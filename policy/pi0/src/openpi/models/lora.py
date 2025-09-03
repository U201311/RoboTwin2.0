import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))

@at.typecheck
class MoEFeedForward(nn.Module):

    
    expert_dim: int  
    hidden_dim: int = 4096  # expert隐藏层维度，与Action expert一致
    num_experts: int = 8  # 专家数量
    top_k: int = 1  # 选择top-k个专家
    # weight_noise_std: float = 0.00  # 添加到expert权重的噪声标准差，用于区分expert

    def setup(self):
        # Router/Gating network
        self.w_gating = self.param(
            "w_gating",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.expert_dim, self.num_experts),
        )
        
        self.b_gating = self.param(
            "b_gating",
            nn.initializers.zeros,
            (self.num_experts,),
        )
        
        # Expert weights (所有expert权重相同，从action expert初始化)
        # Hidden layer weights: gate and up projections for each expert
        self.w_expert_hidden = self.param(
            "w_expert_hidden",
            self._init_democratic_weights,
            (2, self.num_experts, self.expert_dim, self.hidden_dim),
        )
        
        # Output layer weights: down projection for each expert
        self.w_expert_output = self.param(
            "w_expert_output",
            self._init_democratic_weights, 
            (self.num_experts, self.hidden_dim, self.expert_dim),
        )

    def _init_democratic_weights(self, key, shape):
        """初始化民主式权重：所有expert权重相同"""
        # 这里只是占位符初始化，实际权重将由DemocraticMoEWeightLoader加载
        return nn.initializers.lecun_normal()(key, shape)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Args:
            x: 输入张量 (batch_size, seq_len, expert_dim)
            
        Returns:
            output: MoE输出 (batch_size, seq_len, expert_dim)
            loss: 负载均衡损失 (scalar)
            gate_scores: 门控分数 (batch_size, seq_len, num_experts)
            expert_activation_rates: 专家激活率 (num_experts,)
        """
        dtype = x.dtype  
        B, S, expert_dim = x.shape
        x_flat = x.reshape(-1, expert_dim)  # (B*S, expert_dim)

        # 1. Gating computation following reference implementation
        scores = jnp.dot(x_flat, self.w_gating.astype(dtype))  # Raw logits
        # scores = nn.softmax(scores, axis=-1)  # Apply softmax first
        scores = nn.sigmoid(scores)
        original_scores = scores  # Save post-softmax, pre-bias scores
        gating_scores = scores + self.b_gating.astype(dtype)  # Add bias after softmax

        # 2. Gating loss (load balancing) - use post-bias scores
        gate_loss = jnp.var(jnp.mean(gating_scores, axis=0))  

        # 3. Top-k selection
        # Use original scores for top-k selection to get the indices
        _, top_k_indices = jax.lax.top_k(gating_scores, self.top_k)
        # Extract the original scores corresponding to the selected indices
        batch_indices = jnp.arange(original_scores.shape[0])[:, None]  # (B*S, 1)
        top_k_values = original_scores[batch_indices, top_k_indices]  # (B*S, top_k)
        normalized_top_k_values = top_k_values / (jnp.sum(top_k_values, axis=-1, keepdims=True))

        # 4. Expert feedforward with GELU gate (参考MoEgelubiasFeedForward的计算方式)
        # Gate projection: 对所有expert同时计算
        ff_gate = jnp.einsum("bf, efh -> beh", x_flat, self.w_expert_hidden[0].astype(dtype))  # (B*S, num_experts, hidden_dim)
        gate_value = nn.gelu(ff_gate)

        # Up projection: 对所有expert同时计算  
        ff1 = jnp.einsum("bf, efh -> beh", x_flat, self.w_expert_hidden[1].astype(dtype))  # (B*S, num_experts, hidden_dim)
        expert_hidden = gate_value * ff1  # (B*S, num_experts, hidden_dim)

        # Down projection: 对所有expert同时计算
        expert_output = jnp.einsum(
            "beh, ehd -> bed", expert_hidden, self.w_expert_output.astype(dtype)
        )  # (B*S, num_experts, expert_dim)

        # 5. Select top-k expert outputs
        selected_outputs = jnp.take_along_axis(
            expert_output,
            top_k_indices[..., None],  # (B*S, top_k, 1)
            axis=1,
        )  # (B*S, top_k, expert_dim)

        # 6. Weighted combination
        weighted_outputs = jnp.sum(selected_outputs * normalized_top_k_values[..., None], axis=1)  # (B*S, expert_dim)

        # 7. Reshape back and ensure correct dtype
        output = weighted_outputs.reshape(B, S, expert_dim).astype(dtype)
        
        # Reshape gating_scores back to original batch/sequence dimensions
        gate_scores = gating_scores.reshape(B, S, self.num_experts)
        
        # 8. Compute expert activation rates
        # Count how many times each expert is selected
        expert_selections = jnp.zeros(self.num_experts)
        for k in range(self.top_k):
            # Get the k-th selected expert for each token
            selected_experts = top_k_indices[:, k]  # (B*S,)
            # Count selections for each expert
            expert_selections = expert_selections.at[selected_experts].add(1)
        
        # Compute activation rates (fraction of tokens that activated each expert)
        total_selections = B * S * self.top_k  # Total possible selections
        expert_activation_rates = expert_selections / total_selections
        
        return output, gate_loss, gate_scores, expert_activation_rates
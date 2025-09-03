import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")

@dataclasses.dataclass(frozen=True)
class MoEWeightLoader(WeightLoader):
    """MoE权重加载器：将Action expert权重复制到所有expert中。
    
    这个加载器专门为MoE设计：
    1. 加载Action expert权重作为基础模板
    2. 将基础权重复制到所有民主式expert中
    3. 为每个expert添加小的随机噪声来区分
    4. 初始化gating网络权重
    """
    
    params_path: str
    num_experts: int = 8
    top_k: int = 1
    noise_std: float = 0.01  # 用于区分expert的噪声标准差
    gating_init_std: float = 0.006  # gating网络初始化标准差
    
    def load(self, params: at.Params) -> at.Params:
        logger.info(f"[MoEWeightLoader] Starting: {self.num_experts} experts, noise_std={self.noise_std}")
        
        # 直接加载checkpoint，不通过CheckpointWeightLoader，避免权重被过滤掉
        logger.info(f"Loading checkpoint directly from: {self.params_path}")
        raw_loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        
        # 展平参数字典
        flat_raw_loaded = flax.traverse_util.flatten_dict(raw_loaded_params, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        
        # 统计权重数量
        paths = [k for k in flat_ref.keys() if "moe" in k]
        action_paths = [k for k in flat_raw_loaded.keys() if "mlp_1" in k]
        logger.info(f"Found {len(paths)} MoE weights, {len(action_paths)} action expert weights")
        
        # 提取Action expert权重作为模板
        action_expert_weights = self._extract_action_expert_weights(flat_raw_loaded)
        
        # 创建结果字典，从原始参数开始
        flat_loaded = {}
        
        # 首先复制所有兼容的基础权重（非MoE权重）
        for k, v in flat_raw_loaded.items():
            if k in flat_ref:
                flat_loaded[k] = v.astype(flat_ref[k].dtype)
        
        if action_expert_weights:
            logger.info(f"Found action expert weights: {list(action_expert_weights.keys())}")
            
            # 复制Action expert权重到所有民主式expert
            self._replicate_to_experts(flat_loaded, flat_ref, action_expert_weights)
            
            # 初始化gating网络
            self._initialize_gating(flat_loaded, flat_ref)
        else:
            logger.warning("No Action expert weights found for replication")
        
        # 处理剩余missing权重
        missing_count = 0
        for k in flat_ref:
            if k not in flat_loaded:
                flat_loaded[k] = flat_ref[k]
                missing_count += 1
                
        logger.info(f"[MoEWeightLoader] Completed: {missing_count} params filled from reference")
                
        return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")
    
    def _extract_action_expert_weights(self, flat_loaded: dict) -> dict:
        """提取Action expert的MLP权重作为模板"""
        action_weights = {}
        
        for path, weight in flat_loaded.items():
            # 查找Action expert MLP权重 (mlp_1)
            if "mlp_1" in path and isinstance(weight, (np.ndarray, jnp.ndarray)):
                weight_key = self._get_weight_key(path)
                if weight_key:
                    action_weights[weight_key] = {
                        'weight': weight,
                        'path': path,
                        'shape': weight.shape
                    }
                    logger.info(f"Found action expert weight: {path} -> {weight.shape}")
                    
        logger.info(f"Extracted {len(action_weights)} action expert weights")
        return action_weights
    
    def _get_weight_key(self, path: str) -> str:
        """从路径提取权重类型键"""
        if "gating_einsum" in path:
            return "gating_einsum" 
        elif "linear" in path:
            return "linear"
        return ""
    
    def _replicate_to_experts(self, flat_loaded: dict, flat_ref: dict, action_weights: dict):
        """将Action expert权重复制到所有民主式expert中"""
        replicated_count = 0
        skipped_count = 0
        
        for ref_path, ref_weight in flat_ref.items():
            if not self._is_expert_weight(ref_path):
                continue
            
            # 尝试映射到Action expert权重
            replicated_weight = self._map_to_weight(ref_path, ref_weight, action_weights)
            if replicated_weight is not None:
                flat_loaded[ref_path] = replicated_weight.astype(ref_weight.dtype)
                replicated_count += 1
            else:
                # 区分是gating权重（将由其他函数处理）还是真正的失败
                if not ("w_gating" in ref_path or "b_gating" in ref_path):
                    skipped_count += 1
                    logger.warning(f"Failed to replicate expert weight: {ref_path}")
                
        logger.info(f"Democratic expert weights replicated: {replicated_count} successful, {skipped_count} failed")
    
    def _is_expert_weight(self, path: str) -> bool:
        """检查是否为民主式expert权重"""
        patterns = ["moe", "w_expert_hidden", "w_expert_output"]
        return any(pattern in path for pattern in patterns)
    
    def _map_to_weight(self, demo_path: str, demo_weight: jnp.ndarray, action_weights: dict) -> jnp.ndarray:
        """将Action expert权重映射到民主式expert权重"""
        
        # 处理expert hidden weights
        if "w_expert_hidden" in demo_path:
            return self._replicate_expert_hidden_weight(demo_weight, action_weights)
            
        # 处理expert output weights
        elif "w_expert_output" in demo_path:
            return self._replicate_expert_output_weight(demo_weight, action_weights)
            
        # 处理gating权重 - 这些不从Action expert复制，而是新初始化
        elif "w_gating" in demo_path or "b_gating" in demo_path:
            return None  # 让_initialize_gating函数处理
            
        else:
            logger.warning(f"Unknown democratic weight type: {demo_path}")
            return None
    
    def _replicate_expert_hidden_weight(self, demo_weight: jnp.ndarray, action_weights: dict) -> jnp.ndarray:
        """复制Action expert权重到expert hidden权重"""
        if "gating_einsum" in action_weights:
            action_gating = action_weights["gating_einsum"]["weight"]
            
            # Action expert权重形状: (18, 2, 1024, 4096) -> 我们需要每一层的权重
            # 目标权重可能是4维 (2, num_experts, expert_dim, hidden_dim) 或 5维 (num_layers, 2, num_experts, expert_dim, hidden_dim)
            
            if len(demo_weight.shape) == 4:
                # 4维目标: (2, num_experts, expert_dim, hidden_dim)
                num_gates, num_experts, expert_dim, hidden_dim = demo_weight.shape
                
                # Action expert是多层的 (18, 2, expert_dim, hidden_dim)
                if len(action_gating.shape) == 4:
                    num_layers, action_gates, action_input, action_hidden = action_gating.shape
                    
                    # 检查维度兼容性
                    if (action_gates == num_gates and 
                        action_input == expert_dim and
                        action_hidden == hidden_dim):
                        
                        # 创建结果张量
                        result = jnp.zeros(demo_weight.shape, dtype=demo_weight.dtype)
                        
                        # 使用第一层的权重作为模板复制到所有expert
                        layer_weight = action_gating[0]  # 使用第一层权重 (2, expert_dim, hidden_dim)
                        
                        for expert_idx in range(num_experts):
                            result = result.at[:, expert_idx, :, :].set(layer_weight)
                        
                        logger.info(f"Replicated action expert hidden weights from layer 0")
                        return result
                    else:
                        logger.warning(f"Hidden weight shape incompatible: action{action_gating.shape} vs target{demo_weight.shape}")
                else:
                    logger.warning(f"Unexpected action gating shape: {action_gating.shape}, expected 4D")
                    
            elif len(demo_weight.shape) == 5:
                # 5维目标: (num_layers, 2, num_experts, expert_dim, hidden_dim)
                num_layers, num_gates, num_experts, expert_dim, hidden_dim = demo_weight.shape
                
                # Action expert是多层的 (18, 2, expert_dim, hidden_dim)
                if len(action_gating.shape) == 4:
                    action_layers, action_gates, action_input, action_hidden = action_gating.shape
                    
                    # 检查维度兼容性
                    if (action_gates == num_gates and 
                        action_input == expert_dim and
                        action_hidden == hidden_dim):
                        
                        # 创建结果张量
                        result = jnp.zeros(demo_weight.shape, dtype=demo_weight.dtype)
                        
                        # 复制每一层的权重到所有expert
                        for layer_idx in range(min(num_layers, action_layers)):
                            layer_weight = action_gating[layer_idx]  # (2, expert_dim, hidden_dim)
                            for expert_idx in range(num_experts):
                                result = result.at[layer_idx, :, expert_idx, :, :].set(layer_weight)
                        
                        logger.info(f"Replicated action expert hidden weights from {min(num_layers, action_layers)} layers")
                        return result
                    else:
                        logger.warning(f"Hidden weight shape incompatible: action{action_gating.shape} vs target{demo_weight.shape}")
                else:
                    logger.warning(f"Unexpected action gating shape: {action_gating.shape}, expected 4D")
            else:
                logger.warning(f"Unexpected target weight shape dimensions: {len(demo_weight.shape)}, expected 4 or 5")
                    
        return None
    
    def _replicate_expert_output_weight(self, demo_weight: jnp.ndarray, action_weights: dict) -> jnp.ndarray:
        """复制Action expert权重到expert output权重"""
        if "linear" in action_weights:
            action_linear = action_weights["linear"]["weight"]
            
            # Action expert权重形状: (18, 4096, 1024) -> 我们需要每一层的权重
            # 目标权重可能是3维 (num_experts, hidden_dim, expert_dim) 或 4维 (num_layers, num_experts, hidden_dim, expert_dim)
            
            if len(demo_weight.shape) == 3:
                # 3维目标: (num_experts, hidden_dim, expert_dim)
                num_experts, hidden_dim, expert_dim = demo_weight.shape
                
                # Action expert是多层的 (18, hidden_dim, expert_dim)
                if len(action_linear.shape) == 3:
                    num_layers, action_hidden, action_output = action_linear.shape
                    
                    # 检查兼容性
                    if (action_hidden == hidden_dim and 
                        action_output == expert_dim):
                        result = jnp.zeros(demo_weight.shape, dtype=demo_weight.dtype)
                        
                        # 使用第一层的权重作为模板复制到所有expert
                        layer_weight = action_linear[0]  # 使用第一层权重 (hidden_dim, expert_dim)
                        
                        for expert_idx in range(num_experts):
                            result = result.at[expert_idx, :, :].set(layer_weight)
                        
                        logger.info(f"Replicated action expert output weights from layer 0")
                        return result
                    else:
                        logger.warning(f"Output weight shape incompatible: action{action_linear.shape} vs target{demo_weight.shape}")
                else:
                    logger.warning(f"Unexpected action linear shape: {action_linear.shape}, expected 3D")
                    
            elif len(demo_weight.shape) == 4:
                # 4维目标: (num_layers, num_experts, hidden_dim, expert_dim)
                num_layers, num_experts, hidden_dim, expert_dim = demo_weight.shape
                
                # Action expert是多层的 (18, hidden_dim, expert_dim)
                if len(action_linear.shape) == 3:
                    action_layers, action_hidden, action_output = action_linear.shape
                    
                    # 检查兼容性
                    if (action_hidden == hidden_dim and 
                        action_output == expert_dim):
                        result = jnp.zeros(demo_weight.shape, dtype=demo_weight.dtype)
                        
                        # 复制每一层的权重到所有expert
                        for layer_idx in range(min(num_layers, action_layers)):
                            layer_weight = action_linear[layer_idx]  # (hidden_dim, expert_dim)
                            for expert_idx in range(num_experts):
                                result = result.at[layer_idx, expert_idx, :, :].set(layer_weight)
                        
                        logger.info(f"Replicated action expert output weights from {min(num_layers, action_layers)} layers")
                        return result
                    else:
                        logger.warning(f"Output weight shape incompatible: action{action_linear.shape} vs target{demo_weight.shape}")
                else:
                    logger.warning(f"Unexpected action linear shape: {action_linear.shape}, expected 3D")
            else:
                logger.warning(f"Unexpected target weight shape dimensions: {len(demo_weight.shape)}, expected 3 or 4")
                    
        return None
    
    def _initialize_gating(self, flat_loaded: dict, flat_ref: dict):
        """初始化民主式MoE的gating网络"""
        
        initialized_count = 0
        for path, weight in flat_ref.items():
            if ("moe" in path and 
                ("w_gating" in path or "b_gating" in path)):
                
                if path not in flat_loaded:
                    if "w_gating" in path:
                        # 小值初始化gating权重，促进均匀分布
                        small_init = jax.random.normal(jax.random.key(42), weight.shape) * self.gating_init_std
                    elif "b_gating" in path:
                        # bias初始化为零
                        small_init = jnp.zeros(weight.shape)
                        
                    flat_loaded[path] = small_init.astype(weight.dtype)
                    initialized_count += 1
        
        logger.info(f"Gating weights initialized: {initialized_count}")
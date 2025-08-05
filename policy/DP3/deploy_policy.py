# import packages and module here
import sys

import torch
import sapien.core as sapien
import traceback
import os
import numpy as np
from envs import *
from hydra import initialize, compose
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra import main as hydra_main
import pathlib
from omegaconf import OmegaConf

import yaml
from datetime import datetime
import importlib

from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime
import hydra

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

sys.path.append(os.path.join(parent_directory, '3D-Diffusion-Policy'))

from dp3_policy import *


def encode_obs(observation):  # Post-Process Observation
    obs = dict()
    obs['agent_pos'] = observation['joint_action']['vector']
    obs['point_cloud'] = observation['pointcloud']
    return obs


def _create_default_normalizer(model):
    """创建默认的归一化器参数，用于评估时的基本功能"""
    try:
        if hasattr(model, 'normalizer') and hasattr(model.normalizer, 'params_dict'):
            # 为常见的观察空间创建默认归一化参数
            default_params = {
                'agent_pos': {
                    'mean': np.zeros(14),  # 假设是双臂机器人的关节位置
                    'std': np.ones(14)
                },
                'point_cloud': {
                    'mean': np.zeros(3),   # XYZ 坐标
                    'std': np.ones(3)
                }
            }
            
            # 只设置存在的键
            for key, params in default_params.items():
                if key in model.normalizer.params_dict or not model.normalizer.params_dict:
                    model.normalizer.params_dict[key] = params
                    
            print("Default normalizer parameters created")
    except Exception as e:
        print(f"Failed to create default normalizer: {e}")


def get_model(usr_args):
    config_path = "./3D-Diffusion-Policy/diffusion_policy_3d/config"
    config_name = f"{usr_args['config_name']}.yaml"
    print(f"Using config path: {config_name}")

    with initialize(config_path=config_path, version_base='1.2'):
        cfg = compose(config_name=config_name)

    now = datetime.now()
    run_dir = f"data/outputs/{now:%Y.%m.%d}/{now:%H.%M.%S}_{usr_args['config_name']}_{usr_args['task_name']}"

    hydra_runtime_cfg = {
        "job": {
            "override_dirname": usr_args['task_name']
        },
        "run": {
            "dir": run_dir
        },
        "sweep": {
            "dir": run_dir,
            "subdir": "0"
        }
    }

    OmegaConf.set_struct(cfg, False)
    cfg.hydra = hydra_runtime_cfg
    cfg.task_name = usr_args["task_name"]
    cfg.expert_data_num = usr_args["expert_data_num"]
    cfg.raw_task_name = usr_args["task_name"]
    cfg.policy.use_pc_color = usr_args['use_rgb']
    
    # 添加必要的参数到配置中
    cfg.horizon = usr_args.get('horizon', 8)
    cfg.n_obs_steps = usr_args.get('n_obs_steps', 3)
    cfg.n_action_steps = usr_args.get('n_action_steps', 6)
    OmegaConf.set_struct(cfg, True)

    # 使用 hydra 正确实例化 DP3 模型
    DP3_Model = hydra.utils.instantiate(cfg.policy)
    
    # 尝试加载检查点
    if 'checkpoint_path' in usr_args and usr_args['checkpoint_path']:
        checkpoint_path = usr_args['checkpoint_path']
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 检查是否是 Hydra 训练工作流保存的检查点格式
                if 'state_dicts' in checkpoint and 'ema_model' in checkpoint['state_dicts']:
                    # 优先使用 EMA 模型（通常性能更好）
                    ema_state_dict = checkpoint['state_dicts']['ema_model']
                    DP3_Model.load_state_dict(ema_state_dict)
                    print(f"Successfully loaded EMA model from {checkpoint_path}")
                    
                elif 'state_dicts' in checkpoint and 'model' in checkpoint['state_dicts']:
                    # 如果没有 EMA，使用主模型
                    model_state_dict = checkpoint['state_dicts']['model']
                    DP3_Model.load_state_dict(model_state_dict)
                    print(f"Successfully loaded main model from {checkpoint_path}")
                    
                elif 'state_dict' in checkpoint:
                    # 标准 PyTorch 格式
                    DP3_Model.load_state_dict(checkpoint['state_dict'])
                    print(f"Successfully loaded standard checkpoint from {checkpoint_path}")
                    
                elif 'model' in checkpoint:
                    # 另一种常见格式
                    DP3_Model.load_state_dict(checkpoint['model'])
                    print(f"Successfully loaded model from {checkpoint_path}")
                    
                else:
                    # 尝试直接加载（假设整个文件就是 state_dict）
                    DP3_Model.load_state_dict(checkpoint)
                    print(f"Successfully loaded direct state_dict from {checkpoint_path}")
                    
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Using random weights")
        else:
            print(f"Warning: Checkpoint file not found at {checkpoint_path}")
            print("Using random weights")
    else:
        print("Warning: No checkpoint path provided, using random weights")
    
    # 检查和初始化归一化器
    if hasattr(DP3_Model, 'normalizer'):
        if hasattr(DP3_Model.normalizer, 'params_dict') and len(DP3_Model.normalizer.params_dict) == 0:
            print("Warning: Normalizer is empty, attempting to initialize from checkpoint or defaults")
            
            # 尝试从检查点中加载归一化器参数
            if 'checkpoint_path' in usr_args and usr_args['checkpoint_path'] and os.path.exists(usr_args['checkpoint_path']):
                try:
                    checkpoint = torch.load(usr_args['checkpoint_path'], map_location='cpu')
                    
                    # 查找归一化器参数
                    normalizer_found = False
                    if 'state_dicts' in checkpoint:
                        # 检查是否有归一化器的 state_dict
                        for key in checkpoint['state_dicts']:
                            if 'normalizer' in key.lower():
                                print(f"Found normalizer in checkpoint: {key}")
                                normalizer_found = True
                                break
                    
                    if not normalizer_found:
                        print("No normalizer found in checkpoint, creating default normalizer")
                        # 创建默认的归一化器参数（用于评估时不会造成太大影响）
                        _create_default_normalizer(DP3_Model)
                        
                except Exception as e:
                    print(f"Failed to load normalizer from checkpoint: {e}")
                    print("Creating default normalizer")
                    _create_default_normalizer(DP3_Model)
            else:
                print("Creating default normalizer")
                _create_default_normalizer(DP3_Model)
        else:
            print("Normalizer is properly initialized")
    else:
        print("Model has no normalizer attribute")
    
    # 创建 RobotRunner 实例来管理观察缓冲区
    from diffusion_policy_3d.env_runner.robot_runner import RobotRunner
    DP3_Model.env_runner = RobotRunner(
        n_obs_steps=usr_args.get('n_obs_steps', 3),
        n_action_steps=usr_args.get('n_action_steps', 6)
    )
    
    # 为模型添加必要的方法
    def get_action():
        return DP3_Model.env_runner.get_action(DP3_Model)
    
    def update_obs(obs):
        DP3_Model.env_runner.update_obs(obs)
    
    # 绑定方法到模型
    DP3_Model.get_action = get_action
    DP3_Model.update_obs = update_obs
    
    return DP3_Model


def eval(TASK_ENV, model, observation):
    obs = encode_obs(observation)  # Post-Process Observation
    # instruction = TASK_ENV.get_instruction()

    if len(
            model.env_runner.obs
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.env_runner.reset_obs()
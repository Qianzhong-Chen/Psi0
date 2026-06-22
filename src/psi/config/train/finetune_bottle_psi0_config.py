"""Psi0 18-dim EEF fine-tune config for bottle pick-and-place tasks.

Usage:
    CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$HOME/miniconda3/envs/legs \
    .venv-psi/bin/python scripts/train.py finetune_bottle_psi0_config \
        --exp=bottle-pickup \
        --data.train_repo_ids=bottle_pickup_30_uni

All other defaults are set below — override via CLI as needed.
"""
from psi.config.config import LaunchConfig, TrainConfig, LoggingConfig, WandbConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt
from pydantic import Field
import os


class BottleRepackTransform(pt.RealRepackTransform):
    state_key: str = "observation.state_psi0"
    action_key: str = "action.psi0_18"
    action_chunk_size: int = 30


class BottleFieldTransform(pt.ActionStateTransform):
    stat_path: str = "meta/stats.json"
    stat_action_key: str = "action.psi0_18"
    stat_state_key: str = "observation.state_psi0"
    action_norm_type: str = "bounds"
    use_norm_mask: bool = False
    normalize_state: bool = True


class BottleModelTransform(pt.Psi0ModelTransform):
    img_aug: bool = True


class BottleDataTransform(DataTransform):
    repack: BottleRepackTransform = Field(default_factory=BottleRepackTransform)
    field: BottleFieldTransform = Field(default_factory=BottleFieldTransform)
    model: BottleModelTransform = Field(default_factory=BottleModelTransform)


class BottleDataConfig(LerobotDataConfig):
    root_dir: str = os.environ.get(
        "DATA_ROOT",
        os.path.expanduser("~/LEGS/data/lerobot/wigs"),
    )
    transform: BottleDataTransform = Field(default_factory=BottleDataTransform)


class BottleModelConfig(Psi0ModelConfig):
    model_name_or_path: str = "cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k"
    pretrained_action_header_path: str = "cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k"
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    n_conditions: int = 0
    action_chunk_size: int = 30
    action_dim: int = 18
    action_exec_horizon: int = 30
    observation_horizon: int = 1
    odim: int = 15
    view_feature_dim: int = 2048
    tune_vlm: bool = False
    use_film: bool = False
    combined_temb: bool = False
    rtc: bool = True
    max_delay: int = 8


class DynamicLaunchConfig(LaunchConfig):
    data: BottleDataConfig = Field(default_factory=BottleDataConfig)
    model: BottleModelConfig = Field(default_factory=BottleModelConfig)

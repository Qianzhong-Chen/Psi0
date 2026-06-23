from typing import Union, Annotated
from pydantic import BaseModel, Field, model_validator

from psi.config.config import LaunchConfig, TrainConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform
from psi.config import transform as pt

class DynamicDataTransform(DataTransform):
    repack: pt.RealRepackTransform
    field: pt.ActionStateTransform
    model: pt.Psi0ModelTransform

class DynamicDataConfig(LerobotDataConfig):
    transform: DynamicDataTransform

class DynamicTrainConfig(TrainConfig):
    # Non-VLM fine-tune (tune_vlm=False): only the action head trains, so the
    # frozen VLM keeps per-GPU memory low. 64/GPU fits comfortably on an L40S
    # (~46GB); sweep showed no OOM up to 48 with headroom to spare.
    train_batch_size: int = 64

class DynamicLaunchConfig(LaunchConfig):
    train: DynamicTrainConfig
    data: DynamicDataConfig
    model: Psi0ModelConfig
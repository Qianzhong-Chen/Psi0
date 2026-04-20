from __future__ import annotations
import os
from typing_extensions import Self, Annotated
from pydantic import BaseModel, Field, model_validator
import os
import tyro
from typing import Union, Any, TYPE_CHECKING
from tyro.conf import subcommand as cmd
from pathlib import Path
import datetime
from psi.config.transform import DataTransform

class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100


class WandbConfig(BaseModel):
    project: str = "psi"
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    name: str | None = None
    resume: str = "allow"  # allow, must, never

    def model_post_init(self, __context: Any) -> None:
        if self.entity is None:
            self.entity = os.getenv("WANDB_ENTITY", None)


class TrainConfig(BaseModel):
    num_workers: int = 8
    overfit_single_batch: bool = False
    name: str = "human3d"  # "vqvae"
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")  # Environment variable or Path to HF Token

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # "no" for fp32, "fp16", "bf16"
    max_grad_norm: float | None = None  # 1.0
    optimizer_foreach: bool | None = None

    train_batch_size: int = 16
    val_batch_size: int = 16

    val_num_batches: int = 20

    checkpointing_steps: int = 5000
    max_checkpoints_to_keep: int | None = None
    validation_steps: int = 50

    learning_rate: float = 1e-5
    # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            # "num_warmup_steps": 0,
            "betas": (0.9, 0.95),
            "weight_decay": 1e-8,
            "eps": 1e-8,
        }
    )
    scheduler_specific_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {"min_lr": 5.0e-07}
    )

    ## FSDP or DDP
    data_parallel: str = "ddp"  # "deepspeed", "ddp" or "fsdp"
    sharding_strategy: str = "full-shard"
    deepspeed_config: str = "src/InternVLA/config/deepseeds/zero3.json"
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    max_training_steps: int | None = 100_000
    num_train_epochs: int | None = None
    warmup_steps: int | None = 0  # if > 0, overrides warmup_ratio
    warmup_ratio: float | None = 0.05  # used if warmup_steps is not set or 0

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        steps, ratio = self.warmup_steps, self.warmup_ratio
        if (steps is not None and steps > 0) and (ratio is not None and ratio > 0):
            raise ValueError("Only one of warmup_steps or warmup_ratio can be set")

        steps, ratio = self.max_training_steps, self.num_train_epochs
        if (steps is None or steps == 0) and (ratio is None or ratio == 0):
            raise ValueError(
                "At least one of max_training_steps or num_train_epochs must be set"
            )
        if steps is not None and ratio is not None and steps > 0 and ratio > 0:
            raise ValueError(
                "Only one of max_training_steps or num_train_epochs can be set"
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        if self.lr_scheduler_type != "cosine_with_min_lr":
            self.scheduler_specific_kwargs = {}

        if not os.path.isabs(self.deepspeed_config):
            from psi.utils import resolve_path
            self.deepspeed_config = os.path.abspath(
                resolve_path(self.deepspeed_config)
            )

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"
    policy: str | None = None
    action_exec_horizon: int | None = None
    rtc: bool = False
    run_dir: str 
    ckpt_step: int 

    @model_validator(mode="after")
    def set_policy(self):
        if self.policy is None:
            run_dir_path = Path(self.run_dir)
            self.policy = run_dir_path.parts[1]
        return self
        

class DataConfig(BaseModel):
    transform: DataTransform

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> Any:
        raise NotImplementedError
    
class ModelConfig(BaseModel): 
    ...

class DummyModelConfig(ModelConfig):
    # boilerplate model config
    ...
class LaunchConfig(BaseModel):
    # NOTE: This class is used for type hinting only!
    # The real implmentation is DynamicLaunchConfig .

    exp: str = Field(..., description="Name of the experiment", frozen=True)
    seed: int | None = None
    auto_tag_run: bool = False
    eval: bool = False
    debug: bool = False
    timestamp: str | None = None

    log: LoggingConfig
    wandb: WandbConfig
    train: TrainConfig
    data: DataConfig
    model: ModelConfig
    
    def model_post_init(self, __context: Any) -> None:
        def extract_timestamp(folder_name: str) -> str:
            parts = folder_name.split(".")
            return parts[-1] if len(parts) > 1 else ""

        def resolve_run_dir_from_resume_path(resume_path: str) -> str | None:
            """Map .../run, .../run/checkpoints, or .../run/checkpoints/ckpt_N -> .../run."""
            rk_abs = os.path.abspath(os.path.expanduser(resume_path))
            if not os.path.isdir(rk_abs):
                return None
            base = os.path.basename(rk_abs.rstrip(os.sep))
            if base.startswith("ckpt_"):
                return os.path.dirname(os.path.dirname(rk_abs))
            if base == "checkpoints":
                return os.path.dirname(rk_abs)
            return rk_abs

        def list_resumable_runs(trainer_dir: str) -> list[str]:
            """Run folder basenames that belong to this exp and contain at least one ckpt_*."""
            exp_prefix = f"{self.exp}."
            debug_prefix = f"debug-{exp_prefix}"
            filtered: list[str] = []
            if not os.path.isdir(trainer_dir):
                return filtered
            for f in os.listdir(trainer_dir):
                if not (f.startswith(exp_prefix) or f.startswith(debug_prefix)):
                    continue
                run_path = os.path.join(trainer_dir, f)
                if not os.path.isdir(run_path):
                    continue
                ckpt_root = os.path.join(run_path, "checkpoints")
                if not os.path.isdir(ckpt_root):
                    continue
                if not any(
                    d.startswith("ckpt_") and os.path.isdir(os.path.join(ckpt_root, d))
                    for d in os.listdir(ckpt_root)
                ):
                    continue
                filtered.append(f)
            return filtered

        def newest_checkpoint_activity(trainer_dir: str, run_folder: str) -> float:
            ckpt_root = os.path.join(trainer_dir, run_folder, "checkpoints")
            m = os.path.getmtime(ckpt_root)
            for d in os.listdir(ckpt_root):
                if d.startswith("ckpt_"):
                    p = os.path.join(ckpt_root, d)
                    if os.path.isdir(p):
                        m = max(m, os.path.getmtime(p))
            return m

        trainer_dir = os.path.join(self.train.output_dir, self.train.name)

        if self.train.resume_from_checkpoint == "latest":
            auto_resume_success = False
            filtered = list_resumable_runs(trainer_dir)
            runs = dict(
                sorted(
                    {
                        extract_timestamp(f): os.path.join(trainer_dir, f)
                        for f in filtered
                    }.items(),
                    reverse=True,
                )
            )

            if self.timestamp is not None and self.timestamp in runs:
                print(f"Will resume run for specified timestamp: {self.timestamp}")
                self.train.resume_from_checkpoint = runs[self.timestamp]
                auto_resume_success = True
            elif len(filtered) > 0:
                best_folder = max(
                    filtered,
                    key=lambda name: newest_checkpoint_activity(trainer_dir, name),
                )
                ts = extract_timestamp(best_folder)
                print(
                    f"Will auto-resume latest matching experiment run {best_folder} "
                    f"(timestamp {ts}; picked by newest checkpoint activity under {trainer_dir})"
                )
                self.timestamp = ts
                self.train.resume_from_checkpoint = os.path.join(trainer_dir, best_folder)
                auto_resume_success = True

            if not auto_resume_success:
                self.train.resume_from_checkpoint = None

        rk = self.train.resume_from_checkpoint
        if rk not in (None, "latest"):
            run_dir = resolve_run_dir_from_resume_path(str(rk))
            if run_dir is not None:
                ts_candidate = extract_timestamp(os.path.basename(run_dir))
                if ts_candidate.isdigit() and len(ts_candidate) >= 10:
                    self.timestamp = ts_candidate

        is_multinode = (
            "SLURM_NODELIST" in os.environ
            and len(os.environ["SLURM_NODELIST"].split(",")) > 1
        )
        if is_multinode:
            assert self.timestamp is not None, (
                "Timestamp must be provided for multi-node training, eg., "
                '--timestamp=$(date +"%y%m%d%H%M"), or rely on auto-resume from latest.'
            )

        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")

"""Psi0 policy server for REAL-ROBOT deployment with a hard lower-body lock.

Same as ``psi0_serve_simple.py`` (HTTP /act, 15-D state -> 18-D action chunk),
EXCEPT every action returned to the client has its base/locomotion + height dims
overwritten with safe constants, regardless of what the policy predicts:

    action[14] = vx     -> 0.0
    action[15] = vy     -> 0.0
    action[16] = vyaw   -> 0.0
    action[17] = height -> --upright-height (default 0.78, the bottle-ckpt train height)

This is a SERVER-SIDE clamp on purpose: it cannot be bypassed by the client, the
RTC chunk, or a bad checkpoint. The robot's lower body stays still and the torso
stays at a fixed upright height no matter what the upper-body policy does. The
model's internal RTC continuity state (``previous_action``, kept in *normalized*
space) is left untouched, so upper-body RTC blending is unaffected — only the
*emitted* command is clamped.

Use this for real-robot runs of left-arm / fixed-base manipulation checkpoints.
For sim or full loco-manip, use the unmodified ``psi0_serve_simple.py``.

Launch:
    cd ~/LEGS/submodules/Psi0
    .venv-psi/bin/python src/psi/deploy/psi0_serve_real_lockbase.py \
        --host 0.0.0.0 --port 22085 --policy psi0 \
        --run-dir .runs/finetune/<run> --ckpt-step <N> \
        --action-exec-horizon 30 --upright-height 0.78
"""

import sys
import time
import numpy as np
import os.path as osp
from pathlib import Path

import tyro
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from typing import Dict, Any
from fastapi.responses import JSONResponse
from torchvision.transforms import v2
from pydantic import BaseModel, model_validator

from psi.deploy.helpers import *
from psi.config.config import LaunchConfig
from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)

# Action-vector indices of the base + height dims in the 18-D Psi0 action
# [L_6d(0:6), L_grip(6), R_6d(7:13), R_grip(13), vx(14), vy(15), vyaw(16), height(17)].
IDX_VX, IDX_VY, IDX_VYAW, IDX_HEIGHT = 14, 15, 16, 17


class RealLockBaseServerConfig(BaseModel):
    """ServerConfig + real-robot lower-body lock options.

    Kept local to this file so the shared psi.config.ServerConfig (and the
    stock psi0_serve_simple.py) stay untouched.
    """
    host: str = "0.0.0.0"
    port: int = 22085
    device: str = "cuda:0"
    policy: str | None = None
    action_exec_horizon: int | None = None
    rtc: bool = False
    run_dir: str
    ckpt_step: int

    # --- lower-body lock ---
    upright_height: float = 0.78
    """Fixed torso height (m) forced on action[17]. 0.78 = bottle-ckpt train height;
    decoupled_wbc DEFAULT_BASE_HEIGHT is 0.74. Match your data-collection height."""
    lock_base: bool = True
    """Force vx/vy/vyaw -> 0. Disable only if you really want policy locomotion."""
    lock_height: bool = True
    """Force height -> upright_height. Disable to let the policy command height."""

    @model_validator(mode="after")
    def set_policy(self):
        if self.policy is None:
            self.policy = Path(self.run_dir).parts[1]
        return self


class Server:

    def __init__(self, cfg: RealLockBaseServerConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        self.device = torch.device(cfg.device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {cfg.policy} (REAL-ROBOT lock-base build)")

        run_dir = Path(cfg.run_dir)
        ckpt_step = cfg.ckpt_step
        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config_.model_validate_json(conf)
        seed_everything(launch_config.seed or 42)

        from psi.models.psi0 import Psi0Model
        self.model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=cfg.device)
        self.model.to(cfg.device)
        self.model.eval()

        self.maxmin: ActionStateTransform = launch_config.data.transform.field  # type:ignore
        self.repack_transform: SimpleRepackTransform = launch_config.data.transform.repack  # type:ignore
        self.model_transform: Psi0ModelTransform = launch_config.data.transform.model  # type:ignore

        num_params = sum(p.numel() for p in self.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params*1e-6:.3f} Total", ctx_level=1)

        self.Da = launch_config.model.action_dim  # type:ignore
        self.Tp = launch_config.model.action_chunk_size  # type:ignore
        self.Ta = cfg.action_exec_horizon or launch_config.model.action_exec_horizon  # type:ignore
        assert self.Ta <= self.Tp, "action_exec_horizon is too big"
        self.launch_config = launch_config
        self.count = 0

        # --- lower-body lock config ---
        self.lock_base = cfg.lock_base
        self.lock_height = cfg.lock_height
        self.upright_height = float(cfg.upright_height)
        if self.Da < 18:
            overwatch.warning(
                f"action_dim={self.Da} < 18 — no base/height dims to lock; "
                "this server behaves like psi0_serve_simple.")
        overwatch.info(
            f"[LOCK] lock_base={self.lock_base} (vx,vy,vyaw->0) | "
            f"lock_height={self.lock_height} (height->{self.upright_height})")

        self.enable_rtc = cfg.rtc
        if cfg.rtc:
            assert launch_config.model.rtc, "rtc is not supported for this model"  # type:ignore
            self.rtc_max_delay = launch_config.model.max_delay  # type:ignore
            assert self.Tp - self.Ta <= self.rtc_max_delay, "action_exec_horizon too big for rtc_max_delay and action_chunk_size"
            self.previous_action = None
            overwatch.info(f"RTC enabled with max_delay={self.rtc_max_delay}, "
                           f"action_dim={self.Da}, action_chunk_size={self.Tp}, "
                           f"action_exec_horizon={self.Ta}")
        self.last_serve_time = time.monotonic()

    def _apply_lower_body_lock(self, pred_actions: np.ndarray) -> np.ndarray:
        """Overwrite base + height dims with safe constants on EVERY emitted action.

        pred_actions: (Ta, Da) DENORMALIZED action chunk. Mutates in place and
        returns it. No-op for the dims that don't exist (Da < 18)."""
        if self.Da < 15:
            return pred_actions
        if self.lock_base and self.Da > IDX_VYAW:
            pred_actions[:, IDX_VX] = 0.0
            pred_actions[:, IDX_VY] = 0.0
            pred_actions[:, IDX_VYAW] = 0.0
        if self.lock_height and self.Da > IDX_HEIGHT:
            pred_actions[:, IDX_HEIGHT] = self.upright_height
        return pred_actions

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name

            overwatch.info(f"Instruction: {instruction}")

            transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
            t = v2.Compose(transforms)

            states = torch.from_numpy(state_dict["states"].copy())

            if self.maxmin.normalize_state:  # type:ignore
                s_np = states.numpy()
                if self.maxmin.pad_state_dim is not None:
                    s_np = pad_to_len(s_np, self.maxmin.pad_state_dim, dim=1)[0]
                states = torch.from_numpy(self.maxmin.normalize_state_func(s_np)).to(self.device)

            if not self.enable_rtc:
                raw_pred_actions = self.model.predict_action(
                    observations=[[t(Image.fromarray(img)) for img in image_dict.values()]],
                    states=states.unsqueeze(0),
                    instructions=[instruction],
                    num_inference_steps=10,
                    traj2ds=None,
                )
            else:
                current_time = time.monotonic()
                if self.previous_action is None or "reset" in history_dict:
                    overwatch.info("===Reset or first step, without condition===")
                    raw_pred_actions = self.model.predict_action(
                        observations=[[t(Image.fromarray(img)) for img in image_dict.values()]],
                        states=states.unsqueeze(0),
                        instructions=[instruction],
                        num_inference_steps=10,
                        traj2ds=None,
                    )
                else:
                    overwatch.info("RTC enabled, using RTC inference")
                    prev_actions = np.concatenate([
                        self.previous_action[None, self.Ta:, :],
                        np.zeros((1, self.Ta, self.Da), dtype=np.float32),
                    ], axis=1)
                    prev_actions = torch.from_numpy(prev_actions).to(self.device)
                    raw_pred_actions = self.model.predict_action_with_training_rtc_flow(
                        observations=[[t(Image.fromarray(img)) for img in image_dict.values()]],
                        states=states.unsqueeze(0),
                        instructions=[instruction],
                        num_inference_steps=10,
                        traj2ds=None,
                        prev_actions=prev_actions,
                        inference_delay=(self.Tp - self.Ta),
                        max_delay=self.rtc_max_delay,
                    )

            raw_pred_actions = raw_pred_actions.reshape(-1, self.Da).cpu().numpy()  # (Tp, Da)
            pred_actions = self.maxmin.denormalize(raw_pred_actions)  # (Tp, Da)
            # Keep RTC continuity state in NORMALIZED space, BEFORE the lock —
            # so upper-body RTC blending is unaffected by the base clamp.
            self.previous_action = raw_pred_actions.copy().astype(np.float32)
            pred_actions = pred_actions[:self.Ta]
            # HARD lower-body lock on the emitted command (cannot be bypassed).
            pred_actions = self._apply_lower_body_lock(pred_actions)
            overwatch.info(f"Return Action ({pred_actions.shape}) [base/height locked]")

            self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions, 0.0)  # type:ignore
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback
            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        overwatch.info(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            overwatch.warning(f"Server crashed, {e}")
        finally:
            overwatch.info("Server stopped.")
            exit(1)


def serve(cfg: RealLockBaseServerConfig) -> None:
    overwatch.info("Server :: Initializing Psi0 (real-robot lock-base)")
    assert cfg.policy is not None, "which policy to serve?"
    server = Server(cfg)
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(RealLockBaseServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(RealLockBaseServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)

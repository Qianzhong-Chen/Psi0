"""Psi0 policy server for the 36-D loco-manipulation model (NO locking).

For the `bend-pick-wristcam` family (action_dim=36, odim=36): arm(14)+hand(14)
JOINTS + torso rpyh(4) + base vx/vy/vyaw/target_yaw(4). Unlike
psi0_serve_real_lockbase.py, this server does NOT clamp anything — it returns the
policy's raw (denormalized) 36-D action chunk verbatim.

Differences from psi0_serve_simple / psi0_serve_real_lockbase:
  * Loads LaunchConfig from run_config.json DIRECTLY via the training config module
    (`finetune_simple_psi0_config`), because these SIMPLE runs ship NO argv.txt.
  * No base/height/anything lock.
  * Otherwise identical HTTP /act + RTC predict path.

Action layout (36-D), per real/deploy/psi-inference.py:
  [0:14]  hand_cmd   (left hand 7 + right hand 7)
  [14:28] arm_cmd    (left arm 7 + right arm 7)
  [28:32] torso      (roll, pitch, yaw, height)
  [32:36] base       (vx, vy, vyaw, target_yaw)

Launch (no argv.txt needed):
    cd ~/LEGS/submodules/Psi0
    .venv-psi/bin/python src/psi/deploy/psi0_serve_locomanip.py \
        --host 0.0.0.0 --port 8014 --policy psi0 \
        --run-dir .runs/bend-pick-wristcam-v1 --ckpt-step 40000 \
        --action-exec-horizon 30 --config-module finetune_simple_psi0_config
"""

import sys
import time
import importlib
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
from psi.utils import pad_to_len, seed_everything
from psi.utils.overwatch import initialize_overwatch

overwatch = initialize_overwatch(__name__)


class LocoManipServerConfig(BaseModel):
    """ServerConfig for the loco-manip (no-lock) server. Kept local so the shared
    psi.config.ServerConfig and the other servers stay untouched."""
    host: str = "0.0.0.0"
    port: int = 8014
    device: str = "cuda:0"
    policy: str | None = None
    action_exec_horizon: int | None = None
    rtc: bool = False
    """Default OFF. This HTTP server returns a full fresh chunk per /act call
    (sequential playback in the bridge). The training-time RTC flow needs
    action_exec_horizon < action_chunk_size (inference_delay > 0); for ckpts where
    exec==chunk (e.g. bend-pick: 30==30) RTC is not applicable here — keep it off."""
    run_dir: str
    ckpt_step: int
    config_module: str = "finetune_simple_psi0_config"
    """Training config module under psi.config.train whose DynamicLaunchConfig
    validates run_config.json (SIMPLE runs ship no argv.txt)."""

    @model_validator(mode="after")
    def set_policy(self):
        if self.policy is None:
            self.policy = "psi0"
        return self


class Server:

    def __init__(self, cfg: LocoManipServerConfig):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        self.device = torch.device(cfg.device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {cfg.policy} (LOCO-MANIP, no lock)")

        run_dir = Path(cfg.run_dir)
        ckpt_step = cfg.ckpt_step
        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        # Load config directly from run_config.json via the training config module
        # (no argv.txt for SIMPLE runs).
        mod = importlib.import_module(f"psi.config.train.{cfg.config_module}")
        DynamicLaunchConfigClass = getattr(mod, "DynamicLaunchConfig")
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = DynamicLaunchConfigClass.model_validate_json(conf)
        seed_everything(launch_config.seed or 42)
        overwatch.info(f"Loaded config via psi.config.train.{cfg.config_module}")

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
        overwatch.info(f"[LOCO-MANIP] action_dim={self.Da}, chunk={self.Tp}, "
                       f"exec_horizon={self.Ta}, NO locking applied")

        self.enable_rtc = cfg.rtc
        if cfg.rtc:
            assert launch_config.model.rtc, "rtc is not supported for this model"  # type:ignore
            self.rtc_max_delay = launch_config.model.max_delay  # type:ignore
            assert self.Tp - self.Ta <= self.rtc_max_delay, "action_exec_horizon too big for rtc_max_delay and action_chunk_size"
            self.previous_action = None
            overwatch.info(f"RTC enabled with max_delay={self.rtc_max_delay}")
        self.last_serve_time = time.monotonic()

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

            obs = [[t(Image.fromarray(img)) for img in image_dict.values()]]

            if not self.enable_rtc:
                raw_pred_actions = self.model.predict_action(
                    observations=obs, states=states.unsqueeze(0),
                    instructions=[instruction], num_inference_steps=10, traj2ds=None,
                )
            else:
                if self.previous_action is None or "reset" in (history_dict or {}):
                    overwatch.info("===Reset or first step, without condition===")
                    raw_pred_actions = self.model.predict_action(
                        observations=obs, states=states.unsqueeze(0),
                        instructions=[instruction], num_inference_steps=10, traj2ds=None,
                    )
                else:
                    overwatch.info("RTC enabled, using RTC inference")
                    prev_actions = np.concatenate([
                        self.previous_action[None, self.Ta:, :],
                        np.zeros((1, self.Ta, self.Da), dtype=np.float32),
                    ], axis=1)
                    prev_actions = torch.from_numpy(prev_actions).to(self.device)
                    raw_pred_actions = self.model.predict_action_with_training_rtc_flow(
                        observations=obs, states=states.unsqueeze(0),
                        instructions=[instruction], num_inference_steps=10, traj2ds=None,
                        prev_actions=prev_actions,
                        inference_delay=(self.Tp - self.Ta), max_delay=self.rtc_max_delay,
                    )

            raw_pred_actions = raw_pred_actions.reshape(-1, self.Da).cpu().numpy()  # (Tp, Da)
            pred_actions = self.maxmin.denormalize(raw_pred_actions)  # (Tp, Da)
            self.previous_action = raw_pred_actions.copy().astype(np.float32)
            pred_actions = pred_actions[:self.Ta]  # (Ta, Da) — NO lock applied
            overwatch.info(f"Return Action ({pred_actions.shape}) [loco-manip, raw]")

            self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions, 0.0)  # type:ignore
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback
            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    def run(self, host: str = "0.0.0.0", port: int = 8014) -> None:
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


def serve(cfg: LocoManipServerConfig) -> None:
    overwatch.info("Server :: Initializing Psi0 (loco-manip, no lock)")
    server = Server(cfg)
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)


def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(LocoManipServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(LocoManipServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)

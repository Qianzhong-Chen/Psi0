"""Modality config for the G1 universal EEF 18-state / 18-action embodiment.

psi0 18-D EEF schema shared with psi0 / pi05 18-act EEF fine-tunes, split
into 8 modality keys so gr00t-n1.6's per-key projectors get independently
normalised channels (xyz, rpy, grip, base velocity, height all live on
different scales).

State == action layout:
    [0:3]   left_eef_xyz
    [3:6]   left_eef_rpy
    [6:7]   left_grip      (snapped {0.5, 1.0})
    [7:10]  right_eef_xyz
    [10:13] right_eef_rpy
    [13:14] right_grip     (snapped {0.5, 1.0})
    [14:17] base_vel       (state: prev-step cmd vx/vy/vyaw, action: current)
    [17:18] height

Reads dataset's meta/modality.json (expected to be the
``modality_g1_eef_uni.json`` sidecar produced by
``scripts/data/add_g1_eef_uni_sidecar.py``) and validates the key set
against EXPECTED_*_KEYS below. Corresponding parquet columns are
``observation.state_psi0_18`` and ``action.psi0_18``.

Mirror of ``g1_loco_uni.py`` with the EEF schema instead of joint-space.
"""

import json
import os
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


DATASET_PATH = os.environ.get("DATASET_PATH")
if not DATASET_PATH:
    raise RuntimeError("DATASET_PATH must be set to load g1_eef_uni modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
try:
    MODALITY_META = json.load(META_PATH.open("r"))
except Exception:
    raise RuntimeError(f"Failed to load modality.json at {META_PATH}")

EXPECTED_STATE_KEYS = [
    "left_eef_xyz",
    "left_eef_rpy",
    "left_grip",
    "right_eef_xyz",
    "right_eef_rpy",
    "right_grip",
    "base_vel",
    "height",
]
EXPECTED_ACTION_KEYS = EXPECTED_STATE_KEYS  # same 8 keys, same layout

state_keys = list(MODALITY_META.get("state", {}).keys())
action_keys = list(MODALITY_META.get("action", {}).keys())
if set(state_keys) != set(EXPECTED_STATE_KEYS):
    raise RuntimeError(f"modality.json state keys mismatch: {state_keys}")
if set(action_keys) != set(EXPECTED_ACTION_KEYS):
    raise RuntimeError(f"modality.json action keys mismatch: {action_keys}")

ACTION_HORIZON = 16

_default_action_cfg = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)

g1_eef_uni_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=state_keys,
    ),
    "action": ModalityConfig(
        delta_indices=list(range(0, ACTION_HORIZON)),
        modality_keys=action_keys,
        action_configs=[_default_action_cfg for _ in EXPECTED_ACTION_KEYS],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    g1_eef_uni_config, embodiment_tag=EmbodimentTag.G1_EEF_UNI
)

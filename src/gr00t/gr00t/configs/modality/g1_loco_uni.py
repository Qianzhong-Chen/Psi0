"""Modality config for the G1 universal 20-state / 23-action embodiment.

Shared upper-body-joint schema used by psi0 / pi05 / gr00t-n1.6 fine-tunes:
    state_20  = L_arm(7) + R_arm(7) + waist_rpy(3) + L_grip(1) + R_grip(1) + height(1)
    action_23 = state_20 + torso_vx(1) + torso_vy(1) + torso_vyaw(1)

Reads dataset's meta/modality.json (expected to be the
``modality_uni_23act.json`` sidecar produced by
``scripts/data/add_uni_23act_columns.py``) and validates the key set against
EXPECTED_*_KEYS below. The corresponding parquet columns are
``observation.state_uni_20`` and ``action.uni_23``.

Mirror of ``g1_locomanip.py``, with:
    * 6 state keys (no separate hand keys — grippers are 1-D each)
    * 9 action keys (no target_yaw — base velocity is the only command channel)
    * video alias "ego_view" (sidecar maps it to ``observation.images.ego_view``)
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
    raise RuntimeError("DATASET_PATH must be set to load uni modality.json")
META_PATH = Path(DATASET_PATH) / "meta" / "modality.json"
if not META_PATH.exists():
    raise RuntimeError(f"Missing modality.json at {META_PATH}")
try:
    MODALITY_META = json.load(META_PATH.open("r"))
except Exception:
    raise RuntimeError(f"Failed to load modality.json at {META_PATH}")

EXPECTED_STATE_KEYS = [
    "left_arm",
    "right_arm",
    "rpy",
    "left_grip",
    "right_grip",
    "height",
]
EXPECTED_ACTION_KEYS = [
    "left_arm",
    "right_arm",
    "rpy",
    "left_grip",
    "right_grip",
    "height",
    "torso_vx",
    "torso_vy",
    "torso_vyaw",
]

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

g1_loco_uni_config = {
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
    g1_loco_uni_config, embodiment_tag=EmbodimentTag.G1_LOCO_UNI
)

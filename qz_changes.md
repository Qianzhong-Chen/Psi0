# Psi0 Training Setup & Workflow

Notes from setting up Psi0 fine-tuning on this machine (Ubuntu 24.04, L40S GPU).

---

## 1. Environment: `.venv-psi`

```bash
cd ~/LEGS/submodules/Psi0
export PATH="$HOME/.local/bin:$PATH"
uv venv .venv-psi --python 3.10
UV_PROJECT_ENVIRONMENT=.venv-psi GIT_LFS_SKIP_SMUDGE=1 uv sync \
    --group serve --group viz --group psi \
    --index-strategy unsafe-best-match
# flash_attn (required — Qwen3VL uses FlashAttention2):
CUDA_HOME=$HOME/miniconda3/envs/legs VIRTUAL_ENV=.venv-psi uv pip install flash-attn --no-build-isolation
```

**Key:** `CUDA_HOME` must point to `~/miniconda3/envs/legs` (has nvcc) since
`/usr/local/cuda` doesn't exist on this machine. Set it at both install time and
runtime.

**Note:** `CUDA_HOME` is required for the `uv sync` step too (not just flash-attn) —
a dependency builds a CUDA extension and fails with `CUDA_HOME environment variable
is not set` otherwise. Prefix the `uv sync` command with `CUDA_HOME=$HOME/miniconda3/envs/legs`.

### Video decoder

lerobot uses `torchcodec` by default but it's broken on this machine (FFmpeg
version mismatch). **Uninstall torchcodec** and let lerobot fall back to `pyav`:
```bash
VIRTUAL_ENV=.venv-psi uv pip uninstall torchcodec
VIRTUAL_ENV=.venv-psi uv pip install av   # pyav — already installed
```

---

## 2. `.env` file

Copy from `.env.sample` and fill in:
```bash
HF_TOKEN=hf_...
WANDB_API_KEY=wandb_v1_...
WANDB_ENTITY=
PSI_HOME=/home/ANT.AMAZON.COM/qzchen/LEGS/submodules/Psi0
DATA_HOME=/home/ANT.AMAZON.COM/qzchen/LEGS/data
HF_HOME=/home/ANT.AMAZON.COM/qzchen/.cache/huggingface
TORCH_HOME=/home/ANT.AMAZON.COM/qzchen/.cache/torch
UV_CACHE_DIR=/home/ANT.AMAZON.COM/qzchen/.cache/uv
HF_LEROBOT_HOME=/home/ANT.AMAZON.COM/qzchen/LEGS/data/lerobot/wigs
OMP_NUM_THREADS=8
TOKENIZERS_PARALLELISM=false
DEEPSPEED_LOG_LEVEL=warning
CUDA_LAUNCH_BLOCKING=true
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

## 3. Pretrained checkpoints

Downloaded from HuggingFace `USC-PSI-Lab/psi-model` to `cache/checkpoints/psi0/`:
```
cache/checkpoints/psi0/
├── pre.fast.1by1.2601091803.ckpt.ego200k.he30k/   (4 GB — base VLM)
│   ├── model.safetensors
│   ├── config.json, tokenizer.json, ...
└── postpre.1by1.pad36.2601131206.ckpt.he30k/      (1.9 GB — action header)
    └── action_header.safetensors
```

To re-download:
```bash
conda activate sam3d-objects  # or any env with huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('USC-PSI-Lab/psi-model', repo_type='model', local_dir='cache/checkpoints',
    allow_patterns='psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k/*')
snapshot_download('USC-PSI-Lab/psi-model', repo_type='model', local_dir='cache/checkpoints',
    allow_patterns='psi0/postpre.1by1.pad36.2601131206.ckpt.he30k/*')
"
```

---

## 4. Data transfer (collected → LeRobot → training-ready)

### 4.1 Build LeRobot dataset (from LEGS repo)

```bash
cd ~/LEGS

# Render (legs conda env, GPU). NOTE: nvcc MUST be on PATH (see below):
conda activate legs
cd scripts/data
PATH="$HOME/miniconda3/envs/legs/bin:$PATH" CUDA_HOME=$HOME/miniconda3/envs/legs \
  python offline_renderer_mp.py --data-dir ../../data/collected/<NAME> --ply sjc13_1 --workers 8

# Build dataset (.venv-openpi). NOTE: needs GR00T repo on PYTHONPATH + ffprobe:
cd ~/LEGS
PYTHONPATH="$HOME/LEGS/submodules/GR00T-WholeBodyControl" \
  PATH="$HOME/miniconda3/envs/legs/bin:$PATH" \
  scripts/data/wigs2universal.sh data/collected/<NAME> data/lerobot/wigs/<DATASET_NAME> ""
```

**Render gotcha — gsplat "No CUDA toolkit found" / `_C` is None.** gsplat 1.0.0
JIT-compiles its CUDA ext and gates on bare `nvcc` being resolvable. This
machine has no `/usr/local/cuda`; nvcc lives in the `legs` conda env. If `nvcc`
isn't on PATH, gsplat silently disables itself and every frame fails with
`AttributeError: 'NoneType' object has no attribute 'fully_fused_projection_packed_fwd'`.
Fix: prepend `$HOME/miniconda3/envs/legs/bin` to PATH (the cached build at
`~/.cache/torch_extensions/py310_cu128/gsplat_cuda/gsplat_cuda.so` then loads).

**Build env `.venv-openpi` (recreate if missing).** The build scripts need the
OLD lerobot API (`lerobot.common.datasets`), pinned in
`GR00T-WholeBodyControl/decoupled_wbc/pyproject.toml` to git
`a445d9c9da6bea99a8972daa4fe1fdd053d711d2`. `.venv-psi` has the NEW lerobot
(0.3.3) and can't run the builder. Recreate:
```bash
cd ~/LEGS/submodules/Psi0
uv venv .venv-openpi --python 3.10
VIRTUAL_ENV=.venv-openpi uv pip install \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@a445d9c9da6bea99a8972daa4fe1fdd053d711d2" \
  "datasets==3.6.0" "numpy==1.26.4" scipy pandas pillow torch torchvision av pyarrow tqdm \
  --index-strategy unsafe-best-match
# decoupled_wbc import resolves via PYTHONPATH (editable install fails: readme
# path escapes the package dir), so pass PYTHONPATH=<GR00T root> when building.
```
ffprobe/ffmpeg: not in `.venv-openpi`; `wigs2universal.sh` prepends the `legs`
conda bin if ffprobe isn't already on PATH (so just keep `legs` bin on PATH).

### 4.2 Rename video column for Psi0 compatibility

Psi0 expects `observation.images.egocentric` but our builder produces
`observation.images.ego_view`. Fix:
```bash
cd ~/LEGS/data/lerobot/wigs/<DATASET_NAME>
mv videos/chunk-000/observation.images.ego_view videos/chunk-000/observation.images.egocentric
python3 -c "
import json
info = json.load(open('meta/info.json'))
info['features']['observation.images.egocentric'] = info['features'].pop('observation.images.ego_view')
json.dump(info, open('meta/info.json', 'w'), indent=2)
"
```

### 4.3 Upload to S3 (optional)

```bash
aws s3 sync ~/LEGS/data/lerobot/wigs/<DATASET_NAME> \
    s3://qzchen-ws/legs/dataset/<DATASET_NAME> --profile coro-manipulation
```

---

## 5. Training config

The training script uses tyro CLI with config modules at `src/psi/config/train/`.
For 18-dim EEF fine-tuning, use `finetune_real_psi0_config`.

### Key column mappings (our dataset → Psi0 expectations)

| Psi0 expects | Our dataset column | CLI override |
|---|---|---|
| `observation.images.egocentric` | `observation.images.ego_view` | Rename in dataset (§4.2) |
| `states` (repack state_key) | `observation.state_psi0` (15D) | `--data.transform.repack.state-key=observation.state_psi0` |
| `action` (repack action_key) | `action.psi0_18` (18D) | `--data.transform.repack.action-key=action.psi0_18` |
| `action` (stat key) | `action.psi0_18` | `--data.transform.field.stat-action-key=action.psi0_18` |
| `states` (stat key) | `observation.state_psi0` | `--data.transform.field.stat-state-key=observation.state_psi0` |

### Model dimensions

| Parameter | Value | Meaning |
|---|---|---|
| `--model.action-dim` | 18 | L_eef(6)+L_grip(1)+R_eef(6)+R_grip(1)+vx+vy+vyaw+height |
| `--model.odim` | 15 | Same minus base commands (state is observed, not commanded) |
| `--model.action-chunk-size` | 30 | Predict 30 future actions |
| `--model.action-exec-horizon` | 30 | Execute all 30 |

---

## 6. Launch training

```bash
cd ~/LEGS/submodules/Psi0

CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$HOME/miniconda3/envs/legs \
.venv-psi/bin/python scripts/train.py finetune_real_psi0_config \
    --exp=bottle-pickup \
    --train.name=finetune \
    --train.data_parallel=ddp \
    --train.mixed_precision=bf16 \
    --train.train_batch_size=4 \
    --train.num_workers=0 \
    --train.max_training_steps=20000 \
    --train.warmup_steps=500 \
    --train.warmup_ratio=None \
    --train.checkpointing_steps=2500 \
    --train.validation_steps=1000 \
    --train.max_grad_norm=1.0 \
    --train.learning_rate=1e-4 \
    --train.lr_scheduler_type=cosine \
    --train.resume_from_checkpoint=latest \
    --log.report_to=wandb \
    --data.root_dir=$HOME/LEGS/data/lerobot/wigs \
    --data.train_repo_ids=bottle_pickup_30_uni \
    --data.transform.repack.action-chunk-size=30 \
    --data.transform.repack.state-key=observation.state_psi0 \
    --data.transform.repack.action-key=action.psi0_18 \
    --data.transform.field.stat-path=meta/stats.json \
    --data.transform.field.stat-action-key=action.psi0_18 \
    --data.transform.field.stat-state-key=observation.state_psi0 \
    --data.transform.field.action_norm_type=bounds \
    --data.transform.field.no-use-norm-mask \
    --data.transform.field.normalize-state \
    --data.transform.model.img-aug \
    --data.transform.model.resize.size 240 320 \
    --data.transform.model.center_crop.size 240 320 \
    --model.model_name_or_path=cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
    --model.pretrained-action-header-path=cache/checkpoints/psi0/postpre.1by1.pad36.2601131206.ckpt.he30k \
    --model.noise-scheduler=flow \
    --model.train-diffusion-steps=1000 \
    --model.n_conditions=0 \
    --model.action-chunk-size=30 \
    --model.action-dim=18 \
    --model.action-exec-horizon=30 \
    --model.observation-horizon=1 \
    --model.odim=15 \
    --model.view_feature_dim=2048 \
    --model.no-tune-vlm \
    --model.no-use_film \
    --model.no-combined_temb \
    --model.rtc \
    --model.max-delay=8
```

### Key flags

| Flag | Notes |
|---|---|
| `--train.resume_from_checkpoint=latest` | Auto-resume from last checkpoint |
| `--train.num_workers=0` | Single-threaded data loading (avoids pyav multiprocess issues) |
| `--train.warmup_ratio=None` | Must explicitly null since warmup_steps is set (pydantic validation) |
| `CUDA_HOME=$HOME/miniconda3/envs/legs` | Required for deepspeed CUDA version check |
| `--model.no-tune-vlm` | Freeze the VLM backbone (only train action head) |

### Output

- Checkpoints: `.runs/finetune/<exp_name>.<timestamp>/checkpoints/ckpt_<step>/`
- Wandb: `https://wandb.ai/stanford-qianzhong/psi/`
- Logs: `.runs/finetune/<exp_name>.<timestamp>/wandb/*/logs/`

---

## 7. Gotchas / troubleshooting

| Issue | Fix |
|---|---|
| `CUDA_HOME does not exist` (deepspeed) | Set `CUDA_HOME=$HOME/miniconda3/envs/legs` at runtime |
| `flash_attn seems not installed` | Install with `CUDA_HOME` pointing to conda nvcc (see §1) |
| `libtorchcodec` / FFmpeg errors | Uninstall torchcodec; pyav fallback works |
| `Column observation.images.egocentric not found` | Rename video dir + update info.json (§4.2) |
| `Column states not found` | Pass `--data.transform.repack.state-key=observation.state_psi0` |
| `Column action not found` | Pass `--data.transform.repack.action-key=action.psi0_18` |
| `KeyError: 'action'` in stats loading | Pass `--data.transform.field.stat-action-key=action.psi0_18` |
| `Only one of warmup_steps or warmup_ratio` | Add `--train.warmup_ratio=None` |
| `zero-dimensional tensor cannot be concatenated` (eval) | Fixed in `src/psi/trainers/finetune.py`: `evaluate()` now does `accelerator.gather(val_loss["loss"]).reshape(-1)` so the scalar per-step loss is 1-dim before `torch.cat` (single-GPU gather keeps it 0-dim otherwise) |
| Rendered object wrong color / lying down (pitch 90°) | Offline renderer was a SEPARATE mesh path that ignored sim's per-episode variant + glb orientation. See §8. |

---

## 8. The offline renderer is a SEPARATE path from the sim (mesh AND camera)

**Lesson (cost: multiple full 500-ep re-renders + 3k training steps on wrong data).**
`scripts/main/simulate_object.py` (what you see live + records) and
`scripts/data/offline_renderer.py` (what builds the training images) are **two
fully independent rendering paths**. They each hardcode their own URDF, camera
intrinsics, resolution, and mesh-loading logic. Changing the sim does NOT change
the rendered dataset — every sim-side change must be mirrored in the renderer.
There were FOUR separate drifts here (mesh + camera), all silent:

### 8a. Mesh handling drift

- **Bottle lying on its side (~90° pitch):** offline_renderer unconditionally
  applied a +90°-about-x "Y-up→Z-up" flip to every `.glb`. The bottle GLBs are
  already Z-up (`glb_z_up: True` in scene_config); the sim skips the flip, the
  renderer didn't. Flipping an already-upright mesh lays it down.
- **Only one bottle color (no green/pink):** offline_renderer built object
  renderers from the static `get_scene()` config, ignoring the per-episode
  variant that the sim picks at random and records in
  `episode_XXXX_scenario.json["objects"][i]["mesh"]`. So every frame used the
  scene default. It also skipped the green variant's `mesh_extra_rpy` /
  `mesh_scale` (green `.glb` is y-up + ~30% oversize).

**Fix applied** (renderer-only — collected data + scenario.json were correct):
- `load_entity_mesh()` now honors `glb_z_up`, `mesh_extra_rpy`, `mesh_scale`
  (mirrors `simulate_object._load_static_mesh`).
- New `_resolve_object_entries()` + `build_entity_renderers(scenario=...)` use
  the per-episode mesh from the scenario and recover that variant's geometry
  overrides from the scene_cfg `visual_variants` (by matching mesh name).
- Per-episode renderer cache key `(scene, recorded-object-meshes)` so the
  green/pink swap actually takes effect (a scene-only key reused ep 0's mesh
  for the whole dataset).

### 8b. Camera drift (URDF + intrinsics + resolution)

The robot moved to the **ZED 2i rig** (`--camera zed2i`, URDF
`robots/g1_sjc13/urdf/g1_sjc13.urdf`). The renderer was still on the legacy
RealSense setup. Three more silent mismatches:

- **Missing camera boxes:** offline_renderer hardcoded the OLD URDF
  `robots/g1/urdf/g1_29dof_with_hand.urdf`, which has none of the ZED hardware.
  The g1_sjc13 URDF adds `wrist_cam_{left,right}_link` + `zed2i_chest_link` —
  box-primitive visuals that are NOT collected as camera views but must appear
  as **black boxes on the robot** in the ego image. robot_renderer.py already
  renders box primitives; it just had the wrong URDF. Fixed: `URDF_PATH` →
  g1_sjc13 (must match simulate_object's `DEFAULT_URDF`).
- **Wrong intrinsics:** the recorder ALWAYS labels the recorded ego camera
  `"d435"` regardless of the real sensor, and the renderer keyed
  `K = D435_REAL_K if cam_name == "d435"`. So the ZED pose was rendered with
  D435 intrinsics. Fixed: K now comes from a `CAMERA` preset (default `zed2i` →
  `ZED2I_REAL_K`), NOT the recorded key.
- **Wrong resolution/aspect (renderer):** renderer hardcoded `W,H = 640,480`
  (4:3); ZED 2i is `640×360` (16:9). With `IntrinsicsCamera`, K's principal
  point (cx=319.5, cy=179.5) implies 640×360 — rendering it into a 480-tall
  viewport shifts the principal point off-center and distorts/crops. Fixed:
  `EGO_W,EGO_H` from the preset. (fov is ignored when K is set.)
- **Wrong resolution/aspect (BUILDER — separate from the renderer!):**
  `scripts/main/utils/sonic_frame_builder.py` ALSO hardcodes the ego image dims
  `EGO_VIEW_HEIGHT/WIDTH = 480/640`, and `build_sonic_lerobot._load_render_image`
  RESIZES every render to that shape. So even after the renderer produced correct
  640×360 frames, the builder squashed them back into 640×480 (16:9 → 4:3) in the
  LeRobot videos. This is a THIRD copy of the resolution constant. Fixed:
  `EGO_VIEW_HEIGHT = 360`. **Always verify the built video dims**, not just the
  rendered frames: `av.open(...).streams.video[0].width/height`.

Override for legacy d435 datasets: `CAMERA=d435 python offline_renderer.py ...`
AND set `EGO_VIEW_HEIGHT=480` in sonic_frame_builder.py.

**Rule of thumb:** any change to `scene_config.py` / `simulate_object.py`
(`visual_variants`, mesh orientation/scale, camera preset, URDF, intrinsics,
resolution) MUST be mirrored in `offline_renderer.py`
(`load_entity_mesh`, the scenario→entry merge, `URDF_PATH`, `CAMERA_PRESETS`,
`EGO_K`/`EGO_W`/`EGO_H`). The renderer and sim share NO config — they only
agree by hand. Always sanity-render ~10 episodes (both variants) to
`render_test/` and eyeball them **before** rendering all 500 + training:
`OUT_SUBDIR=render_test python offline_renderer.py --episode <id> --ply sjc13_1 ...`
(also needs nvcc on PATH — see §4). Check: correct object color/orientation,
16:9 ZED aspect, and the wrist/chest camera black boxes present.

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

---

## 9. CURRENT DEFAULT: Psi0 training on AWS Batch (LEGS → ws_groot_training)

This is the **standard workflow now** (supersedes the §6 single-GPU local recipe
for real runs — local is only for quick debugging). Model = Psi0 18-D EEF head +
Qwen3VL-2B backbone; the backbone trains as **vision LoRA** (not full-finetune),
action head trains fully.

### 9.1 The default recipe (what "same as last time" means)
- **VLM: vision LoRA** — `tune_vlm=true` + `lora=true`, rank=16, alpha=16,
  dropout=0.0, targets `[q,k,v,o,gate,up,down]_proj` (openpi-style). ~17.4M
  trainable params → fits A100 40GB with bs=24/GPU.
- **RTC: OFF** (`rtc=false` → `--model.no-rtc`). Verify in logs: the launched
  accelerate command shows `--model.no-rtc` (not `--model.rtc`).
- **lr = 5e-5**, cosine, warmup 500, bf16, DDP (data-split only), 8 GPUs.
- **Steps: use N+100, not N** (e.g. 20100 for "20k", 30100 for "30k"). The loop
  stops AT `max_training_steps` and checkpoints at multiples of
  `checkpointing_steps` (2500); asking for exactly 30000 can end at step 29999 and
  **skip the ckpt_30000 save**. The +100 guarantees the round checkpoint lands.
- config module: `finetune_real_psi0_config`; batch=24/GPU; action_dim=18, odim=15,
  chunk/exec=30, obs_horizon=1, noise=flow, diffusion_steps=1000, view_dim=2048,
  norm=bounds, resize/crop 240×320, img-aug on.

### 9.2 The pipeline (config templates live in ws_groot_training)
Framework: `~/ws_groot_training/src/Lab126PRGModelTrainingScripts` (the `train`
CLI → builds a 2-layer Docker image → AWS Batch). Per-dataset artifacts, all
committed as working examples:
- `configs/psi0/legs_<date>_<task>_lora.yaml` — the training config (copy the
  latest, e.g. `legs_0706_locomani_lora.yaml`; change `s3_paths`, `dataset_name`,
  `exp_name`, `s3_upload_path`, `max_training_steps`).
- `launch_psi0_<date>_lora.sh` — refreshes ada creds, uploads config to S3,
  `train submit … --config-uri … --config-region us-west-2`.

Steps for a new dataset:
1. **Build + upload dataset** (from LEGS repo): render → `wigs2universal.sh` →
   `aws s3 sync … s3://qzchen-ws/legs/dataset/<name>_uni`. **Validate first** (see
   §10 — two silent build gotchas that crash the container).
2. **Copy config + launch script**, edit the dataset/step/exp fields.
3. **Submit** (image already in ECR — no rebuild unless Psi0 code changed):
   ```bash
   cd ~/ws_groot_training/src/Lab126PRGModelTrainingScripts
   export WANDB_API_KEY=<key>; export WANDB_PROJECT=psi
   IMG=311141550854.dkr.ecr.us-west-2.amazonaws.com/lab126coromanipulationvla:psi0-training-main-f999406-20260701-141236
   aws s3 cp configs/psi0/<cfg>.yaml s3://qzchen-ws/legs/configs/<cfg>.yaml --profile coro-manipulation --region us-west-2
   train submit configs/psi0/<cfg>.yaml --backend batch --instance p4d \
     --name psi0-lora-<name> --no-push --image "$IMG" \
     --config-uri s3://qzchen-ws/legs/configs/<cfg>.yaml --config-region us-west-2
   ```
   The container runs `python -m training.cli local <cfg>`; it downloads the
   config from `TRAINING_CONFIG_URI`, syncs the dataset, then the psi0 adapter
   (`models/psi0/train.py`) builds an `accelerate launch scripts/train.py …`
   subprocess. Monitor via wandb (project `psi`) — a NEW run appears once it
   clears dataset-load into the training loop.

### 9.3 How LoRA is applied AFTER loading the pretrained ckpt (the mechanism)
Order matters — LoRA wraps the VLM **after** the pretrained weights are loaded, so
adapters start from a trained backbone:
1. `init_qwen3vl_models()` loads the pretrained VLM
   (`--model.model_name_or_path=cache/checkpoints/psi0/pre.fast…`) into `vlm_model`.
2. **Then**, if `train.lora`: `from peft import LoraConfig, get_peft_model` →
   `vlm_model = get_peft_model(vlm_model, LoraConfig(r,alpha,dropout,target_modules,
   bias="none"))`. Asserts `tune_vlm=True` (adapters must be trainable). Log line:
   `Wrapped VLM with LoRA (r=16, alpha=16, targets=[…])`.
3. Action head loads its own pretrained weights
   (`--model.pretrained-action-header-path=…postpre…`) and trains **fully** (LoRA
   is VLM-only). A size-mismatch warning on the header ("only loaded transformer
   blocks") is expected/benign.
4. `init_models()`: when `lora`, does NOT touch `requires_grad` (PEFT already froze
   base + unfroze adapters). Final log: `Model has 507M trainable parameters`
   (~17.4M LoRA + the full action head).
All in `src/psi/trainers/finetune.py`; LoRA hyperparams default in
`src/psi/config/model_psi0.py` (rank/alpha=16, dropout 0, 7 target_modules).

### 9.4 Checkpoints auto-upload to S3, self-contained (code fix in trainer.py)
Psi0 originally saved checkpoints **local-disk only** → lost on Batch node
teardown (the framework doesn't upload for you; gr00t/openpi model code does).
`src/psi/trainers/trainer.py` now, in `save_checkpoint` (main process, background
thread), calls `_upload_checkpoint_to_s3()` which reads
`CHECKPOINT_S3_BUCKET`/`CHECKPOINT_S3_PREFIX` (the framework runner exports these
from the config's `checkpoint.s3_upload_path`) and `aws s3 sync`s
`ckpt_<step>` → `s3://<bucket>/<prefix>/ckpt_<step>`. Before syncing it bundles two
things so the ckpt is **self-contained for deployment**:
- `_bundle_norm_stats()` → copies the dataset's `meta/stats.json` (+ `norm_stats.json`)
  into `<ckpt>/assets/<dataset>/` (openpi-style; psi0 otherwise has NO stats — it
  relies on the lerobot dataset's stats at the transform layer).
- `_bundle_run_config()` → copies the run's `run_config.json` + `argv.txt` into the
  ckpt so it records exactly how it was trained.
Result on S3: `ckpt_<step>/{model.safetensors, optimizer.bin, scheduler.bin,
random_states_*, assets/<dataset>/{stats,norm_stats}.json, run_config.json, argv.txt}`.
**This code lives in the fork and is baked into image `f999406`** — a run only gets
it if the image contains it (rebuild via `train submit --push --model-code-dir
~/LEGS/submodules/Psi0` when Psi0 code changes). Also note: the FINAL checkpoint
can still be lost if the job SUCCEEDS and the node is reclaimed before the last
background sync finishes — the round-number step (§9.1) plus the periodic saves
mean the last *saved* ckpt (e.g. 30000) is safe; only a trailing partial is at risk.

### 9.5 GPU / queue availability (verified 2026-07-07)
| Queue | GPU | Status |
|---|---|---|
| `gr00t-p4d-24xlarge-queue` | A100 40GB | ✅ stable — **default** |
| `robometer-train-p4de-queue` | A100 **80GB** | ✅ works (`--instance p4de --job-queue robometer-train-p4de-queue`) |
| `gr00t-p5-48xlarge-queue` | H100 | ⚠️ `InsufficientInstanceCapacity` (CE pinned to single subnet us-west-2a; needs 2b/2c/2d subnets added) |
| `gr00t-p5en-48xlarge-queue` | H100 | ❌ dead — launch template pins a **deleted** capacity reservation |
| Greenland (WagenDevelopment / P4DInitiative) | A100 40GB only | ✅ ~96 idle A100s; **NO H100** allocated to us |

No working H100 path currently. psi0 A100 step time ≈ **0.9 s/step** (p4d, bs 24×8);
30k steps ≈ 8 h. See LEGS `qz_changes.md` for the H100/Greenland investigation detail.

---

## 10. Two silent LeRobot-build gotchas that crash the AWS container

Both bit datasets built via `wigs2universal.sh` (the standalone path). ALWAYS
validate a freshly-built dataset before submitting (`.venv-openpi` python):

1. **parquet feature `_type: "List"` vs `"Sequence"`.** `build_sonic_lerobot`
   (via the Gr00tDataExporter) can embed HF feature metadata as `"_type": "List"`
   (newer `datasets`), which the container's older `datasets` can't parse →
   `ValueError: Feature type 'List' not found`. Physical arrow type is identical
   (`fixed_size_list`). **Fix:** rewrite each parquet's `b"huggingface"` schema
   metadata `"_type": "List"` → `"Sequence"` (script pattern:
   read_table → replace_schema_metadata → write back). Check:
   `pq.read_schema(f).metadata[b'huggingface']` for `"_type": "List"`.
2. **video key `ego_view` vs `egocentric`.** `build_sonic_lerobot` writes
   `observation.images.ego_view`, but Psi0's transform hardcodes
   `observation.images.egocentric` → `KeyError: Column …egocentric not in the
   dataset`. **Fix (now idempotent in `wigs2universal.sh`):** `mv` the video dir +
   rename the feature key in `meta/info.json`. GR00T reads `modality_gr00t_sonic.json`
   (keeps ego_view, unaffected); psi0 reads `modality_psi0.json` (already `egocentric`).

Validation one-liner: load `data_dir=<ds>/data` with `datasets.load_dataset("parquet")`
(catches gotcha 1), and confirm `info.json` video key + video dir are `egocentric`
(gotcha 2), plus fps and `observation.state_psi0`/`action.psi0_18` present.

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

# Render (legs conda env, GPU):
conda activate legs
cd scripts/data
python offline_renderer_mp.py --data-dir ../../data/collected/<NAME> --ply sjc13_1 --workers 5

# Build dataset (.venv-openpi):
cd ~/LEGS
scripts/data/wigs2universal.sh data/collected/<NAME> data/lerobot/wigs/<DATASET_NAME> ""
```

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

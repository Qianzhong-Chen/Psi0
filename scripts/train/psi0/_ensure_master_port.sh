#!/usr/bin/env bash
# Default MASTER_PORT for torchrun when the caller has not set one.
# A fixed port (e.g. 29500) causes EADDRINUSE when several Slurm jobs land on
# the same node. Mix job id, step id, and shell pid so independent jobs rarely
# collide; override anytime with: export MASTER_PORT=...
if [ -z "${MASTER_PORT:-}" ]; then
  _j="${SLURM_JOB_ID:-0}"
  _s="${SLURM_STEP_ID:-0}"
  _p=$$
  export MASTER_PORT=$((28000 + (_j * 7919 + _s * 997 + _p) % 12000))
fi

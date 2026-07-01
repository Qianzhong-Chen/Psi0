"""Merge a LoRA (PEFT-wrapped) Psi0 checkpoint into a plain-keyed checkpoint.

The AWS LoRA runs save the VLM as an UNMERGED PEFT model, so model.safetensors
has keys like:
    vlm_model.base_model.model.model.…{base_layer,lora_A,lora_B}.default.weight
But psi0_serve_real_lockbase.py -> Psi0Model.from_pretrained loads the VLM into a
PLAIN Qwen3VLForConditionalGeneration with strict=True, expecting:
    vlm_model.model.…weight  (no base_model / base_layer / lora_*)

This script rebuilds the identical PEFT wrap, loads the checkpoint's VLM weights,
merge_and_unload()s the adapters into the base weights (exact, lossless math), and
writes a NEW model.safetensors with plain keys next to the original.

Usage:
    .venv-psi/bin/python scripts/merge_lora_ckpt.py \
        --run-dir .runs/legs_0630_pickup_lora --ckpt-step 17500
Output: <run-dir>/checkpoints/ckpt_<step>_merged/model.safetensors  (+ marker)
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, get_peft_model

from psi.utils.utils import parse_args_to_tyro_config

QWEN3VL_VARIANT = None  # resolved from psi0 module below


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt-step", type=int, required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    src = run_dir / "checkpoints" / f"ckpt_{args.ckpt_step}" / "model.safetensors"
    out_dir = run_dir / "checkpoints" / f"ckpt_{args.ckpt_step}_merged"
    out_dir.mkdir(parents=True, exist_ok=True)
    assert src.exists(), f"missing {src}"

    # Parse the run's config the same way the server does, to get the exact
    # LoRA hyperparameters + target modules used at train time.
    cfg = parse_args_to_tyro_config(run_dir / "argv.txt")
    mcfg = cfg.model
    from psi.models.psi0 import QWEN3VL_VARIANT as VARIANT
    print(f"[merge] variant={VARIANT} rank={mcfg.lora_rank} alpha={mcfg.lora_alpha} "
          f"targets={list(mcfg.lora_target_modules)}")

    state_dict = load_file(str(src), device="cpu")

    # Split off the VLM sub-dict (strip the "vlm_model." prefix, keep PEFT keys).
    vlm_sd, head_sd = {}, {}
    for k, v in state_dict.items():
        if k.startswith("vlm_model."):
            vlm_sd[k[len("vlm_model."):]] = v
        elif k.startswith("action_header."):
            head_sd[k] = v  # keep as-is; passthrough
        else:
            raise AssertionError(f"unexpected top-level key: {k}")

    # Rebuild an empty base VLM, wrap with the SAME LoRA config, load, merge.
    vlm_config = AutoConfig.from_pretrained(VARIANT)
    vlm_config._attn_implementation = "flash_attention_2"
    vlm = Qwen3VLForConditionalGeneration(vlm_config).to(dtype=torch.bfloat16)

    # embed/lm_head tie + possible token-resize (mirror from_pretrained).
    emb_key = "base_model.model.model.language_model.embed_tokens.weight"
    if emb_key in vlm_sd and vlm_sd[emb_key].shape[0] != vlm.lm_head.weight.shape[0]:
        vlm.resize_token_embeddings(vlm_sd[emb_key].shape[0], pad_to_multiple_of=192,
                                    mean_resizing=True)
        print(f"[merge] resized token embeddings -> {vlm.lm_head.weight.shape[0]}")

    lora_cfg = LoraConfig(
        r=mcfg.lora_rank, lora_alpha=mcfg.lora_alpha, lora_dropout=mcfg.lora_dropout,
        target_modules=list(mcfg.lora_target_modules), bias="none",
    )
    vlm = get_peft_model(vlm, lora_cfg)

    # lm_head is tied to embed_tokens in from_pretrained; add it so strict load
    # doesn't miss it (PEFT keeps lm_head outside the adapter).
    if "base_model.model.lm_head.weight" not in vlm_sd and emb_key in vlm_sd:
        vlm_sd["base_model.model.lm_head.weight"] = vlm_sd[emb_key]

    missing, unexpected = vlm.load_state_dict(vlm_sd, strict=False)
    # Only tolerate the tied lm_head / rotary buffers as missing; fail loud otherwise.
    hard_missing = [m for m in missing if "lora" in m or "base_layer" in m]
    assert not hard_missing, f"[merge] LoRA/base keys missing at load: {hard_missing[:8]}"
    assert not unexpected, f"[merge] unexpected keys at load: {unexpected[:8]}"
    print(f"[merge] loaded PEFT VLM (missing={len(missing)} tolerated, unexpected=0)")

    merged = vlm.merge_and_unload()  # folds lora_B@lora_A*scaling into base weights

    # Re-emit plain keys matching from_pretrained's expectation: vlm_model.<plain>.
    # Drop lm_head.weight: it is tied to embed_tokens (shared storage, which
    # safetensors refuses to save), and from_pretrained re-ties it at load
    # (vlm_state_dict["lm_head.weight"] = embed_tokens.weight), so omitting it
    # is exactly what the loader expects.
    out_sd = {}
    for k, v in merged.state_dict().items():
        if k == "lm_head.weight":
            continue
        out_sd[f"vlm_model.{k}"] = v.to(torch.bfloat16).contiguous().clone()
    # action head passthrough (already prefixed "action_header.")
    for k, v in head_sd.items():
        out_sd[k] = v.contiguous()

    n_lora = sum(1 for k in out_sd if "lora" in k.lower() or "base_layer" in k)
    assert n_lora == 0, f"[merge] merged dict still has {n_lora} LoRA keys!"

    save_file(out_sd, str(out_dir / "model.safetensors"))
    (out_dir / "MERGED_FROM").write_text(f"ckpt_{args.ckpt_step} (LoRA merged)\n")
    print(f"[merge] wrote {out_dir/'model.safetensors'} "
          f"({len(out_sd)} keys, 0 LoRA)")


if __name__ == "__main__":
    main()

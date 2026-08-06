#!/usr/bin/env python3
"""
Stability gate, task 1: reproduce and locally isolate the float16 NaN/Inf
failure on seed 222 (acestep-v15-base, this pre-Ampere GPU). Registers a
forward hook on every submodule of the DiT model that checks its output
for NaN/Inf and, on first detection, logs the offending module's
qualified name + dtype/shape and raises immediately -- this works
regardless of where the actual per-step sampling loop lives internally,
since all forward computation flows through these submodules' forward().
"""
import os
import sys
import time

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

sys.path.insert(0, "/var/lib/symthaea/training-runs/ace-step-1.5/repo")

import torch
from loguru import logger
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = "/var/lib/symthaea/training-runs/ace-step-1.5/repo"
CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/repo/checkpoints"
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/trace_out"
os.makedirs(SAVE_DIR, exist_ok=True)

FIRST_BAD = {"name": None, "step": 0}
CALL_COUNTER = {"n": 0}


def make_hook(name):
    def hook(module, inp, out):
        CALL_COUNTER["n"] += 1
        if FIRST_BAD["name"] is not None:
            return
        tensors = out if isinstance(out, (tuple, list)) else [out]
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if torch.isnan(t).any() or torch.isinf(t).any():
                    FIRST_BAD["name"] = name
                    FIRST_BAD["step"] = CALL_COUNTER["n"]
                    nan_n = torch.isnan(t).sum().item()
                    inf_n = torch.isinf(t).sum().item()
                    logger.error(
                        f"[trace_nan] FIRST BAD MODULE: {name} "
                        f"(call #{CALL_COUNTER['n']}), shape={list(t.shape)}, "
                        f"dtype={t.dtype}, nan={nan_n}, inf={inf_n}"
                    )
                    raise RuntimeError(f"NaN/Inf first detected in module: {name}")
    return hook


def main():
    logger.info("Initializing DiT handler (base)...")
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT,
        config_path="acestep-v15-base",
        device="auto",
        offload_to_cpu=True,
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}")
        sys.exit(1)

    logger.info("Initializing LLM handler (0.6B, pt backend)...")
    llm_handler = LLMHandler()
    status_msg, success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR,
        lm_model_path="acestep-5Hz-lm-0.6B",
        backend="pt",
        device="auto",
        offload_to_cpu=True,
        dtype=None,
    )
    if not success:
        logger.error(f"LLM init failed: {status_msg}")
        sys.exit(1)

    n_hooked = 0
    for name, module in dit_handler.model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            module.register_forward_hook(make_hook(name or "<root>"))
            n_hooked += 1
    logger.info(f"Registered NaN-detection hooks on {n_hooked} leaf submodules of dit_handler.model")

    params = GenerationParams(
        task_type="text2music",
        thinking=True,
        use_cot_caption=False,
        use_cot_metas=False,
        use_cot_language=False,
        caption="acapella, clean female vocals, no instruments, pop, a cappella",
        lyrics="I love the summer breeze tonight",
        vocal_language="en",
        bpm=100,
        keyscale="C Major",
        timesignature="4",
        duration=15,
        inference_steps=32,
        guidance_scale=7.0,
        seed=222,
    )
    config = GenerationConfig(batch_size=1, audio_format="wav")

    t0 = time.time()
    try:
        result = generate_music(dit_handler, llm_handler, params=params, config=config, save_dir=SAVE_DIR)
        elapsed = time.time() - t0
        if result.success:
            logger.info(f"UNEXPECTED: seed 222 succeeded this time -- {elapsed:.1f}s (non-determinism present)")
        else:
            logger.error(f"FAILED (no hook fired) -- {elapsed:.1f}s -- {result.status_message}")
    except RuntimeError as e:
        elapsed = time.time() - t0
        logger.error(f"Hook caught it -- {elapsed:.1f}s -- {e}")

    logger.info(f"\nFIRST_BAD = {FIRST_BAD}")
    logger.info(f"Total forward calls before failure: {CALL_COUNTER['n']}")


if __name__ == "__main__":
    main()

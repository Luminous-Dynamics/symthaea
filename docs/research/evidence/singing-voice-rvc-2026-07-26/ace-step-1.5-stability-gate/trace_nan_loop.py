#!/usr/bin/env python3
"""
Stability gate, corrected: the NaN failure is non-deterministic (5/6
trials of the identical seed+phrase succeeded), not tied to a specific
seed. Retry the same (seed=222, phrase) with NaN-detecting hooks active
until a failure is actually caught, to identify which module first
produces NaN/Inf.
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

MAX_ATTEMPTS = 8


def make_hook(name, first_bad, call_counter):
    def hook(module, inp, out):
        call_counter["n"] += 1
        if first_bad["name"] is not None:
            return
        tensors = out if isinstance(out, (tuple, list)) else [out]
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                if torch.isnan(t).any() or torch.isinf(t).any():
                    first_bad["name"] = name
                    first_bad["call"] = call_counter["n"]
                    nan_n = torch.isnan(t).sum().item()
                    inf_n = torch.isinf(t).sum().item()
                    logger.error(
                        f"[trace_nan] FIRST BAD MODULE: {name} "
                        f"(call #{call_counter['n']}), shape={list(t.shape)}, "
                        f"dtype={t.dtype}, nan={nan_n}, inf={inf_n}, "
                        f"input_had_nan={any(torch.isnan(x).any() for x in inp if isinstance(x, torch.Tensor) and x.is_floating_point())}"
                    )
    return hook


def main():
    logger.info("Initializing DiT handler (base)...")
    dit_handler = AceStepHandler()
    status_msg, success = dit_handler.initialize_service(
        project_root=PROJECT_ROOT, config_path="acestep-v15-base",
        device="auto", offload_to_cpu=True,
    )
    if not success:
        logger.error(f"DiT init failed: {status_msg}"); sys.exit(1)

    logger.info("Initializing LLM handler...")
    llm_handler = LLMHandler()
    status_msg, success = llm_handler.initialize(
        checkpoint_dir=CHECKPOINT_DIR, lm_model_path="acestep-5Hz-lm-0.6B",
        backend="pt", device="auto", offload_to_cpu=True, dtype=None,
    )
    if not success:
        logger.error(f"LLM init failed: {status_msg}"); sys.exit(1)

    for attempt in range(MAX_ATTEMPTS):
        first_bad = {"name": None, "call": 0}
        call_counter = {"n": 0}
        handles = []
        for name, module in dit_handler.model.named_modules():
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(make_hook(name or "<root>", first_bad, call_counter))
                handles.append(h)

        params = GenerationParams(
            task_type="text2music",
            thinking=True, use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
            caption="acapella, clean female vocals, no instruments, pop, a cappella",
            lyrics="I love the summer breeze tonight",
            vocal_language="en", bpm=100, keyscale="C Major", timesignature="4",
            duration=15, inference_steps=32, guidance_scale=7.0, seed=222,
        )
        config = GenerationConfig(batch_size=1, audio_format="wav")
        t0 = time.time()
        try:
            result = generate_music(dit_handler, llm_handler, params=params, config=config, save_dir=SAVE_DIR)
            elapsed = time.time() - t0
            ok = result.success
        except Exception as e:
            elapsed = time.time() - t0
            ok = False
            logger.error(f"exception: {e}")

        for h in handles:
            h.remove()

        logger.info(f"attempt {attempt}: {'OK' if ok else 'FAILED'} ({elapsed:.1f}s), "
                     f"first_bad={first_bad}, total_calls={call_counter['n']}")

        if not ok and first_bad["name"] is not None:
            logger.info(f"\n*** CAUGHT IT: first NaN/Inf appeared in module '{first_bad['name']}' "
                        f"at forward-call #{first_bad['call']} of {call_counter['n']} ***")
            break
        if not ok and first_bad["name"] is None:
            logger.warning("Generation failed but no hook caught NaN before the final check -- "
                            "the bad module might not be a leaf module, or NaN arose in a non-hooked op "
                            "(e.g. the scheduler's own arithmetic between DiT calls, not inside the DiT itself).")
            break

    logger.info("Done.")


if __name__ == "__main__":
    main()

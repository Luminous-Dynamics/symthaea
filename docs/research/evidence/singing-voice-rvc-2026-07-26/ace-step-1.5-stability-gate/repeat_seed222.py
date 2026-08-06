#!/usr/bin/env python3
"""
Correction check: the NaN-tracing run found seed 222 succeeded cleanly,
contradicting the earlier "seed 222 fails on all 3 phrases" claim from
the controlled-comparison log. Run the SAME (seed, phrase) pair multiple
times in one process to characterize whether the failure is genuinely
non-deterministic (e.g. cuDNN algorithm selection) rather than a fixed
seed-specific bug.
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

from loguru import logger
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = "/var/lib/symthaea/training-runs/ace-step-1.5/repo"
CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/repo/checkpoints"
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/repeat_out"
os.makedirs(SAVE_DIR, exist_ok=True)

N_TRIALS = 6


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

    results = []
    for trial in range(N_TRIALS):
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
            result = generate_music(dit_handler, llm_handler, params=params, config=config,
                                     save_dir=SAVE_DIR)
            elapsed = time.time() - t0
            status = "OK" if result.success else f"FAILED: {result.status_message}"
        except Exception as e:
            elapsed = time.time() - t0
            status = f"EXCEPTION: {e}"
        logger.info(f"trial {trial}: seed=222 -- {status} ({elapsed:.1f}s)")
        results.append(status)

    ok_count = sum(1 for r in results if r == "OK")
    logger.info(f"\n=== SUMMARY: {ok_count}/{N_TRIALS} succeeded for seed=222, same phrase ===")
    for i, r in enumerate(results):
        logger.info(f"  trial {i}: {r}")


if __name__ == "__main__":
    main()

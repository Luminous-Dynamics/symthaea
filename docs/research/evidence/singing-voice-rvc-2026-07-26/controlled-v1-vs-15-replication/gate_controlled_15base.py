#!/usr/bin/env python3
"""
Controlled v1-vs-1.5 intelligibility replication: 1.5-base side.
Same 5 seeds x 3 phrases as gate_controlled_v1.py. Removes the CoT
confound directly (not a freeze-and-reuse workaround): use_cot_caption,
use_cot_metas, use_cot_language all disabled, bpm/keyscale/timesignature/
vocal_language explicitly pinned -- so the effective conditioning is as
close to identical across seeds/variants as this interface allows.
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
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/controlled_base_out"
os.makedirs(SAVE_DIR, exist_ok=True)

CAPTION = "acapella, clean female vocals, no instruments, pop, a cappella"
PHRASES = [
    ("wont_you_sing_along", "Won't you sing along with me"),
    ("quick_brown_fox", "The quick brown fox jumps over the lazy dog"),
    ("summer_breeze", "I love the summer breeze tonight"),
]
SEEDS = [111, 222, 333, 444, 555]


def main():
    logger.info("Initializing DiT handler (base)...")
    t0 = time.time()
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
    logger.info(f"DiT loaded in {time.time() - t0:.1f}s -- {status_msg}")

    logger.info("Initializing LLM handler (0.6B, pt backend)...")
    t0 = time.time()
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
    logger.info(f"LLM loaded in {time.time() - t0:.1f}s -- {status_msg}")

    for phrase_name, lyrics in PHRASES:
        for seed in SEEDS:
            name = f"{phrase_name}_seed{seed}"
            out_path = os.path.join(SAVE_DIR, f"{name}.wav")
            if os.path.exists(out_path):
                logger.info(f"SKIP {name}")
                continue
            logger.info(f"\n{'='*60}\nGenerating: {name}\n{'='*60}")
            params = GenerationParams(
                task_type="text2music",
                thinking=True,               # still needed for audio-codes generation
                use_cot_caption=False,        # do NOT let the LM rewrite the caption
                use_cot_metas=False,          # do NOT let the LM re-derive bpm/key/timesig
                use_cot_language=False,       # do NOT let the LM re-detect language
                caption=CAPTION,
                lyrics=lyrics,
                vocal_language="en",
                bpm=100,
                keyscale="C Major",
                timesignature="4",
                duration=15,
                inference_steps=32,
                guidance_scale=7.0,
                seed=seed,
            )
            config = GenerationConfig(batch_size=1, audio_format="wav")
            t0 = time.time()
            result = generate_music(dit_handler, llm_handler, params=params, config=config, save_dir=SAVE_DIR)
            elapsed = time.time() - t0
            if result.success:
                logger.info(f"{name} OK -- {elapsed:.1f}s")
                for audio in result.audios:
                    path = audio.get("path")
                    if path:
                        os.replace(path, out_path)
                        logger.info(f"  -> {out_path}")
            else:
                logger.error(f"{name} FAILED -- {elapsed:.1f}s -- {result.status_message}")

    logger.info(f"\nDone. Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()

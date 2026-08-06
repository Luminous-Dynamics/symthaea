#!/usr/bin/env python3
"""ACE-Step 1.5 BASE variant (not turbo) replication of the same 3-phrase
lyric-intelligibility test, since turbo's first-pass result was notably
weaker than v1's -- per the user's own sequence, turbo is only the fast
smoke test, base is the real comparison (more inference steps, CFG
support, generally described as the more capable/research-grade variant)."""
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
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/out_base"
os.makedirs(SAVE_DIR, exist_ok=True)

PHRASES = [
    ("wont_you_sing_along", "Won't you sing along with me"),
    ("quick_brown_fox", "The quick brown fox jumps over the lazy dog"),
    ("chirp_chirp_chirp", "Chirp chirp chirp"),
]
CAPTION = "acapella, clean female vocals, no instruments, pop, a cappella"


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

    for name, lyrics in PHRASES:
        logger.info(f"\n{'='*60}\nGenerating: {name}\n{'='*60}")
        params = GenerationParams(
            task_type="text2music",
            thinking=True,
            caption=CAPTION,
            lyrics=lyrics,
            vocal_language="en",
            duration=15,
            inference_steps=32,
            guidance_scale=7.0,
            seed=111,
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
                    new_path = os.path.join(SAVE_DIR, f"{name}.wav")
                    os.replace(path, new_path)
                    logger.info(f"  -> {new_path}")
        else:
            logger.error(f"{name} FAILED -- {elapsed:.1f}s -- {result.status_message}")

    logger.info(f"\nDone. Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Direct control-mechanism comparison, 1.5 side: cover task, using the SAME
real vocal-shaped references as the v1 side (ascending_vocal.wav,
leap_vocal.wav from Gate B). 2 melodies x 2 phrases x 3 seeds. CoT
confound removed per the controlled-replication methodology. Retries each
failed render once (per the stability gate's finding that failures are
non-deterministic, not seed-specific) and records both attempts.
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
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/cover_compare_out"
REF_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"  # reuse v1's vocal refs
os.makedirs(SAVE_DIR, exist_ok=True)

CAPTION = "acapella, clean female vocals, no instruments, pop, a cappella"
PHRASES = [
    ("wont_you_sing_along", "Won't you sing along with me"),
    ("summer_breeze", "I love the summer breeze tonight"),
]
MELODIES = ["ascending_vocal", "leap_vocal"]
SEEDS = [111, 222, 333]
COVER_STRENGTH = 0.5


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

    for melody in MELODIES:
        for phrase_name, lyrics in PHRASES:
            for seed in SEEDS:
                name = f"{melody}_{phrase_name}_seed{seed}"
                out_path = os.path.join(SAVE_DIR, f"{name}.wav")
                if os.path.exists(out_path):
                    logger.info(f"SKIP {name}")
                    continue
                params = GenerationParams(
                    task_type="cover",
                    thinking=True, use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
                    caption=CAPTION,
                    lyrics=lyrics,
                    vocal_language="en", bpm=100, keyscale="C Major", timesignature="4",
                    duration=15,
                    src_audio=os.path.join(REF_DIR, f"{melody}.wav"),
                    audio_cover_strength=COVER_STRENGTH,
                    inference_steps=32,
                    guidance_scale=7.0,
                    seed=seed,
                )
                config = GenerationConfig(batch_size=1, audio_format="wav")
                ok = False
                for attempt in range(2):  # retry once per the stability gate's finding
                    t0 = time.time()
                    try:
                        result = generate_music(dit_handler, llm_handler, params=params,
                                                 config=config, save_dir=SAVE_DIR)
                        elapsed = time.time() - t0
                        if result.success:
                            for audio in result.audios:
                                p = audio.get("path")
                                if p:
                                    os.replace(p, out_path)
                            logger.info(f"{name} OK (attempt {attempt}) -- {elapsed:.1f}s")
                            ok = True
                            break
                        else:
                            logger.error(f"{name} FAILED (attempt {attempt}) -- {elapsed:.1f}s -- {result.status_message}")
                    except Exception as e:
                        elapsed = time.time() - t0
                        logger.error(f"{name} EXCEPTION (attempt {attempt}) -- {elapsed:.1f}s -- {e}")
                if not ok:
                    logger.error(f"{name}: both attempts failed")

    logger.info(f"\nDone. Output dir: {SAVE_DIR}")


if __name__ == "__main__":
    main()

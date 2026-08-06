#!/usr/bin/env python3
"""
Bounded fix attempt: Qwen3RMSNorm.forward() (used for every norm layer in
this DiT, including the one where NaN was first observed:
decoder.layers.21.self_attn_norm) already upcasts to float32 for the
variance/rsqrt computation, but casts DOWN to float16 BEFORE the final
multiply by self.weight:

    return self.weight * hidden_states.to(input_dtype)   # <- original

This is a known-fragile pattern: if the normalized value is legitimately
large before scaling, downcasting to float16 first can overflow to Inf,
and Inf * weight can produce NaN depending on sign/zero. The more robust
pattern defers the downcast until after the weight multiply:

    return (self.weight * hidden_states).to(input_dtype)  # <- patched

Monkeypatches this globally (affects every RMSNorm layer in the model,
not just layer 21 -- a targeted single-layer patch would need to touch
private per-instance state; patching the class method is simpler and
still "only" changes this one intermediate operation's precision, not
the whole model's dtype) and reruns the exact 6-trial repeat test that
found a 5/6 (83%) success rate unpatched, to see if the fix eliminates
the NaN failures.
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

# --- the patch ---
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm


def _patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)


Qwen3RMSNorm.forward = _patched_forward
logger.info("PATCHED Qwen3RMSNorm.forward: defer float16 downcast until after weight multiply")
# --- end patch ---

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

PROJECT_ROOT = "/var/lib/symthaea/training-runs/ace-step-1.5/repo"
CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/repo/checkpoints"
SAVE_DIR = "/var/lib/symthaea/training-runs/ace-step-1.5/repeat_patched_out"
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
            thinking=True, use_cot_caption=False, use_cot_metas=False, use_cot_language=False,
            caption="acapella, clean female vocals, no instruments, pop, a cappella",
            lyrics="I love the summer breeze tonight",
            vocal_language="en", bpm=100, keyscale="C Major", timesignature="4",
            duration=15, inference_steps=32, guidance_scale=7.0, seed=222,
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
        logger.info(f"trial {trial}: seed=222 (PATCHED) -- {status} ({elapsed:.1f}s)")
        results.append((status, elapsed))

    ok_count = sum(1 for s, _ in results if s == "OK")
    avg_time = sum(e for _, e in results) / len(results)
    logger.info(f"\n=== PATCHED SUMMARY: {ok_count}/{N_TRIALS} succeeded, avg {avg_time:.1f}s/render ===")
    for i, (s, e) in enumerate(results):
        logger.info(f"  trial {i}: {s} ({e:.1f}s)")


if __name__ == "__main__":
    main()

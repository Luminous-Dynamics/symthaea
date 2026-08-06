#!/usr/bin/env python3
"""
Vocal Apprentice plan, recommended next verification: does ACE-Step's base
model (no LoRA -- see correction in VOCAL_APPRENTICE_IMPROVEMENT_PLAN.md,
there is no publicly downloadable Lyric2Vocal checkpoint) produce
intelligible ENGLISH sung vocals at all? Bounded, cheap sanity check before
any further architecture work.

Uses the exact phrase from Gate 3/4 ("won't you sing along with me") for
direct continuity with the existing DiffSinger evidence trail.
"""
import os
import sys

import soundfile as sf
import torchaudio

# torchaudio 2.11's default save() backend requires torchcodec, which in
# turn requires FFmpeg shared libs not on LD_LIBRARY_PATH here. The
# diffusion/decode path is unaffected -- only the final file write breaks.
# Route the actual write through soundfile directly instead of chasing the
# FFmpeg/torchcodec dependency for one save call.
def _save_via_soundfile(path, tensor, sample_rate=48000, format=None, backend=None):
    data = tensor.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T  # torchaudio is (channels, samples); soundfile wants (samples, channels)
    sf.write(path, data, sample_rate)


torchaudio.save = _save_via_soundfile

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step/checkpoints"
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/out"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

model(
    audio_duration=15,
    prompt="acapella, clean female vocals, no instruments, pop, a cappella",
    lyrics="[verse]\nWon't you sing along with me",
    infer_step=60,
    guidance_scale=15,
    scheduler_type="euler",
    cfg_type="apg",
    omega_scale=10,
    manual_seeds="12345",
    guidance_interval=0.5,
    guidance_interval_decay=0.0,
    min_guidance_scale=3,
    use_erg_tag=True,
    use_erg_lyric=True,
    use_erg_diffusion=True,
    oss_steps="",
    guidance_scale_text=0.0,
    guidance_scale_lyric=0.0,
    lora_name_or_path="none",
    lora_weight=1.0,
    save_path=os.path.join(OUTPUT_DIR, "wont_you_sing_along.wav"),
)
print("DONE, wrote output")

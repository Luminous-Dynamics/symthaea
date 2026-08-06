#!/usr/bin/env python3
"""
ACE-Step Controllability Audit -- Gate A: repeatability and identity.
Render the same lyrics + prompt across multiple seeds; the analysis script
(gate_a_analyze.py) measures transcription consistency, timbre proxy
consistency, F0 contour variation, and timing variation.
"""
import os

import soundfile as sf
import torchaudio


def _save_via_soundfile(path, tensor, sample_rate=48000, format=None, backend=None):
    data = tensor.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(path, data, sample_rate)


torchaudio.save = _save_via_soundfile

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step/checkpoints"
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_a_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LYRICS = "[verse]\nWon't you sing along with me"
PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
SEEDS = [111, 222, 333, 444, 555]

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

for seed in SEEDS:
    out_path = os.path.join(OUTPUT_DIR, f"seed_{seed}.wav")
    model(
        audio_duration=15,
        prompt=PROMPT,
        lyrics=LYRICS,
        infer_step=60,
        guidance_scale=15,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10,
        manual_seeds=str(seed),
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
        save_path=out_path,
    )
    print(f"DONE seed={seed}")

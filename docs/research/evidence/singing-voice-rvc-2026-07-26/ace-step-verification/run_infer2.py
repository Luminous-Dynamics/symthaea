#!/usr/bin/env python3
"""Replication test 2: Gate 5's own trivial control phrase ("chirp chirp
chirp"), deliberately never used in real songs, to reduce the chance the
first result was memorized-training-data recall rather than genuine
text-to-vocal synthesis. Also tests a longer, more ordinary sentence."""
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
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

TESTS = [
    ("chirp_chirp_chirp", "[verse]\nChirp chirp chirp"),
    ("novel_sentence", "[verse]\nThe quick brown fox jumps over the lazy dog"),
]

for name, lyrics in TESTS:
    model(
        audio_duration=12,
        prompt="acapella, clean female vocals, no instruments, pop, a cappella",
        lyrics=lyrics,
        infer_step=60,
        guidance_scale=15,
        scheduler_type="euler",
        cfg_type="apg",
        omega_scale=10,
        manual_seeds="54321",
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
        save_path=os.path.join(OUTPUT_DIR, f"{name}.wav"),
    )
    print(f"DONE: {name}")

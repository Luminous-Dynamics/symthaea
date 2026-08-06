#!/usr/bin/env python3
"""
Controlled v1-vs-1.5 intelligibility replication: v1 side.
5 seeds x 3 phrases (won't-you-sing, pangram, and a normal short lyric
phrase replacing the unusual "chirp chirp chirp" for this specific
controlled comparison). Same seeds/phrases reused for the 1.5-base run.
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
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/controlled_v1_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
PHRASES = [
    ("wont_you_sing_along", "Won't you sing along with me"),
    ("quick_brown_fox", "The quick brown fox jumps over the lazy dog"),
    ("summer_breeze", "I love the summer breeze tonight"),
]
SEEDS = [111, 222, 333, 444, 555]

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

for phrase_name, lyrics in PHRASES:
    for seed in SEEDS:
        name = f"{phrase_name}_seed{seed}.wav"
        out_path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(out_path):
            print(f"SKIP {name}")
            continue
        model(
            audio_duration=15,
            prompt=PROMPT,
            lyrics=f"[verse]\n{lyrics}",
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
        print(f"DONE {name}")

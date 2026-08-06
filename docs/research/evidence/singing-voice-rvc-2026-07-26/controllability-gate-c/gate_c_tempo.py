#!/usr/bin/env python3
"""
Gate C: rhythm and duration control (bounded pilot, v1 only -- confirmed
standing baseline). v1's ACEStepPipeline exposes no explicit tempo/BPM/
duration-per-word parameter (unlike 1.5's bpm field) -- the only lever
available is prompt-based tempo descriptors in the caption text. Tests
whether that has any measurable causal effect on pacing (voiced duration,
onset timing, phrase-repetition count within the fixed clip length), or
whether prosody stays internally generated regardless of the prompt.

3 tempo conditions x 3 seeds, fixed lyrics ("won't you sing along with
me" for continuity), no melody reference (isolating the tempo-prompt
variable alone, given audio2audio's own limitations already established
in Gate B).
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
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_c_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LYRICS = "Won't you sing along with me"
SEEDS = [111, 222, 333]
CONDITIONS = {
    "baseline": "acapella, clean female vocals, no instruments, pop, a cappella",
    "slow": "acapella, clean female vocals, no instruments, slow tempo, 60 bpm, ballad, a cappella",
    "fast": "acapella, clean female vocals, no instruments, fast tempo, 150 bpm, upbeat, a cappella",
}

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

for cond_name, prompt in CONDITIONS.items():
    for seed in SEEDS:
        name = f"{cond_name}_seed{seed}.wav"
        out_path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(out_path):
            print(f"SKIP {name}")
            continue
        model(
            audio_duration=15,
            prompt=prompt,
            lyrics=f"[verse]\n{LYRICS}",
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

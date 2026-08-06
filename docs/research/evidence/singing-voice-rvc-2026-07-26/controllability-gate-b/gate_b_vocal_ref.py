#!/usr/bin/env python3
"""
Gate B v1 closing control: does a VOCAL-shaped reference (pitch-shifted
"la", not a sine tone) change the sharp lyrics-vs-strength cliff found
with sine-tone references? Small, bounded scope per user direction:
2 melodies (ascending, leap) x 2 seeds x 3 strengths = 12 renders.
"""
import os

import soundfile as sf
import torch
import torchaudio


def _save_via_soundfile(path, tensor, sample_rate=48000, format=None, backend=None):
    data = tensor.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(path, data, sample_rate)


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    return waveform, sr


torchaudio.save = _save_via_soundfile
torchaudio.load = _load_via_soundfile

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step/checkpoints"
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_b_out"
REF_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LYRICS = "[verse]\nOne two three four five"
PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
SEEDS = [111, 222]
MELODIES = ["ascending_vocal", "leap_vocal"]
STRENGTHS = [0.05, 0.10, 0.15]

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",
    torch_compile=False,
    cpu_offload=True,
)

for melody in MELODIES:
    for seed in SEEDS:
        for strength in STRENGTHS:
            name = f"{melody}_seed{seed}_strength{strength:.2f}.wav"
            out_path = os.path.join(OUTPUT_DIR, name)
            if os.path.exists(out_path):
                print(f"SKIP {name}")
                continue
            model(
                audio_duration=12,
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
                audio2audio_enable=True,
                ref_audio_strength=strength,
                ref_audio_input=os.path.join(REF_DIR, f"{melody}.wav"),
                save_path=out_path,
            )
            print(f"DONE {name}")

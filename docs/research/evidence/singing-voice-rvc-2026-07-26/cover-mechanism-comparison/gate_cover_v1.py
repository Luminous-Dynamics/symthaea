#!/usr/bin/env python3
"""
Direct control-mechanism comparison, v1 side: unconditioned + audio2audio,
using the SAME real vocal-shaped references built for Gate B (pitch-shifted
"la", not sine tones). 2 melodies x 2 phrases x 3 seeds for audio2audio,
plus 2 phrases x 3 seeds for a shared unconditioned baseline (doesn't
depend on melody).
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
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/cover_compare_v1_out"
REF_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
PHRASES = [
    ("wont_you_sing_along", "Won't you sing along with me"),
    ("summer_breeze", "I love the summer breeze tonight"),
]
MELODIES = ["ascending_vocal", "leap_vocal"]
SEEDS = [111, 222, 333]
REF_STRENGTH = 0.35  # matches Gate B's earlier chosen strength for this reference type

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)


def render(name, lyrics, seed, conditioned, melody=None):
    out_path = os.path.join(OUTPUT_DIR, name)
    if os.path.exists(out_path):
        print(f"SKIP {name}")
        return
    kwargs = dict(
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
    if conditioned:
        kwargs["audio2audio_enable"] = True
        kwargs["ref_audio_strength"] = REF_STRENGTH
        kwargs["ref_audio_input"] = os.path.join(REF_DIR, f"{melody}.wav")
    model(**kwargs)
    print(f"DONE {name}")


for phrase_name, lyrics in PHRASES:
    for seed in SEEDS:
        render(f"uncond_{phrase_name}_seed{seed}.wav", lyrics, seed, conditioned=False)
        for melody in MELODIES:
            render(f"{melody}_{phrase_name}_seed{seed}.wav", lyrics, seed, conditioned=True, melody=melody)

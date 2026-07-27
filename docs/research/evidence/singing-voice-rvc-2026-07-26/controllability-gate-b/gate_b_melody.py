#!/usr/bin/env python3
"""
ACE-Step Controllability Audit -- Gate B: melody control (bounded pilot).

Scope, explicitly reduced from the full design (documented, not silently
capped): 2 seeds x 4 melody references x 2 conditions (audio2audio-
conditioned vs. unconditioned baseline, SAME seed per pair) = 16 renders.
One lyric probe only for this pass: "one two three four five" (one
syllable per note -- cleanest diagnostic for melody tracking). The
"won't you sing along with me" real-target phrase and the mismatched/
reversed-reference control are deferred to a follow-up pass if this
pilot shows a measurable conditioning effect worth scaling up.
"""
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _save_via_soundfile(path, tensor, sample_rate=48000, format=None, backend=None):
    data = tensor.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T
    sf.write(path, data, sample_rate)


def _load_via_soundfile(path, *args, **kwargs):
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, channels)
    waveform = torch.from_numpy(data.T)  # -> (channels, samples), torchaudio convention
    return waveform, sr


torchaudio.save = _save_via_soundfile
torchaudio.load = _load_via_soundfile  # music_dcae.load_audio() calls torchaudio.load()
# for the audio2audio reference path -- same torchcodec/FFmpeg gap as save().

from acestep.pipeline_ace_step import ACEStepPipeline

CHECKPOINT_DIR = "/var/lib/symthaea/training-runs/ace-step/checkpoints"
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_b_out"
REF_DIR = "/var/lib/symthaea/training-runs/ace-step/melody_refs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LYRICS = "[verse]\nOne two three four five"
PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
SEEDS = [111, 222]
MELODIES = ["monotone", "ascending", "descending", "leap"]
REF_STRENGTH = 0.35

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)


def render(name, seed, conditioned):
    if os.path.exists(os.path.join(OUTPUT_DIR, name)):
        print(f"SKIP {name} (already exists)")
        return
    kwargs = dict(
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
        save_path=os.path.join(OUTPUT_DIR, name),
    )
    if conditioned:
        kwargs["audio2audio_enable"] = True
        kwargs["ref_audio_strength"] = REF_STRENGTH
        kwargs["ref_audio_input"] = os.path.join(REF_DIR, f"{name.split('_')[0]}.wav")
    model(**kwargs)
    print(f"DONE {name} (conditioned={conditioned})")


for seed in SEEDS:
    # One shared unconditioned baseline per seed (doesn't depend on melody --
    # rendering it once per melody would just waste compute on 3 duplicates).
    render(f"shared_seed{seed}_uncond.wav", seed, conditioned=False)
    for melody in MELODIES:
        render(f"{melody}_seed{seed}_cond.wav", seed, conditioned=True)

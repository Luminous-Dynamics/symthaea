#!/usr/bin/env python3
"""
Gate D: phonetic stress test (bounded, v1 only, baseline caption only --
no control prompts). Answers: which phonetic structures remain reliable
in v1, and which fail predictably? Not another discovery pass -- a
capability-boundary map for future render selection/regression testing.

10 phrases x 3 seeds = 30 renders.
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
OUTPUT_DIR = "/var/lib/symthaea/training-runs/ace-step/gate_d_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROMPT = "acapella, clean female vocals, no instruments, pop, a cappella"
SEEDS = [111, 222, 333]

PHRASES = {
    "positive_control":      "Won't you sing along with me",
    "conversational":        "I love the summer breeze tonight",
    "repeated_syllables":    "Bye bye bye bye baby",
    "rapid_letter_names":    "A B C D E F G",
    "phrase_final_stops":    "Turn off the light and lock it",
    "fricative_heavy":       "She sells seashells by the seashore",
    "consonant_clusters":    "Strong streams splashed strangely",
    "long_sustained_vowels": "Moon over the blue lagoon",
    "short_unstressed":      "It is what it is to me",
    "semantically_unusual":  "The clock ate my umbrella",
}

model = ACEStepPipeline(
    checkpoint_dir=CHECKPOINT_DIR,
    dtype="bfloat16",  # overridden to float16 via ACE_PIPELINE_DTYPE env var
    torch_compile=False,
    cpu_offload=True,
)

for phrase_name, lyrics in PHRASES.items():
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

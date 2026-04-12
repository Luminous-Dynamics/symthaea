#!/usr/bin/env python3
"""Batch CLAP validation — statistical confidence across multiple seeds.

Scores all WAV files in a directory, groups by emotion (filename prefix),
and reports mean ± std CLAP discrimination per emotion.

Usage (from symthaea/ dir):
    # 1. Render batch:
    CARGO_TARGET_DIR=/tmp/muse-rhythm cargo run -p symthaea-muse --example render_batch --release -- 10

    # 2. Score:
    nix-shell -p "python3.withPackages(ps: with ps; [numpy scipy librosa soundfile torch transformers])" \
        --run "python3 scripts/audio-bench/batch_clap.py audio_output/batch/"
"""

import sys
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import ClapProcessor, ClapModel
from collections import defaultdict

DESCRIPTIONS = {
    "joyful": [
        "happy upbeat electronic music with bright synth melodies",
        "joyful energetic instrumental with major key progression",
        "uplifting positive synthesizer music, fast tempo",
    ],
    "contemplative": [
        "peaceful ambient meditation music with slow gentle tones",
        "calm contemplative instrumental, minimal and spacious",
        "serene atmospheric pad music for relaxation",
    ],
    "tense": [
        "dark dramatic electronic music with dissonant synth",
        "anxious tense instrumental with minor key and fast rhythm",
        "suspenseful intense synthesizer music, aggressive",
    ],
    "sorrowful": [
        "sad melancholic ambient music with slow lonely notes",
        "sorrowful mournful instrumental, sparse and quiet",
        "grieving emotional music with deep low tones",
    ],
}

NEGATIVE_DESCRIPTIONS = [
    "birds chirping in a forest",
    "people talking in a restaurant",
    "car engine running on highway",
    "dog barking repeatedly",
]


def to_numpy(x):
    if hasattr(x, 'pooler_output'):
        x = x.pooler_output
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    return np.array(x)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def score_audio(processor, model, audio, sr, emotion):
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000

    inputs = processor(audio=[audio.astype(np.float32)], sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        audio_embed = to_numpy(model.get_audio_features(**inputs))
    if audio_embed.ndim > 1:
        audio_embed = audio_embed[0]

    pos_inputs = processor(text=DESCRIPTIONS[emotion], return_tensors="pt", padding=True)
    with torch.no_grad():
        pos_embeds = to_numpy(model.get_text_features(**pos_inputs))

    neg_inputs = processor(text=NEGATIVE_DESCRIPTIONS, return_tensors="pt", padding=True)
    with torch.no_grad():
        neg_embeds = to_numpy(model.get_text_features(**neg_inputs))

    pos_scores = [cosine_sim(audio_embed, te) for te in pos_embeds]
    neg_scores = [cosine_sim(audio_embed, te) for te in neg_embeds]

    mean_pos = float(np.mean(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    return mean_pos - mean_neg, mean_pos, mean_neg


def grade(score):
    if score > 0.15: return "A"
    if score > 0.08: return "B"
    if score > 0.03: return "C"
    return "F"


def main():
    wav_dir = sys.argv[1] if len(sys.argv) > 1 else "audio_output/batch"

    if not os.path.isdir(wav_dir):
        print(f"ERROR: directory not found: {wav_dir}")
        sys.exit(1)

    # Collect WAV files grouped by emotion
    files_by_emotion = defaultdict(list)
    for f in sorted(os.listdir(wav_dir)):
        if not f.endswith(".wav"):
            continue
        for emotion in DESCRIPTIONS:
            if f.startswith(emotion):
                files_by_emotion[emotion].append(os.path.join(wav_dir, f))
                break

    total = sum(len(v) for v in files_by_emotion.values())
    if total == 0:
        print(f"No WAV files found in {wav_dir}")
        sys.exit(1)

    print(f"Found {total} WAV files across {len(files_by_emotion)} emotions")
    for e, fs in files_by_emotion.items():
        print(f"  {e}: {len(fs)} files")

    print("\nLoading CLAP model...")
    processor = ClapProcessor.from_pretrained("laion/larger_clap_music_and_speech")
    model = ClapModel.from_pretrained("laion/larger_clap_music_and_speech")
    model.eval()
    print("Model loaded.\n")

    # Score all files
    results = defaultdict(list)
    for emotion, paths in sorted(files_by_emotion.items()):
        for path in paths:
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            discrim, pos, neg = score_audio(processor, model, audio, sr, emotion)
            results[emotion].append(discrim)
            g = grade(discrim)
            print(f"  {os.path.basename(path):25s} → {discrim:+.4f} ({g})")

    # Report
    print("\n" + "=" * 80)
    print("  CLAP Batch Validation — Statistical Results")
    print("=" * 80)
    print(f"\n  {'Emotion':15s} | {'N':>3s} | {'Mean':>7s} | {'Std':>6s} | {'Min':>7s} | {'Max':>7s} | {'Grade':>5s} | {'A%':>4s}")
    print("  " + "-" * 70)

    all_means = []
    for emotion in ["joyful", "contemplative", "tense", "sorrowful"]:
        if emotion not in results:
            continue
        scores = results[emotion]
        mean = np.mean(scores)
        std = np.std(scores)
        mn = np.min(scores)
        mx = np.max(scores)
        g = grade(mean)
        pct_a = sum(1 for s in scores if s > 0.15) / len(scores) * 100
        all_means.append(mean)
        print(f"  {emotion:15s} | {len(scores):3d} | {mean:+.4f} | {std:.4f} | {mn:+.4f} | {mx:+.4f} | {g:>5s} | {pct_a:3.0f}%")

    print(f"\n  Overall mean discrimination: {np.mean(all_means):+.4f}")
    print(f"  Overall grade: {grade(np.mean(all_means))}")

    # Grade distribution across all samples
    all_scores = [s for scores in results.values() for s in scores]
    print(f"\n  Grade distribution ({len(all_scores)} samples):")
    for g_name in ["A", "B", "C", "F"]:
        count = sum(1 for s in all_scores if grade(s) == g_name)
        bar = "█" * (count * 2)
        print(f"    {g_name}: {count:3d} ({count/len(all_scores)*100:4.1f}%) {bar}")

    print()


if __name__ == "__main__":
    main()

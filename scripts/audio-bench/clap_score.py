#!/usr/bin/env python3
"""CLAP-based emotional validation for Symthaea compositions.

Uses HuggingFace's CLAP (via transformers) to score how well each WAV
matches its intended emotional description. No torchvision dependency.

Usage (from symthaea/ dir):
    nix-shell -p "python3.withPackages(ps: with ps; [numpy scipy librosa soundfile torch transformers])" \
        --run "python3 scripts/audio-bench/clap_score.py audio_output/demo_*.wav"
"""

import sys
import os
import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import ClapProcessor, ClapModel

# ─── Emotional descriptions for each demo ─────────────────────────────────────

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

# Negative controls — descriptions that SHOULD NOT match
NEGATIVE_DESCRIPTIONS = [
    "birds chirping in a forest",
    "people talking in a restaurant",
    "car engine running on highway",
    "dog barking repeatedly",
]


def load_model():
    """Load CLAP model from HuggingFace (downloads ~600MB first run)."""
    print("Loading CLAP model from HuggingFace...")
    model_id = "laion/larger_clap_music_and_speech"
    processor = ClapProcessor.from_pretrained(model_id)
    model = ClapModel.from_pretrained(model_id)
    model.eval()
    print(f"Model loaded: {model_id}\n")
    return processor, model


def to_numpy(x):
    """Convert model output to numpy, handling BaseModelOutput or tensor."""
    if hasattr(x, 'pooler_output'):
        x = x.pooler_output
    if hasattr(x, 'detach'):
        return x.detach().cpu().numpy()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)


def get_audio_embedding(processor, model, audio, sr):
    """Get CLAP embedding for an audio signal."""
    # CLAP expects 48kHz
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
        sr = 48000
    inputs = processor(audio=[audio], sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)
    arr = to_numpy(audio_embed)
    return arr[0] if arr.ndim > 1 else arr


def get_text_embeddings(processor, model, texts):
    """Get CLAP embeddings for text descriptions."""
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs)
    return to_numpy(text_embed)


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def score_file(processor, model, path: str) -> dict:
    """Score a WAV file against its expected and negative descriptions."""
    name = os.path.basename(path).replace(".wav", "").replace("demo_", "")

    if name not in DESCRIPTIONS:
        print(f"  {name}: no description defined, skipping")
        return None

    # Load audio
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Get embeddings
    audio_embed = get_audio_embedding(processor, model, audio.astype(np.float32), sr)
    pos_embeds = get_text_embeddings(processor, model, DESCRIPTIONS[name])
    neg_embeds = get_text_embeddings(processor, model, NEGATIVE_DESCRIPTIONS)

    pos_scores = [cosine_sim(audio_embed, te) for te in pos_embeds]
    neg_scores = [cosine_sim(audio_embed, te) for te in neg_embeds]

    mean_pos = float(np.mean(pos_scores))
    max_pos = float(np.max(pos_scores))
    mean_neg = float(np.mean(neg_scores))
    discrimination = mean_pos - mean_neg

    return {
        "name": name,
        "mean_positive": mean_pos,
        "max_positive": max_pos,
        "mean_negative": mean_neg,
        "discrimination": discrimination,
    }


def main():
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        d = "audio_output"
        if os.path.isdir(d):
            paths = sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav")])
    if not paths:
        print("Usage: python3 clap_score.py <wav_files...>")
        sys.exit(1)

    processor, model = load_model()

    results = []
    for p in paths:
        r = score_file(processor, model, p)
        if r:
            results.append(r)
            print(f"  Scored: {r['name']} (pos={r['mean_positive']:.4f}, neg={r['mean_negative']:.4f})")

    # Report
    print("\n" + "=" * 80)
    print("  CLAP Emotional Validation — Symthaea Compositions")
    print("=" * 80)
    print(f"\n  {'Composition':15s} | {'Pos Match':>9s} | {'Neg Match':>9s} | {'Discrim':>7s} | {'Grade':>5s}")
    print("  " + "-" * 60)

    for r in results:
        d = r["discrimination"]
        grade = "A" if d > 0.15 else "B" if d > 0.08 else "C" if d > 0.03 else "F"
        print(
            f"  {r['name']:15s} | {r['mean_positive']:9.4f} | {r['mean_negative']:9.4f} | {r['discrimination']:7.4f} | {grade:>5s}"
        )

    if results:
        mean_discrim = np.mean([r["discrimination"] for r in results])
        mean_pos = np.mean([r["mean_positive"] for r in results])
        print(f"\n  Mean Positive Score:  {mean_pos:.4f}")
        print(f"  Mean Discrimination: {mean_discrim:.4f}")
        print(f"  Interpretation: >0.15 = strong, >0.08 = moderate, <0.03 = weak")
    print()


if __name__ == "__main__":
    main()

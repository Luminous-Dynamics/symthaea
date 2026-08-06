#!/usr/bin/env python3
"""Step 4 of synthetic-tumbling-raccoon.md: does swapping in a neural
vocoder (Vocos, charactr/vocos-mel-24khz) close the naturalness gap that
pitch micro-naturalization (Step 2/3) barely moved?

Analysis-resynthesis pass, NOT a WORLD-pipeline rewrite: take the existing
Arm-B (WER-winning, non-naturalized) v10 renders, feed each waveform
through Vocos's own forward() (internal mel extraction + neural decode),
and write the result. This tests whether Vocos's decoder produces a more
natural timbre for the same spectral content WORLD already computed --
without touching the F0/duration control that Arm B already tuned.

Run inside `nix develop .#voice-vocoder` (see flake.nix) -- torch/
torchaudio/vocos are nix-managed, not pip-installed.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from vocos import Vocos

SRC_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_full10")
OUT_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PHRASES = [
    "positive_control", "conversational", "repeated_syllables", "rapid_letter_names",
    "phrase_final_stops", "fricative_heavy", "consonant_clusters", "long_sustained_vowels",
    "short_unstressed", "semantically_unusual",
]

TARGET_SR = 24000  # vocos-mel-24khz's trained sample rate


def main():
    print("Loading Vocos (charactr/vocos-mel-24khz)...")
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()

    results = []
    for phrase in PHRASES:
        src_path = SRC_DIR / f"{phrase}_sung_v10full_b.wav"
        if not src_path.exists():
            print(f"  MISSING: {src_path}")
            continue

        audio, sr = sf.read(str(src_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        wav = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
        if sr != TARGET_SR:
            import torchaudio
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=TARGET_SR)

        with torch.no_grad():
            y_hat = vocos(wav)

        y_hat_np = y_hat.squeeze(0).cpu().numpy()

        n_nan = int(np.isnan(y_hat_np).sum())
        n_inf = int(np.isinf(y_hat_np).sum())
        peak = float(np.max(np.abs(y_hat_np))) if y_hat_np.size else float("nan")
        clipped = int(np.sum(np.abs(y_hat_np) >= 0.999))

        out_path = OUT_DIR / f"{phrase}_sung_v12_vocos.wav"
        sf.write(str(out_path), y_hat_np, TARGET_SR)

        status = "OK"
        if n_nan or n_inf:
            status = "NAN_OR_INF"
        elif peak > 1.5:
            status = "BLOWUP"

        print(f"{phrase:24s} peak={peak:.3f} clipped_samples={clipped:5d} nan={n_nan} inf={n_inf}  [{status}]  -> {out_path.name}")
        results.append({
            "phrase": phrase, "peak": peak, "clipped_samples": clipped,
            "n_nan": n_nan, "n_inf": n_inf, "status": status,
            "src": str(src_path), "out": str(out_path),
        })

    import json
    results_path = OUT_DIR.parent / "v12_vocos_resynth_sanity.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote sanity results to {results_path}")

    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{n_ok}/{len(results)} renders passed the NaN/inf/blowup sanity check.")


if __name__ == "__main__":
    main()

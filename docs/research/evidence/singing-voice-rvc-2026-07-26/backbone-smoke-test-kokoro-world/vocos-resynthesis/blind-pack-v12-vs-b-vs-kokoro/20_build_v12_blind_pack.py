#!/usr/bin/env python3
"""Build the 5-phrase blind Arm-B / v12-Vocos / spoken-Kokoro listening pack,
per the user's explicit request after the v12 Vocos result came back mixed
(DNSMOS up, WER and UTMOS down). Same blinding pattern as the prior
LISTENING_PACK_2026-07-28/01_BLIND_PASS (shuffled neutral clip names, a
withheld key, a response sheet filled in BEFORE unblinding), extended per
this request with:
  - shared peak-safe loudness matching (peak-normalize every clip to a
    common target peak, so no clip is louder/softer purely from gain --
    NOT a perceptual/LUFS loudness match, disclosed as such)
  - randomized clip identities, seeded and reproducible
  - separate intelligibility and naturalness rating fields (the prior pack
    used one combined "quality" field; this one keeps them apart per the
    stated critique that conflating them hides exactly the WER-vs-DNSMOS
    disagreement this pack exists to investigate)

No renderer changes are made here -- this only copies, gain-matches, and
shuffles already-rendered audio.
"""
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

SEED = 20260729  # reproducible shuffle + documented below

PHRASES = [
    "positive_control", "fricative_heavy", "consonant_clusters",
    "short_unstressed", "long_sustained_vowels",
]

SOURCES = {
    "B": {  # Arm B baseline (WER-winning, non-naturalized)
        "dir": Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v10_4arm_matrix_full10"),
        "pattern": "{phrase}_sung_v10full_b.wav",
        "label": "Arm B (event-informed masking, WORLD vocoder)",
    },
    "V": {  # Vocos v12 resynthesis of Arm B
        "dir": Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v12_vocos_resynth"),
        "pattern": "{phrase}_sung_v12_vocos.wav",
        "label": "v12 (Arm B resynthesized through Vocos charactr/vocos-mel-24khz)",
    },
    "K": {  # spoken Kokoro reference (quality anchor, not a sung candidate)
        "dir": Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder/gate2_audio"),
        "pattern": "{phrase}_spoken.wav",
        "label": "spoken Kokoro TTS (reference anchor -- not a singing candidate)",
    },
}

TARGET_PEAK_DBFS = -1.0
TARGET_PEAK = 10 ** (TARGET_PEAK_DBFS / 20.0)

OUT_AUDIO_BASE = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_blind_pack_v12_vs_b_vs_kokoro")
PACK_DIR = OUT_AUDIO_BASE / "01_BLIND_PASS"
KEY_DIR = OUT_AUDIO_BASE / "02_KEY_DO_NOT_OPEN_UNTIL_JUDGED"
PACK_DIR.mkdir(parents=True, exist_ok=True)
KEY_DIR.mkdir(parents=True, exist_ok=True)

# gate2_config.json has the true target text for each phrase id
import sys
sys.path.insert(0, str(Path(__file__).parent))
GATE2_CONFIG = json.loads((Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder") / "gate2_config.json").read_text())
TARGET_TEXT = {p["id"]: p["text"] for p in GATE2_CONFIG["phrases"]}


def peak_normalize(y, target_peak):
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-9:
        return y, 1.0
    gain = target_peak / peak
    return y * gain, gain


def main():
    items = []  # (phrase, condition)
    for phrase in PHRASES:
        for cond in ("B", "V", "K"):
            items.append((phrase, cond))

    rng = random.Random(SEED)
    shuffled = items[:]
    rng.shuffle(shuffled)

    key = {}
    key_rows = []

    for i, (phrase, cond) in enumerate(shuffled, start=1):
        src_dir = SOURCES[cond]["dir"]
        src_path = src_dir / SOURCES[cond]["pattern"].format(phrase=phrase)
        if not src_path.exists():
            print(f"MISSING: {src_path}")
            continue

        y, sr = sf.read(str(src_path), dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        y_norm, gain = peak_normalize(y, TARGET_PEAK)

        clip_name = f"clip_{i:02d}"
        out_path = PACK_DIR / f"{clip_name}.wav"
        sf.write(str(out_path), y_norm, sr)

        peak_after = float(np.max(np.abs(y_norm))) if y_norm.size else 0.0
        dur_s = len(y_norm) / sr

        key[clip_name] = {
            "phrase": phrase,
            "condition": cond,
            "condition_label": SOURCES[cond]["label"],
            "target_text": TARGET_TEXT[phrase],
            "src": str(src_path),
            "gain_applied": round(gain, 4),
            "peak_after_dbfs": round(20 * np.log10(max(peak_after, 1e-9)), 2),
            "duration_s": round(dur_s, 2),
        }
        key_rows.append((clip_name, phrase, cond, TARGET_TEXT[phrase], dur_s))
        print(f"{clip_name} <- {cond}:{phrase:24s} dur={dur_s:.2f}s gain={gain:.3f}")

    (KEY_DIR / "key.json").write_text(json.dumps(key, indent=2))

    key_md_lines = [
        "# Unblinding key — open only after `01_BLIND_PASS/RESPONSE_SHEET.md` is filled in",
        "",
        "| clip | phrase | condition | true target text | duration |",
        "|---|---|---|---|---|",
    ]
    for clip_name, phrase, cond, text, dur in key_rows:
        key_md_lines.append(f"| `{clip_name}` | {phrase} | **{cond}** — {SOURCES[cond]['label']} | \"{text}\" | {dur:.2f}s |")
    key_md_lines.append("")
    key_md_lines.append("## Reveal codes")
    key_md_lines.append("")
    key_md_lines.append("- **B** = Arm B baseline (existing WER-winning render, WORLD vocoder, no naturalization)")
    key_md_lines.append("- **V** = v12 (Arm B resynthesized through Vocos, `charactr/vocos-mel-24khz`)")
    key_md_lines.append("- **K** = spoken Kokoro TTS reference (quality anchor -- naturally will sound best; not a singing candidate, don't score it against B/V on singing quality)")
    (KEY_DIR / "key.md").write_text("\n".join(key_md_lines) + "\n")

    print(f"\nWrote {len(shuffled)} clips to {PACK_DIR}")
    print(f"Wrote key to {KEY_DIR}")


if __name__ == "__main__":
    main()

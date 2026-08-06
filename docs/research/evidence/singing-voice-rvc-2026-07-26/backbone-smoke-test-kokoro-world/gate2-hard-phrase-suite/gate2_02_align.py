#!/usr/bin/env python3
"""Gate 2 stage 2: forced alignment, same method as 02_align.py."""
import json
from pathlib import Path

from ctc_forced_aligner import get_word_stamps

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "gate2_config.json").read_text())

for phrase in config["phrases"]:
    wav_path = BASE / "gate2_audio" / f"{phrase['id']}_spoken.wav"
    txt_path = BASE / "gate2_transcripts" / f"{phrase['id']}.txt"
    try:
        word_timestamps, _model, _lyrics_lines = get_word_stamps(str(wav_path), str(txt_path))
        out = [
            {"word": w["text"], "start": float(w["start"]), "end": float(w["end"])}
            for w in word_timestamps
        ]
        status = "ok"
    except Exception as e:
        out = []
        status = f"FAILED: {e}"
    (BASE / "gate2_alignments" / f"{phrase['id']}.json").write_text(json.dumps(out, indent=2))
    print(f"{phrase['id']}: {status}, {len(out)} words aligned")
    for w in out:
        print(f"  {w['word']:12s} [{w['start']:.3f}, {w['end']:.3f}]")

print("\nGate 2 stage 2 done.")

#!/usr/bin/env python3
"""Stage 2: real forced alignment of Kokoro's OWN known transcript against
its OWN generated speech audio, via torchaudio's MMS_FA model
(ctc_forced_aligner.get_word_stamps). This is a well-posed forced-alignment
task -- unlike the earlier failed attempt (aligning a fixed CSD phone
sequence against real SUNG audio), the target here is natural spoken TTS
output and its own known text, exactly what MMS_FA is designed for.
"""
import json
from pathlib import Path

from ctc_forced_aligner import get_word_stamps

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "config.json").read_text())

for phrase in config["phrases"]:
    wav_path = BASE / "audio" / f"{phrase['id']}_spoken.wav"
    txt_path = BASE / "transcripts" / f"{phrase['id']}.txt"
    word_timestamps, _model, _lyrics_lines = get_word_stamps(str(wav_path), str(txt_path))
    out = [
        {"word": w["text"], "start": float(w["start"]), "end": float(w["end"])}
        for w in word_timestamps
    ]
    out_path = BASE / "alignments" / f"{phrase['id']}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"{phrase['id']}:")
    for w in out:
        print(f"  {w['word']:12s} [{w['start']:.3f}, {w['end']:.3f}]  ({w['end']-w['start']:.3f}s)")

print("\nStage 2 done.")

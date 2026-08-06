#!/usr/bin/env python3
"""Stage 1: render each test phrase as natural spoken Kokoro TTS.

This is the SAME af_heart voice already used elsewhere in this project.
Output is plain natural speech -- no pitch/duration manipulation here.
Also writes a plain-text transcript file per phrase for the forced
aligner (stage 2).
"""
import json
from pathlib import Path

from kokoro import KPipeline
import soundfile as sf

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "config.json").read_text())

pipeline = KPipeline(lang_code="a")

for phrase in config["phrases"]:
    gen = pipeline(phrase["text"], voice="af_heart")
    audio = None
    for _, _, chunk in gen:
        audio = chunk if audio is None else __import__("numpy").concatenate([audio, chunk])
    wav_path = BASE / "audio" / f"{phrase['id']}_spoken.wav"
    sf.write(wav_path, audio, 24000)
    txt_path = BASE / "transcripts" / f"{phrase['id']}.txt"
    txt_path.write_text(phrase["text"] + "\n")
    print(f"{phrase['id']}: wrote {wav_path} ({len(audio)/24000:.2f}s), {txt_path}")

print("\nStage 1 done.")

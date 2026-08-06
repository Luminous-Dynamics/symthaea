#!/usr/bin/env python3
"""Gate 2 stage 1: Kokoro render, same method as 01_kokoro_render.py."""
import json
from pathlib import Path

from kokoro import KPipeline
import soundfile as sf
import numpy as np

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "gate2_config.json").read_text())
pipeline = KPipeline(lang_code="a")

for phrase in config["phrases"]:
    gen = pipeline(phrase["text"], voice="af_heart")
    audio = None
    for _, _, chunk in gen:
        audio = chunk if audio is None else np.concatenate([audio, chunk])
    wav_path = BASE / "gate2_audio" / f"{phrase['id']}_spoken.wav"
    sf.write(wav_path, audio, 24000)
    (BASE / "gate2_transcripts" / f"{phrase['id']}.txt").write_text(phrase["text"] + "\n")
    print(f"{phrase['id']}: {len(audio)/24000:.2f}s")

print("\nGate 2 stage 1 done.")

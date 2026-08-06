#!/usr/bin/env python3
"""Generate an af_heart Kokoro speech corpus for RVC target-speaker
training. RVC learns timbre from clean, varied speech -- doesn't need
phoneme-labeled data the way an SVS acoustic model does, just enough
natural acoustic/prosodic variety for the target-speaker model to
generalize. Source text: "Alice's Adventures in Wonderland" (Lewis
Carroll, public domain, Project Gutenberg EBook #11) -- narrative prose
gives natural sentence variety (statements, dialogue, questions,
exclamations) without hand-authoring hundreds of sentences.
"""
import re
from pathlib import Path

from kokoro import KPipeline
import soundfile as sf
import numpy as np

SRC = Path("/var/lib/symthaea/training-runs/voice-conversion/alice.txt")
raw = SRC.read_text(encoding="utf-8")

start = raw.index("CHAPTER I.\n") + len("CHAPTER I.\n")
end = raw.index("*** END OF THE PROJECT GUTENBERG")
body = raw[start:end]

# Collapse whitespace/newlines, normalize curly quotes/dashes for the G2P.
body = body.replace("’", "'").replace("‘", "'")
body = body.replace("“", '"').replace("”", '"')
body = body.replace("—", " -- ")
body = re.sub(r"\s+", " ", body).strip()

# Simple sentence splitter: split on . ! ? followed by a space+capital,
# but don't bother being perfect -- Kokoro handles run-on fragments fine,
# and RVC training doesn't need linguistically perfect segmentation.
raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", body)
sentences = [s.strip() for s in raw_sentences if 15 <= len(s.strip()) <= 220]

out_dir = Path("/var/lib/symthaea/training-runs/voice-conversion/af_heart_corpus")
out_dir.mkdir(exist_ok=True)

pipeline = KPipeline(lang_code="a")
total_seconds = 0.0
TARGET_SECONDS = 20 * 60  # aim for ~20 minutes

used = 0
for sentence in sentences:
    if total_seconds >= TARGET_SECONDS:
        break
    gen = pipeline(sentence, voice="af_heart")
    audio = np.concatenate([a for _, _, a in gen])
    path = out_dir / f"af_heart_{used:04d}.wav"
    sf.write(path, audio, 24000)
    dur = len(audio) / 24000
    total_seconds += dur
    used += 1
    if used % 20 == 0:
        print(f"[{used}] {total_seconds:.0f}s so far -- last: {sentence[:60]}")

print(f"\nWrote {used} clips, total corpus duration: "
      f"{total_seconds:.1f}s ({total_seconds/60:.1f} min)")

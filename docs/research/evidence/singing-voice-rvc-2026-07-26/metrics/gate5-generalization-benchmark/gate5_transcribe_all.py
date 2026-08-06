#!/usr/bin/env python3
"""Gate 5: transcribe everything in one batched Whisper pass (model
loaded once) -- 9 held-out phrases x 4 checkpoints (vocoder), 2 of those
phrases x 4 checkpoints (Griffin-Lim), 2 controls x 2 checkpoints
(vocoder only), and 9 real-audio baselines.
"""
import glob
import os
from faster_whisper import WhisperModel

DS = "/var/lib/symthaea/training-runs/diffsinger"
OUT_DIR = f"{DS}/gate5_out"
RAW_WAVS = f"{DS}/raw/benchmark-01/wavs"

PHRASE_TEXT = {
    "heldout_wontyou": "won't you sing along with me",
    "heldout_simple_chirp": "chirp chirp chirp",
    "heldout_cluster_windyspring": "and a windy spring time day",
    "heldout_butterfly": "butterfly",
    "heldout_comeflyover": "come and fly and over here",
    "heldout_yellowwait": "yellow and wait",
    "heldout_petalssmile": "petals smile",
    "heldout_singsong": "sing a song and dance along",
    "heldout_comedanceover": "come and dance and over here",
}

print("Loading Whisper model (small, CPU)...")
model = WhisperModel("small", device="cpu", compute_type="int8")


def transcribe(path):
    segs, info = model.transcribe(path, language="en", beam_size=5)
    return " ".join(s.text.strip() for s in segs)


results = {}

# --- Real-audio baselines ---
print("\n### Real-audio baselines ###")
for name, text in PHRASE_TEXT.items():
    path = f"{RAW_WAVS}/{name}.wav"
    if os.path.exists(path):
        t = transcribe(path)
        results[f"real_{name}"] = t
        print(f"real_{name} (gt={text!r}): {t!r}")

# --- Checkpoint sweep ---
print("\n### Checkpoint sweep ###")
for step in (1000, 2000, 4000, 6000):
    for name, text in PHRASE_TEXT.items():
        vpath = f"{OUT_DIR}/{name}_step{step}_vocoder.wav"
        if os.path.exists(vpath):
            t = transcribe(vpath)
            results[f"{name}_step{step}_vocoder"] = t
            print(f"{name}_step{step}_vocoder (gt={text!r}): {t!r}")
        gpath = f"{OUT_DIR}/{name}_step{step}_griffinlim.wav"
        if os.path.exists(gpath):
            t = transcribe(gpath)
            results[f"{name}_step{step}_griffinlim"] = t
            print(f"{name}_step{step}_griffinlim (gt={text!r}): {t!r}")

# --- Controls ---
print("\n### Controls ###")
for step in (2000, 6000):
    for cname in ("control1_seenlyrics_unseenmelody", "control2_unseenlyrics_simplemelody"):
        cpath = f"{OUT_DIR}/{cname}_step{step}_vocoder.wav"
        if os.path.exists(cpath):
            t = transcribe(cpath)
            results[f"{cname}_step{step}"] = t
            print(f"{cname}_step{step}: {t!r}")

print(f"\nTotal transcriptions: {len(results)}")

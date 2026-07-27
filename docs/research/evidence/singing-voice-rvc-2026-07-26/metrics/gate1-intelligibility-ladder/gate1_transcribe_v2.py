#!/usr/bin/env python3
"""Gate 1 v2: transcribe confound-corrected ladder renders with Whisper.
Same caveats as v1's gate1_transcribe.py -- ASR proxy, not a validated
singing-intelligibility metric, not a substitute for human listening.
"""
from faster_whisper import WhisperModel

LADDER_DIR = "/var/lib/symthaea/training-runs/diffsinger/ladder_v2_out"

GROUND_TRUTH = {
    "01_me": "me",
    "02_sing_with_me": "sing with me",
    "03_wont_you_sing_with_me": "won't you sing with me",
    "04_wont_you_sing_along_with_me": "won't you sing along with me",
    "05_now_i_know_my_abc": "now I know my ABC",
    "06_full_closing_phrase": "now I know my ABC won't you sing along with me",
    "07_alphabet": "A B C D E F G H I J K L M N O P Q R S T U V W X Y and Z",
}

print("Loading Whisper model (small, CPU)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

for name in sorted(GROUND_TRUTH.keys()):
    path = f"{LADDER_DIR}/{name}.wav"
    segments, info = model.transcribe(path, language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"\n=== {name} ===")
    print(f"Ground truth: {GROUND_TRUTH[name]!r}")
    print(f"Whisper says: {text!r}")

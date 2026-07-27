#!/usr/bin/env python3
"""Build a real .ds inference file from a held-out CSD test-set item
(en001a, one of csd_en_acoustic.yaml's test_prefixes -- never trained on),
for verifying the trained checkpoint (SING-18 stretch goal). Uses the
ground-truth ph_seq/ph_dur we already generated in convert_csd.py plus a
real F0 curve extracted from the ground-truth wav via parselmouth (same
extractor DiffSinger's own binarizer uses: pe=parselmouth). Acoustic-only
inference needs just ph_seq/ph_dur/f0_seq/f0_timestep -- no note fields.
"""
import csv
import json
import sys

import numpy as np
import parselmouth

HOP_SIZE = 512
SAMPLE_RATE = 44100
F0_TIMESTEP = HOP_SIZE / SAMPLE_RATE
F0_MIN = 65
F0_MAX = 1100

name = sys.argv[1] if len(sys.argv) > 1 else "en001a"
raw_dir = "/var/lib/symthaea/training-runs/diffsinger/raw/benchmark-01"

with open(f"{raw_dir}/transcriptions.csv") as fh:
    reader = csv.DictReader(fh)
    row = next(r for r in reader if r["name"] == name)

wav_path = f"{raw_dir}/wavs/{name}.wav"
sound = parselmouth.Sound(wav_path)
pitch = sound.to_pitch_ac(
    time_step=F0_TIMESTEP, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX
)
f0 = pitch.selected_array["frequency"]
# Fill unvoiced (0 Hz) frames by forward/backward filling from nearest
# voiced frame -- the acoustic model expects a continuous curve.
voiced = f0 > 0
if voiced.any():
    idx = np.where(voiced, np.arange(len(f0)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = f0[idx]
    first_voiced = np.argmax(voiced)
    filled[:first_voiced] = f0[voiced][0]
else:
    filled = np.full_like(f0, 220.0)

ds_entry = {
    "offset": 0.0,
    "text": name,
    "ph_seq": row["ph_seq"],
    "ph_dur": row["ph_dur"],
    "f0_seq": " ".join(f"{x:.1f}" for x in filled),
    "f0_timestep": str(F0_TIMESTEP),
}

out_path = f"/var/lib/symthaea/training-runs/diffsinger/{name}.ds"
with open(out_path, "w") as fh:
    json.dump([ds_entry], fh)
print(f"wrote {out_path}: {len(row['ph_seq'].split())} phonemes, "
      f"{len(filled)} f0 frames ({len(filled) * F0_TIMESTEP:.1f}s)")

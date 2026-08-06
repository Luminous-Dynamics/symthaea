#!/usr/bin/env python3
"""Part 2: how much of the headline melody number survives an ORDER-SENSITIVE metric?

04_evaluate.py scores each voiced frame against its NEAREST target note
(min over the whole melody), which is order-blind. The order-sensitive
variant below splits voiced frames into len(melody) contiguous groups
(mirroring 03_reshape_pyworld.py's one-note-per-word assignment, which
asserts len(words) == len(melody)) and scores each group against the note
it was SUPPOSED to sing.

Run on the real recorded sung renders, so these are the actual pipeline's
outputs, not a reconstruction.
"""
import json
import numpy as np
import pyworld as pw
import soundfile as sf

AUDIO = "/var/lib/symthaea/training-runs/kokoro-world-vocoder/audio"
CONFIG = json.loads(open("/var/lib/symthaea/training-runs/kokoro-world-vocoder/config.json").read())


def f0_of(path):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, _t = pw.harvest(x.astype(np.float64), fs, frame_period=5.0)
    return f0[f0 > 0]


def nearest_note_metric(voiced_f0, melody):
    """Verbatim semantics of 04_evaluate.py: min over ALL targets."""
    log_f0 = np.log2(voiced_f0)
    log_t = np.log2(np.array(melody))
    err = np.min(np.abs(log_f0[:, None] - log_t[None, :]) * 1200.0, axis=1)
    return float(np.median(err)), float(np.mean(err < 50.0))


def expected_note_metric(voiced_f0, melody):
    """Order-sensitive: group i must sing melody[i], not 'whichever is closest'."""
    groups = np.array_split(voiced_f0, len(melody))
    errs = []
    per_group = []
    for g, hz in zip(groups, melody):
        if len(g) == 0:
            continue
        e = np.abs(np.log2(g) - np.log2(hz)) * 1200.0
        errs.append(e)
        per_group.append((hz, float(np.median(np.abs(np.log2(g) - np.log2(hz)) * 1200.0)),
                          float(np.median(g))))
    err = np.concatenate(errs)
    return float(np.median(err)), float(np.mean(err < 50.0)), per_group


print("=" * 84)
print("Order-blind (as shipped) vs order-sensitive, on the REAL recorded sung renders")
print("=" * 84)
for phrase in CONFIG["phrases"]:
    pid, melody = phrase["id"], phrase["melody_hz"]
    v = f0_of(f"{AUDIO}/{pid}_sung.wav")
    m_med, m_frac = nearest_note_metric(v, melody)
    o_med, o_frac, per_group = expected_note_metric(v, melody)
    print(f"\n{pid}  ({len(melody)} notes, \"{phrase['text']}\")")
    print(f"  order-blind    (shipped) : median {m_med:6.1f}c   frac<50c {m_frac:.3f}")
    print(f"  order-sensitive          : median {o_med:6.1f}c   frac<50c {o_frac:.3f}")
    print(f"  per-note (target Hz -> median err / observed median Hz):")
    for hz, err, obs in per_group:
        flag = "  <-- OFF" if err > 50 else ""
        print(f"      {hz:7.2f} Hz -> {err:8.1f}c   observed {obs:7.1f} Hz{flag}")

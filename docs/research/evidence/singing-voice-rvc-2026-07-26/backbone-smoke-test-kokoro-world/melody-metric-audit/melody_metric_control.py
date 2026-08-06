#!/usr/bin/env python3
"""Negative-control audit of 04_evaluate.py's melody_tracking_score.

Question: does "median 4.2 cents to nearest target note" demonstrate that the
render FOLLOWS THE MELODY, or only that its voiced frames sit near SOME note
of the scale?

The metric takes min() over ALL target notes, and 03_reshape_pyworld.py:176
sets every voiced frame of a word to a constant target note. So the metric may
be scoring the vocoder round-tripping the F0 it was just handed, and may be
unable to detect a wrong note at the wrong time.

Controls, all synthesized through the same WORLD path from the same source:
  CORRECT   - notes in the intended order          (261.63 then 392.00)
  SCRAMBLED - notes in REVERSED order              (392.00 then 261.63)  <- objectively wrong melody
  MONOTONE  - every voiced frame on ONE note       (261.63 throughout)   <- no melody at all
  SPOKEN    - untouched natural speech F0                                <- real negative control

If SCRAMBLED and MONOTONE score as well as CORRECT, the metric does not
measure melody following.
"""
import json
import numpy as np
import pyworld as pw
import soundfile as sf

AUDIO = "/var/lib/symthaea/training-runs/kokoro-world-vocoder/audio"
MELODY = [261.63, 392.00]  # config.json hello_world


# --- verbatim copy of 04_evaluate.py's scoring function ---------------------
def melody_tracking_score(x, fs, target_hz_sequence):
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, fs, frame_period=5.0)
    voiced = f0 > 0
    if voiced.sum() < 5:
        return None
    voiced_f0 = f0[voiced]
    log_f0 = np.log2(voiced_f0)
    log_targets = np.log2(np.array(target_hz_sequence))
    cents_err = np.min(np.abs(log_f0[:, None] - log_targets[None, :]) * 1200.0, axis=1)
    return {
        "median_cents_error_to_nearest_target_note": round(float(np.median(cents_err)), 1),
        "fraction_frames_within_50_cents_of_a_target_note": round(float(np.mean(cents_err < 50.0)), 3),
        "target_melody_range_semitones": round(float((log_targets.max() - log_targets.min()) * 12.0), 2),
        "observed_range_semitones": round(float((log_f0.max() - log_f0.min()) * 12.0), 2),
    }


def world_resynth(x, fs, f0_policy):
    """Analyze, replace F0 per policy, resynthesize. Same decomposition as 03."""
    x = x.astype(np.float64)
    f0, t = pw.harvest(x, fs, frame_period=5.0)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    new_f0 = f0_policy(f0.copy())
    y = pw.synthesize(new_f0, sp, ap, fs, frame_period=5.0)
    return y


def policy_notes(note_seq):
    """Split voiced frames into len(note_seq) contiguous groups, one note each.
    Mirrors 03's per-word constant-note assignment."""
    def apply(f0):
        voiced_idx = np.where(f0 > 0)[0]
        if len(voiced_idx) == 0:
            return f0
        groups = np.array_split(voiced_idx, len(note_seq))
        for g, hz in zip(groups, note_seq):
            f0[g] = hz
        return f0
    return apply


def main():
    src, fs = sf.read(f"{AUDIO}/hello_world_spoken.wav")
    if src.ndim > 1:
        src = src.mean(axis=1)

    arms = {
        "CORRECT   (261.63 -> 392.00)": world_resynth(src, fs, policy_notes([261.63, 392.00])),
        "SCRAMBLED (392.00 -> 261.63)": world_resynth(src, fs, policy_notes([392.00, 261.63])),
        "MONOTONE  (261.63 only)     ": world_resynth(src, fs, policy_notes([261.63])),
        "SPOKEN    (untouched F0)    ": src,
    }

    print("=" * 78)
    print("Metric under test: 04_evaluate.py::melody_tracking_score")
    print(f"Target melody scored against: {MELODY}")
    print("=" * 78)

    # Reference: does this harness reproduce the recorded number?
    ref, rfs = sf.read(f"{AUDIO}/hello_world_sung.wav")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)
    s = melody_tracking_score(ref, rfs, MELODY)
    print(f"\n[harness check] recorded hello_world_sung.wav")
    print(f"    {json.dumps(s)}")
    print(f"    results.json recorded: median 4.2c, frac 0.888  <- should match")

    print()
    for name, y in arms.items():
        s = melody_tracking_score(np.asarray(y), fs, MELODY)
        print(f"{name}  median={s['median_cents_error_to_nearest_target_note']:>6}c   "
              f"frac_within_50c={s['fraction_frames_within_50_cents_of_a_target_note']:>5}   "
              f"observed_range={s['observed_range_semitones']:>6} st")

    print()
    print("Interpretation: if SCRAMBLED and MONOTONE score comparably to CORRECT,")
    print("the metric cannot distinguish a right melody from a wrong one.")


if __name__ == "__main__":
    main()

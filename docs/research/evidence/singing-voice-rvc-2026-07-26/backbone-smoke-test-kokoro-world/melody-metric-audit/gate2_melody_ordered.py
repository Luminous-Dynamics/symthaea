#!/usr/bin/env python3
"""Gate 2 (hard 10-phrase suite) scored with a TRUE order-sensitive melody metric.

Gate 2 recorded WER only (`gate2_results.json`) -- melody accuracy on the hard
suite was never measured. This closes that gap and asks the question the WER
numbers alone cannot answer:

    when intelligibility collapses (rapid_letter_names WER 1.0,
    long_sustained_vowels 0.8, fricative_heavy 0.667), does the PITCH
    mechanism collapse with it, or does it hold?

If melody holds while WER collapses, that supports the bundle's existing
"shared Whisper-ASR weakness on this content type" reading and says the
backbone's pitch path is robust. If melody collapses too, it is a backbone
problem and the reading is wrong.

## Why this is a real order-sensitive metric, not an approximation

`gate2_03_reshape.py` synthesizes each word in COMPLETE ISOLATION and
concatenates them with `GAP_S = 0.06` of genuine time-domain silence --
literally `np.zeros` (line 193), not synthesized unvoiced frames. The final
RMS scaling is a constant multiply, so those samples stay exactly zero.

That means true per-word output spans are directly recoverable by splitting
on exact-zero runs. No equal-split approximation (the known limitation of
`melody_metric_ordered.py`), no G2P reconstruction. Word i's segment is then
scored against melody[i] -- the note it was actually supposed to sing.

Segment count is asserted against note count per phrase and reported rather
than forced, so a mis-segmentation shows up as a disclosed anomaly instead of
silently corrupting a score.
"""
import json
import numpy as np
import pyworld as pw
import soundfile as sf

BASE = "/var/lib/symthaea/training-runs/kokoro-world-vocoder"
GAP_S = 0.06
ZERO_EPS = 1e-9
MIN_GAP_FRAC = 0.5  # a real gap is at least half of GAP_S


def split_on_silence(x, fs):
    """Recover per-word spans from the exact-zero inter-word gaps."""
    silent = np.abs(x) < ZERO_EPS
    min_gap = int(GAP_S * MIN_GAP_FRAC * fs)
    spans, run_start = [], None
    gaps = []
    for i, s in enumerate(silent):
        if s and run_start is None:
            run_start = i
        elif not s and run_start is not None:
            if i - run_start >= min_gap:
                gaps.append((run_start, i))
            run_start = None
    if run_start is not None and len(silent) - run_start >= min_gap:
        gaps.append((run_start, len(silent)))

    cursor = 0
    for g0, g1 in gaps:
        if g0 > cursor:
            spans.append((cursor, g0))
        cursor = g1
    if cursor < len(x):
        spans.append((cursor, len(x)))
    return [s for s in spans if s[1] - s[0] > int(0.02 * fs)]


def per_word_melody(path, melody):
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    spans = split_on_silence(x, fs)
    rows, all_err = [], []
    for i, (a, b) in enumerate(spans):
        if i >= len(melody):
            break
        seg = x[a:b].astype(np.float64)
        f0, _t = pw.harvest(seg, fs, frame_period=5.0)
        v = f0[f0 > 0]
        if len(v) < 3:
            rows.append((melody[i], None, None, len(v)))
            continue
        err = np.abs(np.log2(v) - np.log2(melody[i])) * 1200.0
        rows.append((melody[i], float(np.median(err)), float(np.median(v)), len(v)))
        all_err.append(err)
    overall = float(np.median(np.concatenate(all_err))) if all_err else float("nan")
    frac = float(np.mean(np.concatenate(all_err) < 50.0)) if all_err else float("nan")
    return len(spans), rows, overall, frac


def order_blind(path, melody):
    """Verbatim 04_evaluate.py semantics, for side-by-side comparison."""
    x, fs = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f0, _t = pw.harvest(x.astype(np.float64), fs, frame_period=5.0)
    v = f0[f0 > 0]
    lt = np.log2(np.array(melody))
    err = np.min(np.abs(np.log2(v)[:, None] - lt[None, :]) * 1200.0, axis=1)
    return float(np.median(err)), float(np.mean(err < 50.0))


cfg = json.loads(open(f"{BASE}/gate2_config.json").read())
wer = {r["id"]: r["wer"] for r in json.loads(open(f"{BASE}/gate2_results.json").read())}

print("=" * 96)
print("Gate 2 hard suite: recorded WER vs TRUE order-sensitive melody accuracy")
print("(word spans recovered from the exact-zero inter-word gaps, not equal splits)")
print("=" * 96)
print(f"{'phrase':<24}{'WER':>6}{'notes':>7}{'segs':>6}"
      f"{'blind med':>11}{'ORDERED med':>13}{'frac<50c':>10}{'worst note':>12}")
print("-" * 96)

rows_out = []
for ph in cfg["phrases"]:
    pid, melody = ph["id"], ph["melody_hz"]
    path = f"{BASE}/gate2_audio/{pid}_sung.wav"
    nseg, rows, med, frac = per_word_melody(path, melody)
    b_med, _b_frac = order_blind(path, melody)
    worst = max((r[1] for r in rows if r[1] is not None), default=float("nan"))
    flag = "" if nseg == len(melody) else "  <-- SEGMENT/NOTE MISMATCH"
    print(f"{pid:<24}{wer[pid]:>6.3f}{len(melody):>7}{nseg:>6}"
          f"{b_med:>11.1f}{med:>13.1f}{frac:>10.3f}{worst:>12.1f}{flag}")
    rows_out.append({"id": pid, "wer": wer[pid], "n_notes": len(melody),
                     "n_segments": nseg, "order_blind_median_cents": round(b_med, 1),
                     "ordered_median_cents": round(med, 1),
                     "ordered_frac_within_50c": round(frac, 3),
                     "worst_note_cents": round(worst, 1),
                     "per_note": [{"target_hz": r[0],
                                   "median_cents_err": None if r[1] is None else round(r[1], 1),
                                   "observed_median_hz": None if r[2] is None else round(r[2], 1),
                                   "voiced_frames": r[3]} for r in rows]})

print("-" * 96)
ok = [r for r in rows_out if r["n_segments"] == r["n_notes"]]
print(f"phrases with clean segmentation: {len(ok)}/{len(rows_out)}")
if ok:
    hi = [r for r in ok if r["wer"] >= 0.5]
    lo = [r for r in ok if r["wer"] < 0.5]
    print(f"  high-WER (>=0.5, n={len(hi)}): ordered median "
          f"{np.mean([r['ordered_median_cents'] for r in hi]):.1f}c" if hi else "  high-WER: none")
    print(f"  low-WER  (< 0.5, n={len(lo)}): ordered median "
          f"{np.mean([r['ordered_median_cents'] for r in lo]):.1f}c" if lo else "  low-WER: none")
    w = np.array([r["wer"] for r in ok])
    m = np.array([r["ordered_median_cents"] for r in ok])
    if len(ok) > 2 and w.std() > 0 and m.std() > 0:
        print(f"  Pearson r(WER, ordered melody error) = {np.corrcoef(w, m)[0,1]:+.3f}  (n={len(ok)})")

json.dump(rows_out, open("gate2_melody_ordered_results.json", "w"), indent=2)
print("\nWrote gate2_melody_ordered_results.json")

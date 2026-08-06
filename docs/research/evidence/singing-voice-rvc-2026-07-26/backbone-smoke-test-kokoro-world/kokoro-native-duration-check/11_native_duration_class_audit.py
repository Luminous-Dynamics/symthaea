#!/usr/bin/env python3
"""Extends the Kokoro native-duration (pred_dur) validation across
phoneme classes and positions, per the reviewer's explicit request
after the 6-fricative finding. Structure:

1. Rule out an indexing bug: verify every ps[i] maps to a real vocab
   entry (no silent filtering) and pred_dur has exactly len(ps)+2
   entries (bos + one per ps char + eos), for every test phrase.
2. Build a per-phone boundary table (start/end sample from pred_dur)
   with phone-class-appropriate acoustic landmark detectors:
   - fricatives: high-band (>=3kHz) energy-fraction threshold crossing
   - stops: RMS-derivative burst location + post-burst periodicity onset
   - affricates: burst location (stop-like) + frication offset (fricative-like)
   - voiced obstruents: local RMS dip (approximate, disclosed as such)
   - vowels/sonorants: periodicity presence (control -- expect near-zero
     offset, since these are not where the fricative/stop mislocation
     hypothesis applies)
3. Report SIGNED boundary error (landmark_time - pred_dur_nominal_start)
   per phone/class/position, to distinguish a global fixed offset from
   a class-specific or context-dependent one.

This is a measurement/diagnosis pass only -- no synthesis change.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
FS = 24000
FRAME_SAMPLES = 600  # Kokoro's own pred_dur frame hop (25ms), verified exact

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
STOP_CHARS = set("ptk")
FRICATIVE_CHARS = set("fθsʃh")
AFFRICATE_CHARS = set("ʧʤ")
VOICED_OBSTRUENT_CHARS = set("bdgvðzʒ")  # ʤ handled as affricate, not here
STRESS = "ˈˌ"

PHRASES = {
    "fricative_heavy": "she sells seashells by the seashore",
    "consonant_clusters": "strong streams splashed strangely",
    "phrase_final_stops": "turn off the light and lock it",
    "repeated_syllables": "bye bye bye bye baby",
    "long_sustained_vowels": "moon over the blue lagoon",
    "semantically_unusual": "the clock ate my umbrella",
}


def classify_char(c):
    if c in VOWEL_CHARS:
        return "vowel"
    if c in SONORANT_CONSONANT_CHARS:
        return "sonorant"
    if c in STOP_CHARS:
        return "stop"
    if c in FRICATIVE_CHARS:
        return "fricative"
    if c in AFFRICATE_CHARS:
        return "affricate"
    if c in VOICED_OBSTRUENT_CHARS:
        return "voiced_obstruent"
    if c in STRESS or c == " ":
        return "marker"
    return "other"


def frame_windows(y, fs, hop=60, win=240):
    """Fine-grained (2.5ms hop, 10ms window) RMS / high-band-fraction /
    periodicity envelopes over the whole signal, computed once and
    reused for all landmark lookups (cheap, avoids re-windowing per
    query)."""
    n = len(y)
    times = []
    rms = []
    hf = []
    period = []
    half = win // 2
    for start in range(0, n, hop):
        c = start + half
        s0, s1 = max(0, c - half), min(n, c + half)
        seg = y[s0:s1]
        if len(seg) < 8:
            times.append(c / fs)
            rms.append(0.0)
            hf.append(0.0)
            period.append(0.0)
            continue
        rms.append(float(np.sqrt(np.mean(seg**2))))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        total = spec.sum() + 1e-12
        hf.append(float(spec[freqs >= 3000.0].sum() / total))
        # periodicity: normalized autocorrelation peak in 80-400Hz lag range
        seg2 = seg - seg.mean()
        if np.abs(seg2).sum() < 1e-9:
            period.append(0.0)
        else:
            ac = np.correlate(seg2, seg2, mode="full")[len(seg2) - 1:]
            ac0 = ac[0] + 1e-12
            lag_lo, lag_hi = int(fs / 400), min(len(ac) - 1, int(fs / 80))
            if lag_hi <= lag_lo:
                period.append(0.0)
            else:
                period.append(float(np.max(ac[lag_lo:lag_hi]) / ac0))
        times.append(c / fs)
    return np.array(times), np.array(rms), np.array(hf), np.array(period)


def lookup(times, arr, t):
    idx = np.searchsorted(times, t)
    idx = min(max(idx, 0), len(arr) - 1)
    return float(arr[idx])


def find_threshold_crossing(times, arr, t_search_start, t_search_end, thresh, rising):
    """First time in [start,end] where arr crosses thresh (rising: goes
    from below to at/above; falling: goes from at/above to below).
    Returns None if no crossing found."""
    i0 = np.searchsorted(times, t_search_start)
    i1 = np.searchsorted(times, t_search_end)
    i0, i1 = max(0, i0), min(len(arr), i1)
    for i in range(i0, max(i0, i1 - 1)):
        a, b = arr[i], arr[i + 1]
        if rising and a < thresh <= b:
            return float(times[i + 1])
        if not rising and a >= thresh > b:
            return float(times[i + 1])
    return None


def analyze_phrase(pid, text, pipeline):
    r = list(pipeline(text, voice="af_heart"))[0]
    ps = r.phonemes
    pd = r.pred_dur.tolist()
    audio = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)

    # Cross-check against the existing stage-1 spoken.wav where available,
    # to confirm determinism (same voice/text -> same audio) rather than
    # assuming it.
    existing_path = BASE / "gate2_audio" / f"{pid}_spoken.wav"
    if existing_path.exists():
        y_existing, fs_existing = sf.read(str(existing_path))
        assert fs_existing == FS
        match = len(y_existing) == len(audio) and np.allclose(y_existing, audio, atol=1e-6)
        print(f"  [determinism check vs stage-1 spoken.wav: {'MATCH' if match else 'DIFFERS'}]")

    # Rule out indexing bug
    vocab = pipeline.model.vocab
    ids_raw = [vocab.get(p) for p in ps]
    n_none = sum(1 for i in ids_raw if i is None)
    assert n_none == 0, f"{pid}: {n_none} phoneme chars not in vocab -- filtering would break alignment"
    assert len(pd) == len(ps) + 2, f"{pid}: pred_dur length {len(pd)} != len(ps)+2 {len(ps)+2}"

    cum = [0]
    for d in pd:
        cum.append(cum[-1] + d)

    times, rms, hf, period = frame_windows(audio, FS)

    records = []
    for i, ch in enumerate(ps):
        cls = classify_char(ch)
        if cls in ("marker",):
            continue
        pdi = i + 1
        start_s = cum[pdi] * FRAME_SAMPLES
        end_s = cum[pdi + 1] * FRAME_SAMPLES
        t_start, t_end = start_s / FS, end_s / FS
        records.append({
            "phrase": pid, "char": ch, "class": cls,
            "nominal_start_s": t_start, "nominal_end_s": t_end,
            "nominal_dur_ms": (t_end - t_start) * 1000.0,
        })

    # Class-appropriate landmark detection
    search_back, search_fwd = 0.09, 0.06  # seconds
    for rec in records:
        t0, t1 = rec["nominal_start_s"], rec["nominal_end_s"]
        cls = rec["class"]
        if cls == "fricative":
            onset = find_threshold_crossing(times, hf, t0 - search_back, t1 + search_fwd, 0.4, rising=True)
            offset = find_threshold_crossing(times, hf, t0 - search_back, t1 + search_fwd, 0.4, rising=False)
            rec["landmark_onset_s"] = onset
            rec["landmark_offset_s"] = offset
        elif cls == "stop":
            # burst: steepest RMS rise in the search window
            i0 = np.searchsorted(times, t0 - search_back)
            i1 = np.searchsorted(times, t1 + search_fwd)
            i0, i1 = max(0, i0), min(len(rms) - 1, i1)
            if i1 > i0 + 1:
                drms = np.diff(rms[i0:i1])
                burst_i = i0 + int(np.argmax(drms))
                rec["landmark_onset_s"] = float(times[burst_i])
            else:
                rec["landmark_onset_s"] = None
            voicing_on = find_threshold_crossing(times, period, t0, t1 + search_fwd, 0.35, rising=True)
            rec["landmark_offset_s"] = voicing_on
        elif cls == "affricate":
            i0 = np.searchsorted(times, t0 - search_back)
            i1 = np.searchsorted(times, t1)
            i0, i1 = max(0, i0), min(len(rms) - 1, i1)
            if i1 > i0 + 1:
                drms = np.diff(rms[i0:i1])
                burst_i = i0 + int(np.argmax(drms))
                rec["landmark_onset_s"] = float(times[burst_i])
            else:
                rec["landmark_onset_s"] = None
            offset = find_threshold_crossing(times, hf, t0, t1 + search_fwd, 0.4, rising=False)
            rec["landmark_offset_s"] = offset
        elif cls == "voiced_obstruent":
            # approximate: local RMS minimum (voiced obstruents typically
            # dip in amplitude relative to neighboring vowels) -- disclosed
            # as the roughest proxy of the five.
            i0 = np.searchsorted(times, t0 - search_back)
            i1 = np.searchsorted(times, t1 + search_fwd)
            i0, i1 = max(0, i0), min(len(rms) - 1, i1)
            if i1 > i0:
                dip_i = i0 + int(np.argmin(rms[i0:i1]))
                rec["landmark_onset_s"] = float(times[dip_i]) - 0.02  # crude window around dip
            else:
                rec["landmark_onset_s"] = None
            rec["landmark_offset_s"] = None
        else:  # vowel / sonorant -- control
            onset = find_threshold_crossing(times, period, t0 - search_back, t1, 0.35, rising=True)
            rec["landmark_onset_s"] = onset
            rec["landmark_offset_s"] = None

        if rec.get("landmark_onset_s") is not None:
            rec["signed_offset_ms"] = (rec["landmark_onset_s"] - t0) * 1000.0
        else:
            rec["signed_offset_ms"] = None

    return records


def main():
    pipeline = KPipeline(lang_code="a")
    all_records = []
    for pid, text in PHRASES.items():
        print(f"=== {pid}: {text!r} ===")
        recs = analyze_phrase(pid, text, pipeline)
        all_records.extend(recs)

    print()
    print(f"{'phrase':22s} {'char':6s} {'class':16s} {'nom_start_ms':>12s} {'nom_dur_ms':>10s} {'signed_offset_ms':>17s}")
    for r in all_records:
        off = r["signed_offset_ms"]
        off_s = f"{off:8.1f}" if off is not None else "     n/a"
        print(f"{r['phrase']:22s} {r['char']:6s} {r['class']:16s} {r['nominal_start_s']*1000:12.1f} {r['nominal_dur_ms']:10.1f} {off_s:>17s}")

    print()
    print("--- Aggregate signed offset by class (ms; negative = landmark occurs BEFORE nominal start) ---")
    by_class = {}
    for r in all_records:
        if r["signed_offset_ms"] is None:
            continue
        by_class.setdefault(r["class"], []).append(r["signed_offset_ms"])
    for cls, vals in by_class.items():
        arr = np.array(vals)
        print(f"{cls:18s} n={len(arr):3d}  mean={arr.mean():7.1f}  std={arr.std():6.1f}  min={arr.min():7.1f}  max={arr.max():7.1f}")

    out_path = BASE / "native_duration_class_audit.json"
    out_path.write_text(json.dumps(all_records, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

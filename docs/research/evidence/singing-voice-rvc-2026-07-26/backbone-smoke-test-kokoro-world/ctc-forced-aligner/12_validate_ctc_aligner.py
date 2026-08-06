#!/usr/bin/env python3
"""Validates CtcPhoneAligner against:
  1. Kokoro's own pred_dur (discrepancy signal, not ground truth)
  2. The same class-appropriate acoustic landmarks used in the
     native-duration-class audit (fricative hf_frac threshold,
     stop RMS-burst, vowel/sonorant periodicity onset)
  3. Phone order preservation

Reuses the exact phrase set + landmark code from
11_native_duration_class_audit.py so results are directly comparable.
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro import KPipeline

from phone_aligner import CtcPhoneAligner, PhoneSpan

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
FS = 24000
FRAME_SAMPLES = 600

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
STOP_CHARS = set("ptk")
FRICATIVE_CHARS = set("fθsʃh")
AFFRICATE_CHARS = set("ʧʤ")
VOICED_OBSTRUENT_CHARS = set("bdgvðzʒ")
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
    n = len(y)
    times, rms, hf, period = [], [], [], []
    half = win // 2
    for start in range(0, n, hop):
        c = start + half
        s0, s1 = max(0, c - half), min(n, c + half)
        seg = y[s0:s1]
        if len(seg) < 8:
            times.append(c / fs); rms.append(0.0); hf.append(0.0); period.append(0.0)
            continue
        rms.append(float(np.sqrt(np.mean(seg**2))))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        total = spec.sum() + 1e-12
        hf.append(float(spec[freqs >= 3000.0].sum() / total))
        seg2 = seg - seg.mean()
        if np.abs(seg2).sum() < 1e-9:
            period.append(0.0)
        else:
            ac = np.correlate(seg2, seg2, mode="full")[len(seg2) - 1:]
            ac0 = ac[0] + 1e-12
            lag_lo, lag_hi = int(fs / 400), min(len(ac) - 1, int(fs / 80))
            period.append(float(np.max(ac[lag_lo:lag_hi]) / ac0) if lag_hi > lag_lo else 0.0)
        times.append(c / fs)
    return np.array(times), np.array(rms), np.array(hf), np.array(period)


def find_crossing(times, arr, t0, t1, thresh, rising):
    i0, i1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
    i0, i1 = max(0, i0), min(len(arr), i1)
    for i in range(i0, max(i0, i1 - 1)):
        a, b = arr[i], arr[i + 1]
        if rising and a < thresh <= b:
            return float(times[i + 1])
        if not rising and a >= thresh > b:
            return float(times[i + 1])
    return None


def landmark_for(cls, times, rms, hf, period, t0, t1, search_back=0.09, search_fwd=0.06):
    if cls == "fricative":
        return find_crossing(times, hf, t0 - search_back, t1 + search_fwd, 0.4, True)
    if cls == "stop":
        i0, i1 = np.searchsorted(times, t0 - search_back), np.searchsorted(times, t1 + search_fwd)
        i0, i1 = max(0, i0), min(len(rms) - 1, i1)
        if i1 > i0 + 1:
            return float(times[i0 + int(np.argmax(np.diff(rms[i0:i1])))])
        return None
    if cls in ("vowel", "sonorant"):
        return find_crossing(times, period, t0 - search_back, t1, 0.35, True)
    return None


def main():
    pipeline = KPipeline(lang_code="a")
    aligner = CtcPhoneAligner()

    all_rows = []
    for pid, text in PHRASES.items():
        print(f"=== {pid} ===")
        r = list(pipeline(text, voice="af_heart"))[0]
        ps = r.phonemes
        pd = r.pred_dur.tolist()
        audio = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)

        cum = [0]
        for d in pd:
            cum.append(cum[-1] + d)

        # Build native (pred_dur) spans for non-marker chars, keyed by
        # original ps index (needed to re-join with CTC spans below).
        native_by_idx = {}
        for i, ch in enumerate(ps):
            cls = classify_char(ch)
            if cls == "marker":
                continue
            pdi = i + 1
            native_by_idx[i] = {
                "char": ch, "class": cls,
                "start": cum[pdi] * FRAME_SAMPLES, "end": cum[pdi + 1] * FRAME_SAMPLES,
            }

        result = aligner.align(audio, FS, ps)
        if result.global_warnings:
            print("  aligner warnings:", result.global_warnings)
        print("  phone_order_ok:", result.phone_order_ok, " n_ctc_spans:", len(result.spans))

        times, rms, hf, period = frame_windows(audio, FS)

        # transduce() dropped markers in the same order native_by_idx did
        # (both skip STRESS/space) -- re-derive the same ordered index list.
        from misaki_to_espeak import transduce
        triples, _unknown = transduce(ps)
        for k, (char, orig_idx, _tok) in enumerate(triples):
            if k >= len(result.spans):
                break
            span = result.spans[k]
            nat = native_by_idx.get(orig_idx)
            if nat is None:
                continue
            cls = nat["class"]
            t0_ctc, t1_ctc = span.start_sample / FS, span.end_sample / FS
            landmark_t = landmark_for(cls, times, rms, hf, period, t0_ctc, t1_ctc)
            landmark_native_t = landmark_for(cls, times, rms, hf, period, nat["start"] / FS, nat["end"] / FS)
            row = {
                "phrase": pid, "char": char, "class": cls,
                "native_start_ms": nat["start"] / FS * 1000,
                "ctc_start_ms": t0_ctc * 1000,
                "native_vs_ctc_ms": (t0_ctc - nat["start"] / FS) * 1000,
                "ctc_confidence": span.confidence,
                "ctc_landmark_offset_ms": (landmark_t - t0_ctc) * 1000 if landmark_t is not None else None,
                "native_landmark_offset_ms": (landmark_native_t - nat["start"] / FS) * 1000 if landmark_native_t is not None else None,
            }
            all_rows.append(row)

    print()
    hdr = f"{'phrase':22s} {'char':6s} {'class':14s} {'nat_v_ctc_ms':>13s} {'ctc_conf':>9s} {'ctc_land_ms':>12s} {'nat_land_ms':>12s}"
    print(hdr)
    for r in all_rows:
        cl = r["ctc_landmark_offset_ms"]
        nl = r["native_landmark_offset_ms"]
        cl_s = f"{cl:8.1f}" if cl is not None else "     n/a"
        nl_s = f"{nl:8.1f}" if nl is not None else "     n/a"
        print(f"{r['phrase']:22s} {r['char']:6s} {r['class']:14s} {r['native_vs_ctc_ms']:13.1f} {r['ctc_confidence']:9.3f} {cl_s:>12s} {nl_s:>12s}")

    print()
    print("--- CTC landmark offset (should be near 0 if CTC boundary IS the acoustic landmark) ---")
    by_class_ctc, by_class_nat = {}, {}
    for r in all_rows:
        if r["ctc_landmark_offset_ms"] is not None:
            by_class_ctc.setdefault(r["class"], []).append(r["ctc_landmark_offset_ms"])
        if r["native_landmark_offset_ms"] is not None:
            by_class_nat.setdefault(r["class"], []).append(r["native_landmark_offset_ms"])
    for cls in by_class_ctc:
        a = np.array(by_class_ctc[cls])
        b = np.array(by_class_nat.get(cls, []))
        print(f"{cls:14s} CTC:  n={len(a):3d} mean={a.mean():7.1f} std={a.std():6.1f}   |  native: n={len(b):3d} mean={(b.mean() if len(b) else float('nan')):7.1f} std={(b.std() if len(b) else float('nan')):6.1f}")

    out_path = BASE / "ctc_aligner_validation.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

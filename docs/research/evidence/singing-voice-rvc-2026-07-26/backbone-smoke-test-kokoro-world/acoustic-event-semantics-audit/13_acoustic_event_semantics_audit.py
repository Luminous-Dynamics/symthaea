#!/usr/bin/env python3
"""Bounded acoustic-event semantics audit, per the reviewer's explicit
framing: a single "phoneme boundary" is not the object the renderer
needs. Resolves two questions in one pass:

Part 1 (stops): does CTC's phone span correspond to
  [closure_onset, voicing_onset) with the burst inside it, or something
  else? Is the earlier "+17ms mean bias" a real placement error or a
  metric-definition mismatch (landmark = burst, CTC's own convention
  may differ)?

Part 2 (fricatives): do MULTIPLE model-free landmarks (high-band
  energy, spectral flatness, ZCR, spectral flux) agree on the ~40ms
  early-realization skew found earlier (which used high-band energy
  only)? And separately: how much of the STABLE FRICATION CORE (not
  just the onset instant) falls inside the CTC-proposed span -- the
  actually decision-relevant quantity for raw-waveform extraction.

Reuses the same 6-phrase corpus and CtcPhoneAligner already validated.
"""
import json
from pathlib import Path

import numpy as np
from kokoro import KPipeline

from phone_aligner import CtcPhoneAligner
from misaki_to_espeak import transduce

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
# Stops proper (per the reviewer's Part 1 list) also includes voiced
# b/d/g -- treat those as "stop" for THIS audit even though the wider
# codebase buckets them as voiced_obstruent (manner, not voicing, is
# what matters for closure/burst/voicing-onset semantics).
STOP_LIKE_CHARS = set("ptkbdɡ")

PHRASES = {
    "fricative_heavy": "she sells seashells by the seashore",
    "consonant_clusters": "strong streams splashed strangely",
    "phrase_final_stops": "turn off the light and lock it",
    "repeated_syllables": "bye bye bye bye baby",
    "long_sustained_vowels": "moon over the blue lagoon",
    "semantically_unusual": "the clock ate my umbrella",
}


def classify_char(c):
    if c in STOP_LIKE_CHARS:
        return "stop"
    if c in FRICATIVE_CHARS:
        return "fricative"
    if c in AFFRICATE_CHARS:
        return "affricate"
    if c in VOWEL_CHARS:
        return "vowel"
    if c in SONORANT_CONSONANT_CHARS:
        return "sonorant"
    if c in VOICED_OBSTRUENT_CHARS:
        return "voiced_obstruent"
    if c in STRESS or c == " ":
        return "marker"
    return "other"


def analysis_frames(y, fs, hop=30, win=240):
    """Finer hop (1.25ms) than the earlier audits for event-level
    precision. Computes RMS, high-band(>=3kHz) fraction, ZCR, spectral
    flatness (Wiener entropy), and frame-to-frame spectral flux."""
    n = len(y)
    times, rms, hf, zcr, flat, flux = [], [], [], [], [], []
    half = win // 2
    prev_spec = None
    for start in range(0, n, hop):
        c = start + half
        s0, s1 = max(0, c - half), min(n, c + half)
        seg = y[s0:s1]
        if len(seg) < 8:
            times.append(c / fs); rms.append(0.0); hf.append(0.0)
            zcr.append(0.0); flat.append(0.0); flux.append(0.0)
            prev_spec = None
            continue
        rms.append(float(np.sqrt(np.mean(seg**2))))
        zcr.append(float(np.mean(np.abs(np.diff(np.sign(seg)))) / 2.0))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-12
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / fs)
        total = spec.sum()
        hf.append(float(spec[freqs >= 3000.0].sum() / total))
        gmean = np.exp(np.mean(np.log(spec)))
        amean = np.mean(spec)
        flat.append(float(gmean / amean))
        if prev_spec is not None and prev_spec.shape == spec.shape:
            diff = spec - prev_spec
            flux.append(float(np.sqrt(np.sum(np.maximum(diff, 0.0) ** 2))))
        else:
            flux.append(0.0)
        prev_spec = spec
        times.append(c / fs)
    return (np.array(times), np.array(rms), np.array(hf), np.array(zcr),
            np.array(flat), np.array(flux))


def crossing(times, arr, t0, t1, thresh, rising):
    i0, i1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
    i0, i1 = max(0, i0), min(len(arr), i1)
    for i in range(i0, max(i0, i1 - 1)):
        a, b = arr[i], arr[i + 1]
        if rising and a < thresh <= b:
            return float(times[i + 1])
        if not rising and a >= thresh > b:
            return float(times[i + 1])
    return None


def argextreme(times, arr, t0, t1, mode):
    i0, i1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
    i0, i1 = max(0, i0), min(len(arr), i1)
    if i1 <= i0:
        return None
    seg = arr[i0:i1]
    idx = i0 + (int(np.argmax(seg)) if mode == "max" else int(np.argmin(seg)))
    return float(times[idx])


def stable_run(times, arr, t0, t1, thresh, min_dur_s=0.02):
    """Longest contiguous run within [t0,t1] where arr>=thresh for at
    least min_dur_s. Returns (run_start, run_end) or (None, None)."""
    i0, i1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
    i0, i1 = max(0, i0), min(len(arr), i1)
    best = (None, None, 0.0)
    i = i0
    while i < i1:
        if arr[i] >= thresh:
            j = i
            while j < i1 and arr[j] >= thresh:
                j += 1
            dur = times[j - 1] - times[i] if j > i else 0.0
            if dur > best[2]:
                best = (float(times[i]), float(times[j - 1]), dur)
            i = j
        else:
            i += 1
    if best[2] >= min_dur_s:
        return best[0], best[1]
    return None, None


def main():
    pipeline = KPipeline(lang_code="a")
    aligner = CtcPhoneAligner()

    stop_rows, fricative_rows, affricate_rows = [], [], []

    for pid, text in PHRASES.items():
        r = list(pipeline(text, voice="af_heart"))[0]
        ps = r.phonemes
        pd = r.pred_dur.tolist()
        audio = r.audio.numpy() if hasattr(r.audio, "numpy") else np.asarray(r.audio)

        cum = [0]
        for d in pd:
            cum.append(cum[-1] + d)
        native_by_idx = {}
        for i, ch in enumerate(ps):
            cls = classify_char(ch)
            if cls == "marker":
                continue
            pdi = i + 1
            native_by_idx[i] = {"start": cum[pdi] * FRAME_SAMPLES, "end": cum[pdi + 1] * FRAME_SAMPLES, "class": cls}

        result = aligner.align(audio, FS, ps)
        triples, _unknown = transduce(ps)

        times, rms, hf, zcr, flat, flux = analysis_frames(audio, FS)

        for k, (char, orig_idx, _tok) in enumerate(triples):
            if k >= len(result.spans):
                break
            nat = native_by_idx.get(orig_idx)
            if nat is None:
                continue
            cls = nat["class"]
            span = result.spans[k]
            ctc_t0, ctc_t1 = span.start_sample / FS, span.end_sample / FS
            nat_t0, nat_t1 = nat["start"] / FS, nat["end"] / FS

            if cls == "stop":
                search0, search1 = ctc_t0 - 0.10, ctc_t1 + 0.08
                # closure onset: last sustained low-RMS point before the burst
                burst_flux_t = argextreme(times, flux, search0, search1, "max")
                burst_rms_t = argextreme(times, np.diff(np.concatenate([rms[:1], rms])), search0, search1, "max")
                closure_onset_t = None
                if burst_flux_t is not None:
                    i0 = np.searchsorted(times, search0)
                    ib = np.searchsorted(times, burst_flux_t)
                    seg_rms = rms[i0:ib]
                    if len(seg_rms) > 2:
                        # closure = contiguous quiet run ending at the burst
                        thresh = np.percentile(rms[i0:np.searchsorted(times, search1)], 40)
                        j = ib - 1
                        while j > i0 and rms[j] < thresh:
                            j -= 1
                        closure_onset_t = float(times[j])
                voicing_onset_t = crossing(times, zcr, burst_flux_t if burst_flux_t else ctc_t0, search1, 0.15, rising=False) \
                    if burst_flux_t is not None else None
                stop_rows.append({
                    "phrase": pid, "char": char,
                    "native_start_ms": nat_t0 * 1000, "native_end_ms": nat_t1 * 1000,
                    "ctc_start_ms": ctc_t0 * 1000, "ctc_end_ms": ctc_t1 * 1000,
                    "ctc_confidence": span.confidence,
                    "closure_onset_ms": closure_onset_t * 1000 if closure_onset_t else None,
                    "burst_ms": burst_flux_t * 1000 if burst_flux_t else None,
                    "voicing_onset_ms": voicing_onset_t * 1000 if voicing_onset_t else None,
                    "burst_inside_ctc": (burst_flux_t is not None and ctc_t0 <= burst_flux_t <= ctc_t1),
                    "closure_near_ctc_start_ms": (closure_onset_t * 1000 - ctc_t0 * 1000) if closure_onset_t else None,
                })

            elif cls == "fricative":
                search0, search1 = ctc_t0 - 0.10, ctc_t1 + 0.06
                hf_onset = crossing(times, hf, search0, search1, 0.4, rising=True)
                flat_onset = crossing(times, flat, search0, search1, 0.25, rising=True)
                zcr_onset = crossing(times, zcr, search0, search1, 0.25, rising=True)
                flux_peak = argextreme(times, flux, search0, search1, "max")
                core_s, core_e = stable_run(times, hf, search0, search1, 0.4, min_dur_s=0.02)
                overlap = None
                if core_s is not None:
                    inter_s, inter_e = max(core_s, ctc_t0), min(core_e, ctc_t1)
                    inter = max(0.0, inter_e - inter_s)
                    core_dur = core_e - core_s
                    overlap = inter / core_dur if core_dur > 0 else None
                fricative_rows.append({
                    "phrase": pid, "char": char,
                    "native_start_ms": nat_t0 * 1000, "ctc_start_ms": ctc_t0 * 1000, "ctc_end_ms": ctc_t1 * 1000,
                    "ctc_confidence": span.confidence,
                    "hf_onset_offset_ms": (hf_onset - ctc_t0) * 1000 if hf_onset else None,
                    "flat_onset_offset_ms": (flat_onset - ctc_t0) * 1000 if flat_onset else None,
                    "zcr_onset_offset_ms": (zcr_onset - ctc_t0) * 1000 if zcr_onset else None,
                    "flux_peak_offset_ms": (flux_peak - ctc_t0) * 1000 if flux_peak else None,
                    "core_start_ms": core_s * 1000 if core_s else None,
                    "core_end_ms": core_e * 1000 if core_e else None,
                    "core_dur_ms": (core_e - core_s) * 1000 if core_s else None,
                    "core_fraction_inside_ctc_span": overlap,
                })

            elif cls == "affricate":
                search0, search1 = ctc_t0 - 0.10, ctc_t1 + 0.08
                burst_flux_t = argextreme(times, flux, search0, search1, "max")
                core_s, core_e = stable_run(times, hf, search0, search1, 0.4, min_dur_s=0.02)
                affricate_rows.append({
                    "phrase": pid, "char": char, "ctc_start_ms": ctc_t0 * 1000, "ctc_end_ms": ctc_t1 * 1000,
                    "burst_ms": burst_flux_t * 1000 if burst_flux_t else None,
                    "core_start_ms": core_s * 1000 if core_s else None, "core_end_ms": core_e * 1000 if core_e else None,
                    "burst_inside_ctc": (burst_flux_t is not None and ctc_t0 <= burst_flux_t <= ctc_t1),
                })

    print(f"n_stops={len(stop_rows)} n_fricatives={len(fricative_rows)} n_affricates={len(affricate_rows)}")
    print()
    print("=== PART 1: STOPS ===")
    print(f"{'phrase':22s} {'ch':3s} {'CTC[start,end]ms':18s} {'closure_ms':>11s} {'burst_ms':>9s} {'voice_on_ms':>12s} {'burst_in_ctc':>13s}")
    n_burst_in = 0
    for r in stop_rows:
        b_in = r["burst_inside_ctc"]
        n_burst_in += int(b_in)
        cl = f"{r['closure_onset_ms']:.1f}" if r["closure_onset_ms"] is not None else "n/a"
        bu = f"{r['burst_ms']:.1f}" if r["burst_ms"] is not None else "n/a"
        vo = f"{r['voicing_onset_ms']:.1f}" if r["voicing_onset_ms"] is not None else "n/a"
        print(f"{r['phrase']:22s} {r['char']:3s} [{r['ctc_start_ms']:7.1f},{r['ctc_end_ms']:7.1f}] {cl:>11s} {bu:>9s} {vo:>12s} {str(b_in):>13s}")
    print(f"\nburst-inside-CTC-span rate: {n_burst_in}/{len(stop_rows)}")
    closure_deltas = [r["closure_near_ctc_start_ms"] for r in stop_rows if r["closure_near_ctc_start_ms"] is not None]
    if closure_deltas:
        arr = np.array(closure_deltas)
        print(f"closure_onset - ctc_start: mean={arr.mean():.1f}ms std={arr.std():.1f}ms (negative = closure starts before CTC's span)")

    print()
    print("=== PART 2: FRICATIVES ===")
    print(f"{'phrase':22s} {'ch':3s} {'hf':>7s} {'flat':>7s} {'zcr':>7s} {'flux_pk':>8s} {'core_dur_ms':>12s} {'core_in_ctc':>11s}")
    for r in fricative_rows:
        def fmt(x):
            return f"{x:6.1f}" if x is not None else "   n/a"
        cdur = f"{r['core_dur_ms']:.1f}" if r["core_dur_ms"] is not None else "n/a"
        cov = f"{r['core_fraction_inside_ctc_span']:.2f}" if r["core_fraction_inside_ctc_span"] is not None else "n/a"
        print(f"{r['phrase']:22s} {r['char']:3s} {fmt(r['hf_onset_offset_ms'])} {fmt(r['flat_onset_offset_ms'])} {fmt(r['zcr_onset_offset_ms'])} {fmt(r['flux_peak_offset_ms'])} {cdur:>12s} {cov:>11s}")

    print()
    print("--- Part 2 aggregate: onset-vs-CTC-start offset by landmark (ms) ---")
    for key, label in [("hf_onset_offset_ms", "high-band energy"), ("flat_onset_offset_ms", "spectral flatness"),
                        ("zcr_onset_offset_ms", "ZCR"), ("flux_peak_offset_ms", "spectral flux peak")]:
        vals = [r[key] for r in fricative_rows if r[key] is not None]
        if vals:
            arr = np.array(vals)
            print(f"{label:20s} n={len(arr):3d} mean={arr.mean():7.1f} std={arr.std():6.1f}")
    coverage = [r["core_fraction_inside_ctc_span"] for r in fricative_rows if r["core_fraction_inside_ctc_span"] is not None]
    if coverage:
        arr = np.array(coverage)
        print(f"\nstable-core fraction inside CTC span: n={len(arr):3d} mean={arr.mean():.3f} std={arr.std():.3f} min={arr.min():.3f} max={arr.max():.3f}")

    print()
    print("=== affricates (n=%d, descriptive only) ===" % len(affricate_rows))
    for r in affricate_rows:
        print(r)

    out = {"stops": stop_rows, "fricatives": fricative_rows, "affricates": affricate_rows}
    out_path = BASE / "acoustic_event_semantics_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

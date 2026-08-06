#!/usr/bin/env python3
"""Builds the synthesis-event record the reviewer specified, fixing the
two instrumentation limitations disclosed in the acoustic-event
semantics audit:

1. Overlapping search windows for closely-spaced phones (two adjacent
   stops got IDENTICAL closure/burst timestamps because their windows
   overlapped and both found the same nearby transient). Fixed by
   clamping every phone's search window to the neighboring phones' OWN
   CTC span boundaries -- a phone's event search can never cross into
   where its neighbor's CTC span begins/ends.
2. The flux-detector/adjacent-stop confound for fricatives in cluster-
   heavy contexts is mitigated by the same neighbor-clamping (a
   fricative's forward/backward search can no longer wander into an
   adjacent stop's own span).

Produces one SynthesisEvent record per non-marker phone per phrase,
with class-specific sub-events:
  stop:       closure_start, burst_peak, voicing_onset
  fricative:  noise_onset, stable_core_start, stable_core_end, noise_offset
  affricate:  closure_start, burst_peak, stable_core_start, stable_core_end
  other:      acoustic_event_start/end only (periodicity onset), not the
              focus of this pass
"""
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np
from kokoro import KPipeline

from phone_aligner import CtcPhoneAligner
from misaki_to_espeak import transduce

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
FS = 24000
FRAME_SAMPLES = 600
MAX_BACK_S = 0.15
MAX_FWD_S = 0.15

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
STOP_LIKE_CHARS = set("ptkbdɡ")
FRICATIVE_CHARS = set("fθsʃh")
AFFRICATE_CHARS = set("ʧʤ")
VOICED_OBSTRUENT_CHARS = set("vðzʒ")  # b/d/g moved to stop-like; ʤ is affricate
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


@dataclass
class SynthesisEvent:
    phone: str
    phone_class: str
    native_start_ms: float
    native_end_ms: float
    ctc_start_ms: float
    ctc_end_ms: float
    ctc_confidence: float
    event_type: str
    event_confidence: float
    acoustic_event_start_ms: Optional[float] = None
    acoustic_event_end_ms: Optional[float] = None
    preservation_start_ms: Optional[float] = None
    preservation_end_ms: Optional[float] = None
    crossfade_start_ms: Optional[float] = None
    crossfade_end_ms: Optional[float] = None
    # stop sub-events
    closure_start_ms: Optional[float] = None
    burst_peak_ms: Optional[float] = None
    voicing_onset_ms: Optional[float] = None
    # fricative sub-events
    noise_onset_ms: Optional[float] = None
    stable_core_start_ms: Optional[float] = None
    stable_core_end_ms: Optional[float] = None
    noise_offset_ms: Optional[float] = None
    warnings: list = field(default_factory=list)


def analysis_frames(y, fs, hop=30, win=240):
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
        flat.append(float(gmean / np.mean(spec)))
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
    if t1 <= t0:
        return None
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
    if t1 <= t0:
        return None
    i0, i1 = np.searchsorted(times, t0), np.searchsorted(times, t1)
    i0, i1 = max(0, i0), min(len(arr), i1)
    if i1 <= i0:
        return None
    seg = arr[i0:i1]
    idx = i0 + (int(np.argmax(seg)) if mode == "max" else int(np.argmin(seg)))
    return float(times[idx])


def stable_run(times, arr, t0, t1, thresh, min_dur_s=0.02):
    if t1 <= t0:
        return None, None
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


def detect_stop_closure_and_burst(times, rms, back_limit, fwd_limit, min_closure_s=0.015):
    """Replaces the earlier global-flux-argmax burst detector, which was
    found (via direct trace inspection on phrase_final_stops' /k/) to
    pick the loudest spectral change ANYWHERE in the search window --
    often a busy vowel region with more natural modulation than a real
    but acoustically weaker stop burst, landing 100+ms from the true
    burst despite the true burst sitting cleanly inside the CTC span.
    Closure-anchored instead: find low-RMS (near-silence) runs at least
    `min_closure_s` long, take the LAST such run in the window (closest
    to the CTC span, matching where a stop's closure should be), and the
    burst is the first real RMS rise immediately after it ends. Returns
    (closure_start, closure_end, burst_t) -- any element may be None."""
    if fwd_limit <= back_limit:
        return None, None, None
    i0, i1 = np.searchsorted(times, back_limit), np.searchsorted(times, fwd_limit)
    i0, i1 = max(0, i0), min(len(rms), i1)
    if i1 <= i0:
        return None, None, None
    window_rms = rms[i0:i1]
    silence_thresh = max(float(np.percentile(window_rms, 20)), 0.003)
    runs = []
    i = i0
    while i < i1:
        if rms[i] < silence_thresh:
            j = i
            while j < i1 and rms[j] < silence_thresh:
                j += 1
            dur = times[j - 1] - times[i] if j > i else 0.0
            if dur >= min_closure_s:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return None, None, None
    ci0, ci1 = runs[-1]  # last (latest) qualifying closure in the window
    closure_start = float(times[ci0])
    closure_end = float(times[ci1 - 1])
    burst_search_hi = min(fwd_limit, closure_end + 0.05)
    burst_t = crossing(times, rms, closure_end, burst_search_hi, silence_thresh * 3.0, rising=True)
    return closure_start, closure_end, burst_t


def build_events_for_phrase(pid, text, pipeline, aligner):
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

    # Ordered per-phone CTC spans (seconds), for neighbor-clamping.
    ordered = []
    for k, (char, orig_idx, _tok) in enumerate(triples):
        if k >= len(result.spans):
            break
        nat = native_by_idx.get(orig_idx)
        if nat is None:
            continue
        span = result.spans[k]
        ordered.append({
            "char": char, "class": nat["class"],
            "native_start": nat["start"] / FS, "native_end": nat["end"] / FS,
            "ctc_start": span.start_sample / FS, "ctc_end": span.end_sample / FS,
            "ctc_confidence": span.confidence,
        })

    events = []
    for i, item in enumerate(ordered):
        cls = item["class"]
        ctc_t0, ctc_t1 = item["ctc_start"], item["ctc_end"]

        # Phrase-initial/final phones have no real neighbor to clamp
        # against. The original flat MAX_BACK_S/MAX_FWD_S cap turned out
        # to truncate genuinely long phrase-initial events (e.g. "she"
        # /S/'s true frication starts near t=0 and runs ~380ms, far past
        # a 150ms cap) -- extend to the true utterance boundary instead
        # for these edge phones, not an arbitrary flat cap.
        is_phrase_initial = i == 0
        is_phrase_final = i + 1 >= len(ordered)
        prev_end = ordered[i - 1]["ctc_end"] if i > 0 else 0.0
        next_start = ordered[i + 1]["ctc_start"] if i + 1 < len(ordered) else len(audio) / FS
        back_limit = prev_end if is_phrase_initial else max(prev_end, ctc_t0 - MAX_BACK_S)
        fwd_limit = next_start if is_phrase_final else min(next_start, ctc_t1 + MAX_FWD_S)

        ev = SynthesisEvent(
            phone=item["char"], phone_class=cls,
            native_start_ms=item["native_start"] * 1000, native_end_ms=item["native_end"] * 1000,
            ctc_start_ms=ctc_t0 * 1000, ctc_end_ms=ctc_t1 * 1000,
            ctc_confidence=item["ctc_confidence"],
            event_type="none", event_confidence=0.0,
        )
        if is_phrase_initial:
            ev.warnings.append("phrase-initial: search extends to utterance start, not a flat cap")
        if is_phrase_final:
            ev.warnings.append("phrase-final: search extends to utterance end, not a flat cap")

        if cls == "stop":
            closure_t, closure_end_t, burst_t = detect_stop_closure_and_burst(times, rms, back_limit, fwd_limit)
            if burst_t is None:
                # No qualifying closure found in-window -- fall back to
                # the old global-flux-max as a last resort, but flag it
                # explicitly as the less-trustworthy path (known to
                # sometimes pick a louder unrelated region instead of a
                # real but acoustically weak burst).
                burst_t = argextreme(times, flux, back_limit, fwd_limit, "max")
                if burst_t is not None:
                    ev.warnings.append("no closure found -- burst from unconstrained flux-max fallback, lower trust")
            voicing_t = crossing(times, zcr, burst_t if burst_t else ctc_t0, fwd_limit, 0.15, rising=False) \
                if burst_t is not None else None
            ev.event_type = "stop"
            ev.closure_start_ms = closure_t * 1000 if closure_t else None
            ev.burst_peak_ms = burst_t * 1000 if burst_t else None
            ev.voicing_onset_ms = voicing_t * 1000 if voicing_t else None
            ev.acoustic_event_start_ms = ev.closure_start_ms
            ev.acoustic_event_end_ms = ev.voicing_onset_ms
            ev.preservation_start_ms = ev.burst_peak_ms
            ev.preservation_end_ms = (burst_t + 0.03) * 1000 if burst_t else None
            ev.event_confidence = 1.0 if closure_t is not None and burst_t is not None else (0.4 if burst_t is not None else 0.0)
            if burst_t is None:
                ev.warnings.append("no burst found in clamped window")
            elif not (ctc_t0 - 0.01 <= burst_t <= ctc_t1 + 0.15):
                ev.warnings.append("burst far from CTC span even after clamping")

        elif cls in ("fricative", "affricate"):
            noise_onset = crossing(times, hf, back_limit, fwd_limit, 0.4, rising=True)
            core_s, core_e = stable_run(times, hf, back_limit, fwd_limit, 0.4, min_dur_s=0.02)
            noise_offset = crossing(times, hf, core_e if core_e else ctc_t0, fwd_limit, 0.4, rising=False) if core_e else None
            burst_t = None
            if cls == "affricate":
                burst_t = argextreme(times, flux, back_limit, core_s if core_s else ctc_t1, "max")
            ev.event_type = cls
            ev.noise_onset_ms = noise_onset * 1000 if noise_onset else None
            ev.stable_core_start_ms = core_s * 1000 if core_s else None
            ev.stable_core_end_ms = core_e * 1000 if core_e else None
            ev.noise_offset_ms = noise_offset * 1000 if noise_offset else None
            ev.closure_start_ms = None
            ev.burst_peak_ms = burst_t * 1000 if burst_t else None
            ev.acoustic_event_start_ms = ev.noise_onset_ms
            ev.acoustic_event_end_ms = ev.noise_offset_ms
            ev.preservation_start_ms = ev.stable_core_start_ms
            ev.preservation_end_ms = ev.stable_core_end_ms
            ev.event_confidence = 1.0 if core_s is not None else 0.0
            if core_s is None:
                ev.warnings.append("no stable frication core found in clamped window")

        else:
            onset = crossing(times, zcr, back_limit, min(fwd_limit, ctc_t1), 0.15, rising=False)
            ev.event_type = "periodic"
            ev.acoustic_event_start_ms = onset * 1000 if onset else None
            ev.event_confidence = 0.5 if onset is not None else 0.0

        events.append(ev)

    return events


def main():
    pipeline = KPipeline(lang_code="a")
    aligner = CtcPhoneAligner()

    all_events = {}
    for pid, text in PHRASES.items():
        events = build_events_for_phrase(pid, text, pipeline, aligner)
        all_events[pid] = [asdict(e) for e in events]
        print(f"{pid}: {len(events)} events")

    # Spot-check: did clamping fix the duplicate-timestamp bug found in
    # phrase_final_stops (the /t/,/d/ pair that previously shared
    # identical closure/burst timestamps)?
    pfs = all_events["phrase_final_stops"]
    stops = [e for e in pfs if e["event_type"] == "stop"]
    print("\n--- phrase_final_stops stops (duplicate-timestamp check) ---")
    for e in stops:
        print(f"  {e['phone']}: closure={e['closure_start_ms']} burst={e['burst_peak_ms']} voicing={e['voicing_onset_ms']} warnings={e['warnings']}")
    bursts = [e["burst_peak_ms"] for e in stops if e["burst_peak_ms"] is not None]
    n_dupes = len(bursts) - len(set(bursts))
    print(f"duplicate burst timestamps among stops in this phrase: {n_dupes}")

    # Recompute the headline aggregate metrics with clamped windows
    all_flat = [e for pid in all_events for e in all_events[pid]]
    stops_all = [e for e in all_flat if e["event_type"] == "stop"]
    fric_all = [e for e in all_flat if e["event_type"] == "fricative"]
    n_burst_found = sum(1 for e in stops_all if e["burst_peak_ms"] is not None)
    n_burst_far_warn = sum(1 for e in stops_all if "burst far from CTC span even after clamping" in e["warnings"])
    print(f"\nstops: n={len(stops_all)} burst_found={n_burst_found} still_far_after_clamping={n_burst_far_warn}")

    core_covs = []
    for e in fric_all:
        if e["stable_core_start_ms"] is None:
            continue
        cs, ce = e["stable_core_start_ms"] / 1000, e["stable_core_end_ms"] / 1000
        ctc0, ctc1 = e["ctc_start_ms"] / 1000, e["ctc_end_ms"] / 1000
        inter = max(0.0, min(ce, ctc1) - max(cs, ctc0))
        dur = ce - cs
        if dur > 0:
            core_covs.append(inter / dur)
    if core_covs:
        arr = np.array(core_covs)
        print(f"fricatives: n={len(fric_all)} core_found={len(core_covs)} core_coverage mean={arr.mean():.3f} std={arr.std():.3f}")

    out_path = BASE / "synthesis_event_records.json"
    out_path.write_text(json.dumps(all_events, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

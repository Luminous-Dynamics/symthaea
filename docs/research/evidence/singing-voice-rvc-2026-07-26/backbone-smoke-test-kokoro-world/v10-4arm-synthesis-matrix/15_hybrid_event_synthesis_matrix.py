#!/usr/bin/env python3
"""4-arm synthesis matrix, per the reviewer's plan:

  A  existing (proportional-duration) heuristic, mask-only     [v6, unmodified]
  B  hybrid event alignment (synthesis-event-record), mask-only
  C  hybrid event alignment, raw-transient/consonant preservation
  D  hybrid event alignment, high-frequency residual preservation

Arm A is v6 (`03v6_LOCKED_control.py`), run standalone and unmodified --
already the frozen control throughout this arc, not reimplemented here.

Arms B/C/D share one word-loop renderer, structurally identical to v8's
`synthesize_word` for everything EXCEPT how a voiceless-obstruent
phoneme's timing/extraction span is determined: instead of the
proportional `sub_durations` model, each phoneme is looked up in the
phrase-level `events_by_idx` table (built from CtcPhoneAligner +
the class-specific acoustic-event detectors from
`14_synthesis_event_record.py`, including its two 2026-07-28 fixes --
utterance-boundary-aware search and closure-anchored burst detection)
keyed by its position in Kokoro's own `ps` phoneme string. When a valid
event exists, its `preservation_start/end` (real acoustic boundaries)
replace the proportional estimate; when it doesn't (rare -- e.g. an
unmapped/untransduced phone), falls back to the v8 proportional method,
explicitly flagged.

  B: mask-only -- F0 forced to 0 over the event's span (not the
     proportional span), still WORLD-synthesized.
  C: raw waveform extracted directly from the event's preservation span
     (same mechanism as v7/v8, corrected boundaries).
  D: NEW mechanism -- band-split residual. The event's span is rendered
     via BOTH paths (WORLD synthesis low-band + raw source high-band,
     summed), not a block-alternating group choice. This is the "Arm D"
     proposed and repeatedly deferred through v7-v9/v8.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from kokoro import KPipeline

from phone_aligner import CtcPhoneAligner
from misaki_to_espeak import transduce

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
FS = 24000
FRAME_PERIOD_MS = 5.0
FRAME_DT = FRAME_PERIOD_MS / 1000.0
GAP_S = 0.06
FADE_MS = 8.0
CROSSFADE_MS = 10.0
MIN_SYLLABLE_DUR_S = 0.28
STRETCH = 1.2
CONSONANT_NATURAL_MS = 60.0
VOWEL_FLOOR_S = 0.08
GLIDE_MS = 40.0
VIBRATO_RATE_HZ = 5.5
VIBRATO_DEPTH_CENTS = 30.0
VIBRATO_MIN_VOWEL_MS = 150.0
NOTES_CYCLE = [261.63, 293.66, 329.63, 392.00, 440.00]

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
VOICELESS_OBSTRUENT_CHARS = set("ptkfθsʃhʧ")
VOICED_OBSTRUENT_CHARS = set("bdgvðzʒʤ")
STRESS_MARKS = "ˈˌ"


def strip_stress(ps):
    return "".join(c for c in ps if c not in STRESS_MARKS)


def classify(ps):
    out = []
    for c in strip_stress(ps):
        is_vowel = c in VOWEL_CHARS
        if is_vowel or c in SONORANT_CONSONANT_CHARS:
            vclass = "sonorant"
        elif c in VOICELESS_OBSTRUENT_CHARS:
            vclass = "voiceless_obstruent"
        elif c in VOICED_OBSTRUENT_CHARS:
            vclass = "voiced_obstruent"
        else:
            vclass = "voiced_obstruent"
        out.append((c, is_vowel, vclass))
    return out


def syllabify(phonemes):
    if not phonemes:
        return []
    vowel_idx = [i for i, (c, _iv, _vc) in enumerate(phonemes) if c in VOWEL_CHARS]
    if not vowel_idx:
        return [phonemes]
    syllables, start = [], 0
    for k, vi in enumerate(vowel_idx):
        end = vi + 1 if k + 1 < len(vowel_idx) else len(phonemes)
        syllables.append(phonemes[start:end])
        start = end
    return syllables


def sub_durations(phonemes, total_dur_s, consonant_ms):
    n = len(phonemes)
    if n == 0:
        return []
    n_vowels = sum(1 for _, is_v, _vc in phonemes if is_v)
    if n_vowels == 0:
        per = total_dur_s / n
        return [per] * n
    n_consonants = n - n_vowels
    max_c_share = 0.6 * total_dur_s / max(1, n_consonants) if n_consonants else 0.0
    c_dur = min(consonant_ms / 1000.0, max_c_share) if n_consonants else 0.0
    vowel_total = max(VOWEL_FLOOR_S * 0.5, total_dur_s - n_consonants * c_dur)
    v_dur = vowel_total / n_vowels
    return [v_dur if is_v else c_dur for _, is_v, _vc in phonemes]


def resample_frames(arr, n_out):
    t_in = arr.shape[0]
    if t_in == 0:
        return np.zeros((n_out,) + arr.shape[1:], dtype=arr.dtype)
    if t_in == 1:
        return np.repeat(arr, n_out, axis=0)
    x_old = np.linspace(0.0, 1.0, t_in)
    x_new = np.linspace(0.0, 1.0, n_out)
    out = np.empty((n_out,) + arr.shape[1:], dtype=arr.dtype)
    for idx in np.ndindex(arr.shape[1:]):
        out[(slice(None),) + idx] = np.interp(x_new, x_old, arr[(slice(None),) + idx])
    return out


def resample_waveform(samples, n_out):
    if len(samples) == 0:
        return np.zeros(n_out)
    if len(samples) == n_out:
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, samples)


def bandsplit(x, fs, cutoff_hz):
    n = len(x)
    if n < 8:
        return x.copy(), np.zeros_like(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    low_mask = (freqs < cutoff_hz).astype(np.float64)
    low = np.fft.irfft(spec * low_mask, n)
    high = x - low
    return low, high


def blend_linear(a_tail, b_head, n):
    fade_out = np.linspace(1.0, 0.0, n)
    fade_in = np.linspace(0.0, 1.0, n)
    return a_tail * fade_out + b_head * fade_in


def crossfade_concat(chunks, fs, crossfade_ms):
    """Simplified fixed-linear-crossfade concatenation (Arm A's exit
    policy, "current" -- this matrix isn't re-testing exit policy, that
    was v8's own question) -- returns just the waveform."""
    n_default = max(1, int(round(fs * crossfade_ms / 1000.0)))
    if not chunks:
        return np.zeros(1)
    out = chunks[0].copy()
    for nxt in chunks[1:]:
        n = min(len(out), len(nxt), n_default)
        if n <= 0:
            out = np.concatenate([out, nxt])
            continue
        blended = blend_linear(out[-n:], nxt[:n], n)
        out = np.concatenate([out[:-n], blended, nxt[n:]])
    return out


def fade_edges(y, fs, fade_ms):
    n = max(1, int(round(fs * fade_ms / 1000.0)))
    n = min(n, len(y) // 2) if len(y) > 1 else 0
    if n <= 0:
        return y
    y = y.copy()
    y[:n] *= np.linspace(0.0, 1.0, n)
    y[-n:] *= np.linspace(1.0, 0.0, n)
    return y


# --- event-record detectors, copied from 14_synthesis_event_record.py
# (kept import-free/self-contained so this script has no fragile
# cross-script coupling) ---

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
    ci0, ci1 = runs[-1]
    closure_start = float(times[ci0])
    closure_end = float(times[ci1 - 1])
    burst_search_hi = min(fwd_limit, closure_end + 0.05)
    burst_t = crossing(times, rms, closure_end, burst_search_hi, silence_thresh * 3.0, rising=True)
    return closure_start, closure_end, burst_t


MAX_BACK_S = 0.15
MAX_FWD_S = 0.15


def build_events_by_idx(ps, pred_dur, audio, aligner):
    """Returns dict: orig_ps_index -> {class, preservation_start_s,
    preservation_end_s, event_confidence, source}. Only voiceless-
    obstruent-class phones get a real event (stops/fricatives/
    affricates) -- other classes aren't needed for the arms this matrix
    tests and are omitted."""
    cum = [0]
    for d in pred_dur:
        cum.append(cum[-1] + d)
    FRAME_SAMPLES = 600
    native_by_idx = {}
    for i, ch in enumerate(ps):
        cls_tuple = classify(ch)[0] if ch not in (" ",) + tuple(STRESS_MARKS) else None
        if ch == " " or ch in STRESS_MARKS:
            continue
        _c, _iv, vclass = classify(ch)[0]
        pdi = i + 1
        native_by_idx[i] = {
            "start": cum[pdi] * FRAME_SAMPLES / FS, "end": cum[pdi + 1] * FRAME_SAMPLES / FS,
            "vclass": vclass,
        }

    result = aligner.align(audio, FS, ps)
    triples, _unknown = transduce(ps)
    times, rms, hf, zcr, flat, flux = analysis_frames(audio, FS)

    ordered = []
    for k, (char, orig_idx, _tok) in enumerate(triples):
        if k >= len(result.spans):
            break
        nat = native_by_idx.get(orig_idx)
        if nat is None:
            continue
        span = result.spans[k]
        ordered.append({
            "orig_idx": orig_idx, "char": char, "vclass": nat["vclass"],
            "ctc_start": span.start_sample / FS, "ctc_end": span.end_sample / FS,
        })

    STOP_LIKE = set("ptkbdɡ")
    FRICATIVE_LIKE = set("fθsʃh")
    AFFRICATE_LIKE = set("ʧʤ")

    # Found via direct trace inspection on consonant_clusters (v10's
    # severe Arm C/D regression there): two real problems neighbor-
    # clamping the SEARCH WINDOW does not prevent. (1) A phrase-initial
    # phone's search window legitimately reaches back to the true
    # utterance start (per the phrase-boundary fix), but for "strong"'s
    # /s/ this produced a 266ms preservation span -- squeezing 266ms of
    # real source audio into a ~30-60ms target slot via resample_waveform
    # would badly time-distort it. (2) Neighbor-clamping bounds the
    # SEARCH window against a neighbor's CTC span, but the RESULTING
    # preservation spans are derived independently and were found to
    # still overlap each other in time across a word boundary (splashed's
    # final /t/ [1.4375,1.4516] vs strangely's initial /s/
    # [1.4325,1.5175] -- /s/'s span starts BEFORE /t/'s own span ends).
    # Extracting raw audio for both would duplicate overlapping source
    # content when concatenated. Both fixed below: a max-duration cap
    # (clipped from the start, since every observed case had the START
    # erroneously too early, never the end) and a sequential clip against
    # the previously-placed event's own preservation_end.
    MAX_PRESERVATION_DUR_S = 0.12
    prev_preservation_end = None

    events = {}
    for i, item in enumerate(ordered):
        ch = item["char"]
        ctc_t0, ctc_t1 = item["ctc_start"], item["ctc_end"]
        is_first = i == 0
        is_last = i + 1 >= len(ordered)
        prev_end = ordered[i - 1]["ctc_end"] if i > 0 else 0.0
        next_start = ordered[i + 1]["ctc_start"] if i + 1 < len(ordered) else len(audio) / FS
        back_limit = prev_end if is_first else max(prev_end, ctc_t0 - MAX_BACK_S)
        fwd_limit = next_start if is_last else min(next_start, ctc_t1 + MAX_FWD_S)

        pres_s, pres_e, conf = None, None, 0.0
        if ch in STOP_LIKE:
            closure_t, closure_end_t, burst_t = detect_stop_closure_and_burst(times, rms, back_limit, fwd_limit)
            if burst_t is None:
                burst_t = argextreme(times, flux, back_limit, fwd_limit, "max")
                conf = 0.4 if burst_t is not None else 0.0
            else:
                conf = 1.0
            if burst_t is not None:
                pres_s, pres_e = burst_t, min(fwd_limit, burst_t + 0.03)
        elif ch in FRICATIVE_LIKE or ch in AFFRICATE_LIKE:
            core_s, core_e = stable_run(times, hf, back_limit, fwd_limit, 0.4, min_dur_s=0.02)
            if core_s is not None:
                pres_s, pres_e = core_s, core_e
                conf = 1.0
        else:
            continue  # not a class this matrix's arms treat specially

        if pres_s is not None and pres_e is not None and pres_e > pres_s:
            # Fix 1: cap max duration, clipping from the start.
            if pres_e - pres_s > MAX_PRESERVATION_DUR_S:
                pres_s = pres_e - MAX_PRESERVATION_DUR_S
            # Fix 2: never start before the previous event's own end.
            if prev_preservation_end is not None and pres_s < prev_preservation_end:
                pres_s = prev_preservation_end
            if pres_e <= pres_s:
                continue  # clipping consumed the whole span -- skip, fall back to proportional
            prev_preservation_end = pres_e
            events[item["orig_idx"]] = {
                "vclass_detail": "stop" if ch in STOP_LIKE else ("affricate" if ch in AFFRICATE_LIKE else "fricative"),
                "preservation_start_s": pres_s, "preservation_end_s": pres_e,
                "confidence": conf,
            }
    return events


def synthesize_word(f0_seg, sp_seg, ap_seg, f0_full, sp_full, ap_full, x_word, x_full, syll_specs,
                     word_ps_offset, word_index_map, events_by_idx, word_start_sample_full, arm, fs):
    entries = []
    cursor_in_frame = 0
    cursor_in_sample = 0
    local_phoneme_i = 0
    for spec in syll_specs:
        phonemes = spec["phonemes"]
        natural_subdurs = sub_durations(phonemes, spec["natural_dur_s"], CONSONANT_NATURAL_MS)
        target_subdurs = sub_durations(phonemes, spec["target_dur_s"], CONSONANT_NATURAL_MS)
        for (ch, is_v, vclass), nat_d, tgt_d in zip(phonemes, natural_subdurs, target_subdurs):
            full_idx = word_ps_offset + word_index_map[local_phoneme_i]
            local_phoneme_i += 1
            event = events_by_idx.get(full_idx)

            n_in = max(1, round(nat_d / FRAME_DT))
            end_in_frame = min(cursor_in_frame + n_in, len(f0_seg))
            proportional_src_start = cursor_in_sample
            proportional_src_end = min(
                cursor_in_sample + int(round((end_in_frame - cursor_in_frame) * FRAME_DT * fs)),
                len(x_word),
            )

            used_event = False
            mask_frac = (0.0, 1.0)  # fraction of this entry's OWN output span
            # that Arm B masks to F0=0 -- default is "the whole entry"
            # (matching Arm A's uniform per-class masking rule) unless a
            # real event narrows it to the acoustically-verified span.
            if event is not None and vclass == "voiceless_obstruent":
                # Extract from the FULL PHRASE audio at absolute sample
                # coordinates, not word-relative -- found (via
                # events_used dropping to near-zero on fricative_heavy)
                # that MMS_FA's word boundary is frequently narrower than
                # the true event span (e.g. "she" 's /S/ event starts at
                # 5ms, but MMS_FA places the word "she" starting at
                # 322ms), silently rejecting almost every event if
                # extraction were constrained to the word's own slice.
                ev_start_abs = int(round(event["preservation_start_s"] * fs))
                ev_end_abs = int(round(event["preservation_end_s"] * fs))
                if 0 <= ev_start_abs < ev_end_abs <= len(x_full):
                    used_event = True
                    ev_start_wordrel = ev_start_abs - word_start_sample_full
                    ev_end_wordrel = ev_end_abs - word_start_sample_full
                    win = max(1, proportional_src_end - proportional_src_start)
                    f_s = (ev_start_wordrel - proportional_src_start) / win
                    f_e = (ev_end_wordrel - proportional_src_start) / win
                    mask_frac = (max(0.0, min(1.0, f_s)), max(0.0, min(1.0, f_e)))
            if not used_event:
                src_start_sample, src_end_sample = proportional_src_start, proportional_src_end

            cursor_in_sample = proportional_src_end

            if used_event:
                # Found (Arm D's consonant_clusters WER unchanged by the
                # raw-extraction fix, unlike Arm C's) that the WORLD-
                # domain spectral envelope (sp/ap/f0) was STILL being
                # sliced from the OLD, uncorrected proportional window
                # even after the raw waveform extraction was fixed --
                # Arm D's low-band (WORLD) and high-band (raw) components
                # could therefore be built from mismatched time windows.
                # Slice the event's own frame range from the FULL-PHRASE
                # arrays instead, mirroring the raw-extraction fix.
                ev_start_frame_abs = max(0, int(round(event["preservation_start_s"] / FRAME_DT)))
                ev_end_frame_abs = min(len(f0_full), int(round(event["preservation_end_s"] / FRAME_DT)))
                if ev_end_frame_abs > ev_start_frame_abs:
                    seg_f0 = f0_full[ev_start_frame_abs:ev_end_frame_abs]
                    seg_sp = sp_full[ev_start_frame_abs:ev_end_frame_abs]
                    seg_ap = ap_full[ev_start_frame_abs:ev_end_frame_abs]
                else:
                    seg_f0 = f0_seg[cursor_in_frame:end_in_frame]
                    seg_sp = sp_seg[cursor_in_frame:end_in_frame]
                    seg_ap = ap_seg[cursor_in_frame:end_in_frame]
            else:
                seg_f0 = f0_seg[cursor_in_frame:end_in_frame]
                seg_sp = sp_seg[cursor_in_frame:end_in_frame]
                seg_ap = ap_seg[cursor_in_frame:end_in_frame]
            cursor_in_frame = end_in_frame

            if used_event:
                raw_samples = x_full[ev_start_abs:ev_end_abs]
            else:
                raw_samples = x_word[max(0, src_start_sample):max(0, src_end_sample)]

            n_out = max(2, round(tgt_d / FRAME_DT))
            n_out_samples = max(1, round(tgt_d * fs))
            entries.append({
                "ch": ch, "is_vowel": is_v, "vclass": vclass,
                "f0": resample_frames(seg_f0, n_out),
                "sp": resample_frames(seg_sp, n_out),
                "ap": resample_frames(seg_ap, n_out),
                "raw": resample_waveform(raw_samples, n_out_samples),
                "target_hz": spec["target_hz"], "n_out": n_out,
                "n_out_samples": n_out_samples,
                "used_event": used_event,
                "mask_frac": mask_frac,
            })

    n_syll = len(syll_specs)
    syll_frame_ranges = []
    cursor_out = 0
    idx = 0
    for spec in syll_specs:
        n_phonemes_this_syll = len(spec["phonemes"])
        vowel_out_start, vowel_out_end = None, None
        syll_start = cursor_out
        for _ in range(n_phonemes_this_syll):
            e = entries[idx]
            if e["ch"] in VOWEL_CHARS:
                if vowel_out_start is None:
                    vowel_out_start = cursor_out
                vowel_out_end = cursor_out + e["n_out"]
            cursor_out += e["n_out"]
            idx += 1
        syll_frame_ranges.append((syll_start, cursor_out, vowel_out_start, vowel_out_end))

    word_f0_original = np.concatenate([e["f0"] for e in entries]).astype(np.float64)
    target_traj = np.zeros(len(word_f0_original))
    for i, (s, e, vs, ve) in enumerate(syll_frame_ranges):
        target_traj[s:e] = syll_specs[i]["target_hz"]

    glide_frames = max(1, round(GLIDE_MS / 1000.0 / FRAME_DT))
    for i in range(n_syll - 1):
        s0, e0, _, _ = syll_frame_ranges[i]
        s1, e1, _, _ = syll_frame_ranges[i + 1]
        hz0 = syll_specs[i]["target_hz"]
        hz1 = syll_specs[i + 1]["target_hz"]
        n = min(glide_frames, e0 - s0, e1 - s1)
        if n <= 1:
            continue
        half = max(1, n // 2)
        ramp_out = np.linspace(hz0, hz1, 2 * half)
        target_traj[e0 - half:e0] = ramp_out[:half]
        target_traj[s1:s1 + half] = ramp_out[half:]

    t_abs = np.arange(len(word_f0_original)) * FRAME_DT
    vibrato_mult = np.ones(len(word_f0_original))
    for (s, e, vs, ve) in syll_frame_ranges:
        if vs is None or ve is None:
            continue
        vowel_dur_ms = (ve - vs) * FRAME_PERIOD_MS
        if vowel_dur_ms < VIBRATO_MIN_VOWEL_MS:
            continue
        inner_s = vs + round(0.4 * (ve - vs))
        inner_e = vs + round(0.9 * (ve - vs))
        if inner_e <= inner_s:
            continue
        phase = 2 * np.pi * VIBRATO_RATE_HZ * t_abs[inner_s:inner_e]
        cents = VIBRATO_DEPTH_CENTS * np.sin(phase)
        vibrato_mult[inner_s:inner_e] = 2.0 ** (cents / 1200.0)

    cursor = 0
    for e in entries:
        n = e["n_out"]
        seg_target = target_traj[cursor:cursor + n]
        seg_vibrato = vibrato_mult[cursor:cursor + n]
        seg_orig_f0 = e["f0"]
        if e["vclass"] == "sonorant":
            eligible = seg_orig_f0 > 0
            e["final_f0"] = np.where(eligible, seg_target * seg_vibrato, 0.0)
        elif e["vclass"] == "voiceless_obstruent":
            if arm == "B" and e["used_event"]:
                # Event-informed partial mask: zero only the fraction of
                # this entry's own span the real acoustic event actually
                # covers, letting natural (possibly voiced) F0 through
                # elsewhere -- e.g. if the proportional window bled into
                # neighboring vowel material (the exact v9 failure mode),
                # that portion is no longer force-silenced.
                f_s, f_e = e["mask_frac"]
                i_s, i_e = int(round(f_s * n)), int(round(f_e * n))
                mask = np.zeros(n, dtype=bool)
                mask[i_s:i_e] = True
                e["final_f0"] = np.where(mask, 0.0, seg_orig_f0)
            else:
                e["final_f0"] = np.zeros(n)
        else:
            e["final_f0"] = seg_orig_f0
        cursor += n

    # Rendering-method grouping, arm-dependent.
    groups = []
    for e in entries:
        if e["vclass"] == "voiceless_obstruent":
            method = "raw" if arm == "C" else ("residual" if arm == "D" else "world")
        else:
            method = "world"
        if groups and groups[-1][0] == method:
            groups[-1][1].append(e)
        else:
            groups.append((method, [e]))

    rendered_chunks = []
    for method, group_entries in groups:
        if method == "world":
            g_f0 = np.concatenate([e["final_f0"] for e in group_entries]).astype(np.float64)
            g_sp = np.concatenate([e["sp"] for e in group_entries], axis=0).astype(np.float64)
            g_ap = np.concatenate([e["ap"] for e in group_entries], axis=0).astype(np.float64)
            y = pw.synthesize(g_f0, g_sp, g_ap, fs, frame_period=FRAME_PERIOD_MS)
        elif method == "raw":
            y = np.concatenate([e["raw"] for e in group_entries])
        else:  # residual (Arm D): WORLD low-band + raw high-band, summed
            g_f0 = np.concatenate([e["final_f0"] for e in group_entries]).astype(np.float64)
            g_sp = np.concatenate([e["sp"] for e in group_entries], axis=0).astype(np.float64)
            g_ap = np.concatenate([e["ap"] for e in group_entries], axis=0).astype(np.float64)
            y_world = pw.synthesize(g_f0, g_sp, g_ap, fs, frame_period=FRAME_PERIOD_MS)
            y_raw = np.concatenate([e["raw"] for e in group_entries])
            n = min(len(y_world), len(y_raw))
            world_low, _world_high = bandsplit(y_world[:n], fs, 3000.0)
            _raw_low, raw_high = bandsplit(y_raw[:n], fs, 3000.0)
            y = world_low + raw_high
        rendered_chunks.append(y)

    y_word = crossfade_concat(rendered_chunks, fs, CROSSFADE_MS)
    n_events_used = sum(1 for e in entries if e.get("used_event"))
    return y_word, n_events_used


def phoneme_index_map(word_ps):
    """`classify()` strips stress marks before producing its phoneme
    list, so the N-th entry of that list does NOT correspond to the
    N-th character of the original word string whenever a stress mark
    precedes it. Returns, for each post-strip phoneme in order, its true
    index within `word_ps` -- e.g. for "sˈɛlz" (stress mark at index 1),
    returns [0, 2, 3, 4] (skipping index 1). A naive `char_cursor += 1`
    per phoneme was found (via a direct debug probe on fricative_heavy,
    which returned events_used=0 despite build_events_by_idx finding 7
    real events) to silently misalign every phoneme after the first
    stress mark in a word -- this is the fix."""
    return [i for i, c in enumerate(word_ps) if c not in STRESS_MARKS]


def build_config_with_syllable_melody(phrases, pipeline):
    out = []
    note_idx = 0
    for phrase in phrases:
        r = list(pipeline(phrase["text"], voice="af_heart"))[0]
        ps_full = r.phonemes
        ps_words = ps_full.split(" ")
        word_syllables = [syllabify(classify(w)) for w in ps_words]
        n_syllables_total = sum(len(s) for s in word_syllables)
        melody = [NOTES_CYCLE[(note_idx + i) % len(NOTES_CYCLE)] for i in range(n_syllables_total)]
        note_idx += n_syllables_total
        word_offsets = []
        cursor = 0
        for w in ps_words:
            word_offsets.append(cursor)
            cursor += len(w) + 1
        word_index_maps = [phoneme_index_map(w) for w in ps_words]
        out.append({**phrase, "word_syllables": word_syllables, "syllable_melody_hz": melody,
                    "ps_full": ps_full, "pred_dur": r.pred_dur.tolist(), "word_offsets": word_offsets,
                    "word_index_maps": word_index_maps})
    return out


def main(config_path, audio_dir, align_dir, out_suffix, arm):
    config = json.loads((BASE / config_path).read_text())
    pipeline = KPipeline(lang_code="a")
    aligner = CtcPhoneAligner()
    phrases = build_config_with_syllable_melody(config["phrases"], pipeline)

    for phrase in phrases:
        wav_path = BASE / audio_dir / f"{phrase['id']}_spoken.wav"
        align_path = BASE / align_dir / f"{phrase['id']}.json"
        words = json.loads(align_path.read_text())
        if len(words) != len(phrase["word_syllables"]):
            print(f"{phrase['id']}: SKIPPED -- word-count mismatch")
            continue

        x, fs = sf.read(str(wav_path))
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        spoken_rms = float(np.sqrt(np.mean(x**2)))

        f0, t = pw.harvest(x, fs, frame_period=FRAME_PERIOD_MS)
        sp = pw.cheaptrick(x, f0, t, fs)
        ap = pw.d4c(x, f0, t, fs)

        events_by_idx = build_events_by_idx(phrase["ps_full"], phrase["pred_dur"], x.astype(np.float32), aligner)

        melody_cursor = 0
        word_waveforms = []
        gap_samples_n = int(round(GAP_S * fs))
        total_events_used = 0
        for word_idx, (word, sylls) in enumerate(zip(words, phrase["word_syllables"])):
            n_syll = len(sylls)
            syll_specs = []
            start_frame = int(round(word["start"] / FRAME_DT))
            cursor_frame = max(0, min(start_frame, len(f0) - 1))
            end_frame_word = min(int(round(word["end"] / FRAME_DT)), len(f0))
            remaining_frames = max(1, end_frame_word - cursor_frame)
            per_syll_natural_frames = max(1, remaining_frames // n_syll)
            for i, syll_phonemes in enumerate(sylls):
                target_hz = phrase["syllable_melody_hz"][melody_cursor]
                melody_cursor += 1
                n_frames = per_syll_natural_frames if i < n_syll - 1 else (
                    end_frame_word - cursor_frame
                )
                natural_dur_s = max(1, n_frames) * FRAME_DT
                target_dur_s = max(MIN_SYLLABLE_DUR_S, natural_dur_s * STRETCH)
                syll_specs.append({
                    "phonemes": syll_phonemes, "natural_dur_s": natural_dur_s,
                    "target_dur_s": target_dur_s, "target_hz": target_hz,
                })
                cursor_frame += n_frames

            f0_seg = f0[start_frame:end_frame_word]
            sp_seg = sp[start_frame:end_frame_word]
            ap_seg = ap[start_frame:end_frame_word]
            start_sample = int(round(start_frame * FRAME_DT * fs))
            end_sample = int(round(end_frame_word * FRAME_DT * fs))
            x_word = x[start_sample:end_sample]
            word_ps_offset = phrase["word_offsets"][word_idx]
            word_index_map = phrase["word_index_maps"][word_idx]
            y_word, n_used = synthesize_word(
                f0_seg, sp_seg, ap_seg, f0, sp, ap, x_word, x, syll_specs, word_ps_offset, word_index_map,
                events_by_idx, start_sample, arm, fs,
            )
            total_events_used += n_used
            y_word = fade_edges(y_word, fs, FADE_MS)
            word_waveforms.append(y_word)

        gap_samples = np.zeros(gap_samples_n)
        pieces = []
        for i, y_word in enumerate(word_waveforms):
            pieces.append(y_word)
            if i < len(word_waveforms) - 1:
                pieces.append(gap_samples)
        y = np.concatenate(pieces) if pieces else np.zeros(1)

        sung_rms = float(np.sqrt(np.mean(y**2))) + 1e-9
        y = y * (spoken_rms / sung_rms)
        peak = np.abs(y).max()
        if peak > 0.98:
            y = y * (0.98 / peak)

        out_path = BASE / audio_dir / f"{phrase['id']}_{out_suffix}.wav"
        sf.write(str(out_path), y, fs)
        print(f"{phrase['id']}: wrote {out_path} ({len(y)/fs:.2f}s), events_used={total_events_used}")

    print(f"\nArm {arm} ({out_suffix}) done.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "gate2_config.json"
    audio_dir = sys.argv[2] if len(sys.argv) > 2 else "gate2_audio"
    align_dir = sys.argv[3] if len(sys.argv) > 3 else "gate2_alignments"
    out_suffix = sys.argv[4] if len(sys.argv) > 4 else "sung_v10_b"
    arm = sys.argv[5] if len(sys.argv) > 5 else "B"
    main(config_path, audio_dir, align_dir, out_suffix, arm)

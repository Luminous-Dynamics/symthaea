#!/usr/bin/env python3
"""v9 (2026-07-28): phoneme-count-weighted syllable duration split --
root-cause fix for the fricative_heavy "T-shirts"/"T-shirt" regression
found during v8's exit-crossfade ablation.

CONFIRMED ROOT CAUSE (via direct inspection of the spoken source audio's
high-frequency-energy profile, not inference): the word-level natural
duration (from real forced alignment) was being split EQUALLY BY
SYLLABLE COUNT (`per_syll_natural_frames = remaining_frames // n_syll`
in the original `main()`), ignoring each syllable's actual phoneme
content. For "seashells" (/si/ 2 phonemes vs /S held over more phonemes
in `shells` = /SElz/, 4 phonemes) and "seashore" (/si/ vs /SO extended
with r/, similar imbalance), the equal split placed the "sea"/"shells"
(or "sea"/"shore") syllable boundary well INTO the true /sh/ frication
region instead of before it -- confirmed by scanning the spoken audio's
high-band (>=3kHz) energy fraction around the modeled boundary: for
"seashells" the true frication had already dropped from hf_frac~0.7-0.8
to ~0.07 (vowel-like) by the time the modeled raw-/sh/-extraction span
even STARTS (sample 23400), meaning the "raw consonant" slice actually
captured vowel onset, not frication -- while the true frication got
absorbed into the PRECEDING WORLD-synthesized "i" (sea's vowel) group's
source window, contaminating its periodic resynthesis with noise it
was never meant to carry. Same direction (less severe) for "seashore".

FIX: weight each syllable's share of the word's natural frame budget by
its phoneme count (vowels weighted higher than consonants, matching
`sub_durations`' own vowel-gets-more-time assumption) instead of a flat
equal split. This is the ONE changed variable versus v8 Arm A -- exit
policy, gestures, F0 rules, voicing classification are all unchanged.

Original v7/v7b/v8 docstring below, unmodified:
---
v7b (2026-07-28): v7 (waveform-preserved obstruents) + exact frame-
lineage instrumentation, per the reviewer's explicit request after v7's
honest mixed result: "the renderer already knows which source interval
generated each output interval; that mapping should become a first-class
artifact." Every rendered group (a run of consecutive same-method
phonemes) now emits its exact OUTPUT sample range, its CORE-interior
range (excluding the crossfade region shared with neighboring groups --
computed exactly by crossfade_concat_with_lineage, not inferred), its
entry/exit crossfade ranges, and (for raw/waveform-preserved groups) the
exact SOURCE sample range it was extracted from. This is ground truth
from the renderer's own bookkeeping, not an approximate external
mapping. Written out as a `<phrase>_<suffix>_lineage.json` sidecar next
to each rendered WAV.

Original v7 docstring, unchanged below (rendering behavior itself is
IDENTICAL to v7 -- only the lineage export is new):

v7: the reviewer's proposed 5th arm -- preserve the
ORIGINAL Kokoro consonant waveform directly for voiceless obstruents,
crossfaded into the WORLD-rendered sonorant/voiced-obstruent material,
instead of resynthesizing every phoneme through WORLD parameters.

Rationale (reviewer): "F0=0 is necessary but not sufficient" -- WORLD's
resynthesized spectral envelope/aperiodicity for a forced-unvoiced
obstruent can still sound smoothed/muffled relative to the source's own
turbulence and transient structure. Raw-waveform preservation for short
unvoiced consonants retains natural noise character that a parametric
model may be flattening even with the correct voicing decision.

On top of the locked v6 control (03v6_LOCKED_control.py, unmodified):

  - sonorant and voiced-obstruent phonemes: rendered through WORLD
    parameters exactly as in v6 (unchanged).
  - voiceless-obstruent phonemes: the ORIGINAL Kokoro waveform for that
    phoneme's natural time span is extracted directly and used as-is
    (time-domain resampling only if the target/natural duration ratio
    is far from 1.0 -- v3's duration rule already keeps obstruent
    target durations close to natural, so this is the uncommon case).
  - Consecutive same-rendering-method phonemes are batched into one
    chunk (one pw.synthesize call for a WORLD run, one waveform slice
    for a raw run); chunks are concatenated with a short (10ms) linear
    crossfade at every boundary to avoid a hard edge.

Duration allocation, syllabification, gestures, and the sonorant/
voiced-obstruent F0 rules are otherwise IDENTICAL to v6 -- only the
RENDERING METHOD for voiceless-obstruent spans changes.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from kokoro import KPipeline

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")

FRAME_PERIOD_MS = 5.0
FRAME_DT = FRAME_PERIOD_MS / 1000.0
GAP_S = 0.06
FADE_MS = 8.0
CROSSFADE_MS = 10.0  # entry-into-consonant width, and A/C's exit width (unchanged from v7b)
EXIT_POLICY = "A_current"  # set via CLI arg; one of EXIT_POLICIES' keys
MIN_SYLLABLE_DUR_S = 0.28
STRETCH = 1.2
CONSONANT_NATURAL_MS = 60.0
VOWEL_FLOOR_S = 0.08
GLIDE_MS = 40.0
VIBRATO_RATE_HZ = 5.5
VIBRATO_DEPTH_CENTS = 30.0
VIBRATO_MIN_VOWEL_MS = 150.0

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
VOICELESS_OBSTRUENT_CHARS = set("ptkfθsʃhʧ")
VOICED_OBSTRUENT_CHARS = set("bdgvðzʒʤ")
STRESS_MARKS = "ˈˌ"
NOTES_CYCLE = [261.63, 293.66, 329.63, 392.00, 440.00]


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
    """Simple time-domain linear-interpolation resample, used only for
    the uncommon case where a voiceless-obstruent's target duration
    differs substantially from its natural one. Noise/turbulence content
    tolerates this far better than tonal content would (no clear pitch
    to distort), but it is NOT distortion-free -- disclosed, not hidden."""
    if len(samples) == 0:
        return np.zeros(n_out)
    if len(samples) == n_out:
        return samples
    x_old = np.linspace(0.0, 1.0, len(samples))
    x_new = np.linspace(0.0, 1.0, n_out)
    return np.interp(x_new, x_old, samples)


STOP_CHARS = set("ptk")
FRICATIVE_CHARS = set("fθsʃh")
AFFRICATE_CHARS = set("ʧʤ")  # affricates -- treated like a stop (very short) below


def exit_class_for_group(phoneme_str):
    """Classify a raw group's EXIT boundary by its LAST phoneme (the one
    adjacent to the boundary) -- stops/affricates need a very short,
    burst-preserving transition; continuous fricatives can tolerate a
    slightly longer one without losing their defining noise character."""
    last = phoneme_str[-1] if phoneme_str else ""
    if last in STOP_CHARS or last in AFFRICATE_CHARS:
        return "stop"
    return "fricative"


def bandsplit(x, fs, cutoff_hz):
    """Simple FFT-domain brick-wall split -- not phase-linear, but
    adequate for a bounded first test of the multiband CONCEPT."""
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


def blend_equal_power(a_tail, b_head, n):
    t = np.linspace(0.0, 1.0, n)
    fade_out = np.cos(t * np.pi / 2)
    fade_in = np.sin(t * np.pi / 2)
    return a_tail * fade_out + b_head * fade_in


def blend_multiband(a_tail, b_head, n, fs, cutoff_hz=3000.0, high_frac=0.25):
    """Low band (<cutoff_hz) gets the full n-sample linear fade (matching
    the original policy); high band holds the OUTGOING (consonant) side's
    content for most of the window, fading to the incoming (vowel) side
    only in the final high_frac*n samples -- per the reviewer's
    "retain consonant noise until near the boundary" proposal."""
    a_low, a_high = bandsplit(a_tail, fs, cutoff_hz)
    b_low, b_high = bandsplit(b_head, fs, cutoff_hz)
    low_blend = blend_linear(a_low, b_low, n)
    high_n = max(1, int(round(n * high_frac)))
    high_blend = a_high.copy()
    if high_n < n:
        high_blend[-high_n:] = blend_linear(a_high[-high_n:], b_high[-high_n:], high_n)
    else:
        high_blend = blend_linear(a_high, b_high, n)
    return low_blend + high_blend


def policy_A_current(out, nxt, default_n, fs, cls):
    n = min(len(out), len(nxt), default_n)
    return blend_linear(out[-n:], nxt[:n], n), n


def policy_B_short_equal_power(out, nxt, default_n, fs, cls):
    ms = 1.5 if cls == "stop" else 5.0
    n = max(1, int(round(fs * ms / 1000.0)))
    n = min(len(out), len(nxt), n)
    return blend_equal_power(out[-n:], nxt[:n], n), n


def policy_C_multiband(out, nxt, default_n, fs, cls):
    n = min(len(out), len(nxt), default_n)
    return blend_multiband(out[-n:], nxt[:n], n, fs), n


EXIT_POLICIES = {
    "A_current": policy_A_current,
    "B_short_equal_power": policy_B_short_equal_power,
    "C_multiband": policy_C_multiband,
}


def crossfade_concat_with_lineage(chunks, fs, entry_crossfade_ms, exit_policy_fn):
    """Concatenate chunks, applying the ORIGINAL fixed-linear policy at
    every boundary EXCEPT boundaries where the group being LEFT is a raw
    (voiceless-obstruent) group -- those use `exit_policy_fn(out, nxt,
    default_n, fs, exit_class)` instead, per the reviewer's "change
    nothing else" instruction (entry-into-a-consonant stays exactly as
    before; only the exit-from-a-consonant boundary is under test). The
    policy function picks its OWN blend width `n` (not a slice of a
    pre-fixed window -- a genuinely shorter transition needs its own
    window, not a truncation of a longer one) and returns (overlap, n).
    Returns (waveform, bounds, boundary_ns) -- boundary_ns[i] is the
    ACTUAL crossfade width used between chunk i and chunk i+1 (varies
    per-policy, e.g. shorter for policy B), needed downstream to compute
    correct core/entry/exit lineage ranges when policies use different
    widths. `chunks` is a list of (waveform, method, exit_class) tuples."""
    n_fade_default = max(1, int(fs * entry_crossfade_ms / 1000.0))
    if not chunks:
        return np.zeros(1), []
    waves = [c[0] for c in chunks]
    methods = [c[1] for c in chunks]
    exit_classes = [c[2] for c in chunks]
    out = waves[0].copy()
    bounds = [{"start": 0, "end": len(waves[0])}]
    boundary_ns = []  # crossfade width actually used at each boundary (len = len(chunks)-1)
    for i in range(1, len(waves)):
        nxt = waves[i]
        leaving_method = methods[i - 1]
        leaving_exit_class = exit_classes[i - 1]
        if leaving_method == "raw":
            overlap, n = exit_policy_fn(out, nxt, n_fade_default, fs, leaving_exit_class)
        else:
            n = min(len(out), len(nxt), n_fade_default)
            if n <= 1:
                new_start = len(out)
                out = np.concatenate([out, nxt])
                bounds.append({"start": new_start, "end": new_start + len(nxt)})
                boundary_ns.append(0)
                continue
            overlap = blend_linear(out[-n:], nxt[:n], n)
        prev_len = len(out)
        out = np.concatenate([out[:-n], overlap, nxt[n:]])
        new_start = prev_len - n
        bounds.append({"start": new_start, "end": new_start + len(nxt)})
        boundary_ns.append(n)
    return out, bounds, boundary_ns


def fade_edges(y, fs, fade_ms):
    n = max(1, int(fs * fade_ms / 1000.0))
    n = min(n, len(y) // 2) if len(y) > 1 else 0
    if n <= 0:
        return y
    ramp = np.linspace(0.0, 1.0, n)
    y = y.copy()
    y[:n] *= ramp
    y[-n:] *= ramp[::-1]
    return y


def synthesize_word(f0_seg, sp_seg, ap_seg, x_word, syll_specs, fs):
    """Returns the final word waveform, built from alternating
    WORLD-synthesized runs and raw-waveform-preserved runs."""
    # First pass: compute per-phoneme resampled WORLD frames AND the
    # original-waveform sample range (in x_word) for every phoneme,
    # same bookkeeping as v6 plus the raw sample range.
    entries = []  # dicts: vclass, f0/sp/ap (resampled), raw_samples (resampled to target), is_vowel_flag(for gestures)
    cursor_in_frame = 0
    cursor_in_sample = 0
    for spec in syll_specs:
        phonemes = spec["phonemes"]
        natural_subdurs = sub_durations(phonemes, spec["natural_dur_s"], CONSONANT_NATURAL_MS)
        target_subdurs = sub_durations(phonemes, spec["target_dur_s"], CONSONANT_NATURAL_MS)
        for (ch, is_v, vclass), nat_d, tgt_d in zip(phonemes, natural_subdurs, target_subdurs):
            n_in = max(1, round(nat_d / FRAME_DT))
            end_in_frame = min(cursor_in_frame + n_in, len(f0_seg))
            seg_f0 = f0_seg[cursor_in_frame:end_in_frame]
            seg_sp = sp_seg[cursor_in_frame:end_in_frame]
            seg_ap = ap_seg[cursor_in_frame:end_in_frame]

            n_in_samples = int(round((end_in_frame - cursor_in_frame) * FRAME_DT * fs))
            end_in_sample = min(cursor_in_sample + n_in_samples, len(x_word))
            src_start_sample = cursor_in_sample
            raw_samples = x_word[cursor_in_sample:end_in_sample]

            cursor_in_frame = end_in_frame
            cursor_in_sample = end_in_sample

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
                "src_start_sample": src_start_sample, "src_end_sample": end_in_sample,
            })

    # Second pass (WORLD-domain, matching v6 exactly): build the
    # syllable-boundary target trajectory, glides, and vibrato over the
    # WORLD-eligible frames only.
    n_syll = len(syll_specs)
    # Re-derive syllable frame ranges over ALL entries (needed for glide/
    # vibrato placement, matching v6's bookkeeping).
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

    # Assign final F0 per entry (frame-indexed into the flat arrays above).
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
            e["final_f0"] = np.zeros(n)  # only relevant if this entry falls back to WORLD synthesis
        else:  # voiced_obstruent
            e["final_f0"] = seg_orig_f0
        cursor += n

    # Third pass: group consecutive entries by rendering method and
    # render each group (WORLD synthesis for sonorant/voiced-obstruent
    # runs, raw-waveform for voiceless-obstruent runs), then crossfade
    # all groups together.
    groups = []  # list of ("world"|"raw", [entries])
    for e in entries:
        method = "raw" if e["vclass"] == "voiceless_obstruent" else "world"
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
            exit_cls = None
        else:
            y = np.concatenate([e["raw"] for e in group_entries])
            exit_cls = exit_class_for_group("".join(e["ch"] for e in group_entries))
        rendered_chunks.append((y, method, exit_cls))

    exit_policy_fn = EXIT_POLICIES[EXIT_POLICY]
    y_word, bounds, boundary_ns = crossfade_concat_with_lineage(
        rendered_chunks, fs, CROSSFADE_MS, exit_policy_fn
    )

    # Build lineage records: one per group, in WORD-relative output sample
    # coordinates. "core" excludes the crossfade region shared with the
    # PREVIOUS group (entry) and the NEXT group (exit) -- ground truth
    # from crossfade_concat_with_lineage's exact bookkeeping, not inferred.
    # entry/exit widths now use the ACTUAL per-boundary n (boundary_ns),
    # since different exit policies can use different widths.
    lineage = []
    for i, (method, group_entries) in enumerate(groups):
        b = bounds[i]
        has_entry_fade = i > 0
        has_exit_fade = i < len(groups) - 1
        entry_n = boundary_ns[i - 1] if has_entry_fade else 0
        exit_n = boundary_ns[i] if has_exit_fade else 0
        core_start = b["start"] + entry_n
        core_end = b["end"] - exit_n
        core_start = min(core_start, b["end"])
        core_end = max(core_end, b["start"])
        record = {
            "method": method,
            "phonemes": "".join(e["ch"] for e in group_entries),
            "vclass": group_entries[0]["vclass"],
            "output_start": int(b["start"]), "output_end": int(b["end"]),
            "core_start": int(core_start), "core_end": int(core_end),
            "entry_fade": [int(b["start"]), int(b["start"] + entry_n)] if has_entry_fade else None,
            "exit_fade": [int(b["end"] - exit_n), int(b["end"])] if has_exit_fade else None,
        }
        if method == "raw":
            record["source_start_sample"] = int(group_entries[0]["src_start_sample"])
            record["source_end_sample"] = int(group_entries[-1]["src_end_sample"])
        lineage.append(record)

    return y_word, lineage


def build_config_with_syllable_melody(phrases, pipeline):
    out = []
    note_idx = 0
    for phrase in phrases:
        ps_full = " ".join(ps for _gs, ps, _audio in pipeline(phrase["text"], voice="af_heart"))
        ps_words = ps_full.split()
        word_syllables = [syllabify(classify(w)) for w in ps_words]
        n_syllables_total = sum(len(s) for s in word_syllables)
        melody = [NOTES_CYCLE[(note_idx + i) % len(NOTES_CYCLE)] for i in range(n_syllables_total)]
        note_idx += n_syllables_total
        out.append({**phrase, "word_syllables": word_syllables, "syllable_melody_hz": melody})
    return out


def main(config_path, audio_dir, align_dir, out_suffix):
    config = json.loads((BASE / config_path).read_text())
    pipeline = KPipeline(lang_code="a")
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

        melody_cursor = 0
        word_waveforms = []
        phrase_lineage = []
        cumulative_output_sample = 0
        gap_samples_n = int(round(GAP_S * fs))
        for word_idx, (word, sylls) in enumerate(zip(words, phrase["word_syllables"])):
            n_syll = len(sylls)
            syll_specs = []
            start_frame = int(round(word["start"] / FRAME_DT))
            cursor_frame = max(0, min(start_frame, len(f0) - 1))
            end_frame_word = min(int(round(word["end"] / FRAME_DT)), len(f0))
            remaining_frames = max(1, end_frame_word - cursor_frame)
            # v9 fix: weight each syllable's share of the word's natural
            # frame budget by phoneme content (vowel=1.5, consonant=1.0 --
            # a round, disclosed choice, not fit to any specific example)
            # instead of an equal per-syllable-count split. Root cause of
            # the fricative_heavy regression: an equal split placed the
            # "sea"/"shells" boundary well into the true /sh/ frication
            # region for syllables with very different phoneme counts.
            syll_weights = []
            for syll_phonemes in sylls:
                w = sum(1.5 if is_v else 1.0 for _c, is_v, _vc in syll_phonemes)
                syll_weights.append(max(w, 0.5))
            weight_total = sum(syll_weights)
            per_syll_natural_frames = [
                max(1, int(round(remaining_frames * w / weight_total)))
                for w in syll_weights
            ]
            for i, syll_phonemes in enumerate(sylls):
                target_hz = phrase["syllable_melody_hz"][melody_cursor]
                melody_cursor += 1
                n_frames = per_syll_natural_frames[i] if i < n_syll - 1 else (
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
            y_word, word_lineage = synthesize_word(f0_seg, sp_seg, ap_seg, x_word, syll_specs, fs)
            y_word = fade_edges(y_word, fs, FADE_MS)
            word_waveforms.append(y_word)

            # Re-anchor each group's lineage into phrase-level output
            # sample coordinates, and raw groups' source range into the
            # FULL source file's sample coordinates (start_sample is this
            # word's own offset into x).
            for rec in word_lineage:
                rec["word_index"] = word_idx
                rec["word_text"] = word["word"]
                rec["output_start"] += cumulative_output_sample
                rec["output_end"] += cumulative_output_sample
                rec["core_start"] += cumulative_output_sample
                rec["core_end"] += cumulative_output_sample
                if rec["entry_fade"] is not None:
                    rec["entry_fade"] = [v + cumulative_output_sample for v in rec["entry_fade"]]
                if rec["exit_fade"] is not None:
                    rec["exit_fade"] = [v + cumulative_output_sample for v in rec["exit_fade"]]
                if rec["method"] == "raw":
                    rec["source_start_sample"] += start_sample
                    rec["source_end_sample"] += start_sample
                phrase_lineage.append(rec)

            cumulative_output_sample += len(y_word)
            if word_idx < len(words) - 1:
                cumulative_output_sample += gap_samples_n

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
        lineage_path = BASE / audio_dir / f"{phrase['id']}_{out_suffix}_lineage.json"
        lineage_path.write_text(json.dumps({
            "sample_rate": fs,
            "crossfade_ms": CROSSFADE_MS,
            "word_edge_fade_ms": FADE_MS,
            "groups": phrase_lineage,
        }, indent=2))
        print(f"{phrase['id']}: wrote {out_path} ({len(y)/fs:.2f}s), lineage -> {lineage_path}")

    print(f"\nv9 phoneme-weighted syllable split ({EXIT_POLICY}) done.")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    audio_dir = sys.argv[2] if len(sys.argv) > 2 else "audio"
    align_dir = sys.argv[3] if len(sys.argv) > 3 else "alignments"
    out_suffix = sys.argv[4] if len(sys.argv) > 4 else "sung_v9"
    EXIT_POLICY = sys.argv[5] if len(sys.argv) > 5 else "A_current"
    main(config_path, audio_dir, align_dir, out_suffix)

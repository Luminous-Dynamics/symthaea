#!/usr/bin/env python3
"""Stage 3 (v2, 2026-07-28): WORLD-vocoder reshaping, revised after an
external review's direct signal-analysis findings on v1's output (all
independently confirmed before fixing):

  - Severe boundary clicks (up to ~63% full-scale sample jumps at word/gap
    boundaries): root-caused to feeding f0=0 "gap" frames through the same
    pw.synthesize() call as real word content. In WORLD, f0=0 means
    UNVOICED (noise-excited synthesis using whatever spectral envelope is
    given), not silence -- v1's gap frames were repeating the previous
    word's spectral envelope, producing a spurious noise burst discontinuous
    with real neighboring content. Fixed: each word is now synthesized in
    complete ISOLATION (its own pw.synthesize call), concatenated with
    genuine time-domain silence (actual zero samples, not a WORLD unvoiced
    frame) for gaps, with a short fade at each word's edges.
  - 10.5-13.1 dB loudness increase vs. the spoken source (root cause:
    v1 blind-peak-normalized every output to 0.92 regardless of source
    loudness). Fixed: RMS-match the output to the spoken source's RMS,
    with a peak safety cap.
  - Word-level (not phoneme-aware) retiming, flagged as the largest
    structural limitation: a whole multisyllabic word was stretched as one
    block, rather than following singing_bridge.rs's established rule
    ("consonants stay brief, vowels absorb the stretch"). Fixed: each
    word's misaki phoneme string (Kokoro's OWN G2P output -- checked
    directly per the review's suggestion; Kokoro does not expose
    per-phoneme *durations*, only the symbol sequence, so sub-word timing
    remains an estimate, not true forced sub-alignment -- disclosed, not
    hidden) is classified into vowels/consonants; consonants keep a fixed
    nominal duration in both the natural and target timeline, vowels
    absorb 100% of the difference. Applied within each word's own
    real-aligned span (MMS_FA still supplies the one ground-truth
    timing fact used: the word's total start/end).
"""
import json
import re
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from kokoro import KPipeline

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
config = json.loads((BASE / "gate2_config.json").read_text())

FRAME_PERIOD_MS = 5.0
GAP_S = 0.06  # genuine silence between words now, not a synthesized frame
FADE_MS = 8.0  # short edge fade to avoid a raw-edge click at silence boundaries
MIN_WORD_DUR_S = 0.35
STRETCH = 1.2
CONSONANT_NATURAL_MS = 60.0
VOWEL_FLOOR_S = 0.08

# Misaki/Kokoro US phoneme inventory: vowel + diphthong symbols (checked
# directly against this project's own test phrases' actual G2P output --
# not assumed from external misaki docs). Stress marks (ˈˌ) are stripped
# before classification, not treated as phonemes.
VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
STRESS_MARKS = "ˈˌ"


def strip_stress(ps):
    return "".join(c for c in ps if c not in STRESS_MARKS)


def classify(ps):
    """Return a list of (char, is_vowel) for one word's phoneme string."""
    return [(c, c in VOWEL_CHARS) for c in strip_stress(ps)]


def phoneme_sub_durations(phonemes, total_dur_s, consonant_ms):
    """Split total_dur_s across phonemes: consonants get a fixed share
    (consonant_ms, clamped so they can't consume the whole budget for a
    pathologically short word), vowels evenly split whatever remains.
    Falls back to one even split if there are no vowels at all (a
    degenerate all-consonant word, matching singing_bridge.rs's fallback).
    """
    n = len(phonemes)
    if n == 0:
        return []
    n_vowels = sum(1 for _, v in phonemes if v)
    if n_vowels == 0:
        per = total_dur_s / n
        return [per] * n
    n_consonants = n - n_vowels
    max_consonant_share = 0.6 * total_dur_s / max(1, n_consonants) if n_consonants else 0.0
    c_dur = min(consonant_ms / 1000.0, max_consonant_share) if n_consonants else 0.0
    vowel_total = max(VOWEL_FLOOR_S * 0.5, total_dur_s - n_consonants * c_dur)
    v_dur = vowel_total / n_vowels
    return [v_dur if is_v else c_dur for _, is_v in phonemes]


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


pipeline = KPipeline(lang_code="a")

for phrase in config["phrases"]:
    wav_path = BASE / "gate2_audio" / f"{phrase['id']}_spoken.wav"
    align_path = BASE / "gate2_alignments" / f"{phrase['id']}.json"
    words = json.loads(align_path.read_text())
    melody = phrase["melody_hz"]
    if len(words) != len(melody):
        print(f"{phrase['id']}: SKIPPED -- {len(words)} aligned words != {len(melody)} melody "
              f"notes (alignment failure or word-count mismatch, reported not silently dropped)")
        continue

    # Kokoro's own G2P output for this exact phrase, split by word --
    # verified word-count parity (spaces == word boundaries) on every
    # test phrase in this experiment before relying on it.
    word_texts = [w["word"] for w in words]
    ps_full = " ".join(ps for _gs, ps, _audio in pipeline(phrase["text"], voice="af_heart"))
    ps_words = ps_full.split()
    if len(ps_words) != len(words):
        print(f"{phrase['id']}: SKIPPED -- {len(ps_words)} G2P word-tokens != {len(words)} "
              f"aligned words ({ps_words!r} vs {word_texts!r})")
        continue

    x, fs = sf.read(str(wav_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    spoken_rms = float(np.sqrt(np.mean(x**2)))

    f0, t = pw.harvest(x, fs, frame_period=FRAME_PERIOD_MS)
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    frame_dt = FRAME_PERIOD_MS / 1000.0

    word_waveforms = []
    for word, target_hz, ps_word in zip(words, melody, ps_words):
        natural_dur = word["end"] - word["start"]
        target_dur = max(MIN_WORD_DUR_S, natural_dur * STRETCH)
        phonemes = classify(ps_word)

        natural_subdurs = phoneme_sub_durations(phonemes, natural_dur, CONSONANT_NATURAL_MS)
        target_subdurs = phoneme_sub_durations(phonemes, target_dur, CONSONANT_NATURAL_MS)

        start_frame = int(round(word["start"] / frame_dt))
        cursor_frame = max(0, min(start_frame, len(f0) - 1))

        f0_chunks, sp_chunks, ap_chunks = [], [], []
        for (nat_d, tgt_d) in zip(natural_subdurs, target_subdurs):
            n_frames_in = max(1, round(nat_d / frame_dt))
            end_frame = min(cursor_frame + n_frames_in, len(f0))
            seg_f0 = f0[cursor_frame:end_frame]
            seg_sp = sp[cursor_frame:end_frame]
            seg_ap = ap[cursor_frame:end_frame]
            cursor_frame = end_frame

            n_out = max(2, round(tgt_d / frame_dt))
            r_f0 = resample_frames(seg_f0, n_out)
            r_sp = resample_frames(seg_sp, n_out)
            r_ap = resample_frames(seg_ap, n_out)
            voiced = r_f0 > 0
            r_f0 = np.where(voiced, target_hz, 0.0)

            f0_chunks.append(r_f0)
            sp_chunks.append(r_sp)
            ap_chunks.append(r_ap)

        word_f0 = np.concatenate(f0_chunks).astype(np.float64)
        word_sp = np.concatenate(sp_chunks, axis=0).astype(np.float64)
        word_ap = np.concatenate(ap_chunks, axis=0).astype(np.float64)

        y_word = pw.synthesize(word_f0, word_sp, word_ap, fs, frame_period=FRAME_PERIOD_MS)
        y_word = fade_edges(y_word, fs, FADE_MS)
        word_waveforms.append(y_word)

    gap_samples = np.zeros(int(round(GAP_S * fs)))
    pieces = []
    for i, y_word in enumerate(word_waveforms):
        pieces.append(y_word)
        if i < len(word_waveforms) - 1:
            pieces.append(gap_samples)
    y = np.concatenate(pieces)

    # RMS-match to the spoken source (not a blind fixed peak target), with
    # a safety peak cap so RMS-matching can't push a low-source-loudness
    # phrase's peaks over unity.
    sung_rms = float(np.sqrt(np.mean(y**2))) + 1e-9
    y = y * (spoken_rms / sung_rms)
    peak = np.abs(y).max()
    if peak > 0.98:
        y = y * (0.98 / peak)

    out_path = BASE / "gate2_audio" / f"{phrase['id']}_sung.wav"
    sf.write(str(out_path), y, fs)
    print(f"{phrase['id']}: wrote {out_path} ({len(y)/fs:.2f}s), "
          f"rms_match_target={spoken_rms:.4f} final_rms={np.sqrt(np.mean(y**2)):.4f}")

print("\nStage 3 (v2) done.")

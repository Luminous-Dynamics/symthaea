#!/usr/bin/env python3
"""v4 (2026-07-28): preserve obstruent consonants as natural/unvoiced,
only sonorants (vowels/nasals/liquids/glides) carry the note pitch.

Per the reviewer's diagnosis of v3's over-voicing (confirmed by direct
measurement: voiced-frame fraction roughly doubled vs. spoken source,
spectral centroid down 20-40%, ZCR down 57-76% -- consistent with
periodic/harmonic energy replacing noisy consonant/fricative material).
Root cause identified: v3's is_vowel-only classification repitched EVERY
originally-voiced frame to the flat target, including naturally-voiced
consonants (nasals, liquids, glides, voiced obstruents like /z d v/) --
and its duration split used the same fixed nominal consonant-ms in both
the natural and target computations independently, which could drift
apart under the SHARE-based cap rather than staying pinned near the
consonant's own real duration.

Two changes on top of the locked v3 control
(03v3_LOCKED_control.py, NOT modified):

  1. Three-way phoneme classification, not binary vowel/consonant:
     SONORANT (vowel + nasal /m n ng/ + liquid /l r/ + glide /w j/) is
     pitch-bearing and absorbs stretch, matching the reviewer's own
     "vowels; nasals; liquids; glides; other voiced sonorants" list.
     OBSTRUENT (stops, fricatives, affricates -- voiced or unvoiced) is
     forced UNVOICED (target_f0=0) regardless of the source's own
     voicing, per the reviewer's literal pseudocode: "if source_f0==0
     or phoneme_is_unvoiced: target_f0=0". This deliberately turns a
     naturally-voiced obstruent like /z/ or /d/ into aperiodic synthesis
     to remove the harmonic "buzz" WORLD's periodic excitation was
     adding to consonant regions.
  2. Duration: obstruents are clamped to their OWN estimated natural
     duration (not the syllable-wide STRETCH factor) -- sonorants absorb
     100% of the extra target-duration budget, per "consonant onset ->
     preserve near source duration ... vowel nucleus -> absorb most
     stretching."

NOT attempted this pass (deliberately, one bounded increment at a time
per the reviewer's own ablation-not-one-big-change instruction):
context-dependent glide timing (repeated-note/step/leap), vowel-onset-
to-beat alignment, deterministic vibrato jitter, or aligning onset with
the vowel rather than the syllable start.
"""
import json
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
MIN_SYLLABLE_DUR_S = 0.28
STRETCH = 1.2
OBSTRUENT_NATURAL_MS = 60.0
SONORANT_FLOOR_S = 0.08
GLIDE_MS = 40.0
VIBRATO_RATE_HZ = 5.5
VIBRATO_DEPTH_CENTS = 30.0
VIBRATO_MIN_VOWEL_MS = 150.0

VOWEL_CHARS = set("əɜʌIᵻæiɔɪɐuʊɑɛeAOWYɚɝᵊ")
# Nasals, liquids, glides -- pitch-bearing sonorant consonants, per the
# reviewer's explicit list ("nasals; liquids; glides; other voiced
# sonorants"). Checked against this project's own test phrases' actual
# misaki output (m,n,ŋ,l,ɹ,w,j all appear; not assumed from external docs).
SONORANT_CONSONANT_CHARS = set("mnŋlɹrwj")
STRESS_MARKS = "ˈˌ"

NOTES_CYCLE = [261.63, 293.66, 329.63, 392.00, 440.00]


def strip_stress(ps):
    return "".join(c for c in ps if c not in STRESS_MARKS)


def classify(ps):
    """Returns list of (char, phoneme_class) where phoneme_class is
    'sonorant' (vowel or nasal/liquid/glide consonant, pitch-bearing) or
    'obstruent' (stop/fricative/affricate, never pitch-bearing)."""
    out = []
    for c in strip_stress(ps):
        if c in VOWEL_CHARS or c in SONORANT_CONSONANT_CHARS:
            out.append((c, "sonorant"))
        else:
            out.append((c, "obstruent"))
    return out


def syllabify(phonemes):
    """Same maximal-onset rule as v3, now syllabifying around VOWEL
    nuclei specifically (nasals/liquids/glides can be sonorant but are
    never the syllable nucleus in English)."""
    if not phonemes:
        return []
    vowel_idx = [i for i, (c, _cls) in enumerate(phonemes) if c in VOWEL_CHARS]
    if not vowel_idx:
        return [phonemes]
    syllables, start = [], 0
    for k, vi in enumerate(vowel_idx):
        end = vi + 1 if k + 1 < len(vowel_idx) else len(phonemes)
        syllables.append(phonemes[start:end])
        start = end
    return syllables


def sub_durations(phonemes, natural_dur_s, target_dur_s, obstruent_ms):
    """Obstruents are clamped near their OWN estimated natural duration
    in the TARGET timeline too (not scaled by the syllable's overall
    stretch) -- sonorants absorb the entire natural/target difference.
    Falls back to an even split scaled by the same ratio if there are no
    sonorants at all (a degenerate all-obstruent syllable)."""
    n = len(phonemes)
    if n == 0:
        return [], []
    n_sonorant = sum(1 for _, cls in phonemes if cls == "sonorant")
    if n_sonorant == 0:
        ratio = target_dur_s / natural_dur_s if natural_dur_s > 0 else 1.0
        nat = [natural_dur_s / n] * n
        tgt = [d * ratio for d in nat]
        return nat, tgt
    n_obstruent = n - n_sonorant
    max_c_share = 0.6 * natural_dur_s / max(1, n_obstruent) if n_obstruent else 0.0
    c_nat = min(obstruent_ms / 1000.0, max_c_share) if n_obstruent else 0.0
    c_tgt = c_nat  # obstruent target duration pinned to its own natural estimate
    sonorant_nat_total = max(SONORANT_FLOOR_S * 0.5, natural_dur_s - n_obstruent * c_nat)
    sonorant_tgt_total = max(SONORANT_FLOOR_S, target_dur_s - n_obstruent * c_tgt)
    nat_each = sonorant_nat_total / n_sonorant
    tgt_each = sonorant_tgt_total / n_sonorant
    nat = [nat_each if cls == "sonorant" else c_nat for _, cls in phonemes]
    tgt = [tgt_each if cls == "sonorant" else c_tgt for _, cls in phonemes]
    return nat, tgt


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


def synthesize_word(f0_seg, sp_seg, ap_seg, syll_specs, fs):
    all_f0, all_sp, all_ap, all_sonorant = [], [], [], []
    syll_frame_ranges = []
    cursor_in = 0
    cursor_out = 0
    for spec in syll_specs:
        phonemes = spec["phonemes"]
        natural_subdurs, target_subdurs = sub_durations(
            phonemes, spec["natural_dur_s"], spec["target_dur_s"], OBSTRUENT_NATURAL_MS
        )
        vowel_out_start, vowel_out_end = None, None
        for (ch, cls), nat_d, tgt_d in zip(phonemes, natural_subdurs, target_subdurs):
            n_in = max(1, round(nat_d / FRAME_DT))
            end_in = min(cursor_in + n_in, len(f0_seg))
            seg_f0 = f0_seg[cursor_in:end_in]
            seg_sp = sp_seg[cursor_in:end_in]
            seg_ap = ap_seg[cursor_in:end_in]
            cursor_in = end_in

            n_out = max(2, round(tgt_d / FRAME_DT))
            r_f0 = resample_frames(seg_f0, n_out)
            r_sp = resample_frames(seg_sp, n_out)
            r_ap = resample_frames(seg_ap, n_out)
            all_f0.append(r_f0)
            all_sp.append(r_sp)
            all_ap.append(r_ap)
            all_sonorant.append(np.full(n_out, cls == "sonorant"))
            if ch in VOWEL_CHARS:
                if vowel_out_start is None:
                    vowel_out_start = cursor_out
                vowel_out_end = cursor_out + n_out
            cursor_out += n_out
        syll_start = syll_frame_ranges[-1][1] if syll_frame_ranges else 0
        syll_frame_ranges.append((syll_start, cursor_out, vowel_out_start, vowel_out_end))

    word_f0 = np.concatenate(all_f0).astype(np.float64)
    word_sp = np.concatenate(all_sp, axis=0).astype(np.float64)
    word_ap = np.concatenate(all_ap, axis=0).astype(np.float64)
    sonorant_mask = np.concatenate(all_sonorant)
    # Pitch-imposition eligibility per the reviewer's pseudocode: BOTH the
    # source frame must have been genuinely voiced AND the phoneme class
    # must be sonorant. An obstruent never gets a target pitch, even if
    # its source frame happened to read as voiced.
    eligible_mask = (word_f0 > 0) & sonorant_mask

    target_traj = np.zeros(len(word_f0))
    n_syll = len(syll_specs)
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

    t_abs = np.arange(len(word_f0)) * FRAME_DT
    vibrato_mult = np.ones(len(word_f0))
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

    final_f0 = np.where(eligible_mask, target_traj * vibrato_mult, 0.0)
    return final_f0, word_sp, word_ap


def build_config_with_syllable_melody(phrases, pipeline):
    out = []
    note_idx = 0
    for phrase in phrases:
        ps_full = " ".join(ps for _gs, ps, _audio in pipeline(phrase["text"], voice="af_heart"))
        ps_words = ps_full.split()
        word_syllables = []
        for ps_word in ps_words:
            phonemes = classify(ps_word)
            sylls = syllabify(phonemes)
            word_syllables.append(sylls)
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
        for word, sylls in zip(words, phrase["word_syllables"]):
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
                    "phonemes": syll_phonemes,
                    "natural_dur_s": natural_dur_s,
                    "target_dur_s": target_dur_s,
                    "target_hz": target_hz,
                })
                cursor_frame += n_frames

            f0_seg = f0[start_frame:end_frame_word]
            sp_seg = sp[start_frame:end_frame_word]
            ap_seg = ap[start_frame:end_frame_word]
            word_f0, word_sp, word_ap = synthesize_word(f0_seg, sp_seg, ap_seg, syll_specs, fs)
            y_word = pw.synthesize(word_f0, word_sp, word_ap, fs, frame_period=FRAME_PERIOD_MS)
            y_word = fade_edges(y_word, fs, FADE_MS)
            word_waveforms.append(y_word)

        gap_samples = np.zeros(int(round(GAP_S * fs)))
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
        print(f"{phrase['id']}: wrote {out_path} ({len(y)/fs:.2f}s)")

    print("\nv4 (obstruent-preserve) done.")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    audio_dir = sys.argv[2] if len(sys.argv) > 2 else "audio"
    align_dir = sys.argv[3] if len(sys.argv) > 3 else "alignments"
    out_suffix = sys.argv[4] if len(sys.argv) > 4 else "sung_v4"
    main(config_path, audio_dir, align_dir, out_suffix)

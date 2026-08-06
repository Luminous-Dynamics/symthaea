#!/usr/bin/env python3
"""v13: continuous-trajectory WORLD + one global Vocos pass.

Per the reviewer's plan (2026-07-29, after per-word Vocos was correctly
rejected for discarding Vocos's cross-word temporal context): the current
Arm B pipeline synthesizes each WORD independently via its own
`pw.synthesize()` call, fades its edges, then concatenates words with a
fixed 60ms silence gap (`GAP_S`) before a single global Vocos pass (v12).
That means WORLD itself resets state at every word boundary and an
artificial gap is inserted even where the phrase has no real pause --
plausible causes of the "segmented word islands" character every arm in
this arc has shown.

This version builds ONE continuous phrase-length (f0, sp, ap) trajectory
across every word (no per-word `pw.synthesize()`, no artificial silence
gap), overlap-adding each word's parameter arrays onto the previous word's
over a short crossfade window instead of concatenating with a gap, then
calls `pw.synthesize()` ONCE for the whole phrase -- reusing Arm B's exact
per-phoneme F0-assignment/vibrato/event-masking logic (imported, not
reimplemented) so the only architectural change under test is HOW words
are joined, not the note/duration/masking policy itself.

This stage produces WORLD-only output (`*_sung_v13_world_only.wav`) --
Vocos itself needs the nix-managed `voice-vocoder` devShell (not this
pip venv), so the resynthesis pass is a separate stage,
`23_v13_vocos_pass.py`, matching the same two-stage split already used
for v12.

Scope: 3 phrases only (positive_control, fricative_heavy,
long_sustained_vowels), analytics-gated, no listening pack -- per the
policy set this session: build small, check cheaply, only escalate to
listening if the boundary metrics improve without a WER regression.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyworld as pw
import soundfile as sf
from kokoro import KPipeline

sys.path.insert(0, str(Path(__file__).parent))
import importlib
m15 = importlib.import_module("15_hybrid_event_synthesis_matrix")

from phone_aligner import CtcPhoneAligner

BASE = Path("/var/lib/symthaea/training-runs/kokoro-world-vocoder")
FS = m15.FS
FRAME_DT = m15.FRAME_DT
FRAME_PERIOD_MS = m15.FRAME_PERIOD_MS
CONSONANT_NATURAL_MS = m15.CONSONANT_NATURAL_MS
MIN_SYLLABLE_DUR_S = m15.MIN_SYLLABLE_DUR_S
STRETCH = m15.STRETCH
GLIDE_MS = m15.GLIDE_MS
VIBRATO_RATE_HZ = m15.VIBRATO_RATE_HZ
VIBRATO_DEPTH_CENTS = m15.VIBRATO_DEPTH_CENTS
VIBRATO_MIN_VOWEL_MS = m15.VIBRATO_MIN_VOWEL_MS
VOWEL_CHARS = m15.VOWEL_CHARS

JOIN_CROSSFADE_MS = 30.0  # short context-preserving overlap at word boundaries
JOIN_CROSSFADE_FRAMES = max(1, round(JOIN_CROSSFADE_MS / 1000.0 / FRAME_DT))

PHRASES = ["positive_control", "fricative_heavy", "long_sustained_vowels"]

OUT_AUDIO_DIR = Path("/srv/luminous-dynamics/symthaea/audio_output/kokoro_world_vocoder_smoke_test_2026-07-28/v13_continuous_trajectory")
OUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def build_word_entries(f0_seg, sp_seg, ap_seg, f0_full, sp_full, ap_full, x_word, x_full, syll_specs,
                        word_ps_offset, word_index_map, events_by_idx):
    """Arm B's per-phoneme entry construction (target F0, event-informed
    voiceless-obstruent masking), lifted verbatim in spirit from
    `synthesize_word` but stopping BEFORE grouping/pw.synthesize() -- we
    need the per-phoneme f0/sp/ap entries for the whole phrase, not a
    per-word waveform."""
    entries = []
    cursor_in_frame = 0
    cursor_in_sample = 0
    local_phoneme_i = 0
    for spec in syll_specs:
        phonemes = spec["phonemes"]
        natural_subdurs = m15.sub_durations(phonemes, spec["natural_dur_s"], CONSONANT_NATURAL_MS)
        target_subdurs = m15.sub_durations(phonemes, spec["target_dur_s"], CONSONANT_NATURAL_MS)
        for (ch, is_v, vclass), nat_d, tgt_d in zip(phonemes, natural_subdurs, target_subdurs):
            full_idx = word_ps_offset + word_index_map[local_phoneme_i]
            local_phoneme_i += 1
            event = events_by_idx.get(full_idx)

            n_in = max(1, round(nat_d / FRAME_DT))
            end_in_frame = min(cursor_in_frame + n_in, len(f0_seg))
            proportional_src_start = cursor_in_sample
            proportional_src_end = min(
                cursor_in_sample + int(round((end_in_frame - cursor_in_frame) * FRAME_DT * FS)),
                len(x_word),
            )

            cursor_in_sample = proportional_src_end

            used_event = False
            ev_start_frame_abs = ev_end_frame_abs = None
            if event is not None and vclass == "voiceless_obstruent":
                ev_start_abs = int(round(event["preservation_start_s"] * FS))
                ev_end_abs = int(round(event["preservation_end_s"] * FS))
                if 0 <= ev_start_abs < ev_end_abs <= len(x_full):
                    used_event = True
                    ev_start_frame_abs = max(0, int(round(event["preservation_start_s"] / FRAME_DT)))
                    ev_end_frame_abs = min(len(f0_full), int(round(event["preservation_end_s"] / FRAME_DT)))

            if used_event and ev_end_frame_abs is not None and ev_end_frame_abs > ev_start_frame_abs:
                seg_f0 = f0_full[ev_start_frame_abs:ev_end_frame_abs]
                seg_sp = sp_full[ev_start_frame_abs:ev_end_frame_abs]
                seg_ap = ap_full[ev_start_frame_abs:ev_end_frame_abs]
            else:
                used_event = False
                seg_f0 = f0_seg[cursor_in_frame:end_in_frame]
                seg_sp = sp_seg[cursor_in_frame:end_in_frame]
                seg_ap = ap_seg[cursor_in_frame:end_in_frame]
            cursor_in_frame = end_in_frame

            n_out = max(2, round(tgt_d / FRAME_DT))
            entries.append({
                "ch": ch, "is_vowel": is_v, "vclass": vclass,
                "f0": m15.resample_frames(seg_f0, n_out),
                "sp": m15.resample_frames(seg_sp, n_out),
                "ap": m15.resample_frames(seg_ap, n_out),
                "n_out": n_out, "used_event": used_event,
            })
    return entries


def assign_final_f0(entries, syll_specs):
    """Same target-F0/vibrato/glide assignment as Arm B (voiceless
    obstruents fully masked to 0 -- Arm B's event-informed partial mask is
    skipped here for simplicity since none of the 3 target phrases'
    voiceless obstruents are the object of this experiment; the join
    mechanism is)."""
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
            e["final_f0"] = np.zeros(n)
        else:
            e["final_f0"] = seg_orig_f0
        cursor += n
    return entries


def join_continuous(word_param_lists, crossfade_frames):
    """Overlap-add successive words' (f0, sp, ap) frame arrays over
    `crossfade_frames`, instead of concatenating with a silence gap.
    Returns the joined (f0, sp, ap) plus the sample-domain boundary
    positions (for later discontinuity analytics)."""
    if not word_param_lists:
        return np.zeros(1), np.zeros((1, 1)), np.zeros((1, 1)), []

    f0_out, sp_out, ap_out = word_param_lists[0]
    f0_out = f0_out.copy()
    sp_out = sp_out.copy()
    ap_out = ap_out.copy()
    boundary_frames = []  # frame index (into the growing output) of each word join

    for f0_w, sp_w, ap_w in word_param_lists[1:]:
        n = min(crossfade_frames, len(f0_out), len(f0_w))
        if n <= 1:
            boundary_frames.append(len(f0_out))
            f0_out = np.concatenate([f0_out, f0_w])
            sp_out = np.concatenate([sp_out, sp_w], axis=0)
            ap_out = np.concatenate([ap_out, ap_w], axis=0)
            continue

        ramp = np.linspace(0.0, 1.0, n)
        # Overlap the tail of the accumulated output with the head of the
        # next word: linear crossfade in f0 (only where both sides are
        # voiced; unvoiced/voiced transitions just take the voiced side)
        # and in the (sp, ap) envelope/aperiodicity arrays directly.
        tail_f0 = f0_out[-n:]
        head_f0 = f0_w[:n]
        blended_f0 = np.where(
            (tail_f0 > 0) & (head_f0 > 0),
            tail_f0 * (1 - ramp) + head_f0 * ramp,
            np.where(head_f0 > 0, head_f0, tail_f0),
        )
        blended_sp = sp_out[-n:] * (1 - ramp)[:, None] + sp_w[:n] * ramp[:, None]
        blended_ap = ap_out[-n:] * (1 - ramp)[:, None] + ap_w[:n] * ramp[:, None]

        boundary_frames.append(len(f0_out) - n // 2)
        f0_out = np.concatenate([f0_out[:-n], blended_f0, f0_w[n:]])
        sp_out = np.concatenate([sp_out[:-n], blended_sp, sp_w[n:]], axis=0)
        ap_out = np.concatenate([ap_out[:-n], blended_ap, ap_w[n:]], axis=0)

    return f0_out, sp_out, ap_out, boundary_frames


def main():
    config = json.loads((BASE / "gate2_config.json").read_text())
    pipeline = KPipeline(lang_code="a")
    aligner = CtcPhoneAligner()

    target_phrases = [p for p in config["phrases"] if p["id"] in PHRASES]
    phrases = m15.build_config_with_syllable_melody(target_phrases, pipeline)

    results = []
    for phrase in phrases:
        wav_path = BASE / "gate2_audio" / f"{phrase['id']}_spoken.wav"
        align_path = BASE / "gate2_alignments" / f"{phrase['id']}.json"
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
        events_by_idx = m15.build_events_by_idx(phrase["ps_full"], phrase["pred_dur"], x.astype(np.float32), aligner)

        melody_cursor = 0
        word_param_lists = []
        target_hz_sequence = []
        n_words = len(words)
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
                target_hz_sequence.append(target_hz)
                melody_cursor += 1
                n_frames = per_syll_natural_frames if i < n_syll - 1 else (end_frame_word - cursor_frame)
                natural_dur_s = max(1, n_frames) * FRAME_DT
                target_dur_s = max(MIN_SYLLABLE_DUR_S, natural_dur_s * STRETCH)
                syll_specs.append({"phonemes": syll_phonemes, "natural_dur_s": natural_dur_s,
                                    "target_dur_s": target_dur_s, "target_hz": target_hz})
                cursor_frame += n_frames

            f0_seg = f0[start_frame:end_frame_word]
            sp_seg = sp[start_frame:end_frame_word]
            ap_seg = ap[start_frame:end_frame_word]
            start_sample = int(round(start_frame * FRAME_DT * fs))
            end_sample = int(round(end_frame_word * FRAME_DT * fs))
            x_word = x[start_sample:end_sample]
            word_ps_offset = phrase["word_offsets"][word_idx]
            word_index_map = phrase["word_index_maps"][word_idx]

            entries = build_word_entries(f0_seg, sp_seg, ap_seg, f0, sp, ap, x_word, x, syll_specs,
                                          word_ps_offset, word_index_map, events_by_idx)
            entries = assign_final_f0(entries, syll_specs)
            word_f0 = np.concatenate([e["final_f0"] for e in entries]).astype(np.float64)
            word_sp = np.concatenate([e["sp"] for e in entries], axis=0).astype(np.float64)
            word_ap = np.concatenate([e["ap"] for e in entries], axis=0).astype(np.float64)
            word_param_lists.append((word_f0, word_sp, word_ap))

        g_f0, g_sp, g_ap, boundary_frames = join_continuous(word_param_lists, JOIN_CROSSFADE_FRAMES)

        y = pw.synthesize(g_f0, g_sp, g_ap, fs, frame_period=FRAME_PERIOD_MS)

        sung_rms = float(np.sqrt(np.mean(y**2))) + 1e-9
        y = y * (spoken_rms / sung_rms)
        peak = np.abs(y).max()
        if peak > 0.98:
            y = y * (0.98 / peak)

        world_out_path = OUT_AUDIO_DIR / f"{phrase['id']}_sung_v13_world_only.wav"
        sf.write(str(world_out_path), y, fs)

        n_nan = int(np.isnan(y).sum())
        n_inf = int(np.isinf(y).sum())
        wpeak = float(np.max(np.abs(y))) if y.size else float("nan")
        clipped = int(np.sum(np.abs(y) >= 0.999))
        status = "NAN_OR_INF" if (n_nan or n_inf) else ("BLOWUP" if wpeak > 1.5 else "OK")

        boundary_sample_frames = [int(round(bf * FRAME_DT * fs)) for bf in boundary_frames]

        print(f"{phrase['id']:24s} dur={len(y)/fs:.2f}s n_words={n_words} boundaries={len(boundary_frames)} "
              f"world_peak={wpeak:.3f} clipped={clipped} nan={n_nan} inf={n_inf} [{status}] -> {world_out_path.name}")

        results.append({
            "phrase": phrase["id"], "duration_s": round(len(y) / fs, 3),
            "n_words": n_words, "n_boundaries": len(boundary_frames),
            "boundary_sample_frames": boundary_sample_frames,
            "target_hz_sequence": target_hz_sequence,
            "world_out": str(world_out_path),
            "world_peak": wpeak, "clipped_samples": clipped, "n_nan": n_nan, "n_inf": n_inf,
            "status": status,
        })

    (BASE / "v13_continuous_trajectory_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {len(results)} phrases. Results: v13_continuous_trajectory_results.json")
    print("Next: run 23_v13_vocos_pass.py inside `nix develop .#voice-vocoder` for the Vocos stage.")


if __name__ == "__main__":
    main()

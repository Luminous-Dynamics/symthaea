#!/usr/bin/env python3
"""
Step 1 item 2: force-align a KNOWN CSD phone sequence against real sung audio
within a short phrase window, using wav2vec2-lv-60-espeak-cv-ft emissions +
ctc_forced_aligner's generic Viterbi forced_align().

Unlike free recognition, the target sequence is built from CSD's own
ground-truth syllable/phoneme content (translated via phoneme_transducer.py)
-- this is genuine "forced" alignment, not phoneme guessing.

Usage: python3 align_phrase.py <csv_path> <wav_path> <start_row> <end_row>
  start_row/end_row: 0-indexed, inclusive, into the CSV's note rows.
"""
import csv as csvmod
import json
import sys

import numpy as np
import torch
import torchaudio
from ctc_forced_aligner import forced_align, merge_repeats
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

sys.path.insert(0, "/srv/luminous-dynamics/symthaea/docs/research/evidence/singing-voice-rvc-2026-07-26/step1-alignment-spike")
from phoneme_transducer import build_expected_sequence, canonical_target

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
PAD = 0.3  # seconds of context padding on each side of the phrase window


def load_notes(csv_path):
    notes = []
    with open(csv_path) as fh:
        for row in csvmod.DictReader(fh):
            notes.append({
                "start": float(row["start"]),
                "end": float(row["end"]),
                "syllable": row["syllable"],
            })
    return notes


def main():
    csv_path, wav_path, start_row, end_row = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    notes = load_notes(csv_path)
    phrase = notes[start_row:end_row + 1]

    # Build the full CSD phone list for the phrase, remembering which note
    # (and syllable-internal position) each phone belongs to, and where
    # inter-note SP/AP silence gaps fall (for reporting only -- forced_align
    # naturally absorbs real silence into CTC blank, no token needed for it).
    csd_phones = []
    origin = []  # (note_idx, syllable_str) per csd_phones entry
    prev_end = phrase[0]["start"]
    for i, note in enumerate(phrase):
        if note["start"] > prev_end + 1e-4:
            print(f"  [gap {prev_end:.3f}-{note['start']:.3f}s before note {i}: "
                  f"{note['start']-prev_end:.3f}s silence]")
        for ph in note["syllable"].split("_"):
            csd_phones.append(ph)
            origin.append((i, note["syllable"]))
        prev_end = note["end"]

    espeak_seq = build_expected_sequence(csd_phones)
    # csd_phones may include AP/SP (dropped by build_expected_sequence) --
    # recompute a parallel origin list with those dropped too, so indices
    # of espeak_seq line up 1:1 with a filtered origin list.
    origin_phonetic = [o for ph, o in zip(csd_phones, origin) if canonical_target(ph) is not None]
    csd_phonetic = [ph for ph in csd_phones if canonical_target(ph) is not None]
    assert len(espeak_seq) == len(origin_phonetic) == len(csd_phonetic)

    print(f"Phrase notes {start_row}-{end_row}: "
          f"{[n['syllable'] for n in phrase]}")
    print(f"CSD phones (phonetic only): {csd_phonetic}")
    print(f"Espeak target sequence:     {espeak_seq}")

    t0 = max(0.0, phrase[0]["start"] - PAD)
    t1 = phrase[-1]["end"] + PAD

    print(f"\nLoading model {MODEL_ID} ...")
    # Load the vocab+feature-extractor directly rather than via
    # Wav2Vec2Processor -- the bundled Wav2Vec2PhonemeCTCTokenizer tries to
    # spin up a *live* phonemizer/espeak backend for on-the-fly text->phoneme
    # conversion, a capability we don't need (we already have our own target
    # phone sequence) and that fails to init in this sandboxed environment.
    vocab_path = hf_hub_download(MODEL_ID, "vocab.json")
    with open(vocab_path) as fh:
        vocab = json.load(fh)
    blank_id = vocab["<pad>"]
    idx_to_token = {v: k for k, v in vocab.items()}
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    missing = [t for t in espeak_seq if t not in vocab]
    if missing:
        print(f"FATAL: target phones missing from model vocab: {missing}")
        sys.exit(1)
    target_ids = np.asarray([[vocab[t] for t in espeak_seq]], dtype=np.int64)

    print(f"Loading audio slice [{t0:.3f}, {t1:.3f}]s from {wav_path} ...")
    # Plain-stdlib PCM read -- torchaudio.load() in this install requires
    # torchcodec (not installed, and not worth pulling in for one bounded
    # spike; matches the plan's note that torchaudio's audio-I/O/alignment
    # APIs are in active flux around this version).
    import wave
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        assert sampwidth == 2, f"expected 16-bit PCM, got {sampwidth*8}-bit"
        frame_offset = int(t0 * sr)
        num_frames = int((t1 - t0) * sr)
        wf.setpos(max(0, frame_offset))
        raw = wf.readframes(num_frames)
    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)

    with torch.no_grad():
        inputs = feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt")
        logits = model(inputs.input_values).logits  # (1, T, C)
        log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).numpy().astype(np.float32)

    num_model_frames = log_probs.shape[0]
    slice_dur = t1 - t0
    frame_dur = slice_dur / num_model_frames
    print(f"Model frames: {num_model_frames}, slice duration: {slice_dur:.3f}s, "
          f"~{frame_dur*1000:.2f}ms/frame")

    paths, scores = forced_align(
        np.expand_dims(log_probs, axis=0), target_ids, blank=blank_id
    )
    path = paths.squeeze().tolist()
    segments = merge_repeats(path, idx_to_token)
    blank_token = idx_to_token[blank_id]
    real_segments = [s for s in segments if s.label != blank_token]

    print(f"\n{len(segments)} raw segments, {len(real_segments)} non-blank")
    print("\nfull segment trace (incl. blanks), for sustained-vowel sanity check:")
    for s in segments:
        st = t0 + s.start * frame_dur
        en = t0 + (s.end + 1) * frame_dur
        lab = "<blank>" if s.label == blank_token else s.label
        print(f"  {lab:8s} [{st:.3f}, {en:.3f}]  ({(en-st)*1000:.0f}ms, {s.end-s.start+1} frames)")

    # Walk non-blank segments in order and zip them against our expected
    # target sequence -- CTC forced alignment guarantees the same order and
    # count as the target (that's the point of *forced* alignment), so a
    # count mismatch here is a real failure worth surfacing loudly, not
    # silently truncating.
    if len(real_segments) != len(espeak_seq):
        print(f"WARNING: {len(real_segments)} aligned non-blank segments != "
              f"{len(espeak_seq)} expected targets -- reporting what aligned, "
              f"rest is a failure-to-align, not a silent truncation")

    results = []
    for i, seg in enumerate(real_segments[:len(espeak_seq)]):
        start_t = t0 + seg.start * frame_dur
        end_t = t0 + (seg.end + 1) * frame_dur
        note_idx, syll = origin_phonetic[i]
        results.append({
            "csd_phone": csd_phonetic[i],
            "espeak_target": espeak_seq[i],
            "aligned_label": seg.label,
            "start": round(start_t, 4),
            "end": round(end_t, 4),
            "note_idx": note_idx,
            "syllable": syll,
        })

    print("\ncsd_phone  espeak  aligned  start   end     note_syllable")
    for r in results:
        match = "OK" if r["aligned_label"] == r["espeak_target"] else f"MISMATCH(got {r['aligned_label']})"
        print(f"{r['csd_phone']:9s} {r['espeak_target']:6s} {match:20s} "
              f"{r['start']:.3f}  {r['end']:.3f}  note{r['note_idx']}={r['syllable']}")

    out_path = "/var/lib/symthaea/training-runs/ctc-align/last_alignment.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

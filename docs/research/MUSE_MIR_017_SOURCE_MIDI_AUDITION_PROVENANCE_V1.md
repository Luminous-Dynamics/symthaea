# MUSE-MIR-017 — Source MIDI audition provenance and audio artifact integrity (V1)

**Issue:** #11834  
**Parent:** MUSE-MIR-013 structured symbolic import  
**Stack:** MUSE-MIR-016 explicit MIDI measure sidecar  
**Implementation:** `fluid_render.rs`, `muse_studio.rs`, `symthaea-muse-protocol`, Muse Add Music UI  
**Date:** 13 November 2026  
**Status:** implementation and tests authored; not compiled or executed in this environment

## Problem

The previous import audition path took every parsed note-centric `Score`, exported it to a newly generated MIDI file, and rendered that. For MIDI inputs this throws away source event information that may be musically important:
- original event timing and tempo changes;
- note velocities;
- program changes;
- channel/controller events and pedals;
- track/channel organization.

The new MUSE-MIR-013 structured importer correctly returns the legacy note-centric Score plus a notation graph / optional MIDI measure sidecar. But rendering the Score again still would not be a replay of the source MIDI performance.

A second gap existed: `GET /api/music/import/{work_id}/audio` served `audition.wav` without a digest in the import receipt.

## Audition kinds are different evidence claims

V1 makes the distinction explicit in the protocol:

- `source_midi_performance`: the exact uploaded MIDI event stream is passed to FluidSynth for synthesis. It retains the events the renderer understands, but the resulting sound is still determined by the chosen soundfont and synthesizer. It is **not** the original acoustic recording.
- `reconstructed_score`: the note-centric Score is exported/rendered using Muse's fixed reconstruction path. This is the expected path for MusicXML and MuseScore JSON, which provide notation/score data rather than an original acoustic performance. It is not a claim of original playback reconstruction.

If FluidSynth is unavailable or the source MIDI is malformed / unsupported (including Format 2 independent sequences), Studio falls back to the reconstructed-score audition and labels the result accordingly. A fallback is never silently represented as a source-MIDI render.

## Exact source-MIDI path

`fluid_render::render_midi_bytes_to_wav`:
1. parses the exact byte slice with `midly`;
2. accepts Standard MIDI File Format 0 (single track) or Format 1 (parallel tracks) only, with header/track-count consistency for Format 0;
3. rejects Format 2, invalid MIDI bytes and sample rate zero;
4. writes the original bytes unchanged to a unique temporary `.mid` file;
5. invokes the existing FluidSynth renderer on that path;
6. applies the existing shared BS.1770-4 mastering chain;
7. removes the temporary MIDI file.

The source-MIDI audition path does not export or mutate the note-centric Score. If rendering fails, the caller falls back to the reconstructed-score path and sets `audition_kind=reconstructed_score`.

Format 2 is rejected because its tracks are independent sequential patterns, not a parallel multi-part performance timeline. This matches the separate-clock refusal in MUSE-MIR-016.

## Integrity receipt

`ImportedWorkSummary` now includes backward-compatible optional fields:
- `audition_sha256`: SHA-256 of the exact generated `audition.wav` bytes;
- `audition_kind`: `source_midi_performance` or `reconstructed_score`.

The import route calculates the digest from the in-memory WAV bytes it subsequently writes into the atomic staging directory.

The audio endpoint reads the current summary and WAV:
- if a digest is present, it recomputes SHA-256 and refuses a mismatch with HTTP 500;
- if no digest exists (legacy stored summary), it continues to serve the older artifact under an explicitly unverified legacy policy. The UI labels it as a legacy audition with unverified integrity.

The digest proves byte-level consistency with the receipt, not that the render sounds correct or resembles the original performance. Renderer/soundfont qualification remains a separate research problem.

## UI

The Add Music page now labels the playback action accurately:
- “Play source MIDI performance render”;
- “Play reconstructed score audition”;
- “Play legacy audition (integrity unverified)” when old metadata is absent.

It displays the audition SHA-256 for new receipts and retains the existing explanatory instrumentation note.

## Tests authored

### Fluid-render source-MIDI helper

- Format 0 with exactly one track is accepted.
- Format 0 with an invalid multiple-track header is rejected.
- Format 1 parallel tracks are accepted.
- Format 2 independent patterns are rejected.
- Malformed MIDI bytes are rejected before renderer/environment lookup.
- Zero sample rate is rejected before rendering.

### Muse Studio receipt

- exact WAV bytes matching the receipt SHA-256 are accepted;
- a tampered WAV is rejected;
- a legacy summary without an audio digest returns the explicit unverified path.

The test and implementation code is authored, not executed.

## External benchmark implications

This change is about trustworthy audition provenance, not transcription quality. It gives MUSE a correct input to future performance-fidelity evaluation without conflating source MIDI rendering with score reconstruction.

- Klangio's October 2026 vendor benchmark reports mean score F1 0.67 and barline F1 0.72 across its 13 test sets. These are vendor-authored comparison points, not a directly matched independent leaderboard.
- MIREX 2026 Audio-to-Score reports blind MV2H around 0.95 on the familiar Quartets test but around 0.60 on QuartetsMIREX's held-out composer set. That is a generalization issue, not an audition-render issue.
- *A Dual Evaluation for Music Transcription* (August 2026) found notation similarity and rendered-playback similarity favor different transcription pipelines among 24 configurations, with CLEWS the lowest-cost playback metric and the strongest correlation with listener judgments in that piano-focused study.

Sources:
- https://klang.io/blog/music-transcription-benchmark-2026/
- https://futuremirex.com/mirex/wiki/2026:Audio-to-Score_Transcription_Results
- https://arxiv.org/abs/2608.04511
- https://github.com/pingw220/AMT-Dual-Eval

### Qualification targets (not achieved results)

1. 100% byte preservation from uploaded MIDI source to the temporary MIDI file supplied to FluidSynth.
2. Zero source-MIDI audition claims for malformed, Format-2 or unsupported input.
3. 100% exact WAV digest validation on new receipts; zero acceptance of tampered WAVs.
4. 100% explicit reconstructed-score labeling for MusicXML/MuseScore and all MIDI fallback cases.
5. Renderer/soundfont identity, version and configuration become part of a future render receipt before comparing across runs.
6. Only compare rendered playback to an original audio performance when that source audio exists and the renderer configuration is pinned. Do not score a MIDI-only import as audio-to-audio fidelity.

## Verification boundary

- Tests are authored, not compiled or executed.
- No local `cargo`, `rustc` or `rustfmt` toolchain is available in the authoring environment.
- No exact-head CI pass, live FluidSynth render, browser test or acoustic similarity result is claimed.
- Rendering the source MIDI through FluidSynth preserves source event data that FluidSynth supports; it does not recover the original synthesizer, soundfont, room, microphones or recording.

# MUSE-MIR-017 — Source MIDI Audition Provenance

**Date:** 13 November 2026  
**Status:** implementation and deterministic tests authored; exact-head Rust/wasm qualification pending.  
**Tracking issue:** [#11834 — Preserve source MIDI event provenance in imported-work audition](https://github.com/Luminous-Dynamics/symthaea/issues/11834)  
**Base:** [#11831 — MUSE-MIR-016 source-bound MIDI measure sidecar](https://github.com/Luminous-Dynamics/symthaea/pull/11831)

## Problem

Muse already stores the uploaded source file and parses it into a note-centric `Score`. That Score is useful for analysis, editing and notation-aware operations, but it does not preserve every MIDI performance event: program changes, controllers, channel events, track layout and other source-level information are not all represented by the legacy `Score`.

The previous imported-work audition regenerated MIDI from the Score using a fixed Classical palette. That made the audition a reconstruction even when the uploaded source was already MIDI.

The key architectural distinction is:

- **Source artifact:** exact uploaded MIDI bytes and their identity.
- **Analysis projection:** the current note-centric Score and any explicit measure sidecar.
- **Audition artifact:** either a FluidSynth render of the original supported MIDI event stream or a separately disclosed reconstruction from the Score.

These are linked artifacts, not interchangeable representations.

## Implementation

### 1. Source-MIDI performance path

For MIDI uploads only:

1. Parse the upload using `midly::Smf::parse`.
2. Admit simultaneous-sequence Standard MIDI File Format 0 (`SingleTrack`) or Format 1 (`Parallel`) only.
3. Refuse malformed data and Format 2 (`Sequential`) for the source-performance audition path.
4. Materialize the exact uploaded byte slice to a unique temporary `.mid` file.
5. Read it back and require byte-for-byte equality before invoking FluidSynth.
6. Invoke the existing FluidSynth WAV renderer against that path, without exporting or rewriting a MIDI file from the note-centric Score.
7. Read the temporary MIDI again after rendering; ordinary path mutation/substitution causes the source-performance result to be discarded.
8. Remove the temporary file on all ordinary completion paths.

This is local byte-preservation evidence. It does not prove which inode/pages a malicious host or kernel loaded if it changes and restores the path between observations. The trusted renderer-invocation receipt programme remains a separate assurance layer.

### 2. Explicit fallback semantics

The importer records one typed audition kind:

- `source_midi_performance`: supported uploaded Format 0/1 source MIDI was rendered by FluidSynth.
- `reconstructed_score`: the audition was generated from the parsed Score. This is used for MusicXML, MuseScore JSON, unsupported Format 2 MIDI, unavailable FluidSynth, or failed source-MIDI rendering.
- `legacy_unverified`: compatibility default for older persisted summaries that predate the audition-kind and WAV-digest fields.

A source-MIDI performance audition is a newly synthesized SoundFont rendering of MIDI event data. It is **not** the original acoustic recording and does not claim to reproduce the original instrument sample library or room.

The existing reconstructed Score path remains available. The source upload itself is still stored unchanged and remains the authority for the imported source artifact.

### 3. Content-bound WAV artifact

Every newly created imported-work summary stores:

- typed `audition_kind`;
- renderer name;
- instrumentation disclosure appropriate to that kind;
- exact `audition_wav_sha256` of the bytes written as `audition.wav`.

`GET /api/music/import/{id}/audio` recomputes the WAV SHA-256 before serving any newly created artifact. A mismatch or a new-format summary with a missing digest fails closed with HTTP 500.

Legacy records remain readable under an explicit policy:

- missing `audition_kind` defaults to `legacy_unverified`;
- missing WAV digest remains `None`;
- the audio endpoint may serve that legacy file with `X-Muse-Audio-Integrity: legacy-unverified`;
- new records with no digest do not receive the legacy exception;
- verified new records are served with `X-Muse-Audio-Integrity: verified-sha256`.

### 4. UI disclosure

The Add Music page now labels source MIDI performance auditions distinctly from reconstructed Score auditions and legacy unverified audio. It displays the recorded WAV digest when present, or an explicit legacy/unverified message when absent.

## Tests authored

Seven Studio tests cover:

1. Format 0 admission.
2. Format 1 admission.
3. Format 2 sequential rejection.
4. Malformed MIDI rejection.
5. Exact byte equality for the materialized source MIDI input.
6. Acceptance of exact WAV digest.
7. Rejection of tampered WAV bytes and of missing digest on a new-format artifact; explicit legacy missing-digest policy.

Three protocol tests cover:

- legacy summary deserialization to `LegacyUnverified` with no digest;
- source-MIDI audition kind and WAV digest round-trip;
- stable `reconstructed_score` wire spelling.

These tests are authored but not reported as executed until the exact-head workflow completes.

## Validation workflow

`.github/workflows/muse-source-midi-audition.yml` checks out the exact event head, pins Rust 1.96.0, resolves workspace metadata, checks Rust formatting, runs protocol and Studio provenance tests, checks the Leptos UI for `wasm32-unknown-unknown`, and runs Clippy with warnings denied.

The root workspace contains three source/test-only directories matched by `crates/core/*` but lacking Cargo manifests on this branch (`symthaea`, `symthaea-observation-attestation`, `symthaea-iit`). They are explicitly excluded; no placeholder packages are fabricated and no tests are skipped.

## Claim boundary

This work establishes a distinction in the import pipeline and binds the generated WAV bytes to the summary that serves them. It does not establish:

- complete parsing of every MIDI controller or playback semantic into the Score;
- source-format-independent audio equivalence;
- preservation of the original recording or original SoundFont;
- trusted-host or malicious-kernel execution attestation;
- native renderer capability beyond the existing reconstruction path;
- human-perceived fidelity or SOTA superiority.

The source MIDI remains immutable and separately addressable. Richer symbolic event graphs, repeat traversal, MIDI measure sidecars and renderer invocation evidence retain their own schemas and qualification boundaries.

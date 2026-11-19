# MUSE Render Execution Receipt v1

**Status:** Draft integration contract plus standalone Rust verifier authored; qualification pending. No real render or score-to-audio conformance result is claimed.
**Parent:** [MUSE-RENDER-VERIFY-001 / #11389](https://github.com/Luminous-Dynamics/symthaea/issues/11389)

## Why this increment exists

MUSE already has several partial pieces, but they do not yet form one exact score-to-render chain:

- [Score-native edit transaction / PR #11332](https://github.com/Luminous-Dynamics/symthaea/pull/11332) currently implements a velocity-only edit contract with an independent candidate verifier.
- [Part-aware MIDI performance / PR #11541](https://github.com/Luminous-Dynamics/symthaea/pull/11541) defines a MIDI manifest binding the source Score, orchestration timeline, capability bindings, exact MIDI bytes and per-note/part receipts.
- The existing [RendererRecipe](https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-muse/src/piece_recipe.rs) already has fields for renderer version, binary digest, SoundFont digest and render-environment digest.
- The native and optional FluidSynth routes can produce audio, but a successful render call or a WAV hash does not prove that the intended MIDI bytes were consumed or that the score is audible in the output.
- [PR #8944](https://github.com/Luminous-Dynamics/symthaea/pull/8944) prototypes symbolic/audio content binding, but its FullScore projection is limited to one monophonic part and its current locality-dimension guard rejects velocity mutation. It therefore does not yet close the score-edit v1 to rendered-audio path.

This proposal adds the missing **attempt-level join**, not another renderer and not another generic preservation metric.

## Contract boundary

One receipt describes exactly one render attempt. It joins:

`source Score → authorized edit plan → candidate Score → MIDI manifest + exact MIDI bytes → renderer invocation → exact WAV bytes → independent evidence`

The machine-readable shape is in [muse-render-execution-receipt-v1.schema.json](schemas/muse-render-execution-receipt-v1.schema.json); the process record is specified by [muse-render-invocation-v1.schema.json](schemas/muse-render-invocation-v1.schema.json). The MIDI fields intentionally follow the PartId-aware muse-midi-performance-v1 manifest contract, including its 960 ticks-per-quarter resolution and separate score, timeline, instrument-binding, capability, manifest, and MIDI digests. The standalone verifier in tools/muse-render-receipt checks receipt shape, canonical seal, execution-state invariants, and cross-artifact relationships.

### Required joins

1. `source.source_score_sha256` and `source.candidate_score_sha256` identify the exact serialized score artifacts. The candidate digest must equal the score digest carried by the MIDI manifest.
2. `edit.plan_sha256`, `edit.edit_receipt_sha256`, and `edit.verification_receipt_sha256` bind the requested operation, executor change record, and independent score-native verifier. The verifier verdict must be `pass` before rendering is eligible for acceptance.
3. `midi.manifest_sha256` binds the MIDI manifest; `midi.midi_sha256` binds exact Standard MIDI File bytes. The manifest score digest must equal `source.candidate_score_sha256`; its MIDI digest must equal `midi.midi_sha256`.
4. `execution.input_midi_sha256` must equal `midi.midi_sha256`. A completed attempt requires an independently observed zero exit code and a produced audio artifact.
5. `execution.output_audio_sha256` must equal `audio.audio_sha256`. The digest covers exact WAV bytes; decoded PCM format and frame count are recorded separately.
6. Renderer binary, configuration, environment, and relevant sound-bank/asset digests are pinned. A fallback renderer is never silently substituted: it is a distinct renderer identity and a distinct attempt.
7. MIDI-versus-score conformance, rendered-audio-versus-score conformance, protected-audio preservation, and human evaluation remain separate evidence results. One passing layer cannot compensate for another failing or abstaining.

## Execution-state semantics

| Attempt state | Meaning | Acceptance |
|---|---|---|
| `not_submitted` | Dispatch boundary was not crossed | No |
| `submitted` | Submission occurred; completion is not yet known | No |
| `completed` | Renderer terminated successfully and exact output bytes were captured | Not by itself |
| `failed` | A terminal failure was observed and reconciled | No |
| `outcome_unknown` | Timeout, crash or lost observation leaves execution, charge or output uncertain | No |

**Unknown outcome is not retry permission.** Reconcile the original attempt using its stable attempt identity and execution evidence. If another render is required, create a new attempt ID and account for its budget separately. Never rewrite an unknown attempt as failed merely to enable a retry.

Partial WAV bytes may be retained for diagnostics, but a partial output cannot become a completed render by being hashed. A zero exit code also cannot substitute for independent conformance checks.

## Trust and claim boundaries

- A SHA-256 value proves byte identity relative to the bytes actually hashed; it does not prove who produced them.
- The receipt's own digest is computed using the project-specific v1 canonical serialization profile: the typed schema admits strings, booleans, nulls, arrays and integers; object keys are serialized lexicographically in compact JSON; `receipt_sha256` is omitted; the domain tag `luminous-dynamics-muse-render-execution-receipt-v1` plus a NUL delimiter is prepended; then SHA-256 is applied. This is deterministic for this schema but is not claimed as full RFC 8785 JCS. It seals the record but is not an identity signature.
- `unsigned_local` is suitable for development fixtures only. `workflow_observed` means a workflow captured the execution; it is not automatically trusted hardware attestation. `signed` requires an external key registry and signature verification.
- A process invocation record that names an input MIDI digest and output WAV digest is only as trustworthy as the executor that observed the process boundary. Stronger claims require capture inside a controlled runner/container and attestation of runner identity.
- A deterministic MIDI manifest is not proof of acoustic realization. Audio conformance requires an independent observer, calibrated per renderer/asset stratum, plus explicit abstention for unidentifiable voices or instruments.
- Audio preservation does not prove target attainment. A no-op can preserve everything and still fail the requested edit.
- A semantically valid alternative should be judged against the edit contract, not rejected merely because it differs from one reference waveform.

## Initial vertical slice

The first executable slice should be deliberately narrow:

1. Start from a frozen multi-part Score with explicit PartId and source event ordinals.
2. Apply the current velocity-only ScoreEditPlanV1 and require the independent score-edit verifier to pass.
3. Realize the candidate through the PartId-aware MIDI exporter; independently parse the MIDI bytes and verify note pitch, onset, offset, velocity, track and part mapping against the candidate Score and manifest.
4. Render those exact MIDI bytes with one pinned renderer and asset/configuration. Disable silent fallback; a missing FluidSynth binary or SoundFont is an explicit unavailable/abstain outcome.
5. Re-hash the actual MIDI input and WAV output at the execution boundary and emit this receipt.
6. Run known-answer audio checks on a small calibrated patch/pitch/velocity panel. Do not generalize results beyond the qualified panel.
7. Add negative controls: changed MIDI after manifest creation, wrong SoundFont digest, wrong renderer binary, missing/wrong note, octave error, velocity not transmitted, silent WAV, truncated WAV, WAV from a different MIDI, fallback renderer substitution, and outcome-unknown retry attempt.

This is not a claim that the existing velocity edit has already been rendered or qualified. Authored, compiled, executed, and qualified states must remain distinct.

## Standalone verifier

The isolated Rust package at [tools/muse-render-receipt](../../tools/muse-render-receipt/README.md) is outside the root Cargo workspace. It has receipt-only and full-bundle modes, documented in its README.

The full-bundle mode checks exact byte hashes for the Score, edit plan/receipt/verification report, MIDI manifest, orchestration timeline, instrument bindings, capability profiles, MIDI, renderer binary/environment/configuration/asset, invocation, process streams and WAV. It independently recomputes the current velocity-only ScoreEditPlanV1 candidate from the exact source Score and plan, validates the complete edit delta list and verification report counts/issues, then checks every candidate Score note against its MIDI note receipt (ordinal, PartId, pitch, exact onset/offset at 960 PPQ and exporter velocity quantization). The invocation JSON is parsed and checked against the unique attempt ID, renderer identity and hashes, exact MIDI input digest/path, WAV output path and digest, execution timestamps and exit code. It verifies that every timeline segment maps one-to-one to a MIDI track, that instrument renderer/patch/program bindings explain each track, and that every used instrument has a hashed capability profile. It deserializes the PartId-aware MIDI manifest using the exporter's field order and digest recipe; parses the actual Standard MIDI File; and compares track names, program changes, note-on and note-off event multisets against the sidecar. It also checks WAV byte length, sample rate, channel count, frame count, PCM finiteness and non-silence. A structured invocation is still an executor-reported record, not cryptographic proof that an untrusted process consumed the declared input.

The CLI returns exit code 3 for outcome_unknown; this is an unresolved execution, not a passing verification and not permission to retry under the same attempt ID. Receipt-only mode never claims bundle verification. Even a full bundle integrity pass sets acceptance_eligible=false: the verifier does not authenticate signatures, prove that an untrusted renderer consumed the named input, support future score-edit operation classes beyond velocity-only v1, establish score-to-audio perceptual conformance, or authorize the musician's commit.

The package includes known-answer unit tests for receipt tampering, cross-artifact score-digest mismatch, unresolved execution, MIDI sidecar agreement and a wrong-pitch MIDI negative control. A dedicated workflow pins Rust 1.96.0 and runs format/tests without loading the monorepo workspace. The direct dependency versions are exact-pinned to the current root lockfile, but the isolated package does not yet have its own Cargo.lock because the authoring environment has no Cargo toolchain; full transitive dependency freezing is therefore still pending. The workflow has been observed queued; no Rust compile/test pass is claimed. The active environment does not have cargo, rustc or rustfmt, so exact-head local qualification was not possible.

## Qualification targets (prospective)

- 100% rejection of digest-link mismatches across Score, edit receipt, MIDI manifest, MIDI bytes and WAV bytes.
- 100% rejection of silent, truncated, wrong-MIDI and fallback-substitution negative controls.
- 100% exact MIDI event/part mapping on deterministic known-answer fixtures.
- 100% correct terminal-state handling, including no accepted mutation after `outcome_unknown`.
- 100% explicit abstention when renderer/asset identity or verifier observability is missing.
- For the calibrated audio panel, preregister pitch/onset/offset/velocity tolerances on development fixtures and freeze them before held-out confirmation. Do not set universal acoustic thresholds from one patch.
- Human listening remains a separate gate for perceptual change and coherence.

These are acceptance objectives, not current measurements.

## Files

- `schemas/muse-render-execution-receipt-v1.schema.json` — JSON Schema 2020-12.
- `fixtures/muse-render-execution-receipt-v1.completed.example.json` — synthetic shape fixture only; sentinel digests are not real artifacts.
- `fixtures/muse-render-execution-receipt-v1.outcome-unknown.example.json` — synthetic unresolved-execution fixture only.
- `schemas/muse-render-invocation-v1.schema.json` — structured renderer process record.
- `tools/muse-render-receipt/` — isolated Rust receipt/MIDI/WAV integrity verifier and CLI.
- `.github/workflows/muse-render-receipt.yml` — focused Rust format/test workflow.

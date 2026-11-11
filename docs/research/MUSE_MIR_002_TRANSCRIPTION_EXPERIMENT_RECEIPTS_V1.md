# MUSE-MIR-002 — Transcription experiment provenance receipts (V1)

**Parent benchmark:** [MUSE-MIR-001 #11597](https://github.com/Luminous-Dynamics/symthaea/pull/11597)  
**Research programme:** [Perception grounding #10265](https://github.com/Luminous-Dynamics/symthaea/issues/10265)  
**Date:** 11 November 2026  
**Status:** research implementation; focused Rust tests authored, not yet compiled or executed

## Purpose

A transcription score is not a reproducible experiment unless the exact inputs, model, preprocessing, conditioning, runtime, rights and output can be recovered.

MUSE-MIR-002 introduces a model-neutral experiment receipt. It does not run a model, download weights, call a vendor API, or certify model accuracy. Its job is to bind the provenance of one transcription attempt to the MIDI output and the MUSE-MIR-001 evaluation report.

The receipt works for:
- locally executed open-weight models;
- local binaries/containers whose weights are opaque or not separately available;
- hosted inference APIs where weights are inaccessible.

These modes remain distinct in the receipt. A hosted model must not be reported as a locally reproducible open-weight run.

## Receipt contents

Adds `crates/domains/symthaea-muse/src/transcription_experiment.rs`, exported behind the existing `theory` feature.

### Input and dataset identity

The receipt binds:
- benchmark ID, dataset ID/revision, split and sample ID;
- digest and reference to the dataset manifest;
- SHA-256 of the source audio bytes before preprocessing;
- SHA-256 of the canonical model-input bytes after resampling, channel conversion, normalization and chunking.

For tensor-based inference, the model-input digest must use a documented canonical encoding that binds shape, dtype and exact tensor bytes. For chunked hosted requests, it should bind a canonical ordered manifest of the exact chunk payloads, request boundaries and overlap policy, not merely the original WAV file.

### Model, adapter and inference configuration

The receipt records:
- model name, revision, delivery mode and provider;
- artifact SHA-256 for open weights, with the actual model bytes supplied to validation;
- optional code/container revision;
- adapter ID and revision that transform model output into MIDI;
- model license and a reference to the applicable terms;
- a digest and retrievable reference for the complete inference/decode configuration;
- random seed when one was used.

For a multi-file model checkpoint, define a canonical artifact manifest: stable relative paths, file sizes and per-file SHA-256 values sorted by path. Hash that manifest as the checkpoint artifact digest. Do not hash arbitrary archive bytes that can change because of compression metadata unless the archive itself is the exact deployed artifact.

### Preprocessing and conditioning

The first-class preprocessing fields include source/model sample rate, source channel count, channel policy, resampler ID, normalization ID, chunk duration and chunk overlap. A separate digest/reference binds additional preprocessing parameters such as filter coefficients, tensor layout, chunk construction options and frontend configuration.

Conditioning is explicit:
- `None`;
- instrument presence derived from the benchmark ground truth;
- instrument presence from a separate predictor;
- user/operator-provided instrument labels.

Every non-empty conditioning mode requires a digest of the exact conditioning payload. Ground-truth-conditioned and audio-only runs are different experimental conditions and must never share a leaderboard row.

### Runtime

The receipt records provider, device, accelerator, runtime/framework identity, numeric precision, batch size, measured wall time and optional peak memory. Runtime hardware and latency are observations; they do not, by themselves, prove compute-equivalent comparisons.

### Rights and benchmark eligibility

Input rights and model usage terms are separately represented. Scopes distinguish internal-only, research-only, non-commercial research, public benchmark-result publication, public dataset redistribution and commercial product use.

The helper `transcription_experiment_public_benchmark_eligible_v1` is deliberately conservative:
- input rights must be resolved as public domain, owned/licensed or consented research;
- a non-public-domain input needs a rights reference;
- both input rights and model terms must explicitly permit public benchmark-result publication or wider redistribution;
- a model-license reference must be present.

This is a policy signal, not a legal conclusion or license parser. A human must review the actual license, consent and service terms. A non-commercial research dataset or model is not made commercially eligible by recording its name in the receipt.

### Output and evaluator binding

The receipt binds:
- exact predicted Standard MIDI bytes;
- exact part-aware reference manifest digest;
- evaluator schema and algorithm;
- the exact MUSE-MIR-001 report digest.

Validation recomputes hashes for supplied source audio, canonical model input, dataset manifest, inference config, preprocessing config, predicted MIDI and model artifact (when a model artifact digest is claimed). It also recomputes the nested evaluator-report commitment and the experiment-receipt commitment.

## Important trust boundary

SHA-256 commitments detect accidental substitution when the expected digest is held independently. They do **not** prove who created a receipt: anyone who can modify the fields can also calculate a new receipt digest.

For high-trust public benchmark submissions, the next deployment layer should sign the canonical receipt digest with a benchmark-runner identity or managed signing key, publish the signer identity/key history, and retain the signed receipt with CI or benchmark artifacts. Signature verification and key lifecycle are intentionally outside this V1 module.

The evaluator report is nested and its commitment is recomputed, but the receipt validator does not re-run transcription evaluation: it does not receive the full reference MIDI artifact in this API. It checks the report digest and that the report binds the same predicted MIDI, reference manifest, schema and algorithm. This is integrity linkage, not proof that an independent evaluator executed the report.

## Tests authored

Fourteen focused Rust tests currently cover:
1. valid open-weight receipt binding all supplied artifacts;
2. source-audio byte substitution;
3. MIDI output substitution;
4. evaluator report tampering;
5. open-weight artifact requirement;
6. hosted API identity without a locally claimed model artifact;
7. conditioning payload requirement;
8. explicit ground-truth conditioning;
9. unresolved rights exclusion from public benchmark eligibility;
10. non-commercial-only rights exclusion;
11. invalid chunk overlap;
12. receipt metadata tampering;
13. preprocessed model-input and inference-config substitution;
14. model artifact bytes that do not match the claimed checkpoint digest.

These tests are authored, not compiled or run in the authoring environment.

## Integration sequence

1. Freeze the MUSE-MIR-001 evaluator semantics and pin the mir_eval 0.8.2 parity fixture set.
2. Emit one experiment receipt per dataset sample and inference configuration.
3. Store the receipt alongside exact audio, model-input artifact/manifest, prediction MIDI and evaluator report.
4. Aggregate benchmark rows only from receipts that pass structural, digest and rights-policy validation.
5. Keep audio-only, model-conditioned, ground-truth-conditioned and user-conditioned results separate.
6. Add signed runner attestations before treating public benchmark submissions as independently attributable.

## Benchmark reporting rules

Every public comparison should disclose:
- exact dataset revision, split, sample count and rights basis;
- model artifact/checkpoint and adapter revisions;
- model input and preprocessing digest;
- instrument-conditioning source;
- runtime hardware, precision, batch size, wall time and memory (when available);
- note-level onset/full-note scores, onset-only/offset-only, AOR and frame metrics with their exact settings;
- error/abstention counts and empty-reference false-positive performance;
- confidence intervals and held-out genre/instrument slices.

A benchmark row without a valid receipt is exploratory data, not a reproducible leaderboard result.

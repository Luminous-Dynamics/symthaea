# MMAE package-integrity preflight v1

**Date:** 6 November 2026  
**Implementation:** tools/mel-mmae-package-validator/  
**Related programme:** Issue #9284 — Muse paired-target editing and MMAE benchmark

## Why this is separate

The public challenge page documents a paired output package: an edited audio
file for each test item and a JSONL mapping its sample ID to a relative audio
path. The baseline repository writes audio/<id>.wav and submission.jsonl.
The hidden test set and SDK are scheduled for 10 November 2026. This validator
provides a local deterministic preflight without depending on the future SDK.

## Validation boundary

The validator enforces the published reference manifest example and verifies
that the package is internally complete:
- one unique manifest record for every ID supplied by the caller;
- no extra or duplicate sample IDs;
- exact audio/<id>.wav mapping;
- safe package-relative non-symlinked regular paths;
- structurally valid RIFF/WAVE container, non-empty data;
- streaming SHA-256 and byte-length evidence.

It intentionally does not impose audio codec subtype, sample rate, bit depth,
channel count, loudness, duration, musical quality, IFR/CR/EMR or model
eligibility. The public reference layout shows the .wav extension but does
not define these other file constraints; add them only if the released SDK or
organizers specify them.

## Challenge comparison and timing

The official leaderboard states:
- hidden test release and SDK/leaderboard opening: 10 November 2026;
- 500 hidden items in each of the Single Model and Agent tracks;
- primary ranking by overall EMR, ties by IFR then CR;
- public development benchmark: 2,000 examples and 17,741 rubrics.

Registration closed 8 October. Agent Track model components and weights had
to be public before 1 November. The validator cannot determine whether a
team, checkpoint or runtime satisfies these conditions; verify them separately
before claiming official entry.

## Tests and evidence

The standalone crate includes tests for valid packages, expected-ID coverage,
duplicate and extra IDs, duplicate JSON keys, unknown fields, path traversal,
symlinked paths, missing files, empty files, WAV container structure,
malformed RIFF lengths, empty data chunks, count mismatch, missing manifest,
blank JSONL records, expected-ID input and digest stability.

A pinned Rust 1.96.0 workflow formats, tests and runs a generated local
reference package end-to-end. These are package-integrity tests only, not
MMAE evaluation results or official submission evidence.

## Next integration

Once the official SDK is released, add a separate versioned SDK compatibility
adapter that invokes the organizer's validator and records the exact SDK
version, command, exit status and output digest. Keep this local preflight as
an early fail-fast check; do not claim it replaces the SDK.

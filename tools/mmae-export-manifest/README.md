# MMAE content-addressed submission manifest exporter

A Rust-first local exporter for the official Audio Editing Challenge submission.jsonl. It verifies the complete sample-ID mapping, safe relative file paths and exact SHA-256 content identities, then writes the official two-field JSONL manifest.

## What it does

- Requires exactly 500 unique expected IDs from the selected SDK track's expected-ID JSON artifact.
- Requires exactly one output mapping per expected ID.
- Rejects unknown/missing/duplicate IDs, duplicate paths and noncanonical paths.
- Rejects symlinks in the submission root, path components and an existing submission.jsonl.
- Requires existing regular files ending in .wav.
- Recomputes every file's SHA-256 and compares it to the local output mapping.
- Emits one deterministic JSON object per ID, with only the fields id and audio_path.
- Enforces the official SDK's 1 MiB submission.jsonl limit.
- Publishes submission.jsonl atomically and refuses to overwrite different existing content.
- Has no upload or credential interface.

## Inputs

1. The exact track-specific expected IDs JSON array from the official SDK's data/policy artifact.
2. An outputs.jsonl file. Each nonempty line must contain exactly:

~~~json
{"sample_id":"<official-id>","audio_path":"audio/<official-id>.wav","audio_sha256":"sha256:<64-lowercase-hex>"}
~~~

3. An existing submission directory containing the referenced audio files.

The output map is a local provenance format. It is not copied into the official submission manifest.

## Run

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/mmae-export-manifest/Cargo.toml
cargo +1.96.0 test --manifest-path tools/mmae-export-manifest/Cargo.toml
cargo +1.96.0 run --manifest-path tools/mmae-export-manifest/Cargo.toml -- \
  --expected-ids /path/to/Audio-Editing-Challenge-SDK/data/aec_agent.json \
  --outputs outputs.jsonl \
  --submission-dir ./my_submission
~~~

For Track 1, supply the SDK's aec_single_model.json instead. The exact expected-ID file must correspond to the selected track and should be refreshed from the current SDK/server policy before official evaluation.

## Official SDK remains authoritative

This exporter deliberately does not parse WAV encoding or create USTAR archives. The organizer's official MIT-licensed SDK must perform final validation and packaging. It checks the exact two-field UTF-8 submission.jsonl, every required ID, unique safe relative .wav paths, real 16 kHz mono 16-bit PCM RIFF/WAVE data, maximum duration, symlink/path safety, manifest-first USTAR members, padding/terminators and the 2 GiB archive limit.

The SDK fetches current server policy, so a local expected-ID file is not proof that the server policy has not changed. A successful local export is not an official validation result.

## Security and rights

- This tool never accepts, stores, prints or transmits the organizer's evaluation URL or secret key.
- It does not upload, submit, or start an evaluation.
- It does not invent audio for missing/unsupported/failed samples; incomplete sets fail closed.
- The exporter validates audio file bytes against their declared digests but does not establish audio quality or task correctness.
- The official SDK code is MIT licensed. Challenge audio and data rights remain governed by the challenge and source dataset terms.

## Claim boundary

A successful export proves only that 500 mappings match the supplied expected-ID artifact, referenced files exist as safe regular files and their SHA-256 digests match. It does not prove WAV validity, USTAR validity, official policy freshness, competition eligibility, rubric correctness or perceptual quality.

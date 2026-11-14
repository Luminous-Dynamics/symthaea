# MMAE-EXPORT-001 — Content-addressed official submission manifest export

**Date:** 10 November 2026  
**Status:** local Rust manifest exporter and synthetic file-system tests; no official submission or benchmark score claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11519

## Official interface verified from the organizer SDK

The current Audio Editing Challenge SDK README and validator specify:
- Python 3.11+ for the official SDK;
- submission.jsonl plus an audio directory;
- each nonempty UTF-8 JSONL line contains exactly two string fields: id and audio_path;
- every one of the selected track's 500 official IDs exactly once;
- unique canonical relative POSIX .wav paths;
- nonempty little-endian RIFF/WAVE PCM: 16 kHz, mono, 16-bit integer PCM;
- maximum 300 seconds per WAV;
- uncompressed USTAR, manifest first, regular files/directories only, exact WAV members, valid padding and two zero terminator blocks;
- 2 GiB total archive cap including headers and padding;
- current server policy is fetched by the SDK before validation;
- organizer-issued track URL and secret key are required to upload, and upload alone does not start evaluation.

The SDK's validator parses actual RIFF chunks/PCM data and USTAR members. Reimplementing it in Muse would create unnecessary drift and a second source of truth.

## Rust exporter boundary

The Rust adapter accepts:
1. the exact expected-ID JSON array from the selected SDK track artifact/policy;
2. a local outputs.jsonl with sample_id, relative audio_path and expected SHA-256;
3. an existing directory containing those output files.

It checks the expected set is exactly 500 unique IDs, every ID is mapped once, all paths are unique/canonical and safe, each path component is not a symlink, the target is a regular file under the submission root, and the on-disk SHA-256 matches the output mapping. It writes only the exact official two-field manifest rows, in deterministic sample-ID order, using an atomic write and refusing to replace stale different content.

No audio is synthesized for missing/failed/unsupported cases. The export fails closed when an ID or output is missing. This makes capability failures visible rather than hiding them by silently omitting rows.

## Integration pipeline

~~~text
MMAE / VMEA / session-ledger evidence
  -> final output artifact manifest
  -> Rust MMAE manifest exporter
  -> submission.jsonl + audio/
  -> official SDK current-policy validation
  -> official USTAR packer
  -> organizer-issued upload credentials (only when authorized)
  -> official evaluator / hidden rubrics
~~~

The local exporter does not know credentials and does not upload or submit. The official SDK remains authoritative for WAV encoding, RIFF validation, archive format/size and current server policy.

## Negative controls

Synthetic tests cover:
- expected-ID cardinality other than 500;
- missing/extra/duplicate IDs;
- duplicate WAV paths;
- parent traversal, absolute path, wrong suffix and noncanonical path;
- symlinked root, path components and existing submission manifest;
- missing audio file;
- output digest mismatch;
- existing stale manifest content;
- deterministic ordering, exact two-field output and idempotent re-run.

The dummy file payloads used by these library tests are not WAV fixtures. The exporter does not claim to validate WAV encoding; final audio fixtures must pass the organizer's official SDK.

## Claim boundary

A local export pass proves that the submission mapping is complete relative to the supplied expected-ID artifact and that referenced files match their declared SHA-256 digests. It does not prove that the ID artifact is current, that files are valid WAVs, that USTAR packaging is compliant, that challenge entry is authorized, or that output audio satisfies hidden rubrics.

## Related work

- #11506 VMEA-001 instruction/preservation rubric scorer
- #11517 MMAE-TRACE-001 multi-turn session ledger
- #11488 VCCR-001 correction trajectory assurance
- official SDK: https://github.com/Audio-Editing-Challenge/Audio-Editing-Challenge-SDK

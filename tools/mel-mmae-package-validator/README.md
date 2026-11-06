# MMAE submission package validator

A Rust-first, offline preflight for the published ICASSP 2027 Audio Editing
Challenge reference package layout. It checks package completeness and
file/container integrity; it does **not** score musical edits, run the
organizer's hidden rubrics, authenticate a submission, prove challenge
registration, or guarantee that the later organizer SDK will accept a package.

## Reference contract

The challenge website's published example is one JSONL object per sample:

~~~json
{"id":"<sample_id>","audio_path":"audio/<sample_id>.wav"}
~~~

The audio directory and JSONL manifest are packaged together. This tool follows
that reference profile strictly: exact audio/<id>.wav path, no unknown manifest
fields, no duplicate IDs, and every caller-supplied expected ID exactly once.
The caller supplies the released test IDs in a UTF-8 text file with one ID per
line. The optional expected-count check can be set to 500 for a released
track-specific test set; local fixtures can use smaller counts.

## Checks

- The JSONL manifest exists, is UTF-8, has bounded record sizes, and each line
  is an object with exactly id and audio_path.
- Duplicate JSON keys, unknown fields, blank records, unexpected IDs,
  duplicate IDs, missing IDs and count mismatches are rejected.
- Audio paths must exactly match audio/<sample_id>.wav; IDs cannot contain path
  separators, drive delimiters or control characters.
- Manifest and audio paths must be package-relative regular files. Symlinked
  path components, dot/parent segments, non-files and paths outside the
  canonical package root are rejected.
- WAV files must have a consistent RIFF/WAVE chunk table, a fmt chunk of at
  least 16 bytes and non-empty data. SHA-256 and byte length are emitted for
  each verified file.

The WAV check does not constrain codec subtype, sample rate, channel count,
sample width, loudness, duration or perceived quality. These requirements are
not defined by the public submission example and should only be added from an
official SDK/profile. Filesystem checks are a local preflight, not a defense
against a concurrently mutating or hostile package filesystem.

## Run

~~~sh
cargo fmt --check
cargo test --locked
cargo run --locked -- validate \
  --root ./submission-package \
  --expected-ids ./released-test-ids.txt \
  --expected-count 500 \
  --report ./package-preflight.json
~~~

Exit codes: 0 valid package, 2 package rejected, 1 command/input error. The
JSON report records a stable schema version, each rejection, and one
SHA-256/byte-length record per verified WAV.

The preflight report is not a challenge submission receipt. It does not
authenticate team eligibility, model release dates, provenance, copyright or
dataset rights; those must be handled separately under the organizers' rules.

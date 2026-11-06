# REA formal provider-observation envelope fuzzing

This isolated support crate and nested cargo-fuzz workspace exercise the actual Rust formal-envelope parser and typed consistency validator in src/rea_compat/formal.rs.

## Target

- formal_envelope: arbitrary raw bytes enter parse_formal_provider_observation_json, including recursive duplicate-key detection, closed-schema decoding, 4 MiB parser bound, canonical request-digest recomputation, evaluator/rule validation, Evidence reference partition checks, and exact typed-envelope projection validation.

## Valid seed

The support crate deterministically builds a valid formal envelope from the pinned REA Evidence v1 fixture and the producer claim evaluator. The workflow regenerates the exact seed from the checked-out commit and places it in the fuzz corpus. The seed is not hand-authored and its bytes are SHA-256 recorded in the run context.

## Bounds

- Fuzzer max input: 1 MiB.
- Formal parser bound: 4 MiB.
- Per-input timeout: 5 seconds.
- RSS limit: 512 MiB.
- PR smoke: 60 seconds.
- Scheduled/manual campaign: 180 seconds.
- Pinned nightly: nightly-2026-11-03.
- Pinned cargo-fuzz: 0.13.2.

## Evidence limits

A no-crash campaign is bounded evidence about the exact commit, compiler, corpus, runner, sanitizer configuration and duration. It is not an authentication proof, semantic oracle, Lean theorem, parser certification or evidence that the provider observation is true. Crashes and sanitizer findings must be minimized and retained as regressions.

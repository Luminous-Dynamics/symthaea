# mycelix-zkp-core cryptographic fuzz targets

These targets fuzz Symthaea's public Dilithium wrapper boundaries. They are intended for controlled local execution and scheduled CI on owned code, not against third-party services.

## Targets

- `fuzz_keypair_import`: feeds arbitrary byte strings at the exact public/secret key lengths into `DilithiumKeypair::from_bytes`, exercising key encodings and the import-time pair-consistency check.
- `fuzz_signature_verification`: feeds correctly sized arbitrary public-key bytes and bounded attached-signature/message bytes into `verify_signature`, exercising parsers and the underlying verifier when parsing succeeds.

The targets cap source input at 8 KiB and bound generated message suffixes. Fuzzing is not a replacement for known-answer vectors, the matching-key positive control, or the mismatched-key regression test. The keypair-import target is intentionally deferred until the importer hardening change is part of its tested base.

## Prerequisites

Install the Rust nightly toolchain, a C/C++ compiler with LLVM sanitizer support, and `cargo-fuzz`:

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
```

See the [cargo-fuzz project](https://github.com/rust-fuzz/cargo-fuzz) and [Rust Fuzz Book setup guide](https://rust-fuzz.github.io/book/cargo-fuzz/setup.html) for supported-platform requirements.

## Run locally

From this directory:

```sh
cargo +nightly fuzz run fuzz_signature_verification -- -max_len=8192 -max_total_time=600
cargo +nightly fuzz run fuzz_authenticated_signature -- -max_len=8192 -max_total_time=600
```

For a quick smoke campaign, use `-max_total_time=60`. Preserve any crash-inducing input and minimize it with `cargo +nightly fuzz tmin TARGET PATH_TO_CRASH`; check a minimized reproducer into the regression suite before claiming a fix.

## Evidence rules

Record the exact Symthaea commit SHA, Rust toolchain, dependency lockfile digest, target, command, elapsed time, execution count, corpus digest, exit status, and any crashes. A queued job is not a pass. A short successful fuzz campaign proves only that no failure was found within the recorded campaign; it does not prove cryptographic security or constant-time behavior.

The underlying current `pqcrypto` dependency family is marked unmaintained in [RustSec RUSTSEC-2026-0164](https://rustsec.org/advisories/RUSTSEC-2026-0164.html). These fuzz targets improve regression detection while migration to a maintained FIPS 204 ML-DSA implementation remains a separate priority; they do not make the legacy provider suitable for production.

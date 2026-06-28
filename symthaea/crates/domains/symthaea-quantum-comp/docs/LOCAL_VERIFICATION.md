# Local Verification

`alpha.6` keeps a small verification script so researchers can run the same smoke checks locally.

Run:

`./scripts/verify-local.sh`

The script performs:

- format check
- all-feature tests
- baseline binding example
- noise sweep example
- comparative report example
- negative control example
- entanglement proxy example
- robustness summary example
- report export example
- audit control example
- experiment matrix example
- paired significance probe example
- local research receipt example
- QASM feature tests

## Caveat

This repository uses dependency-free report helpers and local smoke tests. These checks do not establish quantum advantage, hardware correctness, physical entanglement, or consciousness. They only verify that the crate's local probes remain deterministic and internally coherent.

## Recommended local environment

Use a pinned Rust toolchain, preferably through your Nix flake or `rust-toolchain.toml` in a larger workspace. When integrating into Symthaea, record the flake lock hash, crate version, and git revision in external Mycelix provenance rather than treating the crate's non-cryptographic fingerprint as a security receipt.

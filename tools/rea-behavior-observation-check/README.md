# REA behavior observation profile checker v0

This standalone Rust workspace checks a synthetic REA behavior-observation sidecar profile. It does not introduce a new Evidence identity or verify the referenced Evidence record.

## Trust boundary

- The profile binds exact argv argument bytes, stdin/stdout/stderr captures, termination, runner environment and sandbox-relative filesystem changes.
- Capture hashes and lengths are recomputed from the bytes included in the profile.
- Every Evidence reference is pinned to `unverified_reference` in v0. The checker does not verify the referenced REA Evidence JSON and cannot authenticate the provider or prove that the runner executed the named binary.
- A follow-on integration must invoke the existing REA compatibility core's `verify_evidence_json` on the exact referenced Evidence bytes and compare the returned Evidence ID, raw-file SHA-256 and subject digest before admission.
- The examples are synthetic test vectors. They are not executions of original target binaries.

## Run

```bash
cargo test --manifest-path tools/rea-behavior-observation-check/Cargo.toml --locked
cargo run --manifest-path tools/rea-behavior-observation-check/Cargo.toml --locked --bin rea-behavior-observation-check -- \
  docs/research/rea-behavior-observation-profile-v0.synthetic.json
cargo clippy --manifest-path tools/rea-behavior-observation-check/Cargo.toml --all-targets --locked -- -D warnings
```

The CLI rejects duplicate JSON object keys and trailing data, validates Draft 2020-12, recomputes capture hashes/lengths, checks filesystem path containment and duplicate identities, and ensures assertions agree with complete captured data.

## Claim ceiling

A successful report is `capture_valid_evidence_admission_unverified`. It establishes only internal consistency of the supplied sidecar bytes and capture fields. It is not an authenticated observation, a verified REA Evidence reference, a complete behavior specification, or proof of behavior outside the exact invocation and environment.
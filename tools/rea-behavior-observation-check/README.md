# REA behavior observation profile checker v0

This standalone Rust workspace checks a synthetic REA behavior-observation sidecar profile. It does not introduce a new Evidence identity or verify the referenced Evidence record.

The sidecar records exact argv bytes, stdin/stdout/stderr capture, termination, environment, runner profile and sandbox-relative filesystem changes. It distinguishes complete, partial, unavailable and not-applicable capture states.

**Critical trust boundary:** every v0 Evidence reference is fixed to `unverified_reference`. The checker can recompute hashes of bytes included in the sidecar, but it does not verify the referenced REA Evidence JSON. A follow-on adapter must call the established `verify_evidence_json` on the exact Evidence bytes and compare the returned Evidence ID, raw file SHA-256 and subject digest before admitting the binding.

Run:

```bash
cargo test --manifest-path tools/rea-behavior-observation-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-behavior-observation-check/Cargo.toml --all-targets --locked -- -D warnings
```

All checked-in examples are synthetic. They are not original-program executions and cannot be reported as black-box observations or reconstruction results.

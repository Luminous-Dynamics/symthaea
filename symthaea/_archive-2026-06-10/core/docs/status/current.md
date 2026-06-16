# Current Status

Status date: 2026-06-15

## Support Boundary

The root is a virtual Cargo workspace with 128 canonical members. The default
development surface is `xtask`, the `symthaea` application, and packages under
`crates/core`.

Packages excluded in the root manifest are not covered by the default workspace
quality gate. They are excluded because they are nested workspaces,
hardware/toolchain-specific projects, independently maintained integrations, or
temporary legacy copies awaiting separate validation.

## Verified During The 2026-06-15 Review

- Cargo metadata resolves with the regenerated workspace lockfile.
- All 212 `crates/symthaea-core/tests/*.rs` files appear as Cargo test targets.
- The workspace path audit passes across all canonical members.
- `symthaea-embeddings` model-integrity tests pass with matching and mismatched
  SHA-256 sidecars.
- `cargo audit` reports no known vulnerabilities in the resolved dependency
  graph. Advisory warnings for unmaintained dependencies remain technical debt.
- Holon authentication, model mmap integrity, and WASM artifact loading were
  hardened.

The final full-build result for this review should be taken from CI or the
review's closing report. The workspace includes GPU, native library, network,
robotics, and large-model surfaces that cannot all be represented by one
portable local command.

## Maturity

| Area | Current classification |
| --- | --- |
| HDC and dynamics primitives | Active research library |
| Main cognitive application | Experimental integration platform |
| Consciousness metrics | Research indicators, not proof of consciousness |
| Ethics and safety modules | Experimental safeguards, not certification |
| ZKP work | Research prototypes and benchmark harnesses |
| Robotics and hardware | Hardware-dependent prototypes |
| Clinical/regulatory use | Not validated or approved |

## Required Gates

```bash
python3 scripts/audit_paths.py
cargo fmt --all -- --check
cargo audit
cargo test -p symthaea-core --lib
cargo test -p symthaea --lib
```

Feature-specific changes must add the smallest relevant package check or test.
Unsafe code, FFI, model loading, authentication, cryptography, and actuator
control require focused tests and review.

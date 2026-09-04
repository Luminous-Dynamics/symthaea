# Qualification status

This crate is a draft security boundary. Source presence is not qualification.

Promotion requires the dedicated `IoT Actuation Guard Device Reality` workflow to complete on the exact pull-request head with:

- Rust 1.96 formatting;
- Rust/Cargo 1.94 check, tests, and strict Clippy;
- fixed Ed25519 verification through the exact anchored public key;
- verifier-key lifecycle regressions;
- no dynamic signature-provider, raw `DeviceRuntimeState` input, final permit, JIT, HAL, network listener, process execution, spawned work, or unsafe code;
- no unreviewed sourced-package drift beyond the already-pending `fips204 0.4.6` admission inherited from upstream.

Until then this crate is candidate source only.

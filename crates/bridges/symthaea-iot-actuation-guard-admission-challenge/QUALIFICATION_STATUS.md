# Qualification status

This crate is candidate source only until the dedicated exact-head workflow completes.

Promotion requires:

- Rust 1.96 formatting;
- Rust/Cargo 1.94 check, tests, and strict Clippy;
- the exact persisted-reservation constructor regression;
- guard-owned OS entropy and wall-time issuance;
- no privileged challenge deserialization or caller-selected nonce/time/expiry;
- canonical response and exact signed-attestation-object binding;
- no `DeviceRuntimeState`, semantic acceptance, final/JIT/HAL, networking, process execution, spawned work, or unsafe code;
- no unreviewed sourced-package drift beyond the already-pending upstream `fips204 0.4.6` admission.

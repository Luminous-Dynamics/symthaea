# LQCD Fast-Lane Candidate Lint Identity Evidence

Authority: **independent engineering evidence only**. This evidence identifies the exact hosted Clippy defects for candidate head `5242b44b50df491a71b047f3c144bdfd8af9afc3`; it does **not** establish causal attribution and does **not** authorize a Rust repair.

## Hosted inputs

- workflow run: `34971672214`
- canonical attempt SHA-256: `1548750b2c944fbc07a2ab81724c816f35207b444bf02834869aab2fe709b694`
- diagnostic SHA-256: `fd703feab65cc0cb3ff0361275ce3a3e9a7f258d81381ac079070303b240093b`
- profile: `lqcd-particle-physics-focused-v2`
- recipe semantics SHA-256: `708bb303573d2a934fb806037f43880fa8c73ae53dc60f3081ef1e90bf55bd69`
- Rust: `1.96.0`
- Clippy: `0.1.96 (ac68faa20c 2026-05-25)`
- command: `cargo clippy --locked -p symthaea-particle-physics --all-targets -- -D warnings`
- all required gates except Clippy: PASS
- terminal disposition: `ClippyFailed`

## Normalized lint identities

1. `d1102e02b674c30bc2dc0905383c03edda62711df7a33c531f8309c5943c34c8` — `clippy::needless_range_loop`, `symmetry_groups.rs:190:18`
2. `0fb3b3e338925603cec9456d3bb2c48030992cea26ec2d4c1f385f6549499234` — `clippy::needless_range_loop`, `symmetry_groups.rs:204:18`
3. `8d802af5cd4b9fb5f2a2440064486d187ebfee11664fc2b90825a849ebd7fa92` — `clippy::needless_range_loop`, `symmetry_groups.rs:205:22`
4. `85a23dcb8a11e213f82c7697f81ccd9499ee5e4c5184c9288a48a15079df5c51` — `clippy::needless_range_loop`, `symmetry_groups.rs:206:26`

Ordered identity-set SHA-256:

`487022e6235e25eb2485dd8b35db508e0b9ac933025bcbb4e294a96a3e58e99a`

## Reproduction

Run the checked-in oracle over the frozen hosted input fixtures. The oracle validates the canonical attempt/diagnostic binding, expected profile, exact Clippy command, Rust/Clippy versions, non-Clippy gate PASS state, and then extracts/hashes the four diagnostics deterministically.

## Claim ceiling

`repair_authorized = false`

Disposition remains `AttributionUnknown` until exact-base execution under equivalent verifier/profile/toolchain semantics is compared against this identity set. Structural evidence from #3426 and byte-identical source evidence make a candidate-caused Rust regression implausible, but they do not substitute for the exact-base execution required by #3404.

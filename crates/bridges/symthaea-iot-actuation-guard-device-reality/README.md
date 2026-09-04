# Symthaea IoT Actuation Guard — Device Reality

This crate is a privileged, non-actuating relying-party boundary for post-reservation device reality.

It consumes a challenge-bound `DeviceAttestationResultV1` only after a `SemanticReservationChallengeV1` exists, and verifies it against guard-owned policy plus anti-rollback trust that contains the exact Ed25519 public key bytes.

It deliberately does not accept `DeviceRuntimeState` as an input. `DeviceRuntimeState` is derived only from a successfully verified signed attestation body.

The crate contains no HAL/device I/O, final-permit/JIT construction, network listener, dynamic verifier provider, caller-selected key, or caller-selected clock.

Promotion requires the dedicated Rust 1.94/1.96 qualification workflow to complete on the exact PR head.

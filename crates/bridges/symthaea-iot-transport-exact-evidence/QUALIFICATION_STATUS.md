# Qualification status

Status: **candidate / not yet compiler-qualified**.

Required focused evidence:

- exact canonical receipt and payload bytes are owned by the opaque capsule;
- construction consumes the original `VerifiedTransportEnvelope`; a borrowed transport proof cannot mint multiple continuation capsules;
- receipt body signing digest, payload digest, decoded envelope digest, peer/session identity, opening time and transport-trust head reproduce the consumed opaque `VerifiedTransportEnvelope`;
- canonical substitution/trailing-byte cases fail closed;
- production source performs no signature verification and accepts no trust registry, current time, final permit, JIT lease or HAL handle;
- the capsule is neither `Clone` nor `Serialize`/`Deserialize`;
- Rust 1.94 package check/tests/strict Clippy and Rust 1.96 formatting complete on the exact head;
- any Cargo.lock change is local/path-only; no new registry or Git package is admitted by this tranche.

No current-trust revalidation or physical-authority claim is made by this crate.

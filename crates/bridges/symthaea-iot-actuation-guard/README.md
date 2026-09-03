# Symthaea IoT Actuation Guard v0.1

This crate is the first privileged **process-boundary ingress** for consequential IoT evidence. It is intentionally non-actuating.

## What v0.1 proves

For one Linux Unix-domain connection at a time, the guard can establish that:

1. the caller UID came from kernel peer credentials and is explicitly allowlisted;
2. the evidence-only IPC frame is bounded and canonical;
3. the Xenia receipt and exact physical-effect payload pass the fixed Ed25519 + ML-DSA-65 verifier under guard-owned current transport trust;
4. the physical-interlock report commits to the same envelope, device and transport-trust generation;
5. the interlock observation does not predate authenticated transport and is fresh at guard-local time.

A positive response is deliberately named `IngressValidatedNoActuation`.

## What v0.1 does not prove

It does **not** authenticate the controller/interlock signature, mutate durable device semantic state, create a final actuator permit, perform the complete JIT revocation fence, hold a HAL/device handle, or cause a physical effect.

Those remain later guard-local stages.

## Process isolation contract

The production crate is Linux-only and forbids unsafe code. It has no TCP path, no client-task spawning, no arbitrary process execution, and no caller-selected verifier.

The socket runtime directory must:

- already exist;
- not be a symlink;
- be writable only by its owner;
- be searchable by the provisioned IPC group;
- have the exact configured group.

The socket is forced to mode `0660` and must inherit that group. Filesystem group membership grants only the ability to reach the socket; the exact kernel peer UID allowlist remains the caller identity gate. This permits the unprivileged/cognitive caller and guard to run as different UIDs.

The guard never silently removes a pre-existing socket path. Runtime-directory lifecycle belongs to the service manager.

## Promotion dependencies

Do not call this boundary qualified until its own exact-head Rust 1.94 check/test/strict-Clippy and Rust 1.96 formatting gates pass. Its fixed Xenia hybrid verifier is stacked on the still-pending #402 crypto-dependency/interoperability gate, and cross-repository Xenia receipt compatibility remains separately gated by the Xenia/Symthaea wire and ML-DSA interoperability qualification PRs.

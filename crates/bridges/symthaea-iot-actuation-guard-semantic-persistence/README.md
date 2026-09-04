# IoT Actuation Guard — Semantic Persistence

This crate performs the first durable **semantic safety** transition in the privileged physical-effect path.

It consumes only a crash-durable `PersistedAdmissionReservation` plus fixed-key authenticated `VerifiedAdmissionDeviceReality`, evaluates the existing device semantic policy against that trusted runtime projection, and persists the successor `DeviceSemanticCheckpointV1` with file fsync, atomic rename, directory fsync, canonical read-back, and exact head confirmation.

A successful `PersistedSemanticAcceptance` is **not physical authority**. Controller observation/interlock, final composition, JIT revocation fencing, and HAL/device I/O remain later stages.

The store is single-operation and opens only against an independently retained current `DeviceSemanticHead`; after success the returned new head must be retained externally before another operation is admitted.

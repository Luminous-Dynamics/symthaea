# Symthaea IoT Actuation Guard — Admission Challenge

This crate issues the first post-reservation device-reality challenge.

A production challenge can be created only from an opaque `PersistedAdmissionReservation`. Its nonce and issuance time are generated inside the crate, its expiry is clipped to the original physical-effect deadline, and the privileged challenge type intentionally cannot be deserialized by guard-side Rust code.

The portable device response contains only one canonical signed `DeviceAttestationResultV1`. Parsing proves byte/correlation properties only; it does not authenticate the signature, create `DeviceRuntimeState`, persist semantic acceptance, create an interlock/final permit, or expose HAL authority.

Promotion requires the dedicated `IoT Actuation Guard Admission Challenge` workflow to pass on the exact PR head.
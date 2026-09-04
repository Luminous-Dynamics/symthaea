# IoT Actuation Guard — Post-Semantic Controller

This crate starts the physical-interlock round trip only after semantic safety has reached crash-durable storage.

`PostSemanticControllerChallengeV1` can be issued only from opaque `PersistedSemanticAcceptance`. It uses OS entropy and guard-local wall time and cannot outlive either the authenticated device-reality result or the original physical-effect deadline.

The controller response contains only a canonical `PostReservationInterlockReportV1` plus raw controller evidence. The previously authenticated device appraisal is not retransmitted; the controller statement binds its exact whole-object digest already committed by the challenge.

Decoded output is correlation evidence only. Current controller-key trust, guard-owned interlock policy, final composition, JIT revocation fencing, and HAL/device I/O remain later stages.

# IoT Validated Actuation v0.1

This tranche defines the product-facing type-state boundary for consequential cyber-physical effects.

The lower authority, execution, accounting, checkpoint and durable-runtime crates intentionally expose serializable decision/evidence objects. Those are useful for audit and persistence, but serialized records are not capabilities. Physical I/O must not treat a caller-constructible `Allow` receipt as authority.

The secure path is:

1. Run the real authority + firmware + safety + exact plan/world evaluator.
2. Mint an opaque `ValidatedActuation` that owns the exact evaluated proposal.
3. Consume that token into durable preparation.
4. Burn the device sequence and reserve one use plus exact risk as `OutcomeUnknown`.
5. Persist the combined action-accounting + device-sequence checkpoint.
6. Mint an `ArmedActuationPermit`; it deliberately exposes no command bytes.
7. Revalidate current authority epoch, expiry, negative authority facts, firmware, safety observations and exact safety-world commitment immediately before egress.
8. Subtract only this permit's already-proven reservation from the temporary authority use-state used for revalidation; every other reservation and delegation escrow remains charged.
9. If the device/gateway already reports this sequence or a later one accepted, retain the effect as ambiguous and charged.
10. Only a clean fresh allow mints the affine `ReadyActuationPermit` that an authenticated egress adapter may consume.

Transport authentication is orthogonal. A future Xenia adapter should require both `ReadyActuationPermit` and Xenia authenticated-session evidence. Neither object should imply the other.

A preflight rejection before network I/O may reconcile the action reservation as not dispatched, but the device sequence remains burned. A sequence number is cheap; reusing it after an uncertain crash/recovery path is not worth the ambiguity.

This layer does not eliminate physical time-of-check/time-of-use risk after host preflight. Device-local safety interlocks and, where appropriate, device-side validation of command expiry/sequence/safety policy remain required for hazardous or fast-changing systems.

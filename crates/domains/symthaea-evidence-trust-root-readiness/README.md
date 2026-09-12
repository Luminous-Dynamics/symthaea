# symthaea-evidence-trust-root-readiness

Final trust-root gate over anchored, signer-governed policy readiness.

Readiness is retained only when the current externally anchored policy tip is also represented by the current valid monotonic trust-store state and the exact tip checkpoint has independently verified attestation evidence.

The gate recomputes lower readiness rather than accepting a caller-created readiness report. It also resolves the current trust-store segment across reviewed recovery history.

Checkpoint attestation must have been independently verified no later than the earliest point in the trusted clock-uncertainty interval; late verification cannot retroactively make an earlier interval ready.

This crate never grants physical authority.

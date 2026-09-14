# symthaea-tpm-reference-runtime-continuity

Thin TPM/reference adapter over the generic verifier runtime-continuity theorem.

This crate does not implement another independence algorithm. It binds two opaque `ContinuousVerifierExecution` capabilities to the already current-head-qualified TPM campaign/obligation roles.

The campaign runtime execution must produce the exact candidate evidence digest. The obligation runtime execution must consume a canonical digest of that exact campaign evidence context and produce a canonical digest of the strict `SafetyEvidenceReceipt` that TPM actually uses.

The two verifier executions must come from distinct process instances and exact reviewed runtime policies. Any role, time, input, output, policy, or parent-qualification mismatch fails closed.

The resulting `RuntimeBackedTpmQualification` is deliberately non-serializable and grants no physical authority.

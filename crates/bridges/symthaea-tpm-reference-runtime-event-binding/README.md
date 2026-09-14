# symthaea-tpm-reference-runtime-event-binding

Challenge-bound semantic successor to the first TPM/runtime adapter.

This bridge binds continuous verifier execution to the **actual verifier event boundary** rather than attributing deterministic downstream adapter work to the verifier process:

- campaign runtime output is the exact `TpmReferenceCampaignVerification` digest stored in the candidate;
- obligation runtime output is a canonical digest of the exact `IndependentVerification` event;
- the strict `SafetyEvidenceReceipt` remains a deterministic downstream derivation and is separately content-bound.

Both verifier computations must answer prospective, signed challenge envelopes issued before their computation starts. Challenge authority, subject, verifier identity, runtime-policy digest, nonce, validity window and role are all exact bindings.

The resulting `ChallengeBoundRuntimeTpmQualification` is non-serializable. Reports and receipts remain evidence only and grant no physical authority.

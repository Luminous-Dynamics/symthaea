# symthaea-domain-awareness-evidence-lifecycle

Bridge from typed `CandidateEvidence` into contract-scoped lifecycle evidence.

The flow is deliberately staged:

1. an assurance subsystem emits a typed candidate,
2. an independent verifier produces a `SafetyEvidenceReceipt`,
3. this bridge verifies that the exact safety case contains the candidate's obligation,
4. the verifier/reviewer supplies an explicit applicability interval and deployment/configuration references,
5. the result becomes a `ScopedSafetyEvidenceReceipt` for lifecycle assessment.

The applicability interval cannot start before both the underlying candidate evidence and the independent verification. Evidence that would already be expired at verification time is rejected.

This bridge never changes `ProofObligation::status`, never discharges the safety case, and never grants physical authority.

```bash
cargo test -p symthaea-domain-awareness-evidence-lifecycle
```

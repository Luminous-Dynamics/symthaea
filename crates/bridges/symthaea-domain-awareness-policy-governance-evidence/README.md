# symthaea-domain-awareness-policy-governance-evidence

Candidate-evidence bridge from the adversarial policy-governance crucible into the canonical DomainAwareness safety contract.

A passing crucible can produce candidates for:

- `DA-031` — current policy is the exact signed lineage tip,
- `DA-032` — signer/key authority is externally governed,
- `DA-033` — policy lineage is externally anchored against rollback.

Each candidate requires the specific positive/negative scenarios relevant to that obligation. Missing, duplicated, failed, or absent required scenarios fail closed.

The output remains `CandidateEvidence` only. This crate cannot create verified receipts, discharge obligations, establish readiness, or grant physical authority.

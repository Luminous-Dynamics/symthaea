# symthaea-domain-awareness-evidence

Typed bridge from concrete assurance artifacts into the canonical `DA-xxx` formal-safety obligations.

The bridge deliberately separates three concepts:

1. **assurance artifact** — a perception/model/sensor/recovery result,
2. **candidate evidence** — a typed statement that the artifact is relevant to a particular reviewed obligation,
3. **verified receipt** — an independent verifier binds that candidate to a receipt id, verifier identity, and verification time.

None of these operations automatically changes `ProofObligation::status` or discharges a safety case.

## Current bindings

The first tranche supports candidate evidence for:

- `DA-004` — explicit selective-classification abstention evidence from a passing perception crucible with observed OOD abstention,
- `DA-010` — time/calibration/lineage auditability from a structurally valid `ObservationEnvelope`,
- `DA-012` — model divergence/incompleteness restricting the ODD model-assurance state,
- `DA-016` — a passing perception crucible with no perception-to-authority bypass,
- `DA-017` — passing negative/background stress exposure,
- `DA-019` — an explicitly reviewed recovery procedure referenced by a requalification authorization.

Every candidate also requires a durable evidence reference and a content digest supplied by the caller's evidence pipeline.

## Verification boundary

`CandidateEvidence::verify(...)` can construct a `SafetyEvidenceReceipt`, but it still does not discharge the corresponding obligation. Strict readiness continues to require both:

- independently verified receipt evidence, and
- explicit workflow discharge/review of the obligation.

This prevents a subsystem from proving its own safety merely by emitting a favorable report.

```bash
cargo test -p symthaea-domain-awareness-evidence
```

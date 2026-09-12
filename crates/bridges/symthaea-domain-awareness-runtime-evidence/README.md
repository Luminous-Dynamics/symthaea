# symthaea-domain-awareness-runtime-evidence

Bridge runtime assurance qualification artifacts into typed DomainAwareness candidate evidence.

This crate binds:

- the runtime evidence-lifecycle crucible to DA-024..DA-027
- the evidence-dependency anti-circularity qualification to DA-028

A passing qualification creates **candidate evidence only**. It does not create a verified receipt, discharge an obligation, or make a safety case ready.

The caller must still provide an exact artifact/content digest through `ArtifactBinding`, and the candidate must still pass through the independent verification and scoped-evidence lifecycle before it can participate in strict readiness.

The runtime binder requires the reviewed destructive and recovery scenarios relevant to each obligation to be present and passing. Duplicate/missing scenario identifiers fail closed.

The dependency qualification runs both positive and negative controls: an ordinary acyclic chain must pass, while an explicit cycle and a same-contract readiness back-dependency must be rejected.

```bash
cargo test -p symthaea-domain-awareness-runtime-evidence
```

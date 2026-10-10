# Cyber Crucible diagnostic trace contract v1

This adds the missing process layer to the public calibration corpus: a trace records how a solver moves from bounded observations through competing hypotheses, evidence gaps, a safe next observation, scope, claim maturity, containment proposal, authority check, outcome verification, recovery obligations, residual uncertainty, and an evidence-bearing report.

## Run locally

```sh
python3 scripts/validate_cyber_crucible_diagnostic_trace.py
python3 -m unittest discover -s tests -p 'test_cyber_crucible_diagnostic_trace.py' -v
```

The standard-library validator checks cross-record semantics and scenario lineage. The GitHub Actions workflow also validates the JSON Schema with the pinned Draft 2020-12 implementation. All trace-local evidence references must resolve. Phase ordering is fixed. State-changing observations require real execution, approved authority, and an authorization reference. Executed containment requires real execution, approved authority, bounded scope, a reference, and verified execution evidence. Recovery cannot be marked verified while obligations remain pending or fail. A supported root-cause claim requires a high-confidence supported hypothesis backed by two independently verified evidence records; supported attribution requires two cited, verified `direct_actor_evidence` records.

## Evidence and authority limits

`validation/cyber_crucible_diagnostic_trace_v1.json` is a **public synthetic illustration**, bound by scenario ID, revision, and digest to the DNS/TLS calibration fixture. It performed no external observations or actions. Its fixture evidence is explicitly `unverified`, confidence is uncalibrated, no remediation is executed, recovery obligations remain pending, root cause and attribution remain unsupported, and `production_authority_granted` is always false.

The validator is not a cryptographic receipt verifier. `integrity_status: verified` is a declared field that must be backed by an upstream verifier before real traces can be trusted; this contract does not itself authenticate receipts or establish digest-to-content binding. The public trace is not held-out qualification evidence and does not implement a hidden-oracle store. Oracle separation, independent evaluator identity, signed/verified receipts, and access control remain future work.

## Deliberate non-claims

A schema-valid trace does not prove the observations happened, the model solved an incident, a containment action is safe in a live environment, or a production operator is qualified. The contract makes omissions and internally inconsistent claims machine-detectable; operational truth still depends on separately authenticated evidence and policy-controlled authority.

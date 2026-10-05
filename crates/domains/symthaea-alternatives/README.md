# symthaea-alternatives

Evidence-first industrial alternatives assessment.

This crate is deliberately read-only and decision-support oriented. It does not declare an alternative globally "better", mutate manufacturing records, or execute physical actions.

The model is:

1. Define the function that must be satisfied.
2. Represent candidate material/process/product pathways.
3. Declare an explicit comparison scale (unit + functional/lifecycle scope) for every burden dimension; candidates cannot define their own cohort.
4. Represent functional-performance values as evidence-linked, scoped measurements rather than bare claims.
5. Declare the physical operating envelope explicitly; a candidate must demonstrate coverage of the required ranges.
6. Keep multi-axis burdens separate.
7. Evaluate hard constraints fail-closed.
8. Compute a conservative Pareto frontier without an aggregate sustainability score.
9. Preserve evidence kind, confidence, canonical authority identity, artifact identity/digest, contradictions, validity windows, and derivation lineage.
10. Derive an explicit qualification ceiling from evidence actually linked to every burden dimension and every required functional metric.
11. Emit a deterministic assessment receipt, including the assessment timestamp when supplied.
12. Require reproducible derivation metadata for simulated/derived evidence.
13. Intersect eligibility across every required function while keeping each requirement assessment separate and auditable.
14. Identify a conservative next-measurement target from unresolved uncertainty.

Source independence is authority-scoped: multiple artifacts or rotated issuer keys under one authority do not become independent sources. The identity contract is not an authenticity proof; Mycelix admission/attestation must establish authority control. The design is intended to compose later with Mycelix manufacturing/BOM/routing records and a federated evidence graph, while remaining independent of Holochain versioning. The Pareto frontier is a candidate comparison set, not a recommendation or deployment authorization. Functional performance and operating capabilities use conservative intervals rather than midpoint-based pass/fail. A required operating envelope must be fully covered; partial temperature/pressure/load coverage is unresolved or failed rather than extrapolated. Time-bounded evidence is ignored by timeless assessments and evaluated only when the caller supplies an explicit assessment timestamp.

## Safety boundary

Model output is a hypothesis or assessment artifact, not a manufacturing authorization.

A simulation cannot promote a candidate to field-qualified status. Missing or conflicting evidence remains visible and lowers the qualification ceiling.

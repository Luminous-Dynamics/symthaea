# symthaea-alternatives

Evidence-first industrial alternatives assessment.

This crate is deliberately read-only and decision-support oriented. It does not declare an alternative globally "better", mutate manufacturing records, or execute physical actions.

The model is:

1. Define the function that must be satisfied.
2. Represent candidate material/process/product pathways.
3. Declare an explicit comparison scale (unit + functional/lifecycle scope) for every burden dimension; candidates cannot define their own cohort.
4. Keep multi-axis burdens separate.
5. Evaluate hard constraints fail-closed.
6. Compute a conservative Pareto frontier without an aggregate sustainability score.
7. Preserve evidence kind, confidence, provenance, source identity, and contradictions.
8. Derive an explicit qualification ceiling from evidence actually linked to each burden dimension.
9. Emit a deterministic assessment receipt.
10. Identify a conservative next-measurement target from unresolved uncertainty.

The design is intended to compose later with Mycelix manufacturing/BOM/routing records and a federated evidence graph, while remaining independent of Holochain versioning. The Pareto frontier is a candidate comparison set, not a recommendation or deployment authorization.

## Safety boundary

Model output is a hypothesis or assessment artifact, not a manufacturing authorization.

A simulation cannot promote a candidate to field-qualified status. Missing or conflicting evidence remains visible and lowers the qualification ceiling.

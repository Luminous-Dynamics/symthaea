# CulinaryDecisionBundleV1 conformance fixtures

These files are synthetic conformance inputs, not measured food data.

- `rank-reversal.json`: structurally valid positive fixture. Candidate alpha wins the baseline cost frontier; beta wins after the common seasonal price shock; neither robustly dominates the other.
- `schema-rejects-private-coordinates.json`: expected to fail JSON Schema validation because exact latitude/longitude fields are disallowed by this consumer-facing geography envelope.
- `semantic-rejects-scenario-digest-mismatch.json`: expected to fail semantic validation because the alpha shock metrics are bound to the baseline digest instead of the registered shock digest.
- `semantic-rejects-simulation-relabelled-observation.json`: expected to fail semantic validation because a synthetic input is mislabeled as an observed objective.
- `semantic-rejects-unknown-gate-on-frontier.json`: expected to fail semantic validation because an unresolved hard gate is reported as ranked/on-frontier.

The negative fixtures are intentionally invalid. The current Rust semantic validator is intended to reject the semantic negative controls and the positive/partial cases are intended to pass its semantic checks. The private-coordinate case additionally requires JSON Schema validation because the Rust function is not a full JSON Schema implementation. Tests are authored but have not been compiled or executed.

- `semantic-rejects-false-scenario-frontier.json`: expected to fail semantic validation because the reported observed-basket frontier hides alpha even though its complete objective vector dominates beta in that scenario.
- `semantic-rejects-false-robust-dominance.json`: expected to fail semantic validation because reciprocal robust-dominance claims are made for candidates whose relative cost reverses across scenarios.
- `valid-scenario-frontier-with-other-scenario-missing.json`: expected to pass semantic checks. Alpha is on the baseline frontier but is globally incomplete and ineligible for the robust frontier because its price is missing under the shock scenario. Scenario-local rankability and all-scenario robust rankability are distinct.

- `semantic-rejects-false-rank-instability-counts.json`: expected to fail semantic validation because a claimed ranked-scenario count does not match the objective evidence and recomputed scenario frontiers.

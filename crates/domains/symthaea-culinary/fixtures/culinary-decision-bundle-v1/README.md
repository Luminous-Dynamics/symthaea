# CulinaryDecisionBundleV1 conformance fixtures

These files are synthetic conformance inputs, not measured food data.

- `rank-reversal.json`: structurally valid positive fixture. Candidate alpha wins the baseline cost frontier; beta wins after the common seasonal price shock; neither robustly dominates the other.
- `schema-rejects-private-coordinates.json`: expected to fail JSON Schema validation because exact latitude/longitude fields are disallowed by this consumer-facing geography envelope.
- `semantic-rejects-scenario-digest-mismatch.json`: expected to fail semantic validation because the alpha shock metrics are bound to the baseline digest instead of the registered shock digest.
- `semantic-rejects-simulation-relabelled-observation.json`: expected to fail semantic validation because a synthetic input is mislabeled as an observed objective.
- `semantic-rejects-unknown-gate-on-frontier.json`: expected to fail semantic validation because an unresolved hard gate is reported as ranked/on-frontier.

The negative fixtures are intentionally invalid. Their presence is useful only if a future independent validator asserts the expected rejection reason. They are not evidence that an executable validator already rejects them.

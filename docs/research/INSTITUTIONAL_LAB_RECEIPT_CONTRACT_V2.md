# Institutional Lab Receipt Contract v2

Status: **content-bound preregistered reference contract / MeasurementOnly**

Tracks: #3054, #3095, #3098, #3101

This is a monotonic hardening of the v1 institutional-lab receipt contract. V1 remains preserved as historical evidence. V2 is the intended cross-implementation target for the future Rust institutional-lab receipt layer.

## Why v2 exists

V1 binds each external experiment profile by `id + revision`. That catches ordinary revision changes but still assumes an external registry never changes the content associated with the same id/revision.

V2 removes that hidden assumption by binding the exact content fingerprint into every profile reference.

The governing distinction is:

`same label + same revision != same experiment unless content commitment also matches`

## ProfileRefV2

Every scenario, mechanism, population, adversary, metric-schema, constitutional-policy and RNG profile reference contains exactly:

- non-empty `id`;
- positive integer `revision`;
- lowercase 64-hex `content_sha256`.

`content_sha256` is an experiment-content fingerprint for this independent reference contract. It is not a claim that SHA-256 is the production governance/signature authority for Symthaea or Mycelix.

Malformed, uppercase, short or otherwise non-canonical commitments fail closed.

## Experiment identity

The v2 schema identifier is:

`symthaea-institutional-lab-receipt-v2`

The exact identity binds:

- schema;
- content-bound scenario reference;
- content-bound mechanism reference;
- content-bound population reference;
- content-bound adversary reference;
- content-bound metric-schema reference;
- content-bound constitutional-policy reference;
- content-bound RNG reference;
- unsigned 64-bit seed.

Changing only `content_sha256` while preserving id/revision changes experiment identity.

## Metric-schema self-binding

The full validated metric schema is carried in the receipt.

V2 independently computes the canonical SHA-256 of that full schema and requires equality with:

`experiment_identity.metric_schema.content_sha256`

Therefore a caller cannot mutate metric semantics under the same id/revision while leaving experiment identity unchanged.

A deliberate metric-schema content change is allowed only when the content commitment changes with it, which creates a different experiment identity and receipt commitment.

## Inherited v1 semantics

V2 intentionally reuses and preserves the v1 rules for:

- exact-key validation;
- metric definitions and metric-value coverage;
- finite metric values and bounds;
- `HigherBetter`, `LowerBetter`, `DescriptiveOnly` directions;
- absence of a canonical aggregate governance/alignment/flourishing score;
- constitutional results (`Satisfied`, `Violated`, `NotEvaluated`);
- derived constitutional validity;
- welfare never compensating for constitutional violation;
- completion states;
- deterministic canonical JSON;
- MeasurementOnly authority ceiling.

The v2 oracle runs the complete v1 self-test before its own hardening tests.

## V2-specific controls

The independent oracle establishes at minimum:

1. exact repeated v2 fixture inputs produce byte-identical output;
2. same id/revision plus changed mechanism content commitment changes experiment identity;
3. same id/revision plus changed scenario content commitment changes experiment identity;
4. malformed profile commitment fails closed;
5. uppercase/non-canonical profile commitment fails closed;
6. carried metric-schema mutation without commitment rebinding fails closed;
7. carried metric-schema mutation with exact rebinding creates a different experiment identity and receipt commitment.

## Frozen v2 fixture commitments

Expected values for the checked-in v2 oracle:

- fixture receipt SHA-256: `84a146109b9f75920ceeb42fab5eaf0ee45b5761679f46d99fead821ba2d0920`
- experiment identity SHA-256: `65c39a0d1d6f655f1e604d729ad1cdd0a95861b9627250b6e0ee82536ea676ba`
- carried metric schema SHA-256: `113b17a9c17248eda9dd2d18e078308912efe65ba804db39d7c4405b41f1f89d`

Same-id/revision substitution controls produce distinct identity commitments, including:

- mechanism-content mutation identity: `e0fc885158ccb9258e10974762210650d274e42606ad4f78e182d3a8ceef0bdf`
- scenario-content mutation identity: `39eff712849c7d98ddde32b9daaf9f0823c98c70e4f13e10972800ababaa9e9b`

## Future Rust requirement

The future `symthaea-institutional-lab` implementation should target v2, not v1, for cross-implementation receipt agreement.

The Rust implementation must eventually prove agreement on checked-in positive and negative fixtures rather than simply reproducing the field names.

Even then:

`content-bound receipt agreement != mechanism safety`

and:

`experiment reproducibility != real-world behavioral validity`.

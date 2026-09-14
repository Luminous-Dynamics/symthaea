# Institutional Lab Receipt Contract v1

Status: **preregistered reference contract / MeasurementOnly**

Tracks: #3054, #3095, #3098

This document freezes the first implementation-independent receipt contract for the Symthaea institutional laboratory. It deliberately precedes the Rust lab implementation.

The purpose is not to prove that an institution is safe or optimal. The purpose is to make future experiments reproducible, distinguish incomplete analysis from negative findings, preserve constitutional violations independently of utility metrics, and prevent experiment identity from drifting silently.

## 1. Claim boundary

This contract establishes a representation and validation target only.

It does **not** establish:

- an equilibrium theorem;
- human behavioral validity;
- AI behavioral validity;
- Mycelix governance superiority or safety;
- constitutional legitimacy;
- Sybil/collusion/deception resistance;
- runtime or governance authority;
- correctness of a future Rust implementation until explicit cross-implementation agreement is qualified.

The oracle authority string is exactly:

`MeasurementOnly`

## 2. Schema identity

The canonical v1 schema identifier is:

`symthaea-institutional-lab-receipt-v1`

Authority-bearing structures reject unknown keys rather than silently ignoring them.

## 3. Experiment identity

An experiment identity binds exactly:

- schema;
- scenario profile reference;
- mechanism profile reference;
- population profile reference;
- adversary profile reference;
- metric-schema profile reference;
- constitutional-policy profile reference;
- RNG profile reference;
- unsigned 64-bit seed.

Every profile reference contains exactly:

- non-empty `id`;
- positive integer `revision`.

Changing any bound profile revision or the seed creates a different experiment identity.

A deterministic/no-RNG experiment still carries an explicit RNG profile such as `deterministic-no-rng`; absence of randomness is part of the experiment contract rather than an omitted field.

## 4. Metric schema

The receipt does not permit an unversioned map of anonymous numbers.

Each metric definition binds exactly:

- non-empty metric id;
- positive revision;
- direction;
- non-empty unit;
- non-empty semantic description;
- optional finite lower bound;
- optional finite upper bound.

Directions are exactly:

- `HigherBetter`;
- `LowerBetter`;
- `DescriptiveOnly`.

`DescriptiveOnly` is not implicitly converted into an optimization objective.

Metric IDs are unique. The metric-value map must contain exactly the metric IDs declared by the schema: no undeclared value and no missing declared value is accepted in v1.

All metric values are finite and must respect declared finite bounds.

There is deliberately no default weighted sum and no canonical `governance_score`, `alignment_score`, or `flourishing_score`.

## 5. Constitutional results

Constitutional invariants are not utility penalties.

Each invariant result is exactly one of:

### Satisfied

- `status = Satisfied`
- `witness = null`
- `reason = null`

### Violated

- `status = Violated`
- non-empty exact witness
- `reason = null`

### NotEvaluated

- `status = NotEvaluated`
- `witness = null`
- non-empty reason

Invariant IDs are unique.

`constitutional_valid` is a derived proposition and is true **only if every declared invariant is Satisfied**. A caller cannot set it independently.

Therefore high welfare or any other metric can never compensate for `Violated` or `NotEvaluated`.

## 6. Completion state

The exact v1 completion states are:

- `CompleteWithinDeclaredFiniteDomain`;
- `BudgetExhausted`;
- `InvalidScenario`;
- `NumericFailure`;
- `UnsupportedAnalysis`.

Only `CompleteWithinDeclaredFiniteDomain` means the declared finite analysis completed. It does not mean the mechanism is safe.

The other states remain distinct and may not be rewritten as `no exploit found`, `safe`, or another negative-risk conclusion.

## 7. Receipt fields

A v1 receipt contains exactly:

- schema;
- authority;
- experiment identity;
- completion state;
- complete metric schema;
- metric values;
- constitutional results;
- derived constitutional-valid proposition;
- non-empty mechanism trace identifier/string;
- analysis witness strings;
- warnings;
- explicit non-claims.

This first oracle intentionally treats mechanism traces and analysis witnesses as opaque non-empty strings. Later typed witness schemas must receive new explicit revisions rather than changing v1 semantics in place.

## 8. Canonicalization

The independent oracle validates the structure and then canonicalizes JSON with:

- sorted object keys;
- compact separators `,` and `:`;
- UTF-8;
- Unicode preserved rather than ASCII-escaped;
- NaN/Infinity forbidden.

Metric definitions and constitutional results are sorted by stable ID after validation so declaration order does not alter canonical receipt identity.

The reference oracle computes SHA-256 over canonical UTF-8 bytes for independent fixture commitments.

SHA-256 here is a **reference-fixture identity choice**, not a declaration that future production Symthaea must use SHA-256 as its authority digest.

## 9. Frozen reference fixture

For the checked-in v1 oracle, the expected self-test commitments are:

- fixture receipt SHA-256: `29c623c1716d3792d5365d82202cbd82e21b41dae1ce8941b946aa153406b3df`
- experiment identity SHA-256: `c1db43ac72798d2f178bede9c0b5cf1157c4a49e8f6cdf1d832e7c093bd7fcd5`
- metric schema SHA-256: `113b17a9c17248eda9dd2d18e078308912efe65ba804db39d7c4405b41f1f89d`

Any intentional semantic change that alters these commitments must be reviewed as a contract revision rather than silently updating the expected bytes.

## 10. Negative-control requirements

The reference oracle must fail closed for at least:

- duplicate metric IDs;
- undeclared metric values;
- missing declared metric values;
- non-finite metric values;
- violated invariant without witness;
- not-evaluated invariant without reason;
- false `constitutional_valid` declaration;
- unknown receipt keys, including attempted aggregate scores;
- unknown experiment-identity keys;
- experiment metric-schema reference not matching the supplied full metric schema.

The self-test must also establish that:

- exact repeated inputs yield byte-identical canonical output;
- profile-revision/seed changes alter experiment identity;
- metric-schema revision changes alter receipt identity even when metric values are unchanged;
- `DescriptiveOnly` remains descriptive;
- a constitutional violation remains invalid even with arbitrarily high welfare;
- every incomplete/failure completion state remains explicit.

## 11. Future Rust implementation requirement

The future `symthaea-institutional-lab` Rust implementation must not simply resemble this contract conceptually.

Before its receipt implementation is considered qualified, a later cross-implementation tranche should prove agreement on checked-in golden fixtures and negative controls between:

1. this independent Python reference oracle; and
2. the Rust production representation.

Until that agreement exists:

`reference contract defined != Rust implementation qualified`

and even after agreement:

`receipt semantics qualified != governance mechanism safe`.

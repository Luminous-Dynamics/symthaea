# MATH-REP-001D1 — Staged v2 property evaluator

Status: **binding candidate; no execution result claimed**

Evaluator identity:

`math-equivalence-property-evaluator-v1`

Generator identity:

`math-equivalence-property-q0-v2`

Normalizer identity:

`symthaea-exact-polynomial-term-v2`

Authority:

`MeasurementOnly`

## Purpose

Bind the exact deterministic generator frozen by MATH-REP-001D0 to the exact v2 normalizer candidate while protecting the separate evaluation seed set from routine development execution.

This tranche changes no normalizer semantics, generator configuration, retrieval policy, runtime mathematical memory, HDC representation, proof search, or theorem authority.

## Staged split discipline

Default example execution uses only development seeds:

```bash
cargo run -p symthaea-core --example math_equivalence_property_v2
```

Default example tests execute the development split and do not execute the evaluation split:

```bash
cargo test -p symthaea-core --example math_equivalence_property_v2
```

The evaluation split is an explicitly ignored test and an explicit CLI mode:

```bash
cargo test -p symthaea-core --example math_equivalence_property_v2 -- --ignored evaluation_split_matches_frozen_oracle
cargo run -p symthaea-core --example math_equivalence_property_v2 -- --evaluation
```

Do not run the evaluation split to diagnose/tune v2 and then present the same split as independent confirmation.

## Exact class totals

For either split, the evaluator requires:

```text
SameNormalForm      392
DifferentNormalForm 120
RefusalContract      96
Total               608
```

A pair that unexpectedly refuses normalization is a failure and is counted separately as an unexpected normalization error.

A refusal case passes only if both its disposition (`Unsupported` or `Rejected`) and frozen receipt-reason class match.

## Per-family evidence

The report preserves individual family summaries for all:

- 24 pair families;
- 8 refusal families.

This prevents a high aggregate result from hiding a systematically broken law or safety/refusal family.

The intended interpretation is therefore layered:

```text
all SameNormalForm families pass?
all DifferentNormalForm families pass?
all refusal families pass?
any unexpected normalization errors?
only then consider aggregate completion
```

## Structured report

The example emits exactly one JSON report envelope containing:

- report version;
- evaluator ID;
- generator ID;
- normalizer ID;
- `MeasurementOnly` authority;
- split identity;
- exact seed list used by that execution;
- total and passed cases;
- `all_passed`;
- class summaries;
- per-family summaries.

Schema:

`.github/schemas/math-equivalence-property-report-v1.schema.json`

The schema fixes the three class totals and requires exactly 24 pair-family and 8 refusal-family entries.

The run's exact Git commit/toolchain/workflow identity must still be bound externally by the research/evidence plane; the report does not attempt an impossible self-reference to its own Git commit SHA.

## Failure law

### Development failure

A development failure may be investigated.

Before changing v2, classify it as one of:

- generator/oracle defect;
- evaluator defect;
- v2 implementation defect with unchanged contract;
- v2 semantic/representation defect requiring a new normalizer identity;
- unsupported resource/environment issue.

Preserve the original exact head and failing output.

### Evaluation failure

An evaluation failure is preserved as result evidence.

Do not:

- alter the evaluation seed;
- alter its label;
- remove the family;
- tune v2 on the failure and rerun under the same claim of independent evaluation.

A representation-semantic change requires a new normalizer identity and fresh evaluation lineage.

## Qualification sequence

The correct sequence is:

```text
1. validate D0 manifest consistency
2. compile/test focused v2 regression harness
3. compile D1 evaluator
4. run 608-case development split
5. classify all development failures
6. if representation semantics stay frozen, record a repaired exact head if needed
7. explicitly run the frozen 608-case evaluation split
8. bind JSON report + exact head/toolchain to evidence
```

Because current draft CI may skip heavy jobs and lightweight governance can remain queue-bound, no skipped/queued workflow state counts as qualification.

## What a perfect generated evaluation would mean

A 608/608 evaluation result would establish only that the exact v2 implementation matched this frozen generated representation contract over the frozen generated distribution.

It would **not** establish:

- that arbitrary mathematical equivalence is decided;
- that normalization is a proof;
- that equivalence-aware retrieval is useful;
- that HDC adds value;
- that theorem-search success improves;
- that any theorem is novel or true.

## Next research gate after mechanics

Only after this mechanics layer qualifies should MATH-REP move to retrieval measurement:

```text
S    syntax-only
N    normal-form-only
S+N  syntax + normal-form fusion
```

with equal candidate corpus, context bytes/items, compute, tie-breaking and downstream proof/search budgets.

The stronger later claim must then be tested on independently fixed theorem/premise data rather than only generated polynomial identities.

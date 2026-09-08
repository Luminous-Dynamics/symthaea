# RCA-003b.3 Qualification Contract

Qualification target: `symthaea-rca-shadow-disposition-policy`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
registered disposition policy
        !=
case evaluation
        !=
shadow disposition
        !=
canonical belief/workspace state
        !=
action authority
        !=
self-improvement promotion
```

Additional cardinality theorem:

```text
distinct evidence-root count
        !=
pairwise-independent evidence-item count
```

## Required positive checks

Hosted qualification must establish:

1. exact lower-layer profile bindings for bound case, independent evidence-set witness, relation eligibility, and interpretation lineage;
2. exact proposition scoping;
3. evidence semantics = issued pairwise-independent evidence items;
4. interpretation semantics = pairwise-independent interpretation-root set;
5. separate evidence-item and interpretation-root thresholds for every outcome class;
6. distinct ancestry-root count cannot satisfy an evidence-item threshold;
7. stronger final support/opposition requirements than their tentative forms;
8. diagnostic-only relation-strength semantics;
9. qualified-current-blocker defeater semantics;
10. unknown interpretation independence forces underdetermination;
11. contested mode preserves qualified support and opposition;
12. exact `EXPERIMENT_CONTRACT_SCHEMA_VERSION` + registered RCA experiment-contract digest is policy-bearing;
13. duplicated corpus/seed/metric/evaluator preregistration fields are absent;
14. resource ceilings are policy-bearing and make evidence-item / interpretation-root requirements feasible;
15. complete explicit BLAKE3 policy identity;
16. persistence revalidation and tamper rejection;
17. rustfmt and strict Clippy.

## Required negative checks

Qualification must fail if the crate gains:

- a `ShadowDispositionV1` or equivalent output enum;
- a case/evidence-witness evaluation API;
- case inputs in policy registration;
- `min_pairwise_independent_evidence_roots` in production policy code;
- candidate/module/root-id/pair-edge count voting semantics;
- a rule that uses distinct ancestry-root cardinality as evidence-item cardinality;
- strength sum/average/normalization/probability semantics;
- duplicated preregistration fields that can disagree with the registered experiment contract;
- a deserialization path that skips policy revalidation;
- a path to canonical belief, GWT/workspace, external action, or self-improvement promotion.

## Adversarial policy cases

Tests must include at least:

- all four lower-layer profiles register exactly;
- evidence-set witness profile drift rejects old policy;
- wrong experiment-contract schema fails closed;
- zero topology requirements fail closed;
- `Supported` cannot be weaker than `TentativelySupported`;
- `Opposed` cannot be weaker than `TentativelyOpposed`;
- contestation cannot disable either surviving side;
- evidence-item and interpretation-root semantics are distinct;
- changing a threshold creates a new policy id;
- changing the registered experiment-contract digest creates a new policy id;
- changing resource ceilings creates a new policy id;
- `max_case_items` below the largest independent-evidence-item requirement fails;
- `max_interpretation_pairs` below `N*(N-1)/2` for the largest required interpretation-root set fails;
- serialized policy tampering fails revalidation.

## Evidence tier

A green focused workflow proves the policy crate compiles/tests/lints and that the frozen source-level authority boundaries remain present on that exact commit. It does **not** prove the future disposition algorithm is correct, because no disposition algorithm belongs to this tranche.

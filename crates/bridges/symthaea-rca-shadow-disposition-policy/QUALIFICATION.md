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

## Required positive checks

Hosted qualification must establish:

1. exact lower-layer profile bindings for bound case, relation eligibility, and interpretation lineage;
2. exact proposition scoping;
3. pairwise-independent-set root semantics;
4. separate evidence-root and interpretation-root thresholds for every outcome class;
5. stronger final support/opposition requirements than their tentative forms;
6. diagnostic-only relation-strength semantics;
7. qualified-current-blocker defeater semantics;
8. unknown interpretation independence forces underdetermination;
9. contested mode preserves qualified support and opposition;
10. evaluation/preregistration/corpus/seed/metric/evaluator identity is policy-bearing;
11. resource ceilings are policy-bearing and make the registered root thresholds feasible;
12. complete explicit BLAKE3 policy identity;
13. persistence revalidation and tamper rejection;
14. rustfmt and strict Clippy.

## Required negative checks

Qualification must fail if the crate gains:

- a `ShadowDispositionV1` or equivalent output enum;
- a case-evaluation API;
- case inputs in policy registration;
- candidate/module-count voting semantics;
- pair-edge-count semantics in place of independent root-set cardinality;
- strength summation/average/normalization/probability semantics;
- a deserialization path that skips policy revalidation;
- a path to canonical belief, GWT/workspace, external action, or self-improvement promotion.

## Adversarial policy cases

Tests must include at least:

- lower-layer profile drift rejects old policy;
- zero root requirements fail closed;
- `Supported` cannot be weaker than `TentativelySupported`;
- `Opposed` cannot be weaker than `TentativelyOpposed`;
- contestation cannot disable either surviving side;
- root thresholds are explicitly pairwise-independent-set cardinalities;
- changing a threshold creates a new policy id;
- changing evaluation identity creates a new policy id;
- changing resource ceilings creates a new policy id;
- `max_case_items` below the largest evidence-root requirement fails;
- `max_interpretation_pairs` below `N*(N-1)/2` for the largest required interpretation-root set fails;
- serialized policy tampering fails revalidation.

## Evidence tier

A green focused workflow proves the policy crate compiles/tests/lints and that the frozen source-level authority boundaries remain present on that exact commit. It does **not** prove the future disposition algorithm is correct, because no disposition algorithm belongs to this tranche.
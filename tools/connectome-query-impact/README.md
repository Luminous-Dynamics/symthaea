# connectome-query-impact

A dependency-free Rust interval and matched-world kernel for measuring how a proposed proofreading correction could change pre-registered scientific query endpoints.

## Paradigm shift

A local segmentation edit is not scientifically important merely because a detector is uncertain, a mask is large, or a neuron has high degree. Its importance depends on whether the correction could materially change a *named scientific result*.

For each pre-registered query and admissible structural scenario, provide:
- baseline endpoint interval before the candidate edit;
- hypothetical endpoint interval after the candidate edit;
- metric ID and units;
- a positive materiality threshold in that metric's own units;
- an optional pre-registered decision threshold;
- query-definition digest and structural scenario ID.

The crate returns a vector of query effects. It never blends different query units into a universal biological-impact score.

## Interval arithmetic

For baseline interval (B=[b_L,b_U]) and post-edit interval (A=[a_L,a_U]), the conservative rectangular outer bound on the change is:

[
A-B=[a_L-b_U, a_U-b_L].
]

For effect interval (E=[e_L,e_U]), minimum absolute effect is zero when (0\in E), otherwise the smaller endpoint magnitude; maximum absolute effect is (max(|e_L|,|e_U|)). Both are normalized by the query's pre-registered positive materiality threshold (m_q).

- `materiality_exceeded_possible`: upper normalized bound is at least 1.
- `materiality_exceeded_guaranteed`: lower normalized bound is at least 1.

The rectangular difference is a conservative outer bound that treats any baseline value as potentially pairable with any post-edit value. When both snapshots share latent structural uncertainty, a matched-world effect set can be narrower. The rectangular result is not a sharp causal or biological effect interval.

## Matched paired-world evidence

An optional `PairedEffectBounds` contains:
- a paired effect interval, which must be contained inside the rectangular outer bound;
- an evidence digest for the matched before/after artifact;
- optional `PairedDecisionWorld` rows with unique world IDs and exact endpoint values measured in the same latent graph world.

Each paired endpoint must lie inside its corresponding marginal interval, and each paired difference must lie inside the paired effect interval (with a 1e-12 floating-point tolerance). Duplicate IDs, empty world sets, out-of-range values, out-of-range differences, and paired decisions without a pre-registered threshold fail closed.

For a pre-registered decision threshold, each world is classified before and after the edit:
- every paired world flips -> `Guaranteed`;
- no paired world flips -> `Impossible`;
- some paired worlds flip -> `Possible`.

If paired decision worlds are absent, threshold flips are calculated conservatively from the marginal interval rectangle.

## Decision-flip scope — no unqualified guarantees

Every scenario reports `decision_flip_scope`; the query summary reports `aggregate_decision_flip_scope`.

| Scope | Meaning |
|---|---|
| `MarginalRectangularProduct` | Flip classification over the Cartesian product of marginal before/after intervals. |
| `SuppliedPairedWorldSet` | Classification only over the matched worlds supplied; no completeness claim. |
| `DeclaredFiniteUniverse` | Classification over an explicitly declared finite universe, only after a separate closure receipt and exact world-ID set equality. |
| `Mixed` | The query contains scenarios with different scopes. |
| `NotApplicable` | No decision threshold was pre-registered. |

A paired result is **not a probability**. "Guaranteed" and "impossible" have meaning only over the scope reported. Even a verified finite universe does not mean all biologically possible connectomes; it is exhaustive only relative to the pinned universe manifest and admissibility policy.

### Finite-universe closure receipt

An optional `PairedWorldSetCoverage` is required to upgrade scope from `SuppliedPairedWorldSet` to `DeclaredFiniteUniverse`. It binds:
- canonical universe ID and universe manifest;
- admissibility policy, enumeration input and enumerator implementation digests;
- expected world-ID manifest, expected IDs and closure proof;
- independent verifier identity and verification receipt;
- exact candidate ID, operation ID, sample hash and task route;
- exact query ID, scenario ID, query-definition digest, metric and unit;
- decision threshold and materiality threshold;
- source volume, base graph snapshot, hypothetical result graph snapshot and candidate-patch digest.

The expected world-ID set must be non-empty, unique and exactly equal to the evaluated paired-world IDs. A missing/extra ID, duplicate, invalid digest, unverified closure flag, or mismatch in query/operation/sample/scenario/graph/patch lineage fails closed.

**Trust boundary:** the Rust core checks receipt structure, digest shape, field equality and exact ID-set equality. It does not hash external artifact bytes or prove enumeration completeness. The ingestion adapter must:
1. recompute every manifest and artifact digest from actual bytes;
2. authenticate the verifier identity against a pinned trust policy;
3. validate the closure proof and verify that the declared universe was finite and frozen before query scoring;
4. independently establish that every admissible world was enumerated exactly once and no unresolved frontier remains.

A boolean such as `closure_verified` is an assertion, not a trust anchor. Without the independent verifier and its evidence, the result must remain supplied-world scope.

## Scenario handling

- Every query/scenario row is preserved and sorted deterministically.
- Structural scenarios have no weights and are never averaged.
- A query-level flip is guaranteed only when every supplied scenario guarantees a flip.
- A query-level flip is impossible only when every supplied scenario rules out a flip within its declared scope.
- If scenarios use different scopes, the aggregate scope is `Mixed`; it is never silently upgraded.
- Missing decision thresholds are reported as `NotConfigured`.

## API and validation

- `Interval::new(lower, upper)`: validate a finite ordered interval.
- `Interval::difference(baseline)`: conservative interval-valued after-minus-before effect.
- `analyze_candidate(&CandidateCorrectionInput)`: validate a candidate and return the canonical per-query impact vector.

Every query must be pre-registered. The same query must retain the same definition digest, metric, unit, materiality threshold and decision threshold across scenarios. Query/scenario pairs and all world IDs must be unique.

## Synthetic fixtures

- Input schema: `research/connectomics/connectome-query-impact-input-v0.1.schema.json`.
- Report schema: `research/connectomics/connectome-query-impact-report-v0.1.schema.json`.
- Fixture schema: `research/connectomics/connectome-query-impact-fixture-v0.1.schema.json`.
- Known-answer cases: `research/connectomics/fixtures/connectome-query-impact-known-answer-v0.1.json`.

The 28 cases cover rectangular and paired effect bounds, paired-world impossible/possible/guaranteed flips, finite-universe closure and exact world-ID equality, mixed query scopes, and fail-closed controls for missing/duplicate worlds, invalid verifier digests, missing closure evidence, and replay across a different query, threshold, sample, candidate, operation, scenario, graph snapshot or patch.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-query-impact/Cargo.toml
```

Thirty-six Rust unit tests are authored for interval arithmetic, threshold classifications, materiality bounds, scenario preservation, query/metric consistency, paired effect containment, matched decision flips, world uniqueness and range validation, exact finite-world closure, and receipt replay rejection.

They have **not** been compiled or executed in the current authoring environment.

## Research and integration

- #9984 can consume impact vectors only after a frozen independent review-cost policy maps query consequences to an explicitly declared loss unit. This crate itself does not rank review work.
- #9987 can acquire additional evidence when an outcome interval spans a decision boundary or materiality threshold.
- #9994 can compare the same query endpoint across observed graph history.
- #10018 ensures a committed correction is independently verified and reversible.
- #10028 binds query-impact and stopping receipts into a cross-stage workflow envelope.
- #10054 defines query-specific scientific consequence analysis.
- #10077 defines matched paired-world decision flips.
- #10091 specifies the finite structural-world closure boundary.

## Benchmarks and proposed targets

Synthetic exhaustive track:
- inject known false merges and false splits into fixed graphs;
- enumerate finite admissible structural worlds independently;
- compare rectangular effects and paired effects against exact before/after query values;
- randomly drop, duplicate, mutate and reorder worlds;
- require zero false finite-universe closure declarations under negative controls;
- measure effect-interval coverage/width, paired-world decision-flip accuracy, closure verifier time and memory.

Prospective proofreading track:
- freeze queries, materiality thresholds and structural admissibility before triage;
- compare random, entropy-only, graph-degree, query-impact and hybrid policies at fixed reviewer-minute budgets;
- report query-material error recall, false-priority cost, reviewer minutes, correction reversals, downstream endpoint error and false query-ready certifications;
- cluster-bootstrap by source volume, animal or acquisition lineage.

Proposed stretch targets (not achieved):
- at least +15 absolute points in query-material error recall at equal reviewer time versus entropy-only prioritization;
- at least 25% fewer reviewer minutes at matched query-material recall;
- nominal interval coverage on held-out volumes;
- zero false finite-universe closure declarations and zero false query-ready certifications in synthetic negative controls;
- 100% abstention when expected world IDs, closure proof, verifier identity or lineage binding is missing.

## Industry comparison

The strongest published pipelines make different tradeoffs, so a single balanced-accuracy leaderboard is incomplete:
- NEURD demonstrates domain-rule-based automated merge cleanup at scale and semi-supervised proofreading speedups; its approach can sacrifice recall to improve precision.
- Autoproof uses learned workflow policies and fragment linking, reporting major effort reductions on a Drosophila reconstruction.
- ConnectomeBench2 V2 VLM results show that fine-tuned VLMs can reach specialist-model performance; the full natural-mix merge-error balanced-accuracy result is a near-tie with the EM-only ViT-B, not a universal VLM win.

Luminous differentiator to test: connect local decisions to query consequence, matched structural worlds, exact closure evidence, human-review cost and finite-horizon stopping in one auditable workflow. This is a research hypothesis, not a demonstrated SOTA result.

## Non-claims

This component does not decide whether a proposed edit is anatomically correct, assign biological probabilities, estimate neural causality or certify all biologically possible structures. It measures the sensitivity of declared graph-derived endpoints to supplied before/after intervals and matched worlds, within the explicitly reported scope. A graph consequence is not proof that an edit should be made.

See [SCI-CONNECTOMICS-060](https://github.com/Luminous-Dynamics/symthaea/issues/10054) and [SCI-CONNECTOMICS-063](https://github.com/Luminous-Dynamics/symthaea/issues/10091).
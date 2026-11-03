# Connectome structural-universe adequacy benchmark v0.1

**Status:** protocol and report schema; implementation and execution are not yet qualified.  
**Issue:** [#10217](https://github.com/Luminous-Dynamics/symthaea/issues/10217)  
**Related:** [#10091](https://github.com/Luminous-Dynamics/symthaea/issues/10091), [PR #10137](https://github.com/Luminous-Dynamics/symthaea/pull/10137), [#10054](https://github.com/Luminous-Dynamics/symthaea/issues/10054), [#9891](https://github.com/Luminous-Dynamics/symthaea/issues/9891), [#9960](https://github.com/Luminous-Dynamics/symthaea/issues/9960)

## 1. The claim this benchmark separates

A finite-world verifier can establish that every member of a declared finite universe was enumerated. It cannot establish that the declared universe contains every biologically or scientifically relevant alternative.

This benchmark therefore reports two independent dispositions:

1. **Enumeration closure:** exact coverage of the candidate's declared universe.
2. **Universe adequacy:** coverage of an independently generated reference challenge set, scoped to a frozen scientific query.

A report can be CLOSED_VERIFIED and ADEQUACY_FAILED. That is the central negative control: perfect enumeration of an incomplete list must not produce a scientifically adequate result.

Even an adequacy pass means only "supported within tested challenges." It is not a proof over every possible biological structure.

## 2. Threat model

The evaluator must detect candidate universes that omit:
- an entire structural error family;
- one specific query-flipping world;
- a continuation, orphan-fragment or missing-neurite alternative;
- a root identity/reassignment alternative;
- a synapse endpoint alternative;
- a correlated detector, registration or morphology failure;
- a compound error whose individual components are non-material but whose interaction changes the query.

The benchmark also tests false ambiguity: a candidate universe that introduces alternatives incompatible with the frozen source evidence can make a query appear less stable than warranted.

The reference generator must be independent of the candidate-universe generator. Reusing the same generator, seeds, hidden labels or pruning rules for both sides invalidates the adequacy estimate.

## 3. Evaluator boundary and freeze order

The candidate generator may receive only:
- permitted source observations and their immutable artifact references;
- the declared query-independent candidate-generation protocol;
- the pre-outcome admissibility policy;
- the authorized evidence available to the candidate at decision time.

It must not receive reference-world identities, evaluator-only truth labels, held-out perturbation seeds, or query outcomes used to score adequacy.

The evaluator-only reference challenge set is frozen before candidate output is opened. A trusted freeze receipt is required for claims about pre-outcome separation; a producer-authored timestamp or boolean does not qualify. Until the trust adapter in [#10204](https://github.com/Luminous-Dynamics/symthaea/issues/10204) is implemented and qualified, `oracle_boundary` must not be reported as `EVALUATOR_ONLY_VERIFIED`.

The evaluator records an append-only sequence:
1. freeze query definition, decision rule, source cohort and challenge-generation protocol;
2. freeze the candidate-universe protocol and admissibility policy;
3. freeze reference challenge manifest and hidden truth;
4. run candidate-universe generation without access to reference data/outcomes;
5. enumerate and independently verify the declared candidate universe;
6. evaluate both candidate and reference worlds using the exact same frozen query implementation;
7. compute the adequacy report in the evaluator boundary.

Where the ordering cannot be authenticated, output ADEQUACY_NOT_ESTIMABLE and preserve the evidence gap.

## 4. World identity and comparability

World IDs must be canonical, content-addressed identifiers with domain separation and a versioned canonicalization rule. In addition, every candidate and reference world carries an evaluator-recomputed `canonical_structural_sha256` produced by the same versioned structural canonicalizer. Structural inclusion is based on canonical structural digest equality, not raw world-ID equality: candidate and reference IDs may differ while the represented graph is equivalent.

The evaluator consumes separate candidate/reference structure sidecars conforming to `connectome-universe-structure-artifacts-v0.1.schema.json`. The report binds the SHA-256 of the exact sidecar bytes plus the canonicalizer identity. For the synthetic V1 representation:
- nodes are records with unique `node_id` and canonical JSON `attributes`;
- directed edges are records with unique `edge_id`, existing `source` / `target` node references and canonical JSON `attributes`;
- nodes are sorted by `node_id`, edges by `edge_id`, and object keys use the pinned serde_json ordered-map encoding;
- the per-world digest is SHA-256 over `CONNECTOME-STRUCTURAL-GRAPH-CANONICAL-V1\\0` followed by canonical JSON bytes for the sorted `{nodes, edges}` object.

The canonicalizer must declare node/edge identity, direction, synapse attributes, coordinate-frame normalization and permitted equivalence transformations before any biological use. The initial fixture is a small synthetic directed graph; it does not establish an anatomical mesh or voxel equivalence canonicalizer. A canonical digest is an identity/equivalence key under that declared canonicalizer, not proof of biological truth.

The evaluator must distinguish:
- **structural inclusion:** a reference structural world is equivalent to a candidate world under the declared canonical equivalence;
- **query-state coverage:** the reference world's query decision state is represented somewhere in the candidate universe, even when its exact structure is not included;
- **evidence compatibility:** whether the reference world is compatible with the observations available to the candidate.

An evidence-incompatible reference world is counted and disclosed but does not count as a missed admissible world. Evidence compatibility and query evaluability are independent axes: count COMPATIBLE / INCOMPATIBLE / INDETERMINATE first, then split COMPATIBLE worlds into query-estimable and query-NOT_ESTIMABLE. A COMPATIBLE world whose query cannot be evaluated remains COMPATIBLE and increments the compatible-query-NOT_ESTIMABLE count; it must not be relabelled evidence-INDETERMINATE. An evidence-INDETERMINATE world remains in its own compatibility category regardless of whether a provisional query output happens to exist. Structural inclusion is determined using the common canonical structural digest. Query-state coverage is a weaker, distinct result: a structurally omitted reference world may still have its decision state represented by a different candidate world.

The initial protocol must not assign probabilities to worlds. Challenge-set frequencies are not population prevalence.

## 5. Query-specific metrics

For every frozen query, report counts and denominators before percentages:

- candidate world count;
- reference challenge world count;
- evidence-compatibility partition: compatible / incompatible / indeterminate reference counts (these three categories sum to all reference worlds);
- query-estimability partition within compatible reference worlds: estimable / NOT_ESTIMABLE counts (these two categories sum to compatible reference worlds);
- exact structural inclusion among all evidence-compatible reference worlds, independent of query evaluability;
- query-state coverage among compatible, query-estimable reference worlds;
- reference worlds that flip the baseline query decision;
- omitted query-flipping worlds;
- false-invariance events (all candidate worlds are estimable and have one decision state, while a compatible reference world has another);
- false-ambiguity events (candidate permits a decision state unsupported by compatible reference worlds);
- evidence-indeterminate reference count;
- evidence-compatible but query-NOT_ESTIMABLE reference count;
- candidate-world NOT_ESTIMABLE / abstention count;
- candidate and reference duplicate canonical-structure groups (different world IDs that canonicalize to the same structure).

Report the same quantities per error family and per source cluster. The reference challenge manifest must predeclare `required_error_families`; the report must contain exactly one family summary for every required family, including families with zero support. Missing family summaries are invalid; a required family with zero estimable compatible support forces ADEQUACY_NOT_ESTIMABLE. A pooled average must not conceal an entire omitted family. Where challenge generation is not probability sampled, use the name **challenge-set coverage**, never population recall.

Also report **false ambiguity**: candidate worlds whose query decision state is not represented by any evidence-compatible, estimable reference world. A candidate universe that invents unsupported query outcomes should not appear safer merely because it contains more worlds. Count affected candidate worlds and report the metric per error family where candidate provenance permits attribution.

Do not reduce structural inclusion, query-state coverage, false-invariance, false-ambiguity and reference-support into one scalar.

## 6. Required synthetic challenge families

Start from exactly known small connectomes and controlled structural transformations:
- false merge of distinct neurites;
- false split of one neurite;
- continuation loss / orphan fragment;
- root reassignment;
- branch insertion, deletion or duplication;
- synapse endpoint reassignment;
- crossing / near-contact ambiguity;
- anisotropic coordinate transform / registration error;
- correlated detector or segmentation failure;
- compound perturbations with non-additive query consequences.

Queries should include edge presence, directed path/reachability, in/out degree, motif count, pathway threshold, cell-type-specific connectivity and a small number of pre-registered graph statistics. Each query must bind its metric, units, decision rule, threshold and implementation digest.

## 7. Mandatory baselines

1. Single best reconstruction.
2. Marginal rectangular uncertainty product.
3. Hand-authored finite alternatives.
4. Existing evidence-conditioned candidate generator.
5. Union of independent candidate channels (morphology, detector disagreement, graph consistency and adjudicated human alternatives), preserving provenance.
6. Evaluator-only oracle reference universe, marked unattainable.
7. Deliberately incomplete universe controls.

All methods must use the same source observations, frozen query and cost/evidence budget. Additional reference information cannot leak into a candidate baseline.

## 8. Adversarial known-answer matrix

- Remove one query-flipping world while preserving a valid, fully resealed Cartesian manifest: closure remains true; adequacy must fail.
- Remove an entire structural error family from the candidate axes.
- Combine two individually non-material errors into one material compound world.
- Make all candidate channels share a correlated failure that the independent reference generator exposes.
- Introduce an evidence-incompatible reference world: count it, but do not classify it as an admissible omission.
- Introduce an evidence-indeterminate world: abstain rather than force a yes/no adequacy label.
- Make a candidate world structurally different but query-equivalent: structural inclusion and query-state coverage must differ.
- Add semantically duplicate worlds with different IDs: canonical equivalence detects duplication.
- Apply anisotropic resampling / coordinate-frame changes: physical equivalence must be handled by a declared transform, not a pixel-index heuristic.
- Ensure candidate and evaluator share no hidden seed, target label or query outcome.

## 9. Acceptance gates (targets, not results)

- 100% detection of deliberately omitted query-flipping worlds in the frozen synthetic adversarial suite.
- Zero false query-invariant reports in known-truth controls.
- At least 95% challenge-set coverage for independently adjudicated, query-material alternatives on supported strata; report cluster-aware uncertainty and exact denominators.
- 100% explicit NOT_ESTIMABLE / abstention for unsupported evidence or unauthenticated oracle separation.
- 100% of error-family denominators visible; no blind family hidden by pooling.
- Independent replay emits byte-identical reports for identical pinned inputs.
- Altered query, source artifact, generator, reference challenge, or candidate universe changes the exact input report/sidecar digests or invalidates verification. The verifier emits `source_report_sha256` over the input report bytes; the report does not contain its own digest.
- No probabilities or population-level recall claims without a defensible sampling design.

## 10. Schema and validation contract

The report shape is defined in `connectome-universe-adequacy-report-v0.1.schema.json`; evaluator-only graph sidecars are defined in `connectome-universe-structure-artifacts-v0.1.schema.json`.

The report binds `evaluator_protocol_sha256`, `rules_sha256` and `canonicalizer_sha256` to versioned protocol identifiers; these are not source-binary attestations. The semantic verifier computes the exact input report digest in its output rather than attempting a self-referential report hash.

The current Rust semantic verifier does **not** independently validate the #10137 Cartesian-closure receipt or #10204 freeze-authentication receipt. It preserves the report's closure field as `claimed_enumeration_closure`, emits `worldset_closure_validation=NOT_PERFORMED_BY_THIS_TOOL`, and hard-disables trusted adequacy promotion. It can independently detect that a supplied, byte-pinned reference challenge exposes omitted or invented query outcomes; absent such a counterexample, it returns ADEQUACY_NOT_ESTIMABLE until the external trust adapters are integrated.

JSON Schema checks structural shape only. A Rust semantic validator must additionally enforce:
- unique candidate and reference world IDs;
- canonical world-ID validity and digest recomputation;
- exact equality between `candidate_universe.world_ids` and `candidate_worlds.world_id` set;
- exact count reconciliation between world arrays and `summary`;
- compatible + incompatible + indeterminate = reference count;
- all count numerators are bounded by the appropriate denominators;
- candidate and reference sidecar raw-byte hashes match the report's structure-manifest digests;
- the pinned evaluator canonicalizer recomputes each world structural digest from sidecar graph bytes; duplicate canonical structures with distinct world IDs are counted separately for candidate and reference worlds;
- query evaluation is deterministic for a canonical structure: structurally equivalent worlds with different query states under the same query/code digest invalidate the report;
- `compatible_reference_count + incompatible_reference_count + indeterminate_reference_count = reference_world_count`;
- `compatible_estimable_reference_count + compatible_query_not_estimable_reference_count = compatible_reference_count`;
- `candidate_not_estimable_count` is recomputed separately from both reference partitions;
- `false_ambiguity_count` is recomputed as estimable candidate worlds whose query state is absent from the compatible, estimable reference query-state set;
- `false_invariance_count` is recomputed only when every candidate world is estimable and all candidate worlds share one decision state, then counts compatible reference worlds with a different state;
- family totals reconcile without duplicate family keys, and every predeclared required family has exactly one summary row;
- required families with zero support force ADEQUACY_NOT_ESTIMABLE;
- false-invariance, false-ambiguity and omitted-flipping counts are independently recomputed;
- query-state coverage is calculated only over query-estimable, evidence-compatible reference worlds;
- structural inclusion is computed over all evidence-compatible reference worlds, including those whose query output is NOT_ESTIMABLE;
- ADEQUACY_SUPPORTED_WITHIN_TESTED_CHALLENGES is impossible when oracle separation is unverified, a material compatible world is omitted, or any required challenge family is absent;
- closure and adequacy dispositions remain independent.

## 11. Claim ceiling

A pass establishes only that the candidate universe covered the frozen, independently generated reference challenge set for the specified query and source evidence under the declared canonicalizer. It does not prove:
- all biologically possible reconstructions were generated;
- source EM or segmentation is anatomically perfect;
- any structural alternative is probable;
- the scientific query is causally identified;
- the result generalizes to another volume, species, query or acquisition protocol.

The benchmark is intended to make universe misspecification visible and falsifiable, not to eliminate biological uncertainty.

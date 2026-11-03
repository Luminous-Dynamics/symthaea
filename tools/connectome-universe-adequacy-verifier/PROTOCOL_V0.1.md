# Connectome universe adequacy verifier protocol V0.1

**Binary:** `connectome-universe-adequacy-verifier`  
**Package:** `tools/connectome-universe-adequacy-verifier`  
**Report schema:** `research/connectomics/connectome-universe-adequacy-report-v0.1.schema.json`  
**Graph sidecar schema:** `research/connectomics/connectome-universe-structure-artifacts-v0.1.schema.json`  
**Rules:** `research/connectomics/CONNECTOME_UNIVERSE_ADEQUACY_BENCHMARK_V0.1.md`  
**Canonicalizer:** `tools/connectome-universe-adequacy-verifier/CANONICALIZER_V0.1.md`

## Input boundary

The CLI accepts exactly three local files:
1. the report;
2. the candidate-only structure sidecar;
3. the evaluator-only reference structure sidecar.

Candidate generation must not have access to evaluator-only reference structures, reference error-family outcomes, or query results used to score adequacy. The CLI does not authenticate that separation: that is an external trust responsibility.

## Validation order

1. Strictly parse JSON and reject unknown/missing keys at every schema-controlled object level.
2. Verify report schema version and the pinned evaluator, rules, canonicalizer, report-schema and graph-sidecar-schema SHA-256 identifiers.
3. Verify the exact raw-byte SHA-256 of each candidate/reference graph sidecar.
4. Recompute each world representation digest from the supplied graph records under canonicalizer V1.
5. Verify world IDs are unique; candidate report IDs, declared candidate-universe IDs and candidate sidecar IDs must match exactly. Reference report IDs and reference sidecar IDs must match exactly.
6. Verify candidate/reference digest strings match independently recomputed graph digests.
7. Verify that a canonical representation under the frozen query has no contradictory determinate query states across candidate/reference records.
8. Independently recompute aggregate and per-error-family counts.
9. Check family support, all denominator partitions and every metric's numerator/denominator constraints.
10. Recompute the challenge-scoped adequacy disposition and reject any mismatch with the report.

## Status definitions

The executable returns `status = COUNTERS_RECOMPUTED`, `trusted_promotion = false`, and `source_report_sha256` over the exact report bytes.

The output preserves the input closure value as `claimed_enumeration_closure` and emits `worldset_closure_validation = NOT_PERFORMED_BY_THIS_TOOL`. It never represents the upstream #10137 closure claim as independently verified.

The output adequacy field is named `universe_adequacy_against_supplied_challenge`:
- `ADEQUACY_FAILED`: at least one query-material counterexample is observed in the supplied, byte-pinned reference challenge (omitted query-flipping world, false invariance, or false ambiguity);
- `ADEQUACY_NOT_ESTIMABLE`: no counterexample is observed, but a required family lacks compatible estimable support, evidence compatibility is indeterminate, a compatible query is not estimable, a candidate query is not estimable, or the external closure/freeze receipts have not been independently authenticated;
- `ADEQUACY_SUPPORTED_WITHIN_TESTED_CHALLENGES`: reserved for a future version. V0.1 hard-disables this outcome because it does not consume authenticated #10137 closure evidence and #10204 pre-outcome freeze-authentication evidence.

A counterexample is reported only against the supplied challenge. It does not establish failure for every possible biological alternative. Absence of a counterexample is not a success certificate.

## Two separate uncertainty partitions

Reference evidence compatibility:
`COMPATIBLE + INCOMPATIBLE + INDETERMINATE = reference_world_count`.

Among evidence-compatible references:
`compatible_estimable_reference_count + compatible_query_not_estimable_reference_count = compatible_reference_count`.

A graph can be evidence-compatible while its downstream query output is not estimable. That state remains COMPATIBLE and increments the query-not-estimable counter. It must not be relabelled evidence-INDETERMINATE.

Structural representation inclusion is calculated over all evidence-compatible references, independent of query estimability. Query-state coverage, query flips, omitted query flips and false invariance are calculated only over compatible, query-estimable references.

## Metric definitions

- **Structural inclusion:** compatible reference worlds whose ID-preserving canonical representation digest appears in the candidate set.
- **Query-state coverage:** compatible, query-estimable reference worlds whose decision state appears in at least one estimable candidate world.
- **Query flip:** compatible, query-estimable reference worlds whose decision differs from the frozen baseline decision.
- **Omitted query flip:** query-flipping reference worlds whose canonical representation is absent from the candidate set.
- **False invariance:** when every candidate world is estimable and all have the same decision state, the count of compatible, query-estimable reference worlds with a different decision state.
- **False ambiguity:** count of estimable candidate worlds whose query decision is absent from the set of compatible, query-estimable reference decision states. If there is no compatible, estimable reference state, false ambiguity is not inferable and is zero; adequacy is NOT_ESTIMABLE.
- **Duplicate structure group:** number of groups with more than one world ID sharing a canonical representation digest, calculated separately for candidate and reference worlds.

## Fail-closed conditions

Reject:
- any report/sidecar object with unexpected or missing fields;
- malformed or mismatched SHA-256 values;
- invalid node/edge references or duplicate IDs;
- sidecar raw-byte hash mismatch;
- report/sidecar ID-set mismatch;
- digest mismatch;
- contradictory query state for the same canonical representation;
- missing/duplicate/extra required-family summary rows;
- aggregate or per-family count mismatch;
- compatibility or query-estimability partition mismatch;
- any numerator exceeding its declared eligible denominator;
- an invalid receipt digest type/value.

## External dependencies intentionally not implemented

V0.1 does not:
- independently re-run the #10137 Cartesian enumeration verifier;
- authenticate the #10204 signer, trust-root, revocation policy, timestamp, transparency log or outcome-ordering proof;
- prove the candidate universe contains every biologically relevant alternative;
- verify expert adjudication or evidence-compatibility labels;
- perform biological mesh/voxel graph equivalence;
- estimate population prevalence from a non-probability challenge set;
- authorize graph edits or scientific conclusions.

Until those external verifiers are integrated, `trusted_promotion` remains false.

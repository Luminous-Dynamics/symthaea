# Bidirectional connectome target-inventory fate protocol V0.1

**Research issue:** [#10328](https://github.com/Luminous-Dynamics/symthaea/issues/10328)  
**Correspondence predecessor:** [#10238](https://github.com/Luminous-Dynamics/symthaea/issues/10238), [PR #10313](https://github.com/Luminous-Dynamics/symthaea/pull/10313)  
**Schema:** `connectome-target-fate-inventory-v0.1.schema.json`

## 1. Purpose

The V0.1 correspondence verifier requires complete source-object coverage but allows target-world objects to remain absent from the mapping rows. This is intentional for partial correspondence, but it leaves a silent-omission channel on the target side.

This protocol adds **bidirectional inventory-fate accounting**. Every object in the exact target manifest receives exactly one explicit fate. It does not force a bijection, choose among ambiguous candidates, or infer biological birth, death, synapse formation, or absence from tissue.

The receipt is an accounting artifact. It is not a biological identity certificate.

## 2. Fate vocabulary

Every target object receives exactly one of these statuses:

- `MAPPED`: a non-ambiguous mapping row includes this target object. The fate points to exactly one `mapping_id`.
- `ALTERNATIVE_CANDIDATE`: the target object appears in one candidate mapping row belonging to an explicit `alternative_set_id`. All admissible candidates remain visible; none is selected.
- `TARGET_ONLY_UNRESOLVED`: the target object is present in the target release but has no currently estimable source correspondence. This is neutral bookkeeping. It does not mean the object was biologically created, newly formed, or absent from the source tissue.
- `OUT_OF_SCOPE`: the target object is excluded by an explicit scope policy. The record binds the policy SHA-256 and a non-empty reason code.
- `REJECTED`: the proposed correspondence is contradicted by a checkable constraint. The rejected proposal stays in the audit trail and references its mapping row.

A future, separately authenticated evidence profile may introduce stronger vocabulary such as `NEWLY_OBSERVED`. V0.1 must not infer that claim from missing correspondence.

## 3. Exact inputs and identity

The fate receipt binds:
- target world ID and release ID;
- exact target-manifest raw-byte SHA-256;
- exact correspondence-receipt SHA-256;
- every target object ID and type;
- a fate record and evidence-ID list for each target object;
- independently recomputable total and per-object-type counters.

The semantic verifier at `tools/connectome-target-fate-verifier/` loads the exact target manifest bytes and the correspondence receipt bytes itself. It must not trust copied object IDs, summary counters, or digest strings without recomputation. The included correspondence envelope fixture is a compact test input for the mapping-table interface; production use must pass the full receipt already checked by the upstream correspondence verifier. The fixture's target-manifest digest is the SHA-256 of the exact bytes in `connectome-target-fate-target-world-v0.1.json`.

## 4. Normative invariants

1. The set of target object IDs in fate records equals the set of target object IDs in the pinned target manifest, exactly.
2. Each target object ID occurs once and only once in the fate inventory.
3. Every fate object's type equals the type in the target manifest.
4. Unknown target IDs, omitted target IDs, duplicate target IDs, and wrong object types are rejected.
5. `MAPPED` references one existing mapping row that contains the target ID, and the mapping row is not an unresolved alternative.
6. `ALTERNATIVE_CANDIDATE` references one existing mapping row and the same non-empty alternative-set ID declared by that row. The referenced mapping must contain the target ID. All candidate mappings in an alternative set must share the relation type and exact source-ID set; candidate target sets must be disjoint.
7. A target object may not be both mapped and target-only, both mapped and out-of-scope, or assigned to two alternative sets.
8. `TARGET_ONLY_UNRESOLVED` has no mapping or alternative-set reference and carries an explicit reason and evidence IDs.
9. `OUT_OF_SCOPE` has no mapping reference, carries a scope-policy digest and a reason code, and is counted explicitly. A scope policy cannot silently hide an in-scope object.
10. `REJECTED` references the rejected mapping and has a reason code. Rejection is not deletion.
11. Summary counters are recomputed from records. Per-object-type counts sum to the exact target inventory total.
12. Source-side `UNMAPPED` objects and target-side `TARGET_ONLY_UNRESOLVED` objects remain separate metrics. Neither can be used as a proxy for biological turnover.
13. Successful structural accounting returns only a recomputed accounting result. It never authenticates referenced evidence artifacts, edit transactions, adjudicator identity, signer authority, or physical registration.

## 5. Known-answer fixture

The fixture includes:
- one mapped segment;
- two alternative neuron candidates in one unresolved alternative set;
- one target-only unresolved synapse;
- one explicitly out-of-scope derived artifact.

Expected counts: 5 target objects; 1 mapped; 2 alternative candidates; 1 target-only unresolved; 1 out of scope; 0 rejected. Per-type counts: 1 segment, 2 neurons, 1 synapse, 1 artifact.

The target-only synapse does not claim synaptogenesis. The alternative neuron candidates do not establish neuron identity. The out-of-scope artifact is not silently discarded.

## 6. Adversarial qualification matrix

The semantic implementation and its unit tests are authored in Rust. It must reject:
- missing, duplicate, unknown, or wrong-type target fate records;
- a mapping reference absent from the correspondence receipt;
- a mapping reference that does not include the target object;
- an alternative-set ID mismatch, mixed source sets, overlapping target candidates, or a candidate assigned to multiple sets;
- a target object marked both mapped and target-only / out-of-scope;
- an out-of-scope record missing a policy digest or reason;
- a target-only record that also carries a mapping ID;
- a producer-authored “new neuron” / “new synapse” claim without an authenticated source-absence protocol;
- any summary counter that differs from independently recomputed counts.

Mutation tests must change one field at a time and assert deterministic rejection.

## 7. Qualification targets (proposed, not achieved)

- 100% target inventory coverage on all valid synthetic fixtures.
- 100% rejection of exhaustive missing / duplicate / unknown / contradictory target-fate mutations.
- 100% preservation of all admitted alternatives; no silent winner selection.
- Zero false biological-newness claims on exhaustive synthetic negatives.
- Deterministic replay for identical raw input bytes.
- Runtime and peak memory reported by source objects, target objects, mapping rows, and alternative-set cardinality.
- Later real-data qualification requires independently adjudicated multi-release edits and object-level annotations; synthetic fixtures are contract tests, not evidence of biological validity.

## 8. Claim ceiling

A valid V0.1 fate inventory demonstrates only exhaustive structural accounting against one pinned target manifest. It does not demonstrate that the mapped objects are biologically identical, that target-only objects are biologically new, or that evidence authorities are authentic. Biological claims require separate evidence authentication and independent adjudication.
\n## 9. Current implementation status\n\nThe Rust CLI verifier and nineteen mutation/known-answer unit tests are authored under `tools/connectome-target-fate-verifier/`. They have not been compiled or executed in the current authoring environment because Rust toolchain binaries are unavailable. The exact raw-byte fixture pins are recorded in the known-answer fixture. Do not interpret authored tests as passing tests; exact-head CI remains the required execution evidence.\n
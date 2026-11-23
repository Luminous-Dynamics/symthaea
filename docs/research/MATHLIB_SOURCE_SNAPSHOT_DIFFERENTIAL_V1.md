# Mathlib source-snapshot dependency differential v1

**Status:** two independent Lean-native extraction jobs passed on PR head `cecdce02555c7cbe70385ddc8aabf2092466483d`; a local artifact comparison has been performed. The machine comparator and exact-head comparison workflow are being added.

## Cohorts

| Cohort | Mathlib source | Lean toolchain | Extractor output |
|---|---|---|---|
| Baseline, LeanGraph-compatible vintage | `8a178386ffc0f5fef0b77738bb5449d50efeea95` | `leanprover/lean4:v4.29.0` / `98dc76e3c0a9b856c9b98726b713fb04fab16740` | Artifact `14830891573`, SHA-256 `5f86d3bd152299250c742f35b8a10a967391253781c0fc225bc8e9091b235304` |
| Candidate, later Mathlib snapshot | `6ebcce37335bcb6bca64596e5d57bb3035648030` | `leanprover/lean4:v4.37.0-rc1` / `0b77e12952ad39077d80d8bd6b66d601f225c321` | Artifact `14832226964`, SHA-256 `8f4f59addb1c00be7e8ecffc65b5084c74ad06c1dd60dee094ed59e29439a8fa` |

These are two source/toolchain cohorts, not a controlled Lean-compiler-only ablation. Mathlib source revisions differ too. The four target declarations were manually inspected at both commits; their theorem statements and visible proof expressions are unchanged, but their surrounding elaboration environments differ.

## Exact per-declaration dependency-set comparison

A direct comparison of the actual artifact JSON files gives:

| Declaration | Metric | v4.29 count | v4.37 count | Intersection | v4.29 only | v4.37 only | Jaccard |
|---|---|---:|---:|---:|---:|---:|---:|
| `Finset.card_map` | type refs | 6 | 6 | 6 | 0 | 0 | 1.000 |
|  | proof-value refs | 6 | 6 | 6 | 0 | 0 | 1.000 |
|  | closure | 10 | 10 | 10 | 0 | 0 | 1.000 |
| `Multiset.map_id` | type refs | 4 | 4 | 4 | 0 | 0 | 1.000 |
|  | proof-value refs | 12 | 12 | 12 | 0 | 0 | 1.000 |
|  | closure | 12 | 12 | 12 | 0 | 0 | 1.000 |
| `Nat.choose_pos` | type refs | 8 | 8 | 8 | 0 | 0 | 1.000 |
|  | proof-value refs | 33 | 10 | 9 | 24 | 1 | 0.265 |
|  | closure | 33 | 10 | 9 | 24 | 1 | 0.265 |
| `Set.countable_union` | type refs | 6 | 6 | 4 | 2 | 2 | 0.500 |
|  | proof-value refs | 22 | 32 | 20 | 2 | 12 | 0.588 |
|  | closure | 22 | 32 | 20 | 2 | 12 | 0.588 |

All four declarations have zero unresolved dependency names in both cohorts. In each output, the union of type and proof-value references reconciles exactly with Lean's merged `ConstantInfo` closure.

### Interpretation

1. **Stable structural cases:** `Finset.card_map` and `Multiset.map_id` have byte-for-byte identical dependency JSON in both cohorts. Their source and dependency representation is stable across these two snapshots.
2. **Stable interface, proof implementation drift:** `Nat.choose_pos` preserves all eight type references. Its proof-value set changes from 33 to 10; 9 names are common, 24 occur only in the v4.29 snapshot, and one helper (`Nat.choose_pos._f`) occurs only in v4.37. Its source theorem and visible recursive proof clauses are unchanged in the inspected files. This is evidence that elaborated proof dependencies are environment/toolchain-sensitive even when source text appears stable; it does not isolate whether the cause is the Lean revision, Mathlib revision, or transitive environment.
3. **Elaborated notation/interface drift:** `Set.countable_union` preserves the same source theorem and visible `simp [union_eq_iUnion, and_comm]` expression. The v4.29 type edges include `Set.instUnion` and `Union.union`; the v4.37 type edges include `Set.instMax` and `Max.max`. This reflects a changed elaboration of the set-operation interface in the two environments. The proof-value dependency set also changes (22 to 32); only 20 names are shared.
4. **Why counts alone mislead:** `Nat.choose_pos` has fewer direct references in the newer cohort, whereas `Set.countable_union` has more. Neither count alone says the proof is simpler, more correct, or better understood.

## Research consequence

The dependency graph must carry a source/toolchain cohort identity and declaration-level dependency provenance. A graph cache keyed only by declaration name is unsafe across Mathlib/Lean snapshots. Persist exact Mathlib commit, Lean toolchain/revision, declaration type identity and dependency provenance with every graph. The benchmark should distinguish proposition/type-interface stability; proof-value dependency stability; elaborated notation and instance resolution; generated/private helper references; filtered/expanded external graph edges; and axiom/opaque-value availability.

This justifies a three-way comparison: Symthaea direct elaborated extraction, LeanGraph raw direct edges at its pinned v4.29 source, and TheoremGraph's historical filtered/expanded dataset. Do not compare the current v4.37 output as though it were the same environment.

## Claim boundary

This is a four-declaration source-snapshot differential. It is not a corpus-level estimate, expert assessment of which dependency is mathematically essential, a controlled compiler-only ablation, or a direct LeanGraph raw-export differential. The v4.29 job uses the same Lean toolchain vintage as the pinned LeanGraph dataset and runs Symthaea's direct extractor; it is a comparability cohort, not LeanGraph execution itself.

## Generated-helper boundary analysis

Both pinned Lean extractors now emit nested helper_nodes records for direct proof-value dependencies whose fully qualified Name contains the root theorem's component sequence and has additional suffix components. This preserves exact private/generated helper identity, declaration kind, type and proof-value dependencies, ConstantInfo closure, unresolved names and edge provenance. The helper remains a separate node; its references are not silently flattened into the theorem's direct edge set.

The differential CLI now validates the complete generated-helper worklist per root declaration and reports helper names plus the union of root and recursively discovered helper proof-value dependencies. The extraction uses fully qualified Lean Name components, a visited set and explicit helper-to-helper references; it does not flatten helper edges into the root's direct edge set. This captures generated-helper boundary depth while retaining a typed node graph. Non-generated dependencies remain direct targets; their own general recursive trust dependencies are reported separately in the trust profile.

## Trust-profile interpretation and comparison

Trust profiles are recursively derived from the loaded Lean environment, walking each reachable ConstantInfo type/value/structural dependency. They distinguish kernel axiom dependencies, unsafe declarations, partial definitions, opaque declarations, explicit sorryAx reachability and unresolved names. A complete trust closure means every reachable name resolved under the pinned environment; it does not mean zero axioms. Legitimate Lean axioms such as extensionality or classical choice are disclosed and interpreted as part of the declared foundation, not automatically counted as defects. A reachable sorryAx, unsafe declaration or partial definition is a separate high-salience boundary for expert review.

The comparison report includes set differences for each trust class and exact sorryAx booleans. No claim of proof soundness follows from the trust profile alone; kernel acceptance and theorem-statement fidelity remain separate.

## 10. Source-grounded explanation of the first two drift cases

The dependency differential was traced to exact source declarations in the two pinned Mathlib cohorts:

- In Mathlib v4.29, Mathlib/Data/Set/Defs.lean (blob cd1214aa75db2ab1a70bf41696f4ce7ea8a05581) defines Set.union and installs it through `instance : Union (Set α) := ⟨Set.union⟩` at line 234.
- In Mathlib v4.37-rc1, Mathlib/Data/Set/Defs.lean (blob b394e615f38aee9aabde418a33931c3f29732736) retains the same Set.union definition but installs it through `instance : Max (Set α) := ⟨Set.union⟩` at line 232.

The source theorem Set.countable_union retains the same visible simp [union_eq_iUnion, and_comm] expression in both cohorts. The Union.union / Set.instUnion versus Max.max / Set.instMax type-edge difference is directly explained by a source-level notation/instance-interface change. It is not evidence that the theorem's mathematical claim changed. The direct type-edge Jaccard of 0.5 is a representation-level result requiring this provenance explanation.

For Nat.choose_pos, the v4.29 direct proof-value set has 33 names and v4.37 has 10. The newer root proof value exposes Nat.choose_pos._f as a generated helper. The nested helper worklist recovers most of the old references: direct proof-value Jaccard is 0.265, while helper-expanded proof-value Jaccard is approximately 0.946. This motivates the report field helper_expansion_explains_direct_drift.

### Helper-expanded stability signal

The comparator now emits helper_expansion_explains_direct_drift. It is true only when:

1. The type dependency set is exactly stable.
2. Direct proof-value dependencies have changed and their Jaccard is below 0.5.
3. The union of root and generated-helper proof dependencies has Jaccard at least 0.9.

This is a diagnostic for dependency representation/repackaging, not proof equivalence, proof-plan fidelity or mathematical essentiality. It does not suppress or replace raw direct-edge differences. On this four-theorem cohort it should trigger for Nat.choose_pos and not for Set.countable_union.

## 5. Typed value-edge semantics correction

The initial four-theorem extraction compared type references with all ConstantInfo.value? references but labelled every value edge proof_dependency. That was accurate for the four root theorem declarations, but not for generated helpers: Nat.choose_pos._f is a definition, so its body references are definition dependencies, not theorem proof dependencies.

The extractor now classifies an available value by declaration kind: theorem_proof, definition_body, opaque_value, other_value, or unavailable. Corresponding edge relations are proof_dependency, definition_dependency, opaque_value_dependency, and other_value_dependency. The generic value_dependencies array is used to reconcile Lean's merged ConstantInfo closure; proof_value_dependencies is theorem-only, while definition and opaque references are separately reported.

The differential comparator reconciles closure as type_dependencies union value_dependencies, keeps the theorem-proof metric separate, and includes definition helper bodies in helper expansion. Legacy snapshots without value_dependencies remain loadable, but are explicitly marked legacy_unclassified and use their old value array only for backward-compatible closure reconciliation.

This aligns the extraction representation more closely with LeanGraph-style typed edge classes such as sig, proof, and def. It does not establish exact equivalence with the external LeanGraph raw export; that still requires a pinned, same-revision edge-by-edge differential.
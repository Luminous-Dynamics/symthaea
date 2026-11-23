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

The differential CLI validates helper-node provenance and closure reconciliation, then reports helper names and the union of root plus one-level helper proof-value dependencies. This is a first boundary diagnostic, not yet a recursively expanded transitive closure. If a helper has further generated children, those require their own nodes in a subsequent expansion pass.

## Trust-profile interpretation and comparison

Trust profiles are recursively derived from the loaded Lean environment, walking each reachable ConstantInfo type/value/structural dependency. They distinguish kernel axiom dependencies, unsafe declarations, partial definitions, opaque declarations, explicit sorryAx reachability and unresolved names. A complete trust closure means every reachable name resolved under the pinned environment; it does not mean zero axioms. Legitimate Lean axioms such as extensionality or classical choice are disclosed and interpreted as part of the declared foundation, not automatically counted as defects. A reachable sorryAx, unsafe declaration or partial definition is a separate high-salience boundary for expert review.

The comparison report includes set differences for each trust class and exact sorryAx booleans. No claim of proof soundness follows from the trust profile alone; kernel acceptance and theorem-statement fidelity remain separate.

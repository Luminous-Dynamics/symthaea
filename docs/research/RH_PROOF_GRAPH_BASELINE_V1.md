# RH proof-graph reference baseline v1

**Purpose:** Use the existing independently produced RH proof-dependency graph to select and explain a tractable mathematical slice, rather than rebuilding a 1.48-million-constant graph before we can begin comprehension work.

**Fixture:** `RH_PROOF_GRAPH_BASELINE_V1.json`  
**Parent:** [MATH-COMPREHENSION-001 #13293](https://github.com/Luminous-Dynamics/symthaea/issues/13293)  
**Lean extractor:** [#13295](https://github.com/Luminous-Dynamics/symthaea/issues/13295)

## 1. Upstream provenance and license

The upstream reference is [james-shore/rh-proof-dependencies](https://github.com/james-shore/rh-proof-dependencies), pinned to commit `70fbb2fc21fed8de3c02d9f6df0b6067bf22e879`, dataset v0.2.1 (2026-11-21 14:48 UTC).

Its NOTICE states that the dataset is CC0-1.0, with named code/input exceptions under Apache-2.0. The Alpöge proof source itself is **not** included and is **not** covered by those dataset licenses. This Symthaea fixture includes only small amounts of derived, attributed graph metadata; it does not copy the proof source or the 17-million-line archive.

Pinned upstream file blobs and documentation are listed in the JSON fixture.

## 2. What the graph measures

The dataset is built from Andrew Sutherland's `lean4export` kernel export of five theorem roots. It records direct references from declaration types and values / recursor rules, then takes their transitive closure. It also reports a comparison against three independently built .olean environments with byte-identical references and declaration ranges.

Upstream v0.2.1 reports:

| Measure | Value |
|---|---:|
| Source files | 175,187 |
| Source lines | 17,374,215 |
| Lines inside declarations used by the kernel proof | 10,737,728 (61.8%) |
| Project declarations in closure | 804,912 of 1,061,400 (75.8%) |
| Kernel constants in five-root closure | 1,484,729 |
| Direct kernel-export references | 160,315,873 |
| Mathlib constants used | 204,488 |
| Lean core constants used | 9,796 |

The five roots are `riemannHypothesis`, `riemannZeta_eq_zero_re`, `LFunction_ne_zero_of_half_lt_re`, `LFunction_eq_zero_re`, and `LFunction_eq_zero_re_of_isPrimitive`. The GRH scope is Dirichlet L-functions; this is not a license to generalize to arbitrary L-function families.

## 3. Why this is useful

The upstream `spine.tsv` and dominator table provide an initial graph-prioritization map. One common chain begins:

`allChi`
→ `UpperTheory.FinalTheorems.everyCharacter`
→ `UpperTheory.FinalTheorems.everyCharacter_of_row`
→ `UpperTheory.LiteratureTheorems.every_character_of_literature_of_vWin`
→ `UpperTheory.Lit.every_character_of_theta_integrals_local_matching`
→ `UpperTheory.Lit.nonempty_hyps_definite_side_of_closed_bound`.

The upstream dominator statistics assign very large reachable subgraphs to these nodes. There is also a separate Galois/local-global branch beginning with `UpperTheory.Gal.Int.galoisLocalGlobalStrong_recParameter`.

These are candidate entry points for human mathematical explanation. They are not explanations themselves.

## 4. Critical distinctions

- **Kernel-used declaration** means a constant is referenced directly or transitively by the proof term.
- **Compiler-required declaration** may include notation, tactics, simp lemmas or instances that leave no term reference. It can be needed to compile without appearing in the kernel closure.
- **Dominator** is a graph-theoretic property of the dependency graph, not proof that the mathematics is essential or that a given lemma is the conceptual crux.
- **Statement cone** is the dependencies of the theorem's type; it is not the full proof dependency closure.
- **Textual label** such as “crux” copied from an external blueprint has not been independently adjudicated by a mathematician in the dataset.

The upstream README explicitly notes these limitations. We preserve them rather than laundering graph structure into mathematical meaning.

## 5. Revised Symthaea strategy

Do not try to rebuild the entire RH dependency graph.

1. Use the CC0 v0.2.1 graph as a reference baseline and candidate-slice selector.
2. Implement our Lean-native extractor first on the four small Mathlib pilot declarations in `MATHLIB_PROOF_COMPREHENSION_PILOT_V1.json`.
3. Compare type/value/inductive/recursor edge conventions against a small sample from this external graph's documented format.
4. For the RH task, select one bounded subgraph from the upstream spine and separately review the mathematical role of every proposed “key” node.
5. Only then attempt to map Rubinstein's inequality (M) to precise Lean declarations. The declaration(s) for (M) have not yet been identified; do not guess from subtree size or a theorem label.
6. Produce a human explanation with each formal fact linked to a declaration, and mark any mathematical interpretation requiring independent review.

This makes the 17-million-line RH proof an opportunity for a proof-comprehension pilot without wasting effort duplicating existing graph extraction.

## 6. References

- Upstream dataset: https://github.com/james-shore/rh-proof-dependencies/tree/70fbb2fc21fed8de3c02d9f6df0b6067bf22e879
- Dataset format: https://github.com/james-shore/rh-proof-dependencies/blob/70fbb2fc21fed8de3c02d9f6df0b6067bf22e879/docs/dataset.md
- Extraction method: https://github.com/james-shore/rh-proof-dependencies/blob/70fbb2fc21fed8de3c02d9f6df0b6067bf22e879/docs/method.md
- License notice: https://github.com/james-shore/rh-proof-dependencies/blob/70fbb2fc21fed8de3c02d9f6df0b6067bf22e879/NOTICE
- Lean / Mathlib RH community verification: https://leanprover-community.github.io/blog/posts/riemann-hypothesis/
- MIT time-stamped verifier ledger: https://math.mit.edu/~drew/grh/
- Rubinstein working explanation: https://www.math.uwaterloo.ca/~mrubinst/alpoge_claude_grh/

## 7. Claim boundary

This file records an attributed external metadata baseline. It is not a Symthaea extraction result, not a human proof explanation, not an endorsement of every mathematical role attributed to declarations, and not a new RH proof.

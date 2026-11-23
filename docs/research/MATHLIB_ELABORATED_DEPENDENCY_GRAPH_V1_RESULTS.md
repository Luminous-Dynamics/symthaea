# Mathlib elaborated dependency extraction — first result

**Status:** Source-pinned extraction executed; independent graph-role review pending.

## Execution provenance

- Symthaea branch: research/math-proof-comprehension-001
- Symthaea commit checked out by extractor job: 2504a533ec972de05973056c3f6645bfb0bd9eb6
- Workflow run: [#45162506144](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45162506144)
- Job: Pinned Mathlib elaborated dependency extraction, job ID 137651174603, conclusion **success**
- Artifact: mathlib-elaborated-dependency-graphs, artifact ID 14811692104
- GitHub artifact SHA-256: 67c317889e2a88a0be8351c6b564f7d629218e0ab7489dc20ef63d5dea445b15
- Artifact created: 2026-11-23 11:10:31 UTC

Pinned source:
- Mathlib: leanprover-community/mathlib4@6ebcce37335bcb6bca64596e5d57bb3035648030
- Lean toolchain: leanprover/lean4:v4.37.0-rc1
- Lean source revision: 0b77e12952ad39077d80d8bd6b66d601f225c321

The artifact came from a successful extraction job even though the sibling finite-reference job in that same earlier workflow run had two stale test assertions and failed. Do not describe the entire run as green. The finite-reference assertions were corrected on a later branch head; the independent extraction job's success is the evidence recorded here.

## Results

| Declaration | Type references | Proof-value references | Lean closure | Typed edges emitted | Unresolved |
|---|---:|---:|---:|---:|---:|
| Nat.choose_pos | 8 | 10 | 10 | 18 | 0 |
| Set.countable_union | 6 | 32 | 32 | 38 | 0 |
| Multiset.map_id | 4 | 12 | 12 | 16 | 0 |
| Finset.card_map | 6 | 6 | 10 | 12 | 0 |
| **Total** | **24** | **60** | **64** | **84** | **0** |

For each declaration:

    set(type_dependencies) union set(proof_value_dependencies)
      == set(ConstantInfo.getUsedConstantsAsSet)

This exact set reconciliation held for all four extracted declarations. The emitted edge count preserves the role-labelled references; the closure count is the number of unique names in Lean's merged view.

### Observations

- The type/proof split is not cosmetic. Set.countable_union has 6 type references and 32 proof-value references. A single untyped adjacency list would conceal where most direct references occur.
- Nat.choose_pos proof-value dependencies include Nat.choose_pos._f, a generated/private helper declaration. A source-token parser is not a trustworthy substitute for elaborated environment inspection.
- Multiset.map_id exposes quotient/induction/setoid machinery in its proof-value references.
- Finset.card_map exposes the proof dependency on Multiset.card_map, while its type has the Finset.map and Function.Embedding references.

## Per-file output digests

The output files are preserved in the GitHub Actions artifact above. Their SHA-256 digests are:

| File | SHA-256 |
|---|---|
| proof-comprehension-extract-Finset.card_map.json | 89d8e4e113ff17af1e26f8cec7b0a76feef79561cf959b6f9d5848ce1e2d6c6a |
| proof-comprehension-extract-Multiset.map_id.json | b79f6ace4bb34faccfdd4385ecfa7f5dddaca2221e8c38a9964647f6b2302381 |
| proof-comprehension-extract-Nat.choose_pos.json | 6f63f12939f019112c04323781966bdb4aa804c3484580959628f09d228b4396 |
| proof-comprehension-extract-Set.countable_union.json | 4629920c02d3f94da401e20eaac483a7284662e6963b308f5beef0bf10fb7e69 |

## Claim boundary

Established: the pinned Lean/Mathlib environment emitted machine-readable, provenance-labelled references from elaborated ConstantInfo; type and proof-value references are separated; the merged closure reconciles with their union for these four declarations; all emitted dependency names resolve in the loaded environment.

Not established: external TheoremGraph/LeanGraph edge agreement, semantic role correctness, theorem-plan fidelity, human comprehension, delayed transfer, model CIRR, or generality across arbitrary Lean declarations. The four items are a small pilot, not a representative estimate of Mathlib.

Next: compare this edge-role view with Lake's flat used-constant traversal and a pinned LeanGraph/TheoremGraph slice; then add inductive/constructor/recursor calibration tasks and expert-reviewed role labels.

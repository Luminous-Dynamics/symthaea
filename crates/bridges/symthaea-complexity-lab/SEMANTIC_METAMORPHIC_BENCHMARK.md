# ETH-025: exact semantics-aware metamorphic benchmark

**Status:** Executed finite Python reference experiment committed. Differential execution against Ganak/Symthaea binaries, official MC format validation, Rust implementation and Lean proofs remain open.

## 1. Research hypothesis

Treat a benchmark as a graph of related instances rather than a bag of unrelated formulas.

A node is a canonical formula plus its exact output contract. An edge is a typed transformation with explicit preconditions and a mathematical relation between outputs:
- preserve the result;
- multiply by an exact coefficient;
- force the result to exact zero;
- preserve projected support while changing hidden witness multiplicity.

This can reveal semantic and numerical errors even where a large instance lacks a cheap independent ground-truth oracle.

Fuzzing and unified test support are not new claims. Fichte and Hecher's 2026 review of the MC competitions explicitly recommends unified fuzzing/testing to expose defaults and corner cases, stronger numerical robustness for weighted modes, and more diverse benchmark curation:
- https://doi.org/10.1016/j.artint.2026.104605

ETH-025 specializes this direction to typed, output-contract-aware transformations across MC, WMC, PMC, PWMC and complex AMC.

## 2. Executed finite experiment

Source:
- reference/semantic_metamorphic.py
- source Git blob: b305e44a4cb46aff44a1b6c0a5f91c7146caf31a
- versioned relation contracts: reference/semantic_metamorphic_relations.json

Result:
- reference/semantic_metamorphic_results.json

Execution environment: Python 3.13.5. Fixed seed: 0xE7025.

Observed:
- 100 randomly generated small base instances for each of five output contracts;
- 3,000 exact direct-oracle metamorphic checks;
- 500 checks for each of six transformation families;
- 200 hidden-witness invariance checks across PMC/PWMC;
- 201 base formulas whose exact output was zero;
- 0 failed relations;
- targeted semantic controls all passed;
- runtime for this small reference suite: approximately 0.35 seconds in this environment.

The direct oracle enumerates all assignments on small finite formulas. The coefficient layer uses exact Fraction pairs, including Gaussian-rational Q(i) coefficients for complex AMC. It does not use floating point.

## 3. Transformation contracts

### Syntax normalization

Reorder clauses, reorder literals, duplicate a clause and add a tautology. Expected exact output is unchanged in all five modes.

### Variable bijections

Permute variable IDs, including the projection set and weight keys. Or complement one variable by flipping its literal occurrences and swapping its positive/negative weights. Expected result is unchanged.

### Free-variable extension

Add a fresh unconstrained variable z.

| Output contract | Expected multiplier |
|---|---|
| #SAT | 2 |
| WMC | w(z)+ + w(z)- |
| PMC | 2 if z is projected; otherwise 1 |
| PWMC | w(z)+ + w(z)- if projected; otherwise 1 |
| Complex AMC | exact complex coefficient sum w(z)+ + w(z)- |

For PWMC a hidden variable is existentially quantified. Its multiplicity and any hidden-variable weight must not enter the projected sum.

### Forced-variable extension

Add a fresh variable with the unit clause z.

- #SAT and PMC are unchanged.
- WMC and complex AMC multiply by w(z).
- PWMC multiplies by w(z) only if the new variable is projected; otherwise it is unchanged.

### Contradictory extension

Add unit clauses z and not-z. Every exact counting output must become exactly zero, including signed and complex weighted modes.

### Hidden-witness amplification

Adding a fresh free hidden variable may double the number of full models, but cannot change the set of supported visible assignments. Both PMC and PWMC must remain unchanged.

## 4. Targeted controls

A formula with two satisfying full assignments but one visible projection gives:
- #SAT = 2;
- PMC = 1;
- PWMC = 2/3 for visible literal weights (2/3, 1/3);
- incorrectly multiplying by hidden-witness count gives 4/3.

The harness also checks that a wrong hidden-free-variable doubling prediction is rejected and that exact signed-rational WMC cancellation and exact Q(i) AMC cancellation both return zero.

## 5. Industry comparison and prior-art boundary

The Model Counting Competition already emphasizes exact versus approximate guarantees, numerical robustness and strict treatment of wrong exact answers. Its 2026 tracks include WMC with negative weights, PMC, mixed WMC/PMC/PWMC, and complex-field AMC:
- https://mccompetition.org/2026/mc_description.html

Ganak's 2026 entry documents why exact-zero controls matter: MPFR can report a nonzero residue when the exact result is zero; its MPQ exact runs are distinct from its probabilistic/approximate configurations:
- https://github.com/meelgroup/mccomp-ganak-appmc-2026

The contribution here is not a new solver, proof system or fuzzing concept. It is a common set of explicit output-relation contracts that can be applied across different solver implementations and tracks.

## 6. Next implementation steps

1. Version the relation schema and make transformation preconditions machine-readable.
2. Emit canonical internal instances and official MC-format files without losing weights, modes or projection declarations.
3. Run every base/transformed pair through exact Ganak modes and Symthaea's future binary; preserve raw stdout, return codes, parser profile and instance digests.
4. Add differential classification: parser routing error, transformation-generator error, exact arithmetic error, solver error, or unsupported input.
5. Add delta-debugging to shrink any failed relation to a minimal formula and transformation while preserving the failure.
6. Add deterministic corpus manifests and solver/version/toolchain digests.
7. Expand from small direct-oracle tests to structured public MC2026 instances and paired transformations; keep exact and approximate/PAC modes separate.

## 7. Benchmarks and acceptance

Minimum:
- 100% exact relation agreement on frozen small direct-oracle cases;
- 100% correct output-contract routing under the selected format profile;
- no wrong exact result accepted;
- exact zero preserved as zero;
- 0 hidden witness multiplications in PMC/PWMC;
- all failure cases minimized and retained as regressions.

Competitive:
- run generated transformations through Ganak exact MPZ/MPQ modes and Symthaea;
- report parser/semantic failures separately from runtime failures;
- measure transformations per second, coverage of mode/weight/projection combinations, bugs found, minimization ratio, false-alarm rate, and regression re-detection;
- require at least one independently reproduced solver bug or a measurable increase in robustness coverage before claiming practical advantage.

An aspirational goal is a reusable semantic test graph that can qualify any implementation of one of these output contracts without knowing its internal algorithm.

## 8. Claim ceiling

The executed result establishes only finite exact relations on small instances using a direct reference oracle. It is not a differential run of Ganak or Symthaea binaries, an official competition parser qualification, a Rust implementation, a Lean proof, a performance comparison, or a better-than-SOTA result.

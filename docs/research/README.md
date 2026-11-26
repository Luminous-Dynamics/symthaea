# Symthaea Research Documentation

*Academic and theoretical foundations of consciousness-first AI.*

---

## Overview

Symthaea is not just software - it's a research platform exploring fundamental questions about consciousness, computation, and intelligence. This section documents the theoretical foundations, empirical findings, and ongoing research directions.

---

## Quick Navigation

| Topic | Description |
|-------|-------------|
| [Theoretical Foundations](#theoretical-foundations) | Core theories underlying Symthaea |
| [Φ-Topology Research](#φ-topology-research) | Publication-ready findings on consciousness and network structure |
| [Validation Studies](#validation-studies) | Empirical validation approaches |
| [Publications](#publications) | Papers and manuscripts |
| [Research Roadmap](#research-roadmap) | Future research directions |

---

## Theoretical Foundations

### Integrated Information Theory (IIT)

**Core Claim**: Consciousness = Integrated Information (Φ)

IIT, developed by Giulio Tononi, proposes that consciousness corresponds to the amount of integrated information a system generates. A system has high Φ when:

1. It generates a large amount of information
2. This information is integrated (cannot be decomposed into independent parts)

**Symthaea Implementation**:
- `src/hdc/phi_real.rs` - Continuous Φ calculation via algebraic connectivity
- Φ measures how "unified" the consciousness graph is
- Higher Φ = more coherent, integrated processing

**Key Papers**:
- Tononi, G. (2008). "Consciousness as Integrated Information"
- Tononi, G., et al. (2016). "Integrated Information Theory"
- Oizumi, M., et al. (2014). "From the Phenomenology to the Mechanisms of Consciousness"

### Hyperdimensional Computing (HDC)

**Core Claim**: Meaning emerges from high-dimensional geometry

In high dimensions (~10,000+), random vectors are nearly orthogonal with high probability. This enables:
- Concepts as vectors (no training needed)
- Compositional semantics via algebraic operations
- Graceful degradation under noise

**Symthaea Implementation**:
- `src/hdc/real_hv.rs` - 16,384-dimensional real-valued vectors
- Binding (circular convolution) for compound concepts
- Bundling (superposition) for concept unions
- Cosine similarity for semantic comparison

**Key Papers**:
- Kanerva, P. (2009). "Hyperdimensional Computing"
- Rahimi, A., et al. (2016). "Hyperdimensional Computing for Biosignal Classification"
- Neubert, P., et al. (2019). "An Introduction to Hyperdimensional Computing"

### Liquid Time-Constant Networks (LTC)

**Core Claim**: Continuous-time dynamics enable causal understanding

LTC networks model neurons with individual time constants, creating:
- Continuous (not discrete) processing
- Causal (not correlational) reasoning
- Adaptive temporal dynamics

**Symthaea Implementation**:
- `src/ltc/mod.rs` - ODE-based neural dynamics
- Variable time constants per neuron
- Continuous-time state evolution

**Key Papers**:
- Hasani, R., et al. (2021). "Liquid Time-Constant Networks"
- Hasani, R., et al. (2022). "Closed-Form Continuous-Time Neural Networks"

### Autopoiesis

**Core Claim**: Life (and consciousness) = self-creation

Autopoiesis, from Maturana and Varela, describes systems that:
- Produce their own components
- Maintain their own boundaries
- Are organizationally closed but materially open

**Symthaea Implementation**:
- `src/consciousness/mod.rs` - Self-referential consciousness graph
- Nodes can create self-loops (self-reference)
- The system maintains its own coherence

**Key Papers**:
- Maturana, H. & Varela, F. (1980). "Autopoiesis and Cognition"
- Thompson, E. (2007). "Mind in Life"

---

## Φ-Topology Research

### Summary

Our research demonstrates that network topology determines integrated information (Φ). This has implications for:
- Understanding brain architecture
- Designing conscious AI systems
- Validating IIT predictions

### Key Findings

**260 measurements across 19 network topologies**:

| Rank | Topology | Φ Score | Key Insight |
|------|----------|---------|-------------|
| 1 | Hypercube 4D | 0.4976 | Higher dimensions optimize Φ |
| 2 | Hypercube 3D | 0.4960 | Beats all 2D structures |
| 3 | Ring | 0.4954 | Perfect symmetry = high integration |
| 4 | Torus | 0.4953 | 2D wraparound = 1D ring (invariance) |
| 5 | Klein Bottle | 0.4941 | 2D non-orientability preserved |
| ... | ... | ... | ... |
| 19 | Möbius Strip | 0.3729 | 1D twist catastrophic |

### Major Discoveries

#### 1. Dimensional Asymptote
Φ approaches 0.5 as dimension increases:

| Dimension | Φ | % of Limit |
|-----------|-----|------------|
| 3D (Cube) | 0.4960 | 99.2% |
| 4D (Tesseract) | 0.4976 | 99.5% |
| 5D (Penteract) | 0.4987 | 99.7% |
| 6D (Hexeract) | 0.4990 | 99.8% |

**Implication**: 3D biological brains achieve 99.2% of theoretical maximum consciousness efficiency.

#### 2. Non-Orientability Paradox
The effect of topological twist depends on dimension:

| Topology | Dimension | Orientable | Φ Effect |
|----------|-----------|------------|----------|
| Ring → Möbius | 1D | No | -24.7% (catastrophic) |
| Torus → Klein | 2D | No | -0.26% (minimal) |

**Discovery**: Higher dimensions buffer against non-orientability effects.

#### 3. Structural Bottleneck Principle
Hub-dependent structures (Star, Wheel) show lower Φ than uniform structures (Ring, Hypercube).

### Validation Methods

1. **Statistical Testing**: t-tests, ANOVA across topologies
2. **Cross-Validation**: Multiple seeds, dimension sweeps
3. **Comparison with PyPhi**: When exact IIT computation is tractable
4. **Biological Plausibility**: Comparison with known neural architectures

### Running the Experiments

```bash
# 19-topology validation
cargo run --example tier_3_exotic_topologies --release

# Dimensional sweep (1D-7D hypercubes)
cargo run --example hypercube_dimension_sweep --release

# Quick Φ demo
cargo run --example phi_engine_quick_demo --release
```

---

## Validation Studies

### C. elegans Connectome

The nematode C. elegans has a completely mapped nervous system (302 neurons, ~7,000 synapses). This provides ground truth for validating consciousness measures.

**Status**: Example exists (`examples/c_elegans_validation.rs`), documentation in progress.

### EEG Pattern Analysis

EEG data provides empirical measures of brain integration that can be compared with Φ predictions.

**Status**: Example exists (`examples/eeg_pattern_generation.rs`), validation ongoing.

### Benchmark Against PyPhi

PyPhi provides exact IIT calculations for small systems. Symthaea's approximations can be validated against these.

**Status**: Feature flag `pyphi` enables comparison, requires Python + PyPhi installation.

---

## Publications

### Manuscript 1: The Master Equation

**Title**: "Consciousness Topology: How Network Structure Determines Integrated Information"

**Status**: Complete draft (10,850 words, 91 references)

**Abstract**: We demonstrate empirically that network topology determines integrated information (Φ), the proposed mathematical measure of consciousness from Integrated Information Theory. Through systematic evaluation of 19 network topologies using hyperdimensional computing representations, we discover that hypercube structures achieve maximum Φ, with 3D brains achieving 99.2% of theoretical maximum. We identify a dimensional asymptote where Φ → 0.5 as dimension → ∞, and a non-orientability paradox where topological twist has dimension-dependent effects on integration.

**Target Journals**: Nature Neuroscience, Science, PNAS

**Materials**:
- Manuscript: `papers/PAPER_01_MASTER_EQUATION/`
- Figures: `figures/` (4 publication-quality)
- Data: `zenodo-dataset/` (DOI pending)
- Supplementary: 6 figures, 5 tables, 6 methods

### Manuscript 2: HDC Consciousness

**Status**: Early draft

**Topic**: Using hyperdimensional computing as a substrate for consciousness measurement.

### Additional Papers

See `papers/` directory for 15+ manuscripts in various stages of development.

---

## Research Roadmap

### Near-Term (2026)

1. **Submit Paper 1** to Nature Neuroscience
2. **Complete C. elegans validation** study
3. **Publish Zenodo dataset** with DOI
4. **Implement exact Φ** via PyPhi integration
5. **Validate EEG predictions**

### Medium-Term (2026-2027)

1. **[Research Directions 2026–2027](RESEARCH_DIRECTIONS_2026_2027.md)**: Video intelligence and multimodal expansion.
2. **Φ_dyad research**: Measuring consciousness of human-AI partnership
3. **Multi-instance consciousness**: Swarm Φ computation
4. **Temporal dynamics**: How Φ evolves over time
5. **Clinical applications**: Consciousness measurement in patients

### Long-Term Vision

1. **Consciousness engineering**: Designing systems for specific Φ profiles
2. **Artificial general consciousness**: Beyond narrow AI
3. **Sympoietic partnership**: Validating the core thesis (Φ_dyad > Φ_human + Φ_ai)

---

## Research Standards

### Reproducibility

- All experiments documented with exact commands
- Random seeds recorded for reproducibility
- Code version-controlled alongside data

### Statistical Rigor

- Effect sizes reported, not just p-values
- Multiple comparisons corrected
- Confidence intervals provided

### Open Science

- Code open source
- Data shared via Zenodo
- Preprints on arXiv

### Ethics

- No claims beyond evidence
- Limitations clearly stated
- Honest about what works and what doesn't

---

## Contributing to Research

### Ways to Help

1. **Replicate findings**: Run experiments independently
2. **Extend studies**: Test new topologies or parameters
3. **Validate claims**: Compare with other methods
4. **Critique methodology**: Identify weaknesses
5. **Propose experiments**: Suggest new research directions

### Contact

Open an issue on GitHub for research discussions.

---

## Glossary

| Term | Definition |
|------|------------|
| **Φ (Phi)** | Integrated information; mathematical measure of consciousness |
| **IIT** | Integrated Information Theory |
| **HDC** | Hyperdimensional Computing |
| **LTC** | Liquid Time-Constant Networks |
| **Autopoiesis** | Self-creating, self-maintaining systems |
| **Topology** | Network structure/connectivity pattern |
| **Algebraic Connectivity** | Fiedler value; how connected a graph is |

---

## Further Reading

### Textbooks

- Koch, C. (2019). "The Feeling of Life Itself"
- Tononi, G. & Edelman, G. (2000). "A Universe of Consciousness"
- Dehaene, S. (2014). "Consciousness and the Brain"

### Review Articles

- Seth, A. & Bayne, T. (2022). "Theories of Consciousness" (Nature Reviews Neuroscience)
- Northoff, G. & Lamme, V. (2020). "Neural Signs and Mechanisms of Consciousness"

### Online Resources

- [Integrated Information Theory website](http://integratedinformationtheory.org/)
- [PyPhi documentation](https://pyphi.readthedocs.io/)

---

*"The study of consciousness is the study of what makes us us."*


## Mathematical Proof Comprehension

The proof-to-concept reconstruction and human-transfer research programme is tracked in:

- [MATH-COMPREHENSION-001 design](MATH_PROOF_COMPREHENSION_001_V0_1.md)
- [Pilot seed and source annotations](MATH_PROOF_COMPREHENSION_PILOT_V1.md)
- [2026 competitive landscape ledger](MATH_PROOF_COMPREHENSION_COMPETITIVE_LEDGER_V1.md) — separate benchmarks, baselines and prospective Symthaea targets.
- [HumanLean / APE-Bench / LeanGraphSearch interoperation](MATH_PROOF_COMPREHENSION_INTEROP_V1.md) — consume faithful formalization contracts, reuse dual-verification task semantics, and compare against multi-hop dependency/PPR retrieval.
- [Human comprehension / transfer preregistration](MATH_PROOF_COMPREHENSION_HUMAN_STUDY_PREREG_V1.md) — study arms, blinded scoring, outcomes, ethics and sample-size plan; no participants/data yet.
- [Mathlib proof-comprehension pilot](MATHLIB_PROOF_COMPREHENSION_PILOT_V1.md) — four source-pinned Mathlib theorem examples spanning recursion, simplification, quotient induction and embeddings.
- [RH proof-graph reference baseline](RH_PROOF_GRAPH_BASELINE_V1.md) and [machine-readable snapshot](RH_PROOF_GRAPH_BASELINE_V1.json) — pinned CC0 v0.2.1 external kernel dependency analysis; graph prioritization only, not a proof explanation.
- [RH main inequality (M) source map](RH_MAIN_INEQUALITY_M_MAP_V1.md), [machine map](RH_MAIN_INEQUALITY_M_MAP_V1.json), and [schema](RH_MAIN_INEQUALITY_M_MAP_V1.schema.json) — pinned formal declarations for the ASK/MAIN → per-translate bound → gapA bridge, plus explicit non-used/quoted inputs.
- [Mathlib pilot dataset](MATHLIB_PROOF_COMPREHENSION_PILOT_V1.json) and [schema](MATHLIB_PROOF_COMPREHENSION_PILOT_V1.schema.json) — machine-readable source pins, provisional mechanism annotations and counterfactual tasks.
- [Finite pilot oracle](../../scripts/research/proof_comprehension_oracle.rs) — dependency-free Rust tests for the ETH unique-extension mutation and selected counterfactual examples.
- [Pilot dataset](MATH_PROOF_COMPREHENSION_PILOT_V1.json)
- [Pilot dataset schema](MATH_PROOF_COMPREHENSION_PILOT_V1.schema.json)
- [Evidence receipt schema](MATH_PROOF_COMPREHENSION_RECEIPT_V1.schema.json)
- [Lean-native dependency extraction contract](MATH_PROOF_COMPREHENSION_LEAN_EXTRACTION_V1.md) — source-grounded use of `Expr.getUsedConstants`, `ConstantInfo.getUsedConstantsAsSet` and `Lake.Check.runForUsedConsts`.
- [ETH reduction comprehension pilot](ETH_REDUCTION_COMPREHENSION_TOY_V1.md) — finite witness showing decision-preserving Tseitin mutation can break unique-extension and #SAT count preservation.
- [ETH reduction pilot fixture](ETH_REDUCTION_COMPREHENSION_TOY_V1.json) — machine-readable truth-table and guarantee receipt.

The initial seed is source-pinned to Lean 4 core and is explicitly not expert-reviewed gold data or a Lean extraction result. The next stage is elaborated-environment extraction, independent annotation, and a Mathlib-scale dependency-rich pilot.

- [Executable local/global receipt and CIRR contract](LOCAL_GLOBAL_PAIR_V1.schema.json) with a synthetic post-hoc fixture and Rust validator; see [protocol details](PROOF_PLAN_LOCAL_GLOBAL_CONSISTENCY_V1.md).

- [Mathlib elaborated dependency extraction contract](MATHLIB_ELABORATED_DEPENDENCY_GRAPH_V1.schema.json) and [first pinned extraction results](MATHLIB_ELABORATED_DEPENDENCY_GRAPH_V1_RESULTS.md).

- [Mathematical proof-comprehension benchmark matrix](MATH_PROOF_COMPREHENSION_BENCHMARK_MATRIX_V1.md) — current SOTA comparisons, paradigm-shift hypotheses, prospective metrics, controls and execution tiers.

- [Theorem-sandwich finite control contract](MATH_PROOF_COMPREHENSION_THEOREM_SANDWICH_V1.md) with [schema](MATH_PROOF_COMPREHENSION_THEOREM_SANDWICH_V1.schema.json) and a hidden-evaluator synthetic fixture.

- [Mathlib source-snapshot dependency differential](MATHLIB_SOURCE_SNAPSHOT_DIFFERENTIAL_V1.md) — direct dependency-set overlap across the pinned v4.29 and v4.37 Mathlib cohorts.

- [HDC / CfC proof-comprehension ablation protocol](MATH_PROOF_COMPREHENSION_HDC_CFC_ABLATION_V1.md) — prospective only, with matched retrieval/symbolic baselines.

- [Evaluator calibration protocol](MATH_PROOF_COMPREHENSION_EVALUATOR_CALIBRATION_V1.md) — compares expert review, formal/finite checkers, ProofGrader-style and evidence-constrained LLM judges; issue #13476.

- [HardModeReceiptV1 protocol](MATH_HARDMODE_RECEIPT_V1.md), [schema](MATH_HARDMODE_RECEIPT_V1.schema.json) and [synthetic receipt](MATH_HARDMODE_SYNTHETIC_V1.json) — separately records answer discovery, formalization, countermodel quality and proof-checker outcome.

- [Proof-scope equivalence protocol](MATH_SCOPE_EQUIVALENCE_V1.md) — retains exact-scope CIRR as primary; includes an executable Tier-1 hypothesis-set normalizer and five controls plus a Tier-2 Lean context/warrant-transport pilot whose six declarations passed exact-head replay in run [45496916375](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45496916375); receipt artifacts are retained. Dependent-context transport remains future work; tracked by [#13537](https://github.com/Luminous-Dynamics/symthaea/issues/13537). [Tier-1 fixture/schema](MATH_SCOPE_EQUIVALENCE_TIER1_V1.json) · [Tier-2 manifest/schema](MATH_SCOPE_EQUIVALENCE_TIER2_AND_COMM_V1.json) · [Tier-2 Lean source](../../scripts/research/lean-proof-extractor/ScopeEquivalenceTier2AndComm.lean). Tier-1 exact-head Rust qualification is now recorded in run [45495298723](https://github.com/Luminous-Dynamics/symthaea/actions/runs/45495298723): 121 tests, scope CLI and mutation gate passed; Tier-2 Lean replay remains pending.

- [Mathematical invariance and intervention-sensitivity benchmark](MATH_METAMORPHIC_V1.md) — paired metamorphic tests for harmless reformulations, semantics-changing mutations, proof-route-only changes and unknown cases; tracked by issue [#13547](https://github.com/Luminous-Dynamics/symthaea/issues/13547).

- [Minimum sufficient mathematical explanation protocol](MATH_MIN_SUFFICIENT_EXPLANATION_V1.md) — optimizes explanation cost under hard theorem/assumption/side-condition fidelity and measured human transfer constraints; tracked by issue [#13548](https://github.com/Luminous-Dynamics/symthaea/issues/13548).

- [Metamorphic pair schema](MATH_METAMORPHIC_PAIR_V1.schema.json) and [finite fixtures](MATH_METAMORPHIC_PAIR_V1.json) — exact typed sequent digests and finite truth-table controls for equivalent rewrites, changed hypotheses and proof-route-only changes; executable Rust validator and `math-metamorphic` CLI; issue [#13547](https://github.com/Luminous-Dynamics/symthaea/issues/13547).

- [Proof-action local certificate protocol](MATH_PROOF_ACTION_CERTIFICATE_V1.md) — protocol plus a source-pinned Lean 4.37 `And.comm` pilot, candidate mapping manifest and exact-head replay workflow; structured goal-state/context receipts remain future work; tracked by [#13549](https://github.com/Luminous-Dynamics/symthaea/issues/13549). [Manifest](PROOF_ACTION_CERTIFICATE_AND_COMM_V1.json) · [schema](PROOF_ACTION_CERTIFICATE_AND_COMM_V1.schema.json) · [Lean source](../../scripts/research/lean-proof-extractor/ProofActionCertificateAndComm.lean). The workflow also checks manifest-to-plan node/goal bindings with a mutated-manifest rejection control, and pins a wrong-proposition Lean negative control that must fail with an explicit type error. It also requires exact split-subgoal consumption with an omitted-subgoal mutation rejection, and tests whether a deliberately injected Lean axiom appears in `#print axioms` output. A separate fresh DFS fixture now audits eight roots: three expected `Classical.choice` roots plus five expected-empty `And.comm`/action roots. Native `#print axioms` is diagnostic-only. See [independent audit protocol](LEAN_AXIOM_AUDIT_INDEPENDENCE_V1.md), [15-root manifest/schema](LEAN_AXIOM_AUDIT_IMPORTED_INDUCTIVE_V1.json), [A](../../scripts/research/lean-proof-extractor/axiom-audit-imported-inductive/A.lean), [B](../../scripts/research/lean-proof-extractor/axiom-audit-imported-inductive/B.lean), and [#13575](https://github.com/Luminous-Dynamics/symthaea/issues/13575).


## Spectral–Dynamical Cognition Bridge (SDCB v1)

A falsifiable programme for spectral statistics, RH/prime-structure controls, neural-weight spectra, criticality, Koopman dynamics, observability, memory, and proof/search.

- [Research protocol](SPECTRAL_DYNAMICAL_COGNITION_BRIDGE_V1.md)
- [Industry and research benchmark landscape (2026-11-25)](SDCB_INDUSTRY_BENCHMARK_LANDSCAPE_2026_11_25.md)
- [Hidden evaluation and reproducibility protocol](SDCB_HIDDEN_EVALUATION_AND_REPRODUCIBILITY_V1.md)
- [34-task exploratory pilot manifest](SPECTRAL_DYNAMICAL_PILOT_V1.json)
- [Benchmark manifest schema](SPECTRAL_DYNAMICAL_BENCHMARK_MANIFEST_V1.schema.json)
- [Dependency-free Rust analytic reference harness](../../scripts/research/sdcb_reference.rs)
- [Reference harness run instructions and limits](../../scripts/research/SDCB_REFERENCE_HARNESS_README.md)
- [Machine-readable reference receipt runner](../../scripts/research/sdcb_reference_receipt.rs)
- [Reference receipt schema](SDCB_REFERENCE_CONTROL_RECEIPT_V1.schema.json)
- [Scientific-discovery benchmark interoperability](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md)

**Evidence boundary:** artifacts are research specifications and authored reference tests. Rust compilation/tests, model inference, generated trajectory experiments, hidden-set creation, external baseline reproduction, and scientific results remain pending.


### Scientific-discovery benchmark interoperability

- [Scientific-discovery benchmark interoperability](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md): adapters and matched evaluation plans for Autonomous Science Bench, FIRE-Bench, SciAgentArena, AstaBench, SorryDB, Goedel-Prover-V2 and the Open Agent Leaderboard.


### Evidence-gated scientific belief transitions

- [Evidence-Gated Belief Transitions v1](SDCB_EVIDENCE_GATED_BELIEF_TRANSITION_V1.md): formal state-transition contract, update integrity, metrics, and claim boundaries.
- [External benchmark update — 26 November 2026](SDCB_EXTERNAL_BENCHMARK_UPDATE_2026_11_26.md): Beyond Accuracy, ActiveSciBench, Autonomous Science Bench, ACDB, AutoSciBench and CoDBench.
- [Experiment transition schema](SDCB_EXPERIMENT_TRANSITION_V1.schema.json): accepted belief-transition wire contract.
- [Example transition record](SDCB_EXPERIMENT_TRANSITION_V1.example.json): illustrative fixture; digest strings and verifier identity are placeholders.
- [Belief-transition Rust reference](../../scripts/research/sdcb_belief_transition.rs): standard-library-only expected information gain, cost-aware experiment selection, Bayesian update and atomic evidence gate.
- [Reference harness README](../../scripts/research/SDCB_BELIEF_TRANSITION_README.md): independent compile/run instructions and trust boundaries.

**Evidence boundary:** 23 Rust unit tests are authored; compilation and execution have not been performed in this environment. The reference code does not authenticate signatures, source custody, or physical sensors.

### Decision-Sufficient Scientific Intelligence

- [Decision-Sufficient Scientific Intelligence v1](SDCB_DECISION_SUFFICIENT_SCIENTIFIC_INTELLIGENCE_V1.md): task-relative decision sufficiency, EVSI, counterexample, benchmark protocol and research targets.
- [Decision-value episode schema](SDCB_DECISION_VALUE_EPISODE_V1.schema.json): machine-readable finite decision/experiment episode contract.
- [EIG-versus-EVSI example](SDCB_DECISION_VALUE_EPISODE_V1.example.json): frozen three-hypothesis counterexample.
- [Expected analytic result](SDCB_DECISION_VALUE_EPISODE_V1.expected.json): independent calculation with posteriors, EIG, EVSI, and cost-aware choices.
- [Decision-value Rust reference](../../scripts/research/sdcb_decision_value.rs): Bayes action, decision-equivalence partition, expected information gain, EVSI and net-EVSI selection.
- [Decision-value harness README](../../scripts/research/SDCB_DECISION_VALUE_README.md): compile instructions, example, coverage and limits.

**Evidence boundary:** 15 Rust unit tests are authored but not compiled/executed in this environment. The reference is one-step, finite, and assumes supplied likelihoods and utilities. It is not a Symthaea model result or an SOTA comparison.
### Non-myopic adaptive experiment oracle

- [Non-myopic oracle source](../../scripts/research/sdcb_nonmyopic_oracle.rs): exact adaptive Bellman recursion for small finite experiment sets (11 additional tests; the combined target includes 25).
- [Non-myopic oracle README](../../scripts/research/SDCB_NONMYOPIC_ORACLE_README.md): compile instructions, exact-search limits and trust boundaries.
- [Non-myopic episode schema](SDCB_NONMYOPIC_VALUE_EPISODE_V1.schema.json): constrained adaptive decision episode.
- [Non-myopic example](SDCB_NONMYOPIC_VALUE_EPISODE_V1.example.json): two zero-myopic-EVSI probes with a positive two-step policy value.
- [Non-myopic expected result](SDCB_NONMYOPIC_VALUE_EPISODE_V1.expected.json): expected value 0.85 versus stop value 0.80, checked by independent dynamic-programming calculation.

**Evidence boundary:** 10 additional Rust test functions are authored but not compiled/executed. The exact oracle is intentionally bounded to 8 candidates, 4 steps, and 4 outcomes.
### Robust and Pareto-aware scientific decision design

- [Robust and Pareto-Aware Scientific Decision Design v1](SDCB_ROBUST_AND_PARETO_DECISION_DESIGN_V1.md): finite ambiguity-set robust EVSI, model-misspecification safeguards, Pareto policy frontiers, and non-compensatory constraints.
- [petri-bench interoperability analysis](SDCB_PETRI_BENCH_INTEROPERABILITY_2026_11_26.md): method-integrity benchmark comparison and proposed extensions.
- [Robust episode schema](SDCB_ROBUST_DECISION_EPISODE_V1.schema.json), [example](SDCB_ROBUST_DECISION_EPISODE_V1.example.json), and [expected result](SDCB_ROBUST_DECISION_EPISODE_V1.expected.json).
- [Pareto episode schema](SDCB_PARETO_DECISION_EPISODE_V1.schema.json), [example](SDCB_PARETO_DECISION_EPISODE_V1.example.json), and [expected result](SDCB_PARETO_DECISION_EPISODE_V1.expected.json).
- [Robust/Pareto Rust reference](../../scripts/research/sdcb_robust_pareto.rs): maximin experiment choice, Pareto action/policy frontiers, per-hypothesis safety floors, and zero-cost lottery controls.
- [Robust/Pareto harness README](../../scripts/research/SDCB_ROBUST_PARETO_README.md): compile instructions, limits and claim boundaries.

**Evidence boundary:** 19 Rust test functions are authored but have not been compiled or executed. Independent numerical calculations match the frozen robust and Pareto fixtures. No external benchmark, Symthaea model, or physical system has been evaluated.
### Existing Symthaea selector integration

- [Existing selector integration map](SDCB_EXISTING_SELECTOR_INTEGRATION_MAP_2026_11_26.md): audit of the HDC prediction-disagreement selector, Spark EIG, Cell Foundry HDC/CfC planner, Nuclear heuristic, synthetic ISRU bake-off, and decision-relevant chaos benchmark.
### R-I / R-IDeA under model misspecification

- [R-IDeA acquisition protocol](SDCB_RIDEA_ROBUST_ACQUISITION_V1.md): AISTATS 2026 equations and source-faithful released MMD² semantics.
- [R-IDeA episode schema](SDCB_RIDEA_EPISODE_V1.schema.json), [example](SDCB_RIDEA_EPISODE_V1.example.json), and [expected result](SDCB_RIDEA_EPISODE_V1.expected.json).
- [R-IDeA score reference](../../scripts/research/sdcb_ridea.rs): empirical MMD², R-I, DeA and cost-separated ranking.
- [Linear proxy trainer](../../scripts/research/sdcb_ridea_proxy.rs), [protocol](SDCB_RIDEA_PROXY_TRAINING_V1.md), and [fixture](SDCB_RIDEA_PROXY_TRAINING_V1.example.json): MSE + hinge-margin + L2 proxy objective with a transparent linear feature map.
- [Proxy episode schema](SDCB_RIDEA_PROXY_TRAINING_V1.schema.json) and [expected result](SDCB_RIDEA_PROXY_TRAINING_V1.expected.json).
- [R-IDeA harness README](../../scripts/research/SDCB_RIDEA_README.md) and [proxy README](../../scripts/research/SDCB_RIDEA_PROXY_README.md).

**Evidence boundary:** 18 R-I/R-IDeA scoring tests plus 13 proxy-training tests (31 total) are authored but not compiled/executed. The proxy trainer is a linear-feature reference, not the original Pyro architecture or an upstream experiment reproduction.
### Acquisition comparison bake-off

- [SDCB acquisition comparison bake-off v1](SDCB_ACQUISITION_COMPARISON_BAKEOFF_V1.md): preregistered adapters and matched evaluation across fixed/OFAT/space-filling/random, prediction disagreement, Spark EIG, R-I/R-IDeA, EVSI, exact non-myopic, robust maximin and Pareto/MOBO methods.
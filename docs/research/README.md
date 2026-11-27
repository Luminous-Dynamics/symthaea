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


## Current embodiment research (2026-11-27)

The polyembodiment programme treats the digital avatar, puppets, abstract forms,
and future physical companions as adapters for one renderer-neutral semantic
event contract.

- [Polyembodiment architecture](SYMTHAEA_POLYEMBODIMENT_ARCHITECTURE_V0_1_2026-11-27.md)
- [Embodiment benchmark plan](SYMTHAEA_EMBODIMENT_BENCHMARK_PLAN_V0_1_2026-11-27.md)
- [Dynamic temporal attention](SYMTHAEA_DYNAMIC_TEMPORAL_ATTENTION_V0_1_2026-11-27.md)
- [Consent and presentation policy](SYMTHAEA_EMBODIMENT_CONSENT_POLICY_V0_1_2026-11-27.md)
- [Deterministic scene compiler](SYMTHAEA_EMBODIMENT_SCENE_COMPILER_V0_1_2026-11-27.md)
- [Bevy 0.20.1 renderer adapter](../../renderers/symthaea-soma-bevy/README.md)
- [Scene frame schema](SYMTHAEA_EMBODIMENT_SCENE_SCHEMA_V1_2026-11-27.json)
- [Scene frame fixtures](SYMTHAEA_EMBODIMENT_SCENE_FIXTURES_V1_2026-11-27.json)
- [Presentation outcome receipts](SYMTHAEA_PRESENTATION_OUTCOME_RECEIPTS_V0_1_2026-11-27.md)
- [Compositor presentation observer](SYMTHAEA_COMPOSITOR_PRESENTATION_OBSERVER_V0_1_2026-11-27.md)
- [Wayland presentation-feedback ledger](../../crates/domains/symthaea-soma/src/embodiment_presentation_feedback.rs)
- [Presentation receipt schema](SYMTHAEA_PRESENTATION_RECEIPT_SCHEMA_V1_2026-11-27.json)
- [Presentation receipt fixtures](SYMTHAEA_PRESENTATION_RECEIPT_FIXTURES_V1_2026-11-27.json)
- [Presentation grant schema](SYMTHAEA_PRESENTATION_CONSENT_GRANT_SCHEMA_V1_2026-11-27.json)
- [Presentation grant fixtures](SYMTHAEA_PRESENTATION_CONSENT_GRANT_FIXTURES_V1_2026-11-27.json)
- [Competitive baselines and differentiation](SYMTHAEA_EMBODIMENT_COMPETITIVE_BASELINES_V0_1_2026-11-27.md)
- [Integrated interaction benchmark protocol](SYMTHAEA_INTERACTION_BENCH_INTEGRATION_V0_1_2026-11-27.md)
- [Visual language and perceptual qualification](SYMTHAEA_VISUAL_LANGUAGE_AND_PERCEPTUAL_QUALIFICATION_V0_1_2026-11-27.md)
- [Deterministic seven-form Bevy layout](../../renderers/symthaea-soma-bevy/src/visual_grammar.rs)
- [Interaction trace schema](SYMTHAEA_INTERACTION_TRACE_SCHEMA_V1_2026-11-27.json)
- [Interaction trace fixtures](SYMTHAEA_INTERACTION_TRACE_FIXTURES_V1_2026-11-27.json)
- [Causal interaction-trace verifier](../../crates/domains/symthaea-soma/src/embodiment_interaction_trace.rs)
- [Trace-verifier CLI](../../crates/domains/symthaea-soma/src/bin/verify_interaction_trace.rs)
- [Turn-control benchmark protocol](SYMTHAEA_TURN_CONTROL_BENCHMARK_V0_1_2026-11-27.md)
- [Cross-form semantic continuity](SYMTHAEA_CROSS_FORM_CONTINUITY_V0_1_2026-11-27.md)
- [Scene continuity descriptor schema](SYMTHAEA_SCENE_CONTINUITY_SCHEMA_V1_2026-11-27.json)
- [Scene continuity fixtures](SYMTHAEA_SCENE_CONTINUITY_FIXTURES_V1_2026-11-27.json)
- [Causal turn decision gate](SYMTHAEA_CAUSAL_TURN_GATE_V0_1_2026-11-27.md)
- [Turn candidate schema](SYMTHAEA_TURN_CANDIDATE_SCHEMA_V1_2026-11-27.json)
- [Turn candidate fixtures](SYMTHAEA_TURN_CANDIDATE_FIXTURES_V1_2026-11-27.json)
- [Event schema](SYMTHAEA_EMBODIMENT_EVENT_SCHEMA_V1_2026-11-27.json)
- [Event fixtures](SYMTHAEA_EMBODIMENT_EVENT_FIXTURES_V1_2026-11-27.json)
- [Embodiment form registry](SYMTHAEA_EMBODIMENT_FORM_REGISTRY_V1_2026-11-27.json)

The renderer-independent event, projection, policy, receipt, scene, turn-control, continuity, interaction-trace verification and compositor-feedback modules live under
`crates/domains/symthaea-soma/src/`. The cross-event trace verifier has a Rust CLI at
`crates/domains/symthaea-soma/src/bin/verify_interaction_trace.rs`; it checks causal relationships but is not a JSON Schema validator.
A standalone Bevy 0.20.1 adapter has transition-aware source code under
`renderers/symthaea-soma-bevy/`, outside the root Cargo workspace; it has not been built or run.
All 160 current regression tests are authored but unexecuted, and no VideoFDB, StreamEval, TurnBench or InteractionBench evaluation result is claimed.

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

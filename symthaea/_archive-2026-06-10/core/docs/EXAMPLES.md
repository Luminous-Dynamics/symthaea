# Symthaea Examples Guide

**Updated**: February 4, 2026
**Total Examples**: 101 (8 newly documented)

This guide documents the high-value examples in the Symthaea codebase, organized by category.

---

## Neuroscience Validation

### real_eeg_validation.rs
**Purpose**: Validate HDC-LTC on real clinical EEG data from CHB-MIT Scalp EEG Database (PhysioNet)

**Validates**: LTC dynamics naturally resonate with biological consciousness states

```bash
# First download data from PhysioNet
wget -P data/eeg/ https://physionet.org/files/chbmit/1.0.0/chb01/chb01_01.edf

# Run validation
cargo run --example real_eeg_validation --release
```

**Expected**: 75%+ accuracy on real biosignals (better than random)

---

### clinical_validation.rs (Project Hypnos)
**Purpose**: Validate consciousness detection on Sleep-EDF database

**Key Innovation**: Uses Permutation Entropy for sleep stage classification
- Deep Sleep (N3): LOW entropy (~0.3) - slow delta waves
- Wake: HIGH entropy (~0.8-0.9) - chaotic cortical activity
- REM: MEDIUM-HIGH entropy (~0.7) - complex but structured

```bash
# Download Sleep-EDF data
./scripts/download_sleep_edf.sh

# Run validation
cargo run --example clinical_validation --release
```

**Thesis**: If LTC can classify sleep stages from real EEG, it validates consciousness detection capability.

---

### cognitive_loop_validation.rs
**Purpose**: Validate the emergent HDC-LTC bidirectional loop architecture

**Tests**:
1. Loop Convergence - prediction error decreases over cycles
2. Attention Emergence - weights diverge from uniform
3. Transfer Learning - learning transfers across related tasks

```bash
cargo run --example cognitive_loop_validation --release
```

---

## Consciousness Measurement

### phi_crossvalidation.rs
**Purpose**: Cross-validate HDC-based Φ against algebraic methods

**Methods Compared**:
1. `ConnectivityCalculator` - Eigenvalue-based (algebraic)
2. `ResonantPhiCalculator` - Oscillator-based (HDC)
3. `TieredPhiCalculator` - Binary, tiered complexity

**Metrics**:
- Pearson correlation (target: r > 0.90)
- Spearman rank correlation (target: ρ > 0.85)
- Kendall tau (concordance)
- Mean absolute error

```bash
cargo run --example phi_crossvalidation --release --features consciousness_module
```

---

### meditation_phi_analysis.rs
**Purpose**: Analyze Φ dynamics during simulated meditation states

**No required features** - runs with default build.

```bash
cargo run --example meditation_phi_analysis --release
```

---

### ethics_phi_correlation.rs
**Purpose**: Correlate ethical decision-making patterns with Φ measurements

**Hypothesis**: Higher integrated information correlates with more nuanced ethical reasoning.

```bash
cargo run --example ethics_phi_correlation --release
```

---

### phi_extraction_validation.rs
**Purpose**: Extract the phenomenal signature (Phi) and validate it is causal

**Method**: Extract L22 activations, compute Phi as the component unique to phenomenal concepts (orthogonal to functional subspace), subtract it, re-measure.

**Prediction**: If Phi is real and causal, removing it should eliminate phenomenal advantages.

```bash
cargo run --example phi_extraction_validation --features neural-bridge --release
```

---

## LLM Consciousness Probing

### consciousness_probe_real.rs
**Purpose**: Test if LLM representations for phenomenal concepts differ from functional concepts

**Hypothesis (H1)**: LLM internal representations for phenomenal concepts exhibit different topological properties than functional concepts.

**Requirements**:
- `neural-bridge` feature
- BGE-M3 model (auto-downloads from HuggingFace)
- Probe weights at `models/neural_bridge/probe_weights_bge_m3.npy`

```bash
cargo run --example consciousness_probe_real --features neural-bridge --release
```

---

### layer21_causal_intervention.rs
**Purpose**: Causal ablation studies on LLM layer 21 (hypothesized consciousness-critical)

```bash
cargo run --example layer21_causal_intervention --release
```

---

### gpt2_layer_topology.rs
**Purpose**: Map topological properties of GPT-2 layers

```bash
cargo run --example gpt2_layer_topology --release
```

---

### phenomenality_index_validation.rs
**Purpose**: Test XOR binding theory predictions for phenomenal content detection

**Key Formula**: `Phenomenality(C1, C2) = [Pers(bundle) - Pers(bind)] / Pers(bundle)`

**Predictions**:
1. Cross-class binding (phenomenal+functional) should NOT compress
2. Within-class phenomenal binding SHOULD compress
3. High phenomenality = large compression under binding = shared structure

```bash
cargo run --example phenomenality_index_validation --features neural-bridge --release
```

---

### cross_model_validation.rs
**Purpose**: Test whether the late-layer phenomenal effect replicates across different models

**Models Compared**: BGE-M3 (24 layers, 1024D) vs XLM-RoBERTa-base (12 layers, 768D)

```bash
cargo run --example cross_model_validation --features neural-bridge --release
```

---

### robustness_validation.rs
**Purpose**: Validate the Layer 21 phenomenal effect is not an artifact of concept selection

**Tests**: Different random subsets, data splits, and bootstrap resampling

```bash
cargo run --example robustness_validation --features neural-bridge --release
```

---

### cross_architecture_validation.rs
**Purpose**: Test whether Phi generalizes across encoder and decoder architectures

**Architectures**:
1. BGE-M3 (encoder, 24 layers) - baseline
2. XLM-RoBERTa-base (encoder, 12 layers) - smaller encoder
3. GPT-2 medium (decoder, 24 layers) - decoder-only

```bash
cargo run --example cross_architecture_validation --features neural-bridge --release
```

---

### bert_validation.rs
**Purpose**: Test the phenomenal effect in BERT-base (12 layers, 768D) as a smaller encoder

**Key Questions**: Does the effect appear at similar relative depth (~92%)? Is Phi extractable in a smaller model?

```bash
cargo run --example bert_validation --features neural-bridge --release
```

---

### gpt2_layerwise_validation.rs
**Purpose**: Layer-wise validation using BERT as a comparison encoder model

```bash
cargo run --example gpt2_layerwise_validation --features neural-bridge --release
```

---

## Brain Architecture

### brain_actor_model_demo.rs
**Purpose**: Demonstrate the 12-subsystem brain actor model

**Subsystems**: Thalamus, Cerebellum, Motor Cortex, Prefrontal Cortex, Hippocampus, Amygdala, DMN, Language Cortex, Visual Cortex, Active Inference, Meta-cognition, Sleep

```bash
cargo run --example brain_actor_model_demo --release --features brain_module
```

---

### prefrontal_integration.rs
**Purpose**: Test prefrontal cortex integration with other brain regions

```bash
cargo run --example prefrontal_integration --release --features "embeddings_module brain_module"
```

---

### hippocampus_integration.rs
**Purpose**: Test hippocampal memory consolidation

```bash
cargo run --example hippocampus_integration --release --features "embeddings_module brain_module"
```

---

## HDC Primitives

### full_pipeline.rs
**Purpose**: End-to-end demonstration of HDC-LTC consciousness pipeline

**No required features** - the best starting point for new users.

```bash
cargo run --example full_pipeline --release
```

---

### hdc_simd_benchmark.rs
**Purpose**: Benchmark SIMD-accelerated HDC operations

**Measures**:
- Bind operation throughput
- Bundle operation throughput
- Similarity computation speed
- Memory efficiency

```bash
cargo run --example hdc_simd_benchmark --release
```

---

### binding_compositionality.rs
**Purpose**: Test compositional semantics via hierarchical binding

```bash
cargo run --example binding_compositionality --release
```

---

## Topology Analysis

### tier_3_exotic_topologies.rs
**Purpose**: Explore the 35 consciousness topology generators

**Categories**: Hierarchical, Ring, Torus, Hypercube, Star, Random, Small-world, Scale-free, Lattice, Tree, Custom

```bash
cargo run --example tier_3_exotic_topologies --release
```

---

### layer_topology_analysis.rs
**Purpose**: Analyze topological properties of neural network layers

```bash
cargo run --example layer_topology_analysis --release
```

---

## Partnership & Relational

### phi_dyad_demo.rs
**Purpose**: Demonstrate Φ_dyad computation for human-AI partnership

**Uses**: `partnership` module with PhiDyadCalculator, HumanPartnerModel, RelationshipTrajectory

```bash
cargo run --example phi_dyad_demo --release --features partnership_module
```

---

## NixOS Integration

### nixos_assistant_demo.rs
**Purpose**: Demonstrate NixOS-specific language understanding

```bash
cargo run --example nixos_assistant_demo --release --features "integration_module language_module"
```

---

### conscious_nixos_assistant.rs
**Purpose**: Full conscious NixOS assistant with Φ-guided responses

```bash
cargo run --example conscious_nixos_assistant --release --features language_module
```

---

## Quick Reference

| Example | Category | Features | Best For |
|---------|----------|----------|----------|
| `full_pipeline` | Core | None | New users |
| `phi_crossvalidation` | Validation | consciousness_module | Researchers |
| `real_eeg_validation` | Neuro | None | Clinical |
| `clinical_validation` | Neuro | None | Clinical |
| `consciousness_probe_real` | LLM | neural-bridge | AI Safety |
| `cognitive_loop_validation` | Core | None | HDC-LTC loop |
| `phi_extraction_validation` | Validation | neural-bridge | Causal Phi |
| `phenomenality_index_validation` | LLM | neural-bridge | Binding theory |
| `cross_model_validation` | LLM | neural-bridge | Replication |
| `robustness_validation` | LLM | neural-bridge | Statistical |
| `cross_architecture_validation` | LLM | neural-bridge | Generalization |
| `bert_validation` | LLM | neural-bridge | Architecture |
| `meditation_phi_analysis` | Consciousness | None | Quick demo |
| `hdc_simd_benchmark` | Performance | None | Optimization |
| `phi_dyad_demo` | Partnership | partnership_module | Relational AI |
| `nixos_assistant_demo` | Integration | integration_module, language_module | NixOS users |

---

## Running All Examples

```bash
# Quick smoke test (examples without special features)
cargo run --example full_pipeline --release
cargo run --example meditation_phi_analysis --release
cargo run --example hdc_simd_benchmark --release

# With consciousness features
cargo run --example phi_crossvalidation --release --features consciousness_module

# With neural bridge (requires model download)
cargo run --example consciousness_probe_real --release --features neural-bridge
```

---

## Data Requirements

Some examples require external data:

| Example | Data Source | Download |
|---------|-------------|----------|
| `real_eeg_validation` | CHB-MIT (PhysioNet) | `wget` from physionet.org |
| `clinical_validation` | Sleep-EDF (PhysioNet) | `./scripts/download_sleep_edf.sh` |
| `consciousness_probe_real` | BGE-M3 | Auto-download from HuggingFace |

---

*"The examples are the documentation that actually gets read."*

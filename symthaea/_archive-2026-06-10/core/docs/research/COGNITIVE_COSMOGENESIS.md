# Cognitive Cosmogenesis Research Design

## Overview
This document outlines the design for a "Cognitive Cosmogenesis" framework within Symthaea. It uses the Lambda-CDM (ΛCDM) model as an architectural grammar to manage the organization of semantic manifolds.

**Core Thesis:** Meaning forms like structure in an expanding, gravitationally-tethered universe. 

## 1. Scope and Distinctions
*   **What is Literal?** The mathematical implementation of ΛCDM-inspired dynamics (expansion, attraction, perturbation, cooling) applied to semantic particles in vector space.
*   **What is Metaphorical?** The mapping of cosmological terms to cognitive components. We do **not** claim physical cosmology explains consciousness; we claim cosmological algorithms optimize semantic organization.

### Mappings
| Cosmology | Cognitive Mapping |
| :--- | :--- |
| Baryonic Matter | Active Concepts / Explicit Thoughts |
| Dark Matter | Latent Attractors / Memory Priors |
| Lambda (Λ) | Novelty Pressure / Expansion Drive |
| Gravity | Attractor-based Semantic Similarity |
| Cosmic Web | RHN/HDC Knowledge Topology |

## 2. Research Goals and Measurement
*   **Primary Objective:** Determine if ΛCDM-inspired dynamics (expansion + attractor gravity) improve semantic cluster separation compared to a standard HDC/random baseline.
*   **Baseline:** Standard HDC clustering without external manifold dynamics.
*   **Measurement:**
    *   **Cluster Separation:** Measured via separation_proxy (mean inter-class / intra-class distance ratio).
    *   **Retrieval Accuracy:** Precision/Recall on query-to-cluster mapping.
    *   **Entropy:** Measure of dispersion/collapse in the manifold.
*   **Falsification:** If ΛCDM-inspired manifold organization performs worse or equal to the baseline under standard benchmarks, the cosmological grammar provides no functional benefit for semantic organization.

## 3. Implementation Plan
*   **Integration:** Feature-gated (`--features cognitive-cosmogenesis`) experimental layer.
*   **Crate:** `crates/symthaea-cosmogenesis` (new experimental crate).
*   **Dependencies:** Clean interface to shared metric perturbation primitives.
*   **Explicit Non-Integration:** This crate must NOT be wired into the cognitive loop, Broca, RHN, memory consolidation, or gravcraft runtime until the clustering assay beats baseline.
*   **Scope:** 
    1. Deterministic particle simulation.
    2. Initialization from HDC vectors.
    3. Manifold evolution simulation (expansion/attraction).
    4. Quantitative assessment of clustering.

## 4. Minimum Viable Experiment
Given N labeled HDC vectors from K semantic classes:

1. Project or initialize them as semantic particles.
2. Run baseline clustering without cosmogenesis.
3. Run ΛCDM-inspired evolution for T steps.
4. Measure cluster separation, entropy, and retrieval accuracy.
5. Compare against baseline.

**Success condition:** The cosmogenesis run improves at least two of: separation_proxy, retrieval precision @K, or cluster stability under perturbation.
**Failure condition:** No improvement over baseline across repeated deterministic seeds.

## 5. Data Structures
```rust
pub struct CognitiveCosmologyParams {
    pub matter_density: f32,
    pub dark_matter_density: f32,
    pub lambda: f32,
    pub attraction_strength: f32,
    pub perturbation_scale: f32,
    pub cooling_rate: f32,
    pub steps: usize,
}

pub struct CognitiveCosmogenesisMetrics {
    pub separation_proxy: f32,
    pub davies_bouldin_index: f32,
    pub retrieval_precision_at_k: f32,
    pub entropy: f32,
    pub cluster_stability: f32,
}
```

## 6. Risks and Mitigation
*   **Metaphor Leakage:** Mitigated by documenting the model strictly as an "algorithmic organization strategy" in all public/internal technical reports.
*   **Scope Creep:** Mitigated by strict separation from physical `gravcraft` logic (no modification of physical navigation) and modularizing the cognitive cosmogenesis code.

## 7. Verification Status
*   **Compile verified:** Yes, via private `CARGO_TARGET_DIR` (2026-06-12).
*   **Test status:** Passed (Stability/Finite-state preservation).
*   **Tuning test:** Included as `#[ignore]` (Separation improvement not yet proven).
*   **Limitation:** No proven semantic separation improvement yet; algorithm is currently at the "stable scaffold" phase.

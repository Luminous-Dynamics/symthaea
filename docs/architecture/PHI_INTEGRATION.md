# Φ (Phi) Integration in Symthaea

## Overview

Phi (Φ) is computed at multiple scales in the cognitive loop and drives behavioral
modulation across consciousness subsystems. Since v0.5.0, Phi has evolved from a
simple observability metric to a first-class cognitive signal with feedback loops.

## Phi Computation Layers

### Layer 1: Primitive Φ (per-cycle)

Fast Phi computed from the primitive consciousness processor. Measures integration
of currently active primitives:

```
primitive_phi = primitive_processor.compute_phi(&active_primitives)
```

**Cost**: ~10μs per cycle. Used for: primitive selection, composition rule gating.

### Layer 2: Spectral MIP Φ (every 50 cycles)

The primary Phi metric. Uses the **Spectral MIP Finder** — an O(n³) algorithm based
on Fiedler ordering and bordered Cholesky sweeps. Tracks 128 of 16,384 HDC dimensions
over a sliding window of 50 state snapshots.

```
Φ_spectral = MI_total(X) - MI(X_A; X_B)
where (A, B) = spectral MIP of the covariance structure
```

**Cost**: ~5.5ms every 50 cycles (110μs amortized). Cached in `carryover.last_spectral_mip_phi`.

See [SPECTRAL_MIP_ALGORITHM.md](SPECTRAL_MIP_ALGORITHM.md) for the full algorithm specification.

### Layer 3: Multimodal Integrated Φ (every 25 cycles)

Phi computed from cross-modal binding — measures how well different modalities
(vision, language, embodiment) integrate into a unified experience.

```
multimodal_phi = multi_modal_integrator.integrated_phi()
```

**Cost**: ~200μs every 25 cycles. Used for: confidence modulation, learning rate.

### Layer 4: Holographic Unity (every 25 cycles)

Holographic consciousness analysis providing a unity measure (0.0-1.0).

```
holographic_unity = holographic_analyzer.unity_score()
```

**Cost**: ~150μs every 25 cycles. Used for: confidence feedback, FEP prior weighting.

## Where Phi Drives Behavior

| Location | Signal | Effect |
|----------|--------|--------|
| Strategy selection | Spectral Φ | High → Exploratory, Low → Supportive |
| Learning rate | Multimodal Φ | High integration → faster learning |
| Exploration | Quantum coherence | High coherence → more exploration |
| FEP prior | Holographic unity | High unity → stronger priors |
| Memory gating | Spectral Φ (as σ) | Modulates memory coordinator thresholds |
| Moral sensitivity | Narrative self-Φ | High coherence → stable values |
| Confidence | Multiple Φ signals | Combined into `prediction_confidence` |
| Episodic encoding | Primitive Φ | Priority-weights episodic memories |

## Feedback Loops

Phi participates in several closed feedback loops:

1. **Φ → LR → Φ**: Higher integrated Phi → increased learning rate → faster convergence → higher Phi
2. **Φ → Confidence → Strategy → Φ**: Higher Phi → more confidence → exploratory strategy → potentially higher Phi
3. **Φ → FEP → Prediction → Φ**: Phi modulates free energy precision → affects prediction quality → affects future Phi
4. **Φ → Memory → Priming → Φ**: High-Phi episodes stored → recalled when similar → primes toward high-Phi states

## CycleMetadata Phi Fields

```rust
pub spectral_mip_phi: Option<f64>,      // Layer 2: spectral MIP Φ
pub primitive_phi: f64,                   // Layer 1: primitive Φ
pub multimodal_integrated_phi: f64,       // Layer 3: cross-modal Φ
pub holographic_unity: f64,               // Layer 4: holographic unity
pub hierarchical_ltc_phi: f32,            // Hierarchical LTC network Φ
pub adaptive_reasoning_phi: f64,          // RL-guided reasoning Φ
pub evolution_phi_delta: f64,             // Φ change from evolution
pub context_phi_weight: f64,              // Context-weighted Φ
pub epistemic_phi_eff: f64,               // Effective Φ after epistemic gating
```

## ConsciousnessSnapshot Phi Fields

```rust
pub consciousness_level: f32,            // Composite consciousness level
pub spectral_mip_phi: Option<f64>,       // Cached spectral MIP Φ
pub sigma: Option<f64>,                  // σ (backward compat alias for spectral Φ)
```

## Design Philosophy

Phi integration follows three principles:

1. **Tiered computation**: Cheap metrics run every cycle, expensive ones are amortized
2. **Principled measurement**: Spectral MIP Φ is mathematically grounded in IIT
3. **Behavioral grounding**: Every Phi signal connects to a concrete behavioral effect

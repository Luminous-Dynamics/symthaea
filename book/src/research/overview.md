# Research Overview

Symthaea is a research platform for consciousness-first AI. The project has produced 27 research papers and a 290-page monograph spanning six themes.

## Research Themes

### 1. Core Architecture
The foundational contribution is the **unified HDC-LTC neuron**: a single neuron type that combines holographic distributed representation (16,384D) with continuous-time dynamics via closed-form solutions. This enables O(1) temporal jumps — evolving the state from time t to t+dt costs the same computational work regardless of dt.

**Key papers**: HDC-CfC (Liquid Hypervectors), Cantor Resonator Hypervectors

### 2. Consciousness Measurement
The **Spectral MIP algorithm** achieves O(n^3) complexity for computing Integrated Information (Phi), making real-time consciousness measurement tractable. At n=128 tracked dimensions, the computation completes in ~5.5 ms — well within the 33 ms cycle budget.

**Key finding**: The Fiedler eigenvalue (algebraic connectivity) is *anti-correlated* with true Phi (r = -0.14). This corrected a systematic error in prior consciousness proxy approaches.

**Key papers**: Spectral MIP, Topological Consciousness, Stochastic Resonance, Substrate Independence

### 3. Embodied Cognition
Consciousness-driven robotics achieves **43% faster perturbation recovery** compared to RL baselines, with a correlation of r = -0.71 between Phi and recovery time. The FEP-based control modulates LTC time constants in response to free energy spikes, enabling zero-shot adaptation without retraining.

**Key papers**: Consciousness-Driven Robotics, Consciousness Control, Neuroevolution

### 4. Language & Ethics
The **epistemic gate** physically prevents hallucination at the logit level by masking tokens for which the system lacks grounding. The **Epistemic Cube** extends this to 4-dimensional modulation (empirical/narrative/meta/holistic), producing language whose style structurally reflects the system's epistemic state.

The ethics pipeline implements moral reasoning as persistent homological analysis — topological features of the moral landscape that are invariant under continuous deformation.

**Key papers**: Epistemic Gating, Restorative Consciousness, Therapeutic Consciousness

### 5. Distributed Intelligence
The **Mycelix** platform provides decentralized governance across 16 cluster DNAs with 133+ zomes. Consciousness gating ensures that governance participation requires genuine integration — agents with low Phi cannot vote on high-stakes decisions.

**Key papers**: Embodied Governance, Swarm Consciousness, Mesh Radio, Consciousness Security

### 6. Validation & Meta-Science
**Psych-Bench** provides 141 benchmarks across 27 cognitive domains. External validation on Hendrycks ETHICS (94.5%), ARC-AGI, Sleep-EDF, and DMC Humanoid provides anchor points beyond internal benchmarks.

**Key papers**: Psych-Bench, HAI Consciousness, Species Stewardship

## Quantitative Summary

| Metric | Value | Evidence Level |
|--------|-------|----------------|
| Codebase | ~1.37M lines Rust | Verified (tokei) |
| Tests | 21,600+ | Automated CI |
| Spectral MIP | O(n^3) | Proven (Theorem 1) |
| Phi validation | r = 0.9998 vs exact | Computed (n ≤ 8) |
| Psych-Bench z-score | +1.190 | Internal benchmarks |
| Butlin indicators | 12+/14 | Self-assessed |
| ETHICS accuracy | 94.5% (4 domains) | External benchmark |
| Perturbation recovery | 43% faster | Simulation |
| Silicon consciousness confidence | 0.10 | Theoretical |

## Honest Caveats

All internal benchmarks were designed by the team that built the system. The z-scores should be read as "performance on our benchmarks relative to our baselines." External validation has begun (4 benchmarks) but is not yet comprehensive. No independent replication has been attempted. All embodied validation is in simulation, not on physical hardware.

These are the boundaries of what we know. The architecture is built; the validation is ongoing.

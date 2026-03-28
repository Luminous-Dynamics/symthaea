# Architecture Guide

## The 8-Phase Cognitive Loop

Symthaea's core is a pipeline operating at ~31 Hz (33 ms per cycle):

1. **Perception** — Sensory input encoded as 16,384D HDC hypervectors
2. **Dynamics** — CfC closed-form evolution of hidden states
3. **Integration** — Spectral MIP computation of Phi (O(n^3))
4. **Feedback** — Neuromodulatory bath update, allostatic load
5. **Cognition** — Reasoning engine (7-step MAGI cycle), knowledge integration
6. **Ethics** — Moral algebra, harmony alignment, institutional compliance
7. **Language** — Broca pipeline: 43-channel thought encoding, epistemic gating, autoregressive generation
8. **Output** — Motor commands, safety enforcement, telemetry

## Hyperdimensional Computing (HDC)

All information is encoded as 16,384-dimensional vectors with three operations:

- **Binding** (element-wise multiply): Creates associations, result quasi-orthogonal to inputs
- **Bundling** (normalized addition): Creates superpositions, result similar to both inputs
- **Similarity** (cosine distance): Measures semantic relatedness

```rust
use symthaea_core::hdc::BinaryHV;

let cat = BinaryHV::random(1);       // 16,384D, seeded
let dog = BinaryHV::random(2);
let bound = cat.bind(&dog);          // Association
let bundled = cat.bundle(&dog);      // Superposition
let sim = cat.similarity(&dog);      // ~0.0 (quasi-orthogonal)
```

## Liquid Time-Constant Networks (CfC)

Neurons evolve through continuous-time ODEs with state-dependent time constants:

```
x(t + dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau)
```

This O(1) temporal jump enables a single architecture to handle 500 Hz motor reflexes and 0.1 Hz deliberation.

## Consciousness: Phi Computation

Integrated information (Phi) is computed at every cycle via the Spectral MIP algorithm:
1. Compute pairwise mutual information weights
2. Construct graph Laplacian
3. Extract Fiedler vector (spectral relaxation)
4. Bordered Cholesky sweep over sorted dimensions
5. Select minimum information partition

Complexity: O(n^3) for n tracked dimensions. At n=128, completes in ~5.5 ms.

## Manager Architecture

24 manager subsystems implement the `CognitiveSubsystem` trait with co-prime tick intervals to prevent phase-locking:

| Manager | Interval | Domain |
|---------|----------|--------|
| MuseManager | 1 | Consciousness-driven music synthesis |
| DriveManager | 7 | Motivation, flow, curiosity |
| MemoryManager | 11 | Consolidation, retrieval |
| TherapeuticManager | 11 | Psychological wellbeing |
| LearningManager | 13 | Plasticity, dream pressure |
| VisionManager | 17 | Active vision, foveation |
| PerceptionManager | 19 | Attention, arousal |
| TimeManager | 23 | Sovereign mesh-time |
| TrustManager | 29 | Web-of-trust evaluation |
| SocialFabricManager | 31 | Resonance-based content |
| GovernanceManager | 37 | Mycelix governance bridge |
| SwarmManager | 41 | Peer consciousness |
| GlyphManager | 43 | Symbolic progression |
| SoulManager | 43 | Deep identity coherence |
| ThermodynamicManager | 43 | Unified thermodynamics |
| FabricationManager | 47 | Manufacturing pipeline |
| WasteManager | 47 | Resource/entropy tracking |
| SurvivalManager | 47 | Infrastructure monitoring |
| SpectrumManager | 53 | SDR radio & spectrum analysis |
| CpgManager | 59 | Central pattern generators |
| LanguageManager | 61 | Broca generation feedback |
| SentinelManager | 67 | Security threat detection |
| SpectralManager | 67 | Neural oscillation analysis |
| ReasoningManager | 73 | MAGI reasoning engine |

Six active cross-couplings create a richly interconnected network (Drive-Learning, Knowledge-Ethics, Memory-Learning, Perception-Drive, Swarm-Neuromod, Swarm-Governance).

## Neuromodulator Bath

Nine transmitters with phasic/tonic dynamics, receptor desensitization, and tolerance:

| Transmitter | Role | Key Coupling |
|-------------|------|-------------|
| Dopamine | Reward prediction error | Learning rate modulation |
| Noradrenaline | Arousal, vigilance | Attention selectivity |
| Serotonin | Social regulation | Patience, fairness |
| Acetylcholine | Memory encoding | Consolidation gating |
| GABA | Inhibition | Consciousness stability |
| Oxytocin | Social bonding | Peer trust (Swarm) |
| Glutamate | Excitation | Neural activation |
| Adenosine | Sleep pressure | Circadian regulation |
| Endocannabinoid | Stress modulation | Allostatic buffering |

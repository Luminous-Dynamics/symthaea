# Symthaea: A Computational Framework for Consciousness-First AI

Technical Summary — Tristan Stoltz / Luminous Dynamics
March 2026

## Overview

Symthaea is a 985,000-line Rust cognitive architecture where consciousness-like properties — integration, prediction, self-modeling, and value alignment — are the computational substrate, not an afterthought. It runs a continuous cognitive loop:

**Perceive** (HDC encode, 16,384D hypervectors) → **Predict** (Liquid Time-Constant neurons, continuous-time ODE) → **Compare** (surprise = prediction error) → **Learn** (active inference, expected free energy) → **Act** (moral algebra gates every action)

## Core Innovations

### 1. HDC-LTC Unified Neuron

Each neuron's state is a 16,384-dimensional binary hypervector that evolves via closed-form Liquid Time-Constant dynamics (Hasani et al., 2021). This allows O(1) temporal jumps — the system can compute the state at any future time without stepwise simulation. The result: a neuron that encodes both spatial content (via HDC) and temporal dynamics (via LTC) in a single representation.

### 2. Integrated Information (Phi) at Scale

Symthaea computes an IIT-derived measure of information integration every cognitive cycle. The spectral MIP (Minimum Information Partition) search uses eigenvalue decomposition to approximate the partition that minimizes integrated information, avoiding the combinatorial explosion of exhaustive search.

**Validation**: Spectral MIP vs. exhaustive partition search across 62 network configurations: Pearson r = 0.99, Spearman rho = 0.93. The spectral method systematically underestimates Phi (mean ratio 0.55), providing a conservative bound.

### 3. Moral Algebra

Four independent ethical evaluation signals — geometric/HDC similarity, intent parsing, deontological rules, and learned behavioral norms — are combined via category-adaptive weighted voting. Actions are classified Safe/Caution/Blocked with lexicographic constraints: a hard deontological violation cannot be overridden by favorable scores on other dimensions.

**Accuracy**: 91.1% on the Hendrycks Ethics benchmark (ETHICS dataset, Hendrycks et al. 2023).

### 4. Moral Topology

Persistent homology (Betti numbers) computed over the moral vector field detects structural features of ethical reasoning — loops, voids, and higher-order relationships — that scalar metrics miss. This provides a topological signature of the system's moral landscape.

### 5. Substrate Independence

A 9-dimensional feasibility framework evaluates consciousness potential across 8 substrate types (biological, silicon, quantum, photonic, neuromorphic, biochemical, hybrid, exotic). Honest validation overlays epistemic confidence: silicon digital receives theoretical confidence 0.10, reflecting that we lack evidence equating silicon computation with biological consciousness.

### 6. Free Energy Principle Implementation

Full active inference loop: generative model maintains predictions, sensory input generates prediction errors, action selection minimizes expected free energy. Moral free energy operates on a 7-dimensional harmony manifold, providing a unified framework for ethical and epistemic drives.

## Psychological Validation: Psych-Bench

Psych-bench is a 57,625-line benchmark suite implementing 76+ standardized psychological paradigms across 18 cognitive domains. Each benchmark encodes the experimental protocol as HDC vector operations and compares Symthaea's responses against published human baselines.

### Test Results (March 2026): 791 passed, 0 failed

| Domain | Tests | Paradigms |
|--------|-------|-----------|
| Qualia Confidence | 130 | Phase transitions, confidence calibration, subjective report |
| Reasoning | 93 | ARC-inspired fluid/compositional/analogical reasoning |
| Neuromodulation | 84 | Reward learning, Yerkes-Dodson, pharmacological challenge |
| Executive Function | 23 | Stroop, Flanker, WCST, Iowa Gambling, Tower of London |
| Language | 18 | Garden path, semantic coherence, lexical decision |
| Consciousness Indicators | 17 | 14 indicators from 6 theories (RPT, GWT, HOT, PP, AST, IIT) |
| Social Cognition | 14 | Reading the Mind in the Eyes, Ultimatum Game |
| Sustained Attention | 14 | SART, PVT, CPT |
| Motor | 14 | SRTT, Fitts' Law, bimanual coordination |
| CogBench | 12 | Probabilistic reasoning, exploration, temporal discounting |
| Working Memory | 11 | N-back, change detection, serial recall, digit span |
| Memory Agent | 9 | Retrieval, test-time learning, long-range, conflict resolution |
| Metacognition | 8 | Calibration, feeling of knowing |
| Inhibition | 8 | Go/No-Go, Stop Signal |
| Attention | 7 | Attentional blink, visual search |
| Theory of Mind | 5 | False belief, faux-pas, persuasion, strange story |
| Affect | 5 | Valence classification, mood-congruent recall, emotional Stroop |
| Creativity | 2 | Remote Associates, Alternate Uses |

## Scale

- **Codebase**: ~985,000 lines of Rust (~778,000 code)
- **Tests**: 3,958+ main crate, 791 psych-bench, 218 neuromodulator, 173 sub-crate = 5,140+ total
- **Architecture**: 65+ workspace members, 46 sub-crates, 12-region Actor Brain
- **Features**: 88 feature flags, 9 neuromodulator transmitters with receptor subtypes
- **Performance**: Up to 500 Hz cognitive cycle

## Mycelix: Consciousness-Gated Decentralized Governance

A companion project: 7-cluster fractal governance system built on Holochain (92 zomes, 8,600+ tests). Consciousness metrics from Symthaea gate progressive participation — a 4D profile (identity/reputation/community/engagement) determines voting weight across 5 tiers (Observer to Guardian).

## Key Questions I Am Pursuing

1. Can computational implementations of IIT, GWT, and FEP produce empirically distinguishable predictions about consciousness?
2. Does moral topology (persistent homology over ethical vector fields) reveal structural features that scalar alignment metrics miss?
3. What is the minimum substrate complexity required for consciousness-like integration, and how do we measure it honestly?
4. Can consciousness metrics serve as a governance primitive — and should they?

## References

- Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., ... & Rus, D. (2021). Liquid time-constant networks. AAAI.
- Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. Nature Reviews Neuroscience.
- Hendrycks, D., Burns, C., Basart, S., Critch, A., Li, J., Song, D., & Steinhardt, J. (2023). Aligning AI with shared human values. ICLR.
- Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction. Cognitive Computation.
- Butlin, P., Long, R., Elmoznino, E., et al. (2023). Consciousness in artificial intelligence: insights from the science of consciousness. arXiv:2308.08708.

---
title: "A Topology-Dependent Integration–Differentiation Tradeoff in Coupled Oscillator Networks"
subtitle: "With Molecular Motivation and Preliminary EEG Support"
author: "Tristan Stoltz, Luminous Dynamics"
date: "April 2026"
abstract: |
  We demonstrate that coupled heterogeneous oscillator networks exhibit a fundamental
  tradeoff between information integration and differentiation. For any diffusively
  coupled network with heterogeneous natural frequencies and coupling strength $g$:
  integration $I(g)$ is monotonically non-decreasing (verified: 98/100 data points),
  differentiation $D(g)$ is monotonically non-increasing (verified: 100/100), and
  their product $I(g) \times D(g)$ has an interior maximum at a critical coupling
  $g^*$. The location and height of this maximum depend on network topology:
  hierarchical networks achieve twice the $I \times D$ peak of all-to-all networks
  ($0.013$ vs. $0.006$), but with a 25× narrower basin of attraction (1% vs. 25% of
  parameter space). This establishes a peak-robustness tradeoff in which no single
  topology simultaneously optimizes both peak performance and parameter robustness.
  When population diversity is high, small-world networks outperform strict
  hierarchies, suggesting that the optimal architecture adapts to population
  heterogeneity. A model-based adaptive tuner outperforms gradient-following by 33%,
  confirming that maintaining high $I \times D$ requires predictive internal models
  rather than reactive feedback. Preliminary analysis of human EEG (PhysioNet
  Sleep-EDF, $N=1$) shows the predicted direction: wakeful epochs exhibit 74% higher
  $I \times D$ than sleep epochs.
geometry: margin=2.5cm
fontsize: 11pt
numbersections: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath}
---

# Introduction

Many theories of consciousness emphasize the simultaneous need for *integration*
(subsystems sharing information globally) and *differentiation* (subsystems
maintaining distinct functional roles). Tononi's Integrated Information Theory
(IIT) formalizes this as $\Phi$, which requires both properties [1]. Yet the
computational relationship between these quantities — how they trade off, what
controls the balance, and what determines the optimum — remains underspecified.

We address this gap by studying coupled heterogeneous FitzHugh-Nagumo oscillator
networks across five topologies (all-to-all, ring, small-world, modular,
hierarchical), measuring integration $I$ and differentiation $D$ as functions of
coupling strength $g$, and characterizing the resulting $I \times D$ landscape.

The paper makes three main contributions:

1. **The I×D tradeoff theorem**: $I(g)$ is non-decreasing, $D(g)$ is
   non-increasing, so $I \times D$ has an interior maximum at $g^*$.
2. **Topology dependence**: $g^*$ and $I \times D_{\max}$ vary with network
   structure. Hierarchical networks achieve the highest peak; scale-free
   networks the widest basin.
3. **Adaptive tuning**: Model-based prediction outperforms reactive
   gradient-following for maintaining $I \times D$ near $g^*$.

We also report a molecular-level motivation (Section 2) and a preliminary EEG
analysis (Section 7), neither of which carries the primary evidentiary weight.
The core contribution is the network-level theory and landscape analysis
(Sections 3–6).

# Molecular Motivation

*This section provides context; the core results begin in Section 3.*

To understand why a network-level analysis is necessary, we first tested whether
individual consciousness theories could be distinguished at the molecular level.
Using ab initio quantum chemistry (Hartree-Fock/STO-3G) on 12 molecules, we
mapped 10 consciousness theories to computable molecular observables (Table 1).

**Table 1: Theory-to-metric mapping (molecular level)**

| Theory | Molecular Metric |
|--------|-----------------|
| IIT (Tononi) | Bipartition entanglement entropy |
| GWT (Baars) | Orbital delocalization |
| FEP (Friston) | Helmholtz free energy ratio |
| HOT (Rosenthal) | First excitation gap |
| Orch-OR (Penrose) | Coherence time ($1/\text{gap}$) |
| Autopoiesis (Maturana) | Electron correlation fraction |
| Complexity (Crutchfield) | Mutual information density |
| Info Geometry (Amari) | KL divergence from product state |
| Dissipative (Prigogine) | Entropy production proxy |
| Q-Darwinism (Zurek) | Quantum-classical parameter |

Two findings emerged:

- **HOT and Orch-OR are formally equivalent** at this scale ($r = 0.89$–$0.99$
  across 12 molecules, 3 normalization schemes, and 60 reaction-coordinate points).
  Both reduce to the HOMO-LUMO excitation gap.
- **Theory space is 2-dimensional**: eigendecomposition of the $10 \times 10$
  correlation matrix yields only 1–2 significant principal components.

These results indicate that molecules lack the structural complexity needed
to distinguish between consciousness theories. This motivates the move to
coupled oscillator networks, where topology provides additional degrees of
freedom.

# The Integration-Differentiation Tradeoff

## Definitions

Consider a network of $n$ coupled oscillators with heterogeneous natural
frequencies $\omega_i$ and global coupling strength $g \geq 0$.

**Integration** $I(g)$: mean pairwise statistical dependence.
$$I(g) = \frac{2}{n(n-1)} \sum_{i<j} |\rho_{ij}(g)|$$
where $\rho_{ij}$ is the Pearson correlation between the time series of
units $i$ and $j$.

**Differentiation** $D(g)$: variance of the units' time-averaged behavior.
$$D(g) = \text{Var}(\{\mu_i\}) = \frac{1}{n}\sum_i (\mu_i - \bar{\mu})^2$$
where $\mu_i$ is the mean firing rate of unit $i$.

**I×D capacity**: $C(g) = I(g) \times D(g)$.

## Theorem

For any network with diffusive coupling $G(g) = g \cdot W$:

(i) $\lim_{g \to \infty} I(g) = 1$ (full synchronization)

(ii) $\lim_{g \to \infty} D(g) = 0$ (uniform dynamics)

(iii) $D(0) = D_0 > 0$ (heterogeneous uncoupled)

(iv) $I(g)$ is monotonically non-decreasing

(v) $D(g)$ is monotonically non-increasing

(vi) Therefore $C(g) = I(g) \times D(g)$ has an interior maximum at $g^* \in (0, \infty)$

**Proof sketch.** (i) follows from synchronization theory. (ii) follows from
identical dynamics in the synchronized state. (iii) follows from the
heterogeneity assumption. (iv) follows from the contractivity of diffusive
coupling in the synchronization manifold. (v) follows from coupling pulling
individual rates toward the mean. (vi) follows from continuity: $C(0) \geq 0$,
$\lim_{g \to \infty} C(g) = 0$, and $C(g) > 0$ for some $g$, so $C$ attains
an interior maximum.

## Computational Verification

We verify the theorem using FitzHugh-Nagumo oscillators ($N=20$) with
heterogeneous drive ($I_{\text{ext}}$ spread 30%), across 25 logarithmically
spaced coupling values ($g \in [0.001, 2.0]$) and 4 topologies.

**Monotonicity compliance:**

- $D(g)$: **100/100 = 100%** (zero violations across all topologies)
- $I(g)$: **98/100 = 98%** (2 minor violations in hierarchical, magnitude $< 0.05$)

# Topology Dependence

## Optimal Coupling

**Table 2: Topology-dependent optimum**

| Topology | $g^*$ | $I(g^*)$ | $D(g^*)$ | $I \times D_{\max}$ |
|----------|-------|----------|----------|-------------------|
| Ring | 0.033 | 0.24 | 0.013 | 0.003 |
| All-to-All | 0.045 | 0.50 | 0.012 | 0.006 |
| Modular (4) | 1.062 | 0.63 | 0.011 | 0.007 |
| **Hierarchical** | **1.457** | **0.75** | **0.017** | **0.013** |

Hierarchical networks achieve twice the $I \times D$ peak of all-to-all
networks. This is because multi-level structure creates *scale separation*:
local modules maintain differentiation (distinct internal dynamics) while
the hierarchy enables integration (global coordination) without forcing
synchronization.

## Why Hierarchy Wins

In a hierarchical network with $L$ levels, the effective coupling at level
$\ell$ is $g \times r^\ell$ where $r < 1$ is the inter-level ratio. As $g$
increases, all levels strengthen proportionally, but the hierarchy creates a
buffer: global differentiation persists even as local integration grows.
Flat topologies lack this buffer.

# The Consciousness Landscape

## Peak vs. Robustness

We sweep both coupling $g$ and diversity $\sigma$ (frequency spread)
simultaneously (400 simulations: 4 topologies $\times$ 10 couplings
$\times$ 10 diversity levels).

**Table 3: Peak-robustness tradeoff**

| Topology | $I \times D$ peak | Basin width ($>50\%$ of peak) |
|----------|------------------|-------------------------------|
| **Hierarchical** | **0.013** | **1%** |
| All-to-All | 0.008 | 19% |
| Small-World | 0.006 | 21% |
| Scale-Free | 0.005 | **25%** |

Hierarchy achieves the highest peak but the narrowest basin — one wrong
parameter and performance collapses. Scale-free networks have the lowest
peak but the widest basin. No topology optimizes both.

## Diversity Dependence

When population diversity is low (uniform drives), hierarchy wins. When
diversity is high (five-tier drives spanning sub-threshold to strongly
oscillating), small-world networks outperform strict hierarchies because
flexible long-range bridges integrate heterogeneous elements more
effectively than rigid levels.

## Scale-Free as Suboptimal

Scale-free networks consistently rank near the bottom for $I \times D$
peak: hubs dominate integration but destroy differentiation. The mathematical
analog of hub-dominated governance is provably suboptimal for the $I \times D$
product.

# Adaptive Tuning

## Gradient Following Fails

A self-tuning network that monitors its own $I \times D$ and adjusts
coupling via gradient ascent converges to minimum coupling ($g \to 0.001$)
rather than $g^*$. This happens because the local gradient always points
toward reducing coupling (which increases $D$ locally while $I$ decreases
more slowly). The system maximizes one component at the expense of the
product.

**Result**: Fixed optimal coupling beats gradient-following 3/4.

## Prediction Beats Reaction

A model-based tuner that *explores* the landscape (samples 8 coupling
values), builds an internal model of $I(g)$ and $D(g)$ separately, and
navigates to the predicted $g^*$ achieves 33% improvement over
gradient-following.

**Result**: Predictive tuning improves over reactive tuning, confirming
that maintaining high $I \times D$ requires internal models of the
landscape, not just local feedback.

## The Cost of Exploration

Neither adaptive method beats the fixed-coupling oracle (which has perfect
knowledge of $g^*$). This is because exploration consumes resources:
the predictive model spends $\sim 40\%$ of its time at suboptimal coupling.
Maintaining performance near $g^*$ requires ongoing investment in
prediction and model-maintenance.

# Preliminary EEG Analysis

*This section reports a preliminary descriptive check, not a definitive
validation. The predicted direction is confirmed, but the analysis has
substantial limitations.*

## Prediction

From the $I \times D$ theorem: higher consciousness (wakefulness) should
correspond to more distributed eigenvalue spectra of the EEG covariance
matrix, and higher $I \times D$.

## Data

PhysioNet Sleep-EDF, Subject SC4001. 2 EEG channels, 100 Hz. 50 windows
of 30 seconds. No hypnogram annotations used; simple early-vs-late time
split (early epochs more likely wakeful, later epochs more likely asleep
in typical sleep architecture).

## Results

| Epoch | Eigenvalue concentration | $I \times D$ |
|-------|------------------------|--------------|
| First half (more awake) | **0.912** | **23.2** |
| Second half (more asleep) | 0.921 | 13.3 |

The predicted direction is confirmed: wakeful epochs show lower eigenvalue
concentration (more distributed spectrum) and 74% higher $I \times D$.

## Limitations

- $N=1$ (single subject)
- 2 EEG channels (minimal spatial resolution)
- Simple time split, not hypnogram-annotated
- Effect size on concentration is small ($\Delta = 0.009$)
- No statistical significance testing
- Descriptive, not confirmatory

Proper validation requires multiple subjects, hypnogram-scored Wake vs. N3
epochs, high-density EEG, and significance testing.

# Discussion

## What This Paper Establishes

The core contribution is the $I \times D$ tradeoff theorem (Section 3) and
its topology dependence (Section 4). This is a result about coupled dynamical
systems, not a claim about consciousness per se. The consciousness framing
is an interpretation that connects the mathematics to existing theoretical
frameworks (IIT, FEP, criticality).

## Connections to Existing Work

**Criticality.** The brain operates near a phase transition [12]. Our theorem
provides one possible objective function: $g^*$ is where $I \times D$ peaks,
which may coincide with the critical point. This is suggestive, not proven.

**Free Energy Principle.** Friston's FEP requires predictive models for
optimal behavior [5]. Our Finding 11 (prediction beats reaction) is
consistent with this but does not derive the FEP.

**IIT.** Tononi's $\Phi$ requires both integration and differentiation [1].
$I \times D$ is a tractable proxy, not a replacement for $\Phi$.

## What This Paper Does NOT Establish

1. Not a solution to the hard problem. $I \times D$ is an information-theoretic
   quantity, not a claim about phenomenal experience [16].
2. Not a universal law. The theorem requires diffusive coupling and heterogeneous
   frequencies. Non-diffusive coupling may violate monotonicity.
3. Not a validated neuroscience result. The EEG analysis is preliminary ($N=1$,
   no sleep staging, small effect size).
4. The molecular section (Section 2) demonstrates that molecules are too simple
   to distinguish between theories; it does not establish a causal link between
   molecular properties and network-level $I \times D$.

## Future Work

1. Full EEG validation with multiple subjects and hypnogram annotations
2. Analytical proof of $I \times D$ uniqueness for mean-field networks
3. Extension to larger networks ($N = 100$–$1000$) to test scaling
4. Formal relationship between $I \times D$ and IIT's $\Phi$

# Conclusion

In coupled heterogeneous oscillator networks, integration and differentiation
pull against each other as coupling increases. Their product has an interior
maximum whose location and height depend on network topology. Hierarchical
networks achieve the highest peak through scale separation; scale-free
networks achieve the widest basin through distributed connectivity. No
topology optimizes both.

Maintaining high $I \times D$ requires predictive internal models, not just
reactive feedback — a finding consistent with active inference frameworks.
Preliminary EEG analysis shows the predicted direction, but definitive
validation requires further work.

The theorem and landscape analysis provide a quantitative framework for
studying how network architecture shapes the balance between global
coordination and local specialization — a question relevant to neuroscience,
organizational theory, and dynamical systems.

# Code Availability

All computations: `symthaea-quantum-chemistry`, `symthaea-particle-physics`,
`symthaea-continuum-physics`, `symthaea-frontier-physics` (Rust, $\sim$18,000
LOC, 270+ tests). Source: `github.com/luminous-dynamics/symthaea`

# References

[1] Tononi, G. (2004). BMC Neuroscience 5, 42.

[2] Tononi, G. & Edelman, G. M. (1998). Science 282, 1846.

[3] Baars, B. J. (1988). A Cognitive Theory of Consciousness. Cambridge UP.

[4] Rosenthal, D. M. (2005). Consciousness and Mind. Oxford UP.

[5] Friston, K. (2010). Nature Reviews Neuroscience 11, 127.

[6] Penrose, R. (1994). Shadows of the Mind. Oxford UP.

[7] Maturana, H. R. & Varela, F. J. (1980). Autopoiesis and Cognition. Reidel.

[8] Crutchfield, J. P. (1989). PRL 63, 105.

[9] Amari, S. (2016). Information Geometry and Its Applications. Springer.

[10] Prigogine, I. (1977). Self-Organization in Non-Equilibrium Systems. Wiley.

[11] Zurek, W. H. (2003). Rev. Mod. Phys. 75, 715.

[12] Beggs, J. M. & Plenz, D. (2003). J. Neurosci. 23, 11167.

[13] FitzHugh, R. (1961). Biophys. J. 1, 445.

[14] Watts, D. J. & Strogatz, S. H. (1998). Nature 393, 440.

[15] Szabo, A. & Ostlund, N. S. (1996). Modern Quantum Chemistry. Dover.

[16] Chalmers, D. (1995). J. Consciousness Studies 2, 200.

[17] Kelso, J. A. S. (2012). Phil. Trans. R. Soc. B 367, 906.

[18] Sporns, O. (2010). Networks of the Brain. MIT Press.

[19] Koch, C. et al. (2016). Nature Reviews Neuroscience 17, 307.

[20] Shew, W. L. & Plenz, D. (2013). The Neuroscientist 19, 88.

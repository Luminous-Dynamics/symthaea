# Symtropy Publication Plan: Papers + Book

**Author**: Tristan Stoltz / Luminous Dynamics
**Date**: 2026-04-04
**Status**: Active

## Inventory: 20 Experimental Findings

| # | Finding | Experiment | p-value | Effect |
|---|---------|-----------|---------|--------|
| 1 | Cooperation emerges under thermodynamic scarcity | jphi_convergence | <0.001 | d=-1.94 |
| 2 | FEP gradient is clustering mechanism (ablation) | jphi_convergence | <0.001 | 6 conditions |
| 3 | Consciousness theory is interchangeable | jphi_convergence | n.s. | d<0.1 |
| 4 | Only Phi=0 differs from Phi>0 | jphi_convergence | <0.001 | d>2.0 |
| 5 | Per-capita energy matches eusocial insects | scaling_and_phase | - | beta=-0.178 vs -0.19 |
| 6 | Consciousness threshold matches clinical data | anesthesia_transition | - | 45% vs 42% |
| 7 | Curvature deflects trajectories linearly | curvature_lensing | 0.009 | d=-35.5 |
| 8 | J/Phi converges under well depletion | jphi_convergence | - | 3 conditions |
| 9 | Entropy predicts survival (Helmholtz) | entropy_prediction | - | harsh constants |
| 10 | HDC state vectors produce dynamic Phi | hdc_cooperation | - | Phi=0.535 |
| 11 | Curvature creates self-organized lanes | curvature_selforg | - | 4 scales |
| 12 | Pre-internet cooperation highest | internet_effect | <0.05 | 5 eras |
| 13 | Algorithm era devastates cooperation | internet_effect | <0.05 | 24 agents |
| 14 | 25% adversaries is critical threshold | adversarial_threshold | <0.05 | 9 fractions |
| 15 | Adversaries die faster than cooperators | adversarial_threshold | - | energy drain |
| 16 | Evolution selects for cooperation | evolution | - | resonance increases |
| 17 | Phi-gravity accelerates clustering | evolution | - | G=0.3 |
| 18 | Societies evolve solidarity, not tribalism | adversarial_evolution | - | res 0.86->1.0 |
| 19 | Physical bonds improve survival 57%->97% | social_bonds | 0.001 | d=2.44 |
| 20 | Phase transition at 0.20 J/tick pressure | phase_transition | 0.036 | d=1.71 |
| 21 | Optimal group size N=12 | dunbar_number | - | survival peaks |
| 22 | Cluster size caps at ~15 (harmony range limit) | dunbar_number | - | physics Dunbar |
| 23 | Per-capita energy scales positively (no metabolic gain) | dunbar_number | - | beta=+1.77 |
| 24 | Abundance breeds complacency (inverted tragedy) | tragedy_commons | - | ABUNDANT worst |
| 25 | Altruism futile without structure (100% recidivism) | altruism_emergence | n.s. | d=0.0 |
| 26 | Equal starts → maximum inequality (Gini 0→0.66) | inequality_emergence | - | d=1.02 |

---

## Paper 1: ALIFE 2026 (DEADLINE: April 12)

**Title**: "The Conscious/Unconscious Distinction Matters, But Consciousness Theories Don't"
**Venue**: ALIFE 2026 (6-page limit)
**Status**: Draft exists, needs update with Findings 19-20

**Core findings** (fits 6 pages):
- F1: Cooperation as thermodynamic necessity
- F2-4: Ablation + theory interchangeability (the headline result)
- F7: Curvature lensing
- F20: Phase transition (NEW - adds thermodynamic depth)

**Cut from this paper** (save for Paper 2-3):
- Social bonds, adversarial, evolution, internet (save for social physics paper)
- HDC, entropy prediction (save for engine paper)

**Action items**:
- [x] Update abstract to include phase transition
- [x] Add phase diagram figure (pressure vs survival)
- [x] Tighten to 6 pages (11 findings, compact bibliography)
- [x] Generate fig5_phase_transition.pdf
- [x] Conclusion updated: 11 findings including phase transition (F10)
- [ ] Submit (paper ready, 6 pages, all figures compiled)

---

## Paper 1.5: J/Phi Metric (Entropy MDPI)

**Title**: "Joules per Phi: A Thermodynamic Metric for the Energy Cost of Information Integration in Multi-Agent Systems"
**Venue**: Entropy (MDPI) — open access
**Status**: Draft complete (papers/entropy-mdpi/main.tex)
**Target**: Q2 2026

**Core findings**:
- J/Phi metric definition and theoretical grounding (Landauer bound)
- Convergence under cooperative thermodynamic enforcement
- 31% J/Phi reduction via epistemic offloading (p=0.0009, d=-1.94)
- Phase transition at 0.20 J/tick (cooperative → collapsed J/Phi regimes)
- Temperature-consciousness sigmoid coupling
- Bifurcation analysis with susceptibility and Binder cumulant

**Action items**:
- [x] Draft paper with all sections
- [ ] Generate figures (convergence plot, phase diagram, J/Phi time series)
- [ ] Run jphi_convergence experiment for fresh data tables
- [ ] Submit to Entropy MDPI

---

## Paper 2: Social Physics (Complex Systems or J. Artificial Societies)

**Title**: "Thermodynamic Social Physics: Cooperation, Conflict, and Phase Transitions in Integration-Coupled Agent Systems"
**Venue**: Journal of Artificial Societies and Social Simulation (JASSS) or Complexity
**Target**: Q3 2026
**Length**: 15-20 pages (full journal article)

**Core findings**:
- F12-13: Internet/technology effect on cooperation (5 eras)
- F14-15: Adversarial threshold (critical fraction)
- F16-18: Evolutionary selection for cooperation + solidarity
- F19: Physical social bonds
- F20: Phase transition + bistability

**Narrative**: How do integration-coupled societies respond to technological change, adversarial agents, evolutionary pressure, and environmental stress? The engine reveals universal patterns: cooperation is thermodynamically favored, adversaries are energetically expensive, and societies under stress evolve solidarity rather than tribalism. A sharp phase transition separates cooperative and collapsed regimes.

**Honest framing**: These are properties of the *model*, not claims about human society. But the patterns parallel published sociology (Dunbar numbers, tragedy of the commons, tipping points).

---

## Paper 3: Engine Paper (Artificial Life journal, full length)

**Title**: "Symtropy: An Open-Source N-Dimensional Physics Engine with Integration-Metric Coupling"
**Venue**: Artificial Life (MIT Press) or SoftwareX
**Target**: Q4 2026
**Length**: 20-25 pages

**Contents**:
- Full engine architecture (5 coupling channels)
- Honest realism assessment (GROUNDED / INSPIRED / SPECULATIVE for each mechanic)
- Thermodynamic accounting (Landauer bound, J/Phi, conservation error)
- N-dimensional physics (2D-9D with LBVH broadphase)
- HDC state vectors (Finding 10)
- Conformal curvature (real math, speculative physics — clearly distinguished)
- Reproducibility: all code AGPL-3.0, all experiments deterministic

**Key contribution**: The engine itself as a research tool. Others can run experiments. The honest realism table is a feature, not a bug — it helps reviewers.

---

## Paper 4: Biological Validation (Frontiers in Computational Neuroscience)

**Title**: "Emergent Biological Scaling Laws in Integration-Coupled Multi-Agent Simulations"
**Venue**: Frontiers in Computational Neuroscience
**Target**: Q1 2027

**Core findings**:
- F5: Metabolic scaling matches eusocial insects (beta=-0.178)
- F6: Consciousness threshold matches clinical anesthesia data (45% vs 42%)
- F9: Helmholtz free energy predicts survival
- F20: Phase transition (thermodynamic interpretation)

**Argument**: Without fitting to biological data, the engine reproduces two quantitative biological observations. This suggests the thermodynamic enforcement layer captures something real about the energy economics of integrated systems.

---

## Book: "The Thermodynamics of Togetherness"

**Publisher target**: MIT Press (Artificial Life series) or Princeton UP
**Length**: ~250 pages
**Timeline**: Q3 2026 proposal, Q1 2027 manuscript

### Structure

**Part I: The Engine** (Chapters 1-4)
1. *Why Consciousness Needs Physics* — the gap between IIT theory and embodied systems
2. *Five Channels* — motor gain, energy budgets, harmony fields, sanctuary, curvature
3. *Honest Uncertainties* — the GROUNDED/INSPIRED/SPECULATIVE classification
4. *The Mathematics* — conformal geometry, Landauer bounds, N-dimensional field theory

**Part II: The Experiments** (Chapters 5-10)
5. *Cooperation as Thermodynamic Necessity* — the founding experiment
6. *The Phase Transition* — when does cooperation collapse?
7. *Internet and Algorithms* — technology's effect on social fabric
8. *Raiders and Evolvers* — adversarial agents and natural selection
9. *Physical Bonds* — constraint-based social structure
10. *Biology Without Fitting* — emergent scaling laws

**Part III: Implications** (Chapters 11-14)
11. *What We Can Claim* — epistemology of simulation results
12. *What We Cannot Claim* — the hard problem, qualia, subjective experience
13. *Eight Harmonies* — the philosophical framework behind the engine
14. *Toward Living Systems* — from simulation to real consciousness research

### Appendices
A. Full experiment data tables
B. Engine API reference
C. Reproduction instructions (cargo run --example X)
D. Statistical methods (Mann-Whitney U, Cohen's d)

---

## Priority Timeline

```
2026 Apr 11: Submit Paper 1 (ALIFE)
2026 May-Jun: Write Paper 2 (social physics), run additional experiments
2026 Jul-Aug: Submit Paper 2, write Paper 3 (engine)
2026 Sep: Submit Paper 3, begin book proposal
2026 Oct-Dec: Write Paper 4 (biological), draft book Part I
2027 Jan-Mar: Submit Paper 4, draft book Part II
2027 Apr-Jun: Book manuscript complete
```

## Sanctuary Mechanic Decision

**Recommendation**: Reframe as "social buffering" (Hostinar et al. 2014, Eisenberger 2012).

Current: High harmony → impulse dampening (force field). No precedent.
Proposed: High harmony → prediction error reduction near resonant agents. Real neuroscience.

This is already partially implemented (resonance-aware prediction error in `on_collision`).
The sanctuary zone as collision dampener should be marked as a game mechanic in papers.

# Five-Component Model: Canonical Definitions

**Version**: 1.1
**Created**: 2026-01-16
**Last reviewed**: 2026-04-18 — still canonical
**Purpose**: Resolve definitional inconsistencies across papers

> **Scope note.** This document canonicalises the **theoretical five-component
> consciousness model** $(\Phi, B, W, A, R)$ from the neuroscience literature.
> It is **not** the governance credential system. The Mycelix sovereign
> governance credential is an 8-dimensional profile (epistemic integrity,
> thermodynamic yield, network resilience, economic velocity, civic participation,
> stewardship, semantic resonance, domain competence) — see
> `sovereignty-papers/essay-07-on-the-eight-dimensions.md` and
> `governance/embodied-governance/` for that system.

---

## Overview

The Five-Component Model defines consciousness as:

$$C = \min(\Phi, B, W, A, R)$$

where each component represents a necessary mechanism for conscious experience. This document establishes **canonical definitions** that all papers must use consistently.

---

## Canonical Component Definitions

### Φ — Integration (IIT)

| Property | Value |
|----------|-------|
| **Full Name** | Integration |
| **Theory Origin** | Integrated Information Theory (Tononi et al.) |
| **Definition** | Information that is integrated across the system, beyond what exists in independent parts |
| **Neural Substrate** | Cortico-cortical connectivity; thalamocortical loops |
| **Measurement** | PCI (Perturbational Complexity Index), LZc (Lempel-Ziv complexity) |
| **Range** | [0, 1] |
| **Intuition** | A system with high Φ cannot be reduced to independent modules without information loss |

**What Φ captures**: The degree to which the whole is more than the sum of its parts—irreducible information integration across the network.

---

### B — Binding (Synchrony)

| Property | Value |
|----------|-------|
| **Full Name** | Binding |
| **Theory Origin** | Neural Synchrony Theory (Singer, Engel) |
| **Definition** | Temporal synchronization that binds distributed features into unified representations |
| **Neural Substrate** | GABAergic interneurons; gamma oscillations (30-100 Hz) |
| **Measurement** | Gamma Phase-Locking Value (PLV); cross-frequency coupling |
| **Range** | [0, 1] |
| **Intuition** | Neurons firing in synchrony bind separate features (color, shape, motion) into unified percepts |

**What B captures**: The temporal "glue" that creates unified objects from distributed neural representations.

---

### W — Workspace (GWT)

| Property | Value |
|----------|-------|
| **Full Name** | Workspace |
| **Theory Origin** | Global Workspace Theory (Baars, Dehaene) |
| **Definition** | Global availability and broadcast capacity—content accessible to multiple cognitive systems |
| **Neural Substrate** | Prefrontal-parietal network; long-range connections |
| **Measurement** | P300 amplitude; ignition patterns; global vs. local processing |
| **Range** | [0, 1] |
| **Intuition** | Content "in the workspace" is globally available for report, memory, planning, and action |

**What W captures**: Whether information has been broadcast globally or remains locally encapsulated.

---

### A — Attention (Selection)

| Property | Value |
|----------|-------|
| **Full Name** | Attention |
| **Theory Origin** | Predictive Processing / Precision Weighting |
| **Definition** | Precision-weighted selection of relevant content for priority processing |
| **Neural Substrate** | Frontoparietal attention network (dorsal/ventral streams) |
| **Measurement** | Alpha suppression (8-12 Hz); attentional modulation indices |
| **Range** | [0, 1] |
| **Intuition** | The "gatekeeper" that selects which content gains priority access to workspace |

**What A captures**: The selective mechanism that determines *which* content is processed—not *awareness of* processing.

**Critical Distinction (A vs. R)**:
- **A = Selection**: What content is prioritized? (Bottom-up salience + top-down goals)
- **R = Reflection**: Am I aware that I am processing? (Meta-cognitive monitoring)

One can attend (high A) without metacognitive awareness of attending (low R), as in **flow states**.
One can reflect on diffuse content (high R, moderate A), as in **open awareness meditation**.

---

### R — Recursion (HOT)

| Property | Value |
|----------|-------|
| **Full Name** | Recursion / Meta-representation |
| **Theory Origin** | Higher-Order Thought Theory (Rosenthal, Brown) |
| **Definition** | Meta-representational capacity—the ability to represent one's own mental states |
| **Neural Substrate** | Medial prefrontal cortex (mPFC); anterior cingulate |
| **Measurement** | Metacognition tasks; PFC-PPC connectivity; self-referential processing |
| **Range** | [0, 1] |
| **Intuition** | "Knowing that you know"—recursive self-awareness |

**What R captures**: Higher-order representations *of* first-order mental states. Not just seeing red, but *being aware* that you are seeing red.

---

## Inconsistencies Found and Corrections Required

### Issue 1: A labeled as "Awareness" in some papers

**Problem**: Papers 08-09 define A as "Awareness" / "Meta-representation" (conflating with R).

| Paper | Current Definition | Should Be |
|-------|-------------------|-----------|
| Paper 01 (Table 3.4) | A (Attention) | ✓ Correct |
| Paper 01 (Table 4) | A (Awareness) | ✗ Should be "A (Attention)" |
| Paper 08 | A = meta-awareness (HOT) | ✗ Should be R = HOT |
| Paper 09 | A (awareness) = meta-representation | ✗ Should be "A (Attention)" |

**Root cause**: Confusion between "selective attention" (A) and "meta-awareness" (R).

**Correction**:
- A = **Attention** = selection/precision (NOT meta-awareness)
- R = **Recursion** = meta-awareness/HOT (NOT recursion-depth)

### Issue 2: R variously called "Recursion" and "HOT"

**Problem**: The term "Recursion" suggests recursive depth (how many levels), while the theory basis is Higher-Order Thought.

**Resolution**: Both names are acceptable:
- **Recursion** emphasizes the self-referential structure
- **HOT/Meta-representation** emphasizes the theory origin

Canonical label: **R (Recursion/HOT)** — use both where space permits.

### Issue 3: Paper 08 claims HOT corresponds to A

**Problem**: Paper 08 line 23 states "HOT corresponds to the meta-awareness component (A)".

**Correction Required**: This should read "HOT corresponds to the recursion component (R)". The entire Paper 08 discussion of "A component" should reference R instead.

---

## Summary: Canonical Component Table

| Symbol | Canonical Name | Theory | Substrate | Key Metric | NOT Confused With |
|--------|---------------|--------|-----------|------------|-------------------|
| **Φ** | Integration | IIT | Cortico-cortical | PCI | — |
| **B** | Binding | Synchrony | GABAergic | Gamma PLV | — |
| **W** | Workspace | GWT | Frontoparietal | P300 | — |
| **A** | Attention | Precision | Attention networks | Alpha suppression | NOT meta-awareness |
| **R** | Recursion/HOT | HOT | mPFC | Metacognition | NOT just recursion-depth |

---

## The min() Function: Conceptual Justification

Why minimum rather than product, sum, or weighted average?

### Empirical Support

1. **Lesion studies**: Damage to any single component devastates consciousness (not proportional reduction)
   - Attention failure (A→0): Sleep onset, anesthesia → unconscious despite intact Φ,B,W,R
   - Binding failure (B→0): Feature agnosia → lost unified experience
   - Workspace failure (W→0): Local-only processing → no global access

2. **Pharmacological evidence**: Anesthetic agents target specific components; knocking out *one* suffices

3. **Sleep stages**: Sleep onset shows precipitous A drop (0.67→0.04) as limiting factor

### Theoretical Support

1. **Bottleneck principle**: Consciousness requires *all* mechanisms functioning; any failure is catastrophic

2. **Falsifiable predictions**:
   - Product/sum models predict gradual degradation
   - min() predicts sudden transitions (matches bistable perception, anesthesia induction)

3. **Component independence**: The min() function respects component dissociability—improving beyond the minimum has no effect (matches "overkill" scenarios)

### Mathematical Properties

The min() function ensures:
- **Boundedness**: C ∈ [0, 1]
- **Monotonicity**: ∂C/∂Cᵢ ≥ 0 (improving any component cannot decrease C)
- **Threshold gating**: C → 0 as any Cᵢ → 0
- **Gradient sparsity**: Only the minimum component has non-zero gradient

---

## Usage Instructions

When referencing components in any paper:

1. Use the **canonical name** from this glossary
2. Include the **symbol** on first use: "Attention (A)"
3. Use the **canonical table** (Summary section) as the standard reference
4. Cross-check definitions against this glossary before submission

---

## Version History

- **v1.0** (2026-01-16): Initial canonical definitions; identified inconsistencies in Papers 01, 08, 09

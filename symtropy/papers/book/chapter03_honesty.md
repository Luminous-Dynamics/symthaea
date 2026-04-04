# Chapter 3: What's Real and What Isn't

*In which we audit our own engine for scientific honesty and find that half of it is invention.*

---

## The Classification

Every mechanic in a simulation engine has a relationship to reality. Some implement well-established physics. Some draw loose analogy from real systems. Some have no precedent whatsoever and exist because a programmer thought they would be interesting.

Most simulation papers do not distinguish between these categories. We do.

We classify every mechanic in Symtropy as one of three types:

**GROUNDED**: Implements established physical or mathematical laws. Can be validated against nature. Uses published constants. If the mechanic is wrong, a physicist can demonstrate how.

**INSPIRED**: Draws loose analogy from a real system but does not faithfully implement its equations. Captures the spirit of a scientific idea without the rigor. If the mechanic is wrong, the error is in the analogy, not the math.

**SPECULATIVE**: Has no published precedent. Invented for this simulation. If the mechanic is wrong, there is nothing to compare it against — it was never claimed to be right.

## The Audit

| Mechanic | Math | Physics | Classification |
|----------|------|---------|---------------|
| Thermodynamics (U, T, S, F) | Correct | Correct | **GROUNDED** |
| Landauer bound (2.87e-21 J/bit) | Exact | Published | **GROUNDED** |
| Conformal geometry (g_ij = e^{2σ}δ_ij) | Correct Riemannian | No physics law couples Φ to spacetime | **GROUNDED** math, **SPECULATIVE** physics |
| FEP gradient descent | Approximation | Loose Friston analogy | **INSPIRED** |
| Harmony field (1/r^{D-1} falloff) | Correct field theory form | No physical field exists | **INSPIRED** |
| Motor gain (4-tier safety) | — | No neuroscience precedent | **INSPIRED** |
| Sanctuary zones (impulse dampening) | — | No "harmony force field" exists | **SPECULATIVE** |
| Φ-gravity (F = GΦ₁Φ₂/r²) | Correct Newtonian form | Tononi never proposed this | **SPECULATIVE** |
| Dimensional leakage | — | Randall-Sundrum misapplied | **SPECULATIVE** |
| Joules-per-Phi metric | Novel | Novel | **SPECULATIVE** (but useful) |

Five of nine mechanics are either INSPIRED or SPECULATIVE. Only the core thermodynamics and the conformal geometry mathematics are fully GROUNDED. The engine is, by this accounting, approximately half invention.

## What This Means for the Findings

Not all findings depend on all mechanics.

The cooperation emergence findings (Chapters 5-6) depend only on the GROUNDED thermodynamic layer (energy budgets, maintenance costs, Helmholtz free energy) and the INSPIRED FEP gradient. They do not use sanctuary zones, Φ-gravity, dimensional leakage, or conformal curvature. The core thesis — cooperation as thermodynamic necessity — rests on the most grounded components of the engine.

The curvature lensing finding (Finding 8: p=0.009) depends on the conformal geometry, which is mathematically GROUNDED but physically SPECULATIVE. The trajectory deflection is real Riemannian geometry. The claim that harmony field energy curves space is pure invention. Finding 36 confirmed that this curvature has zero effect on cooperation — it deflects trajectories but doesn't change social outcomes. The conformal curvature is a mathematical demonstration, not a social mechanism.

The Φ-gravity and sanctuary mechanics are used in a few early experiments but are not invoked in any finding reported in this book. They remain as engine features for potential game design use, but we make no scientific claims about them.

## Why We're Telling You This

The standard practice in simulation papers is to present the engine as a unified whole and let the reader assume that all components are equally grounded. This is misleading. A reader who sees "conformal Riemannian curvature" next to "thermodynamic energy budgets" might reasonably assume both are physics. Only one is.

We classify our own mechanics because:

1. **Reviewers will find what you don't disclose.** An unreported SPECULATIVE mechanic discovered by a reviewer becomes a credibility problem. A disclosed SPECULATIVE mechanic becomes an honest design choice.

2. **Other builders need to know.** If someone extends this engine, they should know which parts are load-bearing science and which are decorative metaphor. Building on a SPECULATIVE foundation without knowing it is how bad science propagates.

3. **The findings are stronger with the distinction.** When we say "cooperation emerges from thermodynamics," we can now demonstrate that this claim rests only on GROUNDED mechanics. The SPECULATIVE components are separable and inert. The thesis does not depend on them.

## The Proposed Standard

We propose that every simulation engine paper include a realism classification table. The format is simple:

| Mechanic | Classification | Evidence |
|----------|---------------|----------|
| [name] | GROUNDED / INSPIRED / SPECULATIVE | [citation or "none"] |

This table should appear in the methods section, before any results. It sets expectations and demonstrates scientific maturity.

The classification requires judgment. Reasonable people may disagree about whether a particular mechanic is INSPIRED or SPECULATIVE — the FEP gradient is a borderline case. The point is not to achieve perfect classification but to *force the exercise of classification*. A builder who must categorize each mechanic will think more carefully about what they're building.

## The Cost of Honesty

Disclosing that half your engine is speculative is uncomfortable. It invites the question: "If half of it is made up, why should I trust any of it?"

The answer is in the separation. The findings reported in this book depend on the GROUNDED half. The SPECULATIVE half is present but inert — it exists in the codebase but does not contribute to the cooperation, phase transition, or social dynamics results. It is engine feature, not scientific claim.

This separation is itself a finding. We *tested* whether the speculative mechanics matter (Finding 36: curvature irrelevant, Finding 29: information irrelevant, Finding 34: memory irrelevant). The negative results are as important as the positive ones — they demonstrate that the engine's core behavior comes from its grounded components, not from the speculative additions that make it look impressive.

If we had not disclosed and tested the classification, we would not know this. The honesty audit strengthened the science.

---

*Next: Chapter 4 — Scaling and Performance: From Twenty Agents to One Thousand*

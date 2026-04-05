# Chapter 2: Five Channels

*In which we describe how an integration metric couples to rigid body physics, and why each coupling exists.*

---

## The Architecture

Symtropy is a rigid body physics engine written in Rust, supporting 2D through 9D simulation with LBVH broadphase collision detection, GJK/EPA narrowphase, and PBD constraint solving. Agents are spheres with position, velocity, mass, and linear damping. Physics runs at 64 Hz with semi-implicit Euler integration and LTC exponential damping (frame-rate independent, never reverses velocity).

On top of this physics layer sits the consciousness-physics coupling — five bidirectional channels through which an integration metric (Φ) modulates rigid body dynamics. These channels are the engine's novel contribution. Each represents a different hypothesis about how internal state could affect physical behavior.

## Channel 1: Motor Gain

Φ modulates force authority through a four-tier safety system:

| Tier | Φ Range | Motor Gain | Interpretation |
|------|---------|-----------|----------------|
| Green | > 0.6 | 100% | Full authority |
| Yellow | 0.3 – 0.6 | 60% | Reduced output |
| Orange | 0.1 – 0.3 | 30% | Minimal control |
| Red | < 0.1 or collapsed | 0% | No motor output |

An agent at Red tier cannot move — it is functionally paralyzed. This creates a survival pressure: maintaining Φ above 0.1 is necessary for locomotion, and locomotion is necessary for reaching energy wells and cooperation partners.

Classification: **INSPIRED**. The four-tier structure borrows from NRC nuclear safety protocols. No neuroscience publication proposes this specific Φ-to-motor mapping. The general principle — that integrated information modulates behavioral output — has loose support in motor control literature (Adams, Shipp & Friston 2013), but the discrete tiers are invention.

## Channel 2: Energy Budget

Every tick, agents pay a maintenance cost proportional to their integration level:

```
cost = base_maintenance × (1.0 + Φ × 0.5)
```

Higher Φ costs more energy. This is the thermodynamic cost of integration — the engine's analogue of the brain's metabolic burden (Raichle & Gusnard 2002). The brain consumes 20% of the body's energy while comprising 2% of its mass; maintaining high integration is expensive.

Energy regenerates through three sources: ambient recovery (slow, ~0.005 J/tick), energy wells (moderate, ~0.12 J/tick when standing on one), and harmony resonance (variable, depends on partner compatibility and proximity).

Classification: **GROUNDED** for the thermodynamic accounting (U, T, S, F tracked per agent, Landauer bound referenced). **INSPIRED** for the Φ-proportional cost (the 0.5 coupling constant is empirical, not derived).

## Channel 3: Harmony Field Friction

Each agent emits a harmony field — eight scalar values representing activation along the Eight Harmonies (Stillness, Play, Craft, Justice, Curiosity, Celebration, Kinship, Stewardship). The field falls off as 1/r^(D-1) with Plummer softening (ε = 1.0) to prevent singularities.

When two agents' harmony fields overlap, the resonance (dot product of their activation vectors, normalized) modulates the collision response. High resonance reduces prediction error from collisions; low resonance increases it. This means collisions between compatible agents are less "surprising" — they produce less behavioral disruption.

Classification: **INSPIRED**. The 1/r^(D-1) falloff is correct field theory. The harmony-as-field-source concept draws loose analogy from McFadden's CEMI field theory (2020), which proposes that the brain's electromagnetic field is conscious. The specific eight-channel structure has no physics precedent.

## Channel 4: Sanctuary Zones

When an agent's Sacred Stillness activation exceeds 0.6, total harmony energy exceeds 2.0, and Φ exceeds 0.3, a sanctuary zone forms — a spatial region where collision impulses are dampened by up to 90%.

In forty-five findings, no experiment ever activated a sanctuary zone. The Φ values produced by the consciousness equation (~0.06-0.09) never reach the 0.3 threshold. The mechanism exists in the codebase but is functionally inert.

Classification: **SPECULATIVE**. No physical system creates a collision-dampening force field through collective harmonization. The closest real-world analogue — social buffering (Hostinar et al. 2014) — operates through stress hormone modulation, not impulse dampening. We retain the mechanism for game design purposes but make no scientific claim about it.

## Channel 5: Conformal Curvature

The harmony field energy parameterizes a conformal Riemannian metric:

```
g_ij(x) = e^{2σ(x)} δ_ij,  where σ(x) = κ × E_harmony(x)
```

This produces geodesic corrections to agent trajectories — harmony "wells" in the field curve the simulation space, deflecting nearby agents. The mathematics is correct (Christoffel symbols, geodesic equation, Ricci scalar all properly derived from the conformal factor). The physics is speculative — no law of nature couples integration metrics to spacetime curvature.

Finding 8 demonstrated measurable trajectory deflection (p = 0.009, d = -35.5) at curvature scale κ = 0.05. Finding 36 demonstrated that this deflection has zero effect on cooperation (d = 0.006). The curvature is mathematically real and socially irrelevant.

Classification: **GROUNDED** mathematics (Carroll 2004, Wald 1984). **SPECULATIVE** physics.

## The Coupling That Matters

Of the five channels, only two contribute to the cooperation findings reported in this book:

1. **Energy budget** (Channel 2): creates the thermodynamic pressure that drives cooperation
2. **Harmony field resonance** (embedded in Channel 3): provides the mutual benefit that sustains cooperation

Motor gain (Channel 1) affects collapsed agents but not active ones (Φ rarely drops below 0.3 in practice). Sanctuary (Channel 4) never activates. Curvature (Channel 5) deflects trajectories without affecting social outcomes.

The engine has five channels, but the book runs on two. We report all five for completeness and honesty — and because the inert channels are themselves a finding. Three coupling mechanisms that seemed theoretically important turned out to be empirically irrelevant. Only the thermodynamic core — energy pressure and mutual benefit — drives the social dynamics that fill the remaining chapters.

---

*Next: Chapter 3 — What's Real and What Isn't*

# Chapter 12: Higher Dimensions and the Geometry of Cooperation

*In which cooperation survives the third dimension, curvature proves beautiful but useless, and we learn that social physics is dimension-independent.*

---

## The 3D Test

Every experiment in this book runs in 2D. The engine supports 2D through 9D — the physics, the harmony fields, the FEP gradient, and the spatial hash are all dimension-generic (using Rust's `const D: usize` generics). But claiming N-dimensional support without testing it is an empty claim.

Finding 30 compared identical experiments in 2D and 3D: 20 agents, 8,000 ticks, 8 seeds, same harmony profiles, same thermodynamic constants.

| Dimension | Survival | Clustering | Cooperation |
|-----------|----------|------------|-------------|
| 2D | 15.2/20 | 5.77 | 956K |
| 3D | 13.4/20 | 14.71 | 617K |

Survival is not significantly different (p=0.32, d=0.27). Cooperation drops 35% (d=1.11, large effect). Clustering loosens 2.5×.

The mechanism is volume. A 100×100 2D arena has 10,000 square units. A 100×100×100 3D arena has 1,000,000 cubic units — 100× the volume. Agents are 100× more dilute. The harmony range (40 units) covers a sphere of volume 268,000 cubic units in 3D vs a circle of 5,027 square units in 2D — the 3D interaction volume is 53× larger, but the arena volume grew 100×, so the probability of a random agent being within harmony range is roughly halved.

Agents cooperate less in 3D because they encounter each other less frequently. But the FEP gradient still drives them toward wells and partners, and the cooperation that does occur is still sufficient for survival. The social physics is dimension-independent; the social geometry is not.

## Curvature: Beautiful Mathematics, Zero Social Effect

The conformal curvature system (Chapter 2, Channel 5) implements correct Riemannian geometry. The conformal factor σ(x) = κ × E_harmony(x) curves the simulation space near high-harmony regions. The geodesic correction:

```
a = -2(v · ∇σ)v + |v|²∇σ
```

produces measurable trajectory deflection. Finding 8 demonstrated this: a test body launched past a stationary high-Φ source deflects proportionally to the curvature scale κ, with Mann-Whitney U = 0.0, p = 0.009, Cohen's d = -35.5. The deflection is real, large, and statistically overwhelming.

Finding 36 tested whether this deflection improves cooperation. Three conditions: flat (κ=0), low curvature (κ=0.01), and high curvature (κ=0.05). Result: d = 0.006 for high curvature vs flat. Survival: 11.7 vs 11.7. Clustering: 6.41 vs 6.90. Cooperation events: identical within noise.

The curvature creates "harmony wells" in the metric — regions where space is compressed, making geodesics curve toward high-harmony areas. But the FEP gradient already drives agents toward these same areas. The curvature adds a second-order correction to trajectories that are already pointing in the right direction. It's like adding a gentle breeze to a river current — technically real, practically invisible.

This is an honest negative result. We invested 85 lines of tensor calculus in a system that produces the strongest p-value in the book (0.009) for trajectory deflection but contributes nothing to the social dynamics that the book is about. The mathematics is impeccable. The social relevance is zero.

For the GROUNDED/INSPIRED/SPECULATIVE classification (Chapter 3), this is the clearest case: GROUNDED mathematics producing SPECULATIVE physics with zero empirical social consequence. We keep it in the engine because it's correct and beautiful. We report it here because honesty requires admitting when beauty is useless.

## What Dimension Teaches

The 3D and curvature results together make a point about the engine's cooperation mechanism: it is *topological*, not *geometric*.

Cooperation depends on connectivity (can agents reach each other?) not on metric (what shape is the space?). In 2D and 3D, agents that can reach resonant partners cooperate; agents that can't, don't. The specific geometry — flat vs curved, 2D vs 3D — changes how quickly agents find each other but not whether they cooperate once they do.

This explains why the gradient topology experiment (Finding 41) found cooperation robust to structural changes: fragmented perception changed the *number* of clusters (5.9 vs 1.2) but not the *fact* of clustering. Obstacles, moving wells, and short sight all changed geometry without changing topology — agents could still reach *some* partners, and some was enough.

The only topology change that kills cooperation is disconnecting the social graph entirely — making it impossible for any agent to reach any resonant partner. This requires either zero harmony range (Finding 33: range is CRITICAL) or zero resonance regeneration (Finding 32: NO_REGEN kills 80%). Every other structural modification preserves the topology and therefore preserves cooperation.

Cooperation is a topological invariant of the engine. Geometry is decoration.

---

*Next: Chapter 13 — What Simulations Can and Cannot Tell Us*

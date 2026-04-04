# Paper 3: The Symtropy Engine

**Title**: "Symtropy: An Open-Source N-Dimensional Physics Engine with Integration-Metric Coupling"
**Target**: Artificial Life (MIT Press) or SoftwareX
**Length**: 20-25 pages

## Abstract (draft)

We present Symtropy, an open-source physics engine where an integration metric (Φ) modulates rigid body dynamics through five bidirectional coupling channels. The engine implements dimension-generic (2D-9D) physics with LBVH broadphase, GJK/EPA narrowphase, PBD+impulse constraint solving, LTC exponential damping, and conformal Riemannian curvature driven by harmony field energy. We provide an honest assessment of each mechanic's scientific grounding, classifying them as GROUNDED (real physics: thermodynamics, Landauer bound, conformal geometry), INSPIRED (loose analogy: FEP gradient, harmony fields), or SPECULATIVE (no precedent: Φ-gravity, sanctuary zones). The engine's thermodynamic accounting layer tracks temperature, entropy, and Helmholtz free energy per agent, achieving <6% energy conservation error. A novel J/Φ metric measures the energetic cost of integration. Across 31 experiment files and 3,000+ simulation runs, the engine produces 20+ emergent findings including cooperation as thermodynamic necessity, metabolic scaling matching biological data, and phase transitions in social organization. All code is AGPL-3.0 with deterministic experiments.

## Structure

### 1. Introduction
- Motivation: consciousness-coupled game engine
- Design goals: real physics (not UI overlay), N-dimensional, open-source
- Contribution: 5 coupling channels, honest realism assessment

### 2. Physics Foundation
- 2.1 Rigid body dynamics (SVector<D>, tensor inertia)
- 2.2 LBVH broadphase (Morton codes, Karras 2012)
- 2.3 GJK/EPA narrowphase (Gram-Schmidt ND normals)
- 2.4 Constraint solving (PBD + velocity correction)
- 2.5 LTC exponential damping (frame-rate independent)

### 3. Integration-Metric Coupling
- 3.1 Channel 1: Motor gain (Φ → force authority)
- 3.2 Channel 2: Energy budget (Φ maintenance cost)
- 3.3 Channel 3: Harmony field friction modulation
- 3.4 Channel 4: Sanctuary impulse dampening
- 3.5 Channel 5: Conformal curvature (g_ij = e^{2σ}δ_ij)

### 4. Thermodynamic Layer
- 4.1 Energy budget (U, T, S, F = U - TS)
- 4.2 Landauer bound and cognitive operation cost
- 4.3 J/Φ metric (novel)
- 4.4 Conservation accounting and error measurement

### 5. Harmony Field Theory
- 5.1 8-channel activation vector
- 5.2 Resonance metric (dot product)
- 5.3 1/r^(D-1) falloff with Plummer softening
- 5.4 Conformal factor σ(x) and geodesic correction

### 6. Agent Decision Making
- 6.1 FEP gradient descent
- 6.2 Φ-gravity (speculative)
- 6.3 HDC state vector substrate (16,384D)

### 7. Honest Realism Assessment
**This is the key contribution: transparent about what's real and what isn't.**

| Mechanic | Math | Physics | Precedent |
|----------|------|---------|-----------|
| Thermodynamics | GROUNDED | GROUNDED | Landauer 1961 |
| Conformal curvature | GROUNDED | SPECULATIVE | Carroll 2004 (math only) |
| FEP gradient | INSPIRED | INSPIRED | Friston 2010 (loose) |
| Harmony fields | INSPIRED | SPECULATIVE | McFadden 2020 (controversial) |
| Motor gain | INSPIRED | SPECULATIVE | NRC analogy |
| Sanctuary | SPECULATIVE | SPECULATIVE | None |
| Φ-gravity | SPECULATIVE | SPECULATIVE | None |
| Dimensional leakage | SPECULATIVE | SPECULATIVE | Randall-Sundrum (misapplied) |

### 8. Experimental Validation
- 8.1 Scorecard: baseline engine metrics
- 8.2 Conservation error reduction (672% → 5.1%)
- 8.3 Curvature lensing (p=0.009)
- 8.4 Biological scaling emergence (beta=-0.178)

### 9. Limitations
- Not a consciousness theory — it's a coupling architecture
- Constants are empirically tuned, not derived
- No learning, no communication, no memory
- Sanctuary and Φ-gravity have zero precedent

### 10. Conclusion
- Contribution: transparent, reproducible, open-source
- Future: learning agents, 3D visualization, real robotics coupling

## Figures (planned)
1. Engine architecture diagram (5 coupling channels)
2. LBVH broadphase visualization
3. Conformal curvature geodesic deflection
4. Conservation error timeline
5. Realism classification table (main contribution)

## Appendices
A. Full ThermodynamicConstants parameter table
B. Harmony channel definitions (Eight Harmonies)
C. Statistical methods
D. Experiment reproduction commands

# The Thermodynamic Cost of Integration: Emergent Cooperation in a Φ-Physics Engine

## Paper Outline — ALIFE 2026

### Abstract (~150 words)

We present Symtropy, a novel physics engine where an integration metric (Φ) is a first-class physical parameter that modulates rigid body dynamics. Unlike conventional game engines where integration metrics are UI overlays, Symtropy's Φ-physics coupling creates five bidirectional channels: motor authority gating, energy budgets, harmony field friction modulation, sanctuary impulse dampening, and integration-parameterized space-time curvature. We demonstrate three key findings: (1) cooperation emerges as a thermodynamic necessity under energy scarcity (p<0.001, Cohen's d=-1.94); (2) maintaining Φ has a measurable energy cost (J/Φ metric); and (3) Φ-parameterized conformal curvature measurably deflects agent trajectories (p=0.009, Cohen's d=-35.5). The engine implements LTC-based exponential damping, 2nd Law entropy tracking, and dimension-correct 1/r^(D-1) field theory with Plummer softening. Code is open-source (AGPL-3.0).

---

### 1. Introduction

- Integration metrics in AI: typically UI labels, not physical forces that modulate dynamics
- IIT (Tononi 2004): consciousness = integrated information (Φ)
- FEP (Friston 2010): agents minimize free energy through action
- Gap: no engine treats Φ as a physics parameter that modulates forces, energy, and geometry
- Contribution: Symtropy — a Rust engine where an integration/coherence metric (Φ) modulates physics

### 2. Architecture

#### 2.1 N-Dimensional Rigid Body Engine
- Const-generic `PhysicsWorld<D>` (2D, 3D, 4D)
- GJK + EPA collision detection (Gram-Schmidt ND normals)
- LBVH broadphase with Morton codes
- LTC exponential damping: v(t+dt) = v(t) × exp(-dt/τ) — frame-rate independent, never reverses velocity
- Semi-implicit Euler with tensor inertia (SVector principal moments)

#### 2.2 Φ-Physics Coupling (5 Channels)
1. **Φ → Motor gain**: Safety tiers (Green/Yellow/Orange/Red) gate force authority
2. **Φ → Energy budget**: Integration maintenance costs energy. Entropy tracked (2nd Law). Helmholtz free energy F = U - TS.
3. **Harmony → Friction**: 1/r^(D-1) harmony fields with Plummer softening modulate collision friction
4. **Sanctuary → Impulse**: High Sacred Stillness dampens impulses (momentum tracked via AtomicU64)
5. **Φ → Curvature**: Conformal metric g_ij = e^{2σ}δ_ij. Geodesic correction with velocity clamp (MAX_GEODESIC_ACCEL).

#### 2.3 Free Energy Gradient
- Agents move to minimize variational free energy
- 4 components: resonance seeking, energy well seeking, danger fleeing, exploration
- Φ modulates social drive: `cooperation_urgency = (1 - energy) × (0.5 + Φ)`

#### 2.4 Thermodynamic Ledger
- Every energy flow tracked: consumption via consume_energy() → record_action()
- Dissipation via damping, friction, collision → record_dissipation()
- Novel metric: J/Φ = Σ(energy × Φ) / Σ|ΔΦ| (Joules per unit integration change)
- Landauer bound reference: 2.87 × 10⁻²¹ J/bit at 310K

### 3. Experiments

#### 3.1 Cooperation Emergence (Section 5 in formal spec)
- **Design**: 12 agents, 10,000 ticks, 10 seeds × 3 conditions
- **Conditions**: FULL (thermo + FEP + offloading), ENERGY_ONLY (thermo only), FREE (no thermo)
- **Measure**: Nearest-neighbor clustering distance
- **Result**: FULL clusters 31% tighter (8.15 vs 11.92)
- **Statistics**: Mann-Whitney U=6.0, z=-3.33, p=0.0009, Cohen's d=-1.94 (large)
- **Interpretation**: Cooperation is thermodynamically necessary — the only condition that produces spatial clustering is the one with all three components (scarcity + gradient + resonance)

#### 3.2 Integration-Parameterized Curvature Lensing
- **Design**: Test body launched past stationary high-Φ source, 5 curvature scales × 5 seeds
- **Measure**: Trajectory deflection from straight line
- **Result**: scale=0.05 → 12.4 unit deflection (62% of closest approach)
- **Statistics**: Mann-Whitney U=0.0, z=-2.61, p=0.009, Cohen's d=-35.5 (massive)
- **Interpretation**: Integration-parameterized curvature deflects trajectories. Higher Φ = stronger geodesic focusing.

#### 3.3 J/Phi Convergence
- **Design**: 10 seeds, 10,000 ticks, convergence detector (window=200, threshold=1e-3)
- **Result**: FULL condition converges at mean tick 1438. ENERGY_ONLY converges at tick 225. FREE never converges.
- **J/Phi values**: FULL = 3.4×10¹⁰ per agent per second (maintaining high Φ is expensive)
- **Interpretation**: The thermodynamic cost of integration stabilizes when all systems interact

#### 3.4 HDC vs Scalar Integration (Preliminary)
- **Design**: Compare MasterConsciousnessEquation (scalar) vs 16,384D HDC state vectors
- **HDC Phi**: Norm-based readout. High inputs → 0.73, low inputs → 0.08 (dynamic, responsive)
- **Previous run**: HDC converges 5/10 vs scalar 0/10 (different convergence dynamics)
- **HDC vector diversity**: 5.3% mean pairwise HDC vector similarity (agents develop unique HDC vector signatures)

### 4. Related Work

- Rapier, Box2D, Bullet: No Φ-physics coupling
- OpenAI Gym/MuJoCo: Reward-based, not thermodynamically grounded
- IIT implementations (Oizumi 2014): Compute Φ but don't couple to physics
- FEP implementations (Active Inference, Friston 2016): Perception-action but not rigid body
- Miegakure (4D game): Dimensional visualization but no Φ coupling
- **Symtropy is the first engine combining IIT-derived Φ, FEP, thermodynamics, and rigid body physics**

### 5. Discussion

- Cooperation as thermodynamic necessity: agents that don't cooperate have their Φ collapse
- Integration as geometry: conformal curvature creates "integration wells"
- The J/Phi metric: first empirical measure of integration maintenance energy cost
- Limitations: integration metric produces near-static output (addressed by HDC approach)
- Future: HDC state vectors as integration substrate, procedural topology response

### 6. Conclusion

An integration/coherence metric (Φ) can be treated as a first-class physical parameter. When coupled to rigid body dynamics through energy budgets, harmony fields, and space-time curvature, it produces emergent cooperation (p<0.001) and measurable geodesic effects (p=0.009). The engine is open-source and extensible to arbitrary dimensions.

### References

- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience.
- Friston, K. (2010). The free-energy principle. Nature Reviews Neuroscience.
- Adams, Shipp & Friston (2013). Predictions not commands. Frontiers Computational Neuroscience.
- McFadden, J. (2020). Integrating information in the brain's EM field. Neuroscience of Consciousness.
- Landauer, R. (1961). Irreversibility and heat generation.
- Karras, T. (2012). Maximizing parallelism in BVH construction. HPG.
- Carroll, S. (2004). Spacetime and Geometry (conformal transformations).

---

## Figures Needed

1. Architecture diagram: 5 coupling channels
2. Clustering comparison: FULL vs FREE (box plot or scatter)
3. Curvature lensing: trajectory at 5 scales (line plot)
4. J/Phi convergence: time series for 3 conditions
5. HDC Φ response: high vs low input dynamics

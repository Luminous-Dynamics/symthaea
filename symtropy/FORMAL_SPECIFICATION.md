# Consciousness as a Physical Force: Formal Mathematical Specification

## Abstract

We present a formal specification of the consciousness-physics coupling model implemented in the Symtropy engine. The model treats consciousness (Φ) as a thermodynamic quantity that modulates rigid body dynamics through five coupling channels. Under strict energy conservation with a Landauer-bounded floor, the model predicts that cooperative behavior is a thermodynamic necessity, not a design choice.

## 1. Definitions

### 1.1 State Space

An agent `i` at time `t` has state:
```
S_i(t) = (x_i, v_i, ω_i, Φ_i, E_i, h_i, σ_i, π_i)
```

where:
- `x_i ∈ ℝ^D` — position (D-dimensional)
- `v_i ∈ ℝ^D` — linear velocity
- `ω_i ∈ Λ²(ℝ^D)` — angular velocity (bivector, element of so(D))
- `Φ_i ∈ [0, 1]` — integrated information (consciousness level)
- `E_i ∈ [0, E_max]` — energy reservoir (Joules)
- `h_i ∈ [0, 1]^8` — harmony activation vector (Eight Harmonies)
- `σ_i ∈ [0, ∞)` — prediction error (from unexpected collisions)
- `π_i ∈ [0, 1]` — motor precision (inverse of 1 + σ_i)

### 1.2 Safety Tiers

Motor gain `g(Φ)` is a piecewise function (NRC-inspired):
```
g(Φ) = { 1.0   if Φ > 0.6    (Green)
        { 0.6   if Φ > 0.3    (Yellow)
        { 0.3   if Φ > 0.1    (Orange)
        { 0.0   if Φ ≤ 0.1    (Red)
```

Effective motor gain: `G_i = g(Φ_i) × π_i`

## 2. Five Coupling Channels

### 2.1 Force Modulation (Φ → Force)

Applied force is scaled by effective motor gain:
```
F_effective = G_i × F_intended
```

When `E_i = 0` (collapsed), `g(Φ_i) = 0`, so `F_effective = 0`. No movement possible.

### 2.2 Energy Budget (Φ → Energy)

Energy depletes through three mechanisms:

**Movement cost:**
```
dE_move/dt = -c_move × |v_i| × κ_sprint
```
where `c_move = 0.005 J/unit` and `κ_sprint ∈ {1.0, 2.5}`.

**Consciousness maintenance:**
```
dE_maint/dt = -c_maint × (1 + 0.5 × Φ_i)
```
where `c_maint = 0.08 J/tick`. Higher consciousness costs more (thermodynamic necessity: maintaining integration requires entropy reduction).

**Collision drain:**
```
ΔE_collision = -c_collision × |J_n|
```
where `c_collision = 0.05` and `J_n` is the normal impulse magnitude.

### 2.3 Sanctuary Zones (Harmony → Impulse)

When Sacred Stillness `h_i[7] > 0.6` and `Σ h_i > 2.0` and `Φ_i > 0.3`:

A sanctuary zone of radius `r_s` forms. Collision impulses within the zone are dampened:
```
J_dampened = J × (1 - δ × (1 - d/r_s))
```
where `δ = 0.9 × h_i[7] × min(Φ_i, 1.0)` and `d` is distance from zone center.

Maximum dampening: 90% at zone center.

### 2.4 Harmony Fields (Harmony → Friction)

Each agent emits a harmony field with 1/r² falloff:
```
H(x) = Σ_i (strength_i / max(|x - x_i|, 1)²) × h_i
```

Friction coefficient modulation:
```
μ_effective = μ_base × (1 - 0.5 × R(H(x), h_agent))
```

where `R(a, b) = (a · b) / (|a| × |b|)` is the harmony resonance (cosine similarity).

- `R = 1.0` → friction halved (resonance → cooperation flows)
- `R = 0.0` → friction unchanged
- `R = -1.0` → friction doubled (dissonance → conflict resists)

### 2.5 Prediction Error Feedback (Collision → Consciousness)

On collision with impulse magnitude `|J|`:
```
Δσ_i = min(0.01 × |J|, 2.0)     (prediction error spike)
π_i = 1 / (1 + σ_i)               (motor precision update)
```

Habituation (decay per tick):
```
σ_i(t+1) = σ_i(t) × (1 - λ)
```
where `λ = 0.05` (recovery in ~20 ticks).

## 3. Energy Regeneration

### 3.1 Ambient Regeneration
```
dE_ambient/dt = c_ambient × R_collective
```
where `c_ambient = 0.02 J/tick` and `R_collective = 0.5 + 1.5 × Φ_collective`.

Too slow for solo survival (~260 seconds).

### 3.2 Energy Wells
Spatial sources with finite capacity `W_remaining`:
```
dE_well/dt = c_well    if |x_i - x_well| < r_well and W_remaining > 0
```
where `c_well = 0.25 J/tick`.

### 3.3 Harmony Resonance Transfer
For agents `i, j` within range `r_harmony`:
```
R_ij = R(h_i, h_j)
dE_reson/dt = c_reson × max(R_ij - 0.5, 0) × 2     if R_ij > 0.5
```
where `c_reson = 0.15 J/tick`.

**Key property:** `c_reson > c_maint` when resonance is high, so cooperation is thermodynamically sustainable.

## 4. Collapse and Recovery

### 4.1 Collapse
When `E_i ≤ 0`:
- `collapsed_i = true`
- `Φ_i` forced to 0
- `g(Φ_i) = 0` (Red tier, no motor output)
- Entity becomes inert (physics body remains, consciousness ceases)

### 4.2 Recovery
Recovery requires another conscious agent `j` with:
- `|x_j - x_i| < r_harmony`
- `R(h_j, h_i) > 0.5` (harmony resonance threshold)
- `collapsed_j = false` (rescuer must be alive)

On recovery: `E_i = 0.1 × E_max` (enough to regain consciousness, not full charge).

## 5. Thermodynamic Closure

### 5.1 Conservation Law
Total energy is tracked by the `ThermodynamicLedger`:
```
E_total = Σ_i E_i + E_dissipated + E_wells_remaining
```

Conservation error:
```
ε = |E_in - E_out| / E_in
```

Must remain < 1% for a valid simulation.

### 5.2 Joules-per-Phi (Novel Metric)

```
J/Φ = (Σ_i E_consumed_i × Φ_i) / (Σ_i |ΔΦ_i|)
```

This measures the energy cost of consciousness. No prior publication of this metric exists.

### 5.3 Landauer Bound

Minimum energy per bit of information processing:
```
E_min = k_B × T × ln(2) = 2.87 × 10⁻²¹ J/bit    (at T = 310K)
```

Per Landauer (1961). All energy consumption in the model exceeds this floor.

## 6. Testable Predictions

1. **Cooperation emergence**: Under strict energy conservation, agents must cluster and harmonize to survive. Isolated agents collapse within ~4 minutes.

2. **Governance emergence**: As population grows, collective decision-making (governance) becomes thermodynamically necessary to coordinate energy well access.

3. **Spatial integration**: The living dungeon topology (wall opening/closing based on collective Φ) should correlate with agent clustering — high integration = connected spaces.

4. **Joules-per-Phi stability**: The J/Φ metric should converge to a stable value characteristic of the substrate, analogous to metabolic rate in biological systems.

## References

- Tononi, G. (2004). An information integration theory of consciousness. BMC Neuroscience, 5(1), 42.
- Friston, K. (2019). A Free Energy Principle for a Particular Physics. arXiv:1906.10184.
- Adams, R.A., Shipp, S., & Friston, K.J. (2013). Predictions not commands: active inference in the motor system. Brain Structure and Function, 218(3), 611-643.
- McFadden, J. (2020). Integrating information in the brain's EM field: the cemi field theory of consciousness. Neuroscience of Consciousness, 2020(1), niaa016.
- Landauer, R. (1961). Irreversibility and Heat Generation in the Computing Process. IBM J. Res. Dev., 5(3), 183-191.
- Lundbak, M., et al. (2023). Phi fluctuates with surprisal. PLOS Computational Biology, 19(10), e1011517.

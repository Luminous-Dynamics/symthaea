# J/Φ metric

> **Status:** stub — full derivation, empirical data, and publication in Phase 5.

> **Terminology caveat (added 2026-07-03):** the `Φ(t)` in this metric is **not** a
> computed IIT (Integrated Information Theory) quantity. It's the fused output of
> Symtropy's Master Consciousness Equation (`symthaea-consciousness-equation`), whose
> own Φ input slot is filled at every call site in this codebase by a locally-defined
> heuristic (e.g. `1.0 - danger_level` in `symtropy-robotics-bridge`, oscillator
> coherence in the Bevy pendulum-swarm examples, or a constant `0.5`) — not by any of
> symthaea-core's IIT-inspired calculators (`SpectralMIPFinder`, `TieredPhi`,
> `ConnectivityCalculator`). So J/Φ measures energy cost per unit of *this equation's
> output*, not per bit of integrated information in the IIT sense. The metric and its
> convergence finding below may still be real and interesting — but read "integrated
> information" throughout this page as "the Master Consciousness Equation's fused
> output," not as a validated Φ.

## Definition

`J/Φ` = Joules per bit of integrated information. A ratio between the **thermodynamic cost** of maintaining a system and the **amount of integration** that system sustains.

Informal: *how expensive is consciousness, in energetic terms?*

Formal:
```
J/Φ = ∫₀ᵀ P(t) dt  /  ∫₀ᵀ Φ(t) dt
```
where `P(t)` is instantaneous power dissipated by the consciousness field's thermodynamic ledger, and `Φ(t)` is integrated information.

## Empirical finding

In `cargo run --example jphi_convergence`, J/Φ converges to a stable substrate-characteristic value — approximately **10⁴ J/Φ** for default Symtropy parameters. The convergence is *independent of starting conditions*: agents initialised with wildly different Φ, energy, or harmony values all settle to the same J/Φ ratio.

This is a novel result with no prior publication that we're aware of. The interpretation is open — candidate hypotheses:

1. Substrate-characteristic constant (analogous to vacuum permittivity)
2. Thermodynamic attractor for any Φ-coupled system
3. Artifact of the specific coupling constants in `ConsciousnessField<D>`

Reproducing the experiment on different parameter sets is ongoing work.

## Relation to Landauer bound

The Landauer bound (`k_B × T × ln 2 ≈ 2.87 × 10⁻²¹ J/bit at 300 K`) is a lower bound on the energy cost of erasing a bit. J/Φ is a different quantity: integration cost, not erasure cost. Empirically J/Φ sits many orders of magnitude above the Landauer bound — integration is more expensive than erasure.

## Implementation

See `symtropy-consciousness-physics/src/thermodynamics.rs`:

- `ThermodynamicLedger` — accumulates `P(t) dt`
- `EntityConsciousness.phi_history` — rolling record of Φ
- `j_phi()` — computes the ratio

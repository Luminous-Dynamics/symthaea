# Φ-coupling

> **Status:** stub — full chapter in Phase 1 of the [roadmap](../roadmap.md).

Symtropy uses Φ (integrated information, Tononi 2004) as its reference implementation of the generic [state-coupling framework](./generic-state-coupling.md).

## What Φ is

Φ is a formal information-theoretic quantity measuring how much a system's integrated information exceeds the sum of its parts. In this codebase, "consciousness" refers **exclusively** to this mathematical object. No claim is made about subjective experience, qualia, or sentience.

## Why Φ specifically

Symthaea's `ConsciousnessEquation` synthesises seven theories — IIT, Free Energy Principle, CEMI field, harmonic binding, global workspace, information integration, recurrent dynamics — into a single scalar Φ value per entity. See `symthaea-consciousness-equation` on crates.io.

## What Φ does in Symtropy

Via `ConsciousnessField<D>`, Φ modulates:

1. **Motor authority** (NRC 4-tier gating)
2. **Energy budget** (thermodynamic cost, Landauer-bound)
3. **Collision impulses** (sanctuary zones)
4. **Friction** (CEMI-inspired `1/r^(D-1)` fields)
5. **Motor precision** (prediction-error feedback)

See [The five coupling channels](./five-channels.md) for the mathematics.

## Key results

- **81.3% tighter clustering** under thermodynamic enforcement — cooperation emerges as a thermodynamic necessity, not hand-coded.
- **J/Φ converges** to a stable substrate-characteristic value.
- **Solo agents collapse within ~4 minutes**; cooperative agents sustain indefinitely.

## Reproducing

```bash
cargo run --example cooperation_emergence -p symtropy-consciousness-physics
cargo run --example jphi_convergence -p symtropy-consciousness-physics
```

See [The 63 experiments](../research/experiments.md).

## Licensing

`ConsciousnessField<D>`, `EntityConsciousness`, `HarmonyField<D>`, `ThermodynamicLedger`, and the 63 experiments live in `symtropy-consciousness-physics` — **AGPL-3.0-or-later**. Commercial licensing available.

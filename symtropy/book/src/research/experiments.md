# The 63 experiments

`symtropy-consciousness-physics/examples/` contains 63 reproducible experiments. All run from a single seed and produce identical results (bit-for-bit on the same CPU).

> **Status:** stub — each experiment gets its own annotated page in Phase 5. This page is the flat index.

## Running any experiment

```bash
cargo run --example <name> -p symtropy-consciousness-physics
```

With features:

```bash
cargo run --example curvature_lensing -p symtropy-consciousness-physics --features consciousness-curvature
cargo run --example hdc_cooperation   -p symtropy-consciousness-physics --features consciousness-hdc
```

## Headline results (reproducible)

- **`cooperation_emergence`** — 81.3 % tighter clustering when thermodynamic enforcement is active vs off. Cooperation emerges as a thermodynamic necessity, not hand-coded.
- **`jphi_convergence`** — J/Φ (Joules per Φ) converges to a stable substrate-characteristic value (~10 K J/Φ). Novel metric, no prior publication.
- **`solo_collapse`** — Prediction 1 of the formal specification: solo agents collapse within ~4 minutes; cooperative agents sustain indefinitely.

## Categories

| Category | Count | Examples |
|---|---|---|
| Cooperation emergence | 8 | `cooperation_emergence`, `cooperation_1000`, `cooperation_drift` |
| Scaling | 5 | `cooperation_1000`, `phase_transition_1000` |
| Economics | 7 | `inequality_emergence`, `tragedy_commons`, `wealth_diffusion` |
| Social networks | 4 | `dunbar_number`, `trust_propagation` |
| Phase transitions | 6 | `anesthesia_transition`, `coherence_collapse` |
| Thermodynamics | 9 | `jphi_convergence`, `landauer_bound`, `energy_audit` |
| Curvature (feature-gated) | 2 | `curvature_lensing`, `curvature_selforg` |
| HDC (feature-gated) | 1 | `hdc_cooperation` |
| Harmony fields | 5 | `sanctuary_zones`, `resonance_friction` |
| Prediction error | 4 | `motor_precision_decay`, `habituation_cycle` |
| Other | 12 | (misc) |

## Determinism guarantee

Every experiment's output is bit-identical across runs on the same machine with the same seed. See [Determinism contract](../core-concepts/determinism.md).

## Publications

See [Publications](./publications.md).

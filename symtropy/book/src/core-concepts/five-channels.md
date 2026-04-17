# The five coupling channels

> **Status:** outline — full mathematical derivation in `../FORMAL_SPECIFICATION.md` (will be ported here in Phase 0).

Any metric couples to physics through five channels. All five are implemented in `ConsciousnessField<D>` and reusable in your own `PhysicsCallback<D>` implementations.

## Channel 1 — Metric → Force

Motor gain is gated by the NRC 4-tier safety system:

- Green (metric ≥ 0.6): 100 % motor authority
- Yellow (0.3 – 0.6): 50 %
- Orange (0.1 – 0.3): 20 %
- Red (< 0.1): 0 %

Gravity and external forces still apply; **motor commands** scale.

## Channel 2 — Metric → Energy

Maintenance cost: `c_maint × (1 + 0.5 × metric)`. Higher metric = higher entropy cost (Landauer-bound enforced: `2.87 × 10⁻²¹ J/bit`).

Energy depletion → Red tier → inert body. Recovery requires harmony resonance with cooperating bodies (channel 3/4 feedback).

## Channel 3 — Harmony → Impulse

Sanctuary zones (high harmony activation, `h[7] > 0.6`) dampen collision impulses by up to **90 %**. This models "safe spaces" as a physical property.

## Channel 4 — Harmony → Friction

`1/r^(D-1)` CEMI-inspired fields modulate friction coefficients in space:

- Resonance (aligned harmony vectors) → friction × 0.5
- Dissonance (anti-aligned) → friction × 2.0

Reference: McFadden (2020), CEMI field theory.

## Channel 5 — Collision → Metric

Prediction error from unexpected collisions temporarily reduces motor precision (habituation over 20 ticks). Based on Adams/Friston (2013) — motor commands as proprioceptive predictions.

Closes the loop:
```
Metric → Motor → Force → Collision → Prediction Error → Motor Precision ↓ → Recovery
```

## The mathematics

See `FORMAL_SPECIFICATION.md` at the repo root for full equations. This page will be updated with inline derivations in Phase 0.

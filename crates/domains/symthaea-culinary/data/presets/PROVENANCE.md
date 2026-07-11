# Preset provenance

Each preset is a **physically valid point within the fixed safety invariants** `crate::validate`
already checks (Phase 1) — the validator itself never changes per style; only which concrete
numbers a style prefers does. Every preset is asserted to validate with zero violations
(`presets::tests::every_preset_validates_cleanly`).

## `french_classical.json`

- **Emulsion φ=0.68** — a classic hollandaise/mayonnaise-range oil-in-egg-yolk ratio, comfortably
  under the random-close-packing break point (0.7405).
- **Coagulation: egg yolk, peak 70 °C** — above this crate's 68 °C yolk set point (McGee, *On Food
  and Cooking*, 2004), a gentle classic hollandaise temperature.
- **Pasteurization: 63 °C, 3 min hold** (after an 8 min ramp) — delivers ≈26 log-reductions against
  the Murphy et al. (2004) Salmonella/poultry model, vastly exceeding the 7-log target; a
  comfortable classical low-and-slow poach/hold.
- **Hydration: 68 %** — a standard French pain de campagne baker's percentage, mid-range of the
  60–85 % bread window (Suas, *Advanced Bread and Pastry*).

## `molecular_gastronomy.json`

- **Emulsion φ=0.72** — deliberately close to (but under) the 0.7405 break point, representing an
  aggressively oil-rich, precision-engineered emulsion.
- **Coagulation: egg yolk, peak 69 °C** — chosen to clear this crate's 68 °C threshold with minimal
  margin. Note: real sous-vide egg technique often targets 63–65 °C for a "jammy" yolk texture, but
  that relies on *time*-at-temperature kinetics this crate's coagulation check does not model (a
  simple peak-threshold check, not a kinetic one) — the preset respects the validator's actual
  model rather than a real-world exception it can't represent yet.
- **Pasteurization: 60 °C, 45 min hold** (after a 10 min ramp) — a real sous-vide low-temperature,
  long-time chicken technique; delivers ≈114 log-reductions, far past the 7-log requirement (sous
  vide chicken breast is commonly held far longer than the safety minimum for texture reasons).
- **Hydration: 50 %** — a laminated-pastry-range baker's percentage, mid-range of the 45–60 %
  pastry window.

## `rustic_fermentation.json`

- **No emulsion/coagulation/pasteurization** — this style is about the bread, not an emulsified
  sauce or egg dish; those fields are simply omitted (`#[serde(default)]` makes this valid JSON).
- **Hydration: 80 %** — a high-hydration rustic/ciabatta-style dough, near the top of the 60–85 %
  bread window (still comfortably inside it).

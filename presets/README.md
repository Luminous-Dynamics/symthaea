# Symthaea Atelier research additions

This directory contains the user-facing design and wardrobe extensions to the apparel engineering digital thread.

- [Personal Garment Studio #12093](https://github.com/Luminous-Dynamics/symthaea/issues/12093)
- [Wardrobe-First Optimization #12094](https://github.com/Luminous-Dynamics/symthaea/issues/12094)
- [Apparel Engineering Digital Thread #11731](https://github.com/Luminous-Dynamics/symthaea/issues/11731)
- [Draft implementation and research PR #11732](https://github.com/Luminous-Dynamics/symthaea/pull/11732)

## Public preset catalogue

The branch contains 17 editable presets:

- Five functional/hybrid presets in this directory: Everyday Core, Heat-Adaptive, Adaptive Ease, Repair-First and Creative / Statement.
- Twelve style directions in `presets/style/`: Streetwear Volume, Classic Tailoring, Utility Workwear, Outdoor Layering, Movement Active, Modest Coverage, Heritage Craft, Formal Occasion, Avant-Garde Sculptural, Vintage Rework, Sensory Calm and Hot Climate Lightweight.

The JSON contracts are in `schemas/design-preset.schema.json` and `schemas/garment-design-brief.schema.json`. A synthetic non-PPE overshirt brief is in `examples/overshirt-design-brief.example.json`.

## Rust research slices

- `tools/atelier-brief-core/` provides typed design-brief validation and fail-closed handoff gating.
- `tools/atelier-wardrobe-core/` provides an exact small-case wardrobe-coverage oracle with protected-item, repair-first, explicit-currency and no-additions semantics.

These are research slices, not a finished user application. Presets are editable starting points; they do not establish fit, comfort, safety, textile performance, environmental benefit or production readiness.

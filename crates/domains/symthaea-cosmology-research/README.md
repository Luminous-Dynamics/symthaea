# symthaea-cosmology-research

Evidence-first contracts for Symthaea cosmology experiments.

This crate deliberately does **not** implement a Boltzmann solver, sampler, or
cosmological likelihood. Mature external engines such as CLASS/CAMB, Cobaya,
and desilike remain the numerical backends. This crate owns the scientific
control layer around them:

- preregistered experiment identity;
- validated Git-object and SHA-256 identities;
- evidence-cube coordinates;
- claim-layer boundaries;
- blinded/holdout contamination tracking;
- DE-001 gate ordering;
- fail-closed claim licensing;
- DE-001A exact-reproduction specifications.

The initial research target is **DE-001: Cosmological Acceleration Anomaly
Localization**. Its first permitted scientific claim is only an
`ObservationalAnomaly`. A better fit of a dynamic-`w` model cannot be silently
promoted into a claim of phenomenological dark-energy dynamics or a physical
mechanism.

DE-001A is narrower still: it can establish only whether Symthaea reproduced a
frozen published constraint under a frozen numerical/data/configuration
subject. A DE-001A PASS does **not** license an anomaly claim.

See:

- `docs/research/DE_001_PREREGISTRATION_2026-09-18.md`
- `docs/research/DE_001A_REPRODUCTION_CONTRACT_2026-09-18.md`

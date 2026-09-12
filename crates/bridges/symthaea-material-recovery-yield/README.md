# Symthaea Material Recovery-Yield Evidence

This crate supplies a bounded circularity input: the fraction of material recovered under one exact declared recovery/recycling process.

It does **not** equate one process yield with universal recyclability.

## Process-specific semantics

Every record binds:

- material identity;
- process identity;
- feedstock form;
- recovered product;
- recovery yield fraction;
- evidence basis;
- conditions note;
- last-updated metadata.

Changing any of those changes the evidence context.

## Evidence basis

Records explicitly distinguish:

- `modeled`;
- `measured_lab`;
- `measured_pilot`;
- `measured_industrial`.

Modeled values emit analytical/model evidence. Measured values emit experiment evidence. Industrial measurement is intentionally not auto-promoted to a universal field-validation claim for every feedstock, facility or product configuration.

## Exact source provenance

The normalized dataset includes source title/version/URI, the SHA-256 of the authoritative source document, normalization metadata and recovery records.

The binding API requires the exact source-document bytes and verifies their SHA-256 before producing evidence.

The exact normalized JSON bytes and the semantic normalized dataset are both content-addressed.

## Candidate/material/process identity

Two material-binding modes exist:

- exact material ID, requiring candidate ID == source material ID;
- explicit mapping with a non-empty review note.

The process ID is always exact. A result for one process cannot silently satisfy another process.

## Discovery evidence semantics

A successful result emits:

- metric `material_recovery_yield_fraction`;
- unit `fraction`;
- model/experiment fidelity determined by the declared evidence basis;
- dataset provenance plus explicit model/experiment evidence kind;
- feedstock/product/conditions context;
- exact source/capture/dataset digests.

No calibrated predictive uncertainty is invented.

## Zero/one boundary

Values must lie in `[0,1]`, but neither endpoint is a universal circularity conclusion. A measured yield of `1.0` under one controlled process does not prove perfect collection, purity, repeated-cycle quality, economics or closed-loop reuse.

## Network-free CLI

    material-recovery-yield <recovery-data.json> <source-document> <candidate-id> <material-id> <process-id> [explicit-mapping-note]

Without the optional note, exact-material-ID mode is used.

## Deliberate non-claims

A receipt does not establish:

- collection rate;
- recovered-product purity;
- economic viability;
- energy/reagent burden;
- repeated-cycle material quality;
- closed-loop reuse;
- universal recyclability;
- life-cycle environmental benefit;
- deployment fitness.

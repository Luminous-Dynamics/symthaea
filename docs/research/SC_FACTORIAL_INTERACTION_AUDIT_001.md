# SC-FACTORIAL-INTERACTION-AUDIT-001 — synthetic 2×2 design qualification

**Tracking issue:** [Symthaea #13109](https://github.com/Luminous-Dynamics/symthaea/issues/13109)  
**Parent research programme:** [Symthaea #13095](https://github.com/Luminous-Dynamics/symthaea/issues/13095)  
**Implementation branch:** `research/sc-factorial-interaction-v0.1`

## Objective

Check whether an intervention experiment really contains the four arms needed to estimate a two-factor interaction:
- control (A=0, B=0);
- A only (A=1, B=0);
- B only (A=0, B=1);
- combined (A=1, B=1).

For a continuous response on a declared additive scale, calculate:

`Delta_AB = y(1,1) - y(1,0) - y(0,1) + y(0,0)`

This is a contrast calculation, not proof of a physical mechanism or causality. Actual causal identification still depends on assignment integrity, intervention delivery, randomization, blocking, specimen independence, measurement validity and assumptions about interference / carryover.

## Current executable contract

- Schema: `data/schemas/sc_factorial_interaction_v0_1.schema.json`
- Synthetic fixture: `tools/superconductivity-discovery-harness/fixtures/factorial_interaction_smoke_v0_1.json`
- Rust auditor: `tools/superconductivity-discovery-harness/src/bin/factorial_interaction_audit.rs`

The v0.1 design is deliberately synthetic and limited to two binary intervention factors, two independent blocks, four unique specimens per block and two response channels. Each block must have a unique batch and campaign. Each arm must have a distinct specimen and realized-state identity, a complete within-block randomization order, and sensor-verified factor delivery readback tied to a SHA-256 source receipt.

The auditor verifies that delivered A/B levels match the declared on/off levels and units. Planned or commanded-only levels are not accepted as evidence of delivered intervention. This is a software audit contract; it does not authorize laboratory equipment or operational procedures.

## Pairing and coherence remain separate

Each specimen has two explicitly distinct outcomes:
- `local_pairing_gap_mev` — local pair-formation / gap response;
- `bulk_superfluid_stiffness_mev` — global phase-coherence response.

The auditor calculates an independent factorial contrast for each channel. It never averages them together or calls a pairing-only interaction a bulk superconductivity improvement.

The synthetic generator truth is intentionally adversarial:
- local pairing-gap interaction: +3 meV;
- bulk superfluid-stiffness interaction: 0 meV.

Thus a correct result is a pairing interaction without a corresponding coherence interaction. This is a fixture arithmetic check only.

## What is verified

The auditor rejects:
- incomplete or duplicated 2×2 arms;
- reused specimen or realized-state IDs;
- mismatched nominal material identity;
- repeated batch/campaign IDs across independent blocks;
- incomplete / duplicated within-block randomization order;
- commanded-only or unverified delivery records;
- factor delivery values inconsistent with declared levels;
- invalid assignment receipt digests;
- missing/duplicate response channels or mixed response units;
- condition-profile changes between blocks;
- computed contrasts that do not match the synthetic generator truth.

The receipt records per-block contrasts and a descriptive mean across blocks. It deliberately does not generate a p-value, confidence interval, population variance, or physical synergy claim from two synthetic blocks.

## Next qualification phase

For physical experiments, compose rather than duplicate the existing owners:
- #12966 for intervention-induced specimen-state transitions;
- #12986 for causal identification / estimand and assumption qualification;
- #12671 for synthesis protocol -> realized phase / microstructure -> property;
- #12677 for pairing versus phase-coherence observation semantics;
- #12946 for value-of-information / sequential experiment choice;
- #13078 and PR #13082 for sample/batch/campaign/lab split integrity.

Before a real multi-knob result is called synergistic, preregister the response scale, contrast, treatment versions, delivered-dose verification, specimen/block structure, primary outcome and negative controls. Estimate batch/lab effects and treatment noncompliance in a separately qualified statistical model. A small 2×2 pilot can reveal a design failure; it does not establish robust transfer.

## Claim ceiling

A passing synthetic run verifies only the factorial design contract and interaction arithmetic for generated data. It does not establish a new superconducting mechanism, increased measured Tc, improved bulk coherence, experimental reproducibility, safe intervention conditions or equipment authorization.

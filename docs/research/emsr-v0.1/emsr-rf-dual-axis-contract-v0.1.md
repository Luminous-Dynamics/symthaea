# EMSR RF Dual-Axis Acquisition and Recognition Contract 0.1

**Status:** Executable synthetic semantic fixtures; no physical RF qualification claim  
**Owner:** Symthaea EMSR-RF-ACQ-001 [#10975](https://github.com/Luminous-Dynamics/symthaea/issues/10975)  
**Parent:** EMSR-RF-OPENSET-001 [#10893](https://github.com/Luminous-Dynamics/symthaea/issues/10893)

## Problem

The legacy four-state vocabulary mixes three recognition outcomes (`KNOWN_CLASS`, `UNKNOWN`, `INDETERMINATE`) with one acquisition-quality outcome (`DEGRADED_ACQUISITION`). The capture chain is a prerequisite for interpreting a signal, not a class in the classifier's label space.

## Dual-axis contract

### Acquisition qualification

`QUALIFIED`, `DEGRADED`, `UNQUALIFIED`, or `UNKNOWN`, bound to an exact receiver/front-end/antenna/configuration/firmware/acquisition profile and evidence for calibration, sample integrity and clock quality.

### Recognition disposition

`KNOWN_CLASS`, `UNKNOWN_CLASS`, or `INDETERMINATE`, bound to the frozen class vocabulary, model digest, input-window digest and uncertainty. `UNKNOWN_CLASS` means a qualified observation lies outside the supported label space; it does not mean malicious, identified, located or attributed.

### Derived consumer disposition

| Acquisition | Recognition | Consumer disposition | Candidate admitted? |
|---|---|---|---|
| QUALIFIED | KNOWN_CLASS | KNOWN_CLASS | Yes, with calibrated uncertainty. |
| QUALIFIED | UNKNOWN_CLASS | UNKNOWN_CLASS | No known label is admitted; preserve the valid unknown observation. |
| QUALIFIED | INDETERMINATE | INDETERMINATE | No. |
| DEGRADED / UNQUALIFIED / UNKNOWN | Any | DEGRADED_ACQUISITION | No; any model class is retained as a non-admitted diagnostic candidate only. |

The acquisition gate is independent of neural confidence. Model confidence cannot repair expired calibration, sample loss, capture clipping, unqualified clock state or a mismatched acquisition profile.

## Compatibility

The existing v0.1 four-way output remains supported as a derived compatibility view:
- `known_class` -> recognition `KNOWN_CLASS` only when acquisition is `QUALIFIED`;
- `unknown` -> recognition `UNKNOWN_CLASS` only when acquisition is `QUALIFIED`;
- `indeterminate` -> recognition `INDETERMINATE` only when acquisition is `QUALIFIED`;
- `degraded_acquisition` -> any non-`QUALIFIED` acquisition state, irrespective of the diagnostic recognition candidate.

Do not store the derived compatibility state as the only source of truth. Preserve both axes.

## Files

- `emsr-rf-dual-axis-contract-v0.1.schema.json`: strict Draft 2020-12 structure and cross-axis conditional constraints.
- `emsr-rf-dual-axis-fixtures-v0.1.json`: 12 synthetic combinations (three recognition dispositions × four acquisition qualification states).

## Required next implementation

The Rust semantic evaluator is implemented in `crates/domains/symthaea-emsr-evidence/src/rf_dual_axis.rs` and is covered by the existing isolated EMSR package workflow. It enforces exact 12/12 fixture discovery, duplicate rejection, strict unknown-field rejection, recomputation of derived disposition from both axes, and mutation tests where high-confidence known-class output is paired with failed/unknown acquisition evidence. It reuses the crate's common-mode dependency semantics and does not weaken dataset rights, split, receipt or digest checks.

## Benchmarks

- ACQ-QUAL: acquisition fault precision/recall, false qualification rate, abstention and detection delay.
- RF-OPEN: known macro-F1, unknown AUROC/AUPR, FPR95, false acceptance at matched known recall, conditioned on qualified captures.
- RF-GEN: held-out receiver/front end, antenna/site, campaign, day and capture pipeline.
- RF-EPI: selective risk/coverage and calibration by acquisition state and held-out domain.
- RF-EDGE: float/quantized parity for the acquisition gate and recognition axis, plus latency/RAM/energy/sample loss.
- RF-AUD: exact input/acquisition/model/profile reconstruction and independent verifier agreement.

Required baselines: hardware/profile QC rules, receiver-native acquisition status, simple signal-quality checks, closed-set classifier without acquisition gating, RF-RPL-style open-set comparator and full dual-axis method. Compare on matched input windows and hardware domains.

## Claim ceiling

This contract and fixture pack do not qualify hardware or physical RF measurements, establish class identity/intent/location, demonstrate GNSS/PNT integrity, or authorize an operational action. The synthetic fixture evaluator now runs as part of the EMSR evidence package tests. The reference decision table is algorithmically distinct within the same Rust package, not cross-language/organizational independence. Physical acquisition qualification and RF model performance remain unmeasured.

# MARINE-CLOCK-DRIFT-QUALIFICATION-001
## Reference-epoch and accumulated timing uncertainty for marine trials

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

## Problem

A timebase can have a stated synchronization uncertainty at one epoch and a non-zero residual rate uncertainty. Comparing every channel only to that initial uncertainty can understate timing error at the end of a long sample window.

The timing contract must bind the reference epoch, correction model, residual drift uncertainty and channel acquisition interval. It must not treat an unknown drift bound as zero.

## Proposed v0.4.0 rule

For a channel sample window [t_start, t_end) and timebase reference epoch t_ref, compute:

`elapsed_max = max(|t_start - t_ref|, |t_end - t_ref|)`

`timing_bound = base_sync_uncertainty + drift_uncertainty_ppm × 10^-6 × elapsed_max`

The channel's declared time-alignment uncertainty must be at least this conservative bound after conversion to seconds.

The drift term represents residual uncertainty in the rate after applying the declared drift correction. It is not the raw measured clock drift. The deterministic offset_seconds and drift_ppm remain separate timebase metadata.

If drift uncertainty is unknown, the trial is INCOMPLETE. If channel uncertainty is below the computed bound, the record is INVALID for the claimed timing quality.

## Versioning

- Preserve v0.1.0, v0.2.0 and v0.3.0 unchanged.
- v0.4.0 adds reference_epoch, a provenance-bearing drift_uncertainty_ppm quantity, and explicit residual-drift semantics.
- Legacy records can be migrated only with a documented source and method; do not backfill zero drift uncertainty.

## Validation and limits

The bound is conservative and deterministic. It is not a proof of GNSS/PPS integrity, clock authenticity, calibration traceability or sensor acquisition truth. Real clock synchronization still requires physical tests and external evidence.

The implementation is tracked in the corresponding GitHub issue.
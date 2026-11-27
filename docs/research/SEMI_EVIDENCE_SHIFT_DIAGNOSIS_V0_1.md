# SEMI-EVIDENCE-SHIFT-001 — evidence-discriminating drift diagnosis v0.1

**Issue:** #13880  
**Implementation:** symthaea-semiconductor::benchmark::evidence_shift_diagnosis  
**Status:** Rust sequential monitor and independent recurrence tests authored; exact-head validation pending.

## Research thesis

A VM residual is an alert signal, not a cause label.

A residual can move because the physical process changed, the measurement system/reference changed, sensor observability changed, event/label timing changed, or the inspection policy changed which wafers became visible. A residual-only detector cannot identify these causes.

Version 0.1 uses explicit per-tool/chamber signal channels:
- K1-qualified VM residual (computed from the immutable streaming prediction receipt and the corrected physical measurement);
- qualified reference-standard residual (shared across scopes only by the same explicit standard identity);
- sensor-observability score (a frozen, content-addressed sensor projection);
- process-event marker (context only; an event is not itself a drift alarm).

CUSUM and EWMA are separate per-channel sequential detectors. Baseline means/scales and thresholds come from a frozen nominal manifest; they are not estimated from the stream being judged.

## Core recurrence

Each numeric channel is standardized against the frozen baseline:

`z_t = (x_t - μ_0) / σ_0`

Two-sided CUSUM:

`C⁺_t = max(0, C⁺_(t-1) + z_t - k)`

`C⁻_t = max(0, C⁻_(t-1) - z_t - k)`

EWMA:

`E_t = λ z_t + (1 - λ) E_(t-1)`

An alert is raised when either CUSUM side reaches its preregistered threshold or absolute EWMA reaches its threshold. The implementation does not adapt thresholds from the live stream.

## Multi-evidence diagnostic logic

The current state machine deliberately abstains:
- residual + reference-standard alert -> COMMON_CAUSE_OR_MIXED_SHIFT;
- reference-standard alert alone -> MEASUREMENT_SYSTEM_SHIFT_SUSPECTED;
- sensor-observability alert -> SENSOR_OBSERVABILITY_SHIFT;
- residual alert + recent process marker + independently stable reference and sensor channels -> PROCESS_SHIFT_SUSPECTED;
- residual alert without the independent evidence required to distinguish causes -> UNRESOLVED;
- missing/stale/invalid evidence or insufficient stable reference windows -> INSUFFICIENT_EVIDENCE;
- all three numeric channels sufficiently observed and non-alerting -> NOMINAL.

A process event marker is contextual evidence, not an alarm. A process-shift diagnosis is a hypothesis, not causal proof. A shared reference standard is pooled only under its explicit standard ID; tool/chamber residual baselines remain separate.

## Evidence binding

VM residuals cannot be caller-supplied arbitrary numbers. The monitor verifies:
- a current K1 label eligibility report;
- report digest and full label-payload digest shape;
- nonzero revision, eligible disposition and downstream permission;
- measurement-event ID agreement between report, observation and prediction receipt;
- immutable streaming prediction receipt digest, schema, clock domain, event time and prediction availability;
- finite prediction and corrected physical value.

The residual is calculated internally:

`r_t = qualified_corrected_measurement_t - prediction_receipt.prediction_value_t`

Reference-standard observations require an explicit standard ID, qualified-evidence declaration and a traceability/evidence SHA-256. Sensor observations require a frozen feature-projection digest. These are software evidence checks; issuer authenticity and physical traceability remain external claims.

## Ordering and replay

- A source sequence is contiguous and is never silently sorted by arrival time.
- Event time is monotone within each tool/chamber scope and explicit shared-reference scope.
- Duplicate observation IDs return a duplicate assessment without changing detector state.
- Invalid K1, prediction, clock, baseline or source evidence is not committed into any CUSUM/EWMA channel.
- Process markers are committed only after the complete associated observation validates.
- Identical frozen event streams and policy digests replay to byte-identical assessments.

Late revocation must be represented as a new assessment version and connected to model snapshot impact closure (#13465); historical assessment receipts are never overwritten.

## Authored tests

The module currently includes:
- positive and negative shift behavior;
- shared reference-standard evidence across otherwise isolated tool scopes;
- common-cause/mixed output when residual and reference channels both shift;
- process-shift hypothesis only with independently stable reference and sensor channels plus a process marker;
- sensor-observability alert;
- unresolved residual-only alert;
- incomplete channel coverage cannot appear nominal;
- maintenance marker alone cannot create an alert;
- unqualified or stale K1 labels excluded from residual detector state;
- prediction receipt digest and measurement-event identity mismatch rejected;
- invalid observation cannot commit a process marker;
- baseline/clock mismatch and event-time regression fail closed;
- duplicate handling and deterministic replay;
- fully observed stable channels produce NOMINAL.

A separate integration test file implements an independent CUSUM/EWMA recurrence and checks known-answer outputs, positive/negative symmetry, deterministic receipts and reference-only attribution. Tests are authored and require exact-head formatting, strict Clippy, crate tests and SEMI-BENCH-001 Reference Oracle validation.

## Benchmarks and targets

Report separately:
- per-channel detection delay;
- false alerts per tool/chamber/week-equivalent;
- event precision/recall and F1;
- worst tool/chamber detection delay;
- top-1/top-3 attribution on identifiable cases;
- abstention rate on non-identifiable cases;
- false process-vs-measurement-system attribution;
- delay sensitivity for 0, 1, 5, 20 and 100 label events;
- cross-tool leakage when reference sharing is not explicitly declared.

Proposed hypotheses (not achievements):
- at least 95% recall on a frozen high-consequence synthetic shift suite at a preregistered false-alert budget;
- no more than one false alert per tool/chamber/week-equivalent on the calibrated nominal synthetic stream;
- at least 90% top-3 attribution on identifiable cases;
- at most 1% false process-vs-measurement attribution on non-identifiable controls, with abstention preferred;
- detect the designated 1.0σ persistent synthetic mean shift within 25 qualified observations, reporting the full power/false-alert curve;
- zero unqualified label updates and zero cross-tool process-baseline contamination.

These are scenario-specific acceptance targets, not universal guarantees.

## Competitive positioning

- M2D2 (Applied Sciences 2025) is a strong multi-machine/multi-modal drift and adaptation comparator on WBG fabrication data; its reported feature explanations are not evidence of physical causality.
- ASMC 2026 cloud VM reports a one-year multi-process HVM dry-etch deployment with ensemble lifecycle, rolling validation and statistical-gated retraining; this remains the production-scale operational comparator.
- NIST OGP-CMP (IEEE TSM 2025) is a dynamic sampling/uncertainty comparator on CMP.
- Classical SPC (Shewhart, CUSUM, EWMA, Hotelling T²/MEWMA) remains mandatory; a learned model must beat simple statistical alarms at matched false-alert burden.
- GRAPE (#13469) covers predictive streaming adaptation. This module adds multi-evidence diagnostic abstention, not a claim of better point prediction.
- The recent hidden-state process-profiling report claims 186 tools, 412 variables and 96M timestamped observations; its provenance and access must be verified before considering its headline metrics.

These datasets and objectives are not one leaderboard.

## Claim ceiling

The module emits sequential statistical alerts and evidence-weighted diagnostic hypotheses. It does not prove process root cause, reference fault, physical sensor failure, yield impact or process safety. Physical attribution requires independent traceable reference standards, crossed measurement-system experiments and domain-expert adjudication. No model update, retraining, APC/R2R, recipe, equipment or wafer-disposition authority is granted.

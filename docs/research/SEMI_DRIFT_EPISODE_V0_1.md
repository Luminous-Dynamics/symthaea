# SEMI-DRIFT-EPISODE-001 — multi-channel evidence-qualified drift ledger v0.1

**Issue:** #13554  
**Implementation:** `symthaea-semiconductor::benchmark::multichannel_drift_episode`  
**Status:** 16 authored Rust unit tests; exact-head validation pending.

## Paradigm

A fab's "drift alarm" is not a root-cause diagnosis. This implementation separates four evidence planes:

1. Input feature distribution — feature-window evidence, before labels are required.
2. Qualified prediction residual — accepts only a current K1 numeric-bound label eligibility report and a prediction-receipt digest.
3. Measurement system — accepts a current K1 calibration requalification report whose current frontier is Qualified and downstream-eligible.
4. Process/context — declared recipe, maintenance or process-event evidence.

Every output explicitly records causal attribution as NOT_ESTABLISHED. The system never promotes simultaneous alarms to a physical cause.

## Detector primitives

- EWMA on caller-standardized input/residual signals.
- Two-sided CUSUM with a declared slack.
- Missingness-rate EWMA.
- Explicit event-change channel for calibration/process events.

Detector policy, feature schema, baseline digest, clock domain, event/observation/frontier times, scope, sufficient counts, evidence receipt and before/after state digests are retained.

This is a first, interpretable detector set. MMD/energy distance and multimodal learned drift features are later benchmark alternatives, not silently mixed into v0.1.

## Evidence requirements

- Input features require FeatureWindow evidence and a valid feature/baseline digest.
- Residual observations require a LabelConsumptionEligibilityReportV1 with:
  - current K1 frontier;
  - EligibleUnderDeclaredPolicy;
  - downstream consumption allowed;
  - report digest reproduced;
  - evidence receipt equal to report digest;
  - exact source label ID;
  - measurement-event ID match;
  - prediction receipt digest.
- Measurement-system observations require a CalibrationRequalificationReportV1 with:
  - exact schema and report digest;
  - measurement-event ID match;
  - current K1 status Qualified;
  - downstream consumption allowed at that frontier;
  - K1 report time not beyond the observation's knowledge frontier;
  - no equipment-control authority.
- UNKNOWN/BLOCKED labels and unqualified calibration reports cannot change detector state. Their denial is recorded in an immutable observation receipt.

A content digest is not issuer authentication. The drift engine verifies report consistency and its declared evidence contract, not who physically issued it.

## Common-cause association

SharedResourceGraphV1 explicitly maps tool/chamber contexts to shared resources (for example, calibration references, shared metrology instruments, recipe versions or common maintenance utilities).

The association evaluator:
- requires a valid, sealed dependency graph;
- rejects duplicate binding tuples;
- returns UNKNOWN if graph inventory is incomplete or an alert context is not represented;
- reports an association candidate only when at least two alerted contexts share the same declared resource ID and resource kind;
- reports NO_DECLARED_SHARED_RESOURCE only when all alerted contexts are represented and no shared resource exists.

The result remains causal-attribution NOT_ESTABLISHED. Coincidence alone never produces a shared-resource association.

## Episode receipts

A sealed episode contains sorted immutable observation receipt digests, active evidence planes, alerting channels and denied-observation count. It rejects tampered/duplicate/out-of-window receipts and cannot be sealed as causal without a future external review receipt type.

## Authored challenge tests

Sixteen tests cover:
1. feature CUSUM alarm after minimum support;
2. UNKNOWN and BLOCKED residuals do not mutate state;
3. non-K1 residual evidence is rejected;
4. tampered K1 report is rejected;
5. duplicate evidence receipt idempotency;
6. out-of-order observation rejection;
7. baseline/schema identity change requires a new channel;
8. sensor missingness-rate tracking;
9. process and calibration remain separate evidence planes;
10. insertion-order-invariant episode digest and explicit no-causality state;
11. tampered observation receipt cannot be sealed;
12. coincidence without declared shared resource does not produce common-cause association;
13. incomplete resource inventory returns UNKNOWN;
14. a context missing from a claimed-complete inventory returns UNKNOWN;
15. qualified calibration evidence is required for the measurement-system plane;
16. UNKNOWN calibration qualification does not update the measurement-system channel.

These are authored tests, not yet reported as passing.

## Benchmark targets (hypotheses)

- 100% residual-monitor refusal for non-K1, UNKNOWN, BLOCKED, report-tampered or prediction-unbound labels.
- 100% measurement-system refusal for missing, stale, malformed, UNKNOWN or REJECTED calibration requalification reports.
- 0 state change for denied observations.
- 0 shared-resource associations from coincidence alone.
- 100% UNKNOWN for incomplete or unrepresented resource inventories.
- >=95% detection of planted sustained shifts within a preregistered horizon at <=1 false alarm per 1,000 in-control wafers, separately by shift family.
- Report detection delay, false alarms, missed shift, evidence coverage and attribution abstention separately.

Synthetic detector thresholds are not universal manufacturing limits. Targets require preregistered seeds, shift sizes, window lengths and in-control distributions.

## Claim ceiling

This is an evidence-qualified change detector and association ledger. It does not infer physical root cause, equipment failure, process excursion, wafer disposition or corrective action; it does not authorize APC/R2R/recipe/equipment control.

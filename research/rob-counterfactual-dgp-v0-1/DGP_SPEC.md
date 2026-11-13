# MCR² v17 DGP contract

## Population and assignment

Each scenario generates 10,000 matched blocks and 20,000 episodes. Each block contains one reference-policy and one anticipatory-policy episode, assigned in randomized slot order before time zero. Assignment receipts and high/low pre-assignment risk strata are observed. Both policies are synthetic safe-policy candidates; no unsafe physical no-intervention arm is defined.

## Potential outcomes

The evaluator oracle records reference and anticipatory event times, competing event time, and both terminal outcomes. Exponential event hazards depend on the risk stratum: reference boundary hazard 0.12 high / 0.02 low; candidate policy changes the high-risk hazard to 0.06 and leaves low-risk hazard unchanged. Competing terminal hazard is 0.004. Fixed horizon is 20.

Each episode is assigned one policy. The observed record reveals only its assigned-policy outcome, if endpoint ascertainment succeeds. The unassigned potential outcome and reference event time remain oracle-only. This is an evaluator design, not a claim that real robot counterfactuals are directly observable.

## Endpoint ascertainment regimes

| Scenario | P(endpoint observed given pre-assignment stratum) |
|---|---|
| Complete follow-up | 1.00 high / 1.00 low |
| MCAR endpoint loss | 0.90 high / 0.90 low |
| Conditional endpoint loss | 0.48 high / 0.96 low |
| Positivity failure | 0.00 high / 0.96 low |

Missing endpoints remain Unknown with an explicit reason (administrative endpoint not obtained or telemetry loss); they are not event-free. Competing terminal events remain separate.

## Evaluation rules

- Ground-truth population risk comes from the evaluator-only oracle.
- Complete-case risk is a transparent biased baseline when missingness depends on observed risk.
- Known-probability endpoint adjustment is an oracle diagnostic using DGP ascertainment probabilities, not a fitted model, survival estimator or generic IPW implementation.
- If any target stratum has zero endpoint observation probability, refuse the full-population point estimate; do not extrapolate through a positivity gap.
- Policy value is evaluated by randomized assignment (ITT) when fixed-horizon assigned outcomes are complete. The simple unadjusted ITT summary refuses incomplete follow-up but preserves denominators and unknown counts.
- Received-intervention contrasts are non-ITT. Receipt is conditional on pre-assignment risk and can reverse the randomized policy-effect direction.
- Observing one assigned policy does not identify an individual episode's counterfactual.
- This endpoint-risk task does not measure landmark time-to-event forecast calibration; that belongs to #9690.

## Integrity

The seed and scenario index determine repeatable per-scenario seeds. Each block has exactly one assignment per policy, within-stratum matching, pre-time-zero assignment and a valid receipt. Observed artifacts may not contain oracle-only fields. Repeat-run determinism, schemas and regression tests are required.

## Scope

All values and units are synthetic. This is a protocol testbed for endpoint semantics and bias/identifiability diagnostics, not a physical robot dynamics model, fitted statistical method, production controller or SOTA claim.

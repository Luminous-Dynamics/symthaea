# MCR² v17 DGP contract

## Population and assignment

For each scenario, generate 10,000 matched blocks and 20,000 episodes. Every block contains one reference-policy and one anticipatory-policy episode, assigned in randomized slot order before time zero. Assignment receipts and the pre-assignment high/low risk stratum are observed covariates. The two policy arms are separately qualified synthetic safe policies; no unsafe physical no-intervention arm is defined.

## Potential outcomes

The evaluator oracle records reference and anticipatory potential event times, competing event time, and both policy-specific terminal outcomes. Event times are generated from exponential hazards conditional on the risk stratum. The reference boundary hazard is 0.12 for high risk and 0.02 for low risk. The candidate policy changes the high-risk hazard to 0.06 and leaves the low-risk hazard unchanged. A competing terminal hazard of 0.004 is shared within each episode's potential paths. A fixed horizon of 20 defines the policy endpoint.

Each episode is assigned only one policy. Its observed record may reveal only that assigned policy's terminal outcome, if endpoint ascertainment succeeds. The unassigned potential outcome and untreated/reference event time remain oracle-only. This split is an evaluator design, not a claim that real robot counterfactuals are directly observable.

## Endpoint ascertainment regimes

| Scenario | Probability endpoint observed given pre-assignment stratum |
|---|---|
| Complete follow-up | 1.00 high / 1.00 low |
| MCAR endpoint loss | 0.90 high / 0.90 low |
| Conditional endpoint loss | 0.48 high / 0.96 low |
| Positivity failure | 0.00 high / 0.96 low |

A missing endpoint is represented as Unknown with a reason (administrative endpoint not obtained or telemetry loss). It is not converted to event-free. A competing terminal event is its own endpoint category and is not silently treated as an ordinary censor.

## Evaluation rules

- Ground-truth population risk comes from the evaluator-only oracle.
- Complete-case risk is a transparent biased baseline when missingness depends on outcome risk.
- The known-probability endpoint adjustment is an oracle diagnostic only, using the known DGP ascertainment probabilities. It is not a fitted model, survival estimator or general-purpose IPW implementation.
- If any target-population stratum has zero endpoint observation probability, refuse a point estimate for the full target population. Do not extrapolate through a positivity gap.
- Policy value is evaluated by randomized assignment (ITT) only when the assigned fixed-horizon endpoints are completely observed. Incomplete ITT is marked non-estimable by this simple unadjusted summary, while denominators and unknown counts remain visible.
- Received-intervention contrasts are explicitly non-ITT. In this DGP, receipt is conditional on pre-assignment risk and can reverse the direction of the randomized policy effect.
- Individual episode-level counterfactuals are not identified by observing one assigned policy.
- This fixed-horizon endpoint-risk task does not assess landmark time-to-event prediction calibration; that is a separate #9690 lane.

## Determinism and integrity

The seed and scenario index define deterministic per-scenario seeds. Each matched block must have exactly one episode assigned to each policy, matching pre-assignment stratum. Assignment time must precede time zero, randomization receipt must validate, and the observed artifact may not contain oracle-only fields. Schema validation, repeat-run determinism, and regression tests are required.

## Scope

All values and units are synthetic. The generator is a protocol testbed for outcome classification and bias/identifiability diagnostics. It is not a physical robot dynamics model, a fitted statistical method, a production controller or a claim of SOTA.

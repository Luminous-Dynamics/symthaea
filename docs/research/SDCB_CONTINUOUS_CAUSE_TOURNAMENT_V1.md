# SDCB continuous common-cause design tournament v1

**Status:** preregistration proposal; no tournament or real-data evaluation has been run.  
**Exact reference:** \`scripts/research/sdcb_continuous_cause_gaussian.rs\`  
**Related work:** #13802 measurement-assurance benchmark plan; #13804 finite binary common-cause planner; #13805 evidence-gated joint belief ledger; #13831 paired-world identifiability oracle.

## 1. Primary claim under test

At identical measurement candidates, prior information, calibration assumptions, resource budgets, and terminal scientific objective, a common-cause-aware selection policy reduces held-out uncertainty or decision regret compared with policies that treat sensor channels as independent, while remaining non-inferior when errors truly are independent.

This programme is a **measurement-selection** tournament, not merely a sensor-prediction contest. It must keep model quality and policy quality separately identifiable.

## 2. Task families

| Family | Generated structure | Required behavior |
|---|---|---|
| G0 independent controls | Independent residual errors; no shared latent factor | Repetition/independent-channel evidence should be used efficiently |
| G1 one shared offset | Multiple science channels load on one shared continuous factor | Do not count channel count as independent evidence |
| G2 multiple common factors | Environment, reference chain, and batch factors have distinct loadings | Select the right independent checks/references |
| G3 common + local | Shared factor plus device-specific latent effects | Preserve common/local uncertainty separately |
| G4 partial ancestry | Devices share some reference-chain links but not all | Do not count a common ancestor twice |
| G5 temporal drift | Shared/local factors evolve over time under frozen OU/Markov generators | Value current check standards over stale references |
| G6 correlated residual misspecification | Residual noise is correlated but the model omits a required latent factor | Detect overconfidence or model inadequacy; do not claim exact calibration |
| G7 correlated measurand | Physical signal is spatially/temporally correlated while instrument errors may be independent | Do not mislabel signal correlation as sensor common cause |
| G8 weak identifiability | Scientific and nuisance loading vectors are nearly collinear | Report poor conditioning and retain uncertainty |
| G9 event-conditioned shift | Ventilation/door/window/occupancy event changes latent environment or sensor response | Reassess measurement validity and select an event-relevant check |
| G10 contaminated reference | The reference/check measurement is biased or shares the suspected chain | Do not treat it as an independent anchor |
| G11 nonlinear boundary | Small linear-Gaussian approximation is inadequate | Route to a richer model or abstain; do not extrapolate v1 guarantees |

G0–G3 and G8 are suitable for exact Gaussian fixtures. G4–G7 and G9–G11 need richer graphs, sequential dynamics, physical metadata, or deliberately misspecified controls.

## 3. Comparator arms

All methods receive the same candidate actions, costs, priors, device/reference ancestry, training evidence, held-out measurements, and hard budgets.

1. Random feasible selection.
2. Fixed-cadence reference allocation.
3. Equal allocation across devices/locations.
4. I.i.d. sensor-noise fusion and EIG (deliberate negative control).
5. Greedy total-information / D-optimal design.
6. A-optimal design (trace of posterior covariance).
7. c-optimal design targeting the scientific parameter.
8. EVSI/decision-focused acquisition once a declared terminal utility is supplied.
9. Hierarchical/random-effects calibration.
10. Correlated-noise Gaussian process / space-time covariance model where the data support it.
11. Robust maximin or distributionally robust selection across frozen covariance/noise scenarios.
12. Common-cause-aware SDCB.
13. Exact exhaustive subset oracle on small cases (evaluator reference only).

Do not compare D-optimal log-determinants, c-optimal target variance, EIG, and EVSI as though they were the same objective. Report each policy against the same realized downstream metric and resource frontier.

## 4. Factorial separation of model and policy

Run three mandatory experiments:

**A — Fixed model, different policies.** Hold the calibration predictor and uncertainty model constant; vary only the action-selection policy.

**B — Fixed policy, different models.** Hold the candidate selector constant; vary i.i.d., hierarchical, common-factor, and correlated residual models.

**C — Joint factorial comparison.** Cross model and policy factors to measure interaction effects.

This isolates improvements due to better prediction/calibration from improvements due to decisions about what to measure.

## 5. Metrics

**Scientific target**
- held-out RMSE/MAE and bias in physical units;
- posterior variance of the declared scientific coordinate;
- interval coverage and interval width;
- target uncertainty reduction per reference cost;
- downstream decision regret/utility where utility is predeclared.

**Measurement-system model**
- coverage and calibration of common/local factor posteriors;
- common-factor attribution and partial-ancestry error;
- residual correlation estimation error;
- false common-cause and missed common-cause rates;
- drift-detection delay and false alarms;
- stale-reference and contaminated-reference detection.

**Selection efficiency**
- science, reference, check-standard, and intervention counts;
- cost, latency, memory, energy, and reference transport/use where available;
- target variance versus cumulative cost;
- regret against exact small-instance oracle;
- selected-set diversity and repeated ancestry exposure.

**Assurance**
- invalid/wrong-chain/stale/duplicate/revoked receipt acceptance;
- calibration-context mismatch acceptance;
- zero-mutation rejection of invalid evidence;
- provenance and uncertainty-budget completeness;
- replay agreement.

Never collapse these into a single weighted score. Publish target-quality/cost/latency Pareto curves and the individual assurance dimensions.

## 6. Preregistered gates (targets, not results)

- 100% agreement with the exact Gaussian oracle on frozen known-answer cases.
- 100% hard-budget compliance in all policy arms.
- Zero false independent-confirmation claims on G1/G2/G4/G10 controls.
- Non-inferiority on G0 independent-error controls (proposed margin: no more than 2% target-variance or decision-quality degradation against the best matched independent baseline).
- At least 10% lower scientific-target posterior variance per equal cost than hypothesis-only EIG/i.i.d. selection on held-out G1–G4/G8 cases; require paired 95% confidence intervals excluding zero.
- Approximate scalable selector regret no greater than 2% versus exact subset oracle on held-out instances within the exact oracle's action cap.
- Nominal 90% and 95% interval coverage assessed separately; no average-only pass can hide factor-specific undercoverage.
- Zero evidence-ledger mutation for rejected receipts.
- At least one independently reproduced real-data result before claiming an empirical advantage.

The exact 100% targets apply only to finite known-answer controls, not arbitrary physical measurements.

## 7. Real-data tournament

### Inha 2026 indoor calibration

Use the paper's S1 reference-efficient, S2 leave-location-out transfer, S3 chronological drift, and S4 event-conditioned tracks. The paper compares iTransformer, TimeXer, DLinear, XLinear, TimeMixer, SenDaL, TESLA, and its lightweight temporal model. Its model has the lowest reported RMSE among listed comparators for four of five S1 variables and four of five S2 variables; it has the lowest reported relative drift increase for all five variables. In S4, the top result varies by condition/model; the proposed model is first or second in the four reported conditions. These are author-reported calibration-model results, not experiment-selection results, and must be reproduced before use as a baseline claim.

The dataset spans six months and five locations in one building. It is useful for reference-budget allocation, location transfer, drift, and event-conditioned calibration; it does not by itself establish hidden reference-chain, power/firmware, or common-cause ground truth. Event metadata are recorded at fixed intervals rather than exact event times, so treat S4 as event-conditioned periods rather than perfectly aligned event response.

Official release: https://github.com/inhaKDD/Indoor-Sensor-Calibration-Dataset  
Paper: https://arxiv.org/abs/2610.11236

### UCI Gas Sensor Array Drift

Use chronological batches for temporal drift/channel robustness only. Sixteen channels from one array are not sixteen independent deployed instruments. The UCI dataset pages have conflicting licence statements: the narrative says research-only/no commercial use, while the structured licence field reports CC BY 4.0. Treat this as unresolved pending written clarification; conservatively handle the data as research-only and do not redistribute or use commercially until clarified.

### Synthetic causal/common-cause tournament

Generate labelled shared environmental, reference-chain, batch, electronics, and local perturbations with exact ground truth. This is the only current route to measuring common-cause attribution error cleanly. Keep injected counterfactual perturbations separate from unmodified observational data.

## 8. Industry comparison and differentiated contribution

- NIST measurement-assurance programmes emphasize cross-comparisons, check standards, control charts, and evidence about the stability of a measurement process; NIST does not mandate one universal recalibration interval.
- Sensor-network metrology research already models space-time-correlated measurement errors and shows that i.i.d. assumptions can change propagated uncertainty.
- 2026 optimal-design research shows that correlated versus uncorrelated observation noise can alter optimal sampling schedules.
- The Inha 2026 work supplies a deployment-oriented calibration benchmark across reference availability, unseen locations, drift, events, and edge efficiency.

Therefore, SDCB should not claim to invent sensor correlation, calibration networks, or optimal experimental design. Its testable differentiation is the integration of:
1. explicit common-cause/reference ancestry and local uncertainty;
2. target-scientific-parameter and decision-value-aware selection;
3. exact finite small-instance oracles for approximate-policy qualification;
4. paired-world identifiability and principled abstention;
5. evidence-bound receipts, independent verification, and zero-mutation invalid transitions.

References:
- NIST recommended calibration interval: https://www.nist.gov/calibrations/recommended-calibration-interval
- NIST metrological traceability TN 2156: https://www.nist.gov/metrology/metrological-traceability
- Kok et al. (2025): https://doi.org/10.1016/j.measen.2024.101793
- Qi & Baker (2026): https://doi.org/10.1016/j.mbs.2025.109571
- Yun et al. (2026): https://arxiv.org/abs/2610.11236

## 9. Reproducibility record

Pin the dataset release/asset digest, source commit, candidate-action registry, prior/noise scenario, split manifest, reference-query transcript, selected action set, exact oracle result, policy/model digest, random seed, hardware/software, resource telemetry, all failures/abstentions, evidence receipts, and licence/redistribution decision. Freeze model and policy before the held-out test.

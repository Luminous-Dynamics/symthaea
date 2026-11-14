# MARINE-OPERATIONAL-PREDICTION-BENCHMARK-001
## Leakage-resistant, physics-aware marine fuel prediction and transfer evaluation

**Status:** benchmark protocol v0.1.0  
**Created:** 2026-11-10  
**Related programmes:** MARINE-WAPS-CO-DESIGN-001 (#11406), MARINE-SERVICE-ROBUSTNESS-001 (#11429), MARINE-COASTAL-HYBRID-REPRODUCTION-001 (#11431)

> This protocol defines a benchmark design, not achieved model performance or a claim to outperform published methods.

## 1. Why this benchmark is necessary

A September 2026 ship-fuel prediction study by Park and Song reports that, on the Poseidon FuelCast ship case, a random split achieved R² = 0.975 ± 0.001 while a purged time-block split achieved R² = 0.943 ± 0.007. The resulting optimism bias was +0.032 R². The same study reports that monotonic constraints eliminated non-physical accumulated-local-effect reversals in its three measured ship datasets without loss of accuracy (reported ΔR² = +0.006), and reports CII prediction MAPE of 6.80% with 80.6% exact rating agreement and 95.8% within one rating grade.

These are author-reported results for three specific vessels, not cargo-fleet performance and not an independent Luminous replication. They nevertheless establish a strong methodological comparator: random-split scores are not sufficient, and physically implausible response surfaces must be tested explicitly.

## 2. Public data: separate the roles

### 2.1 FuelCast daily measured data

The September 2026 study uses the openly available FuelCast dataset with measured daily total fuel-consumption records and weather/operational variables:

| Ship | Type | Propulsion | Period | Rows |
|---|---|---|---:|---:|
| Poseidon | Large cruise ship | Diesel-electric | 12 months | 12,430 |
| Triton | Small cruise ship | Direct-drive diesel | 3 months | 9,575 |
| Ceto | Offshore support vessel | Thruster pods | 6 months | 7,102 |

The target is total daily fuel across onboard consumers, not propulsion-only shaft fuel. The records do not provide the cargo-vessel loading variation necessary for direct transfer to laden/ballast cargo service. Poseidon has no draft or trim variable; the other vessels have limited draft variation. The dataset is therefore appropriate for a public method-reproduction benchmark, not a cargo-vessel or wind-assisted propulsion performance claim.

Dataset: https://huggingface.co/datasets/krohnedigital/FuelCast  
Reported license: CC BY-NC-ND 4.0. Confirm the actual license and permitted commercial/non-commercial use before any ingestion, derivative analysis or distribution. Do not mirror, redistribute, modify or publish derivative data/weights without a rights review and appropriate permission.

### 2.2 EU MRV annual fleet records

The same study uses public EU Monitoring, Reporting and Verification data for annual ship-level diagnosis and fleet benchmarking. It reports 16,273 latest-report vessels and a 2018–2025 panel of 24,845 ships / 100,550 rows for its analysis.

Annual aggregates are valuable for annual emissions/CII diagnosis, fleet segmentation and external consistency checks. They are not time-resolved voyage labels and must not be used to identify daily speed–fuel response or train a voyage fuel predictor by disaggregating annual totals.

Official portal: https://mrv.emsa.europa.eu/

### 2.3 Cargo-vessel evidence remains the missing critical dataset

The coastal hybrid 2026 study uses over 120 voyages from 24 months of AIS/logbook data but its data and code are available from the corresponding author upon reasonable request, rather than as a public raw dataset. This is the most relevant route-specific coastal reproduction target, but an independent raw-data reproduction requires permission and access.

For cargo/WAPS validation, partner data should include voyage- and time-resolved fuel mass flow, shaft torque/RPM, engine and electrical loads, speed through water and over ground, current, wind/waves, draft/trim/loading, route, cargo quantity, WAPS device state/auxiliary power, calibration, timebase and maintenance events.

## 3. Required baseline models

All candidate models must receive the same eligible inputs and be evaluated on identical folds.

1. Naive persistence / voyage-class average baseline.
2. Physics reference: declared calm-water power-law or resistance/propulsion curve, with explicit validity range; not treated as ground truth.
3. Unconstrained Random Forest and XGBoost.
4. Physically constrained Random Forest and XGBoost, where the monotonic direction is justified for the exact input variable and operating slice.
5. Regularized linear or generalized additive baseline for interpretability.
6. Optional sequence models only when dataset size and sampling density justify them; report their extra data demand and compute budget.

Do not compare tuned candidates with an untuned baseline or tune hyperparameters on the final holdout.

## 4. Leakage-resistant split protocol

Minimum evaluation ladder:

- **P0 — reproduction:** recreate the paper's purged time-block split as closely as the published data and implementation allow: contiguous 20% validation block, 2% purge on both sides, five rotated block positions, and split-consistent out-of-fold predictions. This is a protocol reproduction target; it does not require exact source-code reproduction if the code is not publicly available.
- **P1 — future block:** train on earlier blocks and test on later contiguous blocks, including a predeclared purge gap.
- **P2 — voyage holdout:** group all time samples from a complete voyage together; hold out entire voyages.
- **P3 — route/season holdout:** hold out full route patterns, seasons or years.
- **P4 — ship-group holdout:** leave entire ships out when the fleet sample is sufficiently diverse.
- **P5 — external transfer:** evaluate on a partner-provided vessel, route or WAPS installation that was not used for model selection.

Random row splits may be reported only as a diagnostic to quantify optimism bias. They are never the headline generalization result. Time-block folds must be preserved during hyperparameter selection and feature selection.

## 5. Feature and target leakage audit

Before training, build a machine-readable field lineage and exclude any feature that is mathematically derived from the target or contains future information. Examples include fuel per distance, CO₂ per distance, total CO₂ from fuel, daily fuel totals, post-voyage summaries, future AIS-derived route statistics and outcome-derived vessel reports.

Keep separate targets for:
- total onboard fuel mass;
- propulsion-engine fuel;
- shaft energy;
- electrical/hotel energy;
- fuel/primary energy per delivered cargo work;
- emissions and CII indicators.

Never infer propulsion-only energy from total onboard fuel unless the allocation model and its uncertainty are explicit. Never infer daily consumption directly from annual MRV records.

## 6. Physics-informed constraints must be conditional

The 2026 study's monotonic constraints are a strong starting point, but WAPS introduces important exceptions.

Candidate constraints, subject to physical review:
- At fixed hull, loading/draft, route, speed through water, machinery state and environment, increasing propulsion demand should not reduce required delivered shaft power.
- At fixed speed and fixed relative direction, increased adverse relative wind or wave severity may be constrained to non-decreasing added resistance within a declared domain.
- Calm-water power may increase with speed over a declared displacement-hull operating region, but the service-level objective must still account for schedule, current and delivery constraints.
- Current direction, relative wind angle, wave encounter direction, WAPS deployment state, trim, reefing/retraction and controller state must not be assigned a global monotonic sign without a specific physical derivation.

In particular, globally imposing “more wind speed means more fuel” is invalid for a wind-assisted ship: favourable wind direction and device state can reduce engine demand. Likewise, a following current can increase SOG for the same STW. Constraints must be conditional on relative direction, speed basis and device state, and the unconstrainted interactions must remain visible.

Measure monotonicity violations on held-out samples and on controlled counterfactual sweeps. A constraint must not force a physically incorrect response in a favourable-wind or following-sea regime.

## 7. Required metrics

Report per ship, voyage, route, season and operating-condition slice:

- R², RMSE, MAE and signed bias;
- normalized MAE / RAAE and worst normalized error / RMAE;
- P50, P90 and P95 absolute error;
- WAPE or another clearly defined fleet-weighted aggregate;
- 90% and 95% prediction-interval empirical coverage and interval width;
- calibration curves and residual dependence on speed, draft, weather and load;
- physical monotonicity violation count/rate;
- runtime, memory and training data requirement;
- out-of-domain rejection rate and false-feasible rate.

For cargo service, also report fuel/primary energy per delivered tonne-nautical-mile or TEU-nautical-mile, annual delivered work, on-time service and vessel availability. Prediction error in t/day cannot substitute for a service-level energy comparison.

## 8. Initial targets (research goals, not results)

- Reproduce the published +0.032 R² random-split optimism-bias direction on FuelCast; quantify the difference without assuming the exact value will match.
- 0 feature-leakage violations.
- 0 random-row-split scores presented as primary generalization.
- 100% split and preprocessing provenance.
- 0 non-physical monotonic violations in the specifically declared constrained domains, with an explicit report of unconstrained domains.
- Nominal 90% prediction intervals should achieve 85–95% empirical coverage overall and per supported slice; report sharpness.
- Median absolute percentage error <=5% is an aspiration for admitted cargo-vessel held-out shaft-energy prediction, not a FuelCast acceptance requirement.
- Do not publish a fuel-saving claim from predictive accuracy alone; savings require matched counterfactual or physical trial evidence.

## 9. How this can improve on current SOTA

The 2026 physics-informed ship-fuel paper already contributes monotonic tree ensembles, an arithmetic feature-leakage audit, purged time blocks, split-consistent OOF scoring and cross-dataset role separation. We should not claim novelty for those components alone.

Our research opportunity is to extend them across cargo and wind-assisted systems:
- ship/voyage/route/season/port grouped holdouts;
- load/draft/trim and cargo-service state;
- STW vs SOG and vector current/wind/wave direction;
- WAPS device-state-conditioned physics;
- total-fuel vs propulsion-fuel vs shaft-energy boundary separation;
- calibrated uncertainty and explicit out-of-domain abstention;
- degradation/failure modes and forecast uncertainty;
- independent service-level counterfactual/sea-trial qualification.

Any advantage must be shown on blind external cargo-vessel data and must exceed combined measurement and model uncertainty.

## 10. References

- Park, S.-U. & Song, C.-Y. (2026), “Physics-Informed Machine Learning with Monotonic Constraints for Daily Fuel Consumption and CII Prediction of Ships”, Journal of Marine Science and Engineering 14(19), 1774, published 23 September 2026. https://doi.org/10.3390/jmse14191774
- FuelCast dataset, KROHNE Digital, CC BY-NC-ND 4.0 (accessed by the paper 1 May 2026): https://huggingface.co/datasets/krohnedigital/FuelCast
- European Commission / EMSA THETIS-MRV: https://mrv.emsa.europa.eu
- Du et al. (2026), coastal cargo hybrid-electric reproduction target, Applied Sciences 16(10), 4940. https://doi.org/10.3390/app16104940

## 11. Claim ceiling

A score on FuelCast establishes performance only on the measured ship/time ranges permitted by the data rights. EU MRV annual aggregates establish annual fleet-level indicators only. Neither dataset establishes cargo-vessel WAPS benefit, deep-sea generalization or safe operation. Cargo/WAPS claims require cargo-specific matched evidence, relevant physical validation and an independently reviewed measurement chain.

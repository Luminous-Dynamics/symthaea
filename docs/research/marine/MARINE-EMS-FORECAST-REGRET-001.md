# MARINE-EMS-FORECAST-REGRET-001
## Deployable forecast-aware energy management and conditional physics constraints

**Status:** research specification v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

> This specification defines benchmark hypotheses and an experiment contract. It does not describe a validated controller, vessel fuel-saving result or safety-qualified control system.

## 1. Research objective

Compare practical energy-management policies against the offline perfect-foresight optimum without confusing an unattainable oracle with a deployable controller. The benchmark measures policy regret under forecast errors, missing data, delays, model uncertainty, degradation and equipment failures while preserving the same cargo service and operating constraints.

The second objective is to improve fuel/shaft-power prediction through conditional physical structure. A WAPS ship cannot safely inherit global monotonic assumptions about raw wind speed: a favourable wind can provide useful propulsion, while an adverse angle or retracted device may increase demand.

## 2. Industry baseline and scope limits

Park & Song, *Physics-Informed Machine Learning with Monotonic Constraints for Daily Fuel Consumption and CII Prediction of Ships*, JMSE 14(19), 1774 (23 September 2026), DOI 10.3390/jmse14191774, report:
- constrained XGBoost time-block R² 0.943 ± 0.007 on Poseidon daily total fuel;
- random-split R² 0.975 ± 0.001, a +0.032 optimism gap over time-block validation;
- eliminated reported accumulated-local-effects monotonic reversals across three vessels;
- CII prediction MAPE 6.80%, exact rating agreement 80.6% and within one grade 95.8%.

Those records represent two cruise vessels and one offshore support vessel. The target is total daily onboard fuel, not propulsion-only fuel. FuelCast is reported under CC BY-NC-ND 4.0; no cargo/WAPS generalization or unrestricted commercial derivative use is permitted. Use it only for a rights-reviewed methodology reproduction.

EU THETIS/MRV annual ship-level aggregates are suitable for fleet emissions/CII diagnostics and external consistency checks, not daily fuel prediction labels or causal speed-fuel response.

## 2A. New route-risk SOTA comparators (November 2026)

Two recent route-risk studies should be included in the methodology comparison, without ranking their published percentages against one another:

**Battal, Transportation Research Part D 160 (November 2026), Article 105598, DOI 10.1016/j.trd.2026.105598.** The paper reports a distributional-RL/QR-DQN policy with Wasserstein distributionally robust optimization on a Rotterdam–New York corridor. The abstract reports 17.9% lower fuel than slow steaming and 5.1% lower fuel than standard DQN, with about 390 t CO2 per voyage and near-zero adverse-weather exposure. These are author-reported results for one corridor, not an independently reproduced Luminous result or a transferable WAPS forecast.

**Can, Journal of ETA Maritime Science (2 October 2026), DOI 10.4274/jems.2026.95826.** A risk-aware digital twin for the İstanbul–Piraeus route combines AIS observations, a reduced PDE–ODE resistance/navigation model, CVaR and distributionally robust optimization. The article reports a speed change from 20 kn to approximately 16 kn, about 24% lower CO2 and about 26% lower CVaR-based tail risk under its adopted assumptions. The authors explicitly state the reduced-order model is not intended to replace high-fidelity CFD.

These studies expand the benchmark baselines beyond deterministic weather routing and conventional MPC. Their numerical results are **not cross-paper comparable**: vessel, corridor, fuel/CO2 boundary, weather samples, service contract, cost assumptions, and baseline policies differ. Extract the methods and reported case data first, reproduce each published case separately where inputs permit, and then compare algorithm families on one shared frozen service/route/weather corpus.

### Required routing/decision baselines (expanded)

- Deterministic shortest/least-fuel route and conventional weather routing.
- Slow steaming and fixed schedule-speed policy.
- Rule-based speed/energy management.
- Standard DQN or equivalent point-value RL.
- Distributional RL (e.g. QR-DQN) without DRO.
- Distributional RL + Wasserstein DRO (or a faithfully reproduced ambiguity-set variant).
- Risk-aware digital-twin speed optimization with CVaR + DRO.
- Deployable receding-horizon MPC and robust/stochastic MPC.
- Clairvoyant perfect-foresight dynamic programming as an explicitly non-deployable oracle.

For a fair common-case comparison, all methods receive the same forecast snapshots available at decision time, same action frequency, same service and safety constraints, and identical initial vessel/load/SOC state. Score route length and fuel only alongside ETA distribution, on-time probability, adverse-weather exposure, p95 delay, fuel/energy uncertainty, replan latency and hard-constraint violations.

## 3. Baselines and evaluation

B0 measured operating policy; B1 slow steaming/fixed-speed and deterministic weather routing; B2 rule-based EMS; B3 greedy/no-lookahead EMS; B4 standard DQN/point-value RL; B5 distributional RL (QR-DQN); B6 distributional RL + Wasserstein DRO; B7 risk-aware digital twin with CVaR + DRO; B8 deployable receding-horizon/MPC; B9 robust/stochastic MPC; B10 offline perfect-foresight DP oracle; B11 conditional-physics predictor vs unconstrained ML and simple physical baselines.

Freeze voyage, service, forecast snapshot, route, load, metocean, equipment, fuel boundary and model revision. Split by voyage/route/season; purge correlated adjacent windows. Only forecasts available at decision time may enter deployable policies. Observed future weather/reanalysis is not a forecast.

Measure regret against the oracle separately for fuel, primary energy, lifecycle cost and service penalty. Include reserve, SOC, power/ramp, arrival, payload and WAPS operating-envelope hard limits. Report mean, median, p95 and per-condition distributions.

## 4. Conditional physical constraints

Candidate physically meaningful predictors include longitudinal aerodynamic assistance/resistance, apparent-wind angle, device trim/retraction state, longitudinal wave resistance proxy, speed through water and propulsor/shaft load.

Within an explicitly validated operating envelope:
- increasing propulsive longitudinal aerodynamic assistance at fixed service/load/environment/device state should not increase predicted required shaft power absent an identified coupling;
- increasing adverse longitudinal aerodynamic resistance should not decrease predicted propulsion demand;
- fuel/energy balances and battery SOC dynamics should be explicit equations where known;
- no global monotonic rule should be imposed on raw wind speed, angle, side force, yaw, heel or leeway without condition-specific evidence.

Every constraint must have an eligibility domain and negative controls. Conditional constraint violations outside that domain are not scored as if they were valid physical counterexamples; they are out-of-domain.

## 5. Proposed targets

- Zero future-observation leakage.
- 100% known-answer arithmetic/SOC fixture agreement.
- Zero hard-limit violations in the declared controller envelope; explicit fallback/block outside it.
- Time-block/route-held-out R², MAE, RMSE, signed bias, p95 absolute error and uncertainty coverage by condition.
- Zero conditional-physics violations on declared eligible test slices.
- Nominal 90% intervals target 85–95% empirical coverage, with width and stratification reported.
- Deployable controller median regret ≤5% and p95 regret ≤10% as research aspirations vs the clairvoyant reference, on matched service.
- Independent repetition across at least two held-out route/season blocks before a claim is upgraded.

These targets are not achieved results or guaranteed capabilities.

## 6. Data and implementation requirements

Seek permissioned cargo-operator datasets with fuel by propulsion/hotel boundary, shaft torque/RPM, mass fuel flow, speed through water and over ground, loading, wind-device state, metocean and timestamp/calibration records. Preserve FuelCast rights and do not convert annual MRV aggregates into synthetic daily truth.

Keep the controller advisory; no direct navigation or actuator authority. Physical control requires separate hazard analysis, class/flag/owner review and physical qualification.

## 7. Sources

- Park & Song, JMSE 14(19), 1774 (23 September 2026): https://doi.org/10.3390/jmse14191774
- FuelCast dataset: https://huggingface.co/datasets/krohnedigital/FuelCast
- EU THETIS MRV: https://mrv.emsa.europa.eu
- Battal, *Robust low-carbon ship routing under weather uncertainty: a distributional reinforcement learning approach*, Transportation Research Part D 160 (November 2026), 105598: https://doi.org/10.1016/j.trd.2026.105598
- Can, *Risk-Aware Digital Twin-Based Speed Optimization for CO2 Emission Reduction in Short-Sea Shipping: The İstanbul-Piraeus Route*, Journal of ETA Maritime Science (2 October 2026): https://doi.org/10.4274/jems.2026.95826
- Du et al., Applied Sciences 16(10), 4940 (15 May 2026): https://doi.org/10.3390/app16104940

## 8. Claim ceiling

Prediction accuracy is not fuel reduction. Regret vs a clairvoyant model is not evidence of safe autonomous control. No benchmark score implies vessel approval, seaworthiness or regulatory compliance.
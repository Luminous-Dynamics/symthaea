# MARINE-BENCHMARK-EXCHANGE-001
## Versioned marine engineering benchmark manifest and comparability contract

**Status:** research specification v0.1.0  
**Created:** 2026-11-11  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Implementation:** benchmark manifest schema, synthetic fixture and Rust auditor in PR #11410

> This is a protocol and evidence-exchange specification. It is not a benchmark result, certification, physical validation or claim of superiority.

## 1. Problem: marine performance comparisons are often not comparable

A fuel-saving percentage, R² score or CVaR improvement is meaningless outside its vessel, route, cargo service, boundary, weather data, baseline and operating assumptions.

The November 2026 literature makes this particularly visible:
- Distributional QR-DQN plus Wasserstein DRO reports one set of fuel/risk improvements on Rotterdam–New York.
- A CVaR+DRO reduced-order digital twin reports CO2 and tail-risk improvements on İstanbul–Piraeus.
- A separate hybrid-electric study uses a different coastal vessel, route profiles, power boundary and perfect-foresight controller.
- Specialist CFD validation reports observable-specific uncertainty for a defined hull and condition.

These publications cannot be ranked by headline percentages. The field needs benchmark packages that let an independent group reproduce each case and then compare methods on a common problem.

## 2. Benchmark taxonomy

Every benchmark manifest declares one task class:

1. hydrodynamic_validation — resistance, wake, wave profile, propulsor/rudder observables.
2. wind_assist_physics — WAPS force/moment maps and coupled hull/propeller response.
3. fuel_prediction — fuel/shaft/electrical prediction with leakage-resistant holdouts.
4. voyage_routing — route/speed decisions under forecast vintage and weather uncertainty.
5. energy_management — power split, SOC, fuel and reserve policy.
6. design_optimization — vessel architecture and multi-fidelity design.
7. trial_causal_effect — matched interventions and causal treatment effects.
8. regulatory_accounting — versioned statutory calculation and route applicability.
9. fleet_service — vessel assignment, delivery, reliability and throughput.
10. port_energy — berth, shore power, battery and turnaround co-design.

The taxonomy separates a prediction benchmark from a policy benchmark, a physical-model validation case from a design optimization, and a synthetic software test from a measured vessel trial.

## 3. Manifest contract

The v0.1.0 JSON Schema and synthetic fixture are committed at:
- marine-benchmark-manifest-v0.1.0.schema.json
- fixtures/synthetic-marine-benchmark-manifest-v0.1.0.json

Each manifest binds:
- benchmark ID and revision;
- task class and highest evidence/claim stage;
- exact reference case schema/version, case digest, service fingerprint and configuration revision;
- dataset evidence class, unit of analysis, time period, rights/access status, license basis, content digest, source references, transformations and limitations;
- split/holdout strategy and partition unit;
- forecast vintage, operational input availability and explicit prohibition on future observations;
- baseline family, model/version/configuration digest, deployability and oracle-only status;
- metric identity, units, boundary, direction, aggregation, stratification and uncertainty requirements;
- non-compensable hard gates;
- source registry and claim ceiling.

The manifest is a protocol, not a result. Numerical scores belong in separate run receipts that bind the manifest digest, run environment, code commit, input digests, output digests and exact metric calculations.

## 4. Fair-comparison invariants

1. **Same service:** compare policies/designs under the exact same service fingerprint. Environment realization remains a separate declared comparison boundary.
2. **Same information:** deployable policies receive only information available at decision time, with forecast issue time, valid time, latency, revision and ensemble metadata retained.
3. **Independent holdout:** split by voyage, route-season, vessel or purged time block. Adjacent windows from one voyage must not appear in both train and test.
4. **No silent data transformation:** disclose interpolation, filtering, corrections, exclusions, unit conversions and source rights.
5. **Same metric boundary:** a total daily fuel target is not propulsion-only fuel; tank-to-wake CO2 is not well-to-wake GHG; speed-trial error is not full-voyage energy error.
6. **Oracle separation:** perfect-foresight DP is an upper/lower-bound reference, never described as deployable.
7. **Non-compensable gates:** safety, service, reserve, evidence and regulatory hard gates cannot be offset by a lower energy score.
8. **No cross-paper ranking:** results from distinct vessels/routes/baselines stay separate until independently reproduced on the same frozen case.
9. **Uncertainty:** publish distribution, p50/p95/tail metrics, signed bias and interval calibration. Do not report a single mean as a complete performance claim.
10. **Claim stage:** synthetic, published-number reconstruction, published raw-data reproduction, component bench, CFD V&V, model test, sea trial, in-service and fleet transfer are different evidence levels.

## 5. Comparative tracks and target metrics

### Track A — contract/identity
- 100% schema and reference resolution.
- 0 duplicate identifiers or unresolved source/dataset references.
- 100% explicit unit, metric boundary and rights/access declaration.

### Track B — leakage and reproducibility
- 0 future-weather or target-arithmetic leakage in negative controls.
- 100% manifest/run-receipt digest agreement.
- Exact deterministic arithmetic and scoring fixtures.
- Reproducible run environment and data transformations.

### Track C — prediction
- Held-out voyage/route/season and vessel slices.
- R², MAE, RMSE, signed bias, p95 absolute error, calibration and interval width.
- 0 false physical-constraint violations within declared eligible domains.
- Explicit out-of-domain abstention.

### Track D — routing and control
- Fuel/primary energy per delivered cargo work.
- On-time probability, p50/p95 arrival delay and CVaR95 service penalty.
- Adverse-weather exposure, reserve/SOC violations, policy regret vs an explicitly non-deployable oracle.
- Forecast degradation, distribution shift, port congestion, equipment failure and replanning latency.
- No hard-gate violations.

### Track E — design and physical validation
- Pareto results for lifecycle cost, energy, WtW GHG, cargo capacity and reliability.
- Multi-fidelity optimization reports final high-fidelity/physical validation cost and invalidated surrogate wins.
- CFD/test-basin/sea-trial claims are observable-specific and uncertainty-qualified.

Targets must be frozen for each benchmark before scoring. A global threshold is not automatically appropriate for every ship class or measurement system.

## 6. Current implementation

The branch adds:
- Draft 2020-12 manifest schema;
- synthetic route-robustness manifest fixture;
- marine-benchmark-manifest-audit Rust binary;
- adversarial tests for unresolved sources, synthetic-to-physical claim escalation, future-observation leakage, deployable oracle and compensable hard gates;
- exact-head Marine Standalone Contracts workflow smoke step.

The synthetic fixture is expected to return INCOMPLETE because its leakage audits are intentionally not run. This is the correct disposition for a test manifest without physical evidence.

## 7. SOTA comparison use

The manifest standard lets Luminous compare:
- deterministic weather routing and slow steaming;
- standard DQN and distributional QR-DQN;
- QR-DQN plus Wasserstein DRO;
- reduced-order digital twin plus CVaR/DRO;
- MPC and robust/stochastic MPC;
- perfect-foresight DP as a non-deployable oracle;
- matched physical model, CFD and sea-trial observables.

A benchmark exchange is useful only if independent groups can reproduce the case without access to the original optimizer's hidden inputs. The manifest preserves the assumptions required to recreate it, while data rights remain with their owners.

## 8. Claim ceiling

A valid manifest proves only that a benchmark protocol is structurally declared and its references resolve. It does not prove that data are authentic, that a model is correct, that a policy is safe or that any vessel performs better. Performance superiority requires independent reproduction on the same frozen case and evidence stage.

# MARINE-WEATHER-ROUTING-BENCHMARK-001
## Reproduce and extend the open multi-basin weather-routing benchmark

**Status:** reproduction / common-method benchmark specification v0.1.0  
**Created:** 11 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)  
**Related:** #11473 forecast-regret EMS; #11583 benchmark exchange

> The external dataset is a first-principles simulation benchmark evaluated under historical reanalysis, not measured ship fuel data or a sea-trial result. Reproduction of its numbers does not independently validate the underlying physics model.

## 1. Dataset and attribution

- Dataset: Aredah and Rakha, Weather-Routing Benchmark: a multi-basin, multi-vessel instance set for ship weather-routing and speed-optimization research, v1.0.0 (2026).
- Concept DOI: https://doi.org/10.5281/zenodo.21052776
- Public source repository: https://github.com/AhmedAredah/weather-routing-benchmark
- License: CC BY 4.0 (retain attribution, license link and indication of modifications when sharing or adapting).
- Accompanying paper: Energy- and emission-optimal ship routing from a first-principles vessel model under real metocean conditions, listed as under review in CITATION.cff.
- Keep the external dataset unmirrored in Symthaea. Download the released version from its source/Zenodo and record the exact release, file digests, modifications and attribution in every run receipt.

The external README describes 12 routes across all major ocean basins, 4 representative seasonal dates, 2 years (2022 and 2023), 3 hulls (S175 containership, KVLCC2 VLCC, JBC capesize), and 2 objectives (minimum fuel and minimum voyage cost): 96 instances per vessel/objective and 576 priced voyages overall.

## 2. Why this is the right first routing benchmark

Unlike a paper reporting only a single aggregate percentage, this benchmark provides:
- a machine-readable 12-route campaign manifest and geographic corridors;
- standard vessel definition files for three distinct hull forms;
- six per-vessel/objective results CSVs with per-instance fuel, cost, duration, status and runtime;
- exemplar route tracks, significant-wave-height fields and leg-level powering profiles;
- a single explicit weather-blind baseline: calm-water-optimal route scored under the same historical weather;
- two distinct objectives (minimum fuel and minimum cost), so carbon-cost/regulatory exposure does not get confused with pure energy optimization.

The published summary reports:
- S175: route-only mean 5.04%, median 1.91%, p90 14.58%; joint route+speed mean 11.01%.
- KVLCC2: route-only mean 6.14%, median 3.91%, p90 16.07%; joint route+speed mean 10.95%.
- JBC: route-only mean 4.55%, median 1.82%, p90 15.09%; joint route+speed mean 9.57%.

These are author-reported model results, not Luminous results and not measured fuel-saving evidence. They should be reproduced from the per-instance CSVs before using them as a common baseline.

## 3. Evidence and model boundary

### 3.1 Metocean data

The benchmark uses ECMWF ERA5 reanalysis:
- hourly 10 m wind fields at 0.25°;
- significant wave height and mean wave period at 0.5°;
- bathymetry from GMRT and coastlines from Natural Earth.

Raw metocean fields are not redistributed by the benchmark. Reanalysis provides retrospective environmental inputs, not forecasts available at the historical decision time. Therefore:
- use the public package for deterministic retrospective routing algorithm comparison;
- do not report its performance as deployable forecast-aware control;
- add operational forecast vintage, issue/valid time, latency, ensemble and provider only in a separate forecast experiment.

### 3.2 Physics and vessel models

The authors describe a first-principles model using Holtrop–Mennen calm-water resistance, Lang–Mao spectral added resistance, Fujiwara wind resistance, Wageningen B-series propeller operating point and load-dependent SFOC, with engine-MCR, IMO MSC.1/Circ.1228 seakeeping, under-keel-clearance and current-triangle feasibility gates.

Important limitation: the benchmark vessel README says the KVLCC2 and JBC engine SFOC shapes are scaled from the S175 reference. The hull and propeller particulars are from published benchmark definitions, but those SFOC curves are not a substitute for vessel-specific measured engine maps. The model is useful for algorithm comparison on fixed assumptions, not proof of real ship fuel prediction accuracy.

### 3.3 Feasibility is a first-class output

The README reports status ok for 96/96 S175 instances, 75/96 KVLCC2 and 76/96 JBC, partly because deep draft makes some route/canal combinations infeasible irrespective of weather.

A further subset of status-ok instances has an infeasible weather-blind baseline in severe weather; the benchmark excludes undefined savings percentages from its reported summary. The importer must:
- preserve every original row and status;
- distinguish route infeasibility, baseline infeasibility and missing/NA metric;
- never coerce infeasible/undefined savings to 0%;
- report both feasibility counts and conditional savings statistics;
- count a routing-enabled feasible crossing separately from a mere fuel-saving comparison.

## 4. Canonical baseline and metric reproduction

The benchmark README and scripts/summarize_campaign.py define the canonical calculations. The Rust auditor must reproduce them exactly:

**Weather-blind baseline:** calm-optimal path under real weather on the same corridor.

**Route-only (operational, mesh-aware) saving:**

100 × (calmPathRealFuelKg − min(realFuelKg, meshFuelKg when available)) / calmPathRealFuelKg.

**Joint route + speed saving:**

100 × (calmPathRealFuelKg − min(speedOptFuelKg, meshSpeedOptFuelKg, awareSpeedOptFuelKg when available)) / calmPathRealFuelKg.

**Slow-steaming-only contribution:**

100 × (calmPathRealFuelKg − calmSpeedOptFuelKg) / calmPathRealFuelKg.

benefitPct is the non-mesh corridor result and must not be confused with the canonical combined mesh-aware route metric. Missing candidates remain missing. Use only status-ok rows with positive baseline fuel and a defined candidate for the conditional savings summary. Also report total rows, status counts and excluded baselines so the denominator is transparent.

Percentiles must use the external script's linear interpolation convention over sorted samples. Compare rounded outputs within a frozen tolerance; do not silently substitute a different p90 implementation.

## 5. First implementation work package

The isolated Rust tool should accept a local dataset root (not fetch data over the network) and audit:
1. the exact 1.0.0 external source manifest and attribution/CC BY 4.0 notices;
2. 12 unique route names and expected route coordinates/mesh configuration;
3. three vessel definition files and published particulars;
4. six results CSVs, 96 rows each, expected route/date cross product and unique instance keys;
5. result statuses, positive baseline fuel, numeric/NA semantics and no silent zero-fill;
6. route-only, mesh-aware combined, joint route+speed and slow-steaming metrics;
7. per-vessel feasibility counts, conditional sample counts and published headline summaries;
8. routeS runtime distribution and waypoint counts as separate computational metrics;
9. min-cost vs min-fuel divergence, keeping the 2024 bunker/ECA scenario distinct from current statutory EU ETS/FuelEU rules.

The audit should produce machine-readable JSON with dataset identity, source release, file digests, row counts, status distribution, summary reproduction deltas and limitations. A mismatch is surfaced rather than repaired silently.

## 6. How to beat this benchmark fairly

This dataset gives deterministic retrospective common cases. Compare algorithm families with identical input fields and scoring:
- weather-blind calm-optimal baseline;
- deterministic weather-aware corridor routing;
- lateral mesh routing;
- joint route+speed optimization with the same 5% ETA slack;
- multi-objective Pareto routing for fuel, cost, arrival and risk;
- QR-DQN / distributional RL;
- QR-DQN + Wasserstein DRO;
- CVaR/DRO risk-aware route/speed policy;
- MPC / robust MPC;
- oracle-only perfect-foresight policy, when applicable.

Report fuel and cost with ETA/arrival, p90/p95 delay, feasibility, storm-enabled crossings, route distance, environmental exposure, algorithm runtime and evaluation budget. Never compare a 5% ETA-slack policy with an unconstrained policy and call the fuel difference an algorithm win.

### New research extension: forecast-to-reanalysis gap

After reproducing the reanalysis results, add a separate forecast experiment:
- use archived operational forecasts that were actually available at each decision time;
- compare the same routing methods under forecast vintages vs ERA5 reanalysis;
- measure regret, forecast error, delay and robustness per route/season;
- never feed ERA5 future truth into the deployable policy;
- keep ERA5 as retrospective evaluation/environmental truth, not an operational forecast.

This turns a standard routing benchmark into a principled two-stage evaluation: historical idealized routing ability, then real forecast-deployment regret.

## 7. Claim ceiling

A reproduced CSV summary establishes numerical agreement with the external benchmark's published outputs under its model assumptions. It does not independently validate the hydrodynamic model, SFOC curves, weather fields, ship performance or safety constraints. No WAPS savings, fuel prediction superiority or vessel qualification is implied. Transfer requires permissioned measured cargo-vessel data and physical validation.

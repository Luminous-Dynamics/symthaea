# Battery competitive benchmark frontier — 12 November 2026

Status: source-linked research map and proposed internal evaluation programme. All external scores below are published comparator results, not Luminous measurements.

## Executive thesis

Battery research should be judged on **useful, repeatable lifetime service per unit of material, capital, time, energy and risk**—not only initial cell Wh/kg or a single pooled prediction score.

To compete credibly, Luminous needs three benchmark lanes kept separate:

1. **Design discovery:** predict the cycle life of a new material/design combination before a full lifetime test finishes.
2. **Operational prognostics:** estimate SOC/SOH/RUL for an already-existing cell/pack from causal telemetry while preserving censoring and uncertainty.
3. **Manufacturing discovery:** find repeatable process windows that produce the desired structure and cell outcomes across batches, lines and variation.

These lanes share evidence identity, provenance, split integrity and model qualification, but their targets and scores are not interchangeable.

## External benchmark map

| Lane | Published comparator | Scope and reported result | What it does not prove |
|---|---|---|---|
| New-design lifetime discovery | Zhang et al., *Nature* 650 (2026), Discovery Learning | 123 industrial large-format pouch cells; 7.2% cycle-life test error using first-50-cycle physical features for 51% of prototypes; authors estimate 98% time and 95% energy savings under conservative assumptions | Not a Luminous result; does not mean all cells are predictable at cycle 50; its test split/metric must be reproduced exactly |
| Online RUL | *Batteries* 12(11), 484 (11 Nov 2026) | 209-cell source pool; 21 observed-EOL MATR primary-test cells; battery-macro RMSE 26.98 cycles and 92.92% empirical coverage for nominal 90% intervals; protocol-feature RMSE reduction 13.36%, paired p=0.0868 | Point task is based on 21 observed events; other primary-test cells are right-censored. The 82-cell survival analysis is a separate target and metric |
| SOH transfer | Barua et al., *Batteries* 12(9), 342 (2026) | Four NASA cells / 636 cycles; median RMSE 1.52 SOH points on random row splits versus 4.61 points on leave-one-cell-out for the richer predictor set | Small four-cell comparison, not a universal fleet benchmark. Random split is an anti-baseline, not a transfer result |
| Active experiment design | Berkeley Lab / DOE AutoBot (10 Nov 2026) | Explored >5,000 halide-perovskite synthesis combinations and experimentally tested <1%, selecting informative experiments and validating the resulting synthesis/property relationship | Perovskite solar materials, not a battery demonstration. It is a methodological benchmark for experiment efficiency only |
| Battery manufacturing digital twins | Herrero-Ponce et al., *Processes* 14(13), 2158 (2 Jul 2026) | Review identifies strong interdependence among manufacturing stages and gaps in end-to-end integration, standardization, quantitative validation, interoperability and data governance | A review of the field, not a single industrial performance result or vendor benchmark |

### Sources

- Discovery Learning: https://www.nature.com/articles/s41586-025-09951-7
- Censor-aware online RUL: https://www.mdpi.com/2313-0105/12/11/484
- NASA LOCO SOH comparison: https://www.mdpi.com/2313-0105/12/9/342
- AutoBot: https://www.energy.gov/science/bes/articles/accelerating-experiment-design-through-super-intelligence-driven-robotics
- Manufacturing digital-twin review: https://doi.org/10.3390/pr14132158
- IEA Global EV Outlook 2026: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries

## Industry reality: why the product boundary matters

The IEA reports 1.2 TWh of EV battery deployment in 2025, nearly 30% growth year-on-year. Global lithium-ion nameplate manufacturing capacity exceeded 4 TWh by the end of 2025; China accounted for more than 80%, and the IEA notes that reaching near-nominal output can take more than five years for many factories. LFP exceeded 55% of EV battery deployments in 2025 and LFP packs were over 40% cheaper per kWh than NMC on average, although the comparison is partly affected by stationary-storage mix and differing energy-density requirements.

Sources: IEA Global EV Outlook 2026, https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries

The implication is strategic: nominal chemistry performance does not overcome yield, ramp time, process capability, supply-chain concentration, qualification or pack integration. A battery with a higher initial Wh/kg can lose on accepted-cell cost, end-of-life energy, thermal behavior, serviceability or manufacturing throughput.

## Luminous benchmark tracks and targets

### Track A — unseen-design cycle-life discovery

Owner: Symthaea [BATTERY-DESIGN-LIFETIME-TRANSFER-001 #11686](https://github.com/Luminous-Dynamics/symthaea/issues/11686) and draft PR [#11688](https://github.com/Luminous-Dynamics/symthaea/pull/11688).

1. Reproduce the Discovery Learning dataset/feature/cutoff/split/metric exactly before claiming comparability with 7.2%.
2. Then evaluate leave-one-design-combination-out and protocol holdouts with no target-design labels available during model selection/calibration.
3. Internal parity target: ≤7.2% on the exact paper metric only if the original data/split and rights are matched.
4. Stretch target: ≤5% on the exact same task, while maintaining interval calibration, worst-design performance and prospective top-K enrichment. This is a research target, not achieved performance.
5. Measure actual test-time and test-energy savings under the same physical protocol before claiming the paper's 98%/95% estimated savings.
6. Abstain on unsupported chemistry, form factor, protocol or process regimes.

### Track B — existing-cell online health and RUL

Owner: Symthaea [EV-SOH-TRANSFER-BENCH-001 #11252](https://github.com/Luminous-Dynamics/symthaea/issues/11252), with eVTOL study [#11584](https://github.com/Luminous-Dynamics/symthaea/issues/11584).

1. Reproduce the 21-cell MATR strict-80% point task at the published 26.98-cycle battery-macro RMSE and 92.92% nominal-90% empirical interval-coverage comparator, only on the exact paper protocol.
2. Retain a separate 82-cell censor-aware survival evaluation (events and censored observations kept distinct); never pool its outcomes into point RMSE.
3. Reproduce the four-cell NASA LOCO SOH comparison with 4.61-point LOCO median RMSE as the transfer baseline; disclose the tiny independent-cell count.
4. Report per-cell/per-pack error, macro metrics, interval width and coverage, selective risk versus coverage, OOD abstention, and false-safe/false-replacement decision cost.
5. Compare with physical/statistical observers and strong tree ensembles under identical cutoff and label availability before adding larger sequence models.

### Track C — manufacturing process-window discovery

Owner: Symthaea MAT-013 #4344, MAT-PSPP #6224, MAT-UQ #6225 and MAT-INTERFACE #6227; EV consumer adapter in Symtropy [#4791](https://github.com/Luminous-Dynamics/symtropy/issues/4791) / [PR #4792](https://github.com/Luminous-Dynamics/symtropy/pull/4792).

Compare conventional DOE, response-surface models, random selection, single- and multi-objective Bayesian optimization, uncertainty sampling, diversity sampling and mission-conditioned value-of-information. Use equal, frozen budgets and repeat batches.

Report:
- first-pass yield and accepted-good-cell yield by batch/line/recipe revision;
- between-batch and within-roll spatial variability, with confidence intervals;
- electrode loading, thickness, porosity, density, adhesion, moisture and interface state where characterized;
- formation loss, cell energy/power/impedance and lifetime under matched test protocols;
- scrap, material utilization, solvent/water use, process energy, takt time and cost per accepted lifetime kWh;
- process-window width/robustness under measured process variation;
- data/lineage completeness, calibration and invalid-test rates.

Do not claim process causality from observational correlations alone. Controlled interventions or well-supported quasi-experimental designs are required for stronger causal claims.

## The paradigm shifts that should define our product

### 1. From chemistry ranking to mission-conditioned inverse design

Optimize material + electrode + process window + cell + pack + operating policy against a declared mission and lifecycle boundary. Preserve a Pareto frontier; never force a universal winner.

### 2. From digital twin to evidence-bearing digital thread

Each prediction, process record, test, pack and mission outcome resolves to exact versioned configuration and source evidence. Changing a material lot, recipe, firmware or cell configuration triggers targeted requalification rather than silently reusing stale results.

### 3. From 'more data' to decision-value experiments

Choose the next safe, admissible experiment by expected decision change, information gain, experiment cost, time, material, lab bottleneck and uncertainty. Compare with DOE and random baselines. The system proposes; qualified humans/labs authorize and execute.

### 4. From point predictions to robust operating envelopes

Forecast cell/pack capability over time, temperature, ageing, SOC, pulse duration and duty cycle. Report lower-bound capacity/power and the limiting cell/constraint; abstain where support or calibration is missing.

### 5. From model confidence to calibrated, domain-specific uncertainty

Track model, measurement, process and model-form uncertainty separately. A prediction interval calibrated on one chemistry, process or instrument does not automatically transfer to another.

### 6. From successful examples to learning from failures

Retain failed synthesis, non-converged calculations, censored lifetime, process drift, low yield, thermal failures, and negative interventions as separate evidence outcomes. Each should update the corresponding edge of the design → process → structure → property → performance graph.

### 7. From cell Wh/kg to lifecycle service economics

Compare accepted pack cost, useful mission energy, retained energy at end-of-life, charge/thermal constraints, repairability and recoverable materials. Cell-level and pack-level values must never be conflated.

## Competitive positioning

Luminous should not attempt to outbuild CATL or BYD on gigafactory capex and commodity-cell manufacturing. The near-term competitive wedge is a **battery engineering intelligence layer** for developers, labs, smaller manufacturers, pack integrators and new chemistry/process programmes:

- reduce lifetime-test burden through honest unseen-design learning;
- reduce experiment waste through controlled active learning;
- reduce process qualification and root-cause time through exact genealogy and cross-fidelity evidence;
- reduce field risk through calibrated mission-capability/health bounds and explicit abstention;
- make supplier, cell, pack and installed-system evidence interoperable without claiming to certify products.

Commercial proof should begin with one partner dataset and one process/cell problem, not a broad platform claim. The first paid pilot should define a baseline, data rights, exact configuration, independent measurement and measurable savings before/after.

## Rules for SOTA claims

- A paper score is a comparator, not a Luminous achievement.
- Reproduction requires exact dataset release, rights, target definition, feature cutoff, split, model selection and metric.
- A better pooled score does not beat SOTA if worst-design performance, uncertainty or physical holdout gets worse.
- No random-cycle split may stand in for unseen-cell, unseen-design or unseen-line generalization.
- No lifetime point error may include right-censored observations as if they were failures.
- No physical or safety claim may be made from surrogate/simulation evidence.
- Any improvement claim must disclose sample count, independent physical groups, confidence interval, compute/energy, data budget and negative controls.
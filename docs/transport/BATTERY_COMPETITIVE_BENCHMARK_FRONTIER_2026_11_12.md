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

### Track D — joint formation and lifetime charge-policy co-design

Owner: Symthaea [BATTERY-FORMATION-CHARGE-CODESIGN-001 #11735](https://github.com/Luminous-Dynamics/symthaea/issues/11735) / draft [PR #11735](https://github.com/Luminous-Dynamics/symthaea/pull/11735).

This is a distinct interaction benchmark, not another general charge controller or degradation model.

- **Formation baseline:** Attia et al., *Joule* 8(11) (20 November 2024), evaluated 186 cells across 62 formation protocols. The paper reports an average 50% cycle-life increase for its high-first-cycle-current group and attributes the result to altered lithium inventory/electrode utilization under that cell and ageing protocol: https://doi.org/10.1016/j.joule.2024.07.024
- **In-service charge baseline:** a 17 April 2026 *Advanced Science* study reports 20–80% SOC in 15 minutes versus 18 minutes for the manufacturer specification and 25 minutes for a commercial EV charge profile, with less than 5% capacity loss over 500 cycles on one specific commercial cylindrical-cell system. It is not a universal chemistry/format result: https://doi.org/10.1002/advs.75306
- **Manufacturing context:** the October 2026 formation review emphasizes protocol dependence on chemistry, cell format, voltage/current distribution, temperature, pressure and endpoints; no universally optimal protocol exists: https://doi.org/10.1016/j.nxener.2026.100785

The benchmark's novelty is the factorial interaction:
`joint - formation-only - charge-only + reference`.
Compare reference/reference, candidate/reference, reference/candidate and candidate/candidate arms on matched exact cell designs, randomized within independent lot/line blocks. Score charge time, lifetime delivered energy, fade, impedance, thermal/plating diagnostics and formation duration/energy as a Pareto vector.

Reproduce each external study on its own exact cell, split, ageing conditions and metric before claiming parity. Never merge the datasets or claim their published outcomes are additive. A Luminous joint-policy advantage requires a preregistered, lot-held-out physical confirmation set, censor-aware lifetime accounting and evidence of a non-zero interaction with uncertainty. Numeric improvement thresholds are set only after baseline repeatability and power analysis.

The initial Rust contract PR #11735 validates full factorial coverage, immutable protocol identities, declared group-holdout isolation, feature cutoffs, event/censoring semantics and physical evidence/metrology gates. Its synthetic software PASS is not a physical interaction or battery-performance result.

### Track E — decision-value experiment selection

Owner: Symthaea [BATTERY-DESIGN-DECISION-VALUE-001 #11722](https://github.com/Luminous-Dynamics/symthaea/issues/11722) / draft [PR #11722](https://github.com/Luminous-Dynamics/symthaea/pull/11722). This consumes the generic decision-impact benchmark [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803), not another generic VOI engine.

The current inverse-design v0.1 selector is a transparent information-gain-per-normalized-resource heuristic, explicitly not calibrated EVSI. The next benchmark asks whether a selected observation changes the admissible engineering decision:

- A high-entropy experiment can have zero decision value when every possible result selects the same design.
- A low-variance measurement near a safety, manufacturability or mission-feasibility boundary can be decision-critical.
- Expected value of sample information (EVSI) is used only when outcome probabilities come from a declared, admitted, calibrated probability model.
- When only intervals or sets are justified, report distribution-free worst-case regret/robust-feasibility reduction; never turn interval endpoints into pseudo-probabilities.
- Hard evidence, safety, rights and mission gates remain non-compensable. STOP / NO_EXPERIMENT is a valid optimal action.

On identical finite design portfolios and total experiment budgets, compare random, DOE/space-filling, uncertainty sampling, the current information-gain heuristic, EIG when admissible, multi-objective BO/hypervolume improvement where relevant, decision-focused EVSI, distribution-free minimax regret, expert/DOE and STOP. Report normalized decision regret, false-feasible/missed-feasible rates, experiments/material/time/compute to first evidence-supported feasible design, policy sensitivity, failed/abstained-call accounting and top-K prospective enrichment separately.

The initial Rust audit in PR #11722 passed its dedicated known-answer workflow on its original head, but the workflow-syntax policy required adding `ready_for_review` to its PR trigger. Treat the audit and workflow governance as separate qualifications until the new-head rerun completes. The fixture outcomes are synthetic; no selector is yet shown to improve real battery experiments or physical discovery.

### Track F — thermal-propagation evidence transfer and product-change requalification

Owner: Symtropy [EV-BATTERY-THERMAL-PROPAGATION-EVIDENCE-001 #4879](https://github.com/Luminous-Dynamics/symtropy/issues/4879) / draft [PR #4881](https://github.com/Luminous-Dynamics/symtropy/pull/4881).

This is an evidence-assurance lane, not a fire/thermal-runaway simulator or a substitute for standards testing. It closes a gap between individual cell/pack/vehicle test records and decisions about whether those records still apply after design or manufacturing changes.

Research basis:
- UN Regulation No. 100 as consolidated by EU Regulation 2026/1695 includes EV REESS thermal-propagation provisions in 6.15 and Annex 9K, with an application-specific physical-test or systematic risk-management path. The applicability and approval route remain with the competent authority: https://eur-lex.europa.eu/eli/reg/2026/1695/oj/eng/pdf
- IEC 62660-3:2022 is cell/cell-block safety testing, not a standalone pack/vehicle propagation qualification: https://webstore.iec.ch/en/publication/65084
- SAE J2464_202108 describes abuse-test methods across cell/module/pack levels but does not supply pass/fail criteria: https://saemobilus.sae.org/standards/j2464_202108-electric-hybrid-electric-vehicle-rechargeable-energy-storage-system-ress-safety-abuse-testing
- UL 9540A Ed. 6 is a stationary BESS propagation method, including installation-level testing; it is not interchangeable with EV traction REESS evidence: https://www.shopulstandards.com/ProductDetail.aspx?productId=UL9540A_6_S_20260313
- The 2026 cross-standard review recommends a multilevel evidence chain and equivalence assessment across source terms, topology, thermal pathways, protection, enclosure and exposure: https://www.mdpi.com/2313-0105/12/9/375

The benchmark should compare an unstructured document binder, exact-version matching, one-axis configuration matching and a six-domain typed equivalence audit. Report false-green inheritance, detection of changed/unknown domains, result-state preservation, missing metrology/criteria, source-digest completeness and deterministic replay. The software's strongest positive disposition is only **candidate for competent human/authority review**. It cannot determine safety, predict propagation, certify a standard or authorize destructive testing.

Draft PR #4881 adds the Rust/schema/fixture audit. Exact current PR head `84c70844f7eb3910f110f2ed7e0dfd44163203ff` passed its push-triggered focused workflow: Draft 2020-12 schema/fixture validation, locked metadata, formatting, 21/21 tests and strict Clippy (https://github.com/Luminous-Dynamics/symtropy/actions/runs/43258445101). The PR-triggered job remains skipped while the PR is draft. Fixture evidence is synthetic.

## Industry scorecard and benchmark targets

Use public industry numbers to define the comparison boundary, not to claim parity from unlike tests.

### 2025–2026 industrial reality

- Global EV battery deployment reached about **1.2 TWh in 2025**, almost 30% above 2024. China represented 60% of EV battery deployment, and China-headquartered producers supplied almost 75% of global EV battery deployments. This is the scale and supply-chain context, not a performance benchmark. IEA: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- Average battery prices declined 8% in 2025. LFP packs were over 40% cheaper per kWh than NMC on average, but application mix and energy-density requirements matter. China pack prices were 30% below North America and 35% below Europe. IEA's figures are regional/product-mix averages, not a cell-specific procurement quote.
- More than 4 TWh of global lithium-ion nameplate manufacturing capacity existed at end-2025; China held over 80%. The IEA says average production yields above **90%** are needed to compete in today's battery market and notes that new plants may take more than five years to approach nominal output. Treat >90% as an industry-referenced competitive threshold, not as a measured Luminous or partner yield.
- Solid-state programmes are progressing, but the IEA notes that longer range and enhanced safety advantages have not yet been demonstrated at scale in real-world applications. Do not present laboratory cell demonstrations or announced future production dates as product qualification.

IEA reference: https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries

### Aspirational OEM / government programme comparator

USABC's October 6, 2025 High-Energy EV performance goals specify **cell-level** end-of-life targets under declared conditions: 450 Wh/kg usable specific energy, 850 Wh/L usable energy density, 10-minute fast charge to 80% usable energy at 30°C, 750 dynamic-stress-test cycles with 25% fast charging, 15-year calendar life, 800 W/kg 30-second peak discharge, 300 W/kg regenerative power, $60/kWh at 250,000-unit US production, and more than 70% specific usable energy at -20°C without assistance.

These are **aspirational USABC goals**, not today's industry average and not a pack-level guarantee. The benchmark must match cell/pack mass boundary, usable-energy definition, test protocol, temperature, EOL criterion and volume/cost assumptions before reporting a gap.

Sources:
- USABC goals page: https://uscar.org/usabc/goals/
- High-Energy EV targets (version 6 October 2025): https://uscar.org/wp-content/uploads/2025/12/USABC-High-Energy-EV-RFPI.pdf

### Published machine-learning comparator

Discovery Learning reports **7.2% cycle-life test error** on 123 industrial large-format Li-ion pouch cells using features from the first 50 cycles for 51% of prototypes, trained using public datasets of different cell designs. It estimates 98% time and 95% energy savings under its conservative assumptions. This is a strong design-lifetime comparator, but the result is paper-specific: Luminous must reproduce its exact target, feature availability, physical-cell/design split, data rights and metric before claiming parity or improvement. Do not turn its estimate into a measured Luminous saving.

Source: https://www.nature.com/articles/s41586-025-09951-7

### Our competitive tournament

For every candidate method, run the same frozen data/design portfolio and publish separate outcomes:

| Layer | Matched benchmark | Required metrics | Non-comparable shortcuts to reject |
|---|---|---|---|
| Cell design | First-1/5/10/20/50-cycle life transfer | MAE/RMSE, relative error, calibration, interval coverage, worst-design error, time-to-EOL and censor-aware score | Randomly shuffled adjacent cycles; pooled-cell leakage |
| Charging | Same cell, SOC interval, ambient, age and energy boundary | 10–80% / 20–80% time, delivered kWh, thermal derating, energy efficiency, fade and impedance growth | Peak charger kW or different SOC windows presented as equal |
| Formation | Same exact cell design and process window | Formation hours/kWh, first-cycle efficiency, accepted yield, cell life and downstream charge interaction | Coin-cell-to-pouch extrapolation or recipe-only comparison |
| Pack/mission | Same topology, duty trace, ambient and reserve | Useful service, mission completion, lower-bound power/energy, limiting-cell attribution, thermal margins | Cell Wh/kg substituted for pack/vehicle performance |
| Manufacturing | Same line, lot, measurement system and acceptance limits | First-pass yield, scrap/rework, Cp/Cpk, cost per accepted kWh, between-lot variance | One pilot lot used to claim factory capability |
| Safety / evidence | Same application, test level, article, trigger and standard revision | Evidence completeness, transfer blockers, retest triggers, false-green inheritance | Cross-level/cross-standard pass promotion |

### Numeric targets to pursue

**Software/epistemic gates (must be exact):**
- zero source/target/lot/line/protocol leakage in preregistered held-out evaluations;
- zero censored-to-observed EOL promotion;
- zero probability fabricated from interval endpoints;
- zero unsupported product/safety evidence inheritance;
- 100% traceability for all benchmark inputs, model versions, splits, failed/censored/invalid results and resource budgets;
- deterministic report parity under 100 input permutations.

**Research performance targets (not claims):**
- Reproduce Discovery Learning's 7.2% task-specific error before attempting to beat it. Then seek a lower error at equal or smaller data/experiment budgets while preserving worst-design error, calibrated uncertainty and physical holdout.
- Reproduce USABC target dimensions only when we have comparable cell prototypes and valid methods. Treat 450 Wh/kg and 10-minute charge as long-horizon targets, not near-term product claims.
- For formation/charge co-design, demonstrate a preregistered non-additive interaction and a Pareto improvement on independent lots; set the effect threshold after metrology repeatability and power analysis.
- For manufacturing, reach and sustain >90% first-pass yield on a qualified production line before cost-parity claims; report accepted-good-cell cost, not theoretical bill-of-materials cost.
- For sodium-ion, optimize cold-climate useful service and supply resilience; do not attempt to win a lithium-ion energy-density contest on a different system boundary.
- For thermal propagation, software target is zero false-green evidence transfer; physical acceptance remains standard- and authority-specific.

### Track G — low-pressure ASSB practical-cell translation and manufacturability

Owner: Symthaea [ASSB-PRACTICAL-TRANSLATION-001 #11759](https://github.com/Luminous-Dynamics/symthaea/issues/11759), composing SSE-001 #4993, ION-001 #4996, MAT-INTERFACE-001 #6227, MAT-PSPP #6224, MAT-013 #4344 and MAT-UQ #6225.

**Why this is distinct:** material conductivity and symmetric-cell cycling do not establish complete-cell operation, practical pressure, 5 Ah cell performance, manufacturable electrolyte thickness, usable low-temperature energy or repeated-line yield. The missing benchmark is the translation boundary.

Research anchors:
- Lee et al., *Nature Communications*, published 7 November 2026: near-ambient sulfide ASSB operation at room temperature under ≤0.3 MPa and 2.4 mAh/cm² with NMC811 at 0.1C, using coupled composite-electrolyte mechanical reinforcement and porous-carbon current-collector stress relaxation. This is a paper-specific demonstrator, not an EV fast-charge, ≥5 Ah, >1,000-cycle or factory-yield claim. https://www.nature.com/articles/s41467-026-79393-w
- USABC October 21, 2025 solid-electrolyte goals provide an automotive target: material conductivity >10 mS/cm at 30°C and >1 mS/cm at 0°C; electronic conductivity <1×10⁻⁸ mS/cm; critical current density >13 mA/cm²; thermal stability >250°C; <10% conductivity loss after a 72 h water-stability test at −40°C dew point; <$10/kg at a 75 kt/year scale assumption. Cell goals include <2 MPa pressure, >70% of 30°C usable energy at −20°C on a C/3 discharge without assistance, <20 μm electrolyte sheet, >15-year calendar life and >1,000 DST cycles. The RFPI footnote specifies cells of 5 Ah or more. These are USABC development goals, not current average product performance. https://uscar.org/wp-content/uploads/2025/12/USABC-Electrolyte-RFPI.pdf
- IEA Global EV Outlook 2026 reports 1.2 TWh EV battery deployment in 2025, >4 TWh global Li-ion nameplate capacity by year-end, >80% of capacity in China, and says >90% average manufacturing yield is required to compete; new factories may take more than five years to approach nominal output. This is industrial context, not an ASSB-specific yield series. https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- For AI-driven life evaluation, Discovery Learning reports 7.2% cycle-life test error on 123 industrial large-format pouch cells, using first-50-cycle features for 51% of prototypes, and estimates time/energy savings under its own assumptions. Reproduce the exact data split, feature cutoff and target before claiming to match or beat it. https://www.nature.com/articles/s41586-025-09951-7

Paradigm shift:
- Treat ASSB as a **coupled electro-chemo-mechanical-manufacturing system**, not an electrolyte leaderboard.
- Keep fabrication pressure and operating stack pressure separate.
- Compare pressure-only effects only while holding design revision, batch, manufacturing line, process revision and pressure measurement method fixed; stratify a process change as a separate DOE factor.
- Require complete-cell mass and external-volume boundaries including inactive cathode/anode fractions, current collectors, tabs, case and seals.
- Keep sulfide, halide, oxide, polymer/composite, silicon-anode and hybrid solid–liquid lanes separate; hybrid cells cannot be promoted as all-solid-state.
- Split by cell design, cell format, batch, production line and laboratory—not random cycles from the same cell.

Near-term benchmark ladder:
1. Reproduce the 7 November 2026 ≤0.3 MPa / 2.4 mAh/cm² paper's exact test boundary where data rights and methods permit.
2. Match the USABC <2 MPa, <20 μm and ≥5 Ah cell-level goal boundaries; separately report material-level conductivity and stability tests.
3. Progress from symmetric/half-cell mechanistic work to complete full cells; no cell-level Wh/kg or Wh/L claim without complete-cell measured mass/volume and usable discharge energy.
4. Demonstrate independent-cell and independent-batch repeatability, with observed EOL, right-censored and invalid outcomes separate.
5. Qualify manufacturing over multiple lots and lines; target >90% first-pass yield before making scale/cost claims.
6. Benchmark prospective design/lifetime models against random, DOE/factorial, descriptor regression and Discovery Learning-style transfer baselines at equal data, experiment, time and energy budgets.

Draft [PR #11770](https://github.com/Luminous-Dynamics/symthaea/pull/11770) implements the first read-only audit slice on branch `research/assb-practical-translation-v0.1`. Exact head `1075e8fbdd8311c9ebda3146f63485f43bcd1671` passed Draft 2020-12 schema/fixture validation, locked metadata, Rust 1.96 formatting, 23/23 tests and strict Clippy (https://github.com/Luminous-Dynamics/symthaea/actions/runs/43274827165). It rejects fabrication-pressure, batch, line, process-revision, design-revision, method/instrument and uncertainty confounding; audits inactive electrode mass plus complete-cell volume. All evidence remains synthetic; this is not an electrochemical performance claim.

## ASSB benchmark frontier: targets beyond low-pressure headlines

The current low-pressure benchmark lane is **ASSB-PRACTICAL-TRANSLATION-001 #11759** / draft [PR #11770](https://github.com/Luminous-Dynamics/symthaea/pull/11770). Its exact-head focused workflow passed schema/fixture validation, locked metadata, formatting, **23/23 Rust tests** and strict Clippy:
https://github.com/Luminous-Dynamics/symthaea/actions/runs/43274827165

### Three benchmark levels must remain separate

**Current low-pressure research demonstrations**
- Lee et al. (7 November 2026) report room-temperature sulfide ASSB operation at ≤0.3 MPa stack pressure, with NMC811 at 2.4 mAh/cm² and 0.1C. This is an important near-ambient pressure demonstration, but the reported loading and rate are below the ≥5 mAh/cm² and ≥2C challenge proposed for competitive practical SSBs.
- A separate 2026 *Nature Communications* result reports pouch-cell operation at 2.5 MPa, 350 cycles at 1C and 85% retention; its >400 Wh/kg figure is based on solid electrolyte, Li metal and positive-electrode materials, not a fully inclusive inactive-component complete-cell mass boundary. Do not compare that figure directly to complete-cell or pack Wh/kg.

Sources:
- https://www.nature.com/articles/s41467-026-79393-w
- https://www.nature.com/articles/s41467-026-73012-4
- https://doi.org/10.1002/batt.70514

**USABC solid-electrolyte development goals (version 21 October 2025)**
These are explicit automotive-development goals, not a current industry average or a demonstrated commercial baseline.

| Level | Parameter | USABC target |
|---|---|---:|
| Material | Ionic conductivity | >10 mS/cm at 30°C; >1 mS/cm at 0°C |
| Material | Electronic conductivity | <1×10⁻⁸ mS/cm (as stated in the USABC table) |
| Material | Critical current density | >13 mA/cm² |
| Material | Electrochemical window | 0–5 V vs Li/Li⁺ |
| Material | Thermal stability | >250°C |
| Material | Water stability | <10% ionic-conductivity loss after 72 h at −40°C dew point |
| Material | Cost | <$10/kg at 75 kt/year |
| Cell | Capacity | ≥5 Ah |
| Cell | Pressure | <2 MPa |
| Cell | Electrolyte thickness | <20 μm |
| Cell | Calendar life | >15 years |
| Cell | Cycle life | >1,000 cycles |
| Cell | Low-temperature operation | >70% of 30°C usable energy at −20°C, unassisted |

The USABC high-energy EV cell goals add a distinct EOL target set: 450 Wh/kg usable specific energy, 850 Wh/L usable energy density, 10-minute charge to 80% usable energy at 30°C, 750 dynamic-stress-test cycles with 25% fast charging, 15-year calendar life, and $60/kWh at 250,000-unit production assumptions. These goals must not be merged into the solid-electrolyte material targets or treated as pack-level numbers.

Sources:
- USABC Solid Electrolyte RFPI, Appendix A (21 October 2025): https://uscar.org/wp-content/uploads/2025/12/USABC-Electrolyte-RFPI.pdf
- USABC High-Energy EV performance targets (6 October 2025): https://uscar.org/wp-content/uploads/2025/12/USABC-High-Energy-EV-RFPI.pdf
- USABC goals catalogue: https://uscar.org/usabc/goals/

**Research competitiveness challenge**
A March 2026 analysis argues that competitive ASSBs should reach >5 mAh/cm² areal loading and >2C, and that many academic demonstrations remain below 2 mAh/cm² and below 1C. These are proposed practical competitiveness boundaries, not universally standardized pass criteria:
https://doi.org/10.1002/anie.2890151

The benchmark should therefore report a multi-axis vector—pressure, full-cell mass/volume, areal capacity, rate, life, electrolyte thickness, inactive fraction, batch yield and cost—rather than a single record number. The first near-ambient demonstrator should be a reproduction baseline; the USABC and ≥5 mAh/cm² / ≥2C goals are stretch targets.

### Interlaboratory reproducibility is a first-class research lane

A separate programme, **ASSB-ROUNDROBIN-001 #11784**, is now open with a first read-only data/audit slice in branch `research/assb-roundrobin-v0.1`. It composes the ASSB translation audit rather than duplicating it.

The motivation:
- A 2024 *Nature Energy* interlaboratory study supplied common materials to 21 groups and found large variation associated with pressing pressure, pressing duration and In-to-Li ratio. It recommended standard parameter reporting and triplicate data:
  https://www.nature.com/articles/s41560-024-01634-3
- A 2026 fixture study suggests <5% pressure homogeneity, <5% setpoint accuracy MAPE and <5% pressure repeatability CV as candidate engineering metrics. The authors note that quantitative fixture benchmarks were not established and that their numbers are engineering judgement:
  https://doi.org/10.1002/ese3.70581

The round-robin lane distinguishes COMMON_PROTOCOL, LAB_NATIVE and PROTOCOL_DEVIATION arms; counts unique physical cells rather than repeated readings; and reports a synthetic known-answer fixture separately from physical cross-lab candidates. A group with ≥3 labs and ≥3 distinct cells per lab can only qualify as a **candidate for cross-lab variance analysis**, not as proof of reproducibility. It does not yet calculate pooled statistics.

## Additional materials frontier: sodium-ion and zero-excess sodium-metal

A new bounded materials lane has been opened in Symthaea [SODIUM-BATTERY-MISSION-001 #11694](https://github.com/Luminous-Dynamics/symthaea/issues/11694) / draft [PR #11695](https://github.com/Luminous-Dynamics/symthaea/pull/11695).

Keep two technology frontiers separate:

- **Commercial-near sodium-ion** (hard carbon + Prussian White/PBA, layered oxides or polyanionic cathodes) should be compared with LFP/NMC under matched cold-climate, stationary and industrial-fleet missions.
- **Zero-excess sodium-metal** is an experimental post-lithium lane with finite sodium-inventory loss and plating/SEI failure mechanisms. Do not mix it into a commercial sodium-ion leaderboard.

Useful source anchors: IEA's 2026 market/technology assessment; pilot-scale aqueous HC|Prussian White ≈1 Ah pouch cells at ≈120 Wh/kg practical complete-cell specific energy (Maddar et al., DOI 10.1002/aenm.202505344); PNNL's specific localized-high-concentration electrolyte study (HC||PB cell >1,000 cycles at 2C; pouch 82% after 500 cycles); and the 11 November 2026 zero-excess sodium-metal review (demonstrations >200 Wh/kg but demanding ≥99.98% average Coulombic efficiency for a 1,000-cycle/80%-retention challenge under practical conditions). None is a Luminous result or a matched commercial-market comparison.

The sodium benchmark therefore adds **cold-temperature usable capacity including preheat energy**, matched full-cell mass/volume, areal loading, electrolyte-to-capacity ratio, temperature and C-rate, first-pass yield, and failure/censoring semantics. Active-material and cathode-only energy density cannot be promoted to complete-cell or pack claims.

Sources:
- https://www.iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- https://doi.org/10.1002/aenm.202505344
- https://www.pnnl.gov/publications/tailoring-electrolyte-solvation-structure-enhance-rate-capability-cycle-life-and
- https://www.nature.com/articles/s44359-026-00234-6

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
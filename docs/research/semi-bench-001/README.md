# SEMI-BENCH-001 — Semiconductor Engineering & Manufacturing Benchmark Ladder

**Tracking issue:** [Luminous-Dynamics/symthaea#13058](https://github.com/Luminous-Dynamics/symthaea/issues/13058)  
**Version:** 0.1.0  
**Status:** research constitution / proposed benchmark manifest  
**Evidence class:** public datasets, reproducible EDA flows, synthetic controls; no physical-fab result claimed

## Executive thesis

Build an open, leakage-resistant benchmark ladder that measures whether an engineering system can improve semiconductor design and manufacturing decisions—not merely generate plausible engineering language or achieve a high score on a random split.

The ladder spans distinct evidence planes:

1. semiconductor device physics and analytical references;
2. circuit/RTL and physical-design quality;
3. process-data diagnostics and wafer-map inspection;
4. metrology, calibration and equipment evidence;
5. digital-twin forecast fidelity and decision value;
6. independently measured physical process impact;
7. provenance, lifecycle and productive-capability dependencies.

These planes must remain separate. A model can pass one and fail another.

```text
device model accuracy             != fabricated-device validity
RTL-to-GDS PPA                    != fabricated-silicon performance
wafer-map classification          != fab-yield improvement
process prediction                != causal root-cause identification
digital-twin replay               != measured process impact
metrology-bench calibration       != semiconductor process qualification
successful external tapeout       != local fabrication capability
```

There is no single “semiconductor SOTA” number and no scalar `fab_readiness`.

## Programme ownership

This is a benchmark/comparison layer, not a new physical-science or manufacturing ontology.

| Owner | Responsibilities | Existing work to compose |
|---|---|---|
| Symthaea | Device/circuit reasoning, model selection, experimental design, benchmark protocol and independent validators | [ENG-SEMI #5671](https://github.com/Luminous-Dynamics/symthaea/issues/5671), [REF-001 #5868](https://github.com/Luminous-Dynamics/symthaea/issues/5868), [SEMI-FAB #5887](https://github.com/Luminous-Dynamics/symthaea/issues/5887) |
| Symtropy | Physical twin, deterministic replay, constrained optimization, uncertainty, scenario/fault injection and causal-effect evaluation | [GLASS-SILICON #2715](https://github.com/Luminous-Dynamics/symtropy/issues/2715), [GLASS-SILICON-CAUSAL #2747](https://github.com/Luminous-Dynamics/symtropy/issues/2747) |
| Mycelix | Resource/process registry, lot/article/work-order provenance, evidence custody, lifecycle and supplier/dependency records | [MFG-PROC #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085), [CIRC-MFG #3095](https://github.com/Luminous-Dynamics/mycelix/issues/3095) |
| Sol-Atlas | Source, standards, patent, material and technical-knowledge references where a canonical source owner exists | Existing Sol-Atlas/source catalogue work; do not create a duplicate catalogue |

Important boundary: Symtropy’s current Glass/Silicon v0.1 scope covers glass and metallurgical-grade silicon (MG-Si), while explicitly excluding electronic-grade purification, wafer production and semiconductor manufacture. Its results must not be represented as semiconductor-manufacturing results.

Existing semiconductor owners to reuse:

- [SEMI-PROC #5888](https://github.com/Luminous-Dynamics/symthaea/issues/5888): wafer-state and process-flow composition.
- [SEMI-EQP #5889](https://github.com/Luminous-Dynamics/symthaea/issues/5889): tool architecture and capability evidence.
- [SEMI-SUBFAB #5890](https://github.com/Luminous-Dynamics/symthaea/issues/5890): vacuum, gas, thermal, power, exhaust and facility dependency composition.
- [SEMI-MET #5891](https://github.com/Luminous-Dynamics/symthaea/issues/5891): metrology, calibration and inspection.
- [SEMI-EQP-MET #5912](https://github.com/Luminous-Dynamics/symthaea/issues/5912) and [benchmark constitution #5914](https://github.com/Luminous-Dynamics/symthaea/issues/5914): bounded reference inspection/positioning/calibration bench.
- [MFG-PROC replay #5901](https://github.com/Luminous-Dynamics/symthaea/issues/5901): generic manufacturing contracts. Do not create a second process-plan engine.

## Industry frontier and where to compete

### 1. Transistor/device scaling

The frontier includes gate-all-around nanosheets, backside power delivery, and research toward complementary FET (CFET) integration. Imec describes CFET-specific backside-contact and gate-stack modules as well as design-technology co-optimization for future standard-cell configurations.

**Our opportunity:** reproduce bounded device references; connect analytical models, TCAD, compact models, circuit projections and measured device observations; quantify discrepancy instead of tuning every model to agree.

**Not our first target:** attempting to invent or fabricate a leading-edge transistor process. That requires foundry access, process modules, metrology and a scale of capital that a public software research effort does not possess.

### 2. Lithography and patterning

High-NA EUV is a major leading-edge manufacturing frontier. ASML describes how finer-resolution lithography can reduce multi-patterning steps, with potential productivity, energy, chemical and water benefits. Those benefits depend on actual product/process integration; they are not automatic savings for every layer or fab.

**Our opportunity:** pattern/process sensitivity analysis, overlay and defect-metrology data fusion, uncertainty-aware design/process optimization, and reproducible comparison of patterning assumptions. Do not imply an open simulator is equivalent to an ASML scanner or a foundry-qualified lithography process.

### 3. Chiplets and advanced packaging

UCIe 2.0 adds system-level manageability and debug/test support across chiplets and supports 3D packaging, including hybrid-bonding-oriented interconnect scales.

**Our opportunity:** co-optimize chiplet partitioning, interconnect bandwidth/energy, thermal paths, testability, repairability and lifecycle. Make package-level outcomes first-class: known-good-die yield, package yield, thermal resistance, power integrity, test coverage and serviceability.

### 4. Process control and metrology

The industrial process-control stack combines inspection, metrology and data analytics to help manufacturers achieve target yields. SEMI EDA/Interface A and related standards support high-speed equipment data collection and integration.

**Our opportunity:** robust sensor-fusion and virtual metrology that explicitly model calibration, uncertainty, missingness, tool state, recipe/profile identity, data lineage and out-of-distribution conditions. The differentiator should be auditable, calibrated and transportable decision quality—not another unqualified “AI yield predictor.”

### 5. Digital twins and cross-technology co-optimization

Digital twins are valuable only when their state, parameters, observation paths, uncertainty and validation boundaries are explicit. Imec's 2026 strategy highlights cross-technology co-optimization (XTCO), while NIST's digital-twin work emphasizes verification, validation, uncertainty quantification, interoperability and cybersecurity challenges.

**Our opportunity:** combine device, circuit, package, process, equipment and facility constraints without allowing a simulation result to authorize physical action or stand in for a measurement.

## Public benchmark families

### A. Fab-process diagnostics: UCI SECOM

Source: [UCI SECOM dataset](https://archive.ics.uci.edu/dataset/179/secom)

Known dataset facts:
- 1,567 production examples.
- UCI lists 591 attributes; raw sensor values are commonly represented as 590 columns, with label/timestamp data supplied separately.
- 104 failed examples; the minority class is approximately 6.6%.
- Missing sensor values are present.
- Sensor names and detailed process context are anonymized.

Use it for: a small, real-production-data smoke test of process anomaly/risk scoring, feature screening, missingness handling, calibrated abstention and false-alarm tradeoffs.

Do not use it to claim current-fab generalization, production-ready yield prediction or causal root-cause identification. The dataset is small, old, anonymized and has limited process context.

Required baselines:
- prevalence-only and majority-class predictors;
- logistic regression and ridge;
- a tree ensemble;
- a simple anomaly detector;
- a calibrated/abstaining variant.

Required metrics:
- balanced error rate / balanced accuracy;
- AUCPR and minority-class recall;
- false alarms at declared recall/sensitivity levels;
- Brier score and calibration curve;
- coverage, abstention and risk-versus-coverage;
- confidence intervals and failure analysis.

Primary split policy: forward-chaining/time-aware where timestamp semantics permit. Random stratified splits may be reproduced as a secondary diagnostic, but cannot be the headline generalization claim. Freeze split IDs before fitting. Report the number of failures in every partition.

### B. Wafer-map defect patterns: WM-811K

Reference: [Accurate and energy efficient ad-hoc neural network for wafer map classification](https://doi.org/10.1007/s10845-024-02390-7)

Known dataset facts reported in the reference:
- 811,457 wafer maps in total.
- 172,950 labelled maps.
- 25,519 single-pattern labels across Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch and Near-full.
- 147,431 maps carry the mixed/None label.

Use it for: defect-pattern classification, rare-class recognition, partial-label evaluation, low-label learning, calibrated rejection and deployment-cost comparisons.

Critical protocol:
- Keep the eight single-pattern labels and mixed/None label semantics explicit.
- Do not treat mixed/None as “good wafer” or silently discard it.
- Report class counts and exact inclusion rules for every experiment.
- Deduplicate exact image content before split assignment.
- Use wafer/lot/group-disjoint splits if the released identifiers support them. If group identifiers are unavailable, disclose that limitation and do not claim lot-level generalization.
- Keep the published selected-subset accuracy numbers separate from full-distribution results; sample composition and label scope can make headline accuracy incomparable.

Required metrics:
- macro-F1 and balanced accuracy;
- per-class precision, recall and support;
- AUCPR for rare classes;
- calibration and risk-versus-coverage;
- confusion matrix, especially mixed/None errors;
- model parameters, inference latency, peak memory and energy only when instrumented.

Compare classical image features, compact CNN, residual CNN and a small transformer under matched training-data and compute budgets. A Symthaea HDC/CfC approach is a candidate to test, not an assumed winner.

### C. EDA physical-design prediction: CircuitNet

Source: [CircuitNet dataset introduction](https://circuitnet.github.io/intro/intro.html)

CircuitNet reports more than 20,000 samples generated from commercial design-tool runs and includes:
- CircuitNet-N28: RISC-V design basis, with congestion, DRC, IR-drop and net-delay tasks;
- CircuitNet-N14: multiple designs including RISC-V, GPU and ML accelerator; the listed supported tasks include congestion, IR drop and net delay;
- CircuitNet-ISPD15: congestion prediction.

Use it for: congestion, IR-drop and net-delay prediction; design-level transfer; model efficiency.

Required protocol:
- split by complete design, not randomly by adjacent samples;
- preserve node/task boundaries;
- where possible hold out designs and technology variants;
- report any overlap in design lineage or tool configurations;
- reproduce the official dataset baseline before proposing a new model.

Required metrics:
- MAE/RMSE and normalized error;
- tail/threshold recall for critical congestion or IR-drop regions;
- calibration/uncertainty and out-of-domain abstention;
- per-design breakdown, not just pooled averages;
- training/inference compute and memory.

### D. RTL-to-GDS and PPA

Primary open flow: [OpenROAD Flow Scripts metrics](https://openroad-flow-scripts.readthedocs.io/en/latest/contrib/Metrics.html)

OpenROAD Flow Scripts runs reproducible RTL-to-GDS examples and records machine-readable QoR metrics. Its QoR check compares a run against a real baseline build, which is a better model for regression control than reporting one uncontextualized PPA number.

Candidate PDKs:
- [SkyWater SKY130](https://github.com/google/skywater-pdk);
- [GlobalFoundries GF180MCU](https://github.com/google/gf180mcu-pdk);
- [IHP SG13G2](https://github.com/IHP-GmbH/IHP-Open-PDK).

Important limitations:
- SkyWater documentation currently describes the open SKY130 release as experimental preview/alpha and not intended for production use.
- IHP describes SG13G2 as early access and not yet intended for production use.
- PDK status, exact commit, model version, corner, library, rules and supported flow must be pinned for every run.
- Do not compare results across different PDKs, clocks, utilization targets, constraints or tool settings as though they were equivalent.

Metrics:
- area;
- timing (WNS/TNS and critical path);
- power only under explicit, comparable activity/model assumptions;
- congestion and wirelength;
- DRC/LVS status;
- runtime, peak memory and reproducibility;
- failed constraints and unmeasured quantities.

Industry comparison opportunity: IEEE CEDA's public commercial-EDA benchmarking effort describes Siemens Aprisa v2025.2 results on open designs and PDKs, including ASAP7, Nangate45, IHP130 and SKY130HD examples. Use this as a carefully scoped commercial reference where its published artifacts and permissions allow. Do not reuse proprietary scripts or claim that one flow is better without matching the design, PDK, constraints and tool versions.

### E. Device physics: analytical → TCAD → SPICE → measurement

Reuse [ENG-SEMI-REF-001 #5868](https://github.com/Luminous-Dynamics/symthaea/issues/5868), the analytical diode oracle work and the bounded DEVSIM adapter. Keep the current benign reference-device scope.

Report:
- exact subject and datasheet revision;
- equation assumptions and applicability domain;
- solver, mesh, geometry, material, contacts, doping assumptions and model options;
- numerical convergence versus mesh independence;
- analytical↔TCAD residuals;
- compact-model fit residuals on a separate fit/holdout split;
- ngspice circuit prediction residuals;
- measured low-voltage I–V residuals with instrument calibration and uncertainty.

A fit to TCAD is not independent validation of TCAD. A good SPICE fit is not proof of the physical device mechanism. Physical characterization is a separate evidence stage.

### F. Inspection, positioning and calibration bench

Reuse—not duplicate—the exact benchmark contract and corpus from [SEMI-EQP-MET-001 #5912](https://github.com/Luminous-Dynamics/symthaea/issues/5912) and [#5914](https://github.com/Luminous-Dynamics/symthaea/issues/5914).

Progression:
1. synthetic known-answer controls;
2. traceable scale/reference artifact;
3. position command versus observed/reference position;
4. revisit repeatability;
5. absolute accuracy under calibration;
6. remount/refocus sensitivity;
7. raw image/sensor evidence lineage;
8. uncertainty, local field-of-view and coverage;
9. independent physical reference-target campaign.

Do not equate sharpness with accuracy, repeatability with absolute accuracy, a local clean image with whole-wafer cleanliness, or a derived image with a new physical observation.

## Paradigm shifts to test

### 1. Evidence-carrying engineering rather than proposal-only AI

Every recommendation should carry its exact subject/configuration, input digest, assumptions, predicted outputs, uncertainty, applicable constraints, reference baseline, and a reproducible test plan. The system should be able to refuse when data or model coverage is inadequate.

### 2. Cross-layer co-optimization rather than isolated component optimization

Optimize chip, package, cooling, power delivery, interconnect, testability and manufacturing variation as a constrained system. Compare against component-wise optimization under the same evaluation budget. This is our local route to exploring DTCO/XTCO without pretending to reproduce imec's fabrication capabilities.

### 3. Metrology-aware process planning

Treat measurability, calibration burden, uncertainty and spatial coverage as design constraints. A process that produces a nominally superior result but cannot be measured reliably should not be considered qualified.

### 4. Causal process intelligence rather than correlation dashboards

Separate:
- forecast fidelity;
- diagnostic/root-cause utility;
- decision quality under matched budget;
- causal effect of a physical intervention.

Only the last category can support measured energy/yield improvement claims, and only after a preregistered controlled comparison with independently measured outcomes.

### 5. Open reproducibility as a competitive advantage

Publish exact benchmark contracts, baseline scripts, data rights, model/tool versions, result digests, negative controls and claim ceilings. Proprietary incumbents may have more and better fab data; we can compete first on transparency, transportability, calibration and reproducibility.

### 6. Productive-closure engineering

Track which capabilities depend on external fabs, tools, PDKs, metrology, calibration artifacts, imported components, spare parts, software licenses, utilities and specialist maintenance. A working design or instrument is useful capability; it is not the same as local independence or a self-reproducing fab.

## How we compare to industry

| Industry capability | Current practical benchmark for this programme | What would justify a stronger claim |
|---|---|---|
| Leading-edge foundries | Public process/device research, not direct process-node competition | Foundry access, matched device structures, independently measured silicon and statistically credible process/yield data |
| ASML / lithography ecosystem | Pattern sensitivity, overlay/defect data, metrology-aware optimization | Qualified scanner/process integration and physical wafer evidence |
| KLA / process control | Open public datasets, inspection/metrology benchmark and calibration/shift robustness | Partner fab data, matched production comparison, demonstrated false-alarm/escape/yield impact |
| Commercial EDA | OpenROAD and published commercial-EDA benchmark artifacts | Same designs, PDKs, constraints, tool versions and independent repeat runs |
| Advanced packaging | UCIe-oriented architecture and package-level simulation/test contracts | Fabricated test vehicle, measured interconnect/thermal/test results and known-good-die/package-yield accounting |
| Fab digital twins | Held-out public data and deterministic twin replay | Time/campaign-held-out partner data and preregistered physical intervention effect |

Industry sources:
- [ASML 2025 annual report — patterning simplification and sustainability](https://www.asml.com/en/investors/annual-report/2025/strategy-and-stories)
- [SEMI standards and smart manufacturing](https://www.semi.org/en/about/SEMI_Standards)
- [SEMI information and control standards](https://www.semi.org/en/products-services/standards/information_and_control)
- [KLA FY2025 SEC filing](https://www.sec.gov/Archives/edgar/data/319201/000031920126000008/klac-20251231.htm)
- [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)
- [CircuitNet](https://circuitnet.github.io/intro/intro.html)
- [IEEE CEDA commercial EDA benchmark](https://github.com/ieee-ceda-datc/First-Commercial-Benchmarking-SiemensEDA)

## Targets and gates

All numbers below are **research targets**, not measured results.

### Integrity gate

- 100% rejection of seeded data leakage, group leakage, label corruption, stale calibration, bad lineage, wrong units, mismatched PDK/constraints, invalid source identity and out-of-domain controls.
- At least one valid positive control must pass each evaluator; reject-all must not pass.
- Freeze corpus digest, split manifest, toolchain, model identity and benchmark version before result scoring.

### Predictive gate

- Reproduce the selected published/open baseline under its actual protocol.
- Target at least 10% relative improvement in the preregistered primary error metric on grouped/time-held-out distribution shift.
- Require confidence intervals to support the gain.
- No material regression in calibration, minority-class recall, critical-tail recall, compute budget or hard constraints.
- A result that improves average error but degrades rare defect detection must be reported as a tradeoff, not an overall win.

### Decision gate

- Compare with a matched-budget human/simple-rule/established-model baseline.
- Count data acquisition, operator review, compute, runtime, false alarms, retests and rejected decisions.
- Require calibrated abstention and explicit no-go regions.
- Publish Pareto frontiers rather than one composite score that can hide constraint violations.

### Physical-impact gate

- Read-only shadow mode first.
- Preregister the outcome, denominator, treatment, comparator and analysis.
- Use randomized/crossover blocks where practical; otherwise justify a controlled/matched design.
- Meter energy and accepted product; independently measure quality/yield.
- Report uncertainty, downtime, scrap/rework, confounders and adverse effects.
- No energy/yield savings claim until a physical controlled comparison supports it.

### Replication gate

- Independently reproduce the result on a second split/design/regime or physical reference campaign.
- Freeze source, code, environment and result artifacts.
- Publish negative results and benchmark failures, not only winning runs.

## Implementation sequence

| Increment | Deliverable | Claim ceiling |
|---|---|---|
| 001A | This constitution and manifest | Scope and protocol are documented |
| 001B | Data rights/source snapshots, schemas and frozen split manifests | Data integrity and split identity |
| 001C | Independent deterministic baselines/evaluators | Exact benchmark computations reproduced |
| 001D | Symtropy twin/replay adapters and shift/uncertainty evaluation | Twin behavior and decision replay only |
| 001E | Mycelix provenance/result receipts and lifecycle bindings | Traceability of recorded claims |
| 001F | Independent reproduction and public results | Only the exact reproduced benchmark claim |
| 001G | Partner/site physical campaign, if available | Only the measured, qualified physical outcome |

## Reproducibility packet

Every scored run should publish:
- benchmark schema/version and issue/PR;
- dataset source, license, exact version, digest and exclusions;
- split manifest and seed list;
- code commit and build/toolchain identity;
- PDK, libraries, constraints, tool versions and corner where relevant;
- model architecture, configuration, weights digest and training budget;
- hardware, runtime, peak memory and energy-measurement method where available;
- raw result references, derived metrics, confidence intervals and failed controls;
- reviewer disposition and explicit claim ceiling.

## Non-goals

This benchmark constitution does not:
- provide fabrication recipes or hazardous process instructions;
- authorize equipment or fab operation;
- claim a qualified process, foundry-ready design or production PDK;
- claim measured wafer yield improvement;
- imply that open PDKs are production-qualified;
- make a public dataset representative of a modern leading-edge fab;
- establish that any Symthaea/Symtropy/Mycelix recommendation is physically correct without independent evidence.

## References

1. UCI Machine Learning Repository, [SECOM](https://archive.ics.uci.edu/dataset/179/secom).
2. WM-811K reference study, [Journal of Intelligent Manufacturing / Springer](https://doi.org/10.1007/s10845-024-02390-7).
3. CircuitNet, [dataset documentation](https://circuitnet.github.io/intro/intro.html).
4. OpenROAD, [QoR metrics documentation](https://openroad-flow-scripts.readthedocs.io/en/latest/contrib/Metrics.html).
5. SkyWater, [SKY130 PDK status and documentation](https://skywater-pdk.readthedocs.io/en/main/status.html).
6. IHP, [IHP Open PDK repository and status](https://github.com/IHP-GmbH/IHP-Open-PDK).
7. UCIe Consortium, [UCIe specifications](https://www.uciexpress.org/specifications/ucie-3-0).
8. Imec, [CFET roadmap: process modules and standard cells](https://www.imec-int.com/en/articles/advancing-cfet-based-device-roadmap-novel-integration-modules-and-standard-cell).
9. ASML, [2025 annual report strategy and stories](https://www.asml.com/en/investors/annual-report/2025/strategy-and-stories).
10. SEMI, [Standards and smart manufacturing](https://www.semi.org/en/about/SEMI_Standards).
11. KLA, [FY2025 filing](https://www.sec.gov/Archives/edgar/data/319201/000031920126000008/klac-20251231.htm).
12. IEEE CEDA, [First Commercial Benchmarking — Siemens EDA](https://github.com/ieee-ceda-datc/First-Commercial-Benchmarking-SiemensEDA).
13. NIST, [Digital twin research and manufacturing VVUQ challenges](https://doi.org/10.6028/NIST.IR.8620).


## Current 2026 baseline and source-register updates

The follow-on literature review is in [2026-baseline-review.md](2026-baseline-review.md). Two recent anchors materially affect the benchmark protocol:

- **SECOM:** Park & Baek (24 June 2026) report a supplementary SECOM PR-AUC result of 0.2207 for LightGBM with zero-imputed features. Their 60/20/20 stratified experiment had only 21 fail examples in validation and test; a validation-selected precision-at-least-0.90 threshold produced zero test recall. This is a useful reproduction anchor, not a broad fab-transfer claim.
- **WM-811K:** Song et al. (31 March 2026) report PSS-HNet Macro-F1 0.9098 ± 0.0008 and Macro-recall 0.8954 ± 0.0034 under an 80/20 stratified sample-level split. Their iFormer baseline Macro-F1 is 0.9044 ± 0.0010. They identify cross-dataset/process-condition generalization and batch-latency variability as remaining work.

The current source and rights register is [data-source-register-v0.1.json](data-source-register-v0.1.json). It distinguishes metadata verification from an exact downloaded dataset snapshot, and leaves exact file digests null until acquisition is completed. Raw datasets are not checked in.

## Executable reference evaluator

The test-only independent binary metric reference scorer is in `../../benchmarks/semi-bench-001/reference/`. It freezes threshold metrics, average precision, ROC AUC, Brier score, fixed-bin ECE, FPR at target recall and tie-grouped risk-coverage semantics. Eleven local unit tests passed; the path-filtered GitHub Actions workflow is `.github/workflows/semi-bench-reference.yml`.

The Python code is an independent test/reference oracle only. The production implementation remains Rust-first. Synthetic known-answer results are not empirical SECOM or wafer-map results.


## Validation-selected operating-point policy

The reference evaluator now includes a separate validation-only selector that maximizes recall subject to a declared minimum precision, groups equal-score ties, rejects validation/test identity overlap and records an explicit fallback when no observed threshold is feasible. A frozen 10-validation / 10-test known-answer fixture covers this policy. It directly addresses the 2026 SECOM supplementary finding where a high-precision validation operating point produced zero test recall with only 21 positive test observations. The selector does not make a scarce validation sample statistically reliable; confidence intervals and operating-point stability remain required.


## End-to-end EDA benchmark follow-up

The 2026 frontier review is documented in [2026-industry-frontier.md](2026-industry-frontier.md). The new [SEMI-EDA-001 #13069](https://github.com/Luminous-Dynamics/symthaea/issues/13069) workstream adds ChiPBench and PPAPlace as end-to-end placement references. It makes final routed WNS/TNS/PPA the adjudicator and treats HPWL/pre-route predictions as intermediate surrogates that must pass a held-out fidelity gate.

The benchmark scope also maps to NIST's TIDAL manufacturing-intelligence programme and the SEMI proposal for a digital-twin specification in semiconductor manufacturing. These are external architecture/standards references, not Luminous conformance claims.


## Split-manifest admission gate

An independent split validator has been added alongside the metric scorer. It distinguishes group-disjoint, forward-time and stratified sample splits, rejects leakage/identity errors and hashes canonical assignments. This prevents a benchmark report from implying lot independence or temporal generalization when the available dataset only supports a random stratified split. See `benchmarks/semi-bench-001/reference/python/semi_split_manifest_reference.py` and its frozen synthetic known-answer fixture.


## New 2026 inspection, metrology and packaging work

See [2026-inspection-metrology-packaging-frontier.md](2026-inspection-metrology-packaging-frontier.md). New follow-up workstreams:

- [SEMI-INSPECT-001 #13083](https://github.com/Luminous-Dynamics/symthaea/issues/13083): budget-aware adaptive inspection, delayed labels, noisy annotations and latent risk in uninspected wafers.
- [SEMI-MET-002 #13084](https://github.com/Luminous-Dynamics/symthaea/issues/13084): physics-bounded AI SEM dimensional metrology, detection limits and sparse scanning.
- [SEMI-PKG-TEST-001 #13085](https://github.com/Luminous-Dynamics/symthaea/issues/13085): known-good-die, chiplet package test coverage and lifecycle observability.

These extend the programme from classifying inspected examples to optimizing inspection decisions, measuring AI detection limits and engineering package testability.


## Adaptive inspection decision reference

A fourth test-only reference primitive is now included: a budgeted inspection-allocation oracle. It combines modeled defect risk with test sensitivity/specificity, actionability, exposed population, avoidable loss, false-hold cost, direct inspection cost and delay penalty. It computes the exact optimal subset for small finite candidate sets and compares it with a matched-budget risk-only greedy policy. Its synthetic known-answer is part of the SEMI-BENCH-001 workflow; it does not establish physical inspection savings.

See [SEMI-INSPECT-001 #13083](https://github.com/Luminous-Dynamics/symthaea/issues/13083).


### Current executable verification

The updated 31-test reference suite, including the budget-aware inspection-selection oracle, passed in hosted GitHub Actions run [#44890353550](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44890353550). The job also reproduced both the binary-metrics and inspection-decision expected reports byte-for-byte. This is a hosted CI result, not a local test claim.


## Real-data and upstream substrate manufacturing update

The review [2026-real-data-and-upstream-frontier.md](2026-real-data-and-upstream-frontier.md) records an external SECOM benchmark audit, MixedWM38 synthetic-data caveats, 2026 compositional zero-shot literature, and the upstream silicon substrate manufacturing frontier.

New workstreams:
- [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113): leave-combination-out mixed-defect recognition, open-set rejection and synthetic-to-real separation.
- [SEMI-SUBSTRATE-001 #13114](https://github.com/Luminous-Dynamics/symthaea/issues/13114): crystal growth, ingot slicing, wafer machining, material utilization and substrate metrology.

The key change is to treat random-split performance on GAN-augmented MixedWM38 as a data/pipeline check, not a claim of factory generalization. For SECOM, the public external baseline uses a stratified random split and must remain separate from a forward-time evaluation.


## SECOM source integrity and chronology audit

A public UCI-attributed SECOM mirror was audited without importing its raw data into the repository. The audit confirms 1,567 aligned feature/label rows, 590 numeric sensor columns, 41,951 missing values (all rows contain at least one), 104 failures, non-decreasing source timestamp order and 33 repeated timestamp rows. The mirror's Git blob identities are recorded; byte identity to the official UCI download remains unverified.

A contiguous chronological 60/20/20 proposal contains 76/11/17 failures in train/validation/test. This is a source-clock chronology stress split, not an evaluated model result. Validation/test positive counts are small; no lot/tool IDs are present, and timestamps have no timezone. Full audit: [secom-source-integrity-audit-v0.1.json](secom-source-integrity-audit-v0.1.json).

The split validator now supports either offset-aware timestamps or source-local naive timestamps with an explicit clock-domain ID. The latter establishes ordering only within that declared source clock; it does not create UTC or cross-site time comparability.


## Frozen SECOM chronological split manifest

The source audit now has a machine-readable row-level split manifest at [splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json](splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json). It binds all 1,567 row identities, source labels and source-local timestamps to the exact public mirror commit and per-file digests. The split is contiguous by source row/time, with no timestamp group crossing the cut points.

Positive support is 76 failures in train, 11 in validation and 17 in test. This is a frozen chronology stress split—not a trained-model result. The source has no timezone and no lot/tool IDs. It supports only within-source-clock ordering, not UTC, equipment-held-out or lot-held-out claims.


## Latest hosted verification and frozen data split

The current hosted reference workflow passed **35/35 tests** in [GitHub Actions run #44899113316](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44899113316), on executable-content commit `b98ca276e416f22113e6ad3065c14380294c6b3c`. The run includes the source-local chronological SECOM manifest test and byte-for-byte reproduction of the binary-metric and inspection-decision reports. No local test execution is claimed.

The actual row-level chronology manifest is frozen at [the SECOM split file](splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json). It contains 1,567 source-row identities with 76/11/17 failures in train/validation/test. It is not a trained-model result and supports only source-local chronological ordering; no timezone, lot, tool or site identity is inferred.


## Missingness drift follow-up

The SECOM source audit now includes missing-value profiles by chronological split and month. The sensor-cell missingness rate is 5.14% in train, 3.42% in validation and 3.84% in test; it declines from 5.33% in August to 3.77% in October. All rows have at least one missing sensor value, 28 features are over 50% missing, and 52 features have no missing values. This warrants explicit missingness-only and indicator ablations; it is not a process-cause interpretation.

New workstream: [SEMI-SECOM-MISS-001 #13115](https://github.com/Luminous-Dynamics/symthaea/issues/13115), covering missingness-only shortcuts, MAR/MNAR shifts, train/test observation-regime changes and abstention.


## Lot-disjoint, open-world and multi-view wafer frontier

See [2026-wafer-lot-and-multiview-frontier.md](2026-wafer-lot-and-multiview-frontier.md). The new follow-up workstreams separate three capabilities:

- [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122): unseen-lot and time-forward wafer-map generalization, including split-local contrastive pairs and OOD calibration integrity.
- [SEMI-MULTIVIEW-001 #13123](https://github.com/Luminous-Dynamics/symthaea/issues/13123): item-wise CP/DFC map fusion, missing-view robustness and industrial open-world inspection.
- [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113) and [SEMI-OPENSET-001 #13121](https://github.com/Luminous-Dynamics/symthaea/issues/13121): unseen compositions, unknown defects and continual learning.

The 2026 GNN wafer-map paper reports 98.6% accuracy and 89.3% LOCO OOD detection, but its reported split is sample-level stratified rather than lot-disjoint. A review of the released OOD script also found an apparent calibration mismatch: the paper describes a validation-derived threshold, while the public script appears to derive the 5th-percentile threshold from test in-distribution scores. This is a paper/repository reproduction discrepancy to resolve, not evidence that the published result itself is invalid.

The PPAPlace-5000 data release provides 5,000 complete placements with post-GRT timing/power/area labels across ten ChiPBench circuits. Its Hugging Face license metadata is “other”; exact terms must be reviewed before downloading, caching or redistributing the 8.47 GiB release.


## Contrastive-pair lineage gate

The reference layer has been extended with a split-local contrastive-pair validator at `benchmarks/semi-bench-001/reference/python/semi_pair_manifest_reference.py`. It prevents positive/negative pairs from crossing train/validation/test boundaries, prohibits pair identity duplication and requires evidence for same-lot or near-time production-proximity pair labels. This is important for reproducing the 2026 wafer-map GNN that uses lot/time context in contrastive pair construction.

See [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122).


## Contrastive-pair integrity reference

The branch now includes [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122) and a test-only contrastive-pair manifest validator. The validator rejects cross-split pairs, scope mismatches, duplicate pair/edge identities, self-pairs, orphan sample references and unsupported same-lot/time-near production-proximity claims. It binds time-near pairs to a declared clock domain and embargo.

The new validator's 11 tests passed locally, and its CLI reproduced the frozen JSON report byte-for-byte. The complete reference suite now contains 46 tests; the latest hosted GitHub Actions pass still covers the preceding 35-test version. No hosted pass is claimed yet for the new pair validator.


## Latest clean-base verification

The clean-base successor PR [#13126](https://github.com/Luminous-Dynamics/symthaea/pull/13126) has a verified hosted GitHub Actions pass: [run #44917273962](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44917273962), commit `5f8ffa91bac2cb8cee923d0f1968d2f57fd01149`. The full reference suite passed **46/46**, and binary-metric, inspection-decision and contrastive-pair JSON reports all reproduced byte-for-byte. This is hosted CI evidence; no full local run of all 46 tests is claimed.


## Latest verification on the current clean-base PR head

Hosted GitHub Actions run [#44917385489](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44917385489) completed successfully on commit `010faa4a517d4dc46a57a9b0f6b07015a188bf35`. All **46/46** reference tests passed, and the binary-metric, inspection-decision and contrastive-pair reports each matched their frozen expected JSON byte-for-byte. No full local run of all 46 tests is claimed.


## Rust-first semiconductor implementation increment

The earlier planning direction in ENG-SEMI #5671 proposed a semiconductor/device-physics domain crate. That has now been started on the clean-base benchmark branch:
- `crates/domains/symthaea-semiconductor` is the initial Rust domain crate.
- `src/benchmark/binary_metrics.rs` implements the versioned SEMI-BENCH binary-risk scoring contract with grouped ties, threshold metrics, AP/AUROC, Brier/ECE, FPR-at-recall and risk/coverage.
- Its Rust tests load the frozen independent Python input/output fixtures and compare the complete JSON report, plus invalid-input and order-invariance tests.
- The Python scorer remains a test/reference oracle; it is not a production dependency.

The first Rust CI attempt found that the current root workspace cannot be resolved because Cargo's workspace glob includes a missing `crates/core/symthaea/Cargo.toml` on main. The new SEMI-BENCH workflow now tests this crate in a minimal temporary workspace copied from the exact checkout, avoiding an unrelated repository-wide manifest repair. Hosted run [#44933592943](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44933592943) passed on commit `b0d17926ccedd5bd283556aec275302ea294a199`: rustfmt passed, clippy passed with warnings denied, all six Rust unit tests passed, and the Rust CLI reproduced the frozen Python report byte-for-byte. The checks ran in a minimal temporary workspace because the root workspace references a missing `crates/core/symthaea/Cargo.toml`; no unrelated workspace fix was made.

Tracked by [SEMI-RUST-001 #13129](https://github.com/Luminous-Dynamics/symthaea/issues/13129).


## Rule-ordered multimodal zero-shot benchmark

New preprint review and follow-up: [SEMI-ZS-LLM-001 #13146](https://github.com/Luminous-Dynamics/symthaea/issues/13146). Cao & Tao (arXiv:2610.35080, 26 Oct 2026) report a frozen GPT-4o with connected-component highlighting and morphology/location/exclusion rules: 92.9% macro / 93.8% overall on a six-class MixedWM38 subset, and 79.1% macro / 81.3% overall on six classes from WM-811K. On MixedWM38, eight alternate rule orders range 90.1–91.9% overall. This preprint excludes the remaining mixed combinations and is not a lot/time-held-out result. The benchmark therefore measures rule-order robustness, evidence grounding, open-set abstention, cost/latency and hybrid HDC+LMM performance rather than selecting only the best prompt.


## Closed-loop fab decision-support benchmark

The November 2026 human-supervised autonomous decision-support review motivates benchmarking planning, scheduling, dispatch, process-control feedback and metrology as one coupled system. New follow-up [SEMI-FAB-DECISION-001 #13147](https://github.com/Luminous-Dynamics/symthaea/issues/13147) reuses existing process/equipment/subfab/metrology and Symtropy replay owners. It compares fab-native heuristics, constrained optimization, simulation, quality-aware dispatch, inspection value-of-information and bounded LLM assistance. Its staged evidence progression is synthetic DES -> historical replay -> offline counterfactual -> shadow mode -> human-approved pilot -> preregistered outcome study. No process authority or physical yield claim is implied.


## Latest Rust-native integrity gate

Hosted GitHub Actions run [#44934333447](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44934333447) passed on commit `4a7f7029fa8694914f0c85650053d879e399b402`:

- Rust format check: pass.
- Clippy with warnings denied: pass.
- Rust unit tests: **18 passed, 0 failed**.
- Rust binary metrics CLI report: byte-for-byte match with the frozen independent Python reference.
- The Rust split validator verified the synthetic group-disjoint known answer and the actual 1,567-row source-local SECOM chronological manifest, including 940/313/314 row counts and 76/11/17 failure support.
- The Rust contrastive-pair validator passed known-answer, cross-split, duplicate-edge, same-lot, time-embargo and clock-domain tests.
- The Python/reference job in the same run passed 46/46 tests and reproduced its three frozen reports.

These checks ran in a minimal temporary Cargo workspace because the repository root workspace currently references a missing `crates/core/symthaea/Cargo.toml`. The isolated workspace avoided changing unrelated root workspace state. This verifies contract and leakage logic, not a trained model's predictive performance or physical fab outcome.

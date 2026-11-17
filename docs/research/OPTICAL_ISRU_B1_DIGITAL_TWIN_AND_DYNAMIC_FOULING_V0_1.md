# Optical ISRU B1 digital twin and dynamic optical-health model v0.1

**Status:** standalone Rust reference modules authored; exact-head validation pending  
**Parent:** Symthaea #12574 — ISRU-BENCH-002  
**Base:** ISRU-BENCH-001 #12570 / B0 replay-oracle PR #12572  
**Scope:** software benchmark twin only; no physical rig qualification

## 1. Why the B1 layer matters

Autonomous-laboratory research has moved beyond choosing the next parameter point. Three current references establish the direction:

- Autonomous Science Bench 0.1 (1 October 2026) grades the whole model-agent-tool-harness configuration, separates discovery from learning on never-measured candidates, and defines replay-oracle, digital-twin and live-lab backends. Its first pilot has six tasks and warns that three trials per task are insufficient to read small leaderboard differences as meaningful: https://asb.open-ssi.org/
- The 31 July 2026 *Nature Reviews Chemistry* review argues that the next phase of self-driving laboratories depends on scalability, generalizability and provenance-complete experiments across laboratories and domains: https://www.nature.com/articles/s41570-026-00847-2
- The 26 August 2026 review of Bayesian optimization for self-driving materials labs highlights practical complications beyond textbook BO: failed/missing experiments, measurement noise and drift, mixed variables, constraints, multiple objectives, variable cost/fidelity, transfer, asynchronous work and physics-informed priors: https://arxiv.org/abs/2608.26016
- NIST identifies four standards needs for modular autonomous labs: sample management; instrument control and communication; data and knowledge management; and algorithm/model integration: https://www.nist.gov/programs-projects/development-standards-support-modular-and-autonomous-laboratory-ecosystem

The benchmark should therefore qualify the complete experimental decision system: material physics, apparatus state, measurement validity, asynchronous execution, resource use and evidence provenance.

## 2. B1 architecture

The B1 backend preserves the candidate/task/action semantics of B0 and changes the result adapter to a seeded, stateful stochastic twin.

`research/asteroid-optical-isru-v0.2/digital_twin.rs` models:
- four causally ordered stages: characterization, exposure, capture and assay;
- distinct instruments with exclusive locks and different duration scaling;
- asynchronous completion ordered by simulated completion time, including out-of-order completion;
- deterministic keyed stage duration, failure and measurement-noise streams;
- hard experiment, elapsed-time and resource-vector budgets;
- launch-time reservation and completion/failure settlement;
- instrument calibration expiry and explicit recalibration actions;
- instrument maintenance state and cost;
- failure that consumes resources and releases locks without advancing candidate stage lineage;
- candidate/holdout separation, sealed truth and complete holdout prediction requirements;
- explicit `SimulatedIndependentAssay` output type and `simulation_only` marker.

The initial instrument maintenance/recalibration API is intentionally conservative: maintenance operations require no active jobs and are synchronous. B1 does not yet simulate robotic cleaning concurrency or live instrument-control protocols.

### Event and cost semantics

- The instrument lock is acquired at stage launch.
- A multi-resource reservation is recorded before execution.
- The stage result or failure is resolved at its scheduled completion time.
- Reserved maximum duration is released; actual simulated duration is settled to spent resources.
- Failed work consumes its actual time/non-time cost, releases the lock and does not advance the candidate's causal stage.
- Repeated action sequence + same seed + same manifest reproduces the same event/output trace.
- The same candidate cannot have two concurrent stage jobs.

The twin's deterministic noise uses a centered sum of twelve uniform draws (Irwin–Hall approximation) with nominal unit variance. This is explicitly a benchmark approximation, not a calibrated physical sensor-noise model. The fixed seed schedule enables common-random-number comparisons; B2 must estimate actual heteroscedastic and correlated sensor noise from calibration/replication data.

## 3. Dynamic optical-health state is separate from instrument health

The second reference module, `research/asteroid-optical-isru-v0.2/optical_health.rs`, tracks separate state for:

1. **Production optic:** wavelength-dependent throughput of the actual process path.
2. **Optical witness:** representative witness-surface loading and transmission.
3. **QCM:** local areal mass deposited on its crystal.
4. **Clean reference channel:** source output used to normalize the process channel and distinguish source drift from optical-path attenuation.

A deposition event has species-resolved flux (organic film, volatile salt, mineral dust) and explicit transfer factors for the process optic, witness and QCM. These factors are model inputs that must be experimentally characterized for the actual source, view factors, surface temperature and geometry. They are not universal coefficients.

The optical model computes spectral transmission from species-specific extinction coefficients and surface roughness. Inspection produces independent spectral process/reference power channels, witness transmission and QCM areal mass. Source power drift is removed through per-band ratio normalization.

**Non-substitution rule:** QCM mass is a local deposition measurement. It does not become optical transmission. Witness transmission does not automatically equal production-optic throughput. The model preserves all three separately so a policy cannot infer exact process-optic throughput from one QCM number.

### Cleaning and requalification

A cleaning protocol declares species-dependent removal fractions, roughness change, duration, energy and consumables. The receipt records before/after loading on all three surfaces. Cleaning is not assumed harmless: it can alter roughness and the optical transfer function.

After cleaning:
- the optical health model is unqualified;
- the next action cannot be treated as normal production continuation;
- a fresh post-clean inspection is required;
- requalification succeeds only if process spectral throughput, witness loss and QCM mass satisfy the declared thresholds;
- a stale pre-clean reading cannot requalify the optical system.

The model can return Continue, RequestIndependentInspection, CleanAndRequalify or PauseCampaign. These are B1 simulated decisions, not live interlocks.

## 4. Contamination monitoring: current standards and adaptation boundaries

The current ECSS cleanliness standard is **ECSS-Q-ST-70-01C Rev.1 (15 October 2025)**, which cancels and replaces the 2008 issue. Its scope includes cleanliness-sensitive/critical item selection, required cleanliness levels, control through design/development/production/testing/operation and contamination-related failure prevention. It is an applicable design/monitoring reference, not an ISRU acceptance limit: https://ecss.nl/standard/ecss-q-st-70-01c-rev-1-cleanliness-and-contamination-control-15-october-2025/

The published Rev.1 text specifies quantitative monitoring using direct surface measurements, witnesses and/or in-situ QCM. For contamination-critical items it calls for QCM together with witness methods; witness temperature and location should be representative. Visual inspection is qualitative rather than a quantitative cleanliness measurement. This supports independent state channels, not a one-sensor shortcut.

ASTM E1559-09(2022) characterizes outgassing release and deposition on surfaces at specified temperatures using QCM. Its stated test conditions include specimen temperatures of 298–398 K and external chamber pressure below 7 × 10⁻³ Pa. That scope is not automatically transferable to lunar/vacuum ISRU conditions, high-temperature excavation or different geometry: https://store.astm.org/standards/e1559

ECSS-Q-ST-70-05C Rev.2 (1 July 2024) defines direct and indirect detection of organic surface contamination by infrared spectroscopy and includes witness-plate and calibration considerations: https://ecss.nl/standard/ecss-q-st-70-05c-rev-2-detection-of-organic-contamination-of-surfaces-by-infrared-spectroscopy-1-july-2024/

ECSS-Q-ST-70-54C also cautions that simplified witness geometry may overestimate cleaning efficacy compared with real hardware; process validation requires an interpretation for transfer to actual geometry: https://ecss.nl/wp-content/uploads/2017/04/ECSS-Q-ST-70-54C%2815February2017%29.pdf

**Important adaptation boundary:** these standards provide measurement and cleanliness-control practices. Their contaminant limits, thermal conditions, test geometries and qualification criteria must not be copied into an ISRU rig without a separately justified tailoring and validation process.

## 5. Material-regime and pulse-control model

A third standalone module, `research/asteroid-optical-isru-v0.2/material_regime.rs`, now models a minimal coupled thermal/material response:

- incident beam energy versus absorbed energy after measured process-optic transmission;
- heat capacity and active thermal depth;
- spallation onset, volatile-release onset, brittle–ductile transition and melt onset;
- melt-layer growth and its suppression of spallation/surface renewal;
- surface volatile depletion and refresh due to spallation;
- released water/organic/salt vapour versus captured water;
- organic/salt/dust deposition estimates on the optical path; and
- separate gas-analyzer water signal, including moisture introduced by purge gas.

It supports a controlled comparison of equal total incident energy delivered continuously or in pulses with cooling intervals. The model is intentionally small and parameterized; it is not a finite-element heat-transfer, DEM spallation, radiative-transfer or volatile-transport solver.

### Why this is a field-relevant physical effect

The October 2025 optical-mining study reports that beam irradiance distribution, mineral morphology and the brittle–ductile transition affect excavation; organic-vapour deposition on sacrificial glass can progressively block light; and surface melt can inhibit surface refresh and lower water production. Its test-bed results report Nectar A excavation up to about 6 kg/h and a water rate around 16 g/h, while Nectar B water estimates were substantially lower (maximum approximately 4.75 g/h in the reported comparisons). These are specific experimental configurations/simulants, not a universal process benchmark or commercial production rate. The study also describes a wet CO₂ purge contaminating the water signal, which is why the model separates captured water from gas-analyzer signal.

Primary study: *Optical mining of carbonaceous chondrite simulants: Testing and lessons learned in asteroid mining research*, Acta Astronautica 235 (October 2025), DOI 10.1016/j.actaastro.2025.04.033: https://doi.org/10.1016/j.actaastro.2025.04.033

### Industry comparisons—different process families

| Reference | Demonstrated capability | Correct comparison boundary |
|---|---|---|
| Nectar A/B optical mining (2025) | Test-bed excavation and water-release/capture behavior under focused irradiation; strong morphology, surface refresh and optical deposition effects | Relevant physical behavior for the optical-mining model; not directly comparable to oxygen electrolysis |
| NASA/Lunar Resources LR-1 MRE (ICES 2026 presentation; late-2024 campaign) | 25 kg simulant; reported 0.07 kg O₂/h average for 9 h under vacuum with real-time VMOMS gas monitoring | Molten-regolith electrolysis throughput benchmark, not optical water mining |
| NASA CaRD solar-carbothermal prototype (AIAA ASCEND 2026) | Integrated solar concentrator, optical splash plate/shutter, carbothermal reactor, fluid/gas analysis and control; CO detection confirmed carbothermal reaction | Integrated subsystem/interface and energy-accounting benchmark; do not equate this to demonstrated commercial production |

Sources:
- Optical mining paper: https://doi.org/10.1016/j.actaastro.2025.04.033
- NASA MRE / ICES 2026 NTRS record: https://ntrs.nasa.gov/citations/20260006167
- NASA CaRD / AIAA ASCEND 2026 NTRS record: https://ntrs.nasa.gov/citations/20260002864

Luminous's competitive contribution should be judged on fewer experiments to a qualified process envelope, lower false attribution of equipment degradation as material physics, more accepted simulated/physical product per end-to-end resource, and reproducibility across labs—not on claiming higher excavation or oxygen production before running a physical campaign.

### 5.1 Coupled material–optical campaign execution

The standalone `coupled_campaign.rs` module now composes the material-regime and optical-health models into a single pulse transaction.

Causal order:
1. Inspect current source-reference, process-path, witness and QCM state.
2. Apply the health decision gate.
3. Read current process-optic transmission at the selected process band.
4. Apply the pulse using that current transmission.
5. Convert simulated material emissions into a deposition event.
6. Apply independently declared transfer factors to process optic, witness and QCM.
7. Commit material and optical states together only if every step validates.

The material and optical models are cloned into staged state. If material processing or the subsequent deposition event fails, neither state is committed. A later pulse therefore sees the contamination caused by earlier pulses, not a static clean-optic assumption. The integration emits simulation-only receipts and cannot mint physical calibration, custody, assay or authority evidence.

Regression cases cover progressive attenuation of the next pulse, distinct witness/QCM/process-optic observations, wet-purge analyzer bias without extra captured water, and rollback when the deposition/metrology stage rejects the event.

This is a first coupled surrogate, not a validated transport model. In particular, the material model's estimated areal emission load and the optical model's transfer factors/extinction coefficients are calibration parameters. They must be fitted from representative geometry, temperature, species and wavelength measurements before any physical prediction claim.

## 6. Paradigm shifts to prioritize

### A. Joint material–apparatus inference
Fit a causal model with separate latent variables for material state, production-optic state, instrument calibration state, source state and sample state. A falling proxy should not be attributed to material physics until instrument/reference/optic explanations have been tested.

### B. Active metrology and maintenance
The action space should include process another sample, inspect QCM/witness, inspect the process optical path, measure a clean reference, recalibrate, clean/shield, requalify and pause. The optimal action can be a maintenance or diagnostic intervention rather than another material experiment.

### C. Counterfactual apparatus interventions
Randomize shielding, witness placement, source power reference checks and cleaning protocol on matched conditions. Estimate causal effects of contamination controls rather than merely correlating fouling with lower output.

### D. Qualification-aware optimization
The objective is independently accepted product per total campaign resource, with hard constraints on evidence, safety and metrology. Proxy yield and model confidence are not production outcomes.

### E. Cross-lab transfer with an apparatus domain model
Separate material-family transfer from instrument/path transfer. Report which parts of the model transfer, which require a small calibration campaign and which must be relearned.

## 7. B1 benchmark matrix

Compare on identical scenarios, candidate pools, action spaces, seeds and budgets:

- fixed expert/reference workflow;
- random feasible candidate and maintenance actions;
- GP-EI / constrained BO;
- batch BO;
- multi-stage BO with intermediate proxy outputs;
- asynchronous realistic-mean fantasy;
- asynchronous pessimistic-bound / constant-liar;
- seeded random fantasy and no-fantasy asynchronous BO;
- EIG / EVSI material-only policy;
- joint material-plus-apparatus EIG / EVSI;
- full Luminous workflow planner with stage DAG, metrology checks, resource Pareto constraints and asynchronous scheduling.

### Required B1 scenario families

1. Clean stationary apparatus.
2. Source-power drift with clean optic (reference normalization control).
3. Wavelength-dependent optical fouling with stable source.
4. QCM/witness view-factor mismatch.
5. Instrument calibration expiry without optic fouling.
6. Simultaneous calibration expiry and optical fouling.
7. Cleaning that removes contamination but damages roughness/throughput.
8. Intermittent stage failure with partial work and retries.
9. Long-tail stage durations and asynchronous out-of-order outcomes.
10. Holdout material family with different contaminant composition.
11. Transfer of material model across a new instrument/path.
12. Adversarial proxy/holdout leakage and impossible action controls.

### Metrics

Scientific: discovery gate, holdout learning gate, final simulated-assay ranking, predictive interval coverage, log score, Pareto regret.

Optical/metrology: spectral transmission error, source-drift rejection, deposition-mass reconstruction error, witness-to-process-optic transfer error, detection delay, false attribution of material transition to fouling, cleaning recovery fraction, requalification false-pass rate.

Operations: wall-clock to qualified simulated decision, accepted-simulated-product per campaign hour, rig utilization, queue time, calibration/maintenance downtime, failure cost, resource vector, salvage fraction.

Trust: zero holdout leakage, zero simulator-to-physical evidence promotion, zero stale calibration use, zero stage-order violations, zero budget overrun, deterministic replay equality.

### External benchmark anchors and comparison discipline

The comparison set must be updated as the field moves; the following publications add concrete, current anchors:

| External work | What it demonstrates | What B1 must compare (and must not claim) |
|---|---|---|
| BALPI, first published 4 June 2026 | GP classifier/regressor phase mapping; MES, BALD, SMOCU and e-straddle; resolves disconnected regions in CALPHAD oracle worlds | Reproduce e-straddle(c), e-straddle(b), SMOCU and MES on matched hidden phase worlds. Its CALPHAD results are not physical ISRU validation. |
| Cost-aware nanoindentation BO, first published 22 June 2026 | Heteroskedastic GP-BO with drift stabilization, lateral movement and reconfiguration costs; nearly 30× efficiency versus grid mapping in that thin-film workflow | Include instrument stabilization, inspection, calibration and reconfiguration costs. Do not transplant the 30× result as an ISRU target or claim. |
| Cost-aware batch BO with hierarchical/deep GP (2026) | Multi-fidelity/heterotopic query allocation and batch optimization across multiple material properties | Include cheap proxy vs expensive assay actions, batch selection and multi-fidelity baselines; test whether extra model complexity helps only on nonlinear/high-dimensional worlds. |
| NASA LR-1 molten-regolith electrolysis (ICES 2026) | 25 kg simulant processed; reported 0.07 kg O₂/h average for a 9 h vacuum campaign | Industrially relevant sustained-throughput and monitoring comparator for a different extraction route. Not directly comparable to asteroid optical water release/capture. |
| NASA CaRD solar-carbothermal prototype (2026) | Integrated solar concentrator, optical splash plate/shutter, reactor, gas analysis and control; CO detection confirmed reaction | Systems integration, optical protection, gas measurement and energy-boundary comparator. CO confirmation is not a quantified commercial oxygen-production rate. |
| Nectar A/B optical-mining campaign (2025) | Irradiance/morphology-dependent spallation, brittle–ductile transition, melt/surface-refresh effects, organic fouling and purge-water signal contamination | Directly relevant physical failure-mode evidence for the optical-mining model; reported rates remain specific to the test-bed and simulants. |

Sources:
- BALPI: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00459d
- Cost-aware nanoindentation: https://pubs.rsc.org/en/content/articlehtml/2026/dd/d5dd00518c
- Cost-aware batch BO: https://www.nature.com/articles/s41524-026-01981-7
- NASA LR-1 MRE: https://ntrs.nasa.gov/citations/20260006167
- NASA CaRD: https://ntrs.nasa.gov/citations/20260002864
- Nectar A/B: https://doi.org/10.1016/j.actaastro.2025.04.033

A further paradigm shift is to score the campaign against the *first correct qualified decision*, not the best predicted response. This requires action-level baselines that can choose an experiment, reference measurement, QCM/witness inspection, calibration, cleaning/requalification, independent assay or pause. The dedicated decision-directed benchmark is tracked in [ISRU-DECISION-BO-001 #12607](https://github.com/Luminous-Dynamics/symthaea/issues/12607).

## 8. Initial targets (hypotheses, not achieved results)

- Detect an optical-health threshold crossing before the next exposure action in at least 99% of frozen B1 failure/contamination scenarios.
- Zero false requalification passes on a sealed suite containing incomplete cleaning, damaged surfaces, stale readings and missing measurement channels.
- Reduce false material-regime change attribution by at least 50% against a material-only optimizer on paired hidden contamination worlds, while preserving held-out material ranking.
- Achieve at least 1.5× faster time-to-qualified-simulated-decision versus the strongest eligible stage-aware baseline, with a paired 95% lower confidence bound above 1.0.
- Preserve nominal interval coverage on held-out material families and apparatus configurations.
- Report costs of inspection, cleaning, requalification and downtime in the same campaign budget as material exposures.

These remain pre-registered target hypotheses until the B1 tournament runs. They are not physical qualification or performance claims.

## 9. Reproduction

The dedicated workflow compiles and tests standalone Rust references with pinned Rust 1.96, rustfmt, strict Clippy and the Rust test harness:

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/digital_twin.rs -o /tmp/isru-digital-twin
    /tmp/isru-digital-twin

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/optical_health.rs -o /tmp/isru-optical-health
    /tmp/isru-optical-health

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/material_regime.rs -o /tmp/isru-material-regime
    /tmp/isru-material-regime

    rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/coupled_campaign.rs -o /tmp/isru-coupled
    /tmp/isru-coupled

Current regression tests cover asynchronous out-of-order completion, lock contention, stage prerequisites, holdout access, calibration expiry, instrument maintenance, failures, resource/time budgets, deterministic replay, noise moments, optical reference normalization, QCM/witness/process-optic separation, spectral dependence, cleaning residuals, monotonic metrology chronology, post-clean requalification, spallation/melt regime changes, equal-energy pulsed versus continuous delivery, purge-moisture interference, optical-deposition estimates and sample-inventory exhaustion.

## 10. Limitations and next increments

- This is a benchmark model, not a validated digital twin of a specific optical ISRU rig.
- The optical extinction and transfer coefficients are illustrative inputs; they require material-specific and geometry-specific calibration.
- Noise is a deterministic approximation, not an empirical instrument noise distribution.
- Maintenance/recalibration are initially synchronous and require an idle event queue; a later iteration should make them first-class asynchronous tasks.
- The model does not yet represent reactive chemistry, particulate adhesion dynamics, thermal gradients, coating delamination, sensor saturation or a validated vacuum/outgassing transport solver.
- The coupled pulse transaction is an initial deterministic composition. It does not yet model spatially resolved deposition, time-varying contaminant chemistry, temperature-dependent sticking/desorption, particulate impact, or uncertainty propagation across the material-to-surface transfer.
- Signatures, external metrology authority, raw-file custody and physical safety interlocks remain external.
- B2 requires representative terrestrial experiments, independently characterized transfer factors, randomized/blocked validation, independent assay and an external safety authority.

The purpose of B1 is to discover which policies and evidence architecture survive realistic modeled failure modes, then design the physical experiments needed to validate the model. It does not substitute for those experiments.

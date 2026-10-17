# Nanotechnology Trajectory Reliability Benchmark

**Status:** preregistered research protocol; no model, trajectory, or benchmark result is qualified by this document.  
**Date:** 2026-10-16  
**Tracking:** [NANO-008 #7654](https://github.com/Luminous-Dynamics/symthaea/issues/7654)  
**Cross-domain contract:** [Nanotechnology Benchmark Charter](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-001-atomistic-discovery/docs/engineering/NANOTECHNOLOGY_BENCHMARK_CHARTER.md)  
**Atomistic discovery prerequisite:** [NANO-001 #7453](https://github.com/Luminous-Dynamics/symthaea/issues/7453)  
**Selector baseline/evaluator:** [NANO-005 #7570](https://github.com/Luminous-Dynamics/symthaea/issues/7570), [PR #7568](https://github.com/Luminous-Dynamics/symthaea/pull/7568)

## 1. The paradigm shift: qualify trajectories, not just snapshots

A low held-out energy or force error does not establish that a learned potential will produce physically credible trajectories, preserve a nanostructure, or explore its potential-energy surface faithfully. Static prediction, local optimization, structural exploration, and time-evolution are distinct claims.

Two 2026 studies motivate separating them:

- Usman et al., *Benchmarking Machine-Learning Interatomic Potentials for Dynamical Stability in Inorganic Semiconductor Nanocrystals: A CdSe Case Study* (preprint, 14 September 2026), reports a 149-atom chloride-passivated CdSe nanocluster study. Its abstract describes 1,000 training and 2,000 held-out DFT configurations and 1 ns trajectories at 300 K. In that tested setting, the model with the lowest reported force MAE was not the most robust over time; an uncertainty-guided augmentation stabilized one model after 34 added configurations, while other tested models remained unstable under their stated augmentation budgets. These are source-reported findings from one preprint and are not independently reproduced here: https://arxiv.org/abs/2609.15299
- Tahmasbi et al., *Benchmarking universal machine learning interatomic potentials on elemental systems*, combines equilibrium equation-of-state tests with minima-hopping structural searches. Its results report that equilibrium-volume performance and structural fidelity can diverge, including failures on distorted structures: https://www.nature.com/articles/s41524-026-02251-2

A related supported-nanoparticle benchmark explicitly studies the decoupling between energy accuracy and structural exploration: https://doi.org/10.1038/s41524-026-02043-8

**Research hypothesis:** adding trajectory-level reliability, uncertainty calibration, and actual reference-call costs to a nanomaterials benchmark will change which models and acquisition policies qualify, and may reveal whether uncertainty-guided refinement buys real reliability per expensive reference calculation.

This is not a request for a universal “nanotechnology score.” Report a vector of task-specific measurements and hard-gate outcomes.

## 2. Symthaea source-eligibility audit

The existing architecture provides pieces that can participate in this workflow, but the internal molecular dynamics module must not be misrepresented as a nanocrystal reference simulator.

Current source locations:

- [Quantum-chemistry capability description](../../tree/main/crates/domains/symthaea-quantum-chemistry)
- [Current AIMD implementation](../../blob/main/crates/domains/symthaea-quantum-chemistry/src/molecular_dynamics.rs)
- [Organic molecular graph layer](../../tree/main/crates/domains/symthaea-organic-chemistry)
- [Materials component](../../tree/main/crates/domains/symthaea-materials)
- [Nanotechnology benchmark charter](https://github.com/Luminous-Dynamics/symthaea/blob/research/nano-001-atomistic-discovery/docs/engineering/NANOTECHNOLOGY_BENCHMARK_CHARTER.md)

The checked-in `molecular_dynamics.rs` is molecular and non-periodic: it uses a molecular `Molecule`, a restricted Hartree–Fock / STO-3G energy, and central finite-difference forces. Its own module comments disclose that initial velocity components are generated from a deterministic hash and uniform distribution rather than a Maxwell–Boltzmann Gaussian; center-of-mass removal and exact-temperature rescaling are absent; and the short-trajectory energy-drift test tolerates up to 0.1 Hartree. Its default run is 50 steps. These facts are source-level findings, not a new runtime test result.

Therefore:

- Do **not** use this implementation as the reference for a 149-atom periodic/surface nanocrystal claim.
- Do **not** silently infer periodic boundary support, a validated interatomic potential, physical temperature sampling, or reliable nanosecond dynamics from the existence of an AIMD module.
- Use a task-matched, version-pinned external reference method or pre-existing dataset for reference labels. The reference must itself be eligible for the declared chemistry, surface/passivation, charge/spin, boundary conditions, and physical question.
- Keep Symthaea's initial role limited to feature/lineage handling, candidate selection, failure-memory retrieval, resource accounting, orchestration, and independent evidence packaging. Promote a Symthaea scientific model only after an explicit held-out comparison supports that role.

## 3. Frozen benchmark question and units of evaluation

### Primary question

For a declared nanostructure family, can a specified atomistic model reproduce reference-supported structural behavior over a frozen simulation horizon without catastrophic numerical or structural failure, and can uncertainty-guided additions of reference configurations reach that reliability target with fewer **actual** reference calls than eligible baselines?

The first replication candidate is the chloride-passivated CdSe nanocrystal system described in the cited preprint, if its exact inputs, code/configuration, data rights, reference settings, and initial states can be obtained and independently identified. Do not assume that the preprint's complete artifacts are publicly available or that its results can be reproduced from the abstract alone.

If that system cannot be admitted, record a blocker and select an alternative only in a separate reviewed decision. Do not substitute a more convenient material while retaining the original benchmark claim.

### Scientific unit and split

The independent unit is a parent nanostructure/system or a scientifically justified family cluster—not an individual MD frame.

Freeze and record source-backed grouping keys, where available:

- parent composition, cluster identity, size and morphology;
- surface termination/passivation and defect family;
- initial-state / relaxation lineage;
- trajectory identity and random seed;
- reference-method settings, calculation lineage, and any relation to model pretraining/fine-tuning data.

Never randomly split frames from a trajectory between train and test. Related initial and relaxed structures, time-adjacent frames, replicas of the same parent system, and duplicated structures must be grouped or explicitly justified before splitting. A separate family-held-out evaluation is required for any claim of transfer to unseen compositions, sizes, facets, or passivation.

### Frozen run manifest

Every trajectory arm must bind to a machine-readable manifest containing at least:

1. source commit and dirty-tree state, model/checkpoint identity and license eligibility;
2. data/source revision, artifact digests, provenance, use terms, and training-exposure disclosure;
3. parent system and trajectory-group IDs, atomic species, atom count, charge/spin when relevant, cell and periodic boundary conditions when relevant;
4. exact starting coordinates/velocities, temperature, integrator, timestep, thermostat/barostat and parameters, constraints, cutoff/neighbour settings, precision and deterministic flags;
5. declared trajectory duration, replica seed, output cadence and snapshot selection;
6. reference model/method, settings, convergence policy, and which snapshots receive reference evaluations;
7. CPU/GPU model, software versions, elapsed time, peak memory, energy/compute accounting, actual reference calls, and all error/termination reasons;
8. hashes of raw coordinates, raw energy/force outputs, model outputs, derived observables, and metric report.

Do not publish a digest supplied by a dataset page as though it was locally recomputed. Do not redistribute underlying data or model weights before source/use-rights review.

## 4. Evaluate five separate axes

Never combine these into one compensating weighted score. A mandatory failure on dynamic reliability, data rights, domain eligibility, or leakage cannot be offset by a low MAE.

### A. Static prediction and local physical consistency

Where task-matched reference values exist, report separately:

- total-energy MAE/RMSE (eV) and clearly specified per-atom normalization (eV/atom);
- force-component and force-vector error (eV/Å), with weighting and atom aggregation defined;
- stress error only where periodic stress is physically defined and the reference protocol provides it;
- relative-energy ordering for a preregistered comparison set;
- finite outputs, SCF/relaxation convergence, invalid structures, and failures.

Do not compare raw total energies across different compositions or atom counts as a generic ranking. Freeze tolerances from the actual downstream decision and reference repeatability before looking at held-out outcomes.

### B. Trajectory reliability and integration sensitivity

For each declared horizon (t), report:

- **trajectory survival:** fraction of attempts still valid at (t), with invalid, divergent, crashed, or structurally failed runs counted as failures;
- **time-to-failure:** each failure time and right-censored completion, with group-aware uncertainty; do not analyze only surviving trajectories;
- **numerical health:** non-finite coordinate/energy/force count, integrator/solver failure, runaway displacement/energy, and the exact termination category;
- **ensemble behavior:** temperature/energy behavior appropriate to the declared ensemble. For NVE, assess total-energy drift; for thermostatted ensembles, do not incorrectly treat kinetic-plus-potential energy as conserved. Report the timestep-sensitivity check;
- **structural integrity:** composition/atom-count preservation where expected, physically justified interatomic-distance/coordination limits, passivation retention, and structural changes judged against the declared scientific question;
- **trajectory observables:** relevant distribution/function comparisons (for example radial distribution, coordination distribution, size/morphology descriptors, or state-transition statistics), using reference snapshots sampled by a frozen rule;
- **structural displacement/fidelity:** an aligned, permutation-aware distance or domain-specific metric only where it has meaning for the chosen dynamics; do not penalize legitimate thermally activated transitions as failures by default.

The benchmark defines a catastrophic event using material-specific chemistry and numerical criteria before held-out runs begin. A trajectory that changes phase or morphology is not automatically “wrong”: adjudication depends on whether the event is physically plausible under the reference conditions.

### C. Uncertainty and abstention

If a model reports intervals or confidence bounds, separate:

- empirical coverage versus nominal 90% and 95% coverage;
- interval width/sharpness at that coverage;
- calibration by held-out structure/passivation/temperature regime;
- correlation of uncertainty with severe errors or impending instability;
- risk-versus-coverage when the system abstains or requests a more expensive reference;
- out-of-domain detection and the cases where the model should refuse to extrapolate.

Provisional screening target for a nominal 90% interval: 85–95% grouped-holdout coverage, with useful interval width and no critical subgroup collapse. This is a provisional gate, not a universal statistical guarantee; record group counts and uncertainty intervals and establish adequate sample size in the preflight.

HDC similarity, model disagreement, ensemble spread, or free-energy/active-inference quantities must not be relabelled as calibrated probability without direct calibration evidence. Cognitive free energy is not thermodynamic free energy.

### C2. Uncertainty as error triage, not just an interval

A calibrated interval is useful for a discovery campaign only if it helps locate dangerous or decision-changing errors under a fixed budget. Recent work on flexible uncertainty calibration for machine-learned interatomic potentials reports improved uncertainty–error correlation and detection of high-error configurations, including across several atomistic environments: Ho, Ortner & Wang (2026), https://www.nature.com/articles/s41524-026-02080-3. Treat that as a research motivation, not evidence that any Symthaea uncertainty signal is already calibrated.

Calibration method arms should be explicit, not silently substituted under one generic “uncertainty” label. Where the model/reference stack supports them, compare (1) raw uncertainty proxy, (2) one global calibration/rescaling rule, (3) group- or environment-conditional calibration, and (4) a flexible environment-dependent quantile model. A recent 2026 study reports that flexible, environment-dependent calibration improved uncertainty–error alignment and transfer tests for its MACE-based experiments, including a shifted molecular dataset: https://www.nature.com/articles/s41524-026-02080-3. It is evidence that this is a worthy baseline, not a guarantee that the method transfers to Symthaea or every material family.

Freeze a calibration-budget curve (for example 5, 10, 25, 50 and 100 independent calibration configurations where sample availability permits) and report the number of independent parent systems, groups, atoms and trajectory windows at each point. The article's report of efficient calibration in a particular setting must not be generalized into a universal “25 configurations is enough” rule. Calibration and test windows must be separated by the actual independence unit; many atoms or adjacent frames from one parent system do not constitute many independent material examples.

Pre-register the prediction and reference unit: atom/component, whole structure, trajectory window, or candidate-level decision. Do not compare atom-level scores to structure-level outcomes without a fixed aggregation rule. Evaluate at least:

- **Error-ranking utility:** Spearman rank correlation between uncertainty and absolute reference error, plus precision/recall for predeclared worst-error tails (for example, the top 5% and 10% of held-out force-error windows).
- **Budgeted detection:** recall of severe-error windows at fixed reference-acquisition rates (for example 5%, 10%, and 20% of candidate windows), compared with random, ensemble-disagreement, and uncalibrated uncertainty baselines.
- **Reliability warning lead time:** time or saved integration steps between the first predeclared uncertainty alarm and the first reference-confirmed severe error or structural-integrity event. Report missed events and false alarms; a warning after failure is not an early warning.
- **Conditional calibration:** interval coverage and width by material family, element/local environment, surface/passivation, temperature, and in-domain versus shifted groups. Keep calibration and evaluation groups disjoint.
- **Decision impact:** actual reference calls consumed and the total calls required to meet the same trajectory-reliability and structural-fidelity gates, including failed calls, acquisition overhead, and the cost of false alarms/abstention.

For a first screening experiment, a provisional target is at least 90% recall of the preregistered severe-error tail while acquiring references for no more than 20% of eligible windows, without violating the grouped 90%-interval coverage gate. This is a proposed go/no-go target—not a measured performance figure—and must be stress-tested at lower acquisition rates and under group shift. It must be compared with random sampling and at least one strong uncertainty baseline.

Conformal-style marginal coverage assumptions can be weakened by distribution shift, temporal dependence, and chemical-family leakage. Consequently, report observed grouped coverage and calibration sample counts; never present nominal coverage as a guarantee for new nanocrystal chemistry without supporting assumptions and evidence. If the model uses ensemble spread, HDC similarity, novelty, or free-energy-inspired quantities, evaluate each as a separate candidate signal first. Those quantities are not calibrated probabilities by their names alone.

### D. Exploration fidelity

If the use case includes relaxation, minima hopping or another structural search, static MAE and survival are still insufficient. On a frozen set of starts and an explicit call/time budget, report:

- valid relaxation/search completion rate;
- recovery rate for known reference-supported structures/minima;
- rate of unphysical minima under the chosen chemistry-specific criteria;
- energy-ordering agreement for comparable structures;
- diversity of distinct valid basins found, using a frozen structure-equivalence definition;
- disagreement between candidate discovery/search rate and verified structural fidelity.

Use a source-matched algorithm and compute budget. More structures found is not better if they are duplicates or artifacts of a distorted learned potential.

### E. Total cost and data efficiency

Count each actual reference invocation. A failed, timed-out, interrupted, or non-finite invocation still consumes the declared call budget. A proposal rejected before the external solver starts consumes a proposal attempt, not a reference call. Reconcile both counts.

Report wall time, model inference time, external calculation time, peak RAM/VRAM, hardware, and energy/cost where measurable. Report cost per valid trajectory and number of reference configurations needed to reach the reliability threshold. Keep the cost unit consistent within a comparison.

## 5. Study design and acceptance ladder

### Stage 0 — Eligibility and admission

Resolve source/weight/data license, exact revisions, model training exposure, input artifacts, reference method, group keys, resource limits, and applicability domain. Confirm that the model supports each requested element, charge/spin regime, surface chemistry and boundary condition.

**Gate:** all hard provenance and use-rights conditions resolved, and zero unaccounted inputs. Otherwise stop with a typed blocked/unresolved status; no performance run.

### Stage 1 — Reproduce reference and static baselines

Reproduce the task-matched source baseline on the admitted setup. Verify units, starting structures, training/holdout disjointness, metrics, and failure denominator independently. Record both source-reported and locally measured values distinctly.

**Gate:** the baseline meets a preregistered reproduction tolerance. A failed reproduction blocks policy comparisons until understood.

### Stage 2 — Matched trajectory comparison

Run identical frozen starting conditions and declared replica seeds across eligible models, with identical horizons, integrator settings where method parity makes that scientifically appropriate, and per-arm resource caps. Use a justified reference trajectory/snapshot plan. Do not equate numerical convenience with fair physics: if a model needs a different stable timestep, report it and include the resulting cost/coverage consequences.

Report every attempt, including unstable, failed, and out-of-domain trajectories. Estimate uncertainties at the independent system/family level, not by pretending adjacent frames are independent samples.

**Provisional pilot guardrail:** target at least 95% successful completion to the declared horizon across preregistered replicas, with the one-sided 95% lower bound reported and a separately declared structural-integrity gate. The actual horizon, sample size, catastrophe definition, tolerance and confidence procedure must be fixed per task before unsealing results. This guardrail is not a claim that 95% is sufficient for every scientific application.

### Stage 3 — Active-learning value test

Only after eligible static and dynamic baselines reproduce, compare:

1. seeded random reference-configuration acquisition;
2. the canonical descriptor/diversity selector;
3. the strongest task-matched uncertainty/active-learning baseline that can be fairly reproduced;
4. one Symthaea-specific selection policy, followed by a component ablation.

At each cycle, the selector sees only allowed pre-query features and model estimates. Reveal oracle/reference labels only after selection. The updated model may train only on the permitted revealed partition. Keep the final test families inaccessible to policy/model updating.

Primary outcome: actual reference calls needed to reach the preregistered dynamic-reliability threshold. Secondary outcomes: trajectory survival and structural-fidelity curves, static errors, uncertainty-risk curves, diversity, wall time, total cost and failure rate.

**Provisional go/no-go target:** at least 20% fewer actual reference calls to reach the same dynamic-reliability threshold than the strongest eligible baseline, with group-aware uncertainty excluding no improvement, while satisfying every static and dynamic hard gate. This target must be checked against the size/cost of the admitted benchmark before test outcomes are exposed; change only in a documented pre-scoring protocol revision.

### Stage 4 — Replication and promotion

An independent checker reads the frozen manifest and raw output/trajectory receipts, recomputes validity/failure events and metrics, and verifies group/split isolation. Re-running the selector alone is not independent metric verification.

A passing computational benchmark supports only the tested model, task, chemistry, conditions and horizon. It does not establish experimental agreement, synthesizability, safety, device performance or universal transfer. Those are distinct gates.

### Experimental reality-gap gate (separate from computational qualification)

A further paradigm shift is to qualify predictions against the observable that matters in a real material—not merely against another computational label. *UniFFBench* (published 14 July 2026) evaluates universal machine-learning force fields against experimental reference values across 1,500+ mineral systems and reports that impressive computational benchmark performance did not guarantee the density accuracy needed for practical use. It also reports a disconnect between simulation stability and mechanical-property accuracy: https://doi.org/10.1038/s43588-026-01019-4

This study is not a nanocrystal benchmark and its numerical thresholds must not be transferred to CdSe or other nanosystems. It supports a general evaluation principle: whenever a claim extends beyond computational emulation, add a separate experiment-anchored gate with the exact observable and material-specific conditions.

Before an experimental comparison, preregister:
- the measured observable, units, specimen state, composition/morphology, processing/passivation, temperature/pressure/environment, and measurement protocol;
- sample/data provenance, use rights, measurement uncertainty, instrument/batch lineage, missingness, and whether each label is experimental, derived, or simulated;
- a group-aware holdout by specimen/material family or experimental batch, as applicable, preventing near-duplicate measurements or simulations of the same specimen from leaking across splits;
- the physically appropriate acceptance tolerance based on measurement uncertainty and intended use, not an arbitrary universal nanotechnology error threshold;
- an independent metric reconstruction using saved predictions and measurement records.

Do not substitute a DFT value for experimental validation, infer a measured property from trajectory survival, or call a model “experimentally validated” after fitting the same measurements it is scored against. If no matched experimental dataset exists, mark this gate unavailable and keep any result explicitly computational.

## 6. Mandatory negative controls

- **Trajectory-family leak canary:** deliberately place related frames/replicas across a synthetic split; the independent checker must detect forbidden overlap.
- **Initial-state parity:** mutate one arm's initial coordinates, velocities, seed or thermostat config; the report must mark the comparison non-comparable.
- **Uncertainty permutation:** permute uncertainty ranks while preserving the candidate pool; claimed uncertainty-guided gains should vanish or be explained.
- **Random acquisition:** retain fixed random seeds and the full distribution across seeds.
- **Duplicate-structure injection:** duplicate a frame/candidate under a different row ID; canonical identity/family diagnostics must surface the duplicate.
- **Unsupported-domain control:** provide a chemistry or boundary condition outside the declared model envelope; the model must reject or explicitly abstain rather than silently claim support.
- **Failure-cost control:** insert timeout/non-finite/unstable reference calls and ensure they remain in call and cost denominators.
- **Metric tampering:** alter a summary value while preserving raw output digests; independent recomputation must reject the report.
- **Survival-only bias control:** include failed trajectories in the denominator; a run containing only surviving trajectories is invalid for the primary survival metric.

Any leakage-control failure invalidates the affected comparison independent of apparent score.

## 7. Execution strategy while CI is queued

Useful work can proceed without waiting for Actions, but no unavailable execution is implied:

1. Static-review eligibility, data/model provenance, input contracts and metric semantics.
2. Author schemas/manifests and deterministic synthetic fixtures only where a canonical owner and format do not already exist.
3. Verify schema/fixture syntax independently and check the manifest for required fields and contradictions.
4. Keep benchmark code isolated from default production builds and reuse the existing selector, outcome evaluator, matter epistemics, and evidence-plane owners.
5. Acquire real data/model artifacts only after rights/access approval; measure hashes locally; never infer digests or split counts from provider documentation.
6. Execute formatting, tests and numerical comparisons on an exact pinned head when tooling is available. A queued, skipped, or failed-to-start job is not a pass.

No dataset bytes, model weights, raw trajectories, or external solver outputs are included in this protocol. No model was trained, no trajectory was run, and no scientific performance gain is claimed.



## 8. Distribution shift is a first-class test dimension

**Tracking:** [NANO-012 — worst-group trajectory reliability](https://github.com/Luminous-Dynamics/symthaea/issues/7762).

A random held-out frame split can measure interpolation while hiding failures on a new chemical family, surface, temperature regime or defect class. Test distribution shift explicitly and keep each axis distinct from static error and nominal trajectory survival.

Recent research directly motivates this design: Kreiman & Krishnapriyan analyze distribution shifts for universal ML interatomic potentials (*Digital Discovery*, 2026, https://doi.org/10.1039/D5DD00260E); Tahmasbi et al. benchmark generalization on elemental systems (*npj Computational Materials*, 28 July 2026, https://www.nature.com/articles/s41524-026-02251-2); and a 2026 multicomponent-alloy study compares transferability, finite-temperature robustness and uncertainty behavior (https://doi.org/10.1103/1qc9-ypyb). These support the benchmark questions, not any claim that Symthaea currently passes them.

### Frozen axes

Select physically meaningful axes only after the system/reference data are admitted:
- chemistry/composition family;
- surface/facet/coordination and defects;
- temperature or energy regime;
- strain/pressure and structural phase;
- size, interface, passivation or substrate;
- compounded shift, such as an unseen chemistry plus surface class and temperature.

For each test, freeze a shift manifest with a short physical rationale, exact severity definition, source/reference hashes, group IDs, split membership hash, and split policy. Record zero or more *physical perturbation families* separately from the *mechanism of distribution shift*: feature-support shift, label/property-distribution shift, graph-topology shift, physical-regime shift, chemical-environment shift, and measurement-process shift are orthogonal mechanism labels, not synonyms for chemistry or surface families. A change in model inputs does not by itself prove a change in target labels, and an unchanged label range does not prove input support is in-distribution. Record each axis independently and test compounded axes separately.

The NANO-010 shift manifest records zero or more changed physical perturbation families (so a combined temperature/composition shift keeps both labels) and independently enforces mechanism-axis cardinality: no axes for the in-distribution reference group, exactly one axis for a single-factor shift, and at least two axes for a compounded shift. A manifest is still only a frozen taxonomy and split receipt; domain experts must justify that the selected axes and severity correspond to a plausible physical/deployment shift.

Separate train, calibration/validation and untouched test lockbox. An independence key (such as material family, parent structure, substrate, wafer or trajectory origin) must never occur across different splits. Repeated measurements and replicas stay grouped. A new test split cannot share near-duplicate parent structures with training merely because frame IDs differ.

### Required reporting

For each shift family and severity, report static energy/force/stress error distributions; trajectory survival and structural fidelity at declared horizons; empirical coverage and interval width; selective risk/abstention; severe-error recall at fixed acquisition rates; false alarms and warning lead time; and reference cost to reach a fixed reliability target. Show worst-group results and curves against both shift severity and total budget. A pooled mean alone is not an acceptance result.

### Proposed acceptance targets (not results)

- Nominal 90% intervals target 85–95% empirical coverage on independent groups, with interval width and worst-group coverage shown.
- Severe-error triage targets at least 90% recall while reviewing/querying at most 20% of windows; publish false-alarm rate and warning lead time.
- Active adaptation targets at least 20% lower actual reference cost to reach a preregistered trajectory-reliability threshold than the strongest eligible non-adaptive comparator, without increasing physical constraint violations.
- Replicate the finding on at least one separate held-out material/system family before making cross-system generalization claims. Power the group/replicate count for the preregistered effect rather than choosing a sample count after seeing results.

### Mandatory negative controls

The auditor/evaluator must reject: a single independence key shared between train and test; sample IDs duplicated across splits; missing or malformed source/shift digests; shift-severity labels shuffled after outcomes are viewed; and a policy that improves pooled metrics but materially degrades the predeclared worst group. Keep the official benchmark metrics intact; report shift metrics as separately named extensions.

## 9. Prospective rescue-controller benchmark: maximize trustworthy simulation, not snapshot accuracy

**Tracking:** [NANO-013 — prospective trajectory rescue and safe simulation time](https://github.com/Luminous-Dynamics/symthaea/issues/7777).

This extension makes the benchmark an **online decision problem**. At each eligible decision window, a frozen policy decides whether the current simulation may continue, should run more conservatively, should abstain/stop, or should spend a declared reference calculation before resuming. The objective is not the number of frames emitted or a lower static MAE; it is the amount of *reference-validated useful trajectory* produced under a fixed all-in budget, while keeping every material group within physical-integrity limits.

The motivation is now supported by a specific counterexample class: a September 2026 preprint benchmarking ML interatomic potentials on a passivated CdSe nanocrystal reports that force-MAE ranking did not predict 1 ns dynamical robustness in the tested setting, and uncertainty-guided augmentation stabilized one model after 34 extra configurations while other models did not stabilize within their stated augmentation budgets. This is a single-system preprint result, not an independent replication or a Symthaea result: https://arxiv.org/abs/2609.15299. Recent work on flexible uncertainty calibration also motivates evaluating uncertainty on shifted atomistic distributions rather than assuming that model spread is already a calibrated probability: https://www.nature.com/articles/s41524-026-02080-3.

### Preregistered action space

Each policy may select only from actions declared before lockbox evaluation:

1. **Continue** at the declared integration settings.
2. **Conservative continuation**: checkpoint and/or reduce the timestep under a frozen rule. Any restart/replay cost must be counted.
3. **Abstain/stop**: end this trajectory attempt and report the lost opportunity. Policy stops are not automatically physical failures, but neither are they allowed to disappear from the denominator.
4. **Request reference**: evaluate a predeclared snapshot/window using the admitted task-matched reference method, then continue only according to a frozen update/resume rule. Count successful, failed, retried, and rejected reference calls.

The policy receives only information that would have existed at that time: prior coordinates/observables, its own predictions and uncertainty, past reference returns, elapsed cost, and frozen metadata. It may not read future frames, held-out labels, post hoc failure annotations, test-group identity that would not be operationally available, or an evaluation-only oracle flag. The final test lockbox cannot tune trigger thresholds or retraining schedules.

### Separate intervention queries from blinded evaluation audits

An adaptive policy cannot validate itself by asking for expensive references only when its own score is high. That design leaves most accepted/continued windows unevaluated and can make a selective or overconfident policy appear safer simply because it did not inspect its own blind spots.

Use two explicitly separate lanes:

- **Policy lane:** references requested by the controller to decide whether/how to resume. These requests, failures, retries, cache hits and all associated costs belong to the policy's deployable-budget ledger and may influence future policy actions only under the frozen update rule.
- **Evaluation-audit lane:** a precommitted, policy-blind probability sample of eligible windows/systems, selected independently of the policy's query decisions. These reference labels are hidden from the controller until the tested trajectory/policy run is complete and must not change the arm's decisions. Include continued, stopped/abstained and high-/low-risk portions of the eligible population so that false negatives can be estimated; do not evaluate only queried windows or surviving runs.

Freeze the audit sampling frame, strata, randomization seed, inclusion probabilities, missing-label handling and reference method before lockbox runs. Stratified oversampling of rare severe events is permitted only when selection probabilities are retained and the estimator respects them. Report raw sample counts and group coverage. Use inverse-probability weighting or another preregistered design-based estimate when inclusion probabilities differ; a simple unweighted rate is not justified for a nonuniform audit sample.

The evaluation-audit lane is **not free compute**. Report (1) operational policy cost excluding evaluation-only audits, (2) full study cost including evaluation audits, and (3) the audit/reference budget required to make the reported safety claim. Do not smuggle audit labels or their costs into only one arm. The deployment comparison uses matched policy budgets, while the scientific claim about unqueried windows is supported by the independent audit sample and its uncertainty. If no eligible evaluation-audit sample exists, mark unobserved windows as unverified and block a broad reliability claim rather than imputing success.

This separation prevents query-selection bias and survivorship bias. It does not by itself solve dataset shift, reference-model error, or dependence among windows; those remain separate gates.

### Matched-budget comparison arms

Include at least:

- fixed continuation/termination rules with no adaptive triage;
- random query/review at the same actual reference-call budget;
- uncertainty-only trigger;
- novelty-only trigger using a separately declared descriptor/HDC score;
- calibrated uncertainty + out-of-domain/shift diagnostics + all-in cost-aware decision;
- reference-at-every-eligible-window diagnostic, to establish the expensive reference ceiling rather than to compete as a low-cost policy;
- oracle-event trigger only as a labelled diagnostic upper bound, never as an eligible deployable arm.

A policy combining signals must include ablations so that each component's value is separately measurable. Match the evaluation budget and initial conditions across arms. Freeze model/data exposure, policy, thresholds, checkpoint logic, stopping rule, reference settings, and analysis before unblinding.

### Primary and safety endpoints

Report as separate endpoints rather than collapsing them into a weighted score:

- **Useful simulated time per all-in budget**: estimate the total duration of eligible windows passing the task's preregistered reference-fidelity and physical-integrity gates, divided by the policy-lane cost, using the independent evaluation-audit sampling design to estimate unqueried windows. The numerator must not be inferred solely from the controller's own query outcomes. Report a second ratio using full study cost (including evaluation-audit calls). Count every attempted arm and preserve the initial matched cohort; a merely finite trajectory is not necessarily useful.
- **Fixed-horizon survival**: fraction of independent attempts still eligible and physically credible at each preregistered horizon, with group-aware uncertainty.
- **Time and cause of first failure**: retain numerical divergence, unphysical structure, invalid chemistry/passivation, wrong reference-disagreed transition, solver/runtime failure, and other predeclared causes separately.
- **Intervention value**: additional validated time saved by a warning or conservative action; fraction of true severe events detected before the deadline; warning lead time in windows and physical time; missed-event rate; false alarms per 100 eligible windows; number and cost of unnecessary stops/checkpoints.
- **Reference efficiency**: validated time per successful reference call and per *all attempted* reference calls; also report wall-clock, CPU/GPU time, memory, and energy where measured. Do not combine unlike currencies into a single cost number without an explicit conversion policy.
- **Physical observables**: comparison of only the task-relevant structural, dynamical, thermodynamic, or transition observables against the admitted reference. Good numerical survival is not a substitute for physical fidelity.
- **Worst-group harm**: paired candidate-minus-baseline results for every frozen material/system group, plus the minimum group effect and group-clustered intervals. A pooled win cannot compensate for a hard-gate failure in one group.

### Event accounting and censoring rules

Every started trajectory attempt remains in the ledger. Record the first terminal event and timestamp, including valid horizon completion, physical/numerical failure, policy stop, reference-unavailable stop, infrastructure failure, and administrative censoring. Keep administrative censoring distinct from policy stop and from a verified physical failure. Use a competing-event/time-to-event analysis appropriate to the frozen design; do not estimate survival by deleting trajectories that failed early, and do not count a deliberately stopped trajectory as successful past its stop time. A run that crashes before the first scheduled snapshot is still a failed attempt with cost.

All-in cost must include reference setup and retries, failed/invalid reference jobs, policy inference/selection overhead, acquisition and training, cache semantics, checkpoint storage, rollback/rerun time, and unsuccessful candidate attempts. Report the source of each cost and whether it is directly measured, externally billed, or only an estimate. A hash-bound self-reported ledger proves internal consistency, not external compute billing.

### Provisional promotion targets (targets, not results)

1. **Efficiency:** at least 20% more reference-validated useful simulation time per all-in reference/compute budget than the strongest eligible baseline, with a preregistered group-aware interval supporting improvement.
2. **Early warning:** recall at least 90% of the preregistered severe-event class while querying/reviewing at most 20% of eligible windows. Publish false alarms, lead time, missed events, and subgroup rates alongside recall.
3. **Safety non-inferiority:** the fixed-horizon survival and physical-integrity gates must not worsen beyond a task-specific margin fixed from reference repeatability and domain-owner review before held-out outcomes. Any material group breaching a hard physical gate blocks promotion regardless of the pooled gain.
4. **Calibration under shift:** retain the provisional nominal-90% interval target of 85–95% empirical coverage on independent groups, show interval width, and report conditional/worst-group coverage; no calibration claim is inferred from HDC novelty or ensemble spread alone.
5. **Independent transfer:** reproduce the direction of the efficiency/safety result on at least two independent material/system families, including a preregistered out-of-distribution group, before making cross-family claims.

These are provisional go/no-go targets, not universal physical constants. The required number of independent parent systems/replicas must follow a preflight power/precision plan; thousands of correlated windows from one parent nanocrystal do not substitute for independent material groups.

### Required negative controls

- Shuffle the risk score while preserving its marginal distribution; the resulting policy must not retain apparent event-detection advantage.
- Delay the trigger beyond the event onset; warning lead-time benefit must disappear.
- Compare with a matched-budget random trigger and fixed policy.
- Mutate or omit a charged reference-call ledger entry while updating declared hashes; the independent accounting check must still detect counter/cost inconsistency from the attempt records.
- Include a candidate that improves pooled mean but breaches a frozen worst-group physical gate; it must not be promoted.
- Retain failed and inconclusive attempts and predefine how missing reference labels affect eligibility; do not silently remove the most difficult windows.

This protocol qualifies a controller only after baseline reproduction, artifact/provenance audit, exact-head tests, and task-matched scientific validation. It does not qualify the current Symthaea molecular AIMD implementation for nanocrystal dynamics, and it authorizes no physical experiment.

## 10. Decision record template

Each run review should end with a durable decision:

- **Disposition:** blocked / eligible for baseline reproduction / baseline reproduced / dynamic gate failed / policy comparison eligible / policy value supported / inconclusive / value not supported.
- Exact source, data, model, reference method and configuration identities.
- Independent scientific grouping/split manifest and excluded/ambiguous records.
- Energy/force metrics, trajectory survival and failure-time curves, structural observables, uncertainty coverage, abstention, and resource costs—each with units.
- Negative-control status and independent metric-checker receipt.
- Known limitations, untested chemistry/conditions and the single next permitted action.

Keep null and negative outcomes. The useful output of this program is a reliable answer about when a nanomaterial model can or cannot be trusted over time—not a leaderboard that rewards unstable discovery.

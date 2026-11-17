# ISRU decision-directed campaign policy v0.1

**Status:** standalone Rust reference implementation added; dedicated workflow wired; exact-head execution pending  
**Parent:** Symthaea #12607 — ISRU-DECISION-BO-001  
**Implementation:** `research/asteroid-optical-isru-v0.2/decision_policy.rs`  
**Base system:** B1 coupled optical/material twin in PR #12592  
**Claim boundary:** synthetic finite-hypothesis decision model; not calibrated to a physical rig and not a live controller

## 1. Research objective

The controller should not assume that the next best action is another material exposure. It should choose between a material experiment, optical reference measurement, QCM/witness inspection, calibration, cleaning/requalification, independent assay, or a deliberate hold.

The objective is to reduce the expected loss of the *engineering decision* (for example, whether a process envelope is acceptable), not merely to reduce prediction variance or maximize a proxy yield. The action can be metrology or maintenance when the dominant uncertainty is apparatus health rather than material response.

This extends conventional Bayesian optimization in two directions:

1. The action space includes measurements and interventions on the apparatus as well as experiments on the material.
2. The value function is decision loss under asymmetric engineering consequences, with resource cost and hard evidence gates represented explicitly.

The first implementation is intentionally finite and auditable. It accepts a discrete set of hypotheses, decisions, outcomes and likelihoods. It does not fit a Gaussian process or learn likelihoods from data.

## 2. Decision model

Let (H={h_1,ldots,h_n}) be mutually exclusive campaign hypotheses with prior probabilities (p(h_i)). Examples include clean versus fouled optics, source drift versus true material transition, calibration-valid versus calibration-drifted, or conforming versus nonconforming feedstock.

Let (D(s)) be the terminal decisions available under the current safety/evidence state (s), and let (L(d,h)) be the predeclared loss for decision (d) when hypothesis (h) is true.

Current Bayes risk is:

[
R(p,s)=min_{din D(s)}sum_h p(h)L(d,h)
]

For candidate action (a), observations (y) have likelihood (P(ymid h,a)). After observing (y), the posterior is:

[
p(hmid y,a)=
rac{P(ymid h,a)p(h)}
{sum_{h'}P(ymid h',a)p(h')}
]

Expected value of sample information (EVSI) is:

[
operatorname{EVSI}(a)=R(p,s)-
sum_y P(ymid a)R(p(cdotmid y,a),s)
]

Resource cost is represented as a transparent, preregistered weighted sum:

[
C(a)=w_E E_a+w_t t_a+w_m m_a+w_x x_a
]

where the raw receipt preserves energy, elapsed time, consumables and experiment-equivalent cost separately. The first controller uses:

[
V(a)=rac{operatorname{EVSI}(a)-lambda C(a)}{C(a)}
]

It selects only an action with positive net value that clears the declared minimum thresholds. If none qualifies, it returns no selected action; the caller should hold or pause rather than force an experiment.

The cost weights and (lambda) are policy assumptions, not natural constants. They must be preregistered and sensitivity-tested. Report raw resource vectors and Pareto results alongside the scalar ranking so a weighting choice cannot hide a trade-off.

## 3. Hard gates are not soft penalties

The safety/evidence gate is evaluated before value optimization. A high EVSI cannot make an unsafe or unauthorized action eligible.

Current gate dimensions include sample identity, calibration validity, optical-path qualification, thermal-envelope safety, source-reference validity, QCM/witness-chain validity, cleaning-protocol qualification, calibration-standard validity, independent-assay chain validity, and independent-assay pass.

Examples:

- A material exposure is blocked when the process optic is unqualified, thermal envelope is unsafe, source reference is invalid, the instrument is uncalibrated or sample identity is uncertain.
- QCM/witness inspection can remain eligible when material exposure is blocked, provided its own calibration and evidence chain are valid.
- Cleaning/requalification can be eligible while the optical path is unqualified, but only under a qualified cleaning protocol.
- Product acceptance is not an available terminal decision until sample identity, independent assay chain and assay-pass prerequisites are satisfied.
- If no terminal decision is evidence-eligible, evaluation fails closed.

The gate state must come from external, versioned evidence. This module does not authenticate signatures, custody, calibration certificates or safety authority. Nor can a simulation receipt satisfy a physical gate.

## 4. Implementation boundary

The standalone Rust module implements:

- normalized priors and per-hypothesis outcome likelihood validation;
- explicit, asymmetric terminal-decision loss tables;
- outcome-conditioned Bayesian posterior updates;
- expected posterior decision risk and EVSI;
- cost scalarization with the original four resource dimensions retained;
- explicit action and terminal-decision gate evaluation;
- fail-closed behavior for unavailable decisions, malformed likelihoods and invalid costs;
- stable action ordering and deterministic tie-breaking;
- a no-action result when the minimum value threshold is not met;
- a `simulation_only` result marker.

The tests use a synthetic two-hypothesis example. A relatively cheap, informative QCM/witness diagnostic should outrank an expensive material run. A missing thermal-safety gate blocks the material run without blocking the eligible diagnostic. Product acceptance remains unavailable until the independent-assay gate is satisfied.

The likelihood tables in these tests are fixtures, not empirical QCM sensitivity/specificity, and the example cost weights are not prices or measured operating costs. Production use requires empirical likelihood calibration, uncertainty intervals, explicit missing/failure outcomes and external authorization.

## 5. What this adds beyond current methods

### Multi-stage Bayesian optimization

Torresi and Friederich's 2026 *Digital Discovery* paper introduces multi-stage Bayesian optimization (MSBO), allowing flexible sampling across process stages and decisions based on intermediate proxy measurements. It is a strong baseline because a static, end-of-workflow objective can miss informative intermediate signals.

Our additional question is whether the system should change the *kind* of action altogether: for example, pause sample processing and inspect the optic when the observed response could be explained by contamination. MSBO must be included as a baseline, not dismissed or replaced with an in-house approximation.

Reference: https://doi.org/10.1039/D5DD00572H

### Phase-boundary active learning

BALPI (2026) compares MES, BALD, SMOCU and extended-straddle methods for CALPHAD-oracle phase maps. With 16 initial points and 80 iterations, it reports Bayesian errors of 0.038 ± 0.010 for e-straddle(c), 0.041 ± 0.003 for SMOCU and 0.045 ± 0.007 for MES. These are results on its stated phase-diagram benchmark, not physical ISRU performance.

We should reproduce e-straddle(c), SMOCU and MES on phase/regime boundary sub-tasks, then test whether decision-directed EVSI adds value when apparatus state can confound the apparent boundary.

Reference: https://doi.org/10.1039/D5DD00459D

### Instrument-cost-aware optimization

A 2026 real-world nanoindentation study combines heteroskedastic GP modeling with the time costs of drift stabilization, movement and reconfiguration; it reports nearly 30-fold better mapping efficiency than grid indentation for its thin-film workflow. That result is not transferable as an ISRU target, but its method makes one point decisive: instrument overhead must be in the objective.

We must charge optics inspection, reference readings, QCM/witness inspection, calibration, cleaning, requalification, retries and downtime to the same campaign budget as material exposures.

Reference: https://doi.org/10.1039/D5DD00518C

### Physics-informed BO and complete-lab evaluation

The 2026 review of BO in self-driving materials laboratories highlights failed/missing experiments, drift, mixed variables, multiple constraints, variable cost/fidelity, transfer, asynchronous execution and physics-informed priors. Autonomous Science Bench 0.1 (October 2026) scores the whole model-agent-tool-harness system on real laboratory workflows rather than only an isolated acquisition formula.

These motivate a layered comparison: acquisition policy, orchestration/agent, evidence adapter and backend should all be scored, while keeping replay, simulation and physical trials separately labeled.

References:
- https://arxiv.org/abs/2608.26016
- https://asb.open-ssi.org/

## 6. Required benchmark tournament

Every method must see the same candidate pools, hidden-world families, initial observations, action availability, resource budget and paired random seeds. Pin reference versions/commits and licenses. Use matched action semantics; a baseline that cannot inspect/clean/calibrate must be labeled as material-only rather than presented as a full-policy competitor.

### Baselines

1. Random feasible action and LHS/Sobol material-only design.
2. GP-EI, GP-PI, GP-UCB and constrained GP-BO.
3. RF and TPE for categorical/mixed parameter spaces.
4. MES and BALD for information search.
5. BALPI e-straddle(c), e-straddle(b) and SMOCU for regime boundaries.
6. Cost-aware heteroskedastic GP-BO with measured action and apparatus overheads.
7. Multi-stage BO (MSBO) with intermediate proxy observations.
8. Batch and asynchronous BO variants, with their fantasy strategy pinned.
9. Greedy one-step EIG/EVSI over material experiments only.
10. The current joint-action EVSI controller.
11. An oracle diagnostic/action selector, used only as an unattainable upper bound.

Required ablations: no apparatus state; no hard gates; uniform versus heterogeneous costs; no drift/noise model; material-only actions; no independent assay; and no cleaning/requalification action.

### Hidden-world factors

- clean stationary optics; source drift with clean optics; wavelength-dependent and spatially nonuniform fouling;
- QCM/witness/process-optic transfer mismatch;
- calibration drift, expiry and correlated instrument common-cause faults;
- wet-purge gas creating a false water signal;
- sharp, gradual and disconnected material-regime boundaries;
- feedstock batch shift and cross-material transfer failure;
- cleaning with full recovery, partial recovery, roughness damage or misleading witness results;
- missing, delayed, failed and censored measurements;
- asynchronous stages and partial-cost failures;
- holdout material families and an unseen optical geometry;
- adversarial proxy leakage, stale evidence and forbidden-action negative controls.

At least one material family and one instrument/path configuration should be withheld from policy development. The evaluator retains hidden truths and labels.

## 7. Metrics and preregistered targets

**Primary outcome:** total campaign resource cost to the first correct, evidence-qualified terminal decision. Report energy, elapsed time, consumables and exposure count separately, with paired confidence intervals.

**Safety/evidence:** false product-acceptance rate on negative controls; gate violations; stale-calibration actions; invalid custody/assay acceptance; false optical requalification; holdout leakage; deterministic-replay mismatch. The engineering target is zero observed violations in the sealed negative-control suite. This is not a proof of zero real-world failure probability; report binomial confidence bounds and the number of independent negative controls.

**Scientific quality:** regime-boundary F1/IoU and Hausdorff distance where appropriate; posterior Brier/log score; nominal 95% interval coverage on held-out families; decision regret; ranking of independently assayed recovered product.

**Apparatus diagnosis:** source-drift rejection, fouling-detection delay, false material-transition attribution, QCM/witness transfer error, missed-fouling rate, unnecessary cleaning rate, cleaning recovery fraction and requalification time.

**Operations:** total cost to qualified decision, accepted product per end-to-end campaign energy/time, inspection and maintenance overhead, rig utilization, queue time, retries and salvage fraction.

Prospective targets, not achievements:

- zero observed false accepts and zero gate violations in all sealed hard-negative controls;
- at least 1.5× lower median total campaign cost than the strongest eligible full-action baseline, with paired 95% confidence interval for the improvement above 1.0;
- at least 50% lower false attribution of apparatus degradation as a material-regime transition than a material-only policy, without degrading held-out material ranking;
- non-inferior held-out boundary accuracy and 95% predictive-interval coverage within a preregistered tolerance;
- no false requalification pass in scenarios with incomplete cleaning, surface damage, stale measurement or missing channel.

Do not declare success from a single average. Publish per-world-family results, seed-level results, failures/timeouts, confidence intervals, Pareto fronts and ablations. Determine trial count by a power analysis; a small pilot is for debugging, not a leaderboard claim.

## 8. Industry and physical reference comparison

| Reference | Demonstrated result | Correct role in comparison |
|---|---|---|
| Nectar A/B optical mining, Acta Astronautica 235 (2025) | Optical mining tests on carbonaceous chondrite simulants; beam irradiance, mineral morphology and brittle–ductile transition affected excavation; organic-vapour deposition hampered excavation/water production | Direct physical failure-mode anchor for our optical-mining model. Do not treat the reported rates as universal or as a commercial production benchmark. |
| NASA/Lunar Resources LR-1 MRE, ICES 2026 | 25 kg lunar simulant processed under vacuum; 0.07 kg O₂/h average reported over a 9 h sustained molten-regolith electrolysis campaign | A sustained ISRU throughput and monitoring reference for a different process family; not directly comparable to asteroid water release/capture. |
| NASA CaRD solar-carbothermal integrated prototype, 2026 | Integrated concentrator, optical splash plate/shutter, reactor, fluid, gas analysis and control; NASA reports CO detection confirming solar-carbothermal reaction. The project's stated system metric is grams O₂ per kWh reaching the concentrator primary mirror. | Strong systems-integration, energy-boundary and optical-protection comparator. CO detection is not a quantified commercial oxygen production rate. |

Sources:
- Nectar optical-mining study: https://doi.org/10.1016/j.actaastro.2025.04.033
- NASA LR-1 MRE, ICES 2026: https://ntrs.nasa.gov/citations/20260006167
- NASA CaRD, AIAA ASCEND 2026: https://ntrs.nasa.gov/citations/20260002864
- NASA CaRD energy metric presentation: https://ntrs.nasa.gov/citations/20260003603

The competitive claim must not be "we produce more water/oxygen" until measured. Our software claim to earn is: fewer total resources to a correct qualified decision; fewer false attributions; earlier contamination detection; reproducible evidence; and transfer across new batches and apparatus configurations.

## 9. Limitations and next steps

- v0.1 is a finite discrete EVSI calculation, not a fitted GP, deep GP, POMDP or full campaign planner.
- One-step EVSI can be myopic when cleaning changes the system, experiments consume material, or actions unlock later actions. The next algorithmic step is limited-horizon sequential decision value with resource-constrained lookahead, compared against MSBO and a small exact oracle.
- Likelihoods are caller-supplied and must be learned/calibrated from blinded replications. Correlated noise, heteroskedasticity, missing outcomes and uncertainty in the likelihood model need explicit treatment.
- Scalarized cost weights can change rankings; report raw resource vectors and Pareto fronts.
- No physical authority, instrument interlock, chain-of-custody, calibration signature or assay certificate is created by this module.
- The current code tests validate software semantics only. Exact-head workflow status must be inspected independently; skipped or queued jobs are not passes.

Next physical-model increment: replace the lumped thermal proxy with a conservative 1-D finite-volume heat-conduction column, then fit/validate it against instrumented coupons before adding higher-dimensional mechanics or claiming predictive fidelity.


## 10. Action-conditioned state transitions — 17 November 2026

The first observation-only EVSI formulation was too narrow for two important campaign actions:

- **Cleaning/requalification** changes the physical apparatus state. It is not merely a measurement about whether the optic was fouled; a successful cleaning process can move the optic into a recovered state, while incomplete cleaning or surface damage may leave it unqualified.
- **Independent assay** can change evidence eligibility. A successful independent assay can unlock a terminal accept-product decision; a failed assay must not.

The executable policy now factorizes an action outcome as:

[
P(y mid h,a);P(h' mid h,y,a)
]

where (h) is the current hypothesis, (y) the observed outcome and (h') the post-action hypothesis. The outcome also carries a constrained set of gate updates. The next posterior is:

[
p(h' mid y,a)=
rac{sum_h p(h)P(ymid h,a)P(h'mid h,y,a)}
{sum_h p(h)P(ymid h,a)}
]

The expected next decision risk is calculated against terminal decisions eligible under the outcome-specific evidence state. For identity transitions and no gate updates, this reduces to ordinary EVSI. For cleaning, the transition can represent physical recovery. For assay, the outcome can set only the independent-assay-pass gate.

### Gate-update authority boundary

Outcome updates are restricted by action type:
- optical reference may update only the reference-channel validity gate;
- QCM/witness inspection may update only its evidence-chain validity gate;
- calibration may update only instrument calibration validity;
- cleaning/requalification may update only optical-path qualification;
- independent assay may update only independent-assay pass;
- material experiment cannot mint an evidence gate.

Sample identity, thermal safety, cleaning-protocol qualification, calibration-standard validity and independent-assay-chain validity are protected from these updates. They must originate from their appropriate external evidence/authority paths. A diagnostic result cannot declare a machine thermally safe.

### New regression cases

The standalone policy suite now additionally checks:
- an assay-pass outcome can unlock the accept-product decision, while a failed assay cannot;
- cleaning can both transform the fouled hypothesis into a recovered state and update optical qualification;
- hard thermal safety cannot be minted from an optical-reference observation;
- an action cannot update a gate outside its evidence scope;
- transition matrices must normalize and preserve probability mass;
- measurement-only identity transitions still provide ordinary EVSI behavior.

These are still synthetic finite-state fixtures. The model does not prove that a physical cleaning protocol recovers an optic or that an assay result is authentic; it computes decision value only from supplied, independently qualified likelihoods and evidence updates.

### Next algorithmic research

The action-conditioned one-step controller remains myopic. The next step is a finite-horizon campaign planner over resource-constrained action sequences, using the transition-aware controller as the one-step baseline. Compare this against MSBO, batch/asynchronous BO, and a small exact dynamic-programming oracle on finite hidden worlds. To avoid gaming, include material depletion, irreversible sample consumption, calibration expiry, delayed jobs, cleaning downtime, assay cost and failure/abort transitions.

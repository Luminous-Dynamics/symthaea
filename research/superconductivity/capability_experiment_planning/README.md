# Capability-Gated Superconductivity Experiment Planning

**Contract:** SC-CAPABILITY-PLAN-0.1  
**Status:** exact two-parameter synthetic planning oracle and myopic baseline; no real experiment is authorized  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12819

## Why this is distinct

Cost-aware Bayesian optimization already selects measurements using predictive value and cost. The missing challenge is that some actions create capabilities rather than directly returning scientific information.

Examples in a real evidence programme could include instrument qualification, reference calibration, sample preparation, or setting up a data-reduction pipeline. Such an action may have zero immediate information gain, yet unlock a valuable follow-up measurement. A myopic information-gain-per-cost policy can therefore stop at a locally attractive plan and leave the target prediction underidentified.

This benchmark connects the specimen-linked inverse model (#12811) to experiment selection and the synthesis/process work (#12671). It is distinct from compute allocation across DFT/ML funnel stages (#12644).

Relevant research baselines:
- Alvi et al. (2026), *Deep Gaussian process-based cost-aware batch Bayesian optimization for complex materials design campaigns*, npj Computational Materials 12, 105: https://www.nature.com/articles/s41524-026-01981-7
- Hassoon & Dredze (2026 preprint), *Capability-Gated Planning: Cost-to-Goal Discovery and the Limits of Myopic Experiment Selection*: https://arxiv.org/abs/2608.05085
- Jakeman et al. (2026 preprint), *Risk-Aware Goal-Oriented Bayesian Optimal Experimental Design*: https://arxiv.org/abs/2609.08023

These motivate the testbed; they are not evidence that a superconductivity-specific decision policy has already been validated.

## Initial mathematical testbed

The first synthetic version uses two model parameters, a Gaussian prior represented by a precision matrix \(P_0\), and a target prediction direction \(g\). Each measurement action contributes a positive-semidefinite 2×2 Fisher-information matrix \(I_a\). For a selected plan \(S\):

\[
P(S)=P_0+\sum_{a\in S} I_a,
\qquad
V_g(S)=g^T P(S)^{-1}g.
\]

The objective is target predictive variance reduction:

\[
\Delta V_g(S)=V_g(\varnothing)-V_g(S).
\]

This is deliberately a small conjugate-Gaussian benchmark. It is not a general posterior solver and does not claim that Fisher information is the correct likelihood for every physical measurement.

Each action has a cost, required capabilities, capabilities provided, an information contribution, and a hard eligibility flag. A feasible plan must fit within budget, contain only eligible actions, and admit an ordering in which every action's prerequisites are available before that action runs.

Capability actions are constrained to contribute zero direct measurement information. They earn value only by unlocking downstream actions.

## Synthetic adversarial example

Budget: 5 resource units. Prior precision: identity. Goal direction: equal weight on amplitude and phase-stiffness parameters.

Available actions:
- amplitude-sensitive map: cost 2, immediately measures amplitude;
- calorimetry workflow qualification: cost 2, unlocks heat-capacity measurement;
- heat-capacity scan: cost 1, measures the second parameter;
- low-cost transport proxy: cost 1, partially informs both parameters;
- vortex-imaging qualification: cost 3, unlocks vortex mapping;
- vortex map: cost 2;
- unreviewed high-field test: deliberately ineligible due to missing independent safety/facility qualification.

The exact feasible-plan oracle should choose amplitude mapping + calorimetry qualification + heat-capacity scan (total cost 5). A myopic marginal variance-reduction-per-cost policy ignores the zero-immediate-information qualification action and instead selects cheap transport plus amplitude mapping. The benchmark measures the lost target variance reduction.

This is an adversarial synthetic example, not a recommendation to perform or prioritize any actual experiment.

## Implemented planner

`sc_capability_plan_audit`:
- validates a JSON Schema 2020-12 instance;
- checks the prior precision matrix is symmetric positive definite;
- checks each action information matrix is symmetric positive semidefinite;
- rejects duplicate action IDs and invalid capability declarations;
- rejects any capability-only action that claims direct information;
- excludes ineligible actions and records their reason;
- enumerates all action subsets for up to 20 actions;
- checks cost budget and capability closure;
- constructs a feasible ordered sequence for every selected subset;
- computes the exact small-instance oracle maximizing target variance reduction;
- computes a myopic marginal target-variance-reduction-per-cost baseline;
- reports oracle regret / efficiency and exact input hashes.

The exact oracle is exponential in action count and is intended for small benchmark instances. Larger campaigns need a dynamic-programming, branch-and-bound or stochastic-planning implementation and must be compared with this oracle on tractable subsets.

## Hard constraints and limitations

- An ineligible action is never selected, regardless of its information matrix. The synthetic high-field action demonstrates this non-compensable gate.
- Capability requirements must be available initially or provided by an earlier selected action.
- The first objective assumes additive independent Fisher information. It does not yet model shared calibration errors, common-mode noise, correlated observations, stochastic capability failures, destructive sample consumption or instrument queueing.
- Cost is currently a single dimensionless integer resource. Time, instrument-hours, specimen mass/volume and budgeted failure probability are later constraints.
- No actual measurement action is authorized by this benchmark.

## Benchmark ladder

1. Exact enumeration on small action graphs and comparison to greedy baselines.
2. Correlated measurement noise / shared calibration nuisance parameters.
3. Stochastic action outcomes and capability failure probabilities.
4. Multi-resource constraints: cost, instrument time, specimen consumption and transfer logistics.
5. Goal-oriented risk objectives for rare/high-consequence failure modes.
6. Receding-horizon plans with stop / abstain and model-discrepancy detection.
7. Replay on held-out historical measurement campaigns, then prospective evaluation only after institutional review.

## Proposed targets (research targets, not achievements)

- Exact oracle recovered on every small enumerated instance.
- Capability-gated cases demonstrate measurable regret for immediate-gain-per-cost baselines.
- Lower oracle regret than greedy and cost-aware baselines on a frozen benchmark suite.
- Robustness to prior misspecification, correlated errors, capability failures and cost uncertainty.
- Correct exclusion of all ineligible actions, even when their nominal information is arbitrarily large.
- Calibrated target-prediction uncertainty on held-out synthetic ground truth.
- Report plan quality, resource use, runtime, action count, unused budget and failure rates.

## Claim ceiling

This is a synthetic decision-planning benchmark. It does not prescribe physical experiments, replace institutional safety review, authorize synthesis or establish any superconducting material or property.

## v0.2 research track: mechanism / model discrimination

The new `SC-MODEL-DISCRIMINATION-0.1` synthetic contract tests a different question from v0.1. Rather than asking only which action reduces uncertainty *inside one assumed model*, it asks which feasible plan best distinguishes several competing explanations, while preserving an explicit `none_of_above` hypothesis.

The Rust binary `sc_model_discrimination_audit`:
- validates the model prior, requires exactly one `none_of_above` model, and requires every observation action to define a normalized likelihood for every model;
- rejects capability actions that claim to directly observe an outcome;
- computes exact expected mutual information (I(M;Y_S \\mid D)) in nats by enumerating all binary joint outcomes for the selected observation set;
- enumerates feasible action subsets under the budget and capability-order constraints;
- compares the exact oracle with a myopic marginal information-gain-per-cost policy;
- emits SHA-256-bound schema/fixture receipts and negative-control results;
- hard-excludes ineligible actions even when their nominal likelihoods appear highly discriminating.

### Synthetic fixture result (derived from illustrative probabilities only)

The fixture uses four hypothetical model classes: uniform electron-phonon, multiband electron-phonon, finite-(Q) pairing, and `none_of_above`. It makes no claim that these likelihoods are physically calibrated.

Under a six-unit budget, the exact oracle selects:
1. thermodynamic-channel qualification;
2. independent bulk thermodynamic cross-check;
3. field-angle (H_{c2}) signature;
4. stiffness-tensor signature.

The fixture's exact expected model information gain is approximately **0.556959 nats**. The myopic policy cannot select the zero-immediate-information qualification step and therefore misses the thermodynamic cross-check; the two direct measurements yield approximately **0.371839 nats**. A deliberately sharp but ineligible high-field action must be excluded. A low-cost resistivity repeat with identical likelihood under every hypothesis adds zero model information.

These values are algebraic results of a synthetic fixture, not physical evidence, instrument recommendations, or a prediction about any real sample.

### Mathematical and benchmark boundary

For discrete model (M), selected binary outcomes (Y_S), prior (p(M\mid D)), and the fixture's explicit conditional-independence assumption:

[
I(M;Y_S\mid D)=H(M\mid D)-\\mathbb{E}_{Y_S\mid D}[H(M\mid D,Y_S)].
]

The first executable version assumes known likelihood tables, independent outcomes conditional on model identity, binary outcome categories, and at most 12 actions for exact subset enumeration. It does **not** yet model shared calibration errors, correlated observations, likelihood uncertainty, sequential outcome-adaptive policies, or posterior predictive model discrepancy beyond the explicit `none_of_above` class.

Next qualification steps are likelihood calibration on held-out synthetic cases, proper scoring rules (Brier/log score), selective-risk and abstention metrics, robust planning under likelihood misspecification, and replay against specimen-linked historical campaigns only after the underlying evidence and rights are admitted. Report parameter precision, model discrimination, cost and qualified-answer probability as a Pareto surface rather than hiding them in one scalar.

No physical experiment is authorized or performed by this contract. The existing v0.1 target-variance planner remains a separate objective and benchmark.

## v0.3 research track: correlated evidence and shared calibration nuisance

The new `SC-CORRELATED-EVIDENCE-0.1` contract addresses a major failure mode in the v0.2 model-discrimination planner: two measurement outcomes may share a calibration drift, specimen inhomogeneity, batch effect or other common-cause state. Multiplying independently nuisance-marginalized likelihoods can misstate their joint evidential value.

The Rust binary `sc_correlated_evidence_audit` implements a finite, exact synthetic model:
- a shared latent nuisance state is drawn once for the selected observation set;
- observations are conditionally independent only after conditioning on both the physical-model hypothesis and that shared state;
- the oracle marginalizes the shared state *after* multiplying the conditional likelihoods;
- a factorized comparator independently marginalizes the nuisance for each action, deliberately showing the approximation that must not be mistaken for shared-state inference;
- the exact feasible-plan search preserves capability dependencies and non-compensable eligibility;
- a reference measurement can carry zero standalone information about mechanism identity while adding information jointly by identifying shared calibration drift.

### Synthetic fixture result (illustrative likelihoods only)

The fixture contains two calibration states (nominal and shared positive drift), four competing hypotheses including `none_of_above`, and eight actions (seven eligible plus one ineligible). The exact budget-9 oracle selects both thermodynamic repeats, the shared-drift reference, field-angle (H_{c2}), and stiffness-tensor observations, together with the two capability-qualification actions. Its exact expected model information is approximately **0.424973 nats**. The myopic feasible policy cannot unlock either capability chain and stops with only the immediately available field-angle and stiffness observations.

The reference probe alone has **zero** model information because its outcome distribution is identical under every hypothesis. Jointly with a thermodynamic observation it increases model information from approximately 0.070706 to 0.097327 nats by revealing the nuisance state. The independent-per-action nuisance approximation gives a different value from the shared-state calculation on repeated observations (difference exceeds 0.01 nats for the duplicated thermodynamic pair).

These are algebraic results from a synthetic fixture, not calibrated physical likelihoods or measured properties.

### Formal model and limitations

For model (M), one shared nuisance state (C), selected observations (Y_1,\ldots,Y_k), and conditional likelihoods:

[
p(y_1,\ldots,y_k\mid M)=
\sum_c p(c)\prod_{i=1}^{k}p(y_i\mid M,c).
]

The information objective is (I(M;Y_1,\ldots,Y_k\mid D)), evaluated by exact binary-outcome enumeration. The incorrect factorized approximation instead uses (prod_i \sum_c p(c)p(y_i\mid M,c)), which implicitly redraws the same calibration state independently for each action.

The fixture assumes one discrete nuisance state shared by all selected observations and conditional independence given that state and the model. It does not yet model multiple nuisance groups, continuous random effects, partially shared specimen/batch factors, unknown nuisance priors, posterior likelihood uncertainty, adaptive sequential decision trees, or causal dependence among measurement outcomes. Those are the next research requirements before any historical or physical campaign use.

The workflow now tests and lints all three planning binaries and produces a separate correlated-evidence receipt with exact input hashes. No physical experiment is authorized or performed.

## Industry comparison: baseline ladder for the planner

A benchmark that compares only one greedy policy with a small exact oracle is insufficient to establish superiority over modern active learning. The next frozen suite should compare the same candidate pool, cost model, capability graph and evaluation seeds against:

1. random feasible acquisition;
2. greedy target-variance reduction per cost (v0.1);
3. expected improvement / upper-confidence-bound policies where a candidate objective is defined;
4. cost-aware batch Bayesian optimization, including the multi-property and heterotopic-query style used in Alvi et al. (2026, https://www.nature.com/articles/s41524-026-01981-7);
5. D-optimal / log-determinant design and goal-oriented risk-aware optimal experimental design;
6. training-free LLM active learning as a *cross-domain method comparator*, not a superconductivity baseline. Wang et al. (20 May 2026) report over 70% fewer experiments to reach top-performing candidates across four materials datasets (https://www.nature.com/articles/s41524-026-02136-4); this result must not be attributed to superconductors;
7. myopic model-information-gain-per-cost (v0.2);
8. shared-nuisance exact planning and its factorized-likelihood ablation (v0.3);
9. capability-aware receding-horizon planning with an explicit stop/abstain option;
10. exact feasible-plan oracle on bounded instances.

Report paired candidate trajectories and confidence intervals over fixed seeds. Metrics should include oracle regret, target-risk reduction, model Brier/log score, cost-to-qualified-answer, probability of obtaining an answer within budget, false-positive experiment burden, idle capability/resource costs and validated discovery yield. A model with higher information gain but worse downstream decision quality is not necessarily the better experiment planner.

Lee et al.'s 2026 *Self-driving laboratory 2.0* review identifies interoperability, collaboration, generalization, orchestration, safety and creativity as system-level requirements (https://doi.org/10.1039/D5MH01984B). The 2026 critical examination of materials active-learning workflows warns that implicit design choices and hidden biases can dominate reported efficiency (https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00081a). These motivate frozen splits, cost transparency, fail-closed capability admission, robust calibration and exact replay—not leaderboard-only evaluation.

## What v0.3 adds, and what it does not

`sc_correlated_evidence_audit` adds a discrete shared-calibration latent state. It handles one common cause shared across the selected action set, while retaining the v0.2 conditional-independence assumption *given* that state and model. It is not yet a full hierarchical Bayesian model of all sample, batch, instrument and operator effects. Next steps are multiple partially shared nuisance groups, continuous random effects, unknown nuisance-prior inference, likelihood calibration, and outcome-adaptive decision trees. Do not claim real-world calibration robustness until those are validated on source/specimen-linked observations.

## v0.4 research track: expected value of sample information for downstream decisions

The new `SC-DECISION-VALUE-0.1` contract asks a more operational question than the v0.2/v0.3 information-gain objectives:

> Which feasible experiment is expected to improve the downstream decision enough to justify its cost?

The Rust auditor `sc_decision_value_audit` includes:
- four model hypotheses, including `none_of_above`;
- a shared discrete calibration nuisance state, drawn once for the full selected observation set;
- explicit candidate decisions and a non-negative loss for every decision/model pair;
- capability-gated experiments, cost budget and non-compensable eligibility;
- exact feasible-subset oracles for expected value of sample information (EVSI) and mutual information;
- greedy marginal-EVSI-per-cost and marginal-information-per-cost baselines;
- separate SHA-256-bound schema/fixture receipts and adversarial tests.

For posterior (p(M\mid y)), decision (d), and declared loss (L(d,M)), Bayes risk is:

[
R(p)=\min_d\sum_M p(M)L(d,M).
]

For selected measurements (Y_S), expected value of sample information is:

[
\operatorname{EVSI}(S)=R(p_0)-\mathbb E_{Y_S}[R(p(M\mid Y_S))].
]

The exact joint outcome probabilities preserve the v0.3 common-cause model:

[
p(y_{1:k}\mid M)=\sum_c p(c)\prod_i p(y_i\mid M,c).
]

This intentionally does **not** reward information that does not change a downstream decision. The current fixture uses a hard budget plus EVSI maximization, with cost as a tie-breaker; it does not yet fold arbitrary resource costs into the loss. A later multi-resource objective should separately model time, sample consumption, queue occupancy, failure risk and financial cost.

### Synthetic fixture result (declared illustrative losses and likelihoods)

With budget 8 and prior-optimal decision `stop_or_defer` (prior expected loss 2.55), the exact EVSI plan selects:
1. qualify the thermodynamic measurement channel;
2. perform the bulk thermodynamic observation;
3. measure the stiffness tensor.

Its cost is **5 units**, expected posterior decision risk is **1.6568125**, and EVSI is **0.8931875** loss units.

The exact mutual-information plan also buys the field-angle (H_{c2}) observation. It costs **7 units** and has greater information gain, but produces the **same** expected downstream decision-risk reduction (0.8931875). On this fixture, the extra measurement is informationally interesting but decision-irrelevant under the declared loss table.

The myopic EVSI-per-cost policy selects only the immediately available stiffness measurement and achieves EVSI **0.5875**. It misses the zero-immediate-value thermodynamic qualification action and follow-up.

These are exact calculations over the synthetic fixture only. The decision losses are illustrative and are not a real project's economic utility, engineering risk, or laboratory cost function.

### Benchmark implication and limits

Compare EVSI planning against (a) mutual-information maximization, (b) target-variance reduction, (c) expected improvement / knowledge gradient where a scalar candidate objective is meaningful, (d) cost-aware batch Bayesian optimization, (e) D-optimal/log-determinant and risk-aware goal-oriented OED, and (f) random feasible / domain-expert policy baselines. Freeze the candidate pool, priors, loss table, costs, capabilities, measurement likelihoods and stopping rules. Evaluate decision regret and realized decision quality, not just information gained.

The v0.4 finite oracle has known discrete likelihoods and losses, one shared nuisance factor, binary observations and at most ten actions. It does not implement posterior decision trees, unknown loss functions, distributionally robust EVSI, multiple correlated nuisance groups, continuous observations or real-money/sample/time accounting. Its synthetic result does not establish superiority over SOTA. The hosted exact-head workflow now tests/lints all four planning binaries and emits a separate decision-value receipt.

### Connection to the prediction and engineering challenges

The PNAS 2026 programmatic roadmap explicitly separates the Prediction Challenge (many predicted conventional superconductors are not experimentally synthesizable) from the Engineering Challenge (the effect of pressure, nanostructuring, light and other control knobs is not yet predictively understood). It argues for moving beyond (T_c) and dynamic stability alone toward predictive thermodynamics/synthesis, while integrating theory and experiment in a continuous loop: https://doi.org/10.1073/pnas.2520324123.

This means future EVSI loss tables should not encode only “correct mechanism” versus “wrong mechanism.” They should represent downstream outcomes such as target-phase realization, phase purity, pressure/temperature envelope, field/current performance, sample survival, manufacturability and cost. A measurement is valuable when it changes what we should do next: continue synthesis, switch precursors/process, test a mechanism, change the control knob, or stop an unpromising route. The current v0.4 fixture only demonstrates the mathematical decision-value distinction; it contains no synthesis probabilities or application-calibrated losses.

For the next benchmark tier, measure:
- expected regret against an oracle with complete synthetic ground truth;
- realized utility after outcome-adaptive decisions, not only one-shot EVSI;
- cost-to-first-qualified-phase and cost-to-application-qualified candidate;
- false-positive synthesis burden, failed-phase frequency and fraction of candidate spend consumed by infeasible routes;
- value of information under different application weights (high-field magnet, cryogenic device, ambient-pressure conductor), with the loss table preregistered before evaluation.

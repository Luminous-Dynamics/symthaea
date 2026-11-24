# Evidence-directed scientific discovery: paradigm shifts and benchmark strategy v0.1

**Status:** research and evaluation design. This document proposes experiments and stretch targets; it reports no Symthaea/Symtropy superiority result.

## Executive thesis

Build a system that optimizes for **independently qualified knowledge per unit of research expenditure**, not for the number of hypotheses, experiments, tool calls, pages, or low residuals it produces.

Machine-readable external comparator snapshot (as of 24 November 2026): [BH_EGB_EXTERNAL_BENCHMARK_SNAPSHOT_2026_11_24.json](BH_EGB_EXTERNAL_BENCHMARK_SNAPSHOT_2026_11_24.json). Each score retains its benchmark family, protocol, sample size, confidence caveats and claim boundary; no cross-benchmark composite score is defined.

The architecture should separate three decisions:

1. **Research-action allocation (Stage A):** Is another symbolic derivation, source replay, counterexample search, BVP correction, precision increase, independent endpoint route, interval calculation, or proof attempt worth purchasing? Reuse Symthaea #8721 and the actual policy adapter #11171.
2. **Numerical design (Stage B):** Given the chosen action, which parameters, mesh/resolution, precision, endpoint route, and observables most efficiently distinguish the surviving explanations? Reuse the experiment-design package and consume Symtropy's response predictions.
3. **Independent qualification:** Did the result survive source replay, independent arithmetic/numerics, hostile controls, interval or proof obligations, and claim-scope review? The proposer and executor cannot self-certify.

A cheap derivation that removes a hypothesis can dominate an expensive BVP. A numerically discriminating BVP can still be a poor purchase if its outcome would not change the next scientific decision. Neither Stage A nor Stage B should be asked to optimize the other's objective.

## Paradigm shifts worth pursuing

### 1. Decision value instead of curiosity or uncertainty reduction

The next computation should be selected by how much it can change the set of defensible scientific decisions.

For evidence prefix E and legal action a, use the existing prospective VoC owner:

\[
\operatorname{VoC}(a\mid E)=
\mathbb{E}\left[V^*(E\cup O_a)\right]-V^*(E)
-C(a)-C_{\mathrm{opportunity}}(a).
\]

This requires care: calibrated outcome probabilities are allowed only when validated. Otherwise the policy must produce a bounded/ranked qualitative forecast or UNRESOLVED, not manufacture a posterior from model disagreement. The action forecast is frozen before execution and cannot be rewritten after seeing its outcome.

The key metric is decision regret and cost to the next independently qualified decision—not entropy reduction by itself.

### 2. Dependence-aware evidence geometry

Treat the primary observable vector and its covariance as one mathematical object. Derived outputs (determinant, singular values, condition number, residual norms, transformed response entries) must be marked as derived, with their dependence preserved.

The current v0.2 selector consumes one **joint covariance of the stacked hypothesis-output vector** in an explicit hypothesis-major/observable-major order. It derives each pairwise difference covariance from the diagonal and cross-hypothesis blocks:

\[
S_{ij}=\Gamma_{ii}+\Gamma_{jj}-\Gamma_{ij}-\Gamma_{ji}.
\]

The whole joint matrix must be PSD, which guarantees that the derived pair covariances arise from one common covariance model. The caller must justify the cross-hypothesis coupling and shared-error semantics; marginal standard deviations alone are not enough. In singular cases, test support membership separately from the pseudoinverse quadratic form. No silent diagonal jitter. A previous pairwise-matrix double-centering completion attempt was withdrawn after a vector-output counterexample showed it could reject valid joint models.

The current v0.2 extension does not yet model covariance uncertainty sets, full observable lineage, or Symtropy's endpoint Gram metrics. Those are separate follow-ons: #13472 owns derived-observable ancestry/propagation, and Symtropy #7131 owns physical response/basis semantics. Work on correlated optimal designs identifies additional assumptions and verification difficulty as a central open issue, not a cosmetic implementation detail (López-Fidalgo & Wong, 2026).

### 3. Proof-carrying experiment design

Every selected design should return a compact, immutable experiment card:

- rival hypotheses and exact model/source identities;
- primary observables, units, chart and basis identity;
- predicted signatures and full uncertainty provenance;
- nuisance parameters and model-discrepancy assumptions;
- the result that would falsify each hypothesis;
- predeclared solver/measurement settings, tolerances, and exclusions;
- the next decision each possible outcome can change;
- stopping, replicate, pivot, abstain, and escalation rules;
- inclusive resource budget and independent scoring authority.

This changes a recommendation from “run this because it scores highly” into an executable, falsifiable scientific commitment.

### 4. Counterexample-first independent audit

For every proposed derivation, response operator, numerical branch, and research conclusion, run a separately controlled challenge step: mutate signs, normalize differently, change basis, perturb assumptions, seek a counterexample, and replay source identities.

The audit agent should receive the frozen record, not the proposer’s hidden reasoning state or evaluator truth. Keep the scientist/researcher able to steer or reject hypotheses. Recent DiscoveryAgent work is relevant because it combines scientist steering with checkpointed audit agents; in its reported study, scientist+audit collaboration found 14× more valid errors than manual scientist review. That is a study-specific result, not a general multiplier or proof that any particular audit architecture will win.

### 5. Stop when the decision is sufficient

Do not continue because the model remains curious, nor stop because a confidence number crosses an arbitrary threshold. Stop, pivot, or abstain when the remaining admissible outcomes cannot change the qualified conclusion enough to justify further cost.

This yields a decision-sufficiency frontier: remaining uncertainty, downstream decision impact, worst-case downside, and cost of resolving the uncertainty. It can identify cases where the scientifically correct next action is no experiment.

### 6. Separate predictive fit from explanation and theorem status

A model that predicts held-out data may still have the wrong mechanism. A small residual may coexist with a missing branch, an unexamined gauge mode, or an unproven endpoint assumption. Report separately:

- predictive error;
- structural/mechanistic recovery;
- source and derivation replay;
- numerical convergence and original-equation residuals;
- interval-enclosed claims;
- existence, uniqueness, and branch-event proof status;
- independent review and false-promotion rate.

No single “discovery score” should hide these distinctions.

### 7. Reuse cognitive architecture only where it has a falsifiable role

- Ramanujan proposes symbolic identities, simplifications, or lemmas. Independent derivation/EUREKA challenges them.
- HDC retrieves structural analogies and prior episode metadata. Exact ASTs, physical quantities, units, and source digests remain authoritative.
- CfC models research-campaign history and action outcomes, not the radial ODE/BVP solver.
- FEP/VoC allocates research actions. The experiment selector chooses numerical settings only after an action is selected.
- Symtropy executes the declared physical/numerical task. Independent qualification remains separate.

An ablation must show that each cognitive component changes prospective action selection or independently qualified outcomes. Architectural presence is not demonstrated utility.

## What the external field measures today

### End-to-end scientific workflow

- **BioBasicsBench (OpenAI, 6 Nov 2026):** a complementary wet-lab evidence-interpretation benchmark, not a full autonomous discovery campaign. It contains 100 routine wet-lab tasks across protein methods, imaging/cytometry, quantification/assays, nucleic acid methods and molecular cloning. Tasks include raw gels/blots, plate images, microscopy and instrument outputs, and reward correct interpretation, quantitative work, control checking and recognizing when evidence is insufficient. OpenAI reports 37 model/reasoning-effort configurations × 100 tasks × five attempts = 18,500 scored answers. At xhigh, GPT-6 Astra reports 66.7% mean reward (95% whole-task bootstrap CI 60.3–73.1), GPT-6.1 Sol 62.0% (55.6–68.4), and GPT-5.6 Sol 43.8% (37.6–50.0). These are provider-reported evaluations of OpenAI models, not an independent external leaderboard; the intervals for Astra and 6.1 Sol overlap, so the 4.7-point top gap is not evidence of a clear winner. Add it as a separate **evidence-interpretation / uncertainty-recognition** scorecard that complements AS-Bench's sequential experimental design. It can test Symthaea's ability to audit raw observations and refuse unsupported conclusions, but it does not test multi-round experiment allocation, held-out design ranking or EGB numerics.
  https://openai.com/index/measuring-how-ai-agents-handle-wet-lab-science/
- **BIABench (2026):** the open, reproducible counterpart for end-to-end raw bioimage analysis. Its v1.0 dataset has 16 tasks reconstructed from published studies, spanning 11 subtasks/modalities from H&E histology to single-molecule localization microscopy. Every task includes raw inputs and source-grounded expected outputs; evaluation separates an **outcome score** (field-standard comparison to ground truth) from a **process score** (analysis-method selection and QC rubric). Its paper reports no evaluated agent above 0.19 on some 3D/time-lapse tasks and substantial run-to-run unreliability. The v1.0 dataset/repo is public (CC BY 4.0 for BIABench contributions, while each source image dataset retains its own license); evaluation archives contain a canary and are explicitly evaluation-only—never train/fine-tune on them or re-host them. The paper's reproducibility settings are dataset v1.0, basic instruction, three runs per agent/task, and the stated process judge. It tests evidence interpretation, segmentation, reproducibility and abstention, not sequential experiment selection.
  https://arxiv.org/abs/2609.34274
  https://github.com/BIABench/BIABench
  https://huggingface.co/datasets/BIABench/BIABench
- **LABBench2 (2026):** a complementary public biology-research task suite with nearly 1,900 tasks and a public evaluation harness. It is useful for literature/data interpretation, experimental reasoning and scientific knowledge subskills; it is not a full autonomous experimental campaign or raw-image analysis benchmark. Pin the exact dataset and harness revision, retain subtask scores and use it as a separate scorecard.
  https://arxiv.org/abs/2604.09554
  https://github.com/EdisonScientific/labbench2
  https://huggingface.co/datasets/futurehouse/labbench2
- **Autonomous Science Bench (AS-Bench) 0.1:** the closest direct match to our experiment-design objective. It evaluates closed-loop, budgeted campaigns grounded in published lab experiments. A campaign succeeds only when both the *discovery gate* (quality of designs actually measured) and *learning gate* (ranking of designs never measured) pass. The official 0.1 dataset currently lists six campaigns: Nuclease replay, Biosensor replay, Propylene digital twin, Suzuki replay, OER replay, and Coercivity replay. Its current public leaderboard reports a 72.2% resolution rate for GPT-6 Astra/Codex and 66.7% for Opus 5.5/Claude Code. The published results page says ten configurations ran all six tasks with three trials per task; total reported cost was $190 and $173 for those two leading configurations. The benchmark authors explicitly caution that the small gap is not meaningful with only three trials per task. Treat these as configuration-specific pilot aggregates, not statistically established ranking, universal science scores, or matched comparisons with unrelated leaderboards. Five tasks use replay and one uses a digital twin; the live-lab API tier is defined but no v0.1 campaign uses it. Use existing Symthaea #10921 for the adapter. Its two-task Nuclease/Propylene slice is a pilot, not a complete AS-Bench v0.1 reproduction. Do not silently add tasks that appear in the upstream repository but are outside the official six-task dataset without a pinned scope/evaluator decision.
  https://asb.open-ssi.org/about
  https://asb.open-ssi.org/run
  https://github.com/Open-SSI/autonomous-science-bench
  https://arxiv.org/abs/2610.18449
- **SciWorkBench (17 Nov 2026 preprint):** real science tasks across biology, chemistry, materials science, and physics, with a best reported model–harness mean of 57.38. Its failure analysis highlights task-specific procedure translation, multimodal evidence integration, and scientific self-correction. Treat its score as a model-plus-harness result, not a base-model capability constant.
  https://arxiv.org/abs/2611.22490
- **FIRE-Bench (ICML 2026):** verified rediscovery tasks requiring agents to design experiments, execute them, and produce evidence-backed conclusions. The reported best tested system scores 46.7 F1, with substantial run-to-run variance and recurring planning errors.
  https://firebench.github.io/
  https://proceedings.mlr.press/v306/wang26d.html

- **SciWorkBench (17 Nov 2026 preprint):** real science tasks across biology, chemistry, materials science, and physics, with a best reported model–harness mean of 57.38. Its failure analysis highlights task-specific procedure translation, multimodal evidence integration, and scientific self-correction. Treat its score as a model-plus-harness result, not a base-model capability constant.
  https://arxiv.org/abs/2611.22490
- **FIRE-Bench (ICML 2026):** verified rediscovery tasks requiring agents to design experiments, execute them, and produce evidence-backed conclusions. The reported best tested system scores 46.7 F1, with substantial run-to-run variance and recurring failures in design, execution, and evidence-based reasoning.
  https://proceedings.mlr.press/v306/wang26d.html
- **AstaBench (Ai2):** broad tasks across literature, scientific coding/execution, data analysis, and end-to-end discovery. E2E-Bench and E2E-Bench-Hard test the full pipeline from a research task through experiment design/execution, analysis and final research deliverable. Keep its *partial-completion rubric score* separate from exact all-requirements completion. The Spring 2026 Asta v0 report's 3% perfect E2E-Bench-Hard figure is a historical result under that earlier evaluation; a 22 Nov 2026 third-party leaderboard snapshot reports a 0.7 partial-completion score for Opus 4.7, which is not a 70% perfect-completion rate and should not be directly compared to the older 3%. Reproduce the official benchmark version and grader before making a current SOTA comparison.
  https://allenai.org/asta/bench
  https://allenai.org/blog/astabench-update-spring-2026
  https://atlas.village.dev/benchmarks/astabench-2026-e2e-hard

### Interactive law and experiment discovery

- **NewtonBench (ICLR 2026):** 324 tasks across 12 physics domains, using counterfactual law shifts and interactive probes. Reported weaknesses include degradation with system complexity/noise and a tool-assistance paradox where code access can sometimes induce premature exploitation.
  https://arxiv.org/abs/2510.07172
- **Science-Gym:** seven physics/epidemiology simulations that explicitly combine data collection, experiment design, and equation discovery. It is a useful compact harness for the Stage A/Stage B distinction.
  https://doi.org/10.1007/s10994-025-06914-x
- **PhysGym (NeurIPS 2025):** interactive physics environments with controlled prior-knowledge levels, useful for measuring whether Symthaea is discovering structure or merely retrieving supplied background knowledge.
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/42abcf9dffa48ca44fea1c497539a914-Abstract-Datasets_and_Benchmarks_Track.html
- **DiscoverPhysics (2026):** 22 altered-physics worlds with sequential experiments and separate held-out trajectory-prediction and explanation scoring. The public/private split is useful for guarding against memorized laws and for testing the difference between fitting observations and understanding a mechanism.
  https://arxiv.org/abs/2605.26087

### Direct numerical-gravity comparators: same theory first, methods second

**Primary same-problem reference — Kleihaus, Kunz & Radu (2023).** *Phases of Rotating Black Objects in d=5 Einstein–Gauss–Bonnet Theory* studies five-dimensional asymptotically flat EGB black holes with two equal-magnitude angular momenta, exactly the background symmetry class relevant to the current equal-spin EGB campaign. The authors solve the nonperturbative second-order field equations with boundary conditions, report a scan of roughly one thousand numerical data points for the equal-spin existence domain, construct extremal configurations directly, and cross-check their numerical extremal solutions against an analytic near-horizon attractor family. They also state that integration becomes increasingly difficult near the critical endpoint. This is the closest published physical reference for background-branch validation—not a modern agent benchmark and not automatically the same perturbation/response problem as #7131.

Use this as a reproduction/coverage matrix:
- Einstein limit (alpha=0): recover the five-dimensional equal-spin Myers–Perry branch and its limiting behavior.
- Static EGB limit: recover the Schwarzschild–Tangherlini/EGB thermodynamic quantities and the allowed coupling bound.
- Non-extremal equal-spin domain: compare the same normalized quantities (mass, angular momentum, horizon area, entropy, temperature, horizon angular velocity) at matched dimensionless parameters and gauge.
- Extremal boundary: compare horizon expansions and near-horizon attractor data, not only extrapolated global curves.
- First law and asymptotic charges: independently reconstruct mass/angular momentum from far-field coefficients and check the thermodynamic first law along the continuation family.
- Critical endpoint: preserve unresolved intervals where conditioning/integration degrades; do not turn extrapolation into a certified branch endpoint.

The paper reports that the equal-spin solutions are found with a symmetry-reduced radial ansatz and ODEs. Its non-extremal gauge and extremal parametrization differ, so comparisons must include the coordinate/gauge map and normalization, not raw metric-function arrays. The reported ~1,000-point scan is an approximate literature coverage target, not a machine-readable exact benchmark dataset. Reproduce published plots/tables and maintain our own source-pinned grid with solver receipts.

https://www.mdpi.com/2218-1997/9/4/156
https://doi.org/10.3390/universe9040156

**Adjacent numerical-method comparator — Akribeia (2026).** Akribeia pairs physics-informed neural networks with pseudo-spectral refinement to construct rotating higher-curvature black-hole solutions. Its paper was accepted in Physical Review D on 12 November 2026. It covers quadratic and cubic curvature theories and multiple non-vanishing angular momenta, but its published problem set must be matched carefully before calling it a direct EGB comparator. It is meaningful for comparing the neural-to-spectral refinement pipeline, global field-equation residuals, parameterized solution families, and multi-angular-momentum handling—not for general scientific-agent intelligence.

A fair Akribeia comparison must match the theory, dimensions, angular momenta, gauge, boundary conditions, coupling/spin domain, observables, resolution, residual definitions, and independent qualification. Compare source-pinned equations, convergence, endpoint behavior, and interval/proof status—not marketing labels or unmatched residual digits. Akribeia's “certified neural-field” wording and high-precision residuals should not be interpreted as an existence/uniqueness theorem for the Symtropy EGB boundary-value problem without checking the paper's precise certification definition.

https://arxiv.org/abs/2607.07418
https://doi.org/10.1103/3kz2-6dvv

**Adjacent high-spin perturbation comparator — Husken et al. (2026).** A separate PRD paper computes scalar quasinormal-mode corrections for rapidly rotating black holes in scalar-Gauss–Bonnet and dynamical Chern–Simons gravity with pseudo-spectral collocation, reporting spins up to a/M=0.99 and mode-dependent accuracies. This is not the same five-dimensional pure EGB background or spin-2 response operator, but it is a useful numerical-method/perturbation benchmark for endpoint conditions, spectral convergence, and high-spin sensitivity. Keep the field content, dimension, and theory differences explicit.

https://doi.org/10.1103/jrnh-wsxw

## Benchmark programme

### Track A — Stage A research-action allocation

Use the blind episode protocol and existing owners (#8721, #11171, #11060, #5476, #5478). An exact finite-world control is now authored at [BH_EGB_STAGE_A_ACTION_ALLOCATION_CONTROL_V0_1.json](BH_EGB_STAGE_A_ACTION_ALLOCATION_CONTROL_V0_1.json), with its [method and known answers](BH_EGB_STAGE_A_ACTION_ALLOCATION_CONTROL_V0_1.md). It contains:

- a cheap symbolic derivation that identifies which of two states is true and changes the optimal decision (gross EVSI 0.50, cost 0.10, net +0.40);
- a degenerate-parameter BVP and a precision refinement that do not distinguish the states (zero EVSI, negative net value);
- a perfectly discriminating but uneconomic BVP (gross 0.50, cost 0.70, net -0.20);
- a high-net-value action that must still be vetoed because worst-case loss 0.80 breaches the frozen 0.40 risk limit;
- a perfect diagnostic that gains one bit of information but cannot change the optimal downstream decision (EVSI 0, net -0.04).

The fixture uses explicit finite-world priors and deterministic likelihoods to make the expected values exactly checkable. It is a synthetic policy control, not an EGB prior, real action-cost estimate, or calibrated Symthaea VoC forecast. Use it to test the decision-value contract before attempting to calibrate the live predictor.

Compare the same model/backbone and inclusive budgets:

1. no optional tool;
2. fixed schedule;
3. random matched-cost action;
4. static residual/uncertainty threshold;
5. simple variance/disagreement heuristic;
6. heuristic VoC;
7. calibrated prospective VoC;
8. exact enumerated tiny-world oracle (diagnostic upper bound only).

Each forecast must precede action execution and bind to the evidence-prefix digest. Keep separate scores for action selection and outcome interpretation.

Primary metrics: cost to first independently qualified result; decision regret; unnecessary action/tool rate; premature commitment; correct unresolved/abstention rate; forecast calibration where probabilities are valid; solver calls, CPU/GPU, wall time, memory, tokens, money, and human-review burden.

### Track B — Stage B numerical experiment design

On identical fixed hypothesis/design sets, compare:

1. fixed and random designs;
2. the existing scalar variance/disagreement selector (population variance across available hypothesis predictions);
3. a coverage-aware variant of that baseline with a frozen minimum valid-prediction fraction and explicit UNRESOLVED disposition;
4. the original #8034 conditional-independence score after its own qualification;
5. v0.2 covariance-aware maximin Mahalanobis screen from a PSD joint hypothesis-output covariance;
6. robust worst-case design over a predeclared covariance uncertainty set (future work);
7. Gaussian Monte Carlo EIG / expected log Bayes factor only where predictive distributions and priors have been independently justified;
8. interval/set-based discrimination where probability distributions are not defensible;
9. exact tiny-world optimal design oracle.

**Baseline interpretation:** the current `select_most_informative_experiment` implementation in `symthaea-core/src/hdc/conjecture_engine/experiment_selection.rs` computes population variance across predictions and omits hypotheses whose prediction closure returns `None`. This is a useful low-cost disagreement baseline. It is not EIG, has no explicit experiment cost or joint covariance, and its score can be confounded by changing prediction coverage. Keep the legacy implementation intact as a comparator; any coverage-aware arm must publish its frozen eligibility threshold and missing-prediction disposition.

Controls: exact duplicate observables; determinant/condition number derived from the same matrix; common-mode error; invalid/indefinite covariance; individually PSD but jointly impossible pairwise covariances; singular support mismatch and in-support mean; unit rescaling; invertible basis transformation; source/model digest change; covariance miscalibration; nuisance-parameter shifts; observational noise; irrelevant observables; a hypothesis prediction returning None; a candidate supported by only a small subset of hypotheses; a design that cannot change the downstream decision.

Metrics: pairwise discrimination, ranking regret, prediction-coverage fraction, excluded-hypothesis reasons, redundancy/unit/basis invariance, joint covariance PSD validity, covariance coverage, calibration, failure/abstention counts, and cost to the next qualified decision.

### Track C — integrated research workflow

Evaluate end-to-end policy behavior on compatible slices of FIRE-Bench, SciWorkBench, AstaBench, NewtonBench, Science-Gym, PhysGym, and DiscoverPhysics. Do not pool benchmark scores: keep each published protocol and version intact, and report the added adapter's incompatibilities.

Run paired baselines under the same model/harness, tools, context, memory, task information, and resource caps. Include a blind holdout whose generating truth and scoring code never enter the researcher process. Score both partial progress and exact completion; preserve all failed, timed-out, invalid, and abstained attempts.

### Track D — EGB numerical physics

The EGB proving ground should consume Symtropy #7131 rather than recreate its response model. Each episode should bind:

- source/action/fixture SHA and equation-row identity;
- response matrix R, affine offset c, and their derivatives as separate objects;
- horizon/infinity basis identities and Gram metrics;
- physical parameters, units, gauge, ensemble, radial chart, and normalization;
- original-equation and Noether residual vectors;
- independent horizon-outward and infinity-inward propagation;
- discretization, truncation, precision, and model-form uncertainty;
- finite-interval invertibility enclosures or an explicit unresolved status;
- event classification that distinguishes endpoint-map singularity, nonlinear fold, mode-count change, chart singularity, solver failure, and PDE hyperbolicity.

Compare Akribeia only on genuinely matched subproblems. A small BVP residual is not a branch-completeness, existence, uniqueness, or hyperbolicity proof.

## Proposed targets and decision gates

These are goals, not achieved results.

### Contract/mechanical gates

- 100% rejection of the frozen malformed, stale-source, invalid-covariance, non-PSD, digest-tampering, and false-independence fixtures.
- Exact known-answer tolerance of 1e-10 on the synthetic redundancy control, with zero score gain from adding x3=x1+x2.
- Zero silent diagonal jitter or undisclosed covariance repair.
- Unit-rescaling and well-conditioned invertible-basis metamorphic tests preserve the theoretical score within a declared numerical tolerance.
- Every missing/invalid uncertainty model produces a fail-closed or UNRESOLVED disposition, never manufactured confidence.

### Policy gates

- Stage A must beat the strongest simple action policy on a preregistered paired held-out endpoint under matched inclusive cost, with confidence intervals and no worsening of false-promotion or abstention safety measures.
- Stage B must improve decision-relevant discrimination or cost-to-qualified-result over the best qualified simpler design without loss of calibration, redundancy invariance, or false-confidence controls.
- If a static policy is Pareto-equivalent, choose the simpler policy.
- Report separate Stage A and Stage B effects before claiming integrated synergy.

### Stretch external targets

After reproducing the official baselines with the exact model/harness, benchmark revision and grader, target:
- **AS-Bench 0.1 resolution rate above 80%**, against the public 72.2% leader, while reporting discovery and learning gates separately and retaining every failed/incomplete trial;
- **SciWorkBench mean above 65**, against the reported 57.38 best configuration;
- **FIRE-Bench F1 above 60**, against the reported 46.7 best tested configuration;
- **AstaBench E2E-Bench-Hard partial-completion score above 0.80** on the same 2026 scoring/judge snapshot, with exact perfect-completion reported as a separate metric. The historical 3% figure is not the denominator or comparator for the newer partial score;
- **BIABench:** on the exact v1.0/basic-instruction/three-run/process-judge protocol, improve the strongest reproducible baseline by at least 10 percentage points on the combined *outcome* score while not regressing process/QC score; on preselected hard 3D/time-lapse tasks whose published per-task best is at or below 0.19, target over 0.30 outcome score with 10+ runs and uncertainty intervals. Report every task separately, and do not train on evaluation ground truth;
- **LABBench2:** target at least a 5 percentage-point paired improvement over the strongest matched-cost baseline on predeclared held-out subtasks, while preserving all subtask denominators and no-regression guardrails. This is a proposed internal comparison, not a claim against a currently verified leaderboard;
- **BioBasicsBench:** no numeric SOTA target until external task/rubric access enables an independent reproduction. The published provider score is a reference only.

These are provisional stretch goals, not results. For AS-Bench, reproduce its published three-trial-per-task protocol first, then run a separately labelled confirmatory extension with at least 20 paired trials per task and matched resource budgets. Small leaderboard gaps must not be described as significant without matched trials and uncertainty intervals. No combined score across these benchmarks is valid: task units, harnesses, budgets, graders and outcome semantics differ.

### Qualified Knowledge Yield

Retain the proposed operational metric:

\[
\mathrm{QKY}=
\frac{\text{independently qualified claims}}
{\text{total research expenditure}}.
\]

Define expenditure explicitly as a vector, not one currency conversion: solver/tool calls, CPU/GPU time, wall time, memory, model tokens, external queries, money, and human-review hours. Publish the full Pareto frontier with false-promotion rate, unsupported-singleton rate, calibration, proof latency, branch coverage, cost to first qualified result, failure denominators, and human burden. A higher QKY is not acceptable if it is achieved by narrowing claim scope or dropping failed attempts.

## Immediate execution order

1. Qualify the workspace fix (#13080) and then the original #8034 focused crate workflow at exact heads.
2. Review and qualify the additive covariance implementation in #13453 against an independent linear-algebra oracle before merging it.
3. Add arbitrary invertible-basis metamorphic tests and covariance-uncertainty-set controls; the current v0.1 only covers positive diagonal unit changes and point covariance matrices.
4. Implement the Symtropy #7131 typed adapter with endpoint basis/Gram-metric transport and the actual response-map semantics.
5. Pin the actual research-policy callable through #11171, then run Stage A fixed/random/heuristic/VoC comparisons with prospective immutable forecasts.
6. Run a small public EGB pilot to debug the end-to-end evidence path. Do not use that pilot for superiority claims.
7. Freeze a sealed confirmatory set, sample-size/power plan, model/tool/solver pins, budgets, and scoring before evaluating results.
8. Publish results including null findings and failure denominators; claim superiority only where the matched held-out comparison supports it.

## Reusable references

- SciWorkBench: https://arxiv.org/abs/2611.22490
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- AstaBench: https://allenai.org/asta/bench
- Ai2 AstaBench update: https://allenai.org/blog/astabench-update-spring-2026
- NewtonBench: https://arxiv.org/abs/2510.07172
- Science-Gym: https://doi.org/10.1007/s10994-025-06914-x
- PhysGym: https://proceedings.neurips.cc/paper_files/paper/2025/hash/42abcf9dffa48ca44fea1c497539a914-Abstract-Datasets_and-Benchmarks_Track.html
- DiscoverPhysics: https://arxiv.org/abs/2605.26087
- DiscoveryAgent: https://arxiv.org/abs/2611.21950
- Akribeia: https://arxiv.org/abs/2607.07418 ; https://doi.org/10.1103/3kz2-6dvv
- Optimal Designs for Correlated Data: https://doi.org/10.1146/annurev-statistics-042324-012947

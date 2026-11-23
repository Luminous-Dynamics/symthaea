# Evidence-directed scientific discovery: paradigm shifts and benchmark strategy v0.1

**Status:** research and evaluation design. This document proposes experiments and stretch targets; it reports no Symthaea/Symtropy superiority result.

## Executive thesis

Build a system that optimizes for **independently qualified knowledge per unit of research expenditure**, not for the number of hypotheses, experiments, tool calls, pages, or low residuals it produces.

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

For each rival pair, consume a predeclared covariance of the difference vector. Do not infer cross-hypothesis covariance from marginal standard deviations without an explicit coupling model. In singular cases, test support membership separately from the pseudoinverse quadratic form. No silent diagonal jitter.

The current v0.1 extension adds this contract and synthetic control; it does not yet model covariance uncertainty sets or Symtropy's endpoint Gram metrics. Work on correlated optimal designs identifies the additional assumptions and verification difficulty as a central open issue, rather than a cosmetic implementation detail (López-Fidalgo & Wong, 2026).

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

- **SciWorkBench (17 Nov 2026 preprint):** real science tasks across biology, chemistry, materials science, and physics, with a best reported model–harness mean of 57.38. Its failure analysis highlights task-specific procedure translation, multimodal evidence integration, and scientific self-correction. Treat its score as a model-plus-harness result, not a base-model capability constant.
  https://arxiv.org/abs/2611.22490
- **FIRE-Bench (ICML 2026):** verified rediscovery tasks requiring agents to design experiments, execute them, and produce evidence-backed conclusions. The reported best tested system scores 46.7 F1, with substantial run-to-run variance and recurring failures in design, execution, and evidence-based reasoning.
  https://proceedings.mlr.press/v306/wang26d.html
- **AstaBench (Ai2):** thousands of tasks across literature, scientific coding/execution, data analysis, and end-to-end discovery. Ai2's Spring 2026 update reported roughly 53% overall for its Asta v0 system but only 3% perfect end-to-end completion on the then-reported E2E-Bench-Hard slice, despite many runs completing 60–70% of component steps. Later scoring/leaderboard snapshots should be compared only within the exact same benchmark version and rubric.
  https://allenai.org/asta/bench
  https://allenai.org/blog/astabench-update-spring-2026

### Interactive law and experiment discovery

- **NewtonBench (ICLR 2026):** 324 tasks across 12 physics domains, using counterfactual law shifts and interactive probes. Reported weaknesses include degradation with system complexity/noise and a tool-assistance paradox where code access can sometimes induce premature exploitation.
  https://arxiv.org/abs/2510.07172
- **Science-Gym:** seven physics/epidemiology simulations that explicitly combine data collection, experiment design, and equation discovery. It is a useful compact harness for the Stage A/Stage B distinction.
  https://doi.org/10.1007/s10994-025-06914-x
- **PhysGym (NeurIPS 2025):** interactive physics environments with controlled prior-knowledge levels, useful for measuring whether Symthaea is discovering structure or merely retrieving supplied background knowledge.
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/42abcf9dffa48ca44fea1c497539a914-Abstract-Datasets_and_Benchmarks_Track.html
- **DiscoverPhysics (2026):** 22 altered-physics worlds with sequential experiments and separate held-out trajectory-prediction and explanation scoring. The public/private split is useful for guarding against memorized laws and for testing the difference between fitting observations and understanding a mechanism.
  https://arxiv.org/abs/2605.26087

### Direct numerical-gravity comparator

**Akribeia** pairs physics-informed neural networks with pseudo-spectral refinement to construct rotating higher-curvature black-hole solutions; its paper was accepted in Physical Review D on 12 November 2026. It is a meaningful comparator for the numerical-gravity layer, not for general scientific-agent intelligence.

A fair comparison must match the theory, dimensions, angular momenta, gauge, boundary conditions, coupling/spin domain, observables, resolution, residual definitions, and independent qualification. Compare source-pinned equations, convergence, endpoint behavior, and interval/proof status—not marketing labels or unmatched residual digits. Akribeia's reported high-precision residuals do not, by themselves, establish equivalence to Symtropy's response-operator goals or a theorem for this EGB problem.

https://arxiv.org/abs/2607.07418
https://doi.org/10.1103/3kz2-6dvv

## Benchmark programme

### Track A — Stage A research-action allocation

Use the blind episode protocol and existing owners (#8721, #11171, #11060, #5476, #5478). Compare the same model/backbone and inclusive budgets:

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
2. scalar variance/disagreement selector;
3. the original #8034 conditional-independence score after its own qualification;
4. covariance-aware maximin Mahalanobis screen;
5. robust worst-case design over a predeclared covariance uncertainty set (future work);
6. Gaussian Monte Carlo EIG / expected log Bayes factor only where predictive distributions and priors have been independently justified;
7. interval/set-based discrimination where probability distributions are not defensible;
8. exact tiny-world optimal design oracle.

Controls: exact duplicate observables; determinant/condition number derived from the same matrix; common-mode error; invalid/indefinite covariance; singular support mismatch and in-support mean; unit rescaling; invertible basis transformation; source/model digest change; covariance miscalibration; nuisance-parameter shifts; observational noise; irrelevant observables; and a design that cannot change the downstream decision.

Metrics: pairwise discrimination, ranking regret, redundancy/unit/basis invariance, covariance coverage, calibration, failure/abstention counts, and cost to the next qualified decision.

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

After first reproducing the published baselines with the exact model/harness and benchmark revisions, target:
- SciWorkBench mean above 65 (against the reported 57.38 best configuration);
- FIRE-Bench F1 above 60 (against the reported 46.7 best tested configuration);
- at least 10% exact perfect completion on the corresponding E2E-Bench-Hard rubric/snapshot (against Ai2's earlier 3% report).

These are deliberately ambitious, provisional stretch goals. They cannot be called SOTA unless the benchmark version, task set, model, harness, budget, grader, and statistical protocol are matched. Later AstaBench rubric/leaderboard updates must not be conflated with the earlier perfect-completion statistic.

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

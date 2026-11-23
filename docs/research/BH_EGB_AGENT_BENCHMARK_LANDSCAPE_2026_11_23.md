# Scientific research-agent and black-hole solver benchmark landscape — 23 November 2026

**Purpose:** set a current, fair external comparison for the Symthaea × Symtropy EGB campaign. This is a literature/benchmark audit, not a claim that Symthaea has been evaluated on these suites or outperformed any comparator.

## Executive conclusion

The research-agent field is not close to solved. Recent benchmarks show that agents can perform useful component tasks but still struggle to complete end-to-end scientific workflows, design correct experiments, combine evidence, self-correct, and avoid unsupported claims.

The EGB programme has a second, distinct comparison: the mathematical/numerical solution of rotating higher-curvature black holes. A research-agent benchmark cannot establish numerical-gravity superiority, and a small PDE residual cannot establish research-policy superiority. Score these two capabilities separately.

## Current external reference points

| Benchmark / system | Scope | Reported evidence | How to use it |
|---|---|---|---|
| [SciWorkBench](https://arxiv.org/abs/2611.22490), 17 Nov 2026 preprint | 98 real scientific workflow tasks across biology, chemistry, materials and physics | Best reported model–harness pair: Codex / GPT-5.6 Sol, mean 57.38; Claude Code / Opus 5, 57.31. Physics-specific top result reported as 57.23 for Claude Code / Qwen3.8-Max. Authors identify procedure translation, multimodal evidence integration and scientific self-correction as recurrent weaknesses. | Most relevant fresh broad-workflow comparator. Compare model + harness pairs, not model names alone. Its score is rubric-based and preprint evidence; do not present it as a universal science score. |
| [FIRE-Bench](https://proceedings.mlr.press/v306/wang26d.html), ICML 2026 | Rediscovery of 30 verifiable insights from recent ML papers; planning, implementation, execution and claim-level analysis | Strongest tested agents remain below 50 F1, with high run-to-run variance and recurring failures in experiment design, execution and evidence-based reasoning. | Best fit for claim-level evidence and rediscovery. Build a small EGB adapter only where task semantics genuinely match; do not pool scores. |
| [AstaBench](https://allenai.org/asta/bench), living suite; results update 30 Apr 2026 | Thousands of tasks across literature, coding/execution, data analysis and end-to-end discovery | April update: Claude Opus 4.7 scored 58.0% overall at $3.54/problem; GPT-5.5 52.9% at $1.61/problem. On E2E-Bench-Hard, the best agent completed only 3% perfectly end-to-end. AstaBench notes that the new end-to-end scorer penalizes fabricated results and placeholder code more reliably. | Strong industry ecosystem and cost/quality comparator. Use E2E tasks and the cost frontier, not its overall score as a substitute for black-hole physics. |
| [DiscoveryAgent](https://arxiv.org/abs/2611.21950), 16 Nov 2026 preprint | Scientist-steered research with checkpointed trajectories and independent audit agents | Reports scientists collaborating with audit agents found 14× more valid errors than manual review in the study's evaluated setting; also reports strong task-specific predictive results. | Architectural comparator for checkpointed work, audit agents and researcher steering. The 14× figure is study-specific, not a guaranteed general effect. |
| [Akribeia](https://doi.org/10.1103/3kz2-6dvv), PRD accepted 12 Nov 2026 | Neural discovery followed by pseudo-spectral refinement for rotating black holes in higher-curvature gravity | Constructs families with multiple nonzero angular momenta and highly nonlinear/nonperturbative ODE systems; claims globally defined neural fields with extremely high-precision field-equation residual verification. | Direct numerical-gravity competitor. Match theory, dimension, spin, couplings, boundary ensemble, observables, solver budgets and residual definitions before comparing. |

## Three different leaderboards, not one

### 1. Research-workflow quality

Measures whether the system turns a research brief into a valid, reproducible result.

Relevant outcomes:
- correct task decomposition and procedure selection;
- source and data acquisition with provenance;
- legal, discriminating experiments;
- reproducible execution;
- evidence integration and claim traceability;
- self-correction after counterexamples;
- final result completeness.

External comparators: SciWorkBench, AstaBench E2E-Bench, FIRE-Bench.

### 2. Scientific correctness and epistemic discipline

Measures whether the result is actually supported.

Relevant outcomes:
- exact source and fixture binding;
- independent original-equation checks;
- rejection of wrong signs, changed projectors and stale sources;
- calibration and appropriate abstention;
- correct handling of non-identifiability;
- false-promotion rate;
- evidence survival under disconfirm-first review.

This is where the proposed EGB campaign can make a differentiated contribution: score every atomic claim against a specific oracle, not the prose quality of a research report.

### 3. Numerical and mathematical physics quality

Measures whether the computed gravitational solution is correct and what has been proved.

Relevant outcomes:
- matched reference-profile and observable errors;
- original unscaled field-equation residuals by row;
- spectral convergence and tail estimates;
- independent endpoint matching;
- sensitivity derivative convergence;
- interval invertibility/existence bounds;
- branch-event classification;
- separate PDE hyperbolicity qualification.

Direct comparator: Akribeia and specialist spectral/continuation methods. This category must not be inferred from the agent-workflow score.

## Proposed EGB research-agent benchmark

The campaign manifest defines eight episodes and ten action types. Turn these into a small, initially transparent development set and a separate sealed confirmatory set after the episode semantics are stable.

### Episode families

1. Source/fixture identity and action-row replay.
2. Einstein-limit response-map recovery.
3. Full first coupling derivative versus affine offset.
4. Basis covariance and metric-aware conditioning.
5. Akribeia 5D Myers–Perry exact control.
6. Ma–Lü parameter and ensemble conversion.
7. Finite-positive-coupling BVP correction and sensitivity convergence.
8. Local O(a²) analytic Fredholm / implicit-function theorem obligations.

Every episode must include a positive control, a near-miss, at least one hostile mutation, an allowed-action set, the information visible to the policy, a deterministic or independently reviewable oracle, and the exact claim ceiling.

Do not let the policy see hidden truth, sealed challenge identities, evaluator residuals, or future observations.

### Policy arms

At minimum compare:
- researcher-directed action selection;
- fixed schedule;
- random matched-cost actions;
- a simple static residual/uncertainty threshold;
- existing qualified active experiment selector;
- actual Symthaea research-policy incumbent;
- same incumbent with HDC retrieval disabled;
- same incumbent with CfC campaign-state modelling disabled;
- same incumbent with prospective VoC disabled or replaced with a sham;
- exact tiny-world action oracle as an upper-bound diagnostic only.

Hold model/backbone, context, prompts, tools, memory, starting evidence and resource caps constant. Report model–harness pairs. Use paired initial states and seeds; the evaluator must retain the blind split.

### Hard safety and correctness gates

- Zero false promotions on the frozen wrong-source, wrong-sign, wrong-projector, singular-basis and receipt-tampering controls.
- Every result retains its source/action/input/output identities and claim ceiling.
- A missing or skipped workflow is never scored as pass.
- A response-map zero is not automatically a nonlinear fold.
- A stationary BVP pass is not a strong-hyperbolicity result.
- A theorem assistant's suggestion is not a proof; the theorem checker must verify the actual statement and assumptions.
- Correct abstention on underdetermined cases must beat false certainty, but it is not counted as successful resolution.

### Performance goals

Do not choose one magic aggregate score. Compare a vector and Pareto frontier:

1. independently qualified claims per total compute and monetary cost;
2. cost to first exact-source replay;
3. cost to first independent derivation;
4. cost to first interval-certified result;
5. action-forecast calibration and realized decision regret;
6. falsification power at a predeclared false-alarm rate;
7. false-promotion and false-unique rates;
8. branch/parameter coverage, separately from interval-validated coverage;
9. human review time and high-severity omission rate;
10. failed, invalid, timed-out and unresolved attempts in all denominators.

An HDC, CfC or FEP feature earns a place in the stack only if its preregistered ablation improves the validation/cost Pareto frontier without worsening false-promotion, calibration or subgroup performance. Otherwise keep the simpler baseline.

## Industry comparison and differentiation strategy

AstaBench's adoption by the UK AI Security Institute ecosystem and submissions from Elicit, SciSpace, Distyl AI and EvoScientist make it a useful interoperability target. But the Symthaea advantage should not be asserted from architecture novelty or public code volume.

The strongest differentiator to pursue is **reproducible scientific self-correction with independently qualified claims**:
- research checkpoints bind each hypothesis, action forecast, experiment, output and conclusion;
- independent auditors reconstruct claims from source and execution evidence;
- disconfirm-first review searches for counterexamples and high-impact omissions;
- evidence updates cannot overwrite historical claims;
- a learned research policy must demonstrate prospective decision value over simpler schedules at matched cost.

This is a testable research contribution, not yet an achieved capability.

## Fair comparison checklist

Before any public superiority claim, publish:
- benchmark task definitions and licence/adapter status;
- exact model and harness identifiers;
- prompts, tool versions, context and memory budgets;
- compute, wall-time, tokens, solver calls and monetary cost;
- inclusion/exclusion rules and complete denominators;
- paired uncertainty intervals and per-stratum results;
- evaluator independence and contamination controls;
- exact source commits and replayable result artifacts;
- failures, abstentions, invalid runs and unresolved outcomes.

Do not compare a perturbative O(a²) Symtropy profile directly with a fully nonlinear finite-spin Akribeia profile. Begin with the pinned Einstein-limit 5D Myers–Perry control; then build a fair finite-coupling comparison only after the physical ensembles and conventions match.

## References

- SciWorkBench, [arXiv:2611.22490](https://arxiv.org/abs/2611.22490), 17 Nov 2026.
- FIRE-Bench, [PMLR 306 (ICML 2026)](https://proceedings.mlr.press/v306/wang26d.html).
- AstaBench, [benchmark portal](https://allenai.org/asta/bench), and [30 Apr 2026 results update](https://allenai.org/blog/astabench-update-spring-2026).
- DiscoveryAgent, [arXiv:2611.21950](https://arxiv.org/abs/2611.21950), 16 Nov 2026.
- Akribeia, [PRD accepted paper](https://doi.org/10.1103/3kz2-6dvv), 12 Nov 2026.

# Open-World Scientific Discovery — Benchmark Preregistration V0

Date: 2026-11-08  
Status: **design/preregistration template, not an executed or pre-registered campaign**  
Parent lane: [#11042 — Open-World Physics Ansatz](https://github.com/Luminous-Dynamics/symthaea/issues/11042)  
Implementation: [PR #11060](https://github.com/Luminous-Dynamics/symthaea/pull/11060)

## 1. Purpose and claim boundary

This protocol specifies how to compare Symthaea research policies on scientific structure discovery without confusing:

1. fitting coefficients in a supplied model;
2. identifying the structural family;
3. identifying hidden variables and interaction structure;
4. identifying the measurement process;
5. predicting interventions and held-out environments;
6. selecting experiments or tools that change a downstream decision;
7. reproducing an existing result versus producing a novel result.

The aim is **validated discovery per fully inclusive cost**, while preserving uncertainty and non-identifiability. A low training error, plausible explanation, high information score, or passing unit test is insufficient by itself.

This document is not a report of a completed campaign. No model has yet been evaluated under this preregistration. Thresholds below are prospective hypotheses, not achieved performance.

## 2. Scientific hypotheses to test

- **H1 — Structure access matters:** on paired traces, A0 should isolate the coefficient-fitting ceiling; the A2 gap measures the additional cost of discovering the mechanism rather than fitting it.
- **H2 — Experiments should target decisions:** an experiment policy selected by expected terminal decision value should reduce cost-to-validated-decision versus fixed, random and information-only policies, without increasing false-confidence or violating risk limits.
- **H3 — Instrument and mechanism are jointly identifiable only under suitable designs:** explicit sensor-distortion controls should prevent false structural certainty; calibrated instruments or complementary sensors should resolve ambiguity only when the available action semantics make it resolvable.
- **H4 — Generalization is a stronger test than interpolation:** a candidate should survive held-out ranges, interventions, time horizons and related environments; it must also reject transfer into unrelated or genuinely changed mechanisms.
- **H5 — Reproduction is a prerequisite to innovation claims:** a research system must first reproduce a known result under hidden confirmation before a claimed improvement is considered.
- **H6 — Research-process improvement must transfer:** a successor policy must outperform a frozen incumbent on sealed task families under matched inclusive budgets, not merely improve its development examples.

## 3. Benchmark strata — no pooled leaderboard

| Track | Primary benchmark / comparator | Construct | What must be held constant |
|---|---|---|---|
| Ansatz / law discovery | Unsatz; DiscoverPhysics | Discover variables, hidden state, interactions, measurement and governing form; predict unseen interventions and long horizons | Task DSL, world seeds, attempt count, interventions, model/fitter access, compute and scoring |
| Causal graph discovery | ACDB; TCD-Arena / CausalVerse adapters | Recover skeleton, directions and intervention effects under a budget | SCM class, observation/noise semantics, intervention budget, graph density, prior access |
| Global world understanding | WorldTest / AutumnBench-style tasks | Answer reachability, consequence and environment-level questions beyond replaying seen rollouts | World interaction affordances, evidence, question distribution and evaluator |
| Research reproduction | FIRE-Bench; selected Scientist World tasks | Reproduce verifiable published claims before proposing improvements | Source-paper exposure, datasets, code access, environment, baseline and confirmation protocol |
| End-to-end research | FrontierPhysics; AIRS-Bench; SciAgentArena; ScienceAgentBench; AstaBench | Literature, planning, experiments/data, analysis and scientific artifact quality | Tool access, task specification, review rubric, execution limits, human-review budget |
| Invariance / transfer | #8705 + held-out environment adapters | Distinguish invariant mechanisms from nuisance shifts and actual law changes | Environment splits, intervention semantics, source-model access, mechanism families |

Do not average percentages across these tracks. They measure different constructs and have different scoring rules.

## 4. Matched experimental arms

### 4.1 Structure-access treatment

For each deterministic world seed, create one evaluator-owned generating model and one frozen discovery trace.

- **A0:** supply the correct structural family; hide coefficients, initial conditions, noise and sensor parameters as specified.
- **A2:** hide the structural family too.
- **A3:** additionally hide the sensor structure and calibration status.
- **A4:** require extrapolation across horizon, intervention and range.
- **A5:** provide a related source model; include both truly related and unrelated target worlds.

For the primary A0/A2 structural-access contrast, both arms must receive the **same exact observations, intervention history, split, costs and compute cap**. This is a controlled structure-access experiment, not an adaptive-policy comparison.

### 4.2 Adaptive policy comparison

Because adaptive agents may choose different experiments, run a separate policy experiment with the same world seed, starting evidence, legal action set and maximum inclusive budget, while allowing actions to diverge. Record all actions and counterfactual predictions before each outcome. Do not call this an identical-trace A0/A2 comparison.

Compare:
1. no optional tool / no additional experiment;
2. fixed grid;
3. seeded random matched-cost;
4. maximum partition count;
5. minimax worst-case survivors;
6. maximum entropy and minimum expected survivors under their declared uniform grammar prior;
7. exact small-world dynamic-programming oracle for a declared terminal utility (diagnostic upper bound only);
8. existing Symthaea incumbent;
9. a versioned challenger;
10. decision-focused prospective value-of-computation / EVSI policy adapted from existing owners.

Do not implement another general scheduler or probe selector in this lane. Use #8286 for experimental actions, #5476/#5478/#8721 for VoC and #11012 for generic fixed-versus-adaptive orchestration.

## 5. Version-space and utility discipline

A finite grammar can have many syntactic parameterizations of one functional mechanism. Scoring and experiment selection must operate on the correct target level:

- **Syntax / parameterized model:** exact candidate model or parameter set.
- **Functional equivalence class:** models indistinguishable over the declared intervention language.
- **Structural family:** mechanism, relevant variables, latent-state and interaction structure.
- **Measurement process:** sensor transform, delay, censoring, missingness and calibration state.

Do not use uniform probability over raw syntax as a scientific prior without an explicit justification. A grammar with 100 equivalent expressions must not gain 100 times the prior mass of a one-expression family. Use #8669's class-level posterior and prior-splitting controls when integrating probabilities.

Information gain is not decision value. For each decision-focused probe, pre-register:
- current public evidence-prefix digest;
- candidate hypotheses / equivalence classes and probabilities;
- terminal decision and its utility matrix;
- predicted outcome distribution;
- expected change in terminal decision utility (gross EVSI);
- direct probe/tool cost and opportunity cost;
- risk / downside constraint;
- selected action and stop/continue/pivot/replicate/abstain rule.

The exact tiny-world oracle may enumerate future branches only for its declared utility and action set. It is not a general theorem or evidence of physical truth.

## 6. Primary outcomes — report as a vector

### Structure and model
- structural-family precision / recall / exact set match;
- functional-equivalence-class recovery;
- parameterized-model version-space precision / recall;
- variable relevance and interaction recovery;
- latent-state and measurement-map recovery, each separately;
- false unique diagnosis on non-identifiable worlds;
- out-of-support detection and unsupported-claim rate.

### Prediction
- held-out intervention error / proper log score;
- held-out range and long-horizon error;
- predictive interval coverage and sharpness;
- worst-environment regret;
- residual / model-misspecification diagnostics.

### Research judgment
- cost to first independently validated correct structure;
- decision regret versus exact oracle on small worlds;
- EVSI / VoC Brier or log score and calibration;
- premature commitment and unnecessary experiment rate;
- correct stop / pivot / replicate / abstain decisions.

### Transfer and reproduction
- positive-transfer lift on related mechanisms;
- negative-transfer rate on unrelated or changed mechanisms;
- independently reproduced result rate;
- novelty/improvement above a reproduced baseline, with independent confirmation.

### Evidence integrity and cost
- truth leakage, seed/split leakage, stale evidence-prefix or invalid action rate;
- claim-to-evidence trace completeness and independent-verdict reproducibility;
- all failures, timeouts, refusals, missing outputs and invalid attempts;
- model/token/tool costs, experiment costs, wall time, compute, memory, energy (only where actually measured), and human review time.

A critical leakage event, false unique claim on a known non-identifiable task, or invalid confirmation is a **non-compensatory failure**. It cannot be averaged away by a higher prediction score.

## 7. Minimum controls and acceptance gates

Before a confirmatory AI-policy result is interpretable:

1. **Oracle agreement:** 100% agreement on every finite reference-world case that the exact oracle enumerates.
2. **Non-identifiability:** zero false singleton/unique-structure claims on deliberately unidentifiable controls.
3. **A0 ceiling:** demonstrate that the same observations allow the correct-structure parameter-fitting control to succeed before attributing A2 failure to structure discovery.
4. **Blindness:** no evaluator truth, future outcome, confirmation split, oracle result or score before terminal submission.
5. **Evidence integrity:** zero stale evidence-prefix acceptances, unknown/duplicate evidence IDs, invalid intervention-family actions or budget overrun.
6. **Matched costs:** equal caps and full cost accounting across compared arms.
7. **Split integrity:** discovery, calibration, confirmation and transfer splits are disjoint at the world/seed/family level as preregistered.
8. **Denominators:** every attempted rollout remains visible; invalid, timed-out, failed and refused runs are explicitly counted.
9. **Independent review:** deterministic verifier plus domain/expert blockers for claims that require scientific interpretation.
10. **Reproducibility:** exact source SHA, model/config/prompt/tool registry, seed manifest, task revision, raw action trace and independently recomputed score artifact.

The first debugging pilot is not confirmatory. Keep its seed families out of the sealed evaluation. Use at least six structural families and target 20 paired seeds per held-out world for the initial core comparison; finalize the sample count with a preregistered precision/power analysis. Cluster uncertainty by world/task, not by treating correlated rollouts as independent.

## 8. Competitive reference points and stretch hypotheses

These are **aspirational targets** for a matched reproduction, not claims that Symthaea is near these scores. First reproduce the reference protocol and version. A public leaderboard is a moving snapshot, not an eternal industry standard.

| Benchmark | Public reference point | Symthaea stretch hypothesis after exact reproduction |
|---|---|---|
| DiscoverPhysics | Its public leaderboard snapshot (updated 2026-06-10) reports best Pass@1 57.5%, Pass@5 72.7%, explanation 0.83 ± 0.02. Trajectory fit and explanation quality can diverge. | At least +5 absolute percentage points Pass@1 and Pass@5, explanation ≥0.87, with no worse held-out trajectory error; task-clustered interval for Pass@1 improvement must exclude zero. |
| Unsatz | Authors report nine LLMs solve 21–60% across 80 systems / 20 domains. | Exceed 60% on the same 80-system release and improve the weakest structural strata; no cross-paper comparison unless DSL, fitter, costs and scoring match. |
| ACDB | Strongest non-oracle PC + greedy active orientation baseline reports directed F1 42.7%, SHD 4.79 on its current six-level ladder. | Directed F1 ≥47.7% and SHD ≤4.3 under identical intervention budgets, without worsening graph calibration or skeleton quality. |
| FrontierPhysics | 56 tasks, 42 physicists, nine domains; best reported tested setup passes 46/168 rollouts (Pass@1 27.4%) under verifier plus expert-blocker evaluation. | Pass@1 ≥35% on the same task/rubric release, with all critical verifier and expert-blocker criteria satisfied and task-clustered uncertainty reported. |
| WorldTest / AutumnBench | 43 environments / 129 queries; reported human performance materially exceeds tested frontier models. | Match or exceed the strongest published model per query family first; then close the human gap. Do not pool query families or use one average as the sole target. |
| FIRE-Bench / Scientist World | Reproduction of verifiable published findings is a separate construct from novel law discovery; Scientist World's initial release reports 1,000 executable paper-derived environments. | Reproduce a preregistered subset at or above the strongest matched baseline before any innovation lift is counted. Publish per-finding results, including failed reproductions. |

Do not pursue these stretch values at the expense of evidence validity, selective prediction, or cost. Beating a weak or incompatible baseline is not beating SOTA.

## 9. RSI promotion protocol

- Freeze the incumbent source, model, prompt, configuration, tools and budget.
- Permit the challenger to change one declared factor at a time where causal attribution is sought; record joint changes separately.
- Use development tasks for iteration; seal confirmation tasks and holdout families.
- No best-of-N selection on confirmation results. No patching after viewing sealed outcomes.
- Re-run both incumbent and challenger from the same starting conditions and budget.
- Require independent score recomputation and an audit of all failures/exclusions.
- Promote only when the preregistered primary endpoint improves by the practical margin, the interval excludes no improvement, and no safety/integrity/false-confidence guardrail regresses.
- If a simpler static policy is Pareto-equivalent to a learned scheduler, retain the simpler policy.
- Preserve negative results and rejected hypotheses as evidence; do not learn only from successful trajectories.

## 10. Architecture ownership — compose, do not fork

| Existing owner | Reuse boundary |
|---|---|
| #6879 / #6880 ScientificModel / PhysicsIR | Canonical candidate model representation, semantics, units, assumptions and provenance |
| #6868 / #6869 | Physical typing and explicit unknowns |
| #8286 | Noisy SCM active experiment design and intervention semantics |
| #8669 | Posterior over equivalence classes, support omission, class-level scoring and receipts |
| #8705 | Invariant mechanisms, environment transfer and change rejection |
| #5476 / #5478 / #8721 | Prospective value-of-computation, realized utility and tool-allocation comparisons |
| #10877 | Research direction/approach/experimental/interpretive taste |
| #11012 | Generic fixed-versus-adaptive orchestration |
| #9332 / #10756 | Sealed benchmark generation and evaluator quality |
| #6874 / #6875 | Proposal qualification / independent scientific acceptance |
| #8143 / #8150 | Existing open-world physics family-selection v0 |

This lane owns the matched benchmark experiment and thin adapter only. It does not own a second PhysicsIR, posterior engine, active selector, VoC predictor, scheduler, receipt service or evaluator.

## 11. External references

- Unsatz: https://arxiv.org/abs/2611.03713
- DiscoverPhysics: https://arxiv.org/abs/2605.26087 ; leaderboard: https://sampsonml.github.io/DiscoverPhysicsLeaderboard/
- Active Causal Discovery Benchmark: https://arxiv.org/abs/2609.31675
- FrontierPhysics: https://arxiv.org/abs/2611.06910
- WorldTest / AutumnBench: https://proceedings.mlr.press/v306/warrier26a.html
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- Scientist World: https://arxiv.org/abs/2611.08478
- AIRS-Bench: https://ai.meta.com/research/publications/airs-bench-a-suite-of-tasks-for-frontier-ai-research-science-agents/
- SciAgentArena: https://www.microsoft.com/en-us/research/publication/benchmarking-ai-agents-for-addressing-scientific-challenges-across-scales/
- ScienceAgentBench: https://hal.cs.princeton.edu/scienceagentbench
- AstaBench: https://allenai.org/asta/bench

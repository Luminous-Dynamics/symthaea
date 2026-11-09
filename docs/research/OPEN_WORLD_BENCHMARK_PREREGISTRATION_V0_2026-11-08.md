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

### 3.1 Campaign manifest and seed custody

The public campaign manifest is metadata, not the seed ledger. It pins the source commit, agent artifact, policy configuration, task contract, action/compute budgets, common starting-evidence digest and the comparison design. Each pair expands to exactly A0 and A2 with one common agent artifact/configuration and shared budgets.

- `matched_trace` requires an intervention-trace digest and uses identical evidence traces for the structure-access contrast.
- `adaptive_budget` requires common starting evidence and budgets but permits subsequent action histories to diverge.
- The actual seed ledger and sealed confirmation ledger remain evaluator-controlled; the public manifest stores only their content digests. Never commit their contents, low-entropy raw seed values, generating truth or confirmation outcomes.
- The Rust validator checks shape, digest syntax, pair uniqueness, A0/A2 arm order, nonzero budgets and design/trace consistency. It does not authenticate execution, prove seed custody or replace #9332.

## 3A. Methodological-guidance gradient (separate from A0/A2)

For project-level research tasks with an official B1–B4 contract, add a separate methodological-autonomy experiment:

| Level | Policy-visible guidance | Construct measured |
|---|---|---|
| B1 | Full scientific method and procedure | Execution/implementation ceiling with method supplied |
| B2 | Intended method and constraints; implementation details omitted | Ability to operationalize a known method |
| B3 | Objective, inputs, constraints and required outputs only | Autonomous method selection |
| B4 | B3 plus factually correct but non-essential context | Distractor robustness / invariance |

This factor is **not** the A0/A2 structure-access treatment. A0/A2 changes disclosure of the true structural family in a synthetic world; B1–B4 changes methodological instruction. Do not relabel one as the other. Where a task supports both, preregister a factorial design and keep the cells separately identifiable; otherwise run B1–B4 as a standalone adapter over the task owner's official harness.

ASI-Bench has two distinct reference snapshots. The paper reports 18 Agent×Model configurations with B1/B2/B3/B4 means 50.91/29.10/26.62/26.99 and strongest B3 row 51.60. The current supplied leaderboard lists 25 configurations, means 54.92/33.11/30.58/30.74 and overall leading row 55.84; two results are single-run and the board does not supply a scored task-set ID. The current board also says the published evaluation used no external tools. Do not merge these snapshots or infer task-level uncertainty from the macro means.

An additional specialist-system report must be treated separately: Unizen AI states that it locally scored 79.86 across the public 60-task ASI-Bench set using task-specific analytic/first-principles code prepared in advance, and did not submit the private seed42 set for official scoring. This is a vendor-reported, non-official local result, not a directly matched cold-start research-agent result. It motivates two preregistered tracks rather than one:
- **Cold-start generalist:** artifact/policy frozen before the task family is revealed; no per-task code or task-specific tuning after task presentation.
- **Compiled specialist:** task-specific solver/code is allowed, but count and publish the task-knowledge source, precomputation/build time, human engineering time, per-task tuning and artifact size/complexity.

For both tracks, seal task variants/seeds and charge all setup work to a separate inclusive budget. A solver created after seeing the exact public task is not a cold-start discovery success. Compare B1–B4 under a pinned task revision, runner, task set, rounds, model/harness, access and scorer.

Prospective Symthaea gates for an exact B1–B4 adapter:
- improve B3 against a frozen same-model/harness incumbent by a preregistered margin, without relying on greater total resources;
- preserve B1 performance within a predeclared non-inferiority margin;
- keep B4 within a small predeclared tolerance of B3 when the extra facts are specified as non-essential;
- publish each level's valid-artifact rate and cost, not only the overall mean;
- keep refusal, timeout, invalid artifact and failed verifier attempts in the denominator.

Terminal-Bench Science is an adjacent execution/reliability track, not a scientific-law score. Its task-owner 0.1 announcement describes 70 tasks and three independent trials/task; subsequent boards differ by agent/harness and evaluation snapshot. Vals' Nov 5, 2026 Mini-SWE-agent board lists 62.86% for GPT-6 Astra, 52.86% for GPT-6.1 Sol and 47.14% for Opus 5.5; Snorkel's board lists Codex/GPT-6 Astra 68.1% ±3.2 and Claude Code/Opus 5.5 63.3% ±3.3. The task owner's roadmap said 0.2 and 1.0 releases were in preparation after submission close. Treat those scores as distinct snapshots, not one leaderboard. Pin the task release, verifier revision, harness, model, timeout, attempt count and cost accounting before comparison.

## 3B. Law validity, candidate selection and evidence grounding

Add a complementary SciLaws-Bench adapter with two distinct task settings:

- **Real:** fixed scientific data; submit a closed-form law and score held-out numeric fit (S_N) separately from source-grounded scientific validity (S_V).
- **Parallel:** active queries to a residual-calibrated simulator whose hidden law is a newly synthesized structural variant; score hidden structural recovery (S_S) separately from numeric fit.

The upstream corpus reports 118 scientific problems from 381 papers across six disciplines, 291 candidate laws and roughly 8M data points. It contains 66 single-group and 52 multi-group problems; the latter test one shared functional form with per-group parameters and held-out group transfer. These are strong external task assets, but the official Python task packages and judging protocol must be pinned and licensing/runtime compatibility reviewed before an adapter claim.

SciLaws-Bench reports that numeric fit and validity agree in only 53.6% of unequal-fit candidate pairs, and that final candidate selection can discard a better generated candidate. Therefore require a three-part acceptance vector:
1. **Predictive fit:** held-out interpolation, range extrapolation, temporal transfer and group transfer as applicable.
2. **Scientific validity:** frozen, source-grounded rubric with anti-hacking checks and independent reviewer sampling.
3. **Structural recovery:** complete hidden added-term recovery in Parallel, not merely recovery of the published base equation.

Also preserve candidate selection:
- record every generated candidate and its exact model/provenance digest;
- freeze every candidate before selection, with pre-submission predictions and resource costs;
- independently score the best generated candidate and the submitted candidate;
- report generation-to-submission selection regret, candidate-pool coverage and Pareto domination;
- report the exhaustive best-of-N oracle only as an upper bound, never as an agent score.

### Evidence-grounding perturbation

On a subset of Real tasks and on synthetic controls, create paired versions of the same task:
- intact evidence;
- evidence withdrawn (the target-supporting rows/columns/statistic removed);
- evidence reversed (a pre-registered, independently verified transformation that reverses the relevant statistic or causal contrast);
- irrelevant metadata changed while the evidence remains fixed.

The claim should receive credit as *evidence-grounded* only when it is supported on the intact version, appropriately weakens or abstains when the decisive evidence is withdrawn, and changes consistently with the evidence when reversed. The irrelevant-metadata arm should leave the decision invariant. Preserve the original answer score separately. This tests whether the system uses the current evidence rather than recalling or inferring the answer from prior knowledge. It is inspired by evidence-grounded scientific evaluation on a limited single-cell task slice; this preregistration is not a claim of direct reproduction.

Prospective Symthaea target: reduce the best-generated-to-submitted gap under matched candidate-generation budget, without trading away scientific validity or structural recovery. Evidence-grounded accuracy must improve against a frozen incumbent while intact-task accuracy does not regress. Set margins only after a development pilot and freeze them before confirmation.

## 3C. Open-support / misspecification ladder

Separate closed-catalogue recovery from support-misspecification recognition:

| Stratum | Hidden truth relative to candidate catalogue | Required behavior |
|---|---|---|
| F0 | Truth is inside the declared catalogue | Exact compatible version space; identify or report the correct equivalence class |
| F1a | Parameter is valid for the world generator but outside the catalogue and not behaviorally represented by any legal candidate | Reject all incompatible candidates; report no-compatible-model rather than nearest singleton |
| F1b | Exact parameters are outside the catalogue but a legal candidate is observationally equivalent over the complete declared intervention language | Report the compatible candidate/equivalence class; do not claim exact generating-parameter identity or reject it as unknown |
| F2 | Structural family or term is outside the catalogue | Detect misspecification, request discriminating evidence, then propose a typed grammar extension in a later policy track |
| F3 | Measurement process differs from declared sensors | Diagnose instrument/model discrepancy rather than attribute every residual to the physical law |
| F4 | Dynamics change regime or become history/intervention dependent beyond declared model | Detect regime/transport failure and avoid unsupported transfer |
| F5 | Noise/discrepancy violates the declared observation model | Calibrate, model uncertainty or abstain; do not fit away structured error |

The initial F1a control is the gain-7 numeric world: the world generator accepts gains 1–16, while the finite candidate catalogue contains {1,2,3}; gain 7 + identity sensor has no behaviorally equivalent candidate. The F1b control is gain 4 + identity sensor, which is exactly parameter-OOD but behaviorally equivalent to candidate gain 2 + double sensor. Both are paired with F0 controls. The ordinary closed-grammar evaluator remains fail-closed if truth is accidentally omitted from its oracle. Neither F1a nor F1b is structural out-of-grammar discovery.

Required controls: an in-grammar case that must not be rejected; an out-of-catalogue case that must not be forced into a nearest singleton; an evaluator bug/omitted-truth closed-support case that must fail; and a shifted observation that could also arise from sensor or stochastic discrepancies. Detection and subsequent repair are separate outcomes.

## 3D. Reusable executable knowledge: cold-start, specialist and hybrid

ASI-Bench and related project suites expose a distinction between discovering a method and executing a method prepared in advance. Preregister three separate arms:

- **Cold-start generalist:** all method selection and model construction begins after task disclosure. No task-specific prebuilt source, task-specific parameter tables or post-disclosure manual tuning.
- **Compiled specialist:** allow a task-specific program prepared before task disclosure. Record the complete task-knowledge provenance, data/paper access, human and automated engineering time, candidate-generation/tuning budget, compilation and test cost, code size and maintenance cost. Separate public-task development from sealed-task evaluation.
- **Discover–compile–transfer hybrid:** let the researcher discover and validate a method on a source task, compile a bounded reusable capsule from the canonical ScientificModel / PhysicsIR, then freeze the capsule and test it on unseen related and unrelated tasks.

The hybrid is the most ambitious proposed paradigm, but its result is only meaningful if it passes an independent source confirmation before the compiler sees a promotion signal, then passes a disjoint transfer suite. The compiler may not add a second physical ontology, solver IR or evidence authority. Each capsule binds canonical model digest, assumptions, units, validity domain, measurements, parameter range, solver profile, compiler version, test corpus digest, counterexamples, uncertainty/calibration, source-task receipt and prior failed/rejected transfer attempts.

Metrics for all three:
- correctness / verifier pass, scientific validity and false-claim rates;
- source-task discovery quality and cost separately from compiled target execution;
- complete setup, human engineering, tuning, build, validation, execution and maintenance cost;
- first-use cost, amortized cost at K = 1/5/20/100 tasks and measured break-even K;
- held-out related transfer gain, unrelated negative-transfer rate and false-reuse rate;
- selection regret (best qualified capsule vs chosen capsule), support/coverage and abstention;
- codegen reproducibility, model-to-program differential correctness and applicability-envelope violations.

A compiled solver does not count as cold-start discovery merely because it is invoked as an autonomous agent. A solver with excellent fit on one task but no independent confirmation or transfer evidence remains task-specific, not a reusable scientific capability.

The simple amortization diagnostic (K^* = C_{setup}/(c_{baseline}-c_{compiled})) applies only when (c_{baseline}>c_{compiled}) and quality/integrity guardrails are non-inferior. Report total actual costs too; do not extrapolate break-even from a single chosen utilization rate.

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
7. structure-balanced information gain, with equal prior mass per surviving structural family and a uniform conditional prior over that family's compatible parameterizations (a prior-sensitivity control, not a calibrated posterior);
8. exact small-world dynamic-programming oracle for a declared terminal utility (diagnostic upper bound only);
8. existing Symthaea incumbent;
9. a versioned challenger;
10. decision-focused prospective value-of-computation / EVSI policy adapted from existing owners.

Do not implement another general scheduler or probe selector in this lane. Use #8286 for experimental actions, #5476/#5478/#8721 for VoC and #11012 for generic fixed-versus-adaptive orchestration.

## 5. Version-space and utility discipline
### Exact truth versus behavioral support

Report exact generating-parameter membership separately from observational representation. For the current numeric microcosm, behavioral representation is defined narrowly by equality of output signatures over every length-4 intervention sequence in the finite input alphabet {-1,0,1} (81 sequences). This makes the gain-4 / identity-sensor versus gain-2 / double-sensor equivalence test executable and deterministic.

The finite signature check is a bounded equivalence oracle for that declared world and horizon, not proof that two functions agree for all real inputs, arbitrary lengths, unrepresented state histories or other domains. Longer-horizon and continuous-domain equivalence require a separate proof, symbolic normalization, interval/SMT checks or held-out intervention coverage as appropriate. Report exact parameter membership, bounded behavioral representation and structural-family identification as three separate outcomes.


A finite grammar can have many syntactic parameterizations of one functional mechanism. Scoring and experiment selection must operate on the correct target level:

- **Exact generating-parameter membership:** whether the actual parameterized model literally appears in the catalogue.
- **Behavioral representation:** whether at least one legal candidate has the same outputs over the complete declared intervention language, even when its parameters/sensor factorization differ.
- **Functional equivalence class:** the full set of models indistinguishable over the declared intervention language.
- **Structural family:** mechanism, relevant variables, latent-state and interaction structure.
- **Measurement process:** sensor transform, delay, censoring, missingness and calibration state.

Do not use uniform probability over raw syntax as a scientific prior without an explicit justification. A grammar with 100 equivalent expressions must not gain 100 times the prior mass of a one-expression family. Use #8669's class-level posterior and prior-splitting controls when integrating probabilities. The microcosm now includes a structure-balanced one-step information-gain control: equal prior over surviving structural families, with a uniform conditional prior over the surviving parameterizations in each family. This removes raw family-cardinality bias, but the conditional prior is still grammar-dependent: duplicating one equivalent expression can change the within-family predictive distribution. The small control establishes stability only when added parameterizations leave that distribution unchanged. It is a transparent ablation, not representation-invariant Bayesian inference or a replacement for #8669's class-level posterior and prior-splitting controls.

A numeric reference-world control demonstrates why the distinction matters: with 120 parameterized hypotheses across direct and leaky-memory structures, a zero-input probe has structure-balanced information gain 0.3113 bits while nonzero input has 0.2279 bits. Uniform-over-model predictive entropy instead prefers nonzero input (2.1971 bits vs 1.8560 bits at zero). This is a deterministic objective comparison under the declared candidate grammar, not evidence that one policy is universally better.

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

### Law quality, model selection and evidence use

- S_N / held-out prediction score, by task and split type.
- S_V / scientific validity score and each rubric item, with anti-hacking failures.
- S_S / complete hidden structural-term recovery.
- Best-generated versus submitted-candidate score, selection regret and Pareto domination.
- Evidence-grounded accuracy on intact / withdrawn / reversed variants.
- Withdrawn-evidence uncertainty/abstention response and reversed-evidence action/conclusion response.
- Irrelevant-metadata invariance and prior-knowledge-only accuracy.

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
- deterministic algorithmic search-work counters (candidate evaluations, legal actions scored, solver calls) where the implementation can count them exactly; never relabel these counters as FLOPs, CPU seconds or total compute.
- specialist setup/compilation/engineering cost, amortized execution cost, break-even K, capsule retrieval/selection regret and false-reuse rate.

A critical leakage event, false unique claim on a known non-identifiable task, or invalid confirmation is a **non-compensatory failure**. It cannot be averaged away by a higher prediction score.

For the finite v0 protocol, an abstention is considered safe only when (a) the agent submits the exact evaluator-compatible version space and (b) more than one parameterized model remains compatible. An exact singleton accompanied by abstention is recorded as unnecessary abstention. This limited known-support rule does not replace #8669's future calibration and support-omission scoring.

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
| DiscoverPhysics | Leaderboard snapshot reports 22 worlds, 16 models, best Pass@1 57.5%, Pass@5 72.7%, explanation 0.83 ± 0.02. Pinned public commit 33b7fa9: main runner/README/stats explanation threshold 0.75; legacy runner 0.7 with raw error <0.5; leaderboard says 0.9. World inventory: registry 12; README table/sample 11; checked-in config 13 (two unregistered); variance keys 21; denominator 22. | **Parity blocked:** obtain canonical scorer and judge, exact public/private manifest, per-world variance and intended seed pool. Until then, do not compute/compare a 22-world rate. Only after parity and baseline reproduction should Symthaea target ≥+5 pp Pass@1/Pass@5 without worse per-world trajectory error. |
| Unsatz | Authors report nine LLMs solve 21–60% across 80 systems / 20 domains. | Exceed 60% on the same 80-system release and improve the weakest structural strata; no cross-paper comparison unless DSL, fitter, costs and scoring match. |
| ACDB | Strongest non-oracle PC + greedy active orientation baseline reports directed F1 42.7%, SHD 4.79 on its current six-level ladder. | Directed F1 ≥47.7% and SHD ≤4.3 under identical intervention budgets, without worsening graph calibration or skeleton quality. |
| FrontierPhysics | 56 tasks, 42 physicists, nine domains; best reported tested setup passes 46/168 rollouts (Pass@1 27.4%) under verifier plus expert-blocker evaluation. | Pass@1 ≥35% on the same task/rubric release, with all critical verifier and expert-blocker criteria satisfied and task-clustered uncertainty reported. |
| WorldTest / AutumnBench | 43 environments / 129 queries; reported human performance materially exceeds tested frontier models. | Match or exceed the strongest published model per query family first; then close the human gap. Do not pool query families or use one average as the sole target. |
| FIRE-Bench / Scientist World | Reproduction of verifiable published findings is a separate construct from novel law discovery; Scientist World's initial release reports 1,000 executable paper-derived environments. | Reproduce a preregistered subset at or above the strongest matched baseline before any innovation lift is counted. Publish per-finding results, including failed reproductions. |
| ASI-Bench | Paper snapshot (18 configurations): B1/B2/B3/B4 means 50.91/29.10/26.62/26.99. Current supplied board (25 rows): 54.92/33.11/30.58/30.74; two rows are single-run, task-set ID unavailable. | Improve B3 under matched release/harness/cost, preserve B1, and maintain B4≈B3 for verified non-essential distractors. Keep cold-start, specialist and hybrid tracks separate. |
| Terminal-Bench Science | 70 tasks / five domains; current boards differ by harness. Vals Mini-SWE-agent: GPT-6 Astra 62.86%, GPT-6.1 Sol 52.86%, Opus 5.5 47.14%; Snorkel board: Codex/GPT-6 Astra 68.1%±3.2, Claude Code/Opus 5.5 63.3%±3.3. | Do not pool board snapshots. Pin task release, verifier, model/harness, attempts and resource cap; report cost and human rework separately. |

Do not pursue these stretch values at the expense of evidence validity, selective prediction, or cost. Beating a weak or incompatible baseline is not beating SOTA.

The machine-readable [DiscoverPhysics protocol audit receipt](DISCOVERPHYSICS_PROTOCOL_AUDIT_V0_2026-11-08.json) pins commit 33b7fa9 and exact source blob SHAs. It is status `blocked_on_protocol_parity` and disallows benchmark claims until the world/score/seed discrepancies are resolved. The DiscoverPhysics adapter must preserve its native dual output (natural-language explanation plus executable Python law) and held-out trajectory scoring. The public GitHub README lists 11 predefined worlds, Python 3.9+ and JAX, and a YAML benchmark runner; the leaderboard describes 22 total worlds including 11 private. Run the public 11-world slice as a separately named partial reproduction unless authorized access and exact scorer parity for the private worlds are available. The public README currently specifies explanation score ≥0.75 while the leaderboard text specifies ≥0.9; resolve this against the pinned scorer implementation before any comparison. See https://github.com/SampsonML/DiscoverPhysics/ and https://sampsonml.github.io/DiscoverPhysicsLeaderboard/.

FIRE-Bench's public page currently describes 30 paper-derived research tasks and four stages (planning, implementation, execution and analysis); its listed leading result is 46.7 F1 across three trials. Use it as a separate end-to-end reproduction track, with planning-stage failures reported separately; do not treat its F1 as a law-discovery Pass@k equivalent. See https://firebench.github.io/ and https://proceedings.mlr.press/v306/wang26d.html.

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
| #8487 / #8143 / #8150 | Scientific-discovery benchmark protocol and existing open-world family-selection v0 |
| #6879 / #6880 | Canonical ScientificModel / PhysicsIR, equations, units, assumptions and provenance |
| #6868 / #6869 | Physical typing and explicit unknowns |
| #8286 | Noisy SCM active experiment design and intervention semantics |
| #8588 | Hidden regime transitions and boundary identifiability |
| #8607 | Unknown observation noise, adaptive sampling and calibration |
| #8628 | Law-drift vs measurement-drift diagnosis by costed intervention |
| #8667 | Unit/coordinate invariance and paired canonical-law checks |
| #8669 | Posterior over equivalence classes, support omission, prior-splitting and abstention |
| #8671 | Progressive law discovery, intermediate knowledge and negative transfer |
| #8705 | Invariant mechanisms, environment transfer and change rejection |
| #5476 / #5478 / #8721 | Prospective value-of-computation, realized utility and tool-allocation comparisons |
| #10877 / #11012 / #11015 | Research taste and generic fixed-versus-adaptive orchestration |
| #6874 / #6875 | Proposal qualification / independent scientific acceptance |
| #9332 / #10756 | Sealed benchmark generation and evaluator quality |

This lane owns the matched benchmark experiment and thin adapter only. It does not own a second PhysicsIR, posterior engine, active selector, VoC predictor, scheduler, receipt service or evaluator.

## 11. External references

- Unsatz: https://arxiv.org/abs/2611.03713
- DiscoverPhysics: https://arxiv.org/abs/2605.26087 ; leaderboard: https://sampsonml.github.io/DiscoverPhysicsLeaderboard/ ; pinned protocol audit: DISCOVERPHYSICS_PROTOCOL_AUDIT_2026-11-08.md
- Active Causal Discovery Benchmark: https://arxiv.org/abs/2609.31675
- FrontierPhysics: https://arxiv.org/abs/2611.06910
- WorldTest / AutumnBench: https://proceedings.mlr.press/v306/warrier26a.html
- FIRE-Bench: https://proceedings.mlr.press/v306/wang26d.html
- Scientist World: https://arxiv.org/abs/2611.08478
- AIRS-Bench: https://ai.meta.com/research/publications/airs-bench-a-suite-of-tasks-for-frontier-ai-research-science-agents/
- SciAgentArena: https://www.microsoft.com/en-us/research/publication/benchmarking-ai-agents-for-addressing-scientific-challenges-across-scales/
- ScienceAgentBench: https://hal.cs.princeton.edu/scienceagentbench
- AstaBench: https://allenai.org/asta/bench
- SciLaws-Bench: https://yiyihum.github.io/SciLaws-Bench/ ; paper https://arxiv.org/abs/2609.01552
- Open-world misspecification controls and acceptance gates: https://github.com/Luminous-Dynamics/symthaea/issues/11197
- Evidence-grounded scientific evaluation: https://arxiv.org/abs/2610.04915
- Semantic Calibration for long-form scientific QA: https://aclanthology.org/2026.uncertainlp-main.11/
- ASI-Bench: https://arxiv.org/abs/2608.17271 ; leaderboard: https://asibench.apexin.ai/leaderboard/ ; prompt guidance FAQ: https://asibench.apexin.ai/faq/
- Terminal-Bench Science: https://www.terminal-bench-science.ai/announcement ; Vals AI current board: https://vals-ai.com/benchmarks/terminal-bench-science ; Snorkel board: https://snorkel.ai/leaderboard/terminal-bench-science/

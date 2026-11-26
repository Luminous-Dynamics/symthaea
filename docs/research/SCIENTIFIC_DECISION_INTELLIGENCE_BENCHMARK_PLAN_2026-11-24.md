# Scientific Decision Intelligence: Research and Benchmark Plan

**Snapshot:** 2026-11-25  
**Scope:** Symthaea experiment design, active evidence acquisition, decision-focused sensing, numerical qualification and end-to-end scientific workflow execution.  
**Claim status:** Research plan and engineering status only. No external leaderboard result or physical validation is claimed.

## Executive thesis

The useful unit of intelligence is not a prediction, score or tool call. It is a qualified decision improvement obtained from evidence whose acquisition cost, provenance, uncertainty and consequences are explicit.

The programme should connect three distinct capabilities without collapsing their estimands:

1. **Stage B — hypothesis discrimination:** Which candidate experiment best separates surviving models? The v0.2 joint-covariance planner uses prior-free maximin Mahalanobis separation and cost-aware ranking.
2. **Stage A — decision value:** Can the result change the downstream decision enough to justify acquisition? Use Bayesian expected value of sample information (EVSI) when priors and likelihoods are defensible; use set-valued robust minimax improvement when they are not.
3. **Evidence qualification:** Did the observation actually arrive under the permitted protocol, and can an independent evaluator reproduce the result?

A covariance separation score is not EVSI. Mutual information is not decision value. EIG about a parameter is not the same as uncertainty reduction about a model, and none of these authorizes an external action.

## Current public engineering work

| Workstream | Exact evidence established | Claim ceiling |
|---|---|---|
| Symthaea PR #13453 — joint-covariance experiment design v0.2 | Focused workflow passed at source 735b8cc936cc42372de4c9fc8ee8ee8aff0b2dd0: 29 tests, rustfmt, strict Clippy, demos, JSON contracts and committed lockfile gate | Package-level mathematical/contract qualification; not whole-repo CI, physical covariance calibration or EGB validation |
| Symthaea PR #9821 — decision-focused active sensing | Focused workflow passed at source f203c5e4fd62c985c32414e8c038929f031f2d67: 9 tests, rustfmt, strict Clippy and isolated lockfile artifact | Planner and receipt correctness on synthetic tests; no calibrated prospective forecasts or physical energy result |
| Symthaea PR #10947 — AS-Bench compatibility adapter | Focused workflow passed at source f23caa7b8a177f002ee25eaec34404d4e675946f: 38 tests, rustfmt, strict Clippy, pinned manifest and Lab API fixtures | Offline compatibility/request-response contract only; no live API call, campaign execution, verifier access or official submission |
| Symthaea #12546 — finite-world decision evaluator | Focused workflow passed at source d66d216d77d7b9d70cbbd7e66fa75a4bda3f6738: 16 tests, rustfmt and strict Clippy. Includes Bayesian and prior-free robust decision-value modes plus a typed covariance-to-robust adapter. | Executable finite-world evaluator primitives; full chaotic-systems generators, sealed evaluator process, external predictor tournament and physical sensing remain future work |
| Symthaea PR #13545 — robust risk-gate separation | Draft source change at head `ba0c0b4e2ee6c11879017fb040ebe6899c368416`; separates unconstrained minimax value from current/post-observation risk gates and adds three known-answer tests. | **Unqualified:** formatting, compilation, tests and strict Clippy have not run on this head; no external or physical result is claimed |

The focused work is deliberately split across existing owners. #9817/#9821 own active sensing; #12546 owns the paired-world decision evaluator; #12407 owns observation time/mask/modality semantics; #12502/#12505/#12506/#12624 own dynamical reference and predictor baselines; #5476/#5477/#5478 own prospective value-of-computation calibration and causal metareasoning ablations.

## Paradigm shifts

### 1. From information gain to decision value

An observation can have high entropy reduction and zero downstream value. A more detailed perception or more accurate local model can even cause a worse action. The 2026 DEEP benchmark reports that 34–54% of tested escalations that changed downstream loss made the decision worse in its tested settings. It also finds perception-level gain can disagree in sign with downstream decision value.

**Symthaea principle:** optimize task-loss reduction subject to risk, permissions, deadlines and the complete incremental resource vector. Keep information gain as a diagnostic, not the sole objective.

### 2. Two honest uncertainty modes

Use distinct decision-value modes.

- **Probabilistic mode:** calibrated prior + observation likelihood + frozen loss table → posterior Bayes risk → EVSI. Compute EIG separately when the target is parameter/model uncertainty.
- **Prior-free robust mode:** finite possible-state set + outcome-support sets + frozen loss table → current minimax risk and worst-case post-observation minimax risk. Report the guaranteed worst-case loss reduction; do not invent probabilities.

The new research/decision-predictability-benchmark/src/robust.rs module implements the second mode. It deliberately keeps acquisition cost in its own declared units rather than subtracting it from task loss. A single scalar is justified only when an application supplies a defensible conversion rule.

### 3. From pairwise variance fragments to one coherent joint model

The v0.2 covariance planner accepts one PSD covariance over the stacked hypothesis-output vector, then derives all pairwise difference covariances from that common matrix. This is stronger than letting callers supply mutually inconsistent pairwise covariances. The next step is to bind covariance outputs to observation semantics and observable lineage, not to assume the covariance itself reveals a measurement likelihood.

The new typed adapter in the finite-world evaluator preserves Stage-B experiment identity, method identity, D² and covariance digest next to Stage-A robust decision value. It does not use D² in the Stage-A calculation. This creates a reproducible integration seam without conflating estimands.

### 4. From experiment selection to active scientific work allocation

Candidate actions should include more than wet-lab experiments:
- symbolic derivation or proof attempt;
- analytical limit or dimensional check;
- numerical refinement or solver run;
- independent code path or oracle comparison;
- additional sensor modality or physical measurement;
- replication, ablation or adversarial control;
- external expert review.

Every action needs an explicit hypothesis target, predicted outcomes, decision impact, cost vector, deadline and permission contract. The action planner must not authorize execution merely because it predicts positive value.

### 5. From “the run passed” to evidence-carrying scientific state

Pre-commit the question, model version, covariance/likelihood, random seed and decision rule. Record execution intent, actual observation, clock semantics, raw artifacts and independent verification. Unknown execution outcome must trigger reconciliation, not another mutation. Candidate code must never receive evaluator truth or future observations during confirmatory scoring.

### 6. From fixed benchmarks to independently hardened challenge suites

A public benchmark can become saturated or acquire answer shortcuts even when its original tasks are valid. Use a versioned `concept × recipe × instance` representation and a controlled task-hardening loop: discover shortcuts from permitted traces, revise the recipe without changing the scientific construct, independently revalidate the oracle, then freeze a new recipe version and retain older versions for regression measurement.

Compose with:
- [#9332 — adaptive scientific benchmark generation with sealed oracle verification](https://github.com/Luminous-Dynamics/symthaea/issues/9332) for candidate generation, independent task validation, rotating challenge instances and hidden-oracle custody;
- [#10756 — evaluator research and independent target judgments](https://github.com/Luminous-Dynamics/symthaea/issues/10756) for evaluator construct validity, sealed-target transfer and scorer attack resistance;
- [#12567 — information-channel integrity and non-anticipativity](https://github.com/Luminous-Dynamics/symthaea/issues/12567) for candidate/evaluator separation and metadata side-channel controls.

AutoSciBench (arXiv:2610.05140) is a relevant methodological comparator for adaptive benchmark construction, not a leaderboard score. A benchmark generator must not control the final qualification oracle, its own acceptance thresholds, or sealed labels.

## Current external benchmark landscape

The figures below are public reference points, not results from Luminous Dynamics. Benchmarks are different tasks and must not be combined into one league table.

**Verification note (2026-11-25):** Primary benchmark pages/papers were rechecked. AS-Bench reports six campaigns, three trials per campaign per configuration (18 trials/configuration), 180 total trials across ten configurations, and a 72% best result (72.2% on the leaderboard). The internal 83.3% target therefore means 15/18 per configuration, not 150/180. ASI-Bench's homepage snapshot lists GPT-6.1 Sol/Codex ultra at 63.90, while the leaderboard search snapshot lists GPT-5.6 Sol/Codex ultra at 55.84; these supplied ranking snapshots are unreconciled. The leaderboard page's live result rows were loading when opened. Do not treat either score as the single verified current frontier until the official source artifacts are reconciled.

| Benchmark | Public reference point | Why it matters | Comparability warning |
|---|---:|---|---|
| Autonomous Science Bench 0.1 | Best listed resolution 72.2%; six campaigns; 18 trials/configuration (180 across ten configurations) | Closest closed-loop experiment selection and learning benchmark; separate discovery and held-out learning gates | Pilot release; preserve all six campaign rates and both gates; replay/digital-twin provenance, cost and agent time must be reported |
| ASI-Bench | 60 tasks / 11 domains / B1–B4; homepage snapshot lists GPT-6.1 Sol/Codex ultra at 63.90; leaderboard search snapshot lists GPT-5.6 Sol/Codex ultra at 55.84 | Project-level scientific execution under progressively reduced methodological guidance | These are different supplied ranking snapshots, not one reconciled leaderboard. The live leaderboard page was still loading rows when opened; exact scored-task-set identifier remains unverified. Published evaluation uses no external tools |
| SciWorkBench | Best reported mean 57.38 across 98 tasks | Scientific workflows across biology, chemistry, materials and physics | 13 model–harness configurations; follow the official grader and publish per-domain failure categories |
| FIRE-Bench | Best published F1 46.7 across 30 tasks and three trials; 73.6% of analyzed errors attributed to planning | Verifiable scientific rediscovery with atomic claim precision/recall | Claim-level rediscovery is not physical experiment execution |
| BIABench | 16 real-world tasks, 11 analysis subtasks/modalities; separate outcome and process/QC scores | Raw scientific imagery, analysis procedure, controls and measurement reproducibility across 2D, 3D and time-lapse | Difficult 3D/time-lapse tasks remain weak; repeated-run variance high. Process rubric is not correctness ground truth |
| AstaBench | 2,400+ problems; AI Index 2026 reports best score around 0.53 at about $3.40/problem | Broad scientific breadth and cost reference | Secondary comparison; not comparable to closed-loop experiment-resolution rates; 57 agents / 22 agent classes in the AI Index snapshot |
| Decision-Centric Research / RD-Bench | 2,000 rule-generated cases, 10 of 24 decision types; state-aware DCRM 89.4% Top-1; prediction-set coverage 97.1% | Closest typed decision layer: next action, stopping, evidence sufficiency, state completeness and advisor escalation | Closed candidate options and programmatic labels; model-prediction benchmark, not actual research execution or scientific truth |
| FrontierPhysics | 56 end-to-end physics tasks, 42 physicists, nine domains; three trials/task (168 rollouts/configuration). Paper leader Claude Opus 5.5: Pass@1 27.4% (46/168 rollouts), Pass@3 44.6% (25/56 tasks) | Full physics research: literature, plan, experiments, structured results and manuscript with deterministic verifiers + expert blocker rubrics | Paper's 95% task-cluster bootstrap CI: 17.9–36.9% Pass@1 and 32.1–57.1% Pass@3. Keep Pass@1 and Pass@3 distinct; do not mix verifier-only scores or third-party dashboards |
| Scientist World | 1,000 paper-derived executable environments; 99% reproduced-method fidelity within margin, 82% beat paper baselines across environments, 100% independent verification; in its fixed ten-environment evaluation cohort, 84% Improve and 74% Innovate attempts had positive held-out FNI | Research improvement over executable reproductions of published methods, with held-out metrics and traces | 84%/74% compare against reproduced source methods, not original human authors. The 10-task cohort is not all 1,000 environments. FNI depends on source reproduction, metric weights and frozen frontier normalization |
| WorldTest / AutumnBench | 43 interactive environments, 129 environment-level queries; 517 humans and five frontier models | Tests general world-model quality for reachability, intervention consequences, change detection and planning | Adjacent world-model benchmark rather than science-agent score; humans substantially outperform tested models |
| DEEP | 34–54% of tested decision-changing escalations made downstream loss worse | Shows perception/compute improvements can harm final decisions | Robotics/perception domain; methodology transfers, raw numbers do not |
| ActiveArena | 35 active-perception tasks; best reported aggregate ID/OOD 62.97/47.60 | Multi-round evidence acquisition and memory under distribution shift | Robotic manipulation; not a general science-agent score |
| SynTSBench | NeurIPS 2025; programmable time-series features with theoretical-optimum baselines | Separates forecast skill from mathematical limits and irregularity robustness | Synthetic forecasting benchmark, not closed-loop experiment selection |
| NeuroBench chaotic prediction | 14 precomputed Mackey–Glass parameterizations with Lyapunov-time references | Strong nonlinear forecasting baseline | Reproduce exact data, splits, autoregressive rollout and sMAPE; do not infer scientific autonomy from forecast error |
| NGRC-MDS | 2026 IEEE TNNLS work reports improved/matched NG-RC on adaptive Kuramoto, Lorenz and Mackey–Glass cases | Required strong dynamical baseline for multiscale-delay selection | Reproduce published protocols and matched budgets; do not assume transfer to every domain |

Primary links:
- AS-Bench: https://asb.open-ssi.org/about and https://asb.open-ssi.org/leaderboard
- ASI-Bench: https://asibench.apexin.ai/ and https://asibench.apexin.ai/leaderboard/
- SciWorkBench: https://arxiv.org/abs/2611.22490
- FIRE-Bench: https://firebench.github.io/
- BIABench: https://github.com/BIABench/BIABench and https://huggingface.co/datasets/BIABench/BIABench (DOI 10.57967/hf/10561)
- AstaBench summary in Stanford AI Index 2026, Chapter 5: https://hai.stanford.edu/assets/files/ai_index_report_2026_chapter_5_science.pdf
- Decision-Centric Research / RD-Bench: https://arxiv.org/abs/2611.00774
- FrontierPhysics: https://arxiv.org/abs/2611.06910 (Pass@1/Pass@3 table and task-clustered intervals)
- Scientist World: https://arxiv.org/abs/2611.08478
- WorldTest / AutumnBench: https://proceedings.mlr.press/v306/warrier26a.html
- AutoSciBench (adaptive benchmark-generation method, arXiv:2610.05140): https://arxiv.org/abs/2610.05140
- DEEP: https://arxiv.org/abs/2609.35910
- ActiveArena: https://arxiv.org/abs/2609.24124
- SynTSBench: https://proceedings.neurips.cc/paper_files/paper/2025/hash/d60361f3d0021c76626b2597957261d6-Abstract-Datasets_and_Benchmarks_Track.html
- NeuroBench: https://www.nature.com/articles/s41467-025-56739-4
- NGRC-MDS: https://doi.org/10.1109/TNNLS.2026.3706608

A promising theoretical line for the probabilistic arm is the September 2026 paper *Certified Bayesian optimal experimental design: from Expected Information Gain to Signal-to-Noise Ratio*: https://arxiv.org/abs/2609.32282. It derives upper/lower EIG bounds using dimensional logarithmic Sobolev inequalities and Fisher-information/SNR matrices, recovering exact EIG in the linear-Gaussian setting. These bounds require their assumptions and are not a drop-in interpretation of the current prior-free covariance score.

## Benchmark targets — proposals, not achieved results

### Scientific campaign and research workflow

- **AS-Bench:** target at least 15/18 resolved trials (83.3%) per evaluated configuration on the six-campaign v0.1 suite, with both discovery and held-out learning gates passing for every counted success. Each campaign has three trials per configuration; the paper reports 180 trials across ten configurations and a 72% best result. Report all six campaign-specific rates, both gates separately, replay/twin provenance, cost and agent time. Then test a held-out campaign not used for tuning.
- **ASI-Bench:** target ≥70 overall on the same official task-set/scorer/runtime, subject to reconciling the exact scored-task-set identifier. The latest official homepage snapshot lists 63.90. Also target a 25% reduction in the B1-to-B2 score drop, and at least a 10-point improvement over the strongest matched baseline on B2–B4 without sacrificing B1. Use three independent runs and task-level artifacts.
- **SciWorkBench:** target at least 65 mean under the same official harness and grader, with per-domain breakdowns and explicit procedure/evidence/self-correction failure rates.
- **FIRE-Bench:** target at least 55 F1 on the same 30 tasks, while also reporting precision, recall, trial variance and atomic-claim support. Target fewer than half of errors in planning versus the published 73.6% attribution, not just higher final F1.
- **BIABench:** reproduce v1.0/basic/three-run scoring before tuning; keep outcome and process separate. Target mean outcome ≥0.30 on nuclear-pore assembly kinetics (paper's strongest reported score 0.19) and ≥0.20 on 3D puncta quantification (published task-level reference 0.05), with three independent runs and no best-of-three selection. Target fewer than 15% of same-agent/task pairs with >0.2 run-to-run outcome range (paper reports 40%), without lowering task means. Report all task runs, invalid/refused/undecidable cases, outcome weights/tolerances and total costs. The process score uses a VLM rubric; it is a traceability diagnostic, not correctness authority (the paper reports weak agreement with outcome correctness). Outcome metrics themselves use task-specific weights/tolerances derived from source-study ground truth, so independently audit rubric validity and sensitivity before claiming superiority. The HF card marks evaluation.zip ground truth as evaluation-only, carries a canary, and explicitly prohibits training/fine-tuning or re-hosting; do not expose it to Symthaea or its policy-training pipeline. Use [#13473](https://github.com/Luminous-Dynamics/symthaea/issues/13473).

- **AstaBench:** secondary breadth/cost target: at least 0.60 mean score at no more than $3.40 per problem under the same official task set and scoring version. Report the domain and workflow-stage vector, not only the aggregate.

### Typed research decisions, whole-paper improvement and world-model adequacy

These benchmarks close three gaps between local action selection and a complete scientific researcher.

**Decision-Centric Research / RD-Bench**

The paper defines 24 decision types and instantiates ten in 2,000 rule-generated cases, with a held-out research-domain split. Its 0.6B state-aware DCRM reports 89.4% pooled Top-1 across ten types versus 27.8% for the matched question-only model and 36.4% for DeepSeek-V4.1-Flash. Next-action Top-1 is 72.0%; stopping-condition Top-1 is 100% on this closed candidate set. Calibrated prediction sets report 97.1% empirical label coverage at mean size 1.515. The 72% next-action and 100% stopping results are type-level scores, not end-to-end policy success. This is a strong typed-decision baseline, not evidence of open-ended scientific execution: labels are programmatic, action candidates are supplied, and each task is a bounded choice.

Symthaea target, after exact adapter/scorer parity:
- ≥92% pooled Top-1 across the ten official types;
- ≥85% Top-1 for next-action choice, reporting all ten type rows rather than allowing easy stopping/routing cases to hide weakness;
- ≥95% empirical label coverage on held-out domains with mean prediction-set size ≤1.5, plus reliability curves and risk–coverage;
- zero false-safe outputs on frozen evidence-insufficiency and missing-state controls.

Use a state-aware model, question-only model, and frontier baseline on identical case inputs. Publish paired task-level results, seed variation, type/domain strata, candidate-set size and every refusal/invalid case. Never conflate RD-Bench Top-1 with experiment-resolution rate. Compose its Research State / typed decision interface with [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171), its policy regret evaluation with [#13567](https://github.com/Luminous-Dynamics/symthaea/issues/13567), and its prospective traces with [#11015](https://github.com/Luminous-Dynamics/symthaea/pull/11015).

**FrontierPhysics — end-to-end physics research**

The paper's evaluation covers 56 expert-authored tasks across nine physics domains, with three rollouts per task (168 per configuration). A rollout passes only when all deterministic verifier checks and all blocking expert-rubric items pass. Claude Opus 5.5 leads the paper's main result at 27.4% Pass@1 (46/168 rollout slots) and 44.6% Pass@3 (25/56 tasks with at least one passing rollout). Its task-clustered 95% bootstrap intervals are 17.9–36.9% for Pass@1 and 32.1–57.1% for Pass@3; unscored slots count as failures. Keep these metrics separate from verifier-only rates.

Symthaea target: ≥40% Pass@1 and ≥55% Pass@3 under the exact 56-task / 168-rollout protocol, with every slot accounted for and no selective best-of-three reporting. Report task-clustered 95% intervals, nine domain rows, verifier-only versus blocker-rubric failure, and full compute, wall-clock, tool and human-review costs. Require fresh task-family confirmation. Treat this as a physics research milestone, not a universal science score. Use #11171 for the candidate policy, #11042/#11060 for structure-first discovery controls, and #9332/#10756 for hidden evaluator custody and rubric integrity.

**Scientist World — improvement over executable paper baselines**

Scientist World releases 1,000 paper-derived executable environments. Its environment-construction table reports 99% reproduced-method fidelity within margin, 82% outperforming paper baselines across environments (including those without a reimplementable baseline), and 100% independent verification. The model evaluation was run on a fixed cohort of ten lower-compute environments, not all 1,000: 84% of Improve attempts and 74% of Innovate attempts had positive held-out FNI against the reproduced source method. These figures do **not** compare against the original human authors. FNI is weighted and normalized over a frozen frontier cohort; its value depends on the reproduced baseline and metric choices. The paper reports that 37% of intermediate iterations were reverted after later cross-checks.

Symthaea target: on exact matched tasks and constraints, positive held-out FNI on ≥90% of Improve attempts and ≥80% of Innovate attempts, with median FNI above zero in both modes. Preserve the original question, source metrics, data/model scale and compute constraints. Every counted task must pass independent verification; report protected-metric regressions, all rollbacks, failed iterations and inclusive costs. Use a larger sealed environment cohort beyond the public ten-task evaluation cohort for confirmation; disclose overlap and avoid best-of-many held-out selection. No direct scale increase may count as a research-method improvement.

**WorldTest / AutumnBench — generality of the learned world model**

WorldTest asks whether a model supports queries about environment-level structure and counterfactual consequences, not only properties visible in a particular rollout. AutumnBench contains 43 interactive grid-world environments and 129 tasks across masked-frame prediction, change detection and planning; in the published comparison, 517 humans substantially outperformed five frontier models.

Use this as a secondary test of open-world reasoning and intervention understanding. Target ≥90% of the human-median performance on each query family on the published task split, with separate ID/OOD reporting, and zero false-correct claims on frozen impossible-reachability or intervention-under-identifiability controls. Report actual exploration/query cost and uncertainty. It is not a scientific-paper benchmark; it tests whether the agent's internal world model supports counterfactual queries beyond its observed trajectories.

### Research taste: problem, approach, experiment and interpretation

A closed-loop research system needs more than experimental selection. Compose with [#10877 — research direction, concrete ideas, experimental taste and interpretation](https://github.com/Luminous-Dynamics/symthaea/issues/10877), and keep the components separate.

Two recent comparators make the decomposition concrete:

- **IdeaTaste** (arXiv:2611.03424, 2026-11-02) measures retrospective alignment of research proposals with work published by the same communities in 2026, given a frozen 2025 literature snapshot. Its paper evaluates six models, four areas and >90,000 proposals; LLM populations reach about 26% of human-community level on citation matching and 18% on concrete-idea matching. This is a community-matching diagnostic, not a scientific-truth oracle. Do not penalize a potentially valuable novel idea solely because it does not match a later paper.
- **TasteVal** (arXiv:2610.06824, 2026-10-05) isolates experimental planning with a Researcher proposing experiments and a fixed Coder implementing them. It has eight private tasks, 24 recruited experts, and 40 H100-hour / 120 wall-clock-hour limits. Opus 5.5's experimental-compute multiplier is 2.30× (95% CI 1.15–4.37) on seven scored tasks; one refusal is missing, not scored as zero. This is not a 2.30× quality gain: final normalized performance was 1.14× (95% CI 1.01–1.31) on the seven-task subset, and its doubling trend remains about 14.6 months. Researcher inference-token costs are excluded from the multiplier, so it is not inclusive total-cost accounting.

Symthaea should benchmark five distinct constructs:

1. **Direction taste:** choose important, underexplored and feasible problems with explicit opportunity cost.
2. **Approach taste:** propose a concrete mechanism, measurable intermediate effects, and falsifiable predicted consequences.
3. **Experimental taste:** design sequential tests that distinguish mechanisms and change decisions per complete resource budget.
4. **Interpretive taste:** correctly handle positive, null, negative, noisy, conflicting and invalid outcomes; select stop, pivot, replicate or escalate.
5. **Transferable taste:** preserve these abilities on held-out research families and new domains.

Use paired policy comparisons with a fixed Coder: expert researcher + Coder, frontier-model researcher + same Coder, frozen Symthaea policy + same Coder, fixed/threshold/EIG/EVSI policies, matched-cost random policy, and evaluator-only exact tiny-world oracle. Hold model, Coder, tool permissions, initial evidence, task, context and inclusive limits constant; vary research policy only.

Proposed TasteVal promotion targets: ≥2.5× serial experimental-compute multiplier against the recruited-expert baseline, final normalized performance ≥1.0× that baseline, and ≤1.0× the expert baseline's inclusive total cost after adding Researcher inference, fixed-Coder compute, tools and retries. Run all eight tasks or explicitly report any refusal/incomplete task as a separate outcome; preserve the official missing-data treatment and include an additional all-task sensitivity analysis. These targets extend the published metric rather than claiming that its compute multiplier already accounts for inference cost.

Report separately:
- experimental compute and physical/simulation query use;
- final task quality and scientific validity;
- researcher inference tokens, tool/API costs, analysis and verification costs;
- failed experiments, retries, waiting and human intervention;
- prospective forecast calibration and realized decision regret;
- expert judgments, inter-rater agreement and adjudication burden;
- fresh-family transfer.

Do not select the best of many attempts on held-out outcomes, omit refusals or score missing outcomes as zero without an explicit rule. Retain all task denominators. Human expertise is an important comparator but not an infallible oracle; community matching and LLM judges are secondary diagnostics.

Primary links:
- IdeaTaste: https://arxiv.org/abs/2611.03424
- TasteVal: https://arxiv.org/abs/2610.06824
- [#10877](https://github.com/Luminous-Dynamics/symthaea/issues/10877)
- [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171)
- [#11012](https://github.com/Luminous-Dynamics/symthaea/issues/11012)

### Decision and active evidence acquisition

- Demonstrate at least one preregistered task where decision-focused EVSI outperforms entropy-only or parameter-EIG ranking because the observation changes the action.
- Demonstrate a matched negative control where a high-information observation has zero decision value and is not purchased.
- For prior-free tasks, report guaranteed worst-case decision-loss reduction; never relabel it EVSI.
- At matched information, quality and deadlines, target at least 10% lower decision regret than the strongest eligible policy, with paired confidence intervals excluding zero and no material critical-risk regression.
- Alternatively, maintain decision risk while reducing one physically measured, constrained resource dimension by at least 20% without hidden increases in other resources.
- Charge planner overhead, retries, tool calls, waiting, acquisition, conversion, storage, transmission and human review.

### Chaotic dynamics and temporal systems

- Preserve the official NeuroBench Mackey–Glass task and add the pinned Lorenz, Rössler, Mackey–Glass, Hénon and stable/periodic/stochastic controls in #12546.
- Compare persistence, linear AR, tuned ESN, NG-RC, NGRC-MDS, GRU/LSTM/TCN/Transformer or state-space baselines, and HDC/CfC/LTC variants with matched training/compute budgets.
- Report pointwise error-vs-horizon, Lyapunov-normalized horizon only when the reference exponent is qualified, calibration/coverage, attractor statistics, recurrence and regime occupancy, long-rollout failure and decision regret separately.
- Aim for zero false safe/unsafe classifications on frozen analytic negative controls, at least 95% detection on preregistered positive controls, and a separately declared inconclusive category.

### Numerical physics and experiment design

- Reproduce the published same-theory reference grid, analytic limits, solver conventions and residual definitions before making a superiority claim.
- Compare v0.2 covariance rankings against exact linear-Gaussian discrimination cases, a nested Monte Carlo reference on small problems, a nalgebra eigenvalue oracle, and known singular/PSD-support fixtures.
- Track eigenpair residual, orthogonality error, reconstruction error, covariance PSD violations, singular-support false accept/reject, redundancy invariance, unit/basis invariance and design regret.
- Add covariance lineage and cross-observable basis/Gram semantics through #13472 and Symtropy #7131 before importing real observables.
- No EGB or physical-sensor claim until the actual response/basis, boundary conditions, measurement system and independent validation data are pinned.

## Statistical and engineering protocol

Every external comparison should pin:
- benchmark release, task set, scorer, harness, task splits and licenses;
- model/agent version, system prompt, tool versions and allowed tools;
- training/tuning data, compute and action budgets;
- seeds, attempts, failures, refusals, unsupported tasks and denominator;
- raw output artifacts, candidate decisions, action/observation traces and independent verifier receipts.

Use paired seeds and paired confidence intervals; cluster by task or system family instead of treating adjacent windows as independent trials. Report worst-domain and worst-regime results, not only macro averages. Never call skipped, queued, ungraded or unsupported trials successful.

## Immediate next work

1. Keep the prior-free robust evaluator green with its own known-answer tests.
2. Integrate Stage-B covariance discrimination with Stage-A only through a typed adapter that declares predictive support/likelihood and task losses.
3. Keep robust set-valued and Bayesian likelihood-based modes separate in reports.
4. Connect candidate actions to #9817/#9821 and #5476–#5478 without duplicating schedulers.
5. Add sealed evaluator process isolation and observation-event semantics from #12407.
6. Build a reproducible external tournament using the benchmark matrix above; only then make claims about beating a public reference.

The intended scientific contribution is **decision-quality improvement per qualified evidence budget**, not a higher ungrounded information score.

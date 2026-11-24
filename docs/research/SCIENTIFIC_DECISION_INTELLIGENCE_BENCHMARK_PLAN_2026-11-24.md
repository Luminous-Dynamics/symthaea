# Scientific Decision Intelligence: Research and Benchmark Plan

**Snapshot:** 2026-11-24  
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

## Current external benchmark landscape

The figures below are public reference points, not results from Luminous Dynamics. Benchmarks are different tasks and must not be combined into one league table.

| Benchmark | Public reference point | Why it matters | Comparability warning |
|---|---:|---|---|
| Autonomous Science Bench 0.1 | Best listed resolution rate 72.2%; six pilot campaigns; each final score requires discovery and held-out learning gates | Closest closed-loop experiment selection and learning benchmark | Pilot leaderboard; three trials per task; replay and digital-twin provenance differ; cost/time are part of the result |
| ASI-Bench | 60 tasks / 11 domains / B1–B4; latest supplied overall leader 63.90 | Tests project-level scientific execution as methodological guidance is progressively removed | Latest leaderboard says no external tools; exact scored task-set identifier is unverified; preserve B1–B4 and run counts |
| SciWorkBench | Best reported mean 57.38 | Tests scientific workflows across biology, chemistry, materials and physics | Follow the official harness/scorer; reported bottlenecks include procedural translation, multimodal evidence integration and self-correction |
| FIRE-Bench | Best published F1 46.7 across 30 tasks and three trials | Tests verifiable scientific rediscovery with atomic claim precision/recall | Claim-level rediscovery is not the same as running physical experiments |
| DEEP | 34–54% of tested decision-changing escalations made downstream loss worse | Direct evidence that perception/compute gains can harm decisions | Robotics/perception domain; methodology transfers, raw numbers do not |
| ActiveArena | 35 active-perception tasks; reported OOD gap, with best aggregate ID/OOD 62.97/47.60 | Multi-round evidence acquisition and memory under OOD shifts | Robotic manipulation; not a general science-agent score |
| SynTSBench | NeurIPS 2025; programmable time-series features with theoretical-optimum baselines | Separates forecast skill from mathematical limits and irregularity robustness | Synthetic forecasting benchmark, not closed-loop experiment selection |
| NeuroBench chaotic prediction | Mackey–Glass sequence-to-sequence task with 14 parameterized series and Lyapunov-time units | Strong common baseline for chaotic forecast horizon | Preserve the official task and units; do not infer scientific autonomy from forecast error |
| NGRC-MDS | 2026 IEEE TNNLS early-access work reports matching/exceeding optimized NG-RC on adaptive Kuramoto, Lorenz and Mackey–Glass cases | Required strong dynamical baseline for multiscale-delay selection | Reproduce the published protocols and budgets; do not assume transfer to every domain |

Primary links:
- AS-Bench: https://asb.open-ssi.org/about and https://asb.open-ssi.org/leaderboard
- ASI-Bench: https://asibench.apexin.ai/ and https://asibench.apexin.ai/leaderboard/
- SciWorkBench: https://arxiv.org/abs/2611.22490
- FIRE-Bench: https://firebench.github.io/
- DEEP: https://arxiv.org/abs/2609.35910
- ActiveArena: https://arxiv.org/abs/2609.24124
- SynTSBench: https://proceedings.neurips.cc/paper_files/paper/2025/hash/d60361f3d0021c76626b2597957261d6-Abstract-Datasets_and_Benchmarks_Track.html
- NeuroBench: https://www.nature.com/articles/s41467-025-56739-4
- NGRC-MDS: https://doi.org/10.1109/TNNLS.2026.3706608

A promising theoretical line for the probabilistic arm is the September 2026 paper *Certified Bayesian optimal experimental design: from Expected Information Gain to Signal-to-Noise Ratio*: https://arxiv.org/abs/2609.32282. It derives upper/lower EIG bounds using dimensional logarithmic Sobolev inequalities and Fisher-information/SNR matrices, recovering exact EIG in the linear-Gaussian setting. The bounds require their mathematical assumptions and valid Fisher-information bounds; they are not a drop-in interpretation of the current prior-free covariance score.

## Benchmark targets — proposals, not achieved results

### Scientific campaign and research workflow

- **AS-Bench:** target at least 15/18 resolved trials (83.3%) on the six-task, three-trial pilot, with both discovery and held-out learning gates passing for every resolved campaign. Report replay/twin, cost and agent time separately. Then test a held-out campaign not used for tuning.
- **ASI-Bench:** target at least 70 overall on the same task-set/scorer/runtime release, with no B2–B4 collapse (target at least 65 on each autonomous/distractor tier). Use three independent runs and task-level artifacts; do not compare to the current aggregate unless the task-set identifier is reconciled.
- **SciWorkBench:** target at least 65 mean under the same official harness and grader, with per-domain breakdowns and explicit procedure/evidence/self-correction failure rates.
- **FIRE-Bench:** target at least 55 F1 on the same 30 tasks, while also reporting precision, recall, trial variance and atomic-claim support. Target fewer than half of errors in planning, not just higher final F1.

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

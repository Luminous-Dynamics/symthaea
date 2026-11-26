# Active Scientific Intelligence v2 — Identifiability, Intervention and Evidence

**Status:** standard-library-only analytic reference and frozen two-state fixture; not yet compiled or qualified  
**Parent:** [issue #13691 — SDCB v2](https://github.com/Luminous-Dynamics/symthaea/issues/13691)  
**Reference source:** [scripts/research/sdcb_active_science_v2.rs](../../scripts/research/sdcb_active_science_v2.rs)  
**Integration owner:** [issue #13739](https://github.com/Luminous-Dynamics/symthaea/issues/13739)  
**Production integration:** [issue #11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171)

## 1. Research thesis

Static fit is not scientific identification. An unexcited system can make two different mechanisms observationally indistinguishable; a learner must recognize that limitation, select an intervention that separates surviving hypotheses, and preserve uncertainty until a verified observation arrives.

The v2 vertical slice separates four questions:

1. **Observability:** can a selected sensor reconstruct internal state under the assumed dynamics?
2. **Excitation and identifiability:** did the actual input/sensor schedule expose differences between candidate mechanisms?
3. **Decision value:** does learning the distinction change an action or downstream scientific decision? This belongs to DSSI / the decision-predictability evaluator, not to the observability score.
4. **Evidence validity:** was the observation produced by the authorized episode and probe, and does it satisfy the declared measurement contract? A typed fixture receipt is not authentication.

Full-rank observability does not rescue a passive experiment that never excites the part of the mechanism on which hypotheses differ.

## 2. Implemented reference

The Rust reference has no third-party dependencies. It defines:

- bounded 2–4 dimensional discrete-time linear hypotheses, (x_{t+1}=Ax_t+Bu_t);
- one scalar control channel and one or more candidate sensor rows (C);
- candidate-visible passive observation prefix plus finite probe offers with sensor index, input sequence and three separate integer resource dimensions (query, experiment and measurement);
- candidate-visible hypotheses/probes separated from evaluator-only true-hypothesis ID and noise seed;
- deterministic simulation with bounded observation noise;
- pairwise separation margin over the full candidate set;
- exact one-probe maximin selection plus a non-dominated margin/resource Pareto frontier, componentwise budget admission, deterministic tie-breaking and abstention;
- bounded-noise candidate consistency sets;
- observability and controllability rank controls;
- fail-closed checks for duplicate IDs, malformed dimensions, non-finite inputs, budget violations, unknown probes, unknown evaluator truth and stale/misbound receipts.

The committed known-answer fixture contains two candidate mechanisms with the same (A) and (C), but different (B) vectors. Starting at zero with zero input, both sensor traces are identical. A bounded impulse separates their sensor-0 traces:

- mechanism X: ([0,1,0]);
- mechanism Y: ([0,0,1]);
- bounded noise: (delta=0.05);
- maximin separation margin: (1-2delta=0.9).

The candidate episode now includes the passive observation itself, not only the passive input schedule. Both hypotheses remain compatible with that baseline observation. The selector rejects the no-op probe, exposes the low-margin/low-measurement-cost versus high-margin/higher-cost trade-off on the Pareto frontier, and abstains when experiment budget is unavailable. A deterministic 2–4-state shift-system family tests the same equivalence/intervention behavior at each dimension; separate rank controls verify full-rank observability and controllability where constructed. The evaluator-only simulation then emits a bounded-noise receipt; the candidate consistency set must retain the true model and exclude the other model in this fixture.

**Authored coverage:** 23 Rust test functions, including 128 deterministic noise-seed checks. Rust formatting, compilation, tests and Clippy have not been run on this commit. The traces and margins above are hand-derived from the committed matrices, not CI or model results.

## 3. Important limits

This is a reference evaluator seam, not a complete benchmark runner or production adapter.

- The deterministic shift-system generator now creates two-, three- and four-state observational-equivalence episodes. This is an analytic family, not yet a broad nonlinear or randomized corpus.
- Candidate hypotheses are supplied explicitly. Open-world discovery, hypothesis generation, model-family expansion and an `OTHER_UNRESOLVED` hypothesis remain future work.
- The exact maximin selector optimizes one probe's worst pairwise trace separation. It is not Bayesian EVSI, robust EVSI, multi-step planning or a policy-tree oracle.
- The observation receipt has episode/probe identity and a bounded-noise field, but no signature, custody chain, authenticated sensor identity or cryptographic commitment. Production receipts must come from the independently governed evidence verifier.
- The bounded-noise interval rule is a declared deterministic noise model, not a calibrated stochastic likelihood.
- The rank calculation uses numerical Gaussian elimination with a fixed tolerance. It is not a general proof of identifiability for noisy, nonlinear, hybrid or time-varying systems.

The candidate-visible and evaluator-only Rust structs are an API design boundary, not a process-security boundary. Confirmatory evaluation must put truth, seeds, hidden generator parameters and scoring in a separately permissioned evaluator process.

## 4. Paradigm shifts

### 4.1 Identifiability before confidence
The candidate should output a set of models or equivalence class, not force a unique mechanism label. Score false-unique claims, selective risk and calibration separately from prediction accuracy.

### 4.2 Design interventions around mechanism disagreement
Use candidate-model disagreement to propose interventions, but score whether the observation distinguishes the *decision-relevant* alternatives. Prediction disagreement is an acquisition heuristic, not automatically EVSI.

### 4.3 Keep observability, excitation and intervention distinct
An observable pair ((A,C)) says the state is reconstructible under its mathematical assumptions. It does not mean the actual input schedule excited relevant state dimensions, nor that two candidate models were distinguished by acquired data.

### 4.4 Treat unresolved hypotheses as valid outcomes
If multiple mechanisms remain observationally equivalent under available sensors and intervention budget, preserving that equivalence class is the correct result. More data is not automatically worth buying.

### 4.5 Evidence-gated belief transitions
Compose with EGBT [PR #13698](https://github.com/Luminous-Dynamics/symthaea/pull/13698): frozen hypothesis set → prospective probe proposal → authorized run → independently verified observation receipt → recomputable belief transition. Rejected observations leave posterior, model version and budget unchanged.

### 4.6 Decision-sufficient experiment value
Compose with DSSI [PR #13717](https://github.com/Luminous-Dynamics/symthaea/pull/13717): measure EIG and EVSI independently, include exact small finite-horizon policy value, and do not reward experiments that only distinguish hypotheses with identical utilities for the current action set.

### 4.7 Robustness without invented probabilities
Compose with RPSD [PR #13738](https://github.com/Luminous-Dynamics/symthaea/pull/13738) when a finite ambiguity set of complete utility/likelihood scenarios is defensible but scenario weights are not. Keep this distinct from prior-free minimax loss in [PR #13571](https://github.com/Luminous-Dynamics/symthaea/pull/13571).

## 5. External benchmark landscape and matched targets

External scores are reference points, not Symthaea results. Do not combine unlike benchmarks into one overall score.

| Benchmark | Public reference point | What it tests | Proposed Symthaea target |
|---|---:|---|---|
| Beyond Accuracy (2026) | Website: 20 candidate mechanisms, 200 systems, five rounds. Paper: ten LLM agents; separate belief update, hypothesis discrimination, experiment design and intervention prediction | Mechanism identification and the chain from hypotheses to contrastive interventions | Beat strongest matched LLM and BO/LHS controls on identification and each trace-scored reasoning stage; publish all four stage scores |
| ActiveSciBench / LLM-AutoSciLab (NeurIPS 2026) | 57 enzyme-kinetics tasks and 45 GRN tasks; 35.1% symbolic Chem accuracy, 31.1% exact GRN recovery and 72.5% GRN edge F1; reported 2–5× sample efficiency | Budgeted hypothesis-conditioned active discovery | Target ≥50% symbolic Chem accuracy, ≥40% exact GRN recovery, ≥80% edge F1 and ≥3× sample efficiency at same oracle budget; break out difficulty and seed |
| Autonomous Science Bench 0.1 | Six real lab workflows; public leaderboard lists 72.2% resolution for GPT-6 Astra/Codex at $190.23 and 557 agent minutes | End-to-end design–build–test–learn campaigns, tool harness and cost | Target ≥80% native resolution while retaining every campaign denominator; separately improve cost/time Pareto. Replay/twin results do not substitute for live-lab evidence |
| FrontierPhysics (2026 paper) | 56 research tasks, 42 physicists, nine domains, 168 rollouts/configuration; paper reports 27.4% Pass@1 for its top reported configuration | Literature, research planning, experiments, structured output and expert blocker rubrics | Target ≥35% Pass@1 under the exact paper harness and blocking rubrics; report Pass@3 separately with task-cluster confidence intervals |
| DSSI / decision-predictability evaluator | Internal exact finite references, including EIG-vs-EVSI and branch-unlock counterexamples | Decision value, policy value, regret and budgets | Exact agreement on every known-answer case; ≤5% regret against exact small-world policy value before scaling |
| EGBT / claim-evidence integrity | Internal typed transitions and integrity controls | Whether correctly bound, independently verified evidence alone can change scientific state | Zero accepted transitions for misbound, stale, duplicate, over-budget or invalid receipts; zero mutation on rejection |

### Primary external references

- Beyond Accuracy: https://arxiv.org/abs/2611.29467 and https://schwallergroup.github.io/beyond-accuracy/
- LLM-AutoSciLab / ActiveSciBench: https://scientific-discovery.github.io/llm-autoscilab-project/
- Autonomous Science Bench leaderboard and run protocol: https://asb.open-ssi.org/leaderboard and https://asb.open-ssi.org/run
- FrontierPhysics: https://arxiv.org/abs/2611.06910

These are versioned snapshots. Freeze benchmark-native scorer, task set, budgets, harness and trial denominator before claiming improvement.

## 6. Required tournament

Use the same hidden episodes, initial evidence, candidate set, sensor access and inclusive resource budget for every method:

1. passive/no-intervention baseline;
2. random permitted probe;
3. fixed/space-filling probe sequence;
4. existing HDC prediction-disagreement selector;
5. Spark EIG selector, where its likelihood model is valid;
6. existing Cell Foundry HDC/CfC experiment planner;
7. greedy uncertainty reduction / one-step EIG;
8. greedy one-step EVSI;
9. exact DSSI finite-horizon Bayesian oracle on tiny cases;
10. RPSD finite-ambiguity robust selector where its scenario contract applies;
11. prior-free minimax sequential policy oracle where only set-valued supports are justified;
12. candidate HDC/CfC policy, then the production Symthaea researcher once exact callable and model/runtime digest are pinned.

Use the reuse-first source map in [issue #13739](https://github.com/Luminous-Dynamics/symthaea/issues/13739). Do not create a competing production scheduler or relabel a test oracle as the deployed researcher.

## 7. Promotion gates

**Gate A — analytic correctness:** exact rank, observability, controllability, equivalence and trace controls; 100% on frozen known-answer fixtures.

**Gate B — model-selection safety:** zero false-unique mechanism claims on exact observational-equivalence pairs; 100% true-hypothesis coverage under the declared bounded-noise model; calibrated selective risk on noisy near-equivalence cases.

**Gate C — experiment value:** exact regret on small finite worlds; publish EIG, EVSI, robust gain and minimax loss as separate estimands; never exceed a resource dimension.

**Gate D — evidence integrity:** zero belief transitions on corrupted, stale, duplicate, misbound, unauthorized or unverified receipts; accepted and rejected receipts independently replayable.

**Gate E — generalization:** hold out system families and parameter regimes, not only random episode rows; paired seeds; worst-family results; retain all failed controls.

**Gate F — production:** exact callable, model/runtime/tool digests, process-isolated evaluator, permissioned action offers, verifier-bound receipts and rollback/reconciliation. Synthetic fixtures never count as production episodes.

No SOTA or real-world scientific-discovery claim follows from this reference code or its analytic fixture. Those claims require matched benchmark runs and independent reproduction.

# Intervention-Surface Benchmark: External Validation Plan

**Status:** Research plan, 2026-10-22. No external benchmark results are claimed.

## Why the tiny exact fixtures are necessary but insufficient

The current TCCIP evaluator and pairwise utility interaction matrix test equations on six small deterministic Boolean systems. They are useful conformance fixtures but cannot establish a general advantage: controls are still highly artificial, dynamics have no long time horizon, there are no inferred latent representations, no uncertainty estimates, and no real task architectures. The next goal is external construct validity, not adding more toy scalars.

The research programme should produce one **intervention surface** per model and task: outcomes under individual and selected joint interventions, over state distributions, operating regimes, budgets, and predeclared partitions. Candidate metrics consume this shared observation ledger; no metric gets a private simulator, hand-picked partition, or incompatible prior.

## Recent prior art changes the benchmark target

1. **PEID — Partial Effective Information Decomposition for Synergistic Causality** (Yang, Wang, Zhang; arXiv preprint, 5 May 2026): https://arxiv.org/abs/2605.03267. This explicitly targets unique and synergistic causal information under maximum-entropy interventions, and reports an application to air-quality forecasting. We must compare against it when its definitions, code, and assumptions are reproducibly available. A task-utility finite difference is not a substitute for PEID.
2. **CausalVerse** (NeurIPS 2025 Datasets and Benchmarks): https://causal-verse.github.io/ and https://papers.neurips.cc/paper_files/paper/2025/hash/05962abf920ebddac275d26f83fe118a-Abstract-Datasets_and_Benchmarks_Track.html. Its project page reports about 200k images, 140k videos / 300M+ frames, 24 scenes across four domains, plus access to ground-truth causal variables, structures, and generation processes. It offers a possible bridge from exact toy systems to physical, robotic-manipulation, and traffic settings. It is a causal-representation benchmark, not directly a TCCIP leaderboard; adapters must define a comparable task and intervention estimand.
3. **Integrated information in integrate-and-fire networks** (Danilczuk et al., PLOS Computational Biology, 9 March 2026): https://doi.org/10.1371/journal.pcbi.1014085. The IIT 3.0 study reports that dynamics complexity need not track its Φ values, time constants can matter, and noise can affect the result. This motivates explicitly testing noise, time scale, temporal memory, and task utility rather than assuming a single architecture score captures all of them.

The benchmark must not combine results reported under different estimands/units into one leaderboard. For each baseline, state exactly which outcome it predicts or decomposes.

## Benchmark architecture

### Track 0 — exact semantics and falsification (already scaffolded)

- Schema-v3 causal rules, task target, input-state weights, component partition and individual intervention definitions.
- Exhaustive exact task utility for individual and simultaneous independent fair-coin edge cuts.
- Six toy cases for useful XOR, irrelevant connectivity, harmful coupling, prior reversal, and role-matched XOR information.
- Full partition profile plus a 16-row pairwise-intervention oracle.
- Entropy / marginal MI / joint MI reported separately from utility contrasts.

**Qualification limitation:** Rust and committed Python tests still need execution. The independent audit performed during this work was a separate, one-off exhaustive checker, not a compilation or execution of either committed test suite.

### Track 1 — generated structural causal models (SCMs)

Generate held-out model families rather than randomly splitting near-duplicates. Include XOR/parity, AND/OR, common causes, colliders, redundant paths, delayed paths, noisy gates, feedback converted to finite-horizon rollouts, harmful coupling, and sparse/dense mechanisms. For every model, save ground-truth graph/mechanisms, observational samples, permitted intervention set, target utility, prior, and expected counterfactuals.

Split by **mechanism family and graph topology**, not just random seeds. Publish a hidden test set to avoid tuning partition selection and interaction thresholds to its answers.

### Track 2 — configurable physical / robotic / traffic worlds

Build an adapter around CausalVerse's available ground-truth latent variables and causal generation process. Start with one tractable dynamic scene (e.g. spring/collision or robotic manipulation) and expose a small, versioned control/task API. For each task, define:
- state abstraction / causal variables and the validated mappings to simulator state;
- task utility (J), horizon, reset procedure and resource budget;
- legal interventions, including whether a mechanism is disabled, randomized, delayed, or replaced;
- environment split (known dynamics vs held-out parameter, appearance, or domain shifts);
- observable vs privileged state, so method comparisons disclose what causal ground truth they use.

Do not compare our task-utility metric to CausalVerse's published MCC/R² directly; those measure representation recovery. Treat them as one orthogonal track. The new track should evaluate utility/counterfactual prediction using the available simulator interventions.

### Track 3 — a real Symthaea task

Only after Tracks 0–2 qualify, connect a real planning/control component. A proposed first target is a deterministic planning benchmark with controlled edge/message loss; later add a dynamic Symtropy task with recovery and transfer. Keep metric output advisory and non-controlling until evidence supports its use.

## Shared intervention observation ledger

Every row should identify:
- immutable model / mechanisms digest;
- task, utility definition, state-prior digest and environment ID;
- source revision, model version, random seeds and rollout horizon;
- intervention IDs and exact joint-cut set;
- intact utility, post-intervention utility, utility loss, resource cost;
- whether outcome is exact, simulated, or estimated; sample count and uncertainty;
- intervention support / matching quality and all exclusions.

A summary metric is computed only from the ledger. Raw intervention outcomes remain the primary evidence. Never infer causal meaning from ordinary feature masking unless the operation has known intervention semantics.

## Baselines to include

| Question / target | Required comparison |
|---|---|
| Does an edge change task utility? | Raw per-edge intervention effect, paired with intact utility |
| Are pair effects additive? | (Delta^2J_{ij}=J(arnothing)-J(i)-J(j)+J(i,j)), plus additive / no-interaction null |
| Is information carried jointly? | H(Y), individual and joint MI; a stated PID definition; SURD where compatible |
| Is synergistic causal information present? | PEID if implementation/details allow same-model, same-prior comparison |
| Does causal structure recover? | Precision/recall, structural Hamming distance, intervention prediction error; appropriate graph-recovery baselines |
| Does the system resist shifts? | Held-out task utility, regret, failure probability, calibration, and recovery |
| Is this worth computing? | Runtime, memory, tested-pair count, intervention/sample budget and accuracy-cost Pareto frontier |
| Is a formal IIT quantity recovered? | PyPhi under one named, pinned formalism; never treat it as a task-utility gold standard |

## Proposed benchmark gates

These are acceptance proposals, not achieved results.

- Exact small-system quantities: absolute error (le 10^{-12}) wherever the rational expected value and arithmetic permit that tolerance.
- All joint-cut expectations and all six current task/prior fixture summaries match the independent oracle.
- Interaction contrast recovers the declared null, saturation, and compound-failure fixtures with correct sign.
- No unexplained false-positive task-causal credit on disconnected / no-op controls; report false-positive rates across generated null systems, not one example.
- 95% uncertainty-interval coverage between 90% and 98% on simulated, known-truth cases; predeclare estimator and resampling unit.
- On held-out model families, report MAE/RMSE, maximum error, rank correlation, calibration, and error-versus-cost. Do not gate on correlation alone.
- Incremental value required: the intervention surface/profile must improve prediction or architecture decisions beyond simple per-edge effects and pairwise (Delta^2J), or drop the aggregate.
- External-scale target: at equal error and coverage, aim for (ge 10	imes) lower runtime or memory on a declared workload; do not claim this until measured on identical hardware and inputs.

## Failure criteria and anti-gaming rules

- If the profile does not add predictive or decision value over raw edge effects / pairwise differences, remove the TCCIP aggregate rather than defending it.
- If an estimator selects partitions, pairs or thresholds using held-out outcomes, mark the result invalid.
- If a control edge differs materially in target role, fan-in, centrality, delay, or intervention noise, report the mismatch and do not call it fully matched.
- If the candidate flips under a prior or task shift, report a conditional result and never advertise a prior-free intrinsic score.
- If the run lacks source/model/task/intervention identity or has missing strata, return NOT QUALIFIED.
- Do not infer consciousness from task utility, causal synergy, entropy, TCCIP or (Delta^2J).

## Immediate execution sequence

1. Finish the pairwise expectation checks and record exact test counts / execution limitations.
2. Run the committed Python reference tests in an environment with the files; run Rust when a toolchain is available; compare machine-readable outputs.
3. Add an SCM generator with deterministic seeds, family-level splits, and a no-interaction additive control.
4. Implement one CausalVerse adapter on ground-truth dynamics—not the rendered video as a proxy—and specify the task-utility estimand.
5. Compare raw edge effects, pairwise (Delta^2J), PID/SURD, PEID, and applicable causal-graph baselines on the *same model, prior and interventions*.
6. Publish error/cost frontiers and failure cases. Only then decide whether there is a new method or merely a good measurement harness.

## Exact manifest audit performed during development (2026-10-22)

A separately written, standard-library-only exhaustive checker (distinct from `tccip_reference_oracle.py` and the Rust evaluator) was run against local copies of both current TSV manifests. It recomputed individual-cut utilities, TCCIP summaries, every usable partition and the 16 pairwise simultaneous-cut utilities / \(\Delta^2J\) contrasts. All six fixture summaries, all partition counts/ranges, and all 16 pairwise rows matched their declared expectations. The audit produced these input-file SHA-256 digests:

- `tccip_toy_expectations.tsv`: `94cce47ae396e2dee7ce742a039ec8a09f3544293b9bf4a28cf4317eccaf1b7f`
- `tccip_pairwise_utility_interactions.tsv`: `a426a8cfd14a9ee3e2073a18f45439024064945480c2caeba60fa7073bafe9b4`

A reusable checker has now been committed at `validation/tccip_independent_check.py`; its standard reproduction command is in `validation/README.md`. **The digests above bind the copied manifest bytes used by the one-off audit; the newly committed checker has not yet been executed as committed.** This does not replace source-bound test/CI receipts.

## Evidence status

This document is a research plan. CausalVerse and PEID are prior art to evaluate; no benchmark adaptation has been implemented yet. The exact toy manifest and pairwise oracle are authored. The independent finite-state audit run during the current session matched the six task-summary and partition oracles and the XOR pair-contrast expectation. The committed Rust tests have not been compiled, and the committed Python tests have not been run. No SOTA or novelty claim is made.

# SenseBench SCM Graph Recovery Protocol v0.1

**Status:** implementation committed to the public `feat/symthaea-sense-observation-envelope` branch; authored Rust tests have not been executed in the present environment.

## Question

Does active measurement selection improve recovery of causal graph structure under mechanism shift, compared with a strong fixed action and simple baselines? Does the system know when it has enough evidence to submit a graph, and what does it cost to reach that threshold?

This advances the earlier `blind-worlds` suite. That benchmark compares actions among three causal explanations described by hand-authored likelihood rows. The SCM-graph suite instead creates explicit graph hypotheses, generates exact outcome distributions from structural equations, samples intervention outcomes from a separate hidden world, updates a posterior over graphs, and scores the graph that would be submitted.

## Reproduction

Start with an exploratory run:

```sh
SENSEBENCH_CODE_REVISION="$(git rev-parse HEAD)" \
  cargo run -p symthaea-sensebench -- --suite scm-graph --trials 32 --budget 8 --seed 42
```

The workspace CLI emits versioned JSON and requires an exact caller-provided `SENSEBENCH_CODE_REVISION`. Use a clean worktree and record that SHA with the configuration. Do not interpret a queued or skipped workflow as a passing build/test.

## Graph catalogue and worlds

- Enumerate all 25 labeled directed acyclic graphs on three observed variables. Every graph is checked for acyclicity; the catalogue size and unique edge masks have a known-answer test.
- Generate seven graph-family templates: chain and fork (development); collider, sparse random DAG, dense random DAG, random motif, and empty-graph negative control (held out).
- Each of the seven graph-family templates produces four deterministic graph variants. Keep each exact graph instance fixed while crossing it with four mechanism regimes: weak positive effects and strong positive effects (development), plus mixed-sign effects and near-null effects (held out). This gives 28 topology instances and 112 graph × mechanism worlds.
- The held-out partition is factorized: a world is held out if either its graph family or mechanism regime is held out. This yields 16 development worlds (two development topology families × four topology variants × two development mechanisms) and 96 held-out worlds. The same topology instance appears under all four mechanism regimes, enabling paired comparisons that isolate mechanism shifts. The fixed action is tuned only on the 16 development worlds.
- Generate graph topology and mechanism parameters from separate deterministic RNG streams so graph-generator control flow does not influence parameter draws. The true graph and hidden mechanism parameters are not inputs to the policy; the exact source remains inspectable by benchmark auditors.
- All three-variable DAGs are included in the hypothesis catalogue. The held-out split tests shifts among declared topology/mechanism generation regimes, not graph sizes beyond the catalogue. Graph structures and likelihoods remain visible in source and the split is not cryptographically sealed.

The graph models are binary structural causal models. Mechanism regimes have separate amplitude/sign distributions and are crossed with every graph family, specifically to reduce a shortcut in which topology-family labels deterministically reveal the coefficient regime. For each full state, each non-intervened variable's conditional probability is computed from the declared intercept plus the weights of active parents, clamped to the explicit range [0.01, 0.99]. Under a hard intervention, the intervention variable is fixed to the assigned value and its structural equation is bypassed. The evaluator enumerates every three-bit outcome exactly to construct likelihood distributions; each policy's observed state is sampled from the hidden graph and hidden parameter set.

This is deliberately a small, auditable graph-recovery substrate. It does not yet use continuous linear-Gaussian SCMs, latent confounders, temporal graphs, nonlinear mechanism libraries, or larger graphs.

## Policies

All policies share the same initial uniform graph prior, nominal posterior-update rule, observation format, outcome generator, and measurement-count budget.

1. Nominal expected information gain.
2. Predictive outcome entropy, a simple uncertainty-seeking baseline that does not directly score hypothesis discrimination.
3. Model-weighted expected information gain across the declared scenarios.
4. Worst-case robust information-gain selection.
5. Development-fixed action chosen by mean one-step expected information gain across the four development graph-family × mechanism-regime combinations only; it does not use the development worlds' truth graph IDs and never inspects held-out worlds.
6. Uniform random measurement/intervention selection.
7. Passive observation only.
8. Known-mechanism oracle: uses the hidden world's mechanism parameters while retaining the graph-hypothesis task. This is an optimistic upper-bound diagnostic and is not a deployable method.
9. Structure-blind random graph submission, with no measurements. This supplies a graph-recovery floor separate from the random-action baseline.

The current comparison deliberately keeps the posterior update on the same nominal likelihood model for the action-selection policies, to isolate acquisition differences. Consequently, model-weighted/robust selection is not also receiving a privileged model-averaged posterior update. A future factorial ablation should vary acquisition and inference separately.

## Outcomes and metrics

For each world/policy pair, the JSON report includes:

- **Skeleton F1:** adjacency recovery ignoring edge direction.
- **Directed-edge F1:** exact arrow recovery; a reversed arrow counts as a directed false positive and false negative.
- **Structural Hamming distance (SHD):** one edit per unordered variable pair whose state differs, including one edit for a reversal.
- **CPDAG skeleton F1, compelled-direction F1 and edge-mark distance:** score the highest-posterior-mass Markov equivalence class, not an independently thresholded collection of arrows.
- **Markov equivalence-class recovery and posterior class mass:** distinguish learning the observationally identifiable class from selecting one exact DAG inside it.
- **Separate full-DAG and CPDAG claim coverage / false-claim rates:** a confident class-level claim can be appropriate while a single direction remains unresolved.
- Exact graph recovery rate and its Wilson interval.
- Posterior mass on the true graph and multiclass graph-posterior Brier score.
- Graph-claim coverage at the configured posterior threshold.
- False graph claim rate among accepted claims and across all trials.
- Mean measurements/interventions used and counts to threshold when the threshold is reached.
- Per-world metrics, trial-level intervals, and held-out policy summaries. World-level summaries weight generated worlds equally instead of pooling trials as if every observation were an independent world.
- Factorized shift report: 96 any-held-out world rows (28 topology instances), 40 topology-only-shift rows (20 topologies), 16 mechanism-only-shift rows (8 topologies), and 40 joint-shift rows (20 topologies).
- Aggregate intervals first average held-out mechanism regimes within each exact graph topology, then quantify variability across topology instances. Four regime rows sharing one exact graph are not treated as four independent topologies. The post-run result includes each evaluator world ID, truth graph ID and truth edge count for replay/audit; these are evaluator outputs and are not inputs to the measurement policy.

The structure-blind random graph policy reports graph metrics but makes no confidence-qualified claim. Keep its no-measurement behavior separate from the random intervention policy.

Markov-equivalence classes are precomputed for the 25 DAGs; the known-answer catalogue contains 11 classes. Two DAGs are treated as equivalent when they have the same skeleton and the same unshielded colliders, the standard criterion for the causal-sufficiency/faithfulness setting represented by this synthetic benchmark. The predicted CPDAG is the class with greatest summed posterior mass. This keeps the emitted partially directed graph within the catalogue of valid equivalence-class CPDAGs instead of independently thresholding edge directions into a potentially invalid PDAG.

The graph-posterior Brier score scores the posterior over DAGs, not the CPDAG class distribution.

## Negative and known-answer controls

- Exactly 25 unique DAG masks are generated.
- Every generated graph passes the acyclicity check.
- Exact binary SCM outcome distributions sum to one within numerical tolerance.
- Hard interventions assign zero probability to states inconsistent with the imposed value.
- An exactly reversed single edge has skeleton F1 1, directed-edge F1 0, and SHD 1.
- The empty graph exactly matches itself with F1 1 and SHD 0.
- The empty-graph held-out family is retained as a null causal control.
- Repeated runs with identical configuration are required to be byte-stable.

These are authored tests, not executed results. The benchmark has not been compiled or run on the current source head in this environment.

## Proposed qualification gates (not measured capabilities)

1. **Exactness:** all DAG catalogue, distribution normalization, intervention, and structural-metric known-answer tests pass.
2. **Budget fairness:** all actionable policies use the same observation format and measurement budget; report actual costs if non-unit costs are introduced later.
3. **Held-out graph recovery:** compare directed F1, skeleton F1 and SHD against the fixed-design, random-action, passive-only, and structure-blind graph baselines. Do not claim superiority unless paired differences and uncertainty support it.
4. **Graph confidence:** report graph-claim coverage beside false-claim risk. Reject the trivial policy that wins by abstaining on every trial.
5. **Negative controls:** on the empty-graph family, report spurious-edge frequency and false accepted graph claims explicitly.
6. **Factorial generator audit:** seven graph-family templates each generate four graph variants, and each exact graph is evaluated under four mechanism regimes using independent random streams. Confirm that each topology appears under every regime and that the split is 16 development / 96 held out. This reduces the direct graph-family-to-coefficient shortcut, but the seven graph templates remain hand-authored.
7. **World/family inference:** use many independently generated graph families and parameter draws from preregistered distributions. Analyse topology-family and mechanism-regime generalization separately; do not treat 28 crossed combinations as 28 random samples from the universe of SCMs.
8. **Observational equivalence:** the current suite now has a three-variable CPDAG-class scorer and separate compelled-direction metrics. Expand and cross-check it against a trusted CPDAG implementation before larger-graph use; then score interventional direction recovery separately.
9. **Richer SCMs:** add continuous linear-Gaussian and nonlinear/additive-noise families, measurement noise, sampling constraints, intervention artifacts, latent confounding, and selection/censoring mechanisms.
10. **Independent custody:** move the evaluation truth and score implementation to a separate process/corpus, bind test corpus hashes to the protocol, and keep policy-visible fixtures separated from the qualification set.
11. **Physical bridge:** only after software correctness and reproducibility, use replayed sensor streams with injected faults and an independently calibrated reference. Synthetic success alone does not support physical scientific claims.

## Research grounding

- Deb, Shah and Krishnan, [Active Causal Discovery Benchmark: Evaluating LLM Agents Under Budgeted Interventions](https://arxiv.org/abs/2609.31675) (2026), motivates separate graph-structure recovery and intervention-efficiency scores. Its reported results caution against assuming that raw agent complexity beats simpler statistical baselines.
- Herman, Wahl, Ninad and Runge, [Unitless Unrestricted Markov-Consistent SCM Generation: Better Benchmark Datasets for Causal Discovery](https://proceedings.mlr.press/v275/herman25a.html) (2025), discusses artifacts in common synthetic causal graph generators and motivates adversarial generator audits.
- Göbler et al., [causalAssembly: Generating Realistic Production Data for Benchmarking Causal Discovery](https://proceedings.mlr.press/v236/gobler24a.html) (2024), provides a semisynthetic complement once a real process dataset and provenance are available.
- Tang, Sloman and Kaski, [Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html) (AISTATS 2026), motivates the separate SENSE-EXP-002 question of whether acquisition should account for representativeness and model-error amplification in addition to information gain: https://github.com/Luminous-Dynamics/symthaea/issues/7825.

## Non-claims

A passing software test validates this implementation's stated behavior; it does not establish causal discovery in real environments, calibration of physical sensors, causal identification under latent confounding, superiority over state-of-the-art discovery methods, or scientific novelty. The three-node graph universe is a controlled test bed, not the final leaderboard.

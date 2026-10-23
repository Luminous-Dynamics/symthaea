# Scientific-Law Discovery Under Adaptive Design and Misspecification

**Benchmark protocol v0.1 — proposal only; no comparative results are claimed.**

## Objective

Evaluate whether a scientific-discovery system returns a validated law or an honestly unresolved result—not merely a low residual, a small candidate set, or a confident singleton.

The target loop is: generate physically admissible hypotheses; choose an experiment; record a predictive distribution before the outcome; test candidate fit and predictive adequacy separately; expand the hypothesis family after falsification; then validate any replacement on fresh observations. A monitor alarm is evidence against a forecast under its assumptions, not proof of any replacement. No alarm is not proof of adequacy.

## External comparison registry

| System or benchmark | Strength to compare against | Role in our evaluation |
|---|---|---|
| [PySR / SymbolicRegression.jl](https://github.com/MilesCranmer/PySR) | Practical configurable evolutionary symbolic regression, custom operators and complexity penalties | Primary regression baseline on matched data, grammar and compute budgets |
| [AI Feynman](https://doi.org/10.1126/sciadv.aay2631) and [AI Feynman 2.0](https://proceedings.neurips.cc/paper/2020/hash/33a854e247155d590883b93bca53848a-Paper.html) | Physics-informed symmetry, separability, modularity and recursive decomposition | Physics-informed baseline where assumptions and operators align |
| [GenSR (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html) | Learned latent equation space, coarse localization, then evolutionary fine search | Comparator for using continuous latent neighborhoods rather than directly mutating discrete ASTs |
| [EGG-SR (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html) | Equality-graph symbolic equivalence to reduce duplicate search and aggregate equivalent expressions | Comparator for equivalence-aware canonicalization and search-efficiency gains |
| [Exhaustive Symbolic Regression with MDL](https://doi.org/10.1098/rsta.2024.0584) | Bounded search over a declared operator basis, parameter fitting, and minimum-description-length model selection | Strong small-grammar reference; tests whether stochastic search is necessary |
| [SRBench](https://github.com/cavalab/srbench) and [SRBench++](https://doi.org/10.1109/TEVC.2024.3423681) | Broad regression comparison plus task-specific tests: exact form, feature selection, deceptive shortcuts, extrapolation, noise, interpretability | Standard retrospective regression track and robustness suite |
| [ERBench](https://equation-discovery.ti2.fmi.uni-jena.de/index.html) and [ERBench paper](https://arxiv.org/abs/2606.09276) | Equation-recovery suite with public training problems and a hidden evaluation set; emphasizes the distinction between in-domain fit and OOD generalization | External equation-recovery track and leaderboard target; pin benchmark version and report its official metrics separately from custom active-discovery results |
| [TruthInsightBench (2026)](https://arxiv.org/abs/2609.05079) | Blind open-ended scientific-analysis tasks scored from evidence artifacts, including controls, robustness, falsifiability and cross-dataset generalization | Supplementary audit for claim quality and evidence maturity in an agentic discovery system; not a substitute for exact equation-recovery or controlled symbolic-regression metrics |
| [SciLaws-Bench](https://yiyihum.github.io/SciLaws-Bench/) | Separates prediction, scientific validity and structural recovery, including active-query tasks | Science-specific endpoint taxonomy and active-discovery track |
| [RESTART (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/hash/72416ded78a439907ff72165ac9c56e0-Abstract-Conference.html) and [official code](https://github.com/Liyunlun/RESTART) | Boosting identifies unexplained signal; an LLM proposes targeted equation refinements; successful structures are stored for reuse | Direct comparator for post-alarm structured correction; align access to LLMs, library memory, data, and compute |
| [Influence-Guided Symbolic Regression (IGSR, 2026)](https://arxiv.org/abs/2605.29184) | LLM-generated basis terms, per-term influence feedback and Monte Carlo tree search | Hybrid-search comparator; count all model/tool costs and pin source/version |
| [Deliberate Evolution (ICML 2026)](https://proceedings.mlr.press/v306/pang26f.html) | Separates LLM proposal from search control using adaptive operators, analytical diagnosis and reflective memory | Agentic search-control baseline under matched model and token budgets |
| [Bayesian Symbolic Regression with Entropic RL (UAI 2026)](https://proceedings.mlr.press/v337/boussif26a.html) | Bayesian view of expression search to represent uncertainty over symbolic forms | Comparator for uncertainty-aware candidate ranking and diversity |
| [LLM-AutoSciLab / ActiveSciBench](https://scientific-discovery.github.io/llm-autoscilab-project/) | Closed loop of hypothesis generation, experiment selection and mechanism refinement | Agentic discovery baseline where models, tools and budgets can be aligned |
| [LLM-SRBench](https://proceedings.mlr.press/v267/shojaee25a.html) | Symbolic-reasoning and transformed-equation tasks intended to reduce simple memorization | Held-out robustness track with pinned prompts, model versions and query budget |
| Sparse regression / SINDy-style methods | Efficient recovery when the feature dictionary is known or well motivated | Simple strong baseline on dynamical systems; not a universal symbolic-regression comparator |

Published scores are properties of the exact benchmark version, split, operators, priors, tools and budget in each paper. They are not directly comparable with results from Symthaea's small synthetic simulator.

## Tiered benchmark

### Tier A — correctness and negative controls

Test dimensional inference, finite outputs, deterministic seeds, candidate deduplication, measurement-cost accounting, and complete provenance. Include deliberate protocol violations: hidden-truth access, current-outcome leakage into a forecast, reused validation measurements, invalid predictive CDFs, dimensionally invalid recovery proposals, and acceptance after terminal uncertainty. These must result in typed rejection or `unresolved`, never a silent fallback.

### Tier B — controlled structural recovery

Report two task modes separately: **oracle-present discrimination**, where the reference-equivalent law is deliberately included in the hypothesis pool; and **truth-absent recovery**, where the initial pool is generated independently of the hidden reference and is evaluator-audited only after the policy ends. Never pool their success rates. In truth-absent trials, the selector, candidate generator, recovery policy and stop rule must not receive the hidden expression or truth-derived labels.

Start with at least 60 held-out structural worlds across six or more task families and at least 20 paired seeds per world. Include exact in-class laws across expression depths; dimensionally valid near-misses; coefficient-identifiability traps; irrelevant and correlated features; deceptive shortcuts; interpolation versus extrapolation; and observationally equivalent laws that cannot be uniquely identified.

Keep at least three evaluator-only regimes distinct: (1) IID/in-domain, (2) interpolation with held-out combinations or feature correlations, and (3) true range extrapolation where each relevant feature is outside its discovery interval. For range extrapolation, include lower-tail, upper-tail, mixed-tail and scale-shift cases; generate evaluation points from an independent seed; never pass them into candidate generation, parameter fitting, EIG, stopping, or monitor calibration. Log how many reference-valid OOD points were available; do not replace invalid OOD points with in-domain data. Where the reference law itself is undefined or numerically unstable, record the case as not evaluable under the declared domain rather than silently scoring it as successful.

Report exact recovery only when a unique structure is identifiable. Otherwise score recovery of a declared equivalence class. Also report held-out prediction, complexity or description length, unit consistency, OOD prediction, and uncertainty separately. For external equation-recovery comparisons, include the versioned ERBench protocol where feasible and report official leaderboard metrics separately from our active-discovery simulator.

### Tier C — misspecification and active design

Include predeclared cases for omitted additive terms, multiplicative corrections, missing interactions or latent variables, context-conditional shifts that cancel in aggregate, regime shifts, heteroscedastic or heavy-tailed noise, wrong predictive uncertainty, and extrapolation/domain shift.

Keep scientific-law misspecification separate from measurement-distribution misspecification. A PIT alarm establishes evidence of forecast miscalibration only under its monitoring assumptions; it does not identify a missing physical mechanism.

For each world, keep its law, noise process, variable units, admissible actions, domain split, candidate prior, severity scale and budgets in evaluator-only artifacts. Do not expose hidden truth or discrepancy labels to the selector, proposal generator, threshold policy or stopping rule.

### Tier D — recovery after falsification

Compare: stop and return unresolved; restart the original candidate pool; bounded typed reseeding; structured additive corrections; multiplicative corrections; interaction corrections; fitted scalar coefficients versus fixed coefficients/grid search; scenario-robust EIG; and bounded exhaustive/MDL reference search where feasible. Record fitting cost and parameter count. Fit only on discovery observations, never the evaluator-only holdout.

Recovery may use existing observations to propose hypotheses, but a replacement must be evaluated on fresh outcomes. Previously used measurement points remain consumed. Record rejected model IDs, proposed family, dimensional checks, reason for alarm, validation evidence, and final status. An alarm followed by a guessed law is not a successful recovery. A corrected structure with fitted parameters is a new combined structure-plus-parameter hypothesis, and its validation must evaluate that combined hypothesis on fresh points; never treat the coefficient-fit data as independent confirmation.

### Tier E — real-world scientific use

After controlled tiers pass, use reproducible public datasets and expert-reviewed cases. When ground truth is unknown, report predictive performance, stability, calibration, physical constraints and expert assessment—not exact-law accuracy. Preserve data licenses, provenance, and limits on causal interpretation.

## Baselines and assumption checks

At minimum compare random/round-robin selection, nominal EIG, scenario-robust EIG with a frozen scenario set, scheduled random audits, and independent-batch conformal or rank-based audits where calibration assumptions hold. Include finite-horizon sequential monitoring, predictive-rank martingales only under their stated exchangeability conditions, and prequential PIT only when a valid conditional predictive CDF is issued before every outcome. An oracle design or candidate pool may appear only as a labelled upper bound.

Adaptive EIG residuals are not automatically exchangeable. A fitted plug-in CDF is not automatically the true conditional predictive distribution. Invalid assumptions must produce an explicit `invalid_protocol` outcome rather than a misleading alarm-rate result.

Ablate typed generation, experiment selection, adequacy monitoring and recovery independently. Use the same worlds, candidate pools, seeds and noise streams across methods wherever feasible.

## Required endpoints

**Discovery:** exact recovery where identifiable; equivalence-class recovery; held-out and extrapolation error; relevant-feature recovery; dimensional/physical validity; complexity or description length.

**Decision quality:** false confident singleton rate; clean-null false-alarm probability; detection sensitivity by misspecification family and severity; median and 90th-percentile detection delay; calibration; valid recovery after alarm; unresolved rate; fraction of alarms yielding a validated replacement.

**Efficiency:** measurements and measurement cost; total experiments to valid recovery or explicit unresolved outcome; CPU/GPU time, wall time, peak memory, model evaluations, search expansions, and variation across seeds.

**Trust:** hidden-label leakage results; invalid protocol count; rejected-proposal counts; source/config/data digests; exact commit and toolchain; benchmark version, hardware, seeds and per-run machine-readable receipts.

Report per-family and per-severity breakdowns. No single aggregate score should hide failure on extrapolation, context-conditional shifts, or protocol negative controls.

## Statistical protocol and target gates

Pre-register hypotheses, alternatives, query/compute budgets, thresholds, baseline versions, held-out split and analysis before opening the held-out outcomes.

- Use at least 10,000 independently generated clean-null streams for a serious sequential false-alarm audit; publish binomial confidence intervals and assumptions. Smoke tests are regression checks, not confirmation.
- For structural worlds, use paired uncertainty intervals and resample worlds first, then seeds within worlds. Keep comparisons paired by world and seed.
- Publish sensitivity curves over preregistered discrepancy sizes, and null controls using the same adaptive design machinery.
- Publish raw per-world and per-run records, not only pooled summaries. Correct for repeated comparisons or label secondary analyses exploratory.

Initial acceptance thresholds are **targets, not current results**:

1. Clean-world structural recovery declines by no more than 5 percentage points versus the strongest matched baseline.
2. False confident singleton rate falls by at least 20% relatively on the preregistered misspecification mix, with a nonzero baseline and a paired uncertainty interval supporting the effect.
3. Sensitivity reaches at least 80% on practically important alternatives designated before held-out evaluation, with clean-null alarms controlled at the chosen level.
4. Median measurement cost to validated recovery or explicit unresolved status improves by at least 20% versus the strongest simple matched-budget baseline, with world-clustered uncertainty supporting the effect.
5. Negative controls show no hidden-truth leakage or invalid monitor inputs; terminal failures are reported unresolved, not accepted.
6. Oracle-present discrimination and truth-absent recovery are reported as distinct tracks. The truth-absent track has zero reference-equivalent members in the initial hypothesis pool according to evaluator-only audits; trials with accidental inclusion are reported as invalid for the open-ended track, not silently retried or discarded.

If a gate fails, retain and publish the negative result. Do not tune thresholds or task composition after inspection of the held-out set.

## Proposed Symthaea differentiator

The plausible contribution is not simply faster equation generation. The benchmark must first distinguish oracle-present candidate discrimination from truth-absent recovery before making an open-ended discovery claim. It is a combined system: constructive unit-aware generation; experiment selection robust to declared uncertainty scenarios; pre-outcome predictive records; monitoring with explicit statistical assumptions; bounded structural recovery after falsification; fresh validation; and a typed `validated`, `rejected`, or `unresolved` outcome with cost and provenance receipts.

To substantiate the claim, first compete on a shared symbolic-regression suite with strong baselines. Then run a separate held-out active-discovery/recovery benchmark under matched information and compute budgets. Until both pass, describe the combined loop as a research hypothesis—not state of the art.

## Reproducible run manifest

Every run should record benchmark/version and world family; exact commit and source-tree digest; build profile, Rust toolchain and dependency lockfile; algorithm, full configuration, grammar, priors and dimensions; RNG seed and stream IDs; generator and split digests; measurement count/cost, hardware, memory and timings; per-step action and forecast timestamped before observation; observed outcome; monitor state transition and proposal IDs; final status; all structural/predictive metrics; protocol failures; and unresolved reason.

`queued`, `skipped`, `cancelled`, and `in_progress` CI states are not passes. Neither CI status nor a manually replayed simulation substitutes for an exact-head benchmark receipt.

## References

- Udrescu and Tegmark, AI Feynman, Science Advances (2020): https://doi.org/10.1126/sciadv.aay2631
- AI Feynman 2.0, NeurIPS (2020): https://proceedings.neurips.cc/paper/2020/hash/33a854e247155d590883b93bca53848a-Paper.html
- Li et al., GenSR, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html
- Jiang et al., EGG-SR, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/1704fe7aaff33a54802b83a016050ab8-Abstract-Conference.html
- SRBench: https://github.com/cavalab/srbench
- ERBench, Equation Recovery Benchmark and Testsuite (2026): https://arxiv.org/abs/2606.09276
- ERBench leaderboard and benchmark documentation: https://equation-discovery.ti2.fmi.uni-jena.de/index.html
- Yang et al., TruthInsightBench: An Evidence-Grounded Benchmark for Automated Evaluation of Open-Ended Scientific Discovery Agents (2026): https://arxiv.org/abs/2609.05079
- SRBench++, IEEE TEVC: https://doi.org/10.1109/TEVC.2024.3423681
- 2026 symbolic-regression survey, published 20 August 2026: https://doi.org/10.1007/s11831-026-10681-w
- Exhaustive Symbolic Regression and MDL, Philosophical Transactions of the Royal Society A, online 9 April 2026: https://doi.org/10.1098/rsta.2024.0584
- SciLaws-Bench: https://yiyihum.github.io/SciLaws-Bench/
- Li and Pan, Robust Equation Structure Learning with Adaptive Refinement (RESTART), ICLR 2026: https://openreview.net/forum?id=z9TKJhLVKj
- Saveliev et al., Influence-Guided Symbolic Regression (2026): https://arxiv.org/abs/2605.29184
- Pang et al., Deliberate Evolution, ICML 2026: https://proceedings.mlr.press/v306/pang26f.html
- Boussif et al., Bayesian Symbolic Regression with Entropic Reinforcement Learning, UAI 2026: https://proceedings.mlr.press/v337/boussif26a.html
- LLM-AutoSciLab / ActiveSciBench: https://scientific-discovery.github.io/llm-autoscilab-project/
- LLM-SRBench, ICML (2025): https://proceedings.mlr.press/v267/shojaee25a.html
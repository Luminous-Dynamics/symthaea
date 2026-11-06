# Symthaea RSI: frontier comparison, evaluator research and benchmark targets

**Version:** v0 · **Date:** 2026-11-06  
**Owner:** #10756 (evaluator research)  
**Related:** #10497 (R4 ignition), #10691 (algorithmic RSI benchmark), #10717 (causal lineage), #10749 (preregistration semantics)

## Executive position

Symthaea should not compete on a single leaderboard score or claim recursive self-improvement because a task-solving agent improves. The defensible target is a **reproducible research process that produces independently confirmed, transferable improvements, improves its own research procedure, and preserves capability under a fixed inclusive budget**.

The programme needs three coupled but separately measured loops:

1. **Capability-production loop:** research agents alter data, algorithms, model/training methods, tools or systems; each accepted change is tested from a clean start.
2. **Evidence-production loop:** researchers build or improve tasks, measurements and evaluators; their outputs are judged against independent target benchmarks, target disagreement and human adjudication.
3. **Inheritance loop:** accepted changes become the next researcher; later researchers must beat earlier procedures from identical starts and equal total budgets, repeat over full cycles and transfer to fresh task families.

An improvement in one loop is not evidence that the other two improved. This distinction is the central scientific and product advantage Symthaea can pursue.

## 1. Frontier comparison as of 2026-11-06

| System / benchmark | Demonstrated result in the public source | What it proves | What it does not yet prove for Symthaea R4 |
|---|---|---|---|
| **AIDE²** (arXiv:2609.26457) | Seven successive agent improvements in an autonomous eight-day run; transfer to four held-out R&D task families, including an OOD weather-forecasting family; reported reward-hacking rate fell from 55% to 32%. | A serious current reference for iterative self-editing and transfer. | Its separate ignition comparison used three seeds per arm from the same AIDE47 inner-agent start and 50 outer steps; mean endpoints 0.780 vs 0.782 were judged inconclusive. Product-loop progress is not established outer-loop superiority. |
| **Recursive automated R&D** (2026-10-18 release) | NanoChat validation BPB 0.9372 community reference vs 0.9109 system result over 10 B200 seeds; NanoGPT result around 77.3–77.5 s vs a 79.7 s reference; SOL-ExecBench 0.754 vs 0.699 over 235 B200 kernels. | Strong practical comparator for parallel research threads, search, algorithm/system tuning and artifact retention. | These results do not establish same-start R4 ignition. Hardware, task and evaluator must be matched; the NanoGPT official leaderboard timing was still pending in the public artifact README, and only 10 SOL examples were public. |
| **AI4AI-Bench** (arXiv:2608.20318) | Paper reports 0.166 mean and 0.250 best normalized score across 10 algorithm-design tasks and 29 configurations. | Useful frozen training-algorithm research work packages and a separation between run tuning and learning-method changes. | A third-party full code audit dated 2026-11-03 reports score-changing defects in 4/10 tasks, hidden data exposure in two tasks, best-of-three selection on final metrics, and no case classified as research progress under that audit's rubric. Treat the published ranking as provisional pending corrected reproduction. |
| **RSIBench-Data** (arXiv:2607.25886) | Agents improved on their first valid attempt in 58.33% of settings; among runs continuing beyond the best observed score, 78.26% ended below that peak. | Data-strategy iteration can work, but continued search often loses the best result. | Final checkpoint is not a safe proxy for best-so-far; preserving a peak is not proof of repeatable transfer or R4. |
| **EvalResearchBench** (arXiv:2610.04184) | Nine evaluator researchers and 13 candidate models; best evaluator ordered about 75% of candidate pairs as target benchmarks did, below a 91% target-disagreement ceiling. Development-best evaluator was not best on sealed targets; human-designed task samples remained strong. | Evaluator construction is a measurable research skill, not a solved utility. | A strong development grader can fail on sealed families; self-designed evaluation cannot certify itself. |
| **METR / frontier-lab productivity signal** (reported 2026-11-05) | METR estimates benchmark-task capability improvement has accelerated about 1.9× since October 2025, with a 95% confidence interval of 1.5–2.8×; METR emphasizes uncertainty and reports that highly significant fully autonomous contributions are not yet a major contributor to the most important training stages. | AI assistance may be accelerating human AI R&D already. | Increased human+AI productivity is not equivalent to autonomous recursive self-improvement. |

### Comparator interpretation

- **AIDE² is the closest public R3/product-loop reference.** Symthaea should reproduce its meaningful elements—successive inheritance, held-out family transfer, reward-hacking controls—then exceed its evidence quality with same-start ignition, powered confirmation, fully inclusive budget matching and multiple outer-cycle repetitions.
- **Recursive is the strongest engineering/R&D work-package comparator in this set.** Reproduce its task-level results where artifacts and hardware permit, but classify research yield separately from runtime or validation-score tuning.
- **AI4AI-Bench is an algorithm-design task source, not a clean leaderboard target in its audited v1 state.** Symthaea should not claim victory by comparing a clean result to an uncorrected score.
- **RSIBench-Data makes trajectory preservation non-optional.** Report first-valid, best-so-far, final and predeclared selection separately.
- **EvalResearchBench reveals a second-order bottleneck.** A system that improves its researcher but cannot preserve evaluator validity may only become better at optimizing measurement defects.
- **METR's estimate is a productivity signal with high uncertainty, not an autonomous-RSI result.**

## 2. Paradigm shifts

### 2.1 From score maximization to independently confirmed discovery yield

Measure confirmed inherited improvements per fully inclusive cost, not raw accepted patches, benchmark score or number of experiments. Every attempted change—including failures, no-ops, invalid proposals and human rework—stays in the denominator and cost ledger.

### 2.2 From a single champion to a portfolio of falsifiable hypotheses

Retain multiple non-dominated candidate lineages. Give each proposed change a mechanism, predicted intermediate effect, rollback and transplant procedure. Test single-parent mutation, reaction-norm changes across tasks and multi-parent hybridization as separate treatments. A merged lineage is provenance, not causal attribution; use #10717 for intervention and interaction effects.

### 2.3 From self-grading to independently anchored measurement

Evaluator research may use development targets, but the evaluator must be frozen before sealed candidate ranking. Compare against a target portfolio and report the target disagreement ceiling. A candidate-side change to grader, task contract, hidden data, score direction or checkpoint-selection rule is a hard integrity failure.

### 2.4 From one endpoint to complete trajectories

Preserve first-valid, best-so-far, final and predeclared selected artifacts as separate results. Never select the best checkpoint on the sealed confirmation metric. Report per-round development and independent audit values, failures, costs and protected regressions.

### 2.5 From task transfer to mechanism transfer

Hold out entire task families and at least one domain. Test whether a research motif (e.g. memory compression, experiment prioritization or causal debugging) improves a new domain under the same budget, without retuning. Track the causal mechanism, not merely the final artifact score.

### 2.6 From a vague “self-improvement loop” to a claim ladder

Keep separate: product improvement → accumulation → repeatability → better improver under matched budget → outer-loop ignition → inherited recursive progress → fresh-family/domain transfer. No lower tier implies a higher tier.

## 3. Benchmark programme and prospective targets

These targets are hypotheses to be calibrated on pilots, not current Symthaea results.

| Lane | Required benchmark | Prospective target / decision rule |
|---|---|---|
| **Agent self-improvement (R3)** | AIDE²-like iterative code change and four-family held-out transfer, with source/build identity and reward-hacking negative controls. | Match the seven-step/four-family public reference under a comparable budget first; then require independent audit of every retained generation and report all seeds and failed attempts. |
| **Outer-loop ignition (R4)** | Same-start evolved researcher vs fixed/human reference, identical inner-agent artifact, same task population and inclusive total budget. | At least five paired pilot seeds where feasible; confirmatory N from a preregistered precision/power analysis; at least two full-loop reruns and two complete outer cycles for recursive-progress claims; fresh-family transfer and no protected capability regression. |
| **Evaluator research** | EvalResearchBench-style target ranking plus sealed transfer, adversarial leakage and construct-validity mutations. | Aim for ≥0.85 raw pairwise concordance against the frozen target portfolio, while reporting the measured target disagreement ceiling (published reference about 0.91); zero critical false passes; no material development-to-sealed collapse; match or beat the human-designed suite's decision quality at lower inclusive cost. Recalibrate to the measured target portfolio before confirmatory use. |
| **Data-centric research** | RSIBench-Data-style iterative data strategy over fixed model/training stack and budget. | Pilot target ≥70% of settings improve on first valid attempt; retain best-so-far independently; report final separately; confirm any gain from a clean start on held-out task families. |
| **Algorithm redesign** | Corrected, integrity-audited AI4AI task set plus independent successor training from initialization. | First reproduce the shipped baseline and correct task defects. Then target a mean normalized score above the paper's reported 0.250 best only on the corrected task portfolio, with ≥5 paired seed blocks per task where feasible, per-task uncertainty and a classification of tuning vs algorithm redesign. No scalar aggregate may hide a safety or task-family regression. |
| **Automated R&D systems work** | Recursive NanoChat/NanoGPT/SOL-style artifacts, reproduced on matched hardware and exact code/data/evaluator versions. | Demonstrate reproducible paired improvement with uncertainty and exact artifacts. Keep H100/B200 results separate from smaller-device results; no guessed hardware normalization. |
| **Causal attribution** | 2×2 intervention designs, cluster-aware summaries, null controls and lineage-level change-unit analysis. | Attribute gains only to randomized/matched interventions with mechanism predictions and independent confirmation; report interaction effects and cluster-level uncertainty, not parent credit from ancestry alone. |

### Non-compensatory conditions

No performance target can compensate for: sealed-data leakage; candidate-controlled evaluator; test-set checkpoint selection; fabricated evidence; missing failure/cost denominators; invalid source/build lineage; severe protected-capability regression; or a false claim of confirmation/transfer. These are gates, not score penalties.

## 4. Evaluator-research experiment (#10756)

Compare a human-designed compact suite, deterministic rules, frontier-model evaluator builders and Symthaea under matched information and cost. Use separate discovery, development, calibration, sealed target and fresh-family transfer splits.

Measure:
- pairwise ranking concordance, Kendall/Spearman agreement, top-k agreement and rank error;
- performance relative to disagreement among target benchmarks and human adjudicators;
- false promotion, false rejection and abstention at preregistered thresholds;
- task-family/domain coverage, truncation, missing outputs and execution failure;
- construct-validity controls, leakage detection and robustness to scorer optimization;
- confidence calibration, cluster-appropriate uncertainty and target-version stability;
- latency, evaluator calls, tokens, compute, energy and human review cost.

Adversarial controls must include readable hidden seeds, task overlap, scorer gaming, collapsed outputs, unattainable metric optima, noisy baselines, best-of-N sealed checkpoint selection, truncation, budget exhaustion, dropped tool results, missingness manipulation, evaluator-version drift and pseudo-independent correlated tasks.

A high-concordance evaluator is a measurement instrument, not an authority. The target portfolio itself can be wrong or biased; use disagreement ceilings, human review and independent target adjudication.

The first synthetic adversarial-control pack is tracked in `docs/research/fixtures/SYM_RSI_EVALUATOR_RESEARCH_INTEGRITY_FIXTURES_V0_2026-11-06.json`, with its companion Draft 2020-12 schema in `docs/research/fixtures/SYM_RSI_EVALUATOR_RESEARCH_INTEGRITY_FIXTURES_SCHEMA_V0_2026-11-06.json`. It contains 16 hard-gate negative controls and four synthetic positive/inconclusive controls. A custom structural walk found zero schema errors, 16 unique negative-control IDs and the expected hard-gate/non-hard-gate split. These fixtures specify threat mutations and expected dispositions; they are not real task data, sealed tests or a substitute for an independently custodied target portfolio.

## 5. Immediate execution sequence

1. Finish semantic R4 preregistration checks in #10749 and preserve the template's TEMPLATE_NOT_PREREGISTERED state until an actual experiment is frozen.
2. Audit the public Recursive and AI4AI artifacts at exact commit/digest; for AI4AI, reproduce the independent audit's critical cases before using corrected results as a baseline.
3. Build the evaluator-research task/fixture pack in #10756; freeze evaluator artifacts before sealed comparison.
4. Extend #10691 with per-task seed replication, algorithm-change classification, and the evaluator/construct-validity negative-control pack.
5. Run small local smoke tasks for correctness and artifact integrity; reserve expensive accelerator comparisons for exact matched hardware. Local or reduced-compute smoke results are not substitutes for the benchmark's prescribed B300/B200/H100 runs.
6. Use #10717 to test which change units and interactions caused any observed gain; use #10497 to test whether the improver itself gets better.

## 6. Claim boundaries and reproducibility

- A structural or semantic validator pass is not evidence that referenced files, digests, signatures or custodians are authentic.
- A disjointness declaration is not proof of population independence; an independent custodian must inspect the actual manifests.
- Evaluator declarations do not prove independence; source, mounts, credentials, egress, logs and hidden-test custody require independent audit.
- A faster benchmark run is an engineering result, not automatically a new algorithm or R4 result.
- A higher inner-agent score is a product result, not evidence that the outer-loop researcher improved.
- A single positive seed or a post-hoc best checkpoint is not an improvement-process result.

## Sources

1. AIDE², *Recursive self-improvement of AI research agents*: https://arxiv.org/abs/2609.26457
2. Recursive, *First steps toward automated AI research* (2026-10-18): https://recursive.com/updates/first-steps-toward-automated-ai-research
3. Recursive public artifacts: https://github.com/recursive-org/first-steps-toward-automated-ai-research
4. AI4AI-Bench paper: https://arxiv.org/abs/2608.20318
5. Independent AI4AI-Bench code audit (2026-11-03): https://github.com/Open-Benchmark-Review/audits/blob/main/audits/ai4ai-bench/report.md
6. RSIBench-Data: https://arxiv.org/abs/2607.25886
7. EvalResearchBench: https://arxiv.org/abs/2610.04184
8. RSI evidence census: https://arxiv.org/abs/2610.27610
9. METR research productivity coverage (2026-11-05): https://www.reuters.com/technology/anthropics-models-improving-nearly-twice-fast-study-finds-2026-11-05/

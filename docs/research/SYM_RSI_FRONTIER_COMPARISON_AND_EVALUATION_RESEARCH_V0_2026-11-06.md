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
| **RSI-Env** (arXiv:2611.01736, submitted 2026-11-01) | Persistent environment for coordinated model/harness change with a separate Host Controller; seven tasks across Open-General, Capability-Directed and Benchmark-Guided settings. Open-General Self: GPQA 65.2→73.2 and AIME 53.3→56.7; Knowledge Self: SimpleQA 82→85→88 across H0/M0→H1/M0→H1/M1. | New direct comparator for persistent research state, controller isolation, model/harness co-evolution and recursive continuation. | Scores are avg@3/pass@3 on four capability benchmarks; gains coexist with regressions (e.g. Open-General SimpleQA −4.8 points; Knowledge GPQA −12.2 and Terminal-Bench −6.8). Several Self runs used 24.9–28.95 h against 1.19–6.97 h for Teachers under a common 30 h allocation. A common ceiling is not a matched realized-cost comparison; the paper's sustained three-stage sequence is one task-specific case, not a powered R4 ignition comparison. |
| **RSIGym** (arXiv:2610.10310, 2026-10-07) | Service-based environment with Data, Harness and Joint tracks; six research models on five benchmarks. The lead Opus 5 run reports RSI-Index 0.4809 and improves SWE-bench Verified 17.67→50.33, Terminal-Bench 10.49→26.97, AIME 31.67→97.78, GPQA 51.33→83.33 and SkillsBench 1.65→21.21. | Important benchmark for joint data/training/harness research and standardized reusable services. | Its public methodology page says the $500 covers platform services but excludes research-agent inference and final official evaluation; one research run per configuration; final evaluations reuse tasks available during development; avg@3 repeats task attempts, not the research loop. Treat 0.4809 as a useful product-improvement comparator, not a fully inclusive, sealed or repeated-loop R4 result. |
| **RSI-Forge** (arXiv:2610.09426, 2026-10-07) | Generates executable environments from research papers: 210 environments across 18 fields, 90 human-expert reviewed; four models run three successive attempts across 120 environments; at least one model improves after the first attempt in 84% of environments and beats a reproduced paper method in 68/120. | A new comparator for scaling research-task generation and compiling published methods into runnable baselines. | Human experts were more critical than agent judges about shortcut resistance, faithfulness to the paper and whether one idea can exhaust a task. Its task-generation pipeline must itself be evaluated for construct validity, leakage and baseline fidelity. |
| **Harness²** (arXiv:2610.14224, 2026-10-09) | Two-level recursion: task-level Propose–Probe–Compose and a persistent harness-improvement library across tasks. Reports gains across WorkBuddy-Bench, JobBench and Legal Agent Benchmark, with cross-task and cross-executor transfer analyses. | Strong reference for separating task-specific execution-harness edits from reusable evidence-supported lessons about how to improve future tasks. | It explicitly notes behavioral contrasts do not establish task correctness without an independent verifier. Results average three shuffled runs; task execution resets to a common base harness and only the improvement library persists. Treat its transfer findings as a harness-research comparator, not full model/algorithm R4 ignition. |
| **Co-Harness** (arXiv:2607.22688, 2026-07-17) | Alternates harness optimization with model post-training; a HarnessCritic diagnoses failures and proposes local harness updates before training on improved trajectories; reports a 200+ hour autonomous case study. | Comparator for joint harness/weight co-evolution and recovery from long-running research processes. | A long autonomous case study is not itself a same-start, matched-budget, multi-seed improvement-of-the-improver result; measure both model and harness contribution separately. |
| **Recursive automated R&D** (2026-10-18 release) | NanoChat validation BPB 0.9372 community reference vs 0.9109 system result over 10 B200 seeds; NanoGPT result around 77.3–77.5 s vs a 79.7 s reference; SOL-ExecBench 0.754 vs 0.699 over 235 B200 kernels. | Strong practical comparator for parallel research threads, search, algorithm/system tuning and artifact retention. | These results do not establish same-start R4 ignition. Hardware, task and evaluator must be matched; the NanoGPT official leaderboard timing was still pending in the public artifact README, and only 10 SOL examples were public. |
| **AI4AI-Bench** (arXiv:2608.20318) | Paper reports 0.166 mean and 0.250 best normalized score across 10 algorithm-design tasks and 29 configurations. | Useful frozen training-algorithm research work packages and a separation between run tuning and learning-method changes. | A third-party full code audit dated 2026-11-03 reports score-changing defects in 4/10 tasks, hidden data exposure in two tasks, best-of-three selection on final metrics, and no case classified as research progress under that audit's rubric. Treat the published ranking as provisional pending corrected reproduction. |
| **RSIBench-Data** (arXiv:2607.25886) | Agents improved on their first valid attempt in 58.33% of settings; among runs continuing beyond the best observed score, 78.26% ended below that peak. | Data-strategy iteration can work, but continued search often loses the best result. | Final checkpoint is not a safe proxy for best-so-far; preserving a peak is not proof of repeatable transfer or R4. |
| **EvalResearchBench** (arXiv:2610.04184) | Nine evaluator researchers and 13 candidate models; best evaluator ordered about 75% of candidate pairs as target benchmarks did, below a 91% target-disagreement ceiling. Development-best evaluator was not best on sealed targets; human-designed task samples remained strong. | Evaluator construction is a measurable research skill, not a solved utility. | A strong development grader can fail on sealed families; self-designed evaluation cannot certify itself. |
| **METR / frontier-lab productivity signal** (reported 2026-11-05) | METR estimates benchmark-task capability improvement has accelerated about 1.9× since October 2025, with a 95% confidence interval of 1.5–2.8×; METR emphasizes uncertainty and reports that highly significant fully autonomous contributions are not yet a major contributor to the most important training stages. | AI assistance may be accelerating human AI R&D already. | Increased human+AI productivity is not equivalent to autonomous recursive self-improvement. |

### Comparator interpretation

- **RSIGym is the strongest service-based joint data/training/harness comparator.** Reproduce its five-benchmark vector and Data/Harness/Joint ablations, but correct the evidence boundary: include agent inference, final evaluation and human review in total cost, and use a sealed task-family confirmation set rather than relying on the public task reuse.
- **RSI-Forge changes the scale of task supply.** The promising paradigm is to turn papers into executable, independently reproducible research environments. But task generation is a second-order scientific claim: an environment must be faithful to the source method, have an independently reproduced baseline, resist shortcuts and leave more than one meaningful avenue for improvement. Human domain review is a critical component, not a final cosmetic pass.
- **Harness² supplies a useful two-level recursion pattern.** Improve the execution harness for the current task using matched contrastive probes, then update a separate improvement library with only reusable, evidence-supported lessons. Resetting task-specific harness state avoids blindly carrying a specialized solution into unrelated tasks. Its behavioral contrasts remain hypotheses about performance, not correctness proofs.
- **Co-Harness is a direct joint model/harness comparator.** Symthaea should separate gains caused by better training data/model weights from gains caused by the harness and by their interaction. Alternate these changes as distinct treatments, then test whether the joint successor beats the best single-component intervention under the same total budget.
- **RSI-Env is the newest persistent-environment comparator.** Its independent Host Controller, retained failed experiments and model/harness successor registration are worth matching. Its results also sharpen our benchmark: every generation must report a cross-capability vector, not only a target score. The Open-General and Knowledge regressions show why capability retention must be a non-compensatory gate. The same 30-hour allocation ceiling does not make realized resource use equal when Self and Teachers consume very different amounts of time; compare on fully inclusive realized cost as well as allocated caps.
- **The new research-state target:** separate the persistent research record from the current model/harness. After a successor update or full restart, the system should recover validated findings, rejected hypotheses, costs and evidence lineage without accidentally inheriting unqualified conclusions. Measure state-recovery fidelity, provenance completeness, retrieval utility and the rate at which recovered knowledge produces independently confirmed gains on fresh tasks. Do not turn raw memory volume or recall scores into a substitute for useful, verified research continuity.

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

## 4. Research-environment generation and two-level harness research

### Paper-to-environment compiler (compose with #9332 AR-5D)

RSI-Forge demonstrates a pipeline that turns published research into executable environments and independently reproduces a paper's baseline before allowing agent improvement. Symthaea should evaluate this as a separate research capability, not assume task generation is valid because the generated task is executable.

A first qualification pack should include 20–30 papers across several fields, stratified by method type and difficulty, with at least two independent domain reviewers for each scored environment. Require:
- source-method fidelity: every executable operation maps to the paper's stated mechanism, assumptions and data;
- baseline fidelity: independently reproduced paper baseline with variance, environment and resource settings retained;
- evaluator discrimination: known valid modifications and negative controls must be correctly ordered;
- shortcut resistance: metadata-only, seed leakage, answer reconstruction and test-set tuning controls;
- idea non-exhaustion: at least two materially distinct, valid improvement directions remain after the first obvious optimization;
- domain review: blinded expert ratings for construct fidelity, baseline quality, shortcut resistance, reproducibility and scope;
- task-family holdouts: split by source-paper lineage and scientific method family, not randomly by environment row.

Prospective targets: zero critical oracle/seed leakage; at least 90% of environments pass independent baseline-reproduction review; at least 80% receive blinded expert approval for method fidelity and meaningful improvement headroom; and no environment enters a confirmatory split until construct and shortcut audits pass. These thresholds are hypotheses for calibration, not current results.

### Two-level harness learning (compose with Harness²)

Compare:
1. per-task execution-harness optimization with no persistent experience;
2. persistent untyped transcript/history retrieval;
3. persistent verified experience library;
4. direct carry-forward of the previous task's execution harness;
5. shuffled or irrelevant experience library;
6. full two-level Propose–Probe–Compose plus improvement-library update.

Keep the task's execution harness reset to the same base for cross-task transfer tests; only the improvement mechanism's retained knowledge is changed. Add an independent correctness verifier for task deliverables because matched behavioral contrast alone cannot establish correctness.

Measure task outcome, protected capability vector, cost, number of probes, composition precision, reuse rate, false transfer, evidence support for retained lessons and performance on held-out task families and unseen executor models. A useful positive result is not merely a higher current-task score: the learned improvement procedure must help later, unrelated tasks without carrying task-specific assumptions.

### Model–harness co-evolution (compose with Co-Harness / RSIGym)

Run a factorial comparison: data-only, training-setting-only, harness-only, model-weight-only where meaningful, harness→model, model→harness, and joint alternating updates. Keep the original base model and starting harness exact; run the same seed and task families; account for all data-generation, training, inference, evaluator and recovery costs. The interaction term is a first-class result: joint change may be super-additive, redundant or destructive. Do not credit the full joint gain to either component alone.

## 5. Evaluator-research experiment (#10756)

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

## 6. Immediate execution sequence

1. Finish semantic R4 preregistration checks in #10749 and preserve the template's TEMPLATE_NOT_PREREGISTERED state until an actual experiment is frozen.
2. Audit the public Recursive and AI4AI artifacts at exact commit/digest; for AI4AI, reproduce the independent audit's critical cases before using corrected results as a baseline.
3. Build the evaluator-research task/fixture pack in #10756; freeze evaluator artifacts before sealed comparison.
4. Extend #10691 with per-task seed replication, algorithm-change classification, and the evaluator/construct-validity negative-control pack.
5. Run small local smoke tasks for correctness and artifact integrity; reserve expensive accelerator comparisons for exact matched hardware. Local or reduced-compute smoke results are not substitutes for the benchmark's prescribed B300/B200/H100 runs.
6. Use #10717 to test which change units and interactions caused any observed gain; use #10497 to test whether the improver itself gets better.

## 7. Claim boundaries and reproducibility

- A structural or semantic validator pass is not evidence that referenced files, digests, signatures or custodians are authentic.
- A disjointness declaration is not proof of population independence; an independent custodian must inspect the actual manifests.
- Evaluator declarations do not prove independence; source, mounts, credentials, egress, logs and hidden-test custody require independent audit.
- A faster benchmark run is an engineering result, not automatically a new algorithm or R4 result.
- A higher inner-agent score is a product result, not evidence that the outer-loop researcher improved.
- A single positive seed or a post-hoc best checkpoint is not an improvement-process result.

## Sources

1. AIDE², *Recursive self-improvement of AI research agents*: https://arxiv.org/abs/2609.26457
2. RSI-Env, *An Open-Ended Environment for Recursive Self-Improvement of Agents* (2026-11-01): https://arxiv.org/abs/2611.01736
3. Recursive, *First steps toward automated AI research* (2026-10-18): https://recursive.com/updates/first-steps-toward-automated-ai-research
4. Recursive public artifacts: https://github.com/recursive-org/first-steps-toward-automated-ai-research
5. AI4AI-Bench paper: https://arxiv.org/abs/2608.20318
6. Independent AI4AI-Bench code audit (2026-11-03): https://github.com/Open-Benchmark-Review/audits/blob/main/audits/ai4ai-bench/report.md
7. RSIBench-Data: https://arxiv.org/abs/2607.25886
8. EvalResearchBench: https://arxiv.org/abs/2610.04184
9. RSI evidence census: https://arxiv.org/abs/2610.27610
10. METR research productivity coverage (2026-11-05): https://metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/
11. RSIGym project and methodology: https://rsi-index.ai/ and https://github.com/evolvent-ai/RSIGym
12. RSI-Forge: https://arxiv.org/abs/2610.09426
13. Harness²: https://arxiv.org/abs/2610.14224
14. Co-Harness: https://arxiv.org/abs/2607.22688

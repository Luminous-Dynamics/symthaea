# AR-5 — competitive research-system landscape (30 October 2026)

**Status:** desk-research synthesis of public papers/leaderboards. Reported external results are not Symthaea replications and are not directly comparable across task suites, models, budgets or hardware.

## Executive finding

The relevant frontier has moved beyond classic neural architecture search. There are at least five adjacent layers:

1. **Architecture ranking and low-cost screening** — zero-cost NAS proxies on fixed search spaces.
2. **Evolutionary program/algorithm search** — mutation, parent selection, novelty filtering and ensemble-guided proposal generation.
3. **Quality-diversity and direction allocation** — preserve alternative mechanisms instead of selecting only the current global winner.
4. **Long-horizon multi-agent research infrastructure** — isolated workspaces, persistent memory, asynchronous execution, continuous grading and resource control.
5. **End-to-end scientific research evaluation** — whether the system can formulate, implement, execute and defend a result with evidence.

Symthaea's current AR-5B work contributes to layer 3 and aims to build an evidence/verification layer across all five. It does not yet match the full capability of the systems below.

## Public comparison

| System / benchmark | Published or reported result | Main mechanism | What Symthaea should learn |
|---|---|---|---|
| Coral Discovery (preprint, 23 Oct 2026) | Authors report 6 wins, 1 tie, 2 losses vs strongest open autonomous multi-agent comparison on nine matched tasks; reported Kernel result 1103 → 1095 cycles and threshold reached at evaluation 275 vs prior 596; Polyominoes 91.78 vs 87.0 | Verifier-assigned direction/method labels, direction-local records, exploration bonus, NEW-direction option and allocation advice | AR-5B is conceptually adjacent, but v0.1 is not a faithful replication. Reproduce the published scheduler and its ablations before claiming an advantage. Cost, evaluator independence and local-record attribution must match. |
| CodeEvolve (Findings EMNLP 2026) | Authors report matching/surpassing reported AlphaEvolve results on 5/9 AlphaEvolve tasks; beating OpenEvolve and ShinkaEvolve on 6/9 under their matched setup; Qwen3-Coder-30B reportedly beats published AlphaEvolve CirclePackingSquare scores at about one-tenth the cost | Island evolution, inspiration crossover, meta-prompting, depth refinement, CVT-MAP-Elites and weighted LLM ensemble | Implement quality-diversity archive baselines, island structure, cost/token/compute accounting, and measure diversity × progress rather than only score. |
| ShinkaEvolve (ICLR 2026) | Authors report a new circle-packing result with 150 samples | Parent sampling, novelty rejection sampling, bandit-based LLM ensemble selection | Add semantic novelty rejection, sample efficiency and bandit proposal-source selection. Code novelty alone must not count as semantic novelty. |
| CORAL (COLM 2026; open-source framework) | Authors report new SOTA on 10 tasks and 3–10× improvement rates vs fixed evolutionary baselines; four co-evolving agents reportedly improve kernel cycles 1363 → 1103 | Persistent shared memory, asynchronous agents, heartbeat interventions, isolated workspaces, continuous evaluation and resource/session management | Symthaea lacks the full long-running orchestration/runtime layer in AR-5. Add persistent research memory, isolated worktrees, task dispatch, recovery and agent-health budgets only after evaluator isolation is qualified. |
| FIRE-Bench (ICML 2026) | Authors report that even the strongest tested agents achieved less than 50 F1 on rediscovering documented findings, with high run-to-run variance and weaknesses in design, execution and evidence-backed reasoning | Full-cycle rediscovery tasks: question → experiment design → execution → evidence-backed conclusion | Add an agent-science benchmark. A plausible paper, successful code generation or single scalar score is not evidence of discovery. |
| NAS-Bench-201 zero-cost-proxy leaderboard | Listed CIFAR-10 ranking correlations include ParZC 90.4 Spearman / 70.6 Kendall, AZ-NAS 89.2 / 71.2, and InTrain 85.7 / 66.9 | Predict relative network ranking before full training on a fixed benchmark search space | Retain a rigorous within-grammar architecture-screening track. These correlations do not measure open-ended operator invention or held-out scientific transfer. |

### Primary sources

- Coral Discovery, arXiv:2610.33281 (23 October 2026): https://arxiv.org/abs/2610.33281
- CodeEvolve, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1345/
- ShinkaEvolve, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html
- CORAL, COLM 2026 / arXiv:2604.01658: https://arxiv.org/abs/2604.01658
- FIRE-Bench, ICML 2026 / PMLR 306: https://proceedings.mlr.press/v306/wang26d.html
- NAS-Bench-201 CIFAR-10 zero-cost-proxy leaderboard: https://www.sota2.com/research/sota/neural-architecture-search-on-nas-bench-201-cifar-10

## Fair-comparison rules

Reported headline numbers above are source-reported and task-specific. Do not rank these systems by their raw headline metrics. A fair comparison must pin:

- task/problem set and training/test split;
- candidate representation, primitive grammar and search-space size;
- model/backbone, proposal prompts, context and tool access;
- number of proposals, evaluator calls, candidate executions and failed attempts;
- wall-clock, tokens, CPU/GPU hours, memory and energy;
- evaluator access, hidden-label policy and verifier independence;
- stopping rules, random seeds and all hyperparameters;
- exact source revisions, environment manifests and raw receipts.

Use two levels of competition:
- **Mechanism-matched replication:** same task, model, budget, evaluator and published scheduler as the selected baseline.
- **System-level comparison:** best practical pipeline each system can field under a strictly equal total resource budget. Report both; they answer different questions.

## Symthaea gap analysis

### Present work in AR-5

- AR-5A (#9227/#9229): corrects genotype/operator influence and seed dependence in a current neuroevolution path.
- AR-5B (#9230/#9233): adds verifier-grounded direction identity, direction-local credit and a NEW arm.
- AR-5B1 (#9235): begins matched-budget policy comparison on four synthetic landscape families.
- AR-5C (#9319): defines typed primitive invention and verifier-qualified grammar expansion.

These are important foundations, but not a full autonomous discovery system.

### Missing capabilities before a credible SOTA comparison

1. **Faithful baseline implementations:** Coral Discovery scheduler, ShinkaEvolve, CodeEvolve, CORAL and quality-diversity archives, with explicit versions/deviations.
2. **Real proposal generation:** controlled candidate generators, semantic deltas, lineage, parent selection, mutation/crossover and proposal-source attribution.
3. **Persistent multi-agent research memory:** durable hypotheses, failed approaches, verified mechanisms, reusable experiments and provenance. Memory must be queryable without leaking hidden labels.
4. **Cost-aware scheduling:** a strict multi-resource budget over evaluator calls, wall time, tokens, CPU/GPU time, memory and energy. Current AR-5B allocates equal-cost evaluation slots.
5. **Independent verification and evaluator separation:** cryptographically authenticated receipts, hermetic execution, hidden test/evaluation access controls and adversarial audit.
6. **Open-ended benchmark families:** fixed-grammar NAS, runtime-regime search, grammar extension, causal mechanism recovery, held-out transfer and full-cycle rediscovery.
7. **Robust statistics:** paired task seeds, family-level holdouts, confidence intervals, power analysis, negative controls, censored failures and worst-family performance.
8. **Operational reliability:** asynchronous workers, isolated worktrees, crash/retry, health/resource management and immutable experiment registries.

## Proposed paradigm shift

Treat automated discovery as an **evidence-producing research economy**, rather than a single optimizer:

- **Hypothesis layer:** research directions, causal explanations, proposed new primitives and predicted failure modes.
- **Candidate layer:** typed architecture/operator IR and explicit semantic deltas.
- **Allocation layer:** direction-local progress, quality-diversity archives, exploration/NEW options and cost-aware bandit scheduling.
- **Execution layer:** isolated workers, asynchronous runs, resumable state and strict resource budgets.
- **Verification layer:** separate verifier, signed receipts, invariants/proofs, adversarial tests and held-out evaluation.
- **Knowledge layer:** persistent memory of verified and falsified hypotheses with task scope and evidence lineage.
- **Qualification layer:** transfer, reproducibility, resource Pareto frontier and human-reviewed grammar/policy release.

The differentiator should be **causal and semantic accountability**: each proposed improvement must identify the mechanism it changes, what outcome it predicts, what counterexample would falsify it, which controls were used, and where it transfers. A higher score without a reliable causal/semantic delta is a search result, not a new paradigm.

## Benchmark portfolio and target bars

| Track | Core metrics | Target standard before public superiority claims |
|---|---|---|
| Fixed-grammar NAS | final validation/test quality, top-k recall, Spearman/Kendall proxy correlation, regret, training cost | Reproduce NAS-Bench-201 results; test additional search spaces and tasks; never select using locked test labels |
| Runtime-regime discovery | held-out task score, adaptation latency, stability, resource/energy Pareto, worst-shift loss | Paired multi-environment tasks; regime changes and non-stationarity; report worst case |
| Direction allocation | verified capability at fixed total cost, discovery probability vs budget, time/cost to first breakthrough, false novelty, no-new exploration tax | ≥100 paired landscapes over ≥6 families; compare score-only, fixed, random, UCB, Thompson, MAP-Elites, Coral-style and oracle |
| Primitive/grammar discovery | novelty precision, equivalence false positives, property/proof pass rate, held-out transfer beyond existing grammar | Exhaustive finite-domain negative controls; blinded novelty labels; ≥95% precision target after verifier calibration |
| Full-cycle research | rediscovery F1, experiment validity, reproducibility, evidence correctness, cost and run-to-run variance | FIRE-Bench-like tasks with executable intermediate gates and independent evidence checking |
| Multi-agent research infrastructure | improvement per dollar/joule/GPU-hour, worker utilization, useful-memory reuse, failure recovery, duplicate work | Equal-resource single-agent and multi-agent ablations; isolated workspaces and hidden evaluator |

**Provisional stretch goal:** beat the strongest open baseline on a majority of preregistered held-out challenge families while maintaining non-inferiority on null/negative-control families and materially reducing total resource cost. This must be evaluated separately for each track; a win on synthetic direction allocation is not a win on neural architecture quality or scientific discovery.

## Immediate order of work

1. Get #8890 and #9233 onto a compilable, testable branch; preserve the current negative SYM-ARCH-001 results.
2. Run the AR-5B1 synthetic smoke matrix, inspect failures and policy tradeoffs, then expand to ≥6 families and power-sized paired runs.
3. Reproduce the Coral Discovery allocation rule and compare it directly to the Symthaea board; measure whether the additional integrity contract improves trust without hurting efficiency.
4. Add ShinkaEvolve/CodeEvolve proposal-generation and MAP-Elites archive baselines with pinned versions and matched resource limits.
5. Build strict multi-resource scheduling and persistent-memory/worker orchestration as separate factors.
6. Implement AR-5C typed operator IR and independent equivalence/property/transfer verifier.
7. Add a FIRE-Bench-style end-to-end research evaluation before claiming autonomous scientific discovery.

## Claim ceiling

This document is a competitive landscape and engineering roadmap. It contains no Symthaea measured result and no claim of current superiority. The public results cited above are not directly comparable across task setups; they set baselines and research directions, not a single global leaderboard.

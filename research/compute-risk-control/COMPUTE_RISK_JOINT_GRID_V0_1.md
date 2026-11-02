# Joint Compute-Risk Calibration v0.1

**Status:** independent Rust research implementation, deterministic synthetic panel and authored tests. No population-level confidence claim, paper-matched reproduction, Rust test run, or real compute-saving result is claimed.

Tracking: [NEURO-SILICON-021 / #10052](https://github.com/Luminous-Dynamics/symthaea/issues/10052).

## 1. Research anchor and exact risk channels

Wang et al., *Conformal Thinking: Risk Control for Reasoning on a Compute Budget*, ICML 2026, PMLR 306: https://proceedings.mlr.press/v306/wang26hd.html

The paper formulates a two-threshold stopping policy. An upper threshold lambda-plus stops when confidence is high; a dynamic lower threshold lambda-minus(t) can stop a task that appears unsolvable. They represent different errors:

- **Upper false-positive risk:** an upper-threshold stop emits an incorrect answer.
- **Lower false-negative / premature-abandon risk:** a lower-threshold stop abandons an instance even though later reasoning checkpoints would contain a correct answer. The task loss is the fraction of remaining checkpoints that are correct:

    L_minus(t) = 1/(T-t+1) * sum_{k=t}^T I[checkpoint k is correct]

The dynamic lower threshold uses a sigmoid between lower floor l and upper threshold u=lambda-plus:

    lambda_minus(t) = l + (u-l)/(1 + exp(-c * (tokens_t - s * budget)))

**Symthaea adds a third safety channel:** whether the answer actually emitted by the selected stopping policy is wrong. This includes lower-threshold exits and full-budget fallback. Without it, a controller could satisfy two early-stop metrics while returning mostly incorrect answers.

## 2. Author-code audit

Paper PDF: https://arxiv.org/abs/2602.03814

Public author repository: https://github.com/xidulu/reasoning_risk_control

Pinned main commit inspected: `6551e3000dcda3d76491f30cc687d64618807c5b` (2026-05-14). Its README calls the repository preliminary and says the full code/assets will be released soon. The inspected tree contains no LICENSE file. Its lower-threshold helper has an unfinished path: it references an undefined `thresholds` variable and does not use the computed `lower_thresholds`. This Rust implementation is independent; the paper is the algorithmic reference and the preliminary repository is pinned for provenance only.

## 3. Paradigm shift: jointly calibrate all candidate risks

The paper describes a two-step procedure that first selects the upper threshold and then fits the lower threshold. The paper notes that this decoupling loses some theoretical rigor; its appendix discusses how a lower-threshold filter can affect the upper false-positive rate if correct and incorrect validation examples are filtered at different rates.

Symthaea instead freezes a finite grid of complete upper/lower candidates before examining validation outcomes. Each candidate is evaluated on every task. The calibration controls three bounded losses for every candidate:

1. upper wrong-answer loss;
2. lower premature-abandon loss;
3. emitted-answer error across all exit causes, including full-budget fallback.

This is a new joint-grid variant inspired by the paper, not a reproduction of its two-stage algorithm.

## 4. Simultaneous Hoeffding correction across 3G claims

Let G be the number of frozen joint candidates and N the number of independent held-out tasks. There are 3G one-sided risk statements. Allocate delta/(3G) to each statement. The shared Hoeffding correction is:

    c(N,G,delta) = sqrt( ln(3G/delta) / (2N) )

For candidate i:

    UCB_plus_i   = mean(L_plus_i)   + c
    UCB_minus_i  = mean(L_minus_i)  + c
    UCB_output_i = mean(L_output_i) + c

A candidate is feasible only when all three adjusted bounds satisfy their independently preregistered budgets:

    UCB_plus_i   <= epsilon_plus
    UCB_minus_i  <= epsilon_minus
    UCB_output_i <= epsilon_output

A union bound over all 3G statements controls the familywise statistical failure probability by delta, conditional on independent iid task-level observations from the intended validation population, bounded losses, and a candidate grid frozen before calibration outcomes are inspected. The three losses on an individual task may be dependent; the union bound does not multiply probabilities. The receipt now preserves the exact per-claim failure allocation as the rational value `confidence_delta_ppm / (3 * candidate_count)`; it does not round or truncate that allocation to an integer PPM field. The report verifier independently checks the 3G count, allocation denominator, UCB arithmetic, all three feasibility flags, and the selected minimum-compute feasible candidate.

The method enumerates the entire grid and assumes no monotonicity in threshold or risk. Among feasible candidates it selects the lowest mean token fraction, using candidate ID as a deterministic tie-break. If none is feasible it reports that outcome and never relaxes a budget.

### Numerical implementation boundary

The lower sigmoid and Hoeffding logarithm use f64. Threshold and UCB outputs are rounded outward to integer PPM with one extra PPM added to the Hoeffding correction. This is a deterministic engineering guard, not directed-interval arithmetic or a formal numerical proof. A production certificate needs an independently reviewed high-precision/directed-rounding implementation.

## 5. Separate loss definitions and diagnostics

- **Upper wrong-answer loss:** 1 only when the upper threshold fires and the emitted answer at that checkpoint is incorrect; otherwise 0.
- **Lower premature-abandon loss:** at a lower-threshold exit, the fraction of remaining checkpoints (including the exit checkpoint) that would be correct; otherwise 0.
- **Emitted-answer error:** 1 when the answer actually emitted at any exit—including lower stop and full-budget fallback—is wrong; otherwise 0. This is the third simultaneous calibration channel.
- **Upper efficiency loss:** normalized delay after the first correct checkpoint for upper exits.
- **Lower efficiency loss:** fraction of incorrect checkpoints consumed before a lower exit.
- **Token fraction:** tokens used divided by the fixed hard budget.

A full-budget fallback has zero early-stop loss by definition but can still emit a wrong answer; that answer error is counted. A lower stop may carry both emitted-answer error and future-correct loss; these represent different operational failures.

## 6. Synthetic benchmark and independent calculation

The fixture contains 256 deterministic stratified tasks, 32 in each of eight regimes: early solved, quickly solved, delayed solved, never solved/low confidence, overconfident wrong, low-confidence but eventually solvable, oscillatory/delayed, and never solved/medium confidence.

The panel is deterministic and stratified, **not iid**. Therefore its UCBs diagnose arithmetic and candidate rankings only. The report sets `confidence_claim_eligible_under_declared_design=false` and `formal_theorem_verified=false`.

Independent JavaScript replay of the fixture gives a three-channel Hoeffding correction of 96,700 PPM (9.67 percentage points):

| Candidate | Upper empirical risk | Lower empirical risk | Emitted-answer error | Mean token fraction | Feasible at all three UCB budgets? |
|---|---:|---:|---:|---:|---|
| a_fast_aggressive | 12.50% | 10.94% | 75.00% | 28.81% | No |
| b_joint_balanced | 12.50% | 10.94% | 75.00% | 34.38% | No |
| c_late_conservative | 12.50% | 8.25% | 66.41% | 56.40% | No |
| d_full_budget | 0.00% | 0.00% | 25.00% | 100.00% | Yes |

At the declared 23% upper/lower budgets and 35% output-error budget, only the full-budget baseline passes all three numerical UCB thresholds. The experiment therefore demonstrates **no compute saving at the current answer-quality requirement**. That is a negative but useful result: the current confidence signal/candidate grid is insufficient. A stronger verifier or solvability/continuation-value signal is needed before claiming efficiency gains.

A separate Monte Carlo sanity check of the correction used `N=256`, `G=4`, 12 Bernoulli risk streams with true mean 0.5, delta=10%, and 20,000 independently generated calibration families. With the documented outward-rounded correction of 96,700 PPM, all 12 population means were simultaneously below their UCBs in 98.755% of families (95% Wilson interval 98.592%–98.900%). This is a synthetic check of the arithmetic under a simple iid design; it does not validate the stratified fixture, the LLM confidence signal, or a real deployment.

## 7. SOTA and industry comparison: protocol-matched, not one-number leaderboard

The current field has several strong but non-equivalent approaches. Their reported savings are not directly comparable because they vary by model, task, compute regime, budget metric and evaluation protocol. Symthaea should reproduce the methods on common tasks and report the full inference-system cost profile before making a superiority claim.

### Comparator matrix

| Method / reference | Reported contribution | How Symthaea should compare |
|---|---|---|
| Conformal Thinking, ICML 2026 | Calibrates an upper false-positive threshold and a lower farsighted premature-exit threshold against held-out risk targets. | Paper-matched two-step baseline; report both exit risks and actual emitted-answer error. Our joint 3G method is a distinct variant, not the paper's method. |
| Adaptive Thinking / Sonata, ICLR 2026 | Uses predicted self-consistency from prefill hidden states to route compute; the authors report 20–80% fewer thinking tokens at similar accuracy, or up to 5% accuracy improvement at similar token cost on their evaluated models/tasks. | Add a self-consistency/latent-signal baseline; test calibration transfer, inference overhead, hidden-state availability and whether verifier agreement is correlated. |
| Learning When to Think (2026 preprint) | Learns NoThink/Short/Long modes inside GRPO; reports MATH500 accuracy 0.782 vs 0.796 for the base while lowering mean response length from 4,796 to 2,811 tokens (41%); reports larger GSM8K savings in its setup. | Multi-mode learned-router baseline; test our three risk channels on exactly the same checkpoints/tasks and include training cost/amortization. |
| AdaCompute-LLM (2026 preprint) | Solve-then-learn constrained allocation with a Lagrangian oracle; reports up to 12.8% relative accuracy improvement on MATH under matched compute budgets and over 91% imitation accuracy. | Constrained-policy allocation baseline; distinguish average budget feasibility and per-task risk certification. |
| Selective verification over categorical structure (2026 preprint) | Ranks intermediate states and allocates verifier calls selectively; reports 44% fewer verifier calls on MATH than named sampling/search baselines in its experiment. | Verification-cost lane; account for verifier-call cost, correlation, and actual error detection—not just reasoning tokens. |
| When More Thinking Hurts, Findings of ACL 2026 | Finds diminishing marginal returns and overthinking, including reasoning runs that abandon previously correct answers. | Stress tests for correct-answer regression and emitted-answer error; measure the entire checkpoint trajectory, not just final accuracy. |
| Thinking Hard, Not Smart (2026 preprint) | Shows models can allocate a shared budget poorly across multiple tasks, front-loading compute by presentation order rather than value/difficulty. | Add a batch/portfolio budget benchmark; per-question stopping does not establish global resource-allocation quality. |
| Test-Time Scaling in Reasoning LLMs (2026 evaluation/reproducibility preprint) | Separates single-trajectory sequential scaling, leaf-level sampling/terminal reduction, and prefix-level search; emphasizes full-system cost accounting and replay. | Report depth, breadth and search as separate regimes; preserve exact replay artifacts and distinguish candidate-bank diagnostics from end-to-end system results. |

### Required benchmark lanes

**Lane A — Sequential depth:** one reasoning trajectory with upper/lower stopping and full-budget fallback. This is the current crate's scope.

**Lane B — Leaf-level breadth:** multiple independently sampled complete candidates, then voting or verifier-based terminal reduction. Measure generated candidates, duplicate work, aggregation error and all verifier cost.

**Lane C — Prefix-level search:** branching over unfinished reasoning prefixes. Measure tree width/depth, expansion cost, duplicate prefixes, verifier calls, and the cost of discarded branches.

**Lane D — Verifier allocation:** spend limited verification calls on intermediate states. Include uniform verification, confidence-ranked, residual/progress-ranked, self-consistency, independent-verifier and exact synthetic oracle arms. Correlated verifiers must not be counted as independent evidence.

**Lane E — Shared batch budget:** allocate a fixed total token/compute budget across multiple tasks with different difficulty and value. Randomize task presentation order, test order sensitivity, and compare against a finite-horizon allocation oracle. Report score/regret per batch; a per-query risk guarantee does not establish a global portfolio guarantee.

### Matched-cost reporting contract

Do not collapse all inference into “tokens.” For every task and batch, report:

- input/prompt tokens, output/reasoning tokens, verifier tokens and tool tokens separately;
- model forward passes, sampled candidates, expanded prefixes and verifier calls;
- wall-clock latency p50/p90/p95/p99, throughput and queue wait;
- measured GPU/CPU time, peak memory and energy when instrumentation is available;
- hard budget utilization, budget overrun count, and complete cost of the inference system;
- task correctness, three risk channels, subgroup/task-family risk, calibration error and regret to the exact oracle.

For the same reported result, retain the model/checkpoint, prompt, decoding parameters, reasoning regime, verifier identity, candidate grid, calibration split, held-out task IDs, tokenization and resource-accounting method. “Exact replay” and “same distributional behavior” are different reproducibility claims.

### Prospective targets to beat strong baselines

These are targets, not results:

1. At equal end-to-end cost, improve task success by at least 1 percentage point over the best paper-matched adaptive baseline **or** retain accuracy within 1 point while lowering total cost by at least 30%.
2. Keep all three risk UCBs within preregistered budgets on genuinely independent held-out task units; separately report task-family and difficulty strata. No subgroup may be silently hidden by the mean.
3. Reduce verifier calls by at least 30% versus uniform verification at matched emitted-answer error, with correlated-verifier ancestry preserved.
4. On shared-batch tasks, reduce regret to the exact budget-allocation oracle by at least 25% versus sequential first-come/first-served allocation, with randomized order and value/difficulty perturbations.
5. Keep hard compute overruns at zero and report p95/p99 latency; average token savings cannot excuse pathological tail latency.
6. Maintain a full-budget baseline and explicit abstention/fallback. If no policy meets all risk limits, return no feasible candidate rather than weaken the declared risk target.

### Research references and source audit

- Conformal Thinking (ICML 2026): https://proceedings.mlr.press/v306/wang26hd.html
- Adaptive Thinking / Sonata (ICLR 2026, Apple ML Research): https://machinelearning.apple.com/research/adaptive-thinking
- Learning When to Think (2026 preprint): https://arxiv.org/abs/2608.20256
- Adaptive Test-Time Compute Allocation via Constrained Policy Optimization (2026 preprint): https://arxiv.org/abs/2604.14853
- Adaptive Test-Time Compute Allocation via Learned Heuristics over Categorical Structure (2026 preprint): https://arxiv.org/abs/2602.03975
- When More Thinking Hurts (Findings of ACL 2026): https://aclanthology.org/2026.findings-acl.1199/
- Thinking Hard, Not Smart (2026 preprint): https://arxiv.org/abs/2608.07968
- Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility (2026 preprint): https://arxiv.org/abs/2608.04001

The numerical results in this comparison are claims reported by the respective authors in their own experimental protocols, not matched head-to-head measurements. The only current Symthaea result is the deterministic stratified synthetic diagnostic in Section 6, which demonstrates no compute saving at the declared answer-quality requirement.

## 8. Integration

- Symthaea cognitive loop proposes continue/stop and tool/verifier operations.
- #9882 exact resource-vector controller enforces hard token/time/memory/energy caps; those are not statistical alpha budgets.
- #10042 evaluates the risk of actions taken; #10052 evaluates compute-stop decisions.
- #10066 can compose the eventual three stopping claims only after a real validation design and theorem assumptions are verified.
- Symtropy can provide exact finite-horizon synthetic oracles; Mycelix can record provenance but does not make the risk decision.

## 9. Claim ceiling and next research

This is a joint-grid research variant; it does not inherit the paper's theorem by similarity. It does not verify iid/exchangeability, distribution stability, or numerical directed-rounding. The synthetic panel cannot issue a population confidence claim. Next steps are a genuine held-out task harness, independently reviewed high-precision UCB, theorem-to-code audit, and improved evidence/solvability signals. The present result says the current signals do not deliver compute savings at the declared answer-quality budget.

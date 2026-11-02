# Joint Compute-Risk Calibration v0.1

**Status:** independent Rust research implementation, deterministic synthetic panel and authored tests. No population-level confidence claim, paper-matched reproduction, Rust test run, or real compute-saving result is claimed.

Tracking: [NEURO-SILICON-021 / #10052](https://github.com/Luminous-Dynamics/symthaea/issues/10052).

## 1. Research anchor

Wang et al., *Conformal Thinking: Risk Control for Reasoning on a Compute Budget*, ICML 2026, PMLR 306: https://proceedings.mlr.press/v306/wang26hd.html

The paper formulates a two-threshold stopping policy. It uses an upper threshold lambda-plus for stopping when confidence is high, and a dynamic lower threshold lambda-minus(t) for stopping early when an instance appears unsolvable. These two stopping paths represent different failure modes and should not be collapsed into one average.

Upper risk: an upper-threshold stop emits an incorrect answer.

Lower risk: a lower-threshold stop abandons an instance even though later reasoning checkpoints would contain a correct answer. Its loss is the fraction of remaining checkpoints that are correct:

    L_minus(t) = 1/(T-t+1) * sum_{k=t}^T I[checkpoint k is correct]

The paper's dynamic lower threshold uses a sigmoid between a lower floor l and the upper threshold u=lambda-plus:

    lambda_minus(t) = l + (u-l)/(1 + exp(-c * (tokens_t - s * budget)))

Paper source PDF: https://arxiv.org/abs/2602.03814

### Author-code audit

Public repository: https://github.com/xidulu/reasoning_risk_control

Pinned main commit inspected: `6551e3000dcda3d76491f30cc687d64618807c5b` (2026-05-14). Its README calls the code preliminary and says full code/assets will be released soon. The inspected tree has no LICENSE file. Its lower-threshold helper has an unfinished path (references an undefined `thresholds` variable and does not use the computed `lower_thresholds`). This Rust implementation is independent and does not copy that code. The paper is the algorithmic authority.

## 2. Paradigm shift: jointly calibrate the whole stopping policy

The paper describes a two-step procedure that first chooses the upper threshold and then fits the lower-threshold function. The paper notes that this decoupling loses some theoretical rigor; its appendix also discusses how a lower-threshold filter can affect the upper false-positive rate when it removes correct and incorrect validation examples at different rates.

Symthaea v0.1 instead freezes a complete finite grid of joint upper/lower candidates before examining validation outcomes. Each candidate is evaluated on every held-out task. All two-channel candidate claims are corrected simultaneously, and the whole grid is enumerated. There is no binary search and no assumption that risk is monotonic in a threshold parameter.

This is a new joint-grid variant inspired by the paper, not a reproduction of the paper's two-stage algorithm.

## 3. Simultaneous finite-grid correction

Let G be the number of frozen joint candidates and N the number of validation tasks. For each candidate i, compute two bounded task losses in [0,1]: upper wrong-answer loss and lower premature-abandon loss.

There are 2G one-sided risk claims. Allocate statistical failure probability delta/(2G) to each claim. Hoeffding's inequality gives the shared correction:

    c(N,G,delta) = sqrt( ln(2G/delta) / (2N) )

Then for each candidate:

    UCB_plus_i  = mean(L_plus_i)  + c
    UCB_minus_i = mean(L_minus_i) + c

Candidate i is feasible only if both bounds are within their independently preregistered operational budgets:

    UCB_plus_i  <= epsilon_plus
    UCB_minus_i <= epsilon_minus

A union bound over all 2G claims controls the familywise statistical failure probability by delta, conditional on the task-level loss observations being independent and identically distributed from the intended validation population, each loss being bounded, and the candidate grid being fixed before labels are inspected. Dependence between the two losses on the same task is allowed; the union bound does not multiply probabilities.

Among feasible candidates v0.1 selects the lowest average token fraction, with candidate ID as a deterministic tie-break. If none is feasible, it reports no candidate and never relaxes either budget.

### Numerical implementation boundary

The lower sigmoid and Hoeffding logarithm use f64. Threshold and UCB outputs are rounded outward to integer PPM, with one extra PPM added to the Hoeffding correction. This is a deterministic engineering guard, not directed-interval arithmetic or a formal numerical proof. A production certificate needs an independently reviewed high-precision/directed-rounding implementation.

## 4. Distinct losses and diagnostics

- **Upper wrong-answer loss:** 1 only when the upper threshold fires and the emitted answer at that checkpoint is incorrect; otherwise 0.
- **Lower premature-abandon loss:** at a lower-threshold exit, the fraction of remaining checkpoints (including the exit checkpoint) that would be correct; otherwise 0.
- **Emitted-answer error rate:** descriptive error of the answer actually emitted at any exit, including lower stops and full-budget fallback. It is reported separately and is not one of the two calibrated risks in v0.1.
- **Upper efficiency loss:** normalized delay after the first correct checkpoint for upper exits.
- **Lower efficiency loss:** fraction of incorrect checkpoints consumed before a lower exit.
- **Token fraction:** average tokens used divided by the fixed hard budget.

A full-budget fallback has zero early-stopping loss by definition but may still emit a wrong answer; the emitted-answer error metric keeps that visible. A lower stop can have both an emitted wrong answer and positive future-correct loss; these are different diagnostics.

## 5. Synthetic benchmark fixture

The fixture contains 256 deterministic stratified tasks, 32 in each of eight regimes:

1. early solved;
2. quickly solved;
3. delayed solved;
4. never solved with low confidence;
5. overconfident wrong answer;
6. low-confidence but eventually solvable;
7. oscillatory confidence with delayed solution;
8. never solved with medium confidence.

The four preregistered joint candidates explore aggressive, balanced, late/conservative and full-budget stopping behaviors. The panel is intentionally stratified and deterministic; it is **not an iid sample**. Consequently, its Hoeffding UCB numbers test arithmetic and candidate ranking only. The report sets `confidence_claim_eligible_under_declared_design=false` and `formal_theorem_verified=false`.

## 6. Independent reference calculation

An independent JavaScript replay of the fixture generator and stopping rules yields:

| Candidate | Upper empirical risk | Lower empirical risk | Simultaneous UCB correction | Mean token fraction |
|---|---:|---:|---:|---:|
| a_fast_aggressive | 12.50% | 10.94% | 9.25 pp | 28.81% |
| b_joint_balanced | 12.50% | 10.94% | 9.25 pp | 34.38% |
| c_late_conservative | 12.50% | 8.25% | 9.25 pp | 56.40% |
| d_full_budget | 0.00% | 0.00% | 9.25 pp | 100.00% |

All four satisfy the *numerical* 23% upper and lower UCB thresholds on this synthetic panel. The selection rule therefore chooses a_fast_aggressive by its lower token use. This is a panel-specific result, not evidence that it saves compute or meets a population risk target in deployment. The emitted-answer error rate is also reported per candidate in the machine-readable receipt.

## 7. Comparators and industry positioning

Required benchmark arms:

- fixed full-budget reasoning (quality/control baseline);
- fixed 25%, 50%, 75% compute budgets;
- confidence-only upper-threshold stopping (heuristic baseline);
- paper-faithful Conformal Thinking upper/lower two-step method, once a licensed complete reproduction is available or independently reconstructed from the paper;
- Symthaea joint-grid simultaneous-UCB method;
- exact finite-horizon oracle stopping (upper bound only; sees the full outcome trajectory and is not deployable).

Report both risk channels, emitted-answer error, accuracy, task success, mean/p50/p90/p95/p99 tokens, latency, verifier/tool calls and regret to the exact stopping oracle. Include per-task-family results; aggregate averages must not hide the low-confidence-solvable or overconfident-wrong regimes.

Prospective acceptance targets (not achieved results):

- at least 20% lower median token use than the fixed-budget baseline at matched task success;
- no more than 1 percentage point deterioration in emitted-answer accuracy;
- both simultaneous UCBs below their preregistered operational budgets on a genuine held-out iid validation sample;
- zero hard token/time/memory budget overruns (enforced separately by #9882);
- no task family with an unacceptable risk hidden by aggregate performance;
- at least 90% of oracle-achievable compute savings without more than 1pp risk-budget slack, once an exact finite-horizon oracle is defined.

## 8. Integration

- Symthaea cognitive loop proposes continue/stop and tool/verifier operations.
- #9882 exact resource-vector controller enforces hard token/time/memory/energy caps; these are not statistical alpha budgets.
- #10042 policy-risk work evaluates the risk of actions taken; this issue evaluates compute-stop decisions.
- #10066 can compose the eventual two stopping risk claims only after the validation design and certificate assumptions are genuinely satisfied.
- Symtropy can provide exact finite-horizon synthetic oracles; Mycelix can record provenance but does not make the risk decision.

## 9. Claim ceiling and next work

This implementation is a joint-grid research variant. It does not inherit the paper's theorem by similarity, verify iid/exchangeability or distribution stability, or prove numerical correctness. The built-in synthetic panel deliberately cannot issue a population confidence claim. Next steps are a true held-out task harness, an independent high-precision UCB verifier, paper-to-code theorem audit, and a benchmark against fixed compute and paper-faithful stopping baselines.

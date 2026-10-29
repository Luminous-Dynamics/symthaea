# CMAQ-002: Closed-Loop Planning and Distribution Shift

**Status:** standalone Rust evaluator prototype; no agent qualification results.

CMAQ-001 asks whether outcome-aware memory and action predictions improve one-step decisions. This companion asks a harder question: when a planner optimizes a learned model and then acts in the real evaluation dynamics, does the apparent model value survive closed-loop execution?

The evaluator performs finite-horizon, deterministic, model-based receding-horizon planning. It repeatedly plans through the candidate transition/reward/safety model, executes only the first action in the evaluator's ground-truth environment, then replans from the actual successor state. This exposes model errors that an optimizer can actively seek and measures the resulting distribution shift.

## Run

From this directory, use Rust 1.96.0:

- Test: cargo test --locked
- Format check: cargo fmt -- --check
- Score the exact-model fixture: cargo run --locked -- --environment fixtures/environment.tsv --model fixtures/model-exact.tsv --support fixtures/support.tsv --starts fixtures/starts.tsv
- Score the optimizer-exploitation negative control by replacing model-exact.tsv with model-exploit.tsv. It should select the overvalued branch and report constrained regret 3400 milli-utility and a model-to-rollout optimism gap of 7000.

The program emits a deterministic TSV with one row per task. The command requires four inputs: the ground-truth deterministic transition table, candidate predictions for every state-action pair, evaluator-owned support counts, and task start/horizon settings. A duplicate, missing, or unknown key, malformed probability, unsupported state identifier, or out-of-range setting fails closed.

## Input contract

All TSV headers and column order are strict. Reward units are integer milli-utility. Probabilities and safety thresholds use parts per million (0 to 1,000,000). State and task identifiers are whitespace-free tokens.

- Environment: task_id, state_id, action_id, next_state_id, reward_milli, safe, done.
- Model: task_id, state_id, action_id, predicted_next_state_id, predicted_reward_milli, predicted_safe_ppm, predicted_done.
- Support: task_id, state_id, action_id, support_count. This file belongs to the evaluator, not the candidate. Zero means the exact state-action pair was absent from the declared support set.
- Starts: task_id, start_state_id, horizon, gamma_ppm, safety_threshold_ppm.

The model must predict every environment state-action key exactly once. Its predicted successor must name a state in that task's evaluator-defined finite state space. The environment table must contain the complete deterministic transitions for the simulator lane. Do not fabricate unchosen real-world outcomes: unchosen action outcomes are available here only because this is a synthetic/simulator evaluation.

## Metrics

- Raw realized return: discounted reward on the actions actually executed in the ground-truth environment.
- Constrained return: the immediate utility contribution of an actually unsafe action is zero. Safety events remain hard-gate failures; higher reward cannot compensate for an unsafe action.
- Oracle safe return: evaluator-only finite-horizon optimum constrained to actually safe actions and with abstention utility zero. It is an upper bound, never a candidate baseline.
- Constrained regret: max(0, oracle safe return minus constrained return).
- Model plan value: value predicted for the candidate model's best safe plan at the initial state.
- Model-to-rollout optimism gap: initial predicted plan value minus realized constrained return. This is a diagnostic, not a claim of a universal policy-exploitability metric.
- Unsafe executed action count: hard gate; any nonzero count fails.
- Unsupported executed action count/rate: uses evaluator-owned support metadata, not a candidate's self-reported count.
- Executed-action reward MAE and safety Brier score: local prediction diagnostics on state-action pairs actually used by the planner.
- Transition mismatches: executed actions whose predicted successor or done flag differs from the evaluator.

An unsafe transition can still lead to further simulated steps so the full trajectory remains observable. Its immediate contribution to constrained return is zero, and the safety hard gate still fails. This is an evaluation convention, not a claim that all physical environments recover from unsafe actions.

## Qualification protocol

Use this evaluator as one lane, not a universal benchmark score. Pair every candidate and baseline on identical task/seed/horizon tuples. Report task-clustered bootstrap intervals, per-mechanism and worst-family results, all unsafe events, and the full utility/calibration/cost tuple. The current harness emits task-level values but does not calculate confidence intervals or wall-clock, memory, energy, or model-training costs.

Required baselines and controls:

1. A constant/abstain policy and global reward/safety prior.
2. Nearest-support episode retrieval and exact state-action lookup with explicit abstention on missing keys.
3. A transition model trained on the same support data without HDC or recurrence.
4. The same transition/reward model with planning disabled or horizon 1.
5. The intended Symthaea configuration, plus matched HDC-only, recurrent-only, memory-only, world-model-only, and planner ablations.
6. An evaluator-truth model as a simulator upper-bound diagnostic only; it is not a deployable comparison.

Suggested horizons: 1, 3, 5, and 10. Freeze the safety probability threshold, discount, support definition, candidate compute budget, and qualification split before inspecting held-out outcomes. Start with at least 10 development seeds and at least three independently generated causal-law families; use a separate evaluator-controlled qualification panel for claims of cross-law transfer. Use task/mechanism-first resampling so multiple states from one task are not treated as independent observations.

Negative controls should shuffle support reward labels within action while keeping transition/safety labels intact; corrupt predicted successors; inject overconfident false-safe actions; and give an optimizer a deliberately optimistic unsupported branch. The expected outcomes are loss of decision quality, detected transition/safety failures, and higher unsupported-action/exploitation diagnostics. If shuffled or causally irrelevant support performs as well as valid support, the memory-causality hypothesis is not supported.

## How this fits industry evaluation

Recent work argues that world models should be evaluated by their decision utility under intervention and closed-loop execution, not prediction loss alone. The ICLR 2026 World-In-World platform is a relevant external closed-loop comparator; published memory suites such as AMA-Bench and MemoryArena test complementary memory capabilities. Run their published protocols independently and do not compare raw scores across different datasets or metrics. CMAQ-002 is a small deterministic harness for a specific causal chain: model predictions -> optimizer choices -> actual environment rollout -> constrained utility.

This prototype does not yet model stochastic transitions, continuous observations/actions, partial observability, model ensembles, real robotics, or real-world authenticated receipts. It has not yet been compiled or run in this authoring environment, and no Symthaea model or industry baseline has been scored. The smoke fixture is a known-answer test only. No SOTA, production-safety, or empirical improvement claim is made.

# Multi-Step Robust Viability Tubes V1

Status: isolated Rust prototype and deterministic horizon-frontier runner; exact-head Rust qualification is required before these tests count as passing.

## Research question

Can the allostatic viability shield prevent the one-step blind spot where a proposed action is safe on the next transition, but every continuation after that action crosses a safety boundary?

## Paradigm shift: from a safe next state to a viable trajectory

The one-step envelope in `robust_viability.rs` only certifies the immediate successor. That is insufficient when recovery has delay, reserves are depleted over multiple steps, or a short burst creates an unsafe intermediate state that cannot be repaired afterward.

The new module `multistep_viability.rs` enumerates discrete action continuations up to a bounded horizon. A first action is certified only when there is at least one common open-loop continuation sequence whose full reachable interval remains within both declared limits at every intermediate transition.

For horizon (H), an action (a_0) is certified when:

\[
\exists(a_1,\ldots,a_{H-1})\quad
\forall t\in\{1,\ldots,H\},\quad
\underline r_t\ge r_{\min}
\;\land\;
\overline \ell_t\le \ell_{\max}.
\]

Each interval tube is propagated with the declared rectangular action-effect bounds. Any unsafe intermediate tube terminates that candidate sequence; a future recovery action cannot retroactively make the sequence safe.

## Implementation

- `HorizonViableActionSet` reports the actions that have a feasible full-horizon continuation.
- `evaluate_horizon_action_set` supports horizons 1 through 8, and fails closed for invalid horizon values.
- `evaluate_horizon_action_set_with_budget` reports expanded search nodes, transition evaluations, and maximum reached depth. A budget-exhausted search returns no certificate rather than a partial viable set.
- `MAX_EXPANDED_SEARCH_NODES = 10,000` exceeds the worst-case exhaustive 3-action tree at horizon 8 (9,840 continuation nodes under this accounting); the runtime metric is still emitted per operating point and dynamics panel so the real search cost is visible.
- Horizon 1 is required to exactly match the existing one-step viability set.
- The search respects control authority at every continuation step. If emergency recovery is unavailable, it cannot be selected as either the first action or a future continuation.
- If no first action has a robustly safe continuation, the evaluator returns an empty set. The filter returns `None`; it does not label a best-effort action safe.

This is an intentionally conservative **open-loop continuation** check: the future action suffix must work for every state represented by the propagated interval, with no branching based on future observations. A richer controller could use a state-contingent policy and certify more cases, but that needs a formal treatment of partial observability and observation timing.

## Horizon-frontier benchmark

`allostatic-multistep-viability-bench` emits a deterministic Cartesian panel of five operating states × 24 paired synthetic dynamics profiles × horizons 1, 2, 3, 5 and 8, for 600 records. It reports each first action's certification, whether the viable set is empty, expanded nodes, transition evaluations and the maximum search depth. The qualifier runs it twice and requires byte-identical output; it also checks horizon-1 equivalence, nested viable sets, valid schemas, unique keys and the declared search budget. The profiles are shared across operating states to make per-state results paired rather than independently sampled.

The executable is intended to measure horizon-induced capability loss and search cost before integrating it into the mismatch frontier. It is not a simulation of reward-optimal action sequences and does not claim a real-world runtime guarantee.

## Synthetic falsification fixtures

The deterministic unit tests cover seven contracts:
1. Horizon-1 results match the existing one-step viable-action set.
2. A Push that is safe for one step is rejected at horizon 2 when every possible continuation violates the load limit.
3. Increasing the horizon can remove certified actions but cannot add a new one.
4. Emergency-recovery authority is enforced across the whole plan, not just at the first step.
5. Invalid horizons and an already-unsafe initial tube fail closed.
6. A future EmergencyRecover cannot excuse an unsafe intermediate tube.
7. An independent short-horizon action-sequence enumerator agrees with the production recursive search over multiple state tubes, dynamics envelopes and authority settings.

The test dynamics are deliberately synthetic. These tests check the interval-propagation/search contract; they do not demonstrate a safe real-world controller, model calibration, or SOTA performance.

## Next benchmark stage

Integrate the evaluator into the mismatch frontier and extend the true-versus-estimated simulator to compare horizons 1, 2, 3, 5 and 8 under the same held-out perturbations. Keep the four matched calibration variants (global/action-specific × 90%/95%) and report results per operating point.

Report at minimum: realized multi-step false-certification rate, action availability/false-veto rate, no-certified-action rate, task reward, cumulative and p95 violation exposure, recovery with right-censoring, runtime/candidate-sequence count, and calibration coverage on genuinely untouched exchangeable episodes. A longer horizon that only rejects more actions should not be declared better without the safety–availability frontier.

## Proposed gates (targets, not achieved results)

- Zero multi-step robust false certifications on the declared in-envelope synthetic panel, including every intermediate step.
- The horizon-1 set matches the existing one-step filter on every tested state/action.
- As horizon grows, certified sets are nested (or the qualification fails).
- No unsafe intermediate state can be hidden by a later recovery action.
- Maximum runtime work must stay within 10,000 expanded nodes per first-state/horizon evaluation; budget exhaustion is a failed certificate, not a success.
- Report the rate of empty viable-action sets as a first-class operational outcome.
- For an external benchmark, pin Safe-Control-Gym/Safety-Gymnasium version and task details, and compare only on a task with mappable state, action and constraint semantics. The current allostatic scalar simulator is not itself comparable to cart-pole, quadrotor, or robot navigation scores.

## Industry comparison and target baselines

A fair next comparison should use the same dynamics, state/action constraints, disturbance seeds and action frequency for all filters. `Safe-Control-Gym` supports CartPole and quadrotor control with symbolic constraints and injectable input, measurement and parameter/dynamics disturbances; it is a suitable external bridge once a documented mapping from this scalar allostatic state to a physical plant exists. Safety-Gymnasium can provide constrained-control tasks, while SafeRLEval motivates reporting violations by episode frequency, normalized deviation, severity and distribution rather than average cost alone. See [Safe-Control-Gym](https://github.com/learnsyslab/safe-control-gym), [Safety-Gymnasium](https://safety-gymnasium.readthedocs.io/stable/introduction/about_safety_gymnasium.html), and [SafeRLEval / evaluation-metrics paper](https://arxiv.org/abs/2609.15315).

The initial quantitative target is not “beat SOTA” on an unrelated task. It is: zero false certificates on in-envelope trajectories under the same transition contract, no more than 5% false vetoes on truly safe actions where achievable, explicit no-viable-action rate, and worst-case evaluation cost below the fixed budget. Later tests need held-out exchangeable or physically justified calibration data, confidence intervals, and matched-task baselines before a performance-superiority claim.

## Scope boundary

This code adds a separate research evaluator. It does not change the Symthaea live loop, replace prediction/task losses, or alter qualified interoception v0.1 semantics. No claim of formal safety outside the declared reachable-set assumptions is made.
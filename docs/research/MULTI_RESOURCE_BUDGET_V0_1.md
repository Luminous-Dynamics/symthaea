# Vector-Valued Acquisition Budgets v0.1

**Status:** Typed resource registry, complete action-cost vectors, component-wise hard budget checks and Pareto action frontier are committed on `research/multi-resource-budget-v0.1`. Compile/format/test execution and operational integration remain outstanding.

**Tracking:** [#9421](https://github.com/Luminous-Dynamics/symthaea/issues/9421). Uses portfolio-class objective-measurement VoC [#9419](https://github.com/Luminous-Dynamics/symthaea/pull/9419), run-wide pruning-risk ledger [#9329](https://github.com/Luminous-Dynamics/symthaea/pull/9329) and qualification-token follow-up [#9388](https://github.com/Luminous-Dynamics/symthaea/issues/9388).

## 1. Why cost must remain a vector

An objective-measurement action may consume:
- GPU time;
- CPU time;
- target-device occupancy time;
- instrument time;
- joules used while profiling;
- money / cloud credits;
- storage for traces;
- operator time;
- scarce parallel/concurrency slots.

These units cannot be added without an explicit exchange-rate model. A GPU-second is not an instrument-second; energy consumed while measuring is not the energy-per-inference objective. Arbitrary weighted sums hide the policy preference and can rank an action incorrectly for a deployment with a different bottleneck.

The v0.1 contract keeps all resource dimensions separate. The scheduler returns the non-dominated feasible action frontier, not one scalar winner. A final single-action choice requires an explicit budget profile, lexicographic priority or separately versioned resource-conversion model.

## 2. Resource registry

Each resource definition includes:
- stable resource ID;
- unit;
- resource class (accelerator time, CPU time, target-device time, instrument time, energy, money, storage, operator time, concurrency slots, custom);
- SHA-256 for the cost model/instrument accounting that produces quantities.

The registry digest is canonicalized by resource ID. Changing a unit or cost model changes the digest. Registry identities are required by both action-cost and budget vectors.

## 3. Complete action-cost vectors

Every action must state one entry per registered resource, in the frozen registry order:
`(resource_id, unit, units, cost_model_sha256)`.

Zero must be explicit when an action does not consume a resource. Omission is not zero and makes the vector invalid. Each entry must match the exact resource ID, unit and cost model in the registry. At least one resource quantity must be positive.

Integer quantities avoid floating-point unit conversions and rounding. If a cost is measured as a real-valued duration or energy, the importer must pre-register a fixed integer resolution (e.g. microseconds or microjoules) and perform checked conversion in the acquisition layer.

A quantity's unit is the *resource being consumed*. It is independent of the objective value unit. Example: energy per inference (mJ/inference) is a device objective; profiling energy spent while measuring it (microjoules) is a resource budget.

## 4. Hard vector budgets and reserves

A budget binds the resource registry and explicitly supplies one limit for every resource:
- total units;
- already-spent units;
- protected reserve units.

Remaining headroom is computed independently for every resource:

`headroom_r = total_r - spent_r - reserve_r`

Overflow in spent-plus-reserve or an amount exceeding the total budget is a hard error. An action is feasible only when its cost fits every headroom dimension. No borrowing across dimensions occurs, and the scheduler never assumes exchange rates.

If the same instrument is exclusive or has a concurrency limit, model it as an additional resource with explicit integer capacity units. More advanced scheduling with overlapping intervals is a follow-on; v0.1 is an additive budget/capacity check.

## 5. Pareto action frontier

Each action supplies:
- expected reduction in posterior portfolio-class Pareto regret;
- randomized-QMC numerical standard error;
- exact complete resource-cost vector.

The scheduler computes a normal-approximation lower bound:
`LCB = expected_regret_reduction - 1.96 * numerical_standard_error`.

An action with a non-positive LCB is not eligible. This is a conservative diagnostic under the normal approximation; it is not a formal confidence guarantee and not a substitute for empirical model qualification.

Among feasible positive-LCB actions, action A dominates B only when:
- A's LCB benefit is at least B's;
- A consumes no more of any resource dimension;
- A is strictly better on at least one benefit or resource dimension.

All non-dominated actions are returned. When several incomparable actions remain, the caller must provide an explicit priority/budget policy rather than letting a hidden scalarization decide.

## 6. Tests and boundaries

Fourteen authored Rust tests cover:
1. canonical registry digest under resource-order permutation;
2. changed cost model / stale registry rejection;
3. per-resource headroom after spent amounts and reserves;
4. unknown or reordered budget dimensions;
5. overflow-safe reserve arithmetic;
6. spend-plus-reserve above total budget;
7. missing resource dimension (no implicit zero);
8. resource unit and cost-model mismatch;
9. all-zero action cost vector;
10. component-wise feasibility without scalarization;
11. Pareto dominance between actions;
12. non-positive conservative benefit lower-bound abstention;
13. stale registry and action ID mismatch;
14. non-finite benefit and duplicate action IDs.

The current execution environment has no `cargo`, `rustc` or `rustfmt`; these test declarations have not been compiled or executed. The resource vector primitive is not yet wired into the portfolio VoC operational planner, which remains gated on #9388 and #9329/#9104.

## 7. Benchmarks and acceptance

Compare the full portfolio-regret/value/resource frontier—not a hidden weighted scalar—against:
- scalar weighted-cost VoC using a declared exchange-rate model;
- one-resource-at-a-time action schedules;
- lexicographic resource priorities;
- constrained Bayesian optimization / cost-aware HVKG;
- FlexiBO-style objective-specific measurement;
- mixed-integer allocation for deterministic resource costs;
- Pareto action-frontier scheduling with numerical uncertainty.

Report:
- expected regret reduction, lower bound and QMC SE;
- cost in every resource dimension;
- feasible action count, blocked-resource reasons and protected reserves;
- device/instrument utilization;
- final portfolio regret and hypervolume;
- budget utilization and overrun/abstention frequency.

A resource-vector system may legitimately return several incomparable actions. Report that frontier. A single recommendation is justified only after a policy preference or resource budget profile is declared.

## 8. Claim ceiling

The module prevents cross-unit arithmetic and enforces hard resource caps component-wise. It does not decide how a user values instrument time versus energy or money. It does not schedule concurrent jobs, reserve resources transactionally across distributed workers, enforce empirical VoC qualification tokens or the run-wide pruning-risk ledger. No benchmark performance or SOTA improvement is claimed.

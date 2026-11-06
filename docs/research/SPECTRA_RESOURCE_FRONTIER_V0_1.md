# SPECTRA Multi-Resource Feasible Acquisition Frontier v0.1

**Status:** deterministic Rust primitive and test declarations; not compiled or executed.
**Research parent:** #10772
**Method ID:** spectra-componentwise-resource-feasible-pareto-frontier-v1

## Why a vector, not a scalar

An experiment can consume replication units, CPU milliseconds, GPU milliseconds, elapsed wall time, human/participant seconds and money. These quantities are not naturally commensurable. A weighted sum introduces policy choices (and hidden trade-offs) before a decision-maker has said what matters.

The module therefore returns the componentwise Pareto frontier of candidates whose conservative upper-cost envelopes fit every remaining resource budget.

## Contract

A candidate contains:
- a unique stable candidate ID;
- projected certificate-gap contraction (a deterministic search signal, not an inferential guarantee);
- a conservative upper-cost vector;
- explicit qualification for every resource dimension;
- a currency code bound to the frozen budget currency.

Resource dimensions:
- simulation replication units;
- CPU milliseconds;
- GPU milliseconds;
- wall-clock milliseconds;
- human/participant seconds;
- currency micro-units in one frozen ISO-style currency code.

Unknown cost is not zero. Each of the six dimensions must be explicitly qualified; a known zero is represented by an explicit qualification flag and a zero amount. Candidate currency must match the contract currency. Unknown or mismatched dimensions fail closed.

The upper-cost envelope should include fixed setup, simulator initialization, data transfer, retries, orchestration, verification and other costs that are actually incurred by the experiment. The library cannot independently authenticate this evidence.



### Wall-time accounting boundary

CPU/GPU milliseconds, replications, human seconds and currency are additive charges. In v0.1, wall_millis is also treated as an additive charged duration under a frozen serialized execution plan. It is not a parallel-job makespan estimator. If jobs execute concurrently, an adapter must compute the schedule-level maximum elapsed path (including dependencies, queues and retries) separately; it must not sum parallel task durations and call the result wall-clock elapsed time.

## Pareto definition

For candidate i, benefit b_i, upper cost vector c_i, and remaining budget B:

Feasible(i) iff c_ij ≤ B_j for all j.

Candidate i dominates k iff:
- b_i ≥ b_k;
- c_ij ≤ c_kj for every resource dimension j; and
- there is a strict benefit or cost improvement.

If benefit and all cost dimensions are exactly equal, the lexicographically smallest candidate ID is the deterministic canonical representative.

Return:
- the complete set of feasible non-dominated candidates;
- all candidate dispositions: OnParetoFrontier, Dominated, or ExceedsRemainingBudget;
- the dominator ID for dominated candidates;
- the frozen currency and resource budget.

No hidden scalarization is used. A downstream registered decision policy may select among frontier points, but the policy must be explicit and separately replayed.

## Rust implementation

File: tools/spectra-uvsd-fit/src/resource_frontier.rs

- Componentwise ResourceUnits with checked addition and subtraction.
- Conservative ResourceCostUpperBound with explicit six-dimensional qualification.
- ResourceFrontierContract with version, currency and remaining budget.
- Deterministic Pareto filtering, stable tie handling and dispositions for every candidate.
- Currency-bound ResourceChargeLedger with checked accumulation and remaining-budget computation from the original total budget. The frontier contract's remaining_budget is already post-charge and must not be decremented a second time.
- Independent replay by recomputing the entire frontier and comparing every output field.

## Tests declared

- Preserve incomparable candidates and filter a dominated candidate.
- Canonicalize exact benefit/cost duplicates by stable ID.
- Enforce each of the six budget axes separately.
- Input-order invariance and replay mutation rejection.
- Reject missing/unknown cost dimension, currency mismatch, malformed IDs, duplicates, nonfinite/negative benefit and invalid currency.
- Checked ledger overflow and budget overrun.

Rust test declarations are not execution results. cargo, rustc and rustfmt are unavailable in the current environment. No compile/test/fmt/clippy/CI or performance pass is claimed.

## Locked benchmark

Compare replication-only budgets, a pre-registered scalar-weight baseline (diagnostic only), componentwise Pareto filtering, exact small-N oracle, and robust upper-cost filtering under cost-estimation error.

Sweep resource correlations, bottleneck rotations, batch/setup overhead, cost uncertainty, CRN atomicity, and budget exhaustion. Report:
- frontier precision/recall vs exact oracle;
- out-of-budget frontier leakage (target: zero);
- cost-envelope underestimation / coverage;
- dominated fraction and frontier size;
- utilization of each resource dimension;
- decision opportunity loss under pre-registered downstream policies;
- time-to-valid-certificate;
- p50/p95/p99 frontier and replay runtime.

The central acceptance is not "one scalar score wins". It is that SPECTRA exposes the feasible trade-off surface faithfully, does not exceed any declared hard budget, and lets a separately registered policy make the final choice without hiding resource trade-offs.

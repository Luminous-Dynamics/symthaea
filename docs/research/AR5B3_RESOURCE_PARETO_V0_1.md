# AR-5B3 — multi-resource Pareto allocation v0.1

**Status:** single-worker planner and resource ledger authored; tests not compiled or executed.  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/9338  
**Implementation:** crates/domains/symthaea-architecture-engineering/src/resource_pareto.rs  
**Dependency:** AR-5B/AR-5B2 direction allocation and Coral scheduler comparison.

## Why this is separate from Coral

The Coral Discovery formula allocates evaluation slots using a direction yield per evaluation and a UCB-style exploration bonus. It is intentionally not a multi-resource optimizer. Symthaea's direction board v0.1 also allocates evaluation-count slots. Both remain important baselines.

AR-5B3 asks a different question: given hard limits on wall time, CPU, GPU, memory, tokens, tool calls, evaluator calls and energy, which experiment should run next?

Do not divide the score by a weighted cost sum and call the result resource-optimal. A GPU second, a joule, a byte of peak memory and one external tool call are not interchangeable without a declared economic model. v0.1 therefore keeps constraints and Pareto trade-offs separate.

## Implemented v0.1 surface

### Resource contract

ResourceAmounts represents estimates and actual measurements. Each dimension is optional:
- wall time (ms);
- CPU time (ms);
- GPU time (ms);
- peak host memory (bytes);
- peak accelerator memory (bytes);
- energy (millijoules);
- input/output tokens;
- tool calls;
- evaluator calls;
- candidate executions.

None means unknown. It must never be interpreted as zero. ResourceLimits carries independent optional hard caps. A constrained dimension must have a known conservative upper estimate before an action is eligible.

### Feasibility and Pareto selection

Each candidate carries:
- verifier-defined action and direction IDs;
- predicted normalized capability improvement mean;
- calibrated gain standard error;
- number of prior verified attempts;
- a conservative upper bound for every resource dimension.

The acquisition heuristic is:

a_i = mean_gain_i + beta * standard_error_i / sqrt(n_i + 1)

This is an uncertainty-aware proposal score, not a calibrated guarantee. The allocator first removes all candidates that violate any hard resource limit or have unknown estimates for a constrained dimension. It then computes a Pareto frontier that maximizes acquisition while minimizing every resource dimension independently.

An action dominates another only if its acquisition is no worse and every comparable resource cost is no greater, with at least one strict improvement. Unknown versus known cost is incomparable; it is never treated as free. The full non-dominated frontier is retained in the plan.

A deterministic acquisition-then-ID tie-break selects one next action from the frontier. That tie-break does not delete the alternatives or claim that the chosen action is globally optimal.

### Resource ledger

A measured ResourceActionReceipt binds receipt ID, action ID, and measured resource vector with a domain-separated digest. ResourceBudgetState stores complete receipts and reconstructs its cumulative ledger from them at every validation.

Consumable metrics (wall/CPU/GPU time, energy, tokens and calls) are accumulated. Memory values are tracked as sequential high-water peaks in this single-worker v0.1 model. Duplicate receipt/action IDs, digest tampering, counter overflow, unknown use under a hard cap, ledger/receipt mismatch and hard-budget overruns fail closed.

The receipt digest is not an authenticated signature. A production evaluator must independently sign or otherwise authenticate the receipt against a trusted identity and pinned environment. The allocator currently trusts the resource vector supplied by its caller.

## Tests authored

Ten tests cover:
1. unknown/over-limit cost estimates are infeasible;
2. CPU/GPU/energy trade-offs remain incomparable on the frontier;
3. strictly dominated actions are removed;
4. unknown energy fails closed when energy is hard-budgeted;
5. cumulative and peak resource budgets are enforced;
6. unknown budgeted measurements and receipt tampering fail;
7. unknown unbudgeted energy remains unknown, not zero;
8. ledger values reconcile with complete receipts;
9. plan digests detect frontier tampering;
10. deterministic planning and duplicate action rejection.

No test or formatting result is claimed until the exact branch head is compiled and executed.

## Explicit limitations

- Single-worker / one-step planning only. There is no atomic multi-worker reservation, concurrency accounting or refund protocol.
- Peak memory is a sequential high-water mark. Parallel peak memory must be modeled by a concurrency-aware reservation service.
- The acquisition heuristic is a baseline; gain uncertainty calibration is not yet validated.
- Resource estimates are assumed conservative upper bounds; their coverage must be measured.
- Energy may be unknown. If energy is constrained, an unknown energy estimate is infeasible; if unconstrained, it remains explicitly unknown in the ledger.
- Receipts are hash-bound but not signed by this module; evaluator authentication is external.
- The module does not schedule multi-fidelity promotions or train candidates. It chooses among caller-supplied actions.

## Benchmark design

Compare:
- score-only / equal evaluation-count allocation;
- fixed round-robin and random feasible selection;
- global UCB;
- Symthaea DirectionBoard v0.1;
- Coral's faithful per-evaluation scheduler;
- gain-per-wall-time and gain-per-GPU-second scalarized controls;
- the v0.1 Pareto allocator;
- an exact small-instance multi-dimensional knapsack solver as a synthetic oracle upper bound.

Use families where resource preferences reverse:
- CPU-heavy vs GPU-heavy directions;
- fixed setup cost vs cheap repeated iterations;
- memory-capacity and GPU-OOM boundaries;
- low-fidelity screens with late rank reversal;
- tool/API quota constraints;
- energy/battery caps;
- high retry/failure costs;
- cost uncertainty and underestimation attacks;
- Pareto-incomparable capability/latency/energy outcomes;
- no-new-direction null and delayed breakthroughs.

Run both equal-evaluation-count and equal-total-resource experiments. Pin proposals, model, evaluator, prompts, agent count, total resource caps and payoff streams. Every failed, retried, rejected, timed-out or inconclusive attempt remains in the cost denominators.

## Qualification goals (proposed, not achieved)

- Zero hard-budget violations across adversarial and OOM controls.
- Exact plan/receipt/ledger reconciliation.
- Unknown constrained-resource values always fail closed.
- No double spending across concurrent actions (requires later reservation service).
- At least 100 paired landscape/cost instances across six or more independent families for confirmatory claims.
- Capability reported as a resource-conditioned Pareto frontier and worst-family performance, not a single weighted scalar.

## Claim ceiling

A passing synthetic test suite qualifies resource schemas, hard-budget semantics and Pareto bookkeeping. It does not establish improved architecture quality or scientific discovery. A real efficiency claim requires authenticated resource measurements, held-out capability, and repeated comparisons at equal total resource budgets.

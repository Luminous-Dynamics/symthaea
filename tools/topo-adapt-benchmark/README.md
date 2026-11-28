# topo-adapt-benchmark

A deterministic evaluator for the synthetic fixtures owned by TOPO-ADAPT-001
(issue #14123).

## Run

From the repository root:

    cargo test -p topo-adapt-benchmark
    cargo run -p topo-adapt-benchmark

The crate uses the workspace's existing serde, serde_json and blake3 dependencies.
The JSON fixture is embedded at build time from the fixtures directory.

The CLI prints every no-edit/candidate evaluation and a separate
`one_step_candidate_sweep_oracle_upper_bound` decision. The oracle maximizes
served flow, then minimizes declared edit plus one-step maintenance cost; it
has complete candidate visibility and is not a distributed or learned policy.

The CLI also prints the fixture BLAKE3 digest and an optional GITHUB_SHA. When
the source SHA is unavailable it emits null; such a run is not a source-bound
qualification receipt.

## Contract

- Directed integer capacities.
- Exactly one source per episode.
- Forward-time, sink-specific demand schedules.
- Candidate edges remain inactive until selected.
- Graph-edit budget enforced before evaluation.
- Node, edge and intervention references validated.
- Maximum flow calculated with an augmenting-path residual network.
- Minimum cut independently obtained by exhaustive partition enumeration.
- Flow/cut disagreement is a hard error.
- The exhaustive cut oracle is capped at 20 graph nodes in v0.1.

The evaluator reports flow, min-cut, accepted edit count, declared edit cost and
one-step edge-maintenance cost separately. Message bytes and controller updates
are declared fixture budgets but are not charged to a policy until a controller
actually emits message/update traces.

This is a benchmark evaluator, not a learned controller. It does not model
biological transport, bioelectricity, physical repair or industrial actuation.

## Stateful trajectory replay (P0 research API)

The `EpisodeRuntimeState` API is the sequential counterpart to the static
`evaluate_episode` evaluator. It maintains accepted candidate edges, demand
changes, event/action IDs and cumulative edit/maintenance ledgers across ticks.

A caller supplies each tick's event batch and at most one bounded
`PolicyAction`. The runtime:
- applies events to a persistent graph;
- rejects same-tick conflicting mutations to one resource;
- requires strictly increasing ticks and does not allow callers to skip a
  fixture demand/intervention time;
- requires each action to bind to the exact previewed pre-action state digest;
- applies cumulative edit and maximum-active-edge limits;
- treats repeated action/event IDs idempotently;
- computes max-flow and an exhaustive independent min-cut before and after the
  action;
- emits deterministic state and event digests and a per-tick receipt.

The P0 action surface is intentionally small: no-op, add a candidate edge,
defer and abstain. Remove/rewire, capacity changes, richer observation
requests, multi-agent authority and a learned controller are not implemented
in this slice.

Message bytes and controller updates are `None` (unmeasured), not zero. The
runtime must not claim to enforce those budgets until a controller adapter
produces actual message and update traces. Maintenance cost is charged once per
active edge per simulated tick in the current research accounting unit; this
is a synthetic accounting convention, not measured physical energy or money.

Thirteen tests are authored for repeated disruption, the snapshot-reset negative control, persistent runtime-created edges,
cumulative budgets, event-order invariance, conflict rejection, duplicate
action IDs and event IDs, stale pre-state digests, deterministic replay,
demand persistence, skipped-schedule rejection and context-bound state digests.

Run:
    cargo test -p topo-adapt-benchmark


## Exact robust viability and recoverability oracle (P0 reference lane)

The viability module adds two separate exact finite-state calculations:

- solve_robust_viability() computes a finite-horizon kernel with controller-first / disturbance-second quantifier order. All hidden states sharing an observation must accept one common action. It accounts for per-step holding cost, action cost, outcome-specific debit and a protected terminal reserve.
- solve_recoverability(deadline) computes whether a target capability envelope can be reached in at most the deadline, and reports minimum steps, minimum starting budget and a first-action witness. Unreachable remains None, not a fabricated finite recovery time.

The exact oracle is bounded to 64 states, 64 actions, 32 steps, 128 resource units and 16,384 outcome rows. These are small-instance reference limits, not scalability claims. The model is a conservative memoryless observation policy; longer-history belief-state control, stochastic viability, learned policies, physical models and cryptographic authority are outside this slice.

Run:

    cargo test -p topo-adapt-benchmark

Fifteen viability/recoverability tests are authored. The source has not been compiled or executed on this final PR head; CI and formatting results must be checked on that exact head before promotion. The module is a synthetic reference oracle, not a learned policy or industrial/physical qualification.

See docs/research/TOPO_ADAPT_VIABILITY_ORACLE_V0_1.md for the formal contract, benchmark tournament, research hypotheses, falsifiers and promotion gates.

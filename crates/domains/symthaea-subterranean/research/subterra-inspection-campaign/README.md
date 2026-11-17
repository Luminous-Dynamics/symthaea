# subterra-inspection-campaign

A dependency-free Rust reference crate for campaign-level selection over already-scored inspection options.

It is deliberately **not** a second single-action sensor planner. Candidate gross EVSI and likelihoods are produced upstream by the Subterra decision-value adapter and independently assessed models. This crate selects a bounded portfolio under declared constraints.

## Paradigm

Move beyond ranking the next inspection by entropy or one-step EVSI. Select a campaign across multiple assets while accounting for:

- one-time deployment/mobilization bundles shared by several inspections;
- one option per asset/decision, with mandatory critical decisions that cannot be skipped;
- separate energy, duration, data-volume and exposure budgets;
- per-action mission-interruption and robot-loss upper bounds aggregated with the union bound (no independence assumption);
- unknown-state-debt reduction as a separate reported quantity, not silently monetized;
- exact search with deterministic tie-breaking and an explicit node-limit/optimality disposition;
- digest-pinned common-cause and portfolio-value factorization assessments.

## Run

```sh
cargo run --manifest-path Cargo.toml --example synthetic_campaign --offline
cargo fmt --manifest-path Cargo.toml -- --check
cargo test --manifest-path Cargo.toml --all-targets --offline
cargo clippy --manifest-path Cargo.toml --all-targets --offline -- -D warnings
```

## What the solver proves

For a frozen finite candidate set of at most 16 decisions and 64 options, the branch-and-bound search either:
- returns a feasible selection with `OptimalityProven`; or
- returns a feasible incumbent with `NodeLimitReached`, explicitly **not** claiming optimality.

If the node limit is reached before a feasible incumbent exists, the solver returns an error instead of presenting an incomplete plan as feasible. Mandatory decisions must be covered. Shared deployment-bundle fixed resources are charged exactly once.

The objective is the sum of upstream per-decision gross EVSI estimates in a common declared loss unit, subject to independent hard constraints for each resource and risk dimension. The solver does not combine joules, seconds, bytes, exposure or probability into a guessed scalar price.

## Assurance boundary

An additive campaign-value estimate is only marked assurance-eligible when all of the following are present:
- common-cause dependency graph digest;
- fault-domain profile digest;
- common-cause assessment digest;
- portfolio-value/factorization assessment digest;
- inventory marked complete within its declared scope;
- factorization marked complete by that external assessment.

Otherwise the plan is marked `ResearchOnlyInsufficientEvidence`. The crate does not authenticate those digests, prove inventory completeness, or establish that the external factorization is correct. It never infers independence from different robot IDs, sensor IDs or files.

The union bound is conservative only relative to valid supplied probability upper bounds. It does not validate those bounds.

## Synthetic fixture

The executable fixture has three asset decisions and three candidate inspections. Under the same energy ceiling, gross-EVSI, EVSI/energy and EVSI/time greedy each choose one high-value option (19 units); exact campaign search selects two complementary inspections (20 units). The fixture uses synthetic-only digests and an unknown inventory disposition, so its output remains research-only. All values and resource costs are invented arithmetic fixtures, not calibrated inspection performance or field measurements.

## Scope limits

- This crate selects inspection options; it does not compute paths, route geometry, charging-station schedules or fleet execution. Symtropy owns those physical/world and fleet models.
- It does not generate sensor likelihoods, calibrate POD/PFA, assess structural/geotechnical/electrical condition, authenticate Mycelix or Sol-Atlas evidence, or grant work authority.
- Exact search is bounded and exponential in the worst case. Larger campaigns need decomposition or a separately qualified solver, not silent expansion of limits.
- Synthetic passing tests do not establish physical safety, regulatory compliance or mine-operational readiness.

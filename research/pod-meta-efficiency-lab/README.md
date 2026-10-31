# POD-Meta Efficiency Lab

A standalone, dependency-free Rust microbenchmark for the claim:

**Proof-obligation-guided operation selection should produce more independently verified proof progress per unit of computation than generic reasoning allocation.**

This is a deterministic synthetic experiment, not a production scheduler, real FLOP/energy measurement, or state-of-the-art claim.

## Run

~~~sh
cargo test --manifest-path research/pod-meta-efficiency-lab/Cargo.toml
cargo run --manifest-path research/pod-meta-efficiency-lab/Cargo.toml --bin pod-meta-efficiency
cargo fmt --manifest-path research/pod-meta-efficiency-lab/Cargo.toml -- --check
cargo clippy --manifest-path research/pod-meta-efficiency-lab/Cargo.toml --all-targets -- -D warnings
~~~

The runner prints 240 data rows: 8 task profiles × 5 resource budgets × 6 policy arms, plus a header. The seed is frozen at 1729.

## Policy arms

- fixed_depth: executes operations in a fixed catalogue order under the budget.
- generic_voc: ranks operations by generic predicted gain / abstract cost, without conditioning the ranking on the exact unresolved proof obligations.
- budget_matched_random: seed-reproducible random selection from the same operation catalogue under the same total budget.
- proof_obligation_guided: services mandatory verification first, reserves its cost, then selects operations by newly targetable unresolved obligations per abstract cost. An operation's asserted coverage never discharges an obligation by itself.
- unverified_confidence: diagnostic unsafe controller that trusts advertised claims without the independent verifier.
- oracle_minimum_cost: evaluator-informed minimum-cost set-cover upper bound. It is impossible for a planner that cannot inspect evaluator-only outcomes and is never a promotion candidate.

## Information boundary

The planner-visible OperationSpec contains operation ID, advertised obligation/claim mask, generic predicted gain, abstract work-unit cost, and mandatory verification mask.

The evaluator-only verified support mask is stored separately and is not read by the operation chooser. It becomes visible to the controller only after that operation is selected and the simulated independent verifier returns the result.

This provides API-level separation inside one process, not a secure benchmark boundary. A confirmatory benchmark must move hidden labels, seeds and evaluator into a separately permissioned job/service.

## Primary metrics (no composite score)

The runner reports verified obligations and fully verified completion, completion claims and false completion, false-support claims, missed mandatory verification, operations and abstract cost, redundant operation count and wasted-operation cost, cost to first sufficient proof, evaluator-only oracle minimum cost and cost regret, and correct abstention.

An operation that produces no newly verified support is not automatically wasteful: mandatory checks are classified separately and still incur cost. This prevents efficiency metrics from rewarding skipped safety checks.

## Limitations and next gates

- Eight fixed task profiles are structural fixtures, not a hidden or contamination-free leaderboard.
- Abstract work units are not measured FLOPs, GPU time, energy, latency, or monetary cost.
- The random baseline shares the same operation catalogue and total budget, but is budget-matched rather than realized-cost-matched. It is not a learned distribution over industry tool usage.
- Generic VOC uses deterministic supplied predictions, not a calibrated learned predictor.
- The oracle is computed in the same process for diagnostics; it is not a deployable policy.
- The benchmark does not yet model natural-language generation, multiple models, tool latency distributions, real evidence sources or production adapters.

Before any superiority claim, qualify the exact-head Rust crate, add generator/operator-family held-out splits and protected oracle execution, then run matched checkpoints with actual measured all-in resource budgets. Use official external evaluators independently for TEMPO, Vera-Life-Eval, LitTraceQA, GRACE, FaithCoT-Bench and ParaRecover.

The promotion objective is a Pareto frontier over proof correctness, mandatory safety verification, abstention quality and measured resource cost—not a single intelligence number.

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

The runner prints 280 data rows: 8 task profiles × 5 resource budgets × 7 policy arms, plus a header. Each row binds seed 1729 and reports the native POD-Meta scheduler's candidate rejections. Each row binds seed 1729. The full external benchmark mapping and comparability rules are recorded in EXTERNAL_BENCHMARK_MATRIX_V1.md.

## Policy arms

- fixed_order: executes operations in a fixed catalogue order under the budget (a fixed action sequence, not a recurrent-token-depth experiment).
- generic_voc: ranks operations by generic predicted gain / abstract cost, without conditioning the ranking on the exact unresolved proof obligations.
- budget_matched_random: seed-reproducible random selection from the same operation catalogue under the same total budget.
- pod_meta_router: calls the existing `pod-meta-lab::choose_next` implementation with typed obligations, operation kinds, scheduling classes, progress keys, budget and mandatory reserve. The adapter credits only the obligations returned by the native selector; evaluator support masks are intersected with that selection.
- mask_coverage_greedy: a small benchmark-local obligation-mask heuristic retained as a diagnostic ablation, not the production POD-Meta selector.
- unverified_confidence: diagnostic unsafe controller that trusts advertised claims without the independent verifier.
- oracle_minimum_cost: evaluator-informed minimum-cost set-cover upper bound. It is impossible for a planner that cannot inspect evaluator-only outcomes and is never a promotion candidate.

## Information boundary

The planner-visible OperationSpec contains operation ID, advertised obligation/claim mask, generic predicted gain, abstract work-unit cost, and mandatory verification mask. The native POD-Meta arm maps these into its existing typed `Obligation`, `CandidateOperation`, `OperationKind`, `SchedulingClass`, and `SelectionContext` contracts.

The evaluator-only verified support mask is stored separately and is not read by the operation chooser. It becomes visible to the controller only after that operation is selected and the simulated independent verifier returns the result.

This provides API-level separation inside one process, not a secure benchmark boundary. A confirmatory benchmark must move hidden labels, seeds and evaluator into a separately permissioned job/service.

## Primary metrics (no composite score)

The runner reports verified obligations and fully verified completion, completion claims and false completion, false-support claims, native scheduler candidate rejections, missed mandatory verification, operations and abstract cost, rejected-evidence candidate count/cost, redundant operation count and wasted-operation cost, cost to first sufficient proof, evaluator-only oracle minimum cost and cost regret, and correct abstention.

A failed operation that targeted an open obligation is reported as a rejected evidence candidate, not automatically as wasted computation: its verifier rejection can be decision-relevant. An operation that could not target any remaining obligation is classified as redundant/irrelevant. Mandatory checks are separately classified and still incur cost. This prevents efficiency metrics from rewarding skipped safety checks.

## Limitations and next gates

- Eight fixed task profiles are structural fixtures, not a hidden or contamination-free leaderboard.
- Abstract work units are not measured FLOPs, GPU time, energy, latency, or monetary cost.
- The random baseline shares the same operation catalogue and total budget, but is budget-matched rather than realized-cost-matched. It is not a learned distribution over industry tool usage.
- Generic VOC uses deterministic supplied predictions, not a calibrated learned predictor.
- The oracle is computed in the same process for diagnostics; it is not a deployable policy. The POD-Meta adapter is a benchmark mapping into the existing scheduler types, not yet a production cognitive-operation producer or integration.
- The benchmark does not yet model natural-language generation, multiple models, tool latency distributions, real evidence sources or production adapters.

Before any superiority claim, qualify the exact-head Rust crate, add generator/operator-family held-out splits and protected oracle execution, then run matched checkpoints with actual measured all-in resource budgets. Use official external evaluators independently for TEMPO, Vera-Life-Eval, LitTraceQA, GRACE, FaithCoT-Bench and ParaRecover.

The promotion objective is a Pareto frontier over proof correctness, mandatory safety verification, abstention quality and measured resource cost—not a single intelligence number.

# VIS-DVPC-006: Post-Commit Paired-Trace Scorer v1

**Status:** Draft scoring mechanism and synthetic fixtures.  
**Issue:** [#8505](https://github.com/Luminous-Dynamics/symthaea/issues/8505)  
**Parent trace schema:** [VIS-DVPC-005 PR #8492](https://github.com/Luminous-Dynamics/symthaea/pull/8492)  
**Research framing:** [DEEP, arXiv:2609.35910](https://arxiv.org/abs/2609.35910)

## Goal

Score immutable action commitments against paired cheap/full outcome records after policy execution. This is the missing evaluator layer between a trace-admission schema and claims that a perception-compute allocator improves decisions under a budget.

DEEP defines decision value through the change in downstream loss when escalating from cheap to expensive perception. Because that value may be negative, uniform full fidelity is a baseline, not an automatic upper bound. DEEP also charges the allocator for its own computation and evaluates under latency and energy budgets. This scorer adopts those principles but does not claim to reproduce DEEP.

## Inputs

The command requires a single split and three matching TSV files:

```sh
rustc --edition=2024 --test tools/vision-dvpc-trace-scorer.rs -o /tmp/vis-dvpc-006-tests
/tmp/vis-dvpc-006-tests
rustc --edition=2024 -O tools/vision-dvpc-trace-scorer.rs -o /tmp/vis-dvpc-006
/tmp/vis-dvpc-006 test OBSERVATIONS.tsv EVALUATIONS.tsv ACTIONS.tsv
```

Allowed scored splits are `validation`, `test`, and `private_test`; a single invocation must contain only its selected split. The source/split manifests still bind the dataset-wide grouping policy. Training split scoring is not admitted by the CLI.

### Observation and evaluation files

The observation and evaluation schemas are exactly those frozen in [VIS-DVPC-005](VIS_DVPC_005_PAIRED_TRACE_CONTRACT_V1.md). Both files must have the same dataset revision, source/split manifests, utility spec, and license metadata. Trace IDs and sample digests must join one-to-one.

### Action commitment file

The action schema is exact and version-bound by the same dataset metadata:

```text
ACTION_HEADER\ttrace_id\tsample_sha256\tcomparison_id\tpolicy_id\tchoice\tcommitment_index\tlatency_budget_us\tenergy_budget_uj\tdeclared_extra_latency_us\tdeclared_extra_energy_uj\tallocator_latency_us\tallocator_energy_uj\tcommitted
```

Each policy must have exactly one row for every trace, unique contiguous commitment indices from zero, consistent budget ceilings within its run, and the same latency/energy ceilings as the other policies in its `comparison_id`. Every row must be marked `committed=true`. Choices are `cheap` or `full`. Each action row repeats the observation's digest and declared extra-action resource estimates; mismatches fail closed.

`allocator_latency_us` and `allocator_energy_uj` are measured realized allocator overhead per decision, not model estimates. They are charged on *every* decision. When the committed choice is `full`, the evaluator also charges the paired record's actual incremental latency and energy. A `cheap` decision incurs no extra-action cost, but still incurs allocator overhead.

The action file must be frozen before the evaluator outcome file is made available to the policy. This standalone scorer runs after commitment; it does not create the commitment receipt or enforce a secure OS-level process boundary.

## Scores

Each comparison/policy result includes:

- Mean chosen utility and group-balanced utility, averaging within each source group first and then giving each group equal weight.
- Beneficial, harmful and neutral escalations based on the sign of `full_utility - cheap_utility`; missed beneficial opportunities where the policy chose cheap although the full outcome scores higher.
- Cheap-path and chosen-path safety violations as separate counts. Utility gains do not cancel safety regressions.
- CVaR95 incremental decision loss. A full choice contributes `cheap_utility - full_utility`; cheap contributes zero. The measure is the mean of the worst ceil(5% of per-trace losses), with positive values indicating harm.
- Allocator overhead, total incremental latency/energy, fixed run budgets, reservation overruns, and explicit latency/energy budget-violation booleans.
- Paired group-balanced utility delta, CVaR95 delta and safety-violation delta versus `policy_id=all_cheap` in the same `comparison_id`.
- Utility gap to a post-commit hindsight oracle when the policy stays within both resource budgets.

All rows remain visible, including runs that exceed budget. A run that exceeds latency or energy budget is not a valid resource-constrained win. Reservation overruns are separately counted; the scorer does not clip their actual realized costs.

## Deliberate limits in v1

- The scorer computes an evaluator-only **two-resource Pareto-frontier oracle** after committed actions are loaded. It maximizes group-balanced utility under simultaneously binding actual incremental latency and energy ceilings. It may emit `NOT_COMPUTED_FRONTIER_CAP` if the non-dominated frontier exceeds 50,000 states; it never substitutes an approximation and calls it exact.
- The oracle is a non-deployable upper reference with zero allocator overhead and perfect foresight. It optimizes utility only; it does not optimize safety violations or CVaR, which remain separately reported. Utility gaps are `NA` for runs that violate a resource budget.
- No task-specific custom utility learning, confidence intervals, significance tests, or dataset adapter is implemented.
- The scorer checks SHA-256 format and sample digest linkage; the run workflow records actual file SHA-256 values. It does not verify that the source manifest digest matches downloaded media bytes or prove the licensing assertion.
- The scorer reports realized outcomes and should be run once a frozen policy trace is committed. It is not an online model selector.
- The synthetic test fixture has three traces across two groups, deliberately with unequal group sizes, so group-balanced aggregation differs from the raw record mean. It exercises edge behavior and has no statistical power.

## Synthetic fixture design

The three test traces are deliberately discordant and group-imbalanced:
- Trace 0: full fidelity improves utility from 0.40 to 0.55 with no safety regression.
- Trace 1: full fidelity reduces utility from 0.50 to 0.42 and adds one safety violation.
- Trace 2: full fidelity improves utility from 0.10 to 0.20; it shares a source group with trace 0.

Three committed arms are supplied: `all_cheap`, `all_full`, and `value_router` (full on trace 0; cheap on traces 1 and 2). Allocator overhead is non-zero for the router. The shared 25,000 µs latency ceiling makes all-full exceed budget at 35,000 µs, while the router remains within the ceiling at 15,000 µs. The hindsight oracle picks the two beneficial escalations at a combined 20,000 µs and 800 µJ, yielding group-balanced utility 0.437500 on this synthetic fixture. These are fixture arithmetic expectations, not measurements from real traces.

## Benchmark path to external comparison

1. Qualify this scorer with deterministic synthetic tests and schema checks.
2. Add a rights-cleared local adapter that generates cheap/full perception outputs and task outcomes from a frozen trace set. Do not commit raw media.
3. Freeze validation policy selection and reserve a separate test split for final paired comparisons.
4. Compare all-cheap, all-full, matched-budget random, simple score routers, conservative LCB, and candidate policies on the same trace/group set.
5. Charge scheduler and allocator overhead, measure actual latency and energy, publish invalid/over-budget rows, and report group-level paired contrasts by scenario.
6. Treat the current two-resource hindsight oracle as a diagnostic upper reference, then compare against matched-budget random and deployable learned/lightweight routers under the same utility profile. Confirm oracle optimality on small problems before scaling.
7. Evaluate open-loop and closed-loop planner outcomes separately. nuPlan documents both modes, which answer different questions: https://www.nuplan.org/

nuScenes and nuPlan are available under non-commercial-use terms on their public pages; a dataset adapter for a commercial product must verify and obtain the appropriate permission separately. nuScenes terms: https://www.nuscenes.org/data-format.

## Claim boundary

The scorer establishes only post-commit score mechanics. It does not establish a real-data result, detector quality, downstream safety, statistical significance, a deployed allocator, or state of the art.

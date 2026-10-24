# VIS-DVPC-016: Safety-gated budget-sweep Pareto analysis v1

**Status:** Draft reference analyzer.  
**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8595  
**Paired scorer:** https://github.com/Luminous-Dynamics/symthaea/pull/8506  
**Evidence lock:** https://github.com/Luminous-Dynamics/symthaea/pull/8569  
**Research anchor:** DEEP, https://arxiv.org/abs/2609.35910

## Why analyze a frontier rather than one budget?

A single budget point can hide whether a method is generally useful or only wins at a selected operating point. The cost of the allocator itself can change method rankings, so the benchmark must include all-in measured costs and compare the same policy family over multiple predeclared joint latency/energy budgets.

Recent deployment-oriented autonomous-driving research also explores software/hardware co-optimization with inference latency, energy and closed-loop performance, reporting large tradeoffs in its own tasks. Such results motivate multi-objective evaluation but are **not directly comparable** to DVPC unless dataset, hardware, task, budgets and metrics are matched.

This analyzer takes a frozen grid of precomputed policy summaries and classifies a safety-gated frontier. It is not the model, scorer, risk certifier or a benchmark result.

## CLI

Build:

    rustc --edition=2024 -O tools/vision-dvpc-budget-frontier.rs -o /tmp/vis-dvpc-016

Run:

    /tmp/vis-dvpc-016 SWEEP.tsv EXPECTED_TRACE_BUNDLE_LOCK_SHA256 EXPECTED_SWEEP_FILE_SHA256

The first expected hash must come from the separately frozen evidence receipt. The second must be the SHA-256 of the **exact sweep TSV bytes** committed for preregistration. The tool does not generate a replacement hash or accept a changed constraint table as the original experiment.

This is an analyzer over already-produced summaries. It does not recompute upstream risk certificates, verify sensor-file bytes, or prove who created the upstream receipt.

## Required experiment metadata

The exact META key set is required:
- format: exactly VIS-DVPC-BUDGET-SWEEP-001-v1
- experiment ID, dataset ID and immutable dataset revision
- held-out split (test or private_test)
- utility specification SHA-256
- paired evidence bundle-lock SHA-256
- exact paired trace/cluster set SHA-256, repeated in each row to prevent method/budget grid drift
- expected policy count and budget-point count
- fixed evaluated-trace count and independent-cluster count
- harm-rate upper-bound target, CVaR95 incremental-loss ceiling and safety-violation ceiling

Thresholds belong in the frozen sweep file; the sweep file digest must be preregistered independently before test outcomes are viewed. Editing a threshold after outcomes are seen creates a new exploratory analysis, not a replay.

## Grid rows

The exact header is:

    SWEEP_HEADER<TAB>budget_point_id<TAB>policy_id<TAB>latency_budget_us<TAB>energy_budget_uj<TAB>utility_delta_vs_all_cheap<TAB>actual_all_in_latency_us<TAB>actual_all_in_energy_uj<TAB>allocator_latency_us<TAB>allocator_energy_uj<TAB>latency_budget_violations<TAB>energy_budget_violations<TAB>cost_measurement_complete<TAB>safety_violations<TAB>cvar95_incremental_loss<TAB>pooled_harm_upper_bound<TAB>worst_group_harm_upper_bound<TAB>risk_certificate_state<TAB>paired_unit_set_sha256<TAB>evaluated_traces<TAB>independent_clusters

Each budget-point/policy pair appears once. Every budget point must contain the exact same policy set. Ceilings must be consistent across policies for one point and distinct across points. All rows must use the same frozen paired-unit-set digest, evaluated-trace count and independent-cluster count. The paired-unit digest must identify the exact trace/cluster manifest from the same frozen experiment; counts alone do not establish that identical traces were scored.

The policy family must include all-cheap and all-full plus predeclared deployable baselines. Include the same-budget random and lightweight confidence/entropy/value routers when they are available. Preserve infeasible/high-utility/unsafe/over-budget rows; do not remove them to improve the frontier.

## Feasibility gates

A row enters the feasible frontier only if every condition passes:

- measured cost is marked complete;
- both explicit budget-violation counts are zero and all-in actual latency/energy fit their separate ceilings;
- the reported safety-violation count is at or below the frozen ceiling;
- CVaR95 incremental loss does not exceed the frozen tail-loss ceiling;
- pooled and worst-group harm upper bounds do not exceed the frozen risk target;
- the upstream risk-certificate state is CERTIFIED.

The certificate field is only an input assertion tied to upstream receipt evidence. This analyzer does not independently verify the certificate. A CERTIFIED string cannot make an incomplete, shifted, unsafe or over-budget run feasible.

Rows that fail a gate remain in the output as INFEASIBLE, with every applicable failure reason.

## Pareto dominance

Among feasible rows, evaluate three independent objectives:

1. maximize downstream decision-utility delta versus all-cheap;
2. minimize actual all-in latency;
3. minimize actual all-in energy.

Point A dominates B only if A is no worse on all three dimensions and strictly better on at least one. When all three values are exactly equal, lexicographic budget-point/policy ordering deterministically retains one representative and marks the other dominated. The tool never invents a single weighted score, lets extra utility cancel a safety violation, or trades latency slack for an energy violation.

Output includes feasible-frontier, feasible-dominated and infeasible rows; the first deterministic dominator; reasons for failed gates; frontier/infeasible counts; evidence/sweep hashes; and explicit no-claims.

## Interpretation and non-claims

- FEASIBLE_FRONTIER indicates a non-dominated summary point among the provided grid, not a universal optimum.
- FEASIBLE_DOMINATED means another feasible point is at least as good in utility and no worse in each measured resource, with at least one strict improvement (or is the deterministic representative of an objective tie).
- INFEASIBLE means one or more preregistered safety, tail, cost-measurement or hard-budget constraints failed.
- The frontier is only as reliable as its input summaries, their evaluator, risk certificates, hardware meters, and evidence custodian.
- Do not claim SOTA from a small or selected grid. A credible comparison uses paired held-out traces, independent-cluster inference, exact source/data/utility/lock identities, multiple predeclared budget points, published baselines and cross-domain replication.

No third-party data is included. Synthetic tests demonstrate analyzer mechanics only.

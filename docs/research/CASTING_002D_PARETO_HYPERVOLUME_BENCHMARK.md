# CASTING-002D — Constrained Pareto Archive and Hypervolume Benchmark

**Status:** Test-only metric implementation drafted; Rust execution pending  
**Objective directions:** Minimize both synthetic objective coordinates  
**Claim ceiling:** Pareto/hypervolume software semantics; no physical casting result

## Why change the optimization target?

A single weighted score can conceal an unacceptable engineering trade-off. For example, lower energy should not numerically compensate for a defect/quality limit that must be met as a hard requirement. CASTING-002D therefore keeps two concepts separate:

1. **Feasibility:** independent hard requirements decide whether a trial may enter the feasible archive.
2. **Objectives:** among feasible trials, preserve the nondominated front of a synthetic quality/defect proxy and an energy proxy. A chosen production trade-off requires an explicit external decision policy and does not emerge from the Pareto metric by itself.

These two objective values are synthetic units, not a casting defect probability, measured yield, kWh, energy per accepted part, or cost.

## Implemented test-only utility

`physics::casting_multiobjective_benchmark` provides:

- a frozen two-objective minimization reference point;
- an outcome contract for feasible, infeasible and failed attempts;
- atomic validation of IDs, status/objective consistency, objective finiteness, reference-box domain and positive cost;
- deterministic Pareto-front insertion with dominated-point removal and duplicate-objective representative selection;
- exact two-dimensional hypervolume by slice sweep;
- a per-attempt hypervolume versus cumulative-cost trace;
- cost-to-hypervolume-target queries;
- full attempt/status/cost accounting, including failed and infeasible trials.

Every failed or infeasible attempt is retained and charged, but is not allowed onto the feasible Pareto front. Invalid trial inputs are rejected before archive counters, front, trace, or cumulative cost changes.

## Frozen known-answer example

For minimization with a fixed reference point `r = (10, 10)`, the nondominated points

`(2, 8), (4, 4), (7, 2)`

have exact hypervolume

`(4 - 2)(10 - 8) + (7 - 4)(10 - 4) + (10 - 7)(10 - 2) = 46`.

The 11 test-only authored cases in this module cover:

- empty-front hypervolume is zero;
- exact three-point hypervolume;
- dominated candidate removal while retaining its trial and cost;
- front and hypervolume independence from candidate insertion order;
- failed/infeasible trials excluded from the front but charged;
- cost-to-target includes prior failed and infeasible trial expenses;
- non-finite and reference-out-of-domain objectives rejected atomically;
- duplicate IDs, inconsistent payloads and zero cost rejected atomically;
- invalid reference points and invalid hypervolume targets fail closed;
- objective-duplicate representative selection is deterministic;
- cumulative-cost overflow is rejected without partial mutation.

Focused test command:

```sh
cargo test -p symthaea-core casting_multiobjective_benchmark
```

These are authored tests; they have not been compiled or executed in the current authoring environment because Rust/Cargo is unavailable there. Static source inspection is not a substitute for exact-head test evidence.

## Research comparator and next algorithmic layer

Cai et al. (2026), *Efficient Bayesian optimization for cheap and expensive multi-objective problems*, develops a hypervolume-based expected-improvement criterion for conflicting objectives evaluated at different costs. The noisy expected hypervolume improvement work (qNEHVI; NeurIPS 2021) addresses uncertainty in the Pareto frontier and noisy multi-objective evaluations. Evolution-guided BO for constrained multi-objective self-driving labs combines Pareto exploration with reduced sampling in infeasible regions. These are relevant algorithmic comparators; none makes this archive implementation itself a winning optimizer.

- Cai et al. (2026): https://www.sciencedirect.com/science/article/pii/S0307904X2500575X
- Daulton, Balandat & Bakshy (2021), qNEHVI: https://papers.nips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html
- Khan et al. (2024), constrained multi-objective BO: https://doi.org/10.1038/s41524-024-01274-x
- Essam et al. (2026), casting simulation / AI / digital twin review: https://link.springer.com/article/10.1007/s43939-026-00685-5

## Benchmark protocol

Before comparing methods, freeze:
- objective directions, units, normalization and hard constraints;
- reference point and target hypervolume;
- candidate design space and initial observations;
- per-query cost units by evaluation fidelity;
- seeds, final holdout, stopping rule and failed-run handling.

Compare independent single-objective optimization, frozen scalarization baselines, Pareto-aware acquisition, and eventually cheap/expensive multi-fidelity Pareto optimization on identical candidate domains and budgets.

Primary metrics:
- hypervolume versus cumulative cost;
- final hypervolume at a fixed budget;
- cost-to-target hypervolume;
- feasible/infeasible/failed attempt rates;
- objective-specific uncertainty and support coverage;
- run-to-run variability across paired seeds.

Do not compare hypervolume values obtained with different reference points or unaligned objective units. Do not claim physical benefit from synthetic objective functions. A later foundry pilot requires prospective physical measurements, independent metrology, explicit process population, appropriate constraints and a frozen control.

## Boundaries and nonclaims

This module is a test-only Pareto metric, not a multi-objective Bayesian optimizer, material or process model, recipe generator, machine controller, or qualification mechanism. It creates no alternate manufacturing registry or evidence authority. It proves no improvement in defect rate, yield, energy, cost, throughput, safety or industrial competitiveness.
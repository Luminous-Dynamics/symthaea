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

The 11 archive-specific test-only cases cover:

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

## CASTING-002E — Front quality and coverage indicators

Hypervolume is retained as the primary area-based indicator, but it must not be used alone: two approximations with similar hypervolume can differ in coverage and local point distribution. The test-only metric layer adds three complementary diagnostics for **two-objective minimization**:

- **Additive epsilon indicator** against a fixed reference front. Lower is better; negative values are permitted when every reference point is dominated by the approximation. It measures worst-case objective-space gap, not physical utility.
- **IGD+** against a fixed reference front. For each reference point it measures the nearest approximation point using only positive (worse-than-reference) objective deltas, then averages those distances. Lower is better and it is sensitive to missing reference-front regions.
- **Nearest-neighbour spacing**: sample standard deviation of each approximation point's nearest Euclidean-neighbour distance. Lower indicates more even spacing, not a better or more complete front by itself.

All three indicators require comparable objective directions and **the same frozen objective normalization** across methods. Epsilon and IGD+ require the same independent reference-front points. Do not derive normalization, reference points, or the reference front separately for each optimizer after observing results. When analytic ground truth is unavailable, do not fabricate a reference front to enable IGD+; report hypervolume and other reference-free diagnostics while labeling the limitation.

The module now has 17 authored test cases: 11 archive-specific cases plus 6 coverage-metric cases. The new metric tests include hand-calculated epsilon/IGD+/spacing examples, dominance behavior, input-order invariance, empty and too-small fronts, non-finite inputs, and arithmetic-overflow rejection. They are authored only; no Rust test execution is claimed here.

Research rationale:
- Yang, Katt & Kaski, *Multi-Objective Bayesian Optimization via Adaptive ε-Constraints Decomposition*, ICML 2026 (STAGE-BO): https://proceedings.mlr.press/v306/yang26af.html
- *Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science*, Digital Discovery (2026): https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00134c
- Daulton, Balandat & Bakshy, qNEHVI, NeurIPS 2021: https://papers.nips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html

Suggested focused validation remains:

```sh
cargo test -p symthaea-core casting_multiobjective_benchmark
```

CASTING-002E does not add an optimizer or prove SOTA performance. It adds metrics needed to compare a future coverage-seeking optimizer fairly.

## CASTING-002F — Analytical multi-objective reference problems

The test-only module now contains decision-space objective evaluators and deterministic sampled analytical fronts for canonical two-objective **ZDT1, ZDT2, ZDT3 and ZDT4** problems:

- **ZDT1:** continuous convex front, \(f_2 = 1 - \sqrt{f_1}\) on the Pareto set.
- **ZDT2:** continuous concave front, \(f_2 = 1 - f_1^2\).
- **ZDT3:** five disconnected front segments. The segment boundaries are frozen to published approximate values and sampled independently; no interpolation bridges gaps.
- **ZDT4:** same global front shape as ZDT1, but a multi-modal decision landscape, useful for detecting methods that find the easy front geometry but fail to reach the global Pareto set.

The decision-space evaluators implement the standard \(g(x)\) penalty and validate vector length, finite values and each problem's domain. Reference-front generation includes both ends of each segment. The deterministic sample convention is uniform in \(f_1\), not Euclidean arc length; therefore generated front resolution and convention must be frozen for IGD+/epsilon comparisons. These are finite samples of continuous fronts, not the mathematical continuum itself.

### New regression tests

The module now has 23 authored test cases. Six new ZDT tests cover hand-calculated objective values, a ZDT4 non-optimal tail penalty, deterministic and correctly sized fronts, attainable Pareto-set points, nondominance plus zero self-comparison epsilon/IGD+, and fail-closed decision-domain/resolution validation. The previous 17 archive/metric tests remain.

### Benchmark references and industry comparator boundary

- The ZDT formulas and disconnected ZDT3 front intervals follow the widely used pymoo benchmark definitions: https://www.pymoo.org/problems/multi/zdt.html
- COCO's public BBOB-biobj archive offers 55 standardized bi-objective functions and should be a later external validation suite, not silently conflated with these four fixtures: https://coco-platform.org/data-archive/index.html
- BoTorch provides qLogNEHVI, qLogEHVI and qLogNParEGO as strong reference implementations for multi-objective Bayesian optimization: https://botorch.org/docs/multi_objective
- STAGE-BO (ICML 2026) explicitly targets under-covered Pareto regions through adaptive epsilon constraints: https://proceedings.mlr.press/v306/yang26af.html
- TAMO (ICLR 2026) explores amortized in-context proposal generation; proposal latency must be reported separately from pretraining cost and amortized query count: https://proceedings.iclr.cc/paper_files/paper/2026/hash/901b378426e1816eabb776a786c02f41-Abstract-Conference.html

Suggested exact validation command:

    cargo test -p symthaea-core casting_multiobjective_benchmark

The source is still test-only and these test cases have not been compiled or executed in this authoring environment. No method has yet been benchmarked across these fronts, and no physical casting or commercial equivalence claim is supported.

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

The test-only module now contains decision-space objective evaluators and deterministic sampled analytical fronts for canonical two-objective **ZDT1, ZDT2, ZDT3 and ZDT4** problems. Evaluators enforce the standard 30 decision variables for ZDT1/2/3 and 10 for ZDT4; ZDT4 tail variables use the canonical [-10, 10] domain, not [-5, 5].

- **ZDT1:** continuous convex front, \(f_2 = 1 - \sqrt{f_1}\) on the Pareto set.
- **ZDT2:** continuous concave front, \(f_2 = 1 - f_1^2\).
- **ZDT3:** five disconnected front segments. The segment boundaries are frozen to published approximate values and sampled independently; no interpolation bridges gaps. The later four intervals have open lower boundaries, so the reference generator applies a fixed 1e-8 inward offset to avoid including rounded boundary points that are mathematically dominated.
- **ZDT4:** same global front shape as ZDT1, but a multi-modal decision landscape, useful for detecting methods that find the easy front geometry but fail to reach the global Pareto set.

The decision-space evaluators implement the standard \(g(x)\) penalty and validate canonical vector dimension, finite values and each problem's domain. Reference-front generation includes both ends of each segment. The deterministic sample convention is uniform in \(f_1\), not Euclidean arc length; therefore generated front resolution and convention must be frozen for IGD+/epsilon comparisons. These are finite samples of continuous fronts, not the mathematical continuum itself.

### New regression tests

The module now has 24 authored test cases. Seven new ZDT tests cover hand-calculated objective values, a ZDT4 non-optimal tail penalty, deterministic and correctly sized fronts, attainable Pareto-set points, nondominance plus zero self-comparison epsilon/IGD+, fail-closed decision-domain/resolution validation, and end-to-end known hypervolume values for sampled ZDT1/ZDT2 fronts (3.353553390593274 and 3.125 with reference point (2, 2)). The previous 17 archive/metric tests remain.

### Benchmark references and industry comparator boundary

- The ZDT formulas and disconnected ZDT3 front intervals follow the widely used pymoo benchmark definitions: https://www.pymoo.org/problems/multi/zdt.html
- COCO's public BBOB-biobj archive offers 55 standardized bi-objective functions and should be a later external validation suite, not silently conflated with these four fixtures: https://coco-platform.org/data-archive/index.html
- BoTorch provides qLogNEHVI, qLogEHVI and qLogNParEGO as strong reference implementations for multi-objective Bayesian optimization: https://botorch.org/docs/multi_objective
- STAGE-BO (ICML 2026) explicitly targets under-covered Pareto regions through adaptive epsilon constraints: https://proceedings.mlr.press/v306/yang26af.html
- TAMO (ICLR 2026) explores amortized in-context proposal generation; proposal latency must be reported separately from pretraining cost and amortized query count: https://proceedings.iclr.cc/paper_files/paper/2026/hash/901b378426e1816eabb776a786c02f41-Abstract-Conference.html

Suggested exact validation command:

    cargo test -p symthaea-core casting_multiobjective_benchmark

The source is still test-only and these test cases have not been compiled or executed in this authoring environment. No method has yet been benchmarked across these fronts, and no physical casting or commercial equivalence claim is supported.


## CASTING-002G — Largest-gap epsilon-constraint proposal baseline

A new test-only proposal policy provides an auditable bridge from a Pareto archive to a candidate selected under an adaptive epsilon constraint. It is deliberately simpler than STAGE-BO and is not described as Bayesian optimization.

The policy:
1. Sorts the current Pareto front deterministically.
2. Computes adjacent Euclidean gaps using a frozen, positive per-objective scale.
3. Selects the largest normalized gap; ties resolve to the earliest gap in canonical objective order.
4. Sets epsilon to the midpoint of the selected gap endpoints' second-objective values.
5. Accepts only records containing candidate ID, predicted objective means and predictive standard deviations—never hidden evaluator truth.
6. Requires the upper confidence bound \(\mu_2 + \beta\sigma_2\) to satisfy epsilon, then minimizes \(\mu_1-\beta\sigma_1\), with stable candidate-ID tie-breaking.
7. Returns a receipt containing the gap endpoint trial IDs, epsilon, normalized gap, beta, selected ID and predictive bounds.

This is a deterministic surrogate-output selector. The confidence multiplier is only a score parameter; the code does not calibrate predictive intervals, so satisfying an upper confidence bound is not a calibrated probability guarantee. It does **not** fit a surrogate, calculate constrained expected improvement, optimize hypervolume, reproduce STAGE-BO, or guarantee Pareto-front improvement. It must be compared against straightforward epsilon grids, random/grid/LHS baselines, GP-LCB and stronger external methods under identical objective scales and evaluation budgets before any quality claim.

### Added tests and validation limits

The module now has 30 authored test cases (24 existing archive/metric/ZDT tests plus 6 proposal-policy tests). New cases cover known largest-gap selection, uncertainty-aware epsilon filtering, input-order invariance, deterministic ID tie-breaking, invalid data, duplicate candidate IDs, dominated/duplicate front inputs, and the no-candidate-feasible path.

Research context:
- STAGE-BO (ICML 2026) adapts epsilon constraints to target uncovered surrogate-front regions and uses constrained expected improvement: https://proceedings.mlr.press/v306/yang26af.html
- BoTorch qLogNEHVI/qLogEHVI/qLogNParEGO are stronger reference implementations, not claimed reproduced here: https://botorch.org/docs/multi_objective
- COCO BBOB-biobj supplies a standardized scalable bi-objective suite for later external validation: https://numbbo.github.io/coco/testsuites/bbob-biobj

Exact-head Rust execution remains required:

    cargo test -p symthaea-core casting_multiobjective_benchmark

No test execution is claimed from this authoring environment.


## CASTING-002H — Split-conformal uncertainty calibration

The gap policy's \(\beta \sigma\) scores are not probability guarantees unless predictive uncertainty is calibrated. The new test-only split-conformal utility estimates one non-negative multiplier per scalar objective from normalized absolute residuals:

\[
s_i=\frac{|y_i-\mu_i|}{\sigma_i},\qquad
q=s_{(\lceil(n+1)(1-\alpha)\rceil)}.
\]

It fails closed if the exact finite-sample rank exceeds the available calibration sample size rather than silently clipping the quantile. A separate holdout assessment reports empirical interval coverage, mean absolute standardized residual and Gaussian negative log-likelihood; it does not refit the multiplier. Fit a separate multiplier for each objective using a calibration split that was not used to fit the predictive model. When those multipliers are used by a sequential optimizer, measure coverage on selected proposals as a separate post-selection diagnostic; ordinary exchangeable marginal coverage does not imply selection-conditional coverage.

**Guarantee boundary:** under exchangeability between calibration and future observations, split conformal gives finite-sample *marginal* coverage at least \(1-\alpha\) for the constructed intervals. It does not guarantee conditional coverage for every design region or remain valid under arbitrary process drift. It does not establish that the base surrogate is physically accurate. Manufacturing distribution shift requires explicit drift monitoring, recalibration, or methods whose assumptions address the shift.

Research references:
- Deshpande, Marx & Kuleshov (AISTATS 2024), online calibration and conformal prediction in Bayesian optimization: https://proceedings.mlr.press/v238/deshpande24a.html
- Oliveira et al. (JMLR 2024), split conformal and non-exchangeable data: https://jmlr.org/papers/v25/23-1553.html
- Marques & Graziadei (PMLR 2026), skew-adaptive conformal prediction: https://proceedings.mlr.press/v329/c-marques-f-26a.html

Five new authored tests cover exact order-statistic rank, insufficient-sample refusal, bad alpha, invalid/overflowing observations, independent holdout coverage/NLL, and the empty/overflow holdout path. Total: 35 authored test cases in this test-only module.

Focused command:

    cargo test -p symthaea-core casting_multiobjective_benchmark

These tests have not been compiled or run in this authoring environment; exact-head execution is still mandatory.

## CASTING-002I — Weighted conformal under covariate shift

The new test-only weighted conformal fitter augments exchangeable split conformal for an explicitly stated covariate-shift setting. Each calibration score is weighted by a supplied non-negative finite importance weight proportional to target/source covariate-density ratio. The test candidate supplies its own strictly positive finite importance weight.

For normalized absolute-residual scores s_i = |y_i - mu_i| / sigma_i, finite calibration scores must accumulate weight at least

    (1 - alpha) * (sum calibration weights + test-point weight).

The test point contributes its own weight at positive infinity in the weighted conformal distribution. A high target/source weight may leave insufficient finite calibration mass. The implementation explicitly errors in that case; it does not discard test-point mass, clip the quantile, or return a falsely finite interval.

Importance weights are normalized by a common maximum before sums and weighted effective sample size (ESS) are computed; this preserves weighted quantiles and ESS while reducing overflow risk. The receipt reports target test-point weight share, weighted ESS and the finite quantile's weight fraction for audit.

### Statistical limits

The coverage interpretation depends on correctly specified importance ratios and a valid covariate-shift assumption. This does not repair label/conditional drift, unknown or badly estimated density ratios, arbitrary adaptive selection shift, or physically inaccurate surrogate predictions. We do not estimate density ratios here. Post-selection coverage must be evaluated separately. Low weighted ESS or a failed finite quantile is a refusal signal, not a reason to silently relax alpha.

Research:
- Jonkers et al., Conformal Predictive Systems Under Covariate Shift, COPA 2024: https://proceedings.mlr.press/v230/jonkers24a.html
- Conformal prediction with local weights: randomization enables robust guarantees, JRSS B (2025): https://academic.oup.com/jrsssb/article/87/2/549/7889096
- Pournaderi, Sharp training-conditional coverage bounds under covariate shift (27 September 2026): https://arxiv.org/abs/2609.33456

Six new tests cover a hand-calculated quantile including test-point mass, common weight-scale invariance, refusal when finite quantile mass is insufficient, invalid/zero-total weights, underflowing ratios, invalid alpha, and an integration from weighted objective-1 calibration into the epsilon proposal policy. Total: 41 authored test cases in this test-only module.

Focused command:

    cargo test -p symthaea-core casting_multiobjective_benchmark

These tests have not been compiled or run in the current authoring environment. Exact-head execution is still mandatory.

## CASTING-002J — Post-selection and regional coverage audit

A frozen conformal multiplier is now diagnosable on a separate held-out dataset with optimizer-selection and region labels. The audit reports overall, selected-only, unselected-only and per-region empirical coverage, the selected-minus-unselected gap, and the worst observed region.

The known-answer fixture intentionally shows why an aggregate number is inadequate: with multiplier 2 and target coverage 80%, overall coverage is 12/14 (about 85.7%) while optimizer-selected coverage is only 2/4 (50%); unselected coverage is 100%. The audit therefore detects a significant selected-subset reliability warning even though the aggregate exceeds the nominal target. It also reports the region with the lowest observed coverage. These are synthetic held-out observations, not an empirical manufacturing finding.

The implementation does not refit the multiplier, treats each region's minimum count as a frozen precondition, rejects empty selected/unselected strata, and deterministically orders regional summaries. Small groups fail closed rather than producing unstable coverage reports.

Research motivation:
- *Conformal prediction with local weights: randomization enables robust guarantees*, JRSS B (2025): https://academic.oup.com/jrsssb/article/87/2/549/7889096
- Jonkers et al., *Conformal Predictive Systems Under Covariate Shift*, COPA 2024: https://proceedings.mlr.press/v230/jonkers24a.html
- Oliveira et al., *Split Conformal Prediction and Non-Exchangeable Data*, JMLR 2024: https://jmlr.org/papers/v25/23-1553.html

Four new tests cover the masked aggregate-undercoverage example, input-order invariance, insufficient group/missing selection strata, and invalid parameters/scores. Total: 45 authored test cases in this test-only module.

Focused command:

    cargo test -p symthaea-core casting_multiobjective_benchmark

This is an offline diagnostic, not distribution-free conditional coverage, a calibrated drift alarm, safety certification or physical casting validation. Rust tests have not been executed in this environment.


## CASTING-002K — Monte Carlo stress test for covariate and optimizer-selection shift

The prior weighted-calibration implementation accepts supplied importance ratios; the regional audit diagnoses empirical coverage after selection. This experiment connects the two under a known data-generating process, specifically to prevent the false inference that target-marginal calibration automatically makes optimizer-selected candidates reliable.

### Fixed synthetic data-generating process

- Source covariates are uniform on [0, 1].
- Target covariates follow Beta(2, 1), generated by sqrt(U), with exact density ratio w(x) = 2x.
- Conditional standardized residual scores follow an exponential law with scale 0.2 + 2x. The conditional score law is held fixed between source and target; only the covariate distribution changes.
- The optimizer-selection stress slice is predeclared as x >= 0.75.
- The harness uses 128 source calibration observations, nominal 80% coverage, 8,000 replicates and an explicit deterministic seed. Its small local PRNG avoids adding a dependency; it is test fixture machinery only.

### Three methods and required measurements

1. Unweighted split conformal on source calibration.
2. Weighted split conformal using the exact target ratio 2x and each target test point's weight.
3. Selection-conditioned weighted conformal using 2x / (1 - 0.75^2) when x >= 0.75 and zero otherwise, with the matching test-point ratio.

The test reports target-marginal coverage, selected-subset coverage for all three methods, average conformal multiplier as an interval-width proxy, average weighted effective sample size (ESS), and any refusal due to insufficient finite weighted calibration mass. Refusals are counted and asserted to be zero for this overlap-sufficient fixture; they are never removed silently from a coverage denominator. Frozen, broad Monte Carlo envelopes guard against changes that break the intended shift/selection behavior without pretending to prove a population guarantee.

The benchmark is designed to falsify two distinct claims:

- Exact-ratio weighting should improve marginal target coverage versus unweighted calibration under covariate shift.
- Target-marginal weighting alone can still under-cover the selected high-x subset; estimating the selected target distribution via its own ratio should improve selected coverage, at the cost of lower ESS and potentially wider intervals.

The selected ratio is f_target(x | x >= c) / f_source(x) = 2x / (1 - c^2) on the selected support and zero outside. This is valid here because the synthetic selection rule is a known covariate-only threshold and conditional score behavior is unchanged. It is not a recipe for unknown, adaptive, label-dependent or arbitrary concept shift.

Focused reproduction:

    cargo test -p symthaea-core casting_multiobjective_benchmark -- --nocapture

The output is a synthetic regression benchmark, not a real casting data result, a physical-simulation validation or a universal conditional-coverage guarantee. Later physical evaluation must use manufacturing data, frozen train/calibration/test splits, cost-normalized multi-objective comparisons, and separate audits for selection and region. Low ESS, failed weighted finite-mass support, or uncharacterized density ratios should trigger abstention or new calibration data rather than relaxed thresholds.

Research context:
- Jonkers et al., Conformal Predictive Systems Under Covariate Shift, COPA 2024: https://proceedings.mlr.press/v230/jonkers24a.html
- Conformal prediction with local weights: randomization enables robust guarantees, JRSS B 2025: https://academic.oup.com/jrsssb/article/87/2/549/7889096
- Bhattacharyya & Barber, Group-weighted conformal prediction, Electronic Journal of Statistics 2026: https://doi.org/10.1214/26-EJS2506
- Pournaderi, Sharp training-conditional coverage for conformal prediction under covariate shift (27 September 2026): https://arxiv.org/abs/2609.33456

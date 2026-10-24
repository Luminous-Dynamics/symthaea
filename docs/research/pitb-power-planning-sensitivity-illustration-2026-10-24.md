# PITB power-planning sensitivity illustration

**Date:** 2026-10-24  
**Status:** exploratory independent Monte Carlo sensitivity calculation; not a Rust binary run, CI result, real-benchmark power estimate, or model-evaluation result.

## Question

Does the protocol's minimum of 20 independent components by itself establish adequate power for the pre-registered gate? No. Under one explicitly illustrative family-clustered model, 20 components provide only around 44% probability of passing the gate when the true marginal gain is 15 percentage points.

This is a design warning, not a claim about PITB's eventual data. The real eligible component-size distribution cannot be computed until the independently reviewed cross-family relationship/quarantine inventory is complete.

## Model and gate

The exploratory calculation independently implemented the same intended probability model as `plan_pitb_power.rs`:

- A fixed vector of family/component sizes, with one control and treatment outcome generated for each target.
- Control probability drawn from a beta distribution with mean baseline 0.35 and family ICC 0.05.
- A family-level treatment-effect deviation drawn from a Normal distribution with probability-scale SD 0.05 unless a sensitivity row overrides it; treatment and control success probabilities are clipped to [0,1].
- Paired binary outcomes coupled using a fraction of the maximum feasible positive joint-success association. Pair association 0 means independent potential outcomes conditional on the family probabilities; 1 is the Frechet upper bound. This is not a Pearson correlation.
- The gate passes only when the observed absolute success-rate difference is at least +0.10 and the component-clustered paired percentile-bootstrap 95% interval's lower bound is above zero. Degenerate intervals do not pass.
- Each simulated study uses 1,000 component-bootstrap replicates, resampling whole components with replacement and recomputing the target-weighted treatment-minus-control difference, as in PITB's paired success-cost frontier analysis.

For the initial size vector, the 20 component sizes were `2,3,1,4,2,2,3,1,4,2,3,1,2,4,1,3,2,2,4,1`, totaling 47 targets. For larger designs, the same size pattern was repeated and truncated to the stated number of components.

The table was generated in a separate exploratory NumPy simulation with 10,000 simulated studies and 1,000 component-bootstrap resamples per study. The Wilson intervals quantify Monte Carlo uncertainty in the *simulated gate-pass fraction only*; they do not account for uncertainty about whether the assumed statistical model describes PITB. The calculation follows the same paired component-resampling interval estimator as the Rust planner, but uses a different random-number generator. The numerical estimates are not output from the Rust binary, which has not been compiled or executed in this authoring environment.

## Effect-size sensitivity: 20 components, 47 targets

Fixed assumptions unless otherwise stated: baseline 0.35, ICC 0.05, family effect SD 0.05, pair-association fraction 0.5.

| True configured marginal gain | Estimated gate power | Monte Carlo Wilson 95% interval |
|---:|---:|---:|
| +10 percentage points | 26.52% | 25.66–27.39% |
| +15 percentage points | 45.83% | 44.86–46.81% |
| +20 percentage points | 66.75% | 65.82–67.67% |
| +25 percentage points | 82.52% | 81.76–83.25% |

The +10 point row is not expected to deliver 80% power: the gate itself requires the observed estimate to be at least +10 points *and* the lower confidence bound to be positive. Power planning must specify a plausible true effect larger than the threshold if that threshold remains the acceptance rule.

## Component-count sensitivity: +15-point effect

Fixed assumptions: baseline 0.35, ICC 0.05, family effect SD 0.05, pair-association fraction 0.5. Component sizes follow the repeated/truncated size pattern described above.

| Independent components | Total targets | Estimated gate power | Monte Carlo Wilson 95% interval |
|---:|---:|---:|---:|
| 20 | 47 | 45.69% | 44.72–46.67% |
| 30 | 71 | 61.16% | 60.20–62.11% |
| 40 | 94 | 73.62% | 72.75–74.47% |
| 50 | 118 | 82.21% | 81.45–82.95% |
| 60 | 141 | 84.16% | 83.43–84.86% |
| 80 | 188 | 89.89% | 89.28–90.47% |

Within this one synthetic design, roughly 50 independent components reach an 80% chance of passing the configured gate for a +15-point effect. That is a conditional illustration, **not** a prescribed PITB sample size. Cluster composition and treatment-effect variability matter; the eligible real component sizes may be very different.

## Sensitivity to unverified dependence assumptions

| Scenario at configured +15-point gain | Estimated gate power | Monte Carlo Wilson 95% interval |
|---|---:|---:|
| ICC 0.05, family effect SD 0, pair association 0 | 33.11% | 32.19–34.04% |
| ICC 0.05, family effect SD 0.20, pair association 0 | 29.55% | 28.66–30.45% |
| ICC 0.20, family effect SD 0.05, pair association 0.5 | 51.37% | 50.39–52.35% |
| ICC 0.01, family effect SD 0.05, pair association 0.5 | 45.07% | 44.10–46.05% |

The table shows why the pair-dependence and family treatment-effect assumptions must be varied, not picked to obtain a convenient power estimate. The direction and size of the change are model-dependent. No row is an empirical estimate of PITB's ICC, pair association or heterogeneous treatment effect.

## Decision

1. Keep the 20-component rule as a minimum eligibility floor, not a power certification.
2. Once the relationship audit is signed off, freeze the actual eligible component-size vector and publish a preregistered multi-dimensional sensitivity grid over baseline success, expected effect, ICC, family treatment-effect SD, and pair association.
3. Set the desired gate power (80% is a reasonable candidate, not an existing requirement) before test outcomes are unsealed. If the available sample cannot reach the target under defensible assumptions, label the study exploratory or collect more prospective components.
4. Ask an independent statistician to review both the generator and analysis method. Cross-check against a second generative model and use the exact pre-registered final interval method for the final result.
5. Do not turn this illustration, the 20-component floor, or a synthetic CI fixture into a claim that PITB is adequately powered.

## Method references

Simulation-based power planning is particularly relevant when binary outcomes, unequal cluster sizes, and a small number of clusters make simple design-effect calculations unreliable. Guidance recommends sensitivity to ICC and cluster-size assumptions; Monte Carlo comparisons of clustered binary-outcome methods also show that the chosen analysis can influence power.

- [Key considerations for designing, conducting and analysing a cluster randomized trial](https://academic.oup.com/ije/article/52/5/1648/7172835)
- [A comparison of the statistical power of different methods for the analysis of cluster randomization trials with binary outcomes](https://pubmed.ncbi.nlm.nih.gov/17238238/)

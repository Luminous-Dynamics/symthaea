# NANO-017 — Synthetic quantity-of-interest-aware acquisition benchmark

This is a known-model benchmark for testing acquisition decisions before running a scientific campaign. It does not use material data and is not a physical simulation.

## Run

From the repository root:

~~~sh
cargo run -p symthaea-nanotech-audit --bin nano-qoi-acquisition
~~~

Optional arguments are reference budget, number of random baseline trials, and random seed. Defaults are 8, 256, and 0x51a7 (decimal 20903). The JSON report records the seed and trial count so the random baseline is reproducible. Each arm reports the fraction of runs reaching the preregistered synthetic QoI posterior-variance tolerance of 0.03 within budget. It also reports mean calls to tolerance among successful runs; for random acquisition this mean is explicitly conditional on success.

## Known-model scenario

The model contains three latent parameters with identity prior covariance. The target quantity of interest is theta_0. Each reference call gives one noisy linear observation with variance 0.1.

There are 16 candidate observations:
- Eight candidates are strongly informative about theta_0 and have small nuisance-parameter components.
- Eight have higher feature norms and are much more informative about nuisance parameters theta_1 and theta_2, while being weakly informative about theta_0.

This is intentionally constructed so reducing total parameter uncertainty is not identical to reducing uncertainty in the task's target property.

## Compared acquisition rules

1. **QoI-variance greedy:** select the candidate with the largest expected reduction in posterior variance of theta_0.
2. **Parameter-trace greedy:** select the candidate with the largest expected reduction in the trace of the full posterior covariance.
3. **Uniform random:** sample candidates without replacement; report the mean QoI variance and parameter-trace curve across a seeded set of random orders.

For posterior covariance P, a candidate feature vector x and known observation-noise variance sigma-squared, the expected covariance update is:

~~~text
P_next = P - (P x x' P) / (sigma-squared + x' P x)
~~~

For target direction a, expected target variance reduction is:

~~~text
Delta_QoI = (a' P x)^2 / (sigma-squared + x' P x)
~~~

The generic trace objective instead scores the trace of the covariance reduction. No target observations are needed for these covariance calculations under the declared linear-Gaussian model; the model is fully specified by construction.

## Independent numerical sanity values

A separate Python reference calculation of the same closed-form covariance update gives these expected target-posterior variances for the default seed and random trial count. These are **analytic synthetic-harness checks, not Rust test results and not Symthaea materials results**.

| Reference calls | QoI-variance greedy | Parameter-trace greedy | Uniform random mean |
|---:|---:|---:|---:|
| 1 | 0.091224 | 0.998379 | 0.530846 |
| 4 | 0.024390 | 0.047674 | 0.079576 |
| 8 | 0.012346 | 0.024373 | 0.026083 |

The example illustrates the intended failure mode: generic uncertainty reduction can spend early references on nuisance parameters and leave the target property poorly constrained. The target-aware arm should reach a given QoI-variance tolerance in fewer reference calls in this designed scenario. The exact budget advantage on real potentials is unknown and must be measured.

## Validation gates

- Deterministic JSON output for a fixed seed.
- Exact synthetic reference curves checked against the independent numeric values above.
- Target-aware greedy must select a QoI-informative candidate first; generic trace greedy must select a nuisance-informative candidate first.
- Posterior covariance stays symmetric with positive diagonal in the tested updates.
- Budget/trial input validation fails closed.
- Compare against random acquisition under the same candidate set and call budget; report QoI posterior variance and global trace separately.

## Scope ceiling

This is a method unit test, not an interatomic potential, reference-calculation driver, calibration result, data acquisition result or proof that quantity-of-interest acquisition helps real materials research. Before promoting any policy, pair it with NANO-013's independent probability-sample audit, NANO-014's local severe-error detection, NANO-015's risk-constrained decision objective, and NANO-016's valid risk-bound method. The benchmark must eventually be rerun on licensed reference data with parent-system splits, shifted interface/defect families, actual all-in cost, and preregistered QoI accuracy and safety gates.

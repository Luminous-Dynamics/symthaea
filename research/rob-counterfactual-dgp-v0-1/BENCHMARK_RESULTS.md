# MCR² v17 — Synthetic known-truth benchmark results

**Run date:** 2026-11-13  
**Version:** 0.1  
**Result class:** deterministic synthetic reference DGP only. No uncertainty intervals, hypothesis tests, physical data or SOTA claim.

Configuration: seed 20261113; 10,000 matched blocks per scenario; 20,000 episodes per scenario; fixed horizon 20; one reference-safe and one anticipatory-safe assignment per block; assignment before time zero; high/low observed pre-assignment risk strata; competing hazard 0.004.

## Reference-policy fixed-horizon boundary risk

| Scenario | Oracle truth | Complete-case estimate | Complete-case absolute error | Oracle-known observation-probability diagnostic | Diagnostic absolute error |
|---|---:|---:|---:|---:|---:|
| Complete follow-up | 59.69% | 59.69% | 0.00 pp | 59.69% | 0.00 pp |
| MCAR endpoint loss | 60.43% | 60.44% | 0.01 pp | 60.29% | 0.14 pp |
| Conditional endpoint loss | 60.39% | 50.88% | 9.51 pp | 59.99% | 0.40 pp |
| Positivity failure | 60.11% | 31.55% | 28.56 pp | **Refused** | Not identified |

Conditional endpoint observation probability is 0.48 in the high-risk stratum and 0.96 in the low-risk stratum. Under positivity failure, high-risk observation probability is zero and low-risk observation probability is 0.96. The oracle-known calculation is not a fitted censor model, not an operational IPW implementation, and not a substitute for #9690.

## Policy-value lane

Complete-follow-up scenario:

| Measure | Boundary risk / difference |
|---|---:|
| Oracle reference-safe arm | 59.69% |
| Oracle anticipatory-safe arm | 49.68% |
| Randomized fixed-horizon ITT candidate-minus-reference difference | **−10.01 pp** |
| Naive received-intervention-minus-not-received contrast | **+17.30 pp** |

The as-treated contrast reverses direction because receipt is selected by baseline risk. It is deliberately reported as a biased non-ITT comparison, not as treatment value. The randomized ITT is the policy estimand in this synthetic complete-follow-up scenario. Neither result identifies an individual episode's counterfactual.

## Denominators and competing events

| Scenario | Assigned episodes | Complete endpoints | Unknown endpoints |
|---|---:|---:|---:|
| Complete follow-up | 20,000 | 20,000 | 0 |
| MCAR endpoint loss | 20,000 | 17,943 | 2,057 |
| Conditional endpoint loss | 20,000 | 14,349 | 5,651 |
| Positivity failure | 20,000 | 9,627 | 10,373 |

Unknown endpoints are not treated as event-free. Competing terminal events are counted separately. The unadjusted fixed-horizon ITT summary is withheld whenever assigned outcomes are incomplete; the full assigned denominator is retained.

## Local verification

- 15/15 Python regression tests pass.
- JSON Schema validates 64 observed sample records and 64 evaluator-only oracle sample records.
- Clean archive extraction, SHA-256 manifest verification and deterministic report replay pass.
- No Rust compilation, CI workflow result, simulator, ROS 2, HIL, sensor, actuator or physical robot execution is claimed.

## Interpretation

This DGP demonstrates why endpoint observation mechanisms and assignment semantics must be part of the benchmark contract. It does not validate a survival or causal estimator, establish real-world assumptions, quantify uncertainty, or demonstrate anticipatory reflex improvement. The next research step is to compare the synthetic diagnostics against known-truth mechanisms, add sensitivity ranges for unmeasured informative missingness, then bind the independently qualified protocol to instrumented simulation and JointLab evidence.

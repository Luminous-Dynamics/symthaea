# CASTING-001C — Frozen Numerical Oracle and Golden Fixtures

**Status:** Draft research verification tranche, stacked on CASTING-001A and 001B  
**Evidence class:** Synthetic known-answer fixtures only  
**Claim ceiling:** Arithmetic/software contract; no physical casting validation

## Purpose

A candidate module's tests can be self-consistent while implementing the wrong formula, units, or reference behavior. CASTING-001C therefore freezes independent expected values and compares the public Rust API against them. Expected numbers below come from the declared equation and hand-checkable inputs, not from capturing output produced by the implementation under test.

The tests live in `physics/casting_golden_tests.rs` and are registered only under `cfg(test)`. This tranche adds no production API, process registry, recipe, equipment integration, or execution authority.

## Frozen model fixture

For the synthetic fixture only, declare

`t = B m^n`, with `B = 1,200,000 s/m²`, `n = 2`, and thermal modulus `m = V/A` in metres.

| Role | Thermal modulus (m) | Expected time (s) |
|---|---:|---:|
| Training | 0.001 | 1.2 |
| Training | 0.002 | 4.8 |
| Training | 0.004 | 19.2 |
| Reference query | 0.003 | 10.8 |

The fitting oracle requires exponent 2 within absolute tolerance `1e-10`, mold constant 1,200,000 within `1e-5`, and all listed point predictions within `1e-7 s`. The exact three-point relation has a positive calibration domain from 0.001 m through 0.004 m; values outside that interval must not be extrapolated by this API.

## Frozen SI geometry fixture

The geometry ratio oracle is:

- volume: `6e-6 m³`;
- declared effective cooling surface area: `2e-3 m²`;
- expected thermal modulus: `0.003 m`, absolute tolerance `1e-12 m`.

This tests unit arithmetic only. It does not prove that every geometric surface exchanges heat equally; the effective cooling-area convention still requires a documented physical definition.

## Frozen conformal interval fixture

Use the same fitted model and three distinct calibration observations:

| Thermal modulus (m) | Model estimate (s) | Calibration value (s) | Absolute log residual |
|---:|---:|---:|---:|
| 0.0015 | 2.7 | 2.97 | `ln(1.1)` |
| 0.0025 | 7.5 | 7.125 | `abs(ln(0.95))` |
| 0.0035 | 14.7 | 16.17 | `ln(1.1)` |

At `alpha = 0.25`, the finite-sample rank is `ceil((3 + 1) × (1 - 0.25)) = 3`; the frozen radius is `ln(1.1) ≈ 0.09531017980432493`. For query `m = 0.003 m`, the expected estimate is 10.8 s, the lower bound is `10.8 / 1.1 ≈ 9.818181818181818 s`, and the upper bound is `10.8 × 1.1 = 11.88 s`.

Tolerances: quantile `1e-8`, interval bounds and point estimate `1e-7 s`, nominal coverage `1e-12`. With only three synthetic scores, this is a software known-answer case; it is not a useful calibration sample recommendation for a physical process and does not validate exchangeability.

## Negative controls

The test suite includes the following fail-closed and mutation-sensitive controls:

1. Change the last training outcome from 19.2 s to 21.12 s. The refitted prediction at 0.003 m must move by more than 0.1 s from the frozen 10.8 s oracle.
2. Query below the trained domain (0.0009 m). The model must return the outside-domain error.
3. Evaluate one outside-domain and one in-domain held-out observation. Coverage must be 1/2 and the outside-domain point must be recorded as an abstention, not removed from the denominator.
4. Reject zero volume and non-finite effective surface area.

These are fixed negative controls, not a substitute for property-based fuzzing or systematic mutation testing. A future tranche should add automated mutation tests for sign errors, exponent changes, SI scale changes, interval quantile rank changes, and altered coverage denominators.

## Benchmark comparison ladder

A fair future comparison must freeze partitions before fitting and compare, on the same inputs:

1. A predeclared `n = 2` baseline.
2. A learned-exponent Chvorinov baseline.
3. A training-only geometric-mean / median predictor.
4. Where the geometry/process and material assumptions match, a documented independently validated filling/thermal solver.
5. A physics-plus-residual or multi-fidelity candidate only after the previous baselines have reproducible results.

Publish accuracy together with selective coverage/abstentions, empirical interval coverage and width, domain-shift degradation, latency, measured experiment count, and operational effort. For prospective physical trials, add first-pass yield, scrap/rework mass per accepted mass, cycle time, and metered energy per accepted unit. Compare physical and simulated calls separately. Do not compare a sand-cast coupon's numerical errors directly to tensile-strength gains reported for a different squeeze-casting alloy or process.

## Validation state

The tests are authored but were not executed in the authoring environment because Rust/Cargo is unavailable there. GitHub checks on the CASTING-001B predecessor were reported as skipped; that is not passing evidence. Required promotion gates are compilation and tests on the exact latest head, plus review of the numeric/statistical semantics. Synthetic fixture agreement is not evidence of foundry accuracy, process capability, part integrity, yield improvement, energy reduction, or industrial superiority.

## Research context

- [NIST, Digital Twins Workshops Summary Report (21 July 2026)](https://www.nist.gov/publications/digital-twins-workshops-summary-report) — manufacturing digital-twin priorities including interoperability and VVUQ.
- [NIST, Methods of Uncertainty Quantification in Computational Modeling and Simulation (5 October 2026)](https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing) — risk-based credibility and alignment of simulation and metrology practices.
- [Essam et al. (2026), review of computational modeling, AI, and digital twins in metal casting](https://link.springer.com/article/10.1007/s43939-026-00685-5) — identifies reproducible shared benchmarks, quantified uncertainty, cross-population generalization, and industrial validation as gaps.

These references motivate the verification program; no benchmark result or physical benefit is claimed here.

# AR-5B24 Wave 1: Executable Orthogonal/Aligned Shift Benchmark v0.1

**Status:** Rust benchmark implementation draft; tests authored but not executed.  
**Tracking:** [#9971](https://github.com/Luminous-Dynamics/symthaea/issues/9971)  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/functional_shift_benchmark.rs`  
**CLI:** `crates/domains/symthaea-architecture-engineering/examples/functional_shift_benchmark.rs`  
**Branch:** `research/ar5b24-functional-shift-benchmark-v0.1-20261102`  
**Dependency base:** AR-5B23 / PR #9999.

## 1. Purpose

This is the first executable Wave-1 campaign for the uncertainty-aware decision programme. It compares:

- four predeclared bounded decision functionals using one-sided Hoeffding bounds with Bonferroni family allocation;
- a finite-alphabet Weissman TV confidence slack plus empirical TV distance to a nominal distribution;
- the deliberate limitation that a finite functional family does not cover every possible path function.

It is a synthetic categorical benchmark. It does not implement the full policy-frontier LP comparison, target-law transport, a real experiment, or an end-to-end Type-I test.

## 2. Frozen categorical support and functionals

There are eight path categories, indexed 0 through 7.

Nominal distribution:

```text
P0 = (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125)
```

Four predeclared bounded functionals:

- rejection A: `g1 = (0,0,1,1,0,0,1,1)`;
- rejection B: `g2 = (0,1,0,1,0,1,0,1)`;
- scaled evaluator cost: `g3 = (1,1,0,0,1,1,0,0)`;
- scaled decision utility: `g4 = (0,0,0,0,1,1,1,1)`.

Their nominal expectations are all 0.50. The one-sided contract uses upper bounds for g1/g3 and lower bounds for g2/g4. Each function has range [0,1].

A fifth diagnostic function is deliberately omitted from the DFCE vector: `g5 = indicator(category == 0)`. This is not protected by the four-function confidence statement.

## 3. Known-answer scenarios

### F01 — distribution shift orthogonal to the declared decision vector

```text
Q_orth = (0.20, 0.05, 0.15, 0.10, 0.10, 0.15, 0.05, 0.20)
TV(P0, Q_orth) = 0.20
E_Q[g1..g4] = (0.50, 0.50, 0.50, 0.50)
E_P0[g5] = 0.125
E_Q[g5] = 0.20
```

The path distribution moves substantially, while every declared DFCE expectation is unchanged. The omitted fifth function changes by +0.075. This is an explicit negative control against interpreting DFCE coverage as full-distribution coverage.

### F02 — decision-aligned distribution shift

```text
Q_aligned = (0.10, 0.10, 0.15, 0.15, 0.10, 0.10, 0.15, 0.15)
TV(P0, Q_aligned) = 0.10
E_Q[g1..g4] = (0.60, 0.50, 0.40, 0.50)
E_Q[g5] - E_P0[g5] = -0.025
```

Rejection A rises by 0.10 and scaled evaluator cost falls by 0.10. The direct confidence bound must track the changed expectation rather than dismissing it as irrelevant.

## 4. Confidence calculations

For four one-sided [0,1]-bounded functionals, sample size n and familywise failure probability delta:

```text
delta_per_functional = delta / 4
r_DFCE = sqrt(log(4 / delta) / (2*n))
```

For a categorical support of size K, the conservative Weissman finite-alphabet TV concentration slack is:

```text
r_TV_slack = sqrt((K*log(2) + log(1/delta)) / (2*n))
sample_based_TV_radius = min(1, empirical_TV(P_hat, P0) + r_TV_slack)
```

At K=8, n=5,000 and delta=0.01:

- scalar Hoeffding radius: approximately 0.02447747;
- Weissman TV concentration slack: approximately 0.03185961.

The full TV radius also includes the observed empirical TV distance to P0. The confidence scopes differ: TV bounds arbitrary path functionals over the declared finite path support; DFCE bounds only g1–g4.

## 5. Executable interface

The crate exports `functional_shift_benchmark`. The example emits a JSON report containing:

- scenario, seed, sample sizes and confidence allocation;
- exact nominal/target functional expectations;
- omitted-function shift and true TV distance;
- Hoeffding radius and Weissman slack;
- simultaneous DFCE coverage count/rate and Wilson 95% interval;
- mean empirical TV distance;
- mean, median, 5th and 95th percentile sample-based TV radius;
- deterministic request/report SHA-256 identities.

Example commands from the repository root:

```bash
cargo run --locked -p symthaea-architecture-engineering --example functional_shift_benchmark -- \
  --scenario orthogonal --seed 20261102 --replications 20000 --samples 5000

cargo run --locked -p symthaea-architecture-engineering --example functional_shift_benchmark -- \
  --scenario aligned --seed 20261102 --replications 20000 --samples 5000

cargo test --locked -p symthaea-architecture-engineering functional_shift_benchmark
```

The internal SplitMix64 generator is deterministic and suitable for reproducible simulation. It is not cryptographic randomness. The Wilson interval quantifies Monte Carlo uncertainty in the observed replication coverage; it is not the per-replication Hoeffding guarantee.

## 6. Tests authored

The module defines six tests:
1. exact F01 expectation/TV and omitted-function shift;
2. exact F02 expectation/TV;
3. frozen Hoeffding and Weissman formula values;
4. deterministic seeded report reconstruction;
5. zero-replication and invalid-confidence-budget rejection;
6. support-size scaling of the TV slack with fixed functional dimension.

Tests are authored, not run. No Rust compile, rustfmt, Clippy, unit-test, CI, coverage, or performance result is claimed.

## 7. Next benchmark extensions

- Add the independent target-law mismatch negative control, which must not be qualified by an independence flag alone.
- Add a family of omitted-function traps and adaptive functional selection.
- Compare scalar Hoeffding against a theorem-faithful vector empirical-Bernstein implementation (#10009).
- Add policy/model family dimensions and connect each benchmark run to #9999's calibrated frontier adapter.
- Add independent multinomial/exhaustive small-support oracle calculations.
- Extend to cluster/Markov regimes only with a matching concentration theorem; do not relabel dependent data as IID.
- Connect to #10006 when guardrail constraints are inside the LP.

## 8. Claim boundary

The benchmark can demonstrate the support-scaling and functional-scope mechanisms on these frozen synthetic scenarios. It cannot establish that the four declared functions are complete, that calibration and confirmation share a target law, that the Rust implementation is correct before execution, or that DFCE is universally superior to TV/GTV/Wasserstein.
# CMAQ-045 — Predictable conditional rank-spectrum dependence monitor

## The blind spot this targets

CMAQ-038 covers a complete dyadic Haar basis over the finite predictive-rank histogram. A stream can nevertheless preserve an approximately uniform marginal rank distribution while changing its temporal order. Examples include persistence, alternating regimes, burstiness and conditional tail behavior.

CMAQ-045 adds a fixed bank of predictable contexts. Each context is derived solely from ranks observed before the current rank. It then tests the current rank's complete Haar spectrum conditional on that past context.

## Conditional e-process construction

For calibration size n and prior predictive-rank counts N_(t-1,j), under the PRM exchangeability null:

    pi[t,j] = (1 + N[t-1,j]) / (n+t).

For each dyadic Haar contrast h_k taking values in {-1/2, 0, +1/2}, its exact predictive centering is

    mu[t,k] = sum_j pi[t,j] h_k(j)
    z[t,k] = h_k(R_t) - mu[t,k].

For every context c fixed from history F_(t-1), direction d ∈ {-1,+1}, and fixed rate lambda ∈ {1/4, 1/2, 1}, define

    f[t,c,k,d,lambda] =
        1 + lambda * d * 1{c(F_(t-1))} * z[t,k].

Because the context indicator is F_(t-1)-measurable and E[z[t,k] | F_(t-1)] = 0, the conditional expected factor is one. Since |z| <= 1 and lambda <= 1, the factor is nonnegative. Every fixed component is therefore an e-process under the exact predictive-rank null. A normalized, pre-registered mixture over contexts, rank features, directions and rates is also an e-process.

When a context is inactive, its component wealth is left unchanged (factor one). The context cannot inspect the current rank before the bet is applied.

## Frozen context bank

The initial version contains 14 contexts:
- 8 one-hot previous-rank buckets partitioning the n+1 rank buckets into equal-width groups;
- previous two-step movement up;
- previous two-step movement down;
- previous extreme-low rank;
- previous extreme-high rank;
- prior high run of length at least two;
- prior low run of length at least two.

Multiple contexts can be active simultaneously. Context definitions, rank cutoffs, feature basis, betting rates and mixture weights are fixed at construction. No context is invented or tuned after seeing an alarm.

For n calibration observations:
- n+1 predictive rank buckets;
- exactly n nonempty dyadic Haar features;
- 14 frozen contexts;
- 2 betting directions;
- 3 fixed rates;
- 84n component accounts.

State size is O(14n). One update performs an O(n) predictive-prefix pass plus wealth updates for components in active contexts (normally a small subset). The monitor stores rank counts and the last two ranks plus high/low run lengths. It uses log wealth, checked counters, and permanent quarantine.

## Paired dependence probe

Run from the repository root:

    cargo +1.100.0 run --release --manifest-path research/cmaq-045/Cargo.toml --locked --bin dependence_probe -- 1000 2000 20261127 1000 > cmaq-045-results.csv

Arguments are replication count, horizon, seed and number of pre-change null observations. Defaults are 64, 800, 20,261,127 and 300.

The probe compares CMAQ-038's histogram spectrum with CMAQ-045's conditional-context spectrum. Both receive identical rank streams:
- exact Pólya-urn predictive-rank null;
- a stationary-uniform sticky Markov alternative that starts from a uniform rank at the change point, then repeats its previous rank with probability 0.8 and otherwise redraws uniformly over all rank buckets. This initializes the post-change Markov chain in its exact uniform stationary law rather than inheriting the preceding Pólya-urn state.

The Markov transition matrix has the uniform distribution as a stationary marginal but introduces serial persistence. This is a synthetic diagnostic of a temporal-dependence blind spot; it is not a physical dataset, an ImageNet-C/CCTM reproduction or a general claim that every temporal dependency shift is detected.

CSV records per-replication alarm times. Stderr reports null-arm alarms as false alarms, and for the shifted arm separates pre-change false alarms, post-change detections and unconditional misses. An alarm at or before the declared shift time is never counted as a detection; it remains a miss in the unconditional denominator. Median delay is measured from the true onset among post-change detections only and is always accompanied by the miss count. The probe's deterministic SplitMix64 generator is non-cryptographic.

The CI run is a small wiring smoke test only. It cannot establish alpha=.05 control or comparative power.

## Deterministic tests

The unit tests cover:
- unit initial mixture and full frozen component inventory;
- zero active contexts before the first observation;
- context activation determined only by committed prior ranks;
- conditional centering across the full context-feature bank under the exact predictive-rank law;
- nonnegative factors for all ranks and rates;
- invalid rank and counter-overflow state atomicity;
- high/low run activation only after two prior observations;
- movement contexts only after two prior observations;
- permanent quarantine.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-045/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-045/Cargo.toml --locked --all-targets -- -D warnings

## Required full benchmark before a superiority claim

Use paired streams, frozen score mapping, tie protocol, effective calibration blocks and identical alpha/horizon:
- exact Pólya-urn null;
- stationary-uniform sticky Markov ranks;
- balanced alternation;
- short/long bursts;
- conditional high-tail and low-tail transitions;
- regime-dependent dispersion;
- gradual changes in dependence strength;
- marginal-preserving temporal mixtures;
- deliberately context-orthogonal alternatives.

For n in {49, 99, 199, 499, 999}, alpha in {0.01, 0.05, 0.10}, change points 0, 125, 500, 2,000 and 8,000, and horizons matching the published PRM/PF-CCTM protocol:
- at least 10,000 independent full-horizon null paths per cell;
- at least 5,000 alternatives per cell;
- report full-horizon ever-alarm probability with binomial confidence intervals;
- report power, unconditional miss rate and median/p90/p95 delay from true onset;
- report per-family worst case, p50/p95/p99 CPU, memory and update latency;
- compare CMAQ-031, CMAQ-038, CMAQ-040, CMAQ-045 and faithful PRM/CCTM/PF-CCTM arms.

CMAQ-044 paired effective-block inference should be used for delay uncertainty. A candidate is better only if it improves preregistered outcomes without violating null error and without hiding context families where it regresses.

## Validity and claim boundary

The null guarantee is conditional on the fixed calibration reference, correct score-to-rank mapping, valid randomized ties for score atoms, exchangeability of the effective blocks, and correct PRM predictive-rank law. The contexts are predictable transformations of the past and do not create a new null assumption.

The detector only covers the frozen finite context/contrast/rate portfolio. It cannot detect every dependence alternative; an alternative that does not alter any tested conditional contrast may be invisible. It does not authenticate input rank receipts, prove common-cause closure or physical independence, infer hazard meaning, certify a fallback, or establish safety.

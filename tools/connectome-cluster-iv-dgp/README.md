# Deterministic clustered IV DGP and cross-language baseline export

Standalone, dependency-free Rust 2024 generator for clustered instrumental-variable designs with known structural effects, known direct instrument effects, cluster dependence and unit-level endogeneity.

## Structural model

For assignment unit i in cluster g:

```
X_ig = Z_ig' * pi_g + u_g + u_ig
Y_ig = beta * X_ig + Z_ig' * gamma_g + alpha_g + epsilon_ig
```

- beta is the known structural effect of the endogenous treatment.
- pi_g is the cluster-specific first-stage coefficient vector. Its multiplier can vary by cluster, including zero-relevance clusters and strength concentrated in only a few clusters.
- gamma_g is the known direct instrument-to-outcome coefficient vector. It can be exactly zero or cluster-varying, allowing direct effects inside or outside a registered support.
- Cluster shocks (u_g, alpha_g) and unit shocks (u_ig, epsilon_ig) have separately configured endogeneity correlations.
- Instrument columns are equicorrelated with an explicit non-negative pairwise correlation parameter.
- Unequal cluster sizes and cluster-specific noise scales are explicit.

The instrument draw is independent of the cluster and unit disturbances by construction. This is an exogenous-instrument DGP with a deliberately controllable direct outcome path. It is not an instrument-validity proof for biological interventions.

## Assignment level is explicit

The generator distinguishes:

- `WithinCluster`: every row receives its own instrument vector, so instruments vary within clusters.
- `ClusterLevel`: one instrument vector is drawn per cluster and copied to every row in that cluster.

This distinction matters for identification. A cluster-level instrument has no within-cluster variation; an estimator or fixed-effect design that absorbs the cluster cannot identify its first stage. Both regimes must be evaluated separately.

## Configuration

The Rust API is `ClusterIvDgpConfig`. Set:

- scenario ID and seed;
- beta, pi and gamma;
- cluster-size vector;
- per-cluster first-stage, direct-effect and shock multipliers (empty vectors default to one);
- instrument equicorrelation;
- cluster and unit shock standard deviations;
- cluster-level and unit-level treatment/outcome shock correlations;
- instrument assignment level.

The PRNG is SplitMix64 and standard normal values use Box–Muller cosine generation. The algorithm identifier and seed are included in the manifest. Same seed/config on the same IEEE-754 implementation produces byte-identical output. CSV numbers are written to 17 significant decimal digits for round-trip parsing.

## Export

From the repository root:

```sh
cargo run --locked --manifest-path tools/connectome-cluster-iv-dgp/Cargo.toml --bin export-example -- /tmp/connectome-cluster-iv-dgp
```

This writes:

- `/tmp/connectome-cluster-iv-dgp.csv`: one row per assignment/observation unit, including stable unit ID, cluster ID, z0...zN, treatment, outcome, known first-stage and direct-effect components, and realized disturbances;
- `/tmp/connectome-cluster-iv-dgp.manifest.json`: seed, generator algorithm, true beta/pi/gamma, cluster sizes, assignment level, cluster multipliers, DGP assumptions, summary and per-cluster realized truth.

Hash the CSV and manifest separately with SHA-256 before running the estimators or unblinding any score. The generator deliberately does not bundle a hash implementation or infer an experiment-specific split.

## R baseline: clusterIV 0.2.0

The script `examples/clusteriv_020.R` checks the exact package version, imports the same CSV, builds the instrument matrix, and runs all four required CJAR configurations:

- plain variance + shifted/scaled chi-square calibration;
- plain variance + normal calibration;
- leave-two-cluster-out cross-fit variance + chi-square calibration;
- leave-two-cluster-out cross-fit variance + normal calibration.

Each call uses `iv_infer()`, which returns CJIVE, CJAR and CJS from one shared preprocessing pass. The script preserves the complete printed result (including confidence-set topology and effective first-stage/leverage diagnostics) in a separate output file. It has not been executed here.

Install the exact CRAN release in a pinned R environment:

```r
install.packages("clusterIV")
packageVersion("clusterIV")
```

The adapter requires version 0.2.0. Reference: https://ftp.sun.ac.za/ftp/CRAN/web/packages/clusterIV/refman/clusterIV.html.

## Stata 19 baseline

`examples/stata19_weakrobust.do` imports the same CSV, encodes the stable string cluster ID, runs 2SLS with cluster-robust VCE, reports first-stage diagnostics, and explicitly requests both `estat weakrobust, ci ar` and `estat weakrobust, ci clr rseed(2026)`, then repeats AR/CLR under robust VCE separately. Retain the full Stata log and any simulated CLR replication details. It has not been executed here.

Official documentation: https://www.stata.com/new-in-stata/inference-robust-to-weak-instruments/.

## How this fits the benchmark

Use identical CSV rows, cluster IDs, instrument columns, controls and target parameter for:

1. clusterIV CJIVE/CJAR/CJS;
2. Stata 19 weak-robust AR/CLR;
3. Wang–Zhang wild-bootstrap AR when the fixed-cluster/large-within-cluster asymptotic regime is defensible;
4. the independent-row pAR only in its conditional-Gaussian, fixed-design, homoskedastic, independent-row regime;
5. future bounded-direct-effect cluster methods, only after a derivation and simulation calibration.

The direct effect gamma is known in this DGP. Run g=0, an in-bound support, a boundary support and an out-of-bound support. Do not score a method as better because it returns a narrower set with invalid size or coverage.

The exact blocked randomization ITT oracle (PR #9673) is a different estimand and requires separately generated randomization receipts. Do not treat the instrument vector in this DGP as a randomized assignment receipt unless the assignment protocol is explicitly represented and exported.

## Synthetic known-answer fixture

`fixtures/known-answers.json` pins a four-row, two-cluster, two-instrument design with seed 42, cluster-level first-stage multipliers [1.0, 0.5] and cluster-level direct-effect multipliers [1.0, 1.2]. It records exact initial rows, cluster shocks and sample summaries. This permits cross-language parity checks independently of any estimator.

## Limitations

- One endogenous treatment and one scalar outcome.
- One-way clusters; no crossed or multi-way random effects.
- Gaussian disturbances in the reference DGP; heavy-tailed alternatives are a separate future scenario family.
- Instrument equicorrelation is restricted to [0,1); arbitrary covariance matrices are not yet supported.
- The generator produces synthetic instrumental-variable rows, not neuronal observations or an experimental assignment schedule.
- No R, Stata or Rust execution result is claimed in this PR.

## Run tests

```sh
cargo test --locked --manifest-path tools/connectome-cluster-iv-dgp/Cargo.toml
```

There are 27 authored unit tests. They have not been compiled or executed in the current authoring environment. Zero third-party Rust dependencies; isolated Cargo workspace; AGPL-3.0-only.

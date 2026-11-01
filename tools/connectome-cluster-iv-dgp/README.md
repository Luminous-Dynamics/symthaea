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
- per-cluster first-stage, direct-effect and noise multipliers (empty vectors default to one); the noise multiplier scales both cluster-level and unit-level disturbances;
- instrument equicorrelation;
- cluster and unit shock standard deviations;
- cluster-level and unit-level treatment/outcome shock correlations;
- instrument assignment level.

The generator algorithm is `splitmix64-box-muller-cos-open01-v2`: SplitMix64 with 53-bit mantissa uniforms clamped to the open interval (0,1), followed by Box–Muller cosine generation. The algorithm identifier is included in the public manifest; the seed is held in the restricted oracle so it cannot leak structural truth to estimator developers. Same seed/config on the same IEEE-754 implementation produces byte-identical output. CSV numbers are written to 17 significant decimal digits for round-trip parsing.

## Export

From the repository root:

```sh
cargo run --locked --manifest-path tools/connectome-cluster-iv-dgp/Cargo.toml --bin export-example -- /tmp/connectome-cluster-iv-dgp direct-effect-outside-support
```

The second positional argument selects a registered scenario preset. Supported values are `example`, `strong-exact-exclusion`, `weak-exact-exclusion`, `zero-relevance`, `direct-effect-inside-support`, `direct-effect-outside-support`, `direct-effect-at-boundary`, `cluster-specific-direct-effects`, `many-correlated-instruments`, `few-large-clusters`, `many-small-clusters`, `strength-concentrated-in-few-clusters`, `cluster-level-instruments`, and `unequal-heteroskedastic-clusters`. Each preset uses the explicit seed 20261101 in the example exporter; the Rust API accepts arbitrary seeds. The restricted oracle records the registered direct-effect RMS bound (g), realized direct-effect RMS and whether the generated sample is `IN_BOUND`, `OUT_OF_BOUND`, or `UNREGISTERED`. The public manifest does not disclose any of these truth values. The `direct-effect-at-boundary` preset derives g from known synthetic direct-effect truth before running any estimator; it is a boundary calibration fixture, not an empirical control-based bound. The high-dimensional preset uses 25 correlated instruments; the cluster-size presets separate five large clusters from 100 small clusters.

By default, this writes only two estimator-safe artifacts:

- `/tmp/connectome-cluster-iv-dgp.csv`: estimator-visible rows only—stable unit ID, cluster ID, z0...zN, treatment and outcome. It contains no beta/pi/gamma, decomposed direct effect, disturbances or cluster-specific truth multipliers;
- `/tmp/connectome-cluster-iv-dgp.manifest.json`: public layout metadata (row/cluster/instrument counts, cluster sizes, assignment level, generator algorithm and row order). It contains no seed or structural truth and marks the oracle as sealed.

**Ground truth is not written by default.** Only an evaluator-controlled process should request it, using a separate protected path:

```sh
cargo run --locked --manifest-path tools/connectome-cluster-iv-dgp/Cargo.toml --bin export-example -- /tmp/connectome-cluster-iv-dgp direct-effect-outside-support --oracle-out /secure/evaluator/connectome-cluster-iv-dgp.oracle.json
```

That opt-in artifact contains the seed, beta/pi/gamma, registered direct-effect support, cluster shocks and per-row first-stage/direct-effect/error decompositions. The binary prints a warning when it writes this file. Never use a public shared directory as the oracle destination. Hash the observed CSV, public manifest and (when generated) oracle separately with SHA-256. Give only the CSV and public manifest to estimators. Release the oracle only after predictions, model selection and analysis choices are frozen. The generator deliberately does not bundle a hash implementation or infer an experiment-specific split.

## R baseline: clusterIV 0.2.0

The script `examples/clusteriv_020.R` checks the exact package version, imports only the estimator-visible CSV, requires at least three independent clusters for the leave-two-cluster-out cross-fit branch, builds the instrument matrix, and runs all four required CJAR configurations:

- plain variance + shifted/scaled chi-square calibration;
- plain variance + normal calibration;
- leave-two-cluster-out cross-fit variance + chi-square calibration;
- leave-two-cluster-out cross-fit variance + normal calibration.

Each call uses `iv_infer()`, which returns CJIVE, CJAR and CJS from one shared preprocessing pass. The script preserves the complete printed result and serializes the full fitted R object to `.rds` for downstream programmatic inspection of confidence-set topology and diagnostics. The fitted object reports CJIVE, CJAR and CJS together; record `G`, `k`, `F_CJ`, effective first-stage `F_eff`, maximum within-cluster leverage `maxlev`, and full confidence-set topology. Preserve the `conf.set` endpoint matrix as authoritative; `shape`, `n.components`, `unbounded.left` and `unbounded.right` describe its topology. Shapes include `bounded`, `ray`, `two_rays`, `whole_line` and `empty`. CJAR's test is **one-sided** (large positive studentized quadratic form rejects); its confidence set is the inversion of that test. Its `chisq` and `normal` calibrations are asymptotic approximations, not finite-sample exact laws. The polynomial inversion computes set boundaries without a plot-grid search, but numerical roots/endpoints remain floating-point quantities. The cross-fit variance is unbiased at each candidate under its assumptions, but can be non-positive in finite samples; the package uses a conservative acceptance convention in those regions. The adapter does not read the oracle artifact. It also writes session information. It has not been executed here.

Install the exact CRAN release in a pinned R environment:

```r
install.packages("clusterIV")
packageVersion("clusterIV")
```

The adapter requires version 0.2.0. Reference: https://ftp.sun.ac.za/ftp/CRAN/web/packages/clusterIV/refman/clusterIV.html.

## Stata 19 baseline

`examples/stata19_weakrobust.do` imports the same observed-only CSV, encodes the stable string cluster ID, runs 2SLS with cluster-robust VCE, reports first-stage diagnostics, and explicitly requests both `estat weakrobust, ci ar` and `estat weakrobust, ci clr reps(25000) rseed(2026)`, then repeats AR/CLR under heteroskedasticity-robust VCE. It records returned results (`return list`) after each test and preserves a text log. CLR critical values use simulation (25,000 replications here); weak-IV confidence-interval bounds use a grid search and can be disconnected or unbounded. This is a strong commercial comparator, not a finite-sample exact oracle. It has not been executed here.

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

`fixtures/known-answers.json` pins a four-row, two-cluster, two-instrument known-answer dataset with seed 42; it contains the evaluator-side decomposition. `fixtures/small-clustered-iv-known-answers.csv` is its matching observed-only six-column CSV. `fixtures/public-manifest.example.json` and `fixtures/oracle-truth.example.json` demonstrate the separate schemas. Use only the CSV and public manifest for estimator fitting; the oracle fixture is reserved for evaluator-side known-answer checks.

## Blind evaluation and oracle custody

The exporter separates two public artifacts from one opt-in evaluator artifact:

1. **Observed data** (.csv) contains only estimator-visible variables: unit ID, cluster ID, instruments, treatment and outcome.
2. **Public manifest** (.manifest.json) contains data layout and marks oracle custody as SEALED_UNTIL_PREDICTIONS_FROZEN; it excludes the seed, structural parameters and realized decomposition.
3. **Restricted oracle** (.oracle.json) contains the known structural beta, first-stage and direct-effect coefficients, registered support g, realized cluster shocks and row-level decomposition. It is an evaluator artifact, not an estimator input.

The release order is: publish observed CSV + public manifest; hash both; freeze and hash model predictions and selection decisions; only then let the independent evaluator create or access the oracle JSON. The CLI does not write oracle truth unless `--oracle-out` is explicitly supplied, and it rejects an oracle path in the same directory as the public artifacts. The R and Stata adapters read only the observed CSV. Never mount or copy the oracle into the model-fitting environment. The synthetic fixtures/known-answers.json file is intentionally an oracle fixture; the matching small-clustered-iv-known-answers.csv contains only public observed columns.

The DGP's known truth is what makes finite-sample size, bias, power, coverage and false-narrow-set rates measurable. Keeping it separate prevents those metrics from being gamed or accidentally leaked into feature engineering.

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

There are 46 authored library tests plus 3 exporter path-isolation tests (49 total). Coverage includes the 13 named scenario presets, registered direct-effect-support classification, high-dimensional instrument generation, few-large versus many-small cluster regimes, public CSV/manifest non-disclosure, and rejection of an oracle output path sharing the public directory. They have not been compiled or executed in the current authoring environment. Zero third-party Rust dependencies; isolated Cargo workspace; AGPL-3.0-only.

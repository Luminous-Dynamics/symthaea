# Connectome intervention benchmark: clustered weak-IV and assignment-inference comparators

**Status:** benchmark protocol addendum. No baseline has been executed in this repository environment and no biological causal result is claimed.

**Related:** [SCI-CONNECTOMICS-033](https://github.com/Luminous-Dynamics/symthaea/issues/9587), [clustered bounded-direct-effect extension](https://github.com/Luminous-Dynamics/symthaea/issues/9644), [clustered-IV DGP and cross-language export](https://github.com/Luminous-Dynamics/symthaea/pull/9713), [randomized-assignment ITT oracle](https://github.com/Luminous-Dynamics/symthaea/pull/9673), [multi-instrument pAR sensitivity](https://github.com/Luminous-Dynamics/symthaea/pull/9639).

## 1. Keep the estimands separate

A neural optogenetic experiment presents at least three different causal questions:

1. **Randomized assignment ITT:** What is the total effect of assigning the registered stimulation schedule on the measured outcome? This includes all consequences of assignment: opsin activation, direct illumination, heating, extrasynaptic signalling, peripheral effects and measurement paths. It does not require the IV exclusion restriction. PR #9673 implements a synthetic exact blocked-randomization reference for one sharp constant-additive null under uniform complete randomization within the registered strata.
2. **Instrument-mediated structural effect:** What is the effect of source-neuron activity on the downstream outcome under the structural model and instrument assumptions? This needs weak-instrument-robust inference and explicit instrument-validity checks.
3. **Bounded-direct-effect sensitivity:** How does the structural conclusion change when stimulation can affect outcomes through a bounded direct path, rather than assuming exact exclusion? PR #9639 is a multi-instrument, independent-row synthetic pAR reference; it is not clustered animal/session inference.

Never treat an ITT contrast as a source-neuron mediation estimate. Never treat a cluster-robust weak-IV test as automatically robust to nonzero direct effects.

## 2. Existing clustered weak-IV baselines

### CRAN clusterIV 0.2.0

The package provides CJIVE point estimation, cluster-jackknife Anderson–Rubin (CJAR), and cluster-jackknife score (CJS) inference for one endogenous regressor and one-way cluster dependence. It supports multiple excluded instruments, controls, precision weights and high-dimensional fixed effects. The required comparison is:

- `variance="plain"` and `variance="crossfit"` (leave-two-cluster-out);
- CJAR `calibration="chisq"` and `calibration="normal"`;
- CJIVE estimate and interval, CJAR confidence set, and CJS point-null result;
- instrument count `k`, cluster count `G`, `F_CJ`, effective first-stage `F_eff`, maximum within-cluster leverage `maxlev`;
- the authoritative `conf.set` endpoint matrix, plus `shape`, `n.components`, `unbounded.left` and `unbounded.right`.

CJAR is a **one-sided upper-tail** test. The package's shifted/scaled chi-square and normal critical-value calibrations are asymptotic approximations, not finite-sample exact distributions. The grid-free polynomial inversion computes confidence-set boundaries analytically, but numerical roots/endpoints are floating-point values. Cross-fit variance can be non-positive in a finite sample; the documented conservative acceptance convention must be preserved, not patched away.

Possible reported set shapes include `bounded`, `ray`, `two_rays`, `whole_line` and `empty`. Do not collapse a disconnected or unbounded set into one finite interval.

Official package sources:
- [clusterIV 0.2.0 reference manual](https://ftp.sun.ac.za/ftp/CRAN/web/packages/clusterIV/refman/clusterIV.html)
- [clusterIV 0.2.0 NEWS](https://mirrors.ibiblio.org/pub/mirrors/pub/mirrors/CRAN/web/packages/clusterIV/news/news.html)
- [Package README / vector interface](https://mirrors.ibiblio.org/pub/mirrors/pub/mirrors/CRAN/web/packages/clusterIV/readme/README.html)

### Stata 19 estat weakrobust

The commercial comparison should run Anderson–Rubin (AR) and conditional likelihood-ratio (CLR) inference under VCE choices supported by the target design (cluster-robust and heteroskedasticity-robust for this one-way DGP; HAC only when a defensible temporal design is supplied). Record:

- Stata version and exact command;
- AR or CLR, VCE type, confidence level;
- CLR simulation replications and random seed (the current adapter registers 25,000 and seed 2026);
- confidence-set search grid and adaptive expansion settings;
- full topology: finite interval, disconnected union, unbounded, empty or whole real line.

CLR critical values are simulated. Confidence-set bounds use a grid search, not polynomial grid-free inversion. Stata explicitly supports empty, whole-line and union-of-interval confidence sets. Do not report a simulated CLR p-value as exact or a grid-search confidence interval as analytic.

Sources:
- [Stata 19: inference robust to weak instruments](https://www.stata.com/new-in-stata/inference-robust-to-weak-instruments/)
- [Stata weakrobust postestimation manual](https://www.stata.com/manuals/rivregresspostestimation.pdf)

### Wang–Zhang wild-bootstrap AR (2024)

This is a mandatory comparator only in the regime where its fixed-number-of-large-clusters asymptotic assumptions are defensible. Register the bootstrap weights, seed, replications, number of clusters, cluster-size regime, and weak/zero first-stage regime. Do not extrapolate its guarantees to many-small-cluster designs without separate evidence.

Reference: [Wang and Zhang (2024), Journal of Econometrics](https://doi.org/10.1016/j.jeconom.2024.105727).

### Ligtenberg cluster-jackknife AR and score theory

Treat this as the theory baseline behind CJAR/CJS, not as a separate method to rebrand. Reproduce its known-answer cases, plain/cross-fit variance choices and confidence-set topology before attempting extensions.

Reference: [Ligtenberg, cluster-jackknife AR and score tests](https://arxiv.org/abs/2306.08559).

## 3. New shared clustered-IV DGP

PR #9713 provides 13 seeded scenario presets with known `beta`, cluster-varying `pi_g), direct-effect vector `gamma_g), cluster and unit endogeneity, equicorrelated instruments, unequal cluster sizes, heteroskedasticity, zero/weak relevance and both within-cluster and cluster-level instrument assignment.

The export contract is blind by default:

- public CSV: IDs, instruments, treatment, outcome only;
- public manifest: layout and assignment-level metadata only;
- optional evaluator-only oracle JSON: seed, `beta`, `pi`, `gamma`, registered direct-effect support, cluster shocks and per-row truth.

The exporter writes only public artifacts by default. The oracle requires an explicit `--oracle-out` path and is rejected if it shares the public output directory. The benchmark run schema records separate hashes and requires a prediction-freeze receipt before oracle release.

Do not put the oracle fixture or oracle JSON into a model-fitting or baseline container. The four-row known-answer CSV is for parser/parity checks; it has only two clusters and is intentionally not used for the leave-two-cluster-out package comparison.

## 4. Registered DGP evaluation grid

Evaluate every applicable estimator on the same observed CSV rows, clusters, instruments, endogenous treatment and outer split. Required regimes:

- cluster counts: 5, 8, 10, 15, 20, 30, 50, 100; unequal sizes and repeated observations per cluster;
- instrument dimension: 1, 2, 5, 10, 25, 50; strong, weak, zero, cluster-heterogeneous and concentrated first stages;
- dependence: cluster random intercepts/slopes, unit endogeneity, temporal/state dependence, heteroskedastic cluster shocks, heavy tails and non-Gaussian shocks;
- direct effects: exact exclusion, inside support, at boundary, outside support, cluster-specific paths and incorrect control-to-target transport;
- assignment level: instrument variation within cluster versus instrument constant within each cluster;
- leverage: well spread instruments versus strength concentrated in one/few clusters;
- confidence-set cases: finite, empty, disconnected, one-sided/unbounded and full real line.

For nominal 5% tests, report empirical size with Monte Carlo intervals, confidence-set coverage, power over first-stage strength, false-narrow-set rate under zero relevance, topology frequencies and compute cost. A narrower set is not an improvement if size or coverage is wrong.

## 5. Acceptance and claim ceilings

A new bounded-direct-effect clustered method must improve power or set informativeness over CJAR/CJS and wild-bootstrap AR without losing size or coverage. It must maintain coverage for weak/zero first stages and in-bound direct effects; avoid spurious finite/narrow sets when relevance is absent; and preserve full confidence-set topology. If a regime does not pass these gates, mark it `NOT_VALIDATED`.

Keep evidence levels separate:

- deterministic algebra and known-answer tests;
- synthetic Monte Carlo size/coverage calibration;
- observational neural prediction;
- measured biological intervention;
- prospective model-selected experiment.

This repository currently has a DGP generator and adapters, not executed Monte Carlo or biological findings. Any superiority claim requires matched, reproducible estimator runs and independent evaluation.

## 6. Versioned environment registry and pretraining-exposure ledger

A public causal benchmark is not contamination-proof merely because its test split was held out at release time. Once public test examples, generator recipes or derived graphs are visible, later training runs may include them. Treat benchmark freshness and method exposure as explicit evaluation variables, not informal caveats.

The new registry contract records each pool's:
- immutable pool/release version and manifest digest;
- split (train, public validation, reserved validation, sealed evaluation, retired);
- source kind (synthetic SCM, semantic-operational, formula-grounded, biological recording);
- mechanism, noise, observation, intervention, context and outcome family IDs;
- generator name, code commit, config digest and seed commitment;
- visibility, pretraining permission and ground-truth access;
- record count, source version, URI and license.

Reserved/sealed pools must be non-public/restricted, disallow pretraining access and keep ground truth sealed. A planned pool has item_count=0 and must not be represented as qualified data. The example registry includes a planned DANDI pilot only to demonstrate this status; it does not claim any NWB recordings have been materialized or scored.

A method exposure disclosure records code/artifact/weight digests, pretraining cutoff, training sources and the level of direct/derived/family-level/possible/unknown exposure, plus per-pool support status and evidence. Unknown exposure must remain UNKNOWN; absence of a disclosure is not evidence of no overlap.

The rotation policy declares public-pool refresh cadence, maximum public validation age, retirement after exposure, minimum reserved fraction and whether exposure disclosure is mandatory. Benchmark reports must include the exact registry release ID and method-exposure declaration digest.

Schemas:
- \`CONNECTOME_BENCHMARK_ENVIRONMENT_REGISTRY.schema.json\`
- \`CONNECTOME_METHOD_EXPOSURE_DISCLOSURE.schema.json\`

Examples:
- \`CONNECTOME_BENCHMARK_ENVIRONMENT_REGISTRY.example.json\`
- \`CONNECTOME_METHOD_EXPOSURE_DISCLOSURE.example.json\`

These are contract/fixture additions only. The example registry is illustrative and does not claim a real CausalArena-equivalent benchmark release exists in this repository.

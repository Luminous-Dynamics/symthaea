# CMAQ-049 — Contextual paired Fourier-kernel discrepancy e-process

## Purpose

CMAQ-048 monitors a single stream against a frozen reference embedding. Many production comparisons are paired: candidate versus incumbent on the same prompt, two sensors on the same physical event, or two estimators on the same system state.

The null is conditional distribution equality, not simply equality of marginal output distributions. A pooled two-sample discrepancy can miss conditional differences that cancel across contexts.

CMAQ-049 applies the frozen CMAQ-048 Fourier map to both members of a pair, forms a context-blocked feature difference, then feeds that bounded vector into CMAQ-046's anytime-valid approachability e-process.

## Conditional null and finite-feature construction

Let Z_t be a pre-registered context/covariate stratum and let X_t,Y_t be paired responses. The target null is

    H0: Law(X_t | Z_t, F_(t-1)) = Law(Y_t | Z_t, F_(t-1))

For a fixed Fourier feature map phi from CMAQ-048, define

    r_t = 0.5 * e_(Z_t) ⊗ [phi(Y_t) - phi(X_t)].

Because ||phi(x)||_2=1,

    ||r_t||_2 <= 0.5 * (||phi(Y_t)|| + ||phi(X_t)||) <= 1.

Under conditional distribution equality within each registered context,

    E[r_t | Z_t, F_(t-1)] = 0.

The one-hot context block is critical: positive discrepancy in one context and negative discrepancy in another cannot cancel as it could in a pooled marginal feature vector.

**Context assignment must be predictable.** The context ID is derived only from covariates/metadata available before either response is observed. The inventory and mapping are frozen. Post-outcome stratification, selecting contexts after seeing X or Y, or dropping inconvenient contexts is outside the validity contract.

A frozen BoxTarget centered at zero (or a small independently justified tolerance envelope) defines the allowed conditional mean of the full context-block vector. CMAQ-046 creates a predictable OCO normal, support residual and log-domain betting process. Rejection uses the e-process Ville threshold, not a raw distance diagnostic.

This is a finite-feature conditional test. Unequal conditional distributions can share the same finite Fourier embedding; contexts can hide differences within a stratum; a finite feature bank is not universally characteristic for every realization.

## Implementation

The dependency-free Rust crate depends on CMAQ-046 and CMAQ-048 by local path. It provides:
- immutable Fourier map reused exactly from CMAQ-048;
- frozen finite context inventory and map/target/reference digests;
- one-hot context feature blocks;
- half-scaled paired feature difference with L2 bound one;
- exact monotone sequence ID and unique pair ID validation;
- duplicate pair, wrong sequence and unknown context fail closed;
- transactional update: pair ID is consumed only after the CMAQ-046 evidence update succeeds;
- maximum context count and total feature dimension bounds;
- no refitting of the feature map, context inventory, target or alpha during monitoring.

Pair IDs are deduplicated in-process. Durable exactly-once replay protection across process restarts still requires the external evidence ledger. SHA-256 references are shape-checked only; independent signature/authentication remains external.

## Tests

Deterministic tests cover:
- pair-difference L2 bound;
- identical pairs produce the zero vector and no wealth growth;
- swapping X and Y negates the entire contextual feature vector;
- opposing discrepancies in two contexts cancel in the pooled mean but not in the blocked vector;
- conditional mismatch with identical marginal output mixtures accumulates evidence;
- duplicate pair IDs and incorrect sequence are rejected without state mutation;
- unknown context and malformed pair identity fail closed;
- fixed map/context/target/reference bindings;
- context inventory and total feature dimensions are bounded.

Run:

    cargo +1.100.0 test --manifest-path research/cmaq-049/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-049/Cargo.toml --locked --all-targets -- -D warnings

## Paired marginal-cancellation probe

Run:

    cargo +1.100.0 run --release --manifest-path research/cmaq-049/Cargo.toml --locked --bin conditional_pair_probe -- 1000 2000 20261128 > cmaq-049-paired-results.csv

The probe compares:
- exact conditional-pair null: X and Y are independent samples from the same symmetric conditional distribution within each context;
- conditional alternative with equal marginal mixture: context A always pairs 0.2 with 0.8, context B always pairs 0.8 with 0.2. Across contexts, the marginal X and Y mixtures are exactly equal, but the conditional distributions differ.

The context-blocked monitor should detect the registered alternative; the null arm must retain full-horizon Type-I control. CI uses a small number of streams as a wiring smoke test only, not a calibrated false-alarm qualification.

## Full benchmark

Compare against:
- faithful SCPT (Zheng & Zhang, 17 Nov 2026) on paired covariate-conditional streams with its nearest-neighbor/kernel assumptions;
- unconditional/global paired kernel MMD betting without context blocks;
- pooled CMAQ-048 Fourier embedding without the one-hot context inventory;
- raw paired scalar/vector mean-difference e-processes;
- CMAQ-048 one-stream reference embedding monitor.

Register:
- number and construction of strata; context assignment digest and pre-response covariate-only rule;
- common-covariate pairing and pair identity truth;
- feature frequencies, bandwidth, map digest and input preprocessing;
- context-specific null distributions and conditional dependence;
- action/sensor/model build identities and common-cause blocks;
- target, alpha, horizon, reference provenance and evidence ledger semantics.

Primary alternatives:
- context-specific conditional shifts with identical marginal distributions;
- candidate regression isolated to a low-frequency or rare context;
- conditional variance/tail/skew changes;
- covariate-stratified calibration shifts;
- paired sensor bias with common-cause noise;
- context drift, missing pairs and misassigned strata;
- finite-feature null-space alternatives and within-stratum hidden alternatives.

Use at least 10,000 independent full-horizon conditional-pair null paths and 5,000 alternatives per primary cell. Report full-horizon ever-alarm probability with exact/Wilson intervals, unconditional detection and miss fraction, H+1-restricted delay, p50/p90/p95 delay, worst context and common-cause block, pair coverage, p50/p95/p99 ingestion latency, memory and feature/context scaling. Use CMAQ-044 paired effective-block inference.

## What differs from SCPT

SCPT's standard construction works with a measurable positive-definite characteristic kernel bounded in [0,1]. It estimates conditional RKHS mean embeddings with sequential k-nearest neighbors using only earlier pairs, then applies a covariate-indexed predictable betting fraction. Its main proof uses a conditional-independence structure: given the covariate sequence, pairs are mutually independent and each pair's conditional law depends on the sequence through its current covariate. The paper treats broader dependence structures in separate extensions.

CMAQ-049 deliberately takes a different engineering trade:
- no online nearest-neighbor index or growing conditional estimator;
- a pre-registered finite context partition and finite Fourier map;
- a fixed, explicitly dimension-bounded vector with one-hot context blocks;
- an OCO witness over the full blocked feature vector, with a simple support-box null envelope.

The trade is bounded state and an easier deterministic execution contract in exchange for approximation and discretization gaps. It is not automatically more powerful or more general than SCPT. On high-cardinality or continuous covariates, SCPT's local conditional estimator may preserve distinctions that our context partition loses. On low-cardinality, safety-critical modes, our fixed context inventory can be easier to audit and can report per-context feature contributions.

For fair comparison, run SCPT with its native sequential KNN and kernel pipeline on identical paired inputs and covariates. Also compare CMAQ-049 with a pooled finite-feature detector using no context blocks. Report compute and memory against pair count and covariate dimension, not only detection delay.

## External comparison and claim boundary

SCPT (Zheng & Zhang, 17 Nov 2026) tests conditional paired-sample equality and uses nearest-neighbor conditional kernel discrepancy plus test martingales. CMAQ-049 is a bounded finite Fourier-feature/one-hot-context conditional-mean implementation. It is not a reproduction of SCPT and does not inherit SCPT's method-specific results.

The guarantee applies only to the fixed registered Fourier map and context-blocked conditional-mean target. It depends on correct pair matching, predictable context assignment, truthful covariates/context IDs, complete context inventory, valid feature target and a valid conditional null. It does not authenticate device identity, prove physical co-location or establish that paired samples refer to the same underlying event.

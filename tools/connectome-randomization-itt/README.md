# Exact blocked randomization oracle for optogenetic assignment ITT

Standalone, dependency-free Rust 2024 implementation of finite-sample exact Fisher randomization inference for a prespecified binary optogenetic assignment design.

## Why this is a separate causal layer

Randomized light assignment has a causal effect on measured outcomes whether the pathway is opsin-mediated, thermal, optical, extrasynaptic, or another consequence of assignment. This assignment effect (intention-to-treat / reduced form) can be tested **without the instrumental-variable exclusion restriction**.

The effect of source-neuron activity on downstream neural outcomes is a different estimand. That mediated structural effect requires additional assumptions and is handled separately by the weak-IV cluster methods and bounded-direct-effect pAR work.

This oracle prevents an artifact path from being reported as source-neuron mediation merely because a randomized laser assignment predicts a downstream response.

## Implemented randomization designs

One row represents one randomized assignment unit and one predeclared scalar outcome summary. Every row carries a unique unit ID, independent cluster ID (normally animal or animal-session), block/stratum ID, binary assignment receipt, and outcome.

Two modes are supported:

- **Within-cluster blocked:** multiple randomized units occur within each independent animal/session cluster. Every stratum is nested inside one cluster and contains both treatment and control units. The test computes block contrasts, combines blocks within each cluster using block-size weights, then gives every independent cluster equal weight.
- **Cluster-level blocked:** one randomized assignment unit represents one independent cluster, and clusters are randomized within strata. The test computes block contrasts and combines them with block-size weights.

The only implemented assignment mechanism is **uniform complete randomization within each registered stratum**, conditional on the observed treatment count in that stratum. Each legal assignment is equally likely.

## Sharp-null test

The candidate effect `tau` specifies a sharp constant-additive null:

```
Y_i(1) = Y_i(0) + tau
```

Under that sharp null, the no-treatment potential outcome is known for every unit:

```
Y_i(0) = Y_i(observed) - tau * assigned_treatment_i
```

The oracle enumerates every allocation preserving each stratum's observed treatment count, recomputes the predeclared cluster-macro contrast, and returns the exact two-sided tail proportion, with ties included. Because the full allocation space is enumerated, there is no plus-one Monte Carlo correction.

For a constant additive effect confidence set, callers may invert the test over candidate tau values. This crate does not yet implement continuous or grid-free confidence-set inversion. It must not claim a confidence interval from one p-value.

## Fail-closed design requirements

- Exact uniform complete randomization within each registered stratum.
- Both treatment and control assignments in every stratum.
- Within-cluster mode: each stratum is nested inside one cluster.
- Cluster-level mode: exactly one assignment unit per independent cluster.
- At least two independent clusters.
- A prespecified outcome-summary definition and SHA-256 digests for the protocol, allocation receipt and outcome-summary artifact.
- Explicit design assertions for no between-assignment-unit interference and controlled carryover.
- A caller-supplied maximum allocation-space size. Exceeding the cap fails closed; no approximate fallback is hidden behind the exact API.

## Scientific claim ceiling

The test is exact only for the stated sharp constant-additive assignment-effect null under the actual registered assignment mechanism. It is not a general test of a weak/average-effect null.

A rejected sharp null means the assigned stimulation schedule changed the predeclared measured outcome. It does **not** identify the mechanism: opsin activation, direct illumination, heating, imaging artifacts, extrasynaptic signalling and network-mediated effects all remain within the assignment estimand.

It does not establish a direct synaptic edge, source-neuron mediation, or a causal connectome. Between-unit interference and carryover invalidate the default potential-outcome mapping unless the experimental unit and assignment design are defined to absorb them.

The scalar outcome summary and contrast weighting must be registered before looking at the target outcomes. Neurons or time points inside an assignment unit do not become independent randomized units.

## Synthetic known answers

The fixture contains four analytically enumerable examples:

1. Two animals with four randomized trials each and one 2-of-4 block per animal. The exact allocation space contains (6 \times 6 = 36) assignments. The observed equal-animal-weight contrast is -0.5; 28 assignments are at least as extreme, so the exact two-sided p-value is (28/36 = 7/9).
2. Four independently randomized clusters in one 2-of-4 block with outcomes [9, 8, 1, 0] and the top two assigned. There are six assignments, the observed contrast is 8, two assignments are at least as extreme, and the exact two-sided p-value is (2/6 = 1/3). Testing a candidate constant effect of 3 subtracts 3 from the treated outcomes and gives the adjusted contrast 5 with the same exact p-value.

A third example stipulates a direct light/thermal artifact with no opsin-mediated effect; the assignment ITT includes it as part of the total stimulation-assignment effect and does not label it neural mediation. All examples are synthetic known answers and carry the claim ceiling `FINITE_SAMPLE_RANDOMIZATION_SHARP_NULL_ONLY`. They contain no biological observations.

## How it compares with other causal methods

This oracle complements, rather than replaces, the clustered weak-IV ecosystem:

- **CRAN clusterIV 0.2.0 (2026-10-01):** CJIVE, cluster-jackknife Anderson–Rubin (CJAR), and cluster-jackknife score (CJS) inference for one endogenous regressor and one-way clustered errors; multiple instruments, controls, weights and high-dimensional fixed effects; plain and leave-two-cluster-out cross-fit variance; analytic polynomial confidence-set inversion. External source: https://github.com/atal-kat/Clustered-Estimation-and-Inference; package docs: https://www.r-pkg.org/pkg/clusterIV.
- **Stata 19 `estat weakrobust`:** Anderson–Rubin and conditional likelihood-ratio tests with robust, cluster-robust and HAC variance choices; supports nonstandard confidence-set topologies. Documentation: https://www.stata.com/new-in-stata/inference-robust-to-weak-instruments/.
- **Wang and Zhang (2024):** wild-bootstrap IV inference with weak instruments and few large clusters, including a weak-IV-robust AR procedure. https://doi.org/10.1016/j.jeconom.2024.105727.
- **Ligtenberg (2025):** cluster-jackknife AR and score inference with many/weak instruments, now implemented by clusterIV.

The intended combined workflow is:
1. Exact randomization ITT for the total effect of assigned stimulation.
2. CJAR/CJS (and other weak-IV cluster baselines) for structural source-activity effects.
3. Bounded-direct-effect pAR sensitivity to make exclusion assumptions visible.

The ITT and structural effects must never be described as interchangeable.

## Run

From the repository root:

```sh
cargo test --locked --manifest-path tools/connectome-randomization-itt/Cargo.toml
```

Twenty authored Rust unit tests cover the two randomization modes, exact known answers, blocked allocation cardinality, sharp-null adjustment, cluster/stratum integrity, direct-artifact semantics, provenance, carryover/interference assertions and fail-closed allocation limits. The tests have not been compiled or executed in the current authoring environment. No third-party dependencies. Isolated Cargo workspace. Repository license: AGPL-3.0-only.

# SYM-ARCH-002A2 Hierarchical Statistics and Prospective Power v1

**Status:** statistical-infrastructure tranche  
**Scientific claim status:** none  
**Parent plan:** issue #55  
**Stacked dependency:** PR #57 (`research/sym-arch-002a-core-v1`)  

## Purpose

SYM-ARCH-002A2 extends the experimental core with two pieces needed before a future CONFIRM run can be sized or interpreted responsibly:

1. hierarchical candidate-minus-control uncertainty with **generated environments as the independent unit**;
2. prospective sample-size planning from DEV variability without using the observed DEV mean as the assumed confirmatory effect.

This tranche does not choose a CONFIRM sample size because no qualifying DEV campaign is part of this PR. It supplies the instrument that will make that choice later from a frozen DEV artifact and a separately frozen planning effect.

## Hierarchical result schema

`NestedEnvironmentResult` contains:

- one unique environment digest;
- one or more `PairedRunResult` entries;
- a unique nuisance-run digest for each nested representation/learner/stream realization;
- paired candidate/control outcomes from that same nuisance realization.

Validation rejects:

- malformed environment or nuisance digests;
- duplicate environment identities;
- duplicate nuisance-run identities within an environment;
- empty nested-run sets;
- non-finite outcomes.

Nested runs reduce within-environment uncertainty. They do **not** increase the number of independent environments.

## Hierarchical bootstrap

`hierarchical_environment_delta_percentile` performs a two-level paired bootstrap:

1. sample environments with replacement;
2. inside each sampled environment, sample paired nuisance runs with replacement;
3. compute one mean delta for each sampled environment;
4. average environment means with **equal environment weight**;
5. repeat and report a deterministic 95% percentile interval.

An environment with ten nuisance runs therefore does not receive ten times the scientific weight of an environment with one nuisance run.

v1 deliberately calls this a **hierarchical percentile interval**, not BCa. The existing one-level BCa implementation is not silently relabeled as a nested-data method.

## Prospective power planning

`prospective_power_from_dev` is allowed only with a valid exploratory `DEV` manifest. The number of DEV environment results must match the frozen environment-seed manifest.

The planner uses DEV for empirical heterogeneity/noise, but **not for the assumed effect size**. This prevents a tuned or unusually favorable DEV point estimate from becoming a winner's-curse input to CONFIRM sizing.

The preregistered planning configuration includes:

- candidate independent-environment counts;
- nested paired runs planned per future environment;
- Monte Carlo trial count;
- BCa resamples across future environment aggregates;
- target power;
- SESOI;
- a separately frozen `planning_effect`;
- a residual-variance scale;
- gain/regression direction;
- RNG seed.

For a gain claim, `planning_effect` must be strictly greater than `+SESOI`. For a regression-detection plan it must be strictly below `-SESOI`.

### DEV winner-bias guard

Let `d_dev` be the equal-environment DEV mean delta. A sampled DEV run contributes a residual:

`residual = run_delta - d_dev`

The simulated future delta is:

`planning_effect + residual_scale * residual`

Thus DEV contributes the observed environment/nuisance variability while the future-study center remains the independently frozen planning effect.

`residual_scale >= 1` supports conservative sensitivity analysis. A value below one is rejected because it would make the observed DEV variability artificially easier.

## Practical-effect-aligned power criterion

A simulated future study counts as successful only when its environment-level 95% BCa interval clears the same SESOI practical-effect gate used by 002A1.

Power is therefore not defined as merely obtaining `p < 0.05`. It is the probability that the future study will support the **predeclared practically meaningful claim**.

For each candidate environment count, A2 reports:

- Monte Carlo success count;
- point estimate of power;
- Wilson 95% interval for that estimated power.

The selected `minimum_environments`, if any, is the first tested count whose **lower Wilson power bound** clears the target **and remains above target for every larger tested count**. This prevents a single noisy threshold crossing from setting CONFIRM size.

If no tested count satisfies this sustained conservative rule, `minimum_environments = None`; the correct response is to extend the planning grid or reconsider the design, not to weaken the target after looking at results.

## Provenance binding

The power plan records:

- digest of the exact DEV experiment manifest;
- digest over the DEV nested outcomes and complete power configuration.

Changing DEV data, planning effect, SESOI, variance inflation, candidate counts, simulation settings, or seed therefore changes the planning-input digest.

The chosen CONFIRM sample size should later be frozen by referencing this digest before CONFIRM outcomes are observed.

## Acceptance gate

The exact stacked PR head should pass:

1. rustfmt on A2 Rust paths;
2. focused `experiment_statistics` tests;
3. `cargo check -p symthaea-psych-bench --lib`;
4. equal-environment-weight reference test;
5. deterministic hierarchical-bootstrap test;
6. duplicate environment/nuisance identity rejection;
7. DEV-only planning rejection for non-DEV manifests;
8. manifest/result environment-count binding;
9. planning-effect-outside-SESOI validation;
10. deterministic power-plan/provenance-digest test;
11. Wilson-bounded power reporting and sustained-threshold selection.

## Explicit non-claims and limitations

A2 v1 does **not** claim:

- that percentile hierarchical intervals are optimal for every future metric;
- that the empirical DEV variance distribution perfectly represents CONFIRM;
- that a planning effect should be chosen from DEV performance;
- that any particular number of environments is currently required;
- that 002 has demonstrated an architecture advantage;
- that multiple-comparison correction, resource Pareto analysis, or procedural benchmark validity is complete.

The `planning_effect` must be justified and frozen outside this code path. Sensitivity analysis should run multiple preregistered `residual_scale >= 1` conditions where DEV is small or heterogeneous.

## Merge boundary

This PR is statistical infrastructure and may merge on correctness after its exact-head gate passes. It should remain stacked on #57 until #57 merges; after that, rebase/retarget to `main` without changing the statistical contract.

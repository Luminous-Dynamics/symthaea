# Ramanujan Protocol v2: Preregistered Evaluation Protocol

**Status:** Proposed protocol; no benchmark results claimed  
**Scope:** Autonomous invariant discovery, independent numerical generalization, and evidence provenance  
**Related work:** [Issue #6862](https://github.com/Luminous-Dynamics/symthaea/issues/6862), [PR #7372](https://github.com/Luminous-Dynamics/symthaea/pull/7372)

## 1. Claim to test

The primary claim is deliberately narrower than “Symthaea discovers better science”:

> Under equal search and evaluation budgets, Symthaea's cold-discovery mode can recover nontrivial invariants that generalize to unseen initial conditions at a rate that is measurably different from primed discovery, without raising false-discovery rates or relying on prior templates.

This is initially a **within-Symthaea ablation**, not evidence of state-of-the-art performance. A later external comparison should include a well-maintained symbolic-regression baseline such as PySR and, where the task interface permits, methods from SRBench.

## 2. Preregistered hypotheses

- **H1 — Recovery:** Cold discovery recovers ground-truth invariants on held-out conditions more often than a budget-matched random-expression baseline.
- **H2 — Prior sensitivity:** The recovery/generalization gap between cold and primed modes can be quantified, rather than inferred from a few handpicked examples.
- **H3 — Calibration:** Low training fitness predicts held-out conservation only to the extent measured on the sealed test conditions; the benchmark reports false discoveries explicitly.
- **H4 — Cost:** Any gain is reported alongside actual evaluator calls, integration work, and wall-clock cost. Population size times generation count alone is not accepted as proof of equal compute.

A hypothesis is allowed to fail. No post-hoc changes to thresholds, systems, or scoring are allowed in the confirmatory run.

## 3. Benchmark stages

### Stage 0 — Instrumentation and contract tests

Before interpreting benchmark results, expose deterministic counters for at least:
- candidate fitness evaluations;
- **total RHS evaluations**, split into integration calls and candidate-scoring calls, plus attempted/completed numerical integration steps;
- candidate rejection reasons for invalidity, excessive complexity, degeneracy, duplicate formulas, and gradient policy;
- completed and failed trajectories, state updates, and search/holdout wall time.

Counters must include work already spent by a run that later fails. A run rejected at preflight must report zero integration work and an explicit preflight rejection; it must not be conflated with a zero-cost successful search.

Verify that counters count rejected work too. If exact counters are not available, the run may be reported as an exploratory pilot, but not as an equal-computation comparison.

Keep integrity and scientific tests separate: a valid receipt establishes internal consistency of declared evidence, not that the equation is correct.

### Stage 1 — Exploratory synthetic pilot

Use a small fixed suite spanning different invariant structures, for example: harmonic oscillator energy, coupled linear oscillators, pendulum energy, conservative Duffing energy, Kepler energy/angular momentum, and a Lotka–Volterra first integral. Publish the exact system definitions, parameter ranges, units, expected invariants, and known limitations before running the pilot.

Use at least 10 independent search seeds per system and condition. Treat these results as exploratory only: they may expose bugs and refine the harness, but must not be mixed into the confirmatory estimates.

#### Executable single-system instrumentation pilot

An exploratory runner is registered as `ramanujan_protocol_v2_pilot` in `symthaea-core`. From a clean, committed repository root, the intended invocation is:

```text
cargo run --locked -p symthaea-core --example ramanujan_protocol_v2_pilot -- target/ramanujan-protocol-v2-pilot.jsonl 10 42
```

The arguments are a **new output path**, replicate count, and first discovery seed. The runner refuses to overwrite an existing output, rejects dirty/untracked source trees, and emits one hashed JSONL record for each `(seed, condition)` pair plus a final batch manifest listing all per-run digests. The initial instrumentation batch uses the dimensionless one-dimensional harmonic oscillator, 80 candidates, 8 generations, a maximum expression depth of 4, maximum complexity 18, three search trajectories, and three independent holdout trajectories per returned candidate. Conditions are cold, primed, and random-expression control; each replicate shares a discovery seed label and a distinct, deterministic holdout seed namespace.

Each record binds the git revision, Rust and Cargo version strings, available host parallelism, replicate order index, protocol-file digest, Cargo.lock digest, both workspace manifest digests, system-definition digest, candidate formulas, search work counters, per-candidate holdout counters, serialized receipts, oracle classifications, and a content digest. Condition order is counterbalanced deterministically through a three-order Latin square (cold/primed/random; primed/random/cold; random/cold/primed), and the batch manifest records the order used for every replicate. RHS accounting is split into integration RHS calls and candidate-scoring RHS calls, with a total that must equal their sum. This is important because Lie-derivative fitness invokes the system RHS during scoring in addition to the calls made by the integrator. The holdout work totals include both the nominal and half-step stability assessment, while each receipt retains its own separate counter set. The oracle is a tolerance-qualified coefficient-map check for the degree-2 family (a(x^2+v^2)+b): it requires |a|>1e-10, uses a relative coefficient tolerance of 1e-10, and normalizes coefficients below 1e-12 to zero. Formulas outside the supported grammar fail oracle classification rather than being guessed equivalent. Each run has a paired replicate ID shared across cold, primed, and random conditions, while discovery and holdout seeds are distinct within each record. Holdout acceptance and oracle-equivalence are recorded separately.

This runner is an **instrumentation smoke pilot**, not the full Stage 1 system suite or Stage 2 confirmatory test. It now repeats each returned candidate's independent holdout at half the fixed RK4 step size, using the same holdout seed stream, and reports the relative difference in maximum Lie-derivative variance. The initial exploratory stability rule combines a 10% relative tolerance with an absolute variance-difference tolerance of 1e-12 near zero; both are recorded in every candidate record. These are exploratory constants that must be sensitivity-tested and frozen before confirmatory runs, not adjusted after seeing confirmatory results. Each step size gets a separately context-bound receipt. The pilot still covers only one system and must not be used to claim cold discovery superiority. It is designed to make work counts, failed receipts, numerical-step stability, and every returned candidate auditable before the larger suite is frozen.

**Execution caveat:** this invocation has not been run in the present environment. A preceding CI run reported that a `cargo run --locked` lane could not update `Cargo.lock`; preserve `--locked` and resolve that lockfile drift rather than bypassing it. No pilot result should be interpreted until the lockfile state is fixed and the new example and receipt changes pass CI.

### Stage 2 — Confirmatory suite

Freeze the systems, parameter ranges, scoring thresholds, implementation commit, compiler/toolchain, and budgets before running. Use 30 independent search seeds per system and condition as a reproducibility target consistent with SRBench's current methodology. Pair cold and primed trials by system and search-seed label; use separately generated holdout seeds, with no seed reuse between search and holdout.

Recommended initial suite: at least eight systems, including systems with known invariants and systems where the expected outcome is no recoverable nontrivial invariant under the allowed expression grammar. The negative controls matter: a method that always emits plausible-looking formulas must not score well.

## 4. Conditions and controls

Run these conditions with the same expression grammar, fitness definition, maximum work budget, and evaluation environment:

1. **Cold:** no built-in invariant templates, no caller templates, and cold-mode prior settings sanitized as recorded in the receipt.
2. **Primed:** normal built-in templates; explicitly record every enabled prior and seed template.
3. **Random-expression baseline:** sample fresh expressions from the same allowed grammar under the same candidate-evaluation budget, without evolutionary selection, mutation, or crossover. Apply the same in-sample scoring and final candidate shortlist to all three conditions. This estimates how much apparent success arises from the grammar and acceptance test alone.

For Stage 2, add at least one external symbolic-regression baseline if its problem interface can be made equivalent. State any interface mismatch rather than silently treating a non-equivalent baseline as comparable.

## 5. Train/holdout separation

- Seal holdout initial conditions and parameter regimes from discovery and threshold tuning.
- Include both interpolation-like condition shifts and harder extrapolative regimes.
- Evaluate the final frozen candidate list using an evaluator independent of the search's ranking loop.
- Re-evaluate accepted candidates with a second integration step size (for example, halved step size) and check numerical stability of conclusions.
- Exclude constants, functionally constant expressions, malformed formulas, non-finite values, and candidates with insufficient gradient informativeness from successful recovery counts. Record each rejection reason.
- Freeze the gradient-informativeness policy in each receipt: the squared-gradient floor is `0.01` (equivalent to a gradient magnitude at least 10% of that orbit's peak), and the minimum informative-sample fraction is `0.5` on every scored holdout orbit. Both values must be serialized and bound into the context and receipt digests; the independent verifier rejects substituted policy values. Tune or sensitivity-test alternatives only on exploratory/development runs, not the sealed confirmatory holdout.
- Score a system as recovered only when an accepted candidate meets the predeclared symbolic-equivalence or domain-appropriate oracle criterion **and** passes the frozen holdout rule. Numerical near-conservation by itself is not exact symbolic recovery.

The independent holdout seed must differ from the discovery seed. The evaluation harness must also demonstrate that changing a candidate, input condition, or system identity invalidates the associated assessment context and receipt.

## 6. Metrics

Report per system and aggregated, with confidence intervals:

- exact/form-equivalent recovery rate;
- holdout conservation/generalization pass rate;
- false discovery rate, including negative controls;
- held-out invariant residual distribution and sensitivity to the smaller integration step;
- candidate complexity and duplicate/degenerate candidate rates;
- complete versus incomplete holdout coverage;
- fitness-evaluation count, RHS calls, integration steps, wall time, and peak memory where available;
- receipt verification rate and counts of provenance/contract rejection reasons.

Do not report only the best seed, best formula, or mean training score. Publish every attempted run, including failed searches and incomplete validation.

## 7. Preregistered statistical analysis

The following analysis is part of the protocol, not a choice to make after viewing results.

### Primary outcome and comparison

- The unit of analysis is one attempted search run, not each candidate formula. A run is a **recovery success** only when at least one candidate meets the frozen symbolic-equivalence/domain-oracle criterion and passes the independent holdout protocol, including the second-step-size stability check. Empty searches, invalid outputs, and incomplete holdout coverage are failures for the primary recovery outcome and remain in the denominator.
- The primary confirmatory contrast is cold discovery versus the random-expression control. The secondary contrast is cold versus primed discovery.
- Pair runs by system and discovery-seed label. Use the same sealed holdout initial-condition set for the modes within each pair, generated from a dedicated holdout seed namespace that is disjoint from discovery seeds and all tuning runs. This common holdout reduces condition-to-condition variance without allowing holdout results to influence search.
- Report the per-system paired recovery difference and the macro-average across the frozen systems, so a system with many candidates or unusually easy equations cannot dominate the result. Use a 95% stratified bootstrap interval by resampling paired seeds within each system for the fixed-suite estimand. Because a small number of systems limits inference to new system families, any broader generalization claim must be explicitly separated and qualified.
- State the numerator and denominator for every rate. For candidate-level false-discovery rate, count oracle-invalid candidates that pass the numerical holdout rule divided by all candidates that pass the numerical rule. If no candidate passes, report the rate as undefined with a zero accepted-candidate count, not as a measured 0% false-discovery rate.
- Freeze a separate per-run false-discovery guardrail: a run is positive if at least one candidate passes numerical holdout but fails the frozen oracle. Cold is non-inferior to primed only if the upper end of the one-sided 95% stratified paired-bootstrap interval for the cold-minus-primed per-run false-discovery risk is at most **+5 percentage points**. Report candidate-level false-discovery rate as a secondary metric. This margin is a protocol choice, not a universal scientific constant; it must stay frozen once confirmatory runs begin.

### Multiplicity and decision rule

- Use the frozen 30 paired search-seed labels per system for confirmatory estimates. Exploratory pilot runs and any runs used to change the harness, grammar, or thresholds are excluded from confirmatory estimates.
- The primary comparison uses a two-sided 95% interval for the paired macro-average difference. Support superiority of cold over random only if the interval's lower bound is greater than zero and the frozen false-discovery guardrail is not violated. Report effect size in absolute percentage points as well as recovery ratios.
- Treat per-system tests, alternative integration settings, and additional metrics as secondary. Apply Holm correction across families of per-system confirmatory claims; label uncorrected exploratory findings as exploratory.
- Report compute and recovery jointly. Publish the recovery difference beside candidate fitness evaluations, RHS calls, integration steps, expression-evaluation work where instrumented, wall time, and peak memory. A recovery-rate gain with substantially greater work is not an efficiency win; do not extrapolate beyond the budgets actually tested.
- Keep all attempted runs in the data, including crashes, timeouts, invalid receipts, and missing outputs. Distinguish protocol failure, infrastructure failure, scientific non-recovery, and unknown/incomplete outcomes; never drop these rows silently.

## 8. Immutable run artifact

Store one machine-readable record per attempted run with:
- schema version, repository commit, toolchain/platform identity, benchmark suite version;
- system ID and sealed system-definition digest;
- mode, search seed, holdout seed, effective configuration snapshot and digest;
- candidate formula(s), resource counters, integration parameters and frozen threshold;
- all requested/checked trajectory counts, metrics, terminal result and failure code;
- receipt and assessment-context digests; elapsed time and execution outcome.

Keep raw run records append-only. Derive aggregate tables from those records with a separate script and publish its revision. A queued workflow, a skipped benchmark job, or a passing receipt-integrity check is not a scientific benchmark pass.

## 9. Decision rule

The initial pilot is successful as infrastructure when all conditions are explicit, the counters are internally consistent, failed outcomes remain in the dataset, and a verifier can independently recompute the records. It is **not** successful merely because a candidate passes a numerical threshold.

The confirmatory result supports H1/H2 only if the predeclared comparisons improve on the stated metrics under matched budgets, with uncertainty reported across systems and seeds. Otherwise publish the null or negative result and identify which bottleneck failed: search, representation, experiment/condition selection, fitness, numerical integration, or symbolic verification.

## 10. External benchmark grounding

- **SRBench 2025** reports a living, unified comparison of 25 symbolic-regression methods, using standardized containers, hyperparameter tuning, and 30 independent runs per dataset: [SRBench repository](https://github.com/cavalab/srbench).
- **Science-Gym** evaluates autonomous discovery end-to-end, combining data collection, experimental design, and equation discovery across seven simulated physics/epidemiology environments. This is important because a fixed-data symbolic-regression test does not evaluate experiment selection: [Cerrato et al., *Machine Learning* (published 7 January 2026)](https://doi.org/10.1007/s10994-025-06914-x).
- **SRBench++** argues that expression size and exactness alone do not adequately capture scientific interpretability; it adds expert-oriented interpretation dimensions: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/).
- **MechHypoBench** (preprint posted 4 October 2026) evaluates whether AI-generated mechanistic hypotheses imply correct consequences under withheld conditions, using paper-derived mechanisms across 14 fields and datasets containing 17.98 million records. This is a new preprint and should be treated as an emerging design reference, not a settled standard: [arXiv:2610.05197](https://arxiv.org/abs/2610.05197).

The distinctive longer-term question for Symthaea is whether its representation and adaptive condition selection improve reliable discovery **per equal compute and observation budget**, rather than merely increasing the number of generated equations.

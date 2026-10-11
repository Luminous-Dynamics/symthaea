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
- RHS evaluations and numerical integration steps;
- candidates rejected for invalidity, degeneracy, or low gradient informativeness;
- total search and holdout wall time.

Verify that counters count rejected work too. If exact counters are not available, the run may be reported as an exploratory pilot, but not as an equal-computation comparison.

Keep integrity and scientific tests separate: a valid receipt establishes internal consistency of declared evidence, not that the equation is correct.

### Stage 1 — Exploratory synthetic pilot

Use a small fixed suite spanning different invariant structures, for example: harmonic oscillator energy, coupled linear oscillators, pendulum energy, conservative Duffing energy, Kepler energy/angular momentum, and a Lotka–Volterra first integral. Publish the exact system definitions, parameter ranges, units, expected invariants, and known limitations before running the pilot.

Use at least 10 independent search seeds per system and condition. Treat these results as exploratory only: they may expose bugs and refine the harness, but must not be mixed into the confirmatory estimates.

### Stage 2 — Confirmatory suite

Freeze the systems, parameter ranges, scoring thresholds, implementation commit, compiler/toolchain, and budgets before running. Use 30 independent search seeds per system and condition as a reproducibility target consistent with SRBench's current methodology. Pair cold and primed trials by system and search-seed label; use separately generated holdout seeds, with no seed reuse between search and holdout.

Recommended initial suite: at least eight systems, including systems with known invariants and systems where the expected outcome is no recoverable nontrivial invariant under the allowed expression grammar. The negative controls matter: a method that always emits plausible-looking formulas must not score well.

## 4. Conditions and controls

Run these conditions with the same expression grammar, fitness definition, maximum work budget, and evaluation environment:

1. **Cold:** no built-in invariant templates, no caller templates, and cold-mode prior settings sanitized as recorded in the receipt.
2. **Primed:** normal built-in templates; explicitly record every enabled prior and seed template.
3. **Random-expression baseline:** sample from the same allowed grammar under the same candidate-evaluation budget, without selection pressure. This estimates how much apparent success arises from the grammar and acceptance test alone.

For Stage 2, add at least one external symbolic-regression baseline if its problem interface can be made equivalent. State any interface mismatch rather than silently treating a non-equivalent baseline as comparable.

## 5. Train/holdout separation

- Seal holdout initial conditions and parameter regimes from discovery and threshold tuning.
- Include both interpolation-like condition shifts and harder extrapolative regimes.
- Evaluate the final frozen candidate list using an evaluator independent of the search's ranking loop.
- Re-evaluate accepted candidates with a second integration step size (for example, halved step size) and check numerical stability of conclusions.
- Exclude constants, functionally constant expressions, malformed formulas, non-finite values, and candidates with insufficient gradient informativeness from successful recovery counts. Record each rejection reason.
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

## 7. Immutable run artifact

Store one machine-readable record per attempted run with:
- schema version, repository commit, toolchain/platform identity, benchmark suite version;
- system ID and sealed system-definition digest;
- mode, search seed, holdout seed, effective configuration snapshot and digest;
- candidate formula(s), resource counters, integration parameters and frozen threshold;
- all requested/checked trajectory counts, metrics, terminal result and failure code;
- receipt and assessment-context digests; elapsed time and execution outcome.

Keep raw run records append-only. Derive aggregate tables from those records with a separate script and publish its revision. A queued workflow, a skipped benchmark job, or a passing receipt-integrity check is not a scientific benchmark pass.

## 8. Decision rule

The initial pilot is successful as infrastructure when all conditions are explicit, the counters are internally consistent, failed outcomes remain in the dataset, and a verifier can independently recompute the records. It is **not** successful merely because a candidate passes a numerical threshold.

The confirmatory result supports H1/H2 only if the predeclared comparisons improve on the stated metrics under matched budgets, with uncertainty reported across systems and seeds. Otherwise publish the null or negative result and identify which bottleneck failed: search, representation, experiment/condition selection, fitness, numerical integration, or symbolic verification.

## 9. External benchmark grounding

- **SRBench 2025** reports a living, unified comparison of 25 symbolic-regression methods, using standardized containers, hyperparameter tuning, and 30 independent runs per dataset: [SRBench repository](https://github.com/cavalab/srbench).
- **Science-Gym** evaluates autonomous discovery end-to-end, combining data collection, experimental design, and equation discovery across seven simulated physics/epidemiology environments. This is important because a fixed-data symbolic-regression test does not evaluate experiment selection: [Cerrato et al., *Machine Learning* (published 7 January 2026)](https://doi.org/10.1007/s10994-025-06914-x).
- **SRBench++** argues that expression size and exactness alone do not adequately capture scientific interpretability; it adds expert-oriented interpretation dimensions: [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12321164/).

The distinctive longer-term question for Symthaea is whether its representation and adaptive condition selection improve reliable discovery **per equal compute and observation budget**, rather than merely increasing the number of generated equations.

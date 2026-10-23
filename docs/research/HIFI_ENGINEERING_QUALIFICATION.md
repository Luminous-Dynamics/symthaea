# HiFi Engineering: Qualification Contract and Paradigm Shifts

**Status:** Research/benchmark proposal, 2026-10-17  
**Scope of this tranche:** Harden the existing engineering benchmark's numeric acceptance predicate; define the next independent qualification gates.  
**Authority:** This document does not grant engineering, deployment, or safety authority.

## Executive thesis

High-fidelity engineering should not be measured by how realistic a simulation looks, how many solvers are integrated, or how low a single average error happens to be. The core question is:

> For a declared engineering decision, what evidence makes this result credible, what could falsify it, and what uncertainty remains?

A result is not evidence-qualified merely because a solver converged. A model can be numerically correct and physically inapplicable; several solvers can agree because they share assumptions; a low average error can hide dangerous errors near a constraint boundary.

## Compose with existing owners

Do not create a parallel universal VVUQ, provenance, authority, or experiment-selection system.

- [SE-VV #3697](https://github.com/Luminous-Dynamics/symthaea/issues/3697) owns the lifecycle program for verification, validation, credibility, reliability/maintainability and human-systems integration.
- [Systems Engineering #3677](https://github.com/Luminous-Dynamics/symthaea/issues/3677) owns the semantic graph and the evidence-bounded architecture.
- [Decision-impact experiment benchmark #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803) owns comparison of the value of the next evaluation.
- [Elmer FEM adapter #5634](https://github.com/Luminous-Dynamics/symthaea/issues/5634) owns the generic Elmer solver boundary.
- [Multi-fidelity materials uncertainty #6225](https://github.com/Luminous-Dynamics/symthaea/issues/6225) owns its materials-domain specialization.

This file is a focused engineering-curriculum contract and research bridge, not a competing owner.

## Concrete defect fixed in this tranche

The prior `BenchmarkValidation::is_acceptable` predicate accepted a result when `converged` was true and the caller-supplied `relative_error` was below tolerance. It did not check that the reported error agreed with `ground_truth` and `generated_value`; a false or negative reported error could therefore conceal a large mismatch.

The current patch:
- recomputes relative error from the values being compared;
- rejects non-finite values, invalid tolerances, arithmetic overflow, and unsupported zero-reference relative comparisons;
- permits zero reference only when the generated value is exactly zero;
- rejects empty sample/metric identity, unconverged results, negative reported error, and reported/computed disagreement beyond a small floating-point rounding allowance;
- adds adversarial unit-test declarations for these cases.

**Claim boundary:** This protects one numeric acceptance boundary. It does not prove the ground truth is authoritative, the metric units agree, a solver model is applicable, the validation data are independent, or the design is physically safe. The reported-error agreement check uses an eight-`f64::EPSILON` scale allowance; that implementation choice itself should remain explicit and tested.

## Paradigm shifts

### 1. Qualify a model for a use, not in the abstract

Replace a global `model = validated` label with a claim bound to:
- intended decision and operating envelope;
- target quantities, units and acceptance limits;
- solver, model, configuration and geometry revisions;
- reference-data identity and provenance;
- verification evidence, validation evidence and uncertainty limits;
- allowed extrapolation, known exclusions and invalidation conditions.

The same model can be credible for screening and unqualified for a safety-critical release.

### 2. Make independent falsification the default

Every benchmark tier should separate the implementation under test from its expected answer wherever practical. Use analytic solutions, manufactured solutions, separately implemented small oracles, metamorphic properties and protected negative controls. Do not generate both the fixture and expected result from the same production helper.

### 3. Treat disagreement as information

A surrogate-to-CFD, CFD-to-FEM, solver-to-solver, or simulation-to-measurement discrepancy must be recorded rather than hidden by averaging. Check condition, mesh, constitutive law, boundary conditions, units, geometry and shared assumptions before attributing the difference. Higher nominal fidelity is not automatically infallible.

### 4. Optimize decision reliability, not just mean error

Measure false-feasible decisions, constraint margin, worst-case regret, decision changes under plausible perturbations, evaluation cost and missing-evidence rate. When defensible probability distributions exist, report expected decision value; otherwise retain interval or set-based robust bounds instead of inventing probabilities.

### 5. Keep uncertainty sources distinct

Where relevant, distinguish input/parameter uncertainty, numerical error, model-form discrepancy, measurement uncertainty, process variability, out-of-domain applicability and unknown uncertainty. Calibration on one domain or operating regime is not evidence of calibration in another.

### 6. Promote only reproducible, bounded claims

A qualification report should bind its claim to canonical inputs, solver/version, environment, unit conventions, result artifacts, benchmark revision and content digests. A digest identifies bytes; it does not establish truth or authenticate an external source on its own.

## Benchmark ladder and proposed gates

All numeric targets below are **future acceptance targets**, not achieved results. Freeze the corpus, thresholds, reference versions and decision rules before evaluating a candidate.

| Tier | Test family | Proposed acceptance target | What it does not prove |
|---|---|---|---|
| Q0: numeric integrity | NaN/Inf, overflow, negative or inconsistent error, invalid tolerance, zero-reference and missing identity controls | 100% of invalid/control cases rejected; 100% of valid boundary cases match the specified oracle | Ground-truth validity |
| Q1: known-answer oracle | Hand-checked scalar equations and small independently calculated cases | 100% expected classifications on the frozen corpus; zero false accepts on protected negative controls | General solver fidelity |
| Q2: numerical verification | Manufactured solutions, mesh/time-step refinement, conservation and dimensional invariants | Pre-registered convergence order/tolerance met for every supported solver profile; no hidden failed runs | Physical model validity |
| Q3: validation and discrepancy | Held-out reference cases across regimes, geometries and boundary conditions | Report error distribution and subgroup failures; no aggregate metric may hide a failed critical stratum | Extrapolation outside tested coverage |
| Q4: uncertainty calibration | Protected holdout prediction intervals and residual diagnostics by declared domain/regime | Coverage evaluated with sample-size-aware confidence intervals; calibration and sharpness both reported | Universal calibration |
| Q5: decision robustness | Candidate designs near hard constraints under bounded perturbations | Zero false-feasible results on protected hard-constraint controls; worst-case regret and decision flips reported | Real-world reliability without field evidence |
| Q6: prospective evidence | Frozen predictions before new measurement or physical test | Predictions, exclusions and thresholds timestamped/frozen before unblinding; report all misses | Certification or deployment authorization |

### Mandatory scorecard

Never compress this scorecard to one undifferentiated leaderboard number.

- **False accepts / false rejects:** especially hard-constraint and safety-adjacent negative controls.
- **Error:** absolute, signed, relative where defined, tail error and worst case; show units and denominator rules.
- **Numerics:** convergence rate, conservation residual, conditioning/failure rate and sensitivity to solver settings.
- **Uncertainty:** empirical coverage, interval width/sharpness, subgroup calibration and unsupported/OOD rate.
- **Decision quality:** false-feasible rate, robust feasibility, frontier recovery, decision instability and worst-case regret.
- **Cost:** wall-clock, CPU/GPU/memory, physical-test burden and cost per decision improved.
- **Reproducibility:** exact input/model/solver identity, deterministic replay where promised, and complete artifact/digest coverage.
- **Evidence completeness:** missing sources, unknown units, uncalibrated instruments, shared model ancestry and stale revisions remain visible.

## Immediate next work, in order

1. Execute the focused Rust tests introduced with the numeric acceptance fix on the exact PR head; inspect source and test output independently of CI queue status.
2. Extend the independent, hand-authored 14-case oracle with mutation testing and additional domain-specific manufactured-solution fixtures; expected classifications must remain independent of production helpers.
3. Extend the contract from numeric acceptance to typed applicability, units, ground-truth provenance and claim boundaries by composing with SE-VV/ETK owners.
4. Add a small manufactured-solution corpus to one existing solver bridge before broadening solver count.
5. Implement decision-sensitive multifidelity selection through #7803 and the relevant domain owners only after the measurement/discrepancy reports have independently qualified.

## Verification status

When a Rust toolchain is available, the focused commands are:

\`\`\`bash
cargo test --features "engineering-foundations school_learning" --lib benchmark_validation
cargo test --features "engineering-foundations school_learning" --test engineering_benchmark_validation_oracle
\`\`\`

The focused source change, eight unit-test declarations, and a separately located 14-case integration oracle with hand-authored expected outcomes are on a dedicated public-repository branch. Rust execution is **not claimed** in this environment: neither `cargo` nor `rustc` is installed in the available shell, and queued CI jobs are not passes. No engineering capability, solver, physical result, reliability figure or safety certification is claimed by this document.

## References

- NASA, [NASA-STD-7009B: Standard for Models and Simulations](https://standards.nasa.gov/standard/NASA/NASA-STD-7009) (active standard page; issued 2024-03-05). Provides lifecycle guidance for credibility and assessment of modeling/simulation intended use.
- NASA, [NASA-HDBK-7009B: Implementation Guide for NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-HDBK-7009) (issued 2026-02-03). Provides implementation guidance and examples for credible model/simulation practice.
- ASME, [V&V 10: Verification and Validation in Computational Solid Mechanics](https://www.asme.org/codes-standards/find-codes-standards/standard-for-verification-and-validation-in-computational-solid-mechanics) (2019, reaffirmed 2025). Provides a VVUQ framework for computational solid mechanics.

These references inform the process and vocabulary; they are not evidence that this implementation conforms to a standard or that any engineering model is qualified.

# Falsification power reference — design note

**Date:** 2026-11-15  
**Status:** exact-head Rust qualification passed on source-integrated head 02f44f520e344277eae5cbc9a8aeabac8dcf482f; Rust 1.96.0, rustfmt, 11/11 tests and strict Clippy all passed in [focused workflow run #7](https://github.com/Luminous-Dynamics/symthaea/actions/runs/43760015297)  
**Owner:** Symthaea issue #12375  
**Implementation:** research/falsification-power-reference/

## 1. Why this slice exists

The 600,000-trial pilot recorded in #12375 demonstrates that a fixed endpoint reserve can fail a predeclared improvement guard and that late-onset omitted mechanisms can remain almost undetectable under a small observation budget. The next step is not to tune the reserve fraction. It is to quantify the detection ceiling induced by the actual design and each declared material alternative.

This reference is deliberately downstream of the existing open-world microcosm (#11060) and model-class adequacy/refinement evaluator (#12205). It does not duplicate their world, model representation, selector, or refinement logic.

## 2. Mathematical reference

For a fixed design with independent observations and known common Gaussian noise standard deviation sigma, let d_i be the evaluator-known difference between an alternative model and the working model at observation i.

The matched-filter noncentrality is:

    lambda = sqrt(sum_i (d_i / sigma)^2)

For a one-sided level-alpha normal likelihood-ratio test aligned with the known discrepancy direction, let c = Phi_inverse(1 - alpha). The best-case power against this fixed alternative is:

    power = 1 - Phi(c - lambda)

The Rust crate implements the normal quantile and CDF without external dependencies. It also computes the minimum replicate count for a constant known residual.

The discrepancy vectors are evaluator-only. They must never be exposed to Symthaea's candidate-policy process during a blind run.

## 3. Why this is not a scientific adequacy test

This is a deliberately optimistic ceiling:
- the discrepancy direction is known;
- observation noise is iid Gaussian with known sigma;
- the design and stopping rule are fixed;
- nuisance-parameter estimation, model selection, multiple testing and repeated adaptive looks are absent.

It therefore answers a narrow question: could this design distinguish this declared alternative under idealized assumptions? It does not answer whether the candidate model is true, whether a test is calibrated under a composite null, or whether an unknown alternative exists.

The correct negative result is **underpowered / not evaluable**, not **adequate**.

## 4. Integration contract

For each blind campaign:
1. Pin the candidate-model digest, experiment-policy digest, observation IDs and exact design.
2. Evaluator custody supplies a declared alternative envelope and calculates each alternative's residual vector after the design is frozen.
3. Evaluate all alternatives with the same observation IDs, order, observation count and cost budget. The crate checks vector lengths only; the evaluator harness must bind the exact design and cost records.
4. Preserve every alternative row; report minimum power and minimum detectable effect rather than a pooled power mean.
5. If an alternative is below target, mark the campaign underpowered for that alternative. Do not treat a non-rejection as adequacy evidence.
6. Only after the oracle reference is differentially verified should the actual Symthaea research policy be compared with random, fixed grid, EIG/model disagreement, D-optimal and support-aware policies under matched costs.
7. The operational adequacy test must be calibrated independently. Sequential use needs an anytime-valid test or a frozen alpha-spending schedule.

The initial known-answer family is a simple Gaussian mean shift. Future extensions should be separate and independently qualified: estimated nuisance parameters, heteroskedastic/correlated covariance, delayed responses, bounded count data, censored observations, sensor drift, and composite alternative families.

## 5. Cross-project responsibilities

- **Symthaea:** owns hypothesis/model ensembles and the research policy; consumes the result as an evaluator diagnostic, never as an agent-visible oracle.
- **Symtropy:** owns physical/numerical qualification, manufactured solutions, convergence and independent solver comparison. Its simulation residuals must be compared against independently generated alternatives.
- **Independent verification:** checks mathematical assumptions, known-answer results, statistical calibration and the validity of any sequential test.
- **Mycelix:** records source/model/protocol/design/alternative-envelope digests, raw output, evaluator version and independent verification disposition. A signature or content hash proves identity/provenance, not scientific truth.
- **Sol-Atlas:** may consume only domain-scoped, uncertainty-bearing model claims with applicable field evidence and professional approval.

## 6. Acceptance gates

The current v0 API checks equal residual-vector lengths; it does not validate observation identity/order, units, sensor basis or acquisition cost. Those identities must be bound by the evaluator harness. This limitation is deliberate and must remain visible in every integration.


Before calling the crate qualified:
- exact-head cargo fmt --check, cargo test, and strict Clippy;
- differential known-answer comparison with an independent mature statistics implementation or high-precision oracle;
- null calibration over the declared test family;
- monotonicity and boundary/property tests;
- zero-signal, zero-information and non-identifiability controls;
- explicit tests for alternative-envelope omissions and mismatched observation budgets;
- mutation tests for alpha, observation-count, residual-sign and alternative-ID guards.

Before claiming policy improvement:
- same task set, seeds, model/tool access, action/cost budget and stopping rule;
- preregistered primary outcome and alternative strata;
- independent calibration and sealed confirmation panels;
- all timeouts, invalid outputs and underpowered outcomes retained;
- paired task/family-level uncertainty intervals;
- no aggregate score that hides a weak alternative family.

## 7. Benchmark targets

The initial target is not a single percentage. It is:
- type-I error at or below the declared alpha, with confidence intervals reported;
- alternative-specific power curves and minimum detectable effects;
- at least 80% power at alpha <= 5% for each declared material alternative where feasible;
- explicit underpowered disposition when 80% is infeasible under the budget;
- zero false unique-model promotions on exact non-identifiable controls;
- model repair accepted only after sealed prediction confirmation.

These are qualification targets, not achieved results.

## 8. Next work

1. Compile and run this reference on its exact commit.
2. Differentially compare the Gaussian functions and minimum sample sizes against an independent oracle.
3. Add design-specific residual vectors generated from existing #11060 worlds, including zero-information and threshold-onset alternatives.
4. Add estimated-parameter and correlated-noise strata only as separate model families with their own calibration.
5. Connect to #12205 only through an explicit stochastic-power disposition; retain its current deterministic response-table contract.
6. Carry a qualified underpowered disposition into Mycelix provenance and ensure Sol-Atlas cannot silently upgrade it to a validated engineering claim.


The alpha level is per declared alternative. A campaign-level claim that searches across alternatives must separately control family-wise error or use a qualified joint/sequential test; this reference does not do that.

# Temporal Intervention Audit

**Package:** `symthaea-temporal-intervention-audit`  
**Tracking:** NEURO-SILICON-011 / #9561; related to NEURO-SILICON-010 / #9872  
**Status:** evaluator-only research implementation; Rust tests authored, not compiled or executed.

## Purpose

Estimate and test the realized effect of a temporal action using preregistered, randomized, matched treatment/control pairs. This is the missing inferential layer between candidate-predicted action value and evaluator-measured outcomes.

The package is deliberately narrow:

- one frozen candidate model and one treatment policy versus one control policy;
- two matched runs per pair, with one treatment and one control;
- an independently randomized fair-coin assignment of treatment to the first or second run;
- three separate evaluator outcomes: wrong-belief, abstention and missing-output exposure;
- exact sign-flip randomization inference for the sharp null of no treatment effect.

## Estimand and exact test

For pair \(i\) and exposure axis \(j\), define realized benefit as

\[
d_{ij}=Y^{control}_{ij}-Y^{treatment}_{ij}.
\]

Positive benefit means the treatment reduced the corresponding exposure. The candidate's forecast is retained separately and is never substituted for the realized outcome.

Under the sharp null of no treatment effect, and conditional on the paired outcomes, an independently randomized fair-coin assignment makes the sign of each pair difference equally likely. The exact one-sided randomization p-value is

\[
p_j = 2^{-n}\sum_{s\in\{-1,+1\}^{n}}
\mathbf{1}\left\{\sum_i s_i |d_{ij}|\geq \sum_i d_{ij}\right\}.
\]

A two-sided p-value is also returned. Gray-code enumeration evaluates the complete randomization space in \(O(2^n)\) time per axis, with a hard limit of 20 pairs. The report includes a Bonferroni adjustment across the preregistered comparison family; the caller must include all planned policy contrasts in that family size.

This is an exact randomization test of a sharp null under the declared assignment design. It is **not** a confidence interval, a guarantee under arbitrary temporal drift, or proof that the effect generalizes beyond the intervention protocol.

## Pair integrity and failure accounting

Every pair binds task, model, evaluator, policy, context, source-family, randomization-manifest and assignment-receipt digests. The first and second runs must share the same context and source-family digests. Run IDs, pair IDs, context digests and assignment receipts must be unique.

The preregistered pair-ID manifest must match the supplied pair set exactly. A timeout, invalid run or missing response is retained as full-horizon missing-output exposure; an explicit abstention is retained as full-horizon abstention exposure. These outcomes are not silently dropped. Completed outcomes are checked so the three disjoint exposure components do not exceed the episode horizon.

The crate consumes already-validated evaluator records. It does not verify signatures, randomization receipts, process isolation or witness quorums. Those remain upstream responsibilities. A digest is a commitment, not proof that an assignment was genuinely random.

## Interpretation limits

Exactness depends on all of the following:

1. treatment assignment was independently randomized within every pair with probability 0.5;
2. the assignment receipts and preregistration manifest were validated independently;
3. the treatment/control pair is genuinely matched on its declared context;
4. there is no interference between randomized units (or randomization is performed at the correct cluster level);
5. every planned pair, including failures and timeouts, is present.

If pairs are clustered by source family and treatment can affect other pairs in that family, randomize and analyze at the source-family cluster level rather than pretending pair-level independence.

The test evaluates a preregistered treatment/control contrast. Searching across many policies and reporting only the winner invalidates the nominal p-value unless the full search family is included in multiplicity control or evaluated on a separate confirmatory split.

The result is causal only for the randomized intervention, measured outcomes, candidate version, task family, resource budget and episode horizon tested.

## Complexity and limits

- Minimum: 2 complete randomized pairs.
- Maximum exact enumeration: 20 pairs (1,048,576 assignments).
- Three primary exposure axes are always reported separately.
- Multiplicity is controlled with a caller-declared Bonferroni family size.
- No approximate Monte Carlo path is silently substituted above 20 pairs; larger studies need a separately preregistered extension.

## Local validation

From the repository root:

```sh
cargo test --manifest-path research/temporal-intervention-audit/Cargo.toml
```

Twenty-six test functions are authored across the fixed-sample and sequential audit modules. They have not been compiled or executed. No effect-size, p-value, SOTA or hardware claim is made.


## Anytime-valid sequential lane

The exported `sequential` module supports repeated looks for one frozen treatment/control contrast. Each non-tied pair contributes an exact rational betting factor:

[
E_t = (3/2)^{W_t}(1/2)^{L_t},
]

where (W_t) and (L_t) are the numbers of beneficial and harmful paired signs. Ties contribute a factor of one. Under the Fisher sharp null and independently randomized fair-coin assignment orientation within every pair, each nonzero sign is conditionally symmetric and this is a nonnegative martingale.

For (M) predeclared comparisons and familywise level (alpha), a comparison crosses at (E_t ge M/alpha). Ville's inequality and a union bound give time-uniform familywise control across arbitrary stopping times under the stated design. The first crossing is latched in the report; later harmful evidence does not erase the earlier stopping event.

This is not the same as the fixed-sample exact sign-flip p-value above. It is a simple, exact-rational betting baseline that intentionally ignores effect magnitude. It may be less powerful than the terminal randomization test. It is valid only for independently randomized fair-coin pairs under the sharp no-effect null. Fixed-count blocked randomization, clustered assignment, carryover or interference require a design-specific process and are rejected/not supported here.

The state is in-memory. The runner must persist each update and receipt append-only; this crate does not prevent storage rollback, verify assignment signatures, or prove that an assignment was random. Maximum exact state is 64 pairs; the integer e-value is returned as an exact numerator/denominator pair. Fourteen additional sequential-module test functions are authored, including exact finite-null-tree enumeration through 20 looks. They have not been compiled or executed.

Related work: [#8679](https://github.com/Luminous-Dynamics/symthaea/issues/8679) covers general bounded value/risk e-processes; [#9820](https://github.com/Luminous-Dynamics/symthaea/issues/9820) covers induction of fixed-horizon tests; [#9855](https://github.com/Luminous-Dynamics/symthaea/issues/9855) covers finite-filtration optional-stopping/continuation. This module is narrower: it exploits the sign symmetry induced by the paired fair-coin assignment design.

# AR-5B14: Finite Composite-Null Pointwise Optional Continuation v0.1

**Status:** finite two-stage reference draft; source authored, tests not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → #9825 → this branch.  
**Tracking:** #9850.  
**Research anchor:** Koning & van Meer, “Anytime validity is free: inducing sequential tests,” JRSSB 88(4), 1366–1384, published 21 February 2026. https://doi.org/10.1093/jrsssb/qkag050  
**Claim ceiling:** finite family of explicitly supplied null distributions over a common two-stage finite support.

## 1. Research shift: preserve the family, not only its worst-case scalar

A simple-null interim test value can be used as the conditional significance budget for continuation. Composite nulls require more care.

For a null family H, retain a pointwise family of model-specific tests:

- For every P in H: E_P[phi_1^P] <= alpha.
- For every P in H and every declared first-stage history: E_P[phi_2^P | F_1] <= phi_1^P.
- At the terminal path, construct phi_2^H = ess inf over P in H of phi_2^P.

For a finite family, the essential infimum is an ordinary pointwise minimum. For each fixed P, phi_2^H <= phi_2^P pointwise, so:

  E_P[phi_2^H]
  <= E_P[phi_2^P]
  <= E_P[phi_1^P]
  <= alpha.

This establishes the finite construction's type-I error bound under each declared null model. It is the finite analogue of Theorem 6 in the research paper.

Crucially, the interim budget remains model-specific. A single minimum or average of interim budgets is not silently substituted for the family of conditional statements.

## 2. Data model

The plan contains:
- A finite set of uniquely named null models.
- One shared first-stage observable outcome support.
- A model-specific probability distribution over first-stage outcomes.
- A model-specific interim [0,1] test value for each first-stage outcome.
- A model- and history-specific conditional second-stage distribution.
- A model-specific [0,1] terminal rejection value for each shared second-stage outcome.
- A declared global alpha.

All models must share the same observable terminal path support so the pointwise minimum is well-defined. Probability distributions are normalized per model/history. Duplicate or missing model IDs, branch IDs, or path IDs fail closed.

The validator checks every declared conditional branch, including histories with zero probability under a particular model. This is stricter than an almost-sure conditional statement; it avoids silently ignoring an unexercised branch.

The analyzer emits:
- First-stage size and model-specific continued size for every null.
- Conditional branch size and unused conditional budget for every model/history.
- Each model's test value and the pointwise-minimum value at every terminal path.
- The composite terminal-test size under every null model.
- A plan digest and reconstructable report digest.

## 3. Arithmetic fixture (synthetic, not empirical)

Set alpha = 0.05 and use two null models over first-stage histories x and y.

| Quantity | Null A | Null B |
|---|---:|---:|
| P(x) | 0.8 | 0.2 |
| P(y) | 0.2 | 0.8 |
| phi_1(x) | 0.05 | 0.03 |
| phi_1(y) | 0.01 | 0.05 |
| E[phi_1] | 0.042 | 0.046 |

The model-specific conditional continuation sizes are:

| History | Null A | Null B |
|---|---:|---:|
| x | 0.5 × 0.08 = 0.04 <= 0.05 | 0.4 × 0.04 = 0.016 <= 0.03 |
| y | 0.5 × 0.02 = 0.01 <= 0.01 | 0.4 × 0.08 = 0.032 <= 0.05 |

The pathwise minimum is:
- (x, hit): min(0.08, 0.04) = 0.04
- (x, miss): 0
- (y, hit): min(0.02, 0.08) = 0.02
- (y, miss): 0

Its size under each null is:
- Null A: 0.8 × 0.5 × 0.04 + 0.2 × 0.5 × 0.02 = **0.018**
- Null B: 0.2 × 0.4 × 0.04 + 0.8 × 0.4 × 0.02 = **0.0096**

Both are below alpha and below the corresponding first-stage size. The model-specific continuation sizes before taking the pointwise minimum are 0.034 (A) and 0.0288 (B). These are arithmetic fixture values, not measured system performance.

## 4. What this does not establish

This reference does not implement:
- An infinite/continuous null family or measurable essential-infimum construction.
- Arbitrary stopping-time filtrations or conditional anytime-valid processes.
- Composite confidence sequences or simultaneous multiobjective inference.
- Dependent observations, drift, adaptive data reuse, or model misspecification guarantees.
- Proof that a real evaluator's declared conditional distribution is correct.
- Alternative-hypothesis power or improved engineering decisions.

A finite null family is only as protective as the family it contains. Omitting the true null model invalidates the intended coverage claim. A reviewer must assess whether the declared family covers plausible null behaviour.

## 5. Comparative benchmark programme

Compare pointwise composite continuation against:
1. Stop-and-discard after the exploratory stage.
2. A predeclared alpha split (for example, 0.025 + 0.025).
3. A collapsed worst-case interim budget supplied to one shared continuation test.
4. Pointwise model-family continuation using the minimum test.
5. An invalid negative control that restarts at fresh unconditional alpha.
6. A misspecified-family negative control where the generating null is deliberately omitted.

For each frozen scenario, report:
- Empirical and exact type-I error separately for every generating null.
- Maximum null size over the declared family and over held-out omitted nulls.
- Conditional budget overruns by model/history.
- Rejection power, expected decision loss, and calibration under alternatives.
- Number of evaluator calls, sample count, wall time, energy and verification cost.
- Conservatism of the pointwise minimum relative to each model-specific terminal test.
- Reproducibility across independent seeds and exact run receipts.

Do not compare only power or only runtime. A candidate can be better only at equal error control and on a predeclared decision-utility/cost objective. The omitted-null control is essential: coverage of an enumerated family is not robustness to an unknown family.

## 6. Proposed acceptance gates (targets, not results)

- All model-specific first-stage sizes <= alpha.
- Every model/history conditional continuation size <= its own interim budget.
- Shared observable path support is exact across models.
- Pointwise minimum is no greater than each model-specific path value.
- Composite test size <= alpha under every declared null model.
- Missing model, missing history, support mismatch, duplicate ID, invalid probability, over-budget conditional branch, and tampered report all fail closed.
- Canonical input order yields an identical report and digest.
- Independent finite arithmetic agrees with the Rust report to a declared tolerance.
- Held-out omitted-null stress tests are reported separately and never folded into the finite theorem claim.

## 7. Relationship to industry/SOTA methods

This is a reference-oracle and auditability increment, not an assertion of state-of-the-art performance. Benchmark it against alpha-spending procedures, e-process/e-value methods, confidence sequences, group-sequential/conditional-error approaches, and the stop/restart baseline. The distinctive testable hypothesis is not “always more powerful”; it is that retaining model-specific conditional budgets can avoid unnecessary conservatism from collapsing the family too early while preserving a valid pointwise composite terminal test.

The paper also develops sequentialized z- and t-tests and stopping-time continuation. Those should be separate follow-on implementations and compared on matched data-generating processes rather than conflated with this finite reference.

## 8. Exact-head validation

Run from the PR head after obtaining a Rust toolchain:

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering composite_null_continuation -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, test, clippy, CI, benchmark or SOTA pass is claimed.

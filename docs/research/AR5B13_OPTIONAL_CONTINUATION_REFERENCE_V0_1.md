# AR-5B13: Finite Optional-Continuation Significance Ledger v0.1

**Status:** finite two-stage reference draft; tests authored but not compiled or executed.  
**Stack:** #9429 → #9718 → #9719 → #9721 → #9754 → #9782 → #9785 → #9787 → #9807 → #9814 → #9823 → this branch.  
**Research anchor:** Koning & van Meer, “Anytime validity is free: inducing sequential tests,” JRSSB 88(4), 1366–1384, published 21 February 2026. https://doi.org/10.1093/jrsssb/qkag050  
**Claim ceiling:** finite simple-null two-stage conditional-significance ledger. It verifies the tower-property budget contract for explicitly supplied probabilities and randomized test values. It does not implement arbitrary stopping-time filtrations, conditional anytime-valid continuation processes, or composite-null validity.

## 1. Why optional continuation matters

The induced test in #9823 produces an interim [0,1]-valued test outcome. That outcome is not merely a dashboard score: under the null, it can be interpreted as a conditional significance budget for a later test.

Let the initial test outcome phi_1 be valid at level alpha:

```
E_0[phi_1] <= alpha
```

A new conditional test phi_2 is allowed to continue if it satisfies:

```
E_0[phi_2 | F_1] <= phi_1
```

Then the tower property gives:

```
E_0[phi_2]
  = E_0[E_0[phi_2 | F_1]]
 <= E_0[phi_1]
 <= alpha
```

The second test may use a different evaluator, experiment, or measurement process, provided its conditional null validity is justified under the information available at the continuation point.

The paper's Theorem 5 extends this to a sequential test stopped at an arbitrary stopping time and continued with a conditionally anytime-valid test at the observed interim significance value.

## 2. Finite two-stage representation

The v0.1 plan declares:
- a first-stage finite outcome space with null probabilities;
- an interim [0,1] test value for every first-stage outcome;
- one conditional second-stage outcome distribution for every first-stage outcome;
- a [0,1] rejection probability for every second-stage outcome;
- an unconditional significance limit alpha.

The validator checks:
1. First-stage probabilities form a probability measure.
2. The first-stage expected test value is at most alpha.
3. Every first-stage outcome has exactly one continuation branch.
4. Every conditional second-stage distribution is normalized.
5. Every branch's conditional expected rejection probability is at most its interim test value.
6. The unconditional continued size is at most both the first-stage size and alpha.

The report retains branch-level conditional size, remaining conditional budget, joint contribution to continued size, the plan digest and a reconstructable report digest.

## 3. Worked fixture

The synthetic fixture has:
- first-stage outcome **ambiguous** with null probability 0.6 and interim budget 0.07;
- first-stage outcome **strong-evidence** with null probability 0.4 and interim budget 0.02.

The first-stage size is:

```
0.6 * 0.07 + 0.4 * 0.02 = 0.05
```

Conditional continuation sizes:
- ambiguous branch: 0.3 * 0.2 = 0.06 <= 0.07;
- strong-evidence branch: 0.1 * 0.1 = 0.01 <= 0.02.

The unconditional continued size is:

```
0.6 * 0.06 + 0.4 * 0.01 = 0.04 <= 0.05
```

The fixture therefore leaves 0.01 unconditional significance budget unused. It is a constructed probability-table example, not an empirical study.

## 4. Integration opportunities in Symthaea

The architecture can eventually use this contract when:
- a noisy candidate measurement is stopped early and a confirmatory replicate campaign is launched;
- an exploratory test is followed by a different evaluator or higher-fidelity simulation;
- a campaign changes its measurement protocol after an interim diagnostic;
- a policy shifts to a confirmatory benchmark after an adaptive search phase.

Each continuation needs to bind the visible EventTape prefix, first-stage test identity, conditional null model, new test identity, cost model and all evidence. The finite ledger here does not itself prove the second test is conditionally valid; it checks the declared finite contract. That conditional-validity premise must be independently established for the real testing procedure.

## 5. Composite-null warning

The source paper also discusses composite hypotheses. A single conservative composite test value need not always be the best conditional significance budget for continuation. Their pointwise construction maintains a family of tests indexed by null distributions and uses a conservative essential-infimum aggregation.

Do not assume that a simple-null conditional budget automatically extends to a composite null, distribution shift, dependent observations or model uncertainty. A future extension must state the null family and conditional validity for every null distribution.

## 6. Proposed benchmark programme

Compare:
1. Stop-and-discard after the first test.
2. Optional continuation at the interim test value.
3. Conservative continuation at a fixed predeclared alpha split.
4. Optional continuation using a composite-null pointwise family.
5. Invalid controls that continue at a fresh unconditional alpha without conditioning.

Report:
- exact finite null size and conditional branch overrun rate;
- unconditional continued size;
- power at the final decision stage;
- expected sample count and time-to-decision;
- decision loss under asymmetric false-positive/false-negative costs;
- all-in experiment, evaluator, verifier and controller cost;
- performance under changed evaluator, model and noise regimes.

A method is only better if it improves terminal decision utility or cost at equal error control. Do not claim more power merely because the second-stage test is run with a larger conditional alpha on some branches.

## 7. Proposed acceptance targets (not results)

- Every first-stage probability distribution is normalized.
- First-stage expected test value is no greater than alpha.
- Every conditional continuation branch is within its interim budget.
- Unconditional continued size is no greater than first-stage size and alpha.
- Conditional overrun, missing branch, duplicate ID and report tampering negative controls fail closed.
- Exact reconstruction is deterministic after canonical ordering.
- Composite-null and anytime continuation claims require separate proofs and tests.

## 8. Exact-head validation commands

```bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering optional_continuation -- --nocapture
cargo test -p symthaea-architecture-engineering anytime_test_induction -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
```

No compile, test, clippy, CI, benchmark or SOTA pass is claimed.

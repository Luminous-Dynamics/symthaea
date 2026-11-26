# SCI-ANYTIME-IMPERFECT-CERT-001 — Anytime-valid certification with imperfect judges

**Status:** Protocol and mathematical contract. No sequential certification result or empirical qualification is claimed.

**Tracking:** [#9450](https://github.com/Luminous-Dynamics/symthaea/issues/9450).

## 1. Composition boundary

This is the anytime-valid integration of:
- [#9276 — adaptive test allocation](https://github.com/Luminous-Dynamics/symthaea/issues/9276);
- [#9331 — finite-sample imperfect-judge certification](https://github.com/Luminous-Dynamics/symthaea/issues/9331);
- [#9431 — correlated judge ensembles](https://github.com/Luminous-Dynamics/symthaea/issues/9431);
- [#9433 — generalized population estimands](https://github.com/Luminous-Dynamics/symthaea/issues/9433).

It does not own candidate selection, reference construction, general population modeling, or provider identity. It owns the validity of the statistical composition and stopping decision.

## 2. Measurement-channel inversion

For a binary failure label with true prevalence \(p\), judge-positive rate \(q\), sensitivity \(t\) and false-positive rate \(f\):

\[
q=f+(t-f)p,\qquad p=\frac{q-f}{t-f}\quad\text{when }t>f.
\]

For conservative upper-bound inference, construct simultaneous bounds:
- TPR lower bound \(t_L\);
- FPR lower bound \(f_L\);
- FPR upper bound \(f_U\);
- observed judge-positive prevalence upper bound \(q_U\).

The discrimination gate is:

\[
t_L>f_U.
\]

Comparing \(t_L\) with \(f_L\) alone is insufficient: the confidence-compatible interval may still contain a judge with FPR at least as high as TPR.

When the separation gate passes, the initial plug-in envelope is:

\[
p_U=\operatorname{clip}_{[0,1]}
\left(\frac{q_U-f_L}{t_L-f_L}\right).
\]

The derivation is conditional on the binary measurement channel, reference semantics, stable TPR/FPR within the declared stratum, and simultaneous coverage of all four bounds. A separate independent mathematical review must confirm the implementation and edge cases before any certification decision uses it.

## 3. Anytime-valid alpha spending

For total family-wise error budget \(\alpha\), \(S\) predeclared one-sided stream/look sequences, and the \(n\)th observation in one sequence:

\[
\alpha_{s,n}=\frac{\alpha}{S\,n(n+1)}.
\]

Since \(\sum_{n=1}^{\infty}1/[n(n+1)]=1\), the total allocation over all \(S\) streams and all looks is at most \(\alpha\).

For a Bernoulli stream of \(n\) observations with empirical mean \(\bar X_n\), use the one-sided Hoeffding radius:

\[
r_n=\sqrt{\frac{\log(1/\alpha_{s,n})}{2n}}.
\]

The one-sided lower / upper bounds are \(\max(0,\bar X_n-r_n)\) and \(\min(1,\bar X_n+r_n)\). Use four distinct one-sided streams per judge-policy × task-family: TPR lower, FPR lower, FPR upper, and test-positive prevalence upper. The total \(S\) includes all streams across every predeclared arm and family.

This is a transparent, conservative first method, not an assertion that Hoeffding alpha spending is statistically optimal. Mixture / betting confidence sequences and exact fixed-sample alternatives should be compared after independent validation.

## 4. Adaptive acquisition assumptions

V1 permits adaptive choice among predeclared streams, but not outcome-adaptive selection of the next hidden item within a stream.

For each arm × family × measurement type:
- freeze the stream population and sampling frame;
- commit a randomized within-stream draw sequence before outcomes are observed;
- when the allocator chooses a stream, consume the next item from that stream's committed random sequence;
- make the stream choice predictable from prior observations only;
- keep all hard safety controls mandatory;
- preserve positive support for every stratum used in the claim;
- record every selected/unselected stream, draw index, outcome, missingness and retry;
- keep the confirmatory lockbox sealed.

The per-stream Hoeffding confidence sequence remains valid at every within-stream look under the iid Bernoulli stream assumption, even when the scheduler adaptively chooses which predeclared stream to sample next. Cross-stream dependence is handled by the union bound; independence between streams is not required for that bound.

V1 does not cover adaptive within-stream item choice, unequal-probability sampling, finite-bank sampling without replacement, outcome-dependent missingness, or arbitrary covariate-adaptive sampling. Those require a separate martingale / propensity-weighted design. Screening estimates do not become confirmatory population estimates merely because the allocation policy records propensities.

## 5. Status semantics

- **Certified:** all required streams meet minimum counts; simultaneous bounds are valid; TPR lower exceeds FPR upper; every family prevalence upper bound is strictly below the frozen threshold.
- **Inconclusive:** evidence or stream support is insufficient, the TPR/FPR separation gate fails, or the hard budget expires.
- **Not certifiable:** invalid reference, sampling, provenance, plan, execution or report contract.

A confidence interval overlap is inconclusive, not evidence that the judge is definitely inverted. Conversely, a point estimate TPR > FPR does not establish discriminability.

For failure detection, the terminal action for judge abstention must be frozen. V1 defaults to positive/fail-closed and counts abstention as a positive judge action in both positive-reference and negative-reference streams and in test prevalence. Failed/missing calls remain explicit and cannot disappear from denominators.

## 6. Baselines

Compare:
1. Fixed-sample #9331-style certification.
2. Fixed-sample repeated-look Bonferroni.
3. Time-uniform Hoeffding alpha spending.
4. Validated mixture / betting confidence sequences.
5. Uniform stream allocation.
6. Variance / uncertainty-adaptive allocation.
7. Decision-risk / expected-value-of-information allocation.
8. Oracle allocation as a non-deployable diagnostic.

All arms use the same hard-control completion requirements and matched total resource budgets.

## 7. Synthetic known-answer and hostile suite

Include known TPR/FPR/prevalence combinations for:
- clear discrimination;
- overlap where TPR lower exceeds FPR lower but not FPR upper (must remain inconclusive);
- inverted judges and near-random judges;
- high TPR with dangerously high FPR;
- low base-rate rare failures;
- family-specific drift;
- adaptive selection concentrating on an easy stream;
- multiple judge arms and task-family strata;
- early stopping under null and alternative configurations;
- all-abstain, missing, failed and incomplete observations;
- tampered alpha, stream count, draw order, stopping policy and denominators.

Run deterministic Monte Carlo sequential-null experiments. Report Type-I error with its own Monte Carlo interval, power, expected sample size, median/p95 cost and interval width separately. A finite simulation is not a proof of coverage.

## 8. Metrics

Report:
- simultaneous coverage per bound and joint coverage across arms/families;
- Type-I error, power and false-certification probability;
- probability of early certification, inconclusive termination and budget exhaustion;
- calibration and test sample counts by class and family;
- average, median and p95 query / human-review / token / wall-clock cost;
- allocation concentration, selection support and effective independent sample count;
- time to decision and optional-stopping behavior;
- failure-family recall and mandatory-control completion;
- bound width and conservatism versus fixed-sample/exact/mixture-sequence baselines.

## 9. Implementation sequence

1. Independently audit the #9331 measurement-channel bound and discrimination condition.
2. Implement std-only alpha-spending and one-sided Hoeffding bounds with strict invalid-input handling.
3. Implement the four-bound channel envelope and explicit status machine.
4. Add exact known-answer tests, overlapping-interval negative control and deterministic sequential-null simulation.
5. Add a second independently written reference implementation for small cases.
6. Compose with #9276's scheduler only under the frozen per-stream sampling contract.
7. Compare against the fixed-sample #9331 procedure and allocation baselines under matched hard controls and total costs.

The first implementation is a mathematical kernel, not a provider adapter, adjudication platform, GLMM, or empirical certification result.

## 10. Research references

- Feng et al., *Noisy but Valid: Robust Statistical Evaluation of LLMs with Imperfect Judges*, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/08a67eb74bfb9ded9e949dd52973e997-Abstract-Conference.html
- Lee et al., *How to Correctly Report LLM-as-a-Judge Evaluations*, ICML 2026: https://proceedings.mlr.press/v306/lee26bl.html
- Zecchin et al., *Adaptive Learn-then-Test: Statistically Valid and Efficient Hyperparameter Selection*, ICML 2025: https://proceedings.mlr.press/v267/zecchin25a.html
- Saha et al., *LLM-as-a-Judge on a Budget*, AISTATS 2026: https://proceedings.mlr.press/v300/saha26a.html

These results motivate the programme. Their empirical reductions and statistical guarantees do not transfer automatically to Symthaea's candidate pools or task populations.

## 11. Claim boundary

The confidence sequence is valid only under its declared stream model, reference labels and frozen allocation procedure. It does not authenticate judge execution, establish human-label truth, repair task selection bias or prove generalization to an unrepresented population. No certification or superiority claim is made by protocol or synthetic fixture alone.

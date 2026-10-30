# Multi-stream sequential assurance: FDR-sup and asynchronous collection

**Date:** 2026-10-30  
**Status:** independent reproduction of a published two-null counterexample; eight local tests passed; not a general SOTA or production result.

## Why this matters to Symthaea

The anytime assurance monitor in #9072 is currently a single-stream construction. Scaling it to many issuers/risk cells is not merely a matter of running one valid e-process per stream and aggregating their current values.

Tavyrikov, Goeman & de Heide, *Carefree multiple testing with e-processes*, Electronic Journal of Statistics 20(2), 3178–3189 (2026), show:

- e-BH on stopped/current e-process values has an FDR guarantee at a selected vector of stopping times, but the rejection set need not remain stable as evidence is collected asynchronously;
- using the running maximum restores the desired accept-to-reject / carefree behavior, but the running maximum is not itself an e-process;
- under arbitrary dependence, naive e-BH on running maxima does not control FDR-sup;
- applying a valid adjuster to each running maximum yields an e-process again, and e-BH on those adjusted maxima controls FDR-sup at the stated bound;
- the guarantee comes with a potentially substantial power penalty.

Paper: https://doi.org/10.1214/26-EJS2546  
Author full text: https://ris.utwente.nl/ws/portalfiles/portal/499132886/2501.19360v1.pdf

## Independent reproduction

We independently implemented the paper's two-null counterexample, rather than copying an author's implementation.

Both hypotheses are true nulls. Their initial joint values are (1/(2 alpha), 1/(2 alpha)) with probability 2 alpha, otherwise (0,0). At each time, the joint multiplicative increments are (1/2,1/2), (2,1/2), or (1/2,2), each with probability one-third. Each marginal increment has expectation one, but the two streams are dependent.

We simulated 1,000,000 repetitions, 100 update steps, alpha=0.05, seed 20261030. FDR-sup in this all-null construction is the probability of at least one false rejection during the horizon.

| Aggregation | False discoveries | Estimated FDR-sup | Wilson 95% interval |
|---|---:|---:|---:|
| Current-value e-BH | 47,821 / 1,000,000 | 4.7821% | 4.7405–4.8241% |
| Naive e-BH on running maxima | 54,008 / 1,000,000 | **5.4008%** | **5.3567–5.4453%** |
| Adjusted running maximum, A1 | 761 / 1,000,000 | 0.0761% | 0.0709–0.0817% |
| Adjusted running maximum, A2 | 761 / 1,000,000 | 0.0761% | 0.0709–0.0817% |

The naive running-max procedure exceeded its 5% nominal level; the point estimate was 1.08016 times nominal. This reproduces the paper's reported approximately 1.08 alpha behavior in its constructed counterexample. Both adjusters were very conservative on this particular discrete process. Identical A1/A2 counts are a feature of this finite simulation, not evidence that the adjusters are equivalent in general.

The current-value method's result is a simulation of FDR-sup over the common update horizon; it is not an assertion that current-value decisions are monotone or carefree.

## Adjusters

The paper gives two admissible examples (continuous extension at M=1):

`A1(M) = (M - 1 - log M) / log(M)^2`, with `A1(1)=1/2`;

`A2(M) = sqrt(M) - 1`, with `A2(1)=0`.

For each stream, let `M_i(t)=max(1, sup_{s<=t} E_i(s))`. Under the paper's conditions, applying an admissible adjuster to `M_i(t)` produces an e-process. e-BH on the adjusted processes controls FDR-sup under arbitrary dependence, with the theorem's `K0 alpha / K` bound.

The implementation evaluates both adjusters in log space to avoid overflow and uses the continuous boundary values at one.

## Asynchronous-order witness

For alpha=0.05 and two streams, the small deterministic illustration uses current e-values:

- At time t: [30, 15] — e-BH rejects neither.
- After both advance: [15, 30] — e-BH still rejects neither.
- If stream 1 is frozen at its earlier evidence while stream 2 advances: [30, 30] — e-BH rejects both.

This highlights that current-value e-BH does not have the carefree property for arbitrary per-stream evidence schedules. The multiple-testing rule must specify how stopped values, running maxima and future observations are treated.

## Important distinction: FDR versus FWER

The paper's adjusted e-BH result is an FDR-sup guarantee, not a general FWER guarantee. In the all-null counterexample, FDR equals the probability of any false rejection, so it also numerically equals FWER. That equivalence does not hold in mixed-null systems.

For safety-critical issuer/certificate revocation where the objective is to bound the probability of **any** false quarantine among true-null issuers, a separate familywise mode is needed. This branch includes a weighted adjusted-e-Bonferroni helper: for weights w_i>0 summing to one, reject stream i only if its adjusted running-max e-process crosses 1/(alpha_total*w_i). Under the per-stream e-process validity conditions, Ville plus the union bound gives FWER <= alpha_total. This is a conservative option and is not the same objective as FDR optimization.

Recommended API must make the objective explicit:
- FDR mode: adjusted running maxima + the published adjusted e-BH construction.
- FWER mode: predeclared weighted adjusted-e-Bonferroni / alpha spending.
- Never silently substitute one for the other.

## Implementation artifacts

- `validation/multistream_eprocess_fdr_sup.py`: independent A1/A2 implementations, log-domain stability, e-BH, weighted adjusted-e-Bonferroni, asynchronous-order witness and simulator.
- `validation/test_multistream_eprocess_fdr_sup.py`: 8 regression tests.
- `validation/MULTISTREAM_EPROCESS_FDR_SUP_2026-10-30_1M.json`: raw simulation result with Wilson intervals and full protocol.

Local validation: **8/8 tests passed** against the exact committed source/test contents. Git blob identities were checked:
- source: `b8c84f3067489c488e8504befd1daf9395b3b429`
- tests: `ad8498dfd304b68a8e39dca6a8f25272bbd61f9e`

SHA-256:
- source: `9957c3b8feed884775340007090459934bd3b0f5be925a0ba3aaf3a0d2c129c0`
- tests: `b817b438d65c8c6c59560cb12409a906e8e3ed054ac3ecb906665c9808273d4e`
- results: `d3be72be912f319269be3a167f4f79b295fe22c3a87e6da1235ac84f19c2a4ba`

Reproduce:

```sh
PYTHONPATH=validation python -m unittest -v test_multistream_eprocess_fdr_sup.py
PYTHONPATH=validation python validation/multistream_eprocess_fdr_sup.py \
  --output /tmp/multistream-eprocess-fdr-sup.json \
  --repetitions 1000000 --horizon 100 --seed 20261030
```

## What this changes for the assurance architecture

The multi-issuer assurance fabric must not independently monitor streams and then improvise a ranking/aggregation rule. It must define at design time:

1. the null for each issuer (e.g. conditional issuer-bound violation rate <= p0);
2. whether the system's objective is FDR or FWER;
3. the per-stream evidence process and its validity assumptions;
4. the asynchronous evidence-scheduling policy;
5. the aggregation method (adjusted e-BH for FDR-sup, weighted adjusted e-Bonferroni for FWER);
6. global alpha allocation across issuers and epochs;
7. persistent running maxima and no silent resets/recycling.

## Next benchmark gates

1. Reproduce the full paper's 200-stream dependent-normal simulation and compare current e-BH, running-max e-BH, A1 and A2 in matched code.
2. Add mixed-null configurations (varying true-null fraction) and report FDR, FWER, power and rejections over time.
3. Stress adaptive scheduling: greedy frontier allocation, round-robin, random and adversarial scheduling.
4. Test 8, 32, 128, 512 and 4,096 simultaneous issuer streams with dependence structures (independent, block-correlated, common-factor, adversarial).
5. Quantify adjuster power loss against current-value e-BH and safe alpha-spending baselines.
6. Integrate only after the objective is explicit in Rust types; no generic `MultipleTestingResult` should hide whether it is FDR or FWER.

This is a reproduction of one published negative control and a useful implementation gate—not an original theorem, comprehensive benchmark or claim of beating SOTA.

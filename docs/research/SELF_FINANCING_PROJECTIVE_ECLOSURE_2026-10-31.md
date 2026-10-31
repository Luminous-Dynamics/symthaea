# Self-financing projective mergers for dynamic e-closure

**Status:** exact finite-horizon research oracle. The contribution is a candidate path-dependent merger; the stochastic guarantee is conditional on a separately verified one-step e-value contract in a common global filtration. No novelty or SOTA claim is made yet.

## Research motivation

Dynamic e-closure for online hypotheses with any-time-valid evidence (de Heide, 2026) shows that two properties are needed for simultaneous stopped-FDR / SupFDR:
- future-extension coherence across the active-hypothesis horizon;
- time-monotone intersection certificates for SupFDR and setwise persistence.

The paper's pointwise arbitrary-dependence mergers face a strong rigidity result: on an open-ended countable hypothesis horizon, nontrivial symmetric pointwise mergers cannot satisfy its global projectivity requirements. It explicitly leaves direct intersection betting strategies and history-dependent mergers outside that pointwise class as future directions.

This prototype explores a different object: a self-financing portfolio of **one-step coordinate e-value increments**, with predictable shares that may be re-normalized as the active set grows. Past wealth is never rewritten; only the next increment is allocated.

## Construction

Let \(X_{i,t}\geq0\) be the one-step e-value increment for coordinate \(i\) at global update \(t\). Require, under null \(H_i\),

\[
\mathbb E[X_{i,t}\mid\mathcal F_{t-1}]\leq1,
\]

where every coordinate increment is measurable in the **same global filtration**. The active sets \(I_t\) are deterministic and nested in this first prototype.

Choose nonnegative predictable shares \(p_{i,t}\), zero for inactive coordinates, such that

\[
\sum_{i\in I_t}p_{i,t}\leq1.
\]

The implementation compares several predictable share policies. Static priority uses a frozen nonnegative score \(s_i\) and normalizes only over hypotheses active before the update. Equal-active uses \(s_i=1\). AWAIRE-style policies use each coordinate's past wealth \(M_{i,t-1}=\prod_{u<t}X_{i,u}\), its square, or only the largest past wealth coordinate(s). In every case the shares are determined before the current increments are observed and normalized over the active universe, not separately for each intersection.

\[
p_{i,t}=\frac{s_{i,t}}{\sum_{j\in I_t}s_{j,t}}
\quad (i\in I_t),
\qquad p_{i,t}=0\quad(i\notin I_t),
\]

with all-zero scores interpreted as holding cash.

**Prior art:** Ek, Stark, Stuckey and Vukcevic's AWAIRE (2023) already constructs intersection test supermartingales by multiplying predictable convex combinations of component test-supermartingale factors, and studies linear, quadratic and largest-past-wealth weighting. Thus this PR does not claim to invent adaptive self-financing mixtures. The narrower question is how portfolio scope affects future-extension coherence and capacity-aware selection as the hypothesis universe expands.

A separate intersection-local AWAIRE comparator is now implemented in `validation/awaire_dynamic_projective_baseline.py`. For each (S), it normalizes predictable weights over (S\cap I_t), rather than over the whole active universe. Because its weights depend only on the currently active intersection and its past coordinate wealth, it also satisfies the neutral-padding projectivity check. On the two-coordinate fixture, local equal weighting gives (E^{\{1\}}=(1,8,8)), (E^{\{2\}}=(1,1,16)), and (E^{\{1,2\}}=(1,9/2,153/4)). Local past-wealth weighting gives (E^{\{1,2\}}=(1,9/2,12)). These are explicit comparator paths, not a general power ranking.

The comparison must include AWAIRE local linear / quadratic / largest-past-wealth, global static-priority / equal / wealth-adaptive shares, fixed globally summable projective mergers, and the no-capacity dynamic closure menu.

For every finite intersection \(S\), define the self-financing wealth process

\[
E_0^S=1,
\]

\[
E_t^S=E_{t-1}^S
\left[
1-\sum_{i\in S\cap I_t}p_{i,t}
+\sum_{i\in S\cap I_t}p_{i,t}X_{i,t}
\right].
\]

The bracket is a cash-plus-bets portfolio return. It is nonnegative because the share sum over \(S\cap I_t\) is at most one.

### Conditional supermartingale argument

Under the intersection null \(H_S=\bigcap_{i\in S}H_i\), every component \(i\in S\cap I_t\) satisfies the conditional e-value bound. Since the shares are predictable,

\[
\begin{aligned}
\mathbb E[G_t^S\mid\mathcal F_{t-1}]
&=1-\sum_{i\in S\cap I_t}p_{i,t}
+\sum_{i\in S\cap I_t}p_{i,t}
\mathbb E[X_{i,t}\mid\mathcal F_{t-1}]\\
&\leq1.
\end{aligned}
\]

Therefore \(E_t^S=E_{t-1}^S G_t^S\) is a nonnegative test supermartingale under \(H_S\), hence an e-process. No cross-coordinate independence is needed; only the individual conditional contracts in the common global filtration.

This is a short conditional-expectation proof for the proposed construction. It is not a proof that an arbitrary observed increment path satisfies the conditional contract. That belongs to the evidence producer and must be independently documented.

### Future-extension projectivity

The portfolio shares are determined from the global active set and scores, not from the intersection label \(S\). For a future extension \(S'\) and current time \(t\), set \(B=S'\cap I_t\). Since active sets are nested, for every \(u\leq t\),

\[
S'\cap I_u=B\cap I_u.
\]

The portfolio factor at update \(u\) therefore uses exactly the same terms for \(S'\) and \(B\). Thus

\[
E_t^{S'}=E_t^{S'\cap I_t}.
\]

This gives exact neutral-padding projectivity even though the shares are re-normalized over the active set as new hypotheses arrive. The re-normalization affects only future increments; it does not reallocate past wealth.

This path-dependent product construction is outside the **pointwise** merger class in de Heide's rigidity theorem. It is a candidate history-dependent betting strategy, not a contradiction of that theorem.

### Persistent SupFDR certificates

The raw intersection wealth processes may decrease. Apply one common nondecreasing adjuster \(A\) to each running maximum:

\[
\widetilde E_t^S
=A\!\left(1\vee\max_{u\leq t}E_u^S\right),
\qquad
\int_1^\infty\frac{A(x)}{x^2}\,dx\leq1.
\]

The prototype uses exact step adjusters

\[
A(x)=\sum_j a_j\mathbf 1\{x\geq c_j\},
\quad c_j>1,\quad a_j\geq0,
\quad\sum_j a_j/c_j\leq1.
\]

The same function is applied to every intersection path, so exact projectivity is preserved. The adjusted certificates are time-monotone. Under the dynamic e-closure theorem's assumptions, future-extension coherence supplies simultaneous stopped-FDR control and time-monotonicity supplies simultaneous SupFDR / persistence.

The implementation verifies the integral budget exactly but does not verify the coordinate e-process probability contracts.

## Exact two-coordinate fixture

The active sets are \(I_0=\{1\}\), \(I_1=I_2=\{1,2\}\). Equal priority scores give shares:

| Update | Active hypotheses | \(p_{1,t}\) | \(p_{2,t}\) |
|---:|---|---:|---:|
| 0 | {1} | 1 | 0 |
| 1 | {1,2} | 1/2 | 1/2 |
| 2 | {1,2} | 1/2 | 1/2 |

A valid one-step null-martingale construction can generate the all-success sample path used here:
- coordinate 1 has increment 8 with probability 1/8 (otherwise 0) at its first evidence update; other increments are one;
- coordinate 2 has increment 16 with probability 1/16 (otherwise 0) at its first evidence update; other increments are one.

The prototype tests the all-success path \(X_1=(1,8,1)\), \(X_2=(1,1,16)\). The raw intersection wealths are:

| Intersection | Time 0 | Time 1 | Time 2 |
|---|---:|---:|---:|
| {1} | 1 | 9/2 | 9/2 |
| {2} | 1 | 1 | 17/2 |
| {1,2} | 1 | 9/2 | 153/4 |

With alpha \(1/2\), step thresholds (2,4,8,16,32,64), unit increments, and cumulative capacity (0,1,2), the adjuster budget is exactly \(63/64\). The adjusted paths become:

| Intersection | Time 0 | Time 1 | Time 2 |
|---|---:|---:|---:|
| {1} | 0 | 2 | 2 |
| {2} | 0 | 0 | 3 |
| {1,2} | 0 | 2 | 5 |

With utilities (10,8) and costs (1,1), exhaustive selection returns \(\varnothing\), then \(\{1\}\), then \(\{1,2\}\). Each action belongs to every relevant dynamic closure constraint, respects cumulative capacity, and is utility-optimal among all nested feasible choices at that update.

The all-success path has positive probability under the stated one-step martingale construction. This fixture illustrates the algebra; it does not estimate SupFDR or prove the upstream conditional e-value contracts from data.

## Implementation

- `validation/awaire_dynamic_projective_baseline.py`
  - intersection-local predictable mixtures;
  - AWAIRE-style linear, quadratic, and largest-past-wealth policies;
  - independent pathwise future-extension coherence and time-monotonicity checks.
- `validation/test_awaire_dynamic_projective_baseline.py`
  - exact singleton and joint-intersection wealth paths;
  - equal and past-wealth local policies;
  - projectivity, dormant-coordinate, nonnested-activation, and negative-increment checks;
  - a correlated null-increment fixture with exact conditional expectation one for every tested predictable convex combination.


- validation/self_financing_projective_eclosure.py
  - predictable active-set-normalized shares;
  - exact self-financing intersection wealth;
  - exact step adjuster budget;
  - pathwise future-extension projectivity and time monotonicity;
  - dynamic closure membership and nested capacity-aware selection;
  - independent deterministic replay and exhaustive utility optimality.
- validation/test_self_financing_projective_eclosure.py
  - exact two-coordinate process values and decisions;
  - future-score/evidence prefix invariance;
  - valid-but-suboptimal decision negative control;
  - nested active-set, dormant-increment, negative-increment and invalid-adjuster checks;
  - all-zero priority / hold-cash behavior.

Reproduction from repository root:

    PYTHONPATH=validation python -m unittest -v test_self_financing_projective_eclosure.py

The committed test suite has not been executed in this environment. Remote CI is not claimed to pass.

## Scope and open risks

- Requires one global filtration and valid one-step conditional e-value increments for each coordinate; locally valid e-processes must first be lifted into that filtration.
- Active sets are deterministic and nested; random predictable arrivals need a separate extension of the dynamic-closure theorem contract.
- Scores are frozen. More general history-dependent predictable shares are plausible because the conditional expectation proof only needs predictability and a sum-at-most-one constraint, but they need explicit filtration binding and negative controls.
- Cumulative capacity only; no per-period queue stability or delayed-action service guarantee.
- Exact small oracle enumerates \(2^N-1\) intersections and \(2^N\) candidate actions; not a production optimizer.
- No claim of universal novelty: direct intersection betting and history-dependent mergers are identified as an open direction in the dynamic e-closure literature. This is a concrete candidate construction that should be compared with existing betting/portfolio literature and reviewed by a sequential-testing specialist.

## Benchmarks

Compare against:
- AWAIRE-style predictable mixtures (linear past wealth, quadratic past wealth, largest past wealth), including versions normalized within each intersection;
- static global-weight projective mergers;
- fixed global-weight projective mergers from dynamic e-closure;
- self-financing equal-share portfolios (this prototype);
- fixed priority-share portfolios;
- predictable score policies frozen before an epoch;
- online e-closure with one final e-value per arrival;
- adjusted running-max closed e-BH and dynamic e-closure without utility optimization.

Stress:
- independent, common-shock, block and adversarial cross-coordinate dependence;
- staggered arrivals and evidence update schedules;
- null martingale payout distributions with rare large jumps;
- predictable versus deliberately future-leaking shares;
- active-set renormalization versus frozen global weights;
- all-null, sparse and dense alternatives;
- capacity saturation, bursts and resource outages;
- score/value misspecification and utility regret;
- stopping times selected adaptively from the common filtration.

Report simultaneous stopped FDR and SupFDR separately, power, useful discoveries, utility, capital turnover, resource utilization, p50/p95 latency, memory, number of intersection constraints and replay-verification cost. Use exact differential fixtures first, then 5,000 exploratory and 50,000 frozen confirmatory streams. Monte Carlo remains diagnostic; the e-process contract and theorem supply validity.

## References

- de Heide (2026), [Dynamic e-closure for online hypotheses with any-time-valid evidence](https://arxiv.org/abs/2608.09927).
- Tavyrikov, Goeman & de Heide (2026), [Carefree multiple testing with e-processes](https://doi.org/10.1214/26-EJS2546).
- Xu, Fischer & Ramdas (2026), [Online analogs of e-closure and compound e-values](https://proceedings.mlr.press/v337/xu26a.html).

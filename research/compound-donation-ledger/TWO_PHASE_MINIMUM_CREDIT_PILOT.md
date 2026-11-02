# Exact-minimum-credit two-phase allocation: exploratory process-level pilot

## 1. What this benchmark tests

This experiment compares a source bank that waits for a recipient to become fundable against immediate fixed-next routing and no donation.

The core architecture has three separate operations:

1. **Reserve:** when donor i's own adjusted evidence increases, compute the new capped-overshoot reserve increment and deduct it from donor i's score. Place it in an unassigned, source-scoped bank.
2. **Assign:** at the next pre-observation checkpoint, compute the exact score credit required for each eligible candidate to enter the current e-BH rejection set. Assign reserve only if an earlier-source bank can cover the entire deficit.
3. **Credit:** increase the recipient's incoming capital by the assigned amount. Recompute e-BH and verify the target entered the rejection set.

The two-phase policy makes at most one assignment per checkpoint, chooses the smallest fundable deficit (tie-breaking by candidate ID), consumes source tokens oldest-source-first, and does not partially spend a bank that cannot cross an e-BH boundary.

## 2. Valid-null process generator

Each coordinate uses a Bernoulli likelihood-ratio process with null \(p_0=1/2\), alternative design value \(p_1=0.7\), and factors

\[
L(Y)=\begin{cases}
7/5,&Y=1,\\
3/5,&Y=0.
\end{cases}
\]

Under every true-null coordinate, conditional on the complete global past,

\[
\mathbb E[L(Y_{i,t})\mid\mathcal F_{t-1}]
=\frac12\frac75+\frac12\frac35=1.
\]

The simulation uses independent, shared-uniform and balanced-negative constructions. For shared-uniform dependence, a fresh common uniform is drawn each look; every null coordinate still has conditional success probability one half. For balanced-negative dependence, a random half (or floor/ceiling half for odd counts) of the active null coordinates succeeds each look; exchangeability gives conditional marginal probability one half for every null coordinate.

The running-max process is adjusted by

\[
A(x)=\sqrt{x}-1,
\]

which has integral budget one:

\[
\int_1^\infty\frac{\sqrt{x}-1}{x^2}\,dx=1.
\]

To retain exact rational score and reserve arithmetic, the simulator uses a fixed-grid floor of this adjuster. This is a pointwise lower approximation to the valid adjuster and is pathwise nondecreasing, so it does not increase the null expectation.

Candidate weights are fixed before data:

\[
\gamma_i=\frac{1}{(i+1)(i+2)},\quad i=0,\ldots,23,
\qquad \sum_{i=0}^{23}\gamma_i=\frac{24}{25}.
\]

## 3. Policies

- **No donation:** score \(\gamma_i X_{i,t}\).
- **Immediate fixed-next:** reserve the donor's own increment and bind it to the next candidate; credit immediately if active, otherwise wait for activation.
- **Two-phase exact minimum credit:** reserve from the donor's own increment; hold unassigned source escrow; on a later pre-observation checkpoint compute exact e-BH deficits; assign exactly enough to the smallest fundable deficit, using only sources with ID smaller than the recipient.

For a fixed finite active score vector z, e-BH rejects the top \(K\) scores, where

\[
K=\max\{k:z_{(k)}\ge 1/(\alpha k)\}.
\]

The minimum-credit oracle in #10120 computes the smallest additional score credit that makes a specified candidate enter this rejection set. The two-phase policy only spends when the earlier-source bank covers that exact amount.

The routing decision uses only the last committed score vector and bank state before current observations. It does not use current-look outcomes or null labels. Because recipient selection can depend on the whole active score vector, this is the adaptive non-projective mode in #9923; strong horizon projectivity is not claimed.

## 4. Pilot protocol

- 24 hypotheses;
- 50 looks;
- \(\alpha=0.10\);
- fixed global weights above;
- three dependence constructions;
- six scenarios;
- 50 streams/cell for all-null, immediate-bridge and late-bridge;
- 20 streams/cell for far-cluster, no-useful-recipient and switching.

The local exact-rational pilot used deterministic seed schedules recorded in the JSON manifest for this experiment. The checked-in runner and tests are committed separately, but the recorded pilot came from a local equivalent reimplementation of the same LR, adjuster, reserve, exact-credit and source-bank logic. The checked-in script itself was not run from a connected repository checkout.

The one-sided 95% Hoeffding half-width for bounded means is approximately 17.3 percentage points at 50 streams and 27.4 points at 20 streams. No simultaneous confidence correction was applied. These intervals are too wide to establish superiority.

## 5. Main observations

### Late-borderline target

The late-bridge scenario gives the donor and several earlier alternatives time to accumulate source reserve before a useful target activates.

| Dependence | No donation power | Fixed-next power | Two-phase exact-credit power |
|---|---:|---:|---:|
| Independent | 52.00% | 53.00% | 54.14% |
| Shared-uniform | 51.86% | 51.86% | 53.43% |
| Balanced-negative | 49.86% | 50.57% | 51.71% |

In this small pilot, the two-phase method is directionally better in all three dependence constructions. It assigns an average of roughly 2.8–4.2 bank allocations per stream and retains a nontrivial unassigned balance after allocation.

This is the target behaviour the two-phase architecture was meant to enable: capital reserved before the recipient is useful remains available for a later recipient.

### Immediate borderline target

The immediate bridge gives the fixed-next baseline a naturally aligned recipient. Two-phase matches fixed-next under independence, is slightly behind in the shared-uniform cell, and behind in the balanced-negative cell. This is expected: waiting and ranking targets is not free when the immediate next recipient is already useful.

### Negative controls

In far-recipient, no-useful-recipient and switching scenarios, no candidate was fundable under the exact-minimum-credit rule in the pilot. The two-phase method left the source reserve unassigned rather than spreading partial credit that failed to create a discovery. The reserve remains a cost to the donor's compound score, so excessive idle reserve can be a real power cost; this policy is not free.

### Error rates

No false discoveries were observed in these pilot cells, including all-null. This does not establish FDR or SupFDR control. The mathematical claim still depends on globally valid coordinate e-processes, a common filtration, the valid adjuster, fixed globally summable weights, source-only reserve caps, forward-only assignment, and the timewise compound-e-BH self-consistency proof. The simulator cannot authenticate real-world observations, arrivals, or filtration completeness.

## 6. The next research problem: optimize the allocator, not the reserve theorem

The exact-minimum-credit policy answers whether a candidate can be made rejectable with a known amount. It does not identify which fundable candidate gives the highest expected future discovery value.

The next algorithmic line should therefore compare:
1. minimum credit per candidate (implemented);
2. minimum credit divided by a predictable lagged utility score;
3. greedy exact-credit knapsack over a multi-candidate allocation batch;
4. receding-horizon allocation using only a registered utility model, with prior-mixed exploration;
5. a hindsight optimizer only as a non-implementable upper bound.

Every allocator remains subject to the same source-only donor caps, forward prefix budget, immutable assignments, and no incoming re-donation. The utility score must be predictable and cannot encode null labels. Strong projectivity is a separate optional mode.

## 7. Industry comparison and claim boundary

The direct published single-pass baseline remains donation e-LOND by Xu, Fischer & Ramdas (UAI 2026), which has an \(O(\log t)\) rejection-decision algorithm and arbitrary-dependence SupFDR guarantee for one e-value per arriving hypothesis. It is not the same repeated-evidence process contract as this experiment. Dynamic e-closure treats evolving evidence and future-extension coherence; Carefree e-process work requires a valid adjuster for SupFDR rather than raw running suprema.

This pilot is not a matched claim of beating any of those methods. The sample size, limited generator family, one alpha value, fixed hypothesis count and lack of paired confidence intervals preclude it. The next gate is 5,000 exploratory paired streams with all accounting assertions, then at least 50,000 frozen paired streams per confirmatory cell under #9714 and simultaneous uncertainty bounds.

## Claim boundary

This pilot supports one narrow hypothesis: waiting to allocate source-reserved capital can be useful when the recipient becomes borderline only after the reserve has accrued. It does not establish a universal power advantage, FDR/SupFDR theorem beyond #9894, Rust correctness, or SOTA superiority.
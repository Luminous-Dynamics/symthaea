# Exact minimum e-BH credit: an allocation oracle for two-phase escrow

## 1. Objective

The two-phase ledger separates source reservation from recipient assignment. Once reserve is already deducted from donor scores, the assignment policy needs a precise answer:

> What is the smallest additional incoming credit that makes a specific candidate enter the current weighted e-BH rejection set?

A heuristic deficit to the currently observed e-BH threshold is not enough. Adding credit may change the self-consistent rejection count, which changes the e-BH threshold. This note derives the exact finite-family answer for a fixed score vector and level alpha.

This is a deterministic allocation calculation. It does not establish that the score vector is statistically valid; that guarantee comes from the underlying process-valued compound-evidence theorem and its assumptions.

## 2. Definitions

Let the current weighted compound score vector be

\[
z=(z_1,\ldots,z_n),\qquad z_i\ge0,
\]

and let \(\alpha\in(0,1]\). Define

\[
T_k=\frac{1}{\alpha k},\qquad k=1,\ldots,n.
\]

Let \(z_{(1)}\ge\cdots\ge z_{(n)}\) be the descending order statistics and

\[
K(z)=\max\{k:z_{(k)}\ge T_k\},
\]

with \(K=0\) when no rank passes. The step-up e-BH rejection set is the top \(K(z)\) candidates, with a deterministic candidate-ID tie-break.

For target candidate \(j\), define the number of *other* scores meeting the rank-k threshold:

\[
r_{j,k}(z)=\#\{\ell\ne j:z_\ell\ge T_k\}.
\]

Candidate j can enter a rejection set of exactly k hypotheses at threshold \(T_k\) precisely when exactly k-1 other candidates meet the threshold. If fewer meet it, the kth order statistic fails; if more meet it, the resulting step-up set is larger and the relevant larger rank is considered separately. Thus the candidate-specific feasible ranks are

\[
\mathcal K_j(z)=\{k:r_{j,k}(z)=k-1\}.
\]

## 3. Exact minimum-credit formula

If j is already rejected, its required credit is zero. Otherwise,

\[
\boxed{
\Delta_j(z;\alpha)
=
\min_{k\in\mathcal K_j(z)}(T_k-z_j)_+.
}
\]

The feasible set is nonempty. For fixed j, \(r_{j,k}\) is nondecreasing in k, while k-1 increases by exactly one. Hence \(f(k)=r_{j,k}-(k-1)\) can decrease by at most one per step. At k=1, f(k) is nonnegative; at k=n, it is nonpositive. The integer sequence cannot cross from positive to negative without attaining zero.

For the maximum feasible rank \(k^*_j\), the threshold \(T_{k^*_j}\) is the smallest threshold in the feasible set. Since j is not already rejected, \(z_j<T_{k^*_j}\) and

\[
\Delta_j=T_{k^*_j}-z_j.
\]

At this exact credit, candidate j enters the e-BH rejection set. At every smaller breakpoint credit \(T_k-z_j\), it remains outside.

## 4. Batch computation in O(n log n)

The candidate-specific counts can be expressed using the global threshold counts

\[
C_k(z)=\#\{\ell:z_\ell\ge T_k\}.
\]

If \(z_j<T_k\), then \(r_{j,k}=C_k\), so the candidate-specific equality is \(C_k=k-1\). If \(z_j\ge T_k\), then \(r_{j,k}=C_k-1\); any feasible rank in this case means candidate j already lies in the rejection set.

Therefore, for every currently unrejected candidate, the relevant global rank list is

\[
\mathcal S(z)=\{k:C_k(z)=k-1\}.
\]

The candidate-specific feasible ranks are exactly those \(k\in\mathcal S(z)\) for which \(z_j<T_k\).

Algorithm:
1. Sort scores descending in O(n log n).
2. Sweep k=1,...,n. Because T_k decreases, one pointer through the sorted scores computes every C_k in O(n).
3. Store the sorted feasible-rank list S.
4. For each candidate not already rejected, binary-search S for the largest k with \(z_j<T_k\). Return \(T_k-z_j\).

The full vector of minimum-credit amounts is therefore O(n log n), with O(n) memory. The source-bank allocation is a separate prefix-budget step: candidate j can receive only unassigned capital from sources i<j. A greedy fill can rank currently fundable candidates by required amount, then consume source tokens deterministically while preserving each source ID and immutable destination.

The checked-in reference keeps both:
- the exact per-candidate formula (conceptual oracle);
- the batch feasible-rank implementation (O(n log n)).

An independent breakpoint search tests the result by trying all credits 0 and \((T_k-z_j)_+\) and running the actual sorted step-up e-BH rule.

## 5. Integration with two-phase source escrow

At a pre-observation allocation checkpoint:
1. Read the committed score vector through the previous look.
2. Compute the exact required credit \(\Delta_j\) for each unrejected, eligible candidate.
3. Compute its available source bank \(B_j=\sum_{i<j}U_i\), where U_i is unassigned capital reserved from donor i.
4. If \(B_j<\Delta_j\), spend nothing on j; retain the reserve.
5. If \(B_j\ge\Delta_j\), assign exactly \(\Delta_j\) through one or more source-scoped tokens with source i<j. Credit may occur only when the recipient is active.

This is the no-partial-spend rule. It prevents the policy from dispersing small amounts that improve raw scores but do not alter the e-BH rejection set.

The rule is predictable when the score vector, eligible set, and bank state are fixed before the next observations are requested. If it uses histories from higher-ID recipients, it belongs to the adaptive non-projective mode in #9923. The source-cap, own-increment and forward-prefix accounting remain the same; exact horizon projectivity is not claimed.

## 6. Differential validation

File benchmarks/ebh_minimum_credit.py contains:
- exact Fraction e-BH rejection;
- feasible-rank computation;
- O(n log n) batch minimum-credit vector;
- a separate breakpoint-search oracle;
- a source-only bank allocator which refuses partial spending when available capital is insufficient.

File benchmarks/test_ebh_minimum_credit.py covers:
- exhaustive rational score grids for n=1..4;
- randomized rational score vectors for n=5..7;
- threshold ties and inclusive e-BH semantics;
- minimum-credit parity with breakpoint search;
- source-before-destination allocation;
- insufficient-bank no-spend behavior;
- candidate-selection behavior among fundable deficits.

A local independent exact-rational microcheck compared the optimized batch algorithm against the breakpoint oracle on 15,936 candidate/score/alpha cases, including exhaustive n<=4 grids and randomized dimensions up to n=100. All matched. This is a deterministic arithmetic check, not a statistical simulation or Rust benchmark.

## 7. Benchmark plan

Compare immediate fixed-next, static projective priors, heuristic boundary routing, exact-minimum-credit fill-only routing, and prior-mixed fill-only routing on the same valid LR/A2 process paths in #10033. Include:
- immediate borderline candidate;
- late-borderline candidate after reserves have accumulated;
- far-recipient target;
- switching target;
- no-useful-recipient control;
- predictable delayed activation.

Measure SupFDR, fixed-time FDR, terminal power, time-to-first-discovery, exact credit needed, unassigned reserve, assigned/credited amounts, recipient concentration, worst-regime regret, and update/verifier latency. Evaluate FDP after the pre-observation allocation checkpoint as well as after every subsequent evidence update; otherwise the SupFDR estimand can miss a report time.

Use 5,000 exploratory paired streams before freezing a confirmatory grid, then at least 50,000 paired streams per confirmatory statistical cell with preregistered generators and simultaneous uncertainty intervals. Keep the published one-shot donation e-LOND theorem in its own single-arrival lane.

## Claim boundary

This oracle optimizes one deterministic objective: the smallest score credit that makes a specified candidate enter e-BH under a fixed score vector. It does not identify the best candidate to fund, prove power optimality, or alter the compound null-mass theorem. Statistical validity remains conditional on globally valid base e-processes, predictable routing, source-only reserves, forward-only transfers, and the existing timewise self-consistency proof.
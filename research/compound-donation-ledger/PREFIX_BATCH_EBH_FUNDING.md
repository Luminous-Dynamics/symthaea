# Prefix-constrained batch funding for exact e-BH entry

## 1. Problem

At a pre-observation allocation checkpoint, let z be the frozen weighted compound-score vector. The exact minimum additive credit needed to make candidate j enter e-BH is delta_j, computed by the oracle in #10120.

Each source i has unassigned reserve b_i >= 0. Destination j may draw only from earlier sources i<j. We want to select as many currently eligible recipients as possible and fund each selected recipient with exactly its baseline deficit.

For a selected set S, define the prefix demand

\[
D_j(S)=\sum_{\substack{k\in S\\k\le j}}\delta_k
\]

and available source capacity

\[
C_j=\sum_{i<j}b_i.
\]

The set is fundable exactly when

\[
D_j(S)\le C_j\quad\text{for every }j.
\]

Necessity follows because destinations at or before j cannot use sources after j-1. Sufficiency follows from nested source neighborhoods: every earlier recipient's eligible source set is contained in every later recipient's eligible source set. Greedy oldest-source-first assignment realizes any set satisfying these inequalities.

## 2. Maximum-cardinality algorithm

Process destinations in increasing global ID. Maintain:
- a max-heap of currently selected funding deficits;
- the sum of selected deficits;
- the selected candidate IDs.

When destination j has a positive eligible deficit, insert it. Then compare the selected sum with C_j. While it exceeds C_j, evict the largest deficit; ties evict the largest destination ID first.

### Correctness

Induct on j. Before processing j, the current set is maximum-cardinality among feasible subsets of destinations up to j-1. Adding j either preserves feasibility, in which case the cardinality increases by one, or violates the new prefix capacity. If the latter occurs, at least one selected item must be removed. To retain the largest possible number of items, remove only as many as necessary; among single-item deletions, removing the largest deficit minimizes the residual demand. Repeat until feasible. No feasible set can retain one more item than the repaired set at this prefix, because the prefix capacity was violated before each required deletion.

The algorithm is O(n log n): each candidate is inserted once and removed at most once. It maximizes cardinality and, among maximum-cardinality sets, minimizes total selected baseline deficit under the prescribed deterministic tie rule.

### Source assignment

After selection, process destinations in increasing ID. For each destination j, consume remaining source-token balances only from i<j, oldest source first and then source-local token ordinal. Earlier destinations consume only sources that are also eligible for j. Thus the remaining eligible capacity at j is C_j minus earlier selected demands, which is at least delta_j by the prefix condition.

Every fragment keeps:
- source candidate ID;
- source token ID;
- immutable destination ID;
- source-local assignment ordinal;
- exact amount.

No token can be overdrawn, and assignment never changes the donor's already-reserved outflow.

## 3. Relationship to e-BH

The deficits are computed from one frozen score vector. The e-BH rejection set is monotone under coordinatewise score increases: if a candidate is rejected before an allocation, it remains rejected after other scores increase. Therefore, if every selected candidate receives at least its own exact baseline deficit, all selected candidates are rejected after the batch is credited.

This does not mean the procedure is globally cost-optimal. One funded candidate can lower the self-consistent rank threshold and make another candidate reject with less or no additional credit. The baseline-deficit batch ignores that cross-candidate synergy. It is a maximum-cardinality solution for the specified additive deficit model, not an optimizer over every possible joint allocation.

It also maximizes the number of hypotheses made rejectable, not the number of true discoveries. Null labels are never inputs to the selector.

## 4. Executable artifacts

- benchmarks/prefix_batch_ebh_funding.py: exact Fraction selector and source-token assignment.
- benchmarks/test_prefix_batch_ebh_funding.py: brute-force subset oracle and source-flow tests.

A local exact-rational prototype compared the heap selector against brute-force enumeration of every subset on 9,000 random instances with 1–9 destinations. All selected-set cardinalities matched the true subset optimum; each selected set also passed the prefix feasibility check and oldest-source-first assignment. The checked-in test suite mirrors that comparison, but was not run as a repository CI test.

## 5. Benchmark status

A small paired pilot compares no donation, immediate fixed-next, one-target minimum-credit, repeated greedy saturation, and maximum-cardinality batch funding on a conditional-mean-one Bernoulli likelihood-ratio model with an A2 running-maximum adjuster.

Across the 20-stream-per-cell pilot, batch funding did not improve terminal power over one-target greedy in the immediate, late, far-recipient, no-useful-recipient, or switching cases. All observed SupFDR point estimates were zero, but the Hoeffding half-width at n=20 is 27.4 percentage points. This is a weak pilot and is not evidence of calibration or superiority.

The main value of the batch result is algorithmic: it solves the frozen-deficit recipient-set selection problem exactly in O(n log n), rather than searching subsets exponentially. The next statistical step is to design a joint allocator that exploits threshold synergy without using null labels or future observations, and then test whether that added complexity yields actual power.

## Claim boundary

This is a deterministic batch-selection theorem conditional on the exact minimum-credit formula and fixed source balances. It does not prove a new FDR/SupFDR theorem, optimize future power, or beat donation e-LOND/SCORE/dynamic e-closure. Statistical validity remains conditional on the existing globally valid base e-processes, predictable route inputs, source-only reserve caps, forward-only transfers, and time-monotone compound-score argument.
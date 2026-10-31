# Confidence-adaptive CRC: exact recursion, breakpoint capping and duplicate grouping

**Date:** 2026-10-31  
**Status:** 8/8 local regression tests passed on an executable reference copy; synthetic compression pilot matches the ungrouped recursion through two recertification rounds. Not a CIFAR-100 reproduction.

## Research objective

The common-depth CIFAR experiment is reproduced from the published finite-population error-type counts in PR #9358. The next family in Bai & Ramdas (2026), Appendix B.7, is confidence-adaptive: each test feature receives a depth based on its own classifier confidence pair.

For each calibration or query point i, define u_i=1-p_g(x_i), v_i=1-p_f(x_i), with 0<=u_i<=v_i<=1. At threshold t, choose root when t<=u_i, superclass when u_i<t<=v_i, and leaf when t>v_i. The active recursion has to consider every possible nonzero error type for each hypothetical held-out point.

## Exact state representation

For the ungrouped oracle with r active points, a state identifies one held-out point and a type-1/type-2 assignment for every other active point, giving r*2^(r-1) states.

After breakpoint capping, active points with identical (u_i,v_i) are interchangeable within the recursion. Group j has size g_j; represent its assignment by the number of type-2 points, not the full binary assignment. The grouped table cardinality is:

    sum_j g_j * product_{h != j}(g_h + 1)

The implementation retains both paths:
- grouped recursion for scale;
- ungrouped recursion for small exact differential tests.

The grouped-state expander enumerates every point-level assignment represented by a grouped count vector and compares its depth to the ungrouped oracle. It can compare both the initial baseline table and later recertification tables.

## Breakpoint capping

For b=floor(2*((n+1)*alpha-1)), if at least b+2 active points are present, cap all breakpoints above the (b+2)-th smallest active leaf breakpoint tau. The paper establishes that this preserves all CRC baseline depths and all subsequent recertification depths. The implementation has a switch to disable capping and an uncapped oracle for regression parity.

The tests compare all table cells between capped and uncapped recursion at baseline and after three successive recertification steps; comparing only the initial table would not be enough. An additional deterministic randomized property test covers 12 small active-set layouts with repeated pair groups and mixed type-1/type-2 assignments. For every layout, grouped and ungrouped tables match at baseline and after two recertification updates; capped and uncapped tables also match at every cell through those rounds. All intermediate tables remain valid.

## Exact recertification

For grouped state (j,m), test every hypothetical type c in {1,2}. The held-out total includes:
- the hypothetical query with type c;
- every calibration point with type 1;
- every calibration point with type 2.

The per-type slack is s_c=alpha-T_c/(n+1). The update is:

    A_next(D,x) = min(2, a + 2*s_2, 1 + max(a-1,0) + 2*s_1)

All held-out active points and both nonzero hypothetical labels are checked after each round. Type-0 calibration observations remain in the calibration sample-size denominator but contribute zero loss and are not active recursion points.

## Local correctness tests

9/9 tests passed in a local executable reference copy.

The strengthened tests cover:
- the ungrouped state count r*2^(r-1);
- grouped cardinality formula;
- exact grouped-vs-ungrouped baseline table parity;
- grouped-vs-ungrouped parity after one and two recertification rounds;
- exact capped-vs-uncapped parity at baseline and after three recertification rounds;
- validity and nondecreasing depth over eight recertification updates;
- correct query-state construction;
- invalid breakpoint ordering and unsupported error types;
- no cap when the paper's preconditions do not hold.

The committed branch blobs are:
- source: 0517f47ed3098d72901e52197208d276df4e287b
- tests: e4fa2e674a50a13f48b4690ba0821b245bfe4032

The local test run used the matching executable reference copy; the exact fetched GitHub blobs have not yet been run in a separate clean checkout. Remote CI is not claimed.

## Compression pilot

A deterministic synthetic fixture with 12 calibration points and one query point uses four repeated breakpoint groups (u,v) of sizes 4, 3, 3 and 2. One calibration pair matches the query, producing grouped sizes [4,4,3,2] after including the query. With alpha=.25, breakpoint capping threshold is 0.5.

| Metric | Ungrouped | Grouped |
|---|---:|---:|
| Active points | 13 | 13 |
| Table states | 53,248 | 905 |
| Build time | 278 ms | 2.0 ms |

Compression factor: 58.84x on this small fixture.

Every grouped cell was expanded to all 53,248 point-level assignments and compared:
- baseline: zero mismatches;
- one recertification round: zero mismatches;
- two rounds: zero mismatches; maximum floating-point difference 4.44e-16.

Both grouped and ungrouped tables passed every hypothetical-label validity check at all three stages.

This validates the compression algorithm on a repeated-breakpoint synthetic case, not its asymptotic cost or performance on CIFAR data. Actual benefit depends on repeated/quantized confidence pairs and the breakpoint cap.

## What remains blocked

The paper's confidence-adaptive CIFAR result needs individual frozen ResNet-18 probabilities p_f(x_i), p_g(x_i), labels, class hierarchy and exact paired split assignments. The common-depth type histogram is insufficient to reconstruct them. The confidence-adaptive benchmark cannot be honestly claimed until those artifacts are found or regenerated using a pinned training/inference recipe and split manifest.

Paper reference at alpha=.10:
- envelope CRC: risk 0.09052, correct specificity 67.572%, mean node size 19.40;
- 500 recertification rounds: risk 0.09880, correct specificity 68.235%, mean node size 17.03.

These are not reproduced by this branch.

## Next gates

1. Run the exact committed source and tests in a clean branch checkout and preserve hashes.
2. Add property tests over random finite active sets, repeated pairs, ties, and all cap preconditions.
3. Obtain the original confidence cache/split manifest or regenerate frozen ResNet-18 predictions and IDs.
4. Execute the paper's 300 split by 16 query protocol across alpha .02/.05/.10/.15 and recertification horizons 0/1/3/10/20/50/100/200/500.
5. Compare with MAPIE's ConditionalExpectedRiskController under the same expected-risk, calibration split, frozen model and action rules.
6. Report grouped table size, cap rate, build time, per-query latency, memory and empirical expected risk. Keep finite exchangeability and out-of-assumption shift stress separate.

Reference: Bai & Ramdas (2026), arXiv:2610.35821, Appendix B.7: https://arxiv.org/abs/2610.35821

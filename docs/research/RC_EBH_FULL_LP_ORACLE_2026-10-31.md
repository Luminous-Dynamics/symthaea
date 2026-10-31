# RC-eBH exact full-subsets LP oracle

**Date:** 2026-10-31  
**Status:** 6/6 local tests passed; 200 random full-LP comparisons for K=1..5 and 60 additional comparisons for K=6..8. This is independent finite computation, not a proof of the general optimality theorem.

## Why this check matters

The RC-eBH paper claims that the greatest-convex-minorant construction maximizes expected rejection count among all monotone randomized procedures certified by the closed-eBH mean e-collection. A recent AI-generated literature review (not a human peer review) highlighted the duality argument for Proposition 4.2(iii) as the most intricate load-bearing step and suggested an exact K=5 LP check. We implemented that check independently rather than relying on the review's conclusion.

## Full LP constructed from first principles

For ordered e-values e[1]>=...>=e[K], let kappa be a randomized rejection count, q_r=P(kappa>=r), c_r=E[1{kappa>=r}/kappa], and S_r=sum_{j>=r}c_j with S_{K+1}=0.

Then:
- c_r=S_r-S_{r+1};
- q_r=r*S_r-(r-1)*S_{r+1};
- monotone rejection tails are represented by nonnegative, nonincreasing, convex S;
- E[kappa]=sum q_r=S_1+2*sum_{r=2}^K S_r.

For every nonempty subset T of ranks, the closed-eBH certificate is imposed explicitly:

    sum_{r in T}(S_r-S_{r+1}) <= alpha * mean(e_r : r in T)

The LP includes all 2^K-1 subsets, not merely suffix subsets or a cutting-plane approximation. It also imposes S_1<=1 and the distribution-shape constraints. Objective: maximize E[kappa].

RC-eBH is calculated independently from the paper's greatest convex minorant formula. Its tails are checked against all subset constraints, then compared with the full LP optimum.

## Executed results

The suite tests 200 random e-value families (40 each for K=1,2,3,4,5) and 30 more families (10 each for K=6,7,8), with alpha drawn from {.01,.025,.05,.10,.20,.50}. An additional out-of-suite run checked 60 K=6..8 cases.

| Check | Result |
|---|---:|
| Local tests | 5/5 passed |
| Random LP families | 200 |
| Maximum absolute objective gap | 4.44e-16 |
| Maximum difference between RC and LP tails | 3.33e-16 |
| Maximum full-subset violation of RC-eBH (K<=5) | 3.33e-16 |
| Maximum explicit LP constraints (K<=5) | 41 |

Additional K=6..8 run:
- 60 cases, 20 per K, in 0.20 seconds;
- maximum objective gap: 8.88e-16;
- maximum tail difference: 7.77e-16;
- maximum closure violation: 2.22e-16;
- maximum LP constraint count: 271.

Boundary tests include all-zero and tied vectors, a single large e-value, a log-spread from 1e-8 to 1e8, and mixed zero/positive e-values. All matched.

The full LP's rejection-count distributions were nonnegative, summed to one, and had nonincreasing tails in every test.

## Interpretation

For these 200 cases, the LP optimum and the RC-eBH formula agree to floating-point precision, and RC-eBH satisfies every intersection constraint. This is good independent computational evidence for Proposition 4.2(iii), and it gives us a strong small-K regression oracle for future changes.

It does not prove the theorem for arbitrary K, does not replace the paper's analytical proof, and does not validate any sequential/SupFDR claim. In particular, it should not be described as a general theorem proof or a human peer review.

## Reproduction

    PYTHONPATH=validation python -m unittest -v test_rc_ebh_lp_oracle.py

Committed artifacts:
- validation/rc_ebh_lp_oracle.py
- validation/test_rc_ebh_lp_oracle.py
- validation/RC_EBH_FULL_LP_ORACLE_2026-10-31.json
- docs/research/RC_EBH_FULL_LP_ORACLE_2026-10-31.md


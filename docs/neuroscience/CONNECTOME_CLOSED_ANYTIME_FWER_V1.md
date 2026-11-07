# Closed anytime-FWER for hierarchical phase-response mechanisms V1

**Tracking:** #10960 — SCI-CONNECTOMICS-101  
**Composes:** #10898 single-claim e-process, #10948 / #10959 flat phase-response FWER, and the general multiple-testing work in #9110.  
**Guarantee:** finite-family strong anytime-FWER by complete closed testing, assuming elementary e-process validity in one shared global filtration.  
**Status:** exact synthetic intersection-mixture and closure auditor; no biological result is claimed.

## 1. Scientific motivation

The six-leg phase-response programme has mechanistic structure: local rhythm reset, inter-leg transfer, context dependence and candidate sensory/central/biomechanical mechanisms. A flat contrast ledger is a useful safety backstop, but it cannot by itself explain which mechanistic combinations are supported.

A tree or diagram does not establish logical implication. An internal mechanism null must be defined as a precise intersection of child nulls, or have a separately justified custom null set. Union, disjunction, mediation and difference-in-differences hypotheses are not silently treated as intersections.

## 2. V1 intersection e-process

Let H_1,...,H_m be elementary nulls and E_i,t their nonnegative test supermartingales in one common global filtration, each starting at one.

For every nonempty subset I define H_I as the intersection of its elementary nulls. Use fixed weights lambda_{I,i} >= 0 summing to one and frozen before outcomes, and define:

    E_I,t = sum_{i in I} lambda_{I,i} E_i,t

Under H_I, every H_i in I is true. Each component process is therefore a supermartingale in the shared filtration, and the fixed convex mixture E_I,t is also a nonnegative test supermartingale starting at one.

The V1 reference uses equal mixture weights lambda_{I,i}=1/|I|. The exact verifier enumerates every nonempty subset (at most 8 leaves / 255 intersections).

Reject H_I when its running supremum reaches 1/alpha. Reject elementary H_i only when **every** intersection containing i has been rejected.

## 3. Strong anytime-FWER guarantee

Let T be the set of all true elementary nulls. If any true H_i is rejected, closed testing requires rejection of H_T, the intersection of all true nulls. Under H_T, E_T,t is a valid test supermartingale. Thus:

    P(any true elementary rejection)
      <= P(sup_t E_T,t >= 1/alpha)
      <= alpha

by Ville's inequality. This requires arbitrary-dependence robustness only through validity of every component in the same global filtration; it does not require independence among claims.

The running supremum is used for threshold crossing only. It is not converted into an e-value and is not sent to e-BH.

## 4. Exact synthetic known answer

At alpha=1/5, threshold = 5. Elementary paths:

| Null | E-process path |
|---|---|
| H1 local reset | 1, 5, 20 |
| H2 off-diagonal transfer | 1, 4, 15 |
| H3 contact modulation | 1, 3/2, 2 |

For equal-weight mixtures, stage-two intersection e-values are:
- {H1}: 20
- {H2}: 15
- {H3}: 2
- {H1,H2}: 35/2
- {H1,H3}: 11
- {H2,H3}: 17/2
- {H1,H2,H3}: 37/3

All intersections containing H1 or H2 cross by stage 2, so H1 and H2 are rejected. The singleton H3 does not cross, so H3 remains unrejected. This is an arithmetic software fixture, not an empirical type-I estimate.

## 5. Limits and complexity

- V1 supports at most 8 elementary claims; larger families return NOT_SUPPORTED.
- Complete closure is exponential (2^m - 1). Do not silently drop subsets.
- Every claim must share the same global-filtration and confirmation-split commitments.
- Fixed mixture weights cannot be updated from outcomes.
- A custom intersection e-process may be used only with its own theorem-matched validity receipt.
- Custom non-intersection mechanistic hypotheses need separate proofs and are not inferred from parent/child names.
- V1 does not provide FDR or FDR-sup.

## 6. Benchmark comparisons

Compare separate guarantees:
- flat weighted anytime-FWER backstop (#10948 / #10959);
- closed anytime-FWER V1 using intersection e-process mixtures;
- closed testing / Holm / weighted Bonferroni / graphical gatekeeping for fixed-horizon tests;
- Carefree A1/A2 adjuster-based FDR-sup from #9110 under its own theorem;
- asymptotic online FWER under dependent statistics (Jankovic, Fischer & Brannath, 2026), not conflated with finite-sample anytime validity.

The closed-testing principle, graphical gatekeeping and online closure are established ideas. The engineering contribution is exact composition with source-bound connectome claim identities, common-filtration receipts and persistent anytime decisions.

## 7. Claim ceiling

A passing contract establishes exact mixture and closure semantics for the finite synthetic family, conditional on upstream elementary e-process validity and exact null intersection semantics. It does not authenticate those processes, identify true biological mechanisms, establish direct synaptic causation, or establish FDR/FDR-sup.

# Set-valued connectome hypothesis coverage benchmark V1

**Tracking:** #10709 — SCI-CONNECTOMICS-094
**Composes:** #10656, #10657, #10661, #9204, #9248, #9255
**External comparator:** HypoSpace (Chen et al., ICML 2026 spotlight; PMLR 306:15451–15465)
**Status:** synthetic known-answer contract; no biological coverage result or model ranking is claimed.

## 1. Scientific question

Can a candidate-discovery system return a broad, non-redundant set of *admissible* connectome reconstruction alternatives as the finite hypothesis space grows—and does it recover the uncommon alternatives that change the scientific query?

This separates four questions often hidden by a single model score:
1. Are proposals admissible?
2. Do proposals represent distinct biological alternatives rather than syntactic variants?
3. How much of the admissible set has been recovered?
4. Have the query-critical alternatives been recovered?

## 2. Relation to HypoSpace

HypoSpace treats model outputs as samples from finite hypothesis spaces and reports Validity, Uniqueness and Recovery over deterministic, exactly enumerable suites. It shows that high validity can coexist with declining uniqueness and recovery as the admissible space grows. We adopt this as an external comparator, not as a novelty claim.

Our connectomics increment is to canonicalize hypotheses into independently adjudicated biological equivalence classes, measure query-critical and impact-weighted recovery, bind provenance and channel ancestry, and connect missed alternatives to downstream query/certificate effects. The benchmark therefore measures more than syntactic diversity or generic solution coverage.

## 3. Experimental unit and canonicalization

The experimental unit is a frozen query + evidence context + hypothesis universe + candidate-channel run. Every proposal must resolve to a canonical candidate identity or be retained as an unknown proposal. Candidate identities are grouped into equivalence classes using a frozen biological equivalence/adjudication rule before uniqueness or coverage is calculated.

Two candidate records with different IDs but the same equivalence class are not two recovered biological alternatives. An inadmissible or indeterminate proposal cannot increase admissible recovery. A model must not define its own equivalence classes after seeing the score.

## 4. Exact synthetic metrics

Given a finite oracle universe with admissible equivalence-class set A and query-critical subset C:

- **Validity / proposal precision:** count of proposals whose candidate identity is adjudicated ADMISSIBLE divided by all emitted proposals. Also report INADMISSIBLE, INDETERMINATE and unknown proposal counts separately.
- **Uniqueness:** distinct admissible equivalence classes represented divided by admissible proposal count. This penalizes repeated proposals and near-duplicate candidate identities that canonicalize to the same biological alternative.
- **Recovery:** distinct admissible equivalence classes represented divided by |A|, the exactly enumerated admissible set.
- **Query-critical recovery:** distinct recovered classes in C divided by |C|. C includes a class when it flips the frozen query decision or invalidates the frozen policy.
- **Impact-weighted critical recovery:** recovered critical-class impact weight divided by total impact weight across C. Weights are frozen by independent adjudication; generator confidence/frequency is not impact.
- **Mode concentration:** largest proposal share occupied by one admissible equivalence class. Report alongside proposal-class entropy; neither is a completeness certificate.
- **Coverage curve:** report all metrics at each emitted-proposal budget K, not only at a terminal K. Also report total candidate-generation compute and adjudication/reviewer effort.

Use a fixed denominator from the exact oracle. Do not use the channel's own proposal pool as the truth set. Do not use a macro-average across different cardinalities without also showing per-case curves.

## 5. Biological sampled-frame metrics

An actual biological connectome does not have an enumerated ground-truth alternative set. Do not report exact Recovery as though it were known. For biological tasks:

- retain proposal precision/validity only where independent adjudication supports it;
- estimate candidate and query-critical recall only under #9204's independently frozen sampling frame, PSU design, inclusion probabilities and audit coverage, including zero-proposal PSUs;
- preserve the observed union count separately from #9248's model-dependent all-list-missed sensitivity;
- retain indeterminate, disputed, inaccessible and incomplete searches as separate outcomes;
- do not treat a stable global graph invariant or zero observed proposals as proof that no query-critical alternative exists.

## 6. Required baselines and controls

Compare under matched proposal K, evidence access, compute, reviewer time and lockbox rules:

- every individual candidate channel;
- the union of genuinely independent channels;
- a random admissible-proposal baseline;
- a diversity/stratified sampler;
- a query-criticality-aware sampler;
- a HypoSpace-style set-valued sampler adapter (exact version and configuration pinned);
- the exact synthetic oracle upper bound.

Controls include repeated single-mode proposals, syntactic duplicates within one equivalence class, rare high-impact alternatives, query-equivalent but structurally distinct alternatives, invalid candidates, indeterminate adjudications, unknown candidate IDs, shared-source channels and held-out scale tiers.

## 7. Known-answer fixture

The included fixture has three admissible biological equivalence classes and one query-critical class. Two distinct candidate IDs map to the same admissible class. Additional candidates are inadmissible and indeterminate; one proposal is unknown to the frozen universe.

At K=5, the mode-collapsed channel emits three admissible proposal records but covers only two of three admissible classes and misses the only query-critical class. The query-diverse channel covers all three classes and the critical class. Their valid-proposal precision is not identical to their recovery, which is exactly the diagnostic separation this benchmark requires.

The pooled union reports both the increased set coverage and its duplicated proposal burden. It does not receive independence credit merely because it contains multiple channels.

## 8. Scaling and promotion targets

Run cardinality tiers that increase at least tenfold while holding proposal, compute, review and evidence budgets fixed. Report raw curves, marginal new-class yield per proposal/cost, mode concentration, query-critical tail misses and uncertainty. Fit scaling slopes only where the sample sizes and tier counts justify it; do not assert a universal power law from a few synthetic points.

Aspirational promotion targets (not current results):
- 100% metric agreement with exact synthetic oracles;
- zero false query-resolution certificates with an admissible opposing class still compatible;
- at least 95% query-critical recovery on each bounded synthetic promotion tier at the declared budget;
- no more than a 10 percentage-point query-critical-recovery drop under a tenfold hypothesis-space increase at matched total resources;
- at least 20% more query-critical impact recovered per reviewer-hour than the strongest frozen single-channel baseline in a blinded trial;
- zero independence promotions when channel ancestry controls identify shared source, preprocessing, model or reviewer lineages.

Promotion requires uncertainty bounds and per-stratum results. These are targets, not biological performance claims.

## 9. Claim ceiling

Exact Recovery applies only to genuinely enumerable synthetic hypothesis sets. Biological discovery is limited to the declared source frame and sampling design, with an explicit all-list-missed sensitivity envelope. No finite proposal set proves global connectome completeness. No metric in this benchmark establishes causal synaptic transmission or biological truth without independent source evidence and intervention-grounded validation.
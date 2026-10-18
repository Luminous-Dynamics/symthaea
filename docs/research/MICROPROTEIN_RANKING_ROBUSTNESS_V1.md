# MICROPROTEIN-RANKING-001 — Missingness-aware microbial-protein candidate selection

**Status:** synthetic decision benchmark; not organism/strain selection, feeding advice, safety assessment, or production recommendation.  
**Version:** 1.0.0  
**Created:** 2026-10-18

## Research motivation

A 2026 *npj Science of Food* review compared 57 bacterial single-cell-protein (SCP) candidates across safety, amino-acid composition/quality and production efficiency. It reports that 16 candidates had QPS status and 33 had published amino-acid composition data; it also notes that the relevant source literature and data were incomplete for many candidates. The paper explicitly treats safety, protein quality and production efficiency as separate dimensions, and notes that even an amino-acid score is only one aspect of nutritional performance.

Source: Takegawa R, Kono N. “Species selection criteria for sustainable bacterial single cell protein.” *npj Science of Food* 10, 231 (14 May 2026): https://www.nature.com/articles/s41538-026-00828-1

The research lesson is not “rank these organisms from a public table.” It is to make missingness and uncertainty visible, require fit-for-purpose permissions and safety eligibility before ranking, and only call a candidate dominant when the evidence supports that comparison.

## Paradigm shift

**Replace one weighted score with an evidence-gated, uncertainty-aware Pareto frontier.**

A single scalar score can hide critical weaknesses: a low cost may offset a missing required amino-acid profile; high nominal protein may hide low recovery; a simulated energy estimate may be mistaken for measured performance; and an unknown safety or permissions status may be drowned out by favorable metrics.

This benchmark uses three steps:

1. **Hard-gate eligibility first.** Source rights, declared target-scope match, safety, authorization and required nutrition gates are non-compensatory. A fail blocks; an unknown required hard gate is indeterminate.
2. **Require evidence coverage for all declared objectives.** Missing/conflicting/stale required values remain unknown. They are not zero, average, median, or implied by a different organism, strain, process or species.
3. **Report robust dominance, not forced total order.** Candidate A robustly dominates B only if A is no worse on every declared objective across non-overlapping uncertainty intervals and strictly better on at least one objective. If intervals overlap or dimensions are missing, both can remain on the frontier or the cohort must abstain.

For minimizing metric (m), a conservative sufficient condition for A to dominate B is (A_m^{upper} \le B_m^{lower}). For maximizing metric (m), it is (A_m^{lower} \ge B_m^{upper}). The condition must hold across all declared objective metrics; one attractive metric is not enough.

## Synthetic test artifacts

- `microbial-protein-ranking-case-schema-v1.json`: cohort, hard gates, objective profile and interval data schema.
- `microbial-protein-ranking-corpus-schema-v1.json`: corpus container schema.
- `microbial-protein-ranking-corpus-v1.json`: 10 synthetic ranking cohorts.

Every candidate ID, value, bound, source reference, scenario digest, and gate in the corpus is synthetic. No real microbial strain record, amino-acid table, culture parameter, safety status, or production yield is included.

The corpus includes:
- robust dominance with a blocked unsafe candidate that otherwise looks excellent;
- unresolved safety and nutrition gates;
- required amino-acid or resource metrics missing;
- known target-scope mismatch;
- missing permission evidence;
- overlapping uncertainty intervals with no unique winner;
- scenario mismatch;
- all candidates missing required data;
- failure of a hard nutritional constraint that cannot be compensated by cost or environmental metrics.

## Expected decision semantics

Candidate states:
- **eligible:** all hard gates and required metric evidence pass for the declared synthetic comparison;
- **blocked:** known hard-gate failure, source-rights block or known target-scope mismatch;
- **indeterminate:** unresolved required gate or missing/conflicting/stale required metric;
- **not comparable:** incompatible scenario digest, units, source definition or objective profile.

Cohort states:
- **robust frontier:** eligible candidates can be compared under one aligned objective profile; record all frontier and dominated IDs, including ties.
- **incomplete:** unresolved candidate status could change the cohort conclusion; any listed frontier is provisional.
- **abstain—insufficient evidence:** no candidate has the complete required metrics for this decision.
- **not comparable:** cohort inputs do not share the same scenario/objective context.
- **abstain—no eligible candidates:** no candidate passes the hard gates.

Known-invalid candidates can be excluded while ranking remaining eligible candidates, but their exclusion reasons must remain in the receipt. An unresolved hard gate is different: it prevents a global-best claim because the candidate's status is not yet determined.

## Benchmark targets

- Zero hard-gate false accepts across the frozen corpus.
- 100% exact agreement with independent expected candidate states, frontier members, dominated members, cohort status and reason codes.
- Zero silent imputation of missing values or missing permissions.
- Zero Pareto dominance decisions when any required metric is missing, conflicting, stale, unit-mismatched, or scenario-mismatched.
- 100% of compared candidates share the same declared scenario digest and objective profile.
- For interval data, no false robust dominance when the uncertainty intervals overlap on a required objective.
- Deterministic results under candidate, metric and case ordering permutations.
- Receipts include source/scenario digest, exact Git HEAD, evidence classes, gate dispositions, objective units/directions, frontier, exclusions, missingness summary and claims not established.
- No real organism/strain called safe, nutritionally adequate for a particular animal, authorized, or commercially optimal based on this benchmark.

## Implementation boundaries

- Symthaea owns evidence-aware candidate eligibility, robust frontier logic and abstention semantics.
- Reuse the existing nutrition source registry and general bioprotein gate contracts; do not create a parallel nutrition database.
- Symtropy remains owner of exact resource accounting and full-process scenarios.
- Mycelix remains owner of provenance, permissioned sharing and source/batch/assay lineage.
- Sol Atlas may later supply regional energy/water/feedstock context; a mapped resource does not imply availability or permission.
- Real candidate data should enter only after species/strain identity, permissions, measurement methods, target use and jurisdictional safety scope have been verified.

## Non-goals

No lab protocol, strain engineering, live process control, feeding recommendations, data scraping without rights review, safety certification, or operational production choice. The software can verify declared comparison semantics; it cannot authenticate the underlying data or prove biological truth.

## References

- Takegawa & Kono (2026), *npj Science of Food*: https://www.nature.com/articles/s41538-026-00828-1
- Motta de Matos & Cruz (2026), short-term microbial-protein pet-food preference study: https://doi.org/10.1371/journal.pone.0356767
- Precision fermentation TEA/LCA review, *Trends in Food Science & Technology* (2026): https://doi.org/10.1016/j.tifs.2025.105488

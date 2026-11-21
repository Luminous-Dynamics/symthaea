# ds006623 Weighted-Jaccard Reanalysis — Intake Audit V1

**Status:** Metadata and source-code audit only. No participant data have been downloaded or analyzed by this work item.

## 1. Why this is a second target, not a replacement for ISD

The existing CETB ds006623 plan targets the published integration–segregation difference (ISD) analysis. A separate 2026 NeuroImage: Reports article reanalyses the same dataset using full correlation-matrix Weighted Jaccard (WJ), a within-condition floor and group-averaging controls.

These are distinct estimands and analysis pipelines:

- ISD reduces network integration/segregation properties to declared graph metrics over thresholded matrices.
- WJ compares full edge-weight architectures and calibrates between-state similarity against same-state repeatability.

The two methods must be reproduced separately. A WJ reproduction does not reproduce ISD; neither is an independent dataset replication.

## 2. Pinned public sources

### Dataset

- OpenNeuro ds006623 v1.0.0
- DOI 10.18112/openneuro.ds006623.v1.0.0
- CC0 as reported by its dataset descriptor and OpenNeuro release.
- 26 healthy volunteers in the release.

### WJ paper and code

- Paper: Harbert (online 2026-10-04), NeuroImage: Reports, DOI 10.1016/j.ynirp.2026.100417.
- Code repository: https://github.com/nwharbert8-ui/brain-connectivity-wj
- Pinned repository commit: 24bce9bda952311e33369422801acb85a88efcde.
- Repository licence: MIT.
- Primary revised analysis: revision/revision_final_trimmed.py.
- Group aggregation control: revision/revision_group_artifact_fisherz.py.
- Revision notes: revision/README.md.

The revision README says revision_final_trimmed.py produces the primary revised numbers and records the three methodological cautions: no odd/even volume interleaving, equal segment lengths and consistent alphabetical ROI ordering.

## 3. Cohort denominators

Do not flatten the denominators into one participant count:

- Dataset release: 26 volunteers.
- The WJ paper reports two exclusions for incomplete conditions, leaving N=24 for the main analysis.
- The post-LOR trim and requirement for two 300-volume segments in all three conditions leave N=23 for the within-condition floor comparisons.

The exact participant identifiers and reason codes must be read from the author outputs and dataset metadata during reproduction. Do not infer them from a figure.

## 4. Published method details to reproduce

The pinned author revision script declares:

- Conditions: rest_run-1 (awake), imagery_run-2 (post-LOR unresponsiveness), rest_run-2 (early recovery).
- Post-LOR trimming applied to the unresponsiveness run.
- 456 ROI timeseries from the 4S456Parcels XCP-D derivative.
- Spearman correlation matrices, common ROI set and alphabetical ROI ordering.
- 300-volume segments and five repeated placements per participant.
- Two consecutive, non-overlapping segments for each within-condition comparison.
- Absolute, positive-only, negative-only and signed-shift WJ variants.
- Direct sign-flip rates at minimum edge magnitudes 0.0, 0.1, 0.2 and 0.3.
- Net-versus-gross uniformity index.
- Subject-level recovery fraction and bootstrap confidence interval.
- Group averaging by Fisher z transform and a subject-identity permutation control.

These are source-code observations, not an independent rerun.

## 5. Reported paper values to verify, not hard-code

The paper reports a pooled within-condition WJ floor of 0.544 and lower between-state similarities, with a derived normalized excess-distance share of approximately 11–14%. The paper explicitly revises an earlier interpretation that compared between-state values to the perfect-match ceiling.

The normalized 11–14% quantity is a descriptive normalization against the within-state reference. It is not a general variance decomposition, causal attribution fraction or percentage of brain tissue/edges changed.

The reported minimum p-values are limited by the small participant count. The paper itself notes that some two-sided paired tests saturate the exact lower-resolution boundary for N=23. CETB must report paired effect sizes and uncertainty alongside p-values.

## 6. Independent-value assessment

The WJ analysis is methodologically useful because it introduces a measurement-repeatability reference, sensitivity to sign changes, an aggregation control and a length-matched design.

It does not independently validate the biological effect because it uses the same ds006623 cohort. Its own participant count and data lineage overlap with the ISD work. Treat it as a same-dataset analytical comparator, not an external replication.

The RCAF extension should test:
- per-condition floors rather than assuming one pooled floor is adequate;
- segment-duration and guard-gap sensitivity;
- repeated-segment dependence;
- Pearson/Spearman and available XCP-D preprocessing variants;
- subject-level versus group-level conclusions;
- uncertainty clustered at the participant level.

## 7. Execution status

| Task | State |
|---|---|
| Paper metadata / DOI inspected | Complete |
| Author repository README / revision notes inspected | Complete |
| Author code and licence inspected | Complete |
| Author commit pinned | Complete |
| Dataset files downloaded | Not done |
| Author's revised pipeline independently replayed | Not run |
| Published values reproduced | Not run |
| RCAF extensions run | Not run |

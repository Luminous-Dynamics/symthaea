# Query-critical omitted-hypothesis discovery benchmark V1

**Tracking:** #10656 — SCI-CONNECTOMICS-089  
**Parent implementation:** #10603 — adversarial hypothesis-space expansion  
**Composed owners:** #9204, #9248, #9242, #9260, #9344, #9255, #10414, #10416, #10433  
**Status:** benchmark contract and synthetic fixture; no biological evaluation or performance result is claimed.

## 1. Research question

How reliably do independently governed challenge channels discover reconstruction alternatives that were absent from a frozen connectome hypothesis set and that matter to a specific scientific query?

The benchmark connects two currently separate capabilities:

- #10603 can evaluate what happens when an independently supplied alternative reconstruction is appended to the hypothesis universe.
- #9204 can estimate candidate-generation recall outside the frozen proposal pool using an independent probability sample of the declared spatial/neuron population.

The missing evaluation bridge is whether the alternatives found by those independent searches are actually query-critical, and whether shared discovery blind spots make apparent channel diversity misleading.

## 2. The central distinction

A candidate-generation system can have high recall for locally visible proofreading errors while missing a rare reconstruction alternative that reverses a high-consequence scientific conclusion.

Conversely, a newly found biologically admissible graph change may leave a particular query unchanged. Both matter, but they are different endpoints.

The benchmark therefore keeps the following denominators separate:

1. **Adjudicated reconstruction alternatives:** all independently confirmed alternatives found in the audited population.
2. **Query-critical alternatives:** alternatives that change the frozen query decision, invalidate the baseline adaptive policy, or increase its exact minimum worst-case review effort.
3. **Impact-weighted query-critical alternatives:** the query-critical subset weighted under a loss/impact rubric frozen before outcomes.
4. **All-list-missed sensitivity:** alternatives that may remain undiscovered by every observed channel, estimated only under the explicitly declared #9248 model set.

Never substitute candidate precision, candidate-pool recall, or model confidence for query-critical recall.

## 3. Composition architecture

This benchmark is an integration layer, not a replacement for its dependencies.

| Concern | Canonical owner | This benchmark consumes |
|---|---|---|
| Complete eligible spatial/neuron PSU frame | #9260 and #9204 | Frame and sampling-protocol digests |
| Candidate-generation recall outside proposal pool | #9204 | Frozen proposal pool, sample and design-based report |
| All-channel missed-error sensitivity | #9248 | Capture histories and model-sensitivity report |
| Query and plausible reconstruction semantics | #9344 | Frozen query and base hypothesis-set digest |
| Appended-world policy consequences | #10603 | Exact challenge reports and expanded-world digests |
| Query-scoped certificate state | #9255 | Dependency/certificate impact reference |
| Provenance assertion scope and auditability | #10414, #10416, #10433 | Frame-closure and provenance-audit references |
| Prospective workflow treatment effect | #9242 | Preregistered protocol and later outcome report |

Do not duplicate the PSU sampler, finite-population estimator, multiple-systems estimator, fusion-frame closure verifier, or certificate authority.

## 4. Protocol

### 4.1 Freeze the query and baseline

Before any challenge channel runs, freeze:

- source volume and annotation/segmentation revision;
- scientific query specification, implementation and digest;
- query dependency scope and decision threshold/loss;
- base reconstruction-world set and its digest;
- joint observation support, action universe and action costs;
- baseline exact adaptive and fixed-plan outputs from #10603;
- all source, pipeline, model, reviewer and acquisition ancestry relevant to the challenge channels.

The query and policy must not change after challenge proposals are visible.

### 4.2 Freeze discovery channels independently

Candidate channels may include a production detector, an independently developed EM model, morphology/mesh analysis, skeleton continuity analysis, synapse polarity review, orthogonal imaging, and separate blinded human review teams.

For every channel, preserve its source-data lineage, training/pretraining ancestry, preprocessing, model and configuration digests, reviewer exposure, referral paths and acquisition lineage.

Different model names or modalities do not establish independence. Channels sharing a volume, segmentation, mesh derivation, training set, reviewer suggestion or acquisition session must retain those dependence groups.

No channel may see another channel's candidate identities or sampled locations before its own output is frozen.

### 4.3 Independently sample the declared population

Use #9204 and #9260 as the owners of the eligible PSU frame and one-stage stratified SRSWOR design. Include zero-proposal PSUs. Do not construct the audit frame from the union of channel proposals.

The benchmark records the exact frame, protocol, candidate-pool and sample-report digests. Every sampled PSU receives one disposition:

- COMPLETE_SEARCH;
- INCOMPLETE_SEARCH;
- SOURCE_INACCESSIBLE.

Only a completed search can support a negative finding. Incomplete and inaccessible PSUs remain in the denominator and block promotion rather than being silently removed.

The statistical estimator and inclusion-probability arithmetic remain with #9204/#10416.

### 4.4 Blindly adjudicate alternatives

Independent adjudicators determine whether a proposal corresponds to a real, biologically admissible reconstruction alternative. Adjudicators must not see channel identity, rank, confidence, treatment assignment, or whether the alternative changes the query.

Preserve confirmed matches, false positives, ambiguous matches, disputed alternatives and inaccessible evidence as separate states. Every proposal disposition binds an adjudication receipt; a confirmed proposal-to-alternative link additionally binds a blinded match-adjudication receipt. Every alternative binds its independent biological-adjudication receipt and its before/after query decision. Do not force ambiguous identity linkage into a confirmed match.

Every confirmed alternative binds its graph digest, source evidence, adjudication protocol, dependence ancestry, and exact #10603 challenge report.

### 4.5 Evaluate query criticality

For each confirmed alternative, append its reconstruction world to the frozen base using #10603's append-only semantics. Never alter an existing world, query, action, cost or joint-support tuple.

Classify its query consequence from the independently verified challenge report:

- QUERY_DECISION_FLIP;
- ADAPTIVE_POLICY_INVALIDATED;
- ADAPTIVE_COST_INCREASED;
- NO_QUERY_EFFECT;
- NOT_ESTIMABLE.

The categories are mutually exclusive for the primary report, using the first applicable severity in the order above. Preserve the underlying before/after values so the categorization can be independently recomputed.

A world that is biologically admissible but not query-critical is still recorded. It must not be omitted simply because it does not hurt the current query.

## 5. Estimands and metrics

Let the independently adjudicated alternatives in the design-audited population be the reference set. A proposal is counted as a channel capture only after blinded identity adjudication confirms the match.

Report, separately:

- design-weighted candidate recall for all adjudicated alternatives;
- design-weighted query-critical alternative recall;
- impact-weighted query-critical recall under a frozen impact rubric;
- per-channel precision, recall, and proposal burden;
- union-of-channels recall;
- per-error-family, compartment, species, region, image-quality and source-volume estimates;
- exact #10603 policy cost increase and policy-feasibility loss caused by newly admitted worlds;
- reviewer search, adjudication, acquisition, inference, compute, storage and rework costs.

For a design-weighted recall, use the numerator and denominator estimators from the audited PSU design, not a naive ratio over candidate rows. Report the design, inclusion probabilities, PSU denominators, design-aware uncertainty and all nonresponse dispositions. If the design does not support an interval or a subgroup estimate, report NOT_ESTIMABLE.

All-list-missed sensitivity is a separate #9248 output. The observed union of lists is not ground truth, and a capture-recapture/MSE estimate is not ground truth either. Preserve the observed-union estimate and the MSE model-sensitivity range side by side. Return NOT_IDENTIFIABLE when defensible model specifications produce materially incompatible estimates.

## 6. Synthetic known-answer fixture

The checked-in fixture is explicitly synthetic and has four sampled PSUs, three candidate channels and three independently adjudicated alternatives.

- PSU-001 contains a query-flipping alternative found by the EM and morphology channels.
- PSU-002 contains a graph-changing alternative with no effect on the frozen query.
- PSU-003 has zero proposals from every candidate channel, but the independent audit discovers a high-impact alternative that invalidates the baseline adaptive policy.
- PSU-004 is a complete zero-proposal control with no adjudicated alternative.

The EM and morphology channels share raw-volume ancestry. Their two detections of PSU-001's alternative count as two channel captures but do not establish independent evidence.

Expected synthetic outcomes:

- 3 alternatives independently adjudicated;
- 2 query-critical alternatives;
- 1 query-critical alternative found by the candidate channels;
- 1 query-critical alternative missed by every candidate channel and found by independent sampled search;
- 1 biologically admissible but non-query-critical alternative;
- 1 zero-proposal PSU with a discovered alternative;
- no biological or population-level recall claim.

The #10603 challenge references demonstrate one decision flip and one invalidated adaptive policy. They are precomputed synthetic report bindings in this contract fixture, not claims that the external #10603 workflow was run on real biological evidence.

## 7. Required adversarial controls

1. Shared source/preprocessing/reviewer ancestry must not receive independence credit.
2. A zero-proposal PSU must remain in the sample denominator.
3. An incomplete or inaccessible PSU must not be coded as no error found.
4. Ambiguous proposal-to-alternative identity matching must not be counted as a confirmed capture.
5. A query-changing alternative must invalidate the appropriate policy/certificate.
6. A graph change that leaves the query invariant must not be misreported as a query-critical discovery.
7. Cross-volume/source-operation leakage between development and lockbox splits must reject the run.
8. Candidate generation must not see sampled locations, audit outcomes or blinded alternative identities.
9. MSE output must preserve model sensitivity and may return NOT_IDENTIFIABLE.
10. Mutating the frozen query, candidate-pool digest, base hypothesis digest, or #10603 report digest must fail closed.

## 8. Promotion targets

These are preregistered research targets, not current results:

| Measure | Target |
|---|---:|
| Correct query-critical classification on known-answer fixtures | 100% |
| False robust certificate after a query-flipping world is admitted | 0 |
| False independence promotions on shared-ancestry controls | 0 |
| Zero-proposal PSU denominator retention | 100% |
| Incomplete/inaccessible search silently treated as negative | 0 |
| Design-weighted query-critical alternative recall | ≥90% overall |
| High-impact-family query-critical recall | ≥95% |
| Improvement in query-critical impact per total reviewer-hour vs strongest frozen single-channel baseline | ≥20% |
| Increase in false accepted corrections | 0 |
| MSE estimates promoted despite material model sensitivity | 0 |

Recall gates require a design-aware lower confidence bound and adequate effective PSU support. A point estimate above threshold is insufficient. The high-impact family and its impact rubric must be frozen before outcome review.

## 9. Industry comparison

ConnectomeBench2 V2 remains the direct predictive proofreading baseline for split/merge error identification, calibration, and species/region transfer. NEURD, RoboEM and specialist VLMs remain relevant candidate-generation or reconstruction comparators.

This benchmark measures a different contribution: whether the full investigation workflow finds omitted alternatives that matter to a scientific conclusion, including alternatives missed by every production channel. It must not claim classifier superiority or pool its recall denominator with ConnectomeBench2's labelled decisions.

## 10. Claim ceiling and staged execution

The contract can measure query-critical discovery within a declared population frame, sampling design, source-access scope and adjudication protocol. It cannot prove global biological completeness or establish that every undiscovered alternative is represented.

Stage 1 is synthetic contract qualification and exact #10603 integration. Stage 2 is source-frame, PSU and channel-lineage closure using the existing owners. Stage 3 is a blinded, independently adjudicated biological pilot. Stage 4 is a prospective randomized workflow comparison under #9242.

No stage may skip directly from synthetic correctness to biological reliability or SOTA claims.

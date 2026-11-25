# MEL-GROUND-007 — Source-clustered inter-annotator reliability v1

**Status:** synthetic-first implementation in progress; no musician-study result  
**Date:** 25 November 2026  
**Implementation:** `tools/mel-temporal-grounding-qualifier/src/temporal_reliability.rs`  
**Owner:** #13617  
**Consumes:** #13560 / #13563 (independent annotation records), #10296 / #13526 (production interval scorer)

## 1. Why one agreement score is a category error

A musical event query contains at least three different annotation questions:

1. Does the event occur at all?
2. How many separate occurrences are there, and which intervals correspond?
3. Where exactly are the boundaries of each occurrence?

A nominal label coefficient answers only the first question. It cannot establish that listeners agree on occurrence counts or that they place the start/end boundaries similarly. A pairwise interval score cannot establish agreement about whether an event is present in a gradual or masked passage. Boundary tolerance cannot tell us whether the annotators enumerated the same events.

The evaluator therefore reports three distinct reliability layers:

- **Occurrence-state reliability:** Krippendorff's nominal alpha over Present/Absent ratings across multiple query items.
- **Occurrence-set reliability:** symmetric pairwise interval F1, matched IoU and exact occurrence-count agreement for Present/Present pairs.
- **Boundary reliability:** start/end absolute errors in milliseconds and tolerance coverage at preregistered task-specific thresholds.

This does not generate a consensus reference, rank annotators, measure model confidence or convert a small panel into a population claim.

## 2. Dataset design

Each dataset item binds:
- unique item and query IDs;
- a source-work lineage digest (recording/work/session cluster);
- exact audio digest and clock;
- annotation policy version and manifest digest;
- frozen temporal-grounding scoring plan;
- two or more independent, pseudonymous annotator records with Present / Absent / Ambiguous / NotEstimable state and exact half-open intervals.

The dataset requires:
- one shared scoring plan across all items;
- one shared annotation policy version;
- distinct annotation-record digests across the dataset;
- stable annotator identity digests across items;
- unique query IDs;
- a minimum panel size per item;
- a preregistered list of strictly ascending boundary-tolerance thresholds.

A source work may contribute several query items. Those items remain separate observations for descriptive statistics, but all items from that work are kept together during source-cluster resampling. This avoids treating six excerpts from one song as six independent works.

## 3. Occurrence-state reliability: nominal Krippendorff alpha

For each query item, use only Present and Absent ratings. Ambiguous and NotEstimable annotations remain in the dataset/report but are missing for this particular nominal-state calculation; they are never recoded as Absent.

For item (u), let (n_{u,c}) be the number of eligible ratings in category (c). The ordered coincidence contribution is:

[
o_{ck}^{(u)} =
egin{cases}
n_{u,c}(n_{u,c}-1), & c=k \\
n_{u,c}n_{u,k}, & c\ne k
end{cases}
]

Coincidences are accumulated across items with at least two eligible ratings. Observed disagreement is the off-diagonal coincidence mass divided by total coincidence mass. Expected disagreement is derived from the pooled coincidence marginals. The nominal coefficient is:

[
\alpha = 1 - \frac{D_o}{D_e}
]

When (D_e=0), alpha is undefined and reported as null—not 1.0. This occurs, for example, when every eligible label belongs to one category. The report includes observed/expected disagreement, eligible item count, total Present/Absent ratings, coincidence count, unordered rater-pair count and ambiguous/not-estimable exclusions so the coefficient is auditable.

**Interpretation boundary:** alpha is reliability of the binary occurrence-state labels on this panel and item set. It is not interval-boundary agreement, musical usefulness or model accuracy.

Reference: Krippendorff, *Computing Krippendorff's Alpha-Reliability* — https://repository.upenn.edu/asc_papers/43/

## 4. Occurrence-set reliability

For every annotator pair whose state is Present/Present:
- score annotation A as ground truth and B as prediction;
- score annotation B as ground truth and A as prediction;
- use the exact production temporal-grounding scorer and frozen IoU threshold in both directions;
- average the two directions for symmetric interval F1 and matched IoU;
- report exact occurrence-count agreement and absolute occurrence-count difference separately.

This reuses the production one-to-one matcher. It does not merge intervals, average boundaries into a new reference, or hide missing repeated events behind a broad union interval.

For pairwise combinations:
- Present/Present: interval metrics and count metrics are computed.
- Absent/Absent: state agreement is recorded, with no interval metric.
- Present/Absent: a state disagreement is recorded, with no interval metric.
- Any Ambiguous/NotEstimable side: the pair is excluded from nominal-state agreement and interval-set agreement with an explicit reason.

## 5. Boundary reliability

Matched pairwise start/end errors are converted to milliseconds using the exact audio sample rate. The dataset preregisters ascending thresholds (for example 25, 50 and 100 ms); each threshold reports start, end and joint boundary coverage. The evaluator does not select the tolerance after looking at the errors.

Where a reliable score-to-performance map exists, a future adapter may add beat-fraction error. It must remain a separate unit and must not replace millisecond error. Tempo maps and performed time are not interchangeable.

## 6. Source-work-cluster bootstrap

The bootstrap resamples source-work clusters with replacement using a fixed SplitMix64 generator:
- all query items from a selected source work are included together;
- point estimates for pairwise interval/count and boundary metrics are macro-averaged by source-work cluster;
- nominal alpha is recomputed on the resampled set of query items using its coincidence matrix;
- 95% percentile intervals are emitted separately for alpha, pairwise state agreement, interval F1, matched IoU, exact count agreement, start/end boundary MAE, and each preregistered start/end/joint boundary-tolerance rate;
- the seed, replicate count, valid replicate count and source-work cluster count are included.

For a given tolerance, each source work contributes its own within-work boundary-match rate; the estimate is then averaged across source works. Bootstrap replicates resample those same source-work units. This is not the same as treating every note boundary or every query crop as an independent observation.

The current proposed confirmatory floor is 30 independent source-work clusters. This is a reporting warning, not a substitute for a preregistered power analysis or a study-specific sampling design. The report always leaves `population_confidence_claim_permitted=false`; passing the cluster-count floor alone does not authorize a population claim.

### Missingness and ambiguity are not automatically random

Krippendorff's alpha can be computed on incomplete/unbalanced rating matrices, but that algebraic property does not prove the missingness mechanism is harmless. A boundary or event may be marked Ambiguous/NotEstimable precisely because it is gradual, masked, dense or culturally unfamiliar. Those exclusions can therefore depend on the underlying unobserved event and should not be presumed missing completely at random.

The current report preserves and counts Ambiguous/NotEstimable exclusions; it does not convert them to Absent or include them in the nominal coincidence matrix. A real study must report exclusion rates by event family, recording density, source tradition, annotator and other preregistered strata, then run a missingness-sensitivity analysis. Alpha is conditional on the ratings eligible for that calculation; do not interpret it as reliability over excluded cases. The synthetic v1 does not implement a missing-not-at-random correction.

### Keep model performance on a separate axis

The dataset report also embeds the #13560 per-item model-vs-annotator result: raw model prediction, exact artifact and scorer plan, original pseudonymous annotation records, and per-annotator model scores. It adds source-work-macro summaries for Present F1, exact count, one-to-many count-gated F1, Absent claim accuracy, prediction coverage and fully-correct rate.

These model metrics must not be folded into alpha or pairwise human interval agreement. A low alpha may indicate a genuinely ambiguous query; it does not excuse a model's obvious miss. A high alpha does not imply the model is accurate. The report does not calibrate model confidence.

## 7. External benchmark and industry comparison

| Comparator | Current public evidence | What it means for MEL |
|---|---|---|
| TAG-Bench | 1,750 human-verified query/recording pairs over 149.5 hours; 22.1% multi-target. Its best of 21 reported systems reaches 31.2 mIoU and 21.5% Recall@IoU≥0.7; no evaluated system exceeds 13.2% one-to-many count accuracy. https://arxiv.org/abs/2609.01542 | Reproduce the exact official split and all native metrics. The count result makes event enumeration a primary research opportunity, not a secondary diagnostic. |
| RefTAG | Reflection-aware draft → count/boundary diagnosis → revision; reports 38.50 TAG-Bench mIoU and 51.01 mean mIoU over three datasets. https://arxiv.org/abs/2610.16217 | Mandatory strongest temporal-revision baseline. Compare direct prediction, self-revision and evidence-grounded revision under the same inputs and inference budget. |
| MusTBench v3 | Five music-temporal tasks; expert-validated benchmark and four-stage training/optimization recipe. https://arxiv.org/abs/2605.29300 | Reproduce native task metrics; do not substitute the multi-annotator reliability extension for official benchmark scoring. |
| MusicGroundingBench | Exact symbolic-to-audio piano MIDI alignment with note-level and two-bar grounding. https://arxiv.org/abs/2608.29480 | Use for deterministic event/count controls, but not as a proxy for human boundary reliability. |
| SMEBench | 3,600-example paired synthetic edit test set and a real-audio complement across 12 edit operations. https://arxiv.org/abs/2610.36715 | Keep requested-edit fidelity, source preservation and text compatibility separate from perception/grounding metrics. |
| XACLE 2027 | Audio-text alignment evaluation deliberately tests generalization to unseen audio-generation systems; submission deadline listed as 3 Dec 2026. https://xacle.org/2027/ | Adjacent evaluator-generalization challenge. It measures semantic alignment against human judgments, not temporal occurrence localization. |
| Suno Studio 2.0 | Updated 23 Nov 2026: automation, effects, MIDI, synth, user-designed plugins and chat-based clip generation/arrangement in an integrated production environment. https://help.suno.com/en/articles/13670529 | Do not claim MIDI, editing, automation, plugins or chat-directed generation are unique to MEL. Compete on inspectable commitments, verifiable edits, musician veto/branch rights, reproducible lineage and independent observation. |
| ElevenLabs Music v2.5 | Official docs describe audio reference, composition plans and inpainting; its 23 Nov update says v2.5 is now the default. The company reports a 47,885-pair blind preference test favoring v2.5 over v2. https://elevenlabs.io/docs/overview/capabilities/music ; https://elevenlabs.io/blog/music-v2-5-model | Treat generation quality and creator UX as strong commercial baselines. The vendor's preference result is a vendor-reported test, not an independent MEL benchmark. |

The strongest product differentiation is not a claim that Melothaea can generate a better-sounding song than the commercial systems. It is a verifiable music-partner loop:
- forecast the musical action before execution;
- expose uncertainty and alternatives;
- honour accept/steer/veto/branch/ignore as distinct authority states;
- preserve the exact draft and revision history;
- independently verify what was emitted and what was observed;
- preserve disagreement between human listeners instead of manufacturing certainty.

Suno Studio 2.0 and ElevenLabs Music already have substantial creator workflows. MEL should outperform on evidence quality, negotiated agency, temporal grounding and reproducibility—not pretend to lead on features those products demonstrably ship.

## 8. Matched SOTA protocol and proposed targets

The following are proposed programme targets, not achieved results:

1. Reproduce TAG-Bench and RefTAG results on the same permitted data, native split and metric implementation before making any ranking claim.
2. For temporal grounding, target a preregistered improvement of at least **+5 absolute mIoU points** over the strongest reproduced baseline, with a source-work-clustered 95% interval for the paired difference excluding zero and no hard-gate regression.
3. Treat one-to-many count accuracy as a first-class endpoint. Initial target: at least **25%** on the exact TAG-Bench one-to-many subset; stretch: **35%+**. These are prospective goals against the paper's reported ≤13.2% range, not a current MEL score.
4. For a high-agreement, clearly observable event family, target pairwise state alpha of at least **0.80** in the musician panel; if this is not met, report the family as uncertain and revise the annotation protocol rather than forcing consensus. This is a pilot target, not a universal cutoff for all musical events.
5. Report interval F1, matched IoU, count agreement, boundary error and boundary tolerance rates independently. Never average them into a global score.
6. Require at least 30 independent source works for a confirmatory cluster-bootstrap report, or a preregistered study-specific power analysis that justifies a different number.
7. Evaluate held-out source works, performers, instruments, renderers, production conditions and musical traditions. The split unit is source work/performance lineage, not random crops.
8. Run ablations for annotation count, boundary policy, ambiguous-state handling, source evidence, revision actions and model/runtime budget. Include always-one-event, always-absent, duration-prior and local-cue-only negative controls.
9. Use blinded human judgments to validate whether measured interval and count improvements matter to musicians; do not substitute metric gains for user outcomes.

A SOTA claim is eligible only after exact benchmark reproduction, preregistered comparison, held-out source-work evaluation, matched inference budgets, source-clustered statistics, transparent failed-run reporting, and independent musician assessment. A better average with worse repeated-event recall, abstention coverage or agency is not an overall win.

## 9. Sequencing

1. Finish synthetic validation of the per-item multi-annotator adapter (#13560/#13563).
2. Finish synthetic validation of dataset-level alpha, pairwise interval agreement and source-work bootstrap (#13617).
3. Freeze an annotation handbook with positive, absent, gradual, ambiguous and not-estimable musical event policies.
4. Run a pilot with musicians across event families and recording conditions; estimate inter-rater reliability and boundary tolerance before choosing confirmatory sample size.
5. Reproduce TAG-Bench / RefTAG under their own protocols and record exact versions.
6. Run the MEL multi-annotator extension on source-disjoint held-out works, including repeated and overlapping events.
7. Connect these measurements to forecast preview (#10190), negotiated agency (#13313) and the closed-loop output/observation layer (#13480).
8. Only then make a human-benefit or SOTA claim.

## 10. Claim ceiling

This v1 is a deterministic reliability-measurement harness. Its synthetic fixtures verify formulas and replay. It does not authenticate annotator identity, validate the source audio bytes, establish that annotations are independent, determine which annotator is correct, or prove that high annotation agreement implies musical usefulness. No model result, musician-study result, SOTA advantage or population reliability claim is asserted.

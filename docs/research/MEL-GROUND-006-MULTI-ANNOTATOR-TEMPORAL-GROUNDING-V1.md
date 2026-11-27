# MEL-GROUND-006 — Multi-annotator temporal-grounding qualification v1

**Status:** synthetic-first implementation; not a human-study result  
**Date:** 25 November 2026  
**Implementation:** `tools/mel-temporal-grounding-qualifier/src/temporal_consensus.rs`

## 1. Research shift: from one adjudicated timestamp set to a reference distribution

A musical event boundary is not always a single objectively perceived instant. Instrument entrances can be gradual; attacks can be masked; rubato, reverb and overlapping voices can create plausible boundary disagreement. A benchmark that collapses several listeners into one consensus interval can hide the uncertainty that exists in the listening task.

This profile preserves each annotator's state and half-open interval set independently, then scores the *same model prediction* against each annotation through the existing production temporal-grounding scorer. It does not synthesize a majority-voted reference, rank annotators, adjudicate which listener is right, or interpret reference spread as model confidence.

It separates four different objects:

1. **Human-reference variability:** disagreement among independent annotation records.
2. **Model output:** one prediction, including explicit abstention or parse failure.
3. **Model performance:** the distribution of production-scorer results against each eligible annotation.
4. **Model uncertainty:** not represented by this scorer; it requires separately calibrated probability/confidence outputs and a coverage–risk evaluation.

## 2. Implemented v1 contract

The strict input binds query ID/text, exact audio artifact clock/digest, annotation policy and manifest digests, minimum independent panel size, one frozen production scoring plan, one exact-artifact model prediction, and independent annotator records.

Each annotation retains:
- opaque annotator identity digest;
- annotation-record digest;
- Present / Absent / Ambiguous / NotEstimable state;
- its own half-open frame intervals;
- explicit status reason where the state is Ambiguous or NotEstimable.

The report preserves every annotation record, per-annotator TP/FP/FN, interval F1, matched IoU, count diagnostics and boundary errors. Ambiguous/not-estimable labels remain visible but are excluded only from their own accuracy denominator. Absent annotations remain scored negative references.

The report also records explicit claim ceilings:
- annotation manifest authenticated: false;
- annotator receipts authenticated: false;
- pairwise inter-annotator agreement computed: false;
- synthetic consensus reference created: false;
- model confidence calibration computed: false;
- population confidence interval computed: false.

Digest-shaped identifiers are not authentication. The adapter does not verify the WAV bytes or prove that the annotation was independently made by the named person.

## 3. Score and coverage dimensions

Report each dimension independently; do not produce a one-number quality score.

### Per-annotator performance

- Positive event interval precision/recall/F1 and matched interval IoU.
- Exact occurrence-count accuracy and absolute count error.
- One-to-many count-gated F1 on only the annotations with more than one reference occurrence.
- Start/end boundary error in source frames; convert to milliseconds only with the exact artifact sample rate.
- Negative-claim correctness for Absent annotations.
- Full-correct rate and unanimous-correct-across-eligible-panel flag.

### Reference distribution

- Mean, median, min and max positive-reference F1.
- Start/end boundary MAE and P95 across matched events.
- Annotated occurrence-count min/median/max among Present labels.
- Per-annotator results and raw intervals retained; no consensus interval substituted.
- Human-reference disagreement is descriptive unless a separately designed reliability study supports population inference.

### Model coverage

- Attempted panel size.
- Eligible Present/Absent annotations.
- Model abstention and parse-failure counts.
- Overall, positive-only and negative-only prediction coverage.
- Accuracy and coverage stay side by side. An abstention cannot silently remove a difficult eligible annotation from the denominator.

A model's confidence is not inferred from how much humans disagree. To evaluate confidence, require a separately versioned probability or confidence output and proper scoring rules such as Brier score/log loss, calibration curves and risk–coverage curves.

## 4. Benchmark programme and matched competitors

Use each public benchmark for its native task and retain a separate MEL extension. Do not pool scores across incompatible datasets, query families, durations or annotation policies.

| Benchmark / system | Publicly reported scope | How MEL should use it |
|---|---|---|
| TAG-Bench | 1,750 human-verified query-recording pairs; 149.5 audio hours; 387 multi-target queries; 21 systems | Reproduce official mIoU, Recall@IoU, gIoU, midpoint MAE+coverage, count and parse-failure metrics on the permitted official split. Then add the multi-reference MEL extension on separately sourced annotations. |
| RefTAG | Draft → count/boundary action → revision; Re-CoT-SFT and RL; 38.50 TAG-Bench mIoU reported in its paper | Mandatory strongest revision baseline. Match the same query, evidence, model budget and source split. Compare direct-final, self-revision and evidence-grounded revision separately. |
| MusTBench / MusT | Five music-temporal QA tasks and a four-stage encoder/LLM/SFT/RL training recipe | Reproduce its native metrics for the five music tasks; test the MEL adapter on a separate multi-annotator set. |
| MusicGroundingBench | Exact symbolic-to-audio piano MIDI alignment; note-level and two-bar grounding | Use as the known-answer synthetic bridge for exact interval/count controls, while not pretending synthetic alignment models human perceptual uncertainty. |
| ViTAA | Generic audio-annotation visualization and temporal-tolerance agreement tooling | Reference for annotation workflow and tolerance-aware agreement analysis. Its generic Cohen's kappa workflow is not a music-event ground truth standard; MEL must preregister event semantics and tolerance by family. |

### Current TAG-Bench gap

The TAG-Bench project reports its best evaluated system at 31.2 mIoU and 21.5% Recall@IoU≥0.7. Its repository reports a maximum one-to-many count accuracy of 13.2% among evaluated systems. These are benchmark-specific results, not universal music-understanding rankings.

### RefTAG changes the bar

RefTAG reports 38.50% TAG-Bench mIoU, plus improvements from explicit draft/count/boundary revision. That makes a simple “we added a self-reflection step” claim insufficient. The MEL contribution must demonstrate that:
- revisions are measured against the exact production scorer;
- declared actions match the actual prediction delta;
- local evidence beats both no-evidence and yoked-evidence controls;
- gains persist on source-lineage-disjoint held-out songs;
- gains do not come from extra inference budget, different runtime, or easier reference selection;
- multi-annotator score distributions do not hide minority-valid interpretations.

## 5. Experimental protocol required for a defensible SOTA claim

1. **Frozen benchmark version:** pin datasets, licence, query IDs, audio hashes, annotation policy, scorer commit, interval IoU thresholds and parsing grammar.
2. **Official reproduction first:** run every baseline on the exact official test protocol. Report system version, prompt, decoding configuration, runtime, retries, wall time and token/audio budget.
3. **MEL extension separate:** obtain multiple independent annotations for a source-cleared held-out subset. Keep all annotators' raw interval/state records and disagreement, including absent, ambiguous and not-estimable cases.
4. **Source-lineage split:** all clips/edits/crops/alternate mixes from one source work or recording session remain in one split. Avoid leakage across stems, renderer outputs or excerpts.
5. **Counterfactual arms:** direct final; draft→revise without evidence; local evidence; yoked evidence; oracle symbolic alignment where available; and human references. Equalize model/runtime/decoding and declared compute budgets.
6. **Ablations:** remove count diagnostics, boundary diagnostics, retrieved evidence, model-side uncertainty, and transformation metadata one at a time.
7. **Uncertainty:** bootstrap source works, not frames or excerpts. Preserve a separate count-gated cohort. Use power analysis before choosing the confirmatory sample size.
8. **Human adjudication:** only where necessary, keep adjudication as a distinct third annotation record with reason and lineage; never overwrite original annotations.
9. **Report every failure:** parsing, abstention, missing outputs, empty predictions, unavailable evidence and out-of-scope queries remain visible.
10. **Independent replay:** another evaluator must be able to reproduce per-annotator scores and aggregate summaries from raw manifest, raw annotations, prediction, scorer and plan.

## 6. Prospective targets — not achieved results

- 100% deterministic pass on schema, duplicate identity/digest, query/audio mismatch, interval and state/reason negative controls.
- 100% raw annotation-record recovery from the report.
- 0 fabricated consensus intervals.
- Exact parity with the production scorer for every annotator record.
- 100% preservation of ambiguous/not-estimable exclusions as distinct records.
- No model abstention or parse failure silently removed from coverage denominators.
- 100% determinism on fresh-process replay of per-annotator outputs.
- For a confirmatory human study, preregister minimum panel/source-work sizes through power analysis; do not infer sample adequacy from a synthetic six-annotator fixture.

A future model-performance target should be set only after reproducing TAG-Bench/RefTAG and measuring the matched MEL baseline. A reasonable *research ambition* is a ≥5 percentage-point absolute improvement on one preregistered primary metric with a source-clustered 95% interval excluding zero, replicated on a held-out source-work cohort, while no hard-gate or coverage dimension regresses. This is a proposed threshold, not a current result.

## 7. Product and industry positioning

Suno Studio 2.0 now provides MIDI tracks, a piano roll, built-in synthesis, external MIDI input, audio↔MIDI workflows, automation and effects. ElevenLabs Music v2.5 provides composition plans, audio references, and section-level inpainting with retained source chunks. Melothaea cannot differentiate by claiming generic MIDI editing, local regeneration, audio references or chat-based creative control are unique.

The stronger product distinction is **evidence-bearing collaboration**:
- every preview is a versioned forecast, not a post-hoc story;
- every edit has a source-preservation contract and exact parent lineage;
- each temporal claim carries its audio artifact/time map and independent observer status;
- musician veto/steer/branch/ignore rights remain explicit;
- uncertainty and disagreement are visible instead of silently averaged;
- the creator can replay why a change happened and what evidence supported it.

This is a systems-level product hypothesis, not an established commercial advantage. It must be validated against the commercial workflows on matched tasks and with musicians.

## 8. Claim boundary

The current v1 code is an evaluation adapter and synthetic test harness. It does not establish:
- model-level temporal grounding performance;
- human inter-rater reliability or formal pairwise agreement;
- annotator authentication or consent;
- calibration of model confidence;
- causal value of multi-annotator training;
- improved musician agency, trust or ownership;
- an overall SOTA or commercial advantage.

Those require independent annotations, human studies, external benchmark reproduction and authenticated execution/evidence receipts.


## Per-event boundary distribution

For every predicted occurrence index, the adapter derives a reference-boundary distribution from the production scorer's existing prediction↔annotation matches. It reports the number/fraction of eligible Present annotators who match that occurrence and the min/median/max annotated start/end frames. Predicted occurrences with no match in any eligible Present annotation are listed separately.

The model prediction index is only a common alignment key; the adapter does not invent a canonical event ID or merge annotation intervals. A low support rate can indicate a count/identity disagreement or a boundary outside the frozen IoU threshold; inspect the per-annotator records before interpreting it. This distribution is descriptive, not calibrated model uncertainty.


## 3.1 Pairwise temporal concordance (descriptive, not formal IAA)

The implementation also compares every pair of independent annotators. Exact disposition agreement is reported across Present, Absent, Ambiguous and NotEstimable states. For Present/Present pairs only, the adapter uses the production interval matcher in both directions, then reports symmetric event F1, mean matched interval IoU and exact occurrence-count agreement.

This separates three distinct questions:
- Did both annotators believe the event/query was present, absent, ambiguous or not estimable?
- When both marked Present, did they enumerate the same number of occurrences?
- When both marked Present, did their event intervals overlap and align closely?

Pairs with nonmatching dispositions, including Present/Absent, do not receive fabricated interval scores. Ambiguous and NotEstimable remain categorical dispositions and are not treated as absence. The per-pair records are retained so a mean cannot hide a small number of severe disagreements.

This is a descriptive pairwise agreement view for one query, not chance-corrected reliability. The adapter explicitly leaves Cohen's kappa, Fleiss' kappa and Krippendorff's alpha unset. Formal reliability requires a multi-query corpus with preregistered items, raters, tolerance policy, missingness treatment and a justified chance model. For event intervals, report event F1/count agreement separately from matched IoU and boundary deviation; a single kappa over discretized time bins can change substantially with the chosen bin width.



## Canonical integrity commitments — 27 November 2026

The adapter now recomputes two deterministic SHA-256 commitments using the shared production `canonical_json_sha256` implementation:

1. **Annotation record commitment:** includes schema, query ID/text, exact artifact clock/digest, annotation policy version, opaque annotator-ID digest, state, raw intervals and status reason.
2. **Annotation manifest commitment:** includes the same scope plus the sorted list of annotator-ID / annotation-record-digest pairs.

Consequences:
- Reordering the submitted annotation array does not change the manifest digest.
- Mutating an interval, disposition, reason, policy, or panel membership without updating the commitment fails closed.
- All original records remain intact; the digest binds them but does not authenticate a person or prove that an independent listening/annotation process occurred.

The output explicitly distinguishes:
- `annotation_record_digests_recomputed = true`;
- `annotation_manifest_digest_recomputed = true`;
- `annotator_receipts_authenticated = false`;
- `annotation_manifest_authenticated = false`.

This raises the integrity floor but does not replace signatures, trusted timestamping, rater-role/assignment receipts, or consent/purpose verification. Those require the independent provenance owners and a real annotation workflow.

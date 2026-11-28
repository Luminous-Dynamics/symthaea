# MEL-HEAR-009 — Decompose audio perception, alignment and musical reasoning gaps v0

**Status:** research contract and deterministic synthetic fixture; not a model-performance result  
**Date:** 28 November 2026  
**Parent:** #14146  
**Composes:** #13908, #13947, #14005, #14019, #14026, #14028, #14030, #13815, #11389

## Executive result

A system can fail a musical identity question for very different reasons: it may not recover the relevant audio event, align it to the wrong voice or time, lose the distinction while converting audio into symbols, reason incorrectly over correct evidence, or make a poor uncertainty/abstention decision.

A raw-audio task score does not reveal which stage failed. A symbolic-only score does not prove that a model can perceive the recording. MEL-HEAR-009 freezes a matched-condition decomposition so the same musical item and property query can be evaluated with oracle symbolic/event evidence, audio-derived symbolic/event evidence, raw audio, audio plus predicted structure, and a no-evidence control.

The retained synthetic panel is intentionally only one known-answer item. Its purpose is to validate the condition contract and the decomposition counters—not to estimate capability.

## Condition arms

| Arm | Model-visible evidence | Diagnostic role |
|---|---|---|
| `oracle_symbolic` | Independently checked symbolic/event representation sufficient for the target property | Reasoning upper-bound diagnostic |
| `audio_derived_symbolic` | Transcription/event representation and alignment produced from the recording; same downstream reasoner as oracle | Front-end representation + alignment plus fixed reasoning |
| `raw_audio` | Exact materialized audio evidence, no external symbolic representation | End-to-end audio perception/reasoning baseline |
| `audio_plus_predicted_structure` | Same raw audio plus the model's own predicted structure | Test incremental value of structure fusion |
| `no_evidence_control` | Same query and matched policy, no audio or symbolic evidence | Detect prompt priors / answer guessing |

The oracle arm is an *upper-bound diagnostic only for properties represented by its supplied event graph*. A symbolic score is not sufficient oracle evidence for acoustic-only timbre, expressive tone, recording-room properties or performance identity unless an independently validated representation supplies those attributes. Such tasks must be excluded from the oracle-score analysis or use a distinct, explicitly sufficient oracle representation. Never treat an oracle answer label as an oracle representation.

## Ownership boundaries

This contract consumes references; it does not replace upstream authorities:

- #13947/#14019 own property-relative musical identity reduction and paired comparisons.
- #13815 owns recomputation of source bytes, decoded PCM and selected-window evidence.
- #14005 owns the audio-evidence-to-identity-triad join.
- #14030 owns independent musician annotations, disagreement and annotation provenance.
- #14028 owns frozen budget profiles and execution receipts. A budget digest does not prove provider-internal input consumption.
- #14026/#14114 own performer × work dependence, estimands and inference.
- #11389 owns independent score-to-render conformance. A renderer's source score is not automatically a correct transcription of a real human performance.

The qualifier checks structural references and matched conditions. It does not decode audio, transcribe notes, authenticate performers, verify annotation custody, validate provider-internal model input or compute population confidence intervals.

## V0 manifest and checks

The versioned panel manifest binds:
- source-work lineage, performer cohort, pair/triad, split and target property;
- exact query digest and independent reference/annotation manifest references;
- exact #13815 materialization/evidence references;
- sampling-design and #14028 matched-budget profile commitments;
- each condition's model, runtime, prompt, adapter, decoding and execution receipt identities;
- per-condition input-context, evidence stage and reference-visibility audit digest;
- oracle or predicted structure, transcription and alignment references where the arm permits them;
- prediction, confidence and explicit abstention/unparseable dispositions.

Fail-closed checks include:
- exactly one of each of the five condition types;
- one shared frozen budget profile;
- one exact audio evidence manifest shared by raw, audio-derived and fusion arms;
- oracle and audio-derived symbolic arms use the same downstream reasoning-head identity;
- raw-audio, fusion and no-evidence arms use the same model/runtime/prompt/adapter/decoding identity;
- oracle reference is absent from deployable audio arms;
- predicted structure and transcription/alignment reports are present only on arms that receive model-derived structure;
- independent reference labels are not model-visible in any arm;
- all digest-shaped references are syntactically valid and all declared conditions are present.

A digest is a content commitment, not authentication. The report preserves the declared evidence stage; a locally constructed request digest is not a provider-side model-input receipt.

## Diagnostic outputs

For each condition, report item-level:
- whether the reference is scorable;
- correct/incorrect/answered/abstained-or-ambiguous/unparseable counts;
- accuracy and coverage for binary Same/Different references;
- prediction and confidence.

Also report paired diagnostic contrasts:
- oracle correct / audio-derived symbolic incorrect;
- oracle incorrect / audio-derived symbolic correct;
- raw audio correct / fusion incorrect;
- raw audio incorrect / fusion correct;
- no-evidence control correctness.

These flags localize *where to investigate*, but they are not causal component effects by themselves. A difference between oracle and audio-derived arms may reflect transcription, alignment, representation, or residual pipeline differences. A causal claim about a component needs a randomized/frozen ablation that changes only that component while preserving downstream heads, budgets, queries and data.

Ambiguous and NotEstimable references are not converted to binary truth. Abstentions and parse failures remain errors in the primary denominator for clear Same/Different items; their counts remain separately visible.

## Deterministic fixture result

The synthetic one-item fixture is designed to produce:
- oracle symbolic: correct;
- audio-derived symbolic: incorrect;
- raw audio: correct;
- audio plus predicted structure: correct;
- no-evidence control: incorrect.

This checks that the evaluator can represent an oracle-to-predicted-structure gap and distinguish fusion from a prompt-only answer. It does not establish any actual model behavior.

## Prospective benchmark targets (not results)

After a musician-reviewed pilot and frozen held-out real-audio cohort:

- Oracle symbolic reasoning: ≥95% clear-case accuracy for properties that the supplied oracle representation can actually establish.
- Audio-derived symbolic reasoning: ≥90% on supported/observable properties.
- End-to-end raw-audio and fusion: initial target ≥85% on clear, rights-cleared real-performance cases, reported per property and performer/work stratum.
- Calibration: ECE ≤0.05 with subgroup counts, plus risk–coverage and reason-specific abstention.
- Report note/event precision, recall and F1; pitch accuracy; onset/offset tolerance F1; voice/part assignment; occurrence-count accuracy; temporal IoU; and alignment failure separately.
- Every model comparison uses the same frozen capability-budget profile and discloses actual calls, retries, tokens/audio frames, latency, timeouts and unavailable provider measurements.
- Hold out source works and performers; report known-performer/new-work, new-performer/known-work, and both-unseen separately.

These thresholds are prospective pilot targets, not established SOTA numbers. The oracle arm is not a competitor in a public end-to-end leaderboard.

## External comparisons

- MusicListenBench is the most direct property-relative perception comparator: FLIP changes the target music property; STAY changes only the sound. Its November 2026 paper reports a real-recording failure where a post-trained model labels 27 of 42 same-chord two-strum pairs different. https://arxiv.org/abs/2611.29375
- TAG-Bench is an adjacent temporal-grounding test: its 1,750 query/recording pairs span 149.5 hours; its best reported system reaches 31.2 mIoU, 21.5% recall at IoU≥0.7 and no more than 13.2% occurrence-count accuracy. https://arxiv.org/abs/2609.01542
- MMAE is an adjacent edit-instruction benchmark with public development data and separately locked challenge tracks. Keep its operation-success measures separate from property identity and musician authority. https://audio-editing-challenge.github.io/leaderboard/
- Artificial Analysis instrumental/vocal Elo measures blind generation preference, not perception, edit correctness or agency. https://artificialanalysis.ai/methodology/music

## Claim ceiling

The v0 qualifier validates condition structure and diagnostic arithmetic. It does not establish audio-to-score accuracy, musical understanding, human-level hearing, causal component contribution, model superiority or population generalization. Those require exact evidence joins, independent musician annotations, rights-cleared real performances, matched held-out evaluation and dependence-aware uncertainty intervals.

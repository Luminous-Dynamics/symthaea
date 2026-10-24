# Cetacean Benchmark Ladder v1

**Status:** preregistration proposal; proposed targets are not current Symthaea results. No recordings have been scored by this plan. Do not describe acoustic prediction as translation or proven semantic comprehension.

## 1. Paradigm shifts

1. **From “translate the animal” to falsifiable prediction.** Start with observable tasks (detect an event, recover timing/contours, predict the next acoustic unit, predict a blinded behavioral/context label). Meaning claims require additional evidence and must not be inferred from a language-model-like output.
2. **From fixed tokens to structured time series.** Preserve continuous contours, precise inter-click intervals (ICI), rhythm, tempo, rubato, ornamentation, overlap, and turn-taking. Do not force every signal into a universal phoneme inventory.
3. **From random crops to ecological generalization.** Split by source recording and, where metadata permit, session/encounter, individual/social unit, site, season, and recording device. Treat each dimension as a separate, explicit generalization claim.
4. **From one score to an evidence ladder.** Separate signal integrity, event detection, segmentation, acoustic structure, behavior-linked prediction, and meaning. Passing a lower stage never implies a higher one.
5. **From benchmark accuracy to adversarial science.** Compare against identity-only, time-of-day, recorder/site, prevalence, nearest-neighbor, simple signal-processing, and history-only baselines. Include shuffled-label, shuffled-context, and hard-negative controls.
6. **From forced answers to calibrated abstention.** Preserve alternative hypotheses and allow “unknown.” Report calibration and coverage-versus-error; an uncertain answer is preferable to a confident hallucination.
7. **From public-model convenience to legal reproducibility.** The exact audio and annotation license must permit the intended use. A permissively licensed paper or model code does not automatically license the source recordings.

## 2. Evidence gates before any model score

The following are hard gates, not metrics to trade against model accuracy:

- Record the dataset ID, immutable revision/source URL, human-reviewed rights decision, intended use, attribution, and any restrictions on derived artifacts.
- Preserve exact source-recording lineage for each crop, annotation provenance, native sample rate, channel count, sample count, and independently assigned labels.
- Hash the exact local asset bytes. **bioacoustic-corpus-preflight** checks declared metadata; **bioacoustic-asset-audit** additionally streams each local file and verifies SHA-256 against the manifest. Both explicitly stop short of validating the audio header, rights adjudication, or detection performance.
- Freeze the split manifest, annotation protocol, matching tolerance, primary metric, threshold-selection procedure, comparison baselines, exclusion rules, and analysis plan before test evaluation.
- Tune thresholds on validation only. Never change the test threshold after inspecting test labels.
- If source/session/site/individual lineage is unknown, report it as unknown. Do not infer groups from filenames or random crop IDs, and do not claim the corresponding holdout.
- Keep a versioned annotation audit sample with double annotation and adjudication. Report disagreement instead of silently turning ambiguous calls into certain ground truth.

## 3. Task-specific benchmark plan

The numeric objectives below are **proposed engineering targets for a first pilot**, not accepted field-wide standards. Adjust them only in a documented preregistration made before the sealed test set is scored.

| Lane | First task | Primary metrics | Proposed pilot objective | Required controls |
|---|---|---|---|---|
| Common / bottlenose dolphins: whistles | Detect and track whistle contours in continuous audio | Event AP; event F1 under a fixed matching tolerance; onset/offset error; contour frequency MAE; false alarms per hour | Held-out-session event F1 ≥ 0.80 at ≤ 1 false alarm/hour; contour error reported in Hz and as a fraction of the annotated frequency span | Ship / rain / wind / other-species / overlapping-whistle negatives; identity/recorder baseline; blinded human review |
| Dolphins: echolocation clicks | Broadband click-event detection at native high sample rate | Event precision/recall/F1 with preregistered onset tolerance; missed events; false alarms per minute/hour; timing error | Event F1 ≥ 0.90 on a curated held-out pilot, with the actual precision-recall trade-off and uncertainty reported | Electrical artifacts, snapping shrimp where applicable, impulsive machinery, clipping, channel differences; no aliasing through unvalidated downsampling |
| Dolphins: burst-pulsed sounds | Segment pulse trains | Event F1; boundary error; pulse-rate error; per-encounter macro averages | Beat a non-learned transient/pulse baseline by ≥ 5 absolute AP points on held-out encounters, with a grouped confidence interval excluding zero gain | Click-train vs burst-pulse ambiguity; overlapping animals; annotator disagreement |
| Sperm whales: codas / ICI | Detect clicks and segment codas; recover ICI vectors | Click event F1; boundary error; ICI median absolute error; coda macro-F1; held-out sequence log loss | Coda macro-F1 ≥ 0.80 and median absolute ICI error ≤ 1 ms on a sealed, encounter-disjoint pilot | Individual/social-unit leakage, tag vs hydrophone differences, adjacent-coda context, shuffled ICI / shuffled context |
| Baleen whales: songs and calls | Segment units and track contours before classification | Event AP/F1; contour error; song-structure edit distance; false alarms/hour | Establish a source-pinned baseline and confidence intervals first; do not set a universal threshold across species or recording conditions | Seasonal/site shift, vessel noise, call overlap, recorder response |

The stated goals should be accompanied by a grouped 95% confidence interval (bootstrap by the independent unit, such as encounter or session), per-session/per-site slices when supportable, and a clear sample-size statement. A threshold target on one tiny dataset is not a species-wide operating guarantee.

### Cross-cutting pass criteria

A detector/structure model is not considered meaningfully better just because its pooled score is high:

- It must beat a strong, disclosed simple baseline by a preregistered margin (initial proposal: ≥ 5 absolute AP points or ≥ 10% relative error reduction, as appropriate for the task).
- The grouped confidence interval for improvement must exclude zero; otherwise classify the result as inconclusive.
- It must outperform label-shuffle and context-shuffle null controls. If identity, site or recorder prediction explains the result, the intended generalization claim fails.
- Report performance under a fixed low-false-alarm operating point as well as threshold-free metrics; detection in continuous hours is not interchangeable with balanced clip accuracy.
- Report calibration (Brier score and reliability bins) and coverage-versus-error when the model can abstain.
- Report all attempted configurations and test exclusions. No cherry-picked best seed or silent failed run.
- Keep compute, memory, throughput, end-to-end latency, and annotation effort beside predictive scores.

## 4. Sequence and interaction: the harder, higher-value direction

Once low-level events are reliable, preserve **exchanges**, not isolated sound crops. For sperm whales this includes click timing, coda boundaries, tempo, rhythm, rubato, ornamentation, sender uncertainty, overlap, and the preceding/following codas. For dolphins, preserve whistle contour, pulse trains, encounter context and known behavioral observations without assuming every observation identifies the caller or meaning.

Evaluate next-unit prediction with held-out log loss / perplexity **against simple history-only and identity-aware baselines**. Evaluate structure discovery for stability across independent encounters, not merely how plausible clusters look. A model that predicts the next sound has shown predictive structure; it has not yet shown what that sound means.

For a behavior-linked hypothesis, preregister:
- the externally observable outcome or context label;
- the annotation source and blinding procedure;
- the unit of independence and all relevant confounds (individual, site, time, device, observer, group size);
- null models, multiplicity control, uncertainty and replication criteria;
- the exact claim level that a positive result would support.

Start with passive observation and archival data. Any playback, synthetic call broadcast, or other intervention affecting animals requires a separate animal-welfare / ethics review and field protocol; it is outside this benchmark ladder.

## 5. Real data and reproduction constraints

- **DOLPHINFREE:** the 2025 *Earth System Science Data* article describes more than 400 minutes of common-dolphin recordings, approximately 68,000 echolocation clicks, about 4,600 whistle contours and more than 350 pulsed sounds, captured using 512 kHz mono and 256–512 kHz four-hydrophone recordings. This is unusually useful for a multi-task pilot, but admission depends on rights terms for the exact downloadable audio and annotations, not just the paper: https://essd.copernicus.org/articles/17/4495/2025/
- **OpenWhistle longitudinal benchmark (2026; distinct from OpenWhistle-CNN):** its project page describes approximately 180,000 whistles, 114 hours, five years and 8,354 expert labels. Reported baselines include a released Wav2Vec2/OpenWhistle checkpoint with 64.37 ± 2.67 full-set 10-class macro-F1 and 75.76 ± 2.02 detection mAP; Wav2Vec2 pretrained on all audio reports 80.80 ± 1.91 detection mAP in the data-scale comparison. These are benchmark-specific scores, not continuous-audio event-F1 values. Reproduce the exact task/split/metric before comparing, then preregister chronological evaluation: https://sumbre-lab.github.io/openwhistle/ ; 2026 paper: https://arxiv.org/abs/2609.34839
- **Dolph2Vec (2026):** species-specific self-supervised representations trained from five years of longitudinal recordings of five known dolphins, evaluated on signature-whistle classification and detection. Use as an acoustic representation baseline if exact data/model use terms allow it; representation clusters are not semantic labels: https://arxiv.org/abs/2606.12503
- **OpenWhistle-CNN (separate binary-window dataset):** its public card describes 76,516 whistle/noise windows from 482 sessions, 8.5 window-hours and a 261-session test split. It is suitable for the stated window task, but not a substitute for long continuous-recording false-alarm evaluation; verify license scope: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN
- **Sperm-whale dyad sequence prediction (2026):** a study reports that the next coda produced by one whale is predictable from up to ten preceding codas produced by the partner. This motivates a held-out dyad/encounter task with target-history-only, rhythm-only, order-shuffled and identity-only baselines; it does not establish coda meaning: https://doi.org/10.1080/09524622.2026.2684665
- **Coda two-tier structure (2026 preprint):** a 1,483-coda analysis proposes a rhythm-rich lower tier plus upper-tier sequential dependence. Replicate with acoustic nulls and social-unit/encounter holdouts before treating it as robust: https://arxiv.org/abs/2606.16084
- **Cross-site detector shift (2026):** Antarctic minke-whale detection results vary materially by site, motivating site-held-out evaluation and per-site false positives/hour rather than pooled metrics only: https://doi.org/10.1111/mms.70118
- **WhAM:** its code repository notes that reproducing all reported results requires DSWP+CETI annotations, only partly public at publication. A model/code license is not equivalent to the dataset license: https://github.com/Project-CETI/wham

## 6. Run commands and evidence vocabulary

Metadata-only preflight:

    cargo run -p symthaea-communication --bin bioacoustic-corpus-preflight -- corpus-manifest.json --require-source-recording-ids --require-annotations --hold-out-sessions

Actual local byte audit (asset URIs must be clean paths relative to the given root):

    cargo run -p symthaea-communication --bin bioacoustic-asset-audit -- corpus-manifest.json --asset-root ./permitted-corpus

Event-level evaluation of a precomputed prediction bundle:

    cargo run -p symthaea-communication --bin bioacoustic-event-eval -- corpus-manifest.json predictions-test.json

The prediction bundle uses `BioacousticEventPredictionBundle` from `symthaea_communication::bioacoustic_eval` (prediction bundle schema v1; report schema v2, versioned independently). It pins the dataset ID/revision and canonical corpus-manifest BLAKE3, model ID/SHA-256, task, target labels, matching rule, and frozen score threshold. Hex SHA-256 spellings are normalized to lowercase for canonical identity; array order that has no scoring semantics is normalized as well. Its `predictions` array may reference **Test samples only**. Choose `StartTolerance` for point-like events such as click onsets or `TemporalIou` for interval-like events such as whistle segments; select the rule and tolerance before examining Test results. For assets that are crops from the same source recording, populate `source_start_s` with the exact crop offset in the source timeline. Event evaluation rejects overlapping Test crops so duplicated audio cannot inflate the denominator of false alarms per scored hour. The current `BioacousticCorpusManifest` schema is v3 and requires explicit `source_start_s` and `unscored_regions` keys; use `null` / an empty array when unknown / none, rather than silently omitting metadata. Each ordered, non-overlapping unscored region carries a reason and an `annotation_source` provenance reference (e.g. ambiguous caller or annotation uncertainty). Regions are hashed into the manifest identity; they must be inside the sample and must not overlap positive event annotations. Predictions sufficiently covered by such regions are excluded from scoreable prediction sets under a task-specific deterministic rule; the evaluator reports excluded predictions and subtracts unscored-region duration from the false-alarm exposure denominator. Do not treat these regions as negatives; the remaining annotated coverage must still be supported by the annotation protocol.

`exhaustive_annotation_sample_ids` must exactly enumerate every Test recording, including recordings with zero target events. This is an explicit annotation-protocol assertion; the evaluator cannot independently know that unmarked events are truly absent. The evaluator rejects absent/extra Test IDs, sample/source leakage under the declared split policy, non-unique prediction IDs, predictions outside Test, invalid intervals/scores, and a mismatched manifest identity. It uses deterministic one-to-one event matching so duplicate predictions cannot count the same truth event twice. The report emits micro event average precision and precision/recall/F1 at the frozen threshold, per-recording metrics, and macro-F1 averaged across informative Test recordings (recordings with neither target events nor scored predictions are excluded because F1 is undefined). It also reports false alarms per scored hour of non-overlapping audio, per-recording and aggregate counts of predictions omitted because they fell in explicit unscored regions, manifest and prediction-bundle hashes, and explicit false flags for audio-byte verification, annotation-truth authentication, model execution, prediction provenance authentication and threshold-selection authentication. Average precision is tie-aware and uses one-to-one matching for each confidence group. Run the asset audit first; the evaluator scores precomputed outputs and does not run the detector.

Evidence labels:

- **Metadata admitted:** declared manifest, rights references and split relations passed checks.
- **Bytes verified:** every local file's SHA-256 matches its manifest entry. This does not verify decoded sample metadata or authenticity of annotations.
- **Benchmark evaluated:** a frozen model/configuration produced per-sample outputs on the sealed test split and metrics were computed.
- **Independently replicated:** an independent team or separately collected, properly permitted corpus reproduced the preregistered finding.
- **Meaning hypothesis supported:** requires task-specific independent behavioral/context evidence beyond acoustic classification or sequence prediction.

For the current Symthaea research branch, code and tests remain uncompiled/unexecuted in the authoring environment; the first two labels must not be reported as achieved until commands execute on exact data and head.

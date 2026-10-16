# Avian Communication Research Contract v1

**Status:** research plan; all benchmark thresholds below are proposed targets, not measured results or released capabilities.

**Purpose:** establish a falsifiable path from passive bird-audio analysis to evidence-backed claims about signal function and, much later, reciprocal communication. Do not build a bird-to-English translator as the primary abstraction.

## 1. Paradigm shift

Move the research question from **“What sentence does this call mean?”** to:

> Given the acoustic event, its social/ecological context, and a competing set of hypotheses, what observable event is this signal associated with, what would distinguish competing explanations, and does the receiver's behavior change under an appropriate intervention?

This has five consequences:

1. **Discover structure before assigning meaning.** Learn repeatable units, boundaries, sequences, timing and variation without requiring words or human semantic primes.
2. **Separate acoustic identity from function.** Correctly naming a species or clustering a call does not establish what it communicates.
3. **Make context earn its contribution.** Compare context-aware predictions against acoustic-only, identity-only, location-only, shuffled-context and temporal-permutation controls.
4. **Use causal evidence for reference claims.** Natural correlations can generate hypotheses; they do not by themselves establish that a signal refers to a particular external event.
5. **Treat interaction as a later, governed stage.** Passive recordings come first. Playback requires species/population-specific welfare review, stop conditions and reviewed `ExpressionAuthorization`; model confidence is never authorization.

This plan composes the existing `symthaea-stt::discovery`, `symthaea-communication::animal`, `EvaluationPlan`, `DatasetManifest`, `BenchmarkReport`, and immutable run-bundle verification. Do not create a parallel evidence or capability ladder.

## 2. Existing capability ladder and claim ceiling

Use the repository's `CapabilityLevel` semantics:

| Level | Permitted claim |
|---|---|
| `Signal` | A signal/event was detected under the declared conditions. |
| `Unit` | A repeatable acoustic unit or segmentation was found. |
| `Structure` | Recurrence, sequence, timing or transition regularity generalizes to held-out data. |
| `Reference` | An externally grounded referent/function is supported by preregistered intervention and independent replication. |
| `Intent` | A stronger claim about communicative intent has dedicated evidence and alternative explanations are tested; do not infer it from structure alone. |
| `Dialogue` | A bidirectional exchange is context-sensitive and mutually contingent, beats matched/yoked controls, and replicates independently. |

No level implies access to subjective experience. Keep observed call, behavioral outcome, ecological context, model hypothesis and human interpretation as separate fields. Uncertainty or disagreement must not be collapsed into a single confidence label.

## 3. Benchmark suites and baselines

### Suite A — Detection and taxonomic context

Use **BirdVox-full-night v3** as the first primary acoustic dataset. It supplies six continuous recordings, time/frequency call annotations, approximate sensor GPS, UTC start times, and a stated CC BY 4.0 license. This is much more useful for event-detection and recording-lineage tests than randomized clip UUIDs. Its scope is still narrow: six sensors from one night in one region, so it cannot prove cross-season or cross-region generalization. For a cheap secondary check, use **BirdVox-DCASE-20k** (CC BY 4.0) for its 10-second bird-presence/absence task, but treat it as clip-level development evaluation only: its UUIDs deliberately hide source sensor/time origins from participants. Do not fabricate source-recording IDs or claim leakage-safe field generalization from a random UUID split.

Then use a pinned subset of **BEANS** for standardized animal-audio detection/classification, **BEANS-ZERO** for zero-shot generalization to unfamiliar species/vocalizations, and **BirdSet** for multi-label soundscape evaluation and focal-recording-to-soundscape domain shift. For strong event-boundary evaluation, assess **WABAD** only under its declared non-commercial scope and after per-recording terms are reviewed. Compare multiple baselines where lawful and reproducible: simple energy/spectral features, Symthaea HDC/CfC, a taxonomic detector such as BirdNET, and available foundation-model families such as BirdAVES, BirdMAE, Perch 2.0 and BEATs-NLM. The latest 2026 comparative review reports different leaders across BirdSet and BEANS and large gains from attentive probing for some transformers; benchmark encoder and probe as separate factors rather than treating a raw embedding as a classifier. These models are complementary baselines, not semantic oracles.

The global BirdNET assessment used 4,224 one-minute recordings from 67 sites and reported substantial geographic and species-level variation, with recall generally more heterogeneous than precision. That argues against random clip-only splits and aggregate-only reporting.

### Suite B — Unsupervised unit discovery

Run Symthaea's acoustic unit-discovery pipeline beside:
- a simple energy/spectral segmentation baseline;
- a pinned acoustic embedding + clustering baseline, if its model and license are recorded;
- shuffled-time and noise/negative-control inputs.

Score boundary precision/recall/F1 at a preregistered timing tolerance where strong boundary annotations exist. Also report bootstrap segmentation stability, cluster purity/ARI where valid reference labels exist, recurrence precision, and sensitivity to threshold/recording conditions. A visually persuasive spectrogram is not a metric.

### Suite C — Context-conditioned prediction

Use a defined observable target, such as the next annotated call class or a blinded, pre-defined receiver behavior. Compare the full model with:
- acoustic-only;
- context-only;
- identity-only;
- location-only;
- shuffled-context;
- temporal-permutation;
- held-out-individual and held-out-site evaluation.

All data splits must be grouped by original recording/session and must not share source segments. Symthaea now has an animal-specific `AnimalSplitLineage` validator and a shared metadata-only `BioacousticCorpusManifest` admission validator. `validate_bioacoustic_corpus` checks source/rights references, SHA-256 formatting and duplicate hashes, native sample-rate/channel/sample-count metadata, annotation time/frequency bounds, source-recording split overlap, and optional session/site/individual holdouts. The holdout policy requires at least two declared split partitions. `computed_manifest_blake3()` provides an order-canonicalized identity over the manifest, including rights decision, split assignments, annotation metadata and referenced recording hashes. These validators check declared metadata only; they do not authenticate the referenced bytes, label truth, source identities, or the legal correctness of a human rights decision. Train/test individuals and sites must be disjoint for the corresponding generalization claim. If individual identities are unknown, say so and do not claim individual-level generalization.

### Suite D — Grounded function

Only after passive models produce a preregistered, discriminating prediction should a species-appropriate field protocol be considered. Choose one primary behavioral endpoint, randomize candidate/control conditions, use blinded outcome coders where possible, document non-responses and missing observations, and determine sample size from a prospective power analysis. Establish a minimum practically important effect *before* exposing confirmatory outcomes.

No uncontrolled playback, nesting-season experimentation by default, repeated exposure to force a response, or unattended autonomous broadcast. Welfare concerns or adverse behavior trigger the protocol's stop conditions.

### Suite E — Reciprocal interaction

This is a later milestone, not an MVP. A successful exchange must show that the next model response depends on the bird's latest response and context, rather than merely replaying a fixed sequence. Compare against fixed-playback, randomized-playback and yoked-session controls. Require an independently replicated effect before claiming dialogue.

## 4. Proposed benchmark targets

These are initial engineering goals to evaluate and refine during pilot data characterization; they are **not results** and do not substitute for a preregistered study.

The animal communication module now contains `BinaryDetectionCounts` for a declared operating point (precision, recall, specificity, false-positive rate, F1 and balanced accuracy), `binary_average_precision` for score-ranked, tie-aware average precision, `binary_brier_score` for probability quality, `false_alarm_events_per_hour` for exposure-normalized operational reporting, and `select_threshold_at_max_false_positive_rate` plus `apply_binary_threshold` for validation-only threshold selection and frozen application. These are metric implementations only: no BirdVox examples have been evaluated, and no benchmark score is implied. The threshold must be selected on a validation/calibration split and frozen before test evaluation; validation counts must not be reported as test results. For continuous audio, false-alarm counts must already be event-matched and adjacent threshold crossings merged by a pinned rule before computing the hourly rate. Compare Brier score against a declared base-rate predictor to compute Brier skill. Record the metric convention in each run; average precision is not interchangeable with every trapezoidal PR-AUC implementation.

| Dimension | Initial target / gate |
|---|---|
| Dataset integrity | 100% of evaluated samples have stable IDs, source/license attribution, content hashes and immutable split manifests; reject cross-split sample, recording/session, individual or site overlap for each claimed scope. |
| Detection/classification | Publish event-level and recording/site-level average precision / PR-AUC (declared convention), precision, recall, false-positive rate, false alarm events per hour on continuous audio, and per-taxon/site support. Compare against the strongest relevant pinned baseline with grouped-bootstrap 95% confidence intervals; never hide weak taxa in a macro average. |
| Operational threshold | On a validation split, select the highest-recall threshold under a preregistered maximum window-level FPR; freeze it before test evaluation and publish test confusion counts. Report continuous-recording false-alarm events/hour separately using a pinned event-merging rule. |
| Unit segmentation | Boundary F1 ≥ 0.75 at a task-specific, preregistered tolerance on strongly annotated data, plus bootstrap stability ≥ 0.80. Label these proposed floors until pilot annotation quality is assessed. |
| Context value-add | Balanced accuracy at least 0.05 absolute above the strongest required non-contextual control on the held-out evaluation, with paired 95% CI for the difference excluding zero. Use a suitable alternative primary metric if the outcome is not categorical, chosen before testing. |
| Calibration | Expected calibration error ≤ 0.10 and positive Brier skill versus a declared base-rate baseline where probabilities are emitted; report both Brier score and skill with reliability plots and support counts. |
| Operational false alarms | On continuous recordings, report event-matched false alarms per recording-hour with a frozen event-merging gap/cooldown rule, alongside rather than instead of window-level FPR. |
| Unknown/shifted inputs | Publish coverage-risk curves and out-of-distribution metrics; abstain rather than assigning a familiar label when the sample is outside declared support. No absolute threshold is released before calibration on held-out conditions. |
| Grounded function | Preregistered, powered intervention whose primary effect exceeds the predeclared minimum practically important effect, with a confidence interval excluding the null and independent replication. The required effect size is species/task-specific; do not invent one universally. |
| Dialogue | Context-contingent response beats fixed, shuffled and yoked controls under preregistration and independent replication; welfare protocol remains satisfied. No dialogue claim based only on call imitation or a single response. |
| Reproducibility | Exact source/model/toolchain/configuration, dataset manifest, hardware and per-sample results are pinned in an immutable verified run bundle; missing or stale evidence fails closed. |

If a target proves unrealistic on an independently characterized corpus, amend the threshold before looking at confirmatory results and preserve the old target and rationale in the record. Never tune the gate to make a favorite model pass.

## 5. Split, leakage and negative-control requirements

- Partition original recordings and related session/source material before extracting clips or windows.
- Where available, provide separate evaluations for held-out source recordings/sensors, held-out sites, held-out dates/seasons, and held-out individuals. Do not treat these as interchangeable. BirdVox-full-night supports source-recording/sensor isolation but only one recording night; explicitly mark date/season generalization as untested.
- On continuous recordings, report false alarm events per negative recording-hour as well as event recall at a preregistered temporal matching tolerance. Merge nearby threshold crossings into an event only with a pinned gap/cooldown rule; otherwise count inflation can make the rate incomparable. Keep window-level FPR as a separate metric.
- Select thresholds from validation data under a declared operational constraint (for example maximum false-positive rate); freeze that threshold before held-out test scoring. Also report the full ranking metric, so threshold selection does not hide model quality.
- Keep multiple clips/crops from the same source recording on one side of the split. Require non-empty source-recording IDs and run `validate_animal_split_lineage`; the fixture where two differently named crops share one source recording must fail.
- Pin baseline embedding extraction details. AVES/BirdAVES documentation warns that padding can change embeddings even when sequence lengths are supplied; for a padded encoder baseline, use and report the documented safe extraction strategy (including batch-size-one precomputation where required), then add a batch/padding sensitivity check. Baseline preprocessing is part of model identity, not an incidental implementation detail.
- Freeze label ontology and annotation rules before confirmatory scoring; record disagreement and ambiguous labels.
- Include no-call/silence, background-only and non-target sound controls when represented in the task.
- Preserve negative results, null predictions, abstentions, failed model runs and data exclusions.
- Keep taxonomic identification, acoustic clustering, context prediction and communication claims in separate report fields.

## 6. Data and license discipline

Candidate starting points:
- BirdVox-full-night v3 (continuous recordings, annotations, sensor metadata, CC BY 4.0): https://zenodo.org/records/1205569
- BirdVox-DCASE-20k v2 (10-second labels; source UUIDs hide sensor/time origin): https://zenodo.org/records/1208080
- DCASE 2018 Bird Audio Detection task description: https://dcase.community/challenge2018/task-bird-audio-detection
- BEANS benchmark and datasets: https://arxiv.org/abs/2210.12300 and https://github.com/earthspecies/beans
- WABAD data paper: https://doi.org/10.1002/ecy.70317; full dataset record and dataset description: https://zenodo.org/records/17293588
- AVES/BirdAVES reference implementation and documented embedding/batching caveat: https://github.com/earthspecies/aves
- BEANS-ZERO background via Earth Species Project benchmarks: https://earthspecies.org/what-we-do/
- 2026 foundation-model benchmark comparison across BEANS and BirdSet: https://doi.org/10.1016/j.ecoinf.2026.103765
- BirdSet dataset collection and focal/soundscape evaluation setup: https://github.com/DBD-research-group/BirdSet
- BirdNET model documentation (including model scope and license): https://birdnet-team.github.io/BirdNET-Analyzer/stable/models.html
- Global BirdNET evaluation: https://doi.org/10.1016/j.ecolind.2025.114550

**Primary corpus: BirdVox-full-night v3.** Dataset record: https://zenodo.org/records/1205569. The record specifies CC BY 4.0 and includes six continuous mono FLAC recordings, time/frequency annotations, approximate GPS coordinates and UTC start times. The source record includes 35,402 flight-call annotations and flags 29 device-generated beeps as `alarm`; do not count those beeps as positive bird-call events. Keep them available as a labeled hard-negative/device-artifact control. Cite the ICASSP 2018 BirdVox-full-night paper and retain attribution. The six sensors are from a single night; holding out one sensor tests limited sensor/site variation, not generalization to other seasons, ecosystems or continents.

**Secondary clip-level corpus: BirdVox-DCASE-20k v2.** Dataset record: https://zenodo.org/records/1208080. The record specifies CC BY 4.0 and 20,000 ten-second clips, 10,017 positive; however, their random UUIDs obscure source sensor/time origins from participants, and labels were automatically derived from BirdVox-full-night annotations. Use for an established coarse clip-level comparison only. Because source lineage is hidden, do not claim the random train/test split is source-recording independent; do not invent a source-recording ID from the clip UUID. Do not pool its clips with full-night recordings across splits.

**WABAD license-admission gate:** the dataset description on the Zenodo record states CC BY-NC 4.0 for the dataset; the journal article itself is published under CC BY-NC-ND 4.0. These refer to distinct assets and should not be described as a CC BY 4.0 grant. Treat commercial use of the dataset as blocked unless rights holders give clear permission. Before use, verify the terms attached to the exact files and their underlying recording sources, determine whether contemplated derived artifacts or redistribution are permitted, and preserve this decision with source evidence in the corpus manifest. No recordings should be checked into the repo.

WABAD is a high-value candidate for passive validation because it reports 5,047 minutes and 91,931 expert-annotated vocalizations from 1,192 species across 72 sites, with vocalization start/end times and frequency bounds. These counts describe the published dataset, not data already downloaded, audited or usable by this repository.

Pin the exact dataset/version and each artifact's license before use. Public availability is not a blanket redistribution or commercial-use grant. In particular, record the actual model license and intended use for every baseline. Do not commit raw wildlife recordings into the source repository without explicit redistribution rights; publish source references, manifests and permitted derived features instead.

## 7. Implementation sequence

1. **Contract hardening (this tranche):** require all registered context controls; reject incomplete/duplicate/blank split identities, non-finite temporal-permutation evidence, and train/test overlap in source-recording lineage.
2. **Passive corpus audit:** start with BirdVox-full-night v3, build a `BioacousticCorpusManifest`, pin audio/annotation/metadata hashes, exclude device beeps from positive bird events, and split on source recording/sensor. State clearly that six sensors from one night cannot establish temporal or regional generalization. Use BirdVox-DCASE-20k only as a secondary clip-level comparison because its randomized UUIDs conceal lineage. Audit broader taxa/sites with BirdSet/BEANS after their constituent licenses and source lineage are recorded. No communication or training-quality claim yet.
3. **Baseline harness:** build a metadata-first manifest for BirdVox-full-night v3 and its six source recordings, link time/frequency annotations, isolate device alarms, and run a nested sensor-grouped evaluation. Select thresholds only inside each outer fold's training/validation data, freeze them before scoring the held-out sensor, and report both window/event metrics plus false alarm events/hour. Use BirdVox-DCASE-20k only as a secondary clip-level comparison because its source lineage is hidden. Emit per-sample predictions and a separate unit-discovery report using Symthaea's existing immutable run bundle. Metric functions alone do not qualify a run or establish performance.

Once that metadata manifest exists, run the preflight before model work:
```sh
cargo run -p symthaea-communication --bin bioacoustic-corpus-preflight -- birdvox-full-night-manifest.json \
  --require-source-recording-ids --require-annotations
```
The CLI validates declared metadata and emits a canonical manifest BLAKE3 identity. A preflight pass is not a benchmark result and does not authenticate file bytes or the human-reviewed rights decision.
4. **Context prediction:** preregister one observable target and run every required control on held-out identities/sites/time.
5. **Expert review:** reconcile annotation uncertainty and review negative controls before using words like “meaning.”
6. **Intervention proposal:** only after the passive prediction gate passes, submit a species-specific welfare-reviewed protocol.
7. **Reciprocal communication:** only after grounded-function evidence replicates.

## 8. Claim boundary for this research contract

A passing software test establishes only that the declared validator rejects malformed or incomplete evaluation evidence. It does **not** establish that Symthaea recognizes bird calls, understands their meaning, communicates with birds, or is ready to conduct playback. Those are separate empirical milestones and require real, legally usable recordings and appropriately reviewed field evidence.

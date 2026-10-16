# Cetacean Communication Research Contract v1

**Status:** research and engineering plan plus unexecuted source changes. No cetacean dataset has been scored in this work; no existing code path is promoted to a supported or grounded-meaning capability by this contract.

## 1. Existing Symthaea work — reuse it, do not create a parallel project

The public `symthaea-stt` crate already contains:

- `whale.rs`: a streaming sperm-whale coda prototype based on click timing and inter-click intervals (ICI), with HDC signatures and adaptive prototypes;
- `cetacean_classifier.rs`: a frame-level acoustic feature classifier for transient/tonal/noisy events and frequency modulation;
- `cetacean_scorer.rs`: HDC sequence scoring over a small set of hand-specified cetacean acoustic units;
- `temporal_grammar.rs`, `unified_grammar.rs`, and `liquid_hdc.rs`: temporal sequence / liquid-dynamics prototypes;
- `symthaea-bioacoustics`, the acoustic unit-discovery path, and multiple whale-validation / cetacean demo examples.

The repository's own `crates/core/symthaea-stt/README.md` and `docs/communication-capability-inventory.md` describe cetacean modules as research prototypes with synthetic or heuristic labels and cap the evidence claim at acoustic structure. That is the right claim ceiling until source-pinned, recording-disjoint results exist.

**Legacy example integrity correction:** `whale_validation.rs` and `whale_validation_v2.rs` through `v4` previously used file-list-position splits and inferred secondary species labels from filename substrings while printing “SUCCESS” language about learning/generalization. The research branch now labels them exploratory diagnostics, adds evidence-limit warnings, changes misleading headings/summary labels, and removes those success claims. They still do not implement source/encounter-grouped evaluation or independent species ground truth; the outputs must not be cited as validation.

**Immediate audit priority:** inspect and replace validation examples that derive labels from filename substrings or split sorted files by position. A train/test file split is not independent evidence when related crops, encounters or recorders are shared. Generated whistles/codas are software fixtures only—not field or communication evidence.

## 2. Engineering hardening now in the working PR branch

This branch adds guarded paths in `cetacean_classifier.rs`:

- `CetaceanConfig::dolphin_high_rate_research()` supplies an explicitly experimental 512 kHz / 150 kHz upper-band configuration for high-rate dolphin-audio exploration. This makes the declared front-end profile bandwidth-consistent; it does **not** qualify an echolocation-click detector or whistle-contour tracker.
- `CetaceanConfig::validate` checks finite/ordered analysis bands, Nyquist limit, frame/hop geometry, FFT length and threshold ranges;
- `CetaceanConfig::validate_source_sample_rate` rejects source bandwidth incompatible with the declared analysis band and refuses implicit downsampling through the current linear interpolator;
- `CetaceanClassifier::try_new` and `try_classify_audio` provide checked construction and source-rate-aware processing; file classification uses the checked path.

It also adds guarded paths in `whale.rs`:

- `WhaleConfig::for_native_sample_rate()` binds the sperm-whale ICI timing profile to a declared native recording clock without resampling;
- `WhaleConfig::validate` checks the sample clock, timing bounds, click-count limits and overflow-prone LTC time constants;
- `try_with_config` / `try_process_buffer` reject invalid configurations, empty/non-finite input and sample-rate mismatch;
- the coda ICI buffer now retains at most `max_coda_clicks - 1` intervals, matching the advertised maximum number of clicks.

The older infallible APIs remain for compatibility and must not be used to claim checked processing. A high-frequency signal must be analyzed at a supported native rate or passed through a separately validated anti-aliasing resampler. Do not simply resample ultrasonic recordings to 44.1 kHz and then claim click-spectrum analysis.

These edits and their regression tests have **not** been compiled or executed in this session.

The same working branch now adds `symthaea-communication::bioacoustic_corpus`: a metadata-only `BioacousticCorpusManifest` and `validate_bioacoustic_corpus` gate shared by bird and cetacean tracks. It checks rights-decision references and intended-use agreement, hash syntax/duplicates, sample-rate/channel/sample-count validity, annotation time/frequency bounds against native duration/Nyquist, cross-split source-recording leakage, and declared session/site/individual holdout policies. If a holdout is claimed, at least two distinct split partitions are required. `computed_manifest_blake3()` supplies an order-canonicalized manifest identity that changes when split assignments, rights evidence, annotation metadata or recording hashes change. The validator cannot read/hash the actual audio bytes or establish that a human-entered rights decision is legally correct; the caller must compute hashes from exact assets and keep its rights evidence auditable.

## 3. Paradigm shift: one evidence system, separate acoustic tasks

Do not model all cetacean sounds as one universal "cetacean phoneme" classification problem. Use shared provenance, calibration, leakage, null-control and capability-evidence contracts, but task-specific front ends and metrics.

| Research lane | Relevant signal and context | First scientifically defensible output |
|---|---|---|
| Bottlenose / common dolphins: whistles | Sustained frequency contours, harmonics, overlapping calls, individual/encounter metadata where actually known | Contour detection / tracking, contour error and held-out-session generalization |
| Dolphins: echolocation clicks | Short broadband transients, high sample rates, multi-channel geometry where available | Click-event detection and timing; later source localization only with array calibration / ground truth |

**Important distinction:** `WhaleSentinel` is a sperm-whale coda-ICI prototype, not a drop-in dolphin click detector. Its default `min_click_gap` is 20 ms; do not reuse that operating point for faster dolphin click trains without a task-specific configuration and annotated validation.
| Dolphins: burst pulses | Dense pulse trains, broad spectral content and context | Detection / segmentation with class-specific errors, not an assumed social meaning |
| Sperm whales: codas | Click-event sequences and inter-click interval patterns, often social-unit / individual metadata | Coda boundaries, click counts, ICI error and held-out-encounter structure |
| Sperm whales: spectral / "vowel" features | Spectral shape within validated click/coda segments | Feature reproducibility and held-out social-unit classification; not human-phoneme equivalence |
| Baleen whales: songs and calls | Long temporal structure and lower-frequency vocalization bands | Song/call segmentation and detection over continuous audio; report region, season and recorder domain shifts separately |

Paradigm shifts to pursue:

1. **From translation to falsifiable prediction.** First ask whether a pattern predicts a predeclared subsequent acoustic event or an independently annotated behavior beyond strong baselines. Next-event prediction does not itself identify meaning.
2. **From one universal encoder to native-rate, task-aware processing.** Preserve source sampling rates, calibration, channel count and event duration. Treat sample-rate conversion and filtering as versioned model inputs.
3. **From clip accuracy to continuous-monitoring utility.** Measure event recall, timing errors and false alarms per recording-hour. High clip accuracy can hide a detector that fires continuously.
4. **From random clips to provenance-grouped generalization.** Split by original recording, encounter/session, day, recorder/site and known individual according to the actual claim. Keep each related crop in one split.
5. **From model-only results to adversarial controls.** Compare energy/spectral baselines, HDC/CfC/temporal grammar, strong released encoders/models when their terms allow it, and label-shuffled/time-permuted/recording-identity-only controls.
6. **From call similarity to grounded behavior.** Align acoustic predictions with time-synchronized visual / behavioral observations when legitimately available. Require preregistered interventions to advance beyond association; human descriptions remain annotations, not ground truth for an animal's private state.

## 4. Dataset order and license admission

### A. Dolphins — rich acoustic data candidate, but legal admission is still required

**DOLPHINFREE common-dolphin dataset**: the 2025 *Earth System Science Data* data paper reports over 400 minutes of recordings from 2020–2022, around 68,000 echolocation clicks, 4,637 annotated whistle contours and more than 350 pulsed sounds. It describes mono recordings at 512 kHz and four-hydrophone-array recordings at 256 or 512 kHz. This is unusually useful because it combines high-band audio, whistle contours, multichannel data and behavioral observations.

- Data record: https://zenodo.org/records/14637675
- Data paper: https://doi.org/10.5194/essd-17-4495-2025

**License gate:** the paper itself is CC BY 4.0, but the opened Zenodo data record has an empty license field. Do not assume the article license grants reuse rights to every audio recording, annotation and observation file. Confirm dataset/file and underlying source rights with the maintainers before using it in a training/evaluation product or redistributing derived data. The archive is approximately 39.9 GB; do not commit raw recordings to the repository.

### B. Dolphins — a light detection candidate, pending license verification

**OpenWhistle-CNN** exposes whistle/noise windows, recording/session IDs and onset/offset metadata and describes session-disjoint train/validation/test partitions. That makes it a convenient candidate for a whistle detector and a leakage-audit fixture.

- Dataset card: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN

The accessible dataset card does not state a clear license. Do not treat public hosting as a grant of permission: resolve its dataset/model/source-audio terms before admissibility. Verify that source sessions truly do not overlap and that no duplicated crops cross splits.

NOAA SanctSound hourly dolphin-presence records may provide an additional large-scale passive-monitoring task. They are hourly detections derived from PAMGuard output, not event-level annotations; use only for the coarser presence task after pinning the exact NCEI asset, processing provenance and reuse terms: https://catalog.data.gov/dataset/dolphin-sound-production-recorded-at-sanctsound-site-pm05_01

### C. Sperm whales — coda segmentation and sequence benchmarks

- **Project CETI Coda-detector** is a useful method baseline for coda detection / click timing and vocalizer separation. Its public repository includes example recordings from previously published datasets, not a guarantee that full source datasets are openly redistributable: https://github.com/Project-CETI/Coda-detector
- **Project CETI coda-vowel-phonology** provides code and tabular materials relevant to sperm-whale coda spectral features: https://github.com/Project-CETI/coda-vowel-phonology
- **DolphinGemma** is a relevant dolphin-sequence baseline candidate: Google DeepMind describes it as learning patterns and generating dolphin-like acoustic sequences from Wild Dolphin Project data; this does not establish semantic translation or mutual understanding. Pin exact weights / license / version before use: https://deepmind.google/models/gemma/dolphingemma/
- **WhAM** (NeurIPS 2025) is a transformer-based audio-to-audio coda model with embeddings for downstream social-unit and spectral-feature tasks. Its code repository is MIT-licensed, but full DSWP+CETI annotations required to reproduce all published results are not all publicly available. Treat model code, pretrained weights, data and annotations as separate license/access objects. Generation of pseudo-codas is not evidence of communication: https://github.com/Project-CETI/wham
- **Dominica Sperm Whale Project (DSWP)** is identified by WhAM as a publicly available coda dataset, but pin its exact release, source lineage, annotation availability and license before use.

### D. Cross-cetacean and baleen-whale monitoring

DCLDE (Detection, Classification, Localization, and Density Estimation) provides useful passive-acoustic benchmark traditions and competition datasets; terms vary by edition and track, so do not assume a single umbrella license. Use it for species-specific detection/classification only after pinning the exact release and the ground-truth scope.

## 5. Benchmark ladder

All numerical thresholds below are **proposed targets**; they are not measured Symthaea performance and need task-specific pilot characterization.

### Level 0 — input / acoustic integrity

- 100% of samples bind file-content hash, native sample rate, channels, bit depth where known, time interval, recorder/session identity, license and source URI/version.
- Reject a declared analysis upper frequency at or above the applicable source/config Nyquist limit.
- No implicit high-rate downsampling with the current linear interpolator.
- Report clipping, non-finite samples, channel mixdown and filtering/resampling steps.
- Preserve the original audio identity; preprocessing creates a distinct derived-artifact identity.

### Level 1 — detection / segmentation

For dolphin whistles, clicks and burst pulses; sperm-whale clicks/codas; baleen-whale calls/songs, score each task separately:

- Event precision, recall, F1 and average precision (with a declared convention).
- Onset/offset error and event F1 at a preregistered temporal matching tolerance.
- Frequency-contour error for whistles and ICI absolute error for codas.
- False-alarm events per negative recording-hour for continuous recordings, using a frozen threshold-crossing merge/cooldown rule; report window-level FPR separately.
- Per-recording / per-session support and grouped uncertainty. Do not disguise class imbalance with accuracy alone.

Initial engineering gates to evaluate after a manual-label audit:
- Event F1 ≥ 0.80 for the defined detection task on clean, held-out recordings;
- event-based false alarms per negative recording-hour must meet a threshold preregistered for the intended monitoring use;
- whistle-contour and ICI timing tolerances must be derived from annotation precision and scientific task needs, not selected after results are seen.

### Level 2 — structure and predictive value

- Compare HDC/CfC temporal structure against a constant/base-rate predictor, simple ICI/renewal or n-gram model and a suitable self-supervised acoustic representation. For dolphin sequence prediction, include DolphinGemma as a comparison only where the exact model weights, intended-use license and reproducible preprocessing are available; its published role is to learn/predict acoustic sequence structure, not to serve as a proven natural-language translator.
- Report next-event negative log-likelihood / Brier score or task-appropriate predictive metric, against shuffled-event, temporal-permutation and context-only controls.
- Require improvement on held-out recording/session groups with paired/grouped confidence intervals; keep null results.
- For coda families, report balanced accuracy / macro-F1 only when the label ontology and source identity are defensible.

### Level 3 — social identity and behavioral grounding

- Individual or social-unit prediction only when identity labels are independently grounded. Hold out entire encounters / individuals / days—not adjacent clips.
- Behavioral association must beat acoustic-only, behavior-prior-only, identity-only, time-shuffled and site-only controls.
- Report annotator agreement, unknown identity, overlapping callers, event-quality exclusions and uncertainty.
- Do not convert a classifier score into a natural-language "translation."

### Level 4 — reciprocal communication (later)

A communication claim requires a species/population-specific, ethically reviewed, preregistered interaction whose outcome depends on the animal's contingent response; fixed-playback, randomized and yoked-session controls; a welfare stop rule; and independent replication. Acoustic imitation or a generated pseudo-coda does not pass this gate. Do not autonomously broadcast novel sounds to wild cetaceans.

## 6. Code-specific next steps

1. Run the new classifier and whale configuration tests on a real Rust toolchain; inspect formatting / Clippy and do not call skipped CI a pass.
2. Replace the list-position partitions and filename-derived labels inside the now-explicitly-exploratory `whale_validation.rs` and v2–v4 demos with a versioned annotation manifest and recording/encounter-grouped evaluation harness. The display labels have been corrected, but the underlying split/label method is still unsuitable for validation.
3. Use the new shared `BioacousticCorpusManifest` / `validate_bioacoustic_corpus` contract for a Rust-first metadata-only preflight. Populate actual content hashes from the exact assets, record `computed_manifest_blake3()`, resolve source/recording lineage, verify annotation coverage, and record a human-reviewed license decision. Do not copy raw audio into the repo.

When a manifest is prepared, the preflight CLI can validate it without downloading audio:
```sh
cargo run -p symthaea-communication --bin bioacoustic-corpus-preflight -- corpus-manifest.json \
  --require-source-recording-ids --require-annotations
```
Add `--hold-out-sessions`, `--hold-out-sites`, or `--hold-out-individuals` only where the metadata actually supports those claims. The output explicitly states that audio bytes, legal rights and model performance are not independently verified by this preflight.
4. Build the first permissible small subset / metadata preflight only after rights admission; prioritize whistle contours (rich labels) and coda timing as two separate tracks rather than one blended metric.
5. Compare the legacy Symthaea prototypes against explicit baselines and released model families; pin each model's license, weights hash and preprocessing, and record compute/hardware.
6. Keep the shared `symthaea-communication` capability ladder: Signal → Unit → Structure → Reference → Intent → Dialogue. A passing software benchmark can establish only the declared measurement capability; it cannot establish subjective meaning or conscious understanding.

## 7. Claim boundary

This contract and the implementation patches do not establish that Symthaea understands dolphins or whales. The existing cetacean modules remain experimental until pinned datasets, non-leaking splits, independent scoring and the relevant capability-release evidence are qualified. The source-rate guards prevent certain invalid analysis configurations; they do not themselves detect or interpret real-world cetacean sounds.

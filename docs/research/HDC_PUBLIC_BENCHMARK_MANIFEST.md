# Public HDC/VSA Benchmark Manifest

**Status:** metadata-only manifest and schema authored; exact-head Rust validation pending  
**Manifest:** `docs/research/hdc-public-benchmark-manifest.json`  
**Schema:** `docs/research/schemas/hdc-public-benchmark-manifest-v1.schema.json`  
**Validator:** `crates/core/symthaea-core/examples/hdc_public_benchmark_manifest.rs`

## Why this is the next paradigm shift

Symthaea should not argue that a synthetic retrieval experiment proves a field-wide HDC improvement. It should first reproduce public reference tasks, then compare methods under identical data splits, feature pipelines, tuning budgets and resource accounting.

Vergés et al. (2025) define a seven-task HDC/VSA classification suite and a 121-task UCI suite. Their study reports that conventional classifiers—especially SVM—achieved the strongest average accuracy in both suites; the paper reports about a 2.5-point advantage on its HDC suite and 1.5 points on the UCI suite over prototype-based classifiers, and that SVM/LDA inference can be faster than HDC methods in their tested setup. These are results from that paper's implementation and hardware, not measurements of Symthaea. They are a clear warning: representation novelty alone is not a win.

Reference:
- Vergés et al. (2025), *Classification using hyperdimensional computing: a review with comparative analysis*: https://doi.org/10.1007/s10462-025-11181-2
- Reference HDC/VSA implementation: https://github.com/hyperdimensional-computing/torchhd

## What's recorded

The manifest lists the seven reference tasks: CIFAR-10, EMGHandGestures, European Languages, ISOLET, MNIST, PAMAP2, and UCI HAR. Each entry records source citations, upstream licence claims/evidence, preprocessing status, split policy and artifact status.

Important details are deliberately exposed rather than hidden:

- **CIFAR-10 and MNIST:** the reference benchmark uses features from a pretrained two-layer CNN. Raw-pixel results cannot be presented as exact reproduction of that feature-based track without a separately documented reason and experiment. Feature extractor code, weights, normalization and the transformed data digest still need pinning.
- **European Languages:** training text is from Wortschatz, while testing is from Europarl; the reference uses character-trigram counts. This is a cross-corpus evaluation, not a random split. Both corpora require license/terms review before redistributing raw text or derived artifacts.
- **ISOLET:** preserve the original speaker-grouped training/test partition.
- **UCI HAR:** preserve the packaged subject-disjoint split; raw-stream processing is a distinct experiment.
- **EMG and PAMAP2:** preserve subject/session/time structure and prevent temporally adjacent or overlapping windows leaking between train and test.

The paper reports 10,000 dimensions and five repetitions for its comparison. These settings are captured as the initial reference target, not declared as the best setting for every workload. The exact benchmark code revision and each task's transforms still need to be pinned before calling the result a reproduction.

## Dataset provenance and distribution rules

No source data or derived artifact has been downloaded or embedded by this PR. Every `artifact.status` remains `not_materialized`, with null SHA-256 and byte-size fields.

Upstream license strings are evidence to review, not a legal approval by this project. For example, current UCI pages advertise CC BY 4.0 metadata for ISOLET, EMG Data for Gestures, UCI HAR and PAMAP2, while PAMAP2's original README separately describes academic research use. MNIST is reported as CC BY-SA 3.0 in Keras documentation; CIFAR-10 and the combined Wortschatz/Europarl language task require explicit terms review. The manifest therefore records its evidence URL and review state, and prohibits redistribution by this project until usage is reviewed.

Source pages:
- ISOLET: https://archive.ics.uci.edu/dataset/54/isolet
- EMG Data for Gestures: https://archive.ics.uci.edu/dataset/481/emg%2Bdata
- PAMAP2: https://archive.ics.uci.edu/dataset/231/pamap2%2Bphysical%2Bactivity%2Bmonitoring
- UCI HAR: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones
- MNIST license information: https://keras.io/api/datasets/mnist/
- Europarl corpus and provider terms: https://www.statmt.org/europarl/
- Wortschatz source: https://wortschatz.uni-leipzig.de/en/download/

## Comparison tracks

1. **Reference defaults:** reproduce the published default parameters and feature pipeline, with source code/weights/split digests pinned.
2. **Equal-budget tuning:** tune only on train/validation data; disclose the same search budget per method. Never select settings with the test split.
3. **Resource-matched:** compare equal dimension and equal model-byte budgets. Count packed bits properly; include preprocessing, indexes, optimizer state and model metadata.

The initial methods register prototype/MAP/BSC/HRR, published OnlineHD/AdaptHD/DistHD/SparseHD methods, and SVM/LDA/Ridge both on original and shared encoded features. These are **planned comparator identities**, not implemented algorithms and not benchmark results. An adapter must reproduce the named method's operations and protocol before it can become qualified.

## Required reporting

Per task and seed, publish accuracy, macro-F1 and per-class recall; training/update costs; p50/p95/p99 single-query latency; batched throughput; measured peak RSS; actual model bytes; and, where relevant, selective risk, accepted coverage and OOD false-accept rate. Report each task before aggregates, including failures and missing configurations. Add paired uncertainty analysis and correction for multi-task/multi-method claims.

A single scalar ranking is insufficient: task quality, reliability, memory and latency should be separate metrics, then analyzed as a Pareto frontier.

## Validation and claim ceiling

The Rust validator checks the manifest's task inventory, source citations/URLs, split/preprocessing declarations, comparator coverage, licence-review states and fail-closed artifact status. It emits a receipt binding the exact source SHA supplied by CI, plus BLAKE3 digests of the manifest and schema. It is a semantic contract check; it is not a full JSON Schema engine, license audit, dataset downloader, or classifier benchmark.

A passing manifest check only establishes that the metadata contract is internally consistent. Benchmark reproduction, legal review, raw-artifact hashing and all classifier comparisons remain future work. No SOTA claim is made.

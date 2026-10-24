# Cetacean Frontier Research Notes — October 2026

**Purpose:** identify high-value work that can start immediately in Symthaea while CI and lawful-data admission proceed. This is a research synthesis and implementation plan, not evidence that Symthaea has reproduced the cited results.

## Executive decision

Do not start another generic "dolphin translation" model. The highest-leverage near-term work is a reproducible **longitudinal acoustic-structure benchmark** with three connected tracks:

1. **Whistle detection and signature-whistle classification** on the new OpenWhistle longitudinal benchmark, using its published setup as a reproducibility target and a separate chronological/domain-shift evaluation.
2. **Sperm-whale coda sequence prediction across dyads**, testing whether a responder's next-coda features are predictable from a partner's previous coda sequence beyond rhythm-only, identity-only and history-only controls.
3. **Auditable evaluation infrastructure**, now being added in Symthaea: byte-level integrity, explicit event annotations, one-to-one event matching, a frozen test threshold, precision-recall reporting and provenance-bound reports.

The breakthrough would not be a flashy generated call or an unsupported translation. It would be a system that exposes reproducible temporal structure, identifies exactly which context provides predictive information, withstands ecological shift and null controls, and can express uncertainty when the data do not support an answer.

## 1. Research published or surfaced in 2026

### A. OpenWhistle: prioritize longitudinal within-system learning

The OpenWhistle benchmark page describes approximately 180,000 bottlenose-dolphin whistles, 114 hours of audio, a five-year longitudinal span, and 8,354 expert labels, with seven signature-whistle types and three non-signature types. It advertises a NeurIPS 2026 Spotlight and reports a clear advantage for in-domain pretraining on the reported fixed benchmark.

Reported baselines on the project page include:

- A released Wav2Vec2/OpenWhistle checkpoint: full-set 10-class macro-F1 64.37 ± 2.67, natural-six-class macro-F1 59.91 ± 2.57, balanced-six-class macro-F1 81.67 ± 1.80, and detection mAP 75.76 ± 2.02.
- Wav2Vec2 pretraining on 100% of the longitudinal OpenWhistle audio: detection mAP 80.80 ± 1.91 in the site's data-scale comparison.
- The task, splits and uncertainty conventions must be read from the exact benchmark release before comparing against these numbers; macro-F1, window detection mAP and continuous-recording event F1 are not interchangeable.

Sources:
- Project benchmark and data: https://sumbre-lab.github.io/openwhistle/
- 2026 preprint: https://arxiv.org/abs/2609.34839

**Symthaea experiment:** first reproduce a clearly specified public baseline if data/model terms permit. Then preregister chronological splits (train earlier years, validate later, test on the latest held-out period) and any individual/group or recording-site split that is supported by ground-truth metadata. Evaluate how much performance falls as temporal distance increases. Never infer an individual from filename conventions.

### B. Dolph2Vec: use species-specific self-supervision as a baseline, not a claim of semantics

The June 2026 Dolph2Vec preprint describes a five-year longitudinal dataset covering five known dolphins and a Wav2Vec2-based species-specific self-supervised representation model. It studies signature-whistle classification and detection, and reports interpretable embeddings/codebook structure aligned with annotated whistle categories.

Source: https://arxiv.org/abs/2606.12503

**Symthaea experiment:** compare HDC/CfC/temporal representations and suitable acoustic encoders under identical data splits, label budget, preprocessing and scoring. Include label-efficiency curves at fixed annotation budgets (e.g., 1%, 5%, 10%, 25%, 100% of permitted labels); compare repeated-seed variance; test representation stability across years and recording conditions. An interpretable codebook is a hypothesis generator until it is stable and independently validated.

### C. Sperm-whale dyads: make cross-animal response prediction a primary task

A June 2026 study of simultaneous coda sequences from sperm-whale pairs reports that the next coda produced by one whale was predictable from up to ten preceding codas produced by the other. Features included coda duration, the first and second inter-click intervals, and the delay between codas.

Source: https://doi.org/10.1080/09524622.2026.2684665

**Symthaea experiment:** represent every dyadic exchange as a timestamped interaction graph, with caller identity and uncertainty, coda features, inter-coda delay, overlap and preceding history. Test next-coda prediction on held-out encounters/dyads against:

- marginal/base-rate predictor;
- target-whale history-only model;
- partner-history model with order shuffled;
- timing/rhythm-only baseline;
- pair/individual identity-only baseline;
- full sequential model with cross-whale history.

Primary outcomes should be held-out negative log-likelihood (NLL), calibration and paired/grouped improvement over baselines. Use log-loss differences and confidence intervals clustered by encounter/dyad, not a random crop-level split. Prediction supports structure; it does not show what a coda means.

### D. Two-tier structure in sperm-whale codas: replicate with explicit acoustic nulls

A June 2026 preprint analyses 1,483 codas and reports a two-tier, duality-of-patterning-like pattern: click/coda organization is strongly rhythmic at the lower tier, while coda sequences show higher-level dependence. It reports a second-order transfer-entropy lift of 0.132 bits (p = 0.002), and argues that rhythm-only baselines do not reproduce the upper-level sequential signal. It explicitly avoids claiming semantics or human-like phonemes.

Source: https://arxiv.org/abs/2606.16084

Treat this as a replication target, not settled fact. Test it with multiple encoders, rhythm-only controls, tempo scaling, corpus splits grouped by social unit/encounter, appropriate multiple-testing control and nulls designed to preserve the marginal distribution and autocorrelation structure. The key test is whether the upper-level dependence survives acoustic-null recovery and independent data.

### E. Cross-site minke detection: pooled accuracy hides operational collapse

A 2026 open-access study of Antarctic minke-whale detection reports strong site variation: precision-recall AUC is above 0.95 at some sites but around 0.60 at Pilbara; false positives per hour also vary by location and operating point.

Source: https://doi.org/10.1111/mms.70118

**Symthaea experiment:** every detector report should include per-site/per-recorder performance, fixed-operating-point false alarms/hour and an explicitly held-out site where feasible. A model can earn a high pooled score while being unreliable in a new acoustic environment.

### F. Borrow the right evaluation discipline from DCASE bioacoustics

DCASE's few-shot bioacoustic event-detection benchmark uses event-based macro F-measure with temporal IoU and bipartite matching. It explicitly handles unknown/ambiguous labels separately rather than blindly treating them as negatives, and provides per-dataset results because conditions vary across sources: https://dcase.community/challenge2024/task-few-shot-bioacoustic-event-detection

**Symthaea implication:** a credible continuous-audio scorer must support (1) one-to-one event matching, (2) declared event/onset tolerance, (3) unscored regions, (4) event-level precision-recall, (5) exposure time excluding the unscored regions, and (6) per-recording and per-domain breakdowns. The new schema-v3 corpus manifest records explicit unscored intervals and reasons; the event evaluator excludes affected predictions deterministically and reports how many were omitted. Those exclusions remain annotation assertions, not ground-truth authentication.

Do not copy DCASE's exact metric blindly to every cetacean task. Sperm-whale clicks may need onset tolerance and ICI error, whistle segments may need IoU / contour error, and dyadic sequence models require proper scoring rules such as held-out NLL. Metric choice must follow task semantics and be fixed before the Test data is inspected.

## 2. The code gap being closed now

Symthaea's existing public branch contains:

- a metadata-only BioacousticCorpusManifest and rights/split validator;
- bioacoustic-corpus-preflight for declared metadata;
- bioacoustic-asset-audit to verify local file bytes against declared SHA-256 hashes;
- bioacoustic-event-eval to score precomputed event predictions on the Test partition, with explicit exhaustive-annotation assertions, one-to-one matching, event precision/recall/F1, event AP, false alarms/hour and per-recording metrics;
- manifest and prediction-bundle hashes in the report, plus explicit flags for things this evaluator has not authenticated or executed.

The validator was refactored from pairwise recording comparisons to map-based split ownership so larger window corpora do not force O(N²) split validation. Event matching was indexed by time to reduce irrelevant pair comparisons. Authored stress/regression tests are not equivalent to measured runtime; these changes still need to compile and run on a real toolchain.

References:
- PR: https://github.com/Luminous-Dynamics/symthaea/pull/7558
- Implementation issue: https://github.com/Luminous-Dynamics/symthaea/issues/7605

## 3. Next implementation milestones

### Milestone 1 — exact reproducibility
- Pin one permitted benchmark release, licenses/terms for audio, annotations, pretrained weights and derived artifacts.
- Record immutable revision, exact hashes, split membership and annotation source.
- Reproduce the cited baseline with the original split and metric definition before changing the architecture.
- Keep the result, per-sample outputs, configuration and hardware records immutable.

### Milestone 2 — generalization beyond the easy split
- Use chronological evaluation and recording/session/encounter grouping rather than random crops.
- Report performance as a function of temporal distance, source-device/site shift and label budget.
- Add label-shuffled, time-shuffled, identity-only and context-only controls. Preserve hard negatives and unknown labels.
- Use confidence intervals clustered by the actual independent unit.

### Milestone 3 — interaction and sequence
- Build dyadic-coda and contiguous whistle-sequence examples that preserve order, timing and caller uncertainty.
- Compare history-only, rhythm-only, partner-history and full-context models under identical splits.
- Score held-out likelihood/calibration, not just cluster pictures or plausible generated sounds.
- Test whether features discovered in one season/encounter recur in another before naming them as stable acoustic units.

### Milestone 4 — behavioral grounding, only if warranted
- Preregister an externally observable, independently annotated outcome.
- Blind annotators/modelers where practicable and report annotator disagreement.
- Control for individual, pair, site, season, observer, time and recorder effects.
- Keep passive observation as the default. Any playback or intervention requires separate animal-welfare/ethics review and an appropriate field protocol.

## 4. Acceptance gates

- **Reproduction:** published baseline and uncertainty reproduced within a predeclared tolerance under the correct dataset split/metric; otherwise investigate and report the discrepancy.
- **Generalization:** chronological/group-held-out performance is reported, including breakdowns and failures; no claim is promoted from random-crop results alone.
- **Incremental value:** full sequential/context model beats preregistered history-only and identity-only baselines by a defined margin, with an encounter/dyad-clustered confidence interval excluding zero.
- **Null robustness:** shuffled-label/context and acoustic-null controls remove the target effect as predicted; a result that survives a broken negative control is not accepted.
- **Calibration:** report NLL/Brier score or suitable proper scoring rules and reliability, not only classification accuracy.
- **Scientific claim discipline:** event detection or next-coda prediction may establish detection/predictive structure; behavioral association requires independent annotations; communicative meaning requires further task-specific evidence.

## 5. Rights and naming hygiene

The 2026 OpenWhistle longitudinal benchmark is not the same dataset object as the older dolphinteam/OpenWhistle-CNN 76,516-window binary whistle/noise dataset card. Do not conflate their licenses, splits, metadata, scores or available audio. Pin each exact source, model weights and intended use separately.

- OpenWhistle longitudinal benchmark: https://sumbre-lab.github.io/openwhistle/ and https://arxiv.org/abs/2609.34839
- OpenWhistle-CNN window dataset: https://huggingface.co/datasets/dolphinteam/OpenWhistle-CNN
- DOLPHINFREE source/annotation rights still require explicit confirmation before admission: https://zenodo.org/records/14637675

No cetacean data has been downloaded or scored by the changes described here. Current code and tests remain uncompiled/unexecuted in the authoring environment. No current Symthaea detection or comprehension performance is claimed.

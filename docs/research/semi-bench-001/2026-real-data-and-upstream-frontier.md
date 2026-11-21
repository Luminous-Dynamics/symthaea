# SEMI-BENCH-001 — real-data audit and upstream wafer manufacturing frontier

**Review date:** 2026-11-21  
**New workstreams:** [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113), [SEMI-SUBSTRATE-001 #13114](https://github.com/Luminous-Dynamics/symthaea/issues/13114)

This review records two important corrections: public wafer-map benchmarks can be dominated by synthetic balancing, and semiconductor wafer manufacturing begins before device-fab process modules.

## 1. SECOM: an external baseline that is useful, but not yet independent temporal evidence

An openly documented SECOM prototype is available at:
https://github.com/Be1an001/semiconductor-pass-fail-prediction-python

Its own README/model card characterize it as a portfolio screening prototype, not a production process-control system. It documents:
- UCI SECOM, 1,567 rows, 590 loaded sensor columns, 1,463 pass and 104 fail examples;
- a two-stage stratified random 60/20/20 split at \`random_state=42\`;
- missingness filtering and imputation fit on training data only;
- validation-only F2 threshold selection;
- final threshold 0.110;
- held-out split: 11 TP, 56 FP, 10 FN, 237 TN;
- recall 0.5238, precision 0.1642, balanced accuracy 0.6663, PR-AUC 0.2192 and 21.34% review rate.

The external split implementation uses \`sklearn.model_selection.train_test_split\` twice, stratified on the binary label. It is not a forward-time evaluation even though source timestamps are parsed and explored. The model card explicitly identifies lack of time-based validation, small positive support and non-causal feature importance as limitations.

**How we should use it:** a public random-split reproduction anchor only. Freeze the exact three UCI file digests, train/validation/test index digests, preprocessing fit state, model configuration, thresholds and expected outputs. A separately predeclared chronological evaluation is a different track. If the chronological windows contain too few fail examples, label the evaluation underpowered; do not quietly return to random splitting and call it time generalization.

The 2026 Park & Baek supplementary SECOM study reported PR-AUC 0.2207 under its own 60/20/20 setup, but with only 21 failures in each validation/test partition and zero held-out recall at its validation-selected high-precision operating point. These two public reports are close in PR-AUC but have different models/threshold protocols; they should not be interpreted as a matched replication until exact split identities and preprocessing match.

## 2. MixedWM38: separate public data, GAN augmentation and unseen-combination evidence

### Original dataset facts

The upstream MixedWM38 README states that:
- about 38,000 wafer maps are divided into 1 normal, 8 single-defect and 29 mixed-defect patterns;
- the raw map uses 0 for blank/outside, 1 for passing die and 2 for failing die;
- labels are 8-dimensional one-hot/multi-hot vectors for the eight base defect patterns;
- GAN-generated wafers were added to compensate for the highly uneven observed pattern counts.

Source: https://github.com/Junliangwangdhu/WaferMap

A third-party audit in ALEX8642/wafer-mixed reports 38,015 maps, 52×52, 38 combinations, 8 basic labels, 214 undocumented pixel-value-3 observations across 105 maps, and a label-order mapping established by visual inspection. It reports Near-full single support of 149 and Random single support of 866, with neither component present in mixed combinations:
- https://github.com/ALEX8642/wafer-mixed
- https://github.com/ALEX8642/wafer-mixed/blob/main/docs/DATA.md

That project reports test macro-F1 0.9846 and exact match 0.9696 under a stratified-by-combination split. It openly cautions that MixedWM38 is largely GAN-synthesized, near-uniform by design, and saturates quickly; it says its absolute metrics measure pipeline correctness more than real-world difficulty:
https://github.com/ALEX8642/wafer-mixed

Its 70/10/20 sample split is stratified by full 38-type combination, not by lot, tool, site or generator lineage. Its local data is downloaded separately; the repository's MIT code license does not automatically establish the data license or a per-example source provenance record.

### 2026 compositional zero-shot reference

Yang, Jang and Kim, *Classifying mixed-defect from single-defect training in imbalanced wafer maps via diffusion and attention*, Expert Systems with Applications 299, article 130127, published 1 March 2026:
- DOI: https://doi.org/10.1016/j.eswa.2025.130127
- Publisher page: https://www.sciencedirect.com/science/article/pii/S095741742503742X

The paper frames mixed-pattern recognition as generalized zero-shot learning: train on class-imbalanced single-defect maps, and recognize seen singles plus unseen compositions. It introduces class-mixing diffusion (CMDM) and prompt-wise attention score similarity (PASS). It reports a 74.3% harmonic-mean accuracy in its most imbalanced setting and a 3.92 percentage-point gain over the strongest compared diffusion classifier. These are paper-reported results, not reproduced Symthaea results.

This is a more meaningful challenge than a random split across all 38 combinations, because the test specifically measures whether component structure transfers to unseen combinations.

### A derived VQA dataset is not an independent physical corpus

A 2026 Hugging Face dataset, MixedWM38-VQA, contains 38,015 images derived from MixedWM38 plus multi-question and stepwise reasoning annotations:
https://huggingface.co/datasets/wafervqaanon/MixedWM38-VQA

Its dataset card states that supervision is synthetic or semi-synthetic/benchmark-structured and acknowledges that the visual style is cleaner than many real manufacturing environments. It is useful as a derived reasoning/annotation benchmark, but it shares the underlying image lineage and must not be presented as an independent source of physical wafers. The declared MIT dataset-card license does not by itself establish rights in the original MixedWM38 images or every derived annotation.

### Correct benchmark hierarchy

1. Standard interpolation split: reproduces existing work; lowest claim ceiling.
2. Leave-combination-out: measures compositional transfer to unseen mixtures.
3. Leave-component-out/open-set: measures safe rejection when a defect primitive is unseen.
4. Synthetic-to-real: requires per-sample provenance and an independently identified real-origin test set.
5. Cross-dataset: WM-811K and MixedWM38 can be compared only with a documented partial label mapping and no invented labels.
6. Active-learning: compare label efficiency under equal query budgets and noisy annotations.

Do not report MixedWM38 random-split macro-F1 as production generalization. Do not treat a synthetic map as a real wafer observation. Do not treat a component-pattern label as proof of a particular upstream process root cause.

## 3. Upstream substrate manufacturing: expand the material-to-wafer boundary

The January 2026 review *AI-powered semiconductor wafer fabrication: A manufacturing paradigm shift* (International Journal of Machine Tools and Manufacture 214, 104345) explicitly covers crystal growth, ingot slicing and wafer machining:
- DOI: https://doi.org/10.1016/j.ijmachtools.2025.104345
- https://www.sciencedirect.com/science/article/pii/S0890695525001002

It describes the conversion from high-purity crystal to shaped ingot, sliced substrates and precision-machined raw wafers. It reviews AI for growth modelling/quality, ingot-slicing quality and parameter selection, and grinding/lapping/polishing for surface integrity, dimensional accuracy, machining efficiency and tool wear. It also emphasizes the data challenge: more simulation than experimental data in crystal-growth studies, while slicing and wafer-machining research uses more experimental data.

### New paradigm

Measure the full material-to-qualified-wafer chain, not only an isolated process module.

Separate:
- material lot and impurity provenance;
- crystal growth regime and geometry;
- ingot shaping and slicing/kerf loss;
- wafer geometry and crystal quality;
- grinding/lapping/polishing;
- surface/particle/contamination metrology;
- downstream compatibility to a pinned device/foundry profile.

Do not collapse roughness, bow/warp, TTV, resistivity, oxygen/carbon, dislocations, particles, contamination, subsurface damage and flatness into one "quality" scalar.

The key objectives become accepted wafer yield, material utilization, kerf loss, cycle time, tool wear, water/chemical/energy intensity per accepted wafer, measurement uncertainty and downstream compatibility.

This creates a realistic path to add semiconductor capability without claiming we can already produce electronic-grade silicon or operate a production crystal-growth facility. Symtropy's current Glass/Silicon MG-Si scope remains separate; the new issue specifically guards against treating metallurgical-grade silicon modelling as electronic-grade substrate qualification.

## 4. Competitive changes

| Previous headline | Stronger benchmark question |
|---|---|
| Random-split wafer-map macro-F1 | Can the model compose never-seen defect mixtures? |
| MixedWM38 aggregate accuracy | How does it perform on samples with verified real provenance? |
| AI explanation of a defect | Which visual facts are directly supported, and which process hypotheses are uncertain? |
| Model accuracy at a fixed label budget | How many verified labels are needed to achieve a target risk/coverage? |
| Silicon process simulation | Does the full material-to-qualified-wafer chain meet multiple downstream constraints? |
| Best roughness or throughput | What is the Pareto frontier of wafer quality, accepted throughput, material loss and utilities? |

## 5. Workstream integration

- [SEMI-WBM-COMP-001 #13113](https://github.com/Luminous-Dynamics/symthaea/issues/13113): compositional zero-shot and synthetic-to-real wafer-map evaluation.
- [SEMI-SUBSTRATE-001 #13114](https://github.com/Luminous-Dynamics/symthaea/issues/13114): upstream substrate material-to-wafer manufacturing benchmark.
- [SEMI-BENCH-002 #13067](https://github.com/Luminous-Dynamics/symthaea/issues/13067): SECOM baseline reproduction and temporal split audit.
- [SEMI-WM-HDC-001 #13068](https://github.com/Luminous-Dynamics/symthaea/issues/13068): multiscale relational HDC/CfC challenger.

All are research workstreams. The public baseline projects and paper-reported results are comparison material, not reproduced Luminous results.


## SECOM chronological split now frozen

The audited mirror's per-file SHA-256 values and canonical two-file snapshot-manifest digest are now recorded in `secom-source-integrity-audit-v0.1.json`. A row-level chronological split manifest is frozen at `splits/uci-secom-mirror-0d27427a/chronological-60-20-20-v0.1.json`.

It contains 1,567 assignments and uses a declared source-local naive clock domain. Its chronological 60/20/20 support is 76/11/17 failures. The manifest was constructed from the public mirror commit, not from a verified byte-identical official UCI download. It has not yet been used to train or evaluate a model.


## Missingness is a time-varying observation process

The pinned SECOM mirror has 41,951 missing sensor values (4.5375%). Every row contains at least one missing value. The missing-cell fraction changes from 5.14% in the chronological training block to 3.42% in validation and 3.84% in test; monthly it is 5.33% in August, 4.25% in September and 3.77% in October. Twenty-eight sensors are more than 50% missing; 52 have no missing values.

In this mirror, fail rows have a slightly lower average number of missing features than pass rows (25.46 vs 26.86). Do not interpret this as a physical mechanism. It demonstrates that missingness itself can be a time-varying observation process, and that missingness-only baselines and mask-shuffle controls are needed to detect shortcuts.

Follow-up: [SEMI-SECOM-MISS-001 #13115](https://github.com/Luminous-Dynamics/symthaea/issues/13115).

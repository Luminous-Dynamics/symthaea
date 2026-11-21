# Synthetic PCI and Project Hypnos — Evidence Audit V1

**Status:** Static source/documentation audit; no biological dataset was reprocessed here.

## 1. Digital PCI benchmark in symthaea-psych-bench

File: crates/domains/symthaea-psych-bench/src/benchmarks/qualia_confidence/perturbational_complexity.rs

This is a four-state synthetic recurrent network. Each dimension is self-connected; cross-dimensional coupling is introduced by a broadcast-feedback path. The benchmark directly toggles the Global Workspace broadcasting flag between conditions. The comments in the source explicitly state that without broadcasting dimensions decay independently, while with broadcasting a hand-designed differential ring feedback drives cross-dimensional dynamics.

The test suite then checks that the broadcast-enabled condition has higher Lempel–Ziv complexity and propagation depth, including a fixed ratio threshold. These tests verify that the code's intended broadcast mechanism changes the synthetic trajectory. They do not independently test whether a biological system is conscious, and they do not reproduce clinical PCI.

Required interpretation: **synthetic perturbation-propagation architecture diagnostic (legacy digital-PCI label)**. Keep it in the synthetic lane. The condition is broadcast-enabled versus broadcast-disabled, not conscious versus unconscious.

## 2. PCI-labelled HDC example

File: examples/benchmark_pci_validation.rs

The example creates synthetic HDC networks from manually assigned coupling, recurrence and noise parameters. It perturbs a vector and computes entropy over a distribution of pairwise vector similarities. This is a useful synthetic network-response statistic, but it is not the TMS-EEG PCI algorithm.

The example has no human EEG, no TMS instrument, no measured evoked-response time series and no clinical reference labels. It also includes hand-authored expected PCI ranges for named clinical states. Those ranges must not be used to validate a synthetic similarity-entropy value or to support a clinical claim. A correlation across eight hand-parameterized synthetic regimes is an internal model relationship, not external clinical construct validity.

Required interpretation: synthetic HDC perturbation/entropy demonstration only. Do not compare its output to a clinical PCI cutoff.

## 3. Project Hypnos

Historical status document: docs/status/PROJECT_HYPNOS_VALIDATION.md  
Example: examples/clinical_validation.rs

The historical note reports results for two Sleep-EDF participants (SC4001 and SC4002), with a small two-channel EEG feature pipeline and empirically tuned thresholds. Its target labels are sleep stages. The reported N3 / wake / sleep numbers are repository-reported values and have not been independently rerun as part of this audit.

Sleep-stage classification is not consciousness detection. N3, REM and lighter sleep stages do not provide a direct binary label for subjective experience. In particular, REM reports can include vivid experience; lack of an awakening report cannot be recoded as no experience. The dataset's sleep-stage annotations can validate a sleep-stage classifier, not a phenomenal-consciousness detector.

The historical note itself reports weak wake and N2 performance, only two participants and zero N1 detection. It does not provide a prospective multisite cohort, participant-level confidence intervals, or clinical validation.

Required interpretation: exploratory two-subject sleep-stage classification prototype. Remove claims of a validated consciousness biomarker, universal sleep-state ordering or readiness for anesthesia/coma diagnostics.

## 4. PCI threshold boundary

The commonly cited PCI threshold around 0.31 belongs to a specified TMS–EEG method and validation context. It is not a universal threshold for sleep, anesthesia, every patient population, synthetic network complexity, HDC similarity entropy or internal Symthaea Φ.

Clinical TMS–EEG requires a physical perturbation, recording hardware, artifact handling, a specified PCI implementation and appropriate clinical validation. A software analogue can test an architectural hypothesis; it cannot be described as clinical PCI merely because both involve perturbation and complexity.

## 5. Corrective actions

1. Preserve both synthetic experiments, but label them synthetic and mechanism-specific.
2. Keep their tests as software regression tests; do not use them as external consciousness evidence.
3. Correct Project Hypnos wording to sleep-stage classification and retain historical numbers with explicit provenance caveats.
4. Build external-data evaluation around subject-grouped splits, prespecified outcomes, negative controls and independent baselines.
5. Treat clinical deployment as out of scope until prospective, multicentre validation and appropriate ethics/regulatory review are complete.

## Sources

- Casali et al. (2013), PCI: DOI 10.1126/scitranslmed.3006294.
- Ziemann et al. (2026), clinical utility and translational status of TMS–EEG: DOI 10.1016/j.clinph.2025.2111487.
- COGITATE adversarial collaboration (2025): DOI 10.1038/s41586-025-08888-1.

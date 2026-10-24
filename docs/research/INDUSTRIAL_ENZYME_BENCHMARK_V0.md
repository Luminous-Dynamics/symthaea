# Industrial Enzyme Benchmark V0 — Evidence-Bounded Research Contract

**Status:** proposal only; no dataset has been downloaded, frozen, or evaluated by this document.  
**Parent research program:** [BIO-MILLENNIUM-000 #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271).  
**Related work:** [BIO-CEM literature benchmark #5753](https://github.com/Luminous-Dynamics/symthaea/issues/5753); [Mycelix source-bound biology research commons #3327](https://github.com/Luminous-Dynamics/mycelix/issues/3327); [Symtropy deterministic engineering digital-twin boundary #1467](https://github.com/Luminous-Dynamics/symtropy/issues/1467); [Mycelix open health-aid knowledge and dossier provenance #5027](https://github.com/Luminous-Dynamics/mycelix/issues/5027).
**Paradigm-shift roadmap:** [ENZYME_KINETICS_PARADIGM_SHIFT_ROADMAP_V0.md](ENZYME_KINETICS_PARADIGM_SHIFT_ROADMAP_V0.md) — evidence graphs, context-conditioned kinetic claims, transportability tests, uncertainty-aware abstention, and value-of-information planning.  
**Evidence graph contract:** [ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json](ENZYME_EVIDENCE_GRAPH_CONTRACT_V0.json) — node/edge, artifact, rights-reference, and replay invariants. The new [V0 verifier specification](ENZYME_EVIDENCE_GRAPH_VERIFIER_V0.md) documents the implemented structural/local-byte subset and its explicit limits; full replay and scientific-policy enforcement remain future work.

## 1. Question and non-claim

**Question:** Can a provenance-preserving Symthaea-assisted workflow improve the selection or prediction of enzyme-function records over transparent conventional baselines on a frozen, held-out public dataset?

Start with one narrow offline task: estimate a single declared kinetic or activity endpoint **under reported assay conditions**. Choose the endpoint only after a source-data audit. Do not combine \(K_m\), \(k_{cat}\), (k_{cat}/K_m), pH optimum, temperature optimum, conversion, and yield into one target. These endpoints have different meanings and units.

This document does **not** claim that Symthaea can fold proteins, establish enzyme mechanisms, design safe organisms, optimize a live bioreactor, or improve experimental yield. A sequence representation, similarity score, literature hypothesis, model prediction, and experimentally established property are distinct evidence classes.

## 2. Why this first step

Industrial enzymes are a useful starting point because the task can remain offline and low-risk while still measuring whether the stack handles real biological heterogeneity. For v0, narrow the target to **one reported enzyme turnover number (kcat), normalized to s^-1**, rather than blending kcat, Km, kcat/Km, optimum temperature, activity and yield. Every row must preserve the originally reported value/unit as well as the normalized value and the exact conversion rule/factor.

Enzyme outcomes depend on identity and sequence, organism/source, assay method, pH, temperature, substrate, units, and replication. Missing conditions and conflicting reports are part of the evidence—not permission to silently manufacture a complete record.

Use existing scientific resources as sources and specialist tools as replaceable adapters. Do not build a second generic scientific-reasoning stack or new canonical biological type unless an ownership audit proves a lossless representation gap; this follows the compose-first constraint in #6271.

## 3. Candidate sources and admission gate

Sources are candidates, **not yet an admitted corpus**. Separate *pipeline reproduction* from *generalization benchmarking*: a small reproducible dataset can prove the parser/fitting workflow can reproduce published analysis, but cannot by itself show that Symthaea predicts unseen enzymes.

- **Recommended first pilot — DaRUS-3867 / SLAC EnzymeML workflow:** https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3867 (DOI 10.18419/DARUS-3867, version 1.0, Max Häußler, 2024). The official DaRUS record lists CC BY 4.0 and describes raw enzyme-assay absorption data, derived calibration JSON, EnzymeML `.omex` files containing experimental conditions/measurements/estimated parameters/models, and the analysis notebook. It covers small-laccase-catalyzed ABTS oxidation at 25, 30, 35.5, 40 and 45°C and lists 17 files (ZIP 3.8 MB). The linked [EnzymeML/slac_modeling](https://github.com/EnzymeML/slac_modeling) repository exposes `data/raw_data/`, `data/calibrations/`, `data/enzymeml/` and the notebook; however, the repository itself has no LICENSE file, so use the canonical DaRUS DOI record as the data-rights anchor and pin the actual downloaded file hashes rather than treating a GitHub mirror as its own license grant. This is the best current candidate for a *kinetic-analysis reproduction* lane because it exposes both raw readings and intermediate/calculated artifacts. **No files have been downloaded**. Before admission, inspect the file-level metadata/methods, confirm the DataUse terms, cite the dataset and related publication, pin the exact dataset version, chosen filenames, hashes, EnzymeML schema version and fitting configuration. This pilot is not assumed to provide enough independent enzyme/family coverage for the primary generalization benchmark.

- **Secondary reproduction source — DaRUS-3337:** https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3337 (DOI 10.18419/DARUS-3337, version 1.0, Max Häußler, 2023). The record describes a larger 134-file (5.5 MB) JupyterBook thesis dataset with raw measurements and a reproducible analysis. It is a backup/extension if it adds distinct, well-documented workflow cases beyond the SLAC pilot. It does not replace a held-out enzyme-function corpus.

- **Primary-source candidate — BRENDA:** https://www.brenda-enzymes.info/ — curated enzyme functional data. The official site identifies release **2026.1 (4 March 2026)** and CC BY 4.0; the bulk download requires active license acceptance. The official license page also warns that access to digital sequence information may entail benefit-sharing obligations under the UN Convention on Biological Diversity's multilateral mechanism, including potential monetary contributions by commercial users. Therefore **CC BY 4.0 alone is not recorded as a completed rights review**. The manifest must record actual release, accepted terms, attribution, access date and SHA-256, and a human-reviewed rights status that includes any applicable sequence-data benefit-sharing assessment. We have not downloaded or accepted the license for this workstream.
- **STRENDA Guidelines / STRENDA DB:** https://www.beilstein-institut.de/en/projects/strenda/guidelines/ and https://www.beilstein-strenda-db.org/strenda/ — use the guidelines to inform completeness checks for enzyme identity, assay context, units, precision and replication. A reported row does not become complete or reliable merely because it appears in a database.
- **SABIO-RK:** https://sabiork.h-its.org/webservice.jsp — its official documentation describes REST-based programmatic access to kinetic data. Treat it as a secondary source only after its relevant access, use and redistribution terms are independently cleared. Cross-source duplicates must not be treated as independent experiments.
- **Candidate explicitly not admitted — Zenodo forward/backward kcat dataset:** https://zenodo.org/records/21241849 (DOI 10.5281/zenodo.21241849). The page describes a compact dataset compiled from BRENDA and SABIO-RK and says a newer version exists, but the page's license field is blank. It is also prefiltered to enzyme/reaction pairs with both directional kcat values, so selection bias is built into the cohort. It is **not** admitted until the author/source terms, version lineage and redistribution rights are established; convenient downloadability is not a license.
- **Format/tooling reuse — EnzymeML-RS:** https://github.com/EnzymeML/enzymeml-rs. The current public README identifies the project as MIT-licensed and documents EnzymeML document parsing, ODE simulation, parameter fitting and validation, but explicitly says it is under development and not ready for production use. Audit the exact git revision, transitive dependencies, schema/version coverage, parameter-fitting algorithm, unit behavior and tests before adoption. Prefer a narrow pinned adapter and independent fixture-based qualification; do not allow the library's existence or passing self-tests to count as scientific validation. Avoid creating a duplicate parser/model ontology if the library can be safely wrapped.
- **BioModels / SBML:** https://www.ebi.ac.uk/biomodels/ and https://sbml.org/software/ — consider later for independently sourced, published pathway or kinetic model validation. These model files are not a substitute for a consistent enzyme-assay table and should not be mixed into the initial supervised target corpus without a separate track.

The BRENDA statistics page reports 94,073 turnover-number occurrences for release 2026.1. This is a source-database occurrence count, **not** the number of unique, rights-cleared, fully conditioned benchmark observations. The eligible corpus size is unknown until source admission and deduplication are complete.

### Corpus admission checklist

The corpus is frozen only when all items below are recorded:

- Exact source/release identifiers, access dates, URLs, licenses, attribution text, and content hashes.
- Explicit endpoint, target units, transform, inclusion/exclusion rules, assay-condition fields, and label provenance.
- Stable enzyme/protein and organism identifiers where available; source-paper or dataset identity for each observation.
- Duplicate and near-duplicate checks covering repeated database entries, alternate units, repeated publications, the same assay copied through multiple sources, and sequence-family overlap.
- Counts of retained, excluded, ambiguous, missing-context, and conflicting records, with exclusion reasons.
- A record-level manifest that can be independently rechecked without distributing restricted source material.

If licensing, endpoint semantics, units, or observation lineage cannot be resolved, stop at **data-admission-unresolved**. Do not substitute an invented or silently scraped dataset.

### Phase 0: reproduction pilot versus predictive benchmark

Use DaRUS-3867 / SLAC as a separate **kinetic-analysis reproduction gate** once its file-level methods and terms have been reviewed. The phase should: (1) enumerate only the versioned files needed from the canonical Dataverse record; (2) preserve dataset citation, CC BY 4.0 attribution, source file checksums, EnzymeML schema/version, Python requirements of the reference workflow and preprocessing; (3) reproduce the published analysis from raw absorbance data through calibration and concentration conversion to parameter fitting, using the same declared model and assumptions; (4) compare derived parameters, units and residuals against the published EnzymeML/notebook outputs within preregistered tolerances; and (5) record every unreproducible step, ambiguity, optimizer failure or residual discrepancy rather than silently discarding it. Record the reference repository commit separately from the immutable dataset version. Report this only as workflow reproducibility—not biological generalization, novel discovery or independent laboratory replication. Do not submit a guestbook or personal information on the user's behalf.

A successful pilot can qualify the read/parse/unit/fit/serialization pipeline before we spend effort admitting a larger kcat corpus. It does not replace that corpus or the study/family-held-out predictive evaluation.

#### Five-temperature calibration fit and checksum reproduction (not full kinetic-model reproduction)

I checked the five temperature-specific calibration JSON files in the public mirror at a single immutable repository commit [737c57b54f3cf9c9914c53f3d8b7f7790c76b5c5](https://github.com/EnzymeML/slac_modeling/commit/737c57b54f3cf9c9914c53f3d8b7f7790c76b5c5). These were retrieved through the repository API and inspected in memory; none of the files was saved into the workspace.

For each file, I computed MD5 over the returned 7-bit ASCII payload and compared it with the file-level MD5 published by the canonical DaRUS record. All five matched. I then independently refit the declared zero-intercept linear equation, `signal = a × concentration`, including only samples within the embedded calibration range (0–150 µmol/L inclusive). Each file contains 30 samples, with 27 selected and the three 200 µmol/L samples excluded by the stated rule.

| Calibration file | Bytes | Canonical MD5 ✓ | SHA-256 of pinned mirror payload | Refit slope | Refit R² | Refit RMSD |
|---|---:|---|---|---:|---:|---:|
| ABTS_standard_25.0C.json | 5,054 | `c8561fe8784f956d1123bbcd52a65121` | `39ccca8d0288e71f9ea686fdd19b96f04e6761e658a599fee108b897a3c20187` | 0.018329862325174823 | 0.9965435394942497 | 0.052531170831419005 |
| ABTS_standard_30.0C.json | 5,052 | `b9b51ddcd76bf1ee8dcb5c08f258b7a2` | `de7da34ff636397e5ca130b4dbe28916d537290699847a63ce1acdcf33fff4f9` | 0.017314605186480186 | 0.9997826256275483 | 0.012440262679447662 |
| ABTS_standard_35.5C.json | 5,050 | `38e7b2be4eb40a383eec57efa319e897` | `229a013d707ae09dc200588a2b1cd5a2fcfb7831c2f1461663b237e14226a8c5` | 0.017977302593240093 | 0.9961949291160336 | 0.05377164085489366 |
| ABTS_standard_40.0C.json | 5,054 | `7ffcd2249695adb526afe06eee28066b` | `5d2d5f91a5d8cc02508f6591f592687ee711c746143edbb0acdcc83155b112df` | 0.019069431818181815 | 0.9997064382973191 | 0.01589866612109839 |
| ABTS_standard_45.0C.json | 5,052 | `e1960a2f3cfeabb71e291ae4967b6e91` | `b8fcf7ee94bd64b85acd886adc29a2653979ebdfefd4909b00b305f1b3663c6d` | 0.018059865603146853 | 0.9977614756102724 | 0.04146734721174711 |

Across all five files, the independently fitted slopes differ from the embedded reported slopes by less than 4.8×10⁻¹³. The R² values match at the reported precision; RMSD differs only by floating-point rounding (no more than about 1.8×10⁻¹⁷ in these results). These are calibration statistics from the published JSON metadata/data, not new experimental findings. They demonstrate why the declared concentration window must be honored: the 25 °C file alone shows that including the three 200 µmol/L observations changes its zero-intercept slope from 0.018329862325174823 to 0.017570697786690438 and raises RMSD from 0.052531170831419005 to 0.09230840071343391.

The official DaRUS record identifies version 1.0, reports CC BY 4.0, documents raw absorption data plus calibration JSON and EnzymeML `.omex` files across 25–45 °C, and publishes the file-level MD5 values shown above. The mirror README states its data are equivalent to the DaRUS deposit. SHA-256 was computed over each pinned mirror payload in memory using a known-answer-checked implementation, and the SHA-256, Git blob SHA, byte length, and canonical DaRUS MD5 are now frozen as metadata in the five-file reference manifest below. MD5 is used only to compare the payload with published file metadata; it is not collision-resistant and is not the security anchor. The Rust command has not yet been run locally against a saved copy in this checkout. The dedicated workflow is configured to fetch the pinned payloads into its temporary runner workspace and execute synthetic tests plus the real-payload replay; only a completed successful run for the exact current head counts as CI verification. After obtaining the canonical/versioned files through the authorized download path, place them in the ignored local staging directory shown below and confirm each local SHA-256 matches the frozen manifest before invoking the command. This work does **not** yet reproduce absorbance extraction, calibration generation from raw readings, the EnzymeML kinetic fit, all five temperature-dependent kinetic models, or any enzyme-function prediction.

The five-file machine-readable reference index is [SLAC_CALIBRATION_REFERENCE_MANIFEST.json](SLAC_CALIBRATION_REFERENCE_MANIFEST.json). It records, per file, the pinned mirror Git commit/blob, byte length, source SHA-256, canonical DaRUS MD5, declared fitting range, sample counts and reference fit statistics. It is explicitly labelled `reference_only_not_admitted` and contains metadata only—no calibration JSON payloads.

After acquiring the five source files under the official DaRUS terms and placing them in the ignored local staging directory, run the batch replay:

```sh
cargo run -p xtask -- enzyme-calibration-suite-check \
  --manifest docs/research/SLAC_CALIBRATION_REFERENCE_MANIFEST.json
```

The suite verifies every file's byte length, SHA-256, Git blob SHA and canonical MD5, rejects paths escaping the workspace, checks source identity and the exact five-file temperature mapping, then cross-checks each JSON's pH, recorded temperature/unit, wavelength and signal type before recomputing its zero-intercept fit and comparing range, sample counts, slope/R²/RMSD against frozen references. The dedicated GitHub Actions workflow downloads the five files from the immutable mirror commit into its temporary job workspace, prints DaRUS attribution, runs the synthetic Rust tests, then runs this suite command against the real payloads; the source files are never committed by this workflow. Synthetic negative controls cover metadata mismatches even when recalculated file hashes are consistent. The manifest status remains explicitly non-admitted; a passing calibration check is not approval for the predictive benchmark.

For a one-file diagnostic, the replay command is:

```sh
sha256sum research-data/enzyme-benchmark/slac-calibration/ABTS_standard_25.0C.json
# Expected SHA-256 for the pinned mirror payload:
# 39ccca8d0288e71f9ea686fdd19b96f04e6761e658a599fee108b897a3c20187
cargo run -p xtask -- enzyme-calibration-check \
  --input research-data/enzyme-benchmark/slac-calibration/ABTS_standard_25.0C.json \
  --sha256 39ccca8d0288e71f9ea686fdd19b96f04e6761e658a599fee108b897a3c20187
```

The command fails closed on byte-hash mismatch, unsupported equation, duplicate sample IDs, mixed concentration units, invalid declared range, or disagreement between the recomputed fit and embedded slope/R²/RMSD.

#### Kinetic-pipeline checkpoints from the related 30-condition thesis dataset

The related [DaRUS-3337 thesis dataset](https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2FDARUS-3337) and its [SLAC characterization chapter](https://haeussma.github.io/masterthesis/scenarios/SLAC_kinetic_characterization.html) describe a *broader* experiment than the five-temperature calibration subset above. Keep these source versions and scopes separate until their specific OMEX/data-file lineage is verified.

The chapter's reported design comprises 30 kinetic assays across pH 3.0–5.5 and 25–45 °C, initial ABTS concentrations of 0–200 µM, no-enzyme controls, two measurement wavelengths (340 nm for ABTS and 420 nm for oxidized product), and a 15-minute time course with more than 100,000 individual absorbance readings. The reproducible transform is therefore not just a calibration-line fit: map photometer exports to the expected measurement/species IDs, subtract the control-derived blank, apply the *matching pH-and-temperature* standards at the correct wavelength, and do not extrapolate beyond calibration bounds.

The thesis reports a 3.2 absorbance cutoff for calibration points. Its usable upper concentration boundary is pH-dependent: 150 µM at pH 3.0, 3.5 and 5.5, and 175 µM at the other pH values. The five JSON files in the DaRUS-3867 smoke check are the pH-3 calibration files across five temperatures; they do **not** verify the other pH-dependent curves in the 30-condition thesis dataset. The thesis' calibration sample list, calibration-range cutoff, and observations dropped because absorbance was out of range should be reproduced as declared rules and recorded as counts/reasons, not reconstructed by guesswork.

A second, independent step in the broad workflow estimates a per-assay product signal factor \(k\), because no direct product calibration was available. The source models mass balance as \(S_t + kP_t - S_0\) (using its control measurements) and chooses \(k\) to minimize time-course slopes. That factor is a fitted mass-balance parameter—not a unit conversion and not permission to rescale product until the kinetic model looks good. Preserve the source's bounds, residual definition, fitting algorithm and per-assay values as a separate artifact. The chapter flags problematic mass balance at pH 4.0/45 °C and pH 4.5/35 °C, where intended substrate conditions were not achieved and estimated parameters were high uncertainty.

For model reproduction, the chapter ultimately selects **irreversible Michaelis–Menten with time-dependent enzyme inactivation**. It excluded product-inhibition models after negative calculated product concentrations, and substrate-inhibition models when AIC worsened and parameter standard deviations exceeded 100%. For the pH 3.0 / 25 °C example, the chapter reports the following reference results:

| Model | AIC | \(K_m\) (µM) | \(k_{cat}\) (s⁻¹) | Inactivation \(k_i\) (s⁻¹) |
|---|---:|---:|---:|---:|
| Irreversible Michaelis–Menten, no time-dependent inactivation | −162 | 35.473 (5.52% relative uncertainty) | 0.308 (2.07%) | — |
| Irreversible Michaelis–Menten with time-dependent inactivation | −476 | 35.137 (3.00%) | 0.460 (1.95%) | 0.001 (4.47%) |

These are values reported in the *related thesis*, not a claim that the current Rust implementation has reproduced them or that this table alone validates the DaRUS-3867 OMEX model. The chapter reports that omitting inactivation underestimates \(k_{cat}\) by about 33%. That provides a strong future regression test: reproduce the source-selected model and uncertainty treatment, and retain the no-inactivation fit as a prespecified ablation rather than silently using whichever model looks best. It also reports excluding unreliable rows (including pH 4.0/45 °C, pH 4.5/35 °C, and pH 5.5/35 °C) after diagnosing boundary hits, mass-balance outliers, or unestimable uncertainty. Those exclusions must be explicit, frozen, and counted.

**Recommended sequence:** qualify the five-file pH-3 calibration replay first; next reproduce each available temperature-specific OMEX kinetic fit for that same source version; only then expand to the thesis' broader 30-condition pipeline with per-pH calibrations, control corrections, product-factor estimation, model comparison, and explicit failed-fit accounting. Do not blend the two dataset versions or use the reported \(k_{cat}\) table as validation until the matching raw measurements, OMEX model and exact analysis revision have been pinned.

### Executable admission gate

A fail-closed Rust checker is wired into the existing workspace task runner:

```sh
cargo run -p xtask -- enzyme-bench-check --manifest docs/research/ENZYME_BENCHMARK_MANIFEST.json
```

Use [ENZYME_BENCHMARK_MANIFEST_TEMPLATE.json](ENZYME_BENCHMARK_MANIFEST_TEMPLATE.json) and [ENZYME_BENCHMARK_SPLITS_TEMPLATE.json](ENZYME_BENCHMARK_SPLITS_TEMPLATE.json) as non-admitted templates. They are deliberately marked/incomplete so they **must fail** until a human has cleared rights, the real files are frozen and hashed, and the split is populated. Local, rights-gated data artifacts belong in the ignored `research-data/enzyme-benchmark/` directory and must never be committed as part of this PR.

The checker currently enforces: frozen-state and source-rights metadata; manifest path containment under the canonical workspace root as well as workspace-contained corpus/split/preregistration artifacts (including symlink escape checks); a single source ID/release per corpus with every row checked against the reviewed manifest (multi-source pooling is intentionally blocked until rights can be reviewed per source); SHA-256 for corpus, split and preregistration; a single kcat endpoint and normalized s^-1 units; finite positive numeric values and an explicit, arithmetically consistent unit-conversion rule/factor per row; explicit reported/not-reported/not-applicable status for pH, temperature and substrate; unique observation IDs and unambiguous unique CSV column names; exact record count; and row assignment against a frozen split.

On a successful admission run, the checker also emits `ENZYME_METADATA_COVERAGE` rows for twenty optional, STRENDA-inspired descriptors: enzyme purity and form, enzyme concentration, buffer system and concentration, assay pressure, temperature at which pH was measured, substrate concentration range, kinetic equation, total replicate count, independent-experiment count, replicate structure, measurement precision, detection limit, reaction equation, analysis method, raw-measurement-data locator and its SHA-256, initial-rate method, and controls description. Each optional descriptor can carry `{"status":"reported","value":...}`, `{"status":"not_reported"}`, or `{"status":"not_applicable","reason":"..."}`. An absent key is counted separately as `unrecorded`; malformed/unknown statuses and internally inconsistent status/value pairs are counted as `invalid`. The checker enforces an internal accounting invariant: for every optional descriptor, `reported + not_reported + not_applicable + unrecorded + invalid` must equal the total observation count, so the audit cannot silently lose rows in its summary. For `reported`, the checker validates basic value shape: concentrations and pressure require finite positive values with units; experiment/replicate counts are positive integers; precision requires a named measure, finite non-negative value and units; detection limits require a finite non-negative value and units; analysis methods require software, version and method; raw-data locators must be HTTPS URLs without embedded credentials. Enzyme purity, buffer, replicate structure and reaction-equation fields must be nonempty strings. These are structural checks, not validation of a purity method, reaction balance, measurement precision, or scientific truth. These counters are descriptive rather than admission gates: old records remain inspectable without fabricated metadata, and omissions are visible. They are **not** a biological-quality score. This distinction follows the [official STRENDA guidance](https://www.beilstein-strenda-db.org/strenda/public/guidelines.xhtml), which says it aims to improve reporting and reproducibility but does not establish a metric for judging data quality. The current profile is an intentionally small mapping, not a claim of full STRENDA compliance. The endpoint-specific policy is captured in [KCAT_ADMISSION_PROFILE_V0.json](KCAT_ADMISSION_PROFILE_V0.json). It defines four evidence levels: corpus-level integrity admission, context-qualified, primary-comparison candidate, and kinetic-reproduction candidate. The Rust checker emits nested aggregate counts and an `ENZYME_KCAT_ROW_TIER` record for each observation, with a JSON-escaped observation ID and its highest tier (`review_only`, `context_qualified`, `primary_comparison_candidate`, or `kinetic_reproduction_candidate`). Context qualification requires actually reported pH, temperature and substrate plus enzyme form, substrate concentration range, reaction/kinetic equations and buffer/pressure disposition. The primary-comparison tier additionally requires purity, molar enzyme concentration, replicate structure/count, independent experiment count, precision and analysis-method metadata. Kinetic-reproduction candidates additionally require the raw-data URL and SHA-256 string, initial-rate method and controls description. **The current checker validates only the raw-data digest's shape; it does not fetch the URL or verify that digest against the remote payload.** A separate authorized acquisition step must bind URL, exact bytes and SHA-256 before this is evidence of raw-data integrity. These are candidate labels based on declared metadata, not scientific validation or proof of comparability. The JSON is a policy companion, not the runtime source; a unit test checks that its field inventory covers all audited descriptors, and its hard-coded Rust implementation still requires domain review.

#### Reporting-profile gap: structural coverage is not STRENDA conformance

The published [STRENDA Level 1A checklist](https://www.beilstein-institut.de/download/2279/strenda_list_level1a_1_8_20211202.pdf) specifies much richer assay context, including unambiguous enzyme identity, a balanced reaction equation, temperature and pH, pressure/atmosphere where applicable, buffer identity and concentration, salts and other assay components, substrate identity and concentration range, and enzyme/protein concentration. [Level 1B](https://www.beilstein-institut.de/download/2281/strenda_list_level1b_1_8_20211202.pdf) additionally emphasizes the number and nature of independent experiments and measurement precision. The current checker requires only pH, temperature and substrate to be explicitly status-coded and audits a small subset of other descriptors as optional fields. Thus it is **not a Level 1A/1B conformance checker**, even when all twelve optional descriptors are present and correctly shaped.

For a future predictive corpus, the next schema revision should map the relevant STRENDA fields explicitly and then define which are strict admission prerequisites for the chosen endpoint versus fields whose absence is recorded as a limitation. Conditional fields (for example, pressure when atmospheric conditions are not assumed) must encode applicability and rationale, not become universal required values. Never infer missing assay conditions from defaults, a study title, or adjacent records. This profile review should precede any claim that the corpus is sufficiently comparable for a pooled model evaluation. The DaRUS reproduction lane is deliberately separate: this checker still does not parse EnzymeML/OMEX or reproduce the enzyme's kinetic model. A separate `enzyme-calibration-check` command verifies a local calibration JSON's caller-supplied SHA-256 and recomputes the declared zero-intercept linear fit within the source's explicit concentration range; it is a calibration-fit check, not raw-absorbance extraction or kinetic-model validation. Three split modes are supported: `study_holdout`, `sequence_family_holdout`, and the more restrictive `joint_study_and_sequence_family_holdout`. Single-axis tracks report cross-axis overlap (for example, sequence families shared across a study-held-out split) rather than silently claiming that other dimension generalizes; joint mode rejects such overlap. This is an **admission/integrity gate**, not the metric scorer and not evidence that Symthaea has predictive value.

## 4. Observation unit and missingness

The initial modeling row is one *reported measurement under a declared assay context*, not one protein and not one paper. A paper may describe multiple enzymes, substrates, conditions, and replicates.

Each row should reference, where available:

- enzyme/protein identifier and sequence accession or sequence hash;
- EC number and organism/taxonomy/strain;
- substrate/reaction identifier;
- reported endpoint and original value/unit;
- normalized value/unit plus the explicit conversion rule;
- pH, temperature, pressure when applicable, buffer/matrix, and assay-method reference;
- replication count, uncertainty/precision, detection limits, and qualifiers;
- source publication, database record, source release, and extraction/provenance locator;
- explicit missingness/conflict/applicability flags.

Missing is not zero; unreported is not normal; “not detected” is not an exact zero; a literature optimum is not the value at every assay condition. Do not silently impute absent pH, temperature, substrate, or organism context. Preserve contradictory reports as separate observations unless an independently documented harmonization rule justifies a transformation.

## 5. Preregistered evaluation

Freeze the corpus manifest, target, preprocessing, features, candidate pool, split manifest, random seeds, baseline configurations, metric definitions, and budget **before looking at held-out labels**.

### Baselines

At minimum compare:

1. a constant/global-median predictor;
2. an endpoint- and context-aware group baseline, with small-sample fallback defined in advance;
3. a simple descriptor or nearest-neighbour baseline whose inputs and distance rule are fully specified;
4. a published or repository-provided specialist baseline only if the precise task, license, and required inputs are compatible;
5. the Symthaea-assisted method, with all selector/feature/abstention rules frozen before evaluation.

The comparison is allowed to conclude that the simple baseline wins. Symthaea's HDC encoding or cross-domain analogy is an experimental feature, not a presumption of added predictive value.

### Splits and leakage controls

- Group the primary held-out evaluation by source study/publication and sequence-family homology where data supports defensible grouping.
- Keep a random split only as a diagnostic; never present it as evidence of cross-family or cross-study generalization.
- Do not let alternate units, duplicate database records, homologous sequence variants, replicate measurements, or re-hosted copies cross the same held-out boundary unnoticed.
- Report each split's exact identities, exclusions, and overlap-check output.

### Metrics and uncertainty

Choose metrics compatible with the frozen endpoint (for example, MAE/RMSE on a predeclared transformed scale, rank correlation, interval coverage, and calibration where uncertainty is claimed). Report performance by assay-context completeness, major endpoint strata, source study, and in-domain/out-of-domain status. Include confidence intervals and the exact denominator; do not report only the best subgroup or successful predictions.

### Negative controls

- Shuffled-label control.
- Source/study-ID-only nuisance baseline.
- Context-only baseline when context can plausibly explain most of the label variation.
- Duplicate/leakage stress test.
- Deterministic replay of the frozen scoring manifest.

If shuffled labels or source identity appear to produce convincing predictive performance, stop and fix the leakage or scoring defect before making any model claim.

## 6. Role boundaries across the three systems

### Symthaea

Owns the hypothesis and candidate-selection experiment, the frozen model/config manifest, uncertainty and abstention, and the interpretation of benchmark results. The existing molecular-biology encoder is a representational starting point, not a qualified enzyme-function predictor. Keep production implementation Rust-first; any external model is a version-pinned adapter with an explicit dependency and licensing review.

### Mycelix

May hold or reference provenance, source rights, contribution attribution, review receipts, and access/consent policies using existing ownership seams. It must not turn peer agreement, a trust score, a credential, or a governance decision into biological validation. Its relevant data/governance capabilities remain maturity-gated; this document does not claim this workflow is already integrated or deployed.

### Symtropy

May model a separately bounded *process-level* scenario only after appropriate process equations and parameters are sourced and independently qualified. Its current rigid-body physics core must not be presented as a biochemical reaction solver, molecular dynamics engine, protein-folding model, or validated bioreactor simulator. Do not put consciousness/Phi coupling into a scientific reference model unless a separately defined and independently justified hypothesis explicitly requires it.

### Laboratory / physical work

No wet-lab execution is part of this milestone. Any later prospective experiment requires human review, a separate biosafety and risk assessment, a preregistered protocol, suitable oversight and containment, and exact measurement/provenance capture. This benchmark must not generate or authorize pathogen engineering, toxin enhancement, or uncontrolled environmental release.

## 7. Independent verifier and staged exits

The new `xtask enzyme-bench-check` command verifies source admission and dataset/split/preregistration artifact integrity. It does not yet score predictions, prove full source deduplication, or independently compute benchmark metrics; those are subsequent milestones. The eventual scorer must read frozen manifests independently of the candidate generator and recompute metrics, denominators, uncertainty intervals and negative controls. Fail closed on unknown units, missing hashes, unresolved licensing, incomplete provenance or changed data/model heads.

1. **Kinetic workflow reproduction (separate lane):** after source/access terms are reviewed, independently reproduce the selected DaRUS analysis from the frozen raw files; check parameter estimates, residuals and units against predeclared tolerances.
2. **Predictive-data admission:** source rights, release, records, target and units are resolved; content hashes match.
3. **Baseline reproduction:** an independent run reproduces the declared predictive baseline within preregistered numerical tolerances.
4. **Integrity:** duplicate/source/sequence leakage checks and all negative controls behave as expected.
5. **Value test:** Symthaea's comparison against all declared baselines is measured on untouched held-out groups with uncertainty. No advantage is assumed.
6. **Prospective follow-up (separate milestone):** only after the offline evaluation is sound, preregister a human-reviewed, low-risk experiment. Predictions and simulations must remain distinct from new physical measurements.

Terminal statuses must include success, null result, failure, excluded, out-of-domain, and blocked/unresolved cases. A failed run or inconclusive result is retained; it is not silently removed from the denominator.

## 8. Deliverables for the first implementation tranche

- Versioned data-admission and benchmark manifest schema (reuse current evidence/identity owners wherever possible).
- Rust-first deterministic `xtask enzyme-bench-check` admission gate; unit tests use generated **synthetic-only** observations and do not assert biological performance.
- Rights-cleared DaRUS file admission and reproducible kinetic-analysis pilot, reported separately from generalization.
- Rights-cleared source admission and an independently frozen predictive corpus/split/preregistration with exact hashes.
- EnzymeML-RS API/license/schema audit before any parser work.
- Independent scorer/manifest checker plus hostile tests for shuffled labels, duplicate rows, unknown units, source leakage and missing provenance.
- A frozen report containing baseline results, confidence intervals, exclusions, runtime/resource cost and explicit limitations.

**Not in this tranche:** broad sequence generation, wet-lab automation, clinical recommendations, a new canonical biology ontology, a new biochemical simulator, or claims of improved biotech outcomes before measured evaluation.

## 9. References

- DaRUS-3867 official dataset record and CC BY 4.0 attribution metadata: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3867
- EnzymeML-based workflow repository and reference notebook: https://github.com/EnzymeML/slac_modeling
- DaRUS-3337 related reproducible EnzymeML kinetics workflow: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-3337
- BRENDA, download and license/release information: https://www.brenda-enzymes.de/download.php
- BRENDA, database scope and curation: https://www.brenda-enzymes.org/introduction.php
- STRENDA Guidelines: https://www.beilstein-institut.de/en/projects/strenda/guidelines/
- STRENDA DB: https://www.beilstein-strenda-db.org/strenda/
- Wilkinson et al., “The FAIR Guiding Principles for scientific data management and stewardship,” *Scientific Data* (2016): https://doi.org/10.1038/sdata.2016.18
- BioModels and SBML model infrastructure: https://www.ebi.ac.uk/training/online/courses/biomodels-quick-tour/searching-and-getting-data-from-biomodels/ and https://sbml.org/software/

# Industrial Enzyme Benchmark V0 — Evidence-Bounded Research Contract

**Status:** proposal only; no dataset has been downloaded, frozen, or evaluated by this document.  
**Parent research program:** [BIO-MILLENNIUM-000 #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271).  
**Related work:** [BIO-CEM literature benchmark #5753](https://github.com/Luminous-Dynamics/symthaea/issues/5753); [Mycelix source-bound biology research commons #3327](https://github.com/Luminous-Dynamics/mycelix/issues/3327); [Symtropy deterministic engineering digital-twin boundary #1467](https://github.com/Luminous-Dynamics/symtropy/issues/1467); [Mycelix open health-aid knowledge and dossier provenance #5027](https://github.com/Luminous-Dynamics/mycelix/issues/5027).

## 1. Question and non-claim

**Question:** Can a provenance-preserving Symthaea-assisted workflow improve the selection or prediction of enzyme-function records over transparent conventional baselines on a frozen, held-out public dataset?

Start with one narrow offline task: estimate a single declared kinetic or activity endpoint **under reported assay conditions**. Choose the endpoint only after a source-data audit. Do not combine (K_m), (k_{cat}), (k_{cat}/K_m), pH optimum, temperature optimum, conversion, and yield into one target. These endpoints have different meanings and units.

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

### Executable admission gate

A fail-closed Rust checker is wired into the existing workspace task runner:

```sh
cargo run -p xtask -- enzyme-bench-check --manifest docs/research/ENZYME_BENCHMARK_MANIFEST.json
```

Use [ENZYME_BENCHMARK_MANIFEST_TEMPLATE.json](ENZYME_BENCHMARK_MANIFEST_TEMPLATE.json) and [ENZYME_BENCHMARK_SPLITS_TEMPLATE.json](ENZYME_BENCHMARK_SPLITS_TEMPLATE.json) as non-admitted templates. They are deliberately marked/incomplete so they **must fail** until a human has cleared rights, the real files are frozen and hashed, and the split is populated. Local, rights-gated data artifacts belong in the ignored `research-data/enzyme-benchmark/` directory and must never be committed as part of this PR.

The checker currently enforces: frozen-state and source-rights metadata; a single source ID/release per corpus with every row checked against the reviewed manifest (multi-source pooling is intentionally blocked until rights can be reviewed per source); SHA-256 for corpus, split and preregistration; a single kcat endpoint and normalized s^-1 units; finite positive numeric values and an explicit, arithmetically consistent unit-conversion rule/factor per row; explicit reported/not-reported/not-applicable status for pH, temperature and substrate; unique observation IDs and unambiguous unique CSV column names; exact record count; and row assignment against a frozen split. The DaRUS reproduction lane is deliberately separate: this checker does not yet parse EnzymeML or verify the thesis' kinetic parameter fitting, so those remain separate implementation work rather than implied support. Three split modes are supported: `study_holdout`, `sequence_family_holdout`, and the more restrictive `joint_study_and_sequence_family_holdout`. Single-axis tracks report cross-axis overlap (for example, sequence families shared across a study-held-out split) rather than silently claiming that other dimension generalizes; joint mode rejects such overlap. This is an **admission/integrity gate**, not the metric scorer and not evidence that Symthaea has predictive value.

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

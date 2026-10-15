# Industrial Enzyme Benchmark V0 — Evidence-Bounded Research Contract

**Status:** proposal only; no dataset has been downloaded, frozen, or evaluated by this document.  
**Parent research program:** [BIO-MILLENNIUM-000 #6271](https://github.com/Luminous-Dynamics/symthaea/issues/6271).  
**Related work:** [BIO-CEM literature benchmark #5753](https://github.com/Luminous-Dynamics/symthaea/issues/5753); [Symtropy Living World Cell #252](https://github.com/Luminous-Dynamics/symtropy/issues/252); [Mycelix open health-aid knowledge and dossier provenance #5027](https://github.com/Luminous-Dynamics/mycelix/issues/5027).

## 1. Question and non-claim

**Question:** Can a provenance-preserving Symthaea-assisted workflow improve the selection or prediction of enzyme-function records over transparent conventional baselines on a frozen, held-out public dataset?

Start with one narrow offline task: estimate a single declared kinetic or activity endpoint **under reported assay conditions**. Choose the endpoint only after a source-data audit. Do not combine (K_m), (k_{cat}), (k_{cat}/K_m), pH optimum, temperature optimum, conversion, and yield into one target. These endpoints have different meanings and units.

This document does **not** claim that Symthaea can fold proteins, establish enzyme mechanisms, design safe organisms, optimize a live bioreactor, or improve experimental yield. A sequence representation, similarity score, literature hypothesis, model prediction, and experimentally established property are distinct evidence classes.

## 2. Why this first step

Industrial enzymes are a useful starting point because the task can remain offline and low-risk while still measuring whether the stack handles real biological heterogeneity. Enzyme outcomes depend on identity and sequence, organism/source, assay method, pH, temperature, substrate, units, and replication. Missing conditions and conflicting reports are part of the evidence—not permission to silently manufacture a complete record.

Use existing scientific resources as sources and specialist tools as replaceable adapters. Do not build a second generic scientific-reasoning stack or new canonical biological type unless an ownership audit proves a lossless representation gap; this follows the compose-first constraint in #6271.

## 3. Candidate sources and admission gate

Sources are candidates, **not yet an admitted corpus**:

- **BRENDA:** https://www.brenda-enzymes.de/ — curated enzyme functional data. Its download page currently documents release **2026.1 (4 March 2026)** and a **CC BY 4.0** license; the download flow requires explicit acceptance. At dataset freeze, record the release actually downloaded, access date, exact license/terms, required attribution, download URL, and SHA-256. Do not mirror records before the applicable terms are accepted and checked.
- **STRENDA Guidelines / STRENDA DB:** https://www.beilstein-institut.de/en/projects/strenda/guidelines/ and https://www.beilstein-strenda-db.org/strenda/ — use the guidelines to inform completeness checks for enzyme identity, assay context, units, precision, and replication. A reported data row does not become complete or reliable merely because it appears in a database.
- **BioModels / SBML:** https://www.ebi.ac.uk/biomodels/ and https://sbml.org/software/ — consider later for independently sourced, published pathway or kinetic model validation. These model files are not a substitute for a consistent enzyme-assay table and should not be mixed into the initial supervised target corpus without a separate track.

### Corpus admission checklist

The corpus is frozen only when all items below are recorded:

- Exact source/release identifiers, access dates, URLs, licenses, attribution text, and content hashes.
- Explicit endpoint, target units, transform, inclusion/exclusion rules, assay-condition fields, and label provenance.
- Stable enzyme/protein and organism identifiers where available; source-paper or dataset identity for each observation.
- Duplicate and near-duplicate checks covering repeated database entries, alternate units, repeated publications, the same assay copied through multiple sources, and sequence-family overlap.
- Counts of retained, excluded, ambiguous, missing-context, and conflicting records, with exclusion reasons.
- A record-level manifest that can be independently rechecked without distributing restricted source material.

If licensing, endpoint semantics, units, or observation lineage cannot be resolved, stop at **data-admission-unresolved**. Do not substitute an invented or silently scraped dataset.

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

The verifier must read the frozen manifests independently of the candidate generator and recompute row counts, split disjointness, metric values, denominators, and negative controls. It should fail closed on unknown units, missing hashes, unresolved licensing, incomplete provenance, or changed data/model heads.

1. **Admission:** source rights, release, records, target and units are resolved; content hashes match.
2. **Baseline reproduction:** an independent run reproduces a baseline within preregistered numerical tolerances.
3. **Integrity:** duplicate/source/sequence leakage checks and all negative controls behave as expected.
4. **Value test:** Symthaea's comparison against all declared baselines is measured on untouched held-out groups with uncertainty. No advantage is assumed.
5. **Prospective follow-up (separate milestone):** only after the offline evaluation is sound, preregister a human-reviewed, low-risk experiment. Predictions and simulations must remain distinct from new physical measurements.

Terminal statuses must include success, null result, failure, excluded, out-of-domain, and blocked/unresolved cases. A failed run or inconclusive result is retained; it is not silently removed from the denominator.

## 8. Deliverables for the first implementation tranche

- Versioned data-admission and benchmark manifest schema (reuse current evidence/identity owners wherever possible).
- Small rights-cleared fixture corpus with synthetic/known-answer rows for unit tests; clearly mark synthetic rows so they cannot be mistaken for biological measurements.
- Rust-first deterministic validator for schema, units, source identity, and split disjointness.
- Independent scorer/manifest checker plus hostile tests for shuffled labels, duplicates, unknown units, source leakage, and missing provenance.
- A frozen report containing baseline results, confidence intervals, exclusions, runtime/resource cost, and explicit limitations.

**Not in this tranche:** broad sequence generation, wet-lab automation, clinical recommendations, a new canonical biology ontology, a new biochemical simulator, or claims of improved biotech outcomes before measured evaluation.

## 9. References

- BRENDA, download and license/release information: https://www.brenda-enzymes.de/download.php
- BRENDA, database scope and curation: https://www.brenda-enzymes.org/introduction.php
- STRENDA Guidelines: https://www.beilstein-institut.de/en/projects/strenda/guidelines/
- STRENDA DB: https://www.beilstein-strenda-db.org/strenda/
- Wilkinson et al., “The FAIR Guiding Principles for scientific data management and stewardship,” *Scientific Data* (2016): https://doi.org/10.1038/sdata.2016.18
- BioModels and SBML model infrastructure: https://www.ebi.ac.uk/training/online/courses/biomodels-quick-tour/searching-and-getting-data-from-biomodels/ and https://sbml.org/software/

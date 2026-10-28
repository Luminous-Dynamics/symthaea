# Symthaea HIV Reservoir Discovery

**Last literature review:** 2026-10-29  
**Status:** Research protocol and dataset inventory only. No biological dataset is claimed as downloaded, reanalysed, or independently validated by this change.  
**Scope:** Computational analysis of public, de-identified HIV reservoir and latency research data. This is not a clinical tool and does not recommend treatment changes.

## Research thesis

Symthaea should be evaluated as a system for producing **testable, evidence-grounded explanations of HIV persistence**, not as a cure predictor. Its useful contribution would be a measurable improvement in held-out prediction, hypothesis prioritisation, or experiment selection over established statistical and bioinformatics methods.

The first question is deliberately narrow:

> In public research datasets, can Symthaea help explain or predict measured HIV reservoir phenotypes beyond clinical covariates and established transcriptomic baselines, and do the resulting hypotheses survive independent evidence?

A strong result may be a useful negative result: for example, demonstrating that a proposed representation adds no value on cross-sectional data, or that a candidate association does not generalise beyond the cohort in which it was found.

## Why start here?

The Dryad dataset accompanying Dwivedi et al. includes host RNA-seq and HIV reservoir measurements from 191 ART-suppressed people with HIV. The publication reports associations involving P3H3, NBL1 and residual HIV RNA, while also noting that intact-DNA findings were limited by many undetectable values. That makes it a realistic reproduction and evaluation target, not a blank slate for novelty claims.

Source: https://doi.org/10.5061/dryad.k3j9kd5dw  
Publication: https://doi.org/10.1371/journal.ppat.1011114

Additional public GEO studies cover latency reversal, transcriptional state and tissue-specific reservoir biology. They are related evidence sources, but their outcomes, assays and sample units differ. Do not concatenate them into one training table or describe them as interchangeable labels.

## Proposed discovery loop

1. **Register evidence.** Pin dataset accessions, publication versions, file names, checksums and data dictionaries before analysis. Record cohort, assay, biological unit, limitations and reuse restrictions.
2. **Reproduce prior findings.** Attempt a faithful reproduction of the published associations before adding Symthaea-specific methods. Preserve discrepancies and null findings.
3. **Run fair baselines.** Compare a clinical-covariate baseline, established transcriptomic/statistical methods and Symthaea on identical outer folds, samples and outcomes.
4. **Generate mechanism-level hypotheses.** Every hypothesis must link evidence, alternatives, a falsifiable prediction, a proposed discriminating test and an explicit stop condition.
5. **Validate out of cohort.** Use an independent cohort or study where the assay and endpoint are sufficiently comparable. When they are not, treat the study as mechanistic triangulation rather than quantitative replication.
6. **Choose the next experiment by discrimination.** Prefer a test that separates competing explanations, adjusted for cost, feasibility, risk and expected information gain, over one that merely confirms the favoured story.
7. **Preserve provenance.** Mycelix can eventually record dataset digests, code commit, configuration, claims, peer reviews and negative results. Symtropy should simulate only models whose assumptions and calibration have been documented.

## Paradigm shifts to test

- **From association lists to competing causal explanations.** Require at least two plausible explanations and one observable result that distinguishes them.
- **From one pooled dataset to a hierarchy of evidence.** Keep people, cell lines, individual cells, proviral clones and tissue samples as distinct units. Replicates from one donor must never leak across training and test folds.
- **From random splits to study-held-out validation.** Random patient splits are an initial internal test, not proof of generalisation. External study shifts are a central benchmark.
- **From best-score hunting to calibrated decisions.** Report uncertainty, calibration, subgroup performance, model stability, multiple-testing correction and failures.
- **From generic model complexity to matched inductive bias.** A cross-sectional dataset cannot establish a temporal-model advantage. Evaluate CfC/LTC only when repeated or time-resolved observations support temporal prediction; evaluate HDC/representation components with ablations.
- **From predicted therapeutic promise to experiment utility.** Computational associations cannot establish mechanism, safety, cure or clinical benefit.

## Milestones

### M0 — Data and protocol qualification
- Review and pin the files and data dictionary.
- Compute SHA-256 for each downloaded file and retain the source URL, timestamp and licence/terms.
- Audit participant-ID uniqueness, joins, missingness, measurement limits and assay-specific sample units.
- Freeze outcomes, transformations, folds and metrics before modelling.

### M1 — Faithful reproduction
- Reproduce the source paper's reported associations to the extent possible from the released files.
- Report differences in cohort inclusion, covariates, preprocessing, software and effect estimates.
- Treat failure to reproduce as a result requiring investigation, not a reason to tune until it disappears.

### M2 — Fair model comparison
- Compare clinical-only, regularised transcriptomic, established statistical/ML and Symthaea variants.
- Use nested patient-level validation, with all preprocessing and feature selection inside each training fold.
- Publish the full score distribution, confidence intervals, calibration and negative controls.

### M3 — Independent validation and hypothesis ledger
- Validate only endpoint-compatible predictions externally.
- Keep non-comparable datasets as separate mechanistic tasks.
- Require independent evidence and, for causal claims, validation by qualified biomedical researchers before describing a mechanism as established.

## Evidence and safety boundaries

This work must not claim that Symthaea can cure HIV, discover an effective treatment, or establish a causal mechanism from correlation alone. No result in this protocol constitutes evidence for changing ART or any other clinical care. Treatment-interruption decisions and biological experiments belong with qualified clinical or research teams under appropriate oversight.

The source paper itself warns that further work is needed, including functional genomic studies and dedicated assays of the replication-competent reservoir. A prediction of total HIV DNA or residual RNA is not equivalent to detecting intact, replication-competent virus or predicting rebound.

## Files

- [Benchmark protocol](BENCHMARK_PROTOCOL.md) — outcomes, baselines, splits, success criteria and leakage controls.
- [Dataset register](datasets.json) — initial machine-readable source inventory; all entries remain unacquired until their local files are verified.
- [Hypothesis ledger schema](HYPOTHESIS_LEDGER.schema.json) — minimum structure for traceable, falsifiable claims.

## Primary scientific sources

- Dwivedi et al. (2023), *PLOS Pathogens*: https://doi.org/10.1371/journal.ppat.1011114
- Dryad data deposit: https://doi.org/10.5061/dryad.k3j9kd5dw
- Jiang et al., integrated single-cell multiomic analysis of HIV latency reversal, GEO GSE242997 and GSE210146: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE242997
- HIV reservoir transcriptomics using flow-FISH, GEO GSE260588: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE260588
- Brain myeloid reservoir multi-omics, GEO GSE221685 / GSE221686 / GSE221687: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221687
- HIV expression heterogeneity in clones carrying intact latent proviruses, GEO GSE310195: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE310195

# Quantifying Pharmacogenomic Evidence Gaps: An Exploratory Population-Level Equity Analysis

 
> **Research status:** This manuscript describes population-level, hypothesis-generating analysis only. The current Hardy–Weinberg/functional-bin approximation has material discrepancies against published phenotype estimates, and clinically relevant alleles/structural variants are missing. Do not use these outputs for individual genotype interpretation, medication selection, prescribing, dose adjustment, or clinical decision support. Numeric claims remain subject to independent data and methods re-audit.


**Tristan Stoltz**

Luminous Dynamics, Richardson, TX, USA

Correspondence: tristan.stoltz@evolvingresonantcocreationism.com

**Keywords**: pharmacogenomics, health equity, CPIC, ancestry, CYP2D6, CYP2C19, CYP2B6, CYP3A5, dosing guidelines, open-source, admixed populations

---

## Abstract

Pharmacogenomic guidelines can be harder to apply equitably when the underlying evidence and reference data represent populations unevenly. We describe an open-source, exploratory engine that calculates population-level phenotype-frequency approximations from curated allele frequencies under simplified Hardy–Weinberg assumptions and reports per-drug, per-gene, per-group disparity measures. The intended research use is to identify evidence gaps for further investigation, not to interpret an individual's genotype or recommend medication.

The current implementation aggregates alleles into broad functional categories and omits clinically material alleles and structural variants. Its own comparison against published phenotype estimates reveals substantial errors for some gene/population combinations, including severe CYP2D6 discrepancies. Consequently, the computed scores and estimated impacts are hypothesis-generating, not measurements of clinical harm or evidence of dosing validity. Population categories are not genotypes; blending population allele frequencies does not resolve individual diplotypes, phase, copy number, structural variation, or within-group diversity.

Selected allele-frequency comparisons against reference resources can check source consistency, but do not validate phenotype estimates, clinical utility, or patient outcomes. The next qualification steps are independent gene-specific reference annotations, comprehensive variant handling for each supported gene, explicit indeterminate results, external evaluation with prespecified metrics, and—only after those gates—study of any defined clinical use. The project is research-only.


---

## 1. Introduction

### 1.1 The Pharmacogenomics Promise

Pharmacogenomics offers the prospect of matching the right drug, at the right dose, to the right patient based on genetic variation in drug-metabolizing enzymes, transporters, and targets. The Clinical Pharmacogenetics Implementation Consortium (CPIC) has been instrumental in translating pharmacogenomic evidence into actionable clinical guidelines, now covering 34 genes and 164 drugs across 28 active guidelines with over 10,000 citations in the literature [1]. Clinical adoption is accelerating: an estimated 85% of PubMed pharmacogenomic implementation studies reference CPIC guidelines, and pre-emptive genotyping panels are increasingly deployed in major health systems across North America and Europe [2].

The core mechanism is straightforward. Cytochrome P450 (CYP) enzymes metabolize approximately 75% of all clinically used drugs [3]. Genetic polymorphisms in CYP genes produce four canonical metabolizer phenotypes---ultra-rapid (UM), normal (NM), intermediate (IM), and poor (PM)---each associated with distinct pharmacokinetic profiles. A poor metabolizer of codeine, for example, cannot convert it to its active metabolite morphine, rendering the drug ineffective; an ultra-rapid metabolizer converts codeine to morphine at dangerous rates, risking respiratory depression [4]. CPIC guidelines translate these phenotype-pharmacokinetic relationships into dosing recommendations: avoid codeine in ultra-rapid metabolizers, use alternative analgesics in poor metabolizers, and prescribe standard doses to normal metabolizers.

### 1.2 The Representation Crisis

The promise of pharmacogenomics is undermined by a fundamental representation problem. The populations in which guidelines are developed and validated do not reflect the populations to which they are applied. An analysis of the PharmGKB database found that over 63% of population-labeled individuals are of European descent [5]. The GWAS Catalog is even more skewed: 86.5% of participants are of European ancestry, with African representation at just 0.47% [5]. A 2025 scoping review identified 40 published pharmacogenomic analysis tools; every single one was developed in a high-income country, and none included systematic equity assessment [6].

This is not merely a theoretical concern. Allele frequencies for pharmacogenomically actionable variants differ dramatically across ancestry groups. CYP2C19*2, the primary loss-of-function allele for clopidogrel metabolism, occurs at a frequency of 29% in East Asian populations compared with 15% in European populations [7]. CYP3A5*3, the loss-of-function allele critical for tacrolimus dosing in transplant medicine, reaches 94% in European populations but only 18% in African populations [8,9]. CYP2B6*6, the slow-metabolizer allele governing efavirenz clearance in HIV treatment, occurs at 38% in African populations versus 26% in European populations [10,11]. When guidelines are calibrated to European allele frequency distributions, patients from populations with substantially different distributions receive recommendations that may be systematically suboptimal.

### 1.3 Documented Clinical Harm

The consequences of ancestry-biased pharmacogenomic guidelines are not hypothetical. A 2026 study analyzing data from the NIH All of Us Research Program found that 64% of CYP2D6 alleles classified as having pharmacological relevance for "drug response" were most frequent in East Asian populations, not the European populations in which guidelines were primarily validated [12]. An ancestry-linked regulatory haplotype influencing CYP2D6 expression was identified at 45% frequency in East Asian populations versus 2.5% in European populations [13]. CYP2D6 ultra-rapid metabolizer prevalence reaches 29% in Ethiopian populations---a figure frequently cited in isolation---though pan-African estimates are more conservatively 3-5% [14,15]. Even at the lower estimate, this represents a clinically significant population at elevated risk for codeine toxicity and ondansetron treatment failure.

In transplant medicine, approximately 82% of African-ancestry individuals are CYP3A5 expressors (carrying at least one *1 allele) compared with roughly 6% of European-ancestry individuals [9]. Standard tacrolimus dosing, calibrated to the European-predominant non-expressor phenotype, systematically underdoses African-ancestry transplant recipients, contributing to higher rates of graft rejection [9]. In HIV treatment, the standard 600 mg efavirenz dose produces plasma concentrations 3-4 times higher in CYP2B6 poor metabolizers---a phenotype approximately twice as common in African populations as in European populations---causing debilitating central nervous system toxicity including dizziness, insomnia, and psychosis [10,17].

### 1.4 Research Question and Intended Contribution

Published tools such as PharmCAT support genotype interpretation and guideline reporting for defined inputs; pharmacogenomic databases provide allele and guideline resources. The intended research question here is narrower: can a transparent, reproducible population-level audit help identify where allele-frequency sources and guideline reference populations may require further study?

We have not completed a systematic landscape review sufficient to claim that no other tool implements similar analyses or that this is the first open-source tool of its kind. Nor has the current implementation demonstrated clinical impact or superiority over established tools. The present contribution is an exploratory Rust implementation, a record of its assumptions and observed discrepancies, and a qualification plan. Its scores and model-derived group flags are hypothesis-generating only.

## 2. Methods

### 2.1 Allele Frequency Data Sources and Curation

We compiled allele frequency data for 16 star alleles across 6 pharmacogenes from the following primary sources: PharmGKB population-stratified allele frequency tables, the Genome Aggregation Database (gnomAD v4), NCBI Medical Genetics Summaries (NBK574601, AMP Tier 1 evidence), and the Trans-Omics for Precision Medicine (TOPMed) consortium. Gene-specific literature sources included Gaedigk et al. (2017) for CYP2D6 worldwide frequencies [14], Ionova et al. (2020) for CYP2C19 from over 2.2 million direct-to-consumer genetics research participants [7], a 2023 global distribution study for CYP2C9 [20], Desta et al. (2019) for CYP2B6 [10], and multiple meta-analyses for CYP3A5 [8,9].

Allele frequencies were organized across six ancestry groups: European, African, East Asian, South Asian, Native American, and Mixed. We acknowledge that these are pragmatic simplifications of continuous human genetic variation, not biological taxonomies. The "African" category, for example, encompasses enormous genetic diversity from West African to East African to African-American populations. The "Mixed" category is a placeholder for admixed populations, not a biologically meaningful grouping. We use these categories because they correspond to the resolution at which population-level allele frequency data are most commonly reported in the pharmacogenomic literature, and because CPIC guidelines themselves implicitly reference population categories at this scale.

Each allele frequency value was cross-checked against at least two independent published sources. Where sources disagreed, we documented the range. For example, CYP2D6*10 in East Asian populations is reported at 9-44% in AMP summaries, 46.2% in Korean-specific studies, and 43% in our reference value from NCBI NBK574601. One allele frequency was corrected during external validation: CYP3A4*22 in South Asian populations was revised from 0.03 to 0.01 based on gnomAD v4 SAS data (0.009, n=4,810 genomes), as the original value had no supporting literature and was a >3x overestimate (see Section 3.7). Table 1 presents the complete reference allele frequency dataset with source annotations.

**Table 1. Reference allele frequencies by ancestry group.**

| Gene | Allele | Function | European | African | East Asian | South Asian | Native Am. | Mixed | Primary Source |
|------|--------|----------|----------|---------|------------|-------------|------------|-------|----------------|
| CYP2D6 | *4 | No function | 0.19 | 0.04 | 0.01 | 0.12 | 0.08 | 0.10 | PharmGKB/NBK574601 |
| CYP2D6 | *10 | Decreased | 0.02 | 0.05 | 0.43 | 0.20 | 0.05 | 0.10 | PharmGKB/NBK574601 |
| CYP2D6 | *17 | Decreased | 0.00 | 0.19 | 0.00 | 0.02 | 0.01 | 0.05 | PharmGKB/NBK574601 |
| CYP2D6 | *41 | Decreased | 0.09 | 0.04 | 0.02 | 0.07 | 0.04 | 0.05 | PharmGKB/NBK574601 |
| CYP2D6 | *1xN | Increased | 0.02 | 0.05 | 0.01 | 0.04 | 0.02 | 0.03 | Gaedigk 2017 |
| CYP2C19 | *2 | No function | 0.15 | 0.18 | 0.29 | 0.32 | 0.12 | 0.18 | Ionova 2020 |
| CYP2C19 | *17 | Increased | 0.22 | 0.22 | 0.04 | 0.17 | 0.10 | 0.15 | Ionova 2020 |
| CYP2C9 | *2 | Decreased | 0.13 | 0.02 | 0.00 | 0.05 | 0.01 | 0.05 | Hu Genomics 2023 |
| CYP2C9 | *3 | No function | 0.07 | 0.01 | 0.04 | 0.11 | 0.02 | 0.04 | Hu Genomics 2023 |
| CYP3A4 | *22 | Decreased | 0.05 | 0.001 | 0.00 | 0.01 | 0.02 | 0.02 | gnomAD v4 |
| CYP2B6 | *6 | Decreased | 0.26 | 0.38 | 0.18 | 0.30 | 0.22 | 0.28 | Desta 2019/PMC9784060 |
| CYP2B6 | *4 | Increased | 0.04 | 0.01 | 0.07 | 0.05 | 0.03 | 0.04 | Desta 2019 |
| CYP2B6 | *18 | No function | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.01 | Desta 2019 |
| CYP3A5 | *3 | No function | 0.94 | 0.18 | 0.71 | 0.67 | 0.80 | 0.60 | PMC3738061/PMC5600063 |
| CYP3A5 | *6 | No function | 0.00 | 0.13 | 0.00 | 0.00 | 0.00 | 0.03 | PMC3738061 |

*Note: CYP3A4*22 South Asian frequency corrected from 0.03 to 0.01 following gnomAD v4 validation (see Section 3.7). Frequencies are point estimates. Ranges documented in Supplementary Table S1. Native American and Mixed values may reflect limited sample sizes.*

### 2.2 Hardy-Weinberg Phenotype Derivation

In the absence of population-level individual genotype data, we used Hardy-Weinberg equilibrium (HWE) to compute expected diplotype distributions from allele frequencies. For each gene and ancestry group, the procedure was as follows:

1. Sum the frequencies of all cataloged non-normal-function alleles; assign the remainder to *1 (normal function).
2. Classify each allele by functional category: NoFunction, DecreasedFunction, IncreasedFunction, or NormalFunction, according to CPIC's allele functionality table (PharmVar).
3. Aggregate allele frequencies by functional category: let *f*_nf denote the total no-function allele frequency, *f*_df the decreased-function frequency, *f*_if the increased-function frequency, and *f*_n the normal-function frequency (computed as the complement).
4. Compute diplotype-to-phenotype mapping:
   - Poor metabolizer (PM) = *f*_nf^2
   - Ultra-rapid metabolizer (UM) = *f*_if^2 + 2 * *f*_if * *f*_n
   - Intermediate metabolizer (IM) = 2 * *f*_nf * (*f*_n + *f*_df) + *f*_df^2 + 2 * *f*_df * *f*_n
   - Normal metabolizer (NM) = 1 - PM - UM - IM, clamped to [0, 1]

   The IM formula captures all diplotypes containing at least one decreased- or no-function allele that do not qualify as PM or UM: no-function/normal heterozygotes, no-function/decreased heterozygotes, decreased/decreased homozygotes, and decreased/normal heterozygotes. An earlier version of the formula omitted the 2 * *f*_df * *f*_n term (decreased/normal heterozygotes), which caused IM frequencies to be underestimated and NM frequencies to be correspondingly inflated. This was corrected prior to the analyses presented here.
5. Validate: confirm that phenotype fractions sum to 1.0 within a tolerance of 0.01.

We state the limitations of this approach explicitly. HWE assumes random mating, no natural selection, no migration, and no genetic drift within the defined population. Continental-scale ancestry groupings violate all of these assumptions to varying degrees. Admixed populations, which are increasingly common globally and especially in the Americas, cannot be accurately modeled with discrete ancestry bins; we address this limitation with explicit admixed modeling (Section 2.7). For CYP2D6 specifically, the complex gene locus with structural variants, copy number changes, and hybrid alleles makes HWE-based phenotype prediction less reliable than for simpler loci [14]. The validation comparison in Section 3.8 shows that the current outputs can materially disagree with published phenotype estimates. This implementation therefore does not provide validated phenotype-frequency estimates, and it is not known whether derived score rankings preserve the direction or magnitude of any real-world disparity.

### 2.3 Heuristic Population-Score Methodology

For each drug–gene–population tuple, the prototype calculates three **heuristic features** from its simplified phenotype-frequency outputs. None is a clinical probability, calibrated risk estimate, or measure of observed outcomes.

**Toxicity-oriented heuristic index.** For tuples classified in this implementation as prodrugs (codeine, clopidogrel, tamoxifen), the code applies: UM_fraction × 0.8 + PM_fraction × 0.3. For tuples classified as active drugs, it applies: PM_fraction × 0.8 + UM_fraction × 0.2. These fixed weights encode a simplified hypothesis about directions of concern; they do not estimate a patient's adverse-event probability.

**Efficacy-oriented heuristic index.** For the prototype's prodrug class, the code applies PM_fraction × 0.9. For its active-drug class, it applies UM_fraction × 0.7 + PM_fraction × 0.3. This is a designed score, not a validated estimate of therapeutic failure or treatment benefit.

**Reference-distribution similarity proxy.** The prototype calculates total variation distance (TVD) between each population-labeled phenotype vector and the European-labeled reference vector: TVD = (1/2) × sum(|p_i − p_i,ref|). It then stores 1 − TVD in a field named `guideline_applicability`. That field name is not evidence that the resulting quantity measures clinical guideline applicability. The choice of reference group is an analytical convention; applicability would need to be established separately for each guideline and defined context.

The **aggregate equity-gap score** is the mean of the across-group ranges of the two heuristic indices, clamped to [0, 1]. It measures variation in the current code's outputs, not inequity, clinical harm, guideline fit, or observed differences in outcomes.

The prototype labels a population group when its own threshold rules fire: reference-similarity proxy below 0.85, or toxicity-oriented index more than 0.15 above the minimum under an additional population-label rule. These labels are therefore called **model flags** throughout this manuscript; they do not establish that a population is clinically underserved or that a treatment should change.

The weights (0.8, 0.3, 0.9, 0.7, 0.2) are analyst-selected heuristics and have not been calibrated against observed outcomes or independently justified for every listed drug–gene pair. Sensitivity analysis perturbs included values and weights; it can describe internal score stability, but cannot validate the assumptions, resolve omitted variants, or prove that model-derived rankings are correct.

### 2.4 Exploratory Population Scenario Arithmetic

For descriptive examples only, the notebook-like report multiplies a broad population denominator by the current model's phenotype-frequency fraction. The product is an arithmetic scenario, not a count of patients at risk, people prescribed a drug, adverse events, or patients requiring a change in therapy.

The broad denominators used in the prototype are rounded estimates drawn from the cited US population categories. They should not be interpreted as genotype cohorts or exact genetic-ancestry counts. The underlying phenotype fractions inherit all limitations of the simplified Hardy–Weinberg/functional-bin model, including omitted alleles, copy-number and structural variation, and activity-score mismatches.

Accordingly, these products must not be described as epidemiological measurements, clinical impact estimates, or an upper bound on actionable patients without an independently validated phenotype method, defensible population denominators, prescription/indication data, uncertainty intervals, and an explicitly defined context of use.

### 2.5 CPIC Guideline Integration

We encoded 32 CPIC guideline recommendations across 11 drug-gene pairs in 7 clinical domains (Table 2). Each recommendation specifies: drug name, gene name, metabolizer phenotype, dosing action (StandardDose, ReduceDose, IncreaseDose, AvoidDrug, UseAlternative, or TherapeuticDrugMonitoring), dose adjustment factor, alternative drug where applicable, monitoring instructions, evidence level (Strong or Moderate), and source population (European for all encoded guidelines, reflecting the predominant validation population).

**Table 2. Selected guideline-reference records included in the research fixture.**

| Drug | Gene | Clinical Domain | Evidence | Phenotypes Covered | Example reference action in cited guideline |
|------|------|----------------|----------|-------------------|------------|
| Codeine | CYP2D6 | Pain | Strong | UM, NM, PM | Avoid (UM), Alternative (PM) |
| Clopidogrel | CYP2C19 | Cardiology | Strong | PM, NM | Alternative (PM) |
| Warfarin | CYP2C9 | Cardiology | Strong | IM, PM, NM | Reduce dose (IM/PM) |
| Sertraline | CYP2C19 | Psychiatry | Moderate | UM, NM, PM | Increase (UM), Reduce (PM) |
| Efavirenz | CYP2B6 | HIV | Strong | PM, IM, NM, UM | Reduce dose (PM/IM) |
| Tamoxifen | CYP2D6 | Oncology | Strong | PM, IM, NM, UM | Alternative (PM/IM) |
| Atomoxetine | CYP2D6 | Psychiatry | Strong | PM, NM, UM | Reduce 75% (PM) |
| Ondansetron | CYP2D6 | Supportive care | Strong | UM, NM | Alternative (UM) |
| Voriconazole | CYP2C19 | Infectious disease | Strong | PM, UM, NM | Reduce (PM), Alternative (UM) |
| Tacrolimus | CYP3A5 | Transplant | Strong | NM, IM, PM | Increase 2x (NM), 1.5x (IM) |
| Phenytoin | CYP2C9 | Epilepsy | Strong | PM, IM, NM | Reduce (PM/IM) |

The action descriptions above summarize selected published guideline records; they are not actions produced by the current research-projection API. The API returns only reference metadata and an exploratory aggregate flag, with no dose, medication choice, contraindication, or monitoring instruction. The population model is not qualified to interpret individual genotype/diplotype or to augment a patient's care plan.

### 2.6 Implementation

The engine is implemented in Rust (approximately 2,200 lines of code including admixed modeling and validation infrastructure), released under the AGPL-3.0 license. The implementation includes exploratory population allele-frequency entries, CPIC guideline reference records, Hardy–Weinberg phenotype approximations, equity-gap scoring, modeled impact estimates, and admixed-population calculations. These outputs have documented discrepancies and missing-variant limitations and are research-only. The public API includes `equity_analysis(drug, gene)` for aggregate population analysis, `research_guideline_projection(drug, gene, population_context, phenotype_context)` for non-actionable guideline-reference metadata, `population_report(gene)` for modeled distributions, `admixed_metabolizer_distribution(gene, population)` for exploratory admixture calculations, and `admixed_equity_analysis(drug, gene, populations)` for aggregate comparison. The projection type contains no dose, medication choice, contraindication, or treatment instruction; unsupported drug/gene/phenotype tuples return `Indeterminate`. The aggregate model is not qualified for patient-level phenotype interpretation, prescribing, or clinical decision support. Passing software tests does not establish pharmacogenomic accuracy or clinical validity.

### 2.7 Exploratory Admixed-Frequency Approximation

Four illustrative population mixtures were encoded from published average ancestry proportions to test how a simple weighted-frequency calculation behaves. These rounded mixtures are heterogeneous group averages and must not be used as individual ancestry estimates.

For each illustrative mixture, the prototype forms weighted averages of the reference allele frequencies and then applies its existing Hardy–Weinberg/functional-bin approximation. This is a computational scenario, not individual-level admixture inference. It does not model individual genotypes, local ancestry, linkage disequilibrium, phase, the Wahlund effect, or gene-specific diplotype translation. The resulting phenotype-frequency tables are exploratory outputs, not evidence that the true phenotype distribution of a named population has been estimated accurately.

## 3. Results

These components have distinct evidence status. Some selected allele-frequency inputs were compared with public resources and published cohort reports (Supplementary Table S4); those source-consistency checks do not validate completeness or representativeness of the entire allele set. The simplified Hardy–Weinberg/functional-bin phenotype approximation was compared with published estimates (Supplementary Table S5), and those comparisons expose material errors for some gene/population combinations; the model is not phenotype-validated. The heuristic risk weights have not been calibrated against observed clinical outcomes. Perturbing included numerical inputs measures internal sensitivity only and cannot establish that a disparity score or ranking is correct. All derived phenotype, equity, and population scenario outputs remain exploratory pending independent, gene-specific evaluation.

### 3.1 Metabolizer Phenotype Distributions Across Ancestry Groups

Table 3 lists outputs produced by the prototype's simplified HWE/functional-bin calculation for six genes and six broad population labels. The corrected arithmetic term and the CYP3A4*22 source correction address implementation/input consistency; they do not validate the phenotype model. The shown values are model outputs, not observed prevalence, independently validated phenotype estimates, or individual genotype calls. The repository's reference comparison documents material discrepancies and missing-variant limitations.

**Table 3. Simplified model phenotype-frequency outputs (%; not validated prevalence).**

| Gene | Ancestry | UM | NM | IM | PM |
|------|----------|----|----|----|----|
| **CYP2D6** | European | 2.8 | 62.4 | 31.2 | 3.6 |
| | African | 6.6 | 78.2 | 15.1 | 0.2 |
| | East Asian | 1.1 | 76.7 | 22.2 | 0.0 |
| | South Asian | 4.6 | 65.4 | 28.6 | 1.4 |
| | Native Am. | 3.2 | 80.7 | 15.4 | 0.6 |
| | Mixed | 4.1 | 73.5 | 21.4 | 1.0 |
| **CYP2C19** | European | 32.6 | 46.3 | 18.9 | 2.2 |
| | African | 31.2 | 43.9 | 21.6 | 3.2 |
| | East Asian | 5.5 | 47.2 | 38.9 | 8.4 |
| | South Asian | 20.2 | 36.9 | 32.6 | 10.2 |
| | Native Am. | 16.6 | 63.2 | 18.7 | 1.4 |
| | Mixed | 22.4 | 50.3 | 24.1 | 3.2 |
| **CYP2C9** | European | 0.0 | 64.0 | 31.1 | 4.9 |
| | African | 0.0 | 94.1 | 5.8 | 0.1 |
| | East Asian | 0.0 | 84.6 | 15.0 | 0.4 |
| | South Asian | 0.0 | 70.6 | 24.2 | 5.3 |
| | Native Am. | 0.0 | 90.3 | 9.6 | 0.1 |
| | Mixed | 0.0 | 78.4 | 19.2 | 2.4 |
| **CYP2B6** | European | 7.0 | 42.6 | 43.6 | 6.8 |
| | African | 1.7 | 25.2 | 55.4 | 17.6 |
| | East Asian | 10.2 | 48.4 | 38.2 | 3.2 |
| | South Asian | 7.5 | 32.5 | 51.0 | 9.0 |
| | Native Am. | 4.4 | 48.8 | 42.0 | 4.8 |
| | Mixed | 5.8 | 37.0 | 48.4 | 8.8 |
| **CYP3A5** | European | 0.0 | 0.4 | 11.3 | 88.4 |
| | African | 0.0 | 47.6 | 42.1 | 10.3 |
| | East Asian | 0.0 | 8.4 | 41.2 | 50.4 |
| | South Asian | 0.0 | 10.9 | 44.2 | 44.9 |
| | Native Am. | 0.0 | 4.0 | 32.0 | 64.0 |
| | Mixed | 0.0 | 16.0 | 48.0 | 36.0 |

*Values are outputs of this prototype's simplified model, rounded to one decimal place; they are not independent prevalence estimates. UM = ultra-rapid, NM = normal, IM = intermediate, PM = poor under the prototype's own functional-bin approximation, which does not reproduce complete gene-specific CPIC phenotype semantics. CYP2C9 and CYP3A5 show UM = 0% because no increased-function alleles are encoded in this dataset, not because an independent reference established a zero population frequency. CYP3A4 is omitted for brevity.*

These values show how the current implementation responds to its selected inputs. For **CYP2D6**, the model output includes the decreased/normal heterozygote term and reports 31.2% IM for the European-labeled inputs and 15.1% for the African-labeled inputs. The same model reports 3.6% PM for the European-labeled inputs versus a cited reference around 5.4%; the mismatch is consistent with known missing no-function alleles (*3, *5, *6) and other CYP2D6 coverage gaps. These numbers are a discrepancy diagnostic, not a validated group prevalence comparison.

For **CYP2C19**, the prototype outputs differ across labels (2.2% PM for European-labeled inputs, 8.4% for East Asian-labeled inputs, and 10.2% for South Asian-labeled inputs). The missing *3 allele and simplified phenotype mapping are known limitations. These figures do not establish the true prevalence in those populations, the comparative risk of clopidogrel non-response, or the calibration population for every clopidogrel guideline.

For **CYP3A5**, the simplified model gives substantially different outputs for the two input labels, largely reflecting its encoded *3 frequencies. Those outputs should not be described as observed expressor/non-expressor prevalence: the model's genotype/phenotype semantics, allele completeness, population mapping, and external comparability have not been qualified. The current numbers cannot estimate an individual transplant recipient's phenotype or tacrolimus dose requirement.

For **CYP2B6**, the prototype reports PM fractions of 17.6% for African-labeled inputs, 6.8% for European-labeled inputs, and 3.2% for East Asian-labeled inputs, based on the particular *6/*18 records encoded. These are simplified model outputs whose accuracy has not been independently established; they must not be converted into estimates of patients with CNS toxicity or the performance of standard-dose efavirenz.

### 3.2 Equity Gap Analysis

Table 4 reports classifications produced by the current model and its chosen heuristic weights/thresholds for all 11 drug-gene pairs. They are internal outputs, not validated clinical disparities or a rank of real-world patient harm.

**Table 4. Exploratory model flags across selected drug-gene pairs (not clinical risk estimates).**

| Drug | Gene | Domain | Model score tier | Population labels crossing prototype thresholds | Method/coverage warning |
|------|------|--------|----------------|-----------------------------------------------|-------------------------|
| Tacrolimus | CYP3A5 | Transplant | High | African, East Asian, South Asian, Mixed | Expressor interpretation and *3/*6 coverage require independent qualification |
| Efavirenz | CYP2B6 | HIV | High | African, South Asian | Functional-bin PM fraction not independently validated |
| Codeine | CYP2D6 | Pain | Moderate-High | African, East Asian, South Asian | CYP2D6 copy-number/structural variants and diplotype rules incomplete |
| Clopidogrel | CYP2C19 | Cardiology | Moderate-High | East Asian, South Asian, Mixed | *3 and complete star-allele/activity-score interpretation not represented |
| Voriconazole | CYP2C19 | Infectious disease | Moderate-High | East Asian, South Asian, Mixed | Same CYP2C19 coverage and phenotype-definition limits |
| Sertraline | CYP2C19 | Psychiatry | Moderate | East Asian, South Asian | Heuristic weights not outcome-calibrated |
| Tamoxifen | CYP2D6 | Oncology | Moderate | East Asian, South Asian | Model flag is not a treatment or dosing recommendation |
| Atomoxetine | CYP2D6 | Psychiatry | Moderate | East Asian, South Asian | CYP2D6 structural and allele-translation gaps |
| Ondansetron | CYP2D6 | Supportive care | Moderate | African | Prototype UM category does not establish treatment failure |
| Warfarin | CYP2C9 | Cardiology | Moderate | South Asian | Model output omits full multigene dosing context |
| Phenytoin | CYP2C9 | Epilepsy | Moderate | South Asian | Model output omits complete allele coverage and clinical context |

These tiers and flags depend on the present allele list, broad functional bins, heuristic weights, thresholds, and population references. Several phenotype estimates are known to disagree substantially with published comparisons (Section 3.8). The labels therefore mean only that the current code crossed its own threshold; they do not establish that a population is clinically underserved, that a dose is wrong, or that one drug–gene pair causes more harm than another.
### 3.3 Modeled Population Scenario Estimates (Not Clinical Impact)

The table below reproduces aggregate arithmetic from the current exploratory model and population denominators. These are **not estimates of patients who need a dose change, observed adverse events, clinical impact, or treatment benefit**. The phenotype fractions inherit the documented HWE/functional-bin errors and missing-variant limitations described in Sections 3.8 and 4.4. The population denominators are broad modeled categories, not a genotype dataset, and the model does not incorporate indication, actual prescribing, co-medications, organ function, individual genotype, or current clinical context.

| Drug | Gene | Population label | Model denominator (M) | Model phenotype fraction | Arithmetic product (M) |
|------|------|----------------|-----------------------|--------------------------|------------------------|
| Tacrolimus | CYP3A5 | African | 47.0 | 0.897 (expressors) | ~42.2 |
| Tacrolimus | CYP3A5 | European | 195.0 | 0.117 (expressors) | ~22.8 |
| Efavirenz | CYP2B6 | African | 47.0 | 0.193 (PM + UM) | ~9.1 |
| Efavirenz | CYP2B6 | European | 195.0 | 0.138 (PM + UM) | ~26.9 |
| Clopidogrel | CYP2C19 | East Asian | 7.0 | 0.134 (PM + UM) | ~0.9 |
| Clopidogrel | CYP2C19 | South Asian | 6.0 | 0.298 (PM + UM) | ~1.8 |
| Codeine | CYP2D6 | African | 47.0 | 0.068 (PM + UM) | ~3.2 |
| Codeine | CYP2D6 | East Asian | 7.0 | 0.011 (PM + UM) | ~0.1 |

The products in the final column are arithmetic outputs only: denominator multiplied by the modeled fraction. They must not be called “at risk,” “actionable patients,” or estimates of people who need a different treatment until both the phenotype model and the population denominator are independently qualified for a defined use.

### 3.4 Modelled Variation by Drug–Gene Pair

Under the selected allele inputs, broad functional bins, risk weights, and thresholds, the implementation yields different aggregate scores across drug–gene pairs. This is a description of **the model's behaviour**, not evidence that a clinical domain has a measured equity burden or that a population requires a specific intervention.

- **Transplant-related example:** CYP3A5 allele frequency differences motivate careful comparison with independently annotated diplotype distributions. This model's expressor/non-expressor fractions are not qualified to estimate clinical need or dose.
- **HIV-treatment example:** CYP2B6 is relevant to efavirenz pharmacogenomics, but the current allele coverage and simplified phenotype mapping do not establish population prevalence or treatment failure.
- **Pain-management example:** CYP2D6 is relevant to codeine metabolism; missing CYP2D6 alleles and structural/copy-number variation limit this implementation's aggregate phenotype approximation.
- **Cardiovascular example:** CYP2C19 and CYP2C9 are relevant to some antiplatelet and anticoagulation decisions. The model omits variants important to some populations, so the current cross-population outputs are incomplete.
- **Other drug classes:** scores for psychiatry, oncology, and epilepsy remain exploratory and should be treated as prompts for source and method review, not as a clinical ranking.

No model-produced population flag may be converted into an individual inference or prescribing recommendation.

### 3.5 Admixed Population Analysis

To address the limitation that continental ancestry bins poorly represent admixed populations, we extended the engine with a linear admixture model. Table 6 presents metabolizer distributions for the four admixed populations alongside key ancestral groups for CYP2D6, CYP2C19, and CYP3A5.

**Table 6. Metabolizer phenotype distributions for admixed populations (%, HWE-derived from blended allele frequencies).**

| Gene | Population | UM | NM | IM | PM |
|------|-----------|----|----|----|----|
| **CYP2D6** | African American | 5.8 | 75.2 | 18.3 | 0.7 |
| | US Latino/Hispanic | 3.5 | 76.4 | 19.1 | 1.0 |
| | Brazilian | 4.6 | 73.5 | 20.5 | 1.4 |
| | South African Coloured | 4.4 | 73.9 | 20.7 | 1.0 |
| | *African (ref)* | *6.6* | *78.2* | *15.1* | *0.2* |
| | *European (ref)* | *2.8* | *62.4* | *31.2* | *3.6* |
| **CYP2C19** | African American | 30.2 | 44.2 | 22.1 | 3.5 |
| | US Latino/Hispanic | 20.0 | 53.3 | 23.8 | 2.9 |
| | Brazilian | 27.3 | 47.0 | 22.5 | 3.2 |
| | South African Coloured | 21.9 | 44.6 | 27.6 | 5.9 |
| | *East Asian (ref)* | *5.5* | *47.2* | *38.9* | *8.4* |
| | *European (ref)* | *32.6* | *46.3* | *18.9* | *2.2* |
| **CYP3A5** | African American | 0.0 | 26.2 | 46.3 | 27.5 |
| | US Latino/Hispanic | 0.0 | 5.2 | 31.3 | 63.5 |
| | Brazilian | 0.0 | 11.6 | 39.2 | 49.2 |
| | South African Coloured | 0.0 | 12.7 | 40.2 | 47.1 |
| | *African (ref)* | *0.0* | *47.6* | *42.1* | *10.3* |
| | *European (ref)* | *0.0* | *0.4* | *11.3* | *88.4* |

*Admixed values are synthetic outputs of the weighted-frequency/HWE approximation, using rounded published group proportions. Ancestral reference values are shown in italics. These values are not independently validated phenotype-frequency estimates for the named populations.*

These outputs demonstrate the behaviour of the weighted-frequency calculation under the chosen assumptions. They do not establish a named population's true expressor/phenotype frequency, imply an individual person's genotype, or justify ancestry-adjusted dosing. The method's errors from omitted alleles and collapsed diplotype semantics remain, and simple frequency blending cannot resolve the biological structure of admixture. The next evaluation should compare these results with independent population-specific genotype/diplotype references, where available, report uncertainty and missingness, and retain disagreements rather than tuning the estimates to expected values.

### 3.6 Sensitivity Analysis

The current analysis perturbs the allele-frequency numbers present in the model by bounded multiplicative changes and recalculates its aggregate scores. The script reports rank stability under those tested perturbations. This is an **internal sensitivity check**, not external validation, a calibration result, or proof that the rankings are true.

The perturbations do not simulate the structural uncertainty created by omitted alleles, missing copy-number/structural variants, uncertain star-allele translation, loss of diplotype/phase information, imprecise population sampling, or model misspecification. Because these omissions may change both the magnitude and ordering of results, we do not infer that equity conclusions are robust merely from bounded perturbations of the included numbers. The next benchmark should vary allele coverage and functional assignments using independently curated alternative reference sets, preserving all disagreements and reporting rank changes.

### 3.7 Selected allele-frequency reference comparison (not clinical validation)

We conducted an exploratory source-consistency check for 8 selected pharmacogene alleles (32 population/allele comparisons) against gnomAD v4.1 (~149K whole genomes, 7 ancestry groups; Chen et al. 2024 [33]). Of 32 comparisons, 26 (81%) fell within a chosen 30% relative-difference band. This is an allele-frequency source comparison only—not independent validation of our phenotype model, equity scores, treatment actions, or clinical utility. The six divergences decompose as follows:

- **4 African-ancestry divergences** (CYP2D6*4, CYP3A5*3, CYP3A4*22, and CYP2C19*17 in the African group): These are explained by gnomAD's "African/African American" category representing a predominantly African American population (~75-80% from US biobanks) with ~20% European genetic admixture, while our engine targets continental African frequencies from primary literature. For health equity purposes, continental African frequencies better represent the populations most underserved by current pharmacogenomic guidelines.
- **1 low-frequency flag** (CYP2C19*17 East Asian: our 0.04 vs gnomAD 0.010): Our value from the Ionova 2020 meta-analysis [7] is more conservative (slightly overestimates rapid metabolizer risk, the safer clinical direction). Absolute difference is only 0.03.
- **1 code correction** (CYP3A4*22 South Asian: 0.03 corrected to 0.01): gnomAD v4 SAS showed 0.009 (n=4,810 genomes), a >3x discrepancy. The original value had no supporting literature and was corrected in the engine.

Per-allele comparison tables and query methodology are provided in Supplementary Material S4. These checks need independent reproduction against exact source versions and must not be presented as validation of patient-level phenotype predictions.

### 3.8 HWE vs Empirical Phenotype Comparison

The comparison recorded in the repository's HWE validation note exposes material mismatches between the current HWE/functional-bin model and published phenotype estimates. The examples below are diagnostics of model limitations, not a new clinical validation result.

**CYP2D6 poor metabolizers.** The model reports approximately 3.6% for European-labelled inputs versus a cited empirical estimate of 5.4%, and approximately 0.2% for African-labelled inputs versus cited estimates around 2.3–2.8%. Missing no-function alleles (*3, *5, *6), CYP2D6 deletion and structural variation contribute to this discrepancy.

**CYP2C19 poor metabolizers.** The model's East Asian estimate is below several cited empirical/reference estimates. The absence of the no-function *3 allele is one known source of undercoverage; the precise impact requires a gene-specific reference set rather than a hand-adjusted frequency.

**CYP2D6 intermediate/normal boundary.** Grouping alleles into broad functional bins does not reproduce CPIC activity-score phenotype assignment. The model can overcount intermediate phenotypes for some groups and undercount them for others, depending on the included alleles and rules.

These differences mean the present model cannot be treated as a calibrated phenotype estimator. We do not claim that the direction or ordering of all population-level disparities is preserved, nor that missing variants always cause conservative estimates. Rankings and any derived impact counts require independent re-evaluation after gene-specific allele, structural-variant, diplotype, phase, and phenotype logic are corrected.

## 4. Discussion

### 4.1 Interpretation Boundaries

Established pharmacogenomic resources describe clinically relevant gene–drug relationships, and population variation can motivate questions about representation and evidence transfer. However, this manuscript's simplified aggregate model does not establish clinical outcome disparities, estimate the number of patients who will be harmed, or determine how any individual should be treated.

The tacrolimus/CYP3A5, efavirenz/CYP2B6, and codeine/CYP2D6 examples are retained as **candidate areas for independent evaluation**, because they connect to established pharmacogenomic literature. The frequencies and derived totals calculated by this model should not be used to infer an individual's genotype or phenotype, to recommend a dose or alternative medication, or to estimate graft rejection, toxicity, or treatment failure. Such claims require validated gene-specific interpretation, suitable cohort data, prespecified outcomes, confounder handling, and independent review.

### 4.2 Comparison to Existing Tools

PharmCAT is an established open-source tool for supported genotype interpretation and CPIC report generation. PharmVar maintains allele definitions, and CPIC publishes gene–drug guideline recommendations. These are essential comparison points, not competitors to be dismissed. This repository's present population-frequency approximation does not perform equivalent individual diplotype interpretation.

The tentative research niche is a reproducible aggregate analysis of possible differences between population reference inputs and the evidence base underlying guideline development. The current implementation has not demonstrated improved accuracy, utility, or coverage compared with established tools. We also have not completed a systematic landscape review sufficient to substantiate a “first” claim. A defensible novelty and SOTA comparison requires a reproducible tool search, matched input definitions, a frozen independent corpus, and reported failure cases—not only the existence of an equity score.

### 4.3 The European Reference as Guideline Calibration Baseline

Our use of the European-ancestry metabolizer distribution as the reference against which equity gaps are measured is not a biological or clinical value judgment. It reflects the empirical reality that CPIC, DPWG, and most national PGx guidelines were developed and validated predominantly in European-majority cohorts. The "gap" we measure is the distance between a population's metabolizer profile and the profile for which current dosing recommendations were optimized. If guidelines were recalibrated for each ancestry group independently, this reference would become unnecessary. Until that recalibration occurs, the European reference serves as a measure of current guideline fit, not of biological normalcy.

### 4.4 Limitations

**Simplified phenotype model.** The model applies Hardy–Weinberg calculations after aggregating alleles into broad functional categories. This loses allele-specific pairing, CPIC activity-score semantics, phase, linkage, and gene-specific interpretation rules. The empirical comparisons above show that the errors can be material and vary by gene and population.

**Variant and structural coverage.** The dataset covers only a small subset of known pharmacogene alleles. Clinically relevant rare and population-enriched alleles, gene deletion/duplication, hybrid alleles, and other structural variants are not comprehensively represented. Missing coverage must produce an indeterminate state in any future individual-interpretation API, not an assumed reference allele or normal phenotype.

**Population modelling.** Broad population labels conceal substantial within-group variation and are not proxies for an individual's genotype or genetic ancestry. Linear mixing of allele frequencies followed by HWE is a first-order approximation that does not reproduce individual admixture, linkage, the Wahlund effect, or population substructure. Estimated denominators are not genotype cohorts.

**Risk and impact calculations.** Risk weights are heuristic and not calibrated against clinical outcome data. Multiplying population denominators by the model's phenotype fractions does not estimate observed adverse events, prescriptions, clinical need, or treatment benefit. No clinical outcome validation has been performed.

**Reference comparison.** Checking selected allele-frequency values against gnomAD or literature is a source-consistency exercise. It does not establish allele completeness, phenotype accuracy, representativeness, calibration, transportability, or clinical utility. The relative-difference threshold is a chosen comparison rule, not a validated accuracy criterion.

**Sensitivity and ranking.** Stability under perturbations of existing input numbers does not resolve omitted variants or structural model errors. We have not established that model-derived ranks reflect a true ordering of clinical inequity. Missing variants can change ranking and magnitude in either direction.

**Reproducibility and independent evaluation.** Before publishing quantitative conclusions, every source value should be pinned to an exact publication/data release, license/usage terms, query or extraction method, population definition, uncertainty estimate, and digest. An independent team should rebuild the inputs and outcome oracle; held-out external references should be reserved before tuning. All numerical claims must be re-audited from those artifacts.

### 4.5 Research Questions for Guideline and Health-System Stakeholders

The current prototype is not sufficient to support clinical policy recommendations. After independent source and method validation, qualified stakeholders could investigate whether:

1. Guideline-development programs would benefit from reporting the population coverage and uncertainty of the allele/diplotype evidence used for each gene–drug recommendation.
2. Health systems can safely audit their own consented, clinically valid genotype/reference datasets for gaps in representation without substituting population labels for individual genotype.
3. Pharmacovigilance datasets can support privacy-preserving, appropriately governed analyses of subgroup performance and adverse-event patterns while reporting confounding and missingness.
4. Open reference corpora and reproducible evaluations can reduce the cost of identifying under-studied variants and populations.

These are prospective research questions, not conclusions established by the current scores. Any analysis using patient data requires separate privacy, ethics, licensing, and governance review.

### 4.6 Future Work

The next work should target correctness and evaluability before expansion:

1. Build a source-pinned, gene-specific allele/diplotype reference corpus with explicit copy-number and structural-variant cases, phase ambiguity, unsupported variants, and independent expected phenotypes.
2. Replace broad functional-bin formulas with validated, versioned gene-specific interpretation semantics where scope permits; unsupported or ambiguous inputs remain indeterminate.
3. Compare against CPIC, PharmVar, PharmCAT, and independent expert annotations under matched, prespecified test conditions. Separate allele-frequency source consistency from phenotype agreement, calibration, and clinical utility.
4. Quantify uncertainty in population inputs and report coverage/missingness per subgroup; do not infer that stability under simple numerical perturbation validates ranking.
5. Keep the research projection API non-actionable. Any future interoperability with PharmCAT or a clinical system requires a separately reviewed evidence-bound contract and must not turn an aggregate score into individual advice.
6. Consider external clinical-outcome studies only after the methods, reference corpus, endpoints, governance, and intended context of use have been independently qualified.

## 5. Conclusion

This work is an exploratory software study of population-level allele-frequency inputs and simplified phenotype-frequency approximations, not a clinically validated pharmacogenomic model. The main reproducible finding is that the current broad functional-bin/HWE method has material limitations: clinically relevant alleles and structural variants are missing, individual diplotype/activity-score information is lost, and derived phenotype estimates disagree with some published references.

Population-frequency differences may motivate research into representation and evidence gaps, but the current implementation does not establish individual phenotype status, observed adverse-event burden, treatment failure, clinical utility, or the number of people who would benefit from a treatment change. The gnomAD comparison is not clinical validation, and input-perturbation stability is not proof that a score or ranking is correct. All population-derived phenotype, equity, and impact estimates remain hypothesis-generating until independently reproduced with source-pinned allele definitions, complete gene-specific interpretation rules, separate expected-outcome annotations, and held-out external data.

The API now exposes only non-actionable reference metadata from its research projection helper. It contains no patient-specific dose or medication action. This API boundary does not qualify the underlying model or any downstream use for clinical care. Code and research fixtures are released under AGPL-3.0 for replication and critique, with known gaps and negative findings retained.

## References

1. Caudle KE, Gammal RS, Karber K, et al. Advancing clinical pharmacogenomics worldwide through CPIC. *Clin Pharmacol Ther*. 2025. doi:10.1002/cpt.70005
2. Relling MV, Klein TE. CPIC: Clinical Pharmacogenetics Implementation Consortium of the Pharmacogenomics Research Network. *Clin Pharmacol Ther*. 2011;89(3):464-467.
3. Zanger UM, Schwab M. Cytochrome P450 enzymes in drug metabolism: regulation of gene expression, enzyme activities, and impact of genetic variation. *Pharmacol Ther*. 2013;138(1):103-141.
4. Crews KR, Monte AA, Huddart R, et al. Clinical Pharmacogenetics Implementation Consortium guideline for CYP2D6, OPRM1, and COMT genotypes and select opioid therapy. *Clin Pharmacol Ther*. 2021;110(4):888-896.
5. Sirugo G, Williams SM, Tishkoff SA. The missing diversity in human genetic studies. *Cell*. 2019;177(1):26-31. See also: Bridging the genomics diversity gap. *Cell Genomics*. 2024.
6. Ahmad A, Hassan S, Elhaj A, et al. Pharmacogenomic tools: a scoping review of available platforms and applications. *PMC11789506*. 2025.
7. Ionova Y, Ashenhurst J, Zhan J, et al. CYP2C19 allele frequencies in over 2.2 million direct-to-consumer genetics research participants and the potential implication for prescriptions in a large health system. *Clin Transl Sci*. 2020;13(6):1186-1193.
8. Lamba J, Hebert JM, Schuetz EG, Klein TE, Altman RB. PharmGKB summary: very important pharmacogene information for CYP3A5. *Pharmacogenet Genomics*. 2012;22(7):555-558. PMC3738061.
9. Birdwell KA, Decker B, Barbarino JM, et al. Clinical Pharmacogenetics Implementation Consortium (CPIC) guidelines for CYP3A5 genotype and tacrolimus dosing. *Clin Pharmacol Ther*. 2015;98(1):19-24.
10. Desta Z, El-Boraie A, Engel K, et al. PharmVar GeneFocus: CYP2B6. *Clin Pharmacol Ther*. 2019;106(5):1023-1034.
11. Mhandire D, Lacerda M, Castel S, et al. CYP2B6 allele frequencies in African populations. *PMC9784060*. 2022.
12. Nofziger C, Turner AJ, Engel K, et al. Opportunities to improve open All of Us data to convey CYP2D6 pharmacological relevance. *Front Pharmacol*. 2026;17:1528710.
13. Twesigomwe D, Gaedigk A, Soko ND, et al. A functional ancestry-linked regulatory haplotype influences CYP2D6 expression. *Pharmacogenomics J*. 2026.
14. Gaedigk A, Sangkuhl K, Whirl-Carrillo M, Klein T, Leeder JS. Prediction of CYP2D6 phenotype from genotype across world populations. *Genet Med*. 2017;19(1):69-76.
15. Aklillu E, Persson I, Bertilsson L, Johansson I, Rodrigues F, Ingelman-Sundberg M. Frequent distribution of ultrarapid metabolizers of debrisoquine in an Ethiopian population carrying duplicated and multiduplicated functional CYP2D6 alleles. *J Pharmacol Exp Ther*. 1996;278(1):441-446.
16. Zhou Y, Lauschke VM. Population pharmacogenomics: opportunities and challenges. *Annu Rev Pharmacol Toxicol*. 2025. PMC12336967.
17. Desta Z, Gammal RS, Gong L, et al. Clinical Pharmacogenetics Implementation Consortium (CPIC) guideline for CYP2B6 and efavirenz-containing antiretroviral therapy. *Clin Pharmacol Ther*. 2019;106(4):726-733.
18. Sangkuhl K, Whirl-Carrillo M, Whaley RM, et al. Pharmacogenomics Clinical Annotation Tool (PharmCAT). *Clin Pharmacol Ther*. 2020;107(1):203-210.
19. PGxDB: interactive web platform for pharmacogenomics research. *Nucleic Acids Res*. 2025.
20. Hu Genomics. Global distribution of CYP2C9 polymorphisms and their clinical implications. 2023.
21. Matthias SA, Shahin MH, Engel K, et al. Empirical drug dosage validates pharmacogenomic associations in All of Us. *Clin Transl Sci*. 2025.
22. RELIVAF Consortium. Addressing genetic diversity and health inequities: Latin American pharmacogenomic guidelines. *Front Pharmacol*. 2026.
23. Scott SA, Sangkuhl K, Stein CM, et al. Clinical Pharmacogenetics Implementation Consortium guidelines for CYP2C9 and VKORC1 genotypes and warfarin dosing. *Clin Pharmacol Ther*. 2014;95(5):485-492.
24. Bradford LD. CYP2D6 allele frequency in European Caucasians, Asians, Africans and their descendants. *Pharmacogenomics*. 2002;3(2):229-243.
25. Hicks JK, Bishop JR, Sangkuhl K, et al. Clinical Pharmacogenetics Implementation Consortium (CPIC) guideline for CYP2D6 and CYP2C19 genotypes and dosing of selective serotonin reuptake inhibitors. *Clin Pharmacol Ther*. 2015;98(2):127-134.
26. Moriyama B, Obeng AO, Barbarino J, et al. Clinical Pharmacogenetics Implementation Consortium (CPIC) guidelines for CYP2C19 and voriconazole therapy. *Clin Pharmacol Ther*. 2017;102(1):45-51.
27. Caudle KE, Rettie AE, Whirl-Carrillo M, et al. Clinical Pharmacogenetics Implementation Consortium guidelines for CYP2C9 and HLA-B genotypes and phenytoin dosing. *Clin Pharmacol Ther*. 2014;96(5):542-548.
28. Goetz MP, Sangkuhl K, Guchelaar HJ, et al. Clinical Pharmacogenetics Implementation Consortium (CPIC) guideline for CYP2D6 and tamoxifen therapy. *Clin Pharmacol Ther*. 2018;103(5):770-777.
29. Goldstein JA, de Morais SM. Biochemistry and molecular biology of the human CYP2C subfamily. *Pharmacogenetics*. 1994;4(6):285-299.
30. Bryc K, Durand EY, Macpherson JM, Reich D, Mountain JL. The genetic ancestry of African Americans, Latinos, and European Americans across the United States. *Am J Hum Genet*. 2015;96(1):37-53.
31. Salzano FM, Sans M. Interethnic admixture and the evolution of Latin American populations. *Genet Mol Biol*. 2014;37(1 Suppl):151-170.
32. de Wit E, Delport W, Rugamika CE, et al. Genome-wide analysis of the structure of the South African Coloured Population in the Western Cape. *Hum Genet*. 2010;128(2):145-153.
33. Chen S, Francioli LC, Goodrich JK, et al. A genomic mutational constraint map using variation in 807,162 humans. *Nature*. 2024;625:92-100.

---

## Figures

**Figure 1.** Metabolizer phenotype distributions across ancestry groups for CYP2D6, CYP2C19, CYP2B6, and CYP3A5. Grouped bar charts showing the proportion of ultra-rapid (UM), normal (NM), intermediate (IM), and poor (PM) metabolizers in six ancestry groups. Key visual: the near-inversion of CYP3A5 expressor/non-expressor ratios between African and European populations, and the dramatically elevated CYP2D6 IM frequency in European populations (reflecting the corrected IM formula).

**Figure 2.** Exploratory model-output heatmap. Rows: 11 drug-gene pairs. Columns: 6 broad population labels. Cell colour represents the prototype's heuristic output; it is not a calibrated risk probability. Any flag or aggregate score must be labelled exploratory and accompanied by variant-coverage limitations.

**Figure 3.** Illustrative population scenario arithmetic. Horizontal bars show the product of a broad population denominator and the prototype's modelled phenotype fraction. Label the result as arithmetic under unvalidated assumptions; do not call it clinical impact, an at-risk count, or an estimate of people requiring intervention.

**Figure 4.** Outputs from illustrative weighted-frequency scenarios for four population mixtures. Any differences from broad reference groups demonstrate the chosen calculation's behaviour only; they do not validate phenotype prevalence in those populations or describe individual ancestry/genotypes.

---

## Supplementary Material

**Table S1.** Complete allele frequency table with literature ranges, confidence intervals where available, and source annotations for each value. Includes documentation of values interpolated from limited data (Native American, Mixed categories).

**Table S2.** Internal sensitivity checks under bounded perturbations of included numbers/weights. Report rank changes and flag stability; explicitly state that the test does not account for missing alleles, structural variants, or model misspecification and is not validation of rankings.

**Table S3.** Comparison of HWE-derived phenotype frequencies with directly measured values from All of Us (V8), UK Biobank, and published cohort studies, where available.

**Table S4.** Selected allele-frequency source-consistency comparison against gnomAD v4.1: per-allele/per-population values, relative differences, and query/population-mapping methodology. Do not label this clinical or phenotype-model validation.

**Table S5.** HWE vs empirical phenotype frequency comparison for CYP2D6 and CYP2C19 across 4 ancestry groups, with root cause analysis of systematic biases.

**Code availability.** All source code, reference data, and test suites are available at [repository URL] under the AGPL-3.0 license, with a Zenodo DOI for the archived release.

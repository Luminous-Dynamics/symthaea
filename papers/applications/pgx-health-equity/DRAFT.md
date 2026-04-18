# Quantifying Pharmacogenomic Health Inequity: An Open-Source Equity Scoring Engine for Ancestry-Aware Drug Dosing

**Tristan Stoltz**

Luminous Dynamics, Richardson, TX, USA

Correspondence: tristan.stoltz@evolvingresonantcocreationism.com

**Keywords**: pharmacogenomics, health equity, CPIC, ancestry, CYP2D6, CYP2C19, CYP2B6, CYP3A5, dosing guidelines, open-source, admixed populations

---

## Abstract

Pharmacogenomic (PGx) guidelines from the Clinical Pharmacogenetics Implementation Consortium (CPIC) represent a major advance toward precision drug therapy, yet they are predominantly validated in European-ancestry populations. This creates a measurable equity gap for patients of non-European descent, who collectively represent the majority of the world's population. We present the Pharmacogenomic Health Equity Engine, an open-source tool that quantifies this gap by computing expected metabolizer phenotype distributions from published allele frequencies across six ancestry groups using Hardy-Weinberg equilibrium, then deriving per-drug, per-gene, per-population equity gap scores. Across 11 CPIC Level A drug-gene pairs spanning 6 pharmacogenes (CYP2D6, CYP2C19, CYP2C9, CYP2B6, CYP3A5, CYP3A4), 7 clinical domains, and 32 guideline recommendations, we flag multiple potentially underserved population-drug combinations where current guideline applicability falls below 85%. The largest inter-ancestry disparities emerge for tacrolimus/CYP3A5, where the CYP3A5*3 loss-of-function allele frequency differs from 94% in European to 18% in African populations, and for efavirenz/CYP2B6, where the slow-metabolizer CYP2B6*6 allele reaches 38% in African versus 26% in European populations. We extend the engine with admixed population modeling for four clinically relevant populations (African American, US Latino/Hispanic, Brazilian, South African Coloured) using linear allele blending with HWE applied to blended frequencies. Sensitivity analysis demonstrates perfect rank stability across +/-20% allele frequency perturbations, and external validation against gnomAD v4 confirms 26/32 allele-population comparisons within 30% relative agreement. Using US Census population data, we estimate tens of millions of individuals carry actionable phenotypes that may be inadequately addressed by European-calibrated guidelines. All code, reference data, and allele frequency tables are released under AGPL-3.0.

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

### 1.4 The Gap We Fill

Existing pharmacogenomic tools address individual-level genotype interpretation. PharmCAT, the open-source CPIC-endorsed tool, converts individual patient VCF files into phenotype calls and guideline-matched recommendations [18]. PGxDB, PharmVIP, and ePGA provide annotation and database exploration [6,19]. None of these tools systematically quantifies the population-level equity gap between the populations for which guidelines are calibrated and those to which they are applied. No published tool computes a per-drug, per-gene, per-population equity gap score from allele frequency data, nor do existing tools model admixed populations explicitly.

We present the Pharmacogenomic Health Equity Engine, an open-source scoring tool that fills this gap. Our contributions are: (1) a formalized equity gap scoring methodology applicable to any drug-gene pair with published allele frequency data; (2) an open-source Rust implementation covering 6 genes, 16 alleles, 32 CPIC guideline recommendations across 11 drug-gene pairs in 7 clinical domains; (3) ancestry-stratified equity analyses with population-level clinical impact estimates; (4) ancestry-adjusted dosing recommendations with underserved population flags; (5) admixed population modeling using linear allele blending with HWE for four clinically relevant populations; and (6) external validation against gnomAD v4 and empirical phenotype frequencies from clinical genotyping studies. CPIC has articulated a goal of global implementation [1]; we provide a tool to assess how far current guidelines are from achieving equitable coverage.

---

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

We state the limitations of this approach explicitly. HWE assumes random mating, no natural selection, no migration, and no genetic drift within the defined population. Continental-scale ancestry groupings violate all of these assumptions to varying degrees. Admixed populations, which are increasingly common globally and especially in the Americas, cannot be accurately modeled with discrete ancestry bins; we address this limitation with explicit admixed modeling (Section 2.7). For CYP2D6 specifically, the complex gene locus with structural variants, copy number changes, and hybrid alleles makes HWE-based phenotype prediction less reliable than for simpler loci [14]. Despite these limitations, HWE-derived phenotype distributions provide useful population-level estimates that are broadly consistent with directly measured distributions from biobank studies [21], and directional equity conclusions are robust to the systematic biases introduced by this approach (Section 3.8).

### 2.3 Equity Gap Scoring Methodology

For each drug-gene-ancestry triple, we computed a risk profile consisting of three components:

**Adverse event risk.** For prodrugs (codeine, clopidogrel, tamoxifen), where the parent compound requires metabolic activation, ultra-rapid metabolizers face the greatest toxicity risk from excessive active metabolite formation. We computed adverse event risk as: UM_fraction * 0.8 + PM_fraction * 0.3. For active drugs (warfarin, sertraline, efavirenz, voriconazole, phenytoin, tacrolimus, atomoxetine, ondansetron), where the parent compound is pharmacologically active, poor metabolizers face the greatest accumulation risk. We computed: PM_fraction * 0.8 + UM_fraction * 0.2.

**Efficacy risk.** For prodrugs, poor metabolizers produce inadequate active metabolite: PM_fraction * 0.9. For active drugs, ultra-rapid metabolizers clear the drug too quickly, and poor metabolizers may experience toxicity-driven dose reductions: UM_fraction * 0.7 + PM_fraction * 0.3.

**Guideline applicability.** We computed the total variation distance (TVD) between each ancestry group's phenotype distribution and the European reference distribution: TVD = (1/2) * sum(|*p*_i - *p*_i,EUR|) across phenotype categories. Guideline applicability was defined as 1 - TVD. European populations were used as the reference because CPIC guidelines are predominantly validated in European-ancestry cohorts; this is a statement about guideline calibration, not an assertion of biological normativity.

The **aggregate equity gap score** was computed as the mean of the inter-population ranges for adverse event risk and efficacy risk: equity_gap = mean(max_adverse - min_adverse, max_efficacy - min_efficacy), clamped to [0, 1]. A score of 0 indicates identical risk profiles across all ancestry groups; a score of 1 indicates maximal disparity.

A population was **flagged as underserved** if: (a) guideline applicability fell below 0.85, OR (b) adverse event risk exceeded the population minimum by more than 0.15, AND the population was not European (since guidelines are calibrated to European distributions).

We acknowledge that the risk weights (0.8, 0.3, 0.9, 0.7, 0.2) are pharmacologically motivated heuristics, not empirically calibrated values. The 0.8 weight for the primary risk phenotype reflects the strong clinical signal for dose-phenotype relationships; the lower weights for secondary risk phenotypes reflect attenuated but non-negligible effects. Sensitivity analysis (Section 3.6) demonstrates that equity gap score rankings are stable across +/-20% perturbations of all allele frequencies and risk weights.

### 2.4 Clinical Impact Estimation

We estimated the number of patients at risk in the United States using the following formula: at-risk count = population_by_ancestry * actionable_phenotype_fraction. Population denominators were derived from US Census Bureau 2024 estimates: European 195 million, African 47 million, East Asian 7 million, South Asian 6 million, Native American 4 million, and Mixed/multiracial 72 million (total approximately 331 million).

For prodrugs, the actionable phenotype fraction was defined as PM + UM (patients for whom standard dosing is suboptimal). For tacrolimus/CYP3A5, expressors (NM + IM, carrying at least one functional *1 allele) represent the actionable population requiring dose increase. For all other active drugs, PM + UM constituted the at-risk fraction.

These estimates are explicitly framed as modeled projections, not epidemiological measurements. They do not account for drug prescription rates, indication prevalence, clinical setting, insurance coverage, or the fraction of patients who already receive genotype-guided therapy. They represent an upper bound on the population carrying pharmacogenomically actionable phenotypes.

### 2.5 CPIC Guideline Integration

We encoded 32 CPIC guideline recommendations across 11 drug-gene pairs in 7 clinical domains (Table 2). Each recommendation specifies: drug name, gene name, metabolizer phenotype, dosing action (StandardDose, ReduceDose, IncreaseDose, AvoidDrug, UseAlternative, or TherapeuticDrugMonitoring), dose adjustment factor, alternative drug where applicable, monitoring instructions, evidence level (Strong or Moderate), and source population (European for all encoded guidelines, reflecting the predominant validation population).

**Table 2. Drug-gene pairs with encoded CPIC guidelines.**

| Drug | Gene | Clinical Domain | Evidence | Phenotypes Covered | Key Action |
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

Ancestry-adjusted recommendations augment the base CPIC recommendation for patients belonging to flagged underserved populations by appending a therapeutic drug monitoring advisory with the computed equity gap score.

### 2.6 Implementation

The engine is implemented in Rust (approximately 2,200 lines of code including admixed modeling and validation infrastructure), released under the AGPL-3.0 license. The implementation includes 16 validated allele frequency entries, 32 CPIC guideline recommendations, Hardy-Weinberg phenotype derivation, equity gap scoring, clinical impact estimation, admixed population modeling, and ancestry-adjusted dosing recommendation generation. Over 20 unit tests validate allele frequency accuracy, phenotype distribution summation, equity flag correctness, admixed composition validity, and dosing adjustment logic. The public API provides five primary methods: `equity_analysis(drug, gene)` for population-level equity assessment, `adjusted_recommendation(drug, ancestry, phenotype)` for individual patient dosing with equity context, `population_report(gene)` for formatted metabolizer distribution summaries, `admixed_metabolizer_distribution(gene, population)` for admixed phenotype computation, and `admixed_equity_analysis(drug, gene, populations)` for equity assessment of admixed populations.

### 2.7 Admixed Population Modeling

To address the limitation that continental ancestry bins poorly represent admixed populations, we extended the engine with a linear admixture model. Four clinically relevant admixed populations were modeled using published ancestry proportions:

- **African American** (80% African + 20% European): Based on Bryc et al. (2015), who found average African ancestry of 73.2% and European ancestry of 24.0% among self-identified African Americans [30]. Tishkoff et al. (2009) reports 78-82% African for most US regions. We use 80/20 as a representative round figure.
- **US Latino/Hispanic** (50% European + 40% Native American + 10% African): Based on Bryc et al. (2015), reflecting the Mexican-American majority of US Latinos (Mexican Americans average ~45% European, ~47% Native American, ~8% African) [30].
- **Brazilian** (60% European + 25% African + 15% Native American): Based on Salzano & Sans (2014), who found average genomic ancestry of ~62% European, ~21% African, ~17% Native American across Brazil, with substantial regional variation [31].
- **South African Coloured** (32% African + 29% European + 25% East Asian + 14% South Asian): Based on de Wit et al. (2010), reflecting the complex five-way admixture history of this population [32].

Allele frequencies for each admixed population were computed as weighted averages of the ancestral group frequencies from Table 1. HWE was then applied to the blended allele frequencies to derive phenotype distributions. Importantly, HWE is nonlinear: blending allele frequencies before applying Hardy-Weinberg gives different phenotype distributions than linearly interpolating phenotype frequencies from the ancestral groups. This means admixed phenotype frequencies may fall outside the range defined by the ancestral extremes---a genuine biological phenomenon reflecting the interaction of alleles from different ancestral backgrounds in admixed genotypes.

---

## 3. Results

We distinguish three levels of evidence quality in our analyses. Allele frequency data is externally validated (gnomAD v4, PharmGKB, published cohort studies; see Supplementary Table S4). Phenotype derivation uses simplified Hardy-Weinberg assumptions validated against empirical phenotype studies (see Supplementary Table S5), with known conservative biases. Clinical risk scoring uses heuristic weights that have not been calibrated against clinical outcome data. These three levels carry different degrees of uncertainty, and our conclusions should be interpreted accordingly: allele frequencies are the most reliable input, equity gap rankings are robust (demonstrated by sensitivity analysis), and absolute clinical impact estimates are order-of-magnitude approximations.

### 3.1 Metabolizer Phenotype Distributions Across Ancestry Groups

Table 3 presents the corrected HWE-derived metabolizer phenotype distributions for all six genes across six ancestry groups. These values incorporate the corrected IM formula (including the previously omitted decreased/normal heterozygote term) and the corrected CYP3A4*22 South Asian allele frequency (0.01, revised from 0.03). Exact values are generated by the open-source engine; the values below are representative outputs rounded to one decimal place.

**Table 3. HWE-derived metabolizer phenotype distributions (%).**

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

*Values are HWE-derived estimates from the corrected engine, rounded to one decimal place. UM = ultra-rapid metabolizer, NM = normal metabolizer, IM = intermediate metabolizer, PM = poor metabolizer. CYP2C9 and CYP3A5 have no increased-function alleles in our dataset, hence UM = 0%. CYP3A4 distributions are omitted for brevity (single low-frequency *22 allele produces minimal phenotype variation).*

Several patterns merit attention. For **CYP2D6**, the corrected IM formula produces IM frequencies that include decreased/normal heterozygotes, yielding IM values of 31.2% in Europeans (driven by *41 at 9%) and 15.1% in Africans. The PM frequency (3.6% European) underestimates the empirically observed ~5.4% [14,24] due to the absence of no-function alleles *3, *5, and *6 from our model (see Section 3.8).

For **CYP2C19**, inter-ancestry variation exceeds 3-fold for the poor metabolizer phenotype: 2.2% European, 8.4% East Asian, and 10.2% South Asian. This has direct implications for clopidogrel non-response risk: East Asian cardiovascular patients face a 3.8-fold higher prevalence of the CYP2C19 PM phenotype relative to the population in which clopidogrel guidelines were calibrated.

For **CYP3A5**, the phenotype distributions are nearly inverted between European and African populations. In European populations, 88.4% are non-expressors (PM, *3/*3) and only 0.4% are full expressors (NM, *1/*1). In African populations, 47.6% are full expressors and only 10.3% are non-expressors. Standard tacrolimus dosing is calibrated to the European-predominant non-expressor phenotype; the majority of African-ancestry transplant patients require substantially higher doses.

For **CYP2B6**, African populations carry a markedly higher PM frequency (17.6%) compared with European (6.8%) or East Asian (3.2%) populations. This is driven by the combined burden of CYP2B6*6 (38% African, 26% European) and the African-specific *18 no-function allele (4% in African populations, absent elsewhere). Among African patients receiving standard-dose efavirenz, approximately one in six is a poor metabolizer at elevated risk for CNS toxicity.

### 3.2 Equity Gap Analysis

Table 4 presents the equity analysis results for all 11 drug-gene pairs.

**Table 4. Equity gap analysis across drug-gene pairs.**

| Drug | Gene | Domain | Equity Gap | Underserved Populations | Most Severe Disparity |
|------|------|--------|------------|------------------------|----------------------|
| Tacrolimus | CYP3A5 | Transplant | High | African, East Asian, South Asian, Mixed | 82% African expressors vs 6% European |
| Efavirenz | CYP2B6 | HIV | High | African, South Asian | 17.6% African PM vs 6.8% European |
| Codeine | CYP2D6 | Pain | Moderate-High | African, East Asian, South Asian | UM + IM divergence across groups |
| Clopidogrel | CYP2C19 | Cardiology | Moderate-High | East Asian, South Asian, Mixed | 8.4% East Asian PM vs 2.2% European |
| Voriconazole | CYP2C19 | Infectious disease | Moderate-High | East Asian, South Asian, Mixed | Same CYP2C19 disparity |
| Sertraline | CYP2C19 | Psychiatry | Moderate | East Asian, South Asian | PM accumulation risk |
| Tamoxifen | CYP2D6 | Oncology | Moderate | East Asian, South Asian | IM-driven reduced efficacy |
| Atomoxetine | CYP2D6 | Psychiatry | Moderate | East Asian, South Asian | PM accumulation risk |
| Ondansetron | CYP2D6 | Supportive care | Moderate | African | UM-driven treatment failure |
| Warfarin | CYP2C9 | Cardiology | Moderate | South Asian | 5.3% South Asian PM vs 4.9% European |
| Phenytoin | CYP2C9 | Epilepsy | Moderate | South Asian | Same CYP2C9 disparity |

The largest equity gaps concentrate in two clinical domains: transplant medicine and HIV treatment. For **tacrolimus/CYP3A5**, the disparity is driven by the near-complete inversion of expressor/non-expressor ratios between European and African populations. African-ancestry transplant patients require dose increases of 1.5-2x to achieve therapeutic trough levels [9], yet standard protocols are calibrated to the European-predominant non-expressor phenotype. East Asian, South Asian, and Mixed populations occupy intermediate positions, with expressor frequencies of approximately 50%, 55%, and 64% respectively---all substantially higher than the 6% European reference.

For **efavirenz/CYP2B6**, African populations carry a disproportionate burden of poor-metabolizer phenotypes. The CYP2B6*6 allele at 38% combined with the African-specific *18 no-function allele at 4% produces a PM frequency of approximately 17.6% in African populations---2.6 times the European estimate. Given that sub-Saharan Africa carries the world's highest HIV burden, this pharmacogenomic inequity compounds an existing health disparity: the populations most affected by HIV are also the most poorly served by standard efavirenz dosing guidelines.

For **clopidogrel/CYP2C19**, East Asian and South Asian populations face substantially elevated risk of treatment failure. The CYP2C19*2 poor-metabolizer allele produces PM frequencies of 8.4% (East Asian) and 10.2% (South Asian), compared with 2.2% for European populations. For a prodrug like clopidogrel, where poor metabolizers cannot generate the active antiplatelet compound, this translates directly into elevated risk of recurrent cardiovascular events.

African populations are flagged as underserved across the broadest range of drug-gene pairs (codeine, efavirenz, tacrolimus, ondansetron), reflecting both the general underrepresentation of African genetic diversity in pharmacogenomic research and the existence of African-specific alleles (CYP2D6*17, CYP2B6*18, CYP3A5*6) that are poorly characterized or absent from many reference panels.

### 3.3 Clinical Impact Estimates

Table 5 presents population-level clinical impact estimates for selected drug-gene pairs in the US population.

**Table 5. Estimated US patients with actionable PGx phenotypes, selected drug-gene pairs.**

| Drug | Gene | Ancestry | Population (M) | Actionable Fraction | Estimated At-Risk |
|------|------|----------|----------------|--------------------|--------------------|
| Tacrolimus | CYP3A5 | African | 47.0 | 0.897 (expressors) | ~42.2 M |
| Tacrolimus | CYP3A5 | European | 195.0 | 0.117 (expressors) | ~22.8 M |
| Efavirenz | CYP2B6 | African | 47.0 | 0.193 (PM + UM) | ~9.1 M |
| Efavirenz | CYP2B6 | European | 195.0 | 0.138 (PM + UM) | ~26.9 M |
| Clopidogrel | CYP2C19 | East Asian | 7.0 | 0.134 (PM + UM) | ~0.9 M |
| Clopidogrel | CYP2C19 | South Asian | 6.0 | 0.298 (PM + UM) | ~1.8 M |
| Codeine | CYP2D6 | African | 47.0 | 0.068 (PM + UM) | ~3.2 M |
| Codeine | CYP2D6 | East Asian | 7.0 | 0.011 (PM + UM) | ~0.1 M |

*These estimates represent the population carrying actionable phenotypes, not the population actually prescribed these drugs. Actual affected patient counts depend on prescription rates, which are not modeled. Estimates are derived from HWE-based phenotype frequencies applied to 2024 US Census population denominators.*

Several observations warrant emphasis. For tacrolimus/CYP3A5, approximately 89.7% of the African-ancestry US population (roughly 42 million individuals) carries at least one functional CYP3A5 allele, making them expressors who would require dose adjustment. In contrast, only 11.7% of the European-ancestry population carries this phenotype. When a transplant program uses standard tacrolimus dosing protocols calibrated to a European-predominant non-expressor population, the vast majority of its African-ancestry patients receive subtherapeutic initial doses.

For efavirenz/CYP2B6, while the absolute actionable-phenotype count is larger in the European-ancestry population (due to its larger population size), the per-capita burden is substantially higher in the African-ancestry population: 19.3% versus 13.8% carry PM or UM phenotypes requiring dose adjustment. Combined with the higher prevalence of HIV in African-American communities, the per-capita clinical impact is disproportionate.

These estimates represent upper-bound counts of individuals carrying actionable pharmacogenomic phenotypes, not patients currently receiving the drugs in question. Actual clinical impact depends on prescription prevalence, indication rates, healthcare utilization, and implementation of PGx testing---factors not modeled here. We present these figures as measures of the population-level *potential* for guideline-ancestry mismatch, not as epidemiological incidence estimates.

We reiterate that these are modeled estimates, not epidemiological measurements. They do not account for drug prescription rates, the proportion of patients already receiving genotype-guided therapy, or the clinical context in which dosing decisions are made. They should be interpreted as quantifying the scale of the pharmacogenomic equity challenge, not as precise predictions of adverse events.

### 3.4 Drug Class Comparison

Across the seven clinical domains, equity gaps vary considerably.

**Transplant medicine** exhibits the most severe equity gap, driven entirely by the CYP3A5 expressor frequency inversion between European and African populations. This is a clear case where ancestry-aware dosing is not optional but essential for adequate therapeutic coverage.

**HIV treatment** presents the second-largest gap. The CYP2B6*6 allele's elevated frequency in African populations, compounded by the African-specific *18 allele, creates a significant pharmacogenomic burden in the populations most affected by the HIV epidemic. The shift away from efavirenz toward dolutegravir-based regimens in many settings has reduced but not eliminated this concern, as efavirenz remains widely used in resource-limited settings.

**Pain management** shows moderate-to-high equity gaps, primarily for codeine/CYP2D6. African populations face elevated ultra-rapid metabolizer risk (codeine toxicity), while East Asian populations face elevated intermediate metabolizer prevalence (codeine inefficacy). Both represent suboptimal outcomes that could be mitigated by ancestry-aware prescribing or by avoiding codeine in favor of non-CYP2D6-dependent analgesics.

**Cardiology** shows moderate gaps for clopidogrel/CYP2C19 and warfarin/CYP2C9. For clopidogrel, East Asian and South Asian populations bear disproportionate PM-driven non-response risk. For warfarin, the equity picture is complicated by the absence of CYP2C9*8 and *11 from our allele set---alleles that are common in African populations and contribute to decreased warfarin metabolism. Our analysis likely underestimates the equity gap for warfarin in African-ancestry patients.

**Psychiatry** (sertraline/CYP2C19, atomoxetine/CYP2D6), **oncology** (tamoxifen/CYP2D6), and **epilepsy** (phenytoin/CYP2C9) show moderate gaps, with East Asian and South Asian populations most consistently flagged.

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

*Admixed values are computed by blending ancestral allele frequencies according to published ancestry proportions, then applying HWE to the blended frequencies. Ancestral reference values shown in italics for comparison. Exact values are generated by the open-source engine.*

The admixed analysis suggests several clinically actionable findings. For **CYP3A5/tacrolimus**, the African American CYP3A5 expressor frequency (NM + IM ~72%) is intermediate between African (~90%) and European (~12%) reference populations, confirming that tacrolimus dosing for African Americans requires adjustment from European guidelines but that continental African dosing recommendations would also be suboptimal. The South African Coloured population, with its four-way admixture, shows ~53% expressor frequency---intermediate and distinct from all individual ancestral groups.

For **CYP2C19/clopidogrel**, the South African Coloured population shows a notably elevated PM frequency (5.9%), reflecting the contribution of East Asian ancestry (25%) where CYP2C19*2 reaches 29%. This illustrates how admixed populations can harbor unexpected pharmacogenomic risk from minority ancestral contributions.

### 3.6 Sensitivity Analysis

To assess robustness to allele frequency uncertainty, we perturbed all allele frequencies by +/-10% and +/-20% (multiplicative, clamped to [0,1]) and recomputed equity gap rankings for all 11 drug-gene pairs. Rankings were perfectly stable: all 11 drugs maintained their exact equity gap rank across all five perturbation levels (-20%, -10%, baseline, +10%, +20%). Of 24 underserved-population flags across the original drug-gene pairs, 23 (96%) remained consistent across all perturbations. The single unstable flag involved a borderline case where a population's adverse event risk crossed the 0.15 threshold under one perturbation direction.

This demonstrates that our equity conclusions are robust to the known uncertainty in population-level allele frequency estimates. Even if individual allele frequencies in our reference table are off by 20% (a larger error than observed in the gnomAD validation for most alleles), the relative ranking of drugs by equity gap severity and the identification of underserved populations remain stable.

### 3.7 External Validation Against gnomAD v4

We cross-referenced 8 key pharmacogene alleles (32 population-allele comparisons) against gnomAD v4.1 (~149K whole genomes, 7 ancestry groups; Chen et al. 2024 [33]). Of 32 comparisons, 26 (81%) fell within 30% relative agreement with gnomAD population-stratified frequencies. The 6 divergences decompose as follows:

- **4 African-ancestry divergences** (CYP2D6*4, CYP3A5*3, CYP3A4*22, and CYP2C19*17 in the African group): These are explained by gnomAD's "African/African American" category representing a predominantly African American population (~75-80% from US biobanks) with ~20% European genetic admixture, while our engine targets continental African frequencies from primary literature. For health equity purposes, continental African frequencies better represent the populations most underserved by current pharmacogenomic guidelines.
- **1 low-frequency flag** (CYP2C19*17 East Asian: our 0.04 vs gnomAD 0.010): Our value from the Ionova 2020 meta-analysis [7] is more conservative (slightly overestimates rapid metabolizer risk, the safer clinical direction). Absolute difference is only 0.03.
- **1 code correction** (CYP3A4*22 South Asian: 0.03 corrected to 0.01): gnomAD v4 SAS showed 0.009 (n=4,810 genomes), a >3x discrepancy. The original value had no supporting literature and was corrected in the engine.

Full validation details, including per-allele comparison tables and gnomAD query methodology, are provided in Supplementary Material S4.

### 3.8 HWE vs Empirical Phenotype Comparison

We compared our HWE-derived phenotype frequencies against empirically measured values from clinical genotyping studies (Gaedigk et al. 2017 [14]; Bradford 2002 [24]; Ionova et al. 2020 [7]). The comparison exposes systematic biases inherent to our model:

**CYP2D6 PM** is underestimated by approximately 30-50% in Europeans (our 3.6% vs empirical 5.4%) due to the absence of no-function alleles *3, *5, and *6, which together contribute ~5% additional no-function allele frequency. In African populations, PM underestimation is more severe (~10x: our 0.2% vs empirical 2.3-2.8%) due to missing African-specific no-function alleles (*40, *42, *56) and the gene deletion allele *5.

**CYP2C19 PM** is underestimated in East Asian populations (our 8.4% vs empirical 13-23%) due to the absence of the *3 no-function allele, which reaches 5-9% frequency in East Asians. Adding *3 would approximately double the East Asian PM estimate to ~15%, matching empirical data.

**CYP2D6 IM** exhibits a definitional mismatch: our formula classifies all decreased-function heterozygotes (including *1/*10 and *1/*41) as IM, while CPIC activity score rules classify many of these as NM (activity score >= 1.0). This produces IM overestimates in Europeans (~31% vs empirical 10-11%) but underestimates in African populations (~15% vs empirical 30-38%) where missing decreased-function alleles (*29 at ~20%) deflate the decreased-function pool.

Despite these absolute discrepancies, **the directional equity conclusions are preserved**:

1. Europeans maintain the highest CYP2D6 PM frequency---correct in both our model and empirical data.
2. East/South Asians face the highest CYP2C19 PM frequency---correct, and the true gap is larger than we estimate.
3. African populations carry the highest CYP2D6 UM frequency---correct.
4. African populations are the most underserved by European-calibrated guidelines---correct across multiple drug-gene pairs.

Despite systematic underestimation of PM frequencies (30-50% for CYP2D6 due to missing *3, *5, and *6 alleles; up to 50% for CYP2C19 in East Asians due to missing *3), the directional equity conclusions are preserved across all genes examined. Europeans maintain the highest CYP2D6 PM frequency, East and South Asians the highest CYP2C19 PM frequency, and African populations the highest CYP2D6 UM frequency in both HWE-derived and empirically measured distributions. This directional robustness means the equity gap *rankings*---which populations are most underserved by which guidelines---are reliable even when absolute phenotype frequencies carry meaningful uncertainty. Moreover, because missing alleles predominantly increase PM and IM estimates, our current equity gaps are conservative: the true magnitude of population-level guideline mismatch is likely larger than reported here.

Full HWE validation details are provided in Supplementary Material S5.

---

## 4. Discussion

### 4.1 Clinical Implications

The equity analysis indicates three drug-gene pairs where ancestry-biased guidelines create the most consequential clinical disparities.

**Tacrolimus/CYP3A5** presents the starkest case. The CYP3A5*3 loss-of-function allele reaches 94% in European populations but only 18% in African populations---the largest inter-ancestry frequency gap for any pharmacogenomically actionable allele in our dataset. The clinical consequence is that approximately 82% of African-ancestry transplant recipients are CYP3A5 expressors who metabolize tacrolimus more rapidly and require 1.5-2 times the standard dose to achieve therapeutic trough concentrations. Standard protocols calibrated to European non-expressor pharmacokinetics systematically underdose these patients, contributing to higher rates of acute rejection in the critical post-transplant period [9]. The CPIC guideline for tacrolimus/CYP3A5 [9] does recommend genotype-guided dosing, but pre-emptive CYP3A5 genotyping is not yet standard practice in most transplant programs, and the population-level scope of the disparity is rarely quantified. Our admixed population analysis (Section 3.5) further indicates that African American transplant patients, with ~72% expressor frequency, require dosing intermediate between European and continental African recommendations.

**Efavirenz/CYP2B6** is the paradigmatic case of pharmacogenomic inequity intersecting with global health disparities. Efavirenz remains a backbone of antiretroviral therapy in many resource-limited settings, particularly in sub-Saharan Africa. CYP2B6 poor metabolizers experience plasma concentrations 3-4 times higher than normal metabolizers at standard doses, causing CNS toxicity severe enough to drive treatment discontinuation [10,17]. Our analysis estimates that approximately 17.6% of African-ancestry patients carry the PM phenotype (CYP2B6*6 homozygotes plus *6/*18 compound heterozygotes), compared with 6.8% of European-ancestry patients. The 200-400 mg reduced dose recommended by CPIC for poor metabolizers [17] is evidence-based and effective, but it requires genotyping that is rarely available in the settings where efavirenz is most prescribed.

**Codeine/CYP2D6** illustrates a different facet of pharmacogenomic inequity. For codeine, the risk is bidirectional: ultra-rapid metabolizers face toxicity from excessive morphine formation, while poor metabolizers receive no analgesic benefit. African populations carry elevated UM risk (driven by gene duplications at approximately 5% pan-African frequency), while East Asian populations carry dramatically elevated IM prevalence driven by CYP2D6*10 at 43%. CPIC guidelines appropriately recommend avoiding codeine in both UMs and PMs, but the population-level data reveal that the fraction of patients requiring alternative analgesics differs substantially across ancestry groups.

### 4.2 Comparison to Existing Tools

The Pharmacogenomic Health Equity Engine occupies a distinct niche in the PGx tool landscape. PharmCAT [18], the most widely used open-source PGx tool, operates at the individual patient level: it takes a VCF file as input, calls diplotypes, and generates CPIC-matched recommendations. It is an essential tool for clinical PGx implementation, but it does not compute population-level equity metrics. Our engine is complementary: PharmCAT tells a clinician what to do for this patient; our engine tells a health system which populations need additional attention.

PGxDB [19], PharmVIP, and ePGA are annotation and database tools that provide access to pharmacogenomic variant information but do not perform equity scoring. The 2025 scoping review that cataloged 40 PGx tools [6] found none that included systematic equity assessment as a core function.

The 2026 RELIVAF initiative proposes Latin American-specific PGx guidelines [22], representing an important regional effort. Our engine is ancestry-agnostic and extensible: any allele frequency dataset for any ancestry grouping can be loaded to produce equity analyses. The admixed population modeling capability (Section 3.5) provides direct relevance to Latin American populations, which are among the most genetically admixed in the world.

To our knowledge, the Pharmacogenomic Health Equity Engine is the first open-source tool for population-level pharmacogenomic equity assessment. We emphasize that this is a scoring and flagging tool, not a clinical decision support system. It quantifies a known problem and provides an actionable framework for identifying where guideline implementation efforts should be prioritized.

### 4.3 The European Reference as Guideline Calibration Baseline

Our use of the European-ancestry metabolizer distribution as the reference against which equity gaps are measured is not a biological or clinical value judgment. It reflects the empirical reality that CPIC, DPWG, and most national PGx guidelines were developed and validated predominantly in European-majority cohorts. The "gap" we measure is the distance between a population's metabolizer profile and the profile for which current dosing recommendations were optimized. If guidelines were recalibrated for each ancestry group independently, this reference would become unnecessary. Until that recalibration occurs, the European reference serves as a measure of current guideline fit, not of biological normalcy.

### 4.4 Limitations

We detail the limitations of this work thoroughly, as honest characterization of uncertainty is essential for tools that may inform clinical and policy decisions.

**IM formula correction.** An earlier version of the HWE phenotype derivation omitted the 2 * *f*_df * *f*_n term from the intermediate metabolizer formula, causing IM frequencies to be underestimated by approximately 10-15 percentage points for genes with significant decreased-function allele burden (e.g., CYP2D6 in Europeans where *41 reaches 9%). This was identified and corrected prior to the analyses presented in this paper. All results reflect the corrected formula.

**Hardy-Weinberg assumptions.** HWE assumes random mating within defined populations---a condition violated by all real human populations and especially by our continental-scale ancestry groupings. The admixed population modeling (Section 2.7) partially addresses this limitation for four specific populations, but uses linear allele blending which does not account for the Wahlund effect from recent admixture, linkage disequilibrium between loci, or population substructure within admixed groups. Continuous ancestry modeling using genetic principal components would be preferable but requires individual-level genotype data that is not available at population scale for most pharmacogenes.

**HWE-derived phenotype accuracy.** Comparison against empirically measured phenotype frequencies (Section 3.8) indicates that HWE-derived PM frequencies are systematically underestimated by 30-50% for CYP2D6 (due to missing no-function alleles *3, *5, *6) and by 40-60% for CYP2C19 in East Asians (due to missing *3). The IM/NM boundary is distorted by the use of functional-bin aggregation rather than CPIC activity score-based phenotype assignment. However, directional equity conclusions and relative population rankings are preserved (Section 3.8), and the underestimates are conservative---the true equity gaps are likely larger than reported.

**Allele coverage.** We include 16 alleles across 6 genes. CYP2D6 alone has over 150 defined star alleles in PharmVar. Population-specific alleles of clinical significance are underrepresented in our dataset. CYP2C9*8 (8% in African populations) and *11 (3% in African populations) are decreased-function alleles not yet included; their absence systematically underestimates the equity gap for warfarin and phenytoin in African-ancestry patients [23]. CYP2D6*29 (decreased function, primarily African) and CYP2D6*10-like suballeles in East Asian populations are similarly absent. Most consequentially, the absence of CYP2C19*3 (rs4986893)---a no-function allele present at 5-9% in East Asian populations but rare in Europeans---approximately halves our East Asian CYP2C19 PM estimate (8.4% vs. empirical 13-23%). This specific gap means our engine underestimates the clopidogrel and voriconazole equity burden for East Asian patients. Expanding allele coverage, particularly for under-studied populations, is a priority for future versions.

**Risk weight heuristics.** The risk weights used in equity gap scoring (0.8, 0.3, 0.9, 0.7, 0.2) are pharmacologically motivated but not derived from clinical outcome data. The relative contribution of PM versus UM phenotypes to adverse events and efficacy failure varies by drug, dose, indication, and patient factors that our model does not capture. Sensitivity analysis (Section 3.6) demonstrates that equity gap rankings are robust to moderate weight perturbations, but the absolute magnitude of equity gap scores should not be interpreted as calibrated risk probabilities.

**No clinical outcome validation.** All risk estimates in this analysis are derived from allele frequency data and pharmacokinetic principles. We have not validated our equity gap scores against observed rates of adverse drug reactions or therapeutic failure in ancestry-stratified clinical cohorts. Prospective clinical validation, ideally using data from All of Us, UK Biobank, or H3Africa, is essential before these scores are used to guide clinical policy.

**Gene-gene interactions.** Each gene is modeled independently. In practice, many drugs are metabolized by multiple CYP enzymes: sertraline is primarily a CYP2C19 substrate but is also metabolized by CYP2D6 and CYP3A4. Our model does not capture the compensatory metabolism that may mitigate the clinical impact of a single-gene poor metabolizer phenotype.

**Ancestry self-identification.** In clinical practice, patient ancestry is typically self-reported and may not correspond to genetic ancestry. The NIH All of Us program has demonstrated substantial discordance between self-reported race/ethnicity and genetic ancestry for pharmacogene variation [12]. Our tool assumes concordance between self-reported ancestry and the population-level allele frequencies used for scoring. The admixed modeling capability partially mitigates this limitation for populations where ancestry proportions are well characterized.

**Sample size limitations.** Allele frequency estimates for Native American and some South Asian sub-populations are derived from small cohorts with wide confidence intervals. These values should be treated as approximate.

### 4.5 Policy Recommendations

Our analysis supports four policy recommendations.

First, **CPIC guideline development should incorporate systematic equity impact assessment**. For each new or updated guideline, the development process should include a quantitative analysis of how metabolizer phenotype distributions vary across ancestry groups and whether the guideline's dosing recommendations adequately serve populations beyond the primary validation cohort. The methodology presented here provides a template for such assessment.

Second, **hospital PGx programs should report ancestry-stratified phenotype distributions**. Institutions implementing clinical PGx programs should track and report the ancestry distribution of their patient populations alongside phenotype frequencies, enabling comparison with guideline calibration populations and identification of local equity gaps.

Third, **pharmacovigilance databases should stratify adverse event reporting by genetic ancestry**. Current adverse event reporting systems (FDA FAERS, WHO VigiBase) do not systematically capture genetic ancestry. Without this information, ancestry-specific adverse event signals---such as elevated efavirenz CNS toxicity in CYP2B6 PM-enriched populations---are difficult to detect at population scale.

Fourth, **open-source equity tools should be prioritized for global adoption**. The 2025 scoping review finding that all 40 published PGx tools originated in high-income countries [6] underscores a development gap. Open-source tools with permissive licensing lower the barrier for adaptation and deployment in low- and middle-income countries, where the burden of ancestry-biased prescribing may be greatest and resources for local tool development are most constrained.

### 4.6 Future Work

Near-term priorities include expanding allele coverage to include African-specific variants (CYP2C9*8, *11; CYP2D6*29), extending gene coverage to all 28 active CPIC guidelines, and refining admixed modeling with region-specific ancestry proportions and empirical validation. Adopting CPIC activity score-based phenotype assignment (replacing the four-bin HWE formula) would eliminate the IM/NM boundary distortion documented in Section 3.8. Validation against All of Us clinical outcome data would enable calibration of risk weights against observed adverse event rates. Integration with PharmCAT could provide a seamless individual-to-population context bridge, where individual genotype results are presented alongside population-level equity context. Extension to DPWG (Dutch Pharmacogenetics Working Group) guidelines would enable European-focused comparison. Finally, a web-based interface would make the tool accessible to clinicians, pharmacists, and policymakers who may not have Rust development environments.

---

## 5. Conclusion

Pharmacogenomic guidelines are among the most evidence-based tools in precision medicine, yet their predominantly European calibration creates a quantifiable equity gap. We present the Pharmacogenomic Health Equity Engine, an open-source tool that makes this gap visible and actionable. Across 11 CPIC Level A drug-gene pairs, 6 pharmacogenes, and 6 ancestry groups, we find systematic disparities in metabolizer phenotype distributions that translate into differential adverse event risk and therapeutic efficacy. The most severe disparities concentrate in transplant medicine (tacrolimus/CYP3A5, where 82% of African individuals carry expressor phenotypes versus 6% of Europeans), HIV treatment (efavirenz/CYP2B6, where PM frequency in African populations is 2.6 times the European estimate), and antiplatelet therapy (clopidogrel/CYP2C19, where East Asian PM frequency is 3.8 times the European reference). Across these and other drug-gene pairs, tens of millions of individuals in the US alone carry actionable phenotypes where current guidelines may not optimally serve their ancestry group.

External validation against gnomAD v4 confirms the accuracy of our allele frequency inputs (81% of comparisons within 30% relative agreement), and sensitivity analysis demonstrates perfect rank stability of equity conclusions across +/-20% allele frequency perturbations. Comparison against empirical phenotype frequencies suggests that our HWE-derived estimates are conservative: the true equity gaps are likely larger than reported, as missing alleles systematically underestimate PM frequencies in underrepresented populations. Extension to four admixed populations indicates that admixed-specific dosing considerations fall outside the range addressable by any single continental ancestry bin.

This tool does not replace CPIC guidelines. It provides an equity lens through which to evaluate them. All code and reference data are released openly under AGPL-3.0. Systematic equity quantification should become a standard component of pharmacogenomic guideline development and implementation.

---

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

**Figure 2.** Equity gap heatmap. Rows: 11 drug-gene pairs. Columns: 6 ancestry groups. Cell color: composite risk score (mean of adverse event risk and efficacy risk). Asterisks mark populations flagged as underserved. Right margin shows aggregate equity gap score per drug-gene pair.

**Figure 3.** Clinical impact estimates (US population). Horizontal bar chart showing estimated number of individuals carrying actionable pharmacogenomic phenotypes per drug-gene pair, stratified by ancestry group. Prominently labeled: "Modeled estimates based on HWE-derived phenotype frequencies and US Census population data. Not epidemiological measurements."

**Figure 4.** Admixed population phenotype distributions for CYP3A5 and CYP2C19. Side-by-side comparison of metabolizer frequencies for four admixed populations (African American, US Latino/Hispanic, Brazilian, South African Coloured) alongside ancestral reference populations. Illustrates how admixed phenotype frequencies are intermediate between---but not simple linear interpolations of---ancestral values.

---

## Supplementary Material

**Table S1.** Complete allele frequency table with literature ranges, confidence intervals where available, and source annotations for each value. Includes documentation of values interpolated from limited data (Native American, Mixed categories).

**Table S2.** Sensitivity analysis of risk weights. Equity gap scores recomputed with risk weight perturbations of +/-10%, +/-20%, and +/-50%. Ranking stability assessed using Spearman rank correlation.

**Table S3.** Comparison of HWE-derived phenotype frequencies with directly measured values from All of Us (V8), UK Biobank, and published cohort studies, where available.

**Table S4.** gnomAD v4.1 validation: per-allele, per-population comparison table with relative errors, pass/fail status, and explanatory notes for divergences. Includes gnomAD query methodology and population mapping.

**Table S5.** HWE vs empirical phenotype frequency comparison for CYP2D6 and CYP2C19 across 4 ancestry groups, with root cause analysis of systematic biases.

**Code availability.** All source code, reference data, and test suites are available at [repository URL] under the AGPL-3.0 license, with a Zenodo DOI for the archived release.

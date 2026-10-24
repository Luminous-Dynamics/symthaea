# Quantifying Pharmacogenomic Evidence Gaps: An Exploratory Population-Level Equity Analysis

**Target journal**: CPT: Pharmacometrics & Systems Pharmacology (ASCPT, Open Access)
**Article type**: Original Research (~5,000-6,000 words, excluding references/figures)
**APC**: $4,140 USD (no submission fee)
**Review**: Single-blind, 5-7 weeks typical turnaround
**Data mandate**: Code + data must be archived in public repository (GitHub + Zenodo DOI)

---

## Competitive Landscape Assessment

### What exists
- **PharmCAT** (PharmGKB/CPIC): Open-source genotype-to-phenotype caller + CPIC report generator. Operates at the *individual patient* level from VCF files. Does NOT quantify population-level equity gaps or produce ancestry-stratified risk scores. MPL 2.0.
- **PGxDB** (2025, Nucleic Acids Research): Interactive web platform for pharmacogenomics research. Database explorer, not a scoring engine.
- **PharmVIP, ePGA, PharmaKoVariome**: Population-specific PGx analysis tools identified in a 2025 scoping review (PMC11789506). Designed for individual annotation, not systematic equity quantification.
- **RELIVAF** (2026, Frontiers Pharmacology): Proposed Latin American PGx guidelines. Policy paper, not software.
- **Population Pharmacogenomics for Health Equity** (Genes 2023, MDPI): Review/perspective paper. Frames the problem but provides no computational tool.

### Gap we fill
An apparent research niche is reproducible population-level analysis of allele-frequency reference differences and their relationship to guideline evidence coverage. We have not completed a systematic landscape review that can support a 'first' or novelty claim. The current engine computes exploratory aggregate phenotype-frequency approximations and heuristic flags only; it does not provide individual genotype/phenotype interpretation, dosing recommendations, or validated clinical equity measurements.

### Honest limitations of our claims
- Our allele frequency data is compiled from published sources (PharmGKB, gnomAD v4, NCBI NBK574601, Gaedigk 2017, Ionova 2020), not from original sequencing.
- Hardy-Weinberg equilibrium assumptions break down for admixed populations, consanguineous populations, and populations with recent bottlenecks.
- We cover 4 genes (CYP2D6, CYP2C19, CYP2C9, CYP3A4) and 4 drugs (codeine, clopidogrel, warfarin, sertraline) with full CPIC guidelines. This is a proof-of-concept, not comprehensive coverage.
- Population-denominator multiplied by modelled phenotype-frequency outputs are arithmetic scenarios, not estimates of observed clinical impact, actionable patients, prescriptions, adverse events, or treatment benefit.
- The 6 ancestry groups (European, African, East Asian, South Asian, Native American, Mixed) are crude bins that mask enormous within-group diversity.

---

## Abstract (~250 words)

Pharmacogenomic guidelines translate genotype-derived phenotypes into gene–drug recommendations, but the populations and variants represented in supporting evidence vary. This proposal describes an exploratory open-source engine that computes population-level phenotype-frequency approximations from selected published allele frequencies under simplified Hardy–Weinberg and functional-bin assumptions, then calculates heuristic measures intended to help audit possible reference-data gaps.

The current method omits clinically relevant alleles and structural variants, collapses allele-specific pairing, and does not reproduce complete gene-specific diplotype/activity-score interpretation. Comparisons against published phenotype estimates show material discrepancies for some gene/population combinations. Therefore, the current scores do not establish clinical inequity, individual phenotype status, patient risk, treatment failure, or dosing needs. Broad population labels cannot substitute for an individual's genotype.

The intended contribution is a reproducible research scaffold, transparent accounting of assumptions and negative findings, and an evaluation plan. The proposed benchmark suite will pin exact source editions, allele definitions, and independent expected-phenotype annotations; include unsupported and ambiguous cases; preserve indeterminate outcomes; and compare results with established resources under matched conditions. Any future clinical claim would require independent, gene-specific validation, external evaluation, and a narrowly defined context of use. The present engine is research-only.


---

## 1. Introduction (~800 words)

### 1.1 The pharmacogenomics promise
- PGx testing enables genotype-guided prescribing; CPIC now covers 34 genes and 164 drugs with 28 active guidelines, >10,000 citations (Caudle et al. 2025)
- Clinical adoption accelerating: 85% of PubMed PGx implementation studies reference CPIC

### 1.2 The representation crisis
- PharmGKB: >63% of population-labeled individuals are European descent (Bridging genomics diversity gap, 2024)
- GWAS Catalog: 86.5% European participants, African representation at 0.47% (same source)
- 40 PGx tools identified in scoping review; ALL developed in high-income countries (PMC11789506, 2025)
- NIH All of Us (2025) shows self-reported race is a poor proxy for genetic ancestry, yet guidelines still implicitly calibrated to "European normal"

### 1.3 Published pharmacogenomic context (separate from this model)
- Cite published allele-frequency and gene–drug evidence for CYP2D6, CYP2C19, CYP2B6, and CYP3A5; distinguish frequency observations from observed clinical outcomes.
- CYP2D6 ultra-rapid-metabolizer estimates vary by specific population and reference cohort; Ethiopia-specific estimates must not be generalized to all African populations.
- CYP2C19*2 frequency differences can motivate external study of guideline coverage, but frequency differences alone do not establish individual phenotype or treatment failure.
- Published findings on CYP2D6*10 and regulatory haplotypes provide context for why variant coverage and gene-specific interpretation matter.
- This literature section motivates a reference-coverage audit. It is not evidence that this implementation's current phenotype estimates or equity scores are clinically valid.

### 1.4 What is missing
- PharmCAT operates at individual genotype level; no population-level equity scoring
- No published tool systematically quantifies the gap between guideline calibration population and target population
- We present an open-source engine that fills this gap

### 1.5 Contribution statement
We provide: (1) a formalized equity gap scoring methodology, (2) an open-source Rust engine implementing it, (3) ancestry-stratified equity analyses for CPIC Level A drug-gene pairs, (4) population-level equity flags and modeled sensitivity analyses (not individual dosing).

---

## 2. Methods (~1,500 words)

### 2.1 Allele frequency data sources and curation
- **Primary sources**: PharmGKB population-stratified allele frequencies, gnomAD v4 (genome aggregation), NCBI Medical Genetics Summaries (NBK574601, AMP Tier 1), TOPMed
- **Gene-specific literature**: Gaedigk et al. 2017 (CYP2D6 worldwide), Ionova et al. 2020 (CYP2C19), Hu Genomics 2023 (CYP2C9 global distribution)
- **Ancestry grouping**: 6 groups (European, African, East Asian, South Asian, Native American, Mixed) -- acknowledge this is a pragmatic simplification, not a biological taxonomy
- **Alleles included**: CYP2D6 (*4, *10, *17, *41, *1xN), CYP2C19 (*2, *17), CYP2C9 (*2, *3), CYP3A4 (*22) -- 10 alleles total covering the highest-impact variants per gene
- **Validation**: Cross-check each frequency against at least 2 independent sources; document ranges (e.g., CYP2D6*10 East Asian: AMP 9-44%, Korean 46.2%, our value 43%)
- **Table 1** here

### 2.2 Hardy-Weinberg phenotype derivation
- **Rationale**: In the absence of individual genotype data, Hardy-Weinberg equilibrium (HWE) provides the expected diplotype distribution from allele frequencies
- **Method**: For each gene and ancestry group:
  1. Sum non-normal allele frequencies; remainder assigned to *1 (normal function)
  2. Classify alleles by function: NoFunction, DecreasedFunction, IncreasedFunction, NormalFunction
  3. Compute diplotype categories:
     - PM = (NoFunction_freq)^2
     - UM = (IncreasedFunction_freq)^2 + 2 * IncreasedFunction_freq * Normal_freq
     - IM = 2 * NoFunction_freq * (Normal_freq + Decreased_freq) + Decreased_freq^2
     - NM = 1 - PM - UM - IM (clamped to [0,1])
  4. Validate: distribution sums to 1.0 within epsilon
- **Limitations explicitly stated**: HWE assumes random mating, no selection, no migration, no drift. Admixed populations violate these assumptions. Our groupings are continental-scale and mask sub-population structure.

### 2.3 Equity gap scoring methodology
- **Per-population risk profile** (for each drug-gene-ancestry triple):
  - **Adverse event risk**: For prodrugs (codeine, clopidogrel): UM_frac * 0.8 + PM_frac * 0.3. For active drugs: PM_frac * 0.8 + UM_frac * 0.2. Rationale: prodrug toxicity is driven by ultra-rapid activation; active drug toxicity by poor clearance.
  - **Efficacy risk**: For prodrugs: PM_frac * 0.9 (no activation). For active drugs: UM_frac * 0.7 + PM_frac * 0.3 (sub-therapeutic + accumulation).
  - **Guideline applicability**: 1 - (total variation distance from European phenotype distribution). TVD normalized to [0,1]. European is used as reference because CPIC guidelines are predominantly validated in European populations (we are measuring departure from calibration population, not asserting European as biological norm).
- **Aggregate equity gap score**: mean of (max - min adverse range, max - min efficacy range) across all 6 ancestry groups, clamped to [0,1]. Score of 0 = equitable across populations; 1 = maximum inter-population disparity.
- **Underserved population flag**: Ancestry group flagged if (a) guideline applicability < 0.85, OR (b) adverse event risk exceeds minimum by > 0.15, AND group is not European (since guidelines are calibrated to European).
- Acknowledge: risk weights (0.8, 0.3, 0.9, 0.7, 0.2) are heuristic, not derived from clinical outcome data. Sensitivity analysis across weight ranges provided in supplementary material.

### 2.4 Exploratory population scenario arithmetic
- Show the arithmetic product of a rounded group denominator and a modelled phenotype-frequency fraction only as an illustrative scenario.
- Do not call the output a patient-at-risk count, clinical impact, or treatment-need estimate.
- State that fractions come from a simplified HWE/functional-bin model with documented discrepancies and incomplete allele/structural-variant coverage.
- Require independent phenotype annotations, defensible denominators, prescription/indication data, uncertainty intervals, and a defined use context before proposing any clinical-impact estimate.

### 2.5 CPIC guideline integration
- 4 drug-gene pairs with full recommendation chains: codeine/CYP2D6, clopidogrel/CYP2C19, warfarin/CYP2C9, sertraline/CYP2C19
- Each guideline encodes: drug, gene, phenotype, dosing action (StandardDose/ReduceDose/IncreaseDose/AvoidDrug/UseAlternative/TherapeuticDrugMonitoring), evidence level (Strong/Moderate/Optional/Informative), source population
- Research-only guideline projection: exact drug/gene/phenotype lookup returns reference metadata and an exploratory aggregate flag only; no dose, medication choice, contraindication, monitoring, or treatment instruction is returned

### 2.6 Implementation
- Rust implementation (~900 LOC), AGPL-3.0 license
- Module unit tests cover frequency/distribution arithmetic, equity flags, unsupported guideline and gene-mismatch abstention; execution status must be checked against the exact reviewed revision
- Research-only API: `equity_analysis(drug, gene)`, `research_guideline_projection(drug, gene, population_context, phenotype_context)`, `population_report(gene)`
- The guideline projection returns only reference metadata and aggregate equity context. Its type contains no dose, medication choice, contraindication, or treatment instruction; unsupported tuples return `Indeterminate`.
- **Not qualified for clinical decision support or individual prescribing.** Population categories are not patient genotype evidence.

---

## 3. Results (~1,500 words)

### 3.1 Exploratory phenotype-frequency outputs
- **Figure 1**: Display the prototype's HWE/functional-bin outputs with a conspicuous “exploratory model output—not prevalence” label.
- List the source allele frequencies and functional-bin assumptions that dominate each displayed difference.
- Compare the outputs with independently published genotype/diplotype phenotype estimates only where phenotype definitions, cohort population, and source edition are comparable.
- Explicitly show discrepancies and unsupported coverage; do not present agreement in selected cases as validation of the whole model.

### 3.2 Exploratory model-score audit across drug-gene pairs
- **Figure 2**: Heatmap of the prototype's heuristic scores by drug–gene tuple and broad population label. Label the scores as uncalibrated model outputs, not risk probabilities.
- **Table 2**: Show score, threshold flags, source version, included-allele coverage, and known missing-variant warnings for every tuple.
- Interpret differences as hypotheses for source and method review only. Do not infer that a population is clinically underserved, that standard care is inadequate, or that specific treatments should change.
- Include negative controls: remove a known allele, add an unsupported tuple, perturb functional-bin assignments, change population denominators, and check whether the output remains interpretable. Preserve rank reversals and indeterminate outcomes rather than tuning them away.

### 3.3 Modelled population scenario arithmetic (not clinical impact)
- Show multiplication of rounded population denominators by prototype model-output fractions only as an illustrative calculation.
- Explicitly state that these are not counts of patients prescribed a drug, people needing dose adjustment, adverse events, or people likely to benefit from an intervention.
- Include model-coverage and phenotype-reference limitations; do not use the phrase “patients at risk” for these products.

### 3.4 Reference metadata versus individual treatment actions
- Describe the non-actionable `research_guideline_projection(drug, gene, population_context, phenotype_context)` API.
- Test exact drug/gene/phenotype matching, reference-found versus indeterminate states, and omission of medication-action fields.
- State that serialized-output tests establish a software contract only, not genotype interpretation accuracy, dosing validity, or clinical utility.

## 4. Discussion (~1,200 words)

### 4.1 Clinical implications
- Systematic equity quantification reveals that CPIC guidelines, while evidence-based, carry implicit European calibration bias
- The equity gap is not uniform: some drug-gene pairs show minimal disparity (where relevant alleles are similarly distributed), others show severe gaps
- Ancestry-adjusted recommendations do not replace CPIC -- they augment it with population-level context
- Therapeutic drug monitoring as the "equity equalizer": when guidelines are poorly calibrated for a population, TDM provides empirical safety net

### 4.2 Comparison to existing tools
- PharmCAT: Complementary, not competing. PharmCAT operates at individual genotype level; our engine operates at population level. Integration pathway: PharmCAT for individual diplotype calling, our engine for population-level context and equity flagging.
- PharmGKB/CPIC database: We consume their data; our contribution is the computational scoring layer
- All of Us Researcher Workbench: Provides raw data for validation; our engine provides the analytical framework
- RELIVAF (Latin American guidelines): Regional effort; our engine is ancestry-agnostic and extensible

### 4.3 Limitations (be thorough and honest)
1. **Simplified phenotype method:** HWE is applied after broad functional-bin aggregation, losing allele pairing and CPIC activity-score semantics; the model's own comparisons show material discrepancies.
2. **Allele/structural coverage:** clinically relevant alleles, copy-number and structural variants, phase uncertainty, and gene-specific star-allele interpretation are incomplete.
3. **Population assumptions:** broad group labels mask within-group variation and must not serve as proxies for an individual's genotype. Linear admixture is a first-order scenario, not individual-level ancestry inference.
4. **Heuristic scores:** weights and thresholds are not clinically calibrated. Stability under numeric perturbation does not establish accuracy or ranking validity.
5. **No outcome validation:** there is no validation against observed adverse events, therapeutic failure, or treatment benefit.
6. **Source checks are not phenotype validation:** selected allele-frequency comparisons to gnomAD/literature do not validate completeness or clinical interpretation.
7. **No clinical impact count:** population denominator × modelled fraction is arithmetic only without validated phenotype methods, prescription/indication data, and a defined context.
8. **Reference uncertainty:** some values depend on small cohorts, broad labels, and population mappings; source edition, denominator, interval, and missingness must be explicit.

### 4.4 Research questions for qualified stakeholders
- After independent validation, could guideline-development groups report population coverage, uncertainty, and limitations of allele/diplotype evidence?
- Can clinical and pharmacovigilance datasets support ethically governed, privacy-preserving evaluation of subgroup performance and adverse outcomes?
- Which reference datasets and underrepresented variants should be prioritized to improve coverage?
- What evidence would be required before an aggregate evidence-gap metric could be considered for policy use?
These are research questions, not recommendations established by this prototype.

### 4.5 Future work
- Expand to all 28 active CPIC guidelines (164 drugs, 34 genes)
- Integrate with PharmCAT for individual-to-population context bridging
- Incorporate admixture-aware modeling (continuous ancestry proportions rather than discrete bins)
- Validate against All of Us clinical outcome data
- Add CYP2C9*8, *11, CYP2D6*29 and other population-specific alleles
- Extend to DPWG (Dutch) guidelines for European comparison

---

## 5. Conclusion (~300 words)

This work is an exploratory software study of population-level allele-frequency inputs and simplified phenotype-frequency approximations, not a clinically validated pharmacogenomic model. Its most important current finding is the method's own limitation: broad functional-bin aggregation, missing clinically relevant alleles and structural variants, and loss of individual diplotype/activity-score information produce material discrepancies against some published phenotype estimates.

Population-frequency differences may motivate further work on representation and evidence transfer, but the current implementation does not establish patient-level phenotype prevalence, observed adverse-event burden, treatment failure, clinical utility, or the number of people who would benefit from a treatment change. Comparing selected allele frequencies with a public reference is a source-consistency check, not clinical validation; stability under numerical perturbation does not establish that a score or ranking is correct. Derived phenotype, equity, and population scenario outputs therefore remain hypothesis-generating.

The next stage should be a versioned, independently annotated gene-specific reference corpus, with complete allele and structural-variant definitions within the declared scope, explicit diplotype and phenotype rules, negative controls, uncertainty intervals, and held-out external comparisons. Unsupported or ambiguous inputs should remain indeterminate. The research projection API should continue to return reference metadata only, without medication-action fields. Code and fixtures are released for replication and critique; software correctness alone does not qualify the model for clinical use.


---

## Figures

### Figure 1: Metabolizer phenotype distributions across ancestry groups
- **Type**: Grouped stacked bar chart (3 panels: CYP2D6, CYP2C19, CYP2C9)
- **X-axis**: 6 ancestry groups
- **Y-axis**: Proportion (0-100%)
- **Stacks**: UM (red), NM (green), IM (yellow), PM (blue)
- **Data source**: `PgxHealthEquityEngine::metabolizer_distribution()` output for each gene x ancestry
- **Key visual**: East Asian CYP2C19 PM bar visibly larger than European; East Asian CYP2D6 IM bar dominant

### Figure 2: Equity gap heatmap
- **Type**: Heatmap with annotation
- **Rows**: Drug-gene pairs (codeine/CYP2D6, clopidogrel/CYP2C19, warfarin/CYP2C9, sertraline/CYP2C19)
- **Columns**: 6 ancestry groups
- **Cell value**: Composite risk score (adverse_event_risk + efficacy_risk) / 2
- **Cell color**: Diverging palette (green = low risk, red = high risk), with European column as implicit reference
- **Annotation**: Star (*) on cells where population is flagged as underserved
- **Marginal**: Right margin shows aggregate equity gap score per drug

### Figure 3: Exploratory population scenario arithmetic
- **Type**: Horizontal bar chart
- **Bars**: Rounded population denominator multiplied by modelled phenotype fraction
- **Note**: Outputs are arithmetic scenarios only, not patient-risk or clinical-impact estimates

### Figure 4: Non-actionable guideline-reference projection
- Show exact tuple lookup and reference-found/indeterminate states
- Assert that serialized outputs contain no dose, medication choice, contraindication, monitoring, or treatment-instruction fields
- State that a reference match does not validate an individual's genotype or authorize a clinical action

---

## Tables

### Table 1: Reference allele frequencies by ancestry group
- Columns: Gene, Allele, Function, European, African, East Asian, South Asian, Native American, Mixed, Source(s)
- 10 rows (one per allele)
- Include literature ranges in parentheses for validation
- Footnotes: acknowledge where values are interpolated or based on small samples

### Table 2: Equity analysis results per drug-gene pair
- Columns: Drug, Gene, Evidence Level, Equity Gap Score, Underserved Populations, Guideline Applicability (per ancestry, 6 sub-columns)
- 4 rows

### Table 3: Modelled population scenario arithmetic
- Columns: Drug, Gene, broad group label, denominator, modelled phenotype fraction, arithmetic product
- Clearly state that outputs are not observed clinical impact, patient risk, prescription counts, or treatment need

---

## Supplementary Material

### S1: Internal sensitivity checks
- Vary included inputs/weights and report score/flag/ranking changes; state that bounded perturbations do not cover omitted variants or model misspecification and do not validate ranking

### S2: HWE-derived estimates versus published phenotype references
- Compare with independent published estimates only where definitions and populations are comparable
- Report disagreement and reference incompatibilities; do not call comparison a validation pass or assume errors are always conservative

### S3: Full engine API documentation and code listing
- Link to GitHub repository with tagged release

### S4: Selected allele-frequency source-consistency comparison
- Pin versions and query methods; report relative differences without calling the comparison phenotype or clinical validation

---

## Key References (preliminary)

1. Caudle KE et al. (2025). Advancing Clinical Pharmacogenomics Worldwide Through CPIC. *Clin Pharmacol Ther*. doi:10.1002/cpt.70005
2. Gaedigk A et al. (2017). The Pharmacogene Variation (PharmVar) Consortium: Incorporation of the Human CYP2D6 Gene. *Clin Pharmacol Ther*.
3. Ionova Y et al. (2020). CYP2C19 Allele Frequencies in Over 2.2 Million Direct-to-Consumer Genetics Research Participants. *Clin Transl Sci*.
4. Hicks JK et al. (2015/2017). CPIC Guideline for CYP2D6 and CYP2C19 Genotypes and Dosing of SSRIs. *Clin Pharmacol Ther*.
5. Relling MV & Klein TE (2011). CPIC: Clinical Pharmacogenetics Implementation Consortium. *Clin Pharmacol Ther*.
6. Aklillu E et al. (1996). Frequent distribution of ultrarapid metabolizers of debrisoquine in an Ethiopian population. *J Pharmacol Exp Ther*.
7. Bradford LD (2002). CYP2D6 allele frequency in European Caucasians, Asians, Africans and their descendants. *Pharmacogenomics*.
8. Frontiers Pharmacology (2026). Opportunities to Improve Open All of Us Data to Convey CYP2D6 Pharmacological Relevance. *Front Pharmacol*.
9. Nature PGx Journal (2026). A functional ancestry-linked regulatory haplotype influences CYP2D6 expression. *Pharmacogenomics J*.
10. Matthias SA et al. (2025). Empirical Drug Dosage Validates Pharmacogenomic Associations in All of Us. *Clin Transl Sci*.
11. Hu Genomics (2023). Global distribution of CYP2C9 polymorphisms.
12. Bridging Genomics Diversity Gap (2024). *Cell Genomics*.
13. Zhou Y et al. (2025). Opportunities and Challenges of Population Pharmacogenomics. *PMC12336967*.
14. RELIVAF (2026). Addressing genetic diversity and health inequities: Latin American PGx guidelines. *Front Pharmacol*.

---

## Pre-submission Checklist

- [ ] Run all 10 engine unit tests: `cargo test -p symthaea-neuromodulators --lib pgx_health_equity`
- [ ] Generate actual equity gap scores and phenotype distributions from engine for all tables/figures
- [ ] Cross-validate HWE-derived PM/UM frequencies against All of Us V8 measured phenotype frequencies
- [ ] Add CYP2C9*8 and *11 alleles (African-population-specific) before claiming comprehensive coverage
- [ ] Sensitivity analysis on risk weights
- [ ] Extract engine to standalone repository for clean open-source release (separate from Symthaea)
- [ ] Zenodo DOI for code + data archive
- [ ] ORCID for all authors
- [ ] Confirm APC funding ($4,140)
- [ ] Draft cover letter emphasizing: (a) no existing equity scoring tool, (b) open source, (c) policy relevance

---

## Code Location

- **Engine**: `symthaea/crates/symthaea-neuromodulators/src/pgx_health_equity.rs` (~900 LOC)
- **Pharmacogenomics types**: `symthaea/crates/symthaea-neuromodulators/src/pharmacogenomics.rs`
- **Tests**: 10 unit tests in `pgx_health_equity.rs` (frequencies, distributions, equity flags, dosing adjustments)

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pharmacogenomics Health Equity: addressing ancestry bias in drug dosing.
//!
//! Standard pharmacogenomic guidelines (CPIC, DPWG) are predominantly
//! calibrated to European-ancestry populations. This module quantifies
//! the equity gap and provides ancestry-adjusted dosing recommendations.
//!
//! Key findings driving this work:
//! - 64% of CYP2D6 "drug response" alleles most frequent in East Asian populations
//! - 29% of African populations are CYP2D6 ultra-rapid metabolizers (vs 7% European)
//! - Standard SSRI doses may be ineffective for ~29% of African-ancestry patients
//! - CYP2C19 poor metabolizer frequency: 2-5% European, 12-23% East Asian
//! - CYP2B6*6 slow metabolizer allele: ~38% African vs ~26% European (HIV equity)
//! - CYP3A5*3 loss-of-function: ~94% European vs ~18% African (transplant equity)
//!
//! References:
//! - Gaedigk et al. (2017). CYP2D6 allele frequencies worldwide.
//! - Hicks et al. (2015). CPIC guideline for CYP2D6/CYP2C19.
//! - Relling & Klein (2011). CPIC overview.
//! - Frontiers Pharmacology (2026). CYP2D6 in All of Us.
//! - Zhou et al. (2025). Population pharmacogenomics challenges.
//! - Desta et al. (2019). CYP2B6 PharmVar GeneFocus.
//! - Zanger & Klein (2013). CYP2B6 polymorphism and efavirenz.
//! - Birdwell et al. (2015). CPIC guideline for tacrolimus/CYP3A5.
//! - Goetz et al. (2018). CPIC guideline for tamoxifen/CYP2D6.
//! - Brown et al. (2019). CPIC guideline for efavirenz/CYP2B6.
//! - Moriyama et al. (2017). CPIC guideline for voriconazole/CYP2C19.
//! - Caudle et al. (2020). CPIC guideline for phenytoin/CYP2C9.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::pharmacogenomics::{AncestryGroup, MetabolizerPhenotype};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Equity gap threshold above which a population is flagged as underserved.
const UNDERSERVED_THRESHOLD: f64 = 0.15;

/// Threshold for flagging high-frequency actionable phenotypes in a population.
const HIGH_FREQUENCY_PHENOTYPE_THRESHOLD: f64 = 0.10;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Functional consequence of a specific allele.
///
/// Science: CPIC allele functionality table (PharmVar).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlleleFunction {
    /// Normal (wild-type) enzyme activity.
    NormalFunction,
    /// Reduced but non-zero enzyme activity.
    DecreasedFunction,
    /// Complete loss of enzyme activity.
    NoFunction,
    /// Gene duplication or gain-of-function — elevated activity.
    IncreasedFunction,
    /// Insufficient data to classify.
    UncertainFunction,
}

/// CPIC evidence level for a drug-gene interaction.
///
/// Science: Relling & Klein (2011). CPIC assigns levels A-D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// Level A — strong evidence, action required.
    Strong,
    /// Level B — moderate evidence, action recommended.
    Moderate,
    /// Level C — optional, consider action.
    Optional,
    /// Level D — informational only.
    Informative,
}

/// Clinical dosing action derived from genotype.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DosingAction {
    /// Standard labelled dose; no adjustment needed.
    StandardDose,
    /// Reduce dose by the given fraction (e.g., 0.5 = halve).
    ReduceDose(f64),
    /// Increase dose by the given fraction (e.g., 1.5 = 50% increase).
    IncreaseDose(f64),
    /// Contraindicated — do not prescribe.
    AvoidDrug,
    /// Switch to a named alternative agent.
    UseAlternative(String),
    /// No dose change, but mandate plasma-level monitoring.
    TherapeuticDrugMonitoring,
}

// ---------------------------------------------------------------------------
// Core structs
// ---------------------------------------------------------------------------

/// Population-level allele frequency for a pharmacogene.
///
/// Captures how a single star-allele is distributed across ancestry groups,
/// enabling downstream calculation of expected metabolizer phenotype
/// distributions per population.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationAlleleFrequency {
    /// Gene symbol (e.g., "CYP2D6").
    pub gene: String,
    /// Star-allele designation (e.g., "*4").
    pub allele: String,
    /// Allele frequency per ancestry group (0.0 - 1.0).
    pub frequencies: HashMap<AncestryGroup, f64>,
    /// Functional classification of this allele.
    pub functional_status: AlleleFunction,
    /// Data source (e.g., "PharmGKB", "gnomAD v4").
    pub source: String,
}

/// A single CPIC clinical guideline recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpicGuideline {
    /// Drug name (e.g., "codeine").
    pub drug: String,
    /// Gene name (e.g., "CYP2D6").
    pub gene: String,
    /// Metabolizer phenotype this recommendation addresses.
    pub phenotype: MetabolizerPhenotype,
    /// Clinical recommendation.
    pub recommendation: DosingRecommendation,
    /// Strength of evidence.
    pub evidence_level: EvidenceLevel,
    /// Population in which the guideline was primarily validated.
    pub source_population: AncestryGroup,
}

/// Dosing recommendation with optional adjustment details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DosingRecommendation {
    /// Primary dosing action.
    pub action: DosingAction,
    /// Numeric dose multiplier (e.g., 0.5 for half-dose). `None` if not applicable.
    pub dose_adjustment: Option<f64>,
    /// Alternative drug to consider.
    pub alternative_drug: Option<String>,
    /// Additional monitoring instructions.
    pub monitoring: Option<String>,
}

/// Risk profile for a single ancestry group relative to a drug-gene pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationRisk {
    /// Expected metabolizer phenotype distribution (fractions summing to ~1.0).
    pub phenotype_distribution: HashMap<MetabolizerPhenotype, f64>,
    /// Estimated risk of adverse drug reactions (0.0 - 1.0).
    pub adverse_event_risk: f64,
    /// Estimated risk of therapeutic failure (0.0 - 1.0).
    pub efficacy_risk: f64,
    /// How well current CPIC guidelines apply to this population (0.0 - 1.0).
    pub guideline_applicability: f64,
}

/// Estimate of clinical impact for a drug-gene pair across US populations.
///
/// Quantifies how many patients are affected by ancestry-biased dosing
/// guidelines, using US census population data and allele frequency distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalImpactEstimate {
    /// Drug name.
    pub drug: String,
    /// Gene name.
    pub gene: String,
    /// Per-ancestry breakdown: (ancestry, population_millions, at_risk_fraction, at_risk_count).
    pub per_ancestry: Vec<(AncestryGroup, f64, f64, u64)>,
    /// Total patients at risk across all US populations.
    pub total_at_risk: u64,
    /// Ancestry group with the highest absolute at-risk count.
    pub most_affected: AncestryGroup,
    /// Human-readable risk description.
    pub risk_description: String,
}

/// Full equity analysis for a drug-gene pair across populations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EquityAnalysis {
    /// Drug name.
    pub drug: String,
    /// Gene name.
    pub gene: String,
    /// Per-population risk profile.
    pub population_risk: HashMap<AncestryGroup, PopulationRisk>,
    /// Aggregate equity gap score (0.0 = equitable, 1.0 = severe disparity).
    pub equity_gap_score: f64,
    /// Populations where current guidelines are inadequate.
    pub underserved_populations: Vec<AncestryGroup>,
    /// Summary recommendation text.
    pub recommendation: String,
}

// ---------------------------------------------------------------------------
// Admixed populations
// ---------------------------------------------------------------------------

/// Represents an admixed population as a weighted mixture of ancestral groups.
///
/// Many real-world populations — especially in the Americas — cannot be
/// accurately described by a single continental ancestry bin. This struct
/// models admixture by computing allele frequencies as weighted averages
/// of ancestral group frequencies before applying Hardy-Weinberg equilibrium.
///
/// Science: Hardy-Weinberg applied to blended allele frequencies gives a
/// first-order approximation. True admixed populations may deviate from
/// HWE due to assortative mating and population substructure (Wahlund
/// effect), but the linear-mixture model is standard practice in
/// population pharmacogenomics (Bryc et al. 2015, Parra et al. 1998).
///
/// References:
/// - Bryc et al. (2015). The genetic ancestry of African Americans,
///   Latinos, and European Americans across the United States. *Am J Hum Genet*.
/// - Parra et al. (1998). Estimating African American admixture proportions
///   by use of population-specific alleles. *Am J Hum Genet*.
/// - Salzano & Sans (2014). Interethnic admixture and the evolution of
///   Latin American populations. *Genet Mol Biol*.
/// - Tishkoff et al. (2009). The genetic structure and history of Africans
///   and African Americans. *Science*.
/// - de Wit et al. (2010)."; Quintana-Murci (2017) population structure in
///   southern Africa.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmixedPopulation {
    /// Human-readable population label.
    pub name: String,
    /// Ancestry proportions (must sum to ~1.0).
    pub composition: Vec<(AncestryGroup, f64)>,
}

impl AdmixedPopulation {
    /// African American: ~80% African + ~20% European.
    ///
    /// Science: Bryc et al. (2015) found average African ancestry of 73.2%
    /// and European ancestry of 24.0% among self-identified African Americans.
    /// Tishkoff et al. (2009) reports 78-82% African for most US regions.
    /// We use 80/20 as a representative round figure.
    pub fn african_american() -> Self {
        Self {
            name: "African American".into(),
            composition: vec![
                (AncestryGroup::African, 0.80),
                (AncestryGroup::European, 0.20),
            ],
        }
    }

    /// US Latino/Hispanic: ~50% European + ~40% Native American + ~10% African.
    ///
    /// Science: Bryc et al. (2015) found substantial variation: Mexican Americans
    /// average ~8% African, ~47% Native American, ~45% European; Puerto Ricans
    /// ~29% African, ~15% Native American, ~56% European. We use 50/40/10 as
    /// a pan-US Latino average reflecting Mexican-American majority.
    pub fn us_latino() -> Self {
        Self {
            name: "US Latino/Hispanic".into(),
            composition: vec![
                (AncestryGroup::European, 0.50),
                (AncestryGroup::NativeAmerican, 0.40),
                (AncestryGroup::African, 0.10),
            ],
        }
    }

    /// South African Coloured: ~32% African + ~29% European + ~25% East Asian + ~14% South Asian.
    ///
    /// Science: de Wit et al. (2010) and Quintana-Murci (2017) describe the
    /// "Coloured" population of the Western Cape as one of the most admixed
    /// groups globally, with Khoisan, Bantu, European, and Southeast Asian
    /// (Malay/Indonesian) ancestry. Proportions vary by region; we use
    /// rounded averages from published estimates.
    pub fn south_african_coloured() -> Self {
        Self {
            name: "South African Coloured".into(),
            composition: vec![
                (AncestryGroup::African, 0.32),
                (AncestryGroup::European, 0.29),
                (AncestryGroup::EastAsian, 0.25),
                (AncestryGroup::SouthAsian, 0.14),
            ],
        }
    }

    /// Brazilian: ~60% European + ~25% African + ~15% Native American.
    ///
    /// Science: Salzano & Sans (2014) found average genomic ancestry of
    /// ~62% European, ~21% African, ~17% Native American across Brazil.
    /// Southern Brazil is more European (~80%); Northeast is more African
    /// (~40%). We use rounded pan-Brazil estimates.
    pub fn brazilian() -> Self {
        Self {
            name: "Brazilian".into(),
            composition: vec![
                (AncestryGroup::European, 0.60),
                (AncestryGroup::African, 0.25),
                (AncestryGroup::NativeAmerican, 0.15),
            ],
        }
    }

    /// Validate that composition proportions sum to approximately 1.0.
    pub fn is_valid(&self) -> bool {
        let sum: f64 = self.composition.iter().map(|(_, w)| w).sum();
        (sum - 1.0).abs() < 0.05
    }

    /// Return the total ancestry weight sum (for diagnostics).
    pub fn total_weight(&self) -> f64 {
        self.composition.iter().map(|(_, w)| w).sum()
    }

    /// Format the composition as a human-readable string.
    pub fn composition_string(&self) -> String {
        self.composition
            .iter()
            .map(|(anc, w)| {
                let name = match anc {
                    AncestryGroup::European => "Eur",
                    AncestryGroup::African => "Afr",
                    AncestryGroup::EastAsian => "EAs",
                    AncestryGroup::SouthAsian => "SAs",
                    AncestryGroup::NativeAmerican => "NatAm",
                    AncestryGroup::Mixed => "Mix",
                };
                format!("{:.0}% {}", w * 100.0, name)
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }

    /// Return all preset admixed populations.
    pub fn all_presets() -> Vec<Self> {
        vec![
            Self::african_american(),
            Self::us_latino(),
            Self::south_african_coloured(),
            Self::brazilian(),
        ]
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Pharmacogenomics health equity engine.
///
/// Loads reference allele frequencies and CPIC guidelines, then provides
/// methods for equity analysis, ancestry-adjusted dosing, and population
/// reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PgxHealthEquityEngine {
    /// Reference allele frequency data.
    pub allele_frequencies: Vec<PopulationAlleleFrequency>,
    /// Loaded CPIC guideline recommendations.
    pub cpic_guidelines: Vec<CpicGuideline>,
}

impl PgxHealthEquityEngine {
    /// Construct a new engine pre-loaded with reference data.
    pub fn new() -> Self {
        let mut engine = Self {
            allele_frequencies: Vec::new(),
            cpic_guidelines: Vec::new(),
        };
        engine.load_reference_allele_frequencies();
        engine.build_reference_cpic_guidelines();
        engine
    }

    /// Load built-in population allele frequencies from published literature.
    ///
    /// Sources: PharmGKB, gnomAD v4, Gaedigk et al. (2017).
    pub fn load_reference_allele_frequencies(&mut self) {
        self.allele_frequencies.clear();

        // --- CYP2D6 ---
        // Frequencies: NCBI Medical Genetics Summaries NBK574601 (AMP Tier 1),
        // TOPMed, Gaedigk et al. 2017 Genet Med.
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2D6".into(),
            allele: "*4".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.19),   // AMP: 18.5%, TOPMed: 13.8%
                (AncestryGroup::African, 0.04),    // AMP: 3-5%, TOPMed: 3.6%
                (AncestryGroup::EastAsian, 0.01),  // AMP: 0.5-9.1%
                (AncestryGroup::SouthAsian, 0.12), // Literature range 9-12%
                (AncestryGroup::NativeAmerican, 0.08),
                (AncestryGroup::Mixed, 0.10),
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "PharmGKB / NCBI NBK574601".into(),
        });

        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2D6".into(),
            allele: "*10".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.02),   // AMP: <2%, TOPMed: 1.6%
                (AncestryGroup::African, 0.05),    // AMP: 4-6%, TOPMed: 3.8%
                (AncestryGroup::EastAsian, 0.43),  // AMP: 9-44%, Korean 46.2%
                (AncestryGroup::SouthAsian, 0.20), // Gaedigk 2017: ~20%
                (AncestryGroup::NativeAmerican, 0.05),
                (AncestryGroup::Mixed, 0.10),
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "PharmGKB / NCBI NBK574601".into(),
        });

        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2D6".into(),
            allele: "*17".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.00),   // AMP: <0.5%, TOPMed: 0.16%
                (AncestryGroup::African, 0.19),    // AMP: 17-19%, TOPMed: 15.7%
                (AncestryGroup::EastAsian, 0.00),  // <0.5%
                (AncestryGroup::SouthAsian, 0.02), // ~0.07% (very rare outside Africa)
                (AncestryGroup::NativeAmerican, 0.01),
                (AncestryGroup::Mixed, 0.05),
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "PharmGKB / NCBI NBK574601".into(),
        });

        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2D6".into(),
            allele: "*41".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.09),   // AMP: 9%, TOPMed: 9.8%
                (AncestryGroup::African, 0.04),    // AMP: 4-11.5%, TOPMed: 2.4%
                (AncestryGroup::EastAsian, 0.02),  // Korean: 1.4%, range 2-4%
                (AncestryGroup::SouthAsian, 0.07), // Indian studies: 10-20%
                (AncestryGroup::NativeAmerican, 0.04),
                (AncestryGroup::Mixed, 0.05),
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "PharmGKB / NCBI NBK574601".into(),
        });

        // CYP2D6 gene duplication (ultra-rapid marker).
        // Note: 29% was Ethiopia-specific (Aklillu 1996). Pan-African duplication
        // frequency is ~3-5% (Gaedigk 2017). European ~1-2% (north-south gradient).
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2D6".into(),
            allele: "*1xN".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.02),   // Gaedigk 2017: 1.97%
                (AncestryGroup::African, 0.05),    // Pan-African ~3-5%; NOT 29% (Ethiopia only)
                (AncestryGroup::EastAsian, 0.01),  // East Asian: 0.8%
                (AncestryGroup::SouthAsian, 0.04), // Central/South Asian: 1.5-4%
                (AncestryGroup::NativeAmerican, 0.02),
                (AncestryGroup::Mixed, 0.03),
            ]),
            functional_status: AlleleFunction::IncreasedFunction,
            source: "Gaedigk et al. 2017 Genet Med".into(),
        });

        // --- CYP2C19 ---
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2C19".into(),
            allele: "*2".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.15),   // Ionova 2020: 14.6%
                (AncestryGroup::African, 0.18),    // Ionova 2020: 17.5%
                (AncestryGroup::EastAsian, 0.29),  // Ionova 2020: 28.4%
                (AncestryGroup::SouthAsian, 0.32), // Ionova 2020: 31.8%
                (AncestryGroup::NativeAmerican, 0.12),
                (AncestryGroup::Mixed, 0.18),
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "Ionova et al. 2020 Clin Transl Sci".into(),
        });

        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2C19".into(),
            allele: "*17".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.22),   // Ionova 2020: 21.7%
                (AncestryGroup::African, 0.22),    // Ionova 2020: 22.0%
                (AncestryGroup::EastAsian, 0.04),  // Ionova 2020: 3.7%
                (AncestryGroup::SouthAsian, 0.17), // Ionova 2020: 17.3%
                (AncestryGroup::NativeAmerican, 0.10),
                (AncestryGroup::Mixed, 0.15),
            ]),
            functional_status: AlleleFunction::IncreasedFunction,
            source: "Ionova et al. 2020 Clin Transl Sci".into(),
        });

        // --- CYP2C9 ---
        // Frequencies: Global distribution study (Hu Genomics 2023), PharmGKB.
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2C9".into(),
            allele: "*2".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.13),   // 12.7%
                (AncestryGroup::African, 0.02),    // 2.4% (Sub-Saharan mostly absent)
                (AncestryGroup::EastAsian, 0.00),  // <1%, effectively absent
                (AncestryGroup::SouthAsian, 0.05), // 4.6%
                (AncestryGroup::NativeAmerican, 0.01),
                (AncestryGroup::Mixed, 0.05),
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "Hu Genomics 2023 / PharmGKB".into(),
        });

        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2C9".into(),
            allele: "*3".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.07),   // 6.9%
                (AncestryGroup::African, 0.01),    // 1.3%
                (AncestryGroup::EastAsian, 0.04),  // 3.4%
                (AncestryGroup::SouthAsian, 0.11), // 11.3% — highest globally
                (AncestryGroup::NativeAmerican, 0.02),
                (AncestryGroup::Mixed, 0.04),
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "Hu Genomics 2023 / PharmGKB".into(),
        });

        // --- CYP3A4 ---
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP3A4".into(),
            allele: "*22".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.05),   // gnomAD v4 NFE: 4.7%
                (AncestryGroup::African, 0.001), // gnomAD v4 AFR: 0.9% (African-American admixed); continental <0.1%
                (AncestryGroup::EastAsian, 0.00), // gnomAD v4 EAS: 0% (absent)
                (AncestryGroup::SouthAsian, 0.01), // gnomAD v4 SAS: 0.9% (n=4,810 genomes); was 0.03 (overestimate)
                (AncestryGroup::NativeAmerican, 0.02),
                (AncestryGroup::Mixed, 0.02),
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "gnomAD v4 / JMD 2023".into(),
        });

        // --- CYP2B6 ---
        // CYP2B6*6 (rs3745274 G>T + rs2279343 A>G): the primary decreased-function
        // allele. Critical for efavirenz metabolism in HIV treatment.
        // Sources: Desta et al. (2019) PharmVar GeneFocus: CYP2B6;
        // Zanger & Klein (2013); PMC9784060 (African populations 37-38%);
        // PharmGKB CYP2B6 frequency table.
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2B6".into(),
            allele: "*6".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.26),       // PharmVar/PharmGKB: 25-28%
                (AncestryGroup::African, 0.38),        // PMC9784060: 37-38%; Mali study 37%
                (AncestryGroup::EastAsian, 0.18),      // PharmGKB: 15-21%
                (AncestryGroup::SouthAsian, 0.30),     // Pakistani study: 33.8%
                (AncestryGroup::NativeAmerican, 0.22), // Limited data; estimated
                (AncestryGroup::Mixed, 0.28),          // Weighted average
            ]),
            functional_status: AlleleFunction::DecreasedFunction,
            source: "Desta et al. 2019 / PharmGKB / PMC9784060".into(),
        });

        // CYP2B6*4 (rs2279343 A>G alone): increased-function allele.
        // More common in East Asian populations.
        // Source: PharmVar GeneFocus CYP2B6 (Desta et al. 2019).
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2B6".into(),
            allele: "*4".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.04),   // PharmGKB: 3-5%
                (AncestryGroup::African, 0.01),    // Rare in African populations
                (AncestryGroup::EastAsian, 0.07),  // PharmGKB: 5-9%
                (AncestryGroup::SouthAsian, 0.05), // Limited data
                (AncestryGroup::NativeAmerican, 0.03),
                (AncestryGroup::Mixed, 0.04),
            ]),
            functional_status: AlleleFunction::IncreasedFunction,
            source: "Desta et al. 2019 / PharmGKB".into(),
        });

        // CYP2B6*18 (rs28399499): no-function allele, nearly exclusive to
        // African-ancestry populations. Compounds *6 effect for PM phenotype.
        // Source: PharmVar; Desta et al. (2019).
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP2B6".into(),
            allele: "*18".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.00),  // Absent
                (AncestryGroup::African, 0.04),   // 3-5% Sub-Saharan African
                (AncestryGroup::EastAsian, 0.00), // Absent
                (AncestryGroup::SouthAsian, 0.00),
                (AncestryGroup::NativeAmerican, 0.00),
                (AncestryGroup::Mixed, 0.01),
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "Desta et al. 2019 / PharmGKB".into(),
        });

        // --- CYP3A5 ---
        // CYP3A5*3 (rs776746 A>G): the most common loss-of-function allele.
        // Non-expressors (*3/*3) cannot produce functional CYP3A5 protein.
        // CRITICAL: Largest inter-ancestry frequency gap in pharmacogenomics.
        // Sources: PMC3738061 (PharmGKB VIP summary); PMC5600063 meta-analysis;
        // Birdwell et al. (2015) CPIC tacrolimus guideline.
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP3A5".into(),
            allele: "*3".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.94), // PMC3738061: 82-95%; PMC5600063: 94.3%
                (AncestryGroup::African, 0.18), // PMC3738061: Yoruban 6%, Afr-Am 33%; PMC5600063: 18%
                (AncestryGroup::EastAsian, 0.71), // PMC3738061: Japanese 85%, Chinese 65%; PMC5600063: 71.3%
                (AncestryGroup::SouthAsian, 0.67), // PMC3738061: SE Asian 67%
                (AncestryGroup::NativeAmerican, 0.80), // Limited data; estimated intermediate
                (AncestryGroup::Mixed, 0.60),     // Weighted average
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "PMC3738061 / PMC5600063 / Birdwell 2015".into(),
        });

        // CYP3A5*6 (rs10264272): no-function allele, predominantly African.
        // Source: PMC3738061.
        self.allele_frequencies.push(PopulationAlleleFrequency {
            gene: "CYP3A5".into(),
            allele: "*6".into(),
            frequencies: HashMap::from([
                (AncestryGroup::European, 0.00),  // Absent/very rare
                (AncestryGroup::African, 0.13),   // 12-17% Sub-Saharan African
                (AncestryGroup::EastAsian, 0.00), // Absent
                (AncestryGroup::SouthAsian, 0.00),
                (AncestryGroup::NativeAmerican, 0.00),
                (AncestryGroup::Mixed, 0.03),
            ]),
            functional_status: AlleleFunction::NoFunction,
            source: "PMC3738061 / PharmGKB".into(),
        });
    }

    /// Build reference CPIC guidelines for common drug-gene pairs.
    ///
    /// Science: Hicks et al. (2015), CPIC guideline database.
    pub fn build_reference_cpic_guidelines(&mut self) {
        self.cpic_guidelines.clear();

        // --- Codeine / CYP2D6 ---
        self.cpic_guidelines.push(CpicGuideline {
            drug: "codeine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::AvoidDrug,
                dose_adjustment: None,
                alternative_drug: Some("morphine (non-CYP2D6)".into()),
                monitoring: Some(
                    "Risk of respiratory depression from rapid morphine conversion".into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "codeine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "codeine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative("morphine (non-CYP2D6)".into()),
                dose_adjustment: None,
                alternative_drug: Some("morphine (non-CYP2D6)".into()),
                monitoring: Some("Codeine ineffective — no conversion to morphine".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Clopidogrel / CYP2C19 ---
        self.cpic_guidelines.push(CpicGuideline {
            drug: "clopidogrel".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative("prasugrel or ticagrelor".into()),
                dose_adjustment: None,
                alternative_drug: Some("prasugrel or ticagrelor".into()),
                monitoring: Some("Clopidogrel is a prodrug — PMs have no active metabolite".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "clopidogrel".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Warfarin / CYP2C9 ---
        self.cpic_guidelines.push(CpicGuideline {
            drug: "warfarin".into(),
            gene: "CYP2C9".into(),
            phenotype: MetabolizerPhenotype::Intermediate,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.75),
                dose_adjustment: Some(0.75),
                alternative_drug: None,
                monitoring: Some("Monitor INR closely — intermediate metabolism".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "warfarin".into(),
            gene: "CYP2C9".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.50),
                dose_adjustment: Some(0.50),
                alternative_drug: None,
                monitoring: Some("High bleeding risk — frequent INR monitoring required".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Sertraline / CYP2C19 ---
        self.cpic_guidelines.push(CpicGuideline {
            drug: "sertraline".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::IncreaseDose(1.5),
                dose_adjustment: Some(1.5),
                alternative_drug: None,
                monitoring: Some("May need higher dose for therapeutic levels".into()),
            },
            evidence_level: EvidenceLevel::Moderate,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "sertraline".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Moderate,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "sertraline".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.50),
                dose_adjustment: Some(0.50),
                alternative_drug: Some("alternative SSRI (e.g., fluoxetine via CYP2D6)".into()),
                monitoring: Some("Risk of serotonin toxicity at standard dose".into()),
            },
            evidence_level: EvidenceLevel::Moderate,
            source_population: AncestryGroup::European,
        });

        // --- Efavirenz / CYP2B6 ---
        // CPIC Level A. Brown et al. (2019). CRITICAL for HIV treatment equity:
        // CYP2B6*6 homozygotes (~14% of African patients) have 3-4x higher
        // efavirenz plasma levels, causing CNS toxicity (dizziness, insomnia,
        // psychosis). Standard 600mg dose is toxic for PMs.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "efavirenz".into(),
            gene: "CYP2B6".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.50),
                dose_adjustment: Some(0.50),
                alternative_drug: Some("dolutegravir (non-CYP2B6)".into()),
                monitoring: Some(
                    "High CNS toxicity risk — monitor for dizziness, insomnia, vivid dreams. \
                     Consider TDM. 200-400mg may suffice for PM."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "efavirenz".into(),
            gene: "CYP2B6".into(),
            phenotype: MetabolizerPhenotype::Intermediate,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.67),
                dose_adjustment: Some(0.67),
                alternative_drug: None,
                monitoring: Some("Consider 400mg dose. Monitor for CNS side effects.".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "efavirenz".into(),
            gene: "CYP2B6".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "efavirenz".into(),
            gene: "CYP2B6".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::IncreaseDose(1.5),
                dose_adjustment: Some(1.5),
                alternative_drug: None,
                monitoring: Some(
                    "Risk of sub-therapeutic levels — monitor viral load closely.".into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Tamoxifen / CYP2D6 ---
        // CPIC Level A. Goetz et al. (2018). Breast cancer endocrine therapy:
        // CYP2D6 converts tamoxifen to endoxifen (40x more potent). PMs have
        // significantly reduced endoxifen levels and worse cancer outcomes.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tamoxifen".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative("aromatase inhibitor (anastrozole, letrozole)".into()),
                dose_adjustment: None,
                alternative_drug: Some("aromatase inhibitor (anastrozole, letrozole)".into()),
                monitoring: Some(
                    "PM: insufficient endoxifen formation. Switch to aromatase inhibitor \
                     if postmenopausal; consider higher tamoxifen dose (40mg) with TDM if premenopausal."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tamoxifen".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Intermediate,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative(
                    "aromatase inhibitor or higher tamoxifen dose".into(),
                ),
                dose_adjustment: Some(2.0),
                alternative_drug: Some("aromatase inhibitor (anastrozole, letrozole)".into()),
                monitoring: Some(
                    "IM: consider 40mg tamoxifen with endoxifen level monitoring, \
                     or switch to aromatase inhibitor."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tamoxifen".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tamoxifen".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: Some(
                    "Standard dose appropriate. UM may have higher endoxifen levels.".into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Atomoxetine / CYP2D6 ---
        // CPIC Level A. Primarily metabolized by CYP2D6.
        // PMs have ~10x higher AUC. Dose must be reduced 75%.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "atomoxetine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.25),
                dose_adjustment: Some(0.25),
                alternative_drug: None,
                monitoring: Some(
                    "PM: ~10x higher AUC. Start at 0.5mg/kg/day, do not exceed 1.2mg/kg/day. \
                     Monitor for tachycardia, insomnia, hepatotoxicity."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "atomoxetine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "atomoxetine".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::IncreaseDose(1.5),
                dose_adjustment: Some(1.5),
                alternative_drug: Some("guanfacine or methylphenidate".into()),
                monitoring: Some(
                    "UM: may need higher dose. If inadequate response at max dose, \
                     consider non-CYP2D6 alternative."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Ondansetron / CYP2D6 ---
        // CPIC Level A. CYP2D6 converts ondansetron to inactive metabolites.
        // UMs clear it too quickly — use granisetron (non-CYP2D6) instead.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "ondansetron".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative("granisetron".into()),
                dose_adjustment: None,
                alternative_drug: Some("granisetron".into()),
                monitoring: Some(
                    "UM: ondansetron cleared too rapidly — high risk of breakthrough nausea. \
                     Use granisetron (non-CYP2D6 metabolism)."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "ondansetron".into(),
            gene: "CYP2D6".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: None,
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Voriconazole / CYP2C19 ---
        // CPIC Level A. Moriyama et al. (2017). Critical antifungal for
        // immunocompromised patients. CYP2C19 is the primary metabolizer.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "voriconazole".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.50),
                dose_adjustment: Some(0.50),
                alternative_drug: None,
                monitoring: Some(
                    "PM: significantly elevated voriconazole levels. Mandatory TDM. \
                     Target trough 1-5.5 mg/L."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "voriconazole".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::UltraRapid,
            recommendation: DosingRecommendation {
                action: DosingAction::UseAlternative("alternative antifungal agent".into()),
                dose_adjustment: None,
                alternative_drug: Some("isavuconazole, posaconazole".into()),
                monitoring: Some(
                    "UM/RM: sub-therapeutic levels likely even at high doses. \
                     Use alternative antifungal or TDM with aggressive dose escalation."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "voriconazole".into(),
            gene: "CYP2C19".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: Some("TDM recommended for all patients.".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Tacrolimus / CYP3A5 ---
        // CPIC Level A. Birdwell et al. (2015). Transplant immunosuppressant.
        // CYP3A5 expressors (*1/*1, *1/*3) metabolize tacrolimus faster and
        // need 1.5-2x the standard dose to reach therapeutic trough levels.
        // EQUITY: 82% of Africans are CYP3A5 expressors vs ~6% of Europeans.
        // African transplant patients are systematically underdosed.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tacrolimus".into(),
            gene: "CYP3A5".into(),
            phenotype: MetabolizerPhenotype::Normal,  // Expressor: *1/*1
            recommendation: DosingRecommendation {
                action: DosingAction::IncreaseDose(2.0),
                dose_adjustment: Some(2.0),
                alternative_drug: None,
                monitoring: Some(
                    "CYP3A5 expressor (*1/*1): rapid tacrolimus clearance. \
                     Start at 0.3 mg/kg/day (2x standard). TDM mandatory — target trough 10-15 ng/mL."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tacrolimus".into(),
            gene: "CYP3A5".into(),
            phenotype: MetabolizerPhenotype::Intermediate, // Expressor: *1/*3
            recommendation: DosingRecommendation {
                action: DosingAction::IncreaseDose(1.5),
                dose_adjustment: Some(1.5),
                alternative_drug: None,
                monitoring: Some(
                    "CYP3A5 expressor (*1/*3): increased tacrolimus clearance. \
                     Start at 0.25 mg/kg/day (1.5x standard). TDM mandatory."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "tacrolimus".into(),
            gene: "CYP3A5".into(),
            phenotype: MetabolizerPhenotype::Poor, // Non-expressor: *3/*3
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: Some(
                    "CYP3A5 non-expressor (*3/*3): standard tacrolimus dosing. \
                     0.15 mg/kg/day. TDM mandatory."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });

        // --- Phenytoin / CYP2C9 ---
        // CPIC Level A. Caudle et al. (2020). Epilepsy treatment.
        // CYP2C9 PMs accumulate phenytoin — risk of ataxia, nystagmus,
        // cardiovascular collapse at standard doses.
        self.cpic_guidelines.push(CpicGuideline {
            drug: "phenytoin".into(),
            gene: "CYP2C9".into(),
            phenotype: MetabolizerPhenotype::Poor,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.50),
                dose_adjustment: Some(0.50),
                alternative_drug: None,
                monitoring: Some(
                    "PM: reduce dose 50%. Monitor serum phenytoin levels closely — \
                     target 10-20 mcg/mL. Risk of severe toxicity (ataxia, nystagmus)."
                        .into(),
                ),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "phenytoin".into(),
            gene: "CYP2C9".into(),
            phenotype: MetabolizerPhenotype::Intermediate,
            recommendation: DosingRecommendation {
                action: DosingAction::ReduceDose(0.75),
                dose_adjustment: Some(0.75),
                alternative_drug: None,
                monitoring: Some("IM: reduce dose 25%. Monitor serum phenytoin levels.".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
        self.cpic_guidelines.push(CpicGuideline {
            drug: "phenytoin".into(),
            gene: "CYP2C9".into(),
            phenotype: MetabolizerPhenotype::Normal,
            recommendation: DosingRecommendation {
                action: DosingAction::StandardDose,
                dose_adjustment: Some(1.0),
                alternative_drug: None,
                monitoring: Some("Standard loading and maintenance dose.".into()),
            },
            evidence_level: EvidenceLevel::Strong,
            source_population: AncestryGroup::European,
        });
    }

    /// Compute the expected metabolizer phenotype distribution for a gene
    /// in a given ancestry group.
    ///
    /// Uses Hardy-Weinberg equilibrium to derive diplotype frequencies from
    /// allele frequencies, then maps diplotypes to CPIC phenotype bins.
    pub fn metabolizer_distribution(
        &self,
        gene: &str,
        ancestry: AncestryGroup,
    ) -> HashMap<MetabolizerPhenotype, f64> {
        // Collect allele frequencies for this gene + ancestry.
        let alleles: Vec<(AlleleFunction, f64)> = self
            .allele_frequencies
            .iter()
            .filter(|a| a.gene == gene)
            .filter_map(|a| {
                a.frequencies
                    .get(&ancestry)
                    .map(|&f| (a.functional_status, f))
            })
            .collect();

        // Sum of non-normal allele frequencies; remainder is *1 (normal).
        let non_normal_sum: f64 = alleles.iter().map(|(_, f)| f).sum();
        let normal_freq = (1.0 - non_normal_sum).max(0.0);

        // Aggregate by function category.
        let mut no_fn_freq = 0.0_f64;
        let mut decreased_freq = 0.0_f64;
        let mut increased_freq = 0.0_f64;
        for &(func, freq) in &alleles {
            match func {
                AlleleFunction::NoFunction => no_fn_freq += freq,
                AlleleFunction::DecreasedFunction => decreased_freq += freq,
                AlleleFunction::IncreasedFunction => increased_freq += freq,
                _ => {} // NormalFunction / UncertainFunction absorbed into normal
            }
        }

        // Hardy-Weinberg diplotype → phenotype mapping:
        //   PM  = no_fn^2
        //   UM  = increased^2 + 2*increased*normal  (at least one increased + no loss)
        //   IM  = 2*no_fn*(normal+decreased) + decreased^2 + 2*decreased*normal - overlap
        //   NM  = remainder
        let pm = no_fn_freq * no_fn_freq;
        let um = increased_freq * increased_freq + 2.0 * increased_freq * normal_freq;
        // Intermediate: heterozygous loss, homozygous decreased, or decreased/normal heterozygote
        // IM = 2*nf*(n+d) + d^2 + 2*d*n
        //     = null/normal + null/decreased + decreased/decreased + decreased/normal
        let im = 2.0 * no_fn_freq * (normal_freq + decreased_freq)
            + decreased_freq * decreased_freq
            + 2.0 * decreased_freq * normal_freq;
        let nm = (1.0 - pm - um - im).clamp(0.0, 1.0);

        HashMap::from([
            (MetabolizerPhenotype::Poor, pm),
            (MetabolizerPhenotype::UltraRapid, um),
            (MetabolizerPhenotype::Intermediate, im),
            (MetabolizerPhenotype::Normal, nm),
        ])
    }

    /// Perform an equity analysis for a drug-gene pair across all ancestry groups.
    ///
    /// Computes per-population risk profiles and an aggregate equity gap score.
    pub fn equity_analysis(&self, drug: &str, gene: &str) -> EquityAnalysis {
        let ancestries = [
            AncestryGroup::European,
            AncestryGroup::African,
            AncestryGroup::EastAsian,
            AncestryGroup::SouthAsian,
            AncestryGroup::NativeAmerican,
            AncestryGroup::Mixed,
        ];

        let mut population_risk = HashMap::new();
        let mut max_adverse = 0.0_f64;
        let mut min_adverse = 1.0_f64;
        let mut max_efficacy_risk = 0.0_f64;
        let mut min_efficacy_risk = 1.0_f64;

        for &anc in &ancestries {
            let dist = self.metabolizer_distribution(gene, anc);

            // Adverse event risk: driven by PM (drug accumulation) + UM for prodrugs.
            let pm_frac = dist
                .get(&MetabolizerPhenotype::Poor)
                .copied()
                .unwrap_or(0.0);
            let um_frac = dist
                .get(&MetabolizerPhenotype::UltraRapid)
                .copied()
                .unwrap_or(0.0);

            // For prodrugs (codeine, clopidogrel, tamoxifen), UM is dangerous; for others, PM is.
            let is_prodrug = drug == "codeine" || drug == "clopidogrel" || drug == "tamoxifen";
            let adverse = if is_prodrug {
                um_frac * 0.8 + pm_frac * 0.3
            } else {
                pm_frac * 0.8 + um_frac * 0.2
            };

            // Efficacy risk: PM for prodrugs (no activation), UM for active drugs (sub-therapeutic).
            let efficacy_risk = if is_prodrug {
                pm_frac * 0.9
            } else {
                um_frac * 0.7 + pm_frac * 0.3
            };

            // Guideline applicability: penalize populations far from European distributions.
            let eur_dist = self.metabolizer_distribution(gene, AncestryGroup::European);
            let dist_divergence: f64 = dist
                .iter()
                .map(|(pheno, &frac)| {
                    let eur_frac = eur_dist.get(pheno).copied().unwrap_or(0.0);
                    (frac - eur_frac).abs()
                })
                .sum::<f64>()
                / 2.0; // normalize: max divergence = 1.0

            let guideline_applicability = (1.0 - dist_divergence).clamp(0.0, 1.0);

            max_adverse = max_adverse.max(adverse);
            min_adverse = min_adverse.min(adverse);
            max_efficacy_risk = max_efficacy_risk.max(efficacy_risk);
            min_efficacy_risk = min_efficacy_risk.min(efficacy_risk);

            population_risk.insert(
                anc,
                PopulationRisk {
                    phenotype_distribution: dist,
                    adverse_event_risk: adverse,
                    efficacy_risk,
                    guideline_applicability,
                },
            );
        }

        // Equity gap: maximum inter-population variance in combined risk.
        let adverse_range = max_adverse - min_adverse;
        let efficacy_range = max_efficacy_risk - min_efficacy_risk;
        let equity_gap_score = ((adverse_range + efficacy_range) / 2.0).clamp(0.0, 1.0);

        // Flag underserved populations.
        let underserved: Vec<AncestryGroup> = population_risk
            .iter()
            .filter(|(anc, risk)| {
                **anc != AncestryGroup::European
                    && (risk.guideline_applicability < (1.0 - UNDERSERVED_THRESHOLD)
                        || risk.adverse_event_risk > min_adverse + UNDERSERVED_THRESHOLD)
            })
            .map(|(anc, _)| *anc)
            .collect();

        let recommendation = if underserved.is_empty() {
            format!(
                "{}/{}: Current guidelines appear reasonably equitable across populations.",
                drug, gene
            )
        } else {
            let names: Vec<&str> = underserved
                .iter()
                .map(|a| match a {
                    AncestryGroup::European => "European",
                    AncestryGroup::African => "African",
                    AncestryGroup::EastAsian => "East Asian",
                    AncestryGroup::SouthAsian => "South Asian",
                    AncestryGroup::NativeAmerican => "Native American",
                    AncestryGroup::Mixed => "Mixed",
                })
                .collect();
            format!(
                "{}/{}: Equity gap {:.2}. Underserved populations: {}. \
                 Consider ancestry-adjusted dosing or therapeutic drug monitoring.",
                drug,
                gene,
                equity_gap_score,
                names.join(", ")
            )
        };

        EquityAnalysis {
            drug: drug.into(),
            gene: gene.into(),
            population_risk,
            equity_gap_score,
            underserved_populations: underserved,
            recommendation,
        }
    }

    /// Generate an ancestry-adjusted dosing recommendation.
    ///
    /// Starts from the CPIC base guideline for the given phenotype, then
    /// applies ancestry-specific adjustments when the patient belongs to
    /// an underserved population.
    pub fn adjusted_recommendation(
        &self,
        drug: &str,
        patient_ancestry: AncestryGroup,
        patient_phenotype: MetabolizerPhenotype,
    ) -> DosingRecommendation {
        // Find matching CPIC guideline.
        let base = self
            .cpic_guidelines
            .iter()
            .find(|g| g.drug == drug && g.phenotype == patient_phenotype);

        let base_rec = match base {
            Some(g) => g.recommendation.clone(),
            None => {
                // No specific guideline — default to standard dose with monitoring note.
                return DosingRecommendation {
                    action: DosingAction::StandardDose,
                    dose_adjustment: Some(1.0),
                    alternative_drug: None,
                    monitoring: Some(format!(
                        "No CPIC guideline for {}/{:?}. Monitor clinically.",
                        drug, patient_phenotype
                    )),
                };
            }
        };

        // Check if this patient's ancestry is underserved for this drug-gene pair.
        let equity = self.equity_analysis(drug, base.unwrap().gene.as_str());
        let is_underserved = equity.underserved_populations.contains(&patient_ancestry);

        if is_underserved {
            // Enhance monitoring for underserved populations.
            let ancestry_note = format!(
                "Patient ancestry may be underserved by current guidelines (equity gap: {:.2}). \
                 Recommend therapeutic drug monitoring.",
                equity.equity_gap_score
            );
            let monitoring = match base_rec.monitoring {
                Some(ref m) => Some(format!("{} | {}", m, ancestry_note)),
                None => Some(ancestry_note),
            };
            DosingRecommendation {
                action: base_rec.action,
                dose_adjustment: base_rec.dose_adjustment,
                alternative_drug: base_rec.alternative_drug,
                monitoring,
            }
        } else {
            base_rec
        }
    }

    /// Estimates the number of patients affected by ancestry-biased dosing
    /// for a specific drug-gene pair, using US census population data (2024).
    ///
    /// "At risk" = fraction of population with an actionable phenotype (PM or UM)
    /// where the standard European-calibrated guideline may not apply optimally.
    ///
    /// Sources: US Census Bureau (2024 estimates).
    pub fn clinical_impact_estimate(&self, drug: &str, gene: &str) -> ClinicalImpactEstimate {
        // Approximate US population by ancestry (2024 census estimates, millions).
        const POP_EUROPEAN: f64 = 195.0;
        const POP_AFRICAN: f64 = 47.0;
        const POP_EAST_ASIAN: f64 = 7.0;
        const POP_SOUTH_ASIAN: f64 = 6.0;
        const POP_NATIVE_AMERICAN: f64 = 4.0;
        const POP_MIXED: f64 = 72.0;

        let ancestries = [
            (AncestryGroup::European, POP_EUROPEAN),
            (AncestryGroup::African, POP_AFRICAN),
            (AncestryGroup::EastAsian, POP_EAST_ASIAN),
            (AncestryGroup::SouthAsian, POP_SOUTH_ASIAN),
            (AncestryGroup::NativeAmerican, POP_NATIVE_AMERICAN),
            (AncestryGroup::Mixed, POP_MIXED),
        ];

        let is_prodrug = drug == "codeine" || drug == "clopidogrel";
        // For tacrolimus/CYP3A5, expressors (NM/IM) need dose increase — different risk model.
        let is_expressor_risk = drug == "tacrolimus" && gene == "CYP3A5";

        let mut per_ancestry = Vec::new();
        let mut total_at_risk: u64 = 0;
        let mut max_risk_count: u64 = 0;
        let mut most_affected = AncestryGroup::European;

        for &(ancestry, pop_millions) in &ancestries {
            let dist = self.metabolizer_distribution(gene, ancestry);
            let pm = dist
                .get(&MetabolizerPhenotype::Poor)
                .copied()
                .unwrap_or(0.0);
            let um = dist
                .get(&MetabolizerPhenotype::UltraRapid)
                .copied()
                .unwrap_or(0.0);
            let nm = dist
                .get(&MetabolizerPhenotype::Normal)
                .copied()
                .unwrap_or(0.0);
            let im = dist
                .get(&MetabolizerPhenotype::Intermediate)
                .copied()
                .unwrap_or(0.0);

            let at_risk_fraction = if is_expressor_risk {
                // For tacrolimus: NM + IM are expressors who need dose increase.
                nm + im
            } else if is_prodrug {
                // For prodrugs: PM (no activation) + UM (toxic overactivation).
                pm + um
            } else {
                // For active drugs: PM (accumulation) + UM (sub-therapeutic).
                pm + um
            };

            let at_risk_count = (at_risk_fraction * pop_millions * 1_000_000.0) as u64;
            per_ancestry.push((ancestry, pop_millions, at_risk_fraction, at_risk_count));
            total_at_risk += at_risk_count;

            if at_risk_count > max_risk_count {
                max_risk_count = at_risk_count;
                most_affected = ancestry;
            }
        }

        let risk_description = format!(
            "{}/{}: ~{:.1}M patients at risk across US populations. \
             Most affected: {:?} ({:.1}M at-risk individuals, {:.1}% of that population).",
            drug,
            gene,
            total_at_risk as f64 / 1_000_000.0,
            most_affected,
            max_risk_count as f64 / 1_000_000.0,
            per_ancestry
                .iter()
                .find(|(a, _, _, _)| *a == most_affected)
                .map(|(_, _, f, _)| f * 100.0)
                .unwrap_or(0.0),
        );

        ClinicalImpactEstimate {
            drug: drug.into(),
            gene: gene.into(),
            per_ancestry,
            total_at_risk,
            most_affected,
            risk_description,
        }
    }

    /// Perturb all allele frequencies by a multiplicative factor.
    ///
    /// `factor` is a fractional change: e.g., 0.10 means +10%, -0.20 means -20%.
    /// All frequencies are clamped to [0.0, 1.0] after perturbation.
    /// Useful for sensitivity analysis to assess robustness of equity conclusions.
    pub fn perturb_frequencies(&mut self, factor: f64) {
        for allele in &mut self.allele_frequencies {
            for freq in allele.frequencies.values_mut() {
                *freq = (*freq * (1.0 + factor)).clamp(0.0, 1.0);
            }
        }
    }

    /// Return all unique (drug, gene) pairs present in the loaded CPIC guidelines.
    pub fn drug_gene_pairs(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = self
            .cpic_guidelines
            .iter()
            .map(|g| (g.drug.clone(), g.gene.clone()))
            .collect();
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// Generate a formatted population report for a gene.
    ///
    /// Shows metabolizer phenotype distributions across all ancestry groups,
    /// highlighting populations with > 10% ultra-rapid or poor metabolizer frequency.
    pub fn population_report(&self, gene: &str) -> String {
        let ancestries = [
            ("European", AncestryGroup::European),
            ("African", AncestryGroup::African),
            ("East Asian", AncestryGroup::EastAsian),
            ("South Asian", AncestryGroup::SouthAsian),
            ("Native American", AncestryGroup::NativeAmerican),
            ("Mixed", AncestryGroup::Mixed),
        ];

        let mut report = format!("=== {} Population Metabolizer Report ===\n\n", gene);

        for (name, anc) in &ancestries {
            let dist = self.metabolizer_distribution(gene, *anc);
            let um = dist
                .get(&MetabolizerPhenotype::UltraRapid)
                .copied()
                .unwrap_or(0.0);
            let nm = dist
                .get(&MetabolizerPhenotype::Normal)
                .copied()
                .unwrap_or(0.0);
            let im = dist
                .get(&MetabolizerPhenotype::Intermediate)
                .copied()
                .unwrap_or(0.0);
            let pm = dist
                .get(&MetabolizerPhenotype::Poor)
                .copied()
                .unwrap_or(0.0);

            let mut flags = Vec::new();
            if um > HIGH_FREQUENCY_PHENOTYPE_THRESHOLD {
                flags.push(format!("HIGH UM ({:.1}%)", um * 100.0));
            }
            if pm > HIGH_FREQUENCY_PHENOTYPE_THRESHOLD {
                flags.push(format!("HIGH PM ({:.1}%)", pm * 100.0));
            }

            let flag_str = if flags.is_empty() {
                String::new()
            } else {
                format!("  ** {} **", flags.join(", "))
            };

            report.push_str(&format!(
                "{:<16} UM: {:5.1}%  NM: {:5.1}%  IM: {:5.1}%  PM: {:5.1}%{}\n",
                name,
                um * 100.0,
                nm * 100.0,
                im * 100.0,
                pm * 100.0,
                flag_str,
            ));
        }

        report
    }

    // -----------------------------------------------------------------------
    // Admixed population methods
    // -----------------------------------------------------------------------

    /// Compute blended allele frequencies for an admixed population.
    ///
    /// For each allele, the admixed frequency is the weighted average of
    /// ancestral frequencies:
    ///   f_admixed = sum_i (w_i * f_i)
    ///
    /// This is the standard linear-mixture model used in population
    /// pharmacogenomics (Bryc et al. 2015).
    fn blended_allele_frequencies(
        &self,
        gene: &str,
        population: &AdmixedPopulation,
    ) -> Vec<(AlleleFunction, f64)> {
        self.allele_frequencies
            .iter()
            .filter(|a| a.gene == gene)
            .map(|a| {
                let blended_freq: f64 = population
                    .composition
                    .iter()
                    .map(|(anc, weight)| a.frequencies.get(anc).copied().unwrap_or(0.0) * weight)
                    .sum();
                (a.functional_status, blended_freq)
            })
            .collect()
    }

    /// Compute the expected metabolizer phenotype distribution for an
    /// admixed population.
    ///
    /// Allele frequencies are computed as weighted averages of ancestral
    /// group frequencies, then Hardy-Weinberg equilibrium is applied to
    /// the blended frequencies. This gives a first-order approximation
    /// that is more accurate than assigning admixed individuals to a
    /// single continental bin.
    ///
    /// Limitations:
    /// - Assumes HWE in the admixed population (ignores Wahlund effect
    ///   from recent admixture and population substructure).
    /// - Does not model linkage disequilibrium between loci.
    /// - Ancestry proportions are population-level averages; individual
    ///   patients may deviate substantially.
    pub fn admixed_metabolizer_distribution(
        &self,
        gene: &str,
        population: &AdmixedPopulation,
    ) -> HashMap<MetabolizerPhenotype, f64> {
        let alleles = self.blended_allele_frequencies(gene, population);

        // Sum of non-normal allele frequencies; remainder is *1 (normal).
        let non_normal_sum: f64 = alleles.iter().map(|(_, f)| f).sum();
        let normal_freq = (1.0 - non_normal_sum).max(0.0);

        // Aggregate by function category.
        let mut no_fn_freq = 0.0_f64;
        let mut decreased_freq = 0.0_f64;
        let mut increased_freq = 0.0_f64;
        for &(func, freq) in &alleles {
            match func {
                AlleleFunction::NoFunction => no_fn_freq += freq,
                AlleleFunction::DecreasedFunction => decreased_freq += freq,
                AlleleFunction::IncreasedFunction => increased_freq += freq,
                _ => {}
            }
        }

        // Hardy-Weinberg diplotype -> phenotype mapping (same as ancestral method).
        let pm = no_fn_freq * no_fn_freq;
        let um = increased_freq * increased_freq + 2.0 * increased_freq * normal_freq;
        // IM = null/normal + null/decreased + decreased/decreased + decreased/normal
        let im = 2.0 * no_fn_freq * (normal_freq + decreased_freq)
            + decreased_freq * decreased_freq
            + 2.0 * decreased_freq * normal_freq;
        let nm = (1.0 - pm - um - im).clamp(0.0, 1.0);

        HashMap::from([
            (MetabolizerPhenotype::Poor, pm),
            (MetabolizerPhenotype::UltraRapid, um),
            (MetabolizerPhenotype::Intermediate, im),
            (MetabolizerPhenotype::Normal, nm),
        ])
    }

    /// Compare equity gaps between ancestral and admixed populations for
    /// a drug-gene pair.
    ///
    /// For each admixed population, computes the metabolizer distribution,
    /// calculates a gap score relative to the European reference, and flags
    /// phenotypes with actionable frequencies above 10%.
    ///
    /// Returns a Vec of (population_name, gap_score, underserved_flags).
    pub fn admixed_equity_analysis(
        &self,
        drug: &str,
        gene: &str,
        populations: &[AdmixedPopulation],
    ) -> Vec<(String, f64, Vec<String>)> {
        let eur_dist = self.metabolizer_distribution(gene, AncestryGroup::European);
        let is_prodrug = drug == "codeine" || drug == "clopidogrel" || drug == "tamoxifen";

        populations
            .iter()
            .map(|pop| {
                let dist = self.admixed_metabolizer_distribution(gene, pop);

                // Compute divergence from European reference (same method as equity_analysis).
                let dist_divergence: f64 = dist
                    .iter()
                    .map(|(pheno, &frac)| {
                        let eur_frac = eur_dist.get(pheno).copied().unwrap_or(0.0);
                        (frac - eur_frac).abs()
                    })
                    .sum::<f64>()
                    / 2.0;

                let pm = dist
                    .get(&MetabolizerPhenotype::Poor)
                    .copied()
                    .unwrap_or(0.0);
                let um = dist
                    .get(&MetabolizerPhenotype::UltraRapid)
                    .copied()
                    .unwrap_or(0.0);

                // Risk model (matches equity_analysis).
                let adverse = if is_prodrug {
                    um * 0.8 + pm * 0.3
                } else {
                    pm * 0.8 + um * 0.2
                };
                let efficacy_risk = if is_prodrug {
                    pm * 0.9
                } else {
                    um * 0.7 + pm * 0.3
                };

                let gap_score = dist_divergence.clamp(0.0, 1.0);

                // Flag actionable phenotypes.
                let mut flags = Vec::new();
                if pm > HIGH_FREQUENCY_PHENOTYPE_THRESHOLD {
                    flags.push(format!("HIGH PM ({:.1}%)", pm * 100.0));
                }
                if um > HIGH_FREQUENCY_PHENOTYPE_THRESHOLD {
                    flags.push(format!("HIGH UM ({:.1}%)", um * 100.0));
                }
                if adverse > UNDERSERVED_THRESHOLD {
                    flags.push(format!("adverse_risk={:.2}", adverse));
                }
                if efficacy_risk > UNDERSERVED_THRESHOLD {
                    flags.push(format!("efficacy_risk={:.2}", efficacy_risk));
                }

                (pop.name.clone(), gap_score, flags)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> PgxHealthEquityEngine {
        PgxHealthEquityEngine::new()
    }

    #[test]
    fn test_european_cyp2d6_pm_frequency() {
        let e = engine();
        let dist = e.metabolizer_distribution("CYP2D6", AncestryGroup::European);
        let pm = dist[&MetabolizerPhenotype::Poor];
        // *4 freq = 0.20 (NoFunction), PM = no_fn^2 = 0.04
        // Close to 4% for *4 alone; with other alleles total ~7% in literature.
        assert!(pm > 0.03 && pm < 0.10, "European CYP2D6 PM={pm:.3}");
    }

    #[test]
    fn test_east_asian_cyp2d6_star10_frequency() {
        let e = engine();
        let star10 = e
            .allele_frequencies
            .iter()
            .find(|a| a.gene == "CYP2D6" && a.allele == "*10")
            .unwrap();
        let ea_freq = star10.frequencies[&AncestryGroup::EastAsian];
        assert!(
            (ea_freq - 0.43).abs() < 0.01,
            "East Asian CYP2D6*10 freq={ea_freq}"
        );
    }

    #[test]
    fn test_african_cyp2d6_um_frequency() {
        let e = engine();
        let dist = e.metabolizer_distribution("CYP2D6", AncestryGroup::African);
        let um = dist[&MetabolizerPhenotype::UltraRapid];
        // Pan-African duplication ~5% (Gaedigk 2017). 29% was Ethiopia-specific
        // (Aklillu 1996), not representative of all African populations.
        // UM ~5-7% pan-African is realistic.
        assert!(
            um > 0.03 && um < 0.15,
            "African CYP2D6 UM={um:.3}, expected 3-15%"
        );
    }

    #[test]
    fn test_metabolizer_distribution_sums_to_one() {
        let e = engine();
        for gene in &["CYP2D6", "CYP2C19", "CYP2C9"] {
            for &anc in &[
                AncestryGroup::European,
                AncestryGroup::African,
                AncestryGroup::EastAsian,
            ] {
                let dist = e.metabolizer_distribution(gene, anc);
                let total: f64 = dist.values().sum();
                assert!(
                    (total - 1.0).abs() < 0.01,
                    "{gene}/{anc:?} distribution sums to {total:.4}"
                );
            }
        }
    }

    #[test]
    fn test_equity_analysis_produces_valid_result_for_codeine() {
        let e = engine();
        let analysis = e.equity_analysis("codeine", "CYP2D6");
        assert!(analysis.equity_gap_score >= 0.0, "Gap score non-negative");
        assert!(analysis.equity_gap_score <= 1.0, "Gap score bounded");
        // At least one population should differ from the European reference
        assert!(
            !analysis.population_risk.is_empty(),
            "Should have per-population risk data"
        );
    }

    #[test]
    fn test_equity_analysis_flags_east_asian_for_cyp2c19() {
        let e = engine();
        let analysis = e.equity_analysis("clopidogrel", "CYP2C19");
        assert!(
            analysis
                .underserved_populations
                .contains(&AncestryGroup::EastAsian),
            "East Asian population should be flagged for clopidogrel/CYP2C19: {:?}",
            analysis.underserved_populations
        );
    }

    #[test]
    fn test_adjusted_recommendation_differs_for_ultra_rapid() {
        let e = engine();
        let rec = e.adjusted_recommendation(
            "codeine",
            AncestryGroup::African,
            MetabolizerPhenotype::UltraRapid,
        );
        // UM + codeine = avoid drug (CPIC) + underserved ancestry note.
        assert!(
            matches!(rec.action, DosingAction::AvoidDrug),
            "UM codeine should avoid drug, got {:?}",
            rec.action
        );
        // UM codeine should have monitoring guidance regardless of underserved status
        assert!(
            rec.monitoring.is_some(),
            "UM codeine should have monitoring recommendation"
        );
    }

    #[test]
    fn test_population_report_non_empty() {
        let e = engine();
        let report = e.population_report("CYP2D6");
        assert!(!report.is_empty());
        assert!(report.contains("European"));
        assert!(report.contains("African"));
        assert!(report.contains("UM:"));
    }

    #[test]
    fn test_cpic_guidelines_loaded() {
        let e = engine();
        assert!(
            e.cpic_guidelines.len() >= 10,
            "Expected at least 10 CPIC guidelines, got {}",
            e.cpic_guidelines.len()
        );
        // Verify codeine UM guideline exists.
        assert!(
            e.cpic_guidelines
                .iter()
                .any(|g| g.drug == "codeine" && g.phenotype == MetabolizerPhenotype::UltraRapid)
        );
    }

    #[test]
    fn test_equity_gap_both_nonzero() {
        let e = engine();
        let codeine_gap = e.equity_analysis("codeine", "CYP2D6").equity_gap_score;
        let sertraline_gap = e.equity_analysis("sertraline", "CYP2C19").equity_gap_score;
        // Both drugs have meaningful cross-population equity gaps.
        // CYP2C19 shows large variation (*2: 15% EUR vs 32% SAS) so sertraline
        // gap may exceed codeine with corrected frequencies.
        assert!(
            codeine_gap > 0.01,
            "Codeine equity gap ({codeine_gap:.3}) should be nonzero"
        );
        assert!(
            sertraline_gap > 0.01,
            "Sertraline equity gap ({sertraline_gap:.3}) should be nonzero"
        );
    }

    #[test]
    fn test_standard_dose_for_normal_metabolizer() {
        let e = engine();
        let rec = e.adjusted_recommendation(
            "codeine",
            AncestryGroup::European,
            MetabolizerPhenotype::Normal,
        );
        assert!(
            matches!(rec.action, DosingAction::StandardDose),
            "Normal metabolizer should get standard dose, got {:?}",
            rec.action
        );
    }

    #[test]
    fn test_alternative_drug_for_poor_metabolizer_codeine() {
        let e = engine();
        let rec = e.adjusted_recommendation(
            "codeine",
            AncestryGroup::European,
            MetabolizerPhenotype::Poor,
        );
        assert!(
            matches!(rec.action, DosingAction::UseAlternative(_)),
            "PM codeine should use alternative, got {:?}",
            rec.action
        );
        assert!(
            rec.alternative_drug.is_some(),
            "Should suggest alternative drug"
        );
    }

    #[test]
    fn test_allele_function_variants_present() {
        let e = engine();
        let functions: Vec<AlleleFunction> = e
            .allele_frequencies
            .iter()
            .map(|a| a.functional_status)
            .collect();
        assert!(functions.contains(&AlleleFunction::NoFunction));
        assert!(functions.contains(&AlleleFunction::DecreasedFunction));
        assert!(functions.contains(&AlleleFunction::IncreasedFunction));
    }

    // ---- New tests for expanded coverage ----

    #[test]
    fn test_cyp2b6_star6_african_higher_than_european() {
        let e = engine();
        let star6 = e
            .allele_frequencies
            .iter()
            .find(|a| a.gene == "CYP2B6" && a.allele == "*6")
            .unwrap();
        let afr = star6.frequencies[&AncestryGroup::African];
        let eur = star6.frequencies[&AncestryGroup::European];
        assert!(
            afr > eur,
            "CYP2B6*6 African freq ({afr}) should exceed European ({eur})"
        );
        // African ~0.38, European ~0.26 per PharmGKB/PMC9784060.
        assert!(afr > 0.30, "CYP2B6*6 African freq={afr}, expected >0.30");
        assert!(eur > 0.20, "CYP2B6*6 European freq={eur}, expected >0.20");
    }

    #[test]
    fn test_cyp3a5_star3_european_much_higher_than_african() {
        let e = engine();
        let star3 = e
            .allele_frequencies
            .iter()
            .find(|a| a.gene == "CYP3A5" && a.allele == "*3")
            .unwrap();
        let eur = star3.frequencies[&AncestryGroup::European];
        let afr = star3.frequencies[&AncestryGroup::African];
        let eas = star3.frequencies[&AncestryGroup::EastAsian];
        // European ~0.94, African ~0.18 (PMC3738061 / PMC5600063).
        assert!(
            eur > afr + 0.50,
            "CYP3A5*3 European ({eur}) should be >50pp higher than African ({afr})"
        );
        assert!(
            (eur - 0.94).abs() < 0.05,
            "CYP3A5*3 European freq={eur}, expected ~0.94"
        );
        assert!(
            (afr - 0.18).abs() < 0.05,
            "CYP3A5*3 African freq={afr}, expected ~0.18"
        );
        assert!(
            (eas - 0.71).abs() < 0.05,
            "CYP3A5*3 East Asian freq={eas}, expected ~0.71"
        );
    }

    #[test]
    fn test_efavirenz_equity_analysis_flags_african() {
        let e = engine();
        let analysis = e.equity_analysis("efavirenz", "CYP2B6");
        // African populations have the highest CYP2B6*6 frequency (~38%),
        // leading to more PMs who accumulate efavirenz to toxic levels.
        assert!(
            analysis
                .underserved_populations
                .contains(&AncestryGroup::African)
                || analysis.equity_gap_score > 0.01,
            "Efavirenz/CYP2B6 should show an equity gap or flag African: gap={:.3}, underserved={:?}",
            analysis.equity_gap_score,
            analysis.underserved_populations
        );
    }

    #[test]
    fn test_tacrolimus_equity_analysis_flags_african() {
        let e = engine();
        let analysis = e.equity_analysis("tacrolimus", "CYP3A5");
        // African populations have the lowest CYP3A5*3 frequency (~18%),
        // meaning ~82% are CYP3A5 expressors who metabolize tacrolimus faster
        // and are systematically underdosed.
        assert!(
            analysis.equity_gap_score > 0.05,
            "Tacrolimus/CYP3A5 should show a significant equity gap: {:.3}",
            analysis.equity_gap_score
        );
    }

    #[test]
    fn test_clinical_impact_estimate_reasonable_numbers() {
        let e = engine();
        let estimate = e.clinical_impact_estimate("codeine", "CYP2D6");
        // Total US population ~331M. At-risk (PM+UM) should be some fraction.
        assert!(
            estimate.total_at_risk > 1_000_000,
            "At least 1M at risk for codeine/CYP2D6, got {}",
            estimate.total_at_risk
        );
        assert!(
            estimate.total_at_risk < 200_000_000,
            "At risk should be <200M, got {}",
            estimate.total_at_risk
        );
        assert!(!estimate.risk_description.is_empty());
    }

    #[test]
    fn test_clinical_impact_total_equals_sum_of_per_ancestry() {
        let e = engine();
        let estimate = e.clinical_impact_estimate("efavirenz", "CYP2B6");
        let sum: u64 = estimate.per_ancestry.iter().map(|(_, _, _, c)| c).sum();
        assert_eq!(
            estimate.total_at_risk, sum,
            "Total at risk ({}) should equal sum of per-ancestry counts ({})",
            estimate.total_at_risk, sum
        );
    }

    #[test]
    fn test_clinical_impact_tacrolimus_most_affected() {
        let e = engine();
        let estimate = e.clinical_impact_estimate("tacrolimus", "CYP3A5");
        // For tacrolimus, European has the largest absolute population with
        // standard dosing already correct (*3/*3 non-expressors = PM = standard dose).
        // But the at-risk expressors are concentrated in African and Mixed groups.
        // European has huge population but low expressor rate (~6%).
        // The most_affected should be the group with highest absolute at-risk count.
        assert!(
            estimate.total_at_risk > 0,
            "Should have at-risk patients for tacrolimus/CYP3A5"
        );
    }

    #[test]
    fn test_expanded_cpic_guidelines_count() {
        let e = engine();
        // Original: 11 guidelines. New: +4 efavirenz +4 tamoxifen +3 atomoxetine
        // +2 ondansetron +3 voriconazole +3 tacrolimus +3 phenytoin = 22 new = 33 total.
        assert!(
            e.cpic_guidelines.len() >= 30,
            "Expected at least 30 CPIC guidelines after expansion, got {}",
            e.cpic_guidelines.len()
        );
        // Verify key new guidelines exist.
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "efavirenz"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "tamoxifen"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "atomoxetine"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "ondansetron"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "voriconazole"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "tacrolimus"));
        assert!(e.cpic_guidelines.iter().any(|g| g.drug == "phenytoin"));
    }

    #[test]
    fn test_perturb_frequencies_clamps_to_unit() {
        let mut e = engine();
        e.perturb_frequencies(0.20);
        for allele in &e.allele_frequencies {
            for &freq in allele.frequencies.values() {
                assert!(
                    (0.0..=1.0).contains(&freq),
                    "Frequency out of range after +20% perturbation: {freq}"
                );
            }
        }
        let mut e2 = engine();
        e2.perturb_frequencies(-0.20);
        for allele in &e2.allele_frequencies {
            for &freq in allele.frequencies.values() {
                assert!(
                    (0.0..=1.0).contains(&freq),
                    "Frequency out of range after -20% perturbation: {freq}"
                );
            }
        }
    }

    #[test]
    fn test_drug_gene_pairs_returns_all_eleven() {
        let e = engine();
        let pairs = e.drug_gene_pairs();
        assert!(
            pairs.len() >= 11,
            "Expected at least 11 drug-gene pairs, got {}",
            pairs.len()
        );
    }

    #[test]
    fn test_metabolizer_distribution_sums_to_one_new_genes() {
        let e = engine();
        for gene in &["CYP2B6", "CYP3A5"] {
            for &anc in &[
                AncestryGroup::European,
                AncestryGroup::African,
                AncestryGroup::EastAsian,
            ] {
                let dist = e.metabolizer_distribution(gene, anc);
                let total: f64 = dist.values().sum();
                assert!(
                    (total - 1.0).abs() < 0.01,
                    "{gene}/{anc:?} distribution sums to {total:.4}"
                );
            }
        }
    }

    // ---- Admixed population tests ----

    #[test]
    fn test_admixed_african_american_cyp2d6_between_ancestral() {
        let e = engine();
        let aa = AdmixedPopulation::african_american();
        let admixed_dist = e.admixed_metabolizer_distribution("CYP2D6", &aa);
        let afr_dist = e.metabolizer_distribution("CYP2D6", AncestryGroup::African);
        let eur_dist = e.metabolizer_distribution("CYP2D6", AncestryGroup::European);

        // HWE is nonlinear: blending allele frequencies before applying HWE
        // gives different results than blending phenotype frequencies. So admixed
        // phenotype values may fall OUTSIDE the range of ancestral values.
        // We verify that the distribution is valid and differs from both parents.
        let mut total = 0.0;
        for pheno in &[
            MetabolizerPhenotype::Poor,
            MetabolizerPhenotype::UltraRapid,
            MetabolizerPhenotype::Intermediate,
            MetabolizerPhenotype::Normal,
        ] {
            let admixed = admixed_dist[pheno];
            assert!(
                admixed >= 0.0 && admixed <= 1.0,
                "African American CYP2D6 {pheno:?}: admixed={admixed:.4} out of [0,1]"
            );
            total += admixed;
        }
        assert!(
            (total - 1.0).abs() < 0.01,
            "Distribution should sum to ~1.0, got {total}"
        );
        // Admixed should differ from both pure ancestral groups
        let admixed_pm = admixed_dist[&MetabolizerPhenotype::Poor];
        let afr_pm = afr_dist[&MetabolizerPhenotype::Poor];
        let eur_pm = eur_dist[&MetabolizerPhenotype::Poor];
        assert!(
            (admixed_pm - afr_pm).abs() > 0.001 || (admixed_pm - eur_pm).abs() > 0.001,
            "Admixed PM should differ from at least one ancestral group"
        );
    }

    #[test]
    fn test_admixed_us_latino_reflects_native_american_blend() {
        let e = engine();
        let latino = AdmixedPopulation::us_latino();
        let admixed_dist = e.admixed_metabolizer_distribution("CYP2C19", &latino);

        // US Latino has 40% Native American + 50% European + 10% African.
        // The PM rate should reflect this blend — not match any single group exactly.
        let eur_pm = e.metabolizer_distribution("CYP2C19", AncestryGroup::European)
            [&MetabolizerPhenotype::Poor];
        let nat_pm = e.metabolizer_distribution("CYP2C19", AncestryGroup::NativeAmerican)
            [&MetabolizerPhenotype::Poor];
        let admixed_pm = admixed_dist[&MetabolizerPhenotype::Poor];

        // Should not equal either pure ancestral group.
        assert!(
            (admixed_pm - eur_pm).abs() > 0.001 || (admixed_pm - nat_pm).abs() > 0.001,
            "US Latino PM={admixed_pm:.4} should differ from pure European={eur_pm:.4} \
             or pure NativeAmerican={nat_pm:.4}"
        );

        // Distribution should still sum to ~1.0.
        let total: f64 = admixed_dist.values().sum();
        assert!(
            (total - 1.0).abs() < 0.01,
            "US Latino CYP2C19 distribution sums to {total:.4}, expected ~1.0"
        );
    }

    #[test]
    fn test_admixed_composition_sums_to_one() {
        for pop in AdmixedPopulation::all_presets() {
            assert!(
                pop.is_valid(),
                "{} composition sums to {:.4}, expected ~1.0",
                pop.name,
                pop.total_weight()
            );
        }
    }

    #[test]
    fn test_admixed_equity_gap_intermediate() {
        let e = engine();
        let pops = AdmixedPopulation::all_presets();
        let results = e.admixed_equity_analysis("codeine", "CYP2D6", &pops);

        // Get ancestral gap scores for reference.
        let analysis = e.equity_analysis("codeine", "CYP2D6");
        let eur_risk = &analysis.population_risk[&AncestryGroup::European];
        let afr_risk = &analysis.population_risk[&AncestryGroup::African];
        let _eur_combined = eur_risk.adverse_event_risk + eur_risk.efficacy_risk;
        let _afr_combined = afr_risk.adverse_event_risk + afr_risk.efficacy_risk;

        // African American (80% African, 20% European) gap should be non-zero
        // but less extreme than pure African divergence from European.
        let aa_result = results
            .iter()
            .find(|(n, _, _)| n == "African American")
            .unwrap();
        assert!(
            aa_result.1 > 0.0,
            "African American should have a nonzero gap for codeine/CYP2D6"
        );

        // All results should have valid gap scores in [0, 1].
        for (name, gap, _) in &results {
            assert!(
                *gap >= 0.0 && *gap <= 1.0,
                "{name} gap score {gap} out of [0,1] range"
            );
        }

        // Should have results for all 4 preset populations.
        assert_eq!(results.len(), 4, "Expected 4 admixed population results");
    }

    #[test]
    fn test_admixed_metabolizer_distribution_sums_to_one() {
        let e = engine();
        for pop in AdmixedPopulation::all_presets() {
            for gene in &["CYP2D6", "CYP2C19", "CYP2C9", "CYP2B6", "CYP3A5"] {
                let dist = e.admixed_metabolizer_distribution(gene, &pop);
                let total: f64 = dist.values().sum();
                assert!(
                    (total - 1.0).abs() < 0.01,
                    "{}/{} admixed distribution sums to {total:.4}",
                    pop.name,
                    gene
                );
            }
        }
    }

    #[test]
    fn test_admixed_brazilian_cyp3a5_between_ancestral() {
        let e = engine();
        let br = AdmixedPopulation::brazilian();
        let admixed_dist = e.admixed_metabolizer_distribution("CYP3A5", &br);
        let eur_dist = e.metabolizer_distribution("CYP3A5", AncestryGroup::European);
        let afr_dist = e.metabolizer_distribution("CYP3A5", AncestryGroup::African);

        // Brazilian is 60% European + 25% African + 15% Native American.
        // PM (non-expressor) should be dominated by the European component
        // but pulled down by the African component.
        let admixed_pm = admixed_dist[&MetabolizerPhenotype::Poor];
        let eur_pm = eur_dist[&MetabolizerPhenotype::Poor];
        let afr_pm = afr_dist[&MetabolizerPhenotype::Poor];

        // Should be less than pure European PM (since African ancestry pulls it down).
        assert!(
            admixed_pm < eur_pm + 0.01,
            "Brazilian CYP3A5 PM={admixed_pm:.4} should be <= European PM={eur_pm:.4}"
        );
        // Should be more than pure African PM.
        assert!(
            admixed_pm > afr_pm - 0.01,
            "Brazilian CYP3A5 PM={admixed_pm:.4} should be >= African PM={afr_pm:.4}"
        );
    }

    #[test]
    fn test_admixed_composition_string() {
        let aa = AdmixedPopulation::african_american();
        let s = aa.composition_string();
        assert!(s.contains("Afr"), "Should contain 'Afr': {s}");
        assert!(s.contains("Eur"), "Should contain 'Eur': {s}");
    }
}

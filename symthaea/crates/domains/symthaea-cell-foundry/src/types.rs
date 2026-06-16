// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core data types for cell culture control and biological prediction.

use serde::{Deserialize, Serialize};

/// Biological prediction horizons (seconds)
pub const BIO_HORIZONS: &[f32] = &[
    3_600.0,      // 1 hour (metabolic)
    86_400.0,     // 1 day (cell division)
    604_800.0,    // 1 week (differentiation)
    2_592_000.0,  // 1 month (tissue formation)
    7_776_000.0,  // 3 months (organ development)
    23_328_000.0, // 9 months (gestation)
];

/// Default gene expression profile length
pub const DEFAULT_GENE_EXPR_LEN: usize = 32;

/// Broad categories of cell identity used throughout the foundry pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    /// Adult somatic (differentiated) cell, suitable as iPSC or SCNT donor.
    Somatic,
    /// Induced pluripotent stem cell (iPSC) — fully reprogrammed.
    Pluripotent, // iPSC
    /// Primordial germ cell-like cell (PGC-LC) — output of PGC induction.
    PrimordialGerm, // PGC-LC
    /// Oogonium stage in the female germline.
    Oogonium,
    /// Mature oocyte — used as SCNT recipient or IVG endpoint.
    Oocyte,
    /// Spermatogonial stem cell stage in the male germline.
    Spermatogonium,
    /// Primary spermatocyte — entering meiosis.
    Spermatocyte,
    /// Early embryonic cell (e.g., post-SCNT blastocyst).
    Embryonic,
    /// Terminally differentiated cell of a specific tissue lineage.
    Differentiated(DifferentiatedType),
}

/// Specific tissue lineage for a fully differentiated cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferentiatedType {
    /// Neural lineage (neurons, glia).
    Neural,
    /// Cardiac muscle lineage.
    Cardiac,
    /// Hepatocyte / liver cell lineage.
    Hepatic,
    /// Renal tubular or podocyte lineage.
    Renal,
    /// Epithelial (surface barrier) lineage.
    Epithelial,
    /// Mesenchymal stromal lineage.
    Mesenchymal,
}

/// Snapshot of a single cell's biological state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellState {
    /// Identity / lineage of the cell.
    pub cell_type: CellType,
    /// Proportion of living, intact cells (0–1).
    pub viability: f64, // 0-1
    /// Degree of pluripotency; 1.0 = fully pluripotent iPSC.
    pub pluripotency_score: f64, // 0-1 (1 = fully pluripotent)
    /// Number of cell divisions accumulated since culture start.
    pub division_count: u32,
    /// Time in culture since seeding (hours).
    pub age_hours: f64,
    /// Global DNA methylation level (0 = fully demethylated, 1 = fully methylated).
    pub methylation_score: f64, // Global methylation level 0-1
    /// Simplified per-gene expression profile (each value 0–1).
    pub gene_expression: Vec<f64>, // Simplified expression profile
}

impl CellState {
    /// Create a default somatic cell.
    pub fn new_somatic() -> Self {
        Self {
            cell_type: CellType::Somatic,
            viability: 0.95,
            pluripotency_score: 0.0,
            division_count: 0,
            age_hours: 0.0,
            methylation_score: 0.7,
            gene_expression: vec![0.5; DEFAULT_GENE_EXPR_LEN],
        }
    }

    /// Create a default iPSC cell.
    pub fn new_ipsc() -> Self {
        Self {
            cell_type: CellType::Pluripotent,
            viability: 0.90,
            pluripotency_score: 0.95,
            division_count: 0,
            age_hours: 0.0,
            methylation_score: 0.3,
            gene_expression: vec![0.7; DEFAULT_GENE_EXPR_LEN],
        }
    }

    /// Create a default oocyte.
    pub fn new_oocyte() -> Self {
        Self {
            cell_type: CellType::Oocyte,
            viability: 0.85,
            pluripotency_score: 0.0,
            division_count: 0,
            age_hours: 0.0,
            methylation_score: 0.5,
            gene_expression: vec![0.4; DEFAULT_GENE_EXPR_LEN],
        }
    }
}

/// Physical parameters of a cell culture incubation environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CultureEnvironment {
    /// Incubator temperature in °C (target ≈ 37.0).
    pub temperature_celsius: f64, // target ~37.0
    /// CO₂ concentration in percent (target ≈ 5.0).
    pub co2_percent: f64, // target ~5.0
    /// O₂ concentration in percent (target ≈ 20.0, or 5.0 for hypoxic culture).
    pub o2_percent: f64, // target ~20.0 (or 5.0 for hypoxic)
    /// Media pH (target ≈ 7.4).
    pub ph: f64, // target ~7.4
    /// Relative humidity in percent (target ≈ 95.0).
    pub humidity_percent: f64, // target ~95.0
    /// Volume of culture media in the vessel (mL).
    pub media_volume_ml: f64,
    /// Number of times the culture has been passaged (subcultured).
    pub passage_number: u32,
}

impl CultureEnvironment {
    /// Standard culture conditions.
    pub fn standard() -> Self {
        Self {
            temperature_celsius: 37.0,
            co2_percent: 5.0,
            o2_percent: 20.0,
            ph: 7.4,
            humidity_percent: 95.0,
            media_volume_ml: 10.0,
            passage_number: 0,
        }
    }

    /// Hypoxic conditions (low O2, used for stem cell culture).
    pub fn hypoxic() -> Self {
        Self {
            temperature_celsius: 37.0,
            co2_percent: 5.0,
            o2_percent: 5.0,
            ph: 7.4,
            humidity_percent: 95.0,
            media_volume_ml: 10.0,
            passage_number: 0,
        }
    }

    /// Compute deviation from standard conditions (sum of squared fractional errors).
    pub fn deviation_from_standard(&self) -> f64 {
        let std = CultureEnvironment::standard();
        let temp_err =
            (self.temperature_celsius - std.temperature_celsius) / std.temperature_celsius;
        let co2_err = (self.co2_percent - std.co2_percent) / std.co2_percent;
        let ph_err = (self.ph - std.ph) / std.ph;
        let humidity_err = (self.humidity_percent - std.humidity_percent) / std.humidity_percent;
        temp_err * temp_err + co2_err * co2_err + ph_err * ph_err + humidity_err * humidity_err
    }
}

/// Stage of a Yamanaka factor iPSC reprogramming protocol (28-day timeline).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReprogrammingStage {
    /// Day 0: Yamanaka factor delivery initiated.
    Initial, // Day 0: factor delivery
    /// Days 1–3: Mesenchymal-to-Epithelial Transition (MET) begins.
    EarlyResponse, // Day 1-3: MET begins
    /// Days 4–10: Epigenetic remodeling and chromatin opening.
    MidReprogramming, // Day 4-10: epigenetic remodeling
    /// Days 11–20: Pluripotency gene network activation.
    LateReprogramming, // Day 11-20: pluripotency activation
    /// Days 21–28: Stable iPSC colony formation and expansion.
    Stabilization, // Day 21-28: stable iPSC colonies
    /// Protocol finished; iPSC identity validated.
    Complete, // Validated iPSC
    /// Protocol terminated due to cell death or quality failure.
    Failed,
}

/// Developmental stage during in vitro gametogenesis (IVG): iPSC → gamete.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GermCellStage {
    /// Starting iPSC that feeds into the IVG pipeline.
    Ipsc,
    /// Primordial germ cell-like (PGC-LC) induction phase.
    PgcInduction, // PGC-like cell induction
    /// Early germ cell after PGC induction.
    EarlyGerm, // Early germ cell
    /// Cell is transitioning into meiotic prophase I.
    MeioticEntry, // Entering meiosis
    /// Meiosis I (reductional division) in progress.
    MeiosisI,
    /// Meiosis II (equational division) in progress.
    MeiosisII,
    /// IVG complete; mature haploid gamete produced.
    MatureGamete,
    /// Protocol terminated due to checkpoint failure or low viability.
    Failed,
}

/// Quality control checkpoints verified during meiosis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeiosisCheckpoint {
    /// Homologous chromosome synapsis (pairing) in prophase I.
    Synapsis, // Homolog pairing
    /// Crossover / recombination event count verification.
    Crossover, // Recombination
    /// Chromosome segregation quality after meiosis I.
    SegregationI, // First division
    /// Chromatid segregation quality after meiosis II.
    SegregationII, // Second division
}

/// Stage of a Somatic Cell Nuclear Transfer (SCNT) protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScntStage {
    /// Removal of the oocyte's own nucleus.
    Enucleation,
    /// Injection of the donor somatic nucleus into the enucleated oocyte.
    NuclearInjection,
    /// Oocyte activation to initiate nuclear reprogramming.
    Activation,
    /// First mitotic division of the reconstructed embryo.
    FirstDivision,
    /// Development to blastocyst stage (inner cell mass + trophoblast).
    BlastocystFormation,
    /// SCNT successfully completed; embryo is viable.
    Complete,
    /// Protocol terminated due to failed stage transition.
    Failed,
}

/// Expression levels of the four canonical Yamanaka reprogramming factors (each 0–1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamanakaFactors {
    /// OCT4 (POU5F1) expression level (master pluripotency regulator).
    pub oct4: f64, // 0-1 expression level
    /// SOX2 expression level (pluripotency co-factor with OCT4).
    pub sox2: f64,
    /// KLF4 expression level (anti-apoptotic reprogramming factor).
    pub klf4: f64,
    /// c-MYC expression level (proliferation booster; reduced late to limit oncogenesis risk).
    pub c_myc: f64,
}

impl YamanakaFactors {
    /// Total expression level (sum of all four factors).
    pub fn total_expression(&self) -> f64 {
        self.oct4 + self.sox2 + self.klf4 + self.c_myc
    }

    /// Balanced score: how close to equal expression of all four factors (0-1).
    pub fn balance_score(&self) -> f64 {
        let mean = self.total_expression() / 4.0;
        if mean < 1e-9 {
            return 0.0;
        }
        let variance = ((self.oct4 - mean).powi(2)
            + (self.sox2 - mean).powi(2)
            + (self.klf4 - mean).powi(2)
            + (self.c_myc - mean).powi(2))
            / 4.0;
        // Normalized inverse variance (0 variance -> score 1.0)
        1.0 / (1.0 + variance / (mean * mean))
    }
}

/// Epigenetic state of a cell relevant to reprogramming quality and imprinting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticProfile {
    /// Genome-wide CpG methylation fraction (0 = fully demethylated, 1 = fully methylated).
    pub global_methylation: f64,
    /// H19/IGF2 imprinting completeness (0 = erased, 1 = fully imprinted).
    pub h19_igf2_imprinting: f64, // 0 = no imprint, 1 = full imprint
    /// SNRPN locus imprinting completeness (0 = erased, 1 = fully imprinted).
    pub snrpn_imprinting: f64,
    /// Whether X-chromosome inactivation has been re-established.
    pub x_inactivation_status: bool,
    /// Telomere length relative to a healthy newborn baseline (1.0).
    pub telomere_length: f64, // Relative to newborn (1.0)
}

impl EpigeneticProfile {
    /// Create a healthy newborn profile.
    pub fn healthy_newborn() -> Self {
        Self {
            global_methylation: 0.7,
            h19_igf2_imprinting: 1.0,
            snrpn_imprinting: 1.0,
            x_inactivation_status: true,
            telomere_length: 1.0,
        }
    }

    /// Create a partially erased (reprogrammed) profile.
    pub fn partially_erased() -> Self {
        Self {
            global_methylation: 0.3,
            h19_igf2_imprinting: 0.5,
            snrpn_imprinting: 0.4,
            x_inactivation_status: false,
            telomere_length: 0.8,
        }
    }
}

/// Comprehensive quality control report for a cell product.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Whether karyotype analysis found a normal chromosome complement.
    pub karyotype_normal: bool,
    /// Measured expression levels of key pluripotency markers (name, score).
    pub pluripotency_markers: Vec<(String, f64)>,
    /// Cell viability at the time of assessment (0–1).
    pub viability: f64,
    /// Whether the culture passed contamination screening.
    pub contamination_free: bool,
    /// Overall epigenetic quality score (0–1; see `epigenetic_quality_score`).
    pub epigenetic_score: f64,
    /// True when all critical checks pass and the product is release-eligible.
    pub overall_pass: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_horizons_ordering() {
        for i in 1..BIO_HORIZONS.len() {
            assert!(
                BIO_HORIZONS[i] > BIO_HORIZONS[i - 1],
                "BIO_HORIZONS must be strictly increasing"
            );
        }
    }

    #[test]
    fn test_bio_horizons_span_1h_to_9mo() {
        assert!((BIO_HORIZONS[0] - 3600.0).abs() < 1.0);
        assert!((BIO_HORIZONS[BIO_HORIZONS.len() - 1] - 23_328_000.0).abs() < 1.0);
    }

    #[test]
    fn test_cell_state_somatic() {
        let cell = CellState::new_somatic();
        assert_eq!(cell.cell_type, CellType::Somatic);
        assert!(cell.viability > 0.9);
        assert!(cell.pluripotency_score < 0.01);
    }

    #[test]
    fn test_cell_state_ipsc() {
        let cell = CellState::new_ipsc();
        assert_eq!(cell.cell_type, CellType::Pluripotent);
        assert!(cell.pluripotency_score > 0.9);
    }

    #[test]
    fn test_culture_standard() {
        let env = CultureEnvironment::standard();
        assert!((env.temperature_celsius - 37.0).abs() < 0.01);
        assert!((env.co2_percent - 5.0).abs() < 0.01);
        assert!((env.ph - 7.4).abs() < 0.01);
    }

    #[test]
    fn test_culture_deviation_zero_for_standard() {
        let env = CultureEnvironment::standard();
        assert!(env.deviation_from_standard() < 1e-10);
    }

    #[test]
    fn test_culture_deviation_nonzero_for_shifted() {
        let mut env = CultureEnvironment::standard();
        env.temperature_celsius = 38.0;
        assert!(env.deviation_from_standard() > 0.0);
    }

    #[test]
    fn test_yamanaka_total_expression() {
        let factors = YamanakaFactors {
            oct4: 0.8,
            sox2: 0.7,
            klf4: 0.6,
            c_myc: 0.5,
        };
        assert!((factors.total_expression() - 2.6).abs() < 1e-9);
    }

    #[test]
    fn test_yamanaka_balance_perfect() {
        let factors = YamanakaFactors {
            oct4: 0.5,
            sox2: 0.5,
            klf4: 0.5,
            c_myc: 0.5,
        };
        assert!((factors.balance_score() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_yamanaka_balance_imbalanced() {
        let factors = YamanakaFactors {
            oct4: 1.0,
            sox2: 0.0,
            klf4: 0.0,
            c_myc: 0.0,
        };
        assert!(factors.balance_score() < 0.5);
    }

    #[test]
    fn test_epigenetic_healthy_newborn() {
        let profile = EpigeneticProfile::healthy_newborn();
        assert!((profile.h19_igf2_imprinting - 1.0).abs() < 1e-9);
        assert!((profile.snrpn_imprinting - 1.0).abs() < 1e-9);
        assert!((profile.telomere_length - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cell_type_serde_roundtrip() {
        let ct = CellType::Differentiated(DifferentiatedType::Neural);
        let json = serde_json::to_string(&ct).unwrap();
        let deser: CellType = serde_json::from_str(&json).unwrap();
        assert_eq!(ct, deser);
    }

    #[test]
    fn test_quality_report_serde() {
        let report = QualityReport {
            karyotype_normal: true,
            pluripotency_markers: vec![("OCT4".into(), 0.9), ("SOX2".into(), 0.85)],
            viability: 0.95,
            contamination_free: true,
            epigenetic_score: 0.88,
            overall_pass: true,
        };
        let json = serde_json::to_string(&report).unwrap();
        let deser: QualityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.karyotype_normal, deser.karyotype_normal);
        assert_eq!(
            report.pluripotency_markers.len(),
            deser.pluripotency_markers.len()
        );
    }
}

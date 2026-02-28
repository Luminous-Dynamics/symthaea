//! Core data types for cell culture control and biological prediction.

use serde::{Deserialize, Serialize};

/// Biological prediction horizons (seconds)
pub const BIO_HORIZONS: &[f32] = &[
    3_600.0,       // 1 hour (metabolic)
    86_400.0,      // 1 day (cell division)
    604_800.0,     // 1 week (differentiation)
    2_592_000.0,   // 1 month (tissue formation)
    7_776_000.0,   // 3 months (organ development)
    23_328_000.0,  // 9 months (gestation)
];

/// Default gene expression profile length
pub const DEFAULT_GENE_EXPR_LEN: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    Somatic,
    Pluripotent,     // iPSC
    PrimordialGerm,  // PGC-LC
    Oogonium,
    Oocyte,
    Spermatogonium,
    Spermatocyte,
    Embryonic,
    Differentiated(DifferentiatedType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DifferentiatedType {
    Neural,
    Cardiac,
    Hepatic,
    Renal,
    Epithelial,
    Mesenchymal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellState {
    pub cell_type: CellType,
    pub viability: f64,           // 0-1
    pub pluripotency_score: f64,  // 0-1 (1 = fully pluripotent)
    pub division_count: u32,
    pub age_hours: f64,
    pub methylation_score: f64,   // Global methylation level 0-1
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CultureEnvironment {
    pub temperature_celsius: f64, // target ~37.0
    pub co2_percent: f64,         // target ~5.0
    pub o2_percent: f64,          // target ~20.0 (or 5.0 for hypoxic)
    pub ph: f64,                  // target ~7.4
    pub humidity_percent: f64,    // target ~95.0
    pub media_volume_ml: f64,
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
        let temp_err = (self.temperature_celsius - std.temperature_celsius) / std.temperature_celsius;
        let co2_err = (self.co2_percent - std.co2_percent) / std.co2_percent;
        let ph_err = (self.ph - std.ph) / std.ph;
        let humidity_err = (self.humidity_percent - std.humidity_percent) / std.humidity_percent;
        temp_err * temp_err + co2_err * co2_err + ph_err * ph_err + humidity_err * humidity_err
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReprogrammingStage {
    Initial,          // Day 0: factor delivery
    EarlyResponse,    // Day 1-3: MET begins
    MidReprogramming, // Day 4-10: epigenetic remodeling
    LateReprogramming, // Day 11-20: pluripotency activation
    Stabilization,    // Day 21-28: stable iPSC colonies
    Complete,         // Validated iPSC
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GermCellStage {
    Ipsc,
    PgcInduction,   // PGC-like cell induction
    EarlyGerm,      // Early germ cell
    MeioticEntry,   // Entering meiosis
    MeiosisI,
    MeiosisII,
    MatureGamete,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeiosisCheckpoint {
    Synapsis,       // Homolog pairing
    Crossover,      // Recombination
    SegregationI,   // First division
    SegregationII,  // Second division
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScntStage {
    Enucleation,
    NuclearInjection,
    Activation,
    FirstDivision,
    BlastocystFormation,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YamanakaFactors {
    pub oct4: f64, // 0-1 expression level
    pub sox2: f64,
    pub klf4: f64,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpigeneticProfile {
    pub global_methylation: f64,
    pub h19_igf2_imprinting: f64, // 0 = no imprint, 1 = full imprint
    pub snrpn_imprinting: f64,
    pub x_inactivation_status: bool,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub karyotype_normal: bool,
    pub pluripotency_markers: Vec<(String, f64)>,
    pub viability: f64,
    pub contamination_free: bool,
    pub epigenetic_score: f64,
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
        assert_eq!(report.pluripotency_markers.len(), deser.pluripotency_markers.len());
    }
}

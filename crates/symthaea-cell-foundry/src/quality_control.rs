// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quality control assessment for cell products.
//!
//! Provides karyotype checking, pluripotency marker analysis, and overall
//! quality reports that combine cell state with epigenetic profiling.

use crate::epigenetics::epigenetic_quality_score;
use crate::types::{CellState, EpigeneticProfile, QualityReport};

/// Assess overall quality of a cell product.
///
/// Combines karyotype analysis, pluripotency markers, viability, contamination
/// status, and epigenetic profiling into a comprehensive quality report.
pub fn assess_quality(cell: &CellState, epigenetics: &EpigeneticProfile) -> QualityReport {
    let karyotype_normal = check_karyotype(cell);
    let pluripotency_markers = check_pluripotency_markers(cell);
    let viability = cell.viability;
    let contamination_free = check_contamination(cell);
    let epigenetic_score = epigenetic_quality_score(epigenetics);

    // Overall pass requires all critical checks
    let marker_pass = pluripotency_markers.iter().all(|(_, score)| *score > 0.5);
    let overall_pass = karyotype_normal
        && marker_pass
        && viability > 0.7
        && contamination_free
        && epigenetic_score > 0.5;

    QualityReport {
        karyotype_normal,
        pluripotency_markers,
        viability,
        contamination_free,
        epigenetic_score,
        overall_pass,
    }
}

/// Check karyotype normality.
///
/// In a real system this would involve image analysis of chromosome spreads.
/// Here we use proxy indicators: high viability and low division count suggest
/// normal karyotype (fewer opportunities for chromosome instability).
pub fn check_karyotype(cell: &CellState) -> bool {
    // High passage/division count increases aneuploidy risk
    let division_risk = if cell.division_count > 50 {
        0.6 // High risk
    } else if cell.division_count > 20 {
        0.8
    } else {
        0.95
    };

    // Low viability correlates with genomic instability
    let viability_factor = cell.viability.clamp(0.0, 1.0);

    // Combined probability of normal karyotype
    let normal_probability = division_risk * viability_factor;
    normal_probability > 0.7
}

/// Check pluripotency markers.
///
/// Returns marker name and expression level for key pluripotency markers.
/// For iPSCs these should all be high; for somatic cells they should be low.
pub fn check_pluripotency_markers(cell: &CellState) -> Vec<(String, f64)> {
    let base = cell.pluripotency_score;

    // Marker levels derived from overall pluripotency score with marker-specific offsets
    vec![
        ("OCT4".to_string(), (base * 1.0).clamp(0.0, 1.0)),
        ("SOX2".to_string(), (base * 0.95).clamp(0.0, 1.0)),
        ("NANOG".to_string(), (base * 0.90).clamp(0.0, 1.0)),
        ("TRA-1-60".to_string(), (base * 0.85).clamp(0.0, 1.0)),
        ("SSEA-4".to_string(), (base * 0.88).clamp(0.0, 1.0)),
    ]
}

/// Check for contamination (mycoplasma, bacterial, cross-contamination).
///
/// Uses cell state proxy: very low viability or abnormal gene expression
/// patterns suggest contamination.
fn check_contamination(cell: &CellState) -> bool {
    // Extremely low viability suggests contamination
    if cell.viability < 0.1 {
        return false;
    }

    // Check gene expression for anomalous patterns
    if !cell.gene_expression.is_empty() {
        let mean: f64 =
            cell.gene_expression.iter().sum::<f64>() / cell.gene_expression.len() as f64;
        // If all genes are at extreme values, something is wrong
        let all_extreme = cell
            .gene_expression
            .iter()
            .all(|&x| !(0.05..=0.95).contains(&x));
        if all_extreme && cell.gene_expression.len() > 5 {
            return false;
        }
        // Negative expression is biologically impossible
        if cell.gene_expression.iter().any(|&x| x < -0.01) {
            return false;
        }
        // Very high mean with no variance = likely artifact
        if mean > 0.98 && cell.gene_expression.len() > 5 {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DEFAULT_GENE_EXPR_LEN;

    #[test]
    fn test_karyotype_normal_healthy() {
        let cell = CellState::new_somatic();
        assert!(check_karyotype(&cell));
    }

    #[test]
    fn test_karyotype_risk_high_divisions() {
        let mut cell = CellState::new_somatic();
        cell.division_count = 100;
        cell.viability = 0.5;
        assert!(!check_karyotype(&cell));
    }

    #[test]
    fn test_pluripotency_markers_ipsc() {
        let cell = CellState::new_ipsc();
        let markers = check_pluripotency_markers(&cell);
        assert_eq!(markers.len(), 5);
        // All markers should be high for iPSC
        for (name, score) in &markers {
            assert!(
                *score > 0.5,
                "iPSC marker {} should be high, got {}",
                name,
                score
            );
        }
    }

    #[test]
    fn test_pluripotency_markers_somatic() {
        let cell = CellState::new_somatic();
        let markers = check_pluripotency_markers(&cell);
        // All markers should be low/zero for somatic
        for (name, score) in &markers {
            assert!(
                *score < 0.1,
                "Somatic marker {} should be low, got {}",
                name,
                score
            );
        }
    }

    #[test]
    fn test_assess_quality_ipsc_pass() {
        let cell = CellState::new_ipsc();
        let epi = EpigeneticProfile::healthy_newborn();
        let report = assess_quality(&cell, &epi);
        assert!(
            report.overall_pass,
            "Healthy iPSC should pass QC: karyotype={}, viability={}, epi_score={}, contamination_free={}",
            report.karyotype_normal,
            report.viability,
            report.epigenetic_score,
            report.contamination_free
        );
    }

    #[test]
    fn test_assess_quality_low_viability_fail() {
        let mut cell = CellState::new_ipsc();
        cell.viability = 0.3;
        let epi = EpigeneticProfile::healthy_newborn();
        let report = assess_quality(&cell, &epi);
        assert!(!report.overall_pass, "Low viability should fail QC");
    }

    #[test]
    fn test_assess_quality_bad_epigenetics_fail() {
        let cell = CellState::new_ipsc();
        let epi = EpigeneticProfile {
            global_methylation: 0.1,
            h19_igf2_imprinting: 0.2,
            snrpn_imprinting: 0.1,
            x_inactivation_status: false,
            telomere_length: 0.2,
        };
        let report = assess_quality(&cell, &epi);
        assert!(!report.overall_pass, "Bad epigenetics should fail QC");
    }

    #[test]
    fn test_contamination_healthy() {
        let cell = CellState::new_somatic();
        assert!(check_contamination(&cell));
    }

    #[test]
    fn test_contamination_dead_cells() {
        let mut cell = CellState::new_somatic();
        cell.viability = 0.05;
        assert!(!check_contamination(&cell));
    }

    #[test]
    fn test_contamination_extreme_expression() {
        let mut cell = CellState::new_somatic();
        cell.gene_expression = vec![0.99; DEFAULT_GENE_EXPR_LEN];
        assert!(!check_contamination(&cell));
    }

    #[test]
    fn test_quality_report_serde() {
        let cell = CellState::new_ipsc();
        let epi = EpigeneticProfile::healthy_newborn();
        let report = assess_quality(&cell, &epi);
        let json = serde_json::to_string(&report).unwrap();
        let deser: QualityReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.overall_pass, deser.overall_pass);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Chou-Fasman secondary structure prediction algorithm.
//!
//! Physics-based baseline predictor using amino acid propensity tables
//! from Chou & Fasman (1978). Predicts H (helix), E (sheet), or C (coil)
//! for each residue based on sliding-window propensity averages.

use crate::amino_acid::*;

/// Predict secondary structure using the Chou-Fasman propensity method.
///
/// For each residue, computes the average helix, sheet, and coil propensities
/// over a sliding window of size `window` (default 7, centered). Assigns
/// the class with the highest average propensity.
pub fn predict_chou_fasman(sequence: &[AminoAcid]) -> Vec<SecondaryStructure> {
    predict_chou_fasman_window(sequence, 7)
}

/// Chou-Fasman prediction with configurable window size.
pub fn predict_chou_fasman_window(
    sequence: &[AminoAcid],
    window: usize,
) -> Vec<SecondaryStructure> {
    let n = sequence.len();
    if n == 0 {
        return vec![];
    }

    let half = window / 2;
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let start = if i >= half { i - half } else { 0 };
        let end = (i + half + 1).min(n);
        let win = &sequence[start..end];
        let win_len = win.len() as f64;

        let avg_h: f64 = win.iter().map(|&aa| helix_propensity(aa)).sum::<f64>() / win_len;
        let avg_e: f64 = win.iter().map(|&aa| sheet_propensity(aa)).sum::<f64>() / win_len;
        let avg_c: f64 = win.iter().map(|&aa| coil_propensity(aa)).sum::<f64>() / win_len;

        // Check for helix breakers in the window
        let has_breaker = win.iter().any(|&aa| is_helix_breaker(aa));

        let ss = if avg_h > avg_e && avg_h > avg_c && !has_breaker {
            SecondaryStructure::Helix
        } else if avg_e > avg_c {
            SecondaryStructure::Sheet
        } else {
            SecondaryStructure::Coil
        };
        result.push(ss);
    }

    result
}

/// Compute Q3 accuracy: fraction of residues where predicted == actual.
pub fn q3_accuracy(predicted: &[SecondaryStructure], actual: &[SecondaryStructure]) -> f64 {
    assert_eq!(predicted.len(), actual.len(), "sequence length mismatch");
    if predicted.is_empty() {
        return 0.0;
    }
    let correct = predicted
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count();
    correct as f64 / predicted.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_empty() {
        assert!(predict_chou_fasman(&[]).is_empty());
    }

    #[test]
    fn test_predict_single_residue() {
        let seq = parse_sequence("A");
        let pred = predict_chou_fasman(&seq);
        assert_eq!(pred.len(), 1);
    }

    #[test]
    fn test_predict_preserves_length() {
        let seq = parse_sequence("ACDEFGHIKLMNPQRSTVWY");
        let pred = predict_chou_fasman(&seq);
        assert_eq!(pred.len(), seq.len());
    }

    #[test]
    fn test_poly_ala_favors_helix() {
        // Poly-alanine should be strongly helical
        let seq = parse_sequence("AAAAAAAAAA");
        let pred = predict_chou_fasman(&seq);
        let helix_count = pred
            .iter()
            .filter(|&&ss| ss == SecondaryStructure::Helix)
            .count();
        // Most residues should be helix (Ala has highest helix propensity)
        assert!(
            helix_count >= 5,
            "Expected mostly helix for poly-Ala, got {} helix out of {}",
            helix_count,
            pred.len()
        );
    }

    #[test]
    fn test_poly_val_favors_sheet() {
        // Poly-valine should favor sheet (Val has highest sheet propensity)
        let seq = parse_sequence("VVVVVVVVVV");
        let pred = predict_chou_fasman(&seq);
        let sheet_count = pred
            .iter()
            .filter(|&&ss| ss == SecondaryStructure::Sheet)
            .count();
        assert!(
            sheet_count >= 5,
            "Expected mostly sheet for poly-Val, got {} sheet out of {}",
            sheet_count,
            pred.len()
        );
    }

    #[test]
    fn test_proline_breaks_helix() {
        // Pro is a helix breaker — inserting it into poly-Ala should break helix
        let seq = parse_sequence("AAAPAAAA");
        let pred = predict_chou_fasman(&seq);
        // The residues near Pro should not be helix
        let pro_idx = 3;
        assert_ne!(
            pred[pro_idx],
            SecondaryStructure::Helix,
            "Pro should break helix"
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Window-based feature extraction for protein secondary structure prediction.
//!
//! Extracts 15 physicochemical features from a sliding window centered on each
//! residue. Edge residues use padding (repeat the edge residue).
//!
//! Feature design follows the insight that local sequence context determines
//! secondary structure -- the same principle underlying Chou-Fasman (1978),
//! GOR (Garnier et al., 1978), and PSIPRED (Jones, 1999).

use crate::amino_acid::*;

/// Sliding window radius. Total window = 2 * HALF_WINDOW + 1.
pub const HALF_WINDOW: usize = 5;

/// Total window size (11 residues).
pub const WINDOW_SIZE: usize = 2 * HALF_WINDOW + 1;

/// Number of features extracted per residue.
pub const N_FEATURES: usize = 15;

/// Extract 15 physicochemical features for position `pos` in `seq`.
///
/// Features:
/// - 0:  Relative position (0.0 = N-terminus, 1.0 = C-terminus)
/// - 1:  Window average hydrophobicity (Kyte-Doolittle)
/// - 2:  Window average charge (pH 7)
/// - 3:  Window average volume (normalized 0-1)
/// - 4:  Window average helix propensity (Chou-Fasman P_alpha)
/// - 5:  Window average sheet propensity (Chou-Fasman P_beta)
/// - 6:  Helix minus sheet propensity difference (center residue)
/// - 7:  Center residue hydrophobicity
/// - 8:  Center residue helix propensity
/// - 9:  Center residue sheet propensity
/// - 10: Helix breaker count in window (Gly + Pro)
/// - 11: Hydrophobic residue fraction in window
/// - 12: Charged residue fraction in window
/// - 13: Aromatic residue count in window (F, W, Y, H)
/// - 14: Small residue count in window (G, A, S)
pub fn extract_features(seq: &[AminoAcid], pos: usize) -> [f64; N_FEATURES] {
    let len = seq.len();
    debug_assert!(pos < len, "pos {} out of range for seq len {}", pos, len);

    // Collect window residues with edge padding (repeat edge residue)
    let mut window = [AminoAcid::Ala; WINDOW_SIZE];
    for i in 0..WINDOW_SIZE {
        let offset = i as isize - HALF_WINDOW as isize;
        let idx = pos as isize + offset;
        let clamped = idx.clamp(0, (len - 1) as isize) as usize;
        window[i] = seq[clamped];
    }

    let center = seq[pos];
    let win_f = WINDOW_SIZE as f64;

    // Window averages
    let avg_hydro: f64 = window.iter().map(|&aa| hydrophobicity(aa)).sum::<f64>() / win_f;
    let avg_charge: f64 = window.iter().map(|&aa| charge(aa)).sum::<f64>() / win_f;
    let avg_volume: f64 = window.iter().map(|&aa| volume_normalized(aa)).sum::<f64>() / win_f;
    let avg_helix: f64 = window.iter().map(|&aa| helix_propensity(aa)).sum::<f64>() / win_f;
    let avg_sheet: f64 = window.iter().map(|&aa| sheet_propensity(aa)).sum::<f64>() / win_f;

    // Counts
    let breaker_count = window.iter().filter(|&&aa| is_helix_breaker(aa)).count() as f64;
    let hydro_frac = window.iter().filter(|&&aa| is_hydrophobic(aa)).count() as f64 / win_f;
    let charged_frac = window.iter().filter(|&&aa| is_charged(aa)).count() as f64 / win_f;
    let aromatic_count = window.iter().filter(|&&aa| is_aromatic(aa)).count() as f64;
    let small_count = window.iter().filter(|&&aa| is_small(aa)).count() as f64;

    [
        // 0: relative position
        if len <= 1 {
            0.5
        } else {
            pos as f64 / (len - 1) as f64
        },
        // 1-5: window averages
        avg_hydro,
        avg_charge,
        avg_volume,
        avg_helix,
        avg_sheet,
        // 6: propensity difference (center)
        helix_propensity(center) - sheet_propensity(center),
        // 7-9: center residue properties
        hydrophobicity(center),
        helix_propensity(center),
        sheet_propensity(center),
        // 10-14: window counts/fractions
        breaker_count,
        hydro_frac,
        charged_frac,
        aromatic_count,
        small_count,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        let seq = parse_sequence("ACDEFGHIKLMNPQRSTVWY");
        let f = extract_features(&seq, 10);
        assert_eq!(f.len(), N_FEATURES);
    }

    #[test]
    fn test_relative_position() {
        let seq = parse_sequence("ACDEFGHIKLMNPQRSTVWY");
        let f_start = extract_features(&seq, 0);
        let f_end = extract_features(&seq, seq.len() - 1);
        assert!((f_start[0] - 0.0).abs() < 1e-10, "N-term should be 0.0");
        assert!((f_end[0] - 1.0).abs() < 1e-10, "C-term should be 1.0");
    }

    #[test]
    fn test_polyalanine_features() {
        let seq = parse_sequence("AAAAAAAAAAAAAAA");
        let f = extract_features(&seq, 7);
        // All alanine: hydrophobicity should be Ala's value (1.8)
        assert!(
            (f[1] - 1.8).abs() < 0.01,
            "avg hydro should be ~1.8, got {}",
            f[1]
        );
        // Helix propensity of Ala = 1.42
        assert!(
            (f[4] - 1.42).abs() < 0.01,
            "avg helix should be ~1.42, got {}",
            f[4]
        );
        // Ala is a small residue
        assert!(
            f[14] > 5.0,
            "all-Ala window should have many small residues"
        );
    }

    #[test]
    fn test_edge_padding() {
        // Short sequence: padding should not panic
        let seq = parse_sequence("AV");
        let f0 = extract_features(&seq, 0);
        let f1 = extract_features(&seq, 1);
        assert!(f0[0].is_finite());
        assert!(f1[0].is_finite());
    }

    #[test]
    fn test_single_residue() {
        let seq = parse_sequence("G");
        let f = extract_features(&seq, 0);
        assert!(
            (f[0] - 0.5).abs() < 1e-10,
            "single residue position should be 0.5"
        );
        // Gly is a helix breaker
        assert!(f[10] > 0.0, "Gly window should have breakers");
    }

    #[test]
    fn test_features_finite() {
        let seq = parse_sequence("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGK");
        for pos in 0..seq.len() {
            let f = extract_features(&seq, pos);
            for (i, &val) in f.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "feature {} at pos {} is not finite",
                    i,
                    pos
                );
            }
        }
    }
}

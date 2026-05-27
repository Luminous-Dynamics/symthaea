// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evaluation metrics and benchmarks for protein secondary structure prediction.
//!
//! Provides Q3 accuracy (overall three-state accuracy), per-class accuracy
//! (Q_H, Q_E, Q_C), and confusion matrices for comparing predicted vs actual
//! secondary structure assignments.

use crate::amino_acid::SecondaryStructure;

/// Overall three-state accuracy (Q3): fraction of residues correctly predicted.
///
/// Returns 0.0 if the slices are empty or different lengths.
pub fn q3_accuracy(pred: &[SecondaryStructure], actual: &[SecondaryStructure]) -> f64 {
    if pred.is_empty() || pred.len() != actual.len() {
        return 0.0;
    }
    let correct = pred
        .iter()
        .zip(actual.iter())
        .filter(|(p, a)| p == a)
        .count();
    correct as f64 / pred.len() as f64
}

/// Per-class accuracy: (Q_H, Q_E, Q_C).
///
/// Q_X = (correctly predicted X) / (total actual X). Returns 0.0 for any
/// class with zero actual residues.
pub fn per_class_accuracy(
    pred: &[SecondaryStructure],
    actual: &[SecondaryStructure],
) -> (f64, f64, f64) {
    if pred.len() != actual.len() {
        return (0.0, 0.0, 0.0);
    }

    let mut correct_h = 0usize;
    let mut total_h = 0usize;
    let mut correct_e = 0usize;
    let mut total_e = 0usize;
    let mut correct_c = 0usize;
    let mut total_c = 0usize;

    for (p, a) in pred.iter().zip(actual.iter()) {
        match a {
            SecondaryStructure::Helix => {
                total_h += 1;
                if p == a {
                    correct_h += 1;
                }
            }
            SecondaryStructure::Sheet => {
                total_e += 1;
                if p == a {
                    correct_e += 1;
                }
            }
            SecondaryStructure::Coil => {
                total_c += 1;
                if p == a {
                    correct_c += 1;
                }
            }
        }
    }

    let q_h = if total_h > 0 {
        correct_h as f64 / total_h as f64
    } else {
        0.0
    };
    let q_e = if total_e > 0 {
        correct_e as f64 / total_e as f64
    } else {
        0.0
    };
    let q_c = if total_c > 0 {
        correct_c as f64 / total_c as f64
    } else {
        0.0
    };

    (q_h, q_e, q_c)
}

/// 3x3 confusion matrix: `matrix[actual][predicted]`.
///
/// Row/column indices: 0=Helix, 1=Sheet, 2=Coil.
pub fn confusion_matrix(
    pred: &[SecondaryStructure],
    actual: &[SecondaryStructure],
) -> [[usize; 3]; 3] {
    let mut matrix = [[0usize; 3]; 3];

    for (p, a) in pred.iter().zip(actual.iter()) {
        let actual_idx = ss_index(*a);
        let pred_idx = ss_index(*p);
        matrix[actual_idx][pred_idx] += 1;
    }

    matrix
}

fn ss_index(ss: SecondaryStructure) -> usize {
    match ss {
        SecondaryStructure::Helix => 0,
        SecondaryStructure::Sheet => 1,
        SecondaryStructure::Coil => 2,
    }
}

/// Per-class accuracy breakdown for a three-state prediction.
#[derive(Debug, Clone)]
pub struct ClassAccuracy {
    /// Per-class (H, E, C) correct / total counts.
    pub class_correct: [usize; 3],
    pub class_total: [usize; 3],
}

impl ClassAccuracy {
    /// Compute per-class accuracy from predicted vs actual.
    pub fn compute(predicted: &[SecondaryStructure], actual: &[SecondaryStructure]) -> Self {
        let mut correct = [0usize; 3];
        let mut total = [0usize; 3];
        for (p, a) in predicted.iter().zip(actual.iter()) {
            let c = ss_index(*a);
            total[c] += 1;
            if p == a {
                correct[c] += 1;
            }
        }
        ClassAccuracy {
            class_correct: correct,
            class_total: total,
        }
    }

    /// Overall Q3 accuracy.
    pub fn q3(&self) -> f64 {
        let total: usize = self.class_total.iter().sum();
        let correct: usize = self.class_correct.iter().sum();
        if total == 0 {
            0.0
        } else {
            correct as f64 / total as f64
        }
    }

    /// Per-class accuracy for a given class.
    pub fn class_q(&self, class: u8) -> f64 {
        let c = class as usize;
        if self.class_total[c] == 0 {
            0.0
        } else {
            self.class_correct[c] as f64 / self.class_total[c] as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amino_acid::*;
    use crate::cb513::*;
    use crate::chou_fasman::predict_chou_fasman;
    use crate::features::extract_features;
    use crate::rf_ssp::RandomForestClassifier;

    /// Extract feature vectors for all residues in a sequence.
    fn extract_all(seq: &[AminoAcid]) -> Vec<[f64; crate::features::N_FEATURES]> {
        (0..seq.len()).map(|i| extract_features(seq, i)).collect()
    }

    // --- Unit tests for metrics ---

    #[test]
    fn test_q3_perfect() {
        use SecondaryStructure::*;
        let actual = vec![Helix, Sheet, Coil, Helix];
        let pred = actual.clone();
        assert!((q3_accuracy(&pred, &actual) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_q3_empty() {
        assert_eq!(q3_accuracy(&[], &[]), 0.0);
    }

    #[test]
    fn test_q3_length_mismatch() {
        use SecondaryStructure::*;
        assert_eq!(q3_accuracy(&[Helix], &[Helix, Sheet]), 0.0);
    }

    #[test]
    fn test_q3_half_correct() {
        use SecondaryStructure::*;
        let actual = vec![Helix, Helix, Sheet, Sheet];
        let pred = vec![Helix, Sheet, Sheet, Helix];
        assert!((q3_accuracy(&pred, &actual) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_per_class_perfect() {
        use SecondaryStructure::*;
        let actual = vec![Helix, Sheet, Coil];
        let pred = actual.clone();
        let (qh, qe, qc) = per_class_accuracy(&pred, &actual);
        assert!((qh - 1.0).abs() < 1e-10);
        assert!((qe - 1.0).abs() < 1e-10);
        assert!((qc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_class_no_helix() {
        use SecondaryStructure::*;
        let actual = vec![Sheet, Coil, Sheet];
        let pred = vec![Sheet, Coil, Coil];
        let (qh, qe, qc) = per_class_accuracy(&pred, &actual);
        assert_eq!(qh, 0.0); // No helix in actual
        assert!((qe - 0.5).abs() < 1e-10); // 1 of 2 sheets correct
        assert!((qc - 1.0).abs() < 1e-10); // 1 of 1 coil correct
    }

    #[test]
    fn test_confusion_matrix_diagonal() {
        use SecondaryStructure::*;
        let actual = vec![Helix, Sheet, Coil];
        let pred = actual.clone();
        let cm = confusion_matrix(&pred, &actual);
        assert_eq!(cm[0][0], 1); // H->H
        assert_eq!(cm[1][1], 1); // E->E
        assert_eq!(cm[2][2], 1); // C->C
        // Off-diagonal should be zero
        assert_eq!(cm[0][1], 0);
        assert_eq!(cm[0][2], 0);
        assert_eq!(cm[1][0], 0);
        assert_eq!(cm[1][2], 0);
        assert_eq!(cm[2][0], 0);
        assert_eq!(cm[2][1], 0);
    }

    #[test]
    fn test_confusion_matrix_off_diagonal() {
        use SecondaryStructure::*;
        let actual = vec![Helix, Helix, Sheet];
        let pred = vec![Sheet, Helix, Coil];
        let cm = confusion_matrix(&pred, &actual);
        assert_eq!(cm[0][0], 1); // H->H
        assert_eq!(cm[0][1], 1); // H predicted as E
        assert_eq!(cm[1][2], 1); // E predicted as C
    }

    // --- Benchmark tests ---

    #[test]
    fn benchmark_chou_fasman_baseline() {
        let proteins = load_cb513_subset();
        assert!(!proteins.is_empty());

        let mut total_correct = 0usize;
        let mut total_residues = 0usize;
        let mut class_correct = [0usize; 3];
        let mut class_total = [0usize; 3];

        for p in &proteins {
            let pred = predict_chou_fasman(&p.sequence);
            assert_eq!(pred.len(), p.ss.len());

            for (pr, ac) in pred.iter().zip(p.ss.iter()) {
                total_residues += 1;
                let idx = ss_index(*ac);
                class_total[idx] += 1;
                if pr == ac {
                    total_correct += 1;
                    class_correct[idx] += 1;
                }
            }
        }

        let q3 = total_correct as f64 / total_residues as f64;
        let q_h = if class_total[0] > 0 {
            class_correct[0] as f64 / class_total[0] as f64
        } else {
            0.0
        };
        let q_e = if class_total[1] > 0 {
            class_correct[1] as f64 / class_total[1] as f64
        } else {
            0.0
        };
        let q_c = if class_total[2] > 0 {
            class_correct[2] as f64 / class_total[2] as f64
        } else {
            0.0
        };

        eprintln!("--- Chou-Fasman Baseline ---");
        eprintln!("  Total residues: {}", total_residues);
        eprintln!("  Q3: {:.1}%", q3 * 100.0);
        eprintln!(
            "  Q_H: {:.1}%, Q_E: {:.1}%, Q_C: {:.1}%",
            q_h * 100.0,
            q_e * 100.0,
            q_c * 100.0
        );

        // Chou-Fasman typically achieves 50-60% Q3
        assert!(q3 > 0.20, "Chou-Fasman Q3 too low: {:.1}%", q3 * 100.0);
        assert!(total_residues >= 1000);
    }

    #[test]
    fn benchmark_rf_predictor() {
        let proteins = load_cb513_subset();
        let n = proteins.len();
        assert!(n >= 20);

        // 80/20 split at the protein level
        let split = (n * 4) / 5;
        let train_proteins = &proteins[..split];
        let test_proteins = &proteins[split..];

        // Collect training data
        let mut train_features = Vec::new();
        let mut train_labels = Vec::new();
        for p in train_proteins {
            let feats = extract_all(&p.sequence);
            for (f, &ss) in feats.into_iter().zip(p.ss.iter()) {
                train_features.push(f);
                train_labels.push(ss.class_id());
            }
        }

        // Train RF
        let rf = RandomForestClassifier::train(&train_features, &train_labels, 50, 8);

        // Evaluate on test set
        let mut total_correct = 0usize;
        let mut total_residues = 0usize;
        for p in test_proteins {
            for (pos, &actual_ss) in p.ss.iter().enumerate() {
                let feats = extract_features(&p.sequence, pos);
                let pred = SecondaryStructure::from_class_id(rf.predict(&feats));
                total_residues += 1;
                if pred == actual_ss {
                    total_correct += 1;
                }
            }
        }

        let q3 = if total_residues > 0 {
            total_correct as f64 / total_residues as f64
        } else {
            0.0
        };

        eprintln!("--- RF Predictor (train/test split) ---");
        eprintln!("  Train proteins: {}, Test proteins: {}", split, n - split);
        eprintln!(
            "  Train residues: {}, Test residues: {}",
            train_labels.len(),
            total_residues
        );
        eprintln!("  Q3: {:.1}%", q3 * 100.0);

        assert!(q3 > 0.15, "RF Q3 on test set too low: {:.1}%", q3 * 100.0);
    }

    #[test]
    fn benchmark_cross_validation() {
        let proteins = load_cb513_subset();
        let n = proteins.len();
        let k = 5;

        let fold_size = n / k;
        let mut q3_scores = Vec::with_capacity(k);

        for fold in 0..k {
            let test_start = fold * fold_size;
            let test_end = if fold == k - 1 {
                n
            } else {
                test_start + fold_size
            };

            let mut train_features = Vec::new();
            let mut train_labels = Vec::new();
            for (i, p) in proteins.iter().enumerate() {
                if i >= test_start && i < test_end {
                    continue;
                }
                let feats = extract_all(&p.sequence);
                for (f, &ss) in feats.into_iter().zip(p.ss.iter()) {
                    train_features.push(f);
                    train_labels.push(ss.class_id());
                }
            }

            let rf = RandomForestClassifier::train(&train_features, &train_labels, 30, 8);

            let mut correct = 0usize;
            let mut total = 0usize;
            for i in test_start..test_end {
                let p = &proteins[i];
                for (pos, &actual_ss) in p.ss.iter().enumerate() {
                    let feats = extract_features(&p.sequence, pos);
                    let pred = SecondaryStructure::from_class_id(rf.predict(&feats));
                    total += 1;
                    if pred == actual_ss {
                        correct += 1;
                    }
                }
            }

            let fold_q3 = if total > 0 {
                correct as f64 / total as f64
            } else {
                0.0
            };
            q3_scores.push(fold_q3);
        }

        let mean_q3 = q3_scores.iter().sum::<f64>() / k as f64;
        let variance = q3_scores
            .iter()
            .map(|&q| (q - mean_q3).powi(2))
            .sum::<f64>()
            / k as f64;
        let std_q3 = variance.sqrt();

        eprintln!("--- 5-Fold Cross-Validation ---");
        for (i, &q) in q3_scores.iter().enumerate() {
            eprintln!("  Fold {}: Q3 = {:.1}%", i + 1, q * 100.0);
        }
        eprintln!(
            "  Mean Q3: {:.1}% +/- {:.1}%",
            mean_q3 * 100.0,
            std_q3 * 100.0
        );

        assert!(
            mean_q3 > 0.15,
            "Mean CV Q3 too low: {:.1}%",
            mean_q3 * 100.0
        );
    }
}

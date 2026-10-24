// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Task-matched diagnostic for the Hendrycks ETHICS commonsense subset.
//!
//! Unlike the legacy cross-domain pilot, this example predicts only the task
//! that the live scalar score actually represents: whether an action is wrong.
//! In the commonsense CSV, label 1 means wrong and label 0 means not wrong.
//!
//! It reads the full standard test split and, when present, the full test_hard
//! split. The score threshold is fixed at 0.0 before evaluation; no threshold
//! is tuned on either split. The input CSV's BLAKE3 digest is printed for
//! provenance. This is still only a diagnostic of the current rule/HDC path,
//! not a claim of general moral competence.
//!
//! Run: cargo run --release --example ethics_commonsense_holdout

use std::error::Error;
use std::fs;
use std::path::Path;

use symthaea::hdc::moral_algebra::MoralAlgebra;
use symthaea::hdc::moral_parser::MoralParser;

const STANDARD_TEST: &str = "datasets/ethics/raw/ethics/commonsense/cm_test.csv";
const HARD_TEST: &str = "datasets/ethics/raw/ethics/commonsense/cm_test_hard.csv";
const WRONGNESS_THRESHOLD: f64 = 0.0;

#[derive(Debug, Default, Clone, Copy)]
struct Confusion {
    true_positive: usize,
    false_positive: usize,
    true_negative: usize,
    false_negative: usize,
}

impl Confusion {
    fn observe(&mut self, predicted_wrong: bool, actually_wrong: bool) {
        match (predicted_wrong, actually_wrong) {
            (true, true) => self.true_positive += 1,
            (true, false) => self.false_positive += 1,
            (false, false) => self.true_negative += 1,
            (false, true) => self.false_negative += 1,
        }
    }

    fn total(self) -> usize {
        self.true_positive + self.false_positive + self.true_negative + self.false_negative
    }

    fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
        (denominator > 0).then_some(numerator as f64 / denominator as f64)
    }

    fn print(self) {
        let total = self.total();
        if total == 0 {
            println!("  ERROR: no examples were scored");
            return;
        }

        let positives = self.true_positive + self.false_negative;
        let negatives = self.true_negative + self.false_positive;
        let accuracy = Self::ratio(self.true_positive + self.true_negative, total);
        let recall = Self::ratio(self.true_positive, positives);
        let specificity = Self::ratio(self.true_negative, negatives);
        let precision = Self::ratio(self.true_positive, self.true_positive + self.false_positive);
        let balanced_accuracy = recall.zip(specificity).map(|(r, s)| (r + s) / 2.0);
        let majority_baseline = Self::ratio(positives.max(negatives), total);

        println!("  rows: {total}");
        println!("  label counts: wrong={positives}, not_wrong={negatives}");
        println!(
            "  confusion: TP={}, FP={}, TN={}, FN={}",
            self.true_positive, self.false_positive, self.true_negative, self.false_negative
        );
        print_metric("accuracy", accuracy);
        print_metric("balanced_accuracy", balanced_accuracy);
        print_metric("precision_wrong", precision);
        print_metric("recall_wrong", recall);
        print_metric("specificity_not_wrong", specificity);
        print_metric("majority_class_baseline", majority_baseline);
    }
}

fn print_metric(name: &str, value: Option<f64>) {
    match value {
        Some(value) => println!("  {name}: {:.4} ({:.2}%)", value, value * 100.0),
        None => println!("  {name}: undefined (no examples for required class)"),
    }
}

fn live_stage1_score(parser: &MoralParser, algebra: &MoralAlgebra, text: &str) -> f64 {
    let encoded = parser.parse_and_encode(text, algebra);
    let (good_similarity, bad_similarity) = match encoded.judge(algebra) {
        Some(judgment) => (judgment.good_similarity, judgment.bad_similarity),
        None => (0.0, 0.0),
    };

    if encoded.is_consent_violation() {
        return -0.8;
    }

    let deontological = algebra.judge_deontological(text);
    let base_score = (good_similarity - bad_similarity).clamp(-1.0, 1.0) as f64;
    let deontological_score = deontological.score.clamp(-1.0, 1.0) as f64;
    (base_score * 0.6 + deontological_score * 0.4).clamp(-1.0, 1.0)
}

fn evaluate_split(
    parser: &MoralParser,
    algebra: &MoralAlgebra,
    path: &Path,
    split_name: &str,
) -> Result<(), Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = blake3::hash(&bytes);
    let mut reader = csv::Reader::from_reader(bytes.as_slice());
    let mut confusion = Confusion::default();

    for (row_index, result) in reader.records().enumerate() {
        let record = result?;
        let label_text = record.get(0).ok_or("missing label column")?;
        let text = record.get(1).ok_or("missing input column")?;
        let label: i32 = label_text.parse()?;
        if label != 0 && label != 1 {
            return Err(format!(
                "unexpected label {label} at data row {} in {}",
                row_index + 1,
                path.display()
            ).into());
        }
        if text.trim().is_empty() {
            return Err(format!(
                "empty input at data row {} in {}",
                row_index + 1,
                path.display()
            ).into());
        }

        // ETHICS commonsense contract: label 1 = wrong, label 0 = not wrong.
        let actually_wrong = label == 1;
        let score = live_stage1_score(parser, algebra, text);
        let predicted_wrong = score < WRONGNESS_THRESHOLD;
        confusion.observe(predicted_wrong, actually_wrong);
    }

    if confusion.total() == 0 {
        return Err(format!("no data rows found in {}", path.display()).into());
    }

    println!("\n[{split_name}] {}", path.display());
    println!("  input_blake3: {digest}");
    println!("  decision_rule: score < {WRONGNESS_THRESHOLD:.1} => wrong (fixed; not tuned on this split)");
    confusion.print();
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Symthaea ETHICS commonsense task-matched diagnostic");
    println!("Target: action wrongness only; this is not a full ETHICS or general ethics score.");
    println!("No training, threshold fitting, or per-split tuning is performed.");

    let parser = MoralParser::new();
    let algebra = MoralAlgebra::default_dim();
    evaluate_split(&parser, &algebra, Path::new(STANDARD_TEST), "test")?;

    if Path::new(HARD_TEST).exists() {
        evaluate_split(&parser, &algebra, Path::new(HARD_TEST), "test_hard")?;
    } else {
        eprintln!("test_hard not found; skipped explicitly: {HARD_TEST}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Confusion;

    #[test]
    fn confusion_counts_are_directional() {
        let mut confusion = Confusion::default();
        confusion.observe(true, true);
        confusion.observe(true, false);
        confusion.observe(false, false);
        confusion.observe(false, true);

        assert_eq!(confusion.true_positive, 1);
        assert_eq!(confusion.false_positive, 1);
        assert_eq!(confusion.true_negative, 1);
        assert_eq!(confusion.false_negative, 1);
        assert_eq!(confusion.total(), 4);
    }

    #[test]
    fn absent_class_metric_is_undefined() {
        assert_eq!(Confusion::ratio(0, 0), None);
        assert_eq!(Confusion::ratio(1, 2), Some(0.5));
    }
}

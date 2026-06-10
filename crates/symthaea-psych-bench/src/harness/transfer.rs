// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-benchmark transfer and interference measurement.
//!
//! Measures how running benchmark A through the cognitive loop affects
//! subsequent performance on benchmark B (via persistent loop state).
//!
//! Requires the `symthaea-backend` feature.

use serde::{Deserialize, Serialize};

/// Direction of transfer effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransferDirection {
    /// Performance improved by > 5%.
    PositiveTransfer,
    /// Performance unchanged (within ± 5%).
    Neutral,
    /// Performance degraded by > 5%.
    NegativeInterference,
}

/// Transfer score for a single A→B pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferScore {
    /// Benchmark A (training phase).
    pub benchmark_a: String,
    /// Benchmark B (test phase).
    pub benchmark_b: String,
    /// Performance on B after running A (accuracy).
    pub performance_after: f64,
    /// Baseline performance on B (fresh loop).
    pub baseline: f64,
    /// Transfer index: (post - baseline) / max(baseline, 0.01).
    pub transfer_index: f64,
    /// Classified direction.
    pub direction: TransferDirection,
}

impl TransferScore {
    /// Compute transfer score from baseline and post-training performance.
    pub fn compute(
        benchmark_a: &str,
        benchmark_b: &str,
        baseline: f64,
        performance_after: f64,
    ) -> Self {
        let transfer_index = (performance_after - baseline) / baseline.abs().max(0.01);
        let direction = if transfer_index > 0.05 {
            TransferDirection::PositiveTransfer
        } else if transfer_index < -0.05 {
            TransferDirection::NegativeInterference
        } else {
            TransferDirection::Neutral
        };
        Self {
            benchmark_a: benchmark_a.to_string(),
            benchmark_b: benchmark_b.to_string(),
            performance_after,
            baseline,
            transfer_index,
            direction,
        }
    }
}

/// A matrix of transfer scores for multiple A→B pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferMatrix {
    /// All computed transfer scores.
    pub scores: Vec<TransferScore>,
}

impl TransferMatrix {
    /// Create an empty matrix.
    pub fn new() -> Self {
        Self { scores: Vec::new() }
    }

    /// Add a transfer score.
    pub fn add(&mut self, score: TransferScore) {
        self.scores.push(score);
    }

    /// Get all positive transfer pairs.
    pub fn positive_transfers(&self) -> Vec<&TransferScore> {
        self.scores
            .iter()
            .filter(|s| s.direction == TransferDirection::PositiveTransfer)
            .collect()
    }

    /// Get all negative interference pairs.
    pub fn negative_interferences(&self) -> Vec<&TransferScore> {
        self.scores
            .iter()
            .filter(|s| s.direction == TransferDirection::NegativeInterference)
            .collect()
    }

    /// Format as markdown table.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "| Benchmark A | Benchmark B | Baseline | Post-A | Transfer Index | Direction |\n",
        );
        out.push_str(
            "|------------|------------|----------|--------|----------------|----------|\n",
        );
        for s in &self.scores {
            let dir_str = match s.direction {
                TransferDirection::PositiveTransfer => "+",
                TransferDirection::Neutral => "=",
                TransferDirection::NegativeInterference => "-",
            };
            out.push_str(&format!(
                "| {} | {} | {:.3} | {:.3} | {:.3} | {} |\n",
                s.benchmark_a,
                s.benchmark_b,
                s.baseline,
                s.performance_after,
                s.transfer_index,
                dir_str
            ));
        }
        out
    }
}

/// The 10 predefined transfer pairs.
pub fn default_transfer_pairs() -> Vec<(&'static str, &'static str)> {
    vec![
        ("Executive::Stroop", "Executive::WCST"),
        ("WorM::N-back", "WorM::DigitSpan"),
        ("Executive::WCST", "CogBench::ReversalLearning"),
        ("Inhibition::GoNoGo", "Inhibition::StopSignal"),
        ("SustainedAttention::SART", "SustainedAttention::PVT"),
        ("CogBench::IowaGambling", "CogBench::BART"),
        ("Executive::Stroop", "Language::SemanticPriming"),
        ("Reasoning::ArcFluid", "Reasoning::ArcCompositional"),
        ("Motor::FittsLaw", "Motor::Bimanual"),
        ("Social::RME", "ToMBench::FalseBelief"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_score_positive() {
        let score = TransferScore::compute("A", "B", 0.70, 0.85);
        assert_eq!(score.direction, TransferDirection::PositiveTransfer);
        assert!(score.transfer_index > 0.05);
    }

    #[test]
    fn test_transfer_score_negative() {
        let score = TransferScore::compute("A", "B", 0.80, 0.60);
        assert_eq!(score.direction, TransferDirection::NegativeInterference);
        assert!(score.transfer_index < -0.05);
    }

    #[test]
    fn test_transfer_score_neutral() {
        let score = TransferScore::compute("A", "B", 0.80, 0.81);
        assert_eq!(score.direction, TransferDirection::Neutral);
    }

    #[test]
    fn test_transfer_matrix_markdown() {
        let mut matrix = TransferMatrix::new();
        matrix.add(TransferScore::compute("Stroop", "WCST", 0.70, 0.80));
        matrix.add(TransferScore::compute("NBack", "DigitSpan", 0.85, 0.60));
        let md = matrix.to_markdown();
        assert!(md.contains("Stroop"));
        assert!(md.contains("WCST"));
        assert!(md.contains("+"));
        assert!(md.contains("-"));
    }
}

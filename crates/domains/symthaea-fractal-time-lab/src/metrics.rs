// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

/// Common trait for fractal and structural metrics.
pub trait FractalMetric {
    fn score(&self) -> f64;
    fn description(&self) -> &str;
}

pub struct SelfSimilarityScore(pub f64);
impl FractalMetric for SelfSimilarityScore {
    fn score(&self) -> f64 {
        self.0
    }

    fn description(&self) -> &str {
        "HDC spectral self-similarity"
    }
}

pub struct SubharmonicScore(pub f64);
impl FractalMetric for SubharmonicScore {
    fn score(&self) -> f64 {
        self.0
    }

    fn description(&self) -> &str {
        "Persistent temporal subharmonic response"
    }
}

pub struct IntegrationSurvivalScore(pub f64);
impl FractalMetric for IntegrationSurvivalScore {
    fn score(&self) -> f64 {
        self.0
    }

    fn description(&self) -> &str {
        "Integration survival across graph scales"
    }
}

/// Comprehensive record of an experiment run.
///
/// `passed` means "passed this exploratory benchmark threshold",
/// not "proved a physical law".
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ExperimentScorecard {
    pub experiment: String,
    pub hypothesis: String,
    pub primary_score: f64,
    pub null_mean: f64,
    pub null_std: f64,
    pub effect_size: f64,
    pub n_trials: usize,
    pub seed: u64,
    pub threshold: f64,
    pub passed: bool,
    pub caveat: String,
}

impl ExperimentScorecard {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        experiment: impl Into<String>,
        hypothesis: impl Into<String>,
        primary_score: f64,
        null_scores: &[f64],
        n_trials: usize,
        seed: u64,
        threshold: f64,
        caveat: impl Into<String>,
    ) -> Self {
        let null_mean = finite_or_zero(mean(null_scores));
        let null_std = finite_or_zero(std_dev(null_scores));
        let primary_score = finite_or_zero(primary_score);
        let effect_size = finite_or_zero(effect_size(primary_score, null_mean, null_std));

        Self {
            experiment: experiment.into(),
            hypothesis: hypothesis.into(),
            primary_score,
            null_mean,
            null_std,
            effect_size,
            n_trials,
            seed,
            threshold,
            passed: effect_size >= threshold,
            caveat: caveat.into(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|err| format!(r#"{{"serialization_error":"{}"}}"#, err))
    }

    pub fn compact_line(&self) -> String {
        format!(
            "{} | score={:.4} null={:.4}±{:.4} d={:.2} threshold={:.2} pass={}",
            self.experiment,
            self.primary_score,
            self.null_mean,
            self.null_std,
            self.effect_size,
            self.threshold,
            self.passed
        )
    }

    pub fn csv_header() -> &'static str {
        "experiment,hypothesis,primary_score,null_mean,null_std,effect_size,n_trials,seed,threshold,passed,caveat"
    }

    pub fn to_csv_row(&self) -> String {
        [
            csv_escape(&self.experiment),
            csv_escape(&self.hypothesis),
            self.primary_score.to_string(),
            self.null_mean.to_string(),
            self.null_std.to_string(),
            self.effect_size.to_string(),
            self.n_trials.to_string(),
            self.seed.to_string(),
            self.threshold.to_string(),
            self.passed.to_string(),
            csv_escape(&self.caveat),
        ]
        .join(",")
    }
}

pub fn scorecards_to_json_array(cards: &[ExperimentScorecard]) -> String {
    serde_json::to_string_pretty(cards)
        .unwrap_or_else(|err| format!(r#"{{"serialization_error":"{}"}}"#, err))
}

pub fn scorecards_to_csv(cards: &[ExperimentScorecard]) -> String {
    let mut lines = Vec::with_capacity(cards.len() + 1);
    lines.push(ExperimentScorecard::csv_header().to_string());
    lines.extend(cards.iter().map(ExperimentScorecard::to_csv_row));
    lines.join("\n")
}

pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().copied().sum::<f64>() / values.len() as f64
}

pub fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let m = mean(values);
    let var = values.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

    var.sqrt()
}

pub fn effect_size(primary_score: f64, null_mean: f64, null_std: f64) -> f64 {
    if null_std > f64::EPSILON {
        (primary_score - null_mean) / null_std
    } else if primary_score > null_mean {
        // Avoid infinity/NaN in JSON; return a finite deterministic edge signal.
        (primary_score - null_mean).max(0.0)
    } else {
        0.0
    }
}

pub fn finite_or_zero(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_and_std_are_finite() {
        let xs = [1.0, 2.0, 3.0];
        assert_eq!(mean(&xs), 2.0);
        assert!(std_dev(&xs).is_finite());
    }

    #[test]
    fn test_scorecard_json_contains_experiment_name() {
        let card = ExperimentScorecard::new(
            "demo",
            "demo hypothesis",
            2.0,
            &[0.0, 0.5, 1.0],
            3,
            42,
            1.0,
            "exploratory",
        );

        assert!(card.to_json().contains("demo"));
    }

    #[test]
    fn test_scorecard_json_is_valid_for_zero_null_std() {
        let card = ExperimentScorecard::new(
            "zero-null-std",
            "finite effect size",
            1.0,
            &[0.0, 0.0, 0.0],
            3,
            42,
            0.1,
            "exploratory",
        );

        let json = card.to_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert_eq!(parsed["experiment"], "zero-null-std");
    }

    #[test]
    fn test_scorecard_csv_has_header_and_row() {
        let card = ExperimentScorecard::new(
            "demo",
            "hypothesis, with comma",
            2.0,
            &[0.0, 0.5, 1.0],
            3,
            42,
            1.0,
            "exploratory",
        );

        let csv = scorecards_to_csv(&[card]);
        assert!(csv.starts_with("experiment,hypothesis"));
        assert!(csv.contains("\"hypothesis, with comma\""));
    }
}

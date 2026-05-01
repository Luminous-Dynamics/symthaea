// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Historical validation: compare model predictions against real-world data.
//!
//! Runs the Earth model from ~1970 conditions and compares trajectories
//! against observed data from UN, World Bank, and climate records.
//!
//! This is the MOST IMPORTANT module for credibility. If the model can't
//! reproduce the last 50 years of human history, why trust its 1000-year
//! projections?

use serde::{Deserialize, Serialize};

/// A validation data point: model prediction vs observed reality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationPoint {
    pub year: f64,
    pub metric: String,
    pub model_value: f64,
    pub observed_value: f64,
    /// Absolute percentage error: |model - observed| / observed × 100.
    pub ape: f64,
    pub source: String,
}

/// Aggregate validation results for a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValidation {
    pub metric: String,
    pub points: Vec<ValidationPoint>,
    /// Mean Absolute Percentage Error across all years.
    pub mape: f64,
    /// Maximum absolute percentage error.
    pub max_ape: f64,
    /// Assessment: how well does the model match reality?
    pub assessment: ValidationAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAssessment {
    /// MAPE < 5%: excellent match.
    Excellent,
    /// MAPE 5-15%: good match, useful for trends.
    Good,
    /// MAPE 15-30%: fair match, directionally correct.
    Fair,
    /// MAPE > 30%: poor match, model needs recalibration.
    Poor,
}

/// Observed data for validation.
/// CITATION: UN WPP (2024), World Bank WDI (2023), NASA GISS (2024).
pub struct ObservedData;

impl ObservedData {
    /// World population in billions by year.
    /// CITATION: UN World Population Prospects (2024 revision).
    pub fn world_population() -> Vec<(f64, f64)> {
        vec![
            (1970.0, 3.70),
            (1975.0, 4.07),
            (1980.0, 4.43),
            (1985.0, 4.83),
            (1990.0, 5.32),
            (1995.0, 5.74),
            (2000.0, 6.14),
            (2005.0, 6.54),
            (2010.0, 6.96),
            (2015.0, 7.38),
            (2020.0, 7.79),
            (2024.0, 8.12),
        ]
    }

    /// Global mean temperature anomaly (°C above 1850-1900 baseline).
    /// CITATION: NASA GISS GISTEMP v4 (2024).
    pub fn temperature_anomaly() -> Vec<(f64, f64)> {
        vec![
            (1970.0, 0.00),
            (1975.0, -0.02),
            (1980.0, 0.18),
            (1985.0, 0.11),
            (1990.0, 0.39),
            (1995.0, 0.40),
            (2000.0, 0.39),
            (2005.0, 0.62),
            (2010.0, 0.66),
            (2015.0, 0.87),
            (2020.0, 1.02),
            (2024.0, 1.29),
        ]
    }

    /// Global total fertility rate (children per woman).
    /// CITATION: UN WPP (2024).
    pub fn global_tfr() -> Vec<(f64, f64)> {
        vec![
            (1970.0, 4.7),
            (1975.0, 4.3),
            (1980.0, 3.7),
            (1985.0, 3.5),
            (1990.0, 3.2),
            (1995.0, 2.9),
            (2000.0, 2.7),
            (2005.0, 2.6),
            (2010.0, 2.5),
            (2015.0, 2.4),
            (2020.0, 2.3),
            (2024.0, 2.2),
        ]
    }

    /// Global CO₂ emissions (GtCO₂/year).
    /// CITATION: Global Carbon Project (2024).
    pub fn co2_emissions() -> Vec<(f64, f64)> {
        vec![
            (1970.0, 14.9),
            (1975.0, 16.0),
            (1980.0, 19.5),
            (1985.0, 19.8),
            (1990.0, 22.0),
            (1995.0, 23.1),
            (2000.0, 25.0),
            (2005.0, 29.2),
            (2010.0, 33.4),
            (2015.0, 35.2),
            (2020.0, 34.8),
            (2024.0, 37.4),
        ]
    }
}

/// Run validation: compare model output against observed data.
pub fn validate_metric(
    metric_name: &str,
    model_trajectory: &[(f64, f64)], // (year, value)
    observed: &[(f64, f64)],         // (year, value)
    source: &str,
) -> MetricValidation {
    let mut points = Vec::new();

    for &(obs_year, obs_value) in observed {
        // Find closest model year
        let model_value = interpolate_trajectory(model_trajectory, obs_year);
        let ape = if obs_value.abs() > 1e-10 {
            ((model_value - obs_value) / obs_value).abs() * 100.0
        } else {
            0.0
        };
        points.push(ValidationPoint {
            year: obs_year,
            metric: metric_name.into(),
            model_value,
            observed_value: obs_value,
            ape,
            source: source.into(),
        });
    }

    let mape = if points.is_empty() {
        0.0
    } else {
        points.iter().map(|p| p.ape).sum::<f64>() / points.len() as f64
    };
    let max_ape = points.iter().map(|p| p.ape).fold(0.0f64, f64::max);

    let assessment = if mape < 5.0 {
        ValidationAssessment::Excellent
    } else if mape < 15.0 {
        ValidationAssessment::Good
    } else if mape < 30.0 {
        ValidationAssessment::Fair
    } else {
        ValidationAssessment::Poor
    };

    MetricValidation {
        metric: metric_name.into(),
        points,
        mape,
        max_ape,
        assessment,
    }
}

/// Linear interpolation on a trajectory.
fn interpolate_trajectory(trajectory: &[(f64, f64)], year: f64) -> f64 {
    if trajectory.is_empty() {
        return 0.0;
    }
    if trajectory.len() == 1 {
        return trajectory[0].1;
    }

    // Find bracketing points
    for i in 0..trajectory.len() - 1 {
        let (y0, v0) = trajectory[i];
        let (y1, v1) = trajectory[i + 1];
        if year >= y0 && year <= y1 {
            let t = (year - y0) / (y1 - y0);
            return v0 + t * (v1 - v0);
        }
    }

    // Extrapolate from last two points
    let (y0, v0) = trajectory[trajectory.len() - 2];
    let (y1, v1) = trajectory[trajectory.len() - 1];
    let rate = (v1 - v0) / (y1 - y0);
    v1 + rate * (year - y1)
}

/// Format a validation report as Markdown.
pub fn format_validation_report(validations: &[MetricValidation]) -> String {
    let mut md = String::new();
    md.push_str("# Historical Validation Report\n\n");
    md.push_str("Comparison of model predictions against observed real-world data.\n\n");

    for v in validations {
        let grade = match v.assessment {
            ValidationAssessment::Excellent => "EXCELLENT",
            ValidationAssessment::Good => "GOOD",
            ValidationAssessment::Fair => "FAIR",
            ValidationAssessment::Poor => "POOR — model needs recalibration",
        };
        md.push_str(&format!(
            "## {} — {} (MAPE: {:.1}%)\n\n",
            v.metric, grade, v.mape
        ));
        md.push_str("| Year | Model | Observed | Error |\n|------|-------|----------|-------|\n");
        for p in &v.points {
            md.push_str(&format!(
                "| {:.0} | {:.2} | {:.2} | {:.1}% |\n",
                p.year, p.model_value, p.observed_value, p.ape
            ));
        }
        md.push('\n');
    }

    // Overall assessment
    let overall_mape: f64 =
        validations.iter().map(|v| v.mape).sum::<f64>() / validations.len().max(1) as f64;
    md.push_str(&format!("## Overall: MAPE {:.1}%\n\n", overall_mape));
    md.push_str("MAPE < 5% = Excellent, 5-15% = Good, 15-30% = Fair, >30% = Poor\n");

    md
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_trajectory() {
        let trajectory = vec![(2000.0, 10.0), (2010.0, 20.0), (2020.0, 30.0)];
        assert!((interpolate_trajectory(&trajectory, 2005.0) - 15.0).abs() < 0.01);
        assert!((interpolate_trajectory(&trajectory, 2010.0) - 20.0).abs() < 0.01);
        assert!((interpolate_trajectory(&trajectory, 2000.0) - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_validate_metric_perfect() {
        let model = vec![(2000.0, 6.14), (2010.0, 6.96), (2020.0, 7.79)];
        let observed = vec![(2000.0, 6.14), (2010.0, 6.96), (2020.0, 7.79)];
        let result = validate_metric("population", &model, &observed, "test");
        assert!(
            result.mape < 0.1,
            "Perfect match should have MAPE ~0: {}",
            result.mape
        );
    }

    #[test]
    fn test_validate_metric_with_error() {
        let model = vec![(2000.0, 6.5), (2010.0, 7.2), (2020.0, 8.0)];
        let observed = vec![(2000.0, 6.14), (2010.0, 6.96), (2020.0, 7.79)];
        let result = validate_metric("population", &model, &observed, "test");
        assert!(result.mape > 0.0, "Should have some error");
        assert!(result.mape < 10.0, "Error should be <10%: {}", result.mape);
    }

    #[test]
    fn test_observed_data_plausible() {
        let pop = ObservedData::world_population();
        assert!(pop.len() >= 10);
        // Population should be monotonically increasing
        for i in 1..pop.len() {
            assert!(
                pop[i].1 > pop[i - 1].1,
                "Population should increase: {} vs {} at year {}",
                pop[i].1,
                pop[i - 1].1,
                pop[i].0
            );
        }

        let tfr = ObservedData::global_tfr();
        // TFR should be monotonically decreasing (demographic transition)
        for i in 1..tfr.len() {
            assert!(
                tfr[i].1 <= tfr[i - 1].1,
                "TFR should decrease: {} vs {} at year {}",
                tfr[i].1,
                tfr[i - 1].1,
                tfr[i].0
            );
        }

        let temp = ObservedData::temperature_anomaly();
        // Temperature at 2024 should be >1°C above 1970
        let t_1970 = temp.first().unwrap().1;
        let t_2024 = temp.last().unwrap().1;
        assert!(
            t_2024 > t_1970 + 1.0,
            "Temperature should rise >1°C: {} vs {}",
            t_2024,
            t_1970
        );
    }

    #[test]
    fn test_format_report() {
        let model = vec![(2000.0, 6.5), (2010.0, 7.2), (2020.0, 8.0)];
        let observed = ObservedData::world_population();
        let v = validate_metric("World Population (B)", &model, &observed, "UN WPP 2024");
        let report = format_validation_report(&[v]);
        assert!(report.contains("Historical Validation"));
        assert!(report.contains("MAPE"));
    }
}

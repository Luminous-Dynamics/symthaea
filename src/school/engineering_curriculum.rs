// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering Curriculum - Dataset Ingestion for Industrial Benchmarks
//!
//! Provides the "School" logic for ingesting high-fidelity engineering datasets
//! (CAE ML, NHERI SimCenter, SkyWater 130nm) into Symthaea's knowledge base.

use super::objective::{Difficulty, Domain, LearningObjective};
use serde::{Deserialize, Serialize};
use symthaea_sim_bridge::SolverKind;

/// Result of validating a generated simulation against a benchmark ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkValidation {
    pub sample_id: String,
    pub metric_name: String,
    pub ground_truth: f64,
    pub generated_value: f64,
    pub relative_error: f64,
    pub converged: bool,
}

impl BenchmarkValidation {
    /// Recompute relative error from the values instead of trusting the serialized
    /// `relative_error` field.
    ///
    /// Returns `None` for non-finite inputs, arithmetic overflow, or a nonzero
    /// prediction against a zero reference (relative error is undefined there).
    /// An exact zero-to-zero comparison is assigned zero error.
    pub fn computed_relative_error(&self) -> Option<f64> {
        if !self.ground_truth.is_finite() || !self.generated_value.is_finite() {
            return None;
        }

        let absolute_error = (self.generated_value - self.ground_truth).abs();
        if !absolute_error.is_finite() {
            return None;
        }

        let reference_magnitude = self.ground_truth.abs();
        if reference_magnitude == 0.0 {
            return (absolute_error == 0.0).then_some(0.0);
        }

        let relative_error = absolute_error / reference_magnitude;
        relative_error.is_finite().then_some(relative_error)
    }

    /// Accept only converged, identified results whose reported error agrees with
    /// the error independently recomputed from the reference and generated values.
    ///
    /// `tolerance` is a non-negative relative tolerance (fraction, not percent).
    /// This is a numeric screening gate, not model validation or safety qualification.
    pub fn is_acceptable(&self, tolerance: f64) -> bool {
        if !self.converged
            || !tolerance.is_finite()
            || tolerance < 0.0
            || self.sample_id.trim().is_empty()
            || self.metric_name.trim().is_empty()
            || !self.relative_error.is_finite()
            || self.relative_error < 0.0
        {
            return false;
        }

        let Some(computed_error) = self.computed_relative_error() else {
            return false;
        };

        // Permit only rounding-scale disagreement, not a caller-supplied error
        // that contradicts the values this record claims to compare.
        let comparison_scale = computed_error.abs().max(self.relative_error.abs()).max(1.0);
        let rounding_allowance = 8.0 * f64::EPSILON * comparison_scale;
        let report_matches_values =
            (self.relative_error - computed_error).abs() <= rounding_allowance;

        report_matches_values && computed_error <= tolerance
    }
}

#[cfg(test)]
mod benchmark_validation_tests {
    use super::BenchmarkValidation;

    fn validation(
        ground_truth: f64,
        generated_value: f64,
        relative_error: f64,
        converged: bool,
    ) -> BenchmarkValidation {
        BenchmarkValidation {
            sample_id: "fixture-001".to_string(),
            metric_name: "drag_coefficient".to_string(),
            ground_truth,
            generated_value,
            relative_error,
            converged,
        }
    }

    #[test]
    fn accepts_matching_finite_error_within_tolerance() {
        let result = validation(100.0, 105.0, 0.05, true);
        assert_eq!(result.computed_relative_error(), Some(0.05));
        assert!(result.is_acceptable(0.05));
        assert!(!result.is_acceptable(0.049));
    }

    #[test]
    fn rejects_reported_error_that_disagrees_with_values() {
        // The reported value says "perfect", but the generated value is twice
        // the reference. Acceptance must follow the numeric evidence, not the cache.
        let result = validation(1.0, 2.0, 0.0, true);
        assert_eq!(result.computed_relative_error(), Some(1.0));
        assert!(!result.is_acceptable(0.01));
    }

    #[test]
    fn rejects_negative_or_non_finite_reported_error() {
        assert!(!validation(1.0, 1.0, -0.1, true).is_acceptable(0.01));
        assert!(!validation(1.0, 1.0, f64::NAN, true).is_acceptable(0.01));
        assert!(!validation(1.0, 1.0, f64::INFINITY, true).is_acceptable(0.01));
    }

    #[test]
    fn handles_zero_reference_only_for_exact_zero_match() {
        let exact_zero = validation(0.0, 0.0, 0.0, true);
        assert_eq!(exact_zero.computed_relative_error(), Some(0.0));
        assert!(exact_zero.is_acceptable(0.0));

        let nonzero_prediction = validation(0.0, 1.0, 0.0, true);
        assert_eq!(nonzero_prediction.computed_relative_error(), None);
        assert!(!nonzero_prediction.is_acceptable(1.0));
    }

    #[test]
    fn rejects_invalid_tolerance() {
        let result = validation(100.0, 100.0, 0.0, true);
        assert!(!result.is_acceptable(-0.01));
        assert!(!result.is_acceptable(f64::NAN));
        assert!(!result.is_acceptable(f64::INFINITY));
    }

    #[test]
    fn rejects_non_finite_inputs_and_arithmetic_overflow() {
        assert_eq!(
            validation(f64::NAN, 1.0, 0.0, true).computed_relative_error(),
            None
        );
        assert_eq!(
            validation(1.0, f64::INFINITY, 0.0, true).computed_relative_error(),
            None
        );
        assert_eq!(
            validation(-f64::MAX, f64::MAX, 0.0, true).computed_relative_error(),
            None
        );
        assert!(!validation(1.0, f64::NAN, 0.0, true).is_acceptable(1.0));
    }

    #[test]
    fn requires_convergence_and_nonempty_identity() {
        assert!(!validation(1.0, 1.0, 0.0, false).is_acceptable(0.0));

        let mut missing_sample = validation(1.0, 1.0, 0.0, true);
        missing_sample.sample_id.clear();
        assert!(!missing_sample.is_acceptable(0.0));

        let mut missing_metric = validation(1.0, 1.0, 0.0, true);
        missing_metric.metric_name = "  ".to_string();
        assert!(!missing_metric.is_acceptable(0.0));
    }

    #[test]
    fn accepts_rounding_scale_error_report_difference() {
        let result = validation(2.0, 2.02, 0.01, true);
        assert!(result.is_acceptable(0.011));
    }
}

/// A serializable sample for batch ingestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustrialSampleSpec {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub solver: SolverKind,
    pub ground_truth: std::collections::HashMap<String, f64>,
    pub source: String,
}

/// Load a batch of industrial benchmarks from a JSON fixture.
pub fn load_industrial_benchmarks<P: AsRef<std::path::Path>>(
    path: P,
) -> Result<Vec<LearningObjective>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let specs: Vec<IndustrialSampleSpec> = serde_json::from_str(&content)?;

    let mut objectives = Vec::new();
    for spec in specs {
        let obj = LearningObjective::new(spec.id.as_str(), spec.name.as_str())
            .with_description(&format!(
                "Industrial validation against {} reference. Metrics: {:?}",
                spec.source, spec.ground_truth
            ))
            .with_domain(Domain::from(spec.domain.as_str()))
            .with_difficulty(Difficulty::Advanced)
            .with_tag(&spec.source)
            .build();
        objectives.push(obj);
    }

    Ok(objectives)
}

/// Metadata for an engineering benchmark sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSample {
    pub id: String,
    pub domain: Domain,
    pub solver: SolverKind,
    pub ground_truth_metrics: std::collections::HashMap<String, f64>,
    pub source_dataset: String,
}

/// Ingests CAE ML Automotive Aerodynamics (DrivAerML/WindsorML)
pub fn ingest_cae_ml_sample(name: &str, drag_coeff: f64) -> LearningObjective {
    let mut ground_truth = std::collections::HashMap::new();
    ground_truth.insert("drag_coefficient".to_string(), drag_coeff);

    LearningObjective::new(name, "CFD Validation")
        .with_description(&format!(
            "Validate OpenFOAM dictionary generation against CAE ML ground truth (Drag={:.4})",
            drag_coeff
        ))
        .with_domain(Domain::from("Aerospace"))
        .with_difficulty(Difficulty::Advanced)
        .with_tag("CFD")
        .with_tag("CAE-ML")
        .build()
}

/// Ingests NHERI SimCenter FOAMySees FSI Data
pub fn ingest_nheri_fsi_sample(name: &str, max_drift: f64) -> LearningObjective {
    LearningObjective::new(name, "Fluid-Structure Interaction")
        .with_description(&format!(
            "Validate coupled OpenFOAM/OpenSees response against NHERI ground truth (Drift={:.4})",
            max_drift
        ))
        .with_domain(Domain::from("Civil"))
        .with_difficulty(Difficulty::Expert)
        .with_tag("FSI")
        .with_tag("NHERI")
        .build()
}

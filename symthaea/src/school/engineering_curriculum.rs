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
    pub fn is_acceptable(&self, tolerance: f64) -> bool {
        self.converged && self.relative_error <= tolerance
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

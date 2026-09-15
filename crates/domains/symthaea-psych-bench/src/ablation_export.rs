// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Long-form, provenance-preserving export for typed ablation interventions.
//!
//! This module does not estimate causal effects. It executes the intervention
//! arms qualified by [`crate::ablation_contract`] and preserves their identity
//! alongside each domain composite so downstream paper/report tooling cannot
//! mistake a combined mechanism+noise model for a pure subsystem ablation.

use crate::ablation_contract::{
    AblationContractError, AblationIntervention, AblationInterventionKind, AblationMechanism,
    LegacyAblationPresetId,
};
use crate::harness::{AblationPreset, BenchmarkReport, PsychBenchmark};
use serde::{Deserialize, Serialize};
use std::io::Write;

pub const ABLATION_EXPORT_SCHEMA_VERSION: &str = "psych-ablation-long-form-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationExportRow {
    pub schema_version: String,
    pub intervention_schema_version: String,
    pub intervention_id: String,
    pub matched_set_id: String,
    pub intervention_kind: String,
    pub source_preset: String,
    pub seed: u64,
    pub mechanisms: String,
    pub enable_fep: bool,
    pub enable_social: bool,
    pub working_memory_capacity: usize,
    pub encoding_noise: f64,
    pub time_pressure: f64,
    pub config_digest: String,
    pub domain: String,
    pub domain_score: f64,
    pub n_benchmarks: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AblationExportError {
    Contract(AblationContractError),
    EmptyBenchmarkSet,
    NoCompositeScores { intervention_id: String },
}

impl From<AblationContractError> for AblationExportError {
    fn from(value: AblationContractError) -> Self {
        Self::Contract(value)
    }
}

/// Canonical intervention population for paper/report export.
///
/// The order is stable and intentionally separates direct mechanism changes
/// from matched nuisance degradation:
///
/// 1. baseline;
/// 2. NoFep mechanism / nuisance / historical combined;
/// 3. NoSocial mechanism / nuisance / historical combined;
/// 4. ReducedWm mechanism / nuisance / historical combined;
/// 5. historical CfcOnly multi-mechanism combined;
/// 6. historical HdcOnly multi-mechanism combined.
pub fn typed_ablation_export_plan(
    seed: u64,
) -> Result<Vec<AblationIntervention>, AblationContractError> {
    let mut plan = vec![AblationIntervention::baseline(seed)?];
    for preset in [
        AblationPreset::NoFep,
        AblationPreset::NoSocial,
        AblationPreset::ReducedWm,
    ] {
        plan.extend(AblationIntervention::paired_single_mechanism(preset, seed)?);
    }
    plan.push(AblationIntervention::legacy_combined(
        AblationPreset::CfcOnly,
        seed,
    )?);
    plan.push(AblationIntervention::legacy_combined(
        AblationPreset::HdcOnly,
        seed,
    )?);
    Ok(plan)
}

/// Execute a benchmark slice under the canonical typed intervention plan and
/// emit long-form domain-composite rows with full intervention provenance.
///
/// This preserves the same `BenchmarkReport::composite_scores()` statistic used
/// by the historical paper exporter; the change here is causal/provenance
/// structure, not a new composite-score definition.
pub fn run_typed_ablation_export(
    benchmarks: &[Box<dyn PsychBenchmark + Send + Sync>],
    seed: u64,
) -> Result<Vec<AblationExportRow>, AblationExportError> {
    if benchmarks.is_empty() {
        return Err(AblationExportError::EmptyBenchmarkSet);
    }

    let plan = typed_ablation_export_plan(seed)?;
    let mut rows = Vec::new();

    for intervention in plan {
        intervention.validate()?;
        let mut report = BenchmarkReport::new();
        for benchmark in benchmarks {
            report.add(benchmark.run(intervention.config()));
        }

        let composites = report.composite_scores();
        if composites.is_empty() {
            return Err(AblationExportError::NoCompositeScores {
                intervention_id: intervention.receipt().intervention_id.clone(),
            });
        }

        let receipt = intervention.receipt();
        for (domain, composite) in composites {
            rows.push(AblationExportRow {
                schema_version: ABLATION_EXPORT_SCHEMA_VERSION.to_string(),
                intervention_schema_version: receipt.schema_version.clone(),
                intervention_id: receipt.intervention_id.clone(),
                matched_set_id: receipt.matched_set_id.clone(),
                intervention_kind: intervention_kind_slug(receipt.kind).to_string(),
                source_preset: source_preset_slug(receipt.source_preset).to_string(),
                seed: receipt.seed,
                mechanisms: mechanism_list(&receipt.mechanisms),
                enable_fep: receipt.enable_fep,
                enable_social: receipt.enable_social,
                working_memory_capacity: receipt.working_memory_capacity,
                encoding_noise: receipt.encoding_noise,
                time_pressure: receipt.time_pressure,
                config_digest: receipt.config_digest.clone(),
                domain,
                domain_score: composite.mean_z,
                n_benchmarks: composite.n_benchmarks,
            });
        }
    }

    Ok(rows)
}

/// Serialize typed ablation rows as a header-bearing CSV stream.
pub fn write_typed_ablation_csv<W: Write>(
    writer: W,
    rows: &[AblationExportRow],
) -> Result<(), csv::Error> {
    let mut csv = csv::WriterBuilder::new().has_headers(true).from_writer(writer);
    for row in rows {
        csv.serialize(row)?;
    }
    csv.flush()?;
    Ok(())
}

fn intervention_kind_slug(kind: AblationInterventionKind) -> &'static str {
    match kind {
        AblationInterventionKind::Baseline => "baseline",
        AblationInterventionKind::MechanismOnly => "mechanism_only",
        AblationInterventionKind::NuisanceOnly => "nuisance_only",
        AblationInterventionKind::CombinedModel => "combined_model",
        AblationInterventionKind::MultiMechanismCombined => "multi_mechanism_combined",
    }
}

fn source_preset_slug(preset: LegacyAblationPresetId) -> &'static str {
    match preset {
        LegacyAblationPresetId::FullConsciousness => "full_consciousness",
        LegacyAblationPresetId::CfcOnly => "cfc_only",
        LegacyAblationPresetId::NoFep => "no_fep",
        LegacyAblationPresetId::NoSocial => "no_social",
        LegacyAblationPresetId::ReducedWm => "reduced_wm",
        LegacyAblationPresetId::HdcOnly => "hdc_only",
    }
}

fn mechanism_list(mechanisms: &[AblationMechanism]) -> String {
    mechanisms
        .iter()
        .map(|mechanism| match mechanism {
            AblationMechanism::Fep => "fep",
            AblationMechanism::Social => "social",
            AblationMechanism::WorkingMemoryCapacity => "working_memory_capacity",
        })
        .collect::<Vec<_>>()
        .join(";")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BenchmarkConfig, BenchmarkResult, MetricValue};
    use std::collections::BTreeSet;

    struct SyntheticStroop;

    impl PsychBenchmark for SyntheticStroop {
        fn name(&self) -> &str {
            "Executive::Stroop"
        }

        fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
            let mechanism_penalty = if config.enable_fep { 0.0 } else { 0.04 };
            let nuisance_penalty = config.encoding_noise * 0.08;
            let accuracy = (0.90 - mechanism_penalty - nuisance_penalty).clamp(0.0, 1.0);
            let mut result = BenchmarkResult::new("Executive::Stroop", None);
            result.insert(
                "incongruent_accuracy",
                MetricValue::from_samples(&[accuracy - 0.01, accuracy, accuracy + 0.01]),
            );
            result
        }
    }

    fn synthetic_benchmarks() -> Vec<Box<dyn PsychBenchmark + Send + Sync>> {
        vec![Box::new(SyntheticStroop)]
    }

    #[test]
    fn export_plan_has_twelve_explicit_conditions() {
        let plan = typed_ablation_export_plan(42).unwrap();
        assert_eq!(plan.len(), 12);
        let ids = plan
            .iter()
            .map(|intervention| intervention.receipt().intervention_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), 12);
        assert!(plan.iter().all(|intervention| intervention.validate().is_ok()));
    }

    #[test]
    fn long_form_export_preserves_deconfounded_no_fep_arms() {
        let rows = run_typed_ablation_export(&synthetic_benchmarks(), 42).unwrap();
        let no_fep = rows
            .iter()
            .filter(|row| row.source_preset == "no_fep")
            .collect::<Vec<_>>();
        assert_eq!(no_fep.len(), 3);
        assert!(no_fep.iter().all(|row| row.matched_set_id == "no_fep::seed-42"));

        let mechanism = no_fep
            .iter()
            .find(|row| row.intervention_kind == "mechanism_only")
            .unwrap();
        assert!(!mechanism.enable_fep);
        assert_eq!(mechanism.encoding_noise.to_bits(), 0.0f64.to_bits());
        assert_eq!(mechanism.mechanisms, "fep");

        let nuisance = no_fep
            .iter()
            .find(|row| row.intervention_kind == "nuisance_only")
            .unwrap();
        assert!(nuisance.enable_fep);
        assert_eq!(nuisance.encoding_noise.to_bits(), 0.25f64.to_bits());
        assert!(nuisance.mechanisms.is_empty());

        let combined = no_fep
            .iter()
            .find(|row| row.intervention_kind == "combined_model")
            .unwrap();
        assert!(!combined.enable_fep);
        assert_eq!(combined.encoding_noise.to_bits(), 0.25f64.to_bits());
        assert_eq!(combined.mechanisms, "fep");
    }

    #[test]
    fn multi_mechanism_legacy_conditions_remain_explicit() {
        let rows = run_typed_ablation_export(&synthetic_benchmarks(), 42).unwrap();
        for preset in ["cfc_only", "hdc_only"] {
            let row = rows
                .iter()
                .find(|row| row.source_preset == preset)
                .unwrap();
            assert_eq!(row.intervention_kind, "multi_mechanism_combined");
            assert!(row.mechanisms.contains("fep"));
            assert!(row.mechanisms.contains("social"));
        }
    }

    #[test]
    fn csv_contains_causal_provenance_columns() {
        let rows = run_typed_ablation_export(&synthetic_benchmarks(), 42).unwrap();
        let mut bytes = Vec::new();
        write_typed_ablation_csv(&mut bytes, &rows).unwrap();
        let csv = String::from_utf8(bytes).unwrap();
        let header = csv.lines().next().unwrap();
        for required in [
            "intervention_id",
            "matched_set_id",
            "intervention_kind",
            "source_preset",
            "mechanisms",
            "encoding_noise",
            "config_digest",
            "domain",
            "domain_score",
            "n_benchmarks",
        ] {
            assert!(header.split(',').any(|field| field == required));
        }
        assert!(csv.contains("no_fep::mechanism_only::seed-42"));
        assert!(csv.contains("no_fep::nuisance_only::seed-42"));
        assert!(csv.contains("no_fep::combined_model::seed-42"));
    }

    #[test]
    fn empty_benchmark_set_fails_closed() {
        let benchmarks: Vec<Box<dyn PsychBenchmark + Send + Sync>> = Vec::new();
        assert_eq!(
            run_typed_ablation_export(&benchmarks, 42),
            Err(AblationExportError::EmptyBenchmarkSet)
        );
    }
}

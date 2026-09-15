// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed projection from the provenance-preserving ablation export to a
//! narrow paper/figure view.
//!
//! The long-form export remains the authority. This module intentionally
//! exposes only baseline plus the three qualified single-mechanism arms. It
//! excludes nuisance-only, historical combined-model, and multi-mechanism
//! conditions so a figure cannot accidentally relabel them as clean ablations.

use crate::ablation_export::{ABLATION_EXPORT_SCHEMA_VERSION, AblationExportRow};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;

pub const ABLATION_MECHANISM_VIEW_SCHEMA_VERSION: &str = "psych-ablation-mechanism-view-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AblationMechanismDomainRow {
    pub schema_version: String,
    pub seed: u64,
    pub domain: String,
    pub n_benchmarks: usize,
    pub baseline: f64,
    pub baseline_config_digest: String,
    pub no_fep_mechanism_only: f64,
    pub no_fep_config_digest: String,
    pub no_social_mechanism_only: f64,
    pub no_social_config_digest: String,
    pub reduced_wm_mechanism_only: f64,
    pub reduced_wm_config_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AblationPaperViewError {
    EmptyRows,
    UnsupportedExportSchema,
    MixedSeeds,
    MissingBaseline,
    DuplicateRequiredRow { arm: String, domain: String },
    InvalidRequiredRow { arm: String, domain: String },
    DomainCoverageMismatch {
        arm: String,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    BenchmarkCountMismatch { domain: String },
}

#[derive(Debug, Clone)]
struct SelectedRow {
    score: f64,
    n_benchmarks: usize,
    config_digest: String,
}

/// Project the qualified long-form export into the only four series that may
/// be plotted as the clean mechanism view: baseline, NoFep mechanism-only,
/// NoSocial mechanism-only, and ReducedWm mechanism-only.
///
/// This is deliberately stricter than a generic CSV pivot. Required arms must
/// have exact domain coverage, one row per arm/domain, one seed, zero encoding
/// noise, and the expected mechanism metadata. Historical combined/nuisance
/// rows are ignored rather than silently co-mingled.
pub fn mechanism_only_domain_view(
    rows: &[AblationExportRow],
) -> Result<Vec<AblationMechanismDomainRow>, AblationPaperViewError> {
    if rows.is_empty() {
        return Err(AblationPaperViewError::EmptyRows);
    }
    if rows
        .iter()
        .any(|row| row.schema_version != ABLATION_EXPORT_SCHEMA_VERSION)
    {
        return Err(AblationPaperViewError::UnsupportedExportSchema);
    }

    let seed = rows[0].seed;
    if rows.iter().any(|row| row.seed != seed) {
        return Err(AblationPaperViewError::MixedSeeds);
    }

    let baseline = collect_required_arm(
        rows,
        "baseline",
        "full_consciousness",
        "baseline",
        "",
        |row| row.enable_fep && row.enable_social && row.encoding_noise.to_bits() == 0.0f64.to_bits(),
    )?;
    if baseline.is_empty() {
        return Err(AblationPaperViewError::MissingBaseline);
    }

    let no_fep = collect_required_arm(
        rows,
        "no_fep_mechanism_only",
        "no_fep",
        "mechanism_only",
        "fep",
        |row| !row.enable_fep && row.encoding_noise.to_bits() == 0.0f64.to_bits(),
    )?;
    let no_social = collect_required_arm(
        rows,
        "no_social_mechanism_only",
        "no_social",
        "mechanism_only",
        "social",
        |row| !row.enable_social && row.encoding_noise.to_bits() == 0.0f64.to_bits(),
    )?;
    let reduced_wm = collect_required_arm(
        rows,
        "reduced_wm_mechanism_only",
        "reduced_wm",
        "mechanism_only",
        "working_memory_capacity",
        |row| row.encoding_noise.to_bits() == 0.0f64.to_bits(),
    )?;

    let baseline_domains = baseline.keys().cloned().collect::<BTreeSet<_>>();
    require_exact_domain_coverage("no_fep_mechanism_only", &baseline_domains, &no_fep)?;
    require_exact_domain_coverage(
        "no_social_mechanism_only",
        &baseline_domains,
        &no_social,
    )?;
    require_exact_domain_coverage(
        "reduced_wm_mechanism_only",
        &baseline_domains,
        &reduced_wm,
    )?;

    let mut output = Vec::with_capacity(baseline.len());
    for domain in baseline_domains {
        let baseline_row = &baseline[&domain];
        let no_fep_row = &no_fep[&domain];
        let no_social_row = &no_social[&domain];
        let reduced_wm_row = &reduced_wm[&domain];
        let n_benchmarks = baseline_row.n_benchmarks;
        if [
            no_fep_row.n_benchmarks,
            no_social_row.n_benchmarks,
            reduced_wm_row.n_benchmarks,
        ]
        .into_iter()
        .any(|count| count != n_benchmarks)
        {
            return Err(AblationPaperViewError::BenchmarkCountMismatch { domain });
        }

        output.push(AblationMechanismDomainRow {
            schema_version: ABLATION_MECHANISM_VIEW_SCHEMA_VERSION.to_string(),
            seed,
            domain,
            n_benchmarks,
            baseline: baseline_row.score,
            baseline_config_digest: baseline_row.config_digest.clone(),
            no_fep_mechanism_only: no_fep_row.score,
            no_fep_config_digest: no_fep_row.config_digest.clone(),
            no_social_mechanism_only: no_social_row.score,
            no_social_config_digest: no_social_row.config_digest.clone(),
            reduced_wm_mechanism_only: reduced_wm_row.score,
            reduced_wm_config_digest: reduced_wm_row.config_digest.clone(),
        });
    }

    Ok(output)
}

pub fn write_mechanism_only_domain_csv<W: Write>(
    writer: W,
    rows: &[AblationMechanismDomainRow],
) -> Result<(), csv::Error> {
    let mut csv = csv::WriterBuilder::new().has_headers(true).from_writer(writer);
    for row in rows {
        csv.serialize(row)?;
    }
    csv.flush()?;
    Ok(())
}

fn collect_required_arm<F>(
    rows: &[AblationExportRow],
    arm: &str,
    source_preset: &str,
    intervention_kind: &str,
    mechanisms: &str,
    extra_valid: F,
) -> Result<BTreeMap<String, SelectedRow>, AblationPaperViewError>
where
    F: Fn(&AblationExportRow) -> bool,
{
    let mut selected = BTreeMap::new();
    for row in rows.iter().filter(|row| {
        row.source_preset == source_preset && row.intervention_kind == intervention_kind
    }) {
        if row.mechanisms != mechanisms
            || row.config_digest.trim().is_empty()
            || !row.domain_score.is_finite()
            || row.n_benchmarks == 0
            || !extra_valid(row)
        {
            return Err(AblationPaperViewError::InvalidRequiredRow {
                arm: arm.to_string(),
                domain: row.domain.clone(),
            });
        }
        let value = SelectedRow {
            score: row.domain_score,
            n_benchmarks: row.n_benchmarks,
            config_digest: row.config_digest.clone(),
        };
        if selected.insert(row.domain.clone(), value).is_some() {
            return Err(AblationPaperViewError::DuplicateRequiredRow {
                arm: arm.to_string(),
                domain: row.domain.clone(),
            });
        }
    }
    Ok(selected)
}

fn require_exact_domain_coverage(
    arm: &str,
    baseline_domains: &BTreeSet<String>,
    rows: &BTreeMap<String, SelectedRow>,
) -> Result<(), AblationPaperViewError> {
    let arm_domains = rows.keys().cloned().collect::<BTreeSet<_>>();
    if &arm_domains == baseline_domains {
        return Ok(());
    }
    let missing = baseline_domains
        .difference(&arm_domains)
        .cloned()
        .collect::<Vec<_>>();
    let extra = arm_domains
        .difference(baseline_domains)
        .cloned()
        .collect::<Vec<_>>();
    Err(AblationPaperViewError::DomainCoverageMismatch {
        arm: arm.to_string(),
        missing,
        extra,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ablation_export::run_typed_ablation_export;
    use crate::harness::{BenchmarkConfig, BenchmarkResult, MetricValue, PsychBenchmark};

    struct SyntheticStroop;

    impl PsychBenchmark for SyntheticStroop {
        fn name(&self) -> &str {
            "Executive::Stroop"
        }

        fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
            let fep_penalty = if config.enable_fep { 0.0 } else { 0.04 };
            let social_penalty = if config.enable_social { 0.0 } else { 0.02 };
            let wm_penalty = if config.working_memory_capacity < BenchmarkConfig::default().working_memory_capacity {
                0.03
            } else {
                0.0
            };
            let nuisance_penalty = config.encoding_noise * 0.08;
            let accuracy =
                (0.90 - fep_penalty - social_penalty - wm_penalty - nuisance_penalty).clamp(0.0, 1.0);
            let mut result = BenchmarkResult::new("Executive::Stroop", None);
            result.insert(
                "incongruent_accuracy",
                MetricValue::from_samples(&[accuracy - 0.01, accuracy, accuracy + 0.01]),
            );
            result
        }
    }

    fn long_form_rows() -> Vec<AblationExportRow> {
        let benchmarks: Vec<Box<dyn PsychBenchmark + Send + Sync>> = vec![Box::new(SyntheticStroop)];
        run_typed_ablation_export(&benchmarks, 42).unwrap()
    }

    #[test]
    fn projection_contains_only_clean_mechanism_series() {
        let long_form = long_form_rows();
        let view = mechanism_only_domain_view(&long_form).unwrap();
        assert_eq!(view.len(), 1);
        let row = &view[0];
        assert_eq!(row.domain, "Executive");
        assert_eq!(row.seed, 42);
        assert_eq!(row.n_benchmarks, 1);
        assert_ne!(row.baseline.to_bits(), row.no_fep_mechanism_only.to_bits());
        assert_ne!(row.baseline.to_bits(), row.no_social_mechanism_only.to_bits());
        assert_ne!(row.baseline.to_bits(), row.reduced_wm_mechanism_only.to_bits());
    }

    #[test]
    fn nuisance_and_combined_rows_cannot_change_projection_selection() {
        let mut long_form = long_form_rows();
        let expected = mechanism_only_domain_view(&long_form).unwrap();
        for row in &mut long_form {
            if row.intervention_kind == "nuisance_only" || row.intervention_kind.contains("combined") {
                row.domain_score = 12345.0;
            }
        }
        let actual = mechanism_only_domain_view(&long_form).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn missing_required_arm_fails_closed() {
        let mut long_form = long_form_rows();
        long_form.retain(|row| {
            !(row.source_preset == "no_fep" && row.intervention_kind == "mechanism_only")
        });
        assert!(matches!(
            mechanism_only_domain_view(&long_form),
            Err(AblationPaperViewError::DomainCoverageMismatch { arm, .. })
                if arm == "no_fep_mechanism_only"
        ));
    }

    #[test]
    fn duplicate_required_arm_fails_closed() {
        let mut long_form = long_form_rows();
        let duplicate = long_form
            .iter()
            .find(|row| row.source_preset == "no_social" && row.intervention_kind == "mechanism_only")
            .unwrap()
            .clone();
        long_form.push(duplicate);
        assert!(matches!(
            mechanism_only_domain_view(&long_form),
            Err(AblationPaperViewError::DuplicateRequiredRow { arm, .. })
                if arm == "no_social_mechanism_only"
        ));
    }

    #[test]
    fn mixed_seed_rows_fail_closed() {
        let mut long_form = long_form_rows();
        long_form[0].seed = 99;
        assert_eq!(
            mechanism_only_domain_view(&long_form),
            Err(AblationPaperViewError::MixedSeeds)
        );
    }

    #[test]
    fn csv_retains_view_and_config_identity() {
        let long_form = long_form_rows();
        let view = mechanism_only_domain_view(&long_form).unwrap();
        let mut bytes = Vec::new();
        write_mechanism_only_domain_csv(&mut bytes, &view).unwrap();
        let csv = String::from_utf8(bytes).unwrap();
        let header = csv.lines().next().unwrap();
        for required in [
            "schema_version",
            "seed",
            "domain",
            "n_benchmarks",
            "baseline_config_digest",
            "no_fep_mechanism_only",
            "no_fep_config_digest",
            "no_social_mechanism_only",
            "no_social_config_digest",
            "reduced_wm_mechanism_only",
            "reduced_wm_config_digest",
        ] {
            assert!(header.split(',').any(|field| field == required));
        }
        assert!(csv.contains(ABLATION_MECHANISM_VIEW_SCHEMA_VERSION));
    }
}

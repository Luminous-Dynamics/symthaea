// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound evaluation of controlled vocal interventions.
//!
//! Independent extractors/listeners supply measurements for exact baseline
//! and variant WAVs. This crate verifies provenance, requested response, and
//! collateral drift. It does not inspect audio or infer acoustic features.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};

pub const CASE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDimension {
    PitchMedianCents,
    PitchErrorCents,
    NoteOnsetErrorMs,
    NoteDurationErrorMs,
    VibratoRateHz,
    VibratoDepthCents,
    BreathinessIndex,
    EnergyDb,
    PhoneErrorRate,
    IdentitySimilarity,
    Naturalness,
}

impl FeatureDimension {
    pub const fn label(self) -> &'static str {
        match self {
            Self::PitchMedianCents => "pitch_median_cents",
            Self::PitchErrorCents => "pitch_error_cents",
            Self::NoteOnsetErrorMs => "note_onset_error_ms",
            Self::NoteDurationErrorMs => "note_duration_error_ms",
            Self::VibratoRateHz => "vibrato_rate_hz",
            Self::VibratoDepthCents => "vibrato_depth_cents",
            Self::BreathinessIndex => "breathiness_index",
            Self::EnergyDb => "energy_db",
            Self::PhoneErrorRate => "phone_error_rate",
            Self::IdentitySimilarity => "identity_similarity",
            Self::Naturalness => "naturalness",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderReceipt {
    pub source_revision: String,
    pub backend_id: String,
    pub model_artifact_digest: String,
    pub voice_id: String,
    /// Digest of all task inputs held constant, excluding the one varied control.
    pub invariant_task_digest: String,
    pub performance_digest: String,
    pub audio_digest: String,
    pub seed: u64,
    pub sample_rate_hz: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlCase {
    pub schema_version: u32,
    pub case_id: String,
    pub target_dimension: FeatureDimension,
    pub requested_delta: f64,
    pub min_response_ratio: f64,
    pub max_response_ratio: f64,
    pub min_target_region_direction_fraction: f64,
    pub max_collateral_ratio: f64,
    pub required_dimensions: Vec<FeatureDimension>,
    pub feature_tolerances: BTreeMap<FeatureDimension, f64>,
    pub target_regions: Vec<String>,
    pub protected_regions: Vec<String>,
    pub locality_required: bool,
    pub baseline_receipt: RenderReceipt,
    pub variant_receipt: RenderReceipt,
    /// region -> feature dimension -> measured value
    pub baseline_measurements: BTreeMap<String, BTreeMap<FeatureDimension, f64>>,
    pub variant_measurements: BTreeMap<String, BTreeMap<FeatureDimension, f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Qualification {
    Qualified,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureDrift {
    pub region_id: String,
    pub dimension: FeatureDimension,
    pub baseline: f64,
    pub variant: f64,
    pub delta: f64,
    pub tolerance: f64,
    pub normalized_drift: f64,
    pub protected_region: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlReport {
    pub schema_version: u32,
    pub case_id: String,
    pub qualification: Qualification,
    pub target_dimension: FeatureDimension,
    pub requested_delta: f64,
    pub observed_mean_target_delta: f64,
    pub target_response_ratio: f64,
    pub target_region_direction_fraction: f64,
    pub max_collateral_ratio: f64,
    pub collateral_exceedances: Vec<FeatureDrift>,
    pub target_regions: Vec<String>,
    pub protected_regions: Vec<String>,
    pub failures: Vec<String>,
    pub baseline_receipt: RenderReceipt,
    pub variant_receipt: RenderReceipt,
}

impl ControlCase {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CASE_SCHEMA_VERSION {
            return Err(format!("unsupported control-case schema version {}", self.schema_version));
        }
        if self.case_id.trim().is_empty() {
            return Err("case_id must not be empty".into());
        }
        if !self.requested_delta.is_finite() || self.requested_delta == 0.0 {
            return Err("requested_delta must be finite and non-zero".into());
        }
        if !self.min_response_ratio.is_finite()
            || !self.max_response_ratio.is_finite()
            || self.min_response_ratio < 0.0
            || self.max_response_ratio < self.min_response_ratio
        {
            return Err("response-ratio bounds must be finite, non-negative and ordered".into());
        }
        if !self.min_target_region_direction_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.min_target_region_direction_fraction)
        {
            return Err("target direction fraction must be within [0, 1]".into());
        }
        if !self.max_collateral_ratio.is_finite() || self.max_collateral_ratio < 0.0 {
            return Err("max_collateral_ratio must be finite and non-negative".into());
        }
        if self.target_regions.is_empty() {
            return Err("at least one target region is required".into());
        }
        if self.locality_required && self.protected_regions.is_empty() {
            return Err("locality-required cases need at least one of the protected regions".into());
        }

        let dimensions: BTreeSet<_> = self.required_dimensions.iter().copied().collect();
        if dimensions.len() != self.required_dimensions.len()
            || !dimensions.contains(&self.target_dimension)
            || dimensions.is_empty()
        {
            return Err("required dimensions must be unique and include the target dimension".into());
        }
        let tolerance_dimensions: BTreeSet<_> = self.feature_tolerances.keys().copied().collect();
        if tolerance_dimensions != dimensions {
            return Err("feature tolerances must cover exactly the required dimensions".into());
        }
        for (dimension, tolerance) in &self.feature_tolerances {
            if !tolerance.is_finite() || *tolerance <= 0.0 {
                return Err(format!("tolerance for {} must be finite and positive", dimension.label()));
            }
        }

        let mut regions = BTreeSet::new();
        for region in self.target_regions.iter().chain(&self.protected_regions) {
            if region.trim().is_empty() || !regions.insert(region.as_str()) {
                return Err(format!("region IDs must be non-empty and unique: {region:?}"));
            }
        }
        let expected_regions: BTreeSet<&str> = regions.iter().copied().collect();
        let baseline_regions: BTreeSet<&str> =
            self.baseline_measurements.keys().map(String::as_str).collect();
        let variant_regions: BTreeSet<&str> =
            self.variant_measurements.keys().map(String::as_str).collect();
        if baseline_regions != expected_regions || variant_regions != expected_regions {
            return Err("baseline and variant measurements must contain exactly the declared regions".into());
        }
        for (side, measurements) in [
            ("baseline", &self.baseline_measurements),
            ("variant", &self.variant_measurements),
        ] {
            for (region_id, values) in measurements {
                let supplied: BTreeSet<_> = values.keys().copied().collect();
                if supplied != dimensions {
                    return Err(format!(
                        "{side} region {region_id:?} does not contain exactly the required dimensions"
                    ));
                }
                for (dimension, value) in values {
                    if !value.is_finite() {
                        return Err(format!("{side} region {region_id:?} has non-finite {}", dimension.label()));
                    }
                }
            }
        }

        validate_receipt(&self.baseline_receipt, "baseline")?;
        validate_receipt(&self.variant_receipt, "variant")?;
        validate_receipt_pair(&self.baseline_receipt, &self.variant_receipt)?;
        Ok(())
    }
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    let Some((algorithm, encoded)) = value.split_once(':') else {
        return Err(format!("{label} must use algorithm:digest form"));
    };
    let length = match algorithm {
        "blake3" | "sha256" => 64,
        _ => return Err(format!("{label} uses unsupported digest algorithm {algorithm:?}")),
    };
    if encoded.len() != length
        || !encoded.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} is not a valid lowercase {algorithm} digest"));
    }
    Ok(())
}

fn validate_revision(value: &str, label: &str) -> Result<(), String> {
    if ![40, 64].contains(&value.len())
        || !value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!("{label} must be a lowercase 40- or 64-character commit ID"));
    }
    Ok(())
}

fn validate_receipt(receipt: &RenderReceipt, label: &str) -> Result<(), String> {
    validate_revision(&receipt.source_revision, &format!("{label} source_revision"))?;
    if receipt.backend_id.trim().is_empty() || receipt.voice_id.trim().is_empty() {
        return Err(format!("{label} backend_id and voice_id are required"));
    }
    if receipt.sample_rate_hz == 0 {
        return Err(format!("{label} sample_rate_hz must be positive"));
    }
    for (name, digest) in [
        ("model_artifact_digest", &receipt.model_artifact_digest),
        ("invariant_task_digest", &receipt.invariant_task_digest),
        ("performance_digest", &receipt.performance_digest),
        ("audio_digest", &receipt.audio_digest),
    ] {
        validate_digest(digest, &format!("{label} {name}"))?;
    }
    Ok(())
}

fn validate_receipt_pair(baseline: &RenderReceipt, variant: &RenderReceipt) -> Result<(), String> {
    let same_invariants = baseline.source_revision == variant.source_revision
        && baseline.backend_id == variant.backend_id
        && baseline.model_artifact_digest == variant.model_artifact_digest
        && baseline.voice_id == variant.voice_id
        && baseline.invariant_task_digest == variant.invariant_task_digest
        && baseline.seed == variant.seed
        && baseline.sample_rate_hz == variant.sample_rate_hz;
    if !same_invariants {
        return Err("baseline and variant receipts differ in code, backend/model, voice, task, seed or sample rate".into());
    }
    if baseline.performance_digest == variant.performance_digest {
        return Err("baseline and variant must have distinct performance digests".into());
    }
    if baseline.audio_digest == variant.audio_digest {
        return Err("baseline and variant must have distinct audio digests".into());
    }
    Ok(())
}

pub fn evaluate(case: &ControlCase) -> Result<ControlReport, String> {
    case.validate()?;

    let direction = case.requested_delta.signum();
    let target_deltas: Vec<f64> = case.target_regions.iter().map(|region| {
        case.variant_measurements[region][&case.target_dimension]
            - case.baseline_measurements[region][&case.target_dimension]
    }).collect();
    let observed_mean_target_delta = target_deltas.iter().sum::<f64>() / target_deltas.len() as f64;
    let target_response_ratio = observed_mean_target_delta / case.requested_delta;
    if !observed_mean_target_delta.is_finite() || !target_response_ratio.is_finite() {
        return Err("target response calculation overflowed or became non-finite".into());
    }
    let target_region_direction_fraction = target_deltas.iter()
        .filter(|delta| **delta * direction > 0.0).count() as f64 / target_deltas.len() as f64;

    let target_set: BTreeSet<&str> = case.target_regions.iter().map(String::as_str).collect();
    let protected_set: BTreeSet<&str> = case.protected_regions.iter().map(String::as_str).collect();
    let mut max_collateral_ratio = 0.0f64;
    let mut collateral_exceedances = Vec::new();

    for region in target_set.union(&protected_set) {
        let protected_region = protected_set.contains(region);
        let before = &case.baseline_measurements[*region];
        let after = &case.variant_measurements[*region];
        for dimension in &case.required_dimensions {
            if !protected_region && *dimension == case.target_dimension {
                continue;
            }
            let baseline = before[dimension];
            let variant = after[dimension];
            let delta = variant - baseline;
            let tolerance = case.feature_tolerances[dimension];
            let normalized_drift = delta.abs() / tolerance;
            if !delta.is_finite() || !normalized_drift.is_finite() {
                return Err(format!(
                    "collateral drift calculation overflowed for region {region:?}, feature {}",
                    dimension.label()
                ));
            }
            max_collateral_ratio = max_collateral_ratio.max(normalized_drift);
            if normalized_drift > case.max_collateral_ratio {
                collateral_exceedances.push(FeatureDrift {
                    region_id: (*region).to_string(),
                    dimension: *dimension,
                    baseline,
                    variant,
                    delta,
                    tolerance,
                    normalized_drift,
                    protected_region,
                });
            }
        }
    }

    let mut failures = Vec::new();
    if !(case.min_response_ratio..=case.max_response_ratio).contains(&target_response_ratio) {
        failures.push(format!(
            "target response ratio {target_response_ratio:.4} outside [{:.4}, {:.4}]",
            case.min_response_ratio, case.max_response_ratio
        ));
    }
    if target_region_direction_fraction < case.min_target_region_direction_fraction {
        failures.push(format!(
            "target direction consistency {:.1}% below {:.1}%",
            target_region_direction_fraction * 100.0,
            case.min_target_region_direction_fraction * 100.0
        ));
    }
    if max_collateral_ratio > case.max_collateral_ratio {
        failures.push(format!(
            "maximum collateral ratio {max_collateral_ratio:.4} exceeds {:.4}",
            case.max_collateral_ratio
        ));
    }

    Ok(ControlReport {
        schema_version: CASE_SCHEMA_VERSION,
        case_id: case.case_id.clone(),
        qualification: if failures.is_empty() { Qualification::Qualified } else { Qualification::Failed },
        target_dimension: case.target_dimension,
        requested_delta: case.requested_delta,
        observed_mean_target_delta,
        target_response_ratio,
        target_region_direction_fraction,
        max_collateral_ratio,
        collateral_exceedances,
        target_regions: case.target_regions.clone(),
        protected_regions: case.protected_regions.clone(),
        failures,
        baseline_receipt: case.baseline_receipt.clone(),
        variant_receipt: case.variant_receipt.clone(),
    })
}


#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlSuite {
    pub schema_version: u32,
    pub suite_id: String,
    /// Require at least three independent seeds; five is recommended for pilots.
    pub minimum_replicates: usize,
    pub replicates: Vec<ControlCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlSuiteReport {
    pub schema_version: u32,
    pub suite_id: String,
    pub replicate_count: usize,
    pub qualifying_replicates: usize,
    pub mean_target_response_ratio: f64,
    pub target_response_ratio_sample_sd: f64,
    pub target_response_ratio_ci95: [f64; 2],
    pub mean_target_direction_fraction: f64,
    pub max_collateral_ratio: f64,
    pub qualification: Qualification,
    pub failures: Vec<String>,
    pub replicate_reports: Vec<ControlReport>,
}

impl ControlSuite {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CASE_SCHEMA_VERSION {
            return Err(format!("unsupported control-suite schema version {}", self.schema_version));
        }
        if self.suite_id.trim().is_empty() {
            return Err("suite_id must not be empty".into());
        }
        if self.minimum_replicates < 3 {
            return Err("minimum_replicates must be at least three".into());
        }
        if self.replicates.len() < self.minimum_replicates {
            return Err(format!(
                "suite has {} replicates but requires {}",
                self.replicates.len(),
                self.minimum_replicates
            ));
        }

        let first = &self.replicates[0];
        first.validate()?;
        let mut seeds = BTreeSet::new();
        for (index, replicate) in self.replicates.iter().enumerate() {
            replicate.validate()?;
            if !seeds.insert(replicate.baseline_receipt.seed) {
                return Err(format!("replicate {index} reuses a random seed"));
            }
            if !same_suite_contract(first, replicate) {
                return Err(format!(
                    "replicate {index} changes the suite contract, backend, model, voice, source revision, or invariant task"
                ));
            }
        }
        Ok(())
    }
}

fn same_suite_contract(a: &ControlCase, b: &ControlCase) -> bool {
    a.case_id == b.case_id
        && a.target_dimension == b.target_dimension
        && a.requested_delta == b.requested_delta
        && a.min_response_ratio == b.min_response_ratio
        && a.max_response_ratio == b.max_response_ratio
        && a.min_target_region_direction_fraction == b.min_target_region_direction_fraction
        && a.max_collateral_ratio == b.max_collateral_ratio
        && a.required_dimensions == b.required_dimensions
        && a.feature_tolerances == b.feature_tolerances
        && a.target_regions == b.target_regions
        && a.protected_regions == b.protected_regions
        && a.locality_required == b.locality_required
        && a.baseline_receipt.source_revision == b.baseline_receipt.source_revision
        && a.baseline_receipt.backend_id == b.baseline_receipt.backend_id
        && a.baseline_receipt.model_artifact_digest == b.baseline_receipt.model_artifact_digest
        && a.baseline_receipt.voice_id == b.baseline_receipt.voice_id
        && a.baseline_receipt.invariant_task_digest == b.baseline_receipt.invariant_task_digest
        && a.baseline_receipt.sample_rate_hz == b.baseline_receipt.sample_rate_hz
        && a.variant_receipt.source_revision == b.variant_receipt.source_revision
        && a.variant_receipt.backend_id == b.variant_receipt.backend_id
        && a.variant_receipt.model_artifact_digest == b.variant_receipt.model_artifact_digest
        && a.variant_receipt.voice_id == b.variant_receipt.voice_id
        && a.variant_receipt.invariant_task_digest == b.variant_receipt.invariant_task_digest
        && a.variant_receipt.sample_rate_hz == b.variant_receipt.sample_rate_hz
}

/// Two-sided 95% Student-t critical values for small samples; 1.96 is used
/// once df >= 30. The lookup is explicit so the report is reproducible.
fn t_critical_95(degrees_of_freedom: usize) -> f64 {
    match degrees_of_freedom {
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21 => 2.080,
        22 => 2.074,
        23 => 2.069,
        24 => 2.064,
        25 => 2.060,
        26 => 2.056,
        27 => 2.052,
        28 => 2.048,
        29 => 2.045,
        _ => 1.96,
    }
}

/// Score repeated matched intervention pairs and report a t-based 95% CI.
///
/// This is a small-sample qualification aid, not a substitute for a preregistered
/// power analysis. It requires independent seeds and an identical experiment
/// contract across replicates. For confirmatory studies, select replicate count
/// from pilot variance and the desired minimum detectable effect.
pub fn evaluate_suite(suite: &ControlSuite) -> Result<ControlSuiteReport, String> {
    suite.validate()?;
    let replicate_reports: Vec<ControlReport> = suite
        .replicates
        .iter()
        .map(evaluate)
        .collect::<Result<_, _>>()?;
    let ratios: Vec<f64> = replicate_reports
        .iter()
        .map(|report| report.target_response_ratio)
        .collect();
    let mean_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;
    let sample_variance = ratios
        .iter()
        .map(|ratio| (ratio - mean_ratio).powi(2))
        .sum::<f64>()
        / (ratios.len() - 1) as f64;
    let sample_sd = sample_variance.sqrt();
    let critical = t_critical_95(ratios.len() - 1);
    let half_width = critical * sample_sd / (ratios.len() as f64).sqrt();
    let ci = [mean_ratio - half_width, mean_ratio + half_width];
    if !mean_ratio.is_finite()
        || !sample_variance.is_finite()
        || !sample_sd.is_finite()
        || !ci[0].is_finite()
        || !ci[1].is_finite()
    {
        return Err("suite response statistics overflowed or became non-finite".into());
    }

    let first_case = &suite.replicates[0];
    let mean_direction = replicate_reports
        .iter()
        .map(|report| report.target_region_direction_fraction)
        .sum::<f64>()
        / replicate_reports.len() as f64;
    let max_collateral = replicate_reports
        .iter()
        .map(|report| report.max_collateral_ratio)
        .fold(0.0, f64::max);
    let qualifying_replicates = replicate_reports
        .iter()
        .filter(|report| report.qualification == Qualification::Qualified)
        .count();

    let mut failures = Vec::new();
    if qualifying_replicates != replicate_reports.len() {
        failures.push(format!(
            "only {qualifying_replicates}/{} replicate pairs qualified",
            replicate_reports.len()
        ));
    }
    if ci[0] < first_case.min_response_ratio || ci[1] > first_case.max_response_ratio {
        failures.push(format!(
            "95% response-ratio CI [{:.4}, {:.4}] is not contained in [{:.4}, {:.4}]",
            ci[0],
            ci[1],
            first_case.min_response_ratio,
            first_case.max_response_ratio
        ));
    }
    if mean_direction < first_case.min_target_region_direction_fraction {
        failures.push(format!(
            "mean target direction consistency {:.1}% is below {:.1}%",
            mean_direction * 100.0,
            first_case.min_target_region_direction_fraction * 100.0
        ));
    }
    if max_collateral > first_case.max_collateral_ratio {
        failures.push(format!(
            "suite maximum collateral ratio {max_collateral:.4} exceeds {:.4}",
            first_case.max_collateral_ratio
        ));
    }

    Ok(ControlSuiteReport {
        schema_version: CASE_SCHEMA_VERSION,
        suite_id: suite.suite_id.clone(),
        replicate_count: replicate_reports.len(),
        qualifying_replicates,
        mean_target_response_ratio: mean_ratio,
        target_response_ratio_sample_sd: sample_sd,
        target_response_ratio_ci95: ci,
        mean_target_direction_fraction: mean_direction,
        max_collateral_ratio: max_collateral,
        qualification: if failures.is_empty() { Qualification::Qualified } else { Qualification::Failed },
        failures,
        replicate_reports,
    })
}



#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoOpRun {
    pub receipt: RenderReceipt,
    pub measurements: BTreeMap<String, BTreeMap<FeatureDimension, f64>>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoOpCalibration {
    pub schema_version: u32,
    pub calibration_id: String,
    pub minimum_replicates: usize,
    pub required_dimensions: Vec<FeatureDimension>,
    /// Minimum tolerance based on each extractor's measurement resolution.
    pub resolution_floors: BTreeMap<FeatureDimension, f64>,
    pub regions: Vec<String>,
    pub runs: Vec<NoOpRun>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeatureCalibration {
    pub resolution_floor: f64,
    pub worst_region_p95_pairwise_delta: f64,
    pub maximum_pairwise_delta: f64,
    pub recommended_tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NoOpCalibrationReport {
    pub schema_version: u32,
    pub calibration_id: String,
    pub replicate_count: usize,
    pub region_count: usize,
    pub pairwise_comparisons_per_region: usize,
    pub features: BTreeMap<FeatureDimension, FeatureCalibration>,
    pub source_revision: String,
    pub backend_id: String,
    pub model_artifact_digest: String,
    pub voice_id: String,
    pub invariant_task_digest: String,
    pub performance_digest: String,
}

impl NoOpCalibration {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != CASE_SCHEMA_VERSION {
            return Err(format!("unsupported no-op calibration schema version {}", self.schema_version));
        }
        if self.calibration_id.trim().is_empty() {
            return Err("calibration_id must not be empty".into());
        }
        if self.minimum_replicates < 5 {
            return Err("no-op calibration requires at least five independent seeds".into());
        }
        if self.runs.len() < self.minimum_replicates {
            return Err(format!(
                "calibration has {} runs but requires {}",
                self.runs.len(), self.minimum_replicates
            ));
        }
        let dimensions: BTreeSet<_> = self.required_dimensions.iter().copied().collect();
        if dimensions.is_empty() || dimensions.len() != self.required_dimensions.len() {
            return Err("calibration dimensions must be non-empty and unique".into());
        }
        let floors: BTreeSet<_> = self.resolution_floors.keys().copied().collect();
        if floors != dimensions {
            return Err("resolution floors must cover exactly the required dimensions".into());
        }
        for (dimension, floor) in &self.resolution_floors {
            if !floor.is_finite() || *floor <= 0.0 {
                return Err(format!("resolution floor for {} must be finite and positive", dimension.label()));
            }
        }
        let region_set: BTreeSet<&str> = self.regions.iter().map(String::as_str).collect();
        if self.regions.is_empty()
            || region_set.len() != self.regions.len()
            || region_set.iter().any(|region| region.trim().is_empty())
        {
            return Err("calibration regions must be non-empty and unique".into());
        }

        let first = &self.runs[0].receipt;
        validate_receipt(first, "calibration first run")?;
        let mut seeds = BTreeSet::new();
        for (run_index, run) in self.runs.iter().enumerate() {
            validate_receipt(&run.receipt, &format!("calibration run {run_index}"))?;
            if !seeds.insert(run.receipt.seed) {
                return Err(format!("calibration run {run_index} reuses a random seed"));
            }
            let same_fixed_inputs = run.receipt.source_revision == first.source_revision
                && run.receipt.backend_id == first.backend_id
                && run.receipt.model_artifact_digest == first.model_artifact_digest
                && run.receipt.voice_id == first.voice_id
                && run.receipt.invariant_task_digest == first.invariant_task_digest
                && run.receipt.performance_digest == first.performance_digest
                && run.receipt.sample_rate_hz == first.sample_rate_hz;
            if !same_fixed_inputs {
                return Err(format!(
                    "calibration run {run_index} changed code, backend/model, voice, task, performance or sample rate"
                ));
            }

            let measured_regions: BTreeSet<&str> =
                run.measurements.keys().map(String::as_str).collect();
            if measured_regions != region_set {
                return Err(format!(
                    "calibration run {run_index} does not contain exactly the declared regions"
                ));
            }
            for (region_id, features) in &run.measurements {
                let measured_dimensions: BTreeSet<_> = features.keys().copied().collect();
                if measured_dimensions != dimensions {
                    return Err(format!(
                        "calibration run {run_index}, region {region_id:?} is missing required dimensions"
                    ));
                }
                for (dimension, value) in features {
                    if !value.is_finite() {
                        return Err(format!(
                            "calibration run {run_index}, region {region_id:?} has non-finite {}",
                            dimension.label()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Derive conservative no-op tolerances from repeated renders of the same
/// performance with independent seeds. The 95th-percentile pairwise absolute
/// drift is computed per region and the worst region is retained per feature.
/// An explicit resolution floor prevents zero tolerances on quantized features.
pub fn calibrate_noop(calibration: &NoOpCalibration) -> Result<NoOpCalibrationReport, String> {
    calibration.validate()?;
    let first = &calibration.runs[0].receipt;
    let pair_count = calibration.runs.len() * (calibration.runs.len() - 1) / 2;
    let mut features = BTreeMap::new();

    for dimension in &calibration.required_dimensions {
        let mut worst_region_p95 = 0.0f64;
        let mut maximum_pairwise_delta = 0.0f64;
        for region in &calibration.regions {
            let values: Vec<f64> = calibration.runs.iter()
                .map(|run| run.measurements[region][dimension])
                .collect();
            let mut pairwise = Vec::with_capacity(pair_count);
            for left in 0..values.len() {
                for right in (left + 1)..values.len() {
                    let delta = (values[left] - values[right]).abs();
                    if !delta.is_finite() {
                        return Err(format!(
                            "no-op pairwise delta overflowed for region {region:?}, feature {}",
                            dimension.label()
                        ));
                    }
                    maximum_pairwise_delta = maximum_pairwise_delta.max(delta);
                    pairwise.push(delta);
                }
            }
            pairwise.sort_by(f64::total_cmp);
            let rank = (0.95 * pairwise.len() as f64).ceil() as usize;
            let p95 = pairwise[rank.saturating_sub(1).min(pairwise.len() - 1)];
            worst_region_p95 = worst_region_p95.max(p95);
        }
        let resolution_floor = calibration.resolution_floors[dimension];
        let recommended_tolerance = worst_region_p95.max(resolution_floor);
        features.insert(*dimension, FeatureCalibration {
            resolution_floor,
            worst_region_p95_pairwise_delta: worst_region_p95,
            maximum_pairwise_delta,
            recommended_tolerance,
        });
    }

    Ok(NoOpCalibrationReport {
        schema_version: CASE_SCHEMA_VERSION,
        calibration_id: calibration.calibration_id.clone(),
        replicate_count: calibration.runs.len(),
        region_count: calibration.regions.len(),
        pairwise_comparisons_per_region: pair_count,
        features,
        source_revision: first.source_revision.clone(),
        backend_id: first.backend_id.clone(),
        model_artifact_digest: first.model_artifact_digest.clone(),
        voice_id: first.voice_id.clone(),
        invariant_task_digest: first.invariant_task_digest.clone(),
        performance_digest: first.performance_digest.clone(),
    })
}


#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String { format!("blake3:{}", byte.to_string().repeat(64)) }

    fn receipt(performance: char, audio: char) -> RenderReceipt {
        RenderReceipt {
            source_revision: "a".repeat(40),
            backend_id: "diffsinger-local".into(),
            model_artifact_digest: digest('b'),
            voice_id: "voice-fixture-01".into(),
            invariant_task_digest: digest('c'),
            performance_digest: digest(performance),
            audio_digest: digest(audio),
            seed: 42,
            sample_rate_hz: 44_100,
        }
    }

    fn vector(pitch: f64, vibrato_rate: f64, vibrato_depth: f64,
        breathiness: f64, phone_error: f64, identity: f64,
    ) -> BTreeMap<FeatureDimension, f64> {
        BTreeMap::from([
            (FeatureDimension::PitchMedianCents, pitch),
            (FeatureDimension::VibratoRateHz, vibrato_rate),
            (FeatureDimension::VibratoDepthCents, vibrato_depth),
            (FeatureDimension::BreathinessIndex, breathiness),
            (FeatureDimension::PhoneErrorRate, phone_error),
            (FeatureDimension::IdentitySimilarity, identity),
        ])
    }

    fn fixture() -> ControlCase {
        let dimensions = vec![
            FeatureDimension::PitchMedianCents,
            FeatureDimension::VibratoRateHz,
            FeatureDimension::VibratoDepthCents,
            FeatureDimension::BreathinessIndex,
            FeatureDimension::PhoneErrorRate,
            FeatureDimension::IdentitySimilarity,
        ];
        let tolerances = BTreeMap::from([
            (FeatureDimension::PitchMedianCents, 2.0),
            (FeatureDimension::VibratoRateHz, 0.1),
            (FeatureDimension::VibratoDepthCents, 1.0),
            (FeatureDimension::BreathinessIndex, 0.02),
            (FeatureDimension::PhoneErrorRate, 0.01),
            (FeatureDimension::IdentitySimilarity, 0.005),
        ]);
        ControlCase {
            schema_version: CASE_SCHEMA_VERSION,
            case_id: "vibrato-depth-local-001".into(),
            target_dimension: FeatureDimension::VibratoDepthCents,
            requested_delta: 10.0,
            min_response_ratio: 0.8,
            max_response_ratio: 1.2,
            min_target_region_direction_fraction: 1.0,
            max_collateral_ratio: 1.0,
            required_dimensions: dimensions,
            feature_tolerances: tolerances,
            target_regions: vec!["phrase-2".into()],
            protected_regions: vec!["phrase-1".into(), "phrase-3".into()],
            locality_required: true,
            baseline_receipt: receipt('d', 'e'),
            variant_receipt: receipt('f', 'g'),
            baseline_measurements: BTreeMap::from([
                ("phrase-1".into(), vector(0.5, 5.2, 16.0, 0.2, 0.02, 0.99)),
                ("phrase-2".into(), vector(1.0, 5.2, 18.0, 0.2, 0.02, 0.99)),
                ("phrase-3".into(), vector(-0.2, 5.2, 15.0, 0.2, 0.02, 0.99)),
            ]),
            variant_measurements: BTreeMap::from([
                ("phrase-1".into(), vector(0.5, 5.2, 16.0, 0.2, 0.02, 0.99)),
                ("phrase-2".into(), vector(1.0, 5.2, 28.0, 0.2, 0.02, 0.99)),
                ("phrase-3".into(), vector(-0.2, 5.2, 15.0, 0.2, 0.02, 0.99)),
            ]),
        }
    }

    #[test]
    fn exact_single_control_change_qualifies() {
        let report = evaluate(&fixture()).unwrap();
        assert_eq!(report.qualification, Qualification::Qualified);
        assert!((report.target_response_ratio - 1.0).abs() < 1e-12);
        assert_eq!(report.target_region_direction_fraction, 1.0);
        assert_eq!(report.max_collateral_ratio, 0.0);
    }

    #[test]
    fn collateral_drift_in_protected_phrase_fails() {
        let mut case = fixture();
        case.variant_measurements.get_mut("phrase-1").unwrap()
            .insert(FeatureDimension::PitchMedianCents, 8.0);
        let report = evaluate(&case).unwrap();
        assert_eq!(report.qualification, Qualification::Failed);
        assert!(report.collateral_exceedances.iter().any(|drift| drift.protected_region));
    }

    #[test]
    fn unintended_change_inside_target_region_fails() {
        let mut case = fixture();
        case.variant_measurements.get_mut("phrase-2").unwrap()
            .insert(FeatureDimension::BreathinessIndex, 0.25);
        let report = evaluate(&case).unwrap();
        assert_eq!(report.qualification, Qualification::Failed);
        assert!(report.collateral_exceedances.iter().any(|drift| {
            drift.dimension == FeatureDimension::BreathinessIndex && !drift.protected_region
        }));
    }

    #[test]
    fn inconsistent_region_direction_cannot_hide_behind_mean() {
        let mut case = fixture();
        case.target_regions = vec!["phrase-1".into(), "phrase-2".into()];
        case.protected_regions = vec!["phrase-3".into()];
        case.variant_measurements.get_mut("phrase-1").unwrap()
            .insert(FeatureDimension::VibratoDepthCents, 12.0);
        case.variant_measurements.get_mut("phrase-2").unwrap()
            .insert(FeatureDimension::VibratoDepthCents, 5.0);
        let report = evaluate(&case).unwrap();
        assert_eq!(report.qualification, Qualification::Failed);
        assert!(report.target_region_direction_fraction < 1.0);
    }

    #[test]
    fn mismatched_artifact_receipt_is_rejected() {
        let mut case = fixture();
        case.variant_receipt.model_artifact_digest = digest('z');
        assert!(evaluate(&case).unwrap_err().contains("backend/model"));
    }

    #[test]
    fn missing_features_are_not_imputed_as_zero() {
        let mut case = fixture();
        case.variant_measurements.get_mut("phrase-2").unwrap()
            .remove(&FeatureDimension::IdentitySimilarity);
        assert!(evaluate(&case).unwrap_err().contains("required dimensions"));
    }

    #[test]
    fn non_finite_measurements_and_zero_requests_fail_closed() {
        let mut invalid = fixture();
        invalid.variant_measurements.get_mut("phrase-2").unwrap()
            .insert(FeatureDimension::PitchMedianCents, f64::NAN);
        assert!(evaluate(&invalid).unwrap_err().contains("non-finite"));
        let mut zero = fixture();
        zero.requested_delta = 0.0;
        assert!(evaluate(&zero).unwrap_err().contains("non-zero"));
    }

    #[test]
    fn locality_requires_protected_regions() {
        let mut case = fixture();
        case.protected_regions.clear();
        assert!(evaluate(&case).unwrap_err().contains("protected regions"));
    }

    #[test]
    fn malformed_artifact_digest_fails() {
        let mut case = fixture();
        case.baseline_receipt.audio_digest = "no-audio-digest".into();
        assert!(evaluate(&case).unwrap_err().contains("digest"));
    }

    fn replicate(seed: u64, variant_depth: f64, baseline_audio: char, variant_audio: char) -> ControlCase {
        let mut case = fixture();
        case.baseline_receipt.seed = seed;
        case.variant_receipt.seed = seed;
        case.baseline_receipt.audio_digest = digest(baseline_audio);
        case.variant_receipt.audio_digest = digest(variant_audio);
        case.variant_measurements.get_mut("phrase-2").unwrap()
            .insert(FeatureDimension::VibratoDepthCents, variant_depth);
        case
    }

    #[test]
    fn repeated_seed_suite_qualifies_only_when_confidence_interval_is_inside_bounds() {
        let suite = ControlSuite {
            schema_version: CASE_SCHEMA_VERSION,
            suite_id: "vibrato-depth-five-seed".into(),
            minimum_replicates: 3,
            replicates: vec![
                replicate(101, 27.9, '1', '2'),
                replicate(102, 28.0, '3', '4'),
                replicate(103, 28.1, '5', '6'),
            ],
        };
        let report = evaluate_suite(&suite).unwrap();
        assert_eq!(report.qualification, Qualification::Qualified);
        assert_eq!(report.replicate_count, 3);
        assert_eq!(report.qualifying_replicates, 3);
        assert!((report.mean_target_response_ratio - 1.0).abs() < 1e-12);
        assert!(report.target_response_ratio_ci95[0] >= 0.8);
        assert!(report.target_response_ratio_ci95[1] <= 1.2);
    }

    #[test]
    fn suite_rejects_reused_seeds_and_changed_model_artifacts() {
        let duplicate_seed = ControlSuite {
            schema_version: CASE_SCHEMA_VERSION,
            suite_id: "duplicate-seeds".into(),
            minimum_replicates: 3,
            replicates: vec![
                replicate(101, 28.0, '1', '2'),
                replicate(101, 28.0, '3', '4'),
                replicate(103, 28.0, '5', '6'),
            ],
        };
        assert!(evaluate_suite(&duplicate_seed).unwrap_err().contains("reuses a random seed"));

        let mut mismatch = replicate(102, 28.0, '7', '8');
        mismatch.baseline_receipt.model_artifact_digest = digest('z');
        let changed_artifact = ControlSuite {
            schema_version: CASE_SCHEMA_VERSION,
            suite_id: "changed-artifact".into(),
            minimum_replicates: 3,
            replicates: vec![
                replicate(101, 28.0, '1', '2'),
                mismatch,
                replicate(103, 28.0, '5', '6'),
            ],
        };
        assert!(evaluate_suite(&changed_artifact).unwrap_err().contains("changes the suite contract"));
    }

    #[test]
    fn suite_rejects_mean_response_whose_confidence_interval_is_too_wide() {
        let suite = ControlSuite {
            schema_version: CASE_SCHEMA_VERSION,
            suite_id: "high-variance-control".into(),
            minimum_replicates: 3,
            replicates: vec![
                replicate(101, 26.0, '1', '2'),
                replicate(102, 28.0, '3', '4'),
                replicate(103, 30.0, '5', '6'),
            ],
        };
        let report = evaluate_suite(&suite).unwrap();
        assert_eq!(report.qualification, Qualification::Failed);
        assert!(report.failures.iter().any(|failure| failure.contains("95% response-ratio CI")));
    }

    fn noop_calibration() -> NoOpCalibration {
        let values = [0.10, 0.11, 0.09, 0.10, 0.12];
        let runs = values.iter().enumerate().map(|(index, value)| {
            let mut receipt = receipt('d', 'e');
            receipt.seed = 200 + index as u64;
            receipt.performance_digest = digest('d');
            receipt.audio_digest = digest(char::from_digit(index as u32 + 1, 16).unwrap());
            NoOpRun {
                receipt,
                measurements: BTreeMap::from([
                    ("phrase-1".into(), BTreeMap::from([
                        (FeatureDimension::BreathinessIndex, *value),
                        (FeatureDimension::VibratoDepthCents, 16.0),
                    ])),
                    ("phrase-2".into(), BTreeMap::from([
                        (FeatureDimension::BreathinessIndex, *value + 0.01),
                        (FeatureDimension::VibratoDepthCents, 18.0),
                    ])),
                ]),
            }
        }).collect();
        NoOpCalibration {
            schema_version: CASE_SCHEMA_VERSION,
            calibration_id: "noop-calibration-fixture".into(),
            minimum_replicates: 5,
            required_dimensions: vec![
                FeatureDimension::BreathinessIndex,
                FeatureDimension::VibratoDepthCents,
            ],
            resolution_floors: BTreeMap::from([
                (FeatureDimension::BreathinessIndex, 0.005),
                (FeatureDimension::VibratoDepthCents, 0.25),
            ]),
            regions: vec!["phrase-1".into(), "phrase-2".into()],
            runs,
        }
    }

    #[test]
    fn no_op_calibration_derives_repeatability_tolerances() {
        let report = calibrate_noop(&noop_calibration()).unwrap();
        assert_eq!(report.replicate_count, 5);
        assert_eq!(report.pairwise_comparisons_per_region, 10);
        let breathiness = &report.features[&FeatureDimension::BreathinessIndex];
        assert!(breathiness.recommended_tolerance >= breathiness.worst_region_p95_pairwise_delta);
        assert!(breathiness.recommended_tolerance >= 0.005);
        let vibrato = &report.features[&FeatureDimension::VibratoDepthCents];
        assert_eq!(vibrato.recommended_tolerance, 0.25);
    }

    #[test]
    fn no_op_calibration_rejects_reused_seed_or_changed_performance() {
        let mut duplicate_seed = noop_calibration();
        duplicate_seed.runs[1].receipt.seed = duplicate_seed.runs[0].receipt.seed;
        assert!(calibrate_noop(&duplicate_seed).unwrap_err().contains("reuses a random seed"));

        let mut changed_performance = noop_calibration();
        changed_performance.runs[1].receipt.performance_digest = digest('f');
        assert!(calibrate_noop(&changed_performance).unwrap_err().contains("changed code"));
    }

    #[test]
    fn no_op_calibration_rejects_missing_features_and_too_few_repeats() {
        let mut missing = noop_calibration();
        missing.runs[0].measurements.get_mut("phrase-1").unwrap()
            .remove(&FeatureDimension::BreathinessIndex);
        assert!(calibrate_noop(&missing).unwrap_err().contains("missing required dimensions"));

        let mut too_few = noop_calibration();
        too_few.runs.truncate(4);
        assert!(calibrate_noop(&too_few).unwrap_err().contains("requires 5"));
    }

    #[test]
    fn serialized_report_preserves_receipts_and_metrics() {
        let report = evaluate(&fixture()).unwrap();
        let json = serde_json::to_string(&report).unwrap();
        let decoded: ControlReport = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
        assert!(json.contains("vibrato_depth_cents"));
        assert!(json.contains("model_artifact_digest"));
    }
}

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
            return Err("locality-required cases need at least one protected region".into());
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

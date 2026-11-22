// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bounded FDM dimensional benchmark evaluation.
//!
//! This module evaluates declared geometric features from explicit measurements.
//! It deliberately has no printer I/O, slicing, machine-control authority, or
//! universal quality score. A conforming result is limited to the supplied
//! features, limits, evidence class, and measurement model; it is not a claim
//! about strength, fatigue, material properties, or standards conformity.

use crate::crypto_digest::sha256;
use crate::machine::MachineProfile;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Evidence origin is part of the benchmark claim, not presentation metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    Synthetic,
    Replay,
    Simulated,
    Physical,
}

/// Discovery observations may choose candidates; confirmatory observations
/// evaluate a frozen candidate and must not leak into the discovery set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BenchmarkPhase {
    Discovery,
    Confirmatory,
}

/// Identity of the exact applicability envelope for one benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FdmBenchmarkManifest {
    pub protocol_version: String,
    pub run_id: String,
    /// Identity of this physical/synthetic specimen within the campaign.
    pub specimen_id: String,
    /// Coordinate-frame identity for the declared build position/orientation.
    pub build_frame_id: String,
    /// Nominal specimen origin in the declared frame, millimetres.
    pub build_position_mm: [f64; 3],
    /// Nominal specimen orientation in the declared frame, degrees.
    pub build_orientation_deg: [f64; 3],
    /// Canonical machine-profile owner. Its inclusive build bounds are used
    /// directly; this avoids a second, potentially divergent envelope model.
    pub machine_profile: MachineProfile,
    pub phase: BenchmarkPhase,
    pub evidence_class: EvidenceClass,
    /// SHA-256 hex digests are identities, not signatures or safety approvals.
    pub protocol_digest: String,
    pub artifact_digest: String,
    pub machine_profile_digest: String,
    pub process_profile_digest: String,
    /// Digest binding the declared discovery/confirmatory split.
    pub split_digest: String,
    /// IDs of observations used during adaptive candidate selection.
    /// Confirmatory observations may not reuse any of these IDs.
    pub discovery_observation_ids: Vec<String>,
}

/// One critical dimensional feature, with absolute acceptance bounds in mm.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CriticalFeature {
    pub feature_id: String,
    pub lower_limit_mm: f64,
    pub upper_limit_mm: f64,
}

/// One measured feature. Canonical input units are mm; no implicit conversion
/// or default measurement is applied when unit/value/uncertainty is missing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureMeasurement {
    pub measurement_id: String,
    pub feature_id: String,
    pub value_mm: Option<f64>,
    pub uncertainty_mm: Option<f64>,
    pub unit: String,
    pub evidence_class: EvidenceClass,
    /// Source record or synthetic fixture identity.
    pub source_ref: Option<String>,
    /// Required for Physical evidence; this is an identity, not a claim that
    /// the instrument is suitable unless its calibration record is verified.
    pub instrument_id: Option<String>,
    pub calibration_ref: Option<String>,
}

/// Independent resource dimensions. None means unmeasured/unavailable, never 0.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FdmResourceUse {
    pub operator_minutes: Option<f64>,
    pub calibration_filament_g: Option<f64>,
    pub energy_wh: Option<f64>,
}

/// A single run's admitted benchmark inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FdmBenchmarkCase {
    pub manifest: FdmBenchmarkManifest,
    pub critical_features: Vec<CriticalFeature>,
    pub measurements: Vec<FeatureMeasurement>,
    pub resource_use: FdmResourceUse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureDisposition {
    Conforming,
    Nonconforming,
    /// Measurement uncertainty overlaps an acceptance boundary.
    Indeterminate,
    /// A required observation or value is absent.
    Missing,
    /// Identity, units, source class, or numeric input is invalid.
    Invalid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GeometryDisposition {
    Conforming,
    Nonconforming,
    NotEvaluable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FeatureEvaluation {
    pub feature_id: String,
    pub disposition: FeatureDisposition,
    pub value_mm: Option<f64>,
    pub uncertainty_mm: Option<f64>,
    pub lower_interval_mm: Option<f64>,
    pub upper_interval_mm: Option<f64>,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FdmBenchmarkEvaluation {
    /// None means inputs could not be canonically serialized for digesting.
    pub input_digest_sha256: Option<String>,
    pub disposition: GeometryDisposition,
    pub features: Vec<FeatureEvaluation>,
    /// Run/measurement identity and validity problems. Any such issue makes
    /// the overall disposition NotEvaluable.
    pub issues: Vec<String>,
    /// Resource metrics are separate from geometry conformity; invalid or
    /// absent resource measurements do not silently become zero.
    pub resource_issues: Vec<String>,
    pub operator_minutes: Option<f64>,
    pub calibration_filament_g: Option<f64>,
    pub energy_wh: Option<f64>,
}

/// Domain-separated SHA-256 binding to the exact serialized canonical machine profile.
fn machine_profile_sha256(profile: &MachineProfile) -> Option<String> {
    let serialized = serde_json::to_vec(profile).ok()?;
    let domain = b"symthaea.fdm-benchmark.machine-profile.v1\0";
    let mut bytes = Vec::with_capacity(domain.len() + serialized.len());
    bytes.extend_from_slice(domain);
    bytes.extend_from_slice(&serialized);
    Some(sha256(&bytes).to_hex())
}

impl FdmBenchmarkManifest {
    fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();
        for (name, value) in [
            ("protocol_version", self.protocol_version.as_str()),
            ("run_id", self.run_id.as_str()),
            ("specimen_id", self.specimen_id.as_str()),
            ("build_frame_id", self.build_frame_id.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(format!("manifest field {name} is empty"));
            }
        }
        if self.build_position_mm.iter().any(|value| !value.is_finite()) {
            issues.push("build position must contain only finite coordinates".to_owned());
        }
        if self.build_orientation_deg.iter().any(|value| !value.is_finite()) {
            issues.push("build orientation must contain only finite angles".to_owned());
        }
        let machine_profile_valid = match self.machine_profile.validate() {
            Ok(()) => true,
            Err(reason) => {
                issues.push(format!("embedded machine profile is invalid: {reason}"));
                false
            }
        };
        match machine_profile_sha256(&self.machine_profile) {
            Some(actual) if actual == self.machine_profile_digest => {}
            Some(_) => issues.push(
                "machine profile digest does not match the embedded canonical MachineProfile"
                    .to_owned(),
            ),
            None => issues.push(
                "embedded machine profile could not be serialized for SHA-256 binding".to_owned(),
            ),
        }
        if machine_profile_valid
            && self.build_position_mm.iter().all(|value| value.is_finite())
            && (0..3).any(|axis| {
                let position = self.build_position_mm[axis];
                position < self.machine_profile.build_min_mm[axis] as f64
                    || position > self.machine_profile.build_max_mm[axis] as f64
            })
        {
            issues.push(
                "specimen origin is outside the canonical MachineProfile build bounds".to_owned(),
            );
        }
        for (name, value) in [
            ("protocol_digest", self.protocol_digest.as_str()),
            ("artifact_digest", self.artifact_digest.as_str()),
            ("machine_profile_digest", self.machine_profile_digest.as_str()),
            ("process_profile_digest", self.process_profile_digest.as_str()),
            ("split_digest", self.split_digest.as_str()),
        ] {
            if !is_sha256_hex(value) {
                issues.push(format!("manifest field {name} is not a 64-character SHA-256 hex digest"));
            }
        }
        if self.discovery_observation_ids.iter().any(|id| id.trim().is_empty()) {
            issues.push("discovery observation IDs may not be empty".to_owned());
        }
        let mut seen = HashSet::new();
        for id in &self.discovery_observation_ids {
            if !seen.insert(id) {
                issues.push(format!("duplicate discovery observation ID: {id}"));
            }
        }
        issues
    }
}

/// Evaluate geometry using a conservative uncertainty interval.
///
/// A feature is Conforming only when its full [value - uncertainty, value +
/// uncertainty] interval lies inside the declared tolerance. It is
/// Nonconforming only when the interval is wholly outside that tolerance.
/// Boundary-overlapping uncertainty is Indeterminate, not rounded to a pass.
pub fn evaluate_fdm_case(case: &FdmBenchmarkCase) -> FdmBenchmarkEvaluation {
    let input_digest_sha256 = serde_json::to_vec(case)
        .ok()
        .map(|bytes| sha256(&bytes).to_hex());

    let mut issues = case.manifest.validate();
    if input_digest_sha256.is_none() {
        issues.push("input contains a value that cannot be serialized as deterministic JSON".to_owned());
    }
    if case.critical_features.is_empty() {
        issues.push("at least one critical feature is required".to_owned());
    }

    let mut feature_ids = HashSet::new();
    for feature in &case.critical_features {
        if feature.feature_id.trim().is_empty() {
            issues.push("critical feature ID may not be empty".to_owned());
        }
        if !feature_ids.insert(feature.feature_id.clone()) {
            issues.push(format!("duplicate critical feature ID: {}", feature.feature_id));
        }
        if !feature.lower_limit_mm.is_finite()
            || !feature.upper_limit_mm.is_finite()
            || feature.lower_limit_mm >= feature.upper_limit_mm
        {
            issues.push(format!("invalid tolerance bounds for feature {}", feature.feature_id));
        }
    }

    let mut measurement_ids = HashSet::<String>::new();
    let mut seen_discovery = HashSet::<String>::new();
    for id in &case.manifest.discovery_observation_ids {
        seen_discovery.insert(id.clone());
    }
    for measurement in &case.measurements {
        if measurement.measurement_id.trim().is_empty() {
            issues.push("measurement ID may not be empty".to_owned());
        }
        if !measurement_ids.insert(measurement.measurement_id.clone()) {
            issues.push(format!("duplicate measurement ID: {}", measurement.measurement_id));
        }
        if !feature_ids.contains(&measurement.feature_id) {
            issues.push(format!("measurement references unknown feature: {}", measurement.feature_id));
        }
        if measurement.evidence_class != case.manifest.evidence_class {
            issues.push(format!(
                "evidence-class mismatch for measurement {}",
                measurement.measurement_id
            ));
        }
        if measurement.unit != "mm" {
            issues.push(format!(
                "measurement {} uses unit {:?}; this evaluator accepts canonical mm only",
                measurement.measurement_id, measurement.unit
            ));
        }
        if measurement.source_ref.as_deref().is_none_or(|value| value.trim().is_empty()) {
            issues.push(format!("measurement {} has no source reference", measurement.measurement_id));
        }
        if case.manifest.phase == BenchmarkPhase::Confirmatory
            && seen_discovery.contains(&measurement.measurement_id)
        {
            issues.push(format!(
                "confirmatory measurement {} leaks a discovery observation ID",
                measurement.measurement_id
            ));
        }
        if case.manifest.evidence_class == EvidenceClass::Physical {
            if measurement.instrument_id.as_deref().is_none_or(|value| value.trim().is_empty()) {
                issues.push(format!("physical measurement {} has no instrument identity", measurement.measurement_id));
            }
            if measurement.calibration_ref.as_deref().is_none_or(|value| value.trim().is_empty()) {
                issues.push(format!("physical measurement {} has no calibration reference", measurement.measurement_id));
            }
        }
    }

    let resource_issues = validate_resource_use(&case.resource_use);
    if !issues.is_empty() {
        return FdmBenchmarkEvaluation {
            input_digest_sha256,
            disposition: GeometryDisposition::NotEvaluable,
            features: Vec::new(),
            issues,
            resource_issues,
            operator_minutes: case.resource_use.operator_minutes.filter(|v| v.is_finite() && *v >= 0.0),
            calibration_filament_g: case.resource_use.calibration_filament_g.filter(|v| v.is_finite() && *v >= 0.0),
            energy_wh: case.resource_use.energy_wh.filter(|v| v.is_finite() && *v >= 0.0),
        };
    }

    let mut measurements_by_feature: HashMap<&str, Vec<&FeatureMeasurement>> = HashMap::new();
    for measurement in &case.measurements {
        measurements_by_feature
            .entry(measurement.feature_id.as_str())
            .or_default()
            .push(measurement);
    }

    let features: Vec<FeatureEvaluation> = case
        .critical_features
        .iter()
        .map(|feature| {
            let matching = measurements_by_feature
                .get(feature.feature_id.as_str())
                .map(|items| items.as_slice())
                .unwrap_or(&[]);
            evaluate_feature(feature, matching)
        })
        .collect();

    let disposition = if features
        .iter()
        .any(|feature| feature.disposition == FeatureDisposition::Nonconforming)
    {
        GeometryDisposition::Nonconforming
    } else if features
        .iter()
        .all(|feature| feature.disposition == FeatureDisposition::Conforming)
    {
        GeometryDisposition::Conforming
    } else {
        GeometryDisposition::NotEvaluable
    };

    FdmBenchmarkEvaluation {
        input_digest_sha256,
        disposition,
        features,
        issues,
        resource_issues,
        operator_minutes: case.resource_use.operator_minutes.filter(|v| v.is_finite() && *v >= 0.0),
        calibration_filament_g: case.resource_use.calibration_filament_g.filter(|v| v.is_finite() && *v >= 0.0),
        energy_wh: case.resource_use.energy_wh.filter(|v| v.is_finite() && *v >= 0.0),
    }
}

fn evaluate_feature(
    feature: &CriticalFeature,
    matching: &[&FeatureMeasurement],
) -> FeatureEvaluation {
    let invalid = |explanation: &str| FeatureEvaluation {
        feature_id: feature.feature_id.clone(),
        disposition: FeatureDisposition::Invalid,
        value_mm: None,
        uncertainty_mm: None,
        lower_interval_mm: None,
        upper_interval_mm: None,
        explanation: explanation.to_owned(),
    };
    if matching.is_empty() {
        return FeatureEvaluation {
            feature_id: feature.feature_id.clone(),
            disposition: FeatureDisposition::Missing,
            value_mm: None,
            uncertainty_mm: None,
            lower_interval_mm: None,
            upper_interval_mm: None,
            explanation: "required critical measurement is missing; missing is not zero".to_owned(),
        };
    }
    if matching.len() != 1 {
        return invalid("critical feature must have exactly one measurement in this run");
    }

    let measurement = matching[0];
    let (Some(value), Some(uncertainty)) = (measurement.value_mm, measurement.uncertainty_mm) else {
        return FeatureEvaluation {
            feature_id: feature.feature_id.clone(),
            disposition: FeatureDisposition::Missing,
            value_mm: measurement.value_mm,
            uncertainty_mm: measurement.uncertainty_mm,
            lower_interval_mm: None,
            upper_interval_mm: None,
            explanation: "value or uncertainty is missing".to_owned(),
        };
    };

    if !value.is_finite() || !uncertainty.is_finite() || uncertainty < 0.0 {
        return invalid("measurement value must be finite and uncertainty finite and non-negative");
    }
    let lower_interval = value - uncertainty;
    let upper_interval = value + uncertainty;
    if !lower_interval.is_finite() || !upper_interval.is_finite() {
        return invalid("uncertainty interval overflowed");
    }

    let (disposition, explanation) = if lower_interval >= feature.lower_limit_mm
        && upper_interval <= feature.upper_limit_mm
    {
        (
            FeatureDisposition::Conforming,
            "entire measurement uncertainty interval is within declared limits",
        )
    } else if upper_interval < feature.lower_limit_mm || lower_interval > feature.upper_limit_mm {
        (
            FeatureDisposition::Nonconforming,
            "entire measurement uncertainty interval is outside declared limits",
        )
    } else {
        (
            FeatureDisposition::Indeterminate,
            "measurement uncertainty interval overlaps a tolerance boundary",
        )
    };

    FeatureEvaluation {
        feature_id: feature.feature_id.clone(),
        disposition,
        value_mm: Some(value),
        uncertainty_mm: Some(uncertainty),
        lower_interval_mm: Some(lower_interval),
        upper_interval_mm: Some(upper_interval),
        explanation: explanation.to_owned(),
    }
}

fn validate_resource_use(resource: &FdmResourceUse) -> Vec<String> {
    let mut issues = Vec::new();
    for (name, value) in [
        ("operator_minutes", resource.operator_minutes),
        ("calibration_filament_g", resource.calibration_filament_g),
        ("energy_wh", resource.energy_wh),
    ] {
        if value.is_some_and(|v| !v.is_finite() || v < 0.0) {
            issues.push(format!("{name} must be finite and non-negative when present"));
        }
    }
    issues
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(char: char) -> String {
        std::iter::repeat_n(char, 64).collect()
    }

    fn manifest(evidence_class: EvidenceClass, phase: BenchmarkPhase) -> FdmBenchmarkManifest {
        let machine_profile = MachineProfile::default();
        FdmBenchmarkManifest {
            protocol_version: "fdm-quality-v0.1".to_owned(),
            run_id: "run-001".to_owned(),
            specimen_id: "specimen-001".to_owned(),
            build_frame_id: "machine_base_frame".to_owned(),
            build_position_mm: [100.0, 100.0, 0.0],
            build_orientation_deg: [0.0, 0.0, 0.0],
            phase,
            evidence_class,
            protocol_digest: digest('a'),
            artifact_digest: digest('b'),
            machine_profile_digest: machine_profile_sha256(&machine_profile)
                .expect("default machine profile serializes"),
            machine_profile,
            process_profile_digest: digest('d'),
            split_digest: digest('e'),
            discovery_observation_ids: Vec::new(),
        }
    }

    fn feature() -> CriticalFeature {
        CriticalFeature {
            feature_id: "datum-width".to_owned(),
            lower_limit_mm: 9.9,
            upper_limit_mm: 10.1,
        }
    }

    fn measurement(
        evidence_class: EvidenceClass,
        value_mm: Option<f64>,
        uncertainty_mm: Option<f64>,
    ) -> FeatureMeasurement {
        FeatureMeasurement {
            measurement_id: "measurement-001".to_owned(),
            feature_id: "datum-width".to_owned(),
            value_mm,
            uncertainty_mm,
            unit: "mm".to_owned(),
            evidence_class,
            source_ref: Some("fixture:fdm-case-001".to_owned()),
            instrument_id: (evidence_class == EvidenceClass::Physical).then(|| "caliper-001".to_owned()),
            calibration_ref: (evidence_class == EvidenceClass::Physical).then(|| "calibration-record-001".to_owned()),
        }
    }

    fn case(
        evidence_class: EvidenceClass,
        value_mm: Option<f64>,
        uncertainty_mm: Option<f64>,
    ) -> FdmBenchmarkCase {
        FdmBenchmarkCase {
            manifest: manifest(evidence_class, BenchmarkPhase::Confirmatory),
            critical_features: vec![feature()],
            measurements: vec![measurement(evidence_class, value_mm, uncertainty_mm)],
            resource_use: FdmResourceUse {
                operator_minutes: Some(5.0),
                calibration_filament_g: Some(2.0),
                energy_wh: Some(8.0),
            },
        }
    }

    #[test]
    fn uncertainty_interval_inside_limits_is_conforming() {
        let result = evaluate_fdm_case(&case(EvidenceClass::Synthetic, Some(10.0), Some(0.02)));
        assert_eq!(result.disposition, GeometryDisposition::Conforming);
        assert_eq!(result.features[0].disposition, FeatureDisposition::Conforming);
        assert_eq!(result.input_digest_sha256.as_ref().map(String::len), Some(64));
    }

    #[test]
    fn uncertainty_interval_wholly_outside_limits_is_nonconforming() {
        let result = evaluate_fdm_case(&case(EvidenceClass::Synthetic, Some(10.12), Some(0.005)));
        assert_eq!(result.disposition, GeometryDisposition::Nonconforming);
        assert_eq!(result.features[0].disposition, FeatureDisposition::Nonconforming);
    }

    #[test]
    fn uncertainty_crossing_a_tolerance_boundary_is_not_a_pass() {
        let result = evaluate_fdm_case(&case(EvidenceClass::Synthetic, Some(10.11), Some(0.02)));
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert_eq!(result.features[0].disposition, FeatureDisposition::Indeterminate);
    }

    #[test]
    fn missing_measurement_is_not_replaced_with_zero() {
        let result = evaluate_fdm_case(&case(EvidenceClass::Synthetic, None, Some(0.0)));
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert_eq!(result.features[0].disposition, FeatureDisposition::Missing);
    }

    #[test]
    fn non_millimetre_units_fail_closed() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.measurements[0].unit = "inch".to_owned();
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("canonical mm")));
    }

    #[test]
    fn evidence_class_mismatch_fails_closed() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.measurements[0].evidence_class = EvidenceClass::Physical;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("evidence-class mismatch")));
    }

    #[test]
    fn physical_measurement_requires_instrument_and_calibration_identity() {
        let mut input = case(EvidenceClass::Physical, Some(10.0), Some(0.01));
        input.measurements[0].instrument_id = None;
        input.measurements[0].calibration_ref = None;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("no instrument identity")));
        assert!(result.issues.iter().any(|issue| issue.contains("no calibration reference")));
    }

    #[test]
    fn duplicate_discovery_observation_ids_are_rejected() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.discovery_observation_ids = vec![
            "discovery-001".to_owned(),
            "discovery-001".to_owned(),
        ];
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("duplicate discovery observation ID")));
    }

    #[test]
    fn confirmatory_reuse_of_discovery_observation_is_rejected() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.discovery_observation_ids = vec!["measurement-001".to_owned()];
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("leaks a discovery observation ID")));
    }

    #[test]
    fn non_finite_build_position_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.build_position_mm[0] = f64::NAN;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("build position")));
    }


    #[test]
    fn specimen_origin_outside_canonical_machine_profile_bounds_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.build_position_mm[0] = 220.001;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("outside the canonical MachineProfile build bounds")));
    }

    #[test]
    fn invalid_canonical_machine_profile_bounds_block_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.machine_profile.build_min_mm[1] = 220.0;
        input.manifest.machine_profile.build_max_mm[1] = 220.0;
        input.manifest.machine_profile_digest =
            machine_profile_sha256(&input.manifest.machine_profile).unwrap();
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("embedded machine profile is invalid")));
    }

    #[test]
    fn specimen_origin_on_inclusive_machine_profile_boundary_is_accepted() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.build_position_mm = [220.0, 220.0, 250.0];
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::Conforming);
    }

    #[test]
    fn changed_machine_profile_without_matching_digest_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.machine_profile.build_max_mm[0] = 210.0;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("digest does not match")));
    }

    #[test]
    fn non_finite_build_orientation_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.build_orientation_deg[2] = f64::INFINITY;
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("build orientation")));
    }

    #[test]
    fn empty_specimen_or_coordinate_frame_identity_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.specimen_id.clear();
        input.manifest.build_frame_id.clear();
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("specimen_id")));
        assert!(result.issues.iter().any(|issue| issue.contains("build_frame_id")));
    }

    #[test]
    fn invalid_digest_blocks_evaluation() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.manifest.artifact_digest = "not-a-digest".to_owned();
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert!(result.issues.iter().any(|issue| issue.contains("artifact_digest")));
    }

    #[test]
    fn duplicate_measurements_for_critical_feature_are_not_silently_averaged() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        let mut duplicate = input.measurements[0].clone();
        duplicate.measurement_id = "measurement-002".to_owned();
        input.measurements.push(duplicate);
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::NotEvaluable);
        assert_eq!(result.features.len(), 1);
        assert_eq!(result.features[0].disposition, FeatureDisposition::Invalid);
    }

    #[test]
    fn bad_resource_metric_is_reported_separately_from_geometry() {
        let mut input = case(EvidenceClass::Synthetic, Some(10.0), Some(0.0));
        input.resource_use.energy_wh = Some(-1.0);
        let result = evaluate_fdm_case(&input);
        assert_eq!(result.disposition, GeometryDisposition::Conforming);
        assert!(result.resource_issues.iter().any(|issue| issue.contains("energy_wh")));
        assert_eq!(result.energy_wh, None);
    }

    #[test]
    fn digest_changes_when_an_input_changes() {
        let first = evaluate_fdm_case(&case(EvidenceClass::Synthetic, Some(10.0), Some(0.0)));
        let second = evaluate_fdm_case(&case(EvidenceClass::Synthetic, Some(10.01), Some(0.0)));
        assert_ne!(first.input_digest_sha256, second.input_digest_sha256);
    }

    #[derive(serde::Deserialize)]
    struct FrozenFdmFixtureCorpus {
        schema_version: String,
        cases: Vec<FrozenFdmFixtureCase>,
    }

    #[derive(serde::Deserialize)]
    struct FrozenFdmFixtureCase {
        case_id: String,
        value_mm: Option<f64>,
        uncertainty_mm: Option<f64>,
        expected_geometry_disposition: GeometryDisposition,
        expected_feature_disposition: FeatureDisposition,
    }

    #[test]
    fn frozen_fixture_corpus_matches_independent_expected_outcomes() {
        let corpus: FrozenFdmFixtureCorpus = serde_json::from_str(include_str!(
            "../tests/fixtures/fdm_quality_v0_1.json"
        ))
        .expect("frozen FDM fixture JSON must parse");
        assert_eq!(corpus.schema_version, "fdm-quality-v0.1");
        assert!(!corpus.cases.is_empty());

        for fixture in corpus.cases {
            let input = case(EvidenceClass::Synthetic, fixture.value_mm, fixture.uncertainty_mm);
            let result = evaluate_fdm_case(&input);
            assert_eq!(
                result.disposition, fixture.expected_geometry_disposition,
                "geometry result mismatch for frozen case {}",
                fixture.case_id
            );
            assert_eq!(
                result.features[0].disposition, fixture.expected_feature_disposition,
                "feature result mismatch for frozen case {}",
                fixture.case_id
            );
        }
    }

}

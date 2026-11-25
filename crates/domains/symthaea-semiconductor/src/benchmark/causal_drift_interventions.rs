// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semiconductor causal-drift identifiability challenge.
//!
//! This is a synthetic evidence-qualified challenge protocol, not a general
//! causal discovery engine. It distinguishes a small set of deliberately
//! instrumented single-cause interventions and returns UNKNOWN whenever the
//! required independent channels or negative controls are unavailable.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use thiserror::Error;

pub const CAUSAL_DRIFT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-identifiability.v1";
pub const CAUSAL_DRIFT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-report.v1";
pub const CAUSAL_DRIFT_FIXTURE_SCHEMA_V1: &str =
    "luminous.semi-bench.causal-drift-fixture.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDriftChannelV1 {
    PrimaryProcessSensor,
    IndependentProcessSensor,
    PhysicalMetrology,
    ReferenceCheckStandard,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDriftCauseV1 {
    ProcessStateShift,
    PrimarySensorDrift,
    MetrologyReferenceDrift,
    ContextOnlyShift,
    UnknownOrConfounded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalIdentificationStatusV1 {
    IdentifiedWithinDeclaredFixture,
    UnknownNotIdentifiable,
    RejectedInvalidEvidence,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDriftReasonV1 {
    RequiredChannelMissing,
    DuplicateChannel,
    InsufficientBaselineSupport,
    InsufficientPostInterventionSupport,
    NonFiniteObservation,
    MalformedSourceDigest,
    MalformedTransformDigest,
    MalformedQualificationDigest,
    EvidenceNotQualified,
    EvidenceUnavailableAtKnowledgeCutoff,
    ChannelsNotIndependent,
    NegativeControlFailed,
    MultipleCausePattern,
    NoDeclaredCausePattern,
    ContextChangedWithoutMeasuredDrift,
    InterventionManifestMismatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftChannelEvidenceV1 {
    pub channel: CausalDriftChannelV1,
    pub before_values: Vec<f64>,
    pub after_values: Vec<f64>,
    pub source_artifact_sha256: String,
    pub transform_sha256: String,
    pub qualification_receipt_sha256: String,
    /// The independent-source declaration is part of the challenge contract.
    /// It is not proof of physical independence or source authenticity.
    pub independent_source_declared: bool,
    pub qualification_current: bool,
    pub available_at_unix_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftInputV1 {
    pub schema: String,
    pub scenario_id: String,
    pub intervention_event_id: String,
    pub intervention_time_unix_seconds: i64,
    pub diagnosis_knowledge_cutoff_unix_seconds: i64,
    pub context_before: String,
    pub context_after: String,
    pub tolerance: f64,
    pub channels: Vec<CausalDriftChannelEvidenceV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftChannelDeltaV1 {
    pub channel: CausalDriftChannelV1,
    pub before_mean: f64,
    pub after_mean: f64,
    pub delta: f64,
    pub support_before: usize,
    pub support_after: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftDiagnosisV1 {
    pub schema: String,
    pub scenario_id: String,
    pub intervention_event_id: String,
    pub status: CausalIdentificationStatusV1,
    pub cause: Option<CausalDriftCauseV1>,
    pub reason_codes: Vec<CausalDriftReasonV1>,
    pub channel_deltas: Vec<CausalDriftChannelDeltaV1>,
    pub negative_controls_passed: bool,
    pub knowledge_cutoff_unix_seconds: i64,
    pub evidence_input_sha256: String,
    pub diagnosis_sha256: String,
    pub source_authenticity_proven: bool,
    pub physical_root_cause_proven: bool,
    pub intervention_executed: bool,
    pub equipment_control_authority: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftInterventionManifestV1 {
    pub schema: String,
    pub fixture_id: String,
    pub seed: u64,
    pub expected_cause: CausalDriftCauseV1,
    pub process_delta: f64,
    pub primary_sensor_bias_delta: f64,
    pub reference_bias_delta: f64,
    pub context_after: String,
    pub omit_channels: Vec<CausalDriftChannelV1>,
    pub qualification_current: bool,
    pub available_at_unix_seconds: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CausalDriftSyntheticFixtureV1 {
    pub schema: String,
    /// Ground truth is test-oracle material and must never be supplied to diagnosis.
    pub expected_cause: CausalDriftCauseV1,
    pub input: CausalDriftInputV1,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct CausalDriftErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), CausalDriftErrorV1> {
    if value.trim().is_empty() {
        return Err(CausalDriftErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, CausalDriftErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| CausalDriftErrorV1(format!("JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn channel_digest(channel: CausalDriftChannelV1) -> String {
    let encoded = serde_json::to_vec(&channel).expect("enum serialization is infallible");
    sha256_hex(&encoded)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn build_delta(
    evidence: &CausalDriftChannelEvidenceV1,
) -> CausalDriftChannelDeltaV1 {
    let before_mean = mean(&evidence.before_values);
    let after_mean = mean(&evidence.after_values);
    CausalDriftChannelDeltaV1 {
        channel: evidence.channel,
        before_mean,
        after_mean,
        delta: after_mean - before_mean,
        support_before: evidence.before_values.len(),
        support_after: evidence.after_values.len(),
    }
}

fn invalid_report(
    input: &CausalDriftInputV1,
    reasons: Vec<CausalDriftReasonV1>,
) -> Result<CausalDriftDiagnosisV1, CausalDriftErrorV1> {
    let mut report = CausalDriftDiagnosisV1 {
        schema: CAUSAL_DRIFT_REPORT_SCHEMA_V1.into(),
        scenario_id: input.scenario_id.clone(),
        intervention_event_id: input.intervention_event_id.clone(),
        status: CausalIdentificationStatusV1::RejectedInvalidEvidence,
        cause: None,
        reason_codes: reasons,
        channel_deltas: Vec::new(),
        negative_controls_passed: false,
        knowledge_cutoff_unix_seconds: input.diagnosis_knowledge_cutoff_unix_seconds,
        evidence_input_sha256: json_sha256(input)?,
        diagnosis_sha256: String::new(),
        source_authenticity_proven: false,
        physical_root_cause_proven: false,
        intervention_executed: false,
        equipment_control_authority: false,
    };
    report.reason_codes.sort();
    report.reason_codes.dedup();
    report.diagnosis_sha256 = diagnosis_digest(&report)?;
    Ok(report)
}

fn diagnosis_digest(report: &CausalDriftDiagnosisV1) -> Result<String, CausalDriftErrorV1> {
    let mut material = report.clone();
    material.diagnosis_sha256.clear();
    json_sha256(&material)
}

fn near_zero(value: f64, tolerance: f64) -> bool {
    value.abs() <= tolerance
}

fn same_shift(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

/// Identifies only the declared single-cause signatures in a fully qualified
/// synthetic channel panel. This is not causal discovery from arbitrary data.
///
/// Required pattern:
/// - process: both process sensors + physical metrology move together; check
///   standard stays fixed;
/// - primary sensor: only primary sensor moves; independent sensor, metrology
///   and check-standard controls stay fixed;
/// - reference: physical metrology and its check standard move together;
///   independent process sensors stay fixed;
/// - context-only: all measurement channels stay fixed but context changes.
///
/// Missing, unqualified, unavailable or non-independent channels produce
/// UNKNOWN/REJECTED, never an inferred root cause.
pub fn diagnose_causal_drift_v1(
    input: &CausalDriftInputV1,
) -> Result<CausalDriftDiagnosisV1, CausalDriftErrorV1> {
    if input.schema != CAUSAL_DRIFT_SCHEMA_V1 {
        return invalid_report(input, vec![CausalDriftReasonV1::MalformedSourceDigest]);
    }
    require_text(&input.scenario_id, "input.scenario_id")?;
    require_text(&input.intervention_event_id, "input.intervention_event_id")?;
    require_text(&input.context_before, "input.context_before")?;
    require_text(&input.context_after, "input.context_after")?;
    if !input.tolerance.is_finite() || input.tolerance <= 0.0 {
        return invalid_report(input, vec![CausalDriftReasonV1::NonFiniteObservation]);
    }
    if input.intervention_time_unix_seconds > input.diagnosis_knowledge_cutoff_unix_seconds {
        return invalid_report(input, vec![CausalDriftReasonV1::EvidenceUnavailableAtKnowledgeCutoff]);
    }

    let mut seen = BTreeMap::new();
    let mut reasons = Vec::new();
    for item in &input.channels {
        if seen.insert(item.channel, item).is_some() {
            reasons.push(CausalDriftReasonV1::DuplicateChannel);
            continue;
        }
        if !valid_sha256(&item.source_artifact_sha256) {
            reasons.push(CausalDriftReasonV1::MalformedSourceDigest);
        }
        if !valid_sha256(&item.transform_sha256) {
            reasons.push(CausalDriftReasonV1::MalformedTransformDigest);
        }
        if !valid_sha256(&item.qualification_receipt_sha256) {
            reasons.push(CausalDriftReasonV1::MalformedQualificationDigest);
        }
        if !item.qualification_current {
            reasons.push(CausalDriftReasonV1::EvidenceNotQualified);
        }
        if !item.independent_source_declared {
            reasons.push(CausalDriftReasonV1::ChannelsNotIndependent);
        }
        if item.available_at_unix_seconds > input.diagnosis_knowledge_cutoff_unix_seconds {
            reasons.push(CausalDriftReasonV1::EvidenceUnavailableAtKnowledgeCutoff);
        }
        if item.before_values.len() < 2 {
            reasons.push(CausalDriftReasonV1::InsufficientBaselineSupport);
        }
        if item.after_values.len() < 2 {
            reasons.push(CausalDriftReasonV1::InsufficientPostInterventionSupport);
        }
        if item
            .before_values
            .iter()
            .chain(item.after_values.iter())
            .any(|value| !value.is_finite())
        {
            reasons.push(CausalDriftReasonV1::NonFiniteObservation);
        }
    }
    for required in [
        CausalDriftChannelV1::PrimaryProcessSensor,
        CausalDriftChannelV1::IndependentProcessSensor,
        CausalDriftChannelV1::PhysicalMetrology,
        CausalDriftChannelV1::ReferenceCheckStandard,
    ] {
        if !seen.contains_key(&required) {
            reasons.push(CausalDriftReasonV1::RequiredChannelMissing);
        }
    }
    reasons.sort();
    reasons.dedup();
    if !reasons.is_empty() {
        return invalid_report(input, reasons);
    }

    let deltas: BTreeMap<CausalDriftChannelV1, CausalDriftChannelDeltaV1> = seen
        .iter()
        .map(|(channel, item)| (*channel, build_delta(item)))
        .collect();
    let primary = deltas[&CausalDriftChannelV1::PrimaryProcessSensor].delta;
    let independent = deltas[&CausalDriftChannelV1::IndependentProcessSensor].delta;
    let physical = deltas[&CausalDriftChannelV1::PhysicalMetrology].delta;
    let check = deltas[&CausalDriftChannelV1::ReferenceCheckStandard].delta;
    let tolerance = input.tolerance;

    let process_pattern = !near_zero(primary, tolerance)
        && same_shift(primary, independent, tolerance)
        && same_shift(primary, physical, tolerance)
        && near_zero(check, tolerance);
    let sensor_pattern = !near_zero(primary, tolerance)
        && near_zero(independent, tolerance)
        && near_zero(physical, tolerance)
        && near_zero(check, tolerance);
    let reference_pattern = near_zero(primary, tolerance)
        && near_zero(independent, tolerance)
        && !near_zero(physical, tolerance)
        && same_shift(physical, check, tolerance);
    let context_pattern = near_zero(primary, tolerance)
        && near_zero(independent, tolerance)
        && near_zero(physical, tolerance)
        && near_zero(check, tolerance)
        && input.context_before != input.context_after;

    let patterns = [
        (process_pattern, CausalDriftCauseV1::ProcessStateShift),
        (sensor_pattern, CausalDriftCauseV1::PrimarySensorDrift),
        (reference_pattern, CausalDriftCauseV1::MetrologyReferenceDrift),
        (context_pattern, CausalDriftCauseV1::ContextOnlyShift),
    ];
    let matches: Vec<CausalDriftCauseV1> = patterns
        .iter()
        .filter(|(matches, _)| *matches)
        .map(|(_, cause)| *cause)
        .collect();

    let (status, cause, negative_controls_passed) = match matches.as_slice() {
        [cause] => (
            CausalIdentificationStatusV1::IdentifiedWithinDeclaredFixture,
            Some(*cause),
            true,
        ),
        [] => {
            if input.context_before == input.context_after
                && (near_zero(primary, tolerance)
                    && near_zero(independent, tolerance)
                    && near_zero(physical, tolerance)
                    && near_zero(check, tolerance))
            {
                reasons.push(CausalDriftReasonV1::NoDeclaredCausePattern);
            } else {
                reasons.push(CausalDriftReasonV1::NegativeControlFailed);
                reasons.push(CausalDriftReasonV1::NoDeclaredCausePattern);
            }
            (
                CausalIdentificationStatusV1::UnknownNotIdentifiable,
                None,
                false,
            )
        }
        _ => (
            CausalIdentificationStatusV1::UnknownNotIdentifiable,
            None,
            false,
        ),
    };
    if matches.len() > 1 {
        reasons.push(CausalDriftReasonV1::MultipleCausePattern);
    }
    reasons.sort();
    reasons.dedup();

    let mut report = CausalDriftDiagnosisV1 {
        schema: CAUSAL_DRIFT_REPORT_SCHEMA_V1.into(),
        scenario_id: input.scenario_id.clone(),
        intervention_event_id: input.intervention_event_id.clone(),
        status,
        cause,
        reason_codes: reasons,
        channel_deltas: deltas.into_values().collect(),
        negative_controls_passed,
        knowledge_cutoff_unix_seconds: input.diagnosis_knowledge_cutoff_unix_seconds,
        evidence_input_sha256: json_sha256(input)?,
        diagnosis_sha256: String::new(),
        source_authenticity_proven: false,
        physical_root_cause_proven: false,
        intervention_executed: false,
        equipment_control_authority: false,
    };
    report.diagnosis_sha256 = diagnosis_digest(&report)?;
    Ok(report)
}

/// Generates a deterministic paired-intervention challenge. The PRNG affects
/// only small mean-zero noise; the expected causal class remains separate from
/// the observation payload passed to the diagnosis function.
pub fn generate_causal_drift_fixture_v1(
    manifest: &CausalDriftInterventionManifestV1,
) -> Result<CausalDriftSyntheticFixtureV1, CausalDriftErrorV1> {
    if manifest.schema != CAUSAL_DRIFT_FIXTURE_SCHEMA_V1 {
        return Err(CausalDriftErrorV1("fixture schema is unsupported".into()));
    }
    require_text(&manifest.fixture_id, "manifest.fixture_id")?;
    require_text(&manifest.context_after, "manifest.context_after")?;
    for (value, field) in [
        (manifest.process_delta, "process_delta"),
        (manifest.primary_sensor_bias_delta, "primary_sensor_bias_delta"),
        (manifest.reference_bias_delta, "reference_bias_delta"),
    ] {
        finite(value, field)?;
    }

    let mut channels = Vec::new();
    let specs = [
        (CausalDriftChannelV1::PrimaryProcessSensor, manifest.process_delta + manifest.primary_sensor_bias_delta),
        (CausalDriftChannelV1::IndependentProcessSensor, manifest.process_delta),
        (CausalDriftChannelV1::PhysicalMetrology, manifest.process_delta + manifest.reference_bias_delta),
        (CausalDriftChannelV1::ReferenceCheckStandard, manifest.reference_bias_delta),
    ];
    for (channel, after_delta) in specs {
        if manifest.omit_channels.contains(&channel) {
            continue;
        }
        let mut before_values = Vec::new();
        let mut after_values = Vec::new();
        for index in 0..16 {
            let noise = deterministic_noise(manifest.seed, channel, index);
            before_values.push(10.0 + noise);
            after_values.push(10.0 + after_delta + noise);
        }
        let source_seed = format!("{}:{:?}:source", manifest.fixture_id, channel);
        let transform_seed = format!("{}:{:?}:transform", manifest.fixture_id, channel);
        let qualification_seed = format!("{}:{:?}:qualification", manifest.fixture_id, channel);
        channels.push(CausalDriftChannelEvidenceV1 {
            channel,
            before_values,
            after_values,
            source_artifact_sha256: sha256_hex(source_seed.as_bytes()),
            transform_sha256: sha256_hex(transform_seed.as_bytes()),
            qualification_receipt_sha256: sha256_hex(qualification_seed.as_bytes()),
            independent_source_declared: true,
            qualification_current: manifest.qualification_current,
            available_at_unix_seconds: manifest.available_at_unix_seconds,
        });
    }
    let input = CausalDriftInputV1 {
        schema: CAUSAL_DRIFT_SCHEMA_V1.into(),
        scenario_id: manifest.fixture_id.clone(),
        intervention_event_id: format!("intervention-{}", manifest.fixture_id),
        intervention_time_unix_seconds: 100,
        diagnosis_knowledge_cutoff_unix_seconds: manifest.available_at_unix_seconds,
        context_before: "recipe-A".into(),
        context_after: manifest.context_after.clone(),
        tolerance: 0.25,
        channels,
    };
    Ok(CausalDriftSyntheticFixtureV1 {
        schema: CAUSAL_DRIFT_FIXTURE_SCHEMA_V1.into(),
        expected_cause: manifest.expected_cause,
        input,
    })
}

fn deterministic_noise(
    seed: u64,
    channel: CausalDriftChannelV1,
    index: usize,
) -> f64 {
    let channel_tag = match channel {
        CausalDriftChannelV1::PrimaryProcessSensor => 1_u64,
        CausalDriftChannelV1::IndependentProcessSensor => 2,
        CausalDriftChannelV1::PhysicalMetrology => 3,
        CausalDriftChannelV1::ReferenceCheckStandard => 4,
    };
    let mut value = seed
        .wrapping_add(channel_tag.wrapping_mul(0x9e37_79b9))
        .wrapping_add((index as u64).wrapping_mul(0x85eb_ca6b));
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    let unit = (value & 0xffff) as f64 / 65535.0;
    (unit - 0.5) * 0.10
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(
        fixture_id: &str,
        expected_cause: CausalDriftCauseV1,
        process_delta: f64,
        sensor_delta: f64,
        reference_delta: f64,
        context_after: &str,
    ) -> CausalDriftInterventionManifestV1 {
        CausalDriftInterventionManifestV1 {
            schema: CAUSAL_DRIFT_FIXTURE_SCHEMA_V1.into(),
            fixture_id: fixture_id.into(),
            seed: 42,
            expected_cause,
            process_delta,
            primary_sensor_bias_delta: sensor_delta,
            reference_bias_delta: reference_delta,
            context_after: context_after.into(),
            omit_channels: Vec::new(),
            qualification_current: true,
            available_at_unix_seconds: 150,
        }
    }

    #[test]
    fn process_intervention_is_identified_with_independent_controls() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "process-only",
            CausalDriftCauseV1::ProcessStateShift,
            2.0,
            0.0,
            0.0,
            "recipe-A",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::IdentifiedWithinDeclaredFixture
        );
        assert_eq!(result.cause, Some(fixture.expected_cause));
        assert!(result.negative_controls_passed);
        assert!(!result.physical_root_cause_proven);
    }

    #[test]
    fn primary_sensor_intervention_is_identified() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "sensor-only",
            CausalDriftCauseV1::PrimarySensorDrift,
            0.0,
            2.0,
            0.0,
            "recipe-A",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(result.cause, Some(fixture.expected_cause));
    }

    #[test]
    fn reference_intervention_is_identified_by_check_standard() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "reference-only",
            CausalDriftCauseV1::MetrologyReferenceDrift,
            0.0,
            0.0,
            2.0,
            "recipe-A",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(result.cause, Some(fixture.expected_cause));
    }

    #[test]
    fn context_only_change_is_not_mislabeled_as_process_drift() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "context-only",
            CausalDriftCauseV1::ContextOnlyShift,
            0.0,
            0.0,
            0.0,
            "recipe-B",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(result.cause, Some(fixture.expected_cause));
    }

    #[test]
    fn missing_independent_sensor_control_returns_unknown_or_rejected() {
        let mut m = manifest(
            "missing-control",
            CausalDriftCauseV1::PrimarySensorDrift,
            0.0,
            2.0,
            0.0,
            "recipe-A",
        );
        m.omit_channels.push(CausalDriftChannelV1::IndependentProcessSensor);
        let fixture = generate_causal_drift_fixture_v1(&m).unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::RejectedInvalidEvidence
        );
        assert!(result.reason_codes.contains(&CausalDriftReasonV1::RequiredChannelMissing));
        assert!(!result.negative_controls_passed);
    }

    #[test]
    fn unqualified_or_future_evidence_cannot_support_causal_claim() {
        let mut m = manifest(
            "unqualified",
            CausalDriftCauseV1::ProcessStateShift,
            2.0,
            0.0,
            0.0,
            "recipe-A",
        );
        m.qualification_current = false;
        let result = diagnose_causal_drift_v1(
            &generate_causal_drift_fixture_v1(&m).unwrap().input,
        )
        .unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::RejectedInvalidEvidence
        );

        let mut future = generate_causal_drift_fixture_v1(&manifest(
            "future-evidence",
            CausalDriftCauseV1::ProcessStateShift,
            2.0,
            0.0,
            0.0,
            "recipe-A",
        ))
        .unwrap()
        .input;
        future.diagnosis_knowledge_cutoff_unix_seconds = 120;
        let result = diagnose_causal_drift_v1(&future).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::RejectedInvalidEvidence
        );
        assert!(result.reason_codes.contains(&CausalDriftReasonV1::EvidenceUnavailableAtKnowledgeCutoff));
    }

    #[test]
    fn confounded_multiple_cause_pattern_returns_unknown_not_guess() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "confounded",
            CausalDriftCauseV1::UnknownOrConfounded,
            1.0,
            1.0,
            -0.5,
            "recipe-A",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::UnknownNotIdentifiable
        );
        assert_eq!(result.cause, None);
    }

    #[test]
    fn unknown_paired_observation_signature_does_not_claim_process_or_sensor() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "ambiguous",
            CausalDriftCauseV1::UnknownOrConfounded,
            1.0,
            0.0,
            1.0,
            "recipe-A",
        ))
        .unwrap();
        let result = diagnose_causal_drift_v1(&fixture.input).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::UnknownNotIdentifiable
        );
        assert_eq!(result.cause, None);
    }

    #[test]
    fn channel_order_does_not_change_diagnosis_digest() {
        let fixture = generate_causal_drift_fixture_v1(&manifest(
            "order-invariant",
            CausalDriftCauseV1::ProcessStateShift,
            2.0,
            0.0,
            0.0,
            "recipe-A",
        ))
        .unwrap();
        let first = diagnose_causal_drift_v1(&fixture.input).unwrap();
        let mut permuted = fixture.input.clone();
        permuted.channels.reverse();
        let second = diagnose_causal_drift_v1(&permuted).unwrap();
        assert_eq!(first.diagnosis_sha256, second.diagnosis_sha256);
    }

    #[test]
    fn tampered_source_digest_rejects_diagnosis() {
        let mut input = generate_causal_drift_fixture_v1(&manifest(
            "tamper",
            CausalDriftCauseV1::ProcessStateShift,
            2.0,
            0.0,
            0.0,
            "recipe-A",
        ))
        .unwrap()
        .input;
        input.channels[0].source_artifact_sha256 = "not-a-digest".into();
        let result = diagnose_causal_drift_v1(&input).unwrap();
        assert_eq!(
            result.status,
            CausalIdentificationStatusV1::RejectedInvalidEvidence
        );
        assert!(result.reason_codes.contains(&CausalDriftReasonV1::MalformedSourceDigest));
    }
}

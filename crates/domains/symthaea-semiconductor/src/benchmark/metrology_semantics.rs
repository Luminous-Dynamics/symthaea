// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unit-aware and digest-bound binary interpretation of validated metrology.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::benchmark::process_event_graph::{
    MetrologyLabelStatusV1, ProcessEventGraphV1, ProcessEventKindV1,
    validate_process_event_graph_v1,
};

pub const SEMANTICS_SCHEMA_V1: &str = "luminous.semi-bench.binary-measurand-semantics.v1";
pub const CLASSIFICATION_SCHEMA_V1: &str =
    "luminous.semi-bench.validated-metrology-classification.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositiveConditionV1 {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryMeasurandSemanticsV1 {
    pub schema: String,
    pub semantics_id: String,
    pub revision: u64,
    pub target_measurand_id: String,
    pub unit: String,
    pub positive_class_name: String,
    pub negative_class_name: String,
    pub positive_condition: PositiveConditionV1,
    pub threshold: f64,
    /// External authority or controlled specification that defines the rule.
    pub authority_ref: String,
    /// Digest of all fields except this field.
    pub semantics_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ValidatedMetrologyClassificationV1 {
    pub schema: String,
    pub label_id: String,
    pub measurement_event_id: String,
    pub subject_id: String,
    pub target_measurand_id: String,
    pub value: f64,
    pub unit: String,
    pub semantics_id: String,
    pub semantics_revision: u64,
    pub semantics_sha256: String,
    pub positive_condition: PositiveConditionV1,
    pub threshold: f64,
    pub positive_class_name: String,
    pub negative_class_name: String,
    pub derived_positive: bool,
    pub graph_sha256: String,
    pub authority_ref: String,
    pub classification_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct MetrologySemanticsError(pub String);

fn require_text(value: &str, field: &str) -> Result<(), MetrologySemanticsError> {
    if value.trim().is_empty() {
        Err(MetrologySemanticsError(format!(
            "{field}: expected non-empty string"
        )))
    } else {
        Ok(())
    }
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

fn semantics_digest(
    semantics: &BinaryMeasurandSemanticsV1,
) -> Result<String, MetrologySemanticsError> {
    let mut material = semantics.clone();
    material.semantics_sha256.clear();
    let bytes = serde_json::to_vec(&material).map_err(|error| {
        MetrologySemanticsError(format!("semantics serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn classification_digest(
    classification: &ValidatedMetrologyClassificationV1,
) -> Result<String, MetrologySemanticsError> {
    let mut material = classification.clone();
    material.classification_sha256.clear();
    let bytes = serde_json::to_vec(&material).map_err(|error| {
        MetrologySemanticsError(format!("classification serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

pub fn seal_binary_measurand_semantics_v1(
    semantics: &mut BinaryMeasurandSemanticsV1,
) -> Result<(), MetrologySemanticsError> {
    semantics.semantics_sha256.clear();
    validate_semantics_fields(semantics)?;
    semantics.semantics_sha256 = semantics_digest(semantics)?;
    Ok(())
}

fn validate_semantics_fields(
    semantics: &BinaryMeasurandSemanticsV1,
) -> Result<(), MetrologySemanticsError> {
    if semantics.schema != SEMANTICS_SCHEMA_V1 {
        return Err(MetrologySemanticsError(
            "semantics.schema: unsupported".into(),
        ));
    }
    require_text(&semantics.semantics_id, "semantics_id")?;
    require_text(&semantics.target_measurand_id, "target_measurand_id")?;
    require_text(&semantics.unit, "unit")?;
    require_text(&semantics.positive_class_name, "positive_class_name")?;
    require_text(&semantics.negative_class_name, "negative_class_name")?;
    require_text(&semantics.authority_ref, "authority_ref")?;
    if semantics.positive_class_name == semantics.negative_class_name {
        return Err(MetrologySemanticsError(
            "positive and negative class names must differ".into(),
        ));
    }
    if semantics.revision == 0 {
        return Err(MetrologySemanticsError(
            "semantics.revision: revisions start at one".into(),
        ));
    }
    if !semantics.threshold.is_finite() {
        return Err(MetrologySemanticsError(
            "semantics.threshold: expected finite number".into(),
        ));
    }
    Ok(())
}

pub fn validate_binary_measurand_semantics_v1(
    semantics: &BinaryMeasurandSemanticsV1,
) -> Result<(), MetrologySemanticsError> {
    validate_semantics_fields(semantics)?;
    if !valid_sha256(&semantics.semantics_sha256) {
        return Err(MetrologySemanticsError(
            "semantics.semantics_sha256: expected lowercase SHA-256".into(),
        ));
    }
    if semantics_digest(semantics)? != semantics.semantics_sha256 {
        return Err(MetrologySemanticsError(
            "semantics.semantics_sha256: digest does not match rule payload".into(),
        ));
    }
    Ok(())
}

fn derive_positive(
    value: f64,
    condition: PositiveConditionV1,
    threshold: f64,
) -> bool {
    match condition {
        PositiveConditionV1::GreaterThan => value > threshold,
        PositiveConditionV1::GreaterThanOrEqual => value >= threshold,
        PositiveConditionV1::LessThan => value < threshold,
        PositiveConditionV1::LessThanOrEqual => value <= threshold,
    }
}

/// Classifies one physical result using an immutable, unit-aware rule.
///
/// The function validates the complete graph before trusting the label and
/// checks the label's measurement event, subject, measurand, unit, status and
/// finite measured value. There is deliberately no caller-supplied class bit.
pub fn classify_validated_metrology_label_v1(
    graph: &ProcessEventGraphV1,
    label_id: &str,
    semantics: &BinaryMeasurandSemanticsV1,
) -> Result<ValidatedMetrologyClassificationV1, MetrologySemanticsError> {
    validate_binary_measurand_semantics_v1(semantics)?;
    validate_process_event_graph_v1(graph).map_err(|error| {
        MetrologySemanticsError(format!("process event graph rejected: {error}"))
    })?;
    require_text(label_id, "label_id")?;

    let label = graph
        .metrology_labels
        .iter()
        .find(|label| label.label_id == label_id)
        .ok_or_else(|| MetrologySemanticsError("label_id: not present in graph".into()))?;
    if !matches!(&label.status, MetrologyLabelStatusV1::Validated) {
        return Err(MetrologySemanticsError(
            "label.status: only validated metrology results can be classified".into(),
        ));
    }
    if label.measurand_id != semantics.target_measurand_id {
        return Err(MetrologySemanticsError(
            "label.measurand_id does not match classification semantics".into(),
        ));
    }
    if label.unit != semantics.unit {
        return Err(MetrologySemanticsError(
            "label.unit does not match classification semantics".into(),
        ));
    }
    let value = label.value.ok_or_else(|| {
        MetrologySemanticsError("validated label has no numeric value".into())
    })?;
    if !value.is_finite() {
        return Err(MetrologySemanticsError(
            "label.value: expected finite numeric value".into(),
        ));
    }
    let measurement_event = graph
        .events
        .iter()
        .find(|event| event.event_id == label.measurement_event_id)
        .ok_or_else(|| {
            MetrologySemanticsError("measurement_event_id does not resolve".into())
        })?;
    if measurement_event.event_kind != ProcessEventKindV1::Metrology
        || measurement_event.subject_id.as_deref() != Some(label.subject_id.as_str())
    {
        return Err(MetrologySemanticsError(
            "measurement event does not identify this label's subject".into(),
        ));
    }

    let graph_sha256 = validate_process_event_graph_v1(graph)
        .map_err(|error| {
            MetrologySemanticsError(format!("process event graph rejected: {error}"))
        })?
        .graph_sha256;
    let mut classification = ValidatedMetrologyClassificationV1 {
        schema: CLASSIFICATION_SCHEMA_V1.into(),
        label_id: label.label_id.clone(),
        measurement_event_id: label.measurement_event_id.clone(),
        subject_id: label.subject_id.clone(),
        target_measurand_id: label.measurand_id.clone(),
        value,
        unit: label.unit.clone(),
        semantics_id: semantics.semantics_id.clone(),
        semantics_revision: semantics.revision,
        semantics_sha256: semantics.semantics_sha256.clone(),
        positive_condition: semantics.positive_condition,
        threshold: semantics.threshold,
        positive_class_name: semantics.positive_class_name.clone(),
        negative_class_name: semantics.negative_class_name.clone(),
        derived_positive: derive_positive(
            value,
            semantics.positive_condition,
            semantics.threshold,
        ),
        graph_sha256,
        authority_ref: semantics.authority_ref.clone(),
        classification_sha256: String::new(),
    };
    classification.classification_sha256 = classification_digest(&classification)?;
    Ok(classification)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::process_event_graph::{
        ClockBasisV1, MetrologyLabelV1, ProcessEventEdgeV1, ProcessEventGraphV1,
        ProcessEventV1,
    };

    const SOURCE_SHA: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TRANSFORM_SHA: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn semantics(
        condition: PositiveConditionV1,
        threshold: f64,
    ) -> BinaryMeasurandSemanticsV1 {
        let mut rule = BinaryMeasurandSemanticsV1 {
            schema: SEMANTICS_SCHEMA_V1.into(),
            semantics_id: "film-thickness-defect-v1".into(),
            revision: 1,
            target_measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            positive_class_name: "out_of_spec".into(),
            negative_class_name: "in_spec".into(),
            positive_condition: condition,
            threshold,
            authority_ref: "specification:film-thickness-control-plan:v1".into(),
            semantics_sha256: String::new(),
        };
        seal_binary_measurand_semantics_v1(&mut rule).unwrap();
        rule
    }

    fn graph(value: f64, unit: &str, status: MetrologyLabelStatusV1) -> ProcessEventGraphV1 {
        ProcessEventGraphV1 {
            schema: crate::benchmark::process_event_graph::GRAPH_SCHEMA_V1.into(),
            graph_id: "metrology-semantics-test".into(),
            source_artifact_sha256: SOURCE_SHA.into(),
            transform_sha256: TRANSFORM_SHA.into(),
            events: vec![ProcessEventV1 {
                event_id: "metrology-event-1".into(),
                event_kind: ProcessEventKindV1::Metrology,
                subject_id: Some("wafer-17".into()),
                wafer_id: Some("wafer-17".into()),
                lot_id: Some("lot-5".into()),
                carrier_id: None,
                tool_id: Some("metrology-tool-1".into()),
                chamber_id: None,
                module_id: None,
                process_step_id: None,
                recipe_id: None,
                recipe_revision: None,
                event_time: "2026-11-22T08:00:00".into(),
                event_clock_domain_id: "fab-local".into(),
                event_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:00:01".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                source_sequence: Some(1),
                source_artifact_sha256: SOURCE_SHA.into(),
                transform_sha256: TRANSFORM_SHA.into(),
                authority_ref: "lims-event-adapter-v1".into(),
            }],
            edges: Vec::<ProcessEventEdgeV1>::new(),
            metrology_labels: vec![MetrologyLabelV1 {
                label_id: "label-17".into(),
                measurement_event_id: "metrology-event-1".into(),
                subject_id: "wafer-17".into(),
                measurand_id: "film-thickness".into(),
                value: Some(value),
                unit: unit.into(),
                standard_uncertainty: Some(0.1),
                censoring_lower_bound: None,
                censoring_upper_bound: None,
                censoring_basis: None,
                status,
                measurement_time: "2026-11-22T08:00:00".into(),
                measurement_clock_domain_id: "fab-local".into(),
                measurement_clock_basis: ClockBasisV1::SourceLocalNaive,
                available_at: "2026-11-22T08:05:00".into(),
                availability_clock_domain_id: "fab-local".into(),
                availability_clock_basis: ClockBasisV1::SourceLocalNaive,
                instrument_id: Some("metrology-tool-1".into()),
                calibration_ref: Some("cal-17".into()),
                revision: 1,
                supersedes_label_id: None,
                source_artifact_sha256: SOURCE_SHA.into(),
                authority_ref: "lims-result-v1".into(),
            }],
        }
    }

    #[test]
    fn all_four_threshold_operators_have_explicit_boundary_semantics() {
        let input = graph(10.0, "nm", MetrologyLabelStatusV1::Validated);
        assert!(!classify_validated_metrology_label_v1(
            &input,
            "label-17",
            &semantics(PositiveConditionV1::GreaterThan, 10.0)
        )
        .unwrap()
        .derived_positive);
        assert!(classify_validated_metrology_label_v1(
            &input,
            "label-17",
            &semantics(PositiveConditionV1::GreaterThanOrEqual, 10.0)
        )
        .unwrap()
        .derived_positive);
        assert!(!classify_validated_metrology_label_v1(
            &input,
            "label-17",
            &semantics(PositiveConditionV1::LessThan, 10.0)
        )
        .unwrap()
        .derived_positive);
        assert!(classify_validated_metrology_label_v1(
            &input,
            "label-17",
            &semantics(PositiveConditionV1::LessThanOrEqual, 10.0)
        )
        .unwrap()
        .derived_positive);
    }

    #[test]
    fn classification_receipt_binds_value_graph_and_semantics_digest() {
        let input = graph(10.5, "nm", MetrologyLabelStatusV1::Validated);
        let rule = semantics(PositiveConditionV1::GreaterThan, 10.0);
        let first =
            classify_validated_metrology_label_v1(&input, "label-17", &rule).unwrap();
        assert!(first.derived_positive);
        assert_eq!(first.semantics_sha256, rule.semantics_sha256);
        let mut changed = rule.clone();
        changed.threshold = 11.0;
        seal_binary_measurand_semantics_v1(&mut changed).unwrap();
        let second =
            classify_validated_metrology_label_v1(&input, "label-17", &changed).unwrap();
        assert!(!second.derived_positive);
        assert_ne!(first.classification_sha256, second.classification_sha256);
    }

    #[test]
    fn wrong_units_and_measurands_reject() {
        let rule = semantics(PositiveConditionV1::GreaterThan, 10.0);
        assert!(
            classify_validated_metrology_label_v1(
                &graph(10.5, "um", MetrologyLabelStatusV1::Validated),
                "label-17",
                &rule
            )
            .is_err()
        );

        let mut wrong_measurand = graph(10.5, "nm", MetrologyLabelStatusV1::Validated);
        wrong_measurand.metrology_labels[0].measurand_id = "sheet-resistance".into();
        assert!(
            classify_validated_metrology_label_v1(&wrong_measurand, "label-17", &rule).is_err()
        );
    }

    #[test]
    fn non_validated_results_reject() {
        for status in [
            MetrologyLabelStatusV1::Provisional,
            MetrologyLabelStatusV1::Censored,
            MetrologyLabelStatusV1::Invalid,
            MetrologyLabelStatusV1::Unknown,
        ] {
            assert!(
                classify_validated_metrology_label_v1(
                    &graph(10.5, "nm", status),
                    "label-17",
                    &semantics(PositiveConditionV1::GreaterThan, 10.0)
                )
                .is_err()
            );
        }
    }

    #[test]
    fn semantics_digest_tampering_and_invalid_definitions_reject() {
        let mut tampered = semantics(PositiveConditionV1::GreaterThan, 10.0);
        tampered.threshold = 12.0;
        assert!(validate_binary_measurand_semantics_v1(&tampered).is_err());

        let mut invalid = semantics(PositiveConditionV1::GreaterThan, 10.0);
        invalid.unit.clear();
        assert!(seal_binary_measurand_semantics_v1(&mut invalid).is_err());
    }

    #[test]
    fn class_names_must_differ_and_revision_must_be_positive() {
        let mut same_names = semantics(PositiveConditionV1::GreaterThan, 10.0);
        same_names.negative_class_name = same_names.positive_class_name.clone();
        assert!(seal_binary_measurand_semantics_v1(&mut same_names).is_err());

        let mut revision_zero = semantics(PositiveConditionV1::GreaterThan, 10.0);
        revision_zero.revision = 0;
        assert!(seal_binary_measurand_semantics_v1(&mut revision_zero).is_err());
    }
}

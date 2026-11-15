//! Fusion validation-domain split-integrity contract.
//!
//! This module checks declared separation between calibration, measured
//! test-stand evidence and blind held-out observations. It does not inspect
//! raw files, calculate model errors, establish metrology traceability, or
//! prove that the declarations are truthful. Declared digests remain identity
//! metadata until a separate payload verifier fetches and hashes the bytes.

use std::collections::BTreeSet;

use super::{EvidenceClass, ModelUse};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationExposure {
    pub record_id: String,
    pub dataset_id: String,
    pub source_artifact_sha256: String,
    pub regime_id: String,
    pub geometry_sha256: String,
    pub episode_ids: Vec<String>,
    /// Identity of the preprocessing fit; it must be derived from calibration data only.
    pub preprocessing_fit_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasuredObservation {
    pub record_id: String,
    pub evidence_class: EvidenceClass,
    pub dataset_id: String,
    pub source_artifact_sha256: String,
    pub regime_id: String,
    pub geometry_sha256: String,
    pub facility_id: String,
    pub case_id: String,
    pub operating_point_id: String,
    pub episode_ids: Vec<String>,
    pub diagnostic_revision: String,
    pub measurement_procedure_ref: String,
    pub diagnostic_calibration_ref: String,
    pub measurement_uncertainty_ref: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldoutExposure {
    pub used_for_training: bool,
    pub used_for_normalization_fit: bool,
    pub used_for_feature_selection: bool,
    pub used_for_hyperparameter_selection: bool,
    pub used_for_model_or_policy_selection: bool,
    pub used_for_calibration: bool,
}

impl HoldoutExposure {
    fn is_unexposed(&self) -> bool {
        !self.used_for_training
            && !self.used_for_normalization_fit
            && !self.used_for_feature_selection
            && !self.used_for_hyperparameter_selection
            && !self.used_for_model_or_policy_selection
            && !self.used_for_calibration
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlindHoldoutMeasurement {
    pub observation: MeasuredObservation,
    pub exposure: HoldoutExposure,
    /// The holdout is only declared blind until a final evaluation is authorized.
    pub withheld_until_final_evaluation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationDomain {
    pub domain_id: String,
    pub model_revision: String,
    pub regime_id: String,
    pub supported_uses: Vec<ModelUse>,
    pub allowed_geometry_sha256: Vec<String>,
    pub allowed_facility_ids: Vec<String>,
    pub allowed_case_ids: Vec<String>,
    pub allowed_operating_point_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationPlan {
    pub model_revision: String,
    pub requested_use: ModelUse,
    pub domain: ValidationDomain,
    pub calibration: Vec<CalibrationExposure>,
    pub test_stand: MeasuredObservation,
    pub blind_holdout: BlindHoldoutMeasurement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationAdmissibility {
    EligibleForIndependentMetricEvaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationClaimCeiling {
    SplitIntegrityAndDeclaredDomainScopeOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationDomainAudit {
    pub domain_id: String,
    pub model_revision: String,
    pub requested_use: ModelUse,
    pub calibration_record_count: usize,
    pub calibration_episode_count: usize,
    pub test_stand_record_id: String,
    pub holdout_record_id: String,
    pub distinct_holdout_dimensions: Vec<&'static str>,
    pub admissibility: ValidationAdmissibility,
    pub claim_ceiling: ValidationClaimCeiling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationDomainError {
    EmptyIdentity(String),
    InvalidSha256(&'static str),
    DuplicateEpisode(String),
    DuplicateDomainValue {
        field: &'static str,
        value: String,
    },
    DuplicateCalibrationRecord(String),
    UnsupportedModelUse,
    ModelRevisionMismatch,
    EmptyCalibrationSet,
    MissingDomainDimension(&'static str),
    UnsupportedRegime(String),
    OutOfDomain {
        record_id: String,
        dimension: &'static str,
    },
    NonMeasuredObservation {
        record_id: String,
        evidence_class: EvidenceClass,
    },
    MissingMeasurementProvenance {
        record_id: String,
        field: &'static str,
    },
    MissingEpisodeIdentity(String),
    DuplicateEpisodeIdentity(String),
    ReusedDatasetIdentity {
        first_record_id: String,
        second_record_id: String,
    },
    ReusedArtifactIdentity {
        first_record_id: String,
        second_record_id: String,
    },
    ReusedEpisodeIdentity {
        episode_id: String,
        first_record_id: String,
        second_record_id: String,
    },
    HoldoutExposureLeakage(&'static str),
    HoldoutNotWithheld,
    HoldoutNotDistinct,
}

/// Validate source/episode separation and declared domain scope.
///
/// A successful result means only that the supplied declarations satisfy this
/// contract. It does not check payload bytes or evaluate a model against data.
pub fn validate_validation_domain(
    plan: &ValidationPlan,
) -> Result<ValidationDomainAudit, Vec<ValidationDomainError>> {
    let mut errors = Vec::new();

    validate_identity(&plan.model_revision, "model_revision", &mut errors);
    validate_identity(&plan.domain.domain_id, "domain_id", &mut errors);
    validate_identity(
        &plan.domain.model_revision,
        "domain.model_revision",
        &mut errors,
    );
    validate_identity(&plan.domain.regime_id, "domain.regime_id", &mut errors);

    if plan.model_revision != plan.domain.model_revision {
        errors.push(ValidationDomainError::ModelRevisionMismatch);
    }
    if !plan.domain.supported_uses.contains(&plan.requested_use) {
        errors.push(ValidationDomainError::UnsupportedModelUse);
    }

    validate_domain_values(
        &plan.domain.allowed_geometry_sha256,
        "allowed_geometry_sha256",
        true,
        &mut errors,
    );
    validate_domain_values(
        &plan.domain.allowed_facility_ids,
        "allowed_facility_ids",
        false,
        &mut errors,
    );
    validate_domain_values(
        &plan.domain.allowed_case_ids,
        "allowed_case_ids",
        false,
        &mut errors,
    );
    validate_domain_values(
        &plan.domain.allowed_operating_point_ids,
        "allowed_operating_point_ids",
        false,
        &mut errors,
    );

    if plan.calibration.is_empty() {
        errors.push(ValidationDomainError::EmptyCalibrationSet);
    }

    let mut calibration_ids = BTreeSet::new();
    let mut calibration_episodes = BTreeSet::new();
    let mut calibration_datasets = Vec::new();
    let mut calibration_artifacts = Vec::new();

    for record in &plan.calibration {
        validate_identity(&record.record_id, "calibration.record_id", &mut errors);
        validate_identity(&record.dataset_id, "calibration.dataset_id", &mut errors);
        validate_identity(&record.regime_id, "calibration.regime_id", &mut errors);
        validate_identity(
            &record.preprocessing_fit_id,
            "preprocessing_fit_id",
            &mut errors,
        );
        validate_digest(
            &record.source_artifact_sha256,
            "calibration.source_artifact_sha256",
            &mut errors,
        );
        validate_digest(
            &record.geometry_sha256,
            "calibration.geometry_sha256",
            &mut errors,
        );

        if !calibration_ids.insert(record.record_id.as_str()) {
            errors.push(ValidationDomainError::DuplicateCalibrationRecord(
                record.record_id.clone(),
            ));
        }

        if record.regime_id != plan.domain.regime_id {
            errors.push(ValidationDomainError::UnsupportedRegime(
                record.regime_id.clone(),
            ));
        }
        if !plan
            .domain
            .allowed_geometry_sha256
            .contains(&record.geometry_sha256)
        {
            errors.push(ValidationDomainError::OutOfDomain {
                record_id: record.record_id.clone(),
                dimension: "geometry",
            });
        }

        validate_episode_ids(&record.record_id, &record.episode_ids, &mut errors);
        for episode_id in &record.episode_ids {
            calibration_episodes.insert(episode_id.clone());
        }
        calibration_datasets.push((record.dataset_id.as_str(), record.record_id.as_str()));
        calibration_artifacts.push((
            record.source_artifact_sha256.as_str(),
            record.record_id.as_str(),
        ));
    }

    validate_observation(&plan.test_stand, &plan.domain, "test_stand", &mut errors);
    validate_observation(
        &plan.blind_holdout.observation,
        &plan.domain,
        "blind_holdout",
        &mut errors,
    );

    if plan.test_stand.evidence_class != EvidenceClass::MeasuredObservation {
        errors.push(ValidationDomainError::NonMeasuredObservation {
            record_id: plan.test_stand.record_id.clone(),
            evidence_class: plan.test_stand.evidence_class,
        });
    }
    if plan.blind_holdout.observation.evidence_class != EvidenceClass::MeasuredObservation {
        errors.push(ValidationDomainError::NonMeasuredObservation {
            record_id: plan.blind_holdout.observation.record_id.clone(),
            evidence_class: plan.blind_holdout.observation.evidence_class,
        });
    }

    if !plan.blind_holdout.withheld_until_final_evaluation {
        errors.push(ValidationDomainError::HoldoutNotWithheld);
    }
    if !plan.blind_holdout.exposure.is_unexposed() {
        for (exposed, field) in [
            (plan.blind_holdout.exposure.used_for_training, "training"),
            (
                plan.blind_holdout.exposure.used_for_normalization_fit,
                "normalization_fit",
            ),
            (
                plan.blind_holdout.exposure.used_for_feature_selection,
                "feature_selection",
            ),
            (
                plan.blind_holdout
                    .exposure
                    .used_for_hyperparameter_selection,
                "hyperparameter_selection",
            ),
            (
                plan.blind_holdout
                    .exposure
                    .used_for_model_or_policy_selection,
                "model_or_policy_selection",
            ),
            (
                plan.blind_holdout.exposure.used_for_calibration,
                "calibration",
            ),
        ] {
            if exposed {
                errors.push(ValidationDomainError::HoldoutExposureLeakage(field));
            }
        }
    }

    let holdout = &plan.blind_holdout.observation;
    let distinct_holdout_dimensions = [
        (
            plan.test_stand.facility_id != holdout.facility_id,
            "facility",
        ),
        (plan.test_stand.case_id != holdout.case_id, "case"),
        (
            plan.test_stand.operating_point_id != holdout.operating_point_id,
            "operating_point",
        ),
    ]
    .into_iter()
    .filter_map(|(different, dimension)| different.then_some(dimension))
    .collect::<Vec<_>>();

    if distinct_holdout_dimensions.is_empty() {
        errors.push(ValidationDomainError::HoldoutNotDistinct);
    }

    validate_no_overlap(
        &plan.test_stand,
        &plan.calibration,
        &calibration_episodes,
        &calibration_datasets,
        &calibration_artifacts,
        &mut errors,
    );
    validate_no_overlap(
        holdout,
        &plan.calibration,
        &calibration_episodes,
        &calibration_datasets,
        &calibration_artifacts,
        &mut errors,
    );
    validate_pair_overlap(&plan.test_stand, holdout, &mut errors);

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok(ValidationDomainAudit {
        domain_id: plan.domain.domain_id.clone(),
        model_revision: plan.model_revision.clone(),
        requested_use: plan.requested_use,
        calibration_record_count: plan.calibration.len(),
        calibration_episode_count: calibration_episodes.len(),
        test_stand_record_id: plan.test_stand.record_id.clone(),
        holdout_record_id: holdout.record_id.clone(),
        distinct_holdout_dimensions,
        admissibility: ValidationAdmissibility::EligibleForIndependentMetricEvaluation,
        claim_ceiling: ValidationClaimCeiling::SplitIntegrityAndDeclaredDomainScopeOnly,
    })
}

fn validate_no_overlap(
    observation: &MeasuredObservation,
    calibration: &[CalibrationExposure],
    calibration_episodes: &BTreeSet<String>,
    calibration_datasets: &[(&str, &str)],
    calibration_artifacts: &[(&str, &str)],
    errors: &mut Vec<ValidationDomainError>,
) {
    for (dataset_id, calibration_record_id) in calibration_datasets {
        if *dataset_id == observation.dataset_id {
            errors.push(ValidationDomainError::ReusedDatasetIdentity {
                first_record_id: (*calibration_record_id).to_string(),
                second_record_id: observation.record_id.clone(),
            });
        }
    }

    for (artifact_sha256, calibration_record_id) in calibration_artifacts {
        if *artifact_sha256 == observation.source_artifact_sha256 {
            errors.push(ValidationDomainError::ReusedArtifactIdentity {
                first_record_id: (*calibration_record_id).to_string(),
                second_record_id: observation.record_id.clone(),
            });
        }
    }

    for episode_id in &observation.episode_ids {
        if calibration_episodes.contains(episode_id) {
            let calibration_record_id = calibration
                .iter()
                .find(|record| record.episode_ids.contains(episode_id))
                .map_or("unknown", |record| record.record_id.as_str());
            errors.push(ValidationDomainError::ReusedEpisodeIdentity {
                episode_id: episode_id.clone(),
                first_record_id: calibration_record_id.to_string(),
                second_record_id: observation.record_id.clone(),
            });
        }
    }
}

fn validate_pair_overlap(
    test_stand: &MeasuredObservation,
    holdout: &MeasuredObservation,
    errors: &mut Vec<ValidationDomainError>,
) {
    if test_stand.dataset_id == holdout.dataset_id {
        errors.push(ValidationDomainError::ReusedDatasetIdentity {
            first_record_id: test_stand.record_id.clone(),
            second_record_id: holdout.record_id.clone(),
        });
    }
    if test_stand.source_artifact_sha256 == holdout.source_artifact_sha256 {
        errors.push(ValidationDomainError::ReusedArtifactIdentity {
            first_record_id: test_stand.record_id.clone(),
            second_record_id: holdout.record_id.clone(),
        });
    }
    for episode in &holdout.episode_ids {
        if test_stand
            .episode_ids
            .iter()
            .any(|test_episode| test_episode == episode)
        {
            errors.push(ValidationDomainError::ReusedEpisodeIdentity {
                episode_id: episode.clone(),
                first_record_id: test_stand.record_id.clone(),
                second_record_id: holdout.record_id.clone(),
            });
        }
    }
}

fn validate_observation(
    observation: &MeasuredObservation,
    domain: &ValidationDomain,
    role: &'static str,
    errors: &mut Vec<ValidationDomainError>,
) {
    for (value, field) in [
        (&observation.record_id, "record_id"),
        (&observation.dataset_id, "dataset_id"),
        (&observation.regime_id, "regime_id"),
        (&observation.facility_id, "facility_id"),
        (&observation.case_id, "case_id"),
        (&observation.operating_point_id, "operating_point_id"),
        (&observation.diagnostic_revision, "diagnostic_revision"),
        (
            &observation.measurement_procedure_ref,
            "measurement_procedure_ref",
        ),
        (
            &observation.diagnostic_calibration_ref,
            "diagnostic_calibration_ref",
        ),
        (
            &observation.measurement_uncertainty_ref,
            "measurement_uncertainty_ref",
        ),
    ] {
        validate_identity(value, &format!("{role}.{field}"), errors);
    }

    validate_digest(
        &observation.source_artifact_sha256,
        "measurement.source_artifact_sha256",
        errors,
    );
    validate_digest(
        &observation.geometry_sha256,
        "measurement.geometry_sha256",
        errors,
    );
    validate_episode_ids(&observation.record_id, &observation.episode_ids, errors);

    if observation.regime_id != domain.regime_id {
        errors.push(ValidationDomainError::UnsupportedRegime(
            observation.regime_id.clone(),
        ));
    }
    if !domain
        .allowed_geometry_sha256
        .contains(&observation.geometry_sha256)
    {
        errors.push(ValidationDomainError::OutOfDomain {
            record_id: observation.record_id.clone(),
            dimension: "geometry",
        });
    }
    if !domain
        .allowed_facility_ids
        .contains(&observation.facility_id)
    {
        errors.push(ValidationDomainError::OutOfDomain {
            record_id: observation.record_id.clone(),
            dimension: "facility",
        });
    }
    if !domain.allowed_case_ids.contains(&observation.case_id) {
        errors.push(ValidationDomainError::OutOfDomain {
            record_id: observation.record_id.clone(),
            dimension: "case",
        });
    }
    if !domain
        .allowed_operating_point_ids
        .contains(&observation.operating_point_id)
    {
        errors.push(ValidationDomainError::OutOfDomain {
            record_id: observation.record_id.clone(),
            dimension: "operating_point",
        });
    }
}

fn validate_domain_values(
    values: &[String],
    field: &'static str,
    require_sha256: bool,
    errors: &mut Vec<ValidationDomainError>,
) {
    if values.is_empty() {
        errors.push(ValidationDomainError::MissingDomainDimension(field));
    }
    let mut seen = BTreeSet::new();
    for value in values {
        validate_identity(value, field, errors);
        if require_sha256 {
            validate_digest(value, field, errors);
        }
        if !seen.insert(value.as_str()) {
            errors.push(ValidationDomainError::DuplicateDomainValue {
                field,
                value: value.clone(),
            });
        }
    }
}

fn validate_episode_ids(
    record_id: &str,
    episode_ids: &[String],
    errors: &mut Vec<ValidationDomainError>,
) {
    if episode_ids.is_empty() {
        errors.push(ValidationDomainError::MissingEpisodeIdentity(
            record_id.to_string(),
        ));
    }
    let mut seen = BTreeSet::new();
    for episode_id in episode_ids {
        validate_identity(episode_id, "episode_id", errors);
        if !seen.insert(episode_id.as_str()) {
            errors.push(ValidationDomainError::DuplicateEpisodeIdentity(
                episode_id.clone(),
            ));
        }
    }
}

fn validate_identity(value: &str, field: &str, errors: &mut Vec<ValidationDomainError>) {
    if value.trim().is_empty() {
        errors.push(ValidationDomainError::EmptyIdentity(field.to_string()));
    }
}

fn validate_digest(value: &str, field: &'static str, errors: &mut Vec<ValidationDomainError>) {
    if !super::is_sha256(value) {
        errors.push(ValidationDomainError::InvalidSha256(field));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: char) -> String {
        std::iter::repeat_n(value, 64).collect()
    }

    fn calibration() -> CalibrationExposure {
        CalibrationExposure {
            record_id: "calibration-a".into(),
            dataset_id: "calibration-dataset".into(),
            source_artifact_sha256: digest('a'),
            regime_id: "tokamak-regime".into(),
            geometry_sha256: digest('b'),
            episode_ids: vec!["cal-shot-001".into(), "cal-shot-002".into()],
            preprocessing_fit_id: "preprocessing-fit-cal-only".into(),
        }
    }

    fn measurement(
        record_id: &str,
        dataset_id: &str,
        artifact_digest: char,
        facility: &str,
        case_id: &str,
        operating_point: &str,
        episode_id: &str,
    ) -> MeasuredObservation {
        MeasuredObservation {
            record_id: record_id.into(),
            evidence_class: EvidenceClass::MeasuredObservation,
            dataset_id: dataset_id.into(),
            source_artifact_sha256: digest(artifact_digest),
            regime_id: "tokamak-regime".into(),
            geometry_sha256: digest('b'),
            facility_id: facility.into(),
            case_id: case_id.into(),
            operating_point_id: operating_point.into(),
            episode_ids: vec![episode_id.into()],
            diagnostic_revision: "diagnostic-r3".into(),
            measurement_procedure_ref: "procedure-v2".into(),
            diagnostic_calibration_ref: "calibration-certificate-17".into(),
            measurement_uncertainty_ref: "measurement-uncertainty-u7".into(),
        }
    }

    fn plan() -> ValidationPlan {
        ValidationPlan {
            model_revision: "model-r4".into(),
            requested_use: ModelUse::SingleStepPrediction,
            domain: ValidationDomain {
                domain_id: "tokamak-model-domain-v1".into(),
                model_revision: "model-r4".into(),
                regime_id: "tokamak-regime".into(),
                supported_uses: vec![ModelUse::SingleStepPrediction, ModelUse::OpenLoopRollout],
                allowed_geometry_sha256: vec![digest('b')],
                allowed_facility_ids: vec!["facility-a".into(), "facility-b".into()],
                allowed_case_ids: vec!["case-a".into(), "case-b".into()],
                allowed_operating_point_ids: vec!["op-a".into(), "op-b".into()],
            },
            calibration: vec![calibration()],
            test_stand: measurement(
                "test-stand-001",
                "test-dataset",
                'c',
                "facility-a",
                "case-a",
                "op-a",
                "test-shot-001",
            ),
            blind_holdout: BlindHoldoutMeasurement {
                observation: measurement(
                    "holdout-001",
                    "holdout-dataset",
                    'd',
                    "facility-a",
                    "case-a",
                    "op-b",
                    "holdout-shot-001",
                ),
                exposure: HoldoutExposure {
                    used_for_training: false,
                    used_for_normalization_fit: false,
                    used_for_feature_selection: false,
                    used_for_hyperparameter_selection: false,
                    used_for_model_or_policy_selection: false,
                    used_for_calibration: false,
                },
                withheld_until_final_evaluation: true,
            },
        }
    }

    fn has_error(errors: &[ValidationDomainError], expected: ValidationDomainError) -> bool {
        errors.contains(&expected)
    }

    #[test]
    fn same_facility_different_operating_point_is_admissible_for_metric_evaluation() {
        let audit = validate_validation_domain(&plan()).unwrap();
        assert_eq!(
            audit.admissibility,
            ValidationAdmissibility::EligibleForIndependentMetricEvaluation
        );
        assert_eq!(audit.distinct_holdout_dimensions, vec!["operating_point"]);
        assert_eq!(
            audit.claim_ceiling,
            ValidationClaimCeiling::SplitIntegrityAndDeclaredDomainScopeOnly
        );
    }

    #[test]
    fn different_case_at_same_facility_is_admissible() {
        let mut source = plan();
        source.blind_holdout.observation.case_id = "case-b".into();
        source.blind_holdout.observation.operating_point_id = "op-a".into();
        let audit = validate_validation_domain(&source).unwrap();
        assert_eq!(audit.distinct_holdout_dimensions, vec!["case"]);
    }

    #[test]
    fn different_facility_is_admissible() {
        let mut source = plan();
        source.blind_holdout.observation.facility_id = "facility-b".into();
        source.blind_holdout.observation.operating_point_id = "op-a".into();
        let audit = validate_validation_domain(&source).unwrap();
        assert_eq!(audit.distinct_holdout_dimensions, vec!["facility"]);
    }

    #[test]
    fn same_facility_case_and_operating_point_cannot_be_claimed_as_transfer() {
        let mut source = plan();
        source.blind_holdout.observation.operating_point_id = "op-a".into();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::HoldoutNotDistinct
        ));
    }

    #[test]
    fn reused_episode_across_calibration_and_holdout_is_rejected() {
        let mut source = plan();
        source.blind_holdout.observation.episode_ids = vec!["cal-shot-001".into()];
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationDomainError::ReusedEpisodeIdentity { episode_id, .. }
                if episode_id == "cal-shot-001"
        )));
    }

    #[test]
    fn reused_dataset_identity_is_rejected() {
        let mut source = plan();
        source.blind_holdout.observation.dataset_id = "calibration-dataset".into();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ValidationDomainError::ReusedDatasetIdentity { .. }))
        );
    }

    #[test]
    fn reused_artifact_digest_is_rejected() {
        let mut source = plan();
        source.blind_holdout.observation.source_artifact_sha256 =
            source.calibration[0].source_artifact_sha256.clone();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, ValidationDomainError::ReusedArtifactIdentity { .. }))
        );
    }

    #[test]
    fn holdout_exposed_to_model_selection_is_rejected() {
        let mut source = plan();
        source
            .blind_holdout
            .exposure
            .used_for_model_or_policy_selection = true;
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::HoldoutExposureLeakage("model_or_policy_selection")
        ));
    }

    #[test]
    fn synthetic_solver_output_cannot_be_used_as_test_stand_measurement() {
        let mut source = plan();
        source.test_stand.evidence_class = EvidenceClass::ModelOutput;
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::NonMeasuredObservation {
                record_id: "test-stand-001".into(),
                evidence_class: EvidenceClass::ModelOutput,
            }
        ));
    }

    #[test]
    fn missing_measurement_uncertainty_is_rejected_as_incomplete_provenance() {
        let mut source = plan();
        source
            .blind_holdout
            .observation
            .measurement_uncertainty_ref
            .clear();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::EmptyIdentity(
                "blind_holdout.measurement_uncertainty_ref".into()
            )
        ));
    }

    #[test]
    fn geometry_outside_declared_domain_is_rejected() {
        let mut source = plan();
        source.blind_holdout.observation.geometry_sha256 = digest('e');
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::OutOfDomain {
                record_id: "holdout-001".into(),
                dimension: "geometry",
            }
        ));
    }

    #[test]
    fn intended_use_must_be_supported_by_domain_profile() {
        let mut source = plan();
        source.requested_use = ModelUse::OptimizationInsideSurrogate;
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::UnsupportedModelUse
        ));
    }

    #[test]
    fn model_revision_must_match_domain_revision() {
        let mut source = plan();
        source.model_revision = "model-r5".into();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::ModelRevisionMismatch
        ));
    }

    #[test]
    fn reused_episode_between_test_stand_and_holdout_is_rejected() {
        let mut source = plan();
        source.blind_holdout.observation.episode_ids = source.test_stand.episode_ids.clone();
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            ValidationDomainError::ReusedEpisodeIdentity { episode_id, .. }
                if episode_id == "test-shot-001"
        )));
    }

    #[test]
    fn holdout_must_be_explicitly_withheld_until_final_evaluation() {
        let mut source = plan();
        source.blind_holdout.withheld_until_final_evaluation = false;
        let errors = validate_validation_domain(&source).unwrap_err();
        assert!(has_error(
            &errors,
            ValidationDomainError::HoldoutNotWithheld
        ));
    }
    #[test]
    fn frozen_validation_domain_corpus_matches_contract_dispositions() {
        const CORPUS: &str = include_str!("../fixtures/validation-domain-cases-v1.tsv");
        let mut checked = 0usize;

        for line in CORPUS.lines().map(str::trim) {
            if line.is_empty() || line.starts_with('#') || line == "case_id|expected|mutation" {
                continue;
            }
            let fields: Vec<&str> = line.split('|').collect();
            assert_eq!(
                fields.len(),
                3,
                "malformed validation-domain fixture: {line}"
            );
            let [case_id, expected, mutation] = [fields[0], fields[1], fields[2]];
            let mut source = plan();

            match mutation {
                "valid_same_facility_new_op" => {}
                "valid_new_case" => {
                    source.blind_holdout.observation.case_id = "case-b".into();
                    source.blind_holdout.observation.operating_point_id = "op-a".into();
                }
                "valid_new_facility" => {
                    source.blind_holdout.observation.facility_id = "facility-b".into();
                    source.blind_holdout.observation.operating_point_id = "op-a".into();
                }
                "no_scope_difference" => {
                    source.blind_holdout.observation.operating_point_id = "op-a".into();
                }
                "calibration_episode_overlap" => {
                    source.blind_holdout.observation.episode_ids = vec!["cal-shot-001".into()];
                }
                "calibration_dataset_reuse" => {
                    source.blind_holdout.observation.dataset_id = "calibration-dataset".into();
                }
                "calibration_artifact_reuse" => {
                    source.blind_holdout.observation.source_artifact_sha256 =
                        source.calibration[0].source_artifact_sha256.clone();
                }
                "holdout_training_leak" => {
                    source.blind_holdout.exposure.used_for_training = true;
                }
                "holdout_selection_leak" => {
                    source
                        .blind_holdout
                        .exposure
                        .used_for_model_or_policy_selection = true;
                }
                "synthetic_test_stand" => {
                    source.test_stand.evidence_class = EvidenceClass::ModelOutput;
                }
                "missing_uncertainty" => {
                    source
                        .blind_holdout
                        .observation
                        .measurement_uncertainty_ref
                        .clear();
                }
                "geometry_outside_domain" => {
                    source.blind_holdout.observation.geometry_sha256 = digest('e');
                }
                "unsupported_use" => {
                    source.requested_use = ModelUse::OptimizationInsideSurrogate;
                }
                "model_revision_mismatch" => {
                    source.model_revision = "model-r5".into();
                }
                "not_withheld" => {
                    source.blind_holdout.withheld_until_final_evaluation = false;
                }
                "test_holdout_episode_overlap" => {
                    source.blind_holdout.observation.episode_ids =
                        source.test_stand.episode_ids.clone();
                }
                other => panic!("unknown validation-domain mutation {other} for {case_id}"),
            }

            let actual = if validate_validation_domain(&source).is_ok() {
                "ACCEPT"
            } else {
                "REJECT"
            };
            assert_eq!(
                actual, expected,
                "validation-domain case {case_id}/{mutation}"
            );
            checked += 1;
        }

        assert_eq!(
            checked, 16,
            "all frozen validation-domain cases must execute"
        );
    }
}

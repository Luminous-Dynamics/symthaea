//! Owner-evidence contract for qualifying delivered marine services.
//!
//! This module evaluates supplied quantity/quality/authority evidence. It does
//! not implement potable-water chemistry, electrical power-quality physics,
//! calibration systems, raw-data storage, safety engineering or certification.
//! A ContractQualified result describes only the admitted evidence class; a
//! synthetic result is never promoted to a physical measurement claim.

use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvidenceClass {
    Synthetic,
    Assumed,
    ModelEstimate,
    ExternalSource,
    Measured,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PredicateOutcome {
    Pass,
    Fail,
    Unknown,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuthorizationState {
    Authorized,
    Denied,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QualificationDisposition {
    /// All declared quantity, quality, evidence-currentness and authority gates pass.
    ContractQualified,
    /// At least one known required delivery or quality condition fails.
    Unavailable,
    /// Evidence is incomplete or not yet available at the decision time.
    Unknown,
    /// Evidence identity, validity, calibration or structure is invalid.
    InvalidEvidence,
    /// A non-compensable authorization gate explicitly denies service release.
    UnsafeNotAuthorized,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum QualificationReason {
    ServiceIdentityMismatch,
    QualityProfileMismatch,
    ConfigurationMismatch,
    QuantityUnitMismatch,
    ObservationNotYetAvailable,
    OutsideEvidenceValidity,
    QuantityBelowMinimum,
    MissingQualityPredicate(String),
    UnexpectedQualityPredicate(String),
    QualityPredicateFailed(String),
    QualityPredicateUnknown(String),
    QualityPredicateInvalid(String),
    CalibrationMissing,
    CalibrationExpired,
    AuthorizationDenied,
    AuthorizationUnknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ServiceQualityContract {
    pub service_id: String,
    /// Digest of the owner-approved quality profile, not a threshold invented here.
    pub quality_profile_sha256: String,
    pub configuration_sha256: String,
    pub quantity_unit: String,
    pub minimum_rate: f64,
    /// Explicit list of owner-declared predicates required for this service profile.
    pub required_quality_predicates: Vec<String>,
    pub calibrated_measurement_required: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualityPredicateEvidence {
    pub predicate_id: String,
    pub quality_profile_sha256: String,
    pub outcome: PredicateOutcome,
    pub evidence_ref: String,
    pub raw_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualifiedServiceObservation {
    pub service_id: String,
    pub quality_profile_sha256: String,
    pub configuration_sha256: String,
    pub quantity_rate: f64,
    pub quantity_unit: String,
    /// Source event's time at which this observation became available.
    pub observed_at_seconds: u64,
    /// Evidence-validity interval [valid_from_seconds, valid_until_seconds).
    pub valid_from_seconds: u64,
    pub valid_until_seconds: Option<u64>,
    pub evidence_class: EvidenceClass,
    pub source_ref: String,
    pub raw_sha256: String,
    pub instrument_id: Option<String>,
    pub calibration_ref: Option<String>,
    /// Expiry is exclusive: calibration is stale when decision time >= expiry.
    pub calibration_valid_until_seconds: Option<u64>,
    pub authorization: AuthorizationState,
    pub quality_predicates: Vec<QualityPredicateEvidence>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractValidationError {
    EmptyServiceId,
    EmptyQuantityUnit,
    InvalidQualityProfileDigest,
    InvalidConfigurationDigest,
    InvalidMinimumRate,
    EmptyRequiredPredicate,
    DuplicateRequiredPredicate(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ObservationValidationError {
    EmptyServiceId,
    EmptyQuantityUnit,
    InvalidQualityProfileDigest,
    InvalidConfigurationDigest,
    InvalidRawDigest,
    EmptySourceReference,
    InvalidQuantityRate,
    InvalidValidityInterval,
    EmptyPredicateId,
    DuplicatePredicate(String),
    InvalidPredicateProfileDigest(String),
    EmptyPredicateEvidenceReference(String),
    InvalidPredicateRawDigest(String),
    EmptyInstrumentId,
    EmptyCalibrationReference,
    CalibrationExpiryWithoutReference,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QualificationReport {
    pub disposition: QualificationDisposition,
    pub evidence_class: EvidenceClass,
    pub decision_time_seconds: u64,
    pub service_id: String,
    pub observed_quantity_rate: f64,
    pub minimum_quantity_rate: f64,
    pub reasons: Vec<QualificationReason>,
    pub source_ref: String,
    pub raw_sha256: String,
}

pub fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

impl ServiceQualityContract {
    pub fn validate(&self) -> Result<(), ContractValidationError> {
        if self.service_id.trim().is_empty() {
            return Err(ContractValidationError::EmptyServiceId);
        }
        if self.quantity_unit.trim().is_empty() {
            return Err(ContractValidationError::EmptyQuantityUnit);
        }
        if !is_lowercase_sha256(&self.quality_profile_sha256) {
            return Err(ContractValidationError::InvalidQualityProfileDigest);
        }
        if !is_lowercase_sha256(&self.configuration_sha256) {
            return Err(ContractValidationError::InvalidConfigurationDigest);
        }
        if !self.minimum_rate.is_finite() || self.minimum_rate < 0.0 {
            return Err(ContractValidationError::InvalidMinimumRate);
        }
        let mut seen = BTreeSet::new();
        for predicate in &self.required_quality_predicates {
            if predicate.trim().is_empty() {
                return Err(ContractValidationError::EmptyRequiredPredicate);
            }
            if !seen.insert(predicate.as_str()) {
                return Err(ContractValidationError::DuplicateRequiredPredicate(
                    predicate.clone(),
                ));
            }
        }
        Ok(())
    }
}

impl QualifiedServiceObservation {
    pub fn validate(&self) -> Result<(), ObservationValidationError> {
        if self.service_id.trim().is_empty() {
            return Err(ObservationValidationError::EmptyServiceId);
        }
        if self.quantity_unit.trim().is_empty() {
            return Err(ObservationValidationError::EmptyQuantityUnit);
        }
        if !is_lowercase_sha256(&self.quality_profile_sha256) {
            return Err(ObservationValidationError::InvalidQualityProfileDigest);
        }
        if !is_lowercase_sha256(&self.configuration_sha256) {
            return Err(ObservationValidationError::InvalidConfigurationDigest);
        }
        if !is_lowercase_sha256(&self.raw_sha256) {
            return Err(ObservationValidationError::InvalidRawDigest);
        }
        if self.source_ref.trim().is_empty() {
            return Err(ObservationValidationError::EmptySourceReference);
        }
        if !self.quantity_rate.is_finite() || self.quantity_rate < 0.0 {
            return Err(ObservationValidationError::InvalidQuantityRate);
        }
        if self
            .valid_until_seconds
            .is_none_or(|until| until <= self.valid_from_seconds)
        {
            return Err(ObservationValidationError::InvalidValidityInterval);
        }
        if self.instrument_id.as_deref().is_some_and(|id| id.trim().is_empty()) {
            return Err(ObservationValidationError::EmptyInstrumentId);
        }
        if self.calibration_ref.as_deref().is_some_and(|id| id.trim().is_empty()) {
            return Err(ObservationValidationError::EmptyCalibrationReference);
        }
        if self.calibration_valid_until_seconds.is_some() && self.calibration_ref.is_none() {
            return Err(ObservationValidationError::CalibrationExpiryWithoutReference);
        }

        let mut predicate_ids = BTreeSet::new();
        for predicate in &self.quality_predicates {
            if predicate.predicate_id.trim().is_empty() {
                return Err(ObservationValidationError::EmptyPredicateId);
            }
            if !predicate_ids.insert(predicate.predicate_id.as_str()) {
                return Err(ObservationValidationError::DuplicatePredicate(
                    predicate.predicate_id.clone(),
                ));
            }
            if !is_lowercase_sha256(&predicate.quality_profile_sha256) {
                return Err(ObservationValidationError::InvalidPredicateProfileDigest(
                    predicate.predicate_id.clone(),
                ));
            }
            if predicate.evidence_ref.trim().is_empty() {
                return Err(ObservationValidationError::EmptyPredicateEvidenceReference(
                    predicate.predicate_id.clone(),
                ));
            }
            if !is_lowercase_sha256(&predicate.raw_sha256) {
                return Err(ObservationValidationError::InvalidPredicateRawDigest(
                    predicate.predicate_id.clone(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum QualificationValidationError {
    Contract(ContractValidationError),
    Observation(ObservationValidationError),
}

/// Evaluate an owner-supplied observation at a decision time.
///
/// Digests are format-checked, not authenticated. This function cannot fetch
/// source records, authenticate a signer, validate lab results, recalculate
/// uncertainty or independently establish calibration/authority validity.
pub fn evaluate_service_quality(
    contract: &ServiceQualityContract,
    observation: &QualifiedServiceObservation,
    decision_time_seconds: u64,
) -> Result<QualificationReport, QualificationValidationError> {
    contract
        .validate()
        .map_err(QualificationValidationError::Contract)?;
    observation
        .validate()
        .map_err(QualificationValidationError::Observation)?;

    let mut reasons = Vec::new();
    if observation.service_id != contract.service_id {
        reasons.push(QualificationReason::ServiceIdentityMismatch);
    }
    if observation.quality_profile_sha256 != contract.quality_profile_sha256
        || observation
            .quality_predicates
            .iter()
            .any(|predicate| predicate.quality_profile_sha256 != contract.quality_profile_sha256)
    {
        reasons.push(QualificationReason::QualityProfileMismatch);
    }
    if observation.configuration_sha256 != contract.configuration_sha256 {
        reasons.push(QualificationReason::ConfigurationMismatch);
    }
    if observation.quantity_unit != contract.quantity_unit {
        reasons.push(QualificationReason::QuantityUnitMismatch);
    }
    if observation.observed_at_seconds > decision_time_seconds {
        reasons.push(QualificationReason::ObservationNotYetAvailable);
    }
    if decision_time_seconds < observation.valid_from_seconds
        || decision_time_seconds >= observation.valid_until_seconds.unwrap_or(u64::MAX)
    {
        reasons.push(QualificationReason::OutsideEvidenceValidity);
    }
    if observation.quantity_rate < contract.minimum_rate {
        reasons.push(QualificationReason::QuantityBelowMinimum);
    }
    if contract.calibrated_measurement_required
        && observation.evidence_class == EvidenceClass::Measured
    {
        if observation.instrument_id.is_none()
            || observation.calibration_ref.is_none()
            || observation.calibration_valid_until_seconds.is_none()
        {
            reasons.push(QualificationReason::CalibrationMissing);
        } else if observation
            .calibration_valid_until_seconds
            .is_some_and(|until| decision_time_seconds >= until)
        {
            reasons.push(QualificationReason::CalibrationExpired);
        }
    } else if contract.calibrated_measurement_required
        && matches!(
            observation.evidence_class,
            EvidenceClass::Synthetic | EvidenceClass::Assumed | EvidenceClass::ModelEstimate
        )
    {
        // Preserve the synthetic/model evidence class; it cannot be promoted
        // to calibrated physical measurement by copying a digest-shaped field.
        reasons.push(QualificationReason::CalibrationMissing);
    }

    match observation.authorization {
        AuthorizationState::Authorized => {}
        AuthorizationState::Denied => reasons.push(QualificationReason::AuthorizationDenied),
        AuthorizationState::Unknown => reasons.push(QualificationReason::AuthorizationUnknown),
    }

    let predicate_by_id: BTreeMap<&str, &QualityPredicateEvidence> = observation
        .quality_predicates
        .iter()
        .map(|predicate| (predicate.predicate_id.as_str(), predicate))
        .collect();
    let required_ids: BTreeSet<&str> = contract
        .required_quality_predicates
        .iter()
        .map(String::as_str)
        .collect();
    for predicate in &observation.quality_predicates {
        if !required_ids.contains(predicate.predicate_id.as_str()) {
            reasons.push(QualificationReason::UnexpectedQualityPredicate(
                predicate.predicate_id.clone(),
            ));
        }
    }
    for required in &contract.required_quality_predicates {
        match predicate_by_id.get(required.as_str()) {
            None => reasons.push(QualificationReason::MissingQualityPredicate(
                required.clone(),
            )),
            Some(predicate) => match predicate.outcome {
                PredicateOutcome::Pass => {}
                PredicateOutcome::Fail => reasons.push(
                    QualificationReason::QualityPredicateFailed(required.clone()),
                ),
                PredicateOutcome::Unknown => reasons.push(
                    QualificationReason::QualityPredicateUnknown(required.clone()),
                ),
                PredicateOutcome::Invalid => reasons.push(
                    QualificationReason::QualityPredicateInvalid(required.clone()),
                ),
            },
        }
    }

    let disposition = if reasons.iter().any(|reason| {
        matches!(
            reason,
            QualificationReason::ServiceIdentityMismatch
                | QualificationReason::QualityProfileMismatch
                | QualificationReason::ConfigurationMismatch
                | QualificationReason::QuantityUnitMismatch
                | QualificationReason::OutsideEvidenceValidity
                | QualificationReason::CalibrationMissing
                | QualificationReason::CalibrationExpired
                | QualificationReason::QualityPredicateInvalid(_)
                | QualificationReason::UnexpectedQualityPredicate(_)
        )
    }) {
        QualificationDisposition::InvalidEvidence
    } else if reasons.contains(&QualificationReason::AuthorizationDenied) {
        QualificationDisposition::UnsafeNotAuthorized
    } else if reasons.iter().any(|reason| {
        matches!(
            reason,
            QualificationReason::QuantityBelowMinimum
                | QualificationReason::QualityPredicateFailed(_)
        )
    }) {
        QualificationDisposition::Unavailable
    } else if !reasons.is_empty() {
        QualificationDisposition::Unknown
    } else {
        QualificationDisposition::ContractQualified
    };

    Ok(QualificationReport {
        disposition,
        evidence_class: observation.evidence_class,
        decision_time_seconds,
        service_id: observation.service_id.clone(),
        observed_quantity_rate: observation.quantity_rate,
        minimum_quantity_rate: contract.minimum_rate,
        reasons,
        source_ref: observation.source_ref.clone(),
        raw_sha256: observation.raw_sha256.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const PROFILE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CONFIG: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const RAW: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const QUALITY_RAW: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn contract() -> ServiceQualityContract {
        ServiceQualityContract {
            service_id: "potable_water".into(),
            quality_profile_sha256: PROFILE.into(),
            configuration_sha256: CONFIG.into(),
            quantity_unit: "L/h".into(),
            minimum_rate: 40.0,
            required_quality_predicates: vec![
                "potability_profile".into(),
                "microbiology_profile".into(),
            ],
            calibrated_measurement_required: true,
        }
    }

    fn predicate(id: &str, outcome: PredicateOutcome) -> QualityPredicateEvidence {
        QualityPredicateEvidence {
            predicate_id: id.into(),
            quality_profile_sha256: PROFILE.into(),
            outcome,
            evidence_ref: format!("lab-result:{id}"),
            raw_sha256: QUALITY_RAW.into(),
        }
    }

    fn observation() -> QualifiedServiceObservation {
        QualifiedServiceObservation {
            service_id: "potable_water".into(),
            quality_profile_sha256: PROFILE.into(),
            configuration_sha256: CONFIG.into(),
            quantity_rate: 80.0,
            quantity_unit: "L/h".into(),
            observed_at_seconds: 10,
            valid_from_seconds: 0,
            valid_until_seconds: Some(100),
            evidence_class: EvidenceClass::Measured,
            source_ref: "campaign:los18-water-001".into(),
            raw_sha256: RAW.into(),
            instrument_id: Some("flow-meter-01".into()),
            calibration_ref: Some("calibration-epoch-7".into()),
            calibration_valid_until_seconds: Some(90),
            authorization: AuthorizationState::Authorized,
            quality_predicates: vec![
                predicate("potability_profile", PredicateOutcome::Pass),
                predicate("microbiology_profile", PredicateOutcome::Pass),
            ],
        }
    }

    #[test]
    fn quantity_and_all_quality_gates_are_required() {
        let report = evaluate_service_quality(&contract(), &observation(), 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::ContractQualified);
        assert_eq!(report.evidence_class, EvidenceClass::Measured);
    }

    #[test]
    fn sufficient_rate_cannot_compensate_for_failed_quality() {
        let mut input = observation();
        input.quality_predicates[0].outcome = PredicateOutcome::Fail;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::Unavailable);
        assert!(report.reasons.contains(&QualificationReason::QualityPredicateFailed(
            "potability_profile".into()
        )));
    }

    #[test]
    fn missing_quality_is_unknown_not_pass() {
        let mut input = observation();
        input.quality_predicates.pop();
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert!(report.reasons.contains(&QualificationReason::MissingQualityPredicate(
            "microbiology_profile".into()
        )));
    }

    #[test]
    fn unexpected_quality_predicate_invalidates_the_observation() {
        let mut input = observation();
        input.quality_predicates.push(predicate("unregistered-predicate", PredicateOutcome::Pass));
        assert_eq!(
            evaluate_service_quality(&contract(), &input, 50).unwrap().disposition,
            QualificationDisposition::InvalidEvidence
        );
    }

    #[test]
    fn future_observation_is_not_available_at_decision_time() {
        let mut input = observation();
        input.observed_at_seconds = 60;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
        assert!(report.reasons.contains(&QualificationReason::ObservationNotYetAvailable));
    }

    #[test]
    fn expired_calibration_cannot_qualify_measured_evidence() {
        let mut input = observation();
        input.calibration_valid_until_seconds = Some(49);
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::InvalidEvidence);
        assert!(report.reasons.contains(&QualificationReason::CalibrationExpired));
    }

    #[test]
    fn wrong_configuration_or_quality_profile_is_invalid() {
        let mut input = observation();
        input.configuration_sha256 = PROFILE.into();
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::InvalidEvidence);

        let mut wrong_profile = observation();
        wrong_profile.quality_profile_sha256 = CONFIG.into();
        let report = evaluate_service_quality(&contract(), &wrong_profile, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::InvalidEvidence);
    }

    #[test]
    fn unknown_authorization_cannot_qualify_service() {
        let mut input = observation();
        input.authorization = AuthorizationState::Unknown;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::Unknown);
    }

    #[test]
    fn denied_authorization_is_non_compensable() {
        let mut input = observation();
        input.authorization = AuthorizationState::Denied;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::UnsafeNotAuthorized);
    }

    #[test]
    fn synthetic_measurement_cannot_satisfy_calibration_requirement() {
        let mut input = observation();
        input.evidence_class = EvidenceClass::Synthetic;
        input.instrument_id = None;
        input.calibration_ref = None;
        input.calibration_valid_until_seconds = None;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::InvalidEvidence);
        assert_eq!(report.evidence_class, EvidenceClass::Synthetic);
    }

    #[test]
    fn validity_interval_is_half_open() {
        let input = observation();
        assert_eq!(
            evaluate_service_quality(&contract(), &input, 99).unwrap().disposition,
            QualificationDisposition::ContractQualified
        );
        assert_eq!(
            evaluate_service_quality(&contract(), &input, 100).unwrap().disposition,
            QualificationDisposition::InvalidEvidence
        );
    }

    #[test]
    fn malformed_digest_and_duplicate_predicates_are_rejected() {
        let mut input = observation();
        input.raw_sha256 = "not-a-digest".into();
        assert_eq!(
            input.validate(),
            Err(ObservationValidationError::InvalidRawDigest)
        );

        let mut duplicate = observation();
        duplicate
            .quality_predicates
            .push(predicate("potability_profile", PredicateOutcome::Pass));
        assert_eq!(
            duplicate.validate(),
            Err(ObservationValidationError::DuplicatePredicate(
                "potability_profile".into()
            ))
        );
    }

    #[test]
    fn low_quantity_is_unavailable_even_when_quality_passes() {
        let mut input = observation();
        input.quantity_rate = 20.0;
        let report = evaluate_service_quality(&contract(), &input, 50).unwrap();
        assert_eq!(report.disposition, QualificationDisposition::Unavailable);
        assert!(report.reasons.contains(&QualificationReason::QuantityBelowMinimum));
    }

    #[test]
    fn quantity_unit_mismatch_cannot_be_converted_silently() {
        let mut input = observation();
        input.quantity_unit = "gal/min".into();
        assert_eq!(
            evaluate_service_quality(&contract(), &input, 50).unwrap().disposition,
            QualificationDisposition::InvalidEvidence
        );
    }
}

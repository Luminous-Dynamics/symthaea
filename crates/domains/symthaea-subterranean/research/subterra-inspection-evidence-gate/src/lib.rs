//! Fail-closed provenance and freshness admission for campaign inspection options.
//!
//! This is a consumer boundary. It accepts evidence-verification and rights
//! dispositions from separately governed services; it does not authenticate
//! signatures, determine legal rights, calibrate sensors, or validate EVSI.

use subterra_inspection_campaign::InspectionOption;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidencePurpose {
    AssetConditionInspection,
    MaintenancePlanning,
    ResearchEvaluation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightsDisposition {
    AuthorizedForDeclaredPurpose,
    Denied,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiptVerificationDisposition {
    VerifiedByExternalService,
    RejectedByExternalService,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialScopeDisposition {
    CompleteWithinDeclaredScope,
    Partial,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedInspectionContext {
    pub decision_id: String,
    pub option_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub sensor_model_digest: String,
    pub likelihood_model_digest: String,
    pub loss_table_digest: String,
    pub evsi_assessment_digest: String,
    pub source_dataset_digest: String,
    pub purpose: EvidencePurpose,
    /// Frozen by the caller for deterministic, reproducible freshness checks.
    pub evaluation_time_unix_ms: u64,
    pub maximum_observation_age_ms: u64,
    pub allowed_future_clock_skew_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceManifest {
    pub candidate_id: String,
    pub decision_id: String,
    pub option_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub sensor_model_digest: String,
    pub likelihood_model_digest: String,
    pub loss_table_digest: String,
    pub evsi_assessment_digest: String,
    pub source_dataset_digest: String,
    pub observation_receipt_digest: String,
    pub calibration_receipt_digest: String,
    pub rights_receipt_digest: String,
    /// Digest/reference to the external service's signature-verification receipt.
    pub verifier_receipt_digest: String,
    pub observation_captured_at_unix_ms: u64,
    pub evidence_valid_until_unix_ms: u64,
    pub calibration_valid_from_unix_ms: u64,
    pub calibration_valid_until_unix_ms: u64,
    pub rights_valid_until_unix_ms: u64,
    pub rights_purpose: EvidencePurpose,
    pub rights_disposition: RightsDisposition,
    pub receipt_verification: ReceiptVerificationDisposition,
    pub spatial_scope: SpatialScopeDisposition,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateInspectionOption {
    pub option: InspectionOption,
    pub manifest: EvidenceManifest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectionReason {
    EmptyCandidateId,
    EmptyOptionId,
    EmptyRequiredDigest,
    ExternalOptionGateNotPassed,
    DecisionMismatch,
    OptionIdMismatch,
    AssetMismatch,
    AssetRevisionMismatch,
    CoordinateFrameMismatch,
    RegionScopeMismatch,
    SensorModelMismatch,
    LikelihoodModelMismatch,
    LossTableMismatch,
    EvsiAssessmentMismatch,
    RightsPurposeMismatch,
    RightsDenied,
    RightsUnknown,
    ReceiptRejected,
    ReceiptVerificationUnknown,
    MissingObservationReceipt,
    MissingCalibrationReceipt,
    MissingRightsReceipt,
    MissingVerifierReceipt,
    MissingSourceDatasetDigest,
    SourceDatasetMismatch,
    ObservationFromFuture,
    ObservationStale,
    EvidenceExpired,
    CalibrationNotYetValid,
    CalibrationExpired,
    RightsGrantExpired,
    SpatialScopePartial,
    SpatialScopeUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedInspectionOption {
    pub option: InspectionOption,
    /// Preserve the complete evidence manifest with the admitted campaign option.
    pub manifest: EvidenceManifest,
}

impl AdmittedInspectionOption {
    /// Extract the value object expected by the campaign solver. Callers should
    /// persist this wrapper alongside the campaign so the evidence lineage is retained.
    pub fn campaign_option(&self) -> InspectionOption {
        self.option.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceAdmissionReport {
    pub candidate_id: String,
    pub admitted_candidate: Option<AdmittedInspectionOption>,
    pub rejection_reasons: Vec<RejectionReason>,
    /// Retain these references even for rejection so auditors can identify
    /// which upstream objects were presented.
    pub evidence_references: Vec<String>,
}

impl EvidenceAdmissionReport {
    pub fn admitted(&self) -> bool {
        self.admitted_candidate.is_some() && self.rejection_reasons.is_empty()
    }
}

fn missing(value: &str) -> bool {
    value.trim().is_empty()
}

fn push_if_mismatch(reasons: &mut Vec<RejectionReason>, mismatch: bool, reason: RejectionReason) {
    if mismatch {
        reasons.push(reason);
    }
}

pub fn admit_candidate(
    candidate: &CandidateInspectionOption,
    expected: &ExpectedInspectionContext,
) -> EvidenceAdmissionReport {
    let manifest = &candidate.manifest;
    let option = &candidate.option;
    let mut reasons = Vec::new();

    if missing(&manifest.candidate_id) {
        reasons.push(RejectionReason::EmptyCandidateId);
    }
    if missing(&option.option_id) {
        reasons.push(RejectionReason::EmptyOptionId);
    }
    if !option.external_gate_passed {
        reasons.push(RejectionReason::ExternalOptionGateNotPassed);
    }

    for digest in [
        &expected.coordinate_frame_digest,
        &expected.region_scope_digest,
        &expected.sensor_model_digest,
        &expected.likelihood_model_digest,
        &expected.loss_table_digest,
        &expected.evsi_assessment_digest,
        &expected.source_dataset_digest,
        &manifest.coordinate_frame_digest,
        &manifest.region_scope_digest,
        &manifest.sensor_model_digest,
        &manifest.likelihood_model_digest,
        &manifest.loss_table_digest,
        &manifest.evsi_assessment_digest,
    ] {
        if missing(digest) {
            reasons.push(RejectionReason::EmptyRequiredDigest);
            break;
        }
    }

    push_if_mismatch(
        &mut reasons,
        missing(&expected.decision_id)
            || missing(&option.decision_id)
            || missing(&manifest.decision_id)
            || option.decision_id != expected.decision_id
            || manifest.decision_id != expected.decision_id
            || option.decision_id != manifest.decision_id,
        RejectionReason::DecisionMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        missing(&expected.option_id)
            || missing(&manifest.option_id)
            || manifest.option_id != option.option_id
            || manifest.option_id != expected.option_id,
        RejectionReason::OptionIdMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        missing(&expected.asset_id)
            || missing(&manifest.asset_id)
            || manifest.asset_id != expected.asset_id,
        RejectionReason::AssetMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        missing(&expected.asset_revision)
            || missing(&manifest.asset_revision)
            || manifest.asset_revision != expected.asset_revision,
        RejectionReason::AssetRevisionMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.coordinate_frame_digest != expected.coordinate_frame_digest,
        RejectionReason::CoordinateFrameMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.region_scope_digest != expected.region_scope_digest,
        RejectionReason::RegionScopeMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.sensor_model_digest != expected.sensor_model_digest,
        RejectionReason::SensorModelMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.likelihood_model_digest != expected.likelihood_model_digest,
        RejectionReason::LikelihoodModelMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.loss_table_digest != expected.loss_table_digest,
        RejectionReason::LossTableMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.evsi_assessment_digest != expected.evsi_assessment_digest,
        RejectionReason::EvsiAssessmentMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.source_dataset_digest != expected.source_dataset_digest,
        RejectionReason::SourceDatasetMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.rights_purpose != expected.purpose,
        RejectionReason::RightsPurposeMismatch,
    );

    match manifest.rights_disposition {
        RightsDisposition::AuthorizedForDeclaredPurpose => {}
        RightsDisposition::Denied => reasons.push(RejectionReason::RightsDenied),
        RightsDisposition::Unknown => reasons.push(RejectionReason::RightsUnknown),
    }
    match manifest.receipt_verification {
        ReceiptVerificationDisposition::VerifiedByExternalService => {}
        ReceiptVerificationDisposition::RejectedByExternalService => {
            reasons.push(RejectionReason::ReceiptRejected);
        }
        ReceiptVerificationDisposition::Unknown => {
            reasons.push(RejectionReason::ReceiptVerificationUnknown);
        }
    }

    if missing(&manifest.observation_receipt_digest) {
        reasons.push(RejectionReason::MissingObservationReceipt);
    }
    if missing(&manifest.calibration_receipt_digest) {
        reasons.push(RejectionReason::MissingCalibrationReceipt);
    }
    if missing(&manifest.rights_receipt_digest) {
        reasons.push(RejectionReason::MissingRightsReceipt);
    }
    if missing(&manifest.verifier_receipt_digest) {
        reasons.push(RejectionReason::MissingVerifierReceipt);
    }
    if missing(&manifest.source_dataset_digest) {
        reasons.push(RejectionReason::MissingSourceDatasetDigest);
    }

    let latest_allowed_observation = expected
        .evaluation_time_unix_ms
        .saturating_add(expected.allowed_future_clock_skew_ms);
    if manifest.observation_captured_at_unix_ms > latest_allowed_observation {
        reasons.push(RejectionReason::ObservationFromFuture);
    } else if expected
        .evaluation_time_unix_ms
        .saturating_sub(manifest.observation_captured_at_unix_ms)
        > expected.maximum_observation_age_ms
    {
        reasons.push(RejectionReason::ObservationStale);
    }
    if manifest.evidence_valid_until_unix_ms <= expected.evaluation_time_unix_ms {
        reasons.push(RejectionReason::EvidenceExpired);
    }
    if expected.evaluation_time_unix_ms < manifest.calibration_valid_from_unix_ms {
        reasons.push(RejectionReason::CalibrationNotYetValid);
    }
    if manifest.calibration_valid_until_unix_ms <= expected.evaluation_time_unix_ms {
        reasons.push(RejectionReason::CalibrationExpired);
    }
    if manifest.rights_valid_until_unix_ms <= expected.evaluation_time_unix_ms {
        reasons.push(RejectionReason::RightsGrantExpired);
    }
    if manifest.observation_captured_at_unix_ms < manifest.calibration_valid_from_unix_ms
        || manifest.observation_captured_at_unix_ms >= manifest.calibration_valid_until_unix_ms
    {
        reasons.push(RejectionReason::CalibrationNotYetValid);
    }

    match manifest.spatial_scope {
        SpatialScopeDisposition::CompleteWithinDeclaredScope => {}
        SpatialScopeDisposition::Partial => reasons.push(RejectionReason::SpatialScopePartial),
        SpatialScopeDisposition::Unknown => reasons.push(RejectionReason::SpatialScopeUnknown),
    }

    reasons.sort();
    reasons.dedup();

    let admitted = reasons.is_empty();
    EvidenceAdmissionReport {
        candidate_id: manifest.candidate_id.clone(),
        admitted_candidate: admitted.then(|| AdmittedInspectionOption {
            option: option.clone(),
            manifest: manifest.clone(),
        }),
        rejection_reasons: reasons,
        evidence_references: vec![
            manifest.source_dataset_digest.clone(),
            manifest.observation_receipt_digest.clone(),
            manifest.calibration_receipt_digest.clone(),
            manifest.rights_receipt_digest.clone(),
            manifest.verifier_receipt_digest.clone(),
            manifest.evsi_assessment_digest.clone(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subterra_inspection_campaign::ResourceVector;

    const NOW: u64 = 1_800_000_000_000;

    fn context() -> ExpectedInspectionContext {
        ExpectedInspectionContext {
            decision_id: "support-08".into(),
            option_id: "support-08-thermal-followup".into(),
            asset_id: "asset-support-08".into(),
            asset_revision: "rev-17".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            sensor_model_digest: "sha256:sensor-model".into(),
            likelihood_model_digest: "sha256:likelihood".into(),
            loss_table_digest: "sha256:loss-table".into(),
            evsi_assessment_digest: "sha256:evsi".into(),
            source_dataset_digest: "sha256:dataset".into(),
            purpose: EvidencePurpose::AssetConditionInspection,
            evaluation_time_unix_ms: NOW,
            maximum_observation_age_ms: 3_600_000,
            allowed_future_clock_skew_ms: 5_000,
        }
    }

    fn candidate() -> CandidateInspectionOption {
        CandidateInspectionOption {
            option: InspectionOption {
                option_id: "support-08-thermal-followup".into(),
                decision_id: "support-08".into(),
                deployment_bundle_id: "corridor-west".into(),
                gross_evsi_micro_units: 250_000,
                unknown_state_debt_reduction_units: 20,
                variable_resources: ResourceVector {
                    energy_millijoules: 5_000,
                    duration_milliseconds: 20_000,
                    data_bytes: 1_024,
                    exposure_milliunits: 2,
                },
                mission_interruption_upper_ppm: 100,
                robot_loss_upper_ppm: 10,
                external_gate_passed: true,
            },
            manifest: EvidenceManifest {
                candidate_id: "candidate-2026-014".into(),
                decision_id: "support-08".into(),
                option_id: "support-08-thermal-followup".into(),
                asset_id: "asset-support-08".into(),
                asset_revision: "rev-17".into(),
                coordinate_frame_digest: "sha256:frame".into(),
                region_scope_digest: "sha256:region".into(),
                sensor_model_digest: "sha256:sensor-model".into(),
                likelihood_model_digest: "sha256:likelihood".into(),
                loss_table_digest: "sha256:loss-table".into(),
                evsi_assessment_digest: "sha256:evsi".into(),
                source_dataset_digest: "sha256:dataset".into(),
                observation_receipt_digest: "sha256:observation".into(),
                calibration_receipt_digest: "sha256:calibration".into(),
                rights_receipt_digest: "sha256:rights".into(),
                verifier_receipt_digest: "sha256:external-verifier".into(),
                observation_captured_at_unix_ms: NOW - 60_000,
                evidence_valid_until_unix_ms: NOW + 600_000,
                calibration_valid_from_unix_ms: NOW - 600_000,
                calibration_valid_until_unix_ms: NOW + 600_000,
                rights_valid_until_unix_ms: NOW + 600_000,
                rights_purpose: EvidencePurpose::AssetConditionInspection,
                rights_disposition: RightsDisposition::AuthorizedForDeclaredPurpose,
                receipt_verification: ReceiptVerificationDisposition::VerifiedByExternalService,
                spatial_scope: SpatialScopeDisposition::CompleteWithinDeclaredScope,
            },
        }
    }

    #[test]
    fn exact_fresh_verified_candidate_is_admitted() {
        let report = admit_candidate(&candidate(), &context());
        assert!(report.admitted());
        assert!(report.rejection_reasons.is_empty());
        assert!(report.admitted_candidate.is_some());
        let admitted = report.admitted_candidate.as_ref().unwrap();
        assert_eq!(admitted.manifest.asset_revision, "rev-17");
        assert_eq!(
            admitted.campaign_option().option_id,
            "support-08-thermal-followup"
        );
        assert_eq!(report.evidence_references.len(), 6);
    }

    #[test]
    fn stale_observation_is_rejected() {
        let mut item = candidate();
        item.manifest.observation_captured_at_unix_ms = NOW - 3_600_001;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ObservationStale)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn future_dated_observation_beyond_clock_skew_is_rejected() {
        let mut item = candidate();
        item.manifest.observation_captured_at_unix_ms = NOW + 5_001;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ObservationFromFuture)
        );
    }

    #[test]
    fn exact_future_clock_skew_boundary_is_accepted() {
        let mut item = candidate();
        item.manifest.observation_captured_at_unix_ms = NOW + 5_000;
        assert!(admit_candidate(&item, &context()).admitted());
    }

    #[test]
    fn expired_calibration_is_rejected() {
        let mut item = candidate();
        item.manifest.calibration_valid_until_unix_ms = NOW;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::CalibrationExpired)
        );
    }

    #[test]
    fn expired_rights_grant_is_rejected() {
        let mut item = candidate();
        item.manifest.rights_valid_until_unix_ms = NOW;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::RightsGrantExpired)
        );
    }

    #[test]
    fn calibration_not_yet_valid_is_rejected() {
        let mut item = candidate();
        item.manifest.calibration_valid_from_unix_ms = NOW + 1;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::CalibrationNotYetValid)
        );
    }

    #[test]
    fn expired_evidence_is_rejected() {
        let mut item = candidate();
        item.manifest.evidence_valid_until_unix_ms = NOW;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::EvidenceExpired)
        );
    }

    #[test]
    fn asset_revision_mismatch_is_rejected() {
        let mut item = candidate();
        item.manifest.asset_revision = "rev-16".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::AssetRevisionMismatch)
        );
    }

    #[test]
    fn coordinate_frame_mismatch_is_rejected() {
        let mut item = candidate();
        item.manifest.coordinate_frame_digest = "sha256:wrong-frame".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::CoordinateFrameMismatch)
        );
    }

    #[test]
    fn changed_likelihood_or_loss_table_invalidates_evsi() {
        let mut item = candidate();
        item.manifest.likelihood_model_digest = "sha256:new-likelihood".into();
        item.manifest.loss_table_digest = "sha256:new-loss-table".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::LikelihoodModelMismatch)
        );
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::LossTableMismatch)
        );
    }

    #[test]
    fn changed_evsi_assessment_is_rejected() {
        let mut item = candidate();
        item.manifest.evsi_assessment_digest = "sha256:other-evsi".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::EvsiAssessmentMismatch)
        );
    }

    #[test]
    fn denied_or_unknown_rights_are_rejected() {
        let mut denied = candidate();
        denied.manifest.rights_disposition = RightsDisposition::Denied;
        assert!(
            admit_candidate(&denied, &context())
                .rejection_reasons
                .contains(&RejectionReason::RightsDenied)
        );

        let mut unknown = candidate();
        unknown.manifest.rights_disposition = RightsDisposition::Unknown;
        assert!(
            admit_candidate(&unknown, &context())
                .rejection_reasons
                .contains(&RejectionReason::RightsUnknown)
        );
    }

    #[test]
    fn unverified_or_revoked_receipt_disposition_is_rejected() {
        let mut unknown = candidate();
        unknown.manifest.receipt_verification = ReceiptVerificationDisposition::Unknown;
        assert!(
            admit_candidate(&unknown, &context())
                .rejection_reasons
                .contains(&RejectionReason::ReceiptVerificationUnknown)
        );

        let mut rejected = candidate();
        rejected.manifest.receipt_verification =
            ReceiptVerificationDisposition::RejectedByExternalService;
        assert!(
            admit_candidate(&rejected, &context())
                .rejection_reasons
                .contains(&RejectionReason::ReceiptRejected)
        );
    }

    #[test]
    fn partial_or_unknown_spatial_coverage_is_rejected() {
        let mut partial = candidate();
        partial.manifest.spatial_scope = SpatialScopeDisposition::Partial;
        assert!(
            admit_candidate(&partial, &context())
                .rejection_reasons
                .contains(&RejectionReason::SpatialScopePartial)
        );

        let mut unknown = candidate();
        unknown.manifest.spatial_scope = SpatialScopeDisposition::Unknown;
        assert!(
            admit_candidate(&unknown, &context())
                .rejection_reasons
                .contains(&RejectionReason::SpatialScopeUnknown)
        );
    }

    #[test]
    fn missing_verifier_receipt_is_rejected_even_when_status_says_verified() {
        let mut item = candidate();
        item.manifest.verifier_receipt_digest.clear();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::MissingVerifierReceipt)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn option_decision_mismatch_and_external_gate_failure_are_rejected() {
        let mut item = candidate();
        item.option.decision_id = "other-decision".into();
        item.option.external_gate_passed = false;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::DecisionMismatch)
        );
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ExternalOptionGateNotPassed)
        );
    }

    #[test]
    fn candidate_option_id_must_match_expected_and_manifest() {
        let mut item = candidate();
        item.manifest.option_id = "another-action".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::OptionIdMismatch)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn source_dataset_digest_must_match_frozen_context() {
        let mut item = candidate();
        item.manifest.source_dataset_digest = "sha256:substituted-dataset".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::SourceDatasetMismatch)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn missing_required_digest_is_rejected() {
        let mut item = candidate();
        item.manifest.sensor_model_digest.clear();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::EmptyRequiredDigest)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn freshness_evaluation_is_deterministic_for_frozen_context() {
        let item = candidate();
        let expected = context();
        assert_eq!(
            admit_candidate(&item, &expected),
            admit_candidate(&item, &expected)
        );
    }
}

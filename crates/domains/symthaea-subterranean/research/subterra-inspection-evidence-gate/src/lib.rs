//! Fail-closed provenance and freshness admission for campaign inspection options.
//!
//! This is a consumer boundary. It accepts evidence-verification and rights
//! dispositions from separately governed services; it does not authenticate
//! signatures, determine legal rights, calibrate sensors, or validate EVSI.

use std::collections::{BTreeMap, BTreeSet};

use subterra_inspection_campaign::{
    AssuranceBindings, CampaignBudget, CampaignError, CampaignProblem, CampaignReport,
    DecisionRequirement, DeploymentBundle, InspectionOption, ResourceVector, optimize_campaign,
};

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
    pub deployment_bundle_id: String,
    pub deployment_bundle_digest: String,
    pub resource_model_digest: String,
    pub resource_assessment_digest: String,
    pub risk_model_digest: String,
    pub risk_assessment_digest: String,
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
    pub deployment_bundle_id: String,
    pub deployment_bundle_digest: String,
    pub resource_model_digest: String,
    pub resource_assessment_digest: String,
    pub risk_model_digest: String,
    pub risk_assessment_digest: String,
    /// Resource/risk values claimed by the external assessment and matched exactly to the option.
    pub assessed_resources: ResourceVector,
    pub assessed_mission_interruption_upper_ppm: u32,
    pub assessed_robot_loss_upper_ppm: u32,
    pub resource_assessment_valid_until_unix_ms: u64,
    pub risk_assessment_valid_until_unix_ms: u64,
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
    DeploymentBundleMismatch,
    DeploymentBundleDigestMismatch,
    ResourceModelMismatch,
    ResourceAssessmentMismatch,
    ResourceValueMismatch,
    RiskModelMismatch,
    RiskAssessmentMismatch,
    RiskValueMismatch,
    ResourceAssessmentExpired,
    RiskAssessmentExpired,
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
    option: InspectionOption,
    /// Preserve the complete evidence manifest with the admitted campaign option.
    manifest: EvidenceManifest,
}

impl AdmittedInspectionOption {
    /// Read the exact option identity without exposing a bare campaign option.
    pub fn option_id(&self) -> &str {
        &self.option.option_id
    }

    /// Read the full manifest without separating it from this admitted wrapper.
    pub fn manifest(&self) -> &EvidenceManifest {
        &self.manifest
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
        &expected.deployment_bundle_digest,
        &expected.resource_model_digest,
        &expected.resource_assessment_digest,
        &expected.risk_model_digest,
        &expected.risk_assessment_digest,
        &manifest.deployment_bundle_digest,
        &manifest.resource_model_digest,
        &manifest.resource_assessment_digest,
        &manifest.risk_model_digest,
        &manifest.risk_assessment_digest,
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
        missing(&expected.deployment_bundle_id)
            || missing(&manifest.deployment_bundle_id)
            || manifest.deployment_bundle_id != expected.deployment_bundle_id
            || option.deployment_bundle_id != expected.deployment_bundle_id,
        RejectionReason::DeploymentBundleMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.deployment_bundle_digest != expected.deployment_bundle_digest,
        RejectionReason::DeploymentBundleDigestMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.resource_model_digest != expected.resource_model_digest,
        RejectionReason::ResourceModelMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.resource_assessment_digest != expected.resource_assessment_digest,
        RejectionReason::ResourceAssessmentMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.assessed_resources != option.variable_resources,
        RejectionReason::ResourceValueMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.risk_model_digest != expected.risk_model_digest,
        RejectionReason::RiskModelMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.risk_assessment_digest != expected.risk_assessment_digest,
        RejectionReason::RiskAssessmentMismatch,
    );
    push_if_mismatch(
        &mut reasons,
        manifest.assessed_mission_interruption_upper_ppm != option.mission_interruption_upper_ppm
            || manifest.assessed_robot_loss_upper_ppm != option.robot_loss_upper_ppm,
        RejectionReason::RiskValueMismatch,
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
    if manifest.resource_assessment_valid_until_unix_ms <= expected.evaluation_time_unix_ms {
        reasons.push(RejectionReason::ResourceAssessmentExpired);
    }
    if manifest.risk_assessment_valid_until_unix_ms <= expected.evaluation_time_unix_ms {
        reasons.push(RejectionReason::RiskAssessmentExpired);
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
            manifest.deployment_bundle_digest.clone(),
            manifest.resource_model_digest.clone(),
            manifest.resource_assessment_digest.clone(),
            manifest.risk_model_digest.clone(),
            manifest.risk_assessment_digest.clone(),
        ],
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBoundCampaignConfig {
    pub campaign_id: String,
    pub decisions: Vec<DecisionRequirement>,
    pub deployment_bundles: Vec<DeploymentBundle>,
    pub budget: CampaignBudget,
    pub assurance: AssuranceBindings,
    pub maximum_search_nodes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedCampaignCandidate {
    pub candidate_id: String,
    pub option_id: String,
    pub reasons: Vec<RejectionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CampaignAdmissionError {
    EmptyCandidateSet,
    DuplicateContextOptionId(String),
    DuplicateCandidateOptionId(String),
    DuplicateCandidateId(String),
    MissingContextForOption(String),
    RejectedCandidates(Vec<RejectedCampaignCandidate>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBoundCampaignReport {
    pub campaign: CampaignReport,
    /// All admitted manifests, including candidates not selected by the optimizer.
    pub candidate_evidence: Vec<EvidenceManifest>,
    /// Manifests corresponding exactly to options selected by the optimizer.
    pub selected_evidence: Vec<EvidenceManifest>,
}

/// Campaign wrapper that can only be constructed through evidence admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceBoundCampaign {
    problem: CampaignProblem,
    evidence_manifests: Vec<EvidenceManifest>,
}

impl EvidenceBoundCampaign {
    pub fn candidate_count(&self) -> usize {
        self.problem.options.len()
    }

    pub fn optimize(&self) -> Result<EvidenceBoundCampaignReport, CampaignError> {
        let campaign = optimize_campaign(&self.problem)?;
        let selected_ids = campaign
            .selection
            .selected_option_ids
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let selected_evidence = self
            .evidence_manifests
            .iter()
            .filter(|manifest| selected_ids.contains(manifest.option_id.as_str()))
            .cloned()
            .collect();

        Ok(EvidenceBoundCampaignReport {
            campaign,
            candidate_evidence: self.evidence_manifests.clone(),
            selected_evidence,
        })
    }
}

/// Build a campaign only when every supplied candidate is admitted by its
/// exact frozen context. A single rejected candidate aborts the whole build;
/// callers cannot accidentally continue with a partial, silently filtered set.
pub fn build_evidence_bound_campaign(
    mut candidates: Vec<CandidateInspectionOption>,
    expected_contexts: &[ExpectedInspectionContext],
    config: EvidenceBoundCampaignConfig,
) -> Result<EvidenceBoundCampaign, CampaignAdmissionError> {
    if candidates.is_empty() {
        return Err(CampaignAdmissionError::EmptyCandidateSet);
    }

    let mut contexts = BTreeMap::new();
    for context in expected_contexts {
        if contexts
            .insert(context.option_id.as_str(), context)
            .is_some()
        {
            return Err(CampaignAdmissionError::DuplicateContextOptionId(
                context.option_id.clone(),
            ));
        }
    }

    candidates.sort_by(|a, b| a.option.option_id.cmp(&b.option.option_id));
    let mut seen_options = BTreeSet::new();
    let mut seen_candidate_ids = BTreeSet::new();
    let mut admitted_options = Vec::with_capacity(candidates.len());
    let mut evidence_manifests = Vec::with_capacity(candidates.len());
    let mut rejected = Vec::new();

    for candidate in &candidates {
        let candidate_id = candidate.manifest.candidate_id.trim();
        if !candidate_id.is_empty() && !seen_candidate_ids.insert(candidate_id) {
            return Err(CampaignAdmissionError::DuplicateCandidateId(
                candidate_id.to_string(),
            ));
        }
        let option_id = candidate.option.option_id.as_str();
        if !seen_options.insert(option_id) {
            return Err(CampaignAdmissionError::DuplicateCandidateOptionId(
                option_id.to_string(),
            ));
        }
        let Some(context) = contexts.get(option_id) else {
            return Err(CampaignAdmissionError::MissingContextForOption(
                option_id.to_string(),
            ));
        };
        let report = admit_candidate(candidate, context);
        if let Some(admitted) = report.admitted_candidate {
            admitted_options.push(admitted.option);
            evidence_manifests.push(admitted.manifest);
        } else {
            rejected.push(RejectedCampaignCandidate {
                candidate_id: report.candidate_id,
                option_id: candidate.option.option_id.clone(),
                reasons: report.rejection_reasons,
            });
        }
    }

    if !rejected.is_empty() {
        return Err(CampaignAdmissionError::RejectedCandidates(rejected));
    }

    let problem = CampaignProblem {
        campaign_id: config.campaign_id,
        decisions: config.decisions,
        deployment_bundles: config.deployment_bundles,
        options: admitted_options,
        budget: config.budget,
        assurance: config.assurance,
        maximum_search_nodes: config.maximum_search_nodes,
    };
    evidence_manifests.sort_by(|a, b| a.option_id.cmp(&b.option_id));

    Ok(EvidenceBoundCampaign {
        problem,
        evidence_manifests,
    })
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
            deployment_bundle_id: "corridor-west".into(),
            deployment_bundle_digest: "sha256:deployment-bundle".into(),
            resource_model_digest: "sha256:resource-model".into(),
            resource_assessment_digest: "sha256:resource-assessment".into(),
            risk_model_digest: "sha256:risk-model".into(),
            risk_assessment_digest: "sha256:risk-assessment".into(),
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
                deployment_bundle_id: "corridor-west".into(),
                deployment_bundle_digest: "sha256:deployment-bundle".into(),
                resource_model_digest: "sha256:resource-model".into(),
                resource_assessment_digest: "sha256:resource-assessment".into(),
                risk_model_digest: "sha256:risk-model".into(),
                risk_assessment_digest: "sha256:risk-assessment".into(),
                assessed_resources: ResourceVector {
                    energy_millijoules: 5_000,
                    duration_milliseconds: 20_000,
                    data_bytes: 1_024,
                    exposure_milliunits: 2,
                },
                assessed_mission_interruption_upper_ppm: 100,
                assessed_robot_loss_upper_ppm: 10,
                resource_assessment_valid_until_unix_ms: NOW + 600_000,
                risk_assessment_valid_until_unix_ms: NOW + 600_000,
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
        assert_eq!(admitted.manifest().asset_revision, "rev-17");
        assert_eq!(admitted.option_id(), "support-08-thermal-followup");
        assert_eq!(report.evidence_references.len(), 11);
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
    fn altered_resource_vector_cannot_enter_campaign_option() {
        let mut item = candidate();
        item.option.variable_resources.energy_millijoules += 1;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ResourceValueMismatch)
        );
        assert!(report.admitted_candidate.is_none());
    }

    #[test]
    fn altered_risk_bounds_cannot_enter_campaign_option() {
        let mut item = candidate();
        item.option.robot_loss_upper_ppm += 1;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::RiskValueMismatch)
        );
    }

    #[test]
    fn resource_and_risk_assessment_digests_must_match() {
        let mut item = candidate();
        item.manifest.resource_assessment_digest = "sha256:other-resource".into();
        item.manifest.risk_assessment_digest = "sha256:other-risk".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ResourceAssessmentMismatch)
        );
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::RiskAssessmentMismatch)
        );
    }

    #[test]
    fn resource_and_risk_assessments_must_be_fresh() {
        let mut item = candidate();
        item.manifest.resource_assessment_valid_until_unix_ms = NOW;
        item.manifest.risk_assessment_valid_until_unix_ms = NOW;
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::ResourceAssessmentExpired)
        );
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::RiskAssessmentExpired)
        );
    }

    #[test]
    fn deployment_bundle_identity_is_bound_to_the_candidate() {
        let mut item = candidate();
        item.option.deployment_bundle_id = "corridor-east".into();
        let report = admit_candidate(&item, &context());
        assert!(
            report
                .rejection_reasons
                .contains(&RejectionReason::DeploymentBundleMismatch)
        );
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

    fn campaign_budget() -> CampaignBudget {
        CampaignBudget {
            resources: ResourceVector {
                energy_millijoules: 100_000,
                duration_milliseconds: 100_000,
                data_bytes: 100_000,
                exposure_milliunits: 100_000,
            },
            mission_interruption_upper_ppm: 100_000,
            robot_loss_upper_ppm: 100_000,
        }
    }

    fn campaign_assurance() -> AssuranceBindings {
        AssuranceBindings {
            common_cause_graph_digest: "synthetic:graph".into(),
            fault_domain_profile_digest: "synthetic:profile".into(),
            common_cause_assessment_digest: "synthetic:assessment".into(),
            portfolio_value_assessment_digest: "synthetic:factorization".into(),
            inventory_disposition: subterra_inspection_campaign::InventoryDisposition::Unknown,
            additive_value_factorization_complete: false,
        }
    }

    fn campaign_decisions() -> Vec<DecisionRequirement> {
        vec![DecisionRequirement {
            decision_id: "support-08".into(),
            mandatory: false,
        }]
    }

    fn campaign_bundles() -> Vec<DeploymentBundle> {
        vec![DeploymentBundle {
            bundle_id: "corridor-west".into(),
            fixed_resources: ResourceVector::default(),
        }]
    }

    #[test]
    fn campaign_builder_retains_manifests_and_optimizes_only_admitted_options() {
        let built = build_evidence_bound_campaign(
            vec![candidate()],
            &[context()],
            EvidenceBoundCampaignConfig {
                campaign_id: "accepted-test-campaign".into(),
                decisions: campaign_decisions(),
                deployment_bundles: campaign_bundles(),
                budget: campaign_budget(),
                assurance: campaign_assurance(),
                maximum_search_nodes: 1_000,
            },
        )
        .unwrap();
        assert_eq!(built.candidate_count(), 1);
        let report = built.optimize().unwrap();
        assert_eq!(report.campaign.selection.selected_option_ids.len(), 1);
        assert_eq!(report.candidate_evidence.len(), 1);
        assert_eq!(report.selected_evidence.len(), 1);
        assert_eq!(
            report.selected_evidence[0].observation_receipt_digest,
            "sha256:observation"
        );
    }

    #[test]
    fn one_rejected_candidate_aborts_the_entire_campaign_build() {
        let mut stale = candidate();
        stale.manifest.observation_captured_at_unix_ms = NOW - 3_600_001;
        stale.manifest.candidate_id = "candidate-2026-015".into();
        stale.option.option_id = "support-08-stale-followup".into();
        stale.manifest.option_id = "support-08-stale-followup".into();
        let mut stale_context = context();
        stale_context.option_id = "support-08-stale-followup".into();
        let result = build_evidence_bound_campaign(
            vec![candidate(), stale],
            &[context(), stale_context],
            EvidenceBoundCampaignConfig {
                campaign_id: "campaign-with-one-stale-option".into(),
                decisions: campaign_decisions(),
                deployment_bundles: campaign_bundles(),
                budget: campaign_budget(),
                assurance: campaign_assurance(),
                maximum_search_nodes: 1_000,
            },
        );
        assert!(matches!(
            result,
            Err(CampaignAdmissionError::RejectedCandidates(rejections))
                if rejections.len() == 1
                    && rejections[0].reasons.contains(&RejectionReason::ObservationStale)
        ));
    }

    #[test]
    fn campaign_builder_rejects_duplicate_candidate_evidence_identity() {
        let mut second = candidate();
        second.option.option_id = "support-08-other-followup".into();
        second.manifest.option_id = "support-08-other-followup".into();
        let mut second_context = context();
        second_context.option_id = "support-08-other-followup".into();
        let result = build_evidence_bound_campaign(
            vec![candidate(), second],
            &[context(), second_context],
            EvidenceBoundCampaignConfig {
                campaign_id: "duplicate-evidence-identity".into(),
                decisions: campaign_decisions(),
                deployment_bundles: campaign_bundles(),
                budget: campaign_budget(),
                assurance: campaign_assurance(),
                maximum_search_nodes: 1_000,
            },
        );
        assert_eq!(
            result,
            Err(CampaignAdmissionError::DuplicateCandidateId(
                "candidate-2026-014".into()
            ))
        );
    }

    #[test]
    fn campaign_builder_rejects_missing_and_duplicate_contexts() {
        let missing = build_evidence_bound_campaign(
            vec![candidate()],
            &[],
            EvidenceBoundCampaignConfig {
                campaign_id: "missing-context-campaign".into(),
                decisions: campaign_decisions(),
                deployment_bundles: campaign_bundles(),
                budget: campaign_budget(),
                assurance: campaign_assurance(),
                maximum_search_nodes: 1_000,
            },
        );
        assert_eq!(
            missing,
            Err(CampaignAdmissionError::MissingContextForOption(
                "support-08-thermal-followup".into()
            ))
        );

        let duplicate = build_evidence_bound_campaign(
            vec![candidate()],
            &[context(), context()],
            EvidenceBoundCampaignConfig {
                campaign_id: "duplicate-context-campaign".into(),
                decisions: campaign_decisions(),
                deployment_bundles: campaign_bundles(),
                budget: campaign_budget(),
                assurance: campaign_assurance(),
                maximum_search_nodes: 1_000,
            },
        );
        assert_eq!(
            duplicate,
            Err(CampaignAdmissionError::DuplicateContextOptionId(
                "support-08-thermal-followup".into()
            ))
        );
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

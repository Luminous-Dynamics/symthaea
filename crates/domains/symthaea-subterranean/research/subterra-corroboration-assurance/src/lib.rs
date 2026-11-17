//! Fail-closed evidence corroboration under an externally supplied dependency model.
//!
//! This crate consumes the already admitted anchor option and a Mycelix-produced
//! transitive dependency closure. It does not authenticate receipts or construct
//! a second generic dependency graph.

use std::collections::{BTreeMap, BTreeSet};

use subterra_inspection_evidence_gate::{
    AdmittedInspectionOption, EvidencePurpose, RightsDisposition,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InventoryDisposition {
    CompleteWithinDeclaredScope,
    Incomplete,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonCauseDisposition {
    NoSharedClaimCriticalCauseWithinDeclaredScope,
    SharedClaimCriticalCauseDetected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifierSeparationDisposition {
    IndependentWithinDeclaredScope,
    SharedClaimCriticalCauseDetected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultSemanticsDisposition {
    FullyRepresentedByDeclaredModel,
    UnsupportedAlternativeOrRedundantPaths,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DependencyIdentity {
    /// Canonical, namespaced Mycelix fault-domain dimension.
    pub dimension: String,
    /// Exact identity supplied by the upstream transitive dependency closure.
    pub identity: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSource {
    pub source_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub source_dataset_digest: String,
    pub observation_receipt_digest: String,
    pub calibration_receipt_digest: String,
    pub rights_receipt_digest: String,
    pub rights_purpose: EvidencePurpose,
    pub rights_disposition: RightsDisposition,
    pub captured_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub calibration_valid_from_unix_ms: u64,
    pub calibration_valid_until_unix_ms: u64,
    pub rights_valid_until_unix_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDependencyClosure {
    pub source_id: String,
    pub inventory: InventoryDisposition,
    /// Transitive, claim-critical dependencies only; produced by Mycelix LEX-NET-030.
    pub claim_critical_dependencies: Vec<DependencyIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifierDependencyClosure {
    pub verifier_id: String,
    pub inventory: InventoryDisposition,
    /// Transitive, claim-critical dependencies only; produced by Mycelix LEX-NET-030.
    pub claim_critical_dependencies: Vec<DependencyIdentity>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortfolioAssuranceAssessment {
    pub claim_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    /// Externally computed digest of the exact source and observation set.
    pub evidence_set_digest: String,
    pub dependency_graph_digest: String,
    pub fault_domain_profile_digest: String,
    pub assessment_digest: String,
    pub verifier_id: String,
    pub verifier_receipt_digest: String,
    pub purpose: EvidencePurpose,
    pub evaluated_at_unix_ms: u64,
    pub valid_until_unix_ms: u64,
    pub inventory: InventoryDisposition,
    pub common_cause: CommonCauseDisposition,
    pub verifier_separation: VerifierSeparationDisposition,
    pub fault_semantics: FaultSemanticsDisposition,
    pub sources: Vec<SourceDependencyClosure>,
    pub verifier: VerifierDependencyClosure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedCorroborationContext {
    pub claim_id: String,
    pub option_id: String,
    pub asset_id: String,
    pub asset_revision: String,
    pub coordinate_frame_digest: String,
    pub region_scope_digest: String,
    pub evidence_set_digest: String,
    pub dependency_graph_digest: String,
    pub fault_domain_profile_digest: String,
    pub assessment_digest: String,
    pub verifier_id: String,
    pub verifier_receipt_digest: String,
    pub purpose: EvidencePurpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorroborationPolicy {
    pub minimum_distinct_sources: usize,
    pub evaluation_time_unix_ms: u64,
    pub maximum_observation_age_ms: u64,
    pub allowed_future_clock_skew_ms: u64,
    pub maximum_assessment_age_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateCorroborationPortfolio {
    pub admitted_anchor: AdmittedInspectionOption,
    pub sources: Vec<EvidenceSource>,
    pub assessment: PortfolioAssuranceAssessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectionReason {
    EmptyClaimId,
    EmptyOptionId,
    EmptyAssetId,
    EmptyVerifierId,
    EmptySourceId,
    EmptyExpectedDigest,
    EmptyDigest,
    InvalidPolicy,
    OptionIdMismatch,
    ClaimIdMismatch,
    VerifierIdentityMismatch,
    VerifierReusesAcquisitionIdentity,
    RightsPurposeMismatch,
    RightsDenied,
    RightsUnknown,
    MissingRightsReceipt,
    RightsExpired,
    CalibrationNotYetValid,
    CalibrationExpired,
    ObservationOutsideCalibrationValidity,
    InsufficientDistinctSources,
    DuplicateSourceIdentity,
    DuplicateObservationReceipt,
    DuplicateCalibrationReceipt,
    AnchorEvidenceSourceMissing,
    AssetMismatch,
    AssetRevisionMismatch,
    CoordinateFrameMismatch,
    RegionScopeMismatch,
    EvidenceSetMismatch,
    DependencyGraphMismatch,
    FaultDomainProfileMismatch,
    AssessmentDigestMismatch,
    VerifierReceiptMismatch,
    AssessmentFromFuture,
    AssessmentStale,
    AssessmentExpired,
    ObservationFromFuture,
    ObservationStale,
    ObservationExpired,
    SourceInventoryIncomplete,
    SourceInventoryUnknown,
    MissingSourceDependencyClosure,
    DuplicateSourceDependencyClosure,
    EmptySourceDependencyClosure,
    EmptyDependencyIdentity,
    SharedClaimCriticalDependencyBetweenSources,
    VerifierInventoryIncomplete,
    VerifierInventoryUnknown,
    EmptyVerifierDependencyClosure,
    VerifierOverlapsAcquisitionDependency,
    CommonCauseDetected,
    CommonCauseUnknown,
    VerifierNotIndependent,
    VerifierIndependenceUnknown,
    UnsupportedFaultSemantics,
    UnknownFaultSemantics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedCorroborationPortfolio {
    anchor: AdmittedInspectionOption,
    sources: Vec<EvidenceSource>,
    assessment: PortfolioAssuranceAssessment,
}

impl AdmittedCorroborationPortfolio {
    pub fn option_id(&self) -> &str {
        self.anchor.option_id()
    }

    pub fn sources(&self) -> &[EvidenceSource] {
        &self.sources
    }

    pub fn assessment(&self) -> &PortfolioAssuranceAssessment {
        &self.assessment
    }

    pub fn evidence_anchor(&self) -> &AdmittedInspectionOption {
        &self.anchor
    }

    pub fn independence_scope_description(&self) -> &'static str {
        "independent within the complete declared Mycelix dependency/fault model"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorroborationAdmissionReport {
    pub admitted: Option<AdmittedCorroborationPortfolio>,
    pub rejection_reasons: Vec<RejectionReason>,
    pub evidence_references: Vec<String>,
}

impl CorroborationAdmissionReport {
    pub fn accepted(&self) -> bool {
        self.admitted.is_some() && self.rejection_reasons.is_empty()
    }
}

fn missing(value: &str) -> bool {
    value.trim().is_empty()
}

fn add_if(
    reasons: &mut Vec<RejectionReason>,
    condition: bool,
    reason: RejectionReason,
) {
    if condition {
        reasons.push(reason);
    }
}

fn dependency_set(
    dependencies: &[DependencyIdentity],
    reasons: &mut Vec<RejectionReason>,
) -> BTreeSet<DependencyIdentity> {
    let mut result = BTreeSet::new();
    for dependency in dependencies {
        if missing(&dependency.dimension) || missing(&dependency.identity) {
            reasons.push(RejectionReason::EmptyDependencyIdentity);
        } else {
            result.insert(dependency.clone());
        }
    }
    result
}

pub fn admit_corroboration_portfolio(
    candidate: &CandidateCorroborationPortfolio,
    expected: &ExpectedCorroborationContext,
    policy: CorroborationPolicy,
) -> CorroborationAdmissionReport {
    let manifest = candidate.admitted_anchor.manifest();
    let assessment = &candidate.assessment;
    let mut reasons = Vec::new();

    add_if(&mut reasons, missing(&expected.claim_id), RejectionReason::EmptyClaimId);
    add_if(&mut reasons, missing(&expected.option_id), RejectionReason::EmptyOptionId);
    add_if(&mut reasons, missing(&expected.asset_id), RejectionReason::EmptyAssetId);
    add_if(&mut reasons, missing(&expected.verifier_id), RejectionReason::EmptyVerifierId);
    if [
        expected.asset_revision.as_str(),
        expected.coordinate_frame_digest.as_str(),
        expected.region_scope_digest.as_str(),
        expected.evidence_set_digest.as_str(),
        expected.dependency_graph_digest.as_str(),
        expected.fault_domain_profile_digest.as_str(),
        expected.assessment_digest.as_str(),
        expected.verifier_receipt_digest.as_str(),
    ]
    .iter()
    .any(|value| missing(value))
    {
        reasons.push(RejectionReason::EmptyExpectedDigest);
    }
    add_if(
        &mut reasons,
        policy.minimum_distinct_sources == 0,
        RejectionReason::InvalidPolicy,
    );
    add_if(&mut reasons, missing(&assessment.assessment_digest), RejectionReason::EmptyDigest);
    add_if(&mut reasons, missing(&assessment.verifier_receipt_digest), RejectionReason::EmptyDigest);

    add_if(
        &mut reasons,
        candidate.admitted_anchor.option_id() != expected.option_id,
        RejectionReason::OptionIdMismatch,
    );
    add_if(
        &mut reasons,
        manifest.asset_id != expected.asset_id || assessment.asset_id != expected.asset_id,
        RejectionReason::AssetMismatch,
    );
    add_if(
        &mut reasons,
        manifest.asset_revision != expected.asset_revision
            || assessment.asset_revision != expected.asset_revision,
        RejectionReason::AssetRevisionMismatch,
    );
    add_if(
        &mut reasons,
        manifest.coordinate_frame_digest != expected.coordinate_frame_digest
            || assessment.coordinate_frame_digest != expected.coordinate_frame_digest,
        RejectionReason::CoordinateFrameMismatch,
    );
    add_if(
        &mut reasons,
        manifest.region_scope_digest != expected.region_scope_digest
            || assessment.region_scope_digest != expected.region_scope_digest,
        RejectionReason::RegionScopeMismatch,
    );
    add_if(
        &mut reasons,
        assessment.claim_id != expected.claim_id,
        RejectionReason::ClaimIdMismatch,
    );
    add_if(
        &mut reasons,
        assessment.evidence_set_digest != expected.evidence_set_digest,
        RejectionReason::EvidenceSetMismatch,
    );
    add_if(
        &mut reasons,
        assessment.dependency_graph_digest != expected.dependency_graph_digest,
        RejectionReason::DependencyGraphMismatch,
    );
    add_if(
        &mut reasons,
        assessment.fault_domain_profile_digest != expected.fault_domain_profile_digest,
        RejectionReason::FaultDomainProfileMismatch,
    );
    add_if(
        &mut reasons,
        assessment.assessment_digest != expected.assessment_digest,
        RejectionReason::AssessmentDigestMismatch,
    );
    add_if(
        &mut reasons,
        assessment.verifier.verifier_id != expected.verifier_id
            || assessment.verifier_id != expected.verifier_id,
        RejectionReason::VerifierIdentityMismatch,
    );
    add_if(
        &mut reasons,
        assessment.purpose != expected.purpose,
        RejectionReason::RightsPurposeMismatch,
    );
    add_if(
        &mut reasons,
        manifest.rights_purpose != expected.purpose,
        RejectionReason::RightsPurposeMismatch,
    );
    add_if(
        &mut reasons,
        assessment.verifier_receipt_digest != expected.verifier_receipt_digest,
        RejectionReason::VerifierReceiptMismatch,
    );

    add_if(
        &mut reasons,
        candidate.sources.len() < policy.minimum_distinct_sources,
        RejectionReason::InsufficientDistinctSources,
    );

    let mut source_ids = BTreeSet::new();
    let mut observation_receipts = BTreeSet::new();
    let mut calibration_receipts = BTreeSet::new();
    let mut anchor_source_found = false;
    let latest_allowed = policy
        .evaluation_time_unix_ms
        .saturating_add(policy.allowed_future_clock_skew_ms);

    for source in &candidate.sources {
        add_if(&mut reasons, missing(&source.source_id), RejectionReason::EmptySourceId);
        add_if(
            &mut reasons,
            !source_ids.insert(source.source_id.as_str()),
            RejectionReason::DuplicateSourceIdentity,
        );
        add_if(
            &mut reasons,
            !observation_receipts.insert(source.observation_receipt_digest.as_str()),
            RejectionReason::DuplicateObservationReceipt,
        );
        add_if(
            &mut reasons,
            !calibration_receipts.insert(source.calibration_receipt_digest.as_str()),
            RejectionReason::DuplicateCalibrationReceipt,
        );
        add_if(
            &mut reasons,
            missing(&source.observation_receipt_digest)
                || missing(&source.calibration_receipt_digest)
                || missing(&source.rights_receipt_digest)
                || missing(&source.source_dataset_digest),
            RejectionReason::EmptyDigest,
        );
        add_if(
            &mut reasons,
            source.rights_purpose != expected.purpose,
            RejectionReason::RightsPurposeMismatch,
        );
        match source.rights_disposition {
            RightsDisposition::AuthorizedForDeclaredPurpose => {}
            RightsDisposition::Denied => reasons.push(RejectionReason::RightsDenied),
            RightsDisposition::Unknown => reasons.push(RejectionReason::RightsUnknown),
        }
        add_if(
            &mut reasons,
            source.rights_valid_until_unix_ms <= policy.evaluation_time_unix_ms,
            RejectionReason::RightsExpired,
        );

        add_if(
            &mut reasons,
            source.asset_id != expected.asset_id,
            RejectionReason::AssetMismatch,
        );
        add_if(
            &mut reasons,
            source.asset_revision != expected.asset_revision,
            RejectionReason::AssetRevisionMismatch,
        );
        add_if(
            &mut reasons,
            source.coordinate_frame_digest != expected.coordinate_frame_digest,
            RejectionReason::CoordinateFrameMismatch,
        );
        add_if(
            &mut reasons,
            source.region_scope_digest != expected.region_scope_digest,
            RejectionReason::RegionScopeMismatch,
        );

        if source.observation_receipt_digest == manifest.observation_receipt_digest
            && source.calibration_receipt_digest == manifest.calibration_receipt_digest
            && source.source_dataset_digest == manifest.source_dataset_digest
            && source.rights_receipt_digest == manifest.rights_receipt_digest
            && source.captured_at_unix_ms == manifest.observation_captured_at_unix_ms
            && source.valid_until_unix_ms == manifest.evidence_valid_until_unix_ms
            && source.calibration_valid_from_unix_ms == manifest.calibration_valid_from_unix_ms
            && source.calibration_valid_until_unix_ms == manifest.calibration_valid_until_unix_ms
            && source.rights_valid_until_unix_ms == manifest.rights_valid_until_unix_ms
        {
            anchor_source_found = true;
        }

        if source.captured_at_unix_ms > latest_allowed {
            reasons.push(RejectionReason::ObservationFromFuture);
        } else if policy
            .evaluation_time_unix_ms
            .saturating_sub(source.captured_at_unix_ms)
            > policy.maximum_observation_age_ms
        {
            reasons.push(RejectionReason::ObservationStale);
        }
        add_if(
            &mut reasons,
            source.valid_until_unix_ms <= policy.evaluation_time_unix_ms,
            RejectionReason::ObservationExpired,
        );
        if policy.evaluation_time_unix_ms < source.calibration_valid_from_unix_ms {
            reasons.push(RejectionReason::CalibrationNotYetValid);
        }
        if source.calibration_valid_until_unix_ms <= policy.evaluation_time_unix_ms {
            reasons.push(RejectionReason::CalibrationExpired);
        }
        if source.captured_at_unix_ms < source.calibration_valid_from_unix_ms
            || source.captured_at_unix_ms >= source.calibration_valid_until_unix_ms
        {
            reasons.push(RejectionReason::ObservationOutsideCalibrationValidity);
        }
    }

    add_if(
        &mut reasons,
        !anchor_source_found,
        RejectionReason::AnchorEvidenceSourceMissing,
    );

    add_if(
        &mut reasons,
        source_ids.contains(assessment.verifier.verifier_id.as_str())
            || source_ids.contains(assessment.verifier_id.as_str()),
        RejectionReason::VerifierReusesAcquisitionIdentity,
    );

    let mut closures_by_source = BTreeMap::new();
    for closure in &assessment.sources {
        if closures_by_source
            .insert(closure.source_id.as_str(), closure)
            .is_some()
        {
            reasons.push(RejectionReason::DuplicateSourceDependencyClosure);
        }
    }

    let mut dependency_sets = BTreeMap::<&str, BTreeSet<DependencyIdentity>>::new();
    for source_id in &source_ids {
        let Some(closure) = closures_by_source.get(source_id) else {
            reasons.push(RejectionReason::MissingSourceDependencyClosure);
            continue;
        };

        match closure.inventory {
            InventoryDisposition::CompleteWithinDeclaredScope => {}
            InventoryDisposition::Incomplete => {
                reasons.push(RejectionReason::SourceInventoryIncomplete);
            }
            InventoryDisposition::Unknown => {
                reasons.push(RejectionReason::SourceInventoryUnknown);
            }
        }
        if closure.claim_critical_dependencies.is_empty() {
            reasons.push(RejectionReason::EmptySourceDependencyClosure);
        }
        let set = dependency_set(&closure.claim_critical_dependencies, &mut reasons);
        dependency_sets.insert(source_id, set);
    }

    for source_id in closures_by_source.keys() {
        if !source_ids.contains(source_id) {
            reasons.push(RejectionReason::MissingSourceDependencyClosure);
        }
    }

    let source_keys = dependency_sets.keys().copied().collect::<Vec<_>>();
    for left_index in 0..source_keys.len() {
        for right_index in (left_index + 1)..source_keys.len() {
            let left = source_keys[left_index];
            let right = source_keys[right_index];
            let left_set = &dependency_sets[left];
            let right_set = &dependency_sets[right];
            if !left_set.is_disjoint(right_set) {
                reasons.push(RejectionReason::SharedClaimCriticalDependencyBetweenSources);
            }
        }
    }

    match assessment.inventory {
        InventoryDisposition::CompleteWithinDeclaredScope => {}
        InventoryDisposition::Incomplete => reasons.push(RejectionReason::SourceInventoryIncomplete),
        InventoryDisposition::Unknown => reasons.push(RejectionReason::SourceInventoryUnknown),
    }

    match assessment.verifier.inventory {
        InventoryDisposition::CompleteWithinDeclaredScope => {}
        InventoryDisposition::Incomplete => reasons.push(RejectionReason::VerifierInventoryIncomplete),
        InventoryDisposition::Unknown => reasons.push(RejectionReason::VerifierInventoryUnknown),
    }
    if assessment.verifier.claim_critical_dependencies.is_empty() {
        reasons.push(RejectionReason::EmptyVerifierDependencyClosure);
    }
    let verifier_dependencies =
        dependency_set(&assessment.verifier.claim_critical_dependencies, &mut reasons);
    for source_dependencies in dependency_sets.values() {
        if !source_dependencies.is_disjoint(&verifier_dependencies) {
            reasons.push(RejectionReason::VerifierOverlapsAcquisitionDependency);
            break;
        }
    }

    match assessment.common_cause {
        CommonCauseDisposition::NoSharedClaimCriticalCauseWithinDeclaredScope => {}
        CommonCauseDisposition::SharedClaimCriticalCauseDetected => {
            reasons.push(RejectionReason::CommonCauseDetected);
        }
        CommonCauseDisposition::Unknown => reasons.push(RejectionReason::CommonCauseUnknown),
    }
    match assessment.verifier_separation {
        VerifierSeparationDisposition::IndependentWithinDeclaredScope => {}
        VerifierSeparationDisposition::SharedClaimCriticalCauseDetected => {
            reasons.push(RejectionReason::VerifierNotIndependent);
        }
        VerifierSeparationDisposition::Unknown => {
            reasons.push(RejectionReason::VerifierIndependenceUnknown);
        }
    }
    match assessment.fault_semantics {
        FaultSemanticsDisposition::FullyRepresentedByDeclaredModel => {}
        FaultSemanticsDisposition::UnsupportedAlternativeOrRedundantPaths => {
            reasons.push(RejectionReason::UnsupportedFaultSemantics);
        }
        FaultSemanticsDisposition::Unknown => {
            reasons.push(RejectionReason::UnknownFaultSemantics);
        }
    }

    if assessment.evaluated_at_unix_ms > latest_allowed {
        reasons.push(RejectionReason::AssessmentFromFuture);
    } else if policy
        .evaluation_time_unix_ms
        .saturating_sub(assessment.evaluated_at_unix_ms)
        > policy.maximum_assessment_age_ms
    {
        reasons.push(RejectionReason::AssessmentStale);
    }
    add_if(
        &mut reasons,
        assessment.valid_until_unix_ms <= policy.evaluation_time_unix_ms,
        RejectionReason::AssessmentExpired,
    );

    reasons.sort();
    reasons.dedup();

    let admitted = reasons.is_empty();
    let mut references = vec![
        assessment.evidence_set_digest.clone(),
        assessment.dependency_graph_digest.clone(),
        assessment.fault_domain_profile_digest.clone(),
        assessment.assessment_digest.clone(),
        assessment.verifier_receipt_digest.clone(),
    ];
    for source in &candidate.sources {
        references.push(source.observation_receipt_digest.clone());
        references.push(source.calibration_receipt_digest.clone());
        references.push(source.rights_receipt_digest.clone());
        references.push(source.source_dataset_digest.clone());
    }
    references.sort();
    references.dedup();

    CorroborationAdmissionReport {
        admitted: admitted.then(|| AdmittedCorroborationPortfolio {
            anchor: candidate.admitted_anchor.clone(),
            sources: candidate.sources.clone(),
            assessment: assessment.clone(),
        }),
        rejection_reasons: reasons,
        evidence_references: references,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use subterra_inspection_evidence_gate::{
        CandidateInspectionOption, EvidenceManifest, EvidencePurpose, ExpectedInspectionContext,
        ReceiptVerificationDisposition, RightsDisposition, SpatialScopeDisposition,
        admit_candidate,
    };
    use subterra_inspection_campaign::ResourceVector;

    const NOW: u64 = 1_800_000_000_000;

    fn admitted_anchor() -> AdmittedInspectionOption {
        let expected = ExpectedInspectionContext {
            decision_id: "decision-7".into(),
            option_id: "inspect-7".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            sensor_model_digest: "sha256:sensor".into(),
            likelihood_model_digest: "sha256:likelihood".into(),
            loss_table_digest: "sha256:loss".into(),
            evsi_assessment_digest: "sha256:evsi".into(),
            source_dataset_digest: "sha256:dataset-a".into(),
            deployment_bundle_id: "corridor-1".into(),
            deployment_bundle_digest: "sha256:bundle".into(),
            resource_model_digest: "sha256:resource-model".into(),
            resource_assessment_digest: "sha256:resource".into(),
            risk_model_digest: "sha256:risk-model".into(),
            risk_assessment_digest: "sha256:risk".into(),
            purpose: EvidencePurpose::AssetConditionInspection,
            evaluation_time_unix_ms: NOW,
            maximum_observation_age_ms: 3_600_000,
            allowed_future_clock_skew_ms: 5_000,
        };
        let candidate = CandidateInspectionOption {
            option: subterra_inspection_campaign::InspectionOption {
                option_id: "inspect-7".into(),
                decision_id: "decision-7".into(),
                deployment_bundle_id: "corridor-1".into(),
                gross_evsi_micro_units: 100,
                unknown_state_debt_reduction_units: 1,
                variable_resources: ResourceVector {
                    energy_millijoules: 5,
                    duration_milliseconds: 10,
                    data_bytes: 20,
                    exposure_milliunits: 1,
                },
                mission_interruption_upper_ppm: 1,
                robot_loss_upper_ppm: 1,
                external_gate_passed: true,
            },
            manifest: EvidenceManifest {
                candidate_id: "candidate-7".into(),
                decision_id: "decision-7".into(),
                option_id: "inspect-7".into(),
                asset_id: "asset-7".into(),
                asset_revision: "rev-4".into(),
                coordinate_frame_digest: "sha256:frame".into(),
                region_scope_digest: "sha256:region".into(),
                sensor_model_digest: "sha256:sensor".into(),
                likelihood_model_digest: "sha256:likelihood".into(),
                loss_table_digest: "sha256:loss".into(),
                evsi_assessment_digest: "sha256:evsi".into(),
                source_dataset_digest: "sha256:dataset-a".into(),
                deployment_bundle_id: "corridor-1".into(),
                deployment_bundle_digest: "sha256:bundle".into(),
                resource_model_digest: "sha256:resource-model".into(),
                resource_assessment_digest: "sha256:resource".into(),
                risk_model_digest: "sha256:risk-model".into(),
                risk_assessment_digest: "sha256:risk".into(),
                assessed_resources: ResourceVector {
                    energy_millijoules: 5,
                    duration_milliseconds: 10,
                    data_bytes: 20,
                    exposure_milliunits: 1,
                },
                assessed_mission_interruption_upper_ppm: 1,
                assessed_robot_loss_upper_ppm: 1,
                resource_assessment_valid_until_unix_ms: NOW + 10_000,
                risk_assessment_valid_until_unix_ms: NOW + 10_000,
                observation_receipt_digest: "sha256:obs-a".into(),
                calibration_receipt_digest: "sha256:cal-a".into(),
                rights_receipt_digest: "sha256:rights".into(),
                verifier_receipt_digest: "sha256:upstream-verifier".into(),
                observation_captured_at_unix_ms: NOW - 1_000,
                evidence_valid_until_unix_ms: NOW + 10_000,
                calibration_valid_from_unix_ms: NOW - 10_000,
                calibration_valid_until_unix_ms: NOW + 10_000,
                rights_valid_until_unix_ms: NOW + 10_000,
                rights_purpose: EvidencePurpose::AssetConditionInspection,
                rights_disposition: RightsDisposition::AuthorizedForDeclaredPurpose,
                receipt_verification: ReceiptVerificationDisposition::VerifiedByExternalService,
                spatial_scope: SpatialScopeDisposition::CompleteWithinDeclaredScope,
            },
        };
        admit_candidate(&candidate, &expected)
            .admitted_candidate
            .expect("synthetic upstream candidate must be admitted")
    }

    fn source(id: &str, observation: &str, calibration: &str) -> EvidenceSource {
        EvidenceSource {
            source_id: id.into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            source_dataset_digest: format!("sha256:dataset-{id}"),
            observation_receipt_digest: observation.into(),
            calibration_receipt_digest: calibration.into(),
            rights_receipt_digest: format!("sha256:rights-{id}"),
            rights_purpose: EvidencePurpose::AssetConditionInspection,
            rights_disposition: RightsDisposition::AuthorizedForDeclaredPurpose,
            captured_at_unix_ms: NOW - 1_000,
            valid_until_unix_ms: NOW + 10_000,
            calibration_valid_from_unix_ms: NOW - 10_000,
            calibration_valid_until_unix_ms: NOW + 10_000,
            rights_valid_until_unix_ms: NOW + 10_000,
        }
    }

    fn dep(dimension: &str, id: &str) -> DependencyIdentity {
        DependencyIdentity {
            dimension: dimension.into(),
            identity: id.into(),
        }
    }

    fn assessment(sources: &[EvidenceSource]) -> PortfolioAssuranceAssessment {
        let closures = sources
            .iter()
            .map(|source| SourceDependencyClosure {
                source_id: source.source_id.clone(),
                inventory: InventoryDisposition::CompleteWithinDeclaredScope,
                claim_critical_dependencies: vec![
                    dep("robot", &format!("robot-{}", source.source_id)),
                    dep("power", &format!("power-{}", source.source_id)),
                    dep("clock", &format!("clock-{}", source.source_id)),
                    dep("compute", &format!("compute-{}", source.source_id)),
                    dep("network", &format!("network-{}", source.source_id)),
                ],
            })
            .collect();
        PortfolioAssuranceAssessment {
            claim_id: "claim-7".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            evidence_set_digest: "sha256:evidence-set".into(),
            dependency_graph_digest: "sha256:graph".into(),
            fault_domain_profile_digest: "sha256:profile".into(),
            assessment_digest: "sha256:assessment".into(),
            verifier_id: "verifier-7".into(),
            verifier_receipt_digest: "sha256:portfolio-verifier".into(),
            purpose: EvidencePurpose::AssetConditionInspection,
            evaluated_at_unix_ms: NOW - 100,
            valid_until_unix_ms: NOW + 10_000,
            inventory: InventoryDisposition::CompleteWithinDeclaredScope,
            common_cause: CommonCauseDisposition::NoSharedClaimCriticalCauseWithinDeclaredScope,
            verifier_separation: VerifierSeparationDisposition::IndependentWithinDeclaredScope,
            fault_semantics: FaultSemanticsDisposition::FullyRepresentedByDeclaredModel,
            sources: closures,
            verifier: VerifierDependencyClosure {
                verifier_id: "verifier-7".into(),
                inventory: InventoryDisposition::CompleteWithinDeclaredScope,
                claim_critical_dependencies: vec![
                    dep("verifier-host", "verify-host"),
                    dep("verifier-clock", "verify-clock"),
                    dep("verifier-dataset", "sealed-ground-truth"),
                ],
            },
        }
    }

    fn expected() -> ExpectedCorroborationContext {
        ExpectedCorroborationContext {
            claim_id: "claim-7".into(),
            option_id: "inspect-7".into(),
            asset_id: "asset-7".into(),
            asset_revision: "rev-4".into(),
            coordinate_frame_digest: "sha256:frame".into(),
            region_scope_digest: "sha256:region".into(),
            evidence_set_digest: "sha256:evidence-set".into(),
            dependency_graph_digest: "sha256:graph".into(),
            fault_domain_profile_digest: "sha256:profile".into(),
            assessment_digest: "sha256:assessment".into(),
            verifier_id: "verifier-7".into(),
            verifier_receipt_digest: "sha256:portfolio-verifier".into(),
            purpose: EvidencePurpose::AssetConditionInspection,
        }
    }

    fn policy() -> CorroborationPolicy {
        CorroborationPolicy {
            minimum_distinct_sources: 2,
            evaluation_time_unix_ms: NOW,
            maximum_observation_age_ms: 3_600_000,
            allowed_future_clock_skew_ms: 5_000,
            maximum_assessment_age_ms: 3_600_000,
        }
    }

    fn candidate() -> CandidateCorroborationPortfolio {
        let mut anchor_source = source("robot-a-thermal", "sha256:obs-a", "sha256:cal-a");
        anchor_source.source_dataset_digest = "sha256:dataset-a".into();
        anchor_source.rights_receipt_digest = "sha256:rights".into();
        let sources = vec![
            anchor_source,
            source("robot-b-lidar", "sha256:obs-b", "sha256:cal-b"),
        ];
        CandidateCorroborationPortfolio {
            admitted_anchor: admitted_anchor(),
            assessment: assessment(&sources),
            sources,
        }
    }

    #[test]
    fn disjoint_complete_source_and_verifier_closures_are_admitted() {
        let report = admit_corroboration_portfolio(&candidate(), &expected(), policy());
        assert!(report.accepted(), "{:?}", report.rejection_reasons);
        let admitted = report.admitted.unwrap();
        assert_eq!(admitted.sources().len(), 2);
        assert_eq!(
            admitted.independence_scope_description(),
            "independent within the complete declared Mycelix dependency/fault model"
        );
    }

    #[test]
    fn shared_power_dependency_rejects_two_modalities_as_independent() {
        let mut item = candidate();
        item.assessment.sources[1].claim_critical_dependencies[1] =
            dep("power", "shared-robot-battery-bus");
        item.assessment.sources[0].claim_critical_dependencies[1] =
            dep("power", "shared-robot-battery-bus");
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(
            &RejectionReason::SharedClaimCriticalDependencyBetweenSources
        ));
        assert!(!report.accepted());
    }

    #[test]
    fn verifier_sharing_a_clock_with_acquisition_is_rejected() {
        let mut item = candidate();
        item.assessment.verifier.claim_critical_dependencies =
            vec![dep("clock", "clock-robot-a-thermal")];
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(
            &RejectionReason::VerifierOverlapsAcquisitionDependency
        ));
    }

    #[test]
    fn unknown_or_incomplete_source_inventory_is_never_admitted() {
        let mut unknown = candidate();
        unknown.assessment.sources[0].inventory = InventoryDisposition::Unknown;
        let report = admit_corroboration_portfolio(&unknown, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::SourceInventoryUnknown));

        let mut incomplete = candidate();
        incomplete.assessment.inventory = InventoryDisposition::Incomplete;
        let report = admit_corroboration_portfolio(&incomplete, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::SourceInventoryIncomplete));
    }

    #[test]
    fn duplicate_observation_cannot_be_counted_twice() {
        let mut item = candidate();
        item.sources[1].observation_receipt_digest =
            item.sources[0].observation_receipt_digest.clone();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::DuplicateObservationReceipt));
    }

    #[test]
    fn each_source_rights_and_calibration_validity_must_hold() {
        let mut rights = candidate();
        rights.sources[1].rights_disposition = RightsDisposition::Denied;
        let report = admit_corroboration_portfolio(&rights, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::RightsDenied));

        let mut expired_rights = candidate();
        expired_rights.sources[1].rights_valid_until_unix_ms = NOW;
        let report = admit_corroboration_portfolio(&expired_rights, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::RightsExpired));

        let mut expired_calibration = candidate();
        expired_calibration.sources[1].calibration_valid_until_unix_ms = NOW;
        let report = admit_corroboration_portfolio(&expired_calibration, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::CalibrationExpired));
    }

    #[test]
    fn missing_source_rights_receipt_rejects() {
        let mut item = candidate();
        item.sources[1].rights_receipt_digest.clear();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::EmptyDigest));
        assert!(!report.accepted());
    }

    #[test]
    fn unknown_common_cause_or_verifier_separation_rejects() {
        let mut common_cause = candidate();
        common_cause.assessment.common_cause = CommonCauseDisposition::Unknown;
        let report = admit_corroboration_portfolio(&common_cause, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::CommonCauseUnknown));

        let mut verifier = candidate();
        verifier.assessment.verifier_separation = VerifierSeparationDisposition::Unknown;
        let report = admit_corroboration_portfolio(&verifier, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::VerifierIndependenceUnknown));
    }

    #[test]
    fn expected_verifier_identity_and_purpose_are_pinned() {
        let mut item = candidate();
        item.assessment.verifier_id = "different-verifier".into();
        item.assessment.verifier.verifier_id = "different-verifier".into();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::VerifierIdentityMismatch));

        let mut wrong_purpose = candidate();
        wrong_purpose.assessment.purpose = EvidencePurpose::ResearchEvaluation;
        let report = admit_corroboration_portfolio(&wrong_purpose, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::RightsPurposeMismatch));
    }

    #[test]
    fn stale_or_future_source_evidence_is_rejected() {
        let mut stale = candidate();
        stale.sources[1].captured_at_unix_ms = NOW - 3_600_001;
        let report = admit_corroboration_portfolio(&stale, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::ObservationStale));

        let mut future = candidate();
        future.sources[1].captured_at_unix_ms = NOW + 5_001;
        let report = admit_corroboration_portfolio(&future, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::ObservationFromFuture));
    }

    #[test]
    fn wrong_frame_or_evidence_set_is_rejected() {
        let mut wrong_frame = candidate();
        wrong_frame.sources[1].coordinate_frame_digest = "sha256:other-frame".into();
        let report = admit_corroboration_portfolio(&wrong_frame, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::CoordinateFrameMismatch));

        let mut wrong_set = candidate();
        wrong_set.assessment.evidence_set_digest = "sha256:other-set".into();
        let report = admit_corroboration_portfolio(&wrong_set, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::EvidenceSetMismatch));
    }

    #[test]
    fn missing_source_closure_and_unsupported_fault_semantics_reject() {
        let mut missing = candidate();
        missing.assessment.sources.pop();
        let report = admit_corroboration_portfolio(&missing, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::MissingSourceDependencyClosure));

        let mut unsupported = candidate();
        unsupported.assessment.fault_semantics =
            FaultSemanticsDisposition::UnsupportedAlternativeOrRedundantPaths;
        let report = admit_corroboration_portfolio(&unsupported, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::UnsupportedFaultSemantics));
    }

    #[test]
    fn order_and_duplicate_dependency_edges_do_not_change_overlap_result() {
        let mut first = candidate();
        let baseline = admit_corroboration_portfolio(&first, &expected(), policy());
        first.sources.reverse();
        first.assessment.sources.reverse();
        first.assessment.sources[0]
            .claim_critical_dependencies
            .push(first.assessment.sources[0].claim_critical_dependencies[0].clone());
        let reordered = admit_corroboration_portfolio(&first, &expected(), policy());
        assert_eq!(baseline.rejection_reasons, reordered.rejection_reasons);
        assert_eq!(baseline.accepted(), reordered.accepted());
    }

    #[test]
    fn assessment_graph_profile_and_verifier_receipt_are_exactly_pinned() {
        let mut item = candidate();
        item.assessment.dependency_graph_digest = "sha256:other-graph".into();
        item.assessment.verifier_receipt_digest = "sha256:other-verifier".into();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::DependencyGraphMismatch));
        assert!(report.rejection_reasons.contains(&RejectionReason::VerifierReceiptMismatch));
    }

    #[test]
    fn verifier_cannot_reuse_an_acquisition_source_identity() {
        let mut item = candidate();
        item.assessment.verifier_id = "robot-a-thermal".into();
        item.assessment.verifier.verifier_id = "robot-a-thermal".into();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(
            &RejectionReason::VerifierReusesAcquisitionIdentity
        ));
    }

    #[test]
    fn fewer_than_policy_required_sources_is_rejected() {
        let mut item = candidate();
        item.sources.pop();
        item.assessment.sources.pop();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::InsufficientDistinctSources));
        assert!(!report.accepted());
    }

    #[test]
    fn upstream_anchor_observation_must_be_in_corroboration_portfolio() {
        let mut item = candidate();
        item.sources[0].observation_receipt_digest = "sha256:unrelated".into();
        let report = admit_corroboration_portfolio(&item, &expected(), policy());
        assert!(report.rejection_reasons.contains(&RejectionReason::AnchorEvidenceSourceMissing));
    }
}

//! SAWEM execution-authority and bid/offer submission gate.
//!
//! Market analysis, forecasting, optimisation and preparation are distinct from
//! transmission of a bid/offer to the Market Operator. External submission is
//! blocked unless the applicable final rule instrument is effective and known,
//! the participant is commercially admitted/accredited, API permission exists,
//! and the rule's execution-control mode is satisfied.
//!
//! Draft consultation material is never treated as permission. Internal
//! simulation is a separate non-commercial route and cannot produce a
//! commercial-submission authorisation. This is a policy evaluator only: it
//! does not connect to an exchange, sign orders or send bids/offers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketEnvironment {
    InternalSimulation,
    ExternalCommercial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingWorkflowAction {
    Analyse,
    Forecast,
    Optimise,
    PrepareBidOrOffer,
    SubmitBidOrOffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegulatoryInstrumentState {
    Draft,
    InForce,
    Superseded,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmissionControlMode {
    HumanReviewRequired,
    AutonomousSubmissionExplicitlyPermitted,
    SubmissionProhibited,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    Verified,
    Rejected,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccreditationStage {
    DesignApproval,
    FactoryAcceptanceTesting,
    IntegrationTesting,
    MarketSimulation,
    PilotOperation,
    CommercialApproval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateEvidenceRef<'a> {
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub available_at_utc: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegulatoryExecutionProfile<'a> {
    pub instrument_id: &'a str,
    pub instrument_state: RegulatoryInstrumentState,
    pub control_mode: SubmissionControlMode,
    pub effective_from_utc: Option<i64>,
    pub effective_until_utc: Option<i64>,
    pub published_at_utc: Option<i64>,
    pub evidence: Option<GateEvidenceRef<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParticipantExecutionStatus<'a> {
    pub participant_id: &'a str,
    pub admission_status: EvidenceStatus,
    pub admission_evidence: Option<GateEvidenceRef<'a>>,
    /// All prior stages must be Verified before CommercialApproval can qualify.
    pub accreditation_stages: &'a [(AccreditationStage, EvidenceStatus, GateEvidenceRef<'a>)],
    pub api_submission_status: EvidenceStatus,
    pub api_permission_evidence: Option<GateEvidenceRef<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanBidApproval<'a> {
    pub approval_id: &'a str,
    pub participant_id: &'a str,
    pub bid_manifest_sha256: &'a str,
    pub reviewer_id: &'a str,
    pub reviewer_role: &'a str,
    pub reviewer_authority: EvidenceStatus,
    pub reviewer_authority_evidence: Option<GateEvidenceRef<'a>>,
    pub reviewed_at_utc: i64,
    pub approval_evidence: GateEvidenceRef<'a>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionAuthorityInputs<'a> {
    pub environment: MarketEnvironment,
    pub action: TradingWorkflowAction,
    pub evaluation_time_utc: i64,
    pub participant: ParticipantExecutionStatus<'a>,
    pub regulatory_profile: RegulatoryExecutionProfile<'a>,
    /// Digest of the exact bid/offer payload including participant, product,
    /// Trading Period, direction, price, quantity and relevant declarations.
    pub bid_manifest_sha256: &'a str,
    /// When the sender is an automated principal, the human reviewer must be
    /// a different identity. None means no automated principal is declared.
    pub automation_principal_id: Option<&'a str>,
    pub human_approval: Option<HumanBidApproval<'a>>,
    /// Separate verified internal simulation access, not external admission.
    pub internal_simulation_access: EvidenceStatus,
    pub internal_simulation_evidence: Option<GateEvidenceRef<'a>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAuthorityDisposition {
    AnalysisPermitted,
    PreparationPermitted,
    SimulationOnlyPermitted,
    CommercialSubmissionPermittedWithHumanReview,
    CommercialSubmissionPermittedByExplicitRule,
    Blocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAuthorityReason {
    NonSubmissionAction,
    SimulationAccessVerified,
    FinalRuleNotInForce,
    RuleEvidenceMissingOrFutureKnown,
    RuleEffectiveWindowNotApplicable,
    SubmissionProhibited,
    ExecutionControlUnknown,
    ParticipantNotAdmitted,
    CommercialAccreditationIncomplete,
    ApiPermissionNotVerified,
    InvalidBidManifest,
    HumanApprovalMissing,
    HumanApprovalDoesNotMatchBid,
    HumanReviewerNotAuthorised,
    HumanApprovalAfterEvaluationTime,
    HumanApprovalEvidenceInvalid,
    HumanReviewSatisfied,
    SubmissionAuthorityVerified,
    InternalSimulationAccessUnverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionAuthorityDecision<'a> {
    pub disposition: ExecutionAuthorityDisposition,
    pub reason: ExecutionAuthorityReason,
    pub participant_id: &'a str,
    pub bid_manifest_sha256: &'a str,
    pub regulatory_instrument_id: &'a str,
    pub human_approval_id: Option<&'a str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionAuthorityError {
    InvalidReference,
    InvalidDigest,
    InvalidEvidenceTimestamp,
    DuplicateAccreditationStage,
    MissingAccreditationStage,
    InvalidRegulatoryWindow,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn validate_evidence(
    evidence: GateEvidenceRef<'_>,
    evaluation_time_utc: i64,
) -> Result<(), ExecutionAuthorityError> {
    if !nonempty(evidence.source_ref) {
        return Err(ExecutionAuthorityError::InvalidReference);
    }
    if !valid_sha256(evidence.source_sha256) {
        return Err(ExecutionAuthorityError::InvalidDigest);
    }
    if evidence.available_at_utc > evaluation_time_utc {
        return Err(ExecutionAuthorityError::InvalidEvidenceTimestamp);
    }
    Ok(())
}

fn blocked<'a>(
    inputs: ExecutionAuthorityInputs<'a>,
    reason: ExecutionAuthorityReason,
) -> ExecutionAuthorityDecision<'a> {
    ExecutionAuthorityDecision {
        disposition: ExecutionAuthorityDisposition::Blocked,
        reason,
        participant_id: inputs.participant.participant_id,
        bid_manifest_sha256: inputs.bid_manifest_sha256,
        regulatory_instrument_id: inputs.regulatory_profile.instrument_id,
        human_approval_id: None,
    }
}

fn validate_accreditation(
    participant: ParticipantExecutionStatus<'_>,
    evaluation_time_utc: i64,
) -> Result<(), ExecutionAuthorityError> {
    let required = [
        AccreditationStage::DesignApproval,
        AccreditationStage::FactoryAcceptanceTesting,
        AccreditationStage::IntegrationTesting,
        AccreditationStage::MarketSimulation,
        AccreditationStage::PilotOperation,
        AccreditationStage::CommercialApproval,
    ];
    for (index, (stage, _, evidence)) in participant.accreditation_stages.iter().enumerate() {
        if participant.accreditation_stages[..index]
            .iter()
            .any(|(prior, _, _)| prior == stage)
        {
            return Err(ExecutionAuthorityError::DuplicateAccreditationStage);
        }
        validate_evidence(*evidence, evaluation_time_utc)?;
    }
    if required.iter().any(|stage| {
        !participant
            .accreditation_stages
            .iter()
            .any(|(present, _, _)| present == stage)
    }) {
        return Err(ExecutionAuthorityError::MissingAccreditationStage);
    }
    Ok(())
}

/// Evaluate whether the workflow may progress to analysis, preparation,
/// simulation-only submission, or external commercial submission.
pub fn evaluate_execution_authority<'a>(
    inputs: ExecutionAuthorityInputs<'a>,
) -> Result<ExecutionAuthorityDecision<'a>, ExecutionAuthorityError> {
    if !nonempty(inputs.participant.participant_id)
        || !nonempty(inputs.regulatory_profile.instrument_id)
    {
        return Err(ExecutionAuthorityError::InvalidReference);
    }
    if !valid_sha256(inputs.bid_manifest_sha256) {
        return Err(ExecutionAuthorityError::InvalidDigest);
    }

    match inputs.action {
        TradingWorkflowAction::Analyse
        | TradingWorkflowAction::Forecast
        | TradingWorkflowAction::Optimise => {
            return Ok(ExecutionAuthorityDecision {
                disposition: ExecutionAuthorityDisposition::AnalysisPermitted,
                reason: ExecutionAuthorityReason::NonSubmissionAction,
                participant_id: inputs.participant.participant_id,
                bid_manifest_sha256: inputs.bid_manifest_sha256,
                regulatory_instrument_id: inputs.regulatory_profile.instrument_id,
                human_approval_id: None,
            });
        }
        TradingWorkflowAction::PrepareBidOrOffer => {
            return Ok(ExecutionAuthorityDecision {
                disposition: ExecutionAuthorityDisposition::PreparationPermitted,
                reason: ExecutionAuthorityReason::NonSubmissionAction,
                participant_id: inputs.participant.participant_id,
                bid_manifest_sha256: inputs.bid_manifest_sha256,
                regulatory_instrument_id: inputs.regulatory_profile.instrument_id,
                human_approval_id: None,
            });
        }
        TradingWorkflowAction::SubmitBidOrOffer => {}
    }

    if inputs.environment == MarketEnvironment::InternalSimulation {
        if inputs.internal_simulation_access != EvidenceStatus::Verified {
            return Ok(blocked(
                inputs,
                ExecutionAuthorityReason::InternalSimulationAccessUnverified,
            ));
        }
        let Some(evidence) = inputs.internal_simulation_evidence else {
            return Ok(blocked(
                inputs,
                ExecutionAuthorityReason::InternalSimulationAccessUnverified,
            ));
        };
        validate_evidence(evidence, inputs.evaluation_time_utc)?;
        return Ok(ExecutionAuthorityDecision {
            disposition: ExecutionAuthorityDisposition::SimulationOnlyPermitted,
            reason: ExecutionAuthorityReason::SimulationAccessVerified,
            participant_id: inputs.participant.participant_id,
            bid_manifest_sha256: inputs.bid_manifest_sha256,
            regulatory_instrument_id: inputs.regulatory_profile.instrument_id,
            human_approval_id: None,
        });
    }

    let profile = inputs.regulatory_profile;
    match profile.instrument_state {
        RegulatoryInstrumentState::Draft
        | RegulatoryInstrumentState::Superseded
        | RegulatoryInstrumentState::Unknown => {
            return Ok(blocked(inputs, ExecutionAuthorityReason::FinalRuleNotInForce));
        }
        RegulatoryInstrumentState::InForce => {}
    }
    let Some(rule_evidence) = profile.evidence else {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::RuleEvidenceMissingOrFutureKnown,
        ));
    };
    validate_evidence(rule_evidence, inputs.evaluation_time_utc)?;
    let Some(published_at) = profile.published_at_utc else {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::RuleEvidenceMissingOrFutureKnown,
        ));
    };
    if published_at > inputs.evaluation_time_utc || published_at > rule_evidence.available_at_utc {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::RuleEvidenceMissingOrFutureKnown,
        ));
    }
    let Some(effective_from) = profile.effective_from_utc else {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::RuleEffectiveWindowNotApplicable,
        ));
    };
    if effective_from > inputs.evaluation_time_utc
        || profile
            .effective_until_utc
            .map(|end| inputs.evaluation_time_utc > end)
            .unwrap_or(false)
    {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::RuleEffectiveWindowNotApplicable,
        ));
    }
    if profile
        .effective_until_utc
        .is_some_and(|end| end < effective_from)
    {
        return Err(ExecutionAuthorityError::InvalidRegulatoryWindow);
    }

    match profile.control_mode {
        SubmissionControlMode::SubmissionProhibited => {
            return Ok(blocked(inputs, ExecutionAuthorityReason::SubmissionProhibited));
        }
        SubmissionControlMode::Unknown => {
            return Ok(blocked(inputs, ExecutionAuthorityReason::ExecutionControlUnknown));
        }
        SubmissionControlMode::HumanReviewRequired
        | SubmissionControlMode::AutonomousSubmissionExplicitlyPermitted => {}
    }

    if inputs.participant.admission_status != EvidenceStatus::Verified {
        return Ok(blocked(inputs, ExecutionAuthorityReason::ParticipantNotAdmitted));
    }
    let Some(admission_evidence) = inputs.participant.admission_evidence else {
        return Ok(blocked(inputs, ExecutionAuthorityReason::ParticipantNotAdmitted));
    };
    validate_evidence(admission_evidence, inputs.evaluation_time_utc)?;

    validate_accreditation(inputs.participant, inputs.evaluation_time_utc)?;
    if inputs
        .participant
        .accreditation_stages
        .iter()
        .any(|(_, status, _)| *status != EvidenceStatus::Verified)
    {
        return Ok(blocked(
            inputs,
            ExecutionAuthorityReason::CommercialAccreditationIncomplete,
        ));
    }

    if inputs.participant.api_submission_status != EvidenceStatus::Verified {
        return Ok(blocked(inputs, ExecutionAuthorityReason::ApiPermissionNotVerified));
    }
    let Some(api_evidence) = inputs.participant.api_permission_evidence else {
        return Ok(blocked(inputs, ExecutionAuthorityReason::ApiPermissionNotVerified));
    };
    validate_evidence(api_evidence, inputs.evaluation_time_utc)?;

    match profile.control_mode {
        SubmissionControlMode::HumanReviewRequired => {
            let Some(approval) = inputs.human_approval else {
                return Ok(blocked(inputs, ExecutionAuthorityReason::HumanApprovalMissing));
            };
            if !nonempty(approval.approval_id)
                || !nonempty(approval.reviewer_id)
                || !nonempty(approval.reviewer_role)
                || !valid_sha256(approval.bid_manifest_sha256)
            {
                return Err(ExecutionAuthorityError::InvalidReference);
            }
            if approval.participant_id != inputs.participant.participant_id
                || approval.bid_manifest_sha256 != inputs.bid_manifest_sha256
                || inputs.automation_principal_id == Some(approval.reviewer_id)
            {
                return Ok(blocked(
                    inputs,
                    ExecutionAuthorityReason::HumanApprovalDoesNotMatchBid,
                ));
            }
            if approval.reviewer_authority != EvidenceStatus::Verified {
                return Ok(blocked(
                    inputs,
                    ExecutionAuthorityReason::HumanReviewerNotAuthorised,
                ));
            }
            let Some(authority_evidence) = approval.reviewer_authority_evidence else {
                return Ok(blocked(
                    inputs,
                    ExecutionAuthorityReason::HumanReviewerNotAuthorised,
                ));
            };
            validate_evidence(authority_evidence, inputs.evaluation_time_utc)?;
            if approval.reviewed_at_utc > inputs.evaluation_time_utc {
                return Ok(blocked(
                    inputs,
                    ExecutionAuthorityReason::HumanApprovalAfterEvaluationTime,
                ));
            }
            validate_evidence(approval.approval_evidence, inputs.evaluation_time_utc)?;
            if approval.approval_evidence.available_at_utc < approval.reviewed_at_utc {
                return Err(ExecutionAuthorityError::InvalidEvidenceTimestamp);
            }
            Ok(ExecutionAuthorityDecision {
                disposition:
                    ExecutionAuthorityDisposition::CommercialSubmissionPermittedWithHumanReview,
                reason: ExecutionAuthorityReason::HumanReviewSatisfied,
                participant_id: inputs.participant.participant_id,
                bid_manifest_sha256: inputs.bid_manifest_sha256,
                regulatory_instrument_id: profile.instrument_id,
                human_approval_id: Some(approval.approval_id),
            })
        }
        SubmissionControlMode::AutonomousSubmissionExplicitlyPermitted => {
            Ok(ExecutionAuthorityDecision {
                disposition:
                    ExecutionAuthorityDisposition::CommercialSubmissionPermittedByExplicitRule,
                reason: ExecutionAuthorityReason::SubmissionAuthorityVerified,
                participant_id: inputs.participant.participant_id,
                bid_manifest_sha256: inputs.bid_manifest_sha256,
                regulatory_instrument_id: profile.instrument_id,
                human_approval_id: None,
            })
        }
        SubmissionControlMode::SubmissionProhibited | SubmissionControlMode::Unknown => {
            Ok(blocked(inputs, ExecutionAuthorityReason::ExecutionControlUnknown))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const SHA_B: &str = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";

    fn evidence(source: &'static str, digest: &'static str) -> GateEvidenceRef<'static> {
        GateEvidenceRef {
            source_ref: source,
            source_sha256: digest,
            available_at_utc: 10,
        }
    }

    fn accreditation() -> Vec<(AccreditationStage, EvidenceStatus, GateEvidenceRef<'static>)> {
        vec![
            (AccreditationStage::DesignApproval, EvidenceStatus::Verified, evidence("stage/design", SHA_A)),
            (AccreditationStage::FactoryAcceptanceTesting, EvidenceStatus::Verified, evidence("stage/fat", SHA_A)),
            (AccreditationStage::IntegrationTesting, EvidenceStatus::Verified, evidence("stage/integration", SHA_A)),
            (AccreditationStage::MarketSimulation, EvidenceStatus::Verified, evidence("stage/simulation", SHA_A)),
            (AccreditationStage::PilotOperation, EvidenceStatus::Verified, evidence("stage/pilot", SHA_A)),
            (AccreditationStage::CommercialApproval, EvidenceStatus::Verified, evidence("stage/commercial", SHA_A)),
        ]
    }

    fn base_inputs<'a>(
        environment: MarketEnvironment,
        action: TradingWorkflowAction,
        control: SubmissionControlMode,
    ) -> ExecutionAuthorityInputs<'a> {
        let _ = control;
        ExecutionAuthorityInputs {
            environment,
            action,
            evaluation_time_utc: 100,
            participant: ParticipantExecutionStatus {
                participant_id: "participant/001",
                admission_status: EvidenceStatus::Verified,
                admission_evidence: Some(evidence("admission/001", SHA_A)),
                accreditation_stages: &[],
                api_submission_status: EvidenceStatus::Verified,
                api_permission_evidence: Some(evidence("api/permission", SHA_A)),
            },
            regulatory_profile: RegulatoryExecutionProfile {
                instrument_id: "SAWEM-EXECUTION-PROFILE/REV-A",
                instrument_state: RegulatoryInstrumentState::InForce,
                control_mode: control,
                effective_from_utc: Some(50),
                effective_until_utc: None,
                published_at_utc: Some(40),
                evidence: Some(evidence("rule/final", SHA_B)),
            },
            bid_manifest_sha256: SHA_A,
            automation_principal_id: Some("agent/sawem"),
            human_approval: None,
            internal_simulation_access: EvidenceStatus::Verified,
            internal_simulation_evidence: Some(evidence("internal/simulation-access", SHA_A)),
        }
    }

    #[test]
    fn analysis_forecast_optimisation_and_preparation_do_not_require_submission_authority() {
        for action in [
            TradingWorkflowAction::Analyse,
            TradingWorkflowAction::Forecast,
            TradingWorkflowAction::Optimise,
            TradingWorkflowAction::PrepareBidOrOffer,
        ] {
            let result = evaluate_execution_authority(base_inputs(
                MarketEnvironment::ExternalCommercial,
                action,
                SubmissionControlMode::Unknown,
            ))
            .unwrap();
            assert_ne!(result.disposition, ExecutionAuthorityDisposition::Blocked);
        }
    }

    #[test]
    fn draft_rule_does_not_authorise_external_automated_submission() {
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::AutonomousSubmissionExplicitlyPermitted,
        );
        inputs.regulatory_profile.instrument_state = RegulatoryInstrumentState::Draft;
        assert_eq!(
            evaluate_execution_authority(inputs).unwrap().disposition,
            ExecutionAuthorityDisposition::Blocked
        );
    }

    #[test]
    fn unknown_or_prohibited_rule_control_blocks_external_submission() {
        for mode in [
            SubmissionControlMode::Unknown,
            SubmissionControlMode::SubmissionProhibited,
        ] {
            let result = evaluate_execution_authority(base_inputs(
                MarketEnvironment::ExternalCommercial,
                TradingWorkflowAction::SubmitBidOrOffer,
                mode,
            ))
            .unwrap();
            assert_eq!(result.disposition, ExecutionAuthorityDisposition::Blocked);
        }
    }

    #[test]
    fn internal_simulation_is_separate_and_never_authorises_commercial_submission() {
        let inputs = base_inputs(
            MarketEnvironment::InternalSimulation,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::Unknown,
        );
        let result = evaluate_execution_authority(inputs).unwrap();
        assert_eq!(result.disposition, ExecutionAuthorityDisposition::SimulationOnlyPermitted);

        let no_access = ExecutionAuthorityInputs {
            internal_simulation_access: EvidenceStatus::Unknown,
            ..inputs
        };
        assert_eq!(
            evaluate_execution_authority(no_access).unwrap().disposition,
            ExecutionAuthorityDisposition::Blocked
        );
    }

    #[test]
    fn commercial_submission_requires_every_accreditation_stage_verified() {
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        let stages = accreditation();
        inputs.participant.accreditation_stages = &stages;
        inputs.human_approval = Some(HumanBidApproval {
            approval_id: "approval/001",
            participant_id: "participant/001",
            bid_manifest_sha256: SHA_A,
            reviewer_id: "reviewer/001",
            reviewer_role: "authorised-trader",
            reviewer_authority: EvidenceStatus::Verified,
            reviewer_authority_evidence: Some(evidence("reviewer/authority", SHA_A)),
            reviewed_at_utc: 80,
            approval_evidence: evidence("approval/receipt", SHA_A),
        });
        let result = evaluate_execution_authority(inputs).unwrap();
        assert_eq!(
            result.disposition,
            ExecutionAuthorityDisposition::CommercialSubmissionPermittedWithHumanReview
        );

        let mut incomplete = stages.clone();
        incomplete[5].1 = EvidenceStatus::Unknown;
        inputs.participant.accreditation_stages = &incomplete;
        assert_eq!(
            evaluate_execution_authority(inputs).unwrap().disposition,
            ExecutionAuthorityDisposition::Blocked
        );
    }

    #[test]
    fn human_approval_must_bind_exact_bid_and_participant() {
        let stages = accreditation();
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        inputs.participant.accreditation_stages = &stages;
        inputs.human_approval = Some(HumanBidApproval {
            approval_id: "approval/001",
            participant_id: "participant/001",
            bid_manifest_sha256: SHA_B,
            reviewer_id: "reviewer/001",
            reviewer_role: "authorised-trader",
            reviewer_authority: EvidenceStatus::Verified,
            reviewer_authority_evidence: Some(evidence("reviewer/authority", SHA_A)),
            reviewed_at_utc: 80,
            approval_evidence: evidence("approval/receipt", SHA_A),
        });
        assert_eq!(
            evaluate_execution_authority(inputs).unwrap().reason,
            ExecutionAuthorityReason::HumanApprovalDoesNotMatchBid
        );
    }

    #[test]
    fn automated_principal_cannot_self_approve_and_approval_receipt_cannot_predate_review() {
        let stages = accreditation();
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        inputs.participant.accreditation_stages = &stages;
        inputs.human_approval = Some(HumanBidApproval {
            approval_id: "approval/self",
            participant_id: "participant/001",
            bid_manifest_sha256: SHA_A,
            reviewer_id: "agent/sawem",
            reviewer_role: "automated-agent",
            reviewer_authority: EvidenceStatus::Verified,
            reviewer_authority_evidence: Some(evidence("reviewer/authority", SHA_A)),
            reviewed_at_utc: 80,
            approval_evidence: evidence("approval/receipt", SHA_A),
        });
        assert_eq!(
            evaluate_execution_authority(inputs).unwrap().reason,
            ExecutionAuthorityReason::HumanApprovalDoesNotMatchBid
        );

        inputs.human_approval = Some(HumanBidApproval {
            approval_id: "approval/chronology",
            participant_id: "participant/001",
            bid_manifest_sha256: SHA_A,
            reviewer_id: "reviewer/001",
            reviewer_role: "authorised-trader",
            reviewer_authority: EvidenceStatus::Verified,
            reviewer_authority_evidence: Some(evidence("reviewer/authority", SHA_A)),
            reviewed_at_utc: 80,
            approval_evidence: GateEvidenceRef {
                available_at_utc: 70,
                ..evidence("approval/receipt", SHA_A)
            },
        });
        assert_eq!(
            evaluate_execution_authority(inputs),
            Err(ExecutionAuthorityError::InvalidEvidenceTimestamp)
        );
    }

    #[test]
    fn explicitly_effective_autonomous_rule_still_requires_admission_accreditation_and_api() {
        let stages = accreditation();
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::AutonomousSubmissionExplicitlyPermitted,
        );
        inputs.participant.accreditation_stages = &stages;
        let result = evaluate_execution_authority(inputs).unwrap();
        assert_eq!(
            result.disposition,
            ExecutionAuthorityDisposition::CommercialSubmissionPermittedByExplicitRule
        );

        inputs.participant.api_submission_status = EvidenceStatus::Unknown;
        assert_eq!(
            evaluate_execution_authority(inputs).unwrap().disposition,
            ExecutionAuthorityDisposition::Blocked
        );
    }

    #[test]
    fn future_known_rule_or_review_receipts_are_rejected() {
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        inputs.regulatory_profile.evidence = Some(GateEvidenceRef {
            available_at_utc: 101,
            ..evidence("rule/future", SHA_B)
        });
        assert_eq!(
            evaluate_execution_authority(inputs),
            Err(ExecutionAuthorityError::InvalidEvidenceTimestamp)
        );
    }

    #[test]
    fn invalid_digests_missing_stage_and_duplicate_stage_fail_closed() {
        let mut inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        inputs.bid_manifest_sha256 = "bad";
        assert_eq!(
            evaluate_execution_authority(inputs),
            Err(ExecutionAuthorityError::InvalidDigest)
        );

        let duplicate = vec![
            (AccreditationStage::DesignApproval, EvidenceStatus::Verified, evidence("stage/1", SHA_A)),
            (AccreditationStage::DesignApproval, EvidenceStatus::Verified, evidence("stage/2", SHA_A)),
        ];
        inputs = base_inputs(
            MarketEnvironment::ExternalCommercial,
            TradingWorkflowAction::SubmitBidOrOffer,
            SubmissionControlMode::HumanReviewRequired,
        );
        inputs.participant.accreditation_stages = &duplicate;
        assert_eq!(
            evaluate_execution_authority(inputs),
            Err(ExecutionAuthorityError::DuplicateAccreditationStage)
        );
    }
}

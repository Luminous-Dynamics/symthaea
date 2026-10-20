//! Evidence-aware one-step experiment prioritization.
//!
//! This is not an acquisition function for hydrodynamics and not a global portfolio optimizer.
//! It selects the next admissible evaluation using explicit fidelity-release obligations first,
//! then a reviewed conservative lower bound on decision-loss reduction per comparable cost. The
//! planner consumes an externally produced utility estimate; it never infers information gain
//! from interval width or prediction error alone.

use super::fidelity_ledger::{
    EvaluationClass, EvidenceOrigin, EvidenceUseStatus, FidelityAudit,
    FidelityAuditDisposition, FidelityProfile,
};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProposalSupportStatus {
    /// The evaluation is within an independently reviewed declared support envelope.
    WithinDeclaredEnvelope,
    /// The experiment is deliberately targeting a predeclared unsupported regime.
    TargetsDeclaredSupportGap,
    /// Applicability/scope is not established.
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionValueContract {
    pub basis_id: String,
    pub estimator_method_id: String,
    pub estimator_method_revision: String,
    pub semantics_reviewed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentProposal {
    pub proposal_id: String,
    /// A registered fidelity requirement this proposal can satisfy. None means optional research.
    pub requirement_id: Option<String>,
    pub campaign_id: String,
    pub candidate_id: String,
    pub configuration_sha256: String,
    pub operating_case_id: String,
    pub evaluation_class: EvaluationClass,
    pub evidence_origin: EvidenceOrigin,
    pub method_id: String,
    pub method_revision: String,
    pub source_id: String,
    /// Digest of exact source/protocol reference bytes; format check only, not publisher auth.
    pub source_reference_sha256: String,
    pub evidence_use_status: EvidenceUseStatus,
    pub cost_basis_id: String,
    /// Missing/zero cost is never interpreted as free.
    pub cost_units: Option<u64>,
    pub support_status: ProposalSupportStatus,
    /// Conservative lower bound on expected decision-loss reduction in the declared value unit.
    /// This is an externally computed and reviewed quantity; it is not model uncertainty itself.
    pub decision_value_lower_bound_units: Option<i64>,
    pub decision_value_basis_id: String,
    /// Digest of the source evidence / receipt supporting the decision-value estimate.
    pub decision_value_evidence_sha256: Option<String>,
    pub decision_value_method_id: Option<String>,
    pub decision_value_method_revision: Option<String>,
    pub decision_value_semantics_reviewed: bool,
    pub support_assessment_reviewed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalDisposition {
    EligibleForMandatoryRequirement,
    EligibleForOptionalRanking,
    BlockedInvalidIdentityOrDigest,
    BlockedWrongCampaignOrCostBasis,
    BlockedRightsNotPermitted,
    BlockedUnknownSupport,
    BlockedUnreviewedSupport,
    BlockedUtilityUnknownOrUnreviewed,
    BlockedNonPositiveDecisionValue,
    BlockedBudget,
    BlockedUnregisteredRequirement,
    BlockedRequirementMismatch,
    BlockedDuplicateProposalId,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProposalIssue {
    EmptyIdentity { field: &'static str },
    InvalidDigest { field: &'static str },
    MissingOrZeroCost,
    CostBasisMismatch,
    CampaignMismatch,
    RightsNotPermitted { status: EvidenceUseStatus },
    SupportUnknown,
    SupportNotReviewed,
    UnknownDecisionValue,
    InvalidDecisionValueEvidence,
    DecisionValueNotReviewed,
    DecisionValueBasisMismatch,
    NonPositiveDecisionValue { value: i64 },
    UnregisteredRequirement { requirement_id: String },
    RequirementMismatch { requirement_id: String, field: &'static str },
    BudgetExceeded { cost_units: u64, budget_units: u64 },
    DuplicateProposalId { proposal_id: String, count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalAssessment {
    pub proposal_id: String,
    pub disposition: ProposalDisposition,
    pub is_mandatory: bool,
    pub requirement_id: Option<String>,
    pub cost_units: Option<u64>,
    pub decision_value_lower_bound_units: Option<i64>,
    pub issues: Vec<ProposalIssue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentPriorityDisposition {
    SelectedMandatoryEvidence,
    SelectedOptionalExperiment,
    MandatoryRequirementHasNoAdmissibleProposal,
    MandatoryEvidenceExceedsBudget,
    NoBeneficialOptionalProposal,
    InvalidProfileOrAudit,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExperimentPriorityIssue {
    EmptyDecisionValueBasis,
    EmptyDecisionValueEstimatorIdentity { field: &'static str },
    DecisionValueEstimatorContractNotReviewed,
    ProfileAuditIdentityMismatch,
    ProfileAuditManifestMismatch,
    ProfileAuditCostBasisMismatch,
    ProfileDigestInvalid,
    CostCannotBeAudited,
    AuditInvalid,
    AuditRequirementSetMismatch,
    AuditRequirementProgressInconsistent { requirement_id: String },
    MissingMandatoryProposal { requirement_id: String },
    MandatoryProposalOverBudget {
        requirement_id: String,
        cheapest_cost_units: u64,
        budget_units: u64,
    },
    NoAdmissibleProposalForRequirement { requirement_id: String },
    NoPositiveOptionalDecisionValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MandatoryRequirementStatus {
    pub requirement_id: String,
    pub candidate_id: String,
    pub operating_case_id: String,
    pub required_successes: u32,
    pub completed_successes: u32,
    pub remaining_successes: u32,
    pub admissible_proposals_within_budget: usize,
    pub cheapest_admissible_cost_units: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentPriorityReport {
    pub profile_id: String,
    pub campaign_id: String,
    pub cost_basis_id: String,
    pub decision_value_basis_id: String,
    pub decision_value_estimator_method_id: String,
    pub decision_value_estimator_method_revision: String,
    pub disposition: ExperimentPriorityDisposition,
    pub budget_remaining_units: u64,
    pub campaign_spent_units: Option<u128>,
    pub mandatory_requirements_due: Vec<MandatoryRequirementStatus>,
    pub selected_proposal_id: Option<String>,
    pub selected_requirement_id: Option<String>,
    pub selected_cost_units: Option<u64>,
    pub selected_decision_value_lower_bound_units: Option<i64>,
    pub selected_value_per_cost_numerator: Option<u64>,
    pub selected_value_per_cost_denominator: Option<u64>,
    pub assessments: Vec<ProposalAssessment>,
    pub issues: Vec<ExperimentPriorityIssue>,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn validate_profile_audit(
    profile: &FidelityProfile,
    audit: &FidelityAudit,
    value_contract: &DecisionValueContract,
) -> (bool, Vec<ExperimentPriorityIssue>) {
    let mut issues = Vec::new();
    if !nonempty(&value_contract.basis_id) {
        issues.push(ExperimentPriorityIssue::EmptyDecisionValueBasis);
    }
    for (value, field) in [
        (value_contract.estimator_method_id.as_str(), "estimator_method_id"),
        (value_contract.estimator_method_revision.as_str(), "estimator_method_revision"),
    ] {
        if !nonempty(value) {
            issues.push(ExperimentPriorityIssue::EmptyDecisionValueEstimatorIdentity { field });
        }
    }
    if !value_contract.semantics_reviewed {
        issues.push(ExperimentPriorityIssue::DecisionValueEstimatorContractNotReviewed);
    }
    if profile.profile_id != audit.profile_id || profile.campaign_id != audit.campaign_id {
        issues.push(ExperimentPriorityIssue::ProfileAuditIdentityMismatch);
    }
    if profile.manifest_sha256 != audit.profile_manifest_sha256 {
        issues.push(ExperimentPriorityIssue::ProfileAuditManifestMismatch);
    }
    if profile.cost_basis_id != audit.cost_basis_id {
        issues.push(ExperimentPriorityIssue::ProfileAuditCostBasisMismatch);
    }
    if !valid_sha256(&profile.manifest_sha256) {
        issues.push(ExperimentPriorityIssue::ProfileDigestInvalid);
    }
    if audit.total_cost_units.is_none() {
        issues.push(ExperimentPriorityIssue::CostCannotBeAudited);
    }
    if matches!(
        audit.disposition,
        FidelityAuditDisposition::InvalidProfile | FidelityAuditDisposition::InvalidReceipts
    ) {
        issues.push(ExperimentPriorityIssue::AuditInvalid);
    }

    let mut expected = BTreeMap::new();
    for requirement in &profile.requirements {
        if expected.insert(requirement.requirement_id.as_str(), requirement).is_some() {
            issues.push(ExperimentPriorityIssue::AuditRequirementSetMismatch);
        }
    }
    let mut observed = BTreeMap::new();
    for progress in &audit.requirements {
        if observed.insert(progress.requirement_id.as_str(), progress).is_some() {
            issues.push(ExperimentPriorityIssue::AuditRequirementSetMismatch);
        }
    }
    if expected.len() != observed.len()
        || expected.keys().any(|id| !observed.contains_key(id))
        || observed.keys().any(|id| !expected.contains_key(id))
    {
        issues.push(ExperimentPriorityIssue::AuditRequirementSetMismatch);
    }

    for (requirement_id, requirement) in expected {
        if let Some(progress) = observed.get(requirement_id) {
            if progress.candidate_id != requirement.candidate_id
                || progress.operating_case_id != requirement.operating_case_id
                || progress.evaluation_class != requirement.evaluation_class
                || progress.evidence_origin != requirement.evidence_origin
                || progress.required_successes != requirement.required_successes
                || progress.covered != (progress.completed_successes >= progress.required_successes)
                || progress.completed_successes > progress.total_attempts
            {
                issues.push(ExperimentPriorityIssue::AuditRequirementProgressInconsistent {
                    requirement_id: requirement_id.to_string(),
                });
            }
        }
    }
    issues.sort();
    issues.dedup();
    (issues.is_empty(), issues)
}

fn requirement_matches(
    requirement: &super::fidelity_ledger::FidelityRequirement,
    proposal: &ExperimentProposal,
) -> Result<(), &'static str> {
    if requirement.candidate_id != proposal.candidate_id {
        return Err("candidate_id");
    }
    if requirement.configuration_sha256 != proposal.configuration_sha256 {
        return Err("configuration_sha256");
    }
    if requirement.operating_case_id != proposal.operating_case_id {
        return Err("operating_case_id");
    }
    if requirement.evaluation_class != proposal.evaluation_class {
        return Err("evaluation_class");
    }
    if requirement.evidence_origin != proposal.evidence_origin {
        return Err("evidence_origin");
    }
    if requirement.method_id != proposal.method_id {
        return Err("method_id");
    }
    if requirement.method_revision != proposal.method_revision {
        return Err("method_revision");
    }
    Ok(())
}

fn assess_proposal(
    profile: &FidelityProfile,
    proposal: &ExperimentProposal,
    requirements: &BTreeMap<&str, &super::fidelity_ledger::FidelityRequirement>,
    pending_mandatory: &BTreeSet<String>,
    value_contract: &DecisionValueContract,
    budget_remaining_units: u64,
) -> ProposalAssessment {
    let mut issues = Vec::new();
    let mut eligible_class = None;
    let mut is_mandatory = false;

    for (value, field) in [
        (proposal.proposal_id.as_str(), "proposal_id"),
        (proposal.campaign_id.as_str(), "campaign_id"),
        (proposal.candidate_id.as_str(), "candidate_id"),
        (proposal.operating_case_id.as_str(), "operating_case_id"),
        (proposal.method_id.as_str(), "method_id"),
        (proposal.method_revision.as_str(), "method_revision"),
        (proposal.source_id.as_str(), "source_id"),
        (proposal.cost_basis_id.as_str(), "cost_basis_id"),
    ] {
        if !nonempty(value) {
            issues.push(ProposalIssue::EmptyIdentity { field });
        }
    }
    for (value, field) in [
        (proposal.configuration_sha256.as_str(), "configuration_sha256"),
        (proposal.source_reference_sha256.as_str(), "source_reference_sha256"),
    ] {
        if !valid_sha256(value) {
            issues.push(ProposalIssue::InvalidDigest { field });
        }
    }
    if proposal.campaign_id != profile.campaign_id {
        issues.push(ProposalIssue::CampaignMismatch);
    }
    if proposal.cost_basis_id != profile.cost_basis_id {
        issues.push(ProposalIssue::CostBasisMismatch);
    }
    if proposal.evidence_use_status != EvidenceUseStatus::PermittedForDeclaredUse {
        issues.push(ProposalIssue::RightsNotPermitted {
            status: proposal.evidence_use_status,
        });
    }
    if proposal.support_status == ProposalSupportStatus::Unknown {
        issues.push(ProposalIssue::SupportUnknown);
    }
    if !proposal.support_assessment_reviewed {
        issues.push(ProposalIssue::SupportNotReviewed);
    }
    if proposal.cost_units.is_none_or(|cost| cost == 0) {
        issues.push(ProposalIssue::MissingOrZeroCost);
    } else if let Some(cost) = proposal.cost_units {
        if cost > budget_remaining_units {
            issues.push(ProposalIssue::BudgetExceeded {
                cost_units: cost,
                budget_units: budget_remaining_units,
            });
        }
    }

    if let Some(requirement_id) = &proposal.requirement_id {
        if let Some(requirement) = requirements.get(requirement_id.as_str()) {
            match requirement_matches(requirement, proposal) {
                Ok(()) => {
                    is_mandatory = requirement.required_successes > 0
                        && pending_mandatory.contains(requirement_id);
                    eligible_class = Some(if is_mandatory {
                        ProposalDisposition::EligibleForMandatoryRequirement
                    } else {
                        ProposalDisposition::EligibleForOptionalRanking
                    });
                }
                Err(field) => issues.push(ProposalIssue::RequirementMismatch {
                    requirement_id: requirement_id.clone(),
                    field,
                }),
            }
        } else {
            issues.push(ProposalIssue::UnregisteredRequirement {
                requirement_id: requirement_id.clone(),
            });
        }
    } else {
        eligible_class = Some(ProposalDisposition::EligibleForOptionalRanking);
    }

    let utility_is_structurally_valid = match proposal.decision_value_lower_bound_units {
        None => {
            issues.push(ProposalIssue::UnknownDecisionValue);
            false
        }
        Some(value) if value <= 0 => {
            issues.push(ProposalIssue::NonPositiveDecisionValue { value });
            false
        }
        Some(_) => {
            let mut valid = true;
            if proposal.decision_value_basis_id != value_contract.basis_id
                || proposal.decision_value_method_id.as_deref()
                    != Some(value_contract.estimator_method_id.as_str())
                || proposal.decision_value_method_revision.as_deref()
                    != Some(value_contract.estimator_method_revision.as_str())
            {
                issues.push(ProposalIssue::DecisionValueBasisMismatch);
                valid = false;
            }
            match &proposal.decision_value_evidence_sha256 {
                Some(digest) if valid_sha256(digest) => {}
                _ => {
                    issues.push(ProposalIssue::InvalidDecisionValueEvidence);
                    valid = false;
                }
            }
            for field in [
                proposal.decision_value_method_id.as_deref(),
                proposal.decision_value_method_revision.as_deref(),
            ] {
                if field.is_none_or(|value| !nonempty(value)) {
                    issues.push(ProposalIssue::InvalidDecisionValueEvidence);
                    valid = false;
                }
            }
            if !proposal.decision_value_semantics_reviewed || !value_contract.semantics_reviewed {
                issues.push(ProposalIssue::DecisionValueNotReviewed);
                valid = false;
            }
            valid
        }
    };

    let basic_invalid = issues.iter().any(|issue| {
        matches!(
            issue,
            ProposalIssue::EmptyIdentity { .. }
                | ProposalIssue::InvalidDigest { .. }
                | ProposalIssue::MissingOrZeroCost
                | ProposalIssue::CostBasisMismatch
                | ProposalIssue::CampaignMismatch
                | ProposalIssue::RightsNotPermitted { .. }
                | ProposalIssue::SupportUnknown
                | ProposalIssue::SupportNotReviewed
                | ProposalIssue::UnregisteredRequirement { .. }
                | ProposalIssue::RequirementMismatch { .. }
        )
    });
    let over_budget = issues
        .iter()
        .any(|issue| matches!(issue, ProposalIssue::BudgetExceeded { .. }));

    let disposition = if basic_invalid {
        if issues.iter().any(|issue| matches!(issue, ProposalIssue::RightsNotPermitted { .. })) {
            ProposalDisposition::BlockedRightsNotPermitted
        } else if issues.iter().any(|issue| matches!(issue, ProposalIssue::SupportUnknown)) {
            ProposalDisposition::BlockedUnknownSupport
        } else if issues.iter().any(|issue| matches!(issue, ProposalIssue::SupportNotReviewed)) {
            ProposalDisposition::BlockedUnreviewedSupport
        } else if issues.iter().any(|issue| matches!(issue, ProposalIssue::UnregisteredRequirement { .. })) {
            ProposalDisposition::BlockedUnregisteredRequirement
        } else if issues.iter().any(|issue| matches!(issue, ProposalIssue::RequirementMismatch { .. })) {
            ProposalDisposition::BlockedRequirementMismatch
        } else if issues.iter().any(|issue| matches!(issue, ProposalIssue::CampaignMismatch | ProposalIssue::CostBasisMismatch)) {
            ProposalDisposition::BlockedWrongCampaignOrCostBasis
        } else {
            ProposalDisposition::BlockedInvalidIdentityOrDigest
        }
    } else if over_budget {
        ProposalDisposition::BlockedBudget
    } else if is_mandatory
        && eligible_class == Some(ProposalDisposition::EligibleForMandatoryRequirement)
    {
        ProposalDisposition::EligibleForMandatoryRequirement
    } else if eligible_class == Some(ProposalDisposition::EligibleForOptionalRanking) {
        if !utility_is_structurally_valid {
            if issues.iter().any(|issue| matches!(issue, ProposalIssue::NonPositiveDecisionValue { .. })) {
                ProposalDisposition::BlockedNonPositiveDecisionValue
            } else {
                ProposalDisposition::BlockedUtilityUnknownOrUnreviewed
            }
        } else {
            ProposalDisposition::EligibleForOptionalRanking
        }
    } else {
        ProposalDisposition::BlockedInvalidIdentityOrDigest
    };

    issues.sort();
    issues.dedup();
    ProposalAssessment {
        proposal_id: proposal.proposal_id.clone(),
        disposition,
        is_mandatory,
        requirement_id: proposal.requirement_id.clone(),
        cost_units: proposal.cost_units,
        decision_value_lower_bound_units: proposal.decision_value_lower_bound_units,
        issues,
    }
}

fn ratio_order(a: &ProposalAssessment, b: &ProposalAssessment) -> Ordering {
    let a_value = a.decision_value_lower_bound_units.unwrap_or(0).max(0) as u64;
    let b_value = b.decision_value_lower_bound_units.unwrap_or(0).max(0) as u64;
    let a_cost = a.cost_units.unwrap_or(u64::MAX).max(1);
    let b_cost = b.cost_units.unwrap_or(u64::MAX).max(1);

    match (a_value > 0, b_value > 0) {
        (true, true) => {
            let lhs = (a_value as u128) * (b_cost as u128);
            let rhs = (b_value as u128) * (a_cost as u128);
            rhs.cmp(&lhs).then_with(|| a.proposal_id.cmp(&b.proposal_id))
        }
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => a_cost.cmp(&b_cost).then_with(|| a.proposal_id.cmp(&b.proposal_id)),
    }
}

fn proposal_is_mandatory_candidate(assessment: &ProposalAssessment) -> bool {
    assessment.disposition == ProposalDisposition::EligibleForMandatoryRequirement
}

fn proposal_is_optional_candidate(assessment: &ProposalAssessment) -> bool {
    assessment.disposition == ProposalDisposition::EligibleForOptionalRanking
        && assessment.decision_value_lower_bound_units.is_some_and(|value| value > 0)
}

fn select_mandatory(
    due: &[MandatoryRequirementStatus],
    assessments: &[ProposalAssessment],
) -> Option<(String, ProposalAssessment)> {
    let mut feasible: Vec<(&MandatoryRequirementStatus, Vec<&ProposalAssessment>)> = due
        .iter()
        .filter_map(|status| {
            let choices: Vec<&ProposalAssessment> = assessments
                .iter()
                .filter(|assessment| {
                    proposal_is_mandatory_candidate(assessment)
                        && assessment.requirement_id.as_deref() == Some(status.requirement_id.as_str())
                })
                .collect();
            (!choices.is_empty()).then_some((status, choices))
        })
        .collect();
    feasible.sort_by(|(a_status, a_choices), (b_status, b_choices)| {
        a_choices
            .len()
            .cmp(&b_choices.len())
            .then_with(|| a_status.requirement_id.cmp(&b_status.requirement_id))
    });
    let (status, mut choices) = feasible.into_iter().next()?;
    choices.sort_by(|a, b| ratio_order(a, b));
    Some((status.requirement_id.clone(), choices[0].clone()))
}

/// Select one next evaluation under hard fidelity-release gates and a single remaining budget.
///
/// Mandatory evidence is preferred over optional research. If no mandatory gate is outstanding,
/// only proposals with a positive, reviewed decision-value lower bound, established/reviewed scope,
/// permitted evidence use, matching cost basis and exact source/method identities can be ranked.
/// This function does not compute the lower bound; that requires a separate, reviewed estimator.
pub fn prioritize_next_experiment(
    profile: &FidelityProfile,
    audit: &FidelityAudit,
    proposals: &[ExperimentProposal],
    value_contract: &DecisionValueContract,
    budget_remaining_units: u64,
) -> ExperimentPriorityReport {
    let (audit_valid, mut issues) =
        validate_profile_audit(profile, audit, value_contract);
    let requirements: BTreeMap<&str, &super::fidelity_ledger::FidelityRequirement> = profile
        .requirements
        .iter()
        .map(|requirement| (requirement.requirement_id.as_str(), requirement))
        .collect();
    let progress: BTreeMap<&str, &super::fidelity_ledger::RequirementProgress> = audit
        .requirements
        .iter()
        .map(|progress| (progress.requirement_id.as_str(), progress))
        .collect();

    let mut due = Vec::new();
    let mut pending_mandatory = BTreeSet::new();
    for requirement in &profile.requirements {
        if requirement.required_successes == 0 {
            continue;
        }
        let Some(progress) = progress.get(requirement.requirement_id.as_str()) else {
            continue;
        };
        if progress.completed_successes < requirement.required_successes {
            pending_mandatory.insert(requirement.requirement_id.clone());
            due.push(MandatoryRequirementStatus {
                requirement_id: requirement.requirement_id.clone(),
                candidate_id: requirement.candidate_id.clone(),
                operating_case_id: requirement.operating_case_id.clone(),
                required_successes: requirement.required_successes,
                completed_successes: progress.completed_successes,
                remaining_successes: requirement.required_successes - progress.completed_successes,
                admissible_proposals_within_budget: 0,
                cheapest_admissible_cost_units: None,
            });
        }
    }

    let mut proposal_id_counts = BTreeMap::<String, usize>::new();
    for proposal in proposals {
        *proposal_id_counts.entry(proposal.proposal_id.clone()).or_default() += 1;
    }

    let mut assessments: Vec<ProposalAssessment> = proposals
        .iter()
        .map(|proposal| {
            let mut assessment = assess_proposal(
                profile,
                proposal,
                &requirements,
                &pending_mandatory,
                value_contract,
                budget_remaining_units,
            );
            if let Some(count) = proposal_id_counts.get(&proposal.proposal_id).copied() {
                if count > 1 {
                    assessment.disposition = ProposalDisposition::BlockedDuplicateProposalId;
                    assessment.issues.push(ProposalIssue::DuplicateProposalId {
                        proposal_id: proposal.proposal_id.clone(),
                        count,
                    });
                    assessment.issues.sort();
                    assessment.issues.dedup();
                }
            }
            assessment
        })
        .collect();
    assessments.sort_by(|a, b| a.proposal_id.cmp(&b.proposal_id));

    for status in &mut due {
        let choices: Vec<&ProposalAssessment> = assessments
            .iter()
            .filter(|assessment| {
                proposal_is_mandatory_candidate(assessment)
                    && assessment.requirement_id.as_deref() == Some(status.requirement_id.as_str())
            })
            .collect();
        status.admissible_proposals_within_budget = choices.len();
        status.cheapest_admissible_cost_units = choices.iter().filter_map(|a| a.cost_units).min();
    }

    // Surface every outstanding mandatory requirement that has no admissible option under
    // the current budget, even when a different mandatory requirement can be selected now.
    for status in &due {
        if status.admissible_proposals_within_budget > 0 {
            continue;
        }
        let proposals_for_requirement: Vec<&ProposalAssessment> = assessments
            .iter()
            .filter(|assessment| {
                assessment.requirement_id.as_deref() == Some(status.requirement_id.as_str())
            })
            .collect();
        if proposals_for_requirement.is_empty() {
            issues.push(ExperimentPriorityIssue::MissingMandatoryProposal {
                requirement_id: status.requirement_id.clone(),
            });
        } else {
            let budget_blocked_cost = proposals_for_requirement
                .iter()
                .filter(|assessment| assessment.disposition == ProposalDisposition::BlockedBudget)
                .filter_map(|assessment| assessment.cost_units)
                .min();
            if let Some(cost) = budget_blocked_cost {
                issues.push(ExperimentPriorityIssue::MandatoryProposalOverBudget {
                    requirement_id: status.requirement_id.clone(),
                    cheapest_cost_units: cost,
                    budget_units: budget_remaining_units,
                });
            } else {
                issues.push(ExperimentPriorityIssue::NoAdmissibleProposalForRequirement {
                    requirement_id: status.requirement_id.clone(),
                });
            }
        }
    }

    let (disposition, selected, selected_requirement_id) = if !audit_valid {
        (ExperimentPriorityDisposition::InvalidProfileOrAudit, None, None)
    } else if !due.is_empty() {
        if let Some((requirement_id, proposal)) = select_mandatory(&due, &assessments) {
            (
                ExperimentPriorityDisposition::SelectedMandatoryEvidence,
                Some(proposal),
                Some(requirement_id),
            )
        } else {
            let over_budget = issues.iter().any(|issue| {
                matches!(issue, ExperimentPriorityIssue::MandatoryProposalOverBudget { .. })
            });
            (
                if over_budget {
                    ExperimentPriorityDisposition::MandatoryEvidenceExceedsBudget
                } else {
                    ExperimentPriorityDisposition::MandatoryRequirementHasNoAdmissibleProposal
                },
                None,
                None,
            )
        }
    } else {
        let mut optional: Vec<&ProposalAssessment> = assessments
            .iter()
            .filter(|assessment| proposal_is_optional_candidate(assessment))
            .collect();
        optional.sort_by(|a, b| ratio_order(a, b));
        if let Some(best) = optional.first() {
            (
                ExperimentPriorityDisposition::SelectedOptionalExperiment,
                Some((*best).clone()),
                best.requirement_id.clone(),
            )
        } else {
            issues.push(ExperimentPriorityIssue::NoPositiveOptionalDecisionValue);
            (
                ExperimentPriorityDisposition::NoBeneficialOptionalProposal,
                None,
                None,
            )
        }
    };

    let selected_value = selected
        .as_ref()
        .and_then(|proposal| proposal.decision_value_lower_bound_units)
        .filter(|value| *value > 0)
        .map(|value| value as u64);
    let selected_cost = selected.as_ref().and_then(|proposal| proposal.cost_units);
    ExperimentPriorityReport {
        profile_id: profile.profile_id.clone(),
        campaign_id: profile.campaign_id.clone(),
        cost_basis_id: profile.cost_basis_id.clone(),
        decision_value_basis_id: value_contract.basis_id.clone(),
        decision_value_estimator_method_id: value_contract.estimator_method_id.clone(),
        decision_value_estimator_method_revision: value_contract.estimator_method_revision.clone(),
        disposition,
        budget_remaining_units,
        campaign_spent_units: audit.total_cost_units,
        mandatory_requirements_due: due,
        selected_proposal_id: selected.as_ref().map(|proposal| proposal.proposal_id.clone()),
        selected_requirement_id,
        selected_cost_units: selected_cost,
        selected_decision_value_lower_bound_units: selected
            .as_ref()
            .and_then(|proposal| proposal.decision_value_lower_bound_units),
        selected_value_per_cost_numerator: selected_value,
        selected_value_per_cost_denominator: selected_cost,
        assessments,
        issues: {
            issues.sort();
            issues.dedup();
            issues
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fidelity_ledger::{
        audit_fidelity_campaign, AttemptStatus, EvaluationReceipt, FidelityRequirement,
    };

    const CONFIG_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SOURCE: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const OUTPUT: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const VALUE_EVIDENCE: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn requirement(
        requirement_id: &str,
        candidate_id: &str,
        class: EvaluationClass,
        origin: EvidenceOrigin,
        method: &str,
        revision: &str,
        required_successes: u32,
    ) -> FidelityRequirement {
        FidelityRequirement {
            requirement_id: requirement_id.to_string(),
            candidate_id: candidate_id.to_string(),
            configuration_sha256: CONFIG_A.to_string(),
            operating_case_id: "shallow-water".to_string(),
            evaluation_class: class,
            evidence_origin: origin,
            method_id: method.to_string(),
            method_revision: revision.to_string(),
            required_successes,
        }
    }

    fn profile() -> FidelityProfile {
        FidelityProfile {
            profile_id: "marine-priority-profile".to_string(),
            campaign_id: "campaign-1".to_string(),
            manifest_sha256: CONFIG_A.to_string(),
            cost_basis_id: "lab-hours-v1".to_string(),
            requirements: vec![
                requirement(
                    "physical-confirmation",
                    "hull-candidate",
                    EvaluationClass::PhysicalTest,
                    EvidenceOrigin::ExperimentalMeasurement,
                    "towing-tank-protocol",
                    "rev-1",
                    1,
                ),
                requirement(
                    "screening",
                    "hull-candidate",
                    EvaluationClass::LowFidelitySimulation,
                    EvidenceOrigin::CfdSimulation,
                    "low-cfd",
                    "rev-2",
                    0,
                ),
            ],
        }
    }

    fn receipt(req: &FidelityRequirement, receipt_id: &str, cost: u64) -> EvaluationReceipt {
        EvaluationReceipt {
            receipt_id: receipt_id.to_string(),
            campaign_id: "campaign-1".to_string(),
            requirement_id: req.requirement_id.clone(),
            candidate_id: req.candidate_id.clone(),
            configuration_sha256: req.configuration_sha256.clone(),
            operating_case_id: req.operating_case_id.clone(),
            evaluation_class: req.evaluation_class,
            evidence_origin: req.evidence_origin,
            method_id: req.method_id.clone(),
            method_revision: req.method_revision.clone(),
            source_id: "source-1".to_string(),
            source_reference_sha256: SOURCE.to_string(),
            evidence_use_status: EvidenceUseStatus::PermittedForDeclaredUse,
            cost_basis_id: "lab-hours-v1".to_string(),
            cost_units: Some(cost),
            status: AttemptStatus::Complete,
            output_sha256: Some(OUTPUT.to_string()),
        }
    }

    fn value_contract() -> DecisionValueContract {
        DecisionValueContract {
            basis_id: "loss-units-v1".to_string(),
            estimator_method_id: "paired-loss-estimator".to_string(),
            estimator_method_revision: "rev-1".to_string(),
            semantics_reviewed: true,
        }
    }

    fn audit(require_physical: bool) -> (FidelityProfile, FidelityAudit) {
        let profile = profile();
        let receipts = if require_physical {
            vec![]
        } else {
            vec![receipt(&profile.requirements[0], "physical-1", 100)]
        };
        let audit = audit_fidelity_campaign(&profile, &receipts);
        (profile, audit)
    }

    fn proposal(
        proposal_id: &str,
        requirement_id: Option<&str>,
        class: EvaluationClass,
        origin: EvidenceOrigin,
        method_id: &str,
        method_revision: &str,
        cost: Option<u64>,
        value: Option<i64>,
    ) -> ExperimentProposal {
        ExperimentProposal {
            proposal_id: proposal_id.to_string(),
            requirement_id: requirement_id.map(str::to_string),
            campaign_id: "campaign-1".to_string(),
            candidate_id: "hull-candidate".to_string(),
            configuration_sha256: CONFIG_A.to_string(),
            operating_case_id: "shallow-water".to_string(),
            evaluation_class: class,
            evidence_origin: origin,
            method_id: method_id.to_string(),
            method_revision: method_revision.to_string(),
            source_id: "source-protocol".to_string(),
            source_reference_sha256: SOURCE.to_string(),
            evidence_use_status: EvidenceUseStatus::PermittedForDeclaredUse,
            cost_basis_id: "lab-hours-v1".to_string(),
            cost_units: cost,
            support_status: ProposalSupportStatus::WithinDeclaredEnvelope,
            decision_value_lower_bound_units: value,
            decision_value_basis_id: "loss-units-v1".to_string(),
            decision_value_evidence_sha256: value.map(|_| VALUE_EVIDENCE.to_string()),
            decision_value_method_id: value.map(|_| "paired-loss-estimator".to_string()),
            decision_value_method_revision: value.map(|_| "rev-1".to_string()),
            decision_value_semantics_reviewed: value.is_some(),
            support_assessment_reviewed: true,
        }
    }

    fn physical(proposal_id: &str, cost: Option<u64>, value: Option<i64>) -> ExperimentProposal {
        proposal(
            proposal_id,
            Some("physical-confirmation"),
            EvaluationClass::PhysicalTest,
            EvidenceOrigin::ExperimentalMeasurement,
            "towing-tank-protocol",
            "rev-1",
            cost,
            value,
        )
    }

    fn optional(proposal_id: &str, cost: Option<u64>, value: Option<i64>) -> ExperimentProposal {
        proposal(
            proposal_id,
            Some("screening"),
            EvaluationClass::LowFidelitySimulation,
            EvidenceOrigin::CfdSimulation,
            "low-cfd",
            "rev-2",
            cost,
            value,
        )
    }

    #[test]
    fn mandatory_physical_evidence_beats_optional_high_value_low_fidelity_screen() {
        let (profile, audit) = audit(true);
        let proposals = vec![
            physical("tank-run", Some(80), None),
            optional("cheap-screen", Some(1), Some(10_000)),
        ];
        let report = prioritize_next_experiment(&profile, &audit, &proposals, &value_contract(), 100);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::SelectedMandatoryEvidence);
        assert_eq!(report.selected_proposal_id.as_deref(), Some("tank-run"));
    }

    #[test]
    fn mandatory_unknown_utility_is_still_selectable_as_required_evidence() {
        let (profile, audit) = audit(true);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[physical("tank-run", Some(80), None)],
            &value_contract(),
            100,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::SelectedMandatoryEvidence);
        assert_eq!(report.selected_decision_value_lower_bound_units, None);
    }

    #[test]
    fn required_physical_trial_cannot_be_replaced_by_cheaper_optional_cfd() {
        let (profile, audit) = audit(true);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[optional("cheap-screen", Some(1), Some(10_000))],
            &value_contract(),
            100,
        );
        assert_eq!(
            report.disposition,
            ExperimentPriorityDisposition::MandatoryRequirementHasNoAdmissibleProposal
        );
    }

    #[test]
    fn mandatory_budget_overrun_returns_explicit_blocked_status() {
        let (profile, audit) = audit(true);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[physical("tank-run", Some(120), None)],
            &value_contract(),
            100,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::MandatoryEvidenceExceedsBudget);
    }

    #[test]
    fn optional_trials_rank_by_conservative_value_per_comparable_cost() {
        let (profile, audit) = audit(false);
        let proposals = vec![
            optional("low-ratio", Some(20), Some(40)),
            optional("high-ratio", Some(5), Some(20)),
        ];
        let report = prioritize_next_experiment(&profile, &audit, &proposals, &value_contract(), 30);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::SelectedOptionalExperiment);
        assert_eq!(report.selected_proposal_id.as_deref(), Some("high-ratio"));
        assert_eq!(report.selected_value_per_cost_numerator, Some(20));
        assert_eq!(report.selected_value_per_cost_denominator, Some(5));
    }

    #[test]
    fn unknown_utility_is_not_ranked_as_beneficial_optional_work() {
        let (profile, audit) = audit(false);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[optional("unknown", Some(2), None)],
            &value_contract(),
            10,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn non_positive_decision_value_is_not_ranked() {
        let (profile, audit) = audit(false);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[optional("zero", Some(2), Some(0))],
            &value_contract(),
            10,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn restricted_source_cannot_be_ranked_even_with_large_utility() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("restricted", Some(2), Some(10_000));
        proposal.evidence_use_status = EvidenceUseStatus::Restricted;
        let report = prioritize_next_experiment(&profile, &audit, &[proposal], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn unknown_support_cannot_be_ranked() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("unknown-support", Some(2), Some(100));
        proposal.support_status = ProposalSupportStatus::Unknown;
        let report = prioritize_next_experiment(&profile, &audit, &[proposal], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn mismatched_cost_basis_is_blocked() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("wrong-cost-basis", Some(2), Some(100));
        proposal.cost_basis_id = "other-units".to_string();
        let report = prioritize_next_experiment(&profile, &audit, &[proposal], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn duplicate_proposal_ids_are_never_order_selected() {
        let (profile, audit) = audit(false);
        let proposals = vec![
            optional("same-id", Some(2), Some(100)),
            optional("same-id", Some(1), Some(200)),
        ];
        let report = prioritize_next_experiment(&profile, &audit, &proposals, &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
        assert!(report.assessments.iter().all(|item| item.disposition == ProposalDisposition::BlockedDuplicateProposalId));
    }

    #[test]
    fn optional_tie_breaking_is_independent_of_input_order() {
        let (profile, audit) = audit(false);
        let a = optional("proposal-a", Some(5), Some(50));
        let b = optional("proposal-b", Some(10), Some(100));
        let first = prioritize_next_experiment(&profile, &audit, &[b.clone(), a.clone()], &value_contract(), 20);
        let second = prioritize_next_experiment(&profile, &audit, &[a, b], &value_contract(), 20);
        assert_eq!(first.selected_proposal_id, second.selected_proposal_id);
        assert_eq!(first.selected_proposal_id.as_deref(), Some("proposal-a"));
    }

    #[test]
    fn malformed_decision_value_evidence_is_blocked() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("stale-utility", Some(2), Some(100));
        proposal.decision_value_evidence_sha256 = Some("not-a-digest".to_string());
        let report = prioritize_next_experiment(&profile, &audit, &[proposal], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn audit_from_different_manifest_fails_closed() {
        let (profile, mut audit) = audit(false);
        audit.profile_manifest_sha256 = "e".repeat(64);
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[],
            &value_contract(),
            10,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::InvalidProfileOrAudit);
    }

    #[test]
    fn audit_from_different_cost_basis_fails_closed() {
        let (profile, mut audit) = audit(false);
        audit.cost_basis_id = "wrong-units".to_string();
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[],
            &value_contract(),
            10,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::InvalidProfileOrAudit);
    }

    #[test]
    fn unlike_decision_value_estimators_cannot_be_compared() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("wrong-estimator", Some(2), Some(100));
        proposal.decision_value_method_id = Some("different-estimator".to_string());
        let report = prioritize_next_experiment(
            &profile,
            &audit,
            &[proposal],
            &value_contract(),
            10,
        );
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }

    #[test]
    fn audit_profile_mismatch_fails_closed() {
        let (profile, mut audit) = audit(false);
        audit.profile_id = "different-profile".to_string();
        let report = prioritize_next_experiment(&profile, &audit, &[], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::InvalidProfileOrAudit);
    }

    #[test]
    fn targeting_a_declared_gap_requires_an_explicit_review() {
        let (profile, audit) = audit(false);
        let mut proposal = optional("support-gap", Some(2), Some(100));
        proposal.support_status = ProposalSupportStatus::TargetsDeclaredSupportGap;
        proposal.support_assessment_reviewed = false;
        let report = prioritize_next_experiment(&profile, &audit, &[proposal], &value_contract(), 10);
        assert_eq!(report.disposition, ExperimentPriorityDisposition::NoBeneficialOptionalProposal);
    }
}

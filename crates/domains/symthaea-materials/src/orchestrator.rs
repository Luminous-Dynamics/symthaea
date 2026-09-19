// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Human-authorized, dry-run materials discovery orchestration.
//!
//! Proposal, authorization, execution, evidence, and authority are separate states.
//! The initial implementation can plan and budget physical actions but deliberately
//! exposes no enabled physical executor.

use crate::conditioned_property::PropertyArtifactRef;
use crate::discovery_campaign::CampaignBudgets;
use crate::multi_fidelity::EvaluationResourceCost;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const SHA256_HEX_LEN: usize = 64;

/// Broad action class proposed by the discovery loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryActionClass {
    /// Purely computational/data evaluation.
    ComputationalEvaluation,
    /// Proposal for a real physical experiment or synthesis action.
    PhysicalExperiment,
}

/// Acquisition strategy that motivated a proposed next action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcquisitionStrategy {
    /// Prefer high predictive uncertainty.
    UncertaintySampling,
    /// Expected objective improvement.
    ExpectedImprovement,
    /// Expected information gain.
    InformationGain,
    /// Expand the current Pareto frontier.
    ParetoFrontExpansion,
    /// Select a diverse batch of near-optimal candidates.
    DiverseBatch,
    /// Distinguish competing hypotheses/models.
    HypothesisDiscrimination,
    /// Replicate an existing finding.
    Replication,
    /// Resolve contradictory evidence.
    ContradictionResolution,
    /// Explicit other strategy.
    Other(String),
}

/// One proposed next evaluation/experiment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscoveryActionProposal {
    /// Deterministic exact action identity.
    pub proposal_id: String,
    /// Exact MAT-009 campaign identity.
    pub campaign_identity: String,
    /// Exact MAT-007 subject identity.
    pub subject_identity: String,
    /// Action class.
    pub action_class: DiscoveryActionClass,
    /// Evaluator/executor/protocol family identifier.
    pub action_id: String,
    /// Exact input deck/protocol/workflow artifact.
    pub action_artifact: PropertyArtifactRef,
    /// Acquisition strategy.
    pub strategy: AcquisitionStrategy,
    /// Optional non-authoritative acquisition priority.
    pub acquisition_priority: Option<f64>,
    /// Optional estimated information gain/decision value.
    pub expected_information_gain: Option<f64>,
    /// Diversity key used to avoid proposing a batch of near-duplicates.
    pub diversity_key: String,
    /// Human-readable scientific rationale.
    pub rationale: String,
    /// Required material/equipment/resource descriptors.
    pub requirements: Vec<String>,
    /// Known hazard/control descriptors for review; not execution instructions.
    pub hazard_review_items: Vec<String>,
    /// Lower-risk/lower-cost alternatives considered.
    pub alternatives_considered: Vec<String>,
    /// Estimated resource consumption.
    pub estimated_cost: EvaluationResourceCost,
    /// Monotonic proposal schema/version.
    pub proposal_version: u32,
}

impl DiscoveryActionProposal {
    /// Construct and derive the exact proposal identity.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        campaign_identity: String,
        subject_identity: String,
        action_class: DiscoveryActionClass,
        action_id: String,
        action_artifact: PropertyArtifactRef,
        strategy: AcquisitionStrategy,
        acquisition_priority: Option<f64>,
        expected_information_gain: Option<f64>,
        diversity_key: String,
        rationale: String,
        requirements: Vec<String>,
        hazard_review_items: Vec<String>,
        alternatives_considered: Vec<String>,
        estimated_cost: EvaluationResourceCost,
        proposal_version: u32,
    ) -> Result<Self, OrchestratorError> {
        let mut proposal = Self {
            proposal_id: String::new(),
            campaign_identity,
            subject_identity,
            action_class,
            action_id,
            action_artifact,
            strategy,
            acquisition_priority,
            expected_information_gain,
            diversity_key,
            rationale,
            requirements,
            hazard_review_items,
            alternatives_considered,
            estimated_cost,
            proposal_version,
        };
        proposal.validate_without_id()?;
        proposal.proposal_id = proposal.derived_identity();
        Ok(proposal)
    }

    /// Validate stored proposal identity and fields.
    pub fn validate(&self) -> Result<(), OrchestratorError> {
        self.validate_without_id()?;
        if self.proposal_id != self.derived_identity() {
            return Err(OrchestratorError::ProposalIdentityMismatch);
        }
        Ok(())
    }

    /// Whether explicit human authorization is required before execution.
    pub fn requires_human_authorization(&self) -> bool {
        self.action_class == DiscoveryActionClass::PhysicalExperiment
    }

    fn validate_without_id(&self) -> Result<(), OrchestratorError> {
        nonempty("campaign_identity", &self.campaign_identity)?;
        nonempty("subject_identity", &self.subject_identity)?;
        nonempty("action_id", &self.action_id)?;
        artifact(&self.action_artifact)?;
        nonempty("diversity_key", &self.diversity_key)?;
        nonempty("rationale", &self.rationale)?;
        if self.proposal_version == 0 {
            return Err(OrchestratorError::ZeroProposalVersion);
        }
        if let Some(value) = self.acquisition_priority {
            finite("acquisition_priority", value)?;
        }
        if let Some(value) = self.expected_information_gain {
            nonnegative("expected_information_gain", value)?;
        }
        validate_cost(&self.estimated_cost)?;
        unique_nonempty(&self.requirements, "requirement")?;
        unique_nonempty(&self.hazard_review_items, "hazard_review_item")?;
        unique_nonempty(&self.alternatives_considered, "alternative")?;
        Ok(())
    }

    fn derived_identity(&self) -> String {
        format!(
            "materials-action:v1|campaign={}|subject={}|class={}|action={}|artifact={}|strategy={}|version={}",
            token(&self.campaign_identity),
            token(&self.subject_identity),
            action_class_key(self.action_class),
            token(&self.action_id),
            self.action_artifact.artifact_sha256.to_ascii_lowercase(),
            strategy_key(&self.strategy),
            self.proposal_version
        )
    }
}

/// Explicit human authorization bound to one exact physical proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanAuthorization {
    /// Exact proposal identity authorized.
    pub proposal_id: String,
    /// Exact subject identity authorized.
    pub subject_identity: String,
    /// Exact protocol/input artifact digest authorized.
    pub action_artifact_sha256: String,
    /// Stable authorizer/reviewer identity.
    pub authorized_by: String,
    /// Authorization record/version.
    pub authorization_version: u32,
}

impl HumanAuthorization {
    /// Validate that this authorization applies to the exact, unchanged proposal.
    pub fn validate_for(&self, proposal: &DiscoveryActionProposal) -> Result<(), OrchestratorError> {
        proposal.validate()?;
        if !proposal.requires_human_authorization() {
            return Err(OrchestratorError::AuthorizationNotRequiredForComputationalAction);
        }
        nonempty("authorized_by", &self.authorized_by)?;
        if self.authorization_version == 0 {
            return Err(OrchestratorError::ZeroAuthorizationVersion);
        }
        sha256(&self.action_artifact_sha256)?;
        if self.proposal_id != proposal.proposal_id
            || self.subject_identity != proposal.subject_identity
            || !self
                .action_artifact_sha256
                .eq_ignore_ascii_case(&proposal.action_artifact.artifact_sha256)
        {
            return Err(OrchestratorError::AuthorizationDoesNotMatchProposal);
        }
        Ok(())
    }
}

/// Monotonic campaign resource accounting for proposed/approved actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OrchestratorBudgetLedger {
    /// Compute core-hours consumed/reserved.
    pub compute_core_hours: f64,
    /// Physical experiments consumed/reserved.
    pub physical_experiments: u32,
    /// Material mass consumed/reserved in kg.
    pub material_mass_kg: f64,
    /// Number of actions/attempts consumed/reserved.
    pub attempts: u32,
    /// Direct monetary amount when a single currency is established.
    pub direct_cost: Option<f64>,
    /// Currency paired with direct cost.
    pub currency: Option<String>,
}

impl OrchestratorBudgetLedger {
    /// Charge one proposal against a campaign budget, rejecting over-budget actions.
    pub fn charge(
        &mut self,
        budgets: &CampaignBudgets,
        proposal: &DiscoveryActionProposal,
    ) -> Result<(), OrchestratorError> {
        proposal.validate()?;
        let next_compute = self.compute_core_hours + proposal.estimated_cost.compute_core_hours;
        let next_mass = self.material_mass_kg + proposal.estimated_cost.material_mass_kg;
        let next_attempts = self.attempts.saturating_add(1);
        let physical_increment = if proposal.action_class == DiscoveryActionClass::PhysicalExperiment {
            1
        } else {
            0
        };
        let next_physical = self.physical_experiments + physical_increment;

        if budgets
            .max_compute_core_hours
            .is_some_and(|limit| next_compute > limit)
        {
            return Err(OrchestratorError::BudgetExceeded("compute_core_hours"));
        }
        if budgets
            .max_material_mass_kg
            .is_some_and(|limit| next_mass > limit)
        {
            return Err(OrchestratorError::BudgetExceeded("material_mass_kg"));
        }
        if budgets
            .max_attempts
            .is_some_and(|limit| next_attempts > limit)
        {
            return Err(OrchestratorError::BudgetExceeded("attempts"));
        }
        if budgets
            .max_physical_experiments
            .is_some_and(|limit| next_physical > limit)
        {
            return Err(OrchestratorError::BudgetExceeded("physical_experiments"));
        }

        let mut next_cost = self.direct_cost;
        let mut next_currency = self.currency.clone();
        match (
            proposal.estimated_cost.direct_cost,
            proposal.estimated_cost.currency.as_deref(),
        ) {
            (Some(value), Some(currency)) => {
                match next_currency.as_deref() {
                    None => next_currency = Some(currency.to_string()),
                    Some(existing) if existing == currency => {}
                    Some(_) => return Err(OrchestratorError::CurrencyMismatch),
                }
                next_cost = Some(next_cost.unwrap_or(0.0) + value);
                if let Some(limit) = &budgets.max_direct_cost {
                    if limit.currency != currency {
                        return Err(OrchestratorError::CurrencyMismatch);
                    }
                    if next_cost.unwrap_or(0.0) > limit.amount {
                        return Err(OrchestratorError::BudgetExceeded("direct_cost"));
                    }
                }
            }
            (None, None) => {}
            _ => return Err(OrchestratorError::IncompleteMonetaryCost),
        }

        self.compute_core_hours = next_compute;
        self.material_mass_kg = next_mass;
        self.attempts = next_attempts;
        self.physical_experiments = next_physical;
        self.direct_cost = next_cost;
        self.currency = next_currency;
        Ok(())
    }
}

/// Side-effect-free initial orchestrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct DryRunMaterialsOrchestrator;

impl DryRunMaterialsOrchestrator {
    /// Select a diverse proposal batch while preserving caller-supplied priority/order.
    ///
    /// This deliberately does not impose a universal acquisition strategy. The caller
    /// provides proposals already ordered under the campaign-selected strategy; this
    /// function only prevents duplicate diversity neighborhoods/subjects. Rejected
    /// proposals do not reserve a diversity key or subject.
    pub fn diverse_batch(
        &self,
        proposals: &[DiscoveryActionProposal],
        max_items: usize,
    ) -> Result<Vec<DiscoveryActionProposal>, OrchestratorError> {
        if max_items == 0 {
            return Err(OrchestratorError::ZeroBatchSize);
        }
        let mut diversity = HashSet::new();
        let mut subjects = HashSet::new();
        let mut selected = Vec::new();
        for proposal in proposals {
            proposal.validate()?;
            if diversity.contains(&proposal.diversity_key)
                || subjects.contains(&proposal.subject_identity)
            {
                continue;
            }
            diversity.insert(proposal.diversity_key.clone());
            subjects.insert(proposal.subject_identity.clone());
            selected.push(proposal.clone());
            if selected.len() == max_items {
                break;
            }
        }
        Ok(selected)
    }

    /// Validate authorization but refuse physical execution in this initial implementation.
    pub fn request_physical_execution(
        &self,
        proposal: &DiscoveryActionProposal,
        authorization: &HumanAuthorization,
    ) -> Result<(), OrchestratorError> {
        authorization.validate_for(proposal)?;
        Err(OrchestratorError::PhysicalExecutionDisabled)
    }
}

fn validate_cost(cost: &EvaluationResourceCost) -> Result<(), OrchestratorError> {
    nonnegative("compute_core_hours", cost.compute_core_hours)?;
    nonnegative("wall_time_hours", cost.wall_time_hours)?;
    nonnegative("material_mass_kg", cost.material_mass_kg)?;
    match (cost.direct_cost, cost.currency.as_deref()) {
        (Some(value), Some(currency)) => {
            nonnegative("direct_cost", value)?;
            nonempty("currency", currency)
        }
        (None, None) => Ok(()),
        _ => Err(OrchestratorError::IncompleteMonetaryCost),
    }
}

fn artifact(value: &PropertyArtifactRef) -> Result<(), OrchestratorError> {
    nonempty("artifact source_id", &value.source_id)?;
    sha256(&value.artifact_sha256)
}

fn sha256(value: &str) -> Result<(), OrchestratorError> {
    if value.len() != SHA256_HEX_LEN || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Err(OrchestratorError::InvalidSha256)
    } else {
        Ok(())
    }
}

fn unique_nonempty(values: &[String], field: &'static str) -> Result<(), OrchestratorError> {
    let mut seen = HashSet::new();
    for value in values {
        nonempty(field, value)?;
        if !seen.insert(value.as_str()) {
            return Err(OrchestratorError::DuplicateStringValue {
                field,
                value: value.clone(),
            });
        }
    }
    Ok(())
}

fn finite(field: &'static str, value: f64) -> Result<(), OrchestratorError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OrchestratorError::NonFiniteValue { field, value })
    }
}

fn nonnegative(field: &'static str, value: f64) -> Result<(), OrchestratorError> {
    finite(field, value)?;
    if value < 0.0 {
        Err(OrchestratorError::NegativeValue { field, value })
    } else {
        Ok(())
    }
}

fn nonempty(field: &'static str, value: &str) -> Result<(), OrchestratorError> {
    if value.trim().is_empty() {
        Err(OrchestratorError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn action_class_key(value: DiscoveryActionClass) -> &'static str {
    match value {
        DiscoveryActionClass::ComputationalEvaluation => "computational",
        DiscoveryActionClass::PhysicalExperiment => "physical",
    }
}

fn strategy_key(value: &AcquisitionStrategy) -> String {
    match value {
        AcquisitionStrategy::UncertaintySampling => "uncertainty".to_string(),
        AcquisitionStrategy::ExpectedImprovement => "expected-improvement".to_string(),
        AcquisitionStrategy::InformationGain => "information-gain".to_string(),
        AcquisitionStrategy::ParetoFrontExpansion => "pareto-expansion".to_string(),
        AcquisitionStrategy::DiverseBatch => "diverse-batch".to_string(),
        AcquisitionStrategy::HypothesisDiscrimination => "hypothesis-discrimination".to_string(),
        AcquisitionStrategy::Replication => "replication".to_string(),
        AcquisitionStrategy::ContradictionResolution => "contradiction-resolution".to_string(),
        AcquisitionStrategy::Other(value) => format!("other-{}", token(value)),
    }
}

fn token(value: &str) -> String {
    let mut output = String::new();
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.') {
            output.push(byte as char);
        } else {
            output.push_str(&format!("%{byte:02X}"));
        }
    }
    output
}

/// Discovery-orchestrator validation/authorization/budget failure.
#[derive(Debug, Clone, PartialEq)]
pub enum OrchestratorError {
    /// Required text field was empty.
    EmptyField(&'static str),
    /// Numeric value was NaN/infinite.
    NonFiniteValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Numeric value was negative where non-negative was required.
    NegativeValue {
        /// Field name.
        field: &'static str,
        /// Invalid value.
        value: f64,
    },
    /// Artifact SHA-256 was malformed.
    InvalidSha256,
    /// Direct cost and currency were not supplied together.
    IncompleteMonetaryCost,
    /// Duplicate set-like string value.
    DuplicateStringValue {
        /// Field name.
        field: &'static str,
        /// Duplicate value.
        value: String,
    },
    /// Proposal version cannot be zero.
    ZeroProposalVersion,
    /// Stored proposal ID did not match the action definition.
    ProposalIdentityMismatch,
    /// Authorization version cannot be zero.
    ZeroAuthorizationVersion,
    /// Authorization does not bind the exact unchanged proposal.
    AuthorizationDoesNotMatchProposal,
    /// Computational action does not need a physical-action authorization token.
    AuthorizationNotRequiredForComputationalAction,
    /// Initial orchestrator has no enabled physical executor.
    PhysicalExecutionDisabled,
    /// Batch size cannot be zero.
    ZeroBatchSize,
    /// Campaign budget would be exceeded.
    BudgetExceeded(&'static str),
    /// Monetary currencies did not match.
    CurrencyMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery_campaign::MonetaryBudget;

    const A64: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B64: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn artifact_ref(hash: &str) -> PropertyArtifactRef {
        PropertyArtifactRef {
            source_id: "fixture-action".to_string(),
            artifact_sha256: hash.to_string(),
        }
    }

    fn physical(subject: &str, diversity: &str, artifact_hash: &str) -> DiscoveryActionProposal {
        DiscoveryActionProposal::new(
            "campaign-v1".to_string(),
            subject.to_string(),
            DiscoveryActionClass::PhysicalExperiment,
            "cement-compression-test-proposal".to_string(),
            artifact_ref(artifact_hash),
            AcquisitionStrategy::InformationGain,
            Some(0.7),
            Some(0.4),
            diversity.to_string(),
            "resolve strength/process uncertainty".to_string(),
            vec!["reviewed-test-rig".to_string()],
            vec!["campaign-specific hazard review required".to_string()],
            vec!["lower-cost simulation first".to_string()],
            EvaluationResourceCost {
                compute_core_hours: 0.0,
                wall_time_hours: 2.0,
                material_mass_kg: 0.01,
                direct_cost: Some(20.0),
                currency: Some("USD".to_string()),
            },
            1,
        )
        .unwrap()
    }

    fn authorize(proposal: &DiscoveryActionProposal) -> HumanAuthorization {
        HumanAuthorization {
            proposal_id: proposal.proposal_id.clone(),
            subject_identity: proposal.subject_identity.clone(),
            action_artifact_sha256: proposal.action_artifact.artifact_sha256.clone(),
            authorized_by: "human-reviewer-fixture".to_string(),
            authorization_version: 1,
        }
    }

    #[test]
    fn changed_protocol_invalidates_authorization() {
        let proposal = physical("subject-a", "cell-a", A64);
        let authorization = authorize(&proposal);
        let changed = physical("subject-a", "cell-a", B64);
        assert_eq!(
            authorization.validate_for(&changed),
            Err(OrchestratorError::AuthorizationDoesNotMatchProposal)
        );
    }

    #[test]
    fn changed_subject_invalidates_authorization() {
        let proposal = physical("subject-a", "cell-a", A64);
        let authorization = authorize(&proposal);
        let changed = physical("subject-b", "cell-a", A64);
        assert_eq!(
            authorization.validate_for(&changed),
            Err(OrchestratorError::AuthorizationDoesNotMatchProposal)
        );
    }

    #[test]
    fn physical_execution_is_disabled_even_with_valid_authorization() {
        let proposal = physical("subject-a", "cell-a", A64);
        let authorization = authorize(&proposal);
        let orchestrator = DryRunMaterialsOrchestrator;
        assert_eq!(
            orchestrator.request_physical_execution(&proposal, &authorization),
            Err(OrchestratorError::PhysicalExecutionDisabled)
        );
    }

    #[test]
    fn diverse_batch_reserves_only_selected_neighborhoods_and_subjects() {
        let proposals = vec![
            physical("subject-a", "cell-1", A64),
            physical("subject-b", "cell-1", B64),
            physical("subject-a", "cell-2", B64),
            physical("subject-d", "cell-2", A64),
            physical("subject-c", "cell-3", B64),
        ];
        let selected = DryRunMaterialsOrchestrator
            .diverse_batch(&proposals, 3)
            .unwrap();
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].subject_identity, "subject-a");
        assert_eq!(selected[1].subject_identity, "subject-d");
        assert_eq!(selected[2].subject_identity, "subject-c");
    }

    #[test]
    fn budget_accounting_is_monotonic_and_rejects_overrun() {
        let budgets = CampaignBudgets {
            max_compute_core_hours: Some(10.0),
            max_physical_experiments: Some(1),
            max_material_mass_kg: Some(0.02),
            max_direct_cost: Some(MonetaryBudget {
                amount: 25.0,
                currency: "USD".to_string(),
            }),
            max_attempts: Some(2),
        };
        let mut ledger = OrchestratorBudgetLedger::default();
        let first = physical("subject-a", "cell-a", A64);
        ledger.charge(&budgets, &first).unwrap();
        assert_eq!(ledger.physical_experiments, 1);
        assert_eq!(ledger.attempts, 1);
        let second = physical("subject-b", "cell-b", B64);
        assert_eq!(
            ledger.charge(&budgets, &second),
            Err(OrchestratorError::BudgetExceeded("physical_experiments"))
        );
        assert_eq!(ledger.physical_experiments, 1);
        assert_eq!(ledger.attempts, 1);
    }

    #[test]
    fn contradiction_resolution_is_a_strategy_not_evidence() {
        let mut proposal = physical("subject-a", "cell-a", A64);
        proposal.strategy = AcquisitionStrategy::ContradictionResolution;
        proposal.proposal_id = proposal.derived_identity();
        assert!(proposal.validate().is_ok());
        assert_eq!(proposal.strategy, AcquisitionStrategy::ContradictionResolution);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed contract for prospective scientific-decision episodes.
//!
//! Schema/validation prototype only: no fixture reads, action execution, forecast scoring,
//! ledger writes or promotion authority. Candidate-visible proposal types and evaluator-owned
//! resolution types are deliberately separate.
//!
//! Rust types are not a security boundary against hostile candidates. Keep evaluator data in a
//! separate process or restricted custody domain.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u16 = 1;
pub const PROBABILITY_SCALE: u32 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionPhase {
    QuestionSelection,
    HypothesisFormation,
    ExperimentalDesign,
    ProspectivePrediction,
    ExecutionAndObservation,
    Interpretation,
    ResearchPolicyUpdate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceReference {
    pub evidence_id: String,
    pub content_digest: String,
    pub observed_at_ms: u64,
    pub available_at_ms: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceVector {
    /// Some(0) is measured zero; None is unmeasured or unreported.
    pub inference_tokens: Option<u64>,
    pub cpu_millis: Option<u64>,
    pub accelerator_millis: Option<u64>,
    pub wall_clock_millis: Option<u64>,
    pub expert_minutes: Option<u64>,
    /// USD multiplied by one million.
    pub data_cost_microusd: Option<u64>,
}
impl ResourceVector {
    fn is_complete(&self) -> bool {
        self.inference_tokens.is_some()
            && self.cpu_millis.is_some()
            && self.accelerator_millis.is_some()
            && self.wall_clock_millis.is_some()
            && self.expert_minutes.is_some()
            && self.data_cost_microusd.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostCompleteness {
    Complete,
    Partial,
    Unknown,
}

/// The frozen downstream decision/loss contract. This is not an information-gain score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionValueMode {
    BayesianExpectedUtility,
    PriorFreeRobustMinimax,
    DirectOutcomeMetric,
    NotQuantified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionObjectiveReference {
    pub objective_digest: String,
    pub mode: DecisionValueMode,
    pub utility_unit: Option<String>,
    pub loss_table_digest: Option<String>,
}

impl DecisionObjectiveReference {
    fn validate(&self) -> Result<(), String> {
        validate_sha256("decision objective digest", &self.objective_digest)?;
        match self.mode {
            DecisionValueMode::BayesianExpectedUtility
            | DecisionValueMode::PriorFreeRobustMinimax => {
                require_nonempty(
                    "decision utility unit",
                    self.utility_unit.as_deref().unwrap_or_default(),
                )?;
                validate_sha256(
                    "decision loss-table digest",
                    self.loss_table_digest.as_deref().unwrap_or_default(),
                )?;
            }
            DecisionValueMode::DirectOutcomeMetric => {
                require_nonempty(
                    "decision metric unit",
                    self.utility_unit.as_deref().unwrap_or_default(),
                )?;
            }
            DecisionValueMode::NotQuantified => {
                if self.utility_unit.is_some() || self.loss_table_digest.is_some() {
                    return Err("NotQuantified cannot carry utility units or a loss table".into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermittedActionOfferRef {
    /// Upstream AgentView offer ID from #11015; this contract does not mint offers.
    pub offer_id: String,
    /// Exact payload digest on the upstream offer.
    pub action_payload_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateDecisionContext {
    pub schema_version: u16,
    pub episode_id: String,
    pub parent_episode_id: Option<String>,
    pub task_family_id: String,
    pub decision_index: u64,
    pub decision_time_ms: u64,
    pub deadline_ms: Option<u64>,
    pub phase: DecisionPhase,
    pub decision_objective: DecisionObjectiveReference,
    /// Exact serialized ResearchState digest from #13665; reference only, not a second state store.
    pub research_state_sha256: String,
    /// Exact visible evidence-prefix digest from #13665/#11015.
    pub evidence_prefix_sha256: String,
    /// Exact upstream AgentView permitted-action-offer-list digest from #11015.
    pub permitted_action_offer_list_sha256: String,
    /// Digest of this episode envelope, including the upstream identities above.
    pub context_digest: String,
    /// Decision-local references only; artifact/evidence custody stays with its upstream owner.
    pub evidence_available_at_decision: Vec<EvidenceReference>,
    /// Exact upstream offer IDs and their payload digests, projected from #11015 AgentView.
    pub permitted_action_offers: Vec<PermittedActionOfferRef>,
    pub estimated_budget: ResourceVector,
}
impl CandidateDecisionContext {
    pub fn validate(&self) -> Result<(), String> {
        require_nonempty("episode_id", &self.episode_id)?;
        require_nonempty("task_family_id", &self.task_family_id)?;
        validate_sha256("context_digest", &self.context_digest)?;
        validate_sha256("research_state_sha256", &self.research_state_sha256)?;
        validate_sha256("evidence_prefix_sha256", &self.evidence_prefix_sha256)?;
        validate_sha256(
            "permitted_action_offer_list_sha256",
            &self.permitted_action_offer_list_sha256,
        )?;
        self.decision_objective.validate()?;
        if self.schema_version != SCHEMA_VERSION {
            return Err("unsupported schema version".into());
        }
        if self.permitted_action_offers.is_empty() {
            return Err("permitted action offer set is empty".into());
        }
        let mut offer_ids = Vec::with_capacity(self.permitted_action_offers.len());
        for offer in &self.permitted_action_offers {
            require_nonempty("upstream offer_id", &offer.offer_id)?;
            validate_sha256("upstream offer payload digest", &offer.action_payload_sha256)?;
            offer_ids.push(offer.offer_id.clone());
        }
        ensure_unique("permitted_action_offer_ids", &offer_ids)?;
        if self.deadline_ms.is_some_and(|d| d < self.decision_time_ms) {
            return Err("deadline precedes decision time".into());
        }
        for e in &self.evidence_available_at_decision {
            require_nonempty("evidence_id", &e.evidence_id)?;
            validate_sha256("evidence content_digest", &e.content_digest)?;
            if e.observed_at_ms > e.available_at_ms {
                return Err("evidence availability precedes observation".into());
            }
            if e.available_at_ms > self.decision_time_ms {
                return Err(format!("look-ahead leakage in evidence {}", e.evidence_id));
            }
        }
        Ok(())
    }
    pub fn offered_payload_digest(&self, offer_id: &str) -> Option<&str> {
        self.permitted_action_offers
            .iter()
            .find(|offer| offer.offer_id == offer_id)
            .map(|offer| offer.action_payload_sha256.as_str())
    }

    pub fn permits(&self, offer_id: &str) -> bool {
        self.offered_payload_digest(offer_id).is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PredictedOutcome {
    pub outcome_label: String,
    pub probability_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionProbability {
    pub action_id: String,
    /// Digest of the exact action parameters/payload assigned under this identity.
    pub action_payload_digest: String,
    pub probability_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssignmentMechanism {
    Randomized {
        action_probabilities: Vec<ActionProbability>,
        protocol_ref: String,
    },
    Deterministic {
        policy_digest: String,
    },
    Observational {
        selection_process_ref: String,
    },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDesign {
    RandomizedControlled,
    QuasiExperimental,
    Observational,
    MechanisticModel,
    NotClaimed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionProposal {
    pub episode_id: String,
    pub policy_id: String,
    pub policy_version: String,
    pub selected_action_id: String,
    /// Digest of the exact selected payload (for example, an ordered AS-Bench design batch).
    pub selected_action_payload_digest: String,
    pub rationale_digest: String,
    pub assumption_digests: Vec<String>,
    pub predicted_outcomes: Vec<PredictedOutcome>,
    pub assignment_mechanism: AssignmentMechanism,
    pub causal_design: CausalDesign,
    pub identification_assumption_digests: Vec<String>,
}
impl DecisionProposal {
    pub fn validate_against(&self, c: &CandidateDecisionContext) -> Result<(), String> {
        c.validate()?;
        require_nonempty("proposal episode_id", &self.episode_id)?;
        require_nonempty("policy_id", &self.policy_id)?;
        require_nonempty("policy_version", &self.policy_version)?;
        require_nonempty("selected_action_id", &self.selected_action_id)?;
        validate_sha256(
            "selected_action_payload_digest",
            &self.selected_action_payload_digest,
        )?;
        validate_sha256("rationale_digest", &self.rationale_digest)?;
        if self.episode_id != c.episode_id {
            return Err("proposal episode mismatch".into());
        }
        let offered_payload = c
            .offered_payload_digest(&self.selected_action_id)
            .ok_or_else(|| "selected action outside permitted offer set".to_string())?;
        if offered_payload != self.selected_action_payload_digest {
            return Err("selected action payload differs from upstream permitted offer".into());
        }
        if self.predicted_outcomes.is_empty() {
            return Err("predicted outcome distribution is empty".into());
        }
        let mut labels = Vec::new();
        let mut sum = 0_u64;
        for item in &self.predicted_outcomes {
            require_nonempty("outcome label", &item.outcome_label)?;
            if item.probability_micros > PROBABILITY_SCALE {
                return Err("probability exceeds scale".into());
            }
            labels.push(item.outcome_label.clone());
            sum += u64::from(item.probability_micros);
        }
        ensure_unique("outcome labels", &labels)?;
        if sum != u64::from(PROBABILITY_SCALE) {
            return Err("probabilities must sum to 1,000,000".into());
        }
        match &self.assignment_mechanism {
            AssignmentMechanism::Randomized {
                action_probabilities,
                protocol_ref,
            } => {
                require_nonempty("randomization protocol reference", protocol_ref)?;
                if action_probabilities.len() != c.permitted_action_offers.len() {
                    return Err(
                        "randomized policy must log probabilities for every permitted action offer"
                            .into(),
                    );
                }
                let mut action_ids = Vec::with_capacity(action_probabilities.len());
                let mut probability_sum = 0_u64;
                let mut selected_probability = 0_u32;
                for entry in action_probabilities {
                    require_nonempty("randomized action_id", &entry.action_id)?;
                    validate_sha256(
                        "randomized action_payload_digest",
                        &entry.action_payload_digest,
                    )?;
                    let offered_payload = c
                        .offered_payload_digest(&entry.action_id)
                        .ok_or_else(|| {
                            "randomization distribution contains an action outside the permitted offers"
                                .to_string()
                        })?;
                    if offered_payload != entry.action_payload_digest {
                        return Err(
                            "randomization payload differs from upstream permitted offer".into(),
                        );
                    }
                    if entry.probability_micros > PROBABILITY_SCALE {
                        return Err("action propensity exceeds probability scale".into());
                    }
                    if entry.action_id == self.selected_action_id
                        && entry.action_payload_digest == self.selected_action_payload_digest
                    {
                        selected_probability = entry.probability_micros;
                    }
                    action_ids.push(entry.action_id.clone());
                    probability_sum += u64::from(entry.probability_micros);
                }
                ensure_unique("randomized action IDs", &action_ids)?;
                if probability_sum != u64::from(PROBABILITY_SCALE) {
                    return Err("action propensities must sum to 1,000,000".into());
                }
                if selected_probability == 0 {
                    return Err("selected action must have nonzero randomized propensity".into());
                }
            }
            AssignmentMechanism::Deterministic { policy_digest } => {
                validate_sha256("assignment policy digest", policy_digest)?
            }
            AssignmentMechanism::Observational {
                selection_process_ref,
            } => require_nonempty("selection process reference", selection_process_ref)?,
            AssignmentMechanism::Unknown => {}
        }
        ensure_sha256_items("assumption_digests", &self.assumption_digests)?;
        ensure_sha256_items(
            "identification_assumption_digests",
            &self.identification_assumption_digests,
        )?;
        if self.causal_design != CausalDesign::NotClaimed
            && self.identification_assumption_digests.is_empty()
        {
            return Err("causal claim requires identification-assumption digests".into());
        }
        if self.causal_design == CausalDesign::RandomizedControlled
            && !matches!(
                &self.assignment_mechanism,
                AssignmentMechanism::Randomized { .. }
            )
        {
            return Err("randomized-controlled claim requires randomized assignment".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    NotStarted,
    Started,
    Completed,
    Failed,
    TimedOut,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    AwaitingObservation,
    ImmediateObserved,
    DelayedObserved,
    CensoredAtDeadline,
    UnsupportedCounterfactual,
    Indeterminate,
    InvalidExecution,
    NotRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSupport {
    RecordedAction,
    NoRecordedAction,
}

/// Evaluator-owned record. Never serialize this into candidate-process input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorResolution {
    pub episode_id: String,
    pub execution_status: ExecutionStatus,
    pub outcome_status: OutcomeStatus,
    pub action_support: ActionSupport,
    pub recorded_action_id: Option<String>,
    pub recorded_action_payload_digest: Option<String>,
    /// Upstream executor/Lab job receipt reference; this module never creates the receipt.
    pub execution_receipt_ref: Option<String>,
    pub outcome_label: Option<String>,
    pub evaluator_evidence: Vec<EvidenceReference>,
    pub finalized_at_ms: Option<u64>,
    pub actual_cost: ResourceVector,
    pub cost_completeness: CostCompleteness,
    pub independent_adjudication_ref: Option<String>,
}
impl EvaluatorResolution {
    pub fn validate_against(
        &self,
        c: &CandidateDecisionContext,
        p: &DecisionProposal,
    ) -> Result<(), String> {
        c.validate()?;
        p.validate_against(c)?;
        require_nonempty("resolution episode_id", &self.episode_id)?;
        if self.episode_id != c.episode_id {
            return Err("resolution episode mismatch".into());
        }
        if self.finalized_at_ms.is_some_and(|t| t < c.decision_time_ms) {
            return Err("resolution finalized before decision".into());
        }
        for e in &self.evaluator_evidence {
            require_nonempty("evaluator evidence_id", &e.evidence_id)?;
            validate_sha256("evaluator evidence digest", &e.content_digest)?;
            if e.observed_at_ms > e.available_at_ms {
                return Err("evaluator evidence availability precedes observation".into());
            }
            if self.finalized_at_ms.is_some_and(|t| e.available_at_ms > t) {
                return Err("evaluator evidence arrived after finalization".into());
            }
        }
        if self.cost_completeness == CostCompleteness::Complete && !self.actual_cost.is_complete() {
            return Err(
                "complete cost requires every resource dimension, including measured zero".into(),
            );
        }
        if let Some(r) = &self.independent_adjudication_ref {
            require_nonempty("adjudication reference", r)?;
        }
        if let Some(r) = &self.execution_receipt_ref {
            require_nonempty("execution receipt reference", r)?;
        }

        match self.execution_status {
            ExecutionStatus::NotStarted
                if !matches!(
                    self.outcome_status,
                    OutcomeStatus::NotRun | OutcomeStatus::UnsupportedCounterfactual
                ) =>
            {
                return Err("not-started execution must be not-run or unsupported".into());
            }
            ExecutionStatus::Started
                if self.outcome_status != OutcomeStatus::AwaitingObservation =>
            {
                return Err("started execution must await observation".into());
            }
            ExecutionStatus::Failed | ExecutionStatus::TimedOut | ExecutionStatus::Aborted
                if self.outcome_status != OutcomeStatus::InvalidExecution =>
            {
                return Err("failed/timed-out/aborted execution must be invalid_execution".into());
            }
            ExecutionStatus::Completed
                if matches!(
                    self.outcome_status,
                    OutcomeStatus::AwaitingObservation
                        | OutcomeStatus::NotRun
                        | OutcomeStatus::UnsupportedCounterfactual
                ) =>
            {
                return Err("completed execution has incompatible outcome status".into());
            }
            _ => {}
        }

        match self.outcome_status {
            OutcomeStatus::AwaitingObservation => {
                self.require_action(&p.selected_action_id, &p.selected_action_payload_digest)?;
                if self.outcome_label.is_some() || self.finalized_at_ms.is_some() {
                    return Err("awaiting outcome cannot have final label/time".into());
                }
            }
            OutcomeStatus::ImmediateObserved | OutcomeStatus::DelayedObserved => {
                self.require_action(&p.selected_action_id, &p.selected_action_payload_digest)?;
                require_nonempty(
                    "observed outcome label",
                    self.outcome_label.as_deref().unwrap_or_default(),
                )?;
                if self.evaluator_evidence.is_empty() || self.finalized_at_ms.is_none() {
                    return Err("observed outcome requires evidence and finalization time".into());
                }
            }
            OutcomeStatus::CensoredAtDeadline | OutcomeStatus::Indeterminate => {
                self.require_action(&p.selected_action_id, &p.selected_action_payload_digest)?;
                if self.outcome_label.is_some()
                    || self.evaluator_evidence.is_empty()
                    || self.finalized_at_ms.is_none()
                {
                    return Err("censored/indeterminate outcome requires evidence, no label and finalization time".into());
                }
            }
            OutcomeStatus::UnsupportedCounterfactual => {
                if self.execution_status != ExecutionStatus::NotStarted
                    || self.action_support != ActionSupport::NoRecordedAction
                    || self.recorded_action_id.is_some()
                    || self.recorded_action_payload_digest.is_some()
                    || self.execution_receipt_ref.is_some()
                    || self.outcome_label.is_some()
                    || !self.evaluator_evidence.is_empty()
                {
                    return Err(
                        "unsupported counterfactual cannot carry execution receipt or fabricated outcome"
                            .into(),
                    );
                }
            }
            OutcomeStatus::NotRun => {
                if self.action_support != ActionSupport::NoRecordedAction
                    || self.recorded_action_id.is_some()
                    || self.recorded_action_payload_digest.is_some()
                    || self.execution_receipt_ref.is_some()
                    || self.outcome_label.is_some()
                    || !self.evaluator_evidence.is_empty()
                {
                    return Err("not-run cannot carry action, execution receipt or outcome".into());
                }
            }
            OutcomeStatus::InvalidExecution => {
                if self.outcome_label.is_some() {
                    return Err("invalid execution cannot carry scientific label".into());
                }
                if self.action_support == ActionSupport::RecordedAction {
                    self.require_action(&p.selected_action_id, &p.selected_action_payload_digest)?;
                } else if self.recorded_action_id.is_some()
                    || self.recorded_action_payload_digest.is_some()
                    || self.execution_receipt_ref.is_some()
                {
                    return Err(
                        "unrecorded action cannot have an action ID, payload digest or receipt"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
    fn require_action(
        &self,
        expected_id: &str,
        expected_payload_digest: &str,
    ) -> Result<(), String> {
        if self.action_support != ActionSupport::RecordedAction {
            return Err("recorded action support required".into());
        }
        let actual_id = self
            .recorded_action_id
            .as_deref()
            .ok_or_else(|| "recorded action ID required".to_string())?;
        let actual_payload = self
            .recorded_action_payload_digest
            .as_deref()
            .ok_or_else(|| "recorded action payload digest required".to_string())?;
        if actual_id != expected_id || actual_payload != expected_payload_digest {
            return Err("recorded action identity/payload differs from proposal".into());
        }
        let receipt = self
            .execution_receipt_ref
            .as_deref()
            .ok_or_else(|| "recorded action requires an upstream execution receipt reference".to_string())?;
        require_nonempty("execution receipt reference", receipt)?;
        Ok(())
    }
}

fn validate_sha256(field: &str, value: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(format!("{field} must be lowercase SHA-256 hex"));
    }
    Ok(())
}

fn require_nonempty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must not be empty"))
    } else {
        Ok(())
    }
}
fn ensure_unique(field: &str, values: &[String]) -> Result<(), String> {
    for (i, value) in values.iter().enumerate() {
        if values[..i].iter().any(|prior| prior == value) {
            return Err(format!("{field} contains duplicate {value}"));
        }
    }
    Ok(())
}
fn ensure_sha256_items(field: &str, values: &[String]) -> Result<(), String> {
    for value in values {
        validate_sha256(field, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> CandidateDecisionContext {
        CandidateDecisionContext {
            schema_version: SCHEMA_VERSION,
            episode_id: "e1".into(),
            parent_episode_id: None,
            task_family_id: "pilot".into(),
            decision_index: 0,
            decision_time_ms: 100,
            deadline_ms: Some(1_000),
            phase: DecisionPhase::ExperimentalDesign,
            decision_objective: DecisionObjectiveReference {
                objective_digest:
                    "1111111111111111111111111111111111111111111111111111111111111111".into(),
                mode: DecisionValueMode::PriorFreeRobustMinimax,
                utility_unit: Some("normalized_loss_units".into()),
                loss_table_digest: Some(
                    "2222222222222222222222222222222222222222222222222222222222222222".into(),
                ),
            },
            research_state_sha256:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            evidence_prefix_sha256:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
            permitted_action_offer_list_sha256:
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc".into(),
            context_digest: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
                .into(),
            evidence_available_at_decision: vec![EvidenceReference {
                evidence_id: "obs".into(),
                content_digest: "3333333333333333333333333333333333333333333333333333333333333333"
                    .into(),
                observed_at_ms: 80,
                available_at_ms: 90,
            }],
            permitted_action_offers: vec![
                PermittedActionOfferRef {
                    offer_id: "a".into(),
                    action_payload_sha256:
                        "6666666666666666666666666666666666666666666666666666666666666666".into(),
                },
                PermittedActionOfferRef {
                    offer_id: "b".into(),
                    action_payload_sha256:
                        "7777777777777777777777777777777777777777777777777777777777777777".into(),
                },
            ],
            estimated_budget: ResourceVector::default(),
        }
    }
    fn proposal(c: &CandidateDecisionContext) -> DecisionProposal {
        DecisionProposal {
            episode_id: c.episode_id.clone(),
            policy_id: "fixed".into(),
            policy_version: "1".into(),
            selected_action_id: "a".into(),
            selected_action_payload_digest:
                "6666666666666666666666666666666666666666666666666666666666666666".into(),
            rationale_digest: "4444444444444444444444444444444444444444444444444444444444444444"
                .into(),
            assumption_digests: vec![],
            predicted_outcomes: vec![
                PredictedOutcome {
                    outcome_label: "yes".into(),
                    probability_micros: 400_000,
                },
                PredictedOutcome {
                    outcome_label: "no".into(),
                    probability_micros: 600_000,
                },
            ],
            assignment_mechanism: AssignmentMechanism::Randomized {
                action_probabilities: vec![
                    ActionProbability {
                        action_id: "a".into(),
                        action_payload_digest:
                            "6666666666666666666666666666666666666666666666666666666666666666"
                                .into(),
                        probability_micros: 500_000,
                    },
                    ActionProbability {
                        action_id: "b".into(),
                        action_payload_digest:
                            "7777777777777777777777777777777777777777777777777777777777777777"
                                .into(),
                        probability_micros: 500_000,
                    },
                ],
                protocol_ref: "random-protocol".into(),
            },
            causal_design: CausalDesign::RandomizedControlled,
            identification_assumption_digests: vec![
                "5555555555555555555555555555555555555555555555555555555555555555".into(),
            ],
        }
    }
    fn complete_cost() -> ResourceVector {
        ResourceVector {
            inference_tokens: Some(0),
            cpu_millis: Some(1),
            accelerator_millis: Some(0),
            wall_clock_millis: Some(1),
            expert_minutes: Some(0),
            data_cost_microusd: Some(0),
        }
    }
    fn resolution(c: &CandidateDecisionContext) -> EvaluatorResolution {
        EvaluatorResolution {
            episode_id: c.episode_id.clone(),
            execution_status: ExecutionStatus::Completed,
            outcome_status: OutcomeStatus::DelayedObserved,
            action_support: ActionSupport::RecordedAction,
            recorded_action_id: Some("a".into()),
            recorded_action_payload_digest: Some(
                "6666666666666666666666666666666666666666666666666666666666666666".into(),
            ),
            execution_receipt_ref: Some("lab-job-0001".into()),
            outcome_label: Some("yes".into()),
            evaluator_evidence: vec![EvidenceReference {
                evidence_id: "result".into(),
                content_digest: "8888888888888888888888888888888888888888888888888888888888888888"
                    .into(),
                observed_at_ms: 900,
                available_at_ms: 950,
            }],
            finalized_at_ms: Some(960),
            actual_cost: complete_cost(),
            cost_completeness: CostCompleteness::Complete,
            independent_adjudication_ref: Some("independent-review".into()),
        }
    }

    #[test]
    fn requires_exact_upstream_research_state_and_offer_bindings() {
        let mut c = context();
        c.research_state_sha256 = "not-a-digest".into();
        assert!(c.validate().unwrap_err().contains("research_state_sha256"));
        c = context();
        c.evidence_prefix_sha256 = "A".repeat(64);
        assert!(c.validate().unwrap_err().contains("evidence_prefix_sha256"));
        c = context();
        c.permitted_action_offer_list_sha256 = "c".repeat(63);
        assert!(
            c.validate()
                .unwrap_err()
                .contains("permitted_action_offer_list_sha256")
        );
    }

    #[test]
    fn rejects_malformed_action_and_evidence_digests() {
        let c = context();
        let mut p = proposal(&c);
        p.selected_action_payload_digest = "payload-not-a-sha256".into();
        assert!(
            p.validate_against(&c)
                .unwrap_err()
                .contains("selected_action_payload_digest")
        );
        let mut r = resolution(&c);
        r.evaluator_evidence[0].content_digest = "not-a-hash".into();
        assert!(
            r.validate_against(&c, &proposal(&c))
                .unwrap_err()
                .contains("evaluator evidence digest")
        );
    }

    #[test]
    fn rejects_unbound_decision_value_or_loss_contract() {
        let mut c = context();
        c.decision_objective.mode = DecisionValueMode::NotQuantified;
        assert!(c.validate().unwrap_err().contains("NotQuantified"));
    }

    #[test]
    fn rejects_lookahead_evidence() {
        let mut c = context();
        c.evidence_available_at_decision[0].available_at_ms = 101;
        assert!(c.validate().unwrap_err().contains("look-ahead"));
    }
    #[test]
    fn rejects_probability_mass_error() {
        let c = context();
        let mut p = proposal(&c);
        p.predicted_outcomes[0].probability_micros = 399_999;
        assert!(p.validate_against(&c).unwrap_err().contains("sum to"));
    }
    #[test]
    fn randomized_design_requires_full_action_distribution_and_selected_action_positivity() {
        let c = context();
        let mut p = proposal(&c);
        p.assignment_mechanism = AssignmentMechanism::Randomized {
            action_probabilities: vec![ActionProbability {
                action_id: "a".into(),
                action_payload_digest:
                    "6666666666666666666666666666666666666666666666666666666666666666".into(),
                probability_micros: 1_000_000,
            }],
            protocol_ref: "random-protocol".into(),
        };
        assert!(
            p.validate_against(&c)
                .unwrap_err()
                .contains("every permitted action offer")
        );

        p.assignment_mechanism = AssignmentMechanism::Randomized {
            action_probabilities: vec![
                ActionProbability {
                    action_id: "a".into(),
                    action_payload_digest:
                        "6666666666666666666666666666666666666666666666666666666666666666".into(),
                    probability_micros: 0,
                },
                ActionProbability {
                    action_id: "b".into(),
                    action_payload_digest:
                        "7777777777777777777777777777777777777777777777777777777777777777".into(),
                    probability_micros: 1_000_000,
                },
            ],
            protocol_ref: "random-protocol".into(),
        };
        assert!(
            p.validate_against(&c)
                .unwrap_err()
                .contains("selected action must have nonzero")
        );
    }
    #[test]
    fn rejects_selected_payload_not_present_on_upstream_offer() {
        let c = context();
        let mut p = proposal(&c);
        p.selected_action_payload_digest =
            "9999999999999999999999999999999999999999999999999999999999999999".into();
        assert!(
            p.validate_against(&c)
                .unwrap_err()
                .contains("differs from upstream permitted offer")
        );
    }

    #[test]
    fn rejects_unauthorized_action() {
        let c = context();
        let mut p = proposal(&c);
        p.selected_action_id = "off-menu".into();
        assert!(p.validate_against(&c).unwrap_err().contains("outside"));
    }
    #[test]
    fn rejects_result_for_different_action_payload() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.recorded_action_payload_digest =
            Some("9999999999999999999999999999999999999999999999999999999999999999".into());
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("identity/payload differs")
        );
    }

    #[test]
    fn recorded_action_requires_upstream_execution_receipt() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.execution_receipt_ref = None;
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("upstream execution receipt reference")
        );
    }

    #[test]
    fn not_run_and_unsupported_states_cannot_carry_execution_receipts() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.execution_status = ExecutionStatus::NotStarted;
        r.outcome_status = OutcomeStatus::NotRun;
        r.action_support = ActionSupport::NoRecordedAction;
        r.recorded_action_id = None;
        r.recorded_action_payload_digest = None;
        assert!(r.validate_against(&c, &p).is_err());
        r.execution_receipt_ref = None;
        assert!(r.validate_against(&c, &p).is_ok());

        r.outcome_status = OutcomeStatus::UnsupportedCounterfactual;
        r.execution_receipt_ref = Some("lab-job-that-was-never-created".into());
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("execution receipt")
        );
    }

    #[test]
    fn rejects_fabricated_unsupported_outcome() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.execution_status = ExecutionStatus::NotStarted;
        r.outcome_status = OutcomeStatus::UnsupportedCounterfactual;
        r.action_support = ActionSupport::NoRecordedAction;
        r.recorded_action_id = None;
        r.recorded_action_payload_digest = None;
        r.execution_receipt_ref = None;
        r.outcome_label = Some("yes".into());
        r.evaluator_evidence.clear();
        r.finalized_at_ms = None;
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("unsupported counterfactual")
        );
    }
    #[test]
    fn preserves_delayed_censored_and_indeterminate_states() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        assert!(r.validate_against(&c, &p).is_ok());
        r.outcome_status = OutcomeStatus::CensoredAtDeadline;
        r.outcome_label = None;
        assert!(r.validate_against(&c, &p).is_ok());
        r.outcome_status = OutcomeStatus::Indeterminate;
        assert!(r.validate_against(&c, &p).is_ok());
    }
    #[test]
    fn complete_cost_requires_explicit_zero_or_value_for_every_dimension() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.actual_cost.accelerator_millis = None;
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("every resource dimension")
        );
    }
    #[test]
    fn execution_failure_is_not_a_scientific_negative_result() {
        let c = context();
        let p = proposal(&c);
        let mut r = resolution(&c);
        r.execution_status = ExecutionStatus::TimedOut;
        r.outcome_status = OutcomeStatus::DelayedObserved;
        assert!(
            r.validate_against(&c, &p)
                .unwrap_err()
                .contains("invalid_execution")
        );
    }
}

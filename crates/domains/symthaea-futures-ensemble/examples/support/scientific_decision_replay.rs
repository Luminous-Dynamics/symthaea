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
    QuestionSelection, HypothesisFormation, ExperimentalDesign, ProspectivePrediction,
    ExecutionAndObservation, Interpretation, ResearchPolicyUpdate,
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
        self.inference_tokens.is_some() && self.cpu_millis.is_some()
            && self.accelerator_millis.is_some() && self.wall_clock_millis.is_some()
            && self.expert_minutes.is_some() && self.data_cost_microusd.is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CostCompleteness { Complete, Partial, Unknown }

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
    pub context_digest: String,
    pub evidence_available_at_decision: Vec<EvidenceReference>,
    pub permitted_action_ids: Vec<String>,
    pub estimated_budget: ResourceVector,
}
impl CandidateDecisionContext {
    pub fn validate(&self) -> Result<(), String> {
        require_nonempty("episode_id", &self.episode_id)?;
        require_nonempty("task_family_id", &self.task_family_id)?;
        require_nonempty("context_digest", &self.context_digest)?;
        if self.schema_version != SCHEMA_VERSION { return Err("unsupported schema version".into()); }
        if self.permitted_action_ids.is_empty() { return Err("permitted action set is empty".into()); }
        ensure_unique("permitted_action_ids", &self.permitted_action_ids)?;
        if self.deadline_ms.is_some_and(|d| d < self.decision_time_ms) {
            return Err("deadline precedes decision time".into());
        }
        for e in &self.evidence_available_at_decision {
            require_nonempty("evidence_id", &e.evidence_id)?;
            require_nonempty("evidence content_digest", &e.content_digest)?;
            if e.observed_at_ms > e.available_at_ms {
                return Err("evidence availability precedes observation".into());
            }
            if e.available_at_ms > self.decision_time_ms {
                return Err(format!("look-ahead leakage in evidence {}", e.evidence_id));
            }
        }
        Ok(())
    }
    pub fn permits(&self, action: &str) -> bool {
        self.permitted_action_ids.iter().any(|id| id == action)
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
    pub probability_micros: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssignmentMechanism {
    Randomized { action_probabilities: Vec<ActionProbability>, protocol_ref: String },
    Deterministic { policy_digest: String },
    Observational { selection_process_ref: String },
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CausalDesign { RandomizedControlled, QuasiExperimental, Observational, MechanisticModel, NotClaimed }

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionProposal {
    pub episode_id: String,
    pub policy_id: String,
    pub policy_version: String,
    pub selected_action_id: String,
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
        require_nonempty("rationale_digest", &self.rationale_digest)?;
        if self.episode_id != c.episode_id { return Err("proposal episode mismatch".into()); }
        if !c.permits(&self.selected_action_id) { return Err("selected action outside permitted set".into()); }
        if self.predicted_outcomes.is_empty() { return Err("predicted outcome distribution is empty".into()); }
        let mut labels = Vec::new();
        let mut sum = 0_u64;
        for item in &self.predicted_outcomes {
            require_nonempty("outcome label", &item.outcome_label)?;
            if item.probability_micros > PROBABILITY_SCALE { return Err("probability exceeds scale".into()); }
            labels.push(item.outcome_label.clone());
            sum += u64::from(item.probability_micros);
        }
        ensure_unique("outcome labels", &labels)?;
        if sum != u64::from(PROBABILITY_SCALE) { return Err("probabilities must sum to 1,000,000".into()); }
        match &self.assignment_mechanism {
            AssignmentMechanism::Randomized { action_probabilities, protocol_ref } => {
                require_nonempty("randomization protocol reference", protocol_ref)?;
                if action_probabilities.len() != c.permitted_action_ids.len() {
                    return Err("randomized policy must log probabilities for every permitted action".into());
                }
                let mut action_ids = Vec::with_capacity(action_probabilities.len());
                let mut probability_sum = 0_u64;
                let mut selected_probability = 0_u32;
                for entry in action_probabilities {
                    require_nonempty("randomized action_id", &entry.action_id)?;
                    if !c.permits(&entry.action_id) {
                        return Err("randomization distribution contains an action outside the permitted set".into());
                    }
                    if entry.probability_micros > PROBABILITY_SCALE {
                        return Err("action propensity exceeds probability scale".into());
                    }
                    if entry.action_id == self.selected_action_id {
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
            AssignmentMechanism::Deterministic { policy_digest } => require_nonempty("assignment policy digest", policy_digest)?,
            AssignmentMechanism::Observational { selection_process_ref } => require_nonempty("selection process reference", selection_process_ref)?,
            AssignmentMechanism::Unknown => {}
        }
        ensure_nonempty_items("assumption_digests", &self.assumption_digests)?;
        ensure_nonempty_items("identification_assumption_digests", &self.identification_assumption_digests)?;
        if self.causal_design != CausalDesign::NotClaimed && self.identification_assumption_digests.is_empty() {
            return Err("causal claim requires identification-assumption digests".into());
        }
        if self.causal_design == CausalDesign::RandomizedControlled
            && !matches!(&self.assignment_mechanism, AssignmentMechanism::Randomized { .. }) {
            return Err("randomized-controlled claim requires randomized assignment".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus { NotStarted, Started, Completed, Failed, TimedOut, Aborted }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutcomeStatus {
    AwaitingObservation, ImmediateObserved, DelayedObserved, CensoredAtDeadline,
    UnsupportedCounterfactual, Indeterminate, InvalidExecution, NotRun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSupport { ObservedOutcome, NoObservedOutcome }

/// Evaluator-owned record. Never serialize this into candidate-process input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluatorResolution {
    pub episode_id: String,
    pub execution_status: ExecutionStatus,
    pub outcome_status: OutcomeStatus,
    pub action_support: ActionSupport,
    pub observed_action_id: Option<String>,
    pub outcome_label: Option<String>,
    pub evaluator_evidence: Vec<EvidenceReference>,
    pub finalized_at_ms: Option<u64>,
    pub actual_cost: ResourceVector,
    pub cost_completeness: CostCompleteness,
    pub independent_adjudication_ref: Option<String>,
}
impl EvaluatorResolution {
    pub fn validate_against(&self, c: &CandidateDecisionContext, p: &DecisionProposal) -> Result<(), String> {
        c.validate()?;
        p.validate_against(c)?;
        require_nonempty("resolution episode_id", &self.episode_id)?;
        if self.episode_id != c.episode_id { return Err("resolution episode mismatch".into()); }
        if self.finalized_at_ms.is_some_and(|t| t < c.decision_time_ms) {
            return Err("resolution finalized before decision".into());
        }
        for e in &self.evaluator_evidence {
            require_nonempty("evaluator evidence_id", &e.evidence_id)?;
            require_nonempty("evaluator evidence digest", &e.content_digest)?;
            if e.observed_at_ms > e.available_at_ms { return Err("evaluator evidence availability precedes observation".into()); }
            if self.finalized_at_ms.is_some_and(|t| e.available_at_ms > t) {
                return Err("evaluator evidence arrived after finalization".into());
            }
        }
        if self.cost_completeness == CostCompleteness::Complete && !self.actual_cost.is_complete() {
            return Err("complete cost requires every resource dimension, including measured zero".into());
        }
        if let Some(r) = &self.independent_adjudication_ref { require_nonempty("adjudication reference", r)?; }

        match self.execution_status {
            ExecutionStatus::NotStarted if !matches!(self.outcome_status, OutcomeStatus::NotRun | OutcomeStatus::UnsupportedCounterfactual) =>
                return Err("not-started execution must be not-run or unsupported".into()),
            ExecutionStatus::Started if self.outcome_status != OutcomeStatus::AwaitingObservation =>
                return Err("started execution must await observation".into()),
            ExecutionStatus::Failed | ExecutionStatus::TimedOut | ExecutionStatus::Aborted
                if self.outcome_status != OutcomeStatus::InvalidExecution =>
                return Err("failed/timed-out/aborted execution must be invalid_execution".into()),
            ExecutionStatus::Completed if matches!(self.outcome_status, OutcomeStatus::AwaitingObservation | OutcomeStatus::NotRun | OutcomeStatus::UnsupportedCounterfactual) =>
                return Err("completed execution has incompatible outcome status".into()),
            _ => {}
        }

        match self.outcome_status {
            OutcomeStatus::AwaitingObservation => {
                self.require_action(&p.selected_action_id)?;
                if self.outcome_label.is_some() || self.finalized_at_ms.is_some() {
                    return Err("awaiting outcome cannot have final label/time".into());
                }
            }
            OutcomeStatus::ImmediateObserved | OutcomeStatus::DelayedObserved => {
                self.require_action(&p.selected_action_id)?;
                require_nonempty("observed outcome label", self.outcome_label.as_deref().unwrap_or_default())?;
                if self.evaluator_evidence.is_empty() || self.finalized_at_ms.is_none() {
                    return Err("observed outcome requires evidence and finalization time".into());
                }
            }
            OutcomeStatus::CensoredAtDeadline | OutcomeStatus::Indeterminate => {
                self.require_action(&p.selected_action_id)?;
                if self.outcome_label.is_some() || self.evaluator_evidence.is_empty() || self.finalized_at_ms.is_none() {
                    return Err("censored/indeterminate outcome requires evidence, no label and finalization time".into());
                }
            }
            OutcomeStatus::UnsupportedCounterfactual => {
                if self.execution_status != ExecutionStatus::NotStarted
                    || self.action_support != ActionSupport::NoObservedOutcome
                    || self.observed_action_id.is_some() || self.outcome_label.is_some()
                    || !self.evaluator_evidence.is_empty() {
                    return Err("unsupported counterfactual cannot carry execution or fabricated outcome".into());
                }
            }
            OutcomeStatus::NotRun => {
                if self.action_support != ActionSupport::NoObservedOutcome
                    || self.observed_action_id.is_some() || self.outcome_label.is_some()
                    || !self.evaluator_evidence.is_empty() {
                    return Err("not-run cannot carry observed action or outcome".into());
                }
            }
            OutcomeStatus::InvalidExecution => {
                if self.outcome_label.is_some() { return Err("invalid execution cannot carry scientific label".into()); }
                if self.action_support == ActionSupport::ObservedOutcome { self.require_action(&p.selected_action_id)?; }
                else if self.observed_action_id.is_some() { return Err("unobserved action cannot have an action ID".into()); }
            }
        }
        Ok(())
    }
    fn require_action(&self, expected: &str) -> Result<(), String> {
        if self.action_support != ActionSupport::ObservedOutcome { return Err("observed action support required".into()); }
        let actual = self.observed_action_id.as_deref().ok_or_else(|| "observed action ID required".to_string())?;
        if actual != expected { return Err("observed action differs from proposal".into()); }
        Ok(())
    }
}

fn require_nonempty(field: &str, value: &str) -> Result<(), String> {
    if value.trim().is_empty() { Err(format!("{field} must not be empty")) } else { Ok(()) }
}
fn ensure_unique(field: &str, values: &[String]) -> Result<(), String> {
    for (i, value) in values.iter().enumerate() {
        if values[..i].iter().any(|prior| prior == value) { return Err(format!("{field} contains duplicate {value}")); }
    }
    Ok(())
}
fn ensure_nonempty_items(field: &str, values: &[String]) -> Result<(), String> {
    for value in values { require_nonempty(field, value)?; }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn context() -> CandidateDecisionContext {
        CandidateDecisionContext {
            schema_version: SCHEMA_VERSION, episode_id: "e1".into(), parent_episode_id: None,
            task_family_id: "pilot".into(), decision_index: 0, decision_time_ms: 100,
            deadline_ms: Some(1_000), phase: DecisionPhase::ExperimentalDesign,
            context_digest: "sha256:ctx".into(),
            evidence_available_at_decision: vec![EvidenceReference {
                evidence_id: "obs".into(), content_digest: "sha256:obs".into(),
                observed_at_ms: 80, available_at_ms: 90,
            }],
            permitted_action_ids: vec!["a".into(), "b".into()],
            estimated_budget: ResourceVector::default(),
        }
    }
    fn proposal(c: &CandidateDecisionContext) -> DecisionProposal {
        DecisionProposal {
            episode_id: c.episode_id.clone(), policy_id: "fixed".into(), policy_version: "1".into(),
            selected_action_id: "a".into(), rationale_digest: "sha256:rationale".into(),
            assumption_digests: vec![],
            predicted_outcomes: vec![
                PredictedOutcome { outcome_label: "yes".into(), probability_micros: 400_000 },
                PredictedOutcome { outcome_label: "no".into(), probability_micros: 600_000 },
            ],
            assignment_mechanism: AssignmentMechanism::Randomized {
                action_probabilities: vec![
                    ActionProbability { action_id: "a".into(), probability_micros: 500_000 },
                    ActionProbability { action_id: "b".into(), probability_micros: 500_000 },
                ],
                protocol_ref: "random-protocol".into(),
            },
            causal_design: CausalDesign::RandomizedControlled,
            identification_assumption_digests: vec!["sha256:consistency".into()],
        }
    }
    fn complete_cost() -> ResourceVector {
        ResourceVector {
            inference_tokens: Some(0), cpu_millis: Some(1), accelerator_millis: Some(0),
            wall_clock_millis: Some(1), expert_minutes: Some(0), data_cost_microusd: Some(0),
        }
    }
    fn resolution(c: &CandidateDecisionContext) -> EvaluatorResolution {
        EvaluatorResolution {
            episode_id: c.episode_id.clone(), execution_status: ExecutionStatus::Completed,
            outcome_status: OutcomeStatus::DelayedObserved, action_support: ActionSupport::ObservedOutcome,
            observed_action_id: Some("a".into()), outcome_label: Some("yes".into()),
            evaluator_evidence: vec![EvidenceReference {
                evidence_id: "result".into(), content_digest: "sha256:result".into(),
                observed_at_ms: 900, available_at_ms: 950,
            }],
            finalized_at_ms: Some(960), actual_cost: complete_cost(),
            cost_completeness: CostCompleteness::Complete,
            independent_adjudication_ref: Some("independent-review".into()),
        }
    }

    #[test]
    fn rejects_lookahead_evidence() {
        let mut c = context(); c.evidence_available_at_decision[0].available_at_ms = 101;
        assert!(c.validate().unwrap_err().contains("look-ahead"));
    }
    #[test]
    fn rejects_probability_mass_error() {
        let c = context(); let mut p = proposal(&c); p.predicted_outcomes[0].probability_micros = 399_999;
        assert!(p.validate_against(&c).unwrap_err().contains("sum to"));
    }
    #[test]
    fn randomized_design_requires_full_positive_action_distribution() {
        let c = context();
        let mut p = proposal(&c);
        p.assignment_mechanism = AssignmentMechanism::Randomized {
            action_probabilities: vec![
                ActionProbability { action_id: "a".into(), probability_micros: 1_000_000 },
            ],
            protocol_ref: "random-protocol".into(),
        };
        assert!(p.validate_against(&c).unwrap_err().contains("every permitted action"));
    }
    #[test]
    fn rejects_unauthorized_action() {
        let c = context(); let mut p = proposal(&c); p.selected_action_id = "off-menu".into();
        assert!(p.validate_against(&c).unwrap_err().contains("outside"));
    }
    #[test]
    fn rejects_fabricated_unsupported_outcome() {
        let c = context(); let p = proposal(&c); let mut r = resolution(&c);
        r.execution_status = ExecutionStatus::NotStarted; r.outcome_status = OutcomeStatus::UnsupportedCounterfactual;
        r.action_support = ActionSupport::NoObservedOutcome; r.observed_action_id = None;
        r.outcome_label = Some("yes".into()); r.evaluator_evidence.clear(); r.finalized_at_ms = None;
        assert!(r.validate_against(&c, &p).unwrap_err().contains("unsupported counterfactual"));
    }
    #[test]
    fn preserves_delayed_censored_and_indeterminate_states() {
        let c = context(); let p = proposal(&c); let mut r = resolution(&c);
        assert!(r.validate_against(&c, &p).is_ok());
        r.outcome_status = OutcomeStatus::CensoredAtDeadline; r.outcome_label = None;
        assert!(r.validate_against(&c, &p).is_ok());
        r.outcome_status = OutcomeStatus::Indeterminate;
        assert!(r.validate_against(&c, &p).is_ok());
    }
    #[test]
    fn complete_cost_requires_explicit_zero_or_value_for_every_dimension() {
        let c = context(); let p = proposal(&c); let mut r = resolution(&c);
        r.actual_cost.accelerator_millis = None;
        assert!(r.validate_against(&c, &p).unwrap_err().contains("every resource dimension"));
    }
    #[test]
    fn execution_failure_is_not_a_scientific_negative_result() {
        let c = context(); let p = proposal(&c); let mut r = resolution(&c);
        r.execution_status = ExecutionStatus::TimedOut; r.outcome_status = OutcomeStatus::DelayedObserved;
        assert!(r.validate_against(&c, &p).unwrap_err().contains("invalid_execution"));
    }
}

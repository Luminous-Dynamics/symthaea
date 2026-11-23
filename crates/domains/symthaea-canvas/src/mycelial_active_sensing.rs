// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decision-value-based ranking of additional mycelial observations.
//!
//! This module does not maximize entropy reduction and does not actuate a
//! microscope. It computes the expected reduction in a caller-declared Bayes
//! decision loss under explicit action-outcome likelihoods, then subtracts
//! acquisition cost expressed on the same utility scale.
//!
//! Likelihoods are supplied by a calibrated acquisition-response model or a
//! frozen retrospective replay. Missing or non-normalized likelihoods fail
//! closed; they are never silently normalized by this selector.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_fusion_counterfactual::MycelialFusionCandidate;
use crate::mycelial_graph_hypothesis_posterior::{
    MycelialGraphHypothesisPosterior, MycelialGraphHypothesisPosteriorError,
    evaluate_mycelial_graph_hypothesis_posterior,
};
use crate::mycelial_reconstruction_metrics::AnnotatedMycelialSkeleton;

pub const MYCELIAL_ACTIVE_SENSING_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_ACQUISITION_ACTIONS: usize = 64;
pub const MAX_MYCELIAL_ACQUISITION_OUTCOMES_PER_ACTION: usize = 32;
pub const MAX_MYCELIAL_ACQUISITION_DECISIONS: usize = 128;
pub const MAX_MYCELIAL_ACTIVE_SENSING_EVALUATIONS: usize = 1_000_000;
const NORMALIZATION_TOLERANCE: f64 = 1.0e-9;
const MAX_ACTIVE_SENSING_ID_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialHypothesisLoss {
    /// Must exactly match the accepted-candidate vector in the posterior row.
    pub accepted_candidate_ids: Vec<String>,
    pub loss: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAcquisitionDecision {
    pub decision_id: String,
    /// Unit / scale identity for every loss in this decision inventory.
    pub loss_scale_id: String,
    /// One loss for every posterior hypothesis, in exact posterior row order.
    pub losses_by_hypothesis: Vec<MycelialHypothesisLoss>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialHypothesisLikelihood {
    /// Must exactly match the accepted-candidate vector in the posterior row.
    pub accepted_candidate_ids: Vec<String>,
    /// P(this outcome | this exact feasible graph hypothesis).
    pub likelihood: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAcquisitionOutcome {
    pub outcome_id: String,
    /// One likelihood for every posterior hypothesis, in exact posterior row order.
    pub likelihoods_by_hypothesis: Vec<MycelialHypothesisLikelihood>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAcquisitionAction {
    pub action_id: String,
    /// Acquisition cost expressed in exactly the same units as decision loss.
    pub cost_in_loss_units: f64,
    pub cost_scale_id: String,
    pub outcomes: Vec<MycelialAcquisitionOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAcquisitionOutcomeValue {
    pub outcome_id: String,
    /// Prior predictive probability of this outcome under the current posterior.
    pub predictive_probability: f64,
    /// Best decision after observing this outcome; None when predictive mass is zero.
    pub best_decision_id: Option<String>,
    /// Conditional Bayes risk after this outcome; None when predictive mass is zero.
    pub conditional_bayes_risk: Option<f64>,
    /// Joint contribution to expected post-acquisition Bayes risk.
    pub expected_risk_contribution: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAcquisitionActionValue {
    pub action_id: String,
    pub cost_in_loss_units: f64,
    pub expected_posterior_bayes_risk: f64,
    pub gross_value_of_information: f64,
    pub net_value_of_information: f64,
    pub outcomes: Vec<MycelialAcquisitionOutcomeValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialActiveSensingReport {
    pub schema_version: u16,
    pub source_graph_digest: String,
    pub candidate_inventory_digest: String,
    pub posterior_hypothesis_count: usize,
    pub posterior_entropy_nats: f64,
    pub loss_scale_id: String,
    pub decision_inventory_digest: String,
    pub acquisition_action_inventory_digest: String,
    pub baseline_best_decision_id: String,
    pub baseline_bayes_risk: f64,
    /// Highest positive net value; None means no action clears its declared cost.
    pub recommended_action_id: Option<String>,
    pub actions: Vec<MycelialAcquisitionActionValue>,
    /// Explicitly advisory: this report does not control microscope hardware.
    pub advisory_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MycelialActiveSensingError {
    InvalidPosterior(String),
    InvalidDecision(String),
    InvalidAction(String),
    LossScaleMismatch,
    ResourceBudgetExceeded(usize),
    NoDecisionOptions,
    ReportMismatch,
}

impl fmt::Display for MycelialActiveSensingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPosterior(reason) => {
                write!(f, "invalid posterior for active sensing: {reason}")
            }
            Self::InvalidDecision(id) => {
                write!(f, "invalid active-sensing decision option {id}")
            }
            Self::InvalidAction(id) => {
                write!(f, "invalid acquisition action or outcome model {id}")
            }
            Self::LossScaleMismatch => {
                write!(f, "decision losses and acquisition costs use different scales")
            }
            Self::ResourceBudgetExceeded(count) => write!(
                f,
                "active-sensing evaluation budget {count} exceeds {MAX_MYCELIAL_ACTIVE_SENSING_EVALUATIONS}"
            ),
            Self::NoDecisionOptions => {
                write!(f, "active sensing requires at least one decision option")
            }
            Self::ReportMismatch => {
                write!(f, "active-sensing report does not match independent recomputation")
            }
        }
    }
}

impl Error for MycelialActiveSensingError {}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_ACTIVE_SENSING_ID_BYTES
        && value.bytes().all(|byte| !byte.is_ascii_control())
}

fn hash_string(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    use fmt::Write as _;
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn hypothesis_inventory_digest(posterior: &MycelialGraphHypothesisPosterior) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-active-sensing-posterior-v1");
    hash_string(&mut hasher, &posterior.source_graph_digest);
    hash_string(&mut hasher, &posterior.candidate_inventory_digest);
    hasher.update(&(posterior.hypotheses.len() as u64).to_le_bytes());
    for hypothesis in &posterior.hypotheses {
        hasher.update(&(hypothesis.accepted_candidate_ids.len() as u64).to_le_bytes());
        for id in &hypothesis.accepted_candidate_ids {
            hash_string(&mut hasher, id);
        }
        hasher.update(&hypothesis.posterior_probability.to_bits().to_le_bytes());
        match hypothesis.log_prior_probability {
            Some(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn decision_inventory_digest(decisions: &[MycelialAcquisitionDecision]) -> String {
    let mut ordered: Vec<_> = decisions.iter().collect();
    ordered.sort_by(|left, right| left.decision_id.cmp(&right.decision_id));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-active-sensing-decisions-v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for decision in ordered {
        hash_string(&mut hasher, &decision.decision_id);
        hash_string(&mut hasher, &decision.loss_scale_id);
        hasher.update(&(decision.losses_by_hypothesis.len() as u64).to_le_bytes());
        for row in &decision.losses_by_hypothesis {
            hasher.update(&(row.accepted_candidate_ids.len() as u64).to_le_bytes());
            for id in &row.accepted_candidate_ids {
                hash_string(&mut hasher, id);
            }
            hasher.update(&row.loss.to_bits().to_le_bytes());
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn acquisition_action_inventory_digest(actions: &[MycelialAcquisitionAction]) -> String {
    let mut ordered: Vec<_> = actions.iter().collect();
    ordered.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"symthaea-mycelial-active-sensing-actions-v1");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for action in ordered {
        hash_string(&mut hasher, &action.action_id);
        hash_string(&mut hasher, &action.cost_scale_id);
        hasher.update(&action.cost_in_loss_units.to_bits().to_le_bytes());
        hasher.update(&(action.outcomes.len() as u64).to_le_bytes());
        let mut outcomes: Vec<_> = action.outcomes.iter().collect();
        outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));
        for outcome in outcomes {
            hash_string(&mut hasher, &outcome.outcome_id);
            hasher.update(&(outcome.likelihoods_by_hypothesis.len() as u64).to_le_bytes());
            for row in &outcome.likelihoods_by_hypothesis {
                hasher.update(&(row.accepted_candidate_ids.len() as u64).to_le_bytes());
                for id in &row.accepted_candidate_ids {
                    hash_string(&mut hasher, id);
                }
                hasher.update(&row.likelihood.to_bits().to_le_bytes());
            }
        }
    }
    hex_digest(hasher.finalize().as_bytes())
}

fn validate_decisions(
    posterior: &MycelialGraphHypothesisPosterior,
    decisions: &[MycelialAcquisitionDecision],
) -> Result<String, MycelialActiveSensingError> {
    if decisions.is_empty() {
        return Err(MycelialActiveSensingError::NoDecisionOptions);
    }
    if decisions.len() > MAX_MYCELIAL_ACQUISITION_DECISIONS {
        return Err(MycelialActiveSensingError::ResourceBudgetExceeded(
            decisions.len(),
        ));
    }
    let mut decision_ids = BTreeSet::new();
    let scale_id = decisions[0].loss_scale_id.as_str();
    for decision in decisions {
        if !valid_id(&decision.decision_id)
            || !valid_id(&decision.loss_scale_id)
            || decision.loss_scale_id != scale_id
            || !decision_ids.insert(decision.decision_id.as_str())
            || decision.losses_by_hypothesis.len() != posterior.hypotheses.len()
        {
            return Err(MycelialActiveSensingError::InvalidDecision(
                decision.decision_id.clone(),
            ));
        }
        for (row, hypothesis) in decision
            .losses_by_hypothesis
            .iter()
            .zip(&posterior.hypotheses)
        {
            if row.accepted_candidate_ids != hypothesis.accepted_candidate_ids
                || !row.loss.is_finite()
                || row.loss < 0.0
            {
                return Err(MycelialActiveSensingError::InvalidDecision(
                    decision.decision_id.clone(),
                ));
            }
        }
    }
    Ok(scale_id.to_string())
}

fn validate_actions(
    posterior: &MycelialGraphHypothesisPosterior,
    actions: &[MycelialAcquisitionAction],
    loss_scale_id: &str,
    decision_count: usize,
) -> Result<(), MycelialActiveSensingError> {
    if actions.len() > MAX_MYCELIAL_ACQUISITION_ACTIONS {
        return Err(MycelialActiveSensingError::ResourceBudgetExceeded(
            actions.len(),
        ));
    }
    let mut action_ids = BTreeSet::new();
    let mut total_evaluations = decisions_evaluations(actions, posterior.hypotheses.len(), decision_count);
    if total_evaluations > MAX_MYCELIAL_ACTIVE_SENSING_EVALUATIONS {
        return Err(MycelialActiveSensingError::ResourceBudgetExceeded(
            total_evaluations,
        ));
    }

    for action in actions {
        if !valid_id(&action.action_id)
            || !valid_id(&action.cost_scale_id)
            || action.cost_scale_id != loss_scale_id
            || !action.cost_in_loss_units.is_finite()
            || action.cost_in_loss_units < 0.0
            || !action_ids.insert(action.action_id.as_str())
            || action.outcomes.is_empty()
            || action.outcomes.len() > MAX_MYCELIAL_ACQUISITION_OUTCOMES_PER_ACTION
        {
            return Err(MycelialActiveSensingError::InvalidAction(
                action.action_id.clone(),
            ));
        }
        let mut outcome_ids = BTreeSet::new();
        let mut per_hypothesis_totals = vec![0.0_f64; posterior.hypotheses.len()];
        for outcome in &action.outcomes {
            if !valid_id(&outcome.outcome_id)
                || !outcome_ids.insert(outcome.outcome_id.as_str())
                || outcome.likelihoods_by_hypothesis.len() != posterior.hypotheses.len()
            {
                return Err(MycelialActiveSensingError::InvalidAction(
                    action.action_id.clone(),
                ));
            }
            for (index, (row, hypothesis)) in outcome
                .likelihoods_by_hypothesis
                .iter()
                .zip(&posterior.hypotheses)
                .enumerate()
            {
                if row.accepted_candidate_ids != hypothesis.accepted_candidate_ids
                    || !row.likelihood.is_finite()
                    || !(0.0..=1.0).contains(&row.likelihood)
                {
                    return Err(MycelialActiveSensingError::InvalidAction(
                        action.action_id.clone(),
                    ));
                }
                per_hypothesis_totals[index] += row.likelihood;
            }
        }
        if per_hypothesis_totals.iter().any(|sum| {
            !sum.is_finite() || (*sum - 1.0).abs() > NORMALIZATION_TOLERANCE
        }) {
            return Err(MycelialActiveSensingError::InvalidAction(
                action.action_id.clone(),
            ));
        }
    }
    total_evaluations = decisions_evaluations(actions, posterior.hypotheses.len(), decision_count);
    if total_evaluations > MAX_MYCELIAL_ACTIVE_SENSING_EVALUATIONS {
        return Err(MycelialActiveSensingError::ResourceBudgetExceeded(
            total_evaluations,
        ));
    }
    Ok(())
}

fn decisions_evaluations(
    actions: &[MycelialAcquisitionAction],
    hypothesis_count: usize,
    decision_count: usize,
) -> usize {
    actions
        .iter()
        .map(|action| {
            action
                .outcomes
                .len()
                .saturating_mul(hypothesis_count)
                .saturating_mul(decision_count)
        })
        .fold(0usize, usize::saturating_add)
}

fn baseline_decision(
    posterior: &MycelialGraphHypothesisPosterior,
    decisions: &[MycelialAcquisitionDecision],
) -> (String, f64) {
    let mut ordered: Vec<_> = decisions.iter().collect();
    ordered.sort_by(|left, right| left.decision_id.cmp(&right.decision_id));
    let mut best_id = String::new();
    let mut best_risk = f64::INFINITY;
    for decision in ordered {
        let risk: f64 = posterior
            .hypotheses
            .iter()
            .zip(&decision.losses_by_hypothesis)
            .map(|(hypothesis, row)| {
                hypothesis.posterior_probability * row.loss
            })
            .sum();
        if risk < best_risk {
            best_id = decision.decision_id.clone();
            best_risk = risk;
        }
    }
    (best_id, best_risk)
}

fn evaluate_action(
    posterior: &MycelialGraphHypothesisPosterior,
    decisions: &[MycelialAcquisitionDecision],
    action: &MycelialAcquisitionAction,
    baseline_bayes_risk: f64,
) -> MycelialAcquisitionActionValue {
    let mut ordered_decisions: Vec<_> = decisions.iter().collect();
    ordered_decisions.sort_by(|left, right| left.decision_id.cmp(&right.decision_id));
    let mut outcome_values = Vec::with_capacity(action.outcomes.len());
    let mut expected_posterior_bayes_risk = 0.0;
    let mut ordered_outcomes: Vec<_> = action.outcomes.iter().collect();
    ordered_outcomes.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));

    for outcome in ordered_outcomes {
        let predictive_probability: f64 = posterior
            .hypotheses
            .iter()
            .zip(&outcome.likelihoods_by_hypothesis)
            .map(|(hypothesis, row)| {
                hypothesis.posterior_probability * row.likelihood
            })
            .sum();

        let mut best_decision_id = None;
        let mut best_joint_loss = f64::INFINITY;
        for decision in &ordered_decisions {
            let joint_loss: f64 = posterior
                .hypotheses
                .iter()
                .zip(&outcome.likelihoods_by_hypothesis)
                .zip(&decision.losses_by_hypothesis)
                .map(|((hypothesis, likelihood), loss)| {
                    hypothesis.posterior_probability * likelihood.likelihood * loss.loss
                })
                .sum();
            if joint_loss < best_joint_loss {
                best_joint_loss = joint_loss;
                best_decision_id = Some(decision.decision_id.clone());
            }
        }

        let expected_risk_contribution = if predictive_probability > 0.0 {
            best_joint_loss
        } else {
            0.0
        };
        expected_posterior_bayes_risk += expected_risk_contribution;
        let conditional_bayes_risk = (predictive_probability > 0.0)
            .then_some(expected_risk_contribution / predictive_probability);
        if predictive_probability == 0.0 {
            best_decision_id = None;
        }
        outcome_values.push(MycelialAcquisitionOutcomeValue {
            outcome_id: outcome.outcome_id.clone(),
            predictive_probability,
            best_decision_id,
            conditional_bayes_risk,
            expected_risk_contribution,
        });
    }

    let raw_gross_value = baseline_bayes_risk - expected_posterior_bayes_risk;
    let gross_value_of_information = if raw_gross_value > 0.0 {
        raw_gross_value
    } else {
        0.0
    };
    let net_value_of_information = gross_value_of_information - action.cost_in_loss_units;
    MycelialAcquisitionActionValue {
        action_id: action.action_id.clone(),
        cost_in_loss_units: action.cost_in_loss_units,
        expected_posterior_bayes_risk,
        gross_value_of_information,
        net_value_of_information,
        outcomes: outcome_values,
    }
}

pub fn evaluate_mycelial_active_sensing(
    skeleton: &AnnotatedMycelialSkeleton,
    candidates: &[MycelialFusionCandidate],
    posterior: &MycelialGraphHypothesisPosterior,
    decisions: &[MycelialAcquisitionDecision],
    actions: &[MycelialAcquisitionAction],
) -> Result<MycelialActiveSensingReport, MycelialActiveSensingError> {
    posterior
        .validate_against(skeleton, candidates)
        .map_err(|error: MycelialGraphHypothesisPosteriorError| {
            MycelialActiveSensingError::InvalidPosterior(error.to_string())
        })?;
    let loss_scale_id = validate_decisions(posterior, decisions)?;
    validate_actions(
        posterior,
        actions,
        &loss_scale_id,
        decisions.len(),
    )?;

    let (baseline_best_decision_id, baseline_bayes_risk) =
        baseline_decision(posterior, decisions);
    let mut ordered_actions: Vec<_> = actions.iter().collect();
    ordered_actions.sort_by(|left, right| left.action_id.cmp(&right.action_id));
    let action_values: Vec<_> = ordered_actions
        .iter()
        .map(|action| {
            evaluate_action(
                posterior,
                decisions,
                action,
                baseline_bayes_risk,
            )
        })
        .collect();
    let recommended_action_id = action_values
        .iter()
        .filter(|action| action.net_value_of_information > 0.0)
        .max_by(|left, right| {
            left.net_value_of_information
                .total_cmp(&right.net_value_of_information)
                .then_with(|| right.action_id.cmp(&left.action_id))
        })
        .map(|action| action.action_id.clone());

    Ok(MycelialActiveSensingReport {
        schema_version: MYCELIAL_ACTIVE_SENSING_SCHEMA_VERSION,
        source_graph_digest: posterior.source_graph_digest.clone(),
        candidate_inventory_digest: posterior.candidate_inventory_digest.clone(),
        posterior_hypothesis_count: posterior.hypotheses.len(),
        posterior_entropy_nats: posterior.posterior_entropy_nats,
        loss_scale_id,
        decision_inventory_digest: decision_inventory_digest(decisions),
        acquisition_action_inventory_digest: acquisition_action_inventory_digest(actions),
        baseline_best_decision_id,
        baseline_bayes_risk,
        recommended_action_id,
        actions: action_values,
        advisory_only: true,
    })
}

impl MycelialActiveSensingReport {
    pub fn validate_against(
        &self,
        skeleton: &AnnotatedMycelialSkeleton,
        candidates: &[MycelialFusionCandidate],
        posterior: &MycelialGraphHypothesisPosterior,
        decisions: &[MycelialAcquisitionDecision],
        actions: &[MycelialAcquisitionAction],
    ) -> Result<(), MycelialActiveSensingError> {
        let expected =
            evaluate_mycelial_active_sensing(skeleton, candidates, posterior, decisions, actions)?;
        if self != &expected {
            return Err(MycelialActiveSensingError::ReportMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_fusion_counterfactual::{
        MycelialFusionBridgeLengthBasis, MycelialFusionTarget,
    };
    use crate::mycelial_graph_hypothesis_posterior::evaluate_mycelial_graph_hypothesis_posterior;
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION, ReconstructionDimensionality,
        ReconstructionLengthUnit, ReconstructionPoint3,
    };

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn skeleton() -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "frame-voi".to_string(),
            coordinate_frame_id: "microscope-voi".to_string(),
            calibration_digest: digest('b'),
            source_observation_digest: digest('a'),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "human-annotation".to_string(),
            producer_version: "v1".to_string(),
            junctions: vec![
                AnnotatedJunction {
                    id: "A".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 0.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "B".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 1.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "C".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 10.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "D".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 11.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "E".to_string(),
                    kind: AnnotatedJunctionKind::Origin,
                    position: ReconstructionPoint3 { x: 20.0, y: 0.0, z: 0.0 },
                },
                AnnotatedJunction {
                    id: "F".to_string(),
                    kind: AnnotatedJunctionKind::Tip,
                    position: ReconstructionPoint3 { x: 21.0, y: 0.0, z: 0.0 },
                },
            ],
            segments: vec![
                AnnotatedHyphaSegment {
                    id: "ab".to_string(),
                    start_junction_id: "A".to_string(),
                    end_junction_id: "B".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "cd".to_string(),
                    start_junction_id: "C".to_string(),
                    end_junction_id: "D".to_string(),
                    length: 1.0,
                },
                AnnotatedHyphaSegment {
                    id: "ef".to_string(),
                    start_junction_id: "E".to_string(),
                    end_junction_id: "F".to_string(),
                    length: 1.0,
                },
            ],
        }
    }

    fn candidate(id: &str, source: &str, target: &str, probability: f64) -> MycelialFusionCandidate {
        MycelialFusionCandidate {
            candidate_id: id.to_string(),
            source_tip_junction_id: source.to_string(),
            target: MycelialFusionTarget::TipJunction {
                junction_id: target.to_string(),
            },
            bridge_length: 1.0,
            bridge_length_unit: ReconstructionLengthUnit::Micrometer,
            bridge_length_basis: MycelialFusionBridgeLengthBasis::Measured,
            bridge_length_evidence_digest: digest('e'),
            fusion_probability: probability,
            calibration_policy_id: "calibration-v1".to_string(),
            probability_evidence_digest: digest('f'),
            conflict_group_id: None,
        }
    }

    fn posterior_one_candidate(
        source: &AnnotatedMycelialSkeleton,
    ) -> (Vec<MycelialFusionCandidate>, MycelialGraphHypothesisPosterior) {
        let candidates = vec![candidate("bridge", "B", "D", 0.5)];
        let posterior =
            evaluate_mycelial_graph_hypothesis_posterior(source, &candidates).unwrap();
        (candidates, posterior)
    }

    fn loss_row(
        accepted_candidate_ids: &[String],
        loss: f64,
    ) -> MycelialHypothesisLoss {
        MycelialHypothesisLoss {
            accepted_candidate_ids: accepted_candidate_ids.to_vec(),
            loss,
        }
    }

    fn likelihood_row(
        accepted_candidate_ids: &[String],
        likelihood: f64,
    ) -> MycelialHypothesisLikelihood {
        MycelialHypothesisLikelihood {
            accepted_candidate_ids: accepted_candidate_ids.to_vec(),
            likelihood,
        }
    }

    fn binary_decisions(posterior: &MycelialGraphHypothesisPosterior) -> Vec<MycelialAcquisitionDecision> {
        let no_fusion = &posterior.hypotheses[0].accepted_candidate_ids;
        let fusion = &posterior.hypotheses[1].accepted_candidate_ids;
        vec![
            MycelialAcquisitionDecision {
                decision_id: "accept-fusion".to_string(),
                loss_scale_id: "review-loss-v1".to_string(),
                losses_by_hypothesis: vec![loss_row(no_fusion, 1.0), loss_row(fusion, 0.0)],
            },
            MycelialAcquisitionDecision {
                decision_id: "reject-fusion".to_string(),
                loss_scale_id: "review-loss-v1".to_string(),
                losses_by_hypothesis: vec![loss_row(no_fusion, 0.0), loss_row(fusion, 1.0)],
            },
        ]
    }

    #[test]
    fn perfect_observation_is_selected_by_expected_decision_loss_reduction() {
        let source = skeleton();
        let (candidates, posterior) = posterior_one_candidate(&source);
        let decisions = binary_decisions(&posterior);
        let no_fusion = &posterior.hypotheses[0].accepted_candidate_ids;
        let fusion = &posterior.hypotheses[1].accepted_candidate_ids;
        let actions = vec![
            MycelialAcquisitionAction {
                action_id: "perfect-depth-stack".to_string(),
                cost_in_loss_units: 0.1,
                cost_scale_id: "review-loss-v1".to_string(),
                outcomes: vec![
                    MycelialAcquisitionOutcome {
                        outcome_id: "disconnected".to_string(),
                        likelihoods_by_hypothesis: vec![
                            likelihood_row(no_fusion, 1.0),
                            likelihood_row(fusion, 0.0),
                        ],
                    },
                    MycelialAcquisitionOutcome {
                        outcome_id: "connected".to_string(),
                        likelihoods_by_hypothesis: vec![
                            likelihood_row(no_fusion, 0.0),
                            likelihood_row(fusion, 1.0),
                        ],
                    },
                ],
            },
            MycelialAcquisitionAction {
                action_id: "uninformative-repeat".to_string(),
                cost_in_loss_units: 0.1,
                cost_scale_id: "review-loss-v1".to_string(),
                outcomes: vec![
                    MycelialAcquisitionOutcome {
                        outcome_id: "signal-a".to_string(),
                        likelihoods_by_hypothesis: vec![
                            likelihood_row(no_fusion, 0.5),
                            likelihood_row(fusion, 0.5),
                        ],
                    },
                    MycelialAcquisitionOutcome {
                        outcome_id: "signal-b".to_string(),
                        likelihoods_by_hypothesis: vec![
                            likelihood_row(no_fusion, 0.5),
                            likelihood_row(fusion, 0.5),
                        ],
                    },
                ],
            },
        ];
        let report = evaluate_mycelial_active_sensing(
            &source,
            &candidates,
            &posterior,
            &decisions,
            &actions,
        )
        .unwrap();
        assert_eq!(report.baseline_best_decision_id, "accept-fusion");
        close(report.baseline_bayes_risk, 0.5);
        assert_eq!(
            report.recommended_action_id.as_deref(),
            Some("perfect-depth-stack")
        );
        let perfect = report
            .actions
            .iter()
            .find(|action| action.action_id == "perfect-depth-stack")
            .unwrap();
        close(perfect.expected_posterior_bayes_risk, 0.0);
        close(perfect.gross_value_of_information, 0.5);
        close(perfect.net_value_of_information, 0.4);
        report
            .validate_against(&source, &candidates, &posterior, &decisions, &actions)
            .unwrap();
    }

    #[test]
    fn action_that_only_reveals_a_decision_irrelevant_hypothesis_has_zero_value() {
        let source = skeleton();
        let candidates = vec![
            candidate("one", "B", "D", 0.5),
            candidate("two", "F", "B", 0.5),
        ];
        let posterior =
            evaluate_mycelial_graph_hypothesis_posterior(&source, &candidates).unwrap();
        let one = |mask: usize| {
            posterior.hypotheses[mask].accepted_candidate_ids.clone()
        };
        let decisions = vec![
            MycelialAcquisitionDecision {
                decision_id: "accept-one".to_string(),
                loss_scale_id: "review-loss-v1".to_string(),
                losses_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| {
                        loss_row(
                            &hypothesis.accepted_candidate_ids,
                            if hypothesis.accepted_candidate_ids.contains(&"one".to_string()) {
                                0.0
                            } else {
                                1.0
                            },
                        )
                    })
                    .collect(),
            },
            MycelialAcquisitionDecision {
                decision_id: "reject-one".to_string(),
                loss_scale_id: "review-loss-v1".to_string(),
                losses_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| {
                        loss_row(
                            &hypothesis.accepted_candidate_ids,
                            if hypothesis.accepted_candidate_ids.contains(&"one".to_string()) {
                                1.0
                            } else {
                                0.0
                            },
                        )
                    })
                    .collect(),
            },
        ];
        let outcomes = vec![
            MycelialAcquisitionOutcome {
                outcome_id: "two-absent".to_string(),
                likelihoods_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| {
                        likelihood_row(
                            &hypothesis.accepted_candidate_ids,
                            if hypothesis.accepted_candidate_ids.contains(&"two".to_string()) {
                                0.0
                            } else {
                                1.0
                            },
                        )
                    })
                    .collect(),
            },
            MycelialAcquisitionOutcome {
                outcome_id: "two-present".to_string(),
                likelihoods_by_hypothesis: posterior
                    .hypotheses
                    .iter()
                    .map(|hypothesis| {
                        likelihood_row(
                            &hypothesis.accepted_candidate_ids,
                            if hypothesis.accepted_candidate_ids.contains(&"two".to_string()) {
                                1.0
                            } else {
                                0.0
                            },
                        )
                    })
                    .collect(),
            },
        ];
        let actions = vec![MycelialAcquisitionAction {
            action_id: "image-only-for-two".to_string(),
            cost_in_loss_units: 0.0,
            cost_scale_id: "review-loss-v1".to_string(),
            outcomes,
        }];
        let report = evaluate_mycelial_active_sensing(
            &source,
            &candidates,
            &posterior,
            &decisions,
            &actions,
        )
        .unwrap();
        close(report.baseline_bayes_risk, 0.5);
        close(report.actions[0].expected_posterior_bayes_risk, 0.5);
        close(report.actions[0].gross_value_of_information, 0.0);
        assert_eq!(report.recommended_action_id, None);
    }

    #[test]
    fn acquisition_cost_can_make_an_informative_action_not_worthwhile() {
        let source = skeleton();
        let (candidates, posterior) = posterior_one_candidate(&source);
        let decisions = binary_decisions(&posterior);
        let no_fusion = &posterior.hypotheses[0].accepted_candidate_ids;
        let fusion = &posterior.hypotheses[1].accepted_candidate_ids;
        let actions = vec![MycelialAcquisitionAction {
            action_id: "expensive-perfect-measurement".to_string(),
            cost_in_loss_units: 0.6,
            cost_scale_id: "review-loss-v1".to_string(),
            outcomes: vec![
                MycelialAcquisitionOutcome {
                    outcome_id: "negative".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 1.0),
                        likelihood_row(fusion, 0.0),
                    ],
                },
                MycelialAcquisitionOutcome {
                    outcome_id: "positive".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 0.0),
                        likelihood_row(fusion, 1.0),
                    ],
                },
            ],
        }];
        let report = evaluate_mycelial_active_sensing(
            &source,
            &candidates,
            &posterior,
            &decisions,
            &actions,
        )
        .unwrap();
        close(report.actions[0].gross_value_of_information, 0.5);
        close(report.actions[0].net_value_of_information, -0.1);
        assert_eq!(report.recommended_action_id, None);
    }

    #[test]
    fn malformed_or_non_normalized_likelihoods_fail_closed() {
        let source = skeleton();
        let (candidates, posterior) = posterior_one_candidate(&source);
        let decisions = binary_decisions(&posterior);
        let no_fusion = &posterior.hypotheses[0].accepted_candidate_ids;
        let fusion = &posterior.hypotheses[1].accepted_candidate_ids;
        let actions = vec![MycelialAcquisitionAction {
            action_id: "invalid-sensor-model".to_string(),
            cost_in_loss_units: 0.0,
            cost_scale_id: "review-loss-v1".to_string(),
            outcomes: vec![
                MycelialAcquisitionOutcome {
                    outcome_id: "outcome-a".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 0.8),
                        likelihood_row(fusion, 0.8),
                    ],
                },
                MycelialAcquisitionOutcome {
                    outcome_id: "outcome-b".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 0.8),
                        likelihood_row(fusion, 0.2),
                    ],
                },
            ],
        }];
        assert!(matches!(
            evaluate_mycelial_active_sensing(
                &source,
                &candidates,
                &posterior,
                &decisions,
                &actions,
            ),
            Err(MycelialActiveSensingError::InvalidAction(id))
                if id == "invalid-sensor-model"
        ));
    }

    #[test]
    fn tampered_active_sensing_report_is_detected() {
        let source = skeleton();
        let (candidates, posterior) = posterior_one_candidate(&source);
        let decisions = binary_decisions(&posterior);
        let no_fusion = &posterior.hypotheses[0].accepted_candidate_ids;
        let fusion = &posterior.hypotheses[1].accepted_candidate_ids;
        let actions = vec![MycelialAcquisitionAction {
            action_id: "perfect".to_string(),
            cost_in_loss_units: 0.1,
            cost_scale_id: "review-loss-v1".to_string(),
            outcomes: vec![
                MycelialAcquisitionOutcome {
                    outcome_id: "not-fused".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 1.0),
                        likelihood_row(fusion, 0.0),
                    ],
                },
                MycelialAcquisitionOutcome {
                    outcome_id: "fused".to_string(),
                    likelihoods_by_hypothesis: vec![
                        likelihood_row(no_fusion, 0.0),
                        likelihood_row(fusion, 1.0),
                    ],
                },
            ],
        }];
        let mut report = evaluate_mycelial_active_sensing(
            &source,
            &candidates,
            &posterior,
            &decisions,
            &actions,
        )
        .unwrap();
        report.recommended_action_id = None;
        assert_eq!(
            report
                .validate_against(&source, &candidates, &posterior, &decisions, &actions)
                .unwrap_err(),
            MycelialActiveSensingError::ReportMismatch
        );
    }
}

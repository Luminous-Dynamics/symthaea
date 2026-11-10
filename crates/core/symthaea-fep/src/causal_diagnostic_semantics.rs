// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed causal semantics for FEP diagnostic actions.
//!
//! This module prevents an action-conditioned likelihood from being treated
//! as causally identified merely because it is indexed by an action ID.
//! Passive observation, measurement, hard intervention, and mechanism
//! replacement carry different model and provenance requirements.
//!
//! Validation checks declared identifiers, typed fields, digest shape, and
//! one-to-one alignment with diagnostic actions. It produces a canonical
//! descriptor-set digest. It does not authenticate evidence, prove causal
//! identification, execute actions, or authorize physical interventions.
//! Existing SCM and active-probe owners remain responsible for causal models
//! and identification analyses.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::decision_relevant_information::{
    DecisionDiagnosticConfig, DecisionHypothesis, DiagnosticAction,
};

/// How a measurement's possible perturbation of the measured system is handled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementBackAction {
    /// A versioned model predicts the measurement's effect on the system.
    Modelled {
        back_action_model_digest: String,
    },
    /// Evidence bounds back-action as negligible within a declared scope.
    BoundedNegligible {
        negligible_back_action_evidence_digest: String,
    },
}

/// Declared causal semantics for one proposal-only diagnostic action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalDiagnosticActionKind {
    /// Observe an existing process without intentionally changing it.
    PassiveObservation {
        observation_model_digest: String,
        evidence_scope_digest: String,
    },
    /// Apply a measurement protocol with an explicit observation and
    /// back-action contract.
    Measurement {
        measurement_protocol_digest: String,
        observation_model_digest: String,
        back_action: MeasurementBackAction,
    },
    /// A hard intervention represented as do(target = value) under an SCM.
    HardIntervention {
        structural_model_digest: String,
        intervention_specification_digest: String,
        target_variable_id: String,
        intervention_value_digest: String,
        identification_assumptions_digest: String,
        outcome_model_digest: String,
        carryover_model_digest: String,
    },
    /// Explicitly replace a structural mechanism. This is not an ordinary
    /// observation or a shorthand for a hard intervention.
    MechanismReplacement {
        structural_model_digest: String,
        target_mechanism_id: String,
        replacement_mechanism_digest: String,
        intervention_specification_digest: String,
        identification_assumptions_digest: String,
        outcome_model_digest: String,
        carryover_model_digest: String,
    },
}

/// Causal descriptor paired with exactly one ordinary diagnostic action.
///
/// action_id must match one DiagnosticAction.id. This alignment prevents an
/// action's observation likelihood from being attached to a different
/// intervention or measurement specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalDiagnosticActionDescriptor {
    pub action_id: String,
    /// Digest of the complete action-conditioned likelihood column/table
    /// supplied to the decision kernel. The production adapter must recompute
    /// this from the actual hypothesis/action/outcome tensor before accepting
    /// the descriptor; this module only validates and binds the supplied digest.
    pub action_likelihood_table_digest: String,
    pub kind: CausalDiagnosticActionKind,
}

/// Summary receipt for a validated, canonically identified action set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalDiagnosticActionSetReceipt {
    pub action_count: usize,
    pub passive_observation_count: usize,
    pub measurement_count: usize,
    pub hard_intervention_count: usize,
    pub mechanism_replacement_count: usize,
    /// SHA-256 over sorted action IDs, declared costs/risks, and typed claims.
    pub descriptor_set_digest: String,
}

/// Complete digest envelope for a decision-focused diagnostic problem.
///
/// Binds the causal action descriptors, hypothesis identities and priors,
/// action-conditioned likelihood tables, downstream decision-loss table,
/// and task specification. The source revision and evaluator authority belong
/// in the outer run/verification receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CausalDiagnosticDecisionProblemReceipt {
    pub action_set_digest: String,
    pub hypothesis_set_digest: String,
    pub action_likelihood_table_set_digest: String,
    pub decision_loss_table_digest: String,
    pub decision_policy_config_digest: String,
    pub task_specification_digest: String,
    pub decision_problem_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CausalDiagnosticActionError {
    #[error("at least one diagnostic action is required")]
    EmptyActionSet,
    #[error("diagnostic action count {actual} does not match descriptor count {expected}")]
    DescriptorCountMismatch { expected: usize, actual: usize },
    #[error("action {action_index} has an empty or duplicate identifier")]
    InvalidActionIdentifier { action_index: usize },
    #[error("action {action_index} has an invalid observation cost")]
    InvalidObservationCost { action_index: usize },
    #[error("action {action_index} has an invalid safety-risk value")]
    InvalidSafetyRisk { action_index: usize },
    #[error("descriptor {descriptor_index} has an empty or duplicate action identifier")]
    InvalidDescriptorIdentifier { descriptor_index: usize },
    #[error("causal descriptor has no matching diagnostic action: {0}")]
    UnknownAction(String),
    #[error("diagnostic action has no causal descriptor: {0}")]
    MissingDescriptor(String),
    #[error("identifier is empty or invalid: {0}")]
    InvalidIdentifier(&'static str),
    #[error("digest is not a 64-character lowercase hexadecimal SHA-256 value: {0}")]
    InvalidDigest(&'static str),
    #[error("decision hypothesis set is empty or malformed")]
    InvalidHypothesisSet,
    #[error("action index {action_index} is outside 0..{action_count}")]
    InvalidActionIndex {
        action_index: usize,
        action_count: usize,
    },
    #[error("hypothesis {hypothesis_id} has an invalid action-likelihood distribution")]
    InvalidLikelihoodDistribution { hypothesis_id: String },
    #[error("action-likelihood digest does not match the decision-kernel input for action {0}")]
    ActionLikelihoodDigestMismatch(String),
    #[error("decision-loss table is empty, non-finite, negative, or non-rectangular")]
    InvalidDecisionLossTable,
    #[error("decision policy weights or safety threshold are invalid")]
    InvalidDecisionPolicyConfig,
}

/// Compute a canonical digest of one action's complete predictive-likelihood table.
///
/// The table binds sorted hypothesis identifiers, normalized hypothesis priors,
/// outcome-category order, and each P(outcome | hypothesis, action) value. This
/// is the concrete digest a production adapter must compare with the descriptor
/// before passing the matching action to the EVSI kernel.
pub fn compute_action_likelihood_table_digest(
    hypotheses: &[DecisionHypothesis],
    action_index: usize,
) -> Result<String, CausalDiagnosticActionError> {
    if hypotheses.is_empty() {
        return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
    }

    let action_count = hypotheses[0].outcome_probabilities_by_action.len();
    if action_index >= action_count {
        return Err(CausalDiagnosticActionError::InvalidActionIndex {
            action_index,
            action_count,
        });
    }

    let prior_sum: f64 = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .sum();
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > 1e-9 {
        return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
    }

    let outcome_count = hypotheses[0].outcome_probabilities_by_action[action_index].len();
    if outcome_count == 0 {
        return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
    }

    let mut seen_ids = BTreeSet::new();
    let mut ordered: Vec<&DecisionHypothesis> = hypotheses.iter().collect();
    for hypothesis in &ordered {
        if hypothesis.id.trim().is_empty()
            || !seen_ids.insert(hypothesis.id.as_str())
            || !hypothesis.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&hypothesis.prior_probability)
            || hypothesis.outcome_probabilities_by_action.len() != action_count
        {
            return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
        }

        let distribution = &hypothesis.outcome_probabilities_by_action[action_index];
        let mass: f64 = distribution.iter().sum();
        if distribution.len() != outcome_count
            || distribution
                .iter()
                .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
            || !mass.is_finite()
            || (mass - 1.0).abs() > 1e-9
        {
            return Err(CausalDiagnosticActionError::InvalidLikelihoodDistribution {
                hypothesis_id: hypothesis.id.clone(),
            });
        }
    }
    ordered.sort_by(|left, right| left.id.cmp(&right.id));

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-action-likelihood-table-v1\0");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    hasher.update(&(outcome_count as u64).to_le_bytes());
    for hypothesis in ordered {
        hash_field(&mut hasher, "hypothesis_id", &hypothesis.id);
        hasher.update(&hypothesis.prior_probability.to_bits().to_le_bytes());
        for probability in &hypothesis.outcome_probabilities_by_action[action_index] {
            hasher.update(&probability.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Validate the action descriptors and recompute every action-likelihood
/// digest from the exact decision-kernel hypothesis tensor.
///
/// This closes the shape/identity gap between an action's causal descriptor
/// and the predictive distributions actually evaluated by EVSI. It still
/// cannot prove that an upstream causal model generated those distributions
/// from valid observational or interventional assumptions.
pub fn validate_causal_diagnostic_action_likelihood_bindings(
    actions: &[DiagnosticAction],
    descriptors: &[CausalDiagnosticActionDescriptor],
    hypotheses: &[DecisionHypothesis],
) -> Result<CausalDiagnosticActionSetReceipt, CausalDiagnosticActionError> {
    let receipt = validate_causal_diagnostic_action_set(actions, descriptors)?;
    let descriptors_by_id: BTreeMap<&str, &CausalDiagnosticActionDescriptor> = descriptors
        .iter()
        .map(|descriptor| (descriptor.action_id.as_str(), descriptor))
        .collect();

    for (action_index, action) in actions.iter().enumerate() {
        let descriptor = descriptors_by_id
            .get(action.id.as_str())
            .ok_or_else(|| CausalDiagnosticActionError::MissingDescriptor(action.id.clone()))?;
        let expected_digest =
            compute_action_likelihood_table_digest(hypotheses, action_index)?;
        if descriptor.action_likelihood_table_digest != expected_digest {
            return Err(CausalDiagnosticActionError::ActionLikelihoodDigestMismatch(
                action.id.clone(),
            ));
        }
    }
    Ok(receipt)
}

/// Digest hypothesis identities and priors independently of action outcomes.
pub fn compute_hypothesis_set_digest(
    hypotheses: &[DecisionHypothesis],
) -> Result<String, CausalDiagnosticActionError> {
    if hypotheses.is_empty() {
        return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
    }
    let prior_sum: f64 = hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .sum();
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > 1e-9 {
        return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
    }

    let mut seen_ids = BTreeSet::new();
    let mut ordered: Vec<&DecisionHypothesis> = hypotheses.iter().collect();
    for hypothesis in &ordered {
        if hypothesis.id.trim().is_empty()
            || !seen_ids.insert(hypothesis.id.as_str())
            || !hypothesis.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&hypothesis.prior_probability)
        {
            return Err(CausalDiagnosticActionError::InvalidHypothesisSet);
        }
    }
    ordered.sort_by(|left, right| left.id.cmp(&right.id));

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-decision-hypothesis-set-v1\0");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    for hypothesis in ordered {
        hash_field(&mut hasher, "hypothesis_id", &hypothesis.id);
        hasher.update(&hypothesis.prior_probability.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Digest the downstream loss matrix, preserving decision order and exact
/// floating-point values. Losses must be finite, non-negative, and rectangular.
pub fn compute_decision_loss_table_digest(
    hypotheses: &[DecisionHypothesis],
) -> Result<String, CausalDiagnosticActionError> {
    if hypotheses.is_empty() {
        return Err(CausalDiagnosticActionError::InvalidDecisionLossTable);
    }
    let mut seen_ids = BTreeSet::new();
    let mut ordered: Vec<&DecisionHypothesis> = hypotheses.iter().collect();
    for hypothesis in &ordered {
        if hypothesis.id.trim().is_empty()
            || !seen_ids.insert(hypothesis.id.as_str())
        {
            return Err(CausalDiagnosticActionError::InvalidDecisionLossTable);
        }
    }
    ordered.sort_by(|left, right| left.id.cmp(&right.id));

    let decision_count = ordered[0].loss_by_final_decision.len();
    if decision_count == 0 {
        return Err(CausalDiagnosticActionError::InvalidDecisionLossTable);
    }
    for hypothesis in &ordered {
        if hypothesis.loss_by_final_decision.len() != decision_count
            || hypothesis
                .loss_by_final_decision
                .iter()
                .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(CausalDiagnosticActionError::InvalidDecisionLossTable);
        }
    }

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-decision-loss-table-v1\0");
    hasher.update(&(ordered.len() as u64).to_le_bytes());
    hasher.update(&(decision_count as u64).to_le_bytes());
    for hypothesis in ordered {
        hash_field(&mut hasher, "hypothesis_id", &hypothesis.id);
        for loss in &hypothesis.loss_by_final_decision {
            hasher.update(&loss.to_bits().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Digest the cost/risk weights and hard safety threshold used by the
/// decision-focused policy. A change in any of these values changes the
/// decision-problem identity even when observations and task losses stay fixed.
pub fn compute_decision_policy_config_digest(
    config: DecisionDiagnosticConfig,
) -> Result<String, CausalDiagnosticActionError> {
    if !config.observation_cost_weight.is_finite()
        || config.observation_cost_weight < 0.0
        || !config.safety_risk_weight.is_finite()
        || config.safety_risk_weight < 0.0
        || !config.maximum_safety_risk.is_finite()
        || !(0.0..=1.0).contains(&config.maximum_safety_risk)
    {
        return Err(CausalDiagnosticActionError::InvalidDecisionPolicyConfig);
    }

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-decision-policy-config-v1\0");
    hasher.update(&config.observation_cost_weight.to_bits().to_le_bytes());
    hasher.update(&config.safety_risk_weight.to_bits().to_le_bytes());
    hasher.update(&config.maximum_safety_risk.to_bits().to_le_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

/// Bind one complete decision problem before an EVSI result is admitted.
///
/// This verifies the action descriptor set, recomputes the supplied likelihood
/// tables, validates model priors and losses, and binds the caller's task
/// specification digest. It does not prove that the likelihoods or losses
/// accurately represent the world.
pub fn validate_causal_diagnostic_decision_problem(
    actions: &[DiagnosticAction],
    descriptors: &[CausalDiagnosticActionDescriptor],
    hypotheses: &[DecisionHypothesis],
    decision_config: DecisionDiagnosticConfig,
    task_specification_digest: &str,
) -> Result<CausalDiagnosticDecisionProblemReceipt, CausalDiagnosticActionError> {
    let action_set =
        validate_causal_diagnostic_action_likelihood_bindings(actions, descriptors, hypotheses)?;
    let hypothesis_set_digest = compute_hypothesis_set_digest(hypotheses)?;
    let decision_loss_table_digest = compute_decision_loss_table_digest(hypotheses)?;
    let decision_policy_config_digest =
        compute_decision_policy_config_digest(decision_config)?;

    let mut ordered_action_digests: Vec<(String, String)> = actions
        .iter()
        .enumerate()
        .map(|(action_index, action)| {
            Ok((
                action.id.clone(),
                compute_action_likelihood_table_digest(hypotheses, action_index)?,
            ))
        })
        .collect::<Result<_, CausalDiagnosticActionError>>()?;
    ordered_action_digests.sort_by(|left, right| left.0.cmp(&right.0));

    let mut action_likelihood_hasher = Sha256::new();
    action_likelihood_hasher.update(b"symthaea-action-likelihood-table-set-v1\0");
    action_likelihood_hasher.update(&(ordered_action_digests.len() as u64).to_le_bytes());
    for (action_id, likelihood_digest) in ordered_action_digests {
        hash_field(&mut action_likelihood_hasher, "action_id", &action_id);
        hash_field(
            &mut action_likelihood_hasher,
            "action_likelihood_table_digest",
            &likelihood_digest,
        );
    }
    let action_likelihood_table_set_digest =
        format!("{:x}", action_likelihood_hasher.finalize());

    let mut problem_hasher = Sha256::new();
    problem_hasher.update(b"symthaea-causal-diagnostic-decision-problem-v1\0");
    hash_field(
        &mut problem_hasher,
        "action_set_digest",
        &action_set.descriptor_set_digest,
    );
    hash_field(
        &mut problem_hasher,
        "hypothesis_set_digest",
        &hypothesis_set_digest,
    );
    hash_field(
        &mut problem_hasher,
        "action_likelihood_table_set_digest",
        &action_likelihood_table_set_digest,
    );
    hash_field(
        &mut problem_hasher,
        "decision_loss_table_digest",
        &decision_loss_table_digest,
    );
    hash_field(
        &mut problem_hasher,
        "decision_policy_config_digest",
        &decision_policy_config_digest,
    );
    hash_digest(
        &mut problem_hasher,
        "task_specification_digest",
        task_specification_digest,
    )?;

    Ok(CausalDiagnosticDecisionProblemReceipt {
        action_set_digest: action_set.descriptor_set_digest,
        hypothesis_set_digest,
        action_likelihood_table_set_digest,
        decision_loss_table_digest,
        decision_policy_config_digest,
        task_specification_digest: task_specification_digest.to_owned(),
        decision_problem_digest: format!("{:x}", problem_hasher.finalize()),
    })
}

/// Validate a one-to-one typed causal descriptor set and produce its digest.
///
/// A successful result means every action has an explicit semantic kind and
/// well-shaped provenance digests. It does not mean the assumptions are true,
/// authenticated, or sufficient for identification.
pub fn validate_causal_diagnostic_action_set(
    actions: &[DiagnosticAction],
    descriptors: &[CausalDiagnosticActionDescriptor],
) -> Result<CausalDiagnosticActionSetReceipt, CausalDiagnosticActionError> {
    if actions.is_empty() {
        return Err(CausalDiagnosticActionError::EmptyActionSet);
    }
    if actions.len() != descriptors.len() {
        return Err(CausalDiagnosticActionError::DescriptorCountMismatch {
            expected: actions.len(),
            actual: descriptors.len(),
        });
    }

    let mut actions_by_id = BTreeMap::new();
    for (action_index, action) in actions.iter().enumerate() {
        if action.id.trim().is_empty() || actions_by_id.insert(action.id.as_str(), action).is_some()
        {
            return Err(CausalDiagnosticActionError::InvalidActionIdentifier {
                action_index,
            });
        }
        if !action.observation_cost.is_finite() || action.observation_cost < 0.0 {
            return Err(CausalDiagnosticActionError::InvalidObservationCost { action_index });
        }
        if !action.safety_risk.is_finite() || !(0.0..=1.0).contains(&action.safety_risk) {
            return Err(CausalDiagnosticActionError::InvalidSafetyRisk { action_index });
        }
    }

    let mut descriptors_by_id = BTreeMap::new();
    for (descriptor_index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.action_id.trim().is_empty()
            || descriptors_by_id
                .insert(descriptor.action_id.as_str(), descriptor)
                .is_some()
        {
            return Err(CausalDiagnosticActionError::InvalidDescriptorIdentifier {
                descriptor_index,
            });
        }
        if !actions_by_id.contains_key(descriptor.action_id.as_str()) {
            return Err(CausalDiagnosticActionError::UnknownAction(
                descriptor.action_id.clone(),
            ));
        }
    }
    for action_id in actions_by_id.keys() {
        if !descriptors_by_id.contains_key(action_id) {
            return Err(CausalDiagnosticActionError::MissingDescriptor(
                (*action_id).to_owned(),
            ));
        }
    }

    let mut receipt = CausalDiagnosticActionSetReceipt {
        action_count: actions.len(),
        passive_observation_count: 0,
        measurement_count: 0,
        hard_intervention_count: 0,
        mechanism_replacement_count: 0,
        descriptor_set_digest: String::new(),
    };

    let mut hasher = Sha256::new();
    hasher.update(b"symthaea-causal-diagnostic-action-set-v1\0");
    hasher.update(&(descriptors_by_id.len() as u64).to_le_bytes());

    // BTreeMap iteration makes the receipt independent of caller ordering.
    for (action_id, descriptor) in descriptors_by_id {
        let action = actions_by_id
            .get(action_id)
            .ok_or_else(|| CausalDiagnosticActionError::MissingDescriptor(action_id.to_owned()))?;

        hash_field(&mut hasher, "action_id", action_id);
        hash_digest(
            &mut hasher,
            "action_likelihood_table_digest",
            &descriptor.action_likelihood_table_digest,
        )?;
        hasher.update(&action.observation_cost.to_bits().to_le_bytes());
        hasher.update(&action.safety_risk.to_bits().to_le_bytes());

        match &descriptor.kind {
            CausalDiagnosticActionKind::PassiveObservation {
                observation_model_digest,
                evidence_scope_digest,
            } => {
                receipt.passive_observation_count += 1;
                hash_field(&mut hasher, "kind", "passive_observation");
                hash_digest(
                    &mut hasher,
                    "observation_model_digest",
                    observation_model_digest,
                )?;
                hash_digest(&mut hasher, "evidence_scope_digest", evidence_scope_digest)?;
            }
            CausalDiagnosticActionKind::Measurement {
                measurement_protocol_digest,
                observation_model_digest,
                back_action,
            } => {
                receipt.measurement_count += 1;
                hash_field(&mut hasher, "kind", "measurement");
                hash_digest(
                    &mut hasher,
                    "measurement_protocol_digest",
                    measurement_protocol_digest,
                )?;
                hash_digest(
                    &mut hasher,
                    "observation_model_digest",
                    observation_model_digest,
                )?;
                match back_action {
                    MeasurementBackAction::Modelled {
                        back_action_model_digest,
                    } => {
                        hash_field(&mut hasher, "back_action_kind", "modelled");
                        hash_digest(
                            &mut hasher,
                            "back_action_model_digest",
                            back_action_model_digest,
                        )?;
                    }
                    MeasurementBackAction::BoundedNegligible {
                        negligible_back_action_evidence_digest,
                    } => {
                        hash_field(&mut hasher, "back_action_kind", "bounded_negligible");
                        hash_digest(
                            &mut hasher,
                            "negligible_back_action_evidence_digest",
                            negligible_back_action_evidence_digest,
                        )?;
                    }
                }
            }
            CausalDiagnosticActionKind::HardIntervention {
                structural_model_digest,
                intervention_specification_digest,
                target_variable_id,
                intervention_value_digest,
                identification_assumptions_digest,
                outcome_model_digest,
                carryover_model_digest,
            } => {
                receipt.hard_intervention_count += 1;
                hash_field(&mut hasher, "kind", "hard_intervention");
                hash_digest(
                    &mut hasher,
                    "structural_model_digest",
                    structural_model_digest,
                )?;
                hash_digest(
                    &mut hasher,
                    "intervention_specification_digest",
                    intervention_specification_digest,
                )?;
                validate_identifier("target_variable_id", target_variable_id)?;
                hash_field(&mut hasher, "target_variable_id", target_variable_id);
                hash_digest(
                    &mut hasher,
                    "intervention_value_digest",
                    intervention_value_digest,
                )?;
                hash_digest(
                    &mut hasher,
                    "identification_assumptions_digest",
                    identification_assumptions_digest,
                )?;
                hash_digest(&mut hasher, "outcome_model_digest", outcome_model_digest)?;
                hash_digest(
                    &mut hasher,
                    "carryover_model_digest",
                    carryover_model_digest,
                )?;
            }
            CausalDiagnosticActionKind::MechanismReplacement {
                structural_model_digest,
                target_mechanism_id,
                replacement_mechanism_digest,
                intervention_specification_digest,
                identification_assumptions_digest,
                outcome_model_digest,
                carryover_model_digest,
            } => {
                receipt.mechanism_replacement_count += 1;
                hash_field(&mut hasher, "kind", "mechanism_replacement");
                hash_digest(
                    &mut hasher,
                    "structural_model_digest",
                    structural_model_digest,
                )?;
                validate_identifier("target_mechanism_id", target_mechanism_id)?;
                hash_field(&mut hasher, "target_mechanism_id", target_mechanism_id);
                hash_digest(
                    &mut hasher,
                    "replacement_mechanism_digest",
                    replacement_mechanism_digest,
                )?;
                hash_digest(
                    &mut hasher,
                    "intervention_specification_digest",
                    intervention_specification_digest,
                )?;
                hash_digest(
                    &mut hasher,
                    "identification_assumptions_digest",
                    identification_assumptions_digest,
                )?;
                hash_digest(&mut hasher, "outcome_model_digest", outcome_model_digest)?;
                hash_digest(
                    &mut hasher,
                    "carryover_model_digest",
                    carryover_model_digest,
                )?;
            }
        }
    }

    receipt.descriptor_set_digest = format!("{:x}", hasher.finalize());
    Ok(receipt)
}

fn validate_identifier(
    field: &'static str,
    value: &str,
) -> Result<(), CausalDiagnosticActionError> {
    if value.trim().is_empty() {
        return Err(CausalDiagnosticActionError::InvalidIdentifier(field));
    }
    Ok(())
}

fn hash_digest(
    hasher: &mut Sha256,
    field: &'static str,
    digest: &str,
) -> Result<(), CausalDiagnosticActionError> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(CausalDiagnosticActionError::InvalidDigest(field));
    }
    hash_field(hasher, field, digest);
    Ok(())
}

fn hash_field(hasher: &mut Sha256, field: &str, value: &str) {
    let field_bytes = field.as_bytes();
    let value_bytes = value.as_bytes();
    hasher.update(&(field_bytes.len() as u64).to_le_bytes());
    hasher.update(field_bytes);
    hasher.update(&(value_bytes.len() as u64).to_le_bytes());
    hasher.update(value_bytes);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn action(id: &str) -> DiagnosticAction {
        DiagnosticAction {
            id: id.to_owned(),
            observation_cost: 0.1,
            safety_risk: 0.0,
        }
    }

    fn descriptor(
        id: &str,
        kind: CausalDiagnosticActionKind,
    ) -> CausalDiagnosticActionDescriptor {
        CausalDiagnosticActionDescriptor {
            action_id: id.to_owned(),
            action_likelihood_table_digest: digest('c'),
            kind,
        }
    }

    fn four_kind_fixture() -> (
        Vec<DiagnosticAction>,
        Vec<CausalDiagnosticActionDescriptor>,
    ) {
        let actions = vec![
            action("observe"),
            action("measure"),
            action("do-temperature"),
            action("replace-cooling-law"),
        ];
        let descriptors = vec![
            descriptor(
                "observe",
                CausalDiagnosticActionKind::PassiveObservation {
                    observation_model_digest: digest('a'),
                    evidence_scope_digest: digest('b'),
                },
            ),
            descriptor(
                "measure",
                CausalDiagnosticActionKind::Measurement {
                    measurement_protocol_digest: digest('c'),
                    observation_model_digest: digest('d'),
                    back_action: MeasurementBackAction::Modelled {
                        back_action_model_digest: digest('e'),
                    },
                },
            ),
            descriptor(
                "do-temperature",
                CausalDiagnosticActionKind::HardIntervention {
                    structural_model_digest: digest('f'),
                    intervention_specification_digest: digest('1'),
                    target_variable_id: "temperature".into(),
                    intervention_value_digest: digest('2'),
                    identification_assumptions_digest: digest('3'),
                    outcome_model_digest: digest('4'),
                    carryover_model_digest: digest('5'),
                },
            ),
            descriptor(
                "replace-cooling-law",
                CausalDiagnosticActionKind::MechanismReplacement {
                    structural_model_digest: digest('6'),
                    target_mechanism_id: "cooling-law".into(),
                    replacement_mechanism_digest: digest('7'),
                    intervention_specification_digest: digest('8'),
                    identification_assumptions_digest: digest('9'),
                    outcome_model_digest: digest('a'),
                    carryover_model_digest: digest('b'),
                },
            ),
        ];
        (actions, descriptors)
    }

    #[test]
    fn validates_distinct_causal_action_kinds_and_counts() {
        let (actions, descriptors) = four_kind_fixture();
        let receipt = validate_causal_diagnostic_action_set(&actions, &descriptors)
            .expect("complete typed causal action set");

        assert_eq!(receipt.action_count, 4);
        assert_eq!(receipt.passive_observation_count, 1);
        assert_eq!(receipt.measurement_count, 1);
        assert_eq!(receipt.hard_intervention_count, 1);
        assert_eq!(receipt.mechanism_replacement_count, 1);
        assert_eq!(receipt.descriptor_set_digest.len(), 64);
    }

    #[test]
    fn descriptor_digest_is_order_independent_but_binds_action_identity() {
        let (actions, descriptors) = four_kind_fixture();
        let forward = validate_causal_diagnostic_action_set(&actions, &descriptors)
            .expect("forward descriptor order");
        let mut reverse_descriptors = descriptors.clone();
        reverse_descriptors.reverse();
        let reverse = validate_causal_diagnostic_action_set(&actions, &reverse_descriptors)
            .expect("reverse descriptor order");
        assert_eq!(forward.descriptor_set_digest, reverse.descriptor_set_digest);

        let mut renamed_actions = actions;
        renamed_actions[0].id = "different-observation".to_owned();
        assert!(matches!(
            validate_causal_diagnostic_action_set(&renamed_actions, &descriptors),
            Err(CausalDiagnosticActionError::UnknownAction(_))
                | Err(CausalDiagnosticActionError::MissingDescriptor(_))
        ));
    }

    #[test]
    fn descriptor_digest_binds_action_likelihood_table() {
        let (actions, descriptors) = four_kind_fixture();
        let original = validate_causal_diagnostic_action_set(&actions, &descriptors)
            .expect("original likelihood table bindings");

        let mut changed_descriptors = descriptors;
        changed_descriptors[0].action_likelihood_table_digest = digest('d');
        let changed = validate_causal_diagnostic_action_set(&actions, &changed_descriptors)
            .expect("changed likelihood table binding");
        assert_ne!(original.descriptor_set_digest, changed.descriptor_set_digest);
    }

    #[test]
    fn descriptor_digest_binds_cost_and_safety_risk() {
        let (actions, descriptors) = four_kind_fixture();
        let original = validate_causal_diagnostic_action_set(&actions, &descriptors)
            .expect("original set");
        let mut changed_actions = actions;
        changed_actions[0].observation_cost = 0.2;
        let changed = validate_causal_diagnostic_action_set(&changed_actions, &descriptors)
            .expect("changed cost");
        assert_ne!(original.descriptor_set_digest, changed.descriptor_set_digest);
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_mismatched_descriptors() {
        let (actions, descriptors) = four_kind_fixture();
        assert!(matches!(
            validate_causal_diagnostic_action_set(&actions, &descriptors[..3]),
            Err(CausalDiagnosticActionError::DescriptorCountMismatch { .. })
        ));

        let mut unknown = descriptors.clone();
        unknown[0].action_id = "not-an-action".to_owned();
        assert!(matches!(
            validate_causal_diagnostic_action_set(&actions, &unknown),
            Err(CausalDiagnosticActionError::UnknownAction(_))
        ));

        let mut duplicate = descriptors.clone();
        duplicate[1].action_id = "observe".to_owned();
        assert!(matches!(
            validate_causal_diagnostic_action_set(&actions, &duplicate),
            Err(CausalDiagnosticActionError::InvalidDescriptorIdentifier { .. })
        ));
    }

    #[test]
    fn rejects_empty_action_identifiers_and_invalid_cost_or_risk() {
        assert!(matches!(
            validate_causal_diagnostic_action_set(
                &[action(" ")],
                &[descriptor(
                    " ",
                    CausalDiagnosticActionKind::PassiveObservation {
                        observation_model_digest: digest('a'),
                        evidence_scope_digest: digest('b'),
                    }
                )]
            ),
            Err(CausalDiagnosticActionError::InvalidActionIdentifier { .. })
        ));

        let mut invalid_cost = action("cost");
        invalid_cost.observation_cost = f64::NAN;
        assert!(matches!(
            validate_causal_diagnostic_action_set(
                &[invalid_cost],
                &[descriptor(
                    "cost",
                    CausalDiagnosticActionKind::PassiveObservation {
                        observation_model_digest: digest('a'),
                        evidence_scope_digest: digest('b'),
                    }
                )]
            ),
            Err(CausalDiagnosticActionError::InvalidObservationCost { .. })
        ));

        let mut invalid_risk = action("risk");
        invalid_risk.safety_risk = 1.5;
        assert!(matches!(
            validate_causal_diagnostic_action_set(
                &[invalid_risk],
                &[descriptor(
                    "risk",
                    CausalDiagnosticActionKind::PassiveObservation {
                        observation_model_digest: digest('a'),
                        evidence_scope_digest: digest('b'),
                    }
                )]
            ),
            Err(CausalDiagnosticActionError::InvalidSafetyRisk { .. })
        ));
    }

    fn hypothesis_fixture() -> Vec<DecisionHypothesis> {
        vec![
            DecisionHypothesis {
                id: "fault-a".into(),
                prior_probability: 0.4,
                outcome_probabilities_by_action: vec![
                    vec![0.8, 0.2],
                    vec![0.3, 0.7],
                    vec![0.6, 0.4],
                    vec![0.5, 0.5],
                ],
                loss_by_final_decision: vec![0.0, 1.0],
            },
            DecisionHypothesis {
                id: "fault-b".into(),
                prior_probability: 0.6,
                outcome_probabilities_by_action: vec![
                    vec![0.1, 0.9],
                    vec![0.6, 0.4],
                    vec![0.2, 0.8],
                    vec![0.7, 0.3],
                ],
                loss_by_final_decision: vec![1.0, 0.0],
            },
        ]
    }

    fn bind_fixture_likelihood_digests(
        hypotheses: &[DecisionHypothesis],
        descriptors: &mut [CausalDiagnosticActionDescriptor],
    ) {
        for (action_index, descriptor) in descriptors.iter_mut().enumerate() {
            descriptor.action_likelihood_table_digest =
                compute_action_likelihood_table_digest(hypotheses, action_index)
                    .expect("canonical action-likelihood digest");
        }
    }

    #[test]
    fn recomputes_and_validates_action_likelihood_table_digests() {
        let (actions, mut descriptors) = four_kind_fixture();
        let hypotheses = hypothesis_fixture();
        bind_fixture_likelihood_digests(&hypotheses, &mut descriptors);
        validate_causal_diagnostic_action_likelihood_bindings(
            &actions,
            &descriptors,
            &hypotheses,
        )
        .expect("likelihood digests match the EVSI input table");

        let mut altered_hypotheses = hypotheses.clone();
        altered_hypotheses[0].outcome_probabilities_by_action[2] = vec![0.9, 0.1];
        assert!(matches!(
            validate_causal_diagnostic_action_likelihood_bindings(
                &actions,
                &descriptors,
                &altered_hypotheses,
            ),
            Err(CausalDiagnosticActionError::ActionLikelihoodDigestMismatch(_))
        ));
    }

    #[test]
    fn likelihood_table_digest_is_hypothesis_order_independent() {
        let hypotheses = hypothesis_fixture();
        let original =
            compute_action_likelihood_table_digest(&hypotheses, 0).expect("original digest");
        let mut reversed = hypotheses;
        reversed.reverse();
        let reordered =
            compute_action_likelihood_table_digest(&reversed, 0).expect("reordered digest");
        assert_eq!(original, reordered);
    }

    #[test]
    fn invalid_hypothesis_priors_or_likelihoods_fail_closed() {
        let mut invalid_prior = hypothesis_fixture();
        invalid_prior[0].prior_probability = 0.9;
        assert!(matches!(
            compute_action_likelihood_table_digest(&invalid_prior, 0),
            Err(CausalDiagnosticActionError::InvalidHypothesisSet)
        ));

        let mut invalid_likelihood = hypothesis_fixture();
        invalid_likelihood[1].outcome_probabilities_by_action[0] = vec![0.2, 0.2];
        assert!(matches!(
            compute_action_likelihood_table_digest(&invalid_likelihood, 0),
            Err(CausalDiagnosticActionError::InvalidLikelihoodDistribution { .. })
        ));
    }

    #[test]
    fn decision_problem_receipt_binds_hypotheses_likelihoods_losses_and_task() {
        let (actions, mut descriptors) = four_kind_fixture();
        let hypotheses = hypothesis_fixture();
        bind_fixture_likelihood_digests(&hypotheses, &mut descriptors);
        let original = validate_causal_diagnostic_decision_problem(
            &actions,
            &descriptors,
            &hypotheses,
            decision_config(),
            &digest('a'),
        )
        .expect("complete decision-problem receipt");

        let mut altered_losses = hypotheses.clone();
        altered_losses[0].loss_by_final_decision[0] = 0.25;
        let changed_loss = validate_causal_diagnostic_decision_problem(
            &actions,
            &descriptors,
            &altered_losses,
            decision_config(),
            &digest('a'),
        )
        .expect("changed loss table");

        let changed_task = validate_causal_diagnostic_decision_problem(
            &actions,
            &descriptors,
            &hypotheses,
            decision_config(),
            &digest('b'),
        )
        .expect("changed task specification");

        assert_ne!(
            original.decision_loss_table_digest,
            changed_loss.decision_loss_table_digest
        );
        assert_ne!(
            original.decision_problem_digest,
            changed_loss.decision_problem_digest
        );
        assert_ne!(
            original.decision_problem_digest,
            changed_task.decision_problem_digest
        );
    }

    #[test]
    fn policy_config_changes_decision_problem_identity() {
        let (actions, mut descriptors) = four_kind_fixture();
        let hypotheses = hypothesis_fixture();
        bind_fixture_likelihood_digests(&hypotheses, &mut descriptors);
        let config = decision_config();
        let original = validate_causal_diagnostic_decision_problem(
            &actions,
            &descriptors,
            &hypotheses,
            config,
            &digest('a'),
        )
        .expect("default config receipt");

        let changed = validate_causal_diagnostic_decision_problem(
            &actions,
            &descriptors,
            &hypotheses,
            DecisionDiagnosticConfig {
                observation_cost_weight: 2.0,
                ..config
            },
            &digest('a'),
        )
        .expect("changed policy config receipt");

        assert_ne!(
            original.decision_policy_config_digest,
            changed.decision_policy_config_digest
        );
        assert_ne!(
            original.decision_problem_digest,
            changed.decision_problem_digest
        );
    }

    #[test]
    fn decision_problem_receipt_rejects_invalid_loss_table_and_task_digest() {
        let (actions, mut descriptors) = four_kind_fixture();
        let hypotheses = hypothesis_fixture();
        bind_fixture_likelihood_digests(&hypotheses, &mut descriptors);

        let mut invalid_losses = hypotheses.clone();
        invalid_losses[0].loss_by_final_decision[0] = f64::NAN;
        assert!(matches!(
            validate_causal_diagnostic_decision_problem(
                &actions,
                &descriptors,
                &invalid_losses,
                decision_config(),
                &digest('a'),
            ),
            Err(CausalDiagnosticActionError::InvalidDecisionLossTable)
        ));

        assert!(matches!(
            validate_causal_diagnostic_decision_problem(
                &actions,
                &descriptors,
                &hypotheses,
                decision_config(),
                "not-a-digest",
            ),
            Err(CausalDiagnosticActionError::InvalidDigest(
                "task_specification_digest"
            ))
        ));
    }

    #[test]
    fn invalid_decision_policy_config_fails_closed() {
        assert!(matches!(
            compute_decision_policy_config_digest(DecisionDiagnosticConfig {
                observation_cost_weight: f64::NAN,
                ..decision_config()
            }),
            Err(CausalDiagnosticActionError::InvalidDecisionPolicyConfig)
        ));
    }

    #[test]
    fn measurement_requires_typed_back_action_contract() {
        let actions = vec![action("measurement")];
        for back_action in [
            MeasurementBackAction::Modelled {
                back_action_model_digest: digest('a'),
            },
            MeasurementBackAction::BoundedNegligible {
                negligible_back_action_evidence_digest: digest('b'),
            },
        ] {
            let descriptors = vec![descriptor(
                "measurement",
                CausalDiagnosticActionKind::Measurement {
                    measurement_protocol_digest: digest('c'),
                    observation_model_digest: digest('d'),
                    back_action,
                },
            )];
            assert!(validate_causal_diagnostic_action_set(&actions, &descriptors).is_ok());
        }
    }

    #[test]
    fn malformed_provenance_digests_and_empty_target_ids_fail_closed() {
        let actions = vec![action("intervene")];
        let descriptors = vec![descriptor(
            "intervene",
            CausalDiagnosticActionKind::HardIntervention {
                structural_model_digest: digest('a'),
                intervention_specification_digest: digest('b'),
                target_variable_id: "temperature".into(),
                intervention_value_digest: digest('c'),
                identification_assumptions_digest: "not-a-digest".into(),
                outcome_model_digest: digest('e'),
                carryover_model_digest: digest('f'),
            },
        )];
        assert!(matches!(
            validate_causal_diagnostic_action_set(&actions, &descriptors),
            Err(CausalDiagnosticActionError::InvalidDigest(
                "identification_assumptions_digest"
            ))
        ));

        let empty_target = vec![descriptor(
            "intervene",
            CausalDiagnosticActionKind::HardIntervention {
                structural_model_digest: digest('a'),
                intervention_specification_digest: digest('b'),
                target_variable_id: " ".into(),
                intervention_value_digest: digest('c'),
                identification_assumptions_digest: digest('d'),
                outcome_model_digest: digest('e'),
                carryover_model_digest: digest('f'),
            },
        )];
        assert!(matches!(
            validate_causal_diagnostic_action_set(&actions, &empty_target),
            Err(CausalDiagnosticActionError::InvalidIdentifier(
                "target_variable_id"
            ))
        ));
    }
}

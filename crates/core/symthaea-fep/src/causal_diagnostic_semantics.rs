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

use super::decision_relevant_information::DiagnosticAction;

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

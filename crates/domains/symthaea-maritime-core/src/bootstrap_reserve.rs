// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound bootstrap reserve policy for regenerative lineage reproduction.
//!
//! Construction and qualification being simultaneously available is necessary but
//! not sufficient to found a successor epoch. A lineage may consume the last unit
//! of a transfer-critical dependency while those roles are still reported available.
//! This module defines the semantic minimum reserve that must remain on already-
//! qualified predecessor->successor transfer pairs. Runtime quantities remain an
//! upstream dynamic evidence claim; this module does not simulate or authenticate them.

use crate::{
    qualify_regenerative_epoch_handoff, RegenerativeClosureModel, RegenerativeEpochHandoffError,
    RegenerativeEpochHandoffEvidenceV1, RegenerativeGenomeV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REGENERATIVE_BOOTSTRAP_RESERVE_SCHEMA_V1: u8 = 1;

const MAX_ITEMS: usize = 4096;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

/// One semantic reserve requirement on a transfer pair already qualified by the
/// exact epoch-handoff evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReserveRequirementV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    /// Minimum quantity that must remain transferable after the completed source
    /// observation. This is a policy threshold, not a measurement.
    pub minimum_transfer_units: u64,
    /// Opaque evidence/rationale binding for this reserve threshold.
    pub reserve_requirement_binding: String,
}

/// Evidence-bound semantic policy describing which qualified transfers must retain
/// residual bootstrap reserve before a lineage may claim successor reproduction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReservePolicyV1 {
    pub schema_version: u8,
    pub policy_id: String,
    /// Exact handoff-qualification evidence definition this policy refines.
    pub handoff_qualification_evidence_binding: String,
    /// Strictly sorted and unique by `(source_dependency_id, successor_dependency_id)`.
    pub requirements: Vec<RegenerativeBootstrapReserveRequirementV1>,
    pub evidence_binding: String,
}

/// Claimed runtime quantity for one reserve-critical qualified transfer pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReserveQuantityV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    pub available_transfer_units: u64,
}

/// Dynamic evidence presented for bootstrap-reserve assessment.
///
/// `source_final_observation_tick == Some(source_final_tick)` means the caller claims
/// the quantity set and role-overlap result describe a fresh completed source tick.
/// This module validates the shape and cross-evidence identities; it does not
/// authenticate the dynamic source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReserveObservationV1 {
    pub observation_id: String,
    pub dynamic_handoff_receipt_binding: String,
    pub source_final_tick: u64,
    pub source_final_observation_tick: Option<u64>,
    pub construction_and_qualification_available: bool,
    /// Exact coverage of the reserve policy, sorted by transfer pair.
    pub quantities: Vec<RegenerativeBootstrapReserveQuantityV1>,
    pub evidence_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReserveDeficitV1 {
    pub source_dependency_id: String,
    pub successor_dependency_id: String,
    pub minimum_transfer_units: u64,
    pub observed_available_transfer_units: u64,
    pub missing_units: u64,
}

/// Diagnostic assessment of whether one evidence-bound role-overlap observation
/// preserves enough bootstrap reserve to support a successor handoff claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeBootstrapReserveAssessmentV1 {
    pub policy_id: String,
    pub handoff_id: String,
    pub observation_id: String,
    pub source_final_tick: u64,
    pub fresh_completed_observation: bool,
    pub construction_and_qualification_available: bool,
    pub bootstrap_reserve_satisfied: bool,
    pub reproduction_ready: bool,
    pub deficits: Vec<RegenerativeBootstrapReserveDeficitV1>,
    pub dynamic_handoff_receipt_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeBootstrapReserveError {
    UnsupportedSchemaVersion { schema_version: u8 },
    InvalidIdentifier,
    InvalidEvidenceBinding,
    NoReserveRequirements,
    TooManyItems,
    ZeroMinimumReserve { dependency_id: String },
    NonCanonicalRequirementOrder,
    NonCanonicalQuantityOrder,
    HandoffQualificationBindingMismatch,
    DynamicReceiptBindingMismatch,
    QuantityCoverageMismatch,
    RequirementNotInQualifiedHandoff {
        source_dependency_id: String,
        successor_dependency_id: String,
    },
    InvalidHandoff(RegenerativeEpochHandoffError),
}

/// Validate a bootstrap-reserve policy against the exact already-qualified epoch
/// handoff. This composes, rather than weakens, the handoff theorem.
pub fn validate_regenerative_bootstrap_reserve_policy(
    policy: &RegenerativeBootstrapReservePolicyV1,
    handoff: &RegenerativeEpochHandoffEvidenceV1,
    source_genome: &RegenerativeGenomeV1,
    source_model: &RegenerativeClosureModel,
    successor_genome: &RegenerativeGenomeV1,
    successor_model: &RegenerativeClosureModel,
) -> Result<(), RegenerativeBootstrapReserveError> {
    validate_policy_shape(policy)?;
    qualify_regenerative_epoch_handoff(
        handoff,
        source_genome,
        source_model,
        successor_genome,
        successor_model,
    )
    .map_err(RegenerativeBootstrapReserveError::InvalidHandoff)?;

    if policy.handoff_qualification_evidence_binding != handoff.evidence_binding {
        return Err(RegenerativeBootstrapReserveError::HandoffQualificationBindingMismatch);
    }

    let qualified_pairs: BTreeSet<(&str, &str)> = handoff
        .transfer_qualifications
        .iter()
        .map(|transfer| {
            (
                transfer.source_dependency_id.as_str(),
                transfer.successor_dependency_id.as_str(),
            )
        })
        .collect();
    for requirement in &policy.requirements {
        if !qualified_pairs.contains(&(
            requirement.source_dependency_id.as_str(),
            requirement.successor_dependency_id.as_str(),
        )) {
            return Err(
                RegenerativeBootstrapReserveError::RequirementNotInQualifiedHandoff {
                    source_dependency_id: requirement.source_dependency_id.clone(),
                    successor_dependency_id: requirement.successor_dependency_id.clone(),
                },
            );
        }
    }
    Ok(())
}

/// Assess upstream runtime quantity claims against a validated semantic reserve policy.
///
/// This function intentionally does not recompute inventory conservation. Exact
/// quantity truth remains owned by the bound dynamic handoff receipt.
pub fn assess_regenerative_bootstrap_reserve(
    policy: &RegenerativeBootstrapReservePolicyV1,
    handoff: &RegenerativeEpochHandoffEvidenceV1,
    observation: &RegenerativeBootstrapReserveObservationV1,
) -> Result<RegenerativeBootstrapReserveAssessmentV1, RegenerativeBootstrapReserveError> {
    validate_policy_shape(policy)?;
    validate_observation_shape(observation)?;
    if policy.handoff_qualification_evidence_binding != handoff.evidence_binding {
        return Err(RegenerativeBootstrapReserveError::HandoffQualificationBindingMismatch);
    }
    if observation.dynamic_handoff_receipt_binding != handoff.dynamic_handoff_receipt_binding {
        return Err(RegenerativeBootstrapReserveError::DynamicReceiptBindingMismatch);
    }

    let expected_pairs: BTreeSet<(&str, &str)> = policy
        .requirements
        .iter()
        .map(|requirement| {
            (
                requirement.source_dependency_id.as_str(),
                requirement.successor_dependency_id.as_str(),
            )
        })
        .collect();
    let observed_pairs: BTreeSet<(&str, &str)> = observation
        .quantities
        .iter()
        .map(|quantity| {
            (
                quantity.source_dependency_id.as_str(),
                quantity.successor_dependency_id.as_str(),
            )
        })
        .collect();
    if expected_pairs != observed_pairs {
        return Err(RegenerativeBootstrapReserveError::QuantityCoverageMismatch);
    }

    let observed_by_pair: BTreeMap<(&str, &str), u64> = observation
        .quantities
        .iter()
        .map(|quantity| {
            (
                (
                    quantity.source_dependency_id.as_str(),
                    quantity.successor_dependency_id.as_str(),
                ),
                quantity.available_transfer_units,
            )
        })
        .collect();

    let mut deficits = Vec::new();
    for requirement in &policy.requirements {
        let available = observed_by_pair[&(
            requirement.source_dependency_id.as_str(),
            requirement.successor_dependency_id.as_str(),
        )];
        if available < requirement.minimum_transfer_units {
            deficits.push(RegenerativeBootstrapReserveDeficitV1 {
                source_dependency_id: requirement.source_dependency_id.clone(),
                successor_dependency_id: requirement.successor_dependency_id.clone(),
                minimum_transfer_units: requirement.minimum_transfer_units,
                observed_available_transfer_units: available,
                missing_units: requirement.minimum_transfer_units - available,
            });
        }
    }

    let fresh_completed_observation =
        observation.source_final_observation_tick == Some(observation.source_final_tick);
    let bootstrap_reserve_satisfied = deficits.is_empty();
    let reproduction_ready = fresh_completed_observation
        && observation.construction_and_qualification_available
        && bootstrap_reserve_satisfied;

    Ok(RegenerativeBootstrapReserveAssessmentV1 {
        policy_id: policy.policy_id.clone(),
        handoff_id: handoff.handoff_id.clone(),
        observation_id: observation.observation_id.clone(),
        source_final_tick: observation.source_final_tick,
        fresh_completed_observation,
        construction_and_qualification_available: observation
            .construction_and_qualification_available,
        bootstrap_reserve_satisfied,
        reproduction_ready,
        deficits,
        dynamic_handoff_receipt_binding: observation.dynamic_handoff_receipt_binding.clone(),
    })
}

fn validate_policy_shape(
    policy: &RegenerativeBootstrapReservePolicyV1,
) -> Result<(), RegenerativeBootstrapReserveError> {
    if policy.schema_version != REGENERATIVE_BOOTSTRAP_RESERVE_SCHEMA_V1 {
        return Err(RegenerativeBootstrapReserveError::UnsupportedSchemaVersion {
            schema_version: policy.schema_version,
        });
    }
    validate_id(&policy.policy_id)?;
    validate_binding(&policy.handoff_qualification_evidence_binding)?;
    validate_binding(&policy.evidence_binding)?;
    if policy.requirements.is_empty() {
        return Err(RegenerativeBootstrapReserveError::NoReserveRequirements);
    }
    if policy.requirements.len() > MAX_ITEMS {
        return Err(RegenerativeBootstrapReserveError::TooManyItems);
    }
    for requirement in &policy.requirements {
        validate_id(&requirement.source_dependency_id)?;
        validate_id(&requirement.successor_dependency_id)?;
        validate_binding(&requirement.reserve_requirement_binding)?;
        if requirement.minimum_transfer_units == 0 {
            return Err(RegenerativeBootstrapReserveError::ZeroMinimumReserve {
                dependency_id: requirement.source_dependency_id.clone(),
            });
        }
    }
    if policy.requirements.windows(2).any(|pair| {
        (
            pair[0].source_dependency_id.as_str(),
            pair[0].successor_dependency_id.as_str(),
        ) >= (
            pair[1].source_dependency_id.as_str(),
            pair[1].successor_dependency_id.as_str(),
        )
    }) {
        return Err(RegenerativeBootstrapReserveError::NonCanonicalRequirementOrder);
    }
    Ok(())
}

fn validate_observation_shape(
    observation: &RegenerativeBootstrapReserveObservationV1,
) -> Result<(), RegenerativeBootstrapReserveError> {
    validate_id(&observation.observation_id)?;
    validate_binding(&observation.dynamic_handoff_receipt_binding)?;
    validate_binding(&observation.evidence_binding)?;
    if observation.quantities.len() > MAX_ITEMS {
        return Err(RegenerativeBootstrapReserveError::TooManyItems);
    }
    for quantity in &observation.quantities {
        validate_id(&quantity.source_dependency_id)?;
        validate_id(&quantity.successor_dependency_id)?;
    }
    if observation.quantities.windows(2).any(|pair| {
        (
            pair[0].source_dependency_id.as_str(),
            pair[0].successor_dependency_id.as_str(),
        ) >= (
            pair[1].source_dependency_id.as_str(),
            pair[1].successor_dependency_id.as_str(),
        )
    }) {
        return Err(RegenerativeBootstrapReserveError::NonCanonicalQuantityOrder);
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), RegenerativeBootstrapReserveError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeBootstrapReserveError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeBootstrapReserveError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeBootstrapReserveError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

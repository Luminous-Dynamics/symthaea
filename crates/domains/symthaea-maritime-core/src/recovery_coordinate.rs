// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic qualification of a disturbance-recovery coordinate.
//!
//! Nominal regenerative runway and role-support closure do not by themselves say
//! whether a degraded, previously qualified flow can be restored after a modeled
//! disturbance. This module qualifies that separate recovery coordinate without
//! changing the nominal closure theorem. V1 deliberately requires the recovery
//! reserve to sit outside the nominal successor-construction/qualification support
//! closure so using recovery does not silently redefine the scalar runway basis.

use crate::{
    DependencyGovernance, RegenerativeClosureModel, RegenerativeFlowKindV1,
    RegenerativeFlowSupportV1, RegenerativeLineageViabilityProfileV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REGENERATIVE_RECOVERY_COORDINATE_SCHEMA_V1: u8 = 1;
const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRecoveryCoordinatePolicyV1 {
    pub schema_version: u8,
    pub policy_id: String,
    pub evidence_binding: String,
    pub target_dependency_id: String,
    pub flow_kind: RegenerativeFlowKindV1,
    /// Exact healthy flow rate that may be restored after degradation.
    pub qualified_units_per_period: u64,
    pub reserve_dependency_id: String,
    pub reserve_units_per_recovery: u64,
    /// Evidence that recovery restores an already-qualified capability rather than
    /// introducing a new production/recycling capability after failure.
    pub recovery_qualification_binding: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeRecoveryCoordinateReportV1 {
    pub policy_id: String,
    pub policy_evidence_binding: String,
    pub profile_id: String,
    pub closure_model_id: String,
    pub closure_model_evidence_binding: String,
    pub flow_support_id: String,
    pub flow_support_evidence_binding: String,
    pub target_dependency_id: String,
    pub flow_kind: RegenerativeFlowKindV1,
    pub qualified_units_per_period: u64,
    pub reserve_dependency_id: String,
    pub reserve_units_per_recovery: u64,
    pub role_support_dependency_ids: Vec<String>,
    /// True because V1 requires the recovery reserve to remain outside the nominal
    /// construction/qualification support closure.
    pub reserve_outside_nominal_role_support_closure: bool,
    /// Authorization bit for disturbance-conditioned claims that explicitly bind
    /// this recovery coordinate. It does not alter nominal scalar H.
    pub disturbance_recovery_coordinate_qualified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDisturbanceContextV1 {
    pub disturbance_id: String,
    pub evidence_binding: String,
    pub target_dependency_id: String,
    pub flow_kind: RegenerativeFlowKindV1,
    pub degraded_units_per_period: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeDisturbanceRecoveryAuthorizationV1 {
    pub disturbance_id: String,
    pub disturbance_evidence_binding: String,
    pub recovery_policy_id: String,
    pub recovery_policy_evidence_binding: String,
    pub closure_model_id: String,
    pub flow_support_id: String,
    pub target_dependency_id: String,
    pub flow_kind: RegenerativeFlowKindV1,
    pub degraded_units_per_period: u64,
    pub qualified_restore_units_per_period: u64,
    pub disturbance_conditioned_recovery_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativeRecoveryCoordinateError {
    UnsupportedSchemaVersion { schema_version: u8 },
    InvalidIdentifier,
    InvalidEvidenceBinding,
    ProfileInvalid,
    SupportInvalid,
    ZeroQualifiedFlow,
    ZeroReserveCost,
    TargetAndReserveNotDistinct,
    UnknownTargetDependency { dependency_id: String },
    UnknownReserveDependency { dependency_id: String },
    SafeguardedTargetClaimsLocalRecovery { dependency_id: String },
    SafeguardedReserveClaimsLocalRecovery { dependency_id: String },
    TargetOutsideRoleSupportClosure { dependency_id: String },
    RecoveryReserveInsideRoleSupportClosure { dependency_id: String },
    TargetFlowRateMismatch {
        dependency_id: String,
        modeled_units_per_period: u64,
        qualified_units_per_period: u64,
    },
    TargetFlowNotSupportQualified {
        dependency_id: String,
        flow_kind: RegenerativeFlowKindV1,
    },
    StaticRecoveryReserveInsufficient {
        dependency_id: String,
        required_units: u64,
        stockpile_units: u64,
    },
    DisturbanceInvalid,
    DisturbanceRecoverySubjectMismatch,
    DisturbanceDoesNotDegradeQualifiedFlow,
}

/// Qualify one explicit recovery coordinate against the exact healthy successor
/// model and support graph.
pub fn qualify_regenerative_recovery_coordinate(
    policy: &RegenerativeRecoveryCoordinatePolicyV1,
    profile: &RegenerativeLineageViabilityProfileV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> Result<RegenerativeRecoveryCoordinateReportV1, RegenerativeRecoveryCoordinateError> {
    validate_policy(policy)?;
    profile
        .validate_against_model(model)
        .map_err(|_| RegenerativeRecoveryCoordinateError::ProfileInvalid)?;
    support
        .validate_against_model(model)
        .map_err(|_| RegenerativeRecoveryCoordinateError::SupportInvalid)?;

    let role_support = role_support_closure(profile, model, support);
    if !role_support.contains(&policy.target_dependency_id) {
        return Err(RegenerativeRecoveryCoordinateError::TargetOutsideRoleSupportClosure {
            dependency_id: policy.target_dependency_id.clone(),
        });
    }
    if role_support.contains(&policy.reserve_dependency_id) {
        return Err(
            RegenerativeRecoveryCoordinateError::RecoveryReserveInsideRoleSupportClosure {
                dependency_id: policy.reserve_dependency_id.clone(),
            },
        );
    }

    let target = model
        .dependencies
        .iter()
        .find(|dependency| dependency.dependency_id == policy.target_dependency_id)
        .ok_or_else(|| RegenerativeRecoveryCoordinateError::UnknownTargetDependency {
            dependency_id: policy.target_dependency_id.clone(),
        })?;
    let reserve = model
        .dependencies
        .iter()
        .find(|dependency| dependency.dependency_id == policy.reserve_dependency_id)
        .ok_or_else(|| RegenerativeRecoveryCoordinateError::UnknownReserveDependency {
            dependency_id: policy.reserve_dependency_id.clone(),
        })?;
    if target.governance == DependencyGovernance::SafeguardedExternal {
        return Err(
            RegenerativeRecoveryCoordinateError::SafeguardedTargetClaimsLocalRecovery {
                dependency_id: target.dependency_id.clone(),
            },
        );
    }
    if reserve.governance == DependencyGovernance::SafeguardedExternal {
        return Err(
            RegenerativeRecoveryCoordinateError::SafeguardedReserveClaimsLocalRecovery {
                dependency_id: reserve.dependency_id.clone(),
            },
        );
    }

    let modeled_units_per_period = match policy.flow_kind {
        RegenerativeFlowKindV1::Production => target.local_production_units_per_period,
        RegenerativeFlowKindV1::Recycling => target.recycling_units_per_period,
    };
    if modeled_units_per_period != policy.qualified_units_per_period {
        return Err(RegenerativeRecoveryCoordinateError::TargetFlowRateMismatch {
            dependency_id: target.dependency_id.clone(),
            modeled_units_per_period,
            qualified_units_per_period: policy.qualified_units_per_period,
        });
    }
    if !support.claims.iter().any(|claim| {
        claim.dependency_id == policy.target_dependency_id && claim.flow_kind == policy.flow_kind
    }) {
        return Err(RegenerativeRecoveryCoordinateError::TargetFlowNotSupportQualified {
            dependency_id: policy.target_dependency_id.clone(),
            flow_kind: policy.flow_kind,
        });
    }
    if reserve.stockpile_units < policy.reserve_units_per_recovery {
        return Err(
            RegenerativeRecoveryCoordinateError::StaticRecoveryReserveInsufficient {
                dependency_id: reserve.dependency_id.clone(),
                required_units: policy.reserve_units_per_recovery,
                stockpile_units: reserve.stockpile_units,
            },
        );
    }

    Ok(RegenerativeRecoveryCoordinateReportV1 {
        policy_id: policy.policy_id.clone(),
        policy_evidence_binding: policy.evidence_binding.clone(),
        profile_id: profile.profile_id.clone(),
        closure_model_id: model.model_id.clone(),
        closure_model_evidence_binding: model.evidence_binding.clone(),
        flow_support_id: support.support_id.clone(),
        flow_support_evidence_binding: support.evidence_binding.clone(),
        target_dependency_id: policy.target_dependency_id.clone(),
        flow_kind: policy.flow_kind,
        qualified_units_per_period: policy.qualified_units_per_period,
        reserve_dependency_id: policy.reserve_dependency_id.clone(),
        reserve_units_per_recovery: policy.reserve_units_per_recovery,
        role_support_dependency_ids: role_support.into_iter().collect(),
        reserve_outside_nominal_role_support_closure: true,
        disturbance_recovery_coordinate_qualified: true,
    })
}

/// Bind an explicit disturbance to a separately qualified recovery coordinate.
/// This authorizes only a disturbance-conditioned recovery claim; it does not
/// modify nominal lineage horizons or authorize any physical action.
pub fn authorize_regenerative_disturbance_recovery(
    disturbance: &RegenerativeDisturbanceContextV1,
    recovery: &RegenerativeRecoveryCoordinateReportV1,
) -> Result<RegenerativeDisturbanceRecoveryAuthorizationV1, RegenerativeRecoveryCoordinateError> {
    validate_id(&disturbance.disturbance_id)?;
    validate_binding(&disturbance.evidence_binding)?;
    validate_id(&disturbance.target_dependency_id)?;
    if !recovery.disturbance_recovery_coordinate_qualified
        || disturbance.target_dependency_id != recovery.target_dependency_id
        || disturbance.flow_kind != recovery.flow_kind
    {
        return Err(RegenerativeRecoveryCoordinateError::DisturbanceRecoverySubjectMismatch);
    }
    if disturbance.degraded_units_per_period >= recovery.qualified_units_per_period {
        return Err(RegenerativeRecoveryCoordinateError::DisturbanceDoesNotDegradeQualifiedFlow);
    }

    Ok(RegenerativeDisturbanceRecoveryAuthorizationV1 {
        disturbance_id: disturbance.disturbance_id.clone(),
        disturbance_evidence_binding: disturbance.evidence_binding.clone(),
        recovery_policy_id: recovery.policy_id.clone(),
        recovery_policy_evidence_binding: recovery.policy_evidence_binding.clone(),
        closure_model_id: recovery.closure_model_id.clone(),
        flow_support_id: recovery.flow_support_id.clone(),
        target_dependency_id: recovery.target_dependency_id.clone(),
        flow_kind: recovery.flow_kind,
        degraded_units_per_period: disturbance.degraded_units_per_period,
        qualified_restore_units_per_period: recovery.qualified_units_per_period,
        disturbance_conditioned_recovery_authorized: true,
    })
}

fn role_support_closure(
    profile: &RegenerativeLineageViabilityProfileV1,
    model: &RegenerativeClosureModel,
    support: &RegenerativeFlowSupportV1,
) -> BTreeSet<String> {
    let selected_capabilities: BTreeSet<&str> = profile
        .successor_construction_capability_ids
        .iter()
        .chain(profile.successor_qualification_capability_ids.iter())
        .map(String::as_str)
        .collect();
    let mut closure = BTreeSet::new();
    for capability in &model.capabilities {
        if selected_capabilities.contains(capability.capability_id.as_str()) {
            closure.extend(capability.dependency_ids.iter().cloned());
        }
    }
    for _ in 0..model.dependencies.len().max(1) {
        let before = closure.len();
        for claim in &support.claims {
            if closure.contains(&claim.dependency_id) {
                closure.extend(claim.prerequisite_dependency_ids.iter().cloned());
            }
        }
        if closure.len() == before {
            break;
        }
    }
    closure
}

fn validate_policy(
    policy: &RegenerativeRecoveryCoordinatePolicyV1,
) -> Result<(), RegenerativeRecoveryCoordinateError> {
    if policy.schema_version != REGENERATIVE_RECOVERY_COORDINATE_SCHEMA_V1 {
        return Err(RegenerativeRecoveryCoordinateError::UnsupportedSchemaVersion {
            schema_version: policy.schema_version,
        });
    }
    validate_id(&policy.policy_id)?;
    validate_binding(&policy.evidence_binding)?;
    validate_id(&policy.target_dependency_id)?;
    validate_id(&policy.reserve_dependency_id)?;
    validate_binding(&policy.recovery_qualification_binding)?;
    if policy.qualified_units_per_period == 0 {
        return Err(RegenerativeRecoveryCoordinateError::ZeroQualifiedFlow);
    }
    if policy.reserve_units_per_recovery == 0 {
        return Err(RegenerativeRecoveryCoordinateError::ZeroReserveCost);
    }
    if policy.target_dependency_id == policy.reserve_dependency_id {
        return Err(RegenerativeRecoveryCoordinateError::TargetAndReserveNotDistinct);
    }
    Ok(())
}

fn validate_id(value: &str) -> Result<(), RegenerativeRecoveryCoordinateError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeRecoveryCoordinateError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativeRecoveryCoordinateError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativeRecoveryCoordinateError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

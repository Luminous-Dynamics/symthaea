// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-sensitivity surfaces for regenerative lineage viability.
//!
//! A policy family varies normative maturity requirements while holding one exact
//! physical support-qualified lineage state fixed. This module makes that causal
//! separation explicit and fails closed if policy points are non-canonical,
//! disagree about the physical temporal runway, or improve descendant outcomes as
//! maturity requirements become stricter.

use crate::{
    evaluate_policy_normalized_lineage_profile, RegenerativeGenerationCountV1,
    RegenerativeGenerationPolicyProjectionV1, RegenerativeHorizon,
    RegenerativeLineageViabilityReportV1, RegenerativePolicyNormalizedLineageError,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_POLICY_POINTS: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativeReproductionPolicySpecV1 {
    pub reproduction_policy_id: String,
    pub reproduction_policy_evidence_binding: String,
    pub maturity_periods: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativePolicySensitivityPointV1 {
    pub reproduction_policy_id: String,
    pub reproduction_policy_evidence_binding: String,
    pub generation_policy_projection: RegenerativeGenerationPolicyProjectionV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativePolicySensitivitySurfaceV1 {
    pub source_profile_id: String,
    pub genome_id: String,
    pub closure_model_id: String,
    pub flow_support_id: String,
    /// One invariant physical coordinate shared by every policy point.
    pub physical_successor_reproduction_horizon: RegenerativeHorizon,
    pub physical_regenerative_viability_horizon: RegenerativeHorizon,
    pub fully_modeled_successor_reproduction: bool,
    pub fully_modeled_regenerative_viability: bool,
    /// Strictly increasing by maturity periods, with unique policy IDs and bindings.
    pub policy_points: Vec<RegenerativePolicySensitivityPointV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativePolicySensitivityError {
    TooFewPolicies,
    TooManyPolicies,
    NonCanonicalPolicyOrder,
    DuplicatePolicyIdentity,
    DuplicatePolicyEvidenceBinding,
    PhysicalCoordinateMismatch,
    NonMonotoneFoundedDescendants,
    NonMonotoneMaturityCompletedDescendants,
    NonMonotoneReproductionTransitions,
    NormalizedProfile(RegenerativePolicyNormalizedLineageError),
}

pub fn evaluate_regenerative_policy_sensitivity_surface(
    report: &RegenerativeLineageViabilityReportV1,
    policies: &[RegenerativeReproductionPolicySpecV1],
) -> Result<RegenerativePolicySensitivitySurfaceV1, RegenerativePolicySensitivityError> {
    if policies.len() < 2 {
        return Err(RegenerativePolicySensitivityError::TooFewPolicies);
    }
    if policies.len() > MAX_POLICY_POINTS {
        return Err(RegenerativePolicySensitivityError::TooManyPolicies);
    }
    if policies
        .windows(2)
        .any(|pair| pair[0].maturity_periods >= pair[1].maturity_periods)
    {
        return Err(RegenerativePolicySensitivityError::NonCanonicalPolicyOrder);
    }

    let mut policy_ids = BTreeSet::new();
    let mut evidence_bindings = BTreeSet::new();
    for policy in policies {
        if !policy_ids.insert(policy.reproduction_policy_id.as_str()) {
            return Err(RegenerativePolicySensitivityError::DuplicatePolicyIdentity);
        }
        if !evidence_bindings.insert(policy.reproduction_policy_evidence_binding.as_str()) {
            return Err(RegenerativePolicySensitivityError::DuplicatePolicyEvidenceBinding);
        }
    }

    let mut policy_points = Vec::with_capacity(policies.len());
    let mut physical_successor_reproduction_horizon = None;
    let mut physical_regenerative_viability_horizon = None;
    let mut fully_modeled_successor_reproduction = None;
    let mut fully_modeled_regenerative_viability = None;

    for policy in policies {
        let normalized = evaluate_policy_normalized_lineage_profile(
            report,
            policy.reproduction_policy_id.clone(),
            policy.reproduction_policy_evidence_binding.clone(),
            policy.maturity_periods,
        )
        .map_err(RegenerativePolicySensitivityError::NormalizedProfile)?;

        match physical_successor_reproduction_horizon {
            None => {
                physical_successor_reproduction_horizon =
                    Some(normalized.physical_successor_reproduction_horizon);
                physical_regenerative_viability_horizon =
                    Some(normalized.physical_regenerative_viability_horizon);
                fully_modeled_successor_reproduction =
                    Some(normalized.fully_modeled_successor_reproduction);
                fully_modeled_regenerative_viability =
                    Some(normalized.fully_modeled_regenerative_viability);
            }
            Some(expected)
                if expected != normalized.physical_successor_reproduction_horizon
                    || physical_regenerative_viability_horizon
                        != Some(normalized.physical_regenerative_viability_horizon)
                    || fully_modeled_successor_reproduction
                        != Some(normalized.fully_modeled_successor_reproduction)
                    || fully_modeled_regenerative_viability
                        != Some(normalized.fully_modeled_regenerative_viability) =>
            {
                return Err(RegenerativePolicySensitivityError::PhysicalCoordinateMismatch)
            }
            Some(_) => {}
        }

        policy_points.push(RegenerativePolicySensitivityPointV1 {
            reproduction_policy_id: normalized.reproduction_policy_id,
            reproduction_policy_evidence_binding: normalized.reproduction_policy_evidence_binding,
            generation_policy_projection: normalized.generation_policy_projection,
        });
    }

    let surface = RegenerativePolicySensitivitySurfaceV1 {
        source_profile_id: report.profile_id.clone(),
        genome_id: report.genome_id.clone(),
        closure_model_id: report.closure_model_id.clone(),
        flow_support_id: report.flow_support_id.clone(),
        physical_successor_reproduction_horizon:
            physical_successor_reproduction_horizon.expect("policy family is non-empty"),
        physical_regenerative_viability_horizon:
            physical_regenerative_viability_horizon.expect("policy family is non-empty"),
        fully_modeled_successor_reproduction:
            fully_modeled_successor_reproduction.expect("policy family is non-empty"),
        fully_modeled_regenerative_viability:
            fully_modeled_regenerative_viability.expect("policy family is non-empty"),
        policy_points,
    };
    validate_regenerative_policy_sensitivity_surface(&surface)?;
    Ok(surface)
}

pub fn validate_regenerative_policy_sensitivity_surface(
    surface: &RegenerativePolicySensitivitySurfaceV1,
) -> Result<(), RegenerativePolicySensitivityError> {
    if surface.policy_points.len() < 2 {
        return Err(RegenerativePolicySensitivityError::TooFewPolicies);
    }
    if surface.policy_points.len() > MAX_POLICY_POINTS {
        return Err(RegenerativePolicySensitivityError::TooManyPolicies);
    }

    let mut policy_ids = BTreeSet::new();
    let mut evidence_bindings = BTreeSet::new();
    let mut previous_maturity = None;
    let mut previous_counts: Option<(u64, u64, u64)> = None;

    for point in &surface.policy_points {
        if !policy_ids.insert(point.reproduction_policy_id.as_str()) {
            return Err(RegenerativePolicySensitivityError::DuplicatePolicyIdentity);
        }
        if !evidence_bindings.insert(point.reproduction_policy_evidence_binding.as_str()) {
            return Err(RegenerativePolicySensitivityError::DuplicatePolicyEvidenceBinding);
        }

        let projection = &point.generation_policy_projection;
        if projection.temporal_successor_horizon
            != surface.physical_successor_reproduction_horizon
        {
            return Err(RegenerativePolicySensitivityError::PhysicalCoordinateMismatch);
        }
        if previous_maturity.is_some_and(|previous| previous >= projection.maturity_periods) {
            return Err(RegenerativePolicySensitivityError::NonCanonicalPolicyOrder);
        }
        previous_maturity = Some(projection.maturity_periods);

        if let Some(current_counts) = finite_counts(projection) {
            if let Some((previous_founded, previous_matured, previous_transitions)) = previous_counts {
                if current_counts.0 > previous_founded {
                    return Err(RegenerativePolicySensitivityError::NonMonotoneFoundedDescendants);
                }
                if current_counts.1 > previous_matured {
                    return Err(
                        RegenerativePolicySensitivityError::NonMonotoneMaturityCompletedDescendants,
                    );
                }
                if current_counts.2 > previous_transitions {
                    return Err(
                        RegenerativePolicySensitivityError::NonMonotoneReproductionTransitions,
                    );
                }
            }
            previous_counts = Some(current_counts);
        } else {
            previous_counts = None;
        }
    }
    Ok(())
}

fn finite_counts(projection: &RegenerativeGenerationPolicyProjectionV1) -> Option<(u64, u64, u64)> {
    match (
        projection.founded_descendant_generations,
        projection.maturity_completed_descendant_generations,
        projection.descendant_reproduction_transitions,
    ) {
        (
            RegenerativeGenerationCountV1::Finite(founded),
            RegenerativeGenerationCountV1::Finite(matured),
            RegenerativeGenerationCountV1::Finite(transitions),
        ) => Some((founded, matured, transitions)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RegenerativeLineageRoleReportV1, RegenerativeLineageRoleV1};

    fn role(
        role: RegenerativeLineageRoleV1,
        horizon: RegenerativeHorizon,
    ) -> RegenerativeLineageRoleReportV1 {
        RegenerativeLineageRoleReportV1 {
            role,
            conservative_horizon: horizon,
            limiting_requirement_ids: vec![format!("requirement:{role:?}")],
            root_limiting_dependency_ids: vec![format!("dependency:{role:?}")],
            fully_modeled_support: true,
            externally_conditioned_requirement_ids: Vec::new(),
        }
    }

    fn report() -> RegenerativeLineageViabilityReportV1 {
        RegenerativeLineageViabilityReportV1 {
            profile_id: "profile:manta-policy-surface".into(),
            genome_id: "genome:manta-policy-surface".into(),
            closure_model_id: "closure:manta-policy-surface".into(),
            flow_support_id: "support:manta-policy-surface".into(),
            operation: role(
                RegenerativeLineageRoleV1::Operation,
                RegenerativeHorizon::FinitePeriods(8),
            ),
            successor_construction: role(
                RegenerativeLineageRoleV1::SuccessorConstruction,
                RegenerativeHorizon::FinitePeriods(4),
            ),
            successor_qualification: role(
                RegenerativeLineageRoleV1::SuccessorQualification,
                RegenerativeHorizon::FinitePeriods(5),
            ),
            successor_reproduction_horizon: RegenerativeHorizon::FinitePeriods(4),
            regenerative_viability_horizon: RegenerativeHorizon::FinitePeriods(4),
            limiting_roles: vec![RegenerativeLineageRoleV1::SuccessorConstruction],
            fully_modeled_regenerative_viability: true,
        }
    }

    fn policies() -> Vec<RegenerativeReproductionPolicySpecV1> {
        (1..=4)
            .map(|maturity_periods| RegenerativeReproductionPolicySpecV1 {
                reproduction_policy_id: format!("policy-m{maturity_periods}"),
                reproduction_policy_evidence_binding: format!("policy:m{maturity_periods}:v1"),
                maturity_periods,
            })
            .collect()
    }

    #[test]
    fn physical_coordinate_stays_fixed_while_policy_outcomes_contract() {
        let surface = evaluate_regenerative_policy_sensitivity_surface(&report(), &policies()).unwrap();
        assert_eq!(
            surface.physical_successor_reproduction_horizon,
            RegenerativeHorizon::FinitePeriods(4)
        );
        let counts: Vec<(u64, u64, u64)> = surface
            .policy_points
            .iter()
            .map(|point| finite_counts(&point.generation_policy_projection).unwrap())
            .collect();
        assert_eq!(counts, vec![(4, 4, 3), (2, 2, 1), (2, 1, 1), (1, 1, 0)]);
    }

    #[test]
    fn noncanonical_policy_family_fails_closed() {
        let mut duplicate_maturity = policies();
        duplicate_maturity[1].maturity_periods = 1;
        assert_eq!(
            evaluate_regenerative_policy_sensitivity_surface(&report(), &duplicate_maturity),
            Err(RegenerativePolicySensitivityError::NonCanonicalPolicyOrder)
        );

        let mut duplicate_id = policies();
        duplicate_id[1].reproduction_policy_id = duplicate_id[0].reproduction_policy_id.clone();
        assert_eq!(
            evaluate_regenerative_policy_sensitivity_surface(&report(), &duplicate_id),
            Err(RegenerativePolicySensitivityError::DuplicatePolicyIdentity)
        );
    }

    #[test]
    fn deserialized_surface_cannot_change_physics_or_improve_under_stricter_policy() {
        let mut surface = evaluate_regenerative_policy_sensitivity_surface(&report(), &policies()).unwrap();
        surface.policy_points[2].generation_policy_projection.temporal_successor_horizon =
            RegenerativeHorizon::FinitePeriods(5);
        assert_eq!(
            validate_regenerative_policy_sensitivity_surface(&surface),
            Err(RegenerativePolicySensitivityError::PhysicalCoordinateMismatch)
        );

        let mut surface = evaluate_regenerative_policy_sensitivity_surface(&report(), &policies()).unwrap();
        surface.policy_points[2]
            .generation_policy_projection
            .founded_descendant_generations = RegenerativeGenerationCountV1::Finite(3);
        assert_eq!(
            validate_regenerative_policy_sensitivity_surface(&surface),
            Err(RegenerativePolicySensitivityError::NonMonotoneFoundedDescendants)
        );
    }
}

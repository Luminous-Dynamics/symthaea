// SPDX-License-Identifier: AGPL-3.0-or-later
//! Policy-normalized lineage reporting.
//!
//! Physical support-qualified runway and policy-dependent descendant-generation
//! semantics answer different questions. This module carries them side by side so
//! changing a maturity policy cannot masquerade as a change in physical closure.
//! It is diagnostic evidence only and grants no manufacturing, qualification,
//! operating, procurement, or physical-control authority.

use crate::{
    project_regenerative_generation_policy, RegenerativeGenerationPolicyError,
    RegenerativeGenerationPolicyProjectionV1, RegenerativeHorizon,
    RegenerativeLineageViabilityReportV1,
};
use serde::{Deserialize, Serialize};

const MAX_ID_LEN: usize = 256;
const MAX_BINDING_LEN: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegenerativePolicyNormalizedLineageProfileV1 {
    /// Exact source lineage-viability profile identity.
    pub source_profile_id: String,
    pub genome_id: String,
    pub closure_model_id: String,
    pub flow_support_id: String,
    /// Invariant physical/static quantity: latest conservative period where both
    /// successor construction and successor qualification remain supported.
    pub physical_successor_reproduction_horizon: RegenerativeHorizon,
    /// Invariant physical/static quantity including current-platform operation.
    pub physical_regenerative_viability_horizon: RegenerativeHorizon,
    /// Whether construction + qualification support paths are fully modeled rather
    /// than reaching opaque external inputs.
    pub fully_modeled_successor_reproduction: bool,
    /// Whether operation + construction + qualification are all fully modeled.
    pub fully_modeled_regenerative_viability: bool,
    /// Canonical policy identity and evidence/version binding.
    pub reproduction_policy_id: String,
    pub reproduction_policy_evidence_binding: String,
    /// Policy-specific projection derived only from the physical successor
    /// reproduction horizon above.
    pub generation_policy_projection: RegenerativeGenerationPolicyProjectionV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegenerativePolicyNormalizedLineageError {
    InvalidIdentifier,
    InvalidEvidenceBinding,
    GenerationPolicy(RegenerativeGenerationPolicyError),
}

pub fn evaluate_policy_normalized_lineage_profile(
    report: &RegenerativeLineageViabilityReportV1,
    reproduction_policy_id: impl Into<String>,
    reproduction_policy_evidence_binding: impl Into<String>,
    maturity_periods: u64,
) -> Result<RegenerativePolicyNormalizedLineageProfileV1, RegenerativePolicyNormalizedLineageError>
{
    let reproduction_policy_id = reproduction_policy_id.into();
    let reproduction_policy_evidence_binding = reproduction_policy_evidence_binding.into();
    validate_id(&reproduction_policy_id)?;
    validate_binding(&reproduction_policy_evidence_binding)?;

    let generation_policy_projection = project_regenerative_generation_policy(
        report.successor_reproduction_horizon,
        maturity_periods,
    )
    .map_err(RegenerativePolicyNormalizedLineageError::GenerationPolicy)?;

    Ok(RegenerativePolicyNormalizedLineageProfileV1 {
        source_profile_id: report.profile_id.clone(),
        genome_id: report.genome_id.clone(),
        closure_model_id: report.closure_model_id.clone(),
        flow_support_id: report.flow_support_id.clone(),
        physical_successor_reproduction_horizon: report.successor_reproduction_horizon,
        physical_regenerative_viability_horizon: report.regenerative_viability_horizon,
        fully_modeled_successor_reproduction: report.successor_construction.fully_modeled_support
            && report.successor_qualification.fully_modeled_support,
        fully_modeled_regenerative_viability: report.fully_modeled_regenerative_viability,
        reproduction_policy_id,
        reproduction_policy_evidence_binding,
        generation_policy_projection,
    })
}

fn validate_id(value: &str) -> Result<(), RegenerativePolicyNormalizedLineageError> {
    if value.is_empty()
        || value.len() > MAX_ID_LEN
        || value.trim() != value
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativePolicyNormalizedLineageError::InvalidIdentifier)
    } else {
        Ok(())
    }
}

fn validate_binding(value: &str) -> Result<(), RegenerativePolicyNormalizedLineageError> {
    if value.is_empty()
        || value.len() > MAX_BINDING_LEN
        || value.trim() != value
        || !value.contains(':')
        || value.chars().any(char::is_whitespace)
        || value.chars().any(char::is_control)
    {
        Err(RegenerativePolicyNormalizedLineageError::InvalidEvidenceBinding)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RegenerativeGenerationCountV1, RegenerativeLineageRoleReportV1,
        RegenerativeLineageRoleV1,
    };

    fn role(
        role: RegenerativeLineageRoleV1,
        horizon: RegenerativeHorizon,
        fully_modeled_support: bool,
    ) -> RegenerativeLineageRoleReportV1 {
        RegenerativeLineageRoleReportV1 {
            role,
            conservative_horizon: horizon,
            limiting_requirement_ids: vec![format!("requirement:{role:?}")],
            root_limiting_dependency_ids: vec![format!("dependency:{role:?}")],
            fully_modeled_support,
            externally_conditioned_requirement_ids: if fully_modeled_support {
                Vec::new()
            } else {
                vec![format!("requirement:{role:?}")]
            },
        }
    }

    fn report(fully_modeled_qualification: bool) -> RegenerativeLineageViabilityReportV1 {
        RegenerativeLineageViabilityReportV1 {
            profile_id: "profile:manta-policy-normalized".into(),
            genome_id: "genome:manta-policy-normalized".into(),
            closure_model_id: "closure:manta-policy-normalized".into(),
            flow_support_id: "support:manta-policy-normalized".into(),
            operation: role(
                RegenerativeLineageRoleV1::Operation,
                RegenerativeHorizon::FinitePeriods(8),
                true,
            ),
            successor_construction: role(
                RegenerativeLineageRoleV1::SuccessorConstruction,
                RegenerativeHorizon::FinitePeriods(4),
                true,
            ),
            successor_qualification: role(
                RegenerativeLineageRoleV1::SuccessorQualification,
                RegenerativeHorizon::FinitePeriods(4),
                fully_modeled_qualification,
            ),
            successor_reproduction_horizon: RegenerativeHorizon::FinitePeriods(4),
            regenerative_viability_horizon: RegenerativeHorizon::FinitePeriods(4),
            limiting_roles: vec![
                RegenerativeLineageRoleV1::SuccessorConstruction,
                RegenerativeLineageRoleV1::SuccessorQualification,
            ],
            fully_modeled_regenerative_viability: fully_modeled_qualification,
        }
    }

    fn finite(value: RegenerativeGenerationCountV1) -> u64 {
        match value {
            RegenerativeGenerationCountV1::Finite(value) => value,
            RegenerativeGenerationCountV1::IndefiniteUnderStaticModel => {
                panic!("expected finite policy projection")
            }
        }
    }

    #[test]
    fn policy_changes_projection_not_physical_runway() {
        let source = report(true);
        let one = evaluate_policy_normalized_lineage_profile(
            &source,
            "policy-m1",
            "policy:m1:v1",
            1,
        )
        .unwrap();
        let two = evaluate_policy_normalized_lineage_profile(
            &source,
            "policy-m2",
            "policy:m2:v1",
            2,
        )
        .unwrap();
        let four = evaluate_policy_normalized_lineage_profile(
            &source,
            "policy-m4",
            "policy:m4:v1",
            4,
        )
        .unwrap();

        for profile in [&one, &two, &four] {
            assert_eq!(
                profile.physical_successor_reproduction_horizon,
                RegenerativeHorizon::FinitePeriods(4)
            );
            assert_eq!(
                profile.physical_regenerative_viability_horizon,
                RegenerativeHorizon::FinitePeriods(4)
            );
        }
        assert_eq!(
            finite(one.generation_policy_projection.founded_descendant_generations),
            4
        );
        assert_eq!(
            finite(two.generation_policy_projection.founded_descendant_generations),
            2
        );
        assert_eq!(
            finite(four.generation_policy_projection.founded_descendant_generations),
            1
        );
    }

    #[test]
    fn external_conditioning_stays_visible_in_policy_projection() {
        let source = report(false);
        let normalized = evaluate_policy_normalized_lineage_profile(
            &source,
            "policy-m2",
            "policy:m2:v1",
            2,
        )
        .unwrap();
        assert!(!normalized.fully_modeled_successor_reproduction);
        assert!(!normalized.fully_modeled_regenerative_viability);
        assert_eq!(
            normalized.physical_successor_reproduction_horizon,
            RegenerativeHorizon::FinitePeriods(4)
        );
    }

    #[test]
    fn invalid_policy_identity_and_zero_maturity_fail_closed() {
        let source = report(true);
        assert_eq!(
            evaluate_policy_normalized_lineage_profile(
                &source,
                "bad policy",
                "policy:bad:v1",
                1,
            ),
            Err(RegenerativePolicyNormalizedLineageError::InvalidIdentifier)
        );
        assert!(matches!(
            evaluate_policy_normalized_lineage_profile(
                &source,
                "policy-zero",
                "policy:zero:v1",
                0,
            ),
            Err(RegenerativePolicyNormalizedLineageError::GenerationPolicy(
                RegenerativeGenerationPolicyError::ZeroMaturityPeriods
            ))
        ));
    }
}

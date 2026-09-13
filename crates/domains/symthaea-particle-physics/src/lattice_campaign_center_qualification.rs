// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Campaign-bound preregistration and evidence binding for pure-SU(3) `Z_3`
//! center-sector mobility.
//!
//! LQCD-019B/019D established empirically that center-invariant Polyakov
//! convergence and categorical center-sector mobility are distinct evidence
//! dimensions. This module adds that newly discovered requirement without
//! retroactively rewriting the already-frozen LQCD-018H qualification policy.
//!
//! A supplement is valid only for one exact frozen campaign manifest and must be
//! frozen before production authorization. Post-run evidence must cover every
//! planned chain with exactly the planned retained sample count.

use std::collections::BTreeSet;

use crate::lattice_campaign_authority::{
    CampaignExecutionAuthorization, QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE,
};
use crate::lattice_campaign_manifest::PureSu3CampaignManifest;
use crate::lattice_center_symmetry::{
    CenterSectorMobilityAssessment, CenterSectorMobilityPolicy, Z3_CENTER_DIAGNOSTIC_ID,
};

pub const CAMPAIGN_CENTER_QUALIFICATION_SUPPLEMENT_ID: &str =
    "campaign_z3_center_qualification_supplement_v1";
pub const CAMPAIGN_CENTER_QUALIFICATION_EVIDENCE_ID: &str =
    "campaign_z3_center_qualification_evidence_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignCenterQualificationSupplement {
    pub supplement_id: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub policy: CenterSectorMobilityPolicy,
    pub freeze_evidence_id: String,
    pub frozen_policy_artifact_digest: String,
    pub frozen_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AuthorizedCampaignCenterQualification {
    pub supplement_id: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub policy: CenterSectorMobilityPolicy,
    pub frozen_policy_artifact_digest: String,
    pub frozen_at_unix_ns: u128,
    pub campaign_authorized_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundCampaignCenterQualificationEvidence {
    pub evidence_id: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub diagnostic_id: &'static str,
    pub planned_chain_count: usize,
    pub planned_measurements_per_chain: usize,
    pub policy: CenterSectorMobilityPolicy,
    pub center_assessment_artifact_digest: String,
    pub every_chain_meets_classified_fraction: bool,
    pub every_chain_meets_transition_count: bool,
    pub meets_declared_policy: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CampaignCenterQualificationError {
    WrongSupplementId,
    WrongAuthorizationScope,
    CampaignMismatch,
    ManifestDigestMismatch,
    PolyakovMeasurementNotPlanned,
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    InvalidFrozenTimestamp,
    SupplementNotFrozenBeforeAuthorization,
    InvalidMinimumMagnitude(f64),
    InvalidMinimumClassifiedFraction(f64),
    WrongDiagnosticId,
    PolicyMismatch,
    DuplicateAssessmentChain(String),
    MissingAssessmentChain(String),
    UnexpectedAssessmentChain(String),
    SampleCountMismatch { chain_id: String, expected: usize, actual: usize },
    InvalidChainDiagnostics { chain_id: String },
    AssessmentBooleanMismatch,
}

fn require_nonempty(
    value: &str,
    field: &'static str,
) -> Result<(), CampaignCenterQualificationError> {
    if value.trim().is_empty() {
        Err(CampaignCenterQualificationError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(
    value: &str,
    field: &'static str,
) -> Result<(), CampaignCenterQualificationError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(CampaignCenterQualificationError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CampaignCenterQualificationError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn policy_matches(left: &CenterSectorMobilityPolicy, right: &CenterSectorMobilityPolicy) -> bool {
    left.minimum_polyakov_magnitude.to_bits() == right.minimum_polyakov_magnitude.to_bits()
        && left.minimum_classified_fraction.to_bits()
            == right.minimum_classified_fraction.to_bits()
        && left.minimum_sector_transitions_per_chain
            == right.minimum_sector_transitions_per_chain
}

fn validate_policy(
    policy: &CenterSectorMobilityPolicy,
) -> Result<(), CampaignCenterQualificationError> {
    if !policy.minimum_polyakov_magnitude.is_finite() || policy.minimum_polyakov_magnitude < 0.0 {
        return Err(CampaignCenterQualificationError::InvalidMinimumMagnitude(
            policy.minimum_polyakov_magnitude,
        ));
    }
    if !policy.minimum_classified_fraction.is_finite()
        || !(0.0..=1.0).contains(&policy.minimum_classified_fraction)
    {
        return Err(
            CampaignCenterQualificationError::InvalidMinimumClassifiedFraction(
                policy.minimum_classified_fraction,
            ),
        );
    }
    Ok(())
}

/// Freeze the center-mobility requirement against one exact campaign before
/// production authorization. This supplement may be created after the broader
/// campaign manifest was frozen, but never after that campaign was authorized.
pub fn authorize_campaign_center_qualification(
    manifest: &PureSu3CampaignManifest,
    authorization: &CampaignExecutionAuthorization,
    supplement: &CampaignCenterQualificationSupplement,
) -> Result<AuthorizedCampaignCenterQualification, CampaignCenterQualificationError> {
    if supplement.supplement_id != CAMPAIGN_CENTER_QUALIFICATION_SUPPLEMENT_ID {
        return Err(CampaignCenterQualificationError::WrongSupplementId);
    }
    if authorization.authority_scope != QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE {
        return Err(CampaignCenterQualificationError::WrongAuthorizationScope);
    }
    if !manifest.measure_polyakov_loop {
        return Err(CampaignCenterQualificationError::PolyakovMeasurementNotPlanned);
    }
    if supplement.campaign_id != manifest.campaign_id
        || authorization.campaign_id != manifest.campaign_id
    {
        return Err(CampaignCenterQualificationError::CampaignMismatch);
    }
    if supplement.campaign_manifest_artifact_digest
        != authorization.campaign_manifest_artifact_digest
    {
        return Err(CampaignCenterQualificationError::ManifestDigestMismatch);
    }
    require_sha256(
        &supplement.campaign_manifest_artifact_digest,
        "campaign_manifest_artifact_digest",
    )?;
    require_nonempty(&supplement.freeze_evidence_id, "freeze_evidence_id")?;
    require_sha256(
        &supplement.frozen_policy_artifact_digest,
        "frozen_policy_artifact_digest",
    )?;
    validate_policy(&supplement.policy)?;
    if supplement.frozen_at_unix_ns == 0 {
        return Err(CampaignCenterQualificationError::InvalidFrozenTimestamp);
    }
    if supplement.frozen_at_unix_ns >= authorization.authorized_at_unix_ns {
        return Err(CampaignCenterQualificationError::SupplementNotFrozenBeforeAuthorization);
    }

    Ok(AuthorizedCampaignCenterQualification {
        supplement_id: CAMPAIGN_CENTER_QUALIFICATION_SUPPLEMENT_ID,
        campaign_id: manifest.campaign_id.clone(),
        campaign_manifest_artifact_digest: supplement.campaign_manifest_artifact_digest.clone(),
        policy: supplement.policy.clone(),
        frozen_policy_artifact_digest: supplement.frozen_policy_artifact_digest.clone(),
        frozen_at_unix_ns: supplement.frozen_at_unix_ns,
        campaign_authorized_at_unix_ns: authorization.authorized_at_unix_ns,
    })
}

/// Bind post-run center diagnostics to the exact preregistered campaign subject.
/// The public assessment struct is revalidated rather than trusted transitively.
pub fn bind_campaign_center_qualification_evidence(
    manifest: &PureSu3CampaignManifest,
    authorized: &AuthorizedCampaignCenterQualification,
    assessment: &CenterSectorMobilityAssessment,
    center_assessment_artifact_digest: impl Into<String>,
) -> Result<BoundCampaignCenterQualificationEvidence, CampaignCenterQualificationError> {
    if authorized.supplement_id != CAMPAIGN_CENTER_QUALIFICATION_SUPPLEMENT_ID {
        return Err(CampaignCenterQualificationError::WrongSupplementId);
    }
    if authorized.campaign_id != manifest.campaign_id {
        return Err(CampaignCenterQualificationError::CampaignMismatch);
    }
    if assessment.diagnostic_id != Z3_CENTER_DIAGNOSTIC_ID {
        return Err(CampaignCenterQualificationError::WrongDiagnosticId);
    }

    let assessment_policy = CenterSectorMobilityPolicy {
        minimum_polyakov_magnitude: assessment.minimum_polyakov_magnitude,
        minimum_classified_fraction: assessment.minimum_classified_fraction,
        minimum_sector_transitions_per_chain: assessment.minimum_sector_transitions_per_chain,
    };
    if !policy_matches(&authorized.policy, &assessment_policy) {
        return Err(CampaignCenterQualificationError::PolicyMismatch);
    }

    let planned_ids = manifest
        .chains
        .iter()
        .map(|chain| chain.chain_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut observed_ids = BTreeSet::new();
    let mut every_classified = true;
    let mut every_transitions = true;

    for chain in &assessment.chains {
        if !observed_ids.insert(chain.chain_id.as_str()) {
            return Err(CampaignCenterQualificationError::DuplicateAssessmentChain(
                chain.chain_id.clone(),
            ));
        }
        if !planned_ids.contains(chain.chain_id.as_str()) {
            return Err(CampaignCenterQualificationError::UnexpectedAssessmentChain(
                chain.chain_id.clone(),
            ));
        }
        if chain.sample_count != manifest.planned_measurements_per_chain {
            return Err(CampaignCenterQualificationError::SampleCountMismatch {
                chain_id: chain.chain_id.clone(),
                expected: manifest.planned_measurements_per_chain,
                actual: chain.sample_count,
            });
        }
        let sector_sum = chain.sector_counts.iter().sum::<usize>();
        let structurally_valid = chain.diagnostic_id == Z3_CENTER_DIAGNOSTIC_ID
            && chain.classified_count + chain.ambiguous_count == chain.sample_count
            && sector_sum == chain.classified_count
            && chain.sector_transition_count <= chain.classified_count.saturating_sub(1)
            && chain.maximum_classified_sector_dwell <= chain.classified_count
            && chain.maximum_ambiguous_run <= chain.ambiguous_count
            && chain.mean_magnitude.is_finite()
            && chain.mean_magnitude >= 0.0
            && chain.mean_center_aligned_real.is_none_or(f64::is_finite);
        if !structurally_valid {
            return Err(CampaignCenterQualificationError::InvalidChainDiagnostics {
                chain_id: chain.chain_id.clone(),
            });
        }

        let classified_fraction = chain.classified_count as f64 / chain.sample_count as f64;
        every_classified &= classified_fraction >= authorized.policy.minimum_classified_fraction;
        every_transitions &= chain.sector_transition_count
            >= authorized.policy.minimum_sector_transitions_per_chain;
    }

    for planned in planned_ids {
        if !observed_ids.contains(planned) {
            return Err(CampaignCenterQualificationError::MissingAssessmentChain(
                planned.to_string(),
            ));
        }
    }

    let expected_overall = every_classified && every_transitions;
    if assessment.every_chain_meets_classified_fraction != every_classified
        || assessment.every_chain_meets_transition_count != every_transitions
        || assessment.meets_declared_policy != expected_overall
    {
        return Err(CampaignCenterQualificationError::AssessmentBooleanMismatch);
    }

    let center_assessment_artifact_digest = center_assessment_artifact_digest.into();
    require_sha256(
        &center_assessment_artifact_digest,
        "center_assessment_artifact_digest",
    )?;

    Ok(BoundCampaignCenterQualificationEvidence {
        evidence_id: CAMPAIGN_CENTER_QUALIFICATION_EVIDENCE_ID,
        campaign_id: manifest.campaign_id.clone(),
        campaign_manifest_artifact_digest: authorized
            .campaign_manifest_artifact_digest
            .clone(),
        diagnostic_id: Z3_CENTER_DIAGNOSTIC_ID,
        planned_chain_count: manifest.chains.len(),
        planned_measurements_per_chain: manifest.planned_measurements_per_chain,
        policy: authorized.policy.clone(),
        center_assessment_artifact_digest,
        every_chain_meets_classified_fraction: every_classified,
        every_chain_meets_transition_count: every_transitions,
        meets_declared_policy: expected_overall,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_analysis_authority::{ExactHeadCiConclusion, ExactHeadCiEvidence};
    use crate::lattice_campaign_authority::authorize_campaign_scientific_execution;
    use crate::lattice_campaign_manifest::{
        CampaignBenchmarkPlan, CampaignChainPlan, CampaignFlowPlan, CampaignInitialCondition,
        WilsonRectangleSpec, WILSON_PURE_GAUGE_ACTION_ID,
    };
    use crate::lattice_center_symmetry::CenterSectorChainDiagnostics;

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn ci(revision: &str, run_id: u64) -> ExactHeadCiEvidence {
        ExactHeadCiEvidence {
            subject_revision: revision.into(),
            source_tree_digest: digest('8'),
            ci_run_id: run_id,
            ci_receipt_digest: digest('7'),
            toolchain_digest: digest('6'),
            build_profile: "rust-1.96-release".into(),
            conclusion: ExactHeadCiConclusion::Passed,
            qualification_timestamp_unix_ns: 2_500,
        }
    }

    fn manifest() -> PureSu3CampaignManifest {
        PureSu3CampaignManifest {
            campaign_id: "campaign-center".into(),
            dims: [4, 4, 4, 8],
            beta: 6.0,
            action_id: WILSON_PURE_GAUGE_ACTION_ID.into(),
            periodic_boundaries: true,
            sampler_id: "hb-or-v1".into(),
            sampler_config_digest: digest('a'),
            sampler_implementation_revision: "sampler-head".into(),
            sampler_transition_evidence_id: "sampler-theorem".into(),
            sampler_action_parity_evidence_id: "action-parity".into(),
            sampler_exact_head_ci_evidence_id: "sampler-ci".into(),
            rng_algorithm_id: "chacha8-v1".into(),
            rng_implementation_revision: "rng-head".into(),
            rng_qualification_evidence_id: "rng-evidence".into(),
            chains: vec![
                CampaignChainPlan {
                    chain_id: "cold".into(),
                    initial_condition: CampaignInitialCondition::ColdIdentity,
                    initialization_evidence_id: None,
                    initialization_stream_id: "init-0".into(),
                    production_stream_id: "prod-0".into(),
                    tuning_stream_id: None,
                    seed_commitment: digest('b'),
                },
                CampaignChainPlan {
                    chain_id: "hot".into(),
                    initial_condition: CampaignInitialCondition::QualifiedDisordered,
                    initialization_evidence_id: Some("init-evidence".into()),
                    initialization_stream_id: "init-1".into(),
                    production_stream_id: "prod-1".into(),
                    tuning_stream_id: None,
                    seed_commitment: digest('c'),
                },
            ],
            thermalization_cycles: 100,
            measurement_stride_cycles: 10,
            planned_measurements_per_chain: 4,
            measure_plaquette: true,
            measure_polyakov_loop: true,
            wilson_rectangles: vec![WilsonRectangleSpec {
                spatial_direction: 0,
                temporal_direction: 3,
                spatial_extent: 1,
                temporal_extent: 1,
            }],
            flow: CampaignFlowPlan {
                flow_implementation_id: "rk3-v1".into(),
                flow_implementation_revision: "flow-head".into(),
                flow_step_size: 0.001,
                measurement_flow_times: vec![0.01, 0.02],
                energy_operator_id: "energy-v1".into(),
                topology_operator_id: "topology-v1".into(),
                flow_oracle_evidence_id: "flow-oracle".into(),
                flow_exact_head_ci_evidence_id: "flow-ci".into(),
            },
            qualification_policy_artifact_digest: digest('d'),
            qualification_policy_freeze_evidence_id: "policy-freeze".into(),
            qualification_policy_frozen_at_unix_ns: 1_000,
            benchmark: CampaignBenchmarkPlan {
                benchmark_id: "benchmark".into(),
                benchmark_source_digest: digest('e'),
                benchmark_contract_digest: digest('f'),
                required_observable_ids: vec!["plaquette".into(), "polyakov-loop".into()],
            },
            campaign_code_revision: "campaign-head".into(),
            campaign_configuration_digest: digest('1'),
            campaign_freeze_evidence_id: "campaign-freeze".into(),
            campaign_frozen_at_unix_ns: 2_000,
        }
    }

    fn authorization(manifest: &PureSu3CampaignManifest) -> CampaignExecutionAuthorization {
        authorize_campaign_scientific_execution(
            manifest,
            digest('9'),
            &ci("campaign-head", 11),
            &ci("sampler-head", 12),
            &ci("flow-head", 13),
            4_000,
        )
        .unwrap()
    }

    fn supplement() -> CampaignCenterQualificationSupplement {
        CampaignCenterQualificationSupplement {
            supplement_id: CAMPAIGN_CENTER_QUALIFICATION_SUPPLEMENT_ID,
            campaign_id: "campaign-center".into(),
            campaign_manifest_artifact_digest: digest('9'),
            policy: CenterSectorMobilityPolicy {
                minimum_polyakov_magnitude: 0.05,
                minimum_classified_fraction: 0.9,
                minimum_sector_transitions_per_chain: 1,
            },
            freeze_evidence_id: "center-policy-freeze".into(),
            frozen_policy_artifact_digest: digest('4'),
            frozen_at_unix_ns: 3_000,
        }
    }

    fn assessment() -> CenterSectorMobilityAssessment {
        let chain = |id: &str| CenterSectorChainDiagnostics {
            diagnostic_id: Z3_CENTER_DIAGNOSTIC_ID,
            chain_id: id.into(),
            sample_count: 4,
            classified_count: 4,
            ambiguous_count: 0,
            sector_counts: [2, 1, 1],
            sector_transition_count: 2,
            maximum_classified_sector_dwell: 2,
            maximum_ambiguous_run: 0,
            mean_magnitude: 0.3,
            mean_center_aligned_real: Some(0.28),
        };
        CenterSectorMobilityAssessment {
            diagnostic_id: Z3_CENTER_DIAGNOSTIC_ID,
            minimum_polyakov_magnitude: 0.05,
            minimum_classified_fraction: 0.9,
            minimum_sector_transitions_per_chain: 1,
            chains: vec![chain("cold"), chain("hot")],
            every_chain_meets_classified_fraction: true,
            every_chain_meets_transition_count: true,
            meets_declared_policy: true,
        }
    }

    #[test]
    fn supplement_must_precede_campaign_authorization() {
        let manifest = manifest();
        let authorization = authorization(&manifest);
        let mut late = supplement();
        late.frozen_at_unix_ns = authorization.authorized_at_unix_ns;
        assert!(matches!(
            authorize_campaign_center_qualification(&manifest, &authorization, &late),
            Err(CampaignCenterQualificationError::SupplementNotFrozenBeforeAuthorization)
        ));
    }

    #[test]
    fn binds_complete_exact_chain_population() {
        let manifest = manifest();
        let authorization = authorization(&manifest);
        let authorized = authorize_campaign_center_qualification(
            &manifest,
            &authorization,
            &supplement(),
        )
        .unwrap();
        let evidence = bind_campaign_center_qualification_evidence(
            &manifest,
            &authorized,
            &assessment(),
            digest('5'),
        )
        .unwrap();
        assert_eq!(evidence.planned_chain_count, 2);
        assert_eq!(evidence.planned_measurements_per_chain, 4);
        assert!(evidence.meets_declared_policy);
    }

    #[test]
    fn post_hoc_policy_change_fails_closed() {
        let manifest = manifest();
        let authorization = authorization(&manifest);
        let authorized = authorize_campaign_center_qualification(
            &manifest,
            &authorization,
            &supplement(),
        )
        .unwrap();
        let mut changed = assessment();
        changed.minimum_sector_transitions_per_chain = 2;
        assert!(matches!(
            bind_campaign_center_qualification_evidence(
                &manifest,
                &authorized,
                &changed,
                digest('5'),
            ),
            Err(CampaignCenterQualificationError::PolicyMismatch)
        ));
    }

    #[test]
    fn missing_chain_fails_closed() {
        let manifest = manifest();
        let authorization = authorization(&manifest);
        let authorized = authorize_campaign_center_qualification(
            &manifest,
            &authorization,
            &supplement(),
        )
        .unwrap();
        let mut incomplete = assessment();
        incomplete.chains.pop();
        assert!(matches!(
            bind_campaign_center_qualification_evidence(
                &manifest,
                &authorized,
                &incomplete,
                digest('5'),
            ),
            Err(CampaignCenterQualificationError::MissingAssessmentChain(_))
        ));
    }

    #[test]
    fn forged_boolean_summary_fails_closed() {
        let manifest = manifest();
        let authorization = authorization(&manifest);
        let authorized = authorize_campaign_center_qualification(
            &manifest,
            &authorization,
            &supplement(),
        )
        .unwrap();
        let mut forged = assessment();
        forged.every_chain_meets_transition_count = false;
        assert!(matches!(
            bind_campaign_center_qualification_evidence(
                &manifest,
                &authorized,
                &forged,
                digest('5'),
            ),
            Err(CampaignCenterQualificationError::AssessmentBooleanMismatch)
        ));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-run authority and post-run lineage checks for a frozen SU(3) campaign.
//!
//! The campaign manifest is frozen before execution. Exact-head CI can then
//! qualify that already-frozen code subject without mutating the manifest. Only
//! after authorization may production chains start.
//!
//! Authorization != successful execution != statistically promoted ensemble.

use std::collections::BTreeSet;

use crate::lattice_campaign_manifest::{
    CampaignManifestError, PureSu3CampaignManifest, PureSu3CampaignRunRecord,
};

pub const QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE: &str =
    "qualified_campaign_execution_authorization_v1";
pub const QUALIFIED_CAMPAIGN_EXECUTION_RECEIPT_SCOPE: &str =
    "qualified_campaign_execution_complete_v1";

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignExecutionAuthorization {
    pub authority_scope: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub campaign_code_revision: String,
    pub campaign_exact_head_ci_evidence_id: String,
    pub sampler_exact_head_ci_evidence_id: String,
    pub flow_exact_head_ci_evidence_id: String,
    pub qualification_policy_artifact_digest: String,
    pub measured_observable_ids: Vec<String>,
    pub benchmark_required_observable_ids: Vec<String>,
    pub authorized_at_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedCampaignExecutionReceipt {
    pub authority_scope: &'static str,
    pub campaign_id: String,
    pub campaign_manifest_artifact_digest: String,
    pub authorization_timestamp_unix_ns: u128,
    pub started_at_unix_ns: u128,
    pub completed_at_unix_ns: u128,
    pub chain_count: usize,
    pub combined_measurement_artifact_digest: String,
    pub run_receipt_artifact_digest: String,
    pub seed_commitments_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CampaignAuthorityError {
    Manifest(CampaignManifestError),
    EmptyField(&'static str),
    InvalidSha256Digest { field: &'static str, value: String },
    AuthorizationNotAfterCampaignFreeze,
    MissingBenchmarkObservable { observable_id: String },
    WrongAuthorizationScope,
    AuthorizationCampaignMismatch,
    AuthorizationManifestDigestMismatch,
    RunStartedBeforeAuthorization,
    SeedCommitmentMismatch { chain_id: String },
}

impl From<CampaignManifestError> for CampaignAuthorityError {
    fn from(value: CampaignManifestError) -> Self {
        Self::Manifest(value)
    }
}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), CampaignAuthorityError> {
    if value.trim().is_empty() {
        Err(CampaignAuthorityError::EmptyField(field))
    } else {
        Ok(())
    }
}

fn require_sha256(value: &str, field: &'static str) -> Result<(), CampaignAuthorityError> {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return Err(CampaignAuthorityError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    };
    if hex.len() != 64 || !hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(CampaignAuthorityError::InvalidSha256Digest {
            field,
            value: value.to_string(),
        });
    }
    Ok(())
}

fn measured_observable_ids(manifest: &PureSu3CampaignManifest) -> Vec<String> {
    let mut ids = BTreeSet::new();
    if manifest.measure_plaquette {
        ids.insert("plaquette".to_string());
    }
    if manifest.measure_polyakov_loop {
        ids.insert("polyakov-loop".to_string());
    }
    for rectangle in &manifest.wilson_rectangles {
        ids.insert(format!(
            "wilson-loop-r{}-t{}-mu{}-nu{}",
            rectangle.spatial_extent,
            rectangle.temporal_extent,
            rectangle.spatial_direction,
            rectangle.temporal_direction
        ));
        // Family alias is useful for benchmark contracts that compare an
        // orientation-averaged or otherwise aggregated R×T observable.
        ids.insert(format!(
            "wilson-loop-{}x{}",
            rectangle.spatial_extent, rectangle.temporal_extent
        ));
    }
    if !manifest.flow.measurement_flow_times.is_empty() {
        ids.insert("flow-energy".to_string());
        ids.insert("topological-q".to_string());
    }
    ids.into_iter().collect()
}

/// Authorize production execution of an already-frozen campaign.
///
/// The campaign manifest itself remains immutable. This function supplies the
/// orchestration exact-head CI evidence produced after freezing and requires the
/// authorization to precede every production chain.
pub fn authorize_campaign_scientific_execution(
    manifest: &PureSu3CampaignManifest,
    campaign_manifest_artifact_digest: impl Into<String>,
    campaign_exact_head_ci_evidence_id: impl Into<String>,
    authorized_at_unix_ns: u128,
) -> Result<CampaignExecutionAuthorization, CampaignAuthorityError> {
    manifest.validate_for_scientific_execution()?;
    let campaign_manifest_artifact_digest = campaign_manifest_artifact_digest.into();
    let campaign_exact_head_ci_evidence_id = campaign_exact_head_ci_evidence_id.into();
    require_sha256(
        &campaign_manifest_artifact_digest,
        "campaign_manifest_artifact_digest",
    )?;
    require_nonempty(
        &campaign_exact_head_ci_evidence_id,
        "campaign_exact_head_ci_evidence_id",
    )?;
    if authorized_at_unix_ns <= manifest.campaign_frozen_at_unix_ns {
        return Err(CampaignAuthorityError::AuthorizationNotAfterCampaignFreeze);
    }

    let measured = measured_observable_ids(manifest);
    let measured_set = measured.iter().map(String::as_str).collect::<BTreeSet<_>>();
    for observable_id in &manifest.benchmark.required_observable_ids {
        if !measured_set.contains(observable_id.as_str()) {
            return Err(CampaignAuthorityError::MissingBenchmarkObservable {
                observable_id: observable_id.clone(),
            });
        }
    }

    let mut benchmark_required_observable_ids = manifest.benchmark.required_observable_ids.clone();
    benchmark_required_observable_ids.sort();

    Ok(CampaignExecutionAuthorization {
        authority_scope: QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE,
        campaign_id: manifest.campaign_id.clone(),
        campaign_manifest_artifact_digest,
        campaign_code_revision: manifest.campaign_code_revision.clone(),
        campaign_exact_head_ci_evidence_id,
        sampler_exact_head_ci_evidence_id: manifest.sampler_exact_head_ci_evidence_id.clone(),
        flow_exact_head_ci_evidence_id: manifest.flow.flow_exact_head_ci_evidence_id.clone(),
        qualification_policy_artifact_digest: manifest.qualification_policy_artifact_digest.clone(),
        measured_observable_ids: measured,
        benchmark_required_observable_ids,
        authorized_at_unix_ns,
    })
}

/// Validate a completed campaign against both the frozen manifest and the pre-run
/// execution authorization. Seed reveal digests must exactly match their pre-run
/// commitments; this module treats both as SHA-256 digests of the same disclosed
/// seed material.
pub fn bind_qualified_campaign_execution(
    authorization: &CampaignExecutionAuthorization,
    manifest: &PureSu3CampaignManifest,
    run: &PureSu3CampaignRunRecord,
) -> Result<QualifiedCampaignExecutionReceipt, CampaignAuthorityError> {
    if authorization.authority_scope != QUALIFIED_CAMPAIGN_EXECUTION_AUTHORIZATION_SCOPE {
        return Err(CampaignAuthorityError::WrongAuthorizationScope);
    }
    if authorization.campaign_id != manifest.campaign_id {
        return Err(CampaignAuthorityError::AuthorizationCampaignMismatch);
    }
    if authorization.campaign_manifest_artifact_digest != run.campaign_manifest_artifact_digest {
        return Err(CampaignAuthorityError::AuthorizationManifestDigestMismatch);
    }
    run.validate_against(manifest)?;
    if run.started_at_unix_ns <= authorization.authorized_at_unix_ns {
        return Err(CampaignAuthorityError::RunStartedBeforeAuthorization);
    }

    for planned in &manifest.chains {
        let actual = run
            .chains
            .iter()
            .find(|record| record.chain_id == planned.chain_id)
            .expect("run.validate_against guarantees every planned chain exists");
        if actual.seed_reveal_digest != planned.seed_commitment {
            return Err(CampaignAuthorityError::SeedCommitmentMismatch {
                chain_id: planned.chain_id.clone(),
            });
        }
    }

    Ok(QualifiedCampaignExecutionReceipt {
        authority_scope: QUALIFIED_CAMPAIGN_EXECUTION_RECEIPT_SCOPE,
        campaign_id: manifest.campaign_id.clone(),
        campaign_manifest_artifact_digest: authorization
            .campaign_manifest_artifact_digest
            .clone(),
        authorization_timestamp_unix_ns: authorization.authorized_at_unix_ns,
        started_at_unix_ns: run.started_at_unix_ns,
        completed_at_unix_ns: run.completed_at_unix_ns,
        chain_count: run.chains.len(),
        combined_measurement_artifact_digest: run.combined_measurement_artifact_digest.clone(),
        run_receipt_artifact_digest: run.run_receipt_artifact_digest.clone(),
        seed_commitments_verified: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice_campaign_manifest::{
        CampaignBenchmarkPlan, CampaignChainPlan, CampaignChainRunRecord, CampaignFlowPlan,
        CampaignInitialCondition, PURE_SU3_CAMPAIGN_RUN_ID, WILSON_PURE_GAUGE_ACTION_ID,
        WilsonRectangleSpec,
    };

    fn digest(ch: char) -> String {
        format!("sha256:{}", ch.to_string().repeat(64))
    }

    fn manifest() -> PureSu3CampaignManifest {
        PureSu3CampaignManifest {
            campaign_id: "campaign-001".into(),
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
                required_observable_ids: vec!["plaquette".into(), "wilson-loop-1x1".into()],
            },
            campaign_code_revision: "campaign-head".into(),
            campaign_configuration_digest: digest('1'),
            campaign_freeze_evidence_id: "campaign-freeze".into(),
            campaign_frozen_at_unix_ns: 2_000,
        }
    }

    fn run(manifest: &PureSu3CampaignManifest, manifest_digest: &str) -> PureSu3CampaignRunRecord {
        let schedule = manifest.retained_measurement_cycles().unwrap();
        PureSu3CampaignRunRecord {
            run_schema_id: PURE_SU3_CAMPAIGN_RUN_ID,
            campaign_manifest_artifact_digest: manifest_digest.into(),
            campaign_id: manifest.campaign_id.clone(),
            campaign_code_revision: manifest.campaign_code_revision.clone(),
            started_at_unix_ns: 4_000,
            completed_at_unix_ns: 5_000,
            chains: manifest
                .chains
                .iter()
                .map(|chain| CampaignChainRunRecord {
                    chain_id: chain.chain_id.clone(),
                    completed_cycles: manifest.required_completed_cycles().unwrap(),
                    retained_measurement_cycles: schedule.clone(),
                    raw_trajectory_artifact_digest: digest('2'),
                    sampler_trace_artifact_digest: digest('3'),
                    seed_reveal_digest: chain.seed_commitment.clone(),
                })
                .collect(),
            combined_measurement_artifact_digest: digest('4'),
            run_receipt_artifact_digest: digest('5'),
        }
    }

    #[test]
    fn authorization_proves_benchmark_observables_are_measured() {
        let manifest = manifest();
        let authorization = authorize_campaign_scientific_execution(
            &manifest,
            digest('9'),
            "campaign-ci",
            3_000,
        )
        .unwrap();
        assert!(authorization.measured_observable_ids.contains(&"plaquette".into()));
        assert!(authorization.measured_observable_ids.contains(&"wilson-loop-1x1".into()));
    }

    #[test]
    fn missing_benchmark_measurement_fails_before_execution() {
        let mut manifest = manifest();
        manifest.wilson_rectangles.clear();
        assert!(matches!(
            authorize_campaign_scientific_execution(&manifest, digest('9'), "campaign-ci", 3_000),
            Err(CampaignAuthorityError::MissingBenchmarkObservable { observable_id })
                if observable_id == "wilson-loop-1x1"
        ));
    }

    #[test]
    fn run_must_begin_after_authorization() {
        let manifest = manifest();
        let authorization = authorize_campaign_scientific_execution(
            &manifest,
            digest('9'),
            "campaign-ci",
            4_000,
        )
        .unwrap();
        let record = run(&manifest, &digest('9'));
        assert!(matches!(
            bind_qualified_campaign_execution(&authorization, &manifest, &record),
            Err(CampaignAuthorityError::RunStartedBeforeAuthorization)
        ));
    }

    #[test]
    fn seed_reveal_must_match_pre_run_commitment() {
        let manifest = manifest();
        let authorization = authorize_campaign_scientific_execution(
            &manifest,
            digest('9'),
            "campaign-ci",
            3_000,
        )
        .unwrap();
        let mut record = run(&manifest, &digest('9'));
        record.chains[0].seed_reveal_digest = digest('8');
        assert!(matches!(
            bind_qualified_campaign_execution(&authorization, &manifest, &record),
            Err(CampaignAuthorityError::SeedCommitmentMismatch { .. })
        ));
    }

    #[test]
    fn qualified_execution_receipt_stays_below_statistical_promotion() {
        let manifest = manifest();
        let manifest_digest = digest('9');
        let authorization = authorize_campaign_scientific_execution(
            &manifest,
            manifest_digest.clone(),
            "campaign-ci",
            3_000,
        )
        .unwrap();
        let record = run(&manifest, &manifest_digest);
        let receipt = bind_qualified_campaign_execution(&authorization, &manifest, &record).unwrap();
        assert_eq!(
            receipt.authority_scope,
            QUALIFIED_CAMPAIGN_EXECUTION_RECEIPT_SCOPE
        );
        assert!(receipt.seed_commitments_verified);
    }
}

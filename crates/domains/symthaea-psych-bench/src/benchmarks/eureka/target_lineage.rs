// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scoped candidate-subject lineage for EUREKA production targets.
//!
//! V1 candidate identity binds source/config/environment. This V2 composition
//! additionally binds the exact learned target snapshot, adapter contract and
//! typed architectural scope without mutating the V1 constitution in place.

use super::campaign_manifest::{CandidateBindingError, CandidateSubjectBindingV1};
use super::hidden_world::FixtureFamily;
use super::target_contract::{
    EurekaTargetScope, FEP_TARGET_ADAPTER_REVISION, FepTargetContract,
};

pub(super) const TARGET_SCOPED_CANDIDATE_REVISION: &str =
    "EUREKA.002G.TARGET_SCOPED_CANDIDATE.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TargetScopedCandidateError {
    InvalidLegacyCandidate(CandidateBindingError),
    AdapterRevisionMismatch,
    InvalidComponentScope,
}

/// Outcome-independent identity of the exact candidate actually eligible for a
/// target-scoped EUREKA campaign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TargetScopedCandidateBindingV2 {
    legacy_candidate_digest: u64,
    scope: EurekaTargetScope,
    family: FixtureFamily,
    target_contract_digest: u64,
    snapshot_replay_digest: u64,
    adapter_revision: &'static str,
    replay_digest: u64,
}

impl TargetScopedCandidateBindingV2 {
    /// Canonical constructor for the production-FEP component target.
    ///
    /// Scope is derived from the already-validated target contract. There is no
    /// caller parameter through which component evidence can be relabeled as a
    /// full cognitive-loop target.
    pub(super) fn from_production_fep(
        legacy: &CandidateSubjectBindingV1,
        contract: &FepTargetContract,
    ) -> Result<Self, TargetScopedCandidateError> {
        let legacy_candidate_digest = legacy
            .replay_digest()
            .map_err(TargetScopedCandidateError::InvalidLegacyCandidate)?;
        if legacy.target_adapter_revision != FEP_TARGET_ADAPTER_REVISION {
            return Err(TargetScopedCandidateError::AdapterRevisionMismatch);
        }
        if contract.scope() != EurekaTargetScope::ProductionFepComponentSnapshot {
            return Err(TargetScopedCandidateError::InvalidComponentScope);
        }

        let mut binding = Self {
            legacy_candidate_digest,
            scope: contract.scope(),
            family: contract.family(),
            target_contract_digest: contract.replay_digest(),
            snapshot_replay_digest: contract.snapshot_replay_digest(),
            adapter_revision: FEP_TARGET_ADAPTER_REVISION,
            replay_digest: 0,
        };
        binding.replay_digest = scoped_candidate_digest(&binding);
        Ok(binding)
    }

    pub(super) fn legacy_candidate_digest(&self) -> u64 {
        self.legacy_candidate_digest
    }

    pub(super) fn scope(&self) -> EurekaTargetScope {
        self.scope
    }

    pub(super) fn family(&self) -> FixtureFamily {
        self.family
    }

    pub(super) fn target_contract_digest(&self) -> u64 {
        self.target_contract_digest
    }

    pub(super) fn snapshot_replay_digest(&self) -> u64 {
        self.snapshot_replay_digest
    }

    pub(super) fn adapter_revision(&self) -> &'static str {
        self.adapter_revision
    }

    pub(super) fn replay_digest(&self) -> u64 {
        self.replay_digest
    }
}

fn scoped_candidate_digest(binding: &TargetScopedCandidateBindingV2) -> u64 {
    let mut bytes = Vec::new();
    encode_str(&mut bytes, TARGET_SCOPED_CANDIDATE_REVISION);
    bytes.extend_from_slice(&binding.legacy_candidate_digest.to_le_bytes());
    bytes.push(match binding.scope {
        EurekaTargetScope::ProductionFepComponentSnapshot => 1,
        EurekaTargetScope::FullCognitiveLoop => 2,
    });
    bytes.push(match binding.family {
        FixtureFamily::CausalBits => 1,
        FixtureFamily::ResourceFlow => 2,
    });
    bytes.extend_from_slice(&binding.target_contract_digest.to_le_bytes());
    bytes.extend_from_slice(&binding.snapshot_replay_digest.to_le_bytes());
    encode_str(&mut bytes, binding.adapter_revision);
    fnv1a64(&bytes)
}

fn encode_str(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::eureka::analysis_plan::EUREKA_002_ANALYSIS_PLAN_V1;
    use crate::benchmarks::eureka::target_contract::FepTargetContract;
    use symthaea_fep::{
        ActiveInferenceAgent, ActiveInferenceAgentConfig, FepEvaluationSnapshot,
        FepPredictionSession,
    };

    fn sha(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn legacy_candidate() -> CandidateSubjectBindingV1 {
        CandidateSubjectBindingV1 {
            source_commit_hex: std::iter::repeat_n('a', 40).collect(),
            source_tree_sha256: sha('b'),
            candidate_config_sha256: sha('c'),
            target_adapter_revision: FEP_TARGET_ADAPTER_REVISION.to_string(),
            flake_lock_sha256: sha('d'),
            toolchain_evidence_sha256: sha('e'),
            execution_environment_sha256: sha('f'),
            command_runner_sha256: sha('1'),
            analysis_plan_digest: EUREKA_002_ANALYSIS_PLAN_V1.replay_digest(),
        }
    }

    fn snapshot(obs_dim: usize, actions: usize, model_delta: f64) -> FepEvaluationSnapshot {
        let mut agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim,
            num_actions: actions,
            enable_td_learning: true,
            ..Default::default()
        });
        agent.model.transition_matrices[0][0][0] += model_delta;
        FepPredictionSession::from_agent(&agent)
            .freeze_for_evaluation()
            .unwrap()
    }

    #[test]
    fn scope_is_derived_from_component_contract_not_caller_input() {
        let frozen = snapshot(4, 4, 0.0);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let binding = TargetScopedCandidateBindingV2::from_production_fep(
            &legacy_candidate(),
            &contract,
        )
        .unwrap();
        assert_eq!(binding.scope(), EurekaTargetScope::ProductionFepComponentSnapshot);
        assert_eq!(binding.family(), FixtureFamily::ResourceFlow);
        assert_eq!(binding.snapshot_replay_digest(), frozen.replay_digest());
        assert_eq!(binding.target_contract_digest(), contract.replay_digest());
        assert_eq!(binding.adapter_revision(), FEP_TARGET_ADAPTER_REVISION);
    }

    #[test]
    fn adapter_revision_mismatch_fails_closed() {
        let frozen = snapshot(4, 4, 0.0);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let mut legacy = legacy_candidate();
        legacy.target_adapter_revision = "some-other-adapter".to_string();
        assert_eq!(
            TargetScopedCandidateBindingV2::from_production_fep(&legacy, &contract),
            Err(TargetScopedCandidateError::AdapterRevisionMismatch)
        );
    }

    #[test]
    fn learned_model_change_changes_scientific_subject_identity() {
        let a = snapshot(4, 4, 0.0);
        let b = snapshot(4, 4, 0.01);
        let contract_a = FepTargetContract::new(&a, FixtureFamily::ResourceFlow).unwrap();
        let contract_b = FepTargetContract::new(&b, FixtureFamily::ResourceFlow).unwrap();
        let legacy = legacy_candidate();
        let binding_a =
            TargetScopedCandidateBindingV2::from_production_fep(&legacy, &contract_a).unwrap();
        let binding_b =
            TargetScopedCandidateBindingV2::from_production_fep(&legacy, &contract_b).unwrap();
        assert_ne!(binding_a.snapshot_replay_digest(), binding_b.snapshot_replay_digest());
        assert_ne!(binding_a.replay_digest(), binding_b.replay_digest());
    }

    #[test]
    fn legacy_source_or_config_change_changes_scoped_identity() {
        let frozen = snapshot(4, 4, 0.0);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let a = legacy_candidate();
        let mut b = legacy_candidate();
        b.candidate_config_sha256 = sha('9');
        let scoped_a = TargetScopedCandidateBindingV2::from_production_fep(&a, &contract).unwrap();
        let scoped_b = TargetScopedCandidateBindingV2::from_production_fep(&b, &contract).unwrap();
        assert_ne!(scoped_a.legacy_candidate_digest(), scoped_b.legacy_candidate_digest());
        assert_ne!(scoped_a.replay_digest(), scoped_b.replay_digest());
    }

    #[test]
    fn fixture_family_changes_scoped_identity() {
        let frozen = snapshot(5, 5, 0.0);
        let bits = FepTargetContract::new(&frozen, FixtureFamily::CausalBits).unwrap();
        let flow = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let legacy = legacy_candidate();
        let bits_binding =
            TargetScopedCandidateBindingV2::from_production_fep(&legacy, &bits).unwrap();
        let flow_binding =
            TargetScopedCandidateBindingV2::from_production_fep(&legacy, &flow).unwrap();
        assert_ne!(bits_binding.family(), flow_binding.family());
        assert_ne!(bits_binding.replay_digest(), flow_binding.replay_digest());
    }

    #[test]
    fn scoped_identity_is_deterministic_and_outcome_independent() {
        let frozen = snapshot(4, 4, 0.0);
        let contract = FepTargetContract::new(&frozen, FixtureFamily::ResourceFlow).unwrap();
        let legacy = legacy_candidate();
        let a = TargetScopedCandidateBindingV2::from_production_fep(&legacy, &contract).unwrap();
        let b = TargetScopedCandidateBindingV2::from_production_fep(&legacy, &contract).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.replay_digest(), b.replay_digest());
    }
}
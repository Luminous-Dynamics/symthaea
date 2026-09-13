// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 paired scientific-subject contract.
//!
//! This source/test-only layer binds the already-frozen target contract and
//! comparator custody receipt into one pre-HeldOut identity. It owns no world,
//! cannot fit/train either subject, cannot execute predictions, and contains no
//! outcome fields.

use super::analysis_plan::{ANALYSIS_PLAN_REVISION, EUREKA_002_ANALYSIS_PLAN_V1};
use super::v2_comparator_custody::V2ComparatorCustodyReceipt;
use super::v2_target_contract::V2FepTargetContract;

pub(super) const V2_PAIRED_SUBJECT_REVISION: &str = "EUREKA.002.V2.PAIRED_SUBJECT.v2";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2PairedSubjectError {
    PublicSchemaMismatch,
    AnalysisPlanMismatch,
    ZeroTargetCommitment,
    ZeroComparatorCommitment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2PairedSubjectContract {
    public_schema_commitment: [u8; 32],
    target_contract_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2PairedSubjectContract {
    pub(super) fn freeze(
        target: V2FepTargetContract,
        comparator: V2ComparatorCustodyReceipt,
    ) -> Result<Self, V2PairedSubjectError> {
        let public_schema_commitment = validate_schema_pair(
            target.public_schema_commitment(),
            comparator.schema_commitment(),
        )?;
        let analysis_plan_replay_digest = EUREKA_002_ANALYSIS_PLAN_V1.replay_digest();
        let analysis_plan_commitment = EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment();
        validate_analysis_plan_pair(
            analysis_plan_commitment,
            comparator.analysis_plan_commitment(),
        )?;

        let target_contract_commitment = target.commitment();
        let comparator_subject_commitment = comparator.commitment();
        if target_contract_commitment == [0_u8; 32] {
            return Err(V2PairedSubjectError::ZeroTargetCommitment);
        }
        if comparator_subject_commitment == [0_u8; 32] {
            return Err(V2PairedSubjectError::ZeroComparatorCommitment);
        }
        let commitment = paired_subject_commitment(
            public_schema_commitment,
            target_contract_commitment,
            comparator_subject_commitment,
            analysis_plan_replay_digest,
            analysis_plan_commitment,
        );
        Ok(Self {
            public_schema_commitment,
            target_contract_commitment,
            comparator_subject_commitment,
            analysis_plan_replay_digest,
            analysis_plan_commitment,
            commitment,
        })
    }

    pub(super) const fn public_schema_commitment(self) -> [u8; 32] {
        self.public_schema_commitment
    }

    pub(super) const fn target_contract_commitment(self) -> [u8; 32] {
        self.target_contract_commitment
    }

    pub(super) const fn comparator_subject_commitment(self) -> [u8; 32] {
        self.comparator_subject_commitment
    }

    pub(super) const fn analysis_plan_replay_digest(self) -> u64 {
        self.analysis_plan_replay_digest
    }

    pub(super) const fn analysis_plan_commitment(self) -> [u8; 32] {
        self.analysis_plan_commitment
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

fn validate_schema_pair(
    target_schema: [u8; 32],
    comparator_schema: [u8; 32],
) -> Result<[u8; 32], V2PairedSubjectError> {
    if target_schema != comparator_schema {
        return Err(V2PairedSubjectError::PublicSchemaMismatch);
    }
    Ok(target_schema)
}

fn validate_analysis_plan_pair(
    expected: [u8; 32],
    comparator: [u8; 32],
) -> Result<[u8; 32], V2PairedSubjectError> {
    if expected != comparator {
        return Err(V2PairedSubjectError::AnalysisPlanMismatch);
    }
    Ok(expected)
}

fn paired_subject_commitment(
    public_schema_commitment: [u8; 32],
    target_contract_commitment: [u8; 32],
    comparator_subject_commitment: [u8; 32],
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_PAIRED_SUBJECT_REVISION.as_bytes());
    encode_bytes(&mut bytes, ANALYSIS_PLAN_REVISION.as_bytes());
    bytes.extend_from_slice(&analysis_plan_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&analysis_plan_commitment);
    bytes.extend_from_slice(&public_schema_commitment);
    bytes.extend_from_slice(&target_contract_commitment);
    bytes.extend_from_slice(&comparator_subject_commitment);
    *blake3::hash(&bytes).as_bytes()
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_fep::{
        ActiveInferenceAgent, ActiveInferenceAgentConfig, FepHeldOutSubject, FepPredictionSession,
    };

    use super::super::baselines::ShortcutBaselineKind;
    use super::super::hidden_world::PublicAction;
    use super::super::v2_comparator_custody::{
        V2CorpusPartition, V2DevelopmentFitCorpus, V2PublicTransitionEvidence,
    };
    use super::super::v2_public_schema::{
        V2PublicFamily, V2PublicState, public_schema_commitment,
    };

    fn target(model_delta: f64) -> V2FepTargetContract {
        let mut agent = ActiveInferenceAgent::new(ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 4,
            ..Default::default()
        });
        agent.model.transition_matrices[0][0][0] += model_delta;
        let subject = FepHeldOutSubject::seal(
            FepPredictionSession::from_agent(&agent)
                .freeze_for_evaluation()
                .unwrap(),
        );
        V2FepTargetContract::from_heldout_subject(&subject).unwrap()
    }

    fn comparator(
        selected: ShortcutBaselineKind,
        post_delta: i32,
    ) -> V2ComparatorCustodyReceipt {
        let record = V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            [1_u8; 32],
            V2PublicState::new([3, 4, 5, 0]).unwrap(),
            PublicAction::Pulse { slot: 0 },
            V2PublicState::new([2, 5 + post_delta, 5, 0]).unwrap(),
        )
        .unwrap();
        let corpus = V2DevelopmentFitCorpus::freeze(vec![record]).unwrap();
        V2ComparatorCustodyReceipt::freeze(&corpus, selected, [0xA5_u8; 32]).unwrap()
    }

    #[test]
    fn same_schema_target_and_comparator_form_one_nonzero_subject() {
        let target = target(0.0);
        let comparator = comparator(ShortcutBaselineKind::NearestTransition, 0);
        let pair = V2PairedSubjectContract::freeze(target, comparator).unwrap();
        assert_eq!(pair.public_schema_commitment(), public_schema_commitment());
        assert_eq!(
            pair.analysis_plan_replay_digest(),
            EUREKA_002_ANALYSIS_PLAN_V1.replay_digest()
        );
        assert_eq!(
            pair.analysis_plan_commitment(),
            EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment()
        );
        assert_ne!(pair.target_contract_commitment(), [0_u8; 32]);
        assert_ne!(pair.comparator_subject_commitment(), [0_u8; 32]);
        assert_ne!(pair.commitment(), [0_u8; 32]);
    }

    #[test]
    fn schema_mismatch_fails_closed_before_execution() {
        let target_schema = public_schema_commitment();
        let mut comparator_schema = target_schema;
        comparator_schema[0] ^= 0x01;
        assert_eq!(
            validate_schema_pair(target_schema, comparator_schema),
            Err(V2PairedSubjectError::PublicSchemaMismatch)
        );
    }

    #[test]
    fn stale_analysis_plan_fails_closed_before_execution() {
        let expected = EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment();
        let mut stale = expected;
        stale[0] ^= 0x01;
        assert_eq!(
            validate_analysis_plan_pair(expected, stale),
            Err(V2PairedSubjectError::AnalysisPlanMismatch)
        );
    }

    #[test]
    fn changing_learned_target_changes_paired_subject() {
        let comparator = comparator(ShortcutBaselineKind::NearestTransition, 0);
        let a = V2PairedSubjectContract::freeze(target(0.0), comparator).unwrap();
        let comparator = comparator(ShortcutBaselineKind::NearestTransition, 0);
        let b = V2PairedSubjectContract::freeze(target(0.01), comparator).unwrap();
        assert_ne!(a.target_contract_commitment(), b.target_contract_commitment());
        assert_ne!(a.commitment(), b.commitment());
    }

    #[test]
    fn changing_comparator_algorithm_changes_paired_subject() {
        let a = V2PairedSubjectContract::freeze(
            target(0.0),
            comparator(ShortcutBaselineKind::NearestTransition, 0),
        )
        .unwrap();
        let b = V2PairedSubjectContract::freeze(
            target(0.0),
            comparator(ShortcutBaselineKind::SimpleMarkov, 0),
        )
        .unwrap();
        assert_ne!(a.comparator_subject_commitment(), b.comparator_subject_commitment());
        assert_ne!(a.commitment(), b.commitment());
    }

    #[test]
    fn changing_comparator_development_evidence_changes_paired_subject() {
        let a = V2PairedSubjectContract::freeze(
            target(0.0),
            comparator(ShortcutBaselineKind::NearestTransition, 0),
        )
        .unwrap();
        let b = V2PairedSubjectContract::freeze(
            target(0.0),
            comparator(ShortcutBaselineKind::NearestTransition, 1),
        )
        .unwrap();
        assert_ne!(a.comparator_subject_commitment(), b.comparator_subject_commitment());
        assert_ne!(a.commitment(), b.commitment());
    }
}

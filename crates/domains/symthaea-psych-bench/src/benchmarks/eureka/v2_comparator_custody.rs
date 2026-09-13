// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! EUREKA-002 V2 comparator custody grammar.
//!
//! This module is deliberately source/test-only. It freezes canonical public
//! Development evidence and cryptographic comparator identity before any V2
//! comparator builder or HeldOut runner exists. It does not fit a baseline and
//! cannot execute predictions.

use std::collections::BTreeSet;

use super::analysis_plan::{ANALYSIS_PLAN_REVISION, EUREKA_002_ANALYSIS_PLAN_V1};
use super::baselines::ShortcutBaselineKind;
use super::hidden_world::PublicAction;
use super::v2_public_schema::{
    V2PublicFamily, V2PublicSchemaError, V2PublicState, action_index,
    public_schema_commitment,
};

pub(super) const V2_COMPARATOR_FIT_CORPUS_REVISION: &str =
    "EUREKA.002.V2.COMPARATOR_FIT_CORPUS.v2";
pub(super) const V2_COMPARATOR_CUSTODY_REVISION: &str =
    "EUREKA.002.V2.COMPARATOR_CUSTODY.v3";
pub(super) const V2_SHORTCUT_BASELINE_IMPLEMENTATION_REVISION: &str =
    "EUREKA.002.V2.SHORTCUT_BASELINES.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum V2CorpusPartition {
    Development,
    Calibration,
    HeldOut,
    ExternalReplication,
}

impl V2CorpusPartition {
    const fn tag(self) -> u8 {
        match self {
            Self::Development => 1,
            Self::Calibration => 2,
            Self::HeldOut => 3,
            Self::ExternalReplication => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2ComparatorCustodyError {
    PublicSchema(V2PublicSchemaError),
    EmptyFitCorpus,
    ZeroCanonicalRowIdentity,
    FamilyContextMismatch,
    ContextChangedAcrossTransition,
    NonDevelopmentFitEvidence,
    DuplicateCanonicalRowIdentity,
    DuplicateCanonicalTransition,
    ZeroSelectionAuthorizationCommitment,
    ComparatorNotPreregistered,
}

impl From<V2PublicSchemaError> for V2ComparatorCustodyError {
    fn from(value: V2PublicSchemaError) -> Self {
        Self::PublicSchema(value)
    }
}

/// Exact target/comparator-visible transition evidence.
///
/// Hidden mechanism state, evaluator seed, oracle annotations, and model output
/// are structurally absent from this record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2PublicTransitionEvidence {
    family: V2PublicFamily,
    partition: V2CorpusPartition,
    row_identity: [u8; 32],
    pre: V2PublicState,
    action: PublicAction,
    post: V2PublicState,
}

impl V2PublicTransitionEvidence {
    pub(super) fn new(
        family: V2PublicFamily,
        partition: V2CorpusPartition,
        row_identity: [u8; 32],
        pre: V2PublicState,
        action: PublicAction,
        post: V2PublicState,
    ) -> Result<Self, V2ComparatorCustodyError> {
        if row_identity == [0_u8; 32] {
            return Err(V2ComparatorCustodyError::ZeroCanonicalRowIdentity);
        }
        action_index(action)?;
        if !pre.belongs_to(family) || !post.belongs_to(family) {
            return Err(V2ComparatorCustodyError::FamilyContextMismatch);
        }
        if pre.context() != post.context() {
            return Err(V2ComparatorCustodyError::ContextChangedAcrossTransition);
        }
        Ok(Self {
            family,
            partition,
            row_identity,
            pre,
            action,
            post,
        })
    }

    pub(super) const fn row_identity(&self) -> [u8; 32] {
        self.row_identity
    }

    pub(super) const fn partition(&self) -> V2CorpusPartition {
        self.partition
    }
}

/// Canonically ordered Development-only fit evidence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct V2DevelopmentFitCorpus {
    records: Vec<V2PublicTransitionEvidence>,
    schema_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2DevelopmentFitCorpus {
    pub(super) fn freeze(
        mut records: Vec<V2PublicTransitionEvidence>,
    ) -> Result<Self, V2ComparatorCustodyError> {
        if records.is_empty() {
            return Err(V2ComparatorCustodyError::EmptyFitCorpus);
        }
        if records
            .iter()
            .any(|record| record.partition != V2CorpusPartition::Development)
        {
            return Err(V2ComparatorCustodyError::NonDevelopmentFitEvidence);
        }

        let mut seen_ids = BTreeSet::new();
        let mut seen_transitions = BTreeSet::new();
        for record in &records {
            if !seen_ids.insert(record.row_identity) {
                return Err(V2ComparatorCustodyError::DuplicateCanonicalRowIdentity);
            }
            if !seen_transitions.insert(canonical_transition_semantics_bytes(record)) {
                return Err(V2ComparatorCustodyError::DuplicateCanonicalTransition);
            }
        }
        records.sort_by_key(|record| record.row_identity);

        let schema_commitment = public_schema_commitment();
        let commitment = fit_corpus_commitment(schema_commitment, &records);
        Ok(Self {
            records,
            schema_commitment,
            commitment,
        })
    }

    pub(super) fn records(&self) -> &[V2PublicTransitionEvidence] {
        &self.records
    }

    pub(super) const fn schema_commitment(&self) -> [u8; 32] {
        self.schema_commitment
    }

    pub(super) const fn commitment(&self) -> [u8; 32] {
        self.commitment
    }
}

/// Cryptographic identity of the comparator subject *before* any HeldOut query.
///
/// This receipt binds custody only. A later tranche must prove that an actual
/// prediction-only fitted object corresponds exactly to this receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2ComparatorCustodyReceipt {
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    selected: ShortcutBaselineKind,
    selection_authorization_commitment: [u8; 32],
    development_record_count: u32,
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
    commitment: [u8; 32],
}

impl V2ComparatorCustodyReceipt {
    pub(super) fn freeze(
        corpus: &V2DevelopmentFitCorpus,
        selected: ShortcutBaselineKind,
        selection_authorization_commitment: [u8; 32],
    ) -> Result<Self, V2ComparatorCustodyError> {
        if selection_authorization_commitment == [0_u8; 32] {
            return Err(V2ComparatorCustodyError::ZeroSelectionAuthorizationCommitment);
        }
        if !EUREKA_002_ANALYSIS_PLAN_V1
            .eligible_comparators
            .contains(&selected)
        {
            return Err(V2ComparatorCustodyError::ComparatorNotPreregistered);
        }
        let development_record_count =
            u32::try_from(corpus.records.len()).expect("V2 Development corpus fits u32");
        let analysis_plan_replay_digest = EUREKA_002_ANALYSIS_PLAN_V1.replay_digest();
        let analysis_plan_commitment = EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment();
        let commitment = comparator_subject_commitment(
            corpus.schema_commitment,
            corpus.commitment,
            selected,
            selection_authorization_commitment,
            development_record_count,
            analysis_plan_replay_digest,
            analysis_plan_commitment,
        );
        Ok(Self {
            schema_commitment: corpus.schema_commitment,
            fit_corpus_commitment: corpus.commitment,
            selected,
            selection_authorization_commitment,
            development_record_count,
            analysis_plan_replay_digest,
            analysis_plan_commitment,
            commitment,
        })
    }

    pub(super) const fn schema_commitment(self) -> [u8; 32] {
        self.schema_commitment
    }

    pub(super) const fn fit_corpus_commitment(self) -> [u8; 32] {
        self.fit_corpus_commitment
    }

    pub(super) const fn selected(self) -> ShortcutBaselineKind {
        self.selected
    }

    pub(super) const fn development_record_count(self) -> u32 {
        self.development_record_count
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

fn fit_corpus_commitment(
    schema_commitment: [u8; 32],
    records: &[V2PublicTransitionEvidence],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_COMPARATOR_FIT_CORPUS_REVISION.as_bytes());
    bytes.extend_from_slice(&schema_commitment);
    bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
    for record in records {
        encode_record(&mut bytes, record);
    }
    *blake3::hash(&bytes).as_bytes()
}

fn comparator_subject_commitment(
    schema_commitment: [u8; 32],
    fit_corpus_commitment: [u8; 32],
    selected: ShortcutBaselineKind,
    selection_authorization_commitment: [u8; 32],
    development_record_count: u32,
    analysis_plan_replay_digest: u64,
    analysis_plan_commitment: [u8; 32],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_COMPARATOR_CUSTODY_REVISION.as_bytes());
    encode_bytes(
        &mut bytes,
        V2_SHORTCUT_BASELINE_IMPLEMENTATION_REVISION.as_bytes(),
    );
    encode_bytes(&mut bytes, ANALYSIS_PLAN_REVISION.as_bytes());
    bytes.extend_from_slice(&analysis_plan_replay_digest.to_le_bytes());
    bytes.extend_from_slice(&analysis_plan_commitment);
    bytes.extend_from_slice(&schema_commitment);
    bytes.extend_from_slice(&fit_corpus_commitment);
    encode_bytes(&mut bytes, selected.stable_id().as_bytes());
    bytes.extend_from_slice(&selection_authorization_commitment);
    bytes.extend_from_slice(&development_record_count.to_le_bytes());
    *blake3::hash(&bytes).as_bytes()
}

/// Exact transition semantics excluding both row identity and partition.
///
/// This key answers only whether two public transitions carry the same
/// family/pre/action/post semantics. Partition remains bound separately in
/// corpus/evidence commitments. Keeping it out of this key lets Development vs
/// Calibration duplicate content fail locally even when their row IDs differ.
pub(super) fn canonical_transition_semantics_bytes(
    record: &V2PublicTransitionEvidence,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(record.family.tag());
    encode_state(&mut bytes, record.pre);
    bytes.extend_from_slice(
        &(action_index(record.action).expect("record action validated") as u64).to_le_bytes(),
    );
    encode_state(&mut bytes, record.post);
    bytes
}

fn encode_record(bytes: &mut Vec<u8>, record: &V2PublicTransitionEvidence) {
    bytes.push(record.family.tag());
    bytes.push(record.partition.tag());
    bytes.extend_from_slice(&record.row_identity);
    encode_state(bytes, record.pre);
    bytes.extend_from_slice(
        &(action_index(record.action).expect("record action validated") as u64).to_le_bytes(),
    );
    encode_state(bytes, record.post);
}

fn encode_state(bytes: &mut Vec<u8>, state: V2PublicState) {
    for value in state.fields() {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        id: u8,
        partition: V2CorpusPartition,
        family: V2PublicFamily,
        pre: [i32; 4],
        action: PublicAction,
        post: [i32; 4],
    ) -> V2PublicTransitionEvidence {
        let mut row_identity = [0_u8; 32];
        row_identity[0] = id;
        V2PublicTransitionEvidence::new(
            family,
            partition,
            row_identity,
            V2PublicState::new(pre).unwrap(),
            action,
            V2PublicState::new(post).unwrap(),
        )
        .unwrap()
    }

    fn development_records() -> Vec<V2PublicTransitionEvidence> {
        vec![
            record(
                1,
                V2CorpusPartition::Development,
                V2PublicFamily::PublicFlowV2,
                [3, 4, 5, 0],
                PublicAction::Pulse { slot: 0 },
                [2, 5, 5, 0],
            ),
            record(
                2,
                V2CorpusPartition::Development,
                V2PublicFamily::PublicRelayV2,
                [7, 2, 1, 5],
                PublicAction::Pulse { slot: 1 },
                [7, 7, 1, 5],
            ),
            record(
                3,
                V2CorpusPartition::Development,
                V2PublicFamily::PublicFlowV2,
                [8, 9, 4, 1],
                PublicAction::NoOp,
                [9, 9, 4, 1],
            ),
        ]
    }

    #[test]
    fn fit_corpus_commitment_is_order_invariant() {
        let forward = V2DevelopmentFitCorpus::freeze(development_records()).unwrap();
        let mut reversed = development_records();
        reversed.reverse();
        let reverse = V2DevelopmentFitCorpus::freeze(reversed).unwrap();
        assert_eq!(forward.records(), reverse.records());
        assert_eq!(forward.commitment(), reverse.commitment());
    }

    #[test]
    fn changing_public_transition_changes_fit_corpus_commitment() {
        let a = V2DevelopmentFitCorpus::freeze(development_records()).unwrap();
        let mut changed = development_records();
        changed[0] = record(
            1,
            V2CorpusPartition::Development,
            V2PublicFamily::PublicFlowV2,
            [3, 4, 5, 0],
            PublicAction::Pulse { slot: 0 },
            [2, 6, 5, 0],
        );
        let b = V2DevelopmentFitCorpus::freeze(changed).unwrap();
        assert_ne!(a.commitment(), b.commitment());
    }

    #[test]
    fn zero_row_identity_fails_closed() {
        let result = V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            [0_u8; 32],
            V2PublicState::new([1, 2, 3, 0]).unwrap(),
            PublicAction::NoOp,
            V2PublicState::new([1, 2, 3, 0]).unwrap(),
        );
        assert_eq!(result, Err(V2ComparatorCustodyError::ZeroCanonicalRowIdentity));
    }

    #[test]
    fn family_context_mismatch_fails_closed() {
        let result = V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            [1_u8; 32],
            V2PublicState::new([1, 2, 3, 5]).unwrap(),
            PublicAction::NoOp,
            V2PublicState::new([1, 2, 3, 5]).unwrap(),
        );
        assert_eq!(result, Err(V2ComparatorCustodyError::FamilyContextMismatch));
    }

    #[test]
    fn changing_context_within_transition_fails_closed() {
        let result = V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            [1_u8; 32],
            V2PublicState::new([1, 2, 3, 0]).unwrap(),
            PublicAction::NoOp,
            V2PublicState::new([1, 2, 3, 1]).unwrap(),
        );
        assert_eq!(
            result,
            Err(V2ComparatorCustodyError::ContextChangedAcrossTransition)
        );
    }

    #[test]
    fn duplicate_row_identity_fails_closed() {
        let mut records = development_records();
        let duplicate = records[0].clone();
        records.push(duplicate);
        assert_eq!(
            V2DevelopmentFitCorpus::freeze(records),
            Err(V2ComparatorCustodyError::DuplicateCanonicalRowIdentity)
        );
    }

    #[test]
    fn duplicate_transition_under_different_identity_fails_closed() {
        let mut records = development_records();
        let mut duplicate = records[0].clone();
        duplicate.row_identity = [0xA5_u8; 32];
        records.push(duplicate);
        assert_eq!(
            V2DevelopmentFitCorpus::freeze(records),
            Err(V2ComparatorCustodyError::DuplicateCanonicalTransition)
        );
    }

    #[test]
    fn nondevelopment_fit_evidence_fails_closed() {
        for partition in [
            V2CorpusPartition::Calibration,
            V2CorpusPartition::HeldOut,
            V2CorpusPartition::ExternalReplication,
        ] {
            let records = vec![record(
                9,
                partition,
                V2PublicFamily::PublicFlowV2,
                [1, 2, 3, 0],
                PublicAction::NoOp,
                [1, 2, 3, 0],
            )];
            assert_eq!(
                V2DevelopmentFitCorpus::freeze(records),
                Err(V2ComparatorCustodyError::NonDevelopmentFitEvidence)
            );
        }
    }

    #[test]
    fn unsupported_action_cannot_enter_canonical_fit_evidence() {
        let result = V2PublicTransitionEvidence::new(
            V2PublicFamily::PublicFlowV2,
            V2CorpusPartition::Development,
            [7_u8; 32],
            V2PublicState::new([1, 2, 3, 0]).unwrap(),
            PublicAction::Pulse { slot: 3 },
            V2PublicState::new([1, 2, 3, 0]).unwrap(),
        );
        assert_eq!(
            result,
            Err(V2ComparatorCustodyError::PublicSchema(
                V2PublicSchemaError::UnsupportedAction
            ))
        );
    }

    #[test]
    fn zero_selection_authorization_cannot_mint_comparator_subject() {
        let corpus = V2DevelopmentFitCorpus::freeze(development_records()).unwrap();
        assert_eq!(
            V2ComparatorCustodyReceipt::freeze(
                &corpus,
                ShortcutBaselineKind::NearestTransition,
                [0_u8; 32],
            ),
            Err(V2ComparatorCustodyError::ZeroSelectionAuthorizationCommitment)
        );
    }

    #[test]
    fn selected_kind_and_selection_authorization_bind_subject_identity() {
        let corpus = V2DevelopmentFitCorpus::freeze(development_records()).unwrap();
        let auth_a = [0xAA_u8; 32];
        let auth_b = [0xBB_u8; 32];
        let nearest = V2ComparatorCustodyReceipt::freeze(
            &corpus,
            ShortcutBaselineKind::NearestTransition,
            auth_a,
        )
        .unwrap();
        let markov = V2ComparatorCustodyReceipt::freeze(
            &corpus,
            ShortcutBaselineKind::SimpleMarkov,
            auth_a,
        )
        .unwrap();
        let different_auth = V2ComparatorCustodyReceipt::freeze(
            &corpus,
            ShortcutBaselineKind::NearestTransition,
            auth_b,
        )
        .unwrap();
        assert_ne!(nearest.commitment(), markov.commitment());
        assert_ne!(nearest.commitment(), different_auth.commitment());
        assert_eq!(nearest.schema_commitment(), public_schema_commitment());
        assert_eq!(nearest.fit_corpus_commitment(), corpus.commitment());
        assert_eq!(nearest.development_record_count(), 3);
        assert_eq!(
            nearest.analysis_plan_replay_digest(),
            EUREKA_002_ANALYSIS_PLAN_V1.replay_digest()
        );
        assert_eq!(
            nearest.analysis_plan_commitment(),
            EUREKA_002_ANALYSIS_PLAN_V1.cryptographic_commitment()
        );
    }

    #[test]
    fn custody_receipt_is_deterministic_for_same_inputs() {
        let corpus = V2DevelopmentFitCorpus::freeze(development_records()).unwrap();
        let auth = [0x42_u8; 32];
        let a = V2ComparatorCustodyReceipt::freeze(
            &corpus,
            ShortcutBaselineKind::ActionMarginalDelta,
            auth,
        )
        .unwrap();
        let b = V2ComparatorCustodyReceipt::freeze(
            &corpus,
            ShortcutBaselineKind::ActionMarginalDelta,
            auth,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_ne!(a.commitment(), [0_u8; 32]);
    }
}

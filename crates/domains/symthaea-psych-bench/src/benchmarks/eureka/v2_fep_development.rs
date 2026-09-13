// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Development-only production-FEP learning for EUREKA-002 V2.
//!
//! This module receives one target-blind Development plan, learns through the
//! existing detached production FEP session, freezes the learned predictor, and
//! immediately consumes that snapshot into `FepHeldOutSubject`. No trainable
//! FEP capability escapes the returned artifact.

#![allow(dead_code)]

use symthaea::cognitive_loop::CognitiveLoopService;
use symthaea_fep::{
    ActionOutcome, FepHeldOutSubject, FepPredictionSessionError, FrozenPredictionCommitment,
};

use super::v2_corpus_schedule::{V2ScheduleMaterializationError, V2SchedulePartition};
use super::v2_development_order::{
    V2_DEVELOPMENT_ORDER_REVISION, V2_DEVELOPMENT_ROWS_PER_FAMILY,
    V2_DEVELOPMENT_ROWS_TOTAL, V2DevelopmentPlan, materialize_development_plan,
};
use super::v2_public_schema::{
    V2_ACTION_COUNT, V2_OBSERVATION_DIM, V2PublicFamily, public_schema_commitment,
};
use super::v2_target_contract::{
    V2_FEP_TARGET_ADAPTER_REVISION, V2FepAdapter, V2FepTargetContract, V2TargetContractError,
};

pub(super) const V2_FEP_DEVELOPMENT_RUNNER_REVISION: &str =
    "EUREKA.002.V2.FEP_DEVELOPMENT.v1";
const V2_FEP_DEVELOPMENT_MODALITY: &str = "eureka-v2-development";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum V2FepDevelopmentError {
    Materialization(V2ScheduleMaterializationError),
    Session(FepPredictionSessionError),
    TargetContract(V2TargetContractError),
    ObservationDimensionMismatch { expected: usize, actual: usize },
    InsufficientActionCapacity { required: usize, actual: usize },
    WrongDevelopmentCount,
    WrongPartition,
    WrongFamilyPairOrder,
    FamilyHistogramMismatch,
    ActionHistogramMismatch,
    PredictionActionMismatch { expected: usize, actual: usize },
    SourceServiceReplayChanged,
    SourceServiceCommitmentChanged,
    LearnedSubjectContractMismatch,
}

impl From<V2ScheduleMaterializationError> for V2FepDevelopmentError {
    fn from(value: V2ScheduleMaterializationError) -> Self {
        Self::Materialization(value)
    }
}

impl From<FepPredictionSessionError> for V2FepDevelopmentError {
    fn from(value: FepPredictionSessionError) -> Self {
        Self::Session(value)
    }
}

impl From<V2TargetContractError> for V2FepDevelopmentError {
    fn from(value: V2TargetContractError) -> Self {
        Self::TargetContract(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct V2FepDevelopmentReceipt {
    runner_revision: &'static str,
    target_adapter_revision: &'static str,
    development_order_revision: &'static str,
    public_schema_commitment: [u8; 32],
    development_plan_commitment: [u8; 32],
    full_schedule_root: [u8; 32],
    development_corpus_commitment: [u8; 32],
    development_order_root: [u8; 32],
    initial_snapshot_replay_digest: u64,
    initial_snapshot_commitment: FrozenPredictionCommitment,
    learned_snapshot_replay_digest: u64,
    learned_snapshot_commitment: FrozenPredictionCommitment,
    target_contract_commitment: [u8; 32],
    fep_custody_commitment: [u8; 32],
    prediction_trace_root: [u8; 32],
    world_count: u32,
    prediction_count: u32,
    learning_count: u32,
    family_histogram: [u32; 2],
    action_histogram: [u32; V2_ACTION_COUNT as usize],
    commitment: [u8; 32],
}

impl V2FepDevelopmentReceipt {
    pub(super) const fn development_plan_commitment(self) -> [u8; 32] {
        self.development_plan_commitment
    }

    pub(super) const fn full_schedule_root(self) -> [u8; 32] {
        self.full_schedule_root
    }

    pub(super) const fn development_corpus_commitment(self) -> [u8; 32] {
        self.development_corpus_commitment
    }

    pub(super) const fn development_order_root(self) -> [u8; 32] {
        self.development_order_root
    }

    pub(super) const fn initial_snapshot_commitment(self) -> FrozenPredictionCommitment {
        self.initial_snapshot_commitment
    }

    pub(super) const fn learned_snapshot_commitment(self) -> FrozenPredictionCommitment {
        self.learned_snapshot_commitment
    }

    pub(super) const fn learned_snapshot_replay_digest(self) -> u64 {
        self.learned_snapshot_replay_digest
    }

    pub(super) const fn target_contract_commitment(self) -> [u8; 32] {
        self.target_contract_commitment
    }

    pub(super) const fn prediction_trace_root(self) -> [u8; 32] {
        self.prediction_trace_root
    }

    pub(super) const fn world_count(self) -> u32 {
        self.world_count
    }

    pub(super) const fn prediction_count(self) -> u32 {
        self.prediction_count
    }

    pub(super) const fn learning_count(self) -> u32 {
        self.learning_count
    }

    pub(super) const fn family_histogram(self) -> [u32; 2] {
        self.family_histogram
    }

    pub(super) const fn action_histogram(self) -> [u32; V2_ACTION_COUNT as usize] {
        self.action_histogram
    }

    pub(super) const fn commitment(self) -> [u8; 32] {
        self.commitment
    }
}

/// Runner-facing Development artifact. Deliberately not Clone: ownership of the
/// sealed FEP subject remains singular. No snapshot/session/agent field exists.
#[derive(Debug)]
pub(super) struct V2FepDevelopmentArtifact {
    subject: FepHeldOutSubject,
    target_contract: V2FepTargetContract,
    receipt: V2FepDevelopmentReceipt,
}

impl V2FepDevelopmentArtifact {
    pub(super) fn subject(&self) -> &FepHeldOutSubject {
        &self.subject
    }

    pub(super) const fn target_contract(&self) -> V2FepTargetContract {
        self.target_contract
    }

    pub(super) const fn receipt(&self) -> V2FepDevelopmentReceipt {
        self.receipt
    }
}

/// Canonical Development entry point. The target never receives a partition
/// selector or caller-authored training rows.
pub(super) fn run_canonical_v2_fep_development(
    service: &CognitiveLoopService,
) -> Result<V2FepDevelopmentArtifact, V2FepDevelopmentError> {
    let plan = materialize_development_plan()?;
    validate_development_plan(&plan)?;
    run_plan(service, &plan)
}

fn run_plan(
    service: &CognitiveLoopService,
    plan: &V2DevelopmentPlan,
) -> Result<V2FepDevelopmentArtifact, V2FepDevelopmentError> {
    let (initial_snapshot_replay_digest, initial_snapshot_commitment) = {
        let source_session = service.fep_prediction_session();
        if source_session.observation_dim() != V2_OBSERVATION_DIM {
            return Err(V2FepDevelopmentError::ObservationDimensionMismatch {
                expected: V2_OBSERVATION_DIM,
                actual: source_session.observation_dim(),
            });
        }
        if source_session.action_count() < usize::from(V2_ACTION_COUNT) {
            return Err(V2FepDevelopmentError::InsufficientActionCapacity {
                required: usize::from(V2_ACTION_COUNT),
                actual: source_session.action_count(),
            });
        }
        let initial_snapshot = source_session.freeze_for_evaluation()?;
        (
            initial_snapshot.replay_digest(),
            initial_snapshot.commitment(),
        )
    };

    let mut session = service.fep_prediction_session();
    let mut prediction_trace = Vec::new();
    let mut prediction_count = 0_u32;
    let mut learning_count = 0_u32;
    let mut family_histogram = [0_u32; 2];
    let mut action_histogram = [0_u32; V2_ACTION_COUNT as usize];

    for row in plan.ordered_rows().iter().copied() {
        session.reset_transient_state_preserving_model();

        let encoded_pre = V2FepAdapter::encode_state(row.pre());
        let target_action = V2FepAdapter::encode_action(row.action())?;
        session.observe(&encoded_pre, 1.0, V2_FEP_DEVELOPMENT_MODALITY)?;

        // The Development prediction is frozen in the trace before the known
        // Development outcome is fed into learning. It is diagnostic trace
        // evidence only, never confirmatory HeldOut evidence.
        let prediction = session.predict(target_action)?;
        if prediction.action != target_action {
            return Err(V2FepDevelopmentError::PredictionActionMismatch {
                expected: target_action,
                actual: prediction.action,
            });
        }
        encode_prediction_trace(&mut prediction_trace, row.row_identity(), &prediction);
        prediction_count = prediction_count.saturating_add(1);

        let encoded_post = V2FepAdapter::encode_state(row.post());
        session.learn_from_actual(
            target_action,
            &encoded_post,
            1.0,
            V2_FEP_DEVELOPMENT_MODALITY,
        )?;
        learning_count = learning_count.saturating_add(1);
        let family_slot = family_index(row.family());
        family_histogram[family_slot] = family_histogram[family_slot].saturating_add(1);
        action_histogram[target_action] = action_histogram[target_action].saturating_add(1);
    }

    let learned_snapshot = session.freeze_for_evaluation()?;
    let learned_snapshot_replay_digest = learned_snapshot.replay_digest();
    let learned_snapshot_commitment = learned_snapshot.commitment();

    // Detached learning must never mutate the source production service.
    let source_after = service.fep_prediction_session().freeze_for_evaluation()?;
    if source_after.replay_digest() != initial_snapshot_replay_digest {
        return Err(V2FepDevelopmentError::SourceServiceReplayChanged);
    }
    if source_after.commitment() != initial_snapshot_commitment {
        return Err(V2FepDevelopmentError::SourceServiceCommitmentChanged);
    }

    // Consume the trainable snapshot into held-out-only authority immediately.
    let subject = FepHeldOutSubject::seal(learned_snapshot);
    let target_contract = V2FepTargetContract::from_heldout_subject(&subject)?;
    if target_contract.learned_subject_commitment() != subject.commitment()
        || target_contract.learned_subject_commitment() != learned_snapshot_commitment
    {
        return Err(V2FepDevelopmentError::LearnedSubjectContractMismatch);
    }

    let prediction_trace_root = prediction_trace_root(&prediction_trace);
    let world_count = u32::try_from(plan.ordered_rows().len()).expect("512 rows fit u32");
    let commitment = development_receipt_commitment(
        plan,
        initial_snapshot_replay_digest,
        initial_snapshot_commitment,
        learned_snapshot_replay_digest,
        learned_snapshot_commitment,
        target_contract,
        prediction_trace_root,
        world_count,
        prediction_count,
        learning_count,
        family_histogram,
        action_histogram,
    );
    let receipt = V2FepDevelopmentReceipt {
        runner_revision: V2_FEP_DEVELOPMENT_RUNNER_REVISION,
        target_adapter_revision: V2_FEP_TARGET_ADAPTER_REVISION,
        development_order_revision: V2_DEVELOPMENT_ORDER_REVISION,
        public_schema_commitment: public_schema_commitment(),
        development_plan_commitment: plan.commitment(),
        full_schedule_root: plan.full_schedule_root(),
        development_corpus_commitment: plan.development_corpus().commitment(),
        development_order_root: plan.development_order_root(),
        initial_snapshot_replay_digest,
        initial_snapshot_commitment,
        learned_snapshot_replay_digest,
        learned_snapshot_commitment,
        target_contract_commitment: target_contract.commitment(),
        fep_custody_commitment: target_contract.fep_custody_commitment(),
        prediction_trace_root,
        world_count,
        prediction_count,
        learning_count,
        family_histogram,
        action_histogram,
        commitment,
    };

    Ok(V2FepDevelopmentArtifact {
        subject,
        target_contract,
        receipt,
    })
}

fn validate_development_plan(plan: &V2DevelopmentPlan) -> Result<(), V2FepDevelopmentError> {
    let rows = plan.ordered_rows();
    if rows.len() != V2_DEVELOPMENT_ROWS_TOTAL
        || plan.development_corpus().records().len() != V2_DEVELOPMENT_ROWS_TOTAL
    {
        return Err(V2FepDevelopmentError::WrongDevelopmentCount);
    }

    let mut family_histogram = [0_usize; 2];
    let mut action_histogram = [0_usize; V2_ACTION_COUNT as usize];
    for pair in rows.chunks_exact(2) {
        if pair[0].partition() != V2SchedulePartition::Development
            || pair[1].partition() != V2SchedulePartition::Development
        {
            return Err(V2FepDevelopmentError::WrongPartition);
        }
        if pair[0].family() != V2PublicFamily::PublicFlowV2
            || pair[1].family() != V2PublicFamily::PublicRelayV2
        {
            return Err(V2FepDevelopmentError::WrongFamilyPairOrder);
        }
    }
    for row in rows.iter().copied() {
        if row.partition() != V2SchedulePartition::Development {
            return Err(V2FepDevelopmentError::WrongPartition);
        }
        family_histogram[family_index(row.family())] += 1;
        action_histogram[usize::from(row.action_index())] += 1;
    }
    if family_histogram != [V2_DEVELOPMENT_ROWS_PER_FAMILY; 2] {
        return Err(V2FepDevelopmentError::FamilyHistogramMismatch);
    }
    if action_histogram != [128_usize; V2_ACTION_COUNT as usize] {
        return Err(V2FepDevelopmentError::ActionHistogramMismatch);
    }
    Ok(())
}

const fn family_index(family: V2PublicFamily) -> usize {
    match family {
        V2PublicFamily::PublicFlowV2 => 0,
        V2PublicFamily::PublicRelayV2 => 1,
    }
}

fn encode_prediction_trace(
    bytes: &mut Vec<u8>,
    row_identity: [u8; 32],
    prediction: &ActionOutcome,
) {
    bytes.extend_from_slice(&row_identity);
    bytes.extend_from_slice(&(prediction.action as u64).to_le_bytes());
    encode_f64_slice(bytes, &prediction.predicted_next_state.mean);
    encode_f64_slice(bytes, &prediction.predicted_next_state.precision);
    encode_f64_slice(bytes, &prediction.expected_observation);
    bytes.extend_from_slice(&prediction.timestamp.to_le_bytes());
}

fn prediction_trace_root(trace: &[u8]) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(
        &mut bytes,
        b"EUREKA.002.V2.FEP_DEVELOPMENT_PREDICTION_TRACE.v1",
    );
    bytes.extend_from_slice(trace);
    *blake3::hash(&bytes).as_bytes()
}

#[allow(clippy::too_many_arguments)]
fn development_receipt_commitment(
    plan: &V2DevelopmentPlan,
    initial_snapshot_replay_digest: u64,
    initial_snapshot_commitment: FrozenPredictionCommitment,
    learned_snapshot_replay_digest: u64,
    learned_snapshot_commitment: FrozenPredictionCommitment,
    target_contract: V2FepTargetContract,
    prediction_trace_root: [u8; 32],
    world_count: u32,
    prediction_count: u32,
    learning_count: u32,
    family_histogram: [u32; 2],
    action_histogram: [u32; V2_ACTION_COUNT as usize],
) -> [u8; 32] {
    let mut bytes = Vec::new();
    encode_bytes(&mut bytes, V2_FEP_DEVELOPMENT_RUNNER_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_FEP_TARGET_ADAPTER_REVISION.as_bytes());
    encode_bytes(&mut bytes, V2_DEVELOPMENT_ORDER_REVISION.as_bytes());
    bytes.extend_from_slice(&public_schema_commitment());
    bytes.extend_from_slice(&plan.commitment());
    bytes.extend_from_slice(&plan.full_schedule_root());
    bytes.extend_from_slice(&plan.development_corpus().commitment());
    bytes.extend_from_slice(&plan.development_order_root());
    bytes.extend_from_slice(&initial_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(initial_snapshot_commitment.as_bytes());
    bytes.extend_from_slice(&learned_snapshot_replay_digest.to_le_bytes());
    bytes.extend_from_slice(learned_snapshot_commitment.as_bytes());
    bytes.extend_from_slice(&target_contract.commitment());
    bytes.extend_from_slice(&target_contract.fep_custody_commitment());
    bytes.extend_from_slice(&prediction_trace_root);
    bytes.extend_from_slice(&world_count.to_le_bytes());
    bytes.extend_from_slice(&prediction_count.to_le_bytes());
    bytes.extend_from_slice(&learning_count.to_le_bytes());
    for count in family_histogram {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    for count in action_histogram {
        bytes.extend_from_slice(&count.to_le_bytes());
    }
    *blake3::hash(&bytes).as_bytes()
}

fn encode_f64_slice(bytes: &mut Vec<u8>, values: &[f64]) {
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
}

fn encode_bytes(bytes: &mut Vec<u8>, value: &[u8]) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

    fn service() -> CognitiveLoopService {
        CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap()
    }

    #[test]
    fn canonical_runner_trains_exact_development_and_seals_subject() {
        let service = service();
        let plan = materialize_development_plan().unwrap();
        let before = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap();
        let before_replay = before.replay_digest();
        let before_commitment = before.commitment();

        let artifact = run_canonical_v2_fep_development(&service).unwrap();
        let receipt = artifact.receipt();
        let after = service
            .fep_prediction_session()
            .freeze_for_evaluation()
            .unwrap();

        assert_eq!(after.replay_digest(), before_replay);
        assert_eq!(after.commitment(), before_commitment);
        assert_eq!(receipt.world_count(), 512);
        assert_eq!(receipt.prediction_count(), 512);
        assert_eq!(receipt.learning_count(), 512);
        assert_eq!(receipt.family_histogram(), [256, 256]);
        assert_eq!(receipt.action_histogram(), [128, 128, 128, 128]);
        assert_ne!(receipt.prediction_trace_root(), [0_u8; 32]);
        assert_ne!(receipt.commitment(), [0_u8; 32]);
        assert_eq!(receipt.development_plan_commitment(), plan.commitment());
        assert_eq!(receipt.full_schedule_root(), plan.full_schedule_root());
        assert_eq!(
            receipt.development_corpus_commitment(),
            plan.development_corpus().commitment()
        );
        assert_eq!(
            receipt.development_order_root(),
            plan.development_order_root()
        );
        assert_eq!(
            artifact.subject().commitment(),
            receipt.learned_snapshot_commitment()
        );
        assert_eq!(
            artifact.target_contract().learned_subject_commitment(),
            artifact.subject().commitment()
        );
        assert_eq!(
            artifact.target_contract().commitment(),
            receipt.target_contract_commitment()
        );
        assert_ne!(
            receipt.initial_snapshot_commitment(),
            receipt.learned_snapshot_commitment(),
            "default production service has model learning enabled; canonical Development must change predictor identity"
        );
    }

    #[test]
    fn training_source_has_no_confirmatory_execution_reachability() {
        let whole_source = include_str!("v2_fep_development.rs");
        let production_source = whole_source
            .split("#[cfg(test)]")
            .next()
            .expect("Development runner has a production section");
        for forbidden in [
            "materialize_canonical_corpora",
            "execute_calibration_selection",
            "FepEvaluationTrial",
            "FepEvaluationSnapshot",
            "score_consequence",
        ] {
            assert!(
                !production_source.contains(forbidden),
                "Development runner must not reach confirmatory/evaluation authority: {forbidden}"
            );
        }
    }
}

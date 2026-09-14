// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified reasoning-chain transition history.
//!
//! The legacy primitive selector mixes several behavioral learning signals, so its aggregate is not
//! a qualified measurement source. This module keeps a separate measurement-only authority plane.
//!
//! A claimed `ReasoningChain` is never trusted by structure alone. Every execution is replayed
//! through production `ReasoningChain::execute_primitive`. Only a chain whose continuity, outputs,
//! local transition contribution, final state, and accumulated total all agree with production
//! replay can enter history. Admission is transactional: a rejected chain contributes nothing.
//!
//! Important semantic boundary: production currently computes the legacy field
//! `phi_contribution = (1 - HammingSimilarity(input, output)) * 0.1`. After normalization this is a
//! transition-distance statistic. It is **not** IntegrationProxy evidence, IIT Phi, consciousness,
//! reasoning correctness, or expected utility. This module therefore never mutates canonical
//! objective evidence and never calls the V3 planner.

use super::reasoning_active_primitive_evidence::{
    adapt_active_primitive_evidence, ActivePrimitiveEvidenceError, ActivePrimitiveEvidenceReport,
};
use crate::consciousness::primitive_reasoning::{ReasoningChain, TaskType, TransformationType};
use crate::consciousness::ActivePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const VERIFIED_TRANSITION_HISTORY_VERSION: &str = "rq-006x-verified-transition-history-v2";
pub const LOCAL_TRANSITION_CONTRIBUTION_SCALE: f64 = 0.1;
const NUMERIC_TOLERANCE: f64 = 1.0e-12;
const CHAIN_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/verified-transition-chain/v2";
const HISTORY_GENESIS_DOMAIN: &[u8] = b"symthaea/reasoning/verified-transition-history/genesis/v2";
const HISTORY_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea/reasoning/verified-transition-history/observation/v2";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VerifiedTransitionHistoryKey {
    pub candidate_id: String,
    pub task: TaskType,
    pub primitive_encoding_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
struct VerifiedHistoryBucket {
    observation_count: u64,
    total_normalized_transition: f64,
    head_commitment: String,
}

#[derive(Debug, Clone, Default)]
pub struct VerifiedTransitionHistory {
    buckets: HashMap<VerifiedTransitionHistoryKey, VerifiedHistoryBucket>,
    admitted_chains: u64,
    admitted_executions: u64,
    rejected_chains: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransitionMeanEnvelope {
    pub lower: f64,
    pub upper: f64,
}

impl TransitionMeanEnvelope {
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerifiedTransitionSummary {
    pub candidate_id: String,
    pub task: TaskType,
    pub primitive_encoding_digest: String,
    pub observation_count: u64,
    pub total_normalized_transition: f64,
    pub historical_normalized_mean: f64,
    /// Exact range of possible running means after one additional bounded observation in `[0, 1]`.
    /// This is a sensitivity envelope, not a statistical confidence interval.
    pub one_step_update_envelope: TransitionMeanEnvelope,
    pub history_commitment: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedChainAdmission {
    pub chain_commitment: String,
    pub task: TaskType,
    pub execution_count: usize,
    pub distinct_history_keys: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveTransitionHistoryProfile {
    pub candidate_id: String,
    pub primitive_encoding_digest: String,
    pub summary: Option<VerifiedTransitionSummary>,
    /// Same candidate + task history under a different primitive encoding.
    pub stale_identity_history_entries: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActiveTransitionHistoryReport {
    pub adapter_version: String,
    pub task: TaskType,
    pub profiles: Vec<ActiveTransitionHistoryProfile>,
    pub measured_candidates: usize,
    pub unmeasured_candidates: usize,
    pub stale_identity_history_entries: usize,
}

impl VerifiedTransitionHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn admitted_chains(&self) -> u64 {
        self.admitted_chains
    }

    pub fn admitted_executions(&self) -> u64 {
        self.admitted_executions
    }

    pub fn rejected_chains(&self) -> u64 {
        self.rejected_chains
    }

    pub fn observe_chain(
        &mut self,
        chain: &ReasoningChain,
        task: TaskType,
    ) -> Result<VerifiedChainAdmission, VerifiedTransitionHistoryError> {
        let verified = match verify_chain(chain, task) {
            Ok(verified) => verified,
            Err(err) => {
                self.rejected_chains = self.rejected_chains.saturating_add(1);
                return Err(err);
            }
        };

        let mut increments: HashMap<&VerifiedTransitionHistoryKey, u64> = HashMap::new();
        for observation in &verified.observations {
            let increment = increments.entry(&observation.key).or_default();
            *increment = increment
                .checked_add(1)
                .ok_or(VerifiedTransitionHistoryError::ObservationCountOverflow)?;
        }
        for (key, increment) in &increments {
            let current = self
                .buckets
                .get(*key)
                .map(|bucket| bucket.observation_count)
                .unwrap_or(0);
            current
                .checked_add(*increment)
                .ok_or(VerifiedTransitionHistoryError::ObservationCountOverflow)?;
        }
        let execution_increment = u64::try_from(verified.observations.len())
            .map_err(|_| VerifiedTransitionHistoryError::ObservationCountOverflow)?;
        let next_admitted_chains = self
            .admitted_chains
            .checked_add(1)
            .ok_or(VerifiedTransitionHistoryError::ObservationCountOverflow)?;
        let next_admitted_executions = self
            .admitted_executions
            .checked_add(execution_increment)
            .ok_or(VerifiedTransitionHistoryError::ObservationCountOverflow)?;

        let mut touched = HashSet::new();
        for observation in &verified.observations {
            let bucket = self
                .buckets
                .entry(observation.key.clone())
                .or_insert_with(|| VerifiedHistoryBucket {
                    observation_count: 0,
                    total_normalized_transition: 0.0,
                    head_commitment: history_genesis_commitment(&observation.key),
                });
            bucket.observation_count += 1;
            bucket.total_normalized_transition += observation.normalized_transition;
            bucket.head_commitment = advance_history_commitment(
                &bucket.head_commitment,
                &verified.chain_commitment,
                observation.execution_index,
                observation.normalized_transition,
            );
            touched.insert(observation.key.clone());
        }

        self.admitted_chains = next_admitted_chains;
        self.admitted_executions = next_admitted_executions;

        Ok(VerifiedChainAdmission {
            chain_commitment: verified.chain_commitment,
            task,
            execution_count: verified.observations.len(),
            distinct_history_keys: touched.len(),
        })
    }

    pub fn summary(
        &self,
        candidate_id: &str,
        task: TaskType,
        primitive_encoding_digest: &str,
    ) -> Option<VerifiedTransitionSummary> {
        let key = VerifiedTransitionHistoryKey {
            candidate_id: candidate_id.to_owned(),
            task,
            primitive_encoding_digest: primitive_encoding_digest.to_owned(),
        };
        let bucket = self.buckets.get(&key)?;
        Some(summary_from_bucket(&key, bucket))
    }

    fn stale_identity_entries(
        &self,
        candidate_id: &str,
        task: TaskType,
        active_encoding_digest: &str,
    ) -> usize {
        self.buckets
            .keys()
            .filter(|key| {
                key.candidate_id == candidate_id
                    && key.task == task
                    && key.primitive_encoding_digest != active_encoding_digest
            })
            .count()
    }
}

/// Inspect verified transition history for the exact active primitive identities.
///
/// This function returns a separate diagnostic report. It does not modify the active report's
/// `CandidateObjectiveEvidence` and cannot satisfy a V3 objective request.
pub fn inspect_verified_transition_history(
    active_report: &ActivePrimitiveEvidenceReport,
    history: &VerifiedTransitionHistory,
    task: TaskType,
) -> Result<ActiveTransitionHistoryReport, VerifiedTransitionHistoryError> {
    if active_report.profiles.is_empty() {
        return Err(VerifiedTransitionHistoryError::EmptyActiveReport);
    }

    let mut profiles = Vec::with_capacity(active_report.profiles.len());
    let mut measured_candidates = 0usize;
    let mut unmeasured_candidates = 0usize;
    let mut stale_identity_history_entries = 0usize;

    for profile in &active_report.profiles {
        let digest = &profile.observed_primitive.encoding_digest;
        let stale = history.stale_identity_entries(&profile.candidate_id, task, digest);
        stale_identity_history_entries = stale_identity_history_entries.saturating_add(stale);
        let summary = history.summary(&profile.candidate_id, task, digest);
        if summary.is_some() {
            measured_candidates += 1;
        } else {
            unmeasured_candidates += 1;
        }
        profiles.push(ActiveTransitionHistoryProfile {
            candidate_id: profile.candidate_id.clone(),
            primitive_encoding_digest: digest.clone(),
            summary,
            stale_identity_history_entries: stale,
        });
    }

    Ok(ActiveTransitionHistoryReport {
        adapter_version: VERIFIED_TRANSITION_HISTORY_VERSION.into(),
        task,
        profiles,
        measured_candidates,
        unmeasured_candidates,
        stale_identity_history_entries,
    })
}

pub fn inspect_active_primitive_transition_history(
    active: &[ActivePrimitive],
    candidate_ids: &[String],
    history: &VerifiedTransitionHistory,
    task: TaskType,
) -> Result<
    (ActivePrimitiveEvidenceReport, ActiveTransitionHistoryReport),
    VerifiedTransitionHistoryError,
> {
    let active_report = adapt_active_primitive_evidence(active, candidate_ids)?;
    let history_report = inspect_verified_transition_history(&active_report, history, task)?;
    Ok((active_report, history_report))
}

#[derive(Debug, Clone)]
struct StagedObservation {
    key: VerifiedTransitionHistoryKey,
    execution_index: usize,
    normalized_transition: f64,
}

#[derive(Debug, Clone)]
struct VerifiedChain {
    chain_commitment: String,
    observations: Vec<StagedObservation>,
}

fn verify_chain(
    chain: &ReasoningChain,
    task: TaskType,
) -> Result<VerifiedChain, VerifiedTransitionHistoryError> {
    if chain.executions.is_empty() {
        return Err(VerifiedTransitionHistoryError::EmptyChain);
    }
    if !chain.total_phi.is_finite() {
        return Err(VerifiedTransitionHistoryError::InvalidNumericValue {
            field: "chain.total_phi",
            value: chain.total_phi,
        });
    }

    let mut expected_input = chain.question;
    let mut expected_total = 0.0_f64;
    let mut observations = Vec::with_capacity(chain.executions.len());

    for (index, execution) in chain.executions.iter().enumerate() {
        if execution.primitive.name.trim().is_empty() {
            return Err(VerifiedTransitionHistoryError::EmptyCandidateId { index });
        }
        if execution.input != expected_input {
            return Err(VerifiedTransitionHistoryError::ChainDiscontinuity { index });
        }
        if !execution.phi_contribution.is_finite() {
            return Err(VerifiedTransitionHistoryError::InvalidNumericValue {
                field: "execution.phi_contribution",
                value: execution.phi_contribution,
            });
        }

        let mut replay = ReasoningChain::new(execution.input);
        replay
            .execute_primitive(&execution.primitive, execution.transformation)
            .map_err(|err| VerifiedTransitionHistoryError::ReplayFailed {
                index,
                error: err.to_string(),
            })?;
        let canonical = replay
            .executions
            .first()
            .ok_or(VerifiedTransitionHistoryError::ReplayProducedNoExecution { index })?;

        if canonical.output != execution.output {
            return Err(VerifiedTransitionHistoryError::OutputMismatch { index });
        }
        if !close_enough(canonical.phi_contribution, execution.phi_contribution) {
            return Err(VerifiedTransitionHistoryError::ContributionMismatch {
                index,
                expected: canonical.phi_contribution,
                found: execution.phi_contribution,
            });
        }
        if canonical.phi_contribution < -NUMERIC_TOLERANCE
            || canonical.phi_contribution > LOCAL_TRANSITION_CONTRIBUTION_SCALE + NUMERIC_TOLERANCE
        {
            return Err(VerifiedTransitionHistoryError::ContributionOutsideFrozenScale {
                index,
                value: canonical.phi_contribution,
            });
        }

        let normalized_transition =
            (canonical.phi_contribution / LOCAL_TRANSITION_CONTRIBUTION_SCALE).clamp(0.0, 1.0);
        observations.push(StagedObservation {
            key: VerifiedTransitionHistoryKey {
                candidate_id: execution.primitive.name.clone(),
                task,
                primitive_encoding_digest: primitive_encoding_digest(&execution.primitive.encoding.0),
            },
            execution_index: index,
            normalized_transition,
        });
        expected_input = canonical.output;
        expected_total += canonical.phi_contribution;
    }

    if chain.current_state != expected_input {
        return Err(VerifiedTransitionHistoryError::FinalStateMismatch);
    }
    if !close_enough(expected_total, chain.total_phi) {
        return Err(VerifiedTransitionHistoryError::TotalContributionMismatch {
            expected: expected_total,
            found: chain.total_phi,
        });
    }

    Ok(VerifiedChain {
        chain_commitment: chain_commitment(chain, task),
        observations,
    })
}

fn summary_from_bucket(
    key: &VerifiedTransitionHistoryKey,
    bucket: &VerifiedHistoryBucket,
) -> VerifiedTransitionSummary {
    debug_assert!(bucket.observation_count > 0);
    let n = bucket.observation_count as f64;
    let historical_normalized_mean = bucket.total_normalized_transition / n;
    let denominator = n + 1.0;
    let one_step_update_envelope = TransitionMeanEnvelope {
        lower: bucket.total_normalized_transition / denominator,
        upper: (bucket.total_normalized_transition + 1.0) / denominator,
    };
    VerifiedTransitionSummary {
        candidate_id: key.candidate_id.clone(),
        task: key.task,
        primitive_encoding_digest: key.primitive_encoding_digest.clone(),
        observation_count: bucket.observation_count,
        total_normalized_transition: bucket.total_normalized_transition,
        historical_normalized_mean,
        one_step_update_envelope,
        history_commitment: bucket.head_commitment.clone(),
    }
}

fn close_enough(expected: f64, found: f64) -> bool {
    expected.is_finite() && found.is_finite() && (expected - found).abs() <= NUMERIC_TOLERANCE
}

fn primitive_encoding_digest(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn chain_commitment(chain: &ReasoningChain, task: TaskType) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, CHAIN_COMMITMENT_DOMAIN);
    hash_str(&mut hasher, task_slug(task));
    hash_bytes(&mut hasher, &chain.question.0);
    hash_u64(&mut hasher, chain.executions.len() as u64);
    for execution in &chain.executions {
        hash_str(&mut hasher, &execution.primitive.name);
        hash_bytes(&mut hasher, &execution.primitive.encoding.0);
        hash_u64(&mut hasher, transformation_tag(execution.transformation));
        hash_bytes(&mut hasher, &execution.input.0);
        hash_bytes(&mut hasher, &execution.output.0);
        hash_u64(&mut hasher, execution.phi_contribution.to_bits());
    }
    hash_bytes(&mut hasher, &chain.current_state.0);
    hash_u64(&mut hasher, chain.total_phi.to_bits());
    hasher.finalize().to_hex().to_string()
}

fn history_genesis_commitment(key: &VerifiedTransitionHistoryKey) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, HISTORY_GENESIS_DOMAIN);
    hash_str(&mut hasher, &key.candidate_id);
    hash_str(&mut hasher, task_slug(key.task));
    hash_str(&mut hasher, &key.primitive_encoding_digest);
    hasher.finalize().to_hex().to_string()
}

fn advance_history_commitment(
    previous: &str,
    chain_commitment: &str,
    execution_index: usize,
    normalized_transition: f64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, HISTORY_OBSERVATION_DOMAIN);
    hash_str(&mut hasher, previous);
    hash_str(&mut hasher, chain_commitment);
    hash_u64(&mut hasher, execution_index as u64);
    hash_u64(&mut hasher, normalized_transition.to_bits());
    hasher.finalize().to_hex().to_string()
}

const fn transformation_tag(transformation: TransformationType) -> u64 {
    match transformation {
        TransformationType::Bind => 0,
        TransformationType::Bundle => 1,
        TransformationType::Permute => 2,
        TransformationType::Resonate => 3,
        TransformationType::Abstract => 4,
        TransformationType::Ground => 5,
    }
}

const fn task_slug(task: TaskType) -> &'static str {
    match task {
        TaskType::Mathematical => "mathematical",
        TaskType::Physical => "physical",
        TaskType::Geometric => "geometric",
        TaskType::Strategic => "strategic",
        TaskType::MetaCognitive => "metacognitive",
        TaskType::Logical => "logical",
        TaskType::Causal => "causal",
        TaskType::Spatial => "spatial",
        TaskType::Social => "social",
        TaskType::General => "general",
        TaskType::Temporal => "temporal",
        TaskType::Generic => "generic",
        TaskType::Code => "code",
    }
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hash_bytes(hasher, value.as_bytes());
}

fn hash_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

#[derive(Debug)]
pub enum VerifiedTransitionHistoryError {
    EmptyChain,
    EmptyCandidateId { index: usize },
    ChainDiscontinuity { index: usize },
    ReplayFailed { index: usize, error: String },
    ReplayProducedNoExecution { index: usize },
    OutputMismatch { index: usize },
    ContributionMismatch {
        index: usize,
        expected: f64,
        found: f64,
    },
    ContributionOutsideFrozenScale { index: usize, value: f64 },
    FinalStateMismatch,
    TotalContributionMismatch { expected: f64, found: f64 },
    InvalidNumericValue { field: &'static str, value: f64 },
    ObservationCountOverflow,
    EmptyActiveReport,
    ActivePrimitive(ActivePrimitiveEvidenceError),
}

impl fmt::Display for VerifiedTransitionHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "verified transition history requires a non-empty chain"),
            Self::EmptyCandidateId { index } => {
                write!(f, "execution {index} has an empty primitive identity")
            }
            Self::ChainDiscontinuity { index } => write!(
                f,
                "execution {index} input does not equal the authoritative preceding state"
            ),
            Self::ReplayFailed { index, error } => {
                write!(f, "execution {index} production replay failed: {error}")
            }
            Self::ReplayProducedNoExecution { index } => {
                write!(f, "execution {index} production replay emitted no execution")
            }
            Self::OutputMismatch { index } => {
                write!(f, "execution {index} output differs from production replay")
            }
            Self::ContributionMismatch {
                index,
                expected,
                found,
            } => write!(
                f,
                "execution {index} local transition contribution {found} differs from production replay {expected}"
            ),
            Self::ContributionOutsideFrozenScale { index, value } => write!(
                f,
                "execution {index} replayed contribution {value} lies outside frozen [0, 0.1] scale"
            ),
            Self::FinalStateMismatch => write!(
                f,
                "reasoning chain current_state differs from the final replayed execution output"
            ),
            Self::TotalContributionMismatch { expected, found } => write!(
                f,
                "reasoning chain accumulated contribution {found} differs from replayed total {expected}"
            ),
            Self::InvalidNumericValue { field, value } => {
                write!(f, "`{field}` must be finite, got {value}")
            }
            Self::ObservationCountOverflow => {
                write!(f, "verified transition history observation count overflow")
            }
            Self::EmptyActiveReport => {
                write!(f, "verified transition history requires active candidates")
            }
            Self::ActivePrimitive(err) => write!(f, "active primitive evidence error: {err}"),
        }
    }
}

impl std::error::Error for VerifiedTransitionHistoryError {}

impl From<ActivePrimitiveEvidenceError> for VerifiedTransitionHistoryError {
    fn from(value: ActivePrimitiveEvidenceError) -> Self {
        Self::ActivePrimitive(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::{ActivationReason, ActivePrimitive};
    use symthaea_core::hdc::primitive_system::PrimitiveSystem;
    use symthaea_core::hdc::BinaryHV;

    fn primitive(name: &str) -> symthaea_core::hdc::primitive_system::Primitive {
        PrimitiveSystem::global()
            .get(name)
            .unwrap_or_else(|| panic!("fixture primitive `{name}` must exist"))
            .clone()
    }

    fn canonical_chain(name: &str, steps: usize) -> ReasoningChain {
        let mut chain = ReasoningChain::new(BinaryHV::random(41));
        let primitive = primitive(name);
        for _ in 0..steps {
            chain
                .execute_primitive(&primitive, TransformationType::Bundle)
                .unwrap();
        }
        chain
    }

    fn active(name: &str, activation: f64) -> ActivePrimitive {
        ActivePrimitive {
            primitive: primitive(name),
            activation,
            activation_reason: ActivationReason::BottomUp {
                input_similarity: activation,
            },
            duration: 2,
        }
    }

    #[test]
    fn production_replay_admits_canonical_chain_transactionally() {
        let mut history = VerifiedTransitionHistory::new();
        let chain = canonical_chain("NSM_KNOW", 2);
        let admission = history.observe_chain(&chain, TaskType::Generic).unwrap();
        assert_eq!(admission.execution_count, 2);
        assert_eq!(history.admitted_chains(), 1);
        assert_eq!(history.admitted_executions(), 2);
        assert_eq!(history.rejected_chains(), 0);
    }

    #[test]
    fn tampered_chain_is_rejected_without_partial_history() {
        let mut history = VerifiedTransitionHistory::new();
        let mut chain = canonical_chain("NSM_KNOW", 2);
        chain.executions[1].output = BinaryHV::random(99_001);
        assert!(history.observe_chain(&chain, TaskType::Generic).is_err());
        assert_eq!(history.admitted_chains(), 0);
        assert_eq!(history.admitted_executions(), 0);
        assert_eq!(history.rejected_chains(), 1);
    }

    #[test]
    fn summary_has_exact_one_next_observation_envelope() {
        let mut history = VerifiedTransitionHistory::new();
        let chain = canonical_chain("NSM_KNOW", 3);
        history.observe_chain(&chain, TaskType::Generic).unwrap();
        let digest = primitive_encoding_digest(&primitive("NSM_KNOW").encoding.0);
        let summary = history
            .summary("NSM_KNOW", TaskType::Generic, &digest)
            .unwrap();
        assert_eq!(summary.observation_count, 3);
        let expected_width = 1.0 / 4.0;
        assert!((summary.one_step_update_envelope.width() - expected_width).abs() < 1.0e-12);
        assert!((0.0..=1.0).contains(&summary.historical_normalized_mean));
    }

    #[test]
    fn exact_encoding_identity_prevents_history_inheritance() {
        let mut history = VerifiedTransitionHistory::new();
        let chain = canonical_chain("NSM_KNOW", 1);
        history.observe_chain(&chain, TaskType::Generic).unwrap();
        let digest = primitive_encoding_digest(&primitive("NSM_KNOW").encoding.0);
        assert!(history
            .summary("NSM_KNOW", TaskType::Generic, &digest)
            .is_some());
        assert!(history
            .summary("NSM_KNOW", TaskType::Generic, "different-encoding")
            .is_none());
    }

    #[test]
    fn active_history_inspection_does_not_promote_objective_evidence() {
        let mut history = VerifiedTransitionHistory::new();
        history
            .observe_chain(&canonical_chain("NSM_KNOW", 2), TaskType::Generic)
            .unwrap();
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.6)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let (active_report, report) = inspect_active_primitive_transition_history(
            &actives,
            &ids,
            &history,
            TaskType::Generic,
        )
        .unwrap();
        assert_eq!(report.measured_candidates, 1);
        assert_eq!(report.unmeasured_candidates, 1);
        assert_eq!(report.profiles.len(), 2);
        assert!(active_report
            .profiles
            .iter()
            .all(|profile| profile.objective_evidence.observed_axes() == 0));
    }
}

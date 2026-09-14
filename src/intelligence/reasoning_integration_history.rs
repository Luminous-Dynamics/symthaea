// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified reasoning-chain history for candidate-specific IntegrationProxy evidence.
//!
//! The legacy primitive selector intentionally mixes several learning signals, including
//! architecture feedback. That aggregate is useful for behavior, but it is not a qualified
//! measurement source. This module therefore keeps a separate measurement-only authority plane.
//!
//! A claimed `ReasoningChain` is never trusted by structure alone. Every execution is replayed
//! through `ReasoningChain::execute_primitive`, which is the production execution path. Only a
//! chain whose inputs, outputs, local contribution proxy, final state, and accumulated total all
//! agree with that replay can enter the verified history. Admission is transactional: a rejected
//! chain contributes nothing.
//!
//! The historical signal is deliberately called a local transition contribution proxy. It is not
//! IIT Phi, a consciousness measurement, reasoning correctness, or future expected utility.

use super::reasoning_active_primitive_evidence::{
    adapt_active_primitive_evidence, ActivePrimitiveEvidenceError, ActivePrimitiveEvidenceReport,
};
use super::reasoning_context_competition::{ContextCompetitionPolicy, ContextHypothesis};
use super::reasoning_evidence_seeking::{
    plan_with_evidence, EvidenceSeekingPlanReport, EvidenceSeekingPlannerError,
};
use super::reasoning_objective_evidence::{
    CandidateObjectiveEvidence, ObjectiveEvidence, ObjectiveEvidenceError, ScoreInterval,
};
use crate::consciousness::primitive_reasoning::{ReasoningChain, TaskType, TransformationType};
use crate::consciousness::ActivePrimitive;
use std::collections::{HashMap, HashSet};
use std::fmt;

pub const VERIFIED_REASONING_HISTORY_VERSION: &str = "rq-006x-verified-reasoning-history-v1";
/// The production `ReasoningChain` currently computes `(1 - |similarity|) * 0.1`.
/// This constant only normalizes that already-produced local transition proxy to `[0, 1]`.
pub const LOCAL_TRANSITION_CONTRIBUTION_SCALE: f64 = 0.1;
const NUMERIC_TOLERANCE: f64 = 1.0e-12;
const CHAIN_COMMITMENT_DOMAIN: &[u8] = b"symthaea/reasoning/verified-chain/v1";
const HISTORY_GENESIS_DOMAIN: &[u8] = b"symthaea/reasoning/verified-history/genesis/v1";
const HISTORY_OBSERVATION_DOMAIN: &[u8] = b"symthaea/reasoning/verified-history/observation/v1";

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VerifiedReasoningHistoryKey {
    pub candidate_id: String,
    pub task: TaskType,
    pub primitive_encoding_digest: String,
}

#[derive(Debug, Clone, PartialEq)]
struct VerifiedHistoryBucket {
    observation_count: u64,
    total_normalized_contribution: f64,
    head_commitment: String,
}

#[derive(Debug, Clone, Default)]
pub struct VerifiedReasoningHistory {
    buckets: HashMap<VerifiedReasoningHistoryKey, VerifiedHistoryBucket>,
    admitted_chains: u64,
    admitted_executions: u64,
    rejected_chains: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedChainAdmission {
    pub chain_commitment: String,
    pub task: TaskType,
    pub execution_count: usize,
    pub distinct_history_keys: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedIntegrationSummary {
    pub candidate_id: String,
    pub task: TaskType,
    pub primitive_encoding_digest: String,
    pub observation_count: u64,
    pub total_normalized_contribution: f64,
    pub historical_normalized_mean: f64,
    /// Exact range of possible running means after one additional bounded observation in `[0,1]`.
    /// This is not a statistical confidence interval.
    pub one_step_update_envelope: ScoreInterval,
    pub history_commitment: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AppliedVerifiedIntegrationEvidence {
    pub candidate_id: String,
    pub task: TaskType,
    pub primitive_encoding_digest: String,
    pub observation_count: u64,
    pub historical_normalized_mean: f64,
    pub one_step_update_envelope: ScoreInterval,
    pub history_commitment: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedIntegrationApplicationReport {
    pub adapter_version: String,
    pub task: TaskType,
    pub candidates: Vec<CandidateObjectiveEvidence>,
    pub applied: Vec<AppliedVerifiedIntegrationEvidence>,
    pub unmeasured_candidate_ids: Vec<String>,
    /// Histories that match candidate name + task but not the active primitive encoding.
    pub stale_identity_history_entries: usize,
}

impl VerifiedReasoningHistory {
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
    ) -> Result<VerifiedChainAdmission, VerifiedReasoningHistoryError> {
        let verified = match verify_chain(chain, task) {
            Ok(verified) => verified,
            Err(err) => {
                self.rejected_chains = self.rejected_chains.saturating_add(1);
                return Err(err);
            }
        };

        // Preflight every count before mutating any bucket. A chain is all-or-nothing evidence.
        let mut increments: HashMap<&VerifiedReasoningHistoryKey, u64> = HashMap::new();
        for observation in &verified.observations {
            *increments.entry(&observation.key).or_default() = increments
                .get(&observation.key)
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or(VerifiedReasoningHistoryError::ObservationCountOverflow)?;
        }
        for (key, increment) in &increments {
            let current = self
                .buckets
                .get(*key)
                .map(|bucket| bucket.observation_count)
                .unwrap_or(0);
            current
                .checked_add(*increment)
                .ok_or(VerifiedReasoningHistoryError::ObservationCountOverflow)?;
        }
        self.admitted_chains
            .checked_add(1)
            .ok_or(VerifiedReasoningHistoryError::ObservationCountOverflow)?;
        self.admitted_executions
            .checked_add(
                u64::try_from(verified.observations.len())
                    .map_err(|_| VerifiedReasoningHistoryError::ObservationCountOverflow)?,
            )
            .ok_or(VerifiedReasoningHistoryError::ObservationCountOverflow)?;

        let mut touched = HashSet::new();
        for observation in &verified.observations {
            let bucket = self
                .buckets
                .entry(observation.key.clone())
                .or_insert_with(|| VerifiedHistoryBucket {
                    observation_count: 0,
                    total_normalized_contribution: 0.0,
                    head_commitment: history_genesis_commitment(&observation.key),
                });
            bucket.observation_count += 1;
            bucket.total_normalized_contribution += observation.normalized_contribution;
            bucket.head_commitment = advance_history_commitment(
                &bucket.head_commitment,
                &verified.chain_commitment,
                observation.execution_index,
                observation.normalized_contribution,
            );
            touched.insert(observation.key.clone());
        }
        self.admitted_chains += 1;
        self.admitted_executions += u64::try_from(verified.observations.len())
            .map_err(|_| VerifiedReasoningHistoryError::ObservationCountOverflow)?;

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
    ) -> Option<VerifiedIntegrationSummary> {
        let key = VerifiedReasoningHistoryKey {
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

pub fn apply_verified_integration_history(
    active_report: &ActivePrimitiveEvidenceReport,
    history: &VerifiedReasoningHistory,
    task: TaskType,
) -> Result<VerifiedIntegrationApplicationReport, VerifiedReasoningHistoryError> {
    if active_report.profiles.is_empty() {
        return Err(VerifiedReasoningHistoryError::EmptyActiveReport);
    }

    let mut candidates = Vec::with_capacity(active_report.profiles.len());
    let mut applied = Vec::new();
    let mut unmeasured_candidate_ids = Vec::new();
    let mut stale_identity_history_entries = 0usize;

    for profile in &active_report.profiles {
        let mut candidate = profile.objective_evidence.clone();
        candidate.validate()?;
        let digest = &profile.observed_primitive.encoding_digest;
        stale_identity_history_entries = stale_identity_history_entries.saturating_add(
            history.stale_identity_entries(&profile.candidate_id, task, digest),
        );

        if let Some(summary) = history.summary(&profile.candidate_id, task, digest) {
            if candidate.integration_proxy.is_observed() {
                return Err(VerifiedReasoningHistoryError::AxisAlreadyObserved(
                    profile.candidate_id.clone(),
                ));
            }
            let harmonic_before = candidate.harmonic_alignment.clone();
            let epistemic_before = candidate.epistemic_grounding.clone();
            let evidence_refs = vec![
                format!("verified-history:{}", summary.history_commitment),
                format!("candidate-encoding:{}", summary.primitive_encoding_digest),
                format!("task:{}", task_slug(summary.task)),
                format!("observation-count:{}", summary.observation_count),
                format!(
                    "normalized-total-bits:{:016x}",
                    summary.total_normalized_contribution.to_bits()
                ),
            ];
            candidate.integration_proxy = ObjectiveEvidence::observed_interval(
                VERIFIED_REASONING_HISTORY_VERSION,
                evidence_refs,
                summary.one_step_update_envelope.lower,
                summary.one_step_update_envelope.upper,
            )?;
            candidate.validate()?;
            if candidate.harmonic_alignment != harmonic_before
                || candidate.epistemic_grounding != epistemic_before
            {
                return Err(VerifiedReasoningHistoryError::CrossAxisMutation(
                    profile.candidate_id.clone(),
                ));
            }
            applied.push(AppliedVerifiedIntegrationEvidence {
                candidate_id: summary.candidate_id,
                task: summary.task,
                primitive_encoding_digest: summary.primitive_encoding_digest,
                observation_count: summary.observation_count,
                historical_normalized_mean: summary.historical_normalized_mean,
                one_step_update_envelope: summary.one_step_update_envelope,
                history_commitment: summary.history_commitment,
            });
        } else {
            unmeasured_candidate_ids.push(profile.candidate_id.clone());
        }
        candidates.push(candidate);
    }

    Ok(VerifiedIntegrationApplicationReport {
        adapter_version: VERIFIED_REASONING_HISTORY_VERSION.into(),
        task,
        candidates,
        applied,
        unmeasured_candidate_ids,
        stale_identity_history_entries,
    })
}

pub fn plan_active_primitive_with_verified_integration(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    active: &[ActivePrimitive],
    candidate_ids: &[String],
    history: &VerifiedReasoningHistory,
    task: TaskType,
) -> Result<
    (
        ActivePrimitiveEvidenceReport,
        VerifiedIntegrationApplicationReport,
        EvidenceSeekingPlanReport,
    ),
    VerifiedReasoningHistoryError,
> {
    let active_report = adapt_active_primitive_evidence(active, candidate_ids)?;
    let application = apply_verified_integration_history(&active_report, history, task)?;
    let plan = plan_with_evidence(hypotheses, context_policy, &application.candidates)?;
    Ok((active_report, application, plan))
}

#[derive(Debug, Clone)]
struct StagedObservation {
    key: VerifiedReasoningHistoryKey,
    execution_index: usize,
    normalized_contribution: f64,
}

#[derive(Debug, Clone)]
struct VerifiedChain {
    chain_commitment: String,
    observations: Vec<StagedObservation>,
}

fn verify_chain(
    chain: &ReasoningChain,
    task: TaskType,
) -> Result<VerifiedChain, VerifiedReasoningHistoryError> {
    if chain.executions.is_empty() {
        return Err(VerifiedReasoningHistoryError::EmptyChain);
    }
    if !chain.total_phi.is_finite() {
        return Err(VerifiedReasoningHistoryError::InvalidNumericValue {
            field: "chain.total_phi",
            value: chain.total_phi,
        });
    }

    let mut expected_input = chain.question;
    let mut expected_total = 0.0_f64;
    let mut observations = Vec::with_capacity(chain.executions.len());

    for (index, execution) in chain.executions.iter().enumerate() {
        if execution.primitive.name.trim().is_empty() {
            return Err(VerifiedReasoningHistoryError::EmptyCandidateId { index });
        }
        if execution.input != expected_input {
            return Err(VerifiedReasoningHistoryError::ChainDiscontinuity { index });
        }
        if !execution.phi_contribution.is_finite() {
            return Err(VerifiedReasoningHistoryError::InvalidNumericValue {
                field: "execution.phi_contribution",
                value: execution.phi_contribution,
            });
        }

        // Replay through the production executor itself. This avoids a second implementation of
        // transformation or contribution semantics drifting from `execute_primitive`.
        let mut replay = ReasoningChain::new(execution.input);
        replay
            .execute_primitive(&execution.primitive, execution.transformation)
            .map_err(|err| VerifiedReasoningHistoryError::ReplayFailed {
                index,
                error: err.to_string(),
            })?;
        let canonical = replay
            .executions
            .first()
            .ok_or(VerifiedReasoningHistoryError::ReplayProducedNoExecution { index })?;

        if canonical.output != execution.output {
            return Err(VerifiedReasoningHistoryError::OutputMismatch { index });
        }
        if !close_enough(canonical.phi_contribution, execution.phi_contribution) {
            return Err(VerifiedReasoningHistoryError::ContributionMismatch {
                index,
                expected: canonical.phi_contribution,
                found: execution.phi_contribution,
            });
        }
        if canonical.phi_contribution < -NUMERIC_TOLERANCE
            || canonical.phi_contribution > LOCAL_TRANSITION_CONTRIBUTION_SCALE + NUMERIC_TOLERANCE
        {
            return Err(VerifiedReasoningHistoryError::ContributionOutsideFrozenScale {
                index,
                value: canonical.phi_contribution,
            });
        }
        let normalized =
            (canonical.phi_contribution / LOCAL_TRANSITION_CONTRIBUTION_SCALE).clamp(0.0, 1.0);
        let digest = primitive_encoding_digest(&execution.primitive.encoding.0);
        observations.push(StagedObservation {
            key: VerifiedReasoningHistoryKey {
                candidate_id: execution.primitive.name.clone(),
                task,
                primitive_encoding_digest: digest,
            },
            execution_index: index,
            normalized_contribution: normalized,
        });
        expected_input = canonical.output;
        expected_total += canonical.phi_contribution;
    }

    if chain.current_state != expected_input {
        return Err(VerifiedReasoningHistoryError::FinalStateMismatch);
    }
    if !close_enough(expected_total, chain.total_phi) {
        return Err(VerifiedReasoningHistoryError::TotalContributionMismatch {
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
    key: &VerifiedReasoningHistoryKey,
    bucket: &VerifiedHistoryBucket,
) -> VerifiedIntegrationSummary {
    debug_assert!(bucket.observation_count > 0);
    let n = bucket.observation_count as f64;
    let historical_normalized_mean = bucket.total_normalized_contribution / n;
    let denominator = n + 1.0;
    let one_step_update_envelope = ScoreInterval {
        lower: bucket.total_normalized_contribution / denominator,
        upper: (bucket.total_normalized_contribution + 1.0) / denominator,
    };
    VerifiedIntegrationSummary {
        candidate_id: key.candidate_id.clone(),
        task: key.task,
        primitive_encoding_digest: key.primitive_encoding_digest.clone(),
        observation_count: bucket.observation_count,
        total_normalized_contribution: bucket.total_normalized_contribution,
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

fn history_genesis_commitment(key: &VerifiedReasoningHistoryKey) -> String {
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
    normalized_contribution: f64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hash_bytes(&mut hasher, HISTORY_OBSERVATION_DOMAIN);
    hash_str(&mut hasher, previous);
    hash_str(&mut hasher, chain_commitment);
    hash_u64(&mut hasher, execution_index as u64);
    hash_u64(&mut hasher, normalized_contribution.to_bits());
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
pub enum VerifiedReasoningHistoryError {
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
    AxisAlreadyObserved(String),
    CrossAxisMutation(String),
    ActivePrimitive(ActivePrimitiveEvidenceError),
    Objective(ObjectiveEvidenceError),
    Planner(EvidenceSeekingPlannerError),
}

impl fmt::Display for VerifiedReasoningHistoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyChain => write!(f, "verified reasoning history requires a non-empty chain"),
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
                "execution {index} local contribution {found} differs from production replay {expected}"
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
            Self::ObservationCountOverflow => write!(f, "verified history observation count overflow"),
            Self::EmptyActiveReport => write!(f, "verified integration adapter requires active candidates"),
            Self::AxisAlreadyObserved(id) => write!(
                f,
                "candidate `{id}` already has observed integration evidence; explicit invalidation is required before replacement"
            ),
            Self::CrossAxisMutation(id) => write!(
                f,
                "verified integration adapter mutated a non-integration objective for candidate `{id}`"
            ),
            Self::ActivePrimitive(err) => write!(f, "active primitive evidence error: {err}"),
            Self::Objective(err) => write!(f, "objective evidence error: {err}"),
            Self::Planner(err) => write!(f, "V3 planning error: {err}"),
        }
    }
}

impl std::error::Error for VerifiedReasoningHistoryError {}

impl From<ActivePrimitiveEvidenceError> for VerifiedReasoningHistoryError {
    fn from(value: ActivePrimitiveEvidenceError) -> Self {
        Self::ActivePrimitive(value)
    }
}

impl From<ObjectiveEvidenceError> for VerifiedReasoningHistoryError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

impl From<EvidenceSeekingPlannerError> for VerifiedReasoningHistoryError {
    fn from(value: EvidenceSeekingPlannerError) -> Self {
        Self::Planner(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_evidence_seeking::{EvidenceRequestKind, EvidenceSeekingOutcome};
    use crate::consciousness::context_aware_evolution::ReasoningContext;
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
                .execute_primitive(&primitive, TransformationType::Bind)
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

    fn hypothesis(context: ReasoningContext) -> ContextHypothesis {
        ContextHypothesis {
            context,
            support: 0.9,
            source: "fixture-context".into(),
            evidence_refs: vec!["query".into()],
        }
    }

    fn fixture_history(
        candidate_id: &str,
        task: TaskType,
        encoding_digest: &str,
        count: u64,
        normalized_total: f64,
    ) -> VerifiedReasoningHistory {
        let key = VerifiedReasoningHistoryKey {
            candidate_id: candidate_id.into(),
            task,
            primitive_encoding_digest: encoding_digest.into(),
        };
        let mut history = VerifiedReasoningHistory::new();
        history.buckets.insert(
            key.clone(),
            VerifiedHistoryBucket {
                observation_count: count,
                total_normalized_contribution: normalized_total,
                head_commitment: history_genesis_commitment(&key),
            },
        );
        history
    }

    #[test]
    fn canonical_production_chain_is_admitted() {
        let chain = canonical_chain("NSM_KNOW", 2);
        let mut history = VerifiedReasoningHistory::new();
        let admission = history.observe_chain(&chain, TaskType::Logical).unwrap();
        assert_eq!(admission.execution_count, 2);
        assert_eq!(history.admitted_chains(), 1);
        assert_eq!(history.admitted_executions(), 2);
        assert_eq!(history.rejected_chains(), 0);
    }

    #[test]
    fn forged_contribution_is_rejected_transactionally() {
        let mut chain = canonical_chain("NSM_KNOW", 2);
        chain.executions[1].phi_contribution += 0.01;
        let mut history = VerifiedReasoningHistory::new();
        assert!(matches!(
            history.observe_chain(&chain, TaskType::Logical),
            Err(VerifiedReasoningHistoryError::ContributionMismatch { index: 1, .. })
        ));
        assert!(history.buckets.is_empty());
        assert_eq!(history.rejected_chains(), 1);
    }

    #[test]
    fn forged_output_is_rejected() {
        let mut chain = canonical_chain("NSM_KNOW", 1);
        chain.executions[0].output = BinaryHV::random(999);
        let mut history = VerifiedReasoningHistory::new();
        assert!(matches!(
            history.observe_chain(&chain, TaskType::Logical),
            Err(VerifiedReasoningHistoryError::OutputMismatch { index: 0 })
        ));
    }

    #[test]
    fn discontinuous_chain_is_rejected() {
        let mut chain = canonical_chain("NSM_KNOW", 2);
        chain.executions[1].input = BinaryHV::random(1001);
        let mut history = VerifiedReasoningHistory::new();
        assert!(matches!(
            history.observe_chain(&chain, TaskType::Logical),
            Err(VerifiedReasoningHistoryError::ChainDiscontinuity { index: 1 })
        ));
    }

    #[test]
    fn wrong_final_state_and_total_are_rejected() {
        let mut wrong_state = canonical_chain("NSM_KNOW", 1);
        wrong_state.current_state = BinaryHV::random(1234);
        assert!(matches!(
            VerifiedReasoningHistory::new().observe_chain(&wrong_state, TaskType::Logical),
            Err(VerifiedReasoningHistoryError::FinalStateMismatch)
        ));

        let mut wrong_total = canonical_chain("NSM_KNOW", 1);
        wrong_total.total_phi += 0.01;
        assert!(matches!(
            VerifiedReasoningHistory::new().observe_chain(&wrong_total, TaskType::Logical),
            Err(VerifiedReasoningHistoryError::TotalContributionMismatch { .. })
        ));
    }

    #[test]
    fn synthetic_causal_explainer_pattern_is_not_accepted_as_canonical_execution() {
        let primitive = primitive("NSM_KNOW");
        let input = BinaryHV::random(77);
        let mut chain = ReasoningChain::new(input);
        chain.executions.push(crate::consciousness::primitive_reasoning::PrimitiveExecution {
            primitive,
            input,
            output: input.bind(&BinaryHV::random(77)),
            transformation: TransformationType::Bind,
            phi_contribution: 0.07,
            timestamp: 1.0,
        });
        chain.current_state = chain.executions[0].output;
        chain.total_phi = 0.07;
        assert!(VerifiedReasoningHistory::new()
            .observe_chain(&chain, TaskType::Causal)
            .is_err());
    }

    #[test]
    fn one_step_update_envelope_has_exact_sensitivity_width() {
        let digest = primitive_encoding_digest(&primitive("NSM_KNOW").encoding.0);
        let history = fixture_history("NSM_KNOW", TaskType::Logical, &digest, 4, 2.0);
        let summary = history
            .summary("NSM_KNOW", TaskType::Logical, &digest)
            .unwrap();
        assert!((summary.historical_normalized_mean - 0.5).abs() < NUMERIC_TOLERANCE);
        assert!((summary.one_step_update_envelope.lower - 0.4).abs() < NUMERIC_TOLERANCE);
        assert!((summary.one_step_update_envelope.upper - 0.6).abs() < NUMERIC_TOLERANCE);
        assert!((summary.one_step_update_envelope.width() - 0.2).abs() < NUMERIC_TOLERANCE);
    }

    #[test]
    fn additional_verified_observation_narrows_update_envelope() {
        let chain = canonical_chain("NSM_KNOW", 1);
        let digest = primitive_encoding_digest(&primitive("NSM_KNOW").encoding.0);
        let mut history = VerifiedReasoningHistory::new();
        history.observe_chain(&chain, TaskType::Logical).unwrap();
        let first = history
            .summary("NSM_KNOW", TaskType::Logical, &digest)
            .unwrap()
            .one_step_update_envelope
            .width();
        history.observe_chain(&chain, TaskType::Logical).unwrap();
        let second = history
            .summary("NSM_KNOW", TaskType::Logical, &digest)
            .unwrap()
            .one_step_update_envelope
            .width();
        assert!((first - 0.5).abs() < NUMERIC_TOLERANCE);
        assert!((second - (1.0 / 3.0)).abs() < NUMERIC_TOLERANCE);
        assert!(second < first);
    }

    #[test]
    fn no_verified_history_leaves_integration_unknown() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let report = apply_verified_integration_history(
            &active_report,
            &VerifiedReasoningHistory::new(),
            TaskType::Logical,
        )
        .unwrap();
        assert_eq!(report.candidates[0].observed_axes(), 0);
        assert_eq!(report.unmeasured_candidate_ids, vec!["NSM_KNOW".to_string()]);
    }

    #[test]
    fn history_is_bound_to_exact_active_encoding_identity() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let history = fixture_history(
            "NSM_KNOW",
            TaskType::Logical,
            "old-encoding",
            4,
            2.0,
        );
        let report =
            apply_verified_integration_history(&active_report, &history, TaskType::Logical).unwrap();
        assert_eq!(report.candidates[0].observed_axes(), 0);
        assert_eq!(report.stale_identity_history_entries, 1);
    }

    #[test]
    fn adapter_changes_only_integration_axis() {
        let actives = [active("NSM_KNOW", 0.8)];
        let active_report =
            adapt_active_primitive_evidence(&actives, &["NSM_KNOW".into()]).unwrap();
        let digest = active_report.profiles[0]
            .observed_primitive
            .encoding_digest
            .clone();
        let history = fixture_history("NSM_KNOW", TaskType::Logical, &digest, 4, 2.0);
        let report =
            apply_verified_integration_history(&active_report, &history, TaskType::Logical).unwrap();
        assert!(report.candidates[0].integration_proxy.is_observed());
        assert!(!report.candidates[0].harmonic_alignment.is_observed());
        assert!(!report.candidates[0].epistemic_grounding.is_observed());
    }

    #[test]
    fn one_sided_integration_history_remains_underidentified() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let active_report = adapt_active_primitive_evidence(&actives, &ids).unwrap();
        let digest = active_report.profiles[0]
            .observed_primitive
            .encoding_digest
            .clone();
        let history = fixture_history("NSM_KNOW", TaskType::Logical, &digest, 20, 18.0);
        let (_, _, plan) = plan_active_primitive_with_verified_integration(
            &[hypothesis(ReasoningContext::CreativeExploration)],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            &history,
            TaskType::Logical,
        )
        .unwrap();
        assert!(matches!(plan.outcome, EvidenceSeekingOutcome::NeedEvidence { .. }));
    }

    #[test]
    fn overlapping_verified_histories_request_integration_refinement() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let active_report = adapt_active_primitive_evidence(&actives, &ids).unwrap();
        let first_digest = &active_report.profiles[0].observed_primitive.encoding_digest;
        let second_digest = &active_report.profiles[1].observed_primitive.encoding_digest;
        let mut history = fixture_history(
            "NSM_KNOW",
            TaskType::Logical,
            first_digest,
            4,
            2.0,
        );
        let second = fixture_history("NSM_DO", TaskType::Logical, second_digest, 4, 2.0);
        history.buckets.extend(second.buckets);
        let (_, _, plan) = plan_active_primitive_with_verified_integration(
            &[hypothesis(ReasoningContext::CreativeExploration)],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            &history,
            TaskType::Logical,
        )
        .unwrap();
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = plan.outcome else {
            panic!("expected refinement requests");
        };
        assert!(requests.iter().any(|request| matches!(
            &request.kind,
            EvidenceRequestKind::ObjectiveRefinement {
                objective: super::super::reasoning_objective_core::ObjectiveKind::IntegrationProxy,
                ..
            }
        )));
    }

    #[test]
    fn separated_verified_histories_can_identify_integration_heavy_winner() {
        let actives = [active("NSM_KNOW", 0.8), active("NSM_DO", 0.7)];
        let ids = vec!["NSM_KNOW".into(), "NSM_DO".into()];
        let active_report = adapt_active_primitive_evidence(&actives, &ids).unwrap();
        let first_digest = &active_report.profiles[0].observed_primitive.encoding_digest;
        let second_digest = &active_report.profiles[1].observed_primitive.encoding_digest;
        let mut history = fixture_history(
            "NSM_KNOW",
            TaskType::Logical,
            first_digest,
            99,
            90.0,
        );
        let second = fixture_history("NSM_DO", TaskType::Logical, second_digest, 99, 9.0);
        history.buckets.extend(second.buckets);
        let (_, _, plan) = plan_active_primitive_with_verified_integration(
            &[hypothesis(ReasoningContext::CreativeExploration)],
            ContextCompetitionPolicy::development_v1(),
            &actives,
            &ids,
            &history,
            TaskType::Logical,
        )
        .unwrap();
        assert!(matches!(
            plan.outcome,
            EvidenceSeekingOutcome::Selected { ref candidate_id, .. } if candidate_id == "NSM_KNOW"
        ));
    }
}

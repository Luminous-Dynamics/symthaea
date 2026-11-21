// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic synthetic oracle for RMI-Bench Track A.
//!
//! This module measures whether a policy's next symbolic action is sensitive
//! to a controlled change in the current partner observation. It deliberately
//! does NOT score musical quality, complementarity, human ownership,
//! consciousness, or real-world reciprocal influence. Those require separate
//! blinded musical/human evidence and a controlled partner-response track.
//!
//! The oracle uses paired observations from one frozen pre-intervention state.
//! A policy receives the same seed in both arms (common random numbers) and
//! does not receive the intervention label. Exact input/output records are
//! digested for deterministic replay.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt::Write as FmtWrite;

/// Wire/schema version for synthetic RMI trial and result records.
pub const RMI_BENCH_SCHEMA_VERSION: u32 = 1;

/// Which musical dimension is changed by the partner-side intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiTargetFeatureV1 {
    Motif,
    Rhythm,
    Tempo,
    Role,
}

/// Declared musical role in the synthetic Track A observation/action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiRoleV1 {
    Lead,
    Accompaniment,
    Bass,
    Counterline,
}

/// Actor identity inside the frozen synthetic interaction history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiHistoryActorV1 {
    Partner,
    Agent,
}

/// One prior interaction event included in the frozen pre-intervention state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiHistoryEventV1 {
    pub event_id: String,
    pub actor: RmiHistoryActorV1,
    pub features: RmiMusicalFeaturesV1,
}

/// Symbolic features observable in one partner event or emitted by one agent
/// action. Motif values are signed semitone intervals; rhythm is an eight-step
/// onset/rest pattern; tempo is milli-BPM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiMusicalFeaturesV1 {
    pub motif_intervals: [i16; 4],
    pub rhythm_onsets: [bool; 8],
    pub tempo_milli_bpm: u32,
    pub role: RmiRoleV1,
}

impl RmiMusicalFeaturesV1 {
    fn validation_issues(&self, field_prefix: &str) -> Vec<RmiValidationIssueV1> {
        let mut issues = Vec::new();
        if self.motif_intervals.iter().any(|value| !(-24..=24).contains(value)) {
            issues.push(RmiValidationIssueV1::MotifIntervalOutOfRange {
                field: field_prefix.to_string(),
            });
        }
        if !(30_000..=300_000).contains(&self.tempo_milli_bpm) {
            issues.push(RmiValidationIssueV1::TempoOutOfRange {
                field: field_prefix.to_string(),
                tempo_milli_bpm: self.tempo_milli_bpm,
            });
        }
        issues
    }

    pub fn changed_features(&self, other: &Self) -> Vec<RmiTargetFeatureV1> {
        let mut changed = Vec::new();
        if self.motif_intervals != other.motif_intervals {
            changed.push(RmiTargetFeatureV1::Motif);
        }
        if self.rhythm_onsets != other.rhythm_onsets {
            changed.push(RmiTargetFeatureV1::Rhythm);
        }
        if self.tempo_milli_bpm != other.tempo_milli_bpm {
            changed.push(RmiTargetFeatureV1::Tempo);
        }
        if self.role != other.role {
            changed.push(RmiTargetFeatureV1::Role);
        }
        changed
    }

    fn feature_changed(&self, other: &Self, target: RmiTargetFeatureV1) -> bool {
        match target {
            RmiTargetFeatureV1::Motif => self.motif_intervals != other.motif_intervals,
            RmiTargetFeatureV1::Rhythm => self.rhythm_onsets != other.rhythm_onsets,
            RmiTargetFeatureV1::Tempo => self.tempo_milli_bpm != other.tempo_milli_bpm,
            RmiTargetFeatureV1::Role => self.role != other.role,
        }
    }
}

/// One partner observation presented to a policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiPartnerObservationV1 {
    pub source_event_id: String,
    pub features: RmiMusicalFeaturesV1,
}

/// One action returned by a policy. This is a symbolic probe contract, not a
/// complete Score or performed MIDI representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiAgentActionV1 {
    pub action_id: String,
    pub features: RmiMusicalFeaturesV1,
}

/// One subsequent partner action in the controlled simulator track.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiPartnerResponseV1 {
    pub response_id: String,
    pub features: RmiMusicalFeaturesV1,
}

/// A paired counterfactual trial: both observations share the same frozen
/// pre-intervention state and differ only in the declared target feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiInterventionPairV1 {
    pub trial_id: String,
    pub seed: u64,
    /// Exact synthetic history used by both arms. The digest below is
    /// recomputed from this canonical serialized history by the validator.
    pub pre_intervention_history: Vec<RmiHistoryEventV1>,
    pub pre_intervention_state_sha256: String,
    pub target: RmiTargetFeatureV1,
    pub control_observation: RmiPartnerObservationV1,
    pub treatment_observation: RmiPartnerObservationV1,
}

/// Structural or intervention-integrity failure in a Track A pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RmiValidationIssueV1 {
    EmptyTrialId,
    InvalidPreStateDigest,
    EmptySourceEventId { arm: String },
    SourceEventIdMismatch,
    TargetFeatureNotChanged { target: RmiTargetFeatureV1 },
    UncontrolledFeatureChanged { feature: RmiTargetFeatureV1 },
    MotifIntervalOutOfRange { field: String },
    TempoOutOfRange { field: String, tempo_milli_bpm: u32 },
    EmptyPolicyId,
    EmptyPolicyVersion,
    EmptyActionId { arm: String },
}

/// Validate the paired intervention before a policy is invoked.
///
/// This rejects a changed target that is accompanied by uncontrolled changes
/// in other musical dimensions, preventing the evaluator from crediting a
/// policy for an ambiguous intervention.
pub fn validate_rmi_intervention_pair(
    pair: &RmiInterventionPairV1,
) -> Result<(), Vec<RmiValidationIssueV1>> {
    let mut issues = Vec::new();
    if pair.trial_id.trim().is_empty() {
        issues.push(RmiValidationIssueV1::EmptyTrialId);
    }
    let history_bytes = serde_json::to_vec(&pair.pre_intervention_history);
    match history_bytes {
        Ok(bytes) if is_lowercase_sha256(&pair.pre_intervention_state_sha256)
            && sha256_hex(&bytes) == pair.pre_intervention_state_sha256 => {}
        _ => issues.push(RmiValidationIssueV1::InvalidPreStateDigest),
    }
    for event in &pair.pre_intervention_history {
        if event.event_id.trim().is_empty() {
            issues.push(RmiValidationIssueV1::EmptySourceEventId {
                arm: "pre_intervention_history".to_string(),
            });
        }
        issues.extend(event.features.validation_issues("pre_intervention_history"));
    }

    for (arm, observation) in [
        ("control", &pair.control_observation),
        ("treatment", &pair.treatment_observation),
    ] {
        if observation.source_event_id.trim().is_empty() {
            issues.push(RmiValidationIssueV1::EmptySourceEventId {
                arm: arm.to_string(),
            });
        }
        issues.extend(observation.features.validation_issues(arm));
    }

    if pair.control_observation.source_event_id != pair.treatment_observation.source_event_id {
        issues.push(RmiValidationIssueV1::SourceEventIdMismatch);
    }

    let control = &pair.control_observation.features;
    let treatment = &pair.treatment_observation.features;
    if !control.feature_changed(treatment, pair.target) {
        issues.push(RmiValidationIssueV1::TargetFeatureNotChanged {
            target: pair.target,
        });
    }
    for feature in control.changed_features(treatment) {
        if feature != pair.target {
            issues.push(RmiValidationIssueV1::UncontrolledFeatureChanged { feature });
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Policy interface for controlled Track A evaluation.
///
/// The policy receives the current observation and the same seed in both
/// paired arms. It is not told which arm is control/treatment or which feature
/// was manipulated. Production Melothaea adapters can implement this trait
/// later, after their observation/action boundaries are pinned.
pub trait RmiPolicyV1: Sized {
    /// Mutable episode state only. Static weights, binaries and device handles
    /// are identified separately by policy_id/version and adapter provenance.
    type Snapshot: Serialize + DeserializeOwned + PartialEq;
    fn policy_id(&self) -> &str;
    fn policy_version(&self) -> &str;
    /// Every state field that can influence the next action must be included.
    /// Shared process-global state and external side effects are forbidden.
    fn snapshot_state(&self) -> Result<Self::Snapshot, String>;
    /// Restore an independent policy instance from an exact pre-trial snapshot.
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String>;
    /// The evaluator passes an independently restored instance to each arm.
    fn respond(
        &mut self,
        history: &[RmiHistoryEventV1],
        observation: &RmiPartnerObservationV1,
        seed: u64,
    ) -> RmiAgentActionV1;
}

/// Controlled partner policy for measuring agent-to-partner influence.
/// It receives the same frozen history and the agent's current action.
pub trait RmiPartnerPolicyV1: Sized {
    type Snapshot: Serialize + DeserializeOwned + PartialEq;
    fn policy_id(&self) -> &str;
    fn policy_version(&self) -> &str;
    /// Every state field that can influence the next response must be included.
    /// Shared process-global state and external side effects are forbidden.
    fn snapshot_state(&self) -> Result<Self::Snapshot, String>;
    /// Restore an independent partner instance from an exact pre-trial snapshot.
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String>;
    fn respond(
        &mut self,
        history: &[RmiHistoryEventV1],
        agent_action: &RmiAgentActionV1,
        seed: u64,
    ) -> RmiPartnerResponseV1;
}

/// Fixed partner baseline: ignores the agent's current action.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedPartnerResponsePolicyV1 {
    pub id: String,
    pub version: String,
    pub response: RmiPartnerResponseV1,
}

impl RmiPartnerPolicyV1 for FixedPartnerResponsePolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str { &self.id }
    fn policy_version(&self) -> &str { &self.version }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        _agent_action: &RmiAgentActionV1,
        _seed: u64,
    ) -> RmiPartnerResponseV1 {
        self.response.clone()
    }
}

/// Synthetic partner positive control. Its one target-specific transformation
/// exists only to qualify the directionality oracle, not to model human agency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticResponsivePartnerPolicyV1 {
    pub id: String,
    pub version: String,
    pub target: RmiTargetFeatureV1,
}

impl RmiPartnerPolicyV1 for SyntheticResponsivePartnerPolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str { &self.id }
    fn policy_version(&self) -> &str { &self.version }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        agent_action: &RmiAgentActionV1,
        _seed: u64,
    ) -> RmiPartnerResponseV1 {
        let mut features = agent_action.features.clone();
        transform_target_feature(&mut features, self.target);
        RmiPartnerResponseV1 {
            response_id: format!("{}:{}", self.id, agent_action.action_id),
            features,
        }
    }
}

/// Fixed response baseline: returns exactly the same action for every input.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixedResponsePolicyV1 {
    pub id: String,
    pub version: String,
    pub response: RmiAgentActionV1,
}

impl RmiPolicyV1 for FixedResponsePolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn policy_version(&self) -> &str {
        &self.version
    }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        _observation: &RmiPartnerObservationV1,
        _seed: u64,
    ) -> RmiAgentActionV1 {
        self.response.clone()
    }
}

/// One-way prompt baseline: represents a response selected from a frozen
/// initial prompt/state, with no access to the current partner event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OneWayPromptPolicyV1 {
    pub id: String,
    pub version: String,
    pub frozen_response: RmiAgentActionV1,
}

impl RmiPolicyV1 for OneWayPromptPolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn policy_version(&self) -> &str {
        &self.version
    }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        _observation: &RmiPartnerObservationV1,
        _seed: u64,
    ) -> RmiAgentActionV1 {
        self.frozen_response.clone()
    }
}

/// Yoked replay baseline: replays a response captured in a matched prior
/// session. A single paired probe uses that same replay in both arms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YokedReplayPolicyV1 {
    pub id: String,
    pub version: String,
    pub replayed_response: RmiAgentActionV1,
}

impl RmiPolicyV1 for YokedReplayPolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn policy_version(&self) -> &str {
        &self.version
    }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        _observation: &RmiPartnerObservationV1,
        _seed: u64,
    ) -> RmiAgentActionV1 {
        self.replayed_response.clone()
    }
}

/// Imitation-only baseline: copies the observed surface features directly.
///
/// This should be input-sensitive, but the benchmark does not infer musical
/// complementarity, negotiation or co-creative value from copying.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImitationOnlyPolicyV1 {
    pub id: String,
    pub version: String,
}

impl RmiPolicyV1 for ImitationOnlyPolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn policy_version(&self) -> &str {
        &self.version
    }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        observation: &RmiPartnerObservationV1,
        _seed: u64,
    ) -> RmiAgentActionV1 {
        RmiAgentActionV1 {
            action_id: format!("{}:{}", self.id, observation.source_event_id),
            features: observation.features.clone(),
        }
    }
}

/// Apply a known bijective transformation to exactly one configured feature.
/// Shared by synthetic agent and partner positive controls.
fn transform_target_feature(features: &mut RmiMusicalFeaturesV1, target: RmiTargetFeatureV1) {
    match target {
        RmiTargetFeatureV1::Motif => {
            for interval in &mut features.motif_intervals {
                *interval = interval.saturating_neg();
            }
        }
        RmiTargetFeatureV1::Rhythm => features.rhythm_onsets.rotate_left(1),
        RmiTargetFeatureV1::Tempo => {
            // Reflection is a bijection over the admitted [30, 300] BPM range.
            features.tempo_milli_bpm = 330_000 - features.tempo_milli_bpm;
        }
        RmiTargetFeatureV1::Role => {
            features.role = complementary_role(features.role);
        }
    }
}

/// Synthetic positive-control policy.
///
/// It applies a fixed, declared transformation to one configured feature of
/// current input. It proves only that the oracle detects known contingency.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticContingentPolicyV1 {
    pub id: String,
    pub version: String,
    /// One feature this known positive control responds to. This is fixed
    /// policy configuration, not an intervention label passed by the evaluator.
    pub target: RmiTargetFeatureV1,
}

impl RmiPolicyV1 for SyntheticContingentPolicyV1 {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
    fn policy_id(&self) -> &str {
        &self.id
    }

    fn policy_version(&self) -> &str {
        &self.version
    }

    fn respond(
        &mut self,
        _history: &[RmiHistoryEventV1],
        observation: &RmiPartnerObservationV1,
        _seed: u64,
    ) -> RmiAgentActionV1 {
        let mut features = observation.features.clone();
        transform_target_feature(&mut features, self.target);
        RmiAgentActionV1 {
            action_id: format!("{}:{}", self.id, observation.source_event_id),
            features,
        }
    }
}

fn complementary_role(role: RmiRoleV1) -> RmiRoleV1 {
    match role {
        RmiRoleV1::Lead => RmiRoleV1::Accompaniment,
        RmiRoleV1::Accompaniment => RmiRoleV1::Lead,
        RmiRoleV1::Bass => RmiRoleV1::Counterline,
        RmiRoleV1::Counterline => RmiRoleV1::Bass,
    }
}

/// Per-feature difference between two policy actions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiActionDeltaV1 {
    pub motif_changed: bool,
    pub rhythm_changed: bool,
    pub tempo_changed: bool,
    pub role_changed: bool,
}

impl RmiActionDeltaV1 {
    fn between(control: &RmiAgentActionV1, treatment: &RmiAgentActionV1) -> Self {
        Self {
            motif_changed: control.features.motif_intervals
                != treatment.features.motif_intervals,
            rhythm_changed: control.features.rhythm_onsets
                != treatment.features.rhythm_onsets,
            tempo_changed: control.features.tempo_milli_bpm
                != treatment.features.tempo_milli_bpm,
            role_changed: control.features.role != treatment.features.role,
        }
    }

    pub fn targeted_feature_changed(&self, target: RmiTargetFeatureV1) -> bool {
        match target {
            RmiTargetFeatureV1::Motif => self.motif_changed,
            RmiTargetFeatureV1::Rhythm => self.rhythm_changed,
            RmiTargetFeatureV1::Tempo => self.tempo_changed,
            RmiTargetFeatureV1::Role => self.role_changed,
        }
    }

    pub fn collateral_changed_count(&self, target: RmiTargetFeatureV1) -> usize {
        [
            (RmiTargetFeatureV1::Motif, self.motif_changed),
            (RmiTargetFeatureV1::Rhythm, self.rhythm_changed),
            (RmiTargetFeatureV1::Tempo, self.tempo_changed),
            (RmiTargetFeatureV1::Role, self.role_changed),
        ]
        .into_iter()
        .filter(|(feature, changed)| *changed && *feature != target)
        .count()
    }
}

/// Canonical evaluation payload. The digest is computed over this payload's
/// deterministic serde serialization; the envelope itself is not self-hashed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiEvaluationPayloadV1 {
    pub schema_version: u32,
    pub trial_id: String,
    pub seed: u64,
    pub pre_intervention_state_sha256: String,
    pub intervention_pair_sha256: String,
    pub policy_id: String,
    pub policy_version: String,
    pub policy_pre_state_sha256: String,
    pub control_policy_post_state_sha256: String,
    pub treatment_policy_post_state_sha256: String,
    /// True only if a second evaluation with reversed arm order reproduced
    /// both arm-specific actions and post-state digests.
    pub arm_order_swap_verified: bool,
    pub target: RmiTargetFeatureV1,
    pub control_action: RmiAgentActionV1,
    pub treatment_action: RmiAgentActionV1,
    pub action_delta: RmiActionDeltaV1,
    /// True means the action changed in the dimension that was manipulated.
    /// It is input sensitivity only; musical relevance is assessed separately.
    pub targeted_input_sensitivity: bool,
    pub collateral_changed_feature_count: usize,
}

/// Digest envelope for one paired policy evaluation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiEvaluationEnvelopeV1 {
    pub payload: RmiEvaluationPayloadV1,
    pub payload_sha256: String,
}

/// Bidirectional response record for one paired intervention.
/// This measures two directions separately; it is not a quality score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiReciprocalEvaluationPayloadV1 {
    pub schema_version: u32,
    pub trial_id: String,
    pub seed: u64,
    pub agent_evaluation_sha256: String,
    pub partner_policy_id: String,
    pub partner_policy_version: String,
    pub partner_policy_pre_state_sha256: String,
    pub control_partner_post_state_sha256: String,
    pub treatment_partner_post_state_sha256: String,
    /// True only if a reversed partner-arm order reproduced both responses
    /// and post-state digests.
    pub partner_arm_order_swap_verified: bool,
    pub target: RmiTargetFeatureV1,
    /// Partner-to-agent input sensitivity from the first half of this pair.
    pub agent_targeted_sensitivity: bool,
    pub control_partner_response: RmiPartnerResponseV1,
    pub treatment_partner_response: RmiPartnerResponseV1,
    pub partner_action_delta: RmiActionDeltaV1,
    /// Agent-to-partner input sensitivity only; musical relevance remains separate.
    pub targeted_partner_sensitivity: bool,
    pub partner_collateral_changed_feature_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RmiReciprocalEvaluationEnvelopeV1 {
    pub payload: RmiReciprocalEvaluationPayloadV1,
    pub payload_sha256: String,
}

/// Error returned before a trial can be scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RmiEvaluationErrorV1 {
    InvalidIntervention(Vec<RmiValidationIssueV1>),
    EmptyPolicyId,
    EmptyPolicyVersion,
    InvalidActionId { arm: String },
    InvalidActionFeatures { arm: String, issues: Vec<RmiValidationIssueV1> },
    NonDeterministicPolicyOutput { arm: String },
    PolicyStateSnapshotFailure(String),
    PolicyStateRestoreFailure { arm: String, reason: String },
    PolicyStateMismatch { arm: String },
    PolicyIdentityMismatch { arm: String },
    EmptyPartnerPolicyId,
    EmptyPartnerPolicyVersion,
    InvalidPartnerResponseId { arm: String },
    InvalidPartnerResponseFeatures { arm: String, issues: Vec<RmiValidationIssueV1> },
    NonDeterministicPartnerOutput { arm: String },
    PartnerStateSnapshotFailure(String),
    PartnerStateRestoreFailure { arm: String, reason: String },
    PartnerStateMismatch { arm: String },
    PartnerPolicyIdentityMismatch { arm: String },
    SerializationFailure(String),
}

/// Run both arms with identical policy/seed and produce a digest-bound record.
///
/// This function does not claim musical relevance. A downstream blinded
/// annotation or task-specific structural oracle must make that determination.
pub fn evaluate_rmi_pair<P: RmiPolicyV1>(
    policy: &P,
    pair: &RmiInterventionPairV1,
) -> Result<RmiEvaluationEnvelopeV1, RmiEvaluationErrorV1> {
    validate_rmi_intervention_pair(pair).map_err(RmiEvaluationErrorV1::InvalidIntervention)?;
    if policy.policy_id().trim().is_empty() {
        return Err(RmiEvaluationErrorV1::EmptyPolicyId);
    }
    if policy.policy_version().trim().is_empty() {
        return Err(RmiEvaluationErrorV1::EmptyPolicyVersion);
    }

    // Snapshot once, then restore independent instances for both counterfactual
    // arms and for deterministic replay. Never run control then treatment on
    // the same mutable policy object.
    let policy_snapshot = policy
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let policy_pre_state_sha256 = snapshot_sha256_v1(&policy_snapshot)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;

    let mut control_policy = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "control".into(),
            reason,
        })?;
    let mut control_replay = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "control_replay".into(),
            reason,
        })?;
    let mut treatment_policy = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "treatment".into(),
            reason,
        })?;
    let mut treatment_replay = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "treatment_replay".into(),
            reason,
        })?;

    for (arm, restored) in [
        ("control", &control_policy),
        ("control_replay", &control_replay),
        ("treatment", &treatment_policy),
        ("treatment_replay", &treatment_replay),
    ] {
        if restored.policy_id() != policy.policy_id()
            || restored.policy_version() != policy.policy_version()
        {
            return Err(RmiEvaluationErrorV1::PolicyIdentityMismatch {
                arm: arm.to_owned(),
            });
        }
        let restored_snapshot = restored
            .snapshot_state()
            .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
        if restored_snapshot != policy_snapshot {
            return Err(RmiEvaluationErrorV1::PolicyStateMismatch {
                arm: arm.to_owned(),
            });
        }
    }

    let control_action = control_policy.respond(
        &pair.pre_intervention_history,
        &pair.control_observation,
        pair.seed,
    );
    let control_replay_action = control_replay.respond(
        &pair.pre_intervention_history,
        &pair.control_observation,
        pair.seed,
    );
    let control_policy_post_state = control_policy
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let control_policy_post_state_sha256 = snapshot_sha256_v1(&control_policy_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let control_replay_post_state = control_replay
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let control_replay_post_state_sha256 = snapshot_sha256_v1(&control_replay_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if control_action != control_replay_action
        || control_policy_post_state_sha256 != control_replay_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPolicyOutput {
            arm: "control".to_string(),
        });
    }

    let treatment_action = treatment_policy.respond(
        &pair.pre_intervention_history,
        &pair.treatment_observation,
        pair.seed,
    );
    let treatment_replay_action = treatment_replay.respond(
        &pair.pre_intervention_history,
        &pair.treatment_observation,
        pair.seed,
    );
    let treatment_policy_post_state = treatment_policy
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let treatment_policy_post_state_sha256 = snapshot_sha256_v1(&treatment_policy_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let treatment_replay_post_state = treatment_replay
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let treatment_replay_post_state_sha256 = snapshot_sha256_v1(&treatment_replay_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if treatment_action != treatment_replay_action
        || treatment_policy_post_state_sha256 != treatment_replay_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPolicyOutput {
            arm: "treatment".to_string(),
        });
    }

    // Repeat the pair in reverse order. This catches order-coupled shared
    // process state that is not captured by the policy's serialized snapshot.
    let mut treatment_order_probe = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "order_swap_treatment".into(),
            reason,
        })?;
    let mut control_order_probe = P::restore_from_snapshot(&policy_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PolicyStateRestoreFailure {
            arm: "order_swap_control".into(),
            reason,
        })?;
    let reverse_treatment_action = treatment_order_probe.respond(
        &pair.pre_intervention_history,
        &pair.treatment_observation,
        pair.seed,
    );
    let treatment_order_probe_post_state = treatment_order_probe
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let reverse_treatment_post_state_sha256 = snapshot_sha256_v1(&treatment_order_probe_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let reverse_control_action = control_order_probe.respond(
        &pair.pre_intervention_history,
        &pair.control_observation,
        pair.seed,
    );
    let control_order_probe_post_state = control_order_probe
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PolicyStateSnapshotFailure)?;
    let reverse_control_post_state_sha256 = snapshot_sha256_v1(&control_order_probe_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if reverse_treatment_action != treatment_action
        || reverse_treatment_post_state_sha256 != treatment_policy_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPolicyOutput {
            arm: "order_swap_treatment".to_string(),
        });
    }
    if reverse_control_action != control_action
        || reverse_control_post_state_sha256 != control_policy_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPolicyOutput {
            arm: "order_swap_control".to_string(),
        });
    }

    for (arm, action) in [
        ("control", &control_action),
        ("treatment", &treatment_action),
    ] {
        if action.action_id.trim().is_empty() {
            return Err(RmiEvaluationErrorV1::InvalidActionId {
                arm: arm.to_string(),
            });
        }
        let issues = action.features.validation_issues(arm);
        if !issues.is_empty() {
            return Err(RmiEvaluationErrorV1::InvalidActionFeatures {
                arm: arm.to_string(),
                issues,
            });
        }
    }

    let action_delta = RmiActionDeltaV1::between(&control_action, &treatment_action);
    let targeted_input_sensitivity = action_delta.targeted_feature_changed(pair.target);
    let collateral_changed_feature_count =
        action_delta.collateral_changed_count(pair.target);

    let intervention_bytes = serde_json::to_vec(pair)
        .map_err(|error| RmiEvaluationErrorV1::SerializationFailure(error.to_string()))?;
    let intervention_pair_sha256 = sha256_hex(&intervention_bytes);

    let payload = RmiEvaluationPayloadV1 {
        schema_version: RMI_BENCH_SCHEMA_VERSION,
        trial_id: pair.trial_id.clone(),
        seed: pair.seed,
        pre_intervention_state_sha256: pair.pre_intervention_state_sha256.clone(),
        intervention_pair_sha256,
        policy_id: policy.policy_id().to_string(),
        policy_version: policy.policy_version().to_string(),
        policy_pre_state_sha256,
        control_policy_post_state_sha256,
        treatment_policy_post_state_sha256,
        arm_order_swap_verified: true,
        target: pair.target,
        control_action,
        treatment_action,
        action_delta,
        targeted_input_sensitivity,
        collateral_changed_feature_count,
    };
    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|error| RmiEvaluationErrorV1::SerializationFailure(error.to_string()))?;

    Ok(RmiEvaluationEnvelopeV1 {
        payload,
        payload_sha256: sha256_hex(&payload_bytes),
    })
}

/// Evaluate the second direction: how the controlled partner changes its next
/// action in response to the agent's control versus treatment action.
///
/// This simulator track establishes agent-to-partner sensitivity only. A human
/// causal claim requires a separately randomized and ethically approved study.
pub fn evaluate_rmi_reciprocal_pair<P, H>(
    agent_policy: &P,
    partner_policy: &H,
    pair: &RmiInterventionPairV1,
) -> Result<RmiReciprocalEvaluationEnvelopeV1, RmiEvaluationErrorV1>
where
    P: RmiPolicyV1,
    H: RmiPartnerPolicyV1,
{
    let agent = evaluate_rmi_pair(agent_policy, pair)?;
    if partner_policy.policy_id().trim().is_empty() {
        return Err(RmiEvaluationErrorV1::EmptyPartnerPolicyId);
    }
    if partner_policy.policy_version().trim().is_empty() {
        return Err(RmiEvaluationErrorV1::EmptyPartnerPolicyVersion);
    }

    // Independently fork partner policy state for control, treatment and
    // repeatability checks, exactly as done for the agent policy above.
    let partner_snapshot = partner_policy
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let partner_policy_pre_state_sha256 = snapshot_sha256_v1(&partner_snapshot)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;

    let mut control_partner = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "control".into(),
            reason,
        })?;
    let mut control_partner_replay = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "control_replay".into(),
            reason,
        })?;
    let mut treatment_partner = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "treatment".into(),
            reason,
        })?;
    let mut treatment_partner_replay = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "treatment_replay".into(),
            reason,
        })?;

    for (arm, restored) in [
        ("control", &control_partner),
        ("control_replay", &control_partner_replay),
        ("treatment", &treatment_partner),
        ("treatment_replay", &treatment_partner_replay),
    ] {
        if restored.policy_id() != partner_policy.policy_id()
            || restored.policy_version() != partner_policy.policy_version()
        {
            return Err(RmiEvaluationErrorV1::PartnerPolicyIdentityMismatch {
                arm: arm.to_owned(),
            });
        }
        let restored_snapshot = restored
            .snapshot_state()
            .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
        if restored_snapshot != partner_snapshot {
            return Err(RmiEvaluationErrorV1::PartnerStateMismatch {
                arm: arm.to_owned(),
            });
        }
    }

    let control_partner_response = control_partner.respond(
        &pair.pre_intervention_history,
        &agent.payload.control_action,
        pair.seed,
    );
    let control_partner_replay_response = control_partner_replay.respond(
        &pair.pre_intervention_history,
        &agent.payload.control_action,
        pair.seed,
    );
    let control_partner_post_state = control_partner
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let control_partner_post_state_sha256 = snapshot_sha256_v1(&control_partner_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let control_partner_replay_post_state = control_partner_replay
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let control_partner_replay_post_state_sha256 = snapshot_sha256_v1(&control_partner_replay_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if control_partner_response != control_partner_replay_response
        || control_partner_post_state_sha256 != control_partner_replay_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPartnerOutput {
            arm: "control".to_string(),
        });
    }

    let treatment_partner_response = treatment_partner.respond(
        &pair.pre_intervention_history,
        &agent.payload.treatment_action,
        pair.seed,
    );
    let treatment_partner_replay_response = treatment_partner_replay.respond(
        &pair.pre_intervention_history,
        &agent.payload.treatment_action,
        pair.seed,
    );
    let treatment_partner_post_state = treatment_partner
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let treatment_partner_post_state_sha256 = snapshot_sha256_v1(&treatment_partner_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let treatment_partner_replay_post_state = treatment_partner_replay
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let treatment_partner_replay_post_state_sha256 = snapshot_sha256_v1(&treatment_partner_replay_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if treatment_partner_response != treatment_partner_replay_response
        || treatment_partner_post_state_sha256 != treatment_partner_replay_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPartnerOutput {
            arm: "treatment".to_string(),
        });
    }

    let mut treatment_partner_order_probe = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "order_swap_treatment".into(),
            reason,
        })?;
    let mut control_partner_order_probe = H::restore_from_snapshot(&partner_snapshot)
        .map_err(|reason| RmiEvaluationErrorV1::PartnerStateRestoreFailure {
            arm: "order_swap_control".into(),
            reason,
        })?;
    let reverse_treatment_partner_response = treatment_partner_order_probe.respond(
        &pair.pre_intervention_history,
        &agent.payload.treatment_action,
        pair.seed,
    );
    let treatment_partner_order_probe_post_state = treatment_partner_order_probe
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let reverse_treatment_partner_post_state_sha256 = snapshot_sha256_v1(&treatment_partner_order_probe_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    let reverse_control_partner_response = control_partner_order_probe.respond(
        &pair.pre_intervention_history,
        &agent.payload.control_action,
        pair.seed,
    );
    let control_partner_order_probe_post_state = control_partner_order_probe
        .snapshot_state()
        .map_err(RmiEvaluationErrorV1::PartnerStateSnapshotFailure)?;
    let reverse_control_partner_post_state_sha256 = snapshot_sha256_v1(&control_partner_order_probe_post_state)
        .map_err(RmiEvaluationErrorV1::SerializationFailure)?;
    if reverse_treatment_partner_response != treatment_partner_response
        || reverse_treatment_partner_post_state_sha256 != treatment_partner_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPartnerOutput {
            arm: "order_swap_treatment".to_string(),
        });
    }
    if reverse_control_partner_response != control_partner_response
        || reverse_control_partner_post_state_sha256 != control_partner_post_state_sha256
    {
        return Err(RmiEvaluationErrorV1::NonDeterministicPartnerOutput {
            arm: "order_swap_control".to_string(),
        });
    }

    for (arm, response) in [
        ("control", &control_partner_response),
        ("treatment", &treatment_partner_response),
    ] {
        if response.response_id.trim().is_empty() {
            return Err(RmiEvaluationErrorV1::InvalidPartnerResponseId {
                arm: arm.to_string(),
            });
        }
        let issues = response.features.validation_issues(arm);
        if !issues.is_empty() {
            return Err(RmiEvaluationErrorV1::InvalidPartnerResponseFeatures {
                arm: arm.to_string(),
                issues,
            });
        }
    }

    let partner_action_delta = RmiActionDeltaV1::between(
        &RmiAgentActionV1 {
            action_id: control_partner_response.response_id.clone(),
            features: control_partner_response.features.clone(),
        },
        &RmiAgentActionV1 {
            action_id: treatment_partner_response.response_id.clone(),
            features: treatment_partner_response.features.clone(),
        },
    );
    let targeted_partner_sensitivity =
        partner_action_delta.targeted_feature_changed(pair.target);
    let partner_collateral_changed_feature_count =
        partner_action_delta.collateral_changed_count(pair.target);

    let payload = RmiReciprocalEvaluationPayloadV1 {
        schema_version: RMI_BENCH_SCHEMA_VERSION,
        trial_id: pair.trial_id.clone(),
        seed: pair.seed,
        agent_evaluation_sha256: agent.payload_sha256,
        partner_policy_id: partner_policy.policy_id().to_string(),
        partner_policy_version: partner_policy.policy_version().to_string(),
        partner_policy_pre_state_sha256,
        control_partner_post_state_sha256,
        treatment_partner_post_state_sha256,
        partner_arm_order_swap_verified: true,
        target: pair.target,
        agent_targeted_sensitivity: agent.payload.targeted_input_sensitivity,
        control_partner_response,
        treatment_partner_response,
        partner_action_delta,
        targeted_partner_sensitivity,
        partner_collateral_changed_feature_count,
    };
    let payload_bytes = serde_json::to_vec(&payload)
        .map_err(|error| RmiEvaluationErrorV1::SerializationFailure(error.to_string()))?;
    Ok(RmiReciprocalEvaluationEnvelopeV1 {
        payload,
        payload_sha256: sha256_hex(&payload_bytes),
    })
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn snapshot_sha256_v1<T: Serialize>(snapshot: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(snapshot).map_err(|error| error.to_string())?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        // Writing to a String is infallible.
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        static UNTRACKED_PROCESS_CALLS: Cell<u32> = Cell::new(0);
    }

    fn base_features() -> RmiMusicalFeaturesV1 {
        RmiMusicalFeaturesV1 {
            motif_intervals: [2, -1, 3, 0],
            rhythm_onsets: [true, false, true, false, true, false, false, true],
            tempo_milli_bpm: 120_000,
            role: RmiRoleV1::Lead,
        }
    }

    fn pair_for(target: RmiTargetFeatureV1) -> RmiInterventionPairV1 {
        let control = base_features();
        let mut treatment = control.clone();
        match target {
            RmiTargetFeatureV1::Motif => treatment.motif_intervals[0] += 2,
            RmiTargetFeatureV1::Rhythm => treatment.rhythm_onsets[1] = true,
            RmiTargetFeatureV1::Tempo => treatment.tempo_milli_bpm += 10_000,
            RmiTargetFeatureV1::Role => treatment.role = RmiRoleV1::Accompaniment,
        }
        let history = vec![RmiHistoryEventV1 {
            event_id: "prior-event-0".into(),
            actor: RmiHistoryActorV1::Partner,
            features: base_features(),
        }];
        let history_bytes = serde_json::to_vec(&history).unwrap();
        RmiInterventionPairV1 {
            trial_id: format!("synthetic-{:?}", target),
            seed: 42,
            pre_intervention_history: history,
            pre_intervention_state_sha256: sha256_hex(&history_bytes),
            target,
            control_observation: RmiPartnerObservationV1 {
                source_event_id: "partner-event-1".into(),
                features: control,
            },
            treatment_observation: RmiPartnerObservationV1 {
                source_event_id: "partner-event-1".into(),
                features: treatment,
            },
        }
    }

    fn fixed_action() -> RmiAgentActionV1 {
        RmiAgentActionV1 {
            action_id: "fixed-action".into(),
            features: base_features(),
        }
    }

    #[test]
    fn fixed_partner_does_not_show_agent_to_partner_sensitivity() {
        let pair = pair_for(RmiTargetFeatureV1::Role);
        let agent = SyntheticContingentPolicyV1 {
            id: "agent".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Role,
        };
        let partner = FixedPartnerResponsePolicyV1 {
            id: "fixed-partner".into(),
            version: "1".into(),
            response: RmiPartnerResponseV1 {
                response_id: "same-response".into(),
                features: base_features(),
            },
        };
        let result = evaluate_rmi_reciprocal_pair(&agent, &partner, &pair).unwrap();
        assert!(result.payload.agent_evaluation_sha256.len() == 64);
        assert!(!result.payload.targeted_partner_sensitivity);
    }

    #[test]
    fn responsive_partner_positive_control_detects_agent_to_partner_influence() {
        let pair = pair_for(RmiTargetFeatureV1::Role);
        let agent = SyntheticContingentPolicyV1 {
            id: "agent".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Role,
        };
        let partner = SyntheticResponsivePartnerPolicyV1 {
            id: "responsive-partner".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Role,
        };
        let result = evaluate_rmi_reciprocal_pair(&agent, &partner, &pair).unwrap();
        assert!(result.payload.agent_targeted_sensitivity);
        assert!(result.payload.targeted_partner_sensitivity);
        assert_eq!(result.payload.partner_collateral_changed_feature_count, 0);
        assert_eq!(result.payload.agent_evaluation_sha256.len(), 64);
        assert_eq!(result.payload_sha256.len(), 64);
    }

    #[test]
    fn fixed_agent_with_responsive_partner_has_no_agent_to_partner_effect() {
        let pair = pair_for(RmiTargetFeatureV1::Role);
        let agent = FixedResponsePolicyV1 {
            id: "fixed-agent".into(),
            version: "1".into(),
            response: fixed_action(),
        };
        let partner = SyntheticResponsivePartnerPolicyV1 {
            id: "responsive-partner".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Role,
        };
        let result = evaluate_rmi_reciprocal_pair(&agent, &partner, &pair).unwrap();
        assert!(!result.payload.targeted_partner_sensitivity);
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TwoCallCounterPolicy {
        calls: u32,
    }

    impl RmiPolicyV1 for TwoCallCounterPolicy {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
        fn policy_id(&self) -> &str { "two-call-counter-agent" }
        fn policy_version(&self) -> &str { "1" }

        fn respond(
            &mut self,
            _history: &[RmiHistoryEventV1],
            _observation: &RmiPartnerObservationV1,
            _seed: u64,
        ) -> RmiAgentActionV1 {
            // Deliberately ignores the partner observation. The old evaluator
            // called control twice (A/A) then treatment twice (B/B), so its
            // within-arm duplicate checks passed while order manufactured a
            // false treatment effect.
            let call = self.calls;
            self.calls += 1;
            let mut features = base_features();
            if call >= 2 {
                features.role = RmiRoleV1::Accompaniment;
            }
            RmiAgentActionV1 {
                action_id: if call >= 2 { "late-state".into() } else { "early-state".into() },
                features,
            }
        }
    }

    #[test]
    fn policy_state_is_forked_before_each_counterfactual_arm() {
        let policy = TwoCallCounterPolicy { calls: 0 };
        let result = evaluate_rmi_pair(&policy, &pair_for(RmiTargetFeatureV1::Role)).unwrap();

        assert!(!result.payload.targeted_input_sensitivity);
        assert_eq!(
            result.payload.control_action,
            result.payload.treatment_action,
            "state progression alone must not manufacture partner-to-agent sensitivity"
        );
        assert_eq!(
            result.payload.control_policy_post_state_sha256,
            result.payload.treatment_policy_post_state_sha256,
            "both arms start from the same state and take one corresponding transition"
        );
        assert_eq!(result.payload.policy_pre_state_sha256.len(), 64);
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct ProcessGlobalCounterPolicy;

    impl RmiPolicyV1 for ProcessGlobalCounterPolicy {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
        fn policy_id(&self) -> &str { "process-global-counter" }
        fn policy_version(&self) -> &str { "1" }

        fn respond(
            &mut self,
            _history: &[RmiHistoryEventV1],
            _observation: &RmiPartnerObservationV1,
            _seed: u64,
        ) -> RmiAgentActionV1 {
            // This state is intentionally absent from Serialize. Calls 0-1
            // return A and calls 2-3 return B: the old evaluator's control,
            // control-repeat, treatment, treatment-repeat sequence therefore
            // passed its within-arm duplicate checks while fabricating an
            // apparent treatment effect. The reversed-order replay must catch it.
            let call = UNTRACKED_PROCESS_CALLS.with(|counter| {
                let call = counter.get();
                counter.set(call + 1);
                call
            });
            let mut features = base_features();
            if call % 4 >= 2 {
                features.role = RmiRoleV1::Accompaniment;
            }
            RmiAgentActionV1 {
                action_id: if call % 4 >= 2 { "global-B".into() } else { "global-A".into() },
                features,
            }
        }
    }

    #[test]
    fn arm_order_swap_rejects_untracked_process_global_state() {
        UNTRACKED_PROCESS_CALLS.with(|counter| counter.set(0));
        let policy = ProcessGlobalCounterPolicy;
        let result = evaluate_rmi_pair(&policy, &pair_for(RmiTargetFeatureV1::Role));
        assert!(matches!(
            result,
            Err(RmiEvaluationErrorV1::NonDeterministicPolicyOutput { arm })
                if arm == "order_swap_treatment"
        ));
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct TwoCallCounterPartner {
        calls: u32,
    }

    impl RmiPartnerPolicyV1 for TwoCallCounterPartner {
    type Snapshot = Self;
    fn snapshot_state(&self) -> Result<Self::Snapshot, String> { Ok(self.clone()) }
    fn restore_from_snapshot(snapshot: &Self::Snapshot) -> Result<Self, String> { Ok(snapshot.clone()) }
        fn policy_id(&self) -> &str { "two-call-counter-partner" }
        fn policy_version(&self) -> &str { "1" }

        fn respond(
            &mut self,
            _history: &[RmiHistoryEventV1],
            _agent_action: &RmiAgentActionV1,
            _seed: u64,
        ) -> RmiPartnerResponseV1 {
            // Also ignores the agent action; the output transition depends
            // only on an internal serialized call counter.
            let call = self.calls;
            self.calls += 1;
            let mut features = base_features();
            if call >= 2 {
                features.role = RmiRoleV1::Accompaniment;
            }
            RmiPartnerResponseV1 {
                response_id: if call >= 2 { "late-state".into() } else { "early-state".into() },
                features,
            }
        }
    }

    #[test]
    fn partner_state_is_forked_before_each_counterfactual_arm() {
        let agent = FixedResponsePolicyV1 {
            id: "fixed-agent".into(),
            version: "1".into(),
            response: fixed_action(),
        };
        let partner = TwoCallCounterPartner { calls: 0 };
        let result = evaluate_rmi_reciprocal_pair(
            &agent,
            &partner,
            &pair_for(RmiTargetFeatureV1::Role),
        )
        .unwrap();

        assert!(!result.payload.targeted_partner_sensitivity);
        assert_eq!(
            result.payload.control_partner_response,
            result.payload.treatment_partner_response,
            "partner state progression alone must not manufacture agent-to-partner influence"
        );
        assert_eq!(
            result.payload.control_partner_post_state_sha256,
            result.payload.treatment_partner_post_state_sha256
        );
        assert_eq!(result.payload.partner_policy_pre_state_sha256.len(), 64);
    }

    #[test]
    fn fixed_response_does_not_pass_targeted_input_sensitivity() {
        for target in [
            RmiTargetFeatureV1::Motif,
            RmiTargetFeatureV1::Rhythm,
            RmiTargetFeatureV1::Tempo,
            RmiTargetFeatureV1::Role,
        ] {
            let policy = FixedResponsePolicyV1 {
                id: "fixed".into(),
                version: "1".into(),
                response: fixed_action(),
            };
            let result = evaluate_rmi_pair(&policy, &pair_for(target)).unwrap();
            assert!(!result.payload.targeted_input_sensitivity);
        }
    }

    #[test]
    fn one_way_prompt_and_yoked_replay_do_not_pass_contingency() {
        let pair = pair_for(RmiTargetFeatureV1::Motif);
        let one_way = OneWayPromptPolicyV1 {
            id: "one-way".into(),
            version: "1".into(),
            frozen_response: fixed_action(),
        };
        let yoked = YokedReplayPolicyV1 {
            id: "yoked".into(),
            version: "1".into(),
            replayed_response: fixed_action(),
        };
        assert!(
            !evaluate_rmi_pair(&one_way, &pair)
                .unwrap()
                .payload
                .targeted_input_sensitivity
        );
        assert!(
            !evaluate_rmi_pair(&yoked, &pair)
                .unwrap()
                .payload
                .targeted_input_sensitivity
        );
    }

    #[test]
    fn imitation_is_input_sensitive_but_is_not_automatically_musical_collaboration() {
        let pair = pair_for(RmiTargetFeatureV1::Motif);
        let policy = ImitationOnlyPolicyV1 {
            id: "imitation".into(),
            version: "1".into(),
        };
        let result = evaluate_rmi_pair(&policy, &pair).unwrap();
        assert!(result.payload.targeted_input_sensitivity);
        // The result intentionally contains no musical-quality or
        // complementarity verdict: copying is not proof of co-creation.
    }

    #[test]
    fn synthetic_contingent_positive_control_detects_each_target_feature() {
        for target in [
            RmiTargetFeatureV1::Motif,
            RmiTargetFeatureV1::Rhythm,
            RmiTargetFeatureV1::Tempo,
            RmiTargetFeatureV1::Role,
        ] {
            let policy = SyntheticContingentPolicyV1 {
                id: "synthetic-contingent".into(),
                version: "1".into(),
                target,
            };
            let result = evaluate_rmi_pair(&policy, &pair_for(target)).unwrap();
            assert!(
                result.payload.targeted_input_sensitivity,
                "expected target sensitivity for {target:?}"
            );
        }
    }

    #[test]
    fn synthetic_positive_control_changes_only_the_manipulated_output_dimension() {
        for target in [
            RmiTargetFeatureV1::Motif,
            RmiTargetFeatureV1::Rhythm,
            RmiTargetFeatureV1::Tempo,
            RmiTargetFeatureV1::Role,
        ] {
            let policy = SyntheticContingentPolicyV1 {
                id: "synthetic-contingent".into(),
                version: "1".into(),
                target,
            };
            let result = evaluate_rmi_pair(&policy, &pair_for(target)).unwrap();
            assert!(result.payload.action_delta.targeted_feature_changed(target));
            assert_eq!(
                result.payload.collateral_changed_feature_count,
                0,
                "unexpected collateral change for {target:?}"
            );
        }
    }

    #[test]
    fn intervention_rejects_mismatched_source_event_ids() {
        let mut pair = pair_for(RmiTargetFeatureV1::Motif);
        pair.treatment_observation.source_event_id = "different-event".into();
        let issues = validate_rmi_intervention_pair(&pair).unwrap_err();
        assert!(issues.contains(&RmiValidationIssueV1::SourceEventIdMismatch));
    }

    #[test]
    fn intervention_rejects_uncontrolled_feature_changes() {
        let mut pair = pair_for(RmiTargetFeatureV1::Motif);
        pair.treatment_observation.features.role = RmiRoleV1::Bass;
        let issues = validate_rmi_intervention_pair(&pair).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            RmiValidationIssueV1::UncontrolledFeatureChanged {
                feature: RmiTargetFeatureV1::Role
            }
        )));
    }

    #[test]
    fn intervention_rejects_noop_target_change() {
        let mut pair = pair_for(RmiTargetFeatureV1::Tempo);
        pair.treatment_observation.features.tempo_milli_bpm =
            pair.control_observation.features.tempo_milli_bpm;
        let issues = validate_rmi_intervention_pair(&pair).unwrap_err();
        assert!(issues.iter().any(|issue| matches!(
            issue,
            RmiValidationIssueV1::TargetFeatureNotChanged {
                target: RmiTargetFeatureV1::Tempo
            }
        )));
    }

    #[test]
    fn evaluation_is_deterministic_and_digest_bound() {
        let policy = SyntheticContingentPolicyV1 {
            id: "synthetic-contingent".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Rhythm,
        };
        let pair = pair_for(RmiTargetFeatureV1::Rhythm);
        let first = evaluate_rmi_pair(&policy, &pair).unwrap();
        let second = evaluate_rmi_pair(&policy, &pair).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.payload_sha256.len(), 64);
        assert!(is_lowercase_sha256(&first.payload_sha256));
    }

    #[test]
    fn changed_seed_changes_the_record_even_for_a_deterministic_policy() {
        let policy = SyntheticContingentPolicyV1 {
            id: "synthetic-contingent".into(),
            version: "1".into(),
            target: RmiTargetFeatureV1::Motif,
        };
        let first_pair = pair_for(RmiTargetFeatureV1::Motif);
        let mut second_pair = first_pair.clone();
        second_pair.seed += 1;
        let first = evaluate_rmi_pair(&policy, &first_pair).unwrap();
        let second = evaluate_rmi_pair(&policy, &second_pair).unwrap();
        assert_ne!(first.payload_sha256, second.payload_sha256);
    }

    #[test]
    fn changed_history_without_matching_digest_fails_closed() {
        let mut pair = pair_for(RmiTargetFeatureV1::Motif);
        pair.pre_intervention_history[0].features.tempo_milli_bpm += 1_000;
        assert!(matches!(
            evaluate_rmi_pair(
                &SyntheticContingentPolicyV1 {
                    id: "synthetic-contingent".into(),
                    version: "1".into(),
                    target: RmiTargetFeatureV1::Motif
                },
                &pair
            ),
            Err(RmiEvaluationErrorV1::InvalidIntervention(_))
        ));
    }

    #[test]
    fn invalid_pre_state_digest_fails_closed() {
        let mut pair = pair_for(RmiTargetFeatureV1::Role);
        pair.pre_intervention_state_sha256 = "not-a-digest".into();
        assert!(matches!(
            evaluate_rmi_pair(
                &SyntheticContingentPolicyV1 {
                    id: "synthetic-contingent".into(),
                    version: "1".into(),
                    target: RmiTargetFeatureV1::Role
                },
                &pair
            ),
            Err(RmiEvaluationErrorV1::InvalidIntervention(_))
        ));
    }
}

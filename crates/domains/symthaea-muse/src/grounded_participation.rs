// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-to-action authorization bridge for musical participation.
//!
//! This module does not re-run temporal grounding. It consumes a versioned,
//! artifact-bound grounding receipt and checks whether a proposed musical
//! action cites the required grounded event claims from the permitted input
//! prefix. The temporal-grounding scorer remains the independent producer of
//! those claims; this module is the policy/authorization consumer.
//!
//! A model confidence value is not a substitute for a claim receipt. A
//! decision cannot authorize an effect if the cited evidence is stale,
//! ambiguous, unavailable, tied to another artifact, or outside the observed
//! input prefix. WAIT, ASK and ABSTAIN remain distinct from active intervention.
//!
//! V1 uses the repository's serde_json serialization for exact receipt
//! commitments. It does not claim a cross-language canonical JSON standard,
//! authenticate the upstream producer, or prove that the runtime truly
//! enforced its declared input prefix.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const GROUNDED_PARTICIPATION_EVIDENCE_SCHEMA_V1: &str =
    "mel-grounded-participation-evidence-v1";
pub const GROUNDED_PARTICIPATION_RECEIPT_SCHEMA_V1: &str =
    "mel-grounded-participation-receipt-v1";
pub const GROUNDED_PARTICIPATION_POLICY_SCHEMA_V1: &str =
    "mel-grounded-participation-policy-v1";

fn is_lower_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .map_err(|error| error.to_string())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedAudioArtifactV1 {
    pub audio_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub total_frames: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedFrameIntervalV1 {
    pub start_frame: u64,
    pub end_frame: u64,
}

impl GroundedFrameIntervalV1 {
    pub fn is_valid(self) -> bool {
        self.start_frame < self.end_frame
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundedMusicalClaimKindV1 {
    Invitation,
    RetainedFloor,
    ActivePhrase,
    HandoffOrStop,
    IntentionalSpace,
    AmbiguousContext,
    StreamUnavailable,
    CollaboratorVeto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundedClaimStatusV1 {
    Grounded,
    Ambiguous,
    Unavailable,
    Contradicted,
    NotEstimable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedMusicalClaimV1 {
    pub claim_id: String,
    pub kind: GroundedMusicalClaimKindV1,
    pub status: GroundedClaimStatusV1,
    /// Evidence source interval, half-open in the exact artifact's frame clock.
    pub source_interval: GroundedFrameIntervalV1,
    /// Last input frame actually claimed as observed by the grounding producer.
    pub observed_through_frame: u64,
    /// Runner monotonic time at which this claim receipt was committed.
    pub committed_at_ns: u64,
    /// Digest of the underlying temporal-grounding query/receipt for this claim.
    pub grounding_receipt_sha256: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedMusicalEvidenceBundleV1 {
    pub schema_version: String,
    pub session_id: String,
    pub turn_id: String,
    pub artifact: GroundedAudioArtifactV1,
    pub temporal_grounding_evaluator_version: String,
    pub annotation_policy_version: String,
    /// Shared process/session monotonic clock domain used by decision/effect receipts.
    pub runner_clock_domain_id: String,
    /// Exact origin identity for the runner monotonic timestamps below.
    pub runner_origin_id: String,
    /// The permitted input prefix end; the end frame itself is not included.
    pub permitted_prefix_end_frame: u64,
    pub committed_at_ns: u64,
    pub claims: Vec<GroundedMusicalClaimV1>,
    /// SHA-256 of the exact serialized claims vector.
    pub claim_set_sha256: String,
    /// SHA-256 of the complete evidence-bundle payload, excluding this field.
    pub evidence_bundle_sha256: String,
}

#[derive(Serialize)]
struct GroundedEvidenceBundleCommitmentPayloadV1<'a> {
    schema_version: &'a str,
    session_id: &'a str,
    turn_id: &'a str,
    artifact: &'a GroundedAudioArtifactV1,
    temporal_grounding_evaluator_version: &'a str,
    annotation_policy_version: &'a str,
    runner_clock_domain_id: &'a str,
    runner_origin_id: &'a str,
    permitted_prefix_end_frame: u64,
    committed_at_ns: u64,
    claims: &'a [GroundedMusicalClaimV1],
    claim_set_sha256: &'a str,
}

impl GroundedMusicalEvidenceBundleV1 {
    pub fn expected_claim_set_sha256(&self) -> Result<String, String> {
        sha256_json(&self.claims)
    }

    pub fn expected_evidence_bundle_sha256(&self) -> Result<String, String> {
        sha256_json(&GroundedEvidenceBundleCommitmentPayloadV1 {
            schema_version: &self.schema_version,
            session_id: &self.session_id,
            turn_id: &self.turn_id,
            artifact: &self.artifact,
            temporal_grounding_evaluator_version: &self.temporal_grounding_evaluator_version,
            annotation_policy_version: &self.annotation_policy_version,
            runner_clock_domain_id: &self.runner_clock_domain_id,
            runner_origin_id: &self.runner_origin_id,
            permitted_prefix_end_frame: self.permitted_prefix_end_frame,
            committed_at_ns: self.committed_at_ns,
            claims: &self.claims,
            claim_set_sha256: &self.claim_set_sha256,
        })
    }

    pub fn is_commitment_valid(&self) -> bool {
        self.expected_claim_set_sha256()
            .is_ok_and(|expected| expected == self.claim_set_sha256)
            && self.expected_evidence_bundle_sha256()
                .is_ok_and(|expected| expected == self.evidence_bundle_sha256)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum GroundedMusicalActionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
    Ask,
    Abstain,
}

impl GroundedMusicalActionV1 {
    pub fn is_active_intervention(self) -> bool {
        matches!(self, Self::Respond | Self::Continue | Self::Yield)
    }

    pub fn allows_effect_window(self) -> bool {
        matches!(self, Self::Respond | Self::Continue | Self::Yield)
    }

    pub fn requires_grounded_event(self) -> bool {
        matches!(self, Self::Respond | Self::Continue | Self::Yield | Self::Wait)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedParticipationEffectWindowV1 {
    /// Runner-monotonic not-before timestamp, measured from runner_origin_id.
    pub not_before_runner_time_ns: u64,
    /// Expiry timestamp in the same runner clock origin.
    pub expires_at_runner_time_ns: u64,
    pub authorization_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedMusicalDecisionReceiptV1 {
    pub schema_version: String,
    pub session_id: String,
    pub turn_id: String,
    pub decision_id: String,
    pub sequence: u64,
    pub artifact: GroundedAudioArtifactV1,
    pub runner_clock_domain_id: String,
    pub runner_origin_id: String,
    /// Exact claim-set digest from the evidence bundle used by the decision.
    pub evidence_claim_set_sha256: String,
    /// Exact commitment to the full evidence metadata and claim set.
    pub evidence_bundle_sha256: String,
    /// Last input frame available when the decision was made.
    pub observed_through_frame: u64,
    /// Runner monotonic time when the decision was committed.
    pub decision_at_ns: u64,
    pub action: GroundedMusicalActionV1,
    /// Exact evidence claims cited as justification for the action.
    pub cited_claim_ids: Vec<String>,
    pub effect_window: Option<GroundedParticipationEffectWindowV1>,
    /// Veto is an authority state, not an inferred musical feature.
    pub collaborator_veto_active: bool,
    pub policy_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroundedParticipationPolicyV1 {
    pub schema_version: String,
    pub policy_version: String,
    pub require_grounded_claims_for_wait: bool,
    pub maximum_evidence_age_ns: u64,
    pub minimum_effect_lead_ns: u64,
    pub maximum_effect_window_ns: u64,
}

impl Default for GroundedParticipationPolicyV1 {
    fn default() -> Self {
        Self {
            schema_version: GROUNDED_PARTICIPATION_POLICY_SCHEMA_V1.to_owned(),
            policy_version: "mel-grounded-participation-policy-v1".to_owned(),
            require_grounded_claims_for_wait: true,
            maximum_evidence_age_ns: 2_000_000_000,
            minimum_effect_lead_ns: 10_000_000,
            maximum_effect_window_ns: 2_000_000_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroundedParticipationIssueV1 {
    WrongEvidenceSchema,
    WrongDecisionSchema,
    WrongPolicySchema,
    EmptyIdentity { field: String },
    InvalidAudioDigest,
    InvalidAudioClock { field: String },
    InvalidTemporalGroundingVersion,
    InvalidAnnotationPolicyVersion,
    InvalidPrefix,
    InvalidClaimSetDigest,
    EmptyClaimSet,
    DuplicateClaimId { claim_index: usize },
    InvalidClaimId { claim_index: usize },
    InvalidClaimReceiptDigest { claim_index: usize },
    InvalidClaimInterval { claim_index: usize },
    ClaimBeyondArtifact { claim_index: usize },
    ClaimBeyondObservedPrefix { claim_index: usize },
    ClaimCommittedAfterEvidenceBundle { claim_index: usize },
    MissingReasonForNonGroundedClaim { claim_index: usize },
    DuplicateDecisionClaim { claim_index: usize },
    DecisionEvidenceSetEmpty,
    DecisionClaimMissing { claim_id: String },
    DecisionClaimNotGrounded { claim_id: String, status: GroundedClaimStatusV1 },
    DecisionClaimNotObservedByPrefix { claim_id: String },
    DecisionEvidenceTooOld { claim_id: String },
    DecisionClaimCommittedAfterDecision { claim_id: String },
    RequiredEvidenceKindMissing { action: GroundedMusicalActionV1 },
    ContradictoryEvidenceKind { action: GroundedMusicalActionV1 },
    VetoBlocksActiveIntervention,
    UnavailableStreamBlocksActiveIntervention,
    EffectWindowMissing { action: GroundedMusicalActionV1 },
    EffectWindowUnexpected { action: GroundedMusicalActionV1 },
    InvalidEffectWindow,
    EffectWindowBeforeDecision,
    EffectWindowDurationOutsidePolicy,
    EffectAuthorizationIdEmpty,
    RunnerClockMismatch,
    SessionMismatch,
    TurnMismatch,
    ArtifactMismatch { field: String },
    EvidenceCommitmentMismatch,
    PolicyVersionMismatch,
    DecisionBeforeEvidenceBundle,
    MissingEvidenceForAbstainOrAsk,
    AskRequiresAmbiguousGrounding,
    AbstainRequiresNonGroundedEvidence,
    StreamUnavailableRequiresAbstain,
}

impl std::fmt::Display for GroundedParticipationIssueV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for GroundedParticipationIssueV1 {}

fn validate_artifact(
    artifact: &GroundedAudioArtifactV1,
    issues: &mut Vec<GroundedParticipationIssueV1>,
) {
    if !is_lower_hex_digest(&artifact.audio_sha256) {
        issues.push(GroundedParticipationIssueV1::InvalidAudioDigest);
    }
    if artifact.sample_rate_hz == 0 {
        issues.push(GroundedParticipationIssueV1::InvalidAudioClock {
            field: "sample_rate_hz".to_owned(),
        });
    }
    if artifact.channels == 0 {
        issues.push(GroundedParticipationIssueV1::InvalidAudioClock {
            field: "channels".to_owned(),
        });
    }
    if artifact.total_frames == 0 {
        issues.push(GroundedParticipationIssueV1::InvalidAudioClock {
            field: "total_frames".to_owned(),
        });
    }
}

pub fn validate_grounded_evidence_bundle_v1(
    bundle: &GroundedMusicalEvidenceBundleV1,
) -> Vec<GroundedParticipationIssueV1> {
    let mut issues = Vec::new();
    if bundle.schema_version != GROUNDED_PARTICIPATION_EVIDENCE_SCHEMA_V1 {
        issues.push(GroundedParticipationIssueV1::WrongEvidenceSchema);
    }
    for (field, value) in [
        ("session_id", bundle.session_id.as_str()),
        ("turn_id", bundle.turn_id.as_str()),
        ("temporal_grounding_evaluator_version", bundle.temporal_grounding_evaluator_version.as_str()),
        ("annotation_policy_version", bundle.annotation_policy_version.as_str()),
        ("runner_clock_domain_id", bundle.runner_clock_domain_id.as_str()),
        ("runner_origin_id", bundle.runner_origin_id.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(GroundedParticipationIssueV1::EmptyIdentity {
                field: field.to_owned(),
            });
        }
    }
    validate_artifact(&bundle.artifact, &mut issues);
    if bundle.permitted_prefix_end_frame == 0
        || bundle.permitted_prefix_end_frame > bundle.artifact.total_frames
    {
        issues.push(GroundedParticipationIssueV1::InvalidPrefix);
    }
    if bundle.claims.is_empty() {
        issues.push(GroundedParticipationIssueV1::EmptyClaimSet);
    }
    if !is_lower_hex_digest(&bundle.claim_set_sha256)
        || !is_lower_hex_digest(&bundle.evidence_bundle_sha256)
        || !bundle.is_commitment_valid()
    {
        issues.push(GroundedParticipationIssueV1::InvalidClaimSetDigest);
    }

    let mut seen = BTreeSet::new();
    for (index, claim) in bundle.claims.iter().enumerate() {
        if claim.claim_id.trim().is_empty() || !seen.insert(claim.claim_id.as_str()) {
            issues.push(if claim.claim_id.trim().is_empty() {
                GroundedParticipationIssueV1::InvalidClaimId { claim_index: index }
            } else {
                GroundedParticipationIssueV1::DuplicateClaimId { claim_index: index }
            });
        }
        if !is_lower_hex_digest(&claim.grounding_receipt_sha256) {
            issues.push(GroundedParticipationIssueV1::InvalidClaimReceiptDigest {
                claim_index: index,
            });
        }
        if !claim.source_interval.is_valid() {
            issues.push(GroundedParticipationIssueV1::InvalidClaimInterval {
                claim_index: index,
            });
        }
        if claim.source_interval.end_frame > bundle.artifact.total_frames {
            issues.push(GroundedParticipationIssueV1::ClaimBeyondArtifact {
                claim_index: index,
            });
        }
        if claim.observed_through_frame > bundle.permitted_prefix_end_frame
            || claim.source_interval.end_frame > bundle.permitted_prefix_end_frame
        {
            issues.push(GroundedParticipationIssueV1::ClaimBeyondObservedPrefix {
                claim_index: index,
            });
        }
        if claim.committed_at_ns > bundle.committed_at_ns {
            issues.push(GroundedParticipationIssueV1::ClaimCommittedAfterEvidenceBundle {
                claim_index: index,
            });
        }
        if claim.status != GroundedClaimStatusV1::Grounded
            && claim.reason.as_deref().is_none_or(|reason| reason.trim().is_empty())
        {
            issues.push(GroundedParticipationIssueV1::MissingReasonForNonGroundedClaim {
                claim_index: index,
            });
        }
    }
    issues
}

/// Validate that a decision is authorized by grounded claims from the exact
/// artifact and permitted prefix. This function consumes a separately scored
/// evidence bundle; it does not independently verify the source audio bytes.
pub fn validate_grounded_decision_v1(
    bundle: &GroundedMusicalEvidenceBundleV1,
    decision: &GroundedMusicalDecisionReceiptV1,
    policy: &GroundedParticipationPolicyV1,
) -> Vec<GroundedParticipationIssueV1> {
    let mut issues = validate_grounded_evidence_bundle_v1(bundle);
    if decision.schema_version != GROUNDED_PARTICIPATION_RECEIPT_SCHEMA_V1 {
        issues.push(GroundedParticipationIssueV1::WrongDecisionSchema);
    }
    if policy.schema_version != GROUNDED_PARTICIPATION_POLICY_SCHEMA_V1 {
        issues.push(GroundedParticipationIssueV1::WrongPolicySchema);
    }
    for (field, value) in [
        ("decision.session_id", decision.session_id.as_str()),
        ("decision.turn_id", decision.turn_id.as_str()),
        ("decision.decision_id", decision.decision_id.as_str()),
        ("decision.policy_version", decision.policy_version.as_str()),
        ("decision.runner_clock_domain_id", decision.runner_clock_domain_id.as_str()),
        ("decision.runner_origin_id", decision.runner_origin_id.as_str()),
    ] {
        if value.trim().is_empty() {
            issues.push(GroundedParticipationIssueV1::EmptyIdentity {
                field: field.to_owned(),
            });
        }
    }

    let decision_requires_grounded_event = decision.action.is_active_intervention()
        || (decision.action == GroundedMusicalActionV1::Wait
            && policy.require_grounded_claims_for_wait);
    if decision.cited_claim_ids.is_empty() && decision_requires_grounded_event {
        issues.push(GroundedParticipationIssueV1::DecisionEvidenceSetEmpty);
    }
    if decision.session_id != bundle.session_id {
        issues.push(GroundedParticipationIssueV1::SessionMismatch);
    }
    if decision.turn_id != bundle.turn_id {
        issues.push(GroundedParticipationIssueV1::TurnMismatch);
    }
    if decision.runner_clock_domain_id != bundle.runner_clock_domain_id
        || decision.runner_origin_id != bundle.runner_origin_id
    {
        issues.push(GroundedParticipationIssueV1::RunnerClockMismatch);
    }
    if decision.artifact != bundle.artifact {
        issues.push(GroundedParticipationIssueV1::ArtifactMismatch {
            field: "audio_artifact_clock".to_owned(),
        });
    }
    if decision.evidence_claim_set_sha256 != bundle.claim_set_sha256
        || decision.evidence_bundle_sha256 != bundle.evidence_bundle_sha256
    {
        issues.push(GroundedParticipationIssueV1::EvidenceCommitmentMismatch);
    }
    if decision.policy_version != policy.policy_version {
        issues.push(GroundedParticipationIssueV1::PolicyVersionMismatch);
    }
    if decision.observed_through_frame != bundle.permitted_prefix_end_frame
        || decision.observed_through_frame > bundle.artifact.total_frames
    {
        issues.push(GroundedParticipationIssueV1::InvalidPrefix);
    }
    if decision.decision_at_ns < bundle.committed_at_ns {
        issues.push(GroundedParticipationIssueV1::DecisionBeforeEvidenceBundle);
    }
    let evidence_veto_active = bundle.claims.iter().any(|claim| {
        claim.kind == GroundedMusicalClaimKindV1::CollaboratorVeto
            && claim.status == GroundedClaimStatusV1::Grounded
    });
    if (decision.collaborator_veto_active || evidence_veto_active)
        && decision.action.is_active_intervention()
    {
        issues.push(GroundedParticipationIssueV1::VetoBlocksActiveIntervention);
    }

    let mut cited = BTreeSet::new();
    let mut cited_claims = Vec::new();
    for claim_id in &decision.cited_claim_ids {
        if !cited.insert(claim_id.as_str()) {
            issues.push(GroundedParticipationIssueV1::DuplicateDecisionClaim {
                claim_index: cited_claims.len(),
            });
            continue;
        }
        match bundle.claims.iter().find(|claim| claim.claim_id == *claim_id) {
            None => issues.push(GroundedParticipationIssueV1::DecisionClaimMissing {
                claim_id: claim_id.clone(),
            }),
            Some(claim) => {
                let requires_grounded_claim = decision.action.is_active_intervention()
                    || (decision.action == GroundedMusicalActionV1::Wait
                        && policy.require_grounded_claims_for_wait);
                if requires_grounded_claim && claim.status != GroundedClaimStatusV1::Grounded {
                    issues.push(GroundedParticipationIssueV1::DecisionClaimNotGrounded {
                        claim_id: claim_id.clone(),
                        status: claim.status,
                    });
                }
                if claim.observed_through_frame > decision.observed_through_frame
                    || claim.source_interval.end_frame > decision.observed_through_frame
                {
                    issues.push(GroundedParticipationIssueV1::DecisionClaimNotObservedByPrefix {
                        claim_id: claim_id.clone(),
                    });
                }
                if decision.decision_at_ns.saturating_sub(claim.committed_at_ns)
                    > policy.maximum_evidence_age_ns
                {
                    issues.push(GroundedParticipationIssueV1::DecisionEvidenceTooOld {
                        claim_id: claim_id.clone(),
                    });
                }
                if claim.committed_at_ns > decision.decision_at_ns {
                    issues.push(GroundedParticipationIssueV1::DecisionClaimCommittedAfterDecision {
                        claim_id: claim_id.clone(),
                    });
                }
                cited_claims.push(claim);
            }
        }
    }

    let required_kind = match decision.action {
        GroundedMusicalActionV1::Respond => Some(GroundedMusicalClaimKindV1::Invitation),
        GroundedMusicalActionV1::Continue => Some(GroundedMusicalClaimKindV1::ActivePhrase),
        GroundedMusicalActionV1::Yield => Some(GroundedMusicalClaimKindV1::HandoffOrStop),
        GroundedMusicalActionV1::Wait if policy.require_grounded_claims_for_wait => {
            let has_wait_authority = cited_claims.iter().any(|claim| {
                claim.status == GroundedClaimStatusV1::Grounded
                    && matches!(
                        claim.kind,
                        GroundedMusicalClaimKindV1::RetainedFloor
                            | GroundedMusicalClaimKindV1::IntentionalSpace
                            | GroundedMusicalClaimKindV1::CollaboratorVeto
                    )
            });
            if !has_wait_authority {
                issues.push(GroundedParticipationIssueV1::RequiredEvidenceKindMissing {
                    action: decision.action,
                });
            }
            None
        }
        _ => None,
    };
    if let Some(kind) = required_kind {
        if !cited_claims
            .iter()
            .any(|claim| claim.kind == kind && claim.status == GroundedClaimStatusV1::Grounded)
        {
            issues.push(GroundedParticipationIssueV1::RequiredEvidenceKindMissing {
                action: decision.action,
            });
        }
    }
    let has_contradiction = cited_claims.iter().any(|claim| {
        matches!(
            claim.kind,
            GroundedMusicalClaimKindV1::CollaboratorVeto
                | GroundedMusicalClaimKindV1::StreamUnavailable
        ) && claim.status == GroundedClaimStatusV1::Grounded
    });
    if has_contradiction && decision.action.is_active_intervention() {
        issues.push(GroundedParticipationIssueV1::ContradictoryEvidenceKind {
            action: decision.action,
        });
    }

    match decision.action {
        GroundedMusicalActionV1::Ask => {
            if cited_claims.is_empty() {
                issues.push(GroundedParticipationIssueV1::MissingEvidenceForAbstainOrAsk);
            } else if !cited_claims.iter().any(|claim| {
                claim.status == GroundedClaimStatusV1::Ambiguous
                    || claim.kind == GroundedMusicalClaimKindV1::AmbiguousContext
            }) {
                issues.push(GroundedParticipationIssueV1::AskRequiresAmbiguousGrounding);
            }
        }
        GroundedMusicalActionV1::Abstain => {
            if cited_claims.is_empty() {
                issues.push(GroundedParticipationIssueV1::MissingEvidenceForAbstainOrAsk);
            } else if cited_claims.iter().all(|claim| {
                claim.status == GroundedClaimStatusV1::Grounded
                    && claim.kind != GroundedMusicalClaimKindV1::StreamUnavailable
                    && claim.kind != GroundedMusicalClaimKindV1::AmbiguousContext
            }) {
                issues.push(GroundedParticipationIssueV1::AbstainRequiresNonGroundedEvidence);
            }
        }
        _ => {}
    }

    let stream_unavailable = bundle.claims.iter().any(|claim| {
        claim.kind == GroundedMusicalClaimKindV1::StreamUnavailable
            && claim.status == GroundedClaimStatusV1::Grounded
    });
    if stream_unavailable
        && decision.action != GroundedMusicalActionV1::Abstain
        && decision.action != GroundedMusicalActionV1::Ask
    {
        issues.push(GroundedParticipationIssueV1::StreamUnavailableRequiresAbstain);
    }

    match decision.action.allows_effect_window() {
        true => {
            let Some(effect) = decision.effect_window.as_ref() else {
                issues.push(GroundedParticipationIssueV1::EffectWindowMissing {
                    action: decision.action,
                });
                return issues;
            };
            if effect.not_before_runner_time_ns >= effect.expires_at_runner_time_ns {
                issues.push(GroundedParticipationIssueV1::InvalidEffectWindow);
            }
            if effect.not_before_runner_time_ns
                < decision.decision_at_ns.saturating_add(policy.minimum_effect_lead_ns)
            {
                issues.push(GroundedParticipationIssueV1::EffectWindowBeforeDecision);
            }
            if effect.expires_at_runner_time_ns
                .saturating_sub(effect.not_before_runner_time_ns)
                > policy.maximum_effect_window_ns
            {
                issues.push(GroundedParticipationIssueV1::EffectWindowDurationOutsidePolicy);
            }
            if effect.authorization_id.trim().is_empty() {
                issues.push(GroundedParticipationIssueV1::EffectAuthorizationIdEmpty);
            }
        }
        false if decision.effect_window.is_some() => {
            issues.push(GroundedParticipationIssueV1::EffectWindowUnexpected {
                action: decision.action,
            });
        }
        false => {}
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> GroundedAudioArtifactV1 {
        GroundedAudioArtifactV1 {
            audio_sha256: "a".repeat(64),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48_000,
        }
    }

    fn claim(
        id: &str,
        kind: GroundedMusicalClaimKindV1,
        status: GroundedClaimStatusV1,
        end_frame: u64,
    ) -> GroundedMusicalClaimV1 {
        GroundedMusicalClaimV1 {
            claim_id: id.to_owned(),
            kind,
            status,
            source_interval: GroundedFrameIntervalV1 {
                start_frame: end_frame.saturating_sub(2_400),
                end_frame,
            },
            observed_through_frame: end_frame,
            committed_at_ns: 100,
            grounding_receipt_sha256: "b".repeat(64),
            reason: (status != GroundedClaimStatusV1::Grounded)
                .then(|| "independent grounding did not qualify this claim".to_owned()),
        }
    }

    fn bundle() -> GroundedMusicalEvidenceBundleV1 {
        let claims = vec![
            claim(
                "invitation-001",
                GroundedMusicalClaimKindV1::Invitation,
                GroundedClaimStatusV1::Grounded,
                9_600,
            ),
            claim(
                "retains-floor-001",
                GroundedMusicalClaimKindV1::RetainedFloor,
                GroundedClaimStatusV1::Grounded,
                9_600,
            ),
            claim(
                "active-phrase-001",
                GroundedMusicalClaimKindV1::ActivePhrase,
                GroundedClaimStatusV1::Grounded,
                9_600,
            ),
            claim(
                "handoff-001",
                GroundedMusicalClaimKindV1::HandoffOrStop,
                GroundedClaimStatusV1::Grounded,
                9_600,
            ),
        ];
        let mut evidence = GroundedMusicalEvidenceBundleV1 {
            schema_version: GROUNDED_PARTICIPATION_EVIDENCE_SCHEMA_V1.to_owned(),
            session_id: "session-001".to_owned(),
            turn_id: "turn-001".to_owned(),
            artifact: artifact(),
            temporal_grounding_evaluator_version: "mel-temporal-grounding-evaluator-v1".to_owned(),
            annotation_policy_version: "musician-pilot-v1".to_owned(),
            runner_clock_domain_id: "runner-monotonic-v1".to_owned(),
            runner_origin_id: "runner-origin-001".to_owned(),
            permitted_prefix_end_frame: 12_000,
            committed_at_ns: 110,
            claims,
            claim_set_sha256: String::new(),
        };
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        evidence
    }

    fn policy() -> GroundedParticipationPolicyV1 {
        GroundedParticipationPolicyV1::default()
    }

    fn decision(
        evidence: &GroundedMusicalEvidenceBundleV1,
        action: GroundedMusicalActionV1,
        cited_claim_ids: Vec<&str>,
        effect_window: Option<GroundedParticipationEffectWindowV1>,
    ) -> GroundedMusicalDecisionReceiptV1 {
        GroundedMusicalDecisionReceiptV1 {
            schema_version: GROUNDED_PARTICIPATION_RECEIPT_SCHEMA_V1.to_owned(),
            session_id: evidence.session_id.clone(),
            turn_id: evidence.turn_id.clone(),
            decision_id: "decision-001".to_owned(),
            sequence: 1,
            artifact: evidence.artifact.clone(),
            runner_clock_domain_id: evidence.runner_clock_domain_id.clone(),
            runner_origin_id: evidence.runner_origin_id.clone(),
            evidence_claim_set_sha256: evidence.claim_set_sha256.clone(),
            evidence_bundle_sha256: evidence.evidence_bundle_sha256.clone(),
            observed_through_frame: evidence.permitted_prefix_end_frame,
            decision_at_ns: 120,
            action,
            cited_claim_ids: cited_claim_ids.into_iter().map(str::to_owned).collect(),
            effect_window,
            collaborator_veto_active: false,
            policy_version: policy().policy_version,
        }
    }

    fn effect() -> Option<GroundedParticipationEffectWindowV1> {
        Some(GroundedParticipationEffectWindowV1 {
            not_before_runner_time_ns: 10_000_120,
            expires_at_runner_time_ns: 10_100_120,
            authorization_id: "auth-001".to_owned(),
        })
    }

    #[derive(Deserialize)]
    struct GroundedParticipationFixtureSetV1 {
        schema_version: String,
        cases: Vec<GroundedParticipationFixtureCaseV1>,
    }

    #[derive(Deserialize)]
    struct GroundedParticipationFixtureCaseV1 {
        case_id: String,
        expected_disposition: String,
        evidence: GroundedMusicalEvidenceBundleV1,
        decision: GroundedMusicalDecisionReceiptV1,
        policy: GroundedParticipationPolicyV1,
    }

    #[test]
    fn checked_in_evidence_to_action_cases_match_the_authorization_validator() {
        let fixtures: GroundedParticipationFixtureSetV1 = serde_json::from_str(include_str!(
            "../../../../benchmarks/mel/grounded-participation/grounded-participation-cases-v1.json"
        ))
        .expect("grounded participation fixture set must parse");
        assert_eq!(
            fixtures.schema_version,
            "mel-grounded-participation-cases-v1"
        );
        assert_eq!(fixtures.cases.len(), 4);

        for case in fixtures.cases {
            let issues = validate_grounded_decision_v1(
                &case.evidence,
                &case.decision,
                &case.policy,
            );
            match case.expected_disposition.as_str() {
                "accept" => assert!(
                    issues.is_empty(),
                    "{} unexpectedly rejected: {issues:?}",
                    case.case_id
                ),
                "reject_veto" => assert!(
                    issues.contains(&GroundedParticipationIssueV1::VetoBlocksActiveIntervention),
                    "{} must reject a vetoed intervention: {issues:?}",
                    case.case_id
                ),
                other => panic!("{} has unsupported expected disposition {other}", case.case_id),
            }
        }
    }

    #[test]
    fn ask_requires_ambiguous_grounding_instead_of_a_confident_invitation() {
        let evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Ask,
            vec!["invitation-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::AskRequiresAmbiguousGrounding));
    }

    #[test]
    fn unavailable_stream_cannot_authorize_wait_or_active_intervention() {
        let mut evidence = bundle();
        evidence.claims.push(claim(
            "stream-unavailable-001",
            GroundedMusicalClaimKindV1::StreamUnavailable,
            GroundedClaimStatusV1::Grounded,
            9_600,
        ));
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();

        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Wait,
            vec!["retains-floor-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::StreamUnavailableRequiresAbstain));
    }

    #[test]
    fn evidence_bundle_commitment_binds_policy_and_prefix_metadata() {
        let evidence = bundle();
        let mut changed_prefix = evidence.clone();
        changed_prefix.permitted_prefix_end_frame -= 1;
        assert!(!changed_prefix.is_commitment_valid());

        let mut changed_policy = evidence;
        changed_policy.annotation_policy_version = "other-annotation-policy".to_owned();
        assert!(!changed_policy.is_commitment_valid());
    }

    #[derive(Deserialize)]
    struct GroundedParticipationFixtureSetV1 {
        schema_version: String,
        cases: Vec<GroundedParticipationFixtureCaseV1>,
    }

    #[derive(Deserialize)]
    struct GroundedParticipationFixtureCaseV1 {
        case_id: String,
        expected_disposition: String,
        evidence: GroundedMusicalEvidenceBundleV1,
        decision: GroundedMusicalDecisionReceiptV1,
        policy: GroundedParticipationPolicyV1,
    }

    #[test]
    fn checked_in_evidence_to_action_cases_match_the_authorization_validator() {
        let fixtures: GroundedParticipationFixtureSetV1 = serde_json::from_str(include_str!(
            "../../../../benchmarks/mel/grounded-participation/grounded-participation-cases-v1.json"
        ))
        .expect("grounded participation fixture set must parse");
        assert_eq!(
            fixtures.schema_version,
            "mel-grounded-participation-cases-v1"
        );
        assert_eq!(fixtures.cases.len(), 4);

        for case in fixtures.cases {
            let issues = validate_grounded_decision_v1(
                &case.evidence,
                &case.decision,
                &case.policy,
            );
            match case.expected_disposition.as_str() {
                "accept" => assert!(
                    issues.is_empty(),
                    "{} unexpectedly rejected: {issues:?}",
                    case.case_id
                ),
                "reject_veto" => assert!(
                    issues.contains(&GroundedParticipationIssueV1::VetoBlocksActiveIntervention),
                    "{} must reject a vetoed intervention: {issues:?}",
                    case.case_id
                ),
                other => panic!("{} has unsupported expected disposition {other}", case.case_id),
            }
        }
    }

    #[test]
    fn grounded_invitation_authorizes_a_bounded_response() {
        let evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy()).is_empty());
    }

    #[test]
    fn wait_evidence_requirement_flag_changes_the_policy_as_declared() {
        let mut evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Wait,
            Vec::new(),
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::DecisionEvidenceSetEmpty));

        let permissive = GroundedParticipationPolicyV1 {
            require_grounded_claims_for_wait: false,
            ..policy()
        };
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &permissive).is_empty());
    }

    #[test]
    fn retained_floor_grounding_authorizes_wait_without_a_sound_effect() {
        let evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Wait,
            vec!["retains-floor-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy()).is_empty());
    }

    #[test]
    fn veto_blocks_a_grounded_invitation_response() {
        let evidence = bundle();
        let mut receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        receipt.collaborator_veto_active = true;
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::VetoBlocksActiveIntervention));
    }

    #[test]
    fn ambiguous_grounding_cannot_authorize_a_response() {
        let mut evidence = bundle();
        evidence.claims[0].status = GroundedClaimStatusV1::Ambiguous;
        evidence.claims[0].reason = Some("two plausible event interpretations".to_owned());
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        let issues = validate_grounded_decision_v1(&evidence, &receipt, &policy());
        assert!(issues.iter().any(|issue| matches!(
            issue,
            GroundedParticipationIssueV1::DecisionClaimNotGrounded { .. }
        )));
        assert!(issues.contains(&GroundedParticipationIssueV1::RequiredEvidenceKindMissing {
            action: GroundedMusicalActionV1::Respond
        }));
    }

    #[test]
    fn ask_can_cite_ambiguous_grounding_without_authorizing_an_effect() {
        let mut evidence = bundle();
        evidence.claims[0].status = GroundedClaimStatusV1::Ambiguous;
        evidence.claims[0].reason = Some("two plausible handoff interpretations".to_owned());
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Ask,
            vec!["invitation-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy()).is_empty());
    }

    #[test]
    fn abstain_can_cite_unavailable_grounding_as_a_reason_to_withhold() {
        let mut evidence = bundle();
        evidence.claims[0].kind = GroundedMusicalClaimKindV1::StreamUnavailable;
        evidence.claims[0].status = GroundedClaimStatusV1::Grounded;
        evidence.claims[0].reason = None;
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Abstain,
            vec!["invitation-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy()).is_empty());
    }

    #[test]
    fn grounded_stream_unavailability_cannot_be_relabelled_as_wait() {
        let mut evidence = bundle();
        evidence.claims.push(claim(
            "stream-unavailable-001",
            GroundedMusicalClaimKindV1::StreamUnavailable,
            GroundedClaimStatusV1::Grounded,
            9_600,
        ));
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Wait,
            vec!["retains-floor-001"],
            None,
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::StreamUnavailableRequiresAbstain));
    }

    #[test]
    fn full_evidence_bundle_commitment_detects_metadata_tampering() {
        let evidence = bundle();
        let mut changed = evidence.clone();
        changed.permitted_prefix_end_frame -= 1;
        let receipt = decision(
            &changed,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        assert!(validate_grounded_decision_v1(&changed, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::InvalidClaimSetDigest));
    }

    #[test]
    fn effect_window_duration_is_bounded_in_runner_clock_domain() {
        let evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            Some(GroundedParticipationEffectWindowV1 {
                not_before_runner_time_ns: 10_000_120,
                expires_at_runner_time_ns: 10_000_120
                    + policy().maximum_effect_window_ns
                    + 1,
                authorization_id: "auth-001".to_owned(),
            }),
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::EffectWindowDurationOutsidePolicy));
    }

    #[test]
    fn decision_effect_clock_origin_must_match_evidence_origin() {
        let evidence = bundle();
        let mut receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        receipt.runner_origin_id = "different-origin".to_owned();
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::RunnerClockMismatch));
    }

    #[test]
    fn mismatched_audio_artifact_fails_closed() {
        let evidence = bundle();
        let mut receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        receipt.artifact.audio_sha256 = "c".repeat(64);
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::ArtifactMismatch {
                field: "audio_artifact_clock".to_owned()
            }));
    }

    #[test]
    fn decision_cannot_cite_evidence_from_beyond_its_input_prefix() {
        let mut evidence = bundle();
        evidence.permitted_prefix_end_frame = 8_000;
        evidence.claims[0].source_interval.end_frame = 9_600;
        evidence.claims[0].observed_through_frame = 9_600;
        evidence.claim_set_sha256 = evidence.expected_claim_set_sha256().unwrap();
        evidence.evidence_bundle_sha256 = evidence.expected_evidence_bundle_sha256().unwrap();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        let issues = validate_grounded_decision_v1(&evidence, &receipt, &policy());
        assert!(issues.contains(&GroundedParticipationIssueV1::ClaimBeyondObservedPrefix {
            claim_index: 0
        }));
    }

    #[test]
    fn expired_grounding_claim_cannot_authorize_an_action() {
        let evidence = bundle();
        let mut receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        let strict_policy = GroundedParticipationPolicyV1 {
            maximum_evidence_age_ns: 5,
            ..policy()
        };
        receipt.decision_at_ns = 120;
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &strict_policy)
            .contains(&GroundedParticipationIssueV1::DecisionEvidenceTooOld {
                claim_id: "invitation-001".to_owned()
            }));
    }

    #[test]
    fn active_intervention_requires_future_effect_window() {
        let evidence = bundle();
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            Some(GroundedParticipationEffectWindowV1 {
                not_before_runner_time_ns: 121,
                expires_at_runner_time_ns: 10_000_121,
                authorization_id: "auth-001".to_owned(),
            }),
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::EffectWindowBeforeDecision));
    }

    #[test]
    fn evidence_claim_set_tampering_fails_closed() {
        let mut evidence = bundle();
        evidence.claims[0].kind = GroundedMusicalClaimKindV1::RetainedFloor;
        let receipt = decision(
            &evidence,
            GroundedMusicalActionV1::Respond,
            vec!["invitation-001"],
            effect(),
        );
        assert!(validate_grounded_decision_v1(&evidence, &receipt, &policy())
            .contains(&GroundedParticipationIssueV1::InvalidClaimSetDigest));
    }
}

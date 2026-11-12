// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Locality-constrained repair proposal selection.
//!
//! This module connects existing frozen edit contracts, EVAL-017 event
//! influence, EVAL-018 pairwise interaction reports, and EVAL-019 probe plans.
//! It ranks bounded symbolic repair candidates without executing a renderer,
//! mutating a Score, or authorizing an edit.
//!
//! Trust boundaries:
//! - A symbolic candidate is admitted only after the existing frozen-contract
//!   and symbolic-locality verifier accepts its exact source/target Scores.
//! - EVAL-017 and EVAL-018 summaries are constructed only by replaying their
//!   existing report verifiers against the supplied exact evidence inputs.
//! - EVAL-019 plans are replay-verified before their selected pairs or render
//!   cost are consumed.
//! - Missing event influence or pairwise evidence is counted as unknown. It is
//!   never converted to zero risk or a negative interaction.
//! - A selected proposal is NOT an accepted edit. Final acceptance still
//!   requires rendering the candidate and independently verifying the
//!   candidate's exact post-render preservation contract.

use crate::counterfactual_influence::{
    CounterfactualInfluenceInput, CounterfactualInfluenceReport, InfluenceDisposition,
    VerifiedRepeatabilityCalibration, verify_counterfactual_influence,
};
use crate::counterfactual_interaction::{
    PairwiseInteractionInput, PairwiseInteractionReport, verify_pairwise_interaction,
};
use crate::edit_contract::{EditAttribute, ObligationScope, SupportedEditOperation, FrozenEditContract};
use crate::interaction_probe_plan::{
    EventInteractionSupport, InteractionProbeConfig, InteractionProbePlan,
    verify_interaction_probe_plan,
};
use crate::symbolic_audio_binding::{
    AudioPreservationProfile, RendererManifest, audio_preservation_profile_sha256,
    renderer_manifest_sha256, validate_frozen_edit_contract_for_rendering,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::{Score, edit_locality::EditLocalityEvidence};

pub const LOCALITY_REPAIR_PLANNER_VERSION: &str =
    "mel-mr-eval-022-locality-repair-planner-v0.1";
const MAX_REPAIR_CANDIDATES: usize = 256;
const MAX_PROTECTED_EVENTS: usize = 4096;
const MAX_EVIDENCE_ROWS: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPlannerRequest {
    pub request_id: String,
    pub source_score_sha256: String,
    pub instruction_sha256: String,
    pub renderer_manifest_sha256: String,
    pub preservation_profile_sha256: String,
    /// EVAL-017/018 process-level renderer identity and normalized config.
    /// The caller resolves this pair against the renderer manifest in its
    /// trusted adapter before constructing the request.
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    /// EVAL-019 renderer profile identity used by the sparse probe plan.
    pub probe_renderer_profile_sha256: String,
    /// Source event indices that must remain unchanged by the symbolic patch.
    pub protected_event_indices: Vec<usize>,
    /// Hard upper bound for the reusable baseline + single-removal + pair
    /// factorial render campaign estimated by EVAL-019.
    pub render_budget: u128,
    /// If true, a proposal with any unmeasured changed↔protected pair is
    /// retained as NeedsProbes and cannot enter the selectable frontier.
    pub require_complete_protected_pair_evidence: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RepairPlannerError {
    InvalidRequest(&'static str),
    InvalidSha256(&'static str),
    InvalidIdentifier(&'static str),
    TooManyCandidates,
    TooManyProtectedEvents,
    TooManyEvidenceRows,
    DuplicateCandidateId,
    DuplicateProtectedEvent,
    InvalidEventId,
    UnsupportedEditOperation,
    SymbolicEvidenceRejected,
    SourceScoreMismatch,
    InstructionMismatch,
    RendererManifestMismatch,
    RendererEvidenceMismatch,
    ProbePlanMismatch,
    InfluenceReportRejected,
    InteractionReportRejected,
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for RepairPlannerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(reason) => write!(f, "invalid repair-planner request: {reason}"),
            Self::InvalidSha256(field) => write!(f, "invalid SHA-256 field: {field}"),
            Self::InvalidIdentifier(field) => write!(f, "invalid identifier: {field}"),
            Self::TooManyCandidates => write!(f, "repair candidate count exceeds the configured limit"),
            Self::TooManyProtectedEvents => write!(f, "protected event count exceeds the configured limit"),
            Self::TooManyEvidenceRows => write!(f, "evidence row count exceeds the configured limit"),
            Self::DuplicateCandidateId => write!(f, "duplicate repair candidate ID"),
            Self::DuplicateProtectedEvent => write!(f, "duplicate protected event index"),
            Self::InvalidEventId => write!(f, "invalid canonical score event ID"),
            Self::UnsupportedEditOperation => write!(f, "edit operation is outside the current qualified subset"),
            Self::SymbolicEvidenceRejected => write!(f, "frozen edit contract or symbolic locality evidence did not qualify"),
            Self::SourceScoreMismatch => write!(f, "candidate does not share the request's exact source Score"),
            Self::InstructionMismatch => write!(f, "candidate does not share the request's exact instruction"),
            Self::RendererManifestMismatch => write!(f, "candidate renderer manifest differs from the request"),
            Self::RendererEvidenceMismatch => write!(f, "counterfactual evidence does not match the requested renderer context"),
            Self::ProbePlanMismatch => write!(f, "sparse probe plan does not replay for the request context"),
            Self::InfluenceReportRejected => write!(f, "EVAL-017 influence report failed independent replay"),
            Self::InteractionReportRejected => write!(f, "EVAL-018 interaction report failed independent replay"),
            Self::SerializationFailure => write!(f, "could not serialize repair plan"),
            Self::InvalidReceipt => write!(f, "repair-plan receipt digest does not verify"),
            Self::ReplayMismatch => write!(f, "repair plan differs from replayed candidate selection"),
        }
    }
}

impl std::error::Error for RepairPlannerError {}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), RepairPlannerError> {
    if !valid_sha256(value) {
        return Err(RepairPlannerError::InvalidSha256(field));
    }
    Ok(())
}

fn valid_identifier(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= maximum
        && !value.chars().any(char::is_control)
}

fn parse_event_index(value: &str) -> Result<usize, RepairPlannerError> {
    let suffix = value
        .strip_prefix("score_event_index:")
        .ok_or(RepairPlannerError::InvalidEventId)?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(RepairPlannerError::InvalidEventId);
    }
    suffix.parse().map_err(|_| RepairPlannerError::InvalidEventId)
}

fn canonical_pair(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Exact source/target input for symbolic preflight. This step never renders.
pub struct SymbolicRepairCandidateInput<'a> {
    pub candidate_id: &'a str,
    pub instruction_sha256: &'a str,
    pub contract: &'a FrozenEditContract,
    pub source_score: &'a Score,
    pub target_score: &'a Score,
    pub symbolic_evidence: &'a EditLocalityEvidence,
    pub renderer_manifest: &'a RendererManifest,
    pub preservation_profile: &'a AudioPreservationProfile,
}

/// Opaque result of replay-validating the current supported symbolic operation
/// against the exact Score pair and frozen renderer/preservation profiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualifiedSymbolicRepairCandidate {
    candidate_id: String,
    source_score_sha256: String,
    target_score_sha256: String,
    instruction_sha256: String,
    contract_sha256: String,
    renderer_manifest_sha256: String,
    preservation_profile_sha256: String,
    changed_event_indices: Vec<usize>,
    pitch_magnitude_semitones: u64,
}

impl QualifiedSymbolicRepairCandidate {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn changed_event_indices(&self) -> &[usize] {
        &self.changed_event_indices
    }

    pub fn target_score_sha256(&self) -> &str {
        &self.target_score_sha256
    }
}

/// Validate a candidate's frozen contract and symbolic-locality receipt using
/// the existing EVAL-010/EVAL-012 verification path. The current supported
/// contract subset is deliberately limited to PitchAt and Transpose.
pub fn qualify_symbolic_repair_candidate(
    input: SymbolicRepairCandidateInput<'_>,
) -> Result<QualifiedSymbolicRepairCandidate, RepairPlannerError> {
    if !valid_identifier(input.candidate_id, 256) {
        return Err(RepairPlannerError::InvalidIdentifier("candidate_id"));
    }
    validate_sha256("instruction_sha256", input.instruction_sha256)?;
    let operation = input
        .contract
        .validate_supported_operation()
        .map_err(|_| RepairPlannerError::SymbolicEvidenceRejected)?;
    validate_frozen_edit_contract_for_rendering(
        input.contract,
        input.instruction_sha256,
        input.source_score,
        input.target_score,
        input.symbolic_evidence,
        input.renderer_manifest,
        input.preservation_profile,
    )
    .map_err(|_| RepairPlannerError::SymbolicEvidenceRejected)?;

    if input.source_score.events().len() != input.target_score.events().len()
        || input.source_score.events().is_empty()
    {
        return Err(RepairPlannerError::SymbolicEvidenceRejected);
    }

    let source_events = input.source_score.events();
    let target_events = input.target_score.events();
    let mut changed_event_indices = Vec::new();
    let mut pitch_magnitude_semitones = 0u64;
    for (index, (source, target)) in source_events.iter().zip(target_events.iter()).enumerate() {
        let source_pitch = source.pitch.midi();
        let target_pitch = target.pitch.midi();
        if source_pitch != target_pitch {
            changed_event_indices.push(index);
            pitch_magnitude_semitones = pitch_magnitude_semitones
                .saturating_add(i64::from(target_pitch) .abs_diff(i64::from(source_pitch)) as u64);
        }
    }
    if changed_event_indices.is_empty() {
        return Err(RepairPlannerError::SymbolicEvidenceRejected);
    }

    match operation {
        SupportedEditOperation::PitchAt => {
            let pitch_obligation = input
                .contract
                .obligations
                .iter()
                .find(|obligation| obligation.attribute == EditAttribute::Pitch)
                .ok_or(RepairPlannerError::SymbolicEvidenceRejected)?;
            let ObligationScope::Event { event_id } = &pitch_obligation.scope else {
                return Err(RepairPlannerError::SymbolicEvidenceRejected);
            };
            let target_index = parse_event_index(event_id)?;
            if changed_event_indices.as_slice() != [target_index] {
                return Err(RepairPlannerError::SymbolicEvidenceRejected);
            }
        }
        SupportedEditOperation::Transpose => {
            if changed_event_indices.len() != source_events.len() {
                return Err(RepairPlannerError::SymbolicEvidenceRejected);
            }
        }
    }

    let manifest_sha256 = renderer_manifest_sha256(input.renderer_manifest)
        .map_err(|_| RepairPlannerError::RendererManifestMismatch)?;
    let profile_sha256 = audio_preservation_profile_sha256(input.preservation_profile)
        .map_err(|_| RepairPlannerError::SymbolicEvidenceRejected)?;

    Ok(QualifiedSymbolicRepairCandidate {
        candidate_id: input.candidate_id.to_owned(),
        source_score_sha256: input.contract.source_score_sha256.clone(),
        target_score_sha256: input.contract.target_score_sha256.clone(),
        instruction_sha256: input.contract.instruction_sha256.clone(),
        contract_sha256: input.contract.contract_sha256.clone(),
        renderer_manifest_sha256: manifest_sha256,
        preservation_profile_sha256: profile_sha256,
        changed_event_indices,
        pitch_magnitude_semitones,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedEventInfluence {
    event_index: usize,
    source_score_sha256: String,
    renderer_identity_sha256: String,
    renderer_config_sha256: String,
    report_receipt_sha256: String,
    support_frames: u64,
    support_interval_count: usize,
    disposition: InfluenceDisposition,
}

/// Create an influence summary only after EVAL-017 has replayed the full report
/// against the original paired-render inputs and repeatability calibration.
pub fn verify_event_influence(
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
    calibration: &VerifiedRepeatabilityCalibration,
) -> Result<VerifiedEventInfluence, RepairPlannerError> {
    verify_counterfactual_influence(report, input, calibration)
        .map_err(|_| RepairPlannerError::InfluenceReportRejected)?;
    let event_index = parse_event_index(&report.source_event_id)?;
    validate_sha256("influence.source_score_sha256", &report.source_score_sha256)?;
    validate_sha256("influence.renderer_identity_sha256", &report.renderer_identity_sha256)?;
    validate_sha256("influence.renderer_config_sha256", &report.renderer_config_sha256)?;
    validate_sha256("influence.receipt_sha256", &report.receipt_sha256)?;
    let support_frames = report
        .measured_influence_intervals
        .iter()
        .map(|interval| interval.end_frame.saturating_sub(interval.start_frame) as u64)
        .fold(0u64, u64::saturating_add);
    Ok(VerifiedEventInfluence {
        event_index,
        source_score_sha256: report.source_score_sha256.clone(),
        renderer_identity_sha256: report.renderer_identity_sha256.clone(),
        renderer_config_sha256: report.renderer_config_sha256.clone(),
        report_receipt_sha256: report.receipt_sha256.clone(),
        support_frames,
        support_interval_count: report.measured_influence_intervals.len(),
        disposition: report.disposition,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPairwiseInteraction {
    event_a: usize,
    event_b: usize,
    source_score_sha256: String,
    renderer_identity_sha256: String,
    renderer_config_sha256: String,
    report_receipt_sha256: String,
    max_pcm_interaction_rms: f64,
    max_pcm_interaction_peak: f64,
    max_spectral_interaction_rms: f64,
    affected_windows: usize,
}

/// Create a pairwise-risk summary only after the exact four-coalition report
/// replays through EVAL-018's opaque renderer/audio bindings.
pub fn verify_pairwise_interaction_evidence(
    report: &PairwiseInteractionReport,
    input: &PairwiseInteractionInput<'_, '_>,
) -> Result<VerifiedPairwiseInteraction, RepairPlannerError> {
    verify_pairwise_interaction(report, input)
        .map_err(|_| RepairPlannerError::InteractionReportRejected)?;
    let event_a = parse_event_index(&report.source_event_a)?;
    let event_b = parse_event_index(&report.source_event_b)?;
    if event_a == event_b {
        return Err(RepairPlannerError::InvalidEventId);
    }
    validate_sha256("interaction.source_score_sha256", &report.source_score_sha256)?;
    validate_sha256("interaction.renderer_identity_sha256", &report.renderer_identity_sha256)?;
    validate_sha256("interaction.renderer_config_sha256", &report.renderer_config_sha256)?;
    validate_sha256("interaction.receipt_sha256", &report.receipt_sha256)?;

    let mut max_pcm_interaction_rms = 0.0f64;
    let mut max_pcm_interaction_peak = 0.0f64;
    let mut max_spectral_interaction_rms = 0.0f64;
    let mut affected_windows = 0usize;
    for window in &report.windows {
        if !window.pcm_interaction_rms.is_finite()
            || !window.pcm_interaction_peak.is_finite()
            || !window.spectral_magnitude_interaction_rms.is_finite()
            || window.pcm_interaction_rms < 0.0
            || window.pcm_interaction_peak < 0.0
            || window.spectral_magnitude_interaction_rms < 0.0
        {
            return Err(RepairPlannerError::InteractionReportRejected);
        }
        max_pcm_interaction_rms = max_pcm_interaction_rms.max(window.pcm_interaction_rms);
        max_pcm_interaction_peak = max_pcm_interaction_peak.max(window.pcm_interaction_peak);
        max_spectral_interaction_rms =
            max_spectral_interaction_rms.max(window.spectral_magnitude_interaction_rms);
        if window.pcm_interaction_rms > 0.0
            || window.spectral_magnitude_interaction_rms > 0.0
        {
            affected_windows = affected_windows.saturating_add(1);
        }
    }
    Ok(VerifiedPairwiseInteraction {
        event_a: event_a.min(event_b),
        event_b: event_a.max(event_b),
        source_score_sha256: report.source_score_sha256.clone(),
        renderer_identity_sha256: report.renderer_identity_sha256.clone(),
        renderer_config_sha256: report.renderer_config_sha256.clone(),
        report_receipt_sha256: report.receipt_sha256.clone(),
        max_pcm_interaction_rms,
        max_pcm_interaction_peak,
        max_spectral_interaction_rms,
        affected_windows,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedInteractionProbePlan {
    source_score_sha256: String,
    renderer_profile_sha256: String,
    plan_receipt_sha256: String,
    event_count: usize,
    selected_pairs: Vec<(usize, usize)>,
    estimated_render_count: u128,
    all_possible_pair_count: u128,
    unprobed_pair_count: u128,
}

/// Replay EVAL-019's deterministic probe plan before consuming its selection
/// and cached-render budget estimate.
pub fn verify_interaction_probe_plan_evidence(
    plan: &InteractionProbePlan,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    config: &InteractionProbeConfig,
) -> Result<VerifiedInteractionProbePlan, RepairPlannerError> {
    verify_interaction_probe_plan(
        plan,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        config,
    )
    .map_err(|_| RepairPlannerError::ProbePlanMismatch)?;
    validate_sha256("probe.source_score_sha256", &plan.source_score_sha256)?;
    validate_sha256("probe.renderer_profile_sha256", &plan.renderer_profile_sha256)?;
    validate_sha256("probe.receipt_sha256", &plan.receipt_sha256)?;
    let mut selected_pairs = Vec::with_capacity(plan.selected_pair_count);
    for pair in plan
        .candidate_pairs_selected
        .iter()
        .chain(plan.audit_pairs.iter())
    {
        if pair.event_a >= plan.event_count || pair.event_b >= plan.event_count || pair.event_a == pair.event_b {
            return Err(RepairPlannerError::ProbePlanMismatch);
        }
        selected_pairs.push(canonical_pair(pair.event_a, pair.event_b));
    }
    selected_pairs.sort_unstable();
    selected_pairs.dedup();
    Ok(VerifiedInteractionProbePlan {
        source_score_sha256: plan.source_score_sha256.clone(),
        renderer_profile_sha256: plan.renderer_profile_sha256.clone(),
        plan_receipt_sha256: plan.receipt_sha256.clone(),
        event_count: plan.event_count,
        selected_pairs,
        estimated_render_count: plan.estimated_render_count_with_cached_single_removals,
        all_possible_pair_count: plan.all_possible_pair_count,
        unprobed_pair_count: plan.unprobed_pair_count,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepairProposalState {
    ReadyForCandidateRender,
    NeedsAdditionalProbes,
    MissingEventInfluence,
    NoDetectableTargetInfluence,
    OverRenderBudget,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairObjectiveVector {
    /// Number of symbolic source events that the candidate changes.
    pub changed_event_count: usize,
    /// Sum of absolute pitch deltas in semitones across changed events.
    pub pitch_magnitude_semitones: u64,
    /// Measured EVAL-017 influence support frames for changed source events.
    /// This is a planning exposure, not collateral ownership.
    pub measured_influence_support_frames: u64,
    pub measured_influence_interval_count: usize,
    /// Maximum observed EVAL-018 PCM interaction RMS against protected events.
    pub max_protected_pcm_interaction_rms: f64,
    /// Maximum observed EVAL-018 spectral-magnitude interaction RMS.
    pub max_protected_spectral_interaction_rms: f64,
    /// EVAL-019's reusable render estimate for the frozen probe plan.
    pub estimated_probe_render_count: u128,
    /// Protected changed-event pairs not yet backed by a replayed EVAL-018 report.
    /// Unknown pairs are never assigned a zero interaction value.
    pub unmeasured_protected_pairs: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairCandidateProposal {
    pub candidate_id: String,
    pub target_score_sha256: String,
    pub contract_sha256: String,
    pub state: RepairProposalState,
    pub objectives: RepairObjectiveVector,
    pub measured_influence_receipts: Vec<String>,
    pub measured_interaction_receipts: Vec<String>,
    pub missing_influence_event_indices: Vec<usize>,
    /// Changed events whose calibrated removal effect did not exceed the
    /// renderer-specific repeatability noise floor. This is not proof of
    /// inaudibility or zero contribution.
    pub no_detectable_target_influence_event_indices: Vec<usize>,
    pub unmeasured_protected_pairs: Vec<(usize, usize)>,
    /// Pairs selected by EVAL-019 but still awaiting an EVAL-018 render/report.
    pub selected_but_unmeasured_protected_pairs: Vec<(usize, usize)>,
    /// Unmeasured pairs not selected by the current EVAL-019 plan.
    pub unselected_protected_pairs: Vec<(usize, usize)>,
    pub selected_probe_plan_sha256: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepairPlannerConfig {
    /// Maximum proposals admitted to the deterministic comparison.
    pub maximum_candidates: usize,
    /// Maximum number of unmeasured target↔protected pairs tolerated before
    /// a proposal is marked NeedsAdditionalProbes.
    pub maximum_unmeasured_protected_pairs: usize,
}

impl Default for RepairPlannerConfig {
    fn default() -> Self {
        Self {
            maximum_candidates: 64,
            maximum_unmeasured_protected_pairs: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalityRepairPlan {
    pub version: String,
    pub request: RepairPlannerRequest,
    pub config: RepairPlannerConfig,
    pub probe_plan_sha256: String,
    pub proposals: Vec<RepairCandidateProposal>,
    /// Candidate IDs that are not dominated across the full objective vector.
    pub pareto_frontier_candidate_ids: Vec<String>,
    /// Deterministic lexicographic pick from the Pareto frontier. This is a
    /// recommendation only; it is not an acceptance or mutation authorization.
    pub preferred_candidate_id: Option<String>,
    pub probe_all_possible_pair_count: u128,
    pub probe_unprobed_pair_count: u128,
    pub estimated_probe_render_count: u128,
    pub receipt_sha256: String,
}

fn validate_request(request: &RepairPlannerRequest) -> Result<(), RepairPlannerError> {
    for (field, value) in [
        ("source_score_sha256", request.source_score_sha256.as_str()),
        ("instruction_sha256", request.instruction_sha256.as_str()),
        ("renderer_manifest_sha256", request.renderer_manifest_sha256.as_str()),
        ("preservation_profile_sha256", request.preservation_profile_sha256.as_str()),
        ("renderer_identity_sha256", request.renderer_identity_sha256.as_str()),
        ("renderer_config_sha256", request.renderer_config_sha256.as_str()),
        ("probe_renderer_profile_sha256", request.probe_renderer_profile_sha256.as_str()),
    ] {
        validate_sha256(field, value)?;
    }
    if !valid_identifier(&request.request_id, 256) {
        return Err(RepairPlannerError::InvalidIdentifier("request_id"));
    }
    if request.protected_event_indices.len() > MAX_PROTECTED_EVENTS {
        return Err(RepairPlannerError::TooManyProtectedEvents);
    }
    let mut protected = request.protected_event_indices.clone();
    protected.sort_unstable();
    protected.dedup();
    if protected.len() != request.protected_event_indices.len() {
        return Err(RepairPlannerError::DuplicateProtectedEvent);
    }
    Ok(())
}

fn validate_plan_context(
    request: &RepairPlannerRequest,
    candidates: &[QualifiedSymbolicRepairCandidate],
    influences: &[VerifiedEventInfluence],
    interactions: &[VerifiedPairwiseInteraction],
    probe_plan: &VerifiedInteractionProbePlan,
) -> Result<(), RepairPlannerError> {
    validate_request(request)?;
    if candidates.len() > MAX_REPAIR_CANDIDATES {
        return Err(RepairPlannerError::TooManyCandidates);
    }
    if influences.len().saturating_add(interactions.len()) > MAX_EVIDENCE_ROWS {
        return Err(RepairPlannerError::TooManyEvidenceRows);
    }
    if probe_plan.source_score_sha256 != request.source_score_sha256
        || probe_plan.renderer_profile_sha256 != request.probe_renderer_profile_sha256
    {
        return Err(RepairPlannerError::ProbePlanMismatch);
    }
    if request
        .protected_event_indices
        .iter()
        .any(|index| *index >= probe_plan.event_count)
    {
        return Err(RepairPlannerError::InvalidRequest(
            "protected event index is outside the probe plan Score inventory",
        ));
    }
    let mut candidate_ids = std::collections::BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.candidate_id.clone()) {
            return Err(RepairPlannerError::DuplicateCandidateId);
        }
        if candidate.source_score_sha256 != request.source_score_sha256 {
            return Err(RepairPlannerError::SourceScoreMismatch);
        }
        if candidate.instruction_sha256 != request.instruction_sha256 {
            return Err(RepairPlannerError::InstructionMismatch);
        }
        if candidate.renderer_manifest_sha256 != request.renderer_manifest_sha256 {
            return Err(RepairPlannerError::RendererManifestMismatch);
        }
        if candidate.preservation_profile_sha256 != request.preservation_profile_sha256 {
            return Err(RepairPlannerError::SymbolicEvidenceRejected);
        }
    }
    for influence in influences {
        if influence.source_score_sha256 != request.source_score_sha256
            || influence.renderer_identity_sha256 != request.renderer_identity_sha256
            || influence.renderer_config_sha256 != request.renderer_config_sha256
        {
            return Err(RepairPlannerError::RendererEvidenceMismatch);
        }
    }
    for interaction in interactions {
        if interaction.source_score_sha256 != request.source_score_sha256
            || interaction.renderer_identity_sha256 != request.renderer_identity_sha256
            || interaction.renderer_config_sha256 != request.renderer_config_sha256
        {
            return Err(RepairPlannerError::RendererEvidenceMismatch);
        }
    }
    Ok(())
}

/// Compare symbolic candidates using verified source evidence.
///
/// The planner consumes measured reports as risk priors and reports unknown
/// pairs explicitly. It does not use an EVAL-017/018 estimate as a preservation
/// certificate; the selected target candidate must still be rendered and
/// checked with the independent post-render verifier.
pub fn build_locality_repair_plan(
    request: RepairPlannerRequest,
    config: RepairPlannerConfig,
    candidates: &[QualifiedSymbolicRepairCandidate],
    influences: &[VerifiedEventInfluence],
    interactions: &[VerifiedPairwiseInteraction],
    probe_plan: &VerifiedInteractionProbePlan,
) -> Result<LocalityRepairPlan, RepairPlannerError> {
    validate_request(&request)?;
    if config.maximum_candidates == 0
        || config.maximum_candidates > MAX_REPAIR_CANDIDATES
        || config.maximum_unmeasured_protected_pairs > MAX_EVIDENCE_ROWS
    {
        return Err(RepairPlannerError::InvalidRequest("repair planner limits are invalid"));
    }
    if candidates.len() > config.maximum_candidates {
        return Err(RepairPlannerError::TooManyCandidates);
    }
    validate_plan_context(&request, candidates, influences, interactions, probe_plan)?;

    let influence_by_event: std::collections::BTreeMap<usize, &VerifiedEventInfluence> =
        influences.iter().map(|evidence| (evidence.event_index, evidence)).collect();
    let interaction_by_pair: std::collections::BTreeMap<(usize, usize), &VerifiedPairwiseInteraction> =
        interactions
            .iter()
            .map(|evidence| ((evidence.event_a, evidence.event_b), evidence))
            .collect();
    let selected_probe_pairs: std::collections::BTreeSet<(usize, usize)> =
        probe_plan.selected_pairs.iter().copied().collect();
    let protected: std::collections::BTreeSet<usize> =
        request.protected_event_indices.iter().copied().collect();

    let mut proposals = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut reasons = Vec::new();
        let changed: std::collections::BTreeSet<usize> =
            candidate.changed_event_indices.iter().copied().collect();
        if changed.len() != candidate.changed_event_indices.len()
            || changed.iter().any(|index| *index >= probe_plan.event_count)
        {
            return Err(RepairPlannerError::InvalidRequest("candidate changed-event inventory is invalid"));
        }
        if changed.iter().any(|index| protected.contains(index)) {
            return Err(RepairPlannerError::SymbolicEvidenceRejected);
        }

        let mut missing_influence_event_indices = Vec::new();
        let mut no_detectable_target_influence_event_indices = Vec::new();
        let mut measured_influence_support_frames = 0u64;
        let mut measured_influence_interval_count = 0usize;
        let mut measured_influence_receipts = Vec::new();
        for event_index in &candidate.changed_event_indices {
            if let Some(evidence) = influence_by_event.get(event_index) {
                measured_influence_support_frames =
                    measured_influence_support_frames.saturating_add(evidence.support_frames);
                measured_influence_interval_count = measured_influence_interval_count
                    .saturating_add(evidence.support_interval_count);
                if evidence.disposition == InfluenceDisposition::NoEffectAboveNoiseFloor {
                    no_detectable_target_influence_event_indices.push(*event_index);
                }
                measured_influence_receipts.push(evidence.report_receipt_sha256.clone());
            } else {
                missing_influence_event_indices.push(*event_index);
            }
        }
        missing_influence_event_indices.sort_unstable();
        no_detectable_target_influence_event_indices.sort_unstable();
        measured_influence_receipts.sort();

        let mut relevant_pairs = std::collections::BTreeSet::new();
        for changed_event in &changed {
            for protected_event in &protected {
                if changed_event != protected_event {
                    relevant_pairs.insert(canonical_pair(*changed_event, *protected_event));
                }
            }
        }
        let mut unmeasured_protected_pairs = Vec::new();
        let mut selected_but_unmeasured_protected_pairs = Vec::new();
        let mut unselected_protected_pairs = Vec::new();
        let mut max_protected_pcm_interaction_rms = 0.0f64;
        let mut max_protected_spectral_interaction_rms = 0.0f64;
        let mut measured_interaction_receipts = Vec::new();
        for pair in relevant_pairs {
            if let Some(evidence) = interaction_by_pair.get(&pair) {
                max_protected_pcm_interaction_rms =
                    max_protected_pcm_interaction_rms.max(evidence.max_pcm_interaction_rms);
                max_protected_spectral_interaction_rms = max_protected_spectral_interaction_rms
                    .max(evidence.max_spectral_interaction_rms);
                measured_interaction_receipts.push(evidence.report_receipt_sha256.clone());
            } else {
                // A selected pair still awaits an actual four-coalition
                // EVAL-018 result. A non-selected pair remains unprobed.
                // Neither class is interpreted as non-interacting.
                if selected_probe_pairs.contains(&pair) {
                    selected_but_unmeasured_protected_pairs.push(pair);
                } else {
                    unselected_protected_pairs.push(pair);
                }
                unmeasured_protected_pairs.push(pair);
            }
        }
        measured_interaction_receipts.sort();

        let state = if probe_plan.estimated_render_count > request.render_budget {
            reasons.push("EVAL-019 estimated render count exceeds the hard request budget".to_owned());
            RepairProposalState::OverRenderBudget
        } else if !missing_influence_event_indices.is_empty() {
            reasons.push("one or more changed source events lack replay-verified EVAL-017 evidence".to_owned());
            RepairProposalState::MissingEventInfluence
        } else if !no_detectable_target_influence_event_indices.is_empty() {
            reasons.push(
                "one or more changed events have no influence detected above the calibrated repeatability floor; their target audibility is unresolved"
                    .to_owned(),
            );
            RepairProposalState::NoDetectableTargetInfluence
        } else if (request.require_complete_protected_pair_evidence
            && !unmeasured_protected_pairs.is_empty())
            || unmeasured_protected_pairs.len() > config.maximum_unmeasured_protected_pairs
        {
            reasons.push(
                "changed↔protected event pairs remain unmeasured; acquire selected probes and/or expand the EVAL-019 plan"
                    .to_owned(),
            );
            RepairProposalState::NeedsAdditionalProbes
        } else {
            reasons.push("symbolic contract and current evidence qualify this candidate for a new render; post-render acceptance remains separate".to_owned());
            RepairProposalState::ReadyForCandidateRender
        };

        proposals.push(RepairCandidateProposal {
            candidate_id: candidate.candidate_id.clone(),
            target_score_sha256: candidate.target_score_sha256.clone(),
            contract_sha256: candidate.contract_sha256.clone(),
            state,
            objectives: RepairObjectiveVector {
                changed_event_count: candidate.changed_event_indices.len(),
                pitch_magnitude_semitones: candidate.pitch_magnitude_semitones,
                measured_influence_support_frames,
                measured_influence_interval_count,
                max_protected_pcm_interaction_rms,
                max_protected_spectral_interaction_rms,
                estimated_probe_render_count: probe_plan.estimated_render_count,
                unmeasured_protected_pairs: unmeasured_protected_pairs.len(),
            },
            measured_influence_receipts,
            measured_interaction_receipts,
            missing_influence_event_indices,
            no_detectable_target_influence_event_indices,
            unmeasured_protected_pairs,
            selected_but_unmeasured_protected_pairs,
            unselected_protected_pairs,
            selected_probe_plan_sha256: probe_plan.plan_receipt_sha256.clone(),
            reasons,
        });
    }

    proposals.sort_by(|left, right| left.candidate_id.cmp(&right.candidate_id));
    let ready: Vec<&RepairCandidateProposal> = proposals
        .iter()
        .filter(|proposal| proposal.state == RepairProposalState::ReadyForCandidateRender)
        .collect();
    let mut frontier: Vec<&RepairCandidateProposal> = ready
        .iter()
        .copied()
        .filter(|candidate| {
            !ready.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && dominates(&other.objectives, &candidate.objectives)
            })
        })
        .collect();
    frontier.sort_by(|left, right| {
        left.objectives
            .measured_influence_support_frames
            .cmp(&right.objectives.measured_influence_support_frames)
            .then_with(|| left.objectives.changed_event_count.cmp(&right.objectives.changed_event_count))
            .then_with(|| left.objectives.pitch_magnitude_semitones.cmp(&right.objectives.pitch_magnitude_semitones))
            .then_with(|| left.objectives.estimated_probe_render_count.cmp(&right.objectives.estimated_probe_render_count))
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    let pareto_frontier_candidate_ids: Vec<String> =
        frontier.iter().map(|candidate| candidate.candidate_id.clone()).collect();
    let preferred_candidate_id = pareto_frontier_candidate_ids.first().cloned();

    let mut plan = LocalityRepairPlan {
        version: LOCALITY_REPAIR_PLANNER_VERSION.to_owned(),
        request,
        config,
        probe_plan_sha256: probe_plan.plan_receipt_sha256.clone(),
        probe_all_possible_pair_count: probe_plan.all_possible_pair_count,
        probe_unprobed_pair_count: probe_plan.unprobed_pair_count,
        estimated_probe_render_count: probe_plan.estimated_render_count,
        proposals,
        pareto_frontier_candidate_ids,
        preferred_candidate_id,
        receipt_sha256: String::new(),
    };
    plan.receipt_sha256 = plan_receipt_sha256(&plan)?;
    Ok(plan)
}

fn dominates(left: &RepairObjectiveVector, right: &RepairObjectiveVector) -> bool {
    let no_worse = left.changed_event_count <= right.changed_event_count
        && left.pitch_magnitude_semitones <= right.pitch_magnitude_semitones
        && left.measured_influence_support_frames <= right.measured_influence_support_frames
        && left.measured_influence_interval_count <= right.measured_influence_interval_count
        && left.max_protected_pcm_interaction_rms <= right.max_protected_pcm_interaction_rms
        && left.max_protected_spectral_interaction_rms <= right.max_protected_spectral_interaction_rms
        && left.estimated_probe_render_count <= right.estimated_probe_render_count
        && left.unmeasured_protected_pairs <= right.unmeasured_protected_pairs;
    let strictly_better = left.changed_event_count < right.changed_event_count
        || left.pitch_magnitude_semitones < right.pitch_magnitude_semitones
        || left.measured_influence_support_frames < right.measured_influence_support_frames
        || left.measured_influence_interval_count < right.measured_influence_interval_count
        || left.max_protected_pcm_interaction_rms < right.max_protected_pcm_interaction_rms
        || left.max_protected_spectral_interaction_rms < right.max_protected_spectral_interaction_rms
        || left.estimated_probe_render_count < right.estimated_probe_render_count
        || left.unmeasured_protected_pairs < right.unmeasured_protected_pairs;
    no_worse && strictly_better
}

fn plan_receipt_sha256(plan: &LocalityRepairPlan) -> Result<String, RepairPlannerError> {
    let mut unsealed = plan.clone();
    unsealed.receipt_sha256.clear();
    let bytes =
        serde_json::to_vec(&unsealed).map_err(|_| RepairPlannerError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-022-locality-repair-plan-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Verify the content digest and deterministically rebuild the plan from the
/// same opaque qualified candidates and replay-verified evidence summaries.
pub fn verify_locality_repair_plan(
    plan: &LocalityRepairPlan,
    candidates: &[QualifiedSymbolicRepairCandidate],
    influences: &[VerifiedEventInfluence],
    interactions: &[VerifiedPairwiseInteraction],
    probe_plan: &VerifiedInteractionProbePlan,
) -> Result<bool, RepairPlannerError> {
    if plan.receipt_sha256.is_empty()
        || plan_receipt_sha256(plan)? != plan.receipt_sha256
    {
        return Err(RepairPlannerError::InvalidReceipt);
    }
    let expected = build_locality_repair_plan(
        plan.request.clone(),
        plan.config.clone(),
        candidates,
        influences,
        interactions,
        probe_plan,
    )?;
    Ok(expected == *plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        format!("{:x}", Sha256::digest(label.as_bytes()))
    }

    fn symbolic_candidate(id: &str, changed: &[usize], magnitude: u64) -> QualifiedSymbolicRepairCandidate {
        QualifiedSymbolicRepairCandidate {
            candidate_id: id.to_owned(),
            source_score_sha256: digest("source"),
            target_score_sha256: digest(&format!("target-{id}")),
            instruction_sha256: digest("instruction"),
            contract_sha256: digest(&format!("contract-{id}")),
            renderer_manifest_sha256: digest("manifest"),
            preservation_profile_sha256: digest("preservation-profile"),
            changed_event_indices: changed.to_vec(),
            pitch_magnitude_semitones: magnitude,
        }
    }

    fn influence(event_index: usize, support_frames: u64) -> VerifiedEventInfluence {
        VerifiedEventInfluence {
            event_index,
            source_score_sha256: digest("source"),
            renderer_identity_sha256: digest("renderer"),
            renderer_config_sha256: digest("config"),
            report_receipt_sha256: digest(&format!("influence-{event_index}")),
            support_frames,
            support_interval_count: 1,
            disposition: InfluenceDisposition::EffectDetected,
        }
    }

    fn pair(a: usize, b: usize, rms: f64, spectral: f64) -> VerifiedPairwiseInteraction {
        let (event_a, event_b) = canonical_pair(a, b);
        VerifiedPairwiseInteraction {
            event_a,
            event_b,
            source_score_sha256: digest("source"),
            renderer_identity_sha256: digest("renderer"),
            renderer_config_sha256: digest("config"),
            report_receipt_sha256: digest(&format!("pair-{event_a}-{event_b}")),
            max_pcm_interaction_rms: rms,
            max_pcm_interaction_peak: rms * 2.0,
            max_spectral_interaction_rms: spectral,
            affected_windows: 2,
        }
    }

    fn probe_plan() -> VerifiedInteractionProbePlan {
        VerifiedInteractionProbePlan {
            source_score_sha256: digest("source"),
            renderer_profile_sha256: digest("probe-profile"),
            plan_receipt_sha256: digest("probe-plan"),
            event_count: 4,
            selected_pairs: vec![(0, 2), (1, 2), (1, 3)],
            estimated_render_count: 8,
            all_possible_pair_count: 6,
            unprobed_pair_count: 3,
        }
    }

    fn request() -> RepairPlannerRequest {
        RepairPlannerRequest {
            request_id: "repair-request-1".into(),
            source_score_sha256: digest("source"),
            instruction_sha256: digest("instruction"),
            renderer_manifest_sha256: digest("manifest"),
            preservation_profile_sha256: digest("preservation-profile"),
            renderer_identity_sha256: digest("renderer"),
            renderer_config_sha256: digest("config"),
            probe_renderer_profile_sha256: digest("probe-profile"),
            protected_event_indices: vec![2, 3],
            render_budget: 12,
            require_complete_protected_pair_evidence: true,
        }
    }

    #[test]
    fn unknown_event_influence_is_not_interpreted_as_zero() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &[],
            &[],
            &probe_plan(),
        ).unwrap();
        assert_eq!(result.proposals[0].state, RepairProposalState::MissingEventInfluence);
        assert_eq!(result.proposals[0].objectives.measured_influence_support_frames, 0);
        assert_eq!(result.proposals[0].missing_influence_event_indices, vec![0]);
        assert!(result.preferred_candidate_id.is_none());
    }

    #[test]
    fn no_effect_above_noise_floor_is_not_treated_as_zero_target_risk() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let mut no_effect = influence(0, 0);
        no_effect.disposition = InfluenceDisposition::NoEffectAboveNoiseFloor;
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &[no_effect],
            &[],
            &probe_plan(),
        )
        .unwrap();
        assert_eq!(
            result.proposals[0].state,
            RepairProposalState::NoDetectableTargetInfluence
        );
        assert_eq!(
            result.proposals[0].no_detectable_target_influence_event_indices,
            vec![0]
        );
        assert!(result.preferred_candidate_id.is_none());
    }

    #[test]
    fn unknown_protected_interactions_are_reported_as_probe_debt() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let influences = [influence(0, 100)];
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &influences,
            &[],
            &probe_plan(),
        ).unwrap();
        assert_eq!(result.proposals[0].state, RepairProposalState::NeedsAdditionalProbes);
        assert_eq!(result.proposals[0].unmeasured_protected_pairs, vec![(0, 2), (0, 3)]);
        assert_eq!(
            result.proposals[0].selected_but_unmeasured_protected_pairs,
            vec![(0, 2)]
        );
        assert_eq!(
            result.proposals[0].unselected_protected_pairs,
            vec![(0, 3)]
        );
        assert!(result.proposals[0].objectives.unmeasured_protected_pairs > 0);
    }

    #[test]
    fn selected_probe_does_not_count_as_a_measured_interaction() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let influences = [influence(0, 100)];
        // The plan selected (0,2), but no EVAL-018 report exists for it yet.
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig { maximum_candidates: 64, maximum_unmeasured_protected_pairs: 0 },
            &candidates,
            &influences,
            &[],
            &probe_plan(),
        ).unwrap();
        assert_eq!(result.proposals[0].state, RepairProposalState::NeedsAdditionalProbes);
        assert!(result.proposals[0].unmeasured_protected_pairs.contains(&(0, 2)));
    }

    #[test]
    fn complete_event_and_pair_evidence_makes_a_render_proposal() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let influences = [influence(0, 100)];
        let interactions = [pair(0, 2, 0.02, 0.01), pair(0, 3, 0.01, 0.005)];
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &influences,
            &interactions,
            &probe_plan(),
        ).unwrap();
        assert_eq!(result.proposals[0].state, RepairProposalState::ReadyForCandidateRender);
        assert_eq!(result.preferred_candidate_id.as_deref(), Some("c1"));
        assert_eq!(result.proposals[0].objectives.unmeasured_protected_pairs, 0);
        assert_eq!(result.probe_all_possible_pair_count, 6);
        assert_eq!(result.probe_unprobed_pair_count, 3);
        assert_eq!(result.estimated_probe_render_count, 8);
    }

    #[test]
    fn render_budget_is_a_hard_precondition_not_a_soft_penalty() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let influences = [influence(0, 100)];
        let interactions = [pair(0, 2, 0.02, 0.01), pair(0, 3, 0.01, 0.005)];
        let mut small_budget = request();
        small_budget.render_budget = 4;
        let result = build_locality_repair_plan(
            small_budget,
            RepairPlannerConfig::default(),
            &candidates,
            &influences,
            &interactions,
            &probe_plan(),
        ).unwrap();
        assert_eq!(result.proposals[0].state, RepairProposalState::OverRenderBudget);
        assert!(result.preferred_candidate_id.is_none());
    }

    #[test]
    fn pareto_frontier_preserves_tradeoffs_instead_of_collapsing_them() {
        let candidates = [
            symbolic_candidate("low-edit-high-spill", &[0], 1),
            symbolic_candidate("more-edit-low-spill", &[0, 1], 3),
            symbolic_candidate("dominated", &[0, 1], 4),
        ];
        let influences = [influence(0, 50), influence(1, 50)];
        let interactions = [
            pair(0, 2, 0.05, 0.04),
            pair(0, 3, 0.05, 0.04),
            pair(1, 2, 0.001, 0.001),
            pair(1, 3, 0.001, 0.001),
        ];
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &influences,
            &interactions,
            &probe_plan(),
        ).unwrap();
        assert!(result.pareto_frontier_candidate_ids.contains(&"low-edit-high-spill".to_owned()));
        assert!(result.pareto_frontier_candidate_ids.contains(&"more-edit-low-spill".to_owned()));
        assert!(!result.pareto_frontier_candidate_ids.contains(&"dominated".to_owned()));
        assert!(result.preferred_candidate_id.is_some());
    }

    #[test]
    fn protected_event_mutation_rejects_instead_of_becoming_a_candidate() {
        let candidates = [symbolic_candidate("bad", &[2], 1)];
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &[influence(2, 10)],
            &[],
            &probe_plan(),
        );
        assert_eq!(result.unwrap_err(), RepairPlannerError::SymbolicEvidenceRejected);
    }

    #[test]
    fn plan_replays_exactly_and_rejects_tampering() {
        let candidates = [symbolic_candidate("c1", &[0], 2)];
        let influences = [influence(0, 100)];
        let interactions = [pair(0, 2, 0.02, 0.01), pair(0, 3, 0.01, 0.005)];
        let probes = probe_plan();
        let result = build_locality_repair_plan(
            request(),
            RepairPlannerConfig::default(),
            &candidates,
            &influences,
            &interactions,
            &probes,
        ).unwrap();
        assert!(verify_locality_repair_plan(
            &result, &candidates, &influences, &interactions, &probes
        ).unwrap());
        let mut tampered = result.clone();
        tampered.proposals[0].objectives.measured_influence_support_frames += 1;
        assert_eq!(
            verify_locality_repair_plan(
                &tampered, &candidates, &influences, &interactions, &probes
            ).unwrap_err(),
            RepairPlannerError::InvalidReceipt
        );
    }
}

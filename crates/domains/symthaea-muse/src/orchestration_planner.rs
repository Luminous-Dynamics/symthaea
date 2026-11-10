// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, bounded, feasibility-first orchestration search.
//!
//! This v1 kernel consumes source-bound score-demand profiles and opaque,
//! materializer-issued capability receipts. It assigns one distinct qualified
//! instrument instance to each persistent score part. It does not infer part
//! identity, select aesthetic preferences, rewrite notes, split a part across
//! players, or treat role summaries as additional performers.
//!
//! The result is a feasible-candidate set (or explicit abstention/infeasibility
//! status), not a claim that the first candidate is artistically optimal.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use crate::instrument_capability_evidence::{
    CapabilityScopeV1, INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1,
    QualifiedInstrumentCapabilityV1, valid_sha256_hex_v1,
};
use crate::score_demand::{
    DemandIdentityBasis, SCORE_DEMAND_INPUT_DIGEST_SCHEMA, SCORE_DEMAND_PROFILE_SCHEMA,
    ScoreDemandProfile, VoiceDemand,
};

pub const ORCHESTRATION_PLANNER_SCHEMA_V1: &str = "muse-orchestration-planner-v1";
pub const ORCHESTRATION_CANDIDATE_DIGEST_SCHEMA_V1: &str =
    "muse-orchestration-candidate-sha256-v1";
pub const MAX_PLANNER_PARTS_V1: usize = 128;
pub const MAX_PLANNER_INSTRUMENTS_V1: usize = 128;
pub const MAX_PLANNER_SEARCH_NODES_V1: usize = 1_000_000;
pub const MAX_PLANNER_CANDIDATES_V1: usize = 256;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PartRequirementV1 {
    pub part_id: u16,
    /// Authored by the composition/brief layer. Never inferred from VoiceRole.
    pub musical_function: String,
    /// Optional exact performance-scope requirements.
    pub performer_level: Option<String>,
    pub technique: Option<String>,
    pub dynamic: Option<String>,
    /// When true, the chosen capability receipt must include an exact,
    /// qualified renderer/patch/asset scope.
    pub renderer_required: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerLimitsV1 {
    pub max_search_nodes: usize,
    pub max_candidates: usize,
}

impl Default for PlannerLimitsV1 {
    fn default() -> Self {
        Self {
            max_search_nodes: 100_000,
            max_candidates: 64,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrchestrationPlannerRequestV1 {
    pub schema: String,
    /// Caller-supplied expected digest from the trusted source-score boundary.
    pub expected_score_sha256: String,
    pub score_demand: ScoreDemandProfile,
    /// One authored function/requirement record for every persistent PartId.
    pub part_requirements: Vec<PartRequirementV1>,
    /// Constructed only by the instrument-capability materializer.
    pub instruments: Vec<QualifiedInstrumentCapabilityV1>,
    pub limits: PlannerLimitsV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PlannerDispositionV1 {
    /// Search completed and all feasible candidates were enumerated.
    FeasibleComplete,
    /// Valid candidates were found, but the candidate-count cap truncated search.
    CandidateLimitReached,
    /// Search completed with no feasible one-instrument-per-part assignment.
    InfeasibleComplete,
    /// The planner refused to infer missing identities or unsupported inputs.
    Abstained,
    /// Input contract or provenance receipt failed validation.
    InvalidRequest,
    /// Search-node budget was exceeded; partial candidates are discarded.
    BudgetExhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PlannerIssueCodeV1 {
    InvalidSchema,
    InvalidExpectedScoreDigest,
    ScoreDigestMismatch,
    InvalidScoreDemandSchema,
    InvalidScoreDemandDigestSchema,
    IncompletePartIdentity,
    AssignedPartFractionIncomplete,
    NoStablePartDemands,
    InvalidPartDemand,
    DuplicatePartDemand,
    MissingPartRequirement,
    DuplicatePartRequirement,
    UnknownPartRequirement,
    InvalidMusicalFunction,
    InvalidPlannerLimits,
    TooManyParts,
    TooManyInstrumentInstances,
    InvalidCapabilityReceipt,
    DuplicateInstrumentInstanceId,
    NoQualifiedCandidateForPart,
    SearchBudgetExceeded,
    CandidateSerializationFailed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlannerIssueV1 {
    pub code: PlannerIssueCodeV1,
    pub part_id: Option<u16>,
    pub instrument_id: Option<String>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PlannedPartAssignmentV1 {
    pub part_id: u16,
    pub instrument_instance_id: String,
    pub musical_function: String,
    pub instrument_kind: symthaea_music_theory::orchestration::InstrumentKindV1,
    pub sounding_lowest_midi: u8,
    pub sounding_highest_midi: u8,
    pub max_polyphony: u8,
    pub capability_profile_sha256: String,
    pub performance_scope: CapabilityScopeV1,
    pub renderer_scope: Option<CapabilityScopeV1>,
    pub capability_claim_ids: Vec<String>,
    pub evidence_ids: Vec<String>,
    pub qualification_record_ids: Vec<String>,
    pub qualification_record_sha256s: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OrchestrationCandidateV1 {
    pub schema: String,
    pub candidate_digest_schema: String,
    pub score_sha256: String,
    pub candidate_sha256: String,
    /// Sorted by persistent PartId, not by search traversal order.
    pub assignments: Vec<PlannedPartAssignmentV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct OrchestrationPlannerReportV1 {
    pub schema: String,
    pub disposition: PlannerDispositionV1,
    pub score_sha256: Option<String>,
    pub part_count: usize,
    pub qualified_instrument_count: usize,
    pub search_nodes: usize,
    pub candidates: Vec<OrchestrationCandidateV1>,
    pub issues: Vec<PlannerIssueV1>,
}

impl OrchestrationPlannerReportV1 {
    fn new() -> Self {
        Self {
            schema: ORCHESTRATION_PLANNER_SCHEMA_V1.into(),
            disposition: PlannerDispositionV1::InvalidRequest,
            score_sha256: None,
            part_count: 0,
            qualified_instrument_count: 0,
            search_nodes: 0,
            candidates: Vec::new(),
            issues: Vec::new(),
        }
    }

    fn push(
        &mut self,
        code: PlannerIssueCodeV1,
        part_id: Option<u16>,
        instrument_id: Option<&str>,
        detail: impl Into<String>,
    ) {
        self.issues.push(PlannerIssueV1 {
            code,
            part_id,
            instrument_id: instrument_id.map(str::to_owned),
            detail: detail.into(),
        });
    }
}

#[derive(Debug)]
struct SearchState {
    nodes: usize,
    candidates: Vec<OrchestrationCandidateV1>,
    budget_exhausted: bool,
    candidate_limit_reached: bool,
    serialization_failed: bool,
}

fn valid_capability_receipt(
    instrument: &QualifiedInstrumentCapabilityV1,
) -> bool {
    let capability = &instrument.capability;
    let performance_scope = &instrument.performance_scope;
    !capability.id.trim().is_empty()
        && capability.sounding_range.lowest_midi <= capability.sounding_range.highest_midi
        && capability.sounding_range.highest_midi <= 127
        && capability.max_polyphony > 0
        && performance_scope
            .instrument_model
            .as_ref()
            .is_some_and(|model| !model.trim().is_empty())
        && valid_sha256_hex_v1(&instrument.profile_sha256)
        && instrument.profile_digest_schema == INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1
        && instrument.claim_ids.len() >= 2
        && instrument.claim_ids.len() == instrument.evidence_ids.len()
        && instrument.claim_ids.len() == instrument.qualification_record_ids.len()
        && instrument.claim_ids.len() == instrument.qualification_record_sha256s.len()
        && instrument
            .claim_ids
            .iter()
            .all(|value| !value.trim().is_empty())
        && instrument
            .evidence_ids
            .iter()
            .all(|value| !value.trim().is_empty())
        && instrument
            .qualification_record_ids
            .iter()
            .all(|value| !value.trim().is_empty())
        && instrument
            .qualification_record_sha256s
            .iter()
            .all(|value| valid_sha256_hex_v1(value))
}

fn part_demand_is_valid(demand: &VoiceDemand) -> bool {
    demand.part_id.is_some_and(|part_id| part_id != u16::MAX)
        && demand.identity_basis == DemandIdentityBasis::StablePartId
        && demand.note_count > 0
        && demand.min_midi <= demand.max_midi
        && demand.max_midi <= 127
        && demand.pitch_span_semitones == demand.max_midi - demand.min_midi
        && !demand.roles.is_empty()
        && demand.distinct_onset_count > 0
        && demand.distinct_onset_count <= demand.note_count
        && demand.peak_simultaneous_notes > 0
        && demand.peak_simultaneous_notes <= demand.note_count
        && demand.mean_midi.is_finite()
        && demand.mean_midi >= demand.min_midi as f64
        && demand.mean_midi <= demand.max_midi as f64
        && demand.mean_note_duration_beats.is_finite()
        && demand.mean_note_duration_beats > 0.0
        && demand.first_onset_beat.is_finite()
        && demand.last_note_end_beat.is_finite()
        && demand.last_note_end_beat > demand.first_onset_beat
        && demand.active_span_beats.is_finite()
        && demand.active_span_beats > 0.0
        && demand.notes_per_score_beat.is_finite()
        && demand.notes_per_score_beat >= 0.0
        && demand.notes_per_bar.is_none_or(|value| value.is_finite() && value >= 0.0)
}

fn scope_matches_requirement(
    instrument: &QualifiedInstrumentCapabilityV1,
    requirement: &PartRequirementV1,
) -> bool {
    let scope = &instrument.performance_scope;
    let exact_match = |required: &Option<String>, actual: &Option<String>| {
        required.as_ref().is_none_or(|value| actual.as_ref() == Some(value))
    };

    exact_match(&requirement.performer_level, &scope.performer_level)
        && exact_match(&requirement.technique, &scope.technique)
        && exact_match(&requirement.dynamic, &scope.dynamic)
        && (!requirement.renderer_required
            || (instrument.claim_ids.len() >= 3
                && instrument.renderer_scope.as_ref().is_some_and(|renderer| {
                    renderer.renderer_id.as_ref().is_some_and(|value| !value.trim().is_empty())
                        && renderer.patch_id.as_ref().is_some_and(|value| !value.trim().is_empty())
                        && renderer
                            .asset_sha256
                            .as_ref()
                            .is_some_and(|digest| valid_sha256_hex_v1(digest))
                })))
}

fn supports_demand(
    instrument: &QualifiedInstrumentCapabilityV1,
    demand: &VoiceDemand,
    requirement: &PartRequirementV1,
) -> bool {
    scope_matches_requirement(instrument, requirement)
        && demand.min_midi >= instrument.capability.sounding_range.lowest_midi
        && demand.max_midi <= instrument.capability.sounding_range.highest_midi
        && demand.peak_simultaneous_notes as u64
            <= instrument.capability.max_polyphony as u64
}

fn candidate_from_assignment(
    score_sha256: &str,
    demands: &[VoiceDemand],
    requirements: &BTreeMap<u16, &PartRequirementV1>,
    instruments: &[QualifiedInstrumentCapabilityV1],
    selected_indices: &[Option<usize>],
) -> Option<OrchestrationCandidateV1> {
    let mut assignments = Vec::with_capacity(demands.len());
    for (index, demand) in demands.iter().enumerate() {
        let part_id = demand.part_id?;
        let requirement = requirements.get(&part_id)?;
        let instrument = instruments.get(selected_indices.get(index).copied().flatten()?)?;
        assignments.push(PlannedPartAssignmentV1 {
            part_id,
            instrument_instance_id: instrument.capability.id.clone(),
            musical_function: requirement.musical_function.clone(),
            instrument_kind: instrument.capability.kind,
            sounding_lowest_midi: instrument.capability.sounding_range.lowest_midi,
            sounding_highest_midi: instrument.capability.sounding_range.highest_midi,
            max_polyphony: instrument.capability.max_polyphony,
            capability_profile_sha256: instrument.profile_sha256.clone(),
            performance_scope: instrument.performance_scope.clone(),
            renderer_scope: instrument.renderer_scope.clone(),
            capability_claim_ids: instrument.claim_ids.clone(),
            evidence_ids: instrument.evidence_ids.clone(),
            qualification_record_ids: instrument.qualification_record_ids.clone(),
            qualification_record_sha256s: instrument.qualification_record_sha256s.clone(),
        });
    }
    assignments.sort_by_key(|assignment| assignment.part_id);

    #[derive(Serialize)]
    struct Fingerprint<'a> {
        schema: &'a str,
        score_sha256: &'a str,
        assignments: &'a [PlannedPartAssignmentV1],
    }
    let fingerprint = Fingerprint {
        schema: ORCHESTRATION_CANDIDATE_DIGEST_SCHEMA_V1,
        score_sha256,
        assignments: &assignments,
    };
    let bytes = serde_json::to_vec(&fingerprint).ok()?;
    let mut hasher = Sha256::new();
    hasher.update(ORCHESTRATION_CANDIDATE_DIGEST_SCHEMA_V1.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    let candidate_sha256 = format!("{:x}", hasher.finalize());

    Some(OrchestrationCandidateV1 {
        schema: ORCHESTRATION_PLANNER_SCHEMA_V1.into(),
        candidate_digest_schema: ORCHESTRATION_CANDIDATE_DIGEST_SCHEMA_V1.into(),
        score_sha256: score_sha256.into(),
        candidate_sha256,
        assignments,
    })
}

struct SearchContext<'a> {
    score_sha256: &'a str,
    demands: &'a [VoiceDemand],
    requirements: &'a BTreeMap<u16, &'a PartRequirementV1>,
    instruments: &'a [QualifiedInstrumentCapabilityV1],
    options_by_part: &'a [Vec<usize>],
    order: &'a [usize],
    max_nodes: usize,
    max_candidates: usize,
    state: SearchState,
}

impl SearchContext<'_> {
    fn visit(
        &mut self,
        depth: usize,
        selected: &mut [Option<usize>],
        used_instruments: &mut BTreeSet<usize>,
    ) {
        if self.state.budget_exhausted
            || self.state.candidate_limit_reached
            || self.state.serialization_failed
        {
            return;
        }
        if self.state.nodes >= self.max_nodes {
            self.state.budget_exhausted = true;
            return;
        }
        self.state.nodes += 1;

        if depth == self.order.len() {
            if self.state.candidates.len() >= self.max_candidates {
                self.state.candidate_limit_reached = true;
                return;
            }
            if let Some(candidate) = candidate_from_assignment(
                self.score_sha256,
                self.demands,
                self.requirements,
                self.instruments,
                selected,
            ) {
                self.state.candidates.push(candidate);
            } else {
                self.state.serialization_failed = true;
            }
            return;
        }

        let part_index = self.order[depth];
        for &instrument_index in &self.options_by_part[part_index] {
            if used_instruments.contains(&instrument_index) {
                continue;
            }
            used_instruments.insert(instrument_index);
            selected[part_index] = Some(instrument_index);
            self.visit(depth + 1, selected, used_instruments);
            selected[part_index] = None;
            used_instruments.remove(&instrument_index);
            if self.state.budget_exhausted
                || self.state.candidate_limit_reached
                || self.state.serialization_failed
            {
                break;
            }
        }
    }
}

/// Enumerate bounded, deterministic, one-instance-per-part feasible plans.
///
/// This first kernel performs hard range, polyphony and explicit scope checks
/// only. It does not rank by timbral diversity, masking, artistic preference
/// or style; those are later, separately measured objectives. Instruments are
/// never substituted and source scores are never modified.
pub fn plan_orchestration_v1(
    request: &OrchestrationPlannerRequestV1,
) -> OrchestrationPlannerReportV1 {
    let mut report = OrchestrationPlannerReportV1::new();

    if request.schema != ORCHESTRATION_PLANNER_SCHEMA_V1 {
        report.push(
            PlannerIssueCodeV1::InvalidSchema,
            None,
            None,
            "planner request schema is not supported",
        );
    }
    if !valid_sha256_hex_v1(&request.expected_score_sha256) {
        report.push(
            PlannerIssueCodeV1::InvalidExpectedScoreDigest,
            None,
            None,
            "expected source score digest must be 64 lowercase hexadecimal characters",
        );
    }
    if request.score_demand.score_sha256 != request.expected_score_sha256 {
        report.push(
            PlannerIssueCodeV1::ScoreDigestMismatch,
            None,
            None,
            "score-demand profile is bound to a different symbolic Score digest",
        );
    }
    if request.score_demand.schema != SCORE_DEMAND_PROFILE_SCHEMA {
        report.push(
            PlannerIssueCodeV1::InvalidScoreDemandSchema,
            None,
            None,
            "score-demand profile schema is not supported",
        );
    }
    if request.score_demand.input_digest_schema != SCORE_DEMAND_INPUT_DIGEST_SCHEMA {
        report.push(
            PlannerIssueCodeV1::InvalidScoreDemandDigestSchema,
            None,
            None,
            "score-demand digest schema is not supported",
        );
    }
    if !request.score_demand.has_complete_part_identity {
        report.push(
            PlannerIssueCodeV1::IncompletePartIdentity,
            None,
            None,
            "planner refuses role aggregates when any score note lacks a generator-authored PartId",
        );
    }
    if request.score_demand.assigned_part_note_fraction != 1.0 {
        report.push(
            PlannerIssueCodeV1::AssignedPartFractionIncomplete,
            None,
            None,
            "all score notes must carry real PartIds before orchestration",
        );
    }
    if request.limits.max_search_nodes == 0
        || request.limits.max_search_nodes > MAX_PLANNER_SEARCH_NODES_V1
        || request.limits.max_candidates == 0
        || request.limits.max_candidates > MAX_PLANNER_CANDIDATES_V1
    {
        report.push(
            PlannerIssueCodeV1::InvalidPlannerLimits,
            None,
            None,
            "search-node and candidate limits must be within the published bounds",
        );
    }

    if !report.issues.is_empty() {
        report.disposition = if report.issues.iter().all(|issue| {
            matches!(
                issue.code,
                PlannerIssueCodeV1::IncompletePartIdentity
                    | PlannerIssueCodeV1::AssignedPartFractionIncomplete
            )
        }) {
            PlannerDispositionV1::Abstained
        } else {
            PlannerDispositionV1::InvalidRequest
        };
        return report;
    }

    let mut demand_by_part = BTreeMap::<u16, VoiceDemand>::new();
    for demand in &request.score_demand.voice_demands {
        if demand.identity_basis != DemandIdentityBasis::StablePartId {
            // RoleAggregateSummary rows are workload summaries, not performers.
            // Incomplete-identity aggregates would have been rejected above.
            continue;
        }
        if !part_demand_is_valid(demand) {
            report.push(
                PlannerIssueCodeV1::InvalidPartDemand,
                demand.part_id,
                None,
                "stable part demand contains invalid or empty measurements",
            );
            continue;
        }
        let part_id = demand.part_id.expect("validated stable part has an ID");
        if demand_by_part.insert(part_id, demand.clone()).is_some() {
            report.push(
                PlannerIssueCodeV1::DuplicatePartDemand,
                Some(part_id),
                None,
                "score-demand profile contains duplicate stable PartId rows",
            );
        }
    }

    if demand_by_part.is_empty() {
        report.push(
            PlannerIssueCodeV1::NoStablePartDemands,
            None,
            None,
            "no stable part-level demands are available for planning",
        );
    }
    if demand_by_part.len() > MAX_PLANNER_PARTS_V1 {
        report.push(
            PlannerIssueCodeV1::TooManyParts,
            None,
            None,
            "score exceeds the bounded planner part count",
        );
    }

    let mut requirements = BTreeMap::<u16, &PartRequirementV1>::new();
    for requirement in &request.part_requirements {
        if requirement.musical_function.trim().is_empty() {
            report.push(
                PlannerIssueCodeV1::InvalidMusicalFunction,
                Some(requirement.part_id),
                None,
                "musical_function must be caller-authored and non-empty",
            );
        }
        if requirements.insert(requirement.part_id, requirement).is_some() {
            report.push(
                PlannerIssueCodeV1::DuplicatePartRequirement,
                Some(requirement.part_id),
                None,
                "each persistent PartId must have exactly one authored purpose/requirement record",
            );
        }
    }
    for part_id in demand_by_part.keys() {
        if !requirements.contains_key(part_id) {
            report.push(
                PlannerIssueCodeV1::MissingPartRequirement,
                Some(*part_id),
                None,
                "no caller-authored musical function and performance requirements exist for this part",
            );
        }
    }
    for part_id in requirements.keys() {
        if !demand_by_part.contains_key(part_id) {
            report.push(
                PlannerIssueCodeV1::UnknownPartRequirement,
                Some(*part_id),
                None,
                "part requirement refers to a PartId absent from the score-demand profile",
            );
        }
    }

    if request.instruments.is_empty()
        || request.instruments.len() > MAX_PLANNER_INSTRUMENTS_V1
    {
        report.push(
            PlannerIssueCodeV1::TooManyInstrumentInstances,
            None,
            None,
            "qualified instrument inventory must contain 1..=128 instances",
        );
    }
    let mut instrument_ids = BTreeSet::new();
    for instrument in &request.instruments {
        if !valid_capability_receipt(instrument) {
            report.push(
                PlannerIssueCodeV1::InvalidCapabilityReceipt,
                None,
                Some(&instrument.capability.id),
                "instrument capability receipt is malformed or incomplete",
            );
        }
        if !instrument_ids.insert(instrument.capability.id.as_str()) {
            report.push(
                PlannerIssueCodeV1::DuplicateInstrumentInstanceId,
                None,
                Some(&instrument.capability.id),
                "one planner request must contain one capability receipt per concrete instrument instance",
            );
        }
    }

    if !report.issues.is_empty() {
        report.disposition = PlannerDispositionV1::InvalidRequest;
        return report;
    }

    let demands: Vec<VoiceDemand> = demand_by_part.into_values().collect();
    let requirements: BTreeMap<u16, &PartRequirementV1> = requirements;
    report.score_sha256 = Some(request.score_demand.score_sha256.clone());
    report.part_count = demands.len();
    report.qualified_instrument_count = request.instruments.len();

    let mut instruments = request.instruments.clone();
    instruments.sort_by(|a, b| a.capability.id.cmp(&b.capability.id));

    let mut options_by_part = Vec::with_capacity(demands.len());
    for demand in &demands {
        let part_id = demand.part_id.expect("stable demand was validated");
        let requirement = requirements
            .get(&part_id)
            .expect("part requirement completeness was validated");
        let options: Vec<usize> = instruments
            .iter()
            .enumerate()
            .filter(|(_, instrument)| supports_demand(instrument, demand, requirement))
            .map(|(index, _)| index)
            .collect();
        if options.is_empty() {
            report.push(
                PlannerIssueCodeV1::NoQualifiedCandidateForPart,
                Some(part_id),
                None,
                "no exact-scope qualified instrument satisfies the part's sounding range and peak polyphony demand",
            );
        }
        options_by_part.push(options);
    }
    if !report.issues.is_empty() {
        report.disposition = PlannerDispositionV1::InfeasibleComplete;
        return report;
    }

    // Most-constrained part first; ties by PartId. Candidate choices are sorted
    // by concrete instrument instance ID before search.
    let mut order: Vec<usize> = (0..demands.len()).collect();
    order.sort_by(|left, right| {
        options_by_part[*left]
            .len()
            .cmp(&options_by_part[*right].len())
            .then_with(|| {
                demands[*left]
                    .part_id
                    .cmp(&demands[*right].part_id)
            })
    });

    let mut context = SearchContext {
        score_sha256: &request.score_demand.score_sha256,
        demands: &demands,
        requirements: &requirements,
        instruments: &instruments,
        options_by_part: &options_by_part,
        order: &order,
        max_nodes: request.limits.max_search_nodes,
        max_candidates: request.limits.max_candidates,
        state: SearchState {
            nodes: 0,
            candidates: Vec::new(),
            budget_exhausted: false,
            candidate_limit_reached: false,
            serialization_failed: false,
        },
    };
    let mut selected = vec![None; demands.len()];
    let mut used = BTreeSet::new();
    context.visit(0, &mut selected, &mut used);
    report.search_nodes = context.state.nodes;

    if context.state.budget_exhausted {
        report.push(
            PlannerIssueCodeV1::SearchBudgetExceeded,
            None,
            None,
            "search-node budget exhausted; partial candidates were discarded",
        );
        report.disposition = PlannerDispositionV1::BudgetExhausted;
        return report;
    }
    if context.state.serialization_failed {
        report.push(
            PlannerIssueCodeV1::CandidateSerializationFailed,
            None,
            None,
            "candidate fingerprint serialization failed",
        );
        report.disposition = PlannerDispositionV1::InvalidRequest;
        return report;
    }

    report.candidates = context.state.candidates;
    report.disposition = if context.state.candidate_limit_reached {
        PlannerDispositionV1::CandidateLimitReached
    } else if report.candidates.is_empty() {
        PlannerDispositionV1::InfeasibleComplete
    } else {
        PlannerDispositionV1::FeasibleComplete
    };
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instrument_capability_evidence::{
        CapabilityScopeV1, INSTRUMENT_CAPABILITY_EVIDENCE_SCHEMA_V1,
    };
    use crate::score_demand::{
        DemandIdentityBasis, VoiceDemand, SCORE_DEMAND_INPUT_DIGEST_SCHEMA,
        SCORE_DEMAND_PROFILE_SCHEMA,
    };
    use symthaea_music_theory::orchestration::{
        InstrumentCapabilityV1, InstrumentKindV1, SoundingRangeV1,
    };
    use symthaea_music_theory::score::VoiceRole;

    fn sha(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn demand(part_id: u16, min_midi: u8, max_midi: u8, peak: usize) -> VoiceDemand {
        VoiceDemand {
            part_id: Some(part_id),
            roles: vec![VoiceRole::Melody],
            identity_basis: DemandIdentityBasis::StablePartId,
            note_count: 16,
            min_midi,
            max_midi,
            mean_midi: (min_midi as f64 + max_midi as f64) / 2.0,
            pitch_span_semitones: max_midi - min_midi,
            distinct_onset_count: 16,
            mean_note_duration_beats: 1.0,
            peak_simultaneous_notes: peak,
            first_onset_beat: 0.0,
            last_note_end_beat: 16.0,
            active_span_beats: 16.0,
            notes_per_score_beat: 1.0,
            notes_per_bar: Some(4.0),
        }
    }

    fn score_demand(include_role_summary: bool) -> ScoreDemandProfile {
        let mut voice_demands = vec![
            demand(1, 60, 72, 1),
            demand(2, 55, 67, 1),
        ];
        if include_role_summary {
            let mut summary = demand(0, 55, 72, 2);
            summary.part_id = None;
            summary.identity_basis = DemandIdentityBasis::RoleAggregateSummary;
            summary.roles = vec![VoiceRole::Melody];
            voice_demands.push(summary);
        }
        ScoreDemandProfile {
            schema: SCORE_DEMAND_PROFILE_SCHEMA.into(),
            input_digest_schema: SCORE_DEMAND_INPUT_DIGEST_SCHEMA.into(),
            score_sha256: sha('a'),
            meter_beats: 4,
            tempo_bpm: 100.0,
            declared_score_span_beats: 16.0,
            analyzed_span_beats: 16.0,
            has_complete_part_identity: true,
            assigned_part_note_fraction: 1.0,
            voice_demands,
        }
    }

    fn requirement(part_id: u16, function: &str) -> PartRequirementV1 {
        PartRequirementV1 {
            part_id,
            musical_function: function.into(),
            performer_level: Some("professional".into()),
            technique: Some("arco".into()),
            dynamic: Some("mf".into()),
            renderer_required: false,
        }
    }

    fn instrument(
        id: &str,
        kind: InstrumentKindV1,
        low: u8,
        high: u8,
        polyphony: u8,
    ) -> QualifiedInstrumentCapabilityV1 {
        QualifiedInstrumentCapabilityV1 {
            capability: InstrumentCapabilityV1 {
                id: id.into(),
                kind,
                sounding_range: SoundingRangeV1 {
                    lowest_midi: low,
                    highest_midi: high,
                },
                max_polyphony: polyphony,
            },
            profile_sha256: sha(if id == "cello-1" { 'b' } else { 'c' }),
            profile_digest_schema: INSTRUMENT_CAPABILITY_PROFILE_DIGEST_SCHEMA_V1.into(),
            performance_scope: CapabilityScopeV1 {
                instrument_model: Some(format!("{id}-model")),
                performer_level: Some("professional".into()),
                technique: Some("arco".into()),
                dynamic: Some("mf".into()),
                renderer_id: None,
                patch_id: None,
                asset_sha256: None,
            },
            renderer_scope: None,
            claim_ids: vec!["range".into(), "polyphony".into()],
            evidence_ids: vec!["range-evidence".into(), "polyphony-evidence".into()],
            qualification_record_ids: vec!["range-review".into(), "polyphony-review".into()],
            qualification_record_sha256s: vec![sha('d'), sha('e')],
        }
    }

    fn request() -> OrchestrationPlannerRequestV1 {
        OrchestrationPlannerRequestV1 {
            schema: ORCHESTRATION_PLANNER_SCHEMA_V1.into(),
            expected_score_sha256: sha('a'),
            score_demand: score_demand(true),
            part_requirements: vec![
                requirement(1, "primary-theme"),
                requirement(2, "counter-theme"),
            ],
            instruments: vec![
                instrument("cello-1", InstrumentKindV1::Cello, 48, 84, 1),
                instrument("horn-1", InstrumentKindV1::FrenchHorn, 48, 91, 1),
            ],
            limits: PlannerLimitsV1::default(),
        }
    }

    #[test]
    fn enumerates_all_feasible_distinct_instrument_assignments() {
        let report = plan_orchestration_v1(&request());
        assert_eq!(report.disposition, PlannerDispositionV1::FeasibleComplete);
        assert_eq!(report.candidates.len(), 2);
        assert!(report.candidates.iter().all(|candidate| {
            candidate.assignments.len() == 2
                && candidate.assignments[0].part_id == 1
                && candidate.assignments[1].part_id == 2
                && candidate.assignments[0].instrument_instance_id
                    != candidate.assignments[1].instrument_instance_id
        }));
        assert!(report
            .candidates
            .iter()
            .all(|candidate| valid_sha256_hex_v1(&candidate.candidate_sha256)));
    }

    #[test]
    fn role_aggregate_summaries_are_not_counted_as_extra_parts() {
        let report = plan_orchestration_v1(&request());
        assert_eq!(report.part_count, 2);
        assert!(report
            .candidates
            .iter()
            .all(|candidate| candidate.assignments.len() == 2));
    }

    #[test]
    fn range_or_polyphony_violation_produces_no_plan() {
        let mut invalid = request();
        invalid.instruments = vec![
            instrument("small-cello", InstrumentKindV1::Cello, 65, 72, 1),
        ];
        let report = plan_orchestration_v1(&invalid);
        assert_eq!(report.disposition, PlannerDispositionV1::InfeasibleComplete);
        assert!(report.candidates.is_empty());
        assert!(report.issues.iter().any(|issue| {
            issue.code == PlannerIssueCodeV1::NoQualifiedCandidateForPart
        }));
    }

    #[test]
    fn incomplete_source_part_identity_causes_abstention() {
        let mut invalid = request();
        invalid.score_demand.has_complete_part_identity = false;
        invalid.score_demand.assigned_part_note_fraction = 0.5;
        let report = plan_orchestration_v1(&invalid);
        assert_eq!(report.disposition, PlannerDispositionV1::Abstained);
        assert!(report.candidates.is_empty());
    }

    #[test]
    fn mismatched_expected_score_digest_is_rejected() {
        let mut invalid = request();
        invalid.expected_score_sha256 = sha('f');
        let report = plan_orchestration_v1(&invalid);
        assert_eq!(report.disposition, PlannerDispositionV1::InvalidRequest);
        assert!(report.issues.iter().any(|issue| {
            issue.code == PlannerIssueCodeV1::ScoreDigestMismatch
        }));
    }

    #[test]
    fn source_scope_requirements_are_not_inferred_from_voice_role() {
        let mut invalid = request();
        invalid.part_requirements[0].technique = Some("pizzicato".into());
        let report = plan_orchestration_v1(&invalid);
        assert_eq!(report.disposition, PlannerDispositionV1::InfeasibleComplete);
        assert!(report.issues.iter().any(|issue| {
            issue.code == PlannerIssueCodeV1::NoQualifiedCandidateForPart
        }));
    }

    #[test]
    fn renderer_required_part_abstains_from_non_renderer_capabilities() {
        let mut request = request();
        request.part_requirements[0].renderer_required = true;
        let report = plan_orchestration_v1(&request);
        assert_eq!(report.disposition, PlannerDispositionV1::InfeasibleComplete);
        assert!(report.candidates.is_empty());
    }

    #[test]
    fn invalid_schema_and_incomplete_identity_are_not_misreported_as_abstention() {
        let mut request = request();
        request.schema = "unknown-schema".into();
        request.score_demand.has_complete_part_identity = false;
        let report = plan_orchestration_v1(&request);
        assert_eq!(report.disposition, PlannerDispositionV1::InvalidRequest);
    }

    #[test]
    fn search_budget_exhaustion_discards_partial_candidates() {
        let mut limited = request();
        limited.limits.max_search_nodes = 1;
        let report = plan_orchestration_v1(&limited);
        assert_eq!(report.disposition, PlannerDispositionV1::BudgetExhausted);
        assert!(report.candidates.is_empty());
    }

    #[test]
    fn candidate_limit_is_reported_as_incomplete_not_optimal() {
        let mut limited = request();
        limited.limits.max_candidates = 1;
        let report = plan_orchestration_v1(&limited);
        assert_eq!(report.disposition, PlannerDispositionV1::CandidateLimitReached);
        assert_eq!(report.candidates.len(), 1);
    }

    #[test]
    fn frozen_inputs_produce_identical_candidate_digests_and_order() {
        let request = request();
        let first = plan_orchestration_v1(&request);
        let second = plan_orchestration_v1(&request);
        assert_eq!(first.candidates, second.candidates);
        assert_eq!(first.search_nodes, second.search_nodes);
    }
}

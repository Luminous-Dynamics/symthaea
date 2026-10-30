// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, budgeted candidate selection for pairwise musical interaction probes.
//!
//! This module plans experiments; it does not measure interactions and does not
//! infer that an unselected pair is non-interacting. Acoustic support must be
//! derived from a separately verified influence artifact. MIDI note duration
//! alone is not a conservative acoustic-support estimate.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const INTERACTION_PROBE_PLAN_VERSION: &str = "mel-mr-eval-019-interaction-probe-plan-v0.1";
const PLAN_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-019-probe-plan-v1\0";
const MAX_EVENT_COUNT: usize = 100_000;
const MAX_INTERVALS_PER_EVENT: usize = 100_000;
const MAX_ROUTING_DOMAINS_PER_EVENT: usize = 256;
const MAX_ROUTING_GROUPS: usize = 100_000;
const MAX_SAMPLING_ATTEMPTS: usize = 1_000_000;
const MAX_TEMPORAL_CANDIDATE_PAIRS: usize = 2_000_000;
const MAX_TOTAL_SUPPORT_INTERVALS: usize = 2_000_000;
const MAX_TOTAL_ROUTING_MEMBERSHIPS: usize = 1_000_000;
const MAX_TOTAL_ROUTING_SAMPLES: usize = 2_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AcousticSupportInterval {
    pub start_frame: u64,
    pub end_frame: u64,
}

/// Caller-supplied, event-indexed evidence used to construct a probe plan.
///
/// `support_evidence_sha256` must identify the exact replay-verified EVAL-017
/// report from which non-empty support intervals were derived. This planner
/// commits that digest but intentionally does not re-verify the EVAL-017 report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventInteractionSupport {
    pub source_event_index: usize,
    pub acoustic_support: Vec<AcousticSupportInterval>,
    /// Exact renderer route/effect domains. Examples: `track:2`, `bus:chorus-1`,
    /// `master:limiter`. These are identifiers, not inferred musical labels.
    pub routing_domains: Vec<String>,
    /// False means route membership is incomplete/unknown and pruning is unsafe.
    pub routing_complete: bool,
    pub support_evidence_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionProbeConfig {
    /// Maximum candidate pairs sent to the factorial measurement stage.
    pub pair_probe_budget: usize,
    /// Maximum sampled pairs from each shared routing domain (or unknown scope).
    pub routing_group_sample_budget: usize,
    /// Independently reserved pairs sampled outside the candidate policy.
    pub audit_pair_budget: usize,
    /// Bounded rejection-sampling attempts for each sampling operation.
    pub max_sampling_attempts: usize,
    /// Seed for deterministic sampling; this is not a cryptographic RNG.
    pub seed: u64,
    /// Hard cap before temporal-overlap candidate materialization is refused.
    pub max_temporal_candidate_pairs: usize,
}

impl Default for InteractionProbeConfig {
    fn default() -> Self {
        Self {
            pair_probe_budget: 512,
            routing_group_sample_budget: 128,
            audit_pair_budget: 64,
            max_sampling_attempts: 100_000,
            seed: 0x4d45_4c19_2026_1030,
            max_temporal_candidate_pairs: 100_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "kind", content = "domain", rename_all = "snake_case")]
pub enum PairProbeReason {
    AcousticSupportOverlap,
    SharedRoutingDomain(String),
    IncompleteRoutingScope,
    AuditNonCandidate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionProbePair {
    pub event_a: usize,
    pub event_b: usize,
    pub reasons: Vec<PairProbeReason>,
    /// Maximum intersection length among the two events' support intervals.
    pub max_support_overlap_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoutingGroupSampling {
    pub domain_id: String,
    pub member_count: usize,
    /// Number of unordered pairs in the group, before cross-group deduplication.
    pub possible_pair_incidences: u128,
    pub sampled_pair_count: usize,
    pub sampling_attempts: usize,
    pub underfilled: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InteractionProbePlan {
    pub version: String,
    pub source_score_sha256: String,
    pub renderer_profile_sha256: String,
    pub support_and_topology_sha256: String,
    pub total_frames: u64,
    pub event_count: usize,
    pub config: InteractionProbeConfig,

    pub all_possible_pair_count: u128,
    pub temporal_candidate_pair_count: usize,
    pub routing_group_sampling: Vec<RoutingGroupSampling>,
    pub routing_group_pair_incidence_upper_bound: u128,
    pub incomplete_routing_event_count: usize,
    pub incomplete_routing_expands_to_global_scope: bool,
    pub materialized_candidate_pair_count: usize,
    pub candidate_pairs_selected: Vec<InteractionProbePair>,
    /// Materialized candidate proposals omitted only because the probe budget was exhausted.
    pub materialized_candidates_omitted_by_budget: usize,
    pub audit_pairs: Vec<InteractionProbePair>,
    pub audit_sampling_attempts: usize,
    pub audit_pairs_underfilled: usize,
    pub selected_pair_count: usize,
    pub unprobed_pair_count: u128,
    /// Reusable baseline + one single-removal render per event + selected double-removal renders.
    pub estimated_render_count_with_cached_single_removals: u128,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionProbePlanError {
    InvalidSha256(&'static str),
    InvalidFrameCount,
    InvalidEventCount,
    NonCanonicalEventIndex,
    TooManySupportIntervals,
    InvalidSupportInterval,
    MissingSupportEvidence,
    InvalidSupportEvidenceDigest,
    TooManyRoutingDomains,
    InvalidRoutingDomain,
    TooManyRoutingGroups,
    InvalidConfig(&'static str),
    TemporalCandidateBudgetExceeded,
    RoutingSamplingBudgetExceeded,
    SamplingAttemptLimit,
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for InteractionProbePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "interaction probe plan validation failed: {self:?}")
    }
}

impl std::error::Error for InteractionProbePlanError {}

#[derive(Debug, Clone, Serialize)]
struct NormalizedEvent {
    index: usize,
    support: Vec<AcousticSupportInterval>,
    domains: Vec<String>,
    routing_complete: bool,
    support_evidence_sha256: Option<String>,
}

#[derive(Debug, Default)]
struct CandidateAccumulator {
    reasons: BTreeSet<PairProbeReason>,
    max_support_overlap_frames: u64,
}

type PairKey = (usize, usize);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, InteractionProbePlanError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| InteractionProbePlanError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(PLAN_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn pair_key(left: usize, right: usize) -> PairKey {
    if left < right { (left, right) } else { (right, left) }
}

fn pair_count(event_count: usize) -> u128 {
    let n = event_count as u128;
    n.saturating_mul(n.saturating_sub(1)) / 2
}

fn mix_seed(seed: u64, domain: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(domain.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has at least 8 bytes"))
}

/// Small deterministic generator for reproducible planning, not security use.
#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn uniform_index(rng: &mut SplitMix64, upper_exclusive: usize) -> usize {
    let bound = upper_exclusive as u64;
    let rejection_threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next();
        if value >= rejection_threshold {
            return (value % bound) as usize;
        }
    }
}

fn normalize_inputs(
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
) -> Result<Vec<NormalizedEvent>, InteractionProbePlanError> {
    if !is_sha256(source_score_sha256) {
        return Err(InteractionProbePlanError::InvalidSha256("source score"));
    }
    if !is_sha256(renderer_profile_sha256) {
        return Err(InteractionProbePlanError::InvalidSha256("renderer profile"));
    }
    if total_frames == 0 {
        return Err(InteractionProbePlanError::InvalidFrameCount);
    }
    if event_supports.len() < 2 || event_supports.len() > MAX_EVENT_COUNT {
        return Err(InteractionProbePlanError::InvalidEventCount);
    }

    let mut normalized = Vec::with_capacity(event_supports.len());
    let mut total_support_intervals = 0usize;
    let mut total_routing_memberships = 0usize;
    for (expected_index, event) in event_supports.iter().enumerate() {
        if event.source_event_index != expected_index {
            return Err(InteractionProbePlanError::NonCanonicalEventIndex);
        }
        if event.acoustic_support.len() > MAX_INTERVALS_PER_EVENT {
            return Err(InteractionProbePlanError::TooManySupportIntervals);
        }
        total_support_intervals = total_support_intervals.saturating_add(event.acoustic_support.len());
        if total_support_intervals > MAX_TOTAL_SUPPORT_INTERVALS {
            return Err(InteractionProbePlanError::TooManySupportIntervals);
        }
        if !event.acoustic_support.is_empty() {
            let digest = event.support_evidence_sha256.as_deref()
                .ok_or(InteractionProbePlanError::MissingSupportEvidence)?;
            if !is_sha256(digest) {
                return Err(InteractionProbePlanError::InvalidSupportEvidenceDigest);
            }
        } else if let Some(digest) = event.support_evidence_sha256.as_deref() {
            if !is_sha256(digest) {
                return Err(InteractionProbePlanError::InvalidSupportEvidenceDigest);
            }
        }
        if event.routing_domains.len() > MAX_ROUTING_DOMAINS_PER_EVENT {
            return Err(InteractionProbePlanError::TooManyRoutingDomains);
        }
        total_routing_memberships = total_routing_memberships.saturating_add(event.routing_domains.len());
        if total_routing_memberships > MAX_TOTAL_ROUTING_MEMBERSHIPS {
            return Err(InteractionProbePlanError::TooManyRoutingDomains);
        }

        let mut support = event.acoustic_support.clone();
        for interval in &support {
            if interval.start_frame >= interval.end_frame || interval.end_frame > total_frames {
                return Err(InteractionProbePlanError::InvalidSupportInterval);
            }
        }
        support.sort_unstable();
        let mut merged: Vec<AcousticSupportInterval> = Vec::with_capacity(support.len());
        for interval in support {
            if let Some(last) = merged.last_mut() {
                if interval.start_frame <= last.end_frame {
                    last.end_frame = last.end_frame.max(interval.end_frame);
                    continue;
                }
            }
            merged.push(interval);
        }

        let mut domains = BTreeSet::new();
        for domain in &event.routing_domains {
            if domain.is_empty() || domain.len() > 128 || domain.chars().any(char::is_control) {
                return Err(InteractionProbePlanError::InvalidRoutingDomain);
            }
            if !domains.insert(domain.clone()) {
                return Err(InteractionProbePlanError::InvalidRoutingDomain);
            }
        }
        normalized.push(NormalizedEvent {
            index: expected_index,
            support: merged,
            domains: domains.into_iter().collect(),
            routing_complete: event.routing_complete,
            support_evidence_sha256: event.support_evidence_sha256.clone(),
        });
    }
    Ok(normalized)
}

fn validate_config(config: &InteractionProbeConfig) -> Result<(), InteractionProbePlanError> {
    if config.pair_probe_budget == 0 || config.pair_probe_budget > MAX_SAMPLING_ATTEMPTS {
        return Err(InteractionProbePlanError::InvalidConfig("pair_probe_budget is outside the bounded range"));
    }
    if config.routing_group_sample_budget == 0 || config.routing_group_sample_budget > MAX_SAMPLING_ATTEMPTS {
        return Err(InteractionProbePlanError::InvalidConfig("routing_group_sample_budget is outside the bounded range"));
    }
    if config.audit_pair_budget > MAX_SAMPLING_ATTEMPTS {
        return Err(InteractionProbePlanError::InvalidConfig("audit_pair_budget is outside the bounded range"));
    }
    if config.max_sampling_attempts == 0 || config.max_sampling_attempts > MAX_SAMPLING_ATTEMPTS {
        return Err(InteractionProbePlanError::InvalidConfig("max_sampling_attempts is outside the bounded range"));
    }
    if config.max_temporal_candidate_pairs == 0
        || config.max_temporal_candidate_pairs > MAX_TEMPORAL_CANDIDATE_PAIRS
    {
        return Err(InteractionProbePlanError::InvalidConfig("max_temporal_candidate_pairs is outside the bounded range"));
    }
    Ok(())
}

fn add_candidate(
    candidates: &mut BTreeMap<PairKey, CandidateAccumulator>,
    left: usize,
    right: usize,
    reason: PairProbeReason,
    overlap_frames: u64,
) {
    if left == right { return; }
    let entry = candidates.entry(pair_key(left, right)).or_default();
    entry.reasons.insert(reason);
    entry.max_support_overlap_frames = entry.max_support_overlap_frames.max(overlap_frames);
}

fn temporal_candidates(
    events: &[NormalizedEvent],
    max_pairs: usize,
) -> Result<BTreeMap<PairKey, CandidateAccumulator>, InteractionProbePlanError> {
    let mut intervals = Vec::new();
    for event in events {
        for interval in &event.support {
            intervals.push((interval.start_frame, interval.end_frame, event.index));
        }
    }
    intervals.sort_unstable();

    let mut active: Vec<(u64, usize)> = Vec::new();
    let mut candidates = BTreeMap::new();
    for (start, end, event_index) in intervals {
        active.retain(|(active_end, _)| *active_end > start);
        for (active_end, other_index) in &active {
            if *other_index == event_index { continue; }
            let overlap = end.min(*active_end).saturating_sub(start);
            if overlap == 0 { continue; }
            add_candidate(
                &mut candidates,
                event_index,
                *other_index,
                PairProbeReason::AcousticSupportOverlap,
                overlap,
            );
            if candidates.len() > max_pairs {
                return Err(InteractionProbePlanError::TemporalCandidateBudgetExceeded);
            }
        }
        active.push((end, event_index));
    }
    Ok(candidates)
}

fn sample_pairs_from_members(
    members: &[usize],
    wanted: usize,
    seed: u64,
    max_attempts: usize,
) -> (Vec<PairKey>, usize, bool) {
    if members.len() < 2 || wanted == 0 { return (Vec::new(), 0, wanted > 0); }
    let possible = pair_count(members.len());
    let target = (wanted as u128).min(possible) as usize;
    let mut pairs = BTreeSet::new();
    if possible <= target as u128 {
        for left in 0..members.len() {
            for right in (left + 1)..members.len() {
                pairs.insert(pair_key(members[left], members[right]));
            }
        }
        return (pairs.into_iter().collect(), 0, false);
    }

    let mut rng = SplitMix64(seed);
    let mut attempts = 0usize;
    while pairs.len() < target && attempts < max_attempts {
        attempts += 1;
        let left = uniform_index(&mut rng, members.len());
        let right = uniform_index(&mut rng, members.len());
        if left == right { continue; }
        pairs.insert(pair_key(members[left], members[right]));
    }
    let underfilled = pairs.len() < target;
    (pairs.into_iter().collect(), attempts, underfilled)
}

fn is_candidate_pair(left: &NormalizedEvent, right: &NormalizedEvent, global_unknown: bool) -> bool {
    if global_unknown || !left.routing_complete || !right.routing_complete {
        return true;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.support.len() && j < right.support.len() {
        let a = left.support[i];
        let b = right.support[j];
        if a.start_frame < b.end_frame && b.start_frame < a.end_frame {
            return true;
        }
        if a.end_frame <= b.end_frame { i += 1; } else { j += 1; }
    }
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.domains.len() && j < right.domains.len() {
        match left.domains[i].cmp(&right.domains[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn pair_tiebreak(seed: u64, pair: &InteractionProbePair) -> u64 {
    mix_seed(seed, &format!("pair:{}:{}", pair.event_a, pair.event_b))
}

fn select_candidate_pairs(
    candidate_pairs: Vec<InteractionProbePair>,
    budget: usize,
    seed: u64,
) -> Vec<InteractionProbePair> {
    let mut buckets: [Vec<InteractionProbePair>; 3] = std::array::from_fn(|_| Vec::new());
    for pair in candidate_pairs {
        let unknown = pair.reasons.iter().any(|reason| {
            matches!(reason, PairProbeReason::IncompleteRoutingScope)
        });
        let temporal = pair.reasons.iter().any(|reason| {
            matches!(reason, PairProbeReason::AcousticSupportOverlap)
        });
        let bucket = if unknown { 0 } else if temporal { 1 } else { 2 };
        buckets[bucket].push(pair);
    }
    for bucket in &mut buckets {
        bucket.sort_by_key(|pair| {
            let overlap_priority = if pair.reasons.iter().any(|reason| {
                matches!(reason, PairProbeReason::AcousticSupportOverlap)
            }) {
                u64::MAX - pair.max_support_overlap_frames
            } else {
                0
            };
            (overlap_priority, pair_tiebreak(seed, pair))
        });
    }

    let available = buckets.iter().map(Vec::len).sum();
    let mut selected = Vec::with_capacity(budget.min(available));
    let mut cursors = [0usize; 3];
    while selected.len() < budget {
        let mut progressed = false;
        for bucket_index in 0..buckets.len() {
            if let Some(pair) = buckets[bucket_index].get(cursors[bucket_index]) {
                selected.push(pair.clone());
                cursors[bucket_index] += 1;
                progressed = true;
                if selected.len() == budget {
                    break;
                }
            }
        }
        if !progressed {
            break;
        }
    }
    selected
}

fn plan_digest_valid(plan: &InteractionProbePlan) -> Result<bool, InteractionProbePlanError> {
    let mut unsigned = plan.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if receipt.len() != 64 { return Ok(false); }
    Ok(canonical_digest(&unsigned)? == receipt)
}

/// Build a deterministic interaction-probe plan. Every selected pair still
/// requires an exact four-coalition EVAL-018 report; this function does not
/// synthesize the missing renders or classify unselected pairs.
pub fn build_interaction_probe_plan(
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    config: &InteractionProbeConfig,
) -> Result<InteractionProbePlan, InteractionProbePlanError> {
    validate_config(config)?;
    let events = normalize_inputs(source_score_sha256, renderer_profile_sha256, total_frames, event_supports)?;
    let mut candidates = temporal_candidates(&events, config.max_temporal_candidate_pairs)?;
    let temporal_candidate_pair_count = candidates.len();

    let incomplete_routing_event_count = events.iter().filter(|event| !event.routing_complete).count();
    let incomplete_routing_expands_to_global_scope = incomplete_routing_event_count > 0;
    if incomplete_routing_expands_to_global_scope {
        for candidate in candidates.values_mut() {
            candidate.reasons.insert(PairProbeReason::IncompleteRoutingScope);
        }
    }
    let mut group_members: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    if incomplete_routing_expands_to_global_scope {
        group_members.insert("__incomplete_routing_global_scope__".to_owned(), events.iter().map(|event| event.index).collect());
    } else {
        for event in &events {
            for domain in &event.domains {
                group_members.entry(domain.clone()).or_default().push(event.index);
            }
        }
    }
    if group_members.len() > MAX_ROUTING_GROUPS {
        return Err(InteractionProbePlanError::TooManyRoutingGroups);
    }

    let mut routing_group_sampling = Vec::with_capacity(group_members.len());
    let mut routing_group_pair_incidence_upper_bound = 0u128;
    let mut total_routing_samples = 0usize;
    for (domain_id, members) in group_members {
        if members.len() < 2 { continue; }
        let possible_pair_incidences = pair_count(members.len());
        routing_group_pair_incidence_upper_bound = routing_group_pair_incidence_upper_bound
            .saturating_add(possible_pair_incidences);
        let total_sample_cap = config.pair_probe_budget
            .saturating_mul(8)
            .min(MAX_TOTAL_ROUTING_SAMPLES);
        let remaining_sample_capacity = total_sample_cap.saturating_sub(total_routing_samples);
        if remaining_sample_capacity == 0 {
            return Err(InteractionProbePlanError::RoutingSamplingBudgetExceeded);
        }
        let wanted = config.routing_group_sample_budget.min(remaining_sample_capacity);
        let seed = mix_seed(config.seed, &domain_id);
        let (sampled, attempts, underfilled) = sample_pairs_from_members(
            &members,
            wanted,
            seed,
            config.max_sampling_attempts,
        );
        total_routing_samples = total_routing_samples.saturating_add(sampled.len());
        for (left, right) in &sampled {
            let reason = if incomplete_routing_expands_to_global_scope {
                PairProbeReason::IncompleteRoutingScope
            } else {
                PairProbeReason::SharedRoutingDomain(domain_id.clone())
            };
            add_candidate(&mut candidates, *left, *right, reason, 0);
        }
        routing_group_sampling.push(RoutingGroupSampling {
            domain_id,
            member_count: members.len(),
            possible_pair_incidences,
            sampled_pair_count: sampled.len(),
            sampling_attempts: attempts,
            underfilled,
        });
    }

    let materialized_candidate_pair_count = candidates.len();
    let mut candidate_pairs: Vec<_> = candidates.into_iter().map(|((event_a, event_b), accumulator)| {
        InteractionProbePair {
            event_a,
            event_b,
            reasons: accumulator.reasons.into_iter().collect(),
            max_support_overlap_frames: accumulator.max_support_overlap_frames,
        }
    }).collect();
    let candidate_pairs_selected = select_candidate_pairs(
        candidate_pairs,
        config.pair_probe_budget,
        config.seed,
    );
    let materialized_candidates_omitted_by_budget = materialized_candidate_pair_count
        .saturating_sub(candidate_pairs_selected.len());

    let mut audit_pairs = Vec::new();
    let mut audit_sampling_attempts = 0usize;
    if config.audit_pair_budget > 0 && !incomplete_routing_expands_to_global_scope {
        let mut rng = SplitMix64(mix_seed(config.seed, "audit-non-candidate-pairs"));
        let mut sampled = BTreeSet::new();
        while sampled.len() < config.audit_pair_budget
            && audit_sampling_attempts < config.max_sampling_attempts
        {
            audit_sampling_attempts += 1;
            let left = uniform_index(&mut rng, events.len());
            let right = uniform_index(&mut rng, events.len());
            if left == right { continue; }
            let key = pair_key(left, right);
            if sampled.contains(&key) { continue; }
            if is_candidate_pair(&events[key.0], &events[key.1], false) { continue; }
            sampled.insert(key);
        }
        audit_pairs = sampled.into_iter().map(|(event_a, event_b)| InteractionProbePair {
            event_a,
            event_b,
            reasons: vec![PairProbeReason::AuditNonCandidate],
            max_support_overlap_frames: 0,
        }).collect();
    }
    let audit_pairs_underfilled = config.audit_pair_budget.saturating_sub(audit_pairs.len());

    let selected_pair_count = candidate_pairs_selected.len().saturating_add(audit_pairs.len());
    let all_possible_pair_count = pair_count(events.len());
    let unprobed_pair_count = all_possible_pair_count.saturating_sub(selected_pair_count as u128);
    let estimated_render_count_with_cached_single_removals = 1u128
        .saturating_add(events.len() as u128)
        .saturating_add(selected_pair_count as u128);

    let support_and_topology_bytes = serde_json::to_vec(&events)
        .map_err(|_| InteractionProbePlanError::SerializationFailure)?;
    let support_and_topology_sha256 = sha256_hex(&support_and_topology_bytes);

    let mut plan = InteractionProbePlan {
        version: INTERACTION_PROBE_PLAN_VERSION.to_owned(),
        source_score_sha256: source_score_sha256.to_owned(),
        renderer_profile_sha256: renderer_profile_sha256.to_owned(),
        support_and_topology_sha256,
        total_frames,
        event_count: events.len(),
        config: config.clone(),
        all_possible_pair_count,
        temporal_candidate_pair_count,
        routing_group_sampling,
        routing_group_pair_incidence_upper_bound,
        incomplete_routing_event_count,
        incomplete_routing_expands_to_global_scope,
        materialized_candidate_pair_count,
        candidate_pairs_selected,
        materialized_candidates_omitted_by_budget,
        audit_pairs,
        audit_sampling_attempts,
        audit_pairs_underfilled,
        selected_pair_count,
        unprobed_pair_count,
        estimated_render_count_with_cached_single_removals,
        receipt_sha256: String::new(),
    };
    plan.receipt_sha256 = canonical_digest(&plan)?;
    Ok(plan)
}

/// Replay the plan from the exact support/topology inputs and configuration.
pub fn verify_interaction_probe_plan(
    plan: &InteractionProbePlan,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    config: &InteractionProbeConfig,
) -> Result<bool, InteractionProbePlanError> {
    if !plan_digest_valid(plan)? {
        return Err(InteractionProbePlanError::InvalidReceipt);
    }
    let expected = build_interaction_probe_plan(
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        config,
    )?;
    if &expected != plan {
        return Err(InteractionProbePlanError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String { sha256_hex(label.as_bytes()) }

    fn event(index: usize, intervals: &[(u64, u64)], domains: &[&str], routing_complete: bool) -> EventInteractionSupport {
        EventInteractionSupport {
            source_event_index: index,
            acoustic_support: intervals.iter().map(|(start_frame, end_frame)| AcousticSupportInterval { start_frame: *start_frame, end_frame: *end_frame }).collect(),
            routing_domains: domains.iter().map(|domain| (*domain).to_owned()).collect(),
            routing_complete,
            support_evidence_sha256: if intervals.is_empty() { None } else { Some(digest(&format!("support-{index}"))) },
        }
    }

    fn config() -> InteractionProbeConfig {
        InteractionProbeConfig {
            pair_probe_budget: 16,
            routing_group_sample_budget: 16,
            audit_pair_budget: 0,
            max_sampling_attempts: 10_000,
            seed: 7,
            max_temporal_candidate_pairs: 1_000,
        }
    }

    fn plan(events: &[EventInteractionSupport], cfg: &InteractionProbeConfig) -> InteractionProbePlan {
        build_interaction_probe_plan(&digest("score"), &digest("renderer"), 10_000, events, cfg).unwrap()
    }

    #[test]
    fn temporal_support_overlap_creates_candidate_pair() {
        let events = vec![event(0, &[(100, 300)], &[], true), event(1, &[(250, 400)], &[], true)];
        let result = plan(&events, &config());
        assert_eq!(result.temporal_candidate_pair_count, 1);
        let pair = &result.candidate_pairs_selected[0];
        assert_eq!((pair.event_a, pair.event_b), (0, 1));
        assert!(pair.reasons.contains(&PairProbeReason::AcousticSupportOverlap));
        assert_eq!(pair.max_support_overlap_frames, 50);
    }

    #[test]
    fn shared_effect_domain_creates_remote_nonoverlap_candidate() {
        let events = vec![event(0, &[(100, 200)], &["bus:compressor"], true), event(1, &[(800, 900)], &["bus:compressor"], true)];
        let result = plan(&events, &config());
        assert!(result.candidate_pairs_selected[0].reasons.contains(&PairProbeReason::SharedRoutingDomain("bus:compressor".into())));
        assert_eq!(result.temporal_candidate_pair_count, 0);
    }

    #[test]
    fn selection_round_robins_temporal_and_remote_routing_candidates() {
        let mut events: Vec<_> = (0..9)
            .map(|index| event(index, &[(10, 100)], &[], true))
            .collect();
        events.push(event(9, &[(500, 600)], &["bus:master"], true));
        events.push(event(10, &[(700, 800)], &["bus:master"], true));
        let mut cfg = config();
        cfg.pair_probe_budget = 4;
        cfg.routing_group_sample_budget = 1;
        let result = plan(&events, &cfg);
        assert!(result.candidate_pairs_selected.iter().any(|pair| {
            pair.reasons.iter().any(|reason| {
                matches!(reason, PairProbeReason::SharedRoutingDomain(domain) if domain == "bus:master")
            })
        }));
        assert!(result.candidate_pairs_selected.iter().any(|pair| {
            pair.reasons.contains(&PairProbeReason::AcousticSupportOverlap)
        }));
    }

    #[test]
    fn incomplete_routing_expands_scope_and_disables_non_candidate_audits() {
        let mut cfg = config();
        cfg.audit_pair_budget = 4;
        cfg.pair_probe_budget = 2;
        cfg.routing_group_sample_budget = 2;
        let events = vec![
            event(0, &[], &["track:1"], false),
            event(1, &[], &["track:2"], true),
            event(2, &[], &["track:3"], true),
            event(3, &[], &["track:4"], true),
        ];
        let result = plan(&events, &cfg);
        assert!(result.incomplete_routing_expands_to_global_scope);
        assert_eq!(result.incomplete_routing_event_count, 1);
        assert!(result.audit_pairs.is_empty());
        assert_eq!(result.all_possible_pair_count, 6);
        assert!(result.candidate_pairs_selected.iter().all(|pair| pair.reasons.contains(&PairProbeReason::IncompleteRoutingScope)));
        assert_eq!(result.unprobed_pair_count, 4);
    }

    #[test]
    fn deterministic_audit_samples_only_pairs_outside_candidate_policy() {
        let mut cfg = config();
        cfg.pair_probe_budget = 4;
        cfg.routing_group_sample_budget = 2;
        cfg.audit_pair_budget = 3;
        let events = vec![
            event(0, &[(0, 100)], &["track:0"], true),
            event(1, &[(50, 150)], &["track:0"], true),
            event(2, &[(500, 600)], &["track:2"], true),
            event(3, &[(700, 800)], &["track:3"], true),
            event(4, &[(900, 950)], &["track:4"], true),
            event(5, &[(1_000, 1_050)], &["track:5"], true),
        ];
        let first = plan(&events, &cfg);
        let second = plan(&events, &cfg);
        assert_eq!(first, second);
        assert_eq!(first.audit_pairs.len(), 3);
        let normalized = normalize_inputs(&digest("score"), &digest("renderer"), 10_000, &events).unwrap();
        for pair in &first.audit_pairs {
            assert!(!is_candidate_pair(&normalized[pair.event_a], &normalized[pair.event_b], false));
            assert_eq!(pair.reasons, vec![PairProbeReason::AuditNonCandidate]);
        }
        assert!(verify_interaction_probe_plan(&first, &digest("score"), &digest("renderer"), 10_000, &events, &cfg).unwrap());
    }

    #[test]
    fn plan_reports_reusable_render_cost_and_unprobed_pairs() {
        let events: Vec<_> = (0..20).map(|index| event(index, &[], &[&format!("track:{index}")], true)).collect();
        let mut cfg = config();
        cfg.pair_probe_budget = 5;
        cfg.routing_group_sample_budget = 2;
        cfg.audit_pair_budget = 3;
        let result = plan(&events, &cfg);
        assert_eq!(result.selected_pair_count, 3);
        assert_eq!(result.estimated_render_count_with_cached_single_removals, 24);
        assert_eq!(result.unprobed_pair_count, 187);
    }

    #[test]
    fn invalid_support_and_unbound_evidence_fail_closed() {
        let bad_interval = vec![event(0, &[(10, 10)], &[], true), event(1, &[], &[], true)];
        assert_eq!(
            build_interaction_probe_plan(&digest("score"), &digest("renderer"), 100, &bad_interval, &config()).unwrap_err(),
            InteractionProbePlanError::InvalidSupportInterval
        );
        let mut missing = event(0, &[(10, 20)], &[], true);
        missing.support_evidence_sha256 = None;
        let bad_evidence = vec![missing, event(1, &[], &[], true)];
        assert_eq!(
            build_interaction_probe_plan(&digest("score"), &digest("renderer"), 100, &bad_evidence, &config()).unwrap_err(),
            InteractionProbePlanError::MissingSupportEvidence
        );
    }

    #[test]
    fn temporal_candidate_materialization_has_a_hard_cap() {
        let events: Vec<_> = (0..8).map(|index| event(index, &[(10, 100)], &[], true)).collect();
        let mut cfg = config();
        cfg.max_temporal_candidate_pairs = 4;
        assert_eq!(
            build_interaction_probe_plan(&digest("score"), &digest("renderer"), 1_000, &events, &cfg).unwrap_err(),
            InteractionProbePlanError::TemporalCandidateBudgetExceeded
        );
    }

    #[test]
    fn report_digest_tampering_is_rejected() {
        let events = vec![event(0, &[], &["bus:x"], true), event(1, &[], &["bus:x"], true)];
        let mut result = plan(&events, &config());
        result.event_count += 1;
        assert_eq!(
            verify_interaction_probe_plan(&result, &digest("score"), &digest("renderer"), 10_000, &events, &config()).unwrap_err(),
            InteractionProbePlanError::InvalidReceipt
        );
    }
}

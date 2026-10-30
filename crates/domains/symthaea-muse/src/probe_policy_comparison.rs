// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Matched-render-budget comparison of sparse musical interaction probe policies.
//!
//! Compares simple baselines and MEL-MR-EVAL-019 against the same replay-verified
//! EVAL-020 factorial oracle. It reports each policy/seed/budget row separately;
//! it does not pool seeds into false independent observations or infer that an
//! unselected pair is null.

use crate::factorial_ground_truth::{
    FactorialGroundTruthReport, FactorialReferenceConfig, FactorialReferenceError,
    SyntheticEffect, SyntheticEventSignal, verify_factorial_ground_truth,
};
use crate::interaction_probe_benchmark::{
    InteractionProbeBenchmarkConfig, InteractionProbeBenchmarkError,
    contextual_pair_interaction_l2_norm,
};
use crate::interaction_probe_plan::{
    AcousticSupportInterval, EventInteractionSupport, InteractionProbeConfig,
    InteractionProbePlanError, build_interaction_probe_plan,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PROBE_POLICY_COMPARISON_VERSION: &str =
    "mel-mr-eval-024-probe-policy-comparison-v0.1";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-024-policy-comparison-v1\0";
const MAX_COMPARISON_SEEDS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProbeSelectionPolicy {
    NoProbe,
    UniformRandom,
    TemporalOnly,
    RoutingOnly,
    Eval019,
    OracleTopK,
    Exhaustive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePolicyComparisonRow {
    pub policy: ProbeSelectionPolicy,
    pub seed: u64,
    pub requested_pair_budget: usize,
    pub selected_pair_count: usize,
    pub true_positive_count: usize,
    pub false_positive_count: usize,
    pub false_negative_count: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    /// Base + event-removal renders + selected pair renders.
    pub total_render_count: u128,
    pub exhaustive_render_count: u128,
    pub render_cost_fraction_of_exhaustive: f64,
    /// A candidate policy may legitimately spend less than its requested budget.
    pub unused_pair_budget: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePolicyCurveSummary {
    pub policy: ProbeSelectionPolicy,
    pub seed: u64,
    /// Trapezoidal area over observed recall versus actual render-cost fraction.
    /// No extrapolation is made to an unobserved cost range.
    pub observed_auc_recall_cost: Option<f64>,
    pub maximum_observed_render_cost_fraction: Option<f64>,
    /// First observed actual cost at which recall reaches the preregistered target.
    pub cost_fraction_to_target_recall: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePolicyComparisonReport {
    pub version: String,
    pub factorial_report_sha256: String,
    pub source_score_sha256: String,
    pub renderer_profile_sha256: String,
    pub support_and_topology_sha256: String,
    pub factorial_config: FactorialReferenceConfig,
    pub planner_config_template: InteractionProbeConfig,
    pub metric_config: InteractionProbeBenchmarkConfig,
    pub budgets: Vec<usize>,
    pub seeds: Vec<u64>,
    pub event_count: usize,
    pub possible_pair_count: usize,
    pub contextual_positive_pair_count: usize,
    pub global_order_two_harsanyi_count: usize,
    pub higher_order_harsanyi_count: usize,
    pub recall_target: f64,
    pub rows: Vec<ProbePolicyComparisonRow>,
    pub curves: Vec<ProbePolicyCurveSummary>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbePolicyComparisonError {
    InvalidConfig(&'static str),
    EventCountMismatch,
    FrameCountMismatch,
    InvalidSupport,
    MissingSupportEvidence,
    InvalidDigest,
    FactorialReference(FactorialReferenceError),
    ProbePlan(InteractionProbePlanError),
    ProbeBenchmark(InteractionProbeBenchmarkError),
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for ProbePolicyComparisonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "probe-policy comparison validation failed: {self:?}")
    }
}

impl std::error::Error for ProbePolicyComparisonError {}

#[derive(Debug, Clone)]
struct NormalizedSupport {
    intervals: Vec<AcousticSupportInterval>,
    domains: Vec<String>,
    routing_complete: bool,
}

#[derive(Debug, Clone)]
struct PairCandidate {
    left: usize,
    right: usize,
    overlap_frames: u64,
    contextual_norm: f64,
}

#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn below(&mut self, upper_exclusive: usize) -> usize {
        let bound = upper_exclusive as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_digest<T: Serialize>(
    value: &T,
) -> Result<String, ProbePolicyComparisonError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| ProbePolicyComparisonError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn digest_is_valid(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn pair_count(event_count: usize) -> usize {
    event_count.saturating_mul(event_count.saturating_sub(1)) / 2
}

fn pair_key(left: usize, right: usize) -> (usize, usize) {
    if left < right { (left, right) } else { (right, left) }
}

fn pair_tiebreak(seed: u64, left: usize, right: usize) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update((left as u64).to_le_bytes());
    hasher.update((right as u64).to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 contains 8 bytes"))
}

fn normalize_supports(
    event_supports: &[EventInteractionSupport],
    event_count: usize,
    total_frames: u64,
) -> Result<Vec<NormalizedSupport>, ProbePolicyComparisonError> {
    if event_supports.len() != event_count {
        return Err(ProbePolicyComparisonError::EventCountMismatch);
    }
    let mut normalized = Vec::with_capacity(event_count);
    for (expected_index, event) in event_supports.iter().enumerate() {
        if event.source_event_index != expected_index {
            return Err(ProbePolicyComparisonError::InvalidSupport);
        }
        if event.acoustic_support.len() > 100_000
            || event.routing_domains.len() > 256
        {
            return Err(ProbePolicyComparisonError::InvalidSupport);
        }
        let support_digest = event.support_evidence_sha256.as_deref()
            .ok_or(ProbePolicyComparisonError::MissingSupportEvidence)?;
        if !digest_is_valid(support_digest) {
            return Err(ProbePolicyComparisonError::InvalidSupport);
        }
        let mut intervals = event.acoustic_support.clone();
        for interval in &intervals {
            if interval.start_frame >= interval.end_frame || interval.end_frame > total_frames {
                return Err(ProbePolicyComparisonError::InvalidSupport);
            }
        }
        intervals.sort_unstable();
        let mut merged: Vec<AcousticSupportInterval> = Vec::with_capacity(intervals.len());
        for interval in intervals {
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
            if domain.is_empty() || domain.len() > 128 || domain.chars().any(char::is_control)
                || !domains.insert(domain.clone())
            {
                return Err(ProbePolicyComparisonError::InvalidSupport);
            }
        }
        normalized.push(NormalizedSupport {
            intervals: merged,
            domains: domains.into_iter().collect(),
            routing_complete: event.routing_complete,
        });
    }
    Ok(normalized)
}

fn support_overlap(left: &[AcousticSupportInterval], right: &[AcousticSupportInterval]) -> u64 {
    let mut i = 0usize;
    let mut j = 0usize;
    let mut maximum = 0u64;
    while i < left.len() && j < right.len() {
        let a = left[i];
        let b = right[j];
        let overlap = a.end_frame.min(b.end_frame).saturating_sub(
            a.start_frame.max(b.start_frame),
        );
        maximum = maximum.max(overlap);
        if a.end_frame <= b.end_frame {
            i += 1;
        } else {
            j += 1;
        }
    }
    maximum
}

fn share_routing_domain(left: &[String], right: &[String]) -> bool {
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Equal => return true,
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
        }
    }
    false
}

fn all_pairs(event_count: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(pair_count(event_count));
    for left in 0..event_count {
        for right in (left + 1)..event_count {
            pairs.push((left, right));
        }
    }
    pairs
}

fn contextual_truth(
    factorial_report: &FactorialGroundTruthReport,
    event_count: usize,
    metric_config: &InteractionProbeBenchmarkConfig,
) -> Result<(Vec<PairCandidate>, usize, usize), ProbePolicyComparisonError> {
    if !metric_config.minimum_contextual_pair_pcm_l2_norm.is_finite()
        || metric_config.minimum_contextual_pair_pcm_l2_norm <= 0.0
        || !metric_config.minimum_global_order_two_harsanyi_pcm_l2_norm.is_finite()
        || metric_config.minimum_global_order_two_harsanyi_pcm_l2_norm <= 0.0
        || !metric_config.minimum_higher_order_pcm_l2_norm.is_finite()
        || metric_config.minimum_higher_order_pcm_l2_norm <= 0.0
    {
        return Err(ProbePolicyComparisonError::InvalidConfig(
            "interaction thresholds must be finite and positive",
        ));
    }

    let mut pairs = Vec::new();
    for (left, right) in all_pairs(event_count) {
        let norm = contextual_pair_interaction_l2_norm(factorial_report, left, right)
            .map_err(ProbePolicyComparisonError::ProbeBenchmark)?;
        pairs.push(PairCandidate {
            left,
            right,
            overlap_frames: 0,
            contextual_norm: norm,
        });
    }

    let mut global_order_two_count = 0usize;
    let mut higher_order_count = 0usize;
    for dividend in &factorial_report.dividends {
        if dividend.order == 2
            && dividend.pcm_l2_norm > metric_config.minimum_global_order_two_harsanyi_pcm_l2_norm
        {
            global_order_two_count += 1;
        } else if dividend.order >= 3
            && dividend.pcm_l2_norm > metric_config.minimum_higher_order_pcm_l2_norm
        {
            higher_order_count += 1;
        }
    }

    Ok((pairs, global_order_two_count, higher_order_count))
}

fn choose_uniform_random(
    all: &[(usize, usize)],
    budget: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let mut shuffled = all.to_vec();
    let mut rng = SplitMix64(seed);
    for index in (1..shuffled.len()).rev() {
        let other = rng.below(index + 1);
        shuffled.swap(index, other);
    }
    shuffled.truncate(budget.min(shuffled.len()));
    shuffled
}

fn select_temporal(
    pairs: &[PairCandidate],
    budget: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let mut candidates: Vec<_> = pairs.iter()
        .filter(|pair| pair.overlap_frames > 0)
        .collect();
    candidates.sort_by_key(|pair| (
        std::cmp::Reverse(pair.overlap_frames),
        pair_tiebreak(seed, pair.left, pair.right),
    ));
    candidates.into_iter()
        .take(budget)
        .map(|pair| (pair.left, pair.right))
        .collect()
}

fn select_routing(
    pairs: &[PairCandidate],
    supports: &[NormalizedSupport],
    budget: usize,
    seed: u64,
) -> Vec<(usize, usize)> {
    let unknown_routing = supports.iter().any(|support| !support.routing_complete);
    let mut candidates: Vec<_> = pairs.iter()
        .filter(|pair| {
            unknown_routing
                || share_routing_domain(
                    &supports[pair.left].domains,
                    &supports[pair.right].domains,
                )
        })
        .collect();
    candidates.sort_by_key(|pair| pair_tiebreak(seed, pair.left, pair.right));
    candidates.into_iter()
        .take(budget)
        .map(|pair| (pair.left, pair.right))
        .collect()
}

fn select_oracle_top_k(
    pairs: &[PairCandidate],
    budget: usize,
    seed: u64,
    threshold: f64,
) -> Vec<(usize, usize)> {
    let mut candidates: Vec<_> = pairs.iter()
        .filter(|pair| pair.contextual_norm > threshold)
        .collect();
    candidates.sort_by_key(|pair| (
        std::cmp::Reverse(pair.contextual_norm.to_bits()),
        pair_tiebreak(seed, pair.left, pair.right),
    ));
    candidates.into_iter()
        .take(budget)
        .map(|pair| (pair.left, pair.right))
        .collect()
}

fn make_row(
    policy: ProbeSelectionPolicy,
    seed: u64,
    requested_budget: usize,
    selected: Vec<(usize, usize)>,
    truth: &BTreeSet<(usize, usize)>,
    event_count: usize,
    exhaustive_pair_count: usize,
) -> ProbePolicyComparisonRow {
    let selected_set: BTreeSet<_> = selected.iter().copied().collect();
    let true_positive_count = selected_set.intersection(truth).count();
    let false_positive_count = selected_set.len().saturating_sub(true_positive_count);
    let false_negative_count = truth.len().saturating_sub(true_positive_count);
    let precision = if selected_set.is_empty() {
        None
    } else {
        Some(true_positive_count as f64 / selected_set.len() as f64)
    };
    let recall = if truth.is_empty() {
        None
    } else {
        Some(true_positive_count as f64 / truth.len() as f64)
    };
    let exhaustive_render_count = 1u128
        .saturating_add(event_count as u128)
        .saturating_add(exhaustive_pair_count as u128);
    let total_render_count = 1u128
        .saturating_add(event_count as u128)
        .saturating_add(selected_set.len() as u128);
    ProbePolicyComparisonRow {
        policy,
        seed,
        requested_pair_budget: requested_budget,
        selected_pair_count: selected_set.len(),
        true_positive_count,
        false_positive_count,
        false_negative_count,
        precision,
        recall,
        total_render_count,
        exhaustive_render_count,
        render_cost_fraction_of_exhaustive:
            total_render_count as f64 / exhaustive_render_count as f64,
        unused_pair_budget: requested_budget.saturating_sub(selected_set.len()),
    }
}

fn summarize_curves(
    rows: &[ProbePolicyComparisonRow],
    recall_target: f64,
) -> Vec<ProbePolicyCurveSummary> {
    let mut keys = BTreeSet::new();
    for row in rows {
        keys.insert((row.policy, row.seed));
    }

    let mut summaries = Vec::with_capacity(keys.len());
    for (policy, seed) in keys {
        let mut points: Vec<(f64, f64)> = rows.iter()
            .filter(|row| row.policy == policy && row.seed == seed)
            .filter_map(|row| row.recall.map(|recall| {
                (row.render_cost_fraction_of_exhaustive, recall)
            }))
            .collect();
        points.sort_by(|left, right| left.0.total_cmp(&right.0));

        // Different requested budgets can have the same actual render cost.
        // Average those observations at the shared x-coordinate instead of
        // selecting the best one, which would bias an underfilled policy upward.
        let mut unique_points: Vec<(f64, f64, usize)> = Vec::with_capacity(points.len());
        for (cost, recall) in points {
            if let Some(last) = unique_points.last_mut() {
                if (last.0 - cost).abs() <= f64::EPSILON {
                    last.1 += recall;
                    last.2 += 1;
                    continue;
                }
            }
            unique_points.push((cost, recall, 1));
        }
        let unique_points: Vec<(f64, f64)> = unique_points.into_iter()
            .map(|(cost, recall_sum, count)| (cost, recall_sum / count as f64))
            .collect();

        let auc = if unique_points.is_empty() {
            None
        } else {
            let area = unique_points.windows(2)
                .map(|window| {
                    let width = window[1].0 - window[0].0;
                    width * (window[0].1 + window[1].1) * 0.5
                })
                .sum::<f64>();
            Some(area)
        };
        let maximum_observed_render_cost_fraction =
            unique_points.last().map(|point| point.0);
        let cost_fraction_to_target_recall = unique_points.iter()
            .find(|(_, recall)| *recall >= recall_target)
            .map(|(cost, _)| *cost);

        summaries.push(ProbePolicyCurveSummary {
            policy,
            seed,
            observed_auc_recall_cost: auc,
            maximum_observed_render_cost_fraction,
            cost_fraction_to_target_recall,
        });
    }
    summaries
}

fn build_comparison(
    factorial_report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    factorial_config: &FactorialReferenceConfig,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    planner_config: &InteractionProbeConfig,
    budgets: &[usize],
    seeds: &[u64],
    metric_config: &InteractionProbeBenchmarkConfig,
) -> Result<ProbePolicyComparisonReport, ProbePolicyComparisonError> {
    if !digest_is_valid(source_score_sha256) || !digest_is_valid(renderer_profile_sha256) {
        return Err(ProbePolicyComparisonError::InvalidDigest);
    }
    if events.len() < 2
        || events.len() != factorial_report.event_count
        || events.len() != event_supports.len()
    {
        return Err(ProbePolicyComparisonError::EventCountMismatch);
    }
    if total_frames != factorial_report.output_frame_count as u64 {
        return Err(ProbePolicyComparisonError::FrameCountMismatch);
    }
    if budgets.is_empty() || seeds.is_empty() || seeds.len() > MAX_COMPARISON_SEEDS {
        return Err(ProbePolicyComparisonError::InvalidConfig(
            "budgets and 1..=256 seeds are required",
        ));
    }

    verify_factorial_ground_truth(factorial_report, events, effect, factorial_config)
        .map_err(ProbePolicyComparisonError::FactorialReference)?;
    let normalized = normalize_supports(event_supports, events.len(), total_frames)?;
    let all = all_pairs(events.len());
    let possible_pair_count = all.len();
    let mut canonical_budgets = budgets.to_vec();
    canonical_budgets.sort_unstable();
    canonical_budgets.dedup();
    if canonical_budgets.iter().any(|budget| *budget > possible_pair_count) {
        return Err(ProbePolicyComparisonError::InvalidConfig(
            "pair budget cannot exceed all possible pairs",
        ));
    }
    let mut canonical_seeds = seeds.to_vec();
    canonical_seeds.sort_unstable();
    canonical_seeds.dedup();
    if canonical_seeds.is_empty() || canonical_seeds.len() > MAX_COMPARISON_SEEDS {
        return Err(ProbePolicyComparisonError::InvalidConfig("invalid seed set"));
    }

    let (mut pair_candidates, global_order_two_count, higher_order_count) =
        contextual_truth(factorial_report, events.len(), metric_config)?;
    for pair in &mut pair_candidates {
        pair.overlap_frames = support_overlap(
            &normalized[pair.left].intervals,
            &normalized[pair.right].intervals,
        );
    }
    let truth: BTreeSet<_> = pair_candidates.iter()
        .filter(|pair| pair.contextual_norm > metric_config.minimum_contextual_pair_pcm_l2_norm)
        .map(|pair| (pair.left, pair.right))
        .collect();

    // Validate the support evidence and planner configuration even when the
    // requested curve contains only the no-probe point.
    let mut validation_config = planner_config.clone();
    validation_config.pair_probe_budget = 1;
    validation_config.routing_group_sample_budget =
        validation_config.routing_group_sample_budget.max(1);
    validation_config.audit_pair_budget = 0;
    validation_config.seed = canonical_seeds[0];
    build_interaction_probe_plan(
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        &validation_config,
    )
    .map_err(ProbePolicyComparisonError::ProbePlan)?;

    let exhaustive_render_count = 1u128
        .saturating_add(events.len() as u128)
        .saturating_add(possible_pair_count as u128);
    let mut rows = Vec::new();

    for seed in &canonical_seeds {
        rows.push(make_row(
            ProbeSelectionPolicy::NoProbe,
            *seed,
            0,
            Vec::new(),
            &truth,
            events.len(),
            possible_pair_count,
        ));

        for budget in &canonical_budgets {
            let k = *budget;
            if k == 0 {
                continue;
            }

            rows.push(make_row(
                ProbeSelectionPolicy::UniformRandom,
                *seed,
                k,
                choose_uniform_random(&all, k, *seed),
                &truth,
                events.len(),
                possible_pair_count,
            ));
            rows.push(make_row(
                ProbeSelectionPolicy::TemporalOnly,
                *seed,
                k,
                select_temporal(&pair_candidates, k, *seed),
                &truth,
                events.len(),
                possible_pair_count,
            ));
            rows.push(make_row(
                ProbeSelectionPolicy::RoutingOnly,
                *seed,
                k,
                select_routing(&pair_candidates, &normalized, k, *seed),
                &truth,
                events.len(),
                possible_pair_count,
            ));
            rows.push(make_row(
                ProbeSelectionPolicy::OracleTopK,
                *seed,
                k,
                select_oracle_top_k(
                    &pair_candidates,
                    k,
                    *seed,
                    metric_config.minimum_contextual_pair_pcm_l2_norm,
                ),
                &truth,
                events.len(),
                possible_pair_count,
            ));

            let mut eval019_config = planner_config.clone();
            eval019_config.pair_probe_budget = k;
            eval019_config.audit_pair_budget = 0;
            eval019_config.seed = *seed;
            eval019_config.routing_group_sample_budget =
                eval019_config.routing_group_sample_budget.max(1).min(k);
            let plan = build_interaction_probe_plan(
                source_score_sha256,
                renderer_profile_sha256,
                total_frames,
                event_supports,
                &eval019_config,
            )
            .map_err(ProbePolicyComparisonError::ProbePlan)?;
            let eval019_pairs = plan.candidate_pairs_selected.iter()
                .map(|pair| (pair.event_a, pair.event_b))
                .collect();
            rows.push(make_row(
                ProbeSelectionPolicy::Eval019,
                *seed,
                k,
                eval019_pairs,
                &truth,
                events.len(),
                possible_pair_count,
            ));
        }

        rows.push(make_row(
            ProbeSelectionPolicy::Exhaustive,
            *seed,
            possible_pair_count,
            all.clone(),
            &truth,
            events.len(),
            possible_pair_count,
        ));
    }

    let support_bytes = serde_json::to_vec(event_supports)
        .map_err(|_| ProbePolicyComparisonError::SerializationFailure)?;
    let recall_target = 0.95;
    let curves = summarize_curves(&rows, recall_target);
    let mut report = ProbePolicyComparisonReport {
        version: PROBE_POLICY_COMPARISON_VERSION.to_owned(),
        factorial_report_sha256: factorial_report.receipt_sha256.clone(),
        source_score_sha256: source_score_sha256.to_owned(),
        renderer_profile_sha256: renderer_profile_sha256.to_owned(),
        support_and_topology_sha256: sha256_hex(&support_bytes),
        factorial_config: factorial_config.clone(),
        planner_config_template: planner_config.clone(),
        metric_config: metric_config.clone(),
        budgets: canonical_budgets,
        seeds: canonical_seeds,
        event_count: events.len(),
        possible_pair_count,
        contextual_positive_pair_count: truth.len(),
        global_order_two_harsanyi_count: global_order_two_count,
        higher_order_harsanyi_count: higher_order_count,
        recall_target,
        rows,
        curves,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = canonical_digest(&report)?;
    if exhaustive_render_count == 0 {
        return Err(ProbePolicyComparisonError::InvalidConfig("invalid render accounting"));
    }
    Ok(report)
}

/// Build a paired policy comparison over the same source events and exact
/// EVAL-020 oracle. Each row is a seed/policy/budget observation; aggregate
/// statistics must resample fixture families, not individual pairs.
#[allow(clippy::too_many_arguments)]
pub fn compare_probe_policies(
    factorial_report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    factorial_config: &FactorialReferenceConfig,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    planner_config: &InteractionProbeConfig,
    budgets: &[usize],
    seeds: &[u64],
    metric_config: &InteractionProbeBenchmarkConfig,
) -> Result<ProbePolicyComparisonReport, ProbePolicyComparisonError> {
    build_comparison(
        factorial_report,
        events,
        effect,
        factorial_config,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        planner_config,
        budgets,
        seeds,
        metric_config,
    )
}

/// Verify the campaign report by rebuilding every policy row from exact inputs.
#[allow(clippy::too_many_arguments)]
pub fn verify_probe_policy_comparison(
    report: &ProbePolicyComparisonReport,
    factorial_report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    factorial_config: &FactorialReferenceConfig,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    planner_config: &InteractionProbeConfig,
    budgets: &[usize],
    seeds: &[u64],
    metric_config: &InteractionProbeBenchmarkConfig,
) -> Result<bool, ProbePolicyComparisonError> {
    let mut unsigned = report.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if receipt.len() != 64 || canonical_digest(&unsigned)? != receipt {
        return Err(ProbePolicyComparisonError::InvalidReceipt);
    }
    let expected = build_comparison(
        factorial_report,
        events,
        effect,
        factorial_config,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        planner_config,
        budgets,
        seeds,
        metric_config,
    )?;
    if &expected != report {
        return Err(ProbePolicyComparisonError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn event_window(
        index: usize,
        value: f64,
        start_frame: usize,
        end_frame: usize,
    ) -> SyntheticEventSignal {
        let mut samples = vec![0.0; 64];
        samples[start_frame..end_frame].fill(value);
        SyntheticEventSignal {
            source_event_index: index,
            samples,
        }
    }

    fn support(index: usize, start: u64, end: u64) -> EventInteractionSupport {
        EventInteractionSupport {
            source_event_index: index,
            acoustic_support: vec![AcousticSupportInterval {
                start_frame: start,
                end_frame: end,
            }],
            routing_domains: Vec::new(),
            routing_complete: true,
            support_evidence_sha256: Some(digest(&format!("support-{index}"))),
        }
    }

    fn planner_config() -> InteractionProbeConfig {
        InteractionProbeConfig {
            pair_probe_budget: 1,
            routing_group_sample_budget: 1,
            audit_pair_budget: 0,
            max_sampling_attempts: 10_000,
            seed: 17,
            max_temporal_candidate_pairs: 128,
        }
    }

    fn fixture() -> (
        FactorialGroundTruthReport,
        Vec<SyntheticEventSignal>,
        SyntheticEffect,
        FactorialReferenceConfig,
        Vec<EventInteractionSupport>,
    ) {
        let events = vec![
            event_window(0, 0.2, 0, 20),
            event_window(1, 0.3, 16, 30),
            event_window(2, 0.1, 31, 34),
            event_window(3, 0.1, 35, 38),
            event_window(4, 0.1, 39, 42),
            event_window(5, 0.1, 43, 46),
            event_window(6, 0.1, 47, 50),
            event_window(7, 0.1, 51, 54),
        ];
        let effect = SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.5,
            cubic_coefficient: 0.0,
        };
        let config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let report = crate::factorial_ground_truth::build_factorial_ground_truth(
            &events,
            &effect,
            &config,
        ).unwrap();
        let supports = vec![
            support(0, 0, 20),
            support(1, 16, 30),
            support(2, 31, 34),
            support(3, 35, 38),
            support(4, 39, 42),
            support(5, 43, 46),
            support(6, 47, 50),
            support(7, 51, 54),
        ];
        (report, events, effect, config, supports)
    }

    fn find_row<'a>(
        report: &'a ProbePolicyComparisonReport,
        policy: ProbeSelectionPolicy,
        seed: u64,
        budget: usize,
    ) -> &'a ProbePolicyComparisonRow {
        report.rows.iter()
            .find(|row| row.policy == policy && row.seed == seed && row.requested_pair_budget == budget)
            .expect("comparison row")
    }

    #[test]
    fn compares_policies_at_equal_budgets_and_records_actual_cost() {
        let (factorial, events, effect, config, supports) = fixture();
        let planner = planner_config();
        let metric = InteractionProbeBenchmarkConfig::default();
        let score = digest("score");
        let renderer = digest("renderer");
        let seeds = [11u64, 29, 47];
        let report = compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &score,
            &renderer,
            64,
            &supports,
            &planner,
            &[0, 1, 4, 28],
            &seeds,
            &metric,
        ).unwrap();

        assert_eq!(report.contextual_positive_pair_count, 1);
        assert_eq!(report.possible_pair_count, 28);
        for seed in seeds {
            let eval019 = find_row(&report, ProbeSelectionPolicy::Eval019, seed, 1);
            assert_eq!(eval019.selected_pair_count, 1);
            assert_eq!(eval019.true_positive_count, 1);
            assert_eq!(eval019.recall, Some(1.0));

            let temporal = find_row(&report, ProbeSelectionPolicy::TemporalOnly, seed, 1);
            assert_eq!(temporal.true_positive_count, 1);
            assert_eq!(temporal.recall, Some(1.0));

            let routing = find_row(&report, ProbeSelectionPolicy::RoutingOnly, seed, 1);
            assert_eq!(routing.selected_pair_count, 0);
            assert_eq!(routing.precision, None);
            assert_eq!(routing.recall, Some(0.0));

            let oracle = find_row(&report, ProbeSelectionPolicy::OracleTopK, seed, 1);
            assert_eq!(oracle.selected_pair_count, 1);
            assert_eq!(oracle.recall, Some(1.0));

            let exhaustive = find_row(&report, ProbeSelectionPolicy::Exhaustive, seed, 28);
            assert_eq!(exhaustive.selected_pair_count, 28);
            assert_eq!(exhaustive.recall, Some(1.0));
            assert_eq!(exhaustive.render_cost_fraction_of_exhaustive, 1.0);
        }
        let no_probe = find_row(&report, ProbeSelectionPolicy::NoProbe, seeds[0], 0);
        assert_eq!(no_probe.selected_pair_count, 0);
        assert_eq!(no_probe.recall, Some(0.0));
        let eval019_curve = report.curves.iter()
            .find(|curve| curve.policy == ProbeSelectionPolicy::Eval019 && curve.seed == seeds[0])
            .unwrap();
        assert_eq!(eval019_curve.cost_fraction_to_target_recall, Some(10.0 / 37.0));
        let exhaustive_curve = report.curves.iter()
            .find(|curve| curve.policy == ProbeSelectionPolicy::Exhaustive && curve.seed == seeds[0])
            .unwrap();
        assert_eq!(exhaustive_curve.maximum_observed_render_cost_fraction, Some(1.0));
        assert!(verify_probe_policy_comparison(
            &report,
            &factorial,
            &events,
            &effect,
            &config,
            &score,
            &renderer,
            64,
            &supports,
            &planner,
            &[0, 1, 4, 28],
            &seeds,
            &metric,
        ).unwrap());
    }

    #[test]
    fn random_sampling_is_nested_and_seed_reproducible() {
        let (factorial, events, effect, config, supports) = fixture();
        let planner = planner_config();
        let metric = InteractionProbeBenchmarkConfig::default();
        let seeds = [101u64, 202];
        let run = || compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &planner,
            &[1, 2, 4, 8],
            &seeds,
            &metric,
        ).unwrap();
        let first = run();
        let second = run();
        assert_eq!(first, second);
        for seed in seeds {
            let one = find_row(&first, ProbeSelectionPolicy::UniformRandom, seed, 1);
            let two = find_row(&first, ProbeSelectionPolicy::UniformRandom, seed, 2);
            let four = find_row(&first, ProbeSelectionPolicy::UniformRandom, seed, 4);
            assert!(one.recall.unwrap() <= two.recall.unwrap());
            assert!(two.recall.unwrap() <= four.recall.unwrap());
        }
    }

    #[test]
    fn incomplete_routing_never_prunes_routing_only_candidates() {
        let (factorial, events, effect, config, mut supports) = fixture();
        supports[0].routing_complete = false;
        let report = compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &planner_config(),
            &[2],
            &[7],
            &InteractionProbeBenchmarkConfig::default(),
        ).unwrap();
        let routing = find_row(&report, ProbeSelectionPolicy::RoutingOnly, 7, 2);
        assert_eq!(routing.selected_pair_count, 2);
    }

    #[test]
    fn comparison_rejects_missing_support_evidence_for_empty_and_nonempty_intervals() {
        for empty_support in [false, true] {
            let (factorial, events, effect, config, mut supports) = fixture();
            supports[0].support_evidence_sha256 = None;
            if empty_support {
                supports[0].acoustic_support.clear();
            }
            assert_eq!(
                compare_probe_policies(
                    &factorial,
                    &events,
                    &effect,
                    &config,
                    &digest("score"),
                    &digest("renderer"),
                    64,
                    &supports,
                    &planner_config(),
                    &[1],
                    &[41],
                    &InteractionProbeBenchmarkConfig::default(),
                ).unwrap_err(),
                ProbePolicyComparisonError::MissingSupportEvidence,
            );
        }
    }

    #[test]
    fn comparison_keeps_contextual_pairs_separate_from_global_harsanyi_order() {
        let events = vec![
            SyntheticEventSignal { source_event_index: 0, samples: vec![0.2; 64] },
            SyntheticEventSignal { source_event_index: 1, samples: vec![0.3; 64] },
            SyntheticEventSignal { source_event_index: 2, samples: vec![0.4; 64] },
        ];
        let effect = SyntheticEffect::ThreeWayProduct {
            event_indices: [0, 1, 2],
            coefficient: 0.5,
        };
        let config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let factorial = crate::factorial_ground_truth::build_factorial_ground_truth(
            &events,
            &effect,
            &config,
        ).unwrap();
        let supports = vec![
            support(0, 0, 64),
            support(1, 0, 64),
            support(2, 0, 64),
        ];
        let report = compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &planner_config(),
            &[1, 3],
            &[31],
            &InteractionProbeBenchmarkConfig::default(),
        ).unwrap();

        assert_eq!(report.contextual_positive_pair_count, 3);
        assert_eq!(report.global_order_two_harsanyi_count, 0);
        assert_eq!(report.higher_order_harsanyi_count, 1);
        let oracle = find_row(&report, ProbeSelectionPolicy::OracleTopK, 31, 3);
        assert_eq!(oracle.selected_pair_count, 3);
        assert_eq!(oracle.recall, Some(1.0));
    }

    #[test]
    fn support_horizon_uses_fir_output_frames_not_source_frames() {
        let events = vec![
            event_window(0, 0.2, 60, 64),
            event_window(1, 0.3, 60, 64),
        ];
        let effect = SyntheticEffect::LinearFir {
            impulse_response: vec![1.0, 0.5, 0.25, 0.125],
        };
        let config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let factorial = crate::factorial_ground_truth::build_factorial_ground_truth(
            &events,
            &effect,
            &config,
        )
        .unwrap();
        assert_eq!(factorial.input_frame_count, 64);
        assert_eq!(factorial.output_frame_count, 67);

        let supports = vec![support(0, 60, 67), support(1, 60, 67)];
        let planner = planner_config();
        let metric = InteractionProbeBenchmarkConfig::default();
        let score = digest("score");
        let renderer = digest("renderer");
        let seeds = [37u64];
        let budgets = [1usize];
        let report = compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &score,
            &renderer,
            67,
            &supports,
            &planner,
            &budgets,
            &seeds,
            &metric,
        )
        .unwrap();

        assert!(verify_probe_policy_comparison(
            &report,
            &factorial,
            &events,
            &effect,
            &config,
            &score,
            &renderer,
            67,
            &supports,
            &planner,
            &budgets,
            &seeds,
            &metric,
        )
        .unwrap());
        assert_eq!(
            compare_probe_policies(
                &factorial,
                &events,
                &effect,
                &config,
                &score,
                &renderer,
                64,
                &supports,
                &planner,
                &budgets,
                &seeds,
                &metric,
            )
            .unwrap_err(),
            ProbePolicyComparisonError::FrameCountMismatch,
        );

        let out_of_range = vec![support(0, 60, 68), support(1, 60, 67)];
        assert_eq!(
            compare_probe_policies(
                &factorial,
                &events,
                &effect,
                &config,
                &score,
                &renderer,
                67,
                &out_of_range,
                &planner,
                &budgets,
                &seeds,
                &metric,
            )
            .unwrap_err(),
            ProbePolicyComparisonError::InvalidSupport,
        );
    }

    #[test]
    fn report_tampering_is_rejected() {
        let (factorial, events, effect, config, supports) = fixture();
        let planner = planner_config();
        let metric = InteractionProbeBenchmarkConfig::default();
        let score = digest("score");
        let renderer = digest("renderer");
        let seeds = [1u64];
        let mut report = compare_probe_policies(
            &factorial,
            &events,
            &effect,
            &config,
            &score,
            &renderer,
            64,
            &supports,
            &planner,
            &[1],
            &seeds,
            &metric,
        ).unwrap();
        report.rows[0].selected_pair_count += 1;
        assert_eq!(
            verify_probe_policy_comparison(
                &report,
                &factorial,
                &events,
                &effect,
                &config,
                &score,
                &renderer,
                64,
                &supports,
                &planner,
                &[1],
                &seeds,
                &metric,
            ).unwrap_err(),
            ProbePolicyComparisonError::InvalidReceipt,
        );
    }
}

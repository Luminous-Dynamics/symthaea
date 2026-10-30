// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Oracle-backed evaluation of sparse pairwise interaction probe plans.
//!
//! EVAL-019 proposes which pairs to measure. EVAL-020 supplies exhaustive
//! small-scene truth. This module joins them without allowing a probe plan to
//! certify unmeasured pairs: it reports candidate recall/precision, audit
//! discoveries, higher-order interaction mass, and render cost against the
//! exhaustive pairwise reference.

use crate::factorial_ground_truth::{
    FactorialGroundTruthReport, FactorialReferenceConfig, FactorialReferenceError,
    SyntheticEffect, SyntheticEventSignal, verify_factorial_ground_truth,
};
use crate::interaction_probe_plan::{
    EventInteractionSupport, InteractionProbeConfig, InteractionProbePlan,
    InteractionProbePlanError, verify_interaction_probe_plan,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const INTERACTION_PROBE_BENCHMARK_VERSION: &str =
    "mel-mr-eval-023-oracle-probe-audit-v0.1";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-023-oracle-probe-audit-v1\0";

/// PCM-space Harsanyi interaction magnitudes below these floors are treated
/// as numerical/operational nulls for this benchmark fixture. Thresholds are
/// explicit because L2 magnitude scales with fixture length and signal level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionProbeBenchmarkConfig {
    pub minimum_pair_pcm_l2_norm: f64,
    pub minimum_higher_order_pcm_l2_norm: f64,
}

impl Default for InteractionProbeBenchmarkConfig {
    fn default() -> Self {
        Self {
            minimum_pair_pcm_l2_norm: 1.0e-9,
            minimum_higher_order_pcm_l2_norm: 1.0e-9,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionProbeBenchmarkReport {
    pub version: String,
    pub factorial_report_sha256: String,
    pub probe_plan_sha256: String,
    pub config: InteractionProbeBenchmarkConfig,
    pub event_count: usize,
    pub possible_pair_count: u128,

    /// Exact pairwise PCM interactions from EVAL-020, using the configured floor.
    pub ground_truth_interacting_pair_count: usize,
    /// Candidate-policy output only; audit-only pairs are kept in separate fields.
    pub candidate_pair_count: usize,
    pub candidate_true_positive_count: usize,
    pub candidate_false_positive_count: usize,
    pub candidate_false_negative_count: usize,
    /// None means the denominator is empty, not a perfect score.
    pub candidate_precision: Option<f64>,
    pub candidate_recall: Option<f64>,

    pub audit_pair_count: usize,
    pub audit_true_positive_count: usize,
    pub audit_positive_rate: Option<f64>,
    pub selected_pair_count: usize,
    pub selected_true_positive_count: usize,
    pub selected_pair_recall: Option<f64>,
    pub unselected_ground_truth_pair_count: usize,

    /// Pairwise finite differences with all other events held present, matching EVAL-018.
    pub conditional_pair_total_pcm_l2_norm: f64,
    /// Unconditional global order-2 Harsanyi structure, reported separately.
    pub global_order_two_harsanyi_interaction_count: usize,
    pub global_order_two_harsanyi_total_pcm_l2_norm: f64,
    pub nonzero_higher_order_interaction_count: usize,
    pub higher_order_total_pcm_l2_norm: f64,
    /// None when the global order-2 norm is zero; avoids infinity/NaN in receipts.
    pub higher_order_to_global_order_two_l2_ratio: Option<f64>,

    pub planned_render_count: u128,
    pub exhaustive_pairwise_render_count: u128,
    pub render_cost_fraction_of_exhaustive: f64,
    pub selected_pair_fraction: f64,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionProbeBenchmarkError {
    InvalidConfig(&'static str),
    EventCountMismatch,
    FrameCountMismatch,
    InvalidProbeSelection,
    PlanAccountingMismatch,
    FactorialReference(FactorialReferenceError),
    ProbePlan(InteractionProbePlanError),
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for InteractionProbeBenchmarkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "interaction-probe benchmark validation failed: {self:?}")
    }
}

impl std::error::Error for InteractionProbeBenchmarkError {}

fn canonical_digest<T: Serialize>(
    value: &T,
) -> Result<String, InteractionProbeBenchmarkError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| InteractionProbeBenchmarkError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn pair_key(left: usize, right: usize) -> (usize, usize) {
    if left < right { (left, right) } else { (right, left) }
}

fn collect_pairs(
    pairs: &[crate::interaction_probe_plan::InteractionProbePair],
    event_count: usize,
) -> Result<BTreeSet<(usize, usize)>, InteractionProbeBenchmarkError> {
    let mut selected = BTreeSet::new();
    for pair in pairs {
        if pair.event_a >= pair.event_b || pair.event_b >= event_count {
            return Err(InteractionProbeBenchmarkError::InvalidProbeSelection);
        }
        if !selected.insert(pair_key(pair.event_a, pair.event_b)) {
            return Err(InteractionProbeBenchmarkError::InvalidProbeSelection);
        }
    }
    Ok(selected)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

fn contextual_pair_interaction_l2_norm(
    report: &FactorialGroundTruthReport,
    left: usize,
    right: usize,
) -> Result<f64, InteractionProbeBenchmarkError> {
    let all_mask = (1u16 << report.event_count) - 1;
    let mask_without_left = all_mask & !(1u16 << left);
    let mask_without_right = all_mask & !(1u16 << right);
    let mask_without_both = all_mask & !(1u16 << left) & !(1u16 << right);

    let coalition_pcm = |mask: u16| {
        report.coalitions
            .iter()
            .find(|coalition| coalition.coalition_mask == mask)
            .map(|coalition| coalition.pcm_samples.as_slice())
            .ok_or(InteractionProbeBenchmarkError::ReplayMismatch)
    };
    let full = coalition_pcm(all_mask)?;
    let without_left = coalition_pcm(mask_without_left)?;
    let without_right = coalition_pcm(mask_without_right)?;
    let without_both = coalition_pcm(mask_without_both)?;
    if full.len() != without_left.len()
        || full.len() != without_right.len()
        || full.len() != without_both.len()
    {
        return Err(InteractionProbeBenchmarkError::ReplayMismatch);
    }

    let squared_norm = full.iter()
        .zip(without_left)
        .zip(without_right)
        .zip(without_both)
        .map(|(((full, without_left), without_right), without_both)| {
            let residual = *full - *without_left - *without_right + *without_both;
            residual * residual
        })
        .sum::<f64>();
    Ok(squared_norm.sqrt())
}

fn report_digest_valid(
    report: &InteractionProbeBenchmarkReport,
) -> Result<bool, InteractionProbeBenchmarkError> {
    let mut unsigned = report.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if receipt.len() != 64 {
        return Ok(false);
    }
    Ok(canonical_digest(&unsigned)? == receipt)
}

/// Evaluate a replay-verified sparse plan against exact exhaustive coalition
/// truth. total_frames and support intervals use the same frame domain as the
/// source-isolated signals supplied to EVAL-020.
///
/// Candidate metrics exclude the independently sampled audit pairs. The
/// measured metrics include both. Neither metric treats an unprobed pair as
/// a negative example; its status is known only because this synthetic
/// benchmark has the EVAL-020 exhaustive oracle.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_interaction_probe_plan(
    factorial_report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    factorial_config: &FactorialReferenceConfig,
    probe_plan: &InteractionProbePlan,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    probe_config: &InteractionProbeConfig,
    benchmark_config: &InteractionProbeBenchmarkConfig,
) -> Result<InteractionProbeBenchmarkReport, InteractionProbeBenchmarkError> {
    if !benchmark_config.minimum_pair_pcm_l2_norm.is_finite()
        || benchmark_config.minimum_pair_pcm_l2_norm <= 0.0
        || !benchmark_config.minimum_higher_order_pcm_l2_norm.is_finite()
        || benchmark_config.minimum_higher_order_pcm_l2_norm <= 0.0
    {
        return Err(InteractionProbeBenchmarkError::InvalidConfig(
            "PCM interaction floors must be finite and positive",
        ));
    }

    if events.len() < 2
        || events.len() != factorial_report.event_count
        || events.len() != probe_plan.event_count
        || events.len() != event_supports.len()
    {
        return Err(InteractionProbeBenchmarkError::EventCountMismatch);
    }
    if total_frames != factorial_report.input_frame_count as u64 {
        return Err(InteractionProbeBenchmarkError::FrameCountMismatch);
    }

    verify_factorial_ground_truth(
        factorial_report,
        events,
        effect,
        factorial_config,
    )
    .map_err(InteractionProbeBenchmarkError::FactorialReference)?;
    verify_interaction_probe_plan(
        probe_plan,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        probe_config,
    )
    .map_err(InteractionProbeBenchmarkError::ProbePlan)?;

    let candidate_pairs = collect_pairs(
        &probe_plan.candidate_pairs_selected,
        events.len(),
    )?;
    let audit_pairs = collect_pairs(&probe_plan.audit_pairs, events.len())?;
    if candidate_pairs.iter().any(|pair| audit_pairs.contains(pair)) {
        return Err(InteractionProbeBenchmarkError::InvalidProbeSelection);
    }
    let selected_pairs: BTreeSet<_> = candidate_pairs
        .union(&audit_pairs)
        .copied()
        .collect();

    let possible_pair_count = (events.len() as u128)
        .saturating_mul((events.len().saturating_sub(1)) as u128)
        / 2;
    let selected_pair_count = selected_pairs.len();
    if probe_plan.all_possible_pair_count != possible_pair_count
        || probe_plan.selected_pair_count != selected_pair_count
        || probe_plan.unprobed_pair_count
            != possible_pair_count.saturating_sub(selected_pair_count as u128)
    {
        return Err(InteractionProbeBenchmarkError::PlanAccountingMismatch);
    }

    // EVAL-018 probes the pair finite difference with all other source events
    // retained. This is a contextual effect and can include higher-order terms.
    // Do not confuse it with the unconditional order-2 Harsanyi dividend below.
    let mut ground_truth_pairs = BTreeSet::new();
    let mut conditional_pair_l2_squared = 0.0f64;
    for left in 0..events.len() {
        for right in (left + 1)..events.len() {
            let interaction_norm =
                contextual_pair_interaction_l2_norm(factorial_report, left, right)?;
            conditional_pair_l2_squared += interaction_norm * interaction_norm;
            if interaction_norm > benchmark_config.minimum_pair_pcm_l2_norm {
                ground_truth_pairs.insert((left, right));
            }
        }
    }

    let mut global_order_two_harsanyi_interaction_count = 0usize;
    let mut global_order_two_l2_squared = 0.0f64;
    let mut higher_order_l2_squared = 0.0f64;
    let mut nonzero_higher_order_interaction_count = 0usize;
    for dividend in &factorial_report.dividends {
        if dividend.order == 2 {
            global_order_two_l2_squared += dividend.pcm_l2_norm * dividend.pcm_l2_norm;
            if dividend.pcm_l2_norm > benchmark_config.minimum_pair_pcm_l2_norm {
                global_order_two_harsanyi_interaction_count += 1;
            }
        } else if dividend.order >= 3 {
            higher_order_l2_squared += dividend.pcm_l2_norm * dividend.pcm_l2_norm;
            if dividend.pcm_l2_norm > benchmark_config.minimum_higher_order_pcm_l2_norm {
                nonzero_higher_order_interaction_count += 1;
            }
        }
    }

    let candidate_true_positive_count = candidate_pairs
        .intersection(&ground_truth_pairs)
        .count();
    let candidate_false_positive_count =
        candidate_pairs.len().saturating_sub(candidate_true_positive_count);
    let candidate_false_negative_count =
        ground_truth_pairs.len().saturating_sub(candidate_true_positive_count);
    let audit_true_positive_count = audit_pairs
        .intersection(&ground_truth_pairs)
        .count();
    let selected_true_positive_count = selected_pairs
        .intersection(&ground_truth_pairs)
        .count();
    let unselected_ground_truth_pair_count =
        ground_truth_pairs.len().saturating_sub(selected_true_positive_count);

    let planned_render_count = probe_plan.estimated_render_count_with_cached_single_removals;
    let exhaustive_pairwise_render_count = 1u128
        .saturating_add(events.len() as u128)
        .saturating_add(possible_pair_count);
    let expected_planned_render_count = 1u128
        .saturating_add(events.len() as u128)
        .saturating_add(selected_pair_count as u128);
    if planned_render_count != expected_planned_render_count
        || exhaustive_pairwise_render_count == 0
        || planned_render_count > exhaustive_pairwise_render_count
    {
        return Err(InteractionProbeBenchmarkError::PlanAccountingMismatch);
    }

    let conditional_pair_total_pcm_l2_norm = conditional_pair_l2_squared.sqrt();
    let global_order_two_harsanyi_total_pcm_l2_norm = global_order_two_l2_squared.sqrt();
    let higher_order_total_pcm_l2_norm = higher_order_l2_squared.sqrt();
    let higher_order_to_global_order_two_l2_ratio =
        if global_order_two_harsanyi_total_pcm_l2_norm > 0.0 {
            Some(higher_order_total_pcm_l2_norm / global_order_two_harsanyi_total_pcm_l2_norm)
        } else {
            None
        };

    let mut report = InteractionProbeBenchmarkReport {
        version: INTERACTION_PROBE_BENCHMARK_VERSION.to_owned(),
        factorial_report_sha256: factorial_report.receipt_sha256.clone(),
        probe_plan_sha256: probe_plan.receipt_sha256.clone(),
        config: benchmark_config.clone(),
        event_count: events.len(),
        possible_pair_count,
        ground_truth_interacting_pair_count: ground_truth_pairs.len(),
        candidate_pair_count: candidate_pairs.len(),
        candidate_true_positive_count,
        candidate_false_positive_count,
        candidate_false_negative_count,
        candidate_precision: ratio(candidate_true_positive_count, candidate_pairs.len()),
        candidate_recall: ratio(candidate_true_positive_count, ground_truth_pairs.len()),
        audit_pair_count: audit_pairs.len(),
        audit_true_positive_count,
        audit_positive_rate: ratio(audit_true_positive_count, audit_pairs.len()),
        selected_pair_count,
        selected_true_positive_count,
        selected_pair_recall: ratio(selected_true_positive_count, ground_truth_pairs.len()),
        unselected_ground_truth_pair_count,
        conditional_pair_total_pcm_l2_norm,
        global_order_two_harsanyi_interaction_count,
        global_order_two_harsanyi_total_pcm_l2_norm,
        nonzero_higher_order_interaction_count,
        higher_order_total_pcm_l2_norm,
        higher_order_to_global_order_two_l2_ratio,
        planned_render_count,
        exhaustive_pairwise_render_count,
        render_cost_fraction_of_exhaustive:
            planned_render_count as f64 / exhaustive_pairwise_render_count as f64,
        selected_pair_fraction: selected_pair_count as f64 / possible_pair_count as f64,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = canonical_digest(&report)?;
    Ok(report)
}

/// Re-evaluate from the exact oracle inputs and reject report or input drift.
#[allow(clippy::too_many_arguments)]
pub fn verify_interaction_probe_benchmark(
    report: &InteractionProbeBenchmarkReport,
    factorial_report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    factorial_config: &FactorialReferenceConfig,
    probe_plan: &InteractionProbePlan,
    source_score_sha256: &str,
    renderer_profile_sha256: &str,
    total_frames: u64,
    event_supports: &[EventInteractionSupport],
    probe_config: &InteractionProbeConfig,
    benchmark_config: &InteractionProbeBenchmarkConfig,
) -> Result<bool, InteractionProbeBenchmarkError> {
    if !report_digest_valid(report)? {
        return Err(InteractionProbeBenchmarkError::InvalidReceipt);
    }
    let expected = evaluate_interaction_probe_plan(
        factorial_report,
        events,
        effect,
        factorial_config,
        probe_plan,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        event_supports,
        probe_config,
        benchmark_config,
    )?;
    if &expected != report {
        return Err(InteractionProbeBenchmarkError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interaction_probe_plan::{
        AcousticSupportInterval, build_interaction_probe_plan,
    };

    fn digest(label: &str) -> String {
        format!("{:x}", Sha256::digest(label.as_bytes()))
    }

    fn source_event(index: usize, value: f64) -> SyntheticEventSignal {
        SyntheticEventSignal {
            source_event_index: index,
            samples: vec![value; 64],
        }
    }

    fn source_event_window(
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

    fn support(
        index: usize,
        start_frame: u64,
        end_frame: u64,
    ) -> EventInteractionSupport {
        EventInteractionSupport {
            source_event_index: index,
            acoustic_support: vec![AcousticSupportInterval {
                start_frame,
                end_frame,
            }],
            routing_domains: Vec::new(),
            routing_complete: true,
            support_evidence_sha256: Some(digest(&format!("support-{index}"))),
        }
    }

    fn probe_config(audit_pair_budget: usize) -> InteractionProbeConfig {
        InteractionProbeConfig {
            pair_probe_budget: 8,
            routing_group_sample_budget: 8,
            audit_pair_budget,
            max_sampling_attempts: 10_000,
            seed: 0x0230_2026,
            max_temporal_candidate_pairs: 128,
        }
    }

    fn setup(
        events: &[SyntheticEventSignal],
        effect: &SyntheticEffect,
        supports: &[EventInteractionSupport],
        cfg: &InteractionProbeConfig,
    ) -> (FactorialGroundTruthReport, InteractionProbePlan) {
        let factorial_config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let factorial =
            crate::factorial_ground_truth::build_factorial_ground_truth(
                events,
                effect,
                &factorial_config,
            )
            .unwrap();
        let plan = build_interaction_probe_plan(
            &digest("score"),
            &digest("renderer"),
            64,
            supports,
            cfg,
        )
        .unwrap();
        (factorial, plan)
    }

    #[test]
    fn sparse_pair_recall_and_render_savings_are_measured_against_exhaustive_truth() {
        let events = vec![
            source_event_window(0, 0.2, 0, 20),
            source_event_window(1, 0.3, 16, 30),
            source_event_window(2, 0.1, 31, 34),
            source_event_window(3, 0.1, 35, 38),
            source_event_window(4, 0.1, 39, 42),
            source_event_window(5, 0.1, 43, 46),
            source_event_window(6, 0.1, 47, 50),
            source_event_window(7, 0.1, 51, 54),
        ];
        let effect = SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.5,
            cubic_coefficient: 0.0,
        };
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
        let cfg = probe_config(0);
        let (factorial, plan) = setup(&events, &effect, &supports, &cfg);
        let factorial_config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let result = evaluate_interaction_probe_plan(
            &factorial,
            &events,
            &effect,
            &factorial_config,
            &plan,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &cfg,
            &InteractionProbeBenchmarkConfig::default(),
        )
        .unwrap();

        assert_eq!(result.ground_truth_interacting_pair_count, 1);
        assert_eq!(result.candidate_pair_count, 1);
        assert_eq!(result.candidate_true_positive_count, 1);
        assert_eq!(result.candidate_false_positive_count, 0);
        assert_eq!(result.candidate_false_negative_count, 0);
        assert_eq!(result.candidate_precision, Some(1.0));
        assert_eq!(result.candidate_recall, Some(1.0));
        assert_eq!(result.selected_pair_recall, Some(1.0));
        assert_eq!(result.unselected_ground_truth_pair_count, 0);
        assert_eq!(result.planned_render_count, 10);
        assert_eq!(result.exhaustive_pairwise_render_count, 37);
        assert!((result.render_cost_fraction_of_exhaustive - 10.0 / 37.0).abs() < 1.0e-12);
        assert_eq!(result.nonzero_higher_order_interaction_count, 0);
        assert!(verify_interaction_probe_benchmark(
            &result,
            &factorial,
            &events,
            &effect,
            &factorial_config,
            &plan,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &cfg,
            &InteractionProbeBenchmarkConfig::default(),
        )
        .unwrap());
    }

    #[test]
    fn pairwise_probe_report_exposes_a_pure_higher_order_blind_spot() {
        let events = vec![
            source_event(0, 0.2),
            source_event(1, 0.3),
            source_event(2, 0.4),
        ];
        let effect = SyntheticEffect::ThreeWayProduct {
            event_indices: [0, 1, 2],
            coefficient: 0.5,
        };
        let supports = vec![
            support(0, 0, 64),
            support(1, 0, 64),
            support(2, 0, 64),
        ];
        let cfg = probe_config(0);
        let (factorial, plan) = setup(&events, &effect, &supports, &cfg);
        let result = evaluate_interaction_probe_plan(
            &factorial,
            &events,
            &effect,
            &FactorialReferenceConfig {
                sample_rate_hz: 48_000,
                spectral_window_frames: 64,
            },
            &plan,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &cfg,
            &InteractionProbeBenchmarkConfig::default(),
        )
        .unwrap();

        assert_eq!(result.ground_truth_interacting_pair_count, 0);
        assert_eq!(result.ground_truth_interacting_pair_count, 3);
        assert_eq!(result.candidate_pair_count, 3);
        assert_eq!(result.candidate_true_positive_count, 3);
        assert_eq!(result.candidate_false_positive_count, 0);
        assert_eq!(result.candidate_recall, Some(1.0));
        assert_eq!(result.nonzero_higher_order_interaction_count, 1);
        assert!(result.higher_order_total_pcm_l2_norm > 0.0);
        assert_eq!(result.global_order_two_harsanyi_interaction_count, 0);
        assert_eq!(result.higher_order_to_global_order_two_l2_ratio, None);
    }

    #[test]
    fn benchmark_receipt_rejects_report_tampering() {
        let events = vec![
            source_event_window(0, 0.2, 0, 20),
            source_event_window(1, 0.3, 16, 30),
        ];
        let effect = SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.5,
            cubic_coefficient: 0.0,
        };
        let supports = vec![support(0, 0, 20), support(1, 16, 30)];
        let cfg = probe_config(0);
        let (factorial, plan) = setup(&events, &effect, &supports, &cfg);
        let factorial_config = FactorialReferenceConfig {
            sample_rate_hz: 48_000,
            spectral_window_frames: 64,
        };
        let mut result = evaluate_interaction_probe_plan(
            &factorial,
            &events,
            &effect,
            &factorial_config,
            &plan,
            &digest("score"),
            &digest("renderer"),
            64,
            &supports,
            &cfg,
            &InteractionProbeBenchmarkConfig::default(),
        )
        .unwrap();
        result.candidate_false_positive_count += 1;
        assert_eq!(
            verify_interaction_probe_benchmark(
                &result,
                &factorial,
                &events,
                &effect,
                &factorial_config,
                &plan,
                &digest("score"),
                &digest("renderer"),
                64,
                &supports,
                &cfg,
                &InteractionProbeBenchmarkConfig::default(),
            )
            .unwrap_err(),
            InteractionProbeBenchmarkError::InvalidReceipt
        );
    }
}

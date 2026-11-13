// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired, multi-family campaign and composition-clustered analysis for
//! MEL-MR-EVAL-026.
//!
//! Every independently generated composition receives exact EVAL-020
//! factorial truth and a separately generated EVAL-019 support/routing view.
//! EVAL-024 policies are evaluated on the same composition, seeds and budget
//! schedule. Uncertainty is resampled over composition clusters, never over
//! seed/budget rows as if they were independent scenes.
//!
//! This is a synthetic experiment campaign. It does not model a real DAW,
//! FluidSynth routing graph, listener preference, or production audio quality.

use crate::factorial_ground_truth::{
    FactorialGroundTruthReport, FactorialReferenceConfig, FactorialReferenceError, SyntheticEffect,
    SyntheticEventSignal, build_factorial_ground_truth,
};
use crate::interaction_probe_benchmark::InteractionProbeBenchmarkConfig;
use crate::interaction_probe_plan::{
    AcousticSupportInterval, EventInteractionSupport, InteractionProbeConfig,
};
use crate::probe_policy_comparison::{
    ProbePolicyComparisonError, ProbePolicyComparisonRow, ProbePolicyCurveSummary,
    ProbeSelectionPolicy, compare_probe_policies,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PROBE_POLICY_CAMPAIGN_VERSION: &str = "mel-mr-eval-026-paired-multifamily-campaign-v0.1";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-026-paired-multifamily-campaign-v1\0";
const MAX_COMPOSITIONS: usize = 512;
const MAX_BOOTSTRAP_REPLICATES: usize = 20_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionFamily {
    AdditiveNull,
    SparsePairwise,
    Saturation,
    PureThirdOrder,
    LinearFirTail,
    DenseOverlap,
    ClusteredOverlap,
    SharedRouteDisjoint,
    UnderstatedSupport,
    IncompleteRouting,
}

impl InteractionFamily {
    pub const ALL: [Self; 10] = [
        Self::AdditiveNull,
        Self::SparsePairwise,
        Self::Saturation,
        Self::PureThirdOrder,
        Self::LinearFirTail,
        Self::DenseOverlap,
        Self::ClusteredOverlap,
        Self::SharedRouteDisjoint,
        Self::UnderstatedSupport,
        Self::IncompleteRouting,
    ];

    fn split(self) -> CampaignSplit {
        match self {
            Self::PureThirdOrder => CampaignSplit::HeldOutInteractionFamily,
            Self::IncompleteRouting => CampaignSplit::HeldOutRoutingTopology,
            _ => CampaignSplit::Development,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignSplit {
    Development,
    HeldOutInteractionFamily,
    HeldOutRoutingTopology,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignScope {
    Development,
    HeldOutInteractionFamily,
    HeldOutRoutingTopology,
    AllFixtures,
    Family(InteractionFamily),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePolicyCampaignConfig {
    pub families: Vec<InteractionFamily>,
    /// Must be at least seven so every non-held-out family covers 2..=8 events.
    pub compositions_per_family: usize,
    pub frame_count: usize,
    pub seeds: Vec<u64>,
    /// Requested pair-probe budgets. Actual selected counts are retained separately.
    pub pair_budgets: Vec<usize>,
    pub planner_config: InteractionProbeConfig,
    pub factorial_config: FactorialReferenceConfig,
    pub metric_config: InteractionProbeBenchmarkConfig,
    pub recall_target: f64,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
}

impl Default for ProbePolicyCampaignConfig {
    fn default() -> Self {
        let mut planner_config = InteractionProbeConfig::default();
        planner_config.pair_probe_budget = 28;
        planner_config.routing_group_sample_budget = 4;
        planner_config.audit_pair_budget = 2;
        Self {
            families: InteractionFamily::ALL.to_vec(),
            compositions_per_family: 7,
            frame_count: 128,
            seeds: vec![11, 29, 47],
            pair_budgets: vec![0, 1, 2, 4, 8, 16, 28],
            planner_config,
            factorial_config: FactorialReferenceConfig {
                sample_rate_hz: 48_000,
                spectral_window_frames: 64,
            },
            metric_config: InteractionProbeBenchmarkConfig::default(),
            recall_target: 0.95,
            bootstrap_replicates: 2_000,
            bootstrap_seed: 0x4d45_4c26_2026_1113,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignFixtureResult {
    pub fixture_id: String,
    pub composition_index: usize,
    pub family: InteractionFamily,
    pub split: CampaignSplit,
    pub event_count: usize,
    pub frame_count: usize,
    pub source_composition_sha256: String,
    pub effect_profile_sha256: String,
    pub oracle_receipt_sha256: String,
    pub support_topology_sha256: String,
    pub contextual_positive_pair_count: usize,
    pub global_order_two_harsanyi_count: usize,
    pub higher_order_harsanyi_count: usize,
    pub rows: Vec<ProbePolicyComparisonRow>,
    pub curves: Vec<ProbePolicyCurveSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedClusterSummary {
    pub scope: CampaignScope,
    pub candidate_policy: ProbeSelectionPolicy,
    pub baseline_policy: ProbeSelectionPolicy,
    /// Number of independent composition-level means, not seed/budget rows.
    pub independent_composition_count: usize,
    /// Same composition, seed and requested budget, with identical actual render count.
    pub matched_row_pair_count: usize,
    /// Candidate/baseline row pairs excluded because actual render costs differed
    /// or one recall denominator was undefined.
    pub unmatched_row_pair_count: usize,
    /// Each composition has equal weight regardless of how many seed/budget rows it has.
    pub mean_composition_recall_delta: Option<f64>,
    pub bootstrap_95pct_lower: Option<f64>,
    pub bootstrap_95pct_upper: Option<f64>,
    pub bootstrap_replicates: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NullSceneFalseDiscoverySummary {
    pub family: InteractionFamily,
    pub policy: ProbeSelectionPolicy,
    pub composition_count: usize,
    pub evaluated_rows: usize,
    pub rows_with_any_selected_pair: usize,
    pub total_false_positive_pairs: usize,
    pub total_selected_pairs: usize,
    /// Pooled pair-level FDP on all-null scenes. This is 1 whenever any pair
    /// is selected and is therefore not a useful policy-discrimination metric
    /// by itself; composition-macro selection and FP burden are reported too.
    pub false_discovery_rate: Option<f64>,
    /// Equal-weight mean of each composition's fraction of rows selecting
    /// at least one pair. Seeds/budgets remain repeated rows inside a cluster.
    pub composition_macro_any_selection_rate: Option<f64>,
    /// Equal-weight mean false-positive pairs per row, averaged within each
    /// composition before averaging across independent compositions.
    pub composition_macro_false_positive_pairs_per_row: Option<f64>,
    pub any_selection_rate_bootstrap_95pct_lower: Option<f64>,
    pub any_selection_rate_bootstrap_95pct_upper: Option<f64>,
    pub false_positive_burden_bootstrap_95pct_lower: Option<f64>,
    pub false_positive_burden_bootstrap_95pct_upper: Option<f64>,
    pub bootstrap_replicates: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProbePolicyCampaignReport {
    pub version: String,
    pub config: ProbePolicyCampaignConfig,
    pub fixtures: Vec<CampaignFixtureResult>,
    pub paired_summaries: Vec<PairedClusterSummary>,
    pub null_scene_false_discovery: Vec<NullSceneFalseDiscoverySummary>,
    pub fixture_count: usize,
    pub development_fixture_count: usize,
    pub held_out_interaction_family_fixture_count: usize,
    pub held_out_routing_topology_fixture_count: usize,
    pub total_composition_clusters: usize,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbePolicyCampaignError {
    InvalidConfig(&'static str),
    DuplicateFamily,
    DuplicateSeed,
    DuplicateBudget,
    TooManyFixtures,
    FactorialReference(FactorialReferenceError),
    PolicyComparison(ProbePolicyComparisonError),
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for ProbePolicyCampaignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "probe-policy campaign validation failed: {self:?}")
    }
}

impl std::error::Error for ProbePolicyCampaignError {}

#[derive(Debug, Clone)]
struct GeneratedFixture {
    events: Vec<SyntheticEventSignal>,
    effect: SyntheticEffect,
    supports: Vec<EventInteractionSupport>,
    source_digest: String,
    effect_digest: String,
    support_digest: String,
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

    fn below(&mut self, upper: usize) -> usize {
        let bound = upper as u64;
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

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, ProbePolicyCampaignError> {
    let bytes =
        serde_json::to_vec(value).map_err(|_| ProbePolicyCampaignError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn family_seed(family: InteractionFamily) -> u64 {
    match family {
        InteractionFamily::AdditiveNull => 1,
        InteractionFamily::SparsePairwise => 2,
        InteractionFamily::Saturation => 3,
        InteractionFamily::PureThirdOrder => 4,
        InteractionFamily::LinearFirTail => 5,
        InteractionFamily::DenseOverlap => 6,
        InteractionFamily::ClusteredOverlap => 7,
        InteractionFamily::SharedRouteDisjoint => 8,
        InteractionFamily::UnderstatedSupport => 9,
        InteractionFamily::IncompleteRouting => 10,
    }
}

fn scope_seed(scope: CampaignScope) -> u64 {
    match scope {
        CampaignScope::Development => 1,
        CampaignScope::HeldOutInteractionFamily => 2,
        CampaignScope::HeldOutRoutingTopology => 3,
        CampaignScope::AllFixtures => 4,
        CampaignScope::Family(family) => 100 + family_seed(family),
    }
}

fn validate_config(config: &ProbePolicyCampaignConfig) -> Result<(), ProbePolicyCampaignError> {
    if config.families.is_empty() {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "at least one family is required",
        ));
    }
    if !config.families.contains(&InteractionFamily::PureThirdOrder)
        || !config
            .families
            .contains(&InteractionFamily::IncompleteRouting)
        || !config
            .families
            .iter()
            .any(|family| family.split() == CampaignSplit::Development)
    {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "campaign must include development, held-out interaction-family and held-out routing-topology strata",
        ));
    }
    if config.compositions_per_family < 7 || config.compositions_per_family > 64 {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "compositions_per_family must be in 7..=64 to cover event-count strata",
        ));
    }
    if config.frame_count < 64 || config.frame_count > 4_096 {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "frame_count must be in 64..=4096",
        ));
    }
    if config.factorial_config.sample_rate_hz == 0
        || config.factorial_config.spectral_window_frames < 16
        || config.factorial_config.spectral_window_frames > config.frame_count
    {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "invalid factorial reference config",
        ));
    }
    if config.seeds.is_empty() || config.seeds.len() > 64 {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "seeds must contain 1..=64 entries",
        ));
    }
    if config.pair_budgets.is_empty() || config.pair_budgets.len() > 32 {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "pair_budgets must contain 1..=32 entries",
        ));
    }
    if !config.recall_target.is_finite() || (config.recall_target - 0.95).abs() > f64::EPSILON {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "recall_target is frozen at 0.95 by the EVAL-024 comparator version",
        ));
    }
    if config
        .pair_budgets
        .iter()
        .any(|budget| *budget > config.planner_config.pair_probe_budget)
    {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "every requested pair budget must fit the frozen planner_config budget",
        ));
    }
    if config.bootstrap_replicates < 100 || config.bootstrap_replicates > MAX_BOOTSTRAP_REPLICATES {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "bootstrap_replicates must be in 100..=20000",
        ));
    }

    let mut families = BTreeSet::new();
    for family in &config.families {
        if !families.insert(*family) {
            return Err(ProbePolicyCampaignError::DuplicateFamily);
        }
    }
    let mut seeds = BTreeSet::new();
    for seed in &config.seeds {
        if !seeds.insert(*seed) {
            return Err(ProbePolicyCampaignError::DuplicateSeed);
        }
    }
    if config
        .pair_budgets
        .windows(2)
        .any(|window| window[0] >= window[1])
    {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "pair_budgets must be strictly increasing",
        ));
    }
    let mut budgets = BTreeSet::new();
    for budget in &config.pair_budgets {
        if *budget > 28 {
            return Err(ProbePolicyCampaignError::InvalidConfig(
                "pair budget cannot exceed 28 for N<=8",
            ));
        }
        if !budgets.insert(*budget) {
            return Err(ProbePolicyCampaignError::DuplicateBudget);
        }
    }
    let fixture_count = config
        .families
        .len()
        .saturating_mul(config.compositions_per_family);
    if fixture_count > MAX_COMPOSITIONS {
        return Err(ProbePolicyCampaignError::TooManyFixtures);
    }
    if config.families.contains(&InteractionFamily::PureThirdOrder) && config.frame_count < 32 {
        return Err(ProbePolicyCampaignError::InvalidConfig(
            "third-order fixtures require >=32 frames",
        ));
    }
    Ok(())
}

fn active_interval(
    family: InteractionFamily,
    event_index: usize,
    event_count: usize,
    frame_count: usize,
) -> (usize, usize) {
    let edge = 8usize;
    let usable = frame_count.saturating_sub(edge * 2);
    let slot = (usable / event_count.max(1)).max(4);
    let clamp_interval = |start: usize, end: usize| {
        let start = start.min(frame_count.saturating_sub(1));
        let end = end.max(start + 1).min(frame_count);
        (start, end)
    };
    match family {
        InteractionFamily::AdditiveNull | InteractionFamily::SharedRouteDisjoint => {
            let start = edge + event_index * slot;
            clamp_interval(start, start + (slot / 2).max(2))
        }
        InteractionFamily::SparsePairwise
        | InteractionFamily::UnderstatedSupport
        | InteractionFamily::IncompleteRouting => {
            let (start, end) = match event_index {
                0 => (edge, edge + slot * 3),
                1 => (edge + slot * 2, edge + slot * 4),
                _ => {
                    let tail_start = edge + slot * 4;
                    let tail_frames = frame_count.saturating_sub(edge + tail_start);
                    let remaining_events = (event_count - 2).max(1);
                    let tail_slot = (tail_frames / remaining_events).max(1);
                    let start = tail_start + (event_index - 2) * tail_slot;
                    (start, start + (tail_slot / 2).max(1))
                }
            };
            clamp_interval(start, end)
        }
        InteractionFamily::Saturation | InteractionFamily::DenseOverlap => {
            clamp_interval(edge, frame_count - edge)
        }
        InteractionFamily::PureThirdOrder if event_index < 3 => {
            clamp_interval(edge, frame_count / 2)
        }
        InteractionFamily::PureThirdOrder => {
            let start = frame_count / 2 + (event_index - 3) * slot;
            clamp_interval(start, start + (slot / 2).max(2))
        }
        InteractionFamily::LinearFirTail => {
            let start = edge + event_index * slot;
            // The final source event reaches the score boundary so the
            // linear FIR's full L-1 sample convolution tail is observable.
            let end = if event_index + 1 == event_count {
                frame_count
            } else {
                start + (slot / 3).max(2)
            };
            clamp_interval(start, end)
        }
        InteractionFamily::ClusteredOverlap => {
            let split = event_count.div_ceil(2);
            if event_index < split {
                clamp_interval(edge, frame_count / 2)
            } else {
                clamp_interval(frame_count / 2, frame_count - edge)
            }
        }
    }
}

fn make_signal(
    family: InteractionFamily,
    event_index: usize,
    frame_count: usize,
    start: usize,
    end: usize,
    amplitude: f64,
    composition_index: usize,
) -> SyntheticEventSignal {
    let mut samples = vec![0.0; frame_count];
    let polarity = if matches!(
        family,
        InteractionFamily::Saturation | InteractionFamily::PureThirdOrder
    ) {
        1.0
    } else if (event_index + composition_index) % 3 == 0 {
        -1.0
    } else {
        1.0
    };
    for (frame, sample) in samples.iter_mut().enumerate().take(end).skip(start) {
        // A deterministic multi-level envelope avoids a constant-only fixture
        // while remaining exactly reproducible across runs.
        let phase = (frame + event_index * 3 + composition_index * 5) % 8;
        let shape = match phase {
            0 | 1 => 0.75,
            2 | 3 | 4 => 1.0,
            _ => 0.5,
        };
        *sample = polarity * amplitude * shape;
    }
    SyntheticEventSignal {
        source_event_index: event_index,
        samples,
    }
}

fn effect_for(family: InteractionFamily) -> SyntheticEffect {
    match family {
        InteractionFamily::AdditiveNull | InteractionFamily::SharedRouteDisjoint => {
            SyntheticEffect::LinearMix
        }
        InteractionFamily::SparsePairwise
        | InteractionFamily::UnderstatedSupport
        | InteractionFamily::IncompleteRouting => SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.35,
            cubic_coefficient: 0.0,
        },
        InteractionFamily::Saturation => SyntheticEffect::HardClip { threshold: 0.46 },
        InteractionFamily::PureThirdOrder => SyntheticEffect::ThreeWayProduct {
            event_indices: [0, 1, 2],
            coefficient: 0.75,
        },
        InteractionFamily::LinearFirTail => SyntheticEffect::LinearFir {
            impulse_response: vec![1.0, 0.55, 0.25, 0.10, 0.04],
        },
        InteractionFamily::DenseOverlap => SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.28,
            cubic_coefficient: 0.0,
        },
        InteractionFamily::ClusteredOverlap => SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.32,
            cubic_coefficient: 0.0,
        },
    }
}

fn generate_fixture(
    family: InteractionFamily,
    composition_index: usize,
    config: &ProbePolicyCampaignConfig,
) -> Result<GeneratedFixture, ProbePolicyCampaignError> {
    let event_count =
        (2 + composition_index % 7).max(if family == InteractionFamily::PureThirdOrder {
            3
        } else {
            2
        });
    let frame_count = config.frame_count;
    let mut events = Vec::with_capacity(event_count);
    let mut supports = Vec::with_capacity(event_count);
    let mut seed = SplitMix64(
        (composition_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ family_seed(family).wrapping_mul(0xD1B5_4A32_D192_ED03)
            ^ 0x4d45_4c26_2026_1113,
    );

    for event_index in 0..event_count {
        let (start, end) = active_interval(family, event_index, event_count, frame_count);
        let jitter = (seed.below(5) as f64) * 0.005;
        let amplitude = if family == InteractionFamily::Saturation {
            0.255 + jitter
        } else if family == InteractionFamily::PureThirdOrder {
            0.42 + jitter
        } else {
            0.18 + ((composition_index + event_index * 3) % 7) as f64 * 0.012 + jitter
        };
        events.push(make_signal(
            family,
            event_index,
            frame_count,
            start,
            end,
            amplitude,
            composition_index,
        ));

        let mut support_start = start as u64;
        let mut support_end = end as u64;
        if family == InteractionFamily::LinearFirTail {
            // The EVAL-020 reference renderer emits input_frames + (L - 1)
            // output frames for an L-tap FIR. Preserve the complete tail
            // instead of clipping the event support to the input schedule.
            support_end += 4;
        }
        if family == InteractionFamily::UnderstatedSupport {
            // Deliberately understate the real overlap of events 0 and 1.
            // Oracle source signals and factorial truth remain untouched.
            if event_index == 0 {
                support_end = (start + (end - start) / 2).min(frame_count) as u64;
            } else if event_index == 1 {
                support_start = (start + (end - start) / 2).min(frame_count) as u64;
            }
        }
        let mut routing_domains = vec![format!("track:{event_index}")];
        let mut routing_complete = true;
        match family {
            InteractionFamily::SharedRouteDisjoint => {
                routing_domains.push("master:shared-linear-bus".to_owned());
            }
            InteractionFamily::IncompleteRouting => {
                routing_complete = false;
                // A partial observed route is not an independence certificate.
                routing_domains.push(format!("observed-track:{event_index}"));
            }
            _ => {}
        }
        let support_value = (
            event_index,
            support_start,
            support_end,
            &routing_domains,
            routing_complete,
        );
        let support_bytes = serde_json::to_vec(&support_value)
            .map_err(|_| ProbePolicyCampaignError::SerializationFailure)?;
        supports.push(EventInteractionSupport {
            source_event_index: event_index,
            acoustic_support: if support_end > support_start {
                vec![AcousticSupportInterval {
                    start_frame: support_start,
                    end_frame: support_end,
                }]
            } else {
                Vec::new()
            },
            routing_domains,
            routing_complete,
            support_evidence_sha256: Some(sha256_hex(&support_bytes)),
        });
    }

    let effect = effect_for(family);
    let source_bytes =
        serde_json::to_vec(&events).map_err(|_| ProbePolicyCampaignError::SerializationFailure)?;
    let effect_bytes =
        serde_json::to_vec(&effect).map_err(|_| ProbePolicyCampaignError::SerializationFailure)?;
    let support_bytes = serde_json::to_vec(&supports)
        .map_err(|_| ProbePolicyCampaignError::SerializationFailure)?;
    Ok(GeneratedFixture {
        events,
        effect,
        supports,
        source_digest: sha256_hex(&source_bytes),
        effect_digest: sha256_hex(&effect_bytes),
        support_digest: sha256_hex(&support_bytes),
    })
}

fn quantile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let position = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    Some(sorted[lower] * (1.0 - weight) + sorted[upper] * weight)
}

fn bootstrap_interval(
    cluster_means: &[f64],
    replicates: usize,
    seed: u64,
) -> (Option<f64>, Option<f64>) {
    if cluster_means.len() < 2 || replicates == 0 {
        return (None, None);
    }
    let mut rng = SplitMix64(seed);
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0.0;
        for _ in 0..cluster_means.len() {
            sum += cluster_means[rng.below(cluster_means.len())];
        }
        estimates.push(sum / cluster_means.len() as f64);
    }
    estimates.sort_by(f64::total_cmp);
    (quantile(&estimates, 0.025), quantile(&estimates, 0.975))
}

fn scope_matches(fixture: &CampaignFixtureResult, scope: CampaignScope) -> bool {
    match scope {
        CampaignScope::Development => fixture.split == CampaignSplit::Development,
        CampaignScope::HeldOutInteractionFamily => {
            fixture.split == CampaignSplit::HeldOutInteractionFamily
        }
        CampaignScope::HeldOutRoutingTopology => {
            fixture.split == CampaignSplit::HeldOutRoutingTopology
        }
        CampaignScope::AllFixtures => true,
        CampaignScope::Family(family) => fixture.family == family,
    }
}

fn paired_summary(
    fixtures: &[CampaignFixtureResult],
    scope: CampaignScope,
    baseline_policy: ProbeSelectionPolicy,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> PairedClusterSummary {
    let mut cluster_means = Vec::new();
    let mut matched_row_pair_count = 0usize;
    let mut unmatched_row_pair_count = 0usize;
    for fixture in fixtures
        .iter()
        .filter(|fixture| scope_matches(fixture, scope))
    {
        let mut deltas = Vec::new();
        for candidate in fixture
            .rows
            .iter()
            .filter(|row| row.policy == ProbeSelectionPolicy::Eval019)
        {
            let baseline = fixture.rows.iter().find(|row| {
                row.policy == baseline_policy
                    && row.seed == candidate.seed
                    && row.requested_pair_budget == candidate.requested_pair_budget
            });
            let Some(baseline) = baseline else {
                unmatched_row_pair_count += 1;
                continue;
            };
            if candidate.total_render_count != baseline.total_render_count {
                unmatched_row_pair_count += 1;
                continue;
            }
            match (candidate.recall, baseline.recall) {
                (Some(candidate_recall), Some(baseline_recall)) => {
                    deltas.push(candidate_recall - baseline_recall);
                    matched_row_pair_count += 1;
                }
                _ => unmatched_row_pair_count += 1,
            }
        }
        if !deltas.is_empty() {
            cluster_means.push(deltas.iter().sum::<f64>() / deltas.len() as f64);
        }
    }
    let mean = if cluster_means.is_empty() {
        None
    } else {
        Some(cluster_means.iter().sum::<f64>() / cluster_means.len() as f64)
    };
    let (lower, upper) = bootstrap_interval(
        &cluster_means,
        bootstrap_replicates,
        bootstrap_seed ^ scope_seed(scope).rotate_left(11) ^ (baseline_policy as u64),
    );
    PairedClusterSummary {
        scope,
        candidate_policy: ProbeSelectionPolicy::Eval019,
        baseline_policy,
        independent_composition_count: cluster_means.len(),
        matched_row_pair_count,
        unmatched_row_pair_count,
        mean_composition_recall_delta: mean,
        bootstrap_95pct_lower: lower,
        bootstrap_95pct_upper: upper,
        bootstrap_replicates: if cluster_means.len() >= 2 {
            bootstrap_replicates
        } else {
            0
        },
    }
}

fn null_scene_summaries(
    fixtures: &[CampaignFixtureResult],
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Vec<NullSceneFalseDiscoverySummary> {
    let mut result = Vec::new();
    for family in InteractionFamily::ALL {
        let family_fixtures: Vec<&CampaignFixtureResult> = fixtures
            .iter()
            .filter(|fixture| {
                fixture.family == family && fixture.contextual_positive_pair_count == 0
            })
            .collect();
        if family_fixtures.is_empty() {
            continue;
        }
        for policy in [
            ProbeSelectionPolicy::UniformRandom,
            ProbeSelectionPolicy::TemporalOnly,
            ProbeSelectionPolicy::RoutingOnly,
            ProbeSelectionPolicy::Eval019,
        ] {
            let mut evaluated_rows = 0usize;
            let mut rows_with_any_selected_pair = 0usize;
            let mut total_false_positive_pairs = 0usize;
            let mut total_selected_pairs = 0usize;
            let mut composition_selection_rates = Vec::new();
            let mut composition_false_positive_burdens = Vec::new();
            for fixture in &family_fixtures {
                let rows: Vec<_> = fixture.rows.iter().filter(|row| row.policy == policy).collect();
                evaluated_rows += rows.len();
                let selected_rows = rows.iter().filter(|row| row.selected_pair_count > 0).count();
                rows_with_any_selected_pair += selected_rows;
                total_false_positive_pairs +=
                    rows.iter().map(|row| row.false_positive_count).sum::<usize>();
                total_selected_pairs +=
                    rows.iter().map(|row| row.selected_pair_count).sum::<usize>();
                if !rows.is_empty() {
                    let denominator = rows.len() as f64;
                    composition_selection_rates.push(selected_rows as f64 / denominator);
                    composition_false_positive_burdens.push(
                        rows.iter()
                            .map(|row| row.false_positive_count as f64)
                            .sum::<f64>()
                            / denominator,
                    );
                }
            }
            let macro_mean = |values: &[f64]| {
                (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
            };
            let seed = bootstrap_seed
                ^ family_seed(family).rotate_left(17)
                ^ (policy as u64).rotate_left(29);
            let (selection_lower, selection_upper) = bootstrap_interval(
                &composition_selection_rates,
                bootstrap_replicates,
                seed ^ 0x5345_4c45_4354,
            );
            let (burden_lower, burden_upper) = bootstrap_interval(
                &composition_false_positive_burdens,
                bootstrap_replicates,
                seed ^ 0x4641_4c53_4550,
            );
            result.push(NullSceneFalseDiscoverySummary {
                family,
                policy,
                composition_count: composition_selection_rates.len(),
                evaluated_rows,
                rows_with_any_selected_pair,
                total_false_positive_pairs,
                total_selected_pairs,
                false_discovery_rate: if total_selected_pairs == 0 {
                    None
                } else {
                    Some(total_false_positive_pairs as f64 / total_selected_pairs as f64)
                },
                composition_macro_any_selection_rate: macro_mean(&composition_selection_rates),
                composition_macro_false_positive_pairs_per_row: macro_mean(
                    &composition_false_positive_burdens,
                ),
                any_selection_rate_bootstrap_95pct_lower: selection_lower,
                any_selection_rate_bootstrap_95pct_upper: selection_upper,
                false_positive_burden_bootstrap_95pct_lower: burden_lower,
                false_positive_burden_bootstrap_95pct_upper: burden_upper,
                bootstrap_replicates: if composition_selection_rates.len() >= 2 {
                    bootstrap_replicates
                } else {
                    0
                },
            });
        }
    }
    result
}

/// Build the full paired campaign from deterministic synthetic source
/// compositions. The held-out family and routing topology are fixed by the
/// versioned family split: PureThirdOrder and IncompleteRouting.
pub fn build_probe_policy_campaign(
    config: &ProbePolicyCampaignConfig,
) -> Result<ProbePolicyCampaignReport, ProbePolicyCampaignError> {
    validate_config(config)?;
    let mut fixtures = Vec::with_capacity(config.families.len() * config.compositions_per_family);
    for family in &config.families {
        for composition_index in 0..config.compositions_per_family {
            let generated = generate_fixture(*family, composition_index, config)?;
            let oracle: FactorialGroundTruthReport = build_factorial_ground_truth(
                &generated.events,
                &generated.effect,
                &config.factorial_config,
            )
            .map_err(ProbePolicyCampaignError::FactorialReference)?;
            let composition_identity = (
                family,
                composition_index,
                &generated.source_digest,
                &generated.effect_digest,
            );
            let fixture_id = canonical_digest(&composition_identity)?;
            let rendered_frame_count = oracle.output_frame_count as u64;
            let comparison = compare_probe_policies(
                &oracle,
                &generated.events,
                &generated.effect,
                &config.factorial_config,
                &generated.source_digest,
                &generated.effect_digest,
                rendered_frame_count,
                &generated.supports,
                &config.planner_config,
                &config.pair_budgets,
                &config.seeds,
                &config.metric_config,
            )
            .map_err(ProbePolicyCampaignError::PolicyComparison)?;
            let global_order_two_harsanyi_count = oracle
                .dividends
                .iter()
                .filter(|dividend| {
                    dividend.order == 2
                        && dividend.pcm_l2_norm
                            > config
                                .metric_config
                                .minimum_global_order_two_harsanyi_pcm_l2_norm
                })
                .count();
            let higher_order_harsanyi_count = oracle
                .dividends
                .iter()
                .filter(|dividend| {
                    dividend.order >= 3
                        && dividend.pcm_l2_norm
                            > config.metric_config.minimum_higher_order_pcm_l2_norm
                })
                .count();
            let fixture_metadata = (
                &fixture_id,
                family,
                composition_index,
                generated.events.len(),
                rendered_frame_count,
                &generated.source_digest,
                &generated.effect_digest,
                &oracle.receipt_sha256,
                &generated.support_digest,
                oracle.event_count,
            );
            let fixture_id = canonical_digest(&fixture_metadata)?;
            fixtures.push(CampaignFixtureResult {
                fixture_id,
                composition_index,
                family: *family,
                split: family.split(),
                event_count: generated.events.len(),
                frame_count: rendered_frame_count as usize,
                source_composition_sha256: generated.source_digest,
                effect_profile_sha256: generated.effect_digest,
                oracle_receipt_sha256: oracle.receipt_sha256,
                support_topology_sha256: generated.support_digest,
                contextual_positive_pair_count: comparison.contextual_positive_pair_count,
                global_order_two_harsanyi_count,
                higher_order_harsanyi_count,
                rows: comparison.rows,
                curves: comparison.curves,
            });
        }
    }

    let mut paired_summaries = Vec::new();
    for baseline in [
        ProbeSelectionPolicy::UniformRandom,
        ProbeSelectionPolicy::TemporalOnly,
    ] {
        for scope in [
            CampaignScope::Development,
            CampaignScope::HeldOutInteractionFamily,
            CampaignScope::HeldOutRoutingTopology,
        ] {
            paired_summaries.push(paired_summary(
                &fixtures,
                scope,
                baseline,
                config.bootstrap_replicates,
                config.bootstrap_seed,
            ));
        }
        for family in &config.families {
            paired_summaries.push(paired_summary(
                &fixtures,
                CampaignScope::Family(*family),
                baseline,
                config.bootstrap_replicates,
                config.bootstrap_seed,
            ));
        }
    }
    let null_scene_false_discovery = null_scene_summaries(
        &fixtures,
        config.bootstrap_replicates,
        config.bootstrap_seed,
    );
    let development_fixture_count = fixtures
        .iter()
        .filter(|fixture| fixture.split == CampaignSplit::Development)
        .count();
    let held_out_interaction_family_fixture_count = fixtures
        .iter()
        .filter(|fixture| fixture.split == CampaignSplit::HeldOutInteractionFamily)
        .count();
    let held_out_routing_topology_fixture_count = fixtures
        .iter()
        .filter(|fixture| fixture.split == CampaignSplit::HeldOutRoutingTopology)
        .count();
    let mut report = ProbePolicyCampaignReport {
        version: PROBE_POLICY_CAMPAIGN_VERSION.to_owned(),
        config: config.clone(),
        fixture_count: fixtures.len(),
        development_fixture_count,
        held_out_interaction_family_fixture_count,
        held_out_routing_topology_fixture_count,
        total_composition_clusters: fixtures.len(),
        fixtures,
        paired_summaries,
        null_scene_false_discovery,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = canonical_digest(&report)?;
    Ok(report)
}

/// Replay the entire campaign from its frozen configuration and compare every
/// fixture identity, oracle receipt, policy row, cluster statistic and digest.
pub fn verify_probe_policy_campaign(
    report: &ProbePolicyCampaignReport,
) -> Result<bool, ProbePolicyCampaignError> {
    if report.receipt_sha256.len() != 64 {
        return Err(ProbePolicyCampaignError::InvalidReceipt);
    }
    let mut unsigned = report.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if canonical_digest(&unsigned)? != receipt {
        return Err(ProbePolicyCampaignError::InvalidReceipt);
    }
    let expected = build_probe_policy_campaign(&report.config)?;
    if expected != *report {
        return Err(ProbePolicyCampaignError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> ProbePolicyCampaignConfig {
        ProbePolicyCampaignConfig {
            families: vec![
                InteractionFamily::AdditiveNull,
                InteractionFamily::SparsePairwise,
                InteractionFamily::PureThirdOrder,
                InteractionFamily::IncompleteRouting,
            ],
            compositions_per_family: 7,
            frame_count: 128,
            seeds: vec![11, 29],
            pair_budgets: vec![0, 1, 4, 28],
            planner_config: InteractionProbeConfig {
                pair_probe_budget: 28,
                routing_group_sample_budget: 3,
                audit_pair_budget: 1,
                max_sampling_attempts: 10_000,
                seed: 17,
                max_temporal_candidate_pairs: 256,
            },
            factorial_config: FactorialReferenceConfig {
                sample_rate_hz: 48_000,
                spectral_window_frames: 64,
            },
            metric_config: InteractionProbeBenchmarkConfig::default(),
            recall_target: 0.95,
            bootstrap_replicates: 200,
            bootstrap_seed: 91,
        }
    }

    #[test]
    fn campaign_covers_all_event_counts_families_and_frozen_holdouts() {
        let report = build_probe_policy_campaign(&small_config()).unwrap();
        assert_eq!(report.fixture_count, 28);
        assert_eq!(report.development_fixture_count, 14);
        assert_eq!(report.held_out_interaction_family_fixture_count, 7);
        assert_eq!(report.held_out_routing_topology_fixture_count, 7);
        for family in small_config().families {
            let family_fixtures: Vec<&CampaignFixtureResult> = report
                .fixtures
                .iter()
                .filter(|fixture| fixture.family == family)
                .collect();
            let event_counts: BTreeSet<usize> = family_fixtures
                .iter()
                .map(|fixture| fixture.event_count)
                .collect();
            if family == InteractionFamily::PureThirdOrder {
                assert_eq!(event_counts, (3..=8).collect());
            } else {
                assert_eq!(event_counts, (2..=8).collect());
            }
        }
        assert!(report.fixtures.iter().any(|fixture| {
            fixture.family == InteractionFamily::PureThirdOrder
                && fixture.split == CampaignSplit::HeldOutInteractionFamily
        }));
        assert!(report.fixtures.iter().any(|fixture| {
            fixture.family == InteractionFamily::IncompleteRouting
                && fixture.split == CampaignSplit::HeldOutRoutingTopology
        }));
    }

    #[test]
    fn all_generator_families_keep_source_and_support_ranges_bounded() {
        let config = small_config();
        for family in InteractionFamily::ALL {
            for composition_index in [0usize, 6] {
                let fixture = generate_fixture(family, composition_index, &config).unwrap();
                assert!((2..=8).contains(&fixture.events.len()));
                for event in &fixture.events {
                    assert_eq!(event.samples.len(), config.frame_count);
                }
                for support in &fixture.supports {
                    assert!(support.source_event_index < fixture.events.len());
                    for interval in &support.acoustic_support {
                        assert!(interval.start_frame < interval.end_frame);
                        assert!(interval.end_frame <= config.frame_count as u64 + 4);
                    }
                }
            }
        }
    }

    #[test]
    fn campaign_pairs_policies_on_same_compositions_and_seeds() {
        let report = build_probe_policy_campaign(&small_config()).unwrap();
        let primary = report
            .paired_summaries
            .iter()
            .find(|summary| {
                summary.scope == CampaignScope::Development
                    && summary.baseline_policy == ProbeSelectionPolicy::UniformRandom
            })
            .unwrap();
        assert!(primary.independent_composition_count >= 2);
        assert!(primary.matched_row_pair_count > 0);
        assert!(primary.bootstrap_95pct_lower.is_some());
        assert!(primary.bootstrap_95pct_upper.is_some());
    }

    #[test]
    fn support_corruption_changes_probe_view_not_factorial_oracle_truth() {
        let config = small_config();
        let sparse = generate_fixture(InteractionFamily::SparsePairwise, 0, &config).unwrap();
        let oracle_before =
            build_factorial_ground_truth(&sparse.events, &sparse.effect, &config.factorial_config)
                .unwrap();
        let mut corrupted_supports = sparse.supports.clone();
        corrupted_supports[0].acoustic_support[0].end_frame =
            corrupted_supports[0].acoustic_support[0].start_frame + 1;
        corrupted_supports[1].acoustic_support[0].start_frame = corrupted_supports[1]
            .acoustic_support[0]
            .end_frame
            .saturating_sub(1);
        let support_bytes = serde_json::to_vec(&corrupted_supports).unwrap();
        let corrupted_support_digest = sha256_hex(&support_bytes);
        let oracle_after =
            build_factorial_ground_truth(&sparse.events, &sparse.effect, &config.factorial_config)
                .unwrap();
        assert_eq!(oracle_before.input_sha256, oracle_after.input_sha256);
        assert_ne!(sparse.support_digest, corrupted_support_digest);
    }

    #[test]
    fn incomplete_routing_is_a_heldout_topology_and_never_claims_independence() {
        let config = small_config();
        let fixture = generate_fixture(InteractionFamily::IncompleteRouting, 0, &config).unwrap();
        assert!(
            fixture
                .supports
                .iter()
                .any(|support| !support.routing_complete)
        );
        assert_eq!(
            InteractionFamily::IncompleteRouting.split(),
            CampaignSplit::HeldOutRoutingTopology
        );
        let unknown_scope = fixture
            .supports
            .iter()
            .filter(|support| !support.routing_complete)
            .count();
        assert_eq!(unknown_scope, fixture.events.len());
    }

    #[test]
    fn campaign_report_replays_and_tampering_is_rejected() {
        let report = build_probe_policy_campaign(&small_config()).unwrap();
        assert!(verify_probe_policy_campaign(&report).unwrap());
        let mut tampered = report.clone();
        tampered.fixtures[0].rows[0].selected_pair_count += 1;
        assert_eq!(
            verify_probe_policy_campaign(&tampered).unwrap_err(),
            ProbePolicyCampaignError::InvalidReceipt,
        );
    }

    #[test]
    fn config_rejects_duplicate_seeds_budgets_and_families() {
        let mut config = small_config();
        config.seeds.push(config.seeds[0]);
        assert_eq!(
            build_probe_policy_campaign(&config).unwrap_err(),
            ProbePolicyCampaignError::DuplicateSeed,
        );
        config = small_config();
        config.pair_budgets.push(config.pair_budgets[0]);
        assert_eq!(
            build_probe_policy_campaign(&config).unwrap_err(),
            ProbePolicyCampaignError::DuplicateBudget,
        );
        config = small_config();
        config.families.push(config.families[0]);
        assert_eq!(
            build_probe_policy_campaign(&config).unwrap_err(),
            ProbePolicyCampaignError::DuplicateFamily,
        );
    }
}

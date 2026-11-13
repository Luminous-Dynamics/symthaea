// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paired multi-composition campaign and composition-clustered analysis.
//!
//! This module composes EVAL-019, EVAL-020 and EVAL-024. The independent
//! resampling unit is the source-composition lineage. Seeds, budgets, pairs,
//! windows and fixture variants are repeated observations nested in that
//! cluster; they are never treated as independent compositions.

use crate::factorial_ground_truth::{
    FactorialReferenceConfig, FactorialReferenceError, SyntheticEffect, SyntheticEventSignal,
    build_factorial_ground_truth, verify_factorial_ground_truth,
};
use crate::interaction_probe_benchmark::InteractionProbeBenchmarkConfig;
use crate::interaction_probe_plan::{
    AcousticSupportInterval, EventInteractionSupport, InteractionProbeConfig,
};
use crate::probe_policy_comparison::{
    ProbePolicyComparisonError, ProbePolicyComparisonRow, ProbeSelectionPolicy,
    compare_probe_policies,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const INTERACTION_PROBE_CAMPAIGN_VERSION: &str =
    "mel-mr-eval-026-multifamily-campaign-v0.1";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-026-campaign-v1\0";
const MAX_FIXTURES: usize = 4_096;
const MAX_EVENTS: usize = 8;
const MAX_FRAMES: usize = 8_192;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InteractionFixtureFamily {
    AdditiveNull,
    SparsePairwise,
    HardClipping,
    PureHigherOrder,
    LinearFirTail,
    DenseOverlap,
    SharedRoutingDisjointTime,
    UnderstatedSupport,
    IncompleteRouting,
    MixedSalience,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignSplit {
    Development,
    HeldOut,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignFixtureSpec {
    pub fixture_id: String,
    pub composition_lineage_sha256: String,
    pub family: InteractionFixtureFamily,
    pub routing_topology_id: String,
    pub seed: u64,
    pub event_count: usize,
    pub frame_count: usize,
    pub amplitude_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionCampaignFixture {
    pub fixture_id: String,
    pub composition_lineage_sha256: String,
    pub family: InteractionFixtureFamily,
    pub routing_topology_id: String,
    pub split: CampaignSplit,
    pub source_score_sha256: String,
    pub renderer_profile_sha256: String,
    pub total_frames: u64,
    pub events: Vec<SyntheticEventSignal>,
    pub effect: SyntheticEffect,
    pub factorial_config: FactorialReferenceConfig,
    pub event_supports: Vec<EventInteractionSupport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionProbeCampaignConfig {
    pub planner_config: InteractionProbeConfig,
    pub budgets: Vec<usize>,
    pub seeds: Vec<u64>,
    pub metric_config: InteractionProbeBenchmarkConfig,
    pub held_out_families: BTreeSet<InteractionFixtureFamily>,
    pub held_out_routing_topologies: BTreeSet<String>,
    pub minimum_compositions_per_stratum: usize,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
    /// 950 means a two-sided 95% percentile interval.
    pub confidence_level_permille: u16,
}

impl Default for InteractionProbeCampaignConfig {
    fn default() -> Self {
        Self {
            planner_config: InteractionProbeConfig {
                pair_probe_budget: 4,
                routing_group_sample_budget: 4,
                audit_pair_budget: 2,
                max_sampling_attempts: 10_000,
                seed: 0x4d45_4c19_2026_1113,
                max_temporal_candidate_pairs: 4_096,
            },
            budgets: vec![0, 1, 2, 4, 8],
            seeds: vec![11, 29, 47, 71, 103],
            metric_config: InteractionProbeBenchmarkConfig::default(),
            held_out_families: [InteractionFixtureFamily::PureHigherOrder].into_iter().collect(),
            held_out_routing_topologies: ["route:held-out-shared-master".to_owned()].into_iter().collect(),
            minimum_compositions_per_stratum: 3,
            bootstrap_replicates: 2_000,
            bootstrap_seed: 0x4d45_4c19_2026_1126,
            confidence_level_permille: 950,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignFixtureResult {
    pub fixture_id: String,
    pub composition_lineage_sha256: String,
    pub family: InteractionFixtureFamily,
    pub routing_topology_id: String,
    pub split: CampaignSplit,
    pub routing_complete: bool,
    pub event_count: usize,
    pub possible_pair_count: usize,
    pub contextual_positive_pair_count: usize,
    pub global_order_two_harsanyi_count: usize,
    pub higher_order_harsanyi_count: usize,
    pub factorial_report_sha256: String,
    pub comparison_report_sha256: String,
    pub rows: Vec<ProbePolicyComparisonRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CampaignPolicyAggregate {
    pub split: CampaignSplit,
    pub family: InteractionFixtureFamily,
    pub policy: ProbeSelectionPolicy,
    pub requested_pair_budget: usize,
    pub independent_composition_count: usize,
    pub fixture_count: usize,
    pub observation_count: usize,
    pub precision_defined_composition_count: usize,
    pub recall_defined_composition_count: usize,
    pub mean_precision: Option<f64>,
    pub mean_recall: Option<f64>,
    pub mean_selected_pair_count: f64,
    pub mean_render_cost_fraction: f64,
    pub total_true_positives: u128,
    pub total_false_positives: u128,
    pub total_false_negatives: u128,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedPolicyContrast {
    pub split: CampaignSplit,
    pub family: InteractionFixtureFamily,
    pub candidate_policy: ProbeSelectionPolicy,
    pub baseline_policy: ProbeSelectionPolicy,
    pub matched_composition_count: usize,
    pub matched_block_count: usize,
    pub unmatched_or_undefined_block_count: usize,
    pub mean_recall_delta_candidate_minus_baseline: Option<f64>,
    pub confidence_interval_low: Option<f64>,
    pub confidence_interval_high: Option<f64>,
    pub bootstrap_replicates: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutingStratumSummary {
    pub split: CampaignSplit,
    pub family: InteractionFixtureFamily,
    pub routing_complete: bool,
    pub policy: ProbeSelectionPolicy,
    pub requested_pair_budget: usize,
    pub independent_composition_count: usize,
    pub row_count: usize,
    pub true_positive_count: u128,
    pub false_positive_count: u128,
    pub false_negative_count: u128,
    pub missed_interaction_rate: Option<f64>,
    pub false_discovery_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorstFamilyPerformance {
    pub split: CampaignSplit,
    pub policy: ProbeSelectionPolicy,
    pub requested_pair_budget: usize,
    pub family: InteractionFixtureFamily,
    pub mean_recall: f64,
    pub independent_composition_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionProbeCampaignReport {
    pub version: String,
    pub config: InteractionProbeCampaignConfig,
    pub fixture_input_sha256: String,
    pub fixture_count: usize,
    pub independent_composition_count: usize,
    pub development_fixture_count: usize,
    pub held_out_fixture_count: usize,
    pub held_out_families: Vec<InteractionFixtureFamily>,
    pub held_out_routing_topologies: Vec<String>,
    pub fixture_results: Vec<CampaignFixtureResult>,
    pub aggregates: Vec<CampaignPolicyAggregate>,
    pub paired_contrasts: Vec<PairedPolicyContrast>,
    pub routing_summaries: Vec<RoutingStratumSummary>,
    pub worst_family_performance: Vec<WorstFamilyPerformance>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InteractionProbeCampaignError {
    InvalidConfig(&'static str),
    InvalidFixture(&'static str),
    InvalidDigest(&'static str),
    DuplicateFixtureId,
    CompositionLineageCrossesSplits,
    FixtureSplitDoesNotMatchPreregisteredHoldout,
    MissingHeldOutFamily,
    MissingHeldOutRoutingTopology,
    InsufficientIndependentCompositions,
    FactorialReference(FactorialReferenceError),
    ProbeComparison(ProbePolicyComparisonError),
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for InteractionProbeCampaignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "interaction probe campaign validation failed: {self:?}")
    }
}
impl std::error::Error for InteractionProbeCampaignError {}

#[derive(Debug, Clone, Default)]
struct ClusterMetric {
    precision: Vec<f64>,
    recall: Vec<f64>,
    selected_pairs: Vec<f64>,
    render_cost_fraction: Vec<f64>,
    true_positives: u128,
    false_positives: u128,
    false_negatives: u128,
    fixtures: BTreeSet<String>,
    observations: usize,
}

#[derive(Debug, Clone)]
struct DeterministicRng(u64);

impl DeterministicRng {
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

fn digest_is_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, InteractionProbeCampaignError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| InteractionProbeCampaignError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_config(config: &InteractionProbeCampaignConfig) -> Result<(), InteractionProbeCampaignError> {
    if config.budgets.is_empty()
        || config.budgets.len() > 64
        || config.budgets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "budgets must be non-empty, strictly increasing and bounded",
        ));
    }
    if config.seeds.is_empty()
        || config.seeds.len() > 256
        || config.seeds.iter().collect::<BTreeSet<_>>().len() != config.seeds.len()
    {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "seeds must be non-empty, unique and bounded",
        ));
    }
    if !(2..=256).contains(&config.minimum_compositions_per_stratum) {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "each family/split stratum requires at least two compositions",
        ));
    }
    if !(200..=20_000).contains(&config.bootstrap_replicates) {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "bootstrap replicate count must be in 200..=20_000",
        ));
    }
    if !(800..=999).contains(&config.confidence_level_permille)
        || config.held_out_families.is_empty()
        || config.held_out_routing_topologies.is_empty()
    {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "declare held-out families/topologies and an 80%-99.9% confidence level",
        ));
    }
    if config.held_out_routing_topologies.iter().any(|value| {
        value.is_empty() || value.len() > 128 || value.chars().any(char::is_control)
    }) {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "held-out routing topology identifiers must be bounded",
        ));
    }
    Ok(())
}

fn fixture_split(
    family: InteractionFixtureFamily,
    topology: &str,
    config: &InteractionProbeCampaignConfig,
) -> CampaignSplit {
    if config.held_out_families.contains(&family)
        || config.held_out_routing_topologies.contains(topology)
    {
        CampaignSplit::HeldOut
    } else {
        CampaignSplit::Development
    }
}

fn validate_fixtures(
    fixtures: &[InteractionCampaignFixture],
    config: &InteractionProbeCampaignConfig,
) -> Result<(), InteractionProbeCampaignError> {
    if fixtures.is_empty() || fixtures.len() > MAX_FIXTURES {
        return Err(InteractionProbeCampaignError::InvalidConfig(
            "fixture count must be in 1..=4096",
        ));
    }
    let mut ids = BTreeSet::new();
    let mut lineage_split = BTreeMap::<String, CampaignSplit>::new();
    let mut strata = BTreeMap::<(CampaignSplit, InteractionFixtureFamily), BTreeSet<String>>::new();
    let mut held_families = BTreeSet::new();
    let mut held_topologies = BTreeSet::new();

    for fixture in fixtures {
        if fixture.fixture_id.is_empty()
            || fixture.fixture_id.len() > 128
            || !fixture.fixture_id.bytes().all(|b| {
                b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_'
            })
        {
            return Err(InteractionProbeCampaignError::InvalidFixture(
                "fixture ID must be a bounded lowercase ASCII identifier",
            ));
        }
        if !ids.insert(fixture.fixture_id.clone()) {
            return Err(InteractionProbeCampaignError::DuplicateFixtureId);
        }
        for (name, digest) in [
            ("composition lineage", fixture.composition_lineage_sha256.as_str()),
            ("source score", fixture.source_score_sha256.as_str()),
            ("renderer profile", fixture.renderer_profile_sha256.as_str()),
        ] {
            if !digest_is_valid(digest) {
                return Err(InteractionProbeCampaignError::InvalidDigest(name));
            }
        }
        if fixture.routing_topology_id.is_empty()
            || fixture.routing_topology_id.len() > 128
            || fixture.routing_topology_id.chars().any(char::is_control)
            || fixture.events.len() < 2
            || fixture.events.len() > MAX_EVENTS
            || fixture.events.iter().any(|event| {
                event.samples.len() < 16 || event.samples.len() > MAX_FRAMES
            })
            || fixture.event_supports.len() != fixture.events.len()
        {
            return Err(InteractionProbeCampaignError::InvalidFixture(
                "invalid topology, event vector or support coverage",
            ));
        }
        for (index, support) in fixture.event_supports.iter().enumerate() {
            if support.source_event_index != index
                || support.support_evidence_sha256.as_deref().is_none_or(|d| !digest_is_valid(d))
            {
                return Err(InteractionProbeCampaignError::InvalidFixture(
                    "support evidence must be complete, canonical and content-addressed",
                ));
            }
        }
        let expected_split = fixture_split(fixture.family, &fixture.routing_topology_id, config);
        if fixture.split != expected_split {
            return Err(
                InteractionProbeCampaignError::FixtureSplitDoesNotMatchPreregisteredHoldout,
            );
        }
        if let Some(previous) = lineage_split.insert(
            fixture.composition_lineage_sha256.clone(),
            fixture.split,
        ) {
            if previous != fixture.split {
                return Err(InteractionProbeCampaignError::CompositionLineageCrossesSplits);
            }
        }
        strata
            .entry((fixture.split, fixture.family))
            .or_default()
            .insert(fixture.composition_lineage_sha256.clone());
        if fixture.split == CampaignSplit::HeldOut {
            if config.held_out_families.contains(&fixture.family) {
                held_families.insert(fixture.family);
            }
            if config.held_out_routing_topologies.contains(&fixture.routing_topology_id) {
                held_topologies.insert(fixture.routing_topology_id.clone());
            }
        }
    }

    if held_families != config.held_out_families {
        return Err(InteractionProbeCampaignError::MissingHeldOutFamily);
    }
    if held_topologies != config.held_out_routing_topologies {
        return Err(InteractionProbeCampaignError::MissingHeldOutRoutingTopology);
    }
    if strata.values().any(|lineages| {
        lineages.len() < config.minimum_compositions_per_stratum
    }) {
        return Err(InteractionProbeCampaignError::InsufficientIndependentCompositions);
    }
    Ok(())
}

fn support_digest(
    index: usize,
    intervals: &[(u64, u64)],
    domains: &[String],
    complete: bool,
) -> String {
    let bytes = serde_json::to_vec(&(index, intervals, domains, complete)).unwrap_or_default();
    sha256_hex(&bytes)
}

/// Deterministic synthetic fixture generator. Source signals/effect define
/// oracle truth; event_supports are a separate adapter input. The
/// UnderstatedSupport family deliberately corrupts only that adapter input.
pub fn generate_campaign_fixture(
    spec: &CampaignFixtureSpec,
    config: &InteractionProbeCampaignConfig,
) -> Result<InteractionCampaignFixture, InteractionProbeCampaignError> {
    validate_config(config)?;
    if !digest_is_valid(&spec.composition_lineage_sha256)
        || spec.event_count < 2
        || spec.event_count > MAX_EVENTS
        || spec.frame_count < 128
        || spec.frame_count > MAX_FRAMES
        || !spec.amplitude_scale.is_finite()
        || spec.amplitude_scale <= 0.0
        || spec.amplitude_scale > 0.8
        || spec.routing_topology_id.is_empty()
        || spec.routing_topology_id.len() > 128
    {
        return Err(InteractionProbeCampaignError::InvalidFixture(
            "invalid fixture specification",
        ));
    }
    let split = fixture_split(spec.family, &spec.routing_topology_id, config);
    let n = spec.event_count;
    let frames = spec.frame_count;

    let actual: Vec<(usize, usize)> = (0..n)
        .map(|index| match spec.family {
            InteractionFixtureFamily::AdditiveNull
            | InteractionFixtureFamily::SharedRoutingDisjointTime => {
                let start = index * frames / n;
                (start, ((index + 1) * frames / n).max(start + 1))
            }
            InteractionFixtureFamily::SparsePairwise
            | InteractionFixtureFamily::UnderstatedSupport => {
                if index < 2 {
                    (frames / 8, frames * 3 / 8)
                } else {
                    let slots = (n - 2).max(1);
                    let start = frames / 2 + (index - 2) * (frames / 2) / slots;
                    (start, (start + (frames / 2 / slots)).min(frames).max(start + 1))
                }
            }
            InteractionFixtureFamily::LinearFirTail => {
                let start = 4 + index * 12;
                (start, (start + 8).min(frames))
            }
            InteractionFixtureFamily::PureHigherOrder => {
                if index < 3 {
                    (frames / 8, frames * 5 / 8)
                } else {
                    let slots = (n - 3).max(1);
                    let start = frames * 3 / 4 + (index - 3) * (frames / 4) / slots;
                    (start, (start + (frames / 4 / slots)).min(frames).max(start + 1))
                }
            }
            InteractionFixtureFamily::HardClipping
            | InteractionFixtureFamily::DenseOverlap
            | InteractionFixtureFamily::IncompleteRouting => (frames / 8, frames * 5 / 8),
            InteractionFixtureFamily::MixedSalience => {
                if index < 2 {
                    (frames / 8, frames * 5 / 8)
                } else {
                    let slots = (n - 2).max(1);
                    let start = frames * 3 / 4 + (index - 2) * (frames / 4) / slots;
                    (start, (start + (frames / 4 / slots)).min(frames).max(start + 1))
                }
            }
        })
        .collect();

    let mut rng = DeterministicRng(spec.seed ^ 0xa076_1d64_78bd_642f);
    let mut events = Vec::with_capacity(n);
    for (index, (start, end)) in actual.iter().copied().enumerate() {
        let phase = (rng.next() as f64 / u64::MAX as f64) * std::f64::consts::TAU;
        let frequency = 110.0 + index as f64 * 37.0 + (rng.next() % 17) as f64;
        let salience = if spec.family == InteractionFixtureFamily::MixedSalience {
            0.25 + ((index * 7 + 3) % 9) as f64 * 0.16
        } else {
            1.0 + (index % 3) as f64 * 0.07
        };
        let mut samples = vec![0.0; frames];
        for frame in start..end {
            let time = frame as f64 / 48_000.0;
            samples[frame] = spec.amplitude_scale
                * salience
                * (std::f64::consts::TAU * frequency * time + phase).sin();
        }
        events.push(SyntheticEventSignal {
            source_event_index: index,
            samples,
        });
    }

    let effect = match spec.family {
        InteractionFixtureFamily::AdditiveNull
        | InteractionFixtureFamily::SharedRoutingDisjointTime => SyntheticEffect::LinearMix,
        InteractionFixtureFamily::SparsePairwise
        | InteractionFixtureFamily::UnderstatedSupport
        | InteractionFixtureFamily::DenseOverlap => SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.45,
            cubic_coefficient: 0.0,
        },
        InteractionFixtureFamily::HardClipping
        | InteractionFixtureFamily::MixedSalience
        | InteractionFixtureFamily::IncompleteRouting => SyntheticEffect::HardClip {
            threshold: (spec.amplitude_scale * 1.4).max(0.025),
        },
        InteractionFixtureFamily::PureHigherOrder => SyntheticEffect::ThreeWayProduct {
            event_indices: [0, 1, 2],
            coefficient: 0.8,
        },
        InteractionFixtureFamily::LinearFirTail => SyntheticEffect::LinearFir {
            impulse_response: vec![1.0, 0.65, 0.35, 0.15, 0.05],
        },
    };

    let mut event_supports = Vec::with_capacity(n);
    for (index, (start, end)) in actual.iter().copied().enumerate() {
        let declared = if spec.family == InteractionFixtureFamily::UnderstatedSupport {
            if index == 0 {
                (start, start + (end - start) / 5)
            } else if index == 1 {
                (end - (end - start) / 5, end)
            } else {
                (start, end)
            }
        } else {
            (start, end)
        };
        let tail = if spec.family == InteractionFixtureFamily::LinearFirTail { 4 } else { 0 };
        let interval = (declared.0 as u64, (declared.1 + tail).min(frames + tail) as u64);
        let domains = match spec.family {
            InteractionFixtureFamily::SharedRoutingDisjointTime => {
                vec![format!("bus:{}", spec.routing_topology_id)]
            }
            InteractionFixtureFamily::DenseOverlap | InteractionFixtureFamily::HardClipping => {
                vec!["master:limiter".to_owned()]
            }
            _ => vec![format!("track:{index}")],
        };
        let routing_complete = spec.family != InteractionFixtureFamily::IncompleteRouting;
        let interval_vec = vec![interval];
        let evidence = support_digest(index, &interval_vec, &domains, routing_complete);
        event_supports.push(EventInteractionSupport {
            source_event_index: index,
            acoustic_support: vec![AcousticSupportInterval {
                start_frame: interval.0,
                end_frame: interval.1,
            }],
            routing_domains: domains,
            routing_complete,
            support_evidence_sha256: Some(evidence),
        });
    }

    let factorial_config = FactorialReferenceConfig {
        sample_rate_hz: 48_000,
        spectral_window_frames: 64,
    };
    let factorial = build_factorial_ground_truth(&events, &effect, &factorial_config)
        .map_err(InteractionProbeCampaignError::FactorialReference)?;
    let total_frames = factorial.output_frame_count as u64;
    if event_supports.iter().any(|support| {
        support.acoustic_support.iter().any(|interval| interval.end_frame > total_frames)
    }) {
        return Err(InteractionProbeCampaignError::InvalidFixture(
            "generated support exceeds factorial output horizon",
        ));
    }

    let source_bytes = serde_json::to_vec(&(
        &spec.fixture_id,
        &spec.composition_lineage_sha256,
        &events,
        &effect,
        &factorial_config,
        &event_supports,
    ))
    .map_err(|_| InteractionProbeCampaignError::SerializationFailure)?;
    let source_score_sha256 = sha256_hex(&source_bytes);
    let renderer_profile_sha256 = sha256_hex(
        format!("synthetic-renderer-v1:{}", spec.routing_topology_id).as_bytes(),
    );

    Ok(InteractionCampaignFixture {
        fixture_id: spec.fixture_id.clone(),
        composition_lineage_sha256: spec.composition_lineage_sha256.clone(),
        family: spec.family,
        routing_topology_id: spec.routing_topology_id.clone(),
        split,
        source_score_sha256,
        renderer_profile_sha256,
        total_frames,
        events,
        effect,
        factorial_config,
        event_supports,
    })
}

pub fn build_interaction_probe_campaign(
    fixtures: &[InteractionCampaignFixture],
    config: &InteractionProbeCampaignConfig,
) -> Result<InteractionProbeCampaignReport, InteractionProbeCampaignError> {
    validate_config(config)?;
    validate_fixtures(fixtures, config)?;
    let mut ordered = fixtures.to_vec();
    ordered.sort_by(|a, b| a.fixture_id.cmp(&b.fixture_id));
    let fixture_input_sha256 = canonical_digest(&ordered)?;

    let mut fixture_results = Vec::with_capacity(ordered.len());
    for fixture in &ordered {
        let factorial = build_factorial_ground_truth(
            &fixture.events,
            &fixture.effect,
            &fixture.factorial_config,
        )
        .map_err(InteractionProbeCampaignError::FactorialReference)?;
        verify_factorial_ground_truth(
            &factorial,
            &fixture.events,
            &fixture.effect,
            &fixture.factorial_config,
        )
        .map_err(InteractionProbeCampaignError::FactorialReference)?;

        let comparison = compare_probe_policies(
            &factorial,
            &fixture.events,
            &fixture.effect,
            &fixture.factorial_config,
            &fixture.source_score_sha256,
            &fixture.renderer_profile_sha256,
            fixture.total_frames,
            &fixture.event_supports,
            &config.planner_config,
            &config.budgets,
            &config.seeds,
            &config.metric_config,
        )
        .map_err(InteractionProbeCampaignError::ProbeComparison)?;

        fixture_results.push(CampaignFixtureResult {
            fixture_id: fixture.fixture_id.clone(),
            composition_lineage_sha256: fixture.composition_lineage_sha256.clone(),
            family: fixture.family,
            routing_topology_id: fixture.routing_topology_id.clone(),
            split: fixture.split,
            routing_complete: fixture.event_supports.iter().all(|support| support.routing_complete),
            event_count: comparison.event_count,
            possible_pair_count: comparison.possible_pair_count,
            contextual_positive_pair_count: comparison.contextual_positive_pair_count,
            global_order_two_harsanyi_count: comparison.global_order_two_harsanyi_count,
            higher_order_harsanyi_count: comparison.higher_order_harsanyi_count,
            factorial_report_sha256: factorial.receipt_sha256,
            comparison_report_sha256: comparison.receipt_sha256,
            rows: comparison.rows,
        });
    }

    let aggregates = build_aggregates(&fixture_results);
    let paired_contrasts = build_paired_contrasts(&fixture_results, config);
    let routing_summaries = build_routing_summaries(&fixture_results);
    let worst_family_performance = build_worst_family_performance(&aggregates);
    let composition_count = ordered
        .iter()
        .map(|fixture| fixture.composition_lineage_sha256.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let development_fixture_count = fixture_results
        .iter()
        .filter(|fixture| fixture.split == CampaignSplit::Development)
        .count();
    let held_out_fixture_count = fixture_results.len() - development_fixture_count;

    let mut report = InteractionProbeCampaignReport {
        version: INTERACTION_PROBE_CAMPAIGN_VERSION.to_owned(),
        config: config.clone(),
        fixture_input_sha256,
        fixture_count: fixture_results.len(),
        independent_composition_count: composition_count,
        development_fixture_count,
        held_out_fixture_count,
        held_out_families: config.held_out_families.iter().copied().collect(),
        held_out_routing_topologies: config.held_out_routing_topologies.iter().cloned().collect(),
        fixture_results,
        aggregates,
        paired_contrasts,
        routing_summaries,
        worst_family_performance,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = report_digest(&report)?;
    Ok(report)
}

pub fn verify_interaction_probe_campaign(
    report: &InteractionProbeCampaignReport,
    fixtures: &[InteractionCampaignFixture],
    config: &InteractionProbeCampaignConfig,
) -> Result<bool, InteractionProbeCampaignError> {
    if report.receipt_sha256.is_empty() || report_digest(report)? != report.receipt_sha256 {
        return Err(InteractionProbeCampaignError::InvalidReceipt);
    }
    let expected = build_interaction_probe_campaign(fixtures, config)?;
    if &expected != report {
        return Err(InteractionProbeCampaignError::ReplayMismatch);
    }
    Ok(true)
}

fn report_digest(report: &InteractionProbeCampaignReport) -> Result<String, InteractionProbeCampaignError> {
    let mut unsigned = report.clone();
    unsigned.receipt_sha256.clear();
    canonical_digest(&unsigned)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

type AggregateKey = (CampaignSplit, InteractionFixtureFamily, ProbeSelectionPolicy, usize);
type ClusterKey = (CampaignSplit, InteractionFixtureFamily, ProbeSelectionPolicy, usize, String);

fn build_aggregates(fixtures: &[CampaignFixtureResult]) -> Vec<CampaignPolicyAggregate> {
    let mut clusters = BTreeMap::<ClusterKey, ClusterMetric>::new();
    for fixture in fixtures {
        for row in &fixture.rows {
            let key = (
                fixture.split,
                fixture.family,
                row.policy,
                row.requested_pair_budget,
                fixture.composition_lineage_sha256.clone(),
            );
            let metric = clusters.entry(key).or_default();
            metric.observations += 1;
            metric.fixtures.insert(fixture.fixture_id.clone());
            if let Some(value) = row.precision { metric.precision.push(value); }
            if let Some(value) = row.recall { metric.recall.push(value); }
            metric.selected_pairs.push(row.selected_pair_count as f64);
            metric.render_cost_fraction.push(row.render_cost_fraction_of_exhaustive);
            metric.true_positives += row.true_positive_count as u128;
            metric.false_positives += row.false_positive_count as u128;
            metric.false_negatives += row.false_negative_count as u128;
        }
    }

    let mut grouped = BTreeMap::<AggregateKey, Vec<ClusterMetric>>::new();
    for ((split, family, policy, budget, _lineage), metric) in clusters {
        grouped.entry((split, family, policy, budget)).or_default().push(metric);
    }
    grouped
        .into_iter()
        .map(|((split, family, policy, requested_pair_budget), cluster_metrics)| {
            let precision = cluster_metrics.iter().filter_map(|m| mean(&m.precision)).collect::<Vec<_>>();
            let recall = cluster_metrics.iter().filter_map(|m| mean(&m.recall)).collect::<Vec<_>>();
            let pairs = cluster_metrics.iter().filter_map(|m| mean(&m.selected_pairs)).collect::<Vec<_>>();
            let costs = cluster_metrics.iter().filter_map(|m| mean(&m.render_cost_fraction)).collect::<Vec<_>>();
            CampaignPolicyAggregate {
                split,
                family,
                policy,
                requested_pair_budget,
                independent_composition_count: cluster_metrics.len(),
                fixture_count: cluster_metrics.iter().map(|m| m.fixtures.len()).sum(),
                observation_count: cluster_metrics.iter().map(|m| m.observations).sum(),
                precision_defined_composition_count: precision.len(),
                recall_defined_composition_count: recall.len(),
                mean_precision: mean(&precision),
                mean_recall: mean(&recall),
                mean_selected_pair_count: mean(&pairs).unwrap_or(0.0),
                mean_render_cost_fraction: mean(&costs).unwrap_or(0.0),
                total_true_positives: cluster_metrics.iter().map(|m| m.true_positives).sum(),
                total_false_positives: cluster_metrics.iter().map(|m| m.false_positives).sum(),
                total_false_negatives: cluster_metrics.iter().map(|m| m.false_negatives).sum(),
            }
        })
        .collect()
}

fn bootstrap_interval(
    values: &[f64],
    replicates: usize,
    seed: u64,
    confidence_permille: u16,
) -> Option<(f64, f64)> {
    if values.len() < 2 || replicates < 200 { return None; }
    let mut rng = DeterministicRng(seed);
    let mut means = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut sum = 0.0;
        for _ in 0..values.len() { sum += values[rng.below(values.len())]; }
        means.push(sum / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    let tail = (1_000u16 - confidence_permille) as f64 / 2_000.0;
    Some((quantile_sorted(&means, tail), quantile_sorted(&means, 1.0 - tail)))
}

fn quantile_sorted(values: &[f64], q: f64) -> f64 {
    if values.len() == 1 { return values[0]; }
    let position = q.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    let weight = position - lo as f64;
    values[lo] * (1.0 - weight) + values[hi] * weight
}

fn build_paired_contrasts(
    fixtures: &[CampaignFixtureResult],
    config: &InteractionProbeCampaignConfig,
) -> Vec<PairedPolicyContrast> {
    let baselines = [ProbeSelectionPolicy::UniformRandom, ProbeSelectionPolicy::TemporalOnly];
    let mut output = Vec::new();
    for split in [CampaignSplit::Development, CampaignSplit::HeldOut] {
        let families = fixtures.iter().filter(|f| f.split == split)
            .map(|f| f.family).collect::<BTreeSet<_>>();
        for family in families {
            for baseline_policy in baselines {
                let mut cluster_deltas = BTreeMap::<String, Vec<f64>>::new();
                let mut matched_blocks = 0usize;
                let mut unmatched = 0usize;
                for fixture in fixtures.iter().filter(|f| f.split == split && f.family == family) {
                    for candidate in fixture.rows.iter().filter(|r| r.policy == ProbeSelectionPolicy::Eval019) {
                        let baseline = fixture.rows.iter().find(|r| {
                            r.policy == baseline_policy
                                && r.seed == candidate.seed
                                && r.requested_pair_budget == candidate.requested_pair_budget
                        });
                        let Some(baseline) = baseline else { unmatched += 1; continue; };
                        if candidate.total_render_count != baseline.total_render_count
                            || candidate.recall.is_none()
                            || baseline.recall.is_none()
                        {
                            unmatched += 1;
                            continue;
                        }
                        cluster_deltas.entry(fixture.composition_lineage_sha256.clone())
                            .or_default()
                            .push(candidate.recall.unwrap_or(0.0) - baseline.recall.unwrap_or(0.0));
                        matched_blocks += 1;
                    }
                }
                let cluster_values = cluster_deltas.values()
                    .filter_map(|values| mean(values)).collect::<Vec<_>>();
                let mean_delta = mean(&cluster_values);
                let interval = bootstrap_interval(
                    &cluster_values,
                    config.bootstrap_replicates,
                    config.bootstrap_seed
                        ^ (family as u64).wrapping_mul(0x9e37_79b9)
                        ^ (baseline_policy as u64).wrapping_mul(0x85eb_ca6b)
                        ^ (split as u64),
                    config.confidence_level_permille,
                );
                output.push(PairedPolicyContrast {
                    split,
                    family,
                    candidate_policy: ProbeSelectionPolicy::Eval019,
                    baseline_policy,
                    matched_composition_count: cluster_values.len(),
                    matched_block_count: matched_blocks,
                    unmatched_or_undefined_block_count: unmatched,
                    mean_recall_delta_candidate_minus_baseline: mean_delta,
                    confidence_interval_low: interval.map(|i| i.0),
                    confidence_interval_high: interval.map(|i| i.1),
                    bootstrap_replicates: config.bootstrap_replicates,
                });
            }
        }
    }
    output
}

fn build_routing_summaries(fixtures: &[CampaignFixtureResult]) -> Vec<RoutingStratumSummary> {
    type Key = (CampaignSplit, InteractionFixtureFamily, bool, ProbeSelectionPolicy, usize);
    let mut groups = BTreeMap::<Key, Vec<&ProbePolicyComparisonRow>>::new();
    let mut lineages = BTreeMap::<Key, BTreeSet<String>>::new();
    for fixture in fixtures {
        for row in &fixture.rows {
            let key = (fixture.split, fixture.family, fixture.routing_complete,
                row.policy, row.requested_pair_budget);
            groups.entry(key).or_default().push(row);
            lineages.entry(key).or_default().insert(fixture.composition_lineage_sha256.clone());
        }
    }
    groups.into_iter().map(|(key, rows)| {
        let (split, family, routing_complete, policy, requested_pair_budget) = key;
        let tp = rows.iter().map(|r| r.true_positive_count as u128).sum::<u128>();
        let fp = rows.iter().map(|r| r.false_positive_count as u128).sum::<u128>();
        let fn_count = rows.iter().map(|r| r.false_negative_count as u128).sum::<u128>();
        let positives = tp + fn_count;
        let discoveries = tp + fp;
        RoutingStratumSummary {
            split,
            family,
            routing_complete,
            policy,
            requested_pair_budget,
            independent_composition_count: lineages.get(&key).map_or(0, BTreeSet::len),
            row_count: rows.len(),
            true_positive_count: tp,
            false_positive_count: fp,
            false_negative_count: fn_count,
            missed_interaction_rate: (positives > 0).then_some(fn_count as f64 / positives as f64),
            false_discovery_rate: (discoveries > 0).then_some(fp as f64 / discoveries as f64),
        }
    }).collect()
}

fn build_worst_family_performance(
    aggregates: &[CampaignPolicyAggregate],
) -> Vec<WorstFamilyPerformance> {
    type Key = (CampaignSplit, ProbeSelectionPolicy, usize);
    let mut groups = BTreeMap::<Key, Vec<&CampaignPolicyAggregate>>::new();
    for aggregate in aggregates {
        if aggregate.mean_recall.is_some() {
            groups.entry((aggregate.split, aggregate.policy, aggregate.requested_pair_budget))
                .or_default().push(aggregate);
        }
    }
    groups.into_iter().filter_map(|((split, policy, budget), entries)| {
        entries.into_iter().min_by(|a, b| {
            a.mean_recall.unwrap_or(1.0).total_cmp(&b.mean_recall.unwrap_or(1.0))
                .then_with(|| a.family.cmp(&b.family))
        }).map(|worst| WorstFamilyPerformance {
            split,
            policy,
            requested_pair_budget: budget,
            family: worst.family,
            mean_recall: worst.mean_recall.unwrap_or(0.0),
            independent_composition_count: worst.independent_composition_count,
        })
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(label: &str) -> String { sha256_hex(label.as_bytes()) }

    fn config() -> InteractionProbeCampaignConfig {
        InteractionProbeCampaignConfig {
            planner_config: InteractionProbeConfig {
                pair_probe_budget: 3,
                routing_group_sample_budget: 3,
                audit_pair_budget: 1,
                max_sampling_attempts: 5_000,
                seed: 17,
                max_temporal_candidate_pairs: 256,
            },
            budgets: vec![0, 1, 2],
            seeds: vec![3, 7, 11],
            metric_config: InteractionProbeBenchmarkConfig::default(),
            held_out_families: [InteractionFixtureFamily::PureHigherOrder].into_iter().collect(),
            held_out_routing_topologies: ["route:holdout".to_owned()].into_iter().collect(),
            minimum_compositions_per_stratum: 2,
            bootstrap_replicates: 200,
            bootstrap_seed: 99,
            confidence_level_permille: 950,
        }
    }

    fn spec(
        id: &str,
        lineage: &str,
        family: InteractionFixtureFamily,
        topology: &str,
    ) -> CampaignFixtureSpec {
        CampaignFixtureSpec {
            fixture_id: id.to_owned(),
            composition_lineage_sha256: digest(lineage),
            family,
            routing_topology_id: topology.to_owned(),
            seed: 101,
            event_count: if family == InteractionFixtureFamily::PureHigherOrder { 3 } else { 4 },
            frame_count: 128,
            amplitude_scale: 0.12,
        }
    }

    fn fixtures(config: &InteractionProbeCampaignConfig) -> Vec<InteractionCampaignFixture> {
        let mut result = Vec::new();
        for index in 0..2 {
            result.push(generate_campaign_fixture(
                &spec(
                    &format!("dev-add-{index}"),
                    &format!("lineage-dev-add-{index}"),
                    InteractionFixtureFamily::AdditiveNull,
                    "route:dev",
                ),
                config,
            ).unwrap());
            result.push(generate_campaign_fixture(
                &spec(
                    &format!("dev-sparse-{index}"),
                    &format!("lineage-dev-sparse-{index}"),
                    InteractionFixtureFamily::SparsePairwise,
                    "route:dev",
                ),
                config,
            ).unwrap());
            result.push(generate_campaign_fixture(
                &spec(
                    &format!("holdout-high-{index}"),
                    &format!("lineage-holdout-high-{index}"),
                    InteractionFixtureFamily::PureHigherOrder,
                    "route:holdout",
                ),
                config,
            ).unwrap());
        }
        result
    }

    #[test]
    fn deterministic_generator_separates_oracle_truth_from_understated_support() {
        let config = config();
        let normal = generate_campaign_fixture(
            &spec("normal", "lineage-normal", InteractionFixtureFamily::SparsePairwise, "route:dev"),
            &config,
        ).unwrap();
        let understated = generate_campaign_fixture(
            &spec("understated", "lineage-under", InteractionFixtureFamily::UnderstatedSupport, "route:dev"),
            &config,
        ).unwrap();
        assert_eq!(normal.events, understated.events);
        assert_ne!(normal.event_supports, understated.event_supports);
        assert_eq!(normal.total_frames, understated.total_frames);
    }

    #[test]
    fn campaign_pairs_policies_across_independent_compositions_and_replays() {
        let config = config();
        let fixtures = fixtures(&config);
        let report = build_interaction_probe_campaign(&fixtures, &config).unwrap();
        assert_eq!(report.fixture_count, 6);
        assert_eq!(report.independent_composition_count, 6);
        assert_eq!(report.development_fixture_count, 4);
        assert_eq!(report.held_out_fixture_count, 2);
        assert!(!report.aggregates.is_empty());
        assert!(!report.paired_contrasts.is_empty());
        assert!(verify_interaction_probe_campaign(&report, &fixtures, &config).unwrap());
    }

    #[test]
    fn held_out_composition_lineage_cannot_leak_into_development() {
        let config = config();
        let mut fixtures = fixtures(&config);
        let reused_lineage = fixtures[0].composition_lineage_sha256.clone();
        fixtures[0].split = CampaignSplit::Development;
        let held = fixtures.iter_mut()
            .find(|fixture| fixture.split == CampaignSplit::HeldOut).unwrap();
        held.composition_lineage_sha256 = reused_lineage;
        assert_eq!(
            build_interaction_probe_campaign(&fixtures, &config).unwrap_err(),
            InteractionProbeCampaignError::CompositionLineageCrossesSplits
        );
    }

    #[test]
    fn family_and_topology_holdouts_must_match_the_preregistered_split() {
        let config = config();
        let mut fixtures = fixtures(&config);
        fixtures[2].split = CampaignSplit::Development;
        assert_eq!(
            build_interaction_probe_campaign(&fixtures, &config).unwrap_err(),
            InteractionProbeCampaignError::FixtureSplitDoesNotMatchPreregisteredHoldout
        );
    }

    #[test]
    fn same_policy_rows_are_compared_only_at_identical_actual_render_cost() {
        let config = config();
        let report = build_interaction_probe_campaign(&fixtures(&config), &config).unwrap();
        assert!(report.paired_contrasts.iter().all(|contrast| {
            contrast.matched_block_count + contrast.unmatched_or_undefined_block_count > 0
        }));
        assert!(report.paired_contrasts.iter().all(|contrast| {
            contrast.matched_composition_count <= 2
        }));
    }

    #[test]
    fn insufficient_independent_compositions_never_get_cluster_intervals() {
        assert_eq!(
            bootstrap_interval(&[0.5], 200, 7, 950),
            None
        );
        assert!(bootstrap_interval(&[0.1, 0.2, 0.3], 200, 7, 950).is_some());
    }

    #[test]
    fn report_tampering_is_rejected() {
        let config = config();
        let fixtures = fixtures(&config);
        let mut report = build_interaction_probe_campaign(&fixtures, &config).unwrap();
        report.held_out_fixture_count += 1;
        assert_eq!(
            verify_interaction_probe_campaign(&report, &fixtures, &config).unwrap_err(),
            InteractionProbeCampaignError::InvalidReceipt
        );
    }

    #[test]
    fn pure_three_way_family_keeps_global_pairwise_and_higher_order_truth_separate() {
        let config = config();
        let fixture = generate_campaign_fixture(
            &spec("holdout-three", "lineage-three", InteractionFixtureFamily::PureHigherOrder, "route:holdout"),
            &config,
        ).unwrap();
        let report = build_factorial_ground_truth(
            &fixture.events,
            &fixture.effect,
            &fixture.factorial_config,
        ).unwrap();
        assert_eq!(
            report.dividends.iter().filter(|d| d.order == 2 && d.pcm_l2_norm > 1.0e-9).count(),
            0
        );
        assert!(report.dividends.iter().any(|d| d.order >= 3 && d.pcm_l2_norm > 1.0e-9));
    }

    #[test]
    fn incomplete_routing_is_reported_as_its_own_stratum() {
        let config = config();
        let fixture = generate_campaign_fixture(
            &spec("dev-incomplete", "lineage-incomplete", InteractionFixtureFamily::IncompleteRouting, "route:dev"),
            &config,
        ).unwrap();
        assert!(fixture.event_supports.iter().all(|event| !event.routing_complete));
    }
}

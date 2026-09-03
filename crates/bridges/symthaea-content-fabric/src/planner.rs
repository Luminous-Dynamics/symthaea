use std::collections::BTreeMap;

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::model::{
    ActionRefV1, ExternalPlannerCandidateV1, ExternalPlannerRecommendationV1,
    ExternalPlannerRequestV1, HdcCandidateTraceV1, HdcShadowPlanV1, HdcShadowTraceV1,
    EXTERNAL_PLANNER_PROTOCOL_V1, PENALTY_PPM_MAX,
};
use crate::protocol::{selection_satisfies_local_policy, validate_request_v1, ProtocolErrorV1};

pub const ENGINE_ID_V1: &str = "symthaea-hdc-shadow";
pub const ENGINE_VERSION_V1: &str = "0.1.0";

const COST_GOOD_SEED: u64 = 0x4346_3036_4400_0001;
const COST_BAD_SEED: u64 = 0x4346_3036_4400_0011;
const LATENCY_GOOD_SEED: u64 = 0x4346_3036_4400_0002;
const LATENCY_BAD_SEED: u64 = 0x4346_3036_4400_0012;
const ENERGY_GOOD_SEED: u64 = 0x4346_3036_4400_0003;
const ENERGY_BAD_SEED: u64 = 0x4346_3036_4400_0013;
const LOCALITY_GOOD_SEED: u64 = 0x4346_3036_4400_0004;
const LOCALITY_BAD_SEED: u64 = 0x4346_3036_4400_0014;

#[derive(Debug, Clone)]
struct RankedCandidate {
    action: ActionRefV1,
    baseline_rank: u64,
    baseline_weighted_penalty: u64,
    similarity: f32,
}

#[derive(Debug, Clone)]
struct MetricAnchors {
    good: ContinuousHV,
    bad: ContinuousHV,
}

fn local_full_cosine(a: &ContinuousHV, b: &ContinuousHV) -> f32 {
    if a.values.len() != b.values.len() || a.values.is_empty() {
        return -1.0;
    }
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for (&left, &right) in a.values.iter().zip(&b.values) {
        let left = f64::from(left);
        let right = f64::from(right);
        dot += left * right;
        norm_a += left * left;
        norm_b += right * right;
    }
    if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
        return -1.0;
    }
    let value = dot / (norm_a.sqrt() * norm_b.sqrt());
    if value.is_finite() {
        value.clamp(-1.0, 1.0) as f32
    } else {
        -1.0
    }
}

fn quality(penalty_ppm: u32) -> f32 {
    let bounded = penalty_ppm.min(PENALTY_PPM_MAX);
    (PENALTY_PPM_MAX - bounded) as f32 / PENALTY_PPM_MAX as f32
}

fn anchors(good_seed: u64, bad_seed: u64) -> MetricAnchors {
    MetricAnchors {
        good: ContinuousHV::random(HDC_DIMENSION, good_seed),
        bad: ContinuousHV::random(HDC_DIMENSION, bad_seed),
    }
}

fn metric_anchors() -> [MetricAnchors; 4] {
    [
        anchors(COST_GOOD_SEED, COST_BAD_SEED),
        anchors(LATENCY_GOOD_SEED, LATENCY_BAD_SEED),
        anchors(ENERGY_GOOD_SEED, ENERGY_BAD_SEED),
        anchors(LOCALITY_GOOD_SEED, LOCALITY_BAD_SEED),
    ]
}

fn preference_weights(request: &ExternalPlannerRequestV1) -> [f32; 4] {
    [
        f32::from(request.preferences.cost_weight),
        f32::from(request.preferences.latency_weight),
        f32::from(request.preferences.energy_weight),
        f32::from(request.preferences.locality_weight),
    ]
}

fn interpolate_metric(anchor: &MetricAnchors, quality: f32) -> ContinuousHV {
    let weights = [quality, 1.0 - quality];
    ContinuousHV::weighted_bundle(&[&anchor.good, &anchor.bad], &weights)
}

fn ideal_hv(anchors: &[MetricAnchors; 4], weights: &[f32; 4]) -> ContinuousHV {
    let refs = [
        &anchors[0].good,
        &anchors[1].good,
        &anchors[2].good,
        &anchors[3].good,
    ];
    ContinuousHV::weighted_bundle(&refs, weights)
}

fn candidate_hv(
    anchors: &[MetricAnchors; 4],
    preference_weights: &[f32; 4],
    candidate: &ExternalPlannerCandidateV1,
) -> ContinuousHV {
    let metrics = [
        interpolate_metric(&anchors[0], quality(candidate.cost_penalty_ppm)),
        interpolate_metric(&anchors[1], quality(candidate.latency_penalty_ppm)),
        interpolate_metric(&anchors[2], quality(candidate.energy_penalty_ppm)),
        interpolate_metric(&anchors[3], quality(candidate.locality_penalty_ppm)),
    ];
    let refs = [&metrics[0], &metrics[1], &metrics[2], &metrics[3]];
    ContinuousHV::weighted_bundle(&refs, preference_weights)
}

fn hdc_ranking(request: &ExternalPlannerRequestV1) -> Vec<RankedCandidate> {
    if request.preferences.total_weight() == 0 {
        return request
            .candidates
            .iter()
            .map(|candidate| RankedCandidate {
                action: candidate.availability_action.clone(),
                baseline_rank: candidate.baseline_rank,
                baseline_weighted_penalty: candidate.baseline_weighted_penalty,
                similarity: 0.0,
            })
            .collect();
    }

    let anchors = metric_anchors();
    let weights = preference_weights(request);
    let ideal = ideal_hv(&anchors, &weights);
    let mut ranking = request
        .candidates
        .iter()
        .map(|candidate| {
            let encoded = candidate_hv(&anchors, &weights, candidate);
            RankedCandidate {
                action: candidate.availability_action.clone(),
                baseline_rank: candidate.baseline_rank,
                baseline_weighted_penalty: candidate.baseline_weighted_penalty,
                similarity: local_full_cosine(&encoded, &ideal),
            }
        })
        .collect::<Vec<_>>();
    ranking.sort_by(|left, right| {
        right
            .similarity
            .total_cmp(&left.similarity)
            .then_with(|| left.baseline_rank.cmp(&right.baseline_rank))
            .then_with(|| left.action.cmp(&right.action))
    });
    ranking
}

fn policy_preserving_selection(
    request: &ExternalPlannerRequestV1,
    ranking: &[RankedCandidate],
) -> Vec<ActionRefV1> {
    if request.preferences.total_weight() == 0 {
        return request.baseline_selected_availability_actions.clone();
    }

    let mut selected = ranking
        .iter()
        .map(|candidate| candidate.action.clone())
        .collect::<Vec<_>>();
    for candidate in ranking.iter().rev() {
        let tentative = selected
            .iter()
            .filter(|action| *action != &candidate.action)
            .cloned()
            .collect::<Vec<_>>();
        if selection_satisfies_local_policy(request, &tentative) {
            selected = tentative;
        }
    }
    selected
}

pub fn plan_hdc_shadow_v1(
    request: &ExternalPlannerRequestV1,
) -> Result<HdcShadowPlanV1, ProtocolErrorV1> {
    validate_request_v1(request)?;
    let ranking = hdc_ranking(request);
    let selected = policy_preserving_selection(request, &ranking);

    // This local check is advisory defense-in-depth only. Mycelix remains the
    // authority that revalidates the returned subset through CF-06A.
    if !selection_satisfies_local_policy(request, &selected) {
        return Err(ProtocolErrorV1::InvalidBaselineSelection);
    }

    let ranking_actions = ranking
        .iter()
        .map(|candidate| candidate.action.clone())
        .collect::<Vec<_>>();
    let recommendation = ExternalPlannerRecommendationV1 {
        schema_version: EXTERNAL_PLANNER_PROTOCOL_V1,
        request_id: request.id,
        planner_input_id: request.planner_input_id,
        engine_id: ENGINE_ID_V1.to_string(),
        engine_version: ENGINE_VERSION_V1.to_string(),
        ranking: ranking_actions.clone(),
        selected_availability_actions: selected.clone(),
    };

    let hdc_positions = ranking
        .iter()
        .enumerate()
        .map(|(index, candidate)| (candidate.action.clone(), index as u64))
        .collect::<BTreeMap<_, _>>();
    let candidate_traces = ranking
        .iter()
        .map(|candidate| HdcCandidateTraceV1 {
            availability_action: candidate.action.clone(),
            baseline_rank: candidate.baseline_rank,
            hdc_rank: *hdc_positions
                .get(&candidate.action)
                .expect("ranked candidate has an HDC position"),
            similarity_to_ideal: candidate.similarity,
            baseline_weighted_penalty: candidate.baseline_weighted_penalty,
        })
        .collect::<Vec<_>>();
    let baseline_ranking = request
        .candidates
        .iter()
        .map(|candidate| candidate.availability_action.clone())
        .collect::<Vec<_>>();
    let trace = HdcShadowTraceV1 {
        baseline_selected_availability_actions: request
            .baseline_selected_availability_actions
            .clone(),
        hdc_selected_availability_actions: selected,
        ranking_changed: ranking_actions != baseline_ranking,
        selection_changed: recommendation.selected_availability_actions
            != request.baseline_selected_availability_actions,
        candidates: candidate_traces,
    };

    Ok(HdcShadowPlanV1 {
        recommendation,
        trace,
    })
}

pub fn recommend_json_v1(request_json: &str) -> Result<String, ProtocolErrorV1> {
    let request = crate::protocol::decode_request_json_v1(request_json)?;
    let plan = plan_hdc_shadow_v1(&request)?;
    serde_json::to_string(&plan.recommendation)
        .map_err(|error| ProtocolErrorV1::Json(error.to_string()))
}

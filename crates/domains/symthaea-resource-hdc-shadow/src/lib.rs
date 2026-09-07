// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic HDC shadow selection over an evidence-qualified resource context.
//!
//! This is a research selector, not a feasibility or authority layer. It consumes
//! the exact closed feasible set and qualified metric vectors retained by the
//! resource Pareto context, normalizes them through an explicit profile, encodes
//! them with deterministic metric-derived HDC anchors, and emits only the ordinary
//! closed-set `ResourcePlannerProposal` contract.
//!
//! The implementation deliberately mirrors the proven representation pattern from
//! the Content Fabric HDC shadow planner: distinct good/worst anchors, weighted
//! bundling, and a local full-vector cosine independent of runtime-global cognitive
//! throttling. No claim that HDC is superior is made here.

#![deny(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_resource_objective_metric::{
    ObjectiveMetric, ObjectiveMetricSchema, ObjectiveMetricSchemaError, ObjectiveStatistic,
};
use symthaea_resource_objective_normalization::{
    normalize_objective_vector, NormalizationError, NormalizedObjectiveVector,
    ObjectiveNormalizationProfile,
};
use symthaea_resource_pareto::ResourceParetoRanking;
use symthaea_resource_recommendation::{
    validate_resource_planner_proposal, RecommendationError, ResourcePlannerProposal,
    ValidatedResourcePlannerDecision,
};
use symthaea_resource_selectors::canonical_feasible_proposal;
use thiserror::Error;

pub const HDC_SHADOW_SELECTOR_ID: &str = "hdc/exact-metric-shadow-v1";
const ANCHOR_DOMAIN_V1: &[u8] = b"symthaea.resource.hdc.anchor.v1";

/// One exact metric's explicit HDC preference weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdcObjectivePreference {
    pub metric: ObjectiveMetric,
    pub weight: u16,
}

/// Complete exact-metric HDC preference schema.
///
/// Every metric must be present even when its weight is zero, so an omitted
/// dimension cannot silently acquire zero weight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdcPreferenceProfile {
    metric_schema: ObjectiveMetricSchema,
    preferences: BTreeMap<String, HdcObjectivePreference>,
}

impl HdcPreferenceProfile {
    pub fn new(
        preferences: impl IntoIterator<Item = HdcObjectivePreference>,
    ) -> Result<Self, HdcPreferenceError> {
        let collected: Vec<HdcObjectivePreference> = preferences.into_iter().collect();
        if collected.is_empty() {
            return Err(HdcPreferenceError::EmptyProfile);
        }
        let metric_schema = ObjectiveMetricSchema::new(
            collected.iter().map(|preference| preference.metric.clone()),
        )
        .map_err(HdcPreferenceError::InvalidMetricSchema)?;

        let mut by_name = BTreeMap::new();
        for preference in collected {
            let previous = by_name.insert(
                preference.metric.objective_name.clone(),
                preference,
            );
            debug_assert!(previous.is_none());
        }
        Ok(Self {
            metric_schema,
            preferences: by_name,
        })
    }

    pub fn metric_schema(&self) -> &ObjectiveMetricSchema {
        &self.metric_schema
    }

    pub fn get(&self, objective_name: &str) -> Option<&HdcObjectivePreference> {
        self.preferences.get(objective_name)
    }

    pub fn preferences(&self) -> impl Iterator<Item = &HdcObjectivePreference> {
        self.preferences.values()
    }

    pub fn total_weight(&self) -> u64 {
        self.preferences
            .values()
            .map(|preference| u64::from(preference.weight))
            .sum()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum HdcPreferenceError {
    #[error("HDC preference profile must contain at least one exact metric")]
    EmptyProfile,
    #[error("invalid HDC preference metric schema: {0}")]
    InvalidMetricSchema(ObjectiveMetricSchemaError),
}

#[derive(Debug, Clone)]
struct MetricAnchors {
    good: ContinuousHV,
    bad: ContinuousHV,
}

/// Per-candidate research trace. Similarity is diagnostic only and never appears
/// in the recommendation contract.
#[derive(Debug, Clone, PartialEq)]
pub struct HdcCandidateTrace {
    candidate_id: String,
    hdc_rank: usize,
    similarity_to_ideal: f32,
    normalized: NormalizedObjectiveVector,
}

impl HdcCandidateTrace {
    pub fn candidate_id(&self) -> &str {
        &self.candidate_id
    }

    pub fn hdc_rank(&self) -> usize {
        self.hdc_rank
    }

    pub fn similarity_to_ideal(&self) -> f32 {
        self.similarity_to_ideal
    }

    pub fn normalized(&self) -> &NormalizedObjectiveVector {
        &self.normalized
    }
}

/// Complete recommendation-only HDC shadow result bound to the exact source
/// ranking/evidence context and normalization/preference semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct HdcShadowPlan {
    ranking_context: ResourceParetoRanking,
    normalization_profile: ObjectiveNormalizationProfile,
    preference_profile: HdcPreferenceProfile,
    traces: Vec<HdcCandidateTrace>,
    proposal: ResourcePlannerProposal,
    validated_decision: ValidatedResourcePlannerDecision,
    zero_weight_baseline_fallback: bool,
    selection_changed_from_canonical_baseline: bool,
}

impl HdcShadowPlan {
    pub fn ranking_context(&self) -> &ResourceParetoRanking {
        &self.ranking_context
    }

    pub fn normalization_profile(&self) -> &ObjectiveNormalizationProfile {
        &self.normalization_profile
    }

    pub fn preference_profile(&self) -> &HdcPreferenceProfile {
        &self.preference_profile
    }

    pub fn traces(&self) -> &[HdcCandidateTrace] {
        &self.traces
    }

    pub fn proposal(&self) -> &ResourcePlannerProposal {
        &self.proposal
    }

    pub fn validated_decision(&self) -> &ValidatedResourcePlannerDecision {
        &self.validated_decision
    }

    pub fn zero_weight_baseline_fallback(&self) -> bool {
        self.zero_weight_baseline_fallback
    }

    pub fn selection_changed_from_canonical_baseline(&self) -> bool {
        self.selection_changed_from_canonical_baseline
    }
}

/// Produce one deterministic HDC shadow recommendation over every member of the
/// retained closed feasible subset.
pub fn plan_hdc_shadow(
    ranking: &ResourceParetoRanking,
    normalization: &ObjectiveNormalizationProfile,
    preferences: &HdcPreferenceProfile,
) -> Result<HdcShadowPlan, HdcShadowError> {
    validate_context(ranking, normalization, preferences)?;
    let feasible_set = ranking.evidence_set().feasible_set();
    let baseline = canonical_feasible_proposal(feasible_set);

    if preferences.total_weight() == 0 {
        let validated = validate_resource_planner_proposal(feasible_set, baseline.clone())
            .map_err(HdcShadowError::Recommendation)?;
        return Ok(HdcShadowPlan {
            ranking_context: ranking.clone(),
            normalization_profile: normalization.clone(),
            preference_profile: preferences.clone(),
            traces: Vec::new(),
            proposal: baseline,
            validated_decision: validated,
            zero_weight_baseline_fallback: true,
            selection_changed_from_canonical_baseline: false,
        });
    }

    if feasible_set.feasible().is_empty() {
        let proposal = ResourcePlannerProposal::Abstain {
            selector_id: HDC_SHADOW_SELECTOR_ID.into(),
        };
        let validated = validate_resource_planner_proposal(feasible_set, proposal.clone())
            .map_err(HdcShadowError::Recommendation)?;
        return Ok(HdcShadowPlan {
            ranking_context: ranking.clone(),
            normalization_profile: normalization.clone(),
            preference_profile: preferences.clone(),
            traces: Vec::new(),
            selection_changed_from_canonical_baseline: proposal != baseline,
            proposal,
            validated_decision: validated,
            zero_weight_baseline_fallback: false,
        });
    }

    let anchors = build_anchors(ranking);
    let ordered_weights = preference_weights(ranking, preferences)?;
    let ideal = ideal_hv(ranking, &anchors, &ordered_weights)?;

    let mut traces = Vec::new();
    for feasible in feasible_set.feasible() {
        let candidate_id = feasible.candidate_id();
        let qualified = ranking
            .qualified_vector(candidate_id)
            .ok_or_else(|| HdcShadowError::MissingQualifiedVector(candidate_id.to_owned()))?;
        let normalized = normalize_objective_vector(qualified, normalization).map_err(|source| {
            HdcShadowError::Normalization {
                candidate_id: candidate_id.to_owned(),
                source,
            }
        })?;
        let encoded = candidate_hv(ranking, &anchors, &ordered_weights, &normalized)?;
        traces.push(HdcCandidateTrace {
            candidate_id: candidate_id.to_owned(),
            hdc_rank: 0,
            similarity_to_ideal: local_full_cosine(&encoded, &ideal),
            normalized,
        });
    }

    traces.sort_by(|left, right| {
        right
            .similarity_to_ideal
            .total_cmp(&left.similarity_to_ideal)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    for (index, trace) in traces.iter_mut().enumerate() {
        trace.hdc_rank = index;
    }

    let proposal = ResourcePlannerProposal::Recommend {
        selector_id: HDC_SHADOW_SELECTOR_ID.into(),
        candidate_id: traces[0].candidate_id.clone(),
    };
    let validated = validate_resource_planner_proposal(feasible_set, proposal.clone())
        .map_err(HdcShadowError::Recommendation)?;
    let changed = proposal != baseline;

    Ok(HdcShadowPlan {
        ranking_context: ranking.clone(),
        normalization_profile: normalization.clone(),
        preference_profile: preferences.clone(),
        traces,
        proposal,
        validated_decision: validated,
        zero_weight_baseline_fallback: false,
        selection_changed_from_canonical_baseline: changed,
    })
}

fn validate_context(
    ranking: &ResourceParetoRanking,
    normalization: &ObjectiveNormalizationProfile,
    preferences: &HdcPreferenceProfile,
) -> Result<(), HdcShadowError> {
    if normalization.metric_schema() != ranking.metric_schema() {
        return Err(HdcShadowError::NormalizationSchemaMismatch);
    }
    if preferences.metric_schema() != ranking.metric_schema() {
        return Err(HdcShadowError::PreferenceSchemaMismatch);
    }

    for objective in ranking.objectives() {
        let name = &objective.metric.objective_name;
        let band = normalization
            .get(name)
            .ok_or_else(|| HdcShadowError::MissingNormalizationBand(name.clone()))?;
        if band.metric != objective.metric || band.direction != objective.direction {
            return Err(HdcShadowError::NormalizationDirectionMismatch {
                objective_name: name.clone(),
            });
        }
        let preference = preferences
            .get(name)
            .ok_or_else(|| HdcShadowError::MissingPreference(name.clone()))?;
        if preference.metric != objective.metric {
            return Err(HdcShadowError::PreferenceMetricMismatch {
                objective_name: name.clone(),
            });
        }
    }
    Ok(())
}

fn build_anchors(ranking: &ResourceParetoRanking) -> BTreeMap<String, MetricAnchors> {
    ranking
        .objectives()
        .iter()
        .map(|objective| {
            let metric = &objective.metric;
            (
                metric.objective_name.clone(),
                MetricAnchors {
                    good: ContinuousHV::random(HDC_DIMENSION, anchor_seed(metric, b"good")),
                    bad: ContinuousHV::random(HDC_DIMENSION, anchor_seed(metric, b"bad")),
                },
            )
        })
        .collect()
}

fn preference_weights(
    ranking: &ResourceParetoRanking,
    preferences: &HdcPreferenceProfile,
) -> Result<Vec<f32>, HdcShadowError> {
    ranking
        .objectives()
        .iter()
        .map(|objective| {
            preferences
                .get(&objective.metric.objective_name)
                .map(|preference| f32::from(preference.weight))
                .ok_or_else(|| {
                    HdcShadowError::MissingPreference(objective.metric.objective_name.clone())
                })
        })
        .collect()
}

fn ideal_hv(
    ranking: &ResourceParetoRanking,
    anchors: &BTreeMap<String, MetricAnchors>,
    weights: &[f32],
) -> Result<ContinuousHV, HdcShadowError> {
    let refs: Result<Vec<&ContinuousHV>, HdcShadowError> = ranking
        .objectives()
        .iter()
        .map(|objective| {
            anchors
                .get(&objective.metric.objective_name)
                .map(|anchor| &anchor.good)
                .ok_or_else(|| {
                    HdcShadowError::MissingAnchor(objective.metric.objective_name.clone())
                })
        })
        .collect();
    Ok(ContinuousHV::weighted_bundle(&refs?, weights))
}

fn candidate_hv(
    ranking: &ResourceParetoRanking,
    anchors: &BTreeMap<String, MetricAnchors>,
    weights: &[f32],
    normalized: &NormalizedObjectiveVector,
) -> Result<ContinuousHV, HdcShadowError> {
    let mut metrics = Vec::with_capacity(ranking.objectives().len());
    for objective in ranking.objectives() {
        let name = &objective.metric.objective_name;
        let anchor = anchors
            .get(name)
            .ok_or_else(|| HdcShadowError::MissingAnchor(name.clone()))?;
        let quality = normalized
            .get(name)
            .ok_or_else(|| HdcShadowError::MissingNormalizedObjective(name.clone()))?
            .quality() as f32;
        metrics.push(ContinuousHV::weighted_bundle(
            &[&anchor.good, &anchor.bad],
            &[quality, 1.0 - quality],
        ));
    }
    let refs: Vec<&ContinuousHV> = metrics.iter().collect();
    Ok(ContinuousHV::weighted_bundle(&refs, weights))
}

fn local_full_cosine(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
    if left.values.len() != right.values.len() || left.values.is_empty() {
        return -1.0;
    }
    let mut dot = 0.0f64;
    let mut norm_left = 0.0f64;
    let mut norm_right = 0.0f64;
    for (&a, &b) in left.values.iter().zip(&right.values) {
        let a = f64::from(a);
        let b = f64::from(b);
        dot += a * b;
        norm_left += a * a;
        norm_right += b * b;
    }
    if norm_left <= f64::EPSILON || norm_right <= f64::EPSILON {
        return -1.0;
    }
    let similarity = dot / (norm_left.sqrt() * norm_right.sqrt());
    if similarity.is_finite() {
        similarity.clamp(-1.0, 1.0) as f32
    } else {
        -1.0
    }
}

fn anchor_seed(metric: &ObjectiveMetric, role: &[u8]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ANCHOR_DOMAIN_V1);
    update_framed(&mut hasher, role);
    update_framed(&mut hasher, metric.objective_name.as_bytes());
    update_framed(&mut hasher, metric.metric_id.as_bytes());
    update_framed(&mut hasher, metric.unit_id.as_bytes());
    update_statistic(&mut hasher, &metric.statistic);
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_be_bytes(bytes)
}

fn update_framed(hasher: &mut blake3::Hasher, value: &[u8]) {
    let len = u64::try_from(value.len()).expect("in-memory metric field length fits u64");
    hasher.update(&len.to_be_bytes());
    hasher.update(value);
}

fn update_statistic(hasher: &mut blake3::Hasher, statistic: &ObjectiveStatistic) {
    match statistic {
        ObjectiveStatistic::CandidateTotal => hasher.update(&[0]),
        ObjectiveStatistic::CandidateMean => hasher.update(&[1]),
        ObjectiveStatistic::CandidateMinimum => hasher.update(&[2]),
        ObjectiveStatistic::CandidateMaximum => hasher.update(&[3]),
        ObjectiveStatistic::CandidateFinal => hasher.update(&[4]),
        ObjectiveStatistic::CandidateCount => hasher.update(&[5]),
        ObjectiveStatistic::CandidateFraction => hasher.update(&[6]),
        ObjectiveStatistic::PercentileBasisPoints(value) => {
            hasher.update(&[7]);
            hasher.update(&value.to_be_bytes());
        }
    };
}

#[derive(Debug, Error)]
pub enum HdcShadowError {
    #[error("normalization profile metric schema differs from ranking context")]
    NormalizationSchemaMismatch,
    #[error("HDC preference metric schema differs from ranking context")]
    PreferenceSchemaMismatch,
    #[error("missing normalization band for {0}")]
    MissingNormalizationBand(String),
    #[error("normalization metric/direction differs from ranking objective {objective_name}")]
    NormalizationDirectionMismatch { objective_name: String },
    #[error("missing HDC preference for {0}")]
    MissingPreference(String),
    #[error("HDC preference metric differs from ranking objective {objective_name}")]
    PreferenceMetricMismatch { objective_name: String },
    #[error("missing qualified vector for feasible candidate {0}")]
    MissingQualifiedVector(String),
    #[error("normalization failed for {candidate_id}: {source}")]
    Normalization {
        candidate_id: String,
        #[source]
        source: NormalizationError,
    },
    #[error("missing HDC anchor for objective {0}")]
    MissingAnchor(String),
    #[error("normalized vector is missing objective {0}")]
    MissingNormalizedObjective(String),
    #[error("closed-set recommendation validation failed: {0}")]
    Recommendation(#[source] RecommendationError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, TimeZone, Utc};
    use symthaea_operations_research::ObjectiveDirection;
    use symthaea_resource_allocation::{AllocationBook, PlannedAllocation};
    use symthaea_resource_capacity::{
        CapacitySchedule, CapacitySemantics, CapacitySubject, CapacityWindow,
    };
    use symthaea_resource_feasible_set::enumerate_feasible_set;
    use symthaea_resource_model::{
        PortDirection, ResourceAmount, ResourceKind, ResourcePort, ResourceUnit,
    };
    use symthaea_resource_objective_evidence::{
        CandidateObjectiveEvidence, ObjectiveEvidenceClass, ObjectiveEvidenceScope,
        ResourceObjectiveEvidenceSet,
    };
    use symthaea_resource_objective_metric::ObjectiveStatistic;
    use symthaea_resource_objective_normalization::{
        ObjectiveNormalizationBand, OutOfRangePolicy,
    };
    use symthaea_resource_objective_qualification::{
        ObjectiveEvidenceIdentityPolicy, ObjectiveMultipleEvidenceRule,
        ObjectiveQualificationPolicy, ObjectiveQualificationRequest,
    };
    use symthaea_resource_pareto::{rank_feasible_resources, ResourceParetoObjective};
    use symthaea_resource_quality::{
        NumericQualityConstraint, QualityMetric, ResourceQualityProfile,
        ResourceQualityRequirement,
    };
    use symthaea_resource_quality_evidence::{
        QualityEvidenceClass, QualityEvidenceSet, QualityEvidenceWindow, QualitySubject,
    };
    use symthaea_resource_quality_qualification::{
        EvidenceIdentityPolicy, MultipleEvidenceRule, QualityQualificationPolicy,
    };
    use symthaea_resource_recommendation::ValidatedResourcePlannerDecision;
    use symthaea_resource_topology::{ResourceLink, ResourceTopology};

    fn t0() -> chrono::DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 9, 7, 12, 0, 0).unwrap()
    }

    fn power(value: f64) -> ResourceAmount {
        ResourceAmount::new(ResourceKind::Electricity, ResourceUnit::Watt, value).unwrap()
    }

    fn energy() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.joule.v1",
            "si.joule.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn resilience() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "resilience",
            "resilience.fraction.v1",
            "ratio.fraction.v1",
            ObjectiveStatistic::CandidateFraction,
        )
        .unwrap()
    }

    fn energy_kwh() -> ObjectiveMetric {
        ObjectiveMetric::new(
            "energy",
            "energy.total.kwh.v1",
            "energy.kilowatt_hour.v1",
            ObjectiveStatistic::CandidateTotal,
        )
        .unwrap()
    }

    fn ranking(candidates: &[(&str, f64, f64)]) -> ResourceParetoRanking {
        let mut topology = ResourceTopology::default();
        topology
            .add_node(
                "source",
                [ResourcePort {
                    id: "out".into(),
                    direction: PortDirection::Output,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_node(
                "sink",
                [ResourcePort {
                    id: "in".into(),
                    direction: PortDirection::Input,
                    capacity: power(100.0),
                }],
            )
            .unwrap();
        topology
            .add_link(ResourceLink {
                id: "line".into(),
                from_node: "source".into(),
                from_port: "out".into(),
                to_node: "sink".into(),
                to_port: "in".into(),
                capacity: power(100.0),
                loss_fraction: 0.0,
            })
            .unwrap();

        let mut capacities = CapacitySchedule::default();
        for (id, subject) in [
            (
                "source-cap",
                CapacitySubject::Port {
                    node_id: "source".into(),
                    port_id: "out".into(),
                },
            ),
            (
                "link-cap",
                CapacitySubject::Link {
                    link_id: "line".into(),
                },
            ),
            (
                "sink-cap",
                CapacitySubject::Port {
                    node_id: "sink".into(),
                    port_id: "in".into(),
                },
            ),
        ] {
            capacities
                .add_window(
                    &topology,
                    CapacityWindow {
                        id: id.into(),
                        subject,
                        valid_from: t0(),
                        valid_until: t0() + Duration::hours(1),
                        capacity: power(100.0),
                        semantics: CapacitySemantics::Concurrent,
                    },
                )
                .unwrap();
        }

        let mut quality_profile = ResourceQualityProfile::new(power(1.0).key);
        quality_profile
            .set_numeric(QualityMetric::TemperatureCelsius, 40.0)
            .unwrap();
        let mut quality_evidence = QualityEvidenceSet::default();
        quality_evidence
            .insert(
                &topology,
                QualityEvidenceWindow {
                    id: "quality".into(),
                    subject: QualitySubject::Link {
                        link_id: "line".into(),
                    },
                    valid_from: t0(),
                    valid_until: t0() + Duration::hours(1),
                    profile: quality_profile,
                    evidence_class: QualityEvidenceClass::Observed,
                    evidence_ref: "sensor:quality".into(),
                },
            )
            .unwrap();
        let mut quality_requirement = ResourceQualityRequirement::new(power(1.0).key);
        quality_requirement
            .require_numeric(NumericQualityConstraint {
                metric: QualityMetric::TemperatureCelsius,
                minimum: Some(20.0),
                maximum: Some(60.0),
            })
            .unwrap();
        let quality_policy = QualityQualificationPolicy {
            allowed_classes: [QualityEvidenceClass::Observed].into_iter().collect(),
            identity_policy: EvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: MultipleEvidenceRule::RejectMultiple,
        };
        let allocations = candidates
            .iter()
            .enumerate()
            .map(|(index, (id, _, _))| PlannedAllocation {
                id: (*id).into(),
                link_id: "line".into(),
                valid_from: t0(),
                valid_until: t0() + Duration::hours(1),
                sent: power(20.0 + index as f64),
            })
            .collect();
        let feasible = enumerate_feasible_set(
            &topology,
            &capacities,
            &AllocationBook::default(),
            &quality_evidence,
            &quality_requirement,
            &quality_policy,
            allocations,
            16,
        )
        .unwrap();

        let mut evidence = ResourceObjectiveEvidenceSet::new(feasible);
        for (candidate_id, energy_value, resilience_value) in candidates {
            for (suffix, metric, value) in [
                ("energy", energy(), *energy_value),
                ("resilience", resilience(), *resilience_value),
            ] {
                evidence
                    .insert(CandidateObjectiveEvidence {
                        id: format!("{candidate_id}-{suffix}"),
                        candidate_id: (*candidate_id).into(),
                        metric,
                        value,
                        evidence_class: ObjectiveEvidenceClass::Observed,
                        evidence_ref: format!("source:{candidate_id}:{suffix}"),
                        scope: ObjectiveEvidenceScope::CandidateAggregate,
                    })
                    .unwrap();
            }
        }

        let objective_policy = ObjectiveQualificationPolicy {
            allowed_classes: [ObjectiveEvidenceClass::Observed].into_iter().collect(),
            identity_policy: ObjectiveEvidenceIdentityPolicy::Any,
            minimum_evidence: 1,
            multiple_evidence_rule: ObjectiveMultipleEvidenceRule::RejectMultiple,
        };
        let objectives = vec![
            ResourceParetoObjective::new(energy(), ObjectiveDirection::Minimize),
            ResourceParetoObjective::new(resilience(), ObjectiveDirection::Maximize),
        ];
        let requests = vec![
            ObjectiveQualificationRequest {
                metric: energy(),
                policy: objective_policy.clone(),
            },
            ObjectiveQualificationRequest {
                metric: resilience(),
                policy: objective_policy,
            },
        ];
        rank_feasible_resources(&evidence, &objectives, &requests).unwrap()
    }

    fn normalization() -> ObjectiveNormalizationProfile {
        ObjectiveNormalizationProfile::new([
            ObjectiveNormalizationBand {
                metric: energy(),
                direction: ObjectiveDirection::Minimize,
                ideal: 0.0,
                worst: 100.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
            ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 1.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
        ])
        .unwrap()
    }

    fn preferences(energy_weight: u16, resilience_weight: u16) -> HdcPreferenceProfile {
        HdcPreferenceProfile::new([
            HdcObjectivePreference {
                metric: energy(),
                weight: energy_weight,
            },
            HdcObjectivePreference {
                metric: resilience(),
                weight: resilience_weight,
            },
        ])
        .unwrap()
    }

    #[test]
    fn repeated_shadow_plans_are_deterministic() {
        let ranking = ranking(&[("a", 10.0, 0.8), ("b", 50.0, 0.5)]);
        let a = plan_hdc_shadow(&ranking, &normalization(), &preferences(1, 1)).unwrap();
        let b = plan_hdc_shadow(&ranking, &normalization(), &preferences(1, 1)).unwrap();
        assert_eq!(a.proposal(), b.proposal());
        assert_eq!(a.traces(), b.traces());
    }

    #[test]
    fn zero_weight_profile_reproduces_canonical_baseline_exactly() {
        let ranking = ranking(&[("z", 10.0, 0.8), ("a", 50.0, 0.5)]);
        let plan = plan_hdc_shadow(&ranking, &normalization(), &preferences(0, 0)).unwrap();
        assert!(plan.zero_weight_baseline_fallback());
        assert!(plan.traces().is_empty());
        let ValidatedResourcePlannerDecision::Recommend(validated) = plan.validated_decision() else {
            panic!("non-empty baseline should recommend");
        };
        assert_eq!(validated.candidate_id(), "a");
    }

    #[test]
    fn exact_ideal_candidate_ranks_ahead_of_exact_worst_candidate() {
        let ranking = ranking(&[("ideal", 0.0, 1.0), ("worst", 100.0, 0.0)]);
        let plan = plan_hdc_shadow(&ranking, &normalization(), &preferences(1, 1)).unwrap();
        assert_eq!(plan.traces()[0].candidate_id(), "ideal");
        assert!((plan.traces()[0].similarity_to_ideal() - 1.0).abs() < 1e-5);
        let ValidatedResourcePlannerDecision::Recommend(validated) = plan.validated_decision() else {
            panic!("HDC should produce one closed-set recommendation");
        };
        assert_eq!(validated.candidate_id(), "ideal");
        assert_eq!(validated.selector_id(), HDC_SHADOW_SELECTOR_ID);
    }

    #[test]
    fn preference_metric_schema_mismatch_fails_before_hdc_encoding() {
        let ranking = ranking(&[("a", 10.0, 0.8)]);
        let bad = HdcPreferenceProfile::new([
            HdcObjectivePreference {
                metric: energy_kwh(),
                weight: 1,
            },
            HdcObjectivePreference {
                metric: resilience(),
                weight: 1,
            },
        ])
        .unwrap();
        assert!(matches!(
            plan_hdc_shadow(&ranking, &normalization(), &bad),
            Err(HdcShadowError::PreferenceSchemaMismatch)
        ));
    }

    #[test]
    fn normalization_direction_mismatch_fails_before_hdc_encoding() {
        let ranking = ranking(&[("a", 10.0, 0.8)]);
        let wrong = ObjectiveNormalizationProfile::new([
            ObjectiveNormalizationBand {
                metric: energy(),
                direction: ObjectiveDirection::Maximize,
                ideal: 100.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
            ObjectiveNormalizationBand {
                metric: resilience(),
                direction: ObjectiveDirection::Maximize,
                ideal: 1.0,
                worst: 0.0,
                out_of_range: OutOfRangePolicy::Reject,
            },
        ])
        .unwrap();
        assert!(matches!(
            plan_hdc_shadow(&ranking, &wrong, &preferences(1, 1)),
            Err(HdcShadowError::NormalizationDirectionMismatch { objective_name }) if objective_name == "energy"
        ));
    }

    #[test]
    fn metric_semantics_change_anchor_identity() {
        assert_ne!(anchor_seed(&energy(), b"good"), anchor_seed(&energy_kwh(), b"good"));
        assert_ne!(anchor_seed(&energy(), b"good"), anchor_seed(&energy(), b"bad"));
    }
}

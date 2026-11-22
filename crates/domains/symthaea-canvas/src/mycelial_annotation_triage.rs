// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Topology-aware review triage for calibrated mycelial reconstructions.
//!
//! This module exposes predictive uncertainty and graph-structural impact as
//! separate quantities. It provides named deterministic orderings, but never
//! collapses the quantities into an unqualified universal score and never
//! mutates or authorizes accepted topology.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mycelial_reconstruction_calibration::{
    MycelialReconstructionCalibrationReport, ReconstructionCalibrationClass,
};
use crate::mycelial_reconstruction_metrics::AnnotatedMycelialSkeleton;

pub const MYCELIAL_ANNOTATION_TRIAGE_SCHEMA_VERSION: u16 = 1;
pub const MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS: usize = 8_192;
const MAX_TRIAGE_ID_BYTES: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MycelialReviewOrderingPolicy {
    /// Rank high structural consequence first, then uncertainty.
    StructuralImpactThenUncertainty,
    /// Rank uncertainty first, then structural consequence.
    UncertaintyThenStructuralImpact,
    /// Rank high anastomosis probability first, then structural consequence.
    AnastomosisProbabilityThenImpact,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MycelialAnnotationTriageConfig {
    pub ordering_policy: MycelialReviewOrderingPolicy,
    /// Confidence at or below this value is marked as an uncertainty review.
    /// This is an annotation-routing threshold, not a fusion-acceptance threshold.
    pub uncertainty_confidence_threshold: f64,
    /// A positive component delta means removing the junction fragments
    /// one previously connected component into multiple components.
    pub articulation_component_delta_threshold: usize,
    /// A junction is anastomosis-sensitive when either fusion or projected
    /// crossing probability reaches this threshold.
    pub anastomosis_sensitive_probability_threshold: f64,
}

impl Default for MycelialAnnotationTriageConfig {
    fn default() -> Self {
        Self {
            ordering_policy: MycelialReviewOrderingPolicy::StructuralImpactThenUncertainty,
            uncertainty_confidence_threshold: 0.8,
            articulation_component_delta_threshold: 1,
            anastomosis_sensitive_probability_threshold: 0.1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialJunctionReviewItem {
    pub junction_id: String,
    pub predicted_class: ReconstructionCalibrationClass,
    /// Full retained calibrated distribution used to derive all other fields.
    pub probabilities: [f64; 7],
    pub top_label_confidence: f64,
    pub second_highest_probability: f64,
    pub top_two_margin: f64,
    pub predictive_entropy_nats: f64,
    pub anastomosis_probability: f64,
    pub projected_crossing_probability: f64,
    pub uncertain_probability: f64,
    pub incident_segment_ids: Vec<String>,
    pub degree: usize,
    pub components_before_removal: usize,
    pub components_after_removal: usize,
    /// Signed change: components_after - components_before.
    pub component_delta: i64,
    pub cycle_rank_before_removal: usize,
    pub cycle_rank_after_removal: usize,
    /// Cycles removed by deleting this junction (saturating difference).
    pub cycle_rank_loss: usize,
    pub is_articulation_critical: bool,
    pub needs_uncertainty_review: bool,
    pub has_anastomosis_sensitive_evidence: bool,
    /// One-based position under the named queue policy.
    pub review_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MycelialAnnotationTriageTopologySummary {
    pub junction_count: usize,
    pub segment_count: usize,
    pub connected_components: usize,
    pub cycle_rank: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelialAnnotationTriageReport {
    pub schema_version: u16,
    pub source_observation_digest: String,
    pub prediction_graph_digest: String,
    pub calibration_report_digest: String,
    pub calibration_policy_id: String,
    pub ordering_policy: MycelialReviewOrderingPolicy,
    pub uncertainty_confidence_threshold: f64,
    pub articulation_component_delta_threshold: usize,
    pub anastomosis_sensitive_probability_threshold: f64,
    pub topology_before: MycelialAnnotationTriageTopologySummary,
    /// Complete sorted junction inventory, including high-confidence items.
    pub review_items: Vec<MycelialJunctionReviewItem>,
    pub uncertain_review_count: usize,
    pub articulation_critical_count: usize,
    pub anastomosis_sensitive_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MycelialAnnotationTriageError {
    InvalidConfig,
    SourceGraphMismatch,
    CalibrationGraphMismatch,
    TooManyJunctions(usize),
    InvalidTriageReport(&'static str),
}

impl fmt::Display for MycelialAnnotationTriageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "invalid annotation triage configuration"),
            Self::SourceGraphMismatch => {
                write!(f, "calibration and prediction graph source identities differ")
            }
            Self::CalibrationGraphMismatch => {
                write!(f, "calibration report does not bind the supplied prediction graph")
            }
            Self::TooManyJunctions(count) => {
                write!(f, "triage junction count {count} exceeds the configured maximum")
            }
            Self::InvalidTriageReport(reason) => {
                write!(f, "invalid annotation triage report: {reason}")
            }
        }
    }
}

impl Error for MycelialAnnotationTriageError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TopologyMetrics {
    connected_components: usize,
    cycle_rank: usize,
}

fn topology_metrics(
    graph: &AnnotatedMycelialSkeleton,
    excluded_junction_id: Option<&str>,
) -> TopologyMetrics {
    let mut adjacency = BTreeMap::<&str, Vec<&str>>::new();
    for junction in &graph.junctions {
        if Some(junction.id.as_str()) != excluded_junction_id {
            adjacency.insert(junction.id.as_str(), Vec::new());
        }
    }
    let mut edge_count = 0usize;
    for segment in &graph.segments {
        if Some(segment.start_junction_id.as_str()) == excluded_junction_id
            || Some(segment.end_junction_id.as_str()) == excluded_junction_id
        {
            continue;
        }
        if let Some(neighbours) = adjacency.get_mut(segment.start_junction_id.as_str()) {
            neighbours.push(segment.end_junction_id.as_str());
            edge_count += 1;
        }
        if let Some(neighbours) = adjacency.get_mut(segment.end_junction_id.as_str()) {
            neighbours.push(segment.start_junction_id.as_str());
        }
    }
    let mut visited = BTreeSet::new();
    let mut components = 0usize;
    for start in adjacency.keys().copied() {
        if !visited.insert(start) {
            continue;
        }
        components += 1;
        let mut queue = VecDeque::from([start]);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbours) = adjacency.get(current) {
                for neighbour in neighbours {
                    if visited.insert(*neighbour) {
                        queue.push_back(*neighbour);
                    }
                }
            }
        }
    }
    let vertex_count = adjacency.len();
    let cycle_rank = edge_count
        .saturating_add(components)
        .saturating_sub(vertex_count);
    TopologyMetrics {
        connected_components: components,
        cycle_rank,
    }
}

fn signed_difference(after: usize, before: usize) -> i64 {
    after as i64 - before as i64
}

fn top_two(
    probabilities: &[f64; 7],
) -> (ReconstructionCalibrationClass, f64, f64) {
    let mut ranked: Vec<_> = ReconstructionCalibrationClass::ALL
        .into_iter()
        .map(|class| (class, probabilities[class.index()]))
        .collect();
    ranked.sort_by(|left, right| {
        right.1
            .total_cmp(&left.1)
            .then_with(|| left.0.index().cmp(&right.0.index()))
    });
    (
        ranked[0].0,
        ranked[0].1,
        ranked.get(1).map(|entry| entry.1).unwrap_or(0.0),
    )
}

fn entropy(probabilities: &[f64; 7]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

/// Construct a topology-aware annotation queue from a prediction graph and
/// its calibrated probability report. No ground-truth labels are consumed by
/// this ranking; the probability report already carries the prediction graph's
/// source identity and confidence observations.
pub fn build_mycelial_annotation_triage(
    prediction: &AnnotatedMycelialSkeleton,
    calibration: &MycelialReconstructionCalibrationReport,
    config: MycelialAnnotationTriageConfig,
) -> Result<MycelialAnnotationTriageReport, MycelialAnnotationTriageError> {
    prediction
        .validate()
        .map_err(|_| MycelialAnnotationTriageError::SourceGraphMismatch)?;
    calibration
        .validate()
        .map_err(|_| MycelialAnnotationTriageError::CalibrationGraphMismatch)?;
    if prediction.source_observation_digest != calibration.source_observation_digest {
        return Err(MycelialAnnotationTriageError::SourceGraphMismatch);
    }
    let prediction_digest = prediction
        .canonical_digest_hex()
        .map_err(|_| MycelialAnnotationTriageError::SourceGraphMismatch)?;
    if prediction_digest != calibration.prediction_graph_digest {
        return Err(MycelialAnnotationTriageError::CalibrationGraphMismatch);
    }
    if prediction.junctions.len() > MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS
        || !config.uncertainty_confidence_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.uncertainty_confidence_threshold)
        || config.articulation_component_delta_threshold == 0
        || config.articulation_component_delta_threshold > MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS
        || !config.anastomosis_sensitive_probability_threshold.is_finite()
        || !(0.0..=1.0).contains(&config.anastomosis_sensitive_probability_threshold)
    {
        return Err(MycelialAnnotationTriageError::InvalidConfig);
    }

    let before = topology_metrics(prediction, None);
    let topology_before = MycelialAnnotationTriageTopologySummary {
        junction_count: prediction.junctions.len(),
        segment_count: prediction.segments.len(),
        connected_components: before.connected_components,
        cycle_rank: before.cycle_rank,
    };
    let mut observations = BTreeMap::new();
    for observation in &calibration.observations {
        observations.insert(observation.predicted_junction_id.as_str(), observation);
    }

    let mut review_items = Vec::with_capacity(prediction.junctions.len());
    for junction in &prediction.junctions {
        let observation = observations
            .get(junction.id.as_str())
            .ok_or(MycelialAnnotationTriageError::CalibrationGraphMismatch)?;
        let (predicted_class, confidence, second) = top_two(&observation.probabilities);
        let entropy = entropy(&observation.probabilities);
        let incident_segment_ids: Vec<_> = prediction
            .segments
            .iter()
            .filter(|segment| {
                segment.start_junction_id == junction.id
                    || segment.end_junction_id == junction.id
            })
            .map(|segment| segment.id.clone())
            .collect();
        let mut incident_segment_ids = incident_segment_ids;
        incident_segment_ids.sort_unstable();
        let after = topology_metrics(prediction, Some(&junction.id));
        let component_delta = signed_difference(
            after.connected_components,
            before.connected_components,
        );
        let cycle_rank_loss = before
            .cycle_rank
            .saturating_sub(after.cycle_rank);
        let anastomosis_probability =
            observation.probabilities[ReconstructionCalibrationClass::Anastomosis.index()];
        let projected_crossing_probability = observation.probabilities
            [ReconstructionCalibrationClass::ProjectedCrossing.index()];
        let uncertain_probability =
            observation.probabilities[ReconstructionCalibrationClass::Uncertain.index()];
        review_items.push(MycelialJunctionReviewItem {
            junction_id: junction.id.clone(),
            predicted_class,
            probabilities: observation.probabilities,
            top_label_confidence: confidence,
            second_highest_probability: second,
            top_two_margin: confidence - second,
            predictive_entropy_nats: entropy,
            anastomosis_probability,
            projected_crossing_probability,
            uncertain_probability,
            degree: incident_segment_ids.len(),
            incident_segment_ids,
            components_before_removal: before.connected_components,
            components_after_removal: after.connected_components,
            component_delta,
            cycle_rank_before_removal: before.cycle_rank,
            cycle_rank_after_removal: after.cycle_rank,
            cycle_rank_loss,
            is_articulation_critical: component_delta
                >= config.articulation_component_delta_threshold as i64,
            needs_uncertainty_review: confidence <= config.uncertainty_confidence_threshold,
            has_anastomosis_sensitive_evidence: anastomosis_probability
                >= config.anastomosis_sensitive_probability_threshold
                || projected_crossing_probability
                    >= config.anastomosis_sensitive_probability_threshold,
            review_rank: 0,
        });
    }

    review_items.sort_by(|left, right| compare_items(left, right, config.ordering_policy));
    for (index, item) in review_items.iter_mut().enumerate() {
        item.review_rank = index + 1;
    }
    let report = MycelialAnnotationTriageReport {
        schema_version: MYCELIAL_ANNOTATION_TRIAGE_SCHEMA_VERSION,
        source_observation_digest: prediction.source_observation_digest.clone(),
        prediction_graph_digest: prediction_digest,
        calibration_report_digest: calibration
            .canonical_digest_hex()
            .map_err(|_| MycelialAnnotationTriageError::CalibrationGraphMismatch)?,
        calibration_policy_id: calibration.calibration_policy_id.clone(),
        ordering_policy: config.ordering_policy,
        uncertainty_confidence_threshold: config.uncertainty_confidence_threshold,
        articulation_component_delta_threshold: config.articulation_component_delta_threshold,
        anastomosis_sensitive_probability_threshold: config
            .anastomosis_sensitive_probability_threshold,
        topology_before,
        uncertain_review_count: review_items
            .iter()
            .filter(|item| item.needs_uncertainty_review)
            .count(),
        articulation_critical_count: review_items
            .iter()
            .filter(|item| item.is_articulation_critical)
            .count(),
        anastomosis_sensitive_count: review_items
            .iter()
            .filter(|item| item.has_anastomosis_sensitive_evidence)
            .count(),
        review_items,
    };
    report.validate()?;
    Ok(report)
}

fn compare_items(
    left: &MycelialJunctionReviewItem,
    right: &MycelialJunctionReviewItem,
    policy: MycelialReviewOrderingPolicy,
) -> std::cmp::Ordering {
    let impact = || {
        right
            .is_articulation_critical
            .cmp(&left.is_articulation_critical)
            .then_with(|| right.cycle_rank_loss.cmp(&left.cycle_rank_loss))
            .then_with(|| right.component_delta.cmp(&left.component_delta))
            .then_with(|| right.degree.cmp(&left.degree))
    };
    let uncertainty = || {
        right
            .predictive_entropy_nats
            .total_cmp(&left.predictive_entropy_nats)
            .then_with(|| left.top_two_margin.total_cmp(&right.top_two_margin))
    };
    match policy {
        MycelialReviewOrderingPolicy::StructuralImpactThenUncertainty => impact()
            .then_with(uncertainty)
            .then_with(|| left.junction_id.cmp(&right.junction_id)),
        MycelialReviewOrderingPolicy::UncertaintyThenStructuralImpact => uncertainty()
            .then_with(impact)
            .then_with(|| left.junction_id.cmp(&right.junction_id)),
        MycelialReviewOrderingPolicy::AnastomosisProbabilityThenImpact => right
            .anastomosis_probability
            .total_cmp(&left.anastomosis_probability)
            .then_with(impact)
            .then_with(uncertainty)
            .then_with(|| left.junction_id.cmp(&right.junction_id)),
    }
}

impl MycelialAnnotationTriageReport {
    pub fn validate(&self) -> Result<(), MycelialAnnotationTriageError> {
        let mut ids = BTreeSet::new();
        if self.schema_version != MYCELIAL_ANNOTATION_TRIAGE_SCHEMA_VERSION
            || !valid_digest(&self.source_observation_digest)
            || !valid_digest(&self.prediction_graph_digest)
            || !valid_digest(&self.calibration_report_digest)
            || !valid_id(&self.calibration_policy_id)
            || !self.uncertainty_confidence_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.uncertainty_confidence_threshold)
            || self.articulation_component_delta_threshold == 0
            || self.articulation_component_delta_threshold > MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS
            || !self.anastomosis_sensitive_probability_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.anastomosis_sensitive_probability_threshold)
            || self.review_items.len() != self.topology_before.junction_count
            || self.review_items.len() > MAX_MYCELIAL_ANNOTATION_TRIAGE_ITEMS
            || self.review_items.iter().enumerate().any(|(index, item)| {
                !valid_id(&item.junction_id)
                    || !ids.insert(item.junction_id.as_str())
                    || item.review_rank != index + 1
                    || item
                        .probabilities
                        .iter()
                        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
                    || (item.probabilities.iter().sum::<f64>() - 1.0).abs() > 1.0e-6
                    || top_two(&item.probabilities)
                        != (
                            item.predicted_class,
                            item.top_label_confidence,
                            item.second_highest_probability,
                        )
                    || (entropy(&item.probabilities) - item.predictive_entropy_nats).abs()
                        > 1.0e-12
                    || !item.top_label_confidence.is_finite()
                    || !(0.0..=1.0).contains(&item.top_label_confidence)
                    || !item.second_highest_probability.is_finite()
                    || !(0.0..=1.0).contains(&item.second_highest_probability)
                    || !item.top_two_margin.is_finite()
                    || !(0.0..=1.0).contains(&item.top_two_margin)
                    || !item.predictive_entropy_nats.is_finite()
                    || item.predictive_entropy_nats < 0.0
                    || !item.anastomosis_probability.is_finite()
                    || !(0.0..=1.0).contains(&item.anastomosis_probability)
                    || !item.projected_crossing_probability.is_finite()
                    || !(0.0..=1.0).contains(&item.projected_crossing_probability)
                    || !item.uncertain_probability.is_finite()
                    || !(0.0..=1.0).contains(&item.uncertain_probability)
                    || item.incident_segment_ids.windows(2).any(|pair| pair[0] >= pair[1])
                    || item.degree != item.incident_segment_ids.len()
                    || item.components_before_removal != self.topology_before.connected_components
                    || item.cycle_rank_before_removal != self.topology_before.cycle_rank
                    || item.component_delta
                        != signed_difference(
                            item.components_after_removal,
                            item.components_before_removal,
                        )
                    || item.cycle_rank_loss
                        != item
                            .cycle_rank_before_removal
                            .saturating_sub(item.cycle_rank_after_removal)
                    || item.is_articulation_critical
                        != (item.component_delta
                            >= self.articulation_component_delta_threshold as i64)
                    || item.needs_uncertainty_review
                        != (item.top_label_confidence <= self.uncertainty_confidence_threshold)
                    || item.has_anastomosis_sensitive_evidence
                        != (item.anastomosis_probability
                            >= self.anastomosis_sensitive_probability_threshold
                            || item.projected_crossing_probability
                                >= self.anastomosis_sensitive_probability_threshold)
                    || item.components_after_removal > self.topology_before.junction_count
                    || item.cycle_rank_after_removal > self.topology_before.cycle_rank
            })
            || self.review_items.windows(2).any(|pair| {
                compare_items(&pair[0], &pair[1], self.ordering_policy)
                    == std::cmp::Ordering::Greater
            })
            || self.uncertain_review_count
                != self
                    .review_items
                    .iter()
                    .filter(|item| item.needs_uncertainty_review)
                    .count()
            || self.articulation_critical_count
                != self
                    .review_items
                    .iter()
                    .filter(|item| item.is_articulation_critical)
                    .count()
            || self.anastomosis_sensitive_count
                != self
                    .review_items
                    .iter()
                    .filter(|item| item.has_anastomosis_sensitive_evidence)
                    .count()
        {
            return Err(MycelialAnnotationTriageError::InvalidTriageReport(
                "metadata, queue ranking or structural summary is invalid",
            ));
        }
        Ok(())
    }

    pub fn validate_against(
        &self,
        prediction: &AnnotatedMycelialSkeleton,
        calibration: &MycelialReconstructionCalibrationReport,
    ) -> Result<(), MycelialAnnotationTriageError> {
        let expected = build_mycelial_annotation_triage(
            prediction,
            calibration,
            MycelialAnnotationTriageConfig {
                ordering_policy: self.ordering_policy,
                uncertainty_confidence_threshold: self.uncertainty_confidence_threshold,
                articulation_component_delta_threshold: self.articulation_component_delta_threshold,
                anastomosis_sensitive_probability_threshold: self
                    .anastomosis_sensitive_probability_threshold,
            },
        )?;
        if expected != *self {
            return Err(MycelialAnnotationTriageError::InvalidTriageReport(
                "triage metrics do not match recomputation from source graph and calibration evidence",
            ));
        }
        Ok(())
    }

    pub fn canonical_digest_hex(
        &self,
    ) -> Result<String, MycelialAnnotationTriageError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea-mycelial-annotation-triage-v1");
        hasher.update(&self.schema_version.to_le_bytes());
        hasher.update(self.source_observation_digest.as_bytes());
        hasher.update(self.prediction_graph_digest.as_bytes());
        hasher.update(self.calibration_report_digest.as_bytes());
        hash_str(&mut hasher, &self.calibration_policy_id);
        hasher.update(&[match self.ordering_policy {
            MycelialReviewOrderingPolicy::StructuralImpactThenUncertainty => 0,
            MycelialReviewOrderingPolicy::UncertaintyThenStructuralImpact => 1,
            MycelialReviewOrderingPolicy::AnastomosisProbabilityThenImpact => 2,
        }]);
        hasher.update(&self.uncertainty_confidence_threshold.to_bits().to_le_bytes());
        hasher.update(&(self.articulation_component_delta_threshold as u64).to_le_bytes());
        hasher.update(
            &self
                .anastomosis_sensitive_probability_threshold
                .to_bits()
                .to_le_bytes(),
        );
        for value in [
            self.topology_before.junction_count,
            self.topology_before.segment_count,
            self.topology_before.connected_components,
            self.topology_before.cycle_rank,
            self.uncertain_review_count,
            self.articulation_critical_count,
            self.anastomosis_sensitive_count,
        ] {
            hasher.update(&(value as u64).to_le_bytes());
        }
        for item in &self.review_items {
            hash_str(&mut hasher, &item.junction_id);
            hasher.update(&[item.predicted_class.index() as u8]);
            for probability in item.probabilities {
                hasher.update(&probability.to_bits().to_le_bytes());
            }
            for value in [
                item.top_label_confidence,
                item.second_highest_probability,
                item.top_two_margin,
                item.predictive_entropy_nats,
                item.anastomosis_probability,
                item.projected_crossing_probability,
                item.uncertain_probability,
            ] {
                hasher.update(&value.to_bits().to_le_bytes());
            }
            for value in [
                item.degree,
                item.components_before_removal,
                item.components_after_removal,
                item.cycle_rank_before_removal,
                item.cycle_rank_after_removal,
                item.cycle_rank_loss,
            ] {
                hasher.update(&(value as u64).to_le_bytes());
            }
            hasher.update(&item.component_delta.to_le_bytes());
            hasher.update(&[u8::from(item.is_articulation_critical)]);
            hasher.update(&[u8::from(item.needs_uncertainty_review)]);
            hasher.update(&[u8::from(item.has_anastomosis_sensitive_evidence)]);
            hasher.update(&(item.review_rank as u64).to_le_bytes());
            hasher.update(&(item.incident_segment_ids.len() as u64).to_le_bytes());
            for segment_id in &item.incident_segment_ids {
                hash_str(&mut hasher, segment_id);
            }
        }
        Ok(hex_digest(hasher.finalize().as_bytes()))
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty() && value.len() <= MAX_TRIAGE_ID_BYTES && !value.chars().any(char::is_control)
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8; 32]) -> String {
    let mut output = String::with_capacity(64);
    for byte in bytes {
        use fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mycelial_reconstruction_calibration::{
        JunctionClassProbabilityPrediction, ReconstructionCalibrationConfig,
        evaluate_mycelial_reconstruction_calibration,
    };
    use crate::mycelial_reconstruction_metrics::{
        AnnotatedHyphaSegment, AnnotatedJunction, AnnotatedJunctionKind,
        JunctionIdentityMatch, MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
        ReconstructionDimensionality, ReconstructionEvaluationConfig,
        ReconstructionLengthUnit, ReconstructionPoint3, evaluate_mycelial_reconstruction,
    };

    fn skeleton(
        nodes: Vec<AnnotatedJunction>,
        edges: Vec<AnnotatedHyphaSegment>,
    ) -> AnnotatedMycelialSkeleton {
        AnnotatedMycelialSkeleton {
            schema_version: MYCELIAL_RECONSTRUCTION_SCHEMA_VERSION,
            frame_id: "triage-frame-001".to_string(),
            coordinate_frame_id: "triage-stage-v1".to_string(),
            calibration_digest: "c".repeat(64),
            source_observation_digest: "d".repeat(64),
            dimensionality: ReconstructionDimensionality::TwoD,
            length_unit: ReconstructionLengthUnit::Micrometer,
            producer_id: "prediction".to_string(),
            producer_version: "triage-test-v1".to_string(),
            junctions: nodes,
            segments: edges,
        }
    }

    fn node(id: &str, kind: AnnotatedJunctionKind, x: f64, y: f64) -> AnnotatedJunction {
        AnnotatedJunction {
            id: id.to_string(),
            kind,
            position: ReconstructionPoint3 { x, y, z: 0.0 },
        }
    }

    fn edge(id: &str, source: &str, target: &str) -> AnnotatedHyphaSegment {
        AnnotatedHyphaSegment {
            id: id.to_string(),
            start_junction_id: source.to_string(),
            end_junction_id: target.to_string(),
            length: 1.0,
        }
    }

    fn probabilities(
        id: &str,
        class: ReconstructionCalibrationClass,
        confidence: f64,
    ) -> JunctionClassProbabilityPrediction {
        let mut probabilities = [0.0; 7];
        probabilities[class.index()] = confidence;
        probabilities[ReconstructionCalibrationClass::Uncertain.index()] = 1.0 - confidence;
        JunctionClassProbabilityPrediction {
            predicted_junction_id: id.to_string(),
            probabilities,
        }
    }

    fn calibrated_report(
        graph: &AnnotatedMycelialSkeleton,
        low_confidence_id: Option<&str>,
    ) -> MycelialReconstructionCalibrationReport {
        let truth = graph.clone();
        let matches: Vec<_> = graph
            .junctions
            .iter()
            .map(|junction| JunctionIdentityMatch {
                predicted_junction_id: junction.id.clone(),
                ground_truth_junction_id: junction.id.clone(),
            })
            .collect();
        let score = evaluate_mycelial_reconstruction(
            &truth,
            graph,
            "identity-fixture-v1",
            &matches,
            ReconstructionEvaluationConfig {
                maximum_junction_match_distance: 0.01,
                ..ReconstructionEvaluationConfig::default()
            },
        )
        .unwrap();
        let probabilities: Vec<_> = graph
            .junctions
            .iter()
            .map(|junction| {
                if Some(junction.id.as_str()) == low_confidence_id {
                    let mut vector = [0.0; 7];
                    vector[ReconstructionCalibrationClass::Anastomosis.index()] = 0.55;
                    vector[ReconstructionCalibrationClass::ProjectedCrossing.index()] = 0.35;
                    vector[ReconstructionCalibrationClass::Uncertain.index()] = 0.10;
                    JunctionClassProbabilityPrediction {
                        predicted_junction_id: junction.id.clone(),
                        probabilities: vector,
                    }
                } else {
                    probabilities(
                        &junction.id,
                        ReconstructionCalibrationClass::from_junction_kind(junction.kind),
                        0.95,
                    )
                }
            })
            .collect();
        evaluate_mycelial_reconstruction_calibration(
            &truth,
            graph,
            &score,
            "triage-calibration-v1",
            &probabilities,
            ReconstructionCalibrationConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn articulation_and_cycle_impact_are_reported_separately_from_uncertainty() {
        let graph = skeleton(
            vec![
                node("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Branch, 1.0, 0.0),
                node("C", AnnotatedJunctionKind::Tip, 2.0, 0.0),
                node("D", AnnotatedJunctionKind::Anastomosis, 1.0, 1.0),
                node("E", AnnotatedJunctionKind::Tip, 2.0, 1.0),
            ],
            vec![
                edge("ab", "A", "B"),
                edge("bc", "B", "C"),
                edge("bd", "B", "D"),
                edge("de", "D", "E"),
                edge("ec", "E", "C"),
            ],
        );
        let calibration = calibrated_report(&graph, Some("D"));
        let graph_digest = graph.canonical_digest_hex().unwrap();
        let report = build_mycelial_annotation_triage(
            &graph,
            &calibration,
            MycelialAnnotationTriageConfig {
                ordering_policy: MycelialReviewOrderingPolicy::StructuralImpactThenUncertainty,
                uncertainty_confidence_threshold: 0.8,
                articulation_component_delta_threshold: 1,
                ..MycelialAnnotationTriageConfig::default()
            },
        )
        .unwrap();
        let junction_b = report.review_items.iter().find(|item| item.junction_id == "B").unwrap();
        assert!(junction_b.is_articulation_critical);
        assert!(junction_b.component_delta > 0);
        let junction_d = report.review_items.iter().find(|item| item.junction_id == "D").unwrap();
        assert!(junction_d.needs_uncertainty_review);
        assert!(junction_d.cycle_rank_loss > 0);
        report.validate_against(&graph, &calibration).unwrap();
        assert_eq!(graph.canonical_digest_hex().unwrap(), graph_digest);
    }

    #[test]
    fn ordering_policies_change_order_not_measured_impact_values() {
        let graph = skeleton(
            vec![
                node("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Branch, 1.0, 0.0),
                node("C", AnnotatedJunctionKind::Tip, 2.0, 0.0),
                node("D", AnnotatedJunctionKind::Tip, 1.0, 1.0),
            ],
            vec![edge("ab", "A", "B"), edge("bc", "B", "C"), edge("bd", "B", "D")],
        );
        let calibration = calibrated_report(&graph, None);
        let first = build_mycelial_annotation_triage(
            &graph,
            &calibration,
            MycelialAnnotationTriageConfig::default(),
        )
        .unwrap();
        let second = build_mycelial_annotation_triage(
            &graph,
            &calibration,
            MycelialAnnotationTriageConfig {
                ordering_policy: MycelialReviewOrderingPolicy::UncertaintyThenStructuralImpact,
                ..MycelialAnnotationTriageConfig::default()
            },
        )
        .unwrap();
        let by_id = |report: &MycelialAnnotationTriageReport| {
            report
                .review_items
                .iter()
                .map(|item| (item.junction_id.clone(), (item.degree, item.component_delta, item.cycle_rank_loss)))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(by_id(&first), by_id(&second));
    }

    #[test]
    fn calibration_from_a_different_prediction_graph_is_rejected() {
        let graph = skeleton(
            vec![
                node("A", AnnotatedJunctionKind::Origin, 0.0, 0.0),
                node("B", AnnotatedJunctionKind::Tip, 1.0, 0.0),
            ],
            vec![edge("ab", "A", "B")],
        );
        let calibration = calibrated_report(&graph, None);
        let mut changed = graph.clone();
        changed.segments[0].length = 2.0;
        assert!(matches!(
            build_mycelial_annotation_triage(
                &changed,
                &calibration,
                MycelialAnnotationTriageConfig::default()
            ),
            Err(MycelialAnnotationTriageError::CalibrationGraphMismatch)
        ));
    }
}

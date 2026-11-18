//! Decision-sufficient plasma diagnostic portfolio analysis.
//!
//! This is a finite, symbolic design-time oracle. It asks whether the active
//! diagnostics distinguish every pair of hidden worlds that require different
//! decisions. It deliberately does not require full hidden-state identification.
//!
//! UNKNOWN is never informative. A sensor is not independent merely because it
//! has a distinct ID: every sensor declares one or more common-cause fault
//! domains. Exact subset and cut-set enumeration is limited to small candidate
//! sets and is not a reliability estimator or physical observability proof.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const PLASMA_DIAGNOSTIC_PORTFOLIO_SCHEMA: &str =
    "luminous.plasma-diagnostic-portfolio.v0.1";
pub const PLASMA_DIAGNOSTIC_PORTFOLIO_BENCHMARK_ID: &str =
    "PLASMA-DIAG-PORTFOLIO-001";
pub const MAX_EXACT_SENSOR_COUNT: usize = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticPortfolioError {
    pub message: String,
}

impl DiagnosticPortfolioError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DiagnosticPortfolioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DiagnosticPortfolioError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionWorld {
    pub world_id: String,
    pub decision_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorObservation {
    Unknown,
    Known(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorWorldObservation {
    pub world_id: String,
    pub observation: SensorObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticSensor {
    pub sensor_id: String,
    /// One sensor may share multiple failure domains (e.g. clock, power, ADC).
    pub common_mode_groups: Vec<String>,
    /// Optional declared lifecycle resource cost. Unknown is not zero.
    pub resource_cost: Option<u64>,
    /// Present only when resource_cost is known.
    pub resource_unit: Option<String>,
    /// Exactly one observation, including explicit UNKNOWN, for every world.
    pub observations: Vec<SensorWorldObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticPortfolioModel {
    pub schema_version: String,
    pub model_id: String,
    pub decision_catalogue_id: String,
    /// Shared unit for every known resource cost in this model.
    pub resource_unit: Option<String>,
    pub worlds: Vec<DecisionWorld>,
    pub sensors: Vec<DiagnosticSensor>,
}

impl DiagnosticPortfolioModel {
    /// Validate IDs, decision classes, exact world coverage, and cost metadata.
    pub fn validate(&self) -> Result<(), DiagnosticPortfolioError> {
        if self.schema_version != PLASMA_DIAGNOSTIC_PORTFOLIO_SCHEMA {
            return Err(DiagnosticPortfolioError::new(
                "unsupported diagnostic portfolio schema version",
            ));
        }
        non_empty(&self.model_id, "model_id")?;
        non_empty(&self.decision_catalogue_id, "decision_catalogue_id")?;

        if self.worlds.len() < 2 {
            return Err(DiagnosticPortfolioError::new(
                "at least two hidden worlds are required",
            ));
        }
        let mut world_ids = BTreeSet::new();
        let mut decision_ids = BTreeSet::new();
        for world in &self.worlds {
            non_empty(&world.world_id, "world_id")?;
            non_empty(&world.decision_id, "decision_id")?;
            if !world_ids.insert(world.world_id.as_str()) {
                return Err(DiagnosticPortfolioError::new(format!(
                    "duplicate world_id: {}",
                    world.world_id
                )));
            }
            decision_ids.insert(world.decision_id.as_str());
        }
        if decision_ids.len() < 2 {
            return Err(DiagnosticPortfolioError::new(
                "at least two decision classes are required",
            ));
        }

        let mut sensor_ids = BTreeSet::new();
        let mut known_cost_unit: Option<&str> = None;
        for sensor in &self.sensors {
            non_empty(&sensor.sensor_id, "sensor_id")?;
            if !sensor_ids.insert(sensor.sensor_id.as_str()) {
                return Err(DiagnosticPortfolioError::new(format!(
                    "duplicate sensor_id: {}",
                    sensor.sensor_id
                )));
            }
            if sensor.common_mode_groups.is_empty() {
                return Err(DiagnosticPortfolioError::new(format!(
                    "sensor {} must declare at least one common-mode group",
                    sensor.sensor_id
                )));
            }
            let mut sensor_groups = BTreeSet::new();
            for group in &sensor.common_mode_groups {
                non_empty(group, "common_mode_group")?;
                if !sensor_groups.insert(group.as_str()) {
                    return Err(DiagnosticPortfolioError::new(format!(
                        "sensor {} repeats common-mode group {}",
                        sensor.sensor_id, group
                    )));
                }
            }

            match (sensor.resource_cost, sensor.resource_unit.as_deref()) {
                (Some(_), Some(unit)) => {
                    non_empty(unit, "resource_unit")?;
                    if let Some(previous) = known_cost_unit {
                        if previous != unit {
                            return Err(DiagnosticPortfolioError::new(
                                "all known resource costs must use one common unit",
                            ));
                        }
                    } else {
                        known_cost_unit = Some(unit);
                    }
                }
                (Some(_), None) => {
                    return Err(DiagnosticPortfolioError::new(format!(
                        "sensor {} has a resource cost without a unit",
                        sensor.sensor_id
                    )));
                }
                (None, Some(_)) => {
                    return Err(DiagnosticPortfolioError::new(format!(
                        "sensor {} declares a resource unit while its cost is unknown",
                        sensor.sensor_id
                    )));
                }
                (None, None) => {}
            }

            let mut seen_worlds = BTreeSet::new();
            for observation in &sensor.observations {
                non_empty(&observation.world_id, "observation.world_id")?;
                if !world_ids.contains(observation.world_id.as_str()) {
                    return Err(DiagnosticPortfolioError::new(format!(
                        "sensor {} references unknown world {}",
                        sensor.sensor_id, observation.world_id
                    )));
                }
                if !seen_worlds.insert(observation.world_id.as_str()) {
                    return Err(DiagnosticPortfolioError::new(format!(
                        "sensor {} repeats world {}",
                        sensor.sensor_id, observation.world_id
                    )));
                }
                if let SensorObservation::Known(value) = &observation.observation {
                    non_empty(value, "known_observation")?;
                }
            }
            if seen_worlds != world_ids {
                return Err(DiagnosticPortfolioError::new(format!(
                    "sensor {} must declare exactly one observation for every world",
                    sensor.sensor_id
                )));
            }
        }

        match (known_cost_unit, self.resource_unit.as_deref()) {
            (Some(known), Some(model_unit)) if known == model_unit => {}
            (Some(_), _) => {
                return Err(DiagnosticPortfolioError::new(
                    "model resource_unit must match all declared sensor cost units",
                ));
            }
            (None, Some(model_unit)) => {
                non_empty(model_unit, "resource_unit")?;
            }
            (None, None) => {}
        }
        Ok(())
    }

    fn sorted_sensor_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .sensors
            .iter()
            .map(|sensor| sensor.sensor_id.clone())
            .collect();
        ids.sort();
        ids
    }

    fn sensor_index(&self) -> BTreeMap<&str, &DiagnosticSensor> {
        self.sensors
            .iter()
            .map(|sensor| (sensor.sensor_id.as_str(), sensor))
            .collect()
    }
}

fn non_empty(value: &str, field: &str) -> Result<(), DiagnosticPortfolioError> {
    if value.trim().is_empty() {
        return Err(DiagnosticPortfolioError::new(format!(
            "{field} must be non-empty"
        )));
    }
    Ok(())
}

fn ensure_exact_enumeration_size(n: usize) -> Result<(), DiagnosticPortfolioError> {
    if n > MAX_EXACT_SENSOR_COUNT {
        return Err(DiagnosticPortfolioError::new(format!(
            "exact subset enumeration is limited to {MAX_EXACT_SENSOR_COUNT} sensors; received {n}",
        )));
    }
    Ok(())
}

fn subset_for_mask(items: &[String], mask: u64) -> Vec<String> {
    items
        .iter()
        .enumerate()
        .filter(|(index, _)| (mask & (1_u64 << index)) != 0)
        .map(|(_, item)| item.clone())
        .collect()
}

fn is_subset(left: &[String], right: &[String]) -> bool {
    let right: BTreeSet<&str> = right.iter().map(String::as_str).collect();
    left.iter().all(|item| right.contains(item.as_str()))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnresolvedDecisionPair {
    pub left_world_id: String,
    pub right_world_id: String,
    pub left_decision_id: String,
    pub right_decision_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosabilityResult {
    pub diagnosable: bool,
    pub active_sensor_ids: Vec<String>,
    pub unresolved_cross_decision_pairs: Vec<UnresolvedDecisionPair>,
}

/// Cached structural index used by exhaustive subset oracles. World
/// observations are arranged once in canonical model order, so enumerating
/// candidate subsets does not rebuild sensor/world hash maps on every pass.
struct DiagnosabilityContext<'a> {
    model: &'a DiagnosticPortfolioModel,
    sensor_outputs: BTreeMap<String, Vec<SensorObservation>>,
    cross_decision_pairs: Vec<(usize, usize)>,
}

impl<'a> DiagnosabilityContext<'a> {
    fn new(model: &'a DiagnosticPortfolioModel) -> Result<Self, DiagnosticPortfolioError> {
        model.validate()?;
        let mut sensor_outputs = BTreeMap::new();
        for sensor in &model.sensors {
            let outputs: BTreeMap<&str, SensorObservation> = sensor
                .observations
                .iter()
                .map(|item| (item.world_id.as_str(), item.observation.clone()))
                .collect();
            let ordered = model
                .worlds
                .iter()
                .map(|world| {
                    outputs.get(world.world_id.as_str()).cloned().ok_or_else(|| {
                        DiagnosticPortfolioError::new(format!(
                            "sensor {} lost validated observation for world {}",
                            sensor.sensor_id, world.world_id
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            sensor_outputs.insert(sensor.sensor_id.clone(), ordered);
        }

        let mut cross_decision_pairs = Vec::new();
        for (left_index, left) in model.worlds.iter().enumerate() {
            for (right_index, right) in model.worlds.iter().enumerate().skip(left_index + 1) {
                if left.decision_id != right.decision_id {
                    cross_decision_pairs.push((left_index, right_index));
                }
            }
        }

        Ok(Self {
            model,
            sensor_outputs,
            cross_decision_pairs,
        })
    }

    fn diagnose(
        &self,
        active_sensor_ids: &[String],
    ) -> Result<DiagnosabilityResult, DiagnosticPortfolioError> {
        let known_ids: BTreeSet<&str> = self
            .model
            .sensors
            .iter()
            .map(|sensor| sensor.sensor_id.as_str())
            .collect();
        let mut active = BTreeSet::new();
        for sensor_id in active_sensor_ids {
            non_empty(sensor_id, "active_sensor_id")?;
            if !active.insert(sensor_id.as_str()) {
                return Err(DiagnosticPortfolioError::new(format!(
                    "duplicate active sensor_id: {sensor_id}"
                )));
            }
            if !known_ids.contains(sensor_id.as_str()) {
                return Err(DiagnosticPortfolioError::new(format!(
                    "unknown active sensor_id: {sensor_id}"
                )));
            }
        }

        let active_sensor_ids: Vec<String> = active.iter().map(|id| (*id).to_owned()).collect();
        let mut unresolved = Vec::new();
        for (left_index, right_index) in &self.cross_decision_pairs {
            let left = &self.model.worlds[*left_index];
            let right = &self.model.worlds[*right_index];
            let mut separated = false;
            for sensor_id in &active_sensor_ids {
                let outputs = self.sensor_outputs.get(sensor_id).ok_or_else(|| {
                    DiagnosticPortfolioError::new("validated sensor output disappeared")
                })?;
                let left_value = outputs.get(*left_index);
                let right_value = outputs.get(*right_index);
                let (Some(SensorObservation::Known(left_value)), Some(SensorObservation::Known(right_value))) =
                    (left_value, right_value)
                else {
                    continue;
                };
                if left_value != right_value {
                    separated = true;
                    break;
                }
            }
            if !separated {
                unresolved.push(UnresolvedDecisionPair {
                    left_world_id: left.world_id.clone(),
                    right_world_id: right.world_id.clone(),
                    left_decision_id: left.decision_id.clone(),
                    right_decision_id: right.decision_id.clone(),
                });
            }
        }

        Ok(DiagnosabilityResult {
            diagnosable: unresolved.is_empty(),
            active_sensor_ids,
            unresolved_cross_decision_pairs: unresolved,
        })
    }
}

/// Checks whether every cross-decision pair of worlds is separated by at
/// least one active sensor. UNKNOWN on either side is non-informative.
pub fn diagnose(
    model: &DiagnosticPortfolioModel,
    active_sensor_ids: &[String],
) -> Result<DiagnosabilityResult, DiagnosticPortfolioError> {
    DiagnosabilityContext::new(model)?.diagnose(active_sensor_ids)
}

/// Enumerates inclusion-minimal sufficient sensor sets exactly. This is an
/// oracle for small candidate sets, not a heuristic optimizer.
pub fn minimal_sufficient_sensor_sets(
    model: &DiagnosticPortfolioModel,
) -> Result<Vec<Vec<String>>, DiagnosticPortfolioError> {
    let context = DiagnosabilityContext::new(model)?;
    let sensor_ids = model.sorted_sensor_ids();
    ensure_exact_enumeration_size(sensor_ids.len())?;
    let subset_count = 1_u64 << sensor_ids.len();
    let mut sufficient: Vec<Vec<String>> = Vec::new();

    for mask in 0..subset_count {
        let subset = subset_for_mask(&sensor_ids, mask);
        if sufficient.iter().any(|prior| is_subset(prior, &subset)) {
            continue;
        }
        if context.diagnose(&subset)?.diagnosable {
            sufficient.push(subset);
        }
    }
    Ok(sufficient)
}

/// Enumerates inclusion-minimal sensor-loss cut sets relative to a declared
/// diagnosable baseline.
pub fn minimal_sensor_cut_sets(
    model: &DiagnosticPortfolioModel,
    baseline_active_sensor_ids: &[String],
) -> Result<Vec<Vec<String>>, DiagnosticPortfolioError> {
    let context = DiagnosabilityContext::new(model)?;
    let baseline = context.diagnose(baseline_active_sensor_ids)?;
    if !baseline.diagnosable {
        return Err(DiagnosticPortfolioError::new(
            "minimum sensor-cut analysis requires a diagnosable baseline",
        ));
    }
    let baseline_ids = baseline.active_sensor_ids;
    ensure_exact_enumeration_size(baseline_ids.len())?;
    let subset_count = 1_u64 << baseline_ids.len();
    let mut cuts: Vec<Vec<String>> = Vec::new();

    for mask in 1..subset_count {
        let removed = subset_for_mask(&baseline_ids, mask);
        if cuts.iter().any(|prior| is_subset(prior, &removed)) {
            continue;
        }
        let removed_set: BTreeSet<&str> = removed.iter().map(String::as_str).collect();
        let remaining: Vec<String> = baseline_ids
            .iter()
            .filter(|sensor_id| !removed_set.contains(sensor_id.as_str()))
            .cloned()
            .collect();
        if !context.diagnose(&remaining)?.diagnosable {
            cuts.push(removed);
        }
    }
    Ok(cuts)
}

/// A common-mode group loss removes every sensor declaring that group. Sensors
/// may belong to multiple groups (e.g. shared clock and shared acquisition power).
pub fn minimal_common_mode_cut_sets(
    model: &DiagnosticPortfolioModel,
    baseline_active_sensor_ids: &[String],
) -> Result<Vec<Vec<String>>, DiagnosticPortfolioError> {
    let context = DiagnosabilityContext::new(model)?;
    let baseline = context.diagnose(baseline_active_sensor_ids)?;
    if !baseline.diagnosable {
        return Err(DiagnosticPortfolioError::new(
            "common-mode cut analysis requires a diagnosable baseline",
        ));
    }
    let baseline_ids = baseline.active_sensor_ids;
    let sensor_index = model.sensor_index();
    let mut groups = BTreeSet::new();
    for sensor_id in &baseline_ids {
        let sensor = sensor_index
            .get(sensor_id.as_str())
            .copied()
            .ok_or_else(|| DiagnosticPortfolioError::new("sensor index changed"))?;
        groups.extend(sensor.common_mode_groups.iter().cloned());
    }
    let groups: Vec<String> = groups.into_iter().collect();
    ensure_exact_enumeration_size(groups.len())?;
    let subset_count = 1_u64 << groups.len();
    let mut cuts: Vec<Vec<String>> = Vec::new();

    for mask in 1..subset_count {
        let lost_groups = subset_for_mask(&groups, mask);
        if cuts.iter().any(|prior| is_subset(prior, &lost_groups)) {
            continue;
        }
        let lost_set: BTreeSet<&str> = lost_groups.iter().map(String::as_str).collect();
        let remaining: Vec<String> = baseline_ids
            .iter()
            .filter(|sensor_id| {
                sensor_index
                    .get(sensor_id.as_str())
                    .is_some_and(|sensor| {
                        !sensor
                            .common_mode_groups
                            .iter()
                            .any(|group| lost_set.contains(group.as_str()))
                    })
            })
            .cloned()
            .collect();
        if !context.diagnose(&remaining)?.diagnosable {
            cuts.push(lost_groups);
        }
    }
    Ok(cuts)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CostedSufficientPortfolio {
    pub sensor_ids: Vec<String>,
    pub total_resource_cost: u64,
    pub resource_unit: String,
}

/// Finds the exact minimum-cost sufficient set under a declared budget.
/// Refuses to claim cost optimality if any candidate cost is unknown.
pub fn minimum_cost_sufficient_portfolio(
    model: &DiagnosticPortfolioModel,
    resource_budget: u64,
) -> Result<Option<CostedSufficientPortfolio>, DiagnosticPortfolioError> {
    let context = DiagnosabilityContext::new(model)?;
    let sensor_ids = model.sorted_sensor_ids();
    ensure_exact_enumeration_size(sensor_ids.len())?;
    let resource_unit = model
        .resource_unit
        .clone()
        .ok_or_else(|| DiagnosticPortfolioError::new("model resource unit is unknown"))?;
    let sensors = model.sensor_index();
    for sensor_id in &sensor_ids {
        let sensor = sensors
            .get(sensor_id.as_str())
            .copied()
            .ok_or_else(|| DiagnosticPortfolioError::new("sensor index changed"))?;
        if sensor.resource_cost.is_none() {
            return Err(DiagnosticPortfolioError::new(format!(
                "cannot claim minimum-cost optimality: cost for sensor {} is unknown",
                sensor.sensor_id
            )));
        }
    }

    let mut best: Option<CostedSufficientPortfolio> = None;
    let subset_count = 1_u64 << sensor_ids.len();
    for mask in 0..subset_count {
        let subset = subset_for_mask(&sensor_ids, mask);
        if !context.diagnose(&subset)?.diagnosable {
            continue;
        }
        let mut total_cost = 0_u64;
        for sensor_id in &subset {
            let cost = sensors
                .get(sensor_id.as_str())
                .and_then(|sensor| sensor.resource_cost)
                .ok_or_else(|| DiagnosticPortfolioError::new("validated cost disappeared"))?;
            total_cost = total_cost.checked_add(cost).ok_or_else(|| {
                DiagnosticPortfolioError::new("portfolio resource cost overflow")
            })?;
        }
        if total_cost > resource_budget {
            continue;
        }
        let candidate = CostedSufficientPortfolio {
            sensor_ids: subset,
            total_resource_cost: total_cost,
            resource_unit: resource_unit.clone(),
        };
        let replace = best.as_ref().is_none_or(|current| {
            candidate.total_resource_cost < current.total_resource_cost
                || (candidate.total_resource_cost == current.total_resource_cost
                    && (candidate.sensor_ids.len() < current.sensor_ids.len()
                        || (candidate.sensor_ids.len() == current.sensor_ids.len()
                            && candidate.sensor_ids < current.sensor_ids)))
        });
        if replace {
            best = Some(candidate);
        }
    }
    Ok(best)
}

/// A caller-declared numeric interval mapped to one categorical signature.
/// Bounds are [lower_inclusive, upper_exclusive); None means no upper bound.
/// Gaps between bands intentionally classify as UNKNOWN.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericCategoryBand {
    pub lower_inclusive: f64,
    pub upper_exclusive: Option<f64>,
    pub category: String,
}

/// Explicit mapping from one directly measured observation channel to one
/// categorical portfolio sensor. The category schema digest identifies the
/// mapping content; it is not a signature or proof of calibration authenticity.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticCategoryRule {
    pub sensor_id: String,
    pub channel_id: String,
    pub diagnostic_system_id: String,
    pub category_schema_id: String,
    pub category_schema_sha256: String,
    pub requirement: crate::plasma_observability::DecisionChannelRequirement,
    pub bands: Vec<NumericCategoryBand>,
}

impl DiagnosticCategoryRule {
    fn validate(&self) -> Result<(), DiagnosticPortfolioError> {
        non_empty(&self.sensor_id, "category sensor_id")?;
        non_empty(&self.channel_id, "category channel_id")?;
        non_empty(&self.diagnostic_system_id, "category diagnostic_system_id")?;
        non_empty(&self.category_schema_id, "category_schema_id")?;
        if !valid_sha256(&self.category_schema_sha256) {
            return Err(DiagnosticPortfolioError::new(
                "category schema digest must be 64 lowercase hexadecimal characters",
            ));
        }
        if self.requirement.channel_id != self.channel_id {
            return Err(DiagnosticPortfolioError::new(
                "category rule channel_id must match its decision requirement",
            ));
        }
        if self.bands.is_empty() {
            return Err(DiagnosticPortfolioError::new(
                "category rule requires at least one numeric band",
            ));
        }

        let mut previous_upper: Option<f64> = None;
        for (index, band) in self.bands.iter().enumerate() {
            if !band.lower_inclusive.is_finite() {
                return Err(DiagnosticPortfolioError::new(
                    "category band lower bound must be finite",
                ));
            }
            non_empty(&band.category, "category label")?;
            if let Some(upper) = band.upper_exclusive {
                if !upper.is_finite() || upper <= band.lower_inclusive {
                    return Err(DiagnosticPortfolioError::new(
                        "category band upper bound must be finite and greater than lower bound",
                    ));
                }
            }
            if index > 0 {
                if let Some(previous) = previous_upper {
                    if band.lower_inclusive < previous {
                        return Err(DiagnosticPortfolioError::new(
                            "category bands must be ordered and non-overlapping",
                        ));
                    }
                } else {
                    return Err(DiagnosticPortfolioError::new(
                        "an unbounded category band must be the final band",
                    ));
                }
            }
            previous_upper = band.upper_exclusive;
        }
        Ok(())
    }

    fn classify(&self, value: f64) -> Option<&str> {
        self.bands
            .iter()
            .find(|band| {
                value >= band.lower_inclusive
                    && band.upper_exclusive.is_none_or(|upper| value < upper)
            })
            .map(|band| band.category.as_str())
    }
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PortfolioEvidenceBlockReason {
    SyntheticNotMeasurement,
    Unavailable,
    Invalid,
    Delayed,
    Degraded,
    WrongUnits,
    Stale,
    UnapprovedSource,
    CalibrationMismatch,
}

impl From<crate::plasma_observability::AbstentionReason> for PortfolioEvidenceBlockReason {
    fn from(reason: crate::plasma_observability::AbstentionReason) -> Self {
        match reason {
            crate::plasma_observability::AbstentionReason::SyntheticNotMeasurement => {
                Self::SyntheticNotMeasurement
            }
            crate::plasma_observability::AbstentionReason::Unavailable => Self::Unavailable,
            crate::plasma_observability::AbstentionReason::Invalid => Self::Invalid,
            crate::plasma_observability::AbstentionReason::Delayed => Self::Delayed,
            crate::plasma_observability::AbstentionReason::Degraded => Self::Degraded,
            crate::plasma_observability::AbstentionReason::WrongUnits => Self::WrongUnits,
            crate::plasma_observability::AbstentionReason::Stale => Self::Stale,
            crate::plasma_observability::AbstentionReason::UnapprovedSource => Self::UnapprovedSource,
            crate::plasma_observability::AbstentionReason::CalibrationMismatch => {
                Self::CalibrationMismatch
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DiagnosticUnknownReason {
    EvidenceGateBlocked {
        reason: PortfolioEvidenceBlockReason,
    },
    OutsideDeclaredCategoryDomain,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticFrameSensorEvidence {
    pub sensor_id: String,
    pub channel_id: String,
    pub observation: SensorObservation,
    pub unknown_reason: Option<DiagnosticUnknownReason>,
    pub source_id: Option<String>,
    pub source_digest: Option<String>,
    pub calibration_id: Option<String>,
    pub calibration_revision: Option<String>,
    pub observed_at_ns: Option<u64>,
    pub category_schema_id: String,
    pub category_schema_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldCompatibility {
    pub world_id: String,
    pub decision_id: String,
    pub compatible: bool,
    /// Active, directly measured sensor IDs that contradict this world.
    pub contradictory_sensor_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum FrameDecisionDisposition {
    Identified { decision_id: String },
    Ambiguous { decision_ids: Vec<String> },
    NoCompatibleWorld,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticPortfolioFrameResult {
    pub shot_id: String,
    pub frame_id: String,
    pub frame_time_ns: u64,
    pub active_sensor_ids: Vec<String>,
    pub sensor_evidence: Vec<DiagnosticFrameSensorEvidence>,
    /// Design-time diagnosability of the declared complete sensor architecture.
    pub structural_diagnosability: DiagnosabilityResult,
    /// Worlds consistent with this frame's admitted categorical measurements.
    pub world_compatibility: Vec<WorldCompatibility>,
    pub compatible_world_ids: Vec<String>,
    pub compatible_decision_ids: Vec<String>,
    /// A decision is identified only if every compatible world requires it.
    pub frame_decision: FrameDecisionDisposition,
}

/// Evaluate a real observation frame against declared categorical sensor
/// rules. Only channels that pass the shared measured-input gate and fall in a
/// declared category band become Known. All other dispositions become
/// UNKNOWN; synthetic reconstruction is never a new independent sensor.
pub fn evaluate_observation_frame(
    model: &DiagnosticPortfolioModel,
    frame: &crate::plasma_observability::PlasmaObservationFrame,
    rules: &[DiagnosticCategoryRule],
    active_sensor_ids: &[String],
) -> Result<DiagnosticPortfolioFrameResult, DiagnosticPortfolioError> {
    model.validate()?;
    frame
        .validate()
        .map_err(|error| DiagnosticPortfolioError::new(error.message))?;

    if rules.len() != model.sensors.len() {
        return Err(DiagnosticPortfolioError::new(
            "exactly one category rule is required for every candidate sensor",
        ));
    }

    let mut rules_by_sensor = BTreeMap::new();
    let mut channels = BTreeSet::new();
    for rule in rules {
        rule.validate()?;
        if rules_by_sensor.insert(rule.sensor_id.as_str(), rule).is_some() {
            return Err(DiagnosticPortfolioError::new(format!(
                "duplicate category rule for sensor {}",
                rule.sensor_id
            )));
        }
        if !channels.insert(rule.channel_id.as_str()) {
            return Err(DiagnosticPortfolioError::new(format!(
                "channel {} is bound to more than one portfolio sensor",
                rule.channel_id
            )));
        }
        if !model
            .sensors
            .iter()
            .any(|sensor| sensor.sensor_id == rule.sensor_id)
        {
            return Err(DiagnosticPortfolioError::new(format!(
                "category rule references unknown portfolio sensor {}",
                rule.sensor_id
            )));
        }
    }
    for sensor in &model.sensors {
        if !rules_by_sensor.contains_key(sensor.sensor_id.as_str()) {
            return Err(DiagnosticPortfolioError::new(format!(
                "missing category rule for portfolio sensor {}",
                sensor.sensor_id
            )));
        }
    }

    let channels_by_id: BTreeMap<&str, &crate::plasma_observability::DiagnosticChannelObservation> =
        frame
            .channels
            .iter()
            .map(|channel| (channel.channel_id.as_str(), channel))
            .collect();
    let mut sensor_evidence = Vec::with_capacity(model.sensors.len());

    for sensor in &model.sensors {
        let rule = rules_by_sensor
            .get(sensor.sensor_id.as_str())
            .copied()
            .ok_or_else(|| DiagnosticPortfolioError::new("validated category rule disappeared"))?;
        let channel = channels_by_id
            .get(rule.channel_id.as_str())
            .copied()
            .ok_or_else(|| {
                DiagnosticPortfolioError::new(format!(
                    "frame is missing category channel {}",
                    rule.channel_id
                ))
            })?;
        if channel.diagnostic_system_id != rule.diagnostic_system_id {
            return Err(DiagnosticPortfolioError::new(format!(
                "channel {} is bound to diagnostic system {}, expected {}",
                rule.channel_id, channel.diagnostic_system_id, rule.diagnostic_system_id
            )));
        }

        let gate = frame
            .evaluate_decision_requirements(std::slice::from_ref(&rule.requirement))
            .map_err(|error| DiagnosticPortfolioError::new(error.message))?;

        let evidence = match gate {
            crate::plasma_observability::DecisionInputGate::Abstain {
                blocked_channels,
                ..
            } => {
                let blocked = blocked_channels.into_iter().next().ok_or_else(|| {
                    DiagnosticPortfolioError::new(
                        "measurement gate abstained without a blocked-channel reason",
                    )
                })?;
                DiagnosticFrameSensorEvidence {
                    sensor_id: sensor.sensor_id.clone(),
                    channel_id: rule.channel_id.clone(),
                    observation: SensorObservation::Unknown,
                    unknown_reason: Some(DiagnosticUnknownReason::EvidenceGateBlocked {
                        reason: blocked.reason.into(),
                    }),
                    source_id: None,
                    source_digest: None,
                    calibration_id: None,
                    calibration_revision: None,
                    observed_at_ns: None,
                    category_schema_id: rule.category_schema_id.clone(),
                    category_schema_sha256: rule.category_schema_sha256.clone(),
                }
            }
            crate::plasma_observability::DecisionInputGate::MeasuredInputsAvailable(inputs) => {
                let measured = inputs.get(&rule.channel_id).ok_or_else(|| {
                    DiagnosticPortfolioError::new("measured channel disappeared after gate")
                })?;
                if let Some(category) = rule.classify(measured.value()) {
                    DiagnosticFrameSensorEvidence {
                        sensor_id: sensor.sensor_id.clone(),
                        channel_id: rule.channel_id.clone(),
                        observation: SensorObservation::Known(category.to_owned()),
                        unknown_reason: None,
                        source_id: Some(measured.source_id().to_owned()),
                        source_digest: Some(measured.source_digest().to_owned()),
                        calibration_id: Some(measured.calibration_id().to_owned()),
                        calibration_revision: Some(
                            measured.calibration_revision().to_owned(),
                        ),
                        observed_at_ns: Some(measured.observed_at_ns()),
                        category_schema_id: rule.category_schema_id.clone(),
                        category_schema_sha256: rule.category_schema_sha256.clone(),
                    }
                } else {
                    DiagnosticFrameSensorEvidence {
                        sensor_id: sensor.sensor_id.clone(),
                        channel_id: rule.channel_id.clone(),
                        observation: SensorObservation::Unknown,
                        unknown_reason: Some(
                            DiagnosticUnknownReason::OutsideDeclaredCategoryDomain,
                        ),
                        source_id: Some(measured.source_id().to_owned()),
                        source_digest: Some(measured.source_digest().to_owned()),
                        calibration_id: Some(measured.calibration_id().to_owned()),
                        calibration_revision: Some(
                            measured.calibration_revision().to_owned(),
                        ),
                        observed_at_ns: Some(measured.observed_at_ns()),
                        category_schema_id: rule.category_schema_id.clone(),
                        category_schema_sha256: rule.category_schema_sha256.clone(),
                    }
                }
            }
        };

        // A direct-measurement category must never be sourced from a non-measured frame status.
        if !matches!(
            &channel.disposition,
            crate::plasma_observability::ObservationDisposition::Measured { .. }
        ) && !matches!(evidence.observation, SensorObservation::Unknown)
        {
            return Err(DiagnosticPortfolioError::new(
                "non-measured observation was incorrectly promoted to a known category",
            ));
        }
        sensor_evidence.push(evidence);
    }

    let observed_sensor_ids = model
        .sensors
        .iter()
        .map(|sensor| sensor.sensor_id.as_str())
        .collect::<BTreeSet<_>>();
    for sensor_id in active_sensor_ids {
        if !observed_sensor_ids.contains(sensor_id.as_str()) {
            return Err(DiagnosticPortfolioError::new(format!(
                "active sensor {} is not present in the portfolio model",
                sensor_id
            )));
        }
    }

    let structural_diagnosability = diagnose(model, active_sensor_ids)?;
    let active: BTreeSet<&str> = structural_diagnosability
        .active_sensor_ids
        .iter()
        .map(String::as_str)
        .collect();
    let evidence_by_sensor: BTreeMap<&str, &SensorObservation> = sensor_evidence
        .iter()
        .map(|item| (item.sensor_id.as_str(), &item.observation))
        .collect();
    let model_sensor_index = model.sensor_index();

    let mut world_compatibility = Vec::with_capacity(model.worlds.len());
    let mut compatible_world_ids = Vec::new();
    let mut compatible_decisions = BTreeSet::new();

    for world in &model.worlds {
        let mut contradictory_sensor_ids = Vec::new();
        for sensor_id in &active {
            let observed = evidence_by_sensor.get(sensor_id).copied().ok_or_else(|| {
                DiagnosticPortfolioError::new("validated frame sensor evidence disappeared")
            })?;
            let SensorObservation::Known(observed_category) = observed else {
                // UNKNOWN cannot eliminate a world.
                continue;
            };
            let sensor = model_sensor_index.get(sensor_id).copied().ok_or_else(|| {
                DiagnosticPortfolioError::new("validated model sensor disappeared")
            })?;
            let expected = sensor
                .observations
                .iter()
                .find(|item| item.world_id == world.world_id)
                .map(|item| &item.observation)
                .ok_or_else(|| {
                    DiagnosticPortfolioError::new(
                        "validated model world signature disappeared",
                    )
                })?;
            match expected {
                SensorObservation::Known(expected_category)
                    if expected_category != observed_category =>
                {
                    contradictory_sensor_ids.push((*sensor_id).to_owned());
                }
                SensorObservation::Known(_) | SensorObservation::Unknown => {}
            }
        }

        let compatible = contradictory_sensor_ids.is_empty();
        if compatible {
            compatible_world_ids.push(world.world_id.clone());
            compatible_decisions.insert(world.decision_id.clone());
        }
        world_compatibility.push(WorldCompatibility {
            world_id: world.world_id.clone(),
            decision_id: world.decision_id.clone(),
            compatible,
            contradictory_sensor_ids,
        });
    }

    let compatible_decision_ids: Vec<String> = compatible_decisions.into_iter().collect();
    let frame_decision = match compatible_decision_ids.as_slice() {
        [decision_id] => FrameDecisionDisposition::Identified {
            decision_id: decision_id.clone(),
        },
        [] => FrameDecisionDisposition::NoCompatibleWorld,
        decision_ids => FrameDecisionDisposition::Ambiguous {
            decision_ids: decision_ids.to_vec(),
        },
    };

    Ok(DiagnosticPortfolioFrameResult {
        shot_id: frame.shot_id.clone(),
        frame_id: frame.frame_id.clone(),
        frame_time_ns: frame.frame_time_ns,
        active_sensor_ids: structural_diagnosability.active_sensor_ids.clone(),
        sensor_evidence,
        structural_diagnosability,
        world_compatibility,
        compatible_world_ids,
        compatible_decision_ids,
        frame_decision,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn benchmark_manifest() -> serde_json::Value {
        serde_json::from_str(include_str!(
            "../../../../benchmarks/plasma/diagnostic-portfolio-v0.json"
        ))
        .expect("portfolio benchmark manifest should parse")
    }

    fn reference_model() -> DiagnosticPortfolioModel {
        serde_json::from_value(benchmark_manifest()["model"].clone())
            .expect("synthetic portfolio model should deserialize")
    }

    fn frame_and_category_rules() -> (
        crate::plasma_observability::PlasmaObservationFrame,
        Vec<DiagnosticCategoryRule>,
    ) {
        use crate::plasma_observability::{
            CalibrationIdentity, DecisionChannelRequirement, DiagnosticChannelObservation,
            ObservationDisposition, PlasmaObservationFrame, PLASMA_OBSERVATION_SCHEMA,
        };

        let model = reference_model();
        let frame_time_ns = 100_u64;
        let mut expected_channel_ids = Vec::new();
        let mut channels = Vec::new();
        let mut rules = Vec::new();

        for sensor in &model.sensors {
            let channel_id = format!("channel-{}", sensor.sensor_id);
            expected_channel_ids.push(channel_id.clone());
            let source_id = format!("source-{}", sensor.sensor_id);
            let calibration_id = format!("calibration-{}", sensor.sensor_id);
            let units = "synthetic-unit".to_owned();
            let (value, disposition) = if sensor.sensor_id == "failed_diag" {
                (
                    None,
                    ObservationDisposition::Unavailable {
                        reason: "synthetic fixture diagnostic unavailable".to_owned(),
                    },
                )
            } else {
                let value = 0.5;
                (
                    Some(value),
                    ObservationDisposition::Measured {
                        value,
                        units: units.clone(),
                        source_id: source_id.clone(),
                        source_digest: "a".repeat(64),
                        calibration: CalibrationIdentity {
                            calibration_id: calibration_id.clone(),
                            calibration_revision: "rev-1".to_owned(),
                        },
                        observed_at_ns: frame_time_ns,
                    },
                )
            };
            let _ = value;
            channels.push(DiagnosticChannelObservation {
                channel_id: channel_id.clone(),
                diagnostic_system_id: format!("system-{}", sensor.sensor_id),
                disposition,
            });
            rules.push(DiagnosticCategoryRule {
                sensor_id: sensor.sensor_id.clone(),
                channel_id: channel_id.clone(),
                diagnostic_system_id: format!("system-{}", sensor.sensor_id),
                category_schema_id: format!("category-map-{}", sensor.sensor_id),
                category_schema_sha256: "b".repeat(64),
                requirement: DecisionChannelRequirement {
                    channel_id,
                    expected_units: units,
                    max_age_ns: 1_000,
                    allowed_source_ids: vec![source_id],
                    calibration: CalibrationIdentity {
                        calibration_id,
                        calibration_revision: "rev-1".to_owned(),
                    },
                },
                bands: if sensor.sensor_id.starts_with("gen_") {
                    vec![
                        NumericCategoryBand {
                            lower_inclusive: 0.0,
                            upper_exclusive: Some(1.0),
                            category: "H".to_owned(),
                        },
                        NumericCategoryBand {
                            lower_inclusive: 1.0,
                            upper_exclusive: Some(2.0),
                            category: "F".to_owned(),
                        },
                    ]
                } else {
                    vec![
                        NumericCategoryBand {
                            lower_inclusive: 0.0,
                            upper_exclusive: Some(1.0),
                            category: "C".to_owned(),
                        },
                        NumericCategoryBand {
                            lower_inclusive: 1.0,
                            upper_exclusive: Some(2.0),
                            category: "O".to_owned(),
                        },
                    ]
                },
            });
        }

        (
            PlasmaObservationFrame {
                schema_version: PLASMA_OBSERVATION_SCHEMA.to_owned(),
                shot_id: "synthetic-portfolio-shot-001".to_owned(),
                frame_id: "synthetic-portfolio-frame-001".to_owned(),
                clock_domain: "shot-relative-ns".to_owned(),
                frame_time_ns,
                expected_channel_ids,
                channels,
            },
            rules,
        )
    }

    fn set_channel_disposition(
        frame: &mut crate::plasma_observability::PlasmaObservationFrame,
        channel_id: &str,
        disposition: crate::plasma_observability::ObservationDisposition,
    ) {
        let channel = frame
            .channels
            .iter_mut()
            .find(|channel| channel.channel_id == channel_id)
            .expect("fixture channel");
        channel.disposition = disposition;
    }

    fn all_sensor_ids(model: &DiagnosticPortfolioModel) -> Vec<String> {
        model
            .sensors
            .iter()
            .map(|sensor| sensor.sensor_id.clone())
            .collect()
    }

    #[test]
    fn frozen_portfolio_manifest_matches_rust_oracle_outputs() {
        let manifest = benchmark_manifest();
        assert_eq!(
            manifest["schema_version"],
            "luminous.plasma-diagnostic-portfolio-benchmark.v0.1"
        );
        assert_eq!(
            manifest["benchmark_id"],
            PLASMA_DIAGNOSTIC_PORTFOLIO_BENCHMARK_ID
        );
        assert_eq!(manifest["physical_measurements"], false);
        assert_eq!(manifest["resource_costs_are_synthetic_test_values"], true);
        assert_eq!(
            manifest["claim_ceiling"],
            "FINITE_SYMBOLIC_DECISION_SUFFICIENCY_ONLY_NO_PHYSICAL_OBSERVABILITY_OR_SENSOR_RELIABILITY_CLAIM"
        );

        let model = reference_model();
        let all = all_sensor_ids(&model);
        let expected_minimal: Vec<Vec<String>> =
            serde_json::from_value(manifest["expected"]["minimal_sufficient_sensor_sets"].clone())
                .expect("minimal set fixture");
        let expected_sensor_cuts: Vec<Vec<String>> =
            serde_json::from_value(manifest["expected"]["minimal_sensor_cut_sets"].clone())
                .expect("sensor cut fixture");
        let expected_group_cuts: Vec<Vec<String>> =
            serde_json::from_value(manifest["expected"]["minimal_common_mode_cut_sets"].clone())
                .expect("common mode cut fixture");

        assert!(diagnose(&model, &all).expect("full portfolio").diagnosable);
        assert_eq!(minimal_sufficient_sensor_sets(&model).expect("minimal sets"), expected_minimal);
        assert_eq!(minimal_sensor_cut_sets(&model, &all).expect("sensor cuts"), expected_sensor_cuts);
        assert_eq!(
            minimal_common_mode_cut_sets(&model, &all).expect("group cuts"),
            expected_group_cuts
        );
        let expected_cost: CostedSufficientPortfolio =
            serde_json::from_value(manifest["expected"]["minimum_cost_sufficient_portfolio_at_budget_4"].clone())
                .expect("cost fixture");
        assert_eq!(
            minimum_cost_sufficient_portfolio(&model, 4)
                .expect("cost optimizer")
                .expect("feasible"),
            expected_cost
        );
    }

    #[test]
    fn measured_frame_identifies_decision_and_preserves_sensor_provenance() {
        let model = reference_model();
        let (frame, rules) = frame_and_category_rules();
        let active = all_sensor_ids(&model);
        let result = evaluate_observation_frame(&model, &frame, &rules, &active)
            .expect("measured portfolio frame should evaluate");

        assert!(result.structural_diagnosability.diagnosable);
        assert_eq!(
            result.frame_decision,
            FrameDecisionDisposition::Identified {
                decision_id: "PROCEED".to_owned()
            }
        );
        assert_eq!(result.compatible_world_ids, vec!["healthy_closed".to_owned()]);
        let density = result
            .sensor_evidence
            .iter()
            .find(|item| item.sensor_id == "gen_current")
            .expect("density sensor evidence");
        assert_eq!(density.observation, SensorObservation::Known("H".to_owned()));
        assert_eq!(density.source_id.as_deref(), Some("source-gen_current"));
        assert!(density
            .source_digest
            .as_ref()
            .is_some_and(|digest| digest == &"a".repeat(64)));
        assert_eq!(density.calibration_revision.as_deref(), Some("rev-1"));
    }

    #[test]
    fn synthetic_and_unavailable_channels_remain_unknown_and_leave_decision_ambiguous() {
        use crate::plasma_observability::ObservationDisposition;

        let model = reference_model();
        let (mut frame, rules) = frame_and_category_rules();
        set_channel_disposition(
            &mut frame,
            "channel-tie_aux",
            ObservationDisposition::Synthetic {
                value: 0.5,
                units: "synthetic-unit".to_owned(),
                model_id: "diag-reconstruction".to_owned(),
                model_revision: "rev-1".to_owned(),
                input_digest: "c".repeat(64),
            },
        );
        set_channel_disposition(
            &mut frame,
            "channel-tie_optical",
            ObservationDisposition::Unavailable {
                reason: "optical subsystem unavailable".to_owned(),
            },
        );

        let result = evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model),
        )
        .expect("unqualified channels should become UNKNOWN");
        assert_eq!(
            result.frame_decision,
            FrameDecisionDisposition::Ambiguous {
                decision_ids: vec!["BLOCK".to_owned(), "PROCEED".to_owned()]
            }
        );
        for sensor_id in ["tie_aux", "tie_optical"] {
            let evidence = result
                .sensor_evidence
                .iter()
                .find(|item| item.sensor_id == sensor_id)
                .expect("sensor evidence");
            assert_eq!(evidence.observation, SensorObservation::Unknown);
            assert!(evidence.source_id.is_none());
        }
    }

    #[test]
    fn out_of_category_domain_is_unknown_but_an_independent_channel_can_still_identify() {
        use crate::plasma_observability::ObservationDisposition;

        let model = reference_model();
        let (mut frame, rules) = frame_and_category_rules();
        set_channel_disposition(
            &mut frame,
            "channel-tie_aux",
            ObservationDisposition::Measured {
                value: 99.0,
                units: "synthetic-unit".to_owned(),
                source_id: "source-tie_aux".to_owned(),
                source_digest: "a".repeat(64),
                calibration: crate::plasma_observability::CalibrationIdentity {
                    calibration_id: "calibration-tie_aux".to_owned(),
                    calibration_revision: "rev-1".to_owned(),
                },
                observed_at_ns: 100,
            },
        );
        let result = evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model),
        )
        .expect("out-of-domain measurement should remain unknown");
        let evidence = result
            .sensor_evidence
            .iter()
            .find(|item| item.sensor_id == "tie_aux")
            .expect("tie auxiliary evidence");
        assert_eq!(evidence.observation, SensorObservation::Unknown);
        assert_eq!(
            evidence.unknown_reason,
            Some(DiagnosticUnknownReason::OutsideDeclaredCategoryDomain)
        );
        assert_eq!(
            result.frame_decision,
            FrameDecisionDisposition::Identified {
                decision_id: "PROCEED".to_owned()
            }
        );
    }

    #[test]
    fn frame_bridge_rejects_overlapping_bands_duplicate_channels_and_incomplete_rules() {
        let model = reference_model();
        let (frame, mut rules) = frame_and_category_rules();
        rules[0].bands[1].lower_inclusive = 0.5;
        assert!(evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model)
        ).is_err());

        let (frame, mut rules) = frame_and_category_rules();
        rules[1].channel_id = rules[0].channel_id.clone();
        rules[1].requirement.channel_id = rules[0].channel_id.clone();
        assert!(evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model)
        ).is_err());

        let (frame, mut rules) = frame_and_category_rules();
        rules.pop();
        assert!(evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model)
        ).is_err());
    }

    #[test]
    fn frame_bridge_uses_decision_gate_for_units_and_freshness() {
        let model = reference_model();
        let (frame, mut rules) = frame_and_category_rules();
        rules[0].requirement.expected_units = "wrong-unit".to_owned();
        let result = evaluate_observation_frame(
            &model,
            &frame,
            &rules,
            &all_sensor_ids(&model)
        ).expect("unit mismatch should abstain for this channel");
        let evidence = result.sensor_evidence
            .iter()
            .find(|item| item.sensor_id == "gen_current")
            .expect("sensor evidence");
        assert_eq!(evidence.observation, SensorObservation::Unknown);
        assert_eq!(
            evidence.unknown_reason,
            Some(DiagnosticUnknownReason::EvidenceGateBlocked {
                reason: PortfolioEvidenceBlockReason::WrongUnits
            })
        );
    }

    #[test]
    fn matches_independent_pie_009n_reference_semantics() {
        let model = reference_model();
        let all = all_sensor_ids(&model);
        assert!(diagnose(&model, &all).expect("baseline").diagnosable);

        let minimal: BTreeSet<Vec<String>> = minimal_sufficient_sensor_sets(&model)
            .expect("minimal sets")
            .into_iter()
            .collect();
        let expected: BTreeSet<Vec<String>> = [
            vec!["gen_current", "tie_aux"],
            vec!["gen_current", "tie_optical"],
            vec!["gen_vibration", "tie_aux"],
            vec!["gen_vibration", "tie_optical"],
        ]
        .into_iter()
        .map(|items| items.into_iter().map(str::to_owned).collect())
        .collect();
        assert_eq!(minimal, expected);

        let cuts: BTreeSet<Vec<String>> =
            minimal_sensor_cut_sets(&model, &all).expect("sensor cuts").into_iter().collect();
        let expected_cuts: BTreeSet<Vec<String>> = [
            vec!["gen_current", "gen_vibration"],
            vec!["tie_aux", "tie_optical"],
        ]
        .into_iter()
        .map(|items| items.into_iter().map(str::to_owned).collect())
        .collect();
        assert_eq!(cuts, expected_cuts);

        let group_cuts: BTreeSet<Vec<String>> = minimal_common_mode_cut_sets(&model, &all)
            .expect("common-mode cuts")
            .into_iter()
            .collect();
        let expected_group_cuts: BTreeSet<Vec<String>> = [
            vec!["gen_electrical", "gen_mechanical"],
            vec!["optical", "switch_aux"],
        ]
        .into_iter()
        .map(|items| items.into_iter().map(str::to_owned).collect())
        .collect();
        assert_eq!(group_cuts, expected_group_cuts);
    }

    #[test]
    fn unknown_and_mixed_unknown_observations_never_separate_worlds() {
        let model = reference_model();
        let result = diagnose(
            &model,
            &["failed_diag".to_owned(), "tie_aux".to_owned()],
        )
        .expect("diagnose");
        assert!(!result.diagnosable);
        assert!(
            result
                .unresolved_cross_decision_pairs
                .iter()
                .any(|pair| pair.left_world_id == "healthy_closed"
                    && pair.right_world_id == "failed_closed")
        );

        let mut mixed = model.clone();
        let sensor = mixed
            .sensors
            .iter_mut()
            .find(|sensor| sensor.sensor_id == "failed_diag")
            .expect("unknown diagnostic");
        sensor.observations[0].observation = SensorObservation::Unknown;
        sensor.observations[1].observation = SensorObservation::Known("F".to_owned());
        assert!(!diagnose(&mixed, &["failed_diag".to_owned()])
            .expect("mixed unknown")
            .diagnosable);
    }

    #[test]
    fn any_declared_common_mode_group_removes_a_multi_domain_sensor() {
        let mut model = reference_model();
        let sensor = model
            .sensors
            .iter_mut()
            .find(|sensor| sensor.sensor_id == "gen_current")
            .expect("generator current");
        sensor.common_mode_groups.push("shared_adc_power".to_owned());

        let cuts: BTreeSet<Vec<String>> = minimal_common_mode_cut_sets(
            &model,
            &all_sensor_ids(&model),
        )
        .expect("common mode cuts")
        .into_iter()
        .collect();
        assert!(cuts.contains(&vec![
            "gen_mechanical".to_owned(),
            "shared_adc_power".to_owned()
        ]));
    }

    #[test]
    fn exact_cost_oracle_uses_declared_costs_and_hard_budget() {
        let model = reference_model();
        let selected = minimum_cost_sufficient_portfolio(&model, 4)
            .expect("cost selection")
            .expect("feasible portfolio");
        assert_eq!(
            selected.sensor_ids,
            vec!["gen_vibration".to_owned(), "tie_optical".to_owned()]
        );
        assert_eq!(selected.total_resource_cost, 4);
        assert_eq!(selected.resource_unit, "synthetic_resource_unit");
        assert!(
            minimum_cost_sufficient_portfolio(&model, 3)
                .expect("cost selection")
                .is_none()
        );
    }

    #[test]
    fn unknown_costs_never_become_zero_cost_for_optimizer() {
        let mut model = reference_model();
        let sensor = model
            .sensors
            .iter_mut()
            .find(|sensor| sensor.sensor_id == "failed_diag")
            .expect("failed diagnostic");
        sensor.resource_cost = None;
        sensor.resource_unit = None;
        model.resource_unit = Some("synthetic_resource_unit".to_owned());
        assert!(minimum_cost_sufficient_portfolio(&model, 100).is_err());
    }

    #[test]
    fn removing_information_never_improves_decision_diagnosability() {
        let model = reference_model();
        assert!(!diagnose(&model, &[]).expect("empty active set").diagnosable);
        assert!(!diagnose(&model, &["gen_current".to_owned()])
            .expect("single sensor")
            .diagnosable);
        assert!(diagnose(
            &model,
            &["gen_current".to_owned(), "tie_aux".to_owned()]
        )
        .expect("minimal sufficient set")
        .diagnosable);
    }

    #[test]
    fn undiagnosable_baselines_and_malformed_models_fail_closed() {
        let model = reference_model();
        assert!(minimal_sensor_cut_sets(&model, &["failed_diag".to_owned()]).is_err());
        assert!(minimal_common_mode_cut_sets(&model, &["failed_diag".to_owned()]).is_err());

        let mut duplicate = model.clone();
        duplicate.sensors.push(duplicate.sensors[0].clone());
        assert!(duplicate.validate().is_err());

        let mut incomplete = model.clone();
        incomplete.sensors[0].observations.pop();
        assert!(incomplete.validate().is_err());

        let mut empty_value = model;
        empty_value.sensors[0].observations[0].observation =
            SensorObservation::Known(String::new());
        assert!(empty_value.validate().is_err());
    }

    #[test]
    fn exact_enumerators_refuse_unbounded_candidate_sets() {
        let mut model = reference_model();
        let template = model.sensors[0].clone();
        for index in 0..(MAX_EXACT_SENSOR_COUNT + 1 - model.sensors.len()) {
            let mut sensor = template.clone();
            sensor.sensor_id = format!("extra-{index:02}");
            sensor.common_mode_groups = vec![format!("extra-domain-{index:02}")];
            model.sensors.push(sensor);
        }
        assert!(minimal_sufficient_sensor_sets(&model).is_err());
    }
}

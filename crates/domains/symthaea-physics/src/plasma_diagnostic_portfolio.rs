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
    model.validate()?;
    let sensor_ids = model.sorted_sensor_ids();
    ensure_exact_enumeration_size(sensor_ids.len())?;
    let subset_count = 1_u64 << sensor_ids.len();
    let mut sufficient: Vec<Vec<String>> = Vec::new();

    for mask in 0..subset_count {
        let subset = subset_for_mask(&sensor_ids, mask);
        if sufficient.iter().any(|prior| is_subset(prior, &subset)) {
            continue;
        }
        if diagnose(model, &subset)?.diagnosable {
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
    let baseline = diagnose(model, baseline_active_sensor_ids)?;
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
        if !diagnose(model, &remaining)?.diagnosable {
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
    let baseline = diagnose(model, baseline_active_sensor_ids)?;
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
        if !diagnose(model, &remaining)?.diagnosable {
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
    model.validate()?;
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
        if !diagnose(model, &subset)?.diagnosable {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_model() -> DiagnosticPortfolioModel {
        let worlds = vec![
            DecisionWorld {
                world_id: "healthy_closed".to_owned(),
                decision_id: "PROCEED".to_owned(),
            },
            DecisionWorld {
                world_id: "failed_closed".to_owned(),
                decision_id: "BLOCK".to_owned(),
            },
            DecisionWorld {
                world_id: "healthy_open".to_owned(),
                decision_id: "BLOCK".to_owned(),
            },
            DecisionWorld {
                world_id: "failed_open".to_owned(),
                decision_id: "BLOCK".to_owned(),
            },
        ];

        let sensor = |sensor_id: &str,
                      group: &str,
                      cost: u64,
                      values: [&str; 4]| DiagnosticSensor {
            sensor_id: sensor_id.to_owned(),
            common_mode_groups: vec![group.to_owned()],
            resource_cost: Some(cost),
            resource_unit: Some("synthetic_resource_unit".to_owned()),
            observations: worlds
                .iter()
                .zip(values)
                .map(|(world, value)| SensorWorldObservation {
                    world_id: world.world_id.clone(),
                    observation: if value == "UNKNOWN" {
                        SensorObservation::Unknown
                    } else {
                        SensorObservation::Known(value.to_owned())
                    },
                })
                .collect(),
        };

        DiagnosticPortfolioModel {
            schema_version: PLASMA_DIAGNOSTIC_PORTFOLIO_SCHEMA.to_owned(),
            model_id: "synthetic-plasma-decisions-v0".to_owned(),
            decision_catalogue_id: "synthetic-proceed-block-v0".to_owned(),
            resource_unit: Some("synthetic_resource_unit".to_owned()),
            worlds,
            sensors: vec![
                sensor(
                    "gen_current",
                    "gen_electrical",
                    2,
                    ["H", "F", "H", "F"],
                ),
                sensor(
                    "gen_vibration",
                    "gen_mechanical",
                    1,
                    ["H", "F", "H", "F"],
                ),
                sensor("tie_aux", "switch_aux", 4, ["C", "C", "O", "O"]),
                sensor("tie_optical", "optical", 3, ["C", "C", "O", "O"]),
                sensor(
                    "failed_diag",
                    "diag_bus",
                    1,
                    ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
                ),
            ],
        }
    }

    fn all_sensor_ids(model: &DiagnosticPortfolioModel) -> Vec<String> {
        model
            .sensors
            .iter()
            .map(|sensor| sensor.sensor_id.clone())
            .collect()
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

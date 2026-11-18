//! Decision-specific structural diagnosability and sensor-cut analysis.
//!
//! This deterministic symbolic oracle asks whether active sensing paths
//! distinguish every pair of hidden worlds requiring different decisions.
//! It does not infer probabilities, measurement noise, calibration quality,
//! or physical sensor reliability.
//!
//! Unknown is non-informative. It never separates a pair, including when the
//! other world predicts a known value.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Exact subset enumeration is deliberately bounded in this v0.1 oracle.
pub const MAX_EXACT_SENSOR_COUNT: usize = 16;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosabilityError {
    pub message: String,
}

impl DiagnosabilityError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DiagnosabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DiagnosabilityError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionWorld {
    pub world_id: String,
    pub decision_id: String,
}

/// Unknown represents an unavailable or uninformative observation, not a
/// categorical sensor value that can distinguish a world.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum SensorObservation {
    Known(String),
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorldSensorObservation {
    pub world_id: String,
    pub observation: SensorObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticSensor {
    pub sensor_id: String,
    /// Shared acquisition, power, clock, calibration, environment or other
    /// declared common-cause domain. Distinct IDs do not imply independence.
    pub common_mode_group: String,
    pub outcomes: Vec<WorldSensorObservation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosabilityModel {
    pub worlds: Vec<DecisionWorld>,
    pub sensors: Vec<DiagnosticSensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossDecisionPairReport {
    pub left_world_id: String,
    pub right_world_id: String,
    pub left_decision_id: String,
    pub right_decision_id: String,
    pub separating_sensor_ids: Vec<String>,
}

impl CrossDecisionPairReport {
    pub fn is_separated(&self) -> bool {
        !self.separating_sensor_ids.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosabilityResult {
    pub diagnosable: bool,
    pub cross_decision_pairs: Vec<CrossDecisionPairReport>,
}

impl DiagnosabilityResult {
    pub fn unresolved_pairs(&self) -> Vec<&CrossDecisionPairReport> {
        self.cross_decision_pairs
            .iter()
            .filter(|pair| !pair.is_separated())
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorSet {
    pub sensor_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorCutSet {
    pub lost_sensor_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CommonModeCutSet {
    pub lost_common_mode_groups: Vec<String>,
}

impl DiagnosabilityModel {
    /// Require unique worlds/sensors, at least two decision classes, explicit
    /// common-mode groups and one observation (including Unknown) per world.
    pub fn validate(&self) -> Result<(), DiagnosabilityError> {
        if self.worlds.len() < 2 {
            return Err(DiagnosabilityError::new(
                "at least two decision worlds are required",
            ));
        }

        let mut world_ids = BTreeSet::new();
        let mut decisions = BTreeSet::new();
        for world in &self.worlds {
            if world.world_id.trim().is_empty() || world.decision_id.trim().is_empty() {
                return Err(DiagnosabilityError::new(
                    "world IDs and decision IDs must be non-empty",
                ));
            }
            if !world_ids.insert(world.world_id.as_str()) {
                return Err(DiagnosabilityError::new(format!(
                    "duplicate world ID: {}",
                    world.world_id
                )));
            }
            decisions.insert(world.decision_id.as_str());
        }
        if decisions.len() < 2 {
            return Err(DiagnosabilityError::new(
                "at least two distinct decision classes are required",
            ));
        }

        let mut sensor_ids = BTreeSet::new();
        for sensor in &self.sensors {
            if sensor.sensor_id.trim().is_empty()
                || sensor.common_mode_group.trim().is_empty()
            {
                return Err(DiagnosabilityError::new(
                    "sensor ID and common-mode group must be non-empty",
                ));
            }
            if !sensor_ids.insert(sensor.sensor_id.as_str()) {
                return Err(DiagnosabilityError::new(format!(
                    "duplicate sensor ID: {}",
                    sensor.sensor_id
                )));
            }

            let mut observed_worlds = BTreeSet::new();
            for outcome in &sensor.outcomes {
                if outcome.world_id.trim().is_empty()
                    || !world_ids.contains(outcome.world_id.as_str())
                {
                    return Err(DiagnosabilityError::new(format!(
                        "sensor {} references an empty or unknown world",
                        sensor.sensor_id
                    )));
                }
                if !observed_worlds.insert(outcome.world_id.as_str()) {
                    return Err(DiagnosabilityError::new(format!(
                        "sensor {} repeats world {}",
                        sensor.sensor_id, outcome.world_id
                    )));
                }
                if let SensorObservation::Known(value) = &outcome.observation {
                    if value.trim().is_empty() {
                        return Err(DiagnosabilityError::new(format!(
                            "sensor {} uses an empty known value; use Unknown explicitly",
                            sensor.sensor_id
                        )));
                    }
                }
            }
            if observed_worlds != world_ids {
                return Err(DiagnosabilityError::new(format!(
                    "sensor {} must declare exactly one outcome for every world",
                    sensor.sensor_id
                )));
            }
        }
        Ok(())
    }

    /// Evaluate an active sensor set. Duplicate and unknown IDs fail closed.
    pub fn diagnose(
        &self,
        active_sensor_ids: &[&str],
    ) -> Result<DiagnosabilityResult, DiagnosabilityError> {
        self.validate()?;

        let mut active = BTreeSet::new();
        for sensor_id in active_sensor_ids {
            if sensor_id.trim().is_empty() {
                return Err(DiagnosabilityError::new(
                    "active sensor IDs must be non-empty",
                ));
            }
            if !active.insert(*sensor_id) {
                return Err(DiagnosabilityError::new(format!(
                    "duplicate active sensor ID: {sensor_id}"
                )));
            }
        }

        for sensor_id in &active {
            if !self.sensors.iter().any(|sensor| sensor.sensor_id == *sensor_id) {
                return Err(DiagnosabilityError::new(format!(
                    "unknown active sensor ID: {sensor_id}"
                )));
            }
        }

        let mut pairs = Vec::new();
        for (left_index, left) in self.worlds.iter().enumerate() {
            for right in self.worlds.iter().skip(left_index + 1) {
                if left.decision_id == right.decision_id {
                    continue;
                }

                let mut separating_sensor_ids = Vec::new();
                for sensor_id in &active {
                    let sensor = self
                        .sensors
                        .iter()
                        .find(|sensor| sensor.sensor_id == *sensor_id)
                        .expect("active sensor IDs validated above");
                    let left_value = observation_for(sensor, &left.world_id);
                    let right_value = observation_for(sensor, &right.world_id);

                    if let (SensorObservation::Known(left_value), SensorObservation::Known(right_value)) =
                        (left_value, right_value)
                    {
                        if left_value != right_value {
                            separating_sensor_ids.push((*sensor_id).to_owned());
                        }
                    }
                }

                pairs.push(CrossDecisionPairReport {
                    left_world_id: left.world_id.clone(),
                    right_world_id: right.world_id.clone(),
                    left_decision_id: left.decision_id.clone(),
                    right_decision_id: right.decision_id.clone(),
                    separating_sensor_ids,
                });
            }
        }

        let diagnosable = pairs.iter().all(CrossDecisionPairReport::is_separated);
        Ok(DiagnosabilityResult {
            diagnosable,
            cross_decision_pairs: pairs,
        })
    }

    /// Inclusion-minimal sufficient sensor sets by exact enumeration.
    pub fn minimal_sufficient_sensor_sets(
        &self,
    ) -> Result<Vec<SensorSet>, DiagnosabilityError> {
        self.validate()?;
        self.validate_exact_enumeration(self.sensors.len())?;
        let sensor_ids = self.sorted_sensor_ids();
        let mut minimal = Vec::<Vec<String>>::new();

        for size in 0..=sensor_ids.len() {
            for indices in combinations(sensor_ids.len(), size) {
                let candidate: Vec<String> =
                    indices.iter().map(|index| sensor_ids[*index].clone()).collect();
                if contains_subset(&minimal, &candidate) {
                    continue;
                }
                let active: Vec<&str> = candidate.iter().map(String::as_str).collect();
                if self.diagnose(&active)?.diagnosable {
                    minimal.push(candidate);
                }
            }
        }

        Ok(minimal
            .into_iter()
            .map(|sensor_ids| SensorSet { sensor_ids })
            .collect())
    }

    /// Inclusion-minimal sensor-loss cuts relative to a diagnosable baseline.
    pub fn minimal_sensor_cut_sets(
        &self,
        baseline_active_ids: &[&str],
    ) -> Result<Vec<SensorCutSet>, DiagnosabilityError> {
        let baseline = self.canonical_baseline(baseline_active_ids)?;
        self.validate_exact_enumeration(baseline.len())?;
        let baseline_refs: Vec<&str> = baseline.iter().map(String::as_str).collect();
        if !self.diagnose(&baseline_refs)?.diagnosable {
            return Err(DiagnosabilityError::new(
                "sensor-cut baseline must be diagnosable",
            ));
        }

        let mut minimal = Vec::<Vec<String>>::new();
        for size in 1..=baseline.len() {
            for indices in combinations(baseline.len(), size) {
                let removed: Vec<String> =
                    indices.iter().map(|index| baseline[*index].clone()).collect();
                if contains_subset(&minimal, &removed) {
                    continue;
                }
                let removed_set: BTreeSet<&str> =
                    removed.iter().map(String::as_str).collect();
                let active: Vec<&str> = baseline_refs
                    .iter()
                    .copied()
                    .filter(|sensor_id| !removed_set.contains(sensor_id))
                    .collect();
                if !self.diagnose(&active)?.diagnosable {
                    minimal.push(removed);
                }
            }
        }

        Ok(minimal
            .into_iter()
            .map(|lost_sensor_ids| SensorCutSet { lost_sensor_ids })
            .collect())
    }

    /// Inclusion-minimal common-mode group losses relative to a diagnosable
    /// baseline. Group independence is caller-declared and never inferred.
    pub fn minimal_common_mode_cut_sets(
        &self,
        baseline_active_ids: &[&str],
    ) -> Result<Vec<CommonModeCutSet>, DiagnosabilityError> {
        let baseline = self.canonical_baseline(baseline_active_ids)?;
        self.validate_exact_enumeration(baseline.len())?;
        let baseline_refs: Vec<&str> = baseline.iter().map(String::as_str).collect();
        if !self.diagnose(&baseline_refs)?.diagnosable {
            return Err(DiagnosabilityError::new(
                "common-mode cut baseline must be diagnosable",
            ));
        }

        let groups_by_sensor: BTreeMap<&str, &str> = self
            .sensors
            .iter()
            .map(|sensor| {
                (
                    sensor.sensor_id.as_str(),
                    sensor.common_mode_group.as_str(),
                )
            })
            .collect();
        let groups: Vec<String> = baseline
            .iter()
            .map(|sensor_id| groups_by_sensor[sensor_id.as_str()].to_owned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let mut minimal = Vec::<Vec<String>>::new();
        for size in 1..=groups.len() {
            for indices in combinations(groups.len(), size) {
                let removed: Vec<String> =
                    indices.iter().map(|index| groups[*index].clone()).collect();
                if contains_subset(&minimal, &removed) {
                    continue;
                }
                let removed_set: BTreeSet<&str> =
                    removed.iter().map(String::as_str).collect();
                let active: Vec<&str> = baseline_refs
                    .iter()
                    .copied()
                    .filter(|sensor_id| {
                        !removed_set.contains(groups_by_sensor[*sensor_id])
                    })
                    .collect();
                if !self.diagnose(&active)?.diagnosable {
                    minimal.push(removed);
                }
            }
        }

        Ok(minimal
            .into_iter()
            .map(|lost_common_mode_groups| CommonModeCutSet {
                lost_common_mode_groups,
            })
            .collect())
    }

    fn validate_exact_enumeration(
        &self,
        count: usize,
    ) -> Result<(), DiagnosabilityError> {
        if count > MAX_EXACT_SENSOR_COUNT {
            return Err(DiagnosabilityError::new(format!(
                "exact subset enumeration supports at most {MAX_EXACT_SENSOR_COUNT} items",
            )));
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

    fn canonical_baseline(
        &self,
        baseline_active_ids: &[&str],
    ) -> Result<Vec<String>, DiagnosabilityError> {
        self.validate()?;
        let mut baseline = BTreeSet::new();
        for sensor_id in baseline_active_ids {
            if sensor_id.trim().is_empty() {
                return Err(DiagnosabilityError::new(
                    "baseline sensor IDs must be non-empty",
                ));
            }
            if !baseline.insert(*sensor_id) {
                return Err(DiagnosabilityError::new(format!(
                    "duplicate baseline sensor ID: {sensor_id}"
                )));
            }
        }

        let known: BTreeSet<&str> =
            self.sensors.iter().map(|sensor| sensor.sensor_id.as_str()).collect();
        if let Some(unknown) = baseline.iter().find(|sensor_id| !known.contains(**sensor_id)) {
            return Err(DiagnosabilityError::new(format!(
                "baseline references unknown sensor: {unknown}",
            )));
        }
        Ok(baseline.into_iter().map(str::to_owned).collect())
    }
}

fn observation_for<'a>(
    sensor: &'a DiagnosticSensor,
    world_id: &str,
) -> &'a SensorObservation {
    sensor
        .outcomes
        .iter()
        .find(|outcome| outcome.world_id == world_id)
        .map(|outcome| &outcome.observation)
        .expect("validated sensor maps must cover every world")
}

fn contains_subset(existing: &[Vec<String>], candidate: &[String]) -> bool {
    let candidate: BTreeSet<&str> = candidate.iter().map(String::as_str).collect();
    existing.iter().any(|subset| {
        subset
            .iter()
            .all(|sensor_id| candidate.contains(sensor_id.as_str()))
    })
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn visit(
        n: usize,
        k: usize,
        next: usize,
        current: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        if current.len() == k {
            results.push(current.clone());
            return;
        }
        let remaining = k - current.len();
        if remaining > n.saturating_sub(next) {
            return;
        }
        let last_start = n - remaining;
        for index in next..=last_start {
            current.push(index);
            visit(n, k, index + 1, current, results);
            current.pop();
        }
    }

    if k > n {
        return Vec::new();
    }
    let mut results = Vec::new();
    visit(n, k, 0, &mut Vec::new(), &mut results);
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> DiagnosabilityModel {
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

        let sensor = |sensor_id: &str, group: &str, values: [&str; 4]| {
            DiagnosticSensor {
                sensor_id: sensor_id.to_owned(),
                common_mode_group: group.to_owned(),
                outcomes: worlds
                    .iter()
                    .zip(values)
                    .map(|(world, value)| WorldSensorObservation {
                        world_id: world.world_id.clone(),
                        observation: if value == "UNKNOWN" {
                            SensorObservation::Unknown
                        } else {
                            SensorObservation::Known(value.to_owned())
                        },
                    })
                    .collect(),
            }
        };

        DiagnosabilityModel {
            worlds,
            sensors: vec![
                sensor(
                    "gen_current",
                    "gen_electrical",
                    ["HEALTHY", "FAILED", "HEALTHY", "FAILED"],
                ),
                sensor(
                    "gen_vibration",
                    "gen_mechanical",
                    ["HEALTHY", "FAILED", "HEALTHY", "FAILED"],
                ),
                sensor("tie_aux", "switch_aux", ["CLOSED", "CLOSED", "OPEN", "OPEN"]),
                sensor("tie_optical", "optical", ["CLOSED", "CLOSED", "OPEN", "OPEN"]),
                sensor("failed_diag", "diag_bus", ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"]),
            ],
        }
    }

    #[test]
    fn full_sensor_baseline_is_decision_diagnosable() {
        let model = fixture();
        let all = model.sorted_sensor_ids();
        let result = model
            .diagnose(&all.iter().map(String::as_str).collect::<Vec<_>>())
            .expect("valid baseline");
        assert!(result.diagnosable);
        assert!(!result.cross_decision_pairs.is_empty());
        assert!(result.cross_decision_pairs.iter().all(|pair| pair.is_separated()));
    }

    #[test]
    fn exact_minimal_sufficient_sets_match_the_known_answer() {
        let model = fixture();
        let sets = model
            .minimal_sufficient_sensor_sets()
            .expect("exact small-set enumeration");
        let actual: BTreeSet<Vec<String>> =
            sets.into_iter().map(|set| set.sensor_ids).collect();
        let expected = [
            vec!["gen_current".to_owned(), "tie_aux".to_owned()],
            vec!["gen_current".to_owned(), "tie_optical".to_owned()],
            vec!["gen_vibration".to_owned(), "tie_aux".to_owned()],
            vec!["gen_vibration".to_owned(), "tie_optical".to_owned()],
        ]
        .into_iter()
        .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn unknown_never_separates_even_against_a_known_value() {
        let mut model = fixture();
        let sensor = model
            .sensors
            .iter_mut()
            .find(|sensor| sensor.sensor_id == "failed_diag")
            .expect("fixture sensor");
        sensor.outcomes[0].observation = SensorObservation::Unknown;
        sensor.outcomes[1].observation = SensorObservation::Known("FAILED".to_owned());

        let result = model
            .diagnose(&["failed_diag", "tie_aux"])
            .expect("valid unknown outcome");
        assert!(!result.diagnosable);
        assert!(!result.unresolved_pairs().is_empty());
    }

    #[test]
    fn an_always_unknown_channel_cannot_manufacture_certainty() {
        let model = fixture();
        let result = model
            .diagnose(&["failed_diag", "tie_aux"])
            .expect("valid sensor set");
        assert!(!result.diagnosable);
    }

    #[test]
    fn single_sensor_losses_preserve_the_redundant_baseline() {
        let model = fixture();
        let baseline = model.sorted_sensor_ids();
        for lost in ["gen_current", "gen_vibration", "tie_aux", "tie_optical"] {
            let active: Vec<&str> = baseline
                .iter()
                .map(String::as_str)
                .filter(|sensor_id| *sensor_id != lost)
                .collect();
            assert!(
                model.diagnose(&active).expect("single-loss model").diagnosable,
                "single loss {lost} should retain the decision distinction"
            );
        }
    }

    #[test]
    fn exact_minimal_sensor_cuts_match_the_known_answer() {
        let model = fixture();
        let baseline = model.sorted_sensor_ids();
        let cuts = model
            .minimal_sensor_cut_sets(&baseline.iter().map(String::as_str).collect::<Vec<_>>())
            .expect("diagnosable baseline");
        let actual: BTreeSet<Vec<String>> =
            cuts.into_iter().map(|cut| cut.lost_sensor_ids).collect();
        let expected = [
            vec!["gen_current".to_owned(), "gen_vibration".to_owned()],
            vec!["tie_aux".to_owned(), "tie_optical".to_owned()],
        ]
        .into_iter()
        .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn common_mode_cut_sets_respect_declared_fault_domains() {
        let model = fixture();
        let baseline = model.sorted_sensor_ids();
        let cuts = model
            .minimal_common_mode_cut_sets(&baseline.iter().map(String::as_str).collect::<Vec<_>>())
            .expect("diagnosable baseline");
        let actual: BTreeSet<Vec<String>> = cuts
            .into_iter()
            .map(|cut| cut.lost_common_mode_groups)
            .collect();
        let expected = [
            vec!["gen_electrical".to_owned(), "gen_mechanical".to_owned()],
            vec!["optical".to_owned(), "switch_aux".to_owned()],
        ]
        .into_iter()
        .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn malformed_sensor_models_and_baselines_fail_closed() {
        let mut duplicate_sensor = fixture();
        duplicate_sensor.sensors.push(duplicate_sensor.sensors[0].clone());
        assert!(duplicate_sensor.validate().is_err());

        let mut incomplete = fixture();
        incomplete.sensors[0].outcomes.pop();
        assert!(incomplete.validate().is_err());

        let mut duplicate_world = fixture();
        duplicate_world.worlds[1].world_id = duplicate_world.worlds[0].world_id.clone();
        assert!(duplicate_world.validate().is_err());

        let model = fixture();
        assert!(model.diagnose(&["not-a-sensor"]).is_err());
        assert!(model.diagnose(&["tie_aux", "tie_aux"]).is_err());
        assert!(
            model
                .minimal_sensor_cut_sets(&["failed_diag"])
                .is_err(),
            "an undiagnosable baseline must not produce cut sets"
        );
    }

    #[test]
    fn diagnosis_scales_beyond_the_exact_subset_search_bound() {
        let mut model = fixture();
        let template = model.sensors[0].clone();
        let additional = MAX_EXACT_SENSOR_COUNT + 1 - model.sensors.len();
        for index in 0..additional {
            let mut extra = template.clone();
            extra.sensor_id = format!("extra-{index:02}");
            extra.common_mode_group = format!("extra-group-{index:02}");
            model.sensors.push(extra);
        }
        let active: Vec<&str> = model
            .sensors
            .iter()
            .map(|sensor| sensor.sensor_id.as_str())
            .collect();
        assert!(model.diagnose(&active).expect("diagnose large set").diagnosable);
        assert!(model.minimal_sufficient_sensor_sets().is_err());
    }

    #[test]
    fn losing_information_cannot_improve_diagnosability() {
        let model = fixture();
        let all = model.sorted_sensor_ids();
        let full = model
            .diagnose(&all.iter().map(String::as_str).collect::<Vec<_>>())
            .expect("full portfolio");
        for size in 0..all.len() {
            for subset in combinations(all.len(), size) {
                let active: Vec<&str> = subset.iter().map(|index| all[*index].as_str()).collect();
                let reduced = model.diagnose(&active).expect("reduced portfolio");
                if reduced.diagnosable {
                    assert!(full.diagnosable);
                }
            }
        }
    }
}

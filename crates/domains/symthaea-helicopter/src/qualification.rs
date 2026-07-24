// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic flight-qualification campaign evaluation.
//!
//! Qualification is expressed as versioned scenarios, required seed counts,
//! required exercised faults, explicit metric gates, and verified flight-log
//! evidence. Missing evidence is `Incomplete`, never silently converted into a
//! passing zero. Known gate violations remain `Fail` even if other evidence is
//! incomplete.

use serde::{Deserialize, Serialize};

use crate::fault_monitor::HelicopterFaultKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QualificationMetric {
    HoverAltitudeDriftM,
    MaximumAttitudeErrorDeg,
    RecoveryTimeS,
    MaximumPositionErrorM,
    LandingVerticalSpeedMps,
    LandingHorizontalSpeedMps,
    MinimumRotorEnergyMarginJ,
    NavigationRejectedUpdates,
    CriticalFaultCount,
    FuelReserveMarginKg,
    MaximumControlLoopJitterUs,
    MissedControlDeadlines,
    MaximumSensorToActuatorLatencyMs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationDirection {
    AtMost,
    AtLeast,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationGate {
    pub gate_id: String,
    pub metric: QualificationMetric,
    pub direction: QualificationDirection,
    pub threshold: f64,
}

impl QualificationGate {
    fn validate(&self) -> bool {
        !self.gate_id.trim().is_empty() && self.threshold.is_finite()
    }

    fn passes(&self, value: f64) -> bool {
        match self.direction {
            QualificationDirection::AtMost => value <= self.threshold,
            QualificationDirection::AtLeast => value >= self.threshold,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationScenario {
    pub scenario_id: String,
    pub required_seeds: usize,
    pub required_faults: Vec<HelicopterFaultKind>,
    pub gates: Vec<QualificationGate>,
}

impl QualificationScenario {
    fn validate(&self) -> bool {
        !self.scenario_id.trim().is_empty()
            && self.required_seeds > 0
            && !self.gates.is_empty()
            && self.gates.iter().all(QualificationGate::validate)
            && unique_strings(self.gates.iter().map(|gate| gate.gate_id.as_str()))
            && unique_faults(&self.required_faults)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QualificationMetricValue {
    pub metric: QualificationMetric,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationObservation {
    pub scenario_id: String,
    pub seed: u64,
    pub completed: bool,
    pub exercised_faults: Vec<HelicopterFaultKind>,
    pub metrics: Vec<QualificationMetricValue>,
    pub flight_log_digest: Option<String>,
    pub evidence_chain_verified: bool,
}

impl QualificationObservation {
    fn validate(&self) -> bool {
        !self.scenario_id.trim().is_empty()
            && self.metrics.iter().all(|metric| metric.value.is_finite())
            && unique_metrics(&self.metrics)
            && unique_faults(&self.exercised_faults)
            && self
                .flight_log_digest
                .as_ref()
                .is_none_or(|digest| !digest.trim().is_empty())
    }

    fn metric(&self, metric: QualificationMetric) -> Option<f64> {
        self.metrics
            .iter()
            .find(|value| value.metric == metric)
            .map(|value| value.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualificationStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationGateResult {
    pub gate_id: String,
    pub metric: QualificationMetric,
    pub direction: QualificationDirection,
    pub threshold: f64,
    pub status: QualificationStatus,
    /// Maximum value for `AtMost`, minimum value for `AtLeast`.
    pub limiting_value: Option<f64>,
    pub failed_seeds: Vec<u64>,
    pub missing_metric_seeds: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationScenarioReport {
    pub scenario_id: String,
    pub status: QualificationStatus,
    pub distinct_seed_count: usize,
    pub required_seed_count: usize,
    pub duplicate_seed_values: Vec<u64>,
    pub missing_required_faults: Vec<HelicopterFaultKind>,
    pub incomplete_seeds: Vec<u64>,
    pub unverified_evidence_seeds: Vec<u64>,
    pub gate_results: Vec<QualificationGateResult>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualificationReport {
    pub schema_version: String,
    pub status: QualificationStatus,
    pub scenarios: Vec<QualificationScenarioReport>,
}

impl QualificationReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, QualificationError> {
        serde_json::to_vec(self).map_err(|_| QualificationError::SerializationFailed)
    }

    pub fn digest_fnv1a64(&self) -> Result<String, QualificationError> {
        let bytes = self.canonical_json()?;
        let mut hash = 0xcbf29ce484222325_u64;
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        Ok(format!("fnv1a64:{hash:016x}"))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationError {
    InvalidScenario,
    DuplicateScenario,
    InvalidObservation,
    UnknownScenario,
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct QualificationCampaign {
    schema_version: String,
    scenarios: Vec<QualificationScenario>,
}

impl QualificationCampaign {
    pub fn new(
        schema_version: impl Into<String>,
        scenarios: Vec<QualificationScenario>,
    ) -> Result<Self, QualificationError> {
        let schema_version = schema_version.into();
        if schema_version.trim().is_empty()
            || scenarios.is_empty()
            || scenarios.iter().any(|scenario| !scenario.validate())
        {
            return Err(QualificationError::InvalidScenario);
        }
        if !unique_strings(
            scenarios
                .iter()
                .map(|scenario| scenario.scenario_id.as_str()),
        ) {
            return Err(QualificationError::DuplicateScenario);
        }
        Ok(Self {
            schema_version,
            scenarios,
        })
    }

    pub fn evaluate(
        &self,
        observations: &[QualificationObservation],
    ) -> Result<QualificationReport, QualificationError> {
        if observations
            .iter()
            .any(|observation| !observation.validate())
        {
            return Err(QualificationError::InvalidObservation);
        }
        if observations.iter().any(|observation| {
            !self
                .scenarios
                .iter()
                .any(|scenario| scenario.scenario_id == observation.scenario_id)
        }) {
            return Err(QualificationError::UnknownScenario);
        }

        let mut reports = Vec::with_capacity(self.scenarios.len());
        for scenario in &self.scenarios {
            let mut selected: Vec<_> = observations
                .iter()
                .filter(|observation| observation.scenario_id == scenario.scenario_id)
                .collect();
            selected.sort_by_key(|observation| observation.seed);

            let mut seeds: Vec<_> = selected
                .iter()
                .map(|observation| observation.seed)
                .collect();
            seeds.sort_unstable();
            seeds.dedup();
            let mut duplicate_seed_values = Vec::new();
            for window in selected.windows(2) {
                if window[0].seed == window[1].seed
                    && duplicate_seed_values.last().copied() != Some(window[0].seed)
                {
                    duplicate_seed_values.push(window[0].seed);
                }
            }

            let mut exercised_faults: Vec<_> = selected
                .iter()
                .flat_map(|observation| observation.exercised_faults.iter().copied())
                .collect();
            exercised_faults.sort_by_key(|fault| *fault as u8);
            exercised_faults.dedup();
            let missing_required_faults: Vec<_> = scenario
                .required_faults
                .iter()
                .copied()
                .filter(|fault| !exercised_faults.contains(fault))
                .collect();
            let incomplete_seeds: Vec<_> = selected
                .iter()
                .filter(|observation| !observation.completed)
                .map(|observation| observation.seed)
                .collect();
            let unverified_evidence_seeds: Vec<_> = selected
                .iter()
                .filter(|observation| {
                    !observation.evidence_chain_verified || observation.flight_log_digest.is_none()
                })
                .map(|observation| observation.seed)
                .collect();

            let mut gate_results = Vec::with_capacity(scenario.gates.len());
            for gate in &scenario.gates {
                let mut values = Vec::new();
                let mut failed_seeds = Vec::new();
                let mut missing_metric_seeds = Vec::new();
                for observation in &selected {
                    match observation.metric(gate.metric) {
                        Some(value) => {
                            values.push(value);
                            if !gate.passes(value) {
                                failed_seeds.push(observation.seed);
                            }
                        }
                        None => missing_metric_seeds.push(observation.seed),
                    }
                }
                let limiting_value = match gate.direction {
                    QualificationDirection::AtMost => values.into_iter().reduce(f64::max),
                    QualificationDirection::AtLeast => values.into_iter().reduce(f64::min),
                };
                let status = if !failed_seeds.is_empty() {
                    QualificationStatus::Fail
                } else if missing_metric_seeds.len() > 0 || selected.is_empty() {
                    QualificationStatus::Incomplete
                } else {
                    QualificationStatus::Pass
                };
                gate_results.push(QualificationGateResult {
                    gate_id: gate.gate_id.clone(),
                    metric: gate.metric,
                    direction: gate.direction,
                    threshold: gate.threshold,
                    status,
                    limiting_value,
                    failed_seeds,
                    missing_metric_seeds,
                });
            }

            let known_failure = gate_results
                .iter()
                .any(|result| result.status == QualificationStatus::Fail);
            let missing_evidence = !duplicate_seed_values.is_empty()
                || seeds.len() < scenario.required_seeds
                || !missing_required_faults.is_empty()
                || !incomplete_seeds.is_empty()
                || !unverified_evidence_seeds.is_empty()
                || gate_results
                    .iter()
                    .any(|result| result.status == QualificationStatus::Incomplete);
            let status = if known_failure {
                QualificationStatus::Fail
            } else if missing_evidence {
                QualificationStatus::Incomplete
            } else {
                QualificationStatus::Pass
            };
            reports.push(QualificationScenarioReport {
                scenario_id: scenario.scenario_id.clone(),
                status,
                distinct_seed_count: seeds.len(),
                required_seed_count: scenario.required_seeds,
                duplicate_seed_values,
                missing_required_faults,
                incomplete_seeds,
                unverified_evidence_seeds,
                gate_results,
            });
        }

        let status = if reports
            .iter()
            .any(|report| report.status == QualificationStatus::Fail)
        {
            QualificationStatus::Fail
        } else if reports
            .iter()
            .any(|report| report.status == QualificationStatus::Incomplete)
        {
            QualificationStatus::Incomplete
        } else {
            QualificationStatus::Pass
        };
        Ok(QualificationReport {
            schema_version: self.schema_version.clone(),
            status,
            scenarios: reports,
        })
    }
}

fn unique_strings<'a>(values: impl Iterator<Item = &'a str>) -> bool {
    let mut seen = Vec::new();
    for value in values {
        if seen.contains(&value) {
            return false;
        }
        seen.push(value);
    }
    true
}

fn unique_faults(values: &[HelicopterFaultKind]) -> bool {
    values
        .iter()
        .enumerate()
        .all(|(index, value)| !values[..index].contains(value))
}

fn unique_metrics(values: &[QualificationMetricValue]) -> bool {
    values.iter().enumerate().all(|(index, value)| {
        !values[..index]
            .iter()
            .any(|previous| previous.metric == value.metric)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn campaign(required_seeds: usize) -> QualificationCampaign {
        QualificationCampaign::new(
            "symthaea-helicopter-qualification-v1",
            vec![QualificationScenario {
                scenario_id: "tail-loss-landing".to_string(),
                required_seeds,
                required_faults: vec![HelicopterFaultKind::TailRotorUnderSpeed],
                gates: vec![
                    QualificationGate {
                        gate_id: "landing-vertical-speed".to_string(),
                        metric: QualificationMetric::LandingVerticalSpeedMps,
                        direction: QualificationDirection::AtMost,
                        threshold: 4.0,
                    },
                    QualificationGate {
                        gate_id: "rotor-energy-margin".to_string(),
                        metric: QualificationMetric::MinimumRotorEnergyMarginJ,
                        direction: QualificationDirection::AtLeast,
                        threshold: 0.0,
                    },
                ],
            }],
        )
        .unwrap()
    }

    fn observation(seed: u64, vertical_speed: f64) -> QualificationObservation {
        QualificationObservation {
            scenario_id: "tail-loss-landing".to_string(),
            seed,
            completed: true,
            exercised_faults: vec![HelicopterFaultKind::TailRotorUnderSpeed],
            metrics: vec![
                QualificationMetricValue {
                    metric: QualificationMetric::LandingVerticalSpeedMps,
                    value: vertical_speed,
                },
                QualificationMetricValue {
                    metric: QualificationMetric::MinimumRotorEnergyMarginJ,
                    value: 10_000.0,
                },
            ],
            flight_log_digest: Some(format!("fnv1a64:{seed:016x}")),
            evidence_chain_verified: true,
        }
    }

    #[test]
    fn complete_passing_campaign_passes() {
        let observations = vec![observation(1, 2.0), observation(2, 3.0)];
        let report = campaign(2).evaluate(&observations).unwrap();
        assert_eq!(report.status, QualificationStatus::Pass);
        assert!(
            report.scenarios[0]
                .gate_results
                .iter()
                .all(|gate| gate.status == QualificationStatus::Pass)
        );
    }

    #[test]
    fn missing_seed_is_incomplete_not_pass() {
        let report = campaign(2).evaluate(&[observation(1, 2.0)]).unwrap();
        assert_eq!(report.status, QualificationStatus::Incomplete);
    }

    #[test]
    fn known_gate_violation_is_fail_even_with_missing_seed() {
        let report = campaign(2).evaluate(&[observation(1, 8.0)]).unwrap();
        assert_eq!(report.status, QualificationStatus::Fail);
        assert_eq!(report.scenarios[0].gate_results[0].failed_seeds, vec![1]);
    }

    #[test]
    fn missing_fault_coverage_is_incomplete() {
        let mut observation = observation(1, 2.0);
        observation.exercised_faults.clear();
        let report = campaign(1).evaluate(&[observation]).unwrap();
        assert_eq!(report.status, QualificationStatus::Incomplete);
        assert_eq!(
            report.scenarios[0].missing_required_faults,
            vec![HelicopterFaultKind::TailRotorUnderSpeed]
        );
    }

    #[test]
    fn duplicate_seed_is_incomplete() {
        let report = campaign(1)
            .evaluate(&[observation(1, 2.0), observation(1, 2.0)])
            .unwrap();
        assert_eq!(report.status, QualificationStatus::Incomplete);
        assert_eq!(report.scenarios[0].duplicate_seed_values, vec![1]);
    }

    #[test]
    fn report_digest_is_deterministic() {
        let observations = vec![observation(2, 3.0), observation(1, 2.0)];
        let a = campaign(2).evaluate(&observations).unwrap();
        let b = campaign(2).evaluate(&observations).unwrap();
        assert_eq!(a.canonical_json().unwrap(), b.canonical_json().unwrap());
        assert_eq!(a.digest_fnv1a64().unwrap(), b.digest_fnv1a64().unwrap());
    }
}

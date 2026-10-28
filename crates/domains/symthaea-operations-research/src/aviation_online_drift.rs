// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Anytime-valid sequential monitoring of categorical scenario drift.
//!
//! This monitor evaluates a frozen nominal categorical distribution against a
//! predeclared mixture of alternative distributions. The mixture likelihood ratio
//! is an e-process under independent observations from the exact nominal
//! distribution. Crossing 1/alpha has an anytime-valid false-alarm bound under
//! that null. This assumption is not a guarantee under serially dependent airline
//! events, a misspecified scenario taxonomy, or nominal probabilities estimated
//! from the same observations.
//!
//! The frozen reference is never updated by observed outcomes. An outside-support
//! event triggers a separate latched support-shift alarm. A detected alarm is a
//! request to investigate/recalibrate/abstain, not permission to auto-change the
//! radius or take an operational action.

use std::collections::{BTreeMap, BTreeSet};

/// Probability scale in integer parts per million.
pub const PROBABILITY_SCALE_PPM: u32 = 1_000_000;

/// One frozen category probability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftScenarioProbability {
    pub scenario_id: String,
    pub probability_ppm: u32,
}

/// A predeclared alternative distribution in the likelihood-ratio mixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftAlternative {
    pub alternative_id: String,
    pub mixture_weight_ppm: u32,
    pub probabilities: Vec<DriftScenarioProbability>,
}

/// Stable monitor status. Alarm states are latched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftMonitorStatus {
    InControl,
    DriftAlarm,
    SupportShiftAlarm,
}

/// Evidence type supporting the current status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftEvidence {
    None,
    MixtureLikelihoodRatio,
    ScenarioOutsideNominalSupport,
}

/// Invalid input or illegal monitor transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriftMonitorError {
    EmptyNominalDistribution,
    EmptyAlternativeSet,
    EmptyScenarioId,
    EmptyAlternativeId,
    DuplicateScenarioId { scenario_id: String },
    DuplicateAlternativeId { alternative_id: String },
    InvalidProbability { owner_id: String, probability_ppm: u32 },
    ProbabilityMassMismatch { owner_id: String, actual_ppm: u64 },
    AlternativeSupportMismatch { alternative_id: String, scenario_id: String },
    AlternativeMixtureHasZeroSupport { scenario_id: String },
    AlternativeHasZeroProbabilityForNominalSupport { alternative_id: String, scenario_id: String },
    MixtureWeightMassMismatch { actual_ppm: u64 },
    InvalidAlpha { alpha_ppm: u32 },
    EmptyEventId,
    DuplicateEventId { event_id: String },
    NonMonotoneEventTime { previous_epoch_seconds: i64, next_epoch_seconds: i64 },
    AlreadyAlarmed { status: DriftMonitorStatus },
    NumericalFailure,
}

/// Machine-readable result after one observation.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftMonitorReport {
    pub observations_seen: u64,
    pub status: DriftMonitorStatus,
    /// Log of the mixture likelihood-ratio e-value. The corresponding e-value
    /// is exp(log_e_value); log space avoids overflow for long streams.
    pub log_e_value: f64,
    /// Natural-log threshold ln(1/alpha), fixed at monitor creation.
    pub log_alarm_threshold: f64,
    pub evidence: DriftEvidence,
    pub last_event_id: String,
    pub last_scenario_id: String,
    pub last_event_time_epoch_seconds: i64,
}

/// Sequential monitor for a frozen nominal distribution and fixed alternatives.
#[derive(Debug, Clone)]
pub struct OnlineDriftMonitor {
    nominal: BTreeMap<String, u32>,
    alternatives: Vec<ValidatedAlternative>,
    alpha_ppm: u32,
    log_alarm_threshold: f64,
    log_likelihood_ratios: Vec<f64>,
    seen_event_ids: BTreeSet<String>,
    observations_seen: u64,
    last_event_time_epoch_seconds: Option<i64>,
    status: DriftMonitorStatus,
    evidence: DriftEvidence,
    last_event_id: String,
    last_scenario_id: String,
}

#[derive(Debug, Clone)]
struct ValidatedAlternative {
    alternative_id: String,
    mixture_weight_ppm: u32,
    probabilities: BTreeMap<String, u32>,
}

impl OnlineDriftMonitor {
    /// Create a monitor. Nominal probabilities, alternative distributions and
    /// mixture weights are frozen for the lifetime of this instance.
    pub fn new(
        nominal: Vec<DriftScenarioProbability>,
        alternatives: Vec<DriftAlternative>,
        alpha_ppm: u32,
    ) -> Result<Self, DriftMonitorError> {
        if nominal.is_empty() {
            return Err(DriftMonitorError::EmptyNominalDistribution);
        }
        if alternatives.is_empty() {
            return Err(DriftMonitorError::EmptyAlternativeSet);
        }
        if alpha_ppm == 0 || alpha_ppm >= PROBABILITY_SCALE_PPM {
            return Err(DriftMonitorError::InvalidAlpha { alpha_ppm });
        }

        let nominal = validate_distribution("nominal", &nominal)?;
        let nominal_sum: u64 = nominal.values().map(|p| u64::from(*p)).sum();
        if nominal_sum != u64::from(PROBABILITY_SCALE_PPM) {
            return Err(DriftMonitorError::ProbabilityMassMismatch {
                owner_id: "nominal".into(),
                actual_ppm: nominal_sum,
            });
        }

        let mut alternative_ids = BTreeSet::new();
        let mut validated = Vec::with_capacity(alternatives.len());
        let mut weight_sum = 0_u64;
        for alternative in alternatives {
            if alternative.alternative_id.trim().is_empty() {
                return Err(DriftMonitorError::EmptyAlternativeId);
            }
            if !alternative_ids.insert(alternative.alternative_id.clone()) {
                return Err(DriftMonitorError::DuplicateAlternativeId {
                    alternative_id: alternative.alternative_id,
                });
            }
            weight_sum += u64::from(alternative.mixture_weight_ppm);
            let probabilities =
                validate_distribution(&alternative.alternative_id, &alternative.probabilities)?;
            for scenario_id in nominal.keys() {
                if !probabilities.contains_key(scenario_id) {
                    return Err(DriftMonitorError::AlternativeSupportMismatch {
                        alternative_id: alternative.alternative_id.clone(),
                        scenario_id: scenario_id.clone(),
                    });
                }
            }
            for scenario_id in probabilities.keys() {
                if !nominal.contains_key(scenario_id) {
                    return Err(DriftMonitorError::AlternativeSupportMismatch {
                        alternative_id: alternative.alternative_id.clone(),
                        scenario_id: scenario_id.clone(),
                    });
                }
            }
            validated.push(ValidatedAlternative {
                alternative_id: alternative.alternative_id,
                mixture_weight_ppm: alternative.mixture_weight_ppm,
                probabilities,
            });
        }
        if weight_sum != u64::from(PROBABILITY_SCALE_PPM) {
            return Err(DriftMonitorError::MixtureWeightMassMismatch {
                actual_ppm: weight_sum,
            });
        }

        validated.sort_by(|a, b| a.alternative_id.cmp(&b.alternative_id));
        let log_alarm_threshold =
            (f64::from(PROBABILITY_SCALE_PPM) / f64::from(alpha_ppm)).ln();
        let monitor = Self {
            nominal,
            log_likelihood_ratios: vec![0.0; validated.len()],
            alternatives: validated,
            alpha_ppm,
            log_alarm_threshold,
            seen_event_ids: BTreeSet::new(),
            observations_seen: 0,
            last_event_time_epoch_seconds: None,
            status: DriftMonitorStatus::InControl,
            evidence: DriftEvidence::None,
            last_event_id: String::new(),
            last_scenario_id: String::new(),
        };
        debug_assert!(monitor.alpha_ppm > 0);
        Ok(monitor)
    }

    /// Ingest one observed event. Event IDs must be unique and timestamps
    /// non-decreasing. An input error does not partially update monitor state.
    pub fn observe(
        &mut self,
        event_id: &str,
        event_time_epoch_seconds: i64,
        scenario_id: &str,
    ) -> Result<DriftMonitorReport, DriftMonitorError> {
        if self.status != DriftMonitorStatus::InControl {
            return Err(DriftMonitorError::AlreadyAlarmed { status: self.status });
        }
        if event_id.trim().is_empty() {
            return Err(DriftMonitorError::EmptyEventId);
        }
        if self.observations_seen == u64::MAX {
            return Err(DriftMonitorError::NumericalFailure);
        }
        if self.seen_event_ids.contains(event_id) {
            return Err(DriftMonitorError::DuplicateEventId {
                event_id: event_id.into(),
            });
        }
        if let Some(previous) = self.last_event_time_epoch_seconds {
            if event_time_epoch_seconds < previous {
                return Err(DriftMonitorError::NonMonotoneEventTime {
                    previous_epoch_seconds: previous,
                    next_epoch_seconds: event_time_epoch_seconds,
                });
            }
        }

        let nominal_probability = self.nominal.get(scenario_id).copied();
        let Some(p_ppm) = nominal_probability else {
            self.commit_event(event_id, event_time_epoch_seconds, scenario_id);
            self.status = DriftMonitorStatus::SupportShiftAlarm;
            self.evidence = DriftEvidence::ScenarioOutsideNominalSupport;
            return Ok(self.report());
        };
        if p_ppm == 0 {
            self.commit_event(event_id, event_time_epoch_seconds, scenario_id);
            self.status = DriftMonitorStatus::SupportShiftAlarm;
            self.evidence = DriftEvidence::ScenarioOutsideNominalSupport;
            return Ok(self.report());
        }

        let mut next_log_ratios = self.log_likelihood_ratios.clone();
        for (index, alternative) in self.alternatives.iter().enumerate() {
            if !next_log_ratios[index].is_finite() {
                continue;
            }
            let q_ppm = alternative.probabilities[scenario_id];
            if q_ppm == 0 {
                next_log_ratios[index] = f64::NEG_INFINITY;
            } else {
                let increment = (f64::from(q_ppm) / f64::from(p_ppm)).ln();
                let next = next_log_ratios[index] + increment;
                if !next.is_finite() {
                    return Err(DriftMonitorError::NumericalFailure);
                }
                next_log_ratios[index] = next;
            }
        }
        let log_e_value = mixture_log_e_value(&self.alternatives, &next_log_ratios);
        if log_e_value.is_nan() || log_e_value == f64::INFINITY {
            return Err(DriftMonitorError::NumericalFailure);
        }

        // Commit state only after the numeric update has passed validation.
        self.commit_event(event_id, event_time_epoch_seconds, scenario_id);
        self.log_likelihood_ratios = next_log_ratios;
        if log_e_value >= self.log_alarm_threshold {
            self.status = DriftMonitorStatus::DriftAlarm;
            self.evidence = DriftEvidence::MixtureLikelihoodRatio;
        }
        Ok(self.report())
    }

    fn commit_event(&mut self, event_id: &str, time: i64, scenario_id: &str) {
        self.seen_event_ids.insert(event_id.into());
        self.observations_seen += 1;
        self.last_event_time_epoch_seconds = Some(time);
        self.last_event_id = event_id.into();
        self.last_scenario_id = scenario_id.into();
    }

    /// Last report, useful for logs after an alarm.
    pub fn report(&self) -> DriftMonitorReport {
        let log_e_value = mixture_log_e_value(&self.alternatives, &self.log_likelihood_ratios);
        DriftMonitorReport {
            observations_seen: self.observations_seen,
            status: self.status,
            log_e_value,
            log_alarm_threshold: self.log_alarm_threshold,
            evidence: self.evidence,
            last_event_id: self.last_event_id.clone(),
            last_scenario_id: self.last_scenario_id.clone(),
            last_event_time_epoch_seconds: self.last_event_time_epoch_seconds.unwrap_or_default(),
        }
    }

    /// Fixed nominal probabilities in stable scenario-ID order.
    pub fn nominal_probabilities(&self) -> &BTreeMap<String, u32> {
        &self.nominal
    }

    pub fn alpha_ppm(&self) -> u32 {
        self.alpha_ppm
    }

    pub fn observations_seen(&self) -> u64 {
        self.observations_seen
    }
}

fn validate_distribution(
    owner_id: &str,
    distribution: &[DriftScenarioProbability],
) -> Result<BTreeMap<String, u32>, DriftMonitorError> {
    let mut result = BTreeMap::new();
    let mut sum = 0_u64;
    for item in distribution {
        if item.scenario_id.trim().is_empty() {
            return Err(DriftMonitorError::EmptyScenarioId);
        }
        if item.probability_ppm > PROBABILITY_SCALE_PPM {
            return Err(DriftMonitorError::InvalidProbability {
                owner_id: owner_id.into(),
                probability_ppm: item.probability_ppm,
            });
        }
        if result
            .insert(item.scenario_id.clone(), item.probability_ppm)
            .is_some()
        {
            return Err(DriftMonitorError::DuplicateScenarioId {
                scenario_id: item.scenario_id.clone(),
            });
        }
        sum += u64::from(item.probability_ppm);
    }
    if sum != u64::from(PROBABILITY_SCALE_PPM) {
        return Err(DriftMonitorError::ProbabilityMassMismatch {
            owner_id: owner_id.into(),
            actual_ppm: sum,
        });
    }
    Ok(result)
}

fn mixture_log_e_value(
    alternatives: &[ValidatedAlternative],
    log_likelihood_ratios: &[f64],
) -> f64 {
    let terms: Vec<f64> = alternatives
        .iter()
        .zip(log_likelihood_ratios.iter())
        .filter_map(|(alternative, log_lr)| {
            if alternative.mixture_weight_ppm == 0 || *log_lr == f64::NEG_INFINITY {
                None
            } else {
                Some(
                    (f64::from(alternative.mixture_weight_ppm)
                        / f64::from(PROBABILITY_SCALE_PPM))
                        .ln()
                        + *log_lr,
                )
            }
        })
        .collect();
    if terms.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max + terms.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prob(id: &str, ppm: u32) -> DriftScenarioProbability {
        DriftScenarioProbability {
            scenario_id: id.into(),
            probability_ppm: ppm,
        }
    }

    fn monitor(alpha_ppm: u32) -> OnlineDriftMonitor {
        OnlineDriftMonitor::new(
            vec![prob("routine", 900_000), prob("disruption", 100_000)],
            vec![DriftAlternative {
                alternative_id: "disruption-heavy".into(),
                mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("routine", 500_000), prob("disruption", 500_000)],
            }],
            alpha_ppm,
        )
        .unwrap()
    }

    #[test]
    fn starts_at_unit_e_value_and_in_control() {
        let monitor = monitor(50_000);
        assert_eq!(monitor.report().status, DriftMonitorStatus::InControl);
        assert!(monitor.report().log_e_value.abs() < 1e-12);
        assert_eq!(monitor.observations_seen(), 0);
    }

    #[test]
    fn one_expected_rare_event_does_not_cross_five_percent_boundary() {
        let mut monitor = monitor(50_000);
        let report = monitor.observe("e1", 10, "disruption").unwrap();
        assert_eq!(report.status, DriftMonitorStatus::InControl);
        assert!((report.log_e_value.exp() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn sequential_likelihood_ratio_crossing_latches_drift_alarm() {
        let mut monitor = monitor(50_000);
        monitor.observe("e1", 10, "disruption").unwrap();
        let report = monitor.observe("e2", 11, "disruption").unwrap();
        assert_eq!(report.status, DriftMonitorStatus::DriftAlarm);
        assert_eq!(report.evidence, DriftEvidence::MixtureLikelihoodRatio);
        assert!(report.log_e_value >= report.log_alarm_threshold);
    }

    #[test]
    fn stricter_false_alarm_budget_requires_more_evidence() {
        let mut monitor = monitor(10_000);
        for i in 0..2 {
            let report = monitor.observe(&format!("e{i}"), i as i64, "disruption").unwrap();
            assert_eq!(report.status, DriftMonitorStatus::InControl);
        }
        let report = monitor.observe("e2", 2, "disruption").unwrap();
        assert_eq!(report.status, DriftMonitorStatus::DriftAlarm);
    }

    #[test]
    fn repeated_nominal_routine_events_do_not_increase_e_value() {
        let mut monitor = monitor(50_000);
        for i in 0..100 {
            let report = monitor.observe(&format!("routine-{i}"), i, "routine").unwrap();
            assert_eq!(report.status, DriftMonitorStatus::InControl);
        }
        assert!(monitor.report().log_e_value.exp() < 1.0);
    }

    #[test]
    fn unknown_scenario_causes_immediate_support_shift_alarm() {
        let mut monitor = monitor(50_000);
        let report = monitor.observe("e1", 10, "unmodeled-compound-event").unwrap();
        assert_eq!(report.status, DriftMonitorStatus::SupportShiftAlarm);
        assert_eq!(report.evidence, DriftEvidence::ScenarioOutsideNominalSupport);
    }

    #[test]
    fn zero_probability_nominal_atom_causes_support_shift_alarm() {
        let mut monitor = OnlineDriftMonitor::new(
            vec![prob("routine", 1_000_000), prob("known-zero-stress", 0)],
            vec![DriftAlternative {
                alternative_id: "alternative".into(),
                mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("routine", 900_000), prob("known-zero-stress", 100_000)],
            }],
            50_000,
        )
        .unwrap();
        let report = monitor.observe("e1", 10, "known-zero-stress").unwrap();
        assert_eq!(report.status, DriftMonitorStatus::SupportShiftAlarm);
    }

    #[test]
    fn duplicate_event_is_rejected_without_state_change() {
        let mut monitor = monitor(50_000);
        monitor.observe("same", 10, "routine").unwrap();
        let before = monitor.report();
        assert_eq!(
            monitor.observe("same", 11, "disruption"),
            Err(DriftMonitorError::DuplicateEventId { event_id: "same".into() })
        );
        assert_eq!(monitor.report(), before);
    }

    #[test]
    fn decreasing_timestamps_are_rejected_without_state_change() {
        let mut monitor = monitor(50_000);
        monitor.observe("e1", 100, "routine").unwrap();
        let before = monitor.report();
        assert!(matches!(
            monitor.observe("e2", 99, "disruption"),
            Err(DriftMonitorError::NonMonotoneEventTime { .. })
        ));
        assert_eq!(monitor.report(), before);
    }

    #[test]
    fn alarm_is_latched_until_a_new_monitor_is_constructed() {
        let mut monitor = monitor(50_000);
        monitor.observe("e1", 10, "disruption").unwrap();
        monitor.observe("e2", 11, "disruption").unwrap();
        assert!(matches!(
            monitor.observe("e3", 12, "routine"),
            Err(DriftMonitorError::AlreadyAlarmed { status: DriftMonitorStatus::DriftAlarm })
        ));
    }

    #[test]
    fn rejects_invalid_alpha_and_probabilities() {
        for alpha in [0, 1_000_000] {
            assert!(matches!(
                OnlineDriftMonitor::new(
                    vec![prob("routine", 900_000), prob("disruption", 100_000)],
                    vec![DriftAlternative {
                        alternative_id: "alt".into(),
                        mixture_weight_ppm: 1_000_000,
                        probabilities: vec![prob("routine", 500_000), prob("disruption", 500_000)],
                    }],
                    alpha,
                ),
                Err(DriftMonitorError::InvalidAlpha { .. })
            ));
        }
        assert!(matches!(
            OnlineDriftMonitor::new(
                vec![prob("routine", 1_100_000)],
                vec![DriftAlternative {
                    alternative_id: "alt".into(),
                    mixture_weight_ppm: 1_000_000,
                    probabilities: vec![prob("routine", 1_000_000)],
                }],
                50_000,
            ),
            Err(DriftMonitorError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn rejects_invalid_mass_duplicate_ids_and_support_mismatch() {
        assert!(matches!(
            OnlineDriftMonitor::new(vec![prob("x", 900_000)], vec![DriftAlternative {
                alternative_id: "alt".into(), mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("x", 1_000_000)],
            }], 50_000),
            Err(DriftMonitorError::ProbabilityMassMismatch { .. })
        ));
        assert!(matches!(
            OnlineDriftMonitor::new(vec![prob("x", 500_000), prob("x", 500_000)], vec![DriftAlternative {
                alternative_id: "alt".into(), mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("x", 1_000_000)],
            }], 50_000),
            Err(DriftMonitorError::DuplicateScenarioId { .. })
        ));
        assert!(matches!(
            OnlineDriftMonitor::new(vec![prob("x", 1_000_000)], vec![DriftAlternative {
                alternative_id: "alt".into(), mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("other", 1_000_000)],
            }], 50_000),
            Err(DriftMonitorError::AlternativeSupportMismatch { .. })
        ));
    }

    #[test]
    fn row_order_does_not_change_monitor_results() {
        let mut a = monitor(50_000);
        let mut nominal = vec![prob("routine", 900_000), prob("disruption", 100_000)];
        nominal.reverse();
        let mut alternative_probs = vec![prob("routine", 500_000), prob("disruption", 500_000)];
        alternative_probs.reverse();
        let mut b = OnlineDriftMonitor::new(
            nominal,
            vec![DriftAlternative {
                alternative_id: "disruption-heavy".into(),
                mixture_weight_ppm: 1_000_000,
                probabilities: alternative_probs,
            }],
            50_000,
        ).unwrap();
        let a1 = a.observe("e1", 10, "disruption").unwrap();
        let b1 = b.observe("e1", 10, "disruption").unwrap();
        assert_eq!(a1, b1);
    }

    #[test]
    fn explicit_alternative_mixture_weights_must_sum_to_one() {
        assert!(matches!(
            OnlineDriftMonitor::new(
                vec![prob("routine", 900_000), prob("disruption", 100_000)],
                vec![DriftAlternative {
                    alternative_id: "alt".into(),
                    mixture_weight_ppm: 900_000,
                    probabilities: vec![prob("routine", 500_000), prob("disruption", 500_000)],
                }],
                50_000,
            ),
            Err(DriftMonitorError::MixtureWeightMassMismatch { .. })
        ));
    }

    #[test]
    fn large_evidence_stream_stays_in_log_space() {
        let mut monitor = monitor(50_000);
        let first = monitor.observe("e0", 0, "disruption").unwrap();
        assert!(first.log_e_value.is_finite());
        let second = monitor.observe("e1", 1, "disruption").unwrap();
        assert_eq!(second.status, DriftMonitorStatus::DriftAlarm);
        assert!(second.log_e_value.is_finite());
        assert!(matches!(
            monitor.observe("e2", 2, "disruption"),
            Err(DriftMonitorError::AlreadyAlarmed { .. })
        ));
        assert!(monitor.report().log_e_value.is_finite());
    }
 
    #[test]
    fn positive_nominal_categories_must_have_alternative_mixture_coverage() {
        assert!(matches!(
            OnlineDriftMonitor::new(
                vec![prob("routine", 900_000), prob("disruption", 100_000)],
                vec![DriftAlternative {
                    alternative_id: "routine-only".into(),
                    mixture_weight_ppm: 1_000_000,
                    probabilities: vec![prob("routine", 1_000_000), prob("disruption", 0)],
                }],
                50_000,
            ),
            Err(DriftMonitorError::AlternativeMixtureHasZeroSupport { scenario_id }) if scenario_id == "disruption"
        ));
    }

    #[test]
    fn alternatives_must_assign_positive_mass_to_every_nominally_possible_category() {
        assert!(matches!(
            OnlineDriftMonitor::new(
                vec![prob("a", 500_000), prob("b", 500_000)],
                vec![DriftAlternative {
                    alternative_id: "misses-a".into(),
                    mixture_weight_ppm: 1_000_000,
                    probabilities: vec![prob("a", 0), prob("b", 1_000_000)],
                }],
                50_000,
            ),
            Err(DriftMonitorError::AlternativeHasZeroProbabilityForNominalSupport { .. })
        ));
    }

 
    #[test]
    fn empty_event_id_and_duplicate_alternative_id_are_rejected() {
        let mut monitor = monitor(50_000);
        assert_eq!(
            monitor.observe(" ", 10, "routine"),
            Err(DriftMonitorError::EmptyEventId)
        );
        assert_eq!(monitor.observations_seen(), 0);

        let invalid = OnlineDriftMonitor::new(
            vec![prob("routine", 900_000), prob("disruption", 100_000)],
            vec![
                DriftAlternative {
                    alternative_id: "same".into(),
                    mixture_weight_ppm: 500_000,
                    probabilities: vec![prob("routine", 500_000), prob("disruption", 500_000)],
                },
                DriftAlternative {
                    alternative_id: "same".into(),
                    mixture_weight_ppm: 500_000,
                    probabilities: vec![prob("routine", 800_000), prob("disruption", 200_000)],
                },
            ],
            50_000,
        );
        assert!(matches!(invalid, Err(DriftMonitorError::DuplicateAlternativeId { .. })));
    }

    #[test]
    fn nominal_distribution_must_be_covered_by_each_alternative() {
        let invalid = OnlineDriftMonitor::new(
            vec![prob("routine", 900_000), prob("disruption", 100_000)],
            vec![DriftAlternative {
                alternative_id: "broken".into(),
                mixture_weight_ppm: 1_000_000,
                probabilities: vec![prob("routine", 1_000_000), prob("disruption", 0)],
            }],
            50_000,
        );
        assert!(matches!(
            invalid,
            Err(DriftMonitorError::AlternativeHasZeroProbabilityForNominalSupport { .. })
        ));
    }

}

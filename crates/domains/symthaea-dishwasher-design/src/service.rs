//! Resource accounting per independently qualified successful item.
//!
//! Every physical attempt contributes its measured resources, including failed,
//! interrupted and rewash attempts. An item qualifies only when every required
//! endpoint has an independent-assay PASS in the same completed run.
//! This module performs ledger arithmetic; it does not authenticate assays,
//! prove independence or establish appliance performance.

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::quality::{Decision, Endpoint, EvidenceKind};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AttemptResources {
    pub appliance_electricity_kwh: f64,
    pub upstream_hot_water_energy_kwh: f64,
    pub fresh_water_l: f64,
    pub detergent_g: f64,
    pub active_operator_seconds: f64,
}

impl AttemptResources {
    fn validate(self) -> Result<(), ServiceAccountingError> {
        for (field, value) in [
            ("appliance_electricity_kwh", self.appliance_electricity_kwh),
            ("upstream_hot_water_energy_kwh", self.upstream_hot_water_energy_kwh),
            ("fresh_water_l", self.fresh_water_l),
            ("detergent_g", self.detergent_g),
            ("active_operator_seconds", self.active_operator_seconds),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(ServiceAccountingError::InvalidResource(field));
            }
        }
        Ok(())
    }

    fn checked_add(self, other: Self) -> Result<Self, ServiceAccountingError> {
        let total = Self {
            appliance_electricity_kwh: self.appliance_electricity_kwh + other.appliance_electricity_kwh,
            upstream_hot_water_energy_kwh: self.upstream_hot_water_energy_kwh + other.upstream_hot_water_energy_kwh,
            fresh_water_l: self.fresh_water_l + other.fresh_water_l,
            detergent_g: self.detergent_g + other.detergent_g,
            active_operator_seconds: self.active_operator_seconds + other.active_operator_seconds,
        };
        total.validate()?;
        Ok(total)
    }

    fn divided_by(self, divisor: usize) -> Self {
        let d = divisor as f64;
        Self {
            appliance_electricity_kwh: self.appliance_electricity_kwh / d,
            upstream_hot_water_energy_kwh: self.upstream_hot_water_energy_kwh / d,
            fresh_water_l: self.fresh_water_l / d,
            detergent_g: self.detergent_g / d,
            active_operator_seconds: self.active_operator_seconds / d,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ItemEndpointOutcome {
    /// Must equal the physical run ID of the containing attempt.
    pub run_id: String,
    pub item_id: String,
    pub endpoint: Endpoint,
    /// Only independent outcome assays can qualify a successful item.
    pub evidence_kind: EvidenceKind,
    pub decision: Decision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceAttempt {
    /// Strictly increasing sequence in this service ledger.
    pub sequence: u64,
    pub attempt_id: String,
    /// Each physical rewash must have a new run ID.
    pub run_id: String,
    pub started_at_s: u64,
    pub finished_at_s: u64,
    /// Stable IDs for individual service requests, not dish type labels.
    pub item_ids: Vec<String>,
    /// Include allocated preheat/standby/purge overhead in this resource vector.
    pub resources: AttemptResources,
    pub interrupted: bool,
    /// Missing observations are unknown, never passes.
    pub outcomes: Vec<ItemEndpointOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceAccountingError {
    EmptyAttempts,
    EmptyRequiredEndpoints,
    DuplicateRequiredEndpoint(Endpoint),
    InvalidIdentity(&'static str),
    DuplicateAttemptId(String),
    DuplicateRunId(String),
    NonIncreasingSequence { previous: u64, observed: u64 },
    FinishBeforeStart { attempt_id: String },
    InvalidResource(&'static str),
    DuplicateItemInAttempt { attempt_id: String, item_id: String },
    ItemRepeatedAfterQualification { item_id: String, attempt_id: String },
    OutcomeForUnknownItem { attempt_id: String, item_id: String },
    OutcomeRunMismatch { attempt_id: String, expected_run_id: String, observed_run_id: String },
    DuplicateOutcome { attempt_id: String, item_id: String, endpoint: Endpoint },
    ProcessProxyCannotQualify { attempt_id: String, item_id: String, endpoint: Endpoint },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServiceAccountingReport {
    pub attempt_count: usize,
    pub physical_run_count: usize,
    pub unique_item_count: usize,
    pub item_attempt_visits: usize,
    pub reattempted_item_visits: usize,
    pub verified_successful_item_count: usize,
    pub verified_successful_item_ids: Vec<String>,
    pub unqualified_item_ids: Vec<String>,
    /// Includes every failed, interrupted and rewash attempt.
    pub total_resources: AttemptResources,
    /// None when no item qualifies; avoids a misleading zero denominator.
    pub resources_per_verified_item: Option<AttemptResources>,
    /// Wall-clock service window; overlapping attempts are not summed.
    pub service_makespan_s: u64,
}

/// Account all resources against unique items that pass every required endpoint.
///
/// Attempts must be in strictly increasing sequence order. An item can be
/// reattempted after a failed, interrupted, indeterminate or incomplete run.
/// All previous resource consumption remains in the numerator. Reattempting
/// an already-qualified service item is rejected; assign a new service-item ID
/// for a new use request. Outcome records must already be bound to the exact
/// physical run and validated criteria by the caller.
pub fn account_service(
    attempts: &[ServiceAttempt],
    required_endpoints: &[Endpoint],
) -> Result<ServiceAccountingReport, ServiceAccountingError> {
    if attempts.is_empty() { return Err(ServiceAccountingError::EmptyAttempts); }
    if required_endpoints.is_empty() { return Err(ServiceAccountingError::EmptyRequiredEndpoints); }

    let mut required = BTreeSet::new();
    for endpoint in required_endpoints {
        if !required.insert(*endpoint) {
            return Err(ServiceAccountingError::DuplicateRequiredEndpoint(*endpoint));
        }
    }

    let mut attempt_ids = HashSet::new();
    let mut run_ids = HashSet::new();
    let mut seen_items = BTreeSet::new();
    let mut qualified_items = BTreeSet::new();
    let mut item_attempt_visits = 0_usize;
    let mut total_resources = AttemptResources::default();
    let mut earliest_start = u64::MAX;
    let mut latest_finish = 0_u64;
    let mut previous_sequence = None;

    for attempt in attempts {
        if !canonical_non_empty(&attempt.attempt_id) {
            return Err(ServiceAccountingError::InvalidIdentity("attempt_id"));
        }
        if !canonical_non_empty(&attempt.run_id) {
            return Err(ServiceAccountingError::InvalidIdentity("run_id"));
        }
        if !attempt_ids.insert(attempt.attempt_id.as_str()) {
            return Err(ServiceAccountingError::DuplicateAttemptId(attempt.attempt_id.clone()));
        }
        if !run_ids.insert(attempt.run_id.as_str()) {
            return Err(ServiceAccountingError::DuplicateRunId(attempt.run_id.clone()));
        }
        if let Some(previous) = previous_sequence {
            if attempt.sequence <= previous {
                return Err(ServiceAccountingError::NonIncreasingSequence {
                    previous,
                    observed: attempt.sequence,
                });
            }
        }
        previous_sequence = Some(attempt.sequence);

        if attempt.finished_at_s < attempt.started_at_s {
            return Err(ServiceAccountingError::FinishBeforeStart {
                attempt_id: attempt.attempt_id.clone(),
            });
        }
        attempt.resources.validate()?;
        total_resources = total_resources.checked_add(attempt.resources)?;
        earliest_start = earliest_start.min(attempt.started_at_s);
        latest_finish = latest_finish.max(attempt.finished_at_s);

        let mut items_in_attempt = HashSet::new();
        for item_id in &attempt.item_ids {
            if !canonical_non_empty(item_id) {
                return Err(ServiceAccountingError::InvalidIdentity("item_id"));
            }
            if !items_in_attempt.insert(item_id.as_str()) {
                return Err(ServiceAccountingError::DuplicateItemInAttempt {
                    attempt_id: attempt.attempt_id.clone(),
                    item_id: item_id.clone(),
                });
            }
            if qualified_items.contains(item_id) {
                return Err(ServiceAccountingError::ItemRepeatedAfterQualification {
                    item_id: item_id.clone(),
                    attempt_id: attempt.attempt_id.clone(),
                });
            }
            seen_items.insert(item_id.clone());
            item_attempt_visits = item_attempt_visits.checked_add(1)
                .ok_or(ServiceAccountingError::InvalidResource("item_attempt_visits overflow"))?;
        }

        let mut decisions: HashMap<(&str, Endpoint), Decision> = HashMap::new();
        for outcome in &attempt.outcomes {
            if !canonical_non_empty(&outcome.run_id) {
                return Err(ServiceAccountingError::InvalidIdentity("outcome.run_id"));
            }
            if outcome.run_id != attempt.run_id {
                return Err(ServiceAccountingError::OutcomeRunMismatch {
                    attempt_id: attempt.attempt_id.clone(),
                    expected_run_id: attempt.run_id.clone(),
                    observed_run_id: outcome.run_id.clone(),
                });
            }
            if !items_in_attempt.contains(outcome.item_id.as_str()) {
                return Err(ServiceAccountingError::OutcomeForUnknownItem {
                    attempt_id: attempt.attempt_id.clone(),
                    item_id: outcome.item_id.clone(),
                });
            }
            if outcome.evidence_kind != EvidenceKind::IndependentOutcomeAssay {
                return Err(ServiceAccountingError::ProcessProxyCannotQualify {
                    attempt_id: attempt.attempt_id.clone(),
                    item_id: outcome.item_id.clone(),
                    endpoint: outcome.endpoint,
                });
            }
            if decisions.insert((outcome.item_id.as_str(), outcome.endpoint), outcome.decision).is_some() {
                return Err(ServiceAccountingError::DuplicateOutcome {
                    attempt_id: attempt.attempt_id.clone(),
                    item_id: outcome.item_id.clone(),
                    endpoint: outcome.endpoint,
                });
            }
        }

        if !attempt.interrupted {
            for item_id in &attempt.item_ids {
                let all_required_pass = required.iter().all(|endpoint| {
                    decisions.get(&(item_id.as_str(), *endpoint))
                        .is_some_and(|decision| *decision == Decision::Pass)
                });
                if all_required_pass {
                    qualified_items.insert(item_id.clone());
                }
            }
        }
    }

    let unique_item_count = seen_items.len();
    let verified_successful_item_count = qualified_items.len();
    let unqualified_item_ids = seen_items.difference(&qualified_items).cloned().collect();
    let resources_per_verified_item = (verified_successful_item_count > 0)
        .then(|| total_resources.divided_by(verified_successful_item_count));

    Ok(ServiceAccountingReport {
        attempt_count: attempts.len(),
        physical_run_count: run_ids.len(),
        unique_item_count,
        item_attempt_visits,
        reattempted_item_visits: item_attempt_visits.saturating_sub(unique_item_count),
        verified_successful_item_count,
        verified_successful_item_ids: qualified_items.into_iter().collect(),
        unqualified_item_ids,
        total_resources,
        resources_per_verified_item,
        service_makespan_s: latest_finish.saturating_sub(earliest_start),
    })
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUIRED: [Endpoint; 3] = [
        Endpoint::SoilRemovalResidual,
        Endpoint::FinalRinseResidue,
        Endpoint::CrossContamination,
    ];

    fn outcomes(run_id: &str, item_id: &str, decision: Decision) -> Vec<ItemEndpointOutcome> {
        REQUIRED.iter().map(|endpoint| ItemEndpointOutcome {
            run_id: run_id.to_owned(),
            item_id: item_id.to_owned(),
            endpoint: *endpoint,
            evidence_kind: EvidenceKind::IndependentOutcomeAssay,
            decision,
        }).collect()
    }

    fn attempt(
        sequence: u64, attempt_id: &str, run_id: &str, start: u64, finish: u64,
        item_ids: &[&str], resources: AttemptResources, outcomes: Vec<ItemEndpointOutcome>,
    ) -> ServiceAttempt {
        ServiceAttempt {
            sequence,
            attempt_id: attempt_id.to_owned(),
            run_id: run_id.to_owned(),
            started_at_s: start,
            finished_at_s: finish,
            item_ids: item_ids.iter().map(|id| (*id).to_owned()).collect(),
            resources,
            interrupted: false,
            outcomes,
        }
    }

    fn resources(energy: f64, water: f64) -> AttemptResources {
        AttemptResources {
            appliance_electricity_kwh: energy,
            upstream_hot_water_energy_kwh: 0.1,
            fresh_water_l: water,
            detergent_g: 2.0,
            active_operator_seconds: 15.0,
        }
    }

    #[test]
    fn qualifies_only_when_every_required_endpoint_passes() {
        let report = account_service(&[attempt(
            1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7),
            outcomes("run-1", "cup-1", Decision::Pass),
        )], &REQUIRED).unwrap();
        assert_eq!(report.verified_successful_item_ids, vec!["cup-1"]);
        assert_eq!(report.resources_per_verified_item.unwrap().fresh_water_l, 0.7);
    }

    #[test]
    fn failed_attempt_resources_remain_after_successful_rewash() {
        let attempts = [
            attempt(1, "attempt-1", "run-1", 0, 90, &["cup-1"], resources(0.04, 0.8),
                outcomes("run-1", "cup-1", Decision::Fail)),
            attempt(2, "attempt-2", "run-2", 100, 160, &["cup-1"], resources(0.05, 0.7),
                outcomes("run-2", "cup-1", Decision::Pass)),
        ];
        let report = account_service(&attempts, &REQUIRED).unwrap();
        assert_eq!(report.unique_item_count, 1);
        assert_eq!(report.verified_successful_item_count, 1);
        assert_eq!(report.item_attempt_visits, 2);
        assert_eq!(report.reattempted_item_visits, 1);
        assert!((report.total_resources.appliance_electricity_kwh - 0.09).abs() < 1e-12);
        assert!((report.total_resources.fresh_water_l - 1.5).abs() < 1e-12);
        assert_eq!(report.resources_per_verified_item.unwrap().fresh_water_l, 1.5);
        assert_eq!(report.service_makespan_s, 160);
    }

    #[test]
    fn missing_or_indeterminate_endpoint_never_becomes_a_pass() {
        let mut evidence = outcomes("run-1", "cup-1", Decision::Pass);
        evidence.pop();
        evidence[0].decision = Decision::Indeterminate;
        let report = account_service(&[attempt(
            1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7), evidence,
        )], &REQUIRED).unwrap();
        assert_eq!(report.verified_successful_item_count, 0);
        assert_eq!(report.unqualified_item_ids, vec!["cup-1"]);
        assert!(report.resources_per_verified_item.is_none());
    }

    #[test]
    fn a_failed_item_does_not_hide_another_items_success() {
        let mut evidence = outcomes("run-1", "cup-pass", Decision::Pass);
        evidence.extend(outcomes("run-1", "cup-fail", Decision::Fail));
        let report = account_service(&[attempt(
            1, "attempt-1", "run-1", 0, 60, &["cup-pass", "cup-fail"],
            resources(0.1, 1.4), evidence,
        )], &REQUIRED).unwrap();
        assert_eq!(report.verified_successful_item_ids, vec!["cup-pass"]);
        assert_eq!(report.unqualified_item_ids, vec!["cup-fail"]);
        assert_eq!(report.resources_per_verified_item.unwrap().fresh_water_l, 1.4);
    }

    #[test]
    fn interrupted_attempt_consumes_resources_but_releases_nothing() {
        let mut interrupted = attempt(
            1, "attempt-1", "run-1", 0, 30, &["cup-1"], resources(0.03, 0.4),
            outcomes("run-1", "cup-1", Decision::Pass),
        );
        interrupted.interrupted = true;
        let report = account_service(&[interrupted], &REQUIRED).unwrap();
        assert_eq!(report.verified_successful_item_count, 0);
        assert_eq!(report.total_resources.fresh_water_l, 0.4);
    }

    #[test]
    fn process_proxy_cannot_qualify_an_item() {
        let mut evidence = outcomes("run-1", "cup-1", Decision::Pass);
        evidence[0].evidence_kind = EvidenceKind::CalibratedProcessProxy;
        assert!(matches!(
            account_service(&[attempt(
                1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7), evidence,
            )], &REQUIRED),
            Err(ServiceAccountingError::ProcessProxyCannotQualify { .. })
        ));
    }


    #[test]
    fn rejects_outcome_from_a_different_physical_run() {
        let evidence = outcomes("another-run", "cup-1", Decision::Pass);
        assert!(matches!(
            account_service(&[attempt(
                1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7), evidence,
            )], &REQUIRED),
            Err(ServiceAccountingError::OutcomeRunMismatch { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_outcomes_and_duplicate_physical_run_ids() {
        let mut evidence = outcomes("run-1", "cup-1", Decision::Pass);
        evidence.push(evidence[0].clone());
        assert!(matches!(
            account_service(&[attempt(
                1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7), evidence,
            )], &REQUIRED),
            Err(ServiceAccountingError::DuplicateOutcome { .. })
        ));
        let duplicate_runs = [
            attempt(1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7), vec![]),
            attempt(2, "attempt-2", "run-1", 70, 130, &["cup-2"], resources(0.05, 0.7), vec![]),
        ];
        assert!(matches!(
            account_service(&duplicate_runs, &REQUIRED),
            Err(ServiceAccountingError::DuplicateRunId(_))
        ));
    }

    #[test]
    fn rejects_invalid_resources_and_non_increasing_sequence() {
        let invalid = attempt(
            1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(f64::NAN, 0.7),
            outcomes("run-1", "cup-1", Decision::Pass),
        );
        assert!(matches!(
            account_service(&[invalid], &REQUIRED),
            Err(ServiceAccountingError::InvalidResource(_))
        ));
        let a = attempt(1, "attempt-a", "run-a", 0, 60, &["cup-a"], resources(0.05, 0.7), vec![]);
        let b = attempt(1, "attempt-b", "run-b", 70, 130, &["cup-b"], resources(0.05, 0.7), vec![]);
        assert!(matches!(
            account_service(&[a, b], &REQUIRED),
            Err(ServiceAccountingError::NonIncreasingSequence { .. })
        ));
    }

    #[test]
    fn rejects_reattempt_after_an_item_was_already_qualified() {
        let attempts = [
            attempt(1, "attempt-1", "run-1", 0, 60, &["cup-1"], resources(0.05, 0.7),
                outcomes("run-1", "cup-1", Decision::Pass)),
            attempt(2, "attempt-2", "run-2", 70, 130, &["cup-1"], resources(0.05, 0.7),
                outcomes("run-2", "cup-1", Decision::Pass)),
        ];
        assert!(matches!(
            account_service(&attempts, &REQUIRED),
            Err(ServiceAccountingError::ItemRepeatedAfterQualification { .. })
        ));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Standalone asynchronous experiment buffer with synthetic-placeholder isolation.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/async_buffer.rs -o /tmp/isru-async-tests
//   /tmp/isru-async-tests

#![forbid(unsafe_code)]

#[path = "campaign_resources.rs"]
mod campaign_resources;

use campaign_resources::{CampaignBudget, ResourceBasis, ResourceVector};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveSense {
    Maximize,
    Minimize,
}

/// Placeholder policies affect only the acquisition-model fantasy buffer.
/// None of these values are physical observations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaceholderPolicy {
    NoPlaceholder,
    RealisticMean,
    PessimisticBound { beta: f64 },
    SeededUniform { seed: u64, radius_multiplier: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaceholderKind {
    RealisticMean,
    PessimisticBound,
    SeededUniform,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateInstance {
    pub candidate_id: String,
    pub replicate_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobProposal {
    pub job_id: String,
    pub candidate: CandidateInstance,
    pub resource_locks: Vec<String>,
    pub resource_cost: ResourceVector,
    pub predicted_mean: f64,
    pub predicted_standard_deviation: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActualResult {
    pub result_id: String,
    pub observation_id: String,
    pub value: f64,
    pub observed_at_utc_s: i64,
    pub calibration_receipt_id: String,
    pub qualification_receipt_id: String,
    pub recorded_at_utc_s: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FantasyObservation {
    pub job_id: String,
    pub candidate: CandidateInstance,
    pub value: f64,
    pub kind: PlaceholderKind,
    pub policy_seed: Option<u64>,
    pub launch_sequence: u64,
}

/// A synthetic fantasy is intentionally a different type from ActualObservation.
/// No API converts one into the other.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActualObservation {
    pub result_id: String,
    pub observation_id: String,
    pub job_id: String,
    pub candidate: CandidateInstance,
    pub observed_at_utc_s: i64,
    pub recorded_at_utc_s: i64,
    pub calibration_receipt_id: String,
    pub qualification_receipt_id: String,
    pub value_bits: u64,
}

impl ActualObservation {
    pub fn value(&self) -> f64 {
        f64::from_bits(self.value_bits)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InFlightJob {
    pub proposal: JobProposal,
    pub launched_at_utc_s: i64,
    pub recorded_at_utc_s: i64,
    pub fantasy: Option<FantasyObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AsyncEventKind {
    JobLaunched {
        job_id: String,
        candidate: CandidateInstance,
        placeholder_kind: Option<PlaceholderKind>,
        placeholder_value: Option<f64>,
        policy_seed: Option<u64>,
    },
    JobResolved {
        job_id: String,
        result_id: String,
        observation_id: String,
    },
    JobFailed {
        job_id: String,
        failure_receipt_id: String,
    },
    CampaignClosed {
        closeout_receipt_id: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct AsyncEvent {
    pub sequence: u64,
    pub recorded_at_utc_s: i64,
    pub kind: AsyncEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AsyncError {
    EmptyIdentity(&'static str),
    InvalidPrediction,
    InvalidPlaceholderPolicy,
    DuplicateJob(String),
    DuplicateInFlightCandidate(CandidateInstance),
    UnknownJob(String),
    DuplicateResourceLock(String),
    ResourceLockBusy { resource: String, job_id: String },
    ResourceBudgetExceeded(Vec<&'static str>),
    InvalidResource,
    InvalidResult,
    DuplicateResult(String),
    DuplicateObservation(String),
    ObservationBeforeLaunch(String),
    ObservationAfterReceipt(String),
    EventTimeWentBackwards,
    MissingReceipt(&'static str),
    ActiveJobsRemain,
    CampaignClosed,
}

impl fmt::Display for AsyncError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for AsyncError {}

#[derive(Clone, Debug, PartialEq)]
pub struct AsyncScheduler {
    pub budget: CampaignBudget,
    pub spent: ResourceVector,
    pub reserved: ResourceVector,
    pub active_jobs: BTreeMap<String, InFlightJob>,
    pub actual_observations: BTreeMap<String, ActualObservation>,
    pub events: Vec<AsyncEvent>,
    pub closed: bool,
    seen_job_ids: BTreeSet<String>,
    seen_result_ids: BTreeSet<String>,
    next_launch_sequence: u64,
}

fn zero_resources() -> ResourceVector {
    ResourceVector {
        rig_hours: 0.0,
        source_energy_wh: 0.0,
        feedstock_g: 0.0,
        assay_cost: 0.0,
        maintenance_minutes: 0.0,
        consumables_g: 0.0,
    }
}

fn add_resources(left: &ResourceVector, right: &ResourceVector) -> ResourceVector {
    ResourceVector {
        rig_hours: left.rig_hours + right.rig_hours,
        source_energy_wh: left.source_energy_wh + right.source_energy_wh,
        feedstock_g: left.feedstock_g + right.feedstock_g,
        assay_cost: left.assay_cost + right.assay_cost,
        maintenance_minutes: left.maintenance_minutes + right.maintenance_minutes,
        consumables_g: left.consumables_g + right.consumables_g,
    }
}

fn subtract_resources(left: &ResourceVector, right: &ResourceVector) -> ResourceVector {
    ResourceVector {
        rig_hours: left.rig_hours - right.rig_hours,
        source_energy_wh: left.source_energy_wh - right.source_energy_wh,
        feedstock_g: left.feedstock_g - right.feedstock_g,
        assay_cost: left.assay_cost - right.assay_cost,
        maintenance_minutes: left.maintenance_minutes - right.maintenance_minutes,
        consumables_g: left.consumables_g - right.consumables_g,
    }
}

fn over_budget(cost: &ResourceVector, budget: &ResourceVector) -> Vec<&'static str> {
    let mut over = Vec::new();
    if cost.rig_hours > budget.rig_hours {
        over.push("rig_hours");
    }
    if cost.source_energy_wh > budget.source_energy_wh {
        over.push("source_energy_wh");
    }
    if cost.feedstock_g > budget.feedstock_g {
        over.push("feedstock_g");
    }
    if cost.assay_cost > budget.assay_cost {
        over.push("assay_cost");
    }
    if cost.maintenance_minutes > budget.maintenance_minutes {
        over.push("maintenance_minutes");
    }
    if cost.consumables_g > budget.consumables_g {
        over.push("consumables_g");
    }
    over
}

fn validate_policy(policy: PlaceholderPolicy) -> Result<(), AsyncError> {
    match policy {
        PlaceholderPolicy::NoPlaceholder | PlaceholderPolicy::RealisticMean => Ok(()),
        PlaceholderPolicy::PessimisticBound { beta }
            if beta.is_finite() && beta >= 0.0 =>
        {
            Ok(())
        }
        PlaceholderPolicy::SeededUniform {
            seed,
            radius_multiplier,
        } if seed != 0 && radius_multiplier.is_finite() && radius_multiplier >= 0.0 => Ok(()),
        _ => Err(AsyncError::InvalidPlaceholderPolicy),
    }
}

fn next_random(state: &mut u64) -> u64 {
    // xorshift64*: repeatable research randomness, not a cryptographic RNG.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn uniform_unit_interval(seed: u64, launch_sequence: u64) -> f64 {
    let mut state = seed ^ launch_sequence.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    if state == 0 {
        state = 0xa076_1d64_78bd_642f;
    }
    let mantissa = next_random(&mut state) >> 11;
    mantissa as f64 / ((1_u64 << 53) as f64)
}

fn make_fantasy(
    proposal: &JobProposal,
    policy: PlaceholderPolicy,
    objective: ObjectiveSense,
    launch_sequence: u64,
) -> Result<Option<FantasyObservation>, AsyncError> {
    validate_policy(policy)?;
    if !proposal.predicted_mean.is_finite()
        || !proposal.predicted_standard_deviation.is_finite()
        || proposal.predicted_standard_deviation < 0.0
    {
        return Err(AsyncError::InvalidPrediction);
    }

    let (kind, value, policy_seed) = match policy {
        PlaceholderPolicy::NoPlaceholder => return Ok(None),
        PlaceholderPolicy::RealisticMean => (
            PlaceholderKind::RealisticMean,
            proposal.predicted_mean,
            None,
        ),
        PlaceholderPolicy::PessimisticBound { beta } => {
            let offset = beta * proposal.predicted_standard_deviation;
            let value = match objective {
                ObjectiveSense::Maximize => proposal.predicted_mean - offset,
                ObjectiveSense::Minimize => proposal.predicted_mean + offset,
            };
            (PlaceholderKind::PessimisticBound, value, None)
        }
        PlaceholderPolicy::SeededUniform {
            seed,
            radius_multiplier,
        } => {
            let unit = uniform_unit_interval(seed, launch_sequence);
            let centered = 2.0 * unit - 1.0;
            let value = proposal.predicted_mean
                + centered * radius_multiplier * proposal.predicted_standard_deviation;
            (PlaceholderKind::SeededUniform, value, Some(seed))
        }
    };

    if !value.is_finite() {
        return Err(AsyncError::InvalidPrediction);
    }

    Ok(Some(FantasyObservation {
        job_id: proposal.job_id.clone(),
        candidate: proposal.candidate.clone(),
        value,
        kind,
        policy_seed,
        launch_sequence,
    }))
}

impl AsyncScheduler {
    pub fn new(budget: CampaignBudget) -> Result<Self, AsyncError> {
        budget.validate().map_err(|_| AsyncError::InvalidResource)?;
        Ok(Self {
            budget,
            spent: zero_resources(),
            reserved: zero_resources(),
            active_jobs: BTreeMap::new(),
            actual_observations: BTreeMap::new(),
            events: Vec::new(),
            closed: false,
            seen_job_ids: BTreeSet::new(),
            seen_result_ids: BTreeSet::new(),
            next_launch_sequence: 0,
        })
    }

    fn ensure_open(&self) -> Result<(), AsyncError> {
        if self.closed {
            return Err(AsyncError::CampaignClosed);
        }
        Ok(())
    }

    fn ensure_event_time(&self, recorded_at_utc_s: i64) -> Result<(), AsyncError> {
        if self
            .events
            .last()
            .is_some_and(|previous| recorded_at_utc_s < previous.recorded_at_utc_s)
        {
            return Err(AsyncError::EventTimeWentBackwards);
        }
        Ok(())
    }

    fn append_event(
        &mut self,
        recorded_at_utc_s: i64,
        kind: AsyncEventKind,
    ) -> Result<(), AsyncError> {
        self.ensure_event_time(recorded_at_utc_s)?;
        self.events.push(AsyncEvent {
            sequence: self.events.len() as u64,
            recorded_at_utc_s,
            kind,
        });
        Ok(())
    }

    pub fn remaining_budget(&self) -> ResourceVector {
        let committed = add_resources(&self.spent, &self.reserved);
        subtract_resources(&self.budget.resources, &committed)
    }

    /// Exposes optimizer-only synthetic values. This collection is not the
    /// physical-observation store and is never returned by actual_observations().
    pub fn fantasies_for_model(&self) -> Vec<&FantasyObservation> {
        self.active_jobs
            .values()
            .filter_map(|job| job.fantasy.as_ref())
            .collect()
    }

    pub fn active_candidate_count(&self) -> usize {
        self.active_jobs.len()
    }

    pub fn launch_job(
        &mut self,
        proposal: JobProposal,
        policy: PlaceholderPolicy,
        objective: ObjectiveSense,
        launched_at_utc_s: i64,
        recorded_at_utc_s: i64,
    ) -> Result<(), AsyncError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;

        if !nonempty(&proposal.job_id) {
            return Err(AsyncError::EmptyIdentity("job_id"));
        }
        if !nonempty(&proposal.candidate.candidate_id) {
            return Err(AsyncError::EmptyIdentity("candidate_id"));
        }
        if let Some(replicate_id) = &proposal.candidate.replicate_id {
            if !nonempty(replicate_id) {
                return Err(AsyncError::EmptyIdentity("replicate_id"));
            }
        }
        if launched_at_utc_s > recorded_at_utc_s {
            return Err(AsyncError::ObservationAfterReceipt(
                proposal.job_id.clone(),
            ));
        }
        if self.seen_job_ids.contains(&proposal.job_id) {
            return Err(AsyncError::DuplicateJob(proposal.job_id));
        }
        if self
            .active_jobs
            .values()
            .any(|job| job.proposal.candidate == proposal.candidate)
        {
            return Err(AsyncError::DuplicateInFlightCandidate(
                proposal.candidate,
            ));
        }

        proposal
            .resource_cost
            .validate()
            .map_err(|_| AsyncError::InvalidResource)?;

        let mut locks = BTreeSet::new();
        for resource in &proposal.resource_locks {
            if !nonempty(resource) {
                return Err(AsyncError::EmptyIdentity("resource_lock"));
            }
            if !locks.insert(resource.as_str()) {
                return Err(AsyncError::DuplicateResourceLock(resource.clone()));
            }
            let busy = self.active_jobs.values().any(|job| {
                job.proposal.resource_locks.iter().any(|held| held == resource)
            });
            if busy {
                return Err(AsyncError::ResourceLockBusy {
                    resource: resource.clone(),
                    job_id: proposal.job_id.clone(),
                });
            }
        }

        let committed = add_resources(&self.spent, &self.reserved);
        let proposed_total = add_resources(&committed, &proposal.resource_cost);
        let over = over_budget(&proposed_total, &self.budget.resources);
        if !over.is_empty() {
            return Err(AsyncError::ResourceBudgetExceeded(over));
        }

        let launch_sequence = self.next_launch_sequence;
        let fantasy = make_fantasy(&proposal, policy, objective, launch_sequence)?;
        let event_kind = AsyncEventKind::JobLaunched {
            job_id: proposal.job_id.clone(),
            candidate: proposal.candidate.clone(),
            placeholder_kind: fantasy.as_ref().map(|item| item.kind),
            placeholder_value: fantasy.as_ref().map(|item| item.value),
            policy_seed: fantasy.as_ref().and_then(|item| item.policy_seed),
        };

        self.reserved = add_resources(&self.reserved, &proposal.resource_cost);
        self.seen_job_ids.insert(proposal.job_id.clone());
        self.active_jobs.insert(
            proposal.job_id.clone(),
            InFlightJob {
                proposal,
                launched_at_utc_s,
                recorded_at_utc_s,
                fantasy,
            },
        );
        self.next_launch_sequence += 1;
        self.append_event(recorded_at_utc_s, event_kind)
    }

    /// Resolve one in-flight job. Resolution order is independent of launch
    /// order; the event log records receipt time while preserving observation time.
    pub fn resolve_job(
        &mut self,
        job_id: &str,
        result: ActualResult,
    ) -> Result<(), AsyncError> {
        self.ensure_open()?;
        self.ensure_event_time(result.recorded_at_utc_s)?;
        if !nonempty(&result.result_id) {
            return Err(AsyncError::EmptyIdentity("result_id"));
        }
        if !nonempty(&result.observation_id) {
            return Err(AsyncError::EmptyIdentity("observation_id"));
        }
        if !result.value.is_finite() {
            return Err(AsyncError::InvalidResult);
        }
        if !nonempty(&result.calibration_receipt_id) {
            return Err(AsyncError::MissingReceipt("calibration_receipt_id"));
        }
        if !nonempty(&result.qualification_receipt_id) {
            return Err(AsyncError::MissingReceipt("qualification_receipt_id"));
        }
        if self.seen_result_ids.contains(&result.result_id) {
            return Err(AsyncError::DuplicateResult(result.result_id.clone()));
        }
        if self.actual_observations.contains_key(&result.observation_id) {
            return Err(AsyncError::DuplicateObservation(result.observation_id.clone()));
        }

        let job = self
            .active_jobs
            .get(job_id)
            .ok_or_else(|| AsyncError::UnknownJob(job_id.to_string()))?;
        if result.observed_at_utc_s < job.launched_at_utc_s {
            return Err(AsyncError::ObservationBeforeLaunch(job_id.to_string()));
        }
        if result.observed_at_utc_s > result.recorded_at_utc_s {
            return Err(AsyncError::ObservationAfterReceipt(job_id.to_string()));
        }

        let completed = job.clone();
        let new_spent = add_resources(&self.spent, &completed.proposal.resource_cost);
        new_spent
            .validate()
            .map_err(|_| AsyncError::InvalidResource)?;
        let new_reserved = subtract_resources(&self.reserved, &completed.proposal.resource_cost);

        let observation = ActualObservation {
            result_id: result.result_id.clone(),
            observation_id: result.observation_id.clone(),
            job_id: job_id.to_string(),
            candidate: completed.proposal.candidate.clone(),
            observed_at_utc_s: result.observed_at_utc_s,
            recorded_at_utc_s: result.recorded_at_utc_s,
            calibration_receipt_id: result.calibration_receipt_id,
            qualification_receipt_id: result.qualification_receipt_id,
            value_bits: result.value.to_bits(),
        };

        self.active_jobs.remove(job_id);
        self.reserved = new_reserved;
        self.spent = new_spent;
        self.seen_result_ids.insert(result.result_id.clone());
        self.actual_observations
            .insert(result.observation_id.clone(), observation);
        self.append_event(
            result.recorded_at_utc_s,
            AsyncEventKind::JobResolved {
                job_id: job_id.to_string(),
                result_id: result.result_id,
                observation_id: result.observation_id,
            },
        )
    }

    /// A failed experiment releases its locks but consumes the declared cost.
    pub fn fail_job(
        &mut self,
        job_id: &str,
        failure_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), AsyncError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(failure_receipt_id) {
            return Err(AsyncError::MissingReceipt("failure_receipt_id"));
        }
        let job = self
            .active_jobs
            .get(job_id)
            .ok_or_else(|| AsyncError::UnknownJob(job_id.to_string()))?;
        let failed = job.clone();
        let new_spent = add_resources(&self.spent, &failed.proposal.resource_cost);
        new_spent
            .validate()
            .map_err(|_| AsyncError::InvalidResource)?;
        let new_reserved = subtract_resources(&self.reserved, &failed.proposal.resource_cost);

        self.active_jobs.remove(job_id);
        self.spent = new_spent;
        self.reserved = new_reserved;
        self.append_event(
            recorded_at_utc_s,
            AsyncEventKind::JobFailed {
                job_id: job_id.to_string(),
                failure_receipt_id: failure_receipt_id.to_string(),
            },
        )
    }

    pub fn close(
        &mut self,
        closeout_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), AsyncError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(closeout_receipt_id) {
            return Err(AsyncError::MissingReceipt("closeout_receipt_id"));
        }
        if !self.active_jobs.is_empty() {
            return Err(AsyncError::ActiveJobsRemain);
        }
        self.closed = true;
        self.append_event(
            recorded_at_utc_s,
            AsyncEventKind::CampaignClosed {
                closeout_receipt_id: closeout_receipt_id.to_string(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use campaign_resources::{EnergyMeterBoundary, ResourceBasis};

    fn basis() -> ResourceBasis {
        ResourceBasis {
            energy_meter_boundary: EnergyMeterBoundary::ElectricalInput,
            assay_cost_unit: "USD".to_string(),
        }
    }

    fn resources(rig: f64, energy: f64, assay: f64) -> ResourceVector {
        ResourceVector {
            rig_hours: rig,
            source_energy_wh: energy,
            feedstock_g: 1.0,
            assay_cost: assay,
            maintenance_minutes: 2.0,
            consumables_g: 1.0,
        }
    }

    fn scheduler() -> AsyncScheduler {
        AsyncScheduler::new(CampaignBudget {
            resources: ResourceVector {
                rig_hours: 10.0,
                source_energy_wh: 1_000.0,
                feedstock_g: 100.0,
                assay_cost: 500.0,
                maintenance_minutes: 100.0,
                consumables_g: 20.0,
            },
            basis: basis(),
        })
        .unwrap()
    }

    fn proposal(job_id: &str, candidate_id: &str, lock: &str, cost: ResourceVector) -> JobProposal {
        JobProposal {
            job_id: job_id.to_string(),
            candidate: CandidateInstance {
                candidate_id: candidate_id.to_string(),
                replicate_id: None,
            },
            resource_locks: vec![lock.to_string()],
            resource_cost: cost,
            predicted_mean: 10.0,
            predicted_standard_deviation: 2.0,
        }
    }

    #[test]
    fn realistic_mean_placeholder_is_separate_from_actual_observations() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::RealisticMean,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();

        let fantasies = value.fantasies_for_model();
        assert_eq!(fantasies.len(), 1);
        assert_eq!(fantasies[0].value, 10.0);
        assert!(value.actual_observations.is_empty());
    }

    #[test]
    fn pessimistic_bound_respects_objective_direction() {
        let mut maximize = scheduler();
        maximize
            .launch_job(
                proposal("job-max", "candidate-max", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::PessimisticBound { beta: 2.0 },
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert_eq!(maximize.fantasies_for_model()[0].value, 6.0);

        let mut minimize = scheduler();
        minimize
            .launch_job(
                proposal("job-min", "candidate-min", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::PessimisticBound { beta: 2.0 },
                ObjectiveSense::Minimize,
                10,
                10,
            )
            .unwrap();
        assert_eq!(minimize.fantasies_for_model()[0].value, 14.0);
    }

    #[test]
    fn seeded_uniform_placeholder_replays_for_same_seed_and_launch_order() {
        let policy = PlaceholderPolicy::SeededUniform {
            seed: 77,
            radius_multiplier: 2.0,
        };
        let mut first = scheduler();
        let mut second = scheduler();
        let p = proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0));
        first
            .launch_job(p.clone(), policy, ObjectiveSense::Maximize, 10, 10)
            .unwrap();
        second
            .launch_job(p, policy, ObjectiveSense::Maximize, 10, 10)
            .unwrap();
        assert_eq!(
            first.fantasies_for_model()[0].value,
            second.fantasies_for_model()[0].value
        );
    }

    #[test]
    fn no_placeholder_policy_does_not_add_a_synthetic_model_row() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert!(value.fantasies_for_model().is_empty());
        assert!(value.actual_observations.is_empty());
    }

    #[test]
    fn exclusive_locks_prevent_overlapping_jobs_but_allow_distinct_instruments() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::RealisticMean,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        value
            .launch_job(
                proposal("job-2", "candidate-2", "spectrometer", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::RealisticMean,
                ObjectiveSense::Maximize,
                11,
                11,
            )
            .unwrap();
        assert!(matches!(
            value.launch_job(
                proposal("job-3", "candidate-3", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::RealisticMean,
                ObjectiveSense::Maximize,
                12,
                12
            ),
            Err(AsyncError::ResourceLockBusy { .. })
        ));
    }

    #[test]
    fn hard_resource_budget_is_reserved_before_job_launch() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(8.0, 900.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert!(matches!(
            value.launch_job(
                proposal("job-2", "candidate-2", "spectrometer", resources(3.0, 10.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                11,
                11
            ),
            Err(AsyncError::ResourceBudgetExceeded(over)) if over == vec!["rig_hours"]
        ));
        assert_eq!(value.active_candidate_count(), 1);
    }

    #[test]
    fn duplicate_inflight_candidate_is_rejected_unless_replicate_is_distinct() {
        let mut value = scheduler();
        let first = proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0));
        value
            .launch_job(
                first.clone(),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert!(matches!(
            value.launch_job(
                proposal("job-2", "candidate-1", "spectrometer", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                11,
                11
            ),
            Err(AsyncError::DuplicateInFlightCandidate(_))
        ));

        let mut replicate = proposal("job-3", "candidate-1", "spectrometer", resources(1.0, 100.0, 10.0));
        replicate.candidate.replicate_id = Some("replicate-2".to_string());
        value
            .launch_job(
                replicate,
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                12,
                12,
            )
            .unwrap();
        assert_eq!(value.active_candidate_count(), 2);
    }

    #[test]
    fn out_of_order_results_resolve_without_changing_actual_observation_times() {
        let mut value = scheduler();
        for (job, candidate, start, lock) in [
            ("job-1", "candidate-1", 10, "reactor"),
            ("job-2", "candidate-2", 11, "spectrometer"),
        ] {
            value
                .launch_job(
                    proposal(job, candidate, lock, resources(1.0, 100.0, 10.0)),
                    PlaceholderPolicy::PessimisticBound { beta: 1.0 },
                    ObjectiveSense::Maximize,
                    start,
                    start,
                )
                .unwrap();
        }

        value
            .resolve_job(
                "job-2",
                ActualResult {
                    result_id: "result-2".to_string(),
                    observation_id: "observation-2".to_string(),
                    value: 8.5,
                    observed_at_utc_s: 12,
                    calibration_receipt_id: "calibration-2".to_string(),
                    qualification_receipt_id: "qualification-2".to_string(),
                    recorded_at_utc_s: 12,
                },
            )
            .unwrap();
        value
            .resolve_job(
                "job-1",
                ActualResult {
                    result_id: "result-1".to_string(),
                    observation_id: "observation-1".to_string(),
                    value: 9.5,
                    observed_at_utc_s: 11,
                    calibration_receipt_id: "calibration-1".to_string(),
                    qualification_receipt_id: "qualification-1".to_string(),
                    recorded_at_utc_s: 13,
                },
            )
            .unwrap();

        assert_eq!(value.actual_observations["observation-1"].observed_at_utc_s, 11);
        assert_eq!(value.actual_observations["observation-2"].observed_at_utc_s, 12);
        assert!(value.fantasies_for_model().is_empty());
        assert_eq!(value.reserved.rig_hours, 0.0);
        assert_eq!(value.spent.rig_hours, 2.0);
    }

    #[test]
    fn failed_jobs_release_locks_but_consume_resource_budget() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(2.0, 200.0, 20.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        value.fail_job("job-1", "failure-receipt", 12).unwrap();

        assert_eq!(value.active_candidate_count(), 0);
        assert_eq!(value.spent.rig_hours, 2.0);
        assert_eq!(value.reserved.rig_hours, 0.0);
        assert_eq!(value.remaining_budget().rig_hours, 8.0);

        value
            .launch_job(
                proposal("job-2", "candidate-2", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                13,
                13,
            )
            .unwrap();
    }

    #[test]
    fn pending_jobs_block_close_and_closed_scheduler_rejects_new_work() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert_eq!(
            value.close("closeout", 11),
            Err(AsyncError::ActiveJobsRemain)
        );
        value.fail_job("job-1", "failure-receipt", 12).unwrap();
        value.close("closeout", 13).unwrap();
        assert_eq!(
            value.launch_job(
                proposal("job-2", "candidate-2", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                14,
                14
            ),
            Err(AsyncError::CampaignClosed)
        );
    }

    #[test]
    fn actual_result_requires_qualified_receipts_and_cannot_resolve_before_launch() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();

        assert_eq!(
            value.resolve_job(
                "job-1",
                ActualResult {
                    result_id: "result-1".to_string(),
                    observation_id: "obs-1".to_string(),
                    value: 1.0,
                    observed_at_utc_s: 9,
                    calibration_receipt_id: "cal".to_string(),
                    qualification_receipt_id: "qual".to_string(),
                    recorded_at_utc_s: 11,
                },
            ),
            Err(AsyncError::ObservationBeforeLaunch("job-1".to_string()))
        );
        assert_eq!(
            value.resolve_job(
                "job-1",
                ActualResult {
                    result_id: "result-1".to_string(),
                    observation_id: "obs-1".to_string(),
                    value: 1.0,
                    observed_at_utc_s: 11,
                    calibration_receipt_id: String::new(),
                    qualification_receipt_id: "qual".to_string(),
                    recorded_at_utc_s: 12,
                },
            ),
            Err(AsyncError::MissingReceipt("calibration_receipt_id"))
        );
        assert_eq!(value.active_candidate_count(), 1);
        assert!(value.actual_observations.is_empty());
    }

    #[test]
    fn transition_time_rejection_is_atomic() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::NoPlaceholder,
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        let before = value.clone();
        assert_eq!(
            value.fail_job("job-1", "failure", 9),
            Err(AsyncError::EventTimeWentBackwards)
        );
        assert_eq!(value, before);
    }

    #[test]
    fn seeded_policy_rejects_zero_seed_and_invalid_numeric_parameters() {
        let mut value = scheduler();
        assert_eq!(
            value.launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::SeededUniform {
                    seed: 0,
                    radius_multiplier: 1.0,
                },
                ObjectiveSense::Maximize,
                10,
                10
            ),
            Err(AsyncError::InvalidPlaceholderPolicy)
        );
        assert_eq!(value.active_candidate_count(), 0);
    }

    #[test]
    fn actual_observation_value_is_not_the_placeholder_value() {
        let mut value = scheduler();
        value
            .launch_job(
                proposal("job-1", "candidate-1", "reactor", resources(1.0, 100.0, 10.0)),
                PlaceholderPolicy::PessimisticBound { beta: 2.0 },
                ObjectiveSense::Maximize,
                10,
                10,
            )
            .unwrap();
        assert_eq!(value.fantasies_for_model()[0].value, 6.0);
        value
            .resolve_job(
                "job-1",
                ActualResult {
                    result_id: "result-1".to_string(),
                    observation_id: "observation-1".to_string(),
                    value: 12.5,
                    observed_at_utc_s: 11,
                    calibration_receipt_id: "cal".to_string(),
                    qualification_receipt_id: "qual".to_string(),
                    recorded_at_utc_s: 12,
                },
            )
            .unwrap();
        assert_eq!(value.actual_observations["observation-1"].value(), 12.5);
        assert!(value.fantasies_for_model().is_empty());
    }
}

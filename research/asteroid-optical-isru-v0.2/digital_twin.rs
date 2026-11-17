// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// B1 stateful deterministic-seed optical-ISRU digital twin.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/digital_twin.rs -o /tmp/isru-digital-twin
//   /tmp/isru-digital-twin

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

pub const SIMULATION_ONLY: bool = true;
const STAGE_COUNT: usize = 4;
type CandidateMap = BTreeMap<String, CandidateDesign>;
type CandidateTruthMap = BTreeMap<String, CandidateTruth>;
const PPM_SCALE: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StageKind {
    Characterization,
    Exposure,
    Capture,
    Assay,
}

impl StageKind {
    fn index(self) -> usize {
        match self {
            Self::Characterization => 0,
            Self::Exposure => 1,
            Self::Capture => 2,
            Self::Assay => 3,
        }
    }

    fn output_kind(self) -> SimulatedOutputKind {
        match self {
            Self::Characterization | Self::Exposure => SimulatedOutputKind::IntermediateProxy,
            Self::Capture => SimulatedOutputKind::CapturedProduct,
            Self::Assay => SimulatedOutputKind::SimulatedIndependentAssay,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateDesign {
    pub candidate_id: String,
    pub features: Vec<f64>,
    pub parameter_digest: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateTruth {
    /// Simulation-only expected values for characterization, exposure,
    /// capture and assay. These values are not present in the agent API.
    pub stage_means: [f64; STAGE_COUNT],
    pub final_utility: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSeed {
    /// Custodian-side construction input. Never returned by the agent port.
    pub candidate: CandidateDesign,
    pub truth: CandidateTruth,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ResourceCost {
    pub rig_seconds: f64,
    pub source_energy_wh: f64,
    pub feedstock_g: f64,
    pub assay_cost: f64,
    pub maintenance_minutes: f64,
    pub consumables_g: f64,
}

impl ResourceCost {
    fn validate(&self) -> Result<(), TwinError> {
        for (name, value) in [
            ("rig_seconds", self.rig_seconds),
            ("source_energy_wh", self.source_energy_wh),
            ("feedstock_g", self.feedstock_g),
            ("assay_cost", self.assay_cost),
            ("maintenance_minutes", self.maintenance_minutes),
            ("consumables_g", self.consumables_g),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(TwinError::InvalidResource(name));
            }
        }
        Ok(())
    }
}

fn add_cost(left: ResourceCost, right: ResourceCost) -> ResourceCost {
    ResourceCost {
        rig_seconds: left.rig_seconds + right.rig_seconds,
        source_energy_wh: left.source_energy_wh + right.source_energy_wh,
        feedstock_g: left.feedstock_g + right.feedstock_g,
        assay_cost: left.assay_cost + right.assay_cost,
        maintenance_minutes: left.maintenance_minutes + right.maintenance_minutes,
        consumables_g: left.consumables_g + right.consumables_g,
    }
}

fn subtract_cost(left: ResourceCost, right: ResourceCost) -> ResourceCost {
    ResourceCost {
        rig_seconds: left.rig_seconds - right.rig_seconds,
        source_energy_wh: left.source_energy_wh - right.source_energy_wh,
        feedstock_g: left.feedstock_g - right.feedstock_g,
        assay_cost: left.assay_cost - right.assay_cost,
        maintenance_minutes: left.maintenance_minutes - right.maintenance_minutes,
        consumables_g: left.consumables_g - right.consumables_g,
    }
}

fn cost_fits(used: ResourceCost, requested: ResourceCost, budget: ResourceCost) -> bool {
    let total = add_cost(used, requested);
    total.rig_seconds <= budget.rig_seconds
        && total.source_energy_wh <= budget.source_energy_wh
        && total.feedstock_g <= budget.feedstock_g
        && total.assay_cost <= budget.assay_cost
        && total.maintenance_minutes <= budget.maintenance_minutes
        && total.consumables_g <= budget.consumables_g
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwinManifest {
    pub dataset_id: String,
    pub task_version: String,
    pub scenario_digest: String,
    pub seed: u64,
    pub max_experiments: u64,
    pub max_wall_seconds: u64,
    pub resource_budget: ResourceCost,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageSpec {
    pub stage: StageKind,
    pub eligible_instruments: Vec<String>,
    pub base_duration_seconds: u64,
    pub duration_jitter_seconds: u64,
    pub measurement_noise_stddev: f64,
    pub fouling_increment_ppm: u64,
    /// Simulated value bias per ppm of fouling present at launch.
    pub fouling_bias_per_ppm: f64,
    pub resource_cost: ResourceCost,
    pub failure_probability_ppm: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct InstrumentSpec {
    pub instrument_id: String,
    /// Number of operations allowed after calibration before requalification.
    pub calibration_valid_for_jobs: u64,
    pub fouling_threshold_ppm: u64,
    /// Scale factor applied to the stage's nominal duration.
    pub duration_scale_permille: u64,
    pub maintenance_duration_seconds: u64,
    pub recalibration_duration_seconds: u64,
    pub maintenance_cost: ResourceCost,
    pub recalibration_cost: ResourceCost,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstrumentCondition {
    Available,
    Busy,
    CalibrationDue,
    MaintenanceDue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SimulatedOutputKind {
    IntermediateProxy,
    CapturedProduct,
    SimulatedIndependentAssay,
}

/// A B1 output is a synthetic datum by construction. It has no calibration,
/// custody, qualification or authority receipt fields and cannot be passed as
/// an actual Symtropy observation without an explicit, invalid conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct SimulatedObservation {
    pub candidate_id: String,
    pub job_id: String,
    pub instrument_id: String,
    pub stage: StageKind,
    pub output_kind: SimulatedOutputKind,
    pub simulated_value: f64,
    pub completed_at_seconds: u64,
    pub fouling_at_launch_ppm: u64,
    pub simulation_only: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TwinTransition {
    Completed(SimulatedObservation),
    Failed {
        job_id: String,
        candidate_id: String,
        stage: StageKind,
        completed_at_seconds: u64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub enum TwinEventKind {
    JobLaunched {
        job_id: String,
        candidate_id: String,
        instrument_id: String,
        stage: StageKind,
        planned_completion_seconds: u64,
    },
    JobCompleted {
        job_id: String,
        candidate_id: String,
        stage: StageKind,
    },
    JobFailed {
        job_id: String,
        candidate_id: String,
        stage: StageKind,
    },
    InstrumentMaintained {
        instrument_id: String,
    },
    InstrumentRecalibrated {
        instrument_id: String,
    },
    CampaignClosed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwinEvent {
    pub sequence: u64,
    pub at_seconds: u64,
    pub kind: TwinEventKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwinPrediction {
    pub candidate_id: String,
    pub predicted_mean: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwinDiscoveryScore {
    pub best_simulated_assay: Option<f64>,
    pub candidates_assayed: u64,
    pub experiments_started: u64,
    pub elapsed_seconds: u64,
    pub threshold: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TwinLearningScore {
    pub spearman_rank_correlation: Option<f64>,
    pub interval_coverage: f64,
    pub holdout_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwinError {
    EmptyIdentity(&'static str),
    InvalidManifest(&'static str),
    InvalidResource(&'static str),
    InvalidStageSpec(StageKind),
    InvalidInstrumentSpec(String),
    DuplicateStage(StageKind),
    DuplicateInstrument(String),
    DuplicateCandidate(String),
    DuplicateJob(String),
    CandidateOutcomeMismatch(String),
    CandidateOverlap(String),
    TooFewHoldoutCandidates,
    UnknownCandidate(String),
    HoldoutCannotBeMeasured(String),
    InvalidStageOrder {
        candidate_id: String,
        expected: Option<StageKind>,
        requested: StageKind,
    },
    CandidateAlreadyActive(String),
    InstrumentBusy(String),
    UnknownInstrument(String),
    InstrumentNotEligible {
        instrument_id: String,
        stage: StageKind,
    },
    CalibrationDue(String),
    MaintenanceDue(String),
    InstrumentNotDirty(String),
    CalibrationStillValid(String),
    ResourceBudgetExceeded,
    ExperimentBudgetExhausted,
    TimeBudgetExceeded,
    NoActiveJobs,
    CampaignClosed,
    ActiveJobsRemain,
    CampaignNotClosed,
    InvalidThreshold,
    InvalidPrediction(String),
    DuplicatePrediction(String),
    UnexpectedHoldoutPrediction(String),
    MissingHoldoutPrediction(String),
}

#[derive(Clone, Debug)]
struct InstrumentState {
    spec: InstrumentSpec,
    jobs_since_calibration: u64,
    fouling_ppm: u64,
    busy_job_id: Option<String>,
}

#[derive(Clone, Debug)]
struct CandidateProgress {
    next_stage_index: usize,
}

#[derive(Clone, Debug)]
struct InFlightJob {
    job_id: String,
    candidate_id: String,
    instrument_id: String,
    stage: StageKind,
    started_at_seconds: u64,
    finish_at_seconds: u64,
    launch_sequence: u64,
    will_fail: bool,
    fouling_at_launch_ppm: u64,
    reserved_cost: ResourceCost,
    actual_non_time_cost: ResourceCost,
}

pub struct TwinWorld {
    manifest: TwinManifest,
    stages: BTreeMap<StageKind, StageSpec>,
    instruments: BTreeMap<String, InstrumentState>,
    measurable_candidates: BTreeMap<String, CandidateDesign>,
    holdout_candidates: BTreeMap<String, CandidateDesign>,
    sealed_truth: BTreeMap<String, CandidateTruth>,
    progress: BTreeMap<String, CandidateProgress>,
    attempts: BTreeMap<(String, StageKind), u64>,
    seen_job_ids: BTreeSet<String>,
    started_candidates: BTreeSet<String>,
    active_jobs: BTreeMap<String, InFlightJob>,
    completed_observations: Vec<SimulatedObservation>,
    events: Vec<TwinEvent>,
    now_seconds: u64,
    spent: ResourceCost,
    reserved: ResourceCost,
    closed: bool,
}

/// The B1 agent port intentionally exposes catalog metadata and operations
/// only. Sealed candidate truth and holdout outcomes remain evaluator-side.
pub struct TwinAgentApi<'a> {
    world: &'a mut TwinWorld,
}

impl TwinWorld {
    pub fn new(
        manifest: TwinManifest,
        stage_specs: Vec<StageSpec>,
        instrument_specs: Vec<InstrumentSpec>,
        measurable: Vec<CandidateSeed>,
        holdout: Vec<CandidateSeed>,
    ) -> Result<Self, TwinError> {
        if manifest.dataset_id.trim().is_empty() {
            return Err(TwinError::EmptyIdentity("dataset_id"));
        }
        if manifest.task_version.trim().is_empty() {
            return Err(TwinError::EmptyIdentity("task_version"));
        }
        if manifest.scenario_digest.trim().is_empty() {
            return Err(TwinError::EmptyIdentity("scenario_digest"));
        }
        if manifest.max_experiments == 0 {
            return Err(TwinError::InvalidManifest("max_experiments"));
        }
        if manifest.max_wall_seconds == 0 {
            return Err(TwinError::InvalidManifest("max_wall_seconds"));
        }
        manifest.resource_budget.validate()?;

        let mut stages = BTreeMap::new();
        for stage in stage_specs {
            if stage.eligible_instruments.is_empty()
                || stage.base_duration_seconds == 0
                || !stage.measurement_noise_stddev.is_finite()
                || stage.measurement_noise_stddev < 0.0
                || !stage.fouling_bias_per_ppm.is_finite()
                || stage.fouling_bias_per_ppm < 0.0
                || stage.failure_probability_ppm > PPM_SCALE
            {
                return Err(TwinError::InvalidStageSpec(stage.stage));
            }
            stage.resource_cost.validate()?;
            let mut eligible = BTreeSet::new();
            if stage
                .eligible_instruments
                .iter()
                .any(|id| id.trim().is_empty() || !eligible.insert(id))
            {
                return Err(TwinError::InvalidStageSpec(stage.stage));
            }
            if stages.insert(stage.stage, stage.clone()).is_some() {
                return Err(TwinError::DuplicateStage(stage.stage));
            }
        }
        if stages.len() != STAGE_COUNT
            || !(0..STAGE_COUNT).all(|index| stages.contains_key(&stage_from_index(index)))
        {
            return Err(TwinError::InvalidManifest("stage_graph"));
        }

        let mut instruments = BTreeMap::new();
        for spec in instrument_specs {
            if spec.instrument_id.trim().is_empty()
                || spec.calibration_valid_for_jobs == 0
                || spec.fouling_threshold_ppm == 0
                || spec.duration_scale_permille == 0
                || spec.maintenance_duration_seconds == 0
                || spec.recalibration_duration_seconds == 0
            {
                return Err(TwinError::InvalidInstrumentSpec(spec.instrument_id.clone()));
            }
            spec.maintenance_cost.validate()?;
            spec.recalibration_cost.validate()?;
            let id = spec.instrument_id.clone();
            if instruments
                .insert(
                    id.clone(),
                    InstrumentState {
                        spec,
                        jobs_since_calibration: 0,
                        fouling_ppm: 0,
                        busy_job_id: None,
                    },
                )
                .is_some()
            {
                return Err(TwinError::DuplicateInstrument(id));
            }
        }

        for stage in stages.values() {
            for instrument_id in &stage.eligible_instruments {
                if !instruments.contains_key(instrument_id) {
                    return Err(TwinError::UnknownInstrument(instrument_id.clone()));
                }
            }
        }

        let (measurable_candidates, measurable_truth) = split_seeds(measurable)?;
        let (holdout_candidates, holdout_truth) = split_seeds(holdout)?;
        if holdout_candidates.len() < 3 {
            return Err(TwinError::TooFewHoldoutCandidates);
        }
        for id in measurable_candidates.keys() {
            if holdout_candidates.contains_key(id) {
                return Err(TwinError::CandidateOverlap(id.clone()));
            }
        }

        let mut sealed_truth = measurable_truth;
        sealed_truth.extend(holdout_truth);

        let progress = measurable_candidates
            .keys()
            .map(|id| {
                (
                    id.clone(),
                    CandidateProgress {
                        next_stage_index: 0,
                    },
                )
            })
            .collect();

        Ok(Self {
            manifest,
            stages,
            instruments,
            measurable_candidates,
            holdout_candidates,
            sealed_truth,
            progress,
            attempts: BTreeMap::new(),
            seen_job_ids: BTreeSet::new(),
            started_candidates: BTreeSet::new(),
            active_jobs: BTreeMap::new(),
            completed_observations: Vec::new(),
            events: Vec::new(),
            now_seconds: 0,
            spent: ResourceCost::default(),
            reserved: ResourceCost::default(),
            closed: false,
        })
    }

    pub fn agent_api(&mut self) -> TwinAgentApi<'_> {
        TwinAgentApi { world: self }
    }

    fn ensure_open(&self) -> Result<(), TwinError> {
        if self.closed {
            Err(TwinError::CampaignClosed)
        } else {
            Ok(())
        }
    }

    fn instrument_condition(&self, instrument_id: &str) -> Result<InstrumentCondition, TwinError> {
        let instrument = self
            .instruments
            .get(instrument_id)
            .ok_or_else(|| TwinError::UnknownInstrument(instrument_id.to_string()))?;
        if instrument.busy_job_id.is_some() {
            Ok(InstrumentCondition::Busy)
        } else if instrument.jobs_since_calibration >= instrument.spec.calibration_valid_for_jobs {
            Ok(InstrumentCondition::CalibrationDue)
        } else if instrument.fouling_ppm >= instrument.spec.fouling_threshold_ppm {
            Ok(InstrumentCondition::MaintenanceDue)
        } else {
            Ok(InstrumentCondition::Available)
        }
    }

    fn append_event(&mut self, kind: TwinEventKind) {
        self.events.push(TwinEvent {
            sequence: self.events.len() as u64,
            at_seconds: self.now_seconds,
            kind,
        });
    }

    fn launch_stage(
        &mut self,
        job_id: &str,
        candidate_id: &str,
        stage: StageKind,
        instrument_id: &str,
    ) -> Result<u64, TwinError> {
        self.ensure_open()?;
        if job_id.trim().is_empty() {
            return Err(TwinError::EmptyIdentity("job_id"));
        }
        if self.seen_job_ids.contains(job_id) {
            return Err(TwinError::DuplicateJob(job_id.to_string()));
        }
        let progress = self.progress.get(candidate_id).ok_or_else(|| {
            if self.holdout_candidates.contains_key(candidate_id) {
                TwinError::HoldoutCannotBeMeasured(candidate_id.to_string())
            } else {
                TwinError::UnknownCandidate(candidate_id.to_string())
            }
        })?;
        let expected = (progress.next_stage_index < STAGE_COUNT)
            .then_some(stage_from_index(progress.next_stage_index));
        if expected != Some(stage) {
            return Err(TwinError::InvalidStageOrder {
                candidate_id: candidate_id.to_string(),
                expected,
                requested: stage,
            });
        }
        if self
            .active_jobs
            .values()
            .any(|active| active.candidate_id == candidate_id)
        {
            return Err(TwinError::CandidateAlreadyActive(candidate_id.to_string()));
        }
        let stage_spec = self
            .stages
            .get(&stage)
            .ok_or(TwinError::InvalidStageSpec(stage))?
            .clone();
        if !stage_spec
            .eligible_instruments
            .iter()
            .any(|eligible| eligible == instrument_id)
        {
            return Err(TwinError::InstrumentNotEligible {
                instrument_id: instrument_id.to_string(),
                stage,
            });
        }
        match self.instrument_condition(instrument_id)? {
            InstrumentCondition::Available => {}
            InstrumentCondition::Busy => {
                return Err(TwinError::InstrumentBusy(instrument_id.to_string()));
            }
            InstrumentCondition::CalibrationDue => {
                return Err(TwinError::CalibrationDue(instrument_id.to_string()));
            }
            InstrumentCondition::MaintenanceDue => {
                return Err(TwinError::MaintenanceDue(instrument_id.to_string()));
            }
        }

        let is_first_stage = stage == StageKind::Characterization;
        let is_new_experiment = is_first_stage && !self.started_candidates.contains(candidate_id);
        if is_new_experiment
            && self.started_candidates.len() as u64 >= self.manifest.max_experiments
        {
            return Err(TwinError::ExperimentBudgetExhausted);
        }

        let instrument = self
            .instruments
            .get(instrument_id)
            .expect("instrument presence checked");
        let max_unscaled = stage_spec
            .base_duration_seconds
            .saturating_add(stage_spec.duration_jitter_seconds);
        let max_duration = scaled_duration(max_unscaled, instrument.spec.duration_scale_permille);
        let key_attempt = *self
            .attempts
            .get(&(candidate_id.to_string(), stage))
            .unwrap_or(&0);
        let rng_seed = keyed_seed(self.manifest.seed, candidate_id, stage, key_attempt);
        let (duration, will_fail) = sample_job(
            rng_seed,
            &stage_spec,
            instrument.spec.duration_scale_permille,
        );
        let finish_at = self.now_seconds.saturating_add(duration);
        if finish_at > self.manifest.max_wall_seconds
            || self.now_seconds.saturating_add(max_duration) > self.manifest.max_wall_seconds
        {
            return Err(TwinError::TimeBudgetExceeded);
        }

        let mut reserved_cost = stage_spec.resource_cost;
        reserved_cost.rig_seconds = max_duration as f64;
        let mut actual_non_time_cost = stage_spec.resource_cost;
        actual_non_time_cost.rig_seconds = 0.0;
        let committed = add_cost(self.spent, self.reserved);
        if !cost_fits(committed, reserved_cost, self.manifest.resource_budget) {
            return Err(TwinError::ResourceBudgetExceeded);
        }

        let instrument_fouling = instrument.fouling_ppm;
        let launch_sequence = self.events.len() as u64;
        let job = InFlightJob {
            job_id: job_id.to_string(),
            candidate_id: candidate_id.to_string(),
            instrument_id: instrument_id.to_string(),
            stage,
            started_at_seconds: self.now_seconds,
            finish_at_seconds: finish_at,
            launch_sequence,
            will_fail,
            fouling_at_launch_ppm: instrument_fouling,
            reserved_cost,
            actual_non_time_cost,
        };

        self.reserved = add_cost(self.reserved, reserved_cost);
        self.seen_job_ids.insert(job_id.to_string());
        self.attempts
            .insert((candidate_id.to_string(), stage), key_attempt + 1);
        if is_new_experiment {
            self.started_candidates.insert(candidate_id.to_string());
        }
        let instrument_state = self
            .instruments
            .get_mut(instrument_id)
            .expect("instrument presence checked");
        instrument_state.busy_job_id = Some(job_id.to_string());
        instrument_state.jobs_since_calibration =
            instrument_state.jobs_since_calibration.saturating_add(1);
        self.active_jobs.insert(job_id.to_string(), job);
        self.append_event(TwinEventKind::JobLaunched {
            job_id: job_id.to_string(),
            candidate_id: candidate_id.to_string(),
            instrument_id: instrument_id.to_string(),
            stage,
            planned_completion_seconds: finish_at,
        });
        Ok(finish_at)
    }

    fn advance_next_event(&mut self) -> Result<Option<TwinTransition>, TwinError> {
        self.ensure_open()?;
        let next = self
            .active_jobs
            .values()
            .min_by_key(|job| (job.finish_at_seconds, job.launch_sequence))
            .cloned();
        let Some(job) = next else {
            return Ok(None);
        };
        self.now_seconds = job.finish_at_seconds;

        self.active_jobs
            .remove(&job.job_id)
            .expect("selected active job exists");
        let instrument = self
            .instruments
            .get_mut(&job.instrument_id)
            .expect("active job instrument exists");
        instrument.busy_job_id = None;
        instrument.fouling_ppm = instrument
            .fouling_ppm
            .saturating_add(fouling_added(&job, &self.stages));

        let duration = job.finish_at_seconds.saturating_sub(job.started_at_seconds);
        let mut actual_cost = job.actual_non_time_cost;
        actual_cost.rig_seconds = duration as f64;
        self.reserved = subtract_cost(self.reserved, job.reserved_cost);
        self.spent = add_cost(self.spent, actual_cost);

        if job.will_fail {
            self.append_event(TwinEventKind::JobFailed {
                job_id: job.job_id.clone(),
                candidate_id: job.candidate_id.clone(),
                stage: job.stage,
            });
            return Ok(Some(TwinTransition::Failed {
                job_id: job.job_id,
                candidate_id: job.candidate_id,
                stage: job.stage,
                completed_at_seconds: self.now_seconds,
            }));
        }

        let truth = self
            .sealed_truth
            .get(&job.candidate_id)
            .expect("candidate truth fixed at construction");
        let base = if job.stage == StageKind::Assay {
            truth.final_utility
        } else {
            truth.stage_means[job.stage.index()]
        };
        let stage_spec = self.stages.get(&job.stage).expect("stage exists");
        let noise_seed = keyed_seed(
            self.manifest.seed ^ 0xA5A5_A5A5_A5A5_A5A5,
            &job.candidate_id,
            job.stage,
            self.attempts
                .get(&(job.candidate_id.clone(), job.stage))
                .copied()
                .unwrap_or(1)
                .saturating_sub(1),
        );
        let noise = sample_noise(noise_seed, stage_spec.measurement_noise_stddev);
        let fouling_bias = job.fouling_at_launch_ppm as f64 * stage_spec.fouling_bias_per_ppm;
        let observation = SimulatedObservation {
            candidate_id: job.candidate_id.clone(),
            job_id: job.job_id.clone(),
            instrument_id: job.instrument_id.clone(),
            stage: job.stage,
            output_kind: job.stage.output_kind(),
            simulated_value: base + noise - fouling_bias,
            completed_at_seconds: self.now_seconds,
            fouling_at_launch_ppm: job.fouling_at_launch_ppm,
            simulation_only: true,
        };
        self.progress
            .get_mut(&job.candidate_id)
            .expect("candidate progress exists")
            .next_stage_index += 1;
        self.completed_observations.push(observation.clone());
        self.append_event(TwinEventKind::JobCompleted {
            job_id: job.job_id,
            candidate_id: job.candidate_id,
            stage: job.stage,
        });
        Ok(Some(TwinTransition::Completed(observation)))
    }

    fn operate_instrument(
        &mut self,
        instrument_id: &str,
        maintenance: bool,
    ) -> Result<(), TwinError> {
        self.ensure_open()?;
        let instrument = self
            .instruments
            .get(instrument_id)
            .ok_or_else(|| TwinError::UnknownInstrument(instrument_id.to_string()))?
            .clone();
        if instrument.busy_job_id.is_some() {
            return Err(TwinError::InstrumentBusy(instrument_id.to_string()));
        }
        if !self.active_jobs.is_empty() {
            return Err(TwinError::ActiveJobsRemain);
        }

        let (duration, mut cost) = if maintenance {
            if instrument.fouling_ppm < instrument.spec.fouling_threshold_ppm {
                return Err(TwinError::InstrumentNotDirty(instrument_id.to_string()));
            }
            (
                instrument.spec.maintenance_duration_seconds,
                instrument.spec.maintenance_cost,
            )
        } else {
            if instrument.jobs_since_calibration < instrument.spec.calibration_valid_for_jobs {
                return Err(TwinError::CalibrationStillValid(instrument_id.to_string()));
            }
            (
                instrument.spec.recalibration_duration_seconds,
                instrument.spec.recalibration_cost,
            )
        };
        let finish = self.now_seconds.saturating_add(duration);
        if finish > self.manifest.max_wall_seconds {
            return Err(TwinError::TimeBudgetExceeded);
        }
        cost.rig_seconds = duration as f64;
        let committed = add_cost(self.spent, self.reserved);
        if !cost_fits(committed, cost, self.manifest.resource_budget) {
            return Err(TwinError::ResourceBudgetExceeded);
        }
        self.now_seconds = finish;
        self.spent = add_cost(self.spent, cost);
        let state = self
            .instruments
            .get_mut(instrument_id)
            .expect("instrument presence checked");
        if maintenance {
            state.fouling_ppm = 0;
        } else {
            state.jobs_since_calibration = 0;
        }
        self.append_event(if maintenance {
            TwinEventKind::InstrumentMaintained {
                instrument_id: instrument_id.to_string(),
            }
        } else {
            TwinEventKind::InstrumentRecalibrated {
                instrument_id: instrument_id.to_string(),
            }
        });
        Ok(())
    }

    pub fn grade_discovery(&self, threshold: f64) -> Result<TwinDiscoveryScore, TwinError> {
        if !self.closed {
            return Err(TwinError::CampaignNotClosed);
        }
        if !threshold.is_finite() {
            return Err(TwinError::InvalidThreshold);
        }
        let assay_values: Vec<f64> = self
            .completed_observations
            .iter()
            .filter(|obs| obs.stage == StageKind::Assay)
            .map(|obs| obs.simulated_value)
            .collect();
        let best = assay_values.iter().copied().reduce(f64::max);
        Ok(TwinDiscoveryScore {
            best_simulated_assay: best,
            candidates_assayed: assay_values.len() as u64,
            experiments_started: self.started_candidates.len() as u64,
            elapsed_seconds: self.now_seconds,
            threshold,
            passed: best.is_some_and(|value| value >= threshold),
        })
    }

    pub fn grade_learning(
        &self,
        predictions: Vec<TwinPrediction>,
    ) -> Result<TwinLearningScore, TwinError> {
        if !self.closed {
            return Err(TwinError::CampaignNotClosed);
        }
        let mut by_id = BTreeMap::new();
        for prediction in predictions {
            if prediction.candidate_id.trim().is_empty()
                || !prediction.predicted_mean.is_finite()
                || !prediction.interval_lower.is_finite()
                || !prediction.interval_upper.is_finite()
                || prediction.interval_lower > prediction.interval_upper
            {
                return Err(TwinError::InvalidPrediction(prediction.candidate_id));
            }
            let id = prediction.candidate_id.clone();
            if !self.holdout_candidates.contains_key(&id) {
                return Err(TwinError::UnexpectedHoldoutPrediction(id));
            }
            if by_id.insert(id.clone(), prediction).is_some() {
                return Err(TwinError::DuplicatePrediction(id));
            }
        }
        for id in self.holdout_candidates.keys() {
            if !by_id.contains_key(id) {
                return Err(TwinError::MissingHoldoutPrediction(id.clone()));
            }
        }

        let mut predicted = Vec::new();
        let mut actual = Vec::new();
        let mut covered = 0_usize;
        for id in self.holdout_candidates.keys() {
            let prediction = by_id.get(id).expect("complete predictions checked");
            let truth = self.sealed_truth.get(id).expect("holdout truth exists");
            predicted.push(prediction.predicted_mean);
            actual.push(truth.final_utility);
            if truth.final_utility >= prediction.interval_lower
                && truth.final_utility <= prediction.interval_upper
            {
                covered += 1;
            }
        }
        Ok(TwinLearningScore {
            spearman_rank_correlation: spearman(&predicted, &actual),
            interval_coverage: covered as f64 / actual.len() as f64,
            holdout_count: actual.len(),
        })
    }

    pub fn events(&self) -> &[TwinEvent] {
        &self.events
    }

    pub fn spent_resources(&self) -> ResourceCost {
        self.spent
    }

    pub fn reserved_resources(&self) -> ResourceCost {
        self.reserved
    }

    pub fn elapsed_seconds(&self) -> u64 {
        self.now_seconds
    }
}

impl TwinAgentApi<'_> {
    pub fn manifest(&self) -> &TwinManifest {
        &self.world.manifest
    }

    pub fn measurable_catalog(&self) -> Vec<&CandidateDesign> {
        self.world.measurable_candidates.values().collect()
    }

    pub fn holdout_catalog(&self) -> Vec<&CandidateDesign> {
        self.world.holdout_candidates.values().collect()
    }

    pub fn next_stage(&self, candidate_id: &str) -> Result<Option<StageKind>, TwinError> {
        let progress = self.world.progress.get(candidate_id).ok_or_else(|| {
            if self.world.holdout_candidates.contains_key(candidate_id) {
                TwinError::HoldoutCannotBeMeasured(candidate_id.to_string())
            } else {
                TwinError::UnknownCandidate(candidate_id.to_string())
            }
        })?;
        Ok((progress.next_stage_index < STAGE_COUNT)
            .then_some(stage_from_index(progress.next_stage_index)))
    }

    pub fn instrument_condition(
        &self,
        instrument_id: &str,
    ) -> Result<InstrumentCondition, TwinError> {
        self.world.instrument_condition(instrument_id)
    }

    pub fn launch_stage(
        &mut self,
        job_id: &str,
        candidate_id: &str,
        stage: StageKind,
        instrument_id: &str,
    ) -> Result<u64, TwinError> {
        self.world
            .launch_stage(job_id, candidate_id, stage, instrument_id)
    }

    pub fn advance_next_event(&mut self) -> Result<Option<TwinTransition>, TwinError> {
        self.world.advance_next_event()
    }

    pub fn perform_maintenance(&mut self, instrument_id: &str) -> Result<(), TwinError> {
        self.world.operate_instrument(instrument_id, true)
    }

    pub fn recalibrate(&mut self, instrument_id: &str) -> Result<(), TwinError> {
        self.world.operate_instrument(instrument_id, false)
    }

    pub fn close_campaign(&mut self) -> Result<(), TwinError> {
        if self.world.closed {
            return Err(TwinError::CampaignClosed);
        }
        if !self.world.active_jobs.is_empty() {
            return Err(TwinError::ActiveJobsRemain);
        }
        self.world.closed = true;
        self.world.append_event(TwinEventKind::CampaignClosed);
        Ok(())
    }
}

fn stage_from_index(index: usize) -> StageKind {
    match index {
        0 => StageKind::Characterization,
        1 => StageKind::Exposure,
        2 => StageKind::Capture,
        _ => StageKind::Assay,
    }
}

fn scaled_duration(seconds: u64, scale_permille: u64) -> u64 {
    ((seconds as u128 * scale_permille as u128).saturating_add(999) / 1000)
        .max(1)
        .min(u64::MAX as u128) as u64
}

fn split_seeds(seeds: Vec<CandidateSeed>) -> Result<(CandidateMap, CandidateTruthMap), TwinError> {
    let mut candidates = BTreeMap::new();
    let mut truths = BTreeMap::new();
    for seed in seeds {
        let id = seed.candidate.candidate_id.clone();
        if id.trim().is_empty() || seed.candidate.parameter_digest.trim().is_empty() {
            return Err(TwinError::EmptyIdentity("candidate"));
        }
        if seed
            .candidate
            .features
            .iter()
            .any(|value| !value.is_finite())
            || seed
                .truth
                .stage_means
                .iter()
                .any(|value| !value.is_finite())
            || !seed.truth.final_utility.is_finite()
        {
            return Err(TwinError::CandidateOutcomeMismatch(id));
        }
        if candidates.insert(id.clone(), seed.candidate).is_some() {
            return Err(TwinError::DuplicateCandidate(id));
        }
        truths.insert(id, seed.truth);
    }
    Ok((candidates, truths))
}

fn hash_candidate(candidate_id: &str) -> u64 {
    candidate_id
        .as_bytes()
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ *byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

fn keyed_seed(master: u64, candidate_id: &str, stage: StageKind, attempt: u64) -> u64 {
    master
        ^ hash_candidate(candidate_id).rotate_left(13)
        ^ (stage.index() as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ attempt.wrapping_mul(0xbf58_476d_1ce4_e5b9)
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    state.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn sample_job(seed: u64, stage: &StageSpec, duration_scale_permille: u64) -> (u64, bool) {
    let mut state = if seed == 0 {
        0xa076_1d64_78bd_642f
    } else {
        seed
    };
    let jitter = if stage.duration_jitter_seconds == 0 {
        0
    } else if stage.duration_jitter_seconds == u64::MAX {
        next_random(&mut state)
    } else {
        next_random(&mut state) % (stage.duration_jitter_seconds + 1)
    };
    let unscaled = stage.base_duration_seconds.saturating_add(jitter);
    let duration = scaled_duration(unscaled, duration_scale_permille);
    let failure_draw = next_random(&mut state) % PPM_SCALE;
    (duration, failure_draw < stage.failure_probability_ppm)
}

/// Deterministic standard-normal approximation using a centered
/// Irwin-Hall sum of twelve independent uniform draws. Its mean is zero and
/// variance is one; it is not a substitute for a calibrated instrument-noise
/// model, which belongs in the B2 qualification dataset.
fn sample_noise(seed: u64, standard_deviation: f64) -> f64 {
    if standard_deviation == 0.0 {
        return 0.0;
    }
    let mut state = if seed == 0 {
        0xa076_1d64_78bd_642f
    } else {
        seed
    };
    let mut sum = 0.0;
    for _ in 0..12 {
        let uniform = (next_random(&mut state) >> 11) as f64 / ((1_u64 << 53) as f64);
        sum += uniform;
    }
    (sum - 6.0) * standard_deviation
}

fn fouling_added(job: &InFlightJob, stages: &BTreeMap<StageKind, StageSpec>) -> u64 {
    stages
        .get(&job.stage)
        .map(|stage| stage.fouling_increment_ppm)
        .unwrap_or(0)
}

fn ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut output = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = ((start + 1) as f64 + end as f64) / 2.0;
        for index in &order[start..end] {
            output[*index] = rank;
        }
        start = end;
    }
    output
}

fn spearman(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < 3 {
        return None;
    }
    let a = ranks(left);
    let b = ranks(right);
    let mean_a = a.iter().sum::<f64>() / a.len() as f64;
    let mean_b = b.iter().sum::<f64>() / b.len() as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = *x - mean_a;
        let dy = *y - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    let denom = (var_a * var_b).sqrt();
    (denom > f64::EPSILON).then_some(cov / denom)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(rig_seconds: f64, energy: f64) -> ResourceCost {
        ResourceCost {
            rig_seconds,
            source_energy_wh: energy,
            feedstock_g: 1.0,
            assay_cost: 1.0,
            maintenance_minutes: 0.1,
            consumables_g: 0.2,
        }
    }

    fn candidate(id: &str, utility: f64) -> CandidateSeed {
        CandidateSeed {
            candidate: CandidateDesign {
                candidate_id: id.to_string(),
                features: vec![id.len() as f64],
                parameter_digest: format!("sha256:{id}"),
            },
            truth: CandidateTruth {
                stage_means: [utility * 0.2, utility * 0.5, utility * 0.8, utility],
                final_utility: utility,
            },
        }
    }

    fn stage(
        stage: StageKind,
        instruments: &[&str],
        duration: u64,
        jitter: u64,
        noise: f64,
        fouling: u64,
        failure_ppm: u64,
    ) -> StageSpec {
        StageSpec {
            stage,
            eligible_instruments: instruments.iter().map(|s| s.to_string()).collect(),
            base_duration_seconds: duration,
            duration_jitter_seconds: jitter,
            measurement_noise_stddev: noise,
            fouling_increment_ppm: fouling,
            fouling_bias_per_ppm: 0.00001,
            resource_cost: cost(0.0, 2.0),
            failure_probability_ppm: failure_ppm,
        }
    }

    fn instrument(id: &str, valid_jobs: u64, dirty_threshold: u64, scale: u64) -> InstrumentSpec {
        InstrumentSpec {
            instrument_id: id.to_string(),
            calibration_valid_for_jobs: valid_jobs,
            fouling_threshold_ppm: dirty_threshold,
            duration_scale_permille: scale,
            maintenance_duration_seconds: 2,
            recalibration_duration_seconds: 3,
            maintenance_cost: cost(0.0, 1.0),
            recalibration_cost: cost(0.0, 1.0),
        }
    }

    fn world(seed: u64, fail_ppm: u64, valid_jobs: u64, fouling_threshold: u64) -> TwinWorld {
        let instruments = vec![
            instrument("spectrometer-a", valid_jobs, fouling_threshold, 1000),
            instrument("spectrometer-b", valid_jobs, fouling_threshold, 2000),
            instrument("reactor", valid_jobs, fouling_threshold, 1000),
            instrument("capture-train", valid_jobs, fouling_threshold, 1000),
            instrument("assay-lab", valid_jobs, fouling_threshold, 1000),
        ];
        let stages = vec![
            stage(
                StageKind::Characterization,
                &["spectrometer-a", "spectrometer-b"],
                4,
                2,
                0.02,
                2,
                fail_ppm,
            ),
            stage(StageKind::Exposure, &["reactor"], 5, 3, 0.03, 5, fail_ppm),
            stage(
                StageKind::Capture,
                &["capture-train"],
                3,
                1,
                0.01,
                1,
                fail_ppm,
            ),
            stage(StageKind::Assay, &["assay-lab"], 2, 1, 0.01, 0, fail_ppm),
        ];
        TwinWorld::new(
            TwinManifest {
                dataset_id: "isru-b1-dev".to_string(),
                task_version: "0.1.0".to_string(),
                scenario_digest: "sha256:scenario".to_string(),
                seed,
                max_experiments: 3,
                max_wall_seconds: 10_000,
                resource_budget: ResourceCost {
                    rig_seconds: 10_000.0,
                    source_energy_wh: 10_000.0,
                    feedstock_g: 10_000.0,
                    assay_cost: 10_000.0,
                    maintenance_minutes: 10_000.0,
                    consumables_g: 10_000.0,
                },
            },
            stages,
            instruments,
            vec![
                candidate("train-a", 0.3),
                candidate("train-b", 0.8),
                candidate("train-c", 0.6),
            ],
            vec![
                candidate("holdout-a", 0.2),
                candidate("holdout-b", 0.5),
                candidate("holdout-c", 0.9),
            ],
        )
        .unwrap()
    }

    fn complete_all_stages(api: &mut TwinAgentApi<'_>, candidate_id: &str) {
        for (index, stage) in [
            StageKind::Characterization,
            StageKind::Exposure,
            StageKind::Capture,
            StageKind::Assay,
        ]
        .iter()
        .enumerate()
        {
            let instrument_id = match stage {
                StageKind::Characterization => "spectrometer-a",
                StageKind::Exposure => "reactor",
                StageKind::Capture => "capture-train",
                StageKind::Assay => "assay-lab",
            };
            api.launch_stage(
                &format!("{candidate_id}-job-{index}"),
                candidate_id,
                *stage,
                instrument_id,
            )
            .unwrap();
            match api.advance_next_event().unwrap().unwrap() {
                TwinTransition::Completed(observation) => {
                    assert!(observation.simulation_only);
                    assert!(observation.completed_at_seconds > 0);
                }
                TwinTransition::Failed { .. } => panic!("zero-failure fixture unexpectedly failed"),
            }
        }
    }

    #[test]
    fn scaled_duration_saturates_without_integer_overflow() {
        assert_eq!(scaled_duration(u64::MAX, u64::MAX), u64::MAX);
        assert_eq!(scaled_duration(0, 1000), 1);
        assert_eq!(scaled_duration(100, 1500), 150);
    }

    #[test]
    fn two_instruments_allow_parallel_independent_jobs_and_finish_out_of_order() {
        let mut world = world(17, 0, 10, 1_000);
        let mut api = world.agent_api();
        let slow = api
            .launch_stage(
                "slow-job",
                "train-a",
                StageKind::Characterization,
                "spectrometer-b",
            )
            .unwrap();
        let fast = api
            .launch_stage(
                "fast-job",
                "train-b",
                StageKind::Characterization,
                "spectrometer-a",
            )
            .unwrap();
        assert!(slow > fast);
        let first = api.advance_next_event().unwrap().unwrap();
        assert!(matches!(first, TwinTransition::Completed(ref obs) if obs.job_id == "fast-job"));
        let second = api.advance_next_event().unwrap().unwrap();
        assert!(matches!(second, TwinTransition::Completed(ref obs) if obs.job_id == "slow-job"));
    }

    #[test]
    fn one_instrument_cannot_be_reserved_by_two_inflight_jobs() {
        let mut world = world(19, 0, 10, 1_000);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-a",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        assert_eq!(
            api.launch_stage(
                "job-b",
                "train-b",
                StageKind::Characterization,
                "spectrometer-a"
            ),
            Err(TwinError::InstrumentBusy("spectrometer-a".to_string()))
        );
    }

    #[test]
    fn stage_prerequisites_are_enforced() {
        let mut world = world(23, 0, 10, 1_000);
        let mut api = world.agent_api();
        assert_eq!(
            api.launch_stage(
                "exposure-too-soon",
                "train-a",
                StageKind::Exposure,
                "reactor"
            ),
            Err(TwinError::InvalidStageOrder {
                candidate_id: "train-a".to_string(),
                expected: Some(StageKind::Characterization),
                requested: StageKind::Exposure,
            })
        );
    }

    #[test]
    fn holdout_candidates_are_visible_as_metadata_but_not_measurable() {
        let mut world = world(29, 0, 10, 1_000);
        let mut api = world.agent_api();
        assert_eq!(api.holdout_catalog().len(), 3);
        assert_eq!(
            api.launch_stage(
                "holdout-job",
                "holdout-a",
                StageKind::Characterization,
                "spectrometer-a"
            ),
            Err(TwinError::HoldoutCannotBeMeasured("holdout-a".to_string()))
        );
    }

    #[test]
    fn calibration_expiry_blocks_new_work_until_recalibration() {
        let mut world = world(31, 0, 1, 1_000);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-a",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        api.advance_next_event().unwrap();
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::CalibrationDue)
        );
        assert_eq!(
            api.launch_stage(
                "job-b",
                "train-b",
                StageKind::Characterization,
                "spectrometer-a"
            ),
            Err(TwinError::CalibrationDue("spectrometer-a".to_string()))
        );
        api.recalibrate("spectrometer-a").unwrap();
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::Available)
        );
    }

    #[test]
    fn fouling_blocks_instrument_until_maintenance() {
        let mut world = world(37, 0, 10, 2);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-a",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        api.advance_next_event().unwrap();
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::MaintenanceDue)
        );
        assert_eq!(
            api.launch_stage(
                "job-b",
                "train-b",
                StageKind::Characterization,
                "spectrometer-a"
            ),
            Err(TwinError::MaintenanceDue("spectrometer-a".to_string()))
        );
        api.perform_maintenance("spectrometer-a").unwrap();
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::Available)
        );
    }

    #[test]
    fn failed_jobs_consume_budget_release_lock_and_do_not_advance_candidate() {
        let mut world = world(41, PPM_SCALE, 10, 10_000);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-fail",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        assert!(matches!(
            api.advance_next_event().unwrap(),
            Some(TwinTransition::Failed { .. })
        ));
        assert_eq!(
            api.next_stage("train-a"),
            Ok(Some(StageKind::Characterization))
        );
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::Available)
        );
        assert!(api.world.spent.rig_seconds > 0.0);
        assert_eq!(api.world.reserved.rig_seconds, 0.0);
    }

    #[test]
    fn hard_experiment_budget_is_checked_before_launch() {
        let mut world = world(43, 0, 10, 10_000);
        world.manifest.max_experiments = 1;
        let mut api = world.agent_api();
        api.launch_stage(
            "job-a",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        api.advance_next_event().unwrap();
        assert_eq!(
            api.launch_stage(
                "job-b",
                "train-b",
                StageKind::Characterization,
                "spectrometer-b"
            ),
            Err(TwinError::ExperimentBudgetExhausted)
        );
    }

    #[test]
    fn resource_reservations_block_overcommit_before_execution() {
        let mut world = world(47, 0, 10, 10_000);
        world.manifest.resource_budget.source_energy_wh = 1.0;
        let mut api = world.agent_api();
        assert_eq!(
            api.launch_stage(
                "job-a",
                "train-a",
                StageKind::Characterization,
                "spectrometer-a"
            ),
            Err(TwinError::ResourceBudgetExceeded)
        );
        assert_eq!(api.world.active_jobs.len(), 0);
        assert_eq!(api.world.reserved.source_energy_wh, 0.0);
    }

    #[test]
    fn maintenance_and_recalibration_consume_wall_time_and_resources() {
        let mut world = world(53, 0, 1, 2);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-a",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        api.advance_next_event().unwrap();
        let before = api.world.elapsed_seconds();
        api.perform_maintenance("spectrometer-a").unwrap();
        assert!(api.world.elapsed_seconds() >= before + 2);
        api.recalibrate("spectrometer-a").unwrap();
        assert_eq!(
            api.instrument_condition("spectrometer-a"),
            Ok(InstrumentCondition::Available)
        );
    }

    #[test]
    fn simulated_assay_remains_explicitly_simulated() {
        let mut world = world(59, 0, 10, 100_000);
        let mut api = world.agent_api();
        for (i, stage) in [
            StageKind::Characterization,
            StageKind::Exposure,
            StageKind::Capture,
            StageKind::Assay,
        ]
        .iter()
        .enumerate()
        {
            let instrument_id = match stage {
                StageKind::Characterization => "spectrometer-a",
                StageKind::Exposure => "reactor",
                StageKind::Capture => "capture-train",
                StageKind::Assay => "assay-lab",
            };
            api.launch_stage(&format!("job-{i}"), "train-a", *stage, instrument_id)
                .unwrap();
            let transition = api.advance_next_event().unwrap().unwrap();
            if *stage == StageKind::Assay {
                assert!(matches!(
                    transition,
                    TwinTransition::Completed(SimulatedObservation {
                        output_kind: SimulatedOutputKind::SimulatedIndependentAssay,
                        simulation_only: true,
                        ..
                    })
                ));
            }
        }
    }

    #[test]
    fn deterministic_noise_approximation_has_zero_mean_and_unit_variance() {
        let values: Vec<f64> = (1..=10_000).map(|seed| sample_noise(seed, 1.0)).collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        assert!(mean.abs() < 0.04);
        assert!((variance - 1.0).abs() < 0.06);
    }

    #[test]
    fn deterministic_seed_and_action_order_replay_identically() {
        fn replay() -> (Vec<TwinEvent>, Vec<SimulatedObservation>) {
            let mut world = world(61, 50_000, 10, 10_000);
            {
                let mut api = world.agent_api();
                api.launch_stage(
                    "job-a",
                    "train-a",
                    StageKind::Characterization,
                    "spectrometer-a",
                )
                .unwrap();
                api.launch_stage(
                    "job-b",
                    "train-b",
                    StageKind::Characterization,
                    "spectrometer-b",
                )
                .unwrap();
                while api.advance_next_event().unwrap().is_some() {}
                api.close_campaign().unwrap();
            }
            (world.events.clone(), world.completed_observations.clone())
        }
        assert_eq!(replay(), replay());
    }

    #[test]
    fn discovery_and_learning_grading_require_closeout() {
        let mut world = world(67, 0, 10, 100_000);
        assert_eq!(
            world.grade_discovery(0.5),
            Err(TwinError::CampaignNotClosed)
        );
        assert_eq!(
            world.grade_learning(Vec::new()),
            Err(TwinError::CampaignNotClosed)
        );
        {
            let mut api = world.agent_api();
            complete_all_stages(&mut api, "train-a");
            api.close_campaign().unwrap();
        }
        let score = world.grade_discovery(0.1).unwrap();
        assert_eq!(score.candidates_assayed, 1);
        assert!(score.passed);
    }

    #[test]
    fn active_candidate_cannot_start_two_concurrent_stages() {
        let mut world = world(69, 0, 10, 10_000);
        let mut api = world.agent_api();
        api.launch_stage(
            "job-first",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        assert_eq!(
            api.launch_stage(
                "job-second",
                "train-a",
                StageKind::Characterization,
                "spectrometer-b",
            ),
            Err(TwinError::CandidateAlreadyActive("train-a".to_string()))
        );
    }

    #[test]
    fn stage_launch_is_refused_when_worst_case_duration_exceeds_wall_budget() {
        let mut world = world(70, 0, 10, 10_000);
        world.manifest.max_wall_seconds = 5;
        let mut api = world.agent_api();
        assert_eq!(
            api.launch_stage(
                "too-long",
                "train-a",
                StageKind::Characterization,
                "spectrometer-b",
            ),
            Err(TwinError::TimeBudgetExceeded)
        );
    }

    #[test]
    fn retry_after_failed_stage_uses_new_attempt_and_does_not_advance_progress() {
        let mut world = world(72, PPM_SCALE, 10, 100_000);
        let mut api = world.agent_api();
        api.launch_stage(
            "failure-one",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        assert!(matches!(
            api.advance_next_event().unwrap(),
            Some(TwinTransition::Failed { .. })
        ));
        assert_eq!(
            api.next_stage("train-a"),
            Ok(Some(StageKind::Characterization))
        );
        assert_eq!(
            api.launch_stage(
                "failure-one",
                "train-a",
                StageKind::Characterization,
                "spectrometer-a",
            ),
            Err(TwinError::DuplicateJob("failure-one".to_string()))
        );
        api.launch_stage(
            "failure-two",
            "train-a",
            StageKind::Characterization,
            "spectrometer-a",
        )
        .unwrap();
        assert!(matches!(
            api.advance_next_event().unwrap(),
            Some(TwinTransition::Failed { job_id, .. }) if job_id == "failure-two"
        ));
    }

    #[test]
    fn holdout_grader_requires_predictions_for_every_hidden_candidate() {
        let mut world = world(71, 0, 10, 100_000);
        world.agent_api().close_campaign().unwrap();
        assert_eq!(
            world.grade_learning(vec![TwinPrediction {
                candidate_id: "holdout-a".to_string(),
                predicted_mean: 0.2,
                interval_lower: 0.0,
                interval_upper: 0.4,
            }]),
            Err(TwinError::MissingHoldoutPrediction("holdout-b".to_string()))
        );
    }

    #[test]
    fn complete_holdout_predictions_are_scored_without_returning_labels() {
        let mut world = world(73, 0, 10, 100_000);
        world.agent_api().close_campaign().unwrap();
        let score = world
            .grade_learning(vec![
                TwinPrediction {
                    candidate_id: "holdout-a".to_string(),
                    predicted_mean: 0.2,
                    interval_lower: 0.1,
                    interval_upper: 0.3,
                },
                TwinPrediction {
                    candidate_id: "holdout-b".to_string(),
                    predicted_mean: 0.5,
                    interval_lower: 0.4,
                    interval_upper: 0.6,
                },
                TwinPrediction {
                    candidate_id: "holdout-c".to_string(),
                    predicted_mean: 0.9,
                    interval_lower: 0.8,
                    interval_upper: 1.0,
                },
            ])
            .unwrap();
        assert_eq!(score.spearman_rank_correlation, Some(1.0));
        assert_eq!(score.interval_coverage, 1.0);
    }

    #[test]
    fn closeout_refuses_active_jobs_and_blocks_later_mutation() {
        let mut world = world(79, 0, 10, 100_000);
        {
            let mut api = world.agent_api();
            api.launch_stage(
                "job-a",
                "train-a",
                StageKind::Characterization,
                "spectrometer-a",
            )
            .unwrap();
            assert_eq!(api.close_campaign(), Err(TwinError::ActiveJobsRemain));
            api.advance_next_event().unwrap();
            api.close_campaign().unwrap();
            assert_eq!(
                api.launch_stage(
                    "job-b",
                    "train-b",
                    StageKind::Characterization,
                    "spectrometer-b"
                ),
                Err(TwinError::CampaignClosed)
            );
        }
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Standalone stage-aware workflow state and evidence reference.
// Run tests with:
//   rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/stage_workflow.rs -o /tmp/isru-stage-tests
//   /tmp/isru-stage-tests

#![forbid(unsafe_code)]

//! Event-auditable stage workflow state for multi-step experimental campaigns.
//!
//! This reference separates stage execution, pending/qualified measurements,
//! intermediate proxies and independently assayed products. It models causal
//! prerequisites and exclusive instrument locks, and only resumes paused work
//! against the exact declared workflow/sample/batch/protocol/graph lineage.
//!
//! It does not execute instruments, define a physically safe process sequence,
//! prove caller-supplied digests, or authenticate external evidence receipts.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowLineage {
    pub workflow_id: String,
    pub sample_id: String,
    pub batch_id: String,
    pub protocol_digest: String,
    pub graph_digest: String,
}

impl WorkflowLineage {
    fn validate(&self) -> Result<(), WorkflowError> {
        for (name, value) in [
            ("workflow_id", self.workflow_id.as_str()),
            ("sample_id", self.sample_id.as_str()),
            ("batch_id", self.batch_id.as_str()),
            ("protocol_digest", self.protocol_digest.as_str()),
            ("graph_digest", self.graph_digest.as_str()),
        ] {
            if !nonempty(value) {
                return Err(WorkflowError::EmptyIdentity(name));
            }
        }
        Ok(())
    }
}

/// The output type is part of the stage contract. Proxy observations cannot
/// be cast into product observations by changing a status or a label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputKind {
    IntermediateProxy,
    CapturedProduct,
    PurifiedProduct,
    IndependentProductAssay,
    RecoveryEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageSpec {
    pub id: String,
    pub prerequisites: Vec<String>,
    pub output_kind: OutputKind,
    pub requires_qualified_observation: bool,
    pub required_for_product_acceptance: bool,
    pub required_for_campaign_close: bool,
    /// An exclusive instrument / apparatus lock; absent means no exclusive lock.
    pub exclusive_resource: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageGraph {
    pub stages: Vec<StageSpec>,
}

impl StageGraph {
    pub fn validate(&self) -> Result<(), WorkflowError> {
        if self.stages.is_empty() {
            return Err(WorkflowError::EmptyStageGraph);
        }

        let mut by_id = BTreeMap::new();
        for stage in &self.stages {
            if !nonempty(&stage.id) {
                return Err(WorkflowError::EmptyIdentity("stage_id"));
            }
            if let Some(resource) = &stage.exclusive_resource {
                if !nonempty(resource) {
                    return Err(WorkflowError::EmptyIdentity("exclusive_resource"));
                }
            }
            if by_id.insert(stage.id.as_str(), stage).is_some() {
                return Err(WorkflowError::DuplicateStage(stage.id.clone()));
            }
            let mut predecessors = BTreeSet::new();
            for prerequisite in &stage.prerequisites {
                if !nonempty(prerequisite) {
                    return Err(WorkflowError::EmptyIdentity("prerequisite_id"));
                }
                if prerequisite == &stage.id {
                    return Err(WorkflowError::SelfDependency(stage.id.clone()));
                }
                if !predecessors.insert(prerequisite.as_str()) {
                    return Err(WorkflowError::DuplicatePrerequisite {
                        stage_id: stage.id.clone(),
                        prerequisite_id: prerequisite.clone(),
                    });
                }
            }
        }

        for stage in &self.stages {
            for prerequisite in &stage.prerequisites {
                if !by_id.contains_key(prerequisite.as_str()) {
                    return Err(WorkflowError::MissingPrerequisite {
                        stage_id: stage.id.clone(),
                        prerequisite_id: prerequisite.clone(),
                    });
                }
            }
        }

        // Kahn's algorithm: a graph is valid only if all nodes are removed.
        let mut indegree: BTreeMap<&str, usize> = self
            .stages
            .iter()
            .map(|stage| (stage.id.as_str(), stage.prerequisites.len()))
            .collect();
        let mut successors: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for stage in &self.stages {
            for prerequisite in &stage.prerequisites {
                successors
                    .entry(prerequisite.as_str())
                    .or_default()
                    .push(stage.id.as_str());
            }
        }

        let mut ready: VecDeque<&str> = indegree
            .iter()
            .filter_map(|(id, degree)| (*degree == 0).then_some(*id))
            .collect();
        let mut visited = 0;

        while let Some(id) = ready.pop_front() {
            visited += 1;
            if let Some(children) = successors.get(id) {
                for child in children {
                    let degree = indegree.get_mut(child).expect("validated stage");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.push_back(child);
                    }
                }
            }
        }

        if visited != self.stages.len() {
            return Err(WorkflowError::DependencyCycle);
        }
        Ok(())
    }

    fn stage(&self, stage_id: &str) -> Result<&StageSpec, WorkflowError> {
        self.stages
            .iter()
            .find(|stage| stage.id == stage_id)
            .ok_or_else(|| WorkflowError::UnknownStage(stage_id.to_string()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageStatus {
    NotStarted,
    Running,
    PartiallyObserved,
    Paused,
    Resumable,
    Completed,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObservationStatus {
    Pending,
    Qualified,
    Rejected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationRecord {
    pub id: String,
    pub stage_id: String,
    pub kind: OutputKind,
    pub status: ObservationStatus,
    pub observed_at_utc_s: i64,
    pub recorded_at_utc_s: i64,
    pub calibration_receipt_id: Option<String>,
    pub qualification_receipt_id: Option<String>,
    pub rejection_receipt_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageRuntime {
    pub status: StageStatus,
    pub checkpoint_receipt_id: Option<String>,
    pub failure_receipt_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowEventKind {
    StageStarted { stage_id: String },
    ObservationRegistered { observation_id: String, stage_id: String },
    ObservationQualified { observation_id: String },
    ObservationRejected { observation_id: String, rejection_receipt_id: String },
    StagePaused { stage_id: String, checkpoint_receipt_id: String },
    StageMarkedResumable { stage_id: String, checkpoint_receipt_id: String },
    StageResumed { stage_id: String, checkpoint_receipt_id: String },
    StageCompleted { stage_id: String },
    StageFailed { stage_id: String, failure_receipt_id: String },
    ProductAccepted { assay_observation_id: String, independent_assay_receipt_id: String },
    CampaignClosed { closeout_receipt_id: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkflowEvent {
    pub sequence: u64,
    pub recorded_at_utc_s: i64,
    pub kind: WorkflowEventKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkflowError {
    EmptyIdentity(&'static str),
    EmptyStageGraph,
    DuplicateStage(String),
    SelfDependency(String),
    DuplicatePrerequisite { stage_id: String, prerequisite_id: String },
    MissingPrerequisite { stage_id: String, prerequisite_id: String },
    DependencyCycle,
    UnknownStage(String),
    UnknownObservation(String),
    DuplicateObservation(String),
    StageNotReady(String),
    InvalidStageTransition { stage_id: String, status: StageStatus },
    ExclusiveResourceBusy { resource: String, stage_id: String },
    InvalidObservationKind { stage_id: String, expected: OutputKind, actual: OutputKind },
    ObservationNotPending(String),
    ObservationNotQualified(String),
    MissingRequiredObservation(String),
    PendingObservationsRemain(String),
    MissingCheckpoint(String),
    CheckpointMismatch(String),
    LineageMismatch,
    MissingRequiredAcceptanceStage(String),
    IndependentAssayRequired,
    ProductAlreadyAccepted,
    ProductNotAccepted,
    CampaignClosed,
    RequiredStageIncomplete(String),
    ActiveStagesRemain,
    ObservationTimeAfterReceipt(String),
    EventTimeWentBackwards,
    MissingReceipt(&'static str),
}

impl fmt::Display for WorkflowError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl Error for WorkflowError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CampaignWorkflow {
    pub lineage: WorkflowLineage,
    pub graph: StageGraph,
    pub stage_runtime: BTreeMap<String, StageRuntime>,
    pub observations: BTreeMap<String, ObservationRecord>,
    pub accepted_product: bool,
    pub closed: bool,
    pub events: Vec<WorkflowEvent>,
}

impl CampaignWorkflow {
    pub fn new(lineage: WorkflowLineage, graph: StageGraph) -> Result<Self, WorkflowError> {
        lineage.validate()?;
        graph.validate()?;

        if !graph
            .stages
            .iter()
            .any(|stage| stage.required_for_product_acceptance
                && stage.output_kind == OutputKind::IndependentProductAssay)
        {
            return Err(WorkflowError::IndependentAssayRequired);
        }

        let stage_runtime = graph
            .stages
            .iter()
            .map(|stage| {
                (
                    stage.id.clone(),
                    StageRuntime {
                        status: StageStatus::NotStarted,
                        checkpoint_receipt_id: None,
                        failure_receipt_id: None,
                    },
                )
            })
            .collect();

        Ok(Self {
            lineage,
            graph,
            stage_runtime,
            observations: BTreeMap::new(),
            accepted_product: false,
            closed: false,
            events: Vec::new(),
        })
    }

    fn ensure_open(&self) -> Result<(), WorkflowError> {
        if self.closed {
            return Err(WorkflowError::CampaignClosed);
        }
        Ok(())
    }

    fn ensure_event_time(&self, recorded_at_utc_s: i64) -> Result<(), WorkflowError> {
        if self
            .events
            .last()
            .is_some_and(|previous| recorded_at_utc_s < previous.recorded_at_utc_s)
        {
            return Err(WorkflowError::EventTimeWentBackwards);
        }
        Ok(())
    }

    fn append_event(
        &mut self,
        recorded_at_utc_s: i64,
        kind: WorkflowEventKind,
    ) -> Result<(), WorkflowError> {
        self.ensure_event_time(recorded_at_utc_s)?;
        self.events.push(WorkflowEvent {
            sequence: self.events.len() as u64,
            recorded_at_utc_s,
            kind,
        });
        Ok(())
    }

    fn runtime(&self, stage_id: &str) -> Result<&StageRuntime, WorkflowError> {
        self.stage_runtime
            .get(stage_id)
            .ok_or_else(|| WorkflowError::UnknownStage(stage_id.to_string()))
    }

    fn runtime_mut(&mut self, stage_id: &str) -> Result<&mut StageRuntime, WorkflowError> {
        self.stage_runtime
            .get_mut(stage_id)
            .ok_or_else(|| WorkflowError::UnknownStage(stage_id.to_string()))
    }

    fn prerequisites_completed(&self, stage: &StageSpec) -> Result<bool, WorkflowError> {
        for prerequisite in &stage.prerequisites {
            if self.runtime(prerequisite)?.status != StageStatus::Completed {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn resource_busy(&self, resource: &str, except_stage: &str) -> bool {
        self.graph.stages.iter().any(|stage| {
            stage.id != except_stage
                && stage.exclusive_resource.as_deref() == Some(resource)
                && self.stage_runtime.get(&stage.id).is_some_and(|runtime| {
                    runtime.status == StageStatus::Running
                        || runtime.status == StageStatus::PartiallyObserved
                })
        })
    }

    pub fn status(&self, stage_id: &str) -> Result<StageStatus, WorkflowError> {
        Ok(self.runtime(stage_id)?.status)
    }

    /// Start an eligible stage. Multiple independent stages may run concurrently,
    /// but only one may hold a given exclusive resource lock.
    pub fn start_stage(
        &mut self,
        stage_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        let spec = self.graph.stage(stage_id)?.clone();
        if self.runtime(stage_id)?.status != StageStatus::NotStarted {
            return Err(WorkflowError::InvalidStageTransition {
                stage_id: stage_id.to_string(),
                status: self.runtime(stage_id)?.status,
            });
        }
        if !self.prerequisites_completed(&spec)? {
            return Err(WorkflowError::StageNotReady(stage_id.to_string()));
        }
        if let Some(resource) = &spec.exclusive_resource {
            if self.resource_busy(resource, stage_id) {
                return Err(WorkflowError::ExclusiveResourceBusy {
                    resource: resource.clone(),
                    stage_id: stage_id.to_string(),
                });
            }
        }

        self.runtime_mut(stage_id)?.status = StageStatus::Running;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StageStarted {
                stage_id: stage_id.to_string(),
            },
        )
    }

    /// Register a measurement request. Pending is not an observation and does
    /// not count as qualified stage output.
    pub fn register_pending_observation(
        &mut self,
        stage_id: &str,
        observation_id: &str,
        kind: OutputKind,
        observed_at_utc_s: i64,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(observation_id) {
            return Err(WorkflowError::EmptyIdentity("observation_id"));
        }
        if self.observations.contains_key(observation_id) {
            return Err(WorkflowError::DuplicateObservation(observation_id.to_string()));
        }
        if observed_at_utc_s > recorded_at_utc_s {
            return Err(WorkflowError::ObservationTimeAfterReceipt(
                observation_id.to_string(),
            ));
        }
        let spec = self.graph.stage(stage_id)?;
        if spec.output_kind != kind {
            return Err(WorkflowError::InvalidObservationKind {
                stage_id: stage_id.to_string(),
                expected: spec.output_kind,
                actual: kind,
            });
        }

        match self.runtime(stage_id)?.status {
            StageStatus::Running | StageStatus::PartiallyObserved | StageStatus::Paused
            | StageStatus::Resumable => {}
            status => {
                return Err(WorkflowError::InvalidStageTransition {
                    stage_id: stage_id.to_string(),
                    status,
                });
            }
        }

        self.observations.insert(
            observation_id.to_string(),
            ObservationRecord {
                id: observation_id.to_string(),
                stage_id: stage_id.to_string(),
                kind,
                status: ObservationStatus::Pending,
                observed_at_utc_s,
                recorded_at_utc_s,
                calibration_receipt_id: None,
                qualification_receipt_id: None,
                rejection_receipt_id: None,
            },
        );

        if self.runtime(stage_id)?.status == StageStatus::Running {
            self.runtime_mut(stage_id)?.status = StageStatus::PartiallyObserved;
        }
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::ObservationRegistered {
                observation_id: observation_id.to_string(),
                stage_id: stage_id.to_string(),
            },
        )
    }

    /// Attach qualified metrology/authority receipts to an existing pending
    /// observation. Receipt identity is checked for presence, not authenticity.
    pub fn qualify_observation(
        &mut self,
        observation_id: &str,
        calibration_receipt_id: &str,
        qualification_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(calibration_receipt_id) {
            return Err(WorkflowError::MissingReceipt("calibration_receipt_id"));
        }
        if !nonempty(qualification_receipt_id) {
            return Err(WorkflowError::MissingReceipt("qualification_receipt_id"));
        }

        let observation = self
            .observations
            .get_mut(observation_id)
            .ok_or_else(|| WorkflowError::UnknownObservation(observation_id.to_string()))?;
        if observation.status != ObservationStatus::Pending {
            return Err(WorkflowError::ObservationNotPending(
                observation_id.to_string(),
            ));
        }
        if recorded_at_utc_s < observation.recorded_at_utc_s {
            return Err(WorkflowError::EventTimeWentBackwards);
        }
        observation.status = ObservationStatus::Qualified;
        observation.calibration_receipt_id = Some(calibration_receipt_id.to_string());
        observation.qualification_receipt_id = Some(qualification_receipt_id.to_string());

        let stage_id = observation.stage_id.clone();
        let status = self.runtime(&stage_id)?.status;
        if status == StageStatus::Running {
            self.runtime_mut(&stage_id)?.status = StageStatus::PartiallyObserved;
        }
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::ObservationQualified {
                observation_id: observation_id.to_string(),
            },
        )
    }

    pub fn reject_observation(
        &mut self,
        observation_id: &str,
        rejection_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(rejection_receipt_id) {
            return Err(WorkflowError::MissingReceipt("rejection_receipt_id"));
        }
        let observation = self
            .observations
            .get_mut(observation_id)
            .ok_or_else(|| WorkflowError::UnknownObservation(observation_id.to_string()))?;
        if observation.status != ObservationStatus::Pending {
            return Err(WorkflowError::ObservationNotPending(
                observation_id.to_string(),
            ));
        }
        if recorded_at_utc_s < observation.recorded_at_utc_s {
            return Err(WorkflowError::EventTimeWentBackwards);
        }
        observation.status = ObservationStatus::Rejected;
        observation.rejection_receipt_id = Some(rejection_receipt_id.to_string());
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::ObservationRejected {
                observation_id: observation_id.to_string(),
                rejection_receipt_id: rejection_receipt_id.to_string(),
            },
        )
    }

    pub fn pause_stage(
        &mut self,
        stage_id: &str,
        checkpoint_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(checkpoint_receipt_id) {
            return Err(WorkflowError::MissingCheckpoint(stage_id.to_string()));
        }
        match self.runtime(stage_id)?.status {
            StageStatus::Running | StageStatus::PartiallyObserved => {}
            status => {
                return Err(WorkflowError::InvalidStageTransition {
                    stage_id: stage_id.to_string(),
                    status,
                });
            }
        }
        let runtime = self.runtime_mut(stage_id)?;
        runtime.status = StageStatus::Paused;
        runtime.checkpoint_receipt_id = Some(checkpoint_receipt_id.to_string());
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StagePaused {
                stage_id: stage_id.to_string(),
                checkpoint_receipt_id: checkpoint_receipt_id.to_string(),
            },
        )
    }

    pub fn confirm_resumable(
        &mut self,
        stage_id: &str,
        checkpoint_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        let runtime = self.runtime(stage_id)?;
        if runtime.status != StageStatus::Paused {
            return Err(WorkflowError::InvalidStageTransition {
                stage_id: stage_id.to_string(),
                status: runtime.status,
            });
        }
        if runtime.checkpoint_receipt_id.as_deref() != Some(checkpoint_receipt_id) {
            return Err(WorkflowError::CheckpointMismatch(stage_id.to_string()));
        }
        self.runtime_mut(stage_id)?.status = StageStatus::Resumable;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StageMarkedResumable {
                stage_id: stage_id.to_string(),
                checkpoint_receipt_id: checkpoint_receipt_id.to_string(),
            },
        )
    }

    pub fn resume_stage(
        &mut self,
        stage_id: &str,
        lineage: &WorkflowLineage,
        checkpoint_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        lineage.validate()?;
        if lineage != &self.lineage {
            return Err(WorkflowError::LineageMismatch);
        }
        let runtime = self.runtime(stage_id)?;
        if runtime.status != StageStatus::Resumable {
            return Err(WorkflowError::InvalidStageTransition {
                stage_id: stage_id.to_string(),
                status: runtime.status,
            });
        }
        if runtime.checkpoint_receipt_id.as_deref() != Some(checkpoint_receipt_id) {
            return Err(WorkflowError::CheckpointMismatch(stage_id.to_string()));
        }
        let spec = self.graph.stage(stage_id)?;
        if let Some(resource) = &spec.exclusive_resource {
            if self.resource_busy(resource, stage_id) {
                return Err(WorkflowError::ExclusiveResourceBusy {
                    resource: resource.clone(),
                    stage_id: stage_id.to_string(),
                });
            }
        }
        self.runtime_mut(stage_id)?.status = StageStatus::Running;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StageResumed {
                stage_id: stage_id.to_string(),
                checkpoint_receipt_id: checkpoint_receipt_id.to_string(),
            },
        )
    }

    pub fn complete_stage(
        &mut self,
        stage_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        let spec = self.graph.stage(stage_id)?.clone();
        match self.runtime(stage_id)?.status {
            StageStatus::Running | StageStatus::PartiallyObserved => {}
            status => {
                return Err(WorkflowError::InvalidStageTransition {
                    stage_id: stage_id.to_string(),
                    status,
                });
            }
        }

        let observations: Vec<&ObservationRecord> = self
            .observations
            .values()
            .filter(|observation| observation.stage_id == stage_id)
            .collect();
        if observations
            .iter()
            .any(|observation| observation.status == ObservationStatus::Pending)
        {
            return Err(WorkflowError::PendingObservationsRemain(
                stage_id.to_string(),
            ));
        }
        if spec.requires_qualified_observation
            && !observations
                .iter()
                .any(|observation| observation.status == ObservationStatus::Qualified)
        {
            return Err(WorkflowError::MissingRequiredObservation(
                stage_id.to_string(),
            ));
        }
        self.runtime_mut(stage_id)?.status = StageStatus::Completed;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StageCompleted {
                stage_id: stage_id.to_string(),
            },
        )
    }

    pub fn fail_stage(
        &mut self,
        stage_id: &str,
        failure_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(failure_receipt_id) {
            return Err(WorkflowError::MissingReceipt("failure_receipt_id"));
        }
        match self.runtime(stage_id)?.status {
            StageStatus::Running | StageStatus::PartiallyObserved | StageStatus::Paused
            | StageStatus::Resumable => {}
            status => {
                return Err(WorkflowError::InvalidStageTransition {
                    stage_id: stage_id.to_string(),
                    status,
                });
            }
        }
        let runtime = self.runtime_mut(stage_id)?;
        runtime.status = StageStatus::Failed;
        runtime.failure_receipt_id = Some(failure_receipt_id.to_string());
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::StageFailed {
                stage_id: stage_id.to_string(),
                failure_receipt_id: failure_receipt_id.to_string(),
            },
        )
    }

    /// Only a completed IndependentProductAssay observation plus a separate
    /// independent-assay receipt can establish accepted product in this model.
    pub fn accept_product(
        &mut self,
        assay_observation_id: &str,
        independent_assay_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if self.accepted_product {
            return Err(WorkflowError::ProductAlreadyAccepted);
        }
        if !nonempty(independent_assay_receipt_id) {
            return Err(WorkflowError::MissingReceipt(
                "independent_assay_receipt_id",
            ));
        }

        let assay = self
            .observations
            .get(assay_observation_id)
            .ok_or_else(|| WorkflowError::UnknownObservation(assay_observation_id.to_string()))?;
        if assay.kind != OutputKind::IndependentProductAssay
            || assay.status != ObservationStatus::Qualified
            || !self.graph.stage(&assay.stage_id)?.required_for_product_acceptance
        {
            return Err(WorkflowError::IndependentAssayRequired);
        }
        if self.runtime(&assay.stage_id)?.status != StageStatus::Completed {
            return Err(WorkflowError::RequiredStageIncomplete(
                assay.stage_id.clone(),
            ));
        }

        for stage in self.graph.stages.iter().filter(|stage| stage.required_for_product_acceptance) {
            if self.runtime(&stage.id)?.status != StageStatus::Completed {
                return Err(WorkflowError::MissingRequiredAcceptanceStage(stage.id.clone()));
            }
        }

        self.accepted_product = true;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::ProductAccepted {
                assay_observation_id: assay_observation_id.to_string(),
                independent_assay_receipt_id: independent_assay_receipt_id.to_string(),
            },
        )
    }

    pub fn close_campaign(
        &mut self,
        closeout_receipt_id: &str,
        recorded_at_utc_s: i64,
    ) -> Result<(), WorkflowError> {
        self.ensure_open()?;
        self.ensure_event_time(recorded_at_utc_s)?;
        if !nonempty(closeout_receipt_id) {
            return Err(WorkflowError::MissingReceipt("closeout_receipt_id"));
        }
        for stage in self.graph.stages.iter().filter(|stage| stage.required_for_campaign_close) {
            if self.runtime(&stage.id)?.status != StageStatus::Completed {
                return Err(WorkflowError::RequiredStageIncomplete(stage.id.clone()));
            }
        }
        if self.stage_runtime.values().any(|runtime| {
            runtime.status == StageStatus::Running
                || runtime.status == StageStatus::PartiallyObserved
                || runtime.status == StageStatus::Paused
                || runtime.status == StageStatus::Resumable
        }) {
            return Err(WorkflowError::ActiveStagesRemain);
        }
        if self.observations.values().any(|observation| observation.status == ObservationStatus::Pending) {
            return Err(WorkflowError::PendingObservationsRemain(
                "campaign".to_string(),
            ));
        }
        self.closed = true;
        self.append_event(
            recorded_at_utc_s,
            WorkflowEventKind::CampaignClosed {
                closeout_receipt_id: closeout_receipt_id.to_string(),
            },
        )
    }

    pub fn validate_event_log(&self) -> Result<(), WorkflowError> {
        for (index, event) in self.events.iter().enumerate() {
            if event.sequence != index as u64 {
                return Err(WorkflowError::EventTimeWentBackwards);
            }
            if index > 0
                && event.recorded_at_utc_s < self.events[index - 1].recorded_at_utc_s
            {
                return Err(WorkflowError::EventTimeWentBackwards);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lineage() -> WorkflowLineage {
        WorkflowLineage {
            workflow_id: "workflow-001".to_string(),
            sample_id: "sample-001".to_string(),
            batch_id: "batch-001".to_string(),
            protocol_digest: "sha256:protocol".to_string(),
            graph_digest: "sha256:graph".to_string(),
        }
    }

    fn graph() -> StageGraph {
        StageGraph {
            stages: vec![
                StageSpec {
                    id: "baseline".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::IntermediateProxy,
                    requires_qualified_observation: true,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: true,
                    exclusive_resource: Some("spectrometer".to_string()),
                },
                StageSpec {
                    id: "exposure".to_string(),
                    prerequisites: vec!["baseline".to_string()],
                    output_kind: OutputKind::IntermediateProxy,
                    requires_qualified_observation: true,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: true,
                    exclusive_resource: Some("reactor".to_string()),
                },
                StageSpec {
                    id: "capture".to_string(),
                    prerequisites: vec!["exposure".to_string()],
                    output_kind: OutputKind::CapturedProduct,
                    requires_qualified_observation: true,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: true,
                    exclusive_resource: Some("capture-train".to_string()),
                },
                StageSpec {
                    id: "assay".to_string(),
                    prerequisites: vec!["capture".to_string()],
                    output_kind: OutputKind::IndependentProductAssay,
                    requires_qualified_observation: true,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: true,
                    exclusive_resource: Some("assay-lab".to_string()),
                },
            ],
        }
    }

    fn workflow() -> CampaignWorkflow {
        CampaignWorkflow::new(lineage(), graph()).unwrap()
    }

    fn register_and_qualify(
        workflow: &mut CampaignWorkflow,
        stage_id: &str,
        observation_id: &str,
        kind: OutputKind,
        time: i64,
    ) {
        workflow
            .register_pending_observation(stage_id, observation_id, kind, time, time)
            .unwrap();
        workflow
            .qualify_observation(
                observation_id,
                "calibration-receipt",
                "qualification-receipt",
                time + 1,
            )
            .unwrap();
    }

    fn complete_with_observation(
        workflow: &mut CampaignWorkflow,
        stage_id: &str,
        observation_id: &str,
        kind: OutputKind,
        start_time: i64,
    ) {
        workflow.start_stage(stage_id, start_time).unwrap();
        register_and_qualify(workflow, stage_id, observation_id, kind, start_time + 1);
        workflow.complete_stage(stage_id, start_time + 2).unwrap();
    }

    #[test]
    fn valid_causal_graph_is_accepted() {
        assert!(graph().validate().is_ok());
    }

    #[test]
    fn missing_prerequisite_and_cycle_are_rejected() {
        let missing = StageGraph {
            stages: vec![StageSpec {
                id: "exposure".to_string(),
                prerequisites: vec!["absent".to_string()],
                output_kind: OutputKind::IntermediateProxy,
                requires_qualified_observation: false,
                required_for_product_acceptance: false,
                required_for_campaign_close: false,
                exclusive_resource: None,
            }],
        };
        assert!(matches!(
            missing.validate(),
            Err(WorkflowError::MissingPrerequisite { .. })
        ));

        let mut cyclic = graph();
        cyclic.stages[0].prerequisites = vec!["assay".to_string()];
        assert_eq!(cyclic.validate(), Err(WorkflowError::DependencyCycle));
    }

    #[test]
    fn duplicate_stage_and_self_dependency_are_rejected() {
        let mut duplicate = graph();
        duplicate.stages.push(duplicate.stages[0].clone());
        assert!(matches!(
            duplicate.validate(),
            Err(WorkflowError::DuplicateStage(_))
        ));

        let self_link = StageGraph {
            stages: vec![StageSpec {
                id: "solo".to_string(),
                prerequisites: vec!["solo".to_string()],
                output_kind: OutputKind::IntermediateProxy,
                requires_qualified_observation: false,
                required_for_product_acceptance: false,
                required_for_campaign_close: false,
                exclusive_resource: None,
            }],
        };
        assert_eq!(
            self_link.validate(),
            Err(WorkflowError::SelfDependency("solo".to_string()))
        );
    }

    #[test]
    fn downstream_stage_cannot_start_before_prerequisites_complete() {
        let mut value = workflow();
        assert_eq!(
            value.start_stage("exposure", 10),
            Err(WorkflowError::StageNotReady("exposure".to_string()))
        );
    }

    #[test]
    fn independent_stages_can_run_concurrently_but_exclusive_lock_is_unique() {
        let independent = StageGraph {
            stages: vec![
                StageSpec {
                    id: "sensor_a".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::IntermediateProxy,
                    requires_qualified_observation: false,
                    required_for_product_acceptance: false,
                    required_for_campaign_close: false,
                    exclusive_resource: Some("spectrometer".to_string()),
                },
                StageSpec {
                    id: "sensor_b".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::IntermediateProxy,
                    requires_qualified_observation: false,
                    required_for_product_acceptance: false,
                    required_for_campaign_close: false,
                    exclusive_resource: Some("spectrometer".to_string()),
                },
                StageSpec {
                    id: "independent_log".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::RecoveryEvidence,
                    requires_qualified_observation: false,
                    required_for_product_acceptance: false,
                    required_for_campaign_close: false,
                    exclusive_resource: Some("logger".to_string()),
                },
                StageSpec {
                    id: "assay".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::IndependentProductAssay,
                    requires_qualified_observation: false,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: false,
                    exclusive_resource: None,
                },
            ],
        };
        let mut value = CampaignWorkflow::new(lineage(), independent).unwrap();
        value.start_stage("sensor_a", 1).unwrap();
        value.start_stage("independent_log", 2).unwrap();
        assert_eq!(
            value.start_stage("sensor_b", 3),
            Err(WorkflowError::ExclusiveResourceBusy {
                resource: "spectrometer".to_string(),
                stage_id: "sensor_b".to_string(),
            })
        );
    }

    #[test]
    fn pending_observation_blocks_stage_completion_until_qualified_or_rejected() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        value
            .register_pending_observation(
                "baseline",
                "baseline-reading",
                OutputKind::IntermediateProxy,
                11,
                11,
            )
            .unwrap();
        assert_eq!(
            value.complete_stage("baseline", 12),
            Err(WorkflowError::PendingObservationsRemain(
                "baseline".to_string()
            ))
        );
        value
            .qualify_observation(
                "baseline-reading",
                "calibration-receipt",
                "qualification-receipt",
                13,
            )
            .unwrap();
        value.complete_stage("baseline", 14).unwrap();
    }

    #[test]
    fn observation_kind_cannot_be_substituted_for_stage_output_kind() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        assert_eq!(
            value.register_pending_observation(
                "baseline",
                "false-product",
                OutputKind::IndependentProductAssay,
                11,
                11
            ),
            Err(WorkflowError::InvalidObservationKind {
                stage_id: "baseline".to_string(),
                expected: OutputKind::IntermediateProxy,
                actual: OutputKind::IndependentProductAssay
            })
        );
    }

    #[test]
    fn proxy_signal_cannot_be_promoted_to_accepted_product() {
        let mut value = workflow();
        complete_with_observation(
            &mut value,
            "baseline",
            "baseline-proxy",
            OutputKind::IntermediateProxy,
            10,
        );
        assert_eq!(
            value.accept_product("baseline-proxy", "independent-assay-receipt", 20),
            Err(WorkflowError::IndependentAssayRequired)
        );
    }

    #[test]
    fn paused_work_requires_checkpoint_and_exact_lineage_to_resume() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        value
            .register_pending_observation(
                "baseline",
                "pending-baseline",
                OutputKind::IntermediateProxy,
                11,
                11,
            )
            .unwrap();
        value
            .pause_stage("baseline", "checkpoint-1", 12)
            .unwrap();
        assert_eq!(value.status("baseline"), Ok(StageStatus::Paused));
        assert_eq!(
            value.confirm_resumable("baseline", "wrong-checkpoint", 13),
            Err(WorkflowError::CheckpointMismatch("baseline".to_string()))
        );
        value
            .confirm_resumable("baseline", "checkpoint-1", 13)
            .unwrap();

        let mut changed_sample = lineage();
        changed_sample.sample_id = "different-sample".to_string();
        assert_eq!(
            value.resume_stage("baseline", &changed_sample, "checkpoint-1", 14),
            Err(WorkflowError::LineageMismatch)
        );
        value
            .resume_stage("baseline", &lineage(), "checkpoint-1", 15)
            .unwrap();
        assert_eq!(value.status("baseline"), Ok(StageStatus::Running));
    }

    #[test]
    fn asynchronous_result_can_arrive_while_stage_is_paused() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        value
            .register_pending_observation(
                "baseline",
                "late-reading",
                OutputKind::IntermediateProxy,
                11,
                11,
            )
            .unwrap();
        value.pause_stage("baseline", "checkpoint", 12).unwrap();
        value
            .qualify_observation(
                "late-reading",
                "calibration",
                "qualification",
                20,
            )
            .unwrap();
        assert_eq!(value.status("baseline"), Ok(StageStatus::Paused));
        value.confirm_resumable("baseline", "checkpoint", 21).unwrap();
        value
            .resume_stage("baseline", &lineage(), "checkpoint", 22)
            .unwrap();
        value.complete_stage("baseline", 23).unwrap();
    }

    #[test]
    fn rejected_measurement_does_not_satisfy_required_qualified_output() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        value
            .register_pending_observation(
                "baseline",
                "bad-reading",
                OutputKind::IntermediateProxy,
                11,
                11,
            )
            .unwrap();
        value
            .reject_observation("bad-reading", "rejection-receipt", 12)
            .unwrap();
        assert_eq!(
            value.complete_stage("baseline", 13),
            Err(WorkflowError::MissingRequiredObservation(
                "baseline".to_string()
            ))
        );
    }

    #[test]
    fn only_completed_independent_assay_and_required_chain_can_accept_product() {
        let mut value = workflow();
        complete_with_observation(
            &mut value,
            "baseline",
            "baseline-proxy",
            OutputKind::IntermediateProxy,
            10,
        );
        complete_with_observation(
            &mut value,
            "exposure",
            "exposure-proxy",
            OutputKind::IntermediateProxy,
            20,
        );
        complete_with_observation(
            &mut value,
            "capture",
            "captured-product",
            OutputKind::CapturedProduct,
            30,
        );
        value.start_stage("assay", 40).unwrap();
        register_and_qualify(
            &mut value,
            "assay",
            "assay-result",
            OutputKind::IndependentProductAssay,
            41,
        );
        assert_eq!(
            value.accept_product("assay-result", "independent-assay-receipt", 42),
            Err(WorkflowError::RequiredStageIncomplete("assay".to_string()))
        );
        value.complete_stage("assay", 43).unwrap();
        value
            .accept_product("assay-result", "independent-assay-receipt", 44)
            .unwrap();
        assert!(value.accepted_product);
    }

    #[test]
    fn campaign_cannot_close_with_active_stages_or_pending_results() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        assert_eq!(
            value.close_campaign("closeout", 11),
            Err(WorkflowError::RequiredStageIncomplete(
                "baseline".to_string()
            ))
        );
        value.fail_stage("baseline", "failure-receipt", 12).unwrap();
        assert_eq!(
            value.close_campaign("closeout", 13),
            Err(WorkflowError::RequiredStageIncomplete(
                "baseline".to_string()
            ))
        );
    }

    #[test]
    fn duplicate_observation_and_backwards_event_time_are_rejected() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        value
            .register_pending_observation(
                "baseline",
                "same-id",
                OutputKind::IntermediateProxy,
                11,
                11,
            )
            .unwrap();
        assert_eq!(
            value.register_pending_observation(
                "baseline",
                "same-id",
                OutputKind::IntermediateProxy,
                12,
                12
            ),
            Err(WorkflowError::DuplicateObservation("same-id".to_string()))
        );
        let before = value.clone();
        assert_eq!(
            value.pause_stage("baseline", "checkpoint", 10),
            Err(WorkflowError::EventTimeWentBackwards)
        );
        assert_eq!(value, before);
    }

    #[test]
    fn event_log_has_contiguous_sequence_and_non_decreasing_record_times() {
        let mut value = workflow();
        value.start_stage("baseline", 10).unwrap();
        register_and_qualify(
            &mut value,
            "baseline",
            "reading",
            OutputKind::IntermediateProxy,
            11,
        );
        value.complete_stage("baseline", 12).unwrap();
        assert!(value.validate_event_log().is_ok());
        for (index, event) in value.events.iter().enumerate() {
            assert_eq!(event.sequence, index as u64);
        }
    }

    #[test]
    fn late_observation_is_rejected_after_campaign_close() {
        let mini_graph = StageGraph {
            stages: vec![
                StageSpec {
                    id: "assay".to_string(),
                    prerequisites: vec![],
                    output_kind: OutputKind::IndependentProductAssay,
                    requires_qualified_observation: false,
                    required_for_product_acceptance: true,
                    required_for_campaign_close: true,
                    exclusive_resource: None,
                },
            ],
        };
        let mut value = CampaignWorkflow::new(lineage(), mini_graph).unwrap();
        value.start_stage("assay", 10).unwrap();
        value.complete_stage("assay", 11).unwrap();
        value.close_campaign("closeout", 12).unwrap();
        assert_eq!(
            value.register_pending_observation(
                "assay",
                "late-result",
                OutputKind::IndependentProductAssay,
                10,
                13
            ),
            Err(WorkflowError::CampaignClosed)
        );
    }
}

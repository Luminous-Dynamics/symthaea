// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Minimal synthetic mechanical environment for HLB-VSSI-001.
//!
//! The fixture is deliberately a small, deterministic two-state thermal/vibration
//! system. Its purpose is to qualify the benchmark interface and fault taxonomy,
//! not to represent a physical machine or establish industrial validity.
//!
//! The evaluated investigator receives only VssiPublicTask and observations
//! returned for selected experiments. The hidden case and disposition oracle
//! remain inside VssiMechanicalEnvironment. The benchmark runner must keep the
//! environment handle and its evaluator-only methods outside the investigator
//! process; this module does not itself provide a process sandbox.
//!
//! State, parameter, structural, observation-model and out-of-support faults are
//! deliberately disjoint. In particular, the out-of-support case adds an
//! unregistered nonlinear mechanism and must be met with abstention rather than
//! forced classification or a parameter patch.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

/// Fault family selected by the benchmark generator, never included in the
/// public task envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiFixtureCase {
    HealthyControl,
    InitialStateFault,
    ParameterFault,
    StructuralFault,
    ObservationModelFault,
    OpenWorldFault,
}

/// Operation the investigator proposes. It is a proposal only; this fixture
/// never promotes a candidate or mutates a production model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiDisposition {
    NoChange,
    StateUpdate,
    ParameterRepair,
    StructuralRepair,
    ObservationModelRepair,
    Abstain,
}

/// Registered observation or intervention protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VssiExperimentKind {
    PassiveObservation,
    ThermalImpulse,
    AlternatingLoad,
}

/// Nominal two-state mechanical model supplied to the investigator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiMechanicalParameters {
    pub temperature_retention: f64,
    pub temperature_from_vibration: f64,
    pub vibration_from_temperature: f64,
    pub vibration_retention: f64,
    pub load_to_temperature: f64,
    pub load_to_vibration: f64,
}

impl Default for VssiMechanicalParameters {
    fn default() -> Self {
        Self {
            temperature_retention: 0.82,
            temperature_from_vibration: 0.12,
            vibration_from_temperature: 0.18,
            vibration_retention: 0.72,
            load_to_temperature: 0.20,
            load_to_vibration: 0.35,
        }
    }
}

/// One scalar sensor reading. Event time and acquisition time are kept
/// separate so the fixture can expose delayed and asynchronous observations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiSensorReading {
    pub value: f64,
    pub acquired_at: f64,
}

/// One event-time row. Either sensor may be absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VssiObservation {
    pub event_time: f64,
    pub temperature: Option<VssiSensorReading>,
    pub vibration: Option<VssiSensorReading>,
}

/// Public metadata for a registered experiment. Costs are abstract fixture
/// units, not money, energy measurements or calibrated physical risk.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VssiExperimentSpec {
    pub kind: VssiExperimentKind,
    pub observation_cost: f64,
    pub minimum_load: f64,
    pub maximum_load: f64,
    pub duration_ticks: u8,
}

/// Investigator-visible task. It intentionally contains no fault-family or
/// expected-disposition field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VssiPublicTask {
    pub scenario_id: String,
    pub nominal_parameters: VssiMechanicalParameters,
    pub initial_observations: Vec<VssiObservation>,
    pub available_experiments: Vec<VssiExperimentSpec>,
    /// Maximum cumulative cost of investigations in abstract fixture units.
    pub observation_budget: f64,
    /// Declared correlated sensor-noise covariance for the synthetic fixture.
    /// It is a model input, not an empirical calibration claim.
    pub observation_covariance: [[f64; 2]; 2],
}

/// One completed experiment with provenance, budget accounting and a chained digest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VssiExperimentReceipt {
    /// Investigator-visible scenario identity.
    pub scenario_id: String,
    /// Digest of the complete investigator-visible task envelope.
    pub public_task_digest: String,
    /// One-based sequence number within this evaluator-held environment.
    pub experiment_index: u64,
    /// Experiment protocol that produced the observation batch.
    pub kind: VssiExperimentKind,
    /// Digest of the exact registered protocol specification in the public task.
    pub experiment_spec_digest: String,
    /// Abstract fixture cost charged for this protocol.
    pub observation_cost: f64,
    /// Remaining abstract cost budget immediately before the experiment.
    pub budget_before: f64,
    /// Remaining abstract cost budget immediately after the experiment.
    pub budget_after: f64,
    /// Digest of the exact observations released by this experiment.
    pub observations_digest: String,
    /// Previous receipt in the same environment's evidence chain.
    pub previous_receipt_digest: Option<String>,
    /// Domain-separated digest of all receipt claims.
    pub receipt_digest: String,
}

/// Observations released to the investigator together with their receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VssiExperimentResult {
    pub observations: Vec<VssiObservation>,
    pub receipt: VssiExperimentReceipt,
}

impl VssiExperimentReceipt {
    /// Recompute the receipt's claim digest without trusting its stored digest.
    pub fn validate(&self) -> Result<(), VssiFixtureError> {
        let expected_budget_after = (self.budget_before - self.observation_cost).max(0.0);
        if self.scenario_id.trim().is_empty()
            || self.experiment_index == 0
            || !self.observation_cost.is_finite()
            || self.observation_cost < 0.0
            || !self.budget_before.is_finite()
            || self.budget_before < 0.0
            || !self.budget_after.is_finite()
            || self.budget_after < 0.0
            || self.observation_cost > self.budget_before + 1e-12
            || (self.budget_after - expected_budget_after).abs() > 1e-12
        {
            return Err(VssiFixtureError::InvalidReceiptClaims);
        }
        if !is_sha256_hex(&self.public_task_digest)
            || !is_sha256_hex(&self.experiment_spec_digest)
            || !is_sha256_hex(&self.observations_digest)
            || !is_sha256_hex(&self.receipt_digest)
            || self
                .previous_receipt_digest
                .as_ref()
                .is_some_and(|digest| !is_sha256_hex(digest))
        {
            return Err(VssiFixtureError::InvalidReceiptDigest);
        }
        let claims = VssiExperimentReceiptClaims {
            schema_version: "vssi-experiment-receipt-v1",
            scenario_id: &self.scenario_id,
            public_task_digest: &self.public_task_digest,
            experiment_index: self.experiment_index,
            kind: self.kind,
            experiment_spec_digest: &self.experiment_spec_digest,
            observation_cost: self.observation_cost,
            budget_before: self.budget_before,
            budget_after: self.budget_after,
            observations_digest: &self.observations_digest,
            previous_receipt_digest: self.previous_receipt_digest.as_deref(),
        };
        if sha256_serializable(b"symthaea-vssi-experiment-receipt-v1\0", &claims)?
            != self.receipt_digest
        {
            return Err(VssiFixtureError::ReceiptDigestMismatch);
        }
        Ok(())
    }
}

impl VssiExperimentResult {
    /// Recompute both observation and receipt digests for this released batch.
    pub fn validate(&self) -> Result<(), VssiFixtureError> {
        self.receipt.validate()?;
        let digest =
            sha256_serializable(b"symthaea-vssi-observation-batch-v1\0", &self.observations)?;
        if digest != self.receipt.observations_digest {
            return Err(VssiFixtureError::ObservationDigestMismatch);
        }
        Ok(())
    }
}

/// Validate sequence continuity and the hash chain for one environment ledger.
///
/// This establishes internal integrity and ordering, not signer authenticity:
/// the fixture does not hold a signing key or provide an OS process sandbox.
pub fn validate_vssi_experiment_receipt_chain(
    receipts: &[VssiExperimentReceipt],
) -> Result<(), VssiFixtureError> {
    for (index, receipt) in receipts.iter().enumerate() {
        receipt.validate()?;
        if receipt.experiment_index != index as u64 + 1 {
            return Err(VssiFixtureError::ReceiptChainMismatch {
                index: receipt.experiment_index,
            });
        }
        if index == 0 {
            if receipt.previous_receipt_digest.is_some() {
                return Err(VssiFixtureError::ReceiptChainMismatch {
                    index: receipt.experiment_index,
                });
            }
        } else {
            let previous = &receipts[index - 1];
            if receipt.scenario_id != previous.scenario_id
                || receipt.public_task_digest != previous.public_task_digest
                || receipt.previous_receipt_digest.as_deref()
                    != Some(previous.receipt_digest.as_str())
                || (receipt.budget_before - previous.budget_after).abs() > 1e-12
            {
                return Err(VssiFixtureError::ReceiptChainMismatch {
                    index: receipt.experiment_index,
                });
            }
        }
    }
    Ok(())
}

/// Independently validate released experiment batches against the public task.
/// This checks the complete task digest, exact registered protocol specification,
/// one-use protocol identity, observation payload digest and cumulative budget.
/// It does not authenticate an actor or provide an OS process boundary.
pub fn validate_vssi_experiment_results(
    task: &VssiPublicTask,
    results: &[VssiExperimentResult],
) -> Result<(), VssiFixtureError> {
    if task.scenario_id.trim().is_empty()
        || !task.observation_budget.is_finite()
        || task.observation_budget < 0.0
    {
        return Err(VssiFixtureError::InvalidReceiptClaims);
    }
    let mut registered = BTreeSet::new();
    for spec in &task.available_experiments {
        if !registered.insert(spec.kind)
            || !spec.observation_cost.is_finite()
            || spec.observation_cost < 0.0
            || !spec.minimum_load.is_finite()
            || !spec.maximum_load.is_finite()
            || spec.minimum_load > spec.maximum_load
            || spec.duration_ticks == 0
        {
            return Err(VssiFixtureError::ReceiptSpecMismatch);
        }
    }

    let task_digest =
        sha256_serializable(b"symthaea-vssi-public-task-v1\0", task)?;
    let mut previous_receipt_digest: Option<&str> = None;
    let mut remaining_budget = task.observation_budget;
    let mut used_protocols = BTreeSet::new();
    for (index, result) in results.iter().enumerate() {
        result.validate()?;
        let receipt = &result.receipt;
        if receipt.scenario_id != task.scenario_id || receipt.public_task_digest != task_digest {
            return Err(VssiFixtureError::ReceiptTaskMismatch);
        }
        if receipt.experiment_index != index as u64 + 1
            || receipt.previous_receipt_digest.as_deref() != previous_receipt_digest
            || (receipt.budget_before - remaining_budget).abs() > 1e-12
        {
            return Err(VssiFixtureError::ReceiptChainMismatch {
                index: receipt.experiment_index,
            });
        }
        if !used_protocols.insert(receipt.kind) {
            return Err(VssiFixtureError::DuplicateReceiptExperiment);
        }
        let spec = task
            .available_experiments
            .iter()
            .find(|candidate| candidate.kind == receipt.kind)
            .ok_or(VssiFixtureError::ReceiptSpecMismatch)?;
        let expected_spec_digest =
            sha256_serializable(b"symthaea-vssi-experiment-spec-v1\0", spec)?;
        if receipt.experiment_spec_digest != expected_spec_digest
            || (receipt.observation_cost - spec.observation_cost).abs() > 1e-12
            || receipt.observation_cost > remaining_budget + 1e-12
        {
            return Err(VssiFixtureError::ReceiptSpecMismatch);
        }

        for observation in &result.observations {
            if !observation.event_time.is_finite()
                || observation.event_time <= 3.0
                || (observation.temperature.is_none() && observation.vibration.is_none())
            {
                return Err(VssiFixtureError::ReceiptObservationOutOfRange);
            }
            for reading in [observation.temperature, observation.vibration]
                .into_iter()
                .flatten()
            {
                if !reading.value.is_finite()
                    || !reading.acquired_at.is_finite()
                    || reading.acquired_at <= observation.event_time
                {
                    return Err(VssiFixtureError::ReceiptObservationOutOfRange);
                }
            }
        }

        remaining_budget = (remaining_budget - spec.observation_cost).max(0.0);
        if (receipt.budget_after - remaining_budget).abs() > 1e-12 {
            return Err(VssiFixtureError::ReceiptChainMismatch {
                index: receipt.experiment_index,
            });
        }
        previous_receipt_digest = Some(receipt.receipt_digest.as_str());
    }
    Ok(())
}

#[derive(Serialize)]
struct VssiExperimentReceiptClaims<'a> {
    schema_version: &'static str,
    scenario_id: &'a str,
    public_task_digest: &'a str,
    experiment_index: u64,
    kind: VssiExperimentKind,
    experiment_spec_digest: &'a str,
    observation_cost: f64,
    budget_before: f64,
    budget_after: f64,
    observations_digest: &'a str,
    previous_receipt_digest: Option<&'a str>,
}

/// Evaluator-only hidden mechanism. Its fields are private by design.
#[derive(Debug, Clone)]
struct HiddenMechanism {
    case: VssiFixtureCase,
    actual_parameters: VssiMechanicalParameters,
    initial_temperature_offset: f64,
    temperature_sensor_bias: f64,
    vibration_sensor_gain: f64,
    unregistered_temperature_nonlinearity: f64,
}

/// Environment handle. Pass the public task and selected experiment
/// observations to the investigator, but keep this handle in the evaluator.
/// It is intentionally not `Clone`: cloning an environment would fork its
/// one-use intervention ledger and duplicate the available observation budget.
#[derive(Debug)]
pub struct VssiMechanicalEnvironment {
    task: VssiPublicTask,
    hidden: HiddenMechanism,
    spent_observation_cost: f64,
    completed_experiments: Vec<VssiExperimentKind>,
    experiment_receipts: Vec<VssiExperimentReceipt>,
}

/// A disposition score intended only for the independent benchmark evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct VssiDispositionScore {
    pub correct_disposition: bool,
    pub unnecessary_edit: bool,
    pub correct_abstention: bool,
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum VssiFixtureError {
    #[error("scenario identifier must not be empty")]
    EmptyScenarioId,
    #[error("synthetic trajectory became non-finite at tick {tick}")]
    NonFiniteTrajectory { tick: u8 },
    #[error("experiment {kind:?} is not registered for this task")]
    UnregisteredExperiment { kind: VssiExperimentKind },
    #[error("experiment protocol {kind:?} has already been performed")]
    ExperimentAlreadyPerformed { kind: VssiExperimentKind },
    #[error("experiment cost {requested} exceeds remaining budget {remaining}")]
    BudgetExceeded { remaining: f64, requested: f64 },
    #[error("failed to serialize deterministic experiment evidence for hashing")]
    ReceiptSerialization,
    #[error("receipt contains invalid sequence, cost, or budget claims")]
    InvalidReceiptClaims,
    #[error("receipt sequence or previous-digest link is inconsistent at index {index}")]
    ReceiptChainMismatch { index: u64 },
    #[error("receipt contains a malformed SHA-256 digest")]
    InvalidReceiptDigest,
    #[error("experiment receipt digest does not match its claims")]
    ReceiptDigestMismatch,
    #[error("released observations do not match the experiment receipt digest")]
    ObservationDigestMismatch,
    #[error("experiment receipt is not bound to the supplied public task")]
    ReceiptTaskMismatch,
    #[error("experiment receipt does not match its registered experiment specification")]
    ReceiptSpecMismatch,
    #[error("experiment protocol appears more than once in a receipt sequence")]
    DuplicateReceiptExperiment,
    #[error("released observation is outside the allowed post-intervention time/quality contract")]
    ReceiptObservationOutOfRange,
}

impl VssiMechanicalEnvironment {
    /// Create one deterministic task from a hidden benchmark case.
    ///
    /// The case is a generator/evaluator input. Do not forward it to the
    /// evaluated investigator or serialize it into the public task manifest.
    pub fn new(
        scenario_id: impl Into<String>,
        case: VssiFixtureCase,
    ) -> Result<Self, VssiFixtureError> {
        let scenario_id = scenario_id.into();
        if scenario_id.trim().is_empty() {
            return Err(VssiFixtureError::EmptyScenarioId);
        }

        let nominal_parameters = VssiMechanicalParameters::default();
        let hidden = hidden_mechanism(case, nominal_parameters);
        let initial_observations = simulate(&hidden, VssiExperimentKind::PassiveObservation)?
            .into_iter()
            .filter(|observation| observation.event_time <= 3.0)
            .collect();

        let task = VssiPublicTask {
            scenario_id,
            nominal_parameters,
            initial_observations,
            available_experiments: vec![
                VssiExperimentSpec {
                    kind: VssiExperimentKind::PassiveObservation,
                    observation_cost: 0.25,
                    minimum_load: 0.15,
                    maximum_load: 0.15,
                    duration_ticks: 12,
                },
                VssiExperimentSpec {
                    kind: VssiExperimentKind::ThermalImpulse,
                    observation_cost: 0.7,
                    minimum_load: 0.15,
                    maximum_load: 0.95,
                    duration_ticks: 16,
                },
                VssiExperimentSpec {
                    kind: VssiExperimentKind::AlternatingLoad,
                    observation_cost: 1.0,
                    minimum_load: -0.35,
                    maximum_load: 0.75,
                    duration_ticks: 16,
                },
            ],
            observation_budget: 1.2,
            observation_covariance: [[0.04, 0.028], [0.028, 0.05]],
        };

        Ok(Self {
            task,
            hidden,
            spent_observation_cost: 0.0,
            completed_experiments: Vec::new(),
            experiment_receipts: Vec::new(),
        })
    }

    /// Return the investigator-visible task envelope.
    pub fn public_task(&self) -> &VssiPublicTask {
        &self.task
    }

    /// Release the post-baseline observations for one selected experiment.
    ///
    /// Every protocol shares the same tick-0-through-tick-3 baseline. The
    /// selected intervention begins at tick 4. A protocol may be requested
    /// only once per environment; replaying an identical counterfactual would
    /// otherwise return duplicate deterministic observations.
    ///
    /// The compatibility method still returns observations only; every
    /// successful call records a receipt in the evaluator-held ledger.
    pub fn observe_experiment(
        &mut self,
        kind: VssiExperimentKind,
    ) -> Result<Vec<VssiObservation>, VssiFixtureError> {
        Ok(self.observe_experiment_with_receipt(kind)?.observations)
    }

    /// Release observations and return the exact provenance/budget receipt.
    ///
    /// Receipt serialization and hashing complete before the environment is
    /// mutated or observations are released. Failed requests therefore do not
    /// consume budget, advance the receipt chain, or reveal a partial result.
    pub fn observe_experiment_with_receipt(
        &mut self,
        kind: VssiExperimentKind,
    ) -> Result<VssiExperimentResult, VssiFixtureError> {
        let spec = self
            .task
            .available_experiments
            .iter()
            .find(|spec| spec.kind == kind)
            .copied()
            .ok_or(VssiFixtureError::UnregisteredExperiment { kind })?;
        if self.completed_experiments.contains(&kind) {
            return Err(VssiFixtureError::ExperimentAlreadyPerformed { kind });
        }
        let budget_before = self.remaining_observation_budget();
        if spec.observation_cost > budget_before + 1e-12 {
            return Err(VssiFixtureError::BudgetExceeded {
                remaining: budget_before,
                requested: spec.observation_cost,
            });
        }

        let observations = simulate(&self.hidden, kind)?
            .into_iter()
            .filter(|observation| observation.event_time > 3.0)
            .collect::<Vec<_>>();
        let budget_after = (budget_before - spec.observation_cost).max(0.0);
        let public_task_digest =
            sha256_serializable(b"symthaea-vssi-public-task-v1\0", &self.task)?;
        let observations_digest =
            sha256_serializable(b"symthaea-vssi-observation-batch-v1\0", &observations)?;
        let experiment_spec_digest =
            sha256_serializable(b"symthaea-vssi-experiment-spec-v1\0", &spec)?;
        let experiment_index = self.experiment_receipts.len() as u64 + 1;
        let previous_receipt_digest = self
            .experiment_receipts
            .last()
            .map(|receipt| receipt.receipt_digest.clone());
        let claims = VssiExperimentReceiptClaims {
            schema_version: "vssi-experiment-receipt-v1",
            scenario_id: &self.task.scenario_id,
            public_task_digest: &public_task_digest,
            experiment_index,
            kind,
            experiment_spec_digest: &experiment_spec_digest,
            observation_cost: spec.observation_cost,
            budget_before,
            budget_after,
            observations_digest: &observations_digest,
            previous_receipt_digest: previous_receipt_digest.as_deref(),
        };
        let receipt_digest =
            sha256_serializable(b"symthaea-vssi-experiment-receipt-v1\0", &claims)?;
        let receipt = VssiExperimentReceipt {
            scenario_id: self.task.scenario_id.clone(),
            public_task_digest,
            experiment_index,
            kind,
            experiment_spec_digest,
            observation_cost: spec.observation_cost,
            budget_before,
            budget_after,
            observations_digest,
            previous_receipt_digest,
            receipt_digest,
        };

        let result = VssiExperimentResult {
            observations,
            receipt,
        };
        result.validate()?;

        self.spent_observation_cost += spec.observation_cost;
        self.completed_experiments.push(kind);
        self.experiment_receipts.push(result.receipt.clone());
        Ok(result)
    }

    /// Receipts for all successful experiments, in immutable execution order.
    pub fn experiment_receipts(&self) -> &[VssiExperimentReceipt] {
        &self.experiment_receipts
    }

    /// Cumulative abstract experiment cost consumed by this environment.
    pub fn spent_observation_cost(&self) -> f64 {
        self.spent_observation_cost
    }

    /// Remaining abstract experiment budget.
    pub fn remaining_observation_budget(&self) -> f64 {
        (self.task.observation_budget - self.spent_observation_cost).max(0.0)
    }

    /// Unit-test oracle only. The independent campaign evaluator must live
    /// outside the investigator-facing crate/process boundary.
    #[cfg(test)]
    pub(crate) fn score_disposition(
        &self,
        proposed: VssiDisposition,
    ) -> VssiDispositionScore {
        let expected = match self.hidden.case {
            VssiFixtureCase::HealthyControl => VssiDisposition::NoChange,
            VssiFixtureCase::InitialStateFault => VssiDisposition::StateUpdate,
            VssiFixtureCase::ParameterFault => VssiDisposition::ParameterRepair,
            VssiFixtureCase::StructuralFault => VssiDisposition::StructuralRepair,
            VssiFixtureCase::ObservationModelFault => VssiDisposition::ObservationModelRepair,
            VssiFixtureCase::OpenWorldFault => VssiDisposition::Abstain,
        };
        VssiDispositionScore {
            correct_disposition: proposed == expected,
            unnecessary_edit: self.hidden.case == VssiFixtureCase::HealthyControl
                && matches!(
                    proposed,
                    VssiDisposition::StateUpdate
                        | VssiDisposition::ParameterRepair
                        | VssiDisposition::StructuralRepair
                        | VssiDisposition::ObservationModelRepair
                ),
            correct_abstention: self.hidden.case == VssiFixtureCase::OpenWorldFault
                && proposed == VssiDisposition::Abstain,
        }
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_serializable<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<String, VssiFixtureError> {
    let encoded = serde_json::to_vec(value).map_err(|_| VssiFixtureError::ReceiptSerialization)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    Ok(format!("{:x}", hasher.finalize()))
}

fn hidden_mechanism(
    case: VssiFixtureCase,
    nominal: VssiMechanicalParameters,
) -> HiddenMechanism {
    let mut actual_parameters = nominal;
    let mut initial_temperature_offset = 0.0;
    let mut temperature_sensor_bias = 0.0;
    let mut vibration_sensor_gain = 1.0;
    let mut unregistered_temperature_nonlinearity = 0.0;

    match case {
        VssiFixtureCase::HealthyControl => {}
        VssiFixtureCase::InitialStateFault => {
            initial_temperature_offset = 1.15;
        }
        VssiFixtureCase::ParameterFault => {
            actual_parameters.temperature_retention = 0.57;
        }
        VssiFixtureCase::StructuralFault => {
            // The nominal temperature <- vibration causal edge is absent.
            actual_parameters.temperature_from_vibration = 0.0;
        }
        VssiFixtureCase::ObservationModelFault => {
            temperature_sensor_bias = 0.65;
            vibration_sensor_gain = 1.22;
        }
        VssiFixtureCase::OpenWorldFault => {
            // This nonlinear mechanism is deliberately absent from the
            // registered nominal/parameter/structural repair vocabulary.
            unregistered_temperature_nonlinearity = 0.16;
        }
    }

    HiddenMechanism {
        case,
        actual_parameters,
        initial_temperature_offset,
        temperature_sensor_bias,
        vibration_sensor_gain,
        unregistered_temperature_nonlinearity,
    }
}

fn experiment_load(kind: VssiExperimentKind, tick: u8) -> f64 {
    // Every experiment begins from the same pre-intervention operating history.
    if tick < 4 {
        return 0.15;
    }
    match kind {
        VssiExperimentKind::PassiveObservation => 0.15,
        VssiExperimentKind::ThermalImpulse => {
            if (4..=6).contains(&tick) { 0.95 } else { 0.15 }
        }
        VssiExperimentKind::AlternatingLoad => {
            if (tick - 4) % 4 < 2 { 0.75 } else { -0.35 }
        }
    }
}

fn simulate(
    hidden: &HiddenMechanism,
    experiment: VssiExperimentKind,
) -> Result<Vec<VssiObservation>, VssiFixtureError> {
    const TEMPERATURE_TICKS: [u8; 6] = [0, 1, 4, 7, 11, 15];
    const VIBRATION_TICKS: [u8; 6] = [0, 2, 4, 8, 11, 15];
    const COMMON_MODE: [f64; 8] = [0.0, 0.02, -0.015, 0.03, -0.02, 0.012, -0.01, 0.018];
    const TEMPERATURE_NOISE: [f64; 8] = [0.01, -0.01, 0.005, -0.006, 0.008, -0.004, 0.003, -0.007];
    const VIBRATION_NOISE: [f64; 8] = [-0.006, 0.004, -0.003, 0.007, -0.005, 0.002, -0.004, 0.006];

    let parameters = hidden.actual_parameters;
    let mut temperature = 0.25 + hidden.initial_temperature_offset;
    let mut vibration = 0.08;
    let mut observations = Vec::new();

    for tick in 0..16_u8 {
        let common = COMMON_MODE[(tick as usize) % COMMON_MODE.len()];
        let temperature_value = temperature
            + hidden.temperature_sensor_bias
            + common
            + TEMPERATURE_NOISE[(tick as usize) % TEMPERATURE_NOISE.len()];
        let vibration_value = hidden.vibration_sensor_gain * vibration
            + 0.9 * common
            + VIBRATION_NOISE[(tick as usize) % VIBRATION_NOISE.len()];

        let temperature_reading = TEMPERATURE_TICKS.contains(&tick).then_some(
            VssiSensorReading {
                value: temperature_value,
                acquired_at: tick as f64 + 0.12,
            },
        );
        let vibration_reading = VIBRATION_TICKS.contains(&tick).then_some(
            VssiSensorReading {
                value: vibration_value,
                acquired_at: tick as f64 + 0.31,
            },
        );
        if temperature_reading.is_some() || vibration_reading.is_some() {
            observations.push(VssiObservation {
                event_time: tick as f64,
                temperature: temperature_reading,
                vibration: vibration_reading,
            });
        }

        if !temperature_value.is_finite() || !vibration_value.is_finite() {
            return Err(VssiFixtureError::NonFiniteTrajectory { tick });
        }

        let load = experiment_load(experiment, tick);
        let next_temperature = parameters.temperature_retention * temperature
            + parameters.temperature_from_vibration * vibration
            + parameters.load_to_temperature * load
            + hidden.unregistered_temperature_nonlinearity * temperature.powi(2);
        let next_vibration = parameters.vibration_from_temperature * temperature
            + parameters.vibration_retention * vibration
            + parameters.load_to_vibration * load;

        if !next_temperature.is_finite() || !next_vibration.is_finite() {
            return Err(VssiFixtureError::NonFiniteTrajectory { tick });
        }
        temperature = next_temperature;
        vibration = next_vibration;
    }

    Ok(observations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_registered_case_has_an_explicit_expected_disposition() {
        let cases = [
            (VssiFixtureCase::HealthyControl, VssiDisposition::NoChange),
            (VssiFixtureCase::InitialStateFault, VssiDisposition::StateUpdate),
            (VssiFixtureCase::ParameterFault, VssiDisposition::ParameterRepair),
            (VssiFixtureCase::StructuralFault, VssiDisposition::StructuralRepair),
            (
                VssiFixtureCase::ObservationModelFault,
                VssiDisposition::ObservationModelRepair,
            ),
            (VssiFixtureCase::OpenWorldFault, VssiDisposition::Abstain),
        ];

        for (index, (case, expected)) in cases.into_iter().enumerate() {
            let environment =
                VssiMechanicalEnvironment::new(format!("vssi-v0-1-{index}"), case).unwrap();
            assert!(environment.score_disposition(expected).correct_disposition);
        }
    }

    #[test]
    fn healthy_control_penalizes_unnecessary_model_edits() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-healthy-negative-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let score = environment.score_disposition(VssiDisposition::ParameterRepair);
        assert!(!score.correct_disposition);
        assert!(score.unnecessary_edit);
        assert!(!score.correct_abstention);
    }

    #[test]
    fn open_world_fault_requires_abstention() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-open-world-negative-control",
            VssiFixtureCase::OpenWorldFault,
        )
        .unwrap();

        let abstention = environment.score_disposition(VssiDisposition::Abstain);
        let forced_repair = environment.score_disposition(VssiDisposition::ParameterRepair);
        assert!(abstention.correct_abstention);
        assert!(!forced_repair.correct_disposition);
    }

    #[test]
    fn task_envelope_does_not_serialize_the_hidden_fault_label() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-hidden-label-control",
            VssiFixtureCase::StructuralFault,
        )
        .unwrap();

        let serialized = serde_json::to_string(environment.public_task()).unwrap();
        assert!(!serialized.contains("StructuralFault"));
        assert!(!serialized.contains("expected_disposition"));
        assert!(!serialized.contains("fault_family"));
    }

    #[test]
    fn investigation_protocols_produce_distinct_reproducible_traces() {
        let mut pulse_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-pulse-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let pulse = pulse_environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();

        let mut pulse_replay_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-pulse-replay-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let pulse_replay = pulse_replay_environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();

        let mut alternating_environment = VssiMechanicalEnvironment::new(
            "vssi-investigation-alternating-control",
            VssiFixtureCase::ParameterFault,
        )
        .unwrap();
        let alternating = alternating_environment
            .observe_experiment(VssiExperimentKind::AlternatingLoad)
            .unwrap();

        assert_eq!(pulse, pulse_replay);
        assert_ne!(pulse, alternating);
    }

    #[test]
    fn observations_keep_event_time_acquisition_time_and_missingness_separate() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-irregular-observation-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let trace = environment
            .observe_experiment(VssiExperimentKind::PassiveObservation)
            .unwrap();
        assert!(trace
            .iter()
            .any(|row| row.temperature.is_none() && row.vibration.is_some()));
        assert!(trace
            .iter()
            .any(|row| row.temperature.is_some() && row.vibration.is_none()));
        for row in trace {
            if let Some(reading) = row.temperature {
                assert!(reading.acquired_at > row.event_time);
                assert!(reading.value.is_finite());
            }
            if let Some(reading) = row.vibration {
                assert!(reading.acquired_at > row.event_time);
                assert!(reading.value.is_finite());
            }
        }
    }

    #[test]
    fn experiment_receipts_chain_task_identity_observations_and_budget() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-receipt-chain-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let first = environment
            .observe_experiment_with_receipt(VssiExperimentKind::PassiveObservation)
            .unwrap();
        assert_eq!(first.receipt.experiment_index, 1);
        assert_eq!(first.receipt.kind, VssiExperimentKind::PassiveObservation);
        assert_eq!(first.receipt.observation_cost, 0.25);
        assert_eq!(first.receipt.budget_before, 1.2);
        assert!((first.receipt.budget_after - 0.95).abs() < 1e-12);
        assert_eq!(first.receipt.previous_receipt_digest, None);
        assert_eq!(first.receipt.observations_digest.len(), 64);
        assert_eq!(first.receipt.public_task_digest.len(), 64);
        first.validate().unwrap();
        assert_eq!(
            first.receipt.observations_digest,
            sha256_serializable(
                b"symthaea-vssi-observation-batch-v1\0",
                &first.observations
            )
            .unwrap()
        );

        let second = environment
            .observe_experiment_with_receipt(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        assert_eq!(second.receipt.experiment_index, 2);
        assert_eq!(
            second.receipt.previous_receipt_digest.as_deref(),
            Some(first.receipt.receipt_digest.as_str())
        );
        assert!((second.receipt.budget_before - first.receipt.budget_after).abs() < 1e-12);
        assert!(second.receipt.budget_after < second.receipt.budget_before);
        assert_eq!(environment.experiment_receipts().len(), 2);
        assert_eq!(
            environment.experiment_receipts()[1].receipt_digest,
            second.receipt.receipt_digest
        );
        validate_vssi_experiment_receipt_chain(environment.experiment_receipts()).unwrap();

        let mut reordered = environment.experiment_receipts().to_vec();
        reordered.swap(0, 1);
        assert!(matches!(
            validate_vssi_experiment_receipt_chain(&reordered),
            Err(VssiFixtureError::ReceiptChainMismatch { .. })
        ));
    }

    #[test]
    fn experiment_result_detects_observation_and_receipt_tampering() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-receipt-tamper-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();
        let result = environment
            .observe_experiment_with_receipt(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        result.validate().unwrap();

        let mut altered_observations = result.clone();
        altered_observations.observations[0].event_time += 0.25;
        assert_eq!(
            altered_observations.validate(),
            Err(VssiFixtureError::ObservationDigestMismatch)
        );

        let mut altered_receipt = result;
        altered_receipt.receipt.budget_after += 0.01;
        assert_eq!(
            altered_receipt.validate(),
            Err(VssiFixtureError::InvalidReceiptClaims)
        );
    }

    #[test]
    fn failed_experiment_does_not_advance_receipt_chain_or_budget() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-failed-receipt-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();
        let first = environment
            .observe_experiment_with_receipt(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        let receipts_before = environment.experiment_receipts().len();
        let spent_before = environment.spent_observation_cost();

        let rejected =
            environment.observe_experiment_with_receipt(VssiExperimentKind::AlternatingLoad);
        assert!(matches!(rejected, Err(VssiFixtureError::BudgetExceeded { .. })));
        assert_eq!(environment.experiment_receipts().len(), receipts_before);
        assert_eq!(environment.spent_observation_cost(), spent_before);
        assert_eq!(
            environment.experiment_receipts().last().unwrap().receipt_digest,
            first.receipt.receipt_digest
        );
    }

    #[test]
    fn experiment_catalog_declares_bounded_cost_and_input_envelopes() {
        let environment = VssiMechanicalEnvironment::new(
            "vssi-budget-contract-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let task = environment.public_task();
        assert_eq!(task.available_experiments.len(), 3);
        assert_eq!(task.observation_budget, 1.2);
        assert_eq!(task.available_experiments[0].observation_cost, 0.25);
        assert!(task.available_experiments.iter().all(|spec| {
            spec.observation_cost.is_finite()
                && spec.observation_cost >= 0.0
                && spec.minimum_load <= spec.maximum_load
                && spec.duration_ticks == 12
        }));
        assert!(task.observation_covariance[0][1] > 0.0);
        assert_eq!(
            task.observation_covariance[0][1],
            task.observation_covariance[1][0]
        );
    }

    #[test]
    fn experiment_budget_is_enforced_before_observations_are_released() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-budget-enforcement-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        assert!((environment.spent_observation_cost() - 0.7).abs() < 1e-12);
        assert!((environment.remaining_observation_budget() - 0.5).abs() < 1e-12);

        let rejected = environment.observe_experiment(VssiExperimentKind::AlternatingLoad);
        assert!(matches!(rejected, Err(VssiFixtureError::BudgetExceeded { .. })));
        assert!((environment.spent_observation_cost() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn repeating_the_same_experiment_is_rejected_without_duplicate_observations() {
        let mut environment = VssiMechanicalEnvironment::new(
            "vssi-passive-replay-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();

        let first = environment
            .observe_experiment(VssiExperimentKind::PassiveObservation)
            .unwrap();
        let spent_after_first = environment.spent_observation_cost();
        let rejected = environment.observe_experiment(VssiExperimentKind::PassiveObservation);
        assert!(matches!(
            rejected,
            Err(VssiFixtureError::ExperimentAlreadyPerformed {
                kind: VssiExperimentKind::PassiveObservation
            })
        ));
        let pulse = environment
            .observe_experiment(VssiExperimentKind::ThermalImpulse)
            .unwrap();
        assert_ne!(first, pulse);
        assert!(first.iter().all(|row| row.event_time > 3.0));
        assert!(pulse.iter().all(|row| row.event_time > 3.0));
        // Distinct protocols share the same pre-intervention task observations.
        let repeated_initial = VssiMechanicalEnvironment::new(
            "vssi-passive-replay-baseline-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();
        let pulse_initial = VssiMechanicalEnvironment::new(
            "vssi-thermal-baseline-control",
            VssiFixtureCase::HealthyControl,
        )
        .unwrap();
        assert_eq!(
            repeated_initial.public_task().initial_observations,
            pulse_initial.public_task().initial_observations
        );
        assert_eq!(spent_after_first, 0.25);
        assert!(environment.spent_observation_cost() > spent_after_first);
    }

    #[test]
    fn distinct_hidden_cases_are_observably_distinguishable_by_at_least_one_protocol() {
        let cases = [
            VssiFixtureCase::HealthyControl,
            VssiFixtureCase::InitialStateFault,
            VssiFixtureCase::ParameterFault,
            VssiFixtureCase::StructuralFault,
            VssiFixtureCase::ObservationModelFault,
            VssiFixtureCase::OpenWorldFault,
        ];
        let protocols = [
            VssiExperimentKind::PassiveObservation,
            VssiExperimentKind::ThermalImpulse,
            VssiExperimentKind::AlternatingLoad,
        ];

        for left_index in 0..cases.len() {
            for right_index in (left_index + 1)..cases.len() {
                let mut distinguishable = false;
                for protocol in protocols {
                    let mut left = VssiMechanicalEnvironment::new(
                        format!("vssi-separability-left-{left_index}-{right_index}"),
                        cases[left_index],
                    )
                    .unwrap();
                    let mut right = VssiMechanicalEnvironment::new(
                        format!("vssi-separability-right-{left_index}-{right_index}"),
                        cases[right_index],
                    )
                    .unwrap();
                    let left_trace = left.observe_experiment(protocol).unwrap();
                    let right_trace = right.observe_experiment(protocol).unwrap();
                    if left_trace != right_trace {
                        distinguishable = true;
                        break;
                    }
                }
                assert!(
                    distinguishable,
                    "hidden cases {left_index} and {right_index} have identical registered traces"
                );
            }
        }
    }

    #[test]
    fn empty_scenario_identifier_is_rejected() {
        assert_eq!(
            VssiMechanicalEnvironment::new("  ", VssiFixtureCase::HealthyControl).unwrap_err(),
            VssiFixtureError::EmptyScenarioId
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Claim-preserving observation envelope for humanoid capability evaluation.
//!
//! This module intentionally does not define an aggregate humanoid score. It
//! records what was tested, where, under which authority/evidence profile, and
//! which measurements/failures occurred. Qualification remains a separate step.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const HUMANOID_CAPABILITY_OBSERVATION_SCHEMA_V1: &str =
    "symthaea.humanoid.capability-observation.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapabilityDomain {
    Locomotion,
    BalanceRecovery,
    DexterousManipulation,
    MobileManipulation,
    HouseholdTask,
    PerceptionSensing,
    LanguageConditionedTask,
    LongHorizonTaskComposition,
    HumanRobotInteraction,
    PhysicalAssistance,
    SomaticContact,
    PowerEndurance,
    FaultRecovery,
    ConversationalSocial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionSubstrate {
    DeterministicSimulation,
    ExternalBenchmarkSimulation,
    InstrumentedFixture,
    HardwareInLoop,
    ControlledPhysical,
    NonContactHumanFactors,
    HumanContactStudy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementProvenance {
    Measured,
    Derived,
    BenchmarkNative,
    SimulationOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapabilityMeasurementV1 {
    pub metric_id: String,
    pub value: f64,
    pub unit: String,
    pub provenance: MeasurementProvenance,
}

impl CapabilityMeasurementV1 {
    fn validate(&self) -> Result<(), CapabilityObservationError> {
        validate_id(&self.metric_id, CapabilityObservationError::InvalidMetricId)?;
        if !self.value.is_finite() {
            return Err(CapabilityObservationError::NonFiniteMetric);
        }
        if self.unit.trim().is_empty() || self.unit.len() > 64 {
            return Err(CapabilityObservationError::InvalidMetricUnit);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureEventV1 {
    pub failure_id: String,
    pub category: String,
    pub description: String,
}

impl FailureEventV1 {
    fn validate(&self) -> Result<(), CapabilityObservationError> {
        validate_id(&self.failure_id, CapabilityObservationError::InvalidFailureId)?;
        validate_id(&self.category, CapabilityObservationError::InvalidFailureCategory)?;
        if self.description.trim().is_empty() || self.description.len() > 512 {
            return Err(CapabilityObservationError::InvalidFailureDescription);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BenchmarkIdentityV1 {
    Internal {
        protocol_id: String,
        protocol_version: String,
        case_set_id: String,
    },
    External {
        benchmark_id: String,
        benchmark_version: String,
        task_set_id: String,
    },
}

impl BenchmarkIdentityV1 {
    fn validate(&self) -> Result<(), CapabilityObservationError> {
        match self {
            Self::Internal {
                protocol_id,
                protocol_version,
                case_set_id,
            } => {
                validate_id(protocol_id, CapabilityObservationError::InvalidBenchmarkIdentity)?;
                validate_id(protocol_version, CapabilityObservationError::InvalidBenchmarkIdentity)?;
                validate_id(case_set_id, CapabilityObservationError::InvalidBenchmarkIdentity)
            }
            Self::External {
                benchmark_id,
                benchmark_version,
                task_set_id,
            } => {
                validate_id(benchmark_id, CapabilityObservationError::InvalidBenchmarkIdentity)?;
                validate_id(
                    benchmark_version,
                    CapabilityObservationError::InvalidBenchmarkIdentity,
                )?;
                validate_id(task_set_id, CapabilityObservationError::InvalidBenchmarkIdentity)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySubjectIdentityV1 {
    pub source_head: String,
    pub model_or_policy_id: String,
    pub morphology_id: String,
    pub sensor_actuator_profile_id: String,
}

impl CapabilitySubjectIdentityV1 {
    fn validate(&self) -> Result<(), CapabilityObservationError> {
        for value in [
            &self.source_head,
            &self.model_or_policy_id,
            &self.morphology_id,
            &self.sensor_actuator_profile_id,
        ] {
            validate_id(value, CapabilityObservationError::InvalidSubjectIdentity)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunDisposition {
    Completed,
    Failed,
    Inconclusive,
    InfrastructureIndeterminate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityObservationV1 {
    pub run_id: String,
    pub domain: CapabilityDomain,
    pub substrate: ExecutionSubstrate,
    pub subject: CapabilitySubjectIdentityV1,
    pub benchmark: BenchmarkIdentityV1,
    pub environment_id: String,
    pub authority_profile_id: String,
    pub evidence_profile_id: String,
    pub disposition: RunDisposition,
    pub measurements: Vec<CapabilityMeasurementV1>,
    pub failures: Vec<FailureEventV1>,
}

impl HumanoidCapabilityObservationV1 {
    pub fn new(value: Self) -> Result<Self, CapabilityObservationError> {
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), CapabilityObservationError> {
        validate_id(&self.run_id, CapabilityObservationError::InvalidRunId)?;
        self.subject.validate()?;
        self.benchmark.validate()?;
        validate_id(
            &self.environment_id,
            CapabilityObservationError::InvalidEnvironmentId,
        )?;
        validate_id(
            &self.authority_profile_id,
            CapabilityObservationError::InvalidAuthorityProfileId,
        )?;
        validate_id(
            &self.evidence_profile_id,
            CapabilityObservationError::InvalidEvidenceProfileId,
        )?;

        let mut metric_ids = BTreeSet::new();
        for measurement in &self.measurements {
            measurement.validate()?;
            if !metric_ids.insert(measurement.metric_id.as_str()) {
                return Err(CapabilityObservationError::DuplicateMetricId);
            }
        }

        let mut failure_ids = BTreeSet::new();
        for failure in &self.failures {
            failure.validate()?;
            if !failure_ids.insert(failure.failure_id.as_str()) {
                return Err(CapabilityObservationError::DuplicateFailureId);
            }
        }

        if matches!(self.disposition, RunDisposition::Failed) && self.failures.is_empty() {
            return Err(CapabilityObservationError::FailedRunMissingFailureEvidence);
        }

        if matches!(self.substrate, ExecutionSubstrate::ExternalBenchmarkSimulation)
            && !matches!(self.benchmark, BenchmarkIdentityV1::External { .. })
        {
            return Err(CapabilityObservationError::ExternalSubstrateRequiresExternalBenchmark);
        }

        Ok(())
    }

    pub fn has_failures(&self) -> bool {
        !self.failures.is_empty()
    }

    /// Intentionally returns raw measurements rather than an aggregate score.
    pub fn measurements(&self) -> &[CapabilityMeasurementV1] {
        &self.measurements
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityObservationError {
    InvalidRunId,
    InvalidSubjectIdentity,
    InvalidBenchmarkIdentity,
    InvalidEnvironmentId,
    InvalidAuthorityProfileId,
    InvalidEvidenceProfileId,
    InvalidMetricId,
    InvalidMetricUnit,
    NonFiniteMetric,
    DuplicateMetricId,
    InvalidFailureId,
    InvalidFailureCategory,
    InvalidFailureDescription,
    DuplicateFailureId,
    FailedRunMissingFailureEvidence,
    ExternalSubstrateRequiresExternalBenchmark,
}

fn validate_id(value: &str, error: CapabilityObservationError) -> Result<(), CapabilityObservationError> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 192 {
        Err(error)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> CapabilitySubjectIdentityV1 {
        CapabilitySubjectIdentityV1 {
            source_head: "deadbeef".into(),
            model_or_policy_id: "standing-baseline".into(),
            morphology_id: "humanoid-v1".into(),
            sensor_actuator_profile_id: "sim-default".into(),
        }
    }

    fn base_observation() -> HumanoidCapabilityObservationV1 {
        HumanoidCapabilityObservationV1 {
            run_id: "run-001".into(),
            domain: CapabilityDomain::BalanceRecovery,
            substrate: ExecutionSubstrate::DeterministicSimulation,
            subject: subject(),
            benchmark: BenchmarkIdentityV1::Internal {
                protocol_id: "push-recovery".into(),
                protocol_version: "v1".into(),
                case_set_id: "zero-push-eight-directions".into(),
            },
            environment_id: "simple-humanoid-sim".into(),
            authority_profile_id: "simulation-no-human-authority".into(),
            evidence_profile_id: "source-observation-only".into(),
            disposition: RunDisposition::Completed,
            measurements: vec![CapabilityMeasurementV1 {
                metric_id: "recovery_rate".into(),
                value: 1.0,
                unit: "ratio".into(),
                provenance: MeasurementProvenance::Measured,
            }],
            failures: vec![],
        }
    }

    #[test]
    fn valid_internal_observation_is_accepted() {
        assert!(HumanoidCapabilityObservationV1::new(base_observation()).is_ok());
    }

    #[test]
    fn internal_observation_requires_exact_case_set_identity() {
        let mut observation = base_observation();
        observation.benchmark = BenchmarkIdentityV1::Internal {
            protocol_id: "push-recovery".into(),
            protocol_version: "v1".into(),
            case_set_id: "".into(),
        };
        assert_eq!(
            observation.validate(),
            Err(CapabilityObservationError::InvalidBenchmarkIdentity)
        );
    }

    #[test]
    fn failed_run_requires_explicit_failure_evidence() {
        let mut observation = base_observation();
        observation.disposition = RunDisposition::Failed;
        assert_eq!(
            observation.validate(),
            Err(CapabilityObservationError::FailedRunMissingFailureEvidence)
        );
    }

    #[test]
    fn external_substrate_requires_external_benchmark_identity() {
        let mut observation = base_observation();
        observation.substrate = ExecutionSubstrate::ExternalBenchmarkSimulation;
        assert_eq!(
            observation.validate(),
            Err(CapabilityObservationError::ExternalSubstrateRequiresExternalBenchmark)
        );
    }

    #[test]
    fn duplicate_metric_ids_fail_closed() {
        let mut observation = base_observation();
        observation.measurements.push(observation.measurements[0].clone());
        assert_eq!(
            observation.validate(),
            Err(CapabilityObservationError::DuplicateMetricId)
        );
    }

    #[test]
    fn non_finite_metric_fails_closed() {
        let mut observation = base_observation();
        observation.measurements[0].value = f64::NAN;
        assert_eq!(
            observation.validate(),
            Err(CapabilityObservationError::NonFiniteMetric)
        );
    }
}
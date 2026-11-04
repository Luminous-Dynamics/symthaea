// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provenance-bound receipts for one measured hardware-energy run.
//!
//! A receipt binds the exact cumulative counter interval to source, workload,
//! configuration, device, driver, sensor scope, execution parameters and a
//! task-quality gate. The energy value is always the total measured energy in
//! that sensor's declared scope. A failed quality gate does not erase the
//! energy observation; it marks the run ineligible for matched-quality
//! efficiency comparisons.
//!
//! Idle subtraction, campaign-level averages and whole-system aggregation are
//! intentionally outside this per-run receipt. They require separate
//! observations and explicit coverage rules.

use serde::Serialize;

use super::energy_meter::{
    EnergyCounterReading, EnergyDelta, EnergyDeltaError, EnergyMeterDescriptor, EnergyScope,
};

/// Direction used when comparing an observed quality metric with its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EnergyQualityComparison {
    /// Larger values satisfy or improve the quality target.
    AtLeast,
    /// Smaller values satisfy or improve the quality target.
    AtMost,
}

/// Task-quality gate attached to one energy-measurement run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnergyQualityGate {
    /// Name of the independently measured task-quality metric.
    pub metric: String,
    /// Observed value for this exact workload run.
    pub observed_value: f64,
    /// Required minimum or maximum value.
    pub threshold: f64,
    /// Comparison direction.
    pub comparison: EnergyQualityComparison,
}

impl EnergyQualityGate {
    /// Whether the observed value satisfies the declared target.
    pub fn passed(&self) -> bool {
        if !self.observed_value.is_finite() || !self.threshold.is_finite() {
            return false;
        }
        match self.comparison {
            EnergyQualityComparison::AtLeast => self.observed_value >= self.threshold,
            EnergyQualityComparison::AtMost => self.observed_value <= self.threshold,
        }
    }
}

/// Provenance and execution context for one measured run.
///
/// Digests are supplied by the caller and must be 64 hexadecimal characters
/// (for example a BLAKE3 content digest). This type does not calculate or
/// authenticate repository/workload digests itself; it binds the supplied
/// identities to the energy observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EnergyRunContext {
    /// Full source commit identifier (40- or 64-character hexadecimal SHA).
    pub source_commit: String,
    /// Content digest of a dirty source tree, when the run was not clean.
    pub source_tree_digest: Option<String>,
    /// Stable human-readable workload identifier.
    pub workload_id: String,
    /// 256-bit content digest of the workload and frozen input set.
    pub workload_digest: String,
    /// 256-bit content digest of the effective model/runtime configuration.
    pub configuration_digest: String,
    /// Device identity matching the sensor descriptor.
    pub device_id: String,
    /// Device model as reported by the platform/backend.
    pub device_model: String,
    /// Driver version or a backend-specific stable driver identifier.
    pub driver_version: String,
    /// Runtime/backend version, when relevant and available.
    pub runtime_version: Option<String>,
    /// Expected physical scope of the meter.
    pub expected_scope: EnergyScope,
    /// Warm-up iterations before the measured interval.
    pub warmup_iterations: u32,
    /// Number of measured iterations in this run.
    pub measurement_iterations: u32,
    /// Number of input samples processed concurrently per iteration.
    pub batch_size: u32,
    /// Number of concurrent workers/streams used by this run.
    pub concurrency: u32,
    /// Number of task work units completed (for example inferences or tokens).
    pub measured_work_units: u64,
    /// Name of the work unit counted above (for example inference, token, update).
    pub work_unit_label: String,
}

/// Invalid or insufficient data for constructing a measured-run receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnergyReceiptError {
    /// Required metadata is empty.
    EmptyMetadata { field: &'static str },
    /// Source commit is not a 40- or 64-character hexadecimal identifier.
    InvalidSourceCommit,
    /// One of the content digests is not a 64-character hexadecimal value.
    InvalidContentDigest { field: &'static str },
    /// An execution count or work-unit count must be positive.
    InvalidExecutionCount { field: &'static str },
    /// Device identity differs from the measured sensor identity.
    DeviceMismatch,
    /// Expected sensor scope differs from the actual counter scope.
    ScopeMismatch,
    /// A quality metric or threshold is empty/non-finite.
    InvalidQualityGate,
    /// Counter readings cannot form a valid interval.
    InvalidCounterInterval { reason: EnergyDeltaError },
}

/// One provenance-bound measurement of total energy over a single run.
///
/// Fields are private so a valid receipt must be constructed through
/// [EnergyRunReceipt::new]. The receipt is serializable for durable evidence
/// storage; consumers must preserve the original scope and must not reinterpret
/// a device/package counter as system-wide energy.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EnergyRunReceipt {
    schema_version: u16,
    context: EnergyRunContext,
    quality_gate: EnergyQualityGate,
    quality_gate_passed: bool,
    start_reading: EnergyCounterReading,
    end_reading: EnergyCounterReading,
    energy_delta: EnergyDelta,
}

impl EnergyRunReceipt {
    /// Construct a receipt from compatible raw readings and declared context.
    pub fn new(
        context: EnergyRunContext,
        quality_gate: EnergyQualityGate,
        start_reading: EnergyCounterReading,
        end_reading: EnergyCounterReading,
    ) -> Result<Self, EnergyReceiptError> {
        validate_context(&context)?;
        validate_quality_gate(&quality_gate)?;

        if start_reading.descriptor.device_id != context.device_id
            || end_reading.descriptor.device_id != context.device_id
        {
            return Err(EnergyReceiptError::DeviceMismatch);
        }
        if start_reading.descriptor.scope != context.expected_scope
            || end_reading.descriptor.scope != context.expected_scope
        {
            return Err(EnergyReceiptError::ScopeMismatch);
        }

        let energy_delta = EnergyDelta::between(&start_reading, &end_reading)
            .map_err(|reason| EnergyReceiptError::InvalidCounterInterval { reason })?;

        Ok(Self {
            schema_version: 1,
            context,
            quality_gate_passed: quality_gate.passed(),
            quality_gate,
            start_reading,
            end_reading,
            energy_delta,
        })
    }

    /// Receipt schema version.
    pub fn schema_version(&self) -> u16 {
        self.schema_version
    }

    /// Source/workload/device/execution metadata.
    pub fn context(&self) -> &EnergyRunContext {
        &self.context
    }

    /// Task-quality measurement and target.
    pub fn quality_gate(&self) -> &EnergyQualityGate {
        &self.quality_gate
    }

    /// Whether the observed task quality satisfies the requested target.
    ///
    /// Energy remains recorded when this returns false, but the run should not
    /// be included in matched-quality energy-efficiency claims.
    pub fn quality_gate_passed(&self) -> bool {
        self.quality_gate_passed
    }

    /// Raw start counter reading.
    pub fn start_reading(&self) -> &EnergyCounterReading {
        &self.start_reading
    }

    /// Raw end counter reading.
    pub fn end_reading(&self) -> &EnergyCounterReading {
        &self.end_reading
    }

    /// Measured energy delta and sensor scope.
    pub fn energy_delta(&self) -> &EnergyDelta {
        &self.energy_delta
    }

    /// Total observed energy in joules for this sensor's declared scope.
    pub fn total_energy_joules(&self) -> f64 {
        self.energy_delta.energy_joules()
    }

    /// Total observed energy per completed task work unit, in joules.
    ///
    /// The unit is explicitly named by context.work_unit_label. This is not
    /// automatically joules per inference or per token unless that is the
    /// declared unit supplied by the measurement harness.
    pub fn joules_per_work_unit(&self) -> f64 {
        self.total_energy_joules() / self.context.measured_work_units as f64
    }

    /// Whether this receipt qualifies for a matched-quality efficiency report.
    pub fn eligible_for_matched_quality_comparison(&self) -> bool {
        self.quality_gate_passed
    }

    /// Return the sensor descriptor whose scope limits the interpretation.
    pub fn meter_descriptor(&self) -> &EnergyMeterDescriptor {
        self.energy_delta.descriptor()
    }
}

fn validate_context(context: &EnergyRunContext) -> Result<(), EnergyReceiptError> {
    if context.source_commit.is_empty() {
        return Err(EnergyReceiptError::EmptyMetadata {
            field: "source_commit",
        });
    }
    if !matches!(context.source_commit.len(), 40 | 64)
        || !is_hex(&context.source_commit)
    {
        return Err(EnergyReceiptError::InvalidSourceCommit);
    }
    if let Some(digest) = context.source_tree_digest.as_deref() {
        validate_digest(digest, "source_tree_digest")?;
    }
    validate_nonempty(&context.workload_id, "workload_id")?;
    validate_digest(&context.workload_digest, "workload_digest")?;
    validate_digest(&context.configuration_digest, "configuration_digest")?;
    validate_nonempty(&context.device_id, "device_id")?;
    validate_nonempty(&context.device_model, "device_model")?;
    validate_nonempty(&context.driver_version, "driver_version")?;
    validate_nonempty(&context.work_unit_label, "work_unit_label")?;

    if context.measurement_iterations == 0 {
        return Err(EnergyReceiptError::InvalidExecutionCount {
            field: "measurement_iterations",
        });
    }
    if context.batch_size == 0 {
        return Err(EnergyReceiptError::InvalidExecutionCount {
            field: "batch_size",
        });
    }
    if context.concurrency == 0 {
        return Err(EnergyReceiptError::InvalidExecutionCount {
            field: "concurrency",
        });
    }
    if context.measured_work_units == 0 {
        return Err(EnergyReceiptError::InvalidExecutionCount {
            field: "measured_work_units",
        });
    }
    Ok(())
}

fn validate_quality_gate(gate: &EnergyQualityGate) -> Result<(), EnergyReceiptError> {
    if gate.metric.trim().is_empty()
        || !gate.observed_value.is_finite()
        || !gate.threshold.is_finite()
    {
        return Err(EnergyReceiptError::InvalidQualityGate);
    }
    Ok(())
}

fn validate_nonempty(value: &str, field: &'static str) -> Result<(), EnergyReceiptError> {
    if value.trim().is_empty() {
        Err(EnergyReceiptError::EmptyMetadata { field })
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, field: &'static str) -> Result<(), EnergyReceiptError> {
    if value.len() != 64 || !is_hex(value) {
        Err(EnergyReceiptError::InvalidContentDigest { field })
    } else {
        Ok(())
    }
}

fn is_hex(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::energy_meter::{
        EnergyCounterUnit, EnergyMeterBackend, EnergyMeterDescriptor, EnergyScope,
    };

    fn context() -> EnergyRunContext {
        EnergyRunContext {
            source_commit: "a".repeat(40),
            source_tree_digest: None,
            workload_id: "jepa-forward-pass-v1".into(),
            workload_digest: "b".repeat(64),
            configuration_digest: "c".repeat(64),
            device_id: "GPU-TEST-UUID".into(),
            device_model: "Test GPU".into(),
            driver_version: "test-driver-1".into(),
            runtime_version: Some("test-runtime-1".into()),
            expected_scope: EnergyScope::GpuDevice,
            warmup_iterations: 10,
            measurement_iterations: 20,
            batch_size: 4,
            concurrency: 1,
            measured_work_units: 80,
            work_unit_label: "inference".into(),
        }
    }

    fn descriptor() -> EnergyMeterDescriptor {
        EnergyMeterDescriptor {
            backend: EnergyMeterBackend::Nvml,
            sensor_id: "nvml:GPU-TEST-UUID:total-energy".into(),
            device_id: "GPU-TEST-UUID".into(),
            scope: EnergyScope::GpuDevice,
            unit: EnergyCounterUnit::Millijoules,
            wrap_modulus: None,
        }
    }

    fn reading(raw_counter: u64, time_ns: u128) -> EnergyCounterReading {
        EnergyCounterReading {
            descriptor: descriptor(),
            counter_epoch: 1,
            raw_counter,
            monotonic_timestamp_ns: time_ns,
        }
    }

    fn quality_gate(observed_value: f64) -> EnergyQualityGate {
        EnergyQualityGate {
            metric: "prediction_accuracy".into(),
            observed_value,
            threshold: 0.95,
            comparison: EnergyQualityComparison::AtLeast,
        }
    }

    #[test]
    fn receipt_binds_energy_to_source_workload_device_and_quality() {
        let receipt = EnergyRunReceipt::new(
            context(),
            quality_gate(0.97),
            reading(1_000, 1_000),
            reading(1_100, 2_000_000_000),
        )
        .unwrap();

        assert_eq!(receipt.schema_version(), 1);
        assert!(receipt.quality_gate_passed());
        assert!(receipt.eligible_for_matched_quality_comparison());
        assert_eq!(receipt.total_energy_joules(), 0.1);
        assert!((receipt.joules_per_work_unit() - 0.00125).abs() < 1e-12);
        assert_eq!(receipt.meter_descriptor().scope, EnergyScope::GpuDevice);
        assert_eq!(receipt.context().measured_work_units, 80);
    }

    #[test]
    fn failed_quality_gate_preserves_energy_but_disqualifies_comparison() {
        let receipt = EnergyRunReceipt::new(
            context(),
            quality_gate(0.90),
            reading(1_000, 1_000),
            reading(1_100, 2_000_000_000),
        )
        .unwrap();

        assert!(!receipt.quality_gate_passed());
        assert!(!receipt.eligible_for_matched_quality_comparison());
        assert_eq!(receipt.total_energy_joules(), 0.1);
    }

    #[test]
    fn wrong_device_or_scope_is_rejected() {
        let mut wrong_device = reading(1_000, 1_000);
        wrong_device.descriptor.device_id = "GPU-OTHER".into();
        assert_eq!(
            EnergyRunReceipt::new(
                context(),
                quality_gate(0.97),
                wrong_device,
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::DeviceMismatch)
        );

        let mut wrong_scope = reading(1_000, 1_000);
        wrong_scope.descriptor.scope = EnergyScope::SystemWall;
        assert_eq!(
            EnergyRunReceipt::new(
                context(),
                quality_gate(0.97),
                wrong_scope,
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::ScopeMismatch)
        );
    }

    #[test]
    fn malformed_provenance_digests_are_rejected() {
        let mut bad_commit = context();
        bad_commit.source_commit = "not-a-commit".into();
        assert_eq!(
            EnergyRunReceipt::new(
                bad_commit,
                quality_gate(0.97),
                reading(1_000, 1_000),
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::InvalidSourceCommit)
        );

        let mut bad_workload = context();
        bad_workload.workload_digest = "short".into();
        assert_eq!(
            EnergyRunReceipt::new(
                bad_workload,
                quality_gate(0.97),
                reading(1_000, 1_000),
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::InvalidContentDigest {
                field: "workload_digest"
            })
        );
    }

    #[test]
    fn zero_work_units_and_non_finite_quality_are_rejected() {
        let mut no_work = context();
        no_work.measured_work_units = 0;
        assert_eq!(
            EnergyRunReceipt::new(
                no_work,
                quality_gate(0.97),
                reading(1_000, 1_000),
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::InvalidExecutionCount {
                field: "measured_work_units"
            })
        );

        assert_eq!(
            EnergyRunReceipt::new(
                context(),
                quality_gate(f64::NAN),
                reading(1_000, 1_000),
                reading(1_100, 2_000_000_000)
            ),
            Err(EnergyReceiptError::InvalidQualityGate)
        );
    }

    #[test]
    fn driver_epoch_mismatch_is_not_a_measured_interval() {
        let mut end = reading(1_100, 2_000_000_000);
        end.counter_epoch = 2;
        assert_eq!(
            EnergyRunReceipt::new(
                context(),
                quality_gate(0.97),
                reading(1_000, 1_000),
                end
            ),
            Err(EnergyReceiptError::InvalidCounterInterval {
                reason: EnergyDeltaError::CounterEpochChanged
            })
        );
    }
}

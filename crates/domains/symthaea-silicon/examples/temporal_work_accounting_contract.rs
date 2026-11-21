// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned temporal-work accounting contract for continuous-time benchmarks.
//!
//! This example defines a measurement schema, not a neuron or benchmark result.
//! Its smoke receipt deliberately contains no measured wall time or energy.
//!
//! From the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_work_accounting_contract
//!   cargo run --locked -p symthaea-silicon --example temporal_work_accounting_contract
//!
//! Logical time and wall-clock time are separate quantities. Operation counts
//! are reported by class; they are not silently converted to latency or joules.

use serde::{Deserialize, Serialize};

const SCHEMA: &str = "symthaea.temporal-work-ledger";
const SCHEMA_VERSION: &str = "0.1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ReceiptKind {
    ContractSmokeFixture,
    BenchmarkObservation,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OperationCounts {
    scalar_add_sub: u64,
    scalar_multiply: u64,
    scalar_divide: u64,
    scalar_exponential: u64,
    scalar_comparison: u64,
    vector_lanes_updated: u64,
}

impl OperationCounts {
    /// Sum primitive scalar operations without assigning equal hardware cost.
    /// Vector lanes updated remain a separate metric and are not added here.
    fn primitive_scalar_total(self) -> Option<u64> {
        [
            self.scalar_add_sub,
            self.scalar_multiply,
            self.scalar_divide,
            self.scalar_exponential,
            self.scalar_comparison,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalWork {
    input_events: u64,
    accepted_observations: u64,
    fast_state_updates: u64,
    slow_context_refreshes: u64,
    interpolation_evaluations: u64,
    solver_substeps: u64,
    gate_evaluations: u64,
    leak_updates: u64,
    output_decisions: u64,
    independently_verified_correct_decisions: u64,
    state_bytes_read: u64,
    state_bytes_written: u64,
    parameter_bytes_read: u64,
    input_bytes_read: u64,
    peak_scratch_bytes: u64,
    operations: OperationCounts,
}

impl TemporalWork {
    fn state_update_total(&self) -> Option<u64> {
        self.fast_state_updates
            .checked_add(self.slow_context_refreshes)
    }

    fn state_bytes_moved(&self) -> Option<u64> {
        self.state_bytes_read.checked_add(self.state_bytes_written)
    }

    fn total_bytes_touched(&self) -> Option<u64> {
        [
            self.state_bytes_read,
            self.state_bytes_written,
            self.parameter_bytes_read,
            self.input_bytes_read,
        ]
        .into_iter()
        .try_fold(0_u64, u64::checked_add)
    }

    fn primitive_scalar_operations(&self) -> Option<u64> {
        self.operations.primitive_scalar_total()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LatencyQuantiles {
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DirectEnergyMeasurement {
    joules: f64,
    method: String,
    system_boundary: String,
    meter_or_device_id: String,
    same_run_as_performance: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SystemObservation {
    /// None means not measured. It never means zero.
    wall_elapsed_ns: Option<u64>,
    event_to_decision_latency: Option<LatencyQuantiles>,
    deadline_count: u64,
    deadline_misses: u64,
    peak_queue_depth: Option<u64>,
    /// This field accepts direct measurement evidence only, never estimates.
    direct_energy: Option<DirectEnergyMeasurement>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QualityObservation {
    task_metric_name: Option<String>,
    task_metric_value: Option<f64>,
    trajectory_max_abs_error: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TemporalWorkReceipt {
    schema: String,
    schema_version: String,
    kind: ReceiptKind,
    profile_id: String,
    workload_id: String,
    source_revision: String,
    configuration_id: String,
    seed: u64,
    /// Elapsed time represented by the workload, in nanoseconds.
    logical_duration_ns: u64,
    work: TemporalWork,
    quality: QualityObservation,
    system: SystemObservation,
}

impl TemporalWorkReceipt {
    fn validate(&self) -> Result<(), &'static str> {
        if self.schema != SCHEMA || self.schema_version != SCHEMA_VERSION {
            return Err("unsupported schema identity or version");
        }
        for (name, value) in [
            ("profile_id", self.profile_id.as_str()),
            ("workload_id", self.workload_id.as_str()),
            ("source_revision", self.source_revision.as_str()),
            ("configuration_id", self.configuration_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(match name {
                    "profile_id" => "profile identity is empty",
                    "workload_id" => "workload identity is empty",
                    "source_revision" => "source revision identity is empty",
                    _ => "configuration identity is empty",
                });
            }
        }
        if self.logical_duration_ns == 0 {
            return Err("logical duration must be positive");
        }
        if self
            .system
            .wall_elapsed_ns
            .is_some_and(|duration| duration == 0)
        {
            return Err("measured wall duration must be positive");
        }
        if self.work.accepted_observations > self.work.input_events {
            return Err("accepted observations cannot exceed supplied input events");
        }
        if self.work.independently_verified_correct_decisions > self.work.output_decisions {
            return Err("verified correct decisions cannot exceed output decisions");
        }
        if self.system.deadline_misses > self.system.deadline_count {
            return Err("deadline misses cannot exceed declared deadlines");
        }
        if let Some(latency) = self.system.event_to_decision_latency {
            if latency.p50_ns > latency.p95_ns || latency.p95_ns > latency.p99_ns {
                return Err("latency quantiles must be nondecreasing");
            }
        }
        if let Some(value) = self.quality.task_metric_value {
            if !value.is_finite() {
                return Err("task metric must be finite");
            }
        }
        if let Some(error) = self.quality.trajectory_max_abs_error {
            if !error.is_finite() || error < 0.0 {
                return Err("trajectory error must be finite and nonnegative");
            }
        }
        if let Some(energy) = &self.system.direct_energy {
            if self.system.wall_elapsed_ns.is_none() {
                return Err("direct energy requires same-run wall-clock performance observation");
            }
            if !energy.joules.is_finite() || energy.joules < 0.0 {
                return Err("direct energy must be finite and nonnegative");
            }
            if energy.method.trim().is_empty()
                || energy.system_boundary.trim().is_empty()
                || energy.meter_or_device_id.trim().is_empty()
            {
                return Err("direct energy requires method, boundary and meter/device identity");
            }
            if !energy.same_run_as_performance {
                return Err("energy and performance must come from the same run");
            }
        }

        // Reject overflow rather than silently wrapping or saturating evidence.
        self.work
            .state_update_total()
            .ok_or("state update counter overflow")?;
        self.work
            .state_bytes_moved()
            .ok_or("state byte counter overflow")?;
        self.work
            .total_bytes_touched()
            .ok_or("total byte counter overflow")?;
        self.work
            .primitive_scalar_operations()
            .ok_or("primitive operation counter overflow")?;

        Ok(())
    }

    fn logical_seconds(&self) -> Option<f64> {
        (self.logical_duration_ns > 0)
            .then_some(self.logical_duration_ns as f64 / 1.0e9)
    }

    /// Number of fast updates plus slow-context refreshes per logical second.
    fn state_updates_per_logical_second(&self) -> Option<f64> {
        let updates = self.work.state_update_total()?;
        Some(updates as f64 / self.logical_seconds()?)
    }

    /// Sum of primitive scalar operation counts per logical second.
    /// This is not a latency, energy or hardware-cost estimate.
    fn scalar_operations_per_logical_second(&self) -> Option<f64> {
        let operations = self.work.primitive_scalar_operations()?;
        Some(operations as f64 / self.logical_seconds()?)
    }

    /// Logical seconds progressed per measured wall-clock second.
    /// None is returned when wall time was not observed.
    fn real_time_factor(&self) -> Option<f64> {
        let wall_ns = self.system.wall_elapsed_ns.filter(|value| *value > 0)?;
        self.logical_seconds()?;
        Some(self.logical_duration_ns as f64 / wall_ns as f64)
    }

    /// Declared state, parameter and input bytes touched per logical second.
    /// This is accounting, not a cache- or DRAM-traffic measurement.
    fn bytes_touched_per_logical_second(&self) -> Option<f64> {
        let bytes = self.work.total_bytes_touched()?;
        Some(bytes as f64 / self.logical_seconds()?)
    }

    /// Primitive scalar operation count per independently verified correct decision.
    /// Undefined denominators remain None; they are never reported as zero.
    fn scalar_operations_per_correct_decision(&self) -> Option<f64> {
        let operations = self.work.primitive_scalar_operations()?;
        let correct = self.work.independently_verified_correct_decisions;
        (correct > 0).then_some(operations as f64 / correct as f64)
    }

    fn events_per_logical_second(&self) -> Option<f64> {
        Some(self.work.input_events as f64 / self.logical_seconds()?)
    }
}

fn smoke_fixture() -> TemporalWorkReceipt {
    TemporalWorkReceipt {
        schema: SCHEMA.to_owned(),
        schema_version: SCHEMA_VERSION.to_owned(),
        kind: ReceiptKind::ContractSmokeFixture,
        profile_id: "schema-only".to_owned(),
        workload_id: "no-model-no-performance-claim".to_owned(),
        source_revision: "synthetic-contract-fixture-v0.1".to_owned(),
        configuration_id: "no-model-configuration".to_owned(),
        seed: 0,
        logical_duration_ns: 1_000_000_000,
        work: TemporalWork::default(),
        quality: QualityObservation::default(),
        system: SystemObservation::default(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let receipt = smoke_fixture();
    receipt
        .validate()
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    println!("derived real_time_factor: {:?}", receipt.real_time_factor());
    println!(
        "derived scalar_ops_per_correct_decision: {:?}",
        receipt.scalar_operations_per_correct_decision()
    );
    println!("scope: schema smoke fixture only; no model, timing or energy result");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation() -> TemporalWorkReceipt {
        let mut receipt = smoke_fixture();
        receipt.kind = ReceiptKind::BenchmarkObservation;
        receipt.work.input_events = 100;
        receipt.work.accepted_observations = 96;
        receipt.work.fast_state_updates = 200;
        receipt.work.slow_context_refreshes = 20;
        receipt.work.output_decisions = 10;
        receipt.work.independently_verified_correct_decisions = 8;
        receipt.work.operations = OperationCounts {
            scalar_add_sub: 100,
            scalar_multiply: 50,
            scalar_divide: 10,
            scalar_exponential: 5,
            scalar_comparison: 35,
            vector_lanes_updated: 4096,
        };
        receipt
    }

    #[test]
    fn logical_and_wall_time_are_independent() {
        let mut receipt = observation();
        receipt.logical_duration_ns = 10_000_000_000;
        receipt.system.wall_elapsed_ns = Some(2_000_000_000);
        receipt.validate().unwrap();

        assert_eq!(receipt.real_time_factor(), Some(5.0));
        assert_eq!(receipt.state_updates_per_logical_second(), Some(22.0));
    }

    #[test]
    fn zero_logical_duration_has_no_derived_rates() {
        let mut receipt = observation();
        receipt.logical_duration_ns = 0;
        assert_eq!(receipt.state_updates_per_logical_second(), None);
        assert_eq!(receipt.scalar_operations_per_logical_second(), None);
        assert_eq!(receipt.bytes_touched_per_logical_second(), None);
        assert_eq!(receipt.events_per_logical_second(), None);
    }

    #[test]
    fn missing_wall_time_is_unknown_not_zero() {
        let receipt = observation();
        assert_eq!(receipt.real_time_factor(), None);
    }

    #[test]
    fn missing_energy_is_unknown_not_zero() {
        let receipt = observation();
        assert!(receipt.system.direct_energy.is_none());
        receipt.validate().unwrap();
    }

    #[test]
    fn bytes_touched_rate_uses_the_logical_clock_and_checked_totals() {
        let mut receipt = observation();
        receipt.work.state_bytes_read = 100;
        receipt.work.state_bytes_written = 200;
        receipt.work.parameter_bytes_read = 300;
        receipt.work.input_bytes_read = 400;
        assert_eq!(receipt.work.total_bytes_touched(), Some(1000));
        assert_eq!(receipt.bytes_touched_per_logical_second(), Some(1000.0));
        assert_eq!(receipt.events_per_logical_second(), Some(100.0));
    }

    #[test]
    fn undefined_correct_decision_denominator_is_none() {
        let mut receipt = observation();
        receipt.work.independently_verified_correct_decisions = 0;
        assert_eq!(receipt.scalar_operations_per_correct_decision(), None);
    }

    #[test]
    fn primitive_operations_remain_separate_from_vector_lanes() {
        let receipt = observation();
        assert_eq!(receipt.work.primitive_scalar_operations(), Some(200));
        assert_eq!(receipt.work.operations.vector_lanes_updated, 4096);
        assert_eq!(
            receipt.scalar_operations_per_logical_second(),
            Some(200.0)
        );
    }

    #[test]
    fn rejects_impossible_observation_and_decision_counts() {
        let mut receipt = observation();
        receipt.work.accepted_observations = 101;
        assert_eq!(
            receipt.validate(),
            Err("accepted observations cannot exceed supplied input events")
        );

        let mut receipt = observation();
        receipt.work.independently_verified_correct_decisions = 11;
        assert_eq!(
            receipt.validate(),
            Err("verified correct decisions cannot exceed output decisions")
        );
    }

    #[test]
    fn rejects_unordered_latency_quantiles_and_deadline_overcount() {
        let mut receipt = observation();
        receipt.system.event_to_decision_latency = Some(LatencyQuantiles {
            p50_ns: 20,
            p95_ns: 10,
            p99_ns: 30,
        });
        assert_eq!(
            receipt.validate(),
            Err("latency quantiles must be nondecreasing")
        );

        let mut receipt = observation();
        receipt.system.deadline_count = 4;
        receipt.system.deadline_misses = 5;
        assert_eq!(
            receipt.validate(),
            Err("deadline misses cannot exceed declared deadlines")
        );
    }

    #[test]
    fn direct_energy_requires_boundary_identity_and_same_run() {
        let mut receipt = observation();
        receipt.system.wall_elapsed_ns = Some(1_000_000_000);
        receipt.system.direct_energy = Some(DirectEnergyMeasurement {
            joules: 0.25,
            method: "external power meter".to_owned(),
            system_boundary: "host plus accelerator and memory".to_owned(),
            meter_or_device_id: "meter-A".to_owned(),
            same_run_as_performance: false,
        });
        assert_eq!(
            receipt.validate(),
            Err("energy and performance must come from the same run")
        );

        receipt.system.direct_energy.as_mut().unwrap().same_run_as_performance = true;
        receipt.validate().unwrap();
    }

    #[test]
    fn direct_energy_requires_a_wall_clock_performance_observation() {
        let mut receipt = observation();
        receipt.system.direct_energy = Some(DirectEnergyMeasurement {
            joules: 0.25,
            method: "external power meter".to_owned(),
            system_boundary: "host plus accelerator and memory".to_owned(),
            meter_or_device_id: "meter-A".to_owned(),
            same_run_as_performance: true,
        });
        assert_eq!(
            receipt.validate(),
            Err("direct energy requires same-run wall-clock performance observation")
        );
    }

    #[test]
    fn rejects_non_finite_energy_and_negative_trajectory_error() {
        let mut receipt = observation();
        receipt.system.wall_elapsed_ns = Some(1_000_000_000);
        receipt.system.direct_energy = Some(DirectEnergyMeasurement {
            joules: f64::NAN,
            method: "meter".to_owned(),
            system_boundary: "complete system".to_owned(),
            meter_or_device_id: "meter-A".to_owned(),
            same_run_as_performance: true,
        });
        assert_eq!(
            receipt.validate(),
            Err("direct energy must be finite and nonnegative")
        );

        let mut receipt = observation();
        receipt.quality.trajectory_max_abs_error = Some(-0.1);
        assert_eq!(
            receipt.validate(),
            Err("trajectory error must be finite and nonnegative")
        );
    }

    #[test]
    fn rejects_counter_overflow_instead_of_saturating() {
        let mut receipt = observation();
        receipt.work.fast_state_updates = u64::MAX;
        receipt.work.slow_context_refreshes = 1;
        assert_eq!(
            receipt.validate(),
            Err("state update counter overflow")
        );

        let mut receipt = observation();
        receipt.work.operations.scalar_add_sub = u64::MAX;
        receipt.work.operations.scalar_multiply = 1;
        assert_eq!(
            receipt.validate(),
            Err("primitive operation counter overflow")
        );
    }

    #[test]
    fn serialization_is_deterministic_for_a_frozen_receipt() {
        let receipt = observation();
        let first = serde_json::to_string(&receipt).unwrap();
        let second = serde_json::to_string(&receipt).unwrap();
        assert_eq!(first, second);
        assert!(first.contains("\"logical_duration_ns\":1000000000"));
        assert!(first.contains("\"wall_elapsed_ns\":null"));
        assert!(first.contains("\"direct_energy\":null"));
    }

    #[test]
    fn smoke_receipt_has_no_fabricated_performance_metrics() {
        let receipt = smoke_fixture();
        receipt.validate().unwrap();
        assert_eq!(receipt.real_time_factor(), None);
        assert_eq!(receipt.scalar_operations_per_correct_decision(), None);
        assert!(receipt.system.direct_energy.is_none());
    }
}

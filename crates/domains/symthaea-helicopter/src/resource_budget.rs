// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime resource-budget evidence for flight-control deployments.
//!
//! Timing deadlines do not capture memory exhaustion, queue growth, or evidence
//! backpressure. This monitor consumes externally measured resource watermarks
//! and turns them into bounded, replayable qualification evidence. It performs
//! no platform-specific allocation introspection by itself.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceBudgetConfig {
    pub maximum_resident_memory_bytes: u64,
    pub maximum_heap_live_bytes: u64,
    pub maximum_stack_high_water_bytes: u64,
    pub maximum_sensor_queue_depth: usize,
    pub maximum_command_queue_depth: usize,
    pub maximum_evidence_buffer_bytes: u64,
    pub maximum_open_descriptors: u32,
    pub caution_fraction: f64,
    pub maximum_consecutive_caution: u32,
}

impl Default for ResourceBudgetConfig {
    fn default() -> Self {
        Self {
            maximum_resident_memory_bytes: 512 * 1024 * 1024,
            maximum_heap_live_bytes: 256 * 1024 * 1024,
            maximum_stack_high_water_bytes: 8 * 1024 * 1024,
            maximum_sensor_queue_depth: 1_024,
            maximum_command_queue_depth: 128,
            maximum_evidence_buffer_bytes: 64 * 1024 * 1024,
            maximum_open_descriptors: 256,
            caution_fraction: 0.80,
            maximum_consecutive_caution: 10,
        }
    }
}

impl ResourceBudgetConfig {
    pub fn validate(&self) -> Result<(), ResourceBudgetError> {
        if self.maximum_resident_memory_bytes == 0
            || self.maximum_heap_live_bytes == 0
            || self.maximum_stack_high_water_bytes == 0
            || self.maximum_sensor_queue_depth == 0
            || self.maximum_command_queue_depth == 0
            || self.maximum_evidence_buffer_bytes == 0
            || self.maximum_open_descriptors == 0
            || !self.caution_fraction.is_finite()
            || !(0.0..1.0).contains(&self.caution_fraction)
            || self.maximum_consecutive_caution == 0
        {
            return Err(ResourceBudgetError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResourceBudgetObservation {
    pub monotonic_time_s: f64,
    pub resident_memory_bytes: u64,
    pub heap_live_bytes: u64,
    pub stack_high_water_bytes: u64,
    pub sensor_queue_depth: usize,
    pub command_queue_depth: usize,
    pub evidence_buffer_bytes: u64,
    pub open_descriptors: u32,
    pub dropped_telemetry_records: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceBudgetState {
    Healthy,
    Caution,
    Exhausted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceBudgetViolation {
    ResidentMemory,
    HeapLive,
    StackHighWater,
    SensorQueue,
    CommandQueue,
    EvidenceBuffer,
    OpenDescriptors,
    TelemetryDropped,
    PersistentCaution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceBudgetEvidence {
    pub state: ResourceBudgetState,
    pub samples: u64,
    pub violations: Vec<ResourceBudgetViolation>,
    pub consecutive_caution: u32,
    pub peak_resident_memory_bytes: u64,
    pub peak_heap_live_bytes: u64,
    pub peak_stack_high_water_bytes: u64,
    pub peak_sensor_queue_depth: usize,
    pub peak_command_queue_depth: usize,
    pub peak_evidence_buffer_bytes: u64,
    pub peak_open_descriptors: u32,
    pub total_dropped_telemetry_records: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceBudgetError {
    InvalidConfiguration,
    NonFiniteTime,
    TimeWentBackwards,
}

#[derive(Debug, Clone)]
pub struct ResourceBudgetMonitor {
    config: ResourceBudgetConfig,
    last_time_s: Option<f64>,
    evidence: ResourceBudgetEvidence,
}

impl ResourceBudgetMonitor {
    pub fn new(config: ResourceBudgetConfig) -> Result<Self, ResourceBudgetError> {
        config.validate()?;
        Ok(Self {
            config,
            last_time_s: None,
            evidence: ResourceBudgetEvidence {
                state: ResourceBudgetState::Healthy,
                samples: 0,
                violations: Vec::new(),
                consecutive_caution: 0,
                peak_resident_memory_bytes: 0,
                peak_heap_live_bytes: 0,
                peak_stack_high_water_bytes: 0,
                peak_sensor_queue_depth: 0,
                peak_command_queue_depth: 0,
                peak_evidence_buffer_bytes: 0,
                peak_open_descriptors: 0,
                total_dropped_telemetry_records: 0,
            },
        })
    }

    pub fn evidence(&self) -> &ResourceBudgetEvidence {
        &self.evidence
    }

    pub fn observe(
        &mut self,
        observation: ResourceBudgetObservation,
    ) -> Result<&ResourceBudgetEvidence, ResourceBudgetError> {
        self.config.validate()?;
        if !observation.monotonic_time_s.is_finite() || observation.monotonic_time_s < 0.0 {
            return Err(ResourceBudgetError::NonFiniteTime);
        }
        if self
            .last_time_s
            .is_some_and(|last| observation.monotonic_time_s < last)
        {
            return Err(ResourceBudgetError::TimeWentBackwards);
        }
        self.last_time_s = Some(observation.monotonic_time_s);
        self.evidence.samples = self.evidence.samples.saturating_add(1);
        self.evidence.peak_resident_memory_bytes = self
            .evidence
            .peak_resident_memory_bytes
            .max(observation.resident_memory_bytes);
        self.evidence.peak_heap_live_bytes = self
            .evidence
            .peak_heap_live_bytes
            .max(observation.heap_live_bytes);
        self.evidence.peak_stack_high_water_bytes = self
            .evidence
            .peak_stack_high_water_bytes
            .max(observation.stack_high_water_bytes);
        self.evidence.peak_sensor_queue_depth = self
            .evidence
            .peak_sensor_queue_depth
            .max(observation.sensor_queue_depth);
        self.evidence.peak_command_queue_depth = self
            .evidence
            .peak_command_queue_depth
            .max(observation.command_queue_depth);
        self.evidence.peak_evidence_buffer_bytes = self
            .evidence
            .peak_evidence_buffer_bytes
            .max(observation.evidence_buffer_bytes);
        self.evidence.peak_open_descriptors = self
            .evidence
            .peak_open_descriptors
            .max(observation.open_descriptors);
        self.evidence.total_dropped_telemetry_records = self
            .evidence
            .total_dropped_telemetry_records
            .saturating_add(observation.dropped_telemetry_records);

        let mut violations = Vec::new();
        if exceeds(
            observation.resident_memory_bytes,
            self.config.maximum_resident_memory_bytes,
        ) {
            violations.push(ResourceBudgetViolation::ResidentMemory);
        }
        if exceeds(
            observation.heap_live_bytes,
            self.config.maximum_heap_live_bytes,
        ) {
            violations.push(ResourceBudgetViolation::HeapLive);
        }
        if exceeds(
            observation.stack_high_water_bytes,
            self.config.maximum_stack_high_water_bytes,
        ) {
            violations.push(ResourceBudgetViolation::StackHighWater);
        }
        if observation.sensor_queue_depth > self.config.maximum_sensor_queue_depth {
            violations.push(ResourceBudgetViolation::SensorQueue);
        }
        if observation.command_queue_depth > self.config.maximum_command_queue_depth {
            violations.push(ResourceBudgetViolation::CommandQueue);
        }
        if exceeds(
            observation.evidence_buffer_bytes,
            self.config.maximum_evidence_buffer_bytes,
        ) {
            violations.push(ResourceBudgetViolation::EvidenceBuffer);
        }
        if observation.open_descriptors > self.config.maximum_open_descriptors {
            violations.push(ResourceBudgetViolation::OpenDescriptors);
        }
        if observation.dropped_telemetry_records > 0 {
            violations.push(ResourceBudgetViolation::TelemetryDropped);
        }

        let caution = ratio(
            observation.resident_memory_bytes,
            self.config.maximum_resident_memory_bytes,
        ) >= self.config.caution_fraction
            || ratio(
                observation.heap_live_bytes,
                self.config.maximum_heap_live_bytes,
            ) >= self.config.caution_fraction
            || ratio(
                observation.stack_high_water_bytes,
                self.config.maximum_stack_high_water_bytes,
            ) >= self.config.caution_fraction
            || observation.sensor_queue_depth as f64
                / self.config.maximum_sensor_queue_depth as f64
                >= self.config.caution_fraction
            || observation.command_queue_depth as f64
                / self.config.maximum_command_queue_depth as f64
                >= self.config.caution_fraction
            || ratio(
                observation.evidence_buffer_bytes,
                self.config.maximum_evidence_buffer_bytes,
            ) >= self.config.caution_fraction
            || observation.open_descriptors as f64 / self.config.maximum_open_descriptors as f64
                >= self.config.caution_fraction;
        if violations.is_empty() && caution {
            self.evidence.consecutive_caution = self.evidence.consecutive_caution.saturating_add(1);
        } else if !caution {
            self.evidence.consecutive_caution = 0;
        }
        if self.evidence.consecutive_caution >= self.config.maximum_consecutive_caution {
            violations.push(ResourceBudgetViolation::PersistentCaution);
        }
        self.evidence.state = if !violations.is_empty() {
            ResourceBudgetState::Exhausted
        } else if caution {
            ResourceBudgetState::Caution
        } else {
            ResourceBudgetState::Healthy
        };
        self.evidence.violations = violations;
        Ok(&self.evidence)
    }
}

fn exceeds(value: u64, maximum: u64) -> bool {
    value > maximum
}

fn ratio(value: u64, maximum: u64) -> f64 {
    value as f64 / maximum as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(time_s: f64) -> ResourceBudgetObservation {
        ResourceBudgetObservation {
            monotonic_time_s: time_s,
            resident_memory_bytes: 64 * 1024 * 1024,
            heap_live_bytes: 32 * 1024 * 1024,
            stack_high_water_bytes: 1024 * 1024,
            sensor_queue_depth: 10,
            command_queue_depth: 4,
            evidence_buffer_bytes: 1024,
            open_descriptors: 10,
            dropped_telemetry_records: 0,
        }
    }

    #[test]
    fn nominal_resources_are_healthy() {
        let mut monitor = ResourceBudgetMonitor::new(ResourceBudgetConfig::default()).unwrap();
        assert_eq!(
            monitor.observe(observation(0.0)).unwrap().state,
            ResourceBudgetState::Healthy
        );
    }

    #[test]
    fn heap_overrun_is_exhausted() {
        let config = ResourceBudgetConfig::default();
        let mut monitor = ResourceBudgetMonitor::new(config).unwrap();
        let mut sample = observation(0.0);
        sample.heap_live_bytes = config.maximum_heap_live_bytes + 1;
        let evidence = monitor.observe(sample).unwrap();
        assert_eq!(evidence.state, ResourceBudgetState::Exhausted);
        assert!(
            evidence
                .violations
                .contains(&ResourceBudgetViolation::HeapLive)
        );
    }

    #[test]
    fn persistent_caution_becomes_violation() {
        let mut config = ResourceBudgetConfig::default();
        config.maximum_consecutive_caution = 2;
        let mut monitor = ResourceBudgetMonitor::new(config).unwrap();
        let mut sample = observation(0.0);
        sample.heap_live_bytes = (config.maximum_heap_live_bytes as f64 * 0.9) as u64;
        monitor.observe(sample).unwrap();
        sample.monotonic_time_s = 1.0;
        let evidence = monitor.observe(sample).unwrap();
        assert!(
            evidence
                .violations
                .contains(&ResourceBudgetViolation::PersistentCaution)
        );
    }

    #[test]
    fn time_reversal_is_rejected() {
        let mut monitor = ResourceBudgetMonitor::new(ResourceBudgetConfig::default()).unwrap();
        monitor.observe(observation(2.0)).unwrap();
        assert_eq!(
            monitor.observe(observation(1.0)),
            Err(ResourceBudgetError::TimeWentBackwards)
        );
    }
}

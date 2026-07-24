// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Aggregate estimator-health assurance across timing, buffering, and fusion.
//!
//! Individual sensor and navigation components expose local evidence, but flight
//! authority needs one conservative answer about whether the complete estimate
//! chain is healthy. This module combines clock discipline, multi-rate bus, and
//! innovation-consistency evidence without allowing a healthy subcomponent to
//! mask a faulted dependency.

use serde::{Deserialize, Serialize};

use crate::navigation_consistency::{NavigationConsistencyEvidence, NavigationConsistencyState};
use crate::sensor_bus::SensorBusEvidence;
use crate::timebase::{ClockDisciplineEvidence, ClockLockState};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EstimatorHealthConfig {
    pub maximum_bus_rejection_fraction: f64,
    pub maximum_incomplete_snapshot_fraction: f64,
    pub maximum_navigation_rejection_fraction: f64,
    pub maximum_consecutive_inconsistent: u32,
    pub degraded_grace_updates: u32,
    pub recovery_updates: u32,
}

impl Default for EstimatorHealthConfig {
    fn default() -> Self {
        Self {
            maximum_bus_rejection_fraction: 0.05,
            maximum_incomplete_snapshot_fraction: 0.10,
            maximum_navigation_rejection_fraction: 0.25,
            maximum_consecutive_inconsistent: 3,
            degraded_grace_updates: 2,
            recovery_updates: 3,
        }
    }
}

impl EstimatorHealthConfig {
    pub fn validate(&self) -> Result<(), EstimatorHealthError> {
        for value in [
            self.maximum_bus_rejection_fraction,
            self.maximum_incomplete_snapshot_fraction,
            self.maximum_navigation_rejection_fraction,
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(EstimatorHealthError::InvalidConfiguration);
            }
        }
        if self.maximum_consecutive_inconsistent == 0
            || self.degraded_grace_updates == 0
            || self.recovery_updates == 0
        {
            return Err(EstimatorHealthError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatorHealthState {
    Initializing,
    Healthy,
    Degraded,
    Unavailable,
    Faulted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatorHealthReason {
    ClockNotLocked,
    ClockFaulted,
    ExcessiveBusRejections,
    ExcessiveIncompleteSnapshots,
    NavigationSuspect,
    NavigationUnreliable,
    ExcessiveNavigationRejections,
    InnovationStreakExceeded,
    NoSensorMeasurements,
    NoNavigationSamples,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EstimatorHealthAssessment {
    pub state: EstimatorHealthState,
    pub reasons: Vec<EstimatorHealthReason>,
    pub authority_usable: bool,
    pub update_count: u64,
    pub consecutive_degraded: u32,
    pub consecutive_healthy: u32,
    pub bus_rejection_fraction: f64,
    pub incomplete_snapshot_fraction: f64,
    pub navigation_rejection_fraction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorHealthError {
    InvalidConfiguration,
}

#[derive(Debug, Clone)]
pub struct EstimatorHealthManager {
    config: EstimatorHealthConfig,
    update_count: u64,
    consecutive_degraded: u32,
    consecutive_healthy: u32,
    state: EstimatorHealthState,
}

impl EstimatorHealthManager {
    pub fn new(config: EstimatorHealthConfig) -> Result<Self, EstimatorHealthError> {
        config.validate()?;
        Ok(Self {
            config,
            update_count: 0,
            consecutive_degraded: 0,
            consecutive_healthy: 0,
            state: EstimatorHealthState::Initializing,
        })
    }

    pub fn assess(
        &mut self,
        clock: ClockDisciplineEvidence,
        bus: SensorBusEvidence,
        navigation: NavigationConsistencyEvidence,
    ) -> Result<EstimatorHealthAssessment, EstimatorHealthError> {
        self.config.validate()?;
        let bus_total = bus
            .accepted_measurements
            .saturating_add(bus.rejected_measurements);
        let bus_rejection_fraction = fraction(bus.rejected_measurements, bus_total);
        let incomplete_snapshot_fraction = fraction(bus.incomplete_snapshots, bus.snapshots_built);
        let navigation_rejection_fraction =
            fraction(navigation.rejected_samples, navigation.total_samples);

        let mut reasons = Vec::new();
        match clock.lock_state {
            ClockLockState::Uninitialized | ClockLockState::Synchronizing => {
                reasons.push(EstimatorHealthReason::ClockNotLocked)
            }
            ClockLockState::Locked => {}
            ClockLockState::Faulted => reasons.push(EstimatorHealthReason::ClockFaulted),
        }
        if bus_total == 0 {
            reasons.push(EstimatorHealthReason::NoSensorMeasurements);
        }
        if navigation.total_samples == 0 {
            reasons.push(EstimatorHealthReason::NoNavigationSamples);
        }
        if bus_rejection_fraction > self.config.maximum_bus_rejection_fraction {
            reasons.push(EstimatorHealthReason::ExcessiveBusRejections);
        }
        if bus.snapshots_built > 0
            && incomplete_snapshot_fraction > self.config.maximum_incomplete_snapshot_fraction
        {
            reasons.push(EstimatorHealthReason::ExcessiveIncompleteSnapshots);
        }
        match navigation.state {
            NavigationConsistencyState::Healthy => {}
            NavigationConsistencyState::Suspect => {
                reasons.push(EstimatorHealthReason::NavigationSuspect)
            }
            NavigationConsistencyState::Unreliable => {
                reasons.push(EstimatorHealthReason::NavigationUnreliable)
            }
        }
        if navigation.total_samples > 0
            && navigation_rejection_fraction > self.config.maximum_navigation_rejection_fraction
        {
            reasons.push(EstimatorHealthReason::ExcessiveNavigationRejections);
        }
        if navigation.consecutive_inconsistent > self.config.maximum_consecutive_inconsistent {
            reasons.push(EstimatorHealthReason::InnovationStreakExceeded);
        }

        let hard_fault = reasons.iter().any(|reason| {
            matches!(
                reason,
                EstimatorHealthReason::ClockFaulted
                    | EstimatorHealthReason::NavigationUnreliable
                    | EstimatorHealthReason::InnovationStreakExceeded
            )
        });
        let unavailable = reasons.iter().any(|reason| {
            matches!(
                reason,
                EstimatorHealthReason::NoSensorMeasurements
                    | EstimatorHealthReason::NoNavigationSamples
                    | EstimatorHealthReason::ClockNotLocked
            )
        });
        let raw_state = if hard_fault {
            EstimatorHealthState::Faulted
        } else if unavailable {
            EstimatorHealthState::Unavailable
        } else if reasons.is_empty() {
            EstimatorHealthState::Healthy
        } else {
            EstimatorHealthState::Degraded
        };

        self.update_count = self.update_count.saturating_add(1);
        if raw_state == EstimatorHealthState::Healthy {
            self.consecutive_healthy = self.consecutive_healthy.saturating_add(1);
            self.consecutive_degraded = 0;
            if self.consecutive_healthy >= self.config.recovery_updates {
                self.state = EstimatorHealthState::Healthy;
            }
        } else {
            self.consecutive_degraded = self.consecutive_degraded.saturating_add(1);
            self.consecutive_healthy = 0;
            if matches!(
                raw_state,
                EstimatorHealthState::Faulted | EstimatorHealthState::Unavailable
            ) || self.consecutive_degraded >= self.config.degraded_grace_updates
            {
                self.state = raw_state;
            }
        }

        Ok(EstimatorHealthAssessment {
            state: self.state,
            reasons,
            authority_usable: matches!(
                self.state,
                EstimatorHealthState::Healthy | EstimatorHealthState::Degraded
            ),
            update_count: self.update_count,
            consecutive_degraded: self.consecutive_degraded,
            consecutive_healthy: self.consecutive_healthy,
            bus_rejection_fraction,
            incomplete_snapshot_fraction,
            navigation_rejection_fraction,
        })
    }
}

fn fraction(numerator: u64, denominator: u64) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(state: ClockLockState) -> ClockDisciplineEvidence {
        ClockDisciplineEvidence {
            lock_state: state,
            accepted_samples: 10,
            rejected_samples: 0,
            estimated_offset_s: 0.0,
            estimated_drift_ppm: 0.0,
            last_source_time_s: Some(1.0),
            last_host_time_s: Some(1.0),
        }
    }

    fn bus() -> SensorBusEvidence {
        SensorBusEvidence {
            accepted_measurements: 100,
            rejected_measurements: 0,
            dropped_for_capacity: 0,
            snapshots_built: 10,
            incomplete_snapshots: 0,
        }
    }

    fn navigation(state: NavigationConsistencyState) -> NavigationConsistencyEvidence {
        NavigationConsistencyEvidence {
            state,
            total_samples: 20,
            accepted_samples: 20,
            rejected_samples: 0,
            recent_rejection_fraction: 0.0,
            consecutive_inconsistent: 0,
            maximum_observed_nis: 1.0,
            covariance_collapse_count: 0,
            covariance_inflation_count: 0,
            last_source: None,
        }
    }

    #[test]
    fn healthy_chain_recovers_after_required_updates() {
        let mut manager = EstimatorHealthManager::new(EstimatorHealthConfig::default()).unwrap();
        for _ in 0..3 {
            manager
                .assess(
                    clock(ClockLockState::Locked),
                    bus(),
                    navigation(NavigationConsistencyState::Healthy),
                )
                .unwrap();
        }
        assert_eq!(
            manager
                .assess(
                    clock(ClockLockState::Locked),
                    bus(),
                    navigation(NavigationConsistencyState::Healthy)
                )
                .unwrap()
                .state,
            EstimatorHealthState::Healthy
        );
    }

    #[test]
    fn faulted_clock_is_immediately_unusable() {
        let mut manager = EstimatorHealthManager::new(EstimatorHealthConfig::default()).unwrap();
        let assessment = manager
            .assess(
                clock(ClockLockState::Faulted),
                bus(),
                navigation(NavigationConsistencyState::Healthy),
            )
            .unwrap();
        assert_eq!(assessment.state, EstimatorHealthState::Faulted);
        assert!(!assessment.authority_usable);
    }

    #[test]
    fn excessive_incomplete_snapshots_degrade_after_grace() {
        let mut manager = EstimatorHealthManager::new(EstimatorHealthConfig::default()).unwrap();
        let mut bad_bus = bus();
        bad_bus.incomplete_snapshots = 5;
        manager
            .assess(
                clock(ClockLockState::Locked),
                bad_bus,
                navigation(NavigationConsistencyState::Healthy),
            )
            .unwrap();
        let assessment = manager
            .assess(
                clock(ClockLockState::Locked),
                bad_bus,
                navigation(NavigationConsistencyState::Healthy),
            )
            .unwrap();
        assert_eq!(assessment.state, EstimatorHealthState::Degraded);
        assert!(assessment.authority_usable);
    }

    #[test]
    fn missing_navigation_is_unavailable() {
        let mut manager = EstimatorHealthManager::new(EstimatorHealthConfig::default()).unwrap();
        let mut nav = navigation(NavigationConsistencyState::Healthy);
        nav.total_samples = 0;
        nav.accepted_samples = 0;
        let assessment = manager
            .assess(clock(ClockLockState::Locked), bus(), nav)
            .unwrap();
        assert_eq!(assessment.state, EstimatorHealthState::Unavailable);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Navigation runtime: measurement routing, health monitoring, failover.

use crate::measurements::{Measurement, MeasurementModality};
use serde::{Deserialize, Serialize};

/// Navigation health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NavigationHealth {
    Good,
    Degraded,
    Lost,
}

/// Failover mode when primary positioning fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NavigationFailoverMode {
    DeadReckoning,
    PeerOnly,
    None,
}

/// Policy for routing measurements to positioning algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementRoutingPolicy {
    AllSources,
    LocalOnly,
    HighConfidenceOnly,
}

/// Statistics for measurement routing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeasurementRoutingStats {
    pub total_received: u64,
    pub total_accepted: u64,
    pub total_rejected: u64,
}

/// Routes measurements to the appropriate positioning algorithm.
pub struct MeasurementRouter {
    policy: MeasurementRoutingPolicy,
    stats: MeasurementRoutingStats,
    accepted_modalities: Vec<MeasurementModality>,
}

impl MeasurementRouter {
    pub fn new(policy: MeasurementRoutingPolicy) -> Self {
        Self {
            policy,
            stats: MeasurementRoutingStats::default(),
            accepted_modalities: vec![
                MeasurementModality::UwbToF,
                MeasurementModality::WifiRtt,
                MeasurementModality::Gps,
            ],
        }
    }

    pub fn route(&mut self, measurement: &Measurement) -> bool {
        self.stats.total_received += 1;
        let accept = match self.policy {
            MeasurementRoutingPolicy::AllSources => true,
            MeasurementRoutingPolicy::LocalOnly => {
                measurement.provenance == crate::measurements::MeasurementProvenance::Local
            }
            MeasurementRoutingPolicy::HighConfidenceOnly => {
                self.accepted_modalities.contains(&measurement.modality)
            }
        };
        if accept {
            self.stats.total_accepted += 1;
        } else {
            self.stats.total_rejected += 1;
        }
        accept
    }

    pub fn stats(&self) -> &MeasurementRoutingStats {
        &self.stats
    }
}

/// Domain-specific navigation coordinator.
pub struct DomainNavigator {
    health: NavigationHealth,
    failover: NavigationFailoverMode,
    last_fix_us: u64,
    router: MeasurementRouter,
}

impl DomainNavigator {
    pub fn new() -> Self {
        Self {
            health: NavigationHealth::Good,
            failover: NavigationFailoverMode::DeadReckoning,
            last_fix_us: 0,
            router: MeasurementRouter::new(MeasurementRoutingPolicy::AllSources),
        }
    }
    pub fn health(&self) -> NavigationHealth {
        self.health
    }
    pub fn update_fix(&mut self, timestamp_us: u64) {
        self.last_fix_us = timestamp_us;
        self.health = NavigationHealth::Good;
    }
    pub fn check_health(&mut self, current_us: u64) {
        let age = fix_age_s(self.last_fix_us, current_us);
        if age > 30.0 {
            self.health = NavigationHealth::Lost;
        } else if age > 10.0 {
            self.health = NavigationHealth::Degraded;
        }
    }
    pub fn router_mut(&mut self) -> &mut MeasurementRouter {
        &mut self.router
    }
}

impl Default for DomainNavigator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute fix age in seconds from microsecond timestamps.
pub fn fix_age_s(fix_us: u64, current_us: u64) -> f64 {
    if current_us > fix_us {
        (current_us - fix_us) as f64 / 1_000_000.0
    } else {
        0.0
    }
}

/// Convert sigma (standard deviation) to a confidence percentage [0, 100].
pub fn confidence_from_sigma(sigma_m: f64) -> f64 {
    if sigma_m <= 0.0 {
        return 100.0;
    }
    (100.0 * (-sigma_m / 10.0).exp()).clamp(0.0, 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fix_age() {
        assert!((fix_age_s(1_000_000, 2_000_000) - 1.0).abs() < 0.001);
    }
    #[test]
    fn test_confidence() {
        assert!(confidence_from_sigma(0.0) > 99.0);
        assert!(confidence_from_sigma(50.0) < 1.0);
    }
}

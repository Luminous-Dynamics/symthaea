// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Engine — Zero-click proactive support via telemetry analysis

use crate::types::*;

#[derive(Debug)]
pub struct PredictiveEngine {
    /// Free energy threshold above which alerts are generated.
    alert_threshold: f64,
}

impl PredictiveEngine {
    pub fn new() -> Self {
        Self {
            alert_threshold: 5.0,
        }
    }

    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            alert_threshold: threshold,
        }
    }

    /// Assess system state from telemetry and compute expected free energy.
    pub fn assess_system_state(&self, telemetry: &SystemTelemetry) -> PredictionResult {
        // Compute expected free energy from telemetry anomalies
        let mut efe = 0.0;
        let mut failures: Vec<String> = Vec::new();

        // Memory pressure
        if telemetry.conductor_memory_mb > 4096.0 {
            efe += (telemetry.conductor_memory_mb - 4096.0) / 1024.0;
            failures.push("High conductor memory usage".to_string());
        }

        // Low disk
        if telemetry.disk_free_pct < 10.0 {
            efe += (10.0 - telemetry.disk_free_pct) * 0.5;
            failures.push("Low disk space".to_string());
        }

        // Network issues
        if telemetry.network_latency_ms > 500.0 {
            efe += (telemetry.network_latency_ms - 500.0) / 200.0;
            failures.push("High network latency".to_string());
        }

        // No peers
        if telemetry.gossip_peers == 0 {
            efe += 3.0;
            failures.push("No gossip peers connected".to_string());
        }

        // Error rate
        if telemetry.error_rate_per_min > 10.0 {
            efe += telemetry.error_rate_per_min / 5.0;
            failures.push("Elevated error rate".to_string());
        }

        let predicted_failure = if failures.is_empty() {
            None
        } else {
            Some(failures.join("; "))
        };
        let time_horizon = if efe > self.alert_threshold {
            Some(3600)
        } else {
            None
        }; // 1 hour
        let confidence = (efe / 10.0).min(1.0) as f32;

        PredictionResult {
            expected_free_energy: efe,
            predicted_failure,
            time_horizon_secs: time_horizon,
            confidence,
        }
    }

    /// Check if the prediction warrants an alert.
    pub fn should_alert(&self, prediction: &PredictionResult) -> bool {
        prediction.expected_free_energy > self.alert_threshold
    }
}

impl Default for PredictiveEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_telemetry() -> SystemTelemetry {
        SystemTelemetry {
            conductor_memory_mb: 512.0,
            gossip_peers: 5,
            disk_free_pct: 60.0,
            network_latency_ms: 15.0,
            error_rate_per_min: 0.1,
        }
    }

    #[test]
    fn healthy_system_produces_low_free_energy() {
        let engine = PredictiveEngine::new();
        let result = engine.assess_system_state(&healthy_telemetry());
        assert!(
            result.expected_free_energy < 1.0,
            "Healthy system should have low free energy, got {}",
            result.expected_free_energy
        );
        assert!(result.predicted_failure.is_none());
        assert!(!engine.should_alert(&result));
    }

    #[test]
    fn high_memory_produces_alert() {
        let engine = PredictiveEngine::new();
        let telemetry = SystemTelemetry {
            conductor_memory_mb: 8192.0, // 8GB, well above 4096 threshold
            ..healthy_telemetry()
        };
        let result = engine.assess_system_state(&telemetry);
        assert!(result.expected_free_energy > 0.0);
        assert!(
            result
                .predicted_failure
                .as_ref()
                .unwrap()
                .contains("memory")
        );
    }

    #[test]
    fn no_peers_produces_alert() {
        let engine = PredictiveEngine::new();
        let telemetry = SystemTelemetry {
            gossip_peers: 0,
            ..healthy_telemetry()
        };
        let result = engine.assess_system_state(&telemetry);
        assert!(result.expected_free_energy >= 3.0);
        assert!(
            result
                .predicted_failure
                .as_ref()
                .unwrap()
                .contains("gossip peers")
        );
    }

    #[test]
    fn combined_issues_increase_free_energy() {
        let engine = PredictiveEngine::new();
        let single = SystemTelemetry {
            gossip_peers: 0,
            ..healthy_telemetry()
        };
        let combined = SystemTelemetry {
            gossip_peers: 0,
            conductor_memory_mb: 8192.0,
            disk_free_pct: 5.0,
            ..healthy_telemetry()
        };
        let single_result = engine.assess_system_state(&single);
        let combined_result = engine.assess_system_state(&combined);
        assert!(combined_result.expected_free_energy > single_result.expected_free_energy);
    }

    #[test]
    fn should_alert_works_with_threshold() {
        let engine = PredictiveEngine::with_threshold(2.0);
        let telemetry = SystemTelemetry {
            gossip_peers: 0, // adds 3.0 free energy
            ..healthy_telemetry()
        };
        let result = engine.assess_system_state(&telemetry);
        assert!(engine.should_alert(&result));

        let healthy_result = engine.assess_system_state(&healthy_telemetry());
        assert!(!engine.should_alert(&healthy_result));
    }

    #[test]
    fn time_horizon_set_when_above_threshold() {
        let engine = PredictiveEngine::new();
        let telemetry = SystemTelemetry {
            gossip_peers: 0,
            conductor_memory_mb: 8192.0,
            error_rate_per_min: 50.0,
            ..healthy_telemetry()
        };
        let result = engine.assess_system_state(&telemetry);
        assert!(result.expected_free_energy > 5.0);
        assert_eq!(result.time_horizon_secs, Some(3600));
    }
}

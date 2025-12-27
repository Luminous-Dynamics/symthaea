/*!
 * Telemetry Observer - Real-time metrics aggregation
 *
 * Collects aggregated statistics for monitoring and dashboards.
 */

use super::{SymthaeaObserver, types::*, ObserverStats};
use anyhow::Result;
use std::collections::HashMap;

/// Observer that aggregates real-time metrics
///
/// Use for monitoring dashboards and live metrics.
/// Maintains rolling statistics in memory.
pub struct TelemetryObserver {
    stats: ObserverStats,

    // Φ statistics
    phi_measurements: Vec<f64>,
    phi_min: f64,
    phi_max: f64,
    phi_sum: f64,

    // Router statistics
    router_counts: HashMap<String, usize>,
    router_confidence_sums: HashMap<String, f64>,

    // Workspace statistics
    ignition_count: usize,
    coalition_sizes: Vec<usize>,

    // Performance statistics
    language_step_durations: Vec<u64>,

    // Security statistics
    security_denials: usize,
    security_confirmations: usize,
}

impl Default for TelemetryObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryObserver {
    pub fn new() -> Self {
        Self {
            stats: ObserverStats::default(),
            phi_measurements: Vec::new(),
            phi_min: f64::INFINITY,
            phi_max: f64::NEG_INFINITY,
            phi_sum: 0.0,
            router_counts: HashMap::new(),
            router_confidence_sums: HashMap::new(),
            ignition_count: 0,
            coalition_sizes: Vec::new(),
            language_step_durations: Vec::new(),
            security_denials: 0,
            security_confirmations: 0,
        }
    }

    /// Get current Φ statistics
    pub fn phi_stats(&self) -> PhiStats {
        PhiStats {
            count: self.phi_measurements.len(),
            min: self.phi_min,
            max: self.phi_max,
            average: if !self.phi_measurements.is_empty() {
                self.phi_sum / self.phi_measurements.len() as f64
            } else {
                0.0
            },
            latest: self.phi_measurements.last().copied(),
        }
    }

    /// Get router statistics
    pub fn router_stats(&self) -> Vec<RouterStats> {
        let mut stats = Vec::new();

        for (router, count) in &self.router_counts {
            let avg_confidence = self.router_confidence_sums.get(router)
                .map(|sum| sum / *count as f64)
                .unwrap_or(0.0);

            stats.push(RouterStats {
                name: router.clone(),
                count: *count,
                average_confidence: avg_confidence,
            });
        }

        stats.sort_by(|a, b| b.count.cmp(&a.count));
        stats
    }

    /// Get workspace statistics
    pub fn workspace_stats(&self) -> WorkspaceStats {
        WorkspaceStats {
            ignition_count: self.ignition_count,
            average_coalition_size: if !self.coalition_sizes.is_empty() {
                self.coalition_sizes.iter().sum::<usize>() as f64 / self.coalition_sizes.len() as f64
            } else {
                0.0
            },
            max_coalition_size: self.coalition_sizes.iter().max().copied().unwrap_or(0),
        }
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> PerformanceStats {
        PerformanceStats {
            average_language_step_ms: if !self.language_step_durations.is_empty() {
                self.language_step_durations.iter().sum::<u64>() as f64 / self.language_step_durations.len() as f64
            } else {
                0.0
            },
            total_language_steps: self.language_step_durations.len(),
        }
    }

    /// Get security statistics
    pub fn security_stats(&self) -> SecurityStats {
        SecurityStats {
            denials: self.security_denials,
            confirmations: self.security_confirmations,
            total_checks: self.stats.security_checks,
        }
    }

    /// Export all metrics as JSON
    pub fn export_metrics(&self) -> Result<String> {
        let metrics = serde_json::json!({
            "phi": self.phi_stats(),
            "routers": self.router_stats(),
            "workspace": self.workspace_stats(),
            "performance": self.performance_stats(),
            "security": self.security_stats(),
            "total_events": self.stats.total_events,
        });

        Ok(serde_json::to_string_pretty(&metrics)?)
    }
}

impl SymthaeaObserver for TelemetryObserver {
    fn record_router_selection(&mut self, event: RouterSelectionEvent) -> Result<()> {
        self.stats.increment_event("router_selection");

        *self.router_counts.entry(event.selected_router.clone()).or_insert(0) += 1;
        *self.router_confidence_sums.entry(event.selected_router).or_insert(0.0) += event.confidence;

        Ok(())
    }

    fn record_workspace_ignition(&mut self, event: WorkspaceIgnitionEvent) -> Result<()> {
        self.stats.increment_event("workspace_ignition");

        self.ignition_count += 1;
        self.coalition_sizes.push(event.coalition_size);

        // Also track Φ from ignitions
        self.phi_measurements.push(event.phi);
        self.phi_sum += event.phi;
        self.phi_min = self.phi_min.min(event.phi);
        self.phi_max = self.phi_max.max(event.phi);

        Ok(())
    }

    fn record_phi_measurement(&mut self, event: PhiMeasurementEvent) -> Result<()> {
        self.stats.increment_event("phi_measurement");

        self.phi_measurements.push(event.phi);
        self.phi_sum += event.phi;
        self.phi_min = self.phi_min.min(event.phi);
        self.phi_max = self.phi_max.max(event.phi);

        Ok(())
    }

    fn record_primitive_activation(&mut self, event: PrimitiveActivationEvent) -> Result<()> {
        self.stats.increment_event("primitive_activation");
        Ok(())
    }

    fn record_response_generated(&mut self, event: ResponseGeneratedEvent) -> Result<()> {
        self.stats.increment_event("response_generated");
        Ok(())
    }

    fn record_security_check(&mut self, event: SecurityCheckEvent) -> Result<()> {
        self.stats.increment_event("security_check");

        match event.decision {
            SecurityDecision::Denied => self.security_denials += 1,
            SecurityDecision::RequiresConfirmation => self.security_confirmations += 1,
            SecurityDecision::Allowed => {}
        }

        Ok(())
    }

    fn record_error(&mut self, event: ErrorEvent) -> Result<()> {
        self.stats.increment_event("error");
        Ok(())
    }

    fn record_language_step(&mut self, event: LanguageStepEvent) -> Result<()> {
        self.stats.increment_event("language_step");
        self.language_step_durations.push(event.duration_ms);
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn get_stats(&self) -> Result<ObserverStats> {
        Ok(self.stats.clone())
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PhiStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub average: f64,
    pub latest: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RouterStats {
    pub name: String,
    pub count: usize,
    pub average_confidence: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkspaceStats {
    pub ignition_count: usize,
    pub average_coalition_size: f64,
    pub max_coalition_size: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PerformanceStats {
    pub average_language_step_ms: f64,
    pub total_language_steps: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SecurityStats {
    pub denials: usize,
    pub confirmations: usize,
    pub total_checks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_observer() {
        let mut observer = TelemetryObserver::new();

        // Record some events
        observer.record_phi_measurement(PhiMeasurementEvent {
            phi: 0.75,
            ..Default::default()
        }).unwrap();

        observer.record_phi_measurement(PhiMeasurementEvent {
            phi: 0.85,
            ..Default::default()
        }).unwrap();

        observer.record_router_selection(RouterSelectionEvent {
            selected_router: "TestRouter".to_string(),
            confidence: 0.9,
            ..Default::default()
        }).unwrap();

        // Check Φ stats
        let phi_stats = observer.phi_stats();
        assert_eq!(phi_stats.count, 2);
        assert_eq!(phi_stats.min, 0.75);
        assert_eq!(phi_stats.max, 0.85);
        assert!((phi_stats.average - 0.8).abs() < 0.01);

        // Check router stats
        let router_stats = observer.router_stats();
        assert_eq!(router_stats.len(), 1);
        assert_eq!(router_stats[0].name, "TestRouter");
        assert_eq!(router_stats[0].count, 1);

        // Export metrics
        let metrics_json = observer.export_metrics().unwrap();
        assert!(metrics_json.contains("phi"));
        assert!(metrics_json.contains("routers"));
    }
}

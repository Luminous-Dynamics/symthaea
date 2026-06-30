// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::correlation_auditor::CorrelationAuditor;
use crate::stability_audit::StabilityAuditHarvester;

/// Aggregates all system stability telemetry signals for a holistic audit.
pub struct CognitiveResilienceHarvester {
    reasoning_sensor: StabilityAuditHarvester,
    heartbeat_sensor: StabilityAuditHarvester,
    repair_sensor: StabilityAuditHarvester,
    auditor: CorrelationAuditor,
}

impl CognitiveResilienceHarvester {
    pub fn new(window_size: usize) -> Self {
        Self {
            reasoning_sensor: StabilityAuditHarvester::new(window_size),
            heartbeat_sensor: StabilityAuditHarvester::new(window_size),
            repair_sensor: StabilityAuditHarvester::new(window_size),
            auditor: CorrelationAuditor::new(),
        }
    }

    pub fn record_reasoning(&mut self, signal: f64) {
        self.reasoning_sensor.record(signal);
    }
    pub fn record_heartbeat(&mut self, signal: f64) {
        self.heartbeat_sensor.record(signal);
    }
    pub fn record_repair(&mut self, signal: f64) {
        self.repair_sensor.record(signal);
    }

    /// Audit resilience: returns a score (0-1) quantifying systemic balance.
    pub fn audit_resilience(&self) -> f64 {
        let r = self.reasoning_sensor.buffer();
        let h = self.heartbeat_sensor.buffer();
        let p = self.repair_sensor.buffer();

        // 1. Coherence: Reasoning vs Heartbeat
        let coherence = self.auditor.calculate_coupling(h, r);

        // 2. Efficiency: Repair energy vs Reasoning Stability
        let repair_cost = p.iter().sum::<f64>() / (p.len() as f64).max(1.0);
        let stability = r.iter().sum::<f64>() / (r.len() as f64).max(1.0);

        let efficiency = if stability > 0.0 {
            (1.0 - (repair_cost / stability).min(1.0))
        } else {
            1.0
        };

        (coherence * 0.5 + efficiency * 0.5).clamp(0.0, 1.0)
    }
}

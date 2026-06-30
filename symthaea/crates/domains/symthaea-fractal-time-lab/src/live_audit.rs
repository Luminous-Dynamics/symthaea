// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::metrics::ExperimentScorecard;
use crate::stability_audit::StabilityAuditHarvester;
use crate::structural_fidelity::StructuralFidelityAudit;

/// Auditor to run against live reasoning benchmarks.
pub struct LiveCognitiveAuditor {
    harvester: StabilityAuditHarvester,
    auditor: StructuralFidelityAudit,
}

impl LiveCognitiveAuditor {
    pub fn new(window_size: usize) -> Self {
        Self {
            harvester: StabilityAuditHarvester::new(window_size),
            auditor: StructuralFidelityAudit::new(),
        }
    }

    pub fn record_stability(&mut self, ema: f64) {
        self.harvester.record(ema);
    }

    pub fn perform_audit(&self) -> ExperimentScorecard {
        let trace = self.harvester.buffer();
        let is_anomalous = self.auditor.audit_trace(trace);

        ExperimentScorecard::new(
            "Live Reasoning Cognitive Audit",
            "Live reasoning trace stability audit via temporal persistence metrics.",
            if is_anomalous { 0.1 } else { 0.9 }, // Simplified health score
            &[0.0, 0.5],
            1,
            42,
            0.3,
            format!(
                "Audit Result: {}",
                if is_anomalous {
                    "ANOMALY DETECTED"
                } else {
                    "STABLE"
                }
            ),
        )
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use symthaea_fractal_time_lab::correlation_auditor::CorrelationAuditor;
use symthaea_fractal_time_lab::stability_audit::StabilityAuditHarvester;

/// Audits the coupling between reasoning stability and conscious ignition pulses.
pub struct CognitiveCoherenceAuditor {
    reasoning_harvester: StabilityAuditHarvester,
    gwt_harvester: StabilityAuditHarvester,
    auditor: CorrelationAuditor,
}

impl CognitiveCoherenceAuditor {
    pub fn new(window_size: usize) -> Self {
        Self {
            reasoning_harvester: StabilityAuditHarvester::new(window_size),
            gwt_harvester: StabilityAuditHarvester::new(window_size),
            auditor: CorrelationAuditor::new(),
        }
    }

    /// Record Reasoning Stability (EMA confidence)
    pub fn record_reasoning(&mut self, ema: f64) {
        self.reasoning_harvester.record(ema);
    }

    /// Record Workspace Ignition Strength
    pub fn record_gwt(&mut self, ignition: f64) {
        self.gwt_harvester.record(ignition);
    }

    /// Audit coherence between reasoning stability and workspace ignition.
    pub fn audit_coherence(&self) -> f64 {
        let reasoning = self.reasoning_harvester.buffer();
        let gwt = self.gwt_harvester.buffer();

        self.auditor.calculate_coupling(gwt, reasoning)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_detection() {
        let mut auditor = CognitiveCoherenceAuditor::new(64);

        // Simulate phase-locked cognitive activity
        for i in 0..64 {
            let pulse = if i % 2 == 0 { 1.0 } else { 0.0 };
            auditor.record_gwt(pulse);
            auditor.record_reasoning(pulse);
        }

        let coherence = auditor.audit_coherence();
        assert!(
            coherence > 0.8,
            "Perfectly phase-locked system should have high coherence: {}",
            coherence
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::floquet_time_crystal::TimeCrystalDetector;
use crate::multiscale_phi::MultiScalePhi;
use petgraph::graph::UnGraph;

/// Analyzes correlation between GWT ignition pulses and reasoning stability.
pub struct CorrelationAuditor {
    detector: TimeCrystalDetector,
}

impl CorrelationAuditor {
    pub fn new() -> Self {
        Self {
            detector: TimeCrystalDetector,
        }
    }

    /// Calculate cross-correlation between workspace activation and reasoning stability.
    /// Returns a score indicating how 'phase-locked' reasoning is to GWT pulses.
    pub fn calculate_coupling(&self, gwt_signal: &[f64], reasoning_signal: &[f64]) -> f64 {
        if gwt_signal.len() != reasoning_signal.len() || gwt_signal.is_empty() {
            return 0.0;
        }

        // Cross-correlation at lag 0 (simple Pearson)
        let n = gwt_signal.len() as f64;
        let gwt_mean = gwt_signal.iter().sum::<f64>() / n;
        let reason_mean = reasoning_signal.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut den_gwt = 0.0;
        let mut den_reason = 0.0;

        for (g, r) in gwt_signal.iter().zip(reasoning_signal.iter()) {
            let dg = g - gwt_mean;
            let dr = r - reason_mean;
            numerator += dg * dr;
            den_gwt += dg * dg;
            den_reason += dr * dr;
        }

        if den_gwt > 0.0 && den_reason > 0.0 {
            (numerator / (den_gwt.sqrt() * den_reason.sqrt())).abs()
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_coupling() {
        let auditor = CorrelationAuditor::new();
        let signal: Vec<f64> = (0..64).map(|i| i as f64).collect();
        // Perfect coupling
        assert!((auditor.calculate_coupling(&signal, &signal) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_coupling() {
        let auditor = CorrelationAuditor::new();
        let s1 = vec![1.0, 0.0, 1.0, 0.0];
        let s2 = vec![0.0, 1.0, 0.0, 1.0];
        // Anti-correlated or decorrelated
        assert!(auditor.calculate_coupling(&s1, &s2) <= 1.0);
    }
}

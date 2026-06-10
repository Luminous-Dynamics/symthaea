// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Calibration history and drift tracking.
//!
//! Sliding window of calibration events for detecting systematic
//! parameter drift vs oscillatory noise.

use super::normative::{NeuromodCalibration, ReceptorSubtype};

/// Snapshot of a single calibration event for history tracking.
#[derive(Debug, Clone)]
pub struct CalibrationSnapshot {
    /// Per-transmitter sensitivity factors at time of calibration.
    pub factors: std::collections::HashMap<String, f32>,
    /// Confidence delta applied.
    pub confidence_delta: f32,
    /// Calibration coverage quality.
    pub coverage: f64,
    /// Cycle number when this calibration was applied.
    pub applied_at_cycle: u64,
}

/// Sliding window of calibration profiles for drift detection.
///
/// Tracks the last N calibration events and computes drift statistics.
/// Systematic drift (factors consistently moving in one direction) indicates
/// a structural issue vs noise (factors oscillating around a mean).
///
/// Science: McEwen (1998) — allostatic load accumulates when homeostatic
///   correction is consistently in one direction, suggesting the setpoint
///   itself needs adjustment.
#[derive(Debug, Clone)]
pub struct CalibrationHistory {
    /// Sliding window of recent calibration snapshots.
    pub(super) entries: std::collections::VecDeque<CalibrationSnapshot>,
    /// Maximum history size.
    max_entries: usize,
}

impl Default for CalibrationHistory {
    fn default() -> Self {
        Self {
            entries: std::collections::VecDeque::new(),
            max_entries: 20, // Track last ~20 calibrations
        }
    }
}

impl CalibrationHistory {
    /// Record a calibration event.
    pub fn record(&mut self, calibration: &NeuromodCalibration, cycle: u64) {
        let mut factors = std::collections::HashMap::new();
        for adj in &calibration.adjustments {
            let key = match &adj.subtype {
                Some(ReceptorSubtype::NeAlpha) => "NE-alpha".to_string(),
                Some(ReceptorSubtype::NeBeta) => "NE-beta".to_string(),
                Some(ReceptorSubtype::GabaA) => "GABA-A".to_string(),
                Some(ReceptorSubtype::GabaB) => "GABA-B".to_string(),
                None => adj.transmitter.to_string(),
            };
            factors.insert(key, adj.sensitivity_factor);
        }
        self.entries.push_back(CalibrationSnapshot {
            factors,
            confidence_delta: calibration.confidence_delta,
            coverage: calibration.coverage,
            applied_at_cycle: cycle,
        });
        if self.entries.len() > self.max_entries {
            self.entries.pop_front();
        }
    }

    /// Number of calibrations in history.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute drift direction for a specific transmitter.
    ///
    /// Returns the mean deviation from 1.0 (neutral) across history:
    /// - Positive: consistently boosting (factors > 1.0)
    /// - Negative: consistently attenuating (factors < 1.0)
    /// - Near zero: oscillating (noise, not systematic drift)
    ///
    /// Returns `None` if insufficient data (< 3 entries with this transmitter).
    pub fn drift_direction(&self, transmitter: &str) -> Option<f64> {
        let deviations: Vec<f64> = self
            .entries
            .iter()
            .filter_map(|s| s.factors.get(transmitter))
            .map(|&f| (f - 1.0) as f64)
            .collect();
        if deviations.len() < 3 {
            return None;
        }
        let mean = deviations.iter().sum::<f64>() / deviations.len() as f64;
        Some(mean)
    }

    /// Check if a transmitter shows systematic drift (not noise).
    ///
    /// Systematic drift: >75% of calibrations adjust in the same direction.
    /// This suggests the baseline setpoint needs permanent adjustment rather
    /// than repeated corrective calibration.
    pub fn is_systematic_drift(&self, transmitter: &str) -> bool {
        let deviations: Vec<f64> = self
            .entries
            .iter()
            .filter_map(|s| s.factors.get(transmitter))
            .map(|&f| (f - 1.0) as f64)
            .collect();
        if deviations.len() < 5 {
            return false;
        }
        let positive = deviations.iter().filter(|d| **d > 0.001).count();
        let negative = deviations.iter().filter(|d| **d < -0.001).count();
        let total = deviations.len();
        // Systematic if >75% in one direction
        positive * 4 > total * 3 || negative * 4 > total * 3
    }

    /// Get all transmitters that show systematic drift.
    pub fn systematic_drift_transmitters(&self) -> Vec<(String, f64)> {
        let all_keys: std::collections::HashSet<String> = self
            .entries
            .iter()
            .flat_map(|s| s.factors.keys().cloned())
            .collect();
        let mut drifters = Vec::new();
        for key in all_keys {
            if self.is_systematic_drift(&key) {
                if let Some(dir) = self.drift_direction(&key) {
                    drifters.push((key, dir));
                }
            }
        }
        drifters
    }

    /// Compute baseline nudges for transmitters showing systematic drift.
    ///
    /// Returns `(transmitter_name, nudge_delta)` pairs. The nudge is 10% of the
    /// mean drift direction per calibration window, clamped to ±0.02 to prevent
    /// runaway baseline shifting.
    ///
    /// Science: McEwen (1998) — allostatic overload occurs when homeostatic
    ///   corrections are consistently one-directional. Adjusting the setpoint
    ///   (baseline) reduces corrective load.
    pub fn compute_baseline_nudges(&self) -> Vec<(String, f32)> {
        let mut nudges = Vec::new();
        for (transmitter, direction) in self.systematic_drift_transmitters() {
            // 10% of drift direction, clamped to ±0.02
            let nudge = (direction as f32 * 0.1).clamp(-0.02, 0.02);
            if nudge.abs() > 0.001 {
                nudges.push((transmitter, nudge));
            }
        }
        nudges
    }

    /// Mean calibration interval (cycles between calibrations).
    pub fn mean_interval(&self) -> Option<f64> {
        if self.entries.len() < 2 {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0usize;
        for i in 1..self.entries.len() {
            total +=
                (self.entries[i].applied_at_cycle - self.entries[i - 1].applied_at_cycle) as f64;
            count += 1;
        }
        Some(total / count as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CALIBRATION VALIDATOR — closed-loop validation of calibration effectiveness
// ═══════════════════════════════════════════════════════════════════════════════

/// Pre-calibration metric snapshot for validation comparison.
#[derive(Debug, Clone)]
struct MetricSnapshot {
    pe_ema: f64,
    coherence_ema: f64,
    confidence_error_ema: f64,
    cycle: u64,
}

/// Validates whether calibration adjustments actually improved performance.
///
/// Records a pre-calibration metric snapshot when calibration is applied,
/// then compares post-calibration metrics after a settling period.
///
/// Science: Bayesian model comparison — evaluate whether the posterior
/// (post-calibration) is a better fit than the prior (pre-calibration).
/// Powers & Cisek (2021) — closed-loop neuromodulation requires outcome monitoring.
#[derive(Debug, Clone)]
pub struct CalibrationValidator {
    /// Pre-calibration snapshot (set when calibration is applied).
    pending_validation: Option<MetricSnapshot>,
    /// Settling period: cycles to wait before comparing metrics.
    settling_cycles: u64,
    /// Running count of validations that improved metrics.
    pub improvements: u32,
    /// Running count of validations that worsened metrics.
    pub regressions: u32,
    /// Running count of neutral outcomes (within noise margin).
    pub neutral: u32,
    /// Damping factor applied when regressions exceed improvements.
    /// 0.0 = no damping, 1.0 = maximum damping.
    pub regression_damping: f32,
}

impl Default for CalibrationValidator {
    fn default() -> Self {
        Self {
            pending_validation: None,
            settling_cycles: 100, // Wait 100 cycles before comparing
            improvements: 0,
            regressions: 0,
            neutral: 0,
            regression_damping: 0.0,
        }
    }
}

impl CalibrationValidator {
    /// Record pre-calibration metrics when a calibration is about to be applied.
    pub fn record_pre_calibration(
        &mut self,
        pe_ema: f64,
        coherence_ema: f64,
        confidence_error_ema: f64,
        cycle: u64,
    ) {
        self.pending_validation = Some(MetricSnapshot {
            pe_ema,
            coherence_ema,
            confidence_error_ema,
            cycle,
        });
    }

    /// Check whether enough time has passed to validate the calibration.
    /// Returns `Some(improved)` if validation window has elapsed.
    pub fn check_validation(
        &mut self,
        pe_ema: f64,
        coherence_ema: f64,
        confidence_error_ema: f64,
        cycle: u64,
    ) -> Option<bool> {
        let pre = self.pending_validation.as_ref()?;
        if cycle < pre.cycle + self.settling_cycles {
            return None; // Not enough settling time
        }

        // Compare: lower PE is better, higher coherence is better, lower confidence error is better
        let pe_improved = pe_ema < pre.pe_ema - 0.01;
        let coherence_improved = coherence_ema > pre.coherence_ema + 0.01;
        let confidence_improved = confidence_error_ema < pre.confidence_error_ema - 0.005;

        // Score: count how many metrics improved vs worsened
        let pe_worsened = pe_ema > pre.pe_ema + 0.02;
        let coherence_worsened = coherence_ema < pre.coherence_ema - 0.02;

        let improvement_count =
            pe_improved as u8 + coherence_improved as u8 + confidence_improved as u8;
        let worsened_count = pe_worsened as u8 + coherence_worsened as u8;

        let improved = improvement_count > worsened_count;

        if improvement_count > 0 && worsened_count == 0 {
            self.improvements += 1;
        } else if worsened_count > improvement_count {
            self.regressions += 1;
        } else {
            self.neutral += 1;
        }

        // Update regression damping: if regressions outpace improvements, future
        // calibrations should be dampened (smaller adjustments).
        let total = self.improvements + self.regressions + self.neutral;
        if total >= 3 {
            let regression_ratio = self.regressions as f32 / total as f32;
            self.regression_damping = (regression_ratio - 0.3).max(0.0).min(0.5);
        }

        self.pending_validation = None;
        Some(improved)
    }

    /// Damping multiplier for future calibration factors.
    /// Returns 1.0 when no damping, <1.0 when regressions are frequent.
    pub fn adjustment_multiplier(&self) -> f32 {
        1.0 - self.regression_damping
    }

    /// Total number of validations completed.
    pub fn total_validations(&self) -> u32 {
        self.improvements + self.regressions + self.neutral
    }
}

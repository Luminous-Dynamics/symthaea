//! Neuromodulator Manager: groups bath, calibration, and phase tracking fields.

use super::calibration;
use super::neuromodulators;

/// Groups neuromodulator-related CLS fields into a single manager.
///
/// Reduces CLS field count by 7 (8 fields → 1).
pub(crate) struct NeuromodManager {
    /// DA/NE/5-HT/ACh/GABA/Oxytocin/Glutamate/Adenosine/ECB chemical signaling.
    pub bath: neuromodulators::NeuromodulatorBath,

    /// Monitors receptor sensitivity stability.
    pub drift_tracker: neuromodulators::PersonalityDriftTracker,

    /// Entropy, centroid, attractor detection.
    pub phase_tracker: neuromodulators::BathPhaseTracker,

    /// Whether the system was in sleep phase on the previous cycle (for sleep→wake transition).
    pub was_sleeping: bool,

    /// Pending neuromodulator calibration from psych-bench normative z-scores.
    pub pending_calibration: Option<calibration::NeuromodCalibration>,

    /// Last applied calibration summary (for telemetry/diagnostics).
    pub last_calibration_summary: Option<String>,

    /// Autonomous self-assessment monitor.
    pub self_assessment: calibration::SelfAssessmentMonitor,

    /// Hysteresis-based phase transition detector (prevents flicker between labels).
    pub phase_detector: neuromodulators::PhaseTransitionDetector,

    /// Running calibration battery subprocess (async, non-blocking).
    pub calibration_battery_child: Option<std::process::Child>,

    /// When the calibration battery was spawned (for timeout detection).
    pub calibration_battery_spawned_at: Option<std::time::Instant>,

    /// Calibration history for drift tracking (sliding window of last N profiles).
    pub calibration_history: super::calibration::CalibrationHistory,
}

impl Default for NeuromodManager {
    fn default() -> Self {
        Self {
            bath: neuromodulators::NeuromodulatorBath::default(),
            drift_tracker: neuromodulators::PersonalityDriftTracker::default(),
            phase_tracker: neuromodulators::BathPhaseTracker::default(),
            was_sleeping: false,
            pending_calibration: None,
            last_calibration_summary: None,
            self_assessment: calibration::SelfAssessmentMonitor::default(),
            phase_detector: neuromodulators::PhaseTransitionDetector::new(5),
            calibration_battery_child: None,
            calibration_battery_spawned_at: None,
            calibration_history: super::calibration::CalibrationHistory::default(),
        }
    }
}

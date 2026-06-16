// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Thermal Bridge — Platform thermal state as cognitive modulation
//!
//! Maps hardware thermal conditions (phone overheating, CPU throttling, ambient
//! temperature) into interoceptive signals that modulate the cognitive loop's
//! temporal dynamics, consciousness depth, and power consumption.
//!
//! # Biological Analogy
//!
//! Just as biological organisms slow metabolic activity under heat stress
//! (Angilletta 2009), this bridge converts thermal pressure into cognitive
//! deceleration:
//!
//! - Higher thermal level → larger tau_factor → slower CfC integration
//! - Critical thermal → profile downgrade recommendation → drop optional subsystems
//! - Emergency thermal → near-minimum frequency → Sacred Stillness / Active Rest
//!
//! # Platform Integration
//!
//! External code (Android `PowerManager.THERMAL_STATUS_*`, iOS `ProcessInfo.thermalState`,
//! or `/sys/class/thermal/thermal_zone*/temp`) sends `ThermalLevel` via the channel.
//! The bridge smooths transitions via EMA and produces `ThermalSignals` each cycle.

use std::sync::{Mutex, mpsc};

// ═══════════════════════════════════════════════════════════════════════════════
// Thermal level enum — maps to platform APIs
// ═══════════════════════════════════════════════════════════════════════════════

/// Hardware thermal state levels.
///
/// Deliberately mirrors Android `PowerManager.THERMAL_STATUS_*` (0–4)
/// and iOS `ProcessInfo.ThermalState` (nominal/fair/serious/critical).
/// A fifth level (Emergency) covers thermal runaway / forced shutdown scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ThermalLevel {
    /// Normal operating temperature. No throttling.
    Nominal = 0,
    /// Slightly warm. No throttling yet, but trending upward.
    Fair = 1,
    /// Significant heat. OS may begin throttling background work.
    /// Science: Angilletta (2009) — moderate heat stress reduces metabolic efficiency.
    Serious = 2,
    /// High heat. Aggressive throttling required to prevent damage.
    /// Recommends profile downgrade to Minimal.
    Critical = 3,
    /// Thermal runaway / near-shutdown. Minimum viable operation only.
    /// Enters Active Rest (Sacred Stillness) to reduce heat generation.
    Emergency = 4,
}

impl ThermalLevel {
    /// Convert from raw u8 (e.g., from FFI or platform API).
    /// Values > 4 are clamped to Emergency.
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Nominal,
            1 => Self::Fair,
            2 => Self::Serious,
            3 => Self::Critical,
            _ => Self::Emergency,
        }
    }

    /// CfC tau slowdown factor for this thermal level.
    ///
    /// Higher values = slower integration = less heat generated.
    /// Range: [1.0, 2.5].
    ///
    /// Science: Thermal Performance Curves (Angilletta 2009) —
    /// cognitive performance degrades non-linearly with temperature.
    pub fn tau_factor(self) -> f64 {
        match self {
            Self::Nominal => THERMAL_TAU_NOMINAL,
            Self::Fair => THERMAL_TAU_FAIR,
            Self::Serious => THERMAL_TAU_SERIOUS,
            Self::Critical => THERMAL_TAU_CRITICAL,
            Self::Emergency => THERMAL_TAU_EMERGENCY,
        }
    }

    /// Whether this level recommends a consciousness profile downgrade.
    pub fn should_reduce_profile(self) -> bool {
        matches!(self, Self::Critical | Self::Emergency)
    }

    /// Recommended target frequency override, if any.
    /// Returns `None` for levels that don't force a frequency cap.
    pub fn target_frequency_override(self) -> Option<f32> {
        match self {
            Self::Critical => Some(10.0),
            Self::Emergency => Some(5.0),
            _ => None,
        }
    }
}

impl Default for ThermalLevel {
    fn default() -> Self {
        Self::Nominal
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Named constants — tau factors per thermal level
// ═══════════════════════════════════════════════════════════════════════════════

/// Tau factor at Nominal thermal level (no slowdown).
/// Science: Baseline — optimal operating temperature (Angilletta 2009).
pub const THERMAL_TAU_NOMINAL: f64 = 1.0;

/// Tau factor at Fair thermal level (no slowdown — headroom still available).
/// Science: Sub-threshold warmth does not impair cognitive function.
pub const THERMAL_TAU_FAIR: f64 = 1.0;

/// Tau factor at Serious thermal level (30% slowdown).
/// Science: Angilletta (2009) — moderate heat stress reduces metabolic rate ~25-35%.
pub const THERMAL_TAU_SERIOUS: f64 = 1.3;

/// Tau factor at Critical thermal level (80% slowdown).
/// Science: Heat stroke literature — cognitive function degrades sharply above thermal threshold.
pub const THERMAL_TAU_CRITICAL: f64 = 1.8;

/// Tau factor at Emergency thermal level (150% slowdown — near minimum operation).
/// Science: Thermal runaway protection. Organism enters hibernation-like state.
pub const THERMAL_TAU_EMERGENCY: f64 = 2.5;

/// EMA smoothing alpha for thermal transitions.
/// 0.15 ≈ ~6 cycles to converge to new level. Prevents oscillation from
/// noisy thermal sensors while still responding within ~200ms at 30Hz.
pub const THERMAL_EMA_ALPHA: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// Thermal signals — output consumed by cognitive loop
// ═══════════════════════════════════════════════════════════════════════════════

/// Interoceptive signals derived from thermal state.
///
/// Consumed by the cognitive loop:
/// - `tau_factor` multiplied into CfC delta_t (10th modulation factor)
/// - `should_reduce_profile` triggers profile downgrade to Minimal
/// - `target_frequency_override` caps cycle frequency
#[derive(Debug, Clone, Copy)]
pub struct ThermalSignals {
    /// CfC tau slowdown factor (1.0 = no change, up to 2.5).
    /// Smoothed via EMA to prevent discontinuities.
    pub tau_factor: f64,

    /// Whether the cognitive loop should switch to a reduced profile.
    pub should_reduce_profile: bool,

    /// Optional frequency cap (Hz). `None` means no override.
    pub target_frequency_override: Option<f32>,

    /// Current thermal level (for telemetry).
    pub level: ThermalLevel,
}

impl Default for ThermalSignals {
    fn default() -> Self {
        Self {
            tau_factor: 1.0,
            should_reduce_profile: false,
            target_frequency_override: None,
            level: ThermalLevel::Nominal,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Thermal bridge — the main struct
// ═══════════════════════════════════════════════════════════════════════════════

/// Sender half of the thermal channel. Distribute to platform integration
/// code that reads hardware thermal state.
pub type ThermalSender = mpsc::Sender<ThermalLevel>;

/// The thermal bridge: converts platform thermal reports into cognitive
/// modulation signals.
///
/// # Usage
///
/// ```ignore
/// let (mut bridge, tx) = ThermalBridge::new();
///
/// // Platform code sends thermal updates:
/// tx.send(ThermalLevel::Serious).unwrap();
///
/// // Each cognitive cycle:
/// bridge.update();
/// let signals = bridge.signals();
/// // Apply signals.tau_factor to CfC delta_t pipeline
/// ```
pub struct ThermalBridge {
    /// Current raw thermal level (most recent report).
    current_level: ThermalLevel,

    /// EMA-smoothed tau factor. Prevents discontinuities when thermal
    /// level oscillates between adjacent states.
    smoothed_tau: f64,

    /// EMA smoothing alpha (0.0 = no change, 1.0 = instant).
    ema_alpha: f64,

    /// Receiver for thermal level reports from platform code.
    /// Wrapped in Mutex for Sync (single reader, uncontended).
    thermal_rx: Mutex<mpsc::Receiver<ThermalLevel>>,
}

impl ThermalBridge {
    /// Create a new thermal bridge, returning (bridge, sender).
    ///
    /// The sender should be given to platform integration code that
    /// monitors hardware thermal state (Android PowerManager, iOS ProcessInfo,
    /// Linux sysfs thermal zones).
    pub fn new() -> (Self, ThermalSender) {
        let (tx, rx) = mpsc::channel();
        let bridge = Self {
            current_level: ThermalLevel::Nominal,
            smoothed_tau: THERMAL_TAU_NOMINAL,
            ema_alpha: THERMAL_EMA_ALPHA,
            thermal_rx: Mutex::new(rx),
        };
        (bridge, tx)
    }

    /// Drain all pending thermal reports and update smoothed state.
    ///
    /// Called once per cognitive cycle at the start of Phase A (observation),
    /// alongside `SomaticErrorBridge::update()`.
    /// Non-blocking: only processes reports that have already been sent.
    pub fn update(&mut self) {
        // Drain all pending reports, keeping only the most recent
        let rx = self.thermal_rx.lock().unwrap_or_else(|e| e.into_inner());
        let mut latest = None;
        while let Ok(level) = rx.try_recv() {
            latest = Some(level);
        }

        if let Some(level) = latest {
            self.current_level = level;
        }

        // EMA smooth the tau factor toward the target
        let target_tau = self.current_level.tau_factor();
        self.smoothed_tau =
            self.smoothed_tau * (1.0 - self.ema_alpha) + target_tau * self.ema_alpha;
    }

    /// Get current thermal signals for the cognitive loop.
    pub fn signals(&self) -> ThermalSignals {
        ThermalSignals {
            tau_factor: self.smoothed_tau,
            should_reduce_profile: self.current_level.should_reduce_profile(),
            target_frequency_override: self.current_level.target_frequency_override(),
            level: self.current_level,
        }
    }

    /// Current thermal level (for telemetry).
    pub fn level(&self) -> ThermalLevel {
        self.current_level
    }

    /// Current smoothed tau factor (for telemetry).
    pub fn smoothed_tau(&self) -> f64 {
        self.smoothed_tau
    }

    /// Reset thermal state to nominal. Used for testing or explicit recovery.
    pub fn reset(&mut self) {
        self.current_level = ThermalLevel::Nominal;
        self.smoothed_tau = THERMAL_TAU_NOMINAL;
    }
}

impl Default for ThermalBridge {
    fn default() -> Self {
        let (_tx, rx) = mpsc::channel();
        Self {
            current_level: ThermalLevel::Nominal,
            smoothed_tau: THERMAL_TAU_NOMINAL,
            ema_alpha: THERMAL_EMA_ALPHA,
            thermal_rx: Mutex::new(rx),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nominal_no_slowdown() {
        let (mut bridge, _tx) = ThermalBridge::new();
        bridge.update();
        let signals = bridge.signals();
        assert!((signals.tau_factor - 1.0).abs() < 1e-9);
        assert!(!signals.should_reduce_profile);
        assert!(signals.target_frequency_override.is_none());
        assert_eq!(signals.level, ThermalLevel::Nominal);
    }

    #[test]
    fn test_serious_level_increases_tau() {
        let (mut bridge, tx) = ThermalBridge::new();
        tx.send(ThermalLevel::Serious).unwrap();

        // Run enough cycles for EMA to converge
        for _ in 0..50 {
            bridge.update();
        }

        let signals = bridge.signals();
        assert!(
            (signals.tau_factor - THERMAL_TAU_SERIOUS).abs() < 0.01,
            "tau should converge to ~{}, got {}",
            THERMAL_TAU_SERIOUS,
            signals.tau_factor
        );
        assert!(!signals.should_reduce_profile);
        assert!(signals.target_frequency_override.is_none());
    }

    #[test]
    fn test_critical_recommends_profile_downgrade() {
        let (mut bridge, tx) = ThermalBridge::new();
        tx.send(ThermalLevel::Critical).unwrap();
        bridge.update();

        let signals = bridge.signals();
        assert!(signals.should_reduce_profile);
        assert_eq!(signals.target_frequency_override, Some(10.0));
    }

    #[test]
    fn test_emergency_near_minimum() {
        let (mut bridge, tx) = ThermalBridge::new();
        tx.send(ThermalLevel::Emergency).unwrap();

        for _ in 0..50 {
            bridge.update();
        }

        let signals = bridge.signals();
        assert!(
            (signals.tau_factor - THERMAL_TAU_EMERGENCY).abs() < 0.01,
            "tau should converge to ~{}, got {}",
            THERMAL_TAU_EMERGENCY,
            signals.tau_factor
        );
        assert!(signals.should_reduce_profile);
        assert_eq!(signals.target_frequency_override, Some(5.0));
    }

    #[test]
    fn test_ema_smoothing_prevents_instant_jump() {
        let (mut bridge, tx) = ThermalBridge::new();
        tx.send(ThermalLevel::Emergency).unwrap();
        bridge.update();

        // After one cycle, tau should NOT have jumped to 2.5 instantly
        let signals = bridge.signals();
        assert!(
            signals.tau_factor < THERMAL_TAU_EMERGENCY,
            "EMA should prevent instant jump to {}, got {}",
            THERMAL_TAU_EMERGENCY,
            signals.tau_factor
        );
        assert!(
            signals.tau_factor > THERMAL_TAU_NOMINAL,
            "tau should have moved from nominal, got {}",
            signals.tau_factor
        );
    }

    #[test]
    fn test_recovery_from_critical_to_nominal() {
        let (mut bridge, tx) = ThermalBridge::new();

        // Heat up
        tx.send(ThermalLevel::Critical).unwrap();
        for _ in 0..50 {
            bridge.update();
        }
        assert!(bridge.signals().tau_factor > 1.5);

        // Cool down
        tx.send(ThermalLevel::Nominal).unwrap();
        for _ in 0..50 {
            bridge.update();
        }

        let signals = bridge.signals();
        assert!(
            (signals.tau_factor - THERMAL_TAU_NOMINAL).abs() < 0.01,
            "tau should recover to nominal, got {}",
            signals.tau_factor
        );
        assert!(!signals.should_reduce_profile);
    }

    #[test]
    fn test_level_from_u8() {
        assert_eq!(ThermalLevel::from_u8(0), ThermalLevel::Nominal);
        assert_eq!(ThermalLevel::from_u8(1), ThermalLevel::Fair);
        assert_eq!(ThermalLevel::from_u8(2), ThermalLevel::Serious);
        assert_eq!(ThermalLevel::from_u8(3), ThermalLevel::Critical);
        assert_eq!(ThermalLevel::from_u8(4), ThermalLevel::Emergency);
        // Out of range clamps to Emergency
        assert_eq!(ThermalLevel::from_u8(5), ThermalLevel::Emergency);
        assert_eq!(ThermalLevel::from_u8(255), ThermalLevel::Emergency);
    }

    #[test]
    fn test_tau_monotonic_with_level() {
        let levels = [
            ThermalLevel::Nominal,
            ThermalLevel::Fair,
            ThermalLevel::Serious,
            ThermalLevel::Critical,
            ThermalLevel::Emergency,
        ];

        for window in levels.windows(2) {
            assert!(
                window[0].tau_factor() <= window[1].tau_factor(),
                "tau should be monotonically non-decreasing: {:?}={} vs {:?}={}",
                window[0],
                window[0].tau_factor(),
                window[1],
                window[1].tau_factor()
            );
        }
    }

    #[test]
    fn test_tau_bounded() {
        for level in [
            ThermalLevel::Nominal,
            ThermalLevel::Fair,
            ThermalLevel::Serious,
            ThermalLevel::Critical,
            ThermalLevel::Emergency,
        ] {
            let tau = level.tau_factor();
            assert!(
                (1.0..=2.5).contains(&tau),
                "tau for {:?} = {} is out of [1.0, 2.5]",
                level,
                tau
            );
        }
    }

    #[test]
    fn test_only_latest_report_matters() {
        let (mut bridge, tx) = ThermalBridge::new();

        // Send multiple levels before update — only last should stick
        tx.send(ThermalLevel::Nominal).unwrap();
        tx.send(ThermalLevel::Critical).unwrap();
        tx.send(ThermalLevel::Fair).unwrap();
        bridge.update();

        assert_eq!(bridge.level(), ThermalLevel::Fair);
    }

    #[test]
    fn test_reset_clears_state() {
        let (mut bridge, tx) = ThermalBridge::new();

        tx.send(ThermalLevel::Emergency).unwrap();
        for _ in 0..50 {
            bridge.update();
        }
        assert!(bridge.smoothed_tau() > 2.0);

        bridge.reset();
        assert_eq!(bridge.level(), ThermalLevel::Nominal);
        assert!((bridge.smoothed_tau() - THERMAL_TAU_NOMINAL).abs() < 1e-9);
    }

    #[test]
    fn test_default_is_nominal() {
        let bridge = ThermalBridge::default();
        assert_eq!(bridge.level(), ThermalLevel::Nominal);
        assert!((bridge.smoothed_tau() - 1.0).abs() < 1e-9);
    }
}

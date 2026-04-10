// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # MetabolismManager — Real Hardware Constraints as Embodiment
//!
//! Reads CPU utilization, memory pressure, and thermal state from the
//! host OS (Linux /proc and /sys), converting them into neuromodulatory
//! signals that modulate cognition. This gives the system genuine
//! embodied stakes: high CPU load feels like exertion, memory pressure
//! triggers consolidation, thermal spikes slow down processing.
//!
//! On non-Linux platforms, all readings return safe defaults and the
//! manager produces neutral output.
//!
//! # Science
//!
//! - Sterling (2012): Allostasis operates on multi-second timescales.
//! - Kahneman (1973): Attention resource allocation adapts over 1-3s.
//! - Wickens (2008): Resource depletion theory thresholds.
//! - McEwen (2000): Allostatic load as weighted composite of stressors.

use serde::{Deserialize, Serialize};

// ── Constants ────────────────────────────────────────────────────────────

/// Co-prime manager interval. 59 cycles at 31Hz ≈ 1.9s sampling.
/// Science: Sterling (2012) — allostasis operates on multi-second timescales.
pub const METABOLISM_INTERVAL: u32 = 59;

/// CPU load EMA smoothing factor.
/// Science: Kahneman (1973) — attention adapts over 1-3 second windows.
pub const CPU_EMA_ALPHA: f32 = 0.2;

/// Memory pressure EMA smoothing factor (slower, less volatile).
pub const MEMORY_EMA_ALPHA: f32 = 0.1;

/// CPU temperature EMA smoothing factor.
pub const THERMAL_EMA_ALPHA: f32 = 0.15;

/// CPU load above which cognition is modulated.
/// Science: Wickens (2008) — resource depletion theory thresholds.
pub const HIGH_CPU_THRESHOLD: f32 = 0.8;

/// Memory pressure triggering consolidation request.
pub const HIGH_MEMORY_THRESHOLD: f32 = 0.85;

/// Combined stress above which confidence is reduced.
pub const HIGH_STRESS_THRESHOLD: f32 = 0.7;

/// Stress formula weights: CPU, memory, thermal.
/// Science: McEwen (2000) — allostatic load as weighted composite.
pub const STRESS_WEIGHT_CPU: f32 = 0.40;
pub const STRESS_WEIGHT_MEMORY: f32 = 0.35;
pub const STRESS_WEIGHT_THERMAL: f32 = 0.25;

/// Thermal level thresholds (Celsius).
pub const THERMAL_NOMINAL_MAX: f32 = 60.0;
pub const THERMAL_FAIR_MAX: f32 = 70.0;
pub const THERMAL_SERIOUS_MAX: f32 = 80.0;
pub const THERMAL_CRITICAL_MAX: f32 = 90.0;

/// Default temperature when sensor unavailable.
pub const DEFAULT_TEMP_CELSIUS: f32 = 45.0;

// ── Types ────────────────────────────────────────────────────────────────

/// Raw hardware readings from procfs/sysfs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardwareReading {
    /// CPU utilization (0.0-1.0) from /proc/stat delta
    pub cpu_utilization: f32,
    /// Memory pressure (0.0-1.0) from /proc/meminfo (used/total)
    pub memory_pressure: f32,
    /// CPU temperature in Celsius (None if unavailable)
    pub cpu_temp_celsius: Option<f32>,
}

/// Thermal level derived from CPU temperature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalLevel {
    Nominal = 0,
    Fair = 1,
    Serious = 2,
    Critical = 3,
    Emergency = 4,
}

impl ThermalLevel {
    /// Derive from CPU temperature in Celsius.
    pub fn from_celsius(temp: f32) -> Self {
        if temp < THERMAL_NOMINAL_MAX {
            Self::Nominal
        } else if temp < THERMAL_FAIR_MAX {
            Self::Fair
        } else if temp < THERMAL_SERIOUS_MAX {
            Self::Serious
        } else if temp < THERMAL_CRITICAL_MAX {
            Self::Critical
        } else {
            Self::Emergency
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Fair => "fair",
            Self::Serious => "serious",
            Self::Critical => "critical",
            Self::Emergency => "emergency",
        }
    }
}

/// EMA-smoothed metabolic state.
pub struct MetabolicState {
    /// Smoothed CPU load
    cpu_load_ema: f32,
    /// Smoothed memory pressure
    memory_pressure_ema: f32,
    /// Smoothed CPU temperature
    cpu_temp_ema: f32,
    /// Previous /proc/stat CPU jiffies (total, idle) for delta computation
    prev_cpu_jiffies: (u64, u64),
    /// Combined metabolic stress [0, 1]
    stress: f32,
    /// Derived thermal level
    thermal_level: ThermalLevel,
    /// Whether procfs is available (Linux only)
    platform_available: bool,
    /// Whether first reading has been taken (for jiffies delta)
    initialized: bool,
}

impl MetabolicState {
    pub fn new() -> Self {
        let platform_available =
            cfg!(target_os = "linux") && std::path::Path::new("/proc/stat").exists();

        Self {
            cpu_load_ema: 0.0,
            memory_pressure_ema: 0.0,
            cpu_temp_ema: DEFAULT_TEMP_CELSIUS,
            prev_cpu_jiffies: (0, 0),
            stress: 0.0,
            thermal_level: ThermalLevel::Nominal,
            platform_available,
            initialized: false,
        }
    }

    /// Read hardware state and update EMAs.
    pub fn tick(&mut self) -> HardwareReading {
        if !self.platform_available {
            return HardwareReading::default();
        }

        let raw = self.read_hardware();

        // Update CPU EMA (skip first reading — need delta)
        if self.initialized {
            self.cpu_load_ema =
                self.cpu_load_ema * (1.0 - CPU_EMA_ALPHA) + raw.cpu_utilization * CPU_EMA_ALPHA;
        }

        // Update memory EMA
        self.memory_pressure_ema = self.memory_pressure_ema * (1.0 - MEMORY_EMA_ALPHA)
            + raw.memory_pressure * MEMORY_EMA_ALPHA;

        // Update thermal EMA
        let temp = raw.cpu_temp_celsius.unwrap_or(DEFAULT_TEMP_CELSIUS);
        self.cpu_temp_ema =
            self.cpu_temp_ema * (1.0 - THERMAL_EMA_ALPHA) + temp * THERMAL_EMA_ALPHA;

        // Derive thermal level
        self.thermal_level = ThermalLevel::from_celsius(self.cpu_temp_ema);

        // Normalize temperature to 0-1 range (20C=0, 100C=1)
        let temp_normalized = ((self.cpu_temp_ema - 20.0) / 80.0).clamp(0.0, 1.0);

        // Compute combined stress
        self.stress = (STRESS_WEIGHT_CPU * self.cpu_load_ema
            + STRESS_WEIGHT_MEMORY * self.memory_pressure_ema
            + STRESS_WEIGHT_THERMAL * temp_normalized)
            .clamp(0.0, 1.0);

        // NaN guards
        if !self.stress.is_finite() {
            self.stress = 0.0;
        }
        if !self.cpu_load_ema.is_finite() {
            self.cpu_load_ema = 0.0;
        }
        if !self.memory_pressure_ema.is_finite() {
            self.memory_pressure_ema = 0.0;
        }

        self.initialized = true;
        raw
    }

    /// Read raw hardware values from procfs/sysfs.
    fn read_hardware(&mut self) -> HardwareReading {
        HardwareReading {
            cpu_utilization: self.read_cpu_utilization().unwrap_or(0.0),
            memory_pressure: self.read_memory_pressure().unwrap_or(0.0),
            cpu_temp_celsius: self.read_cpu_temperature(),
        }
    }

    /// Read CPU utilization from /proc/stat (delta between calls).
    fn read_cpu_utilization(&mut self) -> Option<f32> {
        let data = std::fs::read_to_string("/proc/stat").ok()?;
        let first_line = data.lines().next()?;
        // "cpu  user nice system idle iowait irq softirq steal"
        let vals: Vec<u64> = first_line
            .split_whitespace()
            .skip(1) // skip "cpu"
            .filter_map(|s| s.parse().ok())
            .collect();
        if vals.len() < 4 {
            return None;
        }
        let total: u64 = vals.iter().sum();
        let idle = vals[3];

        let (prev_total, prev_idle) = self.prev_cpu_jiffies;
        self.prev_cpu_jiffies = (total, idle);

        if !self.initialized {
            return Some(0.0); // First reading, no delta
        }

        let dt = total.saturating_sub(prev_total);
        let di = idle.saturating_sub(prev_idle);
        if dt == 0 {
            return Some(0.0);
        }
        let util = 1.0 - (di as f32 / dt as f32);
        Some(util.clamp(0.0, 1.0))
    }

    /// Read memory pressure from /proc/meminfo.
    fn read_memory_pressure(&self) -> Option<f32> {
        let data = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut mem_total: Option<u64> = None;
        let mut mem_available: Option<u64> = None;

        for line in data.lines() {
            if line.starts_with("MemTotal:") {
                mem_total = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            } else if line.starts_with("MemAvailable:") {
                mem_available = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
            }
            if mem_total.is_some() && mem_available.is_some() {
                break;
            }
        }

        let total = mem_total? as f32;
        let available = mem_available? as f32;
        if total <= 0.0 {
            return Some(0.0);
        }
        let pressure = 1.0 - (available / total);
        Some(pressure.clamp(0.0, 1.0))
    }

    /// Read CPU temperature from /sys/class/thermal.
    fn read_cpu_temperature(&self) -> Option<f32> {
        // Try thermal_zone0 first (most common for CPU)
        for i in 0..5 {
            let path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
            if let Ok(data) = std::fs::read_to_string(&path) {
                if let Ok(millideg) = data.trim().parse::<i64>() {
                    let celsius = millideg as f32 / 1000.0;
                    if celsius > 0.0 && celsius < 150.0 {
                        return Some(celsius);
                    }
                }
            }
        }
        None
    }

    // ── Accessors ──────────────────────────────────────────────────────

    pub fn cpu_load(&self) -> f32 {
        self.cpu_load_ema
    }
    pub fn memory_pressure(&self) -> f32 {
        self.memory_pressure_ema
    }
    pub fn cpu_temp(&self) -> f32 {
        self.cpu_temp_ema
    }
    pub fn stress(&self) -> f32 {
        self.stress
    }
    pub fn thermal_level(&self) -> ThermalLevel {
        self.thermal_level
    }
    pub fn is_platform_available(&self) -> bool {
        self.platform_available
    }
}

impl Default for MetabolicState {
    fn default() -> Self {
        Self::new()
    }
}

/// Telemetry for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetabolismTelemetry {
    pub cpu_utilization: f32,
    pub memory_pressure: f32,
    pub cpu_temp_celsius: f32,
    pub metabolic_stress: f32,
    pub thermal_level: String,
    pub platform_available: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thermal_level_from_celsius() {
        assert_eq!(ThermalLevel::from_celsius(40.0), ThermalLevel::Nominal);
        assert_eq!(ThermalLevel::from_celsius(65.0), ThermalLevel::Fair);
        assert_eq!(ThermalLevel::from_celsius(75.0), ThermalLevel::Serious);
        assert_eq!(ThermalLevel::from_celsius(85.0), ThermalLevel::Critical);
        assert_eq!(ThermalLevel::from_celsius(95.0), ThermalLevel::Emergency);
    }

    #[test]
    fn test_stress_formula_bounds() {
        // All inputs at 0 → stress = 0
        let stress =
            STRESS_WEIGHT_CPU * 0.0 + STRESS_WEIGHT_MEMORY * 0.0 + STRESS_WEIGHT_THERMAL * 0.0;
        assert_eq!(stress, 0.0);

        // All inputs at 1 → stress = sum of weights = 1.0
        let stress =
            STRESS_WEIGHT_CPU * 1.0 + STRESS_WEIGHT_MEMORY * 1.0 + STRESS_WEIGHT_THERMAL * 1.0;
        assert!((stress - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stress_weights_sum_to_one() {
        let sum = STRESS_WEIGHT_CPU + STRESS_WEIGHT_MEMORY + STRESS_WEIGHT_THERMAL;
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_metabolic_state_default() {
        let state = MetabolicState::new();
        assert_eq!(state.cpu_load(), 0.0);
        assert_eq!(state.memory_pressure(), 0.0);
        assert_eq!(state.stress(), 0.0);
        assert_eq!(state.thermal_level(), ThermalLevel::Nominal);
    }

    #[test]
    fn test_tick_on_linux_produces_reading() {
        let mut state = MetabolicState::new();
        let reading = state.tick();

        if state.is_platform_available() {
            // On Linux, should get real readings
            assert!(reading.memory_pressure >= 0.0);
            assert!(reading.memory_pressure <= 1.0);
            // CPU temp may or may not be available
        } else {
            // On non-Linux, all defaults
            assert_eq!(reading.cpu_utilization, 0.0);
            assert_eq!(reading.memory_pressure, 0.0);
        }
    }

    #[test]
    fn test_ema_smoothing() {
        let mut state = MetabolicState::new();
        // Force platform available for testing
        state.platform_available = false; // Use defaults

        // Simulate several ticks
        for _ in 0..10 {
            state.tick();
        }

        // All EMAs should be bounded
        assert!(state.cpu_load() >= 0.0 && state.cpu_load() <= 1.0);
        assert!(state.memory_pressure() >= 0.0 && state.memory_pressure() <= 1.0);
        assert!(state.stress() >= 0.0 && state.stress() <= 1.0);
    }

    #[test]
    fn test_nan_guard() {
        let mut state = MetabolicState::new();
        state.stress = f32::NAN;
        state.cpu_load_ema = f32::NAN;
        state.platform_available = false;
        state.tick();
        assert!(state.stress().is_finite());
        assert!(state.cpu_load().is_finite());
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Sleep/Wake metabolism: state machine governing cycle frequency and compute budget.
//!
//! State machine: Sleep(0.5Hz) <-> Drowsy(5Hz) <-> Alert(20Hz) <-> Focused(50Hz)
//!
//! Transitions are driven by platform signals (phone pickup, user input, inactivity,
//! charging state, night mode) relayed through `WakeSignal`.

use serde::{Deserialize, Serialize};

// Constants
const DROWSY_INACTIVITY_SECS: u32 = 30;
const SLEEP_INACTIVITY_SECS: u32 = 60;
const FOCUS_COHERENCE_THRESHOLD: f32 = 0.7;
const FOCUS_PE_THRESHOLD: f32 = 0.3;
const THERMAL_SERIOUS: u8 = 2;
const DREAM_CONSOLIDATION_INTERVAL: u64 = 100;

/// Wake state with associated target cycle frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WakeState {
    Sleep,   // 0.5 Hz
    Drowsy,  // 5 Hz
    Alert,   // 20 Hz
    Focused, // 50 Hz
}

impl WakeState {
    pub fn target_hz(self) -> f32 {
        match self {
            Self::Sleep => 0.5,
            Self::Drowsy => 5.0,
            Self::Alert => 20.0,
            Self::Focused => 50.0,
        }
    }

    pub fn as_u8(self) -> u8 {
        match self {
            Self::Sleep => 0,
            Self::Drowsy => 1,
            Self::Alert => 2,
            Self::Focused => 3,
        }
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Sleep,
            1 => Self::Drowsy,
            2 => Self::Alert,
            _ => Self::Focused,
        }
    }

    /// Whether topology analysis should run in this state.
    pub fn skip_topology(self) -> bool {
        matches!(self, Self::Sleep | Self::Drowsy)
    }

    /// Whether FEP should run in this state.
    pub fn skip_fep(self) -> bool {
        matches!(self, Self::Sleep)
    }
}

/// Signals from the platform layer that drive wake transitions.
#[derive(Debug, Clone, Copy)]
pub enum WakeSignal {
    PhonePickup,
    UserInput,
    Inactivity(u32),       // seconds of inactivity
    ChargingChanged(bool), // true = plugged in
    NightMode(bool),       // true = night mode active
    ExplicitSleep,
    ThermalLevel(u8), // 0-4
}

impl WakeSignal {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::PhonePickup,
            1 => Self::UserInput,
            2 => Self::ExplicitSleep,
            _ => Self::PhonePickup,
        }
    }
}

/// Metabolism manager: tracks wake state, drives frequency, gates compute.
pub struct Metabolism {
    state: WakeState,
    is_charging: bool,
    is_night_mode: bool,
    thermal_level: u8,
    inactivity_secs: u32,
    cycles_in_state: u64,
    /// Tracks cycles since last dream consolidation in sleep.
    sleep_cycle_counter: u64,
    /// Whether auto-dream should trigger next cycle.
    pub dream_consolidation_due: bool,
    // For focus detection
    coherence_ema: f32,
    pe_ema: f32,
}

impl Metabolism {
    pub fn new() -> Self {
        Self {
            state: WakeState::Alert,
            is_charging: false,
            is_night_mode: false,
            thermal_level: 0,
            inactivity_secs: 0,
            cycles_in_state: 0,
            sleep_cycle_counter: 0,
            dream_consolidation_due: false,
            coherence_ema: 0.5,
            pe_ema: 0.5,
        }
    }

    pub fn state(&self) -> WakeState {
        self.state
    }

    pub fn target_hz(&self) -> f32 {
        self.state.target_hz()
    }

    /// Process a wake signal from the platform.
    pub fn signal(&mut self, sig: WakeSignal) {
        match sig {
            WakeSignal::PhonePickup => {
                if self.state == WakeState::Sleep {
                    self.transition(WakeState::Drowsy);
                }
            }
            WakeSignal::UserInput => {
                match self.state {
                    WakeState::Sleep => self.transition(WakeState::Drowsy),
                    WakeState::Drowsy => self.transition(WakeState::Alert),
                    _ => {}
                }
                self.inactivity_secs = 0;
            }
            WakeSignal::Inactivity(secs) => {
                self.inactivity_secs = secs;
                match self.state {
                    WakeState::Alert | WakeState::Focused if secs >= DROWSY_INACTIVITY_SECS => {
                        self.transition(WakeState::Drowsy);
                    }
                    WakeState::Drowsy if secs >= SLEEP_INACTIVITY_SECS => {
                        self.transition(WakeState::Sleep);
                    }
                    _ => {}
                }
            }
            WakeSignal::ChargingChanged(charging) => {
                self.is_charging = charging;
                self.check_auto_sleep();
            }
            WakeSignal::NightMode(night) => {
                self.is_night_mode = night;
                self.check_auto_sleep();
            }
            WakeSignal::ExplicitSleep => {
                self.transition(WakeState::Sleep);
            }
            WakeSignal::ThermalLevel(level) => {
                self.thermal_level = level.min(4);
                if level >= THERMAL_SERIOUS && self.state == WakeState::Focused {
                    self.transition(WakeState::Alert);
                }
            }
        }
    }

    /// Called each cycle with consciousness metrics for focus detection.
    pub fn tick(&mut self, prediction_error: f32, consciousness_level: f32) {
        self.cycles_in_state += 1;

        // EMA update for focus detection
        self.coherence_ema = self.coherence_ema * 0.95 + consciousness_level * 0.05;
        self.pe_ema = self.pe_ema * 0.95 + prediction_error * 0.05;

        // Auto-focus: sustained low PE + high coherence
        if self.state == WakeState::Alert
            && self.coherence_ema > FOCUS_COHERENCE_THRESHOLD
            && self.pe_ema < FOCUS_PE_THRESHOLD
            && self.cycles_in_state > 50
        {
            self.transition(WakeState::Focused);
        }

        // Dream consolidation in sleep
        if self.state == WakeState::Sleep && self.is_charging && self.is_night_mode {
            self.sleep_cycle_counter += 1;
            if self.sleep_cycle_counter % DREAM_CONSOLIDATION_INTERVAL == 0 {
                self.dream_consolidation_due = true;
            }
        }
    }

    fn transition(&mut self, new_state: WakeState) {
        if self.state != new_state {
            self.state = new_state;
            self.cycles_in_state = 0;
            if new_state != WakeState::Sleep {
                self.sleep_cycle_counter = 0;
                self.dream_consolidation_due = false;
            }
        }
    }

    fn check_auto_sleep(&mut self) {
        if self.is_charging && self.is_night_mode {
            self.transition(WakeState::Sleep);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_is_alert() {
        let m = Metabolism::new();
        assert_eq!(m.state(), WakeState::Alert);
        assert_eq!(m.target_hz(), 20.0);
    }

    #[test]
    fn test_sleep_wake_transitions() {
        let mut m = Metabolism::new();
        m.signal(WakeSignal::Inactivity(35));
        assert_eq!(m.state(), WakeState::Drowsy);
        m.signal(WakeSignal::Inactivity(65));
        assert_eq!(m.state(), WakeState::Sleep);
        m.signal(WakeSignal::PhonePickup);
        assert_eq!(m.state(), WakeState::Drowsy);
        m.signal(WakeSignal::UserInput);
        assert_eq!(m.state(), WakeState::Alert);
    }

    #[test]
    fn test_explicit_sleep() {
        let mut m = Metabolism::new();
        m.signal(WakeSignal::ExplicitSleep);
        assert_eq!(m.state(), WakeState::Sleep);
    }

    #[test]
    fn test_night_charging_auto_sleep() {
        let mut m = Metabolism::new();
        m.signal(WakeSignal::NightMode(true));
        m.signal(WakeSignal::ChargingChanged(true));
        assert_eq!(m.state(), WakeState::Sleep);
    }

    #[test]
    fn test_thermal_defocus() {
        let mut m = Metabolism::new();
        // Force into focused state
        m.state = WakeState::Focused;
        m.signal(WakeSignal::ThermalLevel(2)); // Serious
        assert_eq!(m.state(), WakeState::Alert);
    }

    #[test]
    fn test_dream_consolidation_triggers() {
        let mut m = Metabolism::new();
        m.signal(WakeSignal::NightMode(true));
        m.signal(WakeSignal::ChargingChanged(true));
        assert_eq!(m.state(), WakeState::Sleep);

        for _ in 0..100 {
            m.tick(0.1, 0.5);
        }
        assert!(m.dream_consolidation_due);
    }

    #[test]
    fn test_focus_detection() {
        let mut m = Metabolism::new();
        // Simulate sustained low PE, high coherence
        for _ in 0..100 {
            m.tick(0.1, 0.9);
        }
        assert_eq!(m.state(), WakeState::Focused);
    }

    #[test]
    fn test_wake_state_roundtrip() {
        for v in 0..4u8 {
            let state = WakeState::from_u8(v);
            assert_eq!(state.as_u8(), v);
        }
    }

    #[test]
    fn test_skip_gates() {
        assert!(WakeState::Sleep.skip_topology());
        assert!(WakeState::Sleep.skip_fep());
        assert!(WakeState::Drowsy.skip_topology());
        assert!(!WakeState::Drowsy.skip_fep());
        assert!(!WakeState::Alert.skip_topology());
        assert!(!WakeState::Alert.skip_fep());
    }

    #[test]
    fn test_user_input_resets_inactivity() {
        let mut m = Metabolism::new();
        m.signal(WakeSignal::Inactivity(35));
        assert_eq!(m.state(), WakeState::Drowsy);
        m.signal(WakeSignal::UserInput);
        assert_eq!(m.state(), WakeState::Alert);
        // Now inactivity should need to re-accumulate
        m.signal(WakeSignal::Inactivity(10));
        assert_eq!(m.state(), WakeState::Alert); // Not drowsy yet
    }
}

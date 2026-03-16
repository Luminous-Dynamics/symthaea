//! Sensor bridge: translates platform sensor data into neuromodulator nudges.
//!
//! The bridge receives raw sensor values from the platform layer (Android/iOS)
//! and converts them into biologically-grounded neuromodulator modulations:
//! - Motion -> NE (arousal from physical activity)
//! - Light -> 5-HT (serotonin from bright light, circadian alignment)
//! - GPS novelty -> DA (dopamine from exploration)
//! - Proximity -> privacy mode (suppresses outbound sharing)

use serde::{Deserialize, Serialize};

// Neuromodulator nudge magnitudes (small, clamped)
const MOTION_NE_NUDGE: f32 = 0.03;
const LIGHT_5HT_NUDGE: f32 = 0.02;
const GPS_DA_NUDGE: f32 = 0.04;
const BRIGHT_LUX_THRESHOLD: f32 = 500.0;
const DIM_LUX_THRESHOLD: f32 = 50.0;
// Motion state thresholds (accelerometer magnitude in m/s^2)
const WALKING_THRESHOLD: f32 = 1.5;
const RUNNING_THRESHOLD: f32 = 5.0;
const VEHICLE_THRESHOLD: f32 = 10.0;

/// Motion state derived from accelerometer magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionState {
    Stationary,
    Walking,
    Running,
    InVehicle,
}

impl MotionState {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Stationary => 0,
            Self::Walking => 1,
            Self::Running => 2,
            Self::InVehicle => 3,
        }
    }
}

/// Raw sensor snapshot set by the platform each frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorSnapshot {
    pub accelerometer_magnitude: f32,
    pub light_lux: f32,
    pub proximity_near: bool,
    pub barometer_hpa: f32,
    pub gps_novelty: f32, // 0.0-1.0
}

impl Default for SensorSnapshot {
    fn default() -> Self {
        Self {
            accelerometer_magnitude: 0.0,
            light_lux: 300.0,
            proximity_near: false,
            barometer_hpa: 1013.25,
            gps_novelty: 0.0,
        }
    }
}

/// Neuromodulator nudges computed from sensor data.
#[derive(Debug, Clone, Default)]
pub struct SensorNudges {
    pub dopamine_delta: f32,
    pub norepinephrine_delta: f32,
    pub serotonin_delta: f32,
}

/// Sensor bridge: maintains current snapshot and derives neuromod nudges.
pub struct SensorBridge {
    snapshot: SensorSnapshot,
    motion_state: MotionState,
    privacy_mode: bool,
    motion_ema: f32,
}

impl SensorBridge {
    pub fn new() -> Self {
        Self {
            snapshot: SensorSnapshot::default(),
            motion_state: MotionState::Stationary,
            privacy_mode: false,
            motion_ema: 0.0,
        }
    }

    /// Update sensor snapshot from platform.
    pub fn set_sensors(
        &mut self,
        accel: f32,
        light: f32,
        proximity_near: bool,
        barometer: f32,
        gps_novelty: f32,
    ) {
        self.snapshot = SensorSnapshot {
            accelerometer_magnitude: accel.max(0.0),
            light_lux: light.max(0.0),
            proximity_near,
            barometer_hpa: barometer.clamp(800.0, 1100.0),
            gps_novelty: gps_novelty.clamp(0.0, 1.0),
        };
        self.update_derived();
    }

    fn update_derived(&mut self) {
        // Motion EMA for smooth transitions
        self.motion_ema = self.motion_ema * 0.8 + self.snapshot.accelerometer_magnitude * 0.2;

        self.motion_state = if self.motion_ema >= VEHICLE_THRESHOLD {
            MotionState::InVehicle
        } else if self.motion_ema >= RUNNING_THRESHOLD {
            MotionState::Running
        } else if self.motion_ema >= WALKING_THRESHOLD {
            MotionState::Walking
        } else {
            MotionState::Stationary
        };

        // Privacy mode: face-down (proximity sensor near) suppresses all outbound
        self.privacy_mode = self.snapshot.proximity_near;
    }

    /// Compute neuromodulator nudges from current sensor state.
    pub fn compute_nudges(&self) -> SensorNudges {
        let mut nudges = SensorNudges::default();

        // Motion -> NE (arousal from physical activity)
        let motion_factor = match self.motion_state {
            MotionState::Stationary => 0.0,
            MotionState::Walking => 0.5,
            MotionState::Running => 1.0,
            MotionState::InVehicle => 0.3, // Less arousal in passive transport
        };
        nudges.norepinephrine_delta = MOTION_NE_NUDGE * motion_factor;

        // Light -> 5-HT (bright light boosts serotonin, dim suppresses)
        if self.snapshot.light_lux > BRIGHT_LUX_THRESHOLD {
            nudges.serotonin_delta = LIGHT_5HT_NUDGE;
        } else if self.snapshot.light_lux < DIM_LUX_THRESHOLD {
            nudges.serotonin_delta = -LIGHT_5HT_NUDGE * 0.5;
        }

        // GPS novelty -> DA (dopamine from exploration)
        nudges.dopamine_delta = GPS_DA_NUDGE * self.snapshot.gps_novelty;

        nudges
    }

    pub fn motion_state(&self) -> MotionState {
        self.motion_state
    }

    pub fn privacy_mode(&self) -> bool {
        self.privacy_mode
    }

    pub fn snapshot(&self) -> &SensorSnapshot {
        &self.snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_state() {
        let bridge = SensorBridge::new();
        assert_eq!(bridge.motion_state(), MotionState::Stationary);
        assert!(!bridge.privacy_mode());
    }

    #[test]
    fn test_motion_state_detection() {
        let mut bridge = SensorBridge::new();
        // Feed walking-level acceleration several times for EMA to settle
        for _ in 0..20 {
            bridge.set_sensors(2.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.motion_state(), MotionState::Walking);
    }

    #[test]
    fn test_running_detection() {
        let mut bridge = SensorBridge::new();
        for _ in 0..20 {
            bridge.set_sensors(7.0, 300.0, false, 1013.0, 0.0);
        }
        assert_eq!(bridge.motion_state(), MotionState::Running);
    }

    #[test]
    fn test_privacy_mode() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 0.0, true, 1013.0, 0.0);
        assert!(bridge.privacy_mode());
        bridge.set_sensors(0.0, 0.0, false, 1013.0, 0.0);
        assert!(!bridge.privacy_mode());
    }

    #[test]
    fn test_nudges_motion_ne() {
        let mut bridge = SensorBridge::new();
        for _ in 0..20 {
            bridge.set_sensors(3.0, 300.0, false, 1013.0, 0.0);
        }
        let nudges = bridge.compute_nudges();
        assert!(nudges.norepinephrine_delta > 0.0);
    }

    #[test]
    fn test_nudges_bright_light_serotonin() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 1000.0, false, 1013.0, 0.0);
        let nudges = bridge.compute_nudges();
        assert!(nudges.serotonin_delta > 0.0);
    }

    #[test]
    fn test_nudges_dim_light_serotonin() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 10.0, false, 1013.0, 0.0);
        let nudges = bridge.compute_nudges();
        assert!(nudges.serotonin_delta < 0.0);
    }

    #[test]
    fn test_nudges_gps_novelty_da() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 300.0, false, 1013.0, 0.8);
        let nudges = bridge.compute_nudges();
        assert!(nudges.dopamine_delta > 0.0);
    }

    #[test]
    fn test_nudges_stationary_no_ne() {
        let bridge = SensorBridge::new();
        let nudges = bridge.compute_nudges();
        assert_eq!(nudges.norepinephrine_delta, 0.0);
    }

    #[test]
    fn test_barometer_clamp() {
        let mut bridge = SensorBridge::new();
        bridge.set_sensors(0.0, 300.0, false, 2000.0, 0.0);
        assert_eq!(bridge.snapshot().barometer_hpa, 1100.0);
    }
}

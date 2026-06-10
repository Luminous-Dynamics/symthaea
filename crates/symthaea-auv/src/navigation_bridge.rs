// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Navigation bridge — connects AUV state estimation to the cognitive loop.
//!
//! Simplified: produces measurements from DVL velocity + optional sonar fixes.
//! A full DVL/USBL/INS sensor fusion is future work.

use crate::types::AuvState;
use positioning::{
    Measurement, MeasurementModality, MeasurementProvenance, MeasurementValue, ReferenceFrame,
};

/// A single navigation sample (one tick of sensor data).
#[derive(Debug, Clone, Copy, Default)]
pub struct UnderwaterNavigationSample {
    /// DVL-measured body-frame velocity (if available).
    pub dvl_velocity_body_mps: Option<[f64; 3]>,
    /// Sonar-based relative position fix (if available).
    pub sonar_relative_fix_m: Option<[f64; 3]>,
}

/// Navigation bridge that derives measurements from AUV state + sensor samples.
#[derive(Debug, Clone)]
pub struct AuvNavigationBridge {
    agent_id: String,
    blackout_depth_threshold: f64,
    blackout: bool,
}

impl AuvNavigationBridge {
    /// Create a new navigation bridge.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            blackout_depth_threshold: -300.0,
            blackout: false,
        }
    }

    /// Update blackout status based on depth.
    pub fn update_blackout(&mut self, depth: f64) {
        self.blackout = depth < self.blackout_depth_threshold;
    }

    /// Whether the bridge is in blackout (deep, no GPS/sonar fixes).
    pub fn is_blackout(&self) -> bool {
        self.blackout
    }

    /// Derive measurements from AUV state and a sample.
    pub fn measurements_from_state(
        &self,
        state: &AuvState,
        timestamp: f64,
        sample: &UnderwaterNavigationSample,
    ) -> Vec<Measurement> {
        if self.blackout {
            return Vec::new();
        }
        let mut xyz = [state.position[0], state.position[1], state.depth];
        if let Some(fix) = sample.sonar_relative_fix_m {
            xyz[0] += fix[0] * 0.3;
            xyz[1] += fix[1] * 0.3;
        }
        vec![Measurement {
            modality: MeasurementModality::Gps,
            provenance: MeasurementProvenance::Local,
            frame: ReferenceFrame::Enu,
            value: MeasurementValue::Position {
                xyz,
                sigma: [1.0, 1.0, 1.0],
            },
            timestamp_us: (timestamp * 1_000_000.0) as u64,
            source_id: self.agent_id.clone(),
        }]
    }

    /// Agent identifier.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = AuvNavigationBridge::new("test_auv");
        assert_eq!(bridge.agent_id(), "test_auv");
        assert!(!bridge.is_blackout());
    }

    #[test]
    fn test_blackout_threshold() {
        let mut bridge = AuvNavigationBridge::new("test_auv");
        bridge.update_blackout(-100.0);
        assert!(!bridge.is_blackout());
        bridge.update_blackout(-400.0);
        assert!(bridge.is_blackout());
    }
}

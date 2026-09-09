// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};

/// Broad platform class. Kept intentionally mission-neutral.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaritimePlatformKind {
    AutonomousUnderwaterVehicle,
    UncrewedSurfaceVessel,
    CrewedVessel,
    RemoteSensor,
    LogisticsCraft,
    Other,
}

/// Navigation source quality without assuming any single positioning system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NavigationQuality {
    Nominal,
    Degraded,
    DeadReckoning,
    Unavailable,
}

/// Minimal shared kinematic and energy state for maritime platforms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaritimeState {
    pub platform_id: String,
    pub kind: MaritimePlatformKind,
    pub monotonic_time_ms: u64,
    pub latitude_deg: Option<f64>,
    pub longitude_deg: Option<f64>,
    pub depth_m: Option<f32>,
    pub speed_mps: f32,
    pub heading_deg: f32,
    pub energy_remaining_fraction: f32,
    pub navigation_quality: NavigationQuality,
}

impl MaritimeState {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.platform_id.trim().is_empty() {
            return Err("platform_id must not be empty");
        }
        if !(0.0..=1.0).contains(&self.energy_remaining_fraction) {
            return Err("energy_remaining_fraction must be within [0, 1]");
        }
        if !(0.0..360.0).contains(&self.heading_deg) {
            return Err("heading_deg must be within [0, 360)");
        }
        if let Some(lat) = self.latitude_deg {
            if !(-90.0..=90.0).contains(&lat) {
                return Err("latitude_deg must be within [-90, 90]");
            }
        }
        if let Some(lon) = self.longitude_deg {
            if !(-180.0..=180.0).contains(&lon) {
                return Err("longitude_deg must be within [-180, 180]");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_minimal_state() {
        let state = MaritimeState {
            platform_id: "auv-01".into(),
            kind: MaritimePlatformKind::AutonomousUnderwaterVehicle,
            monotonic_time_ms: 42,
            latitude_deg: Some(-33.9),
            longitude_deg: Some(18.4),
            depth_m: Some(50.0),
            speed_mps: 1.5,
            heading_deg: 270.0,
            energy_remaining_fraction: 0.7,
            navigation_quality: NavigationQuality::Nominal,
        };
        assert_eq!(state.validate(), Ok(()));
    }

    #[test]
    fn rejects_invalid_energy_fraction() {
        let state = MaritimeState {
            platform_id: "auv-01".into(),
            kind: MaritimePlatformKind::AutonomousUnderwaterVehicle,
            monotonic_time_ms: 0,
            latitude_deg: None,
            longitude_deg: None,
            depth_m: None,
            speed_mps: 0.0,
            heading_deg: 0.0,
            energy_remaining_fraction: 1.1,
            navigation_quality: NavigationQuality::Unavailable,
        };
        assert!(state.validate().is_err());
    }
}

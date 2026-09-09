// SPDX-License-Identifier: AGPL-3.0-or-later
use serde::{Deserialize, Serialize};
use symthaea_maritime_core::NavigationQuality;

/// Minimal surface-vessel state owned by the USV domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsvState {
    /// Local tangent-plane position [east, north] in metres.
    pub local_position_m: [f64; 2],
    /// Heading in degrees, [0, 360).
    pub heading_deg: f32,
    /// Body-frame surge speed in m/s.
    pub surge_mps: f32,
    /// Body-frame sway speed in m/s.
    pub sway_mps: f32,
    /// Yaw rate in rad/s.
    pub yaw_rate_rad_s: f32,
    /// Remaining usable energy fraction, [0, 1].
    pub energy_remaining_fraction: f32,
}

impl UsvState {
    pub fn speed_mps(&self) -> f32 {
        self.surge_mps.hypot(self.sway_mps)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.local_position_m.iter().all(|v| v.is_finite()) {
            return Err("local_position_m must be finite");
        }
        if !self.heading_deg.is_finite() || !(0.0..360.0).contains(&self.heading_deg) {
            return Err("heading_deg must be finite and within [0, 360)");
        }
        if !self.surge_mps.is_finite()
            || !self.sway_mps.is_finite()
            || !self.yaw_rate_rad_s.is_finite()
        {
            return Err("USV velocity state must be finite");
        }
        if !self.energy_remaining_fraction.is_finite()
            || !(0.0..=1.0).contains(&self.energy_remaining_fraction)
        {
            return Err("energy_remaining_fraction must be finite and within [0, 1]");
        }
        Ok(())
    }
}

/// Optional geodetic navigation observation supplied by an external positioning stack.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UsvNavigationFix {
    pub latitude_deg: f64,
    pub longitude_deg: f64,
    pub quality: NavigationQuality,
}

impl UsvNavigationFix {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.latitude_deg.is_finite() || !(-90.0..=90.0).contains(&self.latitude_deg) {
            return Err("latitude_deg must be finite and within [-90, 90]");
        }
        if !self.longitude_deg.is_finite() || !(-180.0..=180.0).contains(&self.longitude_deg) {
            return Err("longitude_deg must be finite and within [-180, 180]");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_finite_surface_state() {
        let state = UsvState {
            local_position_m: [12.0, -4.0],
            heading_deg: 90.0,
            surge_mps: 3.0,
            sway_mps: 0.2,
            yaw_rate_rad_s: 0.01,
            energy_remaining_fraction: 0.8,
        };
        assert_eq!(state.validate(), Ok(()));
        assert!(state.speed_mps() > 3.0);
    }

    #[test]
    fn rejects_nonfinite_motion() {
        let state = UsvState {
            local_position_m: [0.0, 0.0],
            heading_deg: 0.0,
            surge_mps: f32::NAN,
            sway_mps: 0.0,
            yaw_rate_rad_s: 0.0,
            energy_remaining_fraction: 1.0,
        };
        assert!(state.validate().is_err());
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::{UsvNavigationFix, UsvState};
use symthaea_maritime_core::{MaritimePlatformKind, MaritimeState, NavigationQuality};

/// Project validated USV-local state into the shared maritime substrate.
///
/// Domain validation is part of the adapter boundary rather than an optional caller
/// precondition. Invalid local motion, invalid geodetic fixes, and invalid derived
/// maritime state fail closed before crossing into `symthaea-maritime-core`.
pub fn to_maritime_state(
    platform_id: impl Into<String>,
    monotonic_time_ms: u64,
    state: &UsvState,
    navigation_fix: Option<UsvNavigationFix>,
) -> Result<MaritimeState, &'static str> {
    state.validate()?;

    let (latitude_deg, longitude_deg, navigation_quality) = match navigation_fix {
        Some(fix) => {
            fix.validate()?;
            (Some(fix.latitude_deg), Some(fix.longitude_deg), fix.quality)
        }
        None => (None, None, NavigationQuality::DeadReckoning),
    };

    let maritime = MaritimeState {
        platform_id: platform_id.into(),
        kind: MaritimePlatformKind::UncrewedSurfaceVessel,
        monotonic_time_ms,
        latitude_deg,
        longitude_deg,
        depth_m: None,
        speed_mps: state.speed_mps(),
        heading_deg: state.heading_deg,
        energy_remaining_fraction: state.energy_remaining_fraction,
        navigation_quality,
    };
    maritime.validate()?;
    Ok(maritime)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> UsvState {
        UsvState {
            local_position_m: [0.0, 0.0],
            heading_deg: 45.0,
            surge_mps: 2.0,
            sway_mps: 0.0,
            yaw_rate_rad_s: 0.0,
            energy_remaining_fraction: 0.6,
        }
    }

    #[test]
    fn absent_external_fix_becomes_dead_reckoning_not_fake_coordinates() {
        let maritime = to_maritime_state("usv-1", 42, &state(), None).unwrap();
        assert_eq!(maritime.latitude_deg, None);
        assert_eq!(maritime.longitude_deg, None);
        assert_eq!(maritime.navigation_quality, NavigationQuality::DeadReckoning);
        assert_eq!(maritime.kind, MaritimePlatformKind::UncrewedSurfaceVessel);
        assert_eq!(maritime.validate(), Ok(()));
    }

    #[test]
    fn invalid_local_state_cannot_cross_adapter_boundary() {
        let mut invalid = state();
        invalid.surge_mps = f32::NAN;
        assert!(to_maritime_state("usv-1", 42, &invalid, None).is_err());
    }

    #[test]
    fn invalid_external_fix_cannot_cross_adapter_boundary() {
        let invalid_fix = UsvNavigationFix {
            latitude_deg: 91.0,
            longitude_deg: 18.4,
            quality: NavigationQuality::Nominal,
        };
        assert!(to_maritime_state("usv-1", 42, &state(), Some(invalid_fix)).is_err());
    }

    #[test]
    fn malformed_platform_identity_cannot_cross_adapter_boundary() {
        assert!(to_maritime_state("   ", 42, &state(), None).is_err());
    }
}

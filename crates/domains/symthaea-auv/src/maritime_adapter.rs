// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adapter from the AUV domain into the shared maritime substrate.
//!
//! The AUV keeps ownership of hydrodynamics, control, chemical sensing and
//! underwater navigation. This adapter only projects already-known platform state
//! into mission-neutral maritime types.

use crate::AuvState;
use symthaea_maritime_core::{MaritimePlatformKind, MaritimeState, NavigationQuality};

/// Context not derivable from the local hydrodynamic state alone.
#[derive(Debug, Clone, PartialEq)]
pub struct AuvMaritimeContext {
    pub platform_id: String,
    pub monotonic_time_ms: u64,
    pub latitude_deg: Option<f64>,
    pub longitude_deg: Option<f64>,
    pub heading_deg: f32,
    pub energy_remaining_fraction: f32,
    pub navigation_quality: NavigationQuality,
}

/// Project AUV-local state through the validated shared maritime boundary.
///
/// Context supplied by navigation/energy integrations is not assumed trustworthy by
/// construction. The final shared state must validate before it leaves this adapter.
pub fn to_maritime_state(
    state: &AuvState,
    context: &AuvMaritimeContext,
) -> Result<MaritimeState, &'static str> {
    let maritime = MaritimeState {
        platform_id: context.platform_id.clone(),
        kind: MaritimePlatformKind::AutonomousUnderwaterVehicle,
        monotonic_time_ms: context.monotonic_time_ms,
        latitude_deg: context.latitude_deg,
        longitude_deg: context.longitude_deg,
        depth_m: Some(state.depth_m() as f32),
        speed_mps: state.speed() as f32,
        heading_deg: context.heading_deg,
        energy_remaining_fraction: context.energy_remaining_fraction,
        navigation_quality: context.navigation_quality,
    };
    maritime.validate()?;
    Ok(maritime)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> AuvMaritimeContext {
        AuvMaritimeContext {
            platform_id: "auv-cape-01".into(),
            monotonic_time_ms: 1000,
            latitude_deg: Some(-33.9249),
            longitude_deg: Some(18.4241),
            heading_deg: 90.0,
            energy_remaining_fraction: 0.8,
            navigation_quality: NavigationQuality::DeadReckoning,
        }
    }

    #[test]
    fn projects_auv_state_without_copying_domain_specific_channels() {
        let state = AuvState::neutral_buoyancy(25.0);
        let maritime = to_maritime_state(&state, &context()).unwrap();
        assert_eq!(maritime.kind, MaritimePlatformKind::AutonomousUnderwaterVehicle);
        assert_eq!(maritime.depth_m, Some(25.0));
        assert_eq!(maritime.navigation_quality, NavigationQuality::DeadReckoning);
        assert_eq!(maritime.validate(), Ok(()));
    }

    #[test]
    fn invalid_context_cannot_cross_adapter_boundary() {
        let state = AuvState::neutral_buoyancy(25.0);
        let mut invalid = context();
        invalid.energy_remaining_fraction = f32::NAN;
        assert!(to_maritime_state(&state, &invalid).is_err());

        invalid = context();
        invalid.longitude_deg = None;
        assert!(to_maritime_state(&state, &invalid).is_err());
    }
}

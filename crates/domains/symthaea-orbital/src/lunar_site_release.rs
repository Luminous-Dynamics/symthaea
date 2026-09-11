// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! LL-004E lunar surface-site -> inertial release-state adapter.
//!
//! This module deliberately does **not** implement lunar orientation itself.
//! It consumes a provenance-bound body-fixed -> inertial orientation snapshot
//! (for example, generated offline from SPICE MOON_ME / MOON_PA kernels) and
//! transforms a declared site-local release vector into the exact inertial
//! frame used by LL-004 moving-target studies.

use serde::{Deserialize, Serialize};

use crate::cislunar_oracle::{EphemerisSource, FrameContract, StateVectorKm};
use crate::cislunar_target::InertialReleaseState;

const SECONDS_PER_DAY: f64 = 86_400.0;
const ORTHONORMAL_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LunarSiteReleaseSpec {
    pub release_id: String,
    pub site_id: String,
    /// Explicit lunar body-fixed frame, e.g. `MOON_ME_DE440` or a synthetic fixture frame.
    pub body_fixed_frame: String,
    /// Planetocentric latitude in degrees.
    pub latitude_deg: f64,
    /// East-positive planetocentric longitude in degrees.
    pub longitude_deg: f64,
    /// Spherical reference radius plus any explicitly supplied site elevation, km.
    pub site_radius_km: f64,
    /// Azimuth clockwise from local north, degrees.
    pub launch_azimuth_deg: f64,
    /// Elevation above the local horizontal plane, degrees.
    pub launch_elevation_deg: f64,
    /// Release speed relative to the rotating lunar surface, km/s.
    pub release_speed_km_s: f64,
    pub epoch_jd: f64,
    pub site_ref: String,
    pub evidence_refs: Vec<String>,
}

impl LunarSiteReleaseSpec {
    pub fn is_well_formed(&self) -> bool {
        !self.release_id.trim().is_empty()
            && !self.site_id.trim().is_empty()
            && !self.body_fixed_frame.trim().is_empty()
            && self.latitude_deg.is_finite()
            && (-90.0..=90.0).contains(&self.latitude_deg)
            && self.longitude_deg.is_finite()
            && self.site_radius_km.is_finite()
            && self.site_radius_km > 0.0
            && self.launch_azimuth_deg.is_finite()
            && self.launch_elevation_deg.is_finite()
            && (-90.0..=90.0).contains(&self.launch_elevation_deg)
            && self.release_speed_km_s.is_finite()
            && self.release_speed_km_s >= 0.0
            && self.epoch_jd.is_finite()
            && self.epoch_jd > 0.0
            && !self.site_ref.trim().is_empty()
            && self
                .evidence_refs
                .iter()
                .all(|reference| !reference.trim().is_empty())
    }
}

/// Externally generated lunar orientation state at one epoch.
///
/// `rotation_body_fixed_to_inertial` is row-major and maps a vector expressed
/// in the declared body-fixed frame into `inertial_frame`. The angular velocity
/// is the body-fixed frame relative to inertial, expressed in the inertial frame.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LunarOrientationState {
    pub epoch_jd: f64,
    pub body_fixed_frame: String,
    pub inertial_frame: FrameContract,
    pub rotation_body_fixed_to_inertial: [[f64; 3]; 3],
    pub angular_velocity_inertial_rad_s: [f64; 3],
    pub source: EphemerisSource,
    /// Maximum permitted absolute epoch difference between this sample and a release.
    pub max_epoch_delta_s: f64,
    pub orientation_ref: String,
    pub evidence_refs: Vec<String>,
}

impl LunarOrientationState {
    pub fn is_well_formed(&self) -> bool {
        self.epoch_jd.is_finite()
            && self.epoch_jd > 0.0
            && !self.body_fixed_frame.trim().is_empty()
            && self.inertial_frame.is_well_formed()
            && matrix_is_rotation(self.rotation_body_fixed_to_inertial)
            && all_finite(self.angular_velocity_inertial_rad_s)
            && self.source.is_well_formed()
            && self.max_epoch_delta_s.is_finite()
            && self.max_epoch_delta_s >= 0.0
            && !self.orientation_ref.trim().is_empty()
            && self
                .evidence_refs
                .iter()
                .all(|reference| !reference.trim().is_empty())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteReleaseError {
    InvalidSite,
    InvalidOrientation,
    BodyFixedFrameMismatch,
    StaleOrientation,
    NumericalFailure,
}

/// Return body-fixed local north/east/up unit vectors for planetocentric coordinates.
///
/// This basis remains finite at the poles because it is constructed analytically
/// from latitude/longitude rather than from a cross product with a nearly parallel axis.
pub fn local_basis_body_fixed(
    latitude_deg: f64,
    longitude_deg: f64,
) -> Result<([f64; 3], [f64; 3], [f64; 3]), SiteReleaseError> {
    if !latitude_deg.is_finite()
        || !(-90.0..=90.0).contains(&latitude_deg)
        || !longitude_deg.is_finite()
    {
        return Err(SiteReleaseError::InvalidSite);
    }
    let lat = latitude_deg.to_radians();
    let lon = longitude_deg.to_radians();
    let (slat, clat) = lat.sin_cos();
    let (slon, clon) = lon.sin_cos();
    let up = [clat * clon, clat * slon, slat];
    let east = [-slon, clon, 0.0];
    let north = [-slat * clon, -slat * slon, clat];
    Ok((north, east, up))
}

/// Convert a lunar site-local release into the inertial state consumed by LL-004.
///
/// No trajectory propagation or launch authority occurs here.
pub fn site_release_to_inertial(
    site: &LunarSiteReleaseSpec,
    orientation: &LunarOrientationState,
    dynamics_ref: impl Into<String>,
    constants_ref: impl Into<String>,
) -> Result<InertialReleaseState, SiteReleaseError> {
    if !site.is_well_formed() {
        return Err(SiteReleaseError::InvalidSite);
    }
    if !orientation.is_well_formed() {
        return Err(SiteReleaseError::InvalidOrientation);
    }
    if site.body_fixed_frame != orientation.body_fixed_frame {
        return Err(SiteReleaseError::BodyFixedFrameMismatch);
    }
    let epoch_delta_s = (site.epoch_jd - orientation.epoch_jd).abs() * SECONDS_PER_DAY;
    if !epoch_delta_s.is_finite() || epoch_delta_s > orientation.max_epoch_delta_s {
        return Err(SiteReleaseError::StaleOrientation);
    }

    let (north, east, up) = local_basis_body_fixed(site.latitude_deg, site.longitude_deg)?;
    let position_body_fixed_km = scale(up, site.site_radius_km);

    let elevation = site.launch_elevation_deg.to_radians();
    let azimuth = site.launch_azimuth_deg.to_radians();
    let horizontal = elevation.cos();
    let local_direction = add(
        add(
            scale(north, horizontal * azimuth.cos()),
            scale(east, horizontal * azimuth.sin()),
        ),
        scale(up, elevation.sin()),
    );
    let relative_velocity_body_fixed_km_s = scale(local_direction, site.release_speed_km_s);

    let position_inertial_km = mat_vec(
        orientation.rotation_body_fixed_to_inertial,
        position_body_fixed_km,
    );
    let relative_velocity_inertial_km_s = mat_vec(
        orientation.rotation_body_fixed_to_inertial,
        relative_velocity_body_fixed_km_s,
    );
    let surface_rotation_velocity_km_s = cross(
        orientation.angular_velocity_inertial_rad_s,
        position_inertial_km,
    );
    let velocity_inertial_km_s = add(
        relative_velocity_inertial_km_s,
        surface_rotation_velocity_km_s,
    );

    if !all_finite(position_inertial_km) || !all_finite(velocity_inertial_km_s) {
        return Err(SiteReleaseError::NumericalFailure);
    }

    let mut evidence_refs = site.evidence_refs.clone();
    evidence_refs.extend(orientation.evidence_refs.iter().cloned());
    evidence_refs.push(site.site_ref.clone());
    evidence_refs.push(orientation.orientation_ref.clone());

    let result = InertialReleaseState {
        release_id: site.release_id.clone(),
        epoch_jd: site.epoch_jd,
        frame: orientation.inertial_frame.clone(),
        state: StateVectorKm {
            position_km: position_inertial_km,
            velocity_km_s: velocity_inertial_km_s,
        },
        dynamics_ref: dynamics_ref.into(),
        constants_ref: constants_ref.into(),
        evidence_refs,
    };
    result
        .is_well_formed()
        .then_some(result)
        .ok_or(SiteReleaseError::NumericalFailure)
}

fn matrix_is_rotation(matrix: [[f64; 3]; 3]) -> bool {
    if matrix.into_iter().flatten().any(|value| !value.is_finite()) {
        return false;
    }
    let rows = matrix;
    for i in 0..3 {
        if (dot(rows[i], rows[i]) - 1.0).abs() > ORTHONORMAL_TOLERANCE {
            return false;
        }
        for j in (i + 1)..3 {
            if dot(rows[i], rows[j]).abs() > ORTHONORMAL_TOLERANCE {
                return false;
            }
        }
    }
    (determinant(matrix) - 1.0).abs() <= 10.0 * ORTHONORMAL_TOLERANCE
}

fn determinant(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn mat_vec(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| x * y).sum()
}

fn scale(v: [f64; 3], scalar: f64) -> [f64; 3] {
    [v[0] * scalar, v[1] * scalar, v[2] * scalar]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn all_finite(v: [f64; 3]) -> bool {
    v.into_iter().all(f64::is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cislunar_oracle::{TimeScale, EphemerisSource};

    fn frame() -> FrameContract {
        FrameContract {
            name: "SYNTHETIC_INERTIAL".into(),
            origin: "Moon".into(),
            axes: "synthetic Cartesian".into(),
            time_scale: TimeScale::Tdb,
            position_unit: "km".into(),
            velocity_unit: "km/s".into(),
            derivative_convention: "inertial derivative".into(),
            contract_ref: "fixture-frame-v1".into(),
        }
    }

    fn source() -> EphemerisSource {
        EphemerisSource {
            provider: "synthetic".into(),
            product: "orientation-fixture".into(),
            version: "v1".into(),
            configuration_ref: "fixture".into(),
        }
    }

    fn site(speed: f64, azimuth: f64, elevation: f64) -> LunarSiteReleaseSpec {
        LunarSiteReleaseSpec {
            release_id: "release-1".into(),
            site_id: "site-1".into(),
            body_fixed_frame: "SYNTHETIC_MOON_FIXED".into(),
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            site_radius_km: 1000.0,
            launch_azimuth_deg: azimuth,
            launch_elevation_deg: elevation,
            release_speed_km_s: speed,
            epoch_jd: 2_460_000.5,
            site_ref: "site-fixture".into(),
            evidence_refs: vec!["site-evidence".into()],
        }
    }

    fn orientation(rotation: [[f64; 3]; 3], omega: [f64; 3]) -> LunarOrientationState {
        LunarOrientationState {
            epoch_jd: 2_460_000.5,
            body_fixed_frame: "SYNTHETIC_MOON_FIXED".into(),
            inertial_frame: frame(),
            rotation_body_fixed_to_inertial: rotation,
            angular_velocity_inertial_rad_s: omega,
            source: source(),
            max_epoch_delta_s: 0.1,
            orientation_ref: "orientation-fixture".into(),
            evidence_refs: vec!["orientation-evidence".into()],
        }
    }

    fn identity() -> [[f64; 3]; 3] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn identity_zero_spin_preserves_equatorial_north_release() {
        let result = site_release_to_inertial(
            &site(0.1, 0.0, 0.0),
            &orientation(identity(), [0.0; 3]),
            "two-body",
            "synthetic",
        )
        .unwrap();
        assert!((result.state.position_km[0] - 1000.0).abs() < 1.0e-12);
        assert!(result.state.position_km[1].abs() < 1.0e-12);
        assert!(result.state.velocity_km_s[0].abs() < 1.0e-12);
        assert!(result.state.velocity_km_s[1].abs() < 1.0e-12);
        assert!((result.state.velocity_km_s[2] - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn z_quarter_turn_rotates_site_position() {
        let rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let result = site_release_to_inertial(
            &site(0.0, 0.0, 0.0),
            &orientation(rotation, [0.0; 3]),
            "two-body",
            "synthetic",
        )
        .unwrap();
        assert!(result.state.position_km[0].abs() < 1.0e-12);
        assert!((result.state.position_km[1] - 1000.0).abs() < 1.0e-12);
    }

    #[test]
    fn body_rotation_velocity_is_added_exactly_once() {
        let result = site_release_to_inertial(
            &site(0.0, 0.0, 0.0),
            &orientation(identity(), [0.0, 0.0, 1.0e-3]),
            "two-body",
            "synthetic",
        )
        .unwrap();
        assert!(result.state.velocity_km_s[0].abs() < 1.0e-12);
        assert!((result.state.velocity_km_s[1] - 1.0).abs() < 1.0e-12);
        assert!(result.state.velocity_km_s[2].abs() < 1.0e-12);
    }

    #[test]
    fn local_basis_remains_orthonormal_near_pole() {
        let (north, east, up) = local_basis_body_fixed(-89.999_999, 37.0).unwrap();
        for axis in [north, east, up] {
            assert!(axis.into_iter().all(f64::is_finite));
            assert!((dot(axis, axis) - 1.0).abs() < 1.0e-12);
        }
        assert!(dot(north, east).abs() < 1.0e-12);
        assert!(dot(north, up).abs() < 1.0e-12);
        assert!(dot(east, up).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_rotation_and_stale_orientation_fail_closed() {
        let bad = orientation([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.0; 3]);
        assert_eq!(
            site_release_to_inertial(&site(0.1, 0.0, 0.0), &bad, "two-body", "synthetic"),
            Err(SiteReleaseError::InvalidOrientation)
        );

        let mut stale_site = site(0.1, 0.0, 0.0);
        stale_site.epoch_jd += 10.0 / SECONDS_PER_DAY;
        assert_eq!(
            site_release_to_inertial(
                &stale_site,
                &orientation(identity(), [0.0; 3]),
                "two-body",
                "synthetic"
            ),
            Err(SiteReleaseError::StaleOrientation)
        );
    }
}

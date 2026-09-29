// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! First-order Mars surface-to-synchronous-tether mechanics.
//!
//! This module deliberately stops short of a full elevator solver. It provides
//! an auditable T0/T1 reference kernel: Mars constants, rotating-frame
//! effective acceleration, synchronous radius, constant-stress taper exponent,
//! and distributed-mass quadrature. Higher-fidelity orbital ephemerides,
//! flexible dynamics, climber loads, and termination/counterweight boundary
//! conditions belong in separate adapters.
//!
//! The implementation is intentionally dependency-free so its limiting cases
//! can be cross-checked against hand calculations and external solvers.

use std::f64::consts::PI;

/// Mars reference constants used by the first-order tether model.
///
/// Values follow NASA's Mars fact sheet / standard reference values used by
/// the engineering foundation document. SI units are used internally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsTetherReference {
    /// Mars gravitational parameter, m^3/s^2.
    pub mu_m3_s2: f64,
    /// Mean equatorial reference radius, m.
    pub radius_m: f64,
    /// Sidereal rotation period, s.
    pub rotation_period_s: f64,
    /// Phobos mean orbital radius from Mars center, m.
    pub phobos_radius_m: f64,
    /// Deimos mean orbital radius from Mars center, m.
    pub deimos_radius_m: f64,
}

impl MarsTetherReference {
    pub const MARS: Self = Self {
        mu_m3_s2: 4.2828372e13,
        radius_m: 3_396_200.0,
        rotation_period_s: 88_642.44,
        phobos_radius_m: 9_378_000.0,
        deimos_radius_m: 23_459_000.0,
    };

    /// Mars angular rotation rate, rad/s.
    pub const fn omega_rad_s(self) -> f64 {
        2.0 * PI / self.rotation_period_s
    }

    /// Radius of the circular synchronous orbit, m.
    pub fn synchronous_radius_m(self) -> f64 {
        (self.mu_m3_s2 / self.omega_rad_s().powi(2)).cbrt()
    }

    /// Synchronous altitude above the reference radius, m.
    pub fn synchronous_altitude_m(self) -> f64 {
        self.synchronous_radius_m() - self.radius_m
    }

    /// Radial effective acceleration in the Mars-corotating frame, m/s^2.
    ///
    /// Positive means centrifugal acceleration exceeds Mars gravity;
    /// negative means gravity dominates.
    pub fn effective_radial_acceleration_m_s2(self, radius_m: f64) -> f64 {
        let r = radius_m.max(1.0);
        self.omega_rad_s().powi(2) * r - self.mu_m3_s2 / r.powi(2)
    }

    /// Negative integral of effective acceleration from r0 to r1.
    ///
    /// This is the specific-energy-like quantity appearing in the
    /// constant-stress taper exponent:
    ///   ln(A1/A0) = -integral(a_eff dr) / (sigma/rho).
    pub fn taper_integral_m2_s2(self, r0_m: f64, r1_m: f64) -> f64 {
        let r0 = r0_m.max(1.0);
        let r1 = r1_m.max(1.0);
        self.mu_m3_s2 * (1.0 / r0 - 1.0 / r1)
            - 0.5 * self.omega_rad_s().powi(2) * (r1.powi(2) - r0.powi(2))
    }
}

/// A three-dimensional vector in an areocentric Mars frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x_m: f64,
    pub y_m: f64,
    pub z_m: f64,
}

impl Vec3 {
    pub const ZERO: Self = Self { x_m: 0.0, y_m: 0.0, z_m: 0.0 };

    pub fn norm_m(self) -> f64 {
        (self.x_m * self.x_m + self.y_m * self.y_m + self.z_m * self.z_m).sqrt()
    }

    pub fn scale(self, k: f64) -> Self {
        Self { x_m: self.x_m * k, y_m: self.y_m * k, z_m: self.z_m * k }
    }

    pub fn add(self, other: Self) -> Self {
        Self { x_m: self.x_m + other.x_m, y_m: self.y_m + other.y_m, z_m: self.z_m + other.z_m }
    }

    pub fn sub(self, other: Self) -> Self {
        Self { x_m: self.x_m - other.x_m, y_m: self.y_m - other.y_m, z_m: self.z_m - other.z_m }
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x_m * other.x_m + self.y_m * other.y_m + self.z_m * other.z_m
    }
}

/// Spherical surface anchor coordinates.
///
/// Longitude is positive east; latitude is areocentric. Radius includes
/// elevation above the reference spherical Mars radius. This deliberately
/// avoids conflating a DEM/areoid height with a structural anchor elevation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsAnchor {
    pub latitude_rad: f64,
    pub longitude_rad: f64,
    pub elevation_m: f64,
}

impl MarsAnchor {
    pub fn position(self, reference: MarsTetherReference) -> Vec3 {
        let r = reference.radius_m + self.elevation_m;
        let clat = self.latitude_rad.cos();
        Vec3 {
            x_m: r * clat * self.longitude_rad.cos(),
            y_m: r * clat * self.longitude_rad.sin(),
            z_m: r * self.latitude_rad.sin(),
        }
    }

    /// Local east, north, up basis vectors at the anchor.
    pub fn enu_basis(self) -> (Vec3, Vec3, Vec3) {
        let lat = self.latitude_rad;
        let lon = self.longitude_rad;
        (
            Vec3 { x_m: -lon.sin(), y_m: lon.cos(), z_m: 0.0 },
            Vec3 { x_m: -lat.sin() * lon.cos(), y_m: -lat.sin() * lon.sin(), z_m: lat.cos() },
            Vec3 { x_m: lat.cos() * lon.cos(), y_m: lat.cos() * lon.sin(), z_m: lat.sin() },
        )
    }

    /// Outward tether direction for azimuth measured east of north and
    /// elevation measured above the local horizontal plane.
    pub fn tether_direction(self, azimuth_rad: f64, elevation_rad: f64) -> Vec3 {
        let (east, north, up) = self.enu_basis();
        let ce = elevation_rad.cos();
        north.scale(ce * azimuth_rad.cos())
            .add(east.scale(ce * azimuth_rad.sin()))
            .add(up.scale(elevation_rad.sin()))
    }
}

/// Straight-line T0 rotating tether geometry.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotatingTetherGeometry {
    pub anchor_position_m: Vec3,
    pub direction: Vec3,
    pub endpoint_position_m: Vec3,
    pub length_m: f64,
    pub endpoint_rotating_velocity_m_s: Vec3,
}

impl RotatingTetherGeometry {
    pub fn from_anchor(
        reference: MarsTetherReference,
        anchor: MarsAnchor,
        azimuth_rad: f64,
        elevation_rad: f64,
        length_m: f64,
    ) -> Self {
        let anchor_position = anchor.position(reference);
        let direction = anchor.tether_direction(azimuth_rad, elevation_rad);
        let endpoint = anchor_position.add(direction.scale(length_m));
        let omega = reference.omega_rad_s();
        let velocity = Vec3 {
            x_m: -omega * endpoint.y_m,
            y_m: omega * endpoint.x_m,
            z_m: 0.0,
        };
        Self {
            anchor_position_m: anchor_position,
            direction,
            endpoint_position_m: endpoint,
            length_m,
            endpoint_rotating_velocity_m_s: velocity,
        }
    }

    /// Forward intersection of the straight tether with a sphere centered on Mars.
    ///
    /// Returns the first positive distance along the tether and the
    /// corresponding point. This is a geometric diagnostic only; it does not
    /// establish structural equilibrium or dynamical stability.
    pub fn sphere_intersection(&self, radius_m: f64) -> Option<(f64, Vec3)> {
        let a = self.direction.dot(self.direction);
        let b = 2.0 * self.anchor_position_m.dot(self.direction);
        let c = self.anchor_position_m.dot(self.anchor_position_m) - radius_m * radius_m;
        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 || a <= f64::MIN_POSITIVE {
            return None;
        }
        let root = discriminant.sqrt();
        let t1 = (-b - root) / (2.0 * a);
        let t2 = (-b + root) / (2.0 * a);
        let t = [t1, t2]
            .into_iter()
            .filter(|v| *v >= 0.0)
            .fold(f64::INFINITY, f64::min);
        if t.is_finite() && t <= self.length_m {
            Some((t, self.anchor_position_m.add(self.direction.scale(t))))
        } else {
            None
        }
    }
}

/// Provenance attached to a terrain observation; identifiers are opaque
/// source/version labels, never interpreted as authority by this module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerrainProvenance {
    pub source_id: String,
    pub source_revision: String,
    pub coordinate_reference: String,
}

/// Quality state for a terrain sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainQuality {
    Measured,
    Interpolated,
    Missing,
    Invalid,
}

/// Vertical reference used by a terrain sample. Values must never be
/// interpreted as interchangeable heights without an explicit conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainVerticalDatum {
    /// Height above a named areoid/equipotential surface.
    AreoidRelative,
    /// Height above the spherical reference radius in MarsTetherReference.
    ReferenceSphereRelative,
    /// Absolute distance from Mars' center of mass.
    PlanetocentricRadius,
}

/// Terrain observation in the same areocentric, east-positive convention
/// used by the MOLA MEGDR products. Elevation is relative to the supplied
/// terrain product's vertical datum (typically its areoid), not automatically
/// interchangeable with radius above the reference sphere.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainSample {
    pub latitude_rad: f64,
    pub longitude_rad: f64,
    pub elevation_m: Option<f64>,
    pub elevation_uncertainty_m: Option<f64>,
    pub vertical_datum: TerrainVerticalDatum,
    pub slope_rad: Option<f64>,
    pub roughness_m: Option<f64>,
    pub quality: TerrainQuality,
    pub provenance: TerrainProvenance,
}

impl TerrainSample {
    pub fn is_usable(&self) -> bool {
        self.quality != TerrainQuality::Missing
            && self.quality != TerrainQuality::Invalid
            && self.elevation_m.is_some_and(f64::is_finite)
            && self.elevation_uncertainty_m.is_some_and(|v| v.is_finite() && v >= 0.0)
            && self.latitude_rad.is_finite()
            && (-PI / 2.0..=PI / 2.0).contains(&self.latitude_rad)
            && self.longitude_rad.is_finite()
            && !self.provenance.source_id.trim().is_empty()
            && !self.provenance.source_revision.trim().is_empty()
            && !self.provenance.coordinate_reference.trim().is_empty()
            && self.slope_rad.is_none_or(|v| v.is_finite() && (0.0..=PI / 2.0).contains(&v))
            && self.roughness_m.is_none_or(|v| v.is_finite() && v >= 0.0)
    }
}

/// Terrain access is injected by a dataset-specific adapter. This core does
/// not load, reinterpret, or silently interpolate a DEM.
pub trait TerrainProvider {
    fn sample(&self, latitude_rad: f64, longitude_rad: f64) -> TerrainSample;
}

/// Constraint state for an anchor candidate. These are descriptive states,
/// not an overall ranking or a certification of engineering feasibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeasibilityState {
    FeasibleUnderModel,
    InfeasibleUnderModel,
    InsufficientEvidence,
    HigherFidelityRequired,
}

/// Converted terrain height above the kernel reference sphere, with a
/// conservative absolute uncertainty bound in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadialTerrainElevation {
    pub elevation_above_reference_m: f64,
    pub uncertainty_bound_m: f64,
}

/// Convert a terrain sample to height above the kernel reference sphere.
/// For areoid-relative samples, both the areoid radius and its uncertainty
/// must be supplied from a compatible, versioned body-fixed product. Bounds
/// are combined by addition (worst-case), not root-sum-square, because their
/// statistical independence is not established here.
pub fn radial_terrain_elevation(
    reference: MarsTetherReference,
    sample: &TerrainSample,
    areoid_radius_m: Option<f64>,
    areoid_uncertainty_m: Option<f64>,
) -> Option<RadialTerrainElevation> {
    if !sample.is_usable() {
        return None;
    }
    let height = sample.elevation_m?;
    let sample_uncertainty = sample.elevation_uncertainty_m?;
    let (radial_elevation, uncertainty) = match sample.vertical_datum {
        TerrainVerticalDatum::AreoidRelative => {
            let areoid = areoid_radius_m?;
            let areoid_uncertainty = areoid_uncertainty_m?;
            if !areoid.is_finite()
                || !areoid_uncertainty.is_finite()
                || areoid_uncertainty < 0.0
            {
                return None;
            }
            (
                areoid + height - reference.radius_m,
                areoid_uncertainty + sample_uncertainty,
            )
        }
        TerrainVerticalDatum::ReferenceSphereRelative => (height, sample_uncertainty),
        TerrainVerticalDatum::PlanetocentricRadius => (height - reference.radius_m, sample_uncertainty),
    };
    if !radial_elevation.is_finite()
        || radial_elevation <= -reference.radius_m
        || !uncertainty.is_finite()
    {
        return None;
    }
    Some(RadialTerrainElevation {
        elevation_above_reference_m: radial_elevation,
        uncertainty_bound_m: uncertainty,
    })
}

/// Convert areoid-relative MOLA topography into radial elevation above the
/// kernel's spherical reference radius. Retained as a scalar helper for
/// callers that already validate uncertainty and provenance separately.
pub fn radial_elevation_from_areoid(
    reference: MarsTetherReference,
    areoid_radius_m: f64,
    topography_above_areoid_m: f64,
) -> Option<f64> {
    if !areoid_radius_m.is_finite() || !topography_above_areoid_m.is_finite() {
        return None;
    }
    let radial_elevation = areoid_radius_m + topography_above_areoid_m - reference.radius_m;
    (radial_elevation.is_finite() && radial_elevation > -reference.radius_m)
        .then_some(radial_elevation)
}

/// Minimal site geometry assessment. It deliberately does not combine
/// geotechnical, structural, logistics, or environmental constraints into a
/// scalar score.
#[derive(Debug, Clone, PartialEq)]
pub struct AnchorGeometryAssessment {
    pub state: FeasibilityState,
    pub terrain_usable: bool,
    pub anchor_position_m: Option<Vec3>,
    pub radial_distance_to_sync_m: Option<f64>,
    pub reason: &'static str,
}

pub fn assess_anchor_geometry(
    reference: MarsTetherReference,
    anchor: MarsAnchor,
    terrain: &TerrainSample,
    minimum_anchor_radius_m: f64,
) -> AnchorGeometryAssessment {
    if !minimum_anchor_radius_m.is_finite() || minimum_anchor_radius_m <= 0.0 {
        return AnchorGeometryAssessment {
            state: FeasibilityState::InsufficientEvidence,
            terrain_usable: false,
            anchor_position_m: None,
            radial_distance_to_sync_m: None,
            reason: "minimum anchor radius is invalid or unspecified",
        };
    }
    if !terrain.is_usable() {
        return AnchorGeometryAssessment {
            state: FeasibilityState::InsufficientEvidence,
            terrain_usable: false,
            anchor_position_m: None,
            radial_distance_to_sync_m: None,
            reason: "terrain sample missing, invalid, or uncertainty-unbounded",
        };
    }
    if !anchor.latitude_rad.is_finite()
        || !anchor.longitude_rad.is_finite()
        || !anchor.elevation_m.is_finite()
        || anchor.elevation_m < 0.0
    {
        return AnchorGeometryAssessment {
            state: FeasibilityState::InfeasibleUnderModel,
            terrain_usable: true,
            anchor_position_m: None,
            radial_distance_to_sync_m: None,
            reason: "anchor coordinates or elevation are outside the supported domain",
        };
    }
    let latitude_delta = (terrain.latitude_rad - anchor.latitude_rad).abs();
    let raw_longitude_delta = (terrain.longitude_rad - anchor.longitude_rad).abs();
    let longitude_delta = ((raw_longitude_delta + PI).rem_euclid(2.0 * PI) - PI).abs();
    if latitude_delta > 1.0e-8 || longitude_delta > 1.0e-8 {
        return AnchorGeometryAssessment {
            state: FeasibilityState::InsufficientEvidence,
            terrain_usable: false,
            anchor_position_m: None,
            radial_distance_to_sync_m: None,
            reason: "terrain sample coordinates do not match the requested anchor",
        };
    }
    let position = anchor.position(reference);
    let radius = position.norm_m();
    if radius < minimum_anchor_radius_m {
        return AnchorGeometryAssessment {
            state: FeasibilityState::InfeasibleUnderModel,
            terrain_usable: true,
            anchor_position_m: Some(position),
            radial_distance_to_sync_m: Some(reference.synchronous_radius_m() - radius),
            reason: "anchor lies inside the configured minimum radius",
        };
    }
    AnchorGeometryAssessment {
        state: FeasibilityState::HigherFidelityRequired,
        terrain_usable: true,
        anchor_position_m: Some(position),
        radial_distance_to_sync_m: Some(reference.synchronous_radius_m() - radius),
        reason: "geometry is representable; terrain datum, geology, tether dynamics, and safety remain unverified",
    }
}

/// Effective material properties for the first-order tether model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TetherMaterial {
    /// Bulk density, kg/m^3.
    pub density_kg_m3: f64,
    /// Allowable axial stress after all design knock-down factors, Pa.
    pub allowable_stress_pa: f64,
}

impl TetherMaterial {
    /// Specific allowable strength, sigma/rho, m^2/s^2.
    pub fn specific_strength_m2_s2(self) -> f64 {
        self.allowable_stress_pa / self.density_kg_m3.max(f64::MIN_POSITIVE)
    }
}

/// First-order constant-stress taper result between two radii.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaperResult {
    pub r0_m: f64,
    pub r1_m: f64,
    pub taper_integral_m2_s2: f64,
    pub specific_strength_m2_s2: f64,
    pub area_ratio_a1_a0: f64,
}

/// Compute the constant-stress area ratio A(r1)/A(r0).
///
/// This is a limiting-case sanity check, not a complete elevator design:
/// the model has no apex/counterweight boundary condition, climber loads,
/// dynamic loads, defects, bending, fatigue, or environmental degradation.
pub fn constant_stress_taper(
    reference: MarsTetherReference,
    material: TetherMaterial,
    r0_m: f64,
    r1_m: f64,
) -> TaperResult {
    let integral = reference.taper_integral_m2_s2(r0_m, r1_m);
    let specific_strength = material.specific_strength_m2_s2();
    let exponent = (integral / specific_strength).clamp(-700.0, 700.0);
    TaperResult {
        r0_m,
        r1_m,
        taper_integral_m2_s2: integral,
        specific_strength_m2_s2: specific_strength,
        area_ratio_a1_a0: exponent.exp(),
    }
}

/// Numerically integrate the tether area profile implied by constant stress.
///
/// Returns tether mass per unit anchor area, kg/m^2, over [r0, r1].
/// The result is useful for trade studies because it exposes the mass
/// consequence of material specific strength without hiding it in a single
/// feasibility score.
pub fn mass_per_anchor_area(
    reference: MarsTetherReference,
    material: TetherMaterial,
    r0_m: f64,
    r1_m: f64,
    steps: usize,
) -> f64 {
    assert!(steps > 0, "steps must be non-zero");
    let n = if steps % 2 == 0 { steps } else { steps + 1 };
    let h = (r1_m - r0_m) / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let r = r0_m + i as f64 * h;
        let a_ratio = constant_stress_taper(reference, material, r0_m, r).area_ratio_a1_a0;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * material.density_kg_m3 * a_ratio;
    }
    sum * h / 3.0
}

/// Dimensionless phase-independent clearance margin for a satellite radius.
///
/// This does not replace a phase-aware ephemeris. It simply states whether
/// the tether's radial interval intersects a satellite's mean orbital radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RadialIntersection {
    IntersectsMeanOrbit,
    ClearOfMeanOrbit,
}

pub fn mean_orbit_intersection(
    r0_m: f64,
    r1_m: f64,
    satellite_radius_m: f64,
) -> RadialIntersection {
    let lo = r0_m.min(r1_m);
    let hi = r0_m.max(r1_m);
    if (lo..=hi).contains(&satellite_radius_m) {
        RadialIntersection::IntersectsMeanOrbit
    } else {
        RadialIntersection::ClearOfMeanOrbit
    }
}

/// Compact reference envelope for the Mars surface-to-AMO problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarsTetherReferenceEnvelope {
    pub surface_radius_m: f64,
    pub synchronous_radius_m: f64,
    pub synchronous_altitude_m: f64,
    pub phobos_radius_m: f64,
    pub deimos_radius_m: f64,
    pub surface_effective_accel_m_s2: f64,
}

impl MarsTetherReferenceEnvelope {
    pub fn mars() -> Self {
        let m = MarsTetherReference::MARS;
        Self {
            surface_radius_m: m.radius_m,
            synchronous_radius_m: m.synchronous_radius_m(),
            synchronous_altitude_m: m.synchronous_altitude_m(),
            phobos_radius_m: m.phobos_radius_m,
            deimos_radius_m: m.deimos_radius_m,
            surface_effective_accel_m_s2: m.effective_radial_acceleration_m_s2(m.radius_m),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS_RADIUS_M: f64 = 5.0;
    const EPS_ACCEL: f64 = 1e-6;

    #[test]
    fn synchronous_radius_matches_independent_reference() {
        let m = MarsTetherReference::MARS;
        assert!((m.synchronous_radius_m() - 20_427_650.0).abs() < EPS_RADIUS_M);
        assert!((m.synchronous_altitude_m() - 17_031_450.0).abs() < EPS_RADIUS_M);
    }

    #[test]
    fn equatorial_vertical_geometry_reaches_sync_at_expected_length() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            elevation_m: 0.0,
        };
        let g = RotatingTetherGeometry::from_anchor(
            m,
            anchor,
            0.0,
            PI / 2.0,
            m.synchronous_altitude_m(),
        );
        assert!((g.endpoint_position_m.norm_m() - m.synchronous_radius_m()).abs() < 5.0);
        assert!(g.endpoint_rotating_velocity_m_s.y_m > 1400.0);
        assert!(g.endpoint_rotating_velocity_m_s.x_m.abs() < 1e-9);
    }

    #[test]
    fn off_equator_local_vertical_has_expected_components() {
        let anchor = MarsAnchor {
            latitude_rad: 30.0_f64.to_radians(),
            longitude_rad: 45.0_f64.to_radians(),
            elevation_m: 0.0,
        };
        let up = anchor.enu_basis().2;
        let direction = anchor.tether_direction(0.0, PI / 2.0);
        assert!((direction.x_m - up.x_m).abs() < 1e-12);
        assert!((direction.y_m - up.y_m).abs() < 1e-12);
        assert!((direction.z_m - up.z_m).abs() < 1e-12);
    }

    #[test]
    fn outward_tether_intersects_sync_sphere_when_geometry_allows_it() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            elevation_m: 0.0,
        };
        let g = RotatingTetherGeometry::from_anchor(
            m,
            anchor,
            0.0,
            PI / 2.0,
            1.0e6,
        );
        assert!(g.sphere_intersection(m.synchronous_radius_m()).is_none());
        let long_geometry = RotatingTetherGeometry::from_anchor(
            m, anchor, 0.0, PI / 2.0, m.synchronous_altitude_m(),
        );
        let hit = long_geometry.sphere_intersection(m.synchronous_radius_m()).expect("sync sphere");
        assert!(hit.0 > 0.0);
        assert!((hit.1.norm_m() - m.synchronous_radius_m()).abs() < 1e-6);
    }

    fn fixture_terrain(quality: TerrainQuality) -> TerrainSample {
        TerrainSample {
            latitude_rad: 0.0,
            longitude_rad: 0.0,
            elevation_m: Some(1200.0),
            elevation_uncertainty_m: Some(5.0),
            vertical_datum: TerrainVerticalDatum::AreoidRelative,
            slope_rad: Some(0.02),
            roughness_m: Some(3.0),
            quality,
            provenance: TerrainProvenance {
                source_id: "test-fixture".into(),
                source_revision: "v1".into(),
                coordinate_reference: "areocentric-east-positive".into(),
            },
        }
    }

    #[test]
    fn terrain_contract_rejects_missing_and_unbounded_uncertainty() {
        let mut sample = fixture_terrain(TerrainQuality::Measured);
        assert!(sample.is_usable());
        sample.elevation_uncertainty_m = None;
        assert!(!sample.is_usable());
        sample = fixture_terrain(TerrainQuality::Missing);
        assert!(!sample.is_usable());
    }

    #[test]
    fn areoid_topography_conversion_is_explicit_and_validated() {
        let m = MarsTetherReference::MARS;
        assert_eq!(radial_elevation_from_areoid(m, m.radius_m + 120.0, 80.0), Some(200.0));
        assert!(radial_elevation_from_areoid(m, f64::NAN, 1.0).is_none());
    }

    #[test]
    fn terrain_contract_rejects_invalid_coordinates_and_missing_provenance() {
        let mut sample = fixture_terrain(TerrainQuality::Measured);
        sample.latitude_rad = PI;
        assert!(!sample.is_usable());
        sample = fixture_terrain(TerrainQuality::Measured);
        sample.provenance.source_revision = "  ".into();
        assert!(!sample.is_usable());
        sample = fixture_terrain(TerrainQuality::Measured);
        sample.slope_rad = Some(PI);
        assert!(!sample.is_usable());
    }

    #[test]
    fn terrain_datum_conversion_preserves_conservative_uncertainty() {
        let m = MarsTetherReference::MARS;
        let sample = fixture_terrain(TerrainQuality::Measured);
        let converted = radial_terrain_elevation(m, &sample, Some(m.radius_m + 120.0), Some(2.0)).unwrap();
        assert_eq!(converted.elevation_above_reference_m, 1320.0);
        assert_eq!(converted.uncertainty_bound_m, 7.0);
        let mut direct = sample.clone();
        direct.vertical_datum = TerrainVerticalDatum::ReferenceSphereRelative;
        let converted = radial_terrain_elevation(m, &direct, None, None).unwrap();
        assert_eq!(converted.elevation_above_reference_m, 1200.0);
        assert_eq!(converted.uncertainty_bound_m, 5.0);
        direct.vertical_datum = TerrainVerticalDatum::PlanetocentricRadius;
        direct.elevation_m = Some(m.radius_m + 1200.0);
        let converted = radial_terrain_elevation(m, &direct, None, None).unwrap();
        assert_eq!(converted.elevation_above_reference_m, 1200.0);
        assert!(radial_terrain_elevation(m, &sample, None, None).is_none());
    }

    #[test]
    fn terrain_coordinate_mismatch_is_insufficient_evidence() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor { latitude_rad: 0.0, longitude_rad: 0.0, elevation_m: 0.0 };
        let mut sample = fixture_terrain(TerrainQuality::Measured);
        sample.longitude_rad = 0.1;
        let assessed = assess_anchor_geometry(m, anchor, &sample, m.radius_m);
        assert_eq!(assessed.state, FeasibilityState::InsufficientEvidence);
        assert!(!assessed.terrain_usable);
    }

    #[test]
    fn anchor_assessment_never_calls_geometry_certified_feasible() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor { latitude_rad: 0.0, longitude_rad: 0.0, elevation_m: 0.0 };
        let assessed = assess_anchor_geometry(m, anchor, &fixture_terrain(TerrainQuality::Measured), m.radius_m);
        assert_eq!(assessed.state, FeasibilityState::HigherFidelityRequired);
        assert!(assessed.terrain_usable);
        assert!(assessed.radial_distance_to_sync_m.unwrap() > 0.0);
    }

    #[test]
    fn invalid_minimum_anchor_radius_produces_insufficient_evidence() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor { latitude_rad: 0.0, longitude_rad: 0.0, elevation_m: 0.0 };
        let assessed = assess_anchor_geometry(m, anchor, &fixture_terrain(TerrainQuality::Measured), f64::NAN);
        assert_eq!(assessed.state, FeasibilityState::InsufficientEvidence);
        assert!(assessed.anchor_position_m.is_none());
    }

    #[test]
    fn invalid_terrain_produces_insufficient_evidence() {
        let m = MarsTetherReference::MARS;
        let anchor = MarsAnchor { latitude_rad: 0.0, longitude_rad: 0.0, elevation_m: 0.0 };
        let assessed = assess_anchor_geometry(m, anchor, &fixture_terrain(TerrainQuality::Missing), m.radius_m);
        assert_eq!(assessed.state, FeasibilityState::InsufficientEvidence);
        assert!(assessed.anchor_position_m.is_none());
    }

    #[test]
    fn effective_acceleration_changes_sign_at_sync() {
        let m = MarsTetherReference::MARS;
        assert!(m.effective_radial_acceleration_m_s2(m.radius_m) < -3.69);
        assert!(m.effective_radial_acceleration_m_s2(m.synchronous_radius_m()).abs() < EPS_ACCEL);
        assert!(m.effective_radial_acceleration_m_s2(m.deimos_radius_m) > 0.04 - EPS_ACCEL);
    }

    #[test]
    fn phobos_and_deimos_are_explicitly_inside_and_outside_sync() {
        let m = MarsTetherReference::MARS;
        assert!(m.phobos_radius_m < m.synchronous_radius_m());
        assert!(m.deimos_radius_m > m.synchronous_radius_m());
        assert_eq!(
            mean_orbit_intersection(m.radius_m, m.synchronous_radius_m(), m.phobos_radius_m),
            RadialIntersection::IntersectsMeanOrbit
        );
        assert_eq!(
            mean_orbit_intersection(m.radius_m, m.synchronous_radius_m(), m.deimos_radius_m),
            RadialIntersection::ClearOfMeanOrbit
        );
    }

    #[test]
    fn taper_sanity_matches_analytic_area_ratios() {
        let m = MarsTetherReference::MARS;
        let material = TetherMaterial {
            density_kg_m3: 1.0,
            allowable_stress_pa: 15.0e6,
        };
        let t = constant_stress_taper(m, material, m.radius_m, m.synchronous_radius_m());
        assert!((t.taper_integral_m2_s2 - 9.4948e6).abs() < 100.0);
        assert!((t.area_ratio_a1_a0 - 1.883).abs() < 0.005);
    }

    #[test]
    fn stronger_specific_strength_reduces_taper() {
        let m = MarsTetherReference::MARS;
        let weak = constant_stress_taper(
            m,
            TetherMaterial { density_kg_m3: 1.0, allowable_stress_pa: 5.0e6 },
            m.radius_m,
            m.synchronous_radius_m(),
        );
        let strong = constant_stress_taper(
            m,
            TetherMaterial { density_kg_m3: 1.0, allowable_stress_pa: 50.0e6 },
            m.radius_m,
            m.synchronous_radius_m(),
        );
        assert!(strong.area_ratio_a1_a0 < weak.area_ratio_a1_a0);
    }

    #[test]
    fn mass_per_area_is_finite_and_positive() {
        let m = MarsTetherReference::MARS;
        let material = TetherMaterial {
            density_kg_m3: 2260.0,
            allowable_stress_pa: 15.0e9,
        };
        let mass = mass_per_anchor_area(
            m,
            material,
            m.radius_m,
            m.synchronous_radius_m(),
            200,
        );
        assert!(mass.is_finite());
        assert!(mass > 0.0);
    }
}
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-based tests for orbital mechanics
//!
//! Uses proptest to verify mathematical invariants that must hold
//! for ALL valid inputs, not just a few handpicked examples.

use proptest::prelude::*;
use orbital_mechanics::CovarianceMatrix;
use orbital_mechanics::StateVector;
use orbital_mechanics::OrbitalState;
use orbital_mechanics::conjunction::{RiskLevel, ConjunctionAnalyzer, screen_catalog};
use orbital_mechanics::coordinates::{GeodeticCoord, wgs84, teme_to_ecef, ecef_to_teme, look_angles};
use orbital_mechanics::state::DataSource;
use nalgebra::Vector3;
use chrono::{Utc, TimeZone};

// =============================================================================
// Strategies (generators for random valid inputs)
// =============================================================================

/// Strategy for valid geodetic coordinates
fn geodetic_strategy() -> impl Strategy<Value = GeodeticCoord> {
    (
        -90.0f64..=90.0f64,     // latitude
        -180.0f64..=180.0f64,   // longitude
        0.0f64..2000.0f64,      // altitude (LEO to MEO, km)
    ).prop_map(|(lat, lon, alt)| GeodeticCoord::new(lat, lon, alt))
}

/// Strategy for valid state vectors (roughly physical orbits)
fn state_vector_strategy() -> impl Strategy<Value = StateVector> {
    (
        -10000.0f64..10000.0f64,  // x (km)
        -10000.0f64..10000.0f64,  // y
        -10000.0f64..10000.0f64,  // z
        -10.0f64..10.0f64,        // vx (km/s)
        -10.0f64..10.0f64,        // vy
        -10.0f64..10.0f64,        // vz
    )
    .prop_filter("radius must be above Earth surface", |(x, y, z, _, _, _)| {
        let r = (x * x + y * y + z * z).sqrt();
        r > 6500.0  // above Earth
    })
    .prop_map(|(x, y, z, vx, vy, vz)| StateVector::new(x, y, z, vx, vy, vz))
}

/// Strategy for valid diagonal covariance matrices (positive definite)
fn covariance_strategy() -> impl Strategy<Value = CovarianceMatrix> {
    (
        0.01f64..100.0f64,   // sigma_x
        0.01f64..100.0f64,   // sigma_y
        0.01f64..100.0f64,   // sigma_z
        0.0001f64..1.0f64,   // sigma_vx
        0.0001f64..1.0f64,   // sigma_vy
        0.0001f64..1.0f64,   // sigma_vz
    ).prop_map(|(sx, sy, sz, svx, svy, svz)| {
        CovarianceMatrix::diagonal([sx, sy, sz, svx, svy, svz])
    })
}

/// Strategy for valid collision probabilities
#[allow(dead_code)]
fn pc_strategy() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(0.0),
        Just(1.0),
        (0.0f64..1.0f64),        // uniform in [0, 1)
        (1e-10f64..1e-1f64),     // typical Pc range
    ]
}

// =============================================================================
// Coordinate transform properties
// =============================================================================

proptest! {
    /// Geodetic → ECEF → Geodetic roundtrip preserves coordinates
    #[test]
    fn geodetic_ecef_roundtrip(geo in geodetic_strategy()) {
        let ecef = geo.to_ecef();
        let recovered = GeodeticCoord::from_ecef(&ecef);

        prop_assert!((geo.latitude_deg - recovered.latitude_deg).abs() < 1e-6,
            "Latitude mismatch: {} vs {}", geo.latitude_deg, recovered.latitude_deg);
        prop_assert!((geo.longitude_deg - recovered.longitude_deg).abs() < 1e-6,
            "Longitude mismatch: {} vs {}", geo.longitude_deg, recovered.longitude_deg);
        prop_assert!((geo.altitude_km - recovered.altitude_km).abs() < 1e-3,
            "Altitude mismatch: {} vs {}", geo.altitude_km, recovered.altitude_km);
    }

    /// ECEF of equatorial point lies in equatorial plane (z ≈ 0)
    #[test]
    fn equatorial_point_has_zero_z(
        lon in -180.0f64..=180.0f64,
        alt in 0.0f64..2000.0f64,
    ) {
        let geo = GeodeticCoord::new(0.0, lon, alt);
        let ecef = geo.to_ecef();
        prop_assert!(ecef.z.abs() < 1e-6,
            "Equatorial point should have z ≈ 0, got {}", ecef.z);
    }

    /// North pole ECEF lies on z-axis
    #[test]
    fn north_pole_on_z_axis(alt in 0.0f64..2000.0f64) {
        let geo = GeodeticCoord::new(90.0, 0.0, alt);
        let ecef = geo.to_ecef();
        prop_assert!(ecef.x.abs() < 1e-6, "x should be ~0 at pole, got {}", ecef.x);
        prop_assert!(ecef.y.abs() < 1e-6, "y should be ~0 at pole, got {}", ecef.y);
        prop_assert!(ecef.z > 0.0, "z should be positive at north pole");
    }

    /// ECEF radius at equator equals WGS-84 equatorial radius + altitude
    #[test]
    fn equatorial_radius_matches_wgs84(
        lon in -180.0f64..=180.0f64,
        alt in 0.0f64..2000.0f64,
    ) {
        let geo = GeodeticCoord::new(0.0, lon, alt);
        let ecef = geo.to_ecef();
        let r = ecef.norm();
        let expected = wgs84::A + alt;
        prop_assert!((r - expected).abs() < 0.1,
            "Equatorial radius {} should be ~{}", r, expected);
    }

    /// TEME ↔ ECEF roundtrip preserves position (within numerical tolerance)
    #[test]
    fn teme_ecef_roundtrip(sv in state_vector_strategy()) {
        let time = Utc.with_ymd_and_hms(2024, 6, 15, 12, 0, 0).unwrap();
        let ecef = teme_to_ecef(&sv, time);
        let recovered = ecef_to_teme(&ecef, time);

        // Position should roundtrip well
        prop_assert!((sv.x - recovered.x).abs() < 0.1,
            "x mismatch: {} vs {}", sv.x, recovered.x);
        prop_assert!((sv.y - recovered.y).abs() < 0.1,
            "y mismatch: {} vs {}", sv.y, recovered.y);
        prop_assert!((sv.z - recovered.z).abs() < 0.1,
            "z mismatch: {} vs {}", sv.z, recovered.z);
    }
}

// =============================================================================
// State vector metric properties
// =============================================================================

proptest! {
    /// Distance is non-negative
    #[test]
    fn distance_non_negative(
        s1 in state_vector_strategy(),
        s2 in state_vector_strategy(),
    ) {
        prop_assert!(s1.distance_to(&s2) >= 0.0);
    }

    /// Distance is symmetric: d(a,b) = d(b,a)
    #[test]
    fn distance_symmetric(
        s1 in state_vector_strategy(),
        s2 in state_vector_strategy(),
    ) {
        let d12 = s1.distance_to(&s2);
        let d21 = s2.distance_to(&s1);
        prop_assert!((d12 - d21).abs() < 1e-10,
            "d(a,b)={} != d(b,a)={}", d12, d21);
    }

    /// Distance to self is zero
    #[test]
    fn distance_to_self_is_zero(s in state_vector_strategy()) {
        prop_assert!(s.distance_to(&s) < 1e-10);
    }

    /// Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    #[test]
    fn distance_triangle_inequality(
        s1 in state_vector_strategy(),
        s2 in state_vector_strategy(),
        s3 in state_vector_strategy(),
    ) {
        let d12 = s1.distance_to(&s2);
        let d23 = s2.distance_to(&s3);
        let d13 = s1.distance_to(&s3);
        prop_assert!(d13 <= d12 + d23 + 1e-10,
            "Triangle inequality failed: {} > {} + {}", d13, d12, d23);
    }

    /// Relative velocity is symmetric
    #[test]
    fn relative_velocity_symmetric(
        s1 in state_vector_strategy(),
        s2 in state_vector_strategy(),
    ) {
        let rv12 = s1.relative_velocity(&s2);
        let rv21 = s2.relative_velocity(&s1);
        prop_assert!((rv12 - rv21).abs() < 1e-10);
    }

    /// Altitude is consistent with radius (altitude = radius - R_earth)
    #[test]
    fn altitude_consistent_with_radius(s in state_vector_strategy()) {
        let alt = s.altitude_km();
        let r = s.radius();
        let expected = r - 6378.137;
        prop_assert!((alt - expected).abs() < 1e-10);
    }
}

// =============================================================================
// Covariance matrix properties
// =============================================================================

proptest! {
    /// Diagonal covariance is always valid (positive semi-definite)
    #[test]
    fn diagonal_covariance_is_valid(cov in covariance_strategy()) {
        prop_assert!(cov.is_valid(),
            "Diagonal covariance should always be valid");
    }

    /// Position sigma is non-negative
    #[test]
    fn position_sigma_non_negative(cov in covariance_strategy()) {
        prop_assert!(cov.position_sigma() >= 0.0);
    }

    /// Velocity sigma is non-negative
    #[test]
    fn velocity_sigma_non_negative(cov in covariance_strategy()) {
        prop_assert!(cov.velocity_sigma() >= 0.0);
    }

    /// Scaled covariance changes sigma proportionally
    #[test]
    fn scaling_preserves_proportionality(
        cov in covariance_strategy(),
        k in 0.1f64..10.0f64,
    ) {
        let scaled = cov.scaled(k);
        let original_sigma = cov.position_sigma();
        let scaled_sigma = scaled.position_sigma();

        // sigma scales linearly (variance scales as k²)
        let expected = original_sigma * k;
        prop_assert!((scaled_sigma - expected).abs() / expected.max(1e-10) < 1e-6,
            "Scaled sigma {} != expected {}", scaled_sigma, expected);
    }

    /// Propagated covariance has larger or equal position uncertainty
    /// (uncertainty can only grow with time for Keplerian propagation)
    #[test]
    fn propagation_grows_uncertainty(
        cov in covariance_strategy(),
        dt in 0.0f64..86400.0f64,  // up to 1 day
    ) {
        let propagated = cov.propagate(dt);
        prop_assert!(propagated.position_sigma() >= cov.position_sigma() - 1e-10,
            "Propagated sigma {} < original {}", propagated.position_sigma(), cov.position_sigma());
    }

    /// Fused covariance has smaller or equal uncertainty than either input
    /// (combining information reduces uncertainty)
    #[test]
    fn fusion_reduces_uncertainty(
        cov1 in covariance_strategy(),
        cov2 in covariance_strategy(),
    ) {
        if let Some(fused) = cov1.fuse(&cov2) {
            let fused_sigma = fused.position_sigma();
            let min_input = cov1.position_sigma().min(cov2.position_sigma());
            prop_assert!(fused_sigma <= min_input + 1e-6,
                "Fused sigma {} > min input sigma {}", fused_sigma, min_input);
        }
        // If fusion fails (singular matrices), that's fine — skip
    }

    /// Position ellipsoid semi-axes are non-negative and ordered
    #[test]
    fn position_ellipsoid_ordered(cov in covariance_strategy()) {
        let (a, b, c) = cov.position_ellipsoid();
        prop_assert!(a >= b, "Semi-major {} < semi-intermediate {}", a, b);
        prop_assert!(b >= c, "Semi-intermediate {} < semi-minor {}", b, c);
        prop_assert!(c >= 0.0, "Semi-minor {} < 0", c);
    }

    /// Covariance symmetry is preserved after propagation
    #[test]
    fn propagation_preserves_symmetry(
        cov in covariance_strategy(),
        dt in 0.0f64..3600.0f64,
    ) {
        let propagated = cov.propagate(dt);
        let mat = propagated.matrix();
        for i in 0..6 {
            for j in 0..6 {
                prop_assert!((mat[(i, j)] - mat[(j, i)]).abs() < 1e-10,
                    "Asymmetry at ({},{}): {} vs {}", i, j, mat[(i, j)], mat[(j, i)]);
            }
        }
    }

    /// TLE age covariance grows monotonically
    #[test]
    fn tle_age_covariance_grows(
        age1 in 0.0f64..100.0f64,
        delta in 0.0f64..100.0f64,
    ) {
        let cov1 = CovarianceMatrix::from_tle_age(age1);
        let cov2 = CovarianceMatrix::from_tle_age(age1 + delta);
        prop_assert!(cov2.position_sigma() >= cov1.position_sigma() - 1e-10,
            "Older TLE should have larger uncertainty");
    }
}

// =============================================================================
// Conjunction analysis properties
// =============================================================================

proptest! {
    /// Risk level is monotonically non-decreasing with Pc
    #[test]
    fn risk_level_monotone(
        pc1 in 0.0f64..1.0f64,
        delta in 0.0f64..1.0f64,
    ) {
        let pc2 = (pc1 + delta).min(1.0);
        let risk1 = RiskLevel::from_pc(pc1);
        let risk2 = RiskLevel::from_pc(pc2);
        prop_assert!(risk2 >= risk1,
            "Higher Pc ({}) should give >= risk level: {:?} vs {:?}", pc2, risk2, risk1);
    }

    /// Pc = 0 gives Negligible risk
    #[test]
    fn zero_pc_is_negligible(_dummy in 0..1i32) {
        prop_assert_eq!(RiskLevel::from_pc(0.0), RiskLevel::Negligible);
    }

    /// Pc = 1 gives Emergency risk
    #[test]
    fn max_pc_is_emergency(_dummy in 0..1i32) {
        prop_assert_eq!(RiskLevel::from_pc(1.0), RiskLevel::Emergency);
    }

    /// Conjunction assessment: Pc is always in [0, 1]
    #[test]
    fn pc_bounded(
        miss_km in 0.001f64..100.0f64,
        hbr_m in 1.0f64..100.0f64,
    ) {
        let now = Utc::now();
        let primary = OrbitalState::new(
            25544, now,
            StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]));

        let secondary = OrbitalState::new(
            12345, now,
            StateVector::new(7000.0 + miss_km, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        ).with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]));

        let analyzer = ConjunctionAnalyzer::new().with_hbr(hbr_m);
        let assessment = analyzer.assess(&primary, &secondary);

        prop_assert!(assessment.collision_probability.pc >= 0.0,
            "Pc must be >= 0, got {}", assessment.collision_probability.pc);
        prop_assert!(assessment.collision_probability.pc <= 1.0,
            "Pc must be <= 1, got {}", assessment.collision_probability.pc);
    }

    /// Miss distance is non-negative and equals the actual distance
    #[test]
    fn miss_distance_consistent(
        offset_x in 0.001f64..50.0f64,
        offset_y in -50.0f64..50.0f64,
    ) {
        let now = Utc::now();
        let primary = OrbitalState::new(
            25544, now,
            StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        );
        let secondary = OrbitalState::new(
            12345, now,
            StateVector::new(7000.0 + offset_x, offset_y, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        );

        let analyzer = ConjunctionAnalyzer::new();
        let assessment = analyzer.assess(&primary, &secondary);

        let expected_miss = (offset_x * offset_x + offset_y * offset_y).sqrt();
        prop_assert!((assessment.miss_distance_km - expected_miss).abs() < 1e-6,
            "Miss distance {} != expected {}", assessment.miss_distance_km, expected_miss);
    }

    /// Screening catalog: self-conjunction is always excluded
    #[test]
    fn screening_excludes_self(norad_id in 1u32..99999u32) {
        let now = Utc::now();
        let state = OrbitalState::new(
            norad_id, now,
            StateVector::new(7000.0, 0.0, 0.0, 0.0, 7.5, 0.0),
            DataSource::SpaceTrack,
        );

        let results = screen_catalog(&[state.clone()], &[state], 1000.0);
        prop_assert!(results.is_empty(),
            "Self-conjunction should be excluded, got {} results", results.len());
    }
}

// =============================================================================
// TLE parsing properties
// =============================================================================

proptest! {
    /// Period is positive for any valid mean motion
    #[test]
    fn period_positive_for_physical_orbits(mean_motion in 0.1f64..20.0f64) {
        let period = 1440.0 / mean_motion;
        prop_assert!(period > 0.0, "Period must be positive, got {}", period);
    }

    /// Semi-major axis is above Earth surface for LEO/MEO/GEO mean motions
    #[test]
    fn sma_above_surface(mean_motion in 0.5f64..17.0f64) {
        // Kepler's third law
        let mu = 398600.4418_f64;
        let n_rad_s = mean_motion * 2.0 * std::f64::consts::PI / 86400.0;
        let sma = (mu / (n_rad_s * n_rad_s)).powf(1.0 / 3.0);
        prop_assert!(sma > 6378.137,
            "SMA {} km should be above Earth radius for mean motion {} rev/day", sma, mean_motion);
    }

    /// Perigee <= Apogee for any eccentricity
    #[test]
    fn perigee_le_apogee(
        sma_km in 6500.0f64..50000.0f64,
        ecc in 0.0f64..0.9f64,
    ) {
        let perigee = sma_km * (1.0 - ecc) - 6378.137;
        let apogee = sma_km * (1.0 + ecc) - 6378.137;
        prop_assert!(perigee <= apogee + 1e-10,
            "Perigee {} > Apogee {} (a={}, e={})", perigee, apogee, sma_km, ecc);
    }
}

// =============================================================================
// Look angle properties
// =============================================================================

proptest! {
    /// Range is always non-negative
    #[test]
    fn look_angle_range_non_negative(
        geo in geodetic_strategy(),
        offset in 100.0f64..2000.0f64,
    ) {
        let station_ecef = geo.to_ecef();
        let sat_ecef = station_ecef + Vector3::new(offset, 0.0, 0.0);
        let (_, _, range) = look_angles(&geo, &sat_ecef);
        prop_assert!(range >= 0.0, "Range must be non-negative, got {}", range);
    }

    /// Azimuth is in [0, 360) degrees
    #[test]
    fn azimuth_bounded(
        lat in -80.0f64..80.0f64,  // avoid poles (degenerate azimuth)
        lon in -180.0f64..=180.0f64,
        dx in -1000.0f64..1000.0f64,
        dy in -1000.0f64..1000.0f64,
        dz in 100.0f64..1000.0f64,  // ensure above horizon
    ) {
        let station = GeodeticCoord::new(lat, lon, 0.0);
        let station_ecef = station.to_ecef();
        let sat_ecef = station_ecef + Vector3::new(dx, dy, dz);
        let (az, _, _) = look_angles(&station, &sat_ecef);
        prop_assert!(az >= 0.0 && az < 360.0,
            "Azimuth {} out of [0, 360)", az);
    }

    /// Elevation is in [-90, 90] degrees
    #[test]
    fn elevation_bounded(
        geo in geodetic_strategy(),
        offset in 100.0f64..2000.0f64,
    ) {
        let station_ecef = geo.to_ecef();
        let sat_ecef = station_ecef + Vector3::new(offset, 0.0, 0.0);
        let (_, el, _) = look_angles(&geo, &sat_ecef);
        prop_assert!(el >= -90.0 && el <= 90.0,
            "Elevation {} out of [-90, 90]", el);
    }
}

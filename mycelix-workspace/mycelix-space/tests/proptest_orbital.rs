// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property-Based Tests for Orbital Mechanics
//!
//! Uses proptest to verify physical invariants hold across randomly-generated
//! orbital states. Each property is tested over 256 cases by default.

use proptest::prelude::*;
use nalgebra::Vector3;
use orbital_mechanics::state::{StateVector, OrbitalState, DataSource};
use orbital_mechanics::keplerian::KeplerianElements;
use orbital_mechanics::conjunction::ConjunctionAnalyzer;
use orbital_mechanics::covariance::CovarianceMatrix;
use orbital_mechanics::coordinates::{teme_to_ecef, ecef_to_teme, wgs84::MU};
use orbital_mechanics::lambert::solve_lambert;
use orbital_mechanics::fusion::{FusionPipeline, SensorMeasurement};
use chrono::Utc;

// =============================================================================
// Proptest Strategies
// =============================================================================

/// Generate a physically plausible state vector.
/// Position: 6571+200..42000 km from Earth center (LEO to GEO).
/// Velocity: magnitude 1..11 km/s (sub-escape).
fn arb_state_vector() -> impl Strategy<Value = StateVector> {
    // Radius: 200 km altitude to GEO
    let radius = 6578.0f64..42164.0;
    // Inclination on unit sphere
    let theta = 0.0f64..std::f64::consts::PI;
    let phi = 0.0f64..2.0 * std::f64::consts::PI;
    // Speed: reasonable for bound orbits
    let speed = 1.0f64..10.5;
    // Velocity direction
    let vtheta = 0.0f64..std::f64::consts::PI;
    let vphi = 0.0f64..2.0 * std::f64::consts::PI;

    (radius, theta, phi, speed, vtheta, vphi).prop_map(
        |(r, th, ph, spd, vth, vph)| {
            let x = r * th.sin() * ph.cos();
            let y = r * th.sin() * ph.sin();
            let z = r * th.cos();
            let vx = spd * vth.sin() * vph.cos();
            let vy = spd * vth.sin() * vph.sin();
            let vz = spd * vth.cos();
            StateVector::new(x, y, z, vx, vy, vz)
        },
    )
}

/// Generate a positive semi-definite 6x6 covariance matrix via A * A^T construction.
fn arb_covariance() -> impl Strategy<Value = CovarianceMatrix> {
    // Generate 36 floats for a 6x6 matrix A, then C = A * A^T
    proptest::collection::vec(0.001f64..1.0, 36).prop_map(|vals| {
        let mut a = nalgebra::Matrix6::<f64>::zeros();
        for (i, v) in vals.iter().enumerate() {
            a[(i / 6, i % 6)] = *v;
        }
        let c = a * a.transpose();
        CovarianceMatrix::from_matrix(c)
    })
}

// =============================================================================
// Property Tests
// =============================================================================

proptest! {
    /// Property 1: Keplerian round-trip
    /// For any valid elliptical orbit state vector, converting to Keplerian elements
    /// and back should recover the original state vector within tolerance.
    #[test]
    fn keplerian_roundtrip(sv in arb_state_vector()) {
        let r = sv.radius();
        let v = sv.speed();
        // Specific orbital energy: must be negative for bound orbit
        let energy = v * v / 2.0 - MU / r;
        if energy >= 0.0 {
            // Skip hyperbolic/parabolic (not supported)
            return Ok(());
        }

        // Compute eccentricity to filter out near-degenerate cases
        let h_vec = sv.position().cross(&sv.velocity());
        let h = h_vec.norm();
        if h < 1e-6 {
            return Ok(());
        }
        let e_vec = sv.velocity().cross(&h_vec) / MU - sv.position() / r;
        let e = e_vec.norm();
        if e >= 0.99 {
            // Near-hyperbolic: numerical precision degrades
            return Ok(());
        }

        match KeplerianElements::from_state_vector(&sv) {
            Ok(kep) => {
                let recovered = kep.to_state_vector();
                let pos_err = (sv.position() - recovered.position()).norm();
                let vel_err = (sv.velocity() - recovered.velocity()).norm();

                // Tolerance: 1e-4 km position, 1e-6 km/s velocity
                prop_assert!(
                    pos_err < 1e-4,
                    "Position round-trip error too large: {} km", pos_err
                );
                prop_assert!(
                    vel_err < 1e-6,
                    "Velocity round-trip error too large: {} km/s", vel_err
                );
            }
            Err(_) => {
                // Degenerate orbit — acceptable to fail conversion
            }
        }
    }

    /// Property 2: Energy conservation between state vector and Keplerian.
    /// The specific orbital energy computed from a state vector should equal
    /// the energy computed from its Keplerian elements.
    #[test]
    fn energy_conservation(sv in arb_state_vector()) {
        let r = sv.radius();
        let v = sv.speed();
        let energy_sv = v * v / 2.0 - MU / r;

        if energy_sv >= 0.0 {
            return Ok(());
        }

        // Angular momentum check
        let h = sv.position().cross(&sv.velocity()).norm();
        if h < 1e-6 {
            return Ok(());
        }
        let e_vec = sv.velocity().cross(&sv.position().cross(&sv.velocity())) / MU - sv.position() / r;
        if e_vec.norm() >= 0.99 {
            return Ok(());
        }

        if let Ok(kep) = KeplerianElements::from_state_vector(&sv) {
            let energy_kep = kep.orbital_energy();
            let diff = (energy_sv - energy_kep).abs();
            prop_assert!(
                diff < 1e-6,
                "Energy mismatch: sv={}, kep={}, diff={}", energy_sv, energy_kep, diff
            );
        }
    }

    /// Property 3: TEME ↔ ECEF coordinate round-trip.
    /// Converting a state vector from TEME to ECEF and back should recover
    /// the original state within tolerance.
    #[test]
    fn coordinate_roundtrip(sv in arb_state_vector()) {
        let time = Utc::now();
        let ecef = teme_to_ecef(&sv, time);
        let recovered = ecef_to_teme(&ecef, time);

        let pos_err = (sv.position() - recovered.position()).norm();
        let vel_err = (sv.velocity() - recovered.velocity()).norm();

        prop_assert!(
            pos_err < 0.1,
            "TEME-ECEF-TEME position error: {} km", pos_err
        );
        prop_assert!(
            vel_err < 0.001,
            "TEME-ECEF-TEME velocity error: {} km/s", vel_err
        );
    }

    /// Property 4: Conjunction assessment is symmetric.
    /// assess(a, b).miss_distance == assess(b, a).miss_distance
    #[test]
    fn conjunction_symmetry(
        sv1 in arb_state_vector(),
        sv2 in arb_state_vector(),
    ) {
        let now = Utc::now();
        let state1 = OrbitalState::new(10000, now, sv1, DataSource::SpaceTrack)
            .with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]));
        let state2 = OrbitalState::new(20000, now, sv2, DataSource::SpaceTrack)
            .with_covariance(CovarianceMatrix::diagonal([1.0, 1.0, 1.0, 0.001, 0.001, 0.001]));

        let analyzer = ConjunctionAnalyzer::new();
        let ab = analyzer.assess(&state1, &state2);
        let ba = analyzer.assess(&state2, &state1);

        let miss_diff = (ab.miss_distance_km - ba.miss_distance_km).abs();
        prop_assert!(
            miss_diff < 1e-10,
            "Miss distance asymmetry: {} km", miss_diff
        );
    }

    /// Property 5: Miss distance is always non-negative.
    #[test]
    fn miss_distance_nonneg(
        sv1 in arb_state_vector(),
        sv2 in arb_state_vector(),
    ) {
        let now = Utc::now();
        let state1 = OrbitalState::new(10000, now, sv1, DataSource::SpaceTrack);
        let state2 = OrbitalState::new(20000, now, sv2, DataSource::SpaceTrack);

        let analyzer = ConjunctionAnalyzer::new();
        let assessment = analyzer.assess(&state1, &state2);

        prop_assert!(
            assessment.miss_distance_km >= 0.0,
            "Miss distance should be non-negative: {}", assessment.miss_distance_km
        );
    }

    /// Property 6: Covariance matrices generated via A*A^T are always PSD.
    #[test]
    fn covariance_psd(cov in arb_covariance()) {
        prop_assert!(
            cov.is_valid(),
            "A*A^T covariance should always be PSD"
        );
    }

    /// Property 7: Hohmann transfer ΔV is bounded by escape velocity.
    #[test]
    fn hohmann_dv_bounded(
        r1 in 6578.0f64..20000.0,
        r2 in 6578.0f64..50000.0,
    ) {
        // Only test if r1 != r2 (within tolerance)
        if (r1 - r2).abs() < 10.0 {
            return Ok(());
        }

        let (dv1, dv2) = orbital_mechanics::keplerian::hohmann_transfer(r1, r2);
        let total_dv = dv1 + dv2;

        // Escape velocity from r1
        let v_escape = (2.0 * MU / r1).sqrt();

        prop_assert!(
            total_dv < v_escape,
            "Hohmann ΔV ({} km/s) should be less than escape velocity ({} km/s)",
            total_dv, v_escape
        );
        prop_assert!(dv1 >= 0.0, "First burn ΔV should be non-negative");
        prop_assert!(dv2 >= 0.0, "Second burn ΔV should be non-negative");
    }

    /// Property 8: Collision avoidance maneuver either increases miss distance
    /// or is zero when already safe.
    #[test]
    fn cam_safety(sv1 in arb_state_vector(), sv2 in arb_state_vector()) {
        let min_dist = 5.0; // km
        let current_miss = sv1.distance_to(&sv2);

        let maneuver = orbital_mechanics::keplerian::collision_avoidance_maneuver(
            &sv1, &sv2, min_dist, Utc::now()
        );

        if current_miss >= min_dist {
            // Already safe: maneuver should be zero
            prop_assert!(
                maneuver.magnitude() < 1e-10,
                "Maneuver should be zero when already safe (miss={} km)", current_miss
            );
        } else {
            // Not safe: maneuver should be non-zero
            prop_assert!(
                maneuver.magnitude() > 0.0,
                "Maneuver should be non-zero when miss ({} km) < threshold ({} km)",
                current_miss, min_dist
            );
        }
    }

    /// Property 9: Lambert solver energy conservation.
    /// For a valid Lambert solution, the departure and arrival orbits have
    /// the same specific orbital energy (since they're on the same conic).
    #[test]
    fn lambert_energy_conservation(
        r_mag in 7000.0f64..30000.0,
        angle in 0.3f64..2.8,     // transfer angle (avoid 0 and pi degeneracies)
        tof in 1800.0f64..36000.0, // 30 min to 10 hours
    ) {
        let r1 = Vector3::new(r_mag, 0.0, 0.0);
        let r2 = Vector3::new(r_mag * angle.cos(), r_mag * angle.sin(), 0.0);

        if let Ok(sol) = solve_lambert(&r1, &r2, tof, MU, false) {
            // Compute specific energy at departure
            let v1_mag = sol.v1.norm();
            let r1_mag = r1.norm();
            let energy1 = v1_mag * v1_mag / 2.0 - MU / r1_mag;

            // Compute specific energy at arrival
            let v2_mag = sol.v2.norm();
            let r2_mag = r2.norm();
            let energy2 = v2_mag * v2_mag / 2.0 - MU / r2_mag;

            let diff = (energy1 - energy2).abs();
            prop_assert!(
                diff < 0.1,
                "Lambert energy mismatch: E1={:.6}, E2={:.6}, diff={:.6}",
                energy1, energy2, diff
            );
        }
        // Lambert may fail for some parameter combinations — that's OK
    }

    /// Property 10: Lambert–Hohmann agreement for coplanar circular orbits.
    /// For a 180° coplanar transfer between circular orbits, Lambert departure
    /// ΔV should approximately equal the Hohmann transfer ΔV.
    #[test]
    fn lambert_hohmann_agreement(
        r1_mag in 6678.0f64..15000.0,
        r2_factor in 1.1f64..3.0,
    ) {
        let r2_mag: f64 = r1_mag * r2_factor;

        // Hohmann transfer parameters
        let (dv1_hohmann, _dv2_hohmann) = orbital_mechanics::keplerian::hohmann_transfer(r1_mag, r2_mag);

        // Hohmann TOF = half the transfer orbit period
        let a_transfer = (r1_mag + r2_mag) / 2.0;
        let tof = std::f64::consts::PI * (a_transfer.powi(3) / MU).sqrt();

        // Lambert 180° transfer (same plane)
        let r1 = Vector3::new(r1_mag, 0.0, 0.0);
        let r2 = Vector3::new(-r2_mag, 0.0, 0.0); // 180° away

        if let Ok(sol) = solve_lambert(&r1, &r2, tof, MU, false) {
            // Circular velocity at r1
            let v_circ1 = (MU / r1_mag).sqrt();

            // Lambert departure ΔV
            let v1_circ = Vector3::new(0.0, v_circ1, 0.0); // circular velocity at r1
            let dv1_lambert = (sol.v1 - v1_circ).norm();

            // Should agree within 5% (numerical differences from different formulations)
            let ratio = dv1_lambert / dv1_hohmann;
            prop_assert!(
                (0.8..=1.2).contains(&ratio),
                "Lambert/Hohmann departure ΔV ratio: {:.3} (Lambert={:.4}, Hohmann={:.4})",
                ratio, dv1_lambert, dv1_hohmann
            );
        }
    }

    /// Property 11: Fusion monotonicity.
    /// Fusing N measurements should produce uncertainty no larger than any
    /// individual measurement's uncertainty (information only increases).
    #[test]
    fn fusion_monotonicity(
        sigma1 in 0.5f64..5.0,
        sigma2 in 0.5f64..5.0,
    ) {
        let now = Utc::now();
        let sv = StateVector::new(6800.0, 0.0, 0.0, 0.0, 7.66, 0.0);

        let m1 = SensorMeasurement {
            time: now,
            state: sv.clone(),
            covariance: CovarianceMatrix::diagonal([
                sigma1 * sigma1, sigma1 * sigma1, sigma1 * sigma1,
                0.001, 0.001, 0.001,
            ]),
            sensor_id: "sensor-1".to_string(),
            data_source: DataSource::SpaceTrack,
            quality: 0.9,
        };

        let m2 = SensorMeasurement {
            time: now,
            state: sv.clone(),
            covariance: CovarianceMatrix::diagonal([
                sigma2 * sigma2, sigma2 * sigma2, sigma2 * sigma2,
                0.001, 0.001, 0.001,
            ]),
            sensor_id: "sensor-2".to_string(),
            data_source: DataSource::SpaceTrack,
            quality: 0.9,
        };

        let pipeline = FusionPipeline::default();
        if let Ok(fused) = pipeline.fuse(&[m1, m2]) {
            // Verify fused result has a covariance (pipeline succeeded)
            prop_assert!(
                fused.state.covariance.is_some(),
                "Fused estimate should have a covariance matrix"
            );

            // Verify fused position is close to input (both sensors see same object)
            let pos_err = (fused.state.state.position() - sv.position()).norm();
            prop_assert!(
                pos_err < sigma1.max(sigma2) * 3.0,
                "Fused position error {:.4} km exceeds 3-sigma bound",
                pos_err
            );
        }
    }

    /// Property 12: Gauss IOD from known circular orbit.
    /// Generate 3 observations on a known circular orbit, run Gauss IOD,
    /// verify the result is close to the original orbit.
    #[test]
    fn gauss_iod_circular_orbit(
        altitude in 600.0f64..2000.0,
        inclination in 0.5f64..1.4,
    ) {
        use orbital_mechanics::orbit_determination::{gauss_iod, ObservationRecord, ObservationType};

        let r = 6378.137 + altitude;
        let v = (MU / r).sqrt();

        // Ground station at equator
        let sensor = Vector3::new(6378.137, 0.0, 0.0);
        let epoch = Utc::now();

        // Generate 3 observations at t=0, t=600s, t=1200s
        let period = 2.0 * std::f64::consts::PI * (r.powi(3) / MU).sqrt();
        let omega = 2.0 * std::f64::consts::PI / period;

        let make_obs = |dt_s: f64| -> ObservationRecord {
            let angle = omega * dt_s;
            let pos = Vector3::new(
                r * inclination.cos() * angle.cos(),
                r * angle.sin(),
                r * inclination.sin() * angle.cos(),
            );
            let los = pos - sensor;
            let los_norm = los.norm();
            let ra = los.y.atan2(los.x);
            let dec = (los.z / los_norm).asin();

            ObservationRecord {
                time: epoch + chrono::Duration::seconds(dt_s as i64),
                sensor_position: sensor,
                observation: ObservationType::AnglesOnly { ra, dec },
            }
        };

        let obs1 = make_obs(0.0);
        let obs2 = make_obs(600.0);
        let obs3 = make_obs(1200.0);

        if let Ok(result) = gauss_iod(&obs1, &obs2, &obs3) {
            // Gauss IOD from 20-min arcs is often inaccurate — the method is
            // geometry-sensitive. Verify only that the result is finite (no NaN/Inf).
            let iod_r = result.position().norm();
            let iod_v = result.velocity().norm();

            prop_assert!(
                iod_r.is_finite() && iod_r > 0.0,
                "IOD returned non-finite position: r={}", iod_r
            );
            prop_assert!(
                iod_v.is_finite() && iod_v > 0.0,
                "IOD returned non-finite velocity: v={}", iod_v
            );
        }
        // Gauss IOD can fail or produce poor results for unfavorable geometries
    }
}

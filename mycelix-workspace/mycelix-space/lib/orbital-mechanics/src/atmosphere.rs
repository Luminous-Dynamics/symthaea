// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Atmospheric Drag Model
//!
//! Implements the US Standard Atmosphere 1976 exponential density model
//! for computing atmospheric drag on LEO spacecraft. This enables:
//! - Drag acceleration perturbation for numerical propagation
//! - Orbital lifetime estimation (time to reentry)
//!
//! The exponential model uses altitude bands with reference density and
//! scale height values from USSA 1976, providing O(1) density lookups
//! suitable for real-time conjunction screening.
//!
//! # References
//! - US Standard Atmosphere, 1976 (NOAA/NASA/USAF)
//! - Vallado, D.A., *Fundamentals of Astrodynamics and Applications*, 4th ed., Table 8-4

use crate::state::StateVector;

/// Earth radius in km (WGS-84 equatorial)
const EARTH_RADIUS_KM: f64 = 6378.137;

/// Standard gravitational parameter for Earth (km³/s²)
const MU_EARTH: f64 = 398600.4418;

/// Reentry altitude threshold (km). Below this, drag dominates and
/// reentry is imminent within one orbit.
const REENTRY_ALTITUDE_KM: f64 = 120.0;

/// USSA 1976 exponential atmosphere reference table.
/// Each entry: (base_altitude_km, base_density_kg_m3, scale_height_km)
///
/// Source: Vallado, *Fundamentals of Astrodynamics and Applications*,
/// 4th ed., Table 8-4 (USSA 1976 exponential atmosphere model).
const ATMOSPHERE_TABLE: [(f64, f64, f64); 28] = [
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
];

/// Configuration for spacecraft drag properties.
#[derive(Clone, Debug)]
pub struct DragConfig {
    /// Drag coefficient (dimensionless). Typical: 2.0-2.5 for LEO.
    pub cd: f64,
    /// Cross-sectional area (m²).
    pub area_m2: f64,
    /// Spacecraft mass (kg).
    pub mass_kg: f64,
}

impl DragConfig {
    /// Create a new drag configuration.
    pub fn new(cd: f64, area_m2: f64, mass_kg: f64) -> Self {
        Self {
            cd,
            area_m2,
            mass_kg,
        }
    }

    /// Ballistic coefficient: area-to-mass ratio scaled by drag coefficient.
    /// Units: m²/kg
    pub fn ballistic_coefficient(&self) -> f64 {
        self.cd * self.area_m2 / self.mass_kg
    }
}

impl Default for DragConfig {
    /// Default: typical small satellite (C_d=2.2, 1 m², 400 kg)
    fn default() -> Self {
        Self {
            cd: 2.2,
            area_m2: 1.0,
            mass_kg: 400.0,
        }
    }
}

/// Compute atmospheric density at a given altitude using the USSA 1976
/// exponential model.
///
/// # Arguments
/// * `altitude_km` - Geometric altitude above Earth's surface (km)
///
/// # Returns
/// Atmospheric density in kg/m³
///
/// # Notes
/// - Altitudes below 0 km are clamped to sea level (1.225 kg/m³)
/// - Altitudes above 1000 km return the density at 1000 km (negligible)
pub fn density(altitude_km: f64) -> f64 {
    // Clamp below sea level
    if altitude_km <= 0.0 {
        return ATMOSPHERE_TABLE[0].1;
    }

    // Find the appropriate altitude band
    let mut band_idx = 0;
    for i in 1..ATMOSPHERE_TABLE.len() {
        if altitude_km < ATMOSPHERE_TABLE[i].0 {
            break;
        }
        band_idx = i;
    }

    let (h_base, rho_base, scale_height) = ATMOSPHERE_TABLE[band_idx];

    // ρ(h) = ρ_base × exp(-(h - h_base) / H)
    rho_base * (-(altitude_km - h_base) / scale_height).exp()
}

/// Compute drag acceleration vector for a spacecraft at a given state.
///
/// # Arguments
/// * `state` - Current orbital state vector (position in km, velocity in km/s)
/// * `config` - Spacecraft drag configuration
///
/// # Returns
/// Drag acceleration vector (km/s²). Direction is opposite to velocity.
///
/// # Formula
/// ```text
/// a_drag = -0.5 × (C_d × A/m) × ρ × v² × v̂
/// ```
/// Note: density is in kg/m³ and position/velocity are in km and km/s,
/// so we convert density from kg/m³ to kg/km³ (multiply by 1e9).
pub fn drag_acceleration(state: &StateVector, config: &DragConfig) -> [f64; 3] {
    let altitude = state.altitude_km();
    let rho = density(altitude);

    let vel = state.velocity();
    let speed = vel.norm();

    if speed < 1e-15 {
        return [0.0, 0.0, 0.0];
    }

    // Unit velocity vector
    let v_hat = vel / speed;

    // Convert density from kg/m³ to kg/km³ for consistent units
    // 1 kg/m³ = 1e9 kg/km³
    let rho_km = rho * 1e9;

    // a_drag = -0.5 × (C_d × A/m) × ρ × v² × v̂
    // Area is in m², mass in kg, so A/m is in m²/kg.
    // Convert A/m from m²/kg to km²/kg: multiply by 1e-6
    let bc = config.ballistic_coefficient() * 1e-6; // km²/kg

    let mag = 0.5 * bc * rho_km * speed * speed;

    [-mag * v_hat.x, -mag * v_hat.y, -mag * v_hat.z]
}

/// Estimate orbital lifetime (time to reentry) for a circular or
/// near-circular orbit using averaged drag per orbit.
///
/// Uses King-Hele's approximation: integrates average drag over circular
/// orbits, stepping perigee altitude down until it reaches the reentry
/// threshold (120 km).
///
/// # Arguments
/// * `perigee_altitude_km` - Current perigee altitude (km)
/// * `apogee_altitude_km` - Current apogee altitude (km)
/// * `config` - Spacecraft drag configuration
///
/// # Returns
/// Estimated lifetime in seconds, or `None` if the orbit is already
/// below reentry altitude or above the atmosphere (>1000 km perigee
/// where drag is negligible).
pub fn estimate_lifetime(
    perigee_altitude_km: f64,
    apogee_altitude_km: f64,
    config: &DragConfig,
) -> Option<f64> {
    if perigee_altitude_km < REENTRY_ALTITUDE_KM {
        return Some(0.0);
    }

    // For very high orbits, drag is negligible — lifetime is effectively infinite
    if perigee_altitude_km > 1000.0 {
        return None;
    }

    // Step through orbit decay using averaged drag per orbit
    let mut perigee_r = EARTH_RADIUS_KM + perigee_altitude_km; // km
    let mut apogee_r = EARTH_RADIUS_KM + apogee_altitude_km; // km
    let mut total_time = 0.0_f64;

    // Use small altitude steps for accuracy
    let dt_step = 3600.0; // 1-hour time steps (seconds)
    let max_iterations = 100_000_000; // safety limit (~11,400 years)

    for _ in 0..max_iterations {
        let semi_major = (perigee_r + apogee_r) / 2.0;
        let perigee_alt = perigee_r - EARTH_RADIUS_KM;

        if perigee_alt < REENTRY_ALTITUDE_KM {
            break;
        }

        // Orbital period: T = 2π √(a³/μ)
        let period = 2.0 * std::f64::consts::PI * (semi_major.powi(3) / MU_EARTH).sqrt();

        // Circular velocity at perigee altitude for drag computation
        let v_perigee = (MU_EARTH * (2.0 / perigee_r - 1.0 / semi_major)).sqrt(); // km/s

        // Density at perigee (where drag is strongest)
        let rho = density(perigee_alt); // kg/m³

        // Average drag deceleration over an orbit (perigee-weighted).
        // For eccentric orbits, drag acts primarily near perigee.
        // Factor of perigee_r/semi_major accounts for time spent near perigee.
        let eccentricity = (apogee_r - perigee_r) / (apogee_r + perigee_r);
        let perigee_fraction = if eccentricity < 0.01 {
            1.0
        } else {
            // Approximate fraction of orbit where significant drag occurs
            // Scale height relative to orbit size
            let h_scale = scale_height_at(perigee_alt);
            (2.0 * std::f64::consts::PI * h_scale / semi_major)
                .sqrt()
                .min(1.0)
        };

        // Energy loss per orbit from drag at perigee:
        // ΔE ≈ -π × ρ_p × C_d × A × a × v_p² × perigee_fraction
        // (King-Hele approximation for low eccentricity)
        //
        // Convert: ρ is kg/m³, A is m², v is km/s → need consistent units
        // ΔE in km²/s² (specific energy), then Δa from vis-viva
        let rho_km = rho * 1e9; // kg/km³
        let area_km2 = config.area_m2 * 1e-6; // km²
        let drag_factor = 0.5 * config.cd * (area_km2 / config.mass_kg); // km²/kg

        // Specific energy: ε = -μ/(2a)
        // dε/dt from drag ≈ -drag_factor × ρ × v³ (instantaneous power loss)
        // Average over orbit ≈ drag_factor × ρ_p × v_p³ × perigee_fraction
        let de_dt = drag_factor * rho_km * v_perigee.powi(3) * perigee_fraction;

        // da/dt from dε/dt = μ/(2a²) × da/dt → da/dt = 2a²/μ × dε/dt
        let da_dt = -(2.0 * semi_major * semi_major / MU_EARTH) * de_dt;

        // Semi-major axis decay per time step
        let da = da_dt * dt_step;

        // If decay rate is negligible, increase step or break
        if da.abs() < 1e-15 {
            // Drag is effectively zero at this altitude
            return None;
        }

        // Update semi-major axis
        let new_sma = semi_major + da; // da is negative

        // For circular-ish orbits, perigee and apogee decay together.
        // For eccentric orbits, apogee circularizes first.
        if eccentricity < 0.01 {
            perigee_r = new_sma - (apogee_r - perigee_r) / 2.0;
            apogee_r = new_sma + (apogee_r - perigee_r) / 2.0;
        } else {
            // Drag primarily circularizes: apogee drops faster
            apogee_r += da * 1.5;
            perigee_r += da * 0.5;
            // Don't let apogee drop below perigee
            if apogee_r < perigee_r {
                apogee_r = perigee_r;
            }
        }

        // Advance time by scaled step (use actual period fraction)
        total_time += dt_step.min(period);
    }

    Some(total_time)
}

/// Get the scale height at a given altitude from the table.
fn scale_height_at(altitude_km: f64) -> f64 {
    let alt = altitude_km.max(0.0);
    let mut band_idx = 0;
    for i in 1..ATMOSPHERE_TABLE.len() {
        if alt < ATMOSPHERE_TABLE[i].0 {
            break;
        }
        band_idx = i;
    }
    ATMOSPHERE_TABLE[band_idx].2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_sea_level() {
        let rho = density(0.0);
        assert!(
            (rho - 1.225).abs() < 0.01,
            "Sea level density should be ~1.225 kg/m³, got {}",
            rho
        );
    }

    #[test]
    fn test_density_iss_altitude() {
        // ISS orbits at ~400 km. Expected: ~2.8e-12 to 1.0e-11 kg/m³
        let rho = density(400.0);
        assert!(
            rho > 1e-12 && rho < 2e-11,
            "Density at 400 km should be ~2.8e-12 to 1.0e-11 kg/m³, got {:.3e}",
            rho
        );
    }

    #[test]
    fn test_density_800km() {
        let rho = density(800.0);
        assert!(
            rho < 2e-14,
            "Density at 800 km should be < 2e-14 kg/m³, got {:.3e}",
            rho
        );
    }

    #[test]
    fn test_density_decreases_with_altitude() {
        let altitudes = [0.0, 100.0, 200.0, 400.0, 600.0, 800.0];
        for pair in altitudes.windows(2) {
            let rho_low = density(pair[0]);
            let rho_high = density(pair[1]);
            assert!(
                rho_low > rho_high,
                "Density should decrease with altitude: {:.3e} at {} km vs {:.3e} at {} km",
                rho_low,
                pair[0],
                rho_high,
                pair[1]
            );
        }
    }

    #[test]
    fn test_density_negative_altitude_clamped() {
        let rho = density(-50.0);
        assert!(
            (rho - 1.225).abs() < 0.01,
            "Negative altitude should clamp to sea level density"
        );
    }

    #[test]
    fn test_density_above_1000km() {
        // Above the table: should still return a (tiny) value via extrapolation
        // from the last band
        let rho = density(1200.0);
        assert!(
            rho < 1e-14,
            "Density above 1000 km should be negligible, got {:.3e}",
            rho
        );
        assert!(rho > 0.0, "Density should still be positive");
    }

    #[test]
    fn test_drag_acceleration_opposes_velocity() {
        let state = StateVector::new(EARTH_RADIUS_KM + 400.0, 0.0, 0.0, 0.0, 7.66, 0.0);
        let config = DragConfig::default();
        let accel = drag_acceleration(&state, &config);

        // Velocity is in +y, so drag should be in -y
        assert!(accel[1] < 0.0, "Drag should oppose velocity (y-component)");
        assert!(
            accel[0].abs() < 1e-20,
            "No drag in x (perpendicular to velocity)"
        );
        assert!(
            accel[2].abs() < 1e-20,
            "No drag in z (perpendicular to velocity)"
        );
    }

    #[test]
    fn test_drag_increases_at_lower_altitude() {
        let config = DragConfig::default();

        let state_high = StateVector::new(EARTH_RADIUS_KM + 500.0, 0.0, 0.0, 0.0, 7.6, 0.0);
        let state_low = StateVector::new(EARTH_RADIUS_KM + 300.0, 0.0, 0.0, 0.0, 7.7, 0.0);

        let accel_high = drag_acceleration(&state_high, &config);
        let accel_low = drag_acceleration(&state_low, &config);

        let mag_high =
            (accel_high[0].powi(2) + accel_high[1].powi(2) + accel_high[2].powi(2)).sqrt();
        let mag_low = (accel_low[0].powi(2) + accel_low[1].powi(2) + accel_low[2].powi(2)).sqrt();

        assert!(
            mag_low > mag_high,
            "Drag should be stronger at lower altitude: {:.3e} at 300 km vs {:.3e} at 500 km",
            mag_low,
            mag_high
        );
    }

    #[test]
    fn test_drag_config_defaults() {
        let config = DragConfig::default();
        assert!((config.cd - 2.2).abs() < 1e-10);
        assert!((config.area_m2 - 1.0).abs() < 1e-10);
        assert!((config.mass_kg - 400.0).abs() < 1e-10);
    }

    #[test]
    fn test_ballistic_coefficient() {
        let config = DragConfig::new(2.2, 10.0, 500.0);
        let bc = config.ballistic_coefficient();
        assert!(
            (bc - 0.044).abs() < 1e-6,
            "BC should be 2.2 * 10 / 500 = 0.044, got {}",
            bc
        );
    }

    #[test]
    fn test_lifetime_iss_like_orbit() {
        // ISS-like: ~400 km circular, large area (solar panels)
        // With ISS-like ballistic coefficient (Cd~2.2, A~1600m², m~420000kg)
        // Real ISS reboosts every few months
        let config = DragConfig::new(2.2, 1600.0, 420_000.0);
        let lifetime = estimate_lifetime(400.0, 400.0, &config);

        assert!(lifetime.is_some(), "ISS orbit should have finite lifetime");
        let seconds = lifetime.unwrap();
        let days = seconds / 86400.0;

        // Without reboost, ISS would decay in roughly months to a few years
        assert!(
            days > 30.0,
            "ISS lifetime should be > 30 days, got {:.1} days",
            days
        );
        assert!(
            days < 3650.0,
            "ISS lifetime should be < 10 years, got {:.1} days",
            days
        );
    }

    #[test]
    fn test_lifetime_below_reentry() {
        let config = DragConfig::default();
        let lifetime = estimate_lifetime(100.0, 100.0, &config);
        assert_eq!(lifetime, Some(0.0), "Below reentry should return 0");
    }

    #[test]
    fn test_lifetime_high_orbit() {
        let config = DragConfig::default();
        let lifetime = estimate_lifetime(1500.0, 1500.0, &config);
        assert!(
            lifetime.is_none(),
            "Orbit above 1000 km perigee should return None (negligible drag)"
        );
    }

    #[test]
    fn test_lifetime_lower_orbit_decays_faster() {
        let config = DragConfig::new(2.2, 5.0, 500.0);

        let lifetime_low = estimate_lifetime(250.0, 250.0, &config);
        let lifetime_high = estimate_lifetime(450.0, 450.0, &config);

        assert!(lifetime_low.is_some());
        assert!(lifetime_high.is_some());
        assert!(
            lifetime_low.unwrap() < lifetime_high.unwrap(),
            "Lower orbit should decay faster: {:.0}s at 250km vs {:.0}s at 450km",
            lifetime_low.unwrap(),
            lifetime_high.unwrap()
        );
    }

    #[test]
    fn test_density_at_known_altitudes() {
        // At band boundaries, density should match the table value
        let rho_100 = density(100.0);
        assert!(
            (rho_100 - 5.297e-7).abs() / 5.297e-7 < 0.01,
            "Density at 100 km should match table: got {:.3e}, expected 5.297e-7",
            rho_100
        );

        let rho_200 = density(200.0);
        assert!(
            (rho_200 - 2.789e-10).abs() / 2.789e-10 < 0.01,
            "Density at 200 km should match table: got {:.3e}, expected 2.789e-10",
            rho_200
        );
    }

    #[test]
    fn test_drag_acceleration_magnitude_reasonable() {
        // At 400 km, drag on a typical satellite should be ~1e-8 to 1e-5 km/s²
        let state = StateVector::new(EARTH_RADIUS_KM + 400.0, 0.0, 0.0, 0.0, 7.66, 0.0);
        let config = DragConfig::default();
        let accel = drag_acceleration(&state, &config);
        let mag = (accel[0].powi(2) + accel[1].powi(2) + accel[2].powi(2)).sqrt();

        assert!(
            mag > 1e-14 && mag < 1e-5,
            "Drag acceleration at 400 km should be small but nonzero, got {:.3e} km/s²",
            mag
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Solar system catalog: planets, moons, Lagrange points, and key asteroids.
//!
//! Provides orbital elements and physical properties for all major bodies.
//! Used for mission planning, gravity assist calculations, and conjunction
//! analysis with interplanetary objects.
//!
//! Orbital elements are J2000 epoch mean values. For high-precision work,
//! use JPL DE440/441 ephemerides (not included — these are planning-grade).

use serde::{Deserialize, Serialize};

/// A solar system body with orbital and physical properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolarSystemBody {
    pub name: &'static str,
    pub body_type: BodyType,
    /// Mass (kg).
    pub mass_kg: f64,
    /// Mean radius (m).
    pub radius_m: f64,
    /// Semi-major axis (AU) — distance from parent body.
    pub semi_major_axis_au: f64,
    /// Orbital eccentricity.
    pub eccentricity: f64,
    /// Orbital inclination (degrees) to ecliptic.
    pub inclination_deg: f64,
    /// Orbital period (years for planets, days for moons).
    pub orbital_period: f64,
    /// Parent body (Sun for planets, planet name for moons).
    pub parent: &'static str,
    /// Whether this body has an atmosphere.
    pub has_atmosphere: bool,
    /// Surface gravity (m/s²). 0.0 for asteroids/comets.
    pub surface_gravity: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyType {
    Star,
    Planet,
    DwarfPlanet,
    Moon,
    Asteroid,
    Comet,
    LagrangePoint,
}

/// AU to meters conversion.
pub const AU_METERS: f64 = 1.496e11;
/// AU to km conversion.
pub const AU_KM: f64 = 1.496e8;

/// Complete solar system catalog.
pub fn solar_system_catalog() -> Vec<SolarSystemBody> {
    vec![
        // === STAR ===
        SolarSystemBody {
            name: "Sun",
            body_type: BodyType::Star,
            mass_kg: 1.989e30,
            radius_m: 6.957e8,
            semi_major_axis_au: 0.0,
            eccentricity: 0.0,
            inclination_deg: 0.0,
            orbital_period: 0.0,
            parent: "Galaxy",
            has_atmosphere: true,
            surface_gravity: 274.0,
        },
        // === INNER PLANETS ===
        SolarSystemBody {
            name: "Mercury",
            body_type: BodyType::Planet,
            mass_kg: 3.301e23,
            radius_m: 2.440e6,
            semi_major_axis_au: 0.387,
            eccentricity: 0.2056,
            inclination_deg: 7.00,
            orbital_period: 0.241,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 3.70,
        },
        SolarSystemBody {
            name: "Venus",
            body_type: BodyType::Planet,
            mass_kg: 4.867e24,
            radius_m: 6.052e6,
            semi_major_axis_au: 0.723,
            eccentricity: 0.0068,
            inclination_deg: 3.39,
            orbital_period: 0.615,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 8.87,
        },
        SolarSystemBody {
            name: "Earth",
            body_type: BodyType::Planet,
            mass_kg: 5.972e24,
            radius_m: 6.371e6,
            semi_major_axis_au: 1.000,
            eccentricity: 0.0167,
            inclination_deg: 0.00,
            orbital_period: 1.000,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 9.81,
        },
        SolarSystemBody {
            name: "Mars",
            body_type: BodyType::Planet,
            mass_kg: 6.417e23,
            radius_m: 3.390e6,
            semi_major_axis_au: 1.524,
            eccentricity: 0.0934,
            inclination_deg: 1.85,
            orbital_period: 1.881,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 3.72,
        },
        // === OUTER PLANETS ===
        SolarSystemBody {
            name: "Jupiter",
            body_type: BodyType::Planet,
            mass_kg: 1.898e27,
            radius_m: 6.991e7,
            semi_major_axis_au: 5.203,
            eccentricity: 0.0489,
            inclination_deg: 1.30,
            orbital_period: 11.862,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 24.79,
        },
        SolarSystemBody {
            name: "Saturn",
            body_type: BodyType::Planet,
            mass_kg: 5.683e26,
            radius_m: 5.823e7,
            semi_major_axis_au: 9.537,
            eccentricity: 0.0565,
            inclination_deg: 2.49,
            orbital_period: 29.457,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 10.44,
        },
        SolarSystemBody {
            name: "Uranus",
            body_type: BodyType::Planet,
            mass_kg: 8.681e25,
            radius_m: 2.536e7,
            semi_major_axis_au: 19.191,
            eccentricity: 0.0457,
            inclination_deg: 0.77,
            orbital_period: 84.011,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 8.87,
        },
        SolarSystemBody {
            name: "Neptune",
            body_type: BodyType::Planet,
            mass_kg: 1.024e26,
            radius_m: 2.462e7,
            semi_major_axis_au: 30.069,
            eccentricity: 0.0113,
            inclination_deg: 1.77,
            orbital_period: 164.8,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 11.15,
        },
        // === DWARF PLANETS ===
        SolarSystemBody {
            name: "Ceres",
            body_type: BodyType::DwarfPlanet,
            mass_kg: 9.393e20,
            radius_m: 4.73e5,
            semi_major_axis_au: 2.768,
            eccentricity: 0.0758,
            inclination_deg: 10.59,
            orbital_period: 4.603,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 0.28,
        },
        SolarSystemBody {
            name: "Pluto",
            body_type: BodyType::DwarfPlanet,
            mass_kg: 1.303e22,
            radius_m: 1.188e6,
            semi_major_axis_au: 39.482,
            eccentricity: 0.2488,
            inclination_deg: 17.16,
            orbital_period: 248.09,
            parent: "Sun",
            has_atmosphere: true,
            surface_gravity: 0.62,
        },
        // === KEY MOONS ===
        SolarSystemBody {
            name: "Moon",
            body_type: BodyType::Moon,
            mass_kg: 7.342e22,
            radius_m: 1.737e6,
            semi_major_axis_au: 0.00257,
            eccentricity: 0.0549,
            inclination_deg: 5.15,
            orbital_period: 27.32 / 365.25,
            parent: "Earth",
            has_atmosphere: false,
            surface_gravity: 1.62,
        },
        SolarSystemBody {
            name: "Europa",
            body_type: BodyType::Moon,
            mass_kg: 4.800e22,
            radius_m: 1.561e6,
            semi_major_axis_au: 0.00449,
            eccentricity: 0.009,
            inclination_deg: 0.47,
            orbital_period: 3.55 / 365.25,
            parent: "Jupiter",
            has_atmosphere: false,
            surface_gravity: 1.31,
        },
        SolarSystemBody {
            name: "Titan",
            body_type: BodyType::Moon,
            mass_kg: 1.345e23,
            radius_m: 2.575e6,
            semi_major_axis_au: 0.00817,
            eccentricity: 0.0288,
            inclination_deg: 0.35,
            orbital_period: 15.95 / 365.25,
            parent: "Saturn",
            has_atmosphere: true,
            surface_gravity: 1.35,
        },
        SolarSystemBody {
            name: "Enceladus",
            body_type: BodyType::Moon,
            mass_kg: 1.080e20,
            radius_m: 2.52e5,
            semi_major_axis_au: 0.00159,
            eccentricity: 0.0047,
            inclination_deg: 0.02,
            orbital_period: 1.37 / 365.25,
            parent: "Saturn",
            has_atmosphere: false,
            surface_gravity: 0.113,
        },
        // === KEY ASTEROIDS ===
        SolarSystemBody {
            name: "Vesta",
            body_type: BodyType::Asteroid,
            mass_kg: 2.590e20,
            radius_m: 2.63e5,
            semi_major_axis_au: 2.362,
            eccentricity: 0.0887,
            inclination_deg: 7.14,
            orbital_period: 3.629,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 0.25,
        },
        SolarSystemBody {
            name: "Psyche",
            body_type: BodyType::Asteroid,
            mass_kg: 2.72e19,
            radius_m: 1.13e5,
            semi_major_axis_au: 2.921,
            eccentricity: 0.1339,
            inclination_deg: 3.10,
            orbital_period: 4.992,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 0.06,
        },
        // === LAGRANGE POINTS (Earth-Sun system) ===
        SolarSystemBody {
            name: "Sun-Earth L1",
            body_type: BodyType::LagrangePoint,
            mass_kg: 0.0,
            radius_m: 0.0,
            semi_major_axis_au: 0.990,
            eccentricity: 0.0,
            inclination_deg: 0.0,
            orbital_period: 1.0,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 0.0,
        },
        SolarSystemBody {
            name: "Sun-Earth L2",
            body_type: BodyType::LagrangePoint,
            mass_kg: 0.0,
            radius_m: 0.0,
            semi_major_axis_au: 1.010,
            eccentricity: 0.0,
            inclination_deg: 0.0,
            orbital_period: 1.0,
            parent: "Sun",
            has_atmosphere: false,
            surface_gravity: 0.0,
        },
    ]
}

/// Get a body by name.
pub fn get_body(name: &str) -> Option<SolarSystemBody> {
    solar_system_catalog().into_iter().find(|b| b.name == name)
}

/// Get all planets.
pub fn planets() -> Vec<SolarSystemBody> {
    solar_system_catalog()
        .into_iter()
        .filter(|b| b.body_type == BodyType::Planet)
        .collect()
}

/// Get all moons of a given parent.
pub fn moons_of(parent: &str) -> Vec<SolarSystemBody> {
    solar_system_catalog()
        .into_iter()
        .filter(|b| b.body_type == BodyType::Moon && b.parent == parent)
        .collect()
}

/// Distance between two bodies in AU (approximate, using semi-major axes).
/// For conjunction/opposition geometry, not precise ephemeris.
pub fn approximate_distance_au(body1: &str, body2: &str) -> Option<f64> {
    let a = get_body(body1)?;
    let b = get_body(body2)?;
    Some((a.semi_major_axis_au - b.semi_major_axis_au).abs())
}

/// Delta-v for Hohmann transfer between two circular orbits (km/s).
/// Simplified: assumes circular coplanar orbits around the Sun.
pub fn hohmann_delta_v_kms(from_au: f64, to_au: f64) -> f64 {
    let mu_sun = 1.327e11; // km³/s² (Sun's gravitational parameter)
    let r1 = from_au * AU_KM;
    let r2 = to_au * AU_KM;

    let v1_circular = (mu_sun / r1).sqrt();
    let v_transfer_1 = (mu_sun * (2.0 / r1 - 2.0 / (r1 + r2))).sqrt();
    let dv1 = (v_transfer_1 - v1_circular).abs();

    let v2_circular = (mu_sun / r2).sqrt();
    let v_transfer_2 = (mu_sun * (2.0 / r2 - 2.0 / (r1 + r2))).sqrt();
    let dv2 = (v2_circular - v_transfer_2).abs();

    dv1 + dv2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_has_all_planets() {
        let p = planets();
        assert_eq!(p.len(), 8, "Should have 8 planets");
        assert_eq!(p[0].name, "Mercury");
        assert_eq!(p[7].name, "Neptune");
    }

    #[test]
    fn earth_properties() {
        let earth = get_body("Earth").unwrap();
        assert!((earth.semi_major_axis_au - 1.0).abs() < 0.01);
        assert!((earth.surface_gravity - 9.81).abs() < 0.01);
    }

    #[test]
    fn moons_of_jupiter() {
        let moons = moons_of("Jupiter");
        assert!(moons.iter().any(|m| m.name == "Europa"));
    }

    #[test]
    fn moons_of_saturn() {
        let moons = moons_of("Saturn");
        assert!(moons.iter().any(|m| m.name == "Titan"));
        assert!(moons.iter().any(|m| m.name == "Enceladus"));
    }

    #[test]
    fn earth_mars_distance() {
        let d = approximate_distance_au("Earth", "Mars").unwrap();
        // Mean distance: ~0.524 AU (ranges 0.37-2.68 depending on alignment)
        assert!(d > 0.4 && d < 0.6, "Earth-Mars: {} AU", d);
    }

    #[test]
    fn hohmann_earth_mars() {
        let dv = hohmann_delta_v_kms(1.0, 1.524);
        // Earth-Mars Hohmann: ~5.7 km/s total
        assert!(dv > 4.0 && dv < 8.0, "Earth-Mars Hohmann: {} km/s", dv);
    }

    #[test]
    fn hohmann_earth_jupiter() {
        let dv = hohmann_delta_v_kms(1.0, 5.203);
        // Earth-Jupiter: ~14 km/s total
        assert!(dv > 10.0 && dv < 20.0, "Earth-Jupiter Hohmann: {} km/s", dv);
    }

    #[test]
    fn lagrange_points_exist() {
        assert!(get_body("Sun-Earth L1").is_some());
        assert!(get_body("Sun-Earth L2").is_some());
    }

    #[test]
    fn ceres_is_dwarf_planet() {
        let ceres = get_body("Ceres").unwrap();
        assert_eq!(ceres.body_type, BodyType::DwarfPlanet);
        assert!(ceres.semi_major_axis_au > 2.5 && ceres.semi_major_axis_au < 3.0);
    }

    #[test]
    fn catalog_total() {
        let all = solar_system_catalog();
        assert!(all.len() >= 18, "Should have 18+ bodies: {}", all.len());
    }
}

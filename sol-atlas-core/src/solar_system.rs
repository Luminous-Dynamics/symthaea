// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Solar system bodies for Sol Atlas visualization.
//! Orbital parameters, visual properties, and position computation.

use serde::{Deserialize, Serialize};

/// A celestial body in the solar system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CelestialBody {
    pub name: String,
    /// Orbital distance from Earth (in globe-radii, for visualization scale).
    pub orbit_radius: f32,
    /// Visual radius of the body (in globe-radii).
    pub visual_radius: f32,
    /// Orbital speed (radians per second).
    pub orbit_speed: f32,
    /// Initial orbital angle offset (radians).
    pub orbit_offset: f32,
    /// Y-axis offset (tilt above/below ecliptic plane).
    pub y_offset: f32,
    /// Is this the Sun (emissive, not lit)?
    pub is_sun: bool,
    /// Texture filename (in assets/textures/).
    pub texture: String,
    /// PBR roughness.
    pub roughness: f32,
    /// PBR metalness.
    pub metalness: f32,
    /// Emissive strength (0.0 = none, 2.0+ = glow).
    pub emission: f32,
}

/// All solar system bodies visible from Earth's perspective.
pub fn solar_system_bodies() -> Vec<CelestialBody> {
    vec![
        // Sizes use log-compressed real ratios so everything is visible:
        // Real: Sun=109×, Jupiter=11.2×, Saturn=9.4×, Venus=0.95×, Mars=0.53×, Moon=0.27×
        // Display: exaggerate small bodies, compress large ones
        CelestialBody {
            name: "Sun".into(),
            orbit_radius: 30.0,
            visual_radius: 3.0,
            orbit_speed: 0.002,
            orbit_offset: 0.0,
            y_offset: 0.0,
            is_sun: true,
            texture: "sun.jpg".into(),
            roughness: 1.0,
            metalness: 0.0,
            emission: 3.0,
        },
        CelestialBody {
            name: "Moon".into(),
            orbit_radius: 2.0,
            visual_radius: 0.08, // exaggerated from 0.27× so it's visible
            orbit_speed: 0.04,
            orbit_offset: 1.2,
            y_offset: 0.05,
            is_sun: false,
            texture: "moon.jpg".into(),
            roughness: 0.95,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Venus".into(),
            orbit_radius: 5.5,
            visual_radius: 0.18,
            orbit_speed: 0.007,
            orbit_offset: 2.4,
            y_offset: 0.0,
            is_sun: false,
            texture: "venus.jpg".into(),
            roughness: 0.8,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Mars".into(),
            orbit_radius: 7.0,
            visual_radius: 0.14,
            orbit_speed: 0.004,
            orbit_offset: 4.1,
            y_offset: 0.0,
            is_sun: false,
            texture: "mars.jpg".into(),
            roughness: 0.9,
            metalness: 0.1,
            emission: 0.0,
        },
        CelestialBody {
            name: "Jupiter".into(),
            orbit_radius: 12.0,
            visual_radius: 1.2, // gas giant — clearly larger than Earth
            orbit_speed: 0.0015,
            orbit_offset: 0.8,
            y_offset: 0.0,
            is_sun: false,
            texture: "jupiter.jpg".into(),
            roughness: 0.7,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Saturn".into(),
            orbit_radius: 16.0,
            visual_radius: 1.0, // slightly smaller than Jupiter
            orbit_speed: 0.001,
            orbit_offset: 3.5,
            y_offset: 0.0,
            is_sun: false,
            texture: "saturn.jpg".into(),
            roughness: 0.75,
            metalness: 0.0,
            emission: 0.0,
        },
    ]
}

/// Compute the 3D position of a celestial body at a given time.
pub fn body_position(body: &CelestialBody, time_secs: f32) -> [f32; 3] {
    let angle = body.orbit_offset + time_secs * body.orbit_speed;
    [
        body.orbit_radius * angle.cos(),
        body.y_offset,
        body.orbit_radius * angle.sin(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solar_system_has_6_bodies() {
        assert_eq!(solar_system_bodies().len(), 6);
    }

    #[test]
    fn test_sun_is_emissive() {
        let bodies = solar_system_bodies();
        let sun = bodies.iter().find(|b| b.name == "Sun").unwrap();
        assert!(sun.is_sun);
        assert!(sun.emission > 1.0);
    }

    #[test]
    fn test_body_position_orbits() {
        let moon = &solar_system_bodies()[1];
        let p0 = body_position(moon, 0.0);
        let p1 = body_position(moon, 100.0);
        // Should be at different positions
        assert!((p0[0] - p1[0]).abs() > 0.01 || (p0[2] - p1[2]).abs() > 0.01);
    }
}

#[cfg(test)]
mod planet_tests {
    use super::*;

    #[test]
    fn all_bodies_have_textures() {
        for body in solar_system_bodies() {
            assert!(!body.texture.is_empty(), "{} has no texture", body.name);
            assert!(body.visual_radius > 0.0, "{} has zero radius", body.name);
            assert!(body.orbit_radius > 0.0, "{} has zero orbit", body.name);
        }
    }

    #[test]
    fn sun_is_largest() {
        let bodies = solar_system_bodies();
        let sun = bodies.iter().find(|b| b.is_sun).unwrap();
        for body in &bodies {
            if !body.is_sun {
                assert!(sun.visual_radius > body.visual_radius,
                    "Sun ({}) should be larger than {} ({})", sun.visual_radius, body.name, body.visual_radius);
            }
        }
    }

    #[test]
    fn jupiter_larger_than_earth() {
        let bodies = solar_system_bodies();
        let jupiter = bodies.iter().find(|b| b.name == "Jupiter").unwrap();
        // Earth is radius 1.0, Jupiter should be > 1.0
        assert!(jupiter.visual_radius > 1.0,
            "Jupiter ({}) should be larger than Earth (1.0)", jupiter.visual_radius);
    }

    #[test]
    fn moon_smallest_planet() {
        let bodies = solar_system_bodies();
        let moon = bodies.iter().find(|b| b.name == "Moon").unwrap();
        for body in &bodies {
            if body.name != "Moon" && !body.is_sun {
                assert!(moon.visual_radius <= body.visual_radius,
                    "Moon ({}) should be <= {} ({})", moon.visual_radius, body.name, body.visual_radius);
            }
        }
    }

    #[test]
    fn body_positions_valid() {
        for body in solar_system_bodies() {
            let pos = body_position(&body, 0.0);
            let r = (pos[0]*pos[0] + pos[1]*pos[1] + pos[2]*pos[2]).sqrt();
            assert!((r - body.orbit_radius).abs() < 1.0,
                "{} position radius {} should be near orbit_radius {}", body.name, r, body.orbit_radius);
        }
    }

    #[test]
    fn body_positions_change_over_time() {
        for body in solar_system_bodies() {
            let p0 = body_position(&body, 0.0);
            let p1 = body_position(&body, 100.0);
            let moved = (p0[0] - p1[0]).abs() + (p0[2] - p1[2]).abs();
            assert!(moved > 0.01, "{} should move over 100 seconds", body.name);
        }
    }
}

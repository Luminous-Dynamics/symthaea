// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Solar system bodies for Terra Atlas visualization.
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
        CelestialBody {
            name: "Sun".into(),
            orbit_radius: 12.0,
            visual_radius: 1.5,
            orbit_speed: 0.005,
            orbit_offset: 0.0,
            y_offset: 0.5,
            is_sun: true,
            texture: "sun.jpg".into(),
            roughness: 1.0,
            metalness: 0.0,
            emission: 3.0,
        },
        CelestialBody {
            name: "Moon".into(),
            orbit_radius: 2.5,
            visual_radius: 0.18,
            orbit_speed: 0.03,
            orbit_offset: 1.2,
            y_offset: 0.15,
            is_sun: false,
            texture: "moon.jpg".into(),
            roughness: 0.95,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Venus".into(),
            orbit_radius: 8.0,
            visual_radius: 0.3,
            orbit_speed: 0.008,
            orbit_offset: 2.4,
            y_offset: -0.3,
            is_sun: false,
            texture: "venus.jpg".into(),
            roughness: 0.8,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Mars".into(),
            orbit_radius: 10.0,
            visual_radius: 0.25,
            orbit_speed: 0.004,
            orbit_offset: 4.1,
            y_offset: 0.2,
            is_sun: false,
            texture: "mars.jpg".into(),
            roughness: 0.9,
            metalness: 0.1,
            emission: 0.0,
        },
        CelestialBody {
            name: "Jupiter".into(),
            orbit_radius: 16.0,
            visual_radius: 0.6,
            orbit_speed: 0.002,
            orbit_offset: 0.8,
            y_offset: -0.1,
            is_sun: false,
            texture: "jupiter.jpg".into(),
            roughness: 0.7,
            metalness: 0.0,
            emission: 0.0,
        },
        CelestialBody {
            name: "Saturn".into(),
            orbit_radius: 20.0,
            visual_radius: 0.5,
            orbit_speed: 0.001,
            orbit_offset: 3.5,
            y_offset: 0.3,
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

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Geophysics
//!
//! **Status**: Exploratory module — HDC encodings for geophysics (seismology, Earth layers, plate tectonics).
//!
//! This module encodes geophysical phenomena: the physics of Earth
//! and planetary systems.
//!
//! ## Key Domains
//!
//! - **Seismology**: Elastic waves, earthquakes, structure
//! - **Geodesy**: Gravity, rotation, shape
//! - **Geomagnetism**: Dynamo, field variations
//! - **Geodynamics**: Mantle convection, plate tectonics
//! - **Planetary physics**: Interior structure, atmospheres
//!
//! ## Physical Principles
//!
//! - Continuum mechanics in heterogeneous media
//! - Thermal convection in viscous fluids
//! - Electromagnetic induction in conductors
//! - Wave propagation in layered media

use super::constants::G;
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Earth's average density
pub const EARTH_DENSITY: f64 = 5515.0; // kg/m³

/// Seismic wave type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeismicWaveType {
    P,        // Primary (compressional)
    S,        // Secondary (shear)
    Love,     // Surface, horizontal shear
    Rayleigh, // Surface, elliptical motion
}

/// Earth layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EarthLayer {
    Crust,
    UpperMantle,
    LowerMantle,
    OuterCore,
    InnerCore,
}

impl EarthLayer {
    /// Approximate depth range in km
    pub fn depth_range_km(&self) -> (f64, f64) {
        match self {
            EarthLayer::Crust => (0.0, 35.0),
            EarthLayer::UpperMantle => (35.0, 670.0),
            EarthLayer::LowerMantle => (670.0, 2890.0),
            EarthLayer::OuterCore => (2890.0, 5150.0),
            EarthLayer::InnerCore => (5150.0, 6371.0),
        }
    }
}

/// An earthquake
#[derive(Debug, Clone)]
pub struct Earthquake {
    pub magnitude: f64, // Moment magnitude
    pub depth_km: f64,
    pub focal_mechanism: FocalMechanism,
    pub moment_nm: f64, // Seismic moment in N·m
    pub vector: ContinuousHV,
}

/// Focal mechanism (fault type)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FocalMechanism {
    StrikeSlip,
    Normal,
    Thrust,
    Oblique,
}

/// Geophysics Encoder
#[derive(Debug, Clone)]
pub struct GeophysicsEncoder {
    genesis: GenesisSeed,

    // Seismology
    pub seismic_wave: ContinuousHV,
    pub p_wave: ContinuousHV,
    pub s_wave: ContinuousHV,
    pub surface_wave: ContinuousHV,
    pub earthquake: ContinuousHV,
    pub fault: ContinuousHV,
    pub rupture: ContinuousHV,
    pub attenuation: ContinuousHV,

    // Earth structure
    pub crust: ContinuousHV,
    pub mantle: ContinuousHV,
    pub core: ContinuousHV,
    pub moho: ContinuousHV, // Mohorovičić discontinuity
    pub cmb: ContinuousHV,  // Core-mantle boundary
    pub lithosphere: ContinuousHV,
    pub asthenosphere: ContinuousHV,

    // Plate tectonics
    pub plate: ContinuousHV,
    pub subduction: ContinuousHV,
    pub spreading: ContinuousHV,
    pub transform: ContinuousHV,
    pub hotspot: ContinuousHV,
    pub convection: ContinuousHV,

    // Geomagnetism
    pub magnetic_field: ContinuousHV,
    pub dynamo: ContinuousHV,
    pub dipole: ContinuousHV,
    pub secular_variation: ContinuousHV,
    pub reversal: ContinuousHV,

    // Geodesy
    pub gravity: ContinuousHV,
    pub geoid: ContinuousHV,
    pub isostasy: ContinuousHV,
    pub rotation: ContinuousHV,
    pub tides: ContinuousHV,

    // Material properties
    pub elastic: ContinuousHV,
    pub viscous: ContinuousHV,
    pub viscoelastic: ContinuousHV,
    pub anisotropy: ContinuousHV,
}

impl GeophysicsEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            genesis: genesis.clone(),

            seismic_wave: genesis.hv("seismo::wave", PHYSICS_DIM),
            p_wave: genesis.hv("seismo::p_wave", PHYSICS_DIM),
            s_wave: genesis.hv("seismo::s_wave", PHYSICS_DIM),
            surface_wave: genesis.hv("seismo::surface_wave", PHYSICS_DIM),
            earthquake: genesis.hv("seismo::earthquake", PHYSICS_DIM),
            fault: genesis.hv("seismo::fault", PHYSICS_DIM),
            rupture: genesis.hv("seismo::rupture", PHYSICS_DIM),
            attenuation: genesis.hv("seismo::attenuation", PHYSICS_DIM),

            crust: genesis.hv("earth::crust", PHYSICS_DIM),
            mantle: genesis.hv("earth::mantle", PHYSICS_DIM),
            core: genesis.hv("earth::core", PHYSICS_DIM),
            moho: genesis.hv("earth::moho", PHYSICS_DIM),
            cmb: genesis.hv("earth::cmb", PHYSICS_DIM),
            lithosphere: genesis.hv("earth::lithosphere", PHYSICS_DIM),
            asthenosphere: genesis.hv("earth::asthenosphere", PHYSICS_DIM),

            plate: genesis.hv("tectonics::plate", PHYSICS_DIM),
            subduction: genesis.hv("tectonics::subduction", PHYSICS_DIM),
            spreading: genesis.hv("tectonics::spreading", PHYSICS_DIM),
            transform: genesis.hv("tectonics::transform", PHYSICS_DIM),
            hotspot: genesis.hv("tectonics::hotspot", PHYSICS_DIM),
            convection: genesis.hv("tectonics::convection", PHYSICS_DIM),

            magnetic_field: genesis.hv("geomag::field", PHYSICS_DIM),
            dynamo: genesis.hv("geomag::dynamo", PHYSICS_DIM),
            dipole: genesis.hv("geomag::dipole", PHYSICS_DIM),
            secular_variation: genesis.hv("geomag::secular", PHYSICS_DIM),
            reversal: genesis.hv("geomag::reversal", PHYSICS_DIM),

            gravity: genesis.hv("geodesy::gravity", PHYSICS_DIM),
            geoid: genesis.hv("geodesy::geoid", PHYSICS_DIM),
            isostasy: genesis.hv("geodesy::isostasy", PHYSICS_DIM),
            rotation: genesis.hv("geodesy::rotation", PHYSICS_DIM),
            tides: genesis.hv("geodesy::tides", PHYSICS_DIM),

            elastic: genesis.hv("material::elastic", PHYSICS_DIM),
            viscous: genesis.hv("material::viscous", PHYSICS_DIM),
            viscoelastic: genesis.hv("material::viscoelastic", PHYSICS_DIM),
            anisotropy: genesis.hv("material::anisotropy", PHYSICS_DIM),
        }
    }

    // ============================================================
    // SEISMOLOGY
    // ============================================================

    /// Create seismic wave
    pub fn create_seismic_wave(&self, wave_type: SeismicWaveType) -> ContinuousHV {
        match wave_type {
            SeismicWaveType::P => self.seismic_wave.bind(&self.p_wave).bind(&self.elastic),
            SeismicWaveType::S => self.seismic_wave.bind(&self.s_wave).bind(&self.elastic),
            SeismicWaveType::Love => self.surface_wave.bind(&self.s_wave),
            SeismicWaveType::Rayleigh => self.surface_wave.bind(&self.p_wave).bind(&self.s_wave),
        }
    }

    /// P-wave velocity: Vp = √((K + 4G/3)/ρ)
    pub fn p_wave_velocity(&self, bulk_mod: f64, shear_mod: f64, density: f64) -> f64 {
        ((bulk_mod + 4.0 * shear_mod / 3.0) / density).sqrt()
    }

    /// S-wave velocity: Vs = √(G/ρ)
    pub fn s_wave_velocity(&self, shear_mod: f64, density: f64) -> f64 {
        (shear_mod / density).sqrt()
    }

    /// Create an earthquake
    pub fn create_earthquake(
        &self,
        magnitude: f64,
        depth_km: f64,
        mechanism: FocalMechanism,
    ) -> Earthquake {
        // Seismic moment from magnitude: log₁₀(M₀) = 1.5M + 9.1
        let moment = 10.0_f64.powf(1.5 * magnitude + 9.1);

        let mech_vec = match mechanism {
            FocalMechanism::StrikeSlip => self.transform.clone(),
            FocalMechanism::Normal => self.spreading.clone(),
            FocalMechanism::Thrust => self.subduction.clone(),
            FocalMechanism::Oblique => self.fault.clone(),
        };

        let vector = self
            .earthquake
            .bind(&self.rupture)
            .bind(&mech_vec)
            .scale(magnitude as f32);

        Earthquake {
            magnitude,
            depth_km,
            focal_mechanism: mechanism,
            moment_nm: moment,
            vector,
        }
    }

    /// Magnitude from seismic moment
    pub fn moment_to_magnitude(&self, moment_nm: f64) -> f64 {
        (2.0 / 3.0) * (moment_nm.log10() - 9.1)
    }

    // ============================================================
    // EARTH STRUCTURE
    // ============================================================

    /// Encode Earth layer
    pub fn earth_layer(&self, layer: EarthLayer) -> ContinuousHV {
        match layer {
            EarthLayer::Crust => self.crust.clone(),
            EarthLayer::UpperMantle => self.mantle.bind(&self.asthenosphere),
            EarthLayer::LowerMantle => self.mantle.bind(&self.viscous),
            EarthLayer::OuterCore => self.core.bind(&self.viscous).bind(&self.dynamo),
            EarthLayer::InnerCore => self.core.bind(&self.elastic),
        }
    }

    /// Discontinuity (Moho, CMB, ICB)
    pub fn discontinuity(&self, depth_km: f64) -> ContinuousHV {
        if depth_km < 50.0 {
            self.moho.clone()
        } else if depth_km < 3000.0 {
            self.cmb.clone()
        } else {
            self.genesis.hv("earth::icb", PHYSICS_DIM)
        }
    }

    /// PREM-like velocity at depth
    pub fn velocity_at_depth(&self, depth_km: f64) -> (f64, f64) {
        // Simplified PREM velocities
        let (vp, vs) = if depth_km < 35.0 {
            (6.5, 3.7) // Crust
        } else if depth_km < 670.0 {
            (8.0 + depth_km * 0.002, 4.5 + depth_km * 0.001) // Upper mantle
        } else if depth_km < 2890.0 {
            (13.0, 7.0) // Lower mantle
        } else if depth_km < 5150.0 {
            (10.0, 0.0) // Outer core (no S-waves)
        } else {
            (11.2, 3.6) // Inner core
        };
        (vp, vs)
    }

    // ============================================================
    // PLATE TECTONICS
    // ============================================================

    /// Plate boundary type
    pub fn plate_boundary(&self, boundary_type: &str) -> ContinuousHV {
        match boundary_type {
            "divergent" => self.spreading.bind(&self.plate),
            "convergent" => self.subduction.bind(&self.plate),
            "transform" => self.transform.bind(&self.plate),
            _ => self.plate.clone(),
        }
    }

    /// Mantle convection encoding
    pub fn mantle_convection(&self) -> ContinuousHV {
        self.convection.bind(&self.mantle).bind(&self.viscous)
    }

    /// Plate velocity (typical: 2-10 cm/yr)
    pub fn plate_velocity_encoding(&self, velocity_cm_yr: f64) -> ContinuousHV {
        self.plate
            .bind(&self.spreading)
            .scale(velocity_cm_yr as f32)
    }

    // ============================================================
    // GEOMAGNETISM
    // ============================================================

    /// Geodynamo encoding
    pub fn geodynamo(&self) -> ContinuousHV {
        self.dynamo
            .bind(&self.core)
            .bind(&self.magnetic_field)
            .bind(&self.convection)
    }

    /// Magnetic field reversal
    pub fn magnetic_reversal(&self) -> ContinuousHV {
        self.reversal.bind(&self.dipole).bind(&self.dynamo)
    }

    /// Dipole moment encoding
    pub fn dipole_moment(&self, moment_am2: f64) -> ContinuousHV {
        // Earth's dipole: ~8×10²² A·m²
        let normalized = (moment_am2 / 8e22) as f32;
        self.dipole.bind(&self.magnetic_field).scale(normalized)
    }

    // ============================================================
    // GEODESY
    // ============================================================

    /// Gravity anomaly encoding
    pub fn gravity_anomaly(&self, mgal: f64) -> ContinuousHV {
        self.gravity.bind(&self.geoid).scale((mgal / 100.0) as f32) // Normalize to typical range
    }

    /// Isostatic compensation
    pub fn isostatic_compensation(&self) -> ContinuousHV {
        self.isostasy
            .bind(&self.gravity)
            .bind(&self.crust)
            .bind(&self.mantle)
    }

    /// Tidal potential
    pub fn tidal_encoding(&self) -> ContinuousHV {
        self.tides.bind(&self.gravity).bind(&self.rotation)
    }

    /// Surface gravity from mass and radius
    pub fn surface_gravity(&self, mass_kg: f64, radius_m: f64) -> f64 {
        G * mass_kg / (radius_m * radius_m)
    }

    // ============================================================
    // MATERIAL PROPERTIES
    // ============================================================

    /// Maxwell viscoelastic model
    pub fn maxwell_viscoelastic(&self, viscosity: f64, shear_mod: f64) -> ContinuousHV {
        let maxwell_time = viscosity / shear_mod;
        self.viscoelastic
            .bind(&self.viscous)
            .bind(&self.elastic)
            .scale(maxwell_time.log10() as f32)
    }

    /// Seismic anisotropy (direction-dependent velocity)
    pub fn seismic_anisotropy(&self, anisotropy_percent: f64) -> ContinuousHV {
        self.anisotropy
            .bind(&self.seismic_wave)
            .scale(anisotropy_percent as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> GeophysicsEncoder {
        let genesis = GenesisSeed::from_phrase("geophysics test");
        GeophysicsEncoder::from_genesis(&genesis)
    }

    #[test]
    fn test_encoder_creation() {
        let encoder = setup();
        assert!(encoder.seismic_wave.norm() > 0.0);
        assert!(encoder.mantle.norm() > 0.0);
    }

    #[test]
    fn test_seismic_velocities() {
        let encoder = setup();

        // Typical upper mantle values
        let vp = encoder.p_wave_velocity(130e9, 70e9, 3300.0);
        let vs = encoder.s_wave_velocity(70e9, 3300.0);

        // Vp should be ~8 km/s
        assert!((vp / 1000.0 - 8.0).abs() < 1.0);
        // Vs should be ~4.5 km/s
        assert!((vs / 1000.0 - 4.5).abs() < 1.0);
        // Vp > Vs always
        assert!(vp > vs);
    }

    #[test]
    fn test_earthquake() {
        let encoder = setup();

        let eq = encoder.create_earthquake(7.0, 20.0, FocalMechanism::Thrust);

        assert_eq!(eq.magnitude, 7.0);
        assert!(eq.moment_nm > 1e18); // M7 has moment ~3×10¹⁹ N·m
    }

    #[test]
    fn test_moment_magnitude() {
        let encoder = setup();

        // M7 earthquake
        let moment = 10.0_f64.powf(1.5 * 7.0 + 9.1);
        let mag = encoder.moment_to_magnitude(moment);

        assert!((mag - 7.0).abs() < 0.1);
    }

    #[test]
    fn test_no_s_waves_in_outer_core() {
        let encoder = setup();

        let (vp, vs) = encoder.velocity_at_depth(3500.0); // Outer core

        // S-waves don't propagate in liquid outer core
        assert!(vs == 0.0);
        assert!(vp > 0.0);
    }

    #[test]
    fn test_surface_gravity() {
        let encoder = setup();

        // Earth: M = 5.97×10²⁴ kg, R = 6.37×10⁶ m
        let g = encoder.surface_gravity(5.97e24, 6.37e6);

        // Should be ~9.8 m/s²
        assert!((g - 9.8).abs() < 0.2);
    }

    #[test]
    fn test_plate_boundaries() {
        let encoder = setup();

        let divergent = encoder.plate_boundary("divergent");
        let convergent = encoder.plate_boundary("convergent");

        // Different boundaries should produce different vectors
        assert!(divergent.norm() > 0.0);
        assert!(convergent.norm() > 0.0);
    }

    #[test]
    fn test_geodynamo() {
        let encoder = setup();

        let dynamo = encoder.geodynamo();

        // Should involve core, convection, and magnetic field
        assert!(dynamo.norm() > 0.0);
    }
}

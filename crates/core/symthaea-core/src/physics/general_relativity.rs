// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # General Relativity
//!
//! **Status**: Exploratory module — HDC encodings for general relativity (spacetime geometry, gravitational waves).
//!
//! This module encodes general relativistic concepts from spacetime geometry
//! to gravitational waves.
//!
//! ## Core Concepts
//!
//! - **Spacetime**: 4D manifold with Lorentzian metric
//! - **Metric Tensor**: g_μν encodes geometry
//! - **Curvature**: Riemann tensor, Ricci tensor, scalar curvature
//! - **Geodesics**: Paths of freely falling objects
//! - **Einstein Equations**: G_μν = 8πG T_μν
//!
//! ## Key Solutions
//!
//! - **Schwarzschild**: Non-rotating black hole
//! - **Kerr**: Rotating black hole
//! - **FLRW**: Cosmological spacetime
//! - **Gravitational Waves**: Ripples in spacetime

use super::constants::{C, G};
use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Spacetime signature
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signature {
    /// (-,+,+,+) mostly plus convention
    MostlyPlus,
    /// (+,-,-,-) mostly minus convention
    MostlyMinus,
}

/// Coordinate system type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinateSystem {
    Cartesian,
    Spherical,
    BoyerLindquist, // For Kerr
    Kruskal,        // Maximal extension
    EddingtonFinkelstein,
}

/// Spacetime type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpacetimeType {
    Flat,          // Minkowski
    Schwarzschild, // Non-rotating BH
    Kerr,          // Rotating BH
    KerrNewman,    // Rotating charged BH
    FLRW,          // Cosmological
    DeSitter,      // Positive cosmological constant
    AntiDeSitter,  // Negative cosmological constant
    GravitationalWave,
}

impl SpacetimeType {
    fn domain(&self) -> &'static str {
        match self {
            SpacetimeType::Flat => "gr::minkowski",
            SpacetimeType::Schwarzschild => "gr::schwarzschild",
            SpacetimeType::Kerr => "gr::kerr",
            SpacetimeType::KerrNewman => "gr::kerr_newman",
            SpacetimeType::FLRW => "gr::flrw",
            SpacetimeType::DeSitter => "gr::de_sitter",
            SpacetimeType::AntiDeSitter => "gr::anti_de_sitter",
            SpacetimeType::GravitationalWave => "gr::gw",
        }
    }
}

/// Gravitational wave polarization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GWPolarization {
    Plus,
    Cross,
}

/// A spacetime geometry
#[derive(Debug, Clone)]
pub struct Spacetime {
    pub name: String,
    pub spacetime_type: SpacetimeType,
    pub dimension: u8,
    /// Metric signature
    pub signature: Signature,
    /// Characteristic length scale (e.g., Schwarzschild radius)
    pub length_scale_m: f64,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// A geodesic (path through spacetime)
#[derive(Debug, Clone)]
pub struct Geodesic {
    /// Type: timelike (massive), null (light), spacelike
    pub causal_type: CausalType,
    /// Proper time/affine parameter
    pub proper_time: f64,
    /// Four-velocity vector
    pub four_velocity: ContinuousHV,
    /// Overall vector
    pub vector: ContinuousHV,
}

/// Causal character of a vector/geodesic
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalType {
    Timelike,  // Massive particles
    Null,      // Light
    Spacelike, // Tachyonic (forbidden for particles)
}

impl CausalType {
    fn domain(&self) -> &'static str {
        match self {
            CausalType::Timelike => "gr::timelike",
            CausalType::Null => "gr::null",
            CausalType::Spacelike => "gr::spacelike",
        }
    }
}

/// Gravitational wave
#[derive(Debug, Clone)]
pub struct GravitationalWave {
    /// Frequency in Hz
    pub frequency_hz: f64,
    /// Strain amplitude (dimensionless)
    pub strain: f64,
    /// Polarization
    pub polarization: GWPolarization,
    /// Source type (binary, supernova, etc.)
    pub source: GWSource,
    /// Vector representation
    pub vector: ContinuousHV,
}

/// Gravitational wave source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GWSource {
    BinaryBlackHole,
    BinaryNeutronStar,
    BlackHoleNeutronStar,
    Supernova,
    Pulsar,
    CosmicStrings,
    Inflation,
}

impl GWSource {
    fn domain(&self) -> &'static str {
        match self {
            GWSource::BinaryBlackHole => "gw::bbh",
            GWSource::BinaryNeutronStar => "gw::bns",
            GWSource::BlackHoleNeutronStar => "gw::bhns",
            GWSource::Supernova => "gw::supernova",
            GWSource::Pulsar => "gw::pulsar",
            GWSource::CosmicStrings => "gw::cosmic_strings",
            GWSource::Inflation => "gw::inflation",
        }
    }
}

/// General Relativity encoder
#[derive(Debug, Clone)]
pub struct GREncoder {
    // Fundamental concepts
    pub spacetime: ContinuousHV,
    pub metric: ContinuousHV,
    pub curvature: ContinuousHV,
    pub geodesic: ContinuousHV,

    // Tensor concepts
    pub riemann: ContinuousHV,       // Riemann curvature tensor
    pub ricci: ContinuousHV,         // Ricci tensor
    pub einstein: ContinuousHV,      // Einstein tensor
    pub weyl: ContinuousHV,          // Weyl conformal tensor
    pub stress_energy: ContinuousHV, // Stress-energy tensor

    // Causal structure
    pub timelike: ContinuousHV,
    pub null: ContinuousHV,
    pub spacelike: ContinuousHV,
    pub light_cone: ContinuousHV,
    pub horizon: ContinuousHV,

    // Spacetime types
    pub flat: ContinuousHV,
    pub curved: ContinuousHV,
    pub singular: ContinuousHV,

    // Gravitational waves
    pub gw_plus: ContinuousHV,
    pub gw_cross: ContinuousHV,
    pub gw_strain: ContinuousHV,

    // Physical effects
    pub frame_dragging: ContinuousHV,
    pub gravitational_redshift: ContinuousHV,
    pub time_dilation: ContinuousHV,
    pub length_contraction: ContinuousHV,
}

impl GREncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            spacetime: genesis.hv("gr::spacetime", PHYSICS_DIM),
            metric: genesis.hv("gr::metric", PHYSICS_DIM),
            curvature: genesis.hv("gr::curvature", PHYSICS_DIM),
            geodesic: genesis.hv("gr::geodesic", PHYSICS_DIM),

            riemann: genesis.hv("gr::riemann_tensor", PHYSICS_DIM),
            ricci: genesis.hv("gr::ricci_tensor", PHYSICS_DIM),
            einstein: genesis.hv("gr::einstein_tensor", PHYSICS_DIM),
            weyl: genesis.hv("gr::weyl_tensor", PHYSICS_DIM),
            stress_energy: genesis.hv("gr::stress_energy", PHYSICS_DIM),

            timelike: genesis.hv(CausalType::Timelike.domain(), PHYSICS_DIM),
            null: genesis.hv(CausalType::Null.domain(), PHYSICS_DIM),
            spacelike: genesis.hv(CausalType::Spacelike.domain(), PHYSICS_DIM),
            light_cone: genesis.hv("gr::light_cone", PHYSICS_DIM),
            horizon: genesis.hv("gr::horizon", PHYSICS_DIM),

            flat: genesis.hv("gr::flat", PHYSICS_DIM),
            curved: genesis.hv("gr::curved", PHYSICS_DIM),
            singular: genesis.hv("gr::singular", PHYSICS_DIM),

            gw_plus: genesis.hv("gr::gw_plus", PHYSICS_DIM),
            gw_cross: genesis.hv("gr::gw_cross", PHYSICS_DIM),
            gw_strain: genesis.hv("gr::gw_strain", PHYSICS_DIM),

            frame_dragging: genesis.hv("gr::frame_dragging", PHYSICS_DIM),
            gravitational_redshift: genesis.hv("gr::gravitational_redshift", PHYSICS_DIM),
            time_dilation: genesis.hv("gr::time_dilation", PHYSICS_DIM),
            length_contraction: genesis.hv("gr::length_contraction", PHYSICS_DIM),
        }
    }

    /// Minkowski (flat) spacetime
    pub fn minkowski(&self) -> Spacetime {
        Spacetime {
            name: "Minkowski".to_string(),
            spacetime_type: SpacetimeType::Flat,
            dimension: 4,
            signature: Signature::MostlyPlus,
            length_scale_m: f64::INFINITY,
            vector: self.spacetime.bind(&self.flat),
        }
    }

    /// Schwarzschild spacetime (non-rotating black hole)
    pub fn schwarzschild(&self, mass_kg: f64) -> Spacetime {
        // Schwarzschild radius: r_s = 2GM/c²
        let rs = 2.0 * G * mass_kg / (C * C);

        let vector = self
            .spacetime
            .bind(&self.curved)
            .bind(&self.horizon)
            .scale((mass_kg / 1e30).ln() as f32 + 1.0); // Log scale for mass

        Spacetime {
            name: "Schwarzschild".to_string(),
            spacetime_type: SpacetimeType::Schwarzschild,
            dimension: 4,
            signature: Signature::MostlyPlus,
            length_scale_m: rs,
            vector,
        }
    }

    /// Kerr spacetime (rotating black hole)
    pub fn kerr(&self, mass_kg: f64, spin: f64) -> Spacetime {
        let rs = 2.0 * G * mass_kg / (C * C);
        let a = spin.abs().min(1.0); // Dimensionless spin parameter

        let vector = self
            .spacetime
            .bind(&self.curved)
            .bind(&self.horizon)
            .bind(&self.frame_dragging.scale(a as f32))
            .scale((mass_kg / 1e30).ln() as f32 + 1.0);

        Spacetime {
            name: "Kerr".to_string(),
            spacetime_type: SpacetimeType::Kerr,
            dimension: 4,
            signature: Signature::MostlyPlus,
            length_scale_m: rs,
            vector,
        }
    }

    /// De Sitter spacetime (positive cosmological constant)
    pub fn de_sitter(&self, hubble_parameter: f64) -> Spacetime {
        let horizon_radius = C / hubble_parameter;

        let vector = self
            .spacetime
            .bind(&self.curved)
            .bind(&self.horizon)
            .scale(hubble_parameter.ln() as f32);

        Spacetime {
            name: "de Sitter".to_string(),
            spacetime_type: SpacetimeType::DeSitter,
            dimension: 4,
            signature: Signature::MostlyPlus,
            length_scale_m: horizon_radius,
            vector,
        }
    }

    /// Create a geodesic
    pub fn create_geodesic(&self, causal_type: CausalType, genesis: &GenesisSeed) -> Geodesic {
        let causal_vec = match causal_type {
            CausalType::Timelike => &self.timelike,
            CausalType::Null => &self.null,
            CausalType::Spacelike => &self.spacelike,
        };

        let four_velocity = genesis.hv("gr::four_velocity", PHYSICS_DIM);
        let vector = self.geodesic.bind(causal_vec).bind(&four_velocity);

        Geodesic {
            causal_type,
            proper_time: 0.0,
            four_velocity,
            vector,
        }
    }

    /// Light ray (null geodesic)
    pub fn light_ray(&self, genesis: &GenesisSeed) -> Geodesic {
        self.create_geodesic(CausalType::Null, genesis)
    }

    /// Massive particle worldline (timelike geodesic)
    pub fn massive_worldline(&self, genesis: &GenesisSeed) -> Geodesic {
        self.create_geodesic(CausalType::Timelike, genesis)
    }

    /// Gravitational wave from binary black hole merger
    pub fn binary_bh_merger(
        &self,
        m1_solar: f64,
        m2_solar: f64,
        distance_mpc: f64,
    ) -> GravitationalWave {
        let m_total = m1_solar + m2_solar;
        let m_chirp = ((m1_solar * m2_solar).powf(3.0 / 5.0)) / m_total.powf(1.0 / 5.0);

        // Characteristic frequency at merger: f ~ c³/(GM)
        let freq = C.powi(3) / (G * m_total * 1.989e30 * std::f64::consts::PI * 10.0);

        // Strain amplitude: h ~ (GM_chirp/c²) / (distance)
        let strain = (G * m_chirp * 1.989e30 / (C * C)) / (distance_mpc * 3.086e22);

        let source_vec = self
            .gw_strain
            .scale(strain.log10() as f32 + 22.0) // Strain is ~10^-22
            .bind(&ContinuousHV::bundle(&[&self.gw_plus, &self.gw_cross]));

        GravitationalWave {
            frequency_hz: freq,
            strain,
            polarization: GWPolarization::Plus,
            source: GWSource::BinaryBlackHole,
            vector: source_vec,
        }
    }

    /// Gravitational wave from binary neutron star merger
    pub fn binary_ns_merger(
        &self,
        m1_solar: f64,
        m2_solar: f64,
        distance_mpc: f64,
    ) -> GravitationalWave {
        let _m_total = m1_solar + m2_solar;
        let freq = 1000.0; // ~kHz for NS mergers
        let strain = 1e-22 * (40.0 / distance_mpc);

        let source_vec = self
            .gw_strain
            .scale(strain.log10() as f32 + 22.0)
            .bind(&self.gw_plus);

        GravitationalWave {
            frequency_hz: freq,
            strain,
            polarization: GWPolarization::Plus,
            source: GWSource::BinaryNeutronStar,
            vector: source_vec,
        }
    }

    /// Einstein field equations: G_μν = 8πG/c⁴ T_μν
    /// Returns a vector representing this relationship
    pub fn einstein_equations(&self) -> ContinuousHV {
        self.einstein.bind(&self.stress_energy)
    }

    /// Geodesic equation (equation of motion in curved spacetime)
    pub fn geodesic_equation(&self) -> ContinuousHV {
        self.geodesic.bind(&self.curvature)
    }

    /// Gravitational redshift factor
    pub fn redshift_factor(&self, r: f64, rs: f64) -> f64 {
        if r <= rs {
            f64::INFINITY
        } else {
            1.0 / (1.0 - rs / r).sqrt()
        }
    }

    /// Time dilation near a mass
    pub fn gravitational_time_dilation(&self, r: f64, rs: f64) -> f64 {
        if r <= rs { 0.0 } else { (1.0 - rs / r).sqrt() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (GREncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("general relativity test");
        let encoder = GREncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_gr_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.spacetime.norm() > 0.0);
        assert!(encoder.curvature.norm() > 0.0);
    }

    #[test]
    fn test_minkowski_spacetime() {
        let (encoder, _) = setup();
        let mink = encoder.minkowski();

        assert_eq!(mink.spacetime_type, SpacetimeType::Flat);
        assert_eq!(mink.dimension, 4);
        assert!(mink.vector.norm() > 0.0);
    }

    #[test]
    fn test_schwarzschild_spacetime() {
        let (encoder, _) = setup();
        // Solar mass black hole
        let bh = encoder.schwarzschild(1.989e30);

        assert_eq!(bh.spacetime_type, SpacetimeType::Schwarzschild);
        // Schwarzschild radius ~ 3 km for 1 solar mass
        assert!((bh.length_scale_m - 2953.0).abs() < 100.0);
    }

    #[test]
    fn test_kerr_has_frame_dragging() {
        let (encoder, _) = setup();
        let schw = encoder.schwarzschild(1.989e30);
        let kerr = encoder.kerr(1.989e30, 0.9);

        // Kerr should have frame dragging character
        let kerr_fd = kerr.vector.similarity(&encoder.frame_dragging);
        let schw_fd = schw.vector.similarity(&encoder.frame_dragging);

        assert!(
            kerr_fd.abs() > schw_fd.abs(),
            "Kerr should have more frame dragging: kerr={}, schw={}",
            kerr_fd,
            schw_fd
        );
    }

    #[test]
    fn test_geodesics() {
        let (encoder, genesis) = setup();

        let light = encoder.light_ray(&genesis);
        let massive = encoder.massive_worldline(&genesis);

        assert_eq!(light.causal_type, CausalType::Null);
        assert_eq!(massive.causal_type, CausalType::Timelike);

        // They should be different
        let sim = light.vector.similarity(&massive.vector);
        assert!(sim < 0.9, "Light and massive geodesics should differ");
    }

    #[test]
    fn test_gravitational_wave() {
        let (encoder, _) = setup();

        // GW150914-like event: 36+29 solar masses at 410 Mpc
        let gw = encoder.binary_bh_merger(36.0, 29.0, 410.0);

        assert_eq!(gw.source, GWSource::BinaryBlackHole);
        assert!(gw.strain > 0.0);
        assert!(gw.frequency_hz > 0.0);
        assert!(gw.vector.norm() > 0.0);
    }

    #[test]
    fn test_einstein_equations() {
        let (encoder, _) = setup();
        let eqs = encoder.einstein_equations();

        // Should combine Einstein and stress-energy tensors
        assert!(eqs.norm() > 0.0);
    }

    #[test]
    fn test_gravitational_redshift() {
        let (encoder, _) = setup();

        // At r = 2*rs, redshift factor = sqrt(2)
        let z = encoder.redshift_factor(6000.0, 3000.0);
        assert!((z - std::f64::consts::SQRT_2).abs() < 0.01);

        // At horizon, redshift is infinite
        let z_horizon = encoder.redshift_factor(3000.0, 3000.0);
        assert!(z_horizon.is_infinite());
    }

    #[test]
    fn test_time_dilation() {
        let (encoder, _) = setup();

        // Far from mass: no dilation
        let td_far = encoder.gravitational_time_dilation(1e10, 3000.0);
        assert!((td_far - 1.0).abs() < 0.001);

        // At r = 2*rs: factor = 1/sqrt(2)
        let td_near = encoder.gravitational_time_dilation(6000.0, 3000.0);
        assert!((td_near - 1.0 / std::f64::consts::SQRT_2).abs() < 0.01);
    }

    #[test]
    fn test_curved_vs_flat() {
        let (encoder, _) = setup();

        let flat = encoder.minkowski();
        let curved = encoder.schwarzschild(1.989e30);

        // Flat should have more similarity to flat concept
        let flat_sim = flat.vector.similarity(&encoder.flat);
        let curved_flat_sim = curved.vector.similarity(&encoder.flat);

        assert!(
            flat_sim > curved_flat_sim,
            "Minkowski should be more flat: flat={}, curved={}",
            flat_sim,
            curved_flat_sim
        );
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Fluid Dynamics
//!
//! This module encodes fluid dynamics concepts from Navier-Stokes to turbulence.
//!
//! ## Governing Equations
//!
//! - **Continuity**: ∂ρ/∂t + ∇·(ρv) = 0
//! - **Navier-Stokes**: ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + f
//! - **Energy**: ρc_p(∂T/∂t + v·∇T) = k∇²T
//!
//! ## Dimensionless Numbers
//!
//! - **Reynolds**: Re = ρvL/μ (inertia/viscosity)
//! - **Mach**: Ma = v/c (compressibility)
//! - **Froude**: Fr = v/√(gL) (inertia/gravity)
//!
//! ## Flow Regimes
//!
//! - Laminar: Smooth, predictable (Re < 2300 in pipes)
//! - Turbulent: Chaotic, mixing (Re > 4000 in pipes)
//! - Transitional: Between laminar and turbulent

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// Flow regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowRegime {
    Laminar,
    Transitional,
    Turbulent,
}

impl FlowRegime {
    fn domain(&self) -> &'static str {
        match self {
            FlowRegime::Laminar => "fluid::laminar",
            FlowRegime::Transitional => "fluid::transitional",
            FlowRegime::Turbulent => "fluid::turbulent",
        }
    }
}

/// Flow type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlowType {
    Steady,
    Unsteady,
    Incompressible,
    Compressible,
    Viscous,
    Inviscid,
    Irrotational,
    Rotational,
}

impl FlowType {
    fn domain(&self) -> &'static str {
        match self {
            FlowType::Steady => "fluid::steady",
            FlowType::Unsteady => "fluid::unsteady",
            FlowType::Incompressible => "fluid::incompressible",
            FlowType::Compressible => "fluid::compressible",
            FlowType::Viscous => "fluid::viscous",
            FlowType::Inviscid => "fluid::inviscid",
            FlowType::Irrotational => "fluid::irrotational",
            FlowType::Rotational => "fluid::rotational",
        }
    }
}

/// Boundary layer type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryLayerType {
    Laminar,
    Turbulent,
    Separation,
}

/// A fluid flow
#[derive(Debug, Clone)]
pub struct FluidFlow {
    pub name: String,
    pub regime: FlowRegime,
    pub reynolds_number: f64,
    pub mach_number: f64,
    pub vector: ContinuousHV,
}

/// A vortex structure
#[derive(Debug, Clone)]
pub struct Vortex {
    pub name: String,
    /// Circulation (m²/s)
    pub circulation: f64,
    /// Core radius (m)
    pub core_radius: f64,
    /// Maximum tangential velocity (m/s)
    pub max_velocity: f64,
    pub vector: ContinuousHV,
}

/// Fluid dynamics encoder
#[derive(Debug, Clone)]
pub struct FluidEncoder {
    // Conservation laws
    pub continuity: ContinuousHV,
    pub momentum: ContinuousHV,
    pub energy: ContinuousHV,

    // Flow properties
    pub velocity: ContinuousHV,
    pub pressure: ContinuousHV,
    pub density: ContinuousHV,
    pub viscosity: ContinuousHV,

    // Flow regimes
    pub laminar: ContinuousHV,
    pub turbulent: ContinuousHV,
    pub transitional: ContinuousHV,

    // Flow types
    pub steady: ContinuousHV,
    pub unsteady: ContinuousHV,
    pub compressible: ContinuousHV,
    pub incompressible: ContinuousHV,

    // Vorticity
    pub vorticity: ContinuousHV,
    pub circulation: ContinuousHV,
    pub vortex: ContinuousHV,

    // Turbulence
    pub reynolds_stress: ContinuousHV,
    pub kolmogorov: ContinuousHV,
    pub cascade: ContinuousHV,
    pub dissipation: ContinuousHV,

    // Boundary layers
    pub boundary_layer: ContinuousHV,
    pub no_slip: ContinuousHV,
    pub separation: ContinuousHV,

    // Waves
    pub gravity_wave: ContinuousHV,
    pub shock_wave: ContinuousHV,
    pub acoustic_wave: ContinuousHV,
}

impl FluidEncoder {
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            continuity: genesis.hv("fluid::continuity", PHYSICS_DIM),
            momentum: genesis.hv("fluid::momentum", PHYSICS_DIM),
            energy: genesis.hv("fluid::energy", PHYSICS_DIM),

            velocity: genesis.hv("fluid::velocity", PHYSICS_DIM),
            pressure: genesis.hv("fluid::pressure", PHYSICS_DIM),
            density: genesis.hv("fluid::density", PHYSICS_DIM),
            viscosity: genesis.hv("fluid::viscosity", PHYSICS_DIM),

            laminar: genesis.hv(FlowRegime::Laminar.domain(), PHYSICS_DIM),
            turbulent: genesis.hv(FlowRegime::Turbulent.domain(), PHYSICS_DIM),
            transitional: genesis.hv(FlowRegime::Transitional.domain(), PHYSICS_DIM),

            steady: genesis.hv(FlowType::Steady.domain(), PHYSICS_DIM),
            unsteady: genesis.hv(FlowType::Unsteady.domain(), PHYSICS_DIM),
            compressible: genesis.hv(FlowType::Compressible.domain(), PHYSICS_DIM),
            incompressible: genesis.hv(FlowType::Incompressible.domain(), PHYSICS_DIM),

            vorticity: genesis.hv("fluid::vorticity", PHYSICS_DIM),
            circulation: genesis.hv("fluid::circulation", PHYSICS_DIM),
            vortex: genesis.hv("fluid::vortex", PHYSICS_DIM),

            reynolds_stress: genesis.hv("turbulence::reynolds_stress", PHYSICS_DIM),
            kolmogorov: genesis.hv("turbulence::kolmogorov", PHYSICS_DIM),
            cascade: genesis.hv("turbulence::cascade", PHYSICS_DIM),
            dissipation: genesis.hv("turbulence::dissipation", PHYSICS_DIM),

            boundary_layer: genesis.hv("fluid::boundary_layer", PHYSICS_DIM),
            no_slip: genesis.hv("fluid::no_slip", PHYSICS_DIM),
            separation: genesis.hv("fluid::separation", PHYSICS_DIM),

            gravity_wave: genesis.hv("fluid::gravity_wave", PHYSICS_DIM),
            shock_wave: genesis.hv("fluid::shock_wave", PHYSICS_DIM),
            acoustic_wave: genesis.hv("fluid::acoustic_wave", PHYSICS_DIM),
        }
    }

    /// Calculate Reynolds number: Re = ρvL/μ
    pub fn reynolds_number(&self, density: f64, velocity: f64, length: f64, viscosity: f64) -> f64 {
        density * velocity * length / viscosity
    }

    /// Calculate Mach number: Ma = v/c
    pub fn mach_number(&self, velocity: f64, sound_speed: f64) -> f64 {
        velocity / sound_speed
    }

    /// Classify flow regime from Reynolds number
    pub fn classify_regime(&self, re: f64) -> FlowRegime {
        if re < 2300.0 {
            FlowRegime::Laminar
        } else if re < 4000.0 {
            FlowRegime::Transitional
        } else {
            FlowRegime::Turbulent
        }
    }

    /// Create a fluid flow
    pub fn create_flow(&self, name: &str, re: f64, ma: f64, genesis: &GenesisSeed) -> FluidFlow {
        let regime = self.classify_regime(re);
        let regime_vec = genesis.hv(regime.domain(), PHYSICS_DIM);

        let compressibility = if ma > 0.3 {
            &self.compressible
        } else {
            &self.incompressible
        };

        let vector = self
            .velocity
            .bind(&regime_vec)
            .bind(compressibility)
            .scale(re.ln() as f32);

        FluidFlow {
            name: name.to_string(),
            regime,
            reynolds_number: re,
            mach_number: ma,
            vector,
        }
    }

    /// Navier-Stokes momentum equation
    pub fn navier_stokes(&self) -> ContinuousHV {
        // ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v
        ContinuousHV::bundle(&[
            &self.momentum,
            &self.velocity,
            &self.pressure,
            &self.viscosity,
        ])
    }

    /// Continuity equation: ∂ρ/∂t + ∇·(ρv) = 0
    pub fn continuity_equation(&self) -> ContinuousHV {
        self.continuity.bind(&self.density).bind(&self.velocity)
    }

    /// Euler equations (inviscid limit of Navier-Stokes)
    pub fn euler_equations(&self) -> ContinuousHV {
        self.navier_stokes().bind(&ContinuousHV::zero(PHYSICS_DIM)) // Zero viscosity
    }

    /// Create a vortex
    pub fn create_vortex(
        &self,
        circulation: f64,
        core_radius: f64,
        _genesis: &GenesisSeed,
    ) -> Vortex {
        // Rankine vortex: v_θ = Γ/(2πr) for r > core
        let max_velocity = circulation / (2.0 * std::f64::consts::PI * core_radius);

        let vector = self
            .vortex
            .bind(&self.vorticity)
            .bind(&self.circulation.scale(circulation as f32));

        Vortex {
            name: "Vortex".to_string(),
            circulation,
            core_radius,
            max_velocity,
            vector,
        }
    }

    /// Kolmogorov microscale: η = (ν³/ε)^(1/4)
    pub fn kolmogorov_scale(&self, viscosity: f64, dissipation_rate: f64) -> f64 {
        (viscosity.powi(3) / dissipation_rate).powf(0.25)
    }

    /// Turbulent energy cascade
    pub fn turbulent_cascade(&self) -> ContinuousHV {
        self.turbulent
            .bind(&self.cascade)
            .bind(&self.kolmogorov)
            .bind(&self.dissipation)
    }

    /// Boundary layer thickness: δ ~ L/√Re
    pub fn boundary_layer_thickness(&self, length: f64, re: f64) -> f64 {
        length / re.sqrt()
    }

    /// Shock wave (Ma > 1)
    pub fn shock(&self, upstream_mach: f64, _genesis: &GenesisSeed) -> ContinuousHV {
        if upstream_mach <= 1.0 {
            ContinuousHV::zero(PHYSICS_DIM)
        } else {
            self.shock_wave
                .bind(&self.compressible)
                .scale(upstream_mach as f32)
        }
    }

    /// Bernoulli's equation: p + ½ρv² + ρgh = const
    pub fn bernoulli(&self) -> ContinuousHV {
        ContinuousHV::bundle(&[&self.pressure, &self.velocity, &self.density])
            .bind(&self.steady)
            .bind(&self.incompressible)
    }

    /// Poiseuille flow (laminar pipe flow)
    pub fn poiseuille_flow(
        &self,
        radius: f64,
        pressure_gradient: f64,
        viscosity: f64,
        genesis: &GenesisSeed,
    ) -> FluidFlow {
        // Q = πR⁴ΔP / (8μL)
        // Mean velocity v = Q/(πR²) = R²ΔP / (8μL)
        let mean_velocity = radius * radius * pressure_gradient / (8.0 * viscosity);
        let re = self.reynolds_number(1000.0, mean_velocity, 2.0 * radius, viscosity);

        self.create_flow("Poiseuille", re, 0.0, genesis)
    }

    /// von Kármán vortex street (oscillating wake behind bluff body)
    pub fn vortex_street(&self, re: f64, genesis: &GenesisSeed) -> ContinuousHV {
        // Vortex shedding occurs for Re ~ 50-200
        self.vortex
            .bind(&self.unsteady)
            .bind(&genesis.hv("fluid::karman", PHYSICS_DIM))
            .scale((re / 100.0).ln() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (FluidEncoder, GenesisSeed) {
        let genesis = GenesisSeed::from_phrase("fluid dynamics test");
        let encoder = FluidEncoder::from_genesis(&genesis);
        (encoder, genesis)
    }

    #[test]
    fn test_fluid_encoder_creation() {
        let (encoder, _) = setup();
        assert!(encoder.velocity.norm() > 0.0);
        assert!(encoder.turbulent.norm() > 0.0);
    }

    #[test]
    fn test_reynolds_number() {
        let (encoder, _) = setup();

        // Water at 1 m/s through 1 cm pipe
        let re = encoder.reynolds_number(1000.0, 1.0, 0.01, 0.001);
        assert!((re - 10000.0).abs() < 1.0);
    }

    #[test]
    fn test_flow_regime_classification() {
        let (encoder, _) = setup();

        assert_eq!(encoder.classify_regime(1000.0), FlowRegime::Laminar);
        assert_eq!(encoder.classify_regime(3000.0), FlowRegime::Transitional);
        assert_eq!(encoder.classify_regime(10000.0), FlowRegime::Turbulent);
    }

    #[test]
    fn test_mach_number() {
        let (encoder, _) = setup();

        // Subsonic
        let ma_sub = encoder.mach_number(100.0, 343.0);
        assert!(ma_sub < 1.0);

        // Supersonic
        let ma_super = encoder.mach_number(500.0, 343.0);
        assert!(ma_super > 1.0);
    }

    #[test]
    fn test_flow_creation() {
        let (encoder, genesis) = setup();

        let laminar = encoder.create_flow("Pipe", 1000.0, 0.1, &genesis);
        let turbulent = encoder.create_flow("Jet", 100000.0, 0.2, &genesis);

        assert_eq!(laminar.regime, FlowRegime::Laminar);
        assert_eq!(turbulent.regime, FlowRegime::Turbulent);
    }

    #[test]
    fn test_navier_stokes() {
        let (encoder, _) = setup();
        let ns = encoder.navier_stokes();

        assert!(ns.norm() > 0.0);
    }

    #[test]
    fn test_vortex() {
        let (encoder, genesis) = setup();

        let vortex = encoder.create_vortex(10.0, 0.1, &genesis);
        assert!(vortex.max_velocity > 0.0);
        assert!(vortex.vector.norm() > 0.0);
    }

    #[test]
    fn test_kolmogorov_scale() {
        let (encoder, _) = setup();

        // Typical atmospheric turbulence
        let eta = encoder.kolmogorov_scale(1.5e-5, 0.01);
        // Should be ~ mm scale
        assert!(eta > 1e-4 && eta < 1e-2);
    }

    #[test]
    fn test_boundary_layer() {
        let (encoder, _) = setup();

        let delta = encoder.boundary_layer_thickness(1.0, 10000.0);
        // δ ~ L/√Re ~ 1/100 = 0.01
        assert!((delta - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_shock() {
        let (encoder, genesis) = setup();

        let subsonic = encoder.shock(0.8, &genesis);
        let supersonic = encoder.shock(2.0, &genesis);

        // No shock for subsonic
        assert!(subsonic.norm() < 0.01);
        // Shock exists for supersonic
        assert!(supersonic.norm() > 0.0);
    }

    #[test]
    fn test_bernoulli() {
        let (encoder, _) = setup();
        let bernoulli = encoder.bernoulli();

        // Should combine pressure, velocity, density
        assert!(bernoulli.norm() > 0.0);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analytical Physics Backend
//!
//! Closed-form stress/strain calculations for simple beam geometries.
//! Covers ~80% of fabrication use cases (brackets, pipes, hinges) without
//! requiring external simulation dependencies.

use crate::simulator::*;
use crate::units::{Meters, Newtons};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Invalid material, section, load, or beam geometry supplied to the analytical backend.
#[derive(Debug, Clone, PartialEq)]
pub enum AnalyticalInputError {
    InvalidMaterial(&'static str),
    InvalidCrossSection(&'static str),
    InvalidBeamLength,
    InvalidForce,
}

impl fmt::Display for AnalyticalInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMaterial(field) => write!(f, "invalid material property: {field}"),
            Self::InvalidCrossSection(field) => {
                write!(f, "invalid cross-section dimension: {field}")
            }
            Self::InvalidBeamLength => write!(f, "beam length must be finite and positive"),
            Self::InvalidForce => write!(f, "force must be finite"),
        }
    }
}

impl std::error::Error for AnalyticalInputError {}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

/// Material properties for analytical calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperties {
    pub name: String,
    /// Young's modulus in Pa
    pub elastic_modulus: f64,
    /// Yield strength in Pa
    pub yield_strength: f64,
    /// Poisson's ratio
    pub poissons_ratio: f64,
    /// Density in kg/m³
    pub density: f64,
}

impl MaterialProperties {
    pub fn pla() -> Self {
        Self {
            name: "PLA".to_string(),
            elastic_modulus: 3.5e9,
            yield_strength: 60.0e6,
            poissons_ratio: 0.36,
            density: 1240.0,
        }
    }

    pub fn steel() -> Self {
        Self {
            name: "Steel".to_string(),
            elastic_modulus: 200.0e9,
            yield_strength: 250.0e6,
            poissons_ratio: 0.30,
            density: 7850.0,
        }
    }

    pub fn aluminum() -> Self {
        Self {
            name: "Aluminum".to_string(),
            elastic_modulus: 69.0e9,
            yield_strength: 270.0e6,
            poissons_ratio: 0.33,
            density: 2700.0,
        }
    }

    /// Validate that the material is physically usable by the closed-form model.
    pub fn validate(&self) -> Result<(), AnalyticalInputError> {
        if !finite_positive(self.elastic_modulus) {
            return Err(AnalyticalInputError::InvalidMaterial("elastic_modulus"));
        }
        if !finite_positive(self.yield_strength) {
            return Err(AnalyticalInputError::InvalidMaterial("yield_strength"));
        }
        if !finite_positive(self.density) {
            return Err(AnalyticalInputError::InvalidMaterial("density"));
        }
        if !self.poissons_ratio.is_finite()
            || self.poissons_ratio <= -1.0
            || self.poissons_ratio >= 0.5
        {
            return Err(AnalyticalInputError::InvalidMaterial("poissons_ratio"));
        }
        Ok(())
    }

    /// Shear modulus G = E / (2(1 + ν))
    pub fn shear_modulus(&self) -> f64 {
        self.elastic_modulus / (2.0 * (1.0 + self.poissons_ratio))
    }

    /// Bulk modulus K = E / (3(1 - 2ν))
    pub fn bulk_modulus(&self) -> f64 {
        self.elastic_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))
    }
}

/// Cross-section geometry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossSection {
    Rectangle {
        width: f64,
        height: f64,
    },
    Circle {
        radius: f64,
    },
    HollowCircle {
        outer_radius: f64,
        inner_radius: f64,
    },
    IBeam {
        flange_width: f64,
        flange_height: f64,
        web_height: f64,
        web_thickness: f64,
    },
}

impl CrossSection {
    /// Validate dimensions and hollow-section ordering.
    pub fn validate(&self) -> Result<(), AnalyticalInputError> {
        let invalid = |field| Err(AnalyticalInputError::InvalidCrossSection(field));
        match self {
            Self::Rectangle { width, height } => {
                if !finite_positive(*width) {
                    return invalid("width");
                }
                if !finite_positive(*height) {
                    return invalid("height");
                }
            }
            Self::Circle { radius } => {
                if !finite_positive(*radius) {
                    return invalid("radius");
                }
            }
            Self::HollowCircle {
                outer_radius,
                inner_radius,
            } => {
                if !finite_positive(*outer_radius) {
                    return invalid("outer_radius");
                }
                if !inner_radius.is_finite() || *inner_radius < 0.0 {
                    return invalid("inner_radius");
                }
                if inner_radius >= outer_radius {
                    return invalid("inner_radius must be smaller than outer_radius");
                }
            }
            Self::IBeam {
                flange_width,
                flange_height,
                web_height,
                web_thickness,
            } => {
                for (name, value) in [
                    ("flange_width", *flange_width),
                    ("flange_height", *flange_height),
                    ("web_height", *web_height),
                    ("web_thickness", *web_thickness),
                ] {
                    if !finite_positive(value) {
                        return invalid(name);
                    }
                }
                if web_thickness > flange_width {
                    return invalid("web_thickness must not exceed flange_width");
                }
            }
        }
        Ok(())
    }

    pub fn area(&self) -> f64 {
        match self {
            CrossSection::Rectangle { width, height } => width * height,
            CrossSection::Circle { radius } => std::f64::consts::PI * radius * radius,
            CrossSection::HollowCircle {
                outer_radius,
                inner_radius,
            } => std::f64::consts::PI * (outer_radius * outer_radius - inner_radius * inner_radius),
            CrossSection::IBeam {
                flange_width,
                flange_height,
                web_height,
                web_thickness,
            } => 2.0 * flange_width * flange_height + web_height * web_thickness,
        }
    }

    pub fn moment_of_inertia(&self) -> f64 {
        match self {
            CrossSection::Rectangle { width, height } => width * height.powi(3) / 12.0,
            CrossSection::Circle { radius } => std::f64::consts::PI * radius.powi(4) / 4.0,
            CrossSection::HollowCircle {
                outer_radius,
                inner_radius,
            } => std::f64::consts::PI * (outer_radius.powi(4) - inner_radius.powi(4)) / 4.0,
            CrossSection::IBeam {
                flange_width,
                flange_height,
                web_height,
                web_thickness,
            } => {
                let total_h = 2.0 * flange_height + web_height;
                let outer = flange_width * total_h.powi(3) / 12.0;
                let inner = (flange_width - web_thickness) * web_height.powi(3) / 12.0;
                outer - inner
            }
        }
    }

    pub fn section_modulus(&self) -> f64 {
        match self {
            CrossSection::Rectangle { height, .. } => self.moment_of_inertia() / (height / 2.0),
            CrossSection::Circle { radius } => self.moment_of_inertia() / radius,
            CrossSection::HollowCircle { outer_radius, .. } => {
                self.moment_of_inertia() / outer_radius
            }
            CrossSection::IBeam {
                flange_height,
                web_height,
                ..
            } => {
                let total_h = 2.0 * flange_height + web_height;
                self.moment_of_inertia() / (total_h / 2.0)
            }
        }
    }
}

/// Analytical result from beam analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticalResult {
    /// Axial stress in pascals.
    pub axial_stress: f64,
    /// Bending stress in pascals.
    pub bending_stress: f64,
    /// Shear stress in pascals.
    pub shear_stress: f64,
    /// Maximum beam deflection in metres.
    pub max_deflection: f64,
    /// Von Mises stress in pascals.
    pub von_mises_stress: f64,
    /// Dimensionless yield safety factor.
    pub safety_factor: f64,
}

/// Analytical physics backend — closed-form beam stress/strain
pub struct AnalyticalBackend {
    pub material: MaterialProperties,
    pub cross_section: CrossSection,
    pub beam_length: f64,
    state: SimState,
}

impl AnalyticalBackend {
    pub fn new(
        material: MaterialProperties,
        cross_section: CrossSection,
        beam_length: f64,
    ) -> Self {
        Self {
            material,
            cross_section,
            beam_length,
            state: SimState {
                time: 0.0,
                positions: vec![[0.0; 3]],
                velocities: vec![[0.0; 3]],
                total_energy: 0.0,
            },
        }
    }

    /// Construct a backend through an explicit SI-unit and validity boundary.
    pub fn new_checked(
        material: MaterialProperties,
        cross_section: CrossSection,
        beam_length: Meters,
    ) -> Result<Self, AnalyticalInputError> {
        material.validate()?;
        cross_section.validate()?;
        if !finite_positive(beam_length.get()) {
            return Err(AnalyticalInputError::InvalidBeamLength);
        }
        Ok(Self::new(material, cross_section, beam_length.get()))
    }

    /// Axial stress σ = F / A
    pub fn axial_stress(&self, force: f64) -> f64 {
        force / self.cross_section.area()
    }

    /// Bending stress σ = M / S where M = F * L/2 for point load at center
    pub fn bending_stress(&self, force: f64) -> f64 {
        let moment = force * self.beam_length / 4.0; // Simply supported, center load
        moment / self.cross_section.section_modulus()
    }

    /// Maximum deflection δ = FL³/(48EI) for simply supported beam, center load
    pub fn max_deflection(&self, force: f64) -> f64 {
        let l = self.beam_length;
        let e = self.material.elastic_modulus;
        let i = self.cross_section.moment_of_inertia();
        force * l.powi(3) / (48.0 * e * i)
    }

    /// Shear stress τ = V*Q/(I*b), simplified to τ_avg = V/A
    pub fn shear_stress(&self, shear_force: f64) -> f64 {
        shear_force / self.cross_section.area()
    }

    /// Full analysis for a given load
    pub fn analyze(&self, axial_force: f64, transverse_force: f64) -> AnalyticalResult {
        let sigma_a = self.axial_stress(axial_force);
        let sigma_b = self.bending_stress(transverse_force);
        let tau = self.shear_stress(transverse_force / 2.0); // Reaction = F/2 each end

        // Von Mises: σ_vm = sqrt(σ² + 3τ²) where σ = σ_axial + σ_bending
        let sigma_total = sigma_a + sigma_b;
        let von_mises = (sigma_total.powi(2) + 3.0 * tau.powi(2)).sqrt();

        let safety_factor = if von_mises > 0.0 {
            self.material.yield_strength / von_mises
        } else {
            f64::INFINITY
        };

        AnalyticalResult {
            axial_stress: sigma_a,
            bending_stress: sigma_b,
            shear_stress: tau,
            max_deflection: self.max_deflection(transverse_force),
            von_mises_stress: von_mises,
            safety_factor,
        }
    }

    /// Analyze typed forces after rejecting non-finite inputs.
    pub fn analyze_checked(
        &self,
        axial_force: Newtons,
        transverse_force: Newtons,
    ) -> Result<AnalyticalResult, AnalyticalInputError> {
        if !axial_force.get().is_finite() || !transverse_force.get().is_finite() {
            return Err(AnalyticalInputError::InvalidForce);
        }
        Ok(self.analyze(axial_force.get(), transverse_force.get()))
    }
}

impl PhysicsBackend for AnalyticalBackend {
    fn step(&mut self, dt: f32, forces: &[ForceHV]) -> SimState {
        self.state.time += dt;

        // Sum force magnitudes for analytical calculation
        let total_force: f32 = forces.iter().map(|f| f.magnitude).sum();
        let deflection = self.max_deflection(total_force as f64);

        self.state.positions = vec![[0.0, deflection as f32, 0.0]];
        self.state.total_energy = 0.5 * total_force * deflection as f32;

        self.state.clone()
    }

    fn get_contacts(&self) -> Vec<ContactPoint> {
        // Simply-supported beam: two support reactions
        vec![
            ContactPoint {
                position: [0.0, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                force_magnitude: 0.0, // Set during step
            },
            ContactPoint {
                position: [self.beam_length as f32, 0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                force_magnitude: 0.0,
            },
        ]
    }

    fn get_deformation(&self) -> Option<DeformationField> {
        let n_points = 10;
        let mut displacements = Vec::new();
        let mut strains = Vec::new();
        for i in 0..n_points {
            let x = (i as f64 / (n_points - 1) as f64) * self.beam_length;
            // Parabolic deflection profile for simply-supported beam
            let y = 4.0
                * self.state.positions[0][1] as f64
                * (x / self.beam_length)
                * (1.0 - x / self.beam_length);
            displacements.push([x as f32, y as f32, 0.0]);
            strains.push((y / self.beam_length).abs() as f32);
        }
        Some(DeformationField {
            displacements,
            strains,
        })
    }

    fn get_reaction_force(&self, _point: [f32; 3]) -> f32 {
        // For simply-supported beam, each support carries half the load
        self.state.total_energy // Simplified
    }

    fn reset(&mut self) {
        self.state = SimState {
            time: 0.0,
            positions: vec![[0.0; 3]],
            velocities: vec![[0.0; 3]],
            total_energy: 0.0,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_constructor_rejects_invalid_material() {
        let mut material = MaterialProperties::steel();
        material.elastic_modulus = f64::NAN;
        let error = AnalyticalBackend::new_checked(
            material,
            CrossSection::Rectangle {
                width: 0.01,
                height: 0.01,
            },
            Meters::positive(0.1).unwrap(),
        )
        .err()
        .expect("invalid material must fail");
        assert_eq!(
            error,
            AnalyticalInputError::InvalidMaterial("elastic_modulus")
        );
    }

    #[test]
    fn checked_constructor_rejects_inverted_hollow_section() {
        let error = AnalyticalBackend::new_checked(
            MaterialProperties::steel(),
            CrossSection::HollowCircle {
                outer_radius: 0.01,
                inner_radius: 0.02,
            },
            Meters::positive(0.1).unwrap(),
        )
        .err()
        .expect("inverted hollow section must fail");
        assert!(matches!(
            error,
            AnalyticalInputError::InvalidCrossSection(_)
        ));
    }

    #[test]
    fn checked_analysis_accepts_typed_si_inputs() {
        let backend = AnalyticalBackend::new_checked(
            MaterialProperties::steel(),
            CrossSection::Rectangle {
                width: 0.01,
                height: 0.02,
            },
            Meters::positive(0.1).unwrap(),
        )
        .unwrap();
        let result = backend
            .analyze_checked(Newtons::new(0.0).unwrap(), Newtons::new(1000.0).unwrap())
            .unwrap();
        assert!(result.safety_factor.is_finite());
        assert!(result.max_deflection >= 0.0);
    }

    #[test]
    fn test_pla_properties() {
        let pla = MaterialProperties::pla();
        assert!((pla.elastic_modulus - 3.5e9).abs() < 1e6);
        assert!(pla.shear_modulus() > 0.0);
        assert!(pla.bulk_modulus() > 0.0);
    }

    #[test]
    fn test_steel_stiffer_than_pla() {
        let steel = MaterialProperties::steel();
        let pla = MaterialProperties::pla();
        assert!(steel.elastic_modulus > pla.elastic_modulus);
    }

    #[test]
    fn test_rectangle_area() {
        let cs = CrossSection::Rectangle {
            width: 0.01,
            height: 0.02,
        };
        assert!((cs.area() - 0.0002).abs() < 1e-10);
    }

    #[test]
    fn test_circle_area() {
        let cs = CrossSection::Circle { radius: 0.01 };
        let expected = std::f64::consts::PI * 0.0001;
        assert!((cs.area() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_hollow_circle() {
        let solid = CrossSection::Circle { radius: 0.02 };
        let hollow = CrossSection::HollowCircle {
            outer_radius: 0.02,
            inner_radius: 0.01,
        };
        assert!(hollow.area() < solid.area());
        assert!(hollow.moment_of_inertia() < solid.moment_of_inertia());
    }

    #[test]
    fn test_axial_stress() {
        let backend = AnalyticalBackend::new(
            MaterialProperties::steel(),
            CrossSection::Rectangle {
                width: 0.01,
                height: 0.01,
            },
            1.0,
        );
        let stress = backend.axial_stress(1000.0); // 1000N on 1cm² = 10 MPa
        assert!((stress - 10.0e6).abs() < 1e3);
    }

    #[test]
    fn test_deflection_steel_vs_pla() {
        let steel_backend = AnalyticalBackend::new(
            MaterialProperties::steel(),
            CrossSection::Rectangle {
                width: 0.01,
                height: 0.01,
            },
            0.5,
        );
        let pla_backend = AnalyticalBackend::new(
            MaterialProperties::pla(),
            CrossSection::Rectangle {
                width: 0.01,
                height: 0.01,
            },
            0.5,
        );
        let force = 100.0;
        assert!(steel_backend.max_deflection(force) < pla_backend.max_deflection(force));
    }

    #[test]
    fn test_analyze_safety_factor() {
        let backend = AnalyticalBackend::new(
            MaterialProperties::steel(),
            CrossSection::Rectangle {
                width: 0.02,
                height: 0.02,
            },
            0.5,
        );
        let result = backend.analyze(100.0, 500.0);
        assert!(
            result.safety_factor > 1.0,
            "Steel beam should be safe under moderate load"
        );
        assert!(result.von_mises_stress > 0.0);
    }

    #[test]
    fn test_physics_backend_step() {
        let mut backend = AnalyticalBackend::new(
            MaterialProperties::pla(),
            CrossSection::Rectangle {
                width: 0.02,
                height: 0.01,
            },
            0.3,
        );
        let force = ForceHV {
            force_vector: tension_hv(),
            magnitude: 50.0,
            application_point: [0.15, 0.0, 0.0],
            expected_resistance: 25.0,
        };
        let state = backend.step(0.01, &[force]);
        assert!(state.time > 0.0);
        assert!(state.total_energy > 0.0);
    }

    #[test]
    fn test_deformation_field() {
        let mut backend = AnalyticalBackend::new(
            MaterialProperties::pla(),
            CrossSection::Rectangle {
                width: 0.02,
                height: 0.01,
            },
            0.3,
        );
        let force = ForceHV {
            force_vector: compression_hv(),
            magnitude: 100.0,
            application_point: [0.15, 0.0, 0.0],
            expected_resistance: 50.0,
        };
        backend.step(0.01, &[force]);
        let df = backend.get_deformation().unwrap();
        assert_eq!(df.displacements.len(), 10);
        assert!(df.max_strain() >= 0.0);
    }

    #[test]
    fn test_surprise_loop() {
        let mut backend = AnalyticalBackend::new(
            MaterialProperties::steel(),
            CrossSection::Rectangle {
                width: 0.02,
                height: 0.02,
            },
            0.5,
        );
        let force = ForceHV {
            force_vector: bending_hv(),
            magnitude: 200.0,
            application_point: [0.25, 0.0, 0.0],
            expected_resistance: 100.0,
        };
        backend.step(0.01, &[force.clone()]);
        let actual = backend.get_reaction_force([0.0, 0.0, 0.0]);
        let surprise = force.surprise(actual);
        // Surprise should be finite and non-negative
        assert!(surprise >= 0.0);
        assert!(surprise.is_finite());
    }
}

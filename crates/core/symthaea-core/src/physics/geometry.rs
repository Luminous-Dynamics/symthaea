// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Geometry Optimization: Form Factor Design for Spark Engine
//!
//! Optimizes the physical geometry of the Spark Engine for:
//! - Thermal efficiency
//! - Structural integrity
//! - Shielding effectiveness
//! - Manufacturing feasibility
//!
//! ## Geometry Options
//!
//! ```text
//! SPHERICAL                    CYLINDRICAL
//! ┌─────────────┐              ┌──────────────────┐
//! │   ╭─────╮   │              │ ╔══════════════╗ │
//! │  ╱       ╲  │              │ ║   Core       ║ │
//! │ │  Core   │ │              │ ║              ║ │
//! │  ╲       ╱  │              │ ╚══════════════╝ │
//! │   ╰─────╯   │              └──────────────────┘
//! └─────────────┘
//! • Minimum surface area       • Easier manufacturing
//! • Uniform stress             • Modular design
//! • Optimal for small scale    • Optimal for industrial
//! ```
//!
//! ## Key Metrics
//!
//! - **Surface-to-volume ratio**: Lower = less heat loss
//! - **Stress concentration**: Lower = safer
//! - **Neutron escape fraction**: Lower = more shielding efficient
//! - **Manufacturing complexity**: Lower = cheaper

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;
use std::f64::consts::PI;

/// Geometry type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryType {
    /// Spherical containment
    Spherical,
    /// Cylindrical containment (aspect ratio = L/D)
    Cylindrical,
    /// Toroidal (donut shape)
    Toroidal,
    /// Cubic/rectangular
    Rectangular,
}

/// Complete geometry specification
#[derive(Debug, Clone)]
pub struct EngineGeometry {
    /// Geometry type
    pub geometry_type: GeometryType,
    /// Core radius (m)
    pub core_radius: f64,
    /// Interface thickness (m)
    pub interface_thickness: f64,
    /// Shell thickness (m)
    pub shell_thickness: f64,
    /// Shielding thickness (m)
    pub shielding_thickness: f64,
    /// Length (for cylindrical) or None (for spherical)
    pub length: Option<f64>,
    /// Total outer radius (m)
    pub outer_radius: f64,
}

impl EngineGeometry {
    /// Create spherical geometry
    pub fn spherical(
        core_radius: f64,
        interface_thickness: f64,
        shell_thickness: f64,
        shielding_thickness: f64,
    ) -> Self {
        let outer_radius =
            core_radius + interface_thickness + shell_thickness + shielding_thickness;
        Self {
            geometry_type: GeometryType::Spherical,
            core_radius,
            interface_thickness,
            shell_thickness,
            shielding_thickness,
            length: None,
            outer_radius,
        }
    }

    /// Create cylindrical geometry with given aspect ratio (L/D)
    pub fn cylindrical(
        core_radius: f64,
        interface_thickness: f64,
        shell_thickness: f64,
        shielding_thickness: f64,
        aspect_ratio: f64,
    ) -> Self {
        let outer_radius =
            core_radius + interface_thickness + shell_thickness + shielding_thickness;
        let length = 2.0 * outer_radius * aspect_ratio;
        Self {
            geometry_type: GeometryType::Cylindrical,
            core_radius,
            interface_thickness,
            shell_thickness,
            shielding_thickness,
            length: Some(length),
            outer_radius,
        }
    }

    /// Total volume (m³)
    pub fn total_volume(&self) -> f64 {
        match self.geometry_type {
            GeometryType::Spherical => (4.0 / 3.0) * PI * self.outer_radius.powi(3),
            GeometryType::Cylindrical => {
                let l = self.length.unwrap_or(2.0 * self.outer_radius);
                PI * self.outer_radius.powi(2) * l
            }
            GeometryType::Toroidal => {
                // V = 2 * π² * R * r² where R = major, r = minor radius
                let r_minor = self.outer_radius;
                let r_major = 2.0 * r_minor; // Typical aspect
                2.0 * PI.powi(2) * r_major * r_minor.powi(2)
            }
            GeometryType::Rectangular => {
                // Cube with edge = 2 * outer_radius
                (2.0 * self.outer_radius).powi(3)
            }
        }
    }

    /// Core volume (m³)
    pub fn core_volume(&self) -> f64 {
        match self.geometry_type {
            GeometryType::Spherical => (4.0 / 3.0) * PI * self.core_radius.powi(3),
            GeometryType::Cylindrical => {
                let l = self.length.unwrap_or(2.0 * self.outer_radius);
                // Subtract end caps thickness
                let core_length = l - 2.0
                    * (self.interface_thickness + self.shell_thickness + self.shielding_thickness);
                PI * self.core_radius.powi(2) * core_length.max(0.0)
            }
            _ => (4.0 / 3.0) * PI * self.core_radius.powi(3), // Approximate
        }
    }

    /// Shell volume (m³)
    pub fn shell_volume(&self) -> f64 {
        let r_inner = self.core_radius + self.interface_thickness;
        let r_outer = r_inner + self.shell_thickness;

        match self.geometry_type {
            GeometryType::Spherical => (4.0 / 3.0) * PI * (r_outer.powi(3) - r_inner.powi(3)),
            GeometryType::Cylindrical => {
                let l = self.length.unwrap_or(2.0 * self.outer_radius);
                PI * (r_outer.powi(2) - r_inner.powi(2)) * l
            }
            _ => (4.0 / 3.0) * PI * (r_outer.powi(3) - r_inner.powi(3)),
        }
    }

    /// Outer surface area (m²)
    pub fn outer_surface_area(&self) -> f64 {
        match self.geometry_type {
            GeometryType::Spherical => 4.0 * PI * self.outer_radius.powi(2),
            GeometryType::Cylindrical => {
                let l = self.length.unwrap_or(2.0 * self.outer_radius);
                // Lateral + 2 end caps
                2.0 * PI * self.outer_radius * l + 2.0 * PI * self.outer_radius.powi(2)
            }
            GeometryType::Toroidal => {
                let r_minor = self.outer_radius;
                let r_major = 2.0 * r_minor;
                4.0 * PI.powi(2) * r_major * r_minor
            }
            GeometryType::Rectangular => 6.0 * (2.0 * self.outer_radius).powi(2),
        }
    }

    /// Surface-to-volume ratio (1/m)
    pub fn surface_to_volume(&self) -> f64 {
        self.outer_surface_area() / self.total_volume()
    }

    /// Compactness score (0-1, higher = more compact)
    /// Sphere has compactness 1.0 (minimum S/V for given volume)
    pub fn compactness(&self) -> f64 {
        // Sphere S/V = 3/r, for same volume:
        let equivalent_radius = (3.0 * self.total_volume() / (4.0 * PI)).powf(1.0 / 3.0);
        let sphere_sv = 3.0 / equivalent_radius;
        sphere_sv / self.surface_to_volume()
    }

    /// Structural factor (0-1, higher = better stress distribution)
    pub fn structural_factor(&self) -> f64 {
        match self.geometry_type {
            GeometryType::Spherical => 1.0,    // Optimal stress distribution
            GeometryType::Cylindrical => 0.85, // Stress concentration at ends
            GeometryType::Toroidal => 0.75,    // Complex stress state
            GeometryType::Rectangular => 0.5,  // High corner stresses
        }
    }

    /// Manufacturing complexity (0-1, lower = easier to manufacture)
    pub fn manufacturing_complexity(&self) -> f64 {
        match self.geometry_type {
            GeometryType::Spherical => 0.7,   // Difficult to machine/weld
            GeometryType::Cylindrical => 0.3, // Standard fabrication
            GeometryType::Toroidal => 0.9,    // Very difficult
            GeometryType::Rectangular => 0.2, // Easiest
        }
    }

    /// Overall geometry score
    pub fn geometry_score(&self, weights: &GeometryWeights) -> f64 {
        let compactness = self.compactness();
        let structural = self.structural_factor();
        let manufacturing = 1.0 - self.manufacturing_complexity();

        compactness * weights.compactness
            + structural * weights.structural
            + manufacturing * weights.manufacturing
    }
}

/// Weights for geometry scoring
#[derive(Debug, Clone)]
pub struct GeometryWeights {
    /// Weight for compactness (thermal efficiency)
    pub compactness: f64,
    /// Weight for structural integrity
    pub structural: f64,
    /// Weight for manufacturing ease
    pub manufacturing: f64,
}

impl Default for GeometryWeights {
    fn default() -> Self {
        Self {
            compactness: 0.4,
            structural: 0.35,
            manufacturing: 0.25,
        }
    }
}

impl GeometryWeights {
    /// Consumer-focused (small, manufacturable)
    pub fn consumer() -> Self {
        Self {
            compactness: 0.5,
            structural: 0.3,
            manufacturing: 0.2,
        }
    }

    /// Industrial-focused (structural, scalable)
    pub fn industrial() -> Self {
        Self {
            compactness: 0.2,
            structural: 0.5,
            manufacturing: 0.3,
        }
    }

    /// Research prototype (cost-optimized)
    pub fn prototype() -> Self {
        Self {
            compactness: 0.1,
            structural: 0.3,
            manufacturing: 0.6,
        }
    }
}

/// Geometry optimization result
#[derive(Debug, Clone)]
pub struct GeometryOptimization {
    /// Optimal geometry
    pub geometry: EngineGeometry,
    /// Geometry score
    pub score: f64,
    /// Comparison with alternatives
    pub alternatives: Vec<(GeometryType, f64)>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Geometry optimizer
#[derive(Debug, Clone)]
pub struct GeometryOptimizer {
    /// Concept vectors
    pub compact: ContinuousHV,
    pub structural: ContinuousHV,
    pub manufacturable: ContinuousHV,
}

impl GeometryOptimizer {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            compact: genesis.hv("geometry::compact", PHYSICS_DIM),
            structural: genesis.hv("geometry::structural", PHYSICS_DIM),
            manufacturable: genesis.hv("geometry::manufacturable", PHYSICS_DIM),
        }
    }

    /// Optimize geometry for given constraints
    pub fn optimize(
        &self,
        power_kw: f64,
        target_power_density_kw_m3: f64,
        weights: &GeometryWeights,
    ) -> GeometryOptimization {
        // Required core volume based on power density
        let core_volume = power_kw / target_power_density_kw_m3;
        let core_radius = (3.0 * core_volume / (4.0 * PI)).powf(1.0 / 3.0);

        // Standard layer thicknesses (scale with core radius)
        let interface = 0.001 + core_radius * 0.01; // 1mm + 1% of radius
        let shell = 0.005 + core_radius * 0.05; // 5mm + 5% of radius
        let shielding = 0.1 + core_radius * 0.2; // 10cm + 20% of radius

        // Evaluate all geometries
        let mut results = Vec::new();

        // Spherical
        let sphere = EngineGeometry::spherical(core_radius, interface, shell, shielding);
        let sphere_score = sphere.geometry_score(weights);
        results.push((GeometryType::Spherical, sphere_score, sphere.clone()));

        // Cylindrical with various aspect ratios
        for aspect in [1.0, 1.5, 2.0, 3.0] {
            let cyl = EngineGeometry::cylindrical(core_radius, interface, shell, shielding, aspect);
            let cyl_score = cyl.geometry_score(weights);
            results.push((GeometryType::Cylindrical, cyl_score, cyl));
        }

        // Find best
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let (best_type, best_score, best_geom) = results[0].clone();

        // Build alternatives list
        let alternatives: Vec<(GeometryType, f64)> =
            results.iter().map(|(t, s, _)| (*t, *s)).collect();

        // Generate recommendations
        let mut recommendations = Vec::new();

        if best_type == GeometryType::Spherical {
            recommendations.push("Spherical geometry optimal for thermal efficiency".to_string());
            recommendations.push("Consider hemispherical access for maintenance".to_string());
        } else {
            recommendations.push(format!(
                "Cylindrical geometry with L/D = {:.1} optimal for this application",
                best_geom.length.unwrap_or(0.0) / (2.0 * best_geom.outer_radius)
            ));
            recommendations.push("End caps require reinforcement for pressure".to_string());
        }

        if best_geom.outer_radius > 0.5 {
            recommendations.push("Large scale - consider modular construction".to_string());
        }

        GeometryOptimization {
            geometry: best_geom,
            score: best_score,
            alternatives,
            recommendations,
        }
    }

    /// Scale geometry for different power levels
    pub fn scale_geometry(&self, base: &EngineGeometry, power_ratio: f64) -> EngineGeometry {
        // Volume scales with power, radius scales with cube root
        let scale = power_ratio.powf(1.0 / 3.0);

        match base.geometry_type {
            GeometryType::Spherical => EngineGeometry::spherical(
                base.core_radius * scale,
                base.interface_thickness * scale,
                base.shell_thickness * scale,
                base.shielding_thickness * scale,
            ),
            GeometryType::Cylindrical => {
                let aspect =
                    base.length.unwrap_or(2.0 * base.outer_radius) / (2.0 * base.outer_radius);
                EngineGeometry::cylindrical(
                    base.core_radius * scale,
                    base.interface_thickness * scale,
                    base.shell_thickness * scale,
                    base.shielding_thickness * scale,
                    aspect,
                )
            }
            _ => base.clone(),
        }
    }
}

/// Dimensional analysis for engine sizing
#[derive(Debug, Clone)]
pub struct EngineDimensions {
    /// Power output (kW)
    pub power_kw: f64,
    /// Total outer diameter (m)
    pub outer_diameter_m: f64,
    /// Total height/length (m)
    pub height_m: f64,
    /// Total mass estimate (kg)
    pub mass_kg: f64,
    /// Power density (kW/m³)
    pub power_density_kw_m3: f64,
    /// Specific power (kW/kg)
    pub specific_power_kw_kg: f64,
}

impl EngineDimensions {
    /// Calculate dimensions from geometry
    pub fn from_geometry(geom: &EngineGeometry, power_kw: f64, avg_density_kg_m3: f64) -> Self {
        let outer_diameter = 2.0 * geom.outer_radius;
        let height = geom.length.unwrap_or(outer_diameter);
        let volume = geom.total_volume();
        let mass = volume * avg_density_kg_m3;

        Self {
            power_kw,
            outer_diameter_m: outer_diameter,
            height_m: height,
            mass_kg: mass,
            power_density_kw_m3: power_kw / volume,
            specific_power_kw_kg: power_kw / mass,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spherical_geometry() {
        let geom = EngineGeometry::spherical(0.05, 0.001, 0.01, 0.1);

        assert!((geom.outer_radius - 0.161).abs() < 0.001);
        assert!(geom.total_volume() > 0.0);
        assert!(geom.core_volume() < geom.total_volume());
        assert!(geom.compactness() > 0.99); // Sphere is optimal
    }

    #[test]
    fn test_cylindrical_geometry() {
        let geom = EngineGeometry::cylindrical(0.05, 0.001, 0.01, 0.1, 2.0);

        assert!(geom.length.is_some());
        assert!(geom.compactness() < 1.0); // Less compact than sphere
        assert!(geom.manufacturing_complexity() < 0.5); // Easier to make
    }

    #[test]
    fn test_geometry_scoring() {
        let sphere = EngineGeometry::spherical(0.05, 0.001, 0.01, 0.1);
        let cyl = EngineGeometry::cylindrical(0.05, 0.001, 0.01, 0.1, 2.0);

        let weights = GeometryWeights::default();

        let sphere_score = sphere.geometry_score(&weights);
        let cyl_score = cyl.geometry_score(&weights);

        // Both should have positive scores
        assert!(sphere_score > 0.0);
        assert!(cyl_score > 0.0);
    }

    #[test]
    fn test_geometry_optimizer() {
        let genesis = GenesisSeed::from_phrase("geometry test");
        let optimizer = GeometryOptimizer::from_genesis(&genesis);

        let result = optimizer.optimize(
            5.0,   // 5 kW
            100.0, // 100 kW/m³ target
            &GeometryWeights::default(),
        );

        assert!(result.score > 0.0);
        assert!(!result.alternatives.is_empty());
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_scaling() {
        let genesis = GenesisSeed::from_phrase("scaling test");
        let optimizer = GeometryOptimizer::from_genesis(&genesis);

        let base = EngineGeometry::spherical(0.05, 0.001, 0.01, 0.1);
        let scaled = optimizer.scale_geometry(&base, 8.0); // 8x power

        // Volume should scale with power
        let volume_ratio = scaled.total_volume() / base.total_volume();
        assert!((volume_ratio - 8.0).abs() < 0.1);
    }

    #[test]
    fn test_dimensions() {
        let geom = EngineGeometry::spherical(0.05, 0.001, 0.01, 0.1);
        let dims = EngineDimensions::from_geometry(&geom, 5.0, 5000.0);

        assert!(dims.outer_diameter_m > 0.0);
        assert!(dims.mass_kg > 0.0);
        assert!(dims.power_density_kw_m3 > 0.0);
        assert!(dims.specific_power_kw_kg > 0.0);
    }
}

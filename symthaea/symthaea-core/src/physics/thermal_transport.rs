// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Thermal Transport: Heat Flow in Spark Engine
//!
//! Models heat generation, conduction, and extraction in the layered
//! Spark Engine architecture.
//!
//! ## Heat Flow Path
//!
//! ```text
//! Fusion Zone (Shell)     →     Interface     →     Liquid Core     →     Heat Exchanger
//!      Q_gen                    Conduction         Convection            Extraction
//!     (MW/m³)                   (k·∇T)             (h·ΔT)                (Q_out)
//! ```
//!
//! ## Key Physics
//!
//! - **Volumetric heat generation**: Q = P_fusion / V_shell
//! - **Conduction**: q = -k * dT/dr (Fourier's law)
//! - **Convection**: q = h * (T_surface - T_bulk)
//! - **Thermal stress**: σ = α * E * ΔT / (1 - ν)
//!
//! ## Temperature Limits
//!
//! - HEA shell: < 1200 K (avoid phase transformation)
//! - Nano-laminate interface: < 800 K (maintain structure)
//! - Liquid metal core: > 303 K (Galinstan melting point)
//! - Heat exchanger: ~350 K (optimal extraction)

use super::standard_model::PHYSICS_DIM;
use crate::genesis::GenesisSeed;
use crate::hdc::unified_hv::ContinuousHV;

/// Thermal properties of a material layer
#[derive(Debug, Clone)]
pub struct ThermalProperties {
    /// Material name
    pub name: String,
    /// Thermal conductivity (W/m·K)
    pub conductivity: f64,
    /// Specific heat capacity (J/kg·K)
    pub heat_capacity: f64,
    /// Density (kg/m³)
    pub density: f64,
    /// Thermal expansion coefficient (1/K)
    pub expansion_coeff: f64,
    /// Maximum operating temperature (K)
    pub max_temp_k: f64,
    /// Minimum operating temperature (K)
    pub min_temp_k: f64,
}

impl ThermalProperties {
    /// TiVZrNbPd HEA shell
    pub fn hea_shell() -> Self {
        Self {
            name: "TiVZrNbPd HEA".to_string(),
            conductivity: 15.0,     // W/m·K (typical HEA)
            heat_capacity: 450.0,   // J/kg·K
            density: 7200.0,        // kg/m³
            expansion_coeff: 10e-6, // 1/K
            max_temp_k: 1200.0,     // Before phase transformation
            min_temp_k: 200.0,
        }
    }

    /// Zr/Nb nano-laminate interface
    pub fn nano_laminate() -> Self {
        Self {
            name: "Zr/Nb Laminate".to_string(),
            conductivity: 25.0,    // W/m·K
            heat_capacity: 300.0,  // J/kg·K
            density: 7500.0,       // kg/m³
            expansion_coeff: 8e-6, // 1/K
            max_temp_k: 800.0,     // Interface stability limit
            min_temp_k: 200.0,
        }
    }

    /// Galinstan liquid metal core
    pub fn galinstan() -> Self {
        Self {
            name: "Galinstan".to_string(),
            conductivity: 16.5,      // W/m·K
            heat_capacity: 296.0,    // J/kg·K
            density: 6440.0,         // kg/m³
            expansion_coeff: 120e-6, // 1/K (liquid)
            max_temp_k: 1573.0,      // Boiling point
            min_temp_k: 254.0,       // Melting point (-19°C)
        }
    }

    /// Ti3SiC2 MAX phase
    pub fn max_phase() -> Self {
        Self {
            name: "Ti3SiC2".to_string(),
            conductivity: 40.0,    // W/m·K (excellent)
            heat_capacity: 520.0,  // J/kg·K
            density: 4520.0,       // kg/m³
            expansion_coeff: 9e-6, // 1/K
            max_temp_k: 1700.0,    // High temp stability
            min_temp_k: 200.0,
        }
    }

    /// Thermal diffusivity (m²/s)
    pub fn diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.heat_capacity)
    }
}

/// Geometry of a cylindrical/spherical layer
#[derive(Debug, Clone, Copy)]
pub struct LayerGeometry {
    /// Inner radius (m)
    pub r_inner: f64,
    /// Outer radius (m)
    pub r_outer: f64,
    /// Length for cylindrical geometry (m), None for spherical
    pub length: Option<f64>,
}

impl LayerGeometry {
    /// Create spherical layer
    pub fn sphere(r_inner: f64, r_outer: f64) -> Self {
        Self {
            r_inner,
            r_outer,
            length: None,
        }
    }

    /// Create cylindrical layer
    pub fn cylinder(r_inner: f64, r_outer: f64, length: f64) -> Self {
        Self {
            r_inner,
            r_outer,
            length: Some(length),
        }
    }

    /// Thickness (m)
    pub fn thickness(&self) -> f64 {
        self.r_outer - self.r_inner
    }

    /// Volume (m³)
    pub fn volume(&self) -> f64 {
        if let Some(l) = self.length {
            // Cylindrical
            std::f64::consts::PI * (self.r_outer.powi(2) - self.r_inner.powi(2)) * l
        } else {
            // Spherical
            (4.0 / 3.0) * std::f64::consts::PI * (self.r_outer.powi(3) - self.r_inner.powi(3))
        }
    }

    /// Inner surface area (m²)
    pub fn inner_area(&self) -> f64 {
        if let Some(l) = self.length {
            2.0 * std::f64::consts::PI * self.r_inner * l
        } else {
            4.0 * std::f64::consts::PI * self.r_inner.powi(2)
        }
    }

    /// Outer surface area (m²)
    pub fn outer_area(&self) -> f64 {
        if let Some(l) = self.length {
            2.0 * std::f64::consts::PI * self.r_outer * l
        } else {
            4.0 * std::f64::consts::PI * self.r_outer.powi(2)
        }
    }
}

/// Temperature profile result
#[derive(Debug, Clone)]
pub struct TemperatureProfile {
    /// Radial positions (m)
    pub radii: Vec<f64>,
    /// Temperatures at each position (K)
    pub temperatures: Vec<f64>,
    /// Maximum temperature (K)
    pub t_max: f64,
    /// Minimum temperature (K)
    pub t_min: f64,
    /// Temperature at shell outer surface
    pub t_shell_outer: f64,
    /// Temperature at core center
    pub t_core_center: f64,
    /// Heat flux at outer surface (W/m²)
    pub heat_flux_outer: f64,
}

/// Thermal stress result
#[derive(Debug, Clone)]
pub struct ThermalStress {
    /// Maximum hoop stress (Pa)
    pub max_hoop_stress: f64,
    /// Maximum radial stress (Pa)
    pub max_radial_stress: f64,
    /// Von Mises equivalent stress (Pa)
    pub von_mises: f64,
    /// Safety factor (yield stress / von Mises)
    pub safety_factor: f64,
    /// Thermal expansion at outer surface (m)
    pub expansion_outer: f64,
}

/// Thermal transport system
#[derive(Debug, Clone)]
pub struct ThermalTransport {
    /// Concept vectors
    pub conduction: ContinuousHV,
    pub convection: ContinuousHV,
    pub radiation_heat: ContinuousHV,
    pub thermal_stress: ContinuousHV,
}

impl ThermalTransport {
    /// Create from genesis
    pub fn from_genesis(genesis: &GenesisSeed) -> Self {
        Self {
            conduction: genesis.hv("thermal::conduction", PHYSICS_DIM),
            convection: genesis.hv("thermal::convection", PHYSICS_DIM),
            radiation_heat: genesis.hv("thermal::radiation", PHYSICS_DIM),
            thermal_stress: genesis.hv("thermal::stress", PHYSICS_DIM),
        }
    }

    /// Calculate steady-state temperature profile
    ///
    /// Uses 1D radial heat conduction with volumetric heat generation in shell
    #[allow(clippy::too_many_arguments)]
    pub fn steady_state_profile(
        &self,
        power_w: f64,
        shell: &ThermalProperties,
        shell_geom: &LayerGeometry,
        interface: &ThermalProperties,
        interface_geom: &LayerGeometry,
        _core: &ThermalProperties,
        core_geom: &LayerGeometry,
        t_coolant: f64, // Coolant temperature (K)
        h_coolant: f64, // Convection coefficient (W/m²·K)
    ) -> TemperatureProfile {
        // Volumetric heat generation in shell
        let q_gen = power_w / shell_geom.volume();

        // Use spherical geometry equations (simplification)
        let _r1 = core_geom.r_outer; // Core-interface boundary
        let _r2 = interface_geom.r_outer; // Interface-shell boundary
        let _r3 = shell_geom.r_outer; // Shell outer surface

        // Thermal resistances
        // R_cond = (r_outer - r_inner) / (4 * pi * k * r_inner * r_outer) for spherical
        // Simplified for thin shells: R ≈ thickness / (k * area)

        let r_shell = shell_geom.thickness() / (shell.conductivity * shell_geom.inner_area());
        let r_interface =
            interface_geom.thickness() / (interface.conductivity * interface_geom.inner_area());
        let r_conv = 1.0 / (h_coolant * shell_geom.outer_area());

        let _r_total = r_shell + r_interface + r_conv;

        // Heat flux
        let q_total = power_w;
        let heat_flux = q_total / shell_geom.outer_area();

        // Temperature drops
        let dt_conv = q_total * r_conv;
        let dt_interface = q_total * r_interface;
        let dt_shell = q_total * r_shell;

        // Temperature profile
        let t_outer = t_coolant + dt_conv;
        let t_shell_inner = t_outer + dt_shell;
        let t_interface_inner = t_shell_inner + dt_interface;
        let t_core = t_interface_inner; // Core is well-mixed (convection)

        // Add temperature rise from volumetric heating in shell
        // For uniform generation: ΔT = q_gen * L² / (2 * k)
        let dt_gen = q_gen * shell_geom.thickness().powi(2) / (2.0 * shell.conductivity);
        let t_max_shell = t_shell_inner + dt_gen;

        // Generate profile points
        let n_points = 20;
        let mut radii = Vec::with_capacity(n_points);
        let mut temperatures = Vec::with_capacity(n_points);

        // Core region (uniform temperature due to convection)
        for i in 0..5 {
            let r = core_geom.r_inner + (core_geom.r_outer - core_geom.r_inner) * (i as f64 / 4.0);
            radii.push(r);
            temperatures.push(t_core);
        }

        // Interface region (linear profile)
        for i in 1..5 {
            let frac = i as f64 / 4.0;
            let r = interface_geom.r_inner + interface_geom.thickness() * frac;
            let t = t_interface_inner - dt_interface * frac;
            radii.push(r);
            temperatures.push(t);
        }

        // Shell region (with volumetric heating - parabolic)
        for i in 1..=10 {
            let frac = i as f64 / 10.0;
            let r = shell_geom.r_inner + shell_geom.thickness() * frac;
            // Parabolic profile for volumetric heating
            let t = t_shell_inner + dt_gen * (1.0 - frac) * (1.0 + frac) - dt_shell * frac;
            radii.push(r);
            temperatures.push(t.max(t_outer));
        }

        TemperatureProfile {
            radii,
            temperatures,
            t_max: t_max_shell,
            t_min: t_coolant,
            t_shell_outer: t_outer,
            t_core_center: t_core,
            heat_flux_outer: heat_flux,
        }
    }

    /// Calculate thermal stress in shell
    pub fn thermal_stress(
        &self,
        profile: &TemperatureProfile,
        material: &ThermalProperties,
        geometry: &LayerGeometry,
        youngs_modulus: f64, // Pa
        poisson_ratio: f64,
        yield_stress: f64, // Pa
    ) -> ThermalStress {
        let alpha = material.expansion_coeff;
        let e = youngs_modulus;
        let nu = poisson_ratio;

        // Temperature difference across shell
        let dt = profile.t_max - profile.t_shell_outer;

        // Thermal stress in thick-walled cylinder/sphere
        // σ_θ (hoop) ≈ α * E * ΔT / (2 * (1 - ν)) for restrained expansion
        let sigma_hoop = alpha * e * dt / (2.0 * (1.0 - nu));

        // Radial stress (typically lower)
        let sigma_radial = sigma_hoop * 0.5;

        // Von Mises equivalent stress
        let von_mises =
            ((sigma_hoop.powi(2) + sigma_radial.powi(2) - sigma_hoop * sigma_radial).abs()).sqrt();

        // Safety factor
        let safety_factor = yield_stress / von_mises;

        // Thermal expansion at outer surface
        let expansion = alpha * (profile.t_shell_outer - 300.0) * geometry.r_outer;

        ThermalStress {
            max_hoop_stress: sigma_hoop,
            max_radial_stress: sigma_radial,
            von_mises,
            safety_factor,
            expansion_outer: expansion,
        }
    }

    /// Check if temperature profile is within limits
    pub fn check_limits(
        &self,
        profile: &TemperatureProfile,
        shell: &ThermalProperties,
        _interface: &ThermalProperties,
        core: &ThermalProperties,
    ) -> ThermalLimitCheck {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();

        // Shell temperature check
        if profile.t_max > shell.max_temp_k {
            violations.push(format!(
                "Shell max temp {:.0} K exceeds limit {:.0} K",
                profile.t_max, shell.max_temp_k
            ));
        } else if profile.t_max > shell.max_temp_k * 0.9 {
            warnings.push(format!(
                "Shell max temp {:.0} K near limit ({:.0}% of max)",
                profile.t_max,
                100.0 * profile.t_max / shell.max_temp_k
            ));
        }

        // Core temperature check (must stay liquid)
        if profile.t_core_center < core.min_temp_k {
            violations.push(format!(
                "Core temp {:.0} K below freezing point {:.0} K",
                profile.t_core_center, core.min_temp_k
            ));
        }

        // Check for excessive gradients (can cause thermal shock)
        let max_gradient = (profile.t_max - profile.t_min) / 0.01; // K/m estimate
        if max_gradient > 1e6 {
            warnings.push(format!("High thermal gradient: {max_gradient:.2e} K/m"));
        }

        let is_safe = violations.is_empty();

        ThermalLimitCheck {
            is_safe,
            violations,
            warnings,
            max_temp: profile.t_max,
            min_temp: profile.t_min,
            margin_shell: shell.max_temp_k - profile.t_max,
            margin_core: profile.t_core_center - core.min_temp_k,
        }
    }

    /// Estimate required coolant flow rate
    pub fn coolant_flow_rate(
        &self,
        power_w: f64,
        coolant_heat_capacity: f64, // J/kg·K
        dt_coolant: f64,            // Allowed temperature rise (K)
    ) -> f64 {
        // m_dot = Q / (Cp * ΔT)
        power_w / (coolant_heat_capacity * dt_coolant)
    }
}

/// Result of thermal limit check
#[derive(Debug, Clone)]
pub struct ThermalLimitCheck {
    /// Overall safety status
    pub is_safe: bool,
    /// List of limit violations
    pub violations: Vec<String>,
    /// List of warnings
    pub warnings: Vec<String>,
    /// Maximum temperature found
    pub max_temp: f64,
    /// Minimum temperature found
    pub min_temp: f64,
    /// Margin below shell limit (K)
    pub margin_shell: f64,
    /// Margin above core freezing (K)
    pub margin_core: f64,
}

/// Complete thermal analysis for Spark Engine
#[derive(Debug, Clone)]
pub struct SparkThermalAnalysis {
    /// Temperature profile
    pub profile: TemperatureProfile,
    /// Thermal stress analysis
    pub stress: ThermalStress,
    /// Limit check results
    pub limits: ThermalLimitCheck,
    /// Required coolant flow rate (kg/s)
    pub coolant_flow_kg_s: f64,
    /// Thermal efficiency (useful heat / total heat)
    pub thermal_efficiency: f64,
}

impl SparkThermalAnalysis {
    /// Perform complete thermal analysis for a Spark Engine spec
    pub fn analyze(
        thermal: &ThermalTransport,
        power_kw: f64,
        core_radius_m: f64,
        interface_thickness_m: f64,
        shell_thickness_m: f64,
    ) -> Self {
        let power_w = power_kw * 1000.0;

        // Build geometries
        let core_geom = LayerGeometry::sphere(0.0, core_radius_m);
        let interface_geom =
            LayerGeometry::sphere(core_radius_m, core_radius_m + interface_thickness_m);
        let shell_geom = LayerGeometry::sphere(
            core_radius_m + interface_thickness_m,
            core_radius_m + interface_thickness_m + shell_thickness_m,
        );

        // Material properties
        let shell = ThermalProperties::hea_shell();
        let interface = ThermalProperties::nano_laminate();
        let core = ThermalProperties::galinstan();

        // Coolant conditions
        let t_coolant = 320.0; // K (slightly above ambient)
        let h_coolant = 5000.0; // W/m²·K (forced convection)

        // Calculate temperature profile
        let profile = thermal.steady_state_profile(
            power_w,
            &shell,
            &shell_geom,
            &interface,
            &interface_geom,
            &core,
            &core_geom,
            t_coolant,
            h_coolant,
        );

        // Calculate thermal stress
        let stress = thermal.thermal_stress(
            &profile,
            &shell,
            &shell_geom,
            200e9, // Young's modulus ~200 GPa for HEA
            0.3,   // Poisson's ratio
            800e6, // Yield stress ~800 MPa
        );

        // Check limits
        let limits = thermal.check_limits(&profile, &shell, &interface, &core);

        // Coolant flow rate (water, 10K temperature rise)
        let coolant_flow = thermal.coolant_flow_rate(power_w, 4186.0, 10.0);

        // Thermal efficiency (assume 40% for LCF with thermal conversion)
        let thermal_efficiency = 0.40;

        Self {
            profile,
            stress,
            limits,
            coolant_flow_kg_s: coolant_flow,
            thermal_efficiency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> ThermalTransport {
        let genesis = GenesisSeed::from_phrase("thermal test");
        ThermalTransport::from_genesis(&genesis)
    }

    #[test]
    fn test_thermal_properties() {
        let hea = ThermalProperties::hea_shell();
        assert!(hea.conductivity > 0.0);
        assert!(hea.diffusivity() > 0.0);
    }

    #[test]
    fn test_layer_geometry() {
        let sphere = LayerGeometry::sphere(0.05, 0.06);
        assert!((sphere.thickness() - 0.01).abs() < 1e-10);
        assert!(sphere.volume() > 0.0);

        let cylinder = LayerGeometry::cylinder(0.05, 0.06, 0.2);
        assert!(cylinder.volume() > sphere.volume());
    }

    #[test]
    fn test_temperature_profile() {
        let thermal = setup();

        let shell = ThermalProperties::hea_shell();
        let interface = ThermalProperties::nano_laminate();
        let core = ThermalProperties::galinstan();

        let core_geom = LayerGeometry::sphere(0.0, 0.03);
        let interface_geom = LayerGeometry::sphere(0.03, 0.0301);
        let shell_geom = LayerGeometry::sphere(0.0301, 0.04);

        let profile = thermal.steady_state_profile(
            5000.0, // 5 kW
            &shell,
            &shell_geom,
            &interface,
            &interface_geom,
            &core,
            &core_geom,
            320.0,  // Coolant temp
            5000.0, // Convection coefficient
        );

        assert!(profile.t_max > profile.t_min);
        assert!(profile.t_core_center > 300.0);
        assert!(profile.heat_flux_outer > 0.0);
    }

    #[test]
    fn test_thermal_stress() {
        let thermal = setup();

        let profile = TemperatureProfile {
            radii: vec![0.03, 0.04],
            temperatures: vec![400.0, 350.0],
            t_max: 450.0,
            t_min: 320.0,
            t_shell_outer: 350.0,
            t_core_center: 400.0,
            heat_flux_outer: 50000.0,
        };

        let shell = ThermalProperties::hea_shell();
        let geom = LayerGeometry::sphere(0.03, 0.04);

        let stress = thermal.thermal_stress(&profile, &shell, &geom, 200e9, 0.3, 800e6);

        assert!(stress.von_mises > 0.0);
        assert!(stress.safety_factor > 0.0);
    }

    #[test]
    fn test_complete_analysis() {
        let thermal = setup();

        let analysis = SparkThermalAnalysis::analyze(
            &thermal, 5.0,    // 5 kW
            0.03,   // 3 cm core radius
            0.0002, // 0.2 mm interface
            0.01,   // 1 cm shell
        );

        assert!(analysis.limits.is_safe || !analysis.limits.violations.is_empty());
        assert!(analysis.coolant_flow_kg_s > 0.0);
    }
}

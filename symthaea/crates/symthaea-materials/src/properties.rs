// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Material property definitions with 5 preset materials.

use serde::{Deserialize, Serialize};

/// Classification of material type for constraint-based search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialCategory {
    /// Metallic alloys (steel, aluminum, titanium).
    Metal,
    /// Ceramic and cite-based materials (concrete).
    Ceramic,
    /// Organic polymer materials.
    Polymer,
    /// Fiber-reinforced composite materials (carbon fiber).
    Composite,
    /// Engineered metamaterial (layered structures with emergent EM properties).
    Metamaterial,
}

/// A material's key engineering properties (8 numeric dimensions).
///
/// Each property maps to one dimension of the 16,384D HDC encoding.
/// Factory methods provide 5 preset reference materials spanning
/// metals, ceramics, and composites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperty {
    /// Human-readable material name.
    pub name: String,
    /// Material classification.
    pub category: MaterialCategory,
    /// Density in kg/m³.
    pub density_kg_m3: f32,
    /// Young's modulus (stiffness) in GPa.
    pub youngs_modulus_gpa: f32,
    /// Yield strength in MPa.
    pub yield_strength_mpa: f32,
    /// Thermal conductivity in W/(m·K).
    pub thermal_conductivity_w_mk: f32,
    /// Specific heat capacity in J/(kg·K).
    pub specific_heat_j_kgk: f32,
    /// Melting point in °C.
    pub melting_point_c: f32,
    /// Corrosion resistance (0.0 = poor, 1.0 = excellent).
    pub corrosion_resistance: f32,
    /// Fatigue limit (endurance limit) in MPa.
    pub fatigue_limit_mpa: f32,
}

impl MaterialProperty {
    /// ASTM A36 structural steel — common low-carbon structural steel.
    pub fn steel_a36() -> Self {
        Self {
            name: "Steel A36".into(),
            category: MaterialCategory::Metal,
            density_kg_m3: 7850.0,
            youngs_modulus_gpa: 200.0,
            yield_strength_mpa: 250.0,
            thermal_conductivity_w_mk: 50.0,
            specific_heat_j_kgk: 486.0,
            melting_point_c: 1425.0,
            corrosion_resistance: 0.3,
            fatigue_limit_mpa: 160.0,
        }
    }

    /// Aluminum 6061-T6 — precipitation-hardened Al-Mg-Si alloy.
    pub fn aluminum_6061() -> Self {
        Self {
            name: "Aluminum 6061".into(),
            category: MaterialCategory::Metal,
            density_kg_m3: 2700.0,
            youngs_modulus_gpa: 69.0,
            yield_strength_mpa: 276.0,
            thermal_conductivity_w_mk: 167.0,
            specific_heat_j_kgk: 896.0,
            melting_point_c: 582.0,
            corrosion_resistance: 0.7,
            fatigue_limit_mpa: 96.0,
        }
    }

    /// Titanium Ti-6Al-4V — alpha-beta titanium alloy (aerospace grade).
    pub fn titanium_ti6al4v() -> Self {
        Self {
            name: "Titanium Ti6Al4V".into(),
            category: MaterialCategory::Metal,
            density_kg_m3: 4430.0,
            youngs_modulus_gpa: 114.0,
            yield_strength_mpa: 880.0,
            thermal_conductivity_w_mk: 6.7,
            specific_heat_j_kgk: 526.0,
            melting_point_c: 1660.0,
            corrosion_resistance: 0.9,
            fatigue_limit_mpa: 510.0,
        }
    }

    /// Concrete C30/37 — standard structural concrete (30 MPa compressive).
    pub fn concrete_c30() -> Self {
        Self {
            name: "Concrete C30".into(),
            category: MaterialCategory::Ceramic,
            density_kg_m3: 2400.0,
            youngs_modulus_gpa: 30.0,
            yield_strength_mpa: 30.0,
            thermal_conductivity_w_mk: 1.7,
            specific_heat_j_kgk: 880.0,
            melting_point_c: 1150.0,
            corrosion_resistance: 0.5,
            fatigue_limit_mpa: 10.0,
        }
    }

    /// Toray T300 carbon fiber composite — high-strength PAN-based CFRP.
    pub fn carbon_fiber_t300() -> Self {
        Self {
            name: "Carbon Fiber T300".into(),
            category: MaterialCategory::Composite,
            density_kg_m3: 1760.0,
            youngs_modulus_gpa: 230.0,
            yield_strength_mpa: 3530.0,
            thermal_conductivity_w_mk: 8.0,
            specific_heat_j_kgk: 710.0,
            melting_point_c: 3650.0,
            corrosion_resistance: 0.95,
            fatigue_limit_mpa: 1500.0,
        }
    }

    /// Bismuth-Magnesium(Zinc) layered metamaterial — "Art's Parts" (Roswell debris claim).
    ///
    /// Alternating layers of Bi (9,780 kg/m³) and Mg-Zn alloy (1,830 kg/m³).
    /// Properties are bulk averages for a ~50/50 layered structure.
    ///
    /// Claimed to be a "terahertz waveguide" for anti-gravity propulsion.
    /// ORNL analysis (2022): terrestrial isotopic signatures, impure Bi layers
    /// with multiple Bi layers (disrupts waveguide function).
    ///
    /// References:
    /// - AARO/ORNL analysis (2022): terrestrial manufacturing byproduct
    /// - Bismuth properties: CRC Handbook of Chemistry and Physics
    /// - Mg-3Zn alloy properties: ASM International
    pub fn bismuth_magnesium_metamaterial() -> Self {
        Self {
            name: "Bi-Mg(Zn) Metamaterial (Art's Parts)".into(),
            category: MaterialCategory::Metamaterial,
            // Bulk average: ~50% Bi (9780) + 50% Mg-Zn (1830)
            density_kg_m3: 5805.0,
            // Bi: 32 GPa, Mg: 45 GPa → layered composite ~35 GPa
            youngs_modulus_gpa: 35.0,
            // Bi is brittle (~4 MPa), Mg-Zn ~130 MPa → limited by weakest layer
            yield_strength_mpa: 20.0,
            // Bi: 7.97 W/(m·K), Mg: 156 W/(m·K) → cross-layer avg ~15
            thermal_conductivity_w_mk: 15.0,
            // Bi: 122, Mg: 1020 → weighted average
            specific_heat_j_kgk: 400.0,
            // Bi melts at 271°C (very low) — limiting factor
            melting_point_c: 271.0,
            // Bi oxidizes slowly; Mg corrodes readily
            corrosion_resistance: 0.3,
            // Bi is extremely brittle → near-zero fatigue life
            fatigue_limit_mpa: 5.0,
        }
    }

    /// All preset reference materials (6 including metamaterial).
    pub fn presets() -> Vec<Self> {
        vec![
            Self::steel_a36(),
            Self::aluminum_6061(),
            Self::titanium_ti6al4v(),
            Self::concrete_c30(),
            Self::carbon_fiber_t300(),
            Self::bismuth_magnesium_metamaterial(),
        ]
    }

    /// Normalize all 8 numeric dimensions to [0, 1] using typical engineering ranges.
    pub fn normalized_values(&self) -> [f32; 8] {
        [
            self.density_kg_m3 / 10000.0,
            self.youngs_modulus_gpa / 400.0,
            self.yield_strength_mpa / 4000.0,
            self.thermal_conductivity_w_mk / 400.0,
            self.specific_heat_j_kgk / 1000.0,
            self.melting_point_c / 4000.0,
            self.corrosion_resistance,
            self.fatigue_limit_mpa / 2000.0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets_count() {
        assert_eq!(MaterialProperty::presets().len(), 6);
    }
    #[test]
    fn test_steel_category() {
        assert_eq!(
            MaterialProperty::steel_a36().category,
            MaterialCategory::Metal
        );
    }
    #[test]
    fn test_concrete_category() {
        assert_eq!(
            MaterialProperty::concrete_c30().category,
            MaterialCategory::Ceramic
        );
    }
    #[test]
    fn test_carbon_fiber_category() {
        assert_eq!(
            MaterialProperty::carbon_fiber_t300().category,
            MaterialCategory::Composite
        );
    }

    #[test]
    fn test_normalized_bounded() {
        for m in MaterialProperty::presets() {
            for v in m.normalized_values() {
                assert!(v >= 0.0 && v <= 1.0, "Out of range: {} for {}", v, m.name);
            }
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let m = MaterialProperty::steel_a36();
        let json = serde_json::to_string(&m).unwrap();
        let m2: MaterialProperty = serde_json::from_str(&json).unwrap();
        assert_eq!(m.name, m2.name);
        assert_eq!(m.category, m2.category);
    }

    #[test]
    fn test_all_factory_methods_return_correct_category() {
        let expected: Vec<(&str, MaterialCategory)> = vec![
            ("Steel A36", MaterialCategory::Metal),
            ("Aluminum 6061", MaterialCategory::Metal),
            ("Titanium Ti6Al4V", MaterialCategory::Metal),
            ("Concrete C30", MaterialCategory::Ceramic),
            ("Carbon Fiber T300", MaterialCategory::Composite),
            (
                "Bi-Mg(Zn) Metamaterial (Art's Parts)",
                MaterialCategory::Metamaterial,
            ),
        ];
        let presets = MaterialProperty::presets();
        assert_eq!(presets.len(), expected.len());
        for (mat, (name, cat)) in presets.iter().zip(expected.iter()) {
            assert_eq!(&mat.name, name, "Wrong name");
            assert_eq!(mat.category, *cat, "Wrong category for {}", mat.name);
        }
    }

    #[test]
    fn test_all_properties_positive() {
        for m in MaterialProperty::presets() {
            assert!(m.density_kg_m3 > 0.0, "density for {}", m.name);
            assert!(m.youngs_modulus_gpa > 0.0, "modulus for {}", m.name);
            assert!(m.yield_strength_mpa > 0.0, "yield for {}", m.name);
            assert!(m.thermal_conductivity_w_mk > 0.0, "thermal for {}", m.name);
            assert!(m.specific_heat_j_kgk > 0.0, "specific heat for {}", m.name);
            assert!(m.melting_point_c > 0.0, "melting for {}", m.name);
            assert!(
                m.corrosion_resistance >= 0.0 && m.corrosion_resistance <= 1.0,
                "corrosion for {}",
                m.name
            );
            assert!(m.fatigue_limit_mpa > 0.0, "fatigue for {}", m.name);
        }
    }

    #[test]
    fn test_presets_unique_names() {
        let presets = MaterialProperty::presets();
        let names: Vec<&str> = presets.iter().map(|m| m.name.as_str()).collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                assert_ne!(names[i], names[j], "Duplicate name: {}", names[i]);
            }
        }
    }

    #[test]
    fn test_normalized_values_length() {
        let m = MaterialProperty::steel_a36();
        assert_eq!(m.normalized_values().len(), 8);
    }
}

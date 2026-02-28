//! Material property definitions with 5 preset materials.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialCategory {
    Metal,
    Ceramic,
    Polymer,
    Composite,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialProperty {
    pub name: String,
    pub category: MaterialCategory,
    pub density_kg_m3: f32,
    pub youngs_modulus_gpa: f32,
    pub yield_strength_mpa: f32,
    pub thermal_conductivity_w_mk: f32,
    pub specific_heat_j_kgk: f32,
    pub melting_point_c: f32,
    pub corrosion_resistance: f32,
    pub fatigue_limit_mpa: f32,
}

impl MaterialProperty {
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

    pub fn presets() -> Vec<Self> {
        vec![
            Self::steel_a36(),
            Self::aluminum_6061(),
            Self::titanium_ti6al4v(),
            Self::concrete_c30(),
            Self::carbon_fiber_t300(),
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

    #[test] fn test_presets_count() { assert_eq!(MaterialProperty::presets().len(), 5); }
    #[test] fn test_steel_category() { assert_eq!(MaterialProperty::steel_a36().category, MaterialCategory::Metal); }
    #[test] fn test_concrete_category() { assert_eq!(MaterialProperty::concrete_c30().category, MaterialCategory::Ceramic); }
    #[test] fn test_carbon_fiber_category() { assert_eq!(MaterialProperty::carbon_fiber_t300().category, MaterialCategory::Composite); }

    #[test]
    fn test_normalized_bounded() {
        for m in MaterialProperty::presets() {
            for v in m.normalized_values() {
                assert!(v >= 0.0 && v <= 1.5, "Out of range: {} for {}", v, m.name);
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
}

//! # Physics Data Encoders
//!
//! HDC encoders for physical measurements. Maps continuous physical
//! quantities into hyperdimensional space while preserving relationships.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for physics encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsEncoderConfig {
    /// Hypervector dimensionality (default: 16384)
    pub dimensions: usize,
    /// Number of quantization levels for continuous values
    pub quantization_levels: usize,
    /// Whether to encode units explicitly
    pub encode_units: bool,
    /// Whether to encode measurement uncertainty
    pub encode_uncertainty: bool,
    /// Domain-specific basis vectors
    pub domain: PhysicsDomain,
}

impl Default for PhysicsEncoderConfig {
    fn default() -> Self {
        Self {
            dimensions: 16384,
            quantization_levels: 1000,
            encode_units: true,
            encode_uncertainty: true,
            domain: PhysicsDomain::General,
        }
    }
}

/// Physics domain for specialized encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsDomain {
    /// General physics
    General,
    /// Nuclear/fusion physics
    Nuclear,
    /// Plasma physics
    Plasma,
    /// Condensed matter
    CondensedMatter,
    /// Particle physics
    Particle,
    /// Astrophysics
    Astrophysics,
}

/// A physical measurement encoded as a hypervector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedMeasurement {
    /// Hypervector representation (simplified as Vec<f32> for portability)
    pub vector: Vec<f32>,
    /// Original value
    pub value: f64,
    /// Physical unit
    pub unit: String,
    /// Uncertainty (if known)
    pub uncertainty: Option<f64>,
    /// Quantity type (temperature, pressure, etc.)
    pub quantity: PhysicalQuantity,
    /// Timestamp (if time-series)
    pub timestamp: Option<f64>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Physical quantity types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicalQuantity {
    // Fundamental
    Temperature,
    Pressure,
    Density,
    Energy,
    Power,
    Time,
    Length,
    Mass,

    // Electromagnetic
    ElectricField,
    MagneticField,
    Current,
    Voltage,

    // Nuclear/Fusion
    CrossSection,
    ReactionRate,
    NeutronFlux,
    NeutronEnergy,
    QFactor,

    // Plasma
    PlasmaTemperature,
    PlasmaDensity,
    ConfinementTime,
    BetaParameter,

    // General
    Dimensionless,
    Other,
}

impl PhysicalQuantity {
    /// Get SI base unit.
    pub fn si_unit(&self) -> &'static str {
        match self {
            Self::Temperature => "K",
            Self::Pressure => "Pa",
            Self::Density => "kg/m³",
            Self::Energy => "J",
            Self::Power => "W",
            Self::Time => "s",
            Self::Length => "m",
            Self::Mass => "kg",
            Self::ElectricField => "V/m",
            Self::MagneticField => "T",
            Self::Current => "A",
            Self::Voltage => "V",
            Self::CrossSection => "barn",
            Self::ReactionRate => "cm³/s",
            Self::NeutronFlux => "n/cm²/s",
            Self::NeutronEnergy => "MeV",
            Self::QFactor => "dimensionless",
            Self::PlasmaTemperature => "keV",
            Self::PlasmaDensity => "m⁻³",
            Self::ConfinementTime => "s",
            Self::BetaParameter => "dimensionless",
            Self::Dimensionless => "",
            Self::Other => "",
        }
    }

    /// Typical order of magnitude (log10 of SI value).
    pub fn typical_magnitude(&self) -> f64 {
        match self {
            Self::Temperature => 2.5,        // ~300 K
            Self::Pressure => 5.0,           // ~100 kPa
            Self::Density => 3.0,            // ~1000 kg/m³
            Self::Energy => 0.0,             // ~1 J
            Self::CrossSection => -24.0,     // ~1 barn = 10⁻²⁴ cm²
            Self::ReactionRate => -20.0,     // ~10⁻²⁰ cm³/s
            Self::NeutronFlux => 12.0,       // ~10¹² n/cm²/s
            Self::PlasmaTemperature => 1.0,  // ~10 keV
            Self::PlasmaDensity => 20.0,     // ~10²⁰ m⁻³
            _ => 0.0,
        }
    }
}

/// Physics-aware HDC encoder.
pub struct PhysicsEncoder {
    config: PhysicsEncoderConfig,
    /// Basis vectors for quantities (lazy-initialized)
    quantity_bases: HashMap<PhysicalQuantity, Vec<f32>>,
    /// Basis vectors for units
    unit_bases: HashMap<String, Vec<f32>>,
    /// Random seed for reproducibility
    seed: u64,
}

impl PhysicsEncoder {
    /// Create new encoder with config.
    pub fn new(config: PhysicsEncoderConfig) -> Self {
        Self {
            config,
            quantity_bases: HashMap::new(),
            unit_bases: HashMap::new(),
            seed: 42,
        }
    }

    /// Create encoder for specific domain.
    pub fn for_domain(domain: PhysicsDomain) -> Self {
        let mut config = PhysicsEncoderConfig::default();
        config.domain = domain;
        Self::new(config)
    }

    /// Encode a physical measurement.
    pub fn encode(&mut self, value: f64, quantity: PhysicalQuantity, unit: &str) -> EncodedMeasurement {
        self.encode_with_uncertainty(value, quantity, unit, None)
    }

    /// Encode with uncertainty.
    pub fn encode_with_uncertainty(
        &mut self,
        value: f64,
        quantity: PhysicalQuantity,
        unit: &str,
        uncertainty: Option<f64>,
    ) -> EncodedMeasurement {
        let d = self.config.dimensions;

        // Get or create basis for this quantity
        let quantity_basis = self.get_quantity_basis(quantity);

        // Encode value using thermometer encoding
        // log-scale for physical quantities that span many orders of magnitude
        let log_value = if value > 0.0 { value.log10() } else { -100.0 };
        let normalized = (log_value - quantity.typical_magnitude() + 10.0) / 20.0; // ±10 orders of magnitude
        let value_vector = self.thermometer_encode(normalized.clamp(0.0, 1.0));

        // Bind value with quantity type
        let combined = self.bind(&value_vector, &quantity_basis);

        // Optionally bind with unit
        let final_vector = if self.config.encode_units {
            let unit_basis = self.get_unit_basis(unit);
            self.bind(&combined, &unit_basis)
        } else {
            combined
        };

        EncodedMeasurement {
            vector: final_vector,
            value,
            unit: unit.to_string(),
            uncertainty,
            quantity,
            timestamp: None,
            metadata: HashMap::new(),
        }
    }

    /// Encode a time series of measurements.
    pub fn encode_timeseries(
        &mut self,
        values: &[(f64, f64)], // (time, value) pairs
        quantity: PhysicalQuantity,
        unit: &str,
    ) -> Vec<EncodedMeasurement> {
        values
            .iter()
            .map(|(t, v)| {
                let mut encoded = self.encode(*v, quantity, unit);
                encoded.timestamp = Some(*t);
                encoded
            })
            .collect()
    }

    /// Compute similarity between two encoded measurements.
    pub fn similarity(&self, a: &EncodedMeasurement, b: &EncodedMeasurement) -> f64 {
        if a.vector.len() != b.vector.len() {
            return 0.0;
        }

        let dot: f32 = a.vector.iter().zip(b.vector.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            (dot / (norm_a * norm_b)) as f64
        } else {
            0.0
        }
    }

    /// Bundle multiple measurements into prototype.
    pub fn bundle(&self, measurements: &[EncodedMeasurement]) -> Vec<f32> {
        if measurements.is_empty() {
            return vec![0.0; self.config.dimensions];
        }

        let d = self.config.dimensions;
        let mut result = vec![0.0f32; d];

        for m in measurements {
            for (i, v) in m.vector.iter().enumerate() {
                if i < d {
                    result[i] += v;
                }
            }
        }

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }

        result
    }

    // Private helper methods

    fn get_quantity_basis(&mut self, quantity: PhysicalQuantity) -> Vec<f32> {
        if let Some(basis) = self.quantity_bases.get(&quantity) {
            return basis.clone();
        }

        let basis = self.random_vector(quantity as u64);
        self.quantity_bases.insert(quantity, basis.clone());
        basis
    }

    fn get_unit_basis(&mut self, unit: &str) -> Vec<f32> {
        if let Some(basis) = self.unit_bases.get(unit) {
            return basis.clone();
        }

        // Hash unit string to seed
        let seed = unit.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let basis = self.random_vector(seed + 1000);
        self.unit_bases.insert(unit.to_string(), basis.clone());
        basis
    }

    fn random_vector(&self, seed: u64) -> Vec<f32> {
        let d = self.config.dimensions;
        let mut vec = Vec::with_capacity(d);

        // Simple LCG for reproducibility
        let mut state = seed ^ self.seed;
        for _ in 0..d {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to ±1
            let bit = (state >> 63) as i32 * 2 - 1;
            vec.push(bit as f32);
        }

        vec
    }

    fn thermometer_encode(&self, normalized: f64) -> Vec<f32> {
        let d = self.config.dimensions;
        let mut vec = vec![-1.0f32; d];

        // Thermometer encoding: fill with +1 up to the value
        let fill_to = (normalized * d as f64) as usize;
        for i in 0..fill_to.min(d) {
            vec[i] = 1.0;
        }

        vec
    }

    fn bind(&self, a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = PhysicsEncoder::new(PhysicsEncoderConfig::default());
        assert_eq!(encoder.config.dimensions, 16384);
    }

    #[test]
    fn test_encode_temperature() {
        let mut encoder = PhysicsEncoder::for_domain(PhysicsDomain::Nuclear);
        let encoded = encoder.encode(300.0, PhysicalQuantity::Temperature, "K");

        assert_eq!(encoded.value, 300.0);
        assert_eq!(encoded.unit, "K");
        assert_eq!(encoded.vector.len(), 16384);
    }

    #[test]
    fn test_similar_values_similar_vectors() {
        let mut encoder = PhysicsEncoder::new(PhysicsEncoderConfig::default());

        let t1 = encoder.encode(300.0, PhysicalQuantity::Temperature, "K");
        let t2 = encoder.encode(310.0, PhysicalQuantity::Temperature, "K");
        let t3 = encoder.encode(3000.0, PhysicalQuantity::Temperature, "K");

        let sim_close = encoder.similarity(&t1, &t2);
        let sim_far = encoder.similarity(&t1, &t3);

        // Close values should be more similar
        assert!(sim_close > sim_far);
    }

    #[test]
    fn test_bundle() {
        let mut encoder = PhysicsEncoder::new(PhysicsEncoderConfig::default());

        let measurements: Vec<_> = (0..10)
            .map(|i| encoder.encode(300.0 + i as f64, PhysicalQuantity::Temperature, "K"))
            .collect();

        let prototype = encoder.bundle(&measurements);
        assert_eq!(prototype.len(), 16384);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoder for waste material properties into 16,384-dimensional hypervectors.
//!
//! Uses 12 orthogonal basis vectors (one per property dimension) with weighted
//! superposition. Waste items with similar properties produce similar hypervectors,
//! enabling HDC-based classification and contamination detection.

use serde::{Deserialize, Serialize};

/// HDC dimension matching Symthaea's standard.
pub const HDC_DIM: usize = 16_384;

/// Number of basis vectors (one per waste property dimension).
const NUM_BASES: usize = 12;

// ---------------------------------------------------------------------------
// Inline HDC primitives (self-contained, no symthaea-core dependency)
// ---------------------------------------------------------------------------

/// Generate a deterministic basis vector of +-1 values using xorshift64.
fn generate_basis(seed: u64, dim: usize) -> Vec<f32> {
    let mut state = seed;
    (0..dim)
        .map(|_| {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state % 2 == 0 { 1.0 } else { -1.0 }
        })
        .collect()
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// Weighted bundle: sum(weight_i * basis_i), then L2-normalize.
fn encode_weighted(bases: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    let dim = bases[0].len();
    let mut result = vec![0.0f32; dim];
    for (basis, &w) in bases.iter().zip(weights.iter()) {
        for (r, &b) in result.iter_mut().zip(basis.iter()) {
            *r += w * b;
        }
    }
    // L2 normalize
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for r in &mut result {
            *r /= norm;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Physical and chemical properties of a waste item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasteProperties {
    /// Material density in g/cm^3 (normalized 0-1 internally).
    pub density: f32,
    /// Color value (0 = black, 1 = white).
    pub color_value: f32,
    /// Texture entropy (0 = smooth, 1 = rough).
    pub texture_entropy: f32,
    /// Surface damage score (0 = pristine, 1 = destroyed).
    pub surface_damage: f32,
    /// Numeric code for material type (0-1 normalized).
    pub material_type_code: f32,
    /// Moisture level (0 = dry, 1 = saturated).
    pub moisture_level: f32,
    /// Temperature (0 = cold, 1 = hot, normalized).
    pub temperature: f32,
    /// Contamination level (0 = clean, 1 = heavily contaminated).
    pub contamination_level: f32,
    /// Decomposition rate (0 = stable, 1 = rapidly decomposing).
    pub decomposition_rate: f32,
    /// Recyclability index (0 = not recyclable, 1 = fully recyclable).
    pub recyclability_index: f32,
    /// Toxicity score (0 = inert, 1 = highly toxic).
    pub toxicity_score: f32,
    /// Overall condition score (0 = poor, 1 = excellent).
    pub condition_score: f32,
}

impl WasteProperties {
    /// Return the 12 property values as a weight vector.
    pub fn as_weights(&self) -> [f32; NUM_BASES] {
        [
            self.density,
            self.color_value,
            self.texture_entropy,
            self.surface_damage,
            self.material_type_code,
            self.moisture_level,
            self.temperature,
            self.contamination_level,
            self.decomposition_rate,
            self.recyclability_index,
            self.toxicity_score,
            self.condition_score,
        ]
    }
}

/// Classification categories for waste materials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasteCategory {
    /// Organic / compostable waste.
    Organic,
    /// PET plastic (resin code 1).
    PlasticPET,
    /// HDPE plastic (resin code 2).
    PlasticHDPE,
    /// PP plastic (resin code 5).
    PlasticPP,
    /// PS plastic (resin code 6).
    PlasticPS,
    /// Other plastics (resin codes 3, 4, 7).
    PlasticOther,
    /// Aluminum cans and foil.
    Aluminum,
    /// Steel / tin cans.
    Steel,
    /// Clear glass.
    GlassClear,
    /// Colored glass (green, brown, blue).
    GlassColored,
    /// Paper (office, newspaper).
    Paper,
    /// Cardboard and corrugated fiberboard.
    Cardboard,
    /// Textiles (clothing, fabric scraps).
    Textiles,
    /// Wood and lumber.
    Wood,
    /// Rubber (tires, gaskets).
    Rubber,
    /// Electronic waste (PCBs, batteries).
    Electronic,
    /// Hazardous waste (chemicals, medical).
    Hazardous,
    /// Mixed / unsorted waste.
    Mixed,
}

impl WasteCategory {
    /// All category variants in order.
    pub const ALL: [WasteCategory; 18] = [
        Self::Organic,
        Self::PlasticPET,
        Self::PlasticHDPE,
        Self::PlasticPP,
        Self::PlasticPS,
        Self::PlasticOther,
        Self::Aluminum,
        Self::Steel,
        Self::GlassClear,
        Self::GlassColored,
        Self::Paper,
        Self::Cardboard,
        Self::Textiles,
        Self::Wood,
        Self::Rubber,
        Self::Electronic,
        Self::Hazardous,
        Self::Mixed,
    ];

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Organic => "Organic",
            Self::PlasticPET => "PlasticPET",
            Self::PlasticHDPE => "PlasticHDPE",
            Self::PlasticPP => "PlasticPP",
            Self::PlasticPS => "PlasticPS",
            Self::PlasticOther => "PlasticOther",
            Self::Aluminum => "Aluminum",
            Self::Steel => "Steel",
            Self::GlassClear => "GlassClear",
            Self::GlassColored => "GlassColored",
            Self::Paper => "Paper",
            Self::Cardboard => "Cardboard",
            Self::Textiles => "Textiles",
            Self::Wood => "Wood",
            Self::Rubber => "Rubber",
            Self::Electronic => "Electronic",
            Self::Hazardous => "Hazardous",
            Self::Mixed => "Mixed",
        }
    }

    /// Reference waste properties for this category (typical / canonical values).
    pub fn reference_properties(self) -> WasteProperties {
        match self {
            Self::Organic => WasteProperties {
                density: 0.4,
                color_value: 0.3,
                texture_entropy: 0.7,
                surface_damage: 0.5,
                material_type_code: 0.05,
                moisture_level: 0.8,
                temperature: 0.5,
                contamination_level: 0.1,
                decomposition_rate: 0.9,
                recyclability_index: 0.3,
                toxicity_score: 0.05,
                condition_score: 0.3,
            },
            Self::PlasticPET => WasteProperties {
                density: 0.55,
                color_value: 0.9,
                texture_entropy: 0.1,
                surface_damage: 0.1,
                material_type_code: 0.15,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.01,
                recyclability_index: 0.9,
                toxicity_score: 0.1,
                condition_score: 0.8,
            },
            Self::PlasticHDPE => WasteProperties {
                density: 0.5,
                color_value: 0.7,
                texture_entropy: 0.15,
                surface_damage: 0.15,
                material_type_code: 0.2,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.01,
                recyclability_index: 0.85,
                toxicity_score: 0.08,
                condition_score: 0.75,
            },
            Self::PlasticPP => WasteProperties {
                density: 0.45,
                color_value: 0.75,
                texture_entropy: 0.12,
                surface_damage: 0.12,
                material_type_code: 0.25,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.01,
                recyclability_index: 0.7,
                toxicity_score: 0.1,
                condition_score: 0.8,
            },
            Self::PlasticPS => WasteProperties {
                density: 0.3,
                color_value: 0.85,
                texture_entropy: 0.08,
                surface_damage: 0.2,
                material_type_code: 0.3,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.01,
                recyclability_index: 0.4,
                toxicity_score: 0.2,
                condition_score: 0.7,
            },
            Self::PlasticOther => WasteProperties {
                density: 0.5,
                color_value: 0.6,
                texture_entropy: 0.2,
                surface_damage: 0.2,
                material_type_code: 0.35,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.05,
                decomposition_rate: 0.02,
                recyclability_index: 0.3,
                toxicity_score: 0.15,
                condition_score: 0.6,
            },
            Self::Aluminum => WasteProperties {
                density: 0.85,
                color_value: 0.85,
                texture_entropy: 0.2,
                surface_damage: 0.3,
                material_type_code: 0.4,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.005,
                recyclability_index: 0.95,
                toxicity_score: 0.02,
                condition_score: 0.7,
            },
            Self::Steel => WasteProperties {
                density: 0.95,
                color_value: 0.5,
                texture_entropy: 0.25,
                surface_damage: 0.4,
                material_type_code: 0.45,
                moisture_level: 0.05,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.02,
                recyclability_index: 0.9,
                toxicity_score: 0.03,
                condition_score: 0.6,
            },
            Self::GlassClear => WasteProperties {
                density: 0.9,
                color_value: 0.95,
                texture_entropy: 0.05,
                surface_damage: 0.3,
                material_type_code: 0.5,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.001,
                recyclability_index: 0.9,
                toxicity_score: 0.01,
                condition_score: 0.65,
            },
            Self::GlassColored => WasteProperties {
                density: 0.9,
                color_value: 0.4,
                texture_entropy: 0.05,
                surface_damage: 0.35,
                material_type_code: 0.55,
                moisture_level: 0.0,
                temperature: 0.4,
                contamination_level: 0.0,
                decomposition_rate: 0.001,
                recyclability_index: 0.8,
                toxicity_score: 0.01,
                condition_score: 0.6,
            },
            Self::Paper => WasteProperties {
                density: 0.3,
                color_value: 0.85,
                texture_entropy: 0.4,
                surface_damage: 0.3,
                material_type_code: 0.6,
                moisture_level: 0.2,
                temperature: 0.45,
                contamination_level: 0.1,
                decomposition_rate: 0.5,
                recyclability_index: 0.7,
                toxicity_score: 0.02,
                condition_score: 0.5,
            },
            Self::Cardboard => WasteProperties {
                density: 0.25,
                color_value: 0.5,
                texture_entropy: 0.5,
                surface_damage: 0.35,
                material_type_code: 0.65,
                moisture_level: 0.15,
                temperature: 0.45,
                contamination_level: 0.1,
                decomposition_rate: 0.45,
                recyclability_index: 0.75,
                toxicity_score: 0.02,
                condition_score: 0.5,
            },
            Self::Textiles => WasteProperties {
                density: 0.35,
                color_value: 0.5,
                texture_entropy: 0.8,
                surface_damage: 0.4,
                material_type_code: 0.7,
                moisture_level: 0.1,
                temperature: 0.45,
                contamination_level: 0.05,
                decomposition_rate: 0.2,
                recyclability_index: 0.5,
                toxicity_score: 0.05,
                condition_score: 0.5,
            },
            Self::Wood => WasteProperties {
                density: 0.5,
                color_value: 0.45,
                texture_entropy: 0.6,
                surface_damage: 0.3,
                material_type_code: 0.75,
                moisture_level: 0.3,
                temperature: 0.45,
                contamination_level: 0.05,
                decomposition_rate: 0.15,
                recyclability_index: 0.4,
                toxicity_score: 0.03,
                condition_score: 0.55,
            },
            Self::Rubber => WasteProperties {
                density: 0.6,
                color_value: 0.15,
                texture_entropy: 0.5,
                surface_damage: 0.5,
                material_type_code: 0.8,
                moisture_level: 0.0,
                temperature: 0.45,
                contamination_level: 0.1,
                decomposition_rate: 0.02,
                recyclability_index: 0.3,
                toxicity_score: 0.2,
                condition_score: 0.4,
            },
            Self::Electronic => WasteProperties {
                density: 0.7,
                color_value: 0.3,
                texture_entropy: 0.6,
                surface_damage: 0.3,
                material_type_code: 0.85,
                moisture_level: 0.0,
                temperature: 0.45,
                contamination_level: 0.2,
                decomposition_rate: 0.01,
                recyclability_index: 0.6,
                toxicity_score: 0.5,
                condition_score: 0.5,
            },
            Self::Hazardous => WasteProperties {
                density: 0.6,
                color_value: 0.4,
                texture_entropy: 0.3,
                surface_damage: 0.2,
                material_type_code: 0.9,
                moisture_level: 0.1,
                temperature: 0.5,
                contamination_level: 0.8,
                decomposition_rate: 0.05,
                recyclability_index: 0.1,
                toxicity_score: 0.9,
                condition_score: 0.3,
            },
            Self::Mixed => WasteProperties {
                density: 0.5,
                color_value: 0.5,
                texture_entropy: 0.5,
                surface_damage: 0.5,
                material_type_code: 0.95,
                moisture_level: 0.3,
                temperature: 0.45,
                contamination_level: 0.4,
                decomposition_rate: 0.3,
                recyclability_index: 0.2,
                toxicity_score: 0.3,
                condition_score: 0.4,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Seeds for the 12 basis vectors, derived from property names.
const BASIS_SEEDS: [u64; NUM_BASES] = [
    0xDEAD_0001, // density
    0xDEAD_0002, // color_value
    0xDEAD_0003, // texture_entropy
    0xDEAD_0004, // surface_damage
    0xDEAD_0005, // material_type_code
    0xDEAD_0006, // moisture_level
    0xDEAD_0007, // temperature
    0xDEAD_0008, // contamination_level
    0xDEAD_0009, // decomposition_rate
    0xDEAD_000A, // recyclability_index
    0xDEAD_000B, // toxicity_score
    0xDEAD_000C, // condition_score
];

/// Encodes waste material properties into 16,384D hypervectors for classification.
///
/// Uses 12 orthogonal basis vectors (one per property dimension) with weighted
/// superposition. The resulting HV can be compared via cosine similarity.
pub struct WasteHdcEncoder {
    bases: Vec<Vec<f32>>,
}

impl WasteHdcEncoder {
    /// Create a new encoder with deterministic basis vectors.
    pub fn new() -> Self {
        let bases: Vec<Vec<f32>> = BASIS_SEEDS
            .iter()
            .map(|&seed| generate_basis(seed, HDC_DIM))
            .collect();
        Self { bases }
    }

    /// Encode waste properties into a 16,384D hypervector.
    pub fn encode(&self, properties: &WasteProperties) -> Vec<f32> {
        let weights = properties.as_weights();
        encode_weighted(&self.bases, &weights)
    }

    /// Cosine similarity between two waste items in HDC space.
    pub fn similarity(&self, a: &WasteProperties, b: &WasteProperties) -> f32 {
        cosine_similarity(&self.encode(a), &self.encode(b))
    }
}

impl Default for WasteHdcEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Database
// ---------------------------------------------------------------------------

/// Database of reference hypervectors for each waste category.
///
/// Stores one canonical HV per category, built from `WasteCategory::reference_properties()`.
/// Supports ranked classification via cosine similarity.
pub struct WasteDatabase {
    encoder: WasteHdcEncoder,
    entries: Vec<(WasteCategory, Vec<f32>)>,
}

impl WasteDatabase {
    /// Build a new database with reference HVs for all 18 categories.
    pub fn new() -> Self {
        let encoder = WasteHdcEncoder::new();
        let entries: Vec<(WasteCategory, Vec<f32>)> = WasteCategory::ALL
            .iter()
            .map(|&cat| {
                let hv = encoder.encode(&cat.reference_properties());
                (cat, hv)
            })
            .collect();
        Self { encoder, entries }
    }

    /// Classify a waste item by its properties. Returns top-k matches ranked by
    /// cosine similarity (descending).
    pub fn classify(
        &self,
        properties: &WasteProperties,
        top_k: usize,
    ) -> Vec<(WasteCategory, f32)> {
        let hv = self.encoder.encode(properties);
        self.classify_hv(&hv, top_k)
    }

    /// Classify from a pre-encoded hypervector.
    pub fn classify_hv(&self, hv: &[f32], top_k: usize) -> Vec<(WasteCategory, f32)> {
        let mut scores: Vec<(WasteCategory, f32)> = self
            .entries
            .iter()
            .map(|(cat, ref_hv)| (*cat, cosine_similarity(hv, ref_hv)))
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(top_k);
        scores
    }

    /// Get the reference HV for a category.
    pub fn reference_hv(&self, category: &WasteCategory) -> Option<&Vec<f32>> {
        self.entries
            .iter()
            .find(|(cat, _)| cat == category)
            .map(|(_, hv)| hv)
    }

    /// Get the encoder for external use.
    pub fn encoder(&self) -> &WasteHdcEncoder {
        &self.encoder
    }

    /// Number of categories in the database.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for WasteDatabase {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn organic_props() -> WasteProperties {
        WasteCategory::Organic.reference_properties()
    }

    fn pet_props() -> WasteProperties {
        WasteCategory::PlasticPET.reference_properties()
    }

    #[test]
    fn test_self_similarity_high() {
        let encoder = WasteHdcEncoder::new();
        let sim = encoder.similarity(&organic_props(), &organic_props());
        assert!(sim > 0.99, "Self-similarity should be > 0.99, got {}", sim);
    }

    #[test]
    fn test_self_similarity_pet() {
        let encoder = WasteHdcEncoder::new();
        let sim = encoder.similarity(&pet_props(), &pet_props());
        assert!(
            sim > 0.99,
            "PET self-similarity should be > 0.99, got {}",
            sim
        );
    }

    #[test]
    fn test_cross_category_discrimination() {
        let encoder = WasteHdcEncoder::new();
        let self_sim = encoder.similarity(&organic_props(), &organic_props());
        let cross_sim = encoder.similarity(&organic_props(), &pet_props());
        assert!(
            cross_sim < self_sim,
            "Cross-category similarity ({}) should be less than self-similarity ({})",
            cross_sim,
            self_sim
        );
    }

    #[test]
    fn test_classify_organic() {
        let db = WasteDatabase::new();
        let results = db.classify(&organic_props(), 3);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].0,
            WasteCategory::Organic,
            "Top match for organic properties should be Organic, got {:?}",
            results[0].0
        );
    }

    #[test]
    fn test_classify_pet() {
        let db = WasteDatabase::new();
        let results = db.classify(&pet_props(), 3);
        assert!(!results.is_empty());
        assert_eq!(
            results[0].0,
            WasteCategory::PlasticPET,
            "Top match for PET properties should be PlasticPET, got {:?}",
            results[0].0
        );
    }

    #[test]
    fn test_classify_aluminum() {
        let db = WasteDatabase::new();
        let props = WasteCategory::Aluminum.reference_properties();
        let results = db.classify(&props, 3);
        assert_eq!(results[0].0, WasteCategory::Aluminum);
    }

    #[test]
    fn test_classify_paper() {
        let db = WasteDatabase::new();
        let props = WasteCategory::Paper.reference_properties();
        let results = db.classify(&props, 3);
        assert_eq!(results[0].0, WasteCategory::Paper);
    }

    #[test]
    fn test_basis_orthogonality() {
        let encoder = WasteHdcEncoder::new();
        for i in 0..NUM_BASES {
            for j in (i + 1)..NUM_BASES {
                let sim = cosine_similarity(&encoder.bases[i], &encoder.bases[j]).abs();
                assert!(
                    sim < 0.1,
                    "Basis vectors {} and {} should be near-orthogonal, similarity = {}",
                    i,
                    j,
                    sim
                );
            }
        }
    }

    #[test]
    fn test_database_contains_all_categories() {
        let db = WasteDatabase::new();
        assert_eq!(db.len(), 18, "Database should contain all 18 categories");
    }

    #[test]
    fn test_confidence_scores_bounded() {
        let db = WasteDatabase::new();
        let results = db.classify(&organic_props(), 18);
        for (cat, score) in &results {
            assert!(
                *score >= -1.0 && *score <= 1.0,
                "Confidence score for {:?} should be in [-1, 1], got {}",
                cat,
                score
            );
        }
    }

    #[test]
    fn test_top_match_confidence_positive() {
        let db = WasteDatabase::new();
        for &cat in &WasteCategory::ALL {
            let results = db.classify(&cat.reference_properties(), 1);
            assert!(
                results[0].1 > 0.0,
                "Top match confidence for {:?} should be positive, got {}",
                cat,
                results[0].1
            );
        }
    }

    #[test]
    fn test_encode_deterministic() {
        let encoder = WasteHdcEncoder::new();
        let hv1 = encoder.encode(&organic_props());
        let hv2 = encoder.encode(&organic_props());
        assert_eq!(hv1, hv2, "Encoding should be deterministic");
    }

    #[test]
    fn test_encode_dimension() {
        let encoder = WasteHdcEncoder::new();
        let hv = encoder.encode(&organic_props());
        assert_eq!(hv.len(), HDC_DIM, "HV should be {} dimensions", HDC_DIM);
    }

    #[test]
    fn test_encoded_hv_normalized() {
        let encoder = WasteHdcEncoder::new();
        let hv = encoder.encode(&organic_props());
        let norm: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "Encoded HV should be L2-normalized, norm = {}",
            norm
        );
    }

    #[test]
    fn test_similar_plastics_closer_than_organic_vs_metal() {
        let encoder = WasteHdcEncoder::new();
        let pet = pet_props();
        let hdpe = WasteCategory::PlasticHDPE.reference_properties();
        let aluminum = WasteCategory::Aluminum.reference_properties();

        let plastic_sim = encoder.similarity(&pet, &hdpe);
        let pet_aluminum = encoder.similarity(&pet, &aluminum);
        assert!(
            plastic_sim > pet_aluminum,
            "PET-HDPE similarity ({}) should exceed PET-Aluminum similarity ({})",
            plastic_sim,
            pet_aluminum
        );
    }

    #[test]
    fn test_database_not_empty() {
        let db = WasteDatabase::new();
        assert!(!db.is_empty());
    }

    #[test]
    fn test_reference_hv_lookup() {
        let db = WasteDatabase::new();
        let hv = db.reference_hv(&WasteCategory::Organic);
        assert!(hv.is_some());
        assert_eq!(hv.unwrap().len(), HDC_DIM);
    }

    #[test]
    fn test_classify_hv_matches_classify() {
        let db = WasteDatabase::new();
        let props = organic_props();
        let hv = db.encoder().encode(&props);
        let results_props = db.classify(&props, 3);
        let results_hv = db.classify_hv(&hv, 3);
        assert_eq!(results_props[0].0, results_hv[0].0);
    }
}

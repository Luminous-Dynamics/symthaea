// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contamination detection for waste streams using HDC outlier detection.
//!
//! Compares a waste item's hypervector against the expected category reference to
//! detect misclassified or contaminated materials. Identifies likely contaminants
//! by checking similarity to all other categories.

use crate::waste_encoder::{WasteCategory, WasteDatabase};
use serde::{Deserialize, Serialize};

/// Contamination similarity threshold. Below this, the item is flagged.
const CONTAMINATION_THRESHOLD: f32 = 0.7;

/// Result of contamination detection for a single waste item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContaminationResult {
    /// Whether the item is flagged as contaminated (not matching expected category).
    pub is_contaminated: bool,
    /// Contamination score (0 = clean match, 1 = completely different).
    pub contamination_score: f32,
    /// Names of detected contaminant categories (if any).
    pub detected_contaminants: Vec<String>,
    /// Confidence in the contamination assessment (0-1).
    pub confidence: f32,
}

/// Detects contaminants in waste streams using HDC outlier detection.
///
/// Given a waste item's hypervector and its expected category, checks whether the
/// item actually belongs to that category. If the cosine similarity is below the
/// threshold, the item is flagged and the most likely actual categories are reported.
pub struct ContaminationDetector;

impl ContaminationDetector {
    /// Create a new detector.
    pub fn new() -> Self {
        Self
    }

    /// Detect contamination by comparing a waste HV against its expected category.
    ///
    /// # Arguments
    /// * `waste_hv` - The encoded hypervector of the waste item.
    /// * `expected_category` - The category this item is supposed to be.
    /// * `database` - Reference database for all categories.
    pub fn detect(
        &self,
        waste_hv: &[f32],
        expected_category: &WasteCategory,
        database: &WasteDatabase,
    ) -> ContaminationResult {
        // Get similarity to expected category
        let expected_hv = match database.reference_hv(expected_category) {
            Some(hv) => hv,
            None => {
                return ContaminationResult {
                    is_contaminated: false,
                    contamination_score: 0.0,
                    detected_contaminants: vec![],
                    confidence: 0.0,
                };
            }
        };

        let similarity_to_expected = cosine_similarity(waste_hv, expected_hv);
        let contamination_score = (1.0 - similarity_to_expected).clamp(0.0, 1.0);
        let is_contaminated = similarity_to_expected < CONTAMINATION_THRESHOLD;

        // Find likely contaminant categories
        let mut detected_contaminants = Vec::new();
        if is_contaminated {
            let matches = database.classify_hv(waste_hv, 3);
            for (cat, sim) in &matches {
                if cat != expected_category && *sim > similarity_to_expected {
                    detected_contaminants.push(cat.name().to_string());
                }
            }
        }

        // Confidence: higher when the gap between expected similarity and threshold is large
        let confidence = if is_contaminated {
            // Clear contamination: low similarity → high confidence
            ((CONTAMINATION_THRESHOLD - similarity_to_expected) / CONTAMINATION_THRESHOLD)
                .clamp(0.0, 1.0)
        } else {
            // Clean item: high similarity → high confidence
            ((similarity_to_expected - CONTAMINATION_THRESHOLD) / (1.0 - CONTAMINATION_THRESHOLD))
                .clamp(0.0, 1.0)
        };

        ContaminationResult {
            is_contaminated,
            contamination_score,
            detected_contaminants,
            confidence,
        }
    }
}

impl Default for ContaminationDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Cosine similarity (duplicated from waste_encoder to keep modules independent).
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::waste_encoder::WasteHdcEncoder;

    #[test]
    fn test_clean_organic_not_contaminated() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();
        let organic_props = WasteCategory::Organic.reference_properties();
        let hv = encoder.encode(&organic_props);

        let detector = ContaminationDetector::new();
        let result = detector.detect(&hv, &WasteCategory::Organic, &db);
        assert!(
            !result.is_contaminated,
            "Clean organic waste should not be flagged as contaminated"
        );
    }

    #[test]
    fn test_plastic_in_organic_stream_contaminated() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();
        // Encode plastic but claim it's organic
        let plastic_props = WasteCategory::PlasticPET.reference_properties();
        let hv = encoder.encode(&plastic_props);

        let detector = ContaminationDetector::new();
        let result = detector.detect(&hv, &WasteCategory::Organic, &db);
        assert!(
            result.is_contaminated,
            "Plastic in organic stream should be flagged as contaminated"
        );
        // Should detect plastic as the contaminant
        let has_plastic = result
            .detected_contaminants
            .iter()
            .any(|c| c.contains("Plastic"));
        assert!(
            has_plastic,
            "Should detect plastic contaminant, got: {:?}",
            result.detected_contaminants
        );
    }

    #[test]
    fn test_contamination_score_bounded() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();
        let props = WasteCategory::Steel.reference_properties();
        let hv = encoder.encode(&props);

        let detector = ContaminationDetector::new();
        let result = detector.detect(&hv, &WasteCategory::Organic, &db);
        assert!(
            result.contamination_score >= 0.0 && result.contamination_score <= 1.0,
            "Contamination score should be in [0, 1], got {}",
            result.contamination_score
        );
    }

    #[test]
    fn test_confidence_bounded() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();

        let detector = ContaminationDetector::new();
        for &cat in &WasteCategory::ALL {
            let hv = encoder.encode(&cat.reference_properties());
            let result = detector.detect(&hv, &cat, &db);
            assert!(
                result.confidence >= 0.0 && result.confidence <= 1.0,
                "{:?} confidence should be in [0, 1], got {}",
                cat,
                result.confidence
            );
        }
    }

    #[test]
    fn test_high_contamination_high_confidence() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();
        // Very different material in wrong stream
        let hazardous_props = WasteCategory::Hazardous.reference_properties();
        let hv = encoder.encode(&hazardous_props);

        let detector = ContaminationDetector::new();
        let result = detector.detect(&hv, &WasteCategory::GlassClear, &db);
        if result.is_contaminated {
            assert!(
                result.confidence > 0.0,
                "Clear contamination should have positive confidence"
            );
        }
    }

    #[test]
    fn test_clean_match_low_contamination_score() {
        let db = WasteDatabase::new();
        let encoder = WasteHdcEncoder::new();
        let props = WasteCategory::Aluminum.reference_properties();
        let hv = encoder.encode(&props);

        let detector = ContaminationDetector::new();
        let result = detector.detect(&hv, &WasteCategory::Aluminum, &db);
        assert!(
            result.contamination_score < 0.1,
            "Clean match should have low contamination score, got {}",
            result.contamination_score
        );
    }
}

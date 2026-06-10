// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear attribution agent: match unknown samples against reference database.

use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::decay_model::IsotopeDecayModel;
use crate::encoder::IsotopicHdcEncoder;
use crate::isotope::{IsotopicSignature, NuclearSource};

/// Result of attributing an unknown isotopic sample to a reference source.
#[derive(Debug, Clone)]
pub struct AttributionResult {
    /// The best-matching reference source category.
    pub matched_source: NuclearSource,
    /// Name of the best-matching reference signature.
    pub matched_name: String,
    /// Attribution confidence (0.0–1.0), combining similarity and margin.
    pub confidence: f32,
    /// All reference similarities, sorted descending.
    pub all_similarities: Vec<(String, f32)>,
}

/// Nuclear attribution agent that matches unknown isotopic samples against
/// a reference database using HDC cosine similarity.
///
/// Pre-encodes all reference signatures at construction time so that
/// attribution calls are O(n) in the number of references, not O(n × d).
pub struct NuclearAttributionAgent {
    references: Vec<IsotopicSignature>,
    /// Pre-computed reference HVs — avoids re-encoding on every attribution call.
    reference_hvs: Vec<ContinuousHV>,
    encoder: IsotopicHdcEncoder,
    decay_model: IsotopeDecayModel,
}

impl NuclearAttributionAgent {
    /// Create an agent with the 5 built-in reference signatures.
    pub fn new() -> Self {
        Self::with_references(IsotopicSignature::references())
    }

    /// # Panics
    ///
    /// Panics if `refs` is empty — at least one reference signature is required.
    pub fn with_references(refs: Vec<IsotopicSignature>) -> Self {
        assert!(
            !refs.is_empty(),
            "NuclearAttributionAgent requires at least one reference signature"
        );
        let encoder = IsotopicHdcEncoder::new();
        let reference_hvs = refs.iter().map(|r| encoder.encode(r)).collect();
        Self {
            references: refs,
            reference_hvs,
            encoder,
            decay_model: IsotopeDecayModel::new(),
        }
    }

    /// Attribute an unknown sample by HDC similarity to references.
    pub fn attribute(&self, unknown: &IsotopicSignature) -> AttributionResult {
        let unknown_hv = self.encoder.encode(unknown);
        let mut similarities: Vec<(String, NuclearSource, f32)> = self
            .references
            .iter()
            .zip(self.reference_hvs.iter())
            .map(|(r, ref_hv)| (r.name.clone(), r.source, unknown_hv.similarity(ref_hv)))
            .collect();
        similarities.sort_by(|a, b| {
            match (a.2.is_nan(), b.2.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Greater, // NaN sorts last
                (false, true) => std::cmp::Ordering::Less,
                (false, false) => b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal),
            }
        });

        let best = &similarities[0];
        let second = if similarities.len() > 1 {
            similarities[1].2
        } else {
            0.0
        };
        let margin = best.2 - second;
        let confidence = (best.2 * 0.7 + margin * 0.3).clamp(0.0, 1.0);

        AttributionResult {
            matched_source: best.1,
            matched_name: best.0.clone(),
            confidence,
            all_similarities: similarities
                .iter()
                .map(|(n, _, s)| (n.clone(), *s))
                .collect(),
        }
    }

    /// Attribute with age estimation from decay model.
    pub fn attribute_with_age(
        &self,
        unknown: &IsotopicSignature,
    ) -> (AttributionResult, crate::decay_model::AgeEstimate) {
        let attr = self.attribute(unknown);
        let age = self.decay_model.estimate_age(unknown);
        (attr, age)
    }
}

impl Default for NuclearAttributionAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_attribution() {
        let agent = NuclearAttributionAgent::new();
        let r = agent.attribute(&IsotopicSignature::highly_enriched_uranium());
        assert_eq!(r.matched_source, NuclearSource::HighlyEnrichedUranium);
        assert!(r.confidence > 0.5);
    }

    #[test]
    fn test_perturbed_heu_attribution() {
        let agent = NuclearAttributionAgent::new();
        let p = IsotopicSignature::highly_enriched_uranium().perturbed(0.05, 42);
        let r = agent.attribute(&p);
        assert_eq!(r.matched_source, NuclearSource::HighlyEnrichedUranium);
    }

    #[test]
    fn test_attribution_result_has_all_similarities() {
        let agent = NuclearAttributionAgent::new();
        let r = agent.attribute(&IsotopicSignature::spent_fuel());
        assert_eq!(r.all_similarities.len(), 5);
    }

    #[test]
    fn test_attribution_sorted_descending() {
        let agent = NuclearAttributionAgent::new();
        let r = agent.attribute(&IsotopicSignature::spent_fuel());
        for i in 1..r.all_similarities.len() {
            assert!(r.all_similarities[i - 1].1 >= r.all_similarities[i].1);
        }
    }

    #[test]
    fn test_attribute_with_age() {
        let agent = NuclearAttributionAgent::new();
        let (attr, age) = agent.attribute_with_age(&IsotopicSignature::spent_fuel());
        assert_eq!(attr.matched_source, NuclearSource::SpentFuel);
        assert!(age.estimated_age_seconds >= 0.0);
    }

    #[test]
    fn test_all_references_self_match() {
        let agent = NuclearAttributionAgent::new();
        for sig in &IsotopicSignature::references() {
            let r = agent.attribute(sig);
            assert_eq!(
                r.matched_source, sig.source,
                "Self-attribution failed for {}",
                sig.name
            );
        }
    }

    // ── Track B: failure-path tests ──────────────────────────────────────

    #[test]
    #[should_panic(expected = "at least one reference signature")]
    fn test_empty_references_panics() {
        NuclearAttributionAgent::with_references(vec![]);
    }

    #[test]
    fn test_single_reference_attribution() {
        let agent = NuclearAttributionAgent::with_references(vec![
            IsotopicSignature::highly_enriched_uranium(),
        ]);
        let r = agent.attribute(&IsotopicSignature::highly_enriched_uranium());
        assert_eq!(r.matched_source, NuclearSource::HighlyEnrichedUranium);
        assert_eq!(r.all_similarities.len(), 1);
    }

    #[test]
    fn test_attribution_confidence_bounded() {
        let agent = NuclearAttributionAgent::new();
        for sig in &IsotopicSignature::references() {
            let r = agent.attribute(sig);
            assert!(
                r.confidence >= 0.0 && r.confidence <= 1.0,
                "Confidence out of [0,1] for {}: {}",
                sig.name,
                r.confidence
            );
        }
    }

    // ── Track C: attribute_with_age edge cases ────────────────────────────

    #[test]
    fn test_attribute_with_age_all_references() {
        let agent = NuclearAttributionAgent::new();
        for sig in &IsotopicSignature::references() {
            let (attr, age) = agent.attribute_with_age(sig);
            assert_eq!(
                attr.matched_source, sig.source,
                "attribute_with_age source mismatch for {}",
                sig.name
            );
            assert!(
                age.estimated_age_seconds.is_finite(),
                "NaN age for {}",
                sig.name
            );
            assert!(
                age.confidence >= 0.0 && age.confidence <= 1.0,
                "confidence out of [0,1] for {}",
                sig.name
            );
        }
    }

    #[test]
    fn test_attribute_with_age_zero_ratios() {
        let agent = NuclearAttributionAgent::new();
        let mut sig = IsotopicSignature::natural_uranium();
        sig.pu241_pu239 = 0.0;
        sig.cs137_activity = 0.0;
        sig.sr90_activity = 0.0;
        let (attr, age) = agent.attribute_with_age(&sig);
        assert_eq!(attr.matched_source, NuclearSource::NaturalUranium);
        assert_eq!(
            age.confidence, 0.0,
            "no usable decay ratios → zero confidence"
        );
    }
}

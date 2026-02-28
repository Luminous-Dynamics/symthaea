//! Nuclear attribution agent: match unknown samples against reference database.

use crate::decay_model::IsotopeDecayModel;
use crate::encoder::IsotopicHdcEncoder;
use crate::isotope::{IsotopicSignature, NuclearSource};

#[derive(Debug, Clone)]
pub struct AttributionResult {
    pub matched_source: NuclearSource,
    pub matched_name: String,
    pub confidence: f32,
    pub all_similarities: Vec<(String, f32)>,
}

pub struct NuclearAttributionAgent {
    references: Vec<IsotopicSignature>,
    encoder: IsotopicHdcEncoder,
    decay_model: IsotopeDecayModel,
}

impl NuclearAttributionAgent {
    pub fn new() -> Self {
        Self {
            references: IsotopicSignature::references(),
            encoder: IsotopicHdcEncoder::new(),
            decay_model: IsotopeDecayModel::new(),
        }
    }

    pub fn with_references(refs: Vec<IsotopicSignature>) -> Self {
        Self {
            references: refs,
            encoder: IsotopicHdcEncoder::new(),
            decay_model: IsotopeDecayModel::new(),
        }
    }

    /// Attribute an unknown sample by HDC similarity to references.
    pub fn attribute(&self, unknown: &IsotopicSignature) -> AttributionResult {
        let unknown_hv = self.encoder.encode(unknown);
        let mut similarities: Vec<(String, NuclearSource, f32)> = self.references.iter()
            .map(|r| {
                let ref_hv = self.encoder.encode(r);
                (r.name.clone(), r.source, unknown_hv.similarity(&ref_hv))
            })
            .collect();
        similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let best = &similarities[0];
        let second = if similarities.len() > 1 { similarities[1].2 } else { 0.0 };
        let margin = best.2 - second;
        let confidence = (best.2 * 0.7 + margin * 0.3).clamp(0.0, 1.0);

        AttributionResult {
            matched_source: best.1,
            matched_name: best.0.clone(),
            confidence,
            all_similarities: similarities.iter().map(|(n, _, s)| (n.clone(), *s)).collect(),
        }
    }

    /// Attribute with age estimation from decay model.
    pub fn attribute_with_age(&self, unknown: &IsotopicSignature) -> (AttributionResult, crate::decay_model::AgeEstimate) {
        let attr = self.attribute(unknown);
        let age = self.decay_model.estimate_age(unknown);
        (attr, age)
    }
}

impl Default for NuclearAttributionAgent { fn default() -> Self { Self::new() } }

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
}

//! Therapeutic Dream Bridge — counterfactual therapeutic reasoning.
//!
//! Implements `DreamableAction` for therapeutic interventions, enabling
//! the dream engine to generate counterfactual therapeutic scenarios:
//! "What if the client used cognitive reappraisal instead of avoidance?"
//!
//! Feature gate: `therapeutic`

use serde::{Deserialize, Serialize};
use symthaea_clinical::therapeutic_modalities::{TherapeuticIntervention, TherapeuticModality};
use symthaea_clinical::rdoc::RDocDomain;
use symthaea_dream::DreamableAction;

/// A dreamable wrapper around a therapeutic intervention.
///
/// Encodes the intervention as a float vector for the dream engine's
/// counterfactual reasoning, while preserving the clinical semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamableTherapeuticAction {
    /// The intervention modality
    pub modality: TherapeuticModality,
    /// Name of the technique
    pub technique: String,
    /// Target RDoC domains (encoded as indices)
    pub target_domains: Vec<u8>,
    /// Intervention intensity (0.0-1.0)
    pub intensity: f32,
    /// Evidence level (0.0-1.0)
    pub evidence_level: f32,
    /// Minimum alliance required
    pub min_alliance: f32,
}

impl DreamableTherapeuticAction {
    /// Create from a `TherapeuticIntervention`.
    pub fn from_intervention(intervention: &TherapeuticIntervention, intensity: f32) -> Self {
        let target_domains: Vec<u8> = intervention
            .target_domains
            .iter()
            .map(|d| match d {
                RDocDomain::NegativeValence => 0,
                RDocDomain::PositiveValence => 1,
                RDocDomain::CognitiveSystems => 2,
                RDocDomain::SocialProcesses => 3,
                RDocDomain::ArousalRegulatory => 4,
                RDocDomain::Sensorimotor => 5,
            })
            .collect();

        Self {
            modality: intervention.modality,
            technique: intervention.technique.clone(),
            target_domains,
            intensity: intensity.clamp(0.0, 1.0),
            evidence_level: intervention.evidence_level,
            min_alliance: intervention.min_alliance,
        }
    }

    /// Generate an alternative intervention (different modality, same targets).
    fn alternative_modality(&self, seed: u64) -> TherapeuticModality {
        let modalities = [
            TherapeuticModality::Cbt,
            TherapeuticModality::Act,
            TherapeuticModality::Dbt,
            TherapeuticModality::Narrative,
            TherapeuticModality::Somatic,
            TherapeuticModality::Psychodynamic,
            TherapeuticModality::Motivational,
            TherapeuticModality::Emdr,
            TherapeuticModality::Ifs,
        ];
        let idx = (seed as usize) % modalities.len();
        let alt = modalities[idx];
        if alt == self.modality {
            modalities[(idx + 1) % modalities.len()]
        } else {
            alt
        }
    }
}

impl DreamableAction for DreamableTherapeuticAction {
    fn perturb(&self, seed: u64) -> Self {
        let alt_modality = self.alternative_modality(seed);

        // Perturb intensity slightly
        let hash = blake3::hash(&seed.to_le_bytes());
        let noise_byte = hash.as_bytes()[0] as f32 / 255.0;
        let perturbed_intensity = (self.intensity + (noise_byte - 0.5) * 0.3).clamp(0.0, 1.0);

        Self {
            modality: alt_modality,
            technique: format!("{} (counterfactual)", self.technique),
            target_domains: self.target_domains.clone(),
            intensity: perturbed_intensity,
            evidence_level: self.evidence_level,
            min_alliance: self.min_alliance,
        }
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        // Predict therapeutic outcome as delta on RDoC dimensions.
        // State is assumed to be [neg_valence, pos_valence, cognitive, social, arousal, sensorimotor].
        let mut outcome = state.to_vec();
        if outcome.len() < 6 {
            outcome.resize(6, 0.0);
        }

        let effect = self.intensity * self.evidence_level;

        for &domain_idx in &self.target_domains {
            let idx = domain_idx as usize;
            if idx < outcome.len() {
                match idx {
                    0 => outcome[idx] -= effect * 0.3, // Reduce negative valence
                    1 => outcome[idx] += effect * 0.2, // Increase positive valence
                    2 => outcome[idx] += effect * 0.15, // Improve cognitive function
                    3 => outcome[idx] += effect * 0.1, // Improve social processing
                    4 => outcome[idx] -= effect * 0.2, // Reduce arousal dysregulation
                    5 => outcome[idx] += effect * 0.05, // Minor sensorimotor improvement
                    _ => {}
                }
                outcome[idx] = outcome[idx].clamp(0.0, 1.0);
            }
        }

        outcome
    }

    fn magnitude(&self) -> f32 {
        // Intervention magnitude based on intensity and number of targeted domains
        self.intensity * (self.target_domains.len() as f32 / 6.0).clamp(0.1, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_action() -> DreamableTherapeuticAction {
        DreamableTherapeuticAction {
            modality: TherapeuticModality::Cbt,
            technique: "cognitive_reappraisal".to_string(),
            target_domains: vec![0, 2], // NegativeValence, CognitiveSystems
            intensity: 0.7,
            evidence_level: 0.85,
            min_alliance: 0.4,
        }
    }

    #[test]
    fn test_perturb_changes_modality() {
        let action = sample_action();
        let perturbed = action.perturb(42);
        assert_ne!(perturbed.modality, action.modality);
        assert!(perturbed.technique.contains("counterfactual"));
    }

    #[test]
    fn test_perturb_deterministic() {
        let action = sample_action();
        let p1 = action.perturb(42);
        let p2 = action.perturb(42);
        assert_eq!(p1.modality, p2.modality);
        assert_eq!(p1.intensity, p2.intensity);
    }

    #[test]
    fn test_predict_outcome_reduces_negative_valence() {
        let action = sample_action();
        let state = vec![0.8, 0.3, 0.5, 0.5, 0.6, 0.5];
        let outcome = action.predict_outcome(&state);
        assert!(outcome[0] < state[0], "negative valence should decrease");
        assert!(outcome[2] > state[2], "cognitive should improve");
    }

    #[test]
    fn test_predict_outcome_clamps() {
        let action = DreamableTherapeuticAction {
            modality: TherapeuticModality::Dbt,
            technique: "distress_tolerance".to_string(),
            target_domains: vec![0, 4],
            intensity: 1.0,
            evidence_level: 1.0,
            min_alliance: 0.2,
        };
        let state = vec![0.05, 0.95, 0.5, 0.5, 0.05, 0.5];
        let outcome = action.predict_outcome(&state);
        for &v in &outcome {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_magnitude_scales_with_domains() {
        let single = DreamableTherapeuticAction {
            modality: TherapeuticModality::Act,
            technique: "defusion".to_string(),
            target_domains: vec![0],
            intensity: 0.5,
            evidence_level: 0.8,
            min_alliance: 0.3,
        };
        let multi = DreamableTherapeuticAction {
            target_domains: vec![0, 1, 2, 3],
            ..single.clone()
        };
        assert!(multi.magnitude() > single.magnitude());
    }

    #[test]
    fn test_from_intervention() {
        use symthaea_clinical::therapeutic_modalities::InterventionLibrary;
        let lib = InterventionLibrary::bootstrap();
        let interventions = lib.by_modality(TherapeuticModality::Cbt);
        assert!(!interventions.is_empty());
        let dreamable = DreamableTherapeuticAction::from_intervention(&interventions[0], 0.6);
        assert_eq!(dreamable.modality, TherapeuticModality::Cbt);
        assert_eq!(dreamable.intensity, 0.6);
    }

    #[test]
    fn test_predict_outcome_short_state() {
        let action = sample_action();
        let state = vec![0.5, 0.5]; // Only 2 elements
        let outcome = action.predict_outcome(&state);
        assert_eq!(outcome.len(), 6); // Padded to 6
    }
}

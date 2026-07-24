//! Grounded animal-communication evaluation and playback governance.

use crate::{CapabilityLevel, CommunicationEvidence, ExpressionAuthorization, ReplicationStatus};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub trait BioacousticEncoder {
    fn provider_id(&self) -> &str;
    fn model_hash(&self) -> &str;
    fn encode(&mut self, mono_audio: &[f32], sample_rate_hz: u32) -> Result<Vec<Vec<f32>>, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationBenchmark {
    pub provider: String,
    pub model_hash: String,
    pub benchmark: String,
    pub score: f64,
    pub pinned_baseline_score: f64,
    pub held_out_tasks: Vec<String>,
}

impl RepresentationBenchmark {
    pub fn passes(&self) -> bool {
        !self.provider.is_empty()
            && self.model_hash.len() >= 16
            && !self.benchmark.is_empty()
            && self.score.is_finite()
            && self.pinned_baseline_score.is_finite()
            && self.score >= self.pinned_baseline_score
            && !self.held_out_tasks.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnimalControl {
    ShuffledContext,
    IdentityOnly,
    LocationOnly,
    AcousticOnly,
    HeldOutIndividuals,
    HeldOutSites,
    TemporalPermutation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextPredictionEvaluation {
    pub task: String,
    pub primary_score: f64,
    pub control_scores: BTreeMap<AnimalControl, f64>,
    pub required_margin: f64,
}

impl ContextPredictionEvaluation {
    pub fn passes(&self) -> bool {
        self.primary_score.is_finite()
            && self.required_margin >= 0.0
            && self
                .control_scores
                .keys()
                .any(|c| c == &AnimalControl::HeldOutIndividuals)
            && self
                .control_scores
                .keys()
                .any(|c| c == &AnimalControl::HeldOutSites)
            && self.control_scores.values().all(|control| {
                control.is_finite() && self.primary_score >= control + self.required_margin
            })
    }
}

pub fn promotable_animal_capability(evidence: &[CommunicationEvidence]) -> CapabilityLevel {
    let intervention = crate::has_grounded_intervention(evidence);
    let independently_replicated = evidence
        .iter()
        .any(|item| item.replication == ReplicationStatus::IndependentlyReplicated);
    if intervention && independently_replicated {
        CapabilityLevel::Reference
    } else {
        CapabilityLevel::Structure
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaybackProtocol {
    pub species: String,
    pub population: String,
    pub welfare_review: String,
    pub stop_conditions: Vec<String>,
    pub authorization: ExpressionAuthorization,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimalEvaluationBundle {
    pub representation_baseline: String,
    pub context_prediction: ContextPredictionEvaluation,
    pub segmentation_stability: f64,
    pub train_individuals: Vec<String>,
    pub test_individuals: Vec<String>,
    pub train_sites: Vec<String>,
    pub test_sites: Vec<String>,
    pub temporal_permutation_score: f64,
}

impl AnimalEvaluationBundle {
    pub fn validate(&self) -> Result<(), String> {
        let train_individuals: std::collections::BTreeSet<_> =
            self.train_individuals.iter().collect();
        let test_individuals: std::collections::BTreeSet<_> =
            self.test_individuals.iter().collect();
        let train_sites: std::collections::BTreeSet<_> = self.train_sites.iter().collect();
        let test_sites: std::collections::BTreeSet<_> = self.test_sites.iter().collect();
        if self.representation_baseline.is_empty()
            || train_individuals.is_empty()
            || test_individuals.is_empty()
            || train_sites.is_empty()
            || test_sites.is_empty()
            || !train_individuals.is_disjoint(&test_individuals)
            || !train_sites.is_disjoint(&test_sites)
            || !(0.0..=1.0).contains(&self.segmentation_stability)
            || !self.context_prediction.passes()
            || self.context_prediction.primary_score
                < self.temporal_permutation_score + self.context_prediction.required_margin
        {
            return Err(
                "animal evaluation lacks disjoint holdouts, controls, or valid scores".into(),
            );
        }
        Ok(())
    }
}

impl PlaybackProtocol {
    pub fn validate(&self) -> Result<(), String> {
        if self.species.is_empty()
            || self.population.is_empty()
            || self.welfare_review.is_empty()
            || self.stop_conditions.is_empty()
            || self.authorization.protocol_id.is_empty()
            || self.authorization.reviewer.is_empty()
        {
            return Err("playback requires species, population, welfare review, stop conditions, and reviewer".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatasetSplit, EvidenceDomain, EvidenceKind, EvidenceRecord, ReplicationStatus};

    fn make_context_prediction(primary: f64, margin: f64) -> ContextPredictionEvaluation {
        ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: primary,
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, primary - margin - 0.01),
                (AnimalControl::HeldOutIndividuals, primary - margin - 0.01),
                (AnimalControl::HeldOutSites, primary - margin - 0.01),
                (AnimalControl::AcousticOnly, primary - margin - 0.01),
            ]),
            required_margin: margin,
        }
    }

    #[test]
    fn empty_controls_never_pass() {
        assert!(
            !ContextPredictionEvaluation {
                task: "context".into(),
                primary_score: 1.0,
                control_scores: BTreeMap::new(),
                required_margin: 0.01
            }
            .passes()
        );
    }

    #[test]
    fn context_prediction_passes_when_all_controls_present() {
        assert!(make_context_prediction(0.80, 0.05).passes());
    }

    #[test]
    fn context_prediction_fails_when_margin_not_met() {
        // Primary 0.80, control at 0.79, required_margin 0.05 — gap is only 0.01.
        let eval = ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: 0.80,
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, 0.79),
                (AnimalControl::HeldOutIndividuals, 0.79),
                (AnimalControl::HeldOutSites, 0.79),
            ]),
            required_margin: 0.05,
        };
        assert!(!eval.passes());
    }

    #[test]
    fn missing_held_out_individuals_fails() {
        let eval = ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: 0.80,
            // No HeldOutIndividuals or HeldOutSites
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, 0.70),
                (AnimalControl::AcousticOnly, 0.70),
            ]),
            required_margin: 0.05,
        };
        assert!(!eval.passes());
    }

    fn make_bundle(overlap_identities: bool) -> AnimalEvaluationBundle {
        let (test_individuals, test_sites) = if overlap_identities {
            (
                vec!["individual-A".into(), "individual-B".into()],
                vec!["site-3".into()],
            )
        } else {
            (
                vec!["individual-C".into(), "individual-D".into()],
                vec!["site-3".into()],
            )
        };
        AnimalEvaluationBundle {
            representation_baseline: "cav".into(),
            context_prediction: make_context_prediction(0.80, 0.05),
            segmentation_stability: 0.85,
            train_individuals: vec!["individual-A".into(), "individual-B".into()],
            test_individuals,
            train_sites: vec!["site-1".into(), "site-2".into()],
            test_sites,
            temporal_permutation_score: 0.60, // well below 0.80 - 0.05 margin
        }
    }

    #[test]
    fn animal_bundle_passes_when_all_conditions_met() {
        assert!(make_bundle(false).validate().is_ok());
    }

    #[test]
    fn animal_bundle_fails_when_train_test_individuals_overlap() {
        assert!(make_bundle(true).validate().is_err());
    }

    #[test]
    fn promotable_capability_requires_intervention_and_replication() {
        fn make_evidence(kind: EvidenceKind, replicated: bool) -> crate::CommunicationEvidence {
            let mut e = crate::CommunicationEvidence {
                id: String::new(),
                dataset_uri: "local:test".into(),
                dataset_hash: "d".into(),
                model_hash: "m".into(),
                lineage: vec![],
                split: DatasetSplit::default(),
                evidence_records: vec![EvidenceRecord {
                    kind,
                    description: "test".into(),
                    result_hash: "r".into(),
                    preregistered: true,
                    independently_replicated: replicated,
                }],
                preregistration_uri: None,
                replication: if replicated {
                    ReplicationStatus::IndependentlyReplicated
                } else {
                    ReplicationStatus::Unreplicated
                },
                calibration: vec![],
                experimental: false,
                domain: EvidenceDomain::AnimalCommunication,
            };
            e.id = e.computed_id().unwrap();
            e
        }
        // Only preregistered intervention + independent replication reaches Reference.
        let evidence = vec![make_evidence(EvidenceKind::Intervention, true)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Reference
        );

        // Observational correlation alone stays at Structure.
        let evidence = vec![make_evidence(EvidenceKind::ObservationalCorrelation, false)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Structure
        );

        // Intervention without independent replication stays at Structure.
        let evidence = vec![make_evidence(EvidenceKind::Intervention, false)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Structure
        );
    }
}

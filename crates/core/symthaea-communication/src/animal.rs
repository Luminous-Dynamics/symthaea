//! Grounded animal-communication evaluation and playback governance.

use crate::{CapabilityLevel, CommunicationEvidence, ExpressionAuthorization, ReplicationStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub trait BioacousticEncoder {
    fn provider_id(&self) -> &str;
    fn model_hash(&self) -> &str;
    fn encode(&mut self, mono_audio: &[f32], sample_rate_hz: u32) -> Result<Vec<Vec<f32>>, String>;
}

/// Declared source lineage for one dataset split. IDs refer to provenance
/// records, not authenticated facts; this type prevents accidental leakage
/// when the caller supplies complete and truthful source metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimalSplitLineage {
    pub split_id: String,
    pub sample_ids: Vec<String>,
    pub source_recording_ids: Vec<String>,
    #[serde(default)]
    pub individual_ids: Vec<String>,
    #[serde(default)]
    pub site_ids: Vec<String>,
}

/// Claims the evaluation intends to test through disjoint holdouts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimalSplitPolicy {
    pub hold_out_individuals: bool,
    pub hold_out_sites: bool,
}

/// Rejects sample/recording leakage and enforces optional individual/site
/// holdouts. Source-recording separation is always required because clips
/// or crops from one recording must not straddle evaluation splits.
pub fn validate_animal_split_lineage(
    splits: &[AnimalSplitLineage],
    policy: &AnimalSplitPolicy,
) -> Result<(), String> {
    if splits.len() < 2 {
        return Err("animal evaluation requires at least two splits".into());
    }

    let mut split_ids = BTreeSet::new();
    for split in splits {
        if split.split_id.trim().is_empty() || !split_ids.insert(&split.split_id) {
            return Err("split identifiers must be non-empty and unique".into());
        }
        validate_unique_ids(&split.sample_ids, "sample", &split.split_id, true)?;
        validate_unique_ids(
            &split.source_recording_ids,
            "source recording",
            &split.split_id,
            true,
        )?;
        validate_unique_ids(
            &split.individual_ids,
            "individual",
            &split.split_id,
            policy.hold_out_individuals,
        )?;
        validate_unique_ids(
            &split.site_ids,
            "site",
            &split.split_id,
            policy.hold_out_sites,
        )?;
    }

    for (index, left) in splits.iter().enumerate() {
        for right in &splits[index + 1..] {
            reject_overlap(
                &left.sample_ids,
                &right.sample_ids,
                "sample",
                &left.split_id,
                &right.split_id,
            )?;
            reject_overlap(
                &left.source_recording_ids,
                &right.source_recording_ids,
                "source recording",
                &left.split_id,
                &right.split_id,
            )?;
            if policy.hold_out_individuals {
                reject_overlap(
                    &left.individual_ids,
                    &right.individual_ids,
                    "individual",
                    &left.split_id,
                    &right.split_id,
                )?;
            }
            if policy.hold_out_sites {
                reject_overlap(
                    &left.site_ids,
                    &right.site_ids,
                    "site",
                    &left.split_id,
                    &right.split_id,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_unique_ids(
    ids: &[String],
    kind: &str,
    split_id: &str,
    required: bool,
) -> Result<(), String> {
    if required && ids.is_empty() {
        return Err(format!("{kind} lineage is required for split {split_id}"));
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.trim().is_empty() || !unique.insert(id) {
            return Err(format!(
                "{kind} IDs must be non-empty and unique in split {split_id}"
            ));
        }
    }
    Ok(())
}

fn reject_overlap(
    left_ids: &[String],
    right_ids: &[String],
    kind: &str,
    left_split: &str,
    right_split: &str,
) -> Result<(), String> {
    let left: BTreeSet<_> = left_ids.iter().collect();
    if right_ids.iter().any(|id| left.contains(id)) {
        return Err(format!(
            "{kind} leakage between splits {left_split} and {right_split}"
        ));
    }
    Ok(())
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
        // Require every preregistered confound control. A missing control is
        // not equivalent to a control that passed.
        let required_controls = [
            AnimalControl::ShuffledContext,
            AnimalControl::IdentityOnly,
            AnimalControl::LocationOnly,
            AnimalControl::AcousticOnly,
            AnimalControl::HeldOutIndividuals,
            AnimalControl::HeldOutSites,
        ];

        !self.task.trim().is_empty()
            && self.primary_score.is_finite()
            && self.required_margin.is_finite()
            && self.required_margin >= 0.0
            && required_controls
                .iter()
                .all(|control| self.control_scores.contains_key(control))
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
        let unique_nonempty = |values: &[String]| {
            let unique: std::collections::BTreeSet<_> = values.iter().collect();
            !values.is_empty()
                && values.iter().all(|value| !value.trim().is_empty())
                && unique.len() == values.len()
        };
        let train_individuals: std::collections::BTreeSet<_> =
            self.train_individuals.iter().collect();
        let test_individuals: std::collections::BTreeSet<_> =
            self.test_individuals.iter().collect();
        let train_sites: std::collections::BTreeSet<_> = self.train_sites.iter().collect();
        let test_sites: std::collections::BTreeSet<_> = self.test_sites.iter().collect();
        if self.representation_baseline.trim().is_empty()
            || !unique_nonempty(&self.train_individuals)
            || !unique_nonempty(&self.test_individuals)
            || !unique_nonempty(&self.train_sites)
            || !unique_nonempty(&self.test_sites)
            || !train_individuals.is_disjoint(&test_individuals)
            || !train_sites.is_disjoint(&test_sites)
            || !(0.0..=1.0).contains(&self.segmentation_stability)
            || !self.temporal_permutation_score.is_finite()
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

    fn split(
        split_id: &str,
        samples: &[&str],
        recordings: &[&str],
        individuals: &[&str],
        sites: &[&str],
    ) -> AnimalSplitLineage {
        AnimalSplitLineage {
            split_id: split_id.into(),
            sample_ids: samples.iter().map(|value| (*value).into()).collect(),
            source_recording_ids: recordings.iter().map(|value| (*value).into()).collect(),
            individual_ids: individuals.iter().map(|value| (*value).into()).collect(),
            site_ids: sites.iter().map(|value| (*value).into()).collect(),
        }
    }

    #[test]
    fn animal_lineage_rejects_two_crops_from_same_source_recording() {
        let splits = vec![
            split("train", &["crop-a"], &["recording-1"], &[], &[]),
            split("test", &["crop-b"], &["recording-1"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    #[test]
    fn animal_lineage_rejects_missing_source_recording_provenance() {
        let splits = vec![
            split("train", &["clip-a"], &[], &[], &[]),
            split("test", &["clip-b"], &["recording-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    #[test]
    fn animal_lineage_requires_disjoint_individuals_when_claimed() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-1"], &["site-b"]),
        ];
        let policy = AnimalSplitPolicy {
            hold_out_individuals: true,
            hold_out_sites: false,
        };
        assert!(validate_animal_split_lineage(&splits, &policy).is_err());
    }

    #[test]
    fn animal_lineage_requires_disjoint_sites_when_claimed() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-2"], &["site-a"]),
        ];
        let policy = AnimalSplitPolicy {
            hold_out_individuals: false,
            hold_out_sites: true,
        };
        assert!(validate_animal_split_lineage(&splits, &policy).is_err());
    }

    #[test]
    fn animal_lineage_allows_known_individual_overlap_when_not_claiming_holdout() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-1"], &["site-b"]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_ok());
    }

    #[test]
    fn animal_lineage_rejects_duplicate_and_blank_ids() {
        let duplicate = vec![
            split("train", &["a", "a"], &["rec-a"], &[], &[]),
            split("test", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&duplicate, &AnimalSplitPolicy::default()).is_err());

        let blank_source = vec![
            split("train", &["a"], &[" "], &[], &[]),
            split("test", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(
            validate_animal_split_lineage(&blank_source, &AnimalSplitPolicy::default()).is_err()
        );
    }

    #[test]
    fn animal_lineage_rejects_duplicate_split_ids() {
        let splits = vec![
            split("same", &["a"], &["rec-a"], &[], &[]),
            split("same", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    fn make_context_prediction(primary: f64, margin: f64) -> ContextPredictionEvaluation {
        ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: primary,
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, primary - margin - 0.01),
                (AnimalControl::IdentityOnly, primary - margin - 0.01),
                (AnimalControl::LocationOnly, primary - margin - 0.01),
                (AnimalControl::AcousticOnly, primary - margin - 0.01),
                (AnimalControl::HeldOutIndividuals, primary - margin - 0.01),
                (AnimalControl::HeldOutSites, primary - margin - 0.01),
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
    fn missing_identity_or_location_control_fails_closed() {
        let mut eval = make_context_prediction(0.80, 0.05);
        eval.control_scores.remove(&AnimalControl::IdentityOnly);
        assert!(!eval.passes());

        let mut eval = make_context_prediction(0.80, 0.05);
        eval.control_scores.remove(&AnimalControl::LocationOnly);
        assert!(!eval.passes());
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
    fn animal_bundle_rejects_non_finite_temporal_permutation() {
        let mut bundle = make_bundle(false);
        bundle.temporal_permutation_score = f64::NAN;
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn animal_bundle_rejects_duplicate_or_empty_split_ids() {
        let mut duplicate = make_bundle(false);
        duplicate.train_sites.push("site-1".into());
        assert!(duplicate.validate().is_err());

        let mut empty = make_bundle(false);
        empty.test_individuals.push(" ".into());
        assert!(empty.validate().is_err());
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

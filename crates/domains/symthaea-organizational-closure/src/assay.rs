//! Deterministic, model-relative assay adequacy for AOBench.
//!
//! This initial evaluator answers a narrow question over a frozen finite model
//! class: did the registered deterministic assays distinguish every pair
//! declared non-equivalent by the target equivalence partition?
//!
//! Missing observations are unresolved, never evidence of equivalence.
//! Stochastic response distributions, power analysis, multiplicity control,
//! and adaptive assay selection are intentionally left for a later tranche.

use crate::ContentDigest;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(pub String);

impl From<&str> for ModelId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssayId(pub String);

impl From<&str> for AssayId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

/// The target equivalence class is frozen before assay evaluation. Two models
/// in the same class are considered equivalent under the declared target
/// relation; different class labels mean they must be distinguishable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateModel {
    pub id: ModelId,
    pub target_equivalence_class: String,
}

impl CandidateModel {
    pub fn new(id: impl Into<ModelId>, target_equivalence_class: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            target_equivalence_class: target_equivalence_class.into(),
        }
    }
}

/// A deterministic response signature for one candidate under one assay.
/// The digest must be computed over a frozen canonical outcome representation
/// by the assay harness, not accepted as proof of authenticity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssayObservation {
    pub model_id: ModelId,
    pub assay_id: AssayId,
    pub response_digest: ContentDigest,
}

impl AssayObservation {
    pub fn new(
        model_id: impl Into<ModelId>,
        assay_id: impl Into<AssayId>,
        response_digest: ContentDigest,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            assay_id: assay_id.into(),
            response_digest,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssayFindingCode {
    EmptyCandidateSet,
    DuplicateModelId,
    EmptyModelId,
    EmptyEquivalenceClass,
    UnknownModelObservation,
    EmptyAssayId,
    DuplicateModelAssayObservation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssayFinding {
    pub code: AssayFindingCode,
    pub model_id: Option<ModelId>,
    pub assay_id: Option<AssayId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPairAssessment {
    pub first: ModelId,
    pub second: ModelId,
    pub first_equivalence_class: String,
    pub second_equivalence_class: String,
    pub distinguishing_assays: Vec<AssayId>,
}

impl ModelPairAssessment {
    pub fn is_resolved(&self) -> bool {
        !self.distinguishing_assays.is_empty()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssayAdequacyReport {
    pub findings: Vec<AssayFinding>,
    pub pair_assessments: Vec<ModelPairAssessment>,
    pub non_equivalent_pair_count: usize,
    pub resolved_pair_count: usize,
}

impl AssayAdequacyReport {
    /// None means the frozen candidate class contains no non-equivalent
    /// pairs, so conservativity is not applicable. Some(true) is a
    /// model-class-relative statement, not a claim about the real world.
    pub fn is_conservative_for_declared_classes(&self) -> Option<bool> {
        if self.non_equivalent_pair_count == 0 {
            return None;
        }
        Some(self.findings.is_empty() && self.resolved_pair_count == self.non_equivalent_pair_count)
    }

    pub fn resolved_pair_fraction(&self) -> Option<f64> {
        if self.non_equivalent_pair_count == 0 {
            return None;
        }
        Some(self.resolved_pair_count as f64 / self.non_equivalent_pair_count as f64)
    }
}

/// Evaluate a deterministic assay panel against a frozen finite candidate
/// model class. Only pairs in different target-equivalence classes must be
/// separated. A pair is resolved if at least one assay has observations for
/// both models and the response digests differ.
///
/// This is a deterministic pilot primitive. It does not implement statistical
/// distinguishability for noisy systems, confidence intervals, test power,
/// or multiplicity correction.
pub fn evaluate_assay_adequacy(
    candidates: &[CandidateModel],
    observations: &[AssayObservation],
) -> AssayAdequacyReport {
    let mut report = AssayAdequacyReport::default();

    if candidates.is_empty() {
        report.findings.push(AssayFinding {
            code: AssayFindingCode::EmptyCandidateSet,
            model_id: None,
            assay_id: None,
            detail: "candidate model set is empty".to_owned(),
        });
        return report;
    }

    let mut candidate_by_id: BTreeMap<ModelId, &CandidateModel> = BTreeMap::new();
    for candidate in candidates {
        if candidate.id.0.trim().is_empty() {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::EmptyModelId,
                model_id: Some(candidate.id.clone()),
                assay_id: None,
                detail: "candidate model ID is empty".to_owned(),
            });
            continue;
        }
        if candidate.target_equivalence_class.trim().is_empty() {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::EmptyEquivalenceClass,
                model_id: Some(candidate.id.clone()),
                assay_id: None,
                detail: "target equivalence class is empty".to_owned(),
            });
            continue;
        }
        if candidate_by_id
            .insert(candidate.id.clone(), candidate)
            .is_some()
        {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::DuplicateModelId,
                model_id: Some(candidate.id.clone()),
                assay_id: None,
                detail: "candidate model ID is duplicated".to_owned(),
            });
        }
    }

    let mut signatures: BTreeMap<(ModelId, AssayId), ContentDigest> = BTreeMap::new();
    for observation in observations {
        if observation.assay_id.0.trim().is_empty() {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::EmptyAssayId,
                model_id: Some(observation.model_id.clone()),
                assay_id: Some(observation.assay_id.clone()),
                detail: "assay ID is empty".to_owned(),
            });
            continue;
        }
        if !candidate_by_id.contains_key(&observation.model_id) {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::UnknownModelObservation,
                model_id: Some(observation.model_id.clone()),
                assay_id: Some(observation.assay_id.clone()),
                detail: "assay observation references an unknown candidate model".to_owned(),
            });
            continue;
        }
        let key = (observation.model_id.clone(), observation.assay_id.clone());
        if signatures
            .insert(key, observation.response_digest)
            .is_some()
        {
            report.findings.push(AssayFinding {
                code: AssayFindingCode::DuplicateModelAssayObservation,
                model_id: Some(observation.model_id.clone()),
                assay_id: Some(observation.assay_id.clone()),
                detail: "candidate has more than one response for the same assay".to_owned(),
            });
        }
    }

    let valid_candidates: Vec<&CandidateModel> = candidate_by_id.values().copied().collect();
    for first_index in 0..valid_candidates.len() {
        for second_index in (first_index + 1)..valid_candidates.len() {
            let first = valid_candidates[first_index];
            let second = valid_candidates[second_index];

            if first.target_equivalence_class == second.target_equivalence_class {
                continue;
            }

            report.non_equivalent_pair_count += 1;
            let mut distinguishing_assays = BTreeSet::new();

            for ((model_id, assay_id), first_digest) in &signatures {
                if model_id != &first.id {
                    continue;
                }
                if let Some(second_digest) = signatures.get(&(second.id.clone(), assay_id.clone()))
                    && first_digest != second_digest
                {
                    distinguishing_assays.insert(assay_id.clone());
                }
            }

            let assessment = ModelPairAssessment {
                first: first.id.clone(),
                second: second.id.clone(),
                first_equivalence_class: first.target_equivalence_class.clone(),
                second_equivalence_class: second.target_equivalence_class.clone(),
                distinguishing_assays: distinguishing_assays.into_iter().collect(),
            };

            if assessment.is_resolved() {
                report.resolved_pair_count += 1;
            }
            report.pair_assessments.push(assessment);
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &[u8]) -> ContentDigest {
        ContentDigest::of(value)
    }

    #[test]
    fn complete_panel_is_conservative_for_frozen_candidate_classes() {
        let candidates = vec![
            CandidateModel::new("model-a", "class-a"),
            CandidateModel::new("model-b", "class-b"),
        ];
        let observations = vec![
            AssayObservation::new("model-a", "remove-producer", digest(b"degraded")),
            AssayObservation::new("model-b", "remove-producer", digest(b"healthy")),
        ];

        let report = evaluate_assay_adequacy(&candidates, &observations);
        assert_eq!(report.non_equivalent_pair_count, 1);
        assert_eq!(report.resolved_pair_count, 1);
        assert_eq!(report.is_conservative_for_declared_classes(), Some(true));
        assert_eq!(report.resolved_pair_fraction(), Some(1.0));
    }

    #[test]
    fn incomplete_panel_leaves_pair_unresolved_not_equivalent() {
        let candidates = vec![
            CandidateModel::new("model-a", "class-a"),
            CandidateModel::new("model-b", "class-b"),
        ];
        let observations = vec![AssayObservation::new(
            "model-a",
            "remove-producer",
            digest(b"degraded"),
        )];

        let report = evaluate_assay_adequacy(&candidates, &observations);
        assert_eq!(report.non_equivalent_pair_count, 1);
        assert_eq!(report.resolved_pair_count, 0);
        assert_eq!(report.is_conservative_for_declared_classes(), Some(false));
        assert!(!report.pair_assessments[0].is_resolved());
    }

    #[test]
    fn target_equivalent_models_need_not_be_distinguished() {
        let candidates = vec![
            CandidateModel::new("model-a", "same-class"),
            CandidateModel::new("model-b", "same-class"),
        ];
        let observations = vec![
            AssayObservation::new("model-a", "assay-1", digest(b"response-a")),
            AssayObservation::new("model-b", "assay-1", digest(b"response-b")),
        ];

        let report = evaluate_assay_adequacy(&candidates, &observations);
        assert_eq!(report.non_equivalent_pair_count, 0);
        assert_eq!(report.is_conservative_for_declared_classes(), None);
        assert!(report.pair_assessments.is_empty());
    }

    #[test]
    fn unknown_or_duplicate_observations_block_qualification() {
        let candidates = vec![
            CandidateModel::new("model-a", "class-a"),
            CandidateModel::new("model-b", "class-b"),
        ];
        let observations = vec![
            AssayObservation::new("model-a", "assay-1", digest(b"a")),
            AssayObservation::new("model-a", "assay-1", digest(b"b")),
            AssayObservation::new("unknown-model", "assay-1", digest(b"unknown")),
            AssayObservation::new("model-b", "assay-1", digest(b"a")),
        ];

        let report = evaluate_assay_adequacy(&candidates, &observations);
        assert_eq!(report.non_equivalent_pair_count, 1);
        assert_eq!(report.resolved_pair_count, 1);
        assert_eq!(report.is_conservative_for_declared_classes(), Some(false));
        assert!(
            report.findings.iter().any(|finding| {
                finding.code == AssayFindingCode::DuplicateModelAssayObservation
            })
        );
        assert!(
            report
                .findings
                .iter()
                .any(|finding| { finding.code == AssayFindingCode::UnknownModelObservation })
        );
    }

    #[test]
    fn empty_candidate_class_is_not_a_pass() {
        let report = evaluate_assay_adequacy(&[], &[]);
        assert_eq!(report.is_conservative_for_declared_classes(), None);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].code, AssayFindingCode::EmptyCandidateSet);
    }
}

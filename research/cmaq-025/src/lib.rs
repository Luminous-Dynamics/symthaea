#![forbid(unsafe_code)]

//! Finite identifiability diagnostics under observational aliasing.
//! Digests are syntax-checked here; an independent boundary must recompute and authenticate them.

use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateOutput {
    Point(i64),
    Set(Vec<i64>),
    Unknown(UnknownReason),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    AmbiguousContext,
    InsufficientSupport,
    OutOfScope,
    IntegrityFailure,
    UnseenMechanism,
}

/// Evaluator-only family and reference fields are appended after candidate inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationRecord {
    /// Digest of complete candidate-visible state, action, context, query, horizon,
    /// and permitted probe evidence.
    pub candidate_view_sha256: String,
    pub environment_family_id: String,
    pub independent_cluster_id: String,
    pub contract_id: String,
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub reference_response: i64,
    pub candidate_output: CandidateOutput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyEvaluation,
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    MixedContract,
    MixedSourceManifest,
    MixedCandidateCheckpoint,
    ClusterAliasedAcrossFamilies,
    CandidateViewOutputMismatch,
    ClusterResponseConflict,
    NonCanonicalPredictionSet,
}

/// Cells are unique (candidate view, cluster) pairs. They are descriptive, not independent
/// samples when a cluster appears in multiple views; inference must cluster at cluster_id.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdentifiabilityScore {
    pub total_records: usize,
    pub candidate_view_groups: usize,
    pub identifiable_view_groups: usize,
    pub unidentifiable_view_groups: usize,
    pub independent_clusters: usize,
    pub environment_families: usize,
    pub view_cluster_cells: usize,
    pub identifiable_view_cluster_cells: usize,
    pub unidentifiable_view_cluster_cells: usize,
    pub point_prediction_cells: usize,
    pub point_error_cells: usize,
    pub set_prediction_cells: usize,
    pub set_miss_cells: usize,
    pub unknown_cells: usize,
    pub ambiguous_unknown_cells: usize,
    pub identifiable_unknown_cells: usize,
    pub ambiguous_point_forecast_groups: usize,
    pub ambiguous_complete_set_groups: usize,
    pub ambiguous_incomplete_set_groups: usize,
    pub ambiguous_unsafe_groups: usize,
    pub irreducible_point_error_floor: usize,
    pub prediction_set_width_sum: usize,
    pub maximum_prediction_set_width: usize,
}

impl IdentifiabilityScore {
    pub fn point_coverage(&self) -> Option<f64> {
        ratio(self.point_prediction_cells, self.view_cluster_cells)
    }
    pub fn point_selective_risk(&self) -> Option<f64> {
        ratio(self.point_error_cells, self.point_prediction_cells)
    }
    pub fn ambiguity_abstention_rate(&self) -> Option<f64> {
        ratio(self.ambiguous_unknown_cells, self.unidentifiable_view_cluster_cells)
    }
    pub fn unnecessary_abstention_rate(&self) -> Option<f64> {
        ratio(self.identifiable_unknown_cells, self.identifiable_view_cluster_cells)
    }
    pub fn prediction_set_miss_rate(&self) -> Option<f64> {
        ratio(self.set_miss_cells, self.set_prediction_cells)
    }
    pub fn irreducible_point_error_rate(&self) -> Option<f64> {
        ratio(self.irreducible_point_error_floor, self.unidentifiable_view_cluster_cells)
    }
}

/// Scores already-formed candidate outputs; oracle data is never supplied to a predictor.
pub fn score_identifiability(
    records: &[EvaluationRecord],
) -> Result<IdentifiabilityScore, ValidationError> {
    let first = records.first().ok_or(ValidationError::EmptyEvaluation)?;
    let mut views: BTreeMap<String, Vec<&EvaluationRecord>> = BTreeMap::new();
    let mut cluster_families: HashMap<String, String> = HashMap::new();
    let mut clusters = HashSet::new();
    let mut families = HashSet::new();

    for record in records {
        for (value, name) in [
            (&record.candidate_view_sha256, "candidate_view_sha256"),
            (&record.environment_family_id, "environment_family_id"),
            (&record.independent_cluster_id, "independent_cluster_id"),
            (&record.contract_id, "contract_id"),
        ] {
            if value.trim().is_empty() {
                return Err(ValidationError::EmptyIdentity(name));
            }
        }
        for (name, digest) in [
            ("candidate_view_sha256", &record.candidate_view_sha256),
            ("contract_sha256", &record.contract_sha256),
            ("source_manifest_sha256", &record.source_manifest_sha256),
            ("candidate_checkpoint_sha256", &record.candidate_checkpoint_sha256),
        ] {
            if !is_sha256_hex(digest) {
                return Err(ValidationError::InvalidSha256(name));
            }
        }
        if record.contract_id != first.contract_id
            || record.contract_sha256 != first.contract_sha256
        {
            return Err(ValidationError::MixedContract);
        }
        if record.source_manifest_sha256 != first.source_manifest_sha256 {
            return Err(ValidationError::MixedSourceManifest);
        }
        if record.candidate_checkpoint_sha256 != first.candidate_checkpoint_sha256 {
            return Err(ValidationError::MixedCandidateCheckpoint);
        }
        if let CandidateOutput::Set(values) = &record.candidate_output {
            if values.is_empty() || values.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(ValidationError::NonCanonicalPredictionSet);
            }
        }
        if let Some(existing) = cluster_families.get(&record.independent_cluster_id) {
            if existing != &record.environment_family_id {
                return Err(ValidationError::ClusterAliasedAcrossFamilies);
            }
        } else {
            cluster_families.insert(
                record.independent_cluster_id.clone(),
                record.environment_family_id.clone(),
            );
        }
        clusters.insert(record.independent_cluster_id.clone());
        families.insert(record.environment_family_id.clone());
        views.entry(record.candidate_view_sha256.clone()).or_default().push(record);
    }

    let mut score = IdentifiabilityScore {
        total_records: records.len(),
        candidate_view_groups: views.len(),
        independent_clusters: clusters.len(),
        environment_families: families.len(),
        ..IdentifiabilityScore::default()
    };

    for group in views.values() {
        let output = &group[0].candidate_output;
        if group.iter().any(|record| &record.candidate_output != output) {
            return Err(ValidationError::CandidateViewOutputMismatch);
        }
        let mut cluster_responses = BTreeMap::new();
        for record in group {
            match cluster_responses.insert(
                record.independent_cluster_id.as_str(),
                record.reference_response,
            ) {
                Some(previous) if previous != record.reference_response => {
                    return Err(ValidationError::ClusterResponseConflict);
                }
                _ => {}
            }
        }
        let count = cluster_responses.len();
        let mut response_counts = BTreeMap::new();
        for response in cluster_responses.values() {
            *response_counts.entry(*response).or_insert(0usize) += 1;
        }
        let ambiguous = response_counts.len() > 1;
        score.view_cluster_cells += count;
        if ambiguous {
            score.unidentifiable_view_groups += 1;
            score.unidentifiable_view_cluster_cells += count;
            let modal = response_counts.values().copied().max().unwrap_or(0);
            score.irreducible_point_error_floor += count - modal;
        } else {
            score.identifiable_view_groups += 1;
            score.identifiable_view_cluster_cells += count;
        }
        match output {
            CandidateOutput::Point(predicted) => {
                score.point_prediction_cells += count;
                score.point_error_cells += cluster_responses
                    .values()
                    .filter(|actual| **actual != *predicted)
                    .count();
                if ambiguous {
                    score.ambiguous_point_forecast_groups += 1;
                    score.ambiguous_unsafe_groups += 1;
                }
            }
            CandidateOutput::Set(values) => {
                score.set_prediction_cells += count;
                score.prediction_set_width_sum += values.len() * count;
                score.maximum_prediction_set_width =
                    score.maximum_prediction_set_width.max(values.len());
                score.set_miss_cells += cluster_responses
                    .values()
                    .filter(|actual| values.binary_search(*actual).is_err())
                    .count();
                if ambiguous {
                    let complete = response_counts
                        .keys()
                        .all(|actual| values.binary_search(actual).is_ok());
                    if complete {
                        score.ambiguous_complete_set_groups += 1;
                    } else {
                        score.ambiguous_incomplete_set_groups += 1;
                        score.ambiguous_unsafe_groups += 1;
                    }
                }
            }
            CandidateOutput::Unknown(_) => {
                score.unknown_cells += count;
                if ambiguous {
                    score.ambiguous_unknown_cells += count;
                } else {
                    score.identifiable_unknown_cells += count;
                }
            }
        }
    }
    Ok(score)
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String {
        c.to_string().repeat(64)
    }

    fn record(
        view: char,
        family: &str,
        cluster: &str,
        response: i64,
        output: CandidateOutput,
    ) -> EvaluationRecord {
        EvaluationRecord {
            candidate_view_sha256: digest(view),
            environment_family_id: family.into(),
            independent_cluster_id: cluster.into(),
            contract_id: "contract-v1".into(),
            contract_sha256: digest('a'),
            source_manifest_sha256: digest('b'),
            candidate_checkpoint_sha256: digest('c'),
            reference_response: response,
            candidate_output: output,
        }
    }

    #[test]
    fn observational_alias_exposes_irreducible_point_error() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('d', "fb", "cb", 20, CandidateOutput::Point(10)),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.unidentifiable_view_groups, 1);
        assert_eq!(score.irreducible_point_error_floor, 1);
        assert_eq!(score.point_error_cells, 1);
        assert_eq!(score.irreducible_point_error_rate(), Some(0.5));
    }

    #[test]
    fn finite_set_covers_all_observed_responses_in_alias_group() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Set(vec![10, 20])),
            record('d', "fb", "cb", 20, CandidateOutput::Set(vec![10, 20])),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.ambiguous_complete_set_groups, 1);
        assert_eq!(score.ambiguous_unsafe_groups, 0);
        assert_eq!(score.set_miss_cells, 0);
        assert_eq!(score.maximum_prediction_set_width, 2);
    }

    #[test]
    fn unknown_is_safe_on_aliases_but_over_abstention_is_measured() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Unknown(UnknownReason::AmbiguousContext)),
            record('d', "fb", "cb", 20, CandidateOutput::Unknown(UnknownReason::AmbiguousContext)),
            record(
                'e',
                "fc",
                "cc",
                30,
                CandidateOutput::Unknown(UnknownReason::InsufficientSupport),
            ),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.ambiguity_abstention_rate(), Some(1.0));
        assert_eq!(score.unnecessary_abstention_rate(), Some(1.0));
    }

    #[test]
    fn benign_family_and_cluster_shift_preserves_identifiability() {
        let records = vec![
            record('d', "train", "train-cluster", 10, CandidateOutput::Point(10)),
            record('d', "heldout", "heldout-cluster", 10, CandidateOutput::Point(10)),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.identifiable_view_groups, 1);
        assert_eq!(score.environment_families, 2);
        assert_eq!(score.independent_clusters, 2);
        assert_eq!(score.point_error_cells, 0);
    }

    #[test]
    fn informative_probe_must_change_candidate_view_to_split_aliases() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('e', "fb", "cb", 20, CandidateOutput::Point(20)),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.unidentifiable_view_groups, 0);
        assert_eq!(score.identifiable_view_groups, 2);
        assert_eq!(score.point_error_cells, 0);
    }

    #[test]
    fn uninformative_probe_leaves_alias_unresolved() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Unknown(UnknownReason::AmbiguousContext)),
            record('d', "fb", "cb", 20, CandidateOutput::Unknown(UnknownReason::AmbiguousContext)),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.unidentifiable_view_groups, 1);
        assert_eq!(score.ambiguous_unknown_cells, 2);
    }

    #[test]
    fn repeated_records_do_not_inflate_cluster_count() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('d', "fb", "cb", 20, CandidateOutput::Point(10)),
        ];
        let score = score_identifiability(&records).unwrap();
        assert_eq!(score.total_records, 3);
        assert_eq!(score.independent_clusters, 2);
        assert_eq!(score.view_cluster_cells, 2);
        assert_eq!(score.irreducible_point_error_floor, 1);
    }

    #[test]
    fn reused_cluster_id_across_families_is_rejected() {
        let records = vec![
            record('d', "fa", "same", 10, CandidateOutput::Point(10)),
            record('e', "fb", "same", 20, CandidateOutput::Point(20)),
        ];
        assert_eq!(
            score_identifiability(&records),
            Err(ValidationError::ClusterAliasedAcrossFamilies)
        );
    }

    #[test]
    fn same_view_with_different_candidate_outputs_is_rejected() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('d', "fb", "cb", 20, CandidateOutput::Point(20)),
        ];
        assert_eq!(
            score_identifiability(&records),
            Err(ValidationError::CandidateViewOutputMismatch)
        );
    }

    #[test]
    fn mixed_checkpoint_and_malformed_digest_are_rejected() {
        let first = record('d', "fa", "ca", 10, CandidateOutput::Point(10));
        let mut second = record('e', "fb", "cb", 20, CandidateOutput::Point(20));
        second.candidate_checkpoint_sha256 = digest('d');
        assert_eq!(
            score_identifiability(&[first.clone(), second]),
            Err(ValidationError::MixedCandidateCheckpoint)
        );
        let mut malformed = first;
        malformed.contract_sha256 = "bad".into();
        assert_eq!(
            score_identifiability(&[malformed]),
            Err(ValidationError::InvalidSha256("contract_sha256"))
        );
    }

    #[test]
    fn conflicting_reference_outcomes_inside_one_cluster_are_rejected() {
        let records = vec![
            record('d', "fa", "ca", 10, CandidateOutput::Point(10)),
            record('d', "fa", "ca", 20, CandidateOutput::Point(10)),
        ];
        assert_eq!(
            score_identifiability(&records),
            Err(ValidationError::ClusterResponseConflict)
        );
    }

    #[test]
    fn prediction_sets_must_be_sorted_unique_and_non_empty() {
        let records = vec![record('d', "fa", "ca", 10, CandidateOutput::Set(vec![20, 10]))];
        assert_eq!(
            score_identifiability(&records),
            Err(ValidationError::NonCanonicalPredictionSet)
        );
    }
}

#![forbid(unsafe_code)]

//! Finite identifiability diagnostics under observational aliasing.
//! Candidate predictions and evaluator-only annotations are separate types.
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

/// Candidate-side result: no family label, cluster label, or reference outcome.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidatePrediction {
    /// Digest of all candidate-visible state, context, action, query, horizon,
    /// authorized evidence, and permitted probe results.
    pub candidate_view_sha256: String,
    pub output: CandidateOutput,
}

/// Evaluator-only annotation; never pass this type to candidate inference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OracleAnnotation {
    pub candidate_view_sha256: String,
    pub environment_family_id: String,
    pub independent_cluster_id: String,
    pub reference_response: i64,
}

/// One immutable custody context applies to the entire evaluation batch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationCustody {
    pub contract_id: String,
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub candidate_checkpoint_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationError {
    EmptyPredictions,
    EmptyAnnotations,
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    UnmatchedOracleView,
    UnmatchedCandidateView,
    ClusterAliasedAcrossFamilies,
    CandidateViewOutputMismatch,
    ClusterResponseConflict,
    NonCanonicalPredictionSet,
}

/// View-cluster cells are descriptive, not independent samples if a cluster
/// appears in multiple views. Statistical inference must aggregate by cluster ID.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IdentifiabilityScore {
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

/// Join already-formed candidate outputs to evaluator-only annotations and score them.
/// CandidatePrediction has no oracle metadata; this function is the only join point.
pub fn score_identifiability(
    predictions: &[CandidatePrediction],
    annotations: &[OracleAnnotation],
    custody: &EvaluationCustody,
) -> Result<IdentifiabilityScore, ValidationError> {
    if predictions.is_empty() {
        return Err(ValidationError::EmptyPredictions);
    }
    if annotations.is_empty() {
        return Err(ValidationError::EmptyAnnotations);
    }

    for (value, name) in [
        (&custody.contract_id, "contract_id"),
        (&custody.contract_sha256, "contract_sha256"),
        (&custody.source_manifest_sha256, "source_manifest_sha256"),
        (&custody.candidate_checkpoint_sha256, "candidate_checkpoint_sha256"),
    ] {
        if value.trim().is_empty() {
            return Err(ValidationError::EmptyIdentity(name));
        }
    }
    for (name, digest) in [
        ("contract_sha256", &custody.contract_sha256),
        ("source_manifest_sha256", &custody.source_manifest_sha256),
        ("candidate_checkpoint_sha256", &custody.candidate_checkpoint_sha256),
    ] {
        if !is_sha256_hex(digest) {
            return Err(ValidationError::InvalidSha256(name));
        }
    }

    let mut predicted: BTreeMap<String, CandidateOutput> = BTreeMap::new();
    for item in predictions {
        if !is_sha256_hex(&item.candidate_view_sha256) {
            return Err(ValidationError::InvalidSha256("candidate_view_sha256"));
        }
        if let CandidateOutput::Set(values) = &item.output {
            if values.is_empty() || values.windows(2).any(|p| p[0] >= p[1]) {
                return Err(ValidationError::NonCanonicalPredictionSet);
            }
        }
        match predicted.entry(item.candidate_view_sha256.clone()) {
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get() != &item.output {
                    return Err(ValidationError::CandidateViewOutputMismatch);
                }
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(item.output.clone());
            }
        }
    }

    let mut oracle: BTreeMap<String, Vec<&OracleAnnotation>> = BTreeMap::new();
    let mut cluster_families: HashMap<String, String> = HashMap::new();
    let mut clusters = HashSet::new();
    let mut families = HashSet::new();
    for item in annotations {
        if !is_sha256_hex(&item.candidate_view_sha256) {
            return Err(ValidationError::InvalidSha256("candidate_view_sha256"));
        }
        if !predicted.contains_key(&item.candidate_view_sha256) {
            return Err(ValidationError::UnmatchedOracleView);
        }
        for (value, name) in [
            (&item.environment_family_id, "environment_family_id"),
            (&item.independent_cluster_id, "independent_cluster_id"),
        ] {
            if value.trim().is_empty() {
                return Err(ValidationError::EmptyIdentity(name));
            }
        }
        match cluster_families.entry(item.independent_cluster_id.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => {
                if entry.get() != &item.environment_family_id {
                    return Err(ValidationError::ClusterAliasedAcrossFamilies);
                }
            }
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(item.environment_family_id.clone());
            }
        }
        clusters.insert(item.independent_cluster_id.clone());
        families.insert(item.environment_family_id.clone());
        oracle.entry(item.candidate_view_sha256.clone()).or_default().push(item);
    }
    if predicted.keys().any(|view| !oracle.contains_key(view)) {
        return Err(ValidationError::UnmatchedCandidateView);
    }

    let mut score = IdentifiabilityScore {
        candidate_view_groups: predicted.len(),
        independent_clusters: clusters.len(),
        environment_families: families.len(),
        ..IdentifiabilityScore::default()
    };

    for (view, group) in &oracle {
        let output = &predicted[view];
        let mut responses = BTreeMap::new();
        for item in group {
            match responses.insert(item.independent_cluster_id.as_str(), item.reference_response) {
                Some(old) if old != item.reference_response => {
                    return Err(ValidationError::ClusterResponseConflict);
                }
                _ => {}
            }
        }
        let count = responses.len();
        let mut frequencies = BTreeMap::new();
        for response in responses.values() {
            *frequencies.entry(*response).or_insert(0usize) += 1;
        }
        let ambiguous = frequencies.len() > 1;
        score.view_cluster_cells += count;
        if ambiguous {
            score.unidentifiable_view_groups += 1;
            score.unidentifiable_view_cluster_cells += count;
            let mode = frequencies.values().copied().max().unwrap_or(0);
            score.irreducible_point_error_floor += count - mode;
        } else {
            score.identifiable_view_groups += 1;
            score.identifiable_view_cluster_cells += count;
        }

        match output {
            CandidateOutput::Point(value) => {
                score.point_prediction_cells += count;
                score.point_error_cells += responses.values().filter(|actual| **actual != *value).count();
                if ambiguous {
                    score.ambiguous_point_forecast_groups += 1;
                    score.ambiguous_unsafe_groups += 1;
                }
            }
            CandidateOutput::Set(values) => {
                score.set_prediction_cells += count;
                score.prediction_set_width_sum += values.len() * count;
                score.maximum_prediction_set_width = score.maximum_prediction_set_width.max(values.len());
                score.set_miss_cells += responses.values().filter(|actual| values.binary_search(*actual).is_err()).count();
                if ambiguous {
                    let covers_observed = frequencies.keys().all(|actual| values.binary_search(actual).is_ok());
                    if covers_observed {
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
    if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) }
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String { c.to_string().repeat(64) }
    fn custody() -> EvaluationCustody {
        EvaluationCustody {
            contract_id: "contract-v1".into(),
            contract_sha256: digest('a'),
            source_manifest_sha256: digest('b'),
            candidate_checkpoint_sha256: digest('c'),
        }
    }
    fn pred(view: char, output: CandidateOutput) -> CandidatePrediction {
        CandidatePrediction { candidate_view_sha256: digest(view), output }
    }
    fn oracle(view: char, family: &str, cluster: &str, response: i64) -> OracleAnnotation {
        OracleAnnotation {
            candidate_view_sha256: digest(view),
            environment_family_id: family.into(),
            independent_cluster_id: cluster.into(),
            reference_response: response,
        }
    }

    #[test]
    fn alias_reports_irreducible_point_error() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![oracle('d', "fa", "ca", 10), oracle('d', "fb", "cb", 20)];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.unidentifiable_view_groups, 1);
        assert_eq!(s.irreducible_point_error_floor, 1);
        assert_eq!(s.point_error_cells, 1);
        assert_eq!(s.irreducible_point_error_rate(), Some(0.5));
    }

    #[test]
    fn finite_set_covers_aliased_observed_responses() {
        let p = vec![pred('d', CandidateOutput::Set(vec![10, 20]))];
        let o = vec![oracle('d', "fa", "ca", 10), oracle('d', "fb", "cb", 20)];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.ambiguous_complete_set_groups, 1);
        assert_eq!(s.ambiguous_unsafe_groups, 0);
        assert_eq!(s.set_miss_cells, 0);
    }

    #[test]
    fn unknown_on_alias_and_identifiable_view_measures_over_abstention() {
        let p = vec![
            pred('d', CandidateOutput::Unknown(UnknownReason::AmbiguousContext)),
            pred('e', CandidateOutput::Unknown(UnknownReason::InsufficientSupport)),
        ];
        let o = vec![
            oracle('d', "fa", "ca", 10), oracle('d', "fb", "cb", 20),
            oracle('e', "fc", "cc", 30),
        ];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.ambiguity_abstention_rate(), Some(1.0));
        assert_eq!(s.unnecessary_abstention_rate(), Some(1.0));
    }

    #[test]
    fn benign_family_shift_remains_identifiable() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![
            oracle('d', "train", "train-cluster", 10),
            oracle('d', "heldout", "heldout-cluster", 10),
        ];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.identifiable_view_groups, 1);
        assert_eq!(s.environment_families, 2);
        assert_eq!(s.independent_clusters, 2);
        assert_eq!(s.point_error_cells, 0);
    }

    #[test]
    fn informative_probe_changes_view_and_separates_responses() {
        let p = vec![pred('d', CandidateOutput::Point(10)), pred('e', CandidateOutput::Point(20))];
        let o = vec![oracle('d', "fa", "ca", 10), oracle('e', "fb", "cb", 20)];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.unidentifiable_view_groups, 0);
        assert_eq!(s.point_error_cells, 0);
    }

    #[test]
    fn uninformative_probe_does_not_split_alias() {
        let p = vec![pred('d', CandidateOutput::Unknown(UnknownReason::AmbiguousContext))];
        let o = vec![oracle('d', "fa", "ca", 10), oracle('d', "fb", "cb", 20)];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.unidentifiable_view_groups, 1);
        assert_eq!(s.ambiguous_unknown_cells, 2);
    }

    #[test]
    fn repeated_annotations_do_not_inflate_cluster_count() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![
            oracle('d', "fa", "ca", 10), oracle('d', "fa", "ca", 10),
            oracle('d', "fb", "cb", 20),
        ];
        let s = score_identifiability(&p, &o, &custody()).unwrap();
        assert_eq!(s.independent_clusters, 2);
        assert_eq!(s.view_cluster_cells, 2);
        assert_eq!(s.irreducible_point_error_floor, 1);
    }

    #[test]
    fn cluster_reuse_across_families_is_rejected() {
        let p = vec![pred('d', CandidateOutput::Point(10)), pred('e', CandidateOutput::Point(20))];
        let o = vec![oracle('d', "fa", "same", 10), oracle('e', "fb", "same", 20)];
        assert_eq!(score_identifiability(&p, &o, &custody()), Err(ValidationError::ClusterAliasedAcrossFamilies));
    }

    #[test]
    fn duplicate_candidate_view_must_have_identical_output() {
        let p = vec![pred('d', CandidateOutput::Point(10)), pred('d', CandidateOutput::Point(20))];
        let o = vec![oracle('d', "fa", "ca", 10)];
        assert_eq!(score_identifiability(&p, &o, &custody()), Err(ValidationError::CandidateViewOutputMismatch));
    }

    #[test]
    fn unmatched_prediction_and_oracle_views_are_rejected() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![oracle('e', "fa", "ca", 10)];
        assert_eq!(score_identifiability(&p, &o, &custody()), Err(ValidationError::UnmatchedOracleView));
        let p2 = vec![pred('d', CandidateOutput::Point(10)), pred('e', CandidateOutput::Point(20))];
        let o2 = vec![oracle('d', "fa", "ca", 10)];
        assert_eq!(score_identifiability(&p2, &o2, &custody()), Err(ValidationError::UnmatchedCandidateView));
    }

    #[test]
    fn conflicting_outcomes_inside_one_cluster_are_rejected() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![oracle('d', "fa", "ca", 10), oracle('d', "fa", "ca", 20)];
        assert_eq!(score_identifiability(&p, &o, &custody()), Err(ValidationError::ClusterResponseConflict));
    }

    #[test]
    fn malformed_custody_and_noncanonical_sets_are_rejected() {
        let p = vec![pred('d', CandidateOutput::Point(10))];
        let o = vec![oracle('d', "fa", "ca", 10)];
        let mut bad_custody = custody();
        bad_custody.contract_sha256 = "bad".into();
        assert_eq!(score_identifiability(&p, &o, &bad_custody), Err(ValidationError::InvalidSha256("contract_sha256")));
        let bad_set = vec![pred('d', CandidateOutput::Set(vec![20, 10]))];
        assert_eq!(score_identifiability(&bad_set, &o, &custody()), Err(ValidationError::NonCanonicalPredictionSet));
    }
}

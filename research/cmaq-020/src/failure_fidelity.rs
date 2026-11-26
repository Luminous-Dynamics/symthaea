use std::collections::{BTreeMap, BTreeSet};

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[derive(Clone, Debug)]
pub struct FailureFidelityContract {
    pub contract_id: String,
    pub contract_sha256: String,
    pub source_manifest_sha256: String,
    pub domain_ids: BTreeSet<String>,
    /// Number of registered execution stages; stage indices are zero-based.
    pub stage_count: u32,
}

impl FailureFidelityContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_id.trim().is_empty() {
            return Err("failure contract ID must be non-empty".to_owned());
        }
        if !is_sha256(&self.contract_sha256) || !is_sha256(&self.source_manifest_sha256) {
            return Err("failure contract and source manifest must use SHA-256 digests".to_owned());
        }
        if self.domain_ids.is_empty()
            || self.domain_ids.iter().any(|domain| domain.trim().is_empty())
        {
            return Err("failure contract must admit at least one non-empty domain".to_owned());
        }
        if self.stage_count == 0 {
            return Err("failure contract stage_count must be positive".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FailureSignature {
    pub reason_id: String,
    pub stage_id: String,
    pub stage_index: u32,
}

impl FailureSignature {
    fn validate(&self, stage_count: u32) -> Result<(), String> {
        if self.reason_id.trim().is_empty() || self.stage_id.trim().is_empty() {
            return Err("failure reason and stage IDs must be non-empty".to_owned());
        }
        if self.stage_index >= stage_count {
            return Err("failure stage index is outside the frozen stage contract".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OutcomeLabel {
    Success,
    Failure(FailureSignature),
}

impl OutcomeLabel {
    fn validate(&self, stage_count: u32) -> Result<(), String> {
        if let Self::Failure(signature) = self {
            signature.validate(stage_count)?;
        }
        Ok(())
    }

    fn is_failure(&self) -> bool {
        matches!(self, Self::Failure(_))
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum BoundarySide {
    LowerPerturbation,
    UpperPerturbation,
}

/// One execution/replay record. Reference labels are evaluator-owned and are
/// never provided to the candidate that generated predicted.
#[derive(Clone, Debug)]
pub struct FailureTraceRecord {
    pub record_id: String,
    pub domain_id: String,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub initial_state_sha256: String,
    pub realized_action_sha256: String,
    pub candidate_checkpoint_sha256: String,
    pub boundary_pair_id: Option<String>,
    pub boundary_side: Option<BoundarySide>,
    pub perturbation_units: Option<i64>,
    pub reference: OutcomeLabel,
    pub predicted: OutcomeLabel,
}

impl FailureTraceRecord {
    fn validate(&self, contract: &FailureFidelityContract) -> Result<(), String> {
        if self.record_id.trim().is_empty()
            || self.domain_id.trim().is_empty()
            || self.environment_family_id.trim().is_empty()
            || self.environment_cluster_id.trim().is_empty()
        {
            return Err("failure record, domain, family and cluster IDs must be non-empty".to_owned());
        }
        if !contract.domain_ids.contains(&self.domain_id) {
            return Err("failure record domain is outside the frozen contract".to_owned());
        }
        for (name, digest) in [
            ("initial_state_sha256", &self.initial_state_sha256),
            ("realized_action_sha256", &self.realized_action_sha256),
            ("candidate_checkpoint_sha256", &self.candidate_checkpoint_sha256),
        ] {
            if !is_sha256(digest) {
                return Err(format!("{name} must be a 64-character hexadecimal digest"));
            }
        }
        self.reference.validate(contract.stage_count)?;
        self.predicted.validate(contract.stage_count)?;
        match (&self.boundary_pair_id, self.boundary_side, self.perturbation_units) {
            (None, None, None) => {}
            (Some(pair_id), Some(_), Some(_)) if !pair_id.trim().is_empty() => {}
            _ => {
                return Err("boundary pair ID, side and perturbation value must be supplied together".to_owned());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FailureClusterCounts {
    pub cases: usize,
    pub reference_failures: usize,
    pub false_successes: usize,
    pub reference_successes: usize,
    pub false_failures: usize,
    pub reason_matches: usize,
    pub stage_matches: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct FailureFidelityScore {
    pub cases: usize,
    pub reference_failures: usize,
    pub false_successes: usize,
    pub failure_capture_recall: Option<f64>,
    pub false_success_rate: Option<f64>,
    pub reason_accuracy: Option<f64>,
    pub stage_accuracy: Option<f64>,
    pub reference_successes: usize,
    pub false_failures: usize,
    pub false_failure_rate: Option<f64>,
    pub boundary_pairs: usize,
    pub boundary_flip_accuracy: Option<f64>,
    pub boundary_pair_outcome_accuracy: Option<f64>,
    pub clusters: BTreeMap<String, FailureClusterCounts>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BoundaryPairCounts {
    flip_matches: bool,
    both_outcomes_match: bool,
}

/// Score failure fidelity independently of final task success.
///
/// False-success rate uses reference failures as the denominator. Reason and
/// stage accuracy also use all reference failures, so a predicted success on
/// a reference failure cannot silently disappear from those metrics.
///
/// Each boundary pair must share initial state, realized action, mechanism
/// family, independent cluster and candidate checkpoint. Cluster summaries
/// are descriptive; #8578 owns inferential resampling at the cluster level.
pub fn score_failure_fidelity(
    records: &[FailureTraceRecord],
    contract: &FailureFidelityContract,
) -> Result<FailureFidelityScore, String> {
    contract.validate()?;
    if records.is_empty() {
        return Err("at least one failure trace record is required".to_owned());
    }

    let mut record_ids = BTreeSet::new();
    let mut frozen_checkpoint: Option<&str> = None;
    let mut cluster_metadata: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    let mut boundary_groups: BTreeMap<&str, Vec<&FailureTraceRecord>> = BTreeMap::new();
    let mut score = FailureFidelityScore {
        cases: records.len(),
        ..FailureFidelityScore::default()
    };

    for record in records {
        record.validate(contract)?;
        if !record_ids.insert(record.record_id.as_str()) {
            return Err(format!("duplicate failure record ID: {}", record.record_id));
        }
        match frozen_checkpoint {
            Some(checkpoint) if checkpoint != record.candidate_checkpoint_sha256 => {
                return Err("one failure evaluation must use exactly one candidate checkpoint".to_owned());
            }
            None => frozen_checkpoint = Some(&record.candidate_checkpoint_sha256),
            _ => {}
        }

        let metadata = (record.domain_id.as_str(), record.environment_family_id.as_str());
        match cluster_metadata.get(record.environment_cluster_id.as_str()) {
            Some(previous) if *previous != metadata => {
                return Err("one failure cluster cannot alias multiple domain/family identities".to_owned());
            }
            None => {
                cluster_metadata.insert(record.environment_cluster_id.as_str(), metadata);
            }
            _ => {}
        }

        let cluster = score.clusters.entry(record.environment_cluster_id.clone()).or_default();
        cluster.cases += 1;
        match (&record.reference, &record.predicted) {
            (OutcomeLabel::Failure(reference), OutcomeLabel::Failure(predicted)) => {
                score.reference_failures += 1;
                cluster.reference_failures += 1;
                if reference.reason_id == predicted.reason_id {
                    cluster.reason_matches += 1;
                }
                if reference.stage_id == predicted.stage_id
                    && reference.stage_index == predicted.stage_index
                {
                    cluster.stage_matches += 1;
                }
            }
            (OutcomeLabel::Failure(_), OutcomeLabel::Success) => {
                score.reference_failures += 1;
                score.false_successes += 1;
                cluster.reference_failures += 1;
                cluster.false_successes += 1;
            }
            (OutcomeLabel::Success, OutcomeLabel::Failure(_)) => {
                score.reference_successes += 1;
                score.false_failures += 1;
                cluster.reference_successes += 1;
                cluster.false_failures += 1;
            }
            (OutcomeLabel::Success, OutcomeLabel::Success) => {
                score.reference_successes += 1;
                cluster.reference_successes += 1;
            }
        }
        if let Some(pair_id) = record.boundary_pair_id.as_deref() {
            boundary_groups.entry(pair_id).or_default().push(record);
        }
    }

    let mut boundary_counts = Vec::new();
    for (pair_id, pair) in boundary_groups {
        if pair.len() != 2 {
            return Err(format!("boundary pair {pair_id} must contain exactly two records"));
        }
        let lower = pair
            .iter()
            .find(|record| record.boundary_side == Some(BoundarySide::LowerPerturbation))
            .ok_or_else(|| format!("boundary pair {pair_id} is missing its lower side"))?;
        let upper = pair
            .iter()
            .find(|record| record.boundary_side == Some(BoundarySide::UpperPerturbation))
            .ok_or_else(|| format!("boundary pair {pair_id} is missing its upper side"))?;
        if lower.domain_id != upper.domain_id
            || lower.environment_family_id != upper.environment_family_id
            || lower.environment_cluster_id != upper.environment_cluster_id
            || lower.initial_state_sha256 != upper.initial_state_sha256
            || lower.realized_action_sha256 != upper.realized_action_sha256
            || lower.candidate_checkpoint_sha256 != upper.candidate_checkpoint_sha256
        {
            return Err(format!("boundary pair {pair_id} does not share its frozen initial state/action/family/cluster/checkpoint"));
        }
        if lower.perturbation_units >= upper.perturbation_units {
            return Err(format!("boundary pair {pair_id} must have strictly ordered perturbation values"));
        }
        let reference_flip = lower.reference.is_failure() != upper.reference.is_failure();
        let predicted_flip = lower.predicted.is_failure() != upper.predicted.is_failure();
        boundary_counts.push(BoundaryPairCounts {
            flip_matches: reference_flip == predicted_flip,
            both_outcomes_match: lower.reference == lower.predicted
                && upper.reference == upper.predicted,
        });
    }

    if score.reference_failures > 0 {
        let reason_matches: usize = score.clusters.values().map(|cluster| cluster.reason_matches).sum();
        let stage_matches: usize = score.clusters.values().map(|cluster| cluster.stage_matches).sum();
        score.failure_capture_recall =
            Some((score.reference_failures - score.false_successes) as f64 / score.reference_failures as f64);
        score.false_success_rate = Some(score.false_successes as f64 / score.reference_failures as f64);
        score.reason_accuracy = Some(reason_matches as f64 / score.reference_failures as f64);
        score.stage_accuracy = Some(stage_matches as f64 / score.reference_failures as f64);
    }
    if score.reference_successes > 0 {
        score.false_failure_rate = Some(score.false_failures as f64 / score.reference_successes as f64);
    }
    if !boundary_counts.is_empty() {
        score.boundary_pairs = boundary_counts.len();
        score.boundary_flip_accuracy = Some(
            boundary_counts.iter().filter(|pair| pair.flip_matches).count() as f64
                / boundary_counts.len() as f64,
        );
        score.boundary_pair_outcome_accuracy = Some(
            boundary_counts.iter().filter(|pair| pair.both_outcomes_match).count() as f64
                / boundary_counts.len() as f64,
        );
    }
    Ok(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> FailureFidelityContract {
        FailureFidelityContract {
            contract_id: "failure-contract-v1".to_owned(),
            contract_sha256: "a".repeat(64),
            source_manifest_sha256: "b".repeat(64),
            domain_ids: ["robot-manipulation-v1".to_owned()].into_iter().collect(),
            stage_count: 5,
        }
    }

    fn failure(reason: &str, stage: &str, stage_index: u32) -> OutcomeLabel {
        OutcomeLabel::Failure(FailureSignature {
            reason_id: reason.to_owned(),
            stage_id: stage.to_owned(),
            stage_index,
        })
    }

    fn record(
        record_id: &str,
        cluster: &str,
        reference: OutcomeLabel,
        predicted: OutcomeLabel,
    ) -> FailureTraceRecord {
        FailureTraceRecord {
            record_id: record_id.to_owned(),
            domain_id: "robot-manipulation-v1".to_owned(),
            environment_family_id: "contact-rich-pick-place".to_owned(),
            environment_cluster_id: cluster.to_owned(),
            initial_state_sha256: "c".repeat(64),
            realized_action_sha256: "d".repeat(64),
            candidate_checkpoint_sha256: "e".repeat(64),
            boundary_pair_id: None,
            boundary_side: None,
            perturbation_units: None,
            reference,
            predicted,
        }
    }

    #[test]
    fn false_success_is_counted_against_reference_failures() {
        let records = vec![
            record("case-fail", "cluster-1", failure("missed_contact", "grasp", 1), OutcomeLabel::Success),
            record("case-success", "cluster-2", OutcomeLabel::Success, OutcomeLabel::Success),
        ];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        assert_eq!(score.reference_failures, 1);
        assert_eq!(score.false_successes, 1);
        assert_eq!(score.failure_capture_recall, Some(0.0));
        assert_eq!(score.false_success_rate, Some(1.0));
        assert_eq!(score.reason_accuracy, Some(0.0));
        assert_eq!(score.stage_accuracy, Some(0.0));
        assert_eq!(score.false_failure_rate, Some(0.0));
    }

    #[test]
    fn reason_and_stage_are_scored_separately_from_failure_detection() {
        let records = vec![record(
            "wrong-reason-stage",
            "cluster-1",
            failure("slip_drop", "transport", 3),
            failure("collision", "grasp", 1),
        )];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        assert_eq!(score.failure_capture_recall, Some(1.0));
        assert_eq!(score.false_success_rate, Some(0.0));
        assert_eq!(score.reason_accuracy, Some(0.0));
        assert_eq!(score.stage_accuracy, Some(0.0));
    }

    #[test]
    fn boundary_pair_scores_flip_detection_and_each_side() {
        let mut lower = record("lower", "cluster-1", OutcomeLabel::Success, OutcomeLabel::Success);
        lower.initial_state_sha256 = "1".repeat(64);
        lower.realized_action_sha256 = "2".repeat(64);
        lower.boundary_pair_id = Some("boundary-1".to_owned());
        lower.boundary_side = Some(BoundarySide::LowerPerturbation);
        lower.perturbation_units = Some(10);
        let mut upper = record("upper", "cluster-1", failure("object_slip", "lift", 2), failure("object_slip", "lift", 2));
        upper.initial_state_sha256 = lower.initial_state_sha256.clone();
        upper.realized_action_sha256 = lower.realized_action_sha256.clone();
        upper.boundary_pair_id = Some("boundary-1".to_owned());
        upper.boundary_side = Some(BoundarySide::UpperPerturbation);
        upper.perturbation_units = Some(11);
        let score = score_failure_fidelity(&[lower, upper], &contract()).expect("valid boundary pair");
        assert_eq!(score.boundary_pairs, 1);
        assert_eq!(score.boundary_flip_accuracy, Some(1.0));
        assert_eq!(score.boundary_pair_outcome_accuracy, Some(1.0));
    }

    #[test]
    fn boundary_pair_does_not_hide_wrong_side_predictions() {
        let mut lower = record("lower", "cluster-1", OutcomeLabel::Success, failure("false", "contact", 0));
        lower.boundary_pair_id = Some("boundary-1".to_owned());
        lower.boundary_side = Some(BoundarySide::LowerPerturbation);
        lower.perturbation_units = Some(0);
        let mut upper = record("upper", "cluster-1", failure("slip", "lift", 2), OutcomeLabel::Success);
        upper.boundary_pair_id = Some("boundary-1".to_owned());
        upper.boundary_side = Some(BoundarySide::UpperPerturbation);
        upper.perturbation_units = Some(1);
        let score = score_failure_fidelity(&[lower, upper], &contract()).expect("valid boundary pair");
        assert_eq!(score.boundary_flip_accuracy, Some(1.0));
        assert_eq!(score.boundary_pair_outcome_accuracy, Some(0.0));
    }

    #[test]
    fn incomplete_boundary_pairs_are_rejected() {
        let mut lower = record("lower", "cluster-1", OutcomeLabel::Success, OutcomeLabel::Success);
        lower.boundary_pair_id = Some("boundary-1".to_owned());
        lower.boundary_side = Some(BoundarySide::LowerPerturbation);
        lower.perturbation_units = Some(0);
        assert!(score_failure_fidelity(&[lower], &contract()).is_err());
    }

    #[test]
    fn boundary_pair_requires_shared_start_action_and_cluster() {
        let mut lower = record("lower", "cluster-1", OutcomeLabel::Success, OutcomeLabel::Success);
        lower.boundary_pair_id = Some("boundary-1".to_owned());
        lower.boundary_side = Some(BoundarySide::LowerPerturbation);
        lower.perturbation_units = Some(0);
        let mut upper = record("upper", "cluster-2", failure("slip", "lift", 2), failure("slip", "lift", 2));
        upper.boundary_pair_id = Some("boundary-1".to_owned());
        upper.boundary_side = Some(BoundarySide::UpperPerturbation);
        upper.perturbation_units = Some(1);
        assert!(score_failure_fidelity(&[lower, upper], &contract()).is_err());
    }

    #[test]
    fn mixed_candidate_checkpoints_and_invalid_digests_are_rejected() {
        let mut first = record("first", "cluster-1", OutcomeLabel::Success, OutcomeLabel::Success);
        let mut second = record("second", "cluster-2", OutcomeLabel::Success, OutcomeLabel::Success);
        second.candidate_checkpoint_sha256 = "invalid".to_owned();
        assert!(score_failure_fidelity(&[first.clone(), second], &contract()).is_err());
        first.realized_action_sha256 = "bad".to_owned();
        assert!(score_failure_fidelity(&[first], &contract()).is_err());
    }

    #[test]
    fn repeated_cases_are_reported_per_cluster_not_collapsed_to_independent_counts() {
        let records = vec![
            record("case-a", "same-cluster", failure("slip", "lift", 2), failure("slip", "lift", 2)),
            record("case-b", "same-cluster", failure("collision", "grasp", 1), OutcomeLabel::Success),
        ];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        let cluster = score.clusters.get("same-cluster").expect("cluster summary");
        assert_eq!(cluster.cases, 2);
        assert_eq!(cluster.reference_failures, 2);
        assert_eq!(cluster.false_successes, 1);
        assert_eq!(score.clusters.len(), 1);
        assert_eq!(score.false_success_rate, Some(0.5));
    }

    #[test]
    fn candidate_false_failure_is_kept_separate_from_false_success() {
        let records = vec![record(
            "false-failure",
            "cluster-1",
            OutcomeLabel::Success,
            failure("timeout", "transport", 4),
        )];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        assert_eq!(score.false_failures, 1);
        assert_eq!(score.false_failure_rate, Some(1.0));
        assert_eq!(score.false_successes, 0);
        assert_eq!(score.false_success_rate, None);
    }
}

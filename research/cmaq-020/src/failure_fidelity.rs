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
    /// Minimum success-rate separation (basis points) that counts as a detected boundary.
    pub minimum_boundary_gap_basis_points: u32,
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
        if self.minimum_boundary_gap_basis_points == 0
            || self.minimum_boundary_gap_basis_points > 10_000
        {
            return Err("minimum boundary gap must be between 1 and 10,000 basis points".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FailureSignature {
    /// None explicitly means the failure reason was indeterminate/unjudgable.
    pub reason_id: Option<String>,
    /// None explicitly means the decisive execution stage could not be judged.
    pub stage_id: Option<String>,
    pub stage_index: Option<u32>,
}

impl FailureSignature {
    fn validate(&self, stage_count: u32) -> Result<(), String> {
        if self.reason_id.as_ref().is_some_and(|reason| reason.trim().is_empty()) {
            return Err("known failure reason IDs must be non-empty".to_owned());
        }
        if self.stage_id.as_ref().is_some_and(|stage| stage.trim().is_empty()) {
            return Err("known failure stage IDs must be non-empty".to_owned());
        }
        match (&self.stage_id, self.stage_index) {
            (None, None) => {}
            (Some(_), Some(index)) if index < stage_count => {}
            (Some(_), Some(_)) => {
                return Err("failure stage index is outside the frozen stage contract".to_owned());
            }
            _ => {
                return Err("unknown stage ID and stage index must both be absent".to_owned());
            }
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
/// Binary action-fork boundary pairs must share initial state, realized action,
/// mechanism family, independent cluster and candidate checkpoint. Cluster
/// summaries are descriptive; #8578 owns inferential resampling.
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



/// Rate-based scene-boundary pair. The reference success-side and failure-side
/// are selected by the benchmark manifest. Both sides use one frozen reference
/// policy/checkpoint, but its closed-loop action traces may differ because the
/// scene variants differ. Candidate counts come from rollouts of one frozen
/// candidate checkpoint.
#[derive(Clone, Debug)]
pub struct BoundaryRatePair {
    pub pair_id: String,
    pub domain_id: String,
    pub environment_family_id: String,
    pub environment_cluster_id: String,
    pub perturbation_dimension_id: String,
    pub success_side_perturbation_units: i64,
    pub failure_side_perturbation_units: i64,
    /// Digest of the pair manifest and its source metadata.
    pub pair_source_manifest_sha256: String,
    /// Canonical scene snapshot digests for the two perturbed variants.
    pub success_side_scene_sha256: String,
    pub failure_side_scene_sha256: String,
    /// Reference policy configuration used for both sides of this pair.
    pub reference_policy_id: String,
    pub reference_policy_checkpoint_sha256: String,
    /// Optional source action traces; if absent, this is not an identical-action comparison.
    pub success_side_action_trace_sha256: Option<String>,
    pub failure_side_action_trace_sha256: Option<String>,
    /// Frozen world-model candidate being evaluated.
    pub candidate_checkpoint_sha256: String,
    pub reference_success_side_successes: u32,
    pub reference_success_side_trials: u32,
    pub reference_failure_side_successes: u32,
    pub reference_failure_side_trials: u32,
    pub predicted_success_side_successes: u32,
    pub predicted_success_side_trials: u32,
    pub predicted_failure_side_successes: u32,
    pub predicted_failure_side_trials: u32,
}

impl BoundaryRatePair {
    fn validate(&self, contract: &FailureFidelityContract) -> Result<(), String> {
        if self.pair_id.trim().is_empty()
            || self.domain_id.trim().is_empty()
            || self.environment_family_id.trim().is_empty()
            || self.environment_cluster_id.trim().is_empty()
            || self.perturbation_dimension_id.trim().is_empty()
        {
            return Err("boundary pair and domain/family/cluster/perturbation IDs must be non-empty".to_owned());
        }
        if !contract.domain_ids.contains(&self.domain_id) {
            return Err("boundary pair domain is outside the frozen contract".to_owned());
        }
        if self.success_side_perturbation_units == self.failure_side_perturbation_units {
            return Err("success-side and failure-side perturbations must differ".to_owned());
        }
        if self.reference_policy_id.trim().is_empty() {
            return Err("reference policy ID must be non-empty".to_owned());
        }
        for (name, digest) in [
            ("pair_source_manifest_sha256", &self.pair_source_manifest_sha256),
            ("success_side_scene_sha256", &self.success_side_scene_sha256),
            ("failure_side_scene_sha256", &self.failure_side_scene_sha256),
            ("reference_policy_checkpoint_sha256", &self.reference_policy_checkpoint_sha256),
            ("candidate_checkpoint_sha256", &self.candidate_checkpoint_sha256),
        ] {
            if !is_sha256(digest) {
                return Err(format!("{name} must be a 64-character hexadecimal digest"));
            }
        }
        if self.success_side_scene_sha256 == self.failure_side_scene_sha256 {
            return Err("success-side and failure-side scene digests must differ".to_owned());
        }
        for (name, digest) in [
            ("success_side_action_trace_sha256", &self.success_side_action_trace_sha256),
            ("failure_side_action_trace_sha256", &self.failure_side_action_trace_sha256),
        ] {
            if digest.as_ref().is_some_and(|value| !is_sha256(value)) {
                return Err(format!("{name} must be a 64-character hexadecimal digest when supplied"));
            }
        }
        for (label, successes, trials) in [
            ("reference success side", self.reference_success_side_successes, self.reference_success_side_trials),
            ("reference failure side", self.reference_failure_side_successes, self.reference_failure_side_trials),
            ("predicted success side", self.predicted_success_side_successes, self.predicted_success_side_trials),
            ("predicted failure side", self.predicted_failure_side_successes, self.predicted_failure_side_trials),
        ] {
            if trials == 0 {
                return Err(format!("{label} must contain at least one rollout"));
            }
            if successes > trials {
                return Err(format!("{label} successes cannot exceed trials"));
            }
        }
        Ok(())
    }

    fn rate_basis_points(successes: u32, trials: u32) -> f64 {
        successes as f64 * 10_000.0 / trials as f64
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BoundaryRateClusterCounts {
    pub pairs: usize,
    pub detected_reference_boundaries: usize,
    pub detected_predicted_boundaries: usize,
    pub correctly_detected_boundaries: usize,
    pub direction_matches: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundaryRatePairMeasurement {
    pub pair_id: String,
    pub environment_cluster_id: String,
    pub pair_source_manifest_sha256: String,
    pub success_side_scene_sha256: String,
    pub failure_side_scene_sha256: String,
    pub reference_policy_id: String,
    pub reference_policy_checkpoint_sha256: String,
    pub success_side_action_trace_sha256: Option<String>,
    pub failure_side_action_trace_sha256: Option<String>,
    pub candidate_checkpoint_sha256: String,
    pub reference_success_side_rate_basis_points: f64,
    pub predicted_success_side_rate_basis_points: f64,
    pub reference_failure_side_rate_basis_points: f64,
    pub predicted_failure_side_rate_basis_points: f64,
    pub reference_gap_basis_points: f64,
    pub predicted_gap_basis_points: f64,
    pub absolute_gap_error_basis_points: f64,
    pub reference_boundary_detected: bool,
    pub predicted_boundary_detected: bool,
    pub boundary_detection_correct: bool,
    pub boundary_direction_correct: bool,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoundaryRateFidelityScore {
    pub pair_measurements: Vec<BoundaryRatePairMeasurement>,
    pub pairs: usize,
    pub detected_reference_boundaries: usize,
    pub detected_predicted_boundaries: usize,
    pub correct_boundary_detection_pairs: usize,
    pub boundary_detection_accuracy: Option<f64>,
    pub boundary_direction_accuracy: Option<f64>,
    /// Mean absolute rate error, measured in basis points (100 bp = 1 pp).
    pub success_side_rate_mae_basis_points: Option<f64>,
    pub failure_side_rate_mae_basis_points: Option<f64>,
    pub boundary_gap_mae_basis_points: Option<f64>,
    pub reference_mean_gap_basis_points: Option<f64>,
    pub predicted_mean_gap_basis_points: Option<f64>,
    pub clusters: BTreeMap<String, BoundaryRateClusterCounts>,
}

/// Score replicated success/failure boundary pairs without collapsing
/// rollouts to one binary label per side.
///
/// The minimum detectable boundary gap is frozen in the contract. The
/// benchmark's success-side and failure-side labels are retained; the
/// perturbation coordinate can increase or decrease depending on the factor.
/// This score is separate from binary boundary-flip scoring above.
pub fn score_boundary_rate_pairs(
    pairs: &[BoundaryRatePair],
    contract: &FailureFidelityContract,
) -> Result<BoundaryRateFidelityScore, String> {
    contract.validate()?;
    if pairs.is_empty() {
        return Err("at least one rate-based boundary pair is required".to_owned());
    }

    let mut pair_ids = BTreeSet::new();
    let mut frozen_checkpoint: Option<&str> = None;
    let mut frozen_reference_policy: Option<(&str, &str)> = None;
    let mut cluster_metadata: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    let threshold = contract.minimum_boundary_gap_basis_points as f64;
    let mut score = BoundaryRateFidelityScore {
        pairs: pairs.len(),
        ..BoundaryRateFidelityScore::default()
    };
    let mut success_error_sum = 0.0;
    let mut failure_error_sum = 0.0;
    let mut gap_error_sum = 0.0;
    let mut reference_gap_sum = 0.0;
    let mut predicted_gap_sum = 0.0;
    let mut direction_matches = 0_usize;

    for pair in pairs {
        pair.validate(contract)?;
        if !pair_ids.insert(pair.pair_id.as_str()) {
            return Err(format!("duplicate boundary pair ID: {}", pair.pair_id));
        }
        match frozen_checkpoint {
            Some(checkpoint) if checkpoint != pair.candidate_checkpoint_sha256 => {
                return Err("one boundary-rate evaluation must use exactly one frozen candidate checkpoint".to_owned());
            }
            None => frozen_checkpoint = Some(&pair.candidate_checkpoint_sha256),
            _ => {}
        }
        match frozen_reference_policy {
            Some((policy_id, checkpoint))
                if policy_id != pair.reference_policy_id
                    || checkpoint != pair.reference_policy_checkpoint_sha256 =>
            {
                return Err("one boundary-rate evaluation must use exactly one reference policy/checkpoint; split policy configurations into separate evaluations".to_owned());
            }
            None => {
                frozen_reference_policy = Some((
                    &pair.reference_policy_id,
                    &pair.reference_policy_checkpoint_sha256,
                ));
            }
            _ => {}
        }

        let metadata = (pair.domain_id.as_str(), pair.environment_family_id.as_str());
        match cluster_metadata.get(pair.environment_cluster_id.as_str()) {
            Some(previous) if *previous != metadata => {
                return Err("one boundary cluster cannot alias multiple domain/family identities".to_owned());
            }
            None => {
                cluster_metadata.insert(pair.environment_cluster_id.as_str(), metadata);
            }
            _ => {}
        }

        let reference_success_rate = BoundaryRatePair::rate_basis_points(
            pair.reference_success_side_successes,
            pair.reference_success_side_trials,
        );
        let reference_failure_rate = BoundaryRatePair::rate_basis_points(
            pair.reference_failure_side_successes,
            pair.reference_failure_side_trials,
        );
        let predicted_success_rate = BoundaryRatePair::rate_basis_points(
            pair.predicted_success_side_successes,
            pair.predicted_success_side_trials,
        );
        let predicted_failure_rate = BoundaryRatePair::rate_basis_points(
            pair.predicted_failure_side_successes,
            pair.predicted_failure_side_trials,
        );

        let reference_gap = reference_success_rate - reference_failure_rate;
        let predicted_gap = predicted_success_rate - predicted_failure_rate;
        let reference_detected = reference_gap >= threshold;
        let predicted_detected = predicted_gap >= threshold;
        let detection_correct = reference_detected == predicted_detected;
        let direction_match = (reference_gap > 0.0) == (predicted_gap > 0.0);

        score.detected_reference_boundaries += reference_detected as usize;
        score.detected_predicted_boundaries += predicted_detected as usize;
        score.correct_boundary_detection_pairs += detection_correct as usize;
        direction_matches += direction_match as usize;
        let success_side_error = (predicted_success_rate - reference_success_rate).abs();
        let failure_side_error = (predicted_failure_rate - reference_failure_rate).abs();
        let gap_error = (predicted_gap - reference_gap).abs();
        success_error_sum += success_side_error;
        failure_error_sum += failure_side_error;
        gap_error_sum += gap_error;
        reference_gap_sum += reference_gap;
        predicted_gap_sum += predicted_gap;
        score.pair_measurements.push(BoundaryRatePairMeasurement {
            pair_id: pair.pair_id.clone(),
            environment_cluster_id: pair.environment_cluster_id.clone(),
            pair_source_manifest_sha256: pair.pair_source_manifest_sha256.clone(),
            success_side_scene_sha256: pair.success_side_scene_sha256.clone(),
            failure_side_scene_sha256: pair.failure_side_scene_sha256.clone(),
            reference_policy_id: pair.reference_policy_id.clone(),
            reference_policy_checkpoint_sha256: pair.reference_policy_checkpoint_sha256.clone(),
            success_side_action_trace_sha256: pair.success_side_action_trace_sha256.clone(),
            failure_side_action_trace_sha256: pair.failure_side_action_trace_sha256.clone(),
            candidate_checkpoint_sha256: pair.candidate_checkpoint_sha256.clone(),
            reference_success_side_rate_basis_points: reference_success_rate,
            predicted_success_side_rate_basis_points: predicted_success_rate,
            reference_failure_side_rate_basis_points: reference_failure_rate,
            predicted_failure_side_rate_basis_points: predicted_failure_rate,
            reference_gap_basis_points: reference_gap,
            predicted_gap_basis_points: predicted_gap,
            absolute_gap_error_basis_points: gap_error,
            reference_boundary_detected: reference_detected,
            predicted_boundary_detected: predicted_detected,
            boundary_detection_correct: detection_correct,
            boundary_direction_correct: direction_match,
        });

        let cluster = score
            .clusters
            .entry(pair.environment_cluster_id.clone())
            .or_default();
        cluster.pairs += 1;
        cluster.detected_reference_boundaries += reference_detected as usize;
        cluster.detected_predicted_boundaries += predicted_detected as usize;
        cluster.correctly_detected_boundaries += detection_correct as usize;
        cluster.direction_matches += direction_match as usize;
    }

    let denominator = pairs.len() as f64;
    score.boundary_detection_accuracy =
        Some(score.correct_boundary_detection_pairs as f64 / denominator);
    score.boundary_direction_accuracy = Some(direction_matches as f64 / denominator);
    score.success_side_rate_mae_basis_points = Some(success_error_sum / denominator);
    score.failure_side_rate_mae_basis_points = Some(failure_error_sum / denominator);
    score.boundary_gap_mae_basis_points = Some(gap_error_sum / denominator);
    score.reference_mean_gap_basis_points = Some(reference_gap_sum / denominator);
    score.predicted_mean_gap_basis_points = Some(predicted_gap_sum / denominator);
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
            minimum_boundary_gap_basis_points: 2_000,
        }
    }

    fn failure(reason: &str, stage: &str, stage_index: u32) -> OutcomeLabel {
        OutcomeLabel::Failure(FailureSignature {
            reason_id: Some(reason.to_owned()),
            stage_id: Some(stage.to_owned()),
            stage_index: Some(stage_index),
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



    fn indeterminate_failure() -> OutcomeLabel {
        OutcomeLabel::Failure(FailureSignature {
            reason_id: None,
            stage_id: None,
            stage_index: None,
        })
    }



    fn rate_pair(pair_id: &str, cluster: &str) -> BoundaryRatePair {
        BoundaryRatePair {
            pair_id: pair_id.to_owned(),
            domain_id: "robot-manipulation-v1".to_owned(),
            environment_family_id: "contact-rich-pick-place".to_owned(),
            environment_cluster_id: cluster.to_owned(),
            perturbation_dimension_id: "position".to_owned(),
            success_side_perturbation_units: 0,
            failure_side_perturbation_units: 2,
            pair_source_manifest_sha256: "b".repeat(64),
            success_side_scene_sha256: "c".repeat(64),
            failure_side_scene_sha256: "d".repeat(64),
            reference_policy_id: "pi05-data-50".to_owned(),
            reference_policy_checkpoint_sha256: "f".repeat(64),
            success_side_action_trace_sha256: None,
            failure_side_action_trace_sha256: None,
            candidate_checkpoint_sha256: "e".repeat(64),
            reference_success_side_successes: 10,
            reference_success_side_trials: 10,
            reference_failure_side_successes: 2,
            reference_failure_side_trials: 10,
            predicted_success_side_successes: 10,
            predicted_success_side_trials: 10,
            predicted_failure_side_successes: 2,
            predicted_failure_side_trials: 10,
        }
    }

    #[test]
    fn rate_boundary_pairs_score_per_side_rates_and_boundary_gap() {
        let pair = rate_pair("pair-1", "cluster-1");
        let score = score_boundary_rate_pairs(&[pair], &contract()).expect("valid rates");
        assert_eq!(score.pairs, 1);
        assert_eq!(score.pair_measurements.len(), 1);
        assert_eq!(score.pair_measurements[0].pair_id, "pair-1");
        assert_eq!(score.pair_measurements[0].environment_cluster_id, "cluster-1");
        assert_eq!(score.pair_measurements[0].pair_source_manifest_sha256, "b".repeat(64));
        assert_eq!(score.pair_measurements[0].candidate_checkpoint_sha256, "e".repeat(64));
        assert_eq!(score.detected_reference_boundaries, 1);
        assert_eq!(score.detected_predicted_boundaries, 1);
        assert_eq!(score.boundary_detection_accuracy, Some(1.0));
        assert_eq!(score.boundary_direction_accuracy, Some(1.0));
        assert_eq!(score.success_side_rate_mae_basis_points, Some(0.0));
        assert_eq!(score.failure_side_rate_mae_basis_points, Some(0.0));
        assert_eq!(score.boundary_gap_mae_basis_points, Some(0.0));
        assert_eq!(score.reference_mean_gap_basis_points, Some(8_000.0));
        assert_eq!(score.predicted_mean_gap_basis_points, Some(8_000.0));
    }

    #[test]
    fn rate_boundary_can_be_detected_but_have_wrong_success_rate_magnitude() {
        let mut pair = rate_pair("pair-1", "cluster-1");
        pair.predicted_success_side_successes = 8;
        pair.predicted_failure_side_successes = 4;
        let score = score_boundary_rate_pairs(&[pair], &contract()).expect("valid rates");
        assert_eq!(score.boundary_detection_accuracy, Some(1.0));
        assert_eq!(score.success_side_rate_mae_basis_points, Some(2_000.0));
        assert_eq!(score.failure_side_rate_mae_basis_points, Some(2_000.0));
        assert_eq!(score.boundary_gap_mae_basis_points, Some(4_000.0));
    }

    #[test]
    fn rate_boundary_detects_missing_or_reversed_failure_boundary() {
        let mut pair = rate_pair("pair-1", "cluster-1");
        pair.predicted_success_side_successes = 5;
        pair.predicted_failure_side_successes = 5;
        let score = score_boundary_rate_pairs(&[pair], &contract()).expect("valid rates");
        assert_eq!(score.detected_reference_boundaries, 1);
        assert_eq!(score.detected_predicted_boundaries, 0);
        assert_eq!(score.boundary_detection_accuracy, Some(0.0));
        assert_eq!(score.boundary_direction_accuracy, Some(0.0));
    }

    #[test]
    fn rate_boundary_requires_positive_rollout_counts_and_valid_success_counts() {
        let mut zero_trials = rate_pair("zero", "cluster-1");
        zero_trials.predicted_success_side_trials = 0;
        assert!(score_boundary_rate_pairs(&[zero_trials], &contract()).is_err());

        let mut too_many_successes = rate_pair("too-many", "cluster-2");
        too_many_successes.reference_failure_side_successes = 11;
        assert!(score_boundary_rate_pairs(&[too_many_successes], &contract()).is_err());
    }

    #[test]
    fn rate_boundary_requires_unique_pairs_valid_digests_and_one_checkpoint() {
        let pair = rate_pair("pair-1", "cluster-1");
        assert!(score_boundary_rate_pairs(&[pair.clone(), pair.clone()], &contract()).is_err());

        let mut invalid_digest = rate_pair("pair-2", "cluster-2");
        invalid_digest.pair_source_manifest_sha256 = "bad".to_owned();
        assert!(score_boundary_rate_pairs(&[invalid_digest], &contract()).is_err());

        let mut mixed_checkpoint = rate_pair("pair-3", "cluster-3");
        mixed_checkpoint.candidate_checkpoint_sha256 = "f".repeat(64);
        assert!(score_boundary_rate_pairs(&[rate_pair("pair-4", "cluster-4"), mixed_checkpoint], &contract()).is_err());

        let mut mixed_reference_policy = rate_pair("pair-5", "cluster-5");
        mixed_reference_policy.reference_policy_id = "fastwam".to_owned();
        assert!(score_boundary_rate_pairs(&[rate_pair("pair-6", "cluster-6"), mixed_reference_policy], &contract()).is_err());

        let mut bad_reference_policy_digest = rate_pair("pair-7", "cluster-7");
        bad_reference_policy_digest.reference_policy_checkpoint_sha256 = "invalid".to_owned();
        assert!(score_boundary_rate_pairs(&[bad_reference_policy_digest], &contract()).is_err());
    }



    #[test]
    fn rate_boundary_rejects_identical_side_scenes_and_invalid_pair_manifest() {
        let mut identical_scenes = rate_pair("same-scene", "cluster-1");
        identical_scenes.failure_side_scene_sha256 = identical_scenes.success_side_scene_sha256.clone();
        assert!(score_boundary_rate_pairs(&[identical_scenes], &contract()).is_err());

        let mut invalid_manifest = rate_pair("invalid-manifest", "cluster-2");
        invalid_manifest.pair_source_manifest_sha256 = "not-a-digest".to_owned();
        assert!(score_boundary_rate_pairs(&[invalid_manifest], &contract()).is_err());
    }

    #[test]
    fn rate_boundary_summaries_preserve_environment_clusters() {
        let pairs = vec![
            rate_pair("pair-1", "cluster-1"),
            rate_pair("pair-2", "cluster-1"),
            rate_pair("pair-3", "cluster-2"),
        ];
        let score = score_boundary_rate_pairs(&pairs, &contract()).expect("valid pairs");
        assert_eq!(score.clusters.len(), 2);
        assert_eq!(score.clusters["cluster-1"].pairs, 2);
        assert_eq!(score.clusters["cluster-2"].pairs, 1);
    }



    #[test]
    fn failure_contract_rejects_invalid_boundary_threshold() {
        let mut zero = contract();
        zero.minimum_boundary_gap_basis_points = 0;
        assert!(zero.validate().is_err());

        let mut above_one_hundred_percent = contract();
        above_one_hundred_percent.minimum_boundary_gap_basis_points = 10_001;
        assert!(above_one_hundred_percent.validate().is_err());
    }

    #[test]
    fn unjudgable_failure_labels_remain_explicit_and_are_not_forced_into_known_classes() {
        let records = vec![
            record(
                "reference-unjudgable",
                "cluster-1",
                indeterminate_failure(),
                indeterminate_failure(),
            ),
            record(
                "candidate-abstains-on-known-failure",
                "cluster-2",
                failure("missed_contact", "grasp", 1),
                indeterminate_failure(),
            ),
        ];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        assert_eq!(score.reference_failures, 2);
        assert_eq!(score.false_successes, 0);
        assert_eq!(score.failure_capture_recall, Some(1.0));
        assert_eq!(score.reason_accuracy, Some(0.5));
        assert_eq!(score.stage_accuracy, Some(0.5));
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
    fn reason_and_stage_errors_are_independent_dimensions() {
        let records = vec![
            record(
                "reason-correct-stage-wrong",
                "cluster-a",
                failure("slip", "transport", 3),
                failure("slip", "grasp", 1),
            ),
            record(
                "reason-wrong-stage-correct",
                "cluster-b",
                failure("collision", "grasp", 1),
                failure("slip", "grasp", 1),
            ),
        ];
        let score = score_failure_fidelity(&records, &contract()).expect("valid records");
        assert_eq!(score.failure_capture_recall, Some(1.0));
        assert_eq!(score.reason_accuracy, Some(0.5));
        assert_eq!(score.stage_accuracy, Some(0.5));
        assert_eq!(score.false_success_rate, Some(0.0));
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

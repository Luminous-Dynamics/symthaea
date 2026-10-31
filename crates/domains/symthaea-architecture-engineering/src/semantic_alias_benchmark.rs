//! Semantic-alias and verifier-corruption controls for direction discovery.
//!
//! The policy-visible challenge contains only proposal IDs, surface forms and descriptor
//! digests. Ground-truth semantic identities and utilities live in a separate adjudication
//! key and are never serialized into the policy-visible challenge.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const SCHEMA: &str = "symthaea.architecture-engineering.semantic-alias-benchmark";
const VERSION: &str = "0.1.0";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticAliasError {
    pub path: String,
    pub message: String,
}

impl SemanticAliasError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SemanticAliasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl Error for SemanticAliasError {}

/// What a search policy is allowed to see. No ground-truth identity or utility is present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyVisibleAliasProposal {
    pub proposal_id: String,
    pub surface_form: String,
    pub descriptor_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyVisibleAliasChallenge {
    pub schema: String,
    pub version: String,
    pub challenge_id: String,
    pub proposals: Vec<PolicyVisibleAliasProposal>,
}

impl PolicyVisibleAliasChallenge {
    pub fn validate(&self) -> Result<(), SemanticAliasError> {
        if self.schema != SCHEMA || self.version != VERSION {
            return Err(SemanticAliasError::new(
                "challenge.schema",
                "unsupported semantic-alias schema/version",
            ));
        }
        if self.challenge_id.trim().is_empty() {
            return Err(SemanticAliasError::new("challenge.challenge_id", "challenge ID must be non-empty"));
        }
        if self.proposals.len() < 2 {
            return Err(SemanticAliasError::new("challenge.proposals", "at least two proposals are required"));
        }
        let mut ids = BTreeSet::new();
        for (index, proposal) in self.proposals.iter().enumerate() {
            if proposal.proposal_id.trim().is_empty() || !ids.insert(proposal.proposal_id.as_str()) {
                return Err(SemanticAliasError::new(
                    format!("challenge.proposals[{index}].proposal_id"),
                    "proposal IDs must be non-empty and unique",
                ));
            }
            if proposal.surface_form.trim().is_empty() {
                return Err(SemanticAliasError::new(
                    format!("challenge.proposals[{index}].surface_form"),
                    "surface form must be non-empty",
                ));
            }
            validate_sha256(
                &proposal.descriptor_sha256,
                &format!("challenge.proposals[{index}].descriptor_sha256"),
            )?;
        }
        Ok(())
    }

    pub fn challenge_sha256(&self) -> Result<String, SemanticAliasError> {
        self.validate()?;
        digest_serializable(self)
    }
}

/// Evaluator-only adjudication data. Do not pass this object to the policy under test.
#[derive(Debug, Clone, PartialEq)]
pub struct AliasAdjudicationKey {
    challenge_sha256: String,
    truth_identity_by_proposal: BTreeMap<String, String>,
    utility_by_identity: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct AliasAdjudicationPayload<'a> {
    challenge_sha256: &'a str,
    truth_identity_by_proposal: &'a BTreeMap<String, String>,
    utility_by_identity: &'a BTreeMap<String, f64>,
}

impl AliasAdjudicationKey {
    fn adjudication_sha256(&self) -> Result<String, SemanticAliasError> {
        digest_serializable(&AliasAdjudicationPayload {
            challenge_sha256: &self.challenge_sha256,
            truth_identity_by_proposal: &self.truth_identity_by_proposal,
            utility_by_identity: &self.utility_by_identity,
        })
    }

    fn validate_for(
        &self,
        challenge: &PolicyVisibleAliasChallenge,
    ) -> Result<(), SemanticAliasError> {
        let expected_ids = challenge.proposals.iter()
            .map(|proposal| proposal.proposal_id.as_str())
            .collect::<BTreeSet<_>>();
        let actual_ids = self.truth_identity_by_proposal.keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected_ids != actual_ids {
            return Err(SemanticAliasError::new(
                "adjudication.truth_identity_by_proposal",
                "adjudication key must contain exactly one identity for every visible proposal",
            ));
        }
        if self.challenge_sha256 != challenge.challenge_sha256()? {
            return Err(SemanticAliasError::new(
                "adjudication.challenge_sha256",
                "adjudication key belongs to a different policy-visible challenge",
            ));
        }
        let identities = self.truth_identity_by_proposal.values()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if identities != self.utility_by_identity.keys().map(String::as_str).collect::<BTreeSet<_>>() {
            return Err(SemanticAliasError::new(
                "adjudication.utility_by_identity",
                "every and only ground-truth identities must have a utility",
            ));
        }
        for (identity, utility) in &self.utility_by_identity {
            if identity.trim().is_empty() || !utility.is_finite() || !(0.0..=1.0).contains(utility) {
                return Err(SemanticAliasError::new(
                    "adjudication.utility_by_identity",
                    "truth identities must be non-empty and utilities finite in [0, 1]",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum AliasVerifierDecision {
    Admitted { canonical_direction_id: String },
    Duplicate { canonical_direction_id: String },
    Rejected,
    Inconclusive,
}

/// Ordered verifier decisions, one per policy-visible proposal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasVerifierTrial {
    pub decisions: Vec<AliasVerifierDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "corruption", rename_all = "snake_case")]
pub enum AliasCorruption {
    /// Incorrectly collapses this genuinely new identity into an already registered direction.
    FalseMergeAsDuplicate {
        proposal_id: String,
        existing_canonical_direction_id: String,
    },
    /// Incorrectly admits an alias of an already seen identity as a new canonical direction.
    FalseNewAdmission {
        proposal_id: String,
        new_canonical_direction_id: String,
    },
    /// Incorrectly rejects the first proposal representing a genuinely new identity.
    FalseNoveltyRejection {
        proposal_id: String,
    },
    /// Replaces a decision with an inconclusive result to test fail-closed handling.
    Inconclusive {
        proposal_id: String,
    },
}

/// Apply explicitly seeded/adjudicated corruptions to a clean verifier trial.
/// The resulting trial remains policy-output-like; the evaluator key is never modified.
pub fn inject_alias_corruptions(
    challenge: &PolicyVisibleAliasChallenge,
    clean_trial: &AliasVerifierTrial,
    corruptions: &[AliasCorruption],
) -> Result<AliasVerifierTrial, SemanticAliasError> {
    challenge.validate()?;
    validate_trial_shape(challenge, clean_trial)?;
    let mut corrupted = clean_trial.clone();
    let index_by_id = challenge.proposals.iter().enumerate()
        .map(|(index, proposal)| (proposal.proposal_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let mut touched = BTreeSet::new();

    for (corruption_index, corruption) in corruptions.iter().enumerate() {
        let proposal_id = match corruption {
            AliasCorruption::FalseMergeAsDuplicate { proposal_id, .. }
            | AliasCorruption::FalseNewAdmission { proposal_id, .. }
            | AliasCorruption::FalseNoveltyRejection { proposal_id }
            | AliasCorruption::Inconclusive { proposal_id } => proposal_id,
        };
        if !touched.insert(proposal_id.as_str()) {
            return Err(SemanticAliasError::new(
                format!("corruptions[{corruption_index}].proposal_id"),
                "at most one corruption may be applied to each proposal in one injection pass",
            ));
        }
        let index = index_by_id.get(proposal_id.as_str()).copied().ok_or_else(|| {
            SemanticAliasError::new(
                format!("corruptions[{corruption_index}].proposal_id"),
                "corruption references an unknown proposal ID",
            )
        })?;
        corrupted.decisions[index] = match corruption {
            AliasCorruption::FalseMergeAsDuplicate { existing_canonical_direction_id, .. } => {
                AliasVerifierDecision::Duplicate {
                    canonical_direction_id: existing_canonical_direction_id.clone(),
                }
            }
            AliasCorruption::FalseNewAdmission { new_canonical_direction_id, .. } => {
                AliasVerifierDecision::Admitted {
                    canonical_direction_id: new_canonical_direction_id.clone(),
                }
            }
            AliasCorruption::FalseNoveltyRejection { .. } => AliasVerifierDecision::Rejected,
            AliasCorruption::Inconclusive { .. } => AliasVerifierDecision::Inconclusive,
        };
    }
    validate_decision_structure(challenge, &corrupted)?;
    Ok(corrupted)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasPairwiseConfusion {
    pub true_duplicate_pairs: u64,
    pub predicted_duplicate_pairs: u64,
    pub true_positive_pairs: u64,
    pub false_merge_pairs: u64,
    pub false_split_pairs: u64,
    pub true_negative_pairs: u64,
    pub duplicate_precision: Option<f64>,
    pub duplicate_recall: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticAliasBenchmarkReport {
    pub schema: String,
    pub version: String,
    pub challenge_id: String,
    pub challenge_sha256: String,
    pub trial_sha256: String,
    /// Digest of evaluator-only truth labels and utility values; never exposes them.
    pub adjudication_sha256: String,
    pub proposal_count: usize,
    pub admitted_decision_count: usize,
    pub duplicate_decision_count: usize,
    pub rejected_decision_count: usize,
    pub inconclusive_decision_count: usize,
    pub false_new_admission_count: usize,
    pub false_novelty_rejection_count: usize,
    pub pairwise: AliasPairwiseConfusion,
    pub allocation_budget: usize,
    pub canonical_slots_created: usize,
    pub distinct_semantic_directions_discovered: usize,
    pub total_distinct_semantic_directions: usize,
    pub oracle_utility: f64,
    pub achieved_utility: f64,
    pub discovery_regret: f64,
    pub receipt_sha256: String,
}

#[derive(Serialize)]
struct SemanticAliasReceiptPayload<'a> {
    schema: &'a str,
    version: &'a str,
    challenge_id: &'a str,
    challenge_sha256: &'a str,
    trial_sha256: &'a str,
    adjudication_sha256: &'a str,
    proposal_count: usize,
    admitted_decision_count: usize,
    duplicate_decision_count: usize,
    rejected_decision_count: usize,
    inconclusive_decision_count: usize,
    false_new_admission_count: usize,
    false_novelty_rejection_count: usize,
    pairwise: &'a AliasPairwiseConfusion,
    allocation_budget: usize,
    canonical_slots_created: usize,
    distinct_semantic_directions_discovered: usize,
    total_distinct_semantic_directions: usize,
    oracle_utility: f64,
    achieved_utility: f64,
    discovery_regret: f64,
}

pub fn evaluate_alias_trial(
    challenge: &PolicyVisibleAliasChallenge,
    adjudication: &AliasAdjudicationKey,
    trial: &AliasVerifierTrial,
    allocation_budget: usize,
) -> Result<SemanticAliasBenchmarkReport, SemanticAliasError> {
    challenge.validate()?;
    adjudication.validate_for(challenge)?;
    validate_trial_shape(challenge, trial)?;
    validate_decision_structure(challenge, trial)?;
    if allocation_budget == 0 {
        return Err(SemanticAliasError::new(
            "allocation_budget",
            "allocation budget must be positive",
        ));
    }

    let mut admitted_decision_count = 0usize;
    let mut duplicate_decision_count = 0usize;
    let mut rejected_decision_count = 0usize;
    let mut inconclusive_decision_count = 0usize;
    let mut false_new_admission_count = 0usize;
    let mut false_novelty_rejection_count = 0usize;
    let mut seen_truth_identities = BTreeSet::new();
    let mut canonical_owner = BTreeMap::<String, String>::new();
    let mut canonical_order = Vec::<(String, String)>::new();
    let mut predicted_ids = Vec::<Option<String>>::with_capacity(trial.decisions.len());

    for (index, (proposal, decision)) in challenge.proposals.iter()
        .zip(&trial.decisions).enumerate()
    {
        let truth_identity = adjudication.truth_identity_by_proposal
            .get(&proposal.proposal_id)
            .ok_or_else(|| SemanticAliasError::new(
                format!("adjudication.proposal[{}]", proposal.proposal_id),
                "truth identity missing",
            ))?;
        let was_seen = !seen_truth_identities.insert(truth_identity.clone());
        match decision {
            AliasVerifierDecision::Admitted { canonical_direction_id } => {
                admitted_decision_count += 1;
                if was_seen {
                    false_new_admission_count += 1;
                }
                if !canonical_owner.contains_key(canonical_direction_id) {
                    canonical_owner.insert(canonical_direction_id.clone(), truth_identity.clone());
                    canonical_order.push((canonical_direction_id.clone(), truth_identity.clone()));
                }
                predicted_ids.push(Some(canonical_direction_id.clone()));
            }
            AliasVerifierDecision::Duplicate { canonical_direction_id } => {
                duplicate_decision_count += 1;
                if !canonical_owner.contains_key(canonical_direction_id) {
                    return Err(SemanticAliasError::new(
                        format!("trial.decisions[{index}].canonical_direction_id"),
                        "duplicate decision must reference a direction admitted earlier in the trial",
                    ));
                }
                predicted_ids.push(Some(canonical_direction_id.clone()));
            }
            AliasVerifierDecision::Rejected => {
                rejected_decision_count += 1;
                if !was_seen {
                    false_novelty_rejection_count += 1;
                }
                predicted_ids.push(None);
            }
            AliasVerifierDecision::Inconclusive => {
                inconclusive_decision_count += 1;
                predicted_ids.push(None);
            }
        }
    }

    let mut true_duplicate_pairs = 0u64;
    let mut predicted_duplicate_pairs = 0u64;
    let mut true_positive_pairs = 0u64;
    let mut false_merge_pairs = 0u64;
    let mut false_split_pairs = 0u64;
    let mut true_negative_pairs = 0u64;

    for left in 0..challenge.proposals.len() {
        for right in (left + 1)..challenge.proposals.len() {
            let left_truth = adjudication.truth_identity_by_proposal
                .get(&challenge.proposals[left].proposal_id)
                .ok_or_else(|| SemanticAliasError::new("adjudication", "left truth identity missing"))?;
            let right_truth = adjudication.truth_identity_by_proposal
                .get(&challenge.proposals[right].proposal_id)
                .ok_or_else(|| SemanticAliasError::new("adjudication", "right truth identity missing"))?;
            let truth_same = left_truth == right_truth;
            let predicted_same = predicted_ids[left].is_some()
                && predicted_ids[right].is_some()
                && predicted_ids[left] == predicted_ids[right];
            match (truth_same, predicted_same) {
                (true, true) => {
                    true_duplicate_pairs = checked_add(true_duplicate_pairs, 1, "pairwise.true_duplicate_pairs")?;
                    predicted_duplicate_pairs = checked_add(predicted_duplicate_pairs, 1, "pairwise.predicted_duplicate_pairs")?;
                    true_positive_pairs = checked_add(true_positive_pairs, 1, "pairwise.true_positive_pairs")?;
                }
                (true, false) => {
                    true_duplicate_pairs = checked_add(true_duplicate_pairs, 1, "pairwise.true_duplicate_pairs")?;
                    false_split_pairs = checked_add(false_split_pairs, 1, "pairwise.false_split_pairs")?;
                }
                (false, true) => {
                    predicted_duplicate_pairs = checked_add(predicted_duplicate_pairs, 1, "pairwise.predicted_duplicate_pairs")?;
                    false_merge_pairs = checked_add(false_merge_pairs, 1, "pairwise.false_merge_pairs")?;
                }
                (false, false) => {
                    true_negative_pairs = checked_add(true_negative_pairs, 1, "pairwise.true_negative_pairs")?;
                }
            }
        }
    }

    let duplicate_precision = if predicted_duplicate_pairs == 0 {
        None
    } else {
        Some(true_positive_pairs as f64 / predicted_duplicate_pairs as f64)
    };
    let duplicate_recall = if true_duplicate_pairs == 0 {
        None
    } else {
        Some(true_positive_pairs as f64 / true_duplicate_pairs as f64)
    };
    let pairwise = AliasPairwiseConfusion {
        true_duplicate_pairs,
        predicted_duplicate_pairs,
        true_positive_pairs,
        false_merge_pairs,
        false_split_pairs,
        true_negative_pairs,
        duplicate_precision,
        duplicate_recall,
    };

    let total_distinct_semantic_directions = adjudication.utility_by_identity.len();
    let distinct_semantic_directions_discovered = canonical_owner.values()
        .collect::<BTreeSet<_>>().len();
    let mut oracle_utilities = adjudication.utility_by_identity.values().copied()
        .collect::<Vec<_>>();
    oracle_utilities.sort_by(|left, right| right.total_cmp(left));
    let oracle_utility = oracle_utilities.iter()
        .take(allocation_budget)
        .sum::<f64>();

    let selected_truths = canonical_order.iter()
        .take(allocation_budget)
        .map(|(_, truth)| truth.clone())
        .collect::<BTreeSet<_>>();
    let achieved_utility = selected_truths.iter()
        .filter_map(|truth| adjudication.utility_by_identity.get(truth))
        .sum::<f64>();
    let discovery_regret = (oracle_utility - achieved_utility).max(0.0);

    let challenge_sha256 = challenge.challenge_sha256()?;
    let trial_sha256 = digest_serializable(trial)?;
    let adjudication_sha256 = adjudication.adjudication_sha256()?;
    let receipt_sha256 = digest_serializable(&SemanticAliasReceiptPayload {
        schema: SCHEMA,
        version: VERSION,
        challenge_id: &challenge.challenge_id,
        challenge_sha256: &challenge_sha256,
        trial_sha256: &trial_sha256,
        adjudication_sha256: &adjudication_sha256,
        proposal_count: challenge.proposals.len(),
        admitted_decision_count,
        duplicate_decision_count,
        rejected_decision_count,
        inconclusive_decision_count,
        false_new_admission_count,
        false_novelty_rejection_count,
        pairwise: &pairwise,
        allocation_budget,
        canonical_slots_created: canonical_order.len(),
        distinct_semantic_directions_discovered,
        total_distinct_semantic_directions,
        oracle_utility,
        achieved_utility,
        discovery_regret,
    })?;

    Ok(SemanticAliasBenchmarkReport {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        challenge_id: challenge.challenge_id.clone(),
        challenge_sha256,
        trial_sha256,
        adjudication_sha256,
        proposal_count: challenge.proposals.len(),
        admitted_decision_count,
        duplicate_decision_count,
        rejected_decision_count,
        inconclusive_decision_count,
        false_new_admission_count,
        false_novelty_rejection_count,
        pairwise,
        allocation_budget,
        canonical_slots_created: canonical_order.len(),
        distinct_semantic_directions_discovered,
        total_distinct_semantic_directions,
        oracle_utility,
        achieved_utility,
        discovery_regret,
        receipt_sha256,
    })
}

fn validate_trial_shape(
    challenge: &PolicyVisibleAliasChallenge,
    trial: &AliasVerifierTrial,
) -> Result<(), SemanticAliasError> {
    if trial.decisions.len() != challenge.proposals.len() {
        return Err(SemanticAliasError::new(
            "trial.decisions",
            "exactly one verifier decision is required per policy-visible proposal",
        ));
    }
    Ok(())
}

fn validate_decision_structure(
    challenge: &PolicyVisibleAliasChallenge,
    trial: &AliasVerifierTrial,
) -> Result<(), SemanticAliasError> {
    let mut admitted = BTreeSet::new();
    for (index, decision) in trial.decisions.iter().enumerate() {
        match decision {
            AliasVerifierDecision::Admitted { canonical_direction_id } => {
                validate_identifier(canonical_direction_id, &format!("trial.decisions[{index}].canonical_direction_id"))?;
                admitted.insert(canonical_direction_id.as_str());
            }
            AliasVerifierDecision::Duplicate { canonical_direction_id } => {
                validate_identifier(canonical_direction_id, &format!("trial.decisions[{index}].canonical_direction_id"))?;
                if !admitted.contains(canonical_direction_id.as_str()) {
                    return Err(SemanticAliasError::new(
                        format!("trial.decisions[{index}].canonical_direction_id"),
                        "duplicate must point to a direction admitted by an earlier proposal",
                    ));
                }
            }
            AliasVerifierDecision::Rejected | AliasVerifierDecision::Inconclusive => {}
        }
    }
    let _ = challenge;
    Ok(())
}

/// Clean reference trial for the canonical alias fixture.
pub fn clean_alias_trial(challenge: &PolicyVisibleAliasChallenge) -> Result<AliasVerifierTrial, SemanticAliasError> {
    challenge.validate()?;
    let trial = AliasVerifierTrial {
        decisions: vec![
            AliasVerifierDecision::Admitted { canonical_direction_id: "dir-a".to_owned() },
            AliasVerifierDecision::Duplicate { canonical_direction_id: "dir-a".to_owned() },
            AliasVerifierDecision::Admitted { canonical_direction_id: "dir-b".to_owned() },
            AliasVerifierDecision::Duplicate { canonical_direction_id: "dir-b".to_owned() },
            AliasVerifierDecision::Admitted { canonical_direction_id: "dir-c".to_owned() },
        ],
    };
    validate_trial_shape(challenge, &trial)?;
    validate_decision_structure(challenge, &trial)?;
    Ok(trial)
}

/// Deterministic challenge with three latent directions and two syntactic aliases.
/// The return value separates policy-visible proposals from evaluator-only truth labels.
pub fn canonical_alias_fixture(
) -> Result<(PolicyVisibleAliasChallenge, AliasAdjudicationKey), SemanticAliasError> {
    let surface_forms = [
        ("p1", "coordinate descent with adaptive diagonal scaling", "identity-a"),
        ("p2", "adaptive diagonal scaling for coordinate descent", "identity-a"),
        ("p3", "low-discrepancy restart scheduling", "identity-b"),
        ("p4", "schedule low-discrepancy restarts", "identity-b"),
        ("p5", "surrogate-guided branch-and-bound", "identity-c"),
    ];
    let proposals = surface_forms.iter().map(|(proposal_id, surface_form, _)| {
        PolicyVisibleAliasProposal {
            proposal_id: (*proposal_id).to_owned(),
            surface_form: (*surface_form).to_owned(),
            descriptor_sha256: digest_bytes(surface_form.as_bytes()),
        }
    }).collect::<Vec<_>>();
    let challenge = PolicyVisibleAliasChallenge {
        schema: SCHEMA.to_owned(),
        version: VERSION.to_owned(),
        challenge_id: "semantic-alias-fixture-v0.1".to_owned(),
        proposals,
    };
    let truth_identity_by_proposal = surface_forms.iter()
        .map(|(proposal_id, _, truth_id)| ((*proposal_id).to_owned(), (*truth_id).to_owned()))
        .collect();
    let utility_by_identity = BTreeMap::from([
        ("identity-a".to_owned(), 0.95),
        ("identity-b".to_owned(), 0.80),
        ("identity-c".to_owned(), 0.70),
    ]);
    let adjudication = AliasAdjudicationKey {
        challenge_sha256: challenge.challenge_sha256()?,
        truth_identity_by_proposal,
        utility_by_identity,
    };
    adjudication.validate_for(&challenge)?;
    Ok((challenge, adjudication))
}

fn validate_identifier(value: &str, path: &str) -> Result<(), SemanticAliasError> {
    if value.trim().is_empty() {
        return Err(SemanticAliasError::new(path, "canonical direction ID must be non-empty"));
    }
    Ok(())
}

fn validate_sha256(value: &str, path: &str) -> Result<(), SemanticAliasError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) {
        return Err(SemanticAliasError::new(path, "expected lowercase 64-character SHA-256 digest"));
    }
    Ok(())
}

fn checked_add(left: u64, right: u64, path: &str) -> Result<u64, SemanticAliasError> {
    left.checked_add(right)
        .ok_or_else(|| SemanticAliasError::new(path, "pair-count aggregation overflowed"))
}

fn digest_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, SemanticAliasError> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        SemanticAliasError::new("serialization", format!("cannot serialize alias receipt: {error}"))
    })?;
    Ok(digest_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_verifier_has_perfect_duplicate_classification_and_zero_regret() {
        let (challenge, adjudication) = canonical_alias_fixture().unwrap();
        let trial = clean_alias_trial(&challenge).unwrap();
        let report = evaluate_alias_trial(&challenge, &adjudication, &trial, 2).unwrap();

        assert_eq!(report.pairwise.true_positive_pairs, 2);
        assert_eq!(report.pairwise.false_merge_pairs, 0);
        assert_eq!(report.pairwise.false_split_pairs, 0);
        assert_eq!(report.pairwise.duplicate_precision, Some(1.0));
        assert_eq!(report.pairwise.duplicate_recall, Some(1.0));
        assert_eq!(report.false_new_admission_count, 0);
        assert_eq!(report.false_novelty_rejection_count, 0);
        assert!((report.discovery_regret - 0.0).abs() < 1e-12);
    }

    #[test]
    fn false_merge_is_detected_and_has_downstream_discovery_regret() {
        let (challenge, adjudication) = canonical_alias_fixture().unwrap();
        let clean = clean_alias_trial(&challenge).unwrap();
        let trial = inject_alias_corruptions(
            &challenge,
            &clean,
            &[
                AliasCorruption::FalseMergeAsDuplicate {
                    proposal_id: "p3".to_owned(),
                    existing_canonical_direction_id: "dir-a".to_owned(),
                },
                AliasCorruption::FalseMergeAsDuplicate {
                    proposal_id: "p4".to_owned(),
                    existing_canonical_direction_id: "dir-a".to_owned(),
                },
            ],
        ).unwrap();
        let report = evaluate_alias_trial(&challenge, &adjudication, &trial, 2).unwrap();

        assert!(report.pairwise.false_merge_pairs > 0);
        assert!(report.discovery_regret > 0.0);
    }

    #[test]
    fn false_new_admission_becomes_a_false_split_and_wastes_budget() {
        let (challenge, adjudication) = canonical_alias_fixture().unwrap();
        let clean = clean_alias_trial(&challenge).unwrap();
        let trial = inject_alias_corruptions(
            &challenge,
            &clean,
            &[AliasCorruption::FalseNewAdmission {
                proposal_id: "p2".to_owned(),
                new_canonical_direction_id: "dir-a-alias".to_owned(),
            }],
        ).unwrap();
        let report = evaluate_alias_trial(&challenge, &adjudication, &trial, 2).unwrap();

        assert_eq!(report.false_new_admission_count, 1);
        assert_eq!(report.pairwise.false_split_pairs, 1);
        assert!(report.discovery_regret > 0.0);
    }

    #[test]
    fn false_novelty_rejection_is_detected() {
        let (challenge, adjudication) = canonical_alias_fixture().unwrap();
        let clean = clean_alias_trial(&challenge).unwrap();
        let trial = inject_alias_corruptions(
            &challenge,
            &clean,
            &[
                AliasCorruption::FalseNoveltyRejection {
                    proposal_id: "p3".to_owned(),
                },
                AliasCorruption::Inconclusive {
                    proposal_id: "p4".to_owned(),
                },
            ],
        ).unwrap();
        let report = evaluate_alias_trial(&challenge, &adjudication, &trial, 2).unwrap();

        assert_eq!(report.false_novelty_rejection_count, 1);
        assert!(report.discovery_regret > 0.0);
    }

    #[test]
    fn unknown_duplicate_targets_and_substituted_challenges_are_rejected() {
        let (challenge, adjudication) = canonical_alias_fixture().unwrap();
        let bad_trial = AliasVerifierTrial {
            decisions: vec![
                AliasVerifierDecision::Duplicate { canonical_direction_id: "never-admitted".to_owned() },
                AliasVerifierDecision::Duplicate { canonical_direction_id: "never-admitted".to_owned() },
                AliasVerifierDecision::Admitted { canonical_direction_id: "dir-b".to_owned() },
                AliasVerifierDecision::Duplicate { canonical_direction_id: "dir-b".to_owned() },
                AliasVerifierDecision::Admitted { canonical_direction_id: "dir-c".to_owned() },
            ],
        };
        assert!(evaluate_alias_trial(&challenge, &adjudication, &bad_trial, 2).is_err());

        let mut substituted = challenge.clone();
        substituted.challenge_id = "substituted".to_owned();
        assert!(evaluate_alias_trial(
            &substituted,
            &adjudication,
            &clean_alias_trial(&challenge).unwrap(),
            2
        ).is_err());
    }
}

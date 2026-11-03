#![forbid(unsafe_code)]

//! SPECTRA-001 T0 structural and descriptive-score qualifier.
//!
//! This crate validates recorded role separation, event ordering, candidate
//! identity and blinded-judge ranking structure. It computes an exact
//! descriptive rank score. It deliberately does not perform inferential
//! statistics, authenticate external receipts, hash external bytes, verify
//! randomization, detect real-world leakage, or establish an ESP mechanism.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const SUITE_SCHEMA_V0_1: &str = "symthaea.spectra.synthetic_suite.v0.1";
pub const CLAIM_CEILING_V0_1: &str = "synthetic_only";
pub const MAX_CANDIDATES: usize = 64;
pub const MAX_JUDGES: usize = 64;
pub const MAX_IDENTIFIER_BYTES: usize = 128;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundedTimeV1 {
    pub clock_domain_id: String,
    pub earliest_ns: u64,
    pub latest_ns: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct JudgeRankingV1 {
    pub judge_id: String,
    /// Candidate IDs in rank order, with the judge's best match first.
    pub ranked_candidate_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrialV1 {
    pub trial_id: String,
    pub participant_id: String,
    pub operator_id: String,
    pub target_custodian_id: String,
    pub evaluator_id: String,
    pub candidate_ids: Vec<String>,
    pub protocol_sha256: String,
    pub target_pool_sha256: String,
    pub target_selection_receipt_sha256: String,
    pub response_sha256: String,
    pub apparatus_locked_at: BoundedTimeV1,
    pub target_selected_at: BoundedTimeV1,
    pub response_sealed_at: BoundedTimeV1,
    pub target_revealed_at: BoundedTimeV1,
    pub judge_rankings: Vec<JudgeRankingV1>,
}

/// Evaluator-only truth. This type must never be passed to a participant-facing
/// policy, telemetry attacker, or blinded judge.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorTruthV1 {
    pub target_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticWorldV1 {
    Null,
    SignalInjected,
    KnownLeakageInjected,
    AdversarialMutation,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LeakageModalityV1 {
    Rf,
    LowFrequencyMagnetic,
    Acoustic,
    Optical,
    Vibration,
    ThermalAirflow,
    PowerConducted,
    NetworkTiming,
    Procedural,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExpectedScoreV1 {
    pub target_rank_sum: u64,
    pub judge_count: u64,
    pub candidate_count: u64,
    /// Positive means the true target is ranked better than the uniform-rank
    /// chance expectation. This is an exact descriptive numerator, not a p-value.
    pub twice_chance_minus_observed_rank_sum: i64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureScenarioV1 {
    pub scenario_id: String,
    pub world: SyntheticWorldV1,
    pub injected_leakage_modality: Option<LeakageModalityV1>,
    pub expected_acceptance: bool,
    pub expected_score: Option<ExpectedScoreV1>,
    pub trial: TrialV1,
    pub evaluator_truth: EvaluatorTruthV1,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct FixtureSuiteV1 {
    pub schema: String,
    pub claim_ceiling: String,
    pub scenarios: Vec<FixtureScenarioV1>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct TrialScoreV1 {
    pub target_rank_sum: u64,
    pub judge_count: u64,
    pub candidate_count: u64,
    pub twice_chance_minus_observed_rank_sum: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorV1 {
    InvalidIdentifier(&'static str),
    InvalidDigest(&'static str),
    TooFewCandidates,
    TooManyCandidates,
    DuplicateCandidate,
    TargetNotPresentExactlyOnce,
    TooFewJudges,
    TooManyJudges,
    DuplicateJudge,
    RoleCollision,
    InvalidRanking,
    InvalidTimeWindow(&'static str),
    ClockDomainMismatch,
    ChronologyUnproven,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixtureErrorV1 {
    Json(String),
    WrongSchema,
    WrongClaimCeiling,
    EmptySuite,
    DuplicateScenarioId,
    ScenarioExpectationMismatch(String),
    WorldLeakageAnnotationMismatch(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixtureVerificationV1 {
    pub checked_scenarios: usize,
    pub accepted_scenarios: usize,
    pub rejected_scenarios: usize,
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value.len() <= MAX_IDENTIFIER_BYTES
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || b"._:-".contains(&byte)
        })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_time_window(value: &BoundedTimeV1) -> bool {
    valid_id(&value.clock_domain_id)
        && value.earliest_ns <= value.latest_ns
}

fn validate_times(trial: &TrialV1, errors: &mut Vec<ValidationErrorV1>) {
    let windows = [
        ("apparatus_locked_at", &trial.apparatus_locked_at),
        ("target_selected_at", &trial.target_selected_at),
        ("response_sealed_at", &trial.response_sealed_at),
        ("target_revealed_at", &trial.target_revealed_at),
    ];

    for &(name, window) in &windows {
        if !valid_time_window(window) {
            errors.push(ValidationErrorV1::InvalidTimeWindow(name));
        }
    }

    if windows
        .iter()
        .any(|(_, window)| !valid_time_window(window))
    {
        return;
    }

    let domain = &trial.apparatus_locked_at.clock_domain_id;
    if windows.iter().any(|(_, window)| &window.clock_domain_id != domain) {
        errors.push(ValidationErrorV1::ClockDomainMismatch);
        return;
    }

    let ordered = trial.apparatus_locked_at.latest_ns
        < trial.target_selected_at.earliest_ns
        && trial.target_selected_at.latest_ns < trial.response_sealed_at.earliest_ns
        && trial.response_sealed_at.latest_ns < trial.target_revealed_at.earliest_ns;

    if !ordered {
        errors.push(ValidationErrorV1::ChronologyUnproven);
    }
}

/// Validate a single trial and return its exact descriptive target-rank score.
///
/// The function trusts neither a shield label nor a digest-shaped string as
/// proof of physical shielding or custody. It checks only the typed structure
/// and recorded chronology supplied to it.
pub fn evaluate_trial_v1(trial: &TrialV1, truth: &EvaluatorTruthV1) -> Result<TrialScoreV1, Vec<ValidationErrorV1>> {
    let mut errors = Vec::new();

    for (name, value) in [
        ("trial_id", trial.trial_id.as_str()),
        ("participant_id", trial.participant_id.as_str()),
        ("operator_id", trial.operator_id.as_str()),
        ("target_custodian_id", trial.target_custodian_id.as_str()),
        ("evaluator_id", trial.evaluator_id.as_str()),
        ] {
        if !valid_id(value) {
            errors.push(ValidationErrorV1::InvalidIdentifier(name));
        }
    }

    for (name, digest) in [
        ("protocol_sha256", trial.protocol_sha256.as_str()),
        ("target_pool_sha256", trial.target_pool_sha256.as_str()),
        (
            "target_selection_receipt_sha256",
            trial.target_selection_receipt_sha256.as_str(),
        ),
        ("response_sha256", trial.response_sha256.as_str()),
    ] {
        if !valid_sha256(digest) {
            errors.push(ValidationErrorV1::InvalidDigest(name));
        }
    }

    let mut role_ids = BTreeSet::new();
    for role in [
        trial.participant_id.as_str(),
        trial.operator_id.as_str(),
        trial.target_custodian_id.as_str(),
        trial.evaluator_id.as_str(),
    ] {
        if valid_id(role) && !role_ids.insert(role) {
            errors.push(ValidationErrorV1::RoleCollision);
        }
    }

    if !valid_id(&truth.target_id) {
        errors.push(ValidationErrorV1::InvalidIdentifier("evaluator_truth.target_id"));
    }

    let candidate_count = trial.candidate_ids.len();
    if candidate_count < 2 {
        errors.push(ValidationErrorV1::TooFewCandidates);
    }
    if candidate_count > MAX_CANDIDATES {
        errors.push(ValidationErrorV1::TooManyCandidates);
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in &trial.candidate_ids {
        if !valid_id(candidate) {
            errors.push(ValidationErrorV1::InvalidIdentifier("candidate_id"));
        }
        if !candidate_ids.insert(candidate.as_str()) {
            errors.push(ValidationErrorV1::DuplicateCandidate);
        }
    }

    if trial.candidate_ids.iter().filter(|id| *id == &truth.target_id).count() != 1 {
        errors.push(ValidationErrorV1::TargetNotPresentExactlyOnce);
    }

    let judge_count = trial.judge_rankings.len();
    if judge_count == 0 {
        errors.push(ValidationErrorV1::TooFewJudges);
    }
    if judge_count > MAX_JUDGES {
        errors.push(ValidationErrorV1::TooManyJudges);
    }

    let mut judge_ids = BTreeSet::new();
    let mut target_rank_sum = 0_u64;

    for ranking in &trial.judge_rankings {
        if !valid_id(&ranking.judge_id) {
            errors.push(ValidationErrorV1::InvalidIdentifier("judge_id"));
        }
        if !judge_ids.insert(ranking.judge_id.as_str()) {
            errors.push(ValidationErrorV1::DuplicateJudge);
        }
        if valid_id(&ranking.judge_id) && !role_ids.insert(ranking.judge_id.as_str()) {
            errors.push(ValidationErrorV1::RoleCollision);
        }

        let ranked: BTreeSet<&str> = ranking
            .ranked_candidate_ids
            .iter()
            .map(String::as_str)
            .collect();

        let has_duplicates = ranked.len() != ranking.ranked_candidate_ids.len();
        let is_permutation = !has_duplicates
            && ranking.ranked_candidate_ids.len() == candidate_count
            && ranked == candidate_ids;

        if !is_permutation {
            errors.push(ValidationErrorV1::InvalidRanking);
            continue;
        }

        if let Some(position) = ranking
            .ranked_candidate_ids
            .iter()
            .position(|candidate| candidate == &truth.target_id)
        {
            target_rank_sum += (position + 1) as u64;
        } else {
            errors.push(ValidationErrorV1::InvalidRanking);
        }
    }

    validate_times(trial, &mut errors);

    if !errors.is_empty() {
        return Err(errors);
    }

    let judge_count = judge_count as u64;
    let candidate_count = candidate_count as u64;
    let chance_twice = (candidate_count + 1) * judge_count;
    let observed_twice = 2 * target_rank_sum;

    Ok(TrialScoreV1 {
        target_rank_sum,
        judge_count,
        candidate_count,
        twice_chance_minus_observed_rank_sum: chance_twice as i64 - observed_twice as i64,
    })
}

/// Parse and verify a frozen synthetic known-answer suite.
///
/// Synthetic world labels are deliberately not passed into the scorer. This
/// makes it explicit that target rank alone cannot distinguish an injected
/// behavioural signal from an injected ordinary-channel leak.
pub fn verify_fixture_suite_json_v1(json: &str) -> Result<FixtureVerificationV1, FixtureErrorV1> {
    let suite: FixtureSuiteV1 =
        serde_json::from_str(json).map_err(|error| FixtureErrorV1::Json(error.to_string()))?;

    if suite.schema != SUITE_SCHEMA_V0_1 {
        return Err(FixtureErrorV1::WrongSchema);
    }
    if suite.claim_ceiling != CLAIM_CEILING_V0_1 {
        return Err(FixtureErrorV1::WrongClaimCeiling);
    }
    if suite.scenarios.is_empty() {
        return Err(FixtureErrorV1::EmptySuite);
    }

    let mut scenario_ids = BTreeSet::new();
    let mut accepted = 0_usize;
    let mut rejected = 0_usize;

    for scenario in &suite.scenarios {
        if !valid_id(&scenario.scenario_id) || !scenario_ids.insert(scenario.scenario_id.as_str()) {
            return Err(FixtureErrorV1::DuplicateScenarioId);
        }

        let leakage_annotation_valid = match &scenario.world {
            SyntheticWorldV1::KnownLeakageInjected => scenario.injected_leakage_modality.is_some(),
            _ => scenario.injected_leakage_modality.is_none(),
        };
        if !leakage_annotation_valid {
            return Err(FixtureErrorV1::WorldLeakageAnnotationMismatch(
                scenario.scenario_id.clone(),
            ));
        }

        match evaluate_trial_v1(&scenario.trial, &scenario.evaluator_truth) {
            Ok(score) => {
                if !scenario.expected_acceptance || scenario.expected_score != Some(score.into()) {
                    return Err(FixtureErrorV1::ScenarioExpectationMismatch(
                        scenario.scenario_id.clone(),
                    ));
                }
                accepted += 1;
            }
            Err(_) => {
                if scenario.expected_acceptance || scenario.expected_score.is_some() {
                    return Err(FixtureErrorV1::ScenarioExpectationMismatch(
                        scenario.scenario_id.clone(),
                    ));
                }
                rejected += 1;
            }
        }
    }

    Ok(FixtureVerificationV1 {
        checked_scenarios: suite.scenarios.len(),
        accepted_scenarios: accepted,
        rejected_scenarios: rejected,
    })
}

impl From<TrialScoreV1> for ExpectedScoreV1 {
    fn from(score: TrialScoreV1) -> Self {
        Self {
            target_rank_sum: score.target_rank_sum,
            judge_count: score.judge_count,
            candidate_count: score.candidate_count,
            twice_chance_minus_observed_rank_sum: score.twice_chance_minus_observed_rank_sum,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = include_str!("../fixtures/qualifier-v0.1.json");

    #[test]
    fn frozen_synthetic_suite_replays_exactly() {
        let report = verify_fixture_suite_json_v1(FIXTURE).expect("fixture suite should replay");
        assert_eq!(report.checked_scenarios, 5);
        assert_eq!(report.accepted_scenarios, 3);
        assert_eq!(report.rejected_scenarios, 2);
    }

    #[test]
    fn signal_and_known_leakage_have_identical_rank_scores() {
        let suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        let signal = suite
            .scenarios
            .iter()
            .find(|scenario| scenario.world == SyntheticWorldV1::SignalInjected)
            .expect("signal scenario");
        let leakage = suite
            .scenarios
            .iter()
            .find(|scenario| scenario.world == SyntheticWorldV1::KnownLeakageInjected)
            .expect("leakage scenario");

        let signal_score = evaluate_trial_v1(&signal.trial, &signal.evaluator_truth).expect("valid signal fixture");
        let leakage_score = evaluate_trial_v1(&leakage.trial, &leakage.evaluator_truth).expect("valid leakage fixture");

        assert_eq!(signal_score, leakage_score);
        assert_eq!(signal_score.twice_chance_minus_observed_rank_sum, 12);
        assert_ne!(signal.injected_leakage_modality, leakage.injected_leakage_modality);
    }

    #[test]
    fn malformed_digest_fails_closed() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.scenarios[0].trial.response_sha256 = "not-a-digest".to_owned();
        assert!(evaluate_trial_v1(&suite.scenarios[0].trial, &suite.scenarios[0].evaluator_truth).is_err());
    }

    #[test]
    fn different_clock_domains_cannot_prove_order() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.scenarios[0].trial.target_revealed_at.clock_domain_id = "unrelated-clock".to_owned();
        let errors = evaluate_trial_v1(&suite.scenarios[0].trial, &suite.scenarios[0].evaluator_truth).expect_err("clock mismatch");
        assert!(errors.contains(&ValidationErrorV1::ClockDomainMismatch));
    }

    #[test]
    fn role_collisions_are_rejected() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.scenarios[0].trial.operator_id =
            suite.scenarios[0].trial.target_custodian_id.clone();
        let errors = evaluate_trial_v1(&suite.scenarios[0].trial, &suite.scenarios[0].evaluator_truth).expect_err("role collision");
        assert!(errors.contains(&ValidationErrorV1::RoleCollision));
    }

    #[test]
    fn candidate_rankings_must_be_exact_permutations() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.scenarios[0].trial.judge_rankings[0]
            .ranked_candidate_ids
            .pop();
        let errors = evaluate_trial_v1(&suite.scenarios[0].trial, &suite.scenarios[0].evaluator_truth).expect_err("incomplete ranking");
        assert!(errors.contains(&ValidationErrorV1::InvalidRanking));
    }

    #[test]
    fn chronology_requires_strictly_separated_uncertainty_windows() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.scenarios[0].trial.target_revealed_at.earliest_ns =
            suite.scenarios[0].trial.response_sealed_at.latest_ns;
        let errors = evaluate_trial_v1(&suite.scenarios[0].trial, &suite.scenarios[0].evaluator_truth).expect_err("overlapping chronology");
        assert!(errors.contains(&ValidationErrorV1::ChronologyUnproven));
    }


    #[test]
    fn participant_facing_trial_has_no_target_label_field() {
        let suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        let scenario = &suite.scenarios[0];
        let public_trial = serde_json::to_value(&scenario.trial).expect("serialize public trial");
        assert!(public_trial.get("target_id").is_none());
        assert!(scenario.trial.candidate_ids.iter().all(|id| id.starts_with("candidate-")));
        assert!(scenario.evaluator_truth.target_id.starts_with("candidate-"));
    }

    #[test]
    fn wrong_claim_ceiling_is_rejected() {
        let mut suite: FixtureSuiteV1 = serde_json::from_str(FIXTURE).expect("valid fixture json");
        suite.claim_ceiling = "empirical_esp_detected".to_owned();
        let json = serde_json::to_string(&suite).expect("serialize suite");
        assert_eq!(
            verify_fixture_suite_json_v1(&json),
            Err(FixtureErrorV1::WrongClaimCeiling)
        );
    }
}

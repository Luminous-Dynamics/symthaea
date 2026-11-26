//! Evidence-gated Bayesian belief transitions for SDCB.
//!
//! Build from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_belief_transition.rs -o /tmp/sdcb-belief-transition
//! Run:
//!   /tmp/sdcb-belief-transition
//!
//! This dependency-free reference checks a narrow contract: a posterior update
//! is permitted only when a verified observation receipt is bound to the exact
//! run, task, system, hypothesis ordering, proposal, version, and budget.
//!
//! It does NOT verify signatures, hash preimages, physical measurement quality,
//! trusted custody, or the correctness of a scientific model. Those require
//! independently governed production components.

use std::collections::HashSet;

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const NUMERIC_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransitionError {
    InvalidIdentity,
    InvalidDigest,
    InvalidDistribution,
    InvalidLikelihoodMatrix,
    InvalidExperiment,
    DuplicateCandidateId,
    StaleProposal,
    IdentityMismatch,
    HypothesisOrderMismatch,
    ProposalMismatch,
    UnverifiedEvidence,
    InvalidVerifier,
    DuplicateExperiment,
    BudgetExceeded,
    CostMismatch,
    InvalidOutcome,
    ImpossibleObservation,
    VersionOverflow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentCandidate {
    pub experiment_id: String,
    pub cost_units: u64,
    /// Row i is P(outcome | hypothesis_i); columns are outcomes.
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentChoice {
    pub experiment_id: String,
    pub expected_information_gain_nats: f64,
    pub information_gain_per_cost: f64,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_distribution(values: &[f64]) -> Result<(), TransitionError> {
    if values.len() < 2
        || values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0)
    {
        return Err(TransitionError::InvalidDistribution);
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() || (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(TransitionError::InvalidDistribution);
    }
    Ok(())
}

fn validate_likelihoods(
    prior: &[f64],
    likelihoods: &[Vec<f64>],
) -> Result<usize, TransitionError> {
    validate_distribution(prior)?;
    if likelihoods.len() != prior.len() || likelihoods.is_empty() {
        return Err(TransitionError::InvalidLikelihoodMatrix);
    }
    let outcome_count = likelihoods[0].len();
    if outcome_count < 2 {
        return Err(TransitionError::InvalidLikelihoodMatrix);
    }
    for row in likelihoods {
        if row.len() != outcome_count {
            return Err(TransitionError::InvalidLikelihoodMatrix);
        }
        if row.iter().any(|value| !value.is_finite() || *value < 0.0 || *value > 1.0) {
            return Err(TransitionError::InvalidLikelihoodMatrix);
        }
        let total: f64 = row.iter().sum();
        if !total.is_finite() || (total - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(TransitionError::InvalidLikelihoodMatrix);
        }
    }
    Ok(outcome_count)
}

fn entropy_nats(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|probability| **probability > 0.0)
        .map(|probability| -probability * probability.ln())
        .sum()
}

/// Expected information gain I(H; O) for a finite hypothesis/outcome model.
pub fn expected_information_gain(
    prior: &[f64],
    likelihoods: &[Vec<f64>],
) -> Result<f64, TransitionError> {
    let outcome_count = validate_likelihoods(prior, likelihoods)?;
    let prior_entropy = entropy_nats(prior);
    let mut expected_posterior_entropy = 0.0;

    for outcome in 0..outcome_count {
        let predictive_probability: f64 = prior
            .iter()
            .enumerate()
            .map(|(hypothesis, probability)| probability * likelihoods[hypothesis][outcome])
            .sum();

        if predictive_probability <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = prior
            .iter()
            .enumerate()
            .map(|(hypothesis, probability)| {
                probability * likelihoods[hypothesis][outcome] / predictive_probability
            })
            .collect();
        expected_posterior_entropy += predictive_probability * entropy_nats(&posterior);
    }

    let gain = prior_entropy - expected_posterior_entropy;
    if !gain.is_finite() {
        return Err(TransitionError::InvalidLikelihoodMatrix);
    }
    if gain < -NUMERIC_TOLERANCE {
        return Err(TransitionError::InvalidLikelihoodMatrix);
    }
    Ok(gain.max(0.0))
}

/// Select the candidate with greatest expected information gain per cost unit.
/// Ties are resolved by lexicographically smaller experiment ID. If every
/// candidate has zero information gain, return None rather than inventing value.
pub fn choose_experiment(
    prior: &[f64],
    candidates: &[ExperimentCandidate],
) -> Result<Option<ExperimentChoice>, TransitionError> {
    validate_distribution(prior)?;
    let mut seen = HashSet::new();
    let mut best: Option<ExperimentChoice> = None;

    for candidate in candidates {
        if candidate.experiment_id.trim().is_empty() || candidate.cost_units == 0 {
            return Err(TransitionError::InvalidExperiment);
        }
        if !seen.insert(candidate.experiment_id.as_str()) {
            return Err(TransitionError::DuplicateCandidateId);
        }

        let gain = expected_information_gain(prior, &candidate.outcome_likelihoods)?;
        let score = gain / candidate.cost_units as f64;
        if !score.is_finite() {
            return Err(TransitionError::InvalidExperiment);
        }
        if gain <= NUMERIC_TOLERANCE {
            continue;
        }

        let choice = ExperimentChoice {
            experiment_id: candidate.experiment_id.clone(),
            expected_information_gain_nats: gain,
            information_gain_per_cost: score,
        };

        let replace = match &best {
            None => true,
            Some(current) if score > current.information_gain_per_cost + NUMERIC_TOLERANCE => true,
            Some(current)
                if (score - current.information_gain_per_cost).abs() <= NUMERIC_TOLERANCE =>
            {
                choice.experiment_id < current.experiment_id
            }
            _ => false,
        };
        if replace {
            best = Some(choice);
        }
    }
    Ok(best)
}

fn posterior_for_outcome(
    prior: &[f64],
    likelihoods: &[Vec<f64>],
    observed_outcome: usize,
) -> Result<(Vec<f64>, f64), TransitionError> {
    let outcome_count = validate_likelihoods(prior, likelihoods)?;
    if observed_outcome >= outcome_count {
        return Err(TransitionError::InvalidOutcome);
    }
    let predictive_probability: f64 = prior
        .iter()
        .enumerate()
        .map(|(hypothesis, probability)| {
            probability * likelihoods[hypothesis][observed_outcome]
        })
        .sum();

    if !predictive_probability.is_finite() || predictive_probability <= 0.0 {
        return Err(TransitionError::ImpossibleObservation);
    }

    let posterior: Vec<f64> = prior
        .iter()
        .enumerate()
        .map(|(hypothesis, probability)| {
            probability * likelihoods[hypothesis][observed_outcome] / predictive_probability
        })
        .collect();
    validate_distribution(&posterior)?;
    Ok((posterior, predictive_probability))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpisodeIdentity {
    pub run_id: String,
    pub task_id: String,
    pub task_digest: String,
    pub system_digest: String,
    pub hypothesis_set_digest: String,
}

impl EpisodeIdentity {
    fn validate(&self) -> Result<(), TransitionError> {
        if self.run_id.trim().is_empty() || self.task_id.trim().is_empty() {
            return Err(TransitionError::InvalidIdentity);
        }
        if !valid_digest(&self.task_digest)
            || !valid_digest(&self.system_digest)
            || !valid_digest(&self.hypothesis_set_digest)
        {
            return Err(TransitionError::InvalidDigest);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentProposal {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub prior_version: u64,
    pub hypothesis_ids: Vec<String>,
    pub cost_units: u64,
    pub outcome_likelihoods: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceReceipt {
    pub identity: EpisodeIdentity,
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub observed_outcome: usize,
    pub cost_units: u64,
    /// Test seam representing a decision already made by an independent verifier.
    /// A production caller must not populate this from untrusted agent output.
    pub independently_verified: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefTransition {
    pub experiment_id: String,
    pub proposal_digest: String,
    pub observation_digest: String,
    pub verifier_id: String,
    pub from_version: u64,
    pub to_version: u64,
    pub prior: Vec<f64>,
    pub posterior: Vec<f64>,
    pub predictive_probability: f64,
    pub cost_units: u64,
    pub budget_before: u64,
    pub budget_after: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BeliefLedger {
    identity: EpisodeIdentity,
    hypothesis_ids: Vec<String>,
    posterior: Vec<f64>,
    version: u64,
    budget_remaining: u64,
    used_experiments: HashSet<String>,
    accepted_transition_count: usize,
}

impl BeliefLedger {
    pub fn new(
        identity: EpisodeIdentity,
        hypothesis_ids: Vec<String>,
        prior: Vec<f64>,
        budget_units: u64,
    ) -> Result<Self, TransitionError> {
        identity.validate()?;
        validate_distribution(&prior)?;
        if hypothesis_ids.len() != prior.len() || hypothesis_ids.len() < 2 {
            return Err(TransitionError::InvalidIdentity);
        }
        let mut unique = HashSet::new();
        if hypothesis_ids
            .iter()
            .any(|hypothesis| hypothesis.trim().is_empty() || !unique.insert(hypothesis.as_str()))
        {
            return Err(TransitionError::InvalidIdentity);
        }
        Ok(Self {
            identity,
            hypothesis_ids,
            posterior: prior,
            version: 0,
            budget_remaining: budget_units,
            used_experiments: HashSet::new(),
            accepted_transition_count: 0,
        })
    }

    pub fn posterior(&self) -> &[f64] {
        &self.posterior
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn budget_remaining(&self) -> u64 {
        self.budget_remaining
    }

    pub fn accepted_transition_count(&self) -> usize {
        self.accepted_transition_count
    }

    /// Apply exactly one Bayesian update, only after all bindings and gates pass.
    /// Validation is completed before any ledger field is mutated.
    pub fn apply_verified(
        &mut self,
        proposal: &ExperimentProposal,
        receipt: &EvidenceReceipt,
    ) -> Result<BeliefTransition, TransitionError> {
        self.identity.validate()?;
        proposal.identity.validate()?;
        receipt.identity.validate()?;

        if proposal.identity != self.identity || receipt.identity != self.identity {
            return Err(TransitionError::IdentityMismatch);
        }
        if proposal.hypothesis_ids != self.hypothesis_ids {
            return Err(TransitionError::HypothesisOrderMismatch);
        }
        if proposal.experiment_id.trim().is_empty()
            || !valid_digest(&proposal.proposal_digest)
            || !valid_digest(&receipt.observation_digest)
        {
            return Err(TransitionError::InvalidDigest);
        }
        if receipt.experiment_id != proposal.experiment_id
            || receipt.proposal_digest != proposal.proposal_digest
        {
            return Err(TransitionError::ProposalMismatch);
        }
        if !receipt.independently_verified {
            return Err(TransitionError::UnverifiedEvidence);
        }
        if receipt.verifier_id.trim().is_empty() {
            return Err(TransitionError::InvalidVerifier);
        }
        if self.used_experiments.contains(&proposal.experiment_id) {
            return Err(TransitionError::DuplicateExperiment);
        }
        if proposal.prior_version != self.version {
            return Err(TransitionError::StaleProposal);
        }
        if proposal.cost_units == 0 || proposal.cost_units != receipt.cost_units {
            return Err(TransitionError::CostMismatch);
        }
        if proposal.cost_units > self.budget_remaining {
            return Err(TransitionError::BudgetExceeded);
        }

        let (next_posterior, predictive_probability) = posterior_for_outcome(
            &self.posterior,
            &proposal.outcome_likelihoods,
            receipt.observed_outcome,
        )?;
        let next_version = self
            .version
            .checked_add(1)
            .ok_or(TransitionError::VersionOverflow)?;
        let budget_after = self.budget_remaining - proposal.cost_units;

        let transition = BeliefTransition {
            experiment_id: proposal.experiment_id.clone(),
            proposal_digest: proposal.proposal_digest.clone(),
            observation_digest: receipt.observation_digest.clone(),
            verifier_id: receipt.verifier_id.clone(),
            from_version: self.version,
            to_version: next_version,
            prior: self.posterior.clone(),
            posterior: next_posterior.clone(),
            predictive_probability,
            cost_units: proposal.cost_units,
            budget_before: self.budget_remaining,
            budget_after,
        };

        self.posterior = next_posterior;
        self.version = next_version;
        self.budget_remaining = budget_after;
        self.used_experiments.insert(proposal.experiment_id.clone());
        self.accepted_transition_count += 1;

        Ok(transition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat(character).take(64).collect()
    }

    fn identity() -> EpisodeIdentity {
        EpisodeIdentity {
            run_id: "run-001".to_string(),
            task_id: "identifiability-pair-001".to_string(),
            task_digest: digest('a'),
            system_digest: digest('b'),
            hypothesis_set_digest: digest('c'),
        }
    }

    fn hypotheses() -> Vec<String> {
        vec!["mechanism-a".to_string(), "mechanism-b".to_string()]
    }

    fn ledger(budget: u64) -> BeliefLedger {
        BeliefLedger::new(identity(), hypotheses(), vec![0.5, 0.5], budget).unwrap()
    }

    fn proposal(ledger: &BeliefLedger) -> ExperimentProposal {
        ExperimentProposal {
            identity: identity(),
            experiment_id: "sensor-x".to_string(),
            proposal_digest: digest('d'),
            prior_version: ledger.version(),
            hypothesis_ids: hypotheses(),
            cost_units: 2,
            outcome_likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
        }
    }

    fn receipt(verified: bool) -> EvidenceReceipt {
        EvidenceReceipt {
            identity: identity(),
            experiment_id: "sensor-x".to_string(),
            proposal_digest: digest('d'),
            observation_digest: digest('e'),
            verifier_id: "independent-verifier-1".to_string(),
            observed_outcome: 0,
            cost_units: 2,
            independently_verified: verified,
        }
    }

    #[test]
    fn expected_information_gain_is_zero_for_uninformative_probe() {
        let gain = expected_information_gain(
            &[0.5, 0.5],
            &[vec![0.5, 0.5], vec![0.5, 0.5]],
        )
        .unwrap();
        assert!(gain.abs() <= NUMERIC_TOLERANCE);
    }

    #[test]
    fn discriminating_probe_has_positive_information_gain() {
        let gain = expected_information_gain(
            &[0.5, 0.5],
            &[vec![0.9, 0.1], vec![0.1, 0.9]],
        )
        .unwrap();
        assert!(gain > 0.0);
        assert!(gain < 2.0_f64.ln());
    }

    #[test]
    fn experiment_selector_uses_information_gain_per_cost() {
        let candidates = vec![
            ExperimentCandidate {
                experiment_id: "exp-expensive".to_string(),
                cost_units: 10,
                outcome_likelihoods: vec![vec![0.99, 0.01], vec![0.01, 0.99]],
            },
            ExperimentCandidate {
                experiment_id: "exp-efficient".to_string(),
                cost_units: 1,
                outcome_likelihoods: vec![vec![0.8, 0.2], vec![0.2, 0.8]],
            },
        ];
        let selected = choose_experiment(&[0.5, 0.5], &candidates)
            .unwrap()
            .unwrap();
        assert_eq!(selected.experiment_id, "exp-efficient");
    }

    #[test]
    fn experiment_selector_abstains_when_all_candidates_are_uninformative() {
        let candidates = vec![ExperimentCandidate {
            experiment_id: "exp-noop".to_string(),
            cost_units: 1,
            outcome_likelihoods: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        }];
        assert_eq!(choose_experiment(&[0.5, 0.5], &candidates).unwrap(), None);
    }

    #[test]
    fn experiment_selector_breaks_ties_deterministically() {
        let candidates = vec![
            ExperimentCandidate {
                experiment_id: "zeta".to_string(),
                cost_units: 1,
                outcome_likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            },
            ExperimentCandidate {
                experiment_id: "alpha".to_string(),
                cost_units: 1,
                outcome_likelihoods: vec![vec![0.9, 0.1], vec![0.1, 0.9]],
            },
        ];
        assert_eq!(
            choose_experiment(&[0.5, 0.5], &candidates)
                .unwrap()
                .unwrap()
                .experiment_id,
            "alpha"
        );
    }

    #[test]
    fn exact_receipt_binding_updates_posterior_once() {
        let mut state = ledger(5);
        let transition = state.apply_verified(&proposal(&state), &receipt(true)).unwrap();
        assert_eq!(transition.from_version, 0);
        assert_eq!(transition.to_version, 1);
        assert!((transition.posterior[0] - 0.9).abs() < 1e-12);
        assert!((transition.posterior[1] - 0.1).abs() < 1e-12);
        assert_eq!(state.version(), 1);
        assert_eq!(state.budget_remaining(), 3);
        assert_eq!(state.accepted_transition_count(), 1);
    }

    #[test]
    fn unverified_evidence_cannot_mutate_belief_or_budget() {
        let mut state = ledger(5);
        let before = state.clone();
        assert_eq!(
            state.apply_verified(&proposal(&state), &receipt(false)),
            Err(TransitionError::UnverifiedEvidence)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn task_or_system_mismatch_is_rejected_atomically() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.identity.system_digest = digest('f');
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::IdentityMismatch)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn wrong_proposal_binding_is_rejected_atomically() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.proposal_digest = digest('f');
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::ProposalMismatch)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn stale_proposal_version_is_rejected() {
        let mut state = ledger(5);
        let mut first = proposal(&state);
        first.experiment_id = "first".to_string();
        first.proposal_digest = digest('d');
        let mut first_receipt = receipt(true);
        first_receipt.experiment_id = "first".to_string();
        state.apply_verified(&first, &first_receipt).unwrap();

        let mut stale = proposal(&state);
        stale.experiment_id = "stale".to_string();
        stale.proposal_digest = digest('f');
        stale.prior_version = 0;
        let mut stale_receipt = receipt(true);
        stale_receipt.experiment_id = "stale".to_string();
        stale_receipt.proposal_digest = digest('f');

        let before = state.clone();
        assert_eq!(
            state.apply_verified(&stale, &stale_receipt),
            Err(TransitionError::StaleProposal)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn duplicate_experiment_is_rejected() {
        let mut state = ledger(5);
        let first = proposal(&state);
        let first_result = receipt(true);
        state.apply_verified(&first, &first_result).unwrap();
        let before = state.clone();
        let mut duplicate = proposal(&state);
        duplicate.prior_version = state.version();
        let duplicate_receipt = receipt(true);
        assert_eq!(
            state.apply_verified(&duplicate, &duplicate_receipt),
            Err(TransitionError::DuplicateExperiment)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn budget_overrun_is_rejected_without_partial_debit() {
        let mut state = ledger(1);
        let before = state.clone();
        assert_eq!(
            state.apply_verified(&proposal(&state), &receipt(true)),
            Err(TransitionError::BudgetExceeded)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn proposal_and_receipt_costs_must_match() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.cost_units = 1;
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::CostMismatch)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn hypothesis_order_must_match_the_frozen_set() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = proposal(&state);
        bad.hypothesis_ids.reverse();
        assert_eq!(
            state.apply_verified(&bad, &receipt(true)),
            Err(TransitionError::HypothesisOrderMismatch)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn impossible_observation_is_rejected() {
        let mut state = BeliefLedger::new(
            identity(),
            hypotheses(),
            vec![1.0, 0.0],
            5,
        )
        .unwrap();
        let mut plan = proposal(&state);
        plan.outcome_likelihoods = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let mut result = receipt(true);
        result.observed_outcome = 0;
        let before = state.clone();
        assert_eq!(
            state.apply_verified(&plan, &result),
            Err(TransitionError::ImpossibleObservation)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn invalid_probability_rows_are_rejected() {
        assert_eq!(
            expected_information_gain(
                &[0.5, 0.5],
                &[vec![0.7, 0.7], vec![0.5, 0.5]],
            ),
            Err(TransitionError::InvalidLikelihoodMatrix)
        );
    }

    #[test]
    fn malformed_digest_is_rejected_before_transition() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.observation_digest = "not-a-digest".to_string();
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::InvalidDigest)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn empty_verifier_identity_is_rejected() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.verifier_id = "  ".to_string();
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::InvalidVerifier)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn proposal_identity_must_match_ledger() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = proposal(&state);
        bad.identity.task_id = "other-task".to_string();
        assert_eq!(
            state.apply_verified(&bad, &receipt(true)),
            Err(TransitionError::IdentityMismatch)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn ledger_rejects_duplicate_hypothesis_ids() {
        let invalid = BeliefLedger::new(
            identity(),
            vec!["same".to_string(), "same".to_string()],
            vec![0.5, 0.5],
            5,
        );
        assert_eq!(invalid, Err(TransitionError::InvalidIdentity));
    }

    #[test]
    fn invalid_prior_is_rejected() {
        assert_eq!(
            BeliefLedger::new(
                identity(),
                hypotheses(),
                vec![0.6, 0.6],
                5,
            ),
            Err(TransitionError::InvalidDistribution)
        );
    }

    #[test]
    fn receipt_outcome_must_exist_in_proposal_model() {
        let mut state = ledger(5);
        let before = state.clone();
        let mut bad = receipt(true);
        bad.observed_outcome = 2;
        assert_eq!(
            state.apply_verified(&proposal(&state), &bad),
            Err(TransitionError::InvalidOutcome)
        );
        assert_eq!(state, before);
    }

    #[test]
    fn accepted_transition_records_predictive_probability_and_budget() {
        let mut state = ledger(7);
        let transition = state.apply_verified(&proposal(&state), &receipt(true)).unwrap();
        assert!((transition.predictive_probability - 0.5).abs() < 1e-12);
        assert_eq!(transition.budget_before, 7);
        assert_eq!(transition.budget_after, 5);
    }
}

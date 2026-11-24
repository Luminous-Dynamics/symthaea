//! Finite-world evaluator primitives for decision-relevant predictability.
//!
//! Candidate-visible episode types contain observations, allowed decisions,
//! probe permissions, resource budget and deadline. Hidden observation
//! likelihoods, state priors and task-loss tables live in EvaluatorWorld.
//!
//! This type boundary is useful for tests and interface design, but it is not
//! a process-security boundary. Confirmatory runs must keep EvaluatorWorld in
//! a separately permissioned evaluator process.

pub mod robust;

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const VALUE_TOLERANCE: f64 = 1.0e-9;
const MAX_STATES: usize = 256;
const MAX_DECISIONS: usize = 256;
const MAX_EVIDENCE: usize = 10_000;
const MAX_PROBES: usize = 256;
const MAX_OUTCOMES: usize = 10_000;

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEpisode {
    pub episode_id: String,
    pub current_step: u64,
    pub decision_deadline_step: u64,
    pub remaining_probe_budget: f64,
    pub maximum_worst_case_loss: f64,
    pub allowed_decision_ids: Vec<String>,
    pub evidence: Vec<ObservedSignal>,
    pub permitted_probes: Vec<PublicProbeOption>,
}

/// Candidate-visible observation. It contains no latent state, likelihood,
/// future label or evaluator loss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedSignal {
    pub channel_id: String,
    pub outcome_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PublicProbeOption {
    pub probe_id: String,
    pub acquisition_cost: f64,
    pub result_delay_steps: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateSubmission {
    pub decision_id: String,
    pub requested_probe_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluatorWorld {
    pub states: Vec<StateModel>,
    pub evidence_channels: Vec<EvidenceChannelModel>,
    pub probes: Vec<ProbeModel>,
    /// Evidence channels are multiplied only because this evaluator contract
    /// explicitly declares conditional independence given the hidden state.
    /// For correlated channels, provide one joint channel instead.
    pub evidence_channels_conditionally_independent: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StateModel {
    pub state_id: String,
    pub prior_probability: f64,
    /// Task loss in [0,1] for each permitted downstream decision.
    pub decision_loss: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvidenceChannelModel {
    pub channel_id: String,
    pub outcomes: Vec<OutcomeLikelihood>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeModel {
    pub probe_id: String,
    pub acquisition_cost: f64,
    pub result_delay_steps: u64,
    pub outcomes: Vec<OutcomeLikelihood>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeLikelihood {
    pub outcome_id: String,
    /// P(this outcome | hidden state), keyed by state ID.
    pub probability_by_state: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeAdmission {
    Eligible,
    NotPermitted,
    OverBudget,
    MissesDecisionDeadline,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbeValue {
    pub probe_id: String,
    pub admission: ProbeAdmission,
    pub current_bayes_risk: f64,
    pub expected_posterior_bayes_risk: f64,
    pub gross_expected_value_of_sample_information: f64,
    pub acquisition_cost: f64,
    pub net_expected_value_of_sample_information: f64,
    /// Mutual information between hidden state and probe outcome, in bits.
    /// Reported as a diagnostic; it is not the purchase objective.
    pub state_information_gain_bits: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvaluation {
    pub episode_id: String,
    pub posterior_by_state: BTreeMap<String, f64>,
    pub candidate_decision_id: String,
    pub candidate_expected_loss: f64,
    pub candidate_worst_case_loss: f64,
    pub candidate_risk_gate_passed: bool,
    pub information_set_bayes_risk: f64,
    pub perfect_information_lower_bound: f64,
    /// Bayes risk minus the perfect-information lower bound. This is
    /// irreducible under the candidate's current information set.
    pub irreducible_information_gap: f64,
    /// Candidate loss minus the best decision available from its information.
    /// For a decision rejected by the risk gate, this is still reported as a
    /// diagnostic; callers must also inspect candidate_risk_gate_passed.
    pub decision_regret: f64,
    pub requested_probe: Option<ProbeValue>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EvaluationError {
    EmptyEpisodeId,
    InvalidStepWindow,
    InvalidBudget(f64),
    InvalidRiskLimit(f64),
    EmptyStateSet,
    TooManyStates(usize),
    InvalidIdentifier {
        field: &'static str,
    },
    DuplicateIdentifier {
        field: &'static str,
        id: String,
    },
    InvalidProbability {
        field: String,
        value: f64,
    },
    ProbabilityMassNotNormalized {
        field: String,
        state_id: String,
        sum: f64,
    },
    MissingStateProbability {
        field: String,
        state_id: String,
    },
    MissingDecisionLoss {
        decision_id: String,
        state_id: String,
    },
    InvalidDecisionLoss {
        decision_id: String,
        state_id: String,
        value: f64,
    },
    TooManyDecisions(usize),
    TooManyEvidenceItems(usize),
    TooManyProbes(usize),
    TooManyOutcomes(usize),
    ConditionalIndependenceNotDeclared,
    UnknownEvidenceChannel(String),
    UnknownEvidenceOutcome {
        channel_id: String,
        outcome_id: String,
    },
    EvidenceHasZeroLikelihood {
        channel_id: String,
        outcome_id: String,
    },
    UnknownDecision(String),
    NoRiskAdmissibleDecision,
    UnknownProbe(String),
    ProbeNotPermitted(String),
    ProbePublicContractMismatch(String),
    NoPosteriorMassForProbeOutcome {
        probe_id: String,
        outcome_id: String,
    },
    NumericalFailure(&'static str),
}

impl Display for EvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyEpisodeId => write!(formatter, "episode ID must be non-empty"),
            Self::InvalidStepWindow => write!(formatter, "decision deadline precedes current step"),
            Self::InvalidBudget(value) => {
                write!(formatter, "remaining probe budget is invalid: {value}")
            }
            Self::InvalidRiskLimit(value) => {
                write!(formatter, "maximum worst-case loss is invalid: {value}")
            }
            Self::EmptyStateSet => write!(formatter, "evaluator world requires at least one state"),
            Self::TooManyStates(count) => {
                write!(formatter, "state count {count} exceeds {MAX_STATES}")
            }
            Self::InvalidIdentifier { field } => {
                write!(formatter, "{field} identifier must be non-empty")
            }
            Self::DuplicateIdentifier { field, id } => {
                write!(formatter, "duplicate {field} identifier: {id}")
            }
            Self::InvalidProbability { field, value } => {
                write!(formatter, "{field} has invalid probability {value}")
            }
            Self::ProbabilityMassNotNormalized {
                field,
                state_id,
                sum,
            } => write!(
                formatter,
                "{field} probabilities for state {state_id} sum to {sum}, expected 1"
            ),
            Self::MissingStateProbability { field, state_id } => {
                write!(formatter, "{field} has no probability for state {state_id}")
            }
            Self::MissingDecisionLoss {
                decision_id,
                state_id,
            } => write!(
                formatter,
                "decision {decision_id} has no loss for state {state_id}"
            ),
            Self::InvalidDecisionLoss {
                decision_id,
                state_id,
                value,
            } => write!(
                formatter,
                "decision {decision_id} has invalid loss {value} for state {state_id}"
            ),
            Self::TooManyDecisions(count) => {
                write!(formatter, "decision count {count} exceeds {MAX_DECISIONS}")
            }
            Self::TooManyEvidenceItems(count) => {
                write!(formatter, "evidence count {count} exceeds {MAX_EVIDENCE}")
            }
            Self::TooManyProbes(count) => {
                write!(formatter, "probe count {count} exceeds {MAX_PROBES}")
            }
            Self::TooManyOutcomes(count) => {
                write!(formatter, "outcome count {count} exceeds {MAX_OUTCOMES}")
            }
            Self::ConditionalIndependenceNotDeclared => write!(
                formatter,
                "multiple evidence channels require an explicit conditional-independence declaration"
            ),
            Self::UnknownEvidenceChannel(channel) => {
                write!(formatter, "unknown evidence channel {channel}")
            }
            Self::UnknownEvidenceOutcome {
                channel_id,
                outcome_id,
            } => write!(
                formatter,
                "unknown outcome {outcome_id} for evidence channel {channel_id}"
            ),
            Self::EvidenceHasZeroLikelihood {
                channel_id,
                outcome_id,
            } => write!(
                formatter,
                "observed outcome {outcome_id} has zero likelihood in channel {channel_id} under every state"
            ),
            Self::UnknownDecision(decision) => write!(formatter, "unknown decision {decision}"),
            Self::NoRiskAdmissibleDecision => write!(
                formatter,
                "no downstream decision passes the frozen worst-case risk limit"
            ),
            Self::UnknownProbe(probe) => write!(formatter, "unknown probe {probe}"),
            Self::ProbeNotPermitted(probe) => write!(
                formatter,
                "probe {probe} is not in the candidate-visible permitted set"
            ),
            Self::ProbePublicContractMismatch(probe) => write!(
                formatter,
                "probe {probe} public cost/deadline contract does not match evaluator contract"
            ),
            Self::NoPosteriorMassForProbeOutcome {
                probe_id,
                outcome_id,
            } => write!(
                formatter,
                "probe {probe_id} outcome {outcome_id} has no posterior mass"
            ),
            Self::NumericalFailure(message) => {
                write!(formatter, "numerical evaluation failed: {message}")
            }
        }
    }
}

impl Error for EvaluationError {}

fn finite_probability(field: String, value: f64) -> Result<(), EvaluationError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(EvaluationError::InvalidProbability { field, value });
    }
    Ok(())
}

fn near_one(value: f64) -> bool {
    (value - 1.0).abs() <= PROBABILITY_TOLERANCE
}

fn validate_outcome_likelihoods(
    field: &str,
    states: &[StateModel],
    outcomes: &[OutcomeLikelihood],
) -> Result<(), EvaluationError> {
    if outcomes.is_empty() {
        return Err(EvaluationError::TooManyOutcomes(0));
    }
    if outcomes.len() > MAX_OUTCOMES {
        return Err(EvaluationError::TooManyOutcomes(outcomes.len()));
    }
    let mut outcome_ids = BTreeSet::new();
    for outcome in outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(EvaluationError::InvalidIdentifier { field: "outcome" });
        }
        if !outcome_ids.insert(outcome.outcome_id.as_str()) {
            return Err(EvaluationError::DuplicateIdentifier {
                field: "outcome",
                id: outcome.outcome_id.clone(),
            });
        }
    }
    for state in states {
        let mut total = 0.0;
        for outcome in outcomes {
            let probability = outcome
                .probability_by_state
                .get(&state.state_id)
                .copied()
                .ok_or_else(|| EvaluationError::MissingStateProbability {
                    field: field.to_owned(),
                    state_id: state.state_id.clone(),
                })?;
            finite_probability(
                format!("{field}.{}.{}", outcome.outcome_id, state.state_id),
                probability,
            )?;
            total += probability;
        }
        if !near_one(total) {
            return Err(EvaluationError::ProbabilityMassNotNormalized {
                field: field.to_owned(),
                state_id: state.state_id.clone(),
                sum: total,
            });
        }
    }
    Ok(())
}

impl EvaluatorWorld {
    pub fn validate(&self) -> Result<(), EvaluationError> {
        if self.states.is_empty() {
            return Err(EvaluationError::EmptyStateSet);
        }
        if self.states.len() > MAX_STATES {
            return Err(EvaluationError::TooManyStates(self.states.len()));
        }
        let mut state_ids = BTreeSet::new();
        let mut prior_total = 0.0;
        for state in &self.states {
            if state.state_id.trim().is_empty() {
                return Err(EvaluationError::InvalidIdentifier { field: "state" });
            }
            if !state_ids.insert(state.state_id.as_str()) {
                return Err(EvaluationError::DuplicateIdentifier {
                    field: "state",
                    id: state.state_id.clone(),
                });
            }
            finite_probability(format!("prior.{}", state.state_id), state.prior_probability)?;
            prior_total += state.prior_probability;
        }
        if !near_one(prior_total) {
            return Err(EvaluationError::NumericalFailure(
                "state prior mass is not normalized",
            ));
        }

        let mut channel_ids = BTreeSet::new();
        for channel in &self.evidence_channels {
            if channel.channel_id.trim().is_empty() {
                return Err(EvaluationError::InvalidIdentifier {
                    field: "evidence channel",
                });
            }
            if !channel_ids.insert(channel.channel_id.as_str()) {
                return Err(EvaluationError::DuplicateIdentifier {
                    field: "evidence channel",
                    id: channel.channel_id.clone(),
                });
            }
            validate_outcome_likelihoods(
                &format!("evidence_channel.{}", channel.channel_id),
                &self.states,
                &channel.outcomes,
            )?;
        }

        let mut probe_ids = BTreeSet::new();
        if self.probes.len() > MAX_PROBES {
            return Err(EvaluationError::TooManyProbes(self.probes.len()));
        }
        for probe in &self.probes {
            if probe.probe_id.trim().is_empty() {
                return Err(EvaluationError::InvalidIdentifier { field: "probe" });
            }
            if !probe_ids.insert(probe.probe_id.as_str()) {
                return Err(EvaluationError::DuplicateIdentifier {
                    field: "probe",
                    id: probe.probe_id.clone(),
                });
            }
            if !probe.acquisition_cost.is_finite() || probe.acquisition_cost < 0.0 {
                return Err(EvaluationError::InvalidBudget(probe.acquisition_cost));
            }
            validate_outcome_likelihoods(
                &format!("probe.{}", probe.probe_id),
                &self.states,
                &probe.outcomes,
            )?;
        }
        Ok(())
    }
}

fn validate_episode(
    episode: &CandidateEpisode,
    world: &EvaluatorWorld,
) -> Result<(), EvaluationError> {
    if episode.episode_id.trim().is_empty() {
        return Err(EvaluationError::EmptyEpisodeId);
    }
    if episode.decision_deadline_step < episode.current_step {
        return Err(EvaluationError::InvalidStepWindow);
    }
    if !episode.remaining_probe_budget.is_finite() || episode.remaining_probe_budget < 0.0 {
        return Err(EvaluationError::InvalidBudget(
            episode.remaining_probe_budget,
        ));
    }
    if !episode.maximum_worst_case_loss.is_finite()
        || !(0.0..=1.0).contains(&episode.maximum_worst_case_loss)
    {
        return Err(EvaluationError::InvalidRiskLimit(
            episode.maximum_worst_case_loss,
        ));
    }
    if episode.allowed_decision_ids.is_empty() || episode.allowed_decision_ids.len() > MAX_DECISIONS
    {
        return Err(EvaluationError::TooManyDecisions(
            episode.allowed_decision_ids.len(),
        ));
    }
    if episode.evidence.len() > MAX_EVIDENCE {
        return Err(EvaluationError::TooManyEvidenceItems(
            episode.evidence.len(),
        ));
    }
    if episode.permitted_probes.len() > MAX_PROBES {
        return Err(EvaluationError::TooManyProbes(
            episode.permitted_probes.len(),
        ));
    }

    let mut decision_ids = BTreeSet::new();
    for decision_id in &episode.allowed_decision_ids {
        if decision_id.trim().is_empty() {
            return Err(EvaluationError::InvalidIdentifier { field: "decision" });
        }
        if !decision_ids.insert(decision_id.as_str()) {
            return Err(EvaluationError::DuplicateIdentifier {
                field: "decision",
                id: decision_id.clone(),
            });
        }
        for state in &world.states {
            let loss = state
                .decision_loss
                .get(decision_id)
                .copied()
                .ok_or_else(|| EvaluationError::MissingDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state.state_id.clone(),
                })?;
            if !loss.is_finite() || !(0.0..=1.0).contains(&loss) {
                return Err(EvaluationError::InvalidDecisionLoss {
                    decision_id: decision_id.clone(),
                    state_id: state.state_id.clone(),
                    value: loss,
                });
            }
        }
    }

    if episode.evidence.len() > 1 && !world.evidence_channels_conditionally_independent {
        return Err(EvaluationError::ConditionalIndependenceNotDeclared);
    }

    for observed in &episode.evidence {
        let channel = world
            .evidence_channels
            .iter()
            .find(|channel| channel.channel_id == observed.channel_id)
            .ok_or_else(|| EvaluationError::UnknownEvidenceChannel(observed.channel_id.clone()))?;
        if !channel
            .outcomes
            .iter()
            .any(|outcome| outcome.outcome_id == observed.outcome_id)
        {
            return Err(EvaluationError::UnknownEvidenceOutcome {
                channel_id: observed.channel_id.clone(),
                outcome_id: observed.outcome_id.clone(),
            });
        }
    }

    let mut permitted_ids = BTreeSet::new();
    for option in &episode.permitted_probes {
        if option.probe_id.trim().is_empty() {
            return Err(EvaluationError::InvalidIdentifier {
                field: "permitted probe",
            });
        }
        if !permitted_ids.insert(option.probe_id.as_str()) {
            return Err(EvaluationError::DuplicateIdentifier {
                field: "permitted probe",
                id: option.probe_id.clone(),
            });
        }
        let probe = world
            .probes
            .iter()
            .find(|probe| probe.probe_id == option.probe_id)
            .ok_or_else(|| EvaluationError::UnknownProbe(option.probe_id.clone()))?;
        if !option.acquisition_cost.is_finite()
            || option.acquisition_cost < 0.0
            || (option.acquisition_cost - probe.acquisition_cost).abs() > VALUE_TOLERANCE
            || option.result_delay_steps != probe.result_delay_steps
        {
            return Err(EvaluationError::ProbePublicContractMismatch(
                option.probe_id.clone(),
            ));
        }
    }
    Ok(())
}

fn posterior_after_evidence(
    episode: &CandidateEpisode,
    world: &EvaluatorWorld,
) -> Result<BTreeMap<String, f64>, EvaluationError> {
    let mut weights: BTreeMap<String, f64> = world
        .states
        .iter()
        .map(|state| (state.state_id.clone(), state.prior_probability))
        .collect();

    for observed in &episode.evidence {
        let channel = world
            .evidence_channels
            .iter()
            .find(|channel| channel.channel_id == observed.channel_id)
            .ok_or_else(|| EvaluationError::UnknownEvidenceChannel(observed.channel_id.clone()))?;
        let likelihood = channel
            .outcomes
            .iter()
            .find(|outcome| outcome.outcome_id == observed.outcome_id)
            .ok_or_else(|| EvaluationError::UnknownEvidenceOutcome {
                channel_id: observed.channel_id.clone(),
                outcome_id: observed.outcome_id.clone(),
            })?;
        for state in &world.states {
            let probability = likelihood.probability_by_state[&state.state_id];
            *weights
                .get_mut(&state.state_id)
                .expect("world state is present in posterior") *= probability;
        }
    }

    let normalizer: f64 = weights.values().sum();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        let last = episode
            .evidence
            .last()
            .expect("caller can have no evidence");
        return Err(EvaluationError::EvidenceHasZeroLikelihood {
            channel_id: last.channel_id.clone(),
            outcome_id: last.outcome_id.clone(),
        });
    }
    for probability in weights.values_mut() {
        *probability /= normalizer;
    }
    Ok(weights)
}

fn decision_risk(
    decision_id: &str,
    posterior: &BTreeMap<String, f64>,
    world: &EvaluatorWorld,
) -> Result<f64, EvaluationError> {
    let mut risk = 0.0;
    for state in &world.states {
        let loss = state
            .decision_loss
            .get(decision_id)
            .copied()
            .ok_or_else(|| EvaluationError::MissingDecisionLoss {
                decision_id: decision_id.to_owned(),
                state_id: state.state_id.clone(),
            })?;
        risk += posterior[&state.state_id] * loss;
    }
    Ok(risk)
}

fn decision_worst_case_loss(
    decision_id: &str,
    world: &EvaluatorWorld,
) -> Result<f64, EvaluationError> {
    world
        .states
        .iter()
        .map(|state| {
            state
                .decision_loss
                .get(decision_id)
                .copied()
                .ok_or_else(|| EvaluationError::MissingDecisionLoss {
                    decision_id: decision_id.to_owned(),
                    state_id: state.state_id.clone(),
                })
        })
        .try_fold(0.0_f64, |maximum, loss| Ok(maximum.max(loss?)))
}

fn admissible_decisions<'a>(
    episode: &'a CandidateEpisode,
    world: &'a EvaluatorWorld,
) -> Result<Vec<&'a str>, EvaluationError> {
    let mut admissible = Vec::new();
    for decision in &episode.allowed_decision_ids {
        if decision_worst_case_loss(decision, world)?
            <= episode.maximum_worst_case_loss + VALUE_TOLERANCE
        {
            admissible.push(decision.as_str());
        }
    }
    if admissible.is_empty() {
        return Err(EvaluationError::NoRiskAdmissibleDecision);
    }
    Ok(admissible)
}

fn bayes_risk(
    posterior: &BTreeMap<String, f64>,
    world: &EvaluatorWorld,
    decisions: &[&str],
) -> Result<f64, EvaluationError> {
    let mut best = f64::INFINITY;
    for decision in decisions {
        best = best.min(decision_risk(decision, posterior, world)?);
    }
    if !best.is_finite() {
        return Err(EvaluationError::NumericalFailure(
            "Bayes risk is not finite",
        ));
    }
    Ok(best)
}

fn perfect_information_risk(
    posterior: &BTreeMap<String, f64>,
    world: &EvaluatorWorld,
    decisions: &[&str],
) -> Result<f64, EvaluationError> {
    let mut total = 0.0;
    for state in &world.states {
        let state_best = decisions
            .iter()
            .map(|decision| state.decision_loss[*decision])
            .fold(f64::INFINITY, f64::min);
        total += posterior[&state.state_id] * state_best;
    }
    if !total.is_finite() {
        return Err(EvaluationError::NumericalFailure(
            "perfect information risk is not finite",
        ));
    }
    Ok(total)
}

fn probe_metrics(
    probe: &ProbeModel,
    posterior: &BTreeMap<String, f64>,
    world: &EvaluatorWorld,
    decisions: &[&str],
) -> Result<(f64, f64, f64), EvaluationError> {
    let current_risk = bayes_risk(posterior, world, decisions)?;
    let mut expected_after_risk = 0.0;
    let mut mutual_information_bits = 0.0;

    for outcome in &probe.outcomes {
        let outcome_probability: f64 = world
            .states
            .iter()
            .map(|state| posterior[&state.state_id] * outcome.probability_by_state[&state.state_id])
            .sum();
        if outcome_probability <= PROBABILITY_TOLERANCE {
            continue;
        }
        let outcome_posterior: BTreeMap<String, f64> = world
            .states
            .iter()
            .map(|state| {
                let joint =
                    posterior[&state.state_id] * outcome.probability_by_state[&state.state_id];
                (state.state_id.clone(), joint / outcome_probability)
            })
            .collect();
        expected_after_risk +=
            outcome_probability * bayes_risk(&outcome_posterior, world, decisions)?;

        for state in &world.states {
            let conditional = outcome.probability_by_state[&state.state_id];
            let joint = posterior[&state.state_id] * conditional;
            if joint > 0.0 && conditional > 0.0 {
                mutual_information_bits += joint * (conditional / outcome_probability).log2();
            }
        }
    }

    let gross_evoi = current_risk - expected_after_risk;
    if !expected_after_risk.is_finite()
        || !gross_evoi.is_finite()
        || !mutual_information_bits.is_finite()
    {
        return Err(EvaluationError::NumericalFailure(
            "probe expected value or information gain is not finite",
        ));
    }
    Ok((
        expected_after_risk,
        gross_evoi,
        mutual_information_bits.max(0.0),
    ))
}

/// Evaluate one candidate decision and, optionally, the expected decision value
/// of a requested next probe.
///
/// The returned posterior, Bayes risks and probe metrics belong to the evaluator.
/// They must not be exposed to the candidate during the episode. This function
/// is deterministic and does not simulate execution of the requested probe.
pub fn evaluate_candidate(
    episode: &CandidateEpisode,
    world: &EvaluatorWorld,
    submission: &CandidateSubmission,
) -> Result<CandidateEvaluation, EvaluationError> {
    world.validate()?;
    validate_episode(episode, world)?;

    if !episode
        .allowed_decision_ids
        .iter()
        .any(|id| id == &submission.decision_id)
    {
        return Err(EvaluationError::UnknownDecision(
            submission.decision_id.clone(),
        ));
    }

    let posterior = posterior_after_evidence(episode, world)?;
    let decisions = admissible_decisions(episode, world)?;
    let candidate_expected_loss = decision_risk(&submission.decision_id, &posterior, world)?;
    let candidate_worst_case_loss = decision_worst_case_loss(&submission.decision_id, world)?;
    let candidate_risk_gate_passed =
        candidate_worst_case_loss <= episode.maximum_worst_case_loss + VALUE_TOLERANCE;
    let information_set_bayes_risk = bayes_risk(&posterior, world, &decisions)?;
    let perfect_information_lower_bound = perfect_information_risk(&posterior, world, &decisions)?;
    let irreducible_information_gap =
        (information_set_bayes_risk - perfect_information_lower_bound).max(0.0);
    let decision_regret = candidate_expected_loss - information_set_bayes_risk;

    let requested_probe = match &submission.requested_probe_id {
        None => None,
        Some(probe_id) => {
            let probe = world
                .probes
                .iter()
                .find(|probe| &probe.probe_id == probe_id)
                .ok_or_else(|| EvaluationError::UnknownProbe(probe_id.clone()))?;
            let option = episode
                .permitted_probes
                .iter()
                .find(|option| &option.probe_id == probe_id);
            let admission = match option {
                None => ProbeAdmission::NotPermitted,
                Some(option)
                    if option.acquisition_cost
                        > episode.remaining_probe_budget + VALUE_TOLERANCE =>
                {
                    ProbeAdmission::OverBudget
                }
                Some(option)
                    if episode
                        .current_step
                        .saturating_add(option.result_delay_steps)
                        > episode.decision_deadline_step =>
                {
                    ProbeAdmission::MissesDecisionDeadline
                }
                Some(_) => ProbeAdmission::Eligible,
            };
            let (expected_after, gross_evoi, information_gain) =
                probe_metrics(probe, &posterior, world, &decisions)?;
            Some(ProbeValue {
                probe_id: probe.probe_id.clone(),
                admission,
                current_bayes_risk: information_set_bayes_risk,
                expected_posterior_bayes_risk: expected_after,
                gross_expected_value_of_sample_information: gross_evoi,
                acquisition_cost: probe.acquisition_cost,
                net_expected_value_of_sample_information: gross_evoi - probe.acquisition_cost,
                state_information_gain_bits: information_gain,
            })
        }
    };

    Ok(CandidateEvaluation {
        episode_id: episode.episode_id.clone(),
        posterior_by_state: posterior,
        candidate_decision_id: submission.decision_id.clone(),
        candidate_expected_loss,
        candidate_worst_case_loss,
        candidate_risk_gate_passed,
        information_set_bayes_risk,
        perfect_information_lower_bound,
        irreducible_information_gap,
        decision_regret,
        requested_probe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(id: &str, prior: f64, losses: &[(&str, f64)]) -> StateModel {
        StateModel {
            state_id: id.to_owned(),
            prior_probability: prior,
            decision_loss: losses
                .iter()
                .map(|(name, loss)| ((*name).to_owned(), *loss))
                .collect(),
        }
    }

    fn outcome(id: &str, likelihoods: &[(&str, f64)]) -> OutcomeLikelihood {
        OutcomeLikelihood {
            outcome_id: id.to_owned(),
            probability_by_state: likelihoods
                .iter()
                .map(|(state, p)| ((*state).to_owned(), *p))
                .collect(),
        }
    }

    fn episode(
        evidence: Vec<ObservedSignal>,
        max_loss: f64,
        budget: f64,
        deadline: u64,
    ) -> CandidateEpisode {
        CandidateEpisode {
            episode_id: "synthetic-episode".into(),
            current_step: 0,
            decision_deadline_step: deadline,
            remaining_probe_budget: budget,
            maximum_worst_case_loss: max_loss,
            allowed_decision_ids: vec!["choose_A".into(), "choose_B".into(), "abstain".into()],
            evidence,
            permitted_probes: vec![PublicProbeOption {
                probe_id: "reference_probe".into(),
                acquisition_cost: 0.1,
                result_delay_steps: 1,
            }],
        }
    }

    fn two_worlds(probe_outcomes: Vec<OutcomeLikelihood>) -> EvaluatorWorld {
        EvaluatorWorld {
            states: vec![
                state(
                    "H_A",
                    0.5,
                    &[("choose_A", 0.0), ("choose_B", 1.0), ("abstain", 0.4)],
                ),
                state(
                    "H_B",
                    0.5,
                    &[("choose_A", 1.0), ("choose_B", 0.0), ("abstain", 0.4)],
                ),
            ],
            evidence_channels: vec![EvidenceChannelModel {
                channel_id: "primary".into(),
                // This fixture models only the passive, observationally
                // equivalent stream. Other outcomes would need probabilities
                // that normalize across the complete channel.
                outcomes: vec![outcome("ambiguous", &[("H_A", 1.0), ("H_B", 1.0)])],
            }],
            probes: vec![ProbeModel {
                probe_id: "reference_probe".into(),
                acquisition_cost: 0.1,
                result_delay_steps: 1,
                outcomes: probe_outcomes,
            }],
            evidence_channels_conditionally_independent: true,
        }
    }

    fn perfect_probe_outcomes() -> Vec<OutcomeLikelihood> {
        vec![
            outcome("supports_A", &[("H_A", 1.0), ("H_B", 0.0)]),
            outcome("supports_B", &[("H_A", 0.0), ("H_B", 1.0)]),
        ]
    }

    #[test]
    fn passive_observational_equivalence_creates_irreducible_decision_risk() {
        let world = two_worlds(perfect_probe_outcomes());
        let task = episode(
            vec![ObservedSignal {
                channel_id: "primary".into(),
                outcome_id: "ambiguous".into(),
            }],
            1.0,
            1.0,
            2,
        );
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "choose_A".into(),
                requested_probe_id: None,
            },
        )
        .unwrap();

        assert!((result.posterior_by_state["H_A"] - 0.5).abs() < VALUE_TOLERANCE);
        assert!((result.posterior_by_state["H_B"] - 0.5).abs() < VALUE_TOLERANCE);
        assert!((result.information_set_bayes_risk - 0.4).abs() < VALUE_TOLERANCE);
        assert!((result.perfect_information_lower_bound - 0.0).abs() < VALUE_TOLERANCE);
        assert!((result.irreducible_information_gap - 0.4).abs() < VALUE_TOLERANCE);
        assert!(result.candidate_risk_gate_passed);
    }

    #[test]
    fn decision_changing_probe_has_positive_net_evoi() {
        let world = two_worlds(perfect_probe_outcomes());
        let task = episode(
            vec![ObservedSignal {
                channel_id: "primary".into(),
                outcome_id: "ambiguous".into(),
            }],
            1.0,
            1.0,
            2,
        );
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: Some("reference_probe".into()),
            },
        )
        .unwrap();
        let probe = result.requested_probe.unwrap();

        assert_eq!(probe.admission, ProbeAdmission::Eligible);
        assert!((probe.current_bayes_risk - 0.4).abs() < VALUE_TOLERANCE);
        assert!((probe.expected_posterior_bayes_risk - 0.0).abs() < VALUE_TOLERANCE);
        assert!((probe.gross_expected_value_of_sample_information - 0.4).abs() < VALUE_TOLERANCE);
        assert!((probe.net_expected_value_of_sample_information - 0.3).abs() < VALUE_TOLERANCE);
        assert!((probe.state_information_gain_bits - 1.0).abs() < VALUE_TOLERANCE);
    }

    #[test]
    fn one_bit_nuisance_probe_has_zero_decision_value() {
        let world = EvaluatorWorld {
            states: vec![
                state("H_A", 0.5, &[("continue", 0.0), ("stop", 1.0)]),
                state("H_B", 0.5, &[("continue", 0.0), ("stop", 1.0)]),
            ],
            evidence_channels: Vec::new(),
            probes: vec![ProbeModel {
                probe_id: "mechanism_diagnostic".into(),
                acquisition_cost: 0.04,
                result_delay_steps: 1,
                outcomes: perfect_probe_outcomes(),
            }],
            evidence_channels_conditionally_independent: true,
        };
        let task = CandidateEpisode {
            episode_id: "nuisance-probe".into(),
            current_step: 0,
            decision_deadline_step: 2,
            remaining_probe_budget: 1.0,
            maximum_worst_case_loss: 1.0,
            allowed_decision_ids: vec!["continue".into(), "stop".into()],
            evidence: Vec::new(),
            permitted_probes: vec![PublicProbeOption {
                probe_id: "mechanism_diagnostic".into(),
                acquisition_cost: 0.04,
                result_delay_steps: 1,
            }],
        };
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "continue".into(),
                requested_probe_id: Some("mechanism_diagnostic".into()),
            },
        )
        .unwrap();
        let probe = result.requested_probe.unwrap();

        assert!((probe.state_information_gain_bits - 1.0).abs() < VALUE_TOLERANCE);
        assert!(probe.gross_expected_value_of_sample_information.abs() < VALUE_TOLERANCE);
        assert!((probe.net_expected_value_of_sample_information + 0.04).abs() < VALUE_TOLERANCE);
    }

    #[test]
    fn risk_limit_vetoes_a_decision_even_when_its_expected_loss_is_competitive() {
        let world = two_worlds(perfect_probe_outcomes());
        let task = episode(
            vec![ObservedSignal {
                channel_id: "primary".into(),
                outcome_id: "ambiguous".into(),
            }],
            0.4,
            1.0,
            2,
        );
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "choose_A".into(),
                requested_probe_id: None,
            },
        )
        .unwrap();

        assert!(!result.candidate_risk_gate_passed);
        assert!((result.candidate_expected_loss - 0.5).abs() < VALUE_TOLERANCE);
        assert!((result.information_set_bayes_risk - 0.4).abs() < VALUE_TOLERANCE);
        assert!(result.decision_regret > 0.0);
    }

    #[test]
    fn over_budget_probe_is_denied_without_erasing_its_estimated_value() {
        let world = two_worlds(perfect_probe_outcomes());
        let task = episode(
            vec![ObservedSignal {
                channel_id: "primary".into(),
                outcome_id: "ambiguous".into(),
            }],
            1.0,
            0.05,
            2,
        );
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: Some("reference_probe".into()),
            },
        )
        .unwrap();
        let probe = result.requested_probe.unwrap();

        assert_eq!(probe.admission, ProbeAdmission::OverBudget);
        assert!((probe.net_expected_value_of_sample_information - 0.3).abs() < VALUE_TOLERANCE);
    }

    #[test]
    fn late_probe_is_denied_before_the_decision_deadline() {
        let world = two_worlds(perfect_probe_outcomes());
        let mut task = episode(Vec::new(), 1.0, 1.0, 0);
        task.current_step = 0;
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: Some("reference_probe".into()),
            },
        )
        .unwrap();
        assert_eq!(
            result.requested_probe.unwrap().admission,
            ProbeAdmission::MissesDecisionDeadline
        );
    }

    #[test]
    fn unknown_or_unpermitted_probe_fails_closed() {
        let world = two_worlds(perfect_probe_outcomes());
        let task = episode(Vec::new(), 1.0, 1.0, 2);
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: Some("not-permitted".into()),
            },
        );
        assert!(matches!(result, Err(EvaluationError::UnknownProbe(_))));

        let mut not_allowed = task.clone();
        not_allowed.permitted_probes.clear();
        let result = evaluate_candidate(
            &not_allowed,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: Some("reference_probe".into()),
            },
        )
        .unwrap();
        assert_eq!(
            result.requested_probe.unwrap().admission,
            ProbeAdmission::NotPermitted
        );
    }

    #[test]
    fn invalid_probe_likelihood_mass_is_rejected() {
        let mut world = two_worlds(vec![
            outcome("left", &[("H_A", 0.8), ("H_B", 0.8)]),
            outcome("right", &[("H_A", 0.1), ("H_B", 0.1)]),
        ]);
        world.probes[0].acquisition_cost = 0.1;
        assert!(matches!(
            world.validate(),
            Err(EvaluationError::ProbabilityMassNotNormalized { .. })
        ));
    }

    #[test]
    fn multiple_evidence_channels_require_declared_independence() {
        let mut world = two_worlds(perfect_probe_outcomes());
        world.evidence_channels.push(EvidenceChannelModel {
            channel_id: "secondary".into(),
            outcomes: vec![outcome("same", &[("H_A", 1.0), ("H_B", 1.0)])],
        });
        world.evidence_channels_conditionally_independent = false;
        let task = episode(
            vec![
                ObservedSignal {
                    channel_id: "primary".into(),
                    outcome_id: "ambiguous".into(),
                },
                ObservedSignal {
                    channel_id: "secondary".into(),
                    outcome_id: "same".into(),
                },
            ],
            1.0,
            1.0,
            2,
        );
        let result = evaluate_candidate(
            &task,
            &world,
            &CandidateSubmission {
                decision_id: "abstain".into(),
                requested_probe_id: None,
            },
        );
        assert!(matches!(
            result,
            Err(EvaluationError::ConditionalIndependenceNotDeclared)
        ));
    }
}

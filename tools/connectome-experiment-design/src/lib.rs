//! Dependency-free reference utilities for information-seeking connectome experiments.
//!
//! This crate implements exact discrete expected information gain (EIG) for a
//! finite hypothesis set and candidate experiments. It is a mathematical
//! selector, not a biological model: callers supply priors, outcome likelihoods,
//! and costs. The code does not estimate or calibrate them.
//!
//! For hypothesis H and candidate experiment e with outcome Y:
//!
//!     EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
//!
//! Entropy and EIG are reported in bits. Cost-normalized EIG is bits per unit of
//! caller-defined cost. It is not interchangeable with downstream decision
//! utility or regret.

use core::fmt;
use std::collections::HashSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const ROUND_OFF_TOLERANCE: f64 = 1.0e-12;

/// Errors are explicit so malformed priors and likelihoods are never silently
/// normalized into plausible-looking experiment rankings.
#[derive(Debug, Clone, PartialEq)]
pub enum DesignError {
    EmptyHypothesisSet,
    PriorCountMismatch { expected: usize, actual: usize },
    InvalidPrior { index: usize, value: f64 },
    PriorMassNotOne { sum: f64 },
    EmptyExperimentId,
    DuplicateExperimentId { experiment_id: String },
    InvalidExperimentCost { cost: f64 },
    NoCandidateExperiments,
    EmptyOutcomeSet,
    DuplicateOutcomeLabel { label: String },
    LikelihoodRowCountMismatch { expected: usize, actual: usize },
    LikelihoodOutcomeCountMismatch {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLikelihood {
        hypothesis_index: usize,
        outcome_index: usize,
        value: f64,
    },
    LikelihoodMassNotOne { hypothesis_index: usize, sum: f64 },
    OutcomeIndexOutOfRange { index: usize, outcome_count: usize },
    ImpossibleObservedOutcome { outcome_index: usize },
    SelectedExperimentNotFound { experiment_id: String },
    EmptyHypothesisId,
    DuplicateHypothesisId { hypothesis_id: String },
    MissingUnresolvedHypothesis,
    ZeroUnresolvedPrior,
    InconsistentHypothesisOrder,
    EmptyTargetQueryId,
    TargetQueryHypothesisCountMismatch { expected: usize, actual: usize },
    EmptyTargetQueryAnswer,
    EmptyDecisionProblemId,
    EmptyDecisionActionSet,
    EmptyDecisionActionId,
    DuplicateDecisionActionId { action_id: String },
    DecisionUtilityCountMismatch { action_id: String, expected: usize, actual: usize },
    NonFiniteDecisionUtility { action_id: String, hypothesis_index: usize },
    EmptyHitObjectiveId,
    HitOutcomeCountMismatch { expected: usize, actual: usize },
    InconsistentHitOutcomeOrder,
    NonFiniteHitOutcomeValue { outcome_index: usize },
    NonFiniteHitThreshold,
    EmptyLikelihoodScenarioSet,
    EmptyLikelihoodScenarioId,
    DuplicateLikelihoodScenarioId { scenario_id: String },
    EmptyCalibrationCohort,
    EmptyCalibrationUnitId,
    EmptyMechanismFamilyId,
    DuplicateCalibrationUnitId { unit_id: String },
    DuplicateInScopeMechanismFamily { mechanism_family_id: String },
    DuplicateOmittedMechanismFamily { mechanism_family_id: String },
    MechanismFamilyLeakage { mechanism_family_id: String },
    NonFiniteNonconformityScore,
    InvalidAdequacyScore,
    NumericalInstability,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis set must not be empty"),
            Self::PriorCountMismatch { expected, actual } => {
                write!(f, "expected {expected} hypothesis priors, received {actual}")
            }
            Self::InvalidPrior { index, value } => write!(f, "prior {index} is invalid: {value}"),
            Self::PriorMassNotOne { sum } => {
                write!(f, "hypothesis priors must sum to one, received {sum}")
            }
            Self::EmptyExperimentId => write!(f, "experiment id must not be empty"),
            Self::DuplicateExperimentId { experiment_id } => {
                write!(f, "duplicate experiment id {experiment_id:?}")
            }
            Self::InvalidExperimentCost { cost } => {
                write!(f, "experiment cost must be finite and positive, received {cost}")
            }
            Self::NoCandidateExperiments => write!(f, "candidate experiment set must not be empty"),
            Self::EmptyOutcomeSet => write!(f, "candidate experiment must declare at least one outcome"),
            Self::DuplicateOutcomeLabel { label } => {
                write!(f, "candidate experiment has duplicate outcome label {label:?}")
            }
            Self::LikelihoodRowCountMismatch { expected, actual } => {
                write!(f, "expected {expected} likelihood rows, received {actual}")
            }
            Self::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} outcome likelihoods; expected {expected}"
            ),
            Self::InvalidLikelihood {
                hypothesis_index,
                outcome_index,
                value,
            } => write!(
                f,
                "likelihood for hypothesis {hypothesis_index}, outcome {outcome_index} is invalid: {value}"
            ),
            Self::LikelihoodMassNotOne { hypothesis_index, sum } => write!(
                f,
                "likelihoods for hypothesis {hypothesis_index} must sum to one, received {sum}"
            ),
            Self::OutcomeIndexOutOfRange { index, outcome_count } => {
                write!(f, "outcome index {index} is outside 0..{outcome_count}")
            }
            Self::ImpossibleObservedOutcome { outcome_index } => {
                write!(f, "outcome {outcome_index} has zero probability under every hypothesis")
            }
            Self::SelectedExperimentNotFound { experiment_id } => {
                write!(f, "selected experiment {experiment_id:?} is not in the candidate set")
            }
            Self::EmptyHypothesisId => write!(f, "hypothesis identifiers must not be empty"),
            Self::DuplicateHypothesisId { hypothesis_id } => {
                write!(f, "duplicate hypothesis identifier {hypothesis_id:?}")
            }
            Self::MissingUnresolvedHypothesis => {
                write!(f, "hypothesis set must include OTHER_OR_UNRESOLVED exactly once")
            }
            Self::ZeroUnresolvedPrior => {
                write!(f, "OTHER_OR_UNRESOLVED must have a positive caller-supplied prior")
            }
            Self::InconsistentHypothesisOrder => {
                write!(f, "all ranked open-world candidates and target queries must use identical hypothesis IDs in identical prior order")
            }
            Self::EmptyTargetQueryId => write!(f, "target causal query identifier must not be empty"),
            Self::TargetQueryHypothesisCountMismatch { expected, actual } => {
                write!(f, "target query has {actual} answer labels; expected {expected} hypotheses")
            }
            Self::EmptyTargetQueryAnswer => {
                write!(f, "target query answer labels must not be empty")
            }
            Self::EmptyDecisionProblemId => write!(f, "decision problem identifier must not be empty"),
            Self::EmptyDecisionActionSet => write!(f, "decision problem must include at least one action"),
            Self::EmptyDecisionActionId => write!(f, "decision action identifiers must not be empty"),
            Self::DuplicateDecisionActionId { action_id } => {
                write!(f, "duplicate decision action identifier {action_id:?}")
            }
            Self::DecisionUtilityCountMismatch { action_id, expected, actual } => {
                write!(f, "decision action {action_id:?} has {actual} utility entries; expected {expected}")
            }
            Self::NonFiniteDecisionUtility { action_id, hypothesis_index } => {
                write!(f, "decision action {action_id:?} has non-finite utility at hypothesis {hypothesis_index}")
            }
            Self::EmptyHitObjectiveId => write!(f, "threshold-hit objective identifier must not be empty"),
            Self::HitOutcomeCountMismatch { expected, actual } => {
                write!(f, "threshold-hit objective has {actual} outcome values; expected {expected}")
            }
            Self::InconsistentHitOutcomeOrder => {
                write!(f, "threshold-hit outcome IDs must exactly match candidate outcome ordering")
            }
            Self::NonFiniteHitOutcomeValue { outcome_index } => {
                write!(f, "threshold-hit outcome value at index {outcome_index} must be finite")
            }
            Self::NonFiniteHitThreshold => write!(f, "threshold-hit value must be finite"),
            Self::EmptyLikelihoodScenarioSet => {
                write!(f, "open-world candidate must declare at least one likelihood scenario")
            }
            Self::EmptyLikelihoodScenarioId => {
                write!(f, "likelihood scenario identifier must not be empty")
            }
            Self::DuplicateLikelihoodScenarioId { scenario_id } => {
                write!(f, "duplicate likelihood scenario identifier {scenario_id:?}")
            }
            Self::EmptyCalibrationCohort => {
                write!(f, "in-scope and omitted-mechanism calibration cohorts must both be non-empty")
            }
            Self::EmptyCalibrationUnitId => {
                write!(f, "calibration unit identifiers must not be empty")
            }
            Self::EmptyMechanismFamilyId => {
                write!(f, "mechanism family identifiers must not be empty")
            }
            Self::DuplicateCalibrationUnitId { unit_id } => {
                write!(f, "duplicate calibration unit identifier {unit_id:?}")
            }
            Self::DuplicateInScopeMechanismFamily { mechanism_family_id } => {
                write!(f, "in-scope calibration must contain one aggregate score per independent family; duplicate {mechanism_family_id:?}")
            }
            Self::DuplicateOmittedMechanismFamily { mechanism_family_id } => {
                write!(f, "omitted-mechanism calibration must contain one aggregate score per independent family; duplicate {mechanism_family_id:?}")
            }
            Self::MechanismFamilyLeakage { mechanism_family_id } => {
                write!(f, "mechanism family {mechanism_family_id:?} appears in both in-scope and omitted calibration cohorts")
            }
            Self::NonFiniteNonconformityScore => {
                write!(f, "nonconformity scores must be finite")
            }
            Self::InvalidAdequacyScore => {
                write!(f, "held-out adequacy score must be finite")
            }
            Self::NumericalInstability => {
                write!(f, "entropy calculation exceeded its numerical tolerance")
            }
        }
    }
}

impl std::error::Error for DesignError {}

/// A candidate experiment's likelihood model.
///
/// likelihoods_by_hypothesis[h][y] is P(Y = outcome[y] | H = h, experiment).
/// Every row must be a normalized probability distribution. The outcome order
/// is shared across all hypotheses.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateExperiment {
    pub id: String,
    /// Caller-defined resource cost. Must be finite and strictly positive.
    pub cost: f64,
    pub outcomes: Vec<String>,
    pub likelihoods_by_hypothesis: Vec<Vec<f64>>,
}

/// Information score for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationGainScore {
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub information_gain_bits: f64,
    pub cost: f64,
    pub information_gain_bits_per_cost: f64,
}

/// Candidate and score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedExperiment {
    pub experiment_id: String,
    pub score: InformationGainScore,
}

/// Opportunity regret relative to the best candidate in a declared set.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub selected_utility_bits_per_cost: f64,
    pub best_utility_bits_per_cost: f64,
    pub regret_bits_per_cost: f64,
}

/// Reserved hypothesis identifier for mechanisms outside the named candidate set.
///
/// The caller must assign this state a positive prior and supply a calibrated
/// likelihood ambiguity set. This crate deliberately does not invent either.
pub const UNRESOLVED_HYPOTHESIS_ID: &str = "OTHER_OR_UNRESOLVED";

/// One admissible conditional-likelihood model for an open-world experiment.
///
/// The likelihood rows are ordered exactly as the hypothesis_ids supplied to
/// OpenWorldCandidateExperiment; columns share the candidate outcome order.
#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodScenario {
    pub id: String,
    pub likelihoods_by_hypothesis: Vec<Vec<f64>>,
}

/// Candidate experiment evaluated across a finite, explicitly declared
/// ambiguity set of possible conditional likelihoods.
///
/// The hypothesis list must contain exactly one OTHER_OR_UNRESOLVED state with
/// positive caller-supplied prior mass. The likelihood scenarios are not
/// learned or calibrated here; they must be prespecified from independent
/// calibration data, held-out mechanism families, or declared sensitivity
/// assumptions.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldCandidateExperiment {
    pub id: String,
    pub cost: f64,
    pub outcomes: Vec<String>,
    pub hypothesis_ids: Vec<String>,
    pub likelihood_scenarios: Vec<LikelihoodScenario>,
}

/// EIG under one admissible likelihood scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioInformationGain {
    pub scenario_id: String,
    pub information_gain_bits: f64,
    pub information_gain_bits_per_cost: f64,
}

/// Worst-/best-case EIG over the candidate's declared likelihood scenarios.
///
/// These are bounds only over the supplied finite ambiguity set. They are not
/// guarantees against arbitrary omitted mechanisms outside that set.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldInformationGainScore {
    pub experiment_id: String,
    pub cost: f64,
    pub unresolved_prior_mass: f64,
    pub worst_case_scenario_id: String,
    pub best_case_scenario_id: String,
    pub worst_case_information_gain_bits: f64,
    pub best_case_information_gain_bits: f64,
    pub worst_case_information_gain_bits_per_cost: f64,
    pub best_case_information_gain_bits_per_cost: f64,
    /// All scenario scores are sorted by scenario ID for deterministic output.
    pub scenario_scores: Vec<ScenarioInformationGain>,
}

/// Open-world experiment after ranking by worst-case EIG per unit cost.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedOpenWorldExperiment {
    pub experiment_id: String,
    pub score: OpenWorldInformationGainScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldExperimentReport {
    pub schema_version: &'static str,
    pub unresolved_hypothesis_id: &'static str,
    pub ranked_experiments: Vec<RankedOpenWorldExperiment>,
}

pub fn evaluate_open_world_report(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
) -> Result<OpenWorldExperimentReport, DesignError> {
    Ok(OpenWorldExperimentReport {
        schema_version: "connectome-open-world-eig-v1",
        unresolved_hypothesis_id: UNRESOLVED_HYPOTHESIS_ID,
        ranked_experiments: rank_open_world_candidates(priors, candidates)?,
    })
}

/// A scientific question represented as an answer class for every
/// candidate mechanism hypothesis, including OTHER_OR_UNRESOLVED.
///
/// The answer labels must encode the declared causal query (for example,
/// positive/negative/unresolved for a target intervention effect). They are
/// not free-form model descriptions. Hypothesis order must exactly match each
/// open-world candidate's hypothesis_ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetCausalQuery {
    pub id: String,
    pub hypothesis_ids: Vec<String>,
    pub answer_by_hypothesis: Vec<String>,
}

/// Query-specific information gain under one admissible likelihood scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioTargetQueryGain {
    pub scenario_id: String,
    pub information_gain_bits: f64,
    pub information_gain_bits_per_cost: f64,
}

/// Worst- and best-case information about a declared causal answer, over the
/// supplied finite likelihood ambiguity set.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldTargetQueryScore {
    pub experiment_id: String,
    pub query_id: String,
    pub cost: f64,
    pub unresolved_prior_mass: f64,
    pub worst_case_scenario_id: String,
    pub best_case_scenario_id: String,
    pub worst_case_information_gain_bits: f64,
    pub best_case_information_gain_bits: f64,
    pub worst_case_information_gain_bits_per_cost: f64,
    pub best_case_information_gain_bits_per_cost: f64,
    pub scenario_scores: Vec<ScenarioTargetQueryGain>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedTargetQueryExperiment {
    pub experiment_id: String,
    pub score: OpenWorldTargetQueryScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TargetQueryExperimentReport {
    pub schema_version: &'static str,
    pub query: TargetCausalQuery,
    pub ranked_experiments: Vec<RankedTargetQueryExperiment>,
}

/// Calculate I(Q;Y) for a categorical target answer Q and experimental
/// observation Y. Hypotheses that imply the same answer are marginalized
/// together; this prevents information about irrelevant within-answer
/// mechanism distinctions from dominating the scientific objective.
fn target_query_information_gain_bits(
    priors: &[f64],
    answer_by_hypothesis: &[String],
    likelihoods_by_hypothesis: &[Vec<f64>],
    outcome_count: usize,
) -> Result<f64, DesignError> {
    if answer_by_hypothesis.len() != priors.len() {
        return Err(DesignError::TargetQueryHypothesisCountMismatch {
            expected: priors.len(),
            actual: answer_by_hypothesis.len(),
        });
    }
    if likelihoods_by_hypothesis.len() != priors.len() {
        return Err(DesignError::LikelihoodRowCountMismatch {
            expected: priors.len(),
            actual: likelihoods_by_hypothesis.len(),
        });
    }

    let mut answer_labels: Vec<&str> = Vec::new();
    let mut answer_index_by_hypothesis = Vec::with_capacity(answer_by_hypothesis.len());
    for answer in answer_by_hypothesis {
        if answer.trim().is_empty() {
            return Err(DesignError::EmptyTargetQueryAnswer);
        }
        let index = match answer_labels.iter().position(|label| *label == answer.as_str()) {
            Some(index) => index,
            None => {
                answer_labels.push(answer.as_str());
                answer_labels.len() - 1
            }
        };
        answer_index_by_hypothesis.push(index);
    }

    let mut answer_priors = vec![0.0; answer_labels.len()];
    let mut joint_answer_outcome =
        vec![vec![0.0; outcome_count]; answer_labels.len()];
    let mut outcome_marginals = vec![0.0; outcome_count];

    for hypothesis_index in 0..priors.len() {
        let answer_index = answer_index_by_hypothesis[hypothesis_index];
        answer_priors[answer_index] += priors[hypothesis_index];

        let likelihood_row = &likelihoods_by_hypothesis[hypothesis_index];
        if likelihood_row.len() != outcome_count {
            return Err(DesignError::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected: outcome_count,
                actual: likelihood_row.len(),
            });
        }
        for outcome_index in 0..outcome_count {
            let joint = priors[hypothesis_index] * likelihood_row[outcome_index];
            joint_answer_outcome[answer_index][outcome_index] += joint;
            outcome_marginals[outcome_index] += joint;
        }
    }

    let prior_entropy = entropy_bits(&answer_priors);
    let mut expected_posterior_entropy = 0.0;
    for (outcome_index, &outcome_mass) in outcome_marginals.iter().enumerate() {
        if outcome_mass <= 0.0 {
            continue;
        }
        let posterior_answers: Vec<f64> = joint_answer_outcome
            .iter()
            .map(|row| row[outcome_index] / outcome_mass)
            .collect();
        expected_posterior_entropy += outcome_mass * entropy_bits(&posterior_answers);
    }

    let mut information_gain = prior_entropy - expected_posterior_entropy;
    if information_gain < 0.0 && information_gain >= -ROUND_OFF_TOLERANCE {
        information_gain = 0.0;
    }
    if information_gain > prior_entropy
        && information_gain <= prior_entropy + ROUND_OFF_TOLERANCE
    {
        information_gain = prior_entropy;
    }
    if !information_gain.is_finite()
        || information_gain < 0.0
        || information_gain > prior_entropy + ROUND_OFF_TOLERANCE
    {
        return Err(DesignError::NumericalInstability);
    }
    Ok(information_gain)
}

/// Evaluate a candidate against a target causal query rather than the full
/// mechanism identity. The result is worst-case query information across the
/// candidate's declared likelihood scenarios; it is not decision utility or
/// an expected treatment benefit.
pub fn evaluate_open_world_target_query(
    priors: &[f64],
    candidate: &OpenWorldCandidateExperiment,
    query: &TargetCausalQuery,
) -> Result<OpenWorldTargetQueryScore, DesignError> {
    if query.id.trim().is_empty() {
        return Err(DesignError::EmptyTargetQueryId);
    }
    if query.hypothesis_ids != candidate.hypothesis_ids {
        return Err(DesignError::InconsistentHypothesisOrder);
    }
    if query.answer_by_hypothesis.len() != candidate.hypothesis_ids.len() {
        return Err(DesignError::TargetQueryHypothesisCountMismatch {
            expected: candidate.hypothesis_ids.len(),
            actual: query.answer_by_hypothesis.len(),
        });
    }

    // Reuse the open-world validator. This checks the positive-prior
    // OTHER_OR_UNRESOLVED state, priors, candidate metadata and every scenario.
    let open_world_score = evaluate_open_world_candidate(priors, candidate)?;

    let mut scenario_scores = Vec::with_capacity(candidate.likelihood_scenarios.len());
    for scenario in &candidate.likelihood_scenarios {
        let information_gain_bits = target_query_information_gain_bits(
            priors,
            &query.answer_by_hypothesis,
            &scenario.likelihoods_by_hypothesis,
            candidate.outcomes.len(),
        )?;
        scenario_scores.push(ScenarioTargetQueryGain {
            scenario_id: scenario.id.clone(),
            information_gain_bits,
            information_gain_bits_per_cost: information_gain_bits / candidate.cost,
        });
    }
    scenario_scores.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut worst = &scenario_scores[0];
    let mut best = &scenario_scores[0];
    for score in &scenario_scores[1..] {
        if score.information_gain_bits_per_cost < worst.information_gain_bits_per_cost {
            worst = score;
        }
        if score.information_gain_bits_per_cost > best.information_gain_bits_per_cost {
            best = score;
        }
    }

    Ok(OpenWorldTargetQueryScore {
        experiment_id: candidate.id.clone(),
        query_id: query.id.clone(),
        cost: candidate.cost,
        unresolved_prior_mass: open_world_score.unresolved_prior_mass,
        worst_case_scenario_id: worst.scenario_id.clone(),
        best_case_scenario_id: best.scenario_id.clone(),
        worst_case_information_gain_bits: worst.information_gain_bits,
        best_case_information_gain_bits: best.information_gain_bits,
        worst_case_information_gain_bits_per_cost: worst.information_gain_bits_per_cost,
        best_case_information_gain_bits_per_cost: best.information_gain_bits_per_cost,
        scenario_scores,
    })
}

/// Rank experiments by worst-case information about a declared causal
/// question per unit cost. All candidates must share the query's exact
/// hypothesis identity and prior ordering.
pub fn rank_open_world_candidates_for_query(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    query: &TargetCausalQuery,
) -> Result<Vec<RankedTargetQueryExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    let reference_hypotheses = &candidates[0].hypothesis_ids;
    if query.hypothesis_ids != *reference_hypotheses {
        return Err(DesignError::InconsistentHypothesisOrder);
    }
    if candidates
        .iter()
        .any(|candidate| candidate.hypothesis_ids != *reference_hypotheses)
    {
        return Err(DesignError::InconsistentHypothesisOrder);
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedTargetQueryExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_open_world_target_query(priors, candidate, query)?,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .worst_case_information_gain_bits_per_cost
            .total_cmp(&left.score.worst_case_information_gain_bits_per_cost)
            .then_with(|| {
                right
                    .score
                    .worst_case_information_gain_bits
                    .total_cmp(&left.score.worst_case_information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Build a versioned goal-oriented experiment-design report.
pub fn evaluate_target_query_report(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    query: TargetCausalQuery,
) -> Result<TargetQueryExperimentReport, DesignError> {
    let ranked_experiments =
        rank_open_world_candidates_for_query(priors, candidates, &query)?;
    Ok(TargetQueryExperimentReport {
        schema_version: "connectome-target-query-eig-v1",
        query,
        ranked_experiments,
    })
}

/// A concrete action available after the scientific uncertainty is resolved.
///
/// Utilities are caller-supplied, outcome-independent values indexed by the
/// exact hypothesis order in DecisionProblem. Include a no-action/control
/// alternative when it is a genuine downstream option. These utilities are
/// not probabilities and may be negative.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAction {
    pub id: String,
    pub utility_by_hypothesis: Vec<f64>,
}

/// A decision problem declares the downstream actions and their utility under
/// every candidate hypothesis, including OTHER_OR_UNRESOLVED.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionProblem {
    pub id: String,
    pub hypothesis_ids: Vec<String>,
    pub actions: Vec<DecisionAction>,
}

/// The optimal downstream action after one observable outcome.
#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeDecision {
    pub outcome_id: String,
    pub outcome_probability: f64,
    pub selected_action_id: String,
    pub posterior_expected_utility: f64,
}

/// Decision value under one admissible likelihood scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioDecisionValue {
    pub scenario_id: String,
    pub current_best_action_id: String,
    pub current_expected_utility: f64,
    pub expected_post_experiment_utility: f64,
    pub expected_value_of_sample_information: f64,
    pub evsi_per_cost: f64,
    pub outcome_decisions: Vec<OutcomeDecision>,
}

/// Worst-/best-case expected value of sample information over the supplied
/// finite likelihood ambiguity set. This is a value of information, not the
/// expected biological benefit of administering the experiment itself.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldDecisionValueScore {
    pub experiment_id: String,
    pub decision_problem_id: String,
    pub cost: f64,
    pub unresolved_prior_mass: f64,
    pub worst_case_scenario_id: String,
    pub best_case_scenario_id: String,
    pub worst_case_evsi: f64,
    pub best_case_evsi: f64,
    pub worst_case_evsi_per_cost: f64,
    pub best_case_evsi_per_cost: f64,
    pub scenario_values: Vec<ScenarioDecisionValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedDecisionValueExperiment {
    pub experiment_id: String,
    pub score: OpenWorldDecisionValueScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionValueExperimentReport {
    pub schema_version: &'static str,
    pub decision_problem: DecisionProblem,
    pub ranked_experiments: Vec<RankedDecisionValueExperiment>,
}

fn validate_decision_problem(
    problem: &DecisionProblem,
    hypothesis_ids: &[String],
) -> Result<(), DesignError> {
    if problem.id.trim().is_empty() {
        return Err(DesignError::EmptyDecisionProblemId);
    }
    if problem.hypothesis_ids.as_slice() != hypothesis_ids {
        return Err(DesignError::InconsistentHypothesisOrder);
    }
    if problem.actions.is_empty() {
        return Err(DesignError::EmptyDecisionActionSet);
    }
    let mut action_ids = HashSet::with_capacity(problem.actions.len());
    for action in &problem.actions {
        if action.id.trim().is_empty() {
            return Err(DesignError::EmptyDecisionActionId);
        }
        if !action_ids.insert(action.id.as_str()) {
            return Err(DesignError::DuplicateDecisionActionId {
                action_id: action.id.clone(),
            });
        }
        if action.utility_by_hypothesis.len() != hypothesis_ids.len() {
            return Err(DesignError::DecisionUtilityCountMismatch {
                action_id: action.id.clone(),
                expected: hypothesis_ids.len(),
                actual: action.utility_by_hypothesis.len(),
            });
        }
        for (hypothesis_index, utility) in action.utility_by_hypothesis.iter().enumerate() {
            if !utility.is_finite() {
                return Err(DesignError::NonFiniteDecisionUtility {
                    action_id: action.id.clone(),
                    hypothesis_index,
                });
            }
        }
    }
    Ok(())
}

/// Evaluate the expected value of sample information (EVSI) for one experiment.
///
/// Current value is the maximum prior expected utility across downstream
/// actions. Post-experiment value averages, over observed outcomes, the maximum
/// posterior expected utility after choosing an action. The information value
/// is their difference. If information is unhelpful, the decision-maker can
/// ignore it, so EVSI cannot be materially negative.
///
/// Likelihood uncertainty is handled by calculating EVSI under every supplied
/// scenario and reporting/ranking the worst case. This is only robust over the
/// declared finite ambiguity set.
pub fn evaluate_open_world_decision_value(
    priors: &[f64],
    candidate: &OpenWorldCandidateExperiment,
    problem: &DecisionProblem,
) -> Result<OpenWorldDecisionValueScore, DesignError> {
    let open_world_score = evaluate_open_world_candidate(priors, candidate)?;
    validate_decision_problem(problem, &candidate.hypothesis_ids)?;

    let prior_mass: f64 = priors.iter().sum();
    if !prior_mass.is_finite() || prior_mass <= 0.0 {
        return Err(DesignError::NumericalInstability);
    }
    let normalized_priors: Vec<f64> = priors.iter().map(|prior| prior / prior_mass).collect();

    let mut current_best_action_id = String::new();
    let mut current_expected_utility = f64::NEG_INFINITY;
    for action in &problem.actions {
        let expected_utility: f64 = normalized_priors
            .iter()
            .zip(&action.utility_by_hypothesis)
            .map(|(prior, utility)| prior * utility)
            .sum();
        if !expected_utility.is_finite() {
            return Err(DesignError::NumericalInstability);
        }
        if expected_utility > current_expected_utility
            || (expected_utility == current_expected_utility
                && (current_best_action_id.is_empty() || action.id < current_best_action_id))
        {
            current_expected_utility = expected_utility;
            current_best_action_id = action.id.clone();
        }
    }

    let mut scenario_values = Vec::with_capacity(candidate.likelihood_scenarios.len());
    for scenario in &candidate.likelihood_scenarios {
        let outcome_count = candidate.outcomes.len();
        let mut outcome_marginals = vec![0.0f64; outcome_count];
        for (hypothesis_index, prior) in normalized_priors.iter().enumerate() {
            for outcome_index in 0..outcome_count {
                outcome_marginals[outcome_index] +=
                    prior * scenario.likelihoods_by_hypothesis[hypothesis_index][outcome_index];
            }
        }

        let total_outcome_mass: f64 = outcome_marginals.iter().sum();
        if !total_outcome_mass.is_finite() || total_outcome_mass <= 0.0 {
            return Err(DesignError::NumericalInstability);
        }
        let mut expected_post_experiment_utility = 0.0f64;
        let mut outcome_decisions = Vec::with_capacity(outcome_count);
        for (outcome_index, raw_outcome_probability) in outcome_marginals.iter().copied().enumerate() {
            let outcome_probability = raw_outcome_probability / total_outcome_mass;
            if outcome_probability <= 0.0 {
                continue;
            }
            let posterior: Vec<f64> = priors
                .iter()
                .enumerate()
                .map(|(hypothesis_index, prior)| {
                    prior * scenario.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                        / raw_outcome_probability
                })
                .collect();

            let mut selected_action_id = String::new();
            let mut posterior_best_utility = f64::NEG_INFINITY;
            for action in &problem.actions {
                let expected_utility: f64 = posterior
                    .iter()
                    .zip(&action.utility_by_hypothesis)
                    .map(|(probability, utility)| probability * utility)
                    .sum();
                if !expected_utility.is_finite() {
                    return Err(DesignError::NumericalInstability);
                }
                if expected_utility > posterior_best_utility
                    || (expected_utility == posterior_best_utility
                        && (selected_action_id.is_empty() || action.id < selected_action_id))
                {
                    posterior_best_utility = expected_utility;
                    selected_action_id = action.id.clone();
                }
            }

            expected_post_experiment_utility += outcome_probability * posterior_best_utility;
            if !expected_post_experiment_utility.is_finite() {
                return Err(DesignError::NumericalInstability);
            }
            outcome_decisions.push(OutcomeDecision {
                outcome_id: candidate.outcomes[outcome_index].clone(),
                outcome_probability,
                selected_action_id,
                posterior_expected_utility: posterior_best_utility,
            });
        }

        let mut evsi = expected_post_experiment_utility - current_expected_utility;
        if evsi < 0.0 && evsi >= -ROUND_OFF_TOLERANCE {
            evsi = 0.0;
        }
        if !evsi.is_finite() || evsi < 0.0 {
            return Err(DesignError::NumericalInstability);
        }
        scenario_values.push(ScenarioDecisionValue {
            scenario_id: scenario.id.clone(),
            current_best_action_id: current_best_action_id.clone(),
            current_expected_utility,
            expected_post_experiment_utility,
            expected_value_of_sample_information: evsi,
            evsi_per_cost: evsi / candidate.cost,
            outcome_decisions,
        });
    }
    scenario_values.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut worst = &scenario_values[0];
    let mut best = &scenario_values[0];
    for value in &scenario_values[1..] {
        if value.evsi_per_cost < worst.evsi_per_cost {
            worst = value;
        }
        if value.evsi_per_cost > best.evsi_per_cost {
            best = value;
        }
    }

    Ok(OpenWorldDecisionValueScore {
        experiment_id: candidate.id.clone(),
        decision_problem_id: problem.id.clone(),
        cost: candidate.cost,
        unresolved_prior_mass: open_world_score.unresolved_prior_mass,
        worst_case_scenario_id: worst.scenario_id.clone(),
        best_case_scenario_id: best.scenario_id.clone(),
        worst_case_evsi: worst.expected_value_of_sample_information,
        best_case_evsi: best.expected_value_of_sample_information,
        worst_case_evsi_per_cost: worst.evsi_per_cost,
        best_case_evsi_per_cost: best.evsi_per_cost,
        scenario_values,
    })
}

/// Rank candidate experiments by worst-case EVSI per unit cost.
pub fn rank_open_world_candidates_for_decision(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    problem: &DecisionProblem,
) -> Result<Vec<RankedDecisionValueExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    if candidates
        .iter()
        .any(|candidate| candidate.hypothesis_ids != problem.hypothesis_ids)
    {
        return Err(DesignError::InconsistentHypothesisOrder);
    }
    validate_decision_problem(problem, &problem.hypothesis_ids)?;

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedDecisionValueExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_open_world_decision_value(priors, candidate, problem)?,
        });
    }
    ranked.sort_by(|left, right| {
        right.score.worst_case_evsi_per_cost
            .total_cmp(&left.score.worst_case_evsi_per_cost)
            .then_with(|| {
                right.score.worst_case_evsi.total_cmp(&left.score.worst_case_evsi)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Build a versioned decision-focused experimental-design report.
pub fn evaluate_decision_value_report(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    decision_problem: DecisionProblem,
) -> Result<DecisionValueExperimentReport, DesignError> {
    let ranked_experiments =
        rank_open_world_candidates_for_decision(priors, candidates, &decision_problem)?;
    Ok(DecisionValueExperimentReport {
        schema_version: "connectome-decision-value-v1",
        decision_problem,
        ranked_experiments,
    })
}

/// Which side of a registered numeric threshold counts as a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitDirection {
    AtLeast,
    AtMost,
}

/// A preregistered threshold-hit target for the discrete outcomes of an
/// experiment. Outcome values are aligned with outcome_ids, which must exactly
/// match the candidate's outcome ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct ThresholdHitObjective {
    pub id: String,
    pub outcome_ids: Vec<String>,
    pub outcome_values: Vec<f64>,
    pub threshold: f64,
    pub direction: HitDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioHitProbability {
    pub scenario_id: String,
    pub probability_of_hit: f64,
    pub probability_of_hit_per_cost: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenWorldHitProbabilityScore {
    pub experiment_id: String,
    pub objective_id: String,
    pub cost: f64,
    pub unresolved_prior_mass: f64,
    pub worst_case_scenario_id: String,
    pub best_case_scenario_id: String,
    pub worst_case_probability_of_hit: f64,
    pub best_case_probability_of_hit: f64,
    pub worst_case_probability_of_hit_per_cost: f64,
    pub best_case_probability_of_hit_per_cost: f64,
    pub scenario_probabilities: Vec<ScenarioHitProbability>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedHitExperiment {
    pub experiment_id: String,
    pub score: OpenWorldHitProbabilityScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HitProbabilityReport {
    pub schema_version: &'static str,
    pub objective: ThresholdHitObjective,
    pub ranked_experiments: Vec<RankedHitExperiment>,
}

/// Evaluate the probability that an experiment outcome crosses a declared
/// threshold. The mixture probability averages over the caller's open-world
/// prior and each likelihood scenario. This is a hit-acquisition objective,
/// not model EIG, EVSI, or a biological utility guarantee.
pub fn evaluate_open_world_hit_probability(
    priors: &[f64],
    candidate: &OpenWorldCandidateExperiment,
    objective: &ThresholdHitObjective,
) -> Result<OpenWorldHitProbabilityScore, DesignError> {
    if objective.id.trim().is_empty() {
        return Err(DesignError::EmptyHitObjectiveId);
    }
    if !objective.threshold.is_finite() {
        return Err(DesignError::NonFiniteHitThreshold);
    }
    if objective.outcome_ids != candidate.outcomes {
        return Err(DesignError::InconsistentHitOutcomeOrder);
    }
    if objective.outcome_values.len() != candidate.outcomes.len() {
        return Err(DesignError::HitOutcomeCountMismatch {
            expected: candidate.outcomes.len(),
            actual: objective.outcome_values.len(),
        });
    }
    for (outcome_index, value) in objective.outcome_values.iter().enumerate() {
        if !value.is_finite() {
            return Err(DesignError::NonFiniteHitOutcomeValue { outcome_index });
        }
    }
    let open_world_score = evaluate_open_world_candidate(priors, candidate)?;

    let hit_indices: Vec<usize> = objective
        .outcome_values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let is_hit = match objective.direction {
                HitDirection::AtLeast => *value >= objective.threshold,
                HitDirection::AtMost => *value <= objective.threshold,
            };
            is_hit.then_some(index)
        })
        .collect();

    let mut scenario_probabilities = Vec::with_capacity(candidate.likelihood_scenarios.len());
    for scenario in &candidate.likelihood_scenarios {
        let mut probability_of_hit = 0.0f64;
        for (hypothesis_index, prior) in priors.iter().enumerate() {
            for outcome_index in &hit_indices {
                probability_of_hit +=
                    prior * scenario.likelihoods_by_hypothesis[hypothesis_index][*outcome_index];
            }
        }
        if !probability_of_hit.is_finite()
            || probability_of_hit < -ROUND_OFF_TOLERANCE
            || probability_of_hit > 1.0 + PROBABILITY_TOLERANCE
        {
            return Err(DesignError::NumericalInstability);
        }
        let probability_of_hit = probability_of_hit.clamp(0.0, 1.0);
        scenario_probabilities.push(ScenarioHitProbability {
            scenario_id: scenario.id.clone(),
            probability_of_hit,
            probability_of_hit_per_cost: probability_of_hit / candidate.cost,
        });
    }
    scenario_probabilities.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));

    let mut worst = &scenario_probabilities[0];
    let mut best = &scenario_probabilities[0];
    for score in &scenario_probabilities[1..] {
        if score.probability_of_hit_per_cost < worst.probability_of_hit_per_cost {
            worst = score;
        }
        if score.probability_of_hit_per_cost > best.probability_of_hit_per_cost {
            best = score;
        }
    }

    Ok(OpenWorldHitProbabilityScore {
        experiment_id: candidate.id.clone(),
        objective_id: objective.id.clone(),
        cost: candidate.cost,
        unresolved_prior_mass: open_world_score.unresolved_prior_mass,
        worst_case_scenario_id: worst.scenario_id.clone(),
        best_case_scenario_id: best.scenario_id.clone(),
        worst_case_probability_of_hit: worst.probability_of_hit,
        best_case_probability_of_hit: best.probability_of_hit,
        worst_case_probability_of_hit_per_cost: worst.probability_of_hit_per_cost,
        best_case_probability_of_hit_per_cost: best.probability_of_hit_per_cost,
        scenario_probabilities,
    })
}

/// Rank experiments by worst-case threshold-hit probability per unit cost.
/// Candidate outcome IDs must share the exact registered objective ordering.
pub fn rank_open_world_candidates_for_hit(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    objective: &ThresholdHitObjective,
) -> Result<Vec<RankedHitExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedHitExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_open_world_hit_probability(priors, candidate, objective)?,
        });
    }
    ranked.sort_by(|left, right| {
        right.score.worst_case_probability_of_hit_per_cost
            .total_cmp(&left.score.worst_case_probability_of_hit_per_cost)
            .then_with(|| {
                right.score.worst_case_probability_of_hit
                    .total_cmp(&left.score.worst_case_probability_of_hit)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Build a versioned threshold-hit report.
pub fn evaluate_hit_probability_report(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
    objective: ThresholdHitObjective,
) -> Result<HitProbabilityReport, DesignError> {
    let ranked_experiments =
        rank_open_world_candidates_for_hit(priors, candidates, &objective)?;
    Ok(HitProbabilityReport {
        schema_version: "connectome-hit-probability-v1",
        objective,
        ranked_experiments,
    })
}

/// One independent mechanism-family aggregate used to calibrate model adequacy.
///
/// Both in-scope and omitted cohorts use exactly one unit per independent
/// mechanism family. Each unit's nonconformity score must be aggregated from
/// that family's animal/session recordings using a preregistered rule. Frames,
/// neurons, trials, and repeated animal/session recordings within a family are
/// not independent calibration units. The held-out test score must use the same
/// family-level aggregation rule.
#[derive(Debug, Clone, PartialEq)]
pub struct AdequacyCalibrationUnit {
    pub unit_id: String,
    pub mechanism_family_id: String,
    /// Higher values must mean worse predictive fit (for example, normalized
    /// negative log predictive density), with normalization fixed in advance.
    pub nonconformity_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdequacyGateConfig {
    /// Target upper bound for false ADEQUATE declarations on exchangeable
    /// omitted-mechanism calibration units.
    pub false_adequacy_alpha: f64,
    /// Minimum one-sided lower confidence bound on in-scope acceptance.
    pub minimum_in_scope_acceptance: f64,
    /// One-sided Wilson lower-bound critical value.
    pub wilson_z_critical: f64,
}

impl Default for AdequacyGateConfig {
    fn default() -> Self {
        Self {
            false_adequacy_alpha: 0.05,
            minimum_in_scope_acceptance: 0.80,
            wilson_z_critical: 1.644_853_626_951_472_2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdequacyAcceptanceCalibration {
    pub accepted_in_scope_units: usize,
    pub total_in_scope_units: usize,
    pub observed_acceptance_rate: f64,
    pub one_sided_wilson_lower_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdequacyCalibrator {
    pub false_adequacy_alpha: f64,
    pub minimum_in_scope_acceptance: f64,
    pub wilson_z_critical: f64,
    pub in_scope_unit_count: usize,
    /// One independent aggregate score per omitted mechanism family.
    pub omitted_mechanism_family_count: usize,
    /// Lower-tail conformal order statistic, one-based; zero means no usable
    /// threshold can be formed at the requested false-adequacy alpha.
    pub lower_order_statistic_k: usize,
    /// A candidate is adequate only when its nonconformity score is strictly
    /// less than this threshold. Strict comparison is conservative under ties.
    pub nonconformity_threshold: Option<f64>,
    pub false_adequacy_probability_bound: f64,
    pub in_scope_acceptance: Option<AdequacyAcceptanceCalibration>,
    /// False unless the threshold exists and the in-scope lower confidence
    /// bound meets the preregistered useful-acceptance floor.
    pub usable: bool,
}

/// Versioned serializable shape for an adequacy calibration result.
#[derive(Debug, Clone, PartialEq)]
pub struct AdequacyCalibrationReport {
    pub schema_version: &'static str,
    pub calibrator: AdequacyCalibrator,
}

/// A high nonconformity score indicates that the candidate set plus its
/// observation model is inadequate or shifted under the declared calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdequacyDecision {
    AdequateUnderCalibration,
    InadequateOrDistributionShift,
    Unresolved,
}

fn wilson_one_sided_lower(
    successes: usize,
    trials: usize,
    z_critical: f64,
) -> Result<f64, DesignError> {
    if trials == 0 || successes > trials || !z_critical.is_finite() || z_critical <= 0.0 {
        return Err(DesignError::InvalidConfig);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z_critical * z_critical;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radicand = p * (1.0 - p) / n + z2 / (4.0 * n * n);
    let lower = (center - z_critical * radicand.sqrt() / denominator).max(0.0);
    if !lower.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(lower)
}

/// Calibrate an open-world adequacy gate from one aggregate score per
/// independent in-scope mechanism family and one per omitted mechanism family.
///
/// The threshold is the k-th smallest omitted-family nonconformity score,
/// where k = floor(alpha * (n_ood + 1)). A candidate is accepted only when its
/// score is strictly smaller than that threshold. Under exchangeability with
/// the omitted-family calibration population, and with ties handled strictly,
/// the false-adequacy probability is bounded by k/(n_ood + 1).
///
/// This guarantee does not transport automatically to arbitrary future
/// mechanisms. The omitted calibration families must be a defensible sample of
/// the mechanism shifts against which the gate will be used. If k=0, or if the
/// in-scope one-sided Wilson lower bound misses its useful-acceptance floor,
/// the calibrator is not usable and all test decisions are UNRESOLVED.
pub fn calibrate_model_adequacy(
    in_scope_units: &[AdequacyCalibrationUnit],
    omitted_mechanism_units: &[AdequacyCalibrationUnit],
    config: AdequacyGateConfig,
) -> Result<AdequacyCalibrator, DesignError> {
    if in_scope_units.is_empty() || omitted_mechanism_units.is_empty() {
        return Err(DesignError::EmptyCalibrationCohort);
    }
    if !config.false_adequacy_alpha.is_finite()
        || config.false_adequacy_alpha <= 0.0
        || config.false_adequacy_alpha >= 1.0
        || !config.minimum_in_scope_acceptance.is_finite()
        || config.minimum_in_scope_acceptance <= 0.0
        || config.minimum_in_scope_acceptance >= 1.0
        || !config.wilson_z_critical.is_finite()
        || config.wilson_z_critical <= 0.0
    {
        return Err(DesignError::InvalidConfig);
    }

    let mut all_unit_ids = HashSet::with_capacity(in_scope_units.len() + omitted_mechanism_units.len());
    let mut in_scope_families = HashSet::new();
    let mut omitted_families = HashSet::new();

    for unit in in_scope_units {
        if unit.unit_id.trim().is_empty() {
            return Err(DesignError::EmptyCalibrationUnitId);
        }
        if unit.mechanism_family_id.trim().is_empty() {
            return Err(DesignError::EmptyMechanismFamilyId);
        }
        if !unit.nonconformity_score.is_finite() {
            return Err(DesignError::NonFiniteNonconformityScore);
        }
        if !all_unit_ids.insert(unit.unit_id.as_str()) {
            return Err(DesignError::DuplicateCalibrationUnitId {
                unit_id: unit.unit_id.clone(),
            });
        }
        if !in_scope_families.insert(unit.mechanism_family_id.as_str()) {
            return Err(DesignError::DuplicateInScopeMechanismFamily {
                mechanism_family_id: unit.mechanism_family_id.clone(),
            });
        }
    }

    for unit in omitted_mechanism_units {
        if unit.unit_id.trim().is_empty() {
            return Err(DesignError::EmptyCalibrationUnitId);
        }
        if unit.mechanism_family_id.trim().is_empty() {
            return Err(DesignError::EmptyMechanismFamilyId);
        }
        if !unit.nonconformity_score.is_finite() {
            return Err(DesignError::NonFiniteNonconformityScore);
        }
        if !all_unit_ids.insert(unit.unit_id.as_str()) {
            return Err(DesignError::DuplicateCalibrationUnitId {
                unit_id: unit.unit_id.clone(),
            });
        }
        if !omitted_families.insert(unit.mechanism_family_id.as_str()) {
            return Err(DesignError::DuplicateOmittedMechanismFamily {
                mechanism_family_id: unit.mechanism_family_id.clone(),
            });
        }
    }

    for family_id in omitted_families {
        if in_scope_families.contains(&family_id) {
            return Err(DesignError::MechanismFamilyLeakage {
                mechanism_family_id: family_id.to_owned(),
            });
        }
    }

    let mut omitted_scores: Vec<f64> = omitted_mechanism_units
        .iter()
        .map(|unit| unit.nonconformity_score)
        .collect();
    omitted_scores.sort_by(f64::total_cmp);

    let n_ood = omitted_scores.len();
    let k = (config.false_adequacy_alpha * (n_ood as f64 + 1.0)).floor() as usize;
    let threshold = if k == 0 {
        None
    } else {
        Some(omitted_scores[k - 1])
    };
    let false_adequacy_probability_bound = k as f64 / (n_ood as f64 + 1.0);

    let in_scope_acceptance = if let Some(threshold) = threshold {
        let accepted = in_scope_units
            .iter()
            .filter(|unit| unit.nonconformity_score < threshold)
            .count();
        let total = in_scope_units.len();
        let lower = wilson_one_sided_lower(accepted, total, config.wilson_z_critical)?;
        Some(AdequacyAcceptanceCalibration {
            accepted_in_scope_units: accepted,
            total_in_scope_units: total,
            observed_acceptance_rate: accepted as f64 / total as f64,
            one_sided_wilson_lower_bound: lower,
        })
    } else {
        None
    };

    let usable = in_scope_acceptance.as_ref().is_some_and(|rate| {
        rate.one_sided_wilson_lower_bound >= config.minimum_in_scope_acceptance
    });

    Ok(AdequacyCalibrator {
        false_adequacy_alpha: config.false_adequacy_alpha,
        minimum_in_scope_acceptance: config.minimum_in_scope_acceptance,
        wilson_z_critical: config.wilson_z_critical,
        in_scope_unit_count: in_scope_units.len(),
        omitted_mechanism_family_count: n_ood,
        lower_order_statistic_k: k,
        nonconformity_threshold: threshold,
        false_adequacy_probability_bound,
        in_scope_acceptance,
        usable,
    })
}

/// Build the versioned output shape for an adequacy calibration.
pub fn calibrate_model_adequacy_report(
    in_scope_units: &[AdequacyCalibrationUnit],
    omitted_mechanism_units: &[AdequacyCalibrationUnit],
    config: AdequacyGateConfig,
) -> Result<AdequacyCalibrationReport, DesignError> {
    Ok(AdequacyCalibrationReport {
        schema_version: "connectome-open-world-adequacy-v1",
        calibrator: calibrate_model_adequacy(
            in_scope_units,
            omitted_mechanism_units,
            config,
        )?,
    })
}

/// Classify one already-aggregated held-out mechanism-family score. This does
/// not name the omitted mechanism and does not distinguish an absent mechanism
/// from observation-kernel, registration, or context-distribution shift.
pub fn assess_model_adequacy(
    calibrator: &AdequacyCalibrator,
    nonconformity_score: f64,
) -> Result<AdequacyDecision, DesignError> {
    if !nonconformity_score.is_finite() {
        return Err(DesignError::InvalidAdequacyScore);
    }
    if !calibrator.usable {
        return Ok(AdequacyDecision::Unresolved);
    }
    let threshold = calibrator
        .nonconformity_threshold
        .ok_or(DesignError::NumericalInstability)?;
    Ok(if nonconformity_score < threshold {
        AdequacyDecision::AdequateUnderCalibration
    } else {
        AdequacyDecision::InadequateOrDistributionShift
    })
}

/// Shannon entropy in bits. Input probabilities must be non-negative and sum
/// to one; all production calls validate this before evaluating entropy.
pub fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| -probability * probability.log2())
        .sum()
}

fn validate_priors(priors: &[f64]) -> Result<(), DesignError> {
    if priors.is_empty() {
        return Err(DesignError::EmptyHypothesisSet);
    }

    let mut sum = 0.0;
    for (index, &prior) in priors.iter().enumerate() {
        if !prior.is_finite() || prior < 0.0 {
            return Err(DesignError::InvalidPrior { index, value: prior });
        }
        sum += prior;
    }

    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DesignError::PriorMassNotOne { sum });
    }
    Ok(())
}

fn validate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<usize, DesignError> {
    validate_priors(priors)?;

    if candidate.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost { cost: candidate.cost });
    }
    if candidate.outcomes.is_empty() {
        return Err(DesignError::EmptyOutcomeSet);
    }

    let mut labels = HashSet::with_capacity(candidate.outcomes.len());
    for label in &candidate.outcomes {
        if !labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateOutcomeLabel { label: label.clone() });
        }
    }

    if candidate.likelihoods_by_hypothesis.len() != priors.len() {
        return Err(DesignError::LikelihoodRowCountMismatch {
            expected: priors.len(),
            actual: candidate.likelihoods_by_hypothesis.len(),
        });
    }

    let outcome_count = candidate.outcomes.len();
    for (hypothesis_index, row) in candidate.likelihoods_by_hypothesis.iter().enumerate() {
        if row.len() != outcome_count {
            return Err(DesignError::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected: outcome_count,
                actual: row.len(),
            });
        }

        let mut row_sum = 0.0;
        for (outcome_index, &likelihood) in row.iter().enumerate() {
            if !likelihood.is_finite() || likelihood < 0.0 {
                return Err(DesignError::InvalidLikelihood {
                    hypothesis_index,
                    outcome_index,
                    value: likelihood,
                });
            }
            row_sum += likelihood;
        }

        if !row_sum.is_finite() || (row_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(DesignError::LikelihoodMassNotOne {
                hypothesis_index,
                sum: row_sum,
            });
        }
    }

    Ok(outcome_count)
}

/// Calculate EIG exactly for a finite discrete hypothesis and outcome space.
pub fn evaluate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<InformationGainScore, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    let prior_entropy = entropy_bits(priors);

    // P(y) = sum_h P(h) P(y | h, e).
    let mut outcome_marginals = vec![0.0; outcome_count];
    for (hypothesis_index, &prior) in priors.iter().enumerate() {
        for (outcome_index, &likelihood) in candidate.likelihoods_by_hypothesis[hypothesis_index]
            .iter()
            .enumerate()
        {
            outcome_marginals[outcome_index] += prior * likelihood;
        }
    }

    // E_y[H(H | y, e)] using Bayes' rule.
    let mut expected_posterior_entropy = 0.0;
    for (outcome_index, &marginal) in outcome_marginals.iter().enumerate() {
        if marginal <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = priors
            .iter()
            .enumerate()
            .map(|(hypothesis_index, &prior)| {
                prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                    / marginal
            })
            .collect();

        expected_posterior_entropy += marginal * entropy_bits(&posterior);
    }

    let mut information_gain = prior_entropy - expected_posterior_entropy;
    if information_gain < 0.0 && information_gain >= -ROUND_OFF_TOLERANCE {
        information_gain = 0.0;
    }
    if information_gain > prior_entropy
        && information_gain <= prior_entropy + ROUND_OFF_TOLERANCE
    {
        information_gain = prior_entropy;
    }
    if !information_gain.is_finite()
        || information_gain < 0.0
        || information_gain > prior_entropy + ROUND_OFF_TOLERANCE
    {
        return Err(DesignError::NumericalInstability);
    }

    Ok(InformationGainScore {
        prior_entropy_bits: prior_entropy,
        expected_posterior_entropy_bits: expected_posterior_entropy,
        information_gain_bits: information_gain,
        cost: candidate.cost,
        information_gain_bits_per_cost: information_gain / candidate.cost,
    })
}

/// Return P(Y = outcomes[outcome_index] | experiment).
pub fn outcome_probability(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<f64, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    if outcome_index >= outcome_count {
        return Err(DesignError::OutcomeIndexOutOfRange {
            index: outcome_index,
            outcome_count,
        });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
        })
        .sum())
}

/// Update the hypothesis prior after observing one candidate outcome.
pub fn posterior_for_observed_outcome(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<Vec<f64>, DesignError> {
    let marginal = outcome_probability(priors, candidate, outcome_index)?;
    if marginal <= 0.0 || !marginal.is_finite() {
        return Err(DesignError::ImpossibleObservedOutcome { outcome_index });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                / marginal
        })
        .collect())
}

/// Rank by EIG per cost, then raw EIG, then ascending ID. The final ID
/// tie-break makes output stable regardless of input order.
pub fn rank_candidates(
    priors: &[f64],
    candidates: &[CandidateExperiment],
) -> Result<Vec<RankedExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_candidate(priors, candidate)?,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .information_gain_bits_per_cost
            .total_cmp(&left.score.information_gain_bits_per_cost)
            .then_with(|| {
                right
                    .score
                    .information_gain_bits
                    .total_cmp(&left.score.information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(ranked)
}

/// Evaluate a candidate using the minimum EIG across its declared
/// likelihood ambiguity scenarios.
///
/// This function refuses to silently collapse the unresolved hypothesis into
/// the named hypotheses: OTHER_OR_UNRESOLVED must be present and receive a
/// positive prior. It also refuses an empty or malformed ambiguity set.
/// The output is robust only to the supplied scenario set, not to every
/// possible missing mechanism.
pub fn evaluate_open_world_candidate(
    priors: &[f64],
    candidate: &OpenWorldCandidateExperiment,
) -> Result<OpenWorldInformationGainScore, DesignError> {
    validate_priors(priors)?;

    if candidate.hypothesis_ids.len() != priors.len() {
        return Err(DesignError::PriorCountMismatch {
            expected: priors.len(),
            actual: candidate.hypothesis_ids.len(),
        });
    }

    let mut hypothesis_ids = HashSet::with_capacity(candidate.hypothesis_ids.len());
    let mut unresolved_index = None;
    for (index, hypothesis_id) in candidate.hypothesis_ids.iter().enumerate() {
        if hypothesis_id.trim().is_empty() {
            return Err(DesignError::EmptyHypothesisId);
        }
        if !hypothesis_ids.insert(hypothesis_id.as_str()) {
            return Err(DesignError::DuplicateHypothesisId {
                hypothesis_id: hypothesis_id.clone(),
            });
        }
        if hypothesis_id.as_str() == UNRESOLVED_HYPOTHESIS_ID {
            unresolved_index = Some(index);
        }
    }

    let unresolved_index = unresolved_index.ok_or(DesignError::MissingUnresolvedHypothesis)?;
    let unresolved_prior_mass = priors[unresolved_index];
    if unresolved_prior_mass <= 0.0 {
        return Err(DesignError::ZeroUnresolvedPrior);
    }

    if candidate.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost { cost: candidate.cost });
    }
    if candidate.outcomes.is_empty() {
        return Err(DesignError::EmptyOutcomeSet);
    }
    if candidate.likelihood_scenarios.is_empty() {
        return Err(DesignError::EmptyLikelihoodScenarioSet);
    }

    let mut scenario_ids = HashSet::with_capacity(candidate.likelihood_scenarios.len());
    let mut scenario_scores = Vec::with_capacity(candidate.likelihood_scenarios.len());
    for scenario in &candidate.likelihood_scenarios {
        if scenario.id.trim().is_empty() {
            return Err(DesignError::EmptyLikelihoodScenarioId);
        }
        if !scenario_ids.insert(scenario.id.as_str()) {
            return Err(DesignError::DuplicateLikelihoodScenarioId {
                scenario_id: scenario.id.clone(),
            });
        }

        let discrete_candidate = CandidateExperiment {
            id: candidate.id.clone(),
            cost: candidate.cost,
            outcomes: candidate.outcomes.clone(),
            likelihoods_by_hypothesis: scenario.likelihoods_by_hypothesis.clone(),
        };
        let score = evaluate_candidate(priors, &discrete_candidate)?;
        scenario_scores.push(ScenarioInformationGain {
            scenario_id: scenario.id.clone(),
            information_gain_bits: score.information_gain_bits,
            information_gain_bits_per_cost: score.information_gain_bits_per_cost,
        });
    }

    scenario_scores.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut worst = &scenario_scores[0];
    let mut best = &scenario_scores[0];
    for score in &scenario_scores[1..] {
        if score.information_gain_bits_per_cost < worst.information_gain_bits_per_cost {
            worst = score;
        }
        if score.information_gain_bits_per_cost > best.information_gain_bits_per_cost {
            best = score;
        }
    }

    Ok(OpenWorldInformationGainScore {
        experiment_id: candidate.id.clone(),
        cost: candidate.cost,
        unresolved_prior_mass,
        worst_case_scenario_id: worst.scenario_id.clone(),
        best_case_scenario_id: best.scenario_id.clone(),
        worst_case_information_gain_bits: worst.information_gain_bits,
        best_case_information_gain_bits: best.information_gain_bits,
        worst_case_information_gain_bits_per_cost: worst.information_gain_bits_per_cost,
        best_case_information_gain_bits_per_cost: best.information_gain_bits_per_cost,
        scenario_scores,
    })
}

/// Rank open-world experiments by worst-case EIG per unit cost, then worst-case
/// raw EIG, then stable experiment ID. All candidates must use the same
/// hypothesis identities and prior ordering.
pub fn rank_open_world_candidates(
    priors: &[f64],
    candidates: &[OpenWorldCandidateExperiment],
) -> Result<Vec<RankedOpenWorldExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }

    let reference_hypotheses = &candidates[0].hypothesis_ids;
    if candidates
        .iter()
        .any(|candidate| candidate.hypothesis_ids != *reference_hypotheses)
    {
        return Err(DesignError::InconsistentHypothesisOrder);
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedOpenWorldExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_open_world_candidate(priors, candidate)?,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .worst_case_information_gain_bits_per_cost
            .total_cmp(&left.score.worst_case_information_gain_bits_per_cost)
            .then_with(|| {
                right
                    .score
                    .worst_case_information_gain_bits
                    .total_cmp(&left.score.worst_case_information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(ranked)
}

/// Compute opportunity regret under the EIG-per-cost objective. This is
/// regret relative to the best experiment in this finite candidate set, not
/// behavioural utility or real-world scientific value.
pub fn selection_regret(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    selected_experiment_id: &str,
) -> Result<SelectionRegret, DesignError> {
    let ranked = rank_candidates(priors, candidates)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;

    let best_utility = best.score.information_gain_bits_per_cost;
    let selected_utility = selected.score.information_gain_bits_per_cost;
    let regret = (best_utility - selected_utility).max(0.0);

    Ok(SelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        selected_utility_bits_per_cost: selected_utility,
        best_utility_bits_per_cost: best_utility,
        regret_bits_per_cost: regret,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_experiment(
        id: &str,
        cost: f64,
        response_given_h0: f64,
        response_given_h1: f64,
    ) -> CandidateExperiment {
        CandidateExperiment {
            id: id.to_owned(),
            cost,
            outcomes: vec!["response".to_owned(), "no_response".to_owned()],
            likelihoods_by_hypothesis: vec![
                vec![response_given_h0, 1.0 - response_given_h0],
                vec![response_given_h1, 1.0 - response_given_h1],
            ],
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15}, expected {expected:.15}, tolerance {tolerance}"
        );
    }

    #[test]
    fn uninformative_experiment_has_zero_eig() {
        let experiment = binary_experiment("uninformative", 1.0, 0.5, 0.5);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.prior_entropy_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 1.0, 1.0e-12);
        close(score.information_gain_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_yields_one_bit_for_equal_binary_prior() {
        let experiment = CandidateExperiment {
            id: "perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["left".to_owned(), "right".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn noisy_ninety_ten_discriminator_matches_analytic_eig() {
        let experiment = binary_experiment("noisy", 1.0, 0.9, 0.1);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 0.531_004_406_410_718_8, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_reduces_asymmetric_prior_to_zero_entropy() {
        let experiment = CandidateExperiment {
            id: "asymmetric-perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["h0".to_owned(), "h1".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.8, 0.2], &experiment).unwrap();
        close(score.prior_entropy_bits, 0.721_928_094_887_362_3, 1.0e-12);
        close(score.information_gain_bits, score.prior_entropy_bits, 1.0e-12);
    }

    #[test]
    fn posterior_update_uses_bayes_rule() {
        let experiment = binary_experiment("bayes", 1.0, 0.8, 0.2);
        let posterior = posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 0).unwrap();
        close(posterior[0], 0.8, 1.0e-12);
        close(posterior[1], 0.2, 1.0e-12);
    }

    #[test]
    fn cost_aware_ranking_can_prefer_less_raw_information() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let ranked = rank_candidates(&[0.5, 0.5], &[expensive, cheap]).unwrap();
        assert_eq!(ranked[0].experiment_id, "cheap");
        assert!(ranked[1].score.information_gain_bits > ranked[0].score.information_gain_bits);
        assert!(
            ranked[0].score.information_gain_bits_per_cost
                > ranked[1].score.information_gain_bits_per_cost
        );
    }

    #[test]
    fn candidate_order_does_not_change_ranked_tie_break() {
        let alpha = binary_experiment("alpha", 1.0, 0.9, 0.1);
        let zeta = binary_experiment("zeta", 1.0, 0.9, 0.1);
        let first = rank_candidates(&[0.5, 0.5], &[zeta.clone(), alpha.clone()]).unwrap();
        let second = rank_candidates(&[0.5, 0.5], &[alpha, zeta]).unwrap();
        assert_eq!(
            first.iter().map(|item| &item.experiment_id).collect::<Vec<_>>(),
            second.iter().map(|item| &item.experiment_id).collect::<Vec<_>>()
        );
        assert_eq!(first[0].experiment_id, "alpha");
    }

    #[test]
    fn selection_regret_is_zero_for_best_candidate_and_positive_otherwise() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let best = selection_regret(&[0.5, 0.5], &[expensive.clone(), cheap.clone()], "cheap")
            .unwrap();
        close(best.regret_bits_per_cost, 0.0, 1.0e-12);
        let regret = selection_regret(&[0.5, 0.5], &[expensive, cheap], "expensive").unwrap();
        assert!(regret.regret_bits_per_cost > 0.0);
        assert_eq!(regret.best_experiment_id, "cheap");
    }

    #[test]
    fn malformed_prior_fails_closed_without_renormalization() {
        let experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        assert_eq!(
            evaluate_candidate(&[0.4, 0.4], &experiment),
            Err(DesignError::PriorMassNotOne { sum: 0.8 })
        );
    }

    #[test]
    fn malformed_likelihood_fails_closed_without_renormalization() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0] = vec![0.9, 0.2];
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::LikelihoodMassNotOne { hypothesis_index: 0, .. })
        ));
    }

    #[test]
    fn non_finite_probability_and_non_positive_cost_are_rejected() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0][0] = f64::NAN;
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::InvalidLikelihood {
                hypothesis_index: 0,
                outcome_index: 0,
                ..
            })
        ));
        let bad_cost = binary_experiment("bad-cost", 0.0, 0.9, 0.1);
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &bad_cost),
            Err(DesignError::InvalidExperimentCost { .. })
        ));
    }

    #[test]
    fn impossible_observed_outcome_has_no_arbitrary_posterior() {
        let experiment = CandidateExperiment {
            id: "impossible".to_owned(),
            cost: 1.0,
            outcomes: vec!["only".to_owned(), "never".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        assert_eq!(
            posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 1),
            Err(DesignError::ImpossibleObservedOutcome { outcome_index: 1 })
        );
    }

    #[test]
    fn duplicate_outcomes_and_unknown_selection_fail_closed() {
        let mut duplicate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        duplicate.outcomes[1] = duplicate.outcomes[0].clone();
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &duplicate),
            Err(DesignError::DuplicateOutcomeLabel { .. })
        ));

        let valid = binary_experiment("valid", 1.0, 0.9, 0.1);
        assert!(matches!(
            selection_regret(&[0.5, 0.5], &[valid], "missing"),
            Err(DesignError::SelectedExperimentNotFound { .. })
        ));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let first = binary_experiment("same", 1.0, 0.9, 0.1);
        let second = binary_experiment("same", 2.0, 0.8, 0.2);
        assert!(matches!(
            rank_candidates(&[0.5, 0.5], &[first, second]),
            Err(DesignError::DuplicateExperimentId { .. })
        ));
    }

    fn open_world_candidate(
        id: &str,
        cost: f64,
        scenario_a: Vec<Vec<f64>>,
        scenario_b: Vec<Vec<f64>>,
    ) -> OpenWorldCandidateExperiment {
        OpenWorldCandidateExperiment {
            id: id.to_owned(),
            cost,
            outcomes: vec!["response".to_owned(), "no_response".to_owned()],
            hypothesis_ids: vec![
                "mechanism_a".to_owned(),
                "mechanism_b".to_owned(),
                UNRESOLVED_HYPOTHESIS_ID.to_owned(),
            ],
            likelihood_scenarios: vec![
                LikelihoodScenario {
                    id: "calibration-a".to_owned(),
                    likelihoods_by_hypothesis: scenario_a,
                },
                LikelihoodScenario {
                    id: "calibration-b".to_owned(),
                    likelihoods_by_hypothesis: scenario_b,
                },
            ],
        }
    }

    #[test]
    fn open_world_requires_explicit_unresolved_hypothesis() {
        let mut candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        candidate.hypothesis_ids[2] = "unresolved-but-not-calibrated".to_owned();
        let priors = [0.45, 0.45, 0.10];
        assert_eq!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::MissingUnresolvedHypothesis)
        );
    }

    #[test]
    fn open_world_requires_positive_unresolved_prior_mass() {
        let candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        let priors = [0.5, 0.5, 0.0];
        assert_eq!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::ZeroUnresolvedPrior)
        );
    }

    #[test]
    fn open_world_reports_worst_and_best_declared_likelihood_scenarios() {
        let candidate = open_world_candidate(
            "open-world",
            2.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.6, 0.4], vec![0.4, 0.6], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        let score = evaluate_open_world_candidate(&priors, &candidate).unwrap();
        assert_eq!(score.unresolved_prior_mass, 0.10);
        assert_eq!(score.scenario_scores.len(), 2);
        assert_eq!(score.worst_case_scenario_id, "calibration-b");
        assert_eq!(score.best_case_scenario_id, "calibration-a");
        assert!(score.worst_case_information_gain_bits > 0.0);
        assert!(score.worst_case_information_gain_bits < score.best_case_information_gain_bits);
        assert_eq!(
            score.worst_case_information_gain_bits_per_cost,
            score.worst_case_information_gain_bits / 2.0
        );
    }

    #[test]
    fn open_world_worst_case_cannot_exceed_best_case() {
        let candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        let score = evaluate_open_world_candidate(&priors, &candidate).unwrap();
        assert_eq!(score.worst_case_information_gain_bits, 0.0);
        assert!(score.best_case_information_gain_bits > 0.0);
    }

    #[test]
    fn open_world_ranking_uses_worst_case_not_nominal_eig() {
        let optimistic_brittle = open_world_candidate(
            "optimistic-brittle",
            1.0,
            vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
            vec![vec![0.5, 0.5], vec![0.5, 0.5], vec![0.5, 0.5]],
        );
        let robust = open_world_candidate(
            "robust",
            1.0,
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
            vec![vec![0.7, 0.3], vec![0.3, 0.7], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        let ranked = rank_open_world_candidates(
            &priors,
            &[optimistic_brittle, robust],
        ).unwrap();
        assert_eq!(ranked[0].experiment_id, "robust");
        assert!(ranked[0].score.worst_case_information_gain_bits > 0.0);
        assert_eq!(ranked[1].score.worst_case_information_gain_bits, 0.0);
    }

    #[test]
    fn open_world_rejects_duplicate_scenario_ids() {
        let mut candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        candidate.likelihood_scenarios[1].id = "calibration-a".to_owned();
        let priors = [0.45, 0.45, 0.10];
        assert_eq!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::DuplicateLikelihoodScenarioId {
                scenario_id: "calibration-a".to_owned()
            })
        );
    }

    #[test]
    fn open_world_rejects_malformed_likelihood_scenario() {
        let candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.3], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        assert!(matches!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::LikelihoodMassNotOne { hypothesis_index: 0, .. })
        ));
    }

    #[test]
    fn open_world_report_wrapper_is_versioned() {
        let candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
            vec![vec![0.7, 0.3], vec![0.3, 0.7], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        let report = evaluate_open_world_report(&priors, &[candidate]).unwrap();
        assert_eq!(report.schema_version, "connectome-open-world-eig-v1");
        assert_eq!(report.unresolved_hypothesis_id, UNRESOLVED_HYPOTHESIS_ID);
        assert_eq!(report.ranked_experiments.len(), 1);
    }

    #[test]
    fn open_world_rejects_duplicate_hypothesis_ids() {
        let mut candidate = open_world_candidate(
            "open-world", 1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        candidate.hypothesis_ids[1] = candidate.hypothesis_ids[0].clone();
        let priors = [0.45, 0.45, 0.10];
        assert!(matches!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::DuplicateHypothesisId { .. })
        ));
    }

    #[test]
    fn open_world_rejects_empty_likelihood_scenario_set() {
        let mut candidate = open_world_candidate(
            "open-world", 1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
        );
        candidate.likelihood_scenarios.clear();
        let priors = [0.45, 0.45, 0.10];
        assert_eq!(
            evaluate_open_world_candidate(&priors, &candidate),
            Err(DesignError::EmptyLikelihoodScenarioSet)
        );
    }

    #[test]
    fn open_world_ranking_rejects_inconsistent_hypothesis_order() {
        let first = open_world_candidate(
            "first", 1.0,
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
            vec![vec![0.7, 0.3], vec![0.3, 0.7], vec![0.5, 0.5]],
        );
        let mut second = open_world_candidate(
            "second", 1.0,
            vec![vec![0.8, 0.2], vec![0.2, 0.8], vec![0.5, 0.5]],
            vec![vec![0.7, 0.3], vec![0.3, 0.7], vec![0.5, 0.5]],
        );
        second.hypothesis_ids.swap(0, 1);
        let priors = [0.45, 0.45, 0.10];
        assert_eq!(
            rank_open_world_candidates(&priors, &[first, second]),
            Err(DesignError::InconsistentHypothesisOrder)
        );
    }

    #[test]
    fn open_world_scenario_order_does_not_change_score() {
        let mut candidate = open_world_candidate(
            "open-world",
            1.0,
            vec![vec![0.9, 0.1], vec![0.1, 0.9], vec![0.5, 0.5]],
            vec![vec![0.6, 0.4], vec![0.4, 0.6], vec![0.5, 0.5]],
        );
        let priors = [0.45, 0.45, 0.10];
        let first = evaluate_open_world_candidate(&priors, &candidate).unwrap();
        candidate.likelihood_scenarios.reverse();
        let second = evaluate_open_world_candidate(&priors, &candidate).unwrap();
        assert_eq!(first, second);
    }

    fn target_query_example() -> TargetCausalQuery {
        TargetCausalQuery {
            id: "increase-functional-output".to_owned(),
            hypothesis_ids: vec![
                "mechanism_a".to_owned(),
                "mechanism_b".to_owned(),
                "mechanism_no_effect".to_owned(),
                UNRESOLVED_HYPOTHESIS_ID.to_owned(),
            ],
            answer_by_hypothesis: vec![
                "positive".to_owned(),
                "positive".to_owned(),
                "negative".to_owned(),
                "unresolved".to_owned(),
            ],
        }
    }

    fn target_query_candidate(
        id: &str,
        likelihoods: Vec<Vec<f64>>,
    ) -> OpenWorldCandidateExperiment {
        OpenWorldCandidateExperiment {
            id: id.to_owned(),
            cost: 1.0,
            outcomes: vec!["response".to_owned(), "no_response".to_owned()],
            hypothesis_ids: vec![
                "mechanism_a".to_owned(),
                "mechanism_b".to_owned(),
                "mechanism_no_effect".to_owned(),
                UNRESOLVED_HYPOTHESIS_ID.to_owned(),
            ],
            likelihood_scenarios: vec![LikelihoodScenario {
                id: "declared-likelihood".to_owned(),
                likelihoods_by_hypothesis: likelihoods,
            }],
        }
    }

    #[test]
    fn target_query_eig_ignores_irrelevant_within_answer_mechanism_information() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "mechanism-discriminator",
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        );
        let query = target_query_example();
        let global = evaluate_open_world_candidate(&priors, &candidate).unwrap();
        let goal = evaluate_open_world_target_query(&priors, &candidate, &query).unwrap();
        assert!((global.worst_case_information_gain_bits - 0.6).abs() < 1.0e-12);
        assert_eq!(goal.worst_case_information_gain_bits, 0.0);
    }

    #[test]
    fn target_query_ranking_can_reverse_global_eig_ranking() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let mechanism_discriminator = target_query_candidate(
            "mechanism-discriminator",
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        );
        let outcome_discriminator = target_query_candidate(
            "outcome-discriminator",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let global = rank_open_world_candidates(
            &priors,
            &[mechanism_discriminator.clone(), outcome_discriminator.clone()],
        ).unwrap();
        assert_eq!(global[0].experiment_id, "mechanism-discriminator");

        let query = target_query_example();
        let goal = rank_open_world_candidates_for_query(
            &priors,
            &[mechanism_discriminator, outcome_discriminator],
            &query,
        ).unwrap();
        assert_eq!(goal[0].experiment_id, "outcome-discriminator");
        assert!(goal[0].score.worst_case_information_gain_bits > 0.43);
        assert_eq!(goal[1].score.worst_case_information_gain_bits, 0.0);
    }

    #[test]
    fn target_query_rejects_hypothesis_order_mismatch() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "candidate",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let mut query = target_query_example();
        query.hypothesis_ids.swap(0, 1);
        assert_eq!(
            evaluate_open_world_target_query(&priors, &candidate, &query),
            Err(DesignError::InconsistentHypothesisOrder)
        );
    }

    #[test]
    fn target_query_rejects_empty_answer_labels() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "candidate",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let mut query = target_query_example();
        query.answer_by_hypothesis[3] = "   ".to_owned();
        assert_eq!(
            evaluate_open_world_target_query(&priors, &candidate, &query),
            Err(DesignError::EmptyTargetQueryAnswer)
        );
    }

    #[test]
    fn target_query_report_is_versioned_and_keeps_query_definition() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "outcome-discriminator",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let query = target_query_example();
        let report = evaluate_target_query_report(&priors, &[candidate], query.clone()).unwrap();
        assert_eq!(report.schema_version, "connectome-target-query-eig-v1");
        assert_eq!(report.query, query);
        assert_eq!(report.ranked_experiments.len(), 1);
    }

    fn decision_problem_example() -> DecisionProblem {
        DecisionProblem {
            id: "choose-functional-output-intervention".to_owned(),
            hypothesis_ids: vec![
                "mechanism_a".to_owned(),
                "mechanism_b".to_owned(),
                "mechanism_no_effect".to_owned(),
                UNRESOLVED_HYPOTHESIS_ID.to_owned(),
            ],
            actions: vec![
                DecisionAction {
                    id: "intervene".to_owned(),
                    utility_by_hypothesis: vec![2.0, 2.0, -2.0, -1.0],
                },
                DecisionAction {
                    id: "do-not-intervene".to_owned(),
                    utility_by_hypothesis: vec![0.0, 0.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn evsi_is_zero_when_experiment_cannot_change_downstream_decision() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "mechanism-discriminator",
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        );
        let problem = decision_problem_example();
        let score = evaluate_open_world_decision_value(&priors, &candidate, &problem).unwrap();
        assert_eq!(score.worst_case_evsi, 0.0);
        assert_eq!(score.worst_case_evsi_per_cost, 0.0);
    }

    #[test]
    fn evsi_rewards_information_that_changes_the_optimal_action() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let mechanism_discriminator = target_query_candidate(
            "mechanism-discriminator",
            vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        );
        let outcome_discriminator = target_query_candidate(
            "outcome-discriminator",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let problem = decision_problem_example();
        let ranked = rank_open_world_candidates_for_decision(
            &priors,
            &[mechanism_discriminator, outcome_discriminator],
            &problem,
        ).unwrap();
        assert_eq!(ranked[0].experiment_id, "outcome-discriminator");
        assert!((ranked[0].score.worst_case_evsi - 0.47).abs() < 1.0e-12);
        assert!((ranked[0].score.scenario_values[0].current_expected_utility - 0.5).abs() < 1.0e-12);
        assert!((ranked[0].score.scenario_values[0].expected_post_experiment_utility - 0.97).abs() < 1.0e-12);
        assert_eq!(ranked[0].score.scenario_values[0].outcome_decisions[0].selected_action_id, "intervene");
        assert_eq!(ranked[0].score.scenario_values[0].outcome_decisions[1].selected_action_id, "do-not-intervene");
        assert_eq!(ranked[1].score.worst_case_evsi, 0.0);
    }

    #[test]
    fn evsi_report_is_versioned_and_preserves_utility_matrix() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "outcome-discriminator",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let problem = decision_problem_example();
        let report = evaluate_decision_value_report(&priors, &[candidate], problem.clone()).unwrap();
        assert_eq!(report.schema_version, "connectome-decision-value-v1");
        assert_eq!(report.decision_problem, problem);
        assert_eq!(report.ranked_experiments.len(), 1);
    }

    #[test]
    fn evsi_uses_worst_case_over_likelihood_ambiguity_set() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let mut candidate = target_query_candidate(
            "candidate",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        candidate.likelihood_scenarios.push(LikelihoodScenario {
            id: "shared-misspecification".to_owned(),
            likelihoods_by_hypothesis: vec![
                vec![0.5, 0.5],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        });
        let problem = decision_problem_example();
        let score = evaluate_open_world_decision_value(&priors, &candidate, &problem).unwrap();
        assert!(score.best_case_evsi > 0.0);
        assert_eq!(score.worst_case_evsi, 0.0);
        assert_eq!(score.worst_case_scenario_id, "shared-misspecification");
    }

    #[test]
    fn evsi_is_stable_to_accepted_prior_floating_point_mass_error() {
        let candidate = target_query_candidate(
            "outcome-discriminator",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let problem = decision_problem_example();
        let exact = evaluate_open_world_decision_value(
            &[0.3, 0.3, 0.3, 0.1],
            &candidate,
            &problem,
        ).unwrap();
        let near = evaluate_open_world_decision_value(
            &[0.3000000001, 0.3000000001, 0.3000000001, 0.1000000001],
            &candidate,
            &problem,
        ).unwrap();
        assert!((exact.worst_case_evsi - near.worst_case_evsi).abs() < 1.0e-12);
    }

    #[test]
    fn evsi_is_zero_when_only_one_downstream_action_exists() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "candidate",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let mut problem = decision_problem_example();
        problem.actions.truncate(1);
        let score = evaluate_open_world_decision_value(&priors, &candidate, &problem).unwrap();
        assert_eq!(score.worst_case_evsi, 0.0);
        assert_eq!(score.worst_case_evsi_per_cost, 0.0);
    }

    #[test]
    fn evsi_rejects_malformed_utility_matrix_and_empty_action_ids() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = target_query_candidate(
            "candidate",
            vec![
                vec![0.9, 0.1],
                vec![0.9, 0.1],
                vec![0.1, 0.9],
                vec![0.5, 0.5],
            ],
        );
        let mut problem = decision_problem_example();
        problem.actions[0].utility_by_hypothesis.pop();
        assert!(matches!(
            evaluate_open_world_decision_value(&priors, &candidate, &problem),
            Err(DesignError::DecisionUtilityCountMismatch { .. })
        ));

        let mut problem = decision_problem_example();
        problem.actions[0].id = " ".to_owned();
        assert_eq!(
            evaluate_open_world_decision_value(&priors, &candidate, &problem),
            Err(DesignError::EmptyDecisionActionId)
        );
    }

    fn hit_objective_example() -> ThresholdHitObjective {
        ThresholdHitObjective {
            id: "functional-output-at-least-0.8".to_owned(),
            outcome_ids: vec!["subthreshold".to_owned(), "hit".to_owned()],
            outcome_values: vec![0.0, 1.0],
            threshold: 0.8,
            direction: HitDirection::AtLeast,
        }
    }

    fn hit_candidate(
        id: &str,
        likelihoods: Vec<Vec<f64>>,
    ) -> OpenWorldCandidateExperiment {
        OpenWorldCandidateExperiment {
            id: id.to_owned(),
            cost: 1.0,
            outcomes: vec!["subthreshold".to_owned(), "hit".to_owned()],
            hypothesis_ids: vec![
                "mechanism_a".to_owned(),
                "mechanism_b".to_owned(),
                "mechanism_no_effect".to_owned(),
                UNRESOLVED_HYPOTHESIS_ID.to_owned(),
            ],
            likelihood_scenarios: vec![LikelihoodScenario {
                id: "known-answer".to_owned(),
                likelihoods_by_hypothesis: likelihoods,
            }],
        }
    }

    #[test]
    fn threshold_hit_probability_finds_likely_target_crossing() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let mechanism_discriminator = hit_candidate(
            "mechanism-discriminator",
            vec![
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.5, 0.5],
                vec![0.5, 0.5],
            ],
        );
        let hit_discriminator = hit_candidate(
            "hit-discriminator",
            vec![
                vec![0.2, 0.8],
                vec![0.2, 0.8],
                vec![0.9, 0.1],
                vec![0.5, 0.5],
            ],
        );
        let objective = hit_objective_example();
        let ranked = rank_open_world_candidates_for_hit(
            &priors,
            &[mechanism_discriminator, hit_discriminator],
            &objective,
        ).unwrap();
        assert_eq!(ranked[0].experiment_id, "hit-discriminator");
        assert!((ranked[0].score.worst_case_probability_of_hit - 0.56).abs() < 1.0e-12);
        assert!((ranked[1].score.worst_case_probability_of_hit - 0.50).abs() < 1.0e-12);
    }

    #[test]
    fn threshold_hit_uses_worst_case_likelihood_scenario() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let mut candidate = hit_candidate(
            "candidate",
            vec![
                vec![0.2, 0.8],
                vec![0.2, 0.8],
                vec![0.9, 0.1],
                vec![0.5, 0.5],
            ],
        );
        candidate.likelihood_scenarios.push(LikelihoodScenario {
            id: "pessimistic".to_owned(),
            likelihoods_by_hypothesis: vec![
                vec![0.8, 0.2],
                vec![0.8, 0.2],
                vec![0.9, 0.1],
                vec![0.5, 0.5],
            ],
        });
        let score = evaluate_open_world_hit_probability(
            &priors,
            &candidate,
            &hit_objective_example(),
        ).unwrap();
        assert!(score.best_case_probability_of_hit > score.worst_case_probability_of_hit);
        assert_eq!(score.worst_case_scenario_id, "pessimistic");
    }

    #[test]
    fn threshold_hit_rejects_outcome_order_mismatch_and_invalid_values() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = hit_candidate(
            "candidate",
            vec![
                vec![0.2, 0.8],
                vec![0.2, 0.8],
                vec![0.9, 0.1],
                vec![0.5, 0.5],
            ],
        );
        let mut objective = hit_objective_example();
        objective.outcome_ids.swap(0, 1);
        assert_eq!(
            evaluate_open_world_hit_probability(&priors, &candidate, &objective),
            Err(DesignError::InconsistentHitOutcomeOrder)
        );

        let mut objective = hit_objective_example();
        objective.outcome_values[1] = f64::NAN;
        assert_eq!(
            evaluate_open_world_hit_probability(&priors, &candidate, &objective),
            Err(DesignError::NonFiniteHitOutcomeValue { outcome_index: 1 })
        );
    }

    #[test]
    fn threshold_hit_report_is_versioned() {
        let priors = [0.3, 0.3, 0.3, 0.1];
        let candidate = hit_candidate(
            "candidate",
            vec![
                vec![0.2, 0.8],
                vec![0.2, 0.8],
                vec![0.9, 0.1],
                vec![0.5, 0.5],
            ],
        );
        let report = evaluate_hit_probability_report(
            &priors,
            &[candidate],
            hit_objective_example(),
        ).unwrap();
        assert_eq!(report.schema_version, "connectome-hit-probability-v1");
        assert_eq!(report.ranked_experiments.len(), 1);
    }

    fn calibration_unit(
        unit_id: &str,
        family_id: &str,
        nonconformity_score: f64,
    ) -> AdequacyCalibrationUnit {
        AdequacyCalibrationUnit {
            unit_id: unit_id.to_owned(),
            mechanism_family_id: family_id.to_owned(),
            nonconformity_score,
        }
    }

    fn adequacy_calibration_example() -> (
        Vec<AdequacyCalibrationUnit>,
        Vec<AdequacyCalibrationUnit>,
    ) {
        let in_scope = (0..20)
            .map(|index| {
                calibration_unit(
                    &format!("in-scope-family-unit-{index:02}"),
                    &format!("in-scope-family-{index:02}"),
                    index as f64 / 10.0,
                )
            })
            .collect();
        let omitted = (0..19)
            .map(|index| {
                calibration_unit(
                    &format!("omitted-family-unit-{index:02}"),
                    &format!("omitted-family-{index:02}"),
                    10.0 + index as f64,
                )
            })
            .collect();
        (in_scope, omitted)
    }

    #[test]
    fn adequacy_report_wrapper_is_versioned() {
        let (in_scope, omitted) = adequacy_calibration_example();
        let report = calibrate_model_adequacy_report(
            &in_scope,
            &omitted,
            AdequacyGateConfig::default(),
        ).unwrap();
        assert_eq!(report.schema_version, "connectome-open-world-adequacy-v1");
        assert_eq!(report.calibrator.lower_order_statistic_k, 1);
        assert!(report.calibrator.usable);
    }

    #[test]
    fn adequacy_gate_needs_nineteen_omitted_families_at_five_percent() {
        let (in_scope, mut omitted) = adequacy_calibration_example();
        omitted.pop();
        let calibrator = calibrate_model_adequacy(
            &in_scope,
            &omitted,
            AdequacyGateConfig::default(),
        )
        .unwrap();
        assert_eq!(calibrator.omitted_mechanism_family_count, 18);
        assert_eq!(calibrator.lower_order_statistic_k, 0);
        assert!(calibrator.nonconformity_threshold.is_none());
        assert!(!calibrator.usable);
        assert_eq!(
            assess_model_adequacy(&calibrator, 0.0).unwrap(),
            AdequacyDecision::Unresolved
        );
    }

    #[test]
    fn adequacy_gate_calibrates_strict_threshold_and_in_scope_floor() {
        let (in_scope, omitted) = adequacy_calibration_example();
        let calibrator = calibrate_model_adequacy(
            &in_scope,
            &omitted,
            AdequacyGateConfig::default(),
        )
        .unwrap();
        assert_eq!(calibrator.omitted_mechanism_family_count, 19);
        assert_eq!(calibrator.lower_order_statistic_k, 1);
        assert_eq!(calibrator.nonconformity_threshold, Some(10.0));
        assert_eq!(calibrator.false_adequacy_probability_bound, 0.05);
        let acceptance = calibrator.in_scope_acceptance.as_ref().unwrap();
        assert_eq!(acceptance.accepted_in_scope_units, 20);
        assert_eq!(acceptance.total_in_scope_units, 20);
        assert!(acceptance.one_sided_wilson_lower_bound >= 0.80);
        assert!(calibrator.usable);
        assert_eq!(
            assess_model_adequacy(&calibrator, 9.99).unwrap(),
            AdequacyDecision::AdequateUnderCalibration
        );
        // Strict inequality makes ties conservative.
        assert_eq!(
            assess_model_adequacy(&calibrator, 10.0).unwrap(),
            AdequacyDecision::InadequateOrDistributionShift
        );
    }

    #[test]
    fn adequacy_gate_abstains_when_in_scope_acceptance_floor_fails() {
        let (mut in_scope, omitted) = adequacy_calibration_example();
        for (index, unit) in in_scope.iter_mut().enumerate() {
            unit.nonconformity_score = 10.0 + index as f64;
        }
        let calibrator = calibrate_model_adequacy(
            &in_scope,
            &omitted,
            AdequacyGateConfig::default(),
        )
        .unwrap();
        assert!(!calibrator.usable);
        assert_eq!(
            assess_model_adequacy(&calibrator, 0.0).unwrap(),
            AdequacyDecision::Unresolved
        );
    }

    #[test]
    fn adequacy_gate_rejects_mechanism_family_leakage() {
        let in_scope = vec![calibration_unit("id-1", "shared-family", 0.1)];
        let omitted = vec![calibration_unit("ood-1", "shared-family", 10.0)];
        assert_eq!(
            calibrate_model_adequacy(
                &in_scope,
                &omitted,
                AdequacyGateConfig::default()
            ),
            Err(DesignError::MechanismFamilyLeakage {
                mechanism_family_id: "shared-family".to_owned()
            })
        );
    }

    #[test]
    fn adequacy_gate_rejects_multiple_units_from_one_in_scope_family() {
        let in_scope = vec![
            calibration_unit("id-1", "same-in-scope-family", 0.1),
            calibration_unit("id-2", "same-in-scope-family", 0.2),
        ];
        let omitted = vec![
            calibration_unit("ood-1", "ood-family-1", 10.0),
        ];
        assert_eq!(
            calibrate_model_adequacy(
                &in_scope,
                &omitted,
                AdequacyGateConfig::default()
            ),
            Err(DesignError::DuplicateInScopeMechanismFamily {
                mechanism_family_id: "same-in-scope-family".to_owned()
            })
        );
    }

    #[test]
    fn adequacy_gate_rejects_multiple_units_from_one_omitted_family() {
        let in_scope = vec![calibration_unit("id-1", "in-family", 0.1)];
        let omitted = vec![
            calibration_unit("ood-1", "same-ood-family", 10.0),
            calibration_unit("ood-2", "same-ood-family", 11.0),
        ];
        assert_eq!(
            calibrate_model_adequacy(
                &in_scope,
                &omitted,
                AdequacyGateConfig::default()
            ),
            Err(DesignError::DuplicateOmittedMechanismFamily {
                mechanism_family_id: "same-ood-family".to_owned()
            })
        );
    }

    #[test]
    fn adequacy_gate_rejects_duplicate_unit_ids_across_cohorts() {
        let in_scope = vec![calibration_unit("shared-unit", "in-family", 0.1)];
        let omitted = vec![calibration_unit("shared-unit", "ood-family", 10.0)];
        assert_eq!(
            calibrate_model_adequacy(
                &in_scope,
                &omitted,
                AdequacyGateConfig::default()
            ),
            Err(DesignError::DuplicateCalibrationUnitId {
                unit_id: "shared-unit".to_owned()
            })
        );
    }

    #[test]
    fn adequacy_gate_rejects_non_finite_and_invalid_test_scores() {
        let in_scope = vec![calibration_unit("id-1", "in-family", 0.1)];
        let omitted = vec![calibration_unit("ood-1", "ood-family", 10.0)];
        let mut bad = in_scope.clone();
        bad[0].nonconformity_score = f64::NAN;
        assert_eq!(
            calibrate_model_adequacy(&bad, &omitted, AdequacyGateConfig::default()),
            Err(DesignError::NonFiniteNonconformityScore)
        );

        let (in_scope, omitted) = adequacy_calibration_example();
        let calibrator = calibrate_model_adequacy(
            &in_scope, &omitted, AdequacyGateConfig::default()
        ).unwrap();
        assert_eq!(
            assess_model_adequacy(&calibrator, f64::INFINITY),
            Err(DesignError::InvalidAdequacyScore)
        );
    }

    #[test]
    fn adequacy_gate_rejects_invalid_config() {
        let (in_scope, omitted) = adequacy_calibration_example();
        let bad = AdequacyGateConfig {
            false_adequacy_alpha: 0.0,
            ..AdequacyGateConfig::default()
        };
        assert_eq!(
            calibrate_model_adequacy(&in_scope, &omitted, bad),
            Err(DesignError::InvalidConfig)
        );
    }

    #[test]
    fn prior_entropy_bounds_information_gain() {
        let experiment = binary_experiment("bounded", 1.0, 1.0, 0.0);
        let score = evaluate_candidate(&[0.9, 0.1], &experiment).unwrap();
        assert!(score.information_gain_bits <= score.prior_entropy_bits);
        assert!(score.information_gain_bits >= 0.0);
    }
}

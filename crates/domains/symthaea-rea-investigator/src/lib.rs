#![forbid(unsafe_code)]

//! Deterministic, fail-closed experiment selection over REA Evidence references.
//!
//! This planner ranks *proposals* only. It never executes experiments, mutates
//! targets, authenticates Evidence producers, or proves causal truth. Provenance
//! classes are caller-supplied classifications; they are not signatures.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const INFORMATION_GAIN_EPSILON: f64 = 1.0e-12;
const EVIDENCE_ID_PREFIX: &str = "ev_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceOrigin {
    IndependentObservation,
    IndependentDerivation,
    ModelGenerated,
    Synthetic,
    ExternalReference,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvidenceProvenance {
    pub evidence_id: String,
    pub origin: EvidenceOrigin,
    /// Parent Evidence IDs for independently derived records.
    pub parent_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: String,
    pub prior_probability: f64,
    pub supporting_evidence_ids: Vec<String>,
    pub contradicting_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Authorization {
    Unknown,
    Denied,
    Authorized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TargetImpact {
    ReadOnlyObservation,
    TemporaryConfiguration,
    ServiceRestart,
    DataMutation,
    PhysicalActuation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodRow {
    pub hypothesis_id: String,
    /// P(outcome | hypothesis), aligned with Experiment.outcomes.
    pub outcome_probabilities: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentProposal {
    pub id: String,
    pub outcomes: Vec<String>,
    pub likelihoods: Vec<LikelihoodRow>,
    pub authorization: Authorization,
    pub reversible: bool,
    pub destructive: bool,
    pub target_impact: TargetImpact,
    pub estimated_time_seconds: f64,
    pub estimated_energy_joules: f64,
    /// Caller-estimated risk in [0, 1].
    pub estimated_risk: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostWeights {
    pub time: f64,
    pub energy: f64,
    pub risk: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlannerConfig {
    pub max_time_seconds: f64,
    pub max_energy_joules: f64,
    pub max_risk: f64,
    pub cost_weights: CostWeights,
    pub max_target_impact: TargetImpact,
    pub minimum_information_gain_bits: f64,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_time_seconds: 300.0,
            max_energy_joules: 100.0,
            max_risk: 0.1,
            cost_weights: CostWeights {
                time: 1.0,
                energy: 1.0,
                risk: 2.0,
            },
            max_target_impact: TargetImpact::ReadOnlyObservation,
            minimum_information_gain_bits: 1.0e-6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannerErrorKind {
    InvalidEvidenceId,
    DuplicateEvidenceId,
    DanglingEvidenceReference,
    EvidenceGraphCycle,
    DuplicateHypothesisId,
    InvalidHypothesisDistribution,
    UngroundedHypothesis,
    DuplicateExperimentId,
    InvalidPlannerConfig,
    InvalidMeasuredDistribution,
    InvalidLikelihoods,
    UnqualifiedObservationEvidence,
    InvalidDecisionMatrix,
    InvalidDecisionConfiguration,
    InvalidDecisionScenario,
    DecisionPolicySearchBudgetExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannerError {
    pub kind: PlannerErrorKind,
    pub message: String,
}

impl PlannerError {
    fn new(kind: PlannerErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl fmt::Display for PlannerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.message)
    }
}

impl std::error::Error for PlannerError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionReason {
    AuthorizationUnknown,
    NotAuthorized,
    Irreversible,
    Destructive,
    TargetImpactExceeded,
    TimeBudgetExceeded,
    EnergyBudgetExceeded,
    RiskBudgetExceeded,
    InvalidOutcomeSpace,
    InvalidLikelihoods,
    NoPredictedInformationGain,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PredictedOutcome {
    pub outcome: String,
    pub probability: f64,
    pub posterior: Vec<(String, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RankedExperiment {
    pub experiment_id: String,
    pub expected_information_gain_bits: f64,
    /// Dimensionless cost, normalized against the configured resource budgets.
    pub normalized_cost: f64,
    /// Expected information gain divided by normalized cost.
    pub utility: f64,
    pub predicted_outcomes: Vec<PredictedOutcome>,
    pub support_evidence_ids: Vec<String>,
    /// Terminal independent observations supporting those records.
    pub independent_root_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RejectedExperiment {
    pub experiment_id: String,
    pub reasons: Vec<RejectionReason>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbstentionReason {
    NoCompetingHypotheses,
    NoIndependentlyGroundedHypothesis,
    NoEligibleExperiment,
    NoDiscriminatingExperiment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvestigationPlan {
    pub selected: Option<RankedExperiment>,
    pub ranked_experiments: Vec<RankedExperiment>,
    pub rejected_experiments: Vec<RejectedExperiment>,
    pub abstention: Option<AbstentionReason>,
    pub prior_entropy_bits: f64,
    pub grounded_hypothesis_ids: Vec<String>,
    /// These provenance classifications are caller supplied, not authenticated.
    pub claim_ceiling: PlannerClaimCeiling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlannerClaimCeiling {
    pub evidence_producer_authenticated: bool,
    pub causal_truth_proven: bool,
    pub selected_experiment_executed: bool,
    pub target_mutated: bool,
    pub likelihood_calibration_verified: bool,
}

impl Default for PlannerClaimCeiling {
    fn default() -> Self {
        Self {
            evidence_producer_authenticated: false,
            causal_truth_proven: false,
            selected_experiment_executed: false,
            target_mutated: false,
            likelihood_calibration_verified: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Probability {
    pub hypothesis_id: String,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MeasuredInformationGain {
    pub prior_entropy_bits: f64,
    pub posterior_entropy_bits: f64,
    /// May be negative when the caller-supplied posterior has greater entropy.
    pub observed_entropy_reduction_bits: f64,
    pub observation_evidence_ids: Vec<String>,
    pub independent_root_evidence_ids: Vec<String>,
    pub evidence_provenance_authenticated: bool,
}

fn valid_evidence_id(value: &str) -> bool {
    let Some(digest) = value.strip_prefix(EVIDENCE_ID_PREFIX) else {
        return false;
    };
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn probability_valid(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn distribution_valid(values: impl Iterator<Item = f64>) -> bool {
    let mut sum = 0.0;
    for value in values {
        if !probability_valid(value) {
            return false;
        }
        sum += value;
    }
    (sum - 1.0).abs() <= PROBABILITY_TOLERANCE
}

fn entropy_bits(probabilities: impl Iterator<Item = f64>) -> f64 {
    probabilities
        .filter(|probability| *probability > 0.0)
        .map(|probability| -probability * probability.log2())
        .sum()
}

fn validate_config(config: PlannerConfig) -> Result<(), PlannerError> {
    let values = [
        config.max_time_seconds,
        config.max_energy_joules,
        config.max_risk,
        config.cost_weights.time,
        config.cost_weights.energy,
        config.cost_weights.risk,
        config.minimum_information_gain_bits,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || config.max_time_seconds <= 0.0
        || config.max_energy_joules <= 0.0
        || config.max_risk <= 0.0
        || config.max_risk > 1.0
        || config.cost_weights.time + config.cost_weights.energy + config.cost_weights.risk <= 0.0
    {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidPlannerConfig,
            "planner budgets, thresholds and weights must be finite and non-negative; budgets must be positive and risk budget must be <= 1",
        ));
    }
    Ok(())
}

fn validate_evidence_catalog<'a>(
    evidence: &'a [EvidenceProvenance],
) -> Result<HashMap<&'a str, &'a EvidenceProvenance>, PlannerError> {
    let mut catalog = HashMap::with_capacity(evidence.len());
    for record in evidence {
        if !valid_evidence_id(&record.evidence_id) {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidEvidenceId,
                format!("invalid REA Evidence ID: {}", record.evidence_id),
            ));
        }
        if catalog
            .insert(record.evidence_id.as_str(), record)
            .is_some()
        {
            return Err(PlannerError::new(
                PlannerErrorKind::DuplicateEvidenceId,
                format!("duplicate Evidence ID: {}", record.evidence_id),
            ));
        }
        let mut seen = HashSet::new();
        for parent in &record.parent_evidence_ids {
            if !valid_evidence_id(parent) {
                return Err(PlannerError::new(
                    PlannerErrorKind::InvalidEvidenceId,
                    format!("invalid parent Evidence ID: {parent}"),
                ));
            }
            if !seen.insert(parent) {
                return Err(PlannerError::new(
                    PlannerErrorKind::DanglingEvidenceReference,
                    format!(
                        "duplicate parent Evidence ID {parent} in {}",
                        record.evidence_id
                    ),
                ));
            }
        }
    }

    for record in evidence {
        for parent in &record.parent_evidence_ids {
            if !catalog.contains_key(parent.as_str()) {
                return Err(PlannerError::new(
                    PlannerErrorKind::DanglingEvidenceReference,
                    format!(
                        "Evidence {} references missing parent {}",
                        record.evidence_id, parent
                    ),
                ));
            }
        }
        if record.origin == EvidenceOrigin::IndependentObservation
            && !record.parent_evidence_ids.is_empty()
        {
            return Err(PlannerError::new(
                PlannerErrorKind::DanglingEvidenceReference,
                format!(
                    "independent observation {} cannot declare derived parent records",
                    record.evidence_id
                ),
            ));
        }
    }

    Ok(catalog)
}

fn evidence_has_independent_root(
    id: &str,
    catalog: &HashMap<&str, &EvidenceProvenance>,
    visiting: &mut HashSet<String>,
    memo: &mut HashMap<String, bool>,
) -> Result<bool, PlannerError> {
    if let Some(result) = memo.get(id) {
        return Ok(*result);
    }
    if !visiting.insert(id.to_owned()) {
        return Err(PlannerError::new(
            PlannerErrorKind::EvidenceGraphCycle,
            format!("cycle in Evidence derivation graph at {id}"),
        ));
    }
    let record = catalog.get(id).ok_or_else(|| {
        PlannerError::new(
            PlannerErrorKind::DanglingEvidenceReference,
            format!("missing Evidence record {id}"),
        )
    })?;

    let result = match record.origin {
        EvidenceOrigin::IndependentObservation => true,
        EvidenceOrigin::IndependentDerivation => {
            if record.parent_evidence_ids.is_empty() {
                false
            } else {
                let mut rooted_parent = false;
                for parent in &record.parent_evidence_ids {
                    rooted_parent |=
                        evidence_has_independent_root(parent, catalog, visiting, memo)?;
                }
                rooted_parent
            }
        }
        EvidenceOrigin::ModelGenerated
        | EvidenceOrigin::Synthetic
        | EvidenceOrigin::ExternalReference => false,
    };

    visiting.remove(id);
    memo.insert(id.to_owned(), result);
    Ok(result)
}

fn collect_independent_roots(
    id: &str,
    catalog: &HashMap<&str, &EvidenceProvenance>,
    roots: &mut BTreeSet<String>,
    visited: &mut HashSet<String>,
) -> Result<(), PlannerError> {
    if !visited.insert(id.to_owned()) {
        return Ok(());
    }
    let record = catalog.get(id).ok_or_else(|| {
        PlannerError::new(
            PlannerErrorKind::DanglingEvidenceReference,
            format!("missing Evidence record {id}"),
        )
    })?;
    match record.origin {
        EvidenceOrigin::IndependentObservation => {
            roots.insert(id.to_owned());
        }
        EvidenceOrigin::IndependentDerivation => {
            for parent in &record.parent_evidence_ids {
                collect_independent_roots(parent, catalog, roots, visited)?;
            }
        }
        EvidenceOrigin::ModelGenerated
        | EvidenceOrigin::Synthetic
        | EvidenceOrigin::ExternalReference => {}
    }
    Ok(())
}

fn validate_hypotheses<'a>(
    hypotheses: &'a [Hypothesis],
    catalog: &HashMap<&str, &EvidenceProvenance>,
) -> Result<
    (
        BTreeMap<&'a str, f64>,
        Vec<String>,
        Vec<String>,
        Vec<String>,
    ),
    PlannerError,
> {
    if hypotheses.is_empty() {
        return Ok((BTreeMap::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let mut priors = BTreeMap::new();
    let mut total = 0.0;
    let mut grounded_ids = Vec::with_capacity(hypotheses.len());
    let mut qualified_support_ids = BTreeSet::new();
    let mut independent_root_ids = BTreeSet::new();
    let mut root_visited = HashSet::new();
    let mut visiting = HashSet::new();
    let mut memo = HashMap::new();

    for hypothesis in hypotheses {
        if hypothesis.id.trim().is_empty() || priors.contains_key(hypothesis.id.as_str()) {
            return Err(PlannerError::new(
                PlannerErrorKind::DuplicateHypothesisId,
                format!("empty or duplicate hypothesis ID: {}", hypothesis.id),
            ));
        }
        if !probability_valid(hypothesis.prior_probability) || hypothesis.prior_probability <= 0.0 {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidHypothesisDistribution,
                format!("hypothesis {} must have a prior in (0, 1]", hypothesis.id),
            ));
        }
        if hypothesis.supporting_evidence_ids.is_empty() {
            return Err(PlannerError::new(
                PlannerErrorKind::UngroundedHypothesis,
                format!(
                    "hypothesis {} has no supporting Evidence references",
                    hypothesis.id
                ),
            ));
        }

        let mut has_independent_root = false;
        for id in hypothesis
            .supporting_evidence_ids
            .iter()
            .chain(hypothesis.contradicting_evidence_ids.iter())
        {
            if !catalog.contains_key(id.as_str()) {
                return Err(PlannerError::new(
                    PlannerErrorKind::DanglingEvidenceReference,
                    format!(
                        "hypothesis {} references missing Evidence {id}",
                        hypothesis.id
                    ),
                ));
            }
        }

        for id in &hypothesis.supporting_evidence_ids {
            let independently_rooted =
                evidence_has_independent_root(id, catalog, &mut visiting, &mut memo)?;
            has_independent_root |= independently_rooted;
            if independently_rooted {
                qualified_support_ids.insert(id.clone());
                collect_independent_roots(
                    id,
                    catalog,
                    &mut independent_root_ids,
                    &mut root_visited,
                )?;
            }
        }
        if !has_independent_root {
            return Err(PlannerError::new(
                PlannerErrorKind::UngroundedHypothesis,
                format!(
                    "hypothesis {} has no independently rooted supporting Evidence; model-generated, synthetic and external-reference records cannot qualify",
                    hypothesis.id
                ),
            ));
        }

        priors.insert(hypothesis.id.as_str(), hypothesis.prior_probability);
        total += hypothesis.prior_probability;
        grounded_ids.push(hypothesis.id.clone());
    }

    if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidHypothesisDistribution,
            format!("hypothesis prior mass must sum to 1.0 (received {total:.12})"),
        ));
    }

    grounded_ids.sort();
    Ok((
        priors,
        grounded_ids,
        qualified_support_ids.into_iter().collect(),
        independent_root_ids.into_iter().collect(),
    ))
}

fn rejected(experiment: &ExperimentProposal, reason: RejectionReason) -> RejectedExperiment {
    RejectedExperiment {
        experiment_id: experiment.id.clone(),
        reasons: vec![reason],
    }
}

fn validate_experiment(
    experiment: &ExperimentProposal,
    hypothesis_ids: &[&str],
    config: PlannerConfig,
) -> Result<(), RejectionReason> {
    if experiment.authorization == Authorization::Unknown {
        return Err(RejectionReason::AuthorizationUnknown);
    }
    if experiment.authorization != Authorization::Authorized {
        return Err(RejectionReason::NotAuthorized);
    }
    if !experiment.reversible {
        return Err(RejectionReason::Irreversible);
    }
    if experiment.destructive {
        return Err(RejectionReason::Destructive);
    }
    if experiment.target_impact > config.max_target_impact {
        return Err(RejectionReason::TargetImpactExceeded);
    }
    if !experiment.estimated_time_seconds.is_finite()
        || experiment.estimated_time_seconds <= 0.0
        || experiment.estimated_time_seconds > config.max_time_seconds
    {
        return Err(RejectionReason::TimeBudgetExceeded);
    }
    if !experiment.estimated_energy_joules.is_finite()
        || experiment.estimated_energy_joules < 0.0
        || experiment.estimated_energy_joules > config.max_energy_joules
    {
        return Err(RejectionReason::EnergyBudgetExceeded);
    }
    if !experiment.estimated_risk.is_finite()
        || !(0.0..=1.0).contains(&experiment.estimated_risk)
        || experiment.estimated_risk > config.max_risk
    {
        return Err(RejectionReason::RiskBudgetExceeded);
    }

    if experiment.outcomes.len() < 2 {
        return Err(RejectionReason::InvalidOutcomeSpace);
    }
    let mut outcomes = HashSet::new();
    if experiment
        .outcomes
        .iter()
        .any(|outcome| outcome.trim().is_empty() || !outcomes.insert(outcome))
    {
        return Err(RejectionReason::InvalidOutcomeSpace);
    }
    if experiment.likelihoods.len() != hypothesis_ids.len() {
        return Err(RejectionReason::InvalidLikelihoods);
    }

    let mut rows = HashMap::new();
    for row in &experiment.likelihoods {
        if rows.insert(row.hypothesis_id.as_str(), row).is_some()
            || !hypothesis_ids.contains(&row.hypothesis_id.as_str())
            || row.outcome_probabilities.len() != experiment.outcomes.len()
            || !distribution_valid(row.outcome_probabilities.iter().copied())
        {
            return Err(RejectionReason::InvalidLikelihoods);
        }
    }
    if hypothesis_ids.iter().any(|id| !rows.contains_key(id)) {
        return Err(RejectionReason::InvalidLikelihoods);
    }

    Ok(())
}

fn evaluate_experiment(
    experiment: &ExperimentProposal,
    priors: &BTreeMap<&str, f64>,
    config: PlannerConfig,
    support_evidence_ids: &[String],
    independent_root_evidence_ids: &[String],
) -> RankedExperiment {
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    let prior_entropy = entropy_bits(priors.values().copied().collect::<Vec<_>>().into_iter());
    let rows: HashMap<&str, &LikelihoodRow> = experiment
        .likelihoods
        .iter()
        .map(|row| (row.hypothesis_id.as_str(), row))
        .collect();

    let mut predicted_outcomes = Vec::with_capacity(experiment.outcomes.len());
    let mut expected_posterior_entropy = 0.0;

    for (outcome_index, outcome) in experiment.outcomes.iter().enumerate() {
        let outcome_probability: f64 = hypothesis_ids
            .iter()
            .map(|id| priors[id] * rows[id].outcome_probabilities[outcome_index])
            .sum();

        let posterior: Vec<(String, f64)> = hypothesis_ids
            .iter()
            .map(|id| {
                let joint = priors[id] * rows[id].outcome_probabilities[outcome_index];
                (
                    (*id).to_owned(),
                    if outcome_probability > 0.0 {
                        joint / outcome_probability
                    } else {
                        0.0
                    },
                )
            })
            .collect();
        let posterior_entropy = entropy_bits(posterior.iter().map(|(_, probability)| *probability));
        expected_posterior_entropy += outcome_probability * posterior_entropy;
        predicted_outcomes.push(PredictedOutcome {
            outcome: outcome.clone(),
            probability: outcome_probability,
            posterior,
        });
    }

    let raw_information_gain = prior_entropy - expected_posterior_entropy;
    let information_gain = if raw_information_gain < 0.0 && raw_information_gain > -1.0e-10 {
        0.0
    } else {
        raw_information_gain.max(0.0)
    };

    let normalized_cost = config.cost_weights.time
        * (experiment.estimated_time_seconds / config.max_time_seconds)
        + config.cost_weights.energy
            * (experiment.estimated_energy_joules / config.max_energy_joules)
        + config.cost_weights.risk * (experiment.estimated_risk / config.max_risk);
    let utility = information_gain / normalized_cost.max(1.0e-12);

    RankedExperiment {
        experiment_id: experiment.id.clone(),
        expected_information_gain_bits: information_gain,
        normalized_cost,
        utility,
        predicted_outcomes,
        support_evidence_ids: support_evidence_ids.to_vec(),
        independent_root_evidence_ids: independent_root_evidence_ids.to_vec(),
    }
}

/// Create a fail-closed experiment plan. No experiment is executed.
///
/// Hypothesis priors and experiment likelihoods are caller-supplied model inputs.
/// Evidence provenance classifications are also caller-supplied and are not
/// authenticated by this planner.
pub fn plan_next_experiment(
    evidence: &[EvidenceProvenance],
    hypotheses: &[Hypothesis],
    experiments: &[ExperimentProposal],
    config: PlannerConfig,
) -> Result<InvestigationPlan, PlannerError> {
    validate_config(config)?;
    let catalog = validate_evidence_catalog(evidence)?;
    let (priors, grounded_hypothesis_ids, support_evidence_ids, independent_root_evidence_ids) =
        match validate_hypotheses(hypotheses, &catalog) {
            Ok(result) => result,
            Err(error) if error.kind == PlannerErrorKind::UngroundedHypothesis => {
                return Ok(InvestigationPlan {
                    selected: None,
                    ranked_experiments: Vec::new(),
                    rejected_experiments: Vec::new(),
                    abstention: Some(AbstentionReason::NoIndependentlyGroundedHypothesis),
                    prior_entropy_bits: 0.0,
                    grounded_hypothesis_ids: Vec::new(),
                    claim_ceiling: PlannerClaimCeiling::default(),
                });
            }
            Err(error) => return Err(error),
        };

    let prior_entropy = entropy_bits(priors.values().copied().collect::<Vec<_>>().into_iter());
    let mut seen_experiments = HashSet::new();
    for experiment in experiments {
        if experiment.id.trim().is_empty() || !seen_experiments.insert(experiment.id.as_str()) {
            return Err(PlannerError::new(
                PlannerErrorKind::DuplicateExperimentId,
                format!("empty or duplicate experiment ID: {}", experiment.id),
            ));
        }
    }

    let mut rejected_experiments = Vec::new();
    let mut ranked_experiments = Vec::new();

    if hypotheses.len() < 2 {
        return Ok(InvestigationPlan {
            selected: None,
            ranked_experiments,
            rejected_experiments,
            abstention: Some(AbstentionReason::NoCompetingHypotheses),
            prior_entropy_bits: prior_entropy,
            grounded_hypothesis_ids,
            claim_ceiling: PlannerClaimCeiling::default(),
        });
    }

    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();
    for experiment in experiments {
        if let Err(reason) = validate_experiment(experiment, &hypothesis_ids, config) {
            rejected_experiments.push(rejected(experiment, reason));
            continue;
        }

        let candidate = evaluate_experiment(
            experiment,
            &priors,
            config,
            &support_evidence_ids,
            &independent_root_evidence_ids,
        );
        if candidate.expected_information_gain_bits <= config.minimum_information_gain_bits {
            rejected_experiments.push(rejected(
                experiment,
                RejectionReason::NoPredictedInformationGain,
            ));
        } else {
            ranked_experiments.push(candidate);
        }
    }

    ranked_experiments.sort_by(|left, right| {
        right
            .utility
            .total_cmp(&left.utility)
            .then_with(|| {
                right
                    .expected_information_gain_bits
                    .total_cmp(&left.expected_information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    rejected_experiments.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    let selected = ranked_experiments.first().cloned();
    let abstention = match &selected {
        Some(_) => None,
        None if experiments.is_empty() => Some(AbstentionReason::NoEligibleExperiment),
        None if rejected_experiments
            .iter()
            .all(|item| item.reasons == [RejectionReason::NoPredictedInformationGain]) =>
        {
            Some(AbstentionReason::NoDiscriminatingExperiment)
        }
        None => Some(AbstentionReason::NoEligibleExperiment),
    };

    Ok(InvestigationPlan {
        selected,
        ranked_experiments,
        rejected_experiments,
        abstention,
        prior_entropy_bits: prior_entropy,
        grounded_hypothesis_ids,
        claim_ceiling: PlannerClaimCeiling::default(),
    })
}

fn probability_map<'a>(
    distribution: &'a [Probability],
) -> Result<BTreeMap<&'a str, f64>, PlannerError> {
    let mut result = BTreeMap::new();
    for item in distribution {
        if item.hypothesis_id.trim().is_empty()
            || !probability_valid(item.probability)
            || result
                .insert(item.hypothesis_id.as_str(), item.probability)
                .is_some()
        {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidMeasuredDistribution,
                "posterior distributions require unique hypothesis IDs and valid probabilities",
            ));
        }
    }
    if !distribution_valid(result.values().copied()) {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidMeasuredDistribution,
            "prior and posterior probability mass must each sum to 1.0",
        ));
    }
    Ok(result)
}

/// Calculate observed posterior entropy reduction after an independent Evidence
/// observation has arrived. This does not authenticate the supplied provenance
/// label or establish that the posterior itself is calibrated.
pub fn measure_observed_information_gain(
    evidence: &[EvidenceProvenance],
    prior: &[Probability],
    posterior: &[Probability],
    observation_evidence_ids: &[String],
) -> Result<MeasuredInformationGain, PlannerError> {
    let catalog = validate_evidence_catalog(evidence)?;
    if prior.is_empty() || posterior.is_empty() || observation_evidence_ids.is_empty() {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidMeasuredDistribution,
            "prior, posterior and observation Evidence references must be non-empty",
        ));
    }

    let prior_map = probability_map(prior)?;
    let posterior_map = probability_map(posterior)?;
    if prior_map.keys().ne(posterior_map.keys()) {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidMeasuredDistribution,
            "prior and posterior must refer to the same hypothesis IDs",
        ));
    }

    let mut visiting = HashSet::new();
    let mut memo = HashMap::new();
    let mut independent_root_ids = BTreeSet::new();
    let mut root_visited = HashSet::new();
    for id in observation_evidence_ids {
        if !catalog.contains_key(id.as_str())
            || !evidence_has_independent_root(id, &catalog, &mut visiting, &mut memo)?
        {
            return Err(PlannerError::new(
                PlannerErrorKind::UnqualifiedObservationEvidence,
                format!(
                    "observed information gain requires independently rooted Evidence; {id} does not qualify"
                ),
            ));
        }
        collect_independent_roots(id, &catalog, &mut independent_root_ids, &mut root_visited)?;
    }

    let prior_entropy = entropy_bits(prior_map.values().copied());
    let posterior_entropy = entropy_bits(posterior_map.values().copied());
    Ok(MeasuredInformationGain {
        prior_entropy_bits: prior_entropy,
        posterior_entropy_bits: posterior_entropy,
        observed_entropy_reduction_bits: prior_entropy - posterior_entropy,
        observation_evidence_ids: observation_evidence_ids.to_vec(),
        independent_root_evidence_ids: independent_root_ids.into_iter().collect(),
        evidence_provenance_authenticated: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence_id(byte: char) -> String {
        format!("ev_{}", byte.to_string().repeat(64))
    }

    fn evidence() -> Vec<EvidenceProvenance> {
        vec![EvidenceProvenance {
            evidence_id: evidence_id('a'),
            origin: EvidenceOrigin::IndependentObservation,
            parent_evidence_ids: Vec::new(),
        }]
    }

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "cause-a".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: Vec::new(),
            },
            Hypothesis {
                id: "cause-b".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: Vec::new(),
            },
        ]
    }

    fn discriminating_experiment(id: &str) -> ExperimentProposal {
        ExperimentProposal {
            id: id.to_owned(),
            outcomes: vec!["yes".to_owned(), "no".to_owned()],
            likelihoods: vec![
                LikelihoodRow {
                    hypothesis_id: "cause-a".to_owned(),
                    outcome_probabilities: vec![0.9, 0.1],
                },
                LikelihoodRow {
                    hypothesis_id: "cause-b".to_owned(),
                    outcome_probabilities: vec![0.1, 0.9],
                },
            ],
            authorization: Authorization::Authorized,
            reversible: true,
            destructive: false,
            target_impact: TargetImpact::ReadOnlyObservation,
            estimated_time_seconds: 10.0,
            estimated_energy_joules: 10.0,
            estimated_risk: 0.01,
        }
    }

    #[test]
    fn symmetric_binary_experiment_has_known_expected_information_gain() {
        let plan = plan_next_experiment(
            &evidence(),
            &hypotheses(),
            &[discriminating_experiment("probe-a")],
            PlannerConfig::default(),
        )
        .unwrap();
        let selected = plan.selected.unwrap();
        assert!((selected.expected_information_gain_bits - 0.5310044064107188).abs() < 1.0e-12);
        assert_eq!(selected.predicted_outcomes.len(), 2);
        assert!((selected.predicted_outcomes[0].posterior[0].1 - 0.9).abs() < 1.0e-12);
        assert!((selected.predicted_outcomes[0].posterior[1].1 - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn uninformative_experiment_is_rejected_and_planner_abstains() {
        let mut experiment = discriminating_experiment("no-signal");
        experiment.likelihoods[0].outcome_probabilities = vec![0.5, 0.5];
        experiment.likelihoods[1].outcome_probabilities = vec![0.5, 0.5];
        let plan = plan_next_experiment(
            &evidence(),
            &hypotheses(),
            &[experiment],
            PlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(AbstentionReason::NoDiscriminatingExperiment)
        );
        assert_eq!(
            plan.rejected_experiments[0].reasons,
            vec![RejectionReason::NoPredictedInformationGain]
        );
    }

    #[test]
    fn model_generated_and_synthetic_evidence_never_ground_hypotheses() {
        for origin in [
            EvidenceOrigin::ModelGenerated,
            EvidenceOrigin::Synthetic,
            EvidenceOrigin::ExternalReference,
        ] {
            let records = vec![EvidenceProvenance {
                evidence_id: evidence_id('b'),
                origin,
                parent_evidence_ids: Vec::new(),
            }];
            let mut candidates = hypotheses();
            for candidate in &mut candidates {
                candidate.supporting_evidence_ids = vec![evidence_id('b')];
            }
            let plan = plan_next_experiment(
                &records,
                &candidates,
                &[discriminating_experiment("probe")],
                PlannerConfig::default(),
            )
            .unwrap();
            assert!(plan.selected.is_none());
            assert_eq!(
                plan.abstention,
                Some(AbstentionReason::NoIndependentlyGroundedHypothesis)
            );
        }
    }

    #[test]
    fn independently_derived_evidence_requires_a_real_independent_parent() {
        let records = vec![
            evidence()[0].clone(),
            EvidenceProvenance {
                evidence_id: evidence_id('b'),
                origin: EvidenceOrigin::IndependentDerivation,
                parent_evidence_ids: vec![evidence_id('a')],
            },
        ];
        let mut candidates = hypotheses();
        for candidate in &mut candidates {
            candidate.supporting_evidence_ids = vec![evidence_id('b')];
        }
        let plan = plan_next_experiment(
            &records,
            &candidates,
            &[discriminating_experiment("probe")],
            PlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_some());
    }

    #[test]
    fn dangling_evidence_and_derivation_cycles_are_rejected() {
        let mut candidates = hypotheses();
        candidates[0].supporting_evidence_ids = vec![evidence_id('b')];
        assert_eq!(
            plan_next_experiment(
                &evidence(),
                &candidates,
                &[discriminating_experiment("probe")],
                PlannerConfig::default()
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::DanglingEvidenceReference
        );

        let records = vec![
            EvidenceProvenance {
                evidence_id: evidence_id('a'),
                origin: EvidenceOrigin::IndependentDerivation,
                parent_evidence_ids: vec![evidence_id('b')],
            },
            EvidenceProvenance {
                evidence_id: evidence_id('b'),
                origin: EvidenceOrigin::IndependentDerivation,
                parent_evidence_ids: vec![evidence_id('a')],
            },
        ];
        assert_eq!(
            plan_next_experiment(
                &records,
                &hypotheses(),
                &[discriminating_experiment("probe")],
                PlannerConfig::default()
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::EvidenceGraphCycle
        );
    }

    #[test]
    fn unauthorized_unknown_irreversible_destructive_and_over_budget_probes_are_never_selected() {
        let mut items = Vec::new();
        let mut unknown = discriminating_experiment("unknown-auth");
        unknown.authorization = Authorization::Unknown;
        items.push(unknown);
        let mut denied = discriminating_experiment("denied");
        denied.authorization = Authorization::Denied;
        items.push(denied);
        let mut irreversible = discriminating_experiment("irreversible");
        irreversible.reversible = false;
        items.push(irreversible);
        let mut destructive = discriminating_experiment("destructive");
        destructive.destructive = true;
        items.push(destructive);
        let mut high_impact = discriminating_experiment("high-impact");
        high_impact.target_impact = TargetImpact::DataMutation;
        items.push(high_impact);
        let mut over_time = discriminating_experiment("over-time");
        over_time.estimated_time_seconds = 9999.0;
        items.push(over_time);
        let mut over_energy = discriminating_experiment("over-energy");
        over_energy.estimated_energy_joules = 9999.0;
        items.push(over_energy);
        let mut over_risk = discriminating_experiment("over-risk");
        over_risk.estimated_risk = 0.9;
        items.push(over_risk);

        let plan =
            plan_next_experiment(&evidence(), &hypotheses(), &items, PlannerConfig::default())
                .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(plan.rejected_experiments.len(), items.len());
        assert!(plan.ranked_experiments.is_empty());
    }

    #[test]
    fn multi_outcome_multi_hypothesis_likelihoods_are_normalized_and_ranked() {
        let mut records = evidence();
        records.push(EvidenceProvenance {
            evidence_id: evidence_id('b'),
            origin: EvidenceOrigin::IndependentObservation,
            parent_evidence_ids: Vec::new(),
        });
        let candidates = vec![
            Hypothesis {
                id: "a".to_owned(),
                prior_probability: 0.4,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: vec![],
            },
            Hypothesis {
                id: "b".to_owned(),
                prior_probability: 0.35,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: vec![],
            },
            Hypothesis {
                id: "c".to_owned(),
                prior_probability: 0.25,
                supporting_evidence_ids: vec![evidence_id('b')],
                contradicting_evidence_ids: vec![],
            },
        ];
        let mut experiment = discriminating_experiment("three-way");
        experiment.outcomes = vec!["x".to_owned(), "y".to_owned(), "z".to_owned()];
        experiment.likelihoods = vec![
            LikelihoodRow {
                hypothesis_id: "a".to_owned(),
                outcome_probabilities: vec![0.8, 0.1, 0.1],
            },
            LikelihoodRow {
                hypothesis_id: "b".to_owned(),
                outcome_probabilities: vec![0.1, 0.8, 0.1],
            },
            LikelihoodRow {
                hypothesis_id: "c".to_owned(),
                outcome_probabilities: vec![0.1, 0.1, 0.8],
            },
        ];
        let plan = plan_next_experiment(
            &records,
            &candidates,
            &[experiment],
            PlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.unwrap().expected_information_gain_bits > 0.0);
    }

    #[test]
    fn malformed_likelihood_rows_are_reported_as_rejected() {
        let mut experiment = discriminating_experiment("bad-row");
        experiment.likelihoods[0].outcome_probabilities = vec![0.8, 0.8];
        let plan = plan_next_experiment(
            &evidence(),
            &hypotheses(),
            &[experiment],
            PlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.rejected_experiments[0].reasons,
            vec![RejectionReason::InvalidLikelihoods]
        );
    }

    #[test]
    fn deterministic_tie_break_uses_experiment_id() {
        let plan = plan_next_experiment(
            &evidence(),
            &hypotheses(),
            &[
                discriminating_experiment("z-probe"),
                discriminating_experiment("a-probe"),
            ],
            PlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(plan.selected.unwrap().experiment_id, "a-probe");
    }

    #[test]
    fn one_hypothesis_abstains_instead_of_inventing_a_competition() {
        let mut only = hypotheses();
        only.pop();
        let plan = plan_next_experiment(
            &evidence(),
            &only,
            &[discriminating_experiment("probe")],
            PlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(AbstentionReason::NoCompetingHypotheses)
        );
    }

    #[test]
    fn observed_information_gain_requires_independent_evidence() {
        let prior = vec![
            Probability {
                hypothesis_id: "cause-a".to_owned(),
                probability: 0.5,
            },
            Probability {
                hypothesis_id: "cause-b".to_owned(),
                probability: 0.5,
            },
        ];
        let posterior = vec![
            Probability {
                hypothesis_id: "cause-a".to_owned(),
                probability: 0.9,
            },
            Probability {
                hypothesis_id: "cause-b".to_owned(),
                probability: 0.1,
            },
        ];
        let measured =
            measure_observed_information_gain(&evidence(), &prior, &posterior, &[evidence_id('a')])
                .unwrap();
        assert!((measured.observed_entropy_reduction_bits - 0.5310044064107188).abs() < 1.0e-12);
        assert!(!measured.evidence_provenance_authenticated);

        let generated = vec![EvidenceProvenance {
            evidence_id: evidence_id('c'),
            origin: EvidenceOrigin::ModelGenerated,
            parent_evidence_ids: vec![],
        }];
        assert_eq!(
            measure_observed_information_gain(&generated, &prior, &posterior, &[evidence_id('c')],)
                .unwrap_err()
                .kind,
            PlannerErrorKind::UnqualifiedObservationEvidence
        );
    }

    #[test]
    fn duplicate_hypothesis_experiment_and_evidence_ids_are_rejected() {
        let mut duplicate_hypotheses = hypotheses();
        duplicate_hypotheses[1].id = duplicate_hypotheses[0].id.clone();
        assert_eq!(
            plan_next_experiment(
                &evidence(),
                &duplicate_hypotheses,
                &[discriminating_experiment("probe")],
                PlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::DuplicateHypothesisId
        );

        assert_eq!(
            plan_next_experiment(
                &evidence(),
                &hypotheses(),
                &[
                    discriminating_experiment("same"),
                    discriminating_experiment("same"),
                ],
                PlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::DuplicateExperimentId
        );

        let duplicate_evidence = vec![evidence()[0].clone(), evidence()[0].clone()];
        assert_eq!(
            plan_next_experiment(
                &duplicate_evidence,
                &hypotheses(),
                &[discriminating_experiment("probe")],
                PlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::DuplicateEvidenceId
        );
    }

    #[test]
    fn invalid_planner_configuration_is_rejected() {
        let mut config = PlannerConfig::default();
        config.max_time_seconds = 0.0;
        assert_eq!(
            plan_next_experiment(&evidence(), &hypotheses(), &[], config)
                .unwrap_err()
                .kind,
            PlannerErrorKind::InvalidPlannerConfig
        );
    }

    #[test]
    fn invalid_prior_mass_is_rejected() {
        let mut candidates = hypotheses();
        candidates[0].prior_probability = 0.7;
        assert_eq!(
            plan_next_experiment(
                &evidence(),
                &candidates,
                &[discriminating_experiment("probe")],
                PlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidHypothesisDistribution
        );
    }
}

/// Robust expected / worst-case / lower-tail EIG planning over likelihood scenarios.
pub mod robust;

/// Decision-aware expected value of sample information and minimax-regret planning.
pub mod decision;

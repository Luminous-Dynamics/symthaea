#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

const EPSILON: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceClass {
    SyntheticKnownTruth,
    CalibrationReceipt,
    PhysicalMeasurement,
    IndependentLaboratory,
    Simulation,
    Surrogate,
    LiteratureReported,
    VendorClaim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceAdmission {
    Candidate,
    AdmittedForResearch,
    AdmittedForEvaluation,
    Blocked,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvidenceRecord {
    pub evidence_id: String,
    pub source_digest: String,
    pub evidence_class: EvidenceClass,
    pub admission: EvidenceAdmission,
    pub claim_scope: String,
    pub validity_domain: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbabilityBasis {
    pub model_id: String,
    pub calibration_evidence_id: String,
    pub validity_domain: String,
    pub calibration_profile: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionAlternative {
    pub alternative_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlternativeLoss {
    pub alternative_id: String,
    pub loss: f64,
    pub feasible: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct World {
    pub world_id: String,
    /// None means set-valued / distribution-free analysis. Never infer a probability.
    pub prior_probability: Option<f64>,
    pub losses: Vec<AlternativeLoss>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutcomeDisposition {
    Observed,
    Failed,
    Censored,
    Invalid,
    Missing,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeLikelihood {
    pub outcome_id: String,
    /// Some(p) is a calibrated likelihood. None means only that the outcome is possible.
    pub probability: Option<f64>,
    pub disposition: OutcomeDisposition,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WorldOutcomes {
    pub world_id: String,
    pub outcomes: Vec<OutcomeLikelihood>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskClass {
    Low,
    Moderate,
    High,
    Unknown,
    Prohibited,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Experiment {
    pub experiment_id: String,
    pub evidence_id: String,
    pub decision_loss_cost: f64,
    pub cost_usd: f64,
    pub duration_hours: f64,
    pub material_grams: f64,
    pub risk_class: RiskClass,
    pub outcomes_by_world: Vec<WorldOutcomes>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkCase {
    pub case_id: String,
    pub decision_loss_unit: String,
    pub alternatives: Vec<DecisionAlternative>,
    pub worlds: Vec<World>,
    pub probability_basis: Option<ProbabilityBasis>,
    pub evidence: Vec<EvidenceRecord>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExperimentBudget {
    pub max_decision_loss_cost: f64,
    pub max_cost_usd: f64,
    pub max_duration_hours: f64,
    pub max_material_grams: f64,
    pub max_risk_class: RiskClass,
    pub allow_no_experiment: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionChoice {
    pub alternative_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutcomeDecisionValue {
    pub outcome_id: String,
    pub predictive_probability: f64,
    pub posterior_choice: DecisionChoice,
    pub posterior_expected_loss: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvsiReport {
    pub case_id: String,
    pub experiment_id: String,
    pub baseline_choice: DecisionChoice,
    pub baseline_expected_loss: f64,
    pub expected_post_observation_loss: f64,
    pub gross_evsi: f64,
    pub decision_loss_cost: f64,
    pub net_decision_value: f64,
    pub outcome_values: Vec<OutcomeDecisionValue>,
    pub world_outcome_hypothesis_count: usize,
    pub failed_world_outcome_hypothesis_count: usize,
    pub censored_world_outcome_hypothesis_count: usize,
    pub invalid_world_outcome_hypothesis_count: usize,
    pub missing_world_outcome_hypothesis_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustOutcomeValue {
    pub outcome_id: String,
    pub possible_world_count: usize,
    pub posterior_worst_case_regret: f64,
    pub posterior_choice: DecisionChoice,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MinimaxRegretReport {
    pub case_id: String,
    pub experiment_id: String,
    pub baseline_choice: DecisionChoice,
    pub baseline_worst_case_regret: f64,
    pub worst_post_observation_regret: f64,
    pub robust_regret_reduction: f64,
    pub decision_loss_cost: f64,
    pub net_robust_value: f64,
    pub outcome_values: Vec<RobustOutcomeValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExperimentSelection {
    Select {
        experiment_id: String,
        net_value_microunits: i64,
        reason: String,
    },
    NoExperiment {
        reason: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuditError(pub String);

impl std::fmt::Display for AuditError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AuditError {}

fn require_nonempty(value: &str, label: &str) -> Result<(), AuditError> {
    if value.trim().is_empty() {
        return Err(AuditError(format!("{label} must not be empty")));
    }
    Ok(())
}

fn validate_unique<'a>(
    values: impl IntoIterator<Item = &'a str>,
    label: &str,
) -> Result<(), AuditError> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_nonempty(value, label)?;
        if !seen.insert(value) {
            return Err(AuditError(format!("duplicate {label}: {value}")));
        }
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), AuditError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(AuditError(format!(
            "{label} must be a lowercase 64-character SHA-256 digest"
        )));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, label: &str) -> Result<(), AuditError> {
    if !value.is_finite() || value < 0.0 {
        return Err(AuditError(format!("{label} must be finite and non-negative")));
    }
    Ok(())
}

fn evidence_registry(
    evidence: &[EvidenceRecord],
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    if evidence.is_empty() {
        return Err(AuditError("evidence registry must not be empty".into()));
    }
    validate_unique(
        evidence.iter().map(|record| record.evidence_id.as_str()),
        "evidence ID",
    )?;
    let mut registry = BTreeMap::new();
    for record in evidence {
        validate_digest(&record.source_digest, "evidence source digest")?;
        require_nonempty(&record.claim_scope, "evidence claim scope")?;
        require_nonempty(&record.validity_domain, "evidence validity domain")?;
        registry.insert(record.evidence_id.as_str(), record);
    }
    Ok(registry)
}

fn require_evidence<'a>(
    registry: &BTreeMap<&'a str, &'a EvidenceRecord>,
    evidence_id: &str,
    owner: &str,
) -> Result<&'a EvidenceRecord, AuditError> {
    let record = registry.get(evidence_id).copied().ok_or_else(|| {
        AuditError(format!("{owner} references unknown evidence {evidence_id}"))
    })?;
    if record.admission != EvidenceAdmission::AdmittedForEvaluation {
        return Err(AuditError(format!(
            "{owner} references evidence {evidence_id} that is not evaluation-admitted"
        )));
    }
    Ok(record)
}

fn validate_case(
    case: &BenchmarkCase,
) -> Result<BTreeMap<&str, &EvidenceRecord>, AuditError> {
    require_nonempty(&case.case_id, "case ID")?;
    require_nonempty(&case.decision_loss_unit, "decision-loss unit")?;
    if case.alternatives.is_empty() || case.worlds.is_empty() {
        return Err(AuditError(
            "at least one decision alternative and one world are required".into(),
        ));
    }
    validate_unique(
        case.alternatives
            .iter()
            .map(|alternative| alternative.alternative_id.as_str()),
        "alternative ID",
    )?;
    validate_unique(
        case.worlds.iter().map(|world| world.world_id.as_str()),
        "world ID",
    )?;
    let expected: BTreeSet<&str> = case
        .alternatives
        .iter()
        .map(|alternative| alternative.alternative_id.as_str())
        .collect();

    for world in &case.worlds {
        let mut actual = BTreeSet::new();
        for loss in &world.losses {
            require_nonempty(&loss.alternative_id, "loss alternative ID")?;
            validate_nonnegative(loss.loss, "decision loss")?;
            if !actual.insert(loss.alternative_id.as_str()) {
                return Err(AuditError(format!(
                    "world {} duplicates loss for {}",
                    world.world_id, loss.alternative_id
                )));
            }
        }
        if actual != expected {
            return Err(AuditError(format!(
                "world {} has missing or unexpected alternative losses",
                world.world_id
            )));
        }
        if let Some(probability) = world.prior_probability {
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err(AuditError(format!(
                    "world {} has an invalid prior probability",
                    world.world_id
                )));
            }
        }
    }

    let registry = evidence_registry(&case.evidence)?;
    if let Some(basis) = &case.probability_basis {
        require_nonempty(&basis.model_id, "probability model ID")?;
        require_nonempty(&basis.validity_domain, "probability validity domain")?;
        require_nonempty(&basis.calibration_profile, "probability calibration profile")?;
        let calibration = require_evidence(
            &registry,
            &basis.calibration_evidence_id,
            "probability basis",
        )?;
        if !matches!(
            calibration.evidence_class,
            EvidenceClass::SyntheticKnownTruth | EvidenceClass::CalibrationReceipt
        ) {
            return Err(AuditError(
                "probability basis must reference a known-truth or calibration receipt".into(),
            ));
        }
    }
    Ok(registry)
}

fn loss_for<'a>(
    world: &'a World,
    alternative_id: &str,
) -> Result<&'a AlternativeLoss, AuditError> {
    world
        .losses
        .iter()
        .find(|loss| loss.alternative_id == alternative_id)
        .ok_or_else(|| {
            AuditError(format!(
                "world {} is missing loss for {}",
                world.world_id, alternative_id
            ))
        })
}

fn validate_experiment(
    case: &BenchmarkCase,
    experiment: &Experiment,
    registry: &BTreeMap<&str, &EvidenceRecord>,
) -> Result<(), AuditError> {
    require_nonempty(&experiment.experiment_id, "experiment ID")?;
    validate_nonnegative(experiment.decision_loss_cost, "decision-loss cost")?;
    validate_nonnegative(experiment.cost_usd, "experiment USD cost")?;
    validate_nonnegative(experiment.duration_hours, "experiment duration")?;
    validate_nonnegative(experiment.material_grams, "experiment material")?;
    require_evidence(registry, &experiment.evidence_id, "experiment")?;

    validate_unique(
        experiment
            .outcomes_by_world
            .iter()
            .map(|item| item.world_id.as_str()),
        "experiment world ID",
    )?;
    let expected_worlds: BTreeSet<&str> =
        case.worlds.iter().map(|world| world.world_id.as_str()).collect();
    let actual_worlds: BTreeSet<&str> = experiment
        .outcomes_by_world
        .iter()
        .map(|item| item.world_id.as_str())
        .collect();
    if expected_worlds != actual_worlds {
        return Err(AuditError(
            "experiment outcome table does not cover every decision world exactly".into(),
        ));
    }

    let probability_modes: BTreeSet<bool> = experiment
        .outcomes_by_world
        .iter()
        .flat_map(|row| row.outcomes.iter().map(|outcome| outcome.probability.is_some()))
        .collect();
    if probability_modes.len() > 1 {
        return Err(AuditError(
            "an experiment cannot mix calibrated likelihoods and set-valued outcomes".into(),
        ));
    }

    let is_probability_model = probability_modes.contains(&true);
    if is_probability_model != case.probability_basis.is_some() {
        return Err(AuditError(
            "probability-bearing outcomes require an explicit probability basis; set-valued outcomes do not".into(),
        ));
    }

    let mut canonical_outcomes: Option<BTreeSet<&str>> = None;
    for row in &experiment.outcomes_by_world {
        validate_unique(
            row.outcomes.iter().map(|outcome| outcome.outcome_id.as_str()),
            "outcome ID within world",
        )?;
        if row.outcomes.is_empty() {
            return Err(AuditError(format!(
                "world {} has no declared experiment outcomes",
                row.world_id
            )));
        }
        let current: BTreeSet<&str> = row
            .outcomes
            .iter()
            .map(|outcome| outcome.outcome_id.as_str())
            .collect();
        if let Some(expected) = &canonical_outcomes {
            if expected != &current {
                return Err(AuditError(
                    "probability experiment must declare the same outcome IDs in every world"
                        .into(),
                ));
            }
        } else if is_probability_model {
            canonical_outcomes = Some(current);
        }

        let total = row
            .outcomes
            .iter()
            .map(|outcome| {
                if let Some(probability) = outcome.probability {
                    if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                        return Err(AuditError(format!(
                            "world {} has an invalid outcome probability",
                            row.world_id
                        )));
                    }
                    Ok(probability)
                } else {
                    Ok(0.0)
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();

        if is_probability_model && (total - 1.0).abs() > EPSILON {
            return Err(AuditError(format!(
                "world {} likelihoods sum to {total}, not 1",
                row.world_id
            )));
        }
    }

    if is_probability_model {
        let prior_total = case
            .worlds
            .iter()
            .map(|world| {
                world.prior_probability.ok_or_else(|| {
                    AuditError(format!(
                        "world {} has no prior probability in EVSI mode",
                        world.world_id
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<f64>();
        if (prior_total - 1.0).abs() > EPSILON {
            return Err(AuditError(format!(
                "world prior probabilities sum to {prior_total}, not 1"
            )));
        }
    } else if case.worlds.iter().any(|world| world.prior_probability.is_some()) {
        return Err(AuditError(
            "set-valued mode must not contain probability-bearing world priors".into(),
        ));
    }
    Ok(())
}

fn expected_decision(
    case: &BenchmarkCase,
    probabilities: &BTreeMap<&str, f64>,
) -> Result<(DecisionChoice, f64), AuditError> {
    let mut best: Option<(String, f64)> = None;
    for alternative in &case.alternatives {
        let mut expected_loss = 0.0;
        let mut feasible_everywhere = true;
        for world in &case.worlds {
            let probability = probabilities.get(world.world_id.as_str()).copied().unwrap_or(0.0);
            if probability <= EPSILON {
                continue;
            }
            let loss = loss_for(world, &alternative.alternative_id)?;
            if !loss.feasible {
                feasible_everywhere = false;
                break;
            }
            expected_loss += probability * loss.loss;
        }
        if feasible_everywhere {
            let candidate = (alternative.alternative_id.clone(), expected_loss);
            if best.as_ref().is_none_or(|current| {
                candidate.1 < current.1 - EPSILON
                    || ((candidate.1 - current.1).abs() <= EPSILON
                        && candidate.0 < current.0)
            }) {
                best = Some(candidate);
            }
        }
    }
    let (alternative_id, loss) = best.ok_or_else(|| {
        AuditError("no decision alternative is feasible across the current support".into())
    })?;
    Ok((DecisionChoice { alternative_id }, loss))
}

fn world_outcome<'a>(
    experiment: &'a Experiment,
    world_id: &str,
) -> Result<&'a WorldOutcomes, AuditError> {
    experiment
        .outcomes_by_world
        .iter()
        .find(|row| row.world_id == world_id)
        .ok_or_else(|| AuditError(format!("missing outcomes for world {world_id}")))
}

/// Evaluate expected value of sample information for a probability-bearing experiment.
///
/// This function requires explicit, evaluation-admitted priors and likelihoods. It does not
/// infer probabilities from intervals, ensembles, bounds, entropy scores or surrogate variance.
pub fn evaluate_evsi(
    case: &BenchmarkCase,
    experiment: &Experiment,
) -> Result<EvsiReport, AuditError> {
    let registry = validate_case(case)?;
    validate_experiment(case, experiment, &registry)?;
    if case.probability_basis.is_none() {
        return Err(AuditError(
            "EVSI requires a declared calibrated probability basis".into(),
        ));
    }

    let prior: BTreeMap<&str, f64> = case
        .worlds
        .iter()
        .map(|world| {
            Ok((
                world.world_id.as_str(),
                world.prior_probability.ok_or_else(|| {
                    AuditError(format!(
                        "world {} has no prior probability",
                        world.world_id
                    ))
                })?,
            ))
        })
        .collect::<Result<_, AuditError>>()?;

    let (baseline_choice, baseline_expected_loss) = expected_decision(case, &prior)?;
    let mut predictive_probabilities = BTreeMap::<String, f64>::new();
    for world in &case.worlds {
        let prior_probability = prior[world.world_id.as_str()];
        for outcome in &world_outcome(experiment, &world.world_id)?.outcomes {
            let likelihood = outcome.probability.ok_or_else(|| {
                AuditError("missing likelihood in probability-bearing experiment".into())
            })?;
            *predictive_probabilities
                .entry(outcome.outcome_id.clone())
                .or_insert(0.0) += prior_probability * likelihood;
        }
    }

    let mut outcome_values = Vec::new();
    let mut expected_post_observation_loss = 0.0;
    for (outcome_id, predictive_probability) in &predictive_probabilities {
        if *predictive_probability <= EPSILON {
            continue;
        }
        let mut posterior = BTreeMap::<&str, f64>::new();
        for world in &case.worlds {
            let likelihood = world_outcome(experiment, &world.world_id)?
                .outcomes
                .iter()
                .find(|outcome| outcome.outcome_id == *outcome_id)
                .and_then(|outcome| outcome.probability)
                .ok_or_else(|| AuditError("missing outcome likelihood".into()))?;
            posterior.insert(
                world.world_id.as_str(),
                prior[world.world_id.as_str()] * likelihood / predictive_probability,
            );
        }
        let (posterior_choice, posterior_expected_loss) = expected_decision(case, &posterior)?;
        expected_post_observation_loss += predictive_probability * posterior_expected_loss;
        outcome_values.push(OutcomeDecisionValue {
            outcome_id: outcome_id.clone(),
            predictive_probability: *predictive_probability,
            posterior_choice,
            posterior_expected_loss,
        });
    }
    outcome_values.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));

    let gross_evsi = baseline_expected_loss - expected_post_observation_loss;
    let dispositions: Vec<OutcomeDisposition> = experiment
        .outcomes_by_world
        .iter()
        .flat_map(|row| row.outcomes.iter().map(|outcome| outcome.disposition))
        .collect();
    let world_outcome_hypothesis_count = dispositions.len();
    let failed_world_outcome_hypothesis_count = dispositions
        .iter()
        .filter(|item| **item == OutcomeDisposition::Failed)
        .count();
    let censored_world_outcome_hypothesis_count = dispositions
        .iter()
        .filter(|item| **item == OutcomeDisposition::Censored)
        .count();
    let invalid_world_outcome_hypothesis_count = dispositions
        .iter()
        .filter(|item| **item == OutcomeDisposition::Invalid)
        .count();
    let missing_world_outcome_hypothesis_count = dispositions
        .iter()
        .filter(|item| **item == OutcomeDisposition::Missing)
        .count();

    Ok(EvsiReport {
        case_id: case.case_id.clone(),
        experiment_id: experiment.experiment_id.clone(),
        baseline_choice,
        baseline_expected_loss,
        expected_post_observation_loss,
        gross_evsi,
        decision_loss_cost: experiment.decision_loss_cost,
        net_decision_value: gross_evsi - experiment.decision_loss_cost,
        outcome_values,
        world_outcome_hypothesis_count,
        failed_world_outcome_hypothesis_count,
        censored_world_outcome_hypothesis_count,
        invalid_world_outcome_hypothesis_count,
        missing_world_outcome_hypothesis_count,
    })
}

fn worst_case_regret(
    case: &BenchmarkCase,
    supported_world_ids: &BTreeSet<&str>,
) -> Result<(DecisionChoice, f64), AuditError> {
    let supported_worlds: Vec<&World> = case
        .worlds
        .iter()
        .filter(|world| supported_world_ids.contains(world.world_id.as_str()))
        .collect();
    if supported_worlds.is_empty() {
        return Err(AuditError("decision support is empty".into()));
    }

    let mut best: Option<(String, f64)> = None;
    for alternative in &case.alternatives {
        let mut feasible_everywhere = true;
        let mut max_regret: f64 = 0.0;
        for world in &supported_worlds {
            let candidate_loss = loss_for(world, &alternative.alternative_id)?;
            if !candidate_loss.feasible {
                feasible_everywhere = false;
                break;
            }
            let oracle_loss = world
                .losses
                .iter()
                .filter(|loss| loss.feasible)
                .map(|loss| loss.loss)
                .fold(f64::INFINITY, f64::min);
            if !oracle_loss.is_finite() {
                return Err(AuditError(format!(
                    "world {} has no feasible decision alternative",
                    world.world_id
                )));
            }
            max_regret = max_regret.max(candidate_loss.loss - oracle_loss);
        }
        if feasible_everywhere {
            let candidate = (alternative.alternative_id.clone(), max_regret);
            if best.as_ref().is_none_or(|current| {
                candidate.1 < current.1 - EPSILON
                    || ((candidate.1 - current.1).abs() <= EPSILON
                        && candidate.0 < current.0)
            }) {
                best = Some(candidate);
            }
        }
    }
    let (alternative_id, regret) = best.ok_or_else(|| {
        AuditError("no alternative is robustly feasible across the current world set".into())
    })?;
    Ok((DecisionChoice { alternative_id }, regret))
}

/// Evaluate distribution-free minimax-regret reduction over an explicit set of possible worlds.
///
/// Outcome likelihoods must be absent. The reported worst post-observation regret is the maximum
/// across every possible experiment outcome, not an average or an invented probability.
pub fn evaluate_minimax_regret(
    case: &BenchmarkCase,
    experiment: &Experiment,
) -> Result<MinimaxRegretReport, AuditError> {
    let registry = validate_case(case)?;
    validate_experiment(case, experiment, &registry)?;
    if experiment
        .outcomes_by_world
        .iter()
        .flat_map(|row| row.outcomes.iter())
        .any(|outcome| outcome.probability.is_some())
    {
        return Err(AuditError(
            "minimax-regret mode requires set-valued outcomes without probabilities".into(),
        ));
    }

    let all_world_ids: BTreeSet<&str> = case
        .worlds
        .iter()
        .map(|world| world.world_id.as_str())
        .collect();
    let (baseline_choice, baseline_worst_case_regret) =
        worst_case_regret(case, &all_world_ids)?;

    let outcome_ids: BTreeSet<&str> = experiment
        .outcomes_by_world
        .iter()
        .flat_map(|row| row.outcomes.iter().map(|outcome| outcome.outcome_id.as_str()))
        .collect();
    if outcome_ids.is_empty() {
        return Err(AuditError("experiment has no possible outcomes".into()));
    }

    let mut outcome_values = Vec::new();
    let mut worst_post_observation_regret: f64 = 0.0;
    for outcome_id in outcome_ids {
        let supported_world_ids: BTreeSet<&str> = experiment
            .outcomes_by_world
            .iter()
            .filter(|row| row.outcomes.iter().any(|outcome| outcome.outcome_id == outcome_id))
            .map(|row| row.world_id.as_str())
            .collect();
        let (posterior_choice, posterior_worst_case_regret) =
            worst_case_regret(case, &supported_world_ids)?;
        worst_post_observation_regret =
            worst_post_observation_regret.max(posterior_worst_case_regret);
        outcome_values.push(RobustOutcomeValue {
            outcome_id: outcome_id.to_string(),
            possible_world_count: supported_world_ids.len(),
            posterior_worst_case_regret,
            posterior_choice,
        });
    }
    outcome_values.sort_by(|left, right| left.outcome_id.cmp(&right.outcome_id));

    let robust_regret_reduction =
        baseline_worst_case_regret - worst_post_observation_regret;
    Ok(MinimaxRegretReport {
        case_id: case.case_id.clone(),
        experiment_id: experiment.experiment_id.clone(),
        baseline_choice,
        baseline_worst_case_regret,
        worst_post_observation_regret,
        robust_regret_reduction,
        decision_loss_cost: experiment.decision_loss_cost,
        net_robust_value: robust_regret_reduction - experiment.decision_loss_cost,
        outcome_values,
    })
}

fn risk_is_admissible(candidate: RiskClass, maximum: RiskClass) -> bool {
    matches!(
        (candidate, maximum),
        (RiskClass::Low, _)
            | (RiskClass::Moderate, RiskClass::Moderate | RiskClass::High)
            | (RiskClass::High, RiskClass::High)
    )
}

fn quantize_microunits(value: f64) -> Result<i64, AuditError> {
    if !value.is_finite() {
        return Err(AuditError("net decision value is not finite".into()));
    }
    let scaled = (value * 1_000_000.0).round();
    if scaled < i64::MIN as f64 || scaled > i64::MAX as f64 {
        return Err(AuditError("net decision value exceeds report range".into()));
    }
    Ok(scaled as i64)
}

/// Select the highest positive net value from already-evaluated reports under a strict budget.
/// The caller must evaluate all experiment candidates using the same case and evaluation mode.
pub fn select_experiment(
    experiments: &[Experiment],
    net_values: &BTreeMap<String, f64>,
    budget: &ExperimentBudget,
) -> Result<ExperimentSelection, AuditError> {
    for (label, value) in [
        ("decision-loss budget", budget.max_decision_loss_cost),
        ("USD budget", budget.max_cost_usd),
        ("duration budget", budget.max_duration_hours),
        ("material budget", budget.max_material_grams),
    ] {
        validate_nonnegative(value, label)?;
    }
    validate_unique(
        experiments.iter().map(|experiment| experiment.experiment_id.as_str()),
        "experiment ID",
    )?;

    let mut eligible: Vec<(&Experiment, f64)> = Vec::new();
    for experiment in experiments {
        let net_value = *net_values.get(&experiment.experiment_id).ok_or_else(|| {
            AuditError(format!(
                "experiment {} has no evaluated net value",
                experiment.experiment_id
            ))
        })?;
        if !net_value.is_finite() {
            return Err(AuditError(format!(
                "experiment {} has non-finite net value",
                experiment.experiment_id
            )));
        }
        if experiment.decision_loss_cost > budget.max_decision_loss_cost
            || experiment.cost_usd > budget.max_cost_usd
            || experiment.duration_hours > budget.max_duration_hours
            || experiment.material_grams > budget.max_material_grams
            || !risk_is_admissible(experiment.risk_class, budget.max_risk_class)
        {
            continue;
        }
        if net_value > EPSILON {
            eligible.push((experiment, net_value));
        }
    }

    eligible.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.experiment_id.cmp(&right.0.experiment_id))
    });
    if let Some((experiment, net_value)) = eligible.first() {
        return Ok(ExperimentSelection::Select {
            experiment_id: experiment.experiment_id.clone(),
            net_value_microunits: quantize_microunits(*net_value)?,
            reason: "highest positive evaluated decision value within declared cost, time, material and risk budgets".into(),
        });
    }
    if budget.allow_no_experiment {
        Ok(ExperimentSelection::NoExperiment {
            reason: "no admissible experiment has positive net decision value within the declared budget".into(),
        })
    } else {
        Err(AuditError(
            "no experiment has positive net decision value within the declared budget".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn evidence() -> Vec<EvidenceRecord> {
        vec![
            EvidenceRecord {
                evidence_id: "CALIBRATION-KNOWN-TRUTH".into(),
                source_digest: digest('a'),
                evidence_class: EvidenceClass::SyntheticKnownTruth,
                admission: EvidenceAdmission::AdmittedForEvaluation,
                claim_scope: "synthetic calibration likelihoods".into(),
                validity_domain: "known-truth fixture only".into(),
            },
            EvidenceRecord {
                evidence_id: "EXPERIMENT-ESTIMATES".into(),
                source_digest: digest('b'),
                evidence_class: EvidenceClass::SyntheticKnownTruth,
                admission: EvidenceAdmission::AdmittedForEvaluation,
                claim_scope: "synthetic experiment outcome tables".into(),
                validity_domain: "known-truth fixture only".into(),
            },
        ]
    }

    fn base_case(probability_mode: bool) -> BenchmarkCase {
        BenchmarkCase {
            case_id: "TWO-DESIGN-TRADEOFF".into(),
            decision_loss_unit: "normalized_mission_loss".into(),
            alternatives: vec![
                DecisionAlternative {
                    alternative_id: "DESIGN-A".into(),
                },
                DecisionAlternative {
                    alternative_id: "DESIGN-B".into(),
                },
                DecisionAlternative {
                    alternative_id: "ABSTAIN".into(),
                },
            ],
            worlds: vec![
                World {
                    world_id: "WORLD-A".into(),
                    prior_probability: probability_mode.then_some(0.5),
                    losses: vec![
                        AlternativeLoss {
                            alternative_id: "DESIGN-A".into(),
                            loss: 0.0,
                            feasible: true,
                        },
                        AlternativeLoss {
                            alternative_id: "DESIGN-B".into(),
                            loss: 10.0,
                            feasible: true,
                        },
                        AlternativeLoss {
                            alternative_id: "ABSTAIN".into(),
                            loss: 6.0,
                            feasible: true,
                        },
                    ],
                },
                World {
                    world_id: "WORLD-B".into(),
                    prior_probability: probability_mode.then_some(0.5),
                    losses: vec![
                        AlternativeLoss {
                            alternative_id: "DESIGN-A".into(),
                            loss: 10.0,
                            feasible: true,
                        },
                        AlternativeLoss {
                            alternative_id: "DESIGN-B".into(),
                            loss: 0.0,
                            feasible: true,
                        },
                        AlternativeLoss {
                            alternative_id: "ABSTAIN".into(),
                            loss: 6.0,
                            feasible: true,
                        },
                    ],
                },
            ],
            probability_basis: probability_mode.then(|| ProbabilityBasis {
                model_id: "SYNTHETIC-EXACT-PRIOR".into(),
                calibration_evidence_id: "CALIBRATION-KNOWN-TRUTH".into(),
                validity_domain: "two-world synthetic test only".into(),
                calibration_profile: "exact fixture frequencies".into(),
            }),
            evidence: evidence(),
        }
    }

    fn experiment(
        id: &str,
        cost: f64,
        risk_class: RiskClass,
        likelihoods: [[f64; 2]; 2],
    ) -> Experiment {
        Experiment {
            experiment_id: id.into(),
            evidence_id: "EXPERIMENT-ESTIMATES".into(),
            decision_loss_cost: cost,
            cost_usd: 100.0,
            duration_hours: 2.0,
            material_grams: 1.0,
            risk_class,
            outcomes_by_world: vec![
                WorldOutcomes {
                    world_id: "WORLD-A".into(),
                    outcomes: vec![
                        OutcomeLikelihood {
                            outcome_id: "RESULT-A".into(),
                            probability: Some(likelihoods[0][0]),
                            disposition: OutcomeDisposition::Observed,
                        },
                        OutcomeLikelihood {
                            outcome_id: "RESULT-B".into(),
                            probability: Some(likelihoods[0][1]),
                            disposition: OutcomeDisposition::Observed,
                        },
                    ],
                },
                WorldOutcomes {
                    world_id: "WORLD-B".into(),
                    outcomes: vec![
                        OutcomeLikelihood {
                            outcome_id: "RESULT-A".into(),
                            probability: Some(likelihoods[1][0]),
                            disposition: OutcomeDisposition::Observed,
                        },
                        OutcomeLikelihood {
                            outcome_id: "RESULT-B".into(),
                            probability: Some(likelihoods[1][1]),
                            disposition: OutcomeDisposition::Observed,
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn decision_irrelevant_high_entropy_experiment_has_zero_evsi() {
        let case = base_case(true);
        let irrelevant = experiment(
            "HIGH-ENTROPY-IRRELEVANT",
            0.0,
            RiskClass::Low,
            [[0.5, 0.5], [0.5, 0.5]],
        );
        let report = evaluate_evsi(&case, &irrelevant).expect("valid experiment");
        assert!((report.gross_evsi).abs() <= EPSILON);
        assert!((report.expected_post_observation_loss - 5.0).abs() <= EPSILON);
        assert_eq!(report.baseline_choice.alternative_id, "DESIGN-A");
    }

    #[test]
    fn decision_boundary_experiment_reduces_expected_loss() {
        let case = base_case(true);
        let informative = experiment(
            "DECISION-BOUNDARY",
            0.5,
            RiskClass::Low,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let report = evaluate_evsi(&case, &informative).expect("valid experiment");
        assert!((report.baseline_expected_loss - 5.0).abs() <= EPSILON);
        assert!((report.expected_post_observation_loss - 1.0).abs() <= EPSILON);
        assert!((report.gross_evsi - 4.0).abs() <= EPSILON);
        assert!((report.net_decision_value - 3.5).abs() <= EPSILON);
    }

    #[test]
    fn failed_censored_invalid_and_missing_outcomes_remain_accounted() {
        let mut case = base_case(true);
        let mut item = experiment(
            "MIXED-OUTCOMES",
            0.1,
            RiskClass::Low,
            [[0.7, 0.3], [0.2, 0.8]],
        );
        item.outcomes_by_world[0].outcomes[0].disposition = OutcomeDisposition::Failed;
        item.outcomes_by_world[0].outcomes[1].disposition = OutcomeDisposition::Censored;
        item.outcomes_by_world[1].outcomes[0].disposition = OutcomeDisposition::Invalid;
        item.outcomes_by_world[1].outcomes[1].disposition = OutcomeDisposition::Missing;
        case.worlds.reverse();
        let report = evaluate_evsi(&case, &item).expect("outcome statuses remain auditable");
        assert_eq!(report.world_outcome_hypothesis_count, 4);
        assert_eq!(report.failed_world_outcome_hypothesis_count, 1);
        assert_eq!(report.censored_world_outcome_hypothesis_count, 1);
        assert_eq!(report.invalid_world_outcome_hypothesis_count, 1);
        assert_eq!(report.missing_world_outcome_hypothesis_count, 1);
    }

    #[test]
    fn interval_or_set_valued_inputs_cannot_be_scored_as_evsi() {
        let case = base_case(false);
        let mut set_experiment = experiment(
            "SET-VALUED",
            0.0,
            RiskClass::Low,
            [[0.5, 0.5], [0.5, 0.5]],
        );
        for row in &mut set_experiment.outcomes_by_world {
            for outcome in &mut row.outcomes {
                outcome.probability = None;
            }
        }
        assert!(evaluate_evsi(&case, &set_experiment).is_err());
    }

    #[test]
    fn minimax_regret_uses_possible_worlds_without_probabilities() {
        let case = base_case(false);
        let experiment = Experiment {
            experiment_id: "PERFECT-DISCRIMINATOR".into(),
            evidence_id: "EXPERIMENT-ESTIMATES".into(),
            decision_loss_cost: 1.0,
            cost_usd: 100.0,
            duration_hours: 2.0,
            material_grams: 1.0,
            risk_class: RiskClass::Low,
            outcomes_by_world: vec![
                WorldOutcomes {
                    world_id: "WORLD-A".into(),
                    outcomes: vec![OutcomeLikelihood {
                        outcome_id: "OUTCOME-A".into(),
                        probability: None,
                        disposition: OutcomeDisposition::Observed,
                    }],
                },
                WorldOutcomes {
                    world_id: "WORLD-B".into(),
                    outcomes: vec![OutcomeLikelihood {
                        outcome_id: "OUTCOME-B".into(),
                        probability: None,
                        disposition: OutcomeDisposition::Observed,
                    }],
                },
            ],
        };
        let report = evaluate_minimax_regret(&case, &experiment).expect("valid set-valued case");
        assert_eq!(report.baseline_choice.alternative_id, "ABSTAIN");
        assert!((report.baseline_worst_case_regret - 6.0).abs() <= EPSILON);
        assert!(report.worst_post_observation_regret.abs() <= EPSILON);
        assert!((report.robust_regret_reduction - 6.0).abs() <= EPSILON);
        assert!((report.net_robust_value - 5.0).abs() <= EPSILON);
    }

    #[test]
    fn probability_rows_must_sum_to_one() {
        let case = base_case(true);
        let mut item = experiment(
            "INVALID-LIKELIHOOD",
            0.0,
            RiskClass::Low,
            [[0.8, 0.1], [0.5, 0.5]],
        );
        item.outcomes_by_world[0].outcomes[0].probability = Some(0.8);
        assert!(evaluate_evsi(&case, &item).is_err());
    }

    #[test]
    fn probability_model_requires_calibration_evidence() {
        let mut case = base_case(true);
        case.evidence[0].admission = EvidenceAdmission::AdmittedForResearch;
        let item = experiment(
            "CALIBRATION-GATED",
            0.0,
            RiskClass::Low,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        assert!(evaluate_evsi(&case, &item).is_err());
    }

    #[test]
    fn high_risk_or_over_budget_experiment_is_not_selected() {
        let case = base_case(true);
        let good = experiment(
            "GOOD",
            1.0,
            RiskClass::Low,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let risky = experiment(
            "RISKY",
            100.0,
            RiskClass::High,
            [[1.0, 0.0], [0.0, 1.0]],
        );
        let good_value = evaluate_evsi(&case, &good).expect("good experiment").net_decision_value;
        let risky_value = evaluate_evsi(&case, &risky).expect("risky experiment").net_decision_value;
        let budget = ExperimentBudget {
            max_decision_loss_cost: 2.0,
            max_cost_usd: 200.0,
            max_duration_hours: 3.0,
            max_material_grams: 2.0,
            max_risk_class: RiskClass::Moderate,
            allow_no_experiment: true,
        };
        let values = BTreeMap::from([
            ("GOOD".to_string(), good_value),
            ("RISKY".to_string(), risky_value),
        ]);
        let result = select_experiment(&[risky, good], &values, &budget).expect("selection works");
        assert!(matches!(result, ExperimentSelection::Select { experiment_id, .. } if experiment_id == "GOOD"));
    }

    #[test]
    fn zero_net_value_keeps_no_experiment_option() {
        let item = experiment(
            "NO-VALUE",
            1.0,
            RiskClass::Low,
            [[0.5, 0.5], [0.5, 0.5]],
        );
        let values = BTreeMap::from([("NO-VALUE".to_string(), -1.0)]);
        let budget = ExperimentBudget {
            max_decision_loss_cost: 2.0,
            max_cost_usd: 200.0,
            max_duration_hours: 3.0,
            max_material_grams: 2.0,
            max_risk_class: RiskClass::Moderate,
            allow_no_experiment: true,
        };
        let result = select_experiment(&[item], &values, &budget).expect("no experiment is allowed");
        assert!(matches!(result, ExperimentSelection::NoExperiment { .. }));
    }

    #[test]
    fn deterministic_evsi_is_invariant_to_world_and_outcome_order() {
        let case = base_case(true);
        let item = experiment(
            "ORDER-INVARIANT",
            0.5,
            RiskClass::Low,
            [[0.9, 0.1], [0.1, 0.9]],
        );
        let first = evaluate_evsi(&case, &item).expect("first evaluation");
        let mut permuted_case = case;
        permuted_case.worlds.reverse();
        let mut permuted_item = item;
        permuted_item.outcomes_by_world.reverse();
        for row in &mut permuted_item.outcomes_by_world {
            row.outcomes.reverse();
        }
        let second = evaluate_evsi(&permuted_case, &permuted_item).expect("permuted evaluation");
        assert!((first.gross_evsi - second.gross_evsi).abs() <= EPSILON);
        assert_eq!(first.baseline_choice, second.baseline_choice);
        assert_eq!(
            first.outcome_values.iter().map(|item| &item.outcome_id).collect::<Vec<_>>(),
            second.outcome_values.iter().map(|item| &item.outcome_id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_design_that_is_infeasible_in_any_supported_world_cannot_be_selected() {
        let mut case = base_case(true);
        let loss = case.worlds[1]
            .losses
            .iter_mut()
            .find(|loss| loss.alternative_id == "DESIGN-A")
            .expect("design A loss");
        loss.feasible = false;
        let item = experiment(
            "CONSTRAINT-CLIFF",
            0.0,
            RiskClass::Low,
            [[1.0, 0.0], [0.0, 1.0]],
        );
        let report = evaluate_evsi(&case, &item).expect("fallback remains feasible");
        assert_eq!(report.baseline_choice.alternative_id, "ABSTAIN");
    }
}

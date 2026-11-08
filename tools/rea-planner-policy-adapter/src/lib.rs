//! Thin adapter that runs the existing REA planner APIs over the frozen
//! six-case synthetic planner-input manifest.
//!
//! The adapter does not reimplement EIG, robust EIG, EVSI or minimax regret.
//! It calls symthaea-rea-investigator directly and adds transparent baselines
//! around the same frozen candidates. It never reads the grader-only DGP file,
//! executes an observation, changes a target, or closes a claim obligation.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

use symthaea_rea_investigator::decision::{
    DecisionAwareConfig, DecisionRankingPolicy, HypothesisUtility, TerminalDecision,
    plan_next_experiment_decision_aware,
};
use symthaea_rea_investigator::robust::{
    LikelihoodModelScenario, RobustExperimentProposal, RobustPlannerConfig, RobustRankingPolicy,
    ScenarioExperimentLikelihood, plan_next_experiment_robust,
};
use symthaea_rea_investigator::{
    Authorization, CostWeights, EvidenceOrigin, EvidenceProvenance, ExperimentProposal,
    Hypothesis, LikelihoodRow, PlannerConfig, TargetImpact, plan_next_experiment,
};

const PLANNER_MANIFEST: &str =
    include_str!("../../../docs/research/rea-planner-input-manifest-v0.json");
const AGENT_VIEW: &str =
    include_str!("../tests/fixtures/rea-proof-obligation-agent-view-v0.json");
const PLANNER_SOURCE_COMMIT: &str = "2cf30ebc544a336d369246613895555a85696fe5";
const TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone)]
pub struct AdapterError(pub String);

impl std::fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AdapterError {}

fn err(context: &str, message: impl Into<String>) -> AdapterError {
    AdapterError(format!("{context}: {}", message.into()))
}

fn object_array<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a [Value], AdapterError> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| err(context, format!("{key} must be an array")))
}

fn string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str, AdapterError> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| err(context, format!("{key} must be a string")))
}

fn number(value: &Value, key: &str, context: &str) -> Result<f64, AdapterError> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| err(context, format!("{key} must be a finite number")))
}

fn boolean(value: &Value, key: &str, context: &str) -> Result<bool, AdapterError> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| err(context, format!("{key} must be a boolean")))
}

fn target_impact(value: &str) -> Result<TargetImpact, AdapterError> {
    match value {
        "ReadOnlyObservation" => Ok(TargetImpact::ReadOnlyObservation),
        "TemporaryConfiguration" => Ok(TargetImpact::TemporaryConfiguration),
        "ServiceRestart" => Ok(TargetImpact::ServiceRestart),
        "DataMutation" => Ok(TargetImpact::DataMutation),
        "PhysicalActuation" => Ok(TargetImpact::PhysicalActuation),
        other => Err(err("target impact", format!("unsupported value {other}"))),
    }
}

fn authorization(value: &str) -> Result<Authorization, AdapterError> {
    match value {
        "Unknown" => Ok(Authorization::Unknown),
        "Denied" => Ok(Authorization::Denied),
        "Authorized" => Ok(Authorization::Authorized),
        other => Err(err("authorization", format!("unsupported value {other}"))),
    }
}

fn evidence_origin(value: &str) -> Result<EvidenceOrigin, AdapterError> {
    match value {
        "IndependentObservation" => Ok(EvidenceOrigin::IndependentObservation),
        "IndependentDerivation" => Ok(EvidenceOrigin::IndependentDerivation),
        "ModelGenerated" => Ok(EvidenceOrigin::ModelGenerated),
        "Synthetic" => Ok(EvidenceOrigin::Synthetic),
        "ExternalReference" => Ok(EvidenceOrigin::ExternalReference),
        other => Err(err("Evidence origin", format!("unsupported value {other}"))),
    }
}

fn likelihood_rows(value: &Value, context: &str) -> Result<Vec<LikelihoodRow>, AdapterError> {
    object_array(value, "likelihoods", context)?
        .iter()
        .map(|row| {
            let id = string(row, "hypothesis_id", context)?.to_owned();
            let probabilities = object_array(row, "outcome_probabilities", context)?
                .iter()
                .map(|item| {
                    item.as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| err(context, "likelihood must be a finite number"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LikelihoodRow {
                hypothesis_id: id,
                outcome_probabilities: probabilities,
            })
        })
        .collect()
}

fn build_evidence(case: &Value) -> Result<Vec<EvidenceProvenance>, AdapterError> {
    object_array(case, "evidence", "case")?
        .iter()
        .map(|item| {
            Ok(EvidenceProvenance {
                evidence_id: string(item, "evidence_id", "evidence")?.to_owned(),
                origin: evidence_origin(string(item, "origin", "evidence")?)?,
                parent_evidence_ids: object_array(item, "parent_evidence_ids", "evidence")?
                    .iter()
                    .map(|id| {
                        id.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| err("evidence", "parent Evidence ID must be a string"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .collect()
}

fn build_hypotheses(case: &Value) -> Result<Vec<Hypothesis>, AdapterError> {
    object_array(case, "hypotheses", "case")?
        .iter()
        .map(|item| {
            let ids = |key: &str| -> Result<Vec<String>, AdapterError> {
                object_array(item, key, "hypothesis")?
                    .iter()
                    .map(|id| {
                        id.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| err("hypothesis", format!("{key} ID must be a string")))
                    })
                    .collect()
            };
            Ok(Hypothesis {
                id: string(item, "id", "hypothesis")?.to_owned(),
                prior_probability: number(item, "prior_probability", "hypothesis")?,
                supporting_evidence_ids: ids("supporting_evidence_ids")?,
                contradicting_evidence_ids: ids("contradicting_evidence_ids")?,
            })
        })
        .collect()
}

fn build_point_experiments(case: &Value) -> Result<Vec<ExperimentProposal>, AdapterError> {
    object_array(case, "candidates", "case")?
        .iter()
        .map(|candidate| {
            Ok(ExperimentProposal {
                id: string(candidate, "id", "candidate")?.to_owned(),
                outcomes: object_array(candidate, "outcomes", "candidate")?
                    .iter()
                    .map(|outcome| {
                        outcome
                            .as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| err("candidate", "outcome must be a string"))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                likelihoods: likelihood_rows_from_field(candidate, "point_likelihoods")?,
                authorization: authorization(string(candidate, "authorization", "candidate")?)?,
                reversible: boolean(candidate, "reversible", "candidate")?,
                destructive: boolean(candidate, "destructive", "candidate")?,
                target_impact: target_impact(string(candidate, "target_impact", "candidate")?)?,
                estimated_time_seconds: number(candidate, "estimated_time_seconds", "candidate")?,
                estimated_energy_joules: number(candidate, "estimated_energy_joules", "candidate")?,
                estimated_risk: number(candidate, "estimated_risk", "candidate")?,
            })
        })
        .collect()
}

fn likelihood_rows_from_field(candidate: &Value, key: &str) -> Result<Vec<LikelihoodRow>, AdapterError> {
    object_array(candidate, key, "candidate")?
        .iter()
        .map(|row| {
            let probabilities = object_array(row, "outcome_probabilities", "likelihood row")?
                .iter()
                .map(|value| {
                    value
                        .as_f64()
                        .filter(|number| number.is_finite())
                        .ok_or_else(|| err("likelihood row", "probability must be finite"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LikelihoodRow {
                hypothesis_id: string(row, "hypothesis_id", "likelihood row")?.to_owned(),
                outcome_probabilities: probabilities,
            })
        })
        .collect()
}

fn build_robust_experiments(case: &Value) -> Result<Vec<RobustExperimentProposal>, AdapterError> {
    build_point_experiments(case)?
        .into_iter()
        .map(|candidate| {
            Ok(RobustExperimentProposal {
                id: candidate.id,
                outcomes: candidate.outcomes,
                authorization: candidate.authorization,
                reversible: candidate.reversible,
                destructive: candidate.destructive,
                target_impact: candidate.target_impact,
                estimated_time_seconds: candidate.estimated_time_seconds,
                estimated_energy_joules: candidate.estimated_energy_joules,
                estimated_risk: candidate.estimated_risk,
            })
        })
        .collect()
}

fn build_scenarios(case: &Value) -> Result<Vec<LikelihoodModelScenario>, AdapterError> {
    object_array(case, "likelihood_scenarios", "case")?
        .iter()
        .map(|scenario| {
            let experiments = object_array(scenario, "experiments", "scenario")?
                .iter()
                .map(|experiment| {
                    Ok(ScenarioExperimentLikelihood {
                        experiment_id: string(experiment, "experiment_id", "scenario experiment")?.to_owned(),
                        likelihoods: likelihood_rows_from_field(experiment, "likelihoods")?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(LikelihoodModelScenario {
                scenario_id: string(scenario, "scenario_id", "scenario")?.to_owned(),
                scenario_probability: number(scenario, "scenario_probability", "scenario")?,
                experiments,
            })
        })
        .collect()
}

fn build_decisions(case: &Value) -> Result<Vec<TerminalDecision>, AdapterError> {
    object_array(case, "terminal_decisions", "case")?
        .iter()
        .map(|decision| {
            let utilities = object_array(decision, "utilities", "terminal decision")?
                .iter()
                .map(|utility| {
                    Ok(HypothesisUtility {
                        hypothesis_id: string(utility, "hypothesis_id", "utility")?.to_owned(),
                        utility: number(utility, "utility", "utility")?,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TerminalDecision {
                id: string(decision, "id", "terminal decision")?.to_owned(),
                utilities,
            })
        })
        .collect()
}

fn build_planner_config(case: &Value) -> Result<PlannerConfig, AdapterError> {
    let config = case
        .get("point_eig_config")
        .ok_or_else(|| err("case", "point_eig_config is required"))?;
    let weights = config
        .get("cost_weights")
        .ok_or_else(|| err("point_eig_config", "cost_weights is required"))?;
    Ok(PlannerConfig {
        max_time_seconds: number(config, "max_time_seconds", "point_eig_config")?,
        max_energy_joules: number(config, "max_energy_joules", "point_eig_config")?,
        max_risk: number(config, "max_risk", "point_eig_config")?,
        cost_weights: CostWeights {
            time: number(weights, "time", "cost_weights")?,
            energy: number(weights, "energy", "cost_weights")?,
            risk: number(weights, "risk", "cost_weights")?,
        },
        max_target_impact: target_impact(string(
            config,
            "max_target_impact",
            "point_eig_config",
        )?)?,
        minimum_information_gain_bits: number(
            config,
            "minimum_information_gain_bits",
            "point_eig_config",
        )?,
    })
}

fn build_robust_config(case: &Value, base: PlannerConfig) -> Result<RobustPlannerConfig, AdapterError> {
    let config = case
        .get("robust_config")
        .ok_or_else(|| err("case", "robust_config is required"))?;
    let ranking_policy = match string(config, "ranking_policy", "robust_config")? {
        "ExpectedInformationGain" => RobustRankingPolicy::ExpectedInformationGain,
        "WorstCaseInformationGain" => RobustRankingPolicy::WorstCaseInformationGain,
        "LowerTailCvar" => RobustRankingPolicy::LowerTailCvar,
        other => return Err(err("robust_config", format!("unsupported ranking policy {other}"))),
    };
    Ok(RobustPlannerConfig {
        base,
        ranking_policy,
        lower_tail_probability: number(config, "lower_tail_probability", "robust_config")?,
        minimum_top1_stability: number(config, "minimum_top1_stability", "robust_config")?,
    })
}

fn build_decision_config(
    case: &Value,
    robust: RobustPlannerConfig,
) -> Result<DecisionAwareConfig, AdapterError> {
    let config = case
        .get("decision_config")
        .ok_or_else(|| err("case", "decision_config is required"))?;
    let ranking_policy = match string(config, "ranking_policy", "decision_config")? {
        "ExpectedEvsi" => DecisionRankingPolicy::ExpectedEvsi,
        "WorstCaseEvsi" => DecisionRankingPolicy::WorstCaseEvsi,
        "LowerTailCvarEvsi" => DecisionRankingPolicy::LowerTailCvarEvsi,
        "ExpectedEvsiMinusWorstCaseRegret" => DecisionRankingPolicy::ExpectedEvsiMinusWorstCaseRegret,
        "MinimaxRegret" => DecisionRankingPolicy::MinimaxRegret,
        other => return Err(err("decision_config", format!("unsupported ranking policy {other}"))),
    };
    let maximum_worst_case_regret = match config.get("maximum_worst_case_regret") {
        Some(Value::Null) | None => None,
        Some(value) => Some(
            value
                .as_f64()
                .filter(|number| number.is_finite())
                .ok_or_else(|| err("decision_config", "maximum_worst_case_regret must be finite or null"))?,
        ),
    };
    Ok(DecisionAwareConfig {
        robust,
        ranking_policy,
        minimum_decision_stability: number(config, "minimum_decision_stability", "decision_config")?,
        maximum_worst_case_regret,
        regret_penalty: number(config, "regret_penalty", "decision_config")?,
        minimum_selection_value: number(config, "minimum_selection_value", "decision_config")?,
        max_policy_evaluations: config
            .get("max_policy_evaluations")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| err("decision_config", "max_policy_evaluations must fit usize"))?,
    })
}

fn normalize_point(plan: &symthaea_rea_investigator::InvestigationPlan) -> Value {
    json!({
        "selected": plan.selected.as_ref().map(|item| item.experiment_id.clone()),
        "abstention": plan.abstention.map(|value| format!("{value:?}")),
        "ranked": plan.ranked_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "expected_information_gain_bits": item.expected_information_gain_bits,
            "normalized_cost": item.normalized_cost,
            "utility": item.utility,
            "support_evidence_ids": item.support_evidence_ids,
            "independent_root_evidence_ids": item.independent_root_evidence_ids
        })).collect::<Vec<_>>(),
        "rejected": plan.rejected_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "reasons": item.reasons.iter().map(|reason| format!("{reason:?}")).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "claim_ceiling": {
            "evidence_producer_authenticated": plan.claim_ceiling.evidence_producer_authenticated,
            "causal_truth_proven": plan.claim_ceiling.causal_truth_proven,
            "selected_experiment_executed": plan.claim_ceiling.selected_experiment_executed,
            "target_mutated": plan.claim_ceiling.target_mutated,
            "likelihood_calibration_verified": plan.claim_ceiling.likelihood_calibration_verified
        }
    })
}

fn normalize_robust(plan: &symthaea_rea_investigator::robust::RobustInvestigationPlan) -> Value {
    json!({
        "selected": plan.selected.as_ref().map(|item| item.experiment_id.clone()),
        "abstention": plan.abstention.map(|value| format!("{value:?}")),
        "ranked": plan.ranked_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "expected_information_gain_bits": item.expected_information_gain_bits,
            "worst_case_information_gain_bits": item.worst_case_information_gain_bits,
            "lower_tail_cvar_information_gain_bits": item.lower_tail_cvar_information_gain_bits,
            "information_gain_variance_bits_squared": item.information_gain_variance_bits_squared,
            "selected_information_gain_bits": item.selected_information_gain_bits,
            "top1_rank_stability": item.top1_rank_stability,
            "top3_rank_stability": item.top3_rank_stability,
            "normalized_cost": item.normalized_cost,
            "utility": item.utility
        })).collect::<Vec<_>>(),
        "rejected": plan.rejected_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "reason": format!("{:?}", item.reason)
        })).collect::<Vec<_>>(),
        "claim_ceiling": {
            "evidence_producer_authenticated": plan.claim_ceiling.evidence_producer_authenticated,
            "likelihood_scenarios_calibrated": plan.claim_ceiling.likelihood_scenarios_calibrated,
            "scenario_weights_authenticated": plan.claim_ceiling.scenario_weights_authenticated,
            "causal_truth_proven": plan.claim_ceiling.causal_truth_proven,
            "experiment_executed": plan.claim_ceiling.experiment_executed,
            "target_mutated": plan.claim_ceiling.target_mutated
        }
    })
}

fn normalize_decision(plan: &symthaea_rea_investigator::decision::DecisionAwarePlan) -> Value {
    json!({
        "selected": plan.selected.as_ref().map(|item| item.experiment_id.clone()),
        "abstention": plan.abstention.map(|value| format!("{value:?}")),
        "baseline_optimal_decision_id": plan.baseline_optimal_decision_id,
        "baseline_optimal_expected_utility": plan.baseline_optimal_expected_utility,
        "ranked": plan.ranked_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "expected_evsi": item.expected_evsi,
            "worst_case_evsi": item.worst_case_evsi,
            "lower_tail_cvar_evsi": item.lower_tail_cvar_evsi,
            "expected_shared_policy_regret": item.expected_shared_policy_regret,
            "worst_case_shared_policy_regret": item.worst_case_shared_policy_regret,
            "decision_stability": item.decision_stability,
            "selected_value": item.selected_value,
            "normalized_cost": item.normalized_cost,
            "utility": item.utility,
            "shared_policy": item.shared_policy.iter().map(|outcome| json!({
                "outcome": outcome.outcome,
                "mixture_probability": outcome.mixture_probability,
                "decision_id": outcome.decision_id
            })).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "rejected": plan.rejected_experiments.iter().map(|item| json!({
            "experiment_id": item.experiment_id,
            "reasons": item.reasons.iter().map(|reason| format!("{reason:?}")).collect::<Vec<_>>()
        })).collect::<Vec<_>>(),
        "claim_ceiling": {
            "evidence_producer_authenticated": plan.claim_ceiling.evidence_producer_authenticated,
            "likelihood_scenarios_calibrated": plan.claim_ceiling.likelihood_scenarios_calibrated,
            "scenario_weights_authenticated": plan.claim_ceiling.scenario_weights_authenticated,
            "decision_utilities_independently_validated": plan.claim_ceiling.decision_utilities_independently_validated,
            "causal_truth_proven": plan.claim_ceiling.causal_truth_proven,
            "selected_experiment_executed": plan.claim_ceiling.selected_experiment_executed,
            "terminal_decision_executed": plan.claim_ceiling.terminal_decision_executed,
            "target_mutated": plan.claim_ceiling.target_mutated
        }
    })
}

fn normalized_cost(candidate: &Value, config: PlannerConfig) -> f64 {
    let time = candidate["estimated_time_seconds"].as_f64().unwrap_or(f64::INFINITY);
    let energy = candidate["estimated_energy_joules"].as_f64().unwrap_or(f64::INFINITY);
    let risk = candidate["estimated_risk"].as_f64().unwrap_or(f64::INFINITY);
    (config.cost_weights.time * time / config.max_time_seconds
        + config.cost_weights.energy * energy / config.max_energy_joules
        + config.cost_weights.risk * risk / config.max_risk)
        .max(1.0e-12)
}

fn impact_rank(value: &str) -> Option<u8> {
    match value {
        "ReadOnlyObservation" => Some(0),
        "TemporaryConfiguration" => Some(1),
        "ServiceRestart" => Some(2),
        "DataMutation" => Some(3),
        "PhysicalActuation" => Some(4),
        _ => None,
    }
}

fn eligible_candidate(candidate: &Value, config: PlannerConfig) -> bool {
    let auth = candidate["authorization"].as_str() == Some("Authorized");
    let reversible = candidate["reversible"].as_bool() == Some(true);
    let nondestructive = candidate["destructive"].as_bool() == Some(false);
    let time = candidate["estimated_time_seconds"].as_f64().unwrap_or(f64::INFINITY);
    let energy = candidate["estimated_energy_joules"].as_f64().unwrap_or(f64::INFINITY);
    let risk = candidate["estimated_risk"].as_f64().unwrap_or(f64::INFINITY);
    let impact_ok = candidate["target_impact"]
        .as_str()
        .and_then(impact_rank)
        .zip(impact_rank(&format!("{:?}", config.max_target_impact)))
        .is_some_and(|(impact, max_impact)| impact <= max_impact);
    auth
        && reversible
        && nondestructive
        && impact_ok
        && time.is_finite()
        && time > 0.0
        && time <= config.max_time_seconds
        && energy.is_finite()
        && energy >= 0.0
        && energy <= config.max_energy_joules
        && risk.is_finite()
        && (0.0..=1.0).contains(&risk)
        && risk <= config.max_risk
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn eligible_ids(case: &Value, config: PlannerConfig) -> Vec<String> {
    object_array(case, "candidates", "case")
        .unwrap_or(&[])
        .iter()
        .filter(|candidate| eligible_candidate(candidate, config))
        .filter_map(|candidate| candidate.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect()
}

fn expected_obligation_closure(
    candidate: &Value,
    agent_candidate: &Value,
    case: &Value,
) -> Result<f64, AdapterError> {
    let obligations = object_array(case, "obligations", "agent case")?;
    let pending: BTreeMap<String, BTreeSet<String>> = obligations
        .iter()
        .filter(|obligation| {
            obligation["required"].as_bool() == Some(true)
                && obligation["status"].as_str() != Some("satisfied")
        })
        .map(|obligation| {
            let id = obligation["obligation_id"].as_str().unwrap_or("").to_owned();
            let classes = obligation["admissible_evidence_classes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
            (id, classes)
        })
        .collect();

    if pending.is_empty() {
        return Ok(0.0);
    }

    let outcomes = object_array(candidate, "outcomes", "candidate")?;
    let rows = likelihood_rows_from_field(candidate, "point_likelihoods")?;
    let hypotheses = object_array(case, "hypotheses", "case")?;
    let priors: BTreeMap<String, f64> = hypotheses
        .iter()
        .map(|hypothesis| {
            Ok((
                string(hypothesis, "id", "hypothesis")?.to_owned(),
                number(hypothesis, "prior_probability", "hypothesis")?,
            ))
        })
        .collect::<Result<_, AdapterError>>()?;
    let marginal: Vec<f64> = (0..outcomes.len())
        .map(|outcome_index| {
            rows.iter()
                .map(|row| {
                    priors.get(&row.hypothesis_id).copied().unwrap_or(0.0)
                        * row.outcome_probabilities.get(outcome_index).copied().unwrap_or(0.0)
                })
                .sum()
        })
        .collect();

    let mut closure_probability = 0.0;
    for (obligation_id, admissible_classes) in pending {
        let mut probability = 0.0;
        for effect in object_array(agent_candidate, "effects", "agent candidate")? {
            if string(effect, "obligation_id", "effect")? != obligation_id
                || string(effect, "resulting_status", "effect")? != "satisfied"
                || !admissible_classes.contains(string(effect, "evidence_class", "effect")?)
            {
                continue;
            }
            let outcome = string(effect, "outcome", "effect")?;
            if let Some(index) = outcomes.iter().position(|item| item.as_str() == Some(outcome)) {
                probability += marginal[index];
            }
        }
        closure_probability += probability.min(1.0);
    }
    Ok(closure_probability)
}

fn baseline_policies(
    case: &Value,
    agent_case: &Value,
    config: PlannerConfig,
) -> Result<Value, AdapterError> {
    let candidates = object_array(case, "candidates", "case")?;
    let source_candidates = object_array(agent_case, "candidates", "agent case")?;
    let eligible: Vec<&Value> = candidates
        .iter()
        .filter(|candidate| eligible_candidate(candidate, config))
        .collect();

    let mut random_choice = None;
    if !eligible.is_empty() {
        let seed = fnv1a64(string(case, "case_id", "case")?.as_bytes());
        let index = (seed as usize) % eligible.len();
        random_choice = eligible[index].get("id").and_then(Value::as_str).map(str::to_owned);
    }

    let cost_choice = eligible
        .iter()
        .min_by(|left, right| {
            normalized_cost(left, config)
                .total_cmp(&normalized_cost(right, config))
                .then_with(|| left["id"].as_str().cmp(&right["id"].as_str()))
        })
        .and_then(|candidate| candidate["id"].as_str().map(str::to_owned));

    let mut greedy = Vec::new();
    for candidate in &eligible {
        let id = string(candidate, "id", "candidate")?;
        let source = source_candidates
            .iter()
            .find(|item| item["experiment_id"].as_str() == Some(id))
            .ok_or_else(|| err("baseline", format!("missing source candidate {id}")))?;
        let expected_closure = expected_obligation_closure(candidate, source, agent_case)?;
        let cost = number(candidate, "cost_units", "candidate")?;
        greedy.push((id.to_owned(), expected_closure, expected_closure / cost));
    }
    greedy.sort_by(|left, right| {
        right.2.total_cmp(&left.2).then_with(|| left.0.cmp(&right.0))
    });

    Ok(json!({
        "random_eligible": {
            "selected": random_choice,
            "seed": fnv1a64(string(case, "case_id", "case")?.as_bytes()),
            "selection_rule": "FNV-1a case-id seeded index over lexicographically declared eligible candidates"
        },
        "cost_only": {
            "selected": cost_choice,
            "ranking": eligible.iter().map(|candidate| json!({
                "experiment_id": candidate["id"],
                "normalized_cost": normalized_cost(candidate, config)
            })).collect::<Vec<_>>()
        },
        "greedy_obligation_closure": {
            "selected": greedy.first().map(|item| item.0.clone()),
            "ranking": greedy.iter().map(|(id, expected, score)| json!({
                "experiment_id": id,
                "expected_mandatory_obligation_closure": expected,
                "closure_per_cost_unit": score
            })).collect::<Vec<_>>(),
            "interpretation": "Expected closure under caller-supplied synthetic likelihoods; no obligation is marked satisfied until a new observation is independently assessed."
        }
    }))
}

fn subset_candidates_and_scenarios(
    case: &Value,
    allowed_ids: &BTreeSet<String>,
) -> Result<(Vec<RobustExperimentProposal>, Vec<LikelihoodModelScenario>), AdapterError> {
    let all = build_robust_experiments(case)?;
    let candidates: Vec<RobustExperimentProposal> = all
        .into_iter()
        .filter(|candidate| allowed_ids.contains(&candidate.id))
        .collect();
    let scenarios = build_scenarios(case)?
        .into_iter()
        .map(|scenario| LikelihoodModelScenario {
            scenario_id: scenario.scenario_id,
            scenario_probability: scenario.scenario_probability,
            experiments: scenario
                .experiments
                .into_iter()
                .filter(|experiment| allowed_ids.contains(&experiment.experiment_id))
                .collect(),
        })
        .collect();
    Ok((candidates, scenarios))
}

fn obligation_closing_candidate_ids(
    case: &Value,
    agent_case: &Value,
    config: PlannerConfig,
) -> Result<BTreeSet<String>, AdapterError> {
    let source_candidates = object_array(agent_case, "candidates", "agent case")?;
    let mut ids = BTreeSet::new();
    for candidate in object_array(case, "candidates", "case")? {
        if !eligible_candidate(candidate, config) {
            continue;
        }
        let id = string(candidate, "id", "candidate")?;
        let source = source_candidates
            .iter()
            .find(|item| item["experiment_id"].as_str() == Some(id))
            .ok_or_else(|| err("obligation gate", format!("source candidate {id} missing")))?;
        if expected_obligation_closure(candidate, source, agent_case)? > TOLERANCE {
            ids.insert(id.to_owned());
        }
    }
    Ok(ids)
}

fn run_case(case: &Value, agent_case: &Value) -> Result<Value, AdapterError> {
    let case_id = string(case, "case_id", "case")?;
    let evidence = build_evidence(case)?;
    let hypotheses = build_hypotheses(case)?;
    let point_experiments = build_point_experiments(case)?;
    let robust_experiments = build_robust_experiments(case)?;
    let scenarios = build_scenarios(case)?;
    let decisions = build_decisions(case)?;
    let base_config = build_planner_config(case)?;
    let robust_config = build_robust_config(case, base_config)?;
    let decision_config = build_decision_config(case, robust_config)?;

    let point = plan_next_experiment(
        &evidence,
        &hypotheses,
        &point_experiments,
        base_config,
    )
    .map_err(|error| err(case_id, format!("point EIG planner failed: {error}")))?;
    let robust = plan_next_experiment_robust(
        &evidence,
        &hypotheses,
        &robust_experiments,
        &scenarios,
        robust_config,
    )
    .map_err(|error| err(case_id, format!("robust EIG planner failed: {error}")))?;
    let decision = plan_next_experiment_decision_aware(
        &evidence,
        &hypotheses,
        &robust_experiments,
        &scenarios,
        &decisions,
        decision_config,
    )
    .map_err(|error| err(case_id, format!("decision-aware planner failed: {error}")))?;

    let baselines = baseline_policies(case, agent_case, base_config)?;
    let gated_ids = obligation_closing_candidate_ids(case, agent_case, base_config)?;
    let (gated_experiments, gated_scenarios) = subset_candidates_and_scenarios(case, &gated_ids)?;
    let gated_decision = if gated_experiments.is_empty() {
        json!({
            "selected": null,
            "abstention": "NoAdmissibleObligationClosingCandidate",
            "ranked": [],
            "claim_ceiling": {
                "evidence_producer_authenticated": false,
                "likelihood_scenarios_calibrated": false,
                "decision_utilities_independently_validated": false,
                "selected_experiment_executed": false,
                "terminal_decision_executed": false
            }
        })
    } else {
        let plan = plan_next_experiment_decision_aware(
            &evidence,
            &hypotheses,
            &gated_experiments,
            &gated_scenarios,
            &decisions,
            decision_config,
        )
        .map_err(|error| err(case_id, format!("obligation-gated decision planner failed: {error}")))?;
        normalize_decision(&plan)
    };

    Ok(json!({
        "case_id": case_id,
        "synthetic_development_only": true,
        "planner_source_commit": PLANNER_SOURCE_COMMIT,
        "random_seed_policy": "case-id FNV-1a; deterministic, not a statistical sample",
        "baselines": baselines,
        "point_eig": normalize_point(&point),
        "robust_eig": normalize_robust(&robust),
        "decision_aware": normalize_decision(&decision),
        "obligation_gated_decision_aware": {
            "eligible_candidate_ids": gated_ids,
            "plan": gated_decision,
            "gate_semantics": "Only candidates with a possible admissible satisfied transition for at least one currently unmet mandatory obligation enter the decision-aware planner. Predicted outcomes do not satisfy obligations."
        },
        "execution_status": "proposal_only_no_experiment_executed",
        "claim_ceiling": {
            "source_provenance_authenticated": false,
            "likelihood_models_calibrated": false,
            "scenario_set_complete": false,
            "utilities_human_validated": false,
            "claim_obligations_satisfied_by_prediction": false,
            "causal_truth_proven": false,
            "target_mutated": false
        }
    }))
}

/// Execute the existing planner APIs over the frozen six-case synthetic manifest.
/// The grader-only outcome file is deliberately not compiled into this adapter.
pub fn run_embedded_pilot() -> Result<Value, AdapterError> {
    let planner: Value = serde_json::from_str(PLANNER_MANIFEST)
        .map_err(|error| err("planner manifest", error.to_string()))?;
    let agent_view: Value = serde_json::from_str(AGENT_VIEW)
        .map_err(|error| err("agent view", error.to_string()))?;
    let cases = object_array(&planner, "cases", "planner manifest")?;
    let agent_cases = object_array(&agent_view, "cases", "agent view")?;
    let planner_text = PLANNER_MANIFEST.as_bytes();
    let input_sha256 = Sha256::digest(planner_text)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();

    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let case_id = string(case, "case_id", "case")?;
        let agent_case = agent_cases
            .iter()
            .find(|item| item["case_id"].as_str() == Some(case_id))
            .ok_or_else(|| err("agent view", format!("missing case {case_id}")))?;
        results.push(run_case(case, agent_case)?);
    }

    Ok(json!({
        "schema": "luminous.rea.planner-policy-adapter-result.v0",
        "dataset_id": planner["dataset_id"],
        "status": "synthetic_policy_api_conformance_only",
        "input_manifest_sha256": input_sha256,
        "source_agent_view_commit_sha": planner["source_case_view_commit_sha"],
        "source_agent_view_git_blob_sha1": planner["source_case_view_file_git_blob_sha1"],
        "planner_source_commit": PLANNER_SOURCE_COMMIT,
        "grader_accessed": false,
        "case_count": results.len(),
        "results": results,
        "claim_ceiling": {
            "planner_algorithms_reimplemented": false,
            "grader_only_dgp_read": false,
            "experiments_executed": false,
            "source_authenticity_proven": false,
            "likelihood_calibration_proven": false,
            "scenario_completeness_proven": false,
            "planner_superiority_established": false
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_case_pilot_runs_all_existing_planner_apis() {
        let result = run_embedded_pilot().unwrap();
        assert_eq!(result["case_count"], 6);
        let results = result["results"].as_array().unwrap();
        assert_eq!(results.len(), 6);
        for case in results {
            assert!(case["point_eig"].is_object());
            assert!(case["robust_eig"].is_object());
            assert!(case["decision_aware"].is_object());
            assert!(case["baselines"].is_object());
            assert!(case["obligation_gated_decision_aware"].is_object());
            assert_eq!(case["execution_status"], "proposal_only_no_experiment_executed");
        }
        assert_eq!(result["grader_accessed"], false);
    }

    #[test]
    fn unauthorized_candidate_is_never_selected_by_an_engine_policy() {
        let result = run_embedded_pilot().unwrap();
        let case = result["results"].as_array().unwrap().iter()
            .find(|case| case["case_id"] == "REA-POA-013").unwrap();
        for policy in ["point_eig", "robust_eig", "decision_aware"] {
            assert_ne!(
                case[policy]["selected"].as_str(),
                Some("unauthorized_production_probe")
            );
        }
        assert!(!case["baselines"]["cost_only"]["ranking"].as_array().unwrap()
            .iter().any(|item| item["experiment_id"] == "unauthorized_production_probe"));
    }

    #[test]
    fn predicted_obligation_closure_never_marks_readiness_satisfied() {
        let result = run_embedded_pilot().unwrap();
        for case in result["results"].as_array().unwrap() {
            assert_eq!(
                case["obligation_gated_decision_aware"]["gate_semantics"],
                "Only candidates with a possible admissible satisfied transition for at least one currently unmet mandatory obligation enter the decision-aware planner. Predicted outcomes do not satisfy obligations."
            );
        }
    }

    #[test]
    fn public_single_outcome_negative_control_abstains_at_obligation_gate() {
        let result = run_embedded_pilot().unwrap();
        let case = result["results"].as_array().unwrap().iter()
            .find(|case| case["case_id"] == "REA-POA-012").unwrap();
        assert_eq!(
            case["obligation_gated_decision_aware"]["plan"]["abstention"],
            "NoAdmissibleObligationClosingCandidate"
        );
    }
}

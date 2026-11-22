// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Finite-state, decision-specific expected value of sample information.
//!
//! This is a planner for proposing a next evidence request. It performs no I/O,
//! contacts no source and grants no authority. Rights, source authority, scope,
//! privacy approval and deadline feasibility are hard admissibility gates;
//! they are never converted into compensable scalar penalties.

use sha2::Digest;

const PROBABILITY_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug)]
pub(crate) struct Hypothesis {
    pub id: String,
    pub prior_probability: f64,
    /// Expected loss under Action A and Action B, respectively.
    pub action_losses: [f64; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct ObservationOutcome {
    pub id: String,
    /// P(this outcome | each hypothesis), in the same order as the prior.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Clone, Debug)]
pub(crate) struct EvidenceQuery {
    pub id: String,
    pub observation_model_digest: String,
    pub outcomes: Vec<ObservationOutcome>,
    /// All three costs must already be normalized to the decision-loss unit.
    pub acquisition_cost: f64,
    pub delay_cost: f64,
    /// Only an approved exposure burden; this never substitutes for privacy approval.
    pub disclosure_exposure_cost: f64,
    /// Upstream gate result, not a self-authenticating authority assertion.
    pub source_authorized: bool,
    /// Upstream decision-scope validator result.
    pub scope_valid: bool,
    /// Separate rights/privacy approval gate; exposure cost is not compensatory.
    pub privacy_approved: bool,
    /// Whether the request can return within the decision's allowed time horizon.
    pub deadline_feasible: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct VoiContext {
    pub hypotheses: Vec<Hypothesis>,
    pub queries: Vec<EvidenceQuery>,
    pub public_seed: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct QueryEvaluation {
    pub query_id: String,
    pub observation_model_digest: String,
    pub admissible: bool,
    pub rejection_reason: Option<String>,
    pub risk_before: f64,
    pub predictive_outcome_entropy_bits: Option<f64>,
    pub expected_post_query_risk: Option<f64>,
    pub expected_risk_reduction: Option<f64>,
    pub acquisition_cost: f64,
    pub delay_cost: f64,
    pub disclosure_exposure_cost: f64,
    pub total_cost: f64,
    pub net_evsi: Option<f64>,
    pub selected: bool,
    pub selected_by_uncertainty_only: bool,
    pub selected_by_random_control: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct VoiPlan {
    pub risk_before: f64,
    pub chosen_query_id: Option<String>,
    pub uncertainty_only_query_id: Option<String>,
    pub random_control_query_id: Option<String>,
    pub stop_reason: &'static str,
    pub evaluations: Vec<QueryEvaluation>,
}

fn validate_context(context: &VoiContext) -> Result<(), String> {
    if context.hypotheses.is_empty() {
        return Err("EVSI context requires at least one hypothesis".into());
    }
    let mut hypothesis_ids = std::collections::BTreeSet::new();
    let mut prior_sum = 0.0_f64;
    for hypothesis in &context.hypotheses {
        if hypothesis.id.is_empty() || !hypothesis_ids.insert(hypothesis.id.as_str()) {
            return Err("hypothesis IDs must be non-empty and unique".into());
        }
        if !hypothesis.prior_probability.is_finite()
            || hypothesis.prior_probability < 0.0
        {
            return Err(format!("hypothesis {} has invalid prior probability", hypothesis.id));
        }
        if hypothesis
            .action_losses
            .iter()
            .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(format!("hypothesis {} has invalid action loss", hypothesis.id));
        }
        prior_sum += hypothesis.prior_probability;
    }
    if !prior_sum.is_finite() || (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(format!("hypothesis prior probabilities sum to {prior_sum}, not 1"));
    }

    let mut query_ids = std::collections::BTreeSet::new();
    for query in &context.queries {
        if query.id.is_empty() || !query_ids.insert(query.id.as_str()) {
            return Err("query IDs must be non-empty and unique".into());
        }
        if query.observation_model_digest.len() != 64
            || !query
                .observation_model_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(format!("query {} has a noncanonical observation-model digest", query.id));
        }
        for (name, value) in [
            ("acquisition_cost", query.acquisition_cost),
            ("delay_cost", query.delay_cost),
            ("disclosure_exposure_cost", query.disclosure_exposure_cost),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("query {} has invalid {name}", query.id));
            }
        }
        if query.outcomes.is_empty() {
            return Err(format!("query {} has no possible outcomes", query.id));
        }
        let mut outcome_ids = std::collections::BTreeSet::new();
        let mut likelihood_sums = vec![0.0_f64; context.hypotheses.len()];
        for outcome in &query.outcomes {
            if outcome.id.is_empty() || !outcome_ids.insert(outcome.id.as_str()) {
                return Err(format!("query {} has empty or duplicate outcome IDs", query.id));
            }
            if outcome.likelihood_by_hypothesis.len() != context.hypotheses.len() {
                return Err(format!("query {} has an outcome with the wrong hypothesis count", query.id));
            }
            for (index, likelihood) in outcome.likelihood_by_hypothesis.iter().enumerate() {
                if !likelihood.is_finite() || !(0.0..=1.0).contains(likelihood) {
                    return Err(format!("query {} has an invalid likelihood", query.id));
                }
                likelihood_sums[index] += likelihood;
            }
        }
        for (index, sum) in likelihood_sums.iter().enumerate() {
            if (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
                return Err(format!(
                    "query {} observation probabilities for hypothesis {} sum to {sum}, not 1",
                    query.id, context.hypotheses[index].id
                ));
            }
        }
    }
    Ok(())
}

fn predictive_outcome_entropy_bits(
    context: &VoiContext,
    query: &EvidenceQuery,
) -> Result<f64, String> {
    let mut entropy = 0.0_f64;
    for outcome in &query.outcomes {
        let probability: f64 = context
            .hypotheses
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(hypothesis, likelihood)| hypothesis.prior_probability * likelihood)
            .sum();
        if probability > 0.0 {
            entropy -= probability * probability.log2();
        }
    }
    if !entropy.is_finite() || entropy < -PROBABILITY_TOLERANCE {
        return Err(format!("query {} has invalid predictive outcome entropy", query.id));
    }
    Ok(entropy.max(0.0))
}

fn stable_query_random_key(seed: u64, query_id: &str) -> u64 {
    let digest = sha2::Sha256::digest(format!("mace-voi-random:{seed}:{query_id}").as_bytes());
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is eight bytes"))
}

fn bayes_risk(hypotheses: &[Hypothesis], probabilities: &[f64]) -> f64 {
    let mut loss_a = 0.0_f64;
    let mut loss_b = 0.0_f64;
    for (hypothesis, probability) in hypotheses.iter().zip(probabilities) {
        loss_a += probability * hypothesis.action_losses[0];
        loss_b += probability * hypothesis.action_losses[1];
    }
    loss_a.min(loss_b)
}

fn expected_post_query_risk(
    hypotheses: &[Hypothesis],
    prior: &[f64],
    query: &EvidenceQuery,
) -> Result<f64, String> {
    let mut expected_risk = 0.0_f64;
    for outcome in &query.outcomes {
        let outcome_probability: f64 = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(probability, likelihood)| probability * likelihood)
            .sum();
        if outcome_probability <= 0.0 {
            continue;
        }
        let posterior: Vec<f64> = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(probability, likelihood)| {
                probability * likelihood / outcome_probability
            })
            .collect();
        let posterior_sum: f64 = posterior.iter().sum();
        if !posterior_sum.is_finite()
            || (posterior_sum - 1.0).abs() > PROBABILITY_TOLERANCE
        {
            return Err(format!(
                "query {} produced a non-normalized posterior for outcome {}",
                query.id, outcome.id
            ));
        }
        expected_risk += outcome_probability * bayes_risk(hypotheses, &posterior);
    }
    if !expected_risk.is_finite() {
        return Err(format!("query {} produced non-finite expected risk", query.id));
    }
    Ok(expected_risk)
}

/// Compute an EVSI plan without consulting realized future observations or
/// evaluator-only scenario outcomes. The observation model is candidate input.
pub(crate) fn plan_next_evidence(context: &VoiContext) -> Result<VoiPlan, String> {
    validate_context(context)?;
    let prior: Vec<f64> = context
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.prior_probability)
        .collect();
    let risk_before = bayes_risk(&context.hypotheses, &prior);
    if !risk_before.is_finite() {
        return Err("prior expected decision risk is non-finite".into());
    }
    let mut evaluations = Vec::with_capacity(context.queries.len());

    for query in &context.queries {
        let rejection_reason = if !query.source_authorized {
            Some("source_not_authorized")
        } else if !query.scope_valid {
            Some("scope_not_valid")
        } else if !query.privacy_approved {
            Some("privacy_not_approved")
        } else if !query.deadline_feasible {
            Some("deadline_not_feasible")
        } else {
            None
        };
        let total_cost = query.acquisition_cost
            + query.delay_cost
            + query.disclosure_exposure_cost;
        if !total_cost.is_finite() {
            return Err(format!("query {} has non-finite total cost", query.id));
        }

        if let Some(reason) = rejection_reason {
            evaluations.push(QueryEvaluation {
                query_id: query.id.clone(),
                observation_model_digest: query.observation_model_digest.clone(),
                admissible: false,
                rejection_reason: Some(reason.to_string()),
                risk_before,
                predictive_outcome_entropy_bits: None,
                expected_post_query_risk: None,
                expected_risk_reduction: None,
                acquisition_cost: query.acquisition_cost,
                delay_cost: query.delay_cost,
                disclosure_exposure_cost: query.disclosure_exposure_cost,
                total_cost,
                net_evsi: None,
                selected: false,
                selected_by_uncertainty_only: false,
                selected_by_random_control: false,
            });
            continue;
        }

        let predictive_entropy = predictive_outcome_entropy_bits(context, query)?;
        let expected_post = expected_post_query_risk(&context.hypotheses, &prior, query)?;
        // Information cannot worsen the Bayes-optimal decision before cost.
        // Clamp only tiny floating-point negatives; material violations reject.
        let raw_reduction = risk_before - expected_post;
        if raw_reduction < -PROBABILITY_TOLERANCE {
            return Err(format!(
                "query {} violates the Bayes information monotonicity invariant",
                query.id
            ));
        }
        let reduction = raw_reduction.max(0.0);
        let net_evsi = reduction - total_cost;
        if !net_evsi.is_finite() {
            return Err(format!("query {} has non-finite net EVSI", query.id));
        }
        evaluations.push(QueryEvaluation {
            query_id: query.id.clone(),
            observation_model_digest: query.observation_model_digest.clone(),
            admissible: true,
            rejection_reason: None,
            risk_before,
            predictive_outcome_entropy_bits: Some(predictive_entropy),
            expected_post_query_risk: Some(expected_post),
            expected_risk_reduction: Some(reduction),
            acquisition_cost: query.acquisition_cost,
            delay_cost: query.delay_cost,
            disclosure_exposure_cost: query.disclosure_exposure_cost,
            total_cost,
            net_evsi: Some(net_evsi),
            selected: false,
            selected_by_uncertainty_only: false,
            selected_by_random_control: false,
        });
    }

    let chosen_index = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| {
            evaluation.admissible && evaluation.net_evsi.is_some_and(|value| value > 0.0)
        })
        .max_by(|(_, left), (_, right)| {
            left.net_evsi
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&right.net_evsi.unwrap_or(f64::NEG_INFINITY))
                .then_with(|| right.total_cost.total_cmp(&left.total_cost))
                .then_with(|| right.query_id.cmp(&left.query_id))
        })
        .map(|(index, _)| index);

    let chosen_query_id = chosen_index.map(|index| {
        evaluations[index].selected = true;
        evaluations[index].query_id.clone()
    });

    // Baseline 1: maximize predictive observation entropy, ignoring decision
    // relevance and acquisition costs, but still obeying hard admissibility gates.
    let uncertainty_only_index = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.admissible)
        .max_by(|(_, left), (_, right)| {
            left.predictive_outcome_entropy_bits
                .unwrap_or(f64::NEG_INFINITY)
                .total_cmp(&right.predictive_outcome_entropy_bits.unwrap_or(f64::NEG_INFINITY))
                .then_with(|| right.query_id.cmp(&left.query_id))
        })
        .map(|(index, _)| index);
    let uncertainty_only_query_id = uncertainty_only_index.map(|index| {
        evaluations[index].selected_by_uncertainty_only = true;
        evaluations[index].query_id.clone()
    });

    // Baseline 2: deterministic random admissible request, keyed by a public seed.
    let random_control_index = evaluations
        .iter()
        .enumerate()
        .filter(|(_, evaluation)| evaluation.admissible)
        .min_by_key(|(_, evaluation)| {
            (
                stable_query_random_key(context.public_seed, &evaluation.query_id),
                evaluation.query_id.clone(),
            )
        })
        .map(|(index, _)| index);
    let random_control_query_id = random_control_index.map(|index| {
        evaluations[index].selected_by_random_control = true;
        evaluations[index].query_id.clone()
    });

    let stop_reason = if context.queries.is_empty() {
        "no_candidate_queries"
    } else if chosen_query_id.is_none() {
        "no_admissible_positive_net_evsi"
    } else {
        "positive_net_evsi_query_selected"
    };

    Ok(VoiPlan {
        risk_before,
        chosen_query_id,
        uncertainty_only_query_id,
        random_control_query_id,
        stop_reason,
        evaluations,
    })
}

/// A deliberately small policy fixture, not a measured agent benchmark.
/// It proves that the planner prefers decision-relevant evidence over a
/// high-entropy but decision-irrelevant query, and never overrides authority.
pub(crate) fn reference_fixture_context() -> VoiContext {
    VoiContext {
        public_seed: 0x4D41_4345,
        hypotheses: vec![
            Hypothesis {
                id: "state_a".into(),
                prior_probability: 0.5,
                action_losses: [0.0, 1.0],
            },
            Hypothesis {
                id: "state_b".into(),
                prior_probability: 0.5,
                action_losses: [1.0, 0.0],
            },
        ],
        queries: vec![
            EvidenceQuery {
                id: "independent_decision_relevant_measurement".into(),
                observation_model_digest: "a".repeat(64),
                outcomes: vec![
                    ObservationOutcome {
                        id: "measurement_supports_a".into(),
                        likelihood_by_hypothesis: vec![0.9, 0.1],
                    },
                    ObservationOutcome {
                        id: "measurement_supports_b".into(),
                        likelihood_by_hypothesis: vec![0.1, 0.9],
                    },
                ],
                acquisition_cost: 0.04,
                delay_cost: 0.02,
                disclosure_exposure_cost: 0.01,
                source_authorized: true,
                scope_valid: true,
                privacy_approved: true,
                deadline_feasible: true,
            },
            EvidenceQuery {
                id: "high_entropy_but_decision_irrelevant".into(),
                observation_model_digest: "b".repeat(64),
                outcomes: vec![
                    ObservationOutcome {
                        id: "irrelevant_signal_zero".into(),
                        likelihood_by_hypothesis: vec![0.25, 0.25],
                    },
                    ObservationOutcome {
                        id: "irrelevant_signal_one".into(),
                        likelihood_by_hypothesis: vec![0.25, 0.25],
                    },
                    ObservationOutcome {
                        id: "irrelevant_signal_two".into(),
                        likelihood_by_hypothesis: vec![0.25, 0.25],
                    },
                    ObservationOutcome {
                        id: "irrelevant_signal_three".into(),
                        likelihood_by_hypothesis: vec![0.25, 0.25],
                    },
                ],
                acquisition_cost: 0.01,
                delay_cost: 0.0,
                disclosure_exposure_cost: 0.0,
                source_authorized: true,
                scope_valid: true,
                privacy_approved: true,
                deadline_feasible: true,
            },
            EvidenceQuery {
                id: "unauthorized_perfect_measurement".into(),
                observation_model_digest: "c".repeat(64),
                outcomes: vec![
                    ObservationOutcome {
                        id: "perfect_supports_a".into(),
                        likelihood_by_hypothesis: vec![1.0, 0.0],
                    },
                    ObservationOutcome {
                        id: "perfect_supports_b".into(),
                        likelihood_by_hypothesis: vec![0.0, 1.0],
                    },
                ],
                acquisition_cost: 0.001,
                delay_cost: 0.0,
                disclosure_exposure_cost: 0.0,
                source_authorized: false,
                scope_valid: true,
                privacy_approved: true,
                deadline_feasible: true,
            },
        ],
    }
}

pub(crate) fn reference_fixture_plan() -> Result<VoiPlan, String> {
    plan_next_evidence(&reference_fixture_context())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_relevant_evidence_beats_high_entropy_irrelevant_evidence() {
        let plan = reference_fixture_plan().unwrap();
        assert_eq!(
            plan.chosen_query_id.as_deref(),
            Some("independent_decision_relevant_measurement")
        );
        assert_eq!(
            plan.uncertainty_only_query_id.as_deref(),
            Some("high_entropy_but_decision_irrelevant")
        );
        assert!(plan.random_control_query_id.is_some());
        let relevant = plan
            .evaluations
            .iter()
            .find(|item| item.query_id == "independent_decision_relevant_measurement")
            .unwrap();
        let irrelevant = plan
            .evaluations
            .iter()
            .find(|item| item.query_id == "high_entropy_but_decision_irrelevant")
            .unwrap();
        assert_eq!(irrelevant.expected_risk_reduction, Some(0.0));
        assert!(irrelevant.net_evsi.unwrap() < 0.0);
        assert!(irrelevant.predictive_outcome_entropy_bits.unwrap()
            > relevant.predictive_outcome_entropy_bits.unwrap());
        assert!(irrelevant.selected_by_uncertainty_only);
        assert!(!irrelevant.selected);
    }

    #[test]
    fn hard_authority_constraint_cannot_be_bought_with_high_voi() {
        let plan = reference_fixture_plan().unwrap();
        let unauthorized = plan
            .evaluations
            .iter()
            .find(|item| item.query_id == "unauthorized_perfect_measurement")
            .unwrap();
        assert!(!unauthorized.admissible);
        assert_eq!(unauthorized.rejection_reason.as_deref(), Some("source_not_authorized"));
        assert!(!unauthorized.selected);
        assert!(!unauthorized.selected_by_uncertainty_only);
        assert!(!unauthorized.selected_by_random_control);
        assert!(unauthorized.net_evsi.is_none());
        assert!(plan.uncertainty_only_query_id.as_deref()
            != Some("unauthorized_perfect_measurement"));
        assert!(plan.random_control_query_id.as_deref()
            != Some("unauthorized_perfect_measurement"));
    }

    #[test]
    fn malformed_observation_model_fails_closed() {
        let mut context = VoiContext {
            public_seed: 1,
            hypotheses: vec![Hypothesis {
                id: "h1".into(),
                prior_probability: 1.0,
                action_losses: [0.0, 1.0],
            }],
            queries: vec![EvidenceQuery {
                id: "bad".into(),
                observation_model_digest: "d".repeat(64),
                outcomes: vec![ObservationOutcome {
                    id: "only_outcome".into(),
                    likelihood_by_hypothesis: vec![0.7],
                }],
                acquisition_cost: 0.0,
                delay_cost: 0.0,
                disclosure_exposure_cost: 0.0,
                source_authorized: true,
                scope_valid: true,
                privacy_approved: true,
                deadline_feasible: true,
            }],
        };
        assert!(plan_next_evidence(&context).is_err());
        context.queries[0].outcomes[0].likelihood_by_hypothesis[0] = 1.0;
        assert!(plan_next_evidence(&context).is_ok());
    }

    #[test]
    fn query_and_outcome_order_do_not_change_the_selected_request() {
        let context = reference_fixture_context();
        let plan = plan_next_evidence(&context).unwrap();

        let mut reversed_context = reference_fixture_context();
        reversed_context.queries.reverse();
        for query in &mut reversed_context.queries {
            query.outcomes.reverse();
        }
        let reversed = plan_next_evidence(&reversed_context).unwrap();
        assert_eq!(plan.chosen_query_id, reversed.chosen_query_id);
        assert_eq!(plan.uncertainty_only_query_id, reversed.uncertainty_only_query_id);
        assert_eq!(plan.random_control_query_id, reversed.random_control_query_id);
        assert!((plan.risk_before - reversed.risk_before).abs() < PROBABILITY_TOLERANCE);
    }
}

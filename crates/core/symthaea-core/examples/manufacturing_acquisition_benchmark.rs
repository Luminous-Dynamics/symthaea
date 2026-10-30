//! Matched-budget demonstration of manufacturing measurement-acquisition policies.
//!
//! This example consumes the Symtropy adapter fixture. The independent held-out evaluator
//! fixture is loaded separately and is never passed into any ranking function. Both fixtures
//! are synthetic and illustrate policy differences; they are not factory evidence or SOTA data.
use std::collections::BTreeMap;

use symthaea_core::hdc::conjecture_engine::{
    expected_information_gain_nats,
    select_experiment_by_information_gain_per_cost,
    select_experiment_by_information_gain_per_cost_with_noise_and_eligibility,
    select_experiment_by_representative_deamplifying_worst_case_eig_per_cost,
    select_experiment_by_worst_case_information_gain_per_cost_with_prior_weights,
};

const ADAPTER: &str =
    include_str!("../src/hdc/conjecture_engine/fixtures/manufacturing-robustness-adapter-synthetic.csv");
const EVALUATOR: &str =
    include_str!("fixtures/manufacturing-acquisition-evaluator-synthetic.csv");
const RANDOM_TRIALS: usize = 1_000;

#[derive(Clone, Debug)]
struct Candidate {
    id: String,
    representativeness: f64,
    amplification_risk: f64,
    observation_std_dev: f64,
    cost: f64,
    cost_unit: String,
    eligible: bool,
}

#[derive(Clone, Copy)]
struct Scenario {
    ood_signal_multiplier: f64,
}

#[derive(Clone, Copy)]
struct HeldOutOutcome {
    decision_success: f64,
    high_confidence_wrong: f64,
    holdout_lots: u32,
}

#[derive(Clone, Copy)]
struct Metrics {
    success: f64,
    false_confidence: f64,
    cost: f64,
    utility_regret: f64,
}

fn parse_candidates() -> Result<Vec<Candidate>, String> {
    let mut lines = ADAPTER.lines();
    let expected = "candidate_id,representativeness_score,amplification_risk,observation_std_uncertainty,acquisition_cost,cost_unit,eligible,eligibility_state,eligibility_receipt_sha256";
    if lines.next() != Some(expected) {
        return Err("unexpected Symtropy candidate-adapter header".to_string());
    }
    let mut candidates = Vec::new();
    for (index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 9 {
            return Err(format!("candidate row {} must have nine fields", index + 2));
        }
        if fields[7] != "ELIGIBLE" || fields[8].len() != 64 {
            return Err(format!("candidate row {} lacks eligibility receipt/state", index + 2));
        }
        let candidate = Candidate {
            id: fields[0].to_string(),
            representativeness: fields[1].parse().map_err(|_| "invalid representativeness")?,
            amplification_risk: fields[2].parse().map_err(|_| "invalid amplification risk")?,
            observation_std_dev: fields[3].parse().map_err(|_| "invalid observation uncertainty")?,
            cost: fields[4].parse().map_err(|_| "invalid cost")?,
            cost_unit: fields[5].to_string(),
            eligible: fields[6].parse().map_err(|_| "invalid eligibility bool")?,
        };
        if !candidate.eligible
            || !(0.0..=1.0).contains(&candidate.representativeness)
            || !(0.0..=1.0).contains(&candidate.amplification_risk)
            || !candidate.observation_std_dev.is_finite()
            || candidate.observation_std_dev <= 0.0
            || !candidate.cost.is_finite()
            || candidate.cost <= 0.0
        {
            return Err(format!("candidate {:?} is not admissible benchmark input", candidate.id));
        }
        candidates.push(candidate);
    }
    if candidates.len() < 2 {
        return Err("benchmark needs at least two candidates".to_string());
    }
    let unit = &candidates[0].cost_unit;
    if candidates.iter().any(|candidate| &candidate.cost_unit != unit) {
        return Err("benchmark candidates must use a common cost unit".to_string());
    }
    Ok(candidates)
}

fn parse_independent_evaluator() -> Result<BTreeMap<String, HeldOutOutcome>, String> {
    let mut lines = EVALUATOR.lines();
    if lines.next()
        != Some("candidate_id,heldout_decision_success_rate,high_confidence_wrong_rate,holdout_lot_count")
    {
        return Err("unexpected independent evaluator header".to_string());
    }
    let mut outcomes = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != 4 {
            return Err(format!("evaluator row {} must have four fields", index + 2));
        }
        let outcome = HeldOutOutcome {
            decision_success: fields[1].parse().map_err(|_| "invalid held-out success rate")?,
            high_confidence_wrong: fields[2].parse().map_err(|_| "invalid false-confidence rate")?,
            holdout_lots: fields[3].parse().map_err(|_| "invalid holdout lot count")?,
        };
        if !(0.0..=1.0).contains(&outcome.decision_success)
            || !(0.0..=1.0).contains(&outcome.high_confidence_wrong)
            || outcome.holdout_lots == 0
        {
            return Err(format!("evaluator row {} has out-of-range values", index + 2));
        }
        outcomes.insert(fields[0].to_string(), outcome);
    }
    if outcomes.is_empty() {
        return Err("independent evaluator is empty".to_string());
    }
    Ok(outcomes)
}

fn prediction(
    hypothesis: &usize,
    candidate: &Candidate,
    scenario: &Scenario,
) -> Option<f64> {
    let scale = if candidate.id == "TX-CAND-OOD-002" {
        scenario.ood_signal_multiplier
    } else {
        0.85 + 0.15 * scenario.ood_signal_multiplier
    };
    let nominal_signal = if candidate.id == "TX-CAND-OOD-002" {
        20.0
    } else {
        2.0
    };
    Some(*hypothesis as f64 * nominal_signal * scale)
}

fn score_base_eig(candidates: &[Candidate], hypotheses: &[usize]) -> Option<(&Candidate, f64)> {
    let base = Scenario {
        ood_signal_multiplier: 1.0,
    };
    let mut best = None;
    for candidate in candidates.iter().filter(|candidate| candidate.eligible) {
        let information = expected_information_gain_nats(
            candidate,
            hypotheses,
            0.005,
            |hypothesis, item| prediction(hypothesis, item, &base),
        )?;
        let score = information / candidate.cost;
        if score.is_finite() && score > 0.0
            && best.is_none_or(|(_, best_score): (&&Candidate, f64)| score > best_score)
        {
            best = Some((candidate, score));
        }
    }
    best
}

fn seeded_random_index(seed: &mut u64, upper: usize) -> usize {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    (*seed as usize) % upper
}

fn evaluate(
    selected: &[&Candidate],
    outcomes: &BTreeMap<String, HeldOutOutcome>,
    oracle_utility: f64,
) -> Result<Metrics, String> {
    if selected.is_empty() {
        return Err("policy selected no candidates".to_string());
    }
    let mut success = 0.0;
    let mut false_confidence = 0.0;
    let mut cost = 0.0;
    let mut lots = 0_u32;
    for candidate in selected {
        let outcome = outcomes
            .get(&candidate.id)
            .ok_or_else(|| format!("no independent evaluator row for {:?}", candidate.id))?;
        success += outcome.decision_success;
        false_confidence += outcome.high_confidence_wrong;
        cost += candidate.cost;
        lots = lots.saturating_add(outcome.holdout_lots);
    }
    let count = selected.len() as f64;
    let _independent_lot_count = lots;
    let utility = (success - false_confidence) / count;
    Ok(Metrics {
        success: success / count,
        false_confidence: false_confidence / count,
        cost: cost / count,
        utility_regret: (oracle_utility - utility).max(0.0),
    })
}

fn print_row(name: &str, candidate: &str, metrics: Metrics, trials: usize) {
    println!(
        "{name},{candidate},{:.4},{:.4},{:.2},{:.4},{trials}",
        metrics.success,
        metrics.false_confidence,
        metrics.cost,
        metrics.utility_regret
    );
}

fn main() -> Result<(), String> {
    let candidates = parse_candidates()?;
    let outcomes = parse_independent_evaluator()?;
    let candidate_ids: Vec<_> = candidates.iter().map(|candidate| candidate.id.as_str()).collect();
    let unit = candidates[0].cost_unit.as_str();
    let best_oracle = outcomes
        .values()
        .map(|outcome| outcome.decision_success - outcome.high_confidence_wrong)
        .fold(f64::NEG_INFINITY, f64::max);
    if !best_oracle.is_finite() {
        return Err("independent evaluator did not define an oracle".to_string());
    }

    let hypotheses = [0_usize, 1];
    let scenarios = [
        Scenario {
            ood_signal_multiplier: 1.0,
        },
        Scenario {
            ood_signal_multiplier: 0.0,
        },
    ];
    let predict = |hypothesis: &usize, candidate: &Candidate, scenario: &Scenario| {
        prediction(hypothesis, candidate, scenario)
    };

    println!(
        "policy,selected_candidate,heldout_decision_success,high_confidence_wrong,mean_acquisition_cost_{unit},utility_regret,trials"
    );

    let mut random_seed = 0x4D46_472D_4143_5101_u64;
    let random_selections: Vec<&Candidate> = (0..RANDOM_TRIALS)
        .map(|_| &candidates[seeded_random_index(&mut random_seed, candidates.len())])
        .collect();
    let random_metrics = evaluate(&random_selections, &outcomes, best_oracle)?;
    print_row("RANDOM", "SEE_SEEDED_MIX", random_metrics, RANDOM_TRIALS);

    let (common, _) = select_experiment_by_information_gain_per_cost(
        &candidates,
        &hypotheses,
        0.005,
        predict,
        |candidate| candidate.cost,
    )
    .ok_or_else(|| "common-noise EIG abstained".to_string())?;
    print_row(
        "COMMON_NOISE_EIG",
        &common.id,
        evaluate(&[common], &outcomes, best_oracle)?,
        1,
    );

    let (heteroscedastic, _) =
        select_experiment_by_information_gain_per_cost_with_noise_and_eligibility(
            &candidates,
            &hypotheses,
            |candidate| candidate.eligible,
            |candidate| candidate.observation_std_dev,
            predict,
            |candidate| candidate.cost,
        )
        .ok_or_else(|| "heteroscedastic EIG abstained".to_string())?;
    print_row(
        "HETEROSCEDASTIC_EIG",
        &heteroscedastic.id,
        evaluate(&[heteroscedastic], &outcomes, best_oracle)?,
        1,
    );

    let (worst_case, _) =
        select_experiment_by_worst_case_information_gain_per_cost_with_prior_weights(
            &candidates,
            &hypotheses,
            &[0.5, 0.5],
            &scenarios,
            |candidate| candidate.eligible,
            |candidate| candidate.observation_std_dev,
            predict,
            |candidate| candidate.cost,
        )
        .ok_or_else(|| "finite-scenario worst-case EIG abstained".to_string())?;
    print_row(
        "WORST_CASE_EIG",
        &worst_case.id,
        evaluate(&[worst_case], &outcomes, best_oracle)?,
        1,
    );

    let (robust, _) =
        select_experiment_by_representative_deamplifying_worst_case_eig_per_cost(
            &candidates,
            &hypotheses,
            &[0.5, 0.5],
            &scenarios,
            |candidate| candidate.eligible,
            |candidate| candidate.representativeness,
            |candidate| candidate.amplification_risk,
            |candidate| candidate.observation_std_dev,
            predict,
            |candidate| candidate.cost,
        )
        .ok_or_else(|| "representative/de-amplifying EIG abstained".to_string())?;
    print_row(
        "REPRESENTATIVE_DEAMPLIFYING_EIG",
        &robust.id,
        evaluate(&[robust], &outcomes, best_oracle)?,
        1,
    );

    eprintln!(
        "synthetic policy comparison complete: candidates={:?}, common_cost_unit={unit}; independent evaluator was not passed to any selector; no physical performance claim",
        candidate_ids
    );
    Ok(())
}

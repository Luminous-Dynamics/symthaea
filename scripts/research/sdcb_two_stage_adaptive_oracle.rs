//! Exact two-stage adaptive experiment-selection oracle for paired worlds.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_two_stage_adaptive_oracle.rs -o /tmp/sdcb-adaptive
//! Run:
//!   /tmp/sdcb-adaptive
//!
//! A second experiment may depend on the first categorical outcome. Experiments
//! are conditionally independent given the paired world. A repeated ID denotes
//! a fresh replicate with the same declared conditional likelihood.
//! The planner enumerates all first actions and all outcome-to-second-action
//! maps, enforcing the hard budget on every possible path.

const MAX_EXPERIMENTS: usize = 8;
const MAX_OUTCOMES: usize = 3;
const PROB_TOL: f64 = 1e-10;
const UTILITY_TOL: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub struct Experiment {
    pub id: String,
    pub cost_units: u64,
    /// P(outcome | world A), in categorical outcome order.
    pub likelihood_world_a: Vec<f64>,
    /// P(outcome | world B), in categorical outcome order.
    pub likelihood_world_b: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveDesign {
    pub world_a_id: String,
    pub world_b_id: String,
    pub prior_world_a: f64,
    /// Utility of abstaining after any observed history (correct decision=1, wrong=0).
    pub abstain_utility: f64,
    pub budget_units: u64,
    pub experiments: Vec<Experiment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidWorldIdentity,
    IdenticalWorlds,
    InvalidPrior,
    InvalidAbstainUtility,
    EmptyExperimentSet,
    TooManyExperiments,
    InvalidExperimentIdentity,
    DuplicateExperimentIdentity,
    InvalidExperimentCost,
    InvalidLikelihood,
    OutcomeDimensionMismatch,
    TooManyOutcomes,
    EnumerationOverflow,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveReport {
    pub first_experiment_id: Option<String>,
    /// Index zero corresponds to first-stage outcome zero. None means stop after stage one.
    pub second_experiments_by_first_outcome: Vec<Option<String>>,
    pub expected_utility: f64,
    pub baseline_utility: f64,
    pub decision_value_gain: f64,
    pub expected_cost_units: f64,
    pub maximum_path_cost_units: u64,
    pub budget_units: u64,
    pub policies_evaluated: u64,
}

fn validate_distribution(values: &[f64]) -> Result<(), Error> {
    if values.len() > MAX_OUTCOMES {
        return Err(Error::TooManyOutcomes);
    }
    if values.len() < 2
        || values.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (values.iter().sum::<f64>() - 1.0).abs() > PROB_TOL
    {
        return Err(Error::InvalidLikelihood);
    }
    Ok(())
}

fn validate_design(design: &AdaptiveDesign) -> Result<usize, Error> {
    if design.world_a_id.trim().is_empty() || design.world_b_id.trim().is_empty() {
        return Err(Error::InvalidWorldIdentity);
    }
    if design.world_a_id == design.world_b_id {
        return Err(Error::IdenticalWorlds);
    }
    if !design.prior_world_a.is_finite()
        || !(0.0..=1.0).contains(&design.prior_world_a)
    {
        return Err(Error::InvalidPrior);
    }
    if !design.abstain_utility.is_finite()
        || !(0.0..=1.0).contains(&design.abstain_utility)
    {
        return Err(Error::InvalidAbstainUtility);
    }
    let n = design.experiments.len();
    if n == 0 {
        return Err(Error::EmptyExperimentSet);
    }
    if n > MAX_EXPERIMENTS {
        return Err(Error::TooManyExperiments);
    }
    let mut outcome_count = None;
    for (i, experiment) in design.experiments.iter().enumerate() {
        if experiment.id.trim().is_empty() {
            return Err(Error::InvalidExperimentIdentity);
        }
        if design.experiments[..i].iter().any(|other| other.id == experiment.id) {
            return Err(Error::DuplicateExperimentIdentity);
        }
        if experiment.cost_units == 0 {
            return Err(Error::InvalidExperimentCost);
        }
        validate_distribution(&experiment.likelihood_world_a)?;
        validate_distribution(&experiment.likelihood_world_b)?;
        if experiment.likelihood_world_a.len() != experiment.likelihood_world_b.len() {
            return Err(Error::OutcomeDimensionMismatch);
        }
        let k = experiment.likelihood_world_a.len();
        match outcome_count {
            None => outcome_count = Some(k),
            Some(existing) if existing != k => return Err(Error::OutcomeDimensionMismatch),
            Some(_) => {}
        }
    }
    let k = outcome_count.ok_or(Error::EmptyExperimentSet)?;
    if k > MAX_OUTCOMES {
        return Err(Error::TooManyOutcomes);
    }
    Ok(k)
}

fn terminal_utility(mass_a: f64, mass_b: f64, abstain: f64) -> f64 {
    mass_a.max(mass_b).max(abstain * (mass_a + mass_b))
}

fn baseline_utility(design: &AdaptiveDesign) -> f64 {
    design.prior_world_a
        .max(1.0 - design.prior_world_a)
        .max(design.abstain_utility)
}

fn evaluate_policy(
    design: &AdaptiveDesign,
    first_index: usize,
    second_map: &[Option<usize>],
) -> Result<(f64, f64, u64), Error> {
    let first = &design.experiments[first_index];
    let outcomes = first.likelihood_world_a.len();
    if second_map.len() != outcomes {
        return Err(Error::OutcomeDimensionMismatch);
    }
    let mut expected_utility = 0.0;
    let mut expected_cost = first.cost_units as f64;
    let mut maximum_path_cost = first.cost_units;

    for y1 in 0..outcomes {
        let p_a1 = design.prior_world_a * first.likelihood_world_a[y1];
        let p_b1 = (1.0 - design.prior_world_a) * first.likelihood_world_b[y1];
        let probability_y1 = p_a1 + p_b1;
        match second_map[y1] {
            None => {
                expected_utility += terminal_utility(
                    p_a1,
                    p_b1,
                    design.abstain_utility,
                );
            }
            Some(second_index) => {
                let second = design.experiments.get(second_index).ok_or(Error::EnumerationOverflow)?;
                let path_cost = first.cost_units
                    .checked_add(second.cost_units)
                    .ok_or(Error::EnumerationOverflow)?;
                if path_cost > design.budget_units {
                    return Err(Error::EnumerationOverflow);
                }
                expected_cost += probability_y1 * second.cost_units as f64;
                maximum_path_cost = maximum_path_cost.max(path_cost);
                for y2 in 0..outcomes {
                    let p_a2 = p_a1 * second.likelihood_world_a[y2];
                    let p_b2 = p_b1 * second.likelihood_world_b[y2];
                    expected_utility += terminal_utility(
                        p_a2,
                        p_b2,
                        design.abstain_utility,
                    );
                }
            }
        }
    }
    if !expected_utility.is_finite() || !expected_cost.is_finite() {
        return Err(Error::EnumerationOverflow);
    }
    Ok((expected_utility, expected_cost, maximum_path_cost))
}

fn policy_key(
    design: &AdaptiveDesign,
    first_index: usize,
    second_map: &[Option<usize>],
) -> String {
    let mut key = design.experiments[first_index].id.clone();
    key.push('|');
    for second in second_map {
        match second {
            Some(index) => key.push_str(&design.experiments[*index].id),
            None => key.push_str("~stop"),
        }
        key.push('|');
    }
    key
}

/// Exact Bayes-optimal two-stage policy over the supplied paired worlds.
/// Each branch may stop or choose any listed second-stage experiment.
/// The cost constraint is pathwise: every possible first-outcome branch
/// must fit within budget. The posterior terminal action may choose A, B,
/// or abstain after the full observed history.
pub fn optimize(design: &AdaptiveDesign) -> Result<AdaptiveReport, Error> {
    let outcome_count = validate_design(design)?;
    let baseline = baseline_utility(design);
    let mut best_utility = baseline;
    let mut best_expected_cost = 0.0;
    let mut best_maximum_path_cost = 0_u64;
    let mut best_first: Option<usize> = None;
    let mut best_map = Vec::<Option<usize>>::new();
    let mut best_key = String::new();
    let mut policies_evaluated = 0_u64;

    let radix = (design.experiments.len() + 1) as u64;
    let mut mapping_count = 1_u64;
    for _ in 0..outcome_count {
        mapping_count = mapping_count.checked_mul(radix).ok_or(Error::EnumerationOverflow)?;
    }

    for first_index in 0..design.experiments.len() {
        let first = &design.experiments[first_index];
        if first.cost_units > design.budget_units {
            continue;
        }
        for mut encoded_map in 0..mapping_count {
            let mut second_map = Vec::with_capacity(outcome_count);
            let mut affordable = true;
            for _ in 0..outcome_count {
                let digit = (encoded_map % radix) as usize;
                encoded_map /= radix;
                let second = if digit == 0 { None } else { Some(digit - 1) };
                if let Some(second_index) = second {
                    if first.cost_units
                        .checked_add(design.experiments[second_index].cost_units)
                        .is_none_or(|cost| cost > design.budget_units)
                    {
                        affordable = false;
                        break;
                    }
                }
                second_map.push(second);
            }
            if !affordable {
                continue;
            }
            policies_evaluated = policies_evaluated
                .checked_add(1)
                .ok_or(Error::EnumerationOverflow)?;
            let (utility, expected_cost, maximum_path_cost) =
                evaluate_policy(design, first_index, &second_map)?;
            let key = policy_key(design, first_index, &second_map);
            let improves = utility > best_utility + UTILITY_TOL;
            let ties = (utility - best_utility).abs() <= UTILITY_TOL;
            let deterministic_tie = ties
                && best_first.is_some()
                && (expected_cost < best_expected_cost - UTILITY_TOL
                    || ((expected_cost - best_expected_cost).abs() <= UTILITY_TOL
                        && (best_key.is_empty() || key < best_key)));
            if improves || deterministic_tie {
                best_utility = utility;
                best_expected_cost = expected_cost;
                best_maximum_path_cost = maximum_path_cost;
                best_first = Some(first_index);
                best_map = second_map;
                best_key = key;
            }
        }
    }

    let second_experiments_by_first_outcome = best_map
        .iter()
        .map(|choice| choice.map(|index| design.experiments[index].id.clone()))
        .collect();

    Ok(AdaptiveReport {
        first_experiment_id: best_first.map(|index| design.experiments[index].id.clone()),
        second_experiments_by_first_outcome,
        expected_utility: best_utility,
        baseline_utility: baseline,
        decision_value_gain: best_utility - baseline,
        expected_cost_units: best_expected_cost,
        maximum_path_cost_units: best_maximum_path_cost,
        budget_units: design.budget_units,
        policies_evaluated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adaptive_fixture(budget_units: u64) -> AdaptiveDesign {
        AdaptiveDesign {
            world_a_id: "scientific-world-a".into(),
            world_b_id: "scientific-world-b".into(),
            prior_world_a: 0.5,
            abstain_utility: 0.6,
            budget_units,
            experiments: vec![
                Experiment {
                    id: "uninformative-control".into(),
                    cost_units: 1,
                    likelihood_world_a: vec![0.5, 0.5],
                    likelihood_world_b: vec![0.5, 0.5],
                },
                Experiment {
                    id: "triage-test".into(),
                    cost_units: 1,
                    likelihood_world_a: vec![0.7, 0.3],
                    likelihood_world_b: vec![0.1, 0.9],
                },
                Experiment {
                    id: "confirm-a".into(),
                    cost_units: 1,
                    likelihood_world_a: vec![0.75, 0.25],
                    likelihood_world_b: vec![0.45, 0.55],
                },
                Experiment {
                    id: "confirm-b".into(),
                    cost_units: 1,
                    likelihood_world_a: vec![0.1, 0.9],
                    likelihood_world_b: vec![0.6, 0.4],
                },
            ],
        }
    }

    #[test]
    fn exact_policy_adapts_second_measurement_to_first_outcome() {
        let report = optimize(&adaptive_fixture(2)).unwrap();
        assert_eq!(report.first_experiment_id.as_deref(), Some("triage-test"));
        assert_eq!(
            report.second_experiments_by_first_outcome,
            vec![Some("confirm-b".to_string()), Some("triage-test".to_string())]
        );
        assert!((report.expected_utility - 0.864).abs() < 1e-12);
        assert!((report.baseline_utility - 0.6).abs() < 1e-12);
        assert!((report.decision_value_gain - 0.264).abs() < 1e-12);
        assert_eq!(report.maximum_path_cost_units, 2);
    }

    #[test]
    fn adaptive_policy_beats_best_fixed_second_experiment() {
        let design = adaptive_fixture(2);
        let report = optimize(&design).unwrap();
        let mut best_fixed = 0.0_f64;
        for first_index in 0..design.experiments.len() {
            for second_index in 0..design.experiments.len() {
                let fixed_map = vec![Some(second_index), Some(second_index)];
                let (utility, _, _) = evaluate_policy(&design, first_index, &fixed_map).unwrap();
                best_fixed = best_fixed.max(utility);
            }
        }
        assert!((best_fixed - 0.86).abs() < 1e-12);
        assert!(report.expected_utility > best_fixed);
    }

    #[test]
    fn every_contingent_branch_obeys_hard_budget() {
        let report = optimize(&adaptive_fixture(1)).unwrap();
        assert!(report.maximum_path_cost_units <= 1);
        assert!(report.expected_cost_units <= 1.0);
    }

    #[test]
    fn zero_budget_returns_no_experiment_baseline() {
        let report = optimize(&adaptive_fixture(0)).unwrap();
        assert_eq!(report.first_experiment_id, None);
        assert_eq!(report.expected_utility, report.baseline_utility);
        assert_eq!(report.expected_cost_units, 0.0);
        assert_eq!(report.maximum_path_cost_units, 0);
    }

    #[test]
    fn malformed_likelihoods_reject() {
        let mut design = adaptive_fixture(2);
        design.experiments[1].likelihood_world_a = vec![0.9, 0.2];
        assert_eq!(optimize(&design), Err(Error::InvalidLikelihood));
    }

    #[test]
    fn mismatched_outcome_dimensions_reject() {
        let mut design = adaptive_fixture(2);
        design.experiments[1].likelihood_world_b = vec![0.5, 0.25, 0.25];
        assert_eq!(optimize(&design), Err(Error::OutcomeDimensionMismatch));
    }

    #[test]
    fn duplicate_experiment_ids_reject() {
        let mut design = adaptive_fixture(2);
        design.experiments[1].id = design.experiments[0].id.clone();
        assert_eq!(optimize(&design), Err(Error::DuplicateExperimentIdentity));
    }

    #[test]
    fn excess_experiment_count_rejects_before_enumeration() {
        let mut design = adaptive_fixture(100);
        while design.experiments.len() <= MAX_EXPERIMENTS {
            design.experiments.push(Experiment {
                id: format!("extra-{}", design.experiments.len()),
                cost_units: 1,
                likelihood_world_a: vec![0.6, 0.4],
                likelihood_world_b: vec![0.4, 0.6],
            });
        }
        assert_eq!(optimize(&design), Err(Error::TooManyExperiments));
    }
}

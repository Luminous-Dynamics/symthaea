use serde_json::Value;
use sha2::{Digest, Sha256};

const FIXTURE: &str =
    include_str!("../../cases/entropy-vs-decision-value.v0.1.json");

#[derive(Debug, Clone)]
pub struct ScenarioReport {
    pub expected_risk: f64,
    pub evsi: f64,
    pub information_gain_bits: f64,
    pub outcome_probabilities: Vec<f64>,
    pub posterior_impact_probabilities: Vec<f64>,
    pub conditional_bayes_risks: Vec<f64>,
    pub bayes_actions: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub baseline_risk: f64,
    pub nuisance_entropy_bits: f64,
    pub perfect_information_evsi: f64,
    pub single_reading: ScenarioReport,
    pub same_source_pair: ScenarioReport,
    pub independent_source_pair: ScenarioReport,
    pub sha256: String,
}

fn number<'a>(root: &'a Value, path: &[&str]) -> Result<f64, String> {
    let mut node = root;
    for part in path {
        node = if let Ok(index) = part.parse::<usize>() {
            node.get(index)
        } else {
            node.get(*part)
        }
        .ok_or_else(|| format!("missing fixture field: {}", path.join(".")))?;
    }
    node.as_f64()
        .ok_or_else(|| format!("fixture field is not numeric: {}", path.join(".")))
}

fn string<'a>(root: &'a Value, path: &[&str]) -> Result<&'a str, String> {
    let mut node = root;
    for part in path {
        node = node
            .get(*part)
            .ok_or_else(|| format!("missing fixture field: {}", path.join(".")))?;
    }
    node.as_str()
        .ok_or_else(|| format!("fixture field is not a string: {}", path.join(".")))
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
}

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) -> Result<(), String> {
    if !actual.is_finite() || !expected.is_finite() || (actual - expected).abs() > tolerance {
        return Err(format!(
            "{label}: actual={actual:.17}, expected={expected:.17}, tolerance={tolerance}"
        ));
    }
    Ok(())
}

fn conditional_decision(
    posterior_impact: f64,
    protect_impact: f64,
    protect_safe: f64,
    no_protect_impact: f64,
    no_protect_safe: f64,
) -> (f64, &'static str) {
    let protect = posterior_impact * protect_impact
        + (1.0 - posterior_impact) * protect_safe;
    let no_protect = posterior_impact * no_protect_impact
        + (1.0 - posterior_impact) * no_protect_safe;
    if protect <= no_protect {
        (protect, "protect")
    } else {
        (no_protect, "no_protect")
    }
}

fn conditional_risk(
    posterior_impact: f64,
    protect_impact: f64,
    protect_safe: f64,
    no_protect_impact: f64,
    no_protect_safe: f64,
) -> f64 {
    conditional_decision(
        posterior_impact,
        protect_impact,
        protect_safe,
        no_protect_impact,
        no_protect_safe,
    )
    .0
}

fn scenario(
    prior_impact: f64,
    conditional_impact: &[f64],
    conditional_safe: &[f64],
    losses: [f64; 4],
) -> Result<ScenarioReport, String> {
    if conditional_impact.len() != conditional_safe.len() || conditional_impact.is_empty() {
        return Err("conditional outcome vectors have incompatible lengths".to_owned());
    }
    let mut joint = Vec::with_capacity(conditional_impact.len());
    let mut posterior = Vec::with_capacity(conditional_impact.len());
    let mut expected_risk = 0.0;
    let mut conditional_risks = Vec::with_capacity(conditional_impact.len());
    let mut bayes_actions = Vec::with_capacity(conditional_impact.len());
    for (&impact_likelihood, &safe_likelihood) in
        conditional_impact.iter().zip(conditional_safe.iter())
    {
        let p_outcome = prior_impact * impact_likelihood
            + (1.0 - prior_impact) * safe_likelihood;
        if p_outcome <= 0.0 {
            return Err("zero-probability outcome cannot be scored".to_owned());
        }
        let p_impact_given_outcome = prior_impact * impact_likelihood / p_outcome;
        let (risk, action) = conditional_decision(
            p_impact_given_outcome,
            losses[0],
            losses[1],
            losses[2],
            losses[3],
        );
        joint.push(p_outcome);
        posterior.push(p_impact_given_outcome);
        conditional_risks.push(risk);
        bayes_actions.push(action.to_owned());
        expected_risk += p_outcome * risk;
    }
    let probability_sum: f64 = joint.iter().sum();
    close(probability_sum, 1.0, 1e-12, "outcome probability sum")?;
    let information_gain = entropy(&joint)
        - prior_impact * entropy(conditional_impact)
        - (1.0 - prior_impact) * entropy(conditional_safe);
    let baseline = conditional_risk(
        prior_impact,
        losses[0],
        losses[1],
        losses[2],
        losses[3],
    );
    Ok(ScenarioReport {
        expected_risk,
        evsi: baseline - expected_risk,
        information_gain_bits: information_gain.max(0.0),
        outcome_probabilities: joint,
        posterior_impact_probabilities: posterior,
        conditional_bayes_risks: conditional_risks,
        bayes_actions,
    })
}

fn shared_source_pair(
    prior_impact: f64,
    bad_calibration_probability: f64,
    per_reading_error_when_good: f64,
    losses: [f64; 4],
) -> Result<ScenarioReport, String> {
    let patterns = ["--", "-+", "+-", "++"];
    let good_error = per_reading_error_when_good;
    let bad_error = 1.0 - good_error;
    let conditional_for_state = |impact_state: bool| -> Vec<f64> {
        patterns
            .iter()
            .map(|pattern| {
                let likelihood_for_error = |error: f64| {
                    pattern
                        .chars()
                        .map(|symbol| {
                            let expected = if impact_state { '+' } else { '-' };
                            if symbol == expected {
                                1.0 - error
                            } else {
                                error
                            }
                        })
                        .product::<f64>()
                };
                (1.0 - bad_calibration_probability) * likelihood_for_error(good_error)
                    + bad_calibration_probability * likelihood_for_error(bad_error)
            })
            .collect()
    };
    scenario(
        prior_impact,
        &conditional_for_state(true),
        &conditional_for_state(false),
        losses,
    )
}

fn independent_source_pair(
    prior_impact: f64,
    marginal_error_probability: f64,
    losses: [f64; 4],
) -> Result<ScenarioReport, String> {
    let patterns = ["--", "-+", "+-", "++"];
    let q = marginal_error_probability;
    let conditional = |impact_state: bool| -> Vec<f64> {
        patterns
            .iter()
            .map(|pattern| {
                pattern
                    .chars()
                    .map(|symbol| {
                        let expected = if impact_state { '+' } else { '-' };
                        if symbol == expected { 1.0 - q } else { q }
                    })
                    .product::<f64>()
            })
            .collect()
    };
    scenario(
        prior_impact,
        &conditional(true),
        &conditional(false),
        losses,
    )
}

fn verify_outcome_rows(
    report: &ScenarioReport,
    fixture: &Value,
    path: &[&str],
) -> Result<(), String> {
    let mut node = fixture;
    for part in path {
        node = if let Ok(index) = part.parse::<usize>() {
            node.get(index)
        } else {
            node.get(*part)
        }
        .ok_or_else(|| format!("missing outcome table: {}", path.join(".")))?;
    }
    let rows = node
        .as_array()
        .ok_or_else(|| format!("outcome table is not an array: {}", path.join(".")))?;
    if rows.len() != report.outcome_probabilities.len() {
        return Err(format!("outcome row count mismatch: {}", path.join(".")));
    }
    let tolerance = number(fixture, &["numerical_tolerance"])?;
    for (index, row) in rows.iter().enumerate() {
        let prefix = format!("{}[{index}]", path.join("."));
        let expected_probability = row
            .get("probability")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing {prefix}.probability"))?;
        let expected_posterior = row
            .get("posterior_impact_probability")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing {prefix}.posterior_impact_probability"))?;
        let expected_risk = row
            .get("conditional_bayes_risk")
            .and_then(Value::as_f64)
            .ok_or_else(|| format!("missing {prefix}.conditional_bayes_risk"))?;
        let expected_action = row
            .get("bayes_action")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("missing {prefix}.bayes_action"))?;
        close(
            report.outcome_probabilities[index],
            expected_probability,
            tolerance,
            &format!("{prefix}.probability"),
        )?;
        close(
            report.posterior_impact_probabilities[index],
            expected_posterior,
            tolerance,
            &format!("{prefix}.posterior"),
        )?;
        close(
            report.conditional_bayes_risks[index],
            expected_risk,
            tolerance,
            &format!("{prefix}.risk"),
        )?;
        if report.bayes_actions[index] != expected_action {
            return Err(format!(
                "{prefix}.action mismatch: actual={}, expected={expected_action}",
                report.bayes_actions[index]
            ));
        }
    }
    Ok(())
}

fn verify_against_fixture(report: &VerificationReport, fixture: &Value) -> Result<(), String> {
    let tolerance = number(fixture, &["numerical_tolerance"])?;
    let close_fixture = |actual: f64, path: &[&str]| -> Result<(), String> {
        close(actual, number(fixture, path)?, tolerance, &path.join("."))
    };

    close_fixture(report.baseline_risk, &["decision_model", "baseline_expected_bayes_risk"])?;
    close_fixture(
        report.nuisance_entropy_bits,
        &["exact_information_value_controls", "0", "expected_entropy_reduction_bits"],
    )?;
    close_fixture(
        report.perfect_information_evsi,
        &["exact_information_value_controls", "1", "expected_value_of_sample_information"],
    )?;
    close_fixture(
        entropy(&[prior, safe_prior]),
        &["exact_information_value_controls", "1", "expected_entropy_reduction_bits"],
    )?;
    let baseline_action = conditional_decision(
        prior,
        losses[0],
        losses[1],
        losses[2],
        losses[3],
    ).1;
    if baseline_action != string(fixture, &["decision_model", "baseline_bayes_action"])? {
        return Err("baseline Bayes action does not match the fixture".to_owned());
    }

    close_fixture(
        report.single_reading.expected_risk,
        &[
            "correlated_observation_control",
            "single_reading",
            "expected_post_observation_bayes_risk",
        ],
    )?;
    close_fixture(
        report.single_reading.evsi,
        &[
            "correlated_observation_control",
            "single_reading",
            "expected_value_of_sample_information",
        ],
    )?;
    close_fixture(
        report.single_reading.information_gain_bits,
        &["correlated_observation_control", "single_reading", "expected_information_gain_bits"],
    )?;

    close_fixture(
        report.same_source_pair.expected_risk,
        &[
            "correlated_observation_control",
            "two_readings_same_source",
            "expected_post_observation_bayes_risk",
        ],
    )?;
    close_fixture(
        report.same_source_pair.evsi,
        &[
            "correlated_observation_control",
            "two_readings_same_source",
            "expected_value_of_sample_information",
        ],
    )?;
    close_fixture(
        report.same_source_pair.information_gain_bits,
        &[
            "correlated_observation_control",
            "two_readings_same_source",
            "expected_information_gain_bits",
        ],
    )?;

    close_fixture(
        report.independent_source_pair.expected_risk,
        &[
            "correlated_observation_control",
            "two_readings_independent_sources",
            "expected_post_observation_bayes_risk",
        ],
    )?;
    close_fixture(
        report.independent_source_pair.evsi,
        &[
            "correlated_observation_control",
            "two_readings_independent_sources",
            "expected_value_of_sample_information",
        ],
    )?;
    close_fixture(
        report.independent_source_pair.information_gain_bits,
        &[
            "correlated_observation_control",
            "two_readings_independent_sources",
            "expected_information_gain_bits",
        ],
    )?;

    close_fixture(
        report.same_source_pair.expected_risk - report.independent_source_pair.expected_risk,
        &["correlated_observation_control", "naive_independence_failure", "risk_underestimation"],
    )?;
    close_fixture(
        report.same_source_pair.evsi - report.independent_source_pair.evsi,
        &["correlated_observation_control", "naive_independence_failure", "evsi_overstatement"],
    )?;

    verify_outcome_rows(
        &report.single_reading,
        fixture,
        &["correlated_observation_control", "single_reading", "outcomes"],
    )?;
    verify_outcome_rows(
        &report.same_source_pair,
        fixture,
        &["correlated_observation_control", "two_readings_same_source", "outcome_table"],
    )?;
    verify_outcome_rows(
        &report.independent_source_pair,
        fixture,
        &["correlated_observation_control", "two_readings_independent_sources", "outcome_table"],
    )?;

    let nuisance_prior = fixture
        .get("exact_information_value_controls")
        .and_then(|items| items.get(0))
        .and_then(|item| item.get("nuisance_prior"))
        .and_then(Value::as_array)
        .ok_or_else(|| "missing nuisance prior".to_owned())?
        .iter()
        .map(|value| value.as_f64().ok_or_else(|| "invalid nuisance prior probability".to_owned()))
        .collect::<Result<Vec<_>, _>>()?;
    close(
        entropy(&nuisance_prior),
        report.nuisance_entropy_bits,
        tolerance,
        "nuisance entropy fixture",
    )?;
    let nuisance_mi = number(
        fixture,
        &[
            "exact_information_value_controls",
            "0",
            "expected_mutual_information_with_decision_state_bits",
        ],
    )?;
    close(nuisance_mi, 0.0, tolerance, "nuisance mutual information")?;
    let nuisance_risk = number(
        fixture,
        &[
            "exact_information_value_controls",
            "0",
            "expected_post_observation_bayes_risk",
        ],
    )?;
    close(
        nuisance_risk,
        report.baseline_risk,
        tolerance,
        "nuisance observation leaves decision risk unchanged",
    )?;
    let nuisance_value = number(
        fixture,
        &[
            "exact_information_value_controls",
            "0",
            "expected_value_of_sample_information",
        ],
    )?;
    close(nuisance_value, 0.0, tolerance, "nuisance observation EVSI")?;
    let perfect_information_risk = number(
        fixture,
        &[
            "exact_information_value_controls",
            "1",
            "expected_post_observation_bayes_risk",
        ],
    )?;
    close(
        perfect_information_risk,
        report.baseline_risk - report.perfect_information_evsi,
        tolerance,
        "perfect-information post-observation risk",
    )?;
    let decision_relevant_value = number(
        fixture,
        &[
            "exact_information_value_controls",
            "1",
            "expected_value_of_sample_information",
        ],
    )?;
    if !(nuisance_value < decision_relevant_value) {
        return Err("decision-value ranking invariant failed".to_owned());
    }
    if number(
        fixture,
        &["correlated_observation_control", "naive_independence_failure", "risk_underestimation"],
    )? <= 0.0
    {
        return Err(
            "correlation negative control must expose positive risk underestimation".to_owned(),
        );
    }
    Ok(())
}

pub fn verify_unknown_dependence(fixture: &Value) -> Result<(), String> {
    if string(
        fixture,
        &["unknown_dependence_control", "mycelix_profile_schema"],
    )? != "mycelix.neo_evidence.source_correlation_profile.v0.1"
    {
        return Err("unknown-dependence profile schema mismatch".to_owned());
    }
    if string(fixture, &["unknown_dependence_control", "assessment"])? != "unknown" {
        return Err("unknown-dependence control must retain unknown assessment".to_owned());
    }
    let independent_permitted = fixture
        .get("unknown_dependence_control")
        .and_then(|value| value.get("independent_product_permitted"))
        .and_then(Value::as_bool)
        .ok_or_else(|| "missing independent_product_permitted flag".to_owned())?;
    let joint_likelihood_available = fixture
        .get("unknown_dependence_control")
        .and_then(|value| value.get("joint_likelihood_available"))
        .and_then(Value::as_bool)
        .ok_or_else(|| "missing joint_likelihood_available flag".to_owned())?;
    let numeric_evsi = fixture
        .get("unknown_dependence_control")
        .and_then(|value| value.get("numeric_evsi"))
        .ok_or_else(|| "missing numeric_evsi field".to_owned())?;
    if independent_permitted || joint_likelihood_available || !numeric_evsi.is_null() {
        return Err(
            "unresolved dependence must not produce an independence product or numeric EVSI"
                .to_owned(),
        );
    }
    if string(
        fixture,
        &["unknown_dependence_control", "required_behavior"],
    )? != "ABSTAIN_DEPENDENCE_UNRESOLVED"
    {
        return Err("unresolved dependence must force explicit abstention".to_owned());
    }
    Ok(())
}

fn verify_fixture_text(text: &str) -> Result<VerificationReport, String> {
    let fixture: Value =
        serde_json::from_str(text).map_err(|error| format!("invalid fixture JSON: {error}"))?;
    if string(&fixture, &["schema_version"])?
        != "symthaea.neo_observation_voi.finite_case.v0.1"
    {
        return Err("unsupported schema_version".to_owned());
    }
    if string(&fixture, &["claim_ceiling"])?
        != "SYNTHETIC_ANALYTIC_ORACLE_ONLY"
    {
        return Err("claim ceiling mismatch".to_owned());
    }
    if string(&fixture, &["data_status"])?
        != "synthetic_exact_finite_case"
    {
        return Err("fixture data_status mismatch".to_owned());
    }
    verify_unknown_dependence(&fixture)?;

    let prior = number(&fixture, &["decision_model", "prior", "impact_mode_probability"])?;
    let safe_prior = number(&fixture, &["decision_model", "prior", "safe_mode_probability"])?;
    let tolerance = number(&fixture, &["numerical_tolerance"])?;
    close(prior + safe_prior, 1.0, tolerance, "prior probability sum")?;

    let losses = [
        number(&fixture, &["decision_model", "actions", "protect", "loss_in_impact_mode"] )?,
        number(&fixture, &["decision_model", "actions", "protect", "loss_in_safe_mode"] )?,
        number(&fixture, &["decision_model", "actions", "no_protect", "loss_in_impact_mode"] )?,
        number(&fixture, &["decision_model", "actions", "no_protect", "loss_in_safe_mode"] )?,
    ];
    let baseline = conditional_risk(prior, losses[0], losses[1], losses[2], losses[3]);
    let perfect_information_risk =
        prior * conditional_risk(1.0, losses[0], losses[1], losses[2], losses[3])
            + safe_prior * conditional_risk(0.0, losses[0], losses[1], losses[2], losses[3]);
    let perfect_information_evsi = baseline - perfect_information_risk;

    let nuisance_entropy = entropy(&[0.5, 0.5]);
    let bad_calibration = number(
        &fixture,
        &[
            "correlated_observation_control",
            "latent_observation_model",
            "shared_calibration_bias_probability",
        ],
    )?;
    let good_error = number(
        &fixture,
        &[
            "correlated_observation_control",
            "latent_observation_model",
            "independent_per_reading_error_probability_given_calibration_state",
        ],
    )?;
    let marginal_error = (1.0 - bad_calibration) * good_error
        + bad_calibration * (1.0 - good_error);

    let single = scenario(
        prior,
        &[marginal_error, 1.0 - marginal_error],
        &[1.0 - marginal_error, marginal_error],
        losses,
    )?;
    let same_source = shared_source_pair(prior, bad_calibration, good_error, losses)?;
    let independent = independent_source_pair(prior, marginal_error, losses)?;

    let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
    let report = VerificationReport {
        baseline_risk: baseline,
        nuisance_entropy_bits: nuisance_entropy,
        perfect_information_evsi,
        single_reading: single,
        same_source_pair: same_source,
        independent_source_pair: independent,
        sha256: digest,
    };
    verify_against_fixture(&report, &fixture)?;
    Ok(report)
}

pub fn verify_frozen_fixture() -> Result<VerificationReport, String> {
    verify_fixture_text(FIXTURE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_fixture_passes_exact_decision_value_oracle() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        close(report.baseline_risk, 1.0, 1e-12, "baseline risk").unwrap();
        close(report.perfect_information_evsi, 0.98, 1e-12, "perfect-info EVSI").unwrap();
    }

    #[test]
    fn entropy_gain_can_be_irrelevant_to_the_decision() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        close(report.nuisance_entropy_bits, 1.0, 1e-12, "nuisance entropy").unwrap();
        close(report.nuisance_entropy_bits, 1.0, 1e-12, "nuisance MI control").unwrap();
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        let nuisance_evsi = number(
            &fixture,
            &["exact_information_value_controls", "0", "expected_value_of_sample_information"],
        ).unwrap();
        close(nuisance_evsi, 0.0, 1e-12, "nuisance EVSI").unwrap();
    }

    #[test]
    fn shared_source_bias_is_not_treated_as_independent_noise() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        close(report.same_source_pair.expected_risk, 0.3869, 1e-12, "same-source risk").unwrap();
        close(
            report.independent_source_pair.expected_risk,
            0.314,
            1e-12,
            "independent-source risk",
        )
        .unwrap();
        assert!(
            report.same_source_pair.expected_risk > report.independent_source_pair.expected_risk
        );
    }

    #[test]
    fn same_source_pair_does_not_overstate_evsi() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        close(report.same_source_pair.evsi, 0.6131, 1e-12, "same-source EVSI").unwrap();
        close(
            report.independent_source_pair.evsi,
            0.686,
            1e-12,
            "independent-source EVSI",
        )
        .unwrap();
        assert!(report.same_source_pair.evsi < report.independent_source_pair.evsi);
    }

    #[test]
    fn outcome_probabilities_are_normalized() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        for (label, probabilities) in [
            ("single", report.single_reading.outcome_probabilities),
            ("same-source", report.same_source_pair.outcome_probabilities),
            ("independent", report.independent_source_pair.outcome_probabilities),
        ] {
            close(probabilities.iter().sum(), 1.0, 1e-12, label).unwrap();
            assert!(probabilities.iter().all(|p| (0.0..=1.0).contains(p)));
        }
    }

    #[test]
    fn posteriors_remain_in_probability_domain() {
        let report = verify_frozen_fixture().expect("fixture should pass");
        for posterior in report
            .single_reading
            .posterior_impact_probabilities
            .iter()
            .chain(report.same_source_pair.posterior_impact_probabilities.iter())
            .chain(report.independent_source_pair.posterior_impact_probabilities.iter())
        {
            assert!((0.0..=1.0).contains(posterior));
        }
    }

    #[test]
    fn generated_finite_models_match_independent_joint_loss_enumerator() {
        fn next_unit(seed: &mut u64) -> f64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((*seed >> 11) as f64) / ((1_u64 << 53) as f64)
        }

        fn normalized(seed: &mut u64) -> Vec<f64> {
            let weights: Vec<f64> = (0..4).map(|_| 0.01 + next_unit(seed)).collect();
            let total: f64 = weights.iter().sum();
            weights.into_iter().map(|weight| weight / total).collect()
        }

        let losses = [1.0, 1.0, 100.0, 0.0];
        let mut seed = 0x4e45_4f56_4f49_u64;

        for _case in 0..128 {
            let prior = 0.005 + 0.20 * next_unit(&mut seed);
            let conditional_impact = normalized(&mut seed);
            let conditional_safe = normalized(&mut seed);
            let evaluated = scenario(
                prior,
                &conditional_impact,
                &conditional_safe,
                losses,
            )
            .expect("positive finite likelihoods should evaluate");

            // Independent identity: score each action directly in joint
            // probability space, then minimize before summing outcomes.
            // This avoids using the evaluator's posterior-normalize-then-risk path.
            let direct_joint_risk: f64 = (0..4)
                .map(|index| {
                    let impact_joint = prior * conditional_impact[index];
                    let safe_joint = (1.0 - prior) * conditional_safe[index];
                    let protect_joint = impact_joint * losses[0] + safe_joint * losses[1];
                    let no_protect_joint = impact_joint * losses[2] + safe_joint * losses[3];
                    protect_joint.min(no_protect_joint)
                })
                .sum();
            let baseline = conditional_risk(prior, losses[0], losses[1], losses[2], losses[3]);
            close(
                evaluated.expected_risk,
                direct_joint_risk,
                1e-12,
                "generated direct joint Bayes risk",
            )
            .unwrap();
            close(
                evaluated.evsi,
                baseline - direct_joint_risk,
                1e-12,
                "generated direct joint EVSI",
            )
            .unwrap();

            // Permuting outcome labels must not change risk or information.
            let reversed_impact: Vec<f64> =
                conditional_impact.iter().rev().copied().collect();
            let reversed_safe: Vec<f64> =
                conditional_safe.iter().rev().copied().collect();
            let permuted = scenario(prior, &reversed_impact, &reversed_safe, losses)
                .expect("permuted positive likelihoods should evaluate");
            close(
                evaluated.expected_risk,
                permuted.expected_risk,
                1e-12,
                "outcome permutation risk invariance",
            )
            .unwrap();
            close(
                evaluated.information_gain_bits,
                permuted.information_gain_bits,
                1e-12,
                "outcome permutation information invariance",
            )
            .unwrap();
        }
    }

    #[test]
    fn nuisance_information_cannot_be_counted_as_decision_value() {
        let invalid = FIXTURE.replace(
            "\"expected_value_of_sample_information\": 0.0",
            "\"expected_value_of_sample_information\": 0.01",
        );
        assert!(verify_fixture_text(&invalid).is_err());
    }

    #[test]
    fn unknown_source_dependence_forces_abstention() {
        let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
        verify_unknown_dependence(&fixture).expect("unknown dependence should force abstention");
        assert_eq!(
            string(&fixture, &["unknown_dependence_control", "required_behavior"]).unwrap(),
            "ABSTAIN_DEPENDENCE_UNRESOLVED"
        );
    }

    #[test]
    fn unknown_source_dependence_cannot_be_promoted_to_independence() {
        let invalid = FIXTURE.replace(
            "\"independent_product_permitted\": false",
            "\"independent_product_permitted\": true",
        );
        assert!(verify_fixture_text(&invalid).is_err());
    }

    #[test]
    fn unresolved_dependence_cannot_emit_numeric_evsi() {
        let invalid = FIXTURE.replace(
            "\"numeric_evsi\": null",
            "\"numeric_evsi\": 0.686",
        );
        assert!(verify_fixture_text(&invalid).is_err());
    }

    #[test]
    fn malformed_json_fails_closed() {
        assert!(verify_fixture_text("{not-json").is_err());
    }

    #[test]
    fn wrong_claim_ceiling_fails_closed() {
        let invalid = FIXTURE.replace(
            "SYNTHETIC_ANALYTIC_ORACLE_ONLY",
            "HISTORICAL_REPLAY",
        );
        assert!(verify_fixture_text(&invalid).is_err());
    }

    #[test]
    fn altered_outcome_value_is_detected() {
        let invalid = FIXTURE.replace(
            "\"expected_post_observation_bayes_risk\": 0.3869",
            "\"expected_post_observation_bayes_risk\": 0.3868",
        );
        assert!(verify_fixture_text(&invalid).is_err());
    }
}

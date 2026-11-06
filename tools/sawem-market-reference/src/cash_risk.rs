//! Exact probability-weighted downside metrics for source-bound BESS scenarios.
//!
//! Scenario probabilities must be supplied explicitly and sum to 100% in
//! basis points. The module computes exact integer numerators for expected
//! net cash and lower-tail CVaR, including partial probability mass at the
//! quantile boundary. It does not estimate scenario probabilities or imply
//! that a synthetic scenario tree is a calibrated real-world distribution.

use std::collections::BTreeSet;

const PROBABILITY_BPS: u128 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CashScenarioOutcome<'a> {
    pub scenario_id: &'a str,
    pub result_source_ref: &'a str,
    pub result_source_sha256: &'a str,
    pub probability_source_ref: &'a str,
    pub probability_source_sha256: &'a str,
    pub result_available_at_utc: i64,
    pub probability_available_at_utc: i64,
    pub probability_bps: u16,
    /// Final net cash in minor ZAR, after the declared model's costs.
    pub net_cash_minor_zar: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CashRiskInputs<'a> {
    pub portfolio_id: &'a str,
    pub evaluation_as_of_utc: i64,
    /// Lower-tail probability in basis points. 500 = worst 5%.
    pub lower_tail_probability_bps: u16,
    pub scenarios: &'a [CashScenarioOutcome<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CashRiskError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidTailProbability,
    InvalidScenarioProbability,
    ProbabilitiesDoNotSumToOne,
    FutureKnownScenario,
    DuplicateScenarioId,
    DuplicateResultSourceReference,
    ArithmeticOverflow,
    EmptyScenarioSet,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CashRiskAssessment<'a> {
    pub portfolio_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub scenario_count: u64,
    pub total_probability_bps: u16,
    pub lower_tail_probability_bps: u16,
    /// Exact expected-value numerator; divide by 10,000 for minor ZAR.
    pub expected_net_cash_numerator_minor_zar_bps: i128,
    pub expected_net_cash_denominator_bps: u128,
    /// Lowest scenario cash value at which cumulative probability reaches the
    /// requested lower-tail probability.
    pub lower_tail_var_minor_zar: i64,
    /// Exact CVaR numerator; divide by lower_tail_probability_bps.
    pub lower_tail_cvar_numerator_minor_zar_bps: i128,
    pub lower_tail_cvar_denominator_bps: u16,
    pub worst_case_net_cash_minor_zar: i64,
    pub probability_of_negative_net_cash_bps: u16,
    pub negative_cash_scenario_count: u64,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn checked_add_i128(current: i128, value: i128) -> Result<i128, CashRiskError> {
    current
        .checked_add(value)
        .ok_or(CashRiskError::ArithmeticOverflow)
}

/// Compute expected net cash, worst-case net cash, P(net cash < 0), lower-tail
/// VaR and lower-tail CVaR using exact probability-basis-point arithmetic.
pub fn evaluate_cash_risk<'a>(
    inputs: CashRiskInputs<'a>,
) -> Result<CashRiskAssessment<'a>, CashRiskError> {
    if !nonempty(inputs.portfolio_id) {
        return Err(CashRiskError::InvalidEvidenceReference);
    }
    if inputs.scenarios.is_empty() {
        return Err(CashRiskError::EmptyScenarioSet);
    }
    if inputs.lower_tail_probability_bps == 0
        || u128::from(inputs.lower_tail_probability_bps) > PROBABILITY_BPS
    {
        return Err(CashRiskError::InvalidTailProbability);
    }

    let mut scenario_ids = BTreeSet::new();
    let mut result_refs = BTreeSet::new();
    let mut total_probability = 0_u32;
    let mut expected_numerator = 0_i128;
    let mut loss_probability = 0_u32;
    let mut negative_scenario_count = 0_u64;
    let mut worst_case = i64::MAX;
    let mut ordered = inputs.scenarios.to_vec();

    for scenario in inputs.scenarios.iter().copied() {
        if !nonempty(scenario.scenario_id)
            || !nonempty(scenario.result_source_ref)
            || !nonempty(scenario.probability_source_ref)
        {
            return Err(CashRiskError::InvalidEvidenceReference);
        }
        if !valid_sha256(scenario.result_source_sha256)
            || !valid_sha256(scenario.probability_source_sha256)
        {
            return Err(CashRiskError::InvalidSourceDigest);
        }
        if scenario.probability_bps == 0 {
            return Err(CashRiskError::InvalidScenarioProbability);
        }
        if scenario.result_available_at_utc > inputs.evaluation_as_of_utc
            || scenario.probability_available_at_utc > inputs.evaluation_as_of_utc
        {
            return Err(CashRiskError::FutureKnownScenario);
        }
        if !scenario_ids.insert(scenario.scenario_id) {
            return Err(CashRiskError::DuplicateScenarioId);
        }
        if !result_refs.insert(scenario.result_source_ref) {
            return Err(CashRiskError::DuplicateResultSourceReference);
        }

        total_probability = total_probability
            .checked_add(u32::from(scenario.probability_bps))
            .ok_or(CashRiskError::ArithmeticOverflow)?;
        let weighted_cash = i128::from(scenario.net_cash_minor_zar)
            .checked_mul(i128::from(scenario.probability_bps))
            .ok_or(CashRiskError::ArithmeticOverflow)?;
        expected_numerator = checked_add_i128(expected_numerator, weighted_cash)?;
        if scenario.net_cash_minor_zar < 0 {
            loss_probability = loss_probability
                .checked_add(u32::from(scenario.probability_bps))
                .ok_or(CashRiskError::ArithmeticOverflow)?;
            negative_scenario_count = negative_scenario_count
                .checked_add(1)
                .ok_or(CashRiskError::ArithmeticOverflow)?;
        }
        worst_case = worst_case.min(scenario.net_cash_minor_zar);
    }

    if u128::from(total_probability) != PROBABILITY_BPS {
        return Err(CashRiskError::ProbabilitiesDoNotSumToOne);
    }

    ordered.sort_by_key(|scenario| scenario.net_cash_minor_zar);
    let tail_target = u32::from(inputs.lower_tail_probability_bps);
    let mut tail_remaining = tail_target;
    let mut cumulative_probability = 0_u32;
    let mut tail_numerator = 0_i128;
    let mut tail_var = ordered[0].net_cash_minor_zar;

    for scenario in ordered {
        let probability = u32::from(scenario.probability_bps);
        let take = probability.min(tail_remaining);
        if take > 0 {
            tail_var = scenario.net_cash_minor_zar;
            let weighted = i128::from(scenario.net_cash_minor_zar)
                .checked_mul(i128::from(take))
                .ok_or(CashRiskError::ArithmeticOverflow)?;
            tail_numerator = checked_add_i128(tail_numerator, weighted)?;
            cumulative_probability = cumulative_probability
                .checked_add(take)
                .ok_or(CashRiskError::ArithmeticOverflow)?;
            tail_remaining = tail_remaining
                .checked_sub(take)
                .ok_or(CashRiskError::ArithmeticOverflow)?;
        }
        if tail_remaining == 0 {
            break;
        }
    }
    if tail_remaining != 0 || cumulative_probability != tail_target {
        return Err(CashRiskError::ArithmeticOverflow);
    }

    Ok(CashRiskAssessment {
        portfolio_id: inputs.portfolio_id,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        scenario_count: u64::try_from(inputs.scenarios.len())
            .map_err(|_| CashRiskError::ArithmeticOverflow)?,
        total_probability_bps: u16::try_from(total_probability)
            .map_err(|_| CashRiskError::ArithmeticOverflow)?,
        lower_tail_probability_bps: inputs.lower_tail_probability_bps,
        expected_net_cash_numerator_minor_zar_bps: expected_numerator,
        expected_net_cash_denominator_bps: PROBABILITY_BPS,
        lower_tail_var_minor_zar: tail_var,
        lower_tail_cvar_numerator_minor_zar_bps: tail_numerator,
        lower_tail_cvar_denominator_bps: inputs.lower_tail_probability_bps,
        worst_case_net_cash_minor_zar: worst_case,
        probability_of_negative_net_cash_bps: u16::try_from(loss_probability)
            .map_err(|_| CashRiskError::ArithmeticOverflow)?,
        negative_cash_scenario_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn scenario(id: &'static str, probability: u16, cash: i64) -> CashScenarioOutcome<'static> {
        CashScenarioOutcome {
            scenario_id: id,
            result_source_ref: id,
            result_source_sha256: SOURCE_SHA,
            probability_source_ref: "probability/manifest/v1",
            probability_source_sha256: SOURCE_SHA,
            result_available_at_utc: 90,
            probability_available_at_utc: 85,
            probability_bps: probability,
            net_cash_minor_zar: cash,
        }
    }

    fn inputs<'a>(scenarios: &'a [CashScenarioOutcome<'a>], tail_bps: u16) -> CashRiskInputs<'a> {
        CashRiskInputs {
            portfolio_id: "battery-001/base-and-stress",
            evaluation_as_of_utc: 100,
            lower_tail_probability_bps: tail_bps,
            scenarios,
        }
    }

    #[test]
    fn expected_cash_var_cvar_and_loss_probability_have_exact_answers() {
        let scenarios = [
            scenario("downside", 1_000, -100),
            scenario("base", 4_000, 0),
            scenario("upside", 5_000, 100),
        ];
        let result = evaluate_cash_risk(inputs(&scenarios, 500)).unwrap();
        assert_eq!(result.expected_net_cash_numerator_minor_zar_bps, 400_000);
        assert_eq!(result.expected_net_cash_denominator_bps, 10_000);
        assert_eq!(result.lower_tail_var_minor_zar, -100);
        assert_eq!(result.lower_tail_cvar_numerator_minor_zar_bps, -50_000);
        assert_eq!(result.lower_tail_cvar_denominator_bps, 500);
        assert_eq!(result.worst_case_net_cash_minor_zar, -100);
        assert_eq!(result.probability_of_negative_net_cash_bps, 1_000);
        assert_eq!(result.negative_cash_scenario_count, 1);
    }

    #[test]
    fn cvar_partially_weights_the_scenario_at_the_tail_boundary() {
        let scenarios = [
            scenario("downside", 1_000, -100),
            scenario("base", 4_000, 0),
            scenario("upside", 5_000, 100),
        ];
        let result = evaluate_cash_risk(inputs(&scenarios, 1_500)).unwrap();
        // Worst 1,000 bps are -100; next 500 bps are 0.
        assert_eq!(result.lower_tail_var_minor_zar, 0);
        assert_eq!(result.lower_tail_cvar_numerator_minor_zar_bps, -100_000);
        assert_eq!(result.lower_tail_cvar_denominator_bps, 1_500);
    }

    #[test]
    fn probabilities_must_sum_exactly_to_one_hundred_percent() {
        let scenarios = [
            scenario("downside", 1_000, -100),
            scenario("base", 4_000, 0),
        ];
        assert_eq!(
            evaluate_cash_risk(inputs(&scenarios, 500)),
            Err(CashRiskError::ProbabilitiesDoNotSumToOne)
        );
    }

    #[test]
    fn future_known_duplicate_and_bad_source_evidence_fail_closed() {
        let future_result = [CashScenarioOutcome {
            result_available_at_utc: 101,
            ..scenario("future-result", 10_000, 100)
        }];
        let future_probability = [CashScenarioOutcome {
            probability_available_at_utc: 101,
            ..scenario("future-probability", 10_000, 100)
        }];
        assert_eq!(
            evaluate_cash_risk(inputs(&future, 500)),
            Err(CashRiskError::FutureKnownScenario)
        );

        let duplicate = [
            scenario("same", 5_000, -100),
            CashScenarioOutcome {
                result_source_ref: "other",
                ..scenario("same", 5_000, 100)
            },
        ];
        assert_eq!(
            evaluate_cash_risk(inputs(&duplicate, 500)),
            Err(CashRiskError::DuplicateScenarioId)
        );

        let bad_hash = [CashScenarioOutcome {
            result_source_sha256: "invalid",
            ..scenario("bad-hash", 10_000, 100)
        }];
        assert_eq!(
            evaluate_cash_risk(inputs(&bad_hash, 500)),
            Err(CashRiskError::InvalidSourceDigest)
        );
    }

    #[test]
    fn zero_probability_and_empty_scenario_sets_are_rejected() {
        assert_eq!(
            evaluate_cash_risk(inputs(&[], 500)),
            Err(CashRiskError::EmptyScenarioSet)
        );
        let zero_probability = [scenario("zero", 0, 100)];
        assert_eq!(
            evaluate_cash_risk(inputs(&zero_probability, 500)),
            Err(CashRiskError::InvalidScenarioProbability)
        );
        assert_eq!(
            evaluate_cash_risk(inputs(&[scenario("single", 10_000, 100)], 0)),
            Err(CashRiskError::InvalidTailProbability)
        );
    }
}

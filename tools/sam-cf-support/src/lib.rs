//! Small, dependency-free qualification harness for counterfactual identification.
//!
//! This is a fixture evaluator, not a causal estimator or a general identifiability
//! engine. It checks that declared support and assumption evidence leads to the
//! conservative disposition preregistered by each analytic fixture.

use std::collections::{BTreeMap, BTreeSet};

const EPSILON: f64 = 1.0e-12;
const IDENTIFICATION_FIXTURES: &str = include_str!("../fixtures/identification-cases-v0.1.tsv");
const CROSS_WORLD_FIXTURES: &str = include_str!("../fixtures/cross-world-couplings-v0.1.tsv");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Disposition {
    PointIdentified,
    SetIdentified,
    AssumptionSensitive,
    UnsupportedIntervention,
    NotIdentified,
    Incomplete,
    OutOfScope,
    FalsePrecisionRejected,
}

impl Disposition {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "POINT_IDENTIFIED" => Ok(Self::PointIdentified),
            "SET_IDENTIFIED" => Ok(Self::SetIdentified),
            "ASSUMPTION_SENSITIVE" => Ok(Self::AssumptionSensitive),
            "UNSUPPORTED_INTERVENTION" => Ok(Self::UnsupportedIntervention),
            "NOT_IDENTIFIED" => Ok(Self::NotIdentified),
            "INCOMPLETE" => Ok(Self::Incomplete),
            "OUT_OF_SCOPE" => Ok(Self::OutOfScope),
            "FALSE_PRECISION_REJECTED" => Ok(Self::FalsePrecisionRejected),
            other => Err(format!("unknown disposition: {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Case {
    id: String,
    estimand: String,
    support: String,
    rho_low: f64,
    rho_high: f64,
    abduction: String,
    factual_tail: String,
    mechanism: String,
    decision: String,
    complete: bool,
    assumptions: String,
    identified_low: f64,
    identified_high: f64,
    reported_low: f64,
    reported_high: f64,
    expected: Disposition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Assessment {
    pub disposition: Disposition,
    /// None means the fixture does not make an action-ranking claim.
    pub decision_invariant: Option<bool>,
}

/// Parse and validate all deterministic identification fixtures.
pub fn parse_cases(input: &str) -> Result<Vec<Case>, String> {
    let mut cases = Vec::new();

    for (line_index, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("case_id|") {
            continue;
        }

        let fields: Vec<&str> = line.split('|').collect();
        if fields.len() != 16 {
            return Err(format!(
                "line {}: expected 16 pipe-delimited fields, found {}",
                line_index + 1,
                fields.len()
            ));
        }

        let parse_number = |index: usize, name: &str| -> Result<f64, String> {
            let value = fields[index]
                .parse::<f64>()
                .map_err(|error| format!("line {}: invalid {name}: {error}", line_index + 1))?;
            if !value.is_finite() {
                return Err(format!("line {}: {name} must be finite", line_index + 1));
            }
            Ok(value)
        };

        let complete = match fields[9] {
            "true" => true,
            "false" => false,
            other => {
                return Err(format!(
                    "line {}: complete must be true or false, got {other}",
                    line_index + 1
                ));
            }
        };

        let case = Case {
            id: fields[0].to_owned(),
            estimand: fields[1].to_owned(),
            support: fields[2].to_owned(),
            rho_low: parse_number(3, "rho_low")?,
            rho_high: parse_number(4, "rho_high")?,
            abduction: fields[5].to_owned(),
            factual_tail: fields[6].to_owned(),
            mechanism: fields[7].to_owned(),
            decision: fields[8].to_owned(),
            complete,
            assumptions: fields[10].to_owned(),
            identified_low: parse_number(11, "identified_low")?,
            identified_high: parse_number(12, "identified_high")?,
            reported_low: parse_number(13, "reported_low")?,
            reported_high: parse_number(14, "reported_high")?,
            expected: Disposition::parse(fields[15])?,
        };

        if case.id.is_empty() {
            return Err(format!(
                "line {}: case_id must not be empty",
                line_index + 1
            ));
        }
        if !matches!(case.estimand.as_str(), "R1" | "R2" | "R3") {
            return Err(format!(
                "line {}: estimand must be R1, R2 or R3",
                line_index + 1
            ));
        }
        if !matches!(
            case.support.as_str(),
            "SUPPORTED" | "UNSUPPORTED" | "GLOBAL_ONLY" | "UNKNOWN"
        ) {
            return Err(format!(
                "line {}: invalid support disposition",
                line_index + 1
            ));
        }
        if !matches!(
            case.abduction.as_str(),
            "IDENTIFIED" | "PARTIAL" | "MISSING" | "NA"
        ) {
            return Err(format!(
                "line {}: invalid abduction disposition",
                line_index + 1
            ));
        }
        if !matches!(
            case.factual_tail.as_str(),
            "PRESENT" | "MISSING" | "NA"
        ) {
            return Err(format!(
                "line {}: invalid factual_tail disposition",
                line_index + 1
            ));
        }
        if !matches!(case.mechanism.as_str(), "IN_SCOPE" | "OUT_OF_SCOPE") {
            return Err(format!("line {}: invalid mechanism scope", line_index + 1));
        }
        if !matches!(case.decision.as_str(), "INVARIANT" | "REVERSAL" | "NA") {
            return Err(format!(
                "line {}: invalid decision disposition",
                line_index + 1
            ));
        }
        if case.assumptions.is_empty() {
            return Err(format!(
                "line {}: assumptions must be explicit",
                line_index + 1
            ));
        }
        if case.rho_low < -1.0 || case.rho_high > 1.0 || case.rho_low > case.rho_high {
            return Err(format!(
                "line {}: cross-world correlation bounds must satisfy -1 <= low <= high <= 1",
                line_index + 1
            ));
        }
        if case.identified_low > case.identified_high {
            return Err(format!(
                "line {}: identified interval is reversed",
                line_index + 1
            ));
        }
        if case.reported_low > case.reported_high {
            return Err(format!(
                "line {}: reported interval is reversed",
                line_index + 1
            ));
        }

        cases.push(case);
    }

    for index in 0..cases.len() {
        if cases[..index]
            .iter()
            .any(|prior| prior.id == cases[index].id)
        {
            return Err(format!("duplicate case_id: {}", cases[index].id));
        }
    }

    if cases.is_empty() {
        return Err("fixture set contains no cases".to_owned());
    }
    Ok(cases)
}

/// Apply the fail-closed fixture disposition rules.
pub fn assess(case: &Case) -> Assessment {
    let decision_invariant = match case.decision.as_str() {
        "INVARIANT" => Some(true),
        "REVERSAL" => Some(false),
        _ => None,
    };

    let base = if !case.complete {
        Disposition::Incomplete
    } else if case.mechanism == "OUT_OF_SCOPE" {
        Disposition::OutOfScope
    } else if case.estimand == "R3"
        && (case.factual_tail != "PRESENT" || matches!(case.abduction.as_str(), "MISSING" | "NA"))
    {
        Disposition::NotIdentified
    } else if case.support != "SUPPORTED" && case.assumptions == "NONE" {
        Disposition::UnsupportedIntervention
    } else if case.decision == "REVERSAL" && case.rho_high - case.rho_low > EPSILON {
        Disposition::AssumptionSensitive
    } else if case.identified_high - case.identified_low > EPSILON
        || case.rho_high - case.rho_low > EPSILON
        || case.support != "SUPPORTED"
        || case.assumptions != "NONE"
        || case.abduction == "PARTIAL"
    {
        Disposition::SetIdentified
    } else {
        Disposition::PointIdentified
    };

    let disposition = if matches!(
        base,
        Disposition::PointIdentified
            | Disposition::SetIdentified
            | Disposition::AssumptionSensitive
    ) && (case.reported_low > case.identified_low + EPSILON
        || case.reported_high < case.identified_high - EPSILON)
    {
        Disposition::FalsePrecisionRejected
    } else {
        base
    };

    Assessment {
        disposition,
        decision_invariant: if disposition == Disposition::FalsePrecisionRejected {
            None
        } else {
            decision_invariant
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_golden_cases_match_their_preregistered_dispositions() {
        let cases = parse_cases(IDENTIFICATION_FIXTURES).expect("valid fixture set");
        assert_eq!(cases.len(), 12);

        for case in &cases {
            let actual = assess(case);
            assert_eq!(
                actual.disposition, case.expected,
                "unexpected disposition for {}",
                case.id
            );
        }
    }

    #[test]
    fn same_marginals_can_hide_different_episode_counterfactuals() {
        let mut marginals = BTreeMap::<(String, String), BTreeMap<u64, f64>>::new();
        let mut conditional = BTreeMap::<String, f64>::new();
        let mut totals = BTreeMap::<String, f64>::new();
        let mut world_ids = BTreeSet::new();

        for (line_index, line) in CROSS_WORLD_FIXTURES.lines().enumerate() {
            if line_index == 0 || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split('|').collect();
            assert_eq!(fields.len(), 5, "bad coupling fixture row");

            let coupling = fields[0];
            let world_id = fields[1];
            assert!(
                world_ids.insert((coupling.to_owned(), world_id.to_owned())),
                "duplicate world id"
            );

            let probability: f64 = fields[2].parse().expect("probability");
            let factual: f64 = fields[3].parse().expect("factual outcome");
            let alternative: f64 = fields[4].parse().expect("alternative outcome");
            assert!(probability.is_finite() && probability > 0.0);
            assert!(factual.is_finite() && alternative.is_finite());

            *totals.entry(coupling.to_owned()).or_insert(0.0) += probability;

            let marginal = marginals
                .entry((coupling.to_owned(), "factual".to_owned()))
                .or_default();
            *marginal.entry(factual.to_bits()).or_insert(0.0) += probability;

            let alternative_marginal = marginals
                .entry((coupling.to_owned(), "alternative".to_owned()))
                .or_default();
            *alternative_marginal
                .entry(alternative.to_bits())
                .or_insert(0.0) += probability;

            if factual.abs() < EPSILON {
                conditional.insert(coupling.to_owned(), alternative);
            }
        }

        assert!(totals.values().all(|total| (*total - 1.0).abs() < EPSILON));

        let positive_factual = marginals
            .get(&("POSITIVE".to_owned(), "factual".to_owned()))
            .expect("positive factual marginal");
        let negative_factual = marginals
            .get(&("NEGATIVE".to_owned(), "factual".to_owned()))
            .expect("negative factual marginal");
        let positive_alternative = marginals
            .get(&("POSITIVE".to_owned(), "alternative".to_owned()))
            .expect("positive alternative marginal");
        let negative_alternative = marginals
            .get(&("NEGATIVE".to_owned(), "alternative".to_owned()))
            .expect("negative alternative marginal");

        assert_eq!(positive_factual, negative_factual);
        assert_eq!(positive_alternative, negative_alternative);
        assert_eq!(conditional.get("POSITIVE"), Some(&0.0));
        assert_eq!(conditional.get("NEGATIVE"), Some(&3.0));
    }

    #[test]
    fn a_wider_identified_set_rejects_a_narrower_claimed_interval() {
        let cases = parse_cases(IDENTIFICATION_FIXTURES).expect("valid fixture set");
        let case = cases
            .iter()
            .find(|case| case.id == "CFI-09")
            .expect("CFI-09");
        assert_eq!(
            assess(case).disposition,
            Disposition::FalsePrecisionRejected
        );
    }

    #[test]
    fn decision_can_be_invariant_even_when_the_outcome_is_set_identified() {
        let cases = parse_cases(IDENTIFICATION_FIXTURES).expect("valid fixture set");
        let case = cases
            .iter()
            .find(|case| case.id == "CFI-10")
            .expect("CFI-10");
        let result = assess(case);
        assert_eq!(result.disposition, Disposition::SetIdentified);
        assert_eq!(result.decision_invariant, Some(true));
    }

    #[test]
    fn malformed_or_reversed_bounds_fail_closed() {
        let malformed = "bad|R3|SUPPORTED|0.8|0.2|IDENTIFIED|PRESENT|IN_SCOPE|INVARIANT|true|NONE|4|4|4|4|POINT_IDENTIFIED";
        assert!(parse_cases(malformed).is_err());
    }

    #[test]
    fn identical_frozen_inputs_produce_identical_assessments() {
        let first = parse_cases(IDENTIFICATION_FIXTURES).expect("valid fixture set");
        let second = parse_cases(IDENTIFICATION_FIXTURES).expect("valid fixture set");
        let left: Vec<Assessment> = first.iter().map(assess).collect();
        let right: Vec<Assessment> = second.iter().map(assess).collect();
        assert_eq!(left, right);
    }
}

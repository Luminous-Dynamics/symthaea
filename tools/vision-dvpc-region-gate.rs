//! VIS-DVPC-031: fail-closed decision gate over value and risk regions.
//! Identification regions and statistical confidence intervals are different evidence types.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;

const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EvidenceState {
    Cleared,
    Rejected,
    Unresolved,
    Missing,
}

impl EvidenceState {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cleared => "CLEARED",
            Self::Rejected => "REJECTED",
            Self::Unresolved => "UNRESOLVED",
            Self::Missing => "MISSING",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    Authorized,
    Denied,
    InsufficientEvidence,
}

impl Decision {
    fn as_str(self) -> &'static str {
        match self {
            Self::Authorized => "AUTHORIZED",
            Self::Denied => "DENIED",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct GateResult {
    decision: Decision,
    value_state: EvidenceState,
    risk_state: EvidenceState,
    value_floor: f64,
    risk_cap: f64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn validate_threshold(value: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(invalid(format!("{name} must be finite and in [0,1]")));
    }
    Ok(())
}

fn validate_interval(interval: Option<(f64, f64)>, name: &str) -> Result<(), String> {
    let Some((lower, upper)) = interval else {
        return Ok(());
    };
    if !lower.is_finite() || !upper.is_finite() {
        return Err(invalid(format!("{name} bounds must be finite")));
    }
    if !(0.0..=1.0).contains(&lower) || !(0.0..=1.0).contains(&upper) {
        return Err(invalid(format!("{name} bounds must be in [0,1]")));
    }
    if lower > upper {
        return Err(invalid(format!("{name} lower bound exceeds upper bound")));
    }
    Ok(())
}

fn assess_value(region: Option<(f64, f64)>, floor: f64) -> EvidenceState {
    match region {
        None => EvidenceState::Missing,
        Some((_, upper)) if upper + EPS < floor => EvidenceState::Rejected,
        Some((lower, _)) if lower + EPS >= floor => EvidenceState::Cleared,
        Some(_) => EvidenceState::Unresolved,
    }
}

fn assess_risk(region: Option<(f64, f64)>, cap: f64) -> EvidenceState {
    match region {
        None => EvidenceState::Missing,
        Some((lower, _)) if lower > cap + EPS => EvidenceState::Rejected,
        Some((_, upper)) if upper <= cap + EPS => EvidenceState::Cleared,
        Some(_) => EvidenceState::Unresolved,
    }
}

fn evaluate(
    value_region: Option<(f64, f64)>,
    risk_region: Option<(f64, f64)>,
    value_floor: f64,
    risk_cap: f64,
) -> Result<GateResult, String> {
    validate_threshold(value_floor, "value_floor")?;
    validate_threshold(risk_cap, "risk_cap")?;
    validate_interval(value_region, "value_region")?;
    validate_interval(risk_region, "risk_region")?;

    let value_state = assess_value(value_region, value_floor);
    let risk_state = assess_risk(risk_region, risk_cap);
    let decision = if value_state == EvidenceState::Rejected || risk_state == EvidenceState::Rejected {
        Decision::Denied
    } else if value_state == EvidenceState::Cleared && risk_state == EvidenceState::Cleared {
        Decision::Authorized
    } else {
        Decision::InsufficientEvidence
    };

    Ok(GateResult {
        decision,
        value_state,
        risk_state,
        value_floor,
        risk_cap,
    })
}

fn parse_bound(value: &str, name: &str) -> Result<Option<f64>, String> {
    if value == "NA" {
        return Ok(None);
    }
    let parsed = value.parse::<f64>().map_err(|_| invalid(format!("invalid {name}")))?;
    if !parsed.is_finite() {
        return Err(invalid(format!("{name} must be finite")));
    }
    Ok(Some(parsed))
}

fn parse_interval(lower: &str, upper: &str, name: &str) -> Result<Option<(f64, f64)>, String> {
    match (parse_bound(lower, name)?, parse_bound(upper, name)?) {
        (None, None) => Ok(None),
        (Some(lower), Some(upper)) => Ok(Some((lower, upper))),
        _ => Err(invalid(format!("{name} endpoints must both be present or both be NA"))),
    }
}

fn render(result: GateResult) -> String {
    let mut output = String::new();
    output.push_str("record_type\tfield\tvalue\n");
    let _ = writeln!(output, "SUMMARY\tdecision\t{}", result.decision.as_str());
    let _ = writeln!(output, "SUMMARY\tvalue_state\t{}", result.value_state.as_str());
    let _ = writeln!(output, "SUMMARY\trisk_state\t{}", result.risk_state.as_str());
    let _ = writeln!(output, "SUMMARY\tvalue_floor\t{:.12}", result.value_floor);
    let _ = writeln!(output, "SUMMARY\trisk_cap\t{:.12}", result.risk_cap);
    output.push_str("LIMIT\tconfidence_interval_semantics\tcaller_must_supply_provenance\n");
    output.push_str("LIMIT\tidentification_region_is_not_a_confidence_interval\ttrue\n");
    output.push_str("LIMIT\tauthorization_is_only_for_the_supplied_value_and_risk_constraints\ttrue\n");
    output
}

fn run(args: &[String]) -> Result<String, String> {
    let [_program, flag, value_lower, value_upper, risk_lower, risk_upper, value_floor, risk_cap] = args else {
        return Err("usage: vision-dvpc-region-gate --evaluate VALUE_LOW VALUE_HIGH RISK_LOW RISK_HIGH VALUE_FLOOR RISK_CAP (use NA NA for missing interval)".into());
    };
    if flag != "--evaluate" {
        return Err("usage: vision-dvpc-region-gate --evaluate VALUE_LOW VALUE_HIGH RISK_LOW RISK_HIGH VALUE_FLOOR RISK_CAP".into());
    }
    let value_region = parse_interval(value_lower, value_upper, "value_region")?;
    let risk_region = parse_interval(risk_lower, risk_upper, "risk_region")?;
    let value_floor = value_floor.parse::<f64>().map_err(|_| invalid("invalid value_floor"))?;
    let risk_cap = risk_cap.parse::<f64>().map_err(|_| invalid("invalid risk_cap"))?;
    Ok(render(evaluate(value_region, risk_region, value_floor, risk_cap)?))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-031 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALUE_FLOOR: f64 = 0.5;
    const RISK_CAP: f64 = 0.05;

    #[test]
    fn authorizes_only_when_entire_value_and_risk_regions_clear() {
        let result = evaluate(Some((0.6, 0.8)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::Authorized);
        assert_eq!(result.value_state, EvidenceState::Cleared);
        assert_eq!(result.risk_state, EvidenceState::Cleared);
    }

    #[test]
    fn denies_when_entire_value_region_is_below_floor() {
        let result = evaluate(Some((0.1, 0.4)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::Denied);
        assert_eq!(result.value_state, EvidenceState::Rejected);
    }

    #[test]
    fn denies_when_entire_risk_region_is_above_cap() {
        let result = evaluate(Some((0.6, 0.8)), Some((0.06, 0.09)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::Denied);
        assert_eq!(result.risk_state, EvidenceState::Rejected);
    }

    #[test]
    fn partial_identification_region_from_dvpc028_requires_more_evidence() {
        let result = evaluate(Some((0.1, 0.6)), Some((0.01, 0.03)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert_eq!(result.value_state, EvidenceState::Unresolved);
        assert_eq!(result.risk_state, EvidenceState::Cleared);
    }

    #[test]
    fn risk_region_that_straddles_cap_requires_more_evidence() {
        let result = evaluate(Some((0.6, 0.8)), Some((0.01, 0.09)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::InsufficientEvidence);
        assert_eq!(result.risk_state, EvidenceState::Unresolved);
    }

    #[test]
    fn inclusive_boundary_authorizes_when_lower_value_equals_floor_and_upper_risk_equals_cap() {
        let result = evaluate(Some((0.5, 0.6)), Some((0.01, 0.05)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::Authorized);
    }

    #[test]
    fn intervals_that_touch_value_floor_but_extend_below_do_not_authorize() {
        let result = evaluate(Some((0.1, 0.5)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(result.decision, Decision::InsufficientEvidence);
    }

    #[test]
    fn missing_bounds_are_not_imputed_or_treated_as_cleared() {
        let missing_value = evaluate(None, Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(missing_value.decision, Decision::InsufficientEvidence);
        let definitely_unsafe = evaluate(None, Some((0.06, 0.09)), VALUE_FLOOR, RISK_CAP).unwrap();
        assert_eq!(definitely_unsafe.decision, Decision::Denied);
    }

    #[test]
    fn malformed_intervals_and_thresholds_fail_closed() {
        assert!(evaluate(Some((0.7, 0.2)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).is_err());
        assert!(evaluate(Some((f64::NAN, 0.8)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).is_err());
        assert!(evaluate(Some((-0.1, 0.8)), Some((0.01, 0.04)), VALUE_FLOOR, RISK_CAP).is_err());
        assert!(evaluate(Some((0.6, 0.8)), Some((0.01, 0.04)), 1.1, RISK_CAP).is_err());
        assert!(parse_interval("0.1", "NA", "value_region").is_err());
    }

    #[test]
    fn render_is_deterministic_and_identification_region_is_not_a_confidence_interval() {
        let result = evaluate(Some((0.1, 0.6)), Some((0.01, 0.03)), VALUE_FLOOR, RISK_CAP).unwrap();
        let text = render(result);
        assert_eq!(text, render(result));
        assert!(text.contains("SUMMARY\tdecision\tINSUFFICIENT_EVIDENCE"));
        assert!(text.contains("identification_region_is_not_a_confidence_interval"));
    }
}

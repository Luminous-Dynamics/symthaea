//! VIS-DVPC-033: compose hash-bound evidence without upgrading diagnostics into authorization.
//! A diagnostic-only bundle is not policy-value certification or a production safety claim.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;
use std::fs;

const UNIVERSE_TRACE_FIELD: &str = "trace_sha256";
const SUPPORT_TRACE_FIELD: &str = "input_sha256";
const ZERO_SUPPORT_SCENARIO: &str = "zero_support";
const TRACE_DIGEST_LEN: usize = 64;

#[derive(Clone, Debug, PartialEq)]
struct BundleReport {
    status: &'static str,
    checks: Vec<(&'static str, bool)>,
    universe_verdict: String,
    trace_sha256: String,
    support_status: String,
    support_sha256: String,
    zero_support_verdict: String,
    zero_support_metrics: String,
    region_kind: String,
    region_lower: f64,
    region_upper: f64,
    gate_decision: String,
    gate_value_state: String,
    gate_risk_state: String,
    value_floor: f64,
    risk_cap: f64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn rows(input: &str) -> Vec<Vec<&str>> {
    input.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split('\t').collect())
        .collect()
}

fn universe_value(input: &str, key: &str) -> Result<String, String> {
    let parsed = rows(input);
    let matches: Vec<&Vec<&str>> = parsed.iter()
        .filter(|row| row.len() == 4 && row[0] == "SUMMARY" && row[1] == "-" && row[2] == key)
        .collect();
    if matches.len() != 1 {
        return Err(invalid(format!("universe report must contain exactly one SUMMARY/-/{key} field")));
    }
    Ok(matches[0][3].to_owned())
}

fn support_value(input: &str, key: &str) -> Result<String, String> {
    let parsed = rows(input);
    let matches: Vec<&Vec<&str>> = parsed.iter()
        .filter(|row| row.len() == 3 && row[0] == "SUMMARY" && row[1] == key)
        .collect();
    if matches.len() != 1 {
        return Err(invalid(format!("support report must contain exactly one SUMMARY/{key} field")));
    }
    Ok(matches[0][2].to_owned())
}

fn gate_value(input: &str, key: &str) -> Result<String, String> {
    let parsed = rows(input);
    let matches: Vec<&Vec<&str>> = parsed.iter()
        .filter(|row| row.len() == 3 && row[0] == "SUMMARY" && row[1] == key)
        .collect();
    if matches.len() != 1 {
        return Err(invalid(format!("gate report must contain exactly one SUMMARY/{key} field")));
    }
    Ok(matches[0][2].to_owned())
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == TRACE_DIGEST_LEN && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_unit_interval(value: &str, name: &str) -> Result<f64, String> {
    let parsed = value.parse::<f64>().map_err(|_| invalid(format!("{name} is not numeric")))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err(invalid(format!("{name} must be finite and in [0,1]")));
    }
    Ok(parsed)
}

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

fn require_ope_scenario(input: &str) -> Result<(String, String), String> {
    let parsed = rows(input);
    let matches: Vec<&Vec<&str>> = parsed.iter()
        .filter(|row| {
            row.len() == 6
                && row[0] == "SCENARIO"
                && row[1] == ZERO_SUPPORT_SCENARIO
                && row[2] == "-"
                && row[4] == "estimator_metrics"
        })
        .collect();
    if matches.len() != 1 {
        return Err(invalid("OPE report must contain exactly one zero_support estimator_metrics record"));
    }
    Ok((matches[0][3].to_owned(), matches[0][5].to_owned()))
}

fn require_ope_region(input: &str, field: &str) -> Result<(String, f64), String> {
    let parsed = rows(input);
    let matches: Vec<&Vec<&str>> = parsed.iter()
        .filter(|row| {
            row.len() == 6
                && row[0] == "IDENTIFICATION_REGION"
                && row[1] == ZERO_SUPPORT_SCENARIO
                && row[2] == "-"
                && row[4] == field
        })
        .collect();
    if matches.len() != 1 {
        return Err(invalid(format!("OPE report must contain exactly one zero_support {field} record")));
    }
    let kind = matches[0][3].to_owned();
    let value = parse_unit_interval(matches[0][5], field)?;
    Ok((kind, value))
}

fn has_ope_limit(input: &str, field: &str, expected: &str) -> bool {
    rows(input).iter().any(|row| {
        row.len() == 6
            && row[0] == "LIMIT"
            && row[1] == "-"
            && row[2] == "-"
            && row[3] == "ALL_SCENARIOS"
            && row[4] == field
            && row[5] == expected
    })
}

fn assess(universe: &str, support: &str, ope: &str, gate: &str) -> Result<BundleReport, String> {
    let universe_verdict = universe_value(universe, "verdict")?;
    let trace_sha256 = universe_value(universe, UNIVERSE_TRACE_FIELD)?;
    let support_status = support_value(support, "status")?;
    let support_sha256 = support_value(support, SUPPORT_TRACE_FIELD)?;
    let (zero_support_verdict, zero_support_metrics) = require_ope_scenario(ope)?;
    let (region_kind_lower, region_lower) = require_ope_region(ope, "lower_bound")?;
    let (region_kind_upper, region_upper) = require_ope_region(ope, "upper_bound")?;
    let gate_decision = gate_value(gate, "decision")?;
    let gate_value_state = gate_value(gate, "value_state")?;
    let gate_risk_state = gate_value(gate, "risk_state")?;
    let value_floor = parse_unit_interval(&gate_value(gate, "value_floor")?, "value_floor")?;
    let risk_cap = parse_unit_interval(&gate_value(gate, "risk_cap")?, "risk_cap")?;

    if !is_sha256_hex(&trace_sha256) || !is_sha256_hex(&support_sha256) {
        return Err(invalid("component trace digests must be 64 hexadecimal characters"));
    }

    let universe_complete = universe_verdict == "COMPLETE_ACTION_UNIVERSE";
    let trace_identity_matches = trace_sha256.eq_ignore_ascii_case(&support_sha256);
    let weak_overlap_remains_diagnostic = support_status == "SUPPORTED_BUT_WEAK_OVERLAP";
    let zero_support_is_not_identified = zero_support_verdict == "NOT_IDENTIFIED";
    let zero_support_metrics_are_suppressed = zero_support_metrics == "SUPPRESSED";
    let region_is_partial_and_coherent = region_kind_lower == "PARTIALLY_IDENTIFIED"
        && region_kind_upper == "PARTIALLY_IDENTIFIED"
        && region_lower <= region_upper
        && approx_eq(region_lower, 0.1)
        && approx_eq(region_upper, 0.6);
    let ope_does_not_claim_confidence_intervals = has_ope_limit(ope, "confidence_intervals", "NOT_COMPUTED");
    let ope_does_not_claim_production_authorization = has_ope_limit(ope, "production_authorization", "NOT_CLAIMED");
    let gate_preserves_uncertainty = gate_decision == "INSUFFICIENT_EVIDENCE"
        && gate_value_state == "UNRESOLVED"
        && gate_risk_state == "CLEARED"
        && approx_eq(value_floor, 0.5)
        && approx_eq(risk_cap, 0.05);
    let authorization_is_prohibited = gate_decision != "AUTHORIZED";

    let checks = vec![
        ("UNIVERSE_COMPLETE", universe_complete),
        ("TRACE_DIGESTS_MATCH", trace_identity_matches),
        ("WEAK_OVERLAP_NOT_UPGRADED", weak_overlap_remains_diagnostic),
        ("ZERO_SUPPORT_NOT_IDENTIFIED", zero_support_is_not_identified),
        ("ZERO_SUPPORT_METRICS_SUPPRESSED", zero_support_metrics_are_suppressed),
        ("PARTIAL_REGION_PRESERVED", region_is_partial_and_coherent),
        ("NO_CONFIDENCE_INTERVAL_CLAIM", ope_does_not_claim_confidence_intervals),
        ("NO_PRODUCTION_AUTHORIZATION_CLAIM", ope_does_not_claim_production_authorization),
        ("DECISION_GATE_PRESERVES_UNCERTAINTY", gate_preserves_uncertainty),
        ("AUTHORIZATION_PROHIBITED", authorization_is_prohibited),
    ];
    let status = if checks.iter().all(|(_, ok)| *ok) { "DIAGNOSTIC_ONLY" } else { "BLOCKED" };

    Ok(BundleReport {
        status,
        checks,
        universe_verdict,
        trace_sha256,
        support_status,
        support_sha256,
        zero_support_verdict,
        zero_support_metrics,
        region_kind: region_kind_lower,
        region_lower,
        region_upper,
        gate_decision,
        gate_value_state,
        gate_risk_state,
        value_floor,
        risk_cap,
    })
}

fn render(report: &BundleReport) -> String {
    let mut out = String::from("record_type\tfield\tvalue\n");
    let _ = writeln!(out, "SUMMARY\tstatus\t{}", report.status);
    let _ = writeln!(out, "SUMMARY\tuniverse_verdict\t{}", report.universe_verdict);
    let _ = writeln!(out, "SUMMARY\tuniverse_trace_sha256\t{}", report.trace_sha256);
    let _ = writeln!(out, "SUMMARY\tsupport_status\t{}", report.support_status);
    let _ = writeln!(out, "SUMMARY\tsupport_input_sha256\t{}", report.support_sha256);
    let _ = writeln!(out, "SUMMARY\tzero_support_verdict\t{}", report.zero_support_verdict);
    let _ = writeln!(out, "SUMMARY\tzero_support_estimator_metrics\t{}", report.zero_support_metrics);
    let _ = writeln!(out, "SUMMARY\tidentification_region_kind\t{}", report.region_kind);
    let _ = writeln!(out, "SUMMARY\tidentification_region_lower\t{:.12}", report.region_lower);
    let _ = writeln!(out, "SUMMARY\tidentification_region_upper\t{:.12}", report.region_upper);
    let _ = writeln!(out, "SUMMARY\tgate_decision\t{}", report.gate_decision);
    let _ = writeln!(out, "SUMMARY\tgate_value_state\t{}", report.gate_value_state);
    let _ = writeln!(out, "SUMMARY\tgate_risk_state\t{}", report.gate_risk_state);
    let _ = writeln!(out, "SUMMARY\tvalue_floor\t{:.12}", report.value_floor);
    let _ = writeln!(out, "SUMMARY\trisk_cap\t{:.12}", report.risk_cap);
    for (name, passed) in &report.checks {
        let _ = writeln!(out, "CHECK\t{name}\t{}", if *passed { "PASS" } else { "FAIL" });
    }
    out.push_str("PROHIBITION\tautomatic_policy_authorization\ttrue\n");
    out.push_str("LIMIT\ttrace_digest_equality_proves_same_bytes_not_manifest_authenticity\ttrue\n");
    out.push_str("LIMIT\tweak_overlap_is_not_statistical_approval\ttrue\n");
    out.push_str("LIMIT\tidentification_region_is_not_a_confidence_interval\ttrue\n");
    out.push_str("LIMIT\tbundle_does_not_claim_real_trace_or_production_validity\ttrue\n");
    out
}

fn run(args: &[String]) -> Result<String, String> {
    let [_program, flag, universe_path, support_path, ope_path, gate_path] = args else {
        return Err("usage: vision-dvpc-evidence-bundle --audit UNIVERSE.tsv SUPPORT.tsv OPE.tsv GATE.tsv".into());
    };
    if flag != "--audit" {
        return Err("usage: vision-dvpc-evidence-bundle --audit UNIVERSE.tsv SUPPORT.tsv OPE.tsv GATE.tsv".into());
    }
    let read = |path: &str| fs::read_to_string(path).map_err(|error| invalid(format!("cannot read {path}: {error}")));
    let universe = read(universe_path)?;
    let support = read(support_path)?;
    let ope = read(ope_path)?;
    let gate = read(gate_path)?;
    Ok(render(&assess(&universe, &support, &ope, &gate)?))
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-033 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const UNIVERSE_OK: &str = "record_type\tcontext_id\tfield\tvalue\nSUMMARY\t-\tverdict\tCOMPLETE_ACTION_UNIVERSE\nSUMMARY\t-\ttrace_sha256\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";
    const SUPPORT_WEAK: &str = "VIS_DVPC_027_AUDIT\tformat\tv1\nSUMMARY\tstatus\tSUPPORTED_BUT_WEAK_OVERLAP\nSUMMARY\tinput_sha256\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\nLIMIT\tno confidence interval or authorization claim is computed\n";
    const OPE_ZERO: &str = "record_type\tscenario_id\testimator\tverdict\tmetric\tvalue\nSCENARIO\tzero_support\t-\tNOT_IDENTIFIED\testimator_metrics\tSUPPRESSED\nIDENTIFICATION_REGION\tzero_support\t-\tPARTIALLY_IDENTIFIED\tlower_bound\t0.100000000000\nIDENTIFICATION_REGION\tzero_support\t-\tPARTIALLY_IDENTIFIED\tupper_bound\t0.600000000000\nLIMIT\t-\t-\tALL_SCENARIOS\tconfidence_intervals\tNOT_COMPUTED\nLIMIT\t-\t-\tALL_SCENARIOS\tproduction_authorization\tNOT_CLAIMED\n";
    const GATE_UNRESOLVED: &str = "record_type\tfield\tvalue\nSUMMARY\tdecision\tINSUFFICIENT_EVIDENCE\nSUMMARY\tvalue_state\tUNRESOLVED\nSUMMARY\trisk_state\tCLEARED\nSUMMARY\tvalue_floor\t0.500000000000\nSUMMARY\trisk_cap\t0.050000000000\n";

    fn evaluate(universe: &str, support: &str, ope: &str, gate: &str) -> BundleReport {
        assess(universe, support, ope, gate).unwrap()
    }

    #[test]
    fn exact_digest_bound_composition_stays_diagnostic_only() {
        let report = evaluate(UNIVERSE_OK, SUPPORT_WEAK, OPE_ZERO, GATE_UNRESOLVED);
        assert_eq!(report.status, "DIAGNOSTIC_ONLY");
        assert_eq!(report.trace_sha256, DIGEST);
        assert!(report.checks.iter().all(|(_, passed)| *passed));
        let rendered = render(&report);
        assert!(rendered.contains("PROHIBITION\tautomatic_policy_authorization\ttrue"));
    }

    #[test]
    fn mismatched_trace_digest_blocks_bundle() {
        let changed = SUPPORT_WEAK.replace(DIGEST, &"b".repeat(64));
        let report = evaluate(UNIVERSE_OK, &changed, OPE_ZERO, GATE_UNRESOLVED);
        assert_eq!(report.status, "BLOCKED");
        assert!(report.checks.iter().any(|(name, passed)| *name == "TRACE_DIGESTS_MATCH" && !passed));
    }

    #[test]
    fn missing_evidence_field_fails_closed() {
        let missing = UNIVERSE_OK.replace("SUMMARY\t-\ttrace_sha256\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n", "");
        assert!(assess(&missing, SUPPORT_WEAK, OPE_ZERO, GATE_UNRESOLVED).is_err());
    }

    #[test]
    fn incomplete_universe_cannot_be_upgraded_to_diagnostic_bundle() {
        let changed = UNIVERSE_OK.replace("COMPLETE_ACTION_UNIVERSE", "INCOMPLETE_ACTION_UNIVERSE");
        assert_eq!(evaluate(&changed, SUPPORT_WEAK, OPE_ZERO, GATE_UNRESOLVED).status, "BLOCKED");
    }

    #[test]
    fn zero_support_metrics_cannot_be_reintroduced() {
        let changed = OPE_ZERO.replace("SUPPRESSED", "0.500000000000");
        let report = evaluate(UNIVERSE_OK, SUPPORT_WEAK, &changed, GATE_UNRESOLVED);
        assert_eq!(report.status, "BLOCKED");
        assert!(report.checks.iter().any(|(name, passed)| *name == "ZERO_SUPPORT_METRICS_SUPPRESSED" && !passed));
    }

    #[test]
    fn authorization_shaped_gate_output_is_blocked() {
        let changed = GATE_UNRESOLVED.replace("INSUFFICIENT_EVIDENCE", "AUTHORIZED")
            .replace("UNRESOLVED", "CLEARED");
        assert_eq!(evaluate(UNIVERSE_OK, SUPPORT_WEAK, OPE_ZERO, &changed).status, "BLOCKED");
    }

    #[test]
    fn support_status_cannot_be_treated_as_approval() {
        let changed = SUPPORT_WEAK.replace("SUPPORTED_BUT_WEAK_OVERLAP", "SUPPORTED_DIAGNOSTIC_ONLY");
        assert_eq!(evaluate(UNIVERSE_OK, &changed, OPE_ZERO, GATE_UNRESOLVED).status, "BLOCKED");
    }

    #[test]
    fn identification_region_cannot_be_relabelled_as_a_confidence_interval() {
        let changed = OPE_ZERO.replace("PARTIALLY_IDENTIFIED", "CONFIDENCE_INTERVAL");
        assert_eq!(evaluate(UNIVERSE_OK, SUPPORT_WEAK, &changed, GATE_UNRESOLVED).status, "BLOCKED");
    }

    #[test]
    fn malformed_digest_fails_closed() {
        let changed = UNIVERSE_OK.replace(DIGEST, "not-a-digest");
        assert!(assess(&changed, SUPPORT_WEAK, OPE_ZERO, GATE_UNRESOLVED).is_err());
    }
}

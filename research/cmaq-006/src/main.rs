use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::process;

const REPORT_SCHEMA: &str = "cmaq-006-decision-gate-v0.1";
// Reviewed policy constant: untrusted episode input cannot lower the model-use bar.
const MINIMUM_NET_LIFT: f64 = 0.02;
const INPUT_HEADER: [&str; 11] = [
    "case_id",
    "model_digest",
    "adequacy_model_digest",
    "utility_model_digest",
    "model_epoch",
    "current_epoch",
    "shift_detected",
    "adequacy_status",
    "net_lift_ci95_low",
    "model_action_authorized",
    "direct_action_authorized",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdequacyStatus {
    Pass,
    Fail,
    Unknown,
}

impl AdequacyStatus {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "PASS" => Ok(Self::Pass),
            "FAIL" => Ok(Self::Fail),
            "UNKNOWN" => Ok(Self::Unknown),
            _ => Err(format!("invalid adequacy_status {value:?}; expected PASS, FAIL, or UNKNOWN")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Decision {
    UseModel,
    DirectFallback,
    Block,
}

impl Decision {
    fn as_str(self) -> &'static str {
        match self {
            Self::UseModel => "USE_MODEL",
            Self::DirectFallback => "DIRECT_FALLBACK",
            Self::Block => "BLOCK",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct GateInput {
    case_id: String,
    model_digest: String,
    adequacy_model_digest: String,
    utility_model_digest: String,
    model_epoch: u64,
    current_epoch: u64,
    shift_detected: bool,
    adequacy_status: AdequacyStatus,
    net_lift_ci95_low: f64,
    model_action_authorized: bool,
    direct_action_authorized: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GateResult {
    case_id: String,
    decision: Decision,
    reason: &'static str,
}

fn token(value: &str, field: &str) -> Result<String, String> {
    if value.is_empty() || value.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(format!("{field} must be a non-empty token without whitespace"));
    }
    Ok(value.to_owned())
}

fn parse_bool(value: &str, field: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("{field} must be exactly true or false")),
    }
}

fn parse_epoch(value: &str, field: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("{field} must be a non-negative integer"))
}

fn parse_finite(value: &str, field: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|_| format!("{field} must be a finite number"))?;
    if !parsed.is_finite() {
        return Err(format!("{field} must be a finite number"));
    }
    Ok(parsed)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn parse_row(fields: &[&str]) -> Result<GateInput, String> {
    if fields.len() != INPUT_HEADER.len() {
        return Err(format!(
            "row has {} fields; expected {}",
            fields.len(),
            INPUT_HEADER.len()
        ));
    }
    let case_id = token(fields[0], "case_id")?;
    let model_digest = token(fields[1], "model_digest")?;
    let adequacy_model_digest = token(fields[2], "adequacy_model_digest")?;
    let utility_model_digest = token(fields[3], "utility_model_digest")?;
    let adequacy_status = AdequacyStatus::parse(fields[7])?;
    let net_lift_ci95_low = parse_finite(fields[8], "net_lift_ci95_low")?;

    Ok(GateInput {
        case_id,
        model_digest,
        adequacy_model_digest,
        utility_model_digest,
        model_epoch: parse_epoch(fields[4], "model_epoch")?,
        current_epoch: parse_epoch(fields[5], "current_epoch")?,
        shift_detected: parse_bool(fields[6], "shift_detected")?,
        adequacy_status,
        net_lift_ci95_low,
        model_action_authorized: parse_bool(fields[9], "model_action_authorized")?,
        direct_action_authorized: parse_bool(fields[10], "direct_action_authorized")?,
    })
}

fn parse_input(input: &str) -> Result<Vec<GateInput>, String> {
    let mut lines = input.lines().enumerate().filter(|(_, line)| !line.trim().is_empty());
    let (_, header) = lines.next().ok_or("empty decision-gate input")?;
    let observed_header = header.split('\t').collect::<Vec<_>>();
    if observed_header.as_slice() != INPUT_HEADER.as_slice() {
        return Err(format!(
            "input header mismatch; expected {}",
            INPUT_HEADER.join("\t")
        ));
    }

    let mut cases = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in lines {
        let fields = line.split('\t').collect::<Vec<_>>();
        let case = parse_row(&fields)
            .map_err(|error| format!("input line {}: {error}", line_index + 1))?;
        if !ids.insert(case.case_id.clone()) {
            return Err(format!("duplicate case_id {:?}", case.case_id));
        }
        cases.push(case);
    }
    if cases.is_empty() {
        return Err("decision-gate input has no cases".to_owned());
    }
    Ok(cases)
}

fn direct_fallback_or_block(input: &GateInput, reason: &'static str) -> GateResult {
    let decision = if input.direct_action_authorized {
        Decision::DirectFallback
    } else {
        Decision::Block
    };
    let reason = if decision == Decision::Block {
        "direct_action_not_authorized"
    } else {
        reason
    };
    GateResult {
        case_id: input.case_id.clone(),
        decision,
        reason,
    }
}

fn assess(input: &GateInput) -> GateResult {
    // Independent action authorization has priority over every model-use signal.
    if !input.direct_action_authorized {
        return GateResult {
            case_id: input.case_id.clone(),
            decision: Decision::Block,
            reason: "direct_action_not_authorized",
        };
    }
    if input.shift_detected {
        return direct_fallback_or_block(input, "shift_detected");
    }
    if input.model_epoch != input.current_epoch {
        return direct_fallback_or_block(input, "stale_model_epoch");
    }
    if !input.model_action_authorized {
        return direct_fallback_or_block(input, "model_action_not_authorized");
    }
    if !valid_sha256(&input.model_digest) {
        return direct_fallback_or_block(input, "invalid_model_digest");
    }
    if !valid_sha256(&input.adequacy_model_digest)
        || !valid_sha256(&input.utility_model_digest)
    {
        return direct_fallback_or_block(input, "invalid_evidence_digest");
    }
    if input.model_digest != input.adequacy_model_digest {
        return direct_fallback_or_block(input, "predictive_evidence_digest_mismatch");
    }
    if input.model_digest != input.utility_model_digest {
        return direct_fallback_or_block(input, "utility_evidence_digest_mismatch");
    }
    if input.adequacy_status != AdequacyStatus::Pass {
        return direct_fallback_or_block(input, "predictive_adequacy_not_pass");
    }
    if input.net_lift_ci95_low <= MINIMUM_NET_LIFT {
        return direct_fallback_or_block(input, "insufficient_net_lift_lower_bound");
    }

    GateResult {
        case_id: input.case_id.clone(),
        decision: Decision::UseModel,
        reason: "all_model_use_gates_passed",
    }
}

fn render(input: &GateInput, result: &GateResult) -> String {
    format!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}",
        REPORT_SCHEMA,
        input.case_id,
        input.model_digest,
        input.adequacy_model_digest,
        input.utility_model_digest,
        input.model_epoch,
        input.current_epoch,
        input.shift_detected,
        input.adequacy_status.as_str(),
        result.decision.as_str(),
        input.net_lift_ci95_low,
        MINIMUM_NET_LIFT,
        input.model_action_authorized,
        input.direct_action_authorized,
        result.reason
    )
}

fn usage() {
    println!("cmaq-006 --input FILE");
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let Some(flag) = args.next() else {
        return Err("missing --input FILE".to_owned());
    };
    if flag == "--help" || flag == "-h" {
        usage();
        return Ok(());
    }
    if flag != "--input" {
        return Err(format!("unknown argument {flag:?}; expected --input FILE"));
    }
    let path = args.next().ok_or("--input requires a file path")?;
    if args.next().is_some() {
        return Err("unexpected extra argument".to_owned());
    }
    let input_text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {path:?}: {error}"))?;
    let cases = parse_input(&input_text)?;
    println!("schema_version\tcase_id\tmodel_digest\tadequacy_model_digest\tutility_model_digest\tmodel_epoch\tcurrent_epoch\tshift_detected\tadequacy_status\tdecision\tnet_lift_ci95_low\tminimum_net_lift\tmodel_action_authorized\tdirect_action_authorized\treason");
    for case in cases {
        let result = assess(&case);
        println!("{}", render(&case, &result));
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("cmaq-006: {error}");
        process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn valid_input() -> GateInput {
        GateInput {
            case_id: "episode_001".to_owned(),
            model_digest: DIGEST.to_owned(),
            adequacy_model_digest: DIGEST.to_owned(),
            utility_model_digest: DIGEST.to_owned(),
            model_epoch: 7,
            current_epoch: 7,
            shift_detected: false,
            adequacy_status: AdequacyStatus::Pass,
            net_lift_ci95_low: 0.12,
            model_action_authorized: true,
            direct_action_authorized: true,
        }
    }

    #[test]
    fn permits_model_only_when_every_evidence_gate_passes() {
        let result = assess(&valid_input());
        assert_eq!(result.decision, Decision::UseModel);
        assert_eq!(result.reason, "all_model_use_gates_passed");
    }

    #[test]
    fn shift_alarm_forces_direct_fallback() {
        let mut input = valid_input();
        input.shift_detected = true;
        assert_eq!(assess(&input).decision, Decision::DirectFallback);
        assert_eq!(assess(&input).reason, "shift_detected");
    }

    #[test]
    fn stale_epoch_forces_direct_fallback() {
        let mut input = valid_input();
        input.model_epoch = 6;
        assert_eq!(assess(&input).reason, "stale_model_epoch");
    }

    #[test]
    fn digest_mismatch_forces_direct_fallback() {
        let mut input = valid_input();
        input.adequacy_model_digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
        assert_eq!(assess(&input).reason, "predictive_evidence_digest_mismatch");
    }

    #[test]
    fn invalid_digest_forces_direct_fallback() {
        let mut input = valid_input();
        input.model_digest = "not-a-digest".to_owned();
        assert_eq!(assess(&input).reason, "invalid_model_digest");
    }

    #[test]
    fn unknown_or_failed_adequacy_forces_direct_fallback() {
        for status in [AdequacyStatus::Unknown, AdequacyStatus::Fail] {
            let mut input = valid_input();
            input.adequacy_status = status;
            assert_eq!(assess(&input).reason, "predictive_adequacy_not_pass");
        }
    }

    #[test]
    fn lower_bound_must_strictly_exceed_minimum_net_lift() {
        for bound in [0.02, 0.0, -0.10] {
            let mut input = valid_input();
            input.net_lift_ci95_low = bound;
            assert_eq!(assess(&input).reason, "insufficient_net_lift_lower_bound");
        }
    }

    #[test]
    fn unauthorized_model_action_falls_back_if_direct_is_authorized() {
        let mut input = valid_input();
        input.model_action_authorized = false;
        assert_eq!(assess(&input).decision, Decision::DirectFallback);
        assert_eq!(assess(&input).reason, "model_action_not_authorized");
    }

    #[test]
    fn no_authorized_direct_action_blocks_even_if_model_claims_to_be_safe() {
        let mut input = valid_input();
        input.direct_action_authorized = false;
        assert_eq!(assess(&input).decision, Decision::Block);
        assert_eq!(assess(&input).reason, "direct_action_not_authorized");
    }

    #[test]
    fn invalid_evidence_digest_forces_direct_fallback() {
        let mut input = valid_input();
        input.utility_model_digest = "g".repeat(64);
        assert_eq!(assess(&input).reason, "invalid_evidence_digest");
    }

    #[test]
    fn utility_evidence_must_refer_to_the_same_model() {
        let mut input = valid_input();
        input.utility_model_digest = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".to_owned();
        assert_eq!(assess(&input).reason, "utility_evidence_digest_mismatch");
    }

    #[test]
    fn parser_rejects_duplicate_case_ids_and_malformed_rows() {
        let header = INPUT_HEADER.join("\t");
        let good = "episode_001\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\t7\t7\tfalse\tPASS\t0.12\t0.02\ttrue\ttrue";
        let duplicate = format!("{header}\n{good}\n{good}\n");
        assert!(parse_input(&duplicate).unwrap_err().contains("duplicate case_id"));
        assert!(parse_input(&format!("{header}\ncase_only\n")).unwrap_err().contains("row has"));
    }

    #[test]
    fn parser_rejects_nonfinite_net_lift_and_unrecognized_boolean() {
        let header = INPUT_HEADER.join("\t");
        let prefix = "episode_001\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\t7\t7\tfalse\tPASS";
        let nonfinite = format!("{header}\n{prefix}\tNaN\ttrue\ttrue\n");
        assert!(parse_input(&nonfinite).unwrap_err().contains("finite number"));
        let malformed_bool = format!("{header}\n{prefix}\t0.12\tyes\ttrue\n");
        assert!(parse_input(&malformed_bool).unwrap_err().contains("exactly true or false"));
    }

    #[test]
    fn output_is_deterministic_and_binds_reason_to_case_id() {
        let input = valid_input();
        let result = assess(&input);
        assert_eq!(render(&input, &result), render(&input, &result));
        assert!(render(&input, &result).starts_with(&format!("{REPORT_SCHEMA}\tepisode_001\t")));
    }
}

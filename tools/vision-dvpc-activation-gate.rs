use std::env;
use std::fmt::Write as _;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-022-campaigns.tsv");
const PPM: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Authorized,
    Denied,
    InsufficientEvidence,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Authorized => "AUTHORIZED",
            Self::Denied => "DENIED",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
        }
    }

    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "AUTHORIZED" => Ok(Self::Authorized),
            "DENIED" => Ok(Self::Denied),
            "INSUFFICIENT_EVIDENCE" => Ok(Self::InsufficientEvidence),
            _ => Err(format!("unknown expected verdict {value:?}")),
        }
    }
}

#[derive(Clone, Debug)]
struct Campaign {
    id: String,
    trials: usize,
    failures: usize,
    alpha_ppm: u64,
    risk_cap_ppm: u64,
    net_value_lcb_milli: i64,
    min_net_value_lcb_milli: i64,
    activation_count: u64,
    eligible_count: u64,
    min_activation_bps: u64,
    provenance_ok: bool,
    action_digest_ok: bool,
    target_calibrated: bool,
    target_shift_detected: bool,
    sealed_touched: bool,
    all_in_cost_units: u64,
    cost_budget_units: u64,
    required_campaign_units: u64,
    available_campaign_units: u64,
    expected: Verdict,
}

fn parse_num<T: std::str::FromStr>(value: &str, field: &str, line: usize) -> Result<T, String> {
    value
        .parse::<T>()
        .map_err(|_| format!("line {line}: invalid {field}: {value:?}"))
}

fn parse_flag(value: &str, field: &str, line: usize) -> Result<bool, String> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("line {line}: {field} must be 0 or 1")),
    }
}

fn parse_campaign(line_text: &str, line: usize) -> Result<Campaign, String> {
    let f: Vec<&str> = line_text.split('\t').collect();
    if f.len() != 20 {
        return Err(format!("line {line}: expected 20 TSV fields, found {}", f.len()));
    }
    if f[0].is_empty() || f[0].chars().any(char::is_whitespace) {
        return Err(format!("line {line}: invalid campaign ID"));
    }
    let campaign = Campaign {
        id: f[0].to_string(),
        trials: parse_num(f[1], "trials", line)?,
        failures: parse_num(f[2], "failures", line)?,
        alpha_ppm: parse_num(f[3], "alpha_ppm", line)?,
        risk_cap_ppm: parse_num(f[4], "risk_cap_ppm", line)?,
        net_value_lcb_milli: parse_num(f[5], "net_value_lcb_milli", line)?,
        min_net_value_lcb_milli: parse_num(f[6], "min_net_value_lcb_milli", line)?,
        activation_count: parse_num(f[7], "activation_count", line)?,
        eligible_count: parse_num(f[8], "eligible_count", line)?,
        min_activation_bps: parse_num(f[9], "min_activation_bps", line)?,
        provenance_ok: parse_flag(f[10], "provenance_ok", line)?,
        action_digest_ok: parse_flag(f[11], "action_digest_ok", line)?,
        target_calibrated: parse_flag(f[12], "target_calibrated", line)?,
        target_shift_detected: parse_flag(f[13], "target_shift_detected", line)?,
        sealed_touched: parse_flag(f[14], "sealed_touched", line)?,
        all_in_cost_units: parse_num(f[15], "all_in_cost_units", line)?,
        cost_budget_units: parse_num(f[16], "cost_budget_units", line)?,
        required_campaign_units: parse_num(f[17], "required_campaign_units", line)?,
        available_campaign_units: parse_num(f[18], "available_campaign_units", line)?,
        expected: Verdict::parse(f[19])?,
    };
    validate_campaign(&campaign)?;
    Ok(campaign)
}

fn validate_campaign(c: &Campaign) -> Result<(), String> {
    if c.trials == 0 || c.failures > c.trials {
        return Err(format!("{}: require trials > 0 and failures <= trials", c.id));
    }
    if c.alpha_ppm == 0 || c.alpha_ppm >= 1_000_000 {
        return Err(format!("{}: alpha must be strictly between 0 and 1", c.id));
    }
    if c.risk_cap_ppm == 0 || c.risk_cap_ppm >= 1_000_000 {
        return Err(format!("{}: risk cap must be strictly between 0 and 1", c.id));
    }
    if c.eligible_count == 0 || c.activation_count > c.eligible_count {
        return Err(format!("{}: activation counts are invalid", c.id));
    }
    if c.min_activation_bps > 10_000 {
        return Err(format!("{}: minimum activation rate exceeds 100%", c.id));
    }
    Ok(())
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n + 1];
    for i in 1..=n {
        out[i] = out[i - 1] + (i as f64).ln();
    }
    out
}

fn binomial_cdf(n: usize, k: usize, p: f64, log_fact: &[f64]) -> f64 {
    if k >= n || p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let log_p = p.ln();
    let log_q = (-p).ln_1p();
    let logs: Vec<f64> = (0..=k)
        .map(|i| {
            log_fact[n] - log_fact[i] - log_fact[n - i]
                + (i as f64) * log_p
                + ((n - i) as f64) * log_q
        })
        .collect();
    let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum: f64 = logs.iter().map(|x| (x - max_log).exp()).sum();
    (max_log.exp() * scaled_sum).clamp(0.0, 1.0)
}

/// One-sided Clopper-Pearson upper confidence bound for a Bernoulli failure rate.
/// Uses the exact binomial tail equation P_p[X <= failures] = alpha, solved by bisection.
fn exact_binomial_upper(n: usize, failures: usize, alpha: f64) -> Result<f64, String> {
    if n == 0 || failures > n || !(0.0 < alpha && alpha < 1.0) {
        return Err("require n > 0, failures <= n, and 0 < alpha < 1".to_string());
    }
    if failures == n {
        return Ok(1.0);
    }
    if failures == 0 {
        return Ok(1.0 - alpha.powf(1.0 / n as f64));
    }
    let log_fact = log_factorials(n);
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..80 {
        let mid = (low + high) / 2.0;
        if binomial_cdf(n, failures, mid, &log_fact) > alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

fn required_zero_failure_trials(alpha: f64, max_risk: f64) -> Result<usize, String> {
    if !(0.0 < alpha && alpha < 1.0 && 0.0 < max_risk && max_risk < 1.0) {
        return Err("require 0 < alpha < 1 and 0 < max_risk < 1".to_string());
    }
    let n = (alpha.ln() / (1.0 - max_risk).ln()).ceil();
    if !n.is_finite() || n < 1.0 || n > usize::MAX as f64 {
        return Err("required trial count is out of range".to_string());
    }
    Ok(n as usize)
}

fn evaluate(c: &Campaign) -> Result<(Verdict, f64), String> {
    let alpha = c.alpha_ppm as f64 / PPM;
    let risk_cap = c.risk_cap_ppm as f64 / PPM;
    let risk_upper = exact_binomial_upper(c.trials, c.failures, alpha)?;

    // Invalid or non-transportable evidence is a hard denial, never a statistical pass.
    if !c.provenance_ok
        || !c.action_digest_ok
        || !c.target_calibrated
        || c.target_shift_detected
        || c.sealed_touched
    {
        return Ok((Verdict::Denied, risk_upper));
    }
    if c.all_in_cost_units > c.cost_budget_units
        || c.required_campaign_units > c.available_campaign_units
        || c.net_value_lcb_milli < c.min_net_value_lcb_milli
    {
        return Ok((Verdict::Denied, risk_upper));
    }
    let activated_bps_lhs = (c.activation_count as u128) * 10_000;
    let minimum_bps_rhs = (c.min_activation_bps as u128) * (c.eligible_count as u128);
    if activated_bps_lhs < minimum_bps_rhs {
        return Ok((Verdict::Denied, risk_upper));
    }
    if risk_upper > risk_cap + 1e-12 {
        return Ok((Verdict::InsufficientEvidence, risk_upper));
    }
    Ok((Verdict::Authorized, risk_upper))
}

fn parsed_fixture() -> Result<Vec<Campaign>, String> {
    let mut campaigns = Vec::new();
    for (idx, raw) in FIXTURE.lines().enumerate() {
        if idx == 0 || raw.trim().is_empty() {
            continue;
        }
        campaigns.push(parse_campaign(raw, idx + 1)?);
    }
    if campaigns.is_empty() {
        return Err("fixture contains no campaign rows".to_string());
    }
    Ok(campaigns)
}

fn score_fixture() -> Result<String, String> {
    let campaigns = parsed_fixture()?;
    let mut output = String::from("case_id\tverdict\trisk_upper_bound\trisk_cap\texpected\n");
    let mut authorized = 0usize;
    let mut denied = 0usize;
    let mut insufficient = 0usize;
    for campaign in &campaigns {
        let (actual, risk_upper) = evaluate(campaign)?;
        if actual != campaign.expected {
            return Err(format!(
                "{}: expected {}, got {} (risk upper {:.9})",
                campaign.id,
                campaign.expected.as_str(),
                actual.as_str(),
                risk_upper
            ));
        }
        match actual {
            Verdict::Authorized => authorized += 1,
            Verdict::Denied => denied += 1,
            Verdict::InsufficientEvidence => insufficient += 1,
        }
        let _ = writeln!(
            output,
            "{}\t{}\t{:.9}\t{:.6}\t{}",
            campaign.id,
            actual.as_str(),
            risk_upper,
            campaign.risk_cap_ppm as f64 / PPM,
            campaign.expected.as_str()
        );
    }
    let _ = writeln!(
        output,
        "SUMMARY\tcampaigns={}\tauthorized={}\tdenied={}\tinsufficient_evidence={}",
        campaigns.len(), authorized, denied, insufficient
    );
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() == 2 && args[1] == "--score" {
        return score_fixture();
    }
    if args.len() == 4 && args[1] == "--zero-failure-required" {
        let alpha_ppm: u64 = parse_num(&args[2], "alpha_ppm", 0)?;
        let risk_ppm: u64 = parse_num(&args[3], "risk_cap_ppm", 0)?;
        let n = required_zero_failure_trials(alpha_ppm as f64 / PPM, risk_ppm as f64 / PPM)?;
        return Ok(format!("required_independent_zero_failure_trials\t{n}\n"));
    }
    Err("usage: vision-dvpc-activation-gate --score | --zero-failure-required ALPHA_PPM RISK_CAP_PPM".into())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-022 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_failure_boundary_is_exact_at_58_and_59_trials() {
        let alpha = 0.05;
        assert!(exact_binomial_upper(58, 0, alpha).unwrap() > 0.05);
        assert!(exact_binomial_upper(59, 0, alpha).unwrap() < 0.05);
        assert_eq!(required_zero_failure_trials(alpha, 0.05).unwrap(), 59);
    }

    #[test]
    fn one_failure_needs_more_than_59_trials_but_100_is_enough_here() {
        assert!(exact_binomial_upper(59, 1, 0.05).unwrap() > 0.05);
        assert!(exact_binomial_upper(100, 1, 0.05).unwrap() < 0.05);
    }

    #[test]
    fn risk_upper_bound_is_monotone_in_failures_and_sample_size() {
        let a = exact_binomial_upper(100, 0, 0.05).unwrap();
        let b = exact_binomial_upper(100, 1, 0.05).unwrap();
        let c = exact_binomial_upper(200, 0, 0.05).unwrap();
        assert!(c < a && a < b);
    }

    #[test]
    fn all_frozen_known_answers_match() {
        let campaigns = parsed_fixture().unwrap();
        assert!(campaigns.len() >= 12);
        for campaign in &campaigns {
            assert_eq!(evaluate(campaign).unwrap().0, campaign.expected, "{}", campaign.id);
        }
    }

    #[test]
    fn tiny_zero_failure_sample_is_not_authorized() {
        let c = parsed_fixture().unwrap().into_iter().find(|x| x.id == "tiny_3_zero").unwrap();
        assert_eq!(evaluate(&c).unwrap().0, Verdict::InsufficientEvidence);
    }

    #[test]
    fn no_op_shift_seal_and_provenance_controls_fail_closed() {
        let campaigns = parsed_fixture().unwrap();
        for id in ["no_op", "source_shift", "sealed_confirmation_touched", "provenance_missing", "digest_mismatch"] {
            let c = campaigns.iter().find(|x| x.id == id).unwrap();
            assert_eq!(evaluate(c).unwrap().0, Verdict::Denied, "{id}");
        }
    }

    #[test]
    fn malformed_failure_count_is_rejected() {
        assert!(exact_binomial_upper(4, 5, 0.05).is_err());
    }

    #[test]
    fn zero_failure_sample_requirement_tightens_with_risk_limit() {
        assert_eq!(required_zero_failure_trials(0.05, 0.05).unwrap(), 59);
        assert_eq!(required_zero_failure_trials(0.05, 0.01).unwrap(), 299);
        assert!(required_zero_failure_trials(0.05, 0.0).is_err());
    }
}

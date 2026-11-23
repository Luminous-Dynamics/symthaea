use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::process::ExitCode;

const HEADER: &str = "schema_version\tscenario_id\tcohort_id\tevidence_class\tevidence_status\tevidence_age_days\tessentials_access_low\tessentials_access_high\tlawful_opportunity_coverage_low\tlawful_opportunity_coverage_high\tfair_pay_adequacy_low\tfair_pay_adequacy_high\tsafe_work_low\tsafe_work_high\twork_stability_low\twork_stability_high\tsocial_protection_coverage_low\tsocial_protection_coverage_high\tlawful_access_time_minutes_low\tlawful_access_time_minutes_high\tcohort_gate_status\tscenario_gate_status";
const SCENARIOS: [&str; 10] = [
    "baseline",
    "high_mean_low_floor",
    "opportunity_without_quality",
    "high_access_friction",
    "threshold_straddle",
    "stale_source_data",
    "conflicted_sources",
    "missing_data",
    "wide_uncertainty",
    "opportunity_gap_hidden_by_mean",
];
const COHORTS: [&str; 5] = [
    "cohort_01",
    "cohort_02",
    "cohort_03",
    "cohort_04",
    "cohort_05",
];
const MIN_ESSENTIALS_ACCESS: f64 = 0.90;
const MIN_LAWFUL_OPPORTUNITY_COVERAGE: f64 = 0.80;
const MIN_FAIR_PAY_ADEQUACY: f64 = 0.80;
const MIN_SAFE_WORK: f64 = 0.90;
const MIN_WORK_STABILITY: f64 = 0.75;
const MIN_SOCIAL_PROTECTION: f64 = 0.75;
const MAX_LAWFUL_ACCESS_TIME_MINUTES: f64 = 60.0;
const MAX_COHORT_COVERAGE_GAP: f64 = 0.15;
const MAX_INTERVAL_WIDTH: f64 = 0.20;
const MAX_TIME_INTERVAL_WIDTH_MINUTES: f64 = 20.0;
const MAX_EVIDENCE_AGE_DAYS: u32 = 180;

#[derive(Clone, Debug)]
struct Range {
    low: Option<f64>,
    high: Option<f64>,
}

impl Range {
    fn parse(low: &str, high: &str, name: &str, line: usize) -> Result<Self, String> {
        if low.is_empty() && high.is_empty() {
            return Ok(Self {
                low: None,
                high: None,
            });
        }
        if low.is_empty() || high.is_empty() {
            return Err(format!("line {line}: half-missing range {name}"));
        }
        let low = low
            .parse::<f64>()
            .map_err(|_| format!("line {line}: invalid {name} lower bound"))?;
        let high = high
            .parse::<f64>()
            .map_err(|_| format!("line {line}: invalid {name} upper bound"))?;
        if !low.is_finite() || !high.is_finite() || low > high {
            return Err(format!("line {line}: invalid {name} interval"));
        }
        Ok(Self {
            low: Some(low),
            high: Some(high),
        })
    }

    fn complete(&self) -> Result<(f64, f64), String> {
        match (self.low, self.high) {
            (Some(low), Some(high)) => Ok((low, high)),
            _ => Err("missing interval".to_string()),
        }
    }

    fn valid_fraction(&self) -> bool {
        matches!((self.low, self.high), (Some(low), Some(high)) if low >= 0.0 && high <= 1.0 && low <= high)
    }

    fn valid_time(&self) -> bool {
        matches!((self.low, self.high), (Some(low), Some(high)) if low >= 0.0 && high >= low)
    }
}

#[derive(Clone, Debug)]
struct Row {
    scenario: String,
    cohort: String,
    evidence_class: String,
    evidence_status: String,
    age_days: u32,
    essentials: Range,
    opportunity: Range,
    pay: Range,
    safe_work: Range,
    stability: Range,
    social_protection: Range,
    access_time: Range,
    recorded_cohort_status: String,
    recorded_scenario_status: String,
}

impl Row {
    fn parse(line: &str, line_number: usize) -> Result<Self, String> {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 22 {
            return Err(format!(
                "line {line_number}: expected 22 columns, found {}",
                f.len()
            ));
        }
        if f[0] != "1" {
            return Err(format!("line {line_number}: unsupported schema version"));
        }
        let age_days = f[5]
            .parse::<u32>()
            .map_err(|_| format!("line {line_number}: invalid evidence age"))?;
        Ok(Self {
            scenario: f[1].to_string(),
            cohort: f[2].to_string(),
            evidence_class: f[3].to_string(),
            evidence_status: f[4].to_string(),
            age_days,
            essentials: Range::parse(f[6], f[7], "essentials access", line_number)?,
            opportunity: Range::parse(f[8], f[9], "lawful opportunity", line_number)?,
            pay: Range::parse(f[10], f[11], "fair pay", line_number)?,
            safe_work: Range::parse(f[12], f[13], "safe work", line_number)?,
            stability: Range::parse(f[14], f[15], "work stability", line_number)?,
            social_protection: Range::parse(f[16], f[17], "social protection", line_number)?,
            access_time: Range::parse(f[18], f[19], "access time", line_number)?,
            recorded_cohort_status: f[20].to_string(),
            recorded_scenario_status: f[21].to_string(),
        })
    }

    fn key(&self) -> (String, String) {
        (self.scenario.clone(), self.cohort.clone())
    }
}

fn higher_is_better(range: &Range, floor: f64) -> &'static str {
    if range.low.is_none() && range.high.is_none() {
        return "missing";
    }
    if !range.valid_fraction() {
        return "invalid";
    }
    let (low, high) = match range.complete() {
        Ok(value) => value,
        Err(_) => return "missing",
    };
    if high < floor {
        "fail"
    } else if low >= floor && high - low <= MAX_INTERVAL_WIDTH {
        "pass"
    } else {
        "uncertain"
    }
}

fn lower_is_better(range: &Range, ceiling: f64) -> &'static str {
    if range.low.is_none() && range.high.is_none() {
        return "missing";
    }
    if !range.valid_time() {
        return "invalid";
    }
    let (low, high) = match range.complete() {
        Ok(value) => value,
        Err(_) => return "missing",
    };
    if low > ceiling {
        "fail"
    } else if high <= ceiling && high - low <= MAX_TIME_INTERVAL_WIDTH_MINUTES {
        "pass"
    } else {
        "uncertain"
    }
}

fn assess_cohort(row: &Row) -> &'static str {
    if row.evidence_status != "usable" || row.age_days > MAX_EVIDENCE_AGE_DAYS {
        return "fixture_indeterminate";
    }
    let checks = [
        higher_is_better(&row.essentials, MIN_ESSENTIALS_ACCESS),
        higher_is_better(&row.opportunity, MIN_LAWFUL_OPPORTUNITY_COVERAGE),
        higher_is_better(&row.pay, MIN_FAIR_PAY_ADEQUACY),
        higher_is_better(&row.safe_work, MIN_SAFE_WORK),
        higher_is_better(&row.stability, MIN_WORK_STABILITY),
        higher_is_better(&row.social_protection, MIN_SOCIAL_PROTECTION),
        lower_is_better(&row.access_time, MAX_LAWFUL_ACCESS_TIME_MINUTES),
    ];
    if checks.contains(&"invalid") {
        return "fixture_indeterminate";
    }
    if checks.contains(&"fail") {
        return "fixture_fail";
    }
    if checks.contains(&"missing") || checks.contains(&"uncertain") {
        return "fixture_indeterminate";
    }
    "fixture_pass"
}

fn group_gap(rows: &[&Row], selector: fn(&Row) -> &Range) -> Option<f64> {
    let mut minimum_low = f64::INFINITY;
    let mut maximum_high = f64::NEG_INFINITY;
    for row in rows {
        let range = selector(row);
        let (low, high) = range.complete().ok()?;
        if !range.valid_fraction() {
            return None;
        }
        minimum_low = minimum_low.min(low);
        maximum_high = maximum_high.max(high);
    }
    Some(maximum_high - minimum_low)
}

fn assess_scenario(rows: &[&Row]) -> &'static str {
    if rows.len() != COHORTS.len()
        || rows
            .iter()
            .any(|r| r.evidence_status != "usable" || r.age_days > MAX_EVIDENCE_AGE_DAYS)
    {
        return "fixture_indeterminate";
    }
    let local: Vec<&str> = rows.iter().map(|r| assess_cohort(r)).collect();
    if local.contains(&"fixture_fail") {
        return "fixture_fail";
    }
    if local.contains(&"fixture_indeterminate") {
        return "fixture_indeterminate";
    }
    let essentials_gap = group_gap(rows, |r| &r.essentials);
    let opportunity_gap = group_gap(rows, |r| &r.opportunity);
    match (essentials_gap, opportunity_gap) {
        (Some(essentials), Some(opportunity))
            if essentials <= MAX_COHORT_COVERAGE_GAP && opportunity <= MAX_COHORT_COVERAGE_GAP =>
        {
            "fixture_pass"
        }
        (Some(_), Some(_)) => "fixture_fail",
        _ => "fixture_indeterminate",
    }
}

fn run(path: &str) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let mut lines = content.lines();
    if lines.next().ok_or_else(|| "artifact empty".to_string())? != HEADER {
        return Err("header does not match frozen viability schema".to_string());
    }
    let mut rows: BTreeMap<(String, String), Row> = BTreeMap::new();
    for (i, line) in lines.enumerate() {
        if line.trim().is_empty() {
            return Err(format!("blank line {}", i + 2));
        }
        let row = Row::parse(line, i + 2)?;
        if !SCENARIOS.contains(&row.scenario.as_str()) {
            return Err(format!("unknown scenario at line {}", i + 2));
        }
        if !COHORTS.contains(&row.cohort.as_str()) {
            return Err(format!("unknown cohort at line {}", i + 2));
        }
        if row.evidence_class != "synthetic_fixture" {
            return Err(format!(
                "non-synthetic evidence class is forbidden in this fixture at line {}",
                i + 2
            ));
        }
        for range in [
            &row.essentials,
            &row.opportunity,
            &row.pay,
            &row.safe_work,
            &row.stability,
            &row.social_protection,
        ] {
            if let (Some(low), Some(high)) = (range.low, range.high) {
                if low < 0.0 || high > 1.0 || low > high {
                    return Err(format!("fractional range out of bounds at line {}", i + 2));
                }
            }
        }
        if row.age_days > 100_000 {
            return Err(format!("implausible evidence age at line {}", i + 2));
        }
        if !["usable", "stale", "conflicted", "unavailable"].contains(&row.evidence_status.as_str())
        {
            return Err(format!("unknown evidence state at line {}", i + 2));
        }
        let key = row.key();
        if rows.insert(key.clone(), row).is_some() {
            return Err(format!("duplicate scenario/cohort pair {key:?}"));
        }
    }

    if rows.len() != SCENARIOS.len() * COHORTS.len() {
        return Err(format!(
            "expected {} rows, got {}",
            SCENARIOS.len() * COHORTS.len(),
            rows.len()
        ));
    }

    let mut scenario_map: BTreeMap<String, Vec<&Row>> = BTreeMap::new();
    for scenario in SCENARIOS {
        for cohort in COHORTS {
            let row = rows
                .get(&(scenario.to_string(), cohort.to_string()))
                .ok_or_else(|| format!("missing row {scenario}/{cohort}"))?;
            let calculated = assess_cohort(row);
            if row.recorded_cohort_status != calculated {
                return Err(format!(
                    "cohort status mismatch for {scenario}/{cohort}: recorded={}, calculated={calculated}",
                    row.recorded_cohort_status
                ));
            }
            scenario_map
                .entry(scenario.to_string())
                .or_default()
                .push(row);
        }
    }

    for scenario in SCENARIOS {
        let group = scenario_map.get(scenario).expect("scenario populated");
        let calculated = assess_scenario(group);
        for row in group {
            if row.recorded_scenario_status != calculated {
                return Err(format!(
                    "scenario gate mismatch for {scenario}: recorded={}, calculated={calculated}",
                    row.recorded_scenario_status
                ));
            }
        }
    }

    let baseline: Vec<&Row> = scenario_map["baseline"].clone();
    if assess_scenario(&baseline) != "fixture_pass" {
        return Err("baseline synthetic fixture must pass all gates".to_string());
    }
    let high_mean: Vec<&Row> = scenario_map["high_mean_low_floor"].clone();
    let mean_essentials = high_mean
        .iter()
        .map(|row| {
            let (low, high) = row
                .essentials
                .complete()
                .expect("fixture intervals are present");
            (low + high) / 2.0
        })
        .sum::<f64>()
        / high_mean.len() as f64;
    if mean_essentials <= MIN_ESSENTIALS_ACCESS || assess_scenario(&high_mean) != "fixture_fail" {
        return Err("high mean must not hide a clearly failing cohort floor".to_string());
    }
    let job_count: Vec<&Row> = scenario_map["opportunity_without_quality"].clone();
    if job_count
        .iter()
        .any(|row| row.opportunity.low.unwrap_or(0.0) < MIN_LAWFUL_OPPORTUNITY_COVERAGE)
        || assess_scenario(&job_count) != "fixture_fail"
    {
        return Err(
            "high opportunity coverage must not compensate for poor quality dimensions".to_string(),
        );
    }
    if assess_scenario(&scenario_map["high_access_friction"]) != "fixture_fail" {
        return Err("high lawful-access friction must veto apparent feasibility".to_string());
    }
    for scenario in [
        "threshold_straddle",
        "stale_source_data",
        "conflicted_sources",
        "missing_data",
        "wide_uncertainty",
    ] {
        if assess_scenario(&scenario_map[scenario]) != "fixture_indeterminate" {
            return Err(format!("{scenario} should remain indeterminate"));
        }
    }
    if assess_scenario(&scenario_map["opportunity_gap_hidden_by_mean"]) != "fixture_fail" {
        return Err("cohort opportunity gap must not be hidden by the mean".to_string());
    }

    println!("VIABILITY_VERIFICATION=PASS");
    println!(
        "CLAIM_CEILING=synthetic gate contract only; not an empirical, causal, legal, or individual assessment"
    );
    println!("SCENARIOS={}", SCENARIOS.len());
    println!("COHORTS_PER_SCENARIO={}", COHORTS.len());
    println!("ROWS={}", rows.len());
    println!(
        "NON_COMPENSATORY_GATES=essentials,opportunity_coverage,fair_pay,safe_work,stability,social_protection,access_time,cohort_gap,evidence_freshness,uncertainty"
    );
    println!(
        "NEGATIVE_CONTROLS=high_mean_low_floor,opportunity_without_quality,high_access_friction,threshold_straddle,stale_source_data,conflicted_sources,missing_data,wide_uncertainty,opportunity_gap_hidden_by_mean"
    );
    for scenario in SCENARIOS {
        println!(
            "SYNTHETIC_FIXTURE scenario={scenario} gate={} note=ILLUSTRATIVE_NOT_EMPIRICAL",
            assess_scenario(&scenario_map[scenario])
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let path = match env::args().nth(1) {
        Some(path) => path,
        None => {
            eprintln!("usage: verify_viability <viability.tsv>");
            return ExitCode::from(2);
        }
    };
    match run(&path) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("VIABILITY_VERIFICATION=FAIL: {error}");
            ExitCode::FAILURE
        }
    }
}

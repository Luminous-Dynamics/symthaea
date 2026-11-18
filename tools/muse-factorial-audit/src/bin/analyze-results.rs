//! Strict importer and report generator for Muse factorial composer measurements.
//!
//! Input rows describe one case/metric outcome. Missing rows, failed executions,
//! and not-applicable metrics are distinct states; only a complete 81-observation
//! block is eligible for contrast estimation.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use muse_factorial_audit::{
    Case, Cohort, ContrastEstimate, Observation, estimate_block, generate_cases,
    summarize_seed_effects,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutcomeStatus {
    Observed,
    NotApplicable,
    Failed,
}

impl OutcomeStatus {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "observed" => Ok(Self::Observed),
            "not_applicable" => Ok(Self::NotApplicable),
            "failed" => Ok(Self::Failed),
            other => Err(format!(
                "unknown outcome status {other:?}; expected observed, not_applicable, or failed"
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct InputRecord {
    case_id: String,
    metric: String,
    status: OutcomeStatus,
    value: Option<f64>,
    artifact_sha256: String,
    note: String,
}

#[derive(Clone, Debug, PartialEq)]
struct CoverageRow {
    cohort: String,
    style: String,
    seed: u64,
    metric: String,
    expected: usize,
    observed: usize,
    not_applicable: usize,
    failed: usize,
    missing: usize,
    disposition: String,
    artifact_sha256s: String,
    notes: String,
}

#[derive(Clone, Debug, PartialEq)]
struct EffectRow {
    cohort: String,
    style: String,
    seed: u64,
    metric: String,
    estimate: ContrastEstimate,
}

#[derive(Clone, Debug, PartialEq)]
struct SummaryRow {
    cohort: String,
    style: String,
    metric: String,
    term: String,
    order: usize,
    seed_count: usize,
    mean: f64,
    sample_sd: Option<f64>,
    minimum: f64,
    maximum: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Analysis {
    coverage: Vec<CoverageRow>,
    effects: Vec<EffectRow>,
    summaries: Vec<SummaryRow>,
    incomplete_blocks: usize,
}

fn validate_metric(metric: &str) -> Result<(), String> {
    if metric.is_empty()
        || !metric
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(format!("invalid metric name {metric:?}"));
    }
    Ok(())
}

fn parse_metrics(value: &str) -> Result<Vec<String>, String> {
    let mut metrics = Vec::new();
    let mut seen = HashSet::new();
    for raw in value.split(',') {
        let metric = raw.trim();
        validate_metric(metric)?;
        if !seen.insert(metric.to_string()) {
            return Err(format!("duplicate metric {metric:?}"));
        }
        metrics.push(metric.to_string());
    }
    if metrics.is_empty() {
        return Err("at least one expected metric is required".to_string());
    }
    Ok(metrics)
}

/// Parse RFC-4180-style comma-separated records, including quoted commas,
/// escaped quotes, CRLF, and embedded newlines. Malformed quote placement
/// rejects the entire input rather than shifting columns.
fn parse_csv(input: &str) -> Result<Vec<Vec<String>>, String> {
    let chars: Vec<char> = input.chars().collect();
    let mut rows = Vec::new();
    let mut row = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut after_quote = false;
    let mut index = 0usize;

    while index < chars.len() {
        let ch = chars[index];
        if in_quotes {
            if ch == '"' {
                if chars.get(index + 1) == Some(&'"') {
                    field.push('"');
                    index += 1;
                } else {
                    in_quotes = false;
                    after_quote = true;
                }
            } else {
                field.push(ch);
            }
        } else if after_quote {
            match ch {
                ',' => {
                    row.push(std::mem::take(&mut field));
                    after_quote = false;
                }
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    if !(row.len() == 1 && row[0].is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                    after_quote = false;
                }
                '\r' => {
                    if chars.get(index + 1) == Some(&'\n') {
                        index += 1;
                    }
                    row.push(std::mem::take(&mut field));
                    if !(row.len() == 1 && row[0].is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                    after_quote = false;
                }
                _ => {
                    return Err(format!(
                        "unexpected character {ch:?} after closing quote at character {index}"
                    ));
                }
            }
        } else {
            match ch {
                '"' if field.is_empty() => in_quotes = true,
                '"' => {
                    return Err(format!(
                        "quote inside an unquoted field at character {index}"
                    ));
                }
                ',' => row.push(std::mem::take(&mut field)),
                '\n' => {
                    row.push(std::mem::take(&mut field));
                    if !(row.len() == 1 && row[0].is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                }
                '\r' => {
                    if chars.get(index + 1) == Some(&'\n') {
                        index += 1;
                    }
                    row.push(std::mem::take(&mut field));
                    if !(row.len() == 1 && row[0].is_empty()) {
                        rows.push(std::mem::take(&mut row));
                    } else {
                        row.clear();
                    }
                }
                _ => field.push(ch),
            }
        }
        index += 1;
    }

    if in_quotes {
        return Err("unterminated quoted field".to_string());
    }
    if !field.is_empty() || !row.is_empty() || after_quote {
        row.push(field);
        if !(row.len() == 1 && row[0].is_empty()) {
            rows.push(row);
        }
    }
    Ok(rows)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn csv_field<'a>(
    row: &'a [String],
    positions: &HashMap<&str, usize>,
    name: &str,
) -> Result<&'a str, String> {
    let index = *positions
        .get(name)
        .ok_or_else(|| format!("missing required CSV column {name:?}"))?;
    row.get(index)
        .map(String::as_str)
        .ok_or_else(|| format!("row is missing column {name:?}"))
}

fn parse_records(input: &str) -> Result<Vec<InputRecord>, String> {
    let mut rows = parse_csv(input)?;
    if rows.is_empty() {
        return Err("observations CSV is empty".to_string());
    }

    let mut header = rows.remove(0);
    if let Some(first) = header.first_mut() {
        *first = first.trim_start_matches('\u{feff}').to_string();
    }

    let allowed = ["case_id", "metric", "status", "value", "artifact_sha256", "note"];
    let required = ["case_id", "metric", "status", "value", "artifact_sha256"];
    let mut positions = HashMap::new();
    for (index, name) in header.iter().enumerate() {
        if !allowed.contains(&name.as_str()) {
            return Err(format!("unexpected CSV column {name:?}"));
        }
        if positions.insert(name.as_str(), index).is_some() {
            return Err(format!("duplicate CSV column {name:?}"));
        }
    }
    for name in required {
        if !positions.contains_key(name) {
            return Err(format!("missing required CSV column {name:?}"));
        }
    }

    let mut records = Vec::with_capacity(rows.len());
    for (offset, row) in rows.iter().enumerate() {
        if row.len() != header.len() {
            return Err(format!(
                "CSV data row {} has {} fields; header has {}",
                offset + 2,
                row.len(),
                header.len()
            ));
        }

        let case_id = csv_field(row, &positions, "case_id")?.trim().to_string();
        let metric = csv_field(row, &positions, "metric")?.trim().to_string();
        let status = OutcomeStatus::parse(csv_field(row, &positions, "status")?.trim())?;
        let value_text = csv_field(row, &positions, "value")?.trim();
        let artifact_sha256 = csv_field(row, &positions, "artifact_sha256")?
            .trim()
            .to_ascii_lowercase();
        let note = positions
            .get("note")
            .and_then(|index| row.get(*index))
            .cloned()
            .unwrap_or_default();

        if case_id.is_empty() {
            return Err(format!("CSV data row {} has an empty case_id", offset + 2));
        }
        validate_metric(&metric)?;

        if !artifact_sha256.is_empty() && !is_sha256(&artifact_sha256) {
            return Err(format!(
                "CSV data row {} has an invalid artifact_sha256",
                offset + 2
            ));
        }

        let value = match status {
            OutcomeStatus::Observed => {
                if artifact_sha256.is_empty() {
                    return Err(format!(
                        "observed row {} must bind a 64-character artifact_sha256",
                        offset + 2
                    ));
                }
                if value_text.is_empty() {
                    return Err(format!("observed row {} has no scalar value", offset + 2));
                }
                let parsed = value_text.parse::<f64>().map_err(|error| {
                    format!("invalid scalar on CSV row {}: {error}", offset + 2)
                })?;
                if !parsed.is_finite() {
                    return Err(format!(
                        "observed row {} has a non-finite scalar",
                        offset + 2
                    ));
                }
                Some(parsed)
            }
            OutcomeStatus::NotApplicable | OutcomeStatus::Failed => {
                if !value_text.is_empty() {
                    return Err(format!(
                        "row {} with status {:?} must not contain a scalar value",
                        offset + 2,
                        status
                    ));
                }
                None
            }
        };

        records.push(InputRecord {
            case_id,
            metric,
            status,
            value,
            artifact_sha256,
            note,
        });
    }
    Ok(records)
}

fn disposition(
    expected: usize,
    observed: usize,
    not_applicable: usize,
    failed: usize,
    missing: usize,
) -> &'static str {
    if observed == expected {
        "complete"
    } else if not_applicable == expected {
        "not_applicable"
    } else if failed == expected {
        "failed"
    } else if missing == expected {
        "missing"
    } else {
        "incomplete"
    }
}

fn analyze(
    cases: &[Case],
    metrics: &[String],
    records: &[InputRecord],
) -> Result<Analysis, String> {
    if cases.is_empty() {
        return Err("the selected cohort has no expected cases".to_string());
    }
    if metrics.is_empty() {
        return Err("at least one expected metric is required".to_string());
    }

    let mut expected_case_ids = HashSet::with_capacity(cases.len());
    for case in cases {
        if !expected_case_ids.insert(case.case_id.as_str()) {
            return Err(format!("duplicate expected case_id {:?}", case.case_id));
        }
    }

    let expected_metrics: HashSet<&str> = metrics.iter().map(String::as_str).collect();
    if expected_metrics.len() != metrics.len() {
        return Err("expected metric list contains duplicates".to_string());
    }
    for metric in metrics {
        validate_metric(metric)?;
    }

    let mut by_case_metric: HashMap<(String, String), &InputRecord> =
        HashMap::with_capacity(records.len());
    for record in records {
        if !expected_case_ids.contains(record.case_id.as_str()) {
            return Err(format!("unexpected case_id {:?}", record.case_id));
        }
        if !expected_metrics.contains(record.metric.as_str()) {
            return Err(format!("unregistered metric {:?}", record.metric));
        }
        let key = (record.case_id.clone(), record.metric.clone());
        if by_case_metric.insert(key, record).is_some() {
            return Err(format!(
                "duplicate observation for case {:?}, metric {:?}",
                record.case_id, record.metric
            ));
        }
    }

    let mut groups: BTreeMap<(String, String, u64, String), Vec<&Case>> = BTreeMap::new();
    for case in cases {
        for metric in metrics {
            groups
                .entry((
                    case.cohort.as_str().to_string(),
                    case.style.clone(),
                    case.seed,
                    metric.clone(),
                ))
                .or_default()
                .push(case);
        }
    }

    let mut coverage = Vec::with_capacity(groups.len());
    let mut effects = Vec::new();
    let mut blocks_for_summary: BTreeMap<
        (String, String, String),
        BTreeMap<u64, Vec<ContrastEstimate>>,
    > = BTreeMap::new();
    let mut incomplete_blocks = 0usize;

    for ((cohort_name, style, seed, metric), mut block_cases) in groups {
        block_cases.sort_by_key(|case| case.cell);
        if block_cases.len() != 81 {
            return Err(format!(
                "expected 81 cases for {cohort_name}/{style}/seed {seed}/{metric}, found {}",
                block_cases.len()
            ));
        }

        let mut observed = 0usize;
        let mut not_applicable = 0usize;
        let mut failed = 0usize;
        let mut missing = 0usize;
        let mut observations = Vec::with_capacity(81);
        let mut artifact_sha256s = Vec::new();
        let mut notes = Vec::new();

        for case in &block_cases {
            let key = (case.case_id.clone(), metric.clone());
            match by_case_metric.get(&key) {
                None => missing += 1,
                Some(record) => {
                    if !record.note.is_empty() {
                        notes.push(format!("{}: {}", record.case_id, record.note));
                    }
                    if !record.artifact_sha256.is_empty() {
                        artifact_sha256s.push(record.artifact_sha256.clone());
                    }
                    match record.status {
                        OutcomeStatus::Observed => {
                            observed += 1;
                            let value = record.value.ok_or_else(|| {
                                format!("observed row {:?} lost its scalar value", record.case_id)
                            })?;
                            observations.push(Observation {
                                cohort: case.cohort,
                                style: case.style.clone(),
                                seed: case.seed,
                                metric: metric.clone(),
                                cell: case.cell,
                                case_id: case.case_id.clone(),
                                value,
                            });
                        }
                        OutcomeStatus::NotApplicable => not_applicable += 1,
                        OutcomeStatus::Failed => failed += 1,
                    }
                }
            }
        }

        let status = disposition(
            block_cases.len(),
            observed,
            not_applicable,
            failed,
            missing,
        );
        if status != "complete" {
            incomplete_blocks += 1;
        } else {
            let first = block_cases[0];
            let estimates = estimate_block(
                first.cohort,
                &style,
                seed,
                &metric,
                &observations,
            )
            .map_err(|error| {
                format!(
                    "contrast estimation rejected {cohort_name}/{style}/seed {seed}/{metric}: {error}"
                )
            })?;
            for estimate in &estimates {
                effects.push(EffectRow {
                    cohort: cohort_name.clone(),
                    style: style.clone(),
                    seed,
                    metric: metric.clone(),
                    estimate: estimate.clone(),
                });
            }
            blocks_for_summary
                .entry((cohort_name.clone(), style.clone(), metric.clone()))
                .or_default()
                .insert(seed, estimates);
        }

        coverage.push(CoverageRow {
            cohort: cohort_name,
            style,
            seed,
            metric,
            expected: block_cases.len(),
            observed,
            not_applicable,
            failed,
            missing,
            disposition: status.to_string(),
            artifact_sha256s: artifact_sha256s.join(";"),
            notes: notes.join(" | "),
        });
    }

    let mut summaries = Vec::new();
    for ((cohort, style, metric), seed_map) in blocks_for_summary {
        let seed_blocks: Vec<Vec<ContrastEstimate>> = seed_map.into_values().collect();
        let summary = summarize_seed_effects(&seed_blocks).map_err(|error| {
            format!("cross-seed summary rejected {cohort}/{style}/{metric}: {error}")
        })?;
        for item in summary {
            summaries.push(SummaryRow {
                cohort: cohort.clone(),
                style: style.clone(),
                metric: metric.clone(),
                term: item.term,
                order: item.order,
                seed_count: item.seed_count,
                mean: item.mean,
                sample_sd: item.sample_sd,
                minimum: item.minimum,
                maximum: item.maximum,
            });
        }
    }

    Ok(Analysis {
        coverage,
        effects,
        summaries,
        incomplete_blocks,
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn csv_record(fields: &[String]) -> String {
    let mut line = fields
        .iter()
        .map(|field| csv_escape(field))
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

fn coverage_csv(rows: &[CoverageRow]) -> String {
    let mut output = String::from(
        "cohort,style,seed,metric,expected,observed,not_applicable,failed,missing,disposition,artifact_sha256s,notes\n",
    );
    for row in rows {
        output.push_str(&csv_record(&[
            row.cohort.clone(),
            row.style.clone(),
            row.seed.to_string(),
            row.metric.clone(),
            row.expected.to_string(),
            row.observed.to_string(),
            row.not_applicable.to_string(),
            row.failed.to_string(),
            row.missing.to_string(),
            row.disposition.clone(),
            row.artifact_sha256s.clone(),
            row.notes.clone(),
        ]));
    }
    output
}

fn effects_csv(rows: &[EffectRow]) -> String {
    let mut output =
        String::from("cohort,style,seed,metric,term,order,estimate\n");
    for row in rows {
        output.push_str(&csv_record(&[
            row.cohort.clone(),
            row.style.clone(),
            row.seed.to_string(),
            row.metric.clone(),
            row.estimate.term.clone(),
            row.estimate.order.to_string(),
            row.estimate.estimate.to_string(),
        ]));
    }
    output
}

fn summary_csv(rows: &[SummaryRow]) -> String {
    let mut output = String::from(
        "cohort,style,metric,term,order,seed_count,mean,sample_sd,minimum,maximum\n",
    );
    for row in rows {
        output.push_str(&csv_record(&[
            row.cohort.clone(),
            row.style.clone(),
            row.metric.clone(),
            row.term.clone(),
            row.order.to_string(),
            row.seed_count.to_string(),
            row.mean.to_string(),
            row.sample_sd.map(|value| value.to_string()).unwrap_or_default(),
            row.minimum.to_string(),
            row.maximum.to_string(),
        ]));
    }
    output
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: analyze-results <discovery|confirmation|both> <observations.csv> <metric[,metric...]> <coverage.csv> <effects.csv> <summary.csv>"
                .to_string(),
        );
    }

    let cohorts = match args[0].as_str() {
        "discovery" => vec![Cohort::Discovery],
        "confirmation" => vec![Cohort::Confirmation],
        "both" => vec![Cohort::Discovery, Cohort::Confirmation],
        other => return Err(format!("unknown cohort selector {other:?}")),
    };
    let metrics = parse_metrics(&args[2])?;

    let input_path = PathBuf::from(&args[1]);
    let output_paths = [
        PathBuf::from(&args[3]),
        PathBuf::from(&args[4]),
        PathBuf::from(&args[5]),
    ];
    if output_paths[0] == output_paths[1]
        || output_paths[0] == output_paths[2]
        || output_paths[1] == output_paths[2]
        || output_paths.iter().any(|path| path == &input_path)
    {
        return Err("input and output paths must all be distinct".to_string());
    }

    let input = fs::read_to_string(&input_path)
        .map_err(|error| format!("cannot read {}: {error}", input_path.display()))?;
    let records = parse_records(&input)?;
    let mut cases = Vec::new();
    for cohort in cohorts {
        cases.extend(generate_cases(cohort));
    }

    let analysis = analyze(&cases, &metrics, &records)?;
    fs::write(&output_paths[0], coverage_csv(&analysis.coverage))
        .map_err(|error| format!("cannot write {}: {error}", output_paths[0].display()))?;
    fs::write(&output_paths[1], effects_csv(&analysis.effects))
        .map_err(|error| format!("cannot write {}: {error}", output_paths[1].display()))?;
    fs::write(&output_paths[2], summary_csv(&analysis.summaries))
        .map_err(|error| format!("cannot write {}: {error}", output_paths[2].display()))?;

    if analysis.incomplete_blocks > 0 {
        return Err(format!(
            "wrote coverage/effect/summary reports, but {} of {} blocks are incomplete; only complete 81/81 observed blocks were analyzed",
            analysis.incomplete_blocks,
            analysis.coverage.len()
        ));
    }

    eprintln!(
        "analyzed {} complete blocks; wrote {} per-seed contrast rows and {} cross-seed summary rows",
        analysis.coverage.len(),
        analysis.effects.len(),
        analysis.summaries.len()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("muse-factorial-audit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use muse_factorial_audit::DISCOVERY_SEEDS;

    fn one_complete_block() -> (Vec<Case>, Vec<InputRecord>) {
        let cases: Vec<Case> = generate_cases(Cohort::Discovery)
            .into_iter()
            .filter(|case| case.style == "Classical" && case.seed == DISCOVERY_SEEDS[0])
            .collect();
        let records = cases
            .iter()
            .map(|case| {
                let linear = [-1.0, 0.0, 1.0];
                let quadratic = [1.0, -2.0, 1.0];
                let value = 7.0
                    + 2.0 * linear[case.cell.valence as usize]
                    + 3.0 * quadratic[case.cell.arousal as usize]
                    + 4.0
                        * linear[case.cell.valence as usize]
                        * quadratic[case.cell.energy as usize];
                InputRecord {
                    case_id: case.case_id.clone(),
                    metric: "response".to_string(),
                    status: OutcomeStatus::Observed,
                    value: Some(value),
                    artifact_sha256: "a".repeat(64),
                    note: String::new(),
                }
            })
            .collect();
        (cases, records)
    }

    #[test]
    fn parser_supports_quoted_commas_newlines_and_escaped_quotes() {
        let parsed = parse_csv("a,b\r\n\"x,y\",\"line one\r\nline two\"\r\n\"say \"\"hi\"\"\",z\r\n")
            .unwrap();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0], vec!["a", "b"]);
        assert_eq!(parsed[1], vec!["x,y", "line one\r\nline two"]);
        assert_eq!(parsed[2], vec!["say \"hi\"", "z"]);
    }

    #[test]
    fn parser_rejects_unclosed_and_misplaced_quotes() {
        assert!(parse_csv("a,b\n\"unclosed,b").is_err());
        assert!(parse_csv("a,b\nbad\"quote,x").is_err());
        assert!(parse_csv("a,b\n\"ok\" trailing,x").is_err());
    }

    #[test]
    fn metrics_are_explicit_and_unique() {
        assert_eq!(
            parse_metrics("note_density,mean_pitch").unwrap(),
            vec!["note_density", "mean_pitch"]
        );
        assert!(parse_metrics("note_density,note_density").is_err());
        assert!(parse_metrics("note density").is_err());
        assert!(parse_metrics("note_density,").is_err());
    }

    #[test]
    fn full_block_produces_eighty_effects_and_seed_summary() {
        let (cases, records) = one_complete_block();
        let analysis = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(analysis.incomplete_blocks, 0);
        assert_eq!(analysis.coverage.len(), 1);
        assert_eq!(analysis.coverage[0].disposition, "complete");
        assert_eq!(analysis.coverage[0].observed, 81);
        assert_eq!(analysis.effects.len(), 80);
        assert_eq!(analysis.summaries.len(), 80);

        let main = analysis
            .effects
            .iter()
            .find(|row| row.estimate.term == "valence_linear")
            .unwrap();
        assert!((main.estimate.estimate - 2.0).abs() < 1.0e-12);
        assert!(analysis.coverage[0].artifact_sha256s.split(';').count() == 81);
    }

    #[test]
    fn not_applicable_is_not_zero_and_blocks_estimation() {
        let (cases, mut records) = one_complete_block();
        let index = records
            .iter()
            .position(|record| record.case_id == cases[0].case_id)
            .unwrap();
        records[index].status = OutcomeStatus::NotApplicable;
        records[index].value = None;
        records[index].artifact_sha256.clear();

        let analysis = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(analysis.incomplete_blocks, 1);
        assert_eq!(analysis.coverage[0].not_applicable, 1);
        assert_eq!(analysis.coverage[0].observed, 80);
        assert_eq!(analysis.effects.len(), 0);
        assert_eq!(analysis.summaries.len(), 0);
    }

    #[test]
    fn missing_and_failed_outcomes_remain_separate() {
        let (cases, mut records) = one_complete_block();
        records.pop();
        let analysis = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(analysis.coverage[0].missing, 1);
        assert_eq!(analysis.coverage[0].failed, 0);
        assert_eq!(analysis.effects.len(), 0);

        let (cases, mut records) = one_complete_block();
        records[0].status = OutcomeStatus::Failed;
        records[0].value = None;
        records[0].artifact_sha256.clear();
        records[0].note = "composer returned an error".to_string();
        let analysis = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(analysis.coverage[0].failed, 1);
        assert_eq!(analysis.coverage[0].missing, 0);
        assert!(analysis.coverage[0].notes.contains("composer returned an error"));
        assert_eq!(analysis.effects.len(), 0);
    }

    #[test]
    fn duplicate_or_unregistered_observations_reject() {
        let (cases, mut records) = one_complete_block();
        records.push(records[0].clone());
        assert!(analyze(&cases, &["response".to_string()], &records).is_err());

        let (cases, mut records) = one_complete_block();
        records[0].metric = "unregistered".to_string();
        assert!(analyze(&cases, &["response".to_string()], &records).is_err());

        let (cases, mut records) = one_complete_block();
        records[0].case_id = "invented-case".to_string();
        assert!(analyze(&cases, &["response".to_string()], &records).is_err());
    }

    #[test]
    fn row_order_does_not_change_reports() {
        let (cases, mut records) = one_complete_block();
        let forward = analyze(&cases, &["response".to_string()], &records).unwrap();
        records.reverse();
        let reverse = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(forward, reverse);
        assert!(coverage_csv(&forward.coverage).starts_with("cohort,style,seed"));
        assert!(effects_csv(&forward.effects).starts_with("cohort,style,seed"));
        assert!(summary_csv(&forward.summaries).starts_with("cohort,style,metric"));
    }

    #[test]
    fn imported_status_and_digest_rules_are_strict() {
        let header = "case_id,metric,status,value,artifact_sha256,note\n";
        let good = format!(
            "{},response,observed,1.25,{},ok\n",
            "DISCOVERY__Classical__s90210__v0a0e0b0",
            "b".repeat(64)
        );
        assert_eq!(parse_records(&(header.to_string() + &good)).unwrap().len(), 1);

        let bad_digest = format!(
            "{},response,observed,1.25,short,broken\n",
            "DISCOVERY__Classical__s90210__v0a0e0b0"
        );
        assert!(parse_records(&(header.to_string() + &bad_digest)).is_err());

        let not_applicable_with_value = format!(
            "{},response,not_applicable,0,,must not impute zero\n",
            "DISCOVERY__Classical__s90210__v0a0e0b0"
        );
        assert!(parse_records(&(header.to_string() + &not_applicable_with_value)).is_err());
    }

    #[test]
    fn output_escapes_notes_and_empty_summary_statistics() {
        let field = csv_escape("failure, with \"quotes\"\nand newline");
        assert_eq!(field, "\"failure, with \"\"quotes\"\"\nand newline\"");
    }
}

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
    analysis_mode: String,
    artifact_sha256s: String,
    notes: String,
}

#[derive(Clone, Debug, PartialEq)]
struct AxisInvarianceRow {
    cohort: String,
    style: String,
    seed: u64,
    axis: String,
    declared_supported: Option<bool>,
    expected_slices: usize,
    complete_slices: usize,
    identical_slices: usize,
    changed_slices: usize,
    unavailable_slices: usize,
    disposition: String,
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
    axis_invariance: Vec<AxisInvarianceRow>,
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

    let allowed = [
        "case_id",
        "metric",
        "status",
        "value",
        "artifact_sha256",
        "note",
    ];
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

fn semantic_digest_from_note(note: &str) -> Option<&str> {
    note.split(';')
        .find_map(|part| part.strip_prefix("semantic_sha256="))
}

fn support_axes_from_note(note: &str) -> Option<String> {
    note.split(';')
        .find_map(|part| part.strip_prefix("support_axes="))
        .map(str::to_string)
}

fn axis_disposition(
    declared_supported: Option<bool>,
    complete_slices: usize,
    identical_slices: usize,
    changed_slices: usize,
    unavailable_slices: usize,
) -> &'static str {
    if unavailable_slices > 0 || complete_slices != 27 {
        return "incomplete";
    }
    match (declared_supported, changed_slices > 0) {
        (Some(false), false) => "negative_control_pass",
        (Some(false), true) => "unexpected_change",
        (Some(true), true) => "response_observed",
        (Some(true), false) => "flat_supported_axis",
        (None, true) => "response_unclassified",
        (None, false) if identical_slices == 27 => "invariant_unclassified",
        (None, false) => "mixed_unclassified",
    }
}

fn axis_invariance_report(
    cases: &[Case],
    digests_by_case: &HashMap<String, String>,
    axes_by_block: &HashMap<(String, String, u64), String>,
) -> Vec<AxisInvarianceRow> {
    const AXES: [&str; 4] = ["valence", "arousal", "energy", "bars"];
    let mut blocks: BTreeMap<(String, String, u64), Vec<&Case>> = BTreeMap::new();
    for case in cases {
        blocks
            .entry((
                case.cohort.as_str().to_string(),
                case.style.clone(),
                case.seed,
            ))
            .or_default()
            .push(case);
    }

    let mut rows = Vec::with_capacity(blocks.len() * AXES.len());
    for ((cohort, style, seed), block_cases) in blocks {
        let block_key = (cohort.clone(), style.clone(), seed);
        let declared_axes = axes_by_block.get(&block_key);
        for (axis_index, axis) in AXES.iter().enumerate() {
            let declared_supported =
                declared_axes.map(|axes| axes.split('|').any(|candidate| candidate == *axis));
            let mut complete_slices = 0usize;
            let mut identical_slices = 0usize;
            let mut changed_slices = 0usize;
            let mut unavailable_slices = 0usize;

            for first in 0_u8..3 {
                for second in 0_u8..3 {
                    for third in 0_u8..3 {
                        let mut other_levels = [first, second, third].into_iter();
                        let mut levels = [0_u8; 4];
                        for (index, level) in levels.iter_mut().enumerate() {
                            if index != axis_index {
                                *level = other_levels.next().expect("three non-target factors");
                            }
                        }

                        let mut slice_digests = Vec::with_capacity(3);
                        for axis_level in 0_u8..3 {
                            levels[axis_index] = axis_level;
                            let cell = muse_factorial_audit::Cell {
                                valence: levels[0],
                                arousal: levels[1],
                                energy: levels[2],
                                bars: levels[3],
                            };
                            let digest = block_cases
                                .iter()
                                .find(|case| case.cell == cell)
                                .and_then(|case| digests_by_case.get(&case.case_id));
                            slice_digests.push(digest);
                        }

                        if slice_digests.iter().all(Option::is_some) {
                            complete_slices += 1;
                            let first_digest = slice_digests[0].expect("slice completeness checked");
                            if slice_digests
                                .iter()
                                .all(|digest| digest.is_some_and(|value| value == first_digest))
                            {
                                identical_slices += 1;
                            } else {
                                changed_slices += 1;
                            }
                        } else {
                            unavailable_slices += 1;
                        }
                    }
                }
            }

            let disposition = axis_disposition(
                declared_supported,
                complete_slices,
                identical_slices,
                changed_slices,
                unavailable_slices,
            );
            rows.push(AxisInvarianceRow {
                cohort: cohort.clone(),
                style: style.clone(),
                seed,
                axis: (*axis).to_string(),
                declared_supported,
                expected_slices: 27,
                complete_slices,
                identical_slices,
                changed_slices,
                unavailable_slices,
                disposition: disposition.to_string(),
            });
        }
    }
    rows
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

    let expected_case_ids: HashSet<String> =
        cases.iter().map(|case| case.case_id.clone()).collect();
    if expected_case_ids.len() != cases.len() {
        return Err("expected case set contains duplicate case IDs".to_string());
    }

    let expected_metrics: HashSet<String> = metrics.iter().cloned().collect();
    if expected_metrics.len() != metrics.len() {
        return Err("expected metric list contains duplicates".to_string());
    }
    for metric in metrics {
        validate_metric(metric)?;
    }

    let case_by_id: HashMap<&str, &Case> = cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect();
    let mut by_case_metric: HashMap<(String, String), &InputRecord> =
        HashMap::with_capacity(records.len());
    let mut digests_by_case: HashMap<String, String> = HashMap::new();
    let mut semantic_digests_by_case: HashMap<String, String> = HashMap::new();
    let mut axes_by_block: HashMap<(String, String, u64), String> = HashMap::new();
    for record in records {
        if !expected_case_ids.contains(&record.case_id) {
            return Err(format!("unexpected case_id {:?}", record.case_id));
        }
        if !expected_metrics.contains(&record.metric) {
            return Err(format!("unregistered metric {:?}", record.metric));
        }
        let key = (record.case_id.clone(), record.metric.clone());
        if by_case_metric.insert(key, record).is_some() {
            return Err(format!(
                "duplicate observation for case {:?}, metric {:?}",
                record.case_id, record.metric
            ));
        }

        if !record.artifact_sha256.is_empty() {
            match digests_by_case.entry(record.case_id.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(record.artifact_sha256.clone());
                }
                std::collections::hash_map::Entry::Occupied(entry)
                    if entry.get() != &record.artifact_sha256 =>
                {
                    return Err(format!(
                        "artifact SHA-256 mismatch across metrics for case {:?}",
                        record.case_id
                    ));
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }

        if let Some(semantic_digest) = semantic_digest_from_note(&record.note) {
            if !is_sha256(semantic_digest) {
                return Err(format!(
                    "semantic score digest is malformed for case {:?}",
                    record.case_id
                ));
            }
            match semantic_digests_by_case.entry(record.case_id.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(semantic_digest.to_string());
                }
                std::collections::hash_map::Entry::Occupied(entry)
                    if entry.get() != semantic_digest =>
                {
                    return Err(format!(
                        "semantic score digest mismatch across metrics for case {:?}",
                        record.case_id
                    ));
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }

        if let Some(case) = case_by_id.get(record.case_id.as_str())
            && let Some(axes) = support_axes_from_note(&record.note)
        {
            let block_key = (
                case.cohort.as_str().to_string(),
                case.style.clone(),
                case.seed,
            );
            match axes_by_block.entry(block_key) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(axes);
                }
                std::collections::hash_map::Entry::Occupied(entry) if entry.get() != &axes => {
                    return Err(format!(
                        "declared support-axis metadata mismatch within case block {:?}",
                        record.case_id
                    ));
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }

    // Prefer the canonical event digest for invariance checks. Older or
    // synthetic inputs without it fall back to the raw artifact digest.
    for (case_id, artifact_digest) in &digests_by_case {
        semantic_digests_by_case
            .entry(case_id.clone())
            .or_insert_with(|| artifact_digest.clone());
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

        let status = disposition(block_cases.len(), observed, not_applicable, failed, missing);
        if status == "not_applicable" {
            // Every expected case explicitly declared this metric unsupported;
            // preserve the block in coverage without treating it as execution failure.
        } else if status != "complete" {
            incomplete_blocks += 1;
        } else if !metric.starts_with("identity_") {
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
            analysis_mode: if metric.starts_with("identity_") {
                "identity_only".to_string()
            } else if status == "complete" {
                "effect_estimated".to_string()
            } else {
                "not_estimated".to_string()
            },
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

    let axis_invariance =
        axis_invariance_report(cases, &semantic_digests_by_case, &axes_by_block);

    Ok(Analysis {
        coverage,
        effects,
        summaries,
        axis_invariance,
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
        "cohort,style,seed,metric,expected,observed,not_applicable,failed,missing,disposition,analysis_mode,artifact_sha256s,notes\n",
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
            row.analysis_mode.clone(),
            row.artifact_sha256s.clone(),
            row.notes.clone(),
        ]));
    }
    output
}

fn axis_invariance_csv(rows: &[AxisInvarianceRow]) -> String {
    let mut output = String::from(
        "cohort,style,seed,axis,declared_supported,expected_slices,complete_slices,identical_slices,changed_slices,unavailable_slices,disposition\n",
    );
    for row in rows {
        output.push_str(&csv_record(&[
            row.cohort.clone(),
            row.style.clone(),
            row.seed.to_string(),
            row.axis.clone(),
            row.declared_supported
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.expected_slices.to_string(),
            row.complete_slices.to_string(),
            row.identical_slices.to_string(),
            row.changed_slices.to_string(),
            row.unavailable_slices.to_string(),
            row.disposition.clone(),
        ]));
    }
    output
}

fn effects_csv(rows: &[EffectRow]) -> String {
    let mut output = String::from("cohort,style,seed,metric,term,order,estimate\n");
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
    let mut output =
        String::from("cohort,style,metric,term,order,seed_count,mean,sample_sd,minimum,maximum\n");
    for row in rows {
        output.push_str(&csv_record(&[
            row.cohort.clone(),
            row.style.clone(),
            row.metric.clone(),
            row.term.clone(),
            row.order.to_string(),
            row.seed_count.to_string(),
            row.mean.to_string(),
            row.sample_sd
                .map(|value| value.to_string())
                .unwrap_or_default(),
            row.minimum.to_string(),
            row.maximum.to_string(),
        ]));
    }
    output
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 7 {
        return Err(
            "usage: analyze-results <discovery|confirmation|both> <observations.csv> <metric[,metric...]> <coverage.csv> <effects.csv> <summary.csv> <axis-invariance.csv>"
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
        PathBuf::from(&args[6]),
    ];
    let unique_outputs: HashSet<&PathBuf> = output_paths.iter().collect();
    if unique_outputs.len() != output_paths.len()
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
    fs::write(&output_paths[3], axis_invariance_csv(&analysis.axis_invariance))
        .map_err(|error| format!("cannot write {}: {error}", output_paths[3].display()))?;

    if analysis.incomplete_blocks > 0 {
        return Err(format!(
            "wrote all reports, but {} of {} blocks contain failed, missing, or mixed outcomes; only complete 81/81 observed blocks were effect-estimated",
            analysis.incomplete_blocks,
            analysis.coverage.len()
        ));
    }

    eprintln!(
        "audited {} expected blocks; wrote {} per-seed contrast rows, {} cross-seed summary rows, and {} axis-invariance rows",
        analysis.coverage.len(),
        analysis.effects.len(),
        analysis.summaries.len(),
        analysis.axis_invariance.len()
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
        let parsed =
            parse_csv("a,b\r\n\"x,y\",\"line one\r\nline two\"\r\n\"say \"\"hi\"\"\",z\r\n")
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
    fn semantic_digest_note_is_extracted_and_validated() {
        let digest = "c".repeat(64);
        assert_eq!(
            semantic_digest_from_note(&format!("source_commit=abc;semantic_sha256={digest};manifest=def")),
            Some(digest.as_str())
        );
        assert_eq!(semantic_digest_from_note("source_commit=abc;manifest=def"), None);
    }

    #[test]
    fn semantic_digest_mismatch_across_metrics_rejects() {
        let (cases, mut records) = one_complete_block();
        let first_semantic = "c".repeat(64);
        let second_semantic = "d".repeat(64);
        for record in &mut records {
            record.note = format!("semantic_sha256={first_semantic}");
        }
        let mut identity_rows: Vec<InputRecord> = records
            .iter()
            .map(|record| {
                let mut identity = record.clone();
                identity.metric = "identity_fingerprint_word".to_string();
                identity.note = format!("semantic_sha256={second_semantic}");
                identity
            })
            .collect();
        identity_rows[0].note = format!("semantic_sha256={first_semantic}");
        records.extend(identity_rows);
        assert!(analyze(
            &cases,
            &["response".to_string(), "identity_fingerprint_word".to_string()],
            &records
        )
        .is_err());
    }

    #[test]
    fn mismatched_artifact_digests_across_metrics_reject() {
        let (cases, mut records) = one_complete_block();
        let mut identity_rows: Vec<InputRecord> = records
            .iter()
            .map(|record| {
                let mut identity = record.clone();
                identity.metric = "identity_fingerprint_word".to_string();
                identity
            })
            .collect();
        identity_rows[0].artifact_sha256 = "b".repeat(64);
        records.extend(identity_rows);
        assert!(analyze(
            &cases,
            &["response".to_string(), "identity_fingerprint_word".to_string()],
            &records
        )
        .is_err());
    }

    #[test]
    fn identity_metrics_are_coverage_audited_but_never_effect_estimated() {
        let (cases, mut records) = one_complete_block();
        let identity_rows: Vec<InputRecord> = records
            .iter()
            .map(|record| {
                let mut identity = record.clone();
                identity.metric = "identity_fingerprint_word".to_string();
                identity.value = Some(12_345.0);
                identity
            })
            .collect();
        records.extend(identity_rows);

        let analysis = analyze(
            &cases,
            &["response".to_string(), "identity_fingerprint_word".to_string()],
            &records,
        )
        .unwrap();
        assert_eq!(analysis.incomplete_blocks, 0);
        assert_eq!(analysis.coverage.len(), 2);
        assert_eq!(analysis.effects.len(), 80);
        assert_eq!(analysis.summaries.len(), 80);
        let identity = analysis
            .coverage
            .iter()
            .find(|row| row.metric == "identity_fingerprint_word")
            .unwrap();
        assert_eq!(identity.disposition, "complete");
        assert_eq!(identity.analysis_mode, "identity_only");
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
    fn fully_not_applicable_metric_is_explicit_but_not_an_execution_failure() {
        let (cases, mut records) = one_complete_block();
        for record in &mut records {
            record.status = OutcomeStatus::NotApplicable;
            record.value = None;
            record.artifact_sha256.clear();
            record.note = "metric not exposed by this grammar plan".to_string();
        }
        let analysis = analyze(&cases, &["response".to_string()], &records).unwrap();
        assert_eq!(analysis.incomplete_blocks, 0);
        assert_eq!(analysis.coverage[0].disposition, "not_applicable");
        assert_eq!(analysis.coverage[0].analysis_mode, "not_estimated");
        assert_eq!(analysis.coverage[0].not_applicable, 81);
        assert!(analysis.effects.is_empty());
        assert!(analysis.summaries.is_empty());
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
        assert!(
            analysis.coverage[0]
                .notes
                .contains("composer returned an error")
        );
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
        assert_eq!(
            parse_records(&(header.to_string() + &good)).unwrap().len(),
            1
        );

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
    fn exact_digests_confirm_supported_response_and_unsupported_invariance() {
        let cases: Vec<Case> = generate_cases(Cohort::Discovery)
            .into_iter()
            .filter(|case| case.style == "Classical" && case.seed == DISCOVERY_SEEDS[0])
            .collect();
        let digests_by_case: HashMap<String, String> = cases
            .iter()
            .map(|case| {
                (
                    case.case_id.clone(),
                    format!("{:064x}", case.cell.valence as u64 + 1),
                )
            })
            .collect();
        let axes_by_block = HashMap::from([(
            (
                "DISCOVERY".to_string(),
                "Classical".to_string(),
                DISCOVERY_SEEDS[0],
            ),
            "valence".to_string(),
        )]);

        let rows = axis_invariance_report(&cases, &digests_by_case, &axes_by_block);
        assert_eq!(rows.len(), 4);
        let valence = rows.iter().find(|row| row.axis == "valence").unwrap();
        assert_eq!(valence.declared_supported, Some(true));
        assert_eq!(valence.complete_slices, 27);
        assert_eq!(valence.changed_slices, 27);
        assert_eq!(valence.identical_slices, 0);
        assert_eq!(valence.disposition, "response_observed");

        let energy = rows.iter().find(|row| row.axis == "energy").unwrap();
        assert_eq!(energy.declared_supported, Some(false));
        assert_eq!(energy.complete_slices, 27);
        assert_eq!(energy.changed_slices, 0);
        assert_eq!(energy.identical_slices, 27);
        assert_eq!(energy.disposition, "negative_control_pass");
    }

    #[test]
    fn exact_invariance_audit_detects_unexpected_change_on_declared_dead_axis() {
        let cases: Vec<Case> = generate_cases(Cohort::Discovery)
            .into_iter()
            .filter(|case| case.style == "Classical" && case.seed == DISCOVERY_SEEDS[0])
            .collect();
        let digests_by_case: HashMap<String, String> = cases
            .iter()
            .map(|case| {
                (
                    case.case_id.clone(),
                    format!("{:064x}", case.cell.energy as u64 + 1),
                )
            })
            .collect();
        let axes_by_block = HashMap::from([(
            (
                "DISCOVERY".to_string(),
                "Classical".to_string(),
                DISCOVERY_SEEDS[0],
            ),
            "valence".to_string(),
        )]);

        let rows = axis_invariance_report(&cases, &digests_by_case, &axes_by_block);
        let energy = rows.iter().find(|row| row.axis == "energy").unwrap();
        assert_eq!(energy.declared_supported, Some(false));
        assert_eq!(energy.complete_slices, 27);
        assert_eq!(energy.changed_slices, 27);
        assert_eq!(energy.identical_slices, 0);
        assert_eq!(energy.disposition, "unexpected_change");
    }

    #[test]
    fn axis_invariance_report_marks_missing_artifacts_incomplete() {
        let cases: Vec<Case> = generate_cases(Cohort::Discovery)
            .into_iter()
            .filter(|case| case.style == "Classical" && case.seed == DISCOVERY_SEEDS[0])
            .collect();
        let mut digests_by_case: HashMap<String, String> = cases
            .iter()
            .map(|case| (case.case_id.clone(), "a".repeat(64)))
            .collect();
        digests_by_case.remove(&cases[0].case_id);
        let axes_by_block = HashMap::from([(
            (
                "DISCOVERY".to_string(),
                "Classical".to_string(),
                DISCOVERY_SEEDS[0],
            ),
            "valence".to_string(),
        )]);

        let rows = axis_invariance_report(&cases, &digests_by_case, &axes_by_block);
        let valence = rows.iter().find(|row| row.axis == "valence").unwrap();
        assert!(valence.unavailable_slices > 0);
        assert_eq!(valence.disposition, "incomplete");
    }

    #[test]
    fn output_escapes_notes_and_empty_summary_statistics() {
        let field = csv_escape("failure, with \"quotes\"\nand newline");
        assert_eq!(field, "\"failure, with \"\"quotes\"\"\nand newline\"");
    }
}

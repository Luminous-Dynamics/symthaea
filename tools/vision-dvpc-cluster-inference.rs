// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! VIS-DVPC-014: paired inference over independent evaluation clusters.
//!
//! This tool consumes one already-aggregated paired delta per independent
//! cluster and predeclared comparison. It does not accept per-frame rows.
//! Percentile intervals are approximate cluster-bootstrap intervals;
//! sign-flip p-values rely on the declared cluster-level symmetry assumption.
//! Neither output is an automatic SOTA verdict.
//!
//! Usage:
//!   rustc --edition=2024 --test tools/vision-dvpc-cluster-inference.rs -o /tmp/vis-dvpc-014-tests
//!   /tmp/vis-dvpc-014-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-cluster-inference.rs -o /tmp/vis-dvpc-014
//!   /tmp/vis-dvpc-014 DELTAS.tsv EXPECTED_BUNDLE_LOCK_SHA256
//!
//! EXPECTED_BUNDLE_LOCK_SHA256 should be taken from the separately frozen
//! VIS-DVPC-012 integrity receipt / preregistration, not read from this TSV.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::fmt::Write as FmtWrite;

const FORMAT_VERSION: &str = "VIS-DVPC-CLUSTER-DELTAS-001-v1";
const RECEIPT_VERSION: &str = "VIS-DVPC-CLUSTER-INFERENCE-001-v1";
const META_KEYS: &[&str] = &[
    "format",
    "experiment_id",
    "dataset_id",
    "dataset_revision",
    "split",
    "cluster_unit",
    "primary_metric",
    "positive_is_better",
    "bundle_lock_sha256",
    "comparison_family_size",
    "bootstrap_seed",
    "bootstrap_replicates",
    "familywise_confidence",
];
const DELTA_HEADER: &[&str] = &[
    "DELTA_HEADER",
    "comparison_id",
    "candidate_id",
    "baseline_id",
    "cluster_id",
    "delta",
];
const MIN_CLUSTERS: usize = 5;
const MIN_BOOTSTRAP_REPLICATES: u64 = 1_000;
const MAX_BOOTSTRAP_REPLICATES: u64 = 1_000_000;
const EXACT_SIGN_FLIP_MAX_CLUSTERS: usize = 16;

#[derive(Debug, Clone)]
struct Table {
    meta: BTreeMap<String, String>,
    rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone)]
struct Comparison {
    candidate_id: String,
    baseline_id: String,
    deltas: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy)]
struct Config {
    family_size: usize,
    seed: u64,
    bootstrap_replicates: u64,
    familywise_confidence: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct ResultRow {
    comparison_id: String,
    candidate_id: String,
    baseline_id: String,
    independent_clusters: usize,
    mean_delta: f64,
    median_delta: f64,
    ci_lower: f64,
    ci_upper: f64,
    positive_clusters: usize,
    negative_clusters: usize,
    tie_clusters: usize,
    min_delta: f64,
    max_delta: f64,
    sign_flip_kind: String,
    sign_flip_p: f64,
    adjusted_p: f64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn validate_token(value: &str, field: &str) -> Result<(), String> {
    if value.is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':'))
    {
        return Err(invalid(format!(
            "{field} must be a non-empty identifier token without whitespace or paths"
        )));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<String, String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(invalid(format!(
            "{field} must be a 64-character hexadecimal SHA-256 digest"
        )));
    }
    Ok(value.to_ascii_lowercase())
}

fn parse_table(input: &str) -> Result<Table, String> {
    let mut meta = BTreeMap::new();
    let mut rows = Vec::new();
    let mut found_header = false;

    for (index, line) in input.lines().enumerate() {
        let line_number = index + 1;
        if line.is_empty() {
            return Err(invalid(format!("line {line_number}: blank lines are forbidden")));
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first().copied() == Some("META") {
            if found_header {
                return Err(invalid(format!(
                    "line {line_number}: metadata must precede the data header"
                )));
            }
            if fields.len() != 3 || fields[1].is_empty() || fields[2].is_empty() {
                return Err(invalid(format!(
                    "line {line_number}: META requires exactly key and value"
                )));
            }
            if meta.insert(fields[1].to_owned(), fields[2].to_owned()).is_some() {
                return Err(invalid(format!(
                    "line {line_number}: duplicate metadata key {}",
                    fields[1]
                )));
            }
            continue;
        }
        if !found_header {
            if fields != DELTA_HEADER {
                return Err(invalid(format!(
                    "line {line_number}: exact DELTA header required; unknown/reordered columns fail closed"
                )));
            }
            found_header = true;
            continue;
        }
        if fields.first().copied() != Some("DELTA") || fields.len() != DELTA_HEADER.len() {
            return Err(invalid(format!(
                "line {line_number}: expected DELTA row with {} fields",
                DELTA_HEADER.len()
            )));
        }
        if fields.iter().any(|field| field.is_empty()) {
            return Err(invalid(format!("line {line_number}: empty fields are forbidden")));
        }
        rows.push(fields.iter().map(|field| (*field).to_owned()).collect());
    }

    if !found_header || rows.is_empty() {
        return Err(invalid("missing DELTA header or data rows"));
    }
    let expected: BTreeSet<&str> = META_KEYS.iter().copied().collect();
    let actual: BTreeSet<&str> = meta.keys().map(String::as_str).collect();
    if actual != expected {
        return Err(invalid("META key set mismatch: missing or unknown fields"));
    }
    if meta.get("format").map(String::as_str) != Some(FORMAT_VERSION) {
        return Err(invalid("unsupported cluster-delta format"));
    }
    if meta.get("positive_is_better").map(String::as_str) != Some("true") {
        return Err(invalid("positive_is_better must be the literal true"));
    }
    if !matches!(meta.get("split").map(String::as_str), Some("test" | "private_test")) {
        return Err(invalid("inference accepts only the predeclared test or private_test split"));
    }

    for key in ["experiment_id", "dataset_id", "dataset_revision", "cluster_unit", "primary_metric"] {
        validate_token(meta.get(key).map(String::as_str).unwrap_or(""), key)?;
    }
    let cluster_unit = meta.get("cluster_unit").map(String::as_str).unwrap_or("");
    if matches!(
        cluster_unit,
        "frame" | "image" | "crop" | "row" | "sample" | "timestamp" | "detection"
    ) {
        return Err(invalid(
            "cluster_unit is too granular; use the independent sequence/scene/session/episode unit",
        ));
    }
    validate_sha256(
        meta.get("bundle_lock_sha256").map(String::as_str).unwrap_or(""),
        "bundle_lock_sha256",
    )?;
    Ok(Table { meta, rows })
}

fn parse_config(meta: &BTreeMap<String, String>) -> Result<Config, String> {
    let family_size = meta["comparison_family_size"]
        .parse::<usize>()
        .map_err(|_| invalid("comparison_family_size must be a positive integer"))?;
    if family_size == 0 {
        return Err(invalid("comparison_family_size must be positive"));
    }
    let seed = meta["bootstrap_seed"]
        .parse::<u64>()
        .map_err(|_| invalid("bootstrap_seed must be a u64"))?;
    let bootstrap_replicates = meta["bootstrap_replicates"]
        .parse::<u64>()
        .map_err(|_| invalid("bootstrap_replicates must be an integer"))?;
    if !(MIN_BOOTSTRAP_REPLICATES..=MAX_BOOTSTRAP_REPLICATES).contains(&bootstrap_replicates) {
        return Err(invalid(format!(
            "bootstrap_replicates must be in {MIN_BOOTSTRAP_REPLICATES}..={MAX_BOOTSTRAP_REPLICATES}"
        )));
    }
    let familywise_confidence = meta["familywise_confidence"]
        .parse::<f64>()
        .map_err(|_| invalid("familywise_confidence must be numeric"))?;
    if !familywise_confidence.is_finite() || !(0.5..1.0).contains(&familywise_confidence) {
        return Err(invalid("familywise_confidence must be finite and in [0.5, 1)"));
    }
    Ok(Config {
        family_size,
        seed,
        bootstrap_replicates,
        familywise_confidence,
    })
}

fn parse_comparisons(rows: &[Vec<String>]) -> Result<BTreeMap<String, Comparison>, String> {
    let mut comparisons: BTreeMap<String, Comparison> = BTreeMap::new();

    for (index, row) in rows.iter().enumerate() {
        let row_number = index + 1;
        let comparison_id = &row[1];
        let candidate_id = &row[2];
        let baseline_id = &row[3];
        let cluster_id = &row[4];
        for (value, field) in [
            (comparison_id.as_str(), "comparison_id"),
            (candidate_id.as_str(), "candidate_id"),
            (baseline_id.as_str(), "baseline_id"),
            (cluster_id.as_str(), "cluster_id"),
        ] {
            validate_token(value, field)?;
        }
        if candidate_id == baseline_id {
            return Err(invalid(format!(
                "data row {row_number}: candidate_id and baseline_id must differ"
            )));
        }
        let delta = row[5].parse::<f64>().map_err(|_| {
            invalid(format!("data row {row_number}: delta must be numeric"))
        })?;
        if !delta.is_finite() {
            return Err(invalid(format!(
                "data row {row_number}: delta must be finite"
            )));
        }

        let comparison = comparisons.entry(comparison_id.clone()).or_insert_with(|| Comparison {
            candidate_id: candidate_id.clone(),
            baseline_id: baseline_id.clone(),
            deltas: BTreeMap::new(),
        });
        if comparison.candidate_id != *candidate_id || comparison.baseline_id != *baseline_id {
            return Err(invalid(format!(
                "comparison {comparison_id} changes candidate/baseline identity across rows"
            )));
        }
        if comparison.deltas.insert(cluster_id.clone(), delta).is_some() {
            return Err(invalid(format!(
                "duplicate cluster {cluster_id} in comparison {comparison_id}"
            )));
        }
    }

    if comparisons.is_empty() {
        return Err(invalid("at least one comparison is required"));
    }
    Ok(comparisons)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    }
}

#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn index(&mut self, len: usize) -> usize {
        ((self.next_u64() as u128 * len as u128) >> 64) as usize
    }
}

fn linear_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

fn adjusted_bootstrap_interval(
    values: &[f64],
    seed: u64,
    replicates: u64,
    family_size: usize,
    familywise_confidence: f64,
) -> (f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut means = Vec::with_capacity(replicates as usize);
    for _ in 0..replicates {
        let mut total = 0.0;
        for _ in 0..values.len() {
            total += values[rng.index(values.len())];
        }
        means.push(total / values.len() as f64);
    }
    means.sort_by(f64::total_cmp);
    // Bonferroni-adjust each two-sided interval across the frozen family.
    let family_alpha = 1.0 - familywise_confidence;
    let per_tail = family_alpha / (2.0 * family_size as f64);
    (
        linear_quantile(&means, per_tail),
        linear_quantile(&means, 1.0 - per_tail),
    )
}

fn sign_flip_test(values: &[f64], seed: u64, monte_carlo_replicates: u64) -> (String, f64) {
    let observed = mean(values).abs();
    let tolerance = 8.0 * f64::EPSILON * observed.max(1.0);
    let mut extreme = 0u64;

    if values.len() <= EXACT_SIGN_FLIP_MAX_CLUSTERS {
        let configurations = 1u64 << values.len();
        for mask in 0..configurations {
            let mut sum = 0.0;
            for (bit, value) in values.iter().enumerate() {
                sum += if (mask & (1u64 << bit)) == 0 { -*value } else { *value };
            }
            if (sum / values.len() as f64).abs() + tolerance >= observed {
                extreme += 1;
            }
        }
        ("exact_sign_flip_symmetric_null".to_owned(), extreme as f64 / configurations as f64)
    } else {
        let mut rng = SplitMix64::new(seed);
        for _ in 0..monte_carlo_replicates {
            let mut sum = 0.0;
            for value in values {
                sum += if (rng.next_u64() & 1) == 0 { -*value } else { *value };
            }
            if (sum / values.len() as f64).abs() + tolerance >= observed {
                extreme += 1;
            }
        }
        let p = (extreme as f64 + 1.0) / (monte_carlo_replicates as f64 + 1.0);
        ("monte_carlo_sign_flip_symmetric_null".to_owned(), p)
    }
}

fn stable_seed(base: u64, comparison_id: &str) -> u64 {
    let mut seed = base ^ 0x6a09e667f3bcc909;
    for byte in comparison_id.bytes() {
        seed = seed.rotate_left(5) ^ u64::from(byte);
        seed = seed.wrapping_mul(0x9e3779b185ebca87);
    }
    seed
}

fn analyze_one(
    comparison_id: &str,
    comparison: &Comparison,
    ordered_clusters: &[String],
    config: Config,
) -> ResultRow {
    let values: Vec<f64> = ordered_clusters
        .iter()
        .map(|cluster| comparison.deltas[cluster])
        .collect();
    let mean_delta = mean(&values);
    let (ci_lower, ci_upper) = adjusted_bootstrap_interval(
        &values,
        stable_seed(config.seed, comparison_id),
        config.bootstrap_replicates,
        config.family_size,
        config.familywise_confidence,
    );
    let (sign_flip_kind, sign_flip_p) = sign_flip_test(
        &values,
        stable_seed(config.seed ^ 0xa5a5a5a5a5a5a5a5, comparison_id),
        config.bootstrap_replicates,
    );
    let positive_clusters = values.iter().filter(|value| **value > 0.0).count();
    let negative_clusters = values.iter().filter(|value| **value < 0.0).count();
    let tie_clusters = values.len() - positive_clusters - negative_clusters;
    ResultRow {
        comparison_id: comparison_id.to_owned(),
        candidate_id: comparison.candidate_id.clone(),
        baseline_id: comparison.baseline_id.clone(),
        independent_clusters: values.len(),
        mean_delta,
        median_delta: median(&values),
        ci_lower,
        ci_upper,
        positive_clusters,
        negative_clusters,
        tie_clusters,
        min_delta: values.iter().copied().min_by(f64::total_cmp).unwrap_or(0.0),
        max_delta: values.iter().copied().max_by(f64::total_cmp).unwrap_or(0.0),
        sign_flip_kind,
        sign_flip_p,
        adjusted_p: (sign_flip_p * config.family_size as f64).min(1.0),
    }
}

fn fmt_number(value: f64) -> String {
    format!("{value:.10}")
}

fn render_receipt(
    table: &Table,
    config: Config,
    results: &[ResultRow],
    cluster_count: usize,
) -> String {
    let mut output = String::new();
    let _ = writeln!(output, "ANALYSIS\tformat\t{RECEIPT_VERSION}");
    let _ = writeln!(output, "ANALYSIS\tstatus\tCLUSTER_LEVEL_INFERENCE_ONLY");
    for key in [
        "experiment_id",
        "dataset_id",
        "dataset_revision",
        "split",
        "cluster_unit",
        "primary_metric",
        "bundle_lock_sha256",
    ] {
        let _ = writeln!(output, "ANALYSIS\t{key}\t{}", table.meta[key]);
    }
    let _ = writeln!(output, "ANALYSIS\tindependent_clusters\t{cluster_count}");
    let _ = writeln!(output, "ANALYSIS\tcomparison_family_size\t{}", config.family_size);
    let _ = writeln!(output, "ANALYSIS\tbootstrap_seed\t{}", config.seed);
    let _ = writeln!(output, "ANALYSIS\tbootstrap_replicates\t{}", config.bootstrap_replicates);
    let _ = writeln!(
        output,
        "ANALYSIS\tfamilywise_confidence\t{}",
        fmt_number(config.familywise_confidence)
    );
    let _ = writeln!(
        output,
        "ANALYSIS\tinterval_method\tBONFERRONI_ADJUSTED_PERCENTILE_CLUSTER_BOOTSTRAP_APPROXIMATE"
    );
    let _ = writeln!(
        output,
        "ANALYSIS\tsign_flip_assumption\tINDEPENDENT_CLUSTER_DELTAS_SYMMETRIC_UNDER_NULL"
    );
    let _ = writeln!(output, "ANALYSIS\tframes_are_independent_units\tfalse");
    let _ = writeln!(output, "ANALYSIS\tsafety_certificate\tNOT_COMPUTED_BY_THIS_TOOL");
    let _ = writeln!(output, "ANALYSIS\tperformance_claim\tNONE");
    let _ = writeln!(
        output,
        "RESULT_HEADER\tcomparison_id\tcandidate_id\tbaseline_id\tindependent_clusters\tmean_delta\tmedian_delta\tci_lower\tci_upper\tpositive_clusters\tnegative_clusters\ttie_clusters\tmin_delta\tmax_delta\tsign_flip_kind\tsign_flip_p\tbonferroni_adjusted_p"
    );
    for result in results {
        let _ = writeln!(
            output,
            "RESULT\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            result.comparison_id,
            result.candidate_id,
            result.baseline_id,
            result.independent_clusters,
            fmt_number(result.mean_delta),
            fmt_number(result.median_delta),
            fmt_number(result.ci_lower),
            fmt_number(result.ci_upper),
            result.positive_clusters,
            result.negative_clusters,
            result.tie_clusters,
            fmt_number(result.min_delta),
            fmt_number(result.max_delta),
            result.sign_flip_kind,
            fmt_number(result.sign_flip_p),
            fmt_number(result.adjusted_p)
        );
    }
    output
}

fn analyze(input: &str, expected_bundle_lock_sha256: &str) -> Result<String, String> {
    let expected_lock = validate_sha256(expected_bundle_lock_sha256, "expected bundle-lock digest")?;
    let table = parse_table(input)?;
    let actual_lock = validate_sha256(&table.meta["bundle_lock_sha256"], "bundle_lock_sha256")?;
    if actual_lock != expected_lock {
        return Err(invalid(
            "cluster-delta metadata does not match independently supplied expected bundle-lock digest",
        ));
    }
    let config = parse_config(&table.meta)?;
    let comparisons = parse_comparisons(&table.rows)?;
    if comparisons.len() != config.family_size {
        return Err(invalid(format!(
            "observed {} comparisons but preregistered family size is {}",
            comparisons.len(),
            config.family_size
        )));
    }

    let first_clusters: BTreeSet<String> = comparisons
        .values()
        .next()
        .expect("non-empty comparisons")
        .deltas
        .keys()
        .cloned()
        .collect();
    if first_clusters.len() < MIN_CLUSTERS {
        return Err(invalid(format!(
            "at least {MIN_CLUSTERS} independent clusters are required for this reference analysis"
        )));
    }
    for (comparison_id, comparison) in &comparisons {
        let cluster_ids: BTreeSet<String> = comparison.deltas.keys().cloned().collect();
        if cluster_ids != first_clusters {
            return Err(invalid(format!(
                "comparison {comparison_id} does not contain the exact shared cluster set; missing/extra clusters fail closed"
            )));
        }
    }

    let cluster_ids: Vec<String> = first_clusters.into_iter().collect();
    let results: Vec<ResultRow> = comparisons
        .iter()
        .map(|(comparison_id, comparison)| analyze_one(comparison_id, comparison, &cluster_ids, config))
        .collect();
    Ok(render_receipt(&table, config, &results, cluster_ids.len()))
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() != 3 {
        return Err(
            "usage: vision-dvpc-cluster-inference DELTAS.tsv EXPECTED_BUNDLE_LOCK_SHA256".to_owned(),
        );
    }
    let input = fs::read_to_string(&args[1])
        .map_err(|error| invalid(format!("cannot read delta table: {error}")))?;
    analyze(&input, &args[2])
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(receipt) => print!("{receipt}"),
        Err(error) => {
            eprintln!("REJECT\treason\t{}", error.replace('\t', " ").replace('\n', " "));
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOCK: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn fixture() -> String {
        let mut content = String::from(
            "META\tformat\tVIS-DVPC-CLUSTER-DELTAS-001-v1\n\
META\texperiment_id\texperiment-2026-01\n\
META\tdataset_id\tsynthetic-fixture\n\
META\tdataset_revision\trev1\n\
META\tsplit\ttest\n\
META\tcluster_unit\tsequence\n\
META\tprimary_metric\tdecision_utility\n\
META\tpositive_is_better\ttrue\n\
META\tbundle_lock_sha256\taaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
META\tcomparison_family_size\t2\n\
META\tbootstrap_seed\t42\n\
META\tbootstrap_replicates\t2000\n\
META\tfamilywise_confidence\t0.95\n\
DELTA_HEADER\tcomparison_id\tcandidate_id\tbaseline_id\tcluster_id\tdelta\n",
        );
        for i in 0..8 {
            let first = (i as f64 + 1.0) / 100.0;
            let second = if i % 2 == 0 { 0.03 } else { -0.01 };
            content.push_str(&format!(
                "DELTA\tcandidate-vs-cheap\tcandidate\tall-cheap\tseq-{i:03}\t{first:.4}\n"
            ));
            content.push_str(&format!(
                "DELTA\tcandidate-vs-random\tcandidate\trandom\tseq-{i:03}\t{second:.4}\n"
            ));
        }
        content
    }

    #[test]
    fn analyzes_paired_cluster_deltas_and_emits_non_claims() {
        let result = analyze(&fixture(), LOCK).expect("valid fixture");
        assert!(result.contains("ANALYSIS\tstatus\tCLUSTER_LEVEL_INFERENCE_ONLY"));
        assert!(result.contains("ANALYSIS\tinterval_method\tBONFERRONI_ADJUSTED_PERCENTILE_CLUSTER_BOOTSTRAP_APPROXIMATE"));
        assert!(result.contains("ANALYSIS\tframes_are_independent_units\tfalse"));
        assert!(result.contains("ANALYSIS\tperformance_claim\tNONE"));
        assert!(result.contains("RESULT\tcandidate-vs-cheap\tcandidate\tall-cheap\t8"));
        assert!(result.contains("RESULT\tcandidate-vs-random\tcandidate\trandom\t8"));
    }

    #[test]
    fn receipt_is_deterministic_under_exact_replay() {
        assert_eq!(
            analyze(&fixture(), LOCK).unwrap(),
            analyze(&fixture(), LOCK).unwrap()
        );
    }

    #[test]
    fn reordered_rows_do_not_change_analysis() {
        let source = fixture();
        let mut lines: Vec<&str> = source.lines().collect();
        let header_index = lines.iter().position(|line| line.starts_with("DELTA_HEADER")).unwrap();
        let meta_and_header = lines[..=header_index].join("\n") + "\n";
        let mut rows: Vec<&str> = lines[header_index + 1..].to_vec();
        rows.reverse();
        let reversed = meta_and_header + &rows.join("\n") + "\n";
        assert_eq!(analyze(&source, LOCK).unwrap(), analyze(&reversed, LOCK).unwrap());
    }

    #[test]
    fn duplicate_cluster_comparison_pair_fails() {
        let mut broken = fixture();
        let row = broken
            .lines()
            .find(|line| line.starts_with("DELTA\tcandidate-vs-cheap"))
            .unwrap()
            .to_owned();
        broken.push_str(&row);
        broken.push('\n');
        assert!(analyze(&broken, LOCK).unwrap_err().contains("duplicate cluster"));
    }

    #[test]
    fn changed_bundle_lock_fails_against_external_expected_digest() {
        let broken = fixture().replace(LOCK, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        assert!(analyze(&broken, LOCK).unwrap_err().contains("expected bundle-lock digest"));
    }

    #[test]
    fn frame_level_clusters_are_rejected_as_pseudoreplication() {
        let broken = fixture().replace("META\tcluster_unit\tsequence", "META\tcluster_unit\tframe");
        assert!(analyze(&broken, LOCK).unwrap_err().contains("too granular"));
    }

    #[test]
    fn nonfinite_delta_fails_closed() {
        let broken = fixture().replace("\t0.0100\n", "\tNaN\n");
        assert!(analyze(&broken, LOCK).unwrap_err().contains("must be finite"));
        let broken_inf = fixture().replace("\t0.0100\n", "\tinf\n");
        assert!(analyze(&broken_inf, LOCK).unwrap_err().contains("must be finite"));
    }

    #[test]
    fn missing_cluster_in_comparison_fails() {
        let mut lines: Vec<&str> = fixture().lines().collect();
        let mut found = false;
        lines.retain(|line| {
            if !found && line.starts_with("DELTA\tcandidate-vs-random") && line.ends_with("seq-007\t-0.0100") {
                found = true;
                false
            } else {
                true
            }
        });
        assert!(found);
        let broken = lines.join("\n") + "\n";
        assert!(analyze(&broken, LOCK).unwrap_err().contains("exact shared cluster set"));
    }

    #[test]
    fn family_size_must_match_predeclared_comparisons() {
        let broken = fixture().replace("META\tcomparison_family_size\t2", "META\tcomparison_family_size\t3");
        assert!(analyze(&broken, LOCK).unwrap_err().contains("pre-registered family size"));
    }

    #[test]
    fn unknown_metadata_or_extra_columns_fail_closed() {
        let broken_meta = fixture().replace(
            "META\tpositive_is_better\ttrue",
            "META\tpositive_is_better\ttrue\nMETA\tposthoc_threshold\t0.0",
        );
        assert!(analyze(&broken_meta, LOCK).unwrap_err().contains("META key set mismatch"));
        let broken_header = fixture().replace(
            "DELTA_HEADER\tcomparison_id\tcandidate_id\tbaseline_id\tcluster_id\tdelta",
            "DELTA_HEADER\tcomparison_id\tcandidate_id\tbaseline_id\tcluster_id\tdelta\tp_value",
        );
        assert!(analyze(&broken_header, LOCK).unwrap_err().contains("exact DELTA header"));
    }

    #[test]
    fn unsupported_bootstrap_counts_and_confidence_fail_closed() {
        let too_few = fixture().replace("META\tbootstrap_replicates\t2000", "META\tbootstrap_replicates\t99");
        assert!(analyze(&too_few, LOCK).unwrap_err().contains("bootstrap_replicates must be in"));
        let bad_confidence = fixture().replace("META\tfamilywise_confidence\t0.95", "META\tfamilywise_confidence\t1.0");
        assert!(analyze(&bad_confidence, LOCK).unwrap_err().contains("familywise_confidence must be finite"));
    }

    #[test]
    fn insufficient_independent_clusters_fails_closed() {
        let mut lines: Vec<String> = fixture().lines().map(str::to_owned).collect();
        lines.retain(|line| !line.ends_with("seq-007\t0.0800") && !line.ends_with("seq-007\t-0.0100"));
        for line in &mut lines {
            // no other data mutation
            let _ = line;
        }
        // Keep four clusters by removing seq-006 too.
        lines.retain(|line| !line.contains("\tseq-006\t"));
        let broken = lines.join("\n") + "\n";
        assert!(analyze(&broken, LOCK).unwrap_err().contains("at least 5 independent clusters"));
    }

    #[test]
    fn comparison_identity_cannot_change_midstream() {
        let broken = fixture().replace(
            "DELTA\tcandidate-vs-cheap\tcandidate\tall-cheap\tseq-001",
            "DELTA\tcandidate-vs-cheap\tcandidate\tdifferent-baseline\tseq-001",
        );
        assert!(analyze(&broken, LOCK).unwrap_err().contains("changes candidate/baseline identity"));
    }

    #[test]
    fn tie_effect_has_unit_sign_flip_p_value() {
        let mut content = fixture();
        for i in 0..8 {
            let from = format!("seq-{i:03}\t{:.4}", (i as f64 + 1.0) / 100.0);
            let to = format!("seq-{i:03}\t0.0000");
            content = content.replace(&from, &to);
            let rfrom = format!(
                "seq-{i:03}\t{:.4}",
                if i % 2 == 0 { 0.03 } else { -0.01 }
            );
            let rto = format!("seq-{i:03}\t0.0000");
            content = content.replace(&rfrom, &rto);
        }
        let result = analyze(&content, LOCK).expect("zero deltas still valid data");
        assert!(result.contains("exact_sign_flip_symmetric_null\t1.0000000000"));
    }
}

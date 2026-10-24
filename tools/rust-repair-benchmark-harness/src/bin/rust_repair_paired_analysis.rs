use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;

const REPORT_VERSION: u32 = 1;
const DEFAULT_SEED: u64 = 20_261_024;
const DEFAULT_BOOTSTRAP_ITERATIONS: usize = 20_000;
const MAX_BOOTSTRAP_ITERATIONS: usize = 1_000_000;

#[derive(Debug)]
struct Cli {
    manifest: PathBuf,
    arm_a: PathBuf,
    arm_b: PathBuf,
    output: PathBuf,
    arm_a_label: String,
    arm_b_label: String,
    seed: u64,
    iterations: usize,
}

#[derive(Debug, Deserialize)]
struct TaskManifest {
    format_version: u32,
    protocol_id: String,
    candidate_jsonl_sha256: String,
    tasks: Vec<ManifestTask>,
}

#[derive(Debug, Deserialize)]
struct ManifestTask {
    instance_id: String,
    repo: String,
}

#[derive(Debug, Deserialize)]
struct ScoreReport {
    format_version: u32,
    protocol_id: String,
    task_manifest_sha256: String,
    candidate_jsonl_sha256: String,
    task_count: usize,
    tasks: Vec<ScoreTask>,
}

#[derive(Debug, Deserialize)]
struct ScoreTask {
    instance_id: String,
    outcome: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Pass,
    Fail,
    Unresolved,
}

impl Outcome {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "pass" => Ok(Self::Pass),
            "fail" => Ok(Self::Fail),
            "unresolved" => Ok(Self::Unresolved),
            other => Err(format!("unknown task outcome {other:?}; expected pass, fail, or unresolved")),
        }
    }

    fn binary(self) -> Option<i8> {
        match self {
            Self::Pass => Some(1),
            Self::Fail => Some(0),
            Self::Unresolved => None,
        }
    }
}

fn usage() -> &'static str {
    r#"Usage:
  cargo run --locked --manifest-path tools/rust-repair-benchmark-harness/Cargo.toml \
    --bin rust-repair-paired-analysis -- \
    --manifest <locked-task-manifest.json> \
    --arm-a <score-report.json> --arm-b <score-report.json> \
    --out <paired-analysis.json> [--arm-a-label NAME] [--arm-b-label NAME] \
    [--seed INTEGER] [--iterations INTEGER]

Compare two preregistered arms on the exact same locked tasks. Unresolved tasks
remain in the full-manifest sensitivity bounds and are never silently dropped."#
}

fn parse_cli() -> Result<Cli, String> {
    let mut args = env::args().skip(1);
    let mut manifest = None;
    let mut arm_a = None;
    let mut arm_b = None;
    let mut output = None;
    let mut arm_a_label = "arm_a".to_string();
    let mut arm_b_label = "arm_b".to_string();
    let mut seed = DEFAULT_SEED;
    let mut iterations = DEFAULT_BOOTSTRAP_ITERATIONS;

    while let Some(arg) = args.next() {
        let value = |args: &mut std::iter::Skip<std::env::Args>, flag: &str| {
            args.next().ok_or_else(|| format!("{flag} requires a value"))
        };
        match arg.as_str() {
            "--manifest" => manifest = Some(PathBuf::from(value(&mut args, "--manifest")?)),
            "--arm-a" => arm_a = Some(PathBuf::from(value(&mut args, "--arm-a")?)),
            "--arm-b" => arm_b = Some(PathBuf::from(value(&mut args, "--arm-b")?)),
            "--out" => output = Some(PathBuf::from(value(&mut args, "--out")?)),
            "--arm-a-label" => arm_a_label = value(&mut args, "--arm-a-label")?,
            "--arm-b-label" => arm_b_label = value(&mut args, "--arm-b-label")?,
            "--seed" => {
                seed = value(&mut args, "--seed")?
                    .parse()
                    .map_err(|_| "--seed must be an unsigned integer".to_string())?;
            }
            "--iterations" => {
                iterations = value(&mut args, "--iterations")?
                    .parse()
                    .map_err(|_| "--iterations must be an unsigned integer".to_string())?;
            }
            "--help" | "-h" => return Err(usage().to_string()),
            other => return Err(format!("unknown argument {other:?}\n\n{}", usage())),
        }
    }

    if !(1..=MAX_BOOTSTRAP_ITERATIONS).contains(&iterations) {
        return Err(format!("--iterations must be in 1..={MAX_BOOTSTRAP_ITERATIONS}"));
    }
    if arm_a_label.trim().is_empty() || arm_b_label.trim().is_empty() {
        return Err("arm labels must not be empty".to_string());
    }
    Ok(Cli {
        manifest: manifest.ok_or_else(|| format!("missing --manifest\n\n{}", usage()))?,
        arm_a: arm_a.ok_or_else(|| format!("missing --arm-a\n\n{}", usage()))?,
        arm_b: arm_b.ok_or_else(|| format!("missing --arm-b\n\n{}", usage()))?,
        output: output.ok_or_else(|| format!("missing --out\n\n{}", usage()))?,
        arm_a_label,
        arm_b_label,
        seed,
        iterations,
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &PathBuf) -> Result<(T, Vec<u8>), String> {
    let bytes = fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let parsed = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    Ok((parsed, bytes))
}

fn score_index(
    report: &ScoreReport,
    label: &str,
    manifest_sha256: &str,
    protocol_id: &str,
    candidate_sha256: &str,
    expected_ids: &BTreeSet<String>,
) -> Result<BTreeMap<String, Outcome>, String> {
    if report.format_version != REPORT_VERSION {
        return Err(format!("{label} score report format_version must be {REPORT_VERSION}"));
    }
    if report.protocol_id != protocol_id {
        return Err(format!("{label} protocol_id differs from the locked task manifest"));
    }
    if report.task_manifest_sha256 != manifest_sha256 {
        return Err(format!("{label} task_manifest_sha256 differs from the exact manifest bytes"));
    }
    if report.candidate_jsonl_sha256 != candidate_sha256 {
        return Err(format!("{label} candidate_jsonl_sha256 differs from the locked manifest"));
    }
    if report.task_count != expected_ids.len() || report.tasks.len() != expected_ids.len() {
        return Err(format!("{label} report task count does not equal the locked manifest count"));
    }

    let mut index = BTreeMap::new();
    for row in &report.tasks {
        if !expected_ids.contains(&row.instance_id) {
            return Err(format!("{label} contains task {:?} outside the locked manifest", row.instance_id));
        }
        let outcome = Outcome::parse(&row.outcome)
            .map_err(|error| format!("{label} task {:?}: {error}", row.instance_id))?;
        if index.insert(row.instance_id.clone(), outcome).is_some() {
            return Err(format!("{label} contains duplicate task {:?}", row.instance_id));
        }
    }
    if index.keys().cloned().collect::<BTreeSet<_>>() != *expected_ids {
        return Err(format!("{label} is missing one or more locked task IDs"));
    }
    Ok(index)
}

fn sensitivity_bounds(a: Outcome, b: Outcome) -> (i8, i8) {
    let a_low = a.binary().unwrap_or(0);
    let a_high = a.binary().unwrap_or(1);
    let b_low = b.binary().unwrap_or(0);
    let b_high = b.binary().unwrap_or(1);
    (a_low - b_high, a_high - b_low)
}

fn paired_delta(a: Outcome, b: Outcome) -> Option<i8> {
    Some(a.binary()? - b.binary()?)
}

#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (((self.next_u64() as u128) * (length as u128)) >> 64) as usize
    }
}

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = (probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted.get(index).copied()
}

fn cluster_bootstrap_ci(
    cluster_deltas: &[Vec<i8>],
    seed: u64,
    iterations: usize,
) -> Option<(f64, f64)> {
    let clusters = cluster_deltas
        .iter()
        .filter(|cluster| !cluster.is_empty())
        .collect::<Vec<_>>();
    if clusters.len() < 2 || iterations == 0 {
        return None;
    }

    let mut rng = SplitMix64::new(seed);
    let mut effects = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let mut delta_sum: i64 = 0;
        let mut task_count: usize = 0;
        for _ in 0..clusters.len() {
            let selected = clusters[rng.index(clusters.len())];
            delta_sum += selected.iter().map(|delta| i64::from(*delta)).sum::<i64>();
            task_count += selected.len();
        }
        if task_count > 0 {
            effects.push(100.0 * delta_sum as f64 / task_count as f64);
        }
    }
    effects.sort_by(f64::total_cmp);
    Some((
        percentile(&effects, 0.025)?,
        percentile(&effects, 0.975)?,
    ))
}

fn run() -> Result<(), String> {
    let cli = parse_cli()?;
    let (manifest, manifest_bytes): (TaskManifest, Vec<u8>) = read_json(&cli.manifest)?;
    let manifest_sha256 = sha256(&manifest_bytes);
    if manifest.format_version != REPORT_VERSION {
        return Err(format!(
            "manifest format_version must be {REPORT_VERSION}, got {}",
            manifest.format_version
        ));
    }
    if manifest.protocol_id.trim().is_empty() {
        return Err("manifest protocol_id must not be empty".to_string());
    }
    if !is_hex_digest(&manifest.candidate_jsonl_sha256) {
        return Err("manifest candidate_jsonl_sha256 must be a 64-character SHA-256 hex digest".to_string());
    }

    let mut expected_ids = BTreeSet::new();
    let mut repositories = BTreeMap::<String, String>::new();
    for task in &manifest.tasks {
        if task.instance_id.trim().is_empty() {
            return Err("manifest contains an empty instance_id".to_string());
        }
        if task.repo.trim().is_empty() {
            return Err(format!("manifest task {:?} has an empty repo", task.instance_id));
        }
        if !expected_ids.insert(task.instance_id.clone()) {
            return Err(format!("manifest contains duplicate task {:?}", task.instance_id));
        }
        repositories.insert(task.instance_id.clone(), task.repo.clone());
    }
    if expected_ids.is_empty() {
        return Err("locked task manifest has no tasks".to_string());
    }

    let (arm_a_report, arm_a_bytes): (ScoreReport, Vec<u8>) = read_json(&cli.arm_a)?;
    let (arm_b_report, arm_b_bytes): (ScoreReport, Vec<u8>) = read_json(&cli.arm_b)?;
    let arm_a = score_index(
        &arm_a_report,
        &cli.arm_a_label,
        &manifest_sha256,
        &manifest.protocol_id,
        &manifest.candidate_jsonl_sha256,
        &expected_ids,
    )?;
    let arm_b = score_index(
        &arm_b_report,
        &cli.arm_b_label,
        &manifest_sha256,
        &manifest.protocol_id,
        &manifest.candidate_jsonl_sha256,
        &expected_ids,
    )?;

    let mut complete_pairs = 0usize;
    let mut complete_a_successes = 0usize;
    let mut complete_b_successes = 0usize;
    let mut wins_a = 0usize;
    let mut wins_b = 0usize;
    let mut ties = 0usize;
    let mut unresolved_a = 0usize;
    let mut unresolved_b = 0usize;
    let mut bounds_lower_sum: i64 = 0;
    let mut bounds_upper_sum: i64 = 0;
    let mut repository_deltas: BTreeMap<String, Vec<i8>> = BTreeMap::new();

    for task_id in &expected_ids {
        let a = *arm_a.get(task_id).expect("validated arm A task set");
        let b = *arm_b.get(task_id).expect("validated arm B task set");
        if a == Outcome::Unresolved {
            unresolved_a += 1;
        }
        if b == Outcome::Unresolved {
            unresolved_b += 1;
        }
        let (low, high) = sensitivity_bounds(a, b);
        bounds_lower_sum += i64::from(low);
        bounds_upper_sum += i64::from(high);

        if let Some(delta) = paired_delta(a, b) {
            complete_pairs += 1;
            complete_a_successes += if a == Outcome::Pass { 1 } else { 0 };
            complete_b_successes += if b == Outcome::Pass { 1 } else { 0 };
            match delta.cmp(&0) {
                std::cmp::Ordering::Greater => wins_a += 1,
                std::cmp::Ordering::Less => wins_b += 1,
                std::cmp::Ordering::Equal => ties += 1,
            }
            repository_deltas
                .entry(repositories.get(task_id).expect("validated manifest repo").clone())
                .or_default()
                .push(delta);
        }
    }

    let cluster_count = repository_deltas.len();
    let cluster_deltas = repository_deltas.values().cloned().collect::<Vec<_>>();
    let ci = cluster_bootstrap_ci(&cluster_deltas, cli.seed, cli.iterations);
    let complete_effect = if complete_pairs > 0 {
        Some(
            100.0 * (complete_a_successes as f64 - complete_b_successes as f64)
                / complete_pairs as f64,
        )
    } else {
        None
    };
    let total_tasks = expected_ids.len();
    let lower_bound = 100.0 * bounds_lower_sum as f64 / total_tasks as f64;
    let upper_bound = 100.0 * bounds_upper_sum as f64 / total_tasks as f64;
    let ci_excludes_zero = ci.map(|(lower, upper)| lower > 0.0 || upper < 0.0);

    let report: Value = json!({
        "format_version": REPORT_VERSION,
        "analysis_protocol_id": "symthaea-rust-repair-paired-analysis",
        "source": {
            "task_manifest_sha256": manifest_sha256,
            "candidate_jsonl_sha256": manifest.candidate_jsonl_sha256,
            "arm_a_score_report_sha256": sha256(&arm_a_bytes),
            "arm_b_score_report_sha256": sha256(&arm_b_bytes),
            "arm_a_label": cli.arm_a_label,
            "arm_b_label": cli.arm_b_label,
            "bootstrap_seed": cli.seed,
            "bootstrap_iterations": cli.iterations
        },
        "task_count": total_tasks,
        "repository_count": repositories.values().collect::<BTreeSet<_>>().len(),
        "complete_pair_count": complete_pairs,
        "complete_pair_repository_count": cluster_count,
        "arm_a_unresolved_count": unresolved_a,
        "arm_b_unresolved_count": unresolved_b,
        "complete_pair": {
            "arm_a_successes": complete_a_successes,
            "arm_b_successes": complete_b_successes,
            "arm_a_wins": wins_a,
            "arm_b_wins": wins_b,
            "ties": ties,
            "arm_a_pass_rate": if complete_pairs > 0 { Some(complete_a_successes as f64 / complete_pairs as f64) } else { None::<f64> },
            "arm_b_pass_rate": if complete_pairs > 0 { Some(complete_b_successes as f64 / complete_pairs as f64) } else { None::<f64> },
            "arm_a_minus_arm_b_percentage_points": complete_effect
        },
        "repository_cluster_bootstrap_95_percent_interval": {
            "status": if cluster_count < 2 { "insufficient_repository_clusters" } else { "computed" },
            "lower_percentage_points": ci.map(|value| value.0),
            "upper_percentage_points": ci.map(|value| value.1),
            "excludes_zero": ci_excludes_zero,
            "cluster_count_below_proposed_20_cluster_gate": cluster_count < 20
        },
        "full_locked_manifest_worst_case_effect_bounds_percentage_points": {
            "lower": lower_bound,
            "upper": upper_bound,
            "includes_unresolved_rows": true
        },
        "interpretation": "The complete-pair effect is diagnostic and excludes unresolved pairs. The full-manifest bounds retain every assigned task. The bootstrap resamples repositories, not individual tasks. This report is not by itself a leaderboard or state-of-the-art claim.",
        "decision_notes": [
            if unresolved_a > 0 || unresolved_b > 0 { "Unresolved outcomes remain in the full-manifest sensitivity bounds; investigate asymmetric missingness before making a causal claim." } else { "Both arms have resolved outcomes for every locked task." },
            if cluster_count < 20 { "Fewer than 20 repositories contributed complete pairs; treat the cluster-bootstrap interval as exploratory, not decision-grade." } else { "At least 20 repositories contributed complete pairs; still inspect task-family coverage and the preregistered analysis plan." },
            if ci_excludes_zero == Some(true) { "The repository-cluster bootstrap interval excludes zero; this is a statistical signal, not proof of generalization or SOTA." } else { "The repository-cluster bootstrap interval does not establish a nonzero effect, or could not be computed." }
        ]
    });

    if let Some(parent) = cli.output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create output directory {}: {error}", parent.display()))?;
    }
    fs::write(&cli.output, serde_json::to_vec_pretty(&report).map_err(|error| error.to_string())?)
        .map_err(|error| format!("cannot write {}: {error}", cli.output.display()))?;

    println!(
        "paired analysis written to {}; tasks={}, complete_pairs={}, A_unresolved={}, B_unresolved={}, complete_pair_effect_pp={:?}",
        cli.output.display(),
        total_tasks,
        complete_pairs,
        unresolved_a,
        unresolved_b,
        complete_effect
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rust-repair-paired-analysis: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_delta_preserves_direction() {
        assert_eq!(paired_delta(Outcome::Pass, Outcome::Fail), Some(1));
        assert_eq!(paired_delta(Outcome::Fail, Outcome::Pass), Some(-1));
        assert_eq!(paired_delta(Outcome::Pass, Outcome::Pass), Some(0));
    }

    #[test]
    fn unresolved_pairs_are_not_complete_pairs() {
        assert_eq!(paired_delta(Outcome::Unresolved, Outcome::Pass), None);
        assert_eq!(paired_delta(Outcome::Fail, Outcome::Unresolved), None);
    }

    #[test]
    fn unresolved_outcomes_expand_full_manifest_bounds() {
        assert_eq!(
            sensitivity_bounds(Outcome::Unresolved, Outcome::Unresolved),
            (-1, 1)
        );
        assert_eq!(
            sensitivity_bounds(Outcome::Pass, Outcome::Unresolved),
            (0, 1)
        );
        assert_eq!(
            sensitivity_bounds(Outcome::Unresolved, Outcome::Fail),
            (0, 1)
        );
    }

    #[test]
    fn resolved_outcomes_have_exact_sensitivity_bounds() {
        assert_eq!(sensitivity_bounds(Outcome::Pass, Outcome::Fail), (1, 1));
        assert_eq!(sensitivity_bounds(Outcome::Fail, Outcome::Pass), (-1, -1));
        assert_eq!(sensitivity_bounds(Outcome::Fail, Outcome::Fail), (0, 0));
    }

    #[test]
    fn outcome_parser_rejects_unknown_states() {
        assert!(Outcome::parse("skipped").is_err());
        assert!(Outcome::parse("not-run").is_err());
    }

    #[test]
    fn splitmix_is_repeatable_for_a_fixed_seed() {
        let mut left = SplitMix64::new(42);
        let mut right = SplitMix64::new(42);
        for _ in 0..20 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }

    #[test]
    fn bootstrap_interval_is_repeatable_and_positive_for_uniform_wins() {
        let groups = vec![vec![1, 1], vec![1], vec![1, 1, 1]];
        let first = cluster_bootstrap_ci(&groups, 7, 500).unwrap();
        let second = cluster_bootstrap_ci(&groups, 7, 500).unwrap();
        assert_eq!(first, second);
        assert_eq!(first, (100.0, 100.0));
    }

    #[test]
    fn bootstrap_requires_two_repositories_with_complete_pairs() {
        assert_eq!(cluster_bootstrap_ci(&[vec![1, -1]], 1, 100), None);
        assert!(cluster_bootstrap_ci(&[vec![1], vec![-1]], 1, 100).is_some());
    }

    #[test]
    fn percentile_uses_nearest_rank_index_deterministically() {
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 0.0), Some(1.0));
        assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 1.0), Some(4.0));
    }
}

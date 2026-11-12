//! Paired, dataset-clustered comparison for cognitive drift detector results.
//!
//! This reports uncertainty over paired evaluation records. It does not infer benchmark
//! superiority from missing runs and does not treat trials from one dataset as independent
//! datasets. Configuration and access-track mismatches fail closed.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub name: String,
    pub version: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub comparison_track: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    pub id: String,
    pub unit: String,
    pub higher_is_better: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BootstrapSpec {
    pub replicates: usize,
    pub seed: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PairedObservation {
    pub dataset_id: String,
    pub seed: u64,
    pub scenario_id: String,
    pub candidate_value: f64,
    pub baseline_value: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonInput {
    pub schema_version: u32,
    pub metric: MetricSpec,
    pub candidate: Arm,
    pub baseline: Arm,
    pub bootstrap: BootstrapSpec,
    pub pairs: Vec<PairedObservation>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DatasetSummary {
    pub dataset_id: String,
    pub paired_records: usize,
    pub candidate_mean: f64,
    pub baseline_mean: f64,
    /// Positive values always favor the candidate, including lower-is-better metrics.
    pub oriented_mean_delta: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ComparisonResult {
    pub result_schema_version: u32,
    pub metric_version: String,
    pub canonical_input_sha256: String,
    /// SHA-256 of this serialized result with this field set to null.
    pub result_sha256: Option<String>,
    pub metric_id: String,
    pub metric_unit: String,
    pub higher_is_better: bool,
    pub candidate_name: String,
    pub candidate_version: String,
    pub candidate_repository_head_sha: String,
    pub candidate_configuration_digest: String,
    pub baseline_name: String,
    pub baseline_version: String,
    pub baseline_repository_head_sha: String,
    pub baseline_configuration_digest: String,
    pub comparison_track: String,
    pub dataset_count: usize,
    pub minimum_seeds_per_dataset: usize,
    /// Coverage only: passing this does not establish superiority over state of the art.
    pub sota_coverage_gate_pass: bool,
    pub sota_coverage_gate_reason: String,
    pub paired_record_count: usize,
    pub bootstrap_method: String,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
    pub candidate_dataset_macro_mean: f64,
    pub baseline_dataset_macro_mean: f64,
    /// Positive values always favor candidate.
    pub dataset_macro_paired_effect: f64,
    pub median_oriented_paired_delta: f64,
    /// A tie contributes 0.5; computed over the matched paired records.
    pub paired_win_fraction: f64,
    pub percentile_95_ci_oriented_effect: [f64; 2],
    /// Descriptive bootstrap fraction, not a frequentist p-value.
    pub bootstrap_fraction_effect_above_zero: f64,
    pub interval_supports_positive_effect: bool,
    pub per_dataset: Vec<DatasetSummary>,
    pub interpretation_guard: String,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<ComparisonResult> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading paired comparison input {}", path.display()))?;
    let input: ComparisonInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing paired comparison input {}", path.display()))?;
    compare(input)
}

pub fn compare(mut input: ComparisonInput) -> anyhow::Result<ComparisonResult> {
    ensure!(input.schema_version == 1, "paired comparison schema_version must be 1");
    validate_arm(&input.candidate, "candidate")?;
    validate_arm(&input.baseline, "baseline")?;
    ensure!(
        input.candidate.comparison_track == input.baseline.comparison_track,
        "candidate and baseline comparison_track must match"
    );
    ensure!(!input.metric.id.trim().is_empty(), "metric.id must not be empty");
    ensure!(!input.metric.unit.trim().is_empty(), "metric.unit must not be empty");
    ensure!(
        (1_000..=100_000).contains(&input.bootstrap.replicates),
        "bootstrap.replicates must be between 1000 and 100000"
    );
    ensure!(!input.pairs.is_empty(), "pairs must not be empty");

    input.pairs.sort_by(|a, b| {
        (&a.dataset_id, &a.seed, &a.scenario_id)
            .cmp(&(&b.dataset_id, &b.seed, &b.scenario_id))
    });

    let mut keys = BTreeSet::new();
    let mut by_dataset: BTreeMap<String, Vec<(u64, String, f64, f64, f64)>> = BTreeMap::new();
    let mut by_dataset_seed: BTreeMap<String, BTreeMap<u64, Vec<(String, f64)>>> = BTreeMap::new();
    let mut scenario_sets: BTreeMap<String, BTreeMap<u64, BTreeSet<String>>> = BTreeMap::new();
    let mut all_oriented_deltas = Vec::with_capacity(input.pairs.len());
    let mut wins = 0.0f64;
    let orientation = if input.metric.higher_is_better { 1.0 } else { -1.0 };

    for (index, pair) in input.pairs.iter().enumerate() {
        ensure!(!pair.dataset_id.trim().is_empty(), "pairs[{index}].dataset_id must not be empty");
        ensure!(!pair.scenario_id.trim().is_empty(), "pairs[{index}].scenario_id must not be empty");
        ensure!(
            pair.candidate_value.is_finite() && pair.baseline_value.is_finite(),
            "pairs[{index}] candidate/baseline values must be finite"
        );
        ensure!(
            keys.insert((pair.dataset_id.clone(), pair.seed, pair.scenario_id.clone())),
            "duplicate paired record key at dataset={}, seed={}, scenario={}",
            pair.dataset_id,
            pair.seed,
            pair.scenario_id
        );
        let delta = (pair.candidate_value - pair.baseline_value) * orientation;
        ensure!(
            delta.is_finite(),
            "pairs[{index}] oriented paired difference overflowed to a non-finite value"
        );
        if delta > 0.0 {
            wins += 1.0;
        } else if delta == 0.0 {
            wins += 0.5;
        }
        all_oriented_deltas.push(delta);
        by_dataset
            .entry(pair.dataset_id.clone())
            .or_default()
            .push((pair.seed, pair.scenario_id.clone(), pair.candidate_value, pair.baseline_value, delta));
        by_dataset_seed
            .entry(pair.dataset_id.clone())
            .or_default()
            .entry(pair.seed)
            .or_default()
            .push((pair.scenario_id.clone(), delta));
        scenario_sets
            .entry(pair.dataset_id.clone())
            .or_default()
            .entry(pair.seed)
            .or_default()
            .insert(pair.scenario_id.clone());
    }

    ensure!(
        by_dataset.len() >= 3,
        "at least three independent dataset clusters are required"
    );
    let unique_seeds: BTreeSet<u64> = input.pairs.iter().map(|pair| pair.seed).collect();
    ensure!(unique_seeds.len() >= 2, "at least two distinct seeds are required");

    // Require a balanced scenario panel for every dataset and seed. Resampling must
    // not change the scenario mix while claiming a paired comparison.
    let mut global_scenario_set: Option<BTreeSet<String>> = None;
    for (dataset_id, seeds) in &scenario_sets {
        let mut iterator = seeds.values();
        let reference = iterator.next().context("dataset has no seed/scenario records")?;
        if let Some(global) = &global_scenario_set {
            ensure!(
                reference == global,
                "unbalanced scenario panel for dataset {dataset_id}; all datasets must include the same scenarios"
            );
        } else {
            global_scenario_set = Some(reference.clone());
        }
        for scenarios in iterator {
            ensure!(
                scenarios == reference,
                "unbalanced scenario panel for dataset {dataset_id}; every seed must include the same scenarios"
            );
        }
    }

    let dataset_ids: Vec<String> = by_dataset.keys().cloned().collect();
    let per_dataset: Vec<DatasetSummary> = by_dataset
        .iter()
        .map(|(dataset_id, records)| {
            DatasetSummary {
                dataset_id: dataset_id.clone(),
                paired_records: records.len(),
                candidate_mean: mean_values(records.iter().map(|x| x.2), records.len()),
                baseline_mean: mean_values(records.iter().map(|x| x.3), records.len()),
                oriented_mean_delta: mean_values(records.iter().map(|x| x.4), records.len()),
            }
        })
        .collect();

    let dataset_count = per_dataset.len();
    let minimum_seeds_per_dataset = by_dataset_seed.values().map(|seeds| seeds.len()).min().unwrap_or(0);
    let sota_coverage_gate_pass = dataset_count >= 7 && minimum_seeds_per_dataset >= 50;
    let sota_coverage_gate_reason = if sota_coverage_gate_pass {
        "Coverage requirement met (at least seven datasets and 50 seeds per dataset); comparative evidence, uncertainty, costs, and baseline fairness still require review.".to_owned()
    } else {
        format!(
            "Not eligible for a state-of-the-art claim: need at least seven datasets and 50 seeds per dataset; observed {dataset_count} datasets and minimum {minimum_seeds_per_dataset} seeds per dataset. This gate measures coverage only."
        )
    };
    let candidate_dataset_macro_mean =
        mean_values(per_dataset.iter().map(|item| item.candidate_mean), dataset_count);
    let baseline_dataset_macro_mean =
        mean_values(per_dataset.iter().map(|item| item.baseline_mean), dataset_count);
    let dataset_macro_paired_effect =
        mean_values(per_dataset.iter().map(|item| item.oriented_mean_delta), dataset_count);

    let mut sorted_deltas = all_oriented_deltas;
    sorted_deltas.sort_by(f64::total_cmp);
    let median_oriented_paired_delta = quantile_sorted(&sorted_deltas, 0.5);
    let paired_win_fraction = wins / input.pairs.len() as f64;

    let canonical_input = serde_json::to_vec(&input)?;
    let canonical_input_sha256 = sha256_hex(&canonical_input);
    let mut rng = SplitMix64::new(input.bootstrap.seed);
    let mut bootstrap_effects = Vec::with_capacity(input.bootstrap.replicates);

    // Hierarchical paired bootstrap: sample datasets, then seeds within each sampled
    // dataset. Keep each seed's entire scenario block intact, preserving scenario pairing.
    // Each sampled dataset receives equal final weight.
    for _ in 0..input.bootstrap.replicates {
        let mut sample_effect_sum = 0.0;
        for _ in 0..dataset_count {
            let sampled_dataset = &dataset_ids[rng.index(dataset_count)];
            let seeds = &by_dataset_seed[sampled_dataset];
            let seed_ids: Vec<u64> = seeds.keys().copied().collect();
            let mut dataset_seed_mean_sum = 0.0;
            for _ in 0..seed_ids.len() {
                let sampled_seed = seed_ids[rng.index(seed_ids.len())];
                let scenario_deltas = &seeds[&sampled_seed];
                let seed_mean = mean_values(
                    scenario_deltas.iter().map(|(_, delta)| *delta),
                    scenario_deltas.len(),
                );
                dataset_seed_mean_sum += seed_mean;
            }
            sample_effect_sum += dataset_seed_mean_sum / seed_ids.len() as f64;
        }
        bootstrap_effects.push(sample_effect_sum / dataset_count as f64);
    }
    bootstrap_effects.sort_by(f64::total_cmp);
    let ci = [
        quantile_sorted(&bootstrap_effects, 0.025),
        quantile_sorted(&bootstrap_effects, 0.975),
    ];
    let bootstrap_fraction_effect_above_zero =
        bootstrap_effects.iter().filter(|delta| **delta > 0.0).count() as f64
            / bootstrap_effects.len() as f64;

    let mut result = ComparisonResult {
        result_schema_version: 1,
        metric_version: "paired-dataset-bootstrap/v1".to_owned(),
        canonical_input_sha256,
        result_sha256: None,
        metric_id: input.metric.id,
        metric_unit: input.metric.unit,
        higher_is_better: input.metric.higher_is_better,
        candidate_name: input.candidate.name,
        candidate_version: input.candidate.version,
        candidate_repository_head_sha: input.candidate.repository_head_sha,
        candidate_configuration_digest: input.candidate.configuration_digest,
        baseline_name: input.baseline.name,
        baseline_version: input.baseline.version,
        baseline_repository_head_sha: input.baseline.repository_head_sha,
        baseline_configuration_digest: input.baseline.configuration_digest,
        comparison_track: input.candidate.comparison_track,
        dataset_count,
        minimum_seeds_per_dataset,
        sota_coverage_gate_pass,
        sota_coverage_gate_reason,
        paired_record_count: input.pairs.len(),
        bootstrap_method: "hierarchical-paired-bootstrap/dataset-then-seed-scenario-block/v1".to_owned(),
        bootstrap_replicates: input.bootstrap.replicates,
        bootstrap_seed: input.bootstrap.seed,
        candidate_dataset_macro_mean,
        baseline_dataset_macro_mean,
        dataset_macro_paired_effect,
        median_oriented_paired_delta,
        paired_win_fraction,
        percentile_95_ci_oriented_effect: ci,
        bootstrap_fraction_effect_above_zero,
        interval_supports_positive_effect: ci[0] > 0.0,
        per_dataset,
        interpretation_guard: "Positive effects favor candidate after direction normalization. The bootstrap fraction is descriptive, not a p-value. A positive interval is not a universal SOTA claim; report per-dataset effects, costs, failure cases, and access-track limits.".to_owned(),
    };
    let result_payload = serde_json::to_vec(&result)?;
    result.result_sha256 = Some(sha256_hex(&result_payload));
    Ok(result)
}

fn validate_arm(arm: &Arm, label: &str) -> anyhow::Result<()> {
    ensure!(!arm.name.trim().is_empty(), "{label}.name must not be empty");
    ensure!(!arm.version.trim().is_empty(), "{label}.version must not be empty");
    ensure!(
        arm.repository_head_sha.len() == 40 || arm.repository_head_sha.len() == 64,
        "{label}.repository_head_sha must be a full 40- or 64-character commit hash"
    );
    ensure!(
        arm.repository_head_sha.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label}.repository_head_sha must be hexadecimal"
    );
    ensure!(
        arm.configuration_digest.len() == 64
            && arm.configuration_digest.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label}.configuration_digest must be a 64-character hexadecimal digest"
    );
    ensure!(
        ["valid-pe-only", "feature-only-unsupervised", "auxiliary-loss-unsupervised", "observed-outcome"]
            .contains(&arm.comparison_track.as_str()),
        "{label}.comparison_track is not a declared evidence-access track"
    );
    Ok(())
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
    }
}

fn mean_values(values: impl Iterator<Item = f64>, count: usize) -> f64 {
    let divisor = count as f64;
    values.map(|value| value / divisor).sum()
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn index(&mut self, upper: usize) -> usize {
        (((self.next_u64() as u128) * (upper as u128)) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(higher_is_better: bool) -> ComparisonInput {
        let mut pairs = Vec::new();
        for dataset in ["dataset-a", "dataset-b", "dataset-c"] {
            for seed in [11, 23, 37] {
                for (scenario, offset) in [("abrupt", 0.0), ("gradual", 0.1)] {
                    let baseline_value = 0.5 + offset;
                    let candidate_value = if higher_is_better {
                        baseline_value + 0.1
                    } else {
                        baseline_value - 0.1
                    };
                    pairs.push(PairedObservation {
                        dataset_id: dataset.to_owned(),
                        seed,
                        scenario_id: scenario.to_owned(),
                        candidate_value,
                        baseline_value,
                    });
                }
            }
        }
        ComparisonInput {
            schema_version: 1,
            metric: MetricSpec {
                id: if higher_is_better { "event-f1" } else { "p95-latency" }.to_owned(),
                unit: if higher_is_better { "fraction" } else { "milliseconds" }.to_owned(),
                higher_is_better,
            },
            candidate: Arm {
                name: "candidate".to_owned(),
                version: "test-candidate-1".to_owned(),
                repository_head_sha: "a".repeat(40),
                configuration_digest: "b".repeat(64),
                comparison_track: "valid-pe-only".to_owned(),
            },
            baseline: Arm {
                name: "baseline".to_owned(),
                version: "test-baseline-1".to_owned(),
                repository_head_sha: "c".repeat(40),
                configuration_digest: "d".repeat(64),
                comparison_track: "valid-pe-only".to_owned(),
            },
            bootstrap: BootstrapSpec {
                replicates: 1000,
                seed: 20261023,
            },
            pairs,
        }
    }

    #[test]
    fn positive_effect_and_interval_favor_candidate() {
        let result = compare(fixture(true)).expect("comparison succeeds");
        assert_eq!(result.dataset_count, 3);
        assert_eq!(result.paired_record_count, 18);
        assert!(!result.sota_coverage_gate_pass);
        assert!(result.sota_coverage_gate_reason.contains("seven datasets and 50 seeds"));
        assert!((result.dataset_macro_paired_effect - 0.1).abs() < 1e-12);
        assert!(result.percentile_95_ci_oriented_effect[0] > 0.0);
        assert!(result.interval_supports_positive_effect);
        assert_eq!(result.result_sha256.as_ref().map(String::len), Some(64));
        assert_eq!(result.canonical_input_sha256.len(), 64);
    }

    #[test]
    fn lower_is_better_metrics_orient_effect_consistently() {
        let result = compare(fixture(false)).expect("comparison succeeds");
        assert!((result.dataset_macro_paired_effect - 0.1).abs() < 1e-12);
        assert!(result.interval_supports_positive_effect);
    }

    #[test]
    fn canonical_input_digest_ignores_pair_order() {
        let first = fixture(true);
        let mut second = first.clone();
        second.pairs.reverse();
        let a = compare(first).expect("first comparison");
        let b = compare(second).expect("reordered comparison");
        assert_eq!(a.canonical_input_sha256, b.canonical_input_sha256);
        assert_eq!(a.result_sha256, b.result_sha256);
    }

    #[test]
    fn bootstrap_is_deterministic_for_pinned_seed() {
        let a = compare(fixture(true)).expect("first comparison");
        let b = compare(fixture(true)).expect("second comparison");
        assert_eq!(a.percentile_95_ci_oriented_effect, b.percentile_95_ci_oriented_effect);
        assert_eq!(a.canonical_input_sha256, b.canonical_input_sha256);
        assert_eq!(a.result_sha256, b.result_sha256);
    }

    #[test]
    fn rejects_mismatched_access_tracks() {
        let mut input = fixture(true);
        input.baseline.comparison_track = "observed-outcome".to_owned();
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_duplicate_paired_records() {
        let mut input = fixture(true);
        input.pairs.push(input.pairs[0].clone());
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_less_than_three_datasets() {
        let mut input = fixture(true);
        input.pairs.retain(|pair| pair.dataset_id != "dataset-c");
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_dataset_with_missing_scenario_panel() {
        let mut input = fixture(true);
        input.pairs.retain(|pair| {
            !(pair.dataset_id == "dataset-a" && pair.scenario_id == "gradual")
        });
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_difference_that_overflows_even_when_inputs_are_finite() {
        let mut input = fixture(true);
        input.pairs[0].candidate_value = f64::MAX;
        input.pairs[0].baseline_value = -f64::MAX;
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_non_finite_metrics() {
        let mut input = fixture(true);
        input.pairs[0].candidate_value = f64::NAN;
        assert!(compare(input).is_err());
    }

    #[test]
    fn rejects_unbalanced_scenario_panel_within_dataset() {
        let mut input = fixture(true);
        input.pairs.retain(|pair| {
            !(pair.dataset_id == "dataset-a" && pair.seed == 11 && pair.scenario_id == "gradual")
        });
        assert!(compare(input).is_err());
    }
}

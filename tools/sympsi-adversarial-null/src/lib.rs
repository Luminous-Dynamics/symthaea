#![forbid(unsafe_code)]

//! Synthetic adversarial-null benchmark for AIA measurement.
//!
//! This crate does not run an AI participant, ingest human data, or test psi.
//! It compares fixed 1/k scoring with conditional target-label randomization
//! while deliberately preserving response-stream preferences and dependence.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const TOOL_VERSION: &str = env!("CARGO_PKG_VERSION");
const DOMAIN: &str = "sympsi.adversarial-null.v0.1";
const INFLATED_FPR_FLOOR: f64 = 0.05;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: String,
    pub synthetic_only: bool,
    pub master_seed: u64,
    pub campaigns_per_scenario: u64,
    pub alpha: f64,
    pub confidence_z: f64,
    pub conditional_randomization_z_critical: f64,
    pub scenario_ids: Vec<String>,
    pub profiles: BTreeMap<String, ProfileSpec>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    pub choice_count: usize,
    pub analysis_trials: usize,
    pub naive_critical_hits: usize,
    pub block_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct Interval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FalsePositiveRate {
    pub campaigns: u64,
    pub rejections: u64,
    pub empirical_rate: f64,
    pub wilson_99: Interval,
    pub upper_limit_for_controlled_null: f64,
    pub disposition: String,
    pub passed_expected_disposition: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScenarioResult {
    pub scenario_id: String,
    pub null_mechanism: String,
    pub profile: String,
    pub choice_count: usize,
    pub analysis_trials: usize,
    pub campaigns: u64,
    pub target_response_independent_by_construction: bool,
    pub response_temporal_dependence: String,
    pub exchangeability_unit: String,
    pub naive_fixed_1_over_k: FalsePositiveRate,
    pub global_conditional_target_randomization: FalsePositiveRate,
    pub block_conditional_target_randomization: FalsePositiveRate,
}

#[derive(Clone, Debug, Serialize)]
pub struct ControlResult {
    pub id: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiptBody {
    pub receipt_schema_version: String,
    pub tool_version: String,
    pub git_head: String,
    pub synthetic_only: bool,
    pub measurement_class: String,
    pub claim_ceiling: String,
    pub manifest_sha256: String,
    pub master_seed: u64,
    pub campaigns_per_scenario: u64,
    pub alpha: f64,
    pub conditional_randomization_z_critical: f64,
    pub scenarios: Vec<ScenarioResult>,
    pub controls: Vec<ControlResult>,
    pub all_controls_passed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiptEnvelope {
    pub receipt: ReceiptBody,
    pub receipt_sha256: String,
}

#[derive(Clone, Copy)]
struct ScenarioSpec {
    id: &'static str,
    mechanism: &'static str,
    response_dependence: &'static str,
    exchangeability: &'static str,
    naive_expected: ExpectedDisposition,
    global_expected: ExpectedDisposition,
    block_expected: ExpectedDisposition,
}

#[derive(Clone, Copy)]
enum ExpectedDisposition {
    Controlled,
    Inflated,
}

const SCENARIOS: [ScenarioSpec; 5] = [
    ScenarioSpec {
        id: "uniform_iid",
        mechanism: "Targets and responses are independent uniform categorical draws.",
        response_dependence: "iid",
        exchangeability: "global",
        naive_expected: ExpectedDisposition::Controlled,
        global_expected: ExpectedDisposition::Controlled,
        block_expected: ExpectedDisposition::Controlled,
    },
    ScenarioSpec {
        id: "static_marginal_alignment",
        mechanism: "Targets and responses are independently drawn from the same fixed nonuniform label distribution.",
        response_dependence: "iid with fixed label preference",
        exchangeability: "global",
        naive_expected: ExpectedDisposition::Inflated,
        global_expected: ExpectedDisposition::Controlled,
        block_expected: ExpectedDisposition::Controlled,
    },
    ScenarioSpec {
        id: "target_imbalance_uniform_response",
        mechanism: "Targets have a nonuniform schedule while responses remain uniform and target-independent.",
        response_dependence: "iid uniform responses",
        exchangeability: "global",
        naive_expected: ExpectedDisposition::Controlled,
        global_expected: ExpectedDisposition::Controlled,
        block_expected: ExpectedDisposition::Controlled,
    },
    ScenarioSpec {
        id: "persistent_response_preference",
        mechanism: "Targets are iid uniform; responses have a target-independent Markov preference with persistent runs.",
        response_dependence: "Markov persistence 0.80 with nonuniform stationary preference",
        exchangeability: "global",
        naive_expected: ExpectedDisposition::Controlled,
        global_expected: ExpectedDisposition::Controlled,
        block_expected: ExpectedDisposition::Controlled,
    },
    ScenarioSpec {
        id: "blockwise_shared_drift",
        mechanism: "Within each preregistered block, targets and responses are independently drawn from the same block-specific nonuniform distribution; the dominant label rotates between blocks.",
        response_dependence: "within-block Markov persistence 0.80 plus block-level drift",
        exchangeability: "within each of 10 preregistered blocks",
        naive_expected: ExpectedDisposition::Inflated,
        global_expected: ExpectedDisposition::Inflated,
        block_expected: ExpectedDisposition::Controlled,
    },
];

#[derive(Clone)]
struct CampaignCounts {
    global_target: Vec<u64>,
    global_response: Vec<u64>,
    global_hits: u64,
    block_target: Vec<Vec<u64>>,
    block_response: Vec<Vec<u64>>,
    block_hits: Vec<u64>,
}

pub fn run_benchmark(
    git_head: &str,
    manifest_bytes: &[u8],
) -> Result<ReceiptEnvelope, String> {
    validate_git_head(git_head)?;
    let manifest: Manifest = serde_json::from_slice(manifest_bytes)
        .map_err(|error| format!("manifest parse failed: {error}"))?;
    validate_manifest(&manifest)?;

    let manifest_sha256 = sha256_hex(manifest_bytes);
    let mut scenarios = Vec::new();
    let mut controls = Vec::new();

    for profile_name in ["four-choice", "five-choice"] {
        let profile = manifest
            .profiles
            .get(profile_name)
            .ok_or_else(|| format!("missing profile {profile_name}"))?;

        for spec in SCENARIOS {
            let result = simulate_scenario(
                spec,
                profile_name,
                profile,
                &manifest,
            )?;
            controls.push(control(
                &format!("{profile_name}:{}:naive-disposition", spec.id),
                result.naive_fixed_1_over_k.passed_expected_disposition,
                "Observed naive fixed-1/k behaviour matches the declared null-world expectation.",
            ));
            controls.push(control(
                &format!("{profile_name}:{}:global-conditional-disposition", spec.id),
                result.global_conditional_target_randomization.passed_expected_disposition,
                "Observed global conditional-randomization behaviour matches the declared exchangeability expectation.",
            ));
            controls.push(control(
                &format!("{profile_name}:{}:block-conditional-disposition", spec.id),
                result.block_conditional_target_randomization.passed_expected_disposition,
                "Observed block-conditional behaviour matches the declared block-exchangeability expectation.",
            ));
            scenarios.push(result);
        }
    }

    let all_controls_passed = controls.iter().all(|item| item.passed);
    let body = ReceiptBody {
        receipt_schema_version: "sympsi.adversarial-null-receipt.v0.1".into(),
        tool_version: TOOL_VERSION.into(),
        git_head: git_head.to_ascii_lowercase(),
        synthetic_only: true,
        measurement_class: "SIMULATED".into(),
        claim_ceiling: "ADVERSARIAL_NULL_DIAGNOSTICS_ONLY_NOT_PSI_OR_AIA_EVIDENCE".into(),
        manifest_sha256,
        master_seed: manifest.master_seed,
        campaigns_per_scenario: manifest.campaigns_per_scenario,
        alpha: manifest.alpha,
        conditional_randomization_z_critical: manifest.conditional_randomization_z_critical,
        scenarios,
        controls,
        all_controls_passed,
    };
    let canonical_body = serde_json::to_vec(&body)
        .map_err(|error| format!("receipt serialization failed: {error}"))?;
    let receipt_sha256 = sha256_hex(&canonical_body);
    Ok(ReceiptEnvelope {
        receipt: body,
        receipt_sha256,
    })
}

fn validate_git_head(head: &str) -> Result<(), String> {
    if head.len() != 40 || !head.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("git head must be an exact 40-character hexadecimal commit SHA".into());
    }
    Ok(())
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.schema_version != "sympsi.adversarial-null.v0.1" || !manifest.synthetic_only {
        return Err("manifest schema or synthetic_only flag is invalid".into());
    }
    if manifest.campaigns_per_scenario < 2_000 {
        return Err("campaign count is below the adversarial-null qualification minimum".into());
    }
    if (manifest.alpha - 0.005).abs() > 1e-12 {
        return Err("this v0.1 benchmark is frozen at alpha=0.005".into());
    }
    if (manifest.conditional_randomization_z_critical - 2.575_829_303_548_900_4).abs() > 1e-12 {
        return Err("conditional randomization z threshold differs from the frozen one-sided 99.5% value".into());
    }
    if !(1.96..=3.5).contains(&manifest.confidence_z) {
        return Err("confidence z-score is outside supported range".into());
    }
    if manifest.profiles.len() != 2 {
        return Err("manifest must contain exactly four-choice and five-choice profiles".into());
    }
    if manifest.scenario_ids != SCENARIOS.iter().map(|scenario| scenario.id.to_string()).collect::<Vec<_>>() {
        return Err("scenario IDs or ordering differ from the frozen scenario catalogue".into());
    }
    let four = manifest.profiles.get("four-choice")
        .ok_or_else(|| "four-choice profile missing".to_string())?;
    let five = manifest.profiles.get("five-choice")
        .ok_or_else(|| "five-choice profile missing".to_string())?;
    validate_profile(four, 4, 7_300, 1_922)?;
    validate_profile(five, 5, 6_300, 1_343)?;
    Ok(())
}

fn validate_profile(
    profile: &ProfileSpec,
    choices: usize,
    trials: usize,
    critical: usize,
) -> Result<(), String> {
    if profile.choice_count != choices
        || profile.analysis_trials != trials
        || profile.naive_critical_hits != critical
        || profile.block_count != 10
        || profile.analysis_trials % profile.block_count != 0
    {
        return Err(format!("profile does not match frozen {choices}-choice reference contract"));
    }
    Ok(())
}

fn simulate_scenario(
    spec: ScenarioSpec,
    profile_name: &str,
    profile: &ProfileSpec,
    manifest: &Manifest,
) -> Result<ScenarioResult, String> {
    let choices = profile.choice_count;
    let n = profile.analysis_trials;
    let block_count = if spec.id == "blockwise_shared_drift" {
        profile.block_count
    } else {
        1
    };
    let block_size = n / block_count;

    let mut naive_rejections = 0_u64;
    let mut global_rejections = 0_u64;
    let mut block_rejections = 0_u64;

    let scenario_seed = derive_seed(
        manifest.master_seed,
        profile_name,
        spec.id,
    );

    for campaign in 0..manifest.campaigns_per_scenario {
        let counts = generate_campaign(
            choices,
            n,
            block_count,
            spec.id,
            scenario_seed.wrapping_add(campaign.wrapping_mul(0x9e37_79b9_7f4a_7c15)),
        );

        if counts.global_hits >= profile.naive_critical_hits as u64 {
            naive_rejections += 1;
        }

        let global_z = conditional_randomization_z(
            &counts.global_target,
            &counts.global_response,
            counts.global_hits,
        );
        if global_z >= manifest.conditional_randomization_z_critical {
            global_rejections += 1;
        }

        let block_z = conditional_block_z(
            &counts.block_target,
            &counts.block_response,
            &counts.block_hits,
        );
        if block_z >= manifest.conditional_randomization_z_critical {
            block_rejections += 1;
        }
    }

    let campaigns = manifest.campaigns_per_scenario;
    let naive = make_fpr(
        naive_rejections,
        campaigns,
        manifest.alpha,
        manifest.confidence_z,
        spec.naive_expected,
    );
    let global = make_fpr(
        global_rejections,
        campaigns,
        manifest.alpha,
        manifest.confidence_z,
        spec.global_expected,
    );
    let block = make_fpr(
        block_rejections,
        campaigns,
        manifest.alpha,
        manifest.confidence_z,
        spec.block_expected,
    );

    let expected_block_count = if spec.id == "blockwise_shared_drift" {
        block_count
    } else {
        1
    };
    let _ = block_size;

    Ok(ScenarioResult {
        scenario_id: spec.id.into(),
        null_mechanism: spec.mechanism.into(),
        profile: profile_name.into(),
        choice_count: choices,
        analysis_trials: n,
        campaigns,
        target_response_independent_by_construction: true,
        response_temporal_dependence: spec.response_dependence.into(),
        exchangeability_unit: if expected_block_count == 1 {
            "whole campaign".into()
        } else {
            format!("{expected_block_count} fixed blocks of {block_size} trials")
        },
        naive_fixed_1_over_k: naive,
        global_conditional_target_randomization: global,
        block_conditional_target_randomization: block,
    })
}

fn generate_campaign(
    choices: usize,
    n: usize,
    block_count: usize,
    scenario_id: &str,
    seed: u64,
) -> CampaignCounts {
    let mut rng = SplitMix64::new(seed);
    let block_size = n / block_count;
    let mut global_target = vec![0_u64; choices];
    let mut global_response = vec![0_u64; choices];
    let mut global_hits = 0_u64;
    let mut block_target = vec![vec![0_u64; choices]; block_count];
    let mut block_response = vec![vec![0_u64; choices]; block_count];
    let mut block_hits = vec![0_u64; block_count];
    let mut previous_response: Option<usize> = None;

    for index in 0..n {
        let block = index / block_size;
        let (target_weights, response_weights, markov) =
            scenario_weights(scenario_id, choices, block);
        let target = rng.categorical(&target_weights);
        let response = if markov {
            match previous_response {
                Some(previous) if rng.unit() < 0.80 => previous,
                _ => rng.categorical(&response_weights),
            }
        } else {
            rng.categorical(&response_weights)
        };
        previous_response = Some(response);

        global_target[target] += 1;
        global_response[response] += 1;
        block_target[block][target] += 1;
        block_response[block][response] += 1;
        if target == response {
            global_hits += 1;
            block_hits[block] += 1;
        }
    }

    CampaignCounts {
        global_target,
        global_response,
        global_hits,
        block_target,
        block_response,
        block_hits,
    }
}

fn scenario_weights(
    scenario_id: &str,
    choices: usize,
    block: usize,
) -> (Vec<f64>, Vec<f64>, bool) {
    let uniform = vec![1.0 / choices as f64; choices];
    let biased = if choices == 4 {
        vec![0.70, 0.10, 0.10, 0.10]
    } else {
        vec![0.60, 0.10, 0.10, 0.10, 0.10]
    };

    match scenario_id {
        "uniform_iid" => (uniform.clone(), uniform, false),
        "static_marginal_alignment" => (biased.clone(), biased, false),
        "target_imbalance_uniform_response" => (biased, uniform, false),
        "persistent_response_preference" => (uniform, biased, true),
        "blockwise_shared_drift" => {
            let dominant = block % choices;
            let mut distribution = vec![0.10; choices];
            distribution[dominant] = if choices == 4 { 0.70 } else { 0.60 };
            (distribution.clone(), distribution, true)
        }
        _ => unreachable!("scenario catalogue is validated before simulation"),
    }
}

/// Mean and variance of the match count when target labels are permuted over
/// fixed response positions, conditional on both category margins.
fn conditional_moments(
    target_counts: &[u64],
    response_counts: &[u64],
) -> (f64, f64) {
    let n: u64 = target_counts.iter().sum();
    if n == 0 || response_counts.iter().sum::<u64>() != n {
        return (0.0, 0.0);
    }
    let nf = n as f64;
    let denominator = (n * n.saturating_sub(1)) as f64;
    let mut mean = 0.0;
    let mut variance = 0.0;

    for category in 0..target_counts.len() {
        let target_count = target_counts[category] as f64;
        let response_count = response_counts[category] as f64;
        let probability = target_count / nf;
        mean += target_count * response_count / nf;
        variance += response_count * probability * (1.0 - probability);
    }

    if n > 1 {
        for category in 0..target_counts.len() {
            let a = target_counts[category] as f64;
            let b = response_counts[category] as f64;
            let p = a / nf;
            let covariance_same = a * (a - 1.0) / denominator - p * p;
            variance += b * (b - 1.0) * covariance_same;

            for other in (category + 1)..target_counts.len() {
                let a_other = target_counts[other] as f64;
                let b_other = response_counts[other] as f64;
                let p_other = a_other / nf;
                let covariance_different =
                    a * a_other / denominator - p * p_other;
                variance += 2.0 * b * b_other * covariance_different;
            }
        }
    }

    (mean, variance.max(0.0))
}

fn conditional_randomization_z(
    target_counts: &[u64],
    response_counts: &[u64],
    observed_hits: u64,
) -> f64 {
    let (mean, variance) = conditional_moments(target_counts, response_counts);
    standardized_score(observed_hits as f64, mean, variance)
}

fn conditional_block_z(
    target_counts: &[Vec<u64>],
    response_counts: &[Vec<u64>],
    observed_hits: &[u64],
) -> f64 {
    let mut total_observed = 0.0;
    let mut total_mean = 0.0;
    let mut total_variance = 0.0;
    for block in 0..target_counts.len() {
        let (mean, variance) =
            conditional_moments(&target_counts[block], &response_counts[block]);
        total_observed += observed_hits[block] as f64;
        total_mean += mean;
        total_variance += variance;
    }
    standardized_score(total_observed, total_mean, total_variance)
}

fn standardized_score(observed: f64, mean: f64, variance: f64) -> f64 {
    if variance <= f64::EPSILON {
        if observed > mean {
            f64::INFINITY
        } else if observed < mean {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else {
        (observed - mean) / variance.sqrt()
    }
}

fn make_fpr(
    rejections: u64,
    campaigns: u64,
    alpha: f64,
    confidence_z: f64,
    expected: ExpectedDisposition,
) -> FalsePositiveRate {
    let rate = rejections as f64 / campaigns as f64;
    let interval = wilson_interval(rejections, campaigns, confidence_z);
    let tolerance =
        4.0 * (alpha * (1.0 - alpha) / campaigns as f64).sqrt()
            + 1.0 / campaigns as f64;
    let (expected_pass, disposition) = match expected {
        ExpectedDisposition::Controlled => (
            interval.upper <= alpha + tolerance,
            "CONTROLLED_WITHIN_PREREGISTERED_MONTE_CARLO_BOUND",
        ),
        ExpectedDisposition::Inflated => (
            interval.lower > INFLATED_FPR_FLOOR,
            "EXPECTED_FALSE_POSITIVE_INFLATION_DETECTED",
        ),
    };
    FalsePositiveRate {
        campaigns,
        rejections,
        empirical_rate: rate,
        wilson_99: interval,
        upper_limit_for_controlled_null: alpha + tolerance,
        disposition: disposition.into(),
        passed_expected_disposition: expected_pass,
    }
}

fn wilson_interval(successes: u64, trials: u64, z: f64) -> Interval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let margin =
        z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    Interval {
        lower: (center - margin).max(0.0),
        upper: (center + margin).min(1.0),
    }
}

fn control(id: &str, passed: bool, detail: &str) -> ControlResult {
    ControlResult {
        id: id.into(),
        passed,
        detail: detail.into(),
    }
}

fn derive_seed(master: u64, profile: &str, scenario: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(DOMAIN.as_bytes());
    hasher.update([0]);
    hasher.update(master.to_be_bytes());
    hasher.update([0]);
    hasher.update(profile.as_bytes());
    hasher.update([0]);
    hasher.update(scenario.as_bytes());
    let digest = hasher.finalize();
    u64::from_be_bytes(digest[..8].try_into().expect("fixed digest prefix"))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    encode_hex(&digest)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    fn categorical(&mut self, weights: &[f64]) -> usize {
        let draw = self.unit();
        let mut cumulative = 0.0;
        for (index, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if draw < cumulative {
                return index;
            }
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_moments_match_small_exact_enumeration_fixture() {
        let targets = [2, 2, 1, 1];
        let responses = [2, 1, 2, 1];
        let (mean, variance) = conditional_moments(&targets, &responses);
        assert!((mean - 2.5).abs() < 1e-12);
        assert!((variance - 0.8333333333333333).abs() < 1e-12);
    }

    #[test]
    fn conditional_randomization_is_invariant_to_response_order_when_margins_match() {
        let target_counts = [30, 20, 20, 30];
        let response_counts = [25, 25, 25, 25];
        let first = conditional_moments(&target_counts, &response_counts);
        let second = conditional_moments(&response_counts, &target_counts);
        assert!((first.0 - second.0).abs() < 1e-12);
        assert!((first.1 - second.1).abs() < 1e-12);
    }

    #[test]
    fn block_conditional_score_sums_block_expectation_and_variance() {
        let targets = vec![vec![70, 10, 10, 10], vec![10, 70, 10, 10]];
        let responses = vec![vec![70, 10, 10, 10], vec![10, 70, 10, 10]];
        let hits = vec![52, 52];
        let z = conditional_block_z(&targets, &responses, &hits);
        assert!(z.abs() < 1e-12);
    }

    #[test]
    fn scenario_expectations_capture_shared_marginal_bias() {
        let four = ProfileSpec {
            choice_count: 4,
            analysis_trials: 7_300,
            naive_critical_hits: 1_922,
            block_count: 10,
        };
        let counts = generate_campaign(
            4,
            7_300,
            1,
            "static_marginal_alignment",
            1234,
        );
        let naive_rate = counts.global_hits as f64 / 7_300.0;
        let global_z = conditional_randomization_z(
            &counts.global_target,
            &counts.global_response,
            counts.global_hits,
        );
        assert!(naive_rate > 0.45);
        assert!(global_z.abs() < 8.0);
        assert_eq!(four.choice_count, 4);
    }

    #[test]
    fn deterministic_campaign_generation_replays() {
        let first = generate_campaign(5, 6_300, 1, "persistent_response_preference", 777);
        let second = generate_campaign(5, 6_300, 1, "persistent_response_preference", 777);
        assert_eq!(first.global_hits, second.global_hits);
        assert_eq!(first.global_target, second.global_target);
        assert_eq!(first.global_response, second.global_response);
    }

    #[test]
    fn manifest_is_strictly_synthetic_and_pins_both_profile_contracts() {
        let manifest: Manifest =
            serde_json::from_slice(include_bytes!("../fixtures/manifest-v0.1.json")).unwrap();
        validate_manifest(&manifest).unwrap();
        let mut invalid = manifest;
        invalid.synthetic_only = false;
        assert!(validate_manifest(&invalid).is_err());
    }

    #[test]
    fn git_head_must_be_exact() {
        assert!(validate_git_head(&"a".repeat(40)).is_ok());
        assert!(validate_git_head("main").is_err());
    }
}

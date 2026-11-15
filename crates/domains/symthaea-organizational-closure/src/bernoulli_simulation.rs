//! Reproducible Monte Carlo harness for the Bernoulli anytime-valid evaluator.
//!
//! SplitMix64 is used only to make synthetic experiments repeatable. It is not
//! cryptographic randomness and is not suitable for allocating real treatment.
//! The simulated policy sees only past observations and public interim bounds;
//! evaluator-only probabilities are used only by the outcome generator and
//! the final evaluator after the trace is closed.

use super::bernoulli_benchmark::{
    BernoulliBenchmarkError, BernoulliBenchmarkSummary, summarize_bernoulli_episodes,
};
use super::bernoulli_sequential::{
    BernoulliArm, BernoulliBoundary, BernoulliClaim, BernoulliOracle,
    BernoulliSequentialReport, BernoulliSequentialScenario, BernoulliSequentialTrace,
    BernoulliTerminalDecision, BernoulliTrial, boundary_for_counts,
    evaluate_bernoulli_sequential,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const BERNOULLI_SIMULATION_PRNG_VERSION: &str = "splitmix64-v1";
pub const BERNOULLI_SIMULATION_POLICY_VERSION: &str = "balanced-anytime-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSimulationStratum {
    pub stratum_id: String,
    pub control_probability: f64,
    pub treatment_probability: f64,
    pub episode_count: usize,
}

impl BernoulliSimulationStratum {
    pub fn new(
        stratum_id: impl Into<String>,
        control_probability: f64,
        treatment_probability: f64,
        episode_count: usize,
    ) -> Self {
        Self {
            stratum_id: stratum_id.into(),
            control_probability,
            treatment_probability,
            episode_count,
        }
    }
}

/// Frozen and seedable synthetic episode manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSimulationManifest {
    pub root_seed: u64,
    pub scenario: BernoulliSequentialScenario,
    pub strata: Vec<BernoulliSimulationStratum>,
}

impl BernoulliSimulationManifest {
    pub fn new(
        root_seed: u64,
        scenario: BernoulliSequentialScenario,
        strata: Vec<BernoulliSimulationStratum>,
    ) -> Self {
        Self { root_seed, scenario, strata }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BernoulliSimulationError {
    EmptyStrata,
    EmptyStratumId,
    DuplicateStratumId,
    ZeroEpisodeCount,
    InvalidProbability,
    InvalidScenario,
    InvalidMetricMargin,
}

impl From<BernoulliBenchmarkError> for BernoulliSimulationError {
    fn from(_: BernoulliBenchmarkError) -> Self {
        Self::InvalidMetricMargin
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSimulationEpisode {
    pub episode_id: String,
    pub episode_seed: u64,
    pub trace: BernoulliSequentialTrace,
    pub report: BernoulliSequentialReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSimulationStratumResult {
    pub stratum_id: String,
    /// Evaluator-only generator parameters; never pass this record to an agent.
    pub control_probability_for_scoring: f64,
    pub treatment_probability_for_scoring: f64,
    pub episodes: Vec<BernoulliSimulationEpisode>,
    pub summary: BernoulliBenchmarkSummary,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BernoulliSimulationSuiteResult {
    pub root_seed: u64,
    pub manifest_digest: String,
    pub prng_version: String,
    pub policy_version: String,
    pub strata: Vec<BernoulliSimulationStratumResult>,
}

/// Small, fully specified SplitMix64 generator. It is deliberately local to
/// the offline benchmark so changing runtime RNG dependencies cannot silently
/// change historical simulation manifests.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }

    fn next_unit_f64(&mut self) -> f64 {
        // Use the high 53 bits so the result is exactly in [0, 1).
        ((self.next_u64() >> 11) as f64) * (1.0 / 9_007_199_254_740_992.0)
    }
}

fn stable_stratum_hash(value: &str) -> u64 {
    // FNV-1a is used only for deterministic seed derivation, not security.
    value.as_bytes().iter().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

fn derive_episode_seed(root_seed: u64, stratum_id: &str, episode_index: usize) -> u64 {
    let index_mix = (episode_index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    let mut mixer = SplitMix64::new(root_seed ^ stable_stratum_hash(stratum_id) ^ index_mix);
    mixer.next_u64()
}

fn digest_manifest(manifest: &BernoulliSimulationManifest) -> String {
    let scenario = &manifest.scenario;
    let mut canonical = format!(
        "bernoulli-simulation-v1|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
        manifest.root_seed,
        scenario.alpha.to_bits(),
        scenario.minimum_effect.to_bits(),
        scenario.max_total_samples,
        scenario.max_samples_per_arm,
        scenario.sample_cost_units,
        scenario.max_cost_units,
        scenario.sample_risk_units,
        scenario.max_risk_units,
        BERNOULLI_SIMULATION_PRNG_VERSION,
        BERNOULLI_SIMULATION_POLICY_VERSION,
        "paired-balanced-allocation",
        "verified-synthetic-evidence"
    );
    let mut strata = manifest.strata.clone();
    strata.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));
    for stratum in strata {
        canonical.push('|');
        canonical.push_str(&stratum.stratum_id.len().to_string());
        canonical.push(':');
        canonical.push_str(&stratum.stratum_id);
        canonical.push(':');
        canonical.push_str(&stratum.control_probability.to_bits().to_string());
        canonical.push(':');
        canonical.push_str(&stratum.treatment_probability.to_bits().to_string());
        canonical.push(':');
        canonical.push_str(&stratum.episode_count.to_string());
    }
    let digest = Sha256::digest(canonical.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_manifest(
    manifest: &BernoulliSimulationManifest,
) -> Result<(), BernoulliSimulationError> {
    if manifest.strata.is_empty() {
        return Err(BernoulliSimulationError::EmptyStrata);
    }
    let scenario = &manifest.scenario;
    if !scenario.alpha.is_finite()
        || scenario.alpha <= 0.0
        || scenario.alpha >= 1.0
        || !scenario.minimum_effect.is_finite()
        || scenario.minimum_effect <= 0.0
        || scenario.minimum_effect > 1.0
        || scenario.max_total_samples == 0
        || scenario.max_samples_per_arm == 0
    {
        return Err(BernoulliSimulationError::InvalidScenario);
    }
    let mut seen = BTreeSet::new();
    for stratum in &manifest.strata {
        if stratum.stratum_id.trim().is_empty() {
            return Err(BernoulliSimulationError::EmptyStratumId);
        }
        if !seen.insert(stratum.stratum_id.clone()) {
            return Err(BernoulliSimulationError::DuplicateStratumId);
        }
        if stratum.episode_count == 0 {
            return Err(BernoulliSimulationError::ZeroEpisodeCount);
        }
        if !stratum.control_probability.is_finite()
            || !(0.0..=1.0).contains(&stratum.control_probability)
            || !stratum.treatment_probability.is_finite()
            || !(0.0..=1.0).contains(&stratum.treatment_probability)
        {
            return Err(BernoulliSimulationError::InvalidProbability);
        }
    }
    Ok(())
}

/// Maximum number of balanced pairs that fit every frozen resource budget.
fn maximum_pairs(scenario: &BernoulliSequentialScenario) -> usize {
    let mut pairs = (scenario.max_total_samples / 2).min(scenario.max_samples_per_arm);
    if scenario.sample_cost_units > 0 {
        let per_pair = scenario.sample_cost_units.saturating_mul(2);
        let affordable = scenario.max_cost_units / per_pair;
        pairs = pairs.min(usize::try_from(affordable).unwrap_or(usize::MAX));
    }
    if scenario.sample_risk_units > 0 {
        let per_pair = scenario.sample_risk_units.saturating_mul(2);
        let affordable = scenario.max_risk_units / per_pair;
        pairs = pairs.min(usize::try_from(affordable).unwrap_or(usize::MAX));
    }
    pairs
}

fn claim_for_boundary(boundary: BernoulliBoundary) -> Option<BernoulliClaim> {
    match boundary {
        BernoulliBoundary::Benefit => Some(BernoulliClaim::Benefit),
        BernoulliBoundary::Harm => Some(BernoulliClaim::Harm),
        BernoulliBoundary::NoMaterialEffect => Some(BernoulliClaim::NoMaterialEffect),
        BernoulliBoundary::Continue => None,
    }
}

/// Generate one episode with a balanced anytime-stopping policy.
///
/// The policy's decision function receives only accumulated synthetic
/// observations and public confidence bounds. The hidden probabilities are
/// captured only by the RNG draw process and the evaluator after closure.
fn simulate_episode(
    scenario: &BernoulliSequentialScenario,
    control_probability: f64,
    treatment_probability: f64,
    episode_seed: u64,
) -> (BernoulliSequentialTrace, BernoulliSequentialReport) {
    let mut rng = SplitMix64::new(episode_seed);
    let mut trials = Vec::new();
    let mut control_successes = 0_usize;
    let mut treatment_successes = 0_usize;
    let mut pairs = 0_usize;
    let max_pairs = maximum_pairs(scenario);
    let mut terminal = BernoulliTerminalDecision::Abstain {
        reason: "balanced policy exhausted its frozen sample or resource budget".into(),
    };

    for _ in 0..max_pairs {
        let control_outcome = rng.next_unit_f64() < control_probability;
        let treatment_outcome = rng.next_unit_f64() < treatment_probability;
        trials.push(BernoulliTrial::qualified(BernoulliArm::Control, control_outcome));
        trials.push(BernoulliTrial::qualified(BernoulliArm::Treatment, treatment_outcome));
        pairs += 1;
        if control_outcome {
            control_successes += 1;
        }
        if treatment_outcome {
            treatment_successes += 1;
        }

        // Deliberately pass only observations and the public scenario into the
        // stopping policy; the oracle probabilities are not arguments here.
        let observed_boundary = boundary_for_counts(
            control_successes,
            pairs,
            treatment_successes,
            pairs,
            scenario.alpha,
            scenario.minimum_effect,
        );
        if let Some(claim) = claim_for_boundary(observed_boundary) {
            terminal = BernoulliTerminalDecision::Stop { claim };
            break;
        }
    }

    let trace = BernoulliSequentialTrace { trials, terminal };
    let oracle = BernoulliOracle::new(control_probability, treatment_probability);
    let report = evaluate_bernoulli_sequential(scenario, &oracle, &trace);
    (trace, report)
}

/// Run a fully reproducible balanced sequential policy over frozen strata.
///
/// Results are grouped by stratum rather than pooled into one headline rate.
/// Each report includes its complete episode trace and evaluator-only score
/// fields; do not send simulation results back to the agent as task input.
pub fn run_bernoulli_simulation_suite(
    manifest: &BernoulliSimulationManifest,
) -> Result<BernoulliSimulationSuiteResult, BernoulliSimulationError> {
    validate_manifest(manifest)?;
    let digest = digest_manifest(manifest);
    let mut strata = manifest.strata.clone();
    strata.sort_by(|left, right| left.stratum_id.cmp(&right.stratum_id));
    let mut results = Vec::with_capacity(strata.len());

    for stratum in strata {
        let mut episodes = Vec::with_capacity(stratum.episode_count);
        for episode_index in 0..stratum.episode_count {
            let episode_seed = derive_episode_seed(
                manifest.root_seed,
                &stratum.stratum_id,
                episode_index,
            );
            let (trace, report) = simulate_episode(
                &manifest.scenario,
                stratum.control_probability,
                stratum.treatment_probability,
                episode_seed,
            );
            episodes.push(BernoulliSimulationEpisode {
                episode_id: format!("{}::{episode_index:06}", stratum.stratum_id),
                episode_seed,
                trace,
                report,
            });
        }
        let reports: Vec<_> = episodes.iter().map(|episode| episode.report.clone()).collect();
        let summary = summarize_bernoulli_episodes(
            &reports,
            manifest.scenario.minimum_effect,
        )?;
        results.push(BernoulliSimulationStratumResult {
            stratum_id: stratum.stratum_id,
            control_probability_for_scoring: stratum.control_probability,
            treatment_probability_for_scoring: stratum.treatment_probability,
            episodes,
            summary,
        });
    }

    Ok(BernoulliSimulationSuiteResult {
        root_seed: manifest.root_seed,
        manifest_digest: digest,
        prng_version: BERNOULLI_SIMULATION_PRNG_VERSION.into(),
        policy_version: BERNOULLI_SIMULATION_POLICY_VERSION.into(),
        strata: results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> BernoulliSimulationManifest {
        BernoulliSimulationManifest::new(
            0xA0B0_C0D0_E0F0_1234,
            BernoulliSequentialScenario::new(0.05, 0.10, 200, 100, 1, 200, 1, 200),
            vec![
                BernoulliSimulationStratum::new("null", 0.50, 0.50, 8),
                BernoulliSimulationStratum::new("benefit-large", 0.10, 0.90, 8),
                BernoulliSimulationStratum::new("harm-large", 0.90, 0.10, 8),
            ],
        )
    }

    #[test]
    fn same_manifest_reproduces_digest_episode_seeds_traces_and_summaries() {
        let manifest = manifest();
        let first = run_bernoulli_simulation_suite(&manifest).unwrap();
        let second = run_bernoulli_simulation_suite(&manifest).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.prng_version, "splitmix64-v1");
        assert_eq!(first.policy_version, "balanced-anytime-v1");
    }

    #[test]
    fn stratum_order_does_not_change_seeded_episodes_or_digest() {
        let forward = manifest();
        let mut reverse = manifest();
        reverse.strata.reverse();
        let a = run_bernoulli_simulation_suite(&forward).unwrap();
        let b = run_bernoulli_simulation_suite(&reverse).unwrap();
        assert_eq!(a.manifest_digest, b.manifest_digest);
        assert_eq!(a.strata, b.strata);
    }

    #[test]
    fn changed_root_seed_changes_episode_streams() {
        let mut first = manifest();
        let mut second = manifest();
        second.root_seed = second.root_seed.wrapping_add(1);
        let a = run_bernoulli_simulation_suite(&first).unwrap();
        let b = run_bernoulli_simulation_suite(&second).unwrap();
        let seeds_a: Vec<_> = a.strata.iter().flat_map(|s| {
            s.episodes.iter().map(|episode| episode.episode_seed)
        }).collect();
        let seeds_b: Vec<_> = b.strata.iter().flat_map(|s| {
            s.episodes.iter().map(|episode| episode.episode_seed)
        }).collect();
        assert_ne!(seeds_a, seeds_b);
    }

    #[test]
    fn every_episode_retains_complete_trace_and_attempted_sample_accounting() {
        let result = run_bernoulli_simulation_suite(&manifest()).unwrap();
        assert_eq!(result.strata.len(), 3);
        for stratum in &result.strata {
            assert_eq!(stratum.episodes.len(), 8);
            for episode in &stratum.episodes {
                assert_eq!(episode.report.attempted_samples, episode.trace.trials.len());
                assert_eq!(
                    episode.report.attempted_samples,
                    episode.report.control.attempted_samples
                        + episode.report.treatment.attempted_samples
                );
                assert!(episode.report.findings.is_empty());
            }
        }
    }

    #[test]
    fn manifest_rejects_duplicate_ids_and_invalid_probabilities() {
        let mut duplicate = manifest();
        duplicate.strata.push(duplicate.strata[0].clone());
        assert_eq!(
            run_bernoulli_simulation_suite(&duplicate),
            Err(BernoulliSimulationError::DuplicateStratumId)
        );

        let mut invalid = manifest();
        invalid.strata[0].control_probability = f64::NAN;
        assert_eq!(
            run_bernoulli_simulation_suite(&invalid),
            Err(BernoulliSimulationError::InvalidProbability)
        );
    }
}

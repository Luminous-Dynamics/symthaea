//! Paired, deterministic statistics for active-learning campaign claims.
//!
//! Acceleration is baseline cost-to-target divided by candidate cost-to-target.
//! Every predeclared seed must contribute a matched pair where both policies
//! reached the same frozen target. A missing/non-target run blocks this interval;
//! callers must report such runs separately rather than silently dropping them.
//!
//! The interval is a percentile non-parametric bootstrap over paired seed-level
//! acceleration factors. It is descriptive evidence for the declared cohort, not
//! a substitute for prospective validation or a universal performance claim.

use crate::campaign::{CampaignError, CampaignRecord, validate_target_reached_pair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const PAIRED_ACCELERATION_STATS_VERSION: u16 = 1;
pub const MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL: usize = 20;
pub const MINIMUM_BOOTSTRAP_RESAMPLES: usize = 1_000;
pub const SEED_MANIFEST_SCHEMA_VERSION: u16 = 1;

/// Canonical, versioned roster of seeds fixed before outcome inspection.
///
/// Hash preimage: UTF-8 bytes of `atomfab-seed-manifest-v1\n` followed by
/// every unique seed in ascending decimal form, one per line. Pin this digest
/// in the preregistration before any outcomes are examined.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedManifestV1 {
    pub schema_version: u16,
    pub seeds: Vec<u64>,
    pub sha256: String,
}

impl SeedManifestV1 {
    pub fn from_seeds(seeds: &[u64]) -> Result<Self, PairedAccelerationError> {
        if seeds.len() < MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL {
            return Err(PairedAccelerationError::TooFewCampaignPairs {
                actual: seeds.len(),
                minimum: MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL,
            });
        }
        let mut canonical = seeds.to_vec();
        canonical.sort_unstable();
        if canonical.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(PairedAccelerationError::DuplicateSeed);
        }
        let sha256 = seed_manifest_sha256(&canonical);
        Ok(Self {
            schema_version: SEED_MANIFEST_SCHEMA_VERSION,
            seeds: canonical,
            sha256,
        })
    }

    pub fn validate(&self) -> Result<(), PairedAccelerationError> {
        if self.schema_version != SEED_MANIFEST_SCHEMA_VERSION
            || self.seeds.len() < MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL
            || self.seeds.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(PairedAccelerationError::InvalidSeedManifest);
        }
        if self.sha256 != seed_manifest_sha256(&self.seeds) {
            return Err(PairedAccelerationError::SeedManifestMismatch);
        }
        Ok(())
    }
}

fn seed_manifest_sha256(seeds: &[u64]) -> String {
    let mut canonical = String::from("atomfab-seed-manifest-v1\n");
    for seed in seeds {
        canonical.push_str(&seed.to_string());
        canonical.push('\n');
    }
    format!("{:x}", Sha256::digest(canonical.as_bytes()))
}

/// Deterministic summary of matched target-reaching campaign pairs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedAccelerationSummary {
    pub schema_version: u16,
    pub baseline_policy_id: String,
    pub candidate_policy_id: String,
    pub predeclared_seeds: Vec<u64>,
    pub seed_manifest_sha256: String,
    pub campaign_pair_count: usize,
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
    pub confidence_level: f64,
    /// Median of per-seed (baseline total cost / candidate total cost).
    /// Values above 1 favor the candidate.
    pub median_acceleration_factor: f64,
    pub lower_confidence_bound: f64,
    pub upper_confidence_bound: f64,
    pub candidate_faster_count: usize,
    pub tied_cost_count: usize,
    pub candidate_slower_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedAccelerationError {
    EmptyPairs,
    TooFewCampaignPairs { actual: usize, minimum: usize },
    TooFewBootstrapResamples { actual: usize, minimum: usize },
    InvalidConfidenceLevel,
    InvalidSeedManifest,
    SeedManifestMismatch,
    SeedRosterMismatch,
    DuplicateSeed,
    MismatchedCohort,
    TargetNotReached,
    InvalidCost,
    NonFiniteStatistic,
    InvalidSummary,
    InvalidCampaign(CampaignError),
}

impl fmt::Display for PairedAccelerationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPairs => write!(f, "paired acceleration summary needs campaign pairs"),
            Self::TooFewCampaignPairs { actual, minimum } => write!(
                f,
                "paired acceleration interval needs at least {minimum} seeds; got {actual}"
            ),
            Self::TooFewBootstrapResamples { actual, minimum } => write!(
                f,
                "paired bootstrap needs at least {minimum} resamples; got {actual}"
            ),
            Self::InvalidConfidenceLevel => {
                write!(
                    f,
                    "confidence level must be finite and strictly between zero and one"
                )
            }
            Self::InvalidSeedManifest => write!(
                f,
                "seed manifest must be sorted, unique, versioned, and contain at least 20 seeds"
            ),
            Self::SeedManifestMismatch => write!(
                f,
                "seed manifest SHA-256 does not match its canonical seed roster"
            ),
            Self::SeedRosterMismatch => write!(
                f,
                "observed campaign seed roster does not exactly match the predeclared manifest"
            ),
            Self::DuplicateSeed => write!(f, "each preregistered seed may appear only once"),
            Self::MismatchedCohort => write!(
                f,
                "paired campaigns must use stable policies, model/data/evaluator/environment, target, budget, cost units and per-policy configuration across seeds"
            ),
            Self::TargetNotReached => write!(
                f,
                "every paired campaign must be completed and reach the identical frozen target"
            ),
            Self::InvalidCost => write!(
                f,
                "both total costs must be finite and strictly positive in a common declared unit"
            ),
            Self::NonFiniteStatistic => {
                write!(f, "bootstrap produced a non-finite statistic")
            }
            Self::InvalidSummary => {
                write!(f, "paired acceleration summary is internally inconsistent")
            }
            Self::InvalidCampaign(error) => write!(f, "invalid campaign record: {error}"),
        }
    }
}

impl Error for PairedAccelerationError {}

impl From<CampaignError> for PairedAccelerationError {
    fn from(value: CampaignError) -> Self {
        match value {
            CampaignError::TargetNotReached | CampaignError::TargetReachedOnNonCompletedRun => {
                Self::TargetNotReached
            }
            other => Self::InvalidCampaign(other),
        }
    }
}

/// Compute a percentile bootstrap interval over matched, target-reaching runs.
///
/// Each tuple is (baseline, candidate) for one seed. The function is deliberately
/// strict: it returns no interval if any supplied pair fails to reach the target.
/// This prevents non-target runs from vanishing through a success-only filter.
///
/// `confidence_level` is typically 0.95. `bootstrap_seed` pins the resampling
/// sequence so the summary can be independently recomputed from the same records.
pub fn bootstrap_median_acceleration(
    pairs: &[(&CampaignRecord, &CampaignRecord)],
    seed_manifest: &SeedManifestV1,
    confidence_level: f64,
    bootstrap_resamples: usize,
    bootstrap_seed: u64,
) -> Result<PairedAccelerationSummary, PairedAccelerationError> {
    if pairs.is_empty() {
        return Err(PairedAccelerationError::EmptyPairs);
    }
    seed_manifest.validate()?;
    if pairs.len() < MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL {
        return Err(PairedAccelerationError::TooFewCampaignPairs {
            actual: pairs.len(),
            minimum: MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL,
        });
    }
    if pairs.len() != seed_manifest.seeds.len() {
        return Err(PairedAccelerationError::SeedRosterMismatch);
    }
    if bootstrap_resamples < MINIMUM_BOOTSTRAP_RESAMPLES {
        return Err(PairedAccelerationError::TooFewBootstrapResamples {
            actual: bootstrap_resamples,
            minimum: MINIMUM_BOOTSTRAP_RESAMPLES,
        });
    }
    if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
        return Err(PairedAccelerationError::InvalidConfidenceLevel);
    }

    let (first_baseline, first_candidate) = pairs[0];
    let baseline_policy_id = first_baseline.policy_id.clone();
    let candidate_policy_id = first_candidate.policy_id.clone();
    let mut seen_seeds = BTreeSet::new();
    let mut run_ids = BTreeSet::new();
    let mut ordered = Vec::with_capacity(pairs.len());

    for (baseline, candidate) in pairs {
        validate_target_reached_pair(baseline, candidate)?;
        if baseline.policy_id != baseline_policy_id
            || candidate.policy_id != candidate_policy_id
            || !same_policy_cohort(first_baseline, baseline)
            || !same_policy_cohort(first_candidate, candidate)
        {
            return Err(PairedAccelerationError::MismatchedCohort);
        }
        if !seen_seeds.insert(baseline.seed) {
            return Err(PairedAccelerationError::DuplicateSeed);
        }
        if !run_ids.insert(baseline.run_id.clone()) || !run_ids.insert(candidate.run_id.clone()) {
            return Err(PairedAccelerationError::MismatchedCohort);
        }

        if baseline.total_cost_unit != candidate.total_cost_unit
            || baseline.max_evaluations != candidate.max_evaluations
            || !baseline.total_cost_value.is_finite()
            || !candidate.total_cost_value.is_finite()
            || baseline.total_cost_value <= 0.0
            || candidate.total_cost_value <= 0.0
        {
            return Err(PairedAccelerationError::InvalidCost);
        }
        let factor = baseline.total_cost_value / candidate.total_cost_value;
        if !factor.is_finite() || factor <= 0.0 {
            return Err(PairedAccelerationError::NonFiniteStatistic);
        }
        ordered.push((baseline.seed, factor));
    }

    // Canonical order makes the output invariant to the input list's ordering.
    ordered.sort_by_key(|(seed, _)| *seed);
    let observed_seeds = ordered.iter().map(|(seed, _)| *seed).collect::<Vec<_>>();
    if observed_seeds != seed_manifest.seeds {
        return Err(PairedAccelerationError::SeedRosterMismatch);
    }
    let factors = ordered
        .iter()
        .map(|(_, factor)| *factor)
        .collect::<Vec<_>>();
    let median_acceleration_factor = median(&factors);

    let candidate_faster_count = factors.iter().filter(|factor| **factor > 1.0).count();
    let tied_cost_count = factors.iter().filter(|factor| **factor == 1.0).count();
    let candidate_slower_count = factors.iter().filter(|factor| **factor < 1.0).count();

    let mut rng = SplitMix64::new(bootstrap_seed);
    let mut bootstrap_medians = Vec::with_capacity(bootstrap_resamples);
    for _ in 0..bootstrap_resamples {
        let mut sample = Vec::with_capacity(factors.len());
        for _ in 0..factors.len() {
            let index = rng.index(factors.len());
            sample.push(factors[index]);
        }
        bootstrap_medians.push(median(&sample));
    }
    bootstrap_medians.sort_by(f64::total_cmp);

    let alpha = 1.0 - confidence_level;
    let lower_confidence_bound = percentile(&bootstrap_medians, alpha / 2.0);
    let upper_confidence_bound = percentile(&bootstrap_medians, 1.0 - alpha / 2.0);
    if !median_acceleration_factor.is_finite()
        || !lower_confidence_bound.is_finite()
        || !upper_confidence_bound.is_finite()
    {
        return Err(PairedAccelerationError::NonFiniteStatistic);
    }

    Ok(PairedAccelerationSummary {
        schema_version: PAIRED_ACCELERATION_STATS_VERSION,
        baseline_policy_id,
        candidate_policy_id,
        predeclared_seeds: seed_manifest.seeds.clone(),
        seed_manifest_sha256: seed_manifest.sha256.clone(),
        campaign_pair_count: pairs.len(),
        bootstrap_resamples,
        bootstrap_seed,
        confidence_level,
        median_acceleration_factor,
        lower_confidence_bound,
        upper_confidence_bound,
        candidate_faster_count,
        tied_cost_count,
        candidate_slower_count,
    })
}

impl PairedAccelerationSummary {
    /// Validate the report after deserialization or before publishing it.
    pub fn validate(&self) -> Result<(), PairedAccelerationError> {
        if self.schema_version != PAIRED_ACCELERATION_STATS_VERSION
            || self.baseline_policy_id.trim().is_empty()
            || self.candidate_policy_id.trim().is_empty()
            || self.baseline_policy_id == self.candidate_policy_id
            || self.campaign_pair_count < MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL
            || self.predeclared_seeds.len() != self.campaign_pair_count
            || self.bootstrap_resamples < MINIMUM_BOOTSTRAP_RESAMPLES
            || !self.confidence_level.is_finite()
            || !(0.0..1.0).contains(&self.confidence_level)
            || !self.median_acceleration_factor.is_finite()
            || self.median_acceleration_factor <= 0.0
            || !self.lower_confidence_bound.is_finite()
            || self.lower_confidence_bound <= 0.0
            || !self.upper_confidence_bound.is_finite()
            || self.upper_confidence_bound <= 0.0
            || self.lower_confidence_bound > self.median_acceleration_factor
            || self.median_acceleration_factor > self.upper_confidence_bound
            || self
                .candidate_faster_count
                .checked_add(self.tied_cost_count)
                .and_then(|count| count.checked_add(self.candidate_slower_count))
                != Some(self.campaign_pair_count)
        {
            return Err(PairedAccelerationError::InvalidSummary);
        }
        let manifest = SeedManifestV1 {
            schema_version: SEED_MANIFEST_SCHEMA_VERSION,
            seeds: self.predeclared_seeds.clone(),
            sha256: self.seed_manifest_sha256.clone(),
        };
        manifest
            .validate()
            .map_err(|_| PairedAccelerationError::InvalidSummary)?;
        Ok(())
    }
}

/// Paired target-attainment statistics that retain failed/non-reaching runs.
///
/// Unlike cost-to-target AF, this estimator accepts completed, failed, or
/// incomplete campaigns as long as each record itself is valid. Every seed in
/// the frozen roster contributes exactly one pair. It reports target reachability
/// and a paired bootstrap interval for candidate minus baseline reach probability,
/// independently of the conditional cost-to-target interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairedTargetReachabilitySummary {
    pub schema_version: u16,
    pub baseline_policy_id: String,
    pub candidate_policy_id: String,
    pub predeclared_seeds: Vec<u64>,
    pub seed_manifest_sha256: String,
    pub campaign_pair_count: usize,
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
    pub confidence_level: f64,
    pub baseline_target_reached_count: usize,
    pub baseline_target_reach_rate: f64,
    pub candidate_target_reached_count: usize,
    pub candidate_target_reach_rate: f64,
    /// Candidate reach rate minus baseline reach rate, in [-1, 1].
    pub reach_rate_difference: f64,
    pub lower_confidence_bound: f64,
    pub upper_confidence_bound: f64,
    pub both_reached_count: usize,
    pub baseline_only_reached_count: usize,
    pub candidate_only_reached_count: usize,
    pub neither_reached_count: usize,
}

impl PairedTargetReachabilitySummary {
    pub fn validate(&self) -> Result<(), PairedAccelerationError> {
        let manifest = SeedManifestV1 {
            schema_version: SEED_MANIFEST_SCHEMA_VERSION,
            seeds: self.predeclared_seeds.clone(),
            sha256: self.seed_manifest_sha256.clone(),
        };
        manifest
            .validate()
            .map_err(|_| PairedAccelerationError::InvalidSummary)?;

        let paired_categories = self
            .both_reached_count
            .checked_add(self.baseline_only_reached_count)
            .and_then(|count| count.checked_add(self.candidate_only_reached_count))
            .and_then(|count| count.checked_add(self.neither_reached_count));
        if self.schema_version != PAIRED_ACCELERATION_STATS_VERSION
            || self.baseline_policy_id.trim().is_empty()
            || self.candidate_policy_id.trim().is_empty()
            || self.baseline_policy_id == self.candidate_policy_id
            || self.campaign_pair_count != self.predeclared_seeds.len()
            || self.bootstrap_resamples < MINIMUM_BOOTSTRAP_RESAMPLES
            || !self.confidence_level.is_finite()
            || !(0.0..1.0).contains(&self.confidence_level)
            || paired_categories != Some(self.campaign_pair_count)
            || self.baseline_target_reached_count
                != self.both_reached_count + self.baseline_only_reached_count
            || self.candidate_target_reached_count
                != self.both_reached_count + self.candidate_only_reached_count
            || !self.baseline_target_reach_rate.is_finite()
            || !self.candidate_target_reach_rate.is_finite()
            || !self.reach_rate_difference.is_finite()
            || !self.lower_confidence_bound.is_finite()
            || !self.upper_confidence_bound.is_finite()
            || self.lower_confidence_bound < -1.0
            || self.upper_confidence_bound > 1.0
            || self.lower_confidence_bound > self.upper_confidence_bound
            || (self.baseline_target_reach_rate
                - self.baseline_target_reached_count as f64 / self.campaign_pair_count as f64)
                .abs()
                > f64::EPSILON
            || (self.candidate_target_reach_rate
                - self.candidate_target_reached_count as f64 / self.campaign_pair_count as f64)
                .abs()
                > f64::EPSILON
            || (self.reach_rate_difference
                - (self.candidate_target_reach_rate - self.baseline_target_reach_rate))
                .abs()
                > f64::EPSILON
        {
            return Err(PairedAccelerationError::InvalidSummary);
        }
        Ok(())
    }
}

/// Summarize target attainment across every preregistered matched seed.
///
/// Failed and incomplete campaigns are valid observations here and count as
/// not reaching the target; they are not filtered from denominators. The
/// percentile bootstrap resamples paired seed outcomes, preserving the pairing.
pub fn summarize_target_reachability(
    pairs: &[(&CampaignRecord, &CampaignRecord)],
    seed_manifest: &SeedManifestV1,
    confidence_level: f64,
    bootstrap_resamples: usize,
    bootstrap_seed: u64,
) -> Result<PairedTargetReachabilitySummary, PairedAccelerationError> {
    if pairs.is_empty() {
        return Err(PairedAccelerationError::EmptyPairs);
    }
    seed_manifest.validate()?;
    if pairs.len() != seed_manifest.seeds.len() {
        return Err(PairedAccelerationError::SeedRosterMismatch);
    }
    if bootstrap_resamples < MINIMUM_BOOTSTRAP_RESAMPLES {
        return Err(PairedAccelerationError::TooFewBootstrapResamples {
            actual: bootstrap_resamples,
            minimum: MINIMUM_BOOTSTRAP_RESAMPLES,
        });
    }
    if !confidence_level.is_finite() || !(0.0..1.0).contains(&confidence_level) {
        return Err(PairedAccelerationError::InvalidConfidenceLevel);
    }

    let (first_baseline, first_candidate) = pairs[0];
    let baseline_policy_id = first_baseline.policy_id.clone();
    let candidate_policy_id = first_candidate.policy_id.clone();
    let mut seen_seeds = BTreeSet::new();
    let mut seen_runs = BTreeSet::new();
    let mut paired_outcomes = Vec::with_capacity(pairs.len());

    for (baseline, candidate) in pairs {
        baseline.validate()?;
        candidate.validate()?;
        crate::campaign::validate_matched_policy_pair(baseline, candidate)?;
        if baseline.policy_id != baseline_policy_id
            || candidate.policy_id != candidate_policy_id
            || baseline.max_evaluations != candidate.max_evaluations
            || !same_policy_cohort(first_baseline, baseline)
            || !same_policy_cohort(first_candidate, candidate)
        {
            return Err(PairedAccelerationError::MismatchedCohort);
        }
        if !seen_seeds.insert(baseline.seed) {
            return Err(PairedAccelerationError::DuplicateSeed);
        }
        if !seen_runs.insert(baseline.run_id.clone()) || !seen_runs.insert(candidate.run_id.clone())
        {
            return Err(PairedAccelerationError::MismatchedCohort);
        }
        paired_outcomes.push((
            baseline.seed,
            baseline.target_reached,
            candidate.target_reached,
        ));
    }

    paired_outcomes.sort_by_key(|(seed, _, _)| *seed);
    let observed_seeds = paired_outcomes
        .iter()
        .map(|(seed, _, _)| *seed)
        .collect::<Vec<_>>();
    if observed_seeds != seed_manifest.seeds {
        return Err(PairedAccelerationError::SeedRosterMismatch);
    }

    let mut both_reached_count = 0usize;
    let mut baseline_only_reached_count = 0usize;
    let mut candidate_only_reached_count = 0usize;
    let mut neither_reached_count = 0usize;
    let mut baseline_target_reached_count = 0usize;
    let mut candidate_target_reached_count = 0usize;
    for (_, baseline_reached, candidate_reached) in &paired_outcomes {
        match (*baseline_reached, *candidate_reached) {
            (true, true) => both_reached_count += 1,
            (true, false) => baseline_only_reached_count += 1,
            (false, true) => candidate_only_reached_count += 1,
            (false, false) => neither_reached_count += 1,
        }
        baseline_target_reached_count += usize::from(u8::from(*baseline_reached));
        candidate_target_reached_count += usize::from(u8::from(*candidate_reached));
    }

    let campaign_pair_count = pairs.len();
    let baseline_target_reach_rate =
        baseline_target_reached_count as f64 / campaign_pair_count as f64;
    let candidate_target_reach_rate =
        candidate_target_reached_count as f64 / campaign_pair_count as f64;
    let reach_rate_difference = candidate_target_reach_rate - baseline_target_reach_rate;

    let mut rng = SplitMix64::new(bootstrap_seed);
    let mut bootstrap_differences = Vec::with_capacity(bootstrap_resamples);
    for _ in 0..bootstrap_resamples {
        let mut baseline_success = 0usize;
        let mut candidate_success = 0usize;
        for _ in 0..paired_outcomes.len() {
            let index = rng.index(paired_outcomes.len());
            baseline_success += usize::from(u8::from(paired_outcomes[index].1));
            candidate_success += usize::from(u8::from(paired_outcomes[index].2));
        }
        let baseline_rate = baseline_success as f64 / campaign_pair_count as f64;
        let candidate_rate = candidate_success as f64 / campaign_pair_count as f64;
        bootstrap_differences.push(candidate_rate - baseline_rate);
    }
    bootstrap_differences.sort_by(f64::total_cmp);

    let alpha = 1.0 - confidence_level;
    let lower_confidence_bound = percentile(&bootstrap_differences, alpha / 2.0);
    let upper_confidence_bound = percentile(&bootstrap_differences, 1.0 - alpha / 2.0);
    let summary = PairedTargetReachabilitySummary {
        schema_version: PAIRED_ACCELERATION_STATS_VERSION,
        baseline_policy_id,
        candidate_policy_id,
        predeclared_seeds: seed_manifest.seeds.clone(),
        seed_manifest_sha256: seed_manifest.sha256.clone(),
        campaign_pair_count,
        bootstrap_resamples,
        bootstrap_seed,
        confidence_level,
        baseline_target_reached_count,
        baseline_target_reach_rate,
        candidate_target_reached_count,
        candidate_target_reach_rate,
        reach_rate_difference,
        lower_confidence_bound,
        upper_confidence_bound,
        both_reached_count,
        baseline_only_reached_count,
        candidate_only_reached_count,
        neither_reached_count,
    };
    summary.validate()?;
    Ok(summary)
}

fn same_policy_cohort(first: &CampaignRecord, current: &CampaignRecord) -> bool {
    first.lane_id == current.lane_id
        && first.policy_id == current.policy_id
        && first.dataset_digest == current.dataset_digest
        && first.model_digest == current.model_digest
        && first.evaluator_digest == current.evaluator_digest
        && first.environment_digest == current.environment_digest
        && first.config_digest == current.config_digest
        && first.target_definition_digest == current.target_definition_digest
        && first.budget_definition_digest == current.budget_definition_digest
        && first.objective_name == current.objective_name
        && first.objective_unit == current.objective_unit
        && first.objective_direction == current.objective_direction
        && first.max_evaluations == current.max_evaluations
        && first.total_cost_unit == current.total_cost_unit
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[middle - 1] + (sorted[middle] - sorted[middle - 1]) / 2.0
    } else {
        sorted[middle]
    }
}

fn percentile(sorted_values: &[f64], probability: f64) -> f64 {
    debug_assert!(!sorted_values.is_empty());
    debug_assert!((0.0..=1.0).contains(&probability));
    let position = probability * (sorted_values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let weight = position - lower as f64;
    sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight
}

/// SplitMix64: small, deterministic PRNG for bootstrap resampling. This is not
/// intended for cryptographic use; the bootstrap sequence is reproducibility data.
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

    fn index(&mut self, length: usize) -> usize {
        ((self.next_u64() as u128 * length as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::campaign::{CampaignStatus, ObjectiveDirection};
    use crate::contract::EvidenceReference;

    fn campaign(policy: &str, seed: u64, cost: f64, target_reached: bool) -> CampaignRecord {
        CampaignRecord {
            schema_version: 1,
            run_id: format!("run-{policy}-{seed}"),
            lane_id: "active-learning-policy".to_owned(),
            policy_id: policy.to_owned(),
            seed,
            dataset_digest: "a".repeat(64),
            model_digest: "b".repeat(64),
            evaluator_digest: "c".repeat(64),
            environment_digest: "9".repeat(64),
            config_digest: if policy == "random" {
                "d".repeat(64)
            } else {
                "8".repeat(64)
            },
            target_definition_digest: "e".repeat(64),
            budget_definition_digest: "f".repeat(64),
            objective_name: "best_validated_objective".to_owned(),
            objective_unit: "score".to_owned(),
            objective_direction: ObjectiveDirection::Maximize,
            used_evaluations: 10,
            max_evaluations: 100,
            elapsed_seconds: 3.0,
            total_cost_value: cost,
            total_cost_unit: "reference_eval_equivalent".to_owned(),
            terminal_status: if target_reached {
                CampaignStatus::Completed
            } else {
                CampaignStatus::Failed
            },
            target_reached,
            best_validated_objective: Some(0.75),
            evidence: vec![EvidenceReference {
                evidence_id: format!("evidence-{policy}-{seed}"),
                sha256: "1".repeat(64),
                role: "campaign_result".to_owned(),
            }],
            failure_reasons: if target_reached {
                Vec::new()
            } else {
                vec!["target_not_reached".to_owned()]
            },
        }
    }

    fn make_pairs(
        ratios: &[f64],
        all_targets_reached: bool,
    ) -> (Vec<CampaignRecord>, Vec<CampaignRecord>) {
        let baseline = ratios
            .iter()
            .enumerate()
            .map(|(index, _)| campaign("random", index as u64 + 1, 100.0, all_targets_reached))
            .collect::<Vec<_>>();
        let candidate = ratios
            .iter()
            .enumerate()
            .map(|(index, factor)| {
                campaign("symthaea_full", index as u64 + 1, 100.0 / factor, true)
            })
            .collect::<Vec<_>>();
        (baseline, candidate)
    }

    fn references<'a>(
        baseline: &'a [CampaignRecord],
        candidate: &'a [CampaignRecord],
    ) -> Vec<(&'a CampaignRecord, &'a CampaignRecord)> {
        baseline.iter().zip(candidate).collect()
    }

    fn seed_manifest(count: usize) -> SeedManifestV1 {
        let seeds = (1..=count as u64).collect::<Vec<_>>();
        SeedManifestV1::from_seeds(&seeds).unwrap()
    }

    fn summarize(
        pairs: &[(&CampaignRecord, &CampaignRecord)],
        count: usize,
        confidence: f64,
        resamples: usize,
        seed: u64,
    ) -> Result<PairedAccelerationSummary, PairedAccelerationError> {
        bootstrap_median_acceleration(pairs, &seed_manifest(count), confidence, resamples, seed)
    }

    #[test]
    fn constant_two_x_acceleration_has_exact_interval() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let summary = summarize(&references(&baseline, &candidate), 20, 0.95, 2_000, 17).unwrap();
        assert_eq!(summary.campaign_pair_count, 20);
        assert_eq!(summary.median_acceleration_factor, 2.0);
        assert_eq!(summary.lower_confidence_bound, 2.0);
        assert_eq!(summary.upper_confidence_bound, 2.0);
        assert_eq!(summary.candidate_faster_count, 20);
        assert_eq!(summary.candidate_slower_count, 0);
    }

    #[test]
    fn summary_is_deterministic_and_input_permutation_invariant() {
        let ratios = (0..20)
            .map(|index| 0.8 + index as f64 * 0.07)
            .collect::<Vec<_>>();
        let (baseline, candidate) = make_pairs(&ratios, true);
        let pairs = references(&baseline, &candidate);
        let forward = summarize(&pairs, 20, 0.95, 4_000, 123).unwrap();
        let reversed = pairs.iter().rev().copied().collect::<Vec<_>>();
        let backward = summarize(&reversed, 20, 0.95, 4_000, 123).unwrap();
        assert_eq!(forward, backward);
        assert_eq!(forward, summarize(&pairs, 20, 0.95, 4_000, 123).unwrap());
    }

    #[test]
    fn non_target_seed_blocks_interval_instead_of_being_silently_dropped() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let mut baseline = baseline;
        baseline[7].terminal_status = CampaignStatus::Failed;
        baseline[7].target_reached = false;
        baseline[7].failure_reasons = vec!["budget_exhausted".to_owned()];
        let err = summarize(&references(&baseline, &candidate), 20, 0.95, 2_000, 1).unwrap_err();
        assert_eq!(err, PairedAccelerationError::TargetNotReached);
    }

    #[test]
    fn duplicate_seed_and_incompatible_cohort_are_rejected() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let mut pairs = references(&baseline, &candidate);
        pairs[1] = (&baseline[0], &candidate[0]);
        assert_eq!(
            summarize(&pairs, 20, 0.95, 2_000, 3).unwrap_err(),
            PairedAccelerationError::DuplicateSeed
        );

        let (baseline, mut candidate) = make_pairs(&[2.0; 20], true);
        candidate[5].model_digest = "7".repeat(64);
        assert_eq!(
            summarize(&references(&baseline, &candidate), 20, 0.95, 2_000, 3).unwrap_err(),
            PairedAccelerationError::InvalidCampaign(CampaignError::MismatchedPolicies)
        );
    }

    #[test]
    fn target_reachability_retains_failed_runs_and_reports_paired_discordance() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let mut baseline = baseline;
        let mut candidate = candidate;
        for index in [1usize, 4, 7, 10, 13, 16] {
            baseline[index].terminal_status = CampaignStatus::Failed;
            baseline[index].target_reached = false;
            baseline[index].failure_reasons = vec!["budget_exhausted".to_owned()];
        }
        for index in [4usize, 10, 15] {
            candidate[index].terminal_status = CampaignStatus::Incomplete;
            candidate[index].target_reached = false;
            candidate[index].failure_reasons = vec!["evaluation_interrupted".to_owned()];
        }

        let summary = summarize_target_reachability(
            &references(&baseline, &candidate),
            &seed_manifest(20),
            0.95,
            2_000,
            91,
        )
        .unwrap();
        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/paired-target-reachability-summary-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(schema["properties"]["predeclared_seeds"]["minItems"], 20);
        assert_eq!(
            schema["properties"]["bootstrap_resamples"]["minimum"],
            1_000
        );
        let encoded = serde_json::to_string(&summary).unwrap();
        let decoded: PairedTargetReachabilitySummary = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, summary);
        assert_eq!(summary.campaign_pair_count, 20);
        assert_eq!(summary.baseline_target_reached_count, 14);
        assert_eq!(summary.candidate_target_reached_count, 17);
        assert_eq!(summary.both_reached_count, 13);
        assert_eq!(summary.baseline_only_reached_count, 1);
        assert_eq!(summary.candidate_only_reached_count, 4);
        assert_eq!(summary.neither_reached_count, 2);
        assert!((summary.reach_rate_difference - 0.15).abs() < 1e-12);
        summary.validate().unwrap();
    }

    #[test]
    fn target_reachability_requires_the_complete_frozen_seed_roster() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let pairs = references(&baseline, &candidate);
        assert_eq!(
            summarize_target_reachability(&pairs[..19], &seed_manifest(20), 0.95, 2_000, 3)
                .unwrap_err(),
            PairedAccelerationError::SeedRosterMismatch
        );
    }

    #[test]
    fn target_reachability_is_deterministic_under_pair_permutation() {
        let (baseline, mut candidate) = make_pairs(&[2.0; 20], true);
        for index in [2usize, 5, 8, 11, 14, 17] {
            candidate[index].terminal_status = CampaignStatus::Failed;
            candidate[index].target_reached = false;
            candidate[index].failure_reasons = vec!["budget_exhausted".to_owned()];
        }
        let pairs = references(&baseline, &candidate);
        let first =
            summarize_target_reachability(&pairs, &seed_manifest(20), 0.95, 3_000, 7).unwrap();
        let reversed = pairs.iter().rev().copied().collect::<Vec<_>>();
        let second =
            summarize_target_reachability(&reversed, &seed_manifest(20), 0.95, 3_000, 7).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn summary_round_trips_and_rejects_internally_inconsistent_reports() {
        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let summary = summarize(&references(&baseline, &candidate), 20, 0.95, 2_000, 17).unwrap();
        summary.validate().unwrap();

        let encoded = serde_json::to_string(&summary).unwrap();
        let decoded: PairedAccelerationSummary = serde_json::from_str(&encoded).unwrap();
        decoded.validate().unwrap();
        assert_eq!(decoded, summary);

        let schema: serde_json::Value = serde_json::from_str(include_str!(
            "../schemas/paired-acceleration-summary-v1.schema.json"
        ))
        .unwrap();
        assert_eq!(schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(schema["properties"]["predeclared_seeds"]["minItems"], 20);
        assert_eq!(
            schema["properties"]["seed_manifest_sha256"]["pattern"],
            "^[0-9a-f]{64}$"
        );

        let seed_schema: serde_json::Value =
            serde_json::from_str(include_str!("../schemas/seed-manifest-v1.schema.json")).unwrap();
        assert_eq!(seed_schema["properties"]["schema_version"]["const"], 1);
        assert_eq!(seed_schema["properties"]["seeds"]["minItems"], 20);

        let seed_fixture: SeedManifestV1 =
            serde_json::from_str(include_str!("../fixtures/seed-manifest-v1.json")).unwrap();
        seed_fixture.validate().unwrap();
        assert_eq!(seed_fixture.seeds, (101..=120).collect::<Vec<_>>());
        assert_eq!(
            schema["properties"]["campaign_pair_count"]["minimum"],
            MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL
        );
        assert_eq!(
            schema["properties"]["bootstrap_resamples"]["minimum"],
            MINIMUM_BOOTSTRAP_RESAMPLES
        );

        let mut invalid = summary;
        invalid.candidate_slower_count = 1;
        assert_eq!(
            invalid.validate().unwrap_err(),
            PairedAccelerationError::InvalidSummary
        );
    }

    #[test]
    fn seed_manifest_is_canonical_and_roster_mismatch_blocks_summary() {
        let manifest = SeedManifestV1::from_seeds(&(101..=120).rev().collect::<Vec<_>>()).unwrap();
        assert_eq!(
            manifest.sha256,
            "c8dd76e3828125d3984e95bef0cd50603cee2bef7933f1fec178866dd973ba59"
        );
        assert_eq!(manifest.seeds, (101..=120).collect::<Vec<_>>());
        manifest.validate().unwrap();

        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let pairs = references(&baseline, &candidate);
        assert_eq!(
            bootstrap_median_acceleration(&pairs, &manifest, 0.95, 2_000, 17).unwrap_err(),
            PairedAccelerationError::SeedRosterMismatch
        );
    }

    #[test]
    fn minimum_sample_and_configuration_gates_are_enforced() {
        let (baseline, candidate) = make_pairs(&[2.0; 19], true);
        assert_eq!(
            summarize(&references(&baseline, &candidate), 20, 0.95, 2_000, 3).unwrap_err(),
            PairedAccelerationError::TooFewCampaignPairs {
                actual: 19,
                minimum: MINIMUM_CAMPAIGNS_FOR_ACCELERATION_INTERVAL
            }
        );

        let (baseline, candidate) = make_pairs(&[2.0; 20], true);
        let pairs = references(&baseline, &candidate);
        assert_eq!(
            summarize(&pairs, 20, 1.0, 2_000, 3).unwrap_err(),
            PairedAccelerationError::InvalidConfidenceLevel
        );
        assert_eq!(
            summarize(&pairs, 20, 0.95, 999, 3).unwrap_err(),
            PairedAccelerationError::TooFewBootstrapResamples {
                actual: 999,
                minimum: MINIMUM_BOOTSTRAP_RESAMPLES
            }
        );
    }
}

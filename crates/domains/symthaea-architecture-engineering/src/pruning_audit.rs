//! Randomized, selection-conditioned audits for multi-fidelity pruning.
//!
//! Rank correlation and prediction-interval coverage are useful diagnostics, but neither
//! directly answers the operational question: how many candidates discarded by this rung
//! would have beaten the candidates that were promoted? This module samples the discarded
//! population uniformly without replacement, evaluates the sample at the same next fidelity
//! as the promoted cohort, and computes an exact one-sided hypergeometric upper confidence
//! bound on the fraction of the discarded population that would have crossed the promotion
//! frontier.
//!
//! The audit result is an aggregate decision, not candidate-level labels. A caller must keep
//! raw sentinel outcomes outside the search policy's information boundary. This module does
//! not itself provide process isolation or cryptographic access control.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

use super::multi_fidelity::{validate_fidelity_ladder, FidelityLevel};

/// A candidate's score measured at the audit target fidelity (maximize is better).
#[derive(Debug, Clone, PartialEq)]
pub struct AuditScore {
    pub candidate_id: String,
    pub fidelity_index: usize,
    pub score: f64,
}

/// Pre-registered risk limits for a pruning audit.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningAuditConfig {
    /// Minimum number of uniformly sampled discarded candidates required to permit pruning.
    pub min_audit_samples: usize,
    /// Largest acceptable upper confidence bound on the false-pruned fraction.
    pub max_false_prune_rate: f64,
    /// One-sided confidence level for the finite-population upper bound (for example 0.95).
    pub confidence_level: f64,
}

/// What the controller should do after observing the audit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningDisposition {
    /// The pre-registered sample-size and finite-population risk limits both passed.
    ContinuePruning,
    /// The sample was too small to support a pruning decision.
    DisablePruningInsufficientAudit,
    /// The upper confidence bound exceeded the pre-registered risk limit.
    DisablePruningObservedRisk,
}

/// Aggregate report returned to the controller. Raw candidate scores are deliberately absent.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningAuditReport {
    pub target_fidelity_index: usize,
    pub selected_frontier_score: f64,
    pub pruned_population_size: usize,
    pub audit_sample_size: usize,
    pub false_pruned_count: usize,
    pub empirical_false_prune_rate: f64,
    pub upper_bound_false_pruned_count: usize,
    pub upper_bound_false_prune_rate: f64,
    pub confidence_level: f64,
    pub sample_seed: u64,
    pub population_sha256: String,
    pub sample_sha256: String,
    pub disposition: PruningDisposition,
}

/// A proof-carrying, deterministic sample selection. Fields are private so callers cannot
/// silently substitute a hand-picked set; persisted receipts can be reconstituted only by
/// verify_pruning_audit_sample, which replays the selection against the full population.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruningAuditSample {
    seed: u64,
    population_ids: Vec<String>,
    sampled_candidate_ids: Vec<String>,
    population_sha256: String,
    sample_sha256: String,
}

impl PruningAuditSample {
    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn population_ids(&self) -> &[String] {
        &self.population_ids
    }

    pub fn sampled_candidate_ids(&self) -> &[String] {
        &self.sampled_candidate_ids
    }

    pub fn population_sha256(&self) -> &str {
        &self.population_sha256
    }

    pub fn sample_sha256(&self) -> &str {
        &self.sample_sha256
    }
}

/// Validation or accounting failure. Invalid audits never return partial decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PruningAuditError {
    pub field: String,
    pub message: String,
}

impl PruningAuditError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for PruningAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for PruningAuditError {}

/// Select a simple random sample without replacement from the candidates that would be
/// pruned. IDs are returned in deterministic draw order. The caller must commit to this
/// sample before observing any target-fidelity audit scores.
fn sample_candidate_ids(
    pruned_candidate_ids: &[String],
    sample_count: usize,
    seed: u64,
) -> Result<Vec<String>, PruningAuditError> {
    if pruned_candidate_ids.is_empty() {
        return Err(PruningAuditError::new(
            "pruned_candidate_ids",
            "must not be empty",
        ));
    }
    if sample_count == 0 || sample_count > pruned_candidate_ids.len() {
        return Err(PruningAuditError::new(
            "sample_count",
            "must be in 1..=pruned_candidate_ids.len()",
        ));
    }

    let mut seen = HashSet::with_capacity(pruned_candidate_ids.len());
    for (index, candidate_id) in pruned_candidate_ids.iter().enumerate() {
        if candidate_id.trim().is_empty() {
            return Err(PruningAuditError::new(
                format!("pruned_candidate_ids[{index}]"),
                "must be a non-empty string",
            ));
        }
        if !seen.insert(candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("pruned_candidate_ids[{index}]"),
                format!("duplicate candidate ID {candidate_id:?}"),
            ));
        }
    }

    let mut sample = pruned_candidate_ids.to_vec();
    sample.sort();
    let mut state = seed;
    for index in (1..sample.len()).rev() {
        let selected = uniform_below(&mut state, (index + 1) as u64);
        sample.swap(index, selected);
    }
    sample.truncate(sample_count);
    Ok(sample)
}

/// Prepare a deterministic, uniform-without-replacement audit sample and bind it to the
/// canonical candidate population. The seed must be independently chosen and committed before
/// target-fidelity audit outcomes are observed.
pub fn prepare_pruning_audit_sample(
    pruned_candidate_ids: &[String],
    sample_count: usize,
    seed: u64,
) -> Result<PruningAuditSample, PruningAuditError> {
    let mut population_ids = pruned_candidate_ids.to_vec();
    population_ids.sort();
    let sampled_candidate_ids = sample_candidate_ids(&population_ids, sample_count, seed)?;
    Ok(PruningAuditSample {
        seed,
        population_sha256: digest_ids(
            b"symthaea.pruning-audit.population.v0.1",
            &population_ids,
        ),
        sample_sha256: digest_ids(
            b"symthaea.pruning-audit.sample.v0.1",
            &sampled_candidate_ids,
        ),
        population_ids,
        sampled_candidate_ids,
    })
}

/// Reconstitute a persisted sample receipt only when replaying the declared seed against the
/// complete candidate population produces exactly the recorded sampled IDs.
pub fn verify_pruning_audit_sample(
    pruned_candidate_ids: &[String],
    sampled_candidate_ids: &[String],
    seed: u64,
) -> Result<PruningAuditSample, PruningAuditError> {
    let verified = prepare_pruning_audit_sample(
        pruned_candidate_ids,
        sampled_candidate_ids.len(),
        seed,
    )?;
    if verified.sampled_candidate_ids != sampled_candidate_ids {
        return Err(PruningAuditError::new(
            "sampled_candidate_ids",
            "receipt does not match the deterministic sample generated from the population and seed",
        ));
    }
    Ok(verified)
}

fn digest_ids(domain: &[u8], ids: &[String]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for candidate_id in ids {
        hasher.update((candidate_id.len() as u64).to_be_bytes());
        hasher.update(candidate_id.as_bytes());
    }
    let digest = hasher.finalize();
    let alphabet = b"0123456789abcdef";
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest.iter().copied() {
        encoded.push(alphabet[(byte >> 4) as usize] as char);
        encoded.push(alphabet[(byte & 0x0f) as usize] as char);
    }
    encoded
}

/// Audit whether a uniformly sampled subset of the pruned population would have crossed the
/// observed promotion frontier at the same target fidelity.
///
/// selected_scores must contain the entire promoted cohort's target-fidelity scores.
/// audit_sample must be the private receipt produced by prepare_pruning_audit_sample (or
/// reconstituted with verify_pruning_audit_sample) from the complete pruned population.
/// audited_pruned_scores must contain exactly one result for every receipt sentinel, evaluated
/// at exactly the same fidelity as selected_scores. Missing or failed sentinel evaluations
/// produce an error; the caller must then fail closed rather than continue pruning.
///
/// A false-pruned sentinel is one whose score is strictly greater than the worst selected
/// score. Ties do not count as false-prunes; deterministic tie-breaking remains the caller's
/// promotion-policy responsibility.
pub fn audit_pruning(
    selected_scores: &[AuditScore],
    audit_sample: &PruningAuditSample,
    audited_pruned_scores: &[AuditScore],
    config: &PruningAuditConfig,
) -> Result<PruningAuditReport, PruningAuditError> {
    let verified = verify_pruning_audit_sample(
        &audit_sample.population_ids,
        &audit_sample.sampled_candidate_ids,
        audit_sample.seed,
    )?;
    if verified.population_sha256 != audit_sample.population_sha256
        || verified.sample_sha256 != audit_sample.sample_sha256
    {
        return Err(PruningAuditError::new(
            "audit_sample",
            "sample receipt digest mismatch",
        ));
    }
    if audited_pruned_scores.len() != verified.sampled_candidate_ids.len() {
        return Err(PruningAuditError::new(
            "audited_pruned_scores",
            "exactly one target-fidelity outcome is required for every sampled sentinel",
        ));
    }
    let expected_ids: HashSet<&str> = verified
        .sampled_candidate_ids
        .iter()
        .map(String::as_str)
        .collect();
    let mut observed_ids = HashSet::with_capacity(audited_pruned_scores.len());
    for (index, observation) in audited_pruned_scores.iter().enumerate() {
        if !observed_ids.insert(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("audited_pruned_scores[{index}].candidate_id"),
                "duplicate audit outcome",
            ));
        }
        if !expected_ids.contains(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("audited_pruned_scores[{index}].candidate_id"),
                "outcome is not part of the committed audit sample",
            ));
        }
    }

    let mut report = audit_pruning_unverified(
        selected_scores,
        &verified.population_ids,
        audited_pruned_scores,
        config,
    )?;
    report.sample_seed = verified.seed;
    report.population_sha256 = verified.population_sha256;
    report.sample_sha256 = verified.sample_sha256;
    Ok(report)
}

fn audit_pruning_unverified(
    selected_scores: &[AuditScore],
    pruned_population_ids: &[String],
    audited_pruned_scores: &[AuditScore],
    config: &PruningAuditConfig,
) -> Result<PruningAuditReport, PruningAuditError> {
    validate_config(config)?;
    if selected_scores.is_empty() {
        return Err(PruningAuditError::new(
            "selected_scores",
            "the promoted cohort must contain at least one target-fidelity score",
        ));
    }
    if pruned_population_ids.is_empty() {
        return Err(PruningAuditError::new(
            "pruned_population_ids",
            "must not be empty when auditing a pruning decision",
        ));
    }
    if audited_pruned_scores.is_empty() {
        return Err(PruningAuditError::new(
            "audited_pruned_scores",
            "at least one randomly selected sentinel is required",
        ));
    }
    if audited_pruned_scores.len() > pruned_population_ids.len() {
        return Err(PruningAuditError::new(
            "audited_pruned_scores",
            "audit sample cannot exceed the pruned population",
        ));
    }

    let (target_fidelity_index, frontier_score, selected_ids) =
        validate_score_cohort(selected_scores, "selected_scores")?;
    let mut pruned_ids = HashSet::with_capacity(pruned_population_ids.len());
    for (index, candidate_id) in pruned_population_ids.iter().enumerate() {
        if candidate_id.trim().is_empty() {
            return Err(PruningAuditError::new(
                format!("pruned_population_ids[{index}]"),
                "must be a non-empty string",
            ));
        }
        if !pruned_ids.insert(candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("pruned_population_ids[{index}]"),
                format!("duplicate candidate ID {candidate_id:?}"),
            ));
        }
        if selected_ids.contains(candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("pruned_population_ids[{index}]"),
                "a candidate cannot be both selected and pruned",
            ));
        }
    }

    let mut audited_ids = HashSet::with_capacity(audited_pruned_scores.len());
    let mut false_pruned_count = 0usize;
    for (index, observation) in audited_pruned_scores.iter().enumerate() {
        let field = format!("audited_pruned_scores[{index}]");
        if observation.candidate_id.trim().is_empty() {
            return Err(PruningAuditError::new(
                format!("{field}.candidate_id"),
                "must be a non-empty string",
            ));
        }
        if !audited_ids.insert(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("{field}.candidate_id"),
                format!("duplicate candidate ID {:?}", observation.candidate_id),
            ));
        }
        if !pruned_ids.contains(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("{field}.candidate_id"),
                "audited candidate is not in the declared pruned population",
            ));
        }
        if selected_ids.contains(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("{field}.candidate_id"),
                "audited candidate overlaps the promoted cohort",
            ));
        }
        if observation.fidelity_index != target_fidelity_index {
            return Err(PruningAuditError::new(
                format!("{field}.fidelity_index"),
                "selected and audited candidates must be evaluated at the same target fidelity",
            ));
        }
        if !observation.score.is_finite() {
            return Err(PruningAuditError::new(
                format!("{field}.score"),
                "must be finite",
            ));
        }
        if observation.score > frontier_score {
            false_pruned_count += 1;
        }
    }

    let population_size = pruned_population_ids.len();
    let sample_size = audited_pruned_scores.len();
    let upper_bound_count = finite_population_upper_bound(
        population_size,
        sample_size,
        false_pruned_count,
        config.confidence_level,
    )?;
    let empirical_rate = false_pruned_count as f64 / sample_size as f64;
    let upper_bound_rate = upper_bound_count as f64 / population_size as f64;

    let disposition = if sample_size < config.min_audit_samples {
        PruningDisposition::DisablePruningInsufficientAudit
    } else if upper_bound_rate <= config.max_false_prune_rate {
        PruningDisposition::ContinuePruning
    } else {
        PruningDisposition::DisablePruningObservedRisk
    };

    Ok(PruningAuditReport {
        target_fidelity_index,
        selected_frontier_score: frontier_score,
        pruned_population_size: population_size,
        audit_sample_size: sample_size,
        false_pruned_count,
        empirical_false_prune_rate: empirical_rate,
        upper_bound_false_pruned_count: upper_bound_count,
        upper_bound_false_prune_rate: upper_bound_rate,
        confidence_level: config.confidence_level,
        sample_seed: 0,
        population_sha256: String::new(),
        sample_sha256: String::new(),
        disposition,
    })
}

/// Calculate the incremental cost for auditing sample_count candidates at the next rung.
/// This difference formula is valid only when higher fidelity resumes the same cumulative
/// evaluation. Fresh restarts or non-nested workloads must be charged using their measured
/// actual costs instead.
pub fn estimate_resumable_audit_cost_units(
    levels: &[FidelityLevel],
    current_fidelity_index: usize,
    sample_count: usize,
) -> Result<u64, PruningAuditError> {
    validate_fidelity_ladder(levels).map_err(|error| {
        PruningAuditError::new(
            format!("fidelity_ladder.{}", error.field),
            error.message,
        )
    })?;
    if current_fidelity_index >= levels.len() - 1 {
        return Err(PruningAuditError::new(
            "current_fidelity_index",
            "must identify a fidelity with a subsequent level",
        ));
    }
    if sample_count == 0 {
        return Err(PruningAuditError::new(
            "sample_count",
            "must be positive",
        ));
    }
    let increment = levels[current_fidelity_index + 1].cumulative_cost_units
        - levels[current_fidelity_index].cumulative_cost_units;
    increment.checked_mul(sample_count as u64).ok_or_else(|| {
        PruningAuditError::new(
            "audit_cost_units",
            "u64 overflow while accounting for audit evaluations",
        )
    })
}

fn validate_config(config: &PruningAuditConfig) -> Result<(), PruningAuditError> {
    if config.min_audit_samples == 0 {
        return Err(PruningAuditError::new(
            "min_audit_samples",
            "must be positive",
        ));
    }
    if !config.max_false_prune_rate.is_finite()
        || !(0.0..=1.0).contains(&config.max_false_prune_rate)
    {
        return Err(PruningAuditError::new(
            "max_false_prune_rate",
            "must be finite and in [0, 1]",
        ));
    }
    if !config.confidence_level.is_finite()
        || config.confidence_level <= 0.0
        || config.confidence_level >= 1.0
    {
        return Err(PruningAuditError::new(
            "confidence_level",
            "must be finite and in (0, 1)",
        ));
    }
    Ok(())
}

fn validate_score_cohort<'a>(
    observations: &'a [AuditScore],
    field: &str,
) -> Result<(usize, f64, HashSet<&'a str>), PruningAuditError> {
    let first = observations.first().ok_or_else(|| {
        PruningAuditError::new(field, "must not be empty")
    })?;
    let fidelity_index = first.fidelity_index;
    let mut frontier_score = f64::INFINITY;
    let mut ids = HashSet::with_capacity(observations.len());
    for (index, observation) in observations.iter().enumerate() {
        let entry = format!("{field}[{index}]");
        if observation.candidate_id.trim().is_empty() {
            return Err(PruningAuditError::new(
                format!("{entry}.candidate_id"),
                "must be a non-empty string",
            ));
        }
        if !ids.insert(observation.candidate_id.as_str()) {
            return Err(PruningAuditError::new(
                format!("{entry}.candidate_id"),
                format!("duplicate candidate ID {:?}", observation.candidate_id),
            ));
        }
        if observation.fidelity_index != fidelity_index {
            return Err(PruningAuditError::new(
                format!("{entry}.fidelity_index"),
                "all selected candidates must use the same target fidelity",
            ));
        }
        if !observation.score.is_finite() {
            return Err(PruningAuditError::new(
                format!("{entry}.score"),
                "must be finite",
            ));
        }
        frontier_score = frontier_score.min(observation.score);
    }
    Ok((fidelity_index, frontier_score, ids))
}

/// Exact conservative upper confidence bound on the number of successes in a finite
/// population, given a simple random sample without replacement.
///
/// Let N be the population size, n the sample size and x the observed successes. This returns
/// the largest K for which P[X <= x | X ~ Hypergeometric(N, K, n)] is at least 1-confidence.
/// The inversion is discrete and therefore conservative.
fn finite_population_upper_bound(
    population_size: usize,
    sample_size: usize,
    observed_successes: usize,
    confidence_level: f64,
) -> Result<usize, PruningAuditError> {
    if population_size == 0
        || sample_size == 0
        || sample_size > population_size
        || observed_successes > sample_size
        || !confidence_level.is_finite()
        || confidence_level <= 0.0
        || confidence_level >= 1.0
    {
        return Err(PruningAuditError::new(
            "finite_population_bound",
            "requires 0 < sample_size <= population_size, successes <= sample_size, and confidence in (0, 1)",
        ));
    }
    if sample_size == population_size {
        return Ok(observed_successes);
    }
    if observed_successes == sample_size {
        return Ok(population_size);
    }

    let log_alpha = (1.0 - confidence_level).ln();
    let mut low = observed_successes;
    let mut high = population_size;
    while low < high {
        let difference = high - low;
        let middle = low + difference / 2 + difference % 2;
        let log_cdf = hypergeometric_log_cdf(
            population_size,
            middle,
            sample_size,
            observed_successes,
        );
        if log_cdf >= log_alpha {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    Ok(low)
}

fn hypergeometric_log_cdf(
    population_size: usize,
    success_population: usize,
    sample_size: usize,
    observed_successes: usize,
) -> f64 {
    let failure_population = population_size - success_population;
    let first = sample_size.saturating_sub(failure_population);
    let last = sample_size.min(success_population).min(observed_successes);
    if first > last {
        return f64::NEG_INFINITY;
    }

    let denominator = log_choose(population_size, sample_size);
    let mut terms = Vec::with_capacity(last - first + 1);
    for successes in first..=last {
        let failures = sample_size - successes;
        terms.push(
            log_choose(success_population, successes)
                + log_choose(failure_population, failures)
                - denominator,
        );
    }
    let maximum = terms
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return maximum;
    }
    let scaled_sum: f64 = terms.iter().map(|term| (term - maximum).exp()).sum();
    maximum + scaled_sum.ln()
}

fn log_choose(population: usize, selected: usize) -> f64 {
    if selected > population {
        return f64::NEG_INFINITY;
    }
    let count = selected.min(population - selected);
    let mut result = 0.0;
    for index in 1..=count {
        result += ((population - count + index) as f64).ln() - (index as f64).ln();
    }
    result
}

fn uniform_below(state: &mut u64, upper_exclusive: u64) -> usize {
    let rejection_threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
    loop {
        let value = next_splitmix64(state);
        if value >= rejection_threshold {
            return (value % upper_exclusive) as usize;
        }
    }
}

fn next_splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(min_audit_samples: usize, max_false_prune_rate: f64) -> PruningAuditConfig {
        PruningAuditConfig {
            min_audit_samples,
            max_false_prune_rate,
            confidence_level: 0.95,
        }
    }

    fn selected_scores(fidelity_index: usize) -> Vec<AuditScore> {
        vec![
            AuditScore { candidate_id: "kept-a".into(), fidelity_index, score: 0.9 },
            AuditScore { candidate_id: "kept-b".into(), fidelity_index, score: 0.8 },
            AuditScore { candidate_id: "kept-c".into(), fidelity_index, score: 0.7 },
        ]
    }

    fn ids(prefix: &str, count: usize) -> Vec<String> {
        (0..count).map(|index| format!("{prefix}-{index}")).collect()
    }

    #[test]
    fn audit_sample_is_reproducible_unique_and_without_replacement() {
        let candidates = ids("pruned", 100);
        let first = prepare_pruning_audit_sample(&candidates, 30, 42).unwrap();
        let second = prepare_pruning_audit_sample(&candidates, 30, 42).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.sampled_candidate_ids().len(), 30);
        assert_eq!(
            first.sampled_candidate_ids().iter().collect::<HashSet<_>>().len(),
            30
        );
        assert!(first.sampled_candidate_ids().iter().all(|id| candidates.contains(id)));
        assert_eq!(
            verify_pruning_audit_sample(
                &candidates,
                first.sampled_candidate_ids(),
                first.seed()
            ).unwrap(),
            first
        );
    }

    #[test]
    fn sample_receipt_rejects_cherry_picking_and_missing_outcomes() {
        let population = ids("pruned", 100);
        let receipt = prepare_pruning_audit_sample(&population, 20, 123).unwrap();
        let mut altered = receipt.sampled_candidate_ids().to_vec();
        let unselected = population
            .iter()
            .find(|candidate_id| !receipt.sampled_candidate_ids().contains(candidate_id))
            .unwrap()
            .clone();
        altered[0] = unselected;
        assert!(verify_pruning_audit_sample(&population, &altered, receipt.seed()).is_err());

        let selected = selected_scores(1);
        let complete: Vec<AuditScore> = receipt
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 1,
                score: 0.1,
            })
            .collect();
        assert!(audit_pruning(
            &selected,
            &receipt,
            &complete[..complete.len() - 1],
            &config(1, 1.0)
        )
        .is_err());
    }

    #[test]
    fn sample_receipt_is_invariant_to_population_input_order() {
        let population = ids("pruned", 80);
        let mut reversed = population.clone();
        reversed.reverse();
        let forward = prepare_pruning_audit_sample(&population, 24, 77).unwrap();
        let backward = prepare_pruning_audit_sample(&reversed, 24, 77).unwrap();
        assert_eq!(forward.population_sha256(), backward.population_sha256());
        assert_eq!(forward.sample_sha256(), backward.sample_sha256());
        assert_eq!(forward.sampled_candidate_ids(), backward.sampled_candidate_ids());
    }

    #[test]
    fn audit_sample_rejects_empty_duplicate_and_out_of_range_inputs() {
        assert!(prepare_pruning_audit_sample(&[], 1, 1).is_err());
        assert!(prepare_pruning_audit_sample(&["a".into()], 0, 1).is_err());
        assert!(prepare_pruning_audit_sample(&["a".into()], 2, 1).is_err());
        assert!(prepare_pruning_audit_sample(&["a".into(), "a".into()], 1, 1).is_err());
        assert!(prepare_pruning_audit_sample(&[" ".into()], 1, 1).is_err());
    }

    #[test]
    fn finite_population_bound_is_exact_for_a_census() {
        assert_eq!(finite_population_upper_bound(8, 8, 2, 0.95).unwrap(), 2);
        assert_eq!(finite_population_upper_bound(8, 8, 0, 0.95).unwrap(), 0);
    }

    #[test]
    fn finite_population_bound_is_monotone_in_observed_false_prunes() {
        let none = finite_population_upper_bound(100, 30, 0, 0.95).unwrap();
        let one = finite_population_upper_bound(100, 30, 1, 0.95).unwrap();
        let two = finite_population_upper_bound(100, 30, 2, 0.95).unwrap();
        assert!(none < one);
        assert!(one < two);
        assert!(two <= 100);
    }

    #[test]
    fn exact_zero_miss_census_qualifies_for_pruning() {
        let selected = selected_scores(2);
        let population = ids("pruned", 40);
        let sample = prepare_pruning_audit_sample(&population, 40, 17).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 2,
                score: 0.1,
            })
            .collect();
        let report = audit_pruning(&selected, &sample, &audited, &config(30, 0.01)).unwrap();
        assert_eq!(report.false_pruned_count, 0);
        assert_eq!(report.upper_bound_false_pruned_count, 0);
        assert_eq!(report.upper_bound_false_prune_rate, 0.0);
        assert_eq!(report.disposition, PruningDisposition::ContinuePruning);
    }

    #[test]
    fn audit_detects_a_candidate_that_beats_the_selected_frontier() {
        let selected = selected_scores(1);
        let population = ids("pruned", 30);
        let sample = prepare_pruning_audit_sample(&population, 30, 1).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .enumerate()
            .map(|(index, candidate_id)| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 1,
                score: if index == 0 { 0.95 } else { 0.1 },
            })
            .collect();
        let report = audit_pruning(&selected, &sample, &audited, &config(30, 0.0)).unwrap();
        assert_eq!(report.false_pruned_count, 1);
        assert_eq!(
            report.disposition,
            PruningDisposition::DisablePruningObservedRisk
        );
    }

    #[test]
    fn insufficient_audit_sample_disables_pruning_even_with_zero_misses() {
        let selected = selected_scores(1);
        let population = ids("pruned", 100);
        let sample = prepare_pruning_audit_sample(&population, 10, 9).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 1,
                score: 0.1,
            })
            .collect();
        let report = audit_pruning(&selected, &sample, &audited, &config(30, 1.0)).unwrap();
        assert_eq!(
            report.disposition,
            PruningDisposition::DisablePruningInsufficientAudit
        );
    }

    #[test]
    fn observed_tail_risk_disables_pruning_under_a_predeclared_limit() {
        let selected = selected_scores(3);
        let population = ids("pruned", 50);
        let sample = prepare_pruning_audit_sample(&population, 50, 5).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 3,
                score: 0.85,
            })
            .collect();
        let report = audit_pruning(&selected, &sample, &audited, &config(30, 0.1)).unwrap();
        assert_eq!(report.false_pruned_count, 50);
        assert_eq!(report.upper_bound_false_prune_rate, 1.0);
        assert_eq!(
            report.disposition,
            PruningDisposition::DisablePruningObservedRisk
        );
    }

    #[test]
    fn ties_at_the_selected_frontier_are_not_counted_as_false_prunes() {
        let selected = selected_scores(2);
        let population = ids("pruned", 30);
        let sample = prepare_pruning_audit_sample(&population, 30, 77).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 2,
                score: 0.7,
            })
            .collect();
        let report = audit_pruning(&selected, &sample, &audited, &config(30, 0.0)).unwrap();
        assert_eq!(report.false_pruned_count, 0);
    }

    #[test]
    fn rejects_duplicate_cohorts_mixed_fidelities_and_non_finite_scores() {
        let mut selected = selected_scores(1);
        selected[1].candidate_id = selected[0].candidate_id.clone();
        let population = ids("pruned", 30);
        let sample = prepare_pruning_audit_sample(&population, 30, 13).unwrap();
        let audited: Vec<AuditScore> = sample
            .sampled_candidate_ids()
            .iter()
            .map(|candidate_id| AuditScore {
                candidate_id: candidate_id.clone(),
                fidelity_index: 1,
                score: 0.1,
            })
            .collect();
        assert!(audit_pruning(&selected, &sample, &audited, &config(30, 0.1)).is_err());

        let selected = selected_scores(1);
        let mut mixed = audited.clone();
        mixed[0].fidelity_index = 2;
        assert!(audit_pruning(&selected, &sample, &mixed, &config(30, 0.1)).is_err());

        let mut invalid = audited.clone();
        invalid[0].score = f64::NAN;
        assert!(audit_pruning(&selected, &sample, &invalid, &config(30, 0.1)).is_err());
    }

    #[test]
    fn rejects_audited_ids_outside_the_pruned_population_or_inside_selected() {
        let selected = selected_scores(1);
        let population = ids("pruned", 30);
        let sample = prepare_pruning_audit_sample(&population, 1, 21).unwrap();
        let outside = vec![AuditScore {
            candidate_id: "unknown".into(),
            fidelity_index: 1,
            score: 0.1,
        }];
        assert!(audit_pruning(&selected, &sample, &outside, &config(1, 0.1)).is_err());

        let overlap_population = vec!["kept-a".to_owned()];
        let overlap_sample = prepare_pruning_audit_sample(&overlap_population, 1, 21).unwrap();
        let overlap = vec![AuditScore {
            candidate_id: "kept-a".into(),
            fidelity_index: 1,
            score: 0.1,
        }];
        assert!(audit_pruning(&selected, &overlap_sample, &overlap, &config(1, 0.1)).is_err());
    }

    #[test]
    fn audit_cost_uses_incremental_cumulative_cost_and_checks_overflow() {
        let levels = vec![
            FidelityLevel { id: "proxy".into(), cumulative_cost_units: 1 },
            FidelityLevel { id: "short".into(), cumulative_cost_units: 4 },
            FidelityLevel { id: "full".into(), cumulative_cost_units: 64 },
        ];
        assert_eq!(
            estimate_resumable_audit_cost_units(&levels, 1, 3).unwrap(),
            180
        );
        assert!(estimate_resumable_audit_cost_units(&levels, 2, 1).is_err());
        assert!(estimate_resumable_audit_cost_units(&levels, 0, 0).is_err());

        let huge = vec![
            FidelityLevel { id: "a".into(), cumulative_cost_units: 1 },
            FidelityLevel { id: "b".into(), cumulative_cost_units: u64::MAX },
        ];
        assert!(estimate_resumable_audit_cost_units(&huge, 0, 2).is_err());
    }

    #[test]
    fn rejects_invalid_risk_configuration_and_empty_audit() {
        let selected = selected_scores(1);
        let population = ids("pruned", 2);
        let sample = prepare_pruning_audit_sample(&population, 1, 8).unwrap();
        let audit = vec![AuditScore {
            candidate_id: sample.sampled_candidate_ids()[0].clone(),
            fidelity_index: 1,
            score: 0.1,
        }];
        assert!(audit_pruning(&selected, &sample, &[], &config(1, 0.1)).is_err());
        assert!(audit_pruning(&selected, &sample, &audit, &config(0, 0.1)).is_err());
        assert!(audit_pruning(&selected, &sample, &audit, &config(1, f64::NAN)).is_err());
        assert_eq!(
            audit_pruning(&selected, &sample, &audit, &config(1, 0.1))
                .unwrap()
                .confidence_level,
            0.95
        );
    }
}

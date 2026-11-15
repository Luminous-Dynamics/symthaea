use std::collections::BTreeSet;

pub const SAMPLING_ALGORITHM: &str = "splitmix64-partial-fisher-yates-v1";
pub const BOUND_METHOD: &str = "exact-one-sided-hypergeometric-bonferroni-v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AuditOutcome {
    ConfirmedFeasible,
    ConfirmedInfeasible,
    Failed,
    NonConverged,
    Unsupported,
    Inconclusive,
}

impl AuditOutcome {
    fn is_potentially_feasible(self) -> bool {
        self != Self::ConfirmedInfeasible
    }

    fn is_unresolved(self) -> bool {
        matches!(
            self,
            Self::Failed | Self::NonConverged | Self::Unsupported | Self::Inconclusive
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenCandidateUniverse {
    /// Complete, strictly lexicographically ordered candidate identity list.
    pub candidate_ids: Vec<String>,
    /// Declared SHA-256 identity of the canonical frozen-universe manifest.
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedCandidate {
    pub candidate_id: String,
    /// Reference result for every accepted candidate, not merely known successes.
    pub outcome: AuditOutcome,
    /// Evidence identity for the reference evaluation, including unresolved runs.
    pub evidence_ref: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedCandidateManifest {
    /// Complete accepted population, in strict lexicographic identity order.
    pub candidates: Vec<AcceptedCandidate>,
    pub manifest_sha256: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditSample {
    pub candidate_id: String,
    pub outcome: AuditOutcome,
    /// Evidence identity for the solver result, including failed attempts.
    pub evidence_ref: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RejectStratum {
    pub stratum_id: String,
    /// Digest of the precise predicate/definition used to form this stratum.
    pub stratum_definition_sha256: String,
    /// Digest of the frozen, lexicographically sorted rejected-candidate ID list.
    pub population_manifest_sha256: String,
    /// Independently checkable record of sample selection and outcomes.
    pub selection_receipt_ref: String,
    /// Complete rejected population for this stratum, in strict lexical order.
    pub rejected_candidate_ids: Vec<String>,
    /// Sample size frozen before any sample outcome is observed.
    pub sample_size: usize,
    /// Sample outcomes in deterministic draw order; count must equal sample_size.
    pub samples: Vec<AuditSample>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RejectAuditPlan {
    pub frozen_universe: FrozenCandidateUniverse,
    /// Seed whose commitment record is declared to predate audit outcomes.
    pub sampling_seed: u64,
    pub seed_commitment_ref: String,
    pub seed_committed_before_outcomes: bool,
    pub familywise_confidence_level: f64,
    pub target_feasible_recall: f64,
    /// Complete accepted population, including infeasible and unresolved outcomes.
    pub accepted: AcceptedCandidateManifest,
    /// Disjoint strata whose union is exactly the rejected portion of the universe.
    pub strata: Vec<RejectStratum>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AuditError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidConfidenceLevel,
    InvalidRecallTarget,
    SeedNotCommittedBeforeOutcomes,
    DuplicateStratumId(String),
    EmptyStratumPopulation(String),
    DuplicateCandidateId(String),
    CandidateOutsideFrozenUniverse(String),
    CandidateMissingFromPartition(String),
    UnsortedCandidatePopulation(String),
    SampleSizeExceedsPopulation(String),
    SampleCountMismatch {
        stratum_id: String,
        declared: usize,
        supplied: usize,
    },
    SampleIdentityMismatch {
        stratum_id: String,
        sample_index: usize,
    },
    InvalidHypergeometricInputs,
    IntegerOverflow,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StratumAuditReport {
    pub stratum_id: String,
    pub population_size: u64,
    pub sample_size: u64,
    pub inclusion_probability: f64,
    pub confirmed_feasible_in_sample: u64,
    pub unresolved_in_sample: u64,
    /// Feasible plus unresolved observations; unresolved is treated as positive.
    pub potential_feasible_in_sample: u64,
    /// Exact upper confidence count for feasible-or-unresolved items in the stratum.
    pub upper_missed_feasible_count: u64,
    pub upper_missed_feasible_rate: f64,
    pub per_stratum_error_probability: f64,
    pub is_full_census: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RejectAuditReport {
    pub bound_method: &'static str,
    pub sampling_algorithm: &'static str,
    pub familywise_confidence_level: f64,
    pub target_feasible_recall: f64,
    pub frozen_candidate_count: u64,
    pub accepted_candidate_count: u64,
    pub accepted_confirmed_feasible_count: u64,
    pub accepted_confirmed_infeasible_count: u64,
    pub accepted_unresolved_count: u64,
    pub total_rejected_population: u64,
    pub total_audited_candidates: u64,
    pub missed_feasible_count_upper_bound: u64,
    /// Conservative lower bound; absent when no feasible population can be bounded.
    pub feasible_recall_lower_bound: Option<f64>,
    pub target_met: bool,
    pub strata: Vec<StratumAuditReport>,
}

/// Reproduce the audit draw for a stratum.
///
/// The population IDs must be frozen, unique and sorted lexicographically.
/// The returned identities are in draw order, not sorted order.
pub fn deterministic_audit_sample_ids(
    stratum_id: &str,
    population_ids: &[String],
    sample_size: usize,
    seed: u64,
) -> Result<Vec<String>, AuditError> {
    if stratum_id.trim().is_empty() {
        return Err(AuditError::EmptyIdentity("stratum_id"));
    }
    if sample_size > population_ids.len() {
        return Err(AuditError::SampleSizeExceedsPopulation(
            stratum_id.to_string(),
        ));
    }
    for pair in population_ids.windows(2) {
        if pair[0] >= pair[1] {
            return Err(AuditError::UnsortedCandidatePopulation(
                stratum_id.to_string(),
            ));
        }
    }

    let mut indices: Vec<usize> = (0..population_ids.len()).collect();
    let mut rng = SplitMix64(seed ^ stable_hash(stratum_id.as_bytes()));

    for index in 0..sample_size {
        let remaining = population_ids.len() - index;
        let bound = u64::try_from(remaining).map_err(|_| AuditError::IntegerOverflow)?;
        let offset = usize::try_from(rng.below(bound)).map_err(|_| AuditError::IntegerOverflow)?;
        indices.swap(index, index + offset);
    }

    Ok(indices[..sample_size]
        .iter()
        .map(|index| population_ids[*index].clone())
        .collect())
}

/// Audit a complete frozen candidate universe using exact finite-population
/// hypergeometric upper bounds with a Bonferroni familywise correction.
///
/// Only ConfirmedInfeasible is a negative observation. Failed, non-converged,
/// unsupported and inconclusive outcomes are all treated as potentially feasible.
/// The accepted population is a complete census; unresolved accepted candidates
/// are included in the recall denominator but not in its lower-bound numerator.
pub fn audit_rejected_candidates(
    plan: &RejectAuditPlan,
) -> Result<RejectAuditReport, Vec<AuditError>> {
    let mut errors = Vec::new();

    if plan.seed_commitment_ref.trim().is_empty() {
        errors.push(AuditError::EmptyIdentity("seed_commitment_ref"));
    }
    if !plan.seed_committed_before_outcomes {
        errors.push(AuditError::SeedNotCommittedBeforeOutcomes);
    }
    if !plan.familywise_confidence_level.is_finite()
        || plan.familywise_confidence_level <= 0.0
        || plan.familywise_confidence_level >= 1.0
    {
        errors.push(AuditError::InvalidConfidenceLevel);
    }
    if !plan.target_feasible_recall.is_finite()
        || plan.target_feasible_recall <= 0.0
        || plan.target_feasible_recall > 1.0
    {
        errors.push(AuditError::InvalidRecallTarget);
    }
    if !is_sha256(&plan.frozen_universe.manifest_sha256) {
        errors.push(AuditError::InvalidSha256("frozen_universe.manifest_sha256"));
    }
    if !is_sha256(&plan.accepted.manifest_sha256) {
        errors.push(AuditError::InvalidSha256("accepted.manifest_sha256"));
    }
    if plan.accepted.evidence_ref.trim().is_empty() {
        errors.push(AuditError::EmptyIdentity("accepted.evidence_ref"));
    }

    let universe_set = validate_sorted_unique(
        &plan.frozen_universe.candidate_ids,
        "frozen-universe",
        &mut errors,
    );
    let accepted_set = validate_accepted_manifest(&plan.accepted, &mut errors);

    let mut stratum_ids = BTreeSet::new();
    let mut rejected_set = BTreeSet::new();

    for stratum in &plan.strata {
        if stratum.stratum_id.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("stratum_id"));
        }
        if !stratum_ids.insert(stratum.stratum_id.clone()) {
            errors.push(AuditError::DuplicateStratumId(stratum.stratum_id.clone()));
        }
        if !is_sha256(&stratum.stratum_definition_sha256) {
            errors.push(AuditError::InvalidSha256("stratum_definition_sha256"));
        }
        if !is_sha256(&stratum.population_manifest_sha256) {
            errors.push(AuditError::InvalidSha256("population_manifest_sha256"));
        }
        if stratum.selection_receipt_ref.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("selection_receipt_ref"));
        }
        if stratum.rejected_candidate_ids.is_empty() {
            errors.push(AuditError::EmptyStratumPopulation(
                stratum.stratum_id.clone(),
            ));
        }
        if stratum.sample_size > stratum.rejected_candidate_ids.len() {
            errors.push(AuditError::SampleSizeExceedsPopulation(
                stratum.stratum_id.clone(),
            ));
        }
        if stratum.samples.len() != stratum.sample_size {
            errors.push(AuditError::SampleCountMismatch {
                stratum_id: stratum.stratum_id.clone(),
                declared: stratum.sample_size,
                supplied: stratum.samples.len(),
            });
        }

        let local_population = validate_sorted_unique(
            &stratum.rejected_candidate_ids,
            &stratum.stratum_id,
            &mut errors,
        );
        for candidate_id in &local_population {
            if !universe_set.contains(candidate_id) {
                errors.push(AuditError::CandidateOutsideFrozenUniverse(
                    candidate_id.clone(),
                ));
            }
            if accepted_set.contains(candidate_id) {
                errors.push(AuditError::DuplicateCandidateId(candidate_id.clone()));
            }
            if !rejected_set.insert(candidate_id.clone()) {
                errors.push(AuditError::DuplicateCandidateId(candidate_id.clone()));
            }
        }

        for sample in &stratum.samples {
            if sample.candidate_id.trim().is_empty() {
                errors.push(AuditError::EmptyIdentity("sample.candidate_id"));
            }
            if sample.evidence_ref.trim().is_empty() {
                errors.push(AuditError::EmptyIdentity("sample.evidence_ref"));
            }
        }

        if stratum.sample_size <= stratum.rejected_candidate_ids.len()
            && stratum.rejected_candidate_ids.len() >= stratum.samples.len()
            && stratum.samples.len() == stratum.sample_size
            && !stratum.rejected_candidate_ids.is_empty()
        {
            match deterministic_audit_sample_ids(
                &stratum.stratum_id,
                &stratum.rejected_candidate_ids,
                stratum.sample_size,
                plan.sampling_seed,
            ) {
                Ok(expected) => {
                    for (index, (expected_id, actual)) in
                        expected.iter().zip(&stratum.samples).enumerate()
                    {
                        if expected_id != &actual.candidate_id {
                            errors.push(AuditError::SampleIdentityMismatch {
                                stratum_id: stratum.stratum_id.clone(),
                                sample_index: index,
                            });
                        }
                    }
                }
                Err(error) => errors.push(error),
            }
        }
    }

    // The accepted population and rejected strata must partition the frozen
    // universe exactly once. No candidate may disappear from the denominator.
    let mut partition = accepted_set.clone();
    for candidate_id in &rejected_set {
        if !partition.insert(candidate_id.clone()) {
            errors.push(AuditError::DuplicateCandidateId(candidate_id.clone()));
        }
    }
    for candidate_id in &universe_set {
        if !partition.contains(candidate_id) {
            errors.push(AuditError::CandidateMissingFromPartition(
                candidate_id.clone(),
            ));
        }
    }
    for candidate_id in &partition {
        if !universe_set.contains(candidate_id) {
            errors.push(AuditError::CandidateOutsideFrozenUniverse(
                candidate_id.clone(),
            ));
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    let accepted_feasible = plan
        .accepted
        .candidates
        .iter()
        .filter(|candidate| candidate.outcome == AuditOutcome::ConfirmedFeasible)
        .count();
    let accepted_infeasible = plan
        .accepted
        .candidates
        .iter()
        .filter(|candidate| candidate.outcome == AuditOutcome::ConfirmedInfeasible)
        .count();
    let accepted_unresolved = plan
        .accepted
        .candidates
        .iter()
        .filter(|candidate| candidate.outcome.is_unresolved())
        .count();

    let familywise_alpha = 1.0 - plan.familywise_confidence_level;
    let stratum_count = plan.strata.len();
    let per_stratum_alpha = if stratum_count == 0 {
        familywise_alpha
    } else {
        familywise_alpha / stratum_count as f64
    };

    let mut reports = Vec::with_capacity(stratum_count);
    let mut total_rejected_population = 0_u64;
    let mut total_audited_candidates = 0_u64;
    let mut missed_feasible_count_upper_bound = 0_u64;

    for stratum in &plan.strata {
        let population_size = stratum.rejected_candidate_ids.len();
        let sample_size = stratum.sample_size;
        let confirmed_feasible = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome == AuditOutcome::ConfirmedFeasible)
            .count();
        let unresolved = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome.is_unresolved())
            .count();
        let potential_feasible = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome.is_potentially_feasible())
            .count();

        let upper_count = upper_hypergeometric_population_successes(
            population_size,
            sample_size,
            potential_feasible,
            per_stratum_alpha,
        )
        .map_err(|error| vec![error])?;

        let population_u64 =
            u64::try_from(population_size).map_err(|_| vec![AuditError::IntegerOverflow])?;
        let sample_u64 =
            u64::try_from(sample_size).map_err(|_| vec![AuditError::IntegerOverflow])?;
        let upper_u64 =
            u64::try_from(upper_count).map_err(|_| vec![AuditError::IntegerOverflow])?;

        total_rejected_population = total_rejected_population
            .checked_add(population_u64)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
        total_audited_candidates = total_audited_candidates
            .checked_add(sample_u64)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
        missed_feasible_count_upper_bound = missed_feasible_count_upper_bound
            .checked_add(upper_u64)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;

        reports.push(StratumAuditReport {
            stratum_id: stratum.stratum_id.clone(),
            population_size: population_u64,
            sample_size: sample_u64,
            inclusion_probability: if population_size == 0 {
                0.0
            } else {
                sample_size as f64 / population_size as f64
            },
            confirmed_feasible_in_sample: confirmed_feasible as u64,
            unresolved_in_sample: unresolved as u64,
            potential_feasible_in_sample: potential_feasible as u64,
            upper_missed_feasible_count: upper_u64,
            upper_missed_feasible_rate: upper_count as f64 / population_size as f64,
            per_stratum_error_probability: per_stratum_alpha,
            is_full_census: sample_size == population_size,
        });
    }

    let accepted_feasible_u64 =
        u64::try_from(accepted_feasible).map_err(|_| vec![AuditError::IntegerOverflow])?;
    let accepted_infeasible_u64 =
        u64::try_from(accepted_infeasible).map_err(|_| vec![AuditError::IntegerOverflow])?;
    let accepted_unresolved_u64 =
        u64::try_from(accepted_unresolved).map_err(|_| vec![AuditError::IntegerOverflow])?;

    // Known feasible accepted candidates are the conservative lower numerator.
    // Accepted unresolved outcomes can still be feasible, so they enter only the
    // denominator. The rejected term is the simultaneous upper bound.
    let recall_denominator = accepted_feasible_u64
        .checked_add(accepted_unresolved_u64)
        .and_then(|value| value.checked_add(missed_feasible_count_upper_bound))
        .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
    let feasible_recall_lower_bound = if recall_denominator == 0 {
        None
    } else {
        Some(accepted_feasible_u64 as f64 / recall_denominator as f64)
    };
    let target_met = feasible_recall_lower_bound
        .map(|recall| recall >= plan.target_feasible_recall)
        .unwrap_or(false);
    let frozen_candidate_count = u64::try_from(plan.frozen_universe.candidate_ids.len())
        .map_err(|_| vec![AuditError::IntegerOverflow])?;
    let accepted_candidate_count = u64::try_from(plan.accepted.candidates.len())
        .map_err(|_| vec![AuditError::IntegerOverflow])?;

    Ok(RejectAuditReport {
        bound_method: BOUND_METHOD,
        sampling_algorithm: SAMPLING_ALGORITHM,
        familywise_confidence_level: plan.familywise_confidence_level,
        target_feasible_recall: plan.target_feasible_recall,
        frozen_candidate_count,
        accepted_candidate_count,
        accepted_confirmed_feasible_count: accepted_feasible_u64,
        accepted_confirmed_infeasible_count: accepted_infeasible_u64,
        accepted_unresolved_count: accepted_unresolved_u64,
        total_rejected_population,
        total_audited_candidates,
        missed_feasible_count_upper_bound,
        feasible_recall_lower_bound,
        target_met,
        strata: reports,
    })
}

fn validate_sorted_unique(
    candidate_ids: &[String],
    label: &str,
    errors: &mut Vec<AuditError>,
) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for candidate_id in candidate_ids {
        if candidate_id.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("candidate_id"));
        }
        if let Some(last) = previous {
            if candidate_id.as_str() <= last {
                errors.push(AuditError::UnsortedCandidatePopulation(label.to_string()));
                break;
            }
        }
        previous = Some(candidate_id);
        if !result.insert(candidate_id.clone()) {
            errors.push(AuditError::DuplicateCandidateId(candidate_id.clone()));
        }
    }
    result
}

fn validate_accepted_manifest(
    manifest: &AcceptedCandidateManifest,
    errors: &mut Vec<AuditError>,
) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for candidate in &manifest.candidates {
        if candidate.candidate_id.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("accepted.candidate_id"));
        }
        if candidate.evidence_ref.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("accepted.candidate.evidence_ref"));
        }
        if let Some(last) = previous {
            if candidate.candidate_id.as_str() <= last {
                errors.push(AuditError::UnsortedCandidatePopulation(
                    "accepted".to_string(),
                ));
                break;
            }
        }
        previous = Some(&candidate.candidate_id);
        if !result.insert(candidate.candidate_id.clone()) {
            errors.push(AuditError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
    }
    result
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn upper_hypergeometric_population_successes(
    population_size: usize,
    sample_size: usize,
    observed_successes: usize,
    alpha: f64,
) -> Result<usize, AuditError> {
    if sample_size > population_size
        || observed_successes > sample_size
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha >= 1.0
    {
        return Err(AuditError::InvalidHypergeometricInputs);
    }
    if sample_size == 0 {
        return Ok(population_size);
    }
    if sample_size == population_size {
        return Ok(observed_successes);
    }

    let mut log_factorials = vec![0.0; population_size + 1];
    for value in 1..=population_size {
        log_factorials[value] = log_factorials[value - 1] + (value as f64).ln();
    }
    let log_alpha = alpha.ln();

    // For fixed N, n and x, P[X <= x | K] is non-increasing in K.
    // Invert the exact lower-tail hypergeometric test to obtain the largest
    // population success count that is not rejected at level alpha.
    let mut lower = observed_successes;
    let mut upper = population_size;
    while lower < upper {
        let middle = lower + (upper - lower).div_ceil(2);
        let log_cdf = hypergeometric_log_cdf_at_most(
            population_size,
            middle,
            sample_size,
            observed_successes,
            &log_factorials,
        );
        if log_cdf > log_alpha {
            lower = middle;
        } else {
            upper = middle - 1;
        }
    }
    Ok(lower)
}

fn hypergeometric_log_cdf_at_most(
    population_size: usize,
    population_successes: usize,
    sample_size: usize,
    observed_successes: usize,
    log_factorials: &[f64],
) -> f64 {
    let minimum = sample_size.saturating_sub(population_size - population_successes);
    let maximum = sample_size.min(population_successes);
    if observed_successes < minimum {
        return f64::NEG_INFINITY;
    }
    let last = observed_successes.min(maximum);
    let log_denominator = log_choose(population_size, sample_size, log_factorials);
    let mut total = f64::NEG_INFINITY;

    for successes in minimum..=last {
        let failures = sample_size - successes;
        let log_probability = log_choose(population_successes, successes, log_factorials)
            + log_choose(
                population_size - population_successes,
                failures,
                log_factorials,
            )
            - log_denominator;
        total = log_add_exp(total, log_probability);
    }
    total.min(0.0)
}

fn log_choose(n: usize, k: usize, log_factorials: &[f64]) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    log_factorials[n] - log_factorials[k] - log_factorials[n - k]
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}

fn hypergeometric_pmf(
    population_size: usize,
    population_successes: usize,
    sample_size: usize,
    observed_successes: usize,
) -> f64 {
    let mut log_factorials = vec![0.0; population_size + 1];
    for value in 1..=population_size {
        log_factorials[value] = log_factorials[value - 1] + (value as f64).ln();
    }
    let minimum = sample_size.saturating_sub(population_size - population_successes);
    let maximum = sample_size.min(population_successes);
    if observed_successes < minimum || observed_successes > maximum {
        return 0.0;
    }
    (log_choose(population_successes, observed_successes, &log_factorials)
        + log_choose(
            population_size - population_successes,
            sample_size - observed_successes,
            &log_factorials,
        )
        - log_choose(population_size, sample_size, &log_factorials))
    .exp()
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }

    fn below(&mut self, bound: u64) -> u64 {
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next();
            if value >= threshold {
                return value % bound;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn ids(prefix: &str, count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("{prefix}-{index:04}"))
            .collect()
    }

    fn samples(
        stratum_id: &str,
        population: &[String],
        sample_size: usize,
        seed: u64,
        outcomes: &[AuditOutcome],
    ) -> Vec<AuditSample> {
        let selected =
            deterministic_audit_sample_ids(stratum_id, population, sample_size, seed).unwrap();
        assert_eq!(selected.len(), outcomes.len());
        selected
            .into_iter()
            .zip(outcomes)
            .enumerate()
            .map(|(index, (candidate_id, outcome))| AuditSample {
                candidate_id,
                outcome: *outcome,
                evidence_ref: format!("solver-run-{stratum_id}-{index}"),
            })
            .collect()
    }

    fn stratum(
        stratum_id: &str,
        population: Vec<String>,
        sample_size: usize,
        selected: Vec<AuditSample>,
    ) -> RejectStratum {
        RejectStratum {
            stratum_id: stratum_id.to_string(),
            stratum_definition_sha256: digest('a'),
            population_manifest_sha256: digest('b'),
            selection_receipt_ref: format!("selection-receipt-{stratum_id}"),
            rejected_candidate_ids: population,
            sample_size,
            samples: selected,
        }
    }

    fn plan(accepted: Vec<AcceptedCandidate>, strata: Vec<RejectStratum>) -> RejectAuditPlan {
        let mut universe = accepted
            .iter()
            .map(|candidate| candidate.candidate_id.clone())
            .collect::<Vec<_>>();
        universe.extend(
            strata
                .iter()
                .flat_map(|stratum| stratum.rejected_candidate_ids.iter().cloned()),
        );
        universe.sort();

        RejectAuditPlan {
            frozen_universe: FrozenCandidateUniverse {
                candidate_ids: universe,
                manifest_sha256: digest('c'),
            },
            sampling_seed: 0x5eed_f00d_cafe_babe,
            seed_commitment_ref: "precommitted-seed-record".to_string(),
            seed_committed_before_outcomes: true,
            familywise_confidence_level: 0.95,
            target_feasible_recall: 0.99,
            accepted: AcceptedCandidateManifest {
                candidates: accepted,
                manifest_sha256: digest('d'),
                evidence_ref: "accepted-population-evidence-manifest".to_string(),
            },
            strata,
        }
    }

    fn accepted_candidate(index: usize, outcome: AuditOutcome) -> AcceptedCandidate {
        AcceptedCandidate {
            candidate_id: format!("accepted-{index:04}"),
            outcome,
            evidence_ref: format!("accepted-solver-run-{index:04}"),
        }
    }

    #[test]
    fn full_census_produces_exact_missed_candidate_bound() {
        let population = ids("core", 4);
        let outcomes = [
            AuditOutcome::ConfirmedFeasible,
            AuditOutcome::ConfirmedInfeasible,
            AuditOutcome::Unsupported,
            AuditOutcome::ConfirmedInfeasible,
        ];
        let selected = samples(
            "core",
            &population,
            population.len(),
            0x5eed_f00d_cafe_babe,
            &outcomes,
        );
        let accepted = (0..99)
            .map(|index| accepted_candidate(index, AuditOutcome::ConfirmedFeasible))
            .collect();
        let report = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("core", population, 4, selected)],
        ))
        .unwrap();

        assert_eq!(report.total_rejected_population, 4);
        assert_eq!(report.total_audited_candidates, 4);
        assert_eq!(report.missed_feasible_count_upper_bound, 2);
        assert_eq!(report.strata[0].unresolved_in_sample, 1);
        assert_eq!(report.feasible_recall_lower_bound, Some(99.0 / 101.0));
    }

    #[test]
    fn accepted_unresolved_outcomes_are_included_in_the_recall_denominator() {
        let population = ids("reject", 2);
        let selected = samples(
            "reject",
            &population,
            2,
            0x5eed_f00d_cafe_babe,
            &[
                AuditOutcome::ConfirmedFeasible,
                AuditOutcome::ConfirmedInfeasible,
            ],
        );
        let accepted = vec![
            accepted_candidate(0, AuditOutcome::ConfirmedFeasible),
            accepted_candidate(1, AuditOutcome::Failed),
            accepted_candidate(2, AuditOutcome::ConfirmedInfeasible),
        ];
        let report = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("reject", population, 2, selected)],
        ))
        .unwrap();

        assert_eq!(report.accepted_confirmed_feasible_count, 1);
        assert_eq!(report.accepted_unresolved_count, 1);
        assert_eq!(report.missed_feasible_count_upper_bound, 1);
        assert_eq!(report.feasible_recall_lower_bound, Some(1.0 / 3.0));
    }

    #[test]
    fn zero_audited_rejects_do_not_produce_a_zero_upper_bound() {
        let population = ids("unseen", 100);
        let accepted = (0..100)
            .map(|index| accepted_candidate(index, AuditOutcome::ConfirmedFeasible))
            .collect();
        let report = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("unseen", population, 0, vec![])],
        ))
        .unwrap();

        assert_eq!(report.total_audited_candidates, 0);
        assert_eq!(report.missed_feasible_count_upper_bound, 100);
        assert!(!report.target_met);
    }

    #[test]
    fn all_unresolved_sample_outcomes_are_conservatively_positive() {
        let population = ids("edge", 6);
        let outcomes = [
            AuditOutcome::ConfirmedInfeasible,
            AuditOutcome::Failed,
            AuditOutcome::NonConverged,
            AuditOutcome::Unsupported,
            AuditOutcome::Inconclusive,
            AuditOutcome::ConfirmedInfeasible,
        ];
        let selected = samples("edge", &population, 6, 0x5eed_f00d_cafe_babe, &outcomes);
        let accepted = (0..100)
            .map(|index| accepted_candidate(index, AuditOutcome::ConfirmedFeasible))
            .collect();
        let report = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("edge", population, 6, selected)],
        ))
        .unwrap();

        assert_eq!(report.strata[0].potential_feasible_in_sample, 4);
        assert_eq!(report.missed_feasible_count_upper_bound, 4);
    }

    #[test]
    fn familywise_correction_becomes_stricter_as_strata_are_added() {
        let population_a = ids("a", 200);
        let population_b = ids("b", 200);
        let outcomes = vec![AuditOutcome::ConfirmedInfeasible; 20];
        let samples_a = samples("a", &population_a, 20, 0x5eed_f00d_cafe_babe, &outcomes);
        let samples_b = samples("b", &population_b, 20, 0x5eed_f00d_cafe_babe, &outcomes);
        let accepted = (0..1_000)
            .map(|index| accepted_candidate(index, AuditOutcome::ConfirmedFeasible))
            .collect();

        let one = audit_rejected_candidates(&plan(
            accepted.clone(),
            vec![stratum("a", population_a.clone(), 20, samples_a.clone())],
        ))
        .unwrap();
        let two = audit_rejected_candidates(&plan(
            accepted,
            vec![
                stratum("a", population_a, 20, samples_a),
                stratum("b", population_b, 20, samples_b),
            ],
        ))
        .unwrap();

        assert!(
            two.strata[0].upper_missed_feasible_rate > one.strata[0].upper_missed_feasible_rate
        );
        assert_eq!(
            two.strata[0].per_stratum_error_probability,
            (1.0 - 0.95) / 2.0
        );
    }

    #[test]
    fn deterministic_sample_is_reproducible_and_seed_sensitive() {
        let population = ids("candidate", 100);
        let first = deterministic_audit_sample_ids("stratum", &population, 20, 0x1111).unwrap();
        let replay = deterministic_audit_sample_ids("stratum", &population, 20, 0x1111).unwrap();
        let other = deterministic_audit_sample_ids("stratum", &population, 20, 0x2222).unwrap();

        assert_eq!(first, replay);
        assert_ne!(first, other);
    }

    #[test]
    fn sample_identity_tampering_is_rejected() {
        let population = ids("core", 10);
        let mut selected = samples(
            "core",
            &population,
            3,
            0x5eed_f00d_cafe_babe,
            &[
                AuditOutcome::ConfirmedInfeasible,
                AuditOutcome::ConfirmedInfeasible,
                AuditOutcome::ConfirmedInfeasible,
            ],
        );
        selected[0].candidate_id = "core-9999".to_string();
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let errors = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("core", population, 3, selected)],
        ))
        .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::SampleIdentityMismatch {
                stratum_id,
                sample_index: 0
            } if stratum_id == "core"
        )));
    }

    #[test]
    fn omitted_sample_result_is_rejected_against_frozen_sample_size() {
        let population = ids("core", 10);
        let selected = samples(
            "core",
            &population,
            3,
            0x5eed_f00d_cafe_babe,
            &[AuditOutcome::ConfirmedInfeasible; 3],
        );
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let errors = audit_rejected_candidates(&plan(
            accepted,
            vec![stratum("core", population, 4, selected)],
        ))
        .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::SampleCountMismatch {
                stratum_id,
                declared: 4,
                supplied: 3
            } if stratum_id == "core"
        )));
    }

    #[test]
    fn duplicated_candidate_across_strata_is_rejected() {
        let population_a = vec!["candidate-0001".to_string(), "candidate-0002".to_string()];
        let population_b = vec!["candidate-0001".to_string(), "candidate-0003".to_string()];
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let errors = audit_rejected_candidates(&plan(
            accepted,
            vec![
                stratum("a", population_a, 0, vec![]),
                stratum("b", population_b, 0, vec![]),
            ],
        ))
        .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::DuplicateCandidateId(id) if id == "candidate-0001"
        )));
    }

    #[test]
    fn a_candidate_missing_from_the_frozen_partition_is_rejected() {
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let mut plan = plan(accepted, vec![]);
        plan.frozen_universe
            .candidate_ids
            .push("ghost-candidate".to_string());
        plan.frozen_universe.candidate_ids.sort();

        let errors = audit_rejected_candidates(&plan).unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::CandidateMissingFromPartition(id) if id == "ghost-candidate"
        )));
    }

    #[test]
    fn accepted_and_rejected_candidate_overlap_is_rejected() {
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let rejected = vec!["accepted-0000".to_string(), "rejected-0001".to_string()];
        let errors =
            audit_rejected_candidates(&plan(accepted, vec![stratum("edge", rejected, 0, vec![])]))
                .unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::DuplicateCandidateId(id) if id == "accepted-0000"
        )));
    }

    #[test]
    fn seed_commitment_must_precede_audit_outcomes() {
        let accepted = vec![accepted_candidate(0, AuditOutcome::ConfirmedFeasible)];
        let population = ids("core", 4);
        let mut invalid = plan(accepted, vec![stratum("core", population, 0, vec![])]);
        invalid.seed_committed_before_outcomes = false;

        let errors = audit_rejected_candidates(&invalid).unwrap_err();
        assert!(errors.contains(&AuditError::SeedNotCommittedBeforeOutcomes));
    }

    #[test]
    fn exact_hypergeometric_bound_collapses_under_full_census() {
        for population_size in 1..=30 {
            for observed in 0..=population_size {
                let upper = upper_hypergeometric_population_successes(
                    population_size,
                    population_size,
                    observed,
                    0.05,
                )
                .unwrap();
                assert_eq!(upper, observed);
            }
        }
    }

    #[test]
    fn exact_hypergeometric_upper_bound_has_at_least_nominal_coverage() {
        let alpha = 0.05;
        for population_size in 1..=18 {
            for sample_size in 0..=population_size {
                for true_successes in 0..=population_size {
                    let minimum = sample_size.saturating_sub(population_size - true_successes);
                    let maximum = sample_size.min(true_successes);
                    let mut noncoverage_probability = 0.0;
                    for observed in minimum..=maximum {
                        let upper = upper_hypergeometric_population_successes(
                            population_size,
                            sample_size,
                            observed,
                            alpha,
                        )
                        .unwrap();
                        if upper < true_successes {
                            noncoverage_probability += hypergeometric_pmf(
                                population_size,
                                true_successes,
                                sample_size,
                                observed,
                            );
                        }
                    }
                    assert!(
                        noncoverage_probability <= alpha + 1.0e-10,
                        "coverage failure N={population_size}, n={sample_size}, K={true_successes}, noncoverage={noncoverage_probability}"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_hypergeometric_bound_is_monotone_in_observed_positives() {
        for observed in 0..10 {
            let current =
                upper_hypergeometric_population_successes(100, 20, observed, 0.05).unwrap();
            let next =
                upper_hypergeometric_population_successes(100, 20, observed + 1, 0.05).unwrap();
            assert!(next >= current);
        }
    }
}

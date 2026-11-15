use std::collections::BTreeSet;

pub const SAMPLING_ALGORITHM: &str = "splitmix64-partial-fisher-yates-v1";
pub const BOUND_METHOD: &str = "one-sided-hoeffding-bonferroni-v1";

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
    fn is_potential_false_negative(self) -> bool {
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
pub struct AcceptedFeasibleCandidate {
    pub candidate_id: String,
    pub evidence_ref: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AcceptedFeasibleManifest {
    /// Explicit, lexicographically ordered reference-confirmed feasible identities.
    pub candidates: Vec<AcceptedFeasibleCandidate>,
    /// Declared identity of the manifest listing those candidates and evidence.
    pub manifest_sha256: String,
    /// Reference to the independently checkable high-fidelity evidence manifest.
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
    /// Sample outcomes in deterministic draw order.
    pub samples: Vec<AuditSample>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RejectAuditPlan {
    /// Seed whose commitment record predates observation of audit outcomes.
    pub sampling_seed: u64,
    pub seed_commitment_ref: String,
    pub seed_committed_before_outcomes: bool,
    pub familywise_confidence_level: f64,
    pub target_feasible_recall: f64,
    pub accepted_feasible: AcceptedFeasibleManifest,
    pub strata: Vec<RejectStratum>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AuditError {
    EmptyIdentity(&'static str),
    InvalidSha256(&'static str),
    InvalidConfidenceLevel,
    InvalidRecallTarget,
    SeedNotCommittedBeforeOutcomes,
    NoRejectedStrata,
    EmptyStratumPopulation(String),
    DuplicateStratumId(String),
    DuplicateCandidateId(String),
    UnsortedCandidatePopulation(String),
    SampleSizeExceedsPopulation(String),
    SampleIdentityMismatch {
        stratum_id: String,
        sample_index: usize,
    },
    IntegerOverflow,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StratumAuditReport {
    pub stratum_id: String,
    pub population_size: u64,
    pub sample_size: u64,
    pub confirmed_feasible_in_sample: u64,
    pub unresolved_in_sample: u64,
    /// Feasible plus unresolved observations; unresolved is treated as positive.
    pub potential_false_negatives_in_sample: u64,
    pub upper_missed_feasible_count: u64,
    pub upper_missed_feasible_rate: f64,
    pub per_stratum_error_probability: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RejectAuditReport {
    pub bound_method: &'static str,
    pub sampling_algorithm: &'static str,
    pub familywise_confidence_level: f64,
    pub target_feasible_recall: f64,
    pub total_rejected_population: u64,
    pub total_audited_candidates: u64,
    pub accepted_feasible_count_lower_bound: u64,
    pub missed_feasible_count_upper_bound: u64,
    /// Conservative lower bound; absent when no feasible population can be bounded.
    pub feasible_recall_lower_bound: Option<f64>,
    pub target_met: bool,
    pub strata: Vec<StratumAuditReport>,
}

/// Reproduce the audit draw for a stratum.
///
/// The population IDs must be frozen and sorted lexicographically. The returned
/// identities are in draw order, not sorted order.
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
    if !is_sha256(&plan.accepted_feasible.manifest_sha256) {
        errors.push(AuditError::InvalidSha256(
            "accepted_feasible.manifest_sha256",
        ));
    }
    if plan.accepted_feasible.evidence_ref.trim().is_empty() {
        errors.push(AuditError::EmptyIdentity("accepted_feasible.evidence_ref"));
    }
    if plan.strata.is_empty() {
        errors.push(AuditError::NoRejectedStrata);
    }

    let mut stratum_ids = BTreeSet::new();
    let mut population_ids = BTreeSet::new();

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
        if stratum.samples.len() > stratum.rejected_candidate_ids.len() {
            errors.push(AuditError::SampleSizeExceedsPopulation(
                stratum.stratum_id.clone(),
            ));
        }

        let mut previous: Option<&str> = None;
        for candidate_id in &stratum.rejected_candidate_ids {
            if candidate_id.trim().is_empty() {
                errors.push(AuditError::EmptyIdentity("rejected_candidate_id"));
            }
            if let Some(last) = previous {
                if candidate_id.as_str() <= last {
                    errors.push(AuditError::UnsortedCandidatePopulation(
                        stratum.stratum_id.clone(),
                    ));
                    break;
                }
            }
            previous = Some(candidate_id);
            if !population_ids.insert(candidate_id.clone()) {
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

        if stratum.samples.len() <= stratum.rejected_candidate_ids.len()
            && !stratum.rejected_candidate_ids.is_empty()
        {
            match deterministic_audit_sample_ids(
                &stratum.stratum_id,
                &stratum.rejected_candidate_ids,
                stratum.samples.len(),
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

    // Explicit accepted identities prevent double-counting a candidate that also
    // appears in the rejected population. The digest's source bytes remain the
    // caller's independent verification responsibility.
    let mut accepted_ids = BTreeSet::new();
    let mut previous_accepted: Option<&str> = None;
    for candidate in &plan.accepted_feasible.candidates {
        if candidate.candidate_id.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity("accepted_feasible.candidate_id"));
        }
        if candidate.evidence_ref.trim().is_empty() {
            errors.push(AuditError::EmptyIdentity(
                "accepted_feasible.candidate.evidence_ref",
            ));
        }
        if let Some(previous) = previous_accepted {
            if candidate.candidate_id.as_str() <= previous {
                errors.push(AuditError::UnsortedCandidatePopulation(
                    "accepted-feasible".to_string(),
                ));
                break;
            }
        }
        previous_accepted = Some(&candidate.candidate_id);
        if !accepted_ids.insert(candidate.candidate_id.clone())
            || population_ids.contains(&candidate.candidate_id)
        {
            errors.push(AuditError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
    }

    let accepted_count = u64::try_from(plan.accepted_feasible.candidates.len())
        .map_err(|_| vec![AuditError::IntegerOverflow])?;
    if !errors.is_empty() {
        return Err(errors);
    }

    let stratum_count = plan.strata.len() as f64;
    let familywise_alpha = 1.0 - plan.familywise_confidence_level;
    let per_stratum_alpha = familywise_alpha / stratum_count;

    let mut reports = Vec::with_capacity(plan.strata.len());
    let mut total_rejected_population = 0_u64;
    let mut total_audited_candidates = 0_u64;
    let mut missed_feasible_count_upper_bound = 0_u64;

    for stratum in &plan.strata {
        let population_size = u64::try_from(stratum.rejected_candidate_ids.len())
            .map_err(|_| vec![AuditError::IntegerOverflow])?;
        let sample_size =
            u64::try_from(stratum.samples.len()).map_err(|_| vec![AuditError::IntegerOverflow])?;
        let confirmed_feasible = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome == AuditOutcome::ConfirmedFeasible)
            .count() as u64;
        let unresolved = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome.is_unresolved())
            .count() as u64;
        let potential_false_negatives = stratum
            .samples
            .iter()
            .filter(|sample| sample.outcome.is_potential_false_negative())
            .count() as u64;

        let upper_rate = if sample_size == 0 {
            1.0
        } else if sample_size == population_size {
            potential_false_negatives as f64 / sample_size as f64
        } else {
            let observed_rate = potential_false_negatives as f64 / sample_size as f64;
            let epsilon = (-per_stratum_alpha.ln() / (2.0 * sample_size as f64)).sqrt();
            (observed_rate + epsilon).min(1.0)
        };

        let upper_count = if sample_size == population_size {
            potential_false_negatives
        } else {
            ((population_size as f64 * upper_rate).ceil() as u64).min(population_size)
        };

        total_rejected_population = total_rejected_population
            .checked_add(population_size)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
        total_audited_candidates = total_audited_candidates
            .checked_add(sample_size)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
        missed_feasible_count_upper_bound = missed_feasible_count_upper_bound
            .checked_add(upper_count)
            .ok_or_else(|| vec![AuditError::IntegerOverflow])?;

        reports.push(StratumAuditReport {
            stratum_id: stratum.stratum_id.clone(),
            population_size,
            sample_size,
            confirmed_feasible_in_sample: confirmed_feasible,
            unresolved_in_sample: unresolved,
            potential_false_negatives_in_sample: potential_false_negatives,
            upper_missed_feasible_count: upper_count,
            upper_missed_feasible_rate: upper_rate,
            per_stratum_error_probability: per_stratum_alpha,
        });
    }

    let recall_denominator = accepted_count
        .checked_add(missed_feasible_count_upper_bound)
        .ok_or_else(|| vec![AuditError::IntegerOverflow])?;
    let feasible_recall_lower_bound = if recall_denominator == 0 {
        None
    } else {
        Some(accepted_count as f64 / recall_denominator as f64)
    };
    let target_met = feasible_recall_lower_bound
        .map(|recall| recall >= plan.target_feasible_recall)
        .unwrap_or(false);

    Ok(RejectAuditReport {
        bound_method: BOUND_METHOD,
        sampling_algorithm: SAMPLING_ALGORITHM,
        familywise_confidence_level: plan.familywise_confidence_level,
        target_feasible_recall: plan.target_feasible_recall,
        total_rejected_population,
        total_audited_candidates,
        accepted_feasible_count_lower_bound: accepted_count,
        missed_feasible_count_upper_bound,
        feasible_recall_lower_bound,
        target_met,
        strata: reports,
    })
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
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

    fn stratum_population(id: &str, count: usize) -> Vec<String> {
        (0..count)
            .map(|index| format!("{id}-candidate-{index:04}"))
            .collect()
    }

    fn sample_outcomes(
        stratum_id: &str,
        population: &[String],
        sample_size: usize,
        seed: u64,
        outcomes: &[AuditOutcome],
    ) -> Vec<AuditSample> {
        let ids =
            deterministic_audit_sample_ids(stratum_id, population, sample_size, seed).unwrap();
        assert_eq!(ids.len(), outcomes.len());
        ids.into_iter()
            .zip(outcomes)
            .enumerate()
            .map(|(index, (candidate_id, outcome))| AuditSample {
                candidate_id,
                outcome: *outcome,
                evidence_ref: format!("solver-run-{stratum_id}-{index}"),
            })
            .collect()
    }

    fn stratum(id: &str, population: Vec<String>, samples: Vec<AuditSample>) -> RejectStratum {
        RejectStratum {
            stratum_id: id.to_string(),
            stratum_definition_sha256: digest('a'),
            population_manifest_sha256: digest('b'),
            selection_receipt_ref: format!("selection-receipt-{id}"),
            rejected_candidate_ids: population,
            samples,
        }
    }

    fn plan(strata: Vec<RejectStratum>, accepted: u64) -> RejectAuditPlan {
        RejectAuditPlan {
            sampling_seed: 0x5eed_f00d_cafe_babe,
            seed_commitment_ref: "precommitted-seed-record".to_string(),
            seed_committed_before_outcomes: true,
            familywise_confidence_level: 0.95,
            target_feasible_recall: 0.99,
            accepted_feasible: AcceptedFeasibleManifest {
                candidates: (0..accepted)
                    .map(|index| AcceptedFeasibleCandidate {
                        candidate_id: format!("accepted-feasible-{index:08}"),
                        evidence_ref: format!("accepted-solver-run-{index:08}"),
                    })
                    .collect(),
                manifest_sha256: digest('c'),
                evidence_ref: "accepted-feasible-evidence-manifest".to_string(),
            },
            strata,
        }
    }

    #[test]
    fn full_census_produces_exact_missed_candidate_bound() {
        let population = stratum_population("core", 4);
        let outcomes = [
            AuditOutcome::ConfirmedFeasible,
            AuditOutcome::ConfirmedInfeasible,
            AuditOutcome::Unsupported,
            AuditOutcome::ConfirmedInfeasible,
        ];
        let samples = sample_outcomes(
            "core",
            &population,
            population.len(),
            0x5eed_f00d_cafe_babe,
            &outcomes,
        );
        let report =
            audit_rejected_candidates(&plan(vec![stratum("core", population, samples)], 99))
                .unwrap();

        assert_eq!(report.total_rejected_population, 4);
        assert_eq!(report.total_audited_candidates, 4);
        assert_eq!(report.missed_feasible_count_upper_bound, 2);
        assert_eq!(report.strata[0].unresolved_in_sample, 1);
        assert_eq!(report.feasible_recall_lower_bound, Some(99.0 / 101.0));
    }

    #[test]
    fn unresolved_solver_results_are_counted_as_potential_false_negatives() {
        let population = stratum_population("edge", 6);
        let outcomes = [
            AuditOutcome::ConfirmedInfeasible,
            AuditOutcome::Failed,
            AuditOutcome::NonConverged,
            AuditOutcome::Unsupported,
            AuditOutcome::Inconclusive,
            AuditOutcome::ConfirmedInfeasible,
        ];
        let samples = sample_outcomes(
            "edge",
            &population,
            population.len(),
            0x5eed_f00d_cafe_babe,
            &outcomes,
        );
        let report =
            audit_rejected_candidates(&plan(vec![stratum("edge", population, samples)], 100))
                .unwrap();

        assert_eq!(report.strata[0].potential_false_negatives_in_sample, 4);
        assert_eq!(report.missed_feasible_count_upper_bound, 4);
    }

    #[test]
    fn empty_audit_cannot_certify_a_high_recall_target() {
        let population = stratum_population("unseen", 100);
        let report =
            audit_rejected_candidates(&plan(vec![stratum("unseen", population, vec![])], 100))
                .unwrap();

        assert_eq!(report.total_audited_candidates, 0);
        assert_eq!(report.missed_feasible_count_upper_bound, 100);
        assert_eq!(report.feasible_recall_lower_bound, Some(0.5));
        assert!(!report.target_met);
    }

    #[test]
    fn familywise_correction_becomes_stricter_as_strata_are_added() {
        let population_a = stratum_population("a", 200);
        let population_b = stratum_population("b", 200);
        let outcomes = vec![AuditOutcome::ConfirmedInfeasible; 20];
        let one_samples = sample_outcomes("a", &population_a, 20, 0x5eed_f00d_cafe_babe, &outcomes);
        let two_samples = sample_outcomes("b", &population_b, 20, 0x5eed_f00d_cafe_babe, &outcomes);

        let one =
            audit_rejected_candidates(&plan(vec![stratum("a", population_a, one_samples)], 1_000))
                .unwrap();
        let two = audit_rejected_candidates(&plan(
            vec![
                stratum(
                    "a",
                    population_a,
                    sample_outcomes(
                        "a",
                        &stratum_population("a", 200),
                        20,
                        0x5eed_f00d_cafe_babe,
                        &outcomes,
                    ),
                ),
                stratum("b", population_b, two_samples),
            ],
            1_000,
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
    fn sample_identity_tampering_is_rejected() {
        let population = stratum_population("core", 10);
        let mut samples = sample_outcomes(
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
        samples[0].candidate_id = "core-candidate-9999".to_string();
        let errors =
            audit_rejected_candidates(&plan(vec![stratum("core", population, samples)], 100))
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
    fn candidate_identity_cannot_appear_in_two_reject_strata() {
        let population_a = vec!["candidate-0001".to_string(), "candidate-0002".to_string()];
        let population_b = vec!["candidate-0001".to_string(), "candidate-0003".to_string()];
        let plan = plan(
            vec![
                stratum("a", population_a, vec![]),
                stratum("b", population_b, vec![]),
            ],
            100,
        );
        let errors = audit_rejected_candidates(&plan).unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::DuplicateCandidateId(id) if id == "candidate-0001"
        )));
    }

    #[test]
    fn seed_commitment_must_precede_audit_outcomes() {
        let population = stratum_population("core", 4);
        let mut invalid = plan(vec![stratum("core", population, vec![])], 100);
        invalid.seed_committed_before_outcomes = false;

        let errors = audit_rejected_candidates(&invalid).unwrap_err();
        assert!(errors.contains(&AuditError::SeedNotCommittedBeforeOutcomes));
    }

    #[test]
    fn sample_evidence_reference_is_mandatory_for_every_outcome() {
        let population = stratum_population("core", 4);
        let mut samples = sample_outcomes(
            "core",
            &population,
            1,
            0x5eed_f00d_cafe_babe,
            &[AuditOutcome::Failed],
        );
        samples[0].evidence_ref.clear();
        let errors =
            audit_rejected_candidates(&plan(vec![stratum("core", population, samples)], 100))
                .unwrap_err();

        assert!(errors.contains(&AuditError::EmptyIdentity("sample.evidence_ref")));
    }

    #[test]
    fn malformed_strata_and_manifest_digests_fail_closed() {
        let population = stratum_population("core", 4);
        let mut bad = stratum("core", population, vec![]);
        bad.population_manifest_sha256 = "not-a-digest".to_string();
        bad.rejected_candidate_ids.reverse();
        let errors = audit_rejected_candidates(&plan(vec![bad], 100)).unwrap_err();

        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::InvalidSha256("population_manifest_sha256")
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            AuditError::UnsortedCandidatePopulation(id) if id == "core"
        )));
    }
}

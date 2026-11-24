//! Posterior-draw aggregation for individual counterfactual effects.
//!
//! This module extends #13520's finite per-pair arithmetic without creating a
//! second branch verifier, effect scorer, calibration engine, or statistical
//! inference implementation.
//!
//! For one restored individual/action pair, every posterior draw contains two
//! sibling model predictions. Both sibling predictions must carry the exact
//! same draw ID and digest. The independently measured branch outcomes and
//! their source run IDs must remain identical across posterior draws; only the
//! candidate's draw-conditioned latent/rollout predictions may vary.
//!
//! A posterior draw is not an independent individual. Results are returned
//! per individual, and callers must retain that cluster identity downstream.
//!
//! Quantiles are weighted empirical inverse-CDF summaries of the finite
//! supplied draw set. They are not calibrated credible intervals, identified
//! sets, or evidence that the supplied posterior is well specified.

use super::individual_effect_retention::{
    evaluate_individual_effect_retention, BranchDisposition, IndividualEffectBranch,
    IndividualEffectComparison, IndividualEffectContext, IndividualEffectError,
    IndividualEffectPair,
};
use std::collections::BTreeSet;
use std::fmt;

pub const POSTERIOR_WEIGHT_TOTAL_PPM: u32 = 1_000_000;
pub const MAX_POSTERIOR_DRAWS_PER_INDIVIDUAL: usize = 4_096;
pub const MAX_POSTERIOR_INDIVIDUALS: usize = 4_096;
pub const MAX_TOTAL_POSTERIOR_DRAWS: usize = 65_536;

const Q05_PPM: u32 = 50_000;
const Q50_PPM: u32 = 500_000;
const Q95_PPM: u32 = 950_000;

/// A branch prediction is explicitly bound to one posterior draw.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorDrawBranch {
    pub posterior_draw_id: u64,
    pub posterior_draw_sha256: [u8; 32],
    /// Existing #13520 branch record. Its realized outcome/run identity is
    /// repeated across posterior draws; its candidate predictions are draw-local.
    pub branch: IndividualEffectBranch,
}

/// Both sibling action predictions under one posterior hypothesis about one
/// individual. The posterior draw is an exogenous-hypothesis identity, not a
/// new independent individual.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorDrawPair {
    pub posterior_draw_id: u64,
    pub posterior_draw_sha256: [u8; 32],
    /// Integer PPM weight. Weights across this individual's draw cohort must
    /// sum exactly to POSTERIOR_WEIGHT_TOTAL_PPM.
    pub weight_ppm: u32,
    pub context: IndividualEffectContext,
    pub baseline: PosteriorDrawBranch,
    pub alternative: PosteriorDrawBranch,
}

/// One individual and one frozen intervention pair, evaluated over posterior
/// draws from q(u_i | factual history).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorIndividualCohort {
    pub draws: Vec<PosteriorDrawPair>,
}

/// One draw's exact paired prediction/effect record. The realized effect is
/// identical across all draws in a cohort and is carried once in the summary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorEffectSampleComparison {
    pub posterior_draw_id: u64,
    pub posterior_draw_sha256: [u8; 32],
    pub weight_ppm: u32,
    pub latent_predicted_effect_units: i128,
    pub rollout_predicted_effect_units: i128,
    pub latent_absolute_error_units: u128,
    pub rollout_absolute_error_units: u128,
    pub latent_sign_matches: bool,
    pub rollout_sign_matches: bool,
}

/// Weighted empirical distribution summary. The weighted mean is represented
/// exactly as weighted_effect_sum_units_ppm / POSTERIOR_WEIGHT_TOTAL_PPM.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorEffectDistribution {
    pub sample_count: u64,
    pub weight_total_ppm: u32,
    pub min_effect_units: i128,
    /// Lower weighted empirical inverse-CDF quantile.
    pub q05_effect_units: i128,
    pub median_effect_units: i128,
    pub q95_effect_units: i128,
    pub max_effect_units: i128,
    pub weighted_effect_sum_units_ppm: i128,
    pub weight_mass_negative_ppm: u32,
    pub weight_mass_zero_ppm: u32,
    pub weight_mass_positive_ppm: u32,
    /// Weighted absolute-error numerator; divide by 1,000,000 for the exact
    /// posterior-weighted mean absolute error in the declared integer units.
    pub weighted_absolute_error_sum_units_ppm: u128,
    /// Posterior weight mass whose predicted effect sign matches the realized
    /// paired effect sign. This is descriptive, not a calibrated probability.
    pub sign_match_weight_ppm: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorIndividualEffectSummary {
    pub individual_id_sha256: [u8; 32],
    pub pair_id: u64,
    pub realized_effect_units: i128,
    pub baseline_applied_at_us: u64,
    pub alternative_applied_at_us: u64,
    pub application_time_delta_us: i128,
    pub latent_effects: PosteriorEffectDistribution,
    pub rollout_effects: PosteriorEffectDistribution,
    /// Canonically ordered by posterior_draw_id for deterministic downstream
    /// serialization/reconstruction.
    pub samples: Vec<PosteriorEffectSampleComparison>,
}

/// Cohort-level manifest. Individual summaries are deliberately not pooled
/// into one distribution: individual heterogeneity and cluster identity remain
/// visible, and posterior draws are never counted as independent subjects.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PosteriorEffectCohortCertificate {
    pub task_sha256: [u8; 32],
    pub intervention_pair_sha256: [u8; 32],
    pub loss_scale_sha256: [u8; 32],
    pub baseline_action_id: String,
    pub alternative_action_id: String,
    pub individual_count: u64,
    pub total_posterior_draw_count: u64,
    pub individuals: Vec<PosteriorIndividualEffectSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PosteriorCohortError {
    EmptyIndividualCohort,
    EmptyDrawSet,
    TooManyDraws { requested: usize, maximum: usize },
    TooManyIndividuals { requested: usize, maximum: usize },
    TooManyTotalDraws { requested: usize, maximum: usize },
    ZeroPosteriorDrawId,
    MissingPosteriorDrawDigest,
    DuplicatePosteriorDrawId { draw_id: u64 },
    DuplicatePosteriorDrawDigest,
    PosteriorDrawBindingMismatch { branch: &'static str, field: &'static str },
    ZeroPosteriorWeight { draw_id: u64 },
    PosteriorWeightSumMismatch { actual_ppm: u64 },
    PairIdentityMismatch { field: &'static str },
    BranchEvidenceMismatch { branch: &'static str, field: &'static str },
    RealizedOutcomeMismatch { field: &'static str },
    DuplicateIndividualId,
    DuplicatePairId { pair_id: u64 },
    InconsistentCohortProfile { field: &'static str },
    ArithmeticOverflow,
    BaseEvaluator(IndividualEffectError),
}

impl fmt::Display for PosteriorCohortError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIndividualCohort => write!(formatter, "posterior-effect cohort requires at least one individual"),
            Self::EmptyDrawSet => write!(formatter, "each individual requires at least one posterior draw"),
            Self::TooManyDraws { requested, maximum } => write!(formatter, "posterior draw count {requested} exceeds the bounded maximum {maximum}"),
            Self::TooManyIndividuals { requested, maximum } => write!(formatter, "individual count {requested} exceeds the bounded maximum {maximum}"),
            Self::TooManyTotalDraws { requested, maximum } => write!(formatter, "total posterior draw count {requested} exceeds the bounded maximum {maximum}"),
            Self::ZeroPosteriorDrawId => write!(formatter, "posterior draw ID must be nonzero"),
            Self::MissingPosteriorDrawDigest => write!(formatter, "posterior draw digest must be nonzero"),
            Self::DuplicatePosteriorDrawId { draw_id } => write!(formatter, "duplicate posterior draw ID {draw_id}"),
            Self::DuplicatePosteriorDrawDigest => write!(formatter, "duplicate posterior draw digest"),
            Self::PosteriorDrawBindingMismatch { branch, field } => write!(formatter, "{branch} branch does not bind the same posterior draw {field}"),
            Self::ZeroPosteriorWeight { draw_id } => write!(formatter, "posterior draw {draw_id} has zero weight"),
            Self::PosteriorWeightSumMismatch { actual_ppm } => write!(formatter, "posterior PPM weights sum to {actual_ppm}, expected {POSTERIOR_WEIGHT_TOTAL_PPM}"),
            Self::PairIdentityMismatch { field } => write!(formatter, "posterior draws do not share frozen pair identity field {field}"),
            Self::BranchEvidenceMismatch { branch, field } => write!(formatter, "{branch} branch's realized evidence identity changed across posterior draws: {field}"),
            Self::RealizedOutcomeMismatch { field } => write!(formatter, "independently measured realized branch outcome changed across posterior draws: {field}"),
            Self::DuplicateIndividualId => write!(formatter, "an individual appears more than once in one posterior-effect cohort"),
            Self::DuplicatePairId { pair_id } => write!(formatter, "duplicate posterior-effect pair ID {pair_id}"),
            Self::InconsistentCohortProfile { field } => write!(formatter, "posterior-effect cohort mixes incompatible {field} profiles"),
            Self::ArithmeticOverflow => write!(formatter, "posterior-effect weighted integer arithmetic overflowed"),
            Self::BaseEvaluator(error) => write!(formatter, "base individual-effect evaluator rejected posterior draw: {error}"),
        }
    }
}

impl std::error::Error for PosteriorCohortError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn same_pair_context(
    left: &IndividualEffectContext,
    right: &IndividualEffectContext,
) -> Result<(), PosteriorCohortError> {
    macro_rules! require_same {
        ($field:ident) => {
            if left.$field != right.$field {
                return Err(PosteriorCohortError::PairIdentityMismatch {
                    field: stringify!($field),
                });
            }
        };
    }

    require_same!(pair_id);
    require_same!(individual_id_sha256);
    require_same!(factual_history_sha256);
    require_same!(branch_point_state_sha256);
    require_same!(exogenous_forcing_manifest_sha256);
    require_same!(intervention_pair_sha256);
    require_same!(task_sha256);
    require_same!(loss_scale_sha256);
    require_same!(evaluator_profile_sha256);
    require_same!(evaluation_split_sha256);
    require_same!(model_configuration_sha256);
    require_same!(latent_representation_profile_sha256);
    // latent_state_sha256 is intentionally allowed to vary: it is the
    // posterior-draw-conditioned latent input for this sibling prediction pair.
    require_same!(latent_readout_profile_sha256);
    require_same!(rollout_profile_sha256);
    require_same!(pairing_receipt_sha256);
    require_same!(pairing_verifier_profile_sha256);
    require_same!(baseline_action_id);
    require_same!(alternative_action_id);
    require_same!(paired_seed);
    Ok(())
}

fn same_branch_evidence(
    first: &PosteriorDrawBranch,
    current: &PosteriorDrawBranch,
    branch_name: &'static str,
) -> Result<(), PosteriorCohortError> {
    macro_rules! require_same {
        ($field:ident) => {
            if first.branch.$field != current.branch.$field {
                return Err(PosteriorCohortError::BranchEvidenceMismatch {
                    branch: branch_name,
                    field: stringify!($field),
                });
            }
        };
    }

    require_same!(run_id);
    require_same!(action_id);
    require_same!(paired_seed);
    require_same!(factual_history_sha256);
    require_same!(branch_point_state_sha256);
    require_same!(exogenous_forcing_manifest_sha256);
    require_same!(model_configuration_sha256);
    require_same!(latent_representation_profile_sha256);
    // latent_state_sha256 and the two prediction values are draw-conditioned;
    // the base evaluator verifies each branch against its own draw context.
    Ok(())
}

fn validate_draw_binding(draw: &PosteriorDrawPair) -> Result<(), PosteriorCohortError> {
    if draw.posterior_draw_id == 0 {
        return Err(PosteriorCohortError::ZeroPosteriorDrawId);
    }
    if digest_is_empty(&draw.posterior_draw_sha256) {
        return Err(PosteriorCohortError::MissingPosteriorDrawDigest);
    }
    if draw.weight_ppm == 0 {
        return Err(PosteriorCohortError::ZeroPosteriorWeight {
            draw_id: draw.posterior_draw_id,
        });
    }

    for (name, branch) in [
        ("baseline", &draw.baseline),
        ("alternative", &draw.alternative),
    ] {
        if branch.posterior_draw_id != draw.posterior_draw_id {
            return Err(PosteriorCohortError::PosteriorDrawBindingMismatch {
                branch: name,
                field: "posterior_draw_id",
            });
        }
        if branch.posterior_draw_sha256 != draw.posterior_draw_sha256 {
            return Err(PosteriorCohortError::PosteriorDrawBindingMismatch {
                branch: name,
                field: "posterior_draw_sha256",
            });
        }
    }
    Ok(())
}

fn validate_same_realized_outcomes(
    first: &IndividualEffectComparison,
    current: &IndividualEffectComparison,
) -> Result<(), PosteriorCohortError> {
    if first.realized_effect_units != current.realized_effect_units {
        return Err(PosteriorCohortError::RealizedOutcomeMismatch {
            field: "realized_effect_units",
        });
    }
    if first.baseline_applied_at_us != current.baseline_applied_at_us {
        return Err(PosteriorCohortError::RealizedOutcomeMismatch {
            field: "baseline_applied_at_us",
        });
    }
    if first.alternative_applied_at_us != current.alternative_applied_at_us {
        return Err(PosteriorCohortError::RealizedOutcomeMismatch {
            field: "alternative_applied_at_us",
        });
    }
    if first.application_time_delta_us != current.application_time_delta_us {
        return Err(PosteriorCohortError::RealizedOutcomeMismatch {
            field: "application_time_delta_us",
        });
    }
    Ok(())
}

fn sign_of(value: i128) -> i8 {
    match value.cmp(&0) {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
        std::cmp::Ordering::Greater => 1,
    }
}

fn weighted_quantile(
    samples: &[PosteriorEffectSampleComparison],
    use_latent: bool,
    quantile_ppm: u32,
) -> Result<i128, PosteriorCohortError> {
    let mut values: Vec<(i128, u32)> = samples
        .iter()
        .map(|sample| {
            (
                if use_latent {
                    sample.latent_predicted_effect_units
                } else {
                    sample.rollout_predicted_effect_units
                },
                sample.weight_ppm,
            )
        })
        .collect();
    values.sort_unstable_by_key(|(effect, _)| *effect);

    let target = u64::from(quantile_ppm);
    let mut cumulative = 0_u64;
    for (effect, weight) in values {
        cumulative = cumulative
            .checked_add(u64::from(weight))
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        if cumulative >= target {
            return Ok(effect);
        }
    }
    Err(PosteriorCohortError::PosteriorWeightSumMismatch {
        actual_ppm: cumulative,
    })
}

fn summarize_distribution(
    samples: &[PosteriorEffectSampleComparison],
    use_latent: bool,
) -> Result<PosteriorEffectDistribution, PosteriorCohortError> {
    if samples.is_empty() {
        return Err(PosteriorCohortError::EmptyDrawSet);
    }
    let mut min_effect = i128::MAX;
    let mut max_effect = i128::MIN;
    let mut weighted_sum = 0_i128;
    let mut negative_weight = 0_u64;
    let mut zero_weight = 0_u64;
    let mut positive_weight = 0_u64;
    let mut weighted_absolute_error = 0_u128;
    let mut sign_match_weight = 0_u64;
    let mut total_weight = 0_u64;

    for sample in samples {
        let (effect, error, sign_matches) = if use_latent {
            (
                sample.latent_predicted_effect_units,
                sample.latent_absolute_error_units,
                sample.latent_sign_matches,
            )
        } else {
            (
                sample.rollout_predicted_effect_units,
                sample.rollout_absolute_error_units,
                sample.rollout_sign_matches,
            )
        };

        min_effect = min_effect.min(effect);
        max_effect = max_effect.max(effect);
        let weight_i128 = i128::from(sample.weight_ppm);
        weighted_sum = weighted_sum
            .checked_add(
                effect
                    .checked_mul(weight_i128)
                    .ok_or(PosteriorCohortError::ArithmeticOverflow)?,
            )
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        weighted_absolute_error = weighted_absolute_error
            .checked_add(
                error
                    .checked_mul(u128::from(sample.weight_ppm))
                    .ok_or(PosteriorCohortError::ArithmeticOverflow)?,
            )
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        total_weight = total_weight
            .checked_add(u64::from(sample.weight_ppm))
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;

        match sign_of(effect) {
            -1 => negative_weight = negative_weight
                .checked_add(u64::from(sample.weight_ppm))
                .ok_or(PosteriorCohortError::ArithmeticOverflow)?,
            0 => zero_weight = zero_weight
                .checked_add(u64::from(sample.weight_ppm))
                .ok_or(PosteriorCohortError::ArithmeticOverflow)?,
            _ => positive_weight = positive_weight
                .checked_add(u64::from(sample.weight_ppm))
                .ok_or(PosteriorCohortError::ArithmeticOverflow)?,
        }
        if sign_matches {
            sign_match_weight = sign_match_weight
                .checked_add(u64::from(sample.weight_ppm))
                .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        }
    }

    if total_weight != u64::from(POSTERIOR_WEIGHT_TOTAL_PPM) {
        return Err(PosteriorCohortError::PosteriorWeightSumMismatch {
            actual_ppm: total_weight,
        });
    }

    Ok(PosteriorEffectDistribution {
        sample_count: u64::try_from(samples.len())
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        weight_total_ppm: POSTERIOR_WEIGHT_TOTAL_PPM,
        min_effect_units: min_effect,
        q05_effect_units: weighted_quantile(samples, use_latent, Q05_PPM)?,
        median_effect_units: weighted_quantile(samples, use_latent, Q50_PPM)?,
        q95_effect_units: weighted_quantile(samples, use_latent, Q95_PPM)?,
        max_effect_units: max_effect,
        weighted_effect_sum_units_ppm: weighted_sum,
        weight_mass_negative_ppm: u32::try_from(negative_weight)
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        weight_mass_zero_ppm: u32::try_from(zero_weight)
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        weight_mass_positive_ppm: u32::try_from(positive_weight)
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        weighted_absolute_error_sum_units_ppm: weighted_absolute_error,
        sign_match_weight_ppm: u32::try_from(sign_match_weight)
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
    })
}

fn same_cohort_profile(
    first: &IndividualEffectContext,
    current: &IndividualEffectContext,
) -> Result<(), PosteriorCohortError> {
    macro_rules! require_same {
        ($field:ident) => {
            if first.$field != current.$field {
                return Err(PosteriorCohortError::InconsistentCohortProfile {
                    field: stringify!($field),
                });
            }
        };
    }

    require_same!(task_sha256);
    require_same!(intervention_pair_sha256);
    require_same!(loss_scale_sha256);
    require_same!(evaluator_profile_sha256);
    require_same!(evaluation_split_sha256);
    require_same!(model_configuration_sha256);
    require_same!(latent_representation_profile_sha256);
    require_same!(latent_readout_profile_sha256);
    require_same!(rollout_profile_sha256);
    require_same!(pairing_verifier_profile_sha256);
    require_same!(baseline_action_id);
    require_same!(alternative_action_id);
    Ok(())
}

/// Evaluate one individual's posterior-draw cohort using #13520 as the only
/// per-pair effect arithmetic authority.
///
/// The actual simulator pair is reused for every posterior draw. Its branch
/// run IDs, applied timestamps, utilities, factual history, restored state and
/// exogenous-forcing manifest must remain unchanged. Candidate latent and
/// imagined predictions may vary by draw, and each draw's exact identity must
/// be present on both sibling branches.
pub fn evaluate_posterior_individual_cohort(
    cohort: &PosteriorIndividualCohort,
) -> Result<PosteriorIndividualEffectSummary, PosteriorCohortError> {
    if cohort.draws.is_empty() {
        return Err(PosteriorCohortError::EmptyDrawSet);
    }
    if cohort.draws.len() > MAX_POSTERIOR_DRAWS_PER_INDIVIDUAL {
        return Err(PosteriorCohortError::TooManyDraws {
            requested: cohort.draws.len(),
            maximum: MAX_POSTERIOR_DRAWS_PER_INDIVIDUAL,
        });
    }

    let first = &cohort.draws[0];
    let mut draw_ids = BTreeSet::new();
    let mut draw_digests = BTreeSet::new();
    let mut total_weight = 0_u64;
    let mut samples = Vec::with_capacity(cohort.draws.len());
    let mut first_comparison: Option<IndividualEffectComparison> = None;

    for draw in &cohort.draws {
        validate_draw_binding(draw)?;
        if !draw_ids.insert(draw.posterior_draw_id) {
            return Err(PosteriorCohortError::DuplicatePosteriorDrawId {
                draw_id: draw.posterior_draw_id,
            });
        }
        if !draw_digests.insert(draw.posterior_draw_sha256) {
            return Err(PosteriorCohortError::DuplicatePosteriorDrawDigest);
        }
        same_pair_context(&first.context, &draw.context)?;
        same_branch_evidence(&first.baseline, &draw.baseline, "baseline")?;
        same_branch_evidence(&first.alternative, &draw.alternative, "alternative")?;

        total_weight = total_weight
            .checked_add(u64::from(draw.weight_ppm))
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        if total_weight > u64::from(POSTERIOR_WEIGHT_TOTAL_PPM) {
            return Err(PosteriorCohortError::PosteriorWeightSumMismatch {
                actual_ppm: total_weight,
            });
        }

        let pair = IndividualEffectPair {
            context: draw.context.clone(),
            baseline: draw.baseline.branch.clone(),
            alternative: draw.alternative.branch.clone(),
        };
        let certificate = evaluate_individual_effect_retention(&[pair])
            .map_err(PosteriorCohortError::BaseEvaluator)?;
        let comparison = certificate
            .comparisons
            .first()
            .ok_or(PosteriorCohortError::EmptyDrawSet)?;

        if let Some(first_value) = &first_comparison {
            validate_same_realized_outcomes(first_value, comparison)?;
        } else {
            first_comparison = Some(comparison.clone());
        }

        samples.push(PosteriorEffectSampleComparison {
            posterior_draw_id: draw.posterior_draw_id,
            posterior_draw_sha256: draw.posterior_draw_sha256,
            weight_ppm: draw.weight_ppm,
            latent_predicted_effect_units: comparison.latent_predicted_effect_units,
            rollout_predicted_effect_units: comparison.rollout_predicted_effect_units,
            latent_absolute_error_units: comparison.latent_absolute_error_units,
            rollout_absolute_error_units: comparison.rollout_absolute_error_units,
            latent_sign_matches: comparison.latent_sign_matches,
            rollout_sign_matches: comparison.rollout_sign_matches,
        });
    }

    if total_weight != u64::from(POSTERIOR_WEIGHT_TOTAL_PPM) {
        return Err(PosteriorCohortError::PosteriorWeightSumMismatch {
            actual_ppm: total_weight,
        });
    }

    samples.sort_unstable_by_key(|sample| sample.posterior_draw_id);
    let comparison = first_comparison.ok_or(PosteriorCohortError::EmptyDrawSet)?;

    Ok(PosteriorIndividualEffectSummary {
        individual_id_sha256: first.context.individual_id_sha256,
        pair_id: first.context.pair_id,
        realized_effect_units: comparison.realized_effect_units,
        baseline_applied_at_us: comparison.baseline_applied_at_us,
        alternative_applied_at_us: comparison.alternative_applied_at_us,
        application_time_delta_us: comparison.application_time_delta_us,
        latent_effects: summarize_distribution(&samples, true)?,
        rollout_effects: summarize_distribution(&samples, false)?,
        samples,
    })
}

/// Evaluate multiple individual cohorts without pooling their posterior draws.
///
/// The resulting individual_count is the number of unique individual
/// identities, never the number of posterior samples. All individuals share
/// the same declared task/action/model/evaluator profile; their histories,
/// state, exogenous manifests and pairing receipts remain individual-specific.
pub fn evaluate_posterior_effect_cohort(
    cohorts: &[PosteriorIndividualCohort],
) -> Result<PosteriorEffectCohortCertificate, PosteriorCohortError> {
    if cohorts.is_empty() {
        return Err(PosteriorCohortError::EmptyIndividualCohort);
    }
    if cohorts.len() > MAX_POSTERIOR_INDIVIDUALS {
        return Err(PosteriorCohortError::TooManyIndividuals {
            requested: cohorts.len(),
            maximum: MAX_POSTERIOR_INDIVIDUALS,
        });
    }

    let first_draw = cohorts[0]
        .draws
        .first()
        .ok_or(PosteriorCohortError::EmptyDrawSet)?;
    let first_context = &first_draw.context;
    let mut individual_ids = BTreeSet::new();
    let mut pair_ids = BTreeSet::new();
    let mut individuals = Vec::with_capacity(cohorts.len());
    let mut total_draws = 0_usize;

    for cohort in cohorts {
        let first = cohort.draws.first().ok_or(PosteriorCohortError::EmptyDrawSet)?;
        same_cohort_profile(first_context, &first.context)?;

        if !individual_ids.insert(first.context.individual_id_sha256) {
            return Err(PosteriorCohortError::DuplicateIndividualId);
        }
        if !pair_ids.insert(first.context.pair_id) {
            return Err(PosteriorCohortError::DuplicatePairId {
                pair_id: first.context.pair_id,
            });
        }

        total_draws = total_draws
            .checked_add(cohort.draws.len())
            .ok_or(PosteriorCohortError::ArithmeticOverflow)?;
        if total_draws > MAX_TOTAL_POSTERIOR_DRAWS {
            return Err(PosteriorCohortError::TooManyTotalDraws {
                requested: total_draws,
                maximum: MAX_TOTAL_POSTERIOR_DRAWS,
            });
        }

        individuals.push(evaluate_posterior_individual_cohort(cohort)?);
    }

    individuals.sort_unstable_by_key(|summary| summary.individual_id_sha256);

    Ok(PosteriorEffectCohortCertificate {
        task_sha256: first_context.task_sha256,
        intervention_pair_sha256: first_context.intervention_pair_sha256,
        loss_scale_sha256: first_context.loss_scale_sha256,
        baseline_action_id: first_context.baseline_action_id.clone(),
        alternative_action_id: first_context.alternative_action_id.clone(),
        individual_count: u64::try_from(individuals.len())
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        total_posterior_draw_count: u64::try_from(total_draws)
            .map_err(|_| PosteriorCohortError::ArithmeticOverflow)?,
        individuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::individual_effect_retention::NonApplicationReason;

    fn digest(seed: u8) -> [u8; 32] {
        let mut value = [seed; 32];
        value[0] = seed;
        value[31] = seed.wrapping_add(1);
        value
    }

    fn make_branch(
        run_id: u64,
        action_id: &str,
        paired_seed: u64,
        context: &IndividualEffectContext,
        applied_at_us: u64,
        utilities: (i64, i64, i64),
    ) -> IndividualEffectBranch {
        IndividualEffectBranch {
            run_id,
            action_id: action_id.to_owned(),
            paired_seed,
            factual_history_sha256: context.factual_history_sha256,
            branch_point_state_sha256: context.branch_point_state_sha256,
            exogenous_forcing_manifest_sha256: context.exogenous_forcing_manifest_sha256,
            model_configuration_sha256: context.model_configuration_sha256,
            latent_representation_profile_sha256: context.latent_representation_profile_sha256,
            latent_state_sha256: context.latent_state_sha256,
            disposition: BranchDisposition::Applied {
                applied_at_us,
                realized_utility_units: utilities.0,
            },
            latent_predicted_utility_units: utilities.1,
            rollout_predicted_utility_units: utilities.2,
        }
    }

    fn make_draw(
        individual_seed: u8,
        pair_id: u64,
        draw_id: u64,
        draw_seed: u8,
        weight_ppm: u32,
        effects: (i64, i64, i64),
    ) -> PosteriorDrawPair {
        let individual_digest = digest(individual_seed);
        let context = IndividualEffectContext {
            pair_id,
            individual_id_sha256: individual_digest,
            factual_history_sha256: digest(individual_seed.wrapping_add(10)),
            branch_point_state_sha256: digest(individual_seed.wrapping_add(20)),
            exogenous_forcing_manifest_sha256: digest(individual_seed.wrapping_add(30)),
            intervention_pair_sha256: digest(40),
            task_sha256: digest(41),
            loss_scale_sha256: digest(42),
            evaluator_profile_sha256: digest(43),
            evaluation_split_sha256: digest(44),
            model_configuration_sha256: digest(45),
            latent_representation_profile_sha256: digest(46),
            latent_state_sha256: digest(draw_seed),
            latent_readout_profile_sha256: digest(47),
            rollout_profile_sha256: digest(48),
            pairing_receipt_sha256: digest(individual_seed.wrapping_add(50)),
            pairing_verifier_profile_sha256: digest(49),
            baseline_action_id: "a0".to_owned(),
            alternative_action_id: "a1".to_owned(),
            paired_seed: 900,
        };

        let baseline = make_branch(
            5_001,
            "a0",
            context.paired_seed,
            &context,
            10_000,
            (100, 1_000, 2_000),
        );
        let alternative = make_branch(
            5_002,
            "a1",
            context.paired_seed,
            &context,
            10_100,
            (
                100_i64 + effects.0,
                1_000_i64 + effects.1,
                2_000_i64 + effects.2,
            ),
        );
        let draw_digest = digest(draw_seed.wrapping_add(70));

        PosteriorDrawPair {
            posterior_draw_id: draw_id,
            posterior_draw_sha256: draw_digest,
            weight_ppm,
            context,
            baseline: PosteriorDrawBranch {
                posterior_draw_id: draw_id,
                posterior_draw_sha256: draw_digest,
                branch: baseline,
            },
            alternative: PosteriorDrawBranch {
                posterior_draw_id: draw_id,
                posterior_draw_sha256: draw_digest,
                branch: alternative,
            },
        }
    }

    fn cohort(draws: Vec<PosteriorDrawPair>) -> PosteriorIndividualCohort {
        PosteriorIndividualCohort { draws }
    }

    #[test]
    fn shared_draw_preserves_bimodal_effect_distribution() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (0, -10, -10)),
            make_draw(1, 101, 2, 52, 500_000, (0, 10, 10)),
        ]);
        let result = evaluate_posterior_individual_cohort(&input).unwrap();

        assert_eq!(result.realized_effect_units, 0);
        assert_eq!(result.latent_effects.min_effect_units, -10);
        assert_eq!(result.latent_effects.q05_effect_units, -10);
        // Lower inverse-CDF median: the first 50% mass is the negative mode.
        assert_eq!(result.latent_effects.median_effect_units, -10);
        assert_eq!(result.latent_effects.q95_effect_units, 10);
        assert_eq!(result.latent_effects.max_effect_units, 10);
        assert_eq!(result.latent_effects.weight_mass_negative_ppm, 500_000);
        assert_eq!(result.latent_effects.weight_mass_positive_ppm, 500_000);
        assert_eq!(result.latent_effects.weight_mass_zero_ppm, 0);
        assert_eq!(result.latent_effects.weighted_effect_sum_units_ppm, 0);
    }

    #[test]
    fn posterior_draw_id_mismatch_on_one_sibling_rejects() {
        let mut draw = make_draw(1, 101, 1, 51, 1_000_000, (2, 1, 1));
        draw.alternative.posterior_draw_id = 2;
        let error = evaluate_posterior_individual_cohort(&cohort(vec![draw])).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::PosteriorDrawBindingMismatch {
                branch: "alternative",
                field: "posterior_draw_id",
            }
        );
    }

    #[test]
    fn posterior_draw_digest_mismatch_on_one_sibling_rejects() {
        let mut draw = make_draw(1, 101, 1, 51, 1_000_000, (2, 1, 1));
        draw.alternative.posterior_draw_sha256 = digest(99);
        let error = evaluate_posterior_individual_cohort(&cohort(vec![draw])).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::PosteriorDrawBindingMismatch {
                branch: "alternative",
                field: "posterior_draw_sha256",
            }
        );
    }

    #[test]
    fn same_realized_branch_outcomes_are_reused_for_each_draw() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            make_draw(1, 101, 2, 52, 500_000, (2, 4, 5)),
        ]);
        let result = evaluate_posterior_individual_cohort(&input).unwrap();
        assert_eq!(result.realized_effect_units, 2);
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.samples[0].latent_predicted_effect_units, 1);
        assert_eq!(result.samples[1].latent_predicted_effect_units, 4);
        assert_eq!(result.baseline_applied_at_us, 10_000);
        assert_eq!(result.alternative_applied_at_us, 10_100);
    }

    #[test]
    fn realized_outcome_drift_across_posterior_draws_rejects() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            make_draw(1, 101, 2, 52, 500_000, (3, 1, 1)),
        ]);
        let error = evaluate_posterior_individual_cohort(&input).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::RealizedOutcomeMismatch {
                field: "realized_effect_units",
            }
        );
    }

    #[test]
    fn applied_timestamp_drift_across_draws_rejects() {
        let mut second = make_draw(1, 101, 2, 52, 500_000, (2, 1, 1));
        second.baseline.branch.disposition = BranchDisposition::Applied {
            applied_at_us: 10_001,
            realized_utility_units: 100,
        };
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            second,
        ]);
        let error = evaluate_posterior_individual_cohort(&input).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::RealizedOutcomeMismatch {
                field: "baseline_applied_at_us",
            }
        );
    }

    #[test]
    fn actual_branch_run_id_drift_across_draws_rejects() {
        let mut second = make_draw(1, 101, 2, 52, 500_000, (2, 1, 1));
        second.baseline.branch.run_id = 5_111;
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            second,
        ]);
        let error = evaluate_posterior_individual_cohort(&input).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::BranchEvidenceMismatch {
                branch: "baseline",
                field: "run_id",
            }
        );
    }

    #[test]
    fn posterior_weights_must_sum_to_one_million_ppm() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 400_000, (2, 1, 1)),
            make_draw(1, 101, 2, 52, 400_000, (2, 3, 3)),
        ]);
        let error = evaluate_posterior_individual_cohort(&input).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::PosteriorWeightSumMismatch { actual_ppm: 800_000 }
        );
    }

    #[test]
    fn duplicate_posterior_draw_id_rejects() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            make_draw(1, 101, 1, 52, 500_000, (2, 3, 3)),
        ]);
        let error = evaluate_posterior_individual_cohort(&input).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::DuplicatePosteriorDrawId { draw_id: 1 }
        );
    }

    #[test]
    fn duplicate_posterior_draw_digest_rejects() {
        let duplicate_digest = digest(121);
        let mut first = make_draw(1, 101, 1, 51, 500_000, (2, 1, 1));
        first.posterior_draw_sha256 = duplicate_digest;
        first.baseline.posterior_draw_sha256 = duplicate_digest;
        first.alternative.posterior_draw_sha256 = duplicate_digest;
        let mut second = make_draw(1, 101, 2, 52, 500_000, (2, 3, 3));
        second.posterior_draw_sha256 = duplicate_digest;
        second.baseline.posterior_draw_sha256 = duplicate_digest;
        second.alternative.posterior_draw_sha256 = duplicate_digest;
        let error = evaluate_posterior_individual_cohort(&cohort(vec![first, second])).unwrap_err();
        assert_eq!(error, PosteriorCohortError::DuplicatePosteriorDrawDigest);
    }

    #[test]
    fn posterior_draws_do_not_inflate_individual_count() {
        let first = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
            make_draw(1, 101, 2, 52, 500_000, (2, 3, 3)),
        ]);
        let second = cohort(vec![
            make_draw(2, 102, 1, 61, 1_000_000, (-1, -1, -1)),
        ]);
        let result = evaluate_posterior_effect_cohort(&[first, second]).unwrap();

        assert_eq!(result.individual_count, 2);
        assert_eq!(result.total_posterior_draw_count, 3);
        assert_eq!(result.individuals.len(), 2);
        assert_ne!(
            result.individuals[0].individual_id_sha256,
            result.individuals[1].individual_id_sha256
        );
    }

    #[test]
    fn cohort_and_draw_input_order_produce_canonical_output() {
        let first = cohort(vec![
            make_draw(1, 101, 2, 52, 500_000, (2, 4, 5)),
            make_draw(1, 101, 1, 51, 500_000, (2, 1, 1)),
        ]);
        let second = cohort(vec![
            make_draw(2, 102, 1, 61, 1_000_000, (-1, -1, -1)),
        ]);
        let forward = evaluate_posterior_effect_cohort(&[first.clone(), second.clone()]).unwrap();
        let reverse = evaluate_posterior_effect_cohort(&[second, first]).unwrap();
        assert_eq!(forward, reverse);
        assert_eq!(forward.individuals[0].samples[0].posterior_draw_id, 1);
    }

    #[test]
    fn cohort_profile_drift_rejects() {
        let first = cohort(vec![
            make_draw(1, 101, 1, 51, 1_000_000, (2, 1, 1)),
        ]);
        let mut second_draw = make_draw(2, 102, 1, 61, 1_000_000, (2, 1, 1));
        second_draw.context.model_configuration_sha256 = digest(88);
        second_draw.baseline.branch.model_configuration_sha256 = digest(88);
        second_draw.alternative.branch.model_configuration_sha256 = digest(88);
        let second = cohort(vec![second_draw]);
        let error = evaluate_posterior_effect_cohort(&[first, second]).unwrap_err();
        assert_eq!(
            error,
            PosteriorCohortError::InconsistentCohortProfile {
                field: "model_configuration_sha256",
            }
        );
    }

    #[test]
    fn missing_or_not_applied_branch_is_not_silently_dropped() {
        let mut draw = make_draw(1, 101, 1, 51, 1_000_000, (2, 1, 1));
        draw.alternative.branch.disposition = BranchDisposition::NotApplied {
            terminal_at_us: 10_500,
            reason: NonApplicationReason::DeadlineMiss,
        };
        let error = evaluate_posterior_individual_cohort(&cohort(vec![draw])).unwrap_err();
        assert!(matches!(
            error,
            PosteriorCohortError::BaseEvaluator(IndividualEffectError::BranchNotApplied { .. })
        ));
    }

    #[test]
    fn latent_and_rollout_distributions_remain_separate() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 500_000, (2, 1, -5)),
            make_draw(1, 101, 2, 52, 500_000, (2, 3, 7)),
        ]);
        let result = evaluate_posterior_individual_cohort(&input).unwrap();
        assert_eq!(result.latent_effects.min_effect_units, 1);
        assert_eq!(result.latent_effects.max_effect_units, 3);
        assert_eq!(result.rollout_effects.min_effect_units, -5);
        assert_eq!(result.rollout_effects.max_effect_units, 7);
    }

    #[test]
    fn weighted_mean_and_error_numerators_are_exact() {
        let input = cohort(vec![
            make_draw(1, 101, 1, 51, 250_000, (2, 0, 0)),
            make_draw(1, 101, 2, 52, 750_000, (2, 4, 6)),
        ]);
        let result = evaluate_posterior_individual_cohort(&input).unwrap();
        assert_eq!(result.latent_effects.weighted_effect_sum_units_ppm, 3_000_000);
        assert_eq!(result.rollout_effects.weighted_effect_sum_units_ppm, 4_500_000);
        assert_eq!(result.latent_effects.weighted_absolute_error_sum_units_ppm, 2_000_000);
        assert_eq!(result.rollout_effects.weighted_absolute_error_sum_units_ppm, 3_500_000);
    }

    #[test]
    fn individual_cohort_requires_exact_weight_total() {
        let error = evaluate_posterior_individual_cohort(&cohort(Vec::new())).unwrap_err();
        assert_eq!(error, PosteriorCohortError::EmptyDrawSet);
    }
}

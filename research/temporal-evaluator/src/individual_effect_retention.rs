//! Evaluator-side comparison of history-conditioned individual effects.
//!
//! This module is deliberately narrower than a causal model or counterfactual
//! simulator. It consumes paired, independently measured branch outcomes and
//! candidate-produced latent-readout / imagined-rollout predictions, then
//! computes exact per-individual effect errors.
//!
//! Branch construction and stochastic correspondence remain owned by
//! Symthaea #13375/#13377 and the external Symtropy verifier. The pairing
//! receipt digests below are references only: this module does not authenticate
//! receipt bytes or prove that the verifier was run. The evaluator launcher
//! must independently retrieve and verify those artifacts before admission.
//!
//! The module is evaluator-only. Never link it into a candidate process or
//! expose the realized branch outcomes / per-individual labels to the candidate.
//! Package separation is not an OS security boundary; #9437 remains required.
//!
//! This module does not estimate abducible share, infer structural causal
//! variables, calibrate predictions, perform statistical inference, or
//! authorize actions. It reports finite paired descriptive errors only.

use std::collections::BTreeSet;
use std::fmt;

pub const MAX_INDIVIDUAL_EFFECT_PAIRS: usize = 4_096;
pub const EFFECT_ACCURACY_PPM: u32 = 1_000_000;

/// Why a paired branch did not produce a realized outcome.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NonApplicationReason {
    DeadlineMiss,
    NoQualifiedAction,
    ExecutionFailure,
    ConsumerRejected,
}

/// Whether the branch action was actually applied.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BranchDisposition {
    Applied {
        applied_at_us: u64,
        realized_utility_units: i64,
    },
    NotApplied {
        terminal_at_us: u64,
        reason: NonApplicationReason,
    },
}

/// Frozen identity for one individual and one two-action intervention pair.
///
/// Every digest is evaluator-side provenance. The candidate must not receive
/// hidden individual IDs, branch outcomes, or evaluator references.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualEffectContext {
    pub pair_id: u64,
    pub individual_id_sha256: [u8; 32],
    pub factual_history_sha256: [u8; 32],
    pub branch_point_state_sha256: [u8; 32],
    /// Shared exogenous forcing commitment for both sibling interventions.
    pub exogenous_forcing_manifest_sha256: [u8; 32],
    pub intervention_pair_sha256: [u8; 32],
    pub task_sha256: [u8; 32],
    pub loss_scale_sha256: [u8; 32],
    pub evaluator_profile_sha256: [u8; 32],
    pub evaluation_split_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub latent_representation_profile_sha256: [u8; 32],
    /// The same history-inferred latent must feed both sibling predictions.
    pub latent_state_sha256: [u8; 32],
    pub latent_readout_profile_sha256: [u8; 32],
    pub rollout_profile_sha256: [u8; 32],
    /// Existing #13377 / Symtropy semantic-pairing receipt reference.
    pub pairing_receipt_sha256: [u8; 32],
    /// Exact verifier profile expected to validate the pairing receipt.
    pub pairing_verifier_profile_sha256: [u8; 32],
    pub baseline_action_id: String,
    pub alternative_action_id: String,
    pub paired_seed: u64,
}

/// One executed or non-executed branch, with both candidate predictions.
///
/// Latent-readout and rollout predictions are kept separate so a loss in the
/// decoded/imagined path can be distinguished from information absent in the
/// frozen latent readout. Predictions are utilities in the exact loss scale
/// named by the parent context.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualEffectBranch {
    pub run_id: u64,
    pub action_id: String,
    pub factual_history_sha256: [u8; 32],
    pub branch_point_state_sha256: [u8; 32],
    pub exogenous_forcing_manifest_sha256: [u8; 32],
    pub model_configuration_sha256: [u8; 32],
    pub latent_representation_profile_sha256: [u8; 32],
    pub latent_state_sha256: [u8; 32],
    pub disposition: BranchDisposition,
    pub latent_predicted_utility_units: i64,
    pub rollout_predicted_utility_units: i64,
}

/// Two sibling interventions on the same restored individual.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualEffectPair {
    pub context: IndividualEffectContext,
    pub baseline: IndividualEffectBranch,
    pub alternative: IndividualEffectBranch,
}

/// Exact sign of an individual treatment effect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EffectSign {
    Negative,
    Zero,
    Positive,
}

impl EffectSign {
    fn of(value: i128) -> Self {
        match value.cmp(&0) {
            std::cmp::Ordering::Less => Self::Negative,
            std::cmp::Ordering::Equal => Self::Zero,
            std::cmp::Ordering::Greater => Self::Positive,
        }
    }
}

/// Per-individual contrast, retained for downstream cluster-level analysis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualEffectComparison {
    pub pair_id: u64,
    pub individual_id_sha256: [u8; 32],
    pub realized_effect_units: i128,
    pub latent_predicted_effect_units: i128,
    pub rollout_predicted_effect_units: i128,
    pub latent_absolute_error_units: u128,
    pub rollout_absolute_error_units: u128,
    pub latent_sign_matches: bool,
    pub rollout_sign_matches: bool,
}

/// Finite descriptive certificate; it contains no confidence interval.
///
/// Error and mean fields are numerators with individual_count as denominator.
/// Keeping integer numerators avoids silently changing the task-loss scale or
/// introducing floating-point rounding into this reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndividualEffectRetentionCertificate {
    pub task_sha256: [u8; 32],
    pub intervention_pair_sha256: [u8; 32],
    pub loss_scale_sha256: [u8; 32],
    pub baseline_action_id: String,
    pub alternative_action_id: String,
    pub individual_count: u64,
    pub realized_effect_sum_units: i128,
    pub latent_predicted_effect_sum_units: i128,
    pub rollout_predicted_effect_sum_units: i128,
    pub latent_effect_bias_sum_units: i128,
    pub rollout_effect_bias_sum_units: i128,
    pub latent_absolute_error_sum_units: u128,
    pub rollout_absolute_error_sum_units: u128,
    pub latent_sign_match_count: u64,
    pub rollout_sign_match_count: u64,
    pub latent_sign_accuracy_ppm: u32,
    pub rollout_sign_accuracy_ppm: u32,
    /// Positive means the rollout has more absolute error than the latent readout.
    pub rollout_minus_latent_absolute_error_sum_units: i128,
    pub population_mean_effect_is_zero: bool,
    pub individual_effect_heterogeneity_present: bool,
    pub comparisons: Vec<IndividualEffectComparison>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IndividualEffectError {
    EmptyPairSet,
    TooManyPairs { requested: usize, maximum: usize },
    ZeroPairId,
    ZeroRunId,
    DuplicatePairId { pair_id: u64 },
    DuplicateIndividualId,
    DuplicateRunId { run_id: u64 },
    MissingDigest { field: &'static str },
    EmptyActionId,
    IdenticalActions,
    BranchActionMismatch { pair_id: u64, expected: String, actual: String },
    BranchProvenanceMismatch { pair_id: u64, action_id: String, field: &'static str },
    SharedProfileMismatch { field: &'static str },
    BranchNotApplied {
        pair_id: u64,
        action_id: String,
        terminal_at_us: u64,
        reason: NonApplicationReason,
    },
    ArithmeticOverflow,
}

impl fmt::Display for IndividualEffectError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPairSet => write!(formatter, "individual-effect evaluation requires at least one pair"),
            Self::TooManyPairs { requested, maximum } => write!(
                formatter,
                "individual-effect pair count {requested} exceeds the bounded maximum {maximum}"
            ),
            Self::ZeroPairId => write!(formatter, "individual-effect pair ID must be nonzero"),
            Self::ZeroRunId => write!(formatter, "branch run ID must be nonzero"),
            Self::DuplicatePairId { pair_id } => write!(formatter, "duplicate individual-effect pair ID {pair_id}"),
            Self::DuplicateIndividualId => write!(formatter, "an individual appears more than once in one evaluation cohort"),
            Self::DuplicateRunId { run_id } => write!(formatter, "duplicate branch run ID {run_id}"),
            Self::MissingDigest { field } => write!(formatter, "required provenance digest {field} is empty"),
            Self::EmptyActionId => write!(formatter, "intervention action IDs must be nonempty"),
            Self::IdenticalActions => write!(formatter, "baseline and alternative action IDs must differ"),
            Self::BranchActionMismatch { pair_id, expected, actual } => write!(
                formatter,
                "pair {pair_id} expected action {expected}, received branch action {actual}"
            ),
            Self::BranchProvenanceMismatch { pair_id, action_id, field } => write!(
                formatter,
                "pair {pair_id} branch {action_id} does not match frozen provenance field {field}"
            ),
            Self::SharedProfileMismatch { field } => write!(
                formatter,
                "individual-effect cohort mixes incompatible {field} profiles"
            ),
            Self::BranchNotApplied { pair_id, action_id, terminal_at_us, reason } => write!(
                formatter,
                "pair {pair_id} action {action_id} was not applied at terminal time {terminal_at_us}: {reason:?}"
            ),
            Self::ArithmeticOverflow => write!(formatter, "individual-effect arithmetic overflowed"),
        }
    }
}

impl std::error::Error for IndividualEffectError {}

fn digest_is_empty(digest: &[u8; 32]) -> bool {
    digest.iter().all(|byte| *byte == 0)
}

fn require_digest(
    digest: &[u8; 32],
    field: &'static str,
) -> Result<(), IndividualEffectError> {
    if digest_is_empty(digest) {
        return Err(IndividualEffectError::MissingDigest { field });
    }
    Ok(())
}

fn validate_context(context: &IndividualEffectContext) -> Result<(), IndividualEffectError> {
    if context.pair_id == 0 {
        return Err(IndividualEffectError::ZeroPairId);
    }
    for (digest, name) in [
        (&context.individual_id_sha256, "individual_id_sha256"),
        (&context.factual_history_sha256, "factual_history_sha256"),
        (&context.branch_point_state_sha256, "branch_point_state_sha256"),
        (&context.exogenous_forcing_manifest_sha256, "exogenous_forcing_manifest_sha256"),
        (&context.intervention_pair_sha256, "intervention_pair_sha256"),
        (&context.task_sha256, "task_sha256"),
        (&context.loss_scale_sha256, "loss_scale_sha256"),
        (&context.evaluator_profile_sha256, "evaluator_profile_sha256"),
        (&context.evaluation_split_sha256, "evaluation_split_sha256"),
        (&context.model_configuration_sha256, "model_configuration_sha256"),
        (&context.latent_representation_profile_sha256, "latent_representation_profile_sha256"),
        (&context.latent_state_sha256, "latent_state_sha256"),
        (&context.latent_readout_profile_sha256, "latent_readout_profile_sha256"),
        (&context.rollout_profile_sha256, "rollout_profile_sha256"),
        (&context.pairing_receipt_sha256, "pairing_receipt_sha256"),
        (&context.pairing_verifier_profile_sha256, "pairing_verifier_profile_sha256"),
    ] {
        require_digest(digest, name)?;
    }
    if context.baseline_action_id.trim().is_empty()
        || context.alternative_action_id.trim().is_empty()
    {
        return Err(IndividualEffectError::EmptyActionId);
    }
    if context.baseline_action_id == context.alternative_action_id {
        return Err(IndividualEffectError::IdenticalActions);
    }
    Ok(())
}

fn validate_branch(
    context: &IndividualEffectContext,
    branch: &IndividualEffectBranch,
    expected_action_id: &str,
) -> Result<i64, IndividualEffectError> {
    if branch.run_id == 0 {
        return Err(IndividualEffectError::ZeroRunId);
    }
    if branch.action_id != expected_action_id {
        return Err(IndividualEffectError::BranchActionMismatch {
            pair_id: context.pair_id,
            expected: expected_action_id.to_owned(),
            actual: branch.action_id.clone(),
        });
    }
    for (actual, expected, field) in [
        (
            &branch.factual_history_sha256,
            &context.factual_history_sha256,
            "factual_history_sha256",
        ),
        (
            &branch.branch_point_state_sha256,
            &context.branch_point_state_sha256,
            "branch_point_state_sha256",
        ),
        (
            &branch.exogenous_forcing_manifest_sha256,
            &context.exogenous_forcing_manifest_sha256,
            "exogenous_forcing_manifest_sha256",
        ),
        (
            &branch.model_configuration_sha256,
            &context.model_configuration_sha256,
            "model_configuration_sha256",
        ),
        (
            &branch.latent_representation_profile_sha256,
            &context.latent_representation_profile_sha256,
            "latent_representation_profile_sha256",
        ),
        (
            &branch.latent_state_sha256,
            &context.latent_state_sha256,
            "latent_state_sha256",
        ),
    ] {
        if actual != expected {
            return Err(IndividualEffectError::BranchProvenanceMismatch {
                pair_id: context.pair_id,
                action_id: branch.action_id.clone(),
                field,
            });
        }
    }
    match branch.disposition {
        BranchDisposition::Applied { realized_utility_units, .. } => Ok(realized_utility_units),
        BranchDisposition::NotApplied { terminal_at_us, reason } => {
            Err(IndividualEffectError::BranchNotApplied {
                pair_id: context.pair_id,
                action_id: branch.action_id.clone(),
                terminal_at_us,
                reason,
            })
        }
    }
}

fn signed_effect(alternative: i64, baseline: i64) -> i128 {
    i128::from(alternative) - i128::from(baseline)
}

fn absolute_error(left: i128, right: i128) -> Result<u128, IndividualEffectError> {
    left.checked_sub(right)
        .map(i128::unsigned_abs)
        .ok_or(IndividualEffectError::ArithmeticOverflow)
}

fn sign_accuracy_ppm(matches: u64, count: u64) -> Result<u32, IndividualEffectError> {
    let scaled = matches
        .checked_mul(u64::from(EFFECT_ACCURACY_PPM))
        .ok_or(IndividualEffectError::ArithmeticOverflow)?;
    u32::try_from(scaled / count).map_err(|_| IndividualEffectError::ArithmeticOverflow)
}

/// Evaluate paired individual effects using already-qualified branch receipts.
///
/// This function checks that both outcomes share the same factual history,
/// restored branch-point state, exogenous forcing commitment, model and latent
/// identity. It also requires nonempty references to the existing pairing
/// verifier receipt/profile. It does not authenticate those receipts; a trusted
/// evaluator wrapper must validate the referenced artifacts first.
pub fn evaluate_individual_effect_retention(
    pairs: &[IndividualEffectPair],
) -> Result<IndividualEffectRetentionCertificate, IndividualEffectError> {
    if pairs.is_empty() {
        return Err(IndividualEffectError::EmptyPairSet);
    }
    if pairs.len() > MAX_INDIVIDUAL_EFFECT_PAIRS {
        return Err(IndividualEffectError::TooManyPairs {
            requested: pairs.len(),
            maximum: MAX_INDIVIDUAL_EFFECT_PAIRS,
        });
    }

    let first = &pairs[0].context;
    let mut pair_ids = BTreeSet::new();
    let mut individual_ids = BTreeSet::new();
    let mut run_ids = BTreeSet::new();

    let mut realized_sum = 0_i128;
    let mut latent_sum = 0_i128;
    let mut rollout_sum = 0_i128;
    let mut latent_bias_sum = 0_i128;
    let mut rollout_bias_sum = 0_i128;
    let mut latent_absolute_error_sum = 0_u128;
    let mut rollout_absolute_error_sum = 0_u128;
    let mut latent_sign_match_count = 0_u64;
    let mut rollout_sign_match_count = 0_u64;
    let mut individual_effects = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let context = &pair.context;
        validate_context(context)?;

        if context.task_sha256 != first.task_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "task" });
        }
        if context.intervention_pair_sha256 != first.intervention_pair_sha256
            || context.baseline_action_id != first.baseline_action_id
            || context.alternative_action_id != first.alternative_action_id
        {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "intervention pair" });
        }
        if context.loss_scale_sha256 != first.loss_scale_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "loss scale" });
        }
        if context.evaluator_profile_sha256 != first.evaluator_profile_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "evaluator profile" });
        }
        if context.evaluation_split_sha256 != first.evaluation_split_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "evaluation split" });
        }
        if context.model_configuration_sha256 != first.model_configuration_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch { field: "model configuration" });
        }
        if context.latent_representation_profile_sha256
            != first.latent_representation_profile_sha256
        {
            return Err(IndividualEffectError::SharedProfileMismatch {
                field: "latent representation profile",
            });
        }
        if context.latent_readout_profile_sha256 != first.latent_readout_profile_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch {
                field: "latent readout profile",
            });
        }
        if context.rollout_profile_sha256 != first.rollout_profile_sha256 {
            return Err(IndividualEffectError::SharedProfileMismatch {
                field: "rollout profile",
            });
        }
        if context.pairing_verifier_profile_sha256
            != first.pairing_verifier_profile_sha256
        {
            return Err(IndividualEffectError::SharedProfileMismatch {
                field: "pairing verifier profile",
            });
        }

        if !pair_ids.insert(context.pair_id) {
            return Err(IndividualEffectError::DuplicatePairId {
                pair_id: context.pair_id,
            });
        }
        if !individual_ids.insert(context.individual_id_sha256) {
            return Err(IndividualEffectError::DuplicateIndividualId);
        }
        if !run_ids.insert(pair.baseline.run_id) {
            return Err(IndividualEffectError::DuplicateRunId {
                run_id: pair.baseline.run_id,
            });
        }
        if !run_ids.insert(pair.alternative.run_id) {
            return Err(IndividualEffectError::DuplicateRunId {
                run_id: pair.alternative.run_id,
            });
        }

        let realized_baseline = validate_branch(
            context,
            &pair.baseline,
            &context.baseline_action_id,
        )?;
        let realized_alternative = validate_branch(
            context,
            &pair.alternative,
            &context.alternative_action_id,
        )?;

        let realized_effect = signed_effect(realized_alternative, realized_baseline);
        let latent_effect = signed_effect(
            pair.alternative.latent_predicted_utility_units,
            pair.baseline.latent_predicted_utility_units,
        );
        let rollout_effect = signed_effect(
            pair.alternative.rollout_predicted_utility_units,
            pair.baseline.rollout_predicted_utility_units,
        );
        let latent_error = absolute_error(latent_effect, realized_effect)?;
        let rollout_error = absolute_error(rollout_effect, realized_effect)?;
        let latent_sign_matches = EffectSign::of(latent_effect) == EffectSign::of(realized_effect);
        let rollout_sign_matches =
            EffectSign::of(rollout_effect) == EffectSign::of(realized_effect);

        realized_sum = realized_sum.checked_add(realized_effect)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        latent_sum = latent_sum.checked_add(latent_effect)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        rollout_sum = rollout_sum.checked_add(rollout_effect)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        latent_bias_sum = latent_bias_sum.checked_add(latent_effect - realized_effect)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        rollout_bias_sum = rollout_bias_sum.checked_add(rollout_effect - realized_effect)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        latent_absolute_error_sum = latent_absolute_error_sum.checked_add(latent_error)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        rollout_absolute_error_sum = rollout_absolute_error_sum.checked_add(rollout_error)
            .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        if latent_sign_matches {
            latent_sign_match_count = latent_sign_match_count.checked_add(1)
                .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        }
        if rollout_sign_matches {
            rollout_sign_match_count = rollout_sign_match_count.checked_add(1)
                .ok_or(IndividualEffectError::ArithmeticOverflow)?;
        }

        individual_effects.push(IndividualEffectComparison {
            pair_id: context.pair_id,
            individual_id_sha256: context.individual_id_sha256,
            realized_effect_units: realized_effect,
            latent_predicted_effect_units: latent_effect,
            rollout_predicted_effect_units: rollout_effect,
            latent_absolute_error_units: latent_error,
            rollout_absolute_error_units: rollout_error,
            latent_sign_matches,
            rollout_sign_matches,
        });
    }

    let individual_count = u64::try_from(pairs.len())
        .map_err(|_| IndividualEffectError::ArithmeticOverflow)?;
    let rollout_minus_latent_absolute_error_sum_units =
        i128::try_from(rollout_absolute_error_sum)
            .and_then(|rollout| {
                i128::try_from(latent_absolute_error_sum)
                    .map(|latent| rollout - latent)
            })
            .map_err(|_| IndividualEffectError::ArithmeticOverflow)?;

    Ok(IndividualEffectRetentionCertificate {
        task_sha256: first.task_sha256,
        intervention_pair_sha256: first.intervention_pair_sha256,
        loss_scale_sha256: first.loss_scale_sha256,
        baseline_action_id: first.baseline_action_id.clone(),
        alternative_action_id: first.alternative_action_id.clone(),
        individual_count,
        realized_effect_sum_units: realized_sum,
        latent_predicted_effect_sum_units: latent_sum,
        rollout_predicted_effect_sum_units: rollout_sum,
        latent_effect_bias_sum_units: latent_bias_sum,
        rollout_effect_bias_sum_units: rollout_bias_sum,
        latent_absolute_error_sum_units: latent_absolute_error_sum,
        rollout_absolute_error_sum_units: rollout_absolute_error_sum,
        latent_sign_match_count,
        rollout_sign_match_count,
        latent_sign_accuracy_ppm: sign_accuracy_ppm(latent_sign_match_count, individual_count)?,
        rollout_sign_accuracy_ppm: sign_accuracy_ppm(rollout_sign_match_count, individual_count)?,
        rollout_minus_latent_absolute_error_sum_units,
        population_mean_effect_is_zero: realized_sum == 0,
        individual_effect_heterogeneity_present: individual_effects
            .first()
            .map(|first_effect| {
                individual_effects
                    .iter()
                    .skip(1)
                    .any(|effect| effect.realized_effect_units != first_effect.realized_effect_units)
            })
            .unwrap_or(false),
        comparisons: individual_effects,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> [u8; 32] {
        let mut value = [0_u8; 32];
        value[0] = seed;
        value[31] = seed.wrapping_add(1);
        value
    }

    fn context(pair_id: u64, individual: u8) -> IndividualEffectContext {
        IndividualEffectContext {
            pair_id,
            individual_id_sha256: digest(individual),
            factual_history_sha256: digest(individual.wrapping_add(10)),
            branch_point_state_sha256: digest(individual.wrapping_add(20)),
            exogenous_forcing_manifest_sha256: digest(individual.wrapping_add(30)),
            intervention_pair_sha256: digest(90),
            task_sha256: digest(91),
            loss_scale_sha256: digest(92),
            evaluator_profile_sha256: digest(93),
            evaluation_split_sha256: digest(94),
            model_configuration_sha256: digest(95),
            latent_representation_profile_sha256: digest(96),
            latent_state_sha256: digest(individual.wrapping_add(40)),
            latent_readout_profile_sha256: digest(97),
            rollout_profile_sha256: digest(98),
            pairing_receipt_sha256: digest(individual.wrapping_add(50)),
            pairing_verifier_profile_sha256: digest(99),
            baseline_action_id: "baseline".to_owned(),
            alternative_action_id: "alternative".to_owned(),
            paired_seed: 1234,
        }
    }

    fn branch(
        run_id: u64,
        action_id: &str,
        context: &IndividualEffectContext,
        realized: i64,
        latent: i64,
        rollout: i64,
    ) -> IndividualEffectBranch {
        IndividualEffectBranch {
            run_id,
            action_id: action_id.to_owned(),
            factual_history_sha256: context.factual_history_sha256,
            branch_point_state_sha256: context.branch_point_state_sha256,
            exogenous_forcing_manifest_sha256: context.exogenous_forcing_manifest_sha256,
            model_configuration_sha256: context.model_configuration_sha256,
            latent_representation_profile_sha256: context.latent_representation_profile_sha256,
            latent_state_sha256: context.latent_state_sha256,
            disposition: BranchDisposition::Applied {
                applied_at_us: run_id * 10,
                realized_utility_units: realized,
            },
            latent_predicted_utility_units: latent,
            rollout_predicted_utility_units: rollout,
        }
    }

    fn pair(
        pair_id: u64,
        individual: u8,
        realized_baseline: i64,
        realized_alternative: i64,
        latent_baseline: i64,
        latent_alternative: i64,
        rollout_baseline: i64,
        rollout_alternative: i64,
    ) -> IndividualEffectPair {
        let context = context(pair_id, individual);
        IndividualEffectPair {
            baseline: branch(
                pair_id * 2,
                "baseline",
                &context,
                realized_baseline,
                latent_baseline,
                rollout_baseline,
            ),
            alternative: branch(
                pair_id * 2 + 1,
                "alternative",
                &context,
                realized_alternative,
                latent_alternative,
                rollout_alternative,
            ),
            context,
        }
    }

    #[test]
    fn f8_mean_cancellation_preserves_individual_contrast() {
        let pairs = vec![
            pair(1, 1, 0, 10, 0, 10, 0, 0),
            pair(2, 2, 0, -10, 0, -10, 0, 0),
        ];
        let certificate = evaluate_individual_effect_retention(&pairs).unwrap();
        assert_eq!(certificate.individual_count, 2);
        assert_eq!(certificate.realized_effect_sum_units, 0);
        assert_eq!(certificate.rollout_predicted_effect_sum_units, 0);
        assert_eq!(certificate.latent_absolute_error_sum_units, 0);
        assert_eq!(certificate.rollout_absolute_error_sum_units, 20);
        assert_eq!(certificate.latent_sign_match_count, 2);
        assert_eq!(certificate.rollout_sign_match_count, 0);
        assert_eq!(certificate.latent_sign_accuracy_ppm, 1_000_000);
        assert_eq!(certificate.rollout_sign_accuracy_ppm, 0);
        assert!(certificate.population_mean_effect_is_zero);
        assert!(certificate.individual_effect_heterogeneity_present);
    }

    #[test]
    fn average_effect_accuracy_does_not_hide_individual_rollout_error() {
        let pairs = vec![
            pair(1, 1, 0, 10, 0, 10, 0, 0),
            pair(2, 2, 0, -10, 0, -10, 0, 0),
        ];
        let result = evaluate_individual_effect_retention(&pairs).unwrap();
        assert_eq!(result.rollout_effect_bias_sum_units, 0);
        assert_eq!(result.rollout_absolute_error_sum_units, 20);
        assert_eq!(result.rollout_minus_latent_absolute_error_sum_units, 20);
    }

    #[test]
    fn exact_latent_and_rollout_effects_are_separated() {
        let pairs = vec![pair(1, 1, -4, 7, -4, 7, -4, 7)];
        let result = evaluate_individual_effect_retention(&pairs).unwrap();
        assert_eq!(result.latent_absolute_error_sum_units, 0);
        assert_eq!(result.rollout_absolute_error_sum_units, 0);
        assert_eq!(result.latent_sign_accuracy_ppm, 1_000_000);
        assert_eq!(result.rollout_sign_accuracy_ppm, 1_000_000);
    }

    #[test]
    fn empty_pair_set_is_rejected() {
        assert_eq!(
            evaluate_individual_effect_retention(&[]),
            Err(IndividualEffectError::EmptyPairSet)
        );
    }

    #[test]
    fn duplicate_individual_is_rejected() {
        let pairs = vec![
            pair(1, 1, 0, 2, 0, 2, 0, 2),
            pair(2, 1, 0, 3, 0, 3, 0, 3),
        ];
        assert_eq!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::DuplicateIndividualId)
        );
    }

    #[test]
    fn duplicate_run_id_is_rejected() {
        let mut pairs = vec![
            pair(1, 1, 0, 2, 0, 2, 0, 2),
            pair(2, 2, 0, 3, 0, 3, 0, 3),
        ];
        pairs[1].baseline.run_id = pairs[0].alternative.run_id;
        assert!(matches!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::DuplicateRunId { .. })
        ));
    }

    #[test]
    fn missing_pairing_receipt_is_rejected() {
        let mut pairs = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        pairs[0].context.pairing_receipt_sha256 = [0; 32];
        assert_eq!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::MissingDigest {
                field: "pairing_receipt_sha256"
            })
        );
    }

    #[test]
    fn branch_point_mismatch_is_rejected() {
        let mut pairs = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        pairs[0].alternative.branch_point_state_sha256 = digest(200);
        assert!(matches!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::BranchProvenanceMismatch {
                field: "branch_point_state_sha256",
                ..
            })
        ));
    }

    #[test]
    fn exogenous_forcing_mismatch_is_rejected() {
        let mut pairs = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        pairs[0].alternative.exogenous_forcing_manifest_sha256 = digest(201);
        assert!(matches!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::BranchProvenanceMismatch {
                field: "exogenous_forcing_manifest_sha256",
                ..
            })
        ));
    }

    #[test]
    fn model_or_latent_identity_mismatch_is_rejected() {
        let mut model_mismatch = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        model_mismatch[0].alternative.model_configuration_sha256 = digest(202);
        assert!(matches!(
            evaluate_individual_effect_retention(&model_mismatch),
            Err(IndividualEffectError::BranchProvenanceMismatch {
                field: "model_configuration_sha256",
                ..
            })
        ));

        let mut latent_mismatch = vec![pair(2, 2, 0, 2, 0, 2, 0, 2)];
        latent_mismatch[0].alternative.latent_state_sha256 = digest(203);
        assert!(matches!(
            evaluate_individual_effect_retention(&latent_mismatch),
            Err(IndividualEffectError::BranchProvenanceMismatch {
                field: "latent_state_sha256",
                ..
            })
        ));
    }

    #[test]
    fn non_applied_branch_is_not_imputed_as_zero() {
        let mut pairs = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        pairs[0].alternative.disposition = BranchDisposition::NotApplied {
            terminal_at_us: 5_000,
            reason: NonApplicationReason::DeadlineMiss,
        };
        assert_eq!(
            evaluate_individual_effect_retention(&pairs),
            Err(IndividualEffectError::BranchNotApplied {
                pair_id: 1,
                action_id: "alternative".to_owned(),
                terminal_at_us: 5_000,
                reason: NonApplicationReason::DeadlineMiss,
            })
        );
    }

    #[test]
    fn incompatible_task_or_loss_scale_is_rejected() {
        let mut task_mismatch = vec![
            pair(1, 1, 0, 2, 0, 2, 0, 2),
            pair(2, 2, 0, 3, 0, 3, 0, 3),
        ];
        task_mismatch[1].context.task_sha256 = digest(222);
        assert!(matches!(
            evaluate_individual_effect_retention(&task_mismatch),
            Err(IndividualEffectError::SharedProfileMismatch { field: "task" })
        ));

        let mut loss_mismatch = vec![
            pair(1, 1, 0, 2, 0, 2, 0, 2),
            pair(2, 2, 0, 3, 0, 3, 0, 3),
        ];
        loss_mismatch[1].context.loss_scale_sha256 = digest(223);
        assert!(matches!(
            evaluate_individual_effect_retention(&loss_mismatch),
            Err(IndividualEffectError::SharedProfileMismatch { field: "loss scale" })
        ));
    }

    #[test]
    fn empty_or_identical_action_ids_are_rejected() {
        let mut empty = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        empty[0].context.alternative_action_id.clear();
        assert_eq!(
            evaluate_individual_effect_retention(&empty),
            Err(IndividualEffectError::EmptyActionId)
        );

        let mut identical = vec![pair(1, 1, 0, 2, 0, 2, 0, 2)];
        identical[0].context.alternative_action_id = "baseline".to_owned();
        assert_eq!(
            evaluate_individual_effect_retention(&identical),
            Err(IndividualEffectError::IdenticalActions)
        );
    }

    #[test]
    fn extreme_signed_utilities_preserve_exact_differences() {
        let pairs = vec![pair(
            1,
            1,
            i64::MIN,
            i64::MAX,
            i64::MIN,
            i64::MAX,
            i64::MIN,
            i64::MAX,
        )];
        let result = evaluate_individual_effect_retention(&pairs).unwrap();
        assert_eq!(result.realized_effect_sum_units, (i64::MAX as i128) - (i64::MIN as i128));
        assert_eq!(result.latent_absolute_error_sum_units, 0);
    }
}

//! Evaluator-side exact paired randomization audit for temporal action values.
//!
//! This package consumes evaluator records whose assignment receipts and
//! signatures have already been checked by an upstream verifier. It is not a
//! signature verifier, process boundary, or proof that randomization occurred.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::fmt;

const PPM: u128 = 1_000_000;
const AXIS_COUNT: u128 = 3;

pub const MIN_RANDOMIZED_PAIRS: usize = 2;
pub const MAX_EXACT_RANDOMIZED_PAIRS: usize = 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposureValues {
    pub wrong_belief_us: u64,
    pub abstention_us: u64,
    pub missing_output_us: u64,
}

impl ExposureValues {
    fn values(self) -> [u64; 3] {
        [
            self.wrong_belief_us,
            self.abstention_us,
            self.missing_output_us,
        ]
    }

    fn validate(self, episode_duration_us: u64) -> Result<(), AuditError> {
        let total = u128::from(self.wrong_belief_us)
            + u128::from(self.abstention_us)
            + u128::from(self.missing_output_us);
        if total > u128::from(episode_duration_us) {
            return Err(AuditError::ExposureExceedsEpisode {
                total_us: total,
                maximum_us: episode_duration_us,
            });
        }
        Ok(())
    }

    fn abstention(duration_us: u64) -> Self {
        Self {
            wrong_belief_us: 0,
            abstention_us: duration_us,
            missing_output_us: 0,
        }
    }

    fn missing_output(duration_us: u64) -> Self {
        Self {
            wrong_belief_us: 0,
            abstention_us: 0,
            missing_output_us: duration_us,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictedBenefit {
    /// Candidate-predicted control exposure minus treatment exposure.
    /// Positive values predict that treatment reduces exposure.
    pub wrong_belief_us: i64,
    pub abstention_us: i64,
    pub missing_output_us: i64,
}

impl PredictedBenefit {
    fn values(self) -> [i64; 3] {
        [
            self.wrong_belief_us,
            self.abstention_us,
            self.missing_output_us,
        ]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OutputFailure {
    Missing,
    Timeout,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArmOutcome {
    Completed(ExposureValues),
    Abstained,
    Failed(OutputFailure),
}

impl ArmOutcome {
    fn exposure(self, episode_duration_us: u64) -> ExposureValues {
        match self {
            Self::Completed(values) => values,
            Self::Abstained => ExposureValues::abstention(episode_duration_us),
            Self::Failed(_) => ExposureValues::missing_output(episode_duration_us),
        }
    }

    fn validate(self, episode_duration_us: u64) -> Result<(), AuditError> {
        match self {
            Self::Completed(values) => values.validate(episode_duration_us),
            Self::Abstained | Self::Failed(_) => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AssignmentScheme {
    /// Treatment is independently assigned to the first or second run by a
    /// fair coin for every pair. Receipts must be validated upstream.
    IndependentFairCoin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizedPair {
    pub pair_id: u64,
    pub first_run_id: u64,
    pub second_run_id: u64,
    pub episode_duration_us: u64,

    pub task_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub evaluator_sha256: [u8; 32],
    pub treatment_policy_sha256: [u8; 32],
    pub control_policy_sha256: [u8; 32],
    pub assignment_manifest_sha256: [u8; 32],

    pub first_context_sha256: [u8; 32],
    pub second_context_sha256: [u8; 32],
    pub first_source_family_sha256: [u8; 32],
    pub second_source_family_sha256: [u8; 32],

    /// Unique commitment to the independently verified assignment receipt.
    pub assignment_receipt_sha256: [u8; 32],
    pub assignment_scheme: AssignmentScheme,
    /// True when the first run received treatment; false when the second did.
    pub treatment_was_first: bool,
    /// Must be exactly 500,000 PPM for the exact sign-flip engine.
    pub treatment_probability_ppm: u32,

    pub first_outcome: ArmOutcome,
    pub second_outcome: ArmOutcome,
    pub predicted_benefit: PredictedBenefit,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizationAuditInput {
    /// Complete preregistered pair set, sorted by strictly increasing ID.
    pub preregistered_pair_ids: Vec<u64>,
    pub pairs: Vec<RandomizedPair>,
    pub familywise_alpha_ppm: u32,
    /// Includes the three exposure axes and every planned policy contrast.
    pub multiplicity_family_size: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExposureAxis {
    WrongBelief,
    Abstention,
    MissingOutput,
}

impl ExposureAxis {
    const ALL: [Self; 3] = [
        Self::WrongBelief,
        Self::Abstention,
        Self::MissingOutput,
    ];

    const fn index(self) -> usize {
        match self {
            Self::WrongBelief => 0,
            Self::Abstention => 1,
            Self::MissingOutput => 2,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AxisAudit {
    pub axis: ExposureAxis,
    pub pair_count: usize,
    pub predicted_benefit_total_us: i128,
    pub realized_benefit_total_us: i128,
    pub mean_predicted_benefit_us: i128,
    pub mean_realized_benefit_us: i128,
    /// Candidate prediction minus evaluator-realized benefit.
    pub mean_signed_prediction_error_us: i128,
    /// Rounded upward so integer reporting cannot understate MAE.
    pub mean_absolute_prediction_error_us_ceil: u128,
    pub wins: usize,
    pub losses: usize,
    pub ties: usize,
    pub exact_one_sided_p_numerator: u64,
    pub exact_two_sided_p_numerator: u64,
    pub exact_p_denominator: u64,
    pub bonferroni_adjusted_one_sided_p_upper_ppm: u32,
    pub familywise_significant: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RandomizationAuditReport {
    pub task_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub evaluator_sha256: [u8; 32],
    pub treatment_policy_sha256: [u8; 32],
    pub control_policy_sha256: [u8; 32],
    pub assignment_manifest_sha256: [u8; 32],
    pub pair_count: usize,
    pub treatment_first_count: usize,
    pub treatment_second_count: usize,
    pub familywise_alpha_ppm: u32,
    pub multiplicity_family_size: u32,
    pub axes: [AxisAudit; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AuditError {
    TooFewPairs { actual: usize, minimum: usize },
    TooManyPairs { actual: usize, maximum: usize },
    InvalidFamilywiseAlpha { alpha_ppm: u32 },
    InvalidMultiplicityFamilySize { actual: u32, minimum: u32 },
    EmptyPreregisteredManifest,
    UnsortedOrDuplicateManifest,
    ManifestMismatch,
    ZeroIdentifier { field: &'static str },
    ZeroDigest { field: &'static str, pair_id: u64 },
    DuplicateIdentifier { field: &'static str, value: u64 },
    DuplicateDigest { field: &'static str, pair_id: u64 },
    SharedDesignMismatch { field: &'static str, pair_id: u64 },
    PairContextMismatch { pair_id: u64 },
    InvalidEpisodeDuration { pair_id: u64 },
    UnsupportedAssignmentScheme { pair_id: u64 },
    InvalidTreatmentProbability { pair_id: u64, actual_ppm: u32 },
    ExposureExceedsEpisode { total_us: u128, maximum_us: u64 },
    ArithmeticOverflow,
}

impl fmt::Display for AuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewPairs { actual, minimum } => {
                write!(f, "randomization audit has {actual} pairs; minimum is {minimum}")
            }
            Self::TooManyPairs { actual, maximum } => {
                write!(f, "randomization audit has {actual} pairs; exact maximum is {maximum}")
            }
            Self::InvalidFamilywiseAlpha { alpha_ppm } => {
                write!(f, "familywise alpha {alpha_ppm} PPM must be in 1..=500000")
            }
            Self::InvalidMultiplicityFamilySize { actual, minimum } => {
                write!(f, "multiplicity family size {actual} is below minimum {minimum}")
            }
            Self::EmptyPreregisteredManifest => write!(f, "preregistered pair manifest is empty"),
            Self::UnsortedOrDuplicateManifest => {
                write!(f, "preregistered pair IDs must be nonzero, unique and strictly increasing")
            }
            Self::ManifestMismatch => {
                write!(f, "observed pair IDs do not exactly match the preregistered manifest")
            }
            Self::ZeroIdentifier { field } => write!(f, "{field} must be nonzero"),
            Self::ZeroDigest { field, pair_id } => {
                write!(f, "{field} digest is zero for pair {pair_id}")
            }
            Self::DuplicateIdentifier { field, value } => {
                write!(f, "duplicate {field} identifier {value}")
            }
            Self::DuplicateDigest { field, pair_id } => {
                write!(f, "duplicate {field} digest at pair {pair_id}")
            }
            Self::SharedDesignMismatch { field, pair_id } => {
                write!(f, "{field} differs from the frozen audit design at pair {pair_id}")
            }
            Self::PairContextMismatch { pair_id } => {
                write!(f, "first and second run contexts or source families differ at pair {pair_id}")
            }
            Self::InvalidEpisodeDuration { pair_id } => {
                write!(f, "episode duration is zero for pair {pair_id}")
            }
            Self::UnsupportedAssignmentScheme { pair_id } => {
                write!(f, "assignment scheme is not independent fair-coin randomization at pair {pair_id}")
            }
            Self::InvalidTreatmentProbability { pair_id, actual_ppm } => {
                write!(f, "treatment probability {actual_ppm} PPM is not 500000 at pair {pair_id}")
            }
            Self::ExposureExceedsEpisode { total_us, maximum_us } => {
                write!(f, "exposure total {total_us}us exceeds episode duration {maximum_us}us")
            }
            Self::ArithmeticOverflow => write!(f, "arithmetic overflow during randomization audit"),
        }
    }
}

impl std::error::Error for AuditError {}

pub fn audit_randomized_interventions(
    input: &RandomizationAuditInput,
) -> Result<RandomizationAuditReport, AuditError> {
    validate_input(input)?;

    let first = &input.pairs[0];
    let mut treatment_first_count = 0usize;

    let mut predicted_totals = [0_i128; 3];
    let mut realized_totals = [0_i128; 3];
    let mut absolute_error_totals = [0_u128; 3];
    let mut wins = [0usize; 3];
    let mut losses = [0usize; 3];
    let mut ties = [0usize; 3];
    let mut magnitudes: [Vec<u128>; 3] = std::array::from_fn(|_| Vec::with_capacity(input.pairs.len()));

    for pair in &input.pairs {
        if pair.treatment_was_first {
            treatment_first_count += 1;
        }

        let first_exposure = pair.first_outcome.exposure(pair.episode_duration_us);
        let second_exposure = pair.second_outcome.exposure(pair.episode_duration_us);
        let (treatment, control) = if pair.treatment_was_first {
            (first_exposure, second_exposure)
        } else {
            (second_exposure, first_exposure)
        };

        let treatment_values = treatment.values();
        let control_values = control.values();
        let predicted_values = pair.predicted_benefit.values();

        for axis_index in 0..3 {
            let realized = i128::from(control_values[axis_index])
                - i128::from(treatment_values[axis_index]);
            let predicted = i128::from(predicted_values[axis_index]);
            let prediction_error = predicted - realized;

            predicted_totals[axis_index] = predicted_totals[axis_index]
                .checked_add(predicted)
                .ok_or(AuditError::ArithmeticOverflow)?;
            realized_totals[axis_index] = realized_totals[axis_index]
                .checked_add(realized)
                .ok_or(AuditError::ArithmeticOverflow)?;
            absolute_error_totals[axis_index] = absolute_error_totals[axis_index]
                .checked_add(prediction_error.unsigned_abs())
                .ok_or(AuditError::ArithmeticOverflow)?;

            match realized.cmp(&0) {
                std::cmp::Ordering::Greater => wins[axis_index] += 1,
                std::cmp::Ordering::Less => losses[axis_index] += 1,
                std::cmp::Ordering::Equal => ties[axis_index] += 1,
            }
            magnitudes[axis_index].push(realized.unsigned_abs());
        }
    }

    let pair_count = input.pairs.len();
    let pair_count_i128 = i128::try_from(pair_count).map_err(|_| AuditError::ArithmeticOverflow)?;
    let axes = std::array::from_fn(|index| {
        let (one_sided, two_sided, denominator) =
            exact_sign_flip_p_values(&magnitudes[index], realized_totals[index]);
        let adjusted = bonferroni_adjusted_ppm(
            one_sided,
            denominator,
            input.multiplicity_family_size,
        );
        AxisAudit {
            axis: ExposureAxis::ALL[index],
            pair_count,
            predicted_benefit_total_us: predicted_totals[index],
            realized_benefit_total_us: realized_totals[index],
            mean_predicted_benefit_us: predicted_totals[index] / pair_count_i128,
            mean_realized_benefit_us: realized_totals[index] / pair_count_i128,
            mean_signed_prediction_error_us: (predicted_totals[index] - realized_totals[index])
                / pair_count_i128,
            mean_absolute_prediction_error_us_ceil: absolute_error_totals[index]
                .saturating_add(pair_count as u128 - 1)
                / pair_count as u128,
            wins: wins[index],
            losses: losses[index],
            ties: ties[index],
            exact_one_sided_p_numerator: one_sided,
            exact_two_sided_p_numerator: two_sided,
            exact_p_denominator: denominator,
            bonferroni_adjusted_one_sided_p_upper_ppm: adjusted,
            familywise_significant: bonferroni_familywise_significant(
                one_sided,
                denominator,
                input.familywise_alpha_ppm,
                input.multiplicity_family_size,
            ),
        }
    });

    Ok(RandomizationAuditReport {
        task_sha256: first.task_sha256,
        model_sha256: first.model_sha256,
        evaluator_sha256: first.evaluator_sha256,
        treatment_policy_sha256: first.treatment_policy_sha256,
        control_policy_sha256: first.control_policy_sha256,
        assignment_manifest_sha256: first.assignment_manifest_sha256,
        pair_count,
        treatment_first_count,
        treatment_second_count: pair_count - treatment_first_count,
        familywise_alpha_ppm: input.familywise_alpha_ppm,
        multiplicity_family_size: input.multiplicity_family_size,
        axes,
    })
}

fn validate_input(input: &RandomizationAuditInput) -> Result<(), AuditError> {
    let pair_count = input.pairs.len();
    if pair_count < MIN_RANDOMIZED_PAIRS {
        return Err(AuditError::TooFewPairs {
            actual: pair_count,
            minimum: MIN_RANDOMIZED_PAIRS,
        });
    }
    if pair_count > MAX_EXACT_RANDOMIZED_PAIRS {
        return Err(AuditError::TooManyPairs {
            actual: pair_count,
            maximum: MAX_EXACT_RANDOMIZED_PAIRS,
        });
    }
    if input.familywise_alpha_ppm == 0 || input.familywise_alpha_ppm > 500_000 {
        return Err(AuditError::InvalidFamilywiseAlpha {
            alpha_ppm: input.familywise_alpha_ppm,
        });
    }
    if input.multiplicity_family_size < AXIS_COUNT as u32 {
        return Err(AuditError::InvalidMultiplicityFamilySize {
            actual: input.multiplicity_family_size,
            minimum: AXIS_COUNT as u32,
        });
    }
    if input.preregistered_pair_ids.is_empty() {
        return Err(AuditError::EmptyPreregisteredManifest);
    }
    if input.preregistered_pair_ids.iter().any(|id| *id == 0)
        || input
            .preregistered_pair_ids
            .windows(2)
            .any(|window| window[0] >= window[1])
    {
        return Err(AuditError::UnsortedOrDuplicateManifest);
    }

    let expected: BTreeSet<u64> = input.preregistered_pair_ids.iter().copied().collect();
    let mut observed = BTreeSet::new();
    for pair in &input.pairs {
        if !observed.insert(pair.pair_id) {
            return Err(AuditError::DuplicateIdentifier {
                field: "pair",
                value: pair.pair_id,
            });
        }
    }
    if expected != observed || expected.len() != pair_count {
        return Err(AuditError::ManifestMismatch);
    }

    let first = &input.pairs[0];
    validate_common_digest("task", first.task_sha256, 0)?;
    validate_common_digest("model", first.model_sha256, 0)?;
    validate_common_digest("evaluator", first.evaluator_sha256, 0)?;
    validate_common_digest("treatment policy", first.treatment_policy_sha256, 0)?;
    validate_common_digest("control policy", first.control_policy_sha256, 0)?;
    validate_common_digest("assignment manifest", first.assignment_manifest_sha256, 0)?;

    let common = (
        first.task_sha256,
        first.model_sha256,
        first.evaluator_sha256,
        first.treatment_policy_sha256,
        first.control_policy_sha256,
        first.assignment_manifest_sha256,
    );

    let mut run_ids = BTreeSet::new();
    let mut context_digests = BTreeSet::new();
    let mut assignment_receipts = BTreeSet::new();

    for pair in &input.pairs {
        if pair.pair_id == 0 {
            return Err(AuditError::ZeroIdentifier { field: "pair" });
        }
        if pair.first_run_id == 0 || pair.second_run_id == 0 {
            return Err(AuditError::ZeroIdentifier { field: "run" });
        }
        if pair.first_run_id == pair.second_run_id {
            return Err(AuditError::DuplicateIdentifier {
                field: "run within pair",
                value: pair.first_run_id,
            });
        }
        for run_id in [pair.first_run_id, pair.second_run_id] {
            if !run_ids.insert(run_id) {
                return Err(AuditError::DuplicateIdentifier {
                    field: "run",
                    value: run_id,
                });
            }
        }
        if pair.episode_duration_us == 0 {
            return Err(AuditError::InvalidEpisodeDuration { pair_id: pair.pair_id });
        }

        validate_pair_digest("first context", pair.first_context_sha256, pair.pair_id)?;
        validate_pair_digest("second context", pair.second_context_sha256, pair.pair_id)?;
        validate_pair_digest(
            "first source family",
            pair.first_source_family_sha256,
            pair.pair_id,
        )?;
        validate_pair_digest(
            "second source family",
            pair.second_source_family_sha256,
            pair.pair_id,
        )?;
        validate_pair_digest(
            "assignment receipt",
            pair.assignment_receipt_sha256,
            pair.pair_id,
        )?;

        if pair.first_context_sha256 != pair.second_context_sha256
            || pair.first_source_family_sha256 != pair.second_source_family_sha256
        {
            return Err(AuditError::PairContextMismatch { pair_id: pair.pair_id });
        }
        if !context_digests.insert(pair.first_context_sha256) {
            return Err(AuditError::DuplicateDigest {
                field: "paired context",
                pair_id: pair.pair_id,
            });
        }
        if !assignment_receipts.insert(pair.assignment_receipt_sha256) {
            return Err(AuditError::DuplicateDigest {
                field: "assignment receipt",
                pair_id: pair.pair_id,
            });
        }

        let actual_common = (
            pair.task_sha256,
            pair.model_sha256,
            pair.evaluator_sha256,
            pair.treatment_policy_sha256,
            pair.control_policy_sha256,
            pair.assignment_manifest_sha256,
        );
        if actual_common != common {
            return Err(AuditError::SharedDesignMismatch {
                field: "task/model/evaluator/policy/assignment manifest",
                pair_id: pair.pair_id,
            });
        }
        if pair.assignment_scheme != AssignmentScheme::IndependentFairCoin {
            return Err(AuditError::UnsupportedAssignmentScheme { pair_id: pair.pair_id });
        }
        if pair.treatment_probability_ppm != 500_000 {
            return Err(AuditError::InvalidTreatmentProbability {
                pair_id: pair.pair_id,
                actual_ppm: pair.treatment_probability_ppm,
            });
        }

        pair.first_outcome.validate(pair.episode_duration_us)?;
        pair.second_outcome.validate(pair.episode_duration_us)?;
    }

    Ok(())
}

fn validate_common_digest(
    field: &'static str,
    digest: [u8; 32],
    pair_id: u64,
) -> Result<(), AuditError> {
    if digest.iter().all(|byte| *byte == 0) {
        return Err(AuditError::ZeroDigest { field, pair_id });
    }
    Ok(())
}

fn validate_pair_digest(
    field: &'static str,
    digest: [u8; 32],
    pair_id: u64,
) -> Result<(), AuditError> {
    if digest.iter().all(|byte| *byte == 0) {
        return Err(AuditError::ZeroDigest { field, pair_id });
    }
    Ok(())
}

fn exact_sign_flip_p_values(magnitudes: &[u128], observed_sum: i128) -> (u64, u64, u64) {
    let pair_count = magnitudes.len();
    let denominator = 1_u64 << pair_count;
    let mut signed_sum: i128 = magnitudes.iter().map(|value| *value as i128).sum();
    let observed_absolute = observed_sum.unsigned_abs();

    let mut previous_gray = 0_u64;
    let mut one_sided = 0_u64;
    let mut two_sided = 0_u64;

    for counter in 0..denominator {
        let gray = counter ^ (counter >> 1);
        if counter > 0 {
            let changed_bit = (gray ^ previous_gray).trailing_zeros() as usize;
            let magnitude = magnitudes[changed_bit] as i128;
            let delta = magnitude * 2;
            if (gray & (1_u64 << changed_bit)) != 0 {
                signed_sum -= delta;
            } else {
                signed_sum += delta;
            }
        }

        if signed_sum >= observed_sum {
            one_sided += 1;
        }
        if signed_sum.unsigned_abs() >= observed_absolute {
            two_sided += 1;
        }
        previous_gray = gray;
    }

    (one_sided, two_sided, denominator)
}

fn bonferroni_adjusted_ppm(
    p_numerator: u64,
    denominator: u64,
    family_size: u32,
) -> u32 {
    let scaled = u128::from(p_numerator)
        * u128::from(family_size)
        * PPM;
    let denominator = u128::from(denominator);
    let rounded_up = scaled.saturating_add(denominator - 1) / denominator;
    rounded_up.min(PPM) as u32
}

fn bonferroni_familywise_significant(
    p_numerator: u64,
    denominator: u64,
    alpha_ppm: u32,
    family_size: u32,
) -> bool {
    u128::from(p_numerator) * u128::from(family_size) * PPM
        <= u128::from(alpha_ppm) * u128::from(denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn completed(wrong: u64, abstention: u64, missing: u64) -> ArmOutcome {
        ArmOutcome::Completed(ExposureValues {
            wrong_belief_us: wrong,
            abstention_us: abstention,
            missing_output_us: missing,
        })
    }

    fn pair(
        id: u64,
        treatment_first: bool,
        control: ArmOutcome,
        treatment: ArmOutcome,
        predicted: PredictedBenefit,
    ) -> RandomizedPair {
        let (first_outcome, second_outcome) = if treatment_first {
            (treatment, control)
        } else {
            (control, treatment)
        };
        RandomizedPair {
            pair_id: id,
            first_run_id: id * 2,
            second_run_id: id * 2 + 1,
            episode_duration_us: 100,
            task_sha256: digest(1),
            model_sha256: digest(2),
            evaluator_sha256: digest(3),
            treatment_policy_sha256: digest(4),
            control_policy_sha256: digest(5),
            assignment_manifest_sha256: digest(6),
            first_context_sha256: digest((20 + id) as u8),
            second_context_sha256: digest((20 + id) as u8),
            first_source_family_sha256: digest(9),
            second_source_family_sha256: digest(9),
            assignment_receipt_sha256: digest((80 + id) as u8),
            assignment_scheme: AssignmentScheme::IndependentFairCoin,
            treatment_was_first: treatment_first,
            treatment_probability_ppm: 500_000,
            first_outcome,
            second_outcome,
            predicted_benefit: predicted,
        }
    }

    fn input(pairs: Vec<RandomizedPair>) -> RandomizationAuditInput {
        let mut ids: Vec<u64> = pairs.iter().map(|pair| pair.pair_id).collect();
        ids.sort_unstable();
        RandomizationAuditInput {
            preregistered_pair_ids: ids,
            pairs,
            familywise_alpha_ppm: 50_000,
            multiplicity_family_size: 3,
        }
    }

    #[test]
    fn strong_consistent_benefit_passes_familywise_exact_test() {
        let pairs = (1..=7)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(100, 0, 0),
                    completed(0, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 90,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        let wrong = &report.axes[0];
        assert_eq!(wrong.mean_realized_benefit_us, 100);
        assert_eq!(wrong.exact_one_sided_p_numerator, 1);
        assert_eq!(wrong.exact_p_denominator, 128);
        assert!(wrong.familywise_significant);
        assert_eq!(wrong.wins, 7);
        assert_eq!(wrong.losses, 0);
    }

    #[test]
    fn zero_effect_has_no_evidence_of_positive_benefit() {
        let pairs = (1..=5)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(50, 0, 0),
                    completed(50, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 0,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        assert_eq!(report.axes[0].mean_realized_benefit_us, 0);
        assert_eq!(report.axes[0].exact_one_sided_p_numerator, 32);
        assert_eq!(report.axes[0].exact_p_denominator, 32);
        assert!(!report.axes[0].familywise_significant);
    }

    #[test]
    fn harmful_treatment_is_not_mistaken_for_a_positive_effect() {
        let pairs = (1..=4)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(10, 0, 0),
                    completed(20, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 25,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        assert_eq!(report.axes[0].mean_realized_benefit_us, -10);
        assert!(!report.axes[0].familywise_significant);
    }

    #[test]
    fn missing_treatment_output_is_full_horizon_missing_exposure() {
        let pairs = vec![
            pair(
                1,
                true,
                completed(0, 0, 0),
                ArmOutcome::Failed(OutputFailure::Timeout),
                PredictedBenefit {
                    wrong_belief_us: 0,
                    abstention_us: 0,
                    missing_output_us: 80,
                },
            ),
            pair(
                2,
                false,
                completed(0, 0, 0),
                ArmOutcome::Failed(OutputFailure::Missing),
                PredictedBenefit {
                    wrong_belief_us: 0,
                    abstention_us: 0,
                    missing_output_us: 80,
                },
            ),
        ];
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        assert_eq!(report.pair_count, 2);
        assert_eq!(report.axes[2].mean_realized_benefit_us, -100);
        assert_eq!(report.axes[2].losses, 2);
    }

    #[test]
    fn abstention_is_separate_from_missing_output() {
        let pairs = vec![
            pair(
                1,
                true,
                completed(0, 0, 0),
                ArmOutcome::Abstained,
                PredictedBenefit {
                    wrong_belief_us: 0,
                    abstention_us: -100,
                    missing_output_us: 0,
                },
            ),
            pair(
                2,
                false,
                completed(0, 0, 0),
                ArmOutcome::Abstained,
                PredictedBenefit {
                    wrong_belief_us: 0,
                    abstention_us: -100,
                    missing_output_us: 0,
                },
            ),
        ];
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        assert_eq!(report.axes[1].mean_realized_benefit_us, -100);
        assert_eq!(report.axes[2].mean_realized_benefit_us, 0);
    }

    #[test]
    fn prediction_error_is_not_confused_with_treatment_effect() {
        let pairs = (1..=2)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(30, 0, 0),
                    completed(10, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 40,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        let report = audit_randomized_interventions(&input(pairs)).unwrap();
        assert_eq!(report.axes[0].mean_realized_benefit_us, 20);
        assert_eq!(report.axes[0].mean_predicted_benefit_us, 40);
        assert_eq!(report.axes[0].mean_signed_prediction_error_us, 20);
        assert_eq!(report.axes[0].mean_absolute_prediction_error_us_ceil, 20);
    }

    #[test]
    fn multiplicity_family_size_controls_the_confirmatory_decision() {
        let pairs = (1..=5)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(100, 0, 0),
                    completed(0, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 100,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        let mut study = input(pairs);
        study.multiplicity_family_size = 4;
        let report = audit_randomized_interventions(&study).unwrap();
        assert!(!report.axes[0].familywise_significant);
        assert!(report.axes[0].bonferroni_adjusted_one_sided_p_upper_ppm > 50_000);
    }

    #[test]
    fn duplicate_run_ids_are_rejected() {
        let mut pairs = vec![
            pair(1, true, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        pairs[1].first_run_id = pairs[0].first_run_id;
        let study = input(pairs);
        assert!(matches!(
            audit_randomized_interventions(&study),
            Err(AuditError::DuplicateIdentifier { field: "run", .. })
        ));
    }

    #[test]
    fn arm_context_mismatch_is_rejected() {
        let mut pairs = vec![
            pair(1, true, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        pairs[0].second_context_sha256 = digest(250);
        let study = input(pairs);
        assert!(matches!(
            audit_randomized_interventions(&study),
            Err(AuditError::PairContextMismatch { pair_id: 1 })
        ));
    }

    #[test]
    fn non_fair_assignment_probability_is_rejected() {
        let pairs = vec![
            pair(1, true, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        let mut study = input(pairs);
        study.pairs[1].treatment_probability_ppm = 250_000;
        assert!(matches!(
            audit_randomized_interventions(&study),
            Err(AuditError::InvalidTreatmentProbability { pair_id: 2, .. })
        ));
    }

    #[test]
    fn overallocated_exposure_is_rejected() {
        let pairs = vec![
            pair(1, true, completed(80, 30, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        let study = input(pairs);
        assert!(matches!(
            audit_randomized_interventions(&study),
            Err(AuditError::ExposureExceedsEpisode { .. })
        ));
    }

    #[test]
    fn missing_preregistered_pair_fails_closed() {
        let pairs = vec![
            pair(1, true, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        let mut study = input(pairs);
        study.preregistered_pair_ids.push(3);
        assert_eq!(
            audit_randomized_interventions(&study),
            Err(AuditError::ManifestMismatch)
        );
    }

    #[test]
    fn zero_assignment_receipt_digest_is_rejected() {
        let mut pairs = vec![
            pair(1, true, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
            pair(2, false, completed(100, 0, 0), completed(0, 0, 0), PredictedBenefit {
                wrong_belief_us: 1, abstention_us: 0, missing_output_us: 0,
            }),
        ];
        pairs[0].assignment_receipt_sha256 = [0; 32];
        let study = input(pairs);
        assert!(matches!(
            audit_randomized_interventions(&study),
            Err(AuditError::ZeroDigest { field: "assignment receipt", pair_id: 1 })
        ));
    }

    #[test]
    fn exact_mode_rejects_more_than_twenty_pairs_without_sampling() {
        let pairs = (1..=21)
            .map(|id| {
                pair(
                    id,
                    id % 2 == 0,
                    completed(100, 0, 0),
                    completed(0, 0, 0),
                    PredictedBenefit {
                        wrong_belief_us: 100,
                        abstention_us: 0,
                        missing_output_us: 0,
                    },
                )
            })
            .collect();
        assert!(matches!(
            audit_randomized_interventions(&input(pairs)),
            Err(AuditError::TooManyPairs { actual: 21, maximum: 20 })
        ));
    }

}

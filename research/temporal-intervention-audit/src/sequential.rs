//! Anytime-valid paired randomization evidence for temporal action effects.
//!
//! This process is valid for repeated looks under the declared design:
//! independently randomized fair-coin treatment orientation within every pair,
//! Fisher's sharp no-effect null, and no interference between randomization
//! units. It is not valid for blocked/fixed-count or clustered assignment
//! unless the actual assignment mechanism is represented by a different test.

use std::collections::BTreeSet;
use std::fmt;

use crate::{
    ArmOutcome, AssignmentScheme, OutputFailure, RandomizedPair,
};

const PPM: u128 = 1_000_000;
const AXIS_COUNT: u32 = 3;
pub const MAX_ANYTIME_PAIRS: u32 = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnytimeAuditConfig {
    pub familywise_alpha_ppm: u32,
    /// Includes all three exposure axes and every predeclared policy contrast.
    pub comparison_family_size: u32,
    /// Arithmetic representation is exact through 64 nonzero pair signs.
    pub maximum_pairs: u32,
    pub task_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub evaluator_sha256: [u8; 32],
    pub treatment_policy_sha256: [u8; 32],
    pub control_policy_sha256: [u8; 32],
    pub assignment_manifest_sha256: [u8; 32],
}

impl AnytimeAuditConfig {
    pub fn validate(&self) -> Result<(), AnytimeAuditError> {
        if self.familywise_alpha_ppm == 0 || self.familywise_alpha_ppm > 500_000 {
            return Err(AnytimeAuditError::InvalidFamilywiseAlpha {
                actual_ppm: self.familywise_alpha_ppm,
            });
        }
        if self.comparison_family_size < AXIS_COUNT {
            return Err(AnytimeAuditError::InvalidComparisonFamilySize {
                actual: self.comparison_family_size,
                minimum: AXIS_COUNT,
            });
        }
        if self.maximum_pairs == 0 || self.maximum_pairs > MAX_ANYTIME_PAIRS {
            return Err(AnytimeAuditError::InvalidMaximumPairs {
                actual: self.maximum_pairs,
                maximum: MAX_ANYTIME_PAIRS,
            });
        }
        for (field, digest) in [
            ("task", self.task_sha256),
            ("model", self.model_sha256),
            ("evaluator", self.evaluator_sha256),
            ("treatment policy", self.treatment_policy_sha256),
            ("control policy", self.control_policy_sha256),
            ("assignment manifest", self.assignment_manifest_sha256),
        ] {
            if digest == [0; 32] {
                return Err(AnytimeAuditError::ZeroConfigDigest { field });
            }
        }
        if self.treatment_policy_sha256 == self.control_policy_sha256 {
            return Err(AnytimeAuditError::IdenticalPolicyDigests);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AnytimeExposureAxis {
    WrongBelief,
    Abstention,
    MissingOutput,
}

impl AnytimeExposureAxis {
    const ALL: [Self; 3] = [
        Self::WrongBelief,
        Self::Abstention,
        Self::MissingOutput,
    ];
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnytimeAxisReport {
    pub axis: AnytimeExposureAxis,
    pub pair_count: u32,
    pub positive_pairs: u32,
    pub negative_pairs: u32,
    pub tied_pairs: u32,
    /// Exact e-value is numerator / denominator.
    pub e_value_numerator: u128,
    pub e_value_denominator: u128,
    /// The comparison crosses when E >= comparison_family_size / alpha.
    pub threshold_numerator: u128,
    pub threshold_denominator: u32,
    pub crossed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnytimeAuditReport {
    pub task_sha256: [u8; 32],
    pub model_sha256: [u8; 32],
    pub evaluator_sha256: [u8; 32],
    pub treatment_policy_sha256: [u8; 32],
    pub control_policy_sha256: [u8; 32],
    pub assignment_manifest_sha256: [u8; 32],
    pub familywise_alpha_ppm: u32,
    pub comparison_family_size: u32,
    pub observed_pairs: u32,
    pub latest_sequence: u64,
    pub axes: [AnytimeAxisReport; 3],
    /// Bonferroni + Ville: any axis/declared comparison may cross at this look.
    pub familywise_crossed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnytimeAuditError {
    InvalidFamilywiseAlpha { actual_ppm: u32 },
    InvalidComparisonFamilySize { actual: u32, minimum: u32 },
    InvalidMaximumPairs { actual: u32, maximum: u32 },
    ZeroConfigDigest { field: &'static str },
    IdenticalPolicyDigests,
    PairLimitReached { maximum: u32 },
    NonSequentialLook { expected: u64, actual: u64 },
    ZeroIdentifier { field: &'static str },
    DuplicateIdentifier { field: &'static str, value: u64 },
    ZeroDigest { field: &'static str },
    DuplicateDigest { field: &'static str },
    DesignDigestMismatch { field: &'static str },
    PairContextMismatch,
    DuplicatePairContext,
    InvalidEpisodeDuration,
    UnsupportedAssignmentScheme,
    InvalidTreatmentProbability { actual_ppm: u32 },
    ExposureExceedsEpisode { measured_us: u128, maximum_us: u64 },
    ArithmeticOverflow,
}

impl fmt::Display for AnytimeAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFamilywiseAlpha { actual_ppm } =>
                write!(f, "familywise alpha {actual_ppm} PPM must be in 1..=500000"),
            Self::InvalidComparisonFamilySize { actual, minimum } =>
                write!(f, "comparison family size {actual} is below minimum {minimum}"),
            Self::InvalidMaximumPairs { actual, maximum } =>
                write!(f, "maximum pair count {actual} must be in 1..={maximum}"),
            Self::ZeroConfigDigest { field } => write!(f, "{field} digest is zero in anytime audit config"),
            Self::IdenticalPolicyDigests => write!(f, "treatment and control policy digests must differ"),
            Self::PairLimitReached { maximum } => write!(f, "anytime audit reached its configured maximum of {maximum} pairs"),
            Self::NonSequentialLook { expected, actual } =>
                write!(f, "sequential look must be {expected}, received {actual}"),
            Self::ZeroIdentifier { field } => write!(f, "{field} identifier must be nonzero"),
            Self::DuplicateIdentifier { field, value } => write!(f, "duplicate {field} identifier {value}"),
            Self::ZeroDigest { field } => write!(f, "{field} digest is zero"),
            Self::DuplicateDigest { field } => write!(f, "duplicate {field} digest"),
            Self::DesignDigestMismatch { field } => write!(f, "{field} digest differs from the frozen anytime audit config"),
            Self::PairContextMismatch => write!(f, "the two runs in a randomized pair do not share context and source-family identity"),
            Self::DuplicatePairContext => write!(f, "paired context digest was reused across randomized pairs"),
            Self::InvalidEpisodeDuration => write!(f, "episode duration must be nonzero"),
            Self::UnsupportedAssignmentScheme => write!(f, "anytime sign process requires independent fair-coin assignment within each pair"),
            Self::InvalidTreatmentProbability { actual_ppm } => write!(f, "treatment probability {actual_ppm} PPM is not 500000"),
            Self::ExposureExceedsEpisode { measured_us, maximum_us } =>
                write!(f, "measured exposure {measured_us}us exceeds episode horizon {maximum_us}us"),
            Self::ArithmeticOverflow => write!(f, "anytime evidence arithmetic overflowed"),
        }
    }
}

impl std::error::Error for AnytimeAuditError {}

/// Incremental exact-rational betting process for one frozen treatment/control
/// contrast. Each nonzero paired sign contributes a factor 3/2 when beneficial
/// and 1/2 when harmful; a tie contributes 1.
///
/// Under the Fisher sharp null and the declared independent fair-coin design,
/// nonzero signs are conditionally symmetric. The process is therefore a
/// nonnegative martingale. Ville's inequality controls crossing at any
/// stopping time. Magnitudes are deliberately ignored; this is a transparent
/// conservative baseline, not a power-optimal betting strategy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnytimeAuditState {
    config: AnytimeAuditConfig,
    observed_pairs: u32,
    positive: [u32; 3],
    negative: [u32; 3],
    ties: [u32; 3],
    pair_ids: BTreeSet<u64>,
    run_ids: BTreeSet<u64>,
    pair_contexts: BTreeSet<[u8; 32]>,
    assignment_receipts: BTreeSet<[u8; 32]>,
}

impl AnytimeAuditState {
    pub fn new(config: AnytimeAuditConfig) -> Result<Self, AnytimeAuditError> {
        config.validate()?;
        Ok(Self {
            config,
            observed_pairs: 0,
            positive: [0; 3],
            negative: [0; 3],
            ties: [0; 3],
            pair_ids: BTreeSet::new(),
            run_ids: BTreeSet::new(),
            pair_contexts: BTreeSet::new(),
            assignment_receipts: BTreeSet::new(),
        })
    }

    /// Append exactly the next sequential pair. Validation is completed before
    /// any state changes, so an invalid pair cannot partially advance the
    /// process. The caller must persist this state/receipt append-only; this
    /// in-memory crate does not protect against storage rollback.
    pub fn observe(
        &mut self,
        sequence: u64,
        pair: &RandomizedPair,
    ) -> Result<AnytimeAuditReport, AnytimeAuditError> {
        if self.observed_pairs >= self.config.maximum_pairs {
            return Err(AnytimeAuditError::PairLimitReached {
                maximum: self.config.maximum_pairs,
            });
        }
        let expected_sequence = u64::from(self.observed_pairs) + 1;
        if sequence != expected_sequence {
            return Err(AnytimeAuditError::NonSequentialLook {
                expected: expected_sequence,
                actual: sequence,
            });
        }

        validate_pair(&self.config, pair, &self.pair_ids, &self.run_ids,
            &self.pair_contexts, &self.assignment_receipts)?;

        let first = outcome_exposure(pair.first_outcome, pair.episode_duration_us);
        let second = outcome_exposure(pair.second_outcome, pair.episode_duration_us);
        let (treatment, control) = if pair.treatment_was_first {
            (first, second)
        } else {
            (second, first)
        };

        let benefits = [
            i128::from(control[0]) - i128::from(treatment[0]),
            i128::from(control[1]) - i128::from(treatment[1]),
            i128::from(control[2]) - i128::from(treatment[2]),
        ];

        // No fallible operation remains after this point.
        self.pair_ids.insert(pair.pair_id);
        self.run_ids.insert(pair.first_run_id);
        self.run_ids.insert(pair.second_run_id);
        self.pair_contexts.insert(pair.first_context_sha256);
        self.assignment_receipts.insert(pair.assignment_receipt_sha256);
        self.observed_pairs += 1;

        for (index, benefit) in benefits.into_iter().enumerate() {
            if benefit > 0 {
                self.positive[index] += 1;
            } else if benefit < 0 {
                self.negative[index] += 1;
            } else {
                self.ties[index] += 1;
            }
        }
        self.report()
    }

    pub fn report(&self) -> Result<AnytimeAuditReport, AnytimeAuditError> {
        let axes = std::array::try_from_fn(|index| {
            let wins = self.positive[index];
            let losses = self.negative[index];
            let tied = self.ties[index];
            let non_ties = wins + losses;
            let numerator = pow3(wins)?;
            let denominator = 1_u128
                .checked_shl(non_ties)
                .ok_or(AnytimeAuditError::ArithmeticOverflow)?;
            let left = numerator
                .checked_mul(u128::from(self.config.familywise_alpha_ppm))
                .ok_or(AnytimeAuditError::ArithmeticOverflow)?;
            let threshold_numerator = u128::from(self.config.comparison_family_size)
                .checked_mul(PPM)
                .ok_or(AnytimeAuditError::ArithmeticOverflow)?;
            let right = denominator
                .checked_mul(threshold_numerator)
                .ok_or(AnytimeAuditError::ArithmeticOverflow)?;
            Ok(AnytimeAxisReport {
                axis: AnytimeExposureAxis::ALL[index],
                pair_count: self.observed_pairs,
                positive_pairs: wins,
                negative_pairs: losses,
                tied_pairs: tied,
                e_value_numerator: numerator,
                e_value_denominator: denominator,
                threshold_numerator,
                threshold_denominator: self.config.familywise_alpha_ppm,
                crossed: left >= right,
            })
        })?;

        Ok(AnytimeAuditReport {
            task_sha256: self.config.task_sha256,
            model_sha256: self.config.model_sha256,
            evaluator_sha256: self.config.evaluator_sha256,
            treatment_policy_sha256: self.config.treatment_policy_sha256,
            control_policy_sha256: self.config.control_policy_sha256,
            assignment_manifest_sha256: self.config.assignment_manifest_sha256,
            familywise_alpha_ppm: self.config.familywise_alpha_ppm,
            comparison_family_size: self.config.comparison_family_size,
            observed_pairs: self.observed_pairs,
            latest_sequence: u64::from(self.observed_pairs),
            familywise_crossed: axes.iter().any(|axis| axis.crossed),
            axes,
        })
    }

    pub const fn observed_pairs(&self) -> u32 {
        self.observed_pairs
    }
}

fn validate_pair(
    config: &AnytimeAuditConfig,
    pair: &RandomizedPair,
    pair_ids: &BTreeSet<u64>,
    run_ids: &BTreeSet<u64>,
    pair_contexts: &BTreeSet<[u8; 32]>,
    assignment_receipts: &BTreeSet<[u8; 32]>,
) -> Result<(), AnytimeAuditError> {
    if pair.pair_id == 0 || pair.first_run_id == 0 || pair.second_run_id == 0 {
        return Err(AnytimeAuditError::ZeroIdentifier { field: "pair/run" });
    }
    if pair.first_run_id == pair.second_run_id {
        return Err(AnytimeAuditError::DuplicateIdentifier {
            field: "run within pair",
            value: pair.first_run_id,
        });
    }
    if pair_ids.contains(&pair.pair_id) {
        return Err(AnytimeAuditError::DuplicateIdentifier {
            field: "pair",
            value: pair.pair_id,
        });
    }
    for id in [pair.first_run_id, pair.second_run_id] {
        if run_ids.contains(&id) {
            return Err(AnytimeAuditError::DuplicateIdentifier {
                field: "run",
                value: id,
            });
        }
    }
    for (field, digest) in [
        ("task", pair.task_sha256),
        ("model", pair.model_sha256),
        ("evaluator", pair.evaluator_sha256),
        ("treatment policy", pair.treatment_policy_sha256),
        ("control policy", pair.control_policy_sha256),
        ("assignment manifest", pair.assignment_manifest_sha256),
        ("first context", pair.first_context_sha256),
        ("second context", pair.second_context_sha256),
        ("first source family", pair.first_source_family_sha256),
        ("second source family", pair.second_source_family_sha256),
        ("assignment receipt", pair.assignment_receipt_sha256),
    ] {
        if digest == [0; 32] {
            return Err(AnytimeAuditError::ZeroDigest { field });
        }
    }
    if pair.first_context_sha256 != pair.second_context_sha256
        || pair.first_source_family_sha256 != pair.second_source_family_sha256
    {
        return Err(AnytimeAuditError::PairContextMismatch);
    }
    if pair_contexts.contains(&pair.first_context_sha256) {
        return Err(AnytimeAuditError::DuplicatePairContext);
    }
    if assignment_receipts.contains(&pair.assignment_receipt_sha256) {
        return Err(AnytimeAuditError::DuplicateDigest {
            field: "assignment receipt",
        });
    }
    for (field, actual, expected) in [
        ("task", pair.task_sha256, config.task_sha256),
        ("model", pair.model_sha256, config.model_sha256),
        ("evaluator", pair.evaluator_sha256, config.evaluator_sha256),
        ("treatment policy", pair.treatment_policy_sha256, config.treatment_policy_sha256),
        ("control policy", pair.control_policy_sha256, config.control_policy_sha256),
        ("assignment manifest", pair.assignment_manifest_sha256, config.assignment_manifest_sha256),
    ] {
        if actual != expected {
            return Err(AnytimeAuditError::DesignDigestMismatch { field });
        }
    }
    if pair.assignment_scheme != AssignmentScheme::IndependentFairCoin {
        return Err(AnytimeAuditError::UnsupportedAssignmentScheme);
    }
    if pair.treatment_probability_ppm != 500_000 {
        return Err(AnytimeAuditError::InvalidTreatmentProbability {
            actual_ppm: pair.treatment_probability_ppm,
        });
    }
    if pair.episode_duration_us == 0 {
        return Err(AnytimeAuditError::InvalidEpisodeDuration);
    }
    validate_outcome(pair.first_outcome, pair.episode_duration_us)?;
    validate_outcome(pair.second_outcome, pair.episode_duration_us)?;
    Ok(())
}

fn validate_outcome(outcome: ArmOutcome, duration_us: u64) -> Result<(), AnytimeAuditError> {
    let exposure = outcome_exposure(outcome, duration_us);
    let total = u128::from(exposure[0]) + u128::from(exposure[1]) + u128::from(exposure[2]);
    if total > u128::from(duration_us) {
        return Err(AnytimeAuditError::ExposureExceedsEpisode {
            measured_us: total,
            maximum_us: duration_us,
        });
    }
    Ok(())
}

fn outcome_exposure(outcome: ArmOutcome, duration_us: u64) -> [u64; 3] {
    match outcome {
        ArmOutcome::Completed(values) => [
            values.wrong_belief_us,
            values.abstention_us,
            values.missing_output_us,
        ],
        ArmOutcome::Abstained => [0, duration_us, 0],
        ArmOutcome::Failed(OutputFailure::Missing | OutputFailure::Timeout | OutputFailure::Invalid) =>
            [0, 0, duration_us],
    }
}

fn pow3(exponent: u32) -> Result<u128, AnytimeAuditError> {
    (0..exponent).try_fold(1_u128, |value, _| {
        value.checked_mul(3).ok_or(AnytimeAuditError::ArithmeticOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExposureValues, PredictedBenefit};

    fn digest(value: u8) -> [u8; 32] {
        [value; 32]
    }

    fn config(maximum_pairs: u32) -> AnytimeAuditConfig {
        AnytimeAuditConfig {
            familywise_alpha_ppm: 50_000,
            comparison_family_size: 3,
            maximum_pairs,
            task_sha256: digest(1),
            model_sha256: digest(2),
            evaluator_sha256: digest(3),
            treatment_policy_sha256: digest(4),
            control_policy_sha256: digest(5),
            assignment_manifest_sha256: digest(6),
        }
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
        treatment: ArmOutcome,
        control: ArmOutcome,
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
            predicted_benefit: PredictedBenefit {
                wrong_belief_us: 0,
                abstention_us: 0,
                missing_output_us: 0,
            },
        }
    }

    fn positive_pair(id: u64) -> RandomizedPair {
        pair(id, id % 2 == 0, completed(0, 0, 0), completed(10, 0, 0))
    }

    fn negative_pair(id: u64) -> RandomizedPair {
        pair(id, id % 2 == 0, completed(10, 0, 0), completed(0, 0, 0))
    }

    #[test]
    fn beneficial_signs_grow_the_exact_rational_e_process() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let mut report = state.observe(1, &positive_pair(1)).unwrap();
        report = state.observe(2, &positive_pair(2)).unwrap();
        report = state.observe(3, &positive_pair(3)).unwrap();
        let axis = &report.axes[0];
        assert_eq!(axis.e_value_numerator, 27);
        assert_eq!(axis.e_value_denominator, 8);
        assert_eq!(axis.positive_pairs, 3);
        assert_eq!(axis.negative_pairs, 0);
    }

    #[test]
    fn harmful_signs_reduce_evidence() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let _ = state.observe(1, &negative_pair(1)).unwrap();
        let report = state.observe(2, &negative_pair(2)).unwrap();
        assert_eq!(report.axes[0].e_value_numerator, 1);
        assert_eq!(report.axes[0].e_value_denominator, 4);
        assert!(!report.axes[0].crossed);
    }

    #[test]
    fn ties_do_not_change_the_e_value() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let tie = pair(1, true, completed(10, 0, 0), completed(10, 0, 0));
        let report = state.observe(1, &tie).unwrap();
        assert_eq!(report.axes[0].e_value_numerator, 1);
        assert_eq!(report.axes[0].e_value_denominator, 1);
        assert_eq!(report.axes[0].tied_pairs, 1);
    }

    #[test]
    fn bonferroni_crossing_uses_the_full_declared_comparison_family() {
        let mut state = AnytimeAuditState::new(config(12)).unwrap();
        for sequence in 1..=10 {
            let report = state.observe(sequence, &positive_pair(sequence)).unwrap();
            assert!(!report.axes[0].crossed);
        }
        let report = state.observe(11, &positive_pair(11)).unwrap();
        assert!(report.axes[0].crossed);
        assert!(report.familywise_crossed);
        assert_eq!(report.axes[0].threshold_numerator, 3_000_000);
        assert_eq!(report.axes[0].threshold_denominator, 50_000);
    }

    #[test]
    fn optional_stopping_crossing_probability_is_bounded_on_a_finite_null_tree() {
        use std::collections::BTreeMap;

        const HORIZON: u32 = 20;
        let mut alive = BTreeMap::<(u32, u32), u128>::from([((0, 0), 1)]);
        let mut crossed_terminal_paths = 0_u128;

        for look in 1..=HORIZON {
            let mut next = BTreeMap::<(u32, u32), u128>::new();
            let mut newly_crossed = 0_u128;
            for ((wins, losses), paths) in alive {
                for (next_wins, next_losses) in [(wins + 1, losses), (wins, losses + 1)] {
                    let numerator = pow3(next_wins).unwrap();
                    let denominator = 1_u128 << (next_wins + next_losses);
                    let crossed = numerator * 50_000
                        >= denominator * 3 * 1_000_000;
                    if crossed {
                        newly_crossed += paths;
                    } else {
                        *next.entry((next_wins, next_losses)).or_default() += paths;
                    }
                }
            }
            crossed_terminal_paths = crossed_terminal_paths * 2 + newly_crossed;
            alive = next;
            let total_terminal_paths = 1_u128 << look;
            assert!(
                crossed_terminal_paths * 3 * 1_000_000
                    <= total_terminal_paths * 50_000
            );
        }
    }

    #[test]
    fn missing_and_abstained_runs_remain_distinct_full_horizon_exposures() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let treatment_missing = pair(
            1,
            true,
            ArmOutcome::Failed(OutputFailure::Timeout),
            completed(0, 0, 0),
        );
        let report = state.observe(1, &treatment_missing).unwrap();
        assert_eq!(report.axes[2].negative_pairs, 1);
        assert_eq!(report.axes[1].tied_pairs, 1);

        let treatment_abstains = pair(2, false, ArmOutcome::Abstained, completed(0, 0, 0));
        let report = state.observe(2, &treatment_abstains).unwrap();
        assert_eq!(report.axes[1].negative_pairs, 1);
        assert_eq!(report.axes[2].tied_pairs, 2);
    }

    #[test]
    fn missing_sequence_is_rejected_without_advancing_state() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        assert!(matches!(
            state.observe(2, &positive_pair(1)),
            Err(AnytimeAuditError::NonSequentialLook { expected: 1, actual: 2 })
        ));
        assert_eq!(state.observed_pairs(), 0);
        assert_eq!(state.observe(1, &positive_pair(1)).unwrap().observed_pairs, 1);
    }

    #[test]
    fn duplicate_pair_run_context_and_assignment_receipts_fail_closed() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let first = positive_pair(1);
        state.observe(1, &first).unwrap();
        assert!(matches!(
            state.observe(2, &first),
            Err(AnytimeAuditError::DuplicateIdentifier { field: "pair", .. })
        ));

        let mut duplicate_run = positive_pair(2);
        duplicate_run.first_run_id = first.first_run_id;
        assert!(matches!(
            state.observe(2, &duplicate_run),
            Err(AnytimeAuditError::DuplicateIdentifier { field: "run", .. })
        ));

        let mut duplicate_context = positive_pair(2);
        duplicate_context.first_context_sha256 = first.first_context_sha256;
        duplicate_context.second_context_sha256 = first.second_context_sha256;
        assert!(matches!(
            state.observe(2, &duplicate_context),
            Err(AnytimeAuditError::DuplicatePairContext)
        ));

        let mut duplicate_receipt = positive_pair(2);
        duplicate_receipt.assignment_receipt_sha256 = first.assignment_receipt_sha256;
        assert!(matches!(
            state.observe(2, &duplicate_receipt),
            Err(AnytimeAuditError::DuplicateDigest { field: "assignment receipt" })
        ));
        assert_eq!(state.observed_pairs(), 1);
    }

    #[test]
    fn non_fair_assignment_is_rejected() {
        let mut state = AnytimeAuditState::new(config(10)).unwrap();
        let mut invalid = positive_pair(1);
        invalid.treatment_probability_ppm = 250_000;
        assert!(matches!(
            state.observe(1, &invalid),
            Err(AnytimeAuditError::InvalidTreatmentProbability { actual_ppm: 250_000 })
        ));
        assert_eq!(state.observed_pairs(), 0);
    }

    #[test]
    fn configuration_requires_distinct_policies_and_predeclared_family() {
        let mut invalid = config(10);
        invalid.control_policy_sha256 = invalid.treatment_policy_sha256;
        assert_eq!(
            AnytimeAuditState::new(invalid),
            Err(AnytimeAuditError::IdenticalPolicyDigests)
        );

        let mut invalid = config(10);
        invalid.comparison_family_size = 2;
        assert!(matches!(
            AnytimeAuditState::new(invalid),
            Err(AnytimeAuditError::InvalidComparisonFamilySize { actual: 2, minimum: 3 })
        ));
    }

    #[test]
    fn exact_integer_evidence_is_safe_at_the_sixty_four_pair_limit() {
        let mut state = AnytimeAuditState::new(config(64)).unwrap();
        let mut report = state.report().unwrap();
        for sequence in 1..=64 {
            report = state.observe(sequence, &positive_pair(sequence)).unwrap();
        }
        assert_eq!(report.axes[0].e_value_numerator, 3_u128.pow(64));
        assert_eq!(report.axes[0].e_value_denominator, 1_u128 << 64);
        assert!(matches!(
            state.observe(65, &positive_pair(65)),
            Err(AnytimeAuditError::PairLimitReached { maximum: 64 })
        ));
    }
}

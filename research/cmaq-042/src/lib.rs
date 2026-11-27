#![forbid(unsafe_code)]

//! Finite-horizon reference-null calibration for path-maximum thresholds.
//!
//! This module calibrates a single predeclared test trajectory against an
//! independent bank of exchangeable reference-null trajectories. Its guarantee
//! is limited to the frozen horizon and supplied null exchangeability contract.
//! It does not replace an unbounded-horizon Ville threshold.

use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub struct ReferencePathMaximum {
    pub trajectory_id: String,
    pub trace_sha256: String,
    /// Log of the maximum pre-registered e-process statistic over 0..=T.
    /// The path must include initial wealth one, so this value is nonnegative.
    pub log_max_e_value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceNullBank {
    pub bank_id: String,
    pub horizon: u64,
    pub calibration_reference_size: usize,
    pub alpha_numerator: u32,
    pub alpha_denominator: u32,
    pub monitor_build_sha256: String,
    pub score_definition_sha256: String,
    pub rank_reference_sha256: String,
    pub null_simulator_sha256: String,
    pub references: Vec<ReferencePathMaximum>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TestPathMaximum {
    pub trajectory_id: String,
    pub trace_sha256: String,
    pub horizon: u64,
    pub calibration_reference_size: usize,
    pub monitor_build_sha256: String,
    pub score_definition_sha256: String,
    pub rank_reference_sha256: String,
    pub log_max_e_value: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CalibrationError {
    EmptyBankId,
    EmptyTrajectoryId,
    ZeroHorizon,
    ZeroCalibrationReferenceSize,
    EmptyReferenceBank,
    InvalidAlpha,
    DuplicateReferenceTrajectory,
    DuplicateReferenceTrace,
    InvalidEvidenceDigest(&'static str),
    NonFiniteLogMaximum,
    LogMaximumBelowInitialWealth,
    ArithmeticOverflow,
    HorizonMismatch,
    CalibrationReferenceSizeMismatch,
    MonitorBuildMismatch,
    ScoreDefinitionMismatch,
    RankReferenceMismatch,
    TestTrajectoryAppearsInReferenceBank,
    TestTraceAppearsInReferenceBank,
    CalibrationAlreadyConsumed,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ThresholdDecision {
    Reject {
        log_test_maximum: f64,
        log_threshold: f64,
        reference_count: usize,
        threshold_rank_one_based: usize,
    },
    DoNotReject {
        log_test_maximum: f64,
        log_threshold: f64,
        reference_count: usize,
        threshold_rank_one_based: usize,
    },
    Disabled {
        log_test_maximum: f64,
        reference_count: usize,
        threshold_rank_one_based: usize,
    },
}

/// Calibrated finite-horizon threshold. Not Clone: one in-memory instance can
/// evaluate at most one test trajectory. Operational one-use enforcement across
/// processes still requires an external, authenticated bank-consumption ledger.
#[derive(Debug)]
pub struct FiniteHorizonGate {
    bank_id: String,
    horizon: u64,
    calibration_reference_size: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
    monitor_build_sha256: String,
    score_definition_sha256: String,
    rank_reference_sha256: String,
    reference_ids: HashSet<String>,
    reference_trace_hashes: HashSet<String>,
    null_simulator_sha256: String,
    reference_count: usize,
    threshold_rank_one_based: usize,
    log_threshold: Option<f64>,
    consumed: bool,
}

impl FiniteHorizonGate {
    pub fn calibrate(bank: ReferenceNullBank) -> Result<Self, CalibrationError> {
        if bank.bank_id.trim().is_empty() {
            return Err(CalibrationError::EmptyBankId);
        }
        if bank.horizon == 0 {
            return Err(CalibrationError::ZeroHorizon);
        }
        if bank.calibration_reference_size == 0 {
            return Err(CalibrationError::ZeroCalibrationReferenceSize);
        }
        if bank.alpha_numerator == 0
            || bank.alpha_denominator == 0
            || bank.alpha_numerator >= bank.alpha_denominator
        {
            return Err(CalibrationError::InvalidAlpha);
        }
        if bank.references.is_empty() {
            return Err(CalibrationError::EmptyReferenceBank);
        }

        validate_digest(&bank.monitor_build_sha256, "monitor_build_sha256")?;
        validate_digest(&bank.score_definition_sha256, "score_definition_sha256")?;
        validate_digest(&bank.rank_reference_sha256, "rank_reference_sha256")?;
        validate_digest(&bank.null_simulator_sha256, "null_simulator_sha256")?;

        let mut reference_ids = HashSet::with_capacity(bank.references.len());
        let mut reference_trace_hashes = HashSet::with_capacity(bank.references.len());
        let mut sorted_maxima = Vec::with_capacity(bank.references.len());
        for reference in &bank.references {
            if reference.trajectory_id.trim().is_empty() {
                return Err(CalibrationError::EmptyTrajectoryId);
            }
            if !reference_ids.insert(reference.trajectory_id.clone()) {
                return Err(CalibrationError::DuplicateReferenceTrajectory);
            }
            validate_digest(&reference.trace_sha256, "reference_trace_sha256")?;
            if !reference_trace_hashes.insert(reference.trace_sha256.clone()) {
                return Err(CalibrationError::DuplicateReferenceTrace);
            }
            validate_log_maximum(reference.log_max_e_value)?;
            sorted_maxima.push(reference.log_max_e_value);
        }

        let reference_count = sorted_maxima.len();
        let threshold_rank_one_based = calibrated_order_rank(
            reference_count,
            bank.alpha_numerator,
            bank.alpha_denominator,
        )?;
        sorted_maxima.sort_by(f64::total_cmp);

        let log_threshold = if threshold_rank_one_based <= reference_count {
            Some(sorted_maxima[threshold_rank_one_based - 1])
        } else {
            None
        };

        Ok(Self {
            bank_id: bank.bank_id,
            horizon: bank.horizon,
            calibration_reference_size: bank.calibration_reference_size,
            alpha_numerator: bank.alpha_numerator,
            alpha_denominator: bank.alpha_denominator,
            monitor_build_sha256: bank.monitor_build_sha256,
            score_definition_sha256: bank.score_definition_sha256,
            rank_reference_sha256: bank.rank_reference_sha256,
            reference_ids,
            reference_trace_hashes,
            null_simulator_sha256: bank.null_simulator_sha256,
            reference_count,
            threshold_rank_one_based,
            log_threshold,
            consumed: false,
        })
    }

    /// Evaluates exactly one test trajectory using a strict exceedance rule.
    ///
    /// A test maximum equal to the reference order statistic is not rejected;
    /// this makes ties conservative. Any binding mismatch fails before the
    /// one-shot gate is consumed.
    pub fn evaluate_once(
        &mut self,
        test: &TestPathMaximum,
    ) -> Result<ThresholdDecision, CalibrationError> {
        if self.consumed {
            return Err(CalibrationError::CalibrationAlreadyConsumed);
        }
        validate_test_path(test)?;

        if self.reference_ids.contains(&test.trajectory_id) {
            return Err(CalibrationError::TestTrajectoryAppearsInReferenceBank);
        }
        if self.reference_trace_hashes.contains(&test.trace_sha256) {
            return Err(CalibrationError::TestTraceAppearsInReferenceBank);
        }
        if test.horizon != self.horizon {
            return Err(CalibrationError::HorizonMismatch);
        }
        if test.calibration_reference_size != self.calibration_reference_size {
            return Err(CalibrationError::CalibrationReferenceSizeMismatch);
        }
        if test.monitor_build_sha256 != self.monitor_build_sha256 {
            return Err(CalibrationError::MonitorBuildMismatch);
        }
        if test.score_definition_sha256 != self.score_definition_sha256 {
            return Err(CalibrationError::ScoreDefinitionMismatch);
        }
        if test.rank_reference_sha256 != self.rank_reference_sha256 {
            return Err(CalibrationError::RankReferenceMismatch);
        }

        let decision = match self.log_threshold {
            Some(log_threshold) if test.log_max_e_value > log_threshold => {
                ThresholdDecision::Reject {
                    log_test_maximum: test.log_max_e_value,
                    log_threshold,
                    reference_count: self.reference_count,
                    threshold_rank_one_based: self.threshold_rank_one_based,
                }
            }
            Some(log_threshold) => ThresholdDecision::DoNotReject {
                log_test_maximum: test.log_max_e_value,
                log_threshold,
                reference_count: self.reference_count,
                threshold_rank_one_based: self.threshold_rank_one_based,
            },
            None => ThresholdDecision::Disabled {
                log_test_maximum: test.log_max_e_value,
                reference_count: self.reference_count,
                threshold_rank_one_based: self.threshold_rank_one_based,
            },
        };
        self.consumed = true;
        Ok(decision)
    }

    pub fn bank_id(&self) -> &str {
        &self.bank_id
    }

    pub fn horizon(&self) -> u64 {
        self.horizon
    }

    pub fn reference_count(&self) -> usize {
        self.reference_count
    }

    pub fn threshold_rank_one_based(&self) -> usize {
        self.threshold_rank_one_based
    }

    pub fn log_threshold(&self) -> Option<f64> {
        self.log_threshold
    }

    pub fn is_disabled(&self) -> bool {
        self.log_threshold.is_none()
    }

    pub fn null_simulator_sha256(&self) -> &str {
        &self.null_simulator_sha256
    }

    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    pub fn alpha(&self) -> (u32, u32) {
        (self.alpha_numerator, self.alpha_denominator)
    }
}

fn validate_test_path(test: &TestPathMaximum) -> Result<(), CalibrationError> {
    if test.trajectory_id.trim().is_empty() {
        return Err(CalibrationError::EmptyTrajectoryId);
    }
    if test.horizon == 0 {
        return Err(CalibrationError::ZeroHorizon);
    }
    if test.calibration_reference_size == 0 {
        return Err(CalibrationError::ZeroCalibrationReferenceSize);
    }
    validate_digest(&test.trace_sha256, "test_trace_sha256")?;
    validate_digest(&test.monitor_build_sha256, "monitor_build_sha256")?;
    validate_digest(&test.score_definition_sha256, "score_definition_sha256")?;
    validate_digest(&test.rank_reference_sha256, "rank_reference_sha256")?;
    validate_log_maximum(test.log_max_e_value)
}

fn validate_log_maximum(value: f64) -> Result<(), CalibrationError> {
    if !value.is_finite() {
        return Err(CalibrationError::NonFiniteLogMaximum);
    }
    if value < 0.0 {
        return Err(CalibrationError::LogMaximumBelowInitialWealth);
    }
    Ok(())
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), CalibrationError> {
    let valid = value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(CalibrationError::InvalidEvidenceDigest(label))
    }
}

/// Returns ceil((1-alpha) * (B+1)) using exact integer arithmetic.
fn calibrated_order_rank(
    reference_count: usize,
    alpha_numerator: u32,
    alpha_denominator: u32,
) -> Result<usize, CalibrationError> {
    let total = (reference_count as u128)
        .checked_add(1)
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    let upper_numerator = u128::from(alpha_denominator - alpha_numerator);
    let product = total
        .checked_mul(upper_numerator)
        .ok_or(CalibrationError::ArithmeticOverflow)?;
    let denominator = u128::from(alpha_denominator);
    let rank = product
        .checked_add(denominator - 1)
        .ok_or(CalibrationError::ArithmeticOverflow)?
        / denominator;
    usize::try_from(rank).map_err(|_| CalibrationError::ArithmeticOverflow)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const SHA_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn bank(reference_count: usize) -> ReferenceNullBank {
        ReferenceNullBank {
            bank_id: "mission-bank-001".into(),
            horizon: 500,
            calibration_reference_size: 99,
            alpha_numerator: 1,
            alpha_denominator: 20,
            monitor_build_sha256: SHA_A.into(),
            score_definition_sha256: SHA_B.into(),
            rank_reference_sha256: SHA_C.into(),
            null_simulator_sha256: SHA_D.into(),
            references: (0..reference_count)
                .map(|index| ReferencePathMaximum {
                    trajectory_id: format!("null-{index}"),
                    trace_sha256: SHA_A.into(),
                    log_max_e_value: index as f64,
                })
                .collect(),
        }
    }

    fn test_path(log_maximum: f64) -> TestPathMaximum {
        TestPathMaximum {
            trajectory_id: "deployment-001".into(),
            trace_sha256: SHA_D.into(),
            horizon: 500,
            calibration_reference_size: 99,
            monitor_build_sha256: SHA_A.into(),
            score_definition_sha256: SHA_B.into(),
            rank_reference_sha256: SHA_C.into(),
            log_max_e_value: log_maximum,
        }
    }

    #[test]
    fn alpha_point_zero_five_with_nineteen_references_uses_nineteenth_order_statistic() {
        let gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        assert_eq!(gate.threshold_rank_one_based(), 19);
        assert_eq!(gate.log_threshold(), Some(18.0));
        assert!(!gate.is_disabled());
    }

    #[test]
    fn insufficient_reference_bank_disables_finite_horizon_rejection() {
        let gate = FiniteHorizonGate::calibrate(bank(18)).unwrap();
        assert_eq!(gate.threshold_rank_one_based(), 19);
        assert_eq!(gate.log_threshold(), None);
        assert!(gate.is_disabled());
    }

    #[test]
    fn strict_tie_at_threshold_does_not_reject() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let decision = gate.evaluate_once(&test_path(18.0)).unwrap();
        assert!(matches!(decision, ThresholdDecision::DoNotReject { .. }));
        assert!(gate.is_consumed());
    }

    #[test]
    fn strict_exceedance_rejects_above_threshold() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let decision = gate.evaluate_once(&test_path(18.000_001)).unwrap();
        assert!(matches!(decision, ThresholdDecision::Reject { .. }));
    }

    #[test]
    fn disabled_threshold_never_rejects() {
        let mut gate = FiniteHorizonGate::calibrate(bank(18)).unwrap();
        let decision = gate.evaluate_once(&test_path(10_000.0)).unwrap();
        assert!(matches!(decision, ThresholdDecision::Disabled { .. }));
    }

    #[test]
    fn horizon_mismatch_fails_without_consuming_gate() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let mut test = test_path(20.0);
        test.horizon = 501;
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::HorizonMismatch)
        );
        assert!(!gate.is_consumed());
    }

    #[test]
    fn build_score_and_rank_reference_mismatches_fail_closed() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let mut test = test_path(20.0);
        test.monitor_build_sha256 = SHA_D.into();
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::MonitorBuildMismatch)
        );

        test = test_path(20.0);
        test.score_definition_sha256 = SHA_D.into();
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::ScoreDefinitionMismatch)
        );

        test = test_path(20.0);
        test.rank_reference_sha256 = SHA_D.into();
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::RankReferenceMismatch)
        );
        assert!(!gate.is_consumed());
    }

    #[test]
    fn calibration_reference_size_mismatch_fails_closed() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let mut test = test_path(20.0);
        test.calibration_reference_size = 100;
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::CalibrationReferenceSizeMismatch)
        );
    }

    #[test]
    fn duplicate_reference_ids_are_rejected() {
        let mut input = bank(19);
        input.references[1].trajectory_id = input.references[0].trajectory_id.clone();
        assert_eq!(
            FiniteHorizonGate::calibrate(input).err(),
            Some(CalibrationError::DuplicateReferenceTrajectory)
        );
    }

    #[test]
    fn duplicate_reference_trace_digests_are_rejected() {
        let mut input = bank(19);
        input.references[1].trace_sha256 = input.references[0].trace_sha256.clone();
        assert_eq!(
            FiniteHorizonGate::calibrate(input).err(),
            Some(CalibrationError::DuplicateReferenceTrace)
        );
    }

    #[test]
    fn malformed_digests_and_non_finite_statistics_are_rejected() {
        let mut input = bank(19);
        input.references[0].trace_sha256 = "bad".into();
        assert_eq!(
            FiniteHorizonGate::calibrate(input).err(),
            Some(CalibrationError::InvalidEvidenceDigest(
                "reference_trace_sha256"
            ))
        );

        let mut input = bank(19);
        input.references[0].log_max_e_value = f64::NAN;
        assert_eq!(
            FiniteHorizonGate::calibrate(input).err(),
            Some(CalibrationError::NonFiniteLogMaximum)
        );

        let mut input = bank(19);
        input.references[0].log_max_e_value = -0.001;
        assert_eq!(
            FiniteHorizonGate::calibrate(input).err(),
            Some(CalibrationError::LogMaximumBelowInitialWealth)
        );
    }

    #[test]
    fn test_path_in_reference_bank_is_rejected() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        let mut test = test_path(20.0);
        test.trajectory_id = "null-0".into();
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::TestTrajectoryAppearsInReferenceBank)
        );

        test = test_path(20.0);
        test.trace_sha256 = SHA_A.into();
        assert_eq!(
            gate.evaluate_once(&test),
            Err(CalibrationError::TestTraceAppearsInReferenceBank)
        );
    }

    #[test]
    fn gate_is_consumed_exactly_once_after_valid_test_path() {
        let mut gate = FiniteHorizonGate::calibrate(bank(19)).unwrap();
        gate.evaluate_once(&test_path(20.0)).unwrap();
        assert_eq!(
            gate.evaluate_once(&test_path(20.0)),
            Err(CalibrationError::CalibrationAlreadyConsumed)
        );
    }

    #[test]
    fn order_rank_uses_exact_integer_ceiling() {
        assert_eq!(calibrated_order_rank(19, 1, 20), Ok(19));
        assert_eq!(calibrated_order_rank(18, 1, 20), Ok(19));
        assert_eq!(calibrated_order_rank(99, 1, 20), Ok(95));
    }

    #[test]
    fn exchangeable_rank_positions_respect_the_alpha_bound() {
        for (reference_count, alpha_numerator, alpha_denominator) in
            [(19usize, 1u32, 20u32), (39, 1, 20), (99, 1, 20), (99, 1, 10)]
        {
            let threshold_rank =
                calibrated_order_rank(reference_count, alpha_numerator, alpha_denominator)
                    .unwrap();
            let mut rejected_positions = 0usize;
            for test_rank in 1..=reference_count + 1 {
                let mut input = bank(reference_count);
                for (index, reference) in input.references.iter_mut().enumerate() {
                    reference.trace_sha256 = format!("{:064x}", index + 1);
                    reference.log_max_e_value = index as f64 + 0.5;
                }
                let mut gate = FiniteHorizonGate::calibrate(input).unwrap();
                let test_log_max = test_rank as f64 - 1.0;
                let decision = gate.evaluate_once(&test_path(test_log_max)).unwrap();
                if matches!(decision, ThresholdDecision::Reject { .. }) {
                    rejected_positions += 1;
                }
            }
            let observed = rejected_positions as f64 / (reference_count + 1) as f64;
            assert!(
                observed <= f64::from(alpha_numerator) / f64::from(alpha_denominator),
                "rank rejection fraction {observed} exceeded alpha"
            );
        }
    }

    #[test]
    fn initial_wealth_zero_is_allowed_as_path_maximum() {
        let mut input = bank(19);
        input.references[0].log_max_e_value = 0.0;
        let gate = FiniteHorizonGate::calibrate(input).unwrap();
        assert_eq!(gate.threshold_rank_one_based(), 19);
    }
}

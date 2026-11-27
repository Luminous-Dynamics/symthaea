//! Block-conformal open-world adequacy calibration.
//!
//! The caller supplies nonconformity scores from a frozen model set and
//! frozen scoring protocol. Larger scores mean poorer predictive fit.
//! Each calibration unit is an independent episode or a block that is
//! exchangeable with the future test block; the unit score is the maximum
//! within-block score. This does NOT make dependent chronological blocks
//! exchangeable. The assumption belongs to the benchmark/calibration design.
//!
//! Fresh calibration epochs and non-reuse of calibration-unit IDs are
//! enforced across adaptive stages. The caller must preserve the consumed
//! epoch and unit ledgers between stages.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidAlpha,
    EmptyCalibrationSet,
    EmptyTestBlock,
    InvalidIdentity,
    DuplicateCalibrationUnit,
    DuplicateCalibrationEpoch,
    DuplicateConsumedUnit,
    DuplicateTestUnit,
    ModelSetMismatch,
    ScoringProtocolMismatch,
    InvalidNonconformityScore,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBlock {
    pub unit_id: String,
    pub max_nonconformity_score: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationFold {
    pub epoch_id: String,
    pub model_set_digest: String,
    pub scoring_protocol_digest: String,
    pub blocks: Vec<CalibrationBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TestBlock {
    pub unit_id: String,
    pub model_set_digest: String,
    pub scoring_protocol_digest: String,
    /// Per-observation nonconformity scores from this test episode/block.
    pub nonconformity_scores: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdequacyAlarm {
    FlaggedOutOfCalibratedEnvelope,
    NotFlagged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConformalAdequacyReport {
    pub calibration_epoch_id: String,
    pub calibration_units: usize,
    pub test_unit_id: String,
    pub test_block_max_score: f64,
    pub conformal_p_value: f64,
    pub requested_false_alarm_level: f64,
    pub minimum_resolvable_p_value: f64,
    pub minimum_calibration_units_for_requested_level: usize,
    /// True only when the rank grid is fine enough to reject at alpha.
    pub significance_level_resolvable: bool,
    pub alarm: AdequacyAlarm,
    /// IDs the caller must append to its consumed ledgers before another
    /// adaptive stage is calibrated.
    pub consumed_calibration_epoch_ids: Vec<String>,
    /// All calibration and test units consumed by this assessment. The caller
    /// must carry the complete set into every later adaptive stage.
    pub consumed_unit_ids: Vec<String>,
}

/// Minimum n for which the smallest non-randomized conformal p-value 1/(n+1)
/// can be no greater than alpha.
pub fn minimum_calibration_units(alpha: f64) -> Result<usize, Error> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::InvalidAlpha);
    }
    Ok(((1.0 / alpha).ceil() as usize).saturating_sub(1))
}

/// Conservative upper-tail conformal p-value. Ties count as at least as
/// nonconforming as the test block and therefore increase the p-value.
pub fn upper_tail_conformal_p_value(
    calibration_scores: &[f64],
    test_score: f64,
) -> Result<f64, Error> {
    if calibration_scores.is_empty() {
        return Err(Error::EmptyCalibrationSet);
    }
    if !test_score.is_finite()
        || calibration_scores.iter().any(|score| !score.is_finite())
    {
        return Err(Error::InvalidNonconformityScore);
    }
    let at_least_as_extreme = calibration_scores.iter()
        .filter(|score| **score >= test_score)
        .count();
    Ok((1 + at_least_as_extreme) as f64 / (calibration_scores.len() + 1) as f64)
}

fn validate_identity(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

/// Identity digests are canonical lowercase SHA-256 hex, not descriptive labels.
fn validate_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
}

fn validate_unique_ids(ids: &[String]) -> Result<(), Error> {
    for (index, id) in ids.iter().enumerate() {
        if !validate_identity(id) {
            return Err(Error::InvalidIdentity);
        }
        if ids[..index].contains(id) {
            return Err(Error::DuplicateCalibrationUnit);
        }
    }
    Ok(())
}

fn validate_consumed_ledgers(
    epoch_ids: &[String],
    unit_ids: &[String],
) -> Result<(), Error> {
    for (index, id) in epoch_ids.iter().enumerate() {
        if !validate_identity(id) {
            return Err(Error::InvalidIdentity);
        }
        if epoch_ids[..index].contains(id) {
            return Err(Error::DuplicateCalibrationEpoch);
        }
    }
    for (index, id) in unit_ids.iter().enumerate() {
        if !validate_identity(id) {
            return Err(Error::InvalidIdentity);
        }
        if unit_ids[..index].contains(id) {
            return Err(Error::DuplicateConsumedUnit);
        }
    }
    Ok(())
}

/// Assess one future episode/block against a frozen calibration fold.
///
/// Validity is finite-sample and distribution-free only if calibration
/// units and the future test unit are exchangeable at the declared unit
/// level, with the same frozen model-set and scoring-protocol digests.
/// Temporal dependence inside a unit is allowed because the unit score is
/// the block maximum; dependence BETWEEN units is not automatically solved.
pub fn assess_block(
    calibration: &CalibrationFold,
    test: &TestBlock,
    alpha: f64,
    previously_consumed_epoch_ids: &[String],
    previously_consumed_unit_ids: &[String],
) -> Result<ConformalAdequacyReport, Error> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(Error::InvalidAlpha);
    }
    if calibration.blocks.is_empty() {
        return Err(Error::EmptyCalibrationSet);
    }
    if test.nonconformity_scores.is_empty() {
        return Err(Error::EmptyTestBlock);
    }
    if !validate_identity(&calibration.epoch_id)
        || !validate_sha256(&calibration.model_set_digest)
        || !validate_sha256(&calibration.scoring_protocol_digest)
        || !validate_identity(&test.unit_id)
    {
        return Err(Error::InvalidIdentity);
    }
    if !validate_sha256(&test.model_set_digest)
        || !validate_sha256(&test.scoring_protocol_digest)
    {
        return Err(Error::InvalidIdentity);
    }
    if test.model_set_digest != calibration.model_set_digest {
        return Err(Error::ModelSetMismatch);
    }
    if test.scoring_protocol_digest != calibration.scoring_protocol_digest {
        return Err(Error::ScoringProtocolMismatch);
    }
    validate_consumed_ledgers(previously_consumed_epoch_ids, previously_consumed_unit_ids)?;
    if previously_consumed_epoch_ids.contains(&calibration.epoch_id) {
        return Err(Error::DuplicateCalibrationEpoch);
    }
    if previously_consumed_unit_ids.contains(&test.unit_id) {
        return Err(Error::DuplicateTestUnit);
    }
    if previously_consumed_unit_ids.iter().any(|id| {
        calibration.blocks.iter().any(|block| &block.unit_id == id)
    }) {
        return Err(Error::DuplicateCalibrationUnit);
    }

    let calibration_ids: Vec<String> = calibration.blocks.iter()
        .map(|block| block.unit_id.clone()).collect();
    validate_unique_ids(&calibration_ids)?;
    if calibration_ids.contains(&test.unit_id) {
        return Err(Error::DuplicateTestUnit);
    }
    if calibration.blocks.iter().any(|block| !block.max_nonconformity_score.is_finite())
        || test.nonconformity_scores.iter().any(|score| !score.is_finite())
    {
        return Err(Error::InvalidNonconformityScore);
    }

    let calibration_scores: Vec<f64> = calibration.blocks.iter()
        .map(|block| block.max_nonconformity_score).collect();
    let test_block_max_score = test.nonconformity_scores.iter()
        .copied().fold(f64::NEG_INFINITY, f64::max);
    let conformal_p_value =
        upper_tail_conformal_p_value(&calibration_scores, test_block_max_score)?;
    let minimum_resolvable_p_value = 1.0 / (calibration_scores.len() + 1) as f64;
    let min_units = minimum_calibration_units(alpha)?;
    let alarm = if conformal_p_value <= alpha {
        AdequacyAlarm::FlaggedOutOfCalibratedEnvelope
    } else {
        AdequacyAlarm::NotFlagged
    };

    Ok(ConformalAdequacyReport {
        calibration_epoch_id: calibration.epoch_id.clone(),
        calibration_units: calibration_scores.len(),
        test_unit_id: test.unit_id.clone(),
        test_block_max_score,
        conformal_p_value,
        requested_false_alarm_level: alpha,
        minimum_resolvable_p_value,
        minimum_calibration_units_for_requested_level: min_units,
        significance_level_resolvable: minimum_resolvable_p_value <= alpha,
        alarm,
        consumed_calibration_epoch_ids: vec![calibration.epoch_id.clone()],
        consumed_unit_ids: {
            let mut ids = calibration_ids;
            ids.push(test.unit_id.clone());
            ids
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> CalibrationFold {
        CalibrationFold {
            epoch_id: "fold-stage-1".into(),
            model_set_digest: "model-set-sha256:abc".into(),
            scoring_protocol_digest: "score-protocol-sha256:def".into(),
            blocks: (1..=19).map(|index| CalibrationBlock {
                unit_id: format!("episode-{index:02}"),
                max_nonconformity_score: index as f64 / 10.0,
            }).collect(),
        }
    }

    fn test_block(unit_id: &str, scores: Vec<f64>) -> TestBlock {
        TestBlock {
            unit_id: unit_id.into(),
            model_set_digest: "model-set-sha256:abc".into(),
            scoring_protocol_digest: "score-protocol-sha256:def".into(),
            nonconformity_scores: scores,
        }
    }

    #[test]
    fn block_maximum_and_rank_p_value_flag_an_extreme_episode() {
        let report = assess_block(
            &calibration(),
            &test_block("test-episode", vec![1.1, 1.8, 2.0, 1.4]),
            0.05,
            &[],
            &[],
        ).unwrap();
        assert_eq!(report.test_block_max_score, 2.0);
        assert_eq!(report.conformal_p_value, 0.05);
        assert_eq!(report.alarm, AdequacyAlarm::FlaggedOutOfCalibratedEnvelope);
        assert!(report.significance_level_resolvable);
    }

    #[test]
    fn plausible_episode_is_not_flagged() {
        let report = assess_block(
            &calibration(),
            &test_block("test-episode", vec![1.1, 1.75, 1.4]),
            0.05,
            &[],
            &[],
        ).unwrap();
        assert_eq!(report.test_block_max_score, 1.75);
        assert_eq!(report.conformal_p_value, 0.15);
        assert_eq!(report.alarm, AdequacyAlarm::NotFlagged);
    }

    #[test]
    fn ties_are_counted_conservatively() {
        let p = upper_tail_conformal_p_value(&[1.0, 2.0, 2.0, 3.0], 2.0).unwrap();
        assert_eq!(p, 0.8);
    }

    #[test]
    fn finite_sample_resolution_is_reported() {
        assert_eq!(minimum_calibration_units(0.05).unwrap(), 19);
        let short_fold = CalibrationFold {
            epoch_id: "short".into(),
            model_set_digest: "models".into(),
            scoring_protocol_digest: "scores".into(),
            blocks: (0..4).map(|i| CalibrationBlock {
                unit_id: format!("u{i}"),
                max_nonconformity_score: i as f64,
            }).collect(),
        };
        let report = assess_block(
            &short_fold,&test_block("new",vec![100.0]),0.05,&[],&[]
        ).unwrap();
        assert_eq!(report.conformal_p_value,0.2);
        assert!(!report.significance_level_resolvable);
        assert_eq!(report.alarm,AdequacyAlarm::NotFlagged);
    }

    #[test]
    fn model_and_scoring_protocol_must_match() {
        let mut wrong_model=test_block("test",vec![2.0]);
        wrong_model.model_set_digest="changed-model".into();
        assert_eq!(
            assess_block(&calibration(),&wrong_model,0.05,&[],&[]),
            Err(Error::ModelSetMismatch)
        );
        let mut wrong_protocol=test_block("test",vec![2.0]);
        wrong_protocol.scoring_protocol_digest="changed-score".into();
        assert_eq!(
            assess_block(&calibration(),&wrong_protocol,0.05,&[],&[]),
            Err(Error::ScoringProtocolMismatch)
        );
    }

    #[test]
    fn adaptive_stages_cannot_reuse_consumed_calibration_folds_or_units() {
        let fold=calibration();
        assert_eq!(
            assess_block(&fold,&test_block("next",vec![2.0]),0.05,
                &["fold-stage-1".into()],&[]),
            Err(Error::DuplicateCalibrationEpoch)
        );
        assert_eq!(
            assess_block(&fold,&test_block("next",vec![2.0]),0.05,
                &[],&["episode-03".into()]),
            Err(Error::DuplicateCalibrationUnit)
        );
    }

    #[test]
    fn test_unit_cannot_be_reused_as_a_calibration_unit() {
        let fold=calibration();
        assert_eq!(
            assess_block(&fold,&test_block("episode-04",vec![2.0]),0.05,&[],&[]),
            Err(Error::DuplicateTestUnit)
        );
    }

    #[test]
    fn invalid_scores_alpha_and_empty_blocks_fail_closed() {
        assert_eq!(minimum_calibration_units(1.0),Err(Error::InvalidAlpha));
        assert_eq!(
            upper_tail_conformal_p_value(&[],1.0),
            Err(Error::EmptyCalibrationSet)
        );
        assert_eq!(
            upper_tail_conformal_p_value(&[1.0],f64::NAN),
            Err(Error::InvalidNonconformityScore)
        );
        let empty_test=test_block("test",vec![]);
        assert_eq!(
            assess_block(&calibration(),&empty_test,0.05,&[],&[]),
            Err(Error::EmptyTestBlock)
        );
    }
}

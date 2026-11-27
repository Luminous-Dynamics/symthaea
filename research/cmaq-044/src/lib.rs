#![forbid(unsafe_code)]

//! Paired, effective-block-level inference for sequential-monitor benchmarks.
//!
//! One input row is one pre-registered common-cause block shared by baseline
//! and candidate. Missed detections receive horizon+1 penalty. Bootstrap
//! resampling occurs only at that block level, preserving within-block pairing.
//! The interval is conditional on blocks being representative/exchangeable;
//! it is not a distribution-free guarantee under arbitrary dependence.

use std::collections::{BTreeMap, HashSet};

const PPM: u32 = 1_000_000;
const US_PER_HOUR_PPM: u128 = 3_600_000_000_000_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExposureInterval {
    pub domain_id: String,
    pub start_us: u64,
    pub end_us: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmOutcome {
    /// Delay from the true shift onset, not the stream start. None is a miss.
    pub detection_delay_us: Option<u64>,
    pub fallback_completed_before_deadline: bool,
    pub safety_violations: u64,
    pub false_warnings: u64,
    pub false_quarantines: u64,
    pub safe_task_completions: u64,
    pub useful_progress_ppm: u32,
    /// Safe operating intervals; deduplicated per asset/clock domain.
    pub safe_exposure: Vec<ExposureInterval>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedBlockOutcome {
    pub effective_block_id: String,
    pub scenario_family: String,
    pub source_pair_sha256: String,
    pub scenario_manifest_sha256: String,
    pub ground_truth_receipt_sha256: String,
    pub score_definition_sha256: String,
    pub calibration_reference_sha256: String,
    pub tie_protocol_sha256: String,
    pub closure_manifest_sha256: String,
    pub baseline_build_sha256: String,
    pub candidate_build_sha256: String,
    pub baseline_configuration_sha256: String,
    pub candidate_configuration_sha256: String,
    pub alpha_numerator: u128,
    pub alpha_denominator: u128,
    pub baseline: ArmOutcome,
    pub candidate: ArmOutcome,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BenchmarkError {
    EmptyEvaluation,
    ZeroHorizon,
    ZeroBootstrapReplicates,
    EmptyBlockId,
    EmptyScenarioFamily,
    DuplicateEffectiveBlock,
    InvalidDigest(&'static str),
    InvalidAlpha,
    BenchmarkContractMismatch,
    DuplicateSourcePair,
    DetectionDelayExceedsHorizon,
    ProgressOutOfRange,
    InvalidExposureInterval,
    ArithmeticOverflow,
    TooFewBlocksForInterval,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArmMetrics {
    pub blocks: usize,
    pub misses: usize,
    pub mean_penalized_delay_us: u64,
    pub timely_fallback_blocks: usize,
    pub safety_violations: u64,
    pub false_warnings: u64,
    pub false_quarantines: u64,
    pub safe_task_completions: u64,
    pub mean_useful_progress_ppm: u32,
    pub safe_exposure_us: u64,
    /// Events per hour, rounded to ppm. None when exposure is zero.
    pub false_warnings_per_hour_ppm: Option<u128>,
    pub false_quarantines_per_hour_ppm: Option<u128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairedMetrics {
    pub blocks: usize,
    /// Candidate minus baseline. Negative favours candidate.
    pub mean_penalized_delay_delta_us: i64,
    /// Percentile cluster-bootstrap interval for the paired mean delta.
    pub bootstrap_95_low_us: Option<i64>,
    pub bootstrap_95_high_us: Option<i64>,
    pub candidate_wins: usize,
    pub ties: usize,
    pub candidate_losses: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FamilyReport {
    pub baseline: ArmMetrics,
    pub candidate: ArmMetrics,
    pub paired: PairedMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkReport {
    pub horizon_us: u64,
    pub bootstrap_replicates: usize,
    pub bootstrap_seed: u64,
    pub baseline: ArmMetrics,
    pub candidate: ArmMetrics,
    pub paired: PairedMetrics,
    pub by_family: BTreeMap<String, FamilyReport>,
}

#[derive(Clone, Debug, Default)]
struct ArmAccumulator {
    misses: usize,
    delay_sum: u128,
    timely_fallback_blocks: usize,
    safety_violations: u128,
    false_warnings: u128,
    false_quarantines: u128,
    safe_task_completions: u128,
    useful_progress_sum: u128,
    safe_exposure: Vec<ExposureInterval>,
}

impl ArmAccumulator {
    fn push(&mut self, arm: &ArmOutcome, horizon_us: u64) -> Result<(), BenchmarkError> {
        let penalty = match arm.detection_delay_us {
            Some(delay) if delay <= horizon_us => delay,
            Some(_) => return Err(BenchmarkError::DetectionDelayExceedsHorizon),
            None => {
                self.misses = self
                    .misses
                    .checked_add(1)
                    .ok_or(BenchmarkError::ArithmeticOverflow)?;
                horizon_us
                    .checked_add(1)
                    .ok_or(BenchmarkError::ArithmeticOverflow)?
            }
        };
        self.delay_sum = self
            .delay_sum
            .checked_add(u128::from(penalty))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;

        if arm.fallback_completed_before_deadline {
            self.timely_fallback_blocks = self
                .timely_fallback_blocks
                .checked_add(1)
                .ok_or(BenchmarkError::ArithmeticOverflow)?;
        }
        self.safety_violations = self
            .safety_violations
            .checked_add(u128::from(arm.safety_violations))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        self.false_warnings = self
            .false_warnings
            .checked_add(u128::from(arm.false_warnings))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        self.false_quarantines = self
            .false_quarantines
            .checked_add(u128::from(arm.false_quarantines))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        self.safe_task_completions = self
            .safe_task_completions
            .checked_add(u128::from(arm.safe_task_completions))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        self.useful_progress_sum = self
            .useful_progress_sum
            .checked_add(u128::from(arm.useful_progress_ppm))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        self.safe_exposure.extend(arm.safe_exposure.iter().cloned());
        Ok(())
    }

    fn finish(
        self,
        block_count: usize,
    ) -> Result<ArmMetrics, BenchmarkError> {
        let safe_exposure_us = union_exposure_us(self.safe_exposure)?;
        let block_denominator =
            u128::try_from(block_count).map_err(|_| BenchmarkError::ArithmeticOverflow)?;
        let mean_delay = self
            .delay_sum
            .checked_add(block_denominator / 2)
            .ok_or(BenchmarkError::ArithmeticOverflow)?
            / block_denominator;
        let mean_progress = self
            .useful_progress_sum
            .checked_add(block_denominator / 2)
            .ok_or(BenchmarkError::ArithmeticOverflow)?
            / block_denominator;

        Ok(ArmMetrics {
            blocks: block_count,
            misses: self.misses,
            mean_penalized_delay_us: u64::try_from(mean_delay)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            timely_fallback_blocks: self.timely_fallback_blocks,
            safety_violations: u64::try_from(self.safety_violations)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            false_warnings: u64::try_from(self.false_warnings)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            false_quarantines: u64::try_from(self.false_quarantines)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            safe_task_completions: u64::try_from(self.safe_task_completions)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            mean_useful_progress_ppm: u32::try_from(mean_progress)
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            safe_exposure_us,
            false_warnings_per_hour_ppm: events_per_hour_ppm(
                self.false_warnings,
                safe_exposure_us,
            )?,
            false_quarantines_per_hour_ppm: events_per_hour_ppm(
                self.false_quarantines,
                safe_exposure_us,
            )?,
        })
    }
}

/// Produces a paired effective-block report. Each input block contributes one
/// pre-registered worst-case outcome per arm. Candidate-baseline delay deltas
/// are bootstrapped as matched block pairs; no episode-level resampling occurs.
pub fn evaluate_paired(
    blocks: &[PairedBlockOutcome],
    horizon_us: u64,
    bootstrap_replicates: usize,
    bootstrap_seed: u64,
) -> Result<BenchmarkReport, BenchmarkError> {
    if blocks.is_empty() {
        return Err(BenchmarkError::EmptyEvaluation);
    }
    if horizon_us == 0 {
        return Err(BenchmarkError::ZeroHorizon);
    }
    if bootstrap_replicates == 0 {
        return Err(BenchmarkError::ZeroBootstrapReplicates);
    }
    if blocks.len() < 2 {
        return Err(BenchmarkError::TooFewBlocksForInterval);
    }

    let mut seen = HashSet::with_capacity(blocks.len());
    let mut source_pairs = HashSet::with_capacity(blocks.len());
    let contract = &blocks[0];
    for block in blocks {
        validate_block(block, horizon_us)?;
        if !seen.insert(block.effective_block_id.as_str()) {
            return Err(BenchmarkError::DuplicateEffectiveBlock);
        }
        if !source_pairs.insert(block.source_pair_sha256.as_str()) {
            return Err(BenchmarkError::DuplicateSourcePair);
        }
        if block.alpha_numerator != contract.alpha_numerator
            || block.alpha_denominator != contract.alpha_denominator
            || block.score_definition_sha256 != contract.score_definition_sha256
            || block.calibration_reference_sha256 != contract.calibration_reference_sha256
            || block.tie_protocol_sha256 != contract.tie_protocol_sha256
        {
            return Err(BenchmarkError::BenchmarkContractMismatch);
        }
    }

    let mut baseline = ArmAccumulator::default();
    let mut candidate = ArmAccumulator::default();
    let mut deltas = Vec::with_capacity(blocks.len());
    let mut by_family_blocks = BTreeMap::<String, Vec<&PairedBlockOutcome>>::new();

    for block in blocks {
        baseline.push(&block.baseline, horizon_us)?;
        candidate.push(&block.candidate, horizon_us)?;
        let baseline_delay = penalized_delay(&block.baseline, horizon_us)?;
        let candidate_delay = penalized_delay(&block.candidate, horizon_us)?;
        deltas.push(
            i64::try_from(candidate_delay)
                .and_then(|candidate_value| {
                    i64::try_from(baseline_delay).map(|baseline_value| {
                        candidate_value - baseline_value
                    })
                })
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
        );
        by_family_blocks
            .entry(block.scenario_family.clone())
            .or_default()
            .push(block);
    }

    let baseline = baseline.finish(blocks.len())?;
    let candidate = candidate.finish(blocks.len())?;
    let paired = paired_metrics(&deltas, bootstrap_replicates, bootstrap_seed)?;

    let mut by_family = BTreeMap::new();
    for (family, family_blocks) in by_family_blocks {
        let mut family_baseline = ArmAccumulator::default();
        let mut family_candidate = ArmAccumulator::default();
        let mut family_deltas = Vec::with_capacity(family_blocks.len());
        for block in &family_blocks {
            family_baseline.push(&block.baseline, horizon_us)?;
            family_candidate.push(&block.candidate, horizon_us)?;
            let baseline_delay = penalized_delay(&block.baseline, horizon_us)?;
            let candidate_delay = penalized_delay(&block.candidate, horizon_us)?;
            family_deltas.push(
                i64::try_from(candidate_delay)
                    .and_then(|candidate_value| {
                        i64::try_from(baseline_delay).map(|baseline_value| {
                            candidate_value - baseline_value
                        })
                    })
                    .map_err(|_| BenchmarkError::ArithmeticOverflow)?,
            );
        }
        let count = family_blocks.len();
        by_family.insert(
            family,
            FamilyReport {
                baseline: family_baseline.finish(count)?,
                candidate: family_candidate.finish(count)?,
                paired: paired_metrics(
                    &family_deltas,
                    bootstrap_replicates,
                    bootstrap_seed ^ 0xD6E8_FEB8_6659_FD93,
                )?,
            },
        );
    }

    Ok(BenchmarkReport {
        horizon_us,
        bootstrap_replicates,
        bootstrap_seed,
        baseline,
        candidate,
        paired,
        by_family,
    })
}

fn validate_block(
    block: &PairedBlockOutcome,
    horizon_us: u64,
) -> Result<(), BenchmarkError> {
    if block.effective_block_id.trim().is_empty() {
        return Err(BenchmarkError::EmptyBlockId);
    }
    if block.scenario_family.trim().is_empty() {
        return Err(BenchmarkError::EmptyScenarioFamily);
    }
    for (digest, label) in [
        (&block.source_pair_sha256, "source_pair_sha256"),
        (&block.scenario_manifest_sha256, "scenario_manifest_sha256"),
        (&block.ground_truth_receipt_sha256, "ground_truth_receipt_sha256"),
        (&block.score_definition_sha256, "score_definition_sha256"),
        (&block.calibration_reference_sha256, "calibration_reference_sha256"),
        (&block.tie_protocol_sha256, "tie_protocol_sha256"),
        (&block.closure_manifest_sha256, "closure_manifest_sha256"),
        (&block.baseline_build_sha256, "baseline_build_sha256"),
        (&block.candidate_build_sha256, "candidate_build_sha256"),
        (&block.baseline_configuration_sha256, "baseline_configuration_sha256"),
        (&block.candidate_configuration_sha256, "candidate_configuration_sha256"),
    ] {
        validate_digest(digest, label)?;
    }
    if block.alpha_numerator == 0 || block.alpha_denominator == 0
        || block.alpha_numerator >= block.alpha_denominator
    {
        return Err(BenchmarkError::InvalidAlpha);
    }
    for arm in [&block.baseline, &block.candidate] {
        if arm.detection_delay_us.is_some_and(|delay| delay > horizon_us) {
            return Err(BenchmarkError::DetectionDelayExceedsHorizon);
        }
        if arm.useful_progress_ppm > PPM {
            return Err(BenchmarkError::ProgressOutOfRange);
        }
        for interval in &arm.safe_exposure {
            if interval.domain_id.trim().is_empty() || interval.end_us <= interval.start_us {
                return Err(BenchmarkError::InvalidExposureInterval);
            }
        }
    }
    Ok(())
}

fn validate_digest(value: &str, label: &'static str) -> Result<(), BenchmarkError> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(BenchmarkError::InvalidDigest(label))
    }
}

fn penalized_delay(
    arm: &ArmOutcome,
    horizon_us: u64,
) -> Result<u64, BenchmarkError> {
    match arm.detection_delay_us {
        Some(delay) if delay <= horizon_us => Ok(delay),
        Some(_) => Err(BenchmarkError::DetectionDelayExceedsHorizon),
        None => horizon_us
            .checked_add(1)
            .ok_or(BenchmarkError::ArithmeticOverflow),
    }
}

fn paired_metrics(
    deltas: &[i64],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<PairedMetrics, BenchmarkError> {
    if deltas.len() < 2 {
        return Err(BenchmarkError::TooFewBlocksForInterval);
    }
    let mut sum = 0i128;
    let mut wins = 0usize;
    let mut ties = 0usize;
    let mut losses = 0usize;
    for delta in deltas {
        sum = sum
            .checked_add(i128::from(*delta))
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
        match delta.cmp(&0) {
            std::cmp::Ordering::Less => wins += 1,
            std::cmp::Ordering::Equal => ties += 1,
            std::cmp::Ordering::Greater => losses += 1,
        }
    }
    let count = i128::try_from(deltas.len()).map_err(|_| BenchmarkError::ArithmeticOverflow)?;
    let mean = round_signed_ratio(sum, count)?;
    let (low, high) = if deltas.len() >= 2 {
        let (low, high) = bootstrap_interval(deltas, bootstrap_replicates, seed)?;
        (Some(low), Some(high))
    } else {
        (None, None)
    };

    Ok(PairedMetrics {
        blocks: deltas.len(),
        mean_penalized_delay_delta_us: mean,
        bootstrap_95_low_us: low,
        bootstrap_95_high_us: high,
        candidate_wins: wins,
        ties,
        candidate_losses: losses,
    })
}

fn bootstrap_interval(
    deltas: &[i64],
    replicates: usize,
    seed: u64,
) -> Result<(i64, i64), BenchmarkError> {
    let mut rng = SplitMix64::new(seed);
    let count = deltas.len();
    let count_u64 = u64::try_from(count).map_err(|_| BenchmarkError::ArithmeticOverflow)?;
    let mut means = Vec::with_capacity(replicates);

    for _ in 0..replicates {
        let mut sum = 0i128;
        for _ in 0..count {
            let index = usize::try_from(rng.below(count_u64))
                .map_err(|_| BenchmarkError::ArithmeticOverflow)?;
            sum = sum
                .checked_add(i128::from(deltas[index]))
                .ok_or(BenchmarkError::ArithmeticOverflow)?;
        }
        means.push(round_signed_ratio(
            sum,
            i128::try_from(count).map_err(|_| BenchmarkError::ArithmeticOverflow)?,
        )?);
    }
    means.sort_unstable();
    let low = percentile_nearest_rank(&means, 2, 100)?;
    let high = percentile_nearest_rank(&means, 98, 100)?;
    Ok((low, high))
}

fn percentile_nearest_rank(
    sorted: &[i64],
    numerator: usize,
    denominator: usize,
) -> Result<i64, BenchmarkError> {
    if sorted.is_empty() || numerator == 0 || numerator > denominator {
        return Err(BenchmarkError::ZeroBootstrapReplicates);
    }
    let product = sorted
        .len()
        .checked_mul(numerator)
        .ok_or(BenchmarkError::ArithmeticOverflow)?;
    let rank = product
        .checked_add(denominator - 1)
        .ok_or(BenchmarkError::ArithmeticOverflow)?
        / denominator;
    sorted
        .get(rank.saturating_sub(1))
        .copied()
        .ok_or(BenchmarkError::ArithmeticOverflow)
}

fn round_signed_ratio(numerator: i128, denominator: i128) -> Result<i64, BenchmarkError> {
    if denominator <= 0 {
        return Err(BenchmarkError::ArithmeticOverflow);
    }
    let half = denominator / 2;
    let rounded = if numerator >= 0 {
        numerator
            .checked_add(half)
            .ok_or(BenchmarkError::ArithmeticOverflow)?
            / denominator
    } else {
        numerator
            .checked_sub(half)
            .ok_or(BenchmarkError::ArithmeticOverflow)?
            / denominator
    };
    i64::try_from(rounded).map_err(|_| BenchmarkError::ArithmeticOverflow)
}

fn events_per_hour_ppm(events: u128, exposure_us: u64) -> Result<Option<u128>, BenchmarkError> {
    if exposure_us == 0 {
        return Ok(None);
    }
    let numerator = events
        .checked_mul(US_PER_HOUR)
        .ok_or(BenchmarkError::ArithmeticOverflow)?;
    Ok(Some(
        numerator
            .checked_add(u128::from(exposure_us) / 2)
            .ok_or(BenchmarkError::ArithmeticOverflow)?
            / u128::from(exposure_us),
    ))
}

fn union_exposure_us(
    mut intervals: Vec<ExposureInterval>,
) -> Result<u64, BenchmarkError> {
    intervals.sort_by(|left, right| {
        left.domain_id
            .cmp(&right.domain_id)
            .then_with(|| left.start_us.cmp(&right.start_us))
            .then_with(|| left.end_us.cmp(&right.end_us))
    });

    let mut total = 0u64;
    let mut current: Option<(String, u64, u64)> = None;
    for interval in intervals {
        let overlaps = current.as_ref().is_some_and(|(domain, _, end)| {
            domain == &interval.domain_id && interval.start_us <= *end
        });
        if overlaps {
            if let Some((_, _, end)) = current.as_mut() {
                *end = (*end).max(interval.end_us);
            }
            continue;
        }
        if let Some((_, start, end)) = current.take() {
            total = total
                .checked_add(end - start)
                .ok_or(BenchmarkError::ArithmeticOverflow)?;
        }
        current = Some((
            interval.domain_id,
            interval.start_us,
            interval.end_us,
        ));
    }
    if let Some((_, start, end)) = current {
        total = total
            .checked_add(end - start)
            .ok_or(BenchmarkError::ArithmeticOverflow)?;
    }
    Ok(total)
}

#[derive(Clone, Debug)]
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

    fn below(&mut self, upper_exclusive: u64) -> u64 {
        let threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return value % upper_exclusive;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn arm(delay: Option<u64>) -> ArmOutcome {
        ArmOutcome {
            detection_delay_us: delay,
            fallback_completed_before_deadline: true,
            safety_violations: 0,
            false_warnings: 0,
            false_quarantines: 0,
            safe_task_completions: 10,
            useful_progress_ppm: 900_000,
            safe_exposure: vec![ExposureInterval {
                domain_id: "vehicle-1".into(),
                start_us: 0,
                end_us: 3_600_000_000,
            }],
        }
    }

    fn block(id: &str, family: &str, baseline: Option<u64>, candidate: Option<u64>) -> PairedBlockOutcome {
        PairedBlockOutcome {
            effective_block_id: id.into(),
            scenario_family: family.into(),
            source_pair_sha256: format!(
                "{:064x}",
                id.bytes().fold(0u64, |accumulator, byte| {
                    accumulator.wrapping_mul(257).wrapping_add(u64::from(byte))
                })
            ),
            scenario_manifest_sha256: SHA_B.into(),
            ground_truth_receipt_sha256: SHA_A.into(),
            score_definition_sha256: SHA_C.into(),
            calibration_reference_sha256: SHA_B.into(),
            tie_protocol_sha256: SHA_C.into(),
            closure_manifest_sha256: SHA_A.into(),
            baseline_build_sha256: SHA_B.into(),
            candidate_build_sha256: SHA_C.into(),
            baseline_configuration_sha256: SHA_B.into(),
            candidate_configuration_sha256: SHA_C.into(),
            alpha_numerator: 1,
            alpha_denominator: 20,
            baseline: arm(baseline),
            candidate: arm(candidate),
        }
    }

    #[test]
    fn misses_are_penalized_with_horizon_plus_one() {
        let report = evaluate_paired(
            &[
                block("b1", "urban", Some(10), None),
                block("b2", "urban", Some(10), Some(11)),
            ],
            100,
            100,
            7,
        )
        .unwrap();
        assert_eq!(report.candidate.misses, 1);
        assert_eq!(report.candidate.mean_penalized_delay_us, 56);
        assert_eq!(report.baseline.mean_penalized_delay_us, 10);
        assert!(report.paired.mean_penalized_delay_delta_us > 0);
    }

    #[test]
    fn paired_delay_difference_favours_faster_candidate() {
        let report = evaluate_paired(
            &[
                block("b1", "urban", Some(100), Some(50)),
                block("b2", "urban", Some(120), Some(70)),
                block("b3", "rural", Some(110), Some(90)),
            ],
            500,
            500,
            42,
        )
        .unwrap();
        assert!(report.paired.mean_penalized_delay_delta_us < 0);
        assert_eq!(report.paired.candidate_wins, 3);
        assert_eq!(report.paired.candidate_losses, 0);
        assert!(report.paired.bootstrap_95_high_us.unwrap() < 0);
    }

    #[test]
    fn bootstrap_is_deterministic_for_a_frozen_seed() {
        let blocks = [
            block("b1", "urban", Some(100), Some(50)),
            block("b2", "urban", Some(120), Some(70)),
            block("b3", "rural", Some(110), Some(90)),
        ];
        let first = evaluate_paired(&blocks, 500, 500, 42).unwrap();
        let second = evaluate_paired(&blocks, 500, 500, 42).unwrap();
        assert_eq!(first.paired, second.paired);
    }

    #[test]
    fn family_breakdown_preserves_regressions() {
        let report = evaluate_paired(
            &[
                block("b1", "urban", Some(100), Some(50)),
                block("b2", "rural", Some(100), Some(180)),
            ],
            500,
            200,
            42,
        )
        .unwrap();
        assert!(report.by_family["urban"].paired.mean_penalized_delay_delta_us < 0);
        assert!(report.by_family["rural"].paired.mean_penalized_delay_delta_us > 0);
        assert_eq!(report.by_family["urban"].paired.bootstrap_95_low_us, None);
        assert_eq!(report.by_family["rural"].paired.bootstrap_95_high_us, None);
    }

    #[test]
    fn false_alarm_rates_use_union_exposure_not_sum_of_overlapping_clips() {
        let mut first = block("b1", "urban", Some(10), Some(10));
        let mut second = block("b2", "urban", Some(10), Some(10));
        first.baseline.false_quarantines = 1;
        first.candidate.false_quarantines = 2;
        second.baseline.false_quarantines = 1;
        second.candidate.false_quarantines = 2;
        second.baseline.safe_exposure[0].start_us = 1_800_000_000;
        second.candidate.safe_exposure[0].start_us = 1_800_000_000;
        let report = evaluate_paired(&[first, second], 100, 100, 9).unwrap();
        assert_eq!(report.baseline.safe_exposure_us, 5_400_000_000);
        assert_eq!(report.candidate.safe_exposure_us, 5_400_000_000);
        assert_eq!(report.baseline.false_quarantines, 2);
        assert_eq!(report.candidate.false_quarantines, 4);
        assert_eq!(report.baseline.false_quarantines_per_hour_ppm, Some(1_333_333));
    }

    #[test]
    fn duplicate_effective_blocks_are_rejected() {
        assert_eq!(
            evaluate_paired(
                &[block("same", "urban", Some(10), Some(10)), block("same", "rural", Some(9), Some(8))],
                100,
                50,
                1
            ),
            Err(BenchmarkError::DuplicateEffectiveBlock)
        );
    }

    #[test]
    fn alpha_calibration_reference_and_tie_contract_must_match_across_blocks() {
        let mut blocks = [
            block("b1", "urban", Some(10), Some(10)),
            block("b2", "urban", Some(10), Some(9)),
        ];
        blocks[1].alpha_numerator = 1;
        blocks[1].alpha_denominator = 10;
        assert_eq!(
            evaluate_paired(&blocks, 100, 50, 1),
            Err(BenchmarkError::BenchmarkContractMismatch)
        );

        blocks[1] = block("b2", "urban", Some(10), Some(9));
        blocks[1].calibration_reference_sha256 = SHA_A.into();
        assert_eq!(
            evaluate_paired(&blocks, 100, 50, 1),
            Err(BenchmarkError::BenchmarkContractMismatch)
        );

        blocks[1] = block("b2", "urban", Some(10), Some(9));
        blocks[1].tie_protocol_sha256 = SHA_A.into();
        assert_eq!(
            evaluate_paired(&blocks, 100, 50, 1),
            Err(BenchmarkError::BenchmarkContractMismatch)
        );
    }

    #[test]
    fn duplicate_source_pair_is_rejected() {
        let first = block("b1", "urban", Some(10), Some(9));
        let mut second = block("b2", "urban", Some(10), Some(8));
        second.source_pair_sha256 = first.source_pair_sha256.clone();
        assert_eq!(
            evaluate_paired(&[first, second], 100, 50, 1),
            Err(BenchmarkError::DuplicateSourcePair)
        );
    }

    #[test]
    fn invalid_alpha_binding_is_rejected() {
        let mut first = block("b1", "urban", Some(10), Some(9));
        let second = block("b2", "urban", Some(10), Some(8));
        first.alpha_numerator = 1;
        first.alpha_denominator = 1;
        assert_eq!(
            evaluate_paired(&[first, second], 100, 50, 1),
            Err(BenchmarkError::InvalidAlpha)
        );
    }

    #[test]
    fn malformed_digests_and_invalid_progress_are_rejected() {
        let mut bad_digest = block("b1", "urban", Some(10), Some(10));
        bad_digest.source_pair_sha256 = "bad".into();
        assert_eq!(
            evaluate_paired(&[bad_digest, block("b2", "urban", Some(10), Some(10))], 100, 50, 1),
            Err(BenchmarkError::InvalidDigest("source_pair_sha256"))
        );

        let mut bad_progress = block("b1", "urban", Some(10), Some(10));
        bad_progress.candidate.useful_progress_ppm = PPM + 1;
        assert_eq!(
            evaluate_paired(&[bad_progress, block("b2", "urban", Some(10), Some(10))], 100, 50, 1),
            Err(BenchmarkError::ProgressOutOfRange)
        );
    }

    #[test]
    fn horizon_and_exposure_bounds_are_enforced() {
        let bad_delay = block("b1", "urban", Some(101), Some(10));
        assert_eq!(
            evaluate_paired(&[bad_delay, block("b2", "urban", Some(10), Some(10))], 100, 50, 1),
            Err(BenchmarkError::DetectionDelayExceedsHorizon)
        );

        let mut bad_exposure = block("b1", "urban", Some(10), Some(10));
        bad_exposure.candidate.safe_exposure[0].end_us = 0;
        assert_eq!(
            evaluate_paired(&[bad_exposure, block("b2", "urban", Some(10), Some(10))], 100, 50, 1),
            Err(BenchmarkError::InvalidExposureInterval)
        );
    }

    #[test]
    fn no_safe_exposure_yields_no_invented_false_alarm_rate() {
        let mut first = block("b1", "hazard", Some(10), Some(10));
        let mut second = block("b2", "hazard", Some(12), Some(11));
        first.baseline.safe_exposure.clear();
        first.candidate.safe_exposure.clear();
        second.baseline.safe_exposure.clear();
        second.candidate.safe_exposure.clear();
        let report = evaluate_paired(&[first, second], 100, 100, 4).unwrap();
        assert_eq!(report.baseline.false_quarantines_per_hour_ppm, None);
    }
}

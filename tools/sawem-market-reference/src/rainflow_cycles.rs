//! Source-bound ASTM E1049-style three-point rainflow counting for BESS SOC.
//!
//! This module identifies SOC reversals, applies the ASTM E1049 three-point
//! counting convention and reports residual half-cycles. It preserves source
//! sample indices and timestamps for every counted range. It does not convert
//! cycle counts into capacity fade or warranty damage; that requires a
//! chemistry- and product-specific, source-bound degradation curve.
//!
//! The caller supplies a maximum admissible sample gap. A larger gap splits
//! the history into independent segments; no cycle is inferred across a gap.
//! The trace is then marked incomplete, and the returned counts are partial.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainflowMethod {
    AstmE1049ThreePointV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainflowCycleKind {
    Full,
    Half,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SocTelemetrySample<'a> {
    pub sample_id: &'a str,
    pub source_ref: &'a str,
    pub source_sha256: &'a str,
    pub sample_time_utc: i64,
    pub observed_at_utc: i64,
    pub available_at_utc: i64,
    pub soc_bps: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RainflowInputs<'a> {
    pub scenario_id: &'a str,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    pub maximum_sample_gap_seconds: u64,
    pub sampling_policy_source_ref: &'a str,
    pub sampling_policy_source_sha256: &'a str,
    pub samples: &'a [SocTelemetrySample<'a>],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RainflowError {
    InvalidEvidenceReference,
    InvalidSourceDigest,
    InvalidHorizon,
    InvalidSampleGap,
    InvalidSample,
    FutureKnownSample,
    SamplesOutOfOrder,
    DuplicateSampleId,
    DuplicateSampleSourceReference,
    ArithmeticOverflow,
    EmptyTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RainflowCycle {
    pub kind: RainflowCycleKind,
    pub range_bps: u16,
    /// Sum of endpoint SOC values. Divide by 2 for exact mean SOC in bps.
    pub mean_soc_sum_bps: u32,
    pub start_sample_index: usize,
    pub end_sample_index: usize,
    pub start_time_utc: i64,
    pub end_time_utc: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RainflowAssessment {
    pub scenario_id: String,
    pub method: RainflowMethod,
    pub evaluation_as_of_utc: i64,
    pub horizon_start_utc: i64,
    pub horizon_end_utc: i64,
    pub maximum_sample_gap_seconds: u64,
    pub sampling_policy_source_ref: String,
    pub sampling_policy_source_sha256: String,
    pub sample_count: u64,
    pub reversal_count: u64,
    pub segment_count: u64,
    pub gap_count: u64,
    pub uncovered_seconds: u64,
    pub trace_complete: bool,
    pub full_cycle_count: u64,
    pub half_cycle_count: u64,
    /// Equivalent full cycles = this numerator / 2.
    pub equivalent_full_cycle_half_units: u128,
    /// Sum(range_bps * cycle_count_half_units). This is a wear-analysis
    /// weighting input, not a degradation or warranty-damage result.
    pub depth_weighted_half_units_bps: u128,
    pub maximum_cycle_depth_bps: u16,
    pub cycles: Vec<RainflowCycle>,
}

#[derive(Debug, Clone, Copy)]
struct Reversal {
    sample_index: usize,
    soc_bps: u16,
    time_utc: i64,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn cycle_between(
    start: Reversal,
    end: Reversal,
    kind: RainflowCycleKind,
) -> RainflowCycle {
    let range = start.soc_bps.abs_diff(end.soc_bps);
    RainflowCycle {
        kind,
        range_bps: range,
        mean_soc_sum_bps: u32::from(start.soc_bps) + u32::from(end.soc_bps),
        start_sample_index: start.sample_index,
        end_sample_index: end.sample_index,
        start_time_utc: start.time_utc,
        end_time_utc: end.time_utc,
    }
}

/// Collapse plateaus to their final sample, then retain endpoints and local
/// extrema. This deterministic tie rule prevents a flat SOC plateau from
/// creating zero-depth cycles.
fn extract_reversals(
    samples: &[SocTelemetrySample<'_>],
    start_index: usize,
    end_index_exclusive: usize,
) -> Vec<Reversal> {
    let mut compact = Vec::<usize>::new();
    for index in start_index..end_index_exclusive {
        if let Some(last) = compact.last_mut() {
            if samples[*last].soc_bps == samples[index].soc_bps {
                *last = index;
                continue;
            }
        }
        compact.push(index);
    }

    if compact.is_empty() {
        return Vec::new();
    }
    if compact.len() == 1 {
        let sample = samples[compact[0]];
        return vec![Reversal {
            sample_index: compact[0],
            soc_bps: sample.soc_bps,
            time_utc: sample.sample_time_utc,
        }];
    }

    let mut reversals = Vec::new();
    let first = samples[compact[0]];
    reversals.push(Reversal {
        sample_index: compact[0],
        soc_bps: first.soc_bps,
        time_utc: first.sample_time_utc,
    });

    for position in 1..compact.len() - 1 {
        let previous = samples[compact[position - 1]].soc_bps;
        let current = samples[compact[position]].soc_bps;
        let next = samples[compact[position + 1]].soc_bps;
        let previous_direction = current.cmp(&previous);
        let next_direction = next.cmp(&current);
        if previous_direction != next_direction {
            let sample = samples[compact[position]];
            reversals.push(Reversal {
                sample_index: compact[position],
                soc_bps: sample.soc_bps,
                time_utc: sample.sample_time_utc,
            });
        }
    }

    let last_index = *compact.last().expect("compact sequence is non-empty");
    let last = samples[last_index];
    reversals.push(Reversal {
        sample_index: last_index,
        soc_bps: last.soc_bps,
        time_utc: last.sample_time_utc,
    });
    reversals
}

/// Count one contiguous reversal segment using the ASTM E1049 three-point
/// rule. Completed inner cycles are full cycles; residue ranges are half
/// cycles. The returned indices refer to the original input sample array.
fn count_reversal_segment(
    reversals: &[Reversal],
    cycles: &mut Vec<RainflowCycle>,
    full_count: &mut u64,
    half_count: &mut u64,
    depth_weighted_half_units: &mut u128,
    maximum_depth: &mut u16,
) -> Result<(), RainflowError> {
    let mut stack = Vec::<Reversal>::new();

    for reversal in reversals.iter().copied() {
        stack.push(reversal);
        loop {
            if stack.len() < 3 {
                break;
            }
            let length = stack.len();
            let y_start = stack[length - 3];
            let y_end = stack[length - 2];
            let x_end = stack[length - 1];
            let y_range = y_start.soc_bps.abs_diff(y_end.soc_bps);
            let x_range = y_end.soc_bps.abs_diff(x_end.soc_bps);
            if x_range < y_range {
                break;
            }

            let kind = if length == 3 {
                RainflowCycleKind::Half
            } else {
                RainflowCycleKind::Full
            };
            cycles.push(cycle_between(y_start, y_end, kind));
            *maximum_depth = (*maximum_depth).max(y_range);
            let count_half_units = match kind {
                RainflowCycleKind::Full => {
                    *full_count = full_count
                        .checked_add(1)
                        .ok_or(RainflowError::ArithmeticOverflow)?;
                    2_u128
                }
                RainflowCycleKind::Half => {
                    *half_count = half_count
                        .checked_add(1)
                        .ok_or(RainflowError::ArithmeticOverflow)?;
                    1_u128
                }
            };
            let weighted = u128::from(y_range)
                .checked_mul(count_half_units)
                .ok_or(RainflowError::ArithmeticOverflow)?;
            *depth_weighted_half_units = depth_weighted_half_units
                .checked_add(weighted)
                .ok_or(RainflowError::ArithmeticOverflow)?;

            if length == 3 {
                stack.remove(0);
            } else {
                stack.drain(length - 3..length - 1);
            }
        }
    }

    for pair in stack.windows(2) {
        let cycle = cycle_between(pair[0], pair[1], RainflowCycleKind::Half);
        let range = cycle.range_bps;
        cycles.push(cycle);
        *half_count = half_count
            .checked_add(1)
            .ok_or(RainflowError::ArithmeticOverflow)?;
        *maximum_depth = (*maximum_depth).max(range);
        *depth_weighted_half_units = depth_weighted_half_units
            .checked_add(u128::from(range))
            .ok_or(RainflowError::ArithmeticOverflow)?;
    }
    Ok(())
}

/// Count source-bound SOC cycles. If a telemetry gap exceeds the explicitly
/// supplied maximum, the trace is split and the result is marked incomplete.
/// No cycle is inferred across the missing interval.
pub fn count_soc_rainflow_cycles(
    inputs: RainflowInputs<'_>,
) -> Result<RainflowAssessment, RainflowError> {
    if !nonempty(inputs.scenario_id)
        || !nonempty(inputs.sampling_policy_source_ref)
    {
        return Err(RainflowError::InvalidEvidenceReference);
    }
    if !valid_sha256(inputs.sampling_policy_source_sha256) {
        return Err(RainflowError::InvalidSourceDigest);
    }
    if inputs.horizon_start_utc >= inputs.horizon_end_utc
        || inputs.evaluation_as_of_utc < inputs.horizon_end_utc
    {
        return Err(RainflowError::InvalidHorizon);
    }
    if inputs.maximum_sample_gap_seconds == 0 {
        return Err(RainflowError::InvalidSampleGap);
    }
    if inputs.samples.is_empty() {
        return Err(RainflowError::EmptyTrace);
    }

    let mut sample_ids = std::collections::BTreeSet::new();
    let mut source_refs = std::collections::BTreeSet::new();
    for (index, sample) in inputs.samples.iter().copied().enumerate() {
        if !nonempty(sample.sample_id) || !nonempty(sample.source_ref) {
            return Err(RainflowError::InvalidEvidenceReference);
        }
        if !valid_sha256(sample.source_sha256) {
            return Err(RainflowError::InvalidSourceDigest);
        }
        if sample.sample_time_utc < inputs.horizon_start_utc
            || sample.sample_time_utc > inputs.horizon_end_utc
            || sample.observed_at_utc < sample.sample_time_utc
            || sample.observed_at_utc > sample.available_at_utc
            || sample.soc_bps > 10_000
        {
            return Err(RainflowError::InvalidSample);
        }
        if sample.available_at_utc > inputs.evaluation_as_of_utc {
            return Err(RainflowError::FutureKnownSample);
        }
        if !sample_ids.insert(sample.sample_id) {
            return Err(RainflowError::DuplicateSampleId);
        }
        if !source_refs.insert(sample.source_ref) {
            return Err(RainflowError::DuplicateSampleSourceReference);
        }
        if index > 0 && sample.sample_time_utc <= inputs.samples[index - 1].sample_time_utc {
            return Err(RainflowError::SamplesOutOfOrder);
        }
    }
    if inputs.samples[0].sample_time_utc != inputs.horizon_start_utc
        || inputs.samples[inputs.samples.len() - 1].sample_time_utc != inputs.horizon_end_utc
    {
        return Err(RainflowError::InvalidHorizon);
    }

    let mut cycles = Vec::new();
    let mut full_cycle_count = 0_u64;
    let mut half_cycle_count = 0_u64;
    let mut depth_weighted_half_units = 0_u128;
    let mut maximum_cycle_depth_bps = 0_u16;
    let mut gap_count = 0_u64;
    let mut uncovered_seconds = 0_u64;
    let mut segment_count = 0_u64;
    let mut reversal_count = 0_u64;
    let mut segment_start = 0_usize;

    for index in 1..inputs.samples.len() {
        let gap = inputs.samples[index]
            .sample_time_utc
            .checked_sub(inputs.samples[index - 1].sample_time_utc)
            .ok_or(RainflowError::ArithmeticOverflow)?;
        let gap_u64 = u64::try_from(gap).map_err(|_| RainflowError::ArithmeticOverflow)?;
        if gap_u64 > inputs.maximum_sample_gap_seconds {
            gap_count = gap_count
                .checked_add(1)
                .ok_or(RainflowError::ArithmeticOverflow)?;
            uncovered_seconds = uncovered_seconds
                .checked_add(gap_u64 - inputs.maximum_sample_gap_seconds)
                .ok_or(RainflowError::ArithmeticOverflow)?;
            let reversals = extract_reversals(inputs.samples, segment_start, index);
            reversal_count = reversal_count
                .checked_add(
                    u64::try_from(reversals.len())
                        .map_err(|_| RainflowError::ArithmeticOverflow)?,
                )
                .ok_or(RainflowError::ArithmeticOverflow)?;
            count_reversal_segment(
                &reversals,
                &mut cycles,
                &mut full_cycle_count,
                &mut half_cycle_count,
                &mut depth_weighted_half_units,
                &mut maximum_cycle_depth_bps,
            )?;
            segment_count = segment_count
                .checked_add(1)
                .ok_or(RainflowError::ArithmeticOverflow)?;
            segment_start = index;
        }
    }

    let reversals = extract_reversals(
        inputs.samples,
        segment_start,
        inputs.samples.len(),
    );
    reversal_count = reversal_count
        .checked_add(
            u64::try_from(reversals.len()).map_err(|_| RainflowError::ArithmeticOverflow)?,
        )
        .ok_or(RainflowError::ArithmeticOverflow)?;
    count_reversal_segment(
        &reversals,
        &mut cycles,
        &mut full_cycle_count,
        &mut half_cycle_count,
        &mut depth_weighted_half_units,
        &mut maximum_cycle_depth_bps,
    )?;
    segment_count = segment_count
        .checked_add(1)
        .ok_or(RainflowError::ArithmeticOverflow)?;

    let equivalent_full_cycle_half_units = u128::from(full_cycle_count)
        .checked_mul(2)
        .and_then(|full| full.checked_add(u128::from(half_cycle_count)))
        .ok_or(RainflowError::ArithmeticOverflow)?;

    Ok(RainflowAssessment {
        scenario_id: inputs.scenario_id.to_owned(),
        method: RainflowMethod::AstmE1049ThreePointV1,
        evaluation_as_of_utc: inputs.evaluation_as_of_utc,
        horizon_start_utc: inputs.horizon_start_utc,
        horizon_end_utc: inputs.horizon_end_utc,
        maximum_sample_gap_seconds: inputs.maximum_sample_gap_seconds,
        sampling_policy_source_ref: inputs.sampling_policy_source_ref.to_owned(),
        sampling_policy_source_sha256: inputs.sampling_policy_source_sha256.to_owned(),
        sample_count: u64::try_from(inputs.samples.len())
            .map_err(|_| RainflowError::ArithmeticOverflow)?,
        reversal_count,
        segment_count,
        gap_count,
        uncovered_seconds,
        trace_complete: gap_count == 0,
        full_cycle_count,
        half_cycle_count,
        equivalent_full_cycle_half_units,
        depth_weighted_half_units_bps: depth_weighted_half_units,
        maximum_cycle_depth_bps,
        cycles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SOURCE_SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn sample(
        id: &'static str,
        time: i64,
        soc_bps: u16,
    ) -> SocTelemetrySample<'static> {
        SocTelemetrySample {
            sample_id: id,
            source_ref: id,
            source_sha256: SOURCE_SHA,
            sample_time_utc: time,
            observed_at_utc: time,
            available_at_utc: time + 1,
            soc_bps,
        }
    }

    fn inputs(samples: &[SocTelemetrySample<'_>]) -> RainflowInputs<'_> {
        RainflowInputs {
            scenario_id: "battery-001/soc-cycle-test",
            evaluation_as_of_utc: 10_000,
            horizon_start_utc: samples[0].sample_time_utc,
            horizon_end_utc: samples[samples.len() - 1].sample_time_utc,
            maximum_sample_gap_seconds: 2,
            sampling_policy_source_ref: "bms/sampling-policy/v1",
            sampling_policy_source_sha256: SOURCE_SHA,
            samples,
        }
    }

    #[test]
    fn matches_reference_astm_three_point_cycle_sequence() {
        let soc = [-2_i16, 1, -3, 5, -1, 3, -4, 4, -3, 1, -2, 3, 2, 6];
        let samples = soc
            .iter()
            .enumerate()
            .map(|(index, value)| {
                sample(
                    match index {
                        0 => "s00",
                        1 => "s01",
                        2 => "s02",
                        3 => "s03",
                        4 => "s04",
                        5 => "s05",
                        6 => "s06",
                        7 => "s07",
                        8 => "s08",
                        9 => "s09",
                        10 => "s10",
                        11 => "s11",
                        12 => "s12",
                        _ => "s13",
                    },
                    index as i64,
                    u16::try_from((*value + 4) * 1_000).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let result = count_soc_rainflow_cycles(inputs(&samples)).unwrap();
        assert_eq!(result.method, RainflowMethod::AstmE1049ThreePointV1);
        assert!(result.trace_complete);
        assert_eq!(result.reversal_count, 14);
        assert_eq!(result.full_cycle_count, 4);
        assert_eq!(result.half_cycle_count, 5);
        assert_eq!(result.equivalent_full_cycle_half_units, 13);
        let observed = result
            .cycles
            .iter()
            .map(|cycle| {
                (
                    cycle.kind,
                    cycle.range_bps,
                    cycle.start_sample_index,
                    cycle.end_sample_index,
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            observed,
            vec![
                (RainflowCycleKind::Half, 3_000, 0, 1),
                (RainflowCycleKind::Half, 4_000, 1, 2),
                (RainflowCycleKind::Full, 4_000, 4, 5),
                (RainflowCycleKind::Half, 8_000, 2, 3),
                (RainflowCycleKind::Full, 3_000, 9, 10),
                (RainflowCycleKind::Full, 1_000, 11, 12),
                (RainflowCycleKind::Full, 7_000, 7, 8),
                (RainflowCycleKind::Half, 9_000, 3, 6),
                (RainflowCycleKind::Half, 10_000, 6, 13),
            ]
        );
    }

    #[test]
    fn flat_plateaus_and_monotone_traces_do_not_create_zero_depth_cycles() {
        let samples = [
            sample("a", 0, 2_000),
            sample("b", 1, 5_000),
            sample("c", 2, 5_000),
            sample("d", 3, 5_000),
            sample("e", 4, 8_000),
        ];
        let result = count_soc_rainflow_cycles(inputs(&samples)).unwrap();
        assert_eq!(result.reversal_count, 2);
        assert_eq!(result.cycles.len(), 1);
        assert_eq!(result.cycles[0].range_bps, 6_000);
        assert_eq!(result.cycles[0].kind, RainflowCycleKind::Half);
        assert_eq!(result.cycles[0].start_sample_index, 0);
        assert_eq!(result.cycles[0].end_sample_index, 4);
    }

    #[test]
    fn telemetry_gap_splits_the_history_and_withholds_complete_trace() {
        let samples = [
            sample("a", 0, 2_000),
            sample("b", 1, 8_000),
            sample("c", 2, 2_000),
            sample("d", 10, 8_000),
            sample("e", 11, 2_000),
        ];
        let result = count_soc_rainflow_cycles(inputs(&samples)).unwrap();
        assert!(!result.trace_complete);
        assert_eq!(result.gap_count, 1);
        assert_eq!(result.uncovered_seconds, 6);
        assert_eq!(result.segment_count, 2);
        assert_eq!(result.cycles.len(), 3);
        assert!(result
            .cycles
            .iter()
            .all(|cycle| cycle.kind == RainflowCycleKind::Half));
    }

    #[test]
    fn future_known_duplicate_and_bad_source_evidence_fail_closed() {
        let future = [
            SocTelemetrySample {
                available_at_utc: 10_001,
                ..sample("future-start", 0, 5_000)
            },
            horizon[0],
        ];
        assert_eq!(
            count_soc_rainflow_cycles(inputs(&future)),
            Err(RainflowError::FutureKnownSample)
        );

        let duplicate_source = [
            sample("same-source", 0, 2_000),
            SocTelemetrySample {
                source_ref: "same-source",
                ..sample("different-id", 1, 8_000)
            },
        ];
        assert_eq!(
            count_soc_rainflow_cycles(inputs(&duplicate_source)),
            Err(RainflowError::DuplicateSampleSourceReference)
        );

        let duplicate_id = [
            sample("same-id", 0, 2_000),
            SocTelemetrySample {
                source_ref: "other-source",
                ..sample("same-id", 1, 8_000)
            },
        ];
        assert_eq!(
            count_soc_rainflow_cycles(inputs(&duplicate_id)),
            Err(RainflowError::DuplicateSampleId)
        );

        let bad_digest = [
            SocTelemetrySample {
                source_sha256: "bad",
                ..sample("bad", 0, 2_000)
            },
            sample("last", 1, 8_000),
        ];
        assert_eq!(
            count_soc_rainflow_cycles(inputs(&bad_digest)),
            Err(RainflowError::InvalidSourceDigest)
        );
    }

    #[test]
    fn invalid_soc_and_nonmonotonic_timestamps_are_rejected() {
        let invalid_soc = [
            SocTelemetrySample {
                soc_bps: 10_001,
                ..sample("invalid", 0, 5_000)
            },
            sample("last", 1, 6_000),
        ];
        assert_eq!(
            count_soc_rainflow_cycles(inputs(&invalid_soc)),
            Err(RainflowError::InvalidSample)
        );

        let out_of_order = [
            sample("first", 0, 2_000),
            sample("second", 1, 8_000),
        ];
        let reversed = [out_of_order[1], out_of_order[0]];
        let invalid = RainflowInputs {
            samples: &reversed,
            ..inputs(&out_of_order)
        };
        assert_eq!(
            count_soc_rainflow_cycles(invalid),
            Err(RainflowError::SamplesOutOfOrder)
        );
    }
}

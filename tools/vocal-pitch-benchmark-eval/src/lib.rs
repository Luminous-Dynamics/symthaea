//! Score-independent evaluation of monophonic pitch tracker outputs.
//!
//! This crate consumes aligned reference and estimated F0 frames. It does not
//! read audio, run pitch trackers, infer voicing thresholds, align recordings,
//! or generate confidence intervals. Sampling and inference must respect the
//! singer/song/session clustering carried in each segment manifest.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const PITCH_BENCHMARK_SCHEMA_V1: &str = "vocal-pitch-benchmark/v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PitchBenchmarkError {
    pub message: String,
}

impl PitchBenchmarkError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for PitchBenchmarkError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for PitchBenchmarkError {}

/// Exact algorithm, data and preprocessing identity for one tracker run.
///
/// The voicing policy identifies the frozen rule that converts tracker
/// confidence into a voiced/unvoiced decision. The confidence threshold must
/// be selected on a disjoint calibration split before this test split is used.
#[derive(Clone, Debug, PartialEq)]
pub struct PitchBenchmarkSpecV1 {
    pub tracker_id: String,
    pub tracker_version: String,
    pub tracker_artifact_sha256: String,
    pub reference_suite_id: String,
    pub reference_suite_version: String,
    pub reference_manifest_sha256: String,
    pub evaluation_split_id: String,
    pub preprocessing_manifest_sha256: String,
    pub frame_alignment_policy_id: String,
    pub voicing_policy_id: String,
    pub frame_period_us: u32,
    pub gross_pitch_error_threshold_cents: f64,
    pub octave_error_tolerance_cents: f64,
    /// The primary independent-pitch track must not use score pitch as an
    /// estimator search prior. Score-conditioned trackers require a separate
    /// explicitly labelled evaluation track.
    pub uses_score_prior: bool,
}

/// Singer/song/session identities are preserved for cluster-aware inference.
/// Frames and segments from one singer or recording session are not
/// independent statistical sampling units.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PitchSegmentMetadataV1 {
    pub segment_id: String,
    pub corpus_id: String,
    pub singer_id: String,
    pub song_id: String,
    pub recording_session_id: String,
    pub recording_condition_id: String,
    pub language_id: String,
    pub register_id: String,
    pub technique_id: String,
    pub snr_stratum_id: String,
    pub source_audio_sha256: String,
}

/// One frame on the frozen common frame grid.
///
/// None means unvoiced/abstained after the tracker-specific voicing policy.
/// A non-None frequency must be finite and strictly positive.
#[derive(Clone, Debug, PartialEq)]
pub struct PitchFrameV1 {
    pub frame_index: u32,
    pub reference_f0_hz: Option<f64>,
    pub estimated_f0_hz: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchSegmentInputV1 {
    pub metadata: PitchSegmentMetadataV1,
    pub frames: Vec<PitchFrameV1>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PitchCountsV1 {
    pub frame_count: u64,
    pub reference_voiced_frames: u64,
    pub estimated_voiced_frames: u64,
    pub jointly_voiced_frames: u64,
    pub false_negative_voicing_frames: u64,
    pub false_positive_voicing_frames: u64,
    pub gross_pitch_error_frames: u64,
    pub octave_error_frames: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchRatesV1 {
    /// FFE@threshold: voicing mismatch OR a gross pitch error, divided by all
    /// evaluated frames. The threshold is fixed in the benchmark specification.
    pub frame_error_rate: f64,
    /// GPE@threshold: gross pitch errors divided by jointly voiced frames.
    /// None when no jointly voiced frame exists.
    pub gross_pitch_error_rate: Option<f64>,
    /// VDE: voicing mismatches divided by all evaluated frames.
    pub voicing_decision_error_rate: f64,
    /// Jointly voiced estimates divided by reference-voiced frames.
    pub reference_voiced_coverage: Option<f64>,
    /// Reference-voiced frames where the estimator abstained/unvoiced.
    pub abstention_rate: Option<f64>,
    /// Estimated-voiced frames where the reference is unvoiced.
    pub false_voiced_rate: Option<f64>,
    /// Octave/subharmonic errors among jointly voiced frames.
    pub octave_error_rate: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchSegmentResultV1 {
    pub metadata: PitchSegmentMetadataV1,
    pub counts: PitchCountsV1,
    pub rates: PitchRatesV1,
    /// Error = 1200 * log2(estimated_f0 / reference_f0), computed only where
    /// both the reference and estimate are voiced.
    pub median_signed_cents_error: Option<f64>,
    pub median_absolute_cents_error: Option<f64>,
    pub p95_absolute_cents_error: Option<f64>,
    pub maximum_absolute_cents_error: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchConditionGroupV1 {
    pub corpus_id: String,
    pub recording_condition_id: String,
    pub segment_count: usize,
    pub counts: PitchCountsV1,
    pub rates: PitchRatesV1,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PitchBenchmarkReportV1 {
    pub schema: String,
    pub spec: PitchBenchmarkSpecV1,
    pub segment_count: usize,
    pub total_counts: PitchCountsV1,
    /// Micro metrics: counts pooled over all frames. Use the group-level
    /// macro metrics below when reproducing equal corpus/condition weighting.
    pub micro_rates: PitchRatesV1,
    pub median_segment_frame_error_rate: f64,
    pub median_segment_voicing_decision_error_rate: f64,
    /// Equal weight across corpus x recording-condition groups. This mirrors
    /// the external benchmark's group weighting but does not turn this
    /// singing-only run into the mixed-corpus leaderboard.
    pub macro_group_frame_error_rate: f64,
    pub macro_group_voicing_decision_error_rate: f64,
    /// Only groups with jointly voiced reference/estimate frames contribute.
    pub macro_group_gross_pitch_error_rate: Option<f64>,
    pub macro_group_gpe_defined_groups: usize,
    pub groups: Vec<PitchConditionGroupV1>,
    pub segments: Vec<PitchSegmentResultV1>,
}

/// Evaluate already aligned tracker estimates against independent reference F0.
///
/// Each segment must contain an exact contiguous frame sequence beginning at
/// frame zero. This function deliberately does not resample, time-warp,
/// interpolate missing frames, normalize audio, or tune voicing thresholds.
pub fn evaluate_pitch_benchmark(
    spec: &PitchBenchmarkSpecV1,
    segments: &[PitchSegmentInputV1],
) -> Result<PitchBenchmarkReportV1, PitchBenchmarkError> {
    validate_spec(spec)?;
    if segments.is_empty() {
        return Err(PitchBenchmarkError::new(
            "at least one held-out singing segment is required",
        ));
    }

    let mut seen_segments = BTreeSet::new();
    let mut segment_results = Vec::with_capacity(segments.len());
    let mut total_counts = PitchCountsV1::default();
    let mut grouped: BTreeMap<(String, String), (PitchCountsV1, usize)> = BTreeMap::new();

    for segment in segments {
        validate_metadata(&segment.metadata)?;
        if !seen_segments.insert(segment.metadata.segment_id.as_str()) {
            return Err(PitchBenchmarkError::new(format!(
                "duplicate segment_id: {}",
                segment.metadata.segment_id
            )));
        }

        let result = evaluate_segment(spec, segment)?;
        add_counts(&mut total_counts, &result.counts);
        let key = (
            segment.metadata.corpus_id.clone(),
            segment.metadata.recording_condition_id.clone(),
        );
        let entry = grouped.entry(key).or_default();
        add_counts(&mut entry.0, &result.counts);
        entry.1 += 1;
        segment_results.push(result);
    }

    let groups: Vec<PitchConditionGroupV1> = grouped
        .into_iter()
        .map(|((corpus_id, recording_condition_id), (counts, segment_count))| {
            PitchConditionGroupV1 {
                corpus_id,
                recording_condition_id,
                segment_count,
                rates: derive_rates(
                    &counts,
                    spec.gross_pitch_error_threshold_cents,
                ),
                counts,
            }
        })
        .collect();

    let macro_group_frame_error_rate =
        mean(groups.iter().map(|group| group.rates.frame_error_rate));
    let macro_group_voicing_decision_error_rate =
        mean(groups.iter().map(|group| group.rates.voicing_decision_error_rate));
    let defined_gpe: Vec<f64> = groups
        .iter()
        .filter_map(|group| group.rates.gross_pitch_error_rate)
        .collect();
    let macro_group_gross_pitch_error_rate =
        (!defined_gpe.is_empty()).then(|| mean(defined_gpe.iter().copied()));

    let median_segment_frame_error_rate = median(
        segment_results
            .iter()
            .map(|result| result.rates.frame_error_rate)
            .collect(),
    );
    let median_segment_voicing_decision_error_rate = median(
        segment_results
            .iter()
            .map(|result| result.rates.voicing_decision_error_rate)
            .collect(),
    );

    Ok(PitchBenchmarkReportV1 {
        schema: PITCH_BENCHMARK_SCHEMA_V1.to_owned(),
        spec: spec.clone(),
        segment_count: segment_results.len(),
        micro_rates: derive_rates(
            &total_counts,
            spec.gross_pitch_error_threshold_cents,
        ),
        total_counts,
        median_segment_frame_error_rate,
        median_segment_voicing_decision_error_rate,
        macro_group_frame_error_rate,
        macro_group_voicing_decision_error_rate,
        macro_group_gross_pitch_error_rate,
        macro_group_gpe_defined_groups: defined_gpe.len(),
        groups,
        segments: segment_results,
    })
}

fn validate_spec(spec: &PitchBenchmarkSpecV1) -> Result<(), PitchBenchmarkError> {
    require_non_empty(&spec.tracker_id, "tracker_id")?;
    require_non_empty(&spec.tracker_version, "tracker_version")?;
    require_digest(&spec.tracker_artifact_sha256, "tracker_artifact_sha256")?;
    require_non_empty(&spec.reference_suite_id, "reference_suite_id")?;
    require_non_empty(&spec.reference_suite_version, "reference_suite_version")?;
    require_digest(&spec.reference_manifest_sha256, "reference_manifest_sha256")?;
    require_non_empty(&spec.evaluation_split_id, "evaluation_split_id")?;
    require_digest(
        &spec.preprocessing_manifest_sha256,
        "preprocessing_manifest_sha256",
    )?;
    require_non_empty(&spec.frame_alignment_policy_id, "frame_alignment_policy_id")?;
    require_non_empty(&spec.voicing_policy_id, "voicing_policy_id")?;

    if spec.frame_period_us == 0 {
        return Err(PitchBenchmarkError::new(
            "frame_period_us must be positive",
        ));
    }
    if !spec.gross_pitch_error_threshold_cents.is_finite()
        || spec.gross_pitch_error_threshold_cents <= 0.0
    {
        return Err(PitchBenchmarkError::new(
            "gross_pitch_error_threshold_cents must be finite and positive",
        ));
    }
    if !spec.octave_error_tolerance_cents.is_finite()
        || spec.octave_error_tolerance_cents <= 0.0
        || spec.octave_error_tolerance_cents >= 1200.0
    {
        return Err(PitchBenchmarkError::new(
            "octave_error_tolerance_cents must be finite and in (0, 1200)",
        ));
    }
    if spec.uses_score_prior {
        return Err(PitchBenchmarkError::new(
            "primary independent-pitch evaluation must not use score pitch as an estimator prior",
        ));
    }
    Ok(())
}

fn validate_metadata(metadata: &PitchSegmentMetadataV1) -> Result<(), PitchBenchmarkError> {
    require_non_empty(&metadata.segment_id, "segment_id")?;
    require_non_empty(&metadata.corpus_id, "segment corpus_id")?;
    require_non_empty(&metadata.singer_id, "singer_id")?;
    require_non_empty(&metadata.song_id, "song_id")?;
    require_non_empty(&metadata.recording_session_id, "recording_session_id")?;
    require_non_empty(
        &metadata.recording_condition_id,
        "recording_condition_id",
    )?;
    require_non_empty(&metadata.language_id, "language_id")?;
    require_non_empty(&metadata.register_id, "register_id")?;
    require_non_empty(&metadata.technique_id, "technique_id")?;
    require_non_empty(&metadata.snr_stratum_id, "snr_stratum_id")?;
    require_digest(&metadata.source_audio_sha256, "source_audio_sha256")?;
    Ok(())
}

fn evaluate_segment(
    spec: &PitchBenchmarkSpecV1,
    segment: &PitchSegmentInputV1,
) -> Result<PitchSegmentResultV1, PitchBenchmarkError> {
    if segment.frames.is_empty() {
        return Err(PitchBenchmarkError::new(format!(
            "segment {} has no pitch frames",
            segment.metadata.segment_id
        )));
    }

    let mut counts = PitchCountsV1::default();
    let mut signed_errors = Vec::new();
    let mut absolute_errors = Vec::new();

    for (expected_index, frame) in segment.frames.iter().enumerate() {
        if frame.frame_index as usize != expected_index {
            return Err(PitchBenchmarkError::new(format!(
                "segment {} frame indices must be contiguous from zero; expected {}, received {}",
                segment.metadata.segment_id, expected_index, frame.frame_index
            )));
        }
        validate_f0(frame.reference_f0_hz, "reference_f0_hz", &segment.metadata.segment_id)?;
        validate_f0(frame.estimated_f0_hz, "estimated_f0_hz", &segment.metadata.segment_id)?;

        counts.frame_count += 1;
        let reference_voiced = frame.reference_f0_hz.is_some();
        let estimated_voiced = frame.estimated_f0_hz.is_some();
        if reference_voiced {
            counts.reference_voiced_frames += 1;
        }
        if estimated_voiced {
            counts.estimated_voiced_frames += 1;
        }

        match (frame.reference_f0_hz, frame.estimated_f0_hz) {
            (Some(reference), Some(estimate)) => {
                counts.jointly_voiced_frames += 1;
                let signed = 1200.0 * (estimate / reference).log2();
                let absolute = signed.abs();
                signed_errors.push(signed);
                absolute_errors.push(absolute);
                if absolute > spec.gross_pitch_error_threshold_cents + 1e-9 {
                    counts.gross_pitch_error_frames += 1;
                }
                if (absolute - 1200.0).abs() <= spec.octave_error_tolerance_cents {
                    counts.octave_error_frames += 1;
                }
            }
            (Some(_), None) => counts.false_negative_voicing_frames += 1,
            (None, Some(_)) => counts.false_positive_voicing_frames += 1,
            (None, None) => {}
        }
    }

    Ok(segment_result(
        segment.metadata.clone(),
        counts,
        signed_errors,
        absolute_errors,
    ))
}

fn segment_result(
    metadata: PitchSegmentMetadataV1,
    counts: PitchCountsV1,
    mut signed_errors: Vec<f64>,
    mut absolute_errors: Vec<f64>,
) -> PitchSegmentResultV1 {
    signed_errors.sort_by(f64::total_cmp);
    absolute_errors.sort_by(f64::total_cmp);
    let signed_median = (!signed_errors.is_empty()).then(|| percentile_sorted(&signed_errors, 0.5));
    let absolute_median =
        (!absolute_errors.is_empty()).then(|| percentile_sorted(&absolute_errors, 0.5));
    let absolute_p95 =
        (!absolute_errors.is_empty()).then(|| percentile_sorted(&absolute_errors, 0.95));
    let absolute_max = absolute_errors.last().copied();

    let rates = derive_rates(&counts);
    PitchSegmentResultV1 {
        metadata,
        counts,
        rates,
        median_signed_cents_error: signed_median,
        median_absolute_cents_error: absolute_median,
        p95_absolute_cents_error: absolute_p95,
        maximum_absolute_cents_error: absolute_max,
    }
}

fn validate_f0(
    f0: Option<f64>,
    field: &str,
    segment_id: &str,
) -> Result<(), PitchBenchmarkError> {
    if let Some(value) = f0
        && (!value.is_finite() || value <= 0.0)
    {
        return Err(PitchBenchmarkError::new(format!(
            "{field} in segment {segment_id} must be finite and positive when voiced"
        )));
    }
    Ok(())
}

fn derive_rates(counts: &PitchCountsV1) -> PitchRatesV1 {
    let vuv_errors = counts.false_negative_voicing_frames + counts.false_positive_voicing_frames;
    let frame_errors = vuv_errors + counts.gross_pitch_error_frames;
    PitchRatesV1 {
        frame_error_rate: frame_errors as f64 / counts.frame_count.max(1) as f64,
        gross_pitch_error_rate: (counts.jointly_voiced_frames > 0).then(|| {
            counts.gross_pitch_error_frames as f64 / counts.jointly_voiced_frames as f64
        }),
        voicing_decision_error_rate: vuv_errors as f64 / counts.frame_count.max(1) as f64,
        reference_voiced_coverage: (counts.reference_voiced_frames > 0).then(|| {
            counts.jointly_voiced_frames as f64 / counts.reference_voiced_frames as f64
        }),
        abstention_rate: (counts.reference_voiced_frames > 0).then(|| {
            counts.false_negative_voicing_frames as f64 / counts.reference_voiced_frames as f64
        }),
        false_voiced_rate: (counts.frame_count > counts.reference_voiced_frames).then(|| {
            counts.false_positive_voicing_frames as f64
                / (counts.frame_count - counts.reference_voiced_frames) as f64
        }),
        octave_error_rate: (counts.jointly_voiced_frames > 0).then(|| {
            counts.octave_error_frames as f64 / counts.jointly_voiced_frames as f64
        }),
    }
}

fn add_counts(total: &mut PitchCountsV1, next: &PitchCountsV1) {
    total.frame_count += next.frame_count;
    total.reference_voiced_frames += next.reference_voiced_frames;
    total.estimated_voiced_frames += next.estimated_voiced_frames;
    total.jointly_voiced_frames += next.jointly_voiced_frames;
    total.false_negative_voicing_frames += next.false_negative_voicing_frames;
    total.false_positive_voicing_frames += next.false_positive_voicing_frames;
    total.gross_pitch_error_frames += next.gross_pitch_error_frames;
    total.octave_error_frames += next.octave_error_frames;
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let values: Vec<f64> = values.collect();
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    percentile_sorted(&values, 0.5)
}

fn percentile_sorted(values: &[f64], quantile: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    values[lower] + (values[upper] - values[lower]) * fraction
}

fn require_digest(value: &str, field: &str) -> Result<(), PitchBenchmarkError> {
    let valid = value
        .strip_prefix("sha256:")
        .is_some_and(|hex| {
            hex.len() == 64
                && hex
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        });
    if !valid {
        return Err(PitchBenchmarkError::new(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), PitchBenchmarkError> {
    if value.trim().is_empty() {
        return Err(PitchBenchmarkError::new(format!("{field} must not be empty")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(character: char) -> String {
        format!("sha256:{}", character.to_string().repeat(64))
    }

    fn spec() -> PitchBenchmarkSpecV1 {
        PitchBenchmarkSpecV1 {
            tracker_id: "swiftf0".to_owned(),
            tracker_version: "pinned-v1".to_owned(),
            tracker_artifact_sha256: digest('a'),
            reference_suite_id: "licensed-vocal-pilot".to_owned(),
            reference_suite_version: "v1".to_owned(),
            reference_manifest_sha256: digest('b'),
            evaluation_split_id: "singer-song-session-heldout".to_owned(),
            preprocessing_manifest_sha256: digest('c'),
            frame_alignment_policy_id: "16ms-center-aligned-v1".to_owned(),
            voicing_policy_id: "confidence>=0.5-calibrated-v1".to_owned(),
            frame_period_us: 16_000,
            gross_pitch_error_threshold_cents: 50.0,
            octave_error_tolerance_cents: 50.0,
            uses_score_prior: false,
        }
    }

    fn metadata(segment: &str, corpus: &str, condition: &str) -> PitchSegmentMetadataV1 {
        PitchSegmentMetadataV1 {
            segment_id: segment.to_owned(),
            corpus_id: corpus.to_owned(),
            singer_id: format!("singer-{segment}"),
            song_id: format!("song-{segment}"),
            recording_session_id: format!("session-{segment}"),
            recording_condition_id: condition.to_owned(),
            language_id: "en".to_owned(),
            register_id: "middle".to_owned(),
            technique_id: "straight-tone".to_owned(),
            snr_stratum_id: "dry-clean".to_owned(),
            source_audio_sha256: digest('d'),
        }
    }

    fn frame(index: u32, reference: Option<f64>, estimate: Option<f64>) -> PitchFrameV1 {
        PitchFrameV1 {
            frame_index: index,
            reference_f0_hz: reference,
            estimated_f0_hz: estimate,
        }
    }

    #[test]
    fn perfect_voicing_and_pitch_produce_zero_error() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![
                    frame(0, Some(220.0), Some(220.0)),
                    frame(1, Some(440.0), Some(440.0)),
                    frame(2, None, None),
                ],
            }],
        )
        .unwrap();
        let result = &report.segments[0];
        assert_eq!(result.counts.frame_count, 3);
        assert_eq!(result.rates.frame_error_rate, 0.0);
        assert_eq!(result.rates.voicing_decision_error_rate, 0.0);
        assert_eq!(result.rates.gross_pitch_error_rate, Some(0.0));
        assert_eq!(result.rates.reference_voiced_coverage, Some(1.0));
    }

    #[test]
    fn fifty_cent_boundary_is_not_a_gross_pitch_error() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(220.0 * 2.0_f64.powf(50.0 / 1200.0)))],
            }],
        )
        .unwrap();
        assert_eq!(report.segments[0].counts.gross_pitch_error_frames, 0);
    }

    #[test]
    fn large_pitch_error_is_counted_in_gpe_and_ffe_but_not_vde() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(220.0 * 2.0_f64.powf(100.0 / 1200.0)))],
            }],
        )
        .unwrap();
        let result = &report.segments[0];
        assert_eq!(result.counts.gross_pitch_error_frames, 1);
        assert_eq!(result.rates.frame_error_rate, 1.0);
        assert_eq!(result.rates.voicing_decision_error_rate, 0.0);
    }

    #[test]
    fn configured_gross_pitch_threshold_is_used_for_every_metric() {
        let mut spec = spec();
        spec.gross_pitch_error_threshold_cents = 100.0;
        let report = evaluate_pitch_benchmark(
            &spec,
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(
                    0,
                    Some(220.0),
                    Some(220.0 * 2.0_f64.powf(75.0 / 1200.0)),
                )],
            }],
        )
        .unwrap();
        assert_eq!(report.segments[0].counts.gross_pitch_error_frames, 0);
        assert_eq!(report.segments[0].rates.frame_error_rate, 0.0);
        assert_eq!(report.segments[0].rates.gross_pitch_error_rate, Some(0.0));
    }

    #[test]
    fn octave_error_is_separately_counted() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(440.0))],
            }],
        )
        .unwrap();
        assert_eq!(report.segments[0].counts.octave_error_frames, 1);
        assert_eq!(report.segments[0].rates.octave_error_rate, Some(1.0));
    }

    #[test]
    fn voicing_mismatch_is_ffe_and_vde() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![
                    frame(0, Some(220.0), None),
                    frame(1, None, Some(330.0)),
                ],
            }],
        )
        .unwrap();
        let result = &report.segments[0];
        assert_eq!(result.counts.false_negative_voicing_frames, 1);
        assert_eq!(result.counts.false_positive_voicing_frames, 1);
        assert_eq!(result.rates.frame_error_rate, 1.0);
        assert_eq!(result.rates.voicing_decision_error_rate, 1.0);
        assert_eq!(result.rates.abstention_rate, Some(1.0));
        assert_eq!(result.rates.false_voiced_rate, Some(1.0));
    }

    #[test]
    fn gpe_is_undefined_when_no_estimated_reference_frames_are_jointly_voiced() {
        let report = evaluate_pitch_benchmark(
            &spec(),
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), None)],
            }],
        )
        .unwrap();
        let result = &report.segments[0];
        assert_eq!(result.rates.gross_pitch_error_rate, None);
        assert_eq!(result.rates.reference_voiced_coverage, Some(0.0));
        assert_eq!(result.rates.median_absolute_cents_error, None);
    }

    #[test]
    fn score_conditioned_tracker_is_rejected_from_primary_independent_track() {
        let mut spec = spec();
        spec.uses_score_prior = true;
        assert!(evaluate_pitch_benchmark(
            &spec,
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(220.0))],
            }]
        )
        .is_err());
    }

    #[test]
    fn duplicate_segment_ids_are_rejected() {
        let segment = PitchSegmentInputV1 {
            metadata: metadata("clip-01", "corpus-a", "clean"),
            frames: vec![frame(0, Some(220.0), Some(220.0))],
        };
        assert!(evaluate_pitch_benchmark(&spec(), &[segment.clone(), segment]).is_err());
    }

    #[test]
    fn frame_indices_must_be_contiguous_and_ordered() {
        let segment = PitchSegmentInputV1 {
            metadata: metadata("clip-01", "corpus-a", "clean"),
            frames: vec![
                frame(0, Some(220.0), Some(220.0)),
                frame(2, Some(330.0), Some(330.0)),
            ],
        };
        assert!(evaluate_pitch_benchmark(&spec(), &[segment]).is_err());
    }

    #[test]
    fn nonpositive_nonfinite_f0_values_are_rejected() {
        let invalid_values = [0.0, -1.0, f64::NAN, f64::INFINITY];
        for invalid in invalid_values {
            let segment = PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(invalid))],
            };
            assert!(evaluate_pitch_benchmark(&spec(), &[segment]).is_err());
        }
    }

    #[test]
    fn missing_metadata_and_invalid_digests_are_rejected() {
        let mut segment = PitchSegmentInputV1 {
            metadata: metadata("clip-01", "corpus-a", "clean"),
            frames: vec![frame(0, Some(220.0), Some(220.0))],
        };
        segment.metadata.singer_id.clear();
        assert!(evaluate_pitch_benchmark(&spec(), &[segment]).is_err());

        let mut spec = spec();
        spec.reference_manifest_sha256 = "untrusted".to_owned();
        assert!(evaluate_pitch_benchmark(
            &spec,
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(220.0))],
            }]
        )
        .is_err());
    }

    #[test]
    fn equal_group_weighting_differs_from_micro_frame_weighting() {
        let long_clean = PitchSegmentInputV1 {
            metadata: metadata("long-clean", "corpus-a", "clean"),
            frames: vec![frame(0, Some(220.0), Some(220.0)); 100],
        };
        let noisy_bad = PitchSegmentInputV1 {
            metadata: metadata("short-noisy", "corpus-b", "noise"),
            frames: vec![frame(0, Some(220.0), Some(220.0 * 2.0_f64.powf(100.0 / 1200.0)))],
        };
        let report = evaluate_pitch_benchmark(&spec(), &[long_clean, noisy_bad]).unwrap();
        assert!(report.macro_group_frame_error_rate > 0.0);
        assert!(report.micro_rates.frame_error_rate < report.macro_group_frame_error_rate);
        assert_eq!(report.groups.len(), 2);
    }

    #[test]
    fn thresholds_must_be_positive_and_frame_grid_is_declared() {
        let mut spec = spec();
        spec.frame_period_us = 0;
        assert!(evaluate_pitch_benchmark(
            &spec,
            &[PitchSegmentInputV1 {
                metadata: metadata("clip-01", "corpus-a", "clean"),
                frames: vec![frame(0, Some(220.0), Some(220.0))],
            }]
        )
        .is_err());
    }

    #[test]
    fn no_segments_is_rejected() {
        assert!(evaluate_pitch_benchmark(&spec(), &[]).is_err());
    }
}

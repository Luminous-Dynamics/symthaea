// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio-derived beat-grid and tempo-map evidence contracts (MEL-DJ-007).
//!
//! This module defines portable evidence records and a deterministic
//! structural validator. It does not decode audio, estimate beats, or prove
//! that a supplied analysis is perceptually correct. Signal and auditory-scene
//! evidence remains owned by MEL-HEAR-011 (#14227).
//!
//! Beat positions use integer source-local frame coordinates. Tempo uses
//! milli-BPM, confidence uses parts per million, and no clock conversion is
//! performed implicitly. A report can retain multiple competing hypotheses
//! or explicitly abstain; consumers must not silently select the first one.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::transition::{
    validate_transition_contract, AnalysisEvidenceRef, AudioSourceRef, TrackWindowRef,
    TransitionContract, MAX_SAFE_JSON_INTEGER,
};

/// Current beat-grid report schema understood by this crate.
pub const BEAT_GRID_SCHEMA_VERSION: u16 = 1;

/// Confidence represented in parts per million, from 0 through 1,000,000.
pub const MAX_CONFIDENCE_PPM: u32 = 1_000_000;

/// Report-level disposition. Resolved means exactly one hypothesis is emitted;
/// Ambiguous means two or more alternatives are retained.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatGridDisposition {
    Resolved,
    Ambiguous,
    Abstained,
    NotEvaluated,
}

/// One detected pulse location, indexed by an integer beat ordinal.
///
/// This does not assert a bar/downbeat location. Downbeat phase is represented
/// separately so pulse tracking and meter interpretation cannot be conflated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PulseAnchor {
    pub frame: u64,
    pub beat_index: i64,
    pub confidence_ppm: u32,
}

/// A downbeat anchor referring to one exact pulse anchor in the same hypothesis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownbeatAnchor {
    pub frame: u64,
    pub beat_index: i64,
    pub bar_index: i64,
    pub confidence_ppm: u32,
}

/// Semantics for values between successive tempo-map points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TempoInterpolation {
    /// Hold the previous BPM value until the next point.
    StepwiseConstant,
    /// Interpolate BPM linearly in source-frame time between points.
    LinearInFrameTime,
}

/// A point in a piecewise tempo map. Tempo is in milli-BPM, avoiding floating
/// point serialization and making boundary comparisons deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TempoMapPoint {
    pub frame: u64,
    pub bpm_milli: u32,
    pub confidence_ppm: u32,
}

/// Meter/grouping hypothesis beginning at a beat ordinal.
///
/// Grouping expresses additive numerator groupings, e.g. 7/8 as [2, 2, 3].
/// It is optional at the hypothesis level: pulse evidence can be useful even
/// when meter or bar phase is unknown.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeterMapSegment {
    pub start_beat_index: i64,
    pub numerator: u8,
    pub denominator: u8,
    pub grouping: Vec<u8>,
    pub confidence_ppm: u32,
}

/// A competing explanation of the source's pulse, tempo, and meter.
///
/// Confidence is a producer-supplied score in parts per million. It is not a
/// calibrated probability unless a separate calibration record establishes that.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatGridHypothesis {
    pub hypothesis_id: String,
    pub confidence_ppm: u32,
    pub pulse_anchors: Vec<PulseAnchor>,
    pub downbeat_anchors: Vec<DownbeatAnchor>,
    pub tempo_interpolation: Option<TempoInterpolation>,
    pub tempo_map: Vec<TempoMapPoint>,
    pub meter_map: Vec<MeterMapSegment>,
}

/// Versioned report binding hypotheses to the source audio and its analysis manifest.
///
/// There is no Default implementation: a decoder cannot manufacture an
/// empty-but-valid report, confidence, source identity, or evidence provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatGridReportV1 {
    pub schema_version: u16,
    pub source: AudioSourceRef,
    pub evidence: AnalysisEvidenceRef,
    pub disposition: BeatGridDisposition,
    pub hypotheses: Vec<BeatGridHypothesis>,
    pub abstention_reason: Option<String>,
}

/// Stable machine-readable validation errors for beat-grid reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BeatGridViolationCode {
    UnsupportedSchemaVersion,
    InvalidSourceSha256,
    InvalidSampleRate,
    InvalidChannelCount,
    EmptySource,
    SourceFrameCountExceedsJsonSafeInteger,
    InvalidEvidenceManifestSha256,
    EvidenceSourceDigestMismatch,
    EmptyAnalysisVersion,
    InvalidEvidenceWindow,
    ReportDispositionMismatch,
    MissingAbstentionReason,
    UnexpectedAbstentionReason,
    EmptyHypothesisId,
    DuplicateHypothesisId,
    HypothesisConfidenceOutOfRange,
    TooFewPulseAnchors,
    PulseAnchorOutOfEvidenceWindow,
    PulseAnchorConfidenceOutOfRange,
    PulseBeatIndexExceedsJsonSafeInteger,
    PulseFramesNotStrictlyIncreasing,
    PulseIndexesNotStrictlyIncreasing,
    DownbeatAnchorOutOfEvidenceWindow,
    DownbeatAnchorConfidenceOutOfRange,
    DownbeatBeatIndexExceedsJsonSafeInteger,
    DownbeatBarIndexExceedsJsonSafeInteger,
    DownbeatFramesNotStrictlyIncreasing,
    DownbeatIndexesNotStrictlyIncreasing,
    DownbeatDoesNotMatchPulse,
    TempoPointOutOfEvidenceWindow,
    TempoPointConfidenceOutOfRange,
    TempoValueMustBePositive,
    TempoFramesNotStrictlyIncreasing,
    TempoInterpolationMismatch,
    MeterSegmentConfidenceOutOfRange,
    MeterStartBeatNotInPulseMap,
    MeterStartBeatExceedsJsonSafeInteger,
    MeterSegmentsNotStrictlyIncreasing,
    InvalidMeterSignature,
    EmptyMeterGrouping,
    InvalidMeterGrouping,
}

/// One stable validation result with a machine-readable field path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BeatGridViolation {
    pub code: BeatGridViolationCode,
    pub path: String,
}

/// Validate source linkage, report disposition, and all source-local temporal claims.
///
/// Validation is structural only. It does not verify detector accuracy,
/// probability calibration, musical correctness, or legal dataset permission.
pub fn validate_beat_grid_report(report: &BeatGridReportV1) -> Vec<BeatGridViolation> {
    let mut violations = Vec::new();

    if report.schema_version != BEAT_GRID_SCHEMA_VERSION {
        push(&mut violations, BeatGridViolationCode::UnsupportedSchemaVersion, "schema_version");
    }

    if !is_lowercase_sha256(&report.source.sha256) {
        push(&mut violations, BeatGridViolationCode::InvalidSourceSha256, "source.sha256");
    }
    if report.source.sample_rate_hz == 0 {
        push(&mut violations, BeatGridViolationCode::InvalidSampleRate, "source.sample_rate_hz");
    }
    if report.source.channels == 0 {
        push(&mut violations, BeatGridViolationCode::InvalidChannelCount, "source.channels");
    }
    if report.source.total_frames == 0 {
        push(&mut violations, BeatGridViolationCode::EmptySource, "source.total_frames");
    }
    if report.source.total_frames > MAX_SAFE_JSON_INTEGER {
        push(
            &mut violations,
            BeatGridViolationCode::SourceFrameCountExceedsJsonSafeInteger,
            "source.total_frames",
        );
    }

    if !is_lowercase_sha256(&report.evidence.manifest_sha256) {
        push(
            &mut violations,
            BeatGridViolationCode::InvalidEvidenceManifestSha256,
            "evidence.manifest_sha256",
        );
    }
    if report.evidence.source_audio_sha256 != report.source.sha256 {
        push(
            &mut violations,
            BeatGridViolationCode::EvidenceSourceDigestMismatch,
            "evidence.source_audio_sha256",
        );
    }
    if report.evidence.analysis_version.trim().is_empty() {
        push(
            &mut violations,
            BeatGridViolationCode::EmptyAnalysisVersion,
            "evidence.analysis_version",
        );
    }
    let evidence_window = report.evidence.covered_window;
    if !evidence_window.is_non_empty() || evidence_window.end_frame > report.source.total_frames {
        push(
            &mut violations,
            BeatGridViolationCode::InvalidEvidenceWindow,
            "evidence.covered_window",
        );
    }

    match report.disposition {
        BeatGridDisposition::Resolved => {
            if report.hypotheses.len() != 1 {
                push(
                    &mut violations,
                    BeatGridViolationCode::ReportDispositionMismatch,
                    "hypotheses",
                );
            }
            if report.abstention_reason.is_some() {
                push(
                    &mut violations,
                    BeatGridViolationCode::UnexpectedAbstentionReason,
                    "abstention_reason",
                );
            }
        }
        BeatGridDisposition::Ambiguous => {
            if report.hypotheses.len() < 2 {
                push(
                    &mut violations,
                    BeatGridViolationCode::ReportDispositionMismatch,
                    "hypotheses",
                );
            }
            if report.abstention_reason.is_some() {
                push(
                    &mut violations,
                    BeatGridViolationCode::UnexpectedAbstentionReason,
                    "abstention_reason",
                );
            }
        }
        BeatGridDisposition::Abstained | BeatGridDisposition::NotEvaluated => {
            if !report.hypotheses.is_empty() {
                push(
                    &mut violations,
                    BeatGridViolationCode::ReportDispositionMismatch,
                    "hypotheses",
                );
            }
            if report.abstention_reason.as_deref().is_none_or(|reason| reason.trim().is_empty()) {
                push(
                    &mut violations,
                    BeatGridViolationCode::MissingAbstentionReason,
                    "abstention_reason",
                );
            }
        }
    }

    let mut hypothesis_ids = BTreeSet::new();
    for (hypothesis_index, hypothesis) in report.hypotheses.iter().enumerate() {
        let base = format!("hypotheses[{hypothesis_index}]");
        if hypothesis.hypothesis_id.trim().is_empty() {
            push(
                &mut violations,
                BeatGridViolationCode::EmptyHypothesisId,
                format!("{base}.hypothesis_id"),
            );
        } else if !hypothesis_ids.insert(hypothesis.hypothesis_id.as_str()) {
            push(
                &mut violations,
                BeatGridViolationCode::DuplicateHypothesisId,
                format!("{base}.hypothesis_id"),
            );
        }

        if hypothesis.confidence_ppm > MAX_CONFIDENCE_PPM {
            push(
                &mut violations,
                BeatGridViolationCode::HypothesisConfidenceOutOfRange,
                format!("{base}.confidence_ppm"),
            );
        }
        if hypothesis.pulse_anchors.len() < 2 {
            push(
                &mut violations,
                BeatGridViolationCode::TooFewPulseAnchors,
                format!("{base}.pulse_anchors"),
            );
        }

        let mut previous_frame = None;
        let mut previous_beat_index = None;
        for (anchor_index, anchor) in hypothesis.pulse_anchors.iter().enumerate() {
            let path = format!("{base}.pulse_anchors[{anchor_index}]");
            if !evidence_window.contains_frame(anchor.frame) {
                push(
                    &mut violations,
                    BeatGridViolationCode::PulseAnchorOutOfEvidenceWindow,
                    format!("{path}.frame"),
                );
            }
            if anchor.beat_index.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
                push(
                    &mut violations,
                    BeatGridViolationCode::PulseBeatIndexExceedsJsonSafeInteger,
                    format!("{path}.beat_index"),
                );
            }
            if anchor.confidence_ppm > MAX_CONFIDENCE_PPM {
                push(
                    &mut violations,
                    BeatGridViolationCode::PulseAnchorConfidenceOutOfRange,
                    format!("{path}.confidence_ppm"),
                );
            }
            if previous_frame.is_some_and(|previous| anchor.frame <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::PulseFramesNotStrictlyIncreasing,
                    format!("{path}.frame"),
                );
            }
            if previous_beat_index.is_some_and(|previous| anchor.beat_index <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::PulseIndexesNotStrictlyIncreasing,
                    format!("{path}.beat_index"),
                );
            }
            previous_frame = Some(anchor.frame);
            previous_beat_index = Some(anchor.beat_index);
        }

        let mut previous_downbeat_frame = None;
        let mut previous_bar_index = None;
        for (anchor_index, anchor) in hypothesis.downbeat_anchors.iter().enumerate() {
            let path = format!("{base}.downbeat_anchors[{anchor_index}]");
            if !evidence_window.contains_frame(anchor.frame) {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatAnchorOutOfEvidenceWindow,
                    format!("{path}.frame"),
                );
            }
            if anchor.beat_index.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatBeatIndexExceedsJsonSafeInteger,
                    format!("{path}.beat_index"),
                );
            }
            if anchor.bar_index.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatBarIndexExceedsJsonSafeInteger,
                    format!("{path}.bar_index"),
                );
            }
            if anchor.confidence_ppm > MAX_CONFIDENCE_PPM {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatAnchorConfidenceOutOfRange,
                    format!("{path}.confidence_ppm"),
                );
            }
            if previous_downbeat_frame.is_some_and(|previous| anchor.frame <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatFramesNotStrictlyIncreasing,
                    format!("{path}.frame"),
                );
            }
            if previous_bar_index.is_some_and(|previous| anchor.bar_index <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatIndexesNotStrictlyIncreasing,
                    format!("{path}.bar_index"),
                );
            }
            if !hypothesis.pulse_anchors.iter().any(|pulse| {
                pulse.frame == anchor.frame && pulse.beat_index == anchor.beat_index
            }) {
                push(
                    &mut violations,
                    BeatGridViolationCode::DownbeatDoesNotMatchPulse,
                    format!("{path}.beat_index"),
                );
            }
            previous_downbeat_frame = Some(anchor.frame);
            previous_bar_index = Some(anchor.bar_index);
        }

        if hypothesis.tempo_map.is_empty() != hypothesis.tempo_interpolation.is_none() {
            push(
                &mut violations,
                BeatGridViolationCode::TempoInterpolationMismatch,
                format!("{base}.tempo_interpolation"),
            );
        }

        let mut previous_tempo_frame = None;
        for (point_index, point) in hypothesis.tempo_map.iter().enumerate() {
            let path = format!("{base}.tempo_map[{point_index}]");
            if !evidence_window.contains_frame(point.frame) {
                push(
                    &mut violations,
                    BeatGridViolationCode::TempoPointOutOfEvidenceWindow,
                    format!("{path}.frame"),
                );
            }
            if point.confidence_ppm > MAX_CONFIDENCE_PPM {
                push(
                    &mut violations,
                    BeatGridViolationCode::TempoPointConfidenceOutOfRange,
                    format!("{path}.confidence_ppm"),
                );
            }
            if point.bpm_milli == 0 {
                push(
                    &mut violations,
                    BeatGridViolationCode::TempoValueMustBePositive,
                    format!("{path}.bpm_milli"),
                );
            }
            if previous_tempo_frame.is_some_and(|previous| point.frame <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::TempoFramesNotStrictlyIncreasing,
                    format!("{path}.frame"),
                );
            }
            previous_tempo_frame = Some(point.frame);
        }

        let pulse_indexes: BTreeSet<i64> = hypothesis
            .pulse_anchors
            .iter()
            .map(|anchor| anchor.beat_index)
            .collect();
        let mut previous_meter_start = None;
        for (segment_index, segment) in hypothesis.meter_map.iter().enumerate() {
            let path = format!("{base}.meter_map[{segment_index}]");
            if segment.confidence_ppm > MAX_CONFIDENCE_PPM {
                push(
                    &mut violations,
                    BeatGridViolationCode::MeterSegmentConfidenceOutOfRange,
                    format!("{path}.confidence_ppm"),
                );
            }
            if segment.start_beat_index.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
                push(
                    &mut violations,
                    BeatGridViolationCode::MeterStartBeatExceedsJsonSafeInteger,
                    format!("{path}.start_beat_index"),
                );
            }
            if !pulse_indexes.contains(&segment.start_beat_index) {
                push(
                    &mut violations,
                    BeatGridViolationCode::MeterStartBeatNotInPulseMap,
                    format!("{path}.start_beat_index"),
                );
            }
            if previous_meter_start.is_some_and(|previous| segment.start_beat_index <= previous) {
                push(
                    &mut violations,
                    BeatGridViolationCode::MeterSegmentsNotStrictlyIncreasing,
                    format!("{path}.start_beat_index"),
                );
            }
            if segment.numerator == 0
                || segment.denominator == 0
                || !segment.denominator.is_power_of_two()
            {
                push(
                    &mut violations,
                    BeatGridViolationCode::InvalidMeterSignature,
                    format!("{path}.numerator_denominator"),
                );
            }
            if segment.grouping.is_empty() {
                push(
                    &mut violations,
                    BeatGridViolationCode::EmptyMeterGrouping,
                    format!("{path}.grouping"),
                );
            } else {
                let sum = segment.grouping.iter().try_fold(0u16, |total, value| {
                    if *value == 0 {
                        None
                    } else {
                        total.checked_add(u16::from(*value))
                    }
                });
                if sum != Some(u16::from(segment.numerator)) {
                    push(
                        &mut violations,
                        BeatGridViolationCode::InvalidMeterGrouping,
                        format!("{path}.grouping"),
                    );
                }
            }
            previous_meter_start = Some(segment.start_beat_index);
        }
    }

    violations
}


/// Stable failures when binding an alignment to its exact source/target grid reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionGridBindingViolationCode {
    InvalidTransitionContract,
    InvalidSourceBeatGridReport,
    InvalidTargetBeatGridReport,
    SourceAudioReferenceMismatch,
    TargetAudioReferenceMismatch,
    SourceAnalysisReferenceMismatch,
    TargetAnalysisReferenceMismatch,
    SourceHypothesisNotFound,
    TargetHypothesisNotFound,
    SourceAnchorOutsidePulseSupport,
    TargetAnchorOutsidePulseSupport,
    BeatPhaseMismatch,
}

/// A typed, stable result from transition-to-beat-grid evidence binding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionGridBindingViolation {
    pub code: TransitionGridBindingViolationCode,
    pub path: String,
}

/// Verify that an explicit transition alignment is bound to the exact source and
/// target audio, analysis manifests, and selected pulse hypotheses.
///
/// This is stronger than checking that hypothesis ID strings are non-empty.
/// It also verifies that both frame anchors are bracketed by pulse anchors in
/// their respective hypotheses and that their circular beat-phase error is no
/// greater than the transition's declared tolerance.
///
/// This function does not establish probability calibration, musical quality,
/// detector accuracy, or execution authority. The caller must retain the
/// validator report and these source documents as evidence.
pub fn verify_transition_grid_binding(
    contract: &TransitionContract,
    source_report: &BeatGridReportV1,
    target_report: &BeatGridReportV1,
) -> Vec<TransitionGridBindingViolation> {
    let Some(alignment) = contract.alignment.as_ref() else {
        return Vec::new();
    };
    let mut violations = Vec::new();

    if !validate_transition_contract(contract).is_empty() {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::InvalidTransitionContract,
            "contract",
        );
        return violations;
    }
    if !validate_beat_grid_report(source_report).is_empty() {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::InvalidSourceBeatGridReport,
            "source_report",
        );
    }
    if !validate_beat_grid_report(target_report).is_empty() {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::InvalidTargetBeatGridReport,
            "target_report",
        );
    }
    if !violations.is_empty() {
        return violations;
    }

    if source_report.source != contract.source.audio {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::SourceAudioReferenceMismatch,
            "source_report.source",
        );
    }
    if source_report.evidence != contract.source.evidence {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::SourceAnalysisReferenceMismatch,
            "source_report.evidence",
        );
    }
    if target_report.source != contract.target.audio {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::TargetAudioReferenceMismatch,
            "target_report.source",
        );
    }
    if target_report.evidence != contract.target.evidence {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::TargetAnalysisReferenceMismatch,
            "target_report.evidence",
        );
    }
    if !violations.is_empty() {
        return violations;
    }

    let source_hypothesis = source_report
        .hypotheses
        .iter()
        .find(|hypothesis| hypothesis.hypothesis_id == alignment.source_grid_hypothesis_id);
    let target_hypothesis = target_report
        .hypotheses
        .iter()
        .find(|hypothesis| hypothesis.hypothesis_id == alignment.target_grid_hypothesis_id);

    let Some(source_hypothesis) = source_hypothesis else {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::SourceHypothesisNotFound,
            "alignment.source_grid_hypothesis_id",
        );
        return violations;
    };
    let Some(target_hypothesis) = target_hypothesis else {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::TargetHypothesisNotFound,
            "alignment.target_grid_hypothesis_id",
        );
        return violations;
    };

    let Some(source_position_milli) = beat_position_milli(
        source_hypothesis,
        alignment.source_anchor_frame,
    ) else {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::SourceAnchorOutsidePulseSupport,
            "alignment.source_anchor_frame",
        );
        return violations;
    };
    let Some(target_position_milli) = beat_position_milli(
        target_hypothesis,
        alignment.target_anchor_frame,
    ) else {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::TargetAnchorOutsidePulseSupport,
            "alignment.target_anchor_frame",
        );
        return violations;
    };

    let source_phase = source_position_milli.rem_euclid(1_000);
    let target_phase = target_position_milli.rem_euclid(1_000);
    let raw_error = (source_phase - target_phase).abs();
    let circular_error = raw_error.min(1_000 - raw_error);
    if circular_error > i128::from(alignment.max_phase_error_milli) {
        push_binding(
            &mut violations,
            TransitionGridBindingViolationCode::BeatPhaseMismatch,
            "alignment.max_phase_error_milli",
        );
    }

    violations
}

/// Estimate a source-local beat position in milli-beats using the two
/// neighboring pulse anchors. Linear interpolation is only used inside a
/// bracketed interval; extrapolation beyond observed pulse support abstains.
fn beat_position_milli(hypothesis: &BeatGridHypothesis, frame: u64) -> Option<i128> {
    // An exact observed pulse is valid even if neighboring pulses are missing.
    if let Some(anchor) = hypothesis
        .pulse_anchors
        .iter()
        .find(|anchor| anchor.frame == frame)
    {
        return Some(i128::from(anchor.beat_index) * 1_000);
    }

    // Interpolate only between adjacent beat ordinals. Do not invent phase
    // through a multi-beat detection gap or extrapolate beyond evidence.
    for pair in hypothesis.pulse_anchors.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if frame <= left.frame || frame >= right.frame {
            continue;
        }
        let frame_span = i128::from(right.frame) - i128::from(left.frame);
        let beat_span = i128::from(right.beat_index) - i128::from(left.beat_index);
        if frame_span <= 0 || beat_span != 1 {
            return None;
        }
        let elapsed_frames = i128::from(frame) - i128::from(left.frame);
        return Some(
            i128::from(left.beat_index) * 1_000
                + elapsed_frames * 1_000 / frame_span,
        );
    }
    None
}

fn push_binding(
    violations: &mut Vec<TransitionGridBindingViolation>,
    code: TransitionGridBindingViolationCode,
    path: impl Into<String>,
) {
    violations.push(TransitionGridBindingViolation {
        code,
        path: path.into(),
    });
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn push(
    violations: &mut Vec<BeatGridViolation>,
    code: BeatGridViolationCode,
    path: impl Into<String>,
) {
    violations.push(BeatGridViolation { code, path: path.into() });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> BeatGridReportV1 {
        let source_sha = "1".repeat(64);
        BeatGridReportV1 {
            schema_version: BEAT_GRID_SCHEMA_VERSION,
            source: AudioSourceRef {
                sha256: source_sha.clone(),
                sample_rate_hz: 48_000,
                channels: 2,
                total_frames: 100_000,
            },
            evidence: AnalysisEvidenceRef {
                manifest_sha256: "a".repeat(64),
                source_audio_sha256: source_sha,
                analysis_version: "auditory-scene-v0.1/config-17".to_string(),
                covered_window: crate::transition::SampleWindow {
                    start_frame: 0,
                    end_frame: 100_000,
                },
            },
            disposition: BeatGridDisposition::Resolved,
            hypotheses: vec![hypothesis("grid-a", 850_000)],
            abstention_reason: None,
        }
    }

    fn hypothesis(id: &str, confidence_ppm: u32) -> BeatGridHypothesis {
        BeatGridHypothesis {
            hypothesis_id: id.to_string(),
            confidence_ppm,
            pulse_anchors: vec![
                PulseAnchor { frame: 0, beat_index: 0, confidence_ppm: 900_000 },
                PulseAnchor { frame: 24_000, beat_index: 1, confidence_ppm: 920_000 },
                PulseAnchor { frame: 48_000, beat_index: 2, confidence_ppm: 910_000 },
            ],
            downbeat_anchors: vec![DownbeatAnchor {
                frame: 0,
                beat_index: 0,
                bar_index: 0,
                confidence_ppm: 800_000,
            }],
            tempo_interpolation: Some(TempoInterpolation::LinearInFrameTime),
            tempo_map: vec![
                TempoMapPoint { frame: 0, bpm_milli: 120_000, confidence_ppm: 880_000 },
                TempoMapPoint { frame: 48_000, bpm_milli: 121_000, confidence_ppm: 810_000 },
            ],
            meter_map: vec![MeterMapSegment {
                start_beat_index: 0,
                numerator: 7,
                denominator: 8,
                grouping: vec![2, 2, 3],
                confidence_ppm: 760_000,
            }],
        }
    }

    fn codes(report: &BeatGridReportV1) -> Vec<BeatGridViolationCode> {
        validate_beat_grid_report(report).into_iter().map(|violation| violation.code).collect()
    }

    #[test]
    fn accepts_source_local_variable_tempo_and_odd_meter_hypothesis() {
        let report = report();
        assert!(validate_beat_grid_report(&report).is_empty());
        assert_eq!(report.hypotheses[0].tempo_map[0].bpm_milli, 120_000);
        assert_eq!(report.hypotheses[0].meter_map[0].grouping, vec![2, 2, 3]);
    }

    #[test]
    fn accepts_multiple_hypotheses_as_ambiguous() {
        let mut report = report();
        report.disposition = BeatGridDisposition::Ambiguous;
        report.hypotheses.push(hypothesis("grid-b-half-time", 720_000));
        assert!(validate_beat_grid_report(&report).is_empty());
    }

    #[test]
    fn accepts_explicit_abstention_without_fake_grid() {
        let mut report = report();
        report.disposition = BeatGridDisposition::Abstained;
        report.hypotheses.clear();
        report.abstention_reason = Some("tempo_and_meter_not_identifiable".to_string());
        assert!(validate_beat_grid_report(&report).is_empty());
    }

    #[test]
    fn rejects_unsupported_version_and_source_digest_mismatch() {
        let mut report = report();
        report.schema_version = BEAT_GRID_SCHEMA_VERSION + 1;
        report.evidence.source_audio_sha256 = "2".repeat(64);
        let found = codes(&report);
        assert!(found.contains(&BeatGridViolationCode::UnsupportedSchemaVersion));
        assert!(found.contains(&BeatGridViolationCode::EvidenceSourceDigestMismatch));
    }

    #[test]
    fn rejects_non_increasing_pulse_frames_and_indexes() {
        let mut report = report();
        report.hypotheses[0].pulse_anchors[1].frame = 0;
        report.hypotheses[0].pulse_anchors[2].beat_index = 1;
        let found = codes(&report);
        assert!(found.contains(&BeatGridViolationCode::PulseFramesNotStrictlyIncreasing));
        assert!(found.contains(&BeatGridViolationCode::PulseIndexesNotStrictlyIncreasing));
    }

    #[test]
    fn rejects_downbeat_not_bound_to_exact_pulse_anchor() {
        let mut report = report();
        report.hypotheses[0].downbeat_anchors[0].frame = 1;
        assert!(codes(&report).contains(&BeatGridViolationCode::DownbeatDoesNotMatchPulse));
    }

    #[test]
    fn rejects_tempo_points_out_of_order_and_nonpositive_tempo() {
        let mut report = report();
        report.hypotheses[0].tempo_map = vec![
            TempoMapPoint { frame: 24_000, bpm_milli: 0, confidence_ppm: 1_000_001 },
            TempoMapPoint { frame: 12_000, bpm_milli: 120_000, confidence_ppm: 800_000 },
        ];
        let found = codes(&report);
        assert!(found.contains(&BeatGridViolationCode::TempoFramesNotStrictlyIncreasing));
        assert!(found.contains(&BeatGridViolationCode::TempoValueMustBePositive));
        assert!(found.contains(&BeatGridViolationCode::TempoPointConfidenceOutOfRange));
    }

    #[test]
    fn rejects_tempo_map_without_interpolation_semantics() {
        let mut report = report();
        report.hypotheses[0].tempo_interpolation = None;
        assert!(codes(&report).contains(&BeatGridViolationCode::TempoInterpolationMismatch));
    }

    #[test]
    fn rejects_meter_grouping_that_does_not_match_numerator() {
        let mut report = report();
        report.hypotheses[0].meter_map[0].grouping = vec![2, 2, 2];
        assert!(codes(&report).contains(&BeatGridViolationCode::InvalidMeterGrouping));
    }

    #[test]
    fn rejects_report_disposition_mismatch() {
        let mut report = report();
        report.disposition = BeatGridDisposition::Abstained;
        report.abstention_reason = Some("insufficient_pulse_evidence".to_string());
        assert!(codes(&report).contains(&BeatGridViolationCode::ReportDispositionMismatch));
    }

    #[test]
    fn rejects_unknown_fields_instead_of_inventing_a_grid() {
        let mut value = serde_json::to_value(report()).unwrap();
        value.as_object_mut().unwrap().insert(
            "automatic_transport_authorized".to_string(),
            serde_json::Value::Bool(true),
        );
        assert!(serde_json::from_value::<BeatGridReportV1>(value).is_err());
    }

    #[test]
    fn rejects_unknown_nested_evidence_fields() {
        let mut value = serde_json::to_value(report()).unwrap();
        value["evidence"]
            .as_object_mut()
            .unwrap()
            .insert("fabricated_confidence".to_string(), serde_json::Value::from(1.0));
        assert!(serde_json::from_value::<BeatGridReportV1>(value).is_err());
    }

    #[test]
    fn accepts_pulse_only_hypothesis_without_tempo_map() {
        let mut report = report();
        report.hypotheses[0].tempo_map.clear();
        report.hypotheses[0].tempo_interpolation = None;
        assert!(validate_beat_grid_report(&report).is_empty());
    }

    #[test]
    fn round_trips_without_floating_point_clock_or_tempo_fields() {
        let report = report();
        let json = serde_json::to_string(&report).unwrap();
        let decoded: BeatGridReportV1 = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, report);
        assert!(!json.contains("bpm_f32"));
        assert!(json.contains("\"bpm_milli\":120000"));
    }

    #[test]
    fn rejects_source_frame_count_above_exact_json_integer_range() {
        let mut report = report();
        report.source.total_frames = MAX_SAFE_JSON_INTEGER + 1;
        assert!(codes(&report)
            .contains(&BeatGridViolationCode::SourceFrameCountExceedsJsonSafeInteger));
    }

    #[test]
    fn rejects_beat_and_bar_indices_above_exact_json_integer_range() {
        let mut report = report();
        report.hypotheses[0].pulse_anchors[0].beat_index =
            (MAX_SAFE_JSON_INTEGER as i64) + 1;
        report.hypotheses[0].downbeat_anchors[0].bar_index =
            (MAX_SAFE_JSON_INTEGER as i64) + 1;
        let found = codes(&report);
        assert!(found.contains(&BeatGridViolationCode::PulseBeatIndexExceedsJsonSafeInteger));
        assert!(found.contains(&BeatGridViolationCode::DownbeatBarIndexExceedsJsonSafeInteger));
    }

    #[test]
    fn rejects_duplicate_hypothesis_ids() {
        let mut report = report();
        report.disposition = BeatGridDisposition::Ambiguous;
        report.hypotheses.push(hypothesis("grid-a", 700_000));
        assert!(codes(&report).contains(&BeatGridViolationCode::DuplicateHypothesisId));
    }

    #[test]
    fn rejects_pulse_anchor_at_exclusive_end_of_evidence_window() {
        let mut report = report();
        report.hypotheses[0].pulse_anchors[2].frame = report.evidence.covered_window.end_frame;
        assert!(codes(&report).contains(&BeatGridViolationCode::PulseAnchorOutOfEvidenceWindow));
    }

    #[test]
    fn rejects_meter_section_that_does_not_start_on_a_known_pulse() {
        let mut report = report();
        report.hypotheses[0].meter_map[0].start_beat_index = 7;
        assert!(codes(&report).contains(&BeatGridViolationCode::MeterStartBeatNotInPulseMap));
    }

    #[test]
    fn rejects_analysis_evidence_window_beyond_source_length() {
        let mut report = report();
        report.evidence.covered_window.end_frame = report.source.total_frames + 1;
        assert!(codes(&report).contains(&BeatGridViolationCode::InvalidEvidenceWindow));
    }

    #[test]
    fn requires_reason_for_not_evaluated_reports() {
        let mut report = report();
        report.disposition = BeatGridDisposition::NotEvaluated;
        report.hypotheses.clear();
        report.abstention_reason = None;
        assert!(codes(&report).contains(&BeatGridViolationCode::MissingAbstentionReason));
    }

    fn track(side: crate::transition::TrackSide) -> TrackWindowRef {
        let (sha, manifest_sha, sample_rate, total_frames) = match side {
            crate::transition::TrackSide::Source => {
                ("1".repeat(64), "a".repeat(64), 44_100, 10_000)
            },
            crate::transition::TrackSide::Target => {
                ("2".repeat(64), "b".repeat(64), 48_000, 20_000)
            },
        };
        TrackWindowRef {
            audio: AudioSourceRef {
                sha256: sha.clone(),
                sample_rate_hz: sample_rate,
                channels: 2,
                total_frames,
            },
            window: crate::transition::SampleWindow {
                start_frame: 100,
                end_frame: 900,
            },
            evidence: AnalysisEvidenceRef {
                manifest_sha256: manifest_sha,
                source_audio_sha256: sha,
                analysis_version: "auditory-scene-v0.1/config-17".to_string(),
                covered_window: crate::transition::SampleWindow {
                    start_frame: 0,
                    end_frame: total_frames,
                },
            },
        }
    }

    fn transition_contract() -> TransitionContract {
        TransitionContract {
            schema_version: crate::transition::TRANSITION_CONTRACT_SCHEMA_VERSION,
            transition_id: "transition-0001".to_string(),
            kind: crate::transition::TransitionKind::BeatmatchedBlend,
            source: track(crate::transition::TrackSide::Source),
            target: track(crate::transition::TrackSide::Target),
            evidence: crate::transition::TransitionEvidenceAssessment {
                timing: crate::transition::EvidenceDisposition::Supported,
                structure: crate::transition::EvidenceDisposition::Ambiguous,
                source_roles: crate::transition::EvidenceDisposition::NotEvaluated,
                target_roles: crate::transition::EvidenceDisposition::NotEvaluated,
                perceptual_fit: crate::transition::EvidenceDisposition::NotEvaluated,
            },
            alignment: Some(crate::transition::TransitionAlignment {
                source_anchor_frame: 400,
                target_anchor_frame: 600,
                source_grid_hypothesis_id: "source-grid".to_string(),
                target_grid_hypothesis_id: "target-grid".to_string(),
                beat_ordinal_milli: 16_000,
                max_phase_error_milli: 100,
                confidence_ppm: 850_000,
            }),
            protected_regions: Vec::new(),
            gain_envelopes: Vec::new(),
        }
    }

    fn report_for_track(
        track: &TrackWindowRef,
        hypothesis_id: &str,
        first_frame: u64,
        second_frame: u64,
    ) -> BeatGridReportV1 {
        BeatGridReportV1 {
            schema_version: BEAT_GRID_SCHEMA_VERSION,
            source: track.audio.clone(),
            evidence: track.evidence.clone(),
            disposition: BeatGridDisposition::Resolved,
            hypotheses: vec![BeatGridHypothesis {
                hypothesis_id: hypothesis_id.to_string(),
                confidence_ppm: 900_000,
                pulse_anchors: vec![
                    PulseAnchor {
                        frame: first_frame,
                        beat_index: 0,
                        confidence_ppm: 900_000,
                    },
                    PulseAnchor {
                        frame: second_frame,
                        beat_index: 1,
                        confidence_ppm: 900_000,
                    },
                ],
                downbeat_anchors: Vec::new(),
                tempo_interpolation: None,
                tempo_map: Vec::new(),
                meter_map: Vec::new(),
            }],
            abstention_reason: None,
        }
    }

    #[test]
    fn verifies_transition_against_exact_source_target_reports_and_phase() {
        let contract = transition_contract();
        let source_report =
            report_for_track(&contract.source, "source-grid", 100, 500);
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 700);
        assert!(verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        )
        .is_empty());
    }

    #[test]
    fn rejects_transition_when_selected_hypothesis_is_missing() {
        let mut contract = transition_contract();
        contract.alignment.as_mut().unwrap().source_grid_hypothesis_id =
            "missing-grid".to_string();
        let source_report =
            report_for_track(&contract.source, "source-grid", 100, 500);
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 700);
        let found = verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        );
        assert_eq!(
            found[0].code,
            TransitionGridBindingViolationCode::SourceHypothesisNotFound
        );
    }

    #[test]
    fn rejects_transition_when_analysis_manifest_does_not_match() {
        let contract = transition_contract();
        let mut source_report =
            report_for_track(&contract.source, "source-grid", 100, 500);
        source_report.evidence.manifest_sha256 = "c".repeat(64);
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 700);
        let found = verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        );
        assert_eq!(
            found[0].code,
            TransitionGridBindingViolationCode::SourceAnalysisReferenceMismatch
        );
    }

    #[test]
    fn rejects_transition_when_source_and_target_beat_phases_exceed_tolerance() {
        let contract = transition_contract();
        let source_report =
            report_for_track(&contract.source, "source-grid", 100, 500);
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 900);
        let found = verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        );
        assert_eq!(
            found[0].code,
            TransitionGridBindingViolationCode::BeatPhaseMismatch
        );
    }

    #[test]
    fn abstains_from_phase_interpolation_across_a_multi_beat_pulse_gap() {
        let contract = transition_contract();
        let source_report = {
            let mut report =
                report_for_track(&contract.source, "source-grid", 100, 900);
            report.hypotheses[0].pulse_anchors[1].beat_index = 2;
            report
        };
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 700);
        let found = verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        );
        assert_eq!(
            found[0].code,
            TransitionGridBindingViolationCode::SourceAnchorOutsidePulseSupport
        );
    }

    #[test]
    fn rejects_alignment_outside_selected_grid_pulse_support() {
        let mut contract = transition_contract();
        contract.alignment.as_mut().unwrap().source_anchor_frame = 800;
        let source_report =
            report_for_track(&contract.source, "source-grid", 100, 500);
        let target_report =
            report_for_track(&contract.target, "target-grid", 300, 700);
        let found = verify_transition_grid_binding(
            &contract,
            &source_report,
            &target_report,
        );
        assert_eq!(
            found[0].code,
            TransitionGridBindingViolationCode::SourceAnchorOutsidePulseSupport
        );
    }

    #[test]
    fn checked_in_valid_fixture_deserializes_and_validates() {
        let json = include_str!("../fixtures/beat-grid-v1/valid.json");
        let report: BeatGridReportV1 = serde_json::from_str(json).unwrap();
        assert!(validate_beat_grid_report(&report).is_empty());
    }

    #[test]
    fn checked_in_ambiguous_fixture_preserves_alternatives() {
        let json = include_str!("../fixtures/beat-grid-v1/ambiguous.json");
        let report: BeatGridReportV1 = serde_json::from_str(json).unwrap();
        assert!(validate_beat_grid_report(&report).is_empty());
        assert_eq!(report.disposition, BeatGridDisposition::Ambiguous);
        assert_eq!(report.hypotheses.len(), 2);
    }

    #[test]
    fn checked_in_abstention_fixture_has_no_grid_hypotheses() {
        let json = include_str!("../fixtures/beat-grid-v1/abstained.json");
        let report: BeatGridReportV1 = serde_json::from_str(json).unwrap();
        assert!(validate_beat_grid_report(&report).is_empty());
        assert!(report.hypotheses.is_empty());
    }
}

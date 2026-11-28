// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Versioned DJ transition contracts and a deterministic, fail-closed validator.
//!
//! This module is deliberately target-neutral: it contains no audio I/O, DSP,
//! model inference, device, browser, or clock-reading code. It describes a
//! proposed transition over immutable source assets and evidence manifests.
//!
//! Validation does not authorize playback. A separate runtime must check
//! fresh execution authority, device state, and runtime preconditions at the
//! point of action. This module also does not establish that the supplied
//! evidence is truthful; it checks structural linkage and consistency only.
//!
//! Time is represented as source-local integer frame indices. Source and
//! target windows are never compared as if they shared a sample clock.

use serde::{Deserialize, Serialize};

/// Current schema version understood by this crate.
pub const TRANSITION_CONTRACT_SCHEMA_VERSION: u16 = 1;

/// Largest integer exactly representable by ECMAScript Number / JSON consumers.
pub const MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Gain values are milli-decibels relative to the deck's declared unity gain.
/// The structural envelope is intentionally generous; a runtime may impose a
/// narrower device-specific range, but must not silently rewrite this contract.
pub const MIN_GAIN_MILLIDECIBELS: i32 = -120_000;
pub const MAX_GAIN_MILLIDECIBELS: i32 = 12_000;

/// A half-open interval of frames, `[start_frame, end_frame)`.
///
/// Frame numbers are local to one source audio asset and its sample clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampleWindow {
    pub start_frame: u64,
    pub end_frame: u64,
}

impl SampleWindow {
    /// True only for a non-empty, ordered half-open interval.
    pub fn is_non_empty(self) -> bool {
        self.start_frame < self.end_frame
    }

    /// Whether this window fully contains another half-open window.
    pub fn contains(self, other: Self) -> bool {
        self.start_frame <= other.start_frame && other.end_frame <= self.end_frame
    }

    /// Whether a frame is in this half-open interval.
    pub fn contains_frame(self, frame: u64) -> bool {
        self.start_frame <= frame && frame < self.end_frame
    }
}

/// Immutable source-audio identity and its local sample clock.
///
/// No audio bytes are carried in a transition contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioSourceRef {
    /// Lowercase, 64-character SHA-256 hexadecimal digest of the source bytes.
    pub sha256: String,
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Total decoded frames per channel for the declared source representation.
    pub total_frames: u64,
}

/// A reference to the exact analysis manifest used to reason about a source.
///
/// The manifest hash identifies the immutable analysis output. Its source hash
/// and covered frame range bind the evidence to a specific source and interval.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisEvidenceRef {
    /// Lowercase, 64-character SHA-256 hexadecimal digest of the manifest.
    pub manifest_sha256: String,
    /// Must equal the adjacent `AudioSourceRef.sha256`.
    pub source_audio_sha256: String,
    /// Non-empty, immutable analysis implementation/configuration identity.
    pub analysis_version: String,
    /// Source-local half-open frame window covered by this evidence manifest.
    pub covered_window: SampleWindow,
}

/// One deck-side source and the portion proposed for a transition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackWindowRef {
    pub audio: AudioSourceRef,
    pub window: SampleWindow,
    pub evidence: AnalysisEvidenceRef,
}

/// The musical form of the proposed hand-off.
///
/// This is descriptive intent; the validator does not score musical quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionKind {
    BeatmatchedBlend,
    PhraseHandoff,
    HardCut,
    DropSwap,
    BreakdownSwap,
    AmbientBridge,
    Other,
}

/// Which side of the transition an authored region or gain envelope addresses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackSide {
    Source,
    Target,
}

/// How much the current analysis supports a class of musical evidence.
///
/// `NotEvaluated` is not success. Consumers must not silently upgrade it to
/// `Supported` or treat missing analysis as a confidence value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceDisposition {
    Supported,
    Ambiguous,
    Unsupported,
    NotEvaluated,
}

/// Explicit evidence state consumed by the transition planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionEvidenceAssessment {
    pub timing: EvidenceDisposition,
    pub structure: EvidenceDisposition,
    pub source_roles: EvidenceDisposition,
    pub target_roles: EvidenceDisposition,
    pub perceptual_fit: EvidenceDisposition,
}

/// An optional alignment hypothesis. Frame anchors are source-local, and the
/// two sides are validated against their own clocks independently. Hypothesis
/// IDs resolve within the corresponding source/target analysis evidence manifests.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionAlignment {
    pub source_anchor_frame: u64,
    pub target_anchor_frame: u64,
    pub source_grid_hypothesis_id: String,
    pub target_grid_hypothesis_id: String,
    /// Transition-timeline beat ordinal in milli-beats; not a track-local index.
    pub transition_beat_ordinal_milli: i64,
    /// Maximum allowed circular phase difference between the selected source/target
    /// pulse hypotheses, in milli-beats (0..=500, half a beat).
    pub max_phase_error_milli: u16,
    /// Maximum permitted absolute tempo difference at the two alignment anchors,
    /// in milli-BPM. Both selected hypotheses must provide tempo-map support.
    pub max_tempo_delta_milli_bpm: u32,
    /// Producer-supplied score in parts per million, 0..=1,000,000.
    /// It is not a calibrated probability unless a separate calibration record says so.
    pub confidence_ppm: u32,
}

/// Musical content that the transition proposal must preserve or avoid masking.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtectedMusicProperty {
    LeadVocal,
    BackingVocal,
    Melody,
    Hook,
    BassLine,
    Kick,
    Snare,
    Percussion,
    HarmonicBed,
    Other,
}

/// A source-local interval that a transition proposal marks as protected.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtectedRegion {
    pub side: TrackSide,
    pub window: SampleWindow,
    pub property: ProtectedMusicProperty,
    /// Non-empty stable reason tag, e.g. `lead_vocal_phrase`.
    pub reason_code: String,
}

/// One gain point at a source-local frame position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GainPoint {
    pub frame: u64,
    pub gain_millidecibels: i32,
}

/// A piecewise-linear gain proposal for one side of the transition.
///
/// The points are data only. No audio is rendered and no gain is applied here.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GainEnvelope {
    pub side: TrackSide,
    pub points: Vec<GainPoint>,
}

/// A complete versioned proposal for one musical transition.
///
/// There is deliberately no `Default` implementation and no authority field:
/// deserializing or validating a proposal cannot manufacture permission to
/// execute it. The execution system must acquire separate, current authority.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionContract {
    pub schema_version: u16,
    pub transition_id: String,
    pub kind: TransitionKind,
    pub source: TrackWindowRef,
    pub target: TrackWindowRef,
    pub evidence: TransitionEvidenceAssessment,
    pub alignment: Option<TransitionAlignment>,
    pub protected_regions: Vec<ProtectedRegion>,
    pub gain_envelopes: Vec<GainEnvelope>,
}

/// Stable machine-readable validation failures.
///
/// The order in `validate_transition_contract` is deterministic: contract
/// identity/version, source, target, optional alignment, protected regions,
/// then gain envelopes in input order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransitionViolationCode {
    UnsupportedSchemaVersion,
    EmptyTransitionId,
    InvalidSourceSha256,
    InvalidSourceSampleRate,
    InvalidSourceChannelCount,
    EmptySourceAsset,
    SourceFrameCountExceedsJsonSafeInteger,
    EmptySourceWindow,
    SourceWindowOutOfBounds,
    InvalidSourceAnalysisSha256,
    SourceAnalysisDigestMismatch,
    EmptySourceAnalysisVersion,
    InvalidSourceAnalysisWindow,
    SourceAnalysisDoesNotCoverWindow,
    InvalidTargetSha256,
    InvalidTargetSampleRate,
    InvalidTargetChannelCount,
    EmptyTargetAsset,
    TargetFrameCountExceedsJsonSafeInteger,
    EmptyTargetWindow,
    TargetWindowOutOfBounds,
    InvalidTargetAnalysisSha256,
    TargetAnalysisDigestMismatch,
    EmptyTargetAnalysisVersion,
    InvalidTargetAnalysisWindow,
    TargetAnalysisDoesNotCoverWindow,
    SourceAlignmentAnchorOutOfWindow,
    TargetAlignmentAnchorOutOfWindow,
    EmptySourceGridHypothesisId,
    EmptyTargetGridHypothesisId,
    AlignmentPhaseToleranceOutOfRange,
    AlignmentBeatOrdinalExceedsJsonSafeInteger,
    AlignmentConfidenceOutOfRange,
    EmptyProtectedRegionReason,
    EmptyProtectedRegionWindow,
    ProtectedRegionOutOfBounds,
    DuplicateGainEnvelopeSide,
    GainEnvelopeTooShort,
    GainPointOutOfWindow,
    GainPointsNotStrictlyIncreasing,
    GainOutOfRange,
}

/// One stable validation result with a human-readable field path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransitionViolation {
    pub code: TransitionViolationCode,
    pub path: String,
}

/// Returns the ordered validation failures for a transition contract.
///
/// This function is pure, deterministic, allocation-bounded by input vector
/// sizes, and has no I/O or authority side effects.
pub fn validate_transition_contract(
    contract: &TransitionContract,
) -> Vec<TransitionViolation> {
    let mut violations = Vec::new();

    if contract.schema_version != TRANSITION_CONTRACT_SCHEMA_VERSION {
        push_violation(
            &mut violations,
            TransitionViolationCode::UnsupportedSchemaVersion,
            "schema_version",
        );
    }
    if contract.transition_id.trim().is_empty() {
        push_violation(
            &mut violations,
            TransitionViolationCode::EmptyTransitionId,
            "transition_id",
        );
    }

    validate_track_window(&contract.source, TrackSide::Source, &mut violations);
    validate_track_window(&contract.target, TrackSide::Target, &mut violations);

    if let Some(alignment) = contract.alignment.as_ref() {
        if !contract.source.window.contains_frame(alignment.source_anchor_frame) {
            push_violation(
                &mut violations,
                TransitionViolationCode::SourceAlignmentAnchorOutOfWindow,
                "alignment.source_anchor_frame",
            );
        }
        if !contract.target.window.contains_frame(alignment.target_anchor_frame) {
            push_violation(
                &mut violations,
                TransitionViolationCode::TargetAlignmentAnchorOutOfWindow,
                "alignment.target_anchor_frame",
            );
        }
        if alignment.source_grid_hypothesis_id.trim().is_empty() {
            push_violation(
                &mut violations,
                TransitionViolationCode::EmptySourceGridHypothesisId,
                "alignment.source_grid_hypothesis_id",
            );
        }
        if alignment.target_grid_hypothesis_id.trim().is_empty() {
            push_violation(
                &mut violations,
                TransitionViolationCode::EmptyTargetGridHypothesisId,
                "alignment.target_grid_hypothesis_id",
            );
        }
        if alignment.max_phase_error_milli > 500 {
            push_violation(
                &mut violations,
                TransitionViolationCode::AlignmentPhaseToleranceOutOfRange,
                "alignment.max_phase_error_milli",
            );
        }
        if alignment.transition_beat_ordinal_milli.unsigned_abs() > MAX_SAFE_JSON_INTEGER {
            push_violation(
                &mut violations,
                TransitionViolationCode::AlignmentBeatOrdinalExceedsJsonSafeInteger,
                "alignment.transition_beat_ordinal_milli",
            );
        }
        if alignment.confidence_ppm > 1_000_000 {
            push_violation(
                &mut violations,
                TransitionViolationCode::AlignmentConfidenceOutOfRange,
                "alignment.confidence_ppm",
            );
        }
    }

    for (index, region) in contract.protected_regions.iter().enumerate() {
        let path = match region.side {
            TrackSide::Source => "protected_regions.source",
            TrackSide::Target => "protected_regions.target",
        };
        if region.reason_code.trim().is_empty() {
            push_violation(
                &mut violations,
                TransitionViolationCode::EmptyProtectedRegionReason,
                format!("{path}[{index}].reason_code"),
            );
        }
        if !region.window.is_non_empty() {
            push_violation(
                &mut violations,
                TransitionViolationCode::EmptyProtectedRegionWindow,
                format!("{path}[{index}].window"),
            );
        } else if !track_window(contract, region.side).window.contains(region.window) {
            push_violation(
                &mut violations,
                TransitionViolationCode::ProtectedRegionOutOfBounds,
                format!("{path}[{index}].window"),
            );
        }
    }

    let mut saw_source_envelope = false;
    let mut saw_target_envelope = false;
    for (index, envelope) in contract.gain_envelopes.iter().enumerate() {
        let duplicate = match envelope.side {
            TrackSide::Source => {
                let old = saw_source_envelope;
                saw_source_envelope = true;
                old
            }
            TrackSide::Target => {
                let old = saw_target_envelope;
                saw_target_envelope = true;
                old
            }
        };
        if duplicate {
            push_violation(
                &mut violations,
                TransitionViolationCode::DuplicateGainEnvelopeSide,
                format!("gain_envelopes[{index}].side"),
            );
        }

        if envelope.points.len() < 2 {
            push_violation(
                &mut violations,
                TransitionViolationCode::GainEnvelopeTooShort,
                format!("gain_envelopes[{index}].points"),
            );
        }

        let window = track_window(contract, envelope.side).window;
        let mut previous_frame: Option<u64> = None;
        for (point_index, point) in envelope.points.iter().enumerate() {
            if !window.contains_frame(point.frame) {
                push_violation(
                    &mut violations,
                    TransitionViolationCode::GainPointOutOfWindow,
                    format!("gain_envelopes[{index}].points[{point_index}].frame"),
                );
            }
            if previous_frame.is_some_and(|previous| point.frame <= previous) {
                push_violation(
                    &mut violations,
                    TransitionViolationCode::GainPointsNotStrictlyIncreasing,
                    format!("gain_envelopes[{index}].points[{point_index}].frame"),
                );
            }
            if !(MIN_GAIN_MILLIDECIBELS..=MAX_GAIN_MILLIDECIBELS)
                .contains(&point.gain_millidecibels)
            {
                push_violation(
                    &mut violations,
                    TransitionViolationCode::GainOutOfRange,
                    format!("gain_envelopes[{index}].points[{point_index}].gain_millidecibels"),
                );
            }
            previous_frame = Some(point.frame);
        }
    }

    violations
}

fn track_window(contract: &TransitionContract, side: TrackSide) -> &TrackWindowRef {
    match side {
        TrackSide::Source => &contract.source,
        TrackSide::Target => &contract.target,
    }
}

fn validate_track_window(
    track: &TrackWindowRef,
    side: TrackSide,
    violations: &mut Vec<TransitionViolation>,
) {
    let (invalid_sha, invalid_rate, invalid_channels, empty_asset, empty_window, out_of_bounds,
        invalid_manifest_sha, digest_mismatch, empty_version, invalid_analysis_window,
        analysis_does_not_cover, prefix) = match side {
        TrackSide::Source => (
            TransitionViolationCode::InvalidSourceSha256,
            TransitionViolationCode::InvalidSourceSampleRate,
            TransitionViolationCode::InvalidSourceChannelCount,
            TransitionViolationCode::EmptySourceAsset,
            TransitionViolationCode::EmptySourceWindow,
            TransitionViolationCode::SourceWindowOutOfBounds,
            TransitionViolationCode::InvalidSourceAnalysisSha256,
            TransitionViolationCode::SourceAnalysisDigestMismatch,
            TransitionViolationCode::EmptySourceAnalysisVersion,
            TransitionViolationCode::InvalidSourceAnalysisWindow,
            TransitionViolationCode::SourceAnalysisDoesNotCoverWindow,
            "source",
        ),
        TrackSide::Target => (
            TransitionViolationCode::InvalidTargetSha256,
            TransitionViolationCode::InvalidTargetSampleRate,
            TransitionViolationCode::InvalidTargetChannelCount,
            TransitionViolationCode::EmptyTargetAsset,
            TransitionViolationCode::EmptyTargetWindow,
            TransitionViolationCode::TargetWindowOutOfBounds,
            TransitionViolationCode::InvalidTargetAnalysisSha256,
            TransitionViolationCode::TargetAnalysisDigestMismatch,
            TransitionViolationCode::EmptyTargetAnalysisVersion,
            TransitionViolationCode::InvalidTargetAnalysisWindow,
            TransitionViolationCode::TargetAnalysisDoesNotCoverWindow,
            "target",
        ),
    };

    if !is_lowercase_sha256(&track.audio.sha256) {
        push_violation(violations, invalid_sha, format!("{prefix}.audio.sha256"));
    }
    if track.audio.sample_rate_hz == 0 {
        push_violation(
            violations,
            invalid_rate,
            format!("{prefix}.audio.sample_rate_hz"),
        );
    }
    if track.audio.channels == 0 {
        push_violation(
            violations,
            invalid_channels,
            format!("{prefix}.audio.channels"),
        );
    }
    if track.audio.total_frames == 0 {
        push_violation(
            violations,
            empty_asset,
            format!("{prefix}.audio.total_frames"),
        );
    }
    if track.audio.total_frames > MAX_SAFE_JSON_INTEGER {
        let code = match side {
            TrackSide::Source => {
                TransitionViolationCode::SourceFrameCountExceedsJsonSafeInteger
            }
            TrackSide::Target => {
                TransitionViolationCode::TargetFrameCountExceedsJsonSafeInteger
            }
        };
        push_violation(
            violations,
            code,
            format!("{prefix}.audio.total_frames"),
        );
    }

    if !track.window.is_non_empty() {
        push_violation(
            violations,
            empty_window,
            format!("{prefix}.window"),
        );
    } else if track.window.end_frame > track.audio.total_frames {
        push_violation(
            violations,
            out_of_bounds,
            format!("{prefix}.window"),
        );
    }

    if !is_lowercase_sha256(&track.evidence.manifest_sha256) {
        push_violation(
            violations,
            invalid_manifest_sha,
            format!("{prefix}.evidence.manifest_sha256"),
        );
    }
    if track.evidence.source_audio_sha256 != track.audio.sha256 {
        push_violation(
            violations,
            digest_mismatch,
            format!("{prefix}.evidence.source_audio_sha256"),
        );
    }
    if track.evidence.analysis_version.trim().is_empty() {
        push_violation(
            violations,
            empty_version,
            format!("{prefix}.evidence.analysis_version"),
        );
    }

    if !track.evidence.covered_window.is_non_empty()
        || track.evidence.covered_window.end_frame > track.audio.total_frames
    {
        push_violation(
            violations,
            invalid_analysis_window,
            format!("{prefix}.evidence.covered_window"),
        );
    } else if track.window.is_non_empty()
        && !track.evidence.covered_window.contains(track.window)
    {
        push_violation(
            violations,
            analysis_does_not_cover,
            format!("{prefix}.evidence.covered_window"),
        );
    }
}

fn is_lowercase_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn push_violation(
    violations: &mut Vec<TransitionViolation>,
    code: TransitionViolationCode,
    path: impl Into<String>,
) {
    violations.push(TransitionViolation {
        code,
        path: path.into(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(hash: &str, sample_rate_hz: u32, frames: u64) -> TrackWindowRef {
        TrackWindowRef {
            audio: AudioSourceRef {
                sha256: hash.to_string(),
                sample_rate_hz,
                channels: 2,
                total_frames: frames,
            },
            window: SampleWindow {
                start_frame: 100,
                end_frame: 900,
            },
            evidence: AnalysisEvidenceRef {
                manifest_sha256: "a".repeat(64),
                source_audio_sha256: hash.to_string(),
                analysis_version: "auditory-scene-v0.1/config-17".to_string(),
                covered_window: SampleWindow {
                    start_frame: 0,
                    end_frame: frames,
                },
            },
        }
    }

    fn valid_contract() -> TransitionContract {
        TransitionContract {
            schema_version: TRANSITION_CONTRACT_SCHEMA_VERSION,
            transition_id: "transition-0001".to_string(),
            kind: TransitionKind::BeatmatchedBlend,
            source: source(&"1".repeat(64), 44_100, 10_000),
            target: source(&"2".repeat(64), 48_000, 20_000),
            evidence: TransitionEvidenceAssessment {
                timing: EvidenceDisposition::Supported,
                structure: EvidenceDisposition::Ambiguous,
                source_roles: EvidenceDisposition::NotEvaluated,
                target_roles: EvidenceDisposition::Unsupported,
                perceptual_fit: EvidenceDisposition::NotEvaluated,
            },
            alignment: Some(TransitionAlignment {
                source_anchor_frame: 400,
                target_anchor_frame: 600,
                source_grid_hypothesis_id: "source-grid-primary".to_string(),
                target_grid_hypothesis_id: "target-grid-primary".to_string(),
                transition_beat_ordinal_milli: 16_000,
                max_phase_error_milli: 100,
                max_tempo_delta_milli_bpm: 1_000,
                confidence_ppm: 850_000,
            }),
            protected_regions: vec![ProtectedRegion {
                side: TrackSide::Source,
                window: SampleWindow {
                    start_frame: 300,
                    end_frame: 500,
                },
                property: ProtectedMusicProperty::LeadVocal,
                reason_code: "lead_vocal_phrase".to_string(),
            }],
            gain_envelopes: vec![
                GainEnvelope {
                    side: TrackSide::Source,
                    points: vec![
                        GainPoint {
                            frame: 200,
                            gain_millidecibels: 0,
                        },
                        GainPoint {
                            frame: 800,
                            gain_millidecibels: -6_000,
                        },
                    ],
                },
                GainEnvelope {
                    side: TrackSide::Target,
                    points: vec![
                        GainPoint {
                            frame: 200,
                            gain_millidecibels: -12_000,
                        },
                        GainPoint {
                            frame: 800,
                            gain_millidecibels: 0,
                        },
                    ],
                },
            ],
        }
    }

    fn codes(contract: &TransitionContract) -> Vec<TransitionViolationCode> {
        validate_transition_contract(contract)
            .into_iter()
            .map(|violation| violation.code)
            .collect()
    }

    #[test]
    fn valid_contract_with_independent_source_clocks_is_accepted() {
        let contract = valid_contract();
        assert!(validate_transition_contract(&contract).is_empty());
        assert_ne!(
            contract.source.audio.sample_rate_hz,
            contract.target.audio.sample_rate_hz
        );
    }

    #[test]
    fn rejects_unknown_schema_version() {
        let mut contract = valid_contract();
        contract.schema_version = TRANSITION_CONTRACT_SCHEMA_VERSION + 1;
        assert_eq!(
            codes(&contract),
            vec![TransitionViolationCode::UnsupportedSchemaVersion]
        );
    }

    #[test]
    fn rejects_empty_transition_id() {
        let mut contract = valid_contract();
        contract.transition_id = "  ".to_string();
        assert_eq!(
            codes(&contract),
            vec![TransitionViolationCode::EmptyTransitionId]
        );
    }

    #[test]
    fn rejects_malformed_source_digest_and_analysis_source_mismatch() {
        let mut contract = valid_contract();
        contract.source.audio.sha256 = "not-a-digest".to_string();
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::InvalidSourceSha256));
        assert!(found.contains(&TransitionViolationCode::SourceAnalysisDigestMismatch));
    }

    #[test]
    fn rejects_frame_count_above_exact_json_integer_range() {
        let mut contract = valid_contract();
        contract.source.audio.total_frames = MAX_SAFE_JSON_INTEGER + 1;
        contract.target.audio.total_frames = MAX_SAFE_JSON_INTEGER + 1;
        let found = codes(&contract);
        assert!(found.contains(
            &TransitionViolationCode::SourceFrameCountExceedsJsonSafeInteger
        ));
        assert!(found.contains(
            &TransitionViolationCode::TargetFrameCountExceedsJsonSafeInteger
        ));
    }

    #[test]
    fn rejects_alignment_ordinal_above_exact_json_integer_range() {
        let mut contract = valid_contract();
        contract.alignment.as_mut().unwrap().transition_beat_ordinal_milli =
            (MAX_SAFE_JSON_INTEGER as i64) + 1;
        assert!(codes(&contract)
            .contains(&TransitionViolationCode::AlignmentBeatOrdinalExceedsJsonSafeInteger));
    }

    #[test]
    fn rejects_empty_and_out_of_bounds_windows() {
        let mut empty = valid_contract();
        empty.source.window = SampleWindow {
            start_frame: 20,
            end_frame: 20,
        };
        assert!(codes(&empty).contains(&TransitionViolationCode::EmptySourceWindow));

        let mut overflow = valid_contract();
        overflow.target.window = SampleWindow {
            start_frame: 100,
            end_frame: overflow.target.audio.total_frames + 1,
        };
        assert!(codes(&overflow).contains(&TransitionViolationCode::TargetWindowOutOfBounds));
    }

    #[test]
    fn rejects_analysis_manifest_that_does_not_cover_the_transition() {
        let mut contract = valid_contract();
        contract.source.evidence.covered_window = SampleWindow {
            start_frame: 150,
            end_frame: 800,
        };
        assert!(codes(&contract)
            .contains(&TransitionViolationCode::SourceAnalysisDoesNotCoverWindow));
    }

    #[test]
    fn rejects_alignment_anchors_outside_their_own_track_clocks() {
        let mut contract = valid_contract();
        contract.alignment.as_mut().unwrap().source_anchor_frame = 900;
        contract.alignment.as_mut().unwrap().target_anchor_frame = 99;
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::SourceAlignmentAnchorOutOfWindow));
        assert!(found.contains(&TransitionViolationCode::TargetAlignmentAnchorOutOfWindow));
    }

    #[test]
    fn rejects_empty_grid_hypothesis_references() {
        let mut contract = valid_contract();
        let alignment = contract.alignment.as_mut().unwrap();
        alignment.source_grid_hypothesis_id = " ".to_string();
        alignment.target_grid_hypothesis_id.clear();
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::EmptySourceGridHypothesisId));
        assert!(found.contains(&TransitionViolationCode::EmptyTargetGridHypothesisId));
    }

    #[test]
    fn rejects_alignment_phase_tolerance_above_half_a_beat() {
        let mut contract = valid_contract();
        contract.alignment.as_mut().unwrap().max_phase_error_milli = 501;
        assert!(codes(&contract)
            .contains(&TransitionViolationCode::AlignmentPhaseToleranceOutOfRange));
    }

    #[test]
    fn rejects_unbounded_alignment_confidence() {
        let mut contract = valid_contract();
        contract.alignment.as_mut().unwrap().confidence_ppm = 1_000_001;
        assert!(codes(&contract)
            .contains(&TransitionViolationCode::AlignmentConfidenceOutOfRange));
    }

    #[test]
    fn rejects_protected_regions_outside_declared_transition_window() {
        let mut contract = valid_contract();
        contract.protected_regions[0].window = SampleWindow {
            start_frame: 50,
            end_frame: 150,
        };
        assert!(codes(&contract).contains(&TransitionViolationCode::ProtectedRegionOutOfBounds));
    }

    #[test]
    fn rejects_non_monotonic_and_out_of_range_gain_points() {
        let mut contract = valid_contract();
        contract.gain_envelopes[0].points = vec![
            GainPoint {
                frame: 500,
                gain_millidecibels: 0,
            },
            GainPoint {
                frame: 400,
                gain_millidecibels: MAX_GAIN_MILLIDECIBELS + 1,
            },
        ];
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::GainPointsNotStrictlyIncreasing));
        assert!(found.contains(&TransitionViolationCode::GainOutOfRange));
    }

    #[test]
    fn rejects_duplicate_gain_envelopes_for_one_side() {
        let mut contract = valid_contract();
        let duplicate = contract.gain_envelopes[0].clone();
        contract.gain_envelopes.push(duplicate);
        assert!(codes(&contract).contains(&TransitionViolationCode::DuplicateGainEnvelopeSide));
    }

    #[test]
    fn rejects_gain_point_at_exclusive_end_frame() {
        let mut contract = valid_contract();
        contract.gain_envelopes[0].points[1].frame = contract.source.window.end_frame;
        assert!(codes(&contract).contains(&TransitionViolationCode::GainPointOutOfWindow));
    }

    #[test]
    fn round_trips_without_creating_authority_or_default_evidence() {
        let contract = valid_contract();
        let json = serde_json::to_string(&contract).unwrap();
        let decoded: TransitionContract = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, contract);
        assert_eq!(decoded.evidence.source_roles, EvidenceDisposition::NotEvaluated);
        assert!(json.contains("\"schema_version\":1"));
        assert!(!json.to_ascii_lowercase().contains("authorization"));
    }

    #[test]
    fn rejects_unknown_contract_fields_instead_of_silently_ignoring_them() {
        let mut value = serde_json::to_value(valid_contract()).unwrap();
        value.as_object_mut().unwrap().insert(
            "execution_authorized".to_string(),
            serde_json::Value::Bool(true),
        );
        assert!(serde_json::from_value::<TransitionContract>(value).is_err());
    }

    #[test]
    fn rejects_unknown_nested_source_fields() {
        let mut value = serde_json::to_value(valid_contract()).unwrap();
        value["source"]["audio"]
            .as_object_mut()
            .unwrap()
            .insert("source_mutated".to_string(), serde_json::Value::Bool(true));
        assert!(serde_json::from_value::<TransitionContract>(value).is_err());
    }

    #[test]
    fn rejects_zero_sample_rate_and_channel_count() {
        let mut contract = valid_contract();
        contract.source.audio.sample_rate_hz = 0;
        contract.target.audio.channels = 0;
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::InvalidSourceSampleRate));
        assert!(found.contains(&TransitionViolationCode::InvalidTargetChannelCount));
    }

    #[test]
    fn rejects_empty_analysis_version_and_invalid_manifest_digest() {
        let mut contract = valid_contract();
        contract.source.evidence.analysis_version = " ".to_string();
        contract.target.evidence.manifest_sha256 = "G".repeat(64);
        let found = codes(&contract);
        assert!(found.contains(&TransitionViolationCode::EmptySourceAnalysisVersion));
        assert!(found.contains(&TransitionViolationCode::InvalidTargetAnalysisSha256));
    }

    #[test]
    fn deserialization_requires_source_evidence_and_does_not_invent_it() {
        let mut value = serde_json::to_value(valid_contract()).unwrap();
        value["source"]
            .as_object_mut()
            .unwrap()
            .remove("evidence");
        assert!(serde_json::from_value::<TransitionContract>(value).is_err());
    }

    #[test]
    fn checked_in_valid_json_fixture_deserializes_and_validates() {
        let json = include_str!("../fixtures/transition-contract-v1/valid.json");
        let contract: TransitionContract = serde_json::from_str(json).unwrap();
        assert!(validate_transition_contract(&contract).is_empty());
        assert_eq!(contract.schema_version, TRANSITION_CONTRACT_SCHEMA_VERSION);
    }

    #[test]
    fn checked_in_unsupported_version_fixture_fails_closed() {
        let json = include_str!(
            "../fixtures/transition-contract-v1/unsupported-schema-version.json"
        );
        let contract: TransitionContract = serde_json::from_str(json).unwrap();
        assert_eq!(
            codes(&contract),
            vec![TransitionViolationCode::UnsupportedSchemaVersion]
        );
    }

    #[test]
    fn checked_in_missing_evidence_fixture_cannot_be_deserialized() {
        let json = include_str!(
            "../fixtures/transition-contract-v1/missing-source-evidence.json"
        );
        assert!(serde_json::from_str::<TransitionContract>(json).is_err());
    }

    #[test]
    fn reports_violations_in_stable_field_order() {
        let mut contract = valid_contract();
        contract.schema_version = 9;
        contract.transition_id.clear();
        contract.source.audio.sample_rate_hz = 0;
        contract.target.audio.channels = 0;
        let found = validate_transition_contract(&contract);
        assert_eq!(
            found.iter().map(|v| v.code).collect::<Vec<_>>(),
            vec![
                TransitionViolationCode::UnsupportedSchemaVersion,
                TransitionViolationCode::EmptyTransitionId,
                TransitionViolationCode::InvalidSourceSampleRate,
                TransitionViolationCode::InvalidTargetChannelCount,
            ]
        );
        assert_eq!(found[0].path, "schema_version");
        assert_eq!(found[1].path, "transition_id");
    }
}

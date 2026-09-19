// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Executable cross-language parity receipts for symbolic pitch metrics.
//!
//! This module deliberately separates three statements:
//! 1. Melothaea has a frozen native metric definition;
//! 2. an exact external implementation produced a frozen reference bundle;
//! 3. the two outputs agree within an explicitly declared tolerance.
//!
//! Only (3), after validating (1) and (2), may receive `Parity` disposition.
//! No parity result is a musical-quality, preference, or product-authority claim.

use crate::evidence_digest::benchmark_symbolic_metrics::{
    BenchmarkScaleModeV1, BenchmarkScaleV1, measure_symbolic_pitch_metrics,
    pitch_in_scale_rate,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_music_theory::{
    Duration, Emphasis, Key, PartId, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
};

pub const MUSPY_PITCH_PARITY_VERSION: &str = "melothaea-muspy-pitch-parity-v1";
pub const MUSPY_REFERENCE_REVISION: &str = "2e1dc660dde6974c1693147f528d7437019d9580";
pub const MUSPY_METRICS_SOURCE_BLOB: &str = "b428fe331f7a8b64380a7d9a12f028e4f12c3033";
pub const DEFAULT_MUSPY_FLOAT_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyReferenceExecutionModeV1 {
    /// Execute the exact audited `muspy/metrics/metrics.py` file while supplying
    /// only the `Music` type import as a minimal stub. Metric function bodies are
    /// loaded from the audited external file, not copied into the runner.
    ExactMetricsFileWithTypeStub,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyPitchReferenceValuesV1 {
    pub n_pitches_used: u16,
    pub n_pitch_classes_used: u16,
    pub pitch_range: u16,
    pub pitch_entropy: f64,
    pub pitch_class_entropy: f64,
    pub scale_consistency: f64,
    pub c_major_pitch_in_scale_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyPitchReferenceFixtureV1 {
    pub fixture_id: String,
    pub midis: Vec<u8>,
    pub external: MuspyPitchReferenceValuesV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyPitchReferenceBundleV1 {
    pub bundle_version: String,
    pub source_revision: String,
    pub metrics_source_blob: String,
    pub execution_mode: MuspyReferenceExecutionModeV1,
    pub python_implementation: String,
    pub python_version: String,
    pub numpy_version: String,
    pub fixtures: Vec<MuspyPitchReferenceFixtureV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyPitchReferenceIssueV1 {
    WrongBundleVersion { found: String },
    WrongSourceRevision { found: String },
    WrongMetricsSourceBlob { found: String },
    EmptyRuntimeField { field: String },
    EmptyFixtureSet,
    FixturesNotCanonical,
    DuplicateFixtureId { fixture_id: String },
    EmptyFixture { fixture_id: String },
    PitchOutsideMidiRange { fixture_id: String, pitch: u8 },
    NonFiniteExternalValue { fixture_id: String, metric_id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyMetricMismatchV1 {
    pub fixture_id: String,
    pub metric_id: String,
    pub rust_value: f64,
    pub external_value: f64,
    pub absolute_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyPitchParityDispositionV1 {
    Parity,
    Mismatch,
    InvalidReference,
    InvalidTolerance,
    NativeMeasurementFailure,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyPitchParityReportV1 {
    pub parity_version: String,
    pub source_revision: String,
    pub metrics_source_blob: String,
    pub float_tolerance: f64,
    pub fixture_count: usize,
    pub issues: Vec<MuspyPitchReferenceIssueV1>,
    pub mismatches: Vec<MuspyMetricMismatchV1>,
    pub disposition: MuspyPitchParityDispositionV1,
}

impl MuspyPitchReferenceBundleV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }

    pub fn validate(&self) -> Vec<MuspyPitchReferenceIssueV1> {
        let mut issues = Vec::new();
        if self.bundle_version != MUSPY_PITCH_PARITY_VERSION {
            issues.push(MuspyPitchReferenceIssueV1::WrongBundleVersion {
                found: self.bundle_version.clone(),
            });
        }
        if self.source_revision != MUSPY_REFERENCE_REVISION {
            issues.push(MuspyPitchReferenceIssueV1::WrongSourceRevision {
                found: self.source_revision.clone(),
            });
        }
        if self.metrics_source_blob != MUSPY_METRICS_SOURCE_BLOB {
            issues.push(MuspyPitchReferenceIssueV1::WrongMetricsSourceBlob {
                found: self.metrics_source_blob.clone(),
            });
        }
        for (field, value) in [
            ("python_implementation", self.python_implementation.as_str()),
            ("python_version", self.python_version.as_str()),
            ("numpy_version", self.numpy_version.as_str()),
        ] {
            if value.trim().is_empty() {
                issues.push(MuspyPitchReferenceIssueV1::EmptyRuntimeField {
                    field: field.into(),
                });
            }
        }
        if self.fixtures.is_empty() {
            issues.push(MuspyPitchReferenceIssueV1::EmptyFixtureSet);
            return issues;
        }

        let ids: Vec<_> = self
            .fixtures
            .iter()
            .map(|fixture| fixture.fixture_id.as_str())
            .collect();
        if !ids.windows(2).all(|pair| pair[0] < pair[1]) {
            issues.push(MuspyPitchReferenceIssueV1::FixturesNotCanonical);
        }
        let mut seen = BTreeSet::new();
        for fixture in &self.fixtures {
            if !seen.insert(fixture.fixture_id.clone()) {
                issues.push(MuspyPitchReferenceIssueV1::DuplicateFixtureId {
                    fixture_id: fixture.fixture_id.clone(),
                });
            }
            if fixture.midis.is_empty() {
                issues.push(MuspyPitchReferenceIssueV1::EmptyFixture {
                    fixture_id: fixture.fixture_id.clone(),
                });
            }
            for pitch in fixture.midis.iter().copied().filter(|pitch| *pitch > 127) {
                issues.push(MuspyPitchReferenceIssueV1::PitchOutsideMidiRange {
                    fixture_id: fixture.fixture_id.clone(),
                    pitch,
                });
            }
            for (metric_id, value) in [
                ("pitch_entropy", fixture.external.pitch_entropy),
                ("pitch_class_entropy", fixture.external.pitch_class_entropy),
                ("scale_consistency", fixture.external.scale_consistency),
                (
                    "c_major_pitch_in_scale_rate",
                    fixture.external.c_major_pitch_in_scale_rate,
                ),
            ] {
                if !value.is_finite() {
                    issues.push(MuspyPitchReferenceIssueV1::NonFiniteExternalValue {
                        fixture_id: fixture.fixture_id.clone(),
                        metric_id: metric_id.into(),
                    });
                }
            }
        }
        issues
    }
}

impl MuspyPitchParityReportV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

pub fn verify_muspy_pitch_reference(
    bundle: &MuspyPitchReferenceBundleV1,
    float_tolerance: f64,
) -> MuspyPitchParityReportV1 {
    let issues = bundle.validate();
    if !float_tolerance.is_finite() || float_tolerance < 0.0 {
        return report(
            bundle,
            0.0,
            issues,
            Vec::new(),
            MuspyPitchParityDispositionV1::InvalidTolerance,
        );
    }
    if !issues.is_empty() {
        return report(
            bundle,
            float_tolerance,
            issues,
            Vec::new(),
            MuspyPitchParityDispositionV1::InvalidReference,
        );
    }

    let c_major = BenchmarkScaleV1 {
        root: 0,
        mode: BenchmarkScaleModeV1::Major,
    };
    let mut mismatches = Vec::new();
    for fixture in &bundle.fixtures {
        let score = score_from_midis(&fixture.midis);
        let native = match measure_symbolic_pitch_metrics(&score) {
            Ok(native) => native,
            Err(_) => {
                return report(
                    bundle,
                    float_tolerance,
                    Vec::new(),
                    mismatches,
                    MuspyPitchParityDispositionV1::NativeMeasurementFailure,
                );
            }
        };
        let c_major_rate = match pitch_in_scale_rate(&score, c_major) {
            Ok(rate) => rate,
            Err(_) => {
                return report(
                    bundle,
                    float_tolerance,
                    Vec::new(),
                    mismatches,
                    MuspyPitchParityDispositionV1::NativeMeasurementFailure,
                );
            }
        };

        compare_exact(
            &mut mismatches,
            fixture,
            "n_pitches_used",
            native.n_pitches_used as f64,
            f64::from(fixture.external.n_pitches_used),
        );
        compare_exact(
            &mut mismatches,
            fixture,
            "n_pitch_classes_used",
            native.n_pitch_classes_used as f64,
            f64::from(fixture.external.n_pitch_classes_used),
        );
        compare_exact(
            &mut mismatches,
            fixture,
            "pitch_range",
            f64::from(native.pitch_range_semitones),
            f64::from(fixture.external.pitch_range),
        );
        compare_float(
            &mut mismatches,
            fixture,
            "pitch_entropy",
            native.pitch_entropy_bits,
            fixture.external.pitch_entropy,
            float_tolerance,
        );
        compare_float(
            &mut mismatches,
            fixture,
            "pitch_class_entropy",
            native.pitch_class_entropy_bits,
            fixture.external.pitch_class_entropy,
            float_tolerance,
        );
        compare_float(
            &mut mismatches,
            fixture,
            "scale_consistency",
            native.scale_consistency,
            fixture.external.scale_consistency,
            float_tolerance,
        );
        compare_float(
            &mut mismatches,
            fixture,
            "c_major_pitch_in_scale_rate",
            c_major_rate,
            fixture.external.c_major_pitch_in_scale_rate,
            float_tolerance,
        );
    }

    let disposition = if mismatches.is_empty() {
        MuspyPitchParityDispositionV1::Parity
    } else {
        MuspyPitchParityDispositionV1::Mismatch
    };
    report(
        bundle,
        float_tolerance,
        Vec::new(),
        mismatches,
        disposition,
    )
}

fn report(
    bundle: &MuspyPitchReferenceBundleV1,
    float_tolerance: f64,
    issues: Vec<MuspyPitchReferenceIssueV1>,
    mismatches: Vec<MuspyMetricMismatchV1>,
    disposition: MuspyPitchParityDispositionV1,
) -> MuspyPitchParityReportV1 {
    MuspyPitchParityReportV1 {
        parity_version: MUSPY_PITCH_PARITY_VERSION.into(),
        source_revision: bundle.source_revision.clone(),
        metrics_source_blob: bundle.metrics_source_blob.clone(),
        float_tolerance,
        fixture_count: bundle.fixtures.len(),
        issues,
        mismatches,
        disposition,
    }
}

fn compare_exact(
    mismatches: &mut Vec<MuspyMetricMismatchV1>,
    fixture: &MuspyPitchReferenceFixtureV1,
    metric_id: &str,
    rust_value: f64,
    external_value: f64,
) {
    if rust_value != external_value {
        mismatches.push(MuspyMetricMismatchV1 {
            fixture_id: fixture.fixture_id.clone(),
            metric_id: metric_id.into(),
            rust_value,
            external_value,
            absolute_error: (rust_value - external_value).abs(),
        });
    }
}

fn compare_float(
    mismatches: &mut Vec<MuspyMetricMismatchV1>,
    fixture: &MuspyPitchReferenceFixtureV1,
    metric_id: &str,
    rust_value: f64,
    external_value: f64,
    tolerance: f64,
) {
    let absolute_error = (rust_value - external_value).abs();
    if absolute_error > tolerance {
        mismatches.push(MuspyMetricMismatchV1 {
            fixture_id: fixture.fixture_id.clone(),
            metric_id: metric_id.into(),
            rust_value,
            external_value,
            absolute_error,
        });
    }
}

fn score_from_midis(midis: &[u8]) -> Score {
    let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
    for (index, midi) in midis.iter().copied().enumerate() {
        score.push(ScoreNote {
            part: PartId(1),
            pitch: Pitch::from_midi(midi),
            onset: Duration::new(index as i64, 1),
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        });
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn fixture(
        fixture_id: &str,
        midis: &[u8],
        n_pitches_used: u16,
        n_pitch_classes_used: u16,
        pitch_range: u16,
        pitch_entropy: f64,
        pitch_class_entropy: f64,
        scale_consistency: f64,
        c_major_pitch_in_scale_rate: f64,
    ) -> MuspyPitchReferenceFixtureV1 {
        MuspyPitchReferenceFixtureV1 {
            fixture_id: fixture_id.into(),
            midis: midis.to_vec(),
            external: MuspyPitchReferenceValuesV1 {
                n_pitches_used,
                n_pitch_classes_used,
                pitch_range,
                pitch_entropy,
                pitch_class_entropy,
                scale_consistency,
                c_major_pitch_in_scale_rate,
            },
        }
    }

    fn closed_form_bundle() -> MuspyPitchReferenceBundleV1 {
        MuspyPitchReferenceBundleV1 {
            bundle_version: MUSPY_PITCH_PARITY_VERSION.into(),
            source_revision: MUSPY_REFERENCE_REVISION.into(),
            metrics_source_blob: MUSPY_METRICS_SOURCE_BLOB.into(),
            execution_mode: MuspyReferenceExecutionModeV1::ExactMetricsFileWithTypeStub,
            python_implementation: "CPython".into(),
            python_version: "qualification-runtime".into(),
            numpy_version: "qualification-runtime".into(),
            fixtures: vec![
                fixture(
                    "c-major-octave",
                    &[60, 64, 67, 72],
                    4,
                    3,
                    12,
                    2.0,
                    1.5,
                    1.0,
                    1.0,
                ),
                fixture(
                    "chromatic-12",
                    &(60..72).collect::<Vec<_>>(),
                    12,
                    12,
                    11,
                    12f64.log2(),
                    12f64.log2(),
                    7.0 / 12.0,
                    7.0 / 12.0,
                ),
                fixture(
                    "extreme-midi-range",
                    &[0, 127],
                    2,
                    2,
                    127,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ),
                fixture(
                    "octave-duplicate",
                    &[60, 72],
                    2,
                    1,
                    12,
                    1.0,
                    0.0,
                    1.0,
                    1.0,
                ),
                fixture(
                    "singleton-c4",
                    &[60],
                    1,
                    1,
                    0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                ),
                fixture(
                    "weighted-c-major-outlier",
                    &[60, 60, 61, 64],
                    3,
                    3,
                    4,
                    1.5,
                    1.5,
                    0.75,
                    0.75,
                ),
            ],
        }
    }

    #[test]
    fn closed_form_reference_bundle_matches_native_metrics() {
        let report = verify_muspy_pitch_reference(
            &closed_form_bundle(),
            DEFAULT_MUSPY_FLOAT_TOLERANCE,
        );
        assert_eq!(report.disposition, MuspyPitchParityDispositionV1::Parity);
        assert!(report.issues.is_empty());
        assert!(report.mismatches.is_empty());
    }

    #[test]
    fn source_identity_is_load_bearing() {
        let mut bundle = closed_form_bundle();
        bundle.metrics_source_blob = "0".repeat(40);
        let report = verify_muspy_pitch_reference(
            &bundle,
            DEFAULT_MUSPY_FLOAT_TOLERANCE,
        );
        assert_eq!(
            report.disposition,
            MuspyPitchParityDispositionV1::InvalidReference
        );
        assert!(matches!(
            report.issues.as_slice(),
            [MuspyPitchReferenceIssueV1::WrongMetricsSourceBlob { .. }]
        ));
    }

    #[test]
    fn one_external_delta_produces_a_named_mismatch() {
        let mut bundle = closed_form_bundle();
        bundle.fixtures[0].external.pitch_entropy += 0.01;
        let report = verify_muspy_pitch_reference(
            &bundle,
            DEFAULT_MUSPY_FLOAT_TOLERANCE,
        );
        assert_eq!(report.disposition, MuspyPitchParityDispositionV1::Mismatch);
        assert_eq!(report.mismatches.len(), 1);
        assert_eq!(report.mismatches[0].fixture_id, "c-major-octave");
        assert_eq!(report.mismatches[0].metric_id, "pitch_entropy");
    }

    #[test]
    fn external_reference_bundle_is_executed_when_qualification_requests_it() {
        let Some(path) = std::env::var_os("MEL_BENCH_002B_REFERENCE") else {
            return;
        };
        let bytes = fs::read(path).expect("read MEL_BENCH_002B_REFERENCE");
        let bundle: MuspyPitchReferenceBundleV1 =
            serde_json::from_slice(&bytes).expect("parse MusPy reference bundle");
        let report = verify_muspy_pitch_reference(
            &bundle,
            DEFAULT_MUSPY_FLOAT_TOLERANCE,
        );
        assert_eq!(
            report.disposition,
            MuspyPitchParityDispositionV1::Parity,
            "external parity report: {report:#?}"
        );
        assert!(report.issues.is_empty());
        assert!(report.mismatches.is_empty());
    }
}

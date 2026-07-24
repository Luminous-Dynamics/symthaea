// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sealed artifact production for blinded cognition studies.
//!
//! This module does not generate or rank musical alternatives. It turns files
//! produced by the frozen study pipeline into a public, arm-blind evidence
//! bundle. Every file is hashed from disk, every WAV is audited, and the audio
//! and recipe commitments must agree with the already-frozen blinded schedule.

use crate::blinded_study::BlindedSchedule;
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::experiment_manifest::FrozenStudyManifest;
use crate::methodology_plan::FrozenMethodologyPlan;
use hound::{SampleFormat, WavReader};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path};

pub const STUDY_ARTIFACT_BUNDLE_VERSION: &str = "symthaea-muse-study-artifacts-v1";
pub const STUDY_ARTIFACT_PLAN_VERSION: &str = "symthaea-muse-study-artifact-plan-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArtifactProductionPlan {
    pub plan_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub schedule_sha256: String,
    pub renderer_binary_sha256: String,
    pub render_environment_sha256: String,
    pub soundfont_sha256: String,
    pub required_sample_rate_hz: u32,
    pub required_channels: u16,
    /// Minimum non-silent absolute peak required for a usable artifact.
    pub minimum_absolute_peak: f64,
    /// Absolute sample peak permitted after rendering/normalization.
    pub maximum_absolute_peak: f64,
    pub minimum_duration_ms: u64,
    pub maximum_duration_ms: u64,
    /// Maximum duration spread among the four arms of one frozen fixture.
    pub maximum_within_fixture_duration_delta_ms: u64,
    /// Command template committed before production. It is evidence, not a
    /// shell command executed by this module.
    pub renderer_command_template: Vec<String>,
    pub entries: Vec<ArtifactSourceFiles>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactSourceFiles {
    pub presentation_id: String,
    pub audio_relative_path: String,
    pub recipe_relative_path: String,
    pub score_relative_path: String,
    pub validation_report_relative_path: String,
    pub midi_relative_path: Option<String>,
    pub renderer_log_relative_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WavAudit {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub sample_format: String,
    pub frame_count: u64,
    pub duration_ms: u64,
    pub absolute_peak: f64,
    pub clipped_sample_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFileEvidence {
    pub relative_path: String,
    pub byte_count: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyArtifactRecord {
    pub presentation_id: String,
    pub audio: ArtifactFileEvidence,
    pub recipe: ArtifactFileEvidence,
    pub score: ArtifactFileEvidence,
    pub validation_report: ArtifactFileEvidence,
    pub midi: Option<ArtifactFileEvidence>,
    pub renderer_log: ArtifactFileEvidence,
    pub wav_audit: WavAudit,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StudyArtifactBundle {
    pub bundle_version: String,
    pub manifest_sha256: String,
    pub methodology_sha256: String,
    pub schedule_sha256: String,
    pub production_plan_sha256: String,
    pub renderer_binary_sha256: String,
    pub render_environment_sha256: String,
    pub soundfont_sha256: String,
    pub records: Vec<StudyArtifactRecord>,
    pub bundle_sha256: String,
}

#[derive(Serialize)]
struct ArtifactBundleCommitment<'a> {
    bundle_version: &'a str,
    manifest_sha256: &'a str,
    methodology_sha256: &'a str,
    schedule_sha256: &'a str,
    production_plan_sha256: &'a str,
    renderer_binary_sha256: &'a str,
    render_environment_sha256: &'a str,
    soundfont_sha256: &'a str,
    records: &'a [StudyArtifactRecord],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StudyArtifactIssue {
    WrongPlanVersion {
        found: String,
    },
    WrongBundleVersion {
        found: String,
    },
    InvalidDigest {
        field: String,
    },
    DigestMismatch {
        field: String,
    },
    InvalidAudioContract {
        field: String,
    },
    EmptyRendererCommand,
    DuplicatePresentation {
        presentation_id: String,
    },
    UnknownPresentation {
        presentation_id: String,
    },
    MissingPresentation {
        presentation_id: String,
    },
    UnsafeRelativePath {
        presentation_id: String,
        field: String,
    },
    MissingFile {
        presentation_id: String,
        field: String,
    },
    FileReadFailed {
        presentation_id: String,
        field: String,
    },
    WavReadFailed {
        presentation_id: String,
    },
    WavContractMismatch {
        presentation_id: String,
        field: String,
    },
    FixtureDurationImbalance {
        fixture_id: String,
        delta_ms: u64,
        allowed_ms: u64,
    },
    ScheduleDigestMismatch {
        presentation_id: String,
        field: String,
    },
    SerializationFailed {
        field: String,
    },
}

pub fn artifact_bundle_commitment(
    bundle: &StudyArtifactBundle,
) -> Result<String, serde_json::Error> {
    canonical_json_sha256(&ArtifactBundleCommitment {
        bundle_version: &bundle.bundle_version,
        manifest_sha256: &bundle.manifest_sha256,
        methodology_sha256: &bundle.methodology_sha256,
        schedule_sha256: &bundle.schedule_sha256,
        production_plan_sha256: &bundle.production_plan_sha256,
        renderer_binary_sha256: &bundle.renderer_binary_sha256,
        render_environment_sha256: &bundle.render_environment_sha256,
        soundfont_sha256: &bundle.soundfont_sha256,
        records: &bundle.records,
    })
}

pub fn seal_study_artifacts(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    schedule: &BlindedSchedule,
    plan: &ArtifactProductionPlan,
    artifact_root: &Path,
) -> Result<StudyArtifactBundle, Vec<StudyArtifactIssue>> {
    let mut issues = validate_plan_authorities(manifest, methodology, schedule, plan);
    let schedule_presentations: BTreeMap<_, _> = schedule
        .presentations
        .iter()
        .map(|presentation| (presentation.presentation_id.as_str(), presentation))
        .collect();
    let mut seen = BTreeSet::new();
    let mut records = Vec::with_capacity(plan.entries.len());

    for entry in &plan.entries {
        if !seen.insert(entry.presentation_id.as_str()) {
            issues.push(StudyArtifactIssue::DuplicatePresentation {
                presentation_id: entry.presentation_id.clone(),
            });
            continue;
        }
        let Some(presentation) = schedule_presentations.get(entry.presentation_id.as_str()) else {
            issues.push(StudyArtifactIssue::UnknownPresentation {
                presentation_id: entry.presentation_id.clone(),
            });
            continue;
        };

        let Some(audio) = read_evidence_file(
            artifact_root,
            &entry.presentation_id,
            "audio_relative_path",
            &entry.audio_relative_path,
            &mut issues,
        ) else {
            continue;
        };
        let Some(recipe) = read_evidence_file(
            artifact_root,
            &entry.presentation_id,
            "recipe_relative_path",
            &entry.recipe_relative_path,
            &mut issues,
        ) else {
            continue;
        };
        let Some(score) = read_evidence_file(
            artifact_root,
            &entry.presentation_id,
            "score_relative_path",
            &entry.score_relative_path,
            &mut issues,
        ) else {
            continue;
        };
        let Some(validation_report) = read_evidence_file(
            artifact_root,
            &entry.presentation_id,
            "validation_report_relative_path",
            &entry.validation_report_relative_path,
            &mut issues,
        ) else {
            continue;
        };
        let Some(renderer_log) = read_evidence_file(
            artifact_root,
            &entry.presentation_id,
            "renderer_log_relative_path",
            &entry.renderer_log_relative_path,
            &mut issues,
        ) else {
            continue;
        };
        let midi = match &entry.midi_relative_path {
            Some(path) => read_evidence_file(
                artifact_root,
                &entry.presentation_id,
                "midi_relative_path",
                path,
                &mut issues,
            ),
            None => None,
        };

        if audio.sha256 != presentation.audio_sha256 {
            issues.push(StudyArtifactIssue::ScheduleDigestMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "audio_sha256".into(),
            });
        }
        if recipe.sha256 != presentation.recipe_sha256 {
            issues.push(StudyArtifactIssue::ScheduleDigestMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "recipe_sha256".into(),
            });
        }

        let audio_path = artifact_root.join(&entry.audio_relative_path);
        let Some(wav_audit) = audit_wav(&audio_path, &entry.presentation_id, &mut issues) else {
            continue;
        };
        if wav_audit.sample_rate_hz != plan.required_sample_rate_hz {
            issues.push(StudyArtifactIssue::WavContractMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "sample_rate_hz".into(),
            });
        }
        if wav_audit.channels != plan.required_channels {
            issues.push(StudyArtifactIssue::WavContractMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "channels".into(),
            });
        }
        if wav_audit.duration_ms < plan.minimum_duration_ms
            || wav_audit.duration_ms > plan.maximum_duration_ms
        {
            issues.push(StudyArtifactIssue::WavContractMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "duration_ms".into(),
            });
        }
        if wav_audit.absolute_peak < plan.minimum_absolute_peak
            || wav_audit.absolute_peak > plan.maximum_absolute_peak
            || wav_audit.clipped_sample_count > 0
        {
            issues.push(StudyArtifactIssue::WavContractMismatch {
                presentation_id: entry.presentation_id.clone(),
                field: "peak_or_clipping".into(),
            });
        }

        records.push(StudyArtifactRecord {
            presentation_id: entry.presentation_id.clone(),
            audio,
            recipe,
            score,
            validation_report,
            midi,
            renderer_log,
            wav_audit,
        });
    }

    for presentation in &schedule.presentations {
        if !seen.contains(presentation.presentation_id.as_str()) {
            issues.push(StudyArtifactIssue::MissingPresentation {
                presentation_id: presentation.presentation_id.clone(),
            });
        }
    }
    let duration_by_presentation: BTreeMap<_, _> = records
        .iter()
        .map(|record| {
            (
                record.presentation_id.as_str(),
                record.wav_audit.duration_ms,
            )
        })
        .collect();
    let mut fixture_durations: BTreeMap<_, Vec<u64>> = BTreeMap::new();
    for presentation in &schedule.presentations {
        if let Some(duration_ms) =
            duration_by_presentation.get(presentation.presentation_id.as_str())
        {
            fixture_durations
                .entry(presentation.key.clone())
                .or_default()
                .push(*duration_ms);
        }
    }
    for (key, durations) in fixture_durations {
        if let (Some(minimum), Some(maximum)) = (durations.iter().min(), durations.iter().max()) {
            let delta = maximum.saturating_sub(*minimum);
            if delta > plan.maximum_within_fixture_duration_delta_ms {
                issues.push(StudyArtifactIssue::FixtureDurationImbalance {
                    fixture_id: key.fixture_id,
                    delta_ms: delta,
                    allowed_ms: plan.maximum_within_fixture_duration_delta_ms,
                });
            }
        }
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    records.sort_by(|left, right| left.presentation_id.cmp(&right.presentation_id));
    let mut bundle = StudyArtifactBundle {
        bundle_version: STUDY_ARTIFACT_BUNDLE_VERSION.into(),
        manifest_sha256: canonical_json_sha256(manifest).map_err(|_| {
            vec![StudyArtifactIssue::SerializationFailed {
                field: "manifest".into(),
            }]
        })?,
        methodology_sha256: canonical_json_sha256(methodology).map_err(|_| {
            vec![StudyArtifactIssue::SerializationFailed {
                field: "methodology".into(),
            }]
        })?,
        schedule_sha256: canonical_json_sha256(schedule).map_err(|_| {
            vec![StudyArtifactIssue::SerializationFailed {
                field: "schedule".into(),
            }]
        })?,
        production_plan_sha256: canonical_json_sha256(plan).map_err(|_| {
            vec![StudyArtifactIssue::SerializationFailed {
                field: "production_plan".into(),
            }]
        })?,
        renderer_binary_sha256: plan.renderer_binary_sha256.clone(),
        render_environment_sha256: plan.render_environment_sha256.clone(),
        soundfont_sha256: plan.soundfont_sha256.clone(),
        records,
        bundle_sha256: String::new(),
    };
    bundle.bundle_sha256 = artifact_bundle_commitment(&bundle).map_err(|_| {
        vec![StudyArtifactIssue::SerializationFailed {
            field: "bundle".into(),
        }]
    })?;
    Ok(bundle)
}

pub fn validate_study_artifact_bundle(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    schedule: &BlindedSchedule,
    plan: &ArtifactProductionPlan,
    bundle: &StudyArtifactBundle,
    artifact_root: &Path,
) -> Vec<StudyArtifactIssue> {
    if bundle.bundle_version != STUDY_ARTIFACT_BUNDLE_VERSION {
        return vec![StudyArtifactIssue::WrongBundleVersion {
            found: bundle.bundle_version.clone(),
        }];
    }
    match seal_study_artifacts(manifest, methodology, schedule, plan, artifact_root) {
        Ok(expected) if expected == *bundle => Vec::new(),
        Ok(expected) => {
            let mut issues = Vec::new();
            for (field, left, right) in [
                (
                    "manifest_sha256",
                    &bundle.manifest_sha256,
                    &expected.manifest_sha256,
                ),
                (
                    "methodology_sha256",
                    &bundle.methodology_sha256,
                    &expected.methodology_sha256,
                ),
                (
                    "schedule_sha256",
                    &bundle.schedule_sha256,
                    &expected.schedule_sha256,
                ),
                (
                    "production_plan_sha256",
                    &bundle.production_plan_sha256,
                    &expected.production_plan_sha256,
                ),
                (
                    "bundle_sha256",
                    &bundle.bundle_sha256,
                    &expected.bundle_sha256,
                ),
            ] {
                if left != right {
                    issues.push(StudyArtifactIssue::DigestMismatch {
                        field: field.into(),
                    });
                }
            }
            if bundle.records != expected.records {
                issues.push(StudyArtifactIssue::DigestMismatch {
                    field: "records".into(),
                });
            }
            issues
        }
        Err(issues) => issues,
    }
}

fn validate_plan_authorities(
    manifest: &FrozenStudyManifest,
    methodology: &FrozenMethodologyPlan,
    schedule: &BlindedSchedule,
    plan: &ArtifactProductionPlan,
) -> Vec<StudyArtifactIssue> {
    let mut issues = Vec::new();
    if plan.plan_version != STUDY_ARTIFACT_PLAN_VERSION {
        issues.push(StudyArtifactIssue::WrongPlanVersion {
            found: plan.plan_version.clone(),
        });
    }
    for (field, expected, actual) in [
        (
            "manifest_sha256",
            canonical_json_sha256(manifest),
            &plan.manifest_sha256,
        ),
        (
            "methodology_sha256",
            canonical_json_sha256(methodology),
            &plan.methodology_sha256,
        ),
        (
            "schedule_sha256",
            canonical_json_sha256(schedule),
            &plan.schedule_sha256,
        ),
    ] {
        match expected {
            Ok(value) if value == *actual => {}
            Ok(_) => issues.push(StudyArtifactIssue::DigestMismatch {
                field: field.into(),
            }),
            Err(_) => issues.push(StudyArtifactIssue::SerializationFailed {
                field: field.into(),
            }),
        }
    }
    for (field, value) in [
        ("renderer_binary_sha256", &plan.renderer_binary_sha256),
        ("render_environment_sha256", &plan.render_environment_sha256),
        ("soundfont_sha256", &plan.soundfont_sha256),
    ] {
        if !is_sha256(value) {
            issues.push(StudyArtifactIssue::InvalidDigest {
                field: field.into(),
            });
        }
    }
    if plan.required_sample_rate_hz == 0 {
        issues.push(StudyArtifactIssue::InvalidAudioContract {
            field: "required_sample_rate_hz".into(),
        });
    }
    if plan.required_channels == 0 || plan.required_channels > 8 {
        issues.push(StudyArtifactIssue::InvalidAudioContract {
            field: "required_channels".into(),
        });
    }
    if !plan.minimum_absolute_peak.is_finite()
        || !plan.maximum_absolute_peak.is_finite()
        || !(0.0..=1.0).contains(&plan.minimum_absolute_peak)
        || !(0.0..=1.0).contains(&plan.maximum_absolute_peak)
        || plan.minimum_absolute_peak == 0.0
        || plan.minimum_absolute_peak >= plan.maximum_absolute_peak
    {
        issues.push(StudyArtifactIssue::InvalidAudioContract {
            field: "absolute_peak_range".into(),
        });
    }
    if plan.minimum_duration_ms == 0
        || plan.maximum_duration_ms < plan.minimum_duration_ms
        || plan.maximum_within_fixture_duration_delta_ms
            > plan
                .maximum_duration_ms
                .saturating_sub(plan.minimum_duration_ms)
    {
        issues.push(StudyArtifactIssue::InvalidAudioContract {
            field: "duration_contract".into(),
        });
    }
    for fixture in &manifest.fixtures {
        if fixture.renderer_sha256 != plan.renderer_binary_sha256 {
            issues.push(StudyArtifactIssue::DigestMismatch {
                field: format!("fixture.{}.renderer_sha256", fixture.key.fixture_id),
            });
        }
        if fixture.soundfont_sha256 != plan.soundfont_sha256 {
            issues.push(StudyArtifactIssue::DigestMismatch {
                field: format!("fixture.{}.soundfont_sha256", fixture.key.fixture_id),
            });
        }
    }
    if plan.renderer_command_template.is_empty()
        || plan
            .renderer_command_template
            .iter()
            .any(|part| part.trim().is_empty())
    {
        issues.push(StudyArtifactIssue::EmptyRendererCommand);
    }
    issues
}

fn read_evidence_file(
    root: &Path,
    presentation_id: &str,
    field: &str,
    relative_path: &str,
    issues: &mut Vec<StudyArtifactIssue>,
) -> Option<ArtifactFileEvidence> {
    let path = Path::new(relative_path);
    if !safe_relative_path(path) {
        issues.push(StudyArtifactIssue::UnsafeRelativePath {
            presentation_id: presentation_id.into(),
            field: field.into(),
        });
        return None;
    }
    let full_path = root.join(path);
    if !full_path.is_file() {
        issues.push(StudyArtifactIssue::MissingFile {
            presentation_id: presentation_id.into(),
            field: field.into(),
        });
        return None;
    }
    match fs::read(&full_path) {
        Ok(bytes) => Some(ArtifactFileEvidence {
            relative_path: relative_path.into(),
            byte_count: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        }),
        Err(_) => {
            issues.push(StudyArtifactIssue::FileReadFailed {
                presentation_id: presentation_id.into(),
                field: field.into(),
            });
            None
        }
    }
}

fn safe_relative_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && !path.is_absolute()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn audit_wav(
    path: &Path,
    presentation_id: &str,
    issues: &mut Vec<StudyArtifactIssue>,
) -> Option<WavAudit> {
    let mut reader = match WavReader::open(path) {
        Ok(reader) => reader,
        Err(_) => {
            issues.push(StudyArtifactIssue::WavReadFailed {
                presentation_id: presentation_id.into(),
            });
            return None;
        }
    };
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        issues.push(StudyArtifactIssue::WavReadFailed {
            presentation_id: presentation_id.into(),
        });
        return None;
    }
    if spec.bits_per_sample == 0 {
        issues.push(StudyArtifactIssue::WavReadFailed {
            presentation_id: presentation_id.into(),
        });
        return None;
    }
    let mut peak = 0.0f64;
    let mut clipped = 0u64;
    let mut samples = 0u64;
    match spec.sample_format {
        SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let Ok(sample) = sample else {
                    issues.push(StudyArtifactIssue::WavReadFailed {
                        presentation_id: presentation_id.into(),
                    });
                    return None;
                };
                if !sample.is_finite() {
                    issues.push(StudyArtifactIssue::WavReadFailed {
                        presentation_id: presentation_id.into(),
                    });
                    return None;
                }
                let magnitude = f64::from(sample.abs());
                peak = peak.max(magnitude);
                if magnitude >= 1.0 {
                    clipped += 1;
                }
                samples += 1;
            }
        }
        SampleFormat::Int => {
            let maximum = if spec.bits_per_sample >= 32 {
                i32::MAX as f64
            } else {
                ((1u64 << (spec.bits_per_sample - 1)) - 1) as f64
            };
            for sample in reader.samples::<i32>() {
                let Ok(sample) = sample else {
                    issues.push(StudyArtifactIssue::WavReadFailed {
                        presentation_id: presentation_id.into(),
                    });
                    return None;
                };
                let magnitude = (sample as i64).unsigned_abs() as f64;
                peak = peak.max(magnitude / maximum);
                if magnitude >= maximum {
                    clipped += 1;
                }
                samples += 1;
            }
        }
    }
    let frame_count = samples / u64::from(spec.channels);
    Some(WavAudit {
        sample_rate_hz: spec.sample_rate,
        channels: spec.channels,
        bits_per_sample: spec.bits_per_sample,
        sample_format: match spec.sample_format {
            SampleFormat::Float => "float".into(),
            SampleFormat::Int => "integer".into(),
        },
        frame_count,
        duration_ms: frame_count.saturating_mul(1_000) / u64::from(spec.sample_rate),
        absolute_peak: peak,
        clipped_sample_count: clipped,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_parent_and_absolute_paths() {
        assert!(safe_relative_path(Path::new("audio/example.wav")));
        assert!(!safe_relative_path(Path::new("../example.wav")));
        assert!(!safe_relative_path(Path::new("/tmp/example.wav")));
    }

    #[test]
    fn wav_audit_reports_peak_duration_and_clipping() {
        let path = std::env::temp_dir().join(format!(
            "symthaea-muse-v9-wav-audit-{}-{}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        writer.write_sample(0i16).unwrap();
        writer.write_sample(16_384i16).unwrap();
        writer.write_sample(i16::MAX).unwrap();
        writer.finalize().unwrap();
        let mut issues = Vec::new();
        let audit = audit_wav(&path, "presentation", &mut issues).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(issues.is_empty());
        assert_eq!(audit.frame_count, 3);
        assert_eq!(audit.sample_rate_hz, 48_000);
        assert_eq!(audit.clipped_sample_count, 1);
        assert_eq!(audit.absolute_peak, 1.0);
    }

    #[test]
    fn bundle_commitment_excludes_only_its_own_digest() {
        let mut bundle = StudyArtifactBundle {
            bundle_version: STUDY_ARTIFACT_BUNDLE_VERSION.into(),
            manifest_sha256: "a".repeat(64),
            methodology_sha256: "b".repeat(64),
            schedule_sha256: "c".repeat(64),
            production_plan_sha256: "d".repeat(64),
            renderer_binary_sha256: "e".repeat(64),
            render_environment_sha256: "f".repeat(64),
            soundfont_sha256: "1".repeat(64),
            records: Vec::new(),
            bundle_sha256: String::new(),
        };
        let digest = artifact_bundle_commitment(&bundle).unwrap();
        bundle.bundle_sha256 = digest.clone();
        assert_eq!(artifact_bundle_commitment(&bundle).unwrap(), digest);
        bundle.soundfont_sha256 = "2".repeat(64);
        assert_ne!(artifact_bundle_commitment(&bundle).unwrap(), digest);
    }
}

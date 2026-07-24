// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Immutable authored-etude registry for the Teach Muse research program.
//!
//! The registry exposes abstractions and auditions, never source note material
//! to normal composition. Artifact hashes are verified before a lesson appears.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde::Deserialize;
use sha2::{Digest, Sha256};
use symthaea_muse_protocol::{
    CompositionLessonStatus, CompositionLessonSummary, LessonArtifactIntegrity,
    TeachingCorpusSummary,
};

const PROTOCOL: &str = "muse-variety-etudes-v1";
const FIRST_WAVE: [&str; 4] = [
    "etude:the-door-remembers",
    "etude:the-missing-thread",
    "etude:breath-between-stones",
    "etude:held-ground",
];

#[derive(Debug)]
pub struct TeachingCorpus {
    pub summary: TeachingCorpusSummary,
    audio_by_lesson: BTreeMap<String, PathBuf>,
}

#[derive(Deserialize)]
struct Manifest {
    protocol_version: String,
    status: String,
    entries: Vec<ManifestEntry>,
}

#[derive(Deserialize)]
struct ManifestEntry {
    slug: String,
    title: String,
    subtitle: String,
    primary_dimension: String,
    strategy: String,
    duration_seconds: f64,
    note_count: usize,
    section_count: usize,
    score_content_sha256: String,
    artifacts: BTreeMap<String, ArtifactRecord>,
}

#[derive(Deserialize)]
struct ArtifactRecord {
    path: String,
    sha256: String,
}

#[derive(Deserialize)]
struct LessonContract {
    protocol_version: String,
    lesson_id: String,
    strategy: String,
    abstract_rule: String,
    expected_effects: Vec<String>,
    applicable_grammars: Vec<String>,
    prohibited_literal_reuse: Vec<String>,
    status: String,
}

#[derive(Deserialize)]
struct ValidationReport {
    protocol_version: String,
    validation_passed: bool,
    exact_score_collisions: usize,
    symbolic_fingerprint: FingerprintValidation,
    errors: Vec<String>,
}

#[derive(Deserialize)]
struct FingerprintValidation {
    near_clone_pairs: usize,
}

static CORPUS: OnceLock<Result<TeachingCorpus, String>> = OnceLock::new();

pub fn corpus() -> Result<&'static TeachingCorpus, String> {
    CORPUS
        .get_or_init(load_corpus)
        .as_ref()
        .map_err(Clone::clone)
}

pub fn audition_path(lesson_id: &str) -> Result<PathBuf, String> {
    corpus()?
        .audio_by_lesson
        .get(lesson_id)
        .cloned()
        .ok_or_else(|| "unknown composition lesson".to_owned())
}

pub fn has_typed_shadow_mapping(lesson_id: &str) -> bool {
    FIRST_WAVE.contains(&lesson_id)
}

fn load_corpus() -> Result<TeachingCorpus, String> {
    let root = resolve_root().ok_or_else(|| {
        "Muse Variety Etudes v1 was not found; set MUSE_VARIETY_ETUDES_DIR".to_owned()
    })?;
    let root = root
        .canonicalize()
        .map_err(|error| format!("could not resolve teaching root: {error}"))?;
    let manifest: Manifest = read_json(&root.join("manifest.json"))?;
    let validation: ValidationReport = read_json(&root.join("validation_report.json"))?;
    if manifest.protocol_version != PROTOCOL || validation.protocol_version != PROTOCOL {
        return Err("unsupported teaching-corpus protocol".to_owned());
    }
    if !validation.validation_passed || !validation.errors.is_empty() {
        return Err("teaching corpus did not pass its supplied validation report".to_owned());
    }
    if manifest.entries.len() != 16 {
        return Err(format!(
            "expected 16 etudes, found {}",
            manifest.entries.len()
        ));
    }

    let mut ids = BTreeSet::new();
    let mut lessons = Vec::with_capacity(manifest.entries.len());
    let mut audio_by_lesson = BTreeMap::new();
    for entry in manifest.entries {
        let lesson_record = entry
            .artifacts
            .get("lesson.json")
            .ok_or_else(|| format!("{} has no lesson contract", entry.slug))?;
        let lesson_path = verified_artifact(&root, lesson_record)?;
        let lesson: LessonContract = read_json(&lesson_path)?;
        if lesson.protocol_version != PROTOCOL || lesson.status != "authored" {
            return Err(format!("{} has an invalid lifecycle boundary", entry.slug));
        }
        if lesson.strategy != entry.strategy || !ids.insert(lesson.lesson_id.clone()) {
            return Err(format!(
                "{} has inconsistent or duplicate lesson identity",
                entry.slug
            ));
        }

        let score_ok = verify_named(&root, &entry.artifacts, "score.json")?;
        let midi_ok = verify_named(&root, &entry.artifacts, "score.mid")?;
        let audio_record = entry
            .artifacts
            .get("audition.wav")
            .ok_or_else(|| format!("{} has no audition", entry.slug))?;
        let audio_path = verified_artifact(&root, audio_record)?;
        audio_by_lesson.insert(lesson.lesson_id.clone(), audio_path);
        let first_shadow_wave = FIRST_WAVE.contains(&lesson.lesson_id.as_str());
        lessons.push(CompositionLessonSummary {
            lesson_id: lesson.lesson_id,
            title: entry.title,
            subtitle: entry.subtitle,
            primary_dimension: entry.primary_dimension,
            strategy: entry.strategy,
            abstract_rule: lesson.abstract_rule,
            expected_effects: lesson.expected_effects,
            applicable_grammars: lesson.applicable_grammars,
            prohibited_literal_reuse: lesson.prohibited_literal_reuse,
            status: CompositionLessonStatus::Authored,
            first_shadow_wave,
            duration_seconds: entry.duration_seconds,
            note_count: entry.note_count,
            section_count: entry.section_count,
            score_content_sha256: entry.score_content_sha256,
            integrity: LessonArtifactIntegrity {
                score_json: score_ok,
                lesson_json: true,
                midi: midi_ok,
                audition_audio: true,
            },
            typed_shadow_mapping: first_shadow_wave,
        });
    }
    Ok(TeachingCorpus {
        summary: TeachingCorpusSummary {
            protocol_version: manifest.protocol_version,
            corpus_status: manifest.status,
            validation_passed: true,
            exact_score_collisions: validation.exact_score_collisions,
            near_clone_pairs: validation.symbolic_fingerprint.near_clone_pairs,
            lessons,
        },
        audio_by_lesson,
    })
}

fn verify_named(
    root: &Path,
    artifacts: &BTreeMap<String, ArtifactRecord>,
    name: &str,
) -> Result<bool, String> {
    let record = artifacts
        .get(name)
        .ok_or_else(|| format!("missing {name}"))?;
    verified_artifact(root, record).map(|_| true)
}

fn verified_artifact(root: &Path, record: &ArtifactRecord) -> Result<PathBuf, String> {
    let path = root.join(&record.path);
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("missing teaching artifact {}: {error}", record.path))?;
    if !canonical.starts_with(root) {
        return Err(format!(
            "teaching artifact escapes corpus root: {}",
            record.path
        ));
    }
    let bytes = std::fs::read(&canonical)
        .map_err(|error| format!("could not read {}: {error}", record.path))?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != record.sha256 {
        return Err(format!("teaching artifact hash mismatch: {}", record.path));
    }
    Ok(canonical)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("{}: {error}", path.display()))
}

fn resolve_root() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("MUSE_VARIETY_ETUDES_DIR") {
        let path = PathBuf::from(path);
        if path.is_dir() {
            return Some(path);
        }
    }
    let local = PathBuf::from("data/music/teaching/muse_variety_etudes_v1");
    if local.is_dir() {
        return Some(local);
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Downloads/muse_variety_etudes_v1"))
        .filter(|path| path.is_dir())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supplied_corpus_is_hash_verified_and_never_auto_promoted() {
        let corpus = corpus().expect("supplied teaching corpus");
        assert_eq!(corpus.summary.lessons.len(), 16);
        assert!(corpus.summary.validation_passed);
        assert_eq!(corpus.summary.exact_score_collisions, 0);
        assert_eq!(corpus.summary.near_clone_pairs, 0);
        assert_eq!(
            corpus
                .summary
                .lessons
                .iter()
                .filter(|lesson| lesson.first_shadow_wave)
                .count(),
            4
        );
        assert!(corpus.summary.lessons.iter().all(|lesson| {
            lesson.status == CompositionLessonStatus::Authored
                && lesson.prohibited_literal_reuse.len() >= 8
        }));
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent integrity verifier for Muse vocal comparison bundles.
//!
//! This tool verifies the consistency of the blind/private split, source/case
//! receipts, backend key, objective report, human-reference receipts, and
//! exact WAV bytes. It does not verify model licensing, model provenance,
//! acoustic correctness, or human ratings.

#![forbid(unsafe_code)]

use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};

const RUN_SCHEMA: &str = "muse-vocal-run/v1";
const PRIVATE_KEY_VERSION: u64 = 2;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlindClip {
    clip_id: String,
    case_id: usize,
    lyrics: String,
    audio: String,
    audio_blake3: String,
    performance_blake3: Option<String>,
    category: String,
    challenge: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunManifest {
    schema: String,
    source_revision: String,
    voice_id: String,
    diffsinger_worker: Option<String>,
    diffsinger_worker_blake3: Option<String>,
    case_count: usize,
    cases: Vec<CaseReceipt>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CaseReceipt {
    case_id: usize,
    seed: u64,
    lyrics: String,
    challenge: String,
    category: String,
    style: String,
    valence: f32,
    arousal: f32,
    composition_energy: f32,
    bars: usize,
    consciousness_level: f32,
    transpose_semitones: f32,
    melody: Vec<NoteReceipt>,
    score_blake3: String,
    performance_canonical_json: String,
    performance_blake3: String,
}

#[derive(Debug, Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct NoteReceipt {
    frequency: f32,
    start_time: f32,
    duration: f32,
    velocity: f32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PrivateBackendKey {
    format_version: u64,
    salt: String,
    entries: Vec<KeyEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct KeyEntry {
    clip_id: String,
    backend: String,
    case_id: usize,
    performance_blake3: Option<String>,
    audio_blake3: String,
    capability_manifest_blake3: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HumanReferenceReceipt {
    case_id: usize,
    singer_pseudonym: String,
    consent_reference: String,
    license_reference: String,
    score_blake3: String,
    audio_blake3: String,
}

#[derive(Debug)]
struct VerifiedBundle {
    case_count: usize,
    clip_count: usize,
    backend_count: usize,
    human_reference_count: usize,
    source_revision: String,
    audio_files_verified: usize,
}

fn fail(message: impl Into<String>) -> String {
    message.into()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| fail(format!("cannot stat {}: {error}", path.display())))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(fail(format!(
            "receipt must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let bytes =
        fs::read(path).map_err(|error| fail(format!("cannot read {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| fail(format!("invalid JSON in {}: {error}", path.display())))
}

fn require_real_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| fail(format!("{label} directory unavailable: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err(fail(format!(
            "{label} must be a real directory, not a symlink"
        )));
    }
    path.canonicalize()
        .map_err(|error| fail(format!("cannot canonicalize {label}: {error}")))
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn valid_git_sha(value: &str) -> bool {
    value.len() == 40
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn safe_audio_file(blind_root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    let mut components = path.components();
    if !matches!(components.next(), Some(Component::Normal(_)))
        || components.next().is_some()
        || path.file_name().and_then(|name| name.to_str()) != Some(relative)
        || !relative.ends_with(".wav")
    {
        return Err(fail(format!(
            "blind audio path must be a simple .wav filename: {relative:?}"
        )));
    }
    let joined = blind_root.join(path);
    let metadata = fs::symlink_metadata(&joined)
        .map_err(|error| fail(format!("missing blind audio {relative:?}: {error}")))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(fail(format!(
            "blind audio must be a regular non-symlink file: {relative:?}"
        )));
    }
    Ok(joined)
}

fn validate_directory_layout(
    directory: &Path,
    allowed_files: &BTreeSet<String>,
    label: &str,
) -> Result<(), String> {
    let children = fs::read_dir(directory)
        .map_err(|error| fail(format!("cannot enumerate {label} directory: {error}")))?;
    for child in children {
        let child = child.map_err(|error| fail(format!("cannot read {label} entry: {error}")))?;
        let name = child
            .file_name()
            .into_string()
            .map_err(|_| fail(format!("{label} directory contains a non-UTF8 filename")))?;
        let metadata = fs::symlink_metadata(child.path())
            .map_err(|error| fail(format!("cannot stat {label} entry {name:?}: {error}")))?;
        if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
            return Err(fail(format!(
                "{label} directory contains a symlink or non-regular entry {name:?}"
            )));
        }
        if !allowed_files.contains(&name) {
            return Err(fail(format!(
                "unexpected file {name:?} in {label} directory; possible blind/private data leak"
            )));
        }
    }
    for required in allowed_files {
        if !directory.join(required).is_file() {
            return Err(fail(format!(
                "required {label} file {required:?} is missing"
            )));
        }
    }
    Ok(())
}

fn file_blake3(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path)
        .map_err(|error| fail(format!("cannot read audio {}: {error}", path.display())))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn expected_blind_id(salt: &str, backend: &str, case_id: usize) -> String {
    let input = format!("muse-vocal-v2:{salt}:{backend}:{case_id}");
    let digest = blake3::hash(input.as_bytes()).to_hex().to_string();
    format!("vocal-{}", &digest[..12])
}

fn validate_case(case: &CaseReceipt) -> Result<(), String> {
    if case.lyrics.trim().is_empty()
        || case.style.trim().is_empty()
        || case.category.trim().is_empty()
        || case.challenge.trim().is_empty()
        || case.bars == 0
        || case.melody.is_empty()
    {
        return Err(fail(format!(
            "case {} has missing identity, score, or generation fields",
            case.case_id
        )));
    }
    let controls = [
        case.valence,
        case.arousal,
        case.composition_energy,
        case.consciousness_level,
        case.transpose_semitones,
    ];
    if controls.iter().any(|value| !value.is_finite())
        || !(0.0..=1.0).contains(&case.valence)
        || !(0.0..=1.0).contains(&case.arousal)
        || !(0.0..=1.0).contains(&case.composition_energy)
        || !(0.0..=1.0).contains(&case.consciousness_level)
    {
        return Err(fail(format!(
            "case {} has invalid or non-finite composition controls",
            case.case_id
        )));
    }
    let mut previous_end = 0.0f32;
    for (index, note) in case.melody.iter().enumerate() {
        if !note.frequency.is_finite()
            || note.frequency <= 0.0
            || !note.start_time.is_finite()
            || note.start_time < 0.0
            || !note.duration.is_finite()
            || note.duration <= 0.0
            || !note.velocity.is_finite()
            || !(0.0..=1.0).contains(&note.velocity)
            || note.start_time < previous_end
        {
            return Err(fail(format!(
                "case {} note {index} has invalid pitch/timing/velocity or overlaps",
                case.case_id
            )));
        }
        let end = note.start_time + note.duration;
        if !end.is_finite() || end <= note.start_time {
            return Err(fail(format!(
                "case {} note {index} has an invalid end time",
                case.case_id
            )));
        }
        previous_end = end;
    }

    let score_bytes = serde_json::to_vec(&case.melody).map_err(|error| {
        fail(format!(
            "cannot encode case {} score: {error}",
            case.case_id
        ))
    })?;
    let observed_score = blake3::hash(&score_bytes).to_hex().to_string();
    if case.score_blake3 != observed_score {
        return Err(fail(format!(
            "case {} score digest mismatch: recorded {}, observed {}",
            case.case_id, case.score_blake3, observed_score
        )));
    }
    if !is_hex_digest(&case.performance_blake3) {
        return Err(fail(format!(
            "case {} has an invalid performance digest",
            case.case_id
        )));
    }
    let performance_value: Value =
        serde_json::from_str(&case.performance_canonical_json).map_err(|error| {
            fail(format!(
                "case {} canonical performance JSON is invalid: {error}",
                case.case_id
            ))
        })?;
    if !performance_value.is_object() {
        return Err(fail(format!(
            "case {} canonical performance receipt must be a JSON object",
            case.case_id
        )));
    }
    let observed_performance = blake3::hash(case.performance_canonical_json.as_bytes())
        .to_hex()
        .to_string();
    if observed_performance != case.performance_blake3 {
        return Err(fail(format!(
            "case {} performance digest mismatch: recorded {}, observed {}",
            case.case_id, case.performance_blake3, observed_performance
        )));
    }
    Ok(())
}

fn verify(blind_dir: &Path, private_dir: &Path) -> Result<VerifiedBundle, String> {
    let blind_root = require_real_directory(blind_dir, "blind")?;
    let private_root = require_real_directory(private_dir, "private")?;
    if blind_root == private_root
        || blind_root.starts_with(&private_root)
        || private_root.starts_with(&blind_root)
    {
        return Err(fail(
            "blind and private bundle directories must be disjoint",
        ));
    }

    let clips: Vec<BlindClip> = read_json(&blind_root.join("blind_manifest.json"))?;
    let run: RunManifest = read_json(&private_root.join("run_manifest.json"))?;
    let key: PrivateBackendKey = read_json(&private_root.join("private_backend_key.json"))?;
    let objectives: Vec<Value> = read_json(&private_root.join("objective_report.json"))?;
    let references: Vec<HumanReferenceReceipt> =
        read_json(&private_root.join("human_reference_receipts.json"))?;

    if run.schema != RUN_SCHEMA || run.case_count != run.cases.len() || run.case_count == 0 {
        return Err(fail("run manifest schema or case_count is invalid"));
    }
    if !valid_git_sha(&run.source_revision) || run.voice_id.trim().is_empty() {
        return Err(fail(
            "run manifest requires an exact lowercase Git SHA and voice ID",
        ));
    }
    match (&run.diffsinger_worker, &run.diffsinger_worker_blake3) {
        (Some(path), Some(digest)) if !path.trim().is_empty() && is_hex_digest(digest) => {}
        (None, None) => {}
        _ => {
            return Err(fail(
                "DiffSinger worker path and digest must either both be present and valid or both be absent",
            ));
        }
    }
    if key.format_version != PRIVATE_KEY_VERSION || !is_hex_digest(&key.salt) {
        return Err(fail("private backend key format or blind salt is invalid"));
    }

    let mut cases = BTreeMap::new();
    for case in &run.cases {
        validate_case(case)?;
        if cases.insert(case.case_id, case).is_some() {
            return Err(fail(format!("duplicate case receipt {}", case.case_id)));
        }
    }
    let expected_case_ids: BTreeSet<_> = (0..run.case_count).collect();
    if cases.keys().copied().collect::<BTreeSet<_>>() != expected_case_ids {
        return Err(fail(
            "run manifest case IDs must be contiguous from zero to case_count - 1",
        ));
    }

    if clips.len() != key.entries.len() {
        return Err(fail(format!(
            "blind clip count {} differs from private key entry count {}",
            clips.len(),
            key.entries.len()
        )));
    }

    let mut clip_by_id = BTreeMap::new();
    let mut audio_names = BTreeSet::new();
    let mut verified_audio = 0usize;
    for clip in &clips {
        if clip.clip_id.trim().is_empty() || clip.lyrics.trim().is_empty() {
            return Err(fail("blind manifest contains an empty clip ID or lyric"));
        }
        if clip_by_id.insert(clip.clip_id.as_str(), clip).is_some() {
            return Err(fail(format!("duplicate blind clip ID {}", clip.clip_id)));
        }
        if !audio_names.insert(clip.audio.as_str()) {
            return Err(fail(format!(
                "multiple blind clips reference the same audio file {:?}",
                clip.audio
            )));
        }
        if !is_hex_digest(&clip.audio_blake3) {
            return Err(fail(format!(
                "clip {} has an invalid audio digest",
                clip.clip_id
            )));
        }
        let audio_path = safe_audio_file(&blind_root, &clip.audio)?;
        let observed_audio = file_blake3(&audio_path)?;
        if observed_audio != clip.audio_blake3 {
            return Err(fail(format!(
                "audio digest mismatch for {}: recorded {}, observed {}",
                clip.clip_id, clip.audio_blake3, observed_audio
            )));
        }
        verified_audio += 1;

        let case = cases
            .get(&clip.case_id)
            .ok_or_else(|| fail(format!("clip {} references unknown case", clip.clip_id)))?;
        if clip.lyrics != case.lyrics
            || clip.challenge != case.challenge
            || clip.category != case.category
        {
            return Err(fail(format!(
                "clip {} does not match case {} lyric/category/challenge receipt",
                clip.clip_id, clip.case_id
            )));
        }
    }
    let mut blind_allowed = BTreeSet::new();
    blind_allowed.insert("blind_manifest.json".to_string());
    blind_allowed.insert("ratings.template.json".to_string());
    blind_allowed.extend(audio_names.iter().map(|name| name.to_string()));
    validate_directory_layout(&blind_root, &blind_allowed, "blind")?;

    let private_allowed = BTreeSet::from([
        "run_manifest.json".to_string(),
        "private_backend_key.json".to_string(),
        "objective_report.json".to_string(),
        "human_reference_receipts.json".to_string(),
        "release_status.json".to_string(),
    ]);
    validate_directory_layout(&private_root, &private_allowed, "private")?;
    let release_status: Value = read_json(&private_root.join("release_status.json"))?;
    for field in ["objective_pass", "human_complete", "release_pass"] {
        if !release_status.get(field).and_then(Value::as_bool).is_some() {
            return Err(fail(format!(
                "release_status.json is missing boolean field {field:?}"
            )));
        }
    }

    let mut key_by_id = BTreeMap::new();
    let mut backend_names = BTreeSet::new();
    let mut model_count = 0usize;
    let mut human_count = 0usize;
    for entry in &key.entries {
        if entry.backend.trim().is_empty()
            || !is_hex_digest(&entry.audio_blake3)
            || key_by_id.insert(entry.clip_id.as_str(), entry).is_some()
        {
            return Err(fail(format!(
                "private backend key has invalid or duplicate entry {}",
                entry.clip_id
            )));
        }
        let clip = clip_by_id.get(entry.clip_id.as_str()).ok_or_else(|| {
            fail(format!(
                "private key references unknown clip {}",
                entry.clip_id
            ))
        })?;
        let expected_id = expected_blind_id(&key.salt, &entry.backend, entry.case_id);
        if expected_id != entry.clip_id {
            return Err(fail(format!(
                "blind ID does not match private salt/backend/case for {}",
                entry.clip_id
            )));
        }
        if entry.case_id != clip.case_id || entry.audio_blake3 != clip.audio_blake3 {
            return Err(fail(format!(
                "private key receipt does not match blind clip {}",
                entry.clip_id
            )));
        }

        if entry.backend == "human-reference" {
            human_count += 1;
            if entry.performance_blake3.is_some()
                || entry.capability_manifest_blake3.is_some()
                || clip.performance_blake3.is_some()
            {
                return Err(fail(format!(
                    "human reference {} must not claim a generated performance or backend capability manifest",
                    entry.clip_id
                )));
            }
        } else {
            model_count += 1;
            let case = cases
                .get(&entry.case_id)
                .ok_or_else(|| fail(format!("model clip {} has no case", entry.clip_id)))?;
            let performance_digest = entry.performance_blake3.as_deref().ok_or_else(|| {
                fail(format!(
                    "model clip {} has no performance digest",
                    entry.clip_id
                ))
            })?;
            let capability_digest =
                entry.capability_manifest_blake3.as_deref().ok_or_else(|| {
                    fail(format!(
                        "model clip {} has no capability digest",
                        entry.clip_id
                    ))
                })?;
            if !is_hex_digest(performance_digest)
                || !is_hex_digest(capability_digest)
                || clip.performance_blake3.as_deref() != Some(performance_digest)
                || performance_digest != case.performance_blake3
            {
                return Err(fail(format!(
                    "performance/capability receipt mismatch for model clip {}",
                    entry.clip_id
                )));
            }
            backend_names.insert(entry.backend.as_str());
        }
    }

    for clip in &clips {
        if !key_by_id.contains_key(clip.clip_id.as_str()) {
            return Err(fail(format!(
                "blind clip {} has no private backend-key entry",
                clip.clip_id
            )));
        }
    }

    if objectives.len() != model_count {
        return Err(fail(format!(
            "objective report count {} differs from generated model clip count {model_count}",
            objectives.len()
        )));
    }
    let mut objective_by_id = BTreeMap::new();
    for objective in &objectives {
        let clip_id = objective
            .get("clip_id")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("objective entry is missing clip_id"))?;
        if objective_by_id.insert(clip_id, objective).is_some() {
            return Err(fail(format!("duplicate objective entry {clip_id}")));
        }
        let key_entry = key_by_id.get(clip_id).ok_or_else(|| {
            fail(format!(
                "objective report references unknown clip {clip_id}"
            ))
        })?;
        if key_entry.backend == "human-reference" {
            return Err(fail(format!(
                "human reference {clip_id} must not have a generated objective entry"
            )));
        }
        let backend = objective
            .get("backend")
            .and_then(Value::as_str)
            .ok_or_else(|| fail(format!("objective {clip_id} is missing backend")))?;
        let case_id = objective
            .get("case_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail(format!("objective {clip_id} has invalid case_id")))?
            as usize;
        let capability_digest = objective
            .get("capability_manifest_blake3")
            .and_then(Value::as_str)
            .ok_or_else(|| fail(format!("objective {clip_id} has no capability digest")))?;
        let capability_canonical_json = objective
            .get("capability_manifest_canonical_json")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                fail(format!(
                    "objective {clip_id} has no canonical capability JSON"
                ))
            })?;
        let decoded_capability: Value =
            serde_json::from_str(capability_canonical_json).map_err(|error| {
                fail(format!(
                    "objective {clip_id} canonical capability JSON is invalid: {error}"
                ))
            })?;
        let capability_value = objective
            .get("capability_manifest")
            .ok_or_else(|| fail(format!("objective {clip_id} has no capability manifest")))?;
        let observed_capability_digest = blake3::hash(capability_canonical_json.as_bytes())
            .to_hex()
            .to_string();
        if backend != key_entry.backend
            || case_id != key_entry.case_id
            || Some(capability_digest) != key_entry.capability_manifest_blake3.as_deref()
            || capability_digest != observed_capability_digest
            || decoded_capability != *capability_value
            || !capability_value.is_object()
            || !objective.get("metrics").is_some_and(Value::is_object)
        {
            return Err(fail(format!(
                "objective receipt does not match private key for {clip_id}"
            )));
        }
    }

    if human_count != references.len() {
        return Err(fail(format!(
            "human-reference key count {human_count} differs from rights-receipt count {}",
            references.len()
        )));
    }
    let mut reference_cases = BTreeSet::new();
    for receipt in &references {
        if receipt.singer_pseudonym.trim().is_empty()
            || receipt.consent_reference.trim().is_empty()
            || receipt.license_reference.trim().is_empty()
            || !is_hex_digest(&receipt.score_blake3)
            || !is_hex_digest(&receipt.audio_blake3)
            || !reference_cases.insert(receipt.case_id)
        {
            return Err(fail(format!(
                "human reference case {} has incomplete/duplicate rights receipt",
                receipt.case_id
            )));
        }
        let key_entry = key
            .entries
            .iter()
            .find(|entry| entry.backend == "human-reference" && entry.case_id == receipt.case_id)
            .ok_or_else(|| {
                fail(format!(
                    "human reference receipt {} has no corresponding private key entry",
                    receipt.case_id
                ))
            })?;
        let clip = clip_by_id
            .get(key_entry.clip_id.as_str())
            .ok_or_else(|| fail("human reference key is missing its blind clip"))?;
        let case = cases
            .get(&receipt.case_id)
            .ok_or_else(|| fail("human reference is missing its score case"))?;
        if receipt.score_blake3 != case.score_blake3
            || receipt.audio_blake3 != clip.audio_blake3
            || receipt.audio_blake3 != key_entry.audio_blake3
        {
            return Err(fail(format!(
                "human-reference score/audio receipt mismatch for case {}",
                receipt.case_id
            )));
        }
    }

    Ok(VerifiedBundle {
        case_count: cases.len(),
        clip_count: clips.len(),
        backend_count: backend_names.len(),
        human_reference_count: human_count,
        source_revision: run.source_revision,
        audio_files_verified: verified_audio,
    })
}

fn arg(args: &[String], name: &str) -> Result<PathBuf, String> {
    let index = args
        .iter()
        .position(|value| value == name)
        .ok_or_else(|| fail(format!("missing argument {name}")))?;
    args.get(index + 1)
        .map(PathBuf::from)
        .ok_or_else(|| fail(format!("missing value after {name}")))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let result = (|| {
        if args.first().map(String::as_str) != Some("verify") {
            return Err(fail(
                "usage: vocal-run-verifier verify --blind-dir DIR --private-dir DIR",
            ));
        }
        let blind_dir = arg(&args, "--blind-dir")?;
        let private_dir = arg(&args, "--private-dir")?;
        let report = verify(&blind_dir, &private_dir)?;
        println!(
            "verified source={} cases={} clips={} backends={} human_references={} wavs={}",
            report.source_revision,
            report.case_count,
            report.clip_count,
            report.backend_count,
            report.human_reference_count,
            report.audio_files_verified
        );
        Ok(())
    })();

    if let Err(error) = result {
        eprintln!("vocal-run-verifier: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    struct Bundle(PathBuf);

    impl Bundle {
        fn new() -> Self {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let root = env::temp_dir().join(format!(
                "symthaea-vocal-run-verify-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("blind")).unwrap();
            fs::create_dir_all(root.join("private")).unwrap();
            Self(root)
        }

        fn write_json(&self, relative: &str, value: &impl serde::Serialize) {
            let path = self.0.join(relative);
            fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
        }

        fn roots(&self) -> (PathBuf, PathBuf) {
            (self.0.join("blind"), self.0.join("private"))
        }
    }

    impl Drop for Bundle {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn digest(bytes: &[u8]) -> String {
        blake3::hash(bytes).to_hex().to_string()
    }

    fn note() -> NoteReceipt {
        NoteReceipt {
            frequency: 220.0,
            start_time: 0.0,
            duration: 0.5,
            velocity: 0.7,
        }
    }

    fn build_fixture(include_human: bool) -> Bundle {
        let bundle = Bundle::new();
        let (blind, private) = bundle.roots();
        let revision = "a".repeat(40);
        let lyrics = "hello world".to_string();
        let melody = vec![note()];
        let score_blake3 = digest(&serde_json::to_vec(&melody).unwrap());
        let performance_canonical_json = r#"{"synthetic_fixture":true}"#.to_string();
        let performance_blake3 = digest(performance_canonical_json.as_bytes());
        let case = CaseReceipt {
            case_id: 0,
            seed: 9001,
            lyrics: lyrics.clone(),
            challenge: "baseline".into(),
            category: "center_legato".into(),
            style: "Classical".into(),
            valence: 0.5,
            arousal: 0.5,
            composition_energy: 0.55,
            bars: 2,
            consciousness_level: 0.72,
            transpose_semitones: 0.0,
            melody,
            score_blake3: score_blake3.clone(),
            performance_canonical_json,
            performance_blake3: performance_blake3.clone(),
        };
        let run = RunManifest {
            schema: RUN_SCHEMA.into(),
            source_revision: revision,
            voice_id: "af_bella".into(),
            diffsinger_worker: None,
            diffsinger_worker_blake3: None,
            case_count: 1,
            cases: vec![case],
        };
        let salt = "c".repeat(64);
        let backend = "kokoro-test";
        let clip_id = expected_blind_id(&salt, backend, 0);
        let audio_name = format!("{clip_id}.wav");
        let audio = b"synthetic-wav-bytes";
        fs::write(blind.join(&audio_name), audio).unwrap();
        let audio_digest = digest(audio);
        let model_clip = BlindClip {
            clip_id: clip_id.clone(),
            case_id: 0,
            lyrics: lyrics.clone(),
            audio: audio_name.clone(),
            audio_blake3: audio_digest.clone(),
            performance_blake3: Some(performance_blake3.clone()),
            category: "center_legato".into(),
            challenge: "baseline".into(),
        };
        let mut clips = vec![model_clip];
        let capability_manifest = serde_json::json!({"backend_id": backend});
        let capability_manifest_canonical_json =
            serde_json::to_string(&capability_manifest).unwrap();
        let capability_manifest_blake3 = digest(capability_manifest_canonical_json.as_bytes());
        let mut entries = vec![KeyEntry {
            clip_id: clip_id.clone(),
            backend: backend.into(),
            case_id: 0,
            performance_blake3: Some(performance_blake3),
            audio_blake3: audio_digest.clone(),
            capability_manifest_blake3: Some(capability_manifest_blake3.clone()),
        }];
        let objectives = vec![serde_json::json!({
            "clip_id": clip_id,
            "backend": backend,
            "case_id": 0,
            "metrics": {"objective_pass": true},
            "capability_manifest": capability_manifest,
            "capability_manifest_canonical_json": capability_manifest_canonical_json,
            "capability_manifest_blake3": capability_manifest_blake3
        })];

        let mut reference_receipts = Vec::new();
        if include_human {
            let human_backend = "human-reference";
            let human_id = expected_blind_id(&salt, human_backend, 0);
            let human_name = format!("{human_id}.wav");
            let human_audio = b"human-anchor-wav-bytes";
            fs::write(blind.join(&human_name), human_audio).unwrap();
            let human_digest = digest(human_audio);
            clips.push(BlindClip {
                clip_id: human_id.clone(),
                case_id: 0,
                lyrics,
                audio: human_name,
                audio_blake3: human_digest.clone(),
                performance_blake3: None,
                category: "center_legato".into(),
                challenge: "baseline".into(),
            });
            entries.push(KeyEntry {
                clip_id: human_id,
                backend: human_backend.into(),
                case_id: 0,
                performance_blake3: None,
                audio_blake3: human_digest.clone(),
                capability_manifest_blake3: None,
            });
            reference_receipts.push(HumanReferenceReceipt {
                case_id: 0,
                singer_pseudonym: "singer-001".into(),
                consent_reference: "consent:001".into(),
                license_reference: "license:001".into(),
                score_blake3,
                audio_blake3: human_digest,
            });
        }
        bundle.write_json("blind/blind_manifest.json", &clips);
        bundle.write_json("private/run_manifest.json", &run);
        bundle.write_json(
            "private/private_backend_key.json",
            &serde_json::json!({
                "format_version": PRIVATE_KEY_VERSION,
                "salt": salt,
                "entries": entries
            }),
        );
        bundle.write_json("private/objective_report.json", &objectives);
        bundle.write_json("private/human_reference_receipts.json", &reference_receipts);
        bundle.write_json(
            "private/release_status.json",
            &serde_json::json!({
                "objective_pass": true,
                "human_complete": include_human,
                "release_pass": false
            }),
        );
        bundle.write_json("blind/ratings.template.json", &serde_json::json!([]));
        bundle
    }

    #[test]
    fn verifies_complete_model_only_bundle() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let report = verify(&blind, &private).unwrap();
        assert_eq!(report.case_count, 1);
        assert_eq!(report.clip_count, 1);
        assert_eq!(report.backend_count, 1);
        assert_eq!(report.human_reference_count, 0);
        assert_eq!(report.audio_files_verified, 1);
    }

    #[test]
    fn verifies_human_reference_with_consent_license_and_score_receipts() {
        let bundle = build_fixture(true);
        let (blind, private) = bundle.roots();
        let report = verify(&blind, &private).unwrap();
        assert_eq!(report.human_reference_count, 1);
        assert_eq!(report.audio_files_verified, 2);
    }

    #[test]
    fn rejects_backend_leak_in_blind_directory() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        fs::write(blind.join("private_backend_key.json"), b"leak").unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("possible blind/private data leak")
        );
    }

    #[test]
    fn rejects_unexpected_private_directory_entries() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        fs::write(private.join("shared-listener-copy.wav"), b"private").unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("possible blind/private data leak")
        );
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlinked_audio() {
        use std::os::unix::fs::symlink;
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let audio = fs::read_dir(&blind)
            .unwrap()
            .filter_map(Result::ok)
            .find(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "wav")
            })
            .unwrap()
            .path();
        let target = private.join("outside.wav");
        fs::write(&target, b"external").unwrap();
        fs::remove_file(&audio).unwrap();
        symlink(&target, &audio).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("non-symlink")
        );
    }

    #[test]
    fn rejects_changed_audio_bytes() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = fs::read_dir(&blind)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(path, b"tampered").unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("audio digest mismatch")
        );
    }

    #[test]
    fn rejects_audio_path_traversal() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = blind.join("blind_manifest.json");
        let mut clips: Vec<BlindClip> = read_json(&path).unwrap();
        clips[0].audio = "../private/private_backend_key.json".into();
        fs::write(path, serde_json::to_vec(&clips).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("simple .wav filename")
        );
    }

    #[test]
    fn rejects_mismatched_backend_or_blind_id() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("private_backend_key.json");
        let mut key: Value = read_json(&path).unwrap();
        key["entries"][0]["backend"] = Value::String("forged-backend".into());
        fs::write(path, serde_json::to_vec(&key).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("blind ID does not match")
        );
    }

    #[test]
    fn rejects_performance_and_score_digest_changes() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("run_manifest.json");
        let mut run: Value = read_json(&path).unwrap();
        run["cases"][0]["score_blake3"] = Value::String("f".repeat(64));
        fs::write(path, serde_json::to_vec(&run).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("score digest mismatch")
        );
    }

    #[test]
    fn rejects_canonical_performance_bytes_that_do_not_match_receipt_digest() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("run_manifest.json");
        let mut run: Value = read_json(&path).unwrap();
        run["cases"][0]["performance_canonical_json"] =
            Value::String(r#"{"version":2,"lyrics":"changed"}"#.into());
        fs::write(&path, serde_json::to_vec(&run).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("performance digest mismatch")
        );
    }

    #[test]
    fn rejects_missing_objective_entry() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        fs::write(private.join("objective_report.json"), b"[]").unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("objective report count")
        );
    }

    #[test]
    fn rejects_missing_model_capability_receipt() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("private_backend_key.json");
        let mut key: Value = read_json(&path).unwrap();
        key["entries"][0]["capability_manifest_blake3"] = Value::Null;
        fs::write(path, serde_json::to_vec(&key).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("no capability digest")
        );
    }

    #[test]
    fn rejects_duplicate_or_unmatched_clip_key_entries() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("private_backend_key.json");
        let mut key: Value = read_json(&path).unwrap();
        let duplicate = key["entries"][0].clone();
        key["entries"].as_array_mut().unwrap().push(duplicate);
        fs::write(path, serde_json::to_vec(&key).unwrap()).unwrap();
        assert!(verify(&blind, &private).unwrap_err().contains("clip count"));
    }

    #[test]
    fn rejects_invalid_revision_and_non_finite_score_values() {
        let bundle = build_fixture(false);
        let (blind, private) = bundle.roots();
        let path = private.join("run_manifest.json");
        let mut run: Value = read_json(&path).unwrap();
        run["source_revision"] = Value::String("not-a-sha".into());
        fs::write(&path, serde_json::to_vec(&run).unwrap()).unwrap();
        assert!(
            verify(&blind, &private)
                .unwrap_err()
                .contains("exact lowercase Git SHA")
        );
    }
}

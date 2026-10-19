//! Build a randomized, anonymized listener pack from renderer-ab-fair evidence.
//!
//! Usage:
//!   SYMTHAEA_BLIND_PACK_SEED=<64-hex-secret> cargo run -p symthaea-muse --features theory --example blind_render_listening_pack -- <evidence-manifest.json> <audio-root> <output-root>
//!
//! The participant pack never contains treatment labels, original WAV paths, or
//! source WAV hashes. The operator answer key is emitted outside that directory.
//! This tool copies WAV bytes unchanged; it does not loudness-normalize or render audio.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const EXPECTED_SCHEMA: &str = "symthaea-muse-renderer-ab-evidence-v1";
const PACK_SCHEMA: &str = "melothaea-blind-listening-pack-v1";
const STYLES: [&str; 8] = [
    "ClassicalRubato",
    "DanceLocked",
    "ProcessExact",
    "FolkLift",
    "JazzLaidBack",
    "DroneElastic",
    "DramaticTiming",
    "ChoralBlend",
];

const ARMS: [(&str, &str); 4] = [
    ("default_default", "fluidsynth_default_profile"),
    ("dialect_room_only", "fluidsynth_room_only"),
    ("dialect_mastering_only", "fluidsynth_mastering_only"),
    ("dialect_room_mastering", "fluidsynth_dialect_profile"),
];

type Result<T> = std::result::Result<T, String>;

#[derive(Debug, Clone)]
struct SourceArm {
    treatment: &'static str,
    path: PathBuf,
    declared_sha256: String,
    declared_bytes: u64,
    sample_rate_hz: u64,
    verified_sha256: String,
    actual_bytes: u64,
}

#[derive(Debug, Clone)]
struct SourceBlock {
    style: String,
    seed_label: String,
    midi_sha256: String,
    arms: Vec<SourceArm>,
}

#[derive(Debug, Clone)]
struct PackedClip {
    sequence: usize,
    block_code: String,
    clip_code: String,
    target_style: String,
    treatment: String,
    source_path: PathBuf,
    source_sha256: String,
    source_bytes: u64,
    participant_filename: String,
    sample_rate_hz: u64,
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_string<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{context}.{key} must be a non-empty string"))
}

fn required_u64(value: &Value, key: &str, context: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{context}.{key} must be a non-negative integer"))
}

fn derived_digest(seed: &str, domain: &str, identity: &str) -> String {
    sha256(format!("{seed}|{domain}|{identity}").as_bytes())
}

fn derived_code(seed: &str, domain: &str, identity: &str) -> String {
    derived_digest(seed, domain, identity)[..20].to_owned()
}

fn score(seed: &str, domain: &str, identity: &str) -> String {
    derived_digest(seed, domain, identity)
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn write_csv_row(output: &mut String, fields: &[String]) {
    output.push_str(
        &fields
            .iter()
            .map(|field| csv_field(field))
            .collect::<Vec<_>>()
            .join(","),
    );
    output.push('\n');
}

fn resolve_wav_path(audio_root: &Path, declared: &str) -> PathBuf {
    let path = PathBuf::from(declared);
    if path.is_absolute() {
        path
    } else {
        audio_root.join(path)
    }
}

fn validate_wave_bytes(bytes: &[u8], context: &str) -> Result<u32> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{context}: expected a non-empty RIFF/WAVE container"));
    }
    let declared_riff_bytes = u32::from_le_bytes(bytes[4..8].try_into().expect("slice has four bytes")) as usize;
    if declared_riff_bytes.checked_add(8) != Some(bytes.len()) {
        return Err(format!("{context}: RIFF size does not match actual file length"));
    }

    let mut offset = 12usize;
    let mut sample_rate_from_header = None;
    let mut has_audio_data = false;
    while offset < bytes.len() {
        if bytes.len() - offset < 8 {
            return Err(format!("{context}: truncated WAVE chunk header"));
        }
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_bytes = u32::from_le_bytes(
            bytes[offset + 4..offset + 8]
                .try_into()
                .expect("slice has four bytes"),
        ) as usize;
        let payload_start = offset + 8;
        let payload_end = payload_start
            .checked_add(chunk_bytes)
            .ok_or_else(|| format!("{context}: WAVE chunk length overflow"))?;
        if payload_end > bytes.len() {
            return Err(format!("{context}: WAVE chunk exceeds file length"));
        }
        if chunk_id == b"fmt " {
            if chunk_bytes < 16 {
                return Err(format!("{context}: WAVE format chunk is shorter than 16 bytes"));
            }
            if sample_rate_from_header.is_some() {
                return Err(format!("{context}: duplicate WAVE format chunks"));
            }
            let rate_start = payload_start + 4;
            let rate = u32::from_le_bytes(
                bytes[rate_start..rate_start + 4]
                    .try_into()
                    .expect("validated format chunk has at least 16 bytes"),
            );
            if rate == 0 {
                return Err(format!("{context}: WAVE sample rate must be positive"));
            }
            sample_rate_from_header = Some(rate);
        }
        if chunk_id == b"data" {
            if chunk_bytes == 0 {
                return Err(format!("{context}: WAVE audio data chunk is empty"));
            }
            if has_audio_data {
                return Err(format!("{context}: duplicate WAVE audio data chunks"));
            }
            has_audio_data = true;
        }
        offset = payload_end
            .checked_add(chunk_bytes % 2)
            .ok_or_else(|| format!("{context}: WAVE padding length overflow"))?;
        if offset > bytes.len() {
            return Err(format!("{context}: WAVE chunk padding exceeds file length"));
        }
    }
    if offset != bytes.len() || !has_audio_data {
        return Err(format!("{context}: WAVE requires complete fmt and non-empty data chunks"));
    }
    sample_rate_from_header
        .ok_or_else(|| format!("{context}: WAVE format chunk is missing"))
}

fn read_blocks(manifest: &Value, audio_root: &Path) -> Result<(Vec<SourceBlock>, String)> {
    if required_string(manifest, "schema", "manifest")? != EXPECTED_SCHEMA {
        return Err(format!("manifest schema must be {EXPECTED_SCHEMA}"));
    }
    if required_string(manifest, "run_status", "manifest")? != "complete" {
        return Err("renderer manifest is not marked complete".to_owned());
    }

    let source_revision = manifest
        .get("source_revision")
        .and_then(|value| value.get("value"))
        .and_then(Value::as_str)
        .ok_or_else(|| "manifest.source_revision.value is missing".to_owned())?;
    if source_revision.len() != 40 && source_revision.len() != 64 {
        return Err("manifest source revision must be 40 or 64 hex characters".to_owned());
    }
    if !source_revision.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("manifest source revision contains non-hex characters".to_owned());
    }

    let cohort = manifest
        .get("cohort")
        .ok_or_else(|| "manifest.cohort is missing".to_owned())?;
    let expected = required_u64(cohort, "expected", "manifest.cohort")?;
    let completed = required_u64(cohort, "completed", "manifest.cohort")?;
    let record_count = required_u64(cohort, "record_count_including_failures", "manifest.cohort")?;

    let records = manifest
        .get("records")
        .and_then(Value::as_array)
        .ok_or_else(|| "manifest.records must be an array".to_owned())?;
    if expected != records.len() as u64
        || completed != records.len() as u64
        || record_count != records.len() as u64
    {
        return Err(format!(
            "manifest cohort counts disagree: expected={expected}, completed={completed}, records_declared={record_count}, records_actual={}",
            records.len()
        ));
    }
    let expected_dialects = manifest
        .get("contract")
        .and_then(|value| value.get("expected_dialect_count"))
        .and_then(Value::as_u64)
        .ok_or_else(|| "manifest.contract.expected_dialect_count is missing".to_owned())?;
    if expected_dialects != STYLES.len() as u64 {
        return Err(format!("expected exactly {} dialects", STYLES.len()));
    }

    let mut blocks = Vec::with_capacity(records.len());
    let mut seen_blocks = BTreeSet::new();
    let mut seen_styles = BTreeSet::new();

    for (row_index, record) in records.iter().enumerate() {
        let context = format!("records[{row_index}]");
        if required_string(record, "status", &context)? != "complete" {
            return Err(format!("{context} is not complete; refusing to drop a failed cell"));
        }
        let style = required_string(record, "style", &context)?.to_owned();
        if !STYLES.contains(&style.as_str()) {
            return Err(format!("{context} has unexpected style {style:?}"));
        }
        let seed_label = match record.get("seed") {
            Some(Value::String(value)) if !value.trim().is_empty() => value.clone(),
            Some(Value::Number(value)) => value.to_string(),
            _ => return Err(format!("{context}.seed must be a non-empty string or number")),
        };
        let block_identity = format!("{style}|{seed_label}");
        if !seen_blocks.insert(block_identity.clone()) {
            return Err(format!("duplicate style/seed cell {block_identity}"));
        }
        seen_styles.insert(style.clone());

        let midi_sha256 = record
            .get("input_midi")
            .and_then(|value| value.get("sha256"))
            .and_then(Value::as_str)
            .filter(|value| valid_sha256(value))
            .ok_or_else(|| format!("{context}.input_midi.sha256 is missing or malformed"))?
            .to_owned();

        let arms_object = record
            .get("arms")
            .ok_or_else(|| format!("{context}.arms is missing"))?;
        let mut arms = Vec::with_capacity(ARMS.len());
        let mut shared_sample_rate = None;

        for (treatment, manifest_arm_name) in ARMS {
            let arm_context = format!("{context}.arms.{manifest_arm_name}");
            let arm = arms_object
                .get(manifest_arm_name)
                .ok_or_else(|| format!("{arm_context} is missing"))?;
            let declared_path = required_string(arm, "wav_path", &arm_context)?;
            let declared_sha256 = required_string(arm, "wav_sha256", &arm_context)?.to_owned();
            if !valid_sha256(&declared_sha256) {
                return Err(format!("{arm_context}.wav_sha256 is malformed"));
            }
            let declared_bytes = required_u64(arm, "wav_bytes", &arm_context)?;
            let sample_rate_hz = required_u64(arm, "sample_rate_hz", &arm_context)?;
            if sample_rate_hz == 0 {
                return Err(format!("{arm_context}.sample_rate_hz must be positive"));
            }
            if let Some(expected_rate) = shared_sample_rate {
                if sample_rate_hz != expected_rate {
                    return Err(format!(
                        "{context} has mismatched sample rates: {sample_rate_hz} versus {expected_rate}"
                    ));
                }
            } else {
                shared_sample_rate = Some(sample_rate_hz);
            }

            let path = resolve_wav_path(audio_root, declared_path);
            let bytes = fs::read(&path)
                .map_err(|error| format!("{arm_context}: cannot read {}: {error}", path.display()))?;
            let wave_sample_rate_hz = validate_wave_bytes(&bytes, &arm_context)? as u64;
            if wave_sample_rate_hz != sample_rate_hz {
                return Err(format!(
                    "{arm_context}: manifest sample rate {sample_rate_hz} Hz disagrees with WAV header {wave_sample_rate_hz} Hz"
                ));
            }
            let actual_bytes = bytes.len() as u64;
            if actual_bytes != declared_bytes {
                return Err(format!(
                    "{arm_context}: byte count mismatch, declared {declared_bytes}, actual {actual_bytes}"
                ));
            }
            let verified_sha256 = sha256(&bytes);
            if verified_sha256 != declared_sha256 {
                return Err(format!("{arm_context}: SHA-256 does not match renderer receipt"));
            }
            arms.push(SourceArm {
                treatment,
                path,
                declared_sha256,
                declared_bytes,
                sample_rate_hz,
                verified_sha256,
                actual_bytes,
            });
        }
        blocks.push(SourceBlock {
            style,
            seed_label,
            midi_sha256,
            arms,
        });
    }

    if seen_styles.len() != STYLES.len() {
        let missing = STYLES
            .iter()
            .filter(|style| !seen_styles.contains(*style))
            .copied()
            .collect::<Vec<_>>();
        return Err(format!("style cohort incomplete; missing: {}", missing.join(", ")));
    }

    // Sort input cells canonically before any randomized display ordering.
    blocks.sort_by(|left, right| {
        left.style
            .cmp(&right.style)
            .then_with(|| left.seed_label.cmp(&right.seed_label))
    });

    Ok((blocks, source_revision.to_owned()))
}

fn build_pack(manifest_path: &Path, audio_root: &Path, output_root: &Path, seed: &str) -> Result<()> {
    if seed.len() != 64 || !seed.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("SYMTHAEA_BLIND_PACK_SEED must be exactly 64 hexadecimal characters (32 bytes)".to_owned());
    }
    if !audio_root.is_dir() {
        return Err(format!("audio root {} is not a directory", audio_root.display()));
    }
    let manifest_bytes = fs::read(manifest_path)
        .map_err(|error| format!("cannot read manifest {}: {error}", manifest_path.display()))?;
    let manifest_sha256 = sha256(&manifest_bytes);
    let manifest: Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("invalid evidence JSON: {error}"))?;

    let (blocks, source_revision) = read_blocks(&manifest, audio_root)?;
    let participant_final = output_root.join("participant-pack");
    let operator_final = output_root.join("operator-only");
    let staging = output_root.join(".melothaea-blind-pack-staging");
    fs::create_dir_all(output_root)
        .map_err(|error| format!("cannot create output root {}: {error}", output_root.display()))?;
    if participant_final.exists() || operator_final.exists() || staging.exists() {
        return Err("output would overwrite an existing pack or staging directory; choose a clean output root".to_owned());
    }

    let participant_stage = staging.join("participant-pack");
    let operator_stage = staging.join("operator-only");
    let stimuli_stage = participant_stage.join("stimuli");
    fs::create_dir_all(&stimuli_stage)
        .map_err(|error| format!("cannot create staging directory: {error}"))?;
    fs::create_dir_all(&operator_stage)
        .map_err(|error| format!("cannot create operator staging directory: {error}"))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&operator_stage, fs::Permissions::from_mode(0o700))
            .map_err(|error| format!("cannot restrict operator-key directory permissions: {error}"))?;
    }

    let outcome = (|| -> Result<()> {
        let mut block_order: Vec<usize> = (0..blocks.len()).collect();
        block_order.sort_by_key(|index| {
            score(
                seed,
                "block-order-v1",
                &format!("{}|{}", blocks[*index].style, blocks[*index].seed_label),
            )
        });

        let mut participant_csv = String::from(
            "sequence,block_code,clip_code,target_style,audio_file,style_fit_1_7,overall_quality_1_7,audible_artifacts_1_7,notes\n",
        );
        let mut key_clips = Vec::new();
        let mut clip_ids = BTreeSet::new();
        let mut sequence = 0usize;

        for block_index in block_order {
            let block = &blocks[block_index];
            let block_identity = format!("{}|{}", block.style, block.seed_label);
            let block_code = format!(
                "block-{}",
                derived_code(seed, "block-code-v1", &block_identity)
            );

            let mut arms: Vec<&SourceArm> = block.arms.iter().collect();
            arms.sort_by_key(|arm| {
                score(
                    seed,
                    "within-block-order-v1",
                    &format!("{block_identity}|{}", arm.treatment),
                )
            });

            for arm in arms {
                sequence += 1;
                let clip_identity = format!("{block_identity}|{}", arm.treatment);
                let clip_code = format!(
                    "clip-{}",
                    derived_code(seed, "clip-code-v1", &clip_identity)
                );
                if !clip_ids.insert(clip_code.clone()) {
                    return Err(format!("randomized clip identifier collision: {clip_code}"));
                }
                let filename = format!("{clip_code}.wav");
                let destination = stimuli_stage.join(&filename);
                let copied = fs::copy(&arm.path, &destination).map_err(|error| {
                    format!("cannot copy verified WAV {}: {error}", arm.path.display())
                })?;
                if copied != arm.actual_bytes {
                    return Err(format!("copied byte count changed for {clip_code}"));
                }
                let copied_bytes = fs::read(&destination)
                    .map_err(|error| format!("cannot re-read copied WAV {clip_code}: {error}"))?;
                if sha256(&copied_bytes) != arm.verified_sha256 {
                    return Err(format!("post-copy byte verification failed for {clip_code}"));
                }

                write_csv_row(
                    &mut participant_csv,
                    &[
                        sequence.to_string(),
                        block_code.clone(),
                        clip_code.clone(),
                        block.style.clone(),
                        format!("stimuli/{filename}"),
                        String::new(),
                        String::new(),
                        String::new(),
                        String::new(),
                    ],
                );

                key_clips.push(json!({
                    "sequence": sequence,
                    "block_code": block_code,
                    "clip_code": clip_code,
                    "target_style": block.style,
                    "renderer_seed_label": block.seed_label,
                    "input_midi_sha256": block.midi_sha256,
                    "treatment_arm": arm.treatment,
                    "original_wav_path": arm.path.display().to_string(),
                    "original_wav_sha256": arm.declared_sha256,
                    "verified_wav_sha256": arm.verified_sha256,
                    "original_wav_bytes": arm.declared_bytes,
                    "copied_wav_bytes": copied,
                    "sample_rate_hz": arm.sample_rate_hz,
                    "participant_relative_path": format!("stimuli/{filename}")
                }));
            }
        }

        fs::write(participant_stage.join("ratings.csv"), participant_csv)
            .map_err(|error| format!("cannot write participant sheet: {error}"))?;

        let participant_manifest = json!({
            "schema": PACK_SCHEMA,
            "status": "ready_for_operator_review",
            "stimulus_count": key_clips.len(),
            "block_count": blocks.len(),
            "randomization_seed_sha256": sha256(seed.as_bytes()),
            "source_evidence_manifest_sha256": manifest_sha256,
            "source_revision_claim": source_revision,
            "participant_blinding": {
                "treatment_labels_in_participant_directory": false,
                "original_wav_paths_in_participant_directory": false,
                "source_wav_hashes_in_participant_directory": false,
                "operator_key_in_participant_directory": false
            },
            "interpretation_limits": [
                "This pack performs no loudness normalization or WAV re-encoding.",
                "A pack-generation pass proves source-byte identity and manifest completeness, not perceptual benefit.",
                "Use a separate controlled playback/presentation layer for level matching and record its gain separately.",
                "The seed digest does not prove random or secret seed generation; use a high-entropy 32-byte seed and protect the operator key."
            ]
        });
        let participant_bytes = serde_json::to_vec_pretty(&participant_manifest)
            .map_err(|error| format!("cannot serialize participant manifest: {error}"))?;
        fs::write(participant_stage.join("pack-manifest.json"), &participant_bytes)
            .map_err(|error| format!("cannot write participant manifest: {error}"))?;
        fs::write(
            participant_stage.join("LISTENER-INSTRUCTIONS.txt"),
            "Melothaea renderer profile listening pilot\n\n\
             For each clip, rate the target-style fit, overall audio quality, and audible artifacts on a 1-7 scale. \
             Use the notes column for brief observations. The same composition may occur in more than one clip; \
             judge each presentation independently before comparing your ratings. Do not infer that higher loudness, \
             brighter spectrum, or a higher crest factor means better quality. Please do not open operator-only data.\n\n\
             This is an exploratory pilot form, not a validated clinical or standards-compliant listening protocol. \
             Playback equipment, listening level, breaks, inclusion rules and primary endpoints should be frozen in \
             the study protocol before collecting confirmatory data.\n",
        ).map_err(|error| format!("cannot write listener instructions: {error}"))?;

        let key = json!({
            "schema": "melothaea-blind-listening-answer-key-v1",
            "access": "operator-only; never distribute with participant-pack",
            "randomization_seed": seed,
            "randomization_seed_sha256": sha256(seed.as_bytes()),
            "source_evidence_manifest_path": manifest_path.display().to_string(),
            "source_evidence_manifest_sha256": manifest_sha256,
            "source_revision_claim": source_revision,
            "clips": key_clips
        });
        let key_bytes = serde_json::to_vec_pretty(&key)
            .map_err(|error| format!("cannot serialize operator key: {error}"))?;
        let key_path = operator_stage.join("answer-key.json");
        fs::write(&key_path, &key_bytes)
            .map_err(|error| format!("cannot write operator key: {error}"))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600))
                .map_err(|error| format!("cannot restrict operator-key permissions: {error}"))?;
        }

        Ok(())
    })();

    if let Err(error) = outcome {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    if let Err(error) = fs::rename(&participant_stage, &participant_final) {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("cannot publish participant pack: {error}"));
    }
    if let Err(error) = fs::rename(&operator_stage, &operator_final) {
        let _ = fs::remove_dir_all(&participant_final);
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("cannot publish operator key: {error}"));
    }
    let _ = fs::remove_dir_all(&staging);

    eprintln!(
        "Created {} anonymous stimuli in {} matched blocks.",
        clip_ids.len(),
        blocks.len()
    );
    eprintln!("Participant deliverable: {}", participant_final.display());
    eprintln!(
        "OPERATOR ONLY — do not distribute: {}",
        operator_final.join("answer-key.json").display()
    );
    eprintln!("Pack status is ready for operator review; no listening result is implied.");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!(
            "Usage: SYMTHAEA_BLIND_PACK_SEED=<64-hex-secret> blind_render_listening_pack <evidence-manifest.json> <audio-root> <output-root>"
        );
        std::process::exit(2);
    }
    let seed = match env::var("SYMTHAEA_BLIND_PACK_SEED") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("SYMTHAEA_BLIND_PACK_SEED must be set; use a high-entropy 32-byte hex value");
            std::process::exit(2);
        }
    };
    if let Err(error) = build_pack(
        Path::new(&args[0]),
        Path::new(&args[1]),
        Path::new(&args[2]),
        &seed,
    ) {
        eprintln!("FAILED: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture_root(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock after UNIX epoch")
            .as_nanos();
        env::temp_dir().join(format!(
            "melothaea-blind-pack-{label}-{}-{nanos}",
            std::process::id()
        ))
    }

    fn make_fixture(root: &Path) -> (PathBuf, Vec<u8>) {
        fs::create_dir_all(root).expect("create fixture root");
        let wave = valid_test_wave();
        fs::write(root.join("fixture.wav"), &wave).expect("write fixture WAV");
        let audio_sha = sha256(&wave);
        let audio_bytes = wave.len() as u64;

        let arm = || {
            json!({
                "wav_path": "fixture.wav",
                "wav_sha256": audio_sha,
                "wav_bytes": audio_bytes,
                "sample_rate_hz": 44_100,
                "metrics": { "integrated_lufs": -16.0 }
            })
        };
        let mut records = Vec::new();
        for (index, style) in STYLES.iter().enumerate() {
            records.push(json!({
                "style": style,
                "seed": index + 1,
                "status": "complete",
                "input_midi": { "sha256": format!("{:064x}", index + 1) },
                "arms": {
                    "fluidsynth_default_profile": arm(),
                    "fluidsynth_room_only": arm(),
                    "fluidsynth_mastering_only": arm(),
                    "fluidsynth_dialect_profile": arm()
                }
            }));
        }

        let manifest = json!({
            "schema": EXPECTED_SCHEMA,
            "run_status": "complete",
            "source_revision": {
                "value": "a".repeat(40),
                "format_valid": true
            },
            "contract": { "expected_dialect_count": STYLES.len() },
            "cohort": {
                "expected": records.len(),
                "completed": records.len(),
                "record_count_including_failures": records.len()
            },
            "records": records,
            "failures": []
        });
        let manifest_path = root.join("renderer-ab-fair-manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");
        (manifest_path, wave)
    }

    #[test]
    fn valid_manifest_has_all_styles_and_verified_audio() {
        let root = fixture_root("valid-manifest");
        let (manifest_path, _) = make_fixture(&root);
        let bytes = fs::read(&manifest_path).expect("read manifest");
        let manifest: Value = serde_json::from_slice(&bytes).expect("parse fixture manifest");
        let (blocks, revision) = read_blocks(&manifest, &root).expect("valid complete cohort");
        assert_eq!(blocks.len(), STYLES.len());
        assert_eq!(revision, "a".repeat(40));
        assert_eq!(blocks.iter().map(|block| block.arms.len()).sum::<usize>(), STYLES.len() * ARMS.len());
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn wav_header_sample_rate_must_match_manifest() {
        let root = fixture_root("sample-rate-mismatch");
        let (manifest_path, _) = make_fixture(&root);
        let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        for record in manifest["records"].as_array_mut().unwrap() {
            record["arms"]["fluidsynth_default_profile"]["sample_rate_hz"] = json!(48_000);
            record["arms"]["fluidsynth_room_only"]["sample_rate_hz"] = json!(48_000);
            record["arms"]["fluidsynth_mastering_only"]["sample_rate_hz"] = json!(48_000);
            record["arms"]["fluidsynth_dialect_profile"]["sample_rate_hz"] = json!(48_000);
        }
        let result = read_blocks(&manifest, &root);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("disagrees with WAV header"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn renderer_receipt_hash_mismatch_fails_closed() {
        let root = fixture_root("hash-mismatch");
        let (manifest_path, _) = make_fixture(&root);
        let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["records"][0]["arms"]["fluidsynth_default_profile"]["wav_sha256"] = Value::String("f".repeat(64));
        let result = read_blocks(&manifest, &root);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("SHA-256"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn failed_render_cell_is_not_silently_dropped() {
        let root = fixture_root("failed-cell");
        let (manifest_path, _) = make_fixture(&root);
        let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["records"][0]["status"] = Value::String("failed".to_owned());
        let result = read_blocks(&manifest, &root);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("refusing to drop a failed cell"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn missing_style_cell_fails_even_if_counts_are_consistent() {
        let root = fixture_root("missing-style");
        let (manifest_path, _) = make_fixture(&root);
        let mut manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        manifest["records"].as_array_mut().unwrap().pop();
        let count = manifest["records"].as_array().unwrap().len() as u64;
        manifest["cohort"]["expected"] = json!(count);
        manifest["cohort"]["completed"] = json!(count);
        manifest["cohort"]["record_count_including_failures"] = json!(count);
        let result = read_blocks(&manifest, &root);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("style cohort incomplete"));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn complete_pack_is_repeatable_blinded_and_byte_preserving() {
        let root = fixture_root("pack");
        let (manifest_path, wave) = make_fixture(&root);
        let audio_root = root.clone();
        let first_output = root.join("out-one");
        let second_output = root.join("out-two");
        let seed = "c".repeat(64);
        build_pack(&manifest_path, &audio_root, &first_output, &seed).expect("first pack");
        build_pack(&manifest_path, &audio_root, &second_output, &seed).expect("second pack");

        let first_participant = first_output.join("participant-pack");
        let first_operator = first_output.join("operator-only");
        let csv_one = fs::read(first_participant.join("ratings.csv")).unwrap();
        let csv_two = fs::read(second_output.join("participant-pack/ratings.csv")).unwrap();
        assert_eq!(csv_one, csv_two);

        let csv = String::from_utf8(csv_one).unwrap();
        for treatment in [
            "default_default",
            "dialect_room_only",
            "dialect_mastering_only",
            "dialect_room_mastering",
        ] {
            assert!(!csv.contains(treatment));
        }
        assert!(!csv.contains("fixture.wav"));
        assert!(!csv.contains(&sha256(&wave)));
        assert!(!first_participant.join("operator-only/answer-key.json").exists());

        let files = fs::read_dir(first_participant.join("stimuli")).unwrap().count();
        assert_eq!(files, STYLES.len() * ARMS.len());
        for entry in fs::read_dir(first_participant.join("stimuli")).unwrap() {
            let path = entry.unwrap().path();
            assert_eq!(fs::read(path).unwrap(), wave);
        }

        let key = fs::read_to_string(first_operator.join("answer-key.json")).unwrap();
        assert!(key.contains("treatment_arm"));
        assert!(key.contains("default_default"));
        assert!(key.contains(&sha256(&wave)));
        assert!(key.contains(&seed));
        let pack_manifest = fs::read_to_string(first_participant.join("pack-manifest.json")).unwrap();
        assert!(!pack_manifest.contains(&seed));
        assert!(pack_manifest.contains(&sha256(seed.as_bytes())));
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn deterministic_codes_repeat_for_same_seed() {
        let a = derived_code(&"a".repeat(64), "clip-code-v1", "block-A|profile");
        let b = derived_code(&"a".repeat(64), "clip-code-v1", "block-A|profile");
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_changes_anonymous_code() {
        let a = derived_code(&"a".repeat(64), "clip-code-v1", "block-A|profile");
        let b = derived_code(&"b".repeat(64), "clip-code-v1", "block-A|profile");
        assert_ne!(a, b);
    }

    #[test]
    fn participant_csv_escapes_quotes_and_commas() {
        assert_eq!(csv_field("jazz, \"late\""), "\"jazz, \"\"late\"\"\"");
        assert_eq!(csv_field("plain"), "plain");
    }

    fn valid_test_wave() -> Vec<u8> {
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&40u32.to_le_bytes()); // file size excluding RIFF header
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&2u16.to_le_bytes()); // stereo
        bytes.extend_from_slice(&44_100u32.to_le_bytes());
        bytes.extend_from_slice(&176_400u32.to_le_bytes());
        bytes.extend_from_slice(&4u16.to_le_bytes()); // block alignment
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        bytes
    }

    #[test]
    fn wave_container_requires_consistent_riff_chunks() {
        assert!(validate_wave_bytes(&valid_test_wave(), "fixture").is_ok());
        assert!(validate_wave_bytes(b"RIFFxxxxWAVE", "fixture").is_err());
        let mut truncated = valid_test_wave();
        truncated.pop();
        assert!(validate_wave_bytes(&truncated, "fixture").is_err());
        let mut no_data = valid_test_wave();
        no_data[40..44].copy_from_slice(&0u32.to_le_bytes());
        assert!(validate_wave_bytes(&no_data, "fixture").is_err());
    }

    #[test]
    fn within_block_order_is_seeded_and_canonical() {
        let seed = "c".repeat(64);
        let mut arms_a = ARMS.map(|(label, _)| label).to_vec();
        let mut arms_b = arms_a.clone();
        arms_a.sort_by_key(|arm| score(&seed, "within-block-order-v1", &format!("piece|1|{arm}")));
        arms_b.reverse();
        arms_b.sort_by_key(|arm| score(&seed, "within-block-order-v1", &format!("piece|1|{arm}")));
        assert_eq!(arms_a, arms_b);
        assert_eq!(arms_a.iter().collect::<BTreeSet<_>>().len(), ARMS.len());
    }
}

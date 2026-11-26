//! Official MusicListenBench v1.0 manifest + JSONL submission importer.
//!
//! Pairs the released item manifests by stable base item ID, verifies
//! perturbation metadata, maps each exact question wording to its semantic
//! A/B relations, hashes locally materialized WAV bytes, and scores the
//! official answer JSONL. No filename-only pairing or guessed answer mapping.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::auditory_invariance::{
    audit_auditory_invariance, AuditoryInvarianceReportV1, AuditoryInvarianceRunV1,
    AuditoryPairV1, AuditoryVariantV1, DatasetSplitV1, FlipFactorV1, ModelAnswerV1,
    MusicalRelationV1, MusicalTaskV1, StayFactorV1,
};

const IMPORT_SCHEMA_V1: &str = "mel-musiclistenbench-official-import-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfficialSubmissionMetaV1 {
    pub model: String,
    pub version: String,
    pub readout: String,
    pub used_training_split: String,
    pub date: String,
    pub code_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct OfficialMusicListenBenchImportReportV1 {
    pub schema: &'static str,
    pub official_submission_meta: OfficialSubmissionMetaV1,
    /// Fingerprint of the three exact local evaluation-manifest files.
    /// This is not asserted to be a signed Hugging Face/Git release identity.
    pub input_manifest_sha256: String,
    pub clean_eval_manifest_sha256: String,
    pub perturb_eval_manifest_sha256: String,
    pub perturb_metadata_sha256: String,
    pub submission_jsonl_sha256: String,
    pub clean_manifest_items: usize,
    pub perturb_manifest_items: usize,
    pub perturb_metadata_items: usize,
    pub submitted_answer_items: usize,
    pub missing_answer_items: usize,
    pub missing_answer_examples: Vec<String>,
    pub official_scoring_report: AuditoryInvarianceReportV1,
    /// WAV bytes have been read and SHA-256 hashed from the supplied local roots.
    pub audio_bytes_sha256_recomputed: bool,
    /// JSONL model/version metadata is self-reported, not signed inference evidence.
    pub inference_receipt_authenticated: bool,
}

#[derive(Debug, Clone)]
struct OfficialRow {
    variant_id: String,
    base_item_id: String,
    task: MusicalTaskV1,
    condition: VariantConditionV1,
    factor: Option<String>,
    voice_path: String,
    question: String,
    gold: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum VariantConditionV1 {
    Clean,
    Flip,
    Stay,
}

#[derive(Debug, Clone, Deserialize)]
struct PerturbMetadataV1 {
    voice: String,
    task: String,
    item_id: String,
    condition: String,
    factor_changed: String,
    gold: String,
}

#[derive(Debug, Clone)]
struct SubmissionAnswers {
    meta: OfficialSubmissionMetaV1,
    answers: BTreeMap<String, ModelAnswerV1>,
    raw_item_count: usize,
}

pub fn import_official_submission(
    clean_eval_bytes: &[u8],
    perturb_eval_bytes: &[u8],
    perturb_metadata_bytes: &[u8],
    submission_jsonl_bytes: &[u8],
    clean_audio_root: &Path,
    perturb_audio_root: &Path,
) -> Result<OfficialMusicListenBenchImportReportV1, String> {
    let clean_eval: Vec<Value> = serde_json::from_slice(clean_eval_bytes)
        .map_err(|error| format!("invalid official clean eval JSON: {error}"))?;
    let perturb_eval: Vec<Value> = serde_json::from_slice(perturb_eval_bytes)
        .map_err(|error| format!("invalid official perturb eval JSON: {error}"))?;
    let perturb_metadata = parse_jsonl::<PerturbMetadataV1>(perturb_metadata_bytes, "perturb metadata")?;
    let submission = parse_submission(submission_jsonl_bytes)?;

    if clean_eval.len() != 1_000 {
        return Err(format!("MusicListenBench v1.0 rendered clean manifest must contain 1,000 items; got {}", clean_eval.len()));
    }
    if perturb_eval.len() != 2_000 {
        return Err(format!("MusicListenBench v1.0 rendered FLIP/STAY manifest must contain 2,000 items; got {}", perturb_eval.len()));
    }
    if perturb_metadata.len() != perturb_eval.len() {
        return Err(format!("perturb metadata count {} differs from perturb eval count {}", perturb_metadata.len(), perturb_eval.len()));
    }

    let mut metadata_by_voice = BTreeMap::new();
    for (index, item) in perturb_metadata.iter().enumerate() {
        if metadata_by_voice.insert(item.voice.as_str(), item).is_some() {
            return Err(format!("duplicate perturb metadata voice at row {index}: {}", item.voice));
        }
    }

    let mut rows = Vec::with_capacity(3_000);
    for (index, item) in clean_eval.iter().enumerate() {
        rows.push(parse_official_row(item, VariantConditionV1::Clean, None, None)
            .map_err(|error| format!("clean eval row {index}: {error}"))?);
    }

    let mut seen_metadata = BTreeSet::new();
    for (index, item) in perturb_eval.iter().enumerate() {
        let voice = voice_path(item)?;
        let metadata = metadata_by_voice
            .get(voice.as_str())
            .ok_or_else(|| format!("perturb eval row {index} lacks exact metadata voice match: {voice}"))?;
        if !seen_metadata.insert(voice.as_str()) {
            return Err(format!("duplicate perturb eval voice: {voice}"));
        }
        let metadata_condition = parse_condition(&metadata.condition)?;
        let row = parse_official_row(item, metadata_condition, Some(&metadata.item_id), Some(&metadata.factor_changed))
            .map_err(|error| format!("perturb eval row {index}: {error}"))?;
        if metadata.task != task_name(row.task) {
            return Err(format!("perturb metadata task mismatch for {voice}"));
        }
        if metadata.gold.trim().to_ascii_uppercase() != row.gold {
            return Err(format!("perturb metadata gold answer mismatch for {voice}"));
        }
        rows.push(row);
    }
    if seen_metadata.len() != metadata_by_voice.len() {
        return Err("perturb metadata contains entries not present in eval_perturb.json".into());
    }

    let expected_ids: BTreeSet<_> = rows.iter().map(|row| row.variant_id.as_str()).collect();
    let unknown: Vec<_> = submission.answers
        .keys()
        .filter(|item_id| !expected_ids.contains(item_id.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(format!("submission contains {} unknown item IDs (first: {})", unknown.len(), unknown[0]));
    }

    let missing: Vec<_> = expected_ids
        .iter()
        .filter(|item_id| !submission.answers.contains_key(**item_id))
        .map(|item_id| (*item_id).to_owned())
        .collect();

    let mut grouped: BTreeMap<String, BTreeMap<VariantConditionV1, AuditoryVariantV1>> = BTreeMap::new();
    let mut pair_meta: BTreeMap<String, (MusicalTaskV1, String, Option<String>, Option<String>)> = BTreeMap::new();
    for row in &rows {
        let root = match row.condition {
            VariantConditionV1::Clean => clean_audio_root,
            VariantConditionV1::Flip | VariantConditionV1::Stay => perturb_audio_root,
        };
        let audio_sha256 = hash_audio_file(root, &row.voice_path)?;
        let (option_a_relation, option_b_relation) = option_relations(row.task, &row.question)?;
        let expected_relation = answer_relation(&row.gold, option_a_relation, option_b_relation)?;
        let model_answer = submission.answers.get(&row.variant_id).copied().unwrap_or(ModelAnswerV1::Unparseable);
        let variant = AuditoryVariantV1 {
            variant_id: row.variant_id.clone(),
            audio_sha256,
            question: row.question.clone(),
            question_sha256: sha256_hex(row.question.as_bytes()),
            option_a_relation,
            option_b_relation,
            expected_relation,
            model_answer,
        };
        let group = grouped.entry(row.base_item_id.clone()).or_default();
        if group.insert(row.condition, variant).is_some() {
            return Err(format!("duplicate {:?} variant for base item {}", row.condition, row.base_item_id));
        }
        let flip = if row.condition == VariantConditionV1::Flip {
            row.factor.clone()
        } else {
            None
        };
        let stay = if row.condition == VariantConditionV1::Stay {
            row.factor.clone()
        } else {
            None
        };
        let candidate = (row.task, row.base_item_id.clone(), flip, stay);
        let entry = pair_meta.entry(row.base_item_id.clone()).or_insert(candidate.clone());
        if entry.0 != candidate.0 {
            return Err(format!("task changed within base item {}", row.base_item_id));
        }
        if candidate.2.is_some() {
            if entry.2.is_some() {
                return Err(format!("duplicate FLIP factor for {}", row.base_item_id));
            }
            entry.2 = candidate.2;
        }
        if candidate.3.is_some() {
            if entry.3.is_some() {
                return Err(format!("duplicate STAY factor for {}", row.base_item_id));
            }
            entry.3 = candidate.3;
        }
    }

    if grouped.len() != 1_000 {
        return Err(format!("expected 1,000 underlying clean/FLIP/STAY groups; found {}", grouped.len()));
    }

    let mut pairs = Vec::with_capacity(grouped.len());
    for (base_item_id, variants) in grouped {
        let clean = variants.get(&VariantConditionV1::Clean)
            .ok_or_else(|| format!("base item {base_item_id} has no CLEAN variant"))?.clone();
        let flip = variants.get(&VariantConditionV1::Flip)
            .ok_or_else(|| format!("base item {base_item_id} has no FLIP variant"))?.clone();
        let stay = variants.get(&VariantConditionV1::Stay)
            .ok_or_else(|| format!("base item {base_item_id} has no STAY variant"))?.clone();
        let (task, _, flip_name, stay_name) = pair_meta.remove(&base_item_id)
            .ok_or_else(|| format!("missing factor metadata for {base_item_id}"))?;
        let flip_factor = parse_flip_factor(flip_name.as_deref().ok_or_else(|| format!("missing FLIP factor for {base_item_id}"))?)?;
        let stay_factor = parse_stay_factor(stay_name.as_deref().ok_or_else(|| format!("missing STAY factor for {base_item_id}"))?)?;
        pairs.push(AuditoryPairV1 {
            item_id: base_item_id.clone(),
            pair_id: base_item_id.clone(),
            source_lineage_sha256: sha256_hex(base_item_id.as_bytes()),
            task,
            flip_factor,
            stay_factor,
            clean,
            flip,
            stay,
        });
    }

    let manifest_sha256 = manifest_fingerprint(&[
        ("data/eval.json", clean_eval_bytes),
        ("data/eval_perturb.json", perturb_eval_bytes),
        ("data/eval_perturb.meta.jsonl", perturb_metadata_bytes),
    ]);
    let model_identity_sha256 = sha256_hex(
        format!("{}\n{}", submission.meta.model, submission.meta.version).as_bytes(),
    );
    let run = AuditoryInvarianceRunV1 {
        schema: "mel-auditory-flip-stay-run-v1".into(),
        dataset_id: "amaai-lab/MusicListenBench".into(),
        dataset_version: "1.0".into(),
        dataset_revision_sha256: manifest_sha256.clone(),
        split: DatasetSplitV1::RenderedHeldOut,
        model_identity_sha256,
        system_prompt_sha256: None,
        runtime_identity_sha256: None,
        decoding_config_sha256: None,
        pairs,
    };
    let official_scoring_report = audit_auditory_invariance(&run)?;

    Ok(OfficialMusicListenBenchImportReportV1 {
        schema: IMPORT_SCHEMA_V1,
        official_submission_meta: submission.meta,
        input_manifest_sha256: manifest_sha256,
        clean_eval_manifest_sha256: sha256_hex(clean_eval_bytes),
        perturb_eval_manifest_sha256: sha256_hex(perturb_eval_bytes),
        perturb_metadata_sha256: sha256_hex(perturb_metadata_bytes),
        submission_jsonl_sha256: sha256_hex(submission_jsonl_bytes),
        clean_manifest_items: clean_eval.len(),
        perturb_manifest_items: perturb_eval.len(),
        perturb_metadata_items: perturb_metadata.len(),
        submitted_answer_items: submission.raw_item_count,
        missing_answer_items: missing.len(),
        missing_answer_examples: missing.iter().take(20).cloned().collect(),
        official_scoring_report,
        audio_bytes_sha256_recomputed: true,
        inference_receipt_authenticated: false,
    })
}

fn parse_jsonl<T: for<'de> Deserialize<'de>>(bytes: &[u8], label: &str) -> Result<Vec<T>, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("{label} is not UTF-8: {error}"))?;
    text.lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(index, line)| serde_json::from_str(line).map_err(|error| format!("{label} line {}: {error}", index + 1)))
        .collect()
}

fn parse_submission(bytes: &[u8]) -> Result<SubmissionAnswers, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| format!("submission is not UTF-8: {error}"))?;
    let mut meta: Option<OfficialSubmissionMetaV1> = None;
    let mut answers = BTreeMap::new();
    let mut raw_item_count = 0;
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("submission line {} is invalid JSON: {error}", index + 1))?;
        if let Some(meta_value) = value.get("meta") {
            if meta.is_some() || raw_item_count != 0 {
                return Err("submission must have exactly one metadata header before answer rows".into());
            }
            let parsed: OfficialSubmissionMetaV1 = serde_json::from_value(meta_value.clone())
                .map_err(|error| format!("invalid official submission metadata: {error}"))?;
            validate_submission_meta(&parsed)?;
            meta = Some(parsed);
            continue;
        }
        let item_id = value.get("item_id").and_then(Value::as_str)
            .ok_or_else(|| format!("submission line {} lacks string item_id", index + 1))?;
        if item_id.trim().is_empty() {
            return Err(format!("submission line {} has empty item_id", index + 1));
        }
        let answer = value.get("answer").and_then(Value::as_str).map(str::trim).map(str::to_ascii_uppercase);
        let model_answer = match answer.as_deref() {
            Some("A") => ModelAnswerV1::A,
            Some("B") => ModelAnswerV1::B,
            _ => ModelAnswerV1::Unparseable,
        };
        if answers.insert(item_id.to_owned(), model_answer).is_some() {
            return Err(format!("duplicate submission answer for item_id {item_id}"));
        }
        raw_item_count += 1;
    }
    Ok(SubmissionAnswers {
        meta: meta.ok_or_else(|| "submission lacks the required metadata header".to_string())?,
        answers,
        raw_item_count,
    })
}

fn validate_submission_meta(meta: &OfficialSubmissionMetaV1) -> Result<(), String> {
    for (label, value) in [
        ("model", meta.model.as_str()),
        ("version", meta.version.as_str()),
        ("readout", meta.readout.as_str()),
        ("used_training_split", meta.used_training_split.as_str()),
        ("date", meta.date.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(format!("submission metadata {label} must not be empty"));
        }
    }
    if meta.readout != "logprob" && meta.readout != "generated" {
        return Err("submission metadata readout must be 'logprob' or 'generated'".into());
    }
    Ok(())
}

fn parse_official_row(
    value: &Value,
    expected_condition: VariantConditionV1,
    expected_base_id: Option<&str>,
    expected_factor: Option<&str>,
) -> Result<OfficialRow, String> {
    let voice = voice_path(value)?;
    let question = value.get("conversations")
        .and_then(Value::as_array)
        .and_then(|messages| messages.first())
        .and_then(|message| message.get("value"))
        .and_then(Value::as_str)
        .ok_or_else(|| "conversation has no human question text".to_string())?
        .to_owned();
    let gold = value.get("conversations")
        .and_then(Value::as_array)
        .and_then(|messages| messages.get(1))
        .and_then(|message| message.get("value"))
        .and_then(Value::as_str)
        .ok_or_else(|| "conversation has no gold answer".to_string())?
        .trim()
        .to_ascii_uppercase();
    if gold != "A" && gold != "B" {
        return Err(format!("official gold answer is not A/B: {gold}"));
    }
    let variant_id = variant_id_from_voice(&voice)?;
    let (base_item_id, condition, factor) = parse_variant_id(&variant_id)?;
    if condition != expected_condition {
        return Err(format!("variant {variant_id} has condition {condition:?}, expected {expected_condition:?}"));
    }
    if expected_base_id.is_some_and(|expected| expected != base_item_id) {
        return Err(format!("metadata item_id mismatch for {variant_id}"));
    }
    if expected_factor.is_some_and(|expected| factor.as_deref() != Some(expected)) {
        return Err(format!("metadata factor mismatch for {variant_id}"));
    }
    let task = task_from_voice(&voice)?;
    let (option_a, option_b) = option_relations(task, &question)?;
    let _ = answer_relation(&gold, option_a, option_b)?;
    Ok(OfficialRow {
        variant_id,
        base_item_id,
        task,
        condition,
        factor,
        voice_path: voice,
        question,
        gold,
    })
}

fn parse_variant_id(variant_id: &str) -> Result<(String, VariantConditionV1, Option<String>), String> {
    if let Some(base) = variant_id.strip_suffix("__base") {
        return Ok((base.to_owned(), VariantConditionV1::Clean, None));
    }
    if let Some((base, factor)) = variant_id.rsplit_once("__flip__") {
        if !base.is_empty() && !factor.is_empty() {
            return Ok((base.to_owned(), VariantConditionV1::Flip, Some(factor.to_owned())));
        }
    }
    if let Some((base, factor)) = variant_id.rsplit_once("__stay__") {
        if !base.is_empty() && !factor.is_empty() {
            return Ok((base.to_owned(), VariantConditionV1::Stay, Some(factor.to_owned())));
        }
    }
    Err(format!("unsupported MusicListenBench item ID: {variant_id}"))
}

fn option_relations(
    task: MusicalTaskV1,
    question: &str,
) -> Result<(MusicalRelationV1, MusicalRelationV1), String> {
    let lower = question.to_ascii_lowercase();
    match task {
        MusicalTaskV1::Melody | MusicalTaskV1::Harmony | MusicalTaskV1::Timbre => {
            if !lower.contains("only answer 'a' for the same, 'b' for different") {
                return Err("unsupported same/different instruction template".into());
            }
            Ok((MusicalRelationV1::Same, MusicalRelationV1::Different))
        }
        MusicalTaskV1::Rhythm => {
            if !lower.contains("only answer 'a' for the first, 'b' for the second") {
                return Err("unsupported rhythm instruction template".into());
            }
            let asks_slower = [
                "slower",
                "more leisurely",
                "more slowly",
                "slower pace",
                "feels slower",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            let asks_faster = [
                "faster",
                "more rapidly",
                "more rapid",
                "faster pace",
                "faster rhythm",
                "more speed",
            ]
            .iter()
            .any(|needle| lower.contains(needle));
            match (asks_faster, asks_slower) {
                (true, false) => Ok((MusicalRelationV1::ClipAFaster, MusicalRelationV1::ClipBFaster)),
                (false, true) => Ok((MusicalRelationV1::ClipBFaster, MusicalRelationV1::ClipAFaster)),
                _ => Err("rhythm question does not identify exactly one faster/slower direction".into()),
            }
        }
    }
}

fn answer_relation(
    answer: &str,
    option_a: MusicalRelationV1,
    option_b: MusicalRelationV1,
) -> Result<MusicalRelationV1, String> {
    match answer {
        "A" => Ok(option_a),
        "B" => Ok(option_b),
        _ => Err(format!("answer must be A/B, got {answer}")),
    }
}

fn voice_path(value: &Value) -> Result<String, String> {
    let voice_value = value.get("voice").ok_or_else(|| "row has no voice field".to_string())?;
    let voice = if let Some(path) = voice_value.as_str() {
        path
    } else if let Some(path) = voice_value.as_array().and_then(|items| items.first()).and_then(Value::as_str) {
        path
    } else {
        return Err("voice must be a string or a non-empty string array".into());
    };
    let path = Path::new(voice);
    if path.is_absolute() || path.components().any(|component| !matches!(component, Component::Normal(_))) {
        return Err(format!("voice path is not a safe relative path: {voice}"));
    }
    Ok(voice.to_owned())
}

fn variant_id_from_voice(voice: &str) -> Result<String, String> {
    let path = Path::new(voice);
    let file = path.file_name().and_then(|name| name.to_str())
        .ok_or_else(|| format!("voice path has no UTF-8 filename: {voice}"))?;
    let id = file.strip_suffix(".wav").ok_or_else(|| format!("voice filename has no .wav suffix: {voice}"))?;
    Ok(id.to_owned())
}

fn task_from_voice(voice: &str) -> Result<MusicalTaskV1, String> {
    let task = voice.split('/').next().unwrap_or_default();
    match task {
        "Q1_melody" => Ok(MusicalTaskV1::Melody),
        "Q2_harmony" => Ok(MusicalTaskV1::Harmony),
        "Q3_timbre" => Ok(MusicalTaskV1::Timbre),
        "Q4_rhythm" => Ok(MusicalTaskV1::Rhythm),
        _ => Err(format!("unsupported MusicListenBench task folder: {task}")),
    }
}

fn task_name(task: MusicalTaskV1) -> &'static str {
    match task {
        MusicalTaskV1::Melody => "Q1_melody",
        MusicalTaskV1::Harmony => "Q2_harmony",
        MusicalTaskV1::Timbre => "Q3_timbre",
        MusicalTaskV1::Rhythm => "Q4_rhythm",
    }
}

fn parse_condition(value: &str) -> Result<VariantConditionV1, String> {
    match value {
        "flip" => Ok(VariantConditionV1::Flip),
        "stay" => Ok(VariantConditionV1::Stay),
        _ => Err(format!("unsupported perturb condition: {value}")),
    }
}

fn parse_flip_factor(value: &str) -> Result<FlipFactorV1, String> {
    match value {
        "note_identity" => Ok(FlipFactorV1::NoteIdentity),
        "third_quality" => Ok(FlipFactorV1::ThirdQuality),
        "instrument_family" => Ok(FlipFactorV1::InstrumentFamily),
        "pulse_rate" => Ok(FlipFactorV1::PulseRate),
        _ => Err(format!("unsupported MusicListenBench FLIP factor: {value}")),
    }
}

fn parse_stay_factor(value: &str) -> Result<StayFactorV1, String> {
    match value {
        "room" => Ok(StayFactorV1::Room),
        "eq" => Ok(StayFactorV1::Eq),
        "hiss" => Ok(StayFactorV1::Hiss),
        "transposition" => Ok(StayFactorV1::Transposition),
        _ => Err(format!("unsupported MusicListenBench STAY factor: {value}")),
    }
}

fn hash_audio_file(root: &Path, relative: &str) -> Result<String, String> {
    let relative_path = Path::new(relative);
    if relative_path.is_absolute()
        || relative_path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe audio path: {relative}"));
    }
    let full_path: PathBuf = root.join(relative_path);
    let bytes = fs::read(&full_path)
        .map_err(|error| format!("cannot read audio file {}: {error}", full_path.display()))?;
    Ok(sha256_hex(&bytes))
}

fn manifest_fingerprint(files: &[(&str, &[u8])]) -> String {
    let mut hasher = Sha256::new();
    for (name, bytes) in files {
        hasher.update((name.len() as u64).to_be_bytes());
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_be_bytes());
        hasher.update(sha256_bytes(bytes));
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_flip_stay_item_ids_pair_by_exact_base_id() {
        assert_eq!(
            parse_variant_id("Q1_melody__test__100__000001__base").unwrap(),
            (
                "Q1_melody__test__100__000001".into(),
                VariantConditionV1::Clean,
                None,
            )
        );
        assert_eq!(
            parse_variant_id("Q1_melody__test__100__000001__flip__note_identity").unwrap(),
            (
                "Q1_melody__test__100__000001".into(),
                VariantConditionV1::Flip,
                Some("note_identity".into()),
            )
        );
        assert_eq!(
            parse_variant_id("Q1_melody__test__100__000001__stay__hiss").unwrap(),
            (
                "Q1_melody__test__100__000001".into(),
                VariantConditionV1::Stay,
                Some("hiss".into()),
            )
        );
    }

    #[test]
    fn exact_released_question_templates_map_to_semantic_relations() {
        for task in [
            MusicalTaskV1::Melody,
            MusicalTaskV1::Harmony,
            MusicalTaskV1::Timbre,
        ] {
            assert_eq!(
                option_relations(
                    task,
                    "Two melodic clips are joined by a brief silence. Same melody, or not? Only answer 'A' for the same, 'B' for different. Do not add any explanation. <audio>"
                )
                .unwrap(),
                (MusicalRelationV1::Same, MusicalRelationV1::Different)
            );
        }
        let faster_questions = [
            "Which segment moves faster?",
            "Which segment plays more rapidly?",
            "Which part proceeds at a faster pace?",
            "Which part has a faster rhythm?",
            "Which clip carries more speed?",
        ];
        for question in faster_questions {
            let full = format!("{question} Only answer 'A' for the first, 'B' for the second. <audio>");
            assert_eq!(
                option_relations(MusicalTaskV1::Rhythm, &full).unwrap(),
                (MusicalRelationV1::ClipAFaster, MusicalRelationV1::ClipBFaster),
                "{question}"
            );
        }
        let slower_questions = [
            "Which segment moves slower?",
            "Which clip runs more leisurely?",
            "Which audio feels slower?",
            "Which part proceeds at a slower pace?",
        ];
        for question in slower_questions {
            let full = format!("{question} Only answer 'A' for the first, 'B' for the second. <audio>");
            assert_eq!(
                option_relations(MusicalTaskV1::Rhythm, &full).unwrap(),
                (MusicalRelationV1::ClipBFaster, MusicalRelationV1::ClipAFaster),
                "{question}"
            );
        }
    }

    #[test]
    fn gold_letter_decoding_produces_the_semantic_relation() {
        assert_eq!(
            answer_relation("A", MusicalRelationV1::ClipBFaster, MusicalRelationV1::ClipAFaster).unwrap(),
            MusicalRelationV1::ClipBFaster
        );
        assert_eq!(
            answer_relation("B", MusicalRelationV1::ClipBFaster, MusicalRelationV1::ClipAFaster).unwrap(),
            MusicalRelationV1::ClipAFaster
        );
        assert!(answer_relation("C", MusicalRelationV1::Same, MusicalRelationV1::Different).is_err());
    }

    #[test]
    fn official_submission_jsonl_preserves_metadata_and_invalid_answers() {
        let jsonl = concat!(
            "{\"meta\":{\"model\":\"test-model\",\"version\":\"v1\",\"readout\":\"generated\",\"used_training_split\":\"none\",\"date\":\"2026-11-26\"}}\n",
            "{\"item_id\":\"Q1_melody__test__1__000000__base\",\"answer\":\"A\"}\n",
            "{\"item_id\":\"Q1_melody__test__1__000000__flip__note_identity\",\"answer\":\"invalid\"}\n"
        );
        let submission = parse_submission(jsonl.as_bytes()).unwrap();
        assert_eq!(submission.meta.model, "test-model");
        assert_eq!(submission.raw_item_count, 2);
        assert_eq!(
            submission.answers["Q1_melody__test__1__000000__base"],
            ModelAnswerV1::A
        );
        assert_eq!(
            submission.answers["Q1_melody__test__1__000000__flip__note_identity"],
            ModelAnswerV1::Unparseable
        );
    }

    #[test]
    fn manifest_fingerprint_is_ordered_by_declared_file_names() {
        let a = manifest_fingerprint(&[("a.json", b"one"), ("b.json", b"two")]);
        let b = manifest_fingerprint(&[("b.json", b"two"), ("a.json", b"one")]);
        assert_ne!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn unsafe_audio_paths_are_rejected_before_file_access() {
        assert!(voice_path(&serde_json::json!({"voice":"../escape.wav"})).is_err());
        assert!(voice_path(&serde_json::json!({"voice":"/tmp/escape.wav"})).is_err());
    }
}

fn sha256_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

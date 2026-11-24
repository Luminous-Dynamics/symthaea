//! Deterministic symbolic scoring for MUSE paired edit evidence v1.
//! The initial implementation supports explicit one-to-one velocity, pitch and rational-duration edits.
//! Unsupported operations and correspondence relations fail closed.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::time::Instant;
use symthaea_music_theory::{
    pitch::Pitch,
    rhythm::Duration,
    score::{Emphasis, PartId, Score, ScoreNote, VoiceRole},
};

const CASE_SCHEMA: &str = "muse-edit-evidence-case-v1";
const RESULT_SCHEMA: &str = "muse-edit-evidence-result-v1";
const CASE_DIGEST_SCHEMA: &str = "muse-edit-evidence-case-serde-json-sha256-v1";
const SCORE_DIGEST_SCHEMA: &str = "muse-score-serde-json-sha256-v1";
const CASE_DIGEST_DOMAIN: &[u8] = b"luminous-dynamics-muse-edit-evidence-case-v1";
const RESULT_DIGEST_DOMAIN: &[u8] = b"luminous-dynamics-muse-edit-evidence-result-v1";

#[derive(Clone, Copy)]
struct Scope {
    part: u16,
    start_num: i64,
    start_den: i64,
    end_num: i64,
    end_den: i64,
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    fn normalize(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort_unstable();
                let mut sorted = serde_json::Map::new();
                for key in keys {
                    let child = object.get(key).expect("key came from this object");
                    sorted.insert(key.clone(), normalize(child));
                }
                Value::Object(sorted)
            }
            Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
            _ => value.clone(),
        }
    }
    serde_json::to_vec(&normalize(value))
        .map_err(|error| format!("canonical JSON serialization failed: {error}"))
}

fn score_sha256(score: &Score) -> Result<String, String> {
    let bytes = serde_json::to_vec(score)
        .map_err(|error| format!("could not serialize typed Score: {error}"))?;
    Ok(sha256(&bytes))
}

fn case_sha256(case: &Value) -> Result<String, String> {
    let bytes = canonical_json_bytes(case)?;
    let mut hash = Sha256::new();
    hash.update(CASE_DIGEST_DOMAIN);
    hash.update([0]);
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_score(score: &Score, label: &str) -> Result<(), String> {
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0
        || score.meter == 0
        || score.key.tonic.value() > 11
        || score.total_beats.den() <= 0
        || score.total_beats.num() < 0
        || score.total_beats.num() > 1_000_000_000
        || score.total_beats.den() > 1_000_000
    {
        return Err(format!("{label} has invalid tempo, key, meter or bounded total duration"));
    }
    for (index, note) in score.notes.iter().enumerate() {
        if note.pitch.midi() > 127
            || note.onset.den() <= 0 || note.onset.num() < 0
            || note.onset.num() > 1_000_000_000 || note.onset.den() > 1_000_000
            || note.duration.den() <= 0 || note.duration.num() <= 0
            || note.duration.num() > 1_000_000_000 || note.duration.den() > 1_000_000
            || !note.velocity.is_finite() || !(0.0..=1.0).contains(&note.velocity)
            || !note.section_intensity.is_finite() || note.section_intensity < 0.0
        {
            return Err(format!("{label}.notes[{index}] has invalid MIDI pitch, bounded rational timing or dynamics"));
        }
        let note_end = note.onset + note.duration;
        if cmp_fraction(
            note_end.num(),
            note_end.den(),
            score.total_beats.num(),
            score.total_beats.den(),
        ) == Ordering::Greater {
            return Err(format!("{label}.notes[{index}] ends beyond Score.total_beats"));
        }
    }
    Ok(())
}

fn required<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("{label} is missing {key}"))
}

fn check_object_shape(
    value: &Value,
    required_fields: &[&str],
    allowed_fields: &[&str],
    label: &str,
) -> Result<(), String> {
    let object = value.as_object().ok_or_else(|| format!("{label} must be an object"))?;
    for field in required_fields {
        if !object.contains_key(*field) {
            return Err(format!("{label} is missing required field {field}"));
        }
    }
    for field in object.keys() {
        if !allowed_fields.contains(&field.as_str()) {
            return Err(format!("{label} contains unsupported field {field}"));
        }
    }
    Ok(())
}

fn check_rational_shape(value: &Value, label: &str) -> Result<(), String> {
    check_object_shape(value, &["numerator", "denominator"], &["numerator", "denominator"], label)
}

fn check_scope_shape(value: &Value, label: &str) -> Result<(), String> {
    check_object_shape(value, &["part_id", "start", "end"], &["part_id", "start", "end"], label)?;
    check_rational_shape(required(value, "start", label)?, &format!("{label}.start"))?;
    check_rational_shape(required(value, "end", label)?, &format!("{label}.end"))
}

fn check_scope_array(value: &Value, label: &str, minimum: usize) -> Result<(), String> {
    let entries = value.as_array().ok_or_else(|| format!("{label} must be an array"))?;
    if entries.len() < minimum {
        return Err(format!("{label} requires at least {minimum} entries"));
    }
    for (index, entry) in entries.iter().enumerate() {
        check_scope_shape(entry, &format!("{label}[{index}]"))?;
    }
    Ok(())
}

/// Check all case objects against the v1 wire contract's closed-field sets
/// before using any values. The release pipeline still validates the full
/// JSON Schema (including ranges, patterns and enums) with a pinned validator.
fn validate_case_wire_shape(case: &Value) -> Result<(), String> {
    check_object_shape(
        case,
        &["schema", "case_id", "dataset_version", "split", "source_work_id", "rights", "source", "instruction", "edit_contract", "target", "evaluation"],
        &["schema", "case_id", "dataset_version", "split", "source_work_id", "composer_id", "rights", "source", "instruction", "edit_contract", "target", "render_profile", "evaluation"],
        "case",
    )?;
    let rights = required(case, "rights", "case")?;
    check_object_shape(
        rights,
        &["basis", "license_id", "derivative_policy"],
        &["basis", "license_id", "attribution", "consent_record_sha256", "derivative_policy"],
        "rights",
    )?;

    let source = required(case, "source", "case")?;
    check_object_shape(source, &["audio", "score_sha256", "stems"], &["audio", "score_sha256", "stems"], "source")?;
    let audio = required(source, "audio", "source")?;
    check_object_shape(audio, &["sha256", "duration_ms", "sample_rate_hz", "channels", "encoding"], &["sha256", "duration_ms", "sample_rate_hz", "channels", "encoding"], "source.audio")?;
    let stems = required(source, "stems", "source")?.as_array()
        .ok_or("source.stems must be an array")?;
    for (index, stem) in stems.iter().enumerate() {
        check_object_shape(stem, &["track_id", "audio_sha256", "score_part_id"], &["track_id", "audio_sha256", "score_part_id"], &format!("source.stems[{index}]"))?;
    }

    let instruction = required(case, "instruction", "case")?;
    check_object_shape(instruction, &["text", "locale", "operation_family", "strength"], &["text", "locale", "operation_family", "strength"], "instruction")?;

    let edit = required(case, "edit_contract", "case")?;
    check_object_shape(edit, &["target_scopes", "protected_scopes", "parameters"], &["target_scopes", "protected_scopes", "parameters"], "edit_contract")?;
    check_scope_array(required(edit, "target_scopes", "edit_contract")?, "edit_contract.target_scopes", 1)?;
    check_scope_array(required(edit, "protected_scopes", "edit_contract")?, "edit_contract.protected_scopes", 0)?;
    check_object_shape(
        required(edit, "parameters", "edit_contract")?,
        &[],
        &["scale_numerator", "scale_denominator", "semitones", "offset_beats", "tempo_multiplier", "gain_db", "pan", "instrument_id", "articulation", "description"],
        "edit_contract.parameters",
    )?;

    let target = required(case, "target", "case")?;
    check_object_shape(target, &["target_kind", "expected_score_sha256", "expected_audio_sha256", "acceptable_audio_sha256", "event_correspondence"], &["target_kind", "expected_score_sha256", "expected_audio_sha256", "acceptable_audio_sha256", "event_correspondence"], "target")?;
    let acceptable = required(target, "acceptable_audio_sha256", "target")?;
    if !acceptable.is_array() {
        return Err("target.acceptable_audio_sha256 must be an array".into());
    }
    let mappings = required(target, "event_correspondence", "target")?.as_array()
        .ok_or("target.event_correspondence must be an array")?;
    for (index, mapping) in mappings.iter().enumerate() {
        check_object_shape(mapping, &["source_note_ordinals", "target_note_ordinals", "relation"], &["source_note_ordinals", "target_note_ordinals", "relation"], &format!("target.event_correspondence[{index}]"))?;
    }

    if let Some(profile) = case.get("render_profile") {
        if !profile.is_null() {
            check_object_shape(
                profile,
                &["renderer_id", "renderer_version", "binary_sha256", "environment_sha256", "configuration_sha256", "asset_sha256"],
                &["renderer_id", "renderer_version", "binary_sha256", "environment_sha256", "configuration_sha256", "asset_sha256"],
                "render_profile",
            )?;
        }
    }

    let evaluation = required(case, "evaluation", "case")?;
    check_object_shape(evaluation, &["target_metrics", "preservation_metrics", "correspondence_required", "tolerances", "human_evaluation"], &["target_metrics", "preservation_metrics", "correspondence_required", "tolerances", "human_evaluation"], "evaluation")?;
    for field in ["target_metrics", "preservation_metrics"] {
        let metrics = required(evaluation, field, "evaluation")?
            .as_array()
            .ok_or_else(|| format!("evaluation.{field} must be an array"))?;
        if metrics.is_empty() {
            return Err(format!("evaluation.{field} must not be empty"));
        }
    }
    let tolerances = required(evaluation, "tolerances", "evaluation")?;
    check_object_shape(tolerances, &["onset_ms", "pitch_cents", "velocity_0_to_127", "minimum_correspondence_coverage"], &["onset_ms", "pitch_cents", "velocity_0_to_127", "minimum_correspondence_coverage"], "evaluation.tolerances")?;
    let human = required(evaluation, "human_evaluation", "evaluation")?;
    check_object_shape(human, &["required", "raters_per_item", "questions"], &["required", "raters_per_item", "questions"], "evaluation.human_evaluation")?;
    if !required(human, "questions", "evaluation.human_evaluation")?.is_array() {
        return Err("evaluation.human_evaluation.questions must be an array".into());
    }
    Ok(())
}

/// Reject unknown typed-Score fields before Serde's default permissive
/// deserializer can silently discard them.
fn validate_score_wire_shape(value: &Value, label: &str) -> Result<(), String> {
    check_object_shape(value, &["key", "tempo_bpm", "meter", "notes", "total_beats"], &["key", "tempo_bpm", "meter", "notes", "total_beats"], label)?;
    check_object_shape(required(value, "key", label)?, &["tonic", "tonality"], &["tonic", "tonality"], &format!("{label}.key"))?;
    check_rational_shape(required(value, "total_beats", label)?, &format!("{label}.total_beats"))?;
    let notes = required(value, "notes", label)?.as_array()
        .ok_or_else(|| format!("{label}.notes must be an array"))?;
    for (index, note) in notes.iter().enumerate() {
        let note_label = format!("{label}.notes[{index}]");
        check_object_shape(
            note,
            &["part", "pitch", "onset", "duration", "velocity", "role", "emphasis", "section_intensity"],
            &["part", "pitch", "onset", "duration", "velocity", "role", "emphasis", "section_intensity"],
            &note_label,
        )?;
        check_object_shape(required(note, "pitch", &note_label)?, &["midi"], &["midi"], &format!("{note_label}.pitch"))?;
        check_rational_shape(required(note, "onset", &note_label)?, &format!("{note_label}.onset"))?;
        check_rational_shape(required(note, "duration", &note_label)?, &format!("{note_label}.duration"))?;
    }
    Ok(())
}

fn deserialize_score(bytes: &[u8], label: &str) -> Result<Score, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid {label} Score JSON: {error}"))?;
    validate_score_wire_shape(&value, label)?;
    serde_json::from_value(value)
        .map_err(|error| format!("invalid typed {label} Score: {error}"))
}

fn string_field(value: &Value, key: &str, label: &str) -> Result<String, String> {
    required(value, key, label)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{label}.{key} must be a string"))
}

fn integer_field(value: &Value, key: &str, label: &str) -> Result<i64, String> {
    required(value, key, label)?
        .as_i64()
        .ok_or_else(|| format!("{label}.{key} must be an integer"))
}

fn optional_hash(value: &Value, key: &str, label: &str) -> Result<Option<String>, String> {
    match required(value, key, label)? {
        Value::Null => Ok(None),
        Value::String(hash) if is_sha256(hash) => Ok(Some(hash.clone())),
        _ => Err(format!("{label}.{key} must be null or a lowercase SHA-256 digest")),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn cmp_fraction(a_num: i64, a_den: i64, b_num: i64, b_den: i64) -> Ordering {
    (a_num as i128 * b_den as i128).cmp(&(b_num as i128 * a_den as i128))
}

fn parse_scopes(value: &Value, label: &str) -> Result<Vec<Scope>, String> {
    let entries = value.as_array().ok_or_else(|| format!("{label} must be an array"))?;
    let mut scopes = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let item = format!("{label}[{index}]");
        let part = integer_field(entry, "part_id", &item)?;
        let start = required(entry, "start", &item)?;
        let end = required(entry, "end", &item)?;
        let start_num = integer_field(start, "numerator", "scope.start")?;
        let start_den = integer_field(start, "denominator", "scope.start")?;
        let end_num = integer_field(end, "numerator", "scope.end")?;
        let end_den = integer_field(end, "denominator", "scope.end")?;
        if !(0..=65534).contains(&part)
            || start_num < 0 || end_num < 0
            || start_num > 1_000_000_000 || end_num > 1_000_000_000
            || start_den <= 0 || end_den <= 0
            || start_den > 1_000_000 || end_den > 1_000_000
        {
            return Err(format!("{item} has an invalid part or bounded rational time"));
        }
        if cmp_fraction(start_num, start_den, end_num, end_den) != Ordering::Less {
            return Err(format!("{item} must have start < end"));
        }
        scopes.push(Scope {
            part: part as u16,
            start_num,
            start_den,
            end_num,
            end_den,
        });
    }
    Ok(scopes)
}

fn scope_intersects(scope: Scope, note: &ScoreNote) -> bool {
    if note.part.0 != scope.part {
        return false;
    }
    let note_end = note.onset + note.duration;
    cmp_fraction(note.onset.num(), note.onset.den(), scope.end_num, scope.end_den)
        == Ordering::Less
        && cmp_fraction(note_end.num(), note_end.den(), scope.start_num, scope.start_den)
            == Ordering::Greater
}

fn scope_contains(scope: Scope, note: &ScoreNote) -> bool {
    if !scope_intersects(scope, note) {
        return false;
    }
    let note_end = note.onset + note.duration;
    cmp_fraction(scope.start_num, scope.start_den, note.onset.num(), note.onset.den())
        != Ordering::Greater
        && cmp_fraction(note_end.num(), note_end.den(), scope.end_num, scope.end_den)
            != Ordering::Greater
}

/// A note must be wholly contained in a matching scope or wholly outside it.
/// Boundary-crossing events are ambiguous and fail closed.
fn in_scopes(scopes: &[Scope], note: &ScoreNote, label: &str) -> Result<bool, String> {
    let mut contained = false;
    for scope in scopes.iter().copied() {
        if scope_intersects(scope, note) {
            if !scope_contains(scope, note) {
                return Err(format!("{label}: note crosses an authorized time-scope boundary"));
            }
            contained = true;
        }
    }
    Ok(contained)
}

fn scopes_overlap(a: Scope, b: Scope) -> bool {
    a.part == b.part
        && cmp_fraction(a.start_num, a.start_den, b.end_num, b.end_den) == Ordering::Less
        && cmp_fraction(b.start_num, b.start_den, a.end_num, a.end_den) == Ordering::Less
}

fn non_velocity_equal(a: &ScoreNote, b: &ScoreNote) -> bool {
    a.part == b.part
        && a.pitch == b.pitch
        && a.onset == b.onset
        && a.duration == b.duration
        && a.role == b.role
        && a.emphasis == b.emphasis
        && a.section_intensity.to_bits() == b.section_intensity.to_bits()
}

fn exact_equal(a: &ScoreNote, b: &ScoreNote) -> bool {
    non_velocity_equal(a, b) && a.velocity.to_bits() == b.velocity.to_bits()
}

fn non_pitch_equal(a: &ScoreNote, b: &ScoreNote) -> bool {
    a.part == b.part
        && a.velocity.to_bits() == b.velocity.to_bits()
        && a.onset == b.onset
        && a.duration == b.duration
        && a.role == b.role
        && a.emphasis == b.emphasis
        && a.section_intensity.to_bits() == b.section_intensity.to_bits()
}

fn non_duration_equal(a: &ScoreNote, b: &ScoreNote) -> bool {
    a.part == b.part
        && a.velocity.to_bits() == b.velocity.to_bits()
        && a.pitch == b.pitch
        && a.onset == b.onset
        && a.role == b.role
        && a.emphasis == b.emphasis
        && a.section_intensity.to_bits() == b.section_intensity.to_bits()
}

fn gcd_i128(mut a: i128, mut b: i128) -> i128 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.abs().max(1)
}

fn scaled_duration(value: Duration, numerator: i64, denominator: i64) -> Option<Duration> {
    if numerator <= 0 || denominator <= 0 {
        return None;
    }
    let raw_num = value.num() as i128 * numerator as i128;
    let raw_den = value.den() as i128 * denominator as i128;
    let divisor = gcd_i128(raw_num, raw_den);
    let reduced_num = i64::try_from(raw_num / divisor).ok()?;
    let reduced_den = i64::try_from(raw_den / divisor).ok()?;
    (reduced_num > 0 && reduced_den > 0).then(|| Duration::new(reduced_num, reduced_den))
}


fn score_metadata_equal(a: &Score, b: &Score) -> bool {
    a.key == b.key
        && a.tempo_bpm.to_bits() == b.tempo_bpm.to_bits()
        && a.meter == b.meter
        && a.total_beats == b.total_beats
}

fn metric(id: &str, numerator: usize, denominator: usize, threshold: f64, lower: bool) -> Value {
    let score = if denominator == 0 { None } else { Some(numerator as f64 / denominator as f64) };
    // An empty denominator is missing evidence, never a free pass.
    let passed = score.map(|actual| {
        if lower { actual <= threshold } else { actual >= threshold }
    }).unwrap_or(false);
    json!({
        "metric_id": id,
        "numerator": numerator,
        "denominator": denominator,
        "score": score,
        "threshold": threshold,
        "passed": passed
    })
}

fn unavailable_metric(id: &str) -> Value {
    json!({
        "metric_id": id,
        "numerator": 0,
        "denominator": 0,
        "score": null,
        "threshold": null,
        "passed": false
    })
}

fn seal_digest(result: &Value) -> Result<String, String> {
    let mut unsealed = result.clone();
    unsealed["artifacts"]["evaluation_receipt_sha256"] = Value::Null;
    let bytes = canonical_json_bytes(&unsealed)?;
    let mut hash = Sha256::new();
    hash.update(RESULT_DIGEST_DOMAIN);
    hash.update([0]);
    hash.update(bytes);
    Ok(format!("{:x}", hash.finalize()))
}

/// Recompute the domain-separated evaluation receipt seal.
pub fn verify_result_seal(result: &Value) -> Result<(), String> {
    let recorded = result
        .pointer("/artifacts/evaluation_receipt_sha256")
        .and_then(Value::as_str)
        .ok_or("result is missing evaluation receipt digest")?;
    if !is_sha256(recorded) || recorded != seal_digest(result)? {
        return Err("evaluation receipt seal is invalid".into());
    }
    Ok(())
}

/// Evaluate a v1 symbolic edit. Score bytes must deserialize as a typed
/// symthaea_music_theory Score. Optional audio is hashed as raw artifact bytes.
#[allow(clippy::too_many_lines)] // One v1 scoring transaction keeps all gates in one auditable path.
pub fn evaluate_symbolic_case(
    case_bytes: &[u8],
    source_score_bytes: &[u8],
    candidate_score_bytes: &[u8],
    candidate_audio_bytes: Option<&[u8]>,
) -> Result<Value, String> {
    let started = Instant::now();
    let case: Value = serde_json::from_slice(case_bytes)
        .map_err(|error| format!("invalid case JSON: {error}"))?;
    if string_field(&case, "schema", "case")? != CASE_SCHEMA {
        return Err("unsupported case schema".into());
    }
    validate_case_wire_shape(&case)?;
    let case_id = string_field(&case, "case_id", "case")?;
    let case_digest = case_sha256(&case)?;
    let instruction = required(&case, "instruction", "case")?;
    let operation = string_field(instruction, "operation_family", "instruction")?;
    if !matches!(operation.as_str(), "velocity" | "pitch" | "duration") {
        return Err(format!("abstain: operation family {operation} is not implemented"));
    }

    let source = deserialize_score(source_score_bytes, "source")?;
    let candidate = deserialize_score(candidate_score_bytes, "candidate")?;
    validate_score(&source, "source Score")?;
    validate_score(&candidate, "candidate Score")?;
    let source_digest = score_sha256(&source)?;
    let candidate_digest = score_sha256(&candidate)?;
    let source_binding = required(&case, "source", "case")?;
    let expected_source = optional_hash(source_binding, "score_sha256", "source")?
        .ok_or("source.score_sha256 is required for symbolic scoring")?;
    if source_digest != expected_source {
        return Err(format!("source Score digest mismatch: expected {expected_source}, got {source_digest}"));
    }
    if source.notes.is_empty() || source.notes.len() > 100_000 || candidate.notes.len() > 100_000 {
        return Err("source must contain 1..=100,000 notes and candidate must contain <=100,000".into());
    }

    let edit = required(&case, "edit_contract", "case")?;
    let target_scopes = parse_scopes(required(edit, "target_scopes", "edit_contract")?, "target_scopes")?;
    let protected_scopes = parse_scopes(required(edit, "protected_scopes", "edit_contract")?, "protected_scopes")?;
    if target_scopes.is_empty() {
        return Err("symbolic edit requires at least one target scope".into());
    }
    if target_scopes.iter().any(|a| protected_scopes.iter().copied().any(|b| scopes_overlap(*a, b))) {
        return Err("target and protected scopes overlap".into());
    }

    let params = required(edit, "parameters", "edit_contract")?;
    let scale_parameters = if matches!(operation.as_str(), "velocity" | "duration") {
        let numerator = integer_field(params, "scale_numerator", "parameters")?;
        let denominator = integer_field(params, "scale_denominator", "parameters")?;
        if !(1..=1024).contains(&numerator)
            || !(1..=1024).contains(&denominator)
            || numerator == denominator
        {
            return Err("edit scale numerator/denominator must be in 1..=1024 and differ".into());
        }
        Some((numerator, denominator))
    } else {
        None
    };
    let pitch_shift = if operation == "pitch" {
        let semitones = integer_field(params, "semitones", "parameters")?;
        if !(-127..=127).contains(&semitones) || semitones == 0 {
            return Err("pitch edit semitone shift must be non-zero and in -127..=127".into());
        }
        Some(semitones)
    } else {
        None
    };
    let evaluation = required(&case, "evaluation", "case")?;
    let tolerances = required(evaluation, "tolerances", "evaluation")?;
    let velocity_tolerance = required(tolerances, "velocity_0_to_127", "tolerances")?
        .as_u64().filter(|v| *v <= 127)
        .ok_or("velocity_0_to_127 must be an integer in 0..=127")? as f32 / 127.0;
    let minimum_coverage = required(tolerances, "minimum_correspondence_coverage", "tolerances")?
        .as_f64().filter(|v| (0.0..=1.0).contains(v))
        .ok_or("minimum_correspondence_coverage must be in [0,1]")?;

    let target = required(&case, "target", "case")?;
    let target_kind = string_field(target, "target_kind", "target")?;
    if !matches!(target_kind.as_str(), "exact_score_and_audio" | "score_target_audio_variable") {
        return Err(format!("abstain: target kind {target_kind} is outside symbolic-score evaluation"));
    }

    // Use only explicit, digest-scoped event correspondence. Never infer event
    // identity from nearest pitch or onset.
    let mappings = required(target, "event_correspondence", "target")?
        .as_array().ok_or("event_correspondence must be an array")?;
    let mut source_to_target = vec![None; source.notes.len()];
    let mut source_relation_modify = vec![None; source.notes.len()];
    let mut target_to_source = vec![None; candidate.notes.len()];
    for (index, mapping) in mappings.iter().enumerate() {
        let relation = string_field(mapping, "relation", &format!("mapping[{index}]"))?;
        if !matches!(relation.as_str(), "preserve" | "modify") {
            return Err(format!("abstain: correspondence relation {relation} is unsupported"));
        }
        let sources = required(mapping, "source_note_ordinals", "mapping")?
            .as_array().ok_or("source_note_ordinals must be an array")?;
        let targets = required(mapping, "target_note_ordinals", "mapping")?
            .as_array().ok_or("target_note_ordinals must be an array")?;
        if sources.len() != 1 || targets.len() != 1 {
            return Err("v1 symbolic scorer requires one-to-one correspondence".into());
        }
        let si = usize::try_from(sources[0].as_u64().ok_or("source ordinal must be non-negative")?)
            .map_err(|_| "source ordinal exceeds this target's addressable range")?;
        let ti = usize::try_from(targets[0].as_u64().ok_or("target ordinal must be non-negative")?)
            .map_err(|_| "target ordinal exceeds this target's addressable range")?;
        if si >= source_to_target.len() || ti >= target_to_source.len() {
            return Err("event correspondence ordinal is outside its bound Score".into());
        }
        if source_to_target[si].replace(ti).is_some() || target_to_source[ti].replace(si).is_some() {
            return Err("duplicate source or target event correspondence".into());
        }
        source_relation_modify[si] = Some(relation == "modify");
    }

    let mapped_sources = source_to_target.iter().filter(|item| item.is_some()).count();
    let mapped_targets = target_to_source.iter().filter(|item| item.is_some()).count();
    let coverage = mapped_sources as f64 / source.notes.len() as f64;
    let mut target_expected = 0usize;
    let mut target_correct = 0usize;
    let mut protected_expected = 0usize;
    let mut protected_exact = 0usize;
    let mut outside_expected = 0usize;
    let mut outside_exact = 0usize;
    let unmapped_candidates = candidate.notes.len().saturating_sub(mapped_targets);
    let mut collateral_changes = unmapped_candidates;
    let mut scope_pass = true;
    let mut edit_pass = true;

    for (si, source_note) in source.notes.iter().enumerate() {
        let in_target = in_scopes(&target_scopes, source_note, "target scope")?;
        let in_protected = in_scopes(&protected_scopes, source_note, "protected scope")?;
        if in_target && in_protected {
            return Err("source note falls inside both target and protected scopes".into());
        }
        if !source_note.part.is_assigned() {
            scope_pass = false;
        }
        let Some(ti) = source_to_target[si] else {
            if in_target {
                target_expected += 1;
            } else {
                outside_expected += 1;
                collateral_changes += 1;
                if in_protected { protected_expected += 1; }
            }
            edit_pass = false;
            continue;
        };
        let candidate_note = &candidate.notes[ti];
        if source_relation_modify[si] != Some(in_target) {
            scope_pass = false;
        }

        if in_target {
            target_expected += 1;
            // The candidate event must also remain wholly inside its authorized
            // scope, especially for duration edits that can move a note-off.
            let candidate_scope_ok = in_scopes(&target_scopes, candidate_note, "candidate target scope")?
                && !in_scopes(&protected_scopes, candidate_note, "candidate protected scope")?;
            let operation_ok = candidate_scope_ok && match operation.as_str() {
                "velocity" => {
                    let (numerator, denominator) = scale_parameters.expect("velocity scale validated");
                    let expected_velocity = source_note.velocity * numerator as f32 / denominator as f32;
                    source_note.velocity.is_finite()
                        && candidate_note.velocity.is_finite()
                        && expected_velocity.is_finite()
                        && (0.0..=1.0).contains(&expected_velocity)
                        && (candidate_note.velocity - expected_velocity).abs() <= velocity_tolerance
                        && non_velocity_equal(source_note, candidate_note)
                }
                "duration" => {
                    let (numerator, denominator) = scale_parameters.expect("duration scale validated");
                    scaled_duration(source_note.duration, numerator, denominator)
                        .map(|expected_duration| candidate_note.duration == expected_duration)
                        .unwrap_or(false)
                        && non_duration_equal(source_note, candidate_note)
                }
                "pitch" => {
                    let semitones = pitch_shift.expect("pitch shift validated");
                    let expected_midi = source_note.pitch.midi() as i32 + semitones as i32;
                    (0..=127).contains(&expected_midi)
                        && candidate_note.pitch == Pitch::from_midi(expected_midi as u8)
                        && non_pitch_equal(source_note, candidate_note)
                }
                _ => false,
            };
            if operation_ok {
                target_correct += 1;
            } else {
                edit_pass = false;
            }
        } else {
            outside_expected += 1;
            let exact = exact_equal(source_note, candidate_note);
            if exact { outside_exact += 1; } else { collateral_changes += 1; edit_pass = false; }
            if in_protected {
                protected_expected += 1;
                if exact { protected_exact += 1; }
            }
        }
    }

    if !score_metadata_equal(&source, &candidate) {
        scope_pass = false;
        edit_pass = false;
    }
    if mapped_targets != candidate.notes.len() { edit_pass = false; }
    if !scope_pass { edit_pass = false; }

    let requested_targets = required(evaluation, "target_metrics", "evaluation")?
        .as_array().ok_or("target_metrics must be an array")?;
    let requested_preservation = required(evaluation, "preservation_metrics", "evaluation")?
        .as_array().ok_or("preservation_metrics must be an array")?;
    for required_metric in [
        "protected_event_agreement",
        "outside_scope_event_agreement",
        "collateral_change_rate",
    ] {
        if !requested_preservation.iter().any(|entry| entry.as_str() == Some(required_metric)) {
            return Err(format!("evaluation.preservation_metrics must include {required_metric}"));
        }
    }
    let primary_metric = match operation.as_str() {
        "velocity" => "velocity_accuracy",
        "pitch" => "pitch_accuracy",
        "duration" => "duration_accuracy",
        _ => unreachable!("supported operation validated above"),
    };
    let has_primary_metric = requested_targets.iter().any(|request| request.as_str() == Some(primary_metric));
    if !has_primary_metric {
        return Err(format!("evaluation.target_metrics must include {primary_metric}"));
    }
    let mut target_metrics = Vec::new();
    for request in requested_targets {
        let id = request.as_str().ok_or("target metric IDs must be strings")?;
        target_metrics.push(match id {
            "exact_event_accuracy" => metric(id, target_correct, target_expected, 1.0, false),
            "velocity_accuracy" if operation == "velocity" => metric(id, target_correct, target_expected, 1.0, false),
            "pitch_accuracy" if operation == "pitch" => metric(id, target_correct, target_expected, 1.0, false),
            "duration_accuracy" if operation == "duration" => metric(id, target_correct, target_expected, 1.0, false),
            "instruction_rubric_pass" => continue,
            _ => unavailable_metric(id),
        });
    }
    if !edit_pass {
        target_metrics.push(json!({
            "metric_id": "symbolic_edit_contract",
            "numerator": 0,
            "denominator": 1,
            "score": 0.0,
            "threshold": 1.0,
            "passed": false
        }));
    }

    let expected_score = optional_hash(target, "expected_score_sha256", "target")?;
    if let Some(expected) = expected_score {
        let pass = expected == candidate_digest;
        target_metrics.push(json!({
            "metric_id": "expected_score_digest_match",
            "numerator": if pass { 1 } else { 0 },
            "denominator": 1,
            "score": if pass { 1.0 } else { 0.0 },
            "threshold": 1.0,
            "passed": pass
        }));
    }

    let audio_digest = candidate_audio_bytes.map(sha256);
    let expected_audio = optional_hash(target, "expected_audio_sha256", "target")?;
    let acceptable_audio = required(target, "acceptable_audio_sha256", "target")?
        .as_array().ok_or("acceptable_audio_sha256 must be an array")?;
    if target_kind == "exact_score_and_audio" {
        let pass = audio_digest.as_deref().map(|hash| {
            expected_audio.as_deref() == Some(hash)
                || acceptable_audio.iter().any(|entry| entry.as_str() == Some(hash))
        }).unwrap_or(false);
        target_metrics.push(json!({
            "metric_id": "audio_exact_match",
            "numerator": if pass { 1 } else { 0 },
            "denominator": 1,
            "score": if pass { 1.0 } else { 0.0 },
            "threshold": 1.0,
            "passed": pass
        }));
    }

    // Instruction-language evidence is a separate conjunct, not a symbolic
    // target metric. Keep its not_run result visible without conflating it
    // with the deterministic score-space operation gate.
    let target_pass = target_metrics.iter()
        .all(|item| item.get("passed").and_then(Value::as_bool) == Some(true))
        && target_expected > 0;
    let mut preservation_metrics = Vec::new();
    for request in requested_preservation {
        let id = request.as_str().ok_or("preservation metric IDs must be strings")?;
        preservation_metrics.push(match id {
            "protected_event_agreement" => metric(id, protected_exact, protected_expected, 1.0, false),
            "outside_scope_event_agreement" => metric(id, outside_exact, outside_expected, 1.0, false),
            "collateral_change_rate" => metric(
                id,
                collateral_changes,
                outside_expected.saturating_add(unmapped_candidates).max(1),
                0.0,
                true,
            ),
            _ => unavailable_metric(id),
        });
    }
    let preservation_pass = preservation_metrics.iter()
        .all(|item| item.get("passed").and_then(Value::as_bool) == Some(true));
    let correspondence_required = required(evaluation, "correspondence_required", "evaluation")?
        .as_bool().ok_or("correspondence_required must be boolean")?;
    if !correspondence_required {
        return Err("abstain: v1 symbolic scoring requires explicit event correspondence".into());
    }
    let correspondence_pass = mapped_sources == source.notes.len()
        && mapped_targets == candidate.notes.len()
        && coverage >= minimum_coverage;
    let human = required(evaluation, "human_evaluation", "evaluation")?;
    let human_required = required(human, "required", "human_evaluation")?
        .as_bool().ok_or("human_evaluation.required must be boolean")?;

    let mut result = json!({
        "schema": RESULT_SCHEMA,
        "case_id": case_id,
        "system": {
            "system_id": "muse-symbolic-edit-evaluator",
            "version": "0.3.0",
            "track": "symbolic_editor",
            "compute_budget_ms": started.elapsed().as_millis().min(u64::MAX as u128) as u64,
            "candidate_budget": 1
        },
        "artifacts": {
            "case_sha256": case_digest,
            "case_digest_schema": CASE_DIGEST_SCHEMA,
            "source_score_sha256": source_digest,
            "score_digest_schema": SCORE_DIGEST_SCHEMA,
            "candidate_audio_sha256": audio_digest,
            "candidate_score_sha256": candidate_digest,
            "render_receipt_sha256": null,
            "evaluation_receipt_sha256": null
        },
        "integrity": {
            "verdict": "not_run",
            "verifier_id": null,
            "verifier_version": null,
            "receipt_sha256": null
        },
        "target": target_metrics,
        "preservation": preservation_metrics,
        "correspondence": {
            "expected_source_events": source.notes.len(),
            "mapped_source_events": mapped_sources,
            "unresolved_source_events": source.notes.len().saturating_sub(mapped_sources),
            "coverage": coverage,
            "minimum_coverage_passed": correspondence_pass
        },
        "instruction_compatibility": {
            "verdict": "not_run",
            "evaluator_id": null,
            "evaluator_version": null,
            "receipt_sha256": null
        },
        "human_evaluation": {
            "status": if human_required { "pending" } else { "not_required" },
            "rater_count": 0,
            "questions": []
        },
        "outcome": {
            "target_pass": target_pass,
            "preservation_pass": preservation_pass,
            "correspondence_pass": correspondence_pass,
            "integrity_pass": false,
            "strict_sample_exact_match": false,
            "human_gate_pass": if human_required { Value::Null } else { Value::Bool(true) },
            "abstention_reason": "Natural-language instruction compatibility, render integrity and perceptual evaluation are outside this symbolic scorer."
        }
    });
    let seal = seal_digest(&result)?;
    result["artifacts"]["evaluation_receipt_sha256"] = Value::String(seal);
    verify_result_seal(&result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{harmony::Key, pitch::PitchClass};

    fn note(part: u16, beat: i64, velocity: f32) -> ScoreNote {
        ScoreNote {
            part: PartId(part),
            pitch: Pitch::from_midi(60 + beat as u8),
            onset: Duration::new(beat, 1),
            duration: Duration::quarter(),
            velocity,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score(candidate: bool) -> Score {
        let mut value = Score::new(Key::major(PitchClass::C), 120.0, 4);
        value.push(note(10, 0, if candidate { 0.8 * 8.0 / 9.0 } else { 0.8 }));
        value.push(note(10, 1, if candidate { 0.7 * 8.0 / 9.0 } else { 0.7 }));
        value.push(note(20, 0, 0.5));
        value
    }

    fn pitch_edited_score() -> Score {
        let mut value = score(false);
        for note in value.notes.iter_mut().take(2) {
            note.pitch = note.pitch.transpose(2);
        }
        value
    }

    fn case_for(source: &Score) -> Value {
        json!({
            "schema": CASE_SCHEMA,
            "case_id": "synthetic.velocity.part10.001",
            "dataset_version": "test",
            "split": "development",
            "source_work_id": "work-1",
            "rights": {"basis": "synthetic_owned", "license_id": "test", "derivative_policy": "private_evaluation"},
            "source": {
                "audio": {
                    "sha256": "1111111111111111111111111111111111111111111111111111111111111111",
                    "duration_ms": 1000, "sample_rate_hz": 44100, "channels": 2, "encoding": "wav_pcm16"
                },
                "score_sha256": score_sha256(source).unwrap(),
                "stems": []
            },
            "instruction": {"text": "Make part 10 softer.", "locale": "en", "operation_family": "velocity", "strength": "moderate"},
            "edit_contract": {
                "target_scopes": [{"part_id": 10, "start": {"numerator": 0, "denominator": 1}, "end": {"numerator": 2, "denominator": 1}}],
                "protected_scopes": [{"part_id": 20, "start": {"numerator": 0, "denominator": 1}, "end": {"numerator": 2, "denominator": 1}}],
                "parameters": {"scale_numerator": 8, "scale_denominator": 9}
            },
            "target": {
                "target_kind": "score_target_audio_variable",
                "expected_score_sha256": null, "expected_audio_sha256": null, "acceptable_audio_sha256": [],
                "event_correspondence": [
                    {"source_note_ordinals": [0], "target_note_ordinals": [0], "relation": "modify"},
                    {"source_note_ordinals": [1], "target_note_ordinals": [1], "relation": "modify"},
                    {"source_note_ordinals": [2], "target_note_ordinals": [2], "relation": "preserve"}
                ]
            },
            "evaluation": {
                "target_metrics": ["exact_event_accuracy", "velocity_accuracy", "instruction_rubric_pass"],
                "preservation_metrics": ["protected_event_agreement", "outside_scope_event_agreement", "collateral_change_rate"],
                "correspondence_required": true,
                "tolerances": {"onset_ms": 10, "pitch_cents": 25, "velocity_0_to_127": 0, "minimum_correspondence_coverage": 1.0},
                "human_evaluation": {"required": true, "raters_per_item": 3, "questions": ["edit_perceptibility", "instruction_recognition", "musical_validity"]}
            }
        })
    }

    fn bytes<T: serde::Serialize>(value: &T) -> Vec<u8> {
        serde_json::to_vec(value).unwrap()
    }

    fn reverse_object_keys(value: &Value) -> Value {
        match value {
            Value::Object(object) => {
                let mut keys: Vec<&String> = object.keys().collect();
                keys.sort_unstable_by(|left, right| right.cmp(left));
                let mut reversed = serde_json::Map::new();
                for key in keys {
                    reversed.insert(key.clone(), reverse_object_keys(object.get(key).unwrap()));
                }
                Value::Object(reversed)
            }
            Value::Array(items) => Value::Array(items.iter().map(reverse_object_keys).collect()),
            _ => value.clone(),
        }
    }


    #[test]
    fn case_digest_is_independent_of_json_object_key_order() {
        let first: Value = serde_json::from_str(r#"{"z":1,"a":{"y":2,"b":3},"items":[{"d":4,"c":5}]}"#).unwrap();
        let second: Value = serde_json::from_str(r#"{"items":[{"c":5,"d":4}],"a":{"b":3,"y":2},"z":1}"#).unwrap();
        assert_eq!(case_sha256(&first).unwrap(), case_sha256(&second).unwrap());
    }

    #[test]
    fn velocity_edit_passes_symbolic_gates_but_strict_emr_remains_false() {
        let source = score(false);
        let candidate = score(true);
        let result = evaluate_symbolic_case(&bytes(&case_for(&source)), &bytes(&source), &bytes(&candidate), None).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert!(result["outcome"]["correspondence_pass"].as_bool().unwrap());
        assert!(!result["outcome"]["strict_sample_exact_match"].as_bool().unwrap());
        assert_eq!(result["instruction_compatibility"]["verdict"], "not_run");
        assert_eq!(result["integrity"]["verdict"], "not_run");
        verify_result_seal(&result).unwrap();
        let reordered = reverse_object_keys(&result);
        verify_result_seal(&reordered).unwrap();
        let mut tampered = result.clone();
        tampered["artifacts"]["case_sha256"] = json!("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        assert!(verify_result_seal(&tampered).is_err());
    }

    #[test]
    fn wrong_velocity_fails_target_gate() {
        let source = score(false);
        let mut candidate = score(true);
        candidate.notes[0].velocity = 0.99;
        let result = evaluate_symbolic_case(&bytes(&case_for(&source)), &bytes(&source), &bytes(&candidate), None).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
    }

    #[test]
    fn protected_note_mutation_fails_preservation_gate() {
        let source = score(false);
        let mut candidate = score(true);
        candidate.notes[2].velocity = 0.51;
        let result = evaluate_symbolic_case(&bytes(&case_for(&source)), &bytes(&source), &bytes(&candidate), None).unwrap();
        assert!(!result["outcome"]["preservation_pass"].as_bool().unwrap());
    }

    #[test]
    fn duplicate_correspondence_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["target"]["event_correspondence"][1]["source_note_ordinals"] = json!([0]);
        let error = evaluate_symbolic_case(&bytes(&case), &bytes(&source), &bytes(&candidate), None).unwrap_err();
        assert!(error.contains("duplicate source"));
    }

    #[test]
    fn stale_source_digest_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["source"]["score_sha256"] = json!("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        let error = evaluate_symbolic_case(&bytes(&case), &bytes(&source), &bytes(&candidate), None).unwrap_err();
        assert!(error.contains("source Score digest mismatch"));
    }

    #[test]
    fn unknown_case_wire_field_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["unrecognized_policy"] = json!("silently ignore me");
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("unsupported field"));
    }

    #[test]
    fn empty_preservation_metric_list_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["evaluation"]["preservation_metrics"] = json!([]);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("preservation_metrics must not be empty"));
    }

    #[test]
    fn omitted_protected_metric_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["evaluation"]["preservation_metrics"] = json!([
            "outside_scope_event_agreement",
            "collateral_change_rate"
        ]);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("must include protected_event_agreement"));
    }

    #[test]
    fn instruction_only_target_metric_list_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["evaluation"]["target_metrics"] = json!(["instruction_rubric_pass"]);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("must include exact_event_accuracy"));
    }

    #[test]
    fn midi_pitch_outside_legal_range_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut source_value = serde_json::to_value(&source).unwrap();
        source_value["notes"][0]["pitch"]["midi"] = json!(200);
        let error = evaluate_symbolic_case(
            &bytes(&case_for(&source)),
            &bytes(&source_value),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("invalid MIDI pitch, rational timing or dynamics") || error.contains("invalid rational timing or dynamics"));
    }

    #[test]
    fn unknown_score_wire_field_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut score_value = serde_json::to_value(&source).unwrap();
        score_value["notes"][0]["future_midi_control"] = json!(7);
        let error = evaluate_symbolic_case(
            &bytes(&case_for(&source)),
            &bytes(&score_value),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("unsupported field"));
    }

    #[test]
    fn malformed_rational_score_is_rejected() {
        let source = score(false);
        let candidate = score(true);
        let mut malformed = serde_json::to_value(&source).unwrap();
        malformed["notes"][0]["onset"]["den"] = json!(0);
        let error = evaluate_symbolic_case(
            &bytes(&case_for(&source)),
            &bytes(&malformed),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("invalid rational timing"));
    }

    #[test]
    fn no_op_scale_is_rejected() {
        let source = score(false);
        let mut case = case_for(&source);
        case["edit_contract"]["parameters"]["scale_numerator"] = json!(1);
        case["edit_contract"]["parameters"]["scale_denominator"] = json!(1);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&source),
            None,
        ).unwrap_err();
        assert!(error.contains("different from 1"));
    }

    #[test]
    fn wrong_direction_scale_fails_target_gate() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["edit_contract"]["parameters"]["scale_numerator"] = json!(9);
        case["edit_contract"]["parameters"]["scale_denominator"] = json!(8);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
    }

    #[test]
    fn empty_protected_scope_does_not_vacuously_pass() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["edit_contract"]["protected_scopes"] = json!([{
            "part_id": 99,
            "start": {"numerator": 0, "denominator": 1},
            "end": {"numerator": 2, "denominator": 1}
        }]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert_eq!(result["preservation"][0]["score"], Value::Null);
    }

    #[test]
    fn inserted_candidate_note_counts_as_collateral_change() {
        let source = score(false);
        let mut candidate = score(true);
        candidate.push(note(30, 2, 0.4));
        let result = evaluate_symbolic_case(
            &bytes(&case_for(&source)),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(!result["outcome"]["preservation_pass"].as_bool().unwrap());
    }

    #[test]
    fn duration_edit_crossing_target_boundary_is_rejected() {
        let mut source = score(false);
        source.notes[0].duration = Duration::new(3, 2);
        source.notes[1].duration = Duration::quarter();
        source.total_beats = Duration::new(2, 1);
        let mut candidate = source.clone();
        candidate.notes[0].duration = Duration::new(3, 4);
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("duration");
        case["edit_contract"]["target_scopes"][0]["end"] = json!({"numerator": 1, "denominator": 1});
        case["edit_contract"]["parameters"] = json!({"scale_numerator": 1, "scale_denominator": 2});
        case["evaluation"]["target_metrics"] = json!(["duration_accuracy"]);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("crosses an authorized time-scope boundary"));
    }

    #[test]
    fn exact_duration_scaling_passes_symbolic_gates() {
        let source = score(false);
        let mut candidate = source.clone();
        for note in candidate.notes.iter_mut().take(2) {
            note.duration = Duration::new(1, 2);
        }
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("duration");
        case["instruction"]["text"] = json!("Make the part 10 notes half as long.");
        case["edit_contract"]["parameters"] = json!({"scale_numerator": 1, "scale_denominator": 2});
        case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "duration_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert!(result["outcome"]["correspondence_pass"].as_bool().unwrap());
        assert!(!result["outcome"]["strict_sample_exact_match"].as_bool().unwrap());
    }

    #[test]
    fn inverse_duration_edit_restores_original_score_digest() {
        let original = score(false);
        let mut shortened = original.clone();
        for note in shortened.notes.iter_mut().take(2) {
            note.duration = Duration::new(1, 2);
        }
        let mut inverse_case = case_for(&shortened);
        inverse_case["case_id"] = json!("synthetic.duration.inverse.001");
        inverse_case["instruction"]["operation_family"] = json!("duration");
        inverse_case["instruction"]["text"] = json!("Restore the target notes to their original length.");
        inverse_case["edit_contract"]["parameters"] = json!({"scale_numerator": 2, "scale_denominator": 1});
        inverse_case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "duration_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&inverse_case),
            &bytes(&shortened),
            &bytes(&original),
            None,
        ).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert_eq!(
            result["artifacts"]["candidate_score_sha256"],
            json!(score_sha256(&original).unwrap())
        );
    }

    #[test]
    fn expanded_duration_crossing_target_boundary_is_rejected() {
        let mut source = score(false);
        source.notes[0].duration = Duration::new(1, 2);
        source.total_beats = Duration::new(2, 1);
        let mut candidate = source.clone();
        candidate.notes[0].duration = Duration::new(3, 2);
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("duration");
        case["edit_contract"]["target_scopes"][0]["end"] = json!({"numerator": 1, "denominator": 1});
        case["edit_contract"]["parameters"] = json!({"scale_numerator": 3, "scale_denominator": 1});
        case["evaluation"]["target_metrics"] = json!(["duration_accuracy"]);
        let error = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap_err();
        assert!(error.contains("candidate target scope") || error.contains("crosses an authorized time-scope boundary"));
    }

    #[test]
    fn incorrect_duration_scale_fails_target_gate() {
        let source = score(false);
        let mut candidate = source.clone();
        for note in candidate.notes.iter_mut().take(2) {
            note.duration = Duration::new(1, 2);
        }
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("duration");
        case["edit_contract"]["parameters"] = json!({"scale_numerator": 2, "scale_denominator": 1});
        case["evaluation"]["target_metrics"] = json!(["duration_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
    }

    #[test]
    fn exact_pitch_transposition_passes_symbolic_gates() {
        let source = score(false);
        let candidate = pitch_edited_score();
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("pitch");
        case["instruction"]["text"] = json!("Raise part 10 by two semitones.");
        case["edit_contract"]["parameters"] = json!({"semitones": 2});
        case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "pitch_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert!(result["outcome"]["correspondence_pass"].as_bool().unwrap());
        assert!(!result["outcome"]["strict_sample_exact_match"].as_bool().unwrap());
    }

    #[test]
    fn pitch_edit_in_wrong_direction_fails_target_gate() {
        let source = score(false);
        let candidate = pitch_edited_score();
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("pitch");
        case["edit_contract"]["parameters"] = json!({"semitones": -2});
        case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "pitch_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
    }

    #[test]
    fn pitch_transposition_beyond_midi_range_fails_target_gate() {
        let mut source = score(false);
        source.notes[0].pitch = Pitch::from_midi(126);
        let mut candidate = source.clone();
        candidate.notes[0].pitch = Pitch::from_midi(127); // Saturation is not a valid +2 edit.
        candidate.notes[1].pitch = candidate.notes[1].pitch.transpose(2);
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("pitch");
        case["edit_contract"]["parameters"] = json!({"semitones": 2});
        case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "pitch_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(!result["outcome"]["target_pass"].as_bool().unwrap());
    }

    #[test]
    fn inverse_pitch_edit_restores_original_score() {
        let original = score(false);
        let transposed = pitch_edited_score();
        let mut inverse_case = case_for(&transposed);
        inverse_case["case_id"] = json!("synthetic.pitch.inverse.001");
        inverse_case["instruction"]["operation_family"] = json!("pitch");
        inverse_case["instruction"]["text"] = json!("Lower part 10 by two semitones.");
        inverse_case["edit_contract"]["parameters"] = json!({"semitones": -2});
        inverse_case["evaluation"]["target_metrics"] = json!(["exact_event_accuracy", "pitch_accuracy"]);
        let result = evaluate_symbolic_case(
            &bytes(&inverse_case),
            &bytes(&transposed),
            &bytes(&original),
            None,
        ).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert_eq!(
            result["artifacts"]["candidate_score_sha256"],
            json!(score_sha256(&original).unwrap())
        );
    }

    #[test]
    fn explicit_event_map_supports_candidate_note_reordering() {
        let source = score(false);
        let mut candidate = score(true);
        candidate.notes.swap(0, 2);
        let mut case = case_for(&source);
        case["target"]["event_correspondence"] = json!([
            {"source_note_ordinals": [0], "target_note_ordinals": [2], "relation": "modify"},
            {"source_note_ordinals": [1], "target_note_ordinals": [1], "relation": "modify"},
            {"source_note_ordinals": [2], "target_note_ordinals": [0], "relation": "preserve"}
        ]);
        let result = evaluate_symbolic_case(
            &bytes(&case),
            &bytes(&source),
            &bytes(&candidate),
            None,
        ).unwrap();
        assert!(result["outcome"]["target_pass"].as_bool().unwrap());
        assert!(result["outcome"]["preservation_pass"].as_bool().unwrap());
        assert!(result["outcome"]["correspondence_pass"].as_bool().unwrap());
    }

    #[test]
    fn unsupported_edit_family_abstains() {
        let source = score(false);
        let candidate = score(true);
        let mut case = case_for(&source);
        case["instruction"]["operation_family"] = json!("harmony");
        let error = evaluate_symbolic_case(&bytes(&case), &bytes(&source), &bytes(&candidate), None).unwrap_err();
        assert!(error.contains("abstain"));
    }
}

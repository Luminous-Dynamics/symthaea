//! Independent integrity verification for MUSE score-to-render receipts.
//!
//! This tool checks byte/link integrity and MIDI/sidecar consistency. It does
//! not decide whether an edit should be accepted or whether listeners find it
//! musical. Execution identity and signature trust remain separate.

use midly::{Format, MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::Path;

pub const RECEIPT_SCHEMA_V1: &str = "muse-render-execution-receipt-v1";
pub const RECEIPT_DIGEST_RECIPE_V1: &str = "muse-render-execution-receipt-jcs-sha256-v1";
pub const RECEIPT_DIGEST_DOMAIN_V1: &[u8] = b"luminous-dynamics-muse-render-execution-receipt-v1";
pub const MIDI_MANIFEST_SCHEMA_V1: &str = "muse-midi-performance-v1";
pub const MIDI_MANIFEST_DIGEST_SCHEMA_V1: &str = "muse-midi-performance-manifest-serde-json-sha256-v1";
pub const MIDI_SCORE_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";
pub const MIDI_TIMELINE_DIGEST_SCHEMA_V1: &str = "muse-orchestration-timeline-serde-json-sha256-v1";
pub const MIDI_BINDING_DIGEST_SCHEMA_V1: &str = "muse-midi-instrument-bindings-serde-json-sha256-v1";
pub const MIDI_CAPABILITY_DIGEST_SCHEMA_V1: &str = "muse-midi-instrument-capabilities-serde-json-sha256-v1";
pub const MIDI_TICKS_PER_QUARTER_V1: u16 = 960;
pub const SCORE_EDIT_SCHEMA_V1: &str = "score-edit-plan-v1";
pub const SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReceiptV1 {
    pub schema: String,
    pub attempt_id: String,
    pub attempt_state: String,
    pub source: SourceBindingV1,
    pub edit: EditBindingV1,
    pub midi: MidiBindingV1,
    pub renderer: RendererBindingV1,
    pub execution: ExecutionBindingV1,
    pub audio: AudioBindingV1,
    pub verification: VerificationBindingV1,
    pub attestation: AttestationV1,
    pub digest_recipe: String,
    pub receipt_sha256: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceBindingV1 {
    pub source_score_sha256: String,
    pub candidate_score_sha256: String,
    pub score_digest_schema: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditBindingV1 {
    pub plan_sha256: String,
    pub edit_receipt_sha256: String,
    pub verifier_id: String,
    pub verifier_version: String,
    pub verification_receipt_sha256: String,
    pub verdict: String,
    pub authorized_operation_ids: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MidiBindingV1 {
    pub manifest_schema: String,
    pub manifest_digest_schema: String,
    pub manifest_sha256: String,
    pub score_digest_schema: String,
    pub score_sha256: String,
    pub timeline_digest_schema: String,
    pub timeline_sha256: String,
    pub binding_digest_schema: String,
    pub binding_sha256: String,
    pub capability_digest_schema: String,
    pub capability_sha256: String,
    pub midi_sha256: String,
    pub byte_length: usize,
    pub ticks_per_quarter: u16,
    pub tempo_microseconds_per_quarter: u32,
    pub meter_numerator: u8,
    pub meter_denominator: u8,
    pub note_count: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RendererBindingV1 {
    pub renderer_id: String,
    pub renderer_version: String,
    pub binary_sha256: String,
    pub environment_sha256: String,
    pub configuration_sha256: String,
    pub asset_sha256: Option<String>,
    pub fallback_used: bool,
    pub fallback_reason: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionBindingV1 {
    pub state: String,
    pub executor_id: String,
    pub input_midi_sha256: Option<String>,
    pub output_audio_sha256: Option<String>,
    pub exit_code: Option<i32>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub stdout_sha256: Option<String>,
    pub stderr_sha256: Option<String>,
    pub invocation_sha256: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioBindingV1 {
    pub disposition: String,
    pub container: String,
    pub audio_sha256: Option<String>,
    pub byte_length: Option<usize>,
    pub sample_rate_hz: Option<u32>,
    pub channels: Option<u16>,
    pub frames: Option<u64>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VerificationBindingV1 {
    pub midi_score_conformance: EvidenceResultV1,
    pub rendered_score_conformance: EvidenceResultV1,
    pub protected_audio_preservation: EvidenceResultV1,
    pub human_evaluation: EvidenceResultV1,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceResultV1 {
    pub verdict: String,
    pub verifier_id: Option<String>,
    pub verifier_version: Option<String>,
    pub receipt_sha256: Option<String>,
    pub reason: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttestationV1 {
    pub kind: String,
    pub identity: String,
    pub key_id: Option<String>,
    pub signature: Option<String>,
}

/// Mirrors the repository's PartId-aware MIDI performance manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MidiPerformanceManifestV1 {
    pub schema: String,
    pub score_digest_schema: String,
    pub score_sha256: String,
    pub timeline_digest_schema: String,
    pub timeline_sha256: String,
    pub binding_digest_schema: String,
    pub binding_sha256: String,
    pub capability_digest_schema: String,
    pub capability_sha256: String,
    pub midi_sha256: String,
    pub ticks_per_quarter: u16,
    pub tempo_microseconds_per_quarter: u32,
    pub meter_numerator: u8,
    pub meter_denominator: u8,
    pub tracks: Vec<MidiPerformanceTrackReceiptV1>,
    pub notes: Vec<MidiPerformanceNoteReceiptV1>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MidiPerformanceTrackReceiptV1 {
    pub track_index: u16,
    pub track_name: String,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub renderer_id: String,
    pub patch_id: String,
    pub gm_program: u8,
    pub midi_channel: u8,
    pub segment_start_tick: u64,
    pub segment_end_tick: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MidiPerformanceNoteReceiptV1 {
    pub source_note_ordinal: usize,
    pub part_id: u16,
    pub segment_id: String,
    pub instrument_id: String,
    pub track_index: u16,
    pub start_tick: u64,
    pub end_tick: u64,
    pub pitch_midi: u8,
    pub velocity_midi: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderInvocationV1 {
    pub schema: String,
    pub attempt_id: String,
    pub renderer_id: String,
    pub renderer_version: String,
    pub executable_sha256: String,
    pub environment_sha256: String,
    pub configuration_sha256: String,
    pub asset_sha256: Option<String>,
    pub input_midi_path: String,
    pub input_midi_sha256: String,
    pub output_audio_path: String,
    pub output_audio_sha256: String,
    pub arguments: Vec<String>,
    pub started_at: String,
    pub finished_at: String,
    pub exit_code: i32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BundleReportV1 {
    pub attempt_id: String,
    pub attempt_state: String,
    pub receipt_seal_verified: bool,
    pub bundle_integrity_verified: bool,
    pub midi_event_manifest_verified: bool,
    pub wav_integrity_verified: bool,
    pub acceptance_eligible: bool,
    pub trust_level: String,
    pub notes: Vec<String>,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn domain_sha256(schema: &[u8], bytes: &[u8]) -> String {
    let mut hash = Sha256::new();
    hash.update(schema);
    hash.update([0]);
    hash.update(bytes);
    format!("{:x}", hash.finalize())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Receipt v1 uses a project-specific canonical serialization profile:
/// typed fields contain strings, booleans, nulls, arrays and integers only;
/// serde_json's default map representation orders object keys lexicographically;
/// compact serialization is then domain-separated before hashing. This is
/// deterministic for this schema, but is not claimed as full RFC 8785 JCS.
fn canonical_receipt_bytes(value: &Value) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| format!("canonical JSON serialization failed: {e}"))
}

fn receipt_digest(value: &Value) -> Result<String, String> {
    let mut unsealed = value.clone();
    unsealed
        .as_object_mut()
        .ok_or("receipt root must be a JSON object")?
        .remove("receipt_sha256");
    Ok(domain_sha256(RECEIPT_DIGEST_DOMAIN_V1, &canonical_receipt_bytes(&unsealed)?))
}

fn require_fields(value: &Value, fields: &[&str], label: &str) -> Result<(), String> {
    let object = value.as_object().ok_or_else(|| format!("{label} must be an object"))?;
    for field in fields {
        if !object.contains_key(*field) {
            return Err(format!("{label} is missing required field {field}"));
        }
    }
    Ok(())
}

fn validate_receipt_shape(value: &Value) -> Result<(), String> {
    require_fields(value, &["schema","attempt_id","attempt_state","source","edit","midi","renderer","execution","audio","verification","attestation","digest_recipe","receipt_sha256"], "receipt")?;
    let source = &value["source"];
    require_fields(source, &["source_score_sha256","candidate_score_sha256","score_digest_schema"], "source")?;
    let edit = &value["edit"];
    require_fields(edit, &["plan_sha256","edit_receipt_sha256","verifier_id","verifier_version","verification_receipt_sha256","verdict","authorized_operation_ids"], "edit")?;
    let midi = &value["midi"];
    require_fields(midi, &["manifest_schema","manifest_digest_schema","manifest_sha256","score_digest_schema","score_sha256","timeline_digest_schema","timeline_sha256","binding_digest_schema","binding_sha256","capability_digest_schema","capability_sha256","midi_sha256","byte_length","ticks_per_quarter","tempo_microseconds_per_quarter","meter_numerator","meter_denominator","note_count"], "midi")?;
    let renderer = &value["renderer"];
    require_fields(renderer, &["renderer_id","renderer_version","binary_sha256","environment_sha256","configuration_sha256","asset_sha256","fallback_used","fallback_reason"], "renderer")?;
    let execution = &value["execution"];
    require_fields(execution, &["state","executor_id","input_midi_sha256","output_audio_sha256","exit_code","started_at","finished_at","stdout_sha256","stderr_sha256","invocation_sha256"], "execution")?;
    let audio = &value["audio"];
    require_fields(audio, &["disposition","container","audio_sha256","byte_length","sample_rate_hz","channels","frames"], "audio")?;
    let verification = &value["verification"];
    require_fields(verification, &["midi_score_conformance","rendered_score_conformance","protected_audio_preservation","human_evaluation"], "verification")?;
    for name in ["midi_score_conformance","rendered_score_conformance","protected_audio_preservation","human_evaluation"] {
        require_fields(&verification[name], &["verdict","verifier_id","verifier_version","receipt_sha256","reason"], name)?;
    }
    require_fields(&value["attestation"], &["kind","identity","key_id","signature"], "attestation")?;
    Ok(())
}

pub fn verify_receipt_bytes(bytes: &[u8]) -> Result<ReceiptV1, String> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| format!("invalid receipt JSON: {e}"))?;
    validate_receipt_shape(&value)?;
    let receipt: ReceiptV1 = serde_json::from_value(value.clone())
        .map_err(|e| format!("receipt shape invalid: {e}"))?;
    validate_receipt(&receipt)?;
    let actual = receipt_digest(&value)?;
    if actual != receipt.receipt_sha256 {
        return Err(format!("receipt seal mismatch: recorded {}, recomputed {actual}", receipt.receipt_sha256));
    }
    Ok(receipt)
}

pub fn validate_receipt(r: &ReceiptV1) -> Result<(), String> {
    if r.schema != RECEIPT_SCHEMA_V1 || r.digest_recipe != RECEIPT_DIGEST_RECIPE_V1 {
        return Err("unsupported receipt schema or digest recipe".into());
    }
    if r.attempt_id.trim().is_empty() || !valid_sha256(&r.receipt_sha256) {
        return Err("attempt ID or receipt digest is invalid".into());
    }
    let required = [
        r.source.source_score_sha256.as_str(), r.source.candidate_score_sha256.as_str(),
        r.edit.plan_sha256.as_str(), r.edit.edit_receipt_sha256.as_str(),
        r.edit.verification_receipt_sha256.as_str(), r.midi.manifest_sha256.as_str(),
        r.midi.score_sha256.as_str(), r.midi.timeline_sha256.as_str(),
        r.midi.binding_sha256.as_str(), r.midi.capability_sha256.as_str(),
        r.midi.midi_sha256.as_str(), r.renderer.binary_sha256.as_str(),
        r.renderer.environment_sha256.as_str(), r.renderer.configuration_sha256.as_str(),
    ];
    if required.into_iter().any(|hash| !valid_sha256(hash)) {
        return Err("one or more required SHA-256 digests are invalid".into());
    }
    let optional = [
        r.renderer.asset_sha256.as_deref(), r.execution.input_midi_sha256.as_deref(),
        r.execution.output_audio_sha256.as_deref(), r.execution.stdout_sha256.as_deref(),
        r.execution.stderr_sha256.as_deref(), r.execution.invocation_sha256.as_deref(),
        r.audio.audio_sha256.as_deref(),
        r.verification.midi_score_conformance.receipt_sha256.as_deref(),
        r.verification.rendered_score_conformance.receipt_sha256.as_deref(),
        r.verification.protected_audio_preservation.receipt_sha256.as_deref(),
        r.verification.human_evaluation.receipt_sha256.as_deref(),
    ];
    if optional.into_iter().flatten().any(|hash| !valid_sha256(hash)) {
        return Err("one or more optional SHA-256 digests are invalid".into());
    }
    if r.source.score_digest_schema != MIDI_SCORE_DIGEST_SCHEMA_V1
        || r.midi.manifest_schema != MIDI_MANIFEST_SCHEMA_V1
        || r.midi.manifest_digest_schema != MIDI_MANIFEST_DIGEST_SCHEMA_V1
        || r.midi.score_digest_schema != MIDI_SCORE_DIGEST_SCHEMA_V1
        || r.midi.timeline_digest_schema != MIDI_TIMELINE_DIGEST_SCHEMA_V1
        || r.midi.binding_digest_schema != MIDI_BINDING_DIGEST_SCHEMA_V1
        || r.midi.capability_digest_schema != MIDI_CAPABILITY_DIGEST_SCHEMA_V1
    {
        return Err("unsupported Score/MIDI digest schema".into());
    }
    if r.source.candidate_score_sha256 != r.midi.score_sha256 {
        return Err("candidate Score digest does not match MIDI manifest Score digest".into());
    }
    if r.midi.ticks_per_quarter != MIDI_TICKS_PER_QUARTER_V1
        || r.midi.tempo_microseconds_per_quarter == 0 || r.midi.meter_numerator == 0
        || r.midi.meter_denominator != 4 || r.midi.note_count == 0 || r.midi.byte_length == 0
    {
        return Err("MIDI timing, meter, note count, or byte length is invalid".into());
    }
    if r.edit.verdict != "pass" || r.edit.verifier_id.trim().is_empty()
        || r.edit.verifier_version.trim().is_empty() || r.edit.authorized_operation_ids.is_empty()
    {
        return Err("edit lacks a passing independent verifier or operation IDs".into());
    }
    if r.renderer.renderer_id.trim().is_empty() || r.renderer.renderer_version.trim().is_empty() {
        return Err("renderer identity/version is empty".into());
    }
    if r.renderer.fallback_used && r.renderer.fallback_reason.as_deref().unwrap_or("").trim().is_empty() {
        return Err("renderer fallback requires an explicit reason".into());
    }
    if !matches!(r.attestation.kind.as_str(), "unsigned_local" | "workflow_observed" | "signed")
        || r.attestation.identity.trim().is_empty()
    {
        return Err("attestation kind or identity is invalid".into());
    }
    if r.attestation.kind == "signed"
        && (r.attestation.key_id.as_deref().unwrap_or("").is_empty()
            || r.attestation.signature.as_deref().unwrap_or("").is_empty())
    {
        return Err("signed attestation requires key_id and signature".into());
    }
    for evidence in [
        &r.verification.midi_score_conformance, &r.verification.rendered_score_conformance,
        &r.verification.protected_audio_preservation, &r.verification.human_evaluation,
    ] {
        if !matches!(evidence.verdict.as_str(), "pass" | "fail" | "abstain" | "not_run") {
            return Err("verification channel has an invalid verdict".into());
        }
    }

    match r.attempt_state.as_str() {
        "not_submitted" => {
            if r.execution.state != "not_started" || r.execution.input_midi_sha256.is_some()
                || r.execution.output_audio_sha256.is_some() || r.execution.exit_code.is_some()
                || r.audio.disposition != "not_produced" || r.audio.audio_sha256.is_some()
            { return Err("not_submitted state contains execution/output fields".into()); }
        }
        "submitted" => {
            if !matches!(r.execution.state.as_str(), "submitted" | "running")
                || r.execution.output_audio_sha256.is_some()
                || matches!(r.audio.disposition.as_str(), "produced" | "partial")
            { return Err("submitted state contains terminal execution/output evidence".into()); }
        }
        "completed" => {
            if r.execution.state != "completed" || r.execution.exit_code != Some(0)
                || r.execution.input_midi_sha256.as_deref() != Some(r.midi.midi_sha256.as_str())
                || r.execution.output_audio_sha256.as_deref() != r.audio.audio_sha256.as_deref()
                || r.audio.disposition != "produced" || r.audio.container != "wav"
                || r.audio.audio_sha256.is_none() || r.audio.byte_length.unwrap_or(0) == 0
                || r.audio.sample_rate_hz.unwrap_or(0) == 0 || r.audio.channels.unwrap_or(0) == 0
                || r.audio.frames.unwrap_or(0) == 0 || r.execution.finished_at.is_none()
                || r.execution.started_at.is_none() || r.execution.invocation_sha256.is_none()
                || r.execution.executor_id.trim().is_empty()
            { return Err("completed state lacks reconciled execution or WAV evidence".into()); }
        }
        "failed" => {
            if r.execution.state != "failed" || r.execution.finished_at.is_none()
                || r.audio.disposition == "produced"
            { return Err("failed state lacks a reconciled terminal failure".into()); }
        }
        "outcome_unknown" => {
            if r.execution.state != "unknown"
                || r.verification.rendered_score_conformance.verdict == "pass"
                || r.verification.protected_audio_preservation.verdict == "pass"
            { return Err("unknown outcome cannot claim completed render/preservation evidence".into()); }
        }
        other => return Err(format!("unsupported attempt state: {other}")),
    }
    Ok(())
}

fn read(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("could not read {label} {}: {e}", path.display()))
}
fn check_file(path: &Path, expected: &str, label: &str) -> Result<(), String> {
    let mut file = fs::File::open(path)
        .map_err(|e| format!("could not open {label} {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)
            .map_err(|e| format!("could not hash {label} {}: {e}", path.display()))?;
        if count == 0 { break; }
        hasher.update(&buffer[..count]);
    }
    let actual = format!("{:x}", hasher.finalize());
    if actual != expected { return Err(format!("{label} SHA-256 mismatch: expected {expected}, observed {actual}")); }
    Ok(())
}
fn json_file(path: &Path, label: &str) -> Result<Value, String> {
    serde_json::from_slice(&read(path, label)?).map_err(|e| format!("{label} JSON invalid: {e}"))
}
fn string_field<'a>(v: &'a Value, field: &str, label: &str) -> Result<&'a str, String> {
    v.get(field).and_then(Value::as_str).ok_or_else(|| format!("{label}.{field} must be a string"))
}
fn string_array(v: &Value, field: &str, label: &str) -> Result<Vec<String>, String> {
    v.get(field).and_then(Value::as_array).ok_or_else(|| format!("{label}.{field} must be an array"))?
        .iter().map(|item| item.as_str().map(str::to_owned)
        .ok_or_else(|| format!("{label}.{field} contains a non-string"))).collect()
}

fn verify_edit_bundle(root: &Path, r: &ReceiptV1) -> Result<(), String> {
    let pp = root.join("edit-plan.json");
    let rp = root.join("edit-receipt.json");
    let vp = root.join("edit-verification.json");
    check_file(&pp, &r.edit.plan_sha256, "edit plan")?;
    check_file(&rp, &r.edit.edit_receipt_sha256, "edit receipt")?;
    check_file(&vp, &r.edit.verification_receipt_sha256, "edit verification")?;
    let plan = json_file(&pp, "edit plan")?;
    let edit = json_file(&rp, "edit receipt")?;
    let verification = json_file(&vp, "edit verification")?;
    if string_field(&plan, "schema", "plan")? != SCORE_EDIT_SCHEMA_V1
        || string_field(&plan, "input_digest_schema", "plan")? != SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1
        || string_field(&plan, "input_score_sha256", "plan")? != r.source.source_score_sha256
        || string_field(&edit, "schema", "edit receipt")? != SCORE_EDIT_SCHEMA_V1
        || string_field(&edit, "input_digest_schema", "edit receipt")? != SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1
        || string_field(&edit, "input_score_sha256", "edit receipt")? != r.source.source_score_sha256
        || string_field(&verification, "input_score_sha256", "verification")? != r.source.source_score_sha256
        || verification.get("verified").and_then(Value::as_bool) != Some(true)
    { return Err("edit plan/receipt/verifier source binding or verdict mismatch".into()); }

    let plan_ops = plan.get("operations").and_then(Value::as_array).ok_or("plan.operations missing")?;
    let op_ids = plan_ops.iter().map(|op| string_field(op, "id", "operation").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    let windows = plan.get("protected_windows").and_then(Value::as_array)
        .ok_or("plan.protected_windows missing")?;
    let window_ids = windows.iter().map(|w| string_field(w, "id", "protected window").map(str::to_owned))
        .collect::<Result<Vec<_>, _>>()?;
    if op_ids != string_array(&edit, "operation_ids", "edit receipt")?
        || op_ids != r.edit.authorized_operation_ids
        || window_ids != string_array(&edit, "protected_window_ids", "edit receipt")?
    { return Err("operation or protected-window IDs differ across the edit transaction".into()); }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OnKey { track: usize, channel: u8, pitch: u8, velocity: u8, tick: u64 }
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OffKey { track: usize, channel: u8, pitch: u8, tick: u64 }
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ProgramKey { track: usize, channel: u8, program: u8, tick: u64 }
fn inc<K: Ord>(m: &mut BTreeMap<K, usize>, k: K) { *m.entry(k).or_insert(0) += 1; }

pub fn verify_midi_event_manifest(m: &MidiPerformanceManifestV1, bytes: &[u8]) -> Result<(), String> {
    let smf = Smf::parse(bytes).map_err(|e| format!("MIDI parse failed: {e}"))?;
    if smf.header.format != Format::Parallel { return Err("MIDI must use Type 1 parallel format".into()); }
    match smf.header.timing {
        Timing::Metrical(ticks) if u32::from(ticks.as_int()) == u32::from(m.ticks_per_quarter) => {}
        _ => return Err("MIDI ticks-per-quarter mismatch".into()),
    }
    if smf.tracks.len() != m.tracks.len() + 1 { return Err("MIDI track count mismatch".into()); }

    let mut track_map = BTreeMap::new();
    let mut expected_names = BTreeMap::new();
    let mut expected_programs = BTreeMap::new();
    for t in &m.tracks {
        let index = usize::from(t.track_index);
        if index == 0 || index >= smf.tracks.len() || t.midi_channel > 15 || t.gm_program > 127
            || t.segment_start_tick >= t.segment_end_tick
            || t.track_name.trim().is_empty() || t.segment_id.trim().is_empty()
        { return Err("invalid track receipt".into()); }
        if track_map.insert(index, t).is_some() { return Err("duplicate MIDI track index".into()); }
        expected_names.insert(index, t.track_name.as_str());
        inc(&mut expected_programs, ProgramKey {
            track: index, channel: t.midi_channel, program: t.gm_program, tick: t.segment_start_tick,
        });
    }

    let mut expected_on = BTreeMap::new();
    let mut expected_off = BTreeMap::new();
    let mut ordinals = std::collections::BTreeSet::new();
    for n in &m.notes {
        let index = usize::from(n.track_index);
        let t = track_map.get(&index).ok_or("note references unknown track")?;
        if !ordinals.insert(n.source_note_ordinal) || n.start_tick >= n.end_tick
            || n.start_tick < t.segment_start_tick || n.end_tick > t.segment_end_tick
            || n.pitch_midi > 127 || !(1..=127).contains(&n.velocity_midi)
            || n.part_id != t.part_id || n.segment_id != t.segment_id || n.instrument_id != t.instrument_id
        { return Err("invalid or inconsistent MIDI note receipt".into()); }
        inc(&mut expected_on, OnKey { track:index, channel:t.midi_channel, pitch:n.pitch_midi, velocity:n.velocity_midi, tick:n.start_tick });
        inc(&mut expected_off, OffKey { track:index, channel:t.midi_channel, pitch:n.pitch_midi, tick:n.end_tick });
    }

    if track_map.len() != m.tracks.len() || (0..m.notes.len()).any(|ordinal| !ordinals.contains(&ordinal)) {
        return Err("MIDI manifest track or source-note ordinal coverage is incomplete".into());
    }

    let mut actual_on = BTreeMap::new();
    let mut actual_off = BTreeMap::new();
    let mut actual_programs = BTreeMap::new();
    let mut actual_names = BTreeMap::new();
    let mut tempo = None;
    let mut meter = None;
    for (index, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0_u64;
        for event in track {
            tick = tick.checked_add(u64::from(event.delta.as_int())).ok_or("MIDI tick overflow")?;
            match &event.kind {
                TrackEventKind::Meta(MetaMessage::TrackName(bytes)) => {
                    let name = std::str::from_utf8(bytes).map_err(|_| "track name is not UTF-8")?;
                    if actual_names.insert(index, name.to_owned()).is_some() { return Err("duplicate track name".into()); }
                }
                TrackEventKind::Meta(MetaMessage::Tempo(value)) => {
                    if index != 0 || tempo.replace(value.as_int()).is_some() { return Err("invalid/duplicate tempo meta event".into()); }
                }
                TrackEventKind::Meta(MetaMessage::TimeSignature(n, exponent, _, _)) => {
                    let denominator = 1_u32.checked_shl(u32::from(*exponent)).ok_or("time signature overflow")?;
                    if index != 0 || meter.replace((*n, denominator)).is_some() { return Err("invalid/duplicate meter meta event".into()); }
                }
                TrackEventKind::Midi { channel, message } => {
                    let channel = channel.as_int();
                    match message {
                        MidiMessage::ProgramChange { program } => inc(&mut actual_programs, ProgramKey {
                            track:index, channel, program:program.as_int(), tick,
                        }),
                        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => inc(&mut actual_on, OnKey {
                            track:index, channel, pitch:key.as_int(), velocity:vel.as_int(), tick,
                        }),
                        MidiMessage::NoteOn { key, .. } | MidiMessage::NoteOff { key, .. } => inc(&mut actual_off, OffKey {
                            track:index, channel, pitch:key.as_int(), tick,
                        }),
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
    if actual_names.get(&0).map(String::as_str) != Some("Muse Tempo and Meter")
        || actual_names.len() != smf.tracks.len()
    { return Err("tempo/part track names missing or incomplete".into()); }
    for index in 1..smf.tracks.len() {
        if !track_map.contains_key(&index) {
            return Err(format!("MIDI manifest is missing track index {index}"));
        }
    }
    for (index, name) in expected_names {
        if actual_names.get(&index).map(String::as_str) != Some(name) { return Err(format!("track {index} name mismatch")); }
    }
    if tempo != Some(m.tempo_microseconds_per_quarter)
        || meter != Some((m.meter_numerator, u32::from(m.meter_denominator)))
    { return Err("MIDI tempo or meter does not match sidecar".into()); }
    if actual_programs != expected_programs { return Err("MIDI program changes do not match track sidecar".into()); }
    if actual_on != expected_on { return Err("MIDI note-on events do not match note sidecar".into()); }
    if actual_off != expected_off { return Err("MIDI note-off events do not match note sidecar".into()); }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MidiInstrumentBindingV1 {
    instrument_id: String,
    renderer_id: String,
    patch_id: String,
    gm_program: u8,
    asset_sha256: Option<String>,
}

fn rational_beat_to_ticks(value: &Value) -> Result<u64, String> {
    let numerator = value.get("numerator").and_then(Value::as_u64)
        .ok_or("timeline beat numerator must be a non-negative integer")?;
    let denominator = value.get("denominator").and_then(Value::as_u64)
        .filter(|value| *value > 0)
        .ok_or("timeline beat denominator must be a positive integer")?;
    let scaled = u128::from(numerator) * u128::from(MIDI_TICKS_PER_QUARTER_V1);
    if scaled % u128::from(denominator) != 0 {
        return Err("timeline beat is not exactly representable at 960 PPQ".into());
    }
    u64::try_from(scaled / u128::from(denominator))
        .map_err(|_| "timeline beat exceeds MIDI tick range".into())
}

fn verify_timeline_binding_bundle(
    root: &Path,
    receipt: &ReceiptV1,
    manifest: &MidiPerformanceManifestV1,
) -> Result<(), String> {
    let timeline_bytes = read(&root.join("orchestration-timeline.json"), "orchestration timeline")?;
    if domain_sha256(MIDI_TIMELINE_DIGEST_SCHEMA_V1.as_bytes(), &timeline_bytes) != receipt.midi.timeline_sha256 {
        return Err("orchestration timeline digest mismatch".into());
    }
    let timeline: Value = serde_json::from_slice(&timeline_bytes)
        .map_err(|e| format!("orchestration timeline JSON invalid: {e}"))?;
    if string_field(&timeline, "schema", "timeline")? != "orchestration-timeline-v1"
        || string_field(&timeline, "score_digest_schema", "timeline")? != MIDI_SCORE_DIGEST_SCHEMA_V1
        || string_field(&timeline, "score_sha256", "timeline")? != receipt.source.candidate_score_sha256
    {
        return Err("orchestration timeline schema/source Score binding mismatch".into());
    }
    let segments = timeline.get("segments").and_then(Value::as_array)
        .ok_or("timeline.segments must be an array")?;
    if segments.len() != manifest.tracks.len() {
        return Err("timeline segment count differs from MIDI track count".into());
    }
    let mut by_id = BTreeMap::new();
    for segment in segments {
        let id = string_field(segment, "id", "timeline segment")?.to_owned();
        if by_id.insert(id, segment).is_some() {
            return Err("timeline contains duplicate segment IDs".into());
        }
    }
    let mut referenced_segments = std::collections::BTreeSet::new();
    for track in &manifest.tracks {
        if !referenced_segments.insert(track.segment_id.as_str()) {
            return Err("multiple MIDI tracks reference the same timeline segment".into());
        }
        let segment = by_id.get(&track.segment_id)
            .ok_or_else(|| format!("MIDI track references missing timeline segment {}", track.segment_id))?;
        let part_id = segment.get("part_id").and_then(Value::as_u64)
            .ok_or("timeline segment part_id must be an integer")?;
        let instrument_id = string_field(segment, "instrument_id", "timeline segment")?;
        let start = rational_beat_to_ticks(segment.get("start").ok_or("timeline segment start missing")?)?;
        let end = rational_beat_to_ticks(segment.get("end").ok_or("timeline segment end missing")?)?;
        if part_id != u64::from(track.part_id) || instrument_id != track.instrument_id
            || start != track.segment_start_tick || end != track.segment_end_tick || start >= end
        {
            return Err(format!("timeline segment {} does not explain its MIDI track", track.segment_id));
        }
    }

    if referenced_segments.len() != by_id.len() {
        return Err("timeline contains an unreferenced orchestration segment".into());
    }

    let binding_bytes = read(&root.join("instrument-bindings.json"), "instrument bindings")?;
    if domain_sha256(MIDI_BINDING_DIGEST_SCHEMA_V1.as_bytes(), &binding_bytes) != receipt.midi.binding_sha256 {
        return Err("instrument-binding digest mismatch".into());
    }
    let bindings: Vec<MidiInstrumentBindingV1> = serde_json::from_slice(&binding_bytes)
        .map_err(|e| format!("instrument bindings JSON invalid: {e}"))?;
    if bindings.windows(2).any(|pair| pair[0].instrument_id >= pair[1].instrument_id) {
        return Err("instrument bindings must be unique and sorted by instrument_id".into());
    }
    let binding_map: BTreeMap<&str, &MidiInstrumentBindingV1> =
        bindings.iter().map(|binding| (binding.instrument_id.as_str(), binding)).collect();
    let used_instruments: std::collections::BTreeSet<&str> =
        manifest.tracks.iter().map(|track| track.instrument_id.as_str()).collect();
    if binding_map.len() != bindings.len() || binding_map.len() != used_instruments.len()
        || binding_map.keys().any(|id| !used_instruments.contains(id))
    {
        return Err("instrument bindings are duplicated, unexpected, or incomplete".into());
    }
    for track in &manifest.tracks {
        let binding = binding_map.get(track.instrument_id.as_str())
            .ok_or_else(|| format!("missing binding for instrument {}", track.instrument_id))?;
        if binding.renderer_id != track.renderer_id || binding.patch_id != track.patch_id
            || binding.gm_program != track.gm_program
            || binding.asset_sha256.as_deref().is_some_and(|hash| !valid_sha256(hash))
            || binding.asset_sha256.as_deref().is_some_and(|hash| Some(hash) != receipt.renderer.asset_sha256.as_deref())
        {
            return Err(format!("instrument binding does not match MIDI track or renderer asset {}", track.track_index));
        }
    }

    let capability_bytes = read(&root.join("instrument-capabilities.json"), "instrument capabilities")?;
    if domain_sha256(MIDI_CAPABILITY_DIGEST_SCHEMA_V1.as_bytes(), &capability_bytes) != receipt.midi.capability_sha256 {
        return Err("instrument-capability digest mismatch".into());
    }
    let capabilities: Value = serde_json::from_slice(&capability_bytes)
        .map_err(|e| format!("instrument capabilities JSON invalid: {e}"))?;
    let capability_array = capabilities.as_array()
        .ok_or("instrument capabilities must be a JSON array")?;
    let mut capability_ids = Vec::new();
    for capability in capability_array {
        capability_ids.push(string_field(capability, "id", "instrument capability")?.to_owned());
    }
    if capability_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err("instrument capabilities must be unique and sorted by ID".into());
    }
    for track in &manifest.tracks {
        if capability_ids.binary_search(&track.instrument_id).is_err() {
            return Err(format!("missing capability profile for instrument {}", track.instrument_id));
        }
    }
    Ok(())
}

fn verify_midi_bundle(root: &Path, r: &ReceiptV1) -> Result<(), String> {
    let bytes = read(&root.join("midi-manifest.json"), "MIDI manifest")?;
    let m: MidiPerformanceManifestV1 = serde_json::from_slice(&bytes)
        .map_err(|e| format!("MIDI manifest JSON invalid: {e}"))?;
    if m.tracks.iter().any(|track| track.renderer_id != r.renderer.renderer_id) {
        return Err("MIDI instrument binding names a different renderer identity".into());
    }
    if m.schema != MIDI_MANIFEST_SCHEMA_V1 || m.score_digest_schema != MIDI_SCORE_DIGEST_SCHEMA_V1
        || m.timeline_digest_schema != MIDI_TIMELINE_DIGEST_SCHEMA_V1
        || m.binding_digest_schema != MIDI_BINDING_DIGEST_SCHEMA_V1
        || m.capability_digest_schema != MIDI_CAPABILITY_DIGEST_SCHEMA_V1
        || m.ticks_per_quarter != MIDI_TICKS_PER_QUARTER_V1
    { return Err("unsupported MIDI manifest/digest recipe".into()); }
    verify_timeline_binding_bundle(root, r, &m)?;
    let encoded = serde_json::to_vec(&m).map_err(|e| format!("manifest serialization failed: {e}"))?;
    if domain_sha256(MIDI_MANIFEST_DIGEST_SCHEMA_V1.as_bytes(), &encoded) != r.midi.manifest_sha256 {
        return Err("MIDI manifest digest mismatch".into());
    }
    let midi = read(&root.join("performance.mid"), "MIDI")?;
    if sha256_hex(&midi) != r.midi.midi_sha256 || midi.len() != r.midi.byte_length
        || m.midi_sha256 != r.midi.midi_sha256 || m.score_sha256 != r.source.candidate_score_sha256
        || m.timeline_sha256 != r.midi.timeline_sha256 || m.binding_sha256 != r.midi.binding_sha256
        || m.capability_sha256 != r.midi.capability_sha256
        || m.tempo_microseconds_per_quarter != r.midi.tempo_microseconds_per_quarter
        || m.meter_numerator != r.midi.meter_numerator || m.meter_denominator != r.midi.meter_denominator
        || m.notes.len() != r.midi.note_count
    { return Err("MIDI bytes or manifest fields do not match render receipt".into()); }
    verify_midi_event_manifest(&m, &midi)
}

fn verify_render_invocation(root: &Path, r: &ReceiptV1) -> Result<(), String> {
    let path = root.join("render-invocation.json");
    let bytes = read(&path, "render invocation")?;
    let expected_sha = r.execution.invocation_sha256.as_deref().ok_or("missing invocation SHA")?;
    if sha256_hex(&bytes) != expected_sha {
        return Err("render invocation SHA-256 mismatch".into());
    }
    let invocation: RenderInvocationV1 = serde_json::from_slice(&bytes)
        .map_err(|e| format!("render invocation JSON invalid: {e}"))?;
    if invocation.schema != "muse-render-invocation-v1"
        || invocation.attempt_id != r.attempt_id
        || invocation.renderer_id != r.renderer.renderer_id
        || invocation.renderer_version != r.renderer.renderer_version
        || invocation.executable_sha256 != r.renderer.binary_sha256
        || invocation.environment_sha256 != r.renderer.environment_sha256
        || invocation.configuration_sha256 != r.renderer.configuration_sha256
        || invocation.asset_sha256 != r.renderer.asset_sha256
        || invocation.input_midi_path != "performance.mid"
        || invocation.input_midi_sha256 != r.midi.midi_sha256
        || invocation.output_audio_path != "output.wav"
        || invocation.output_audio_sha256 != r.audio.audio_sha256.as_deref().unwrap_or("")
        || invocation.output_audio_sha256 != r.execution.output_audio_sha256.as_deref().unwrap_or("")
        || invocation.started_at != r.execution.started_at.as_deref().unwrap_or("")
        || invocation.finished_at != r.execution.finished_at.as_deref().unwrap_or("")
        || Some(invocation.exit_code) != r.execution.exit_code
        || r.execution.input_midi_sha256.as_deref() != Some(invocation.input_midi_sha256.as_str())
        || r.execution.output_audio_sha256.is_none()
        || invocation.arguments.is_empty()
    {
        return Err("render invocation fields do not match receipt renderer, MIDI input, WAV output, and execution result".into());
    }
    Ok(())
}

fn verify_wav(root: &Path, r: &ReceiptV1) -> Result<(), String> {
    let path = root.join("output.wav");
    let expected_bytes = r.audio.byte_length.ok_or("missing WAV byte length")?;
    let actual_bytes = fs::metadata(&path).map_err(|e| format!("could not stat WAV: {e}"))?.len() as usize;
    if actual_bytes != expected_bytes { return Err("WAV byte length does not match receipt".into()); }
    check_file(&path, r.audio.audio_sha256.as_deref().ok_or("missing audio SHA")?, "WAV")?;
    let mut reader = hound::WavReader::open(&path).map_err(|e| format!("WAV parse failed: {e}"))?;
    let spec = reader.spec();
    if spec.sample_rate != r.audio.sample_rate_hz.unwrap_or(0) || spec.channels != r.audio.channels.unwrap_or(0) {
        return Err("WAV sample rate/channel count mismatch".into());
    }
    let mut count = 0_u64;
    let mut non_silent = false;
    match spec.sample_format {
        hound::SampleFormat::Float => {
            for sample in reader.samples::<f32>() {
                let x = sample.map_err(|e| format!("WAV sample decode failed: {e}"))?;
                if !x.is_finite() { return Err("WAV contains NaN/Inf".into()); }
                non_silent |= x.abs() > 1.0e-8;
                count += 1;
            }
        }
        hound::SampleFormat::Int => {
            for sample in reader.samples::<i32>() {
                let x = sample.map_err(|e| format!("WAV sample decode failed: {e}"))?;
                non_silent |= x != 0;
                count += 1;
            }
        }
    }
    let channels = u64::from(spec.channels);
    if channels == 0 || count == 0 || count % channels != 0 || count / channels != r.audio.frames.unwrap_or(0) {
        return Err("WAV sample count/frame count mismatch".into());
    }
    if !non_silent { return Err("rendered WAV is silent".into()); }
    Ok(())
}

pub fn verify_bundle(root: &Path, r: &ReceiptV1) -> Result<BundleReportV1, String> {
    validate_receipt(r)?;
    if r.attempt_state == "outcome_unknown" {
        return Ok(BundleReportV1 {
            attempt_id:r.attempt_id.clone(), attempt_state:r.attempt_state.clone(),
            receipt_seal_verified:true, bundle_integrity_verified:false,
            midi_event_manifest_verified:false, wav_integrity_verified:false,
            acceptance_eligible:false, trust_level:"unresolved_execution".into(),
            notes:vec!["unknown outcome is not retry permission; reconcile the original attempt".into()],
        });
    }
    if r.attempt_state != "completed" { return Err("bundle verification requires completed state".into()); }

    check_file(&root.join("source-score.json"), &r.source.source_score_sha256, "source Score")?;
    check_file(&root.join("candidate-score.json"), &r.source.candidate_score_sha256, "candidate Score")?;
    verify_edit_bundle(root, r)?;
    verify_midi_bundle(root, r)?;
    for (file, hash, label) in [
        ("renderer-binary", r.renderer.binary_sha256.as_str(), "renderer binary"),
        ("renderer-environment.json", r.renderer.environment_sha256.as_str(), "renderer environment"),
        ("renderer-config.json", r.renderer.configuration_sha256.as_str(), "renderer config"),
    ] { check_file(&root.join(file), hash, label)?; }
    if let Some(hash) = r.renderer.asset_sha256.as_deref() {
        check_file(&root.join("renderer-asset.bin"), hash, "renderer asset")?;
    }
    verify_render_invocation(root, r)?;
    if let Some(hash) = r.execution.stdout_sha256.as_deref() { check_file(&root.join("render.stdout"), hash, "renderer stdout")?; }
    if let Some(hash) = r.execution.stderr_sha256.as_deref() { check_file(&root.join("render.stderr"), hash, "renderer stderr")?; }
    verify_wav(root, r)?;

    let mut notes = vec!["integrity verification is not an edit-acceptance or human-quality decision".into()];
    if r.renderer.fallback_used { notes.push("renderer fallback used; treat as a distinct realization stratum".into()); }
    let trust_level = match r.attestation.kind.as_str() {
        "unsigned_local" => "unsigned_local_integrity_only",
        "workflow_observed" => "workflow_observed_unverified",
        "signed" => "signature_fields_present_but_unverified",
        _ => "unknown",
    };
    notes.push("this verifier does not validate signatures, trusted runner identity, publication rights, or musician commit authorization".into());
    Ok(BundleReportV1 {
        attempt_id:r.attempt_id.clone(), attempt_state:r.attempt_state.clone(),
        receipt_seal_verified:true, bundle_integrity_verified:true,
        midi_event_manifest_verified:true, wav_integrity_verified:true,
        acceptance_eligible:false, trust_level:trust_level.into(), notes,
    })
}

pub fn receipt_only_report(r: &ReceiptV1) -> BundleReportV1 {
    BundleReportV1 {
        attempt_id:r.attempt_id.clone(), attempt_state:r.attempt_state.clone(),
        receipt_seal_verified:true, bundle_integrity_verified:false,
        midi_event_manifest_verified:false, wav_integrity_verified:false,
        acceptance_eligible:false,
        trust_level:if r.attempt_state == "outcome_unknown" { "unresolved_execution" } else { "receipt_only" }.into(),
        notes:vec!["receipt-only mode does not read or re-hash the artifact bundle".into()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn d(c: char) -> String { std::iter::repeat(c).take(64).collect() }
    fn seal(mut value: Value) -> Value {
        let hash = receipt_digest(&value).unwrap();
        value["receipt_sha256"] = Value::String(hash);
        value
    }
    fn sample_receipt() -> Vec<u8> {
        let candidate = d('b');
        let midi = d('1');
        let audio = d('8');
        let value = json!({
            "schema":RECEIPT_SCHEMA_V1,"attempt_id":"test.completed.1","attempt_state":"completed",
            "source":{"source_score_sha256":d('a'),"candidate_score_sha256":candidate,"score_digest_schema":MIDI_SCORE_DIGEST_SCHEMA_V1},
            "edit":{"plan_sha256":d('c'),"edit_receipt_sha256":d('d'),"verifier_id":"test-verifier","verifier_version":"1","verification_receipt_sha256":d('e'),"verdict":"pass","authorized_operation_ids":["op-1"]},
            "midi":{"manifest_schema":MIDI_MANIFEST_SCHEMA_V1,"manifest_digest_schema":MIDI_MANIFEST_DIGEST_SCHEMA_V1,"manifest_sha256":d('f'),
                "score_digest_schema":MIDI_SCORE_DIGEST_SCHEMA_V1,"score_sha256":candidate,
                "timeline_digest_schema":MIDI_TIMELINE_DIGEST_SCHEMA_V1,"timeline_sha256":d('2'),
                "binding_digest_schema":MIDI_BINDING_DIGEST_SCHEMA_V1,"binding_sha256":d('3'),
                "capability_digest_schema":MIDI_CAPABILITY_DIGEST_SCHEMA_V1,"capability_sha256":d('4'),
                "midi_sha256":midi,"byte_length":64,"ticks_per_quarter":MIDI_TICKS_PER_QUARTER_V1,
                "tempo_microseconds_per_quarter":500000,"meter_numerator":4,"meter_denominator":4,"note_count":1},
            "renderer":{"renderer_id":"test-renderer","renderer_version":"1","binary_sha256":d('5'),"environment_sha256":d('6'),
                "configuration_sha256":d('7'),"asset_sha256":null,"fallback_used":false,"fallback_reason":null},
            "execution":{"state":"completed","executor_id":"test","input_midi_sha256":midi,"output_audio_sha256":audio,"exit_code":0,
                "started_at":"2026-11-19T10:00:00Z","finished_at":"2026-11-19T10:00:02Z",
                "stdout_sha256":null,"stderr_sha256":null,"invocation_sha256":d('9')},
            "audio":{"disposition":"produced","container":"wav","audio_sha256":audio,"byte_length":128,"sample_rate_hz":44100,"channels":2,"frames":32},
            "verification":{
                "midi_score_conformance":{"verdict":"not_run","verifier_id":null,"verifier_version":null,"receipt_sha256":null,"reason":"test"},
                "rendered_score_conformance":{"verdict":"not_run","verifier_id":null,"verifier_version":null,"receipt_sha256":null,"reason":"test"},
                "protected_audio_preservation":{"verdict":"not_run","verifier_id":null,"verifier_version":null,"receipt_sha256":null,"reason":"test"},
                "human_evaluation":{"verdict":"not_run","verifier_id":null,"verifier_version":null,"receipt_sha256":null,"reason":"test"}},
            "attestation":{"kind":"unsigned_local","identity":"unit-test","key_id":null,"signature":null},
            "digest_recipe":RECEIPT_DIGEST_RECIPE_V1,"receipt_sha256":d('0')
        });
        serde_json::to_vec(&seal(value)).unwrap()
    }

    #[test]
    fn receipt_seal_verifies_and_tampering_is_detected() {
        let bytes = sample_receipt();
        assert_eq!(verify_receipt_bytes(&bytes).unwrap().attempt_state, "completed");
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["attempt_id"] = Value::String("tampered".into());
        assert!(verify_receipt_bytes(&serde_json::to_vec(&changed).unwrap()).unwrap_err().contains("seal mismatch"));
    }
    #[test]
    fn candidate_score_and_midi_score_must_match() {
        let mut value: Value = serde_json::from_slice(&sample_receipt()).unwrap();
        value["midi"]["score_sha256"] = Value::String(d('z'));
        let bytes = serde_json::to_vec(&seal(value)).unwrap();
        assert!(verify_receipt_bytes(&bytes).unwrap_err().contains("candidate Score digest"));
    }
    #[test]
    fn outcome_unknown_is_not_a_success() {
        let mut value: Value = serde_json::from_slice(&sample_receipt()).unwrap();
        value["attempt_state"] = Value::String("outcome_unknown".into());
        value["execution"]["state"] = Value::String("unknown".into());
        value["execution"]["output_audio_sha256"] = Value::Null;
        value["execution"]["exit_code"] = Value::Null;
        value["execution"]["finished_at"] = Value::Null;
        value["audio"]["disposition"] = Value::String("unknown".into());
        value["audio"]["audio_sha256"] = Value::Null;
        value["audio"]["byte_length"] = Value::Null;
        value["audio"]["sample_rate_hz"] = Value::Null;
        value["audio"]["channels"] = Value::Null;
        value["audio"]["frames"] = Value::Null;
        let receipt = verify_receipt_bytes(&serde_json::to_vec(&seal(value)).unwrap()).unwrap();
        assert_eq!(receipt.attempt_state, "outcome_unknown");
        assert_eq!(receipt.execution.state, "unknown");
    }

    fn vlq(mut value: u32) -> Vec<u8> {
        let mut buffer = [0_u8;4];
        let mut index = 3;
        buffer[index] = (value & 0x7f) as u8;
        while { value >>= 7; value != 0 } {
            index -= 1;
            buffer[index] = ((value & 0x7f) as u8) | 0x80;
        }
        buffer[index..].to_vec()
    }
    fn track(out: &mut Vec<u8>, data: &[u8]) {
        out.extend_from_slice(b"MTrk");
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(data);
    }
    fn simple_midi() -> Vec<u8> {
        let mut out = b"MThd".to_vec();
        out.extend_from_slice(&6_u32.to_be_bytes());
        out.extend_from_slice(&1_u16.to_be_bytes());
        out.extend_from_slice(&2_u16.to_be_bytes());
        out.extend_from_slice(&MIDI_TICKS_PER_QUARTER_V1.to_be_bytes());
        let mut tempo = vec![0,0xff,0x03,20];
        tempo.extend_from_slice(b"Muse Tempo and Meter");
        tempo.extend_from_slice(&[0,0xff,0x51,3,0x07,0xa1,0x20]);
        tempo.extend_from_slice(&[0,0xff,0x58,4,4,2,24,8]);
        tempo.extend_from_slice(&[0,0xff,0x2f,0]);
        track(&mut out,&tempo);
        let name = b"Part 1 seg-a inst-a";
        let mut part = vec![0,0xff,0x03,name.len() as u8];
        part.extend_from_slice(name);
        part.extend_from_slice(&[0,0xc0,0,0,0x90,60,90]);
        part.extend_from_slice(&vlq(480));
        part.extend_from_slice(&[0x80,60,0,0,0xff,0x2f,0]);
        track(&mut out,&part);
        out
    }
    fn manifest() -> MidiPerformanceManifestV1 {
        let bytes = simple_midi();
        MidiPerformanceManifestV1 {
            schema:MIDI_MANIFEST_SCHEMA_V1.into(),score_digest_schema:MIDI_SCORE_DIGEST_SCHEMA_V1.into(),
            score_sha256:d('b'),timeline_digest_schema:MIDI_TIMELINE_DIGEST_SCHEMA_V1.into(),timeline_sha256:d('2'),
            binding_digest_schema:MIDI_BINDING_DIGEST_SCHEMA_V1.into(),binding_sha256:d('3'),
            capability_digest_schema:MIDI_CAPABILITY_DIGEST_SCHEMA_V1.into(),capability_sha256:d('4'),
            midi_sha256:sha256_hex(&bytes),ticks_per_quarter:960,tempo_microseconds_per_quarter:500000,
            meter_numerator:4,meter_denominator:4,
            tracks:vec![MidiPerformanceTrackReceiptV1 {
                track_index:1,track_name:"Part 1 seg-a inst-a".into(),part_id:1,segment_id:"seg-a".into(),
                instrument_id:"inst-a".into(),renderer_id:"test-renderer".into(),patch_id:"test-patch".into(),
                gm_program:0,midi_channel:0,segment_start_tick:0,segment_end_tick:960,
            }],
            notes:vec![MidiPerformanceNoteReceiptV1 {
                source_note_ordinal:0,part_id:1,segment_id:"seg-a".into(),instrument_id:"inst-a".into(),
                track_index:1,start_tick:0,end_tick:480,pitch_midi:60,velocity_midi:90,
            }],
        }
    }
    #[test]
    fn midi_event_manifest_matches_real_smf_bytes() {
        verify_midi_event_manifest(&manifest(), &simple_midi()).unwrap();
    }
    #[test]
    fn complete_bundle_integrity_passes_then_rejects_midi_tampering() {
        let root = std::env::temp_dir().join(format!("muse-render-receipt-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let source_bytes = br#"{"score":"source-fixture"}"#.to_vec();
        let candidate_bytes = br#"{"score":"candidate-fixture"}"#.to_vec();
        std::fs::write(root.join("source-score.json"), &source_bytes).unwrap();
        std::fs::write(root.join("candidate-score.json"), &candidate_bytes).unwrap();
        let source_sha = sha256_hex(&source_bytes);
        let candidate_sha = sha256_hex(&candidate_bytes);

        let plan_bytes = serde_json::to_vec(&json!({
            "schema":SCORE_EDIT_SCHEMA_V1,
            "input_digest_schema":SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1,
            "input_score_sha256":source_sha.clone(),
            "operations":[{"id":"op-1"}],
            "protected_windows":[{"id":"keep-1"}]
        })).unwrap();
        let edit_bytes = serde_json::to_vec(&json!({
            "schema":SCORE_EDIT_SCHEMA_V1,
            "input_digest_schema":SCORE_EDIT_INPUT_DIGEST_SCHEMA_V1,
            "input_score_sha256":source_sha,
            "operation_ids":["op-1"],
            "protected_window_ids":["keep-1"],
            "changes":[]
        })).unwrap();
        let verification_bytes = serde_json::to_vec(&json!({
            "input_score_sha256":source_sha,
            "verified":true
        })).unwrap();
        std::fs::write(root.join("edit-plan.json"), &plan_bytes).unwrap();
        std::fs::write(root.join("edit-receipt.json"), &edit_bytes).unwrap();
        std::fs::write(root.join("edit-verification.json"), &verification_bytes).unwrap();

        let midi_bytes = simple_midi();
        let midi_sha = sha256_hex(&midi_bytes);
        std::fs::write(root.join("performance.mid"), &midi_bytes).unwrap();
        let timeline_bytes = serde_json::to_vec(&json!({
            "schema":"orchestration-timeline-v1",
            "score_digest_schema":MIDI_SCORE_DIGEST_SCHEMA_V1,
            "score_sha256":candidate_sha.clone(),
            "segments":[{
                "id":"seg-a","part_id":1,"instrument_id":"inst-a",
                "start":{"numerator":0,"denominator":1},
                "end":{"numerator":1,"denominator":1},
                "musical_function":"melody","handoff":null
            }]
        })).unwrap();
        let binding_bytes = serde_json::to_vec(&json!([{
            "instrument_id":"inst-a","renderer_id":"test-renderer",
            "patch_id":"test-patch","gm_program":0,"asset_sha256":null
        }])).unwrap();
        let capability_bytes = serde_json::to_vec(&json!([{"id":"inst-a"}])).unwrap();
        std::fs::write(root.join("orchestration-timeline.json"), &timeline_bytes).unwrap();
        std::fs::write(root.join("instrument-bindings.json"), &binding_bytes).unwrap();
        std::fs::write(root.join("instrument-capabilities.json"), &capability_bytes).unwrap();

        let mut midi_manifest = manifest();
        midi_manifest.score_sha256 = candidate_sha.clone();
        midi_manifest.midi_sha256 = midi_sha.clone();
        midi_manifest.timeline_sha256 = domain_sha256(MIDI_TIMELINE_DIGEST_SCHEMA_V1.as_bytes(), &timeline_bytes);
        midi_manifest.binding_sha256 = domain_sha256(MIDI_BINDING_DIGEST_SCHEMA_V1.as_bytes(), &binding_bytes);
        midi_manifest.capability_sha256 = domain_sha256(MIDI_CAPABILITY_DIGEST_SCHEMA_V1.as_bytes(), &capability_bytes);
        let manifest_bytes = serde_json::to_vec(&midi_manifest).unwrap();
        let manifest_sha = domain_sha256(MIDI_MANIFEST_DIGEST_SCHEMA_V1.as_bytes(), &manifest_bytes);
        std::fs::write(root.join("midi-manifest.json"), &manifest_bytes).unwrap();

        let renderer_binary = b"synthetic-renderer-binary".to_vec();
        let renderer_environment = b"{\"runtime\":\"synthetic\"}".to_vec();
        let renderer_config = b"{\"sample_rate\":44100}".to_vec();
        std::fs::write(root.join("renderer-binary"), &renderer_binary).unwrap();
        std::fs::write(root.join("renderer-environment.json"), &renderer_environment).unwrap();
        std::fs::write(root.join("renderer-config.json"), &renderer_config).unwrap();
        let invocation_template = |output_sha: &str| serde_json::to_vec(&json!({
            "schema":"muse-render-invocation-v1",
            "attempt_id":"synthetic-bundle.1",
            "renderer_id":"test-renderer",
            "renderer_version":"1",
            "executable_sha256":sha256_hex(&renderer_binary),
            "environment_sha256":sha256_hex(&renderer_environment),
            "configuration_sha256":sha256_hex(&renderer_config),
            "asset_sha256":null,
            "input_midi_path":"performance.mid",
            "input_midi_sha256":midi_sha.clone(),
            "output_audio_path":"output.wav",
            "output_audio_sha256":output_sha,
            "arguments":["--input","performance.mid","--output","output.wav"],
            "started_at":"2026-11-19T10:00:00Z",
            "finished_at":"2026-11-19T10:00:01Z",
            "exit_code":0
        })).unwrap();
        let wav_path = root.join("output.wav");
        let spec = hound::WavSpec {
            channels:2, sample_rate:44100, bits_per_sample:16,
            sample_format:hound::SampleFormat::Int
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for sample in 0..64_i16 {
            writer.write_sample::<i16>(1000 - sample).unwrap();
        }
        writer.finalize().unwrap();
        let wav_bytes = std::fs::read(&wav_path).unwrap();
        let wav_sha = sha256_hex(&wav_bytes);
        let invocation = invocation_template(&wav_sha);
        std::fs::write(root.join("render-invocation.json"), &invocation).unwrap();

        let d = |c: char| std::iter::repeat(c).take(64).collect::<String>();
        let mut receipt = ReceiptV1 {
            schema:RECEIPT_SCHEMA_V1.into(),
            attempt_id:"synthetic-bundle.1".into(),
            attempt_state:"completed".into(),
            source:SourceBindingV1 {
                source_score_sha256:source_sha.clone(),
                candidate_score_sha256:candidate_sha.clone(),
                score_digest_schema:MIDI_SCORE_DIGEST_SCHEMA_V1.into()
            },
            edit:EditBindingV1 {
                plan_sha256:sha256_hex(&plan_bytes),
                edit_receipt_sha256:sha256_hex(&edit_bytes),
                verifier_id:"synthetic-independent-verifier".into(),
                verifier_version:"test-1".into(),
                verification_receipt_sha256:sha256_hex(&verification_bytes),
                verdict:"pass".into(),
                authorized_operation_ids:vec!["op-1".into()]
            },
            midi:MidiBindingV1 {
                manifest_schema:MIDI_MANIFEST_SCHEMA_V1.into(),
                manifest_digest_schema:MIDI_MANIFEST_DIGEST_SCHEMA_V1.into(),
                manifest_sha256:manifest_sha,
                score_digest_schema:MIDI_SCORE_DIGEST_SCHEMA_V1.into(),
                score_sha256:candidate_sha,
                timeline_digest_schema:MIDI_TIMELINE_DIGEST_SCHEMA_V1.into(),
                timeline_sha256:midi_manifest.timeline_sha256.clone(),
                binding_digest_schema:MIDI_BINDING_DIGEST_SCHEMA_V1.into(),
                binding_sha256:midi_manifest.binding_sha256.clone(),
                capability_digest_schema:MIDI_CAPABILITY_DIGEST_SCHEMA_V1.into(),
                capability_sha256:midi_manifest.capability_sha256.clone(),
                midi_sha256:midi_sha.clone(),
                byte_length:midi_bytes.len(),
                ticks_per_quarter:960,
                tempo_microseconds_per_quarter:500000,
                meter_numerator:4,meter_denominator:4,note_count:1
            },
            renderer:RendererBindingV1 {
                renderer_id:"test-renderer".into(),renderer_version:"1".into(),
                binary_sha256:sha256_hex(&renderer_binary),
                environment_sha256:sha256_hex(&renderer_environment),
                configuration_sha256:sha256_hex(&renderer_config),
                asset_sha256:None,fallback_used:false,fallback_reason:None
            },
            execution:ExecutionBindingV1 {
                state:"completed".into(),executor_id:"synthetic-runner".into(),
                input_midi_sha256:Some(midi_sha.clone()),output_audio_sha256:Some(wav_sha.clone()),
                exit_code:Some(0),started_at:Some("2026-11-19T10:00:00Z".into()),
                finished_at:Some("2026-11-19T10:00:01Z".into()),
                stdout_sha256:None,stderr_sha256:None,
                invocation_sha256:Some(sha256_hex(&invocation))
            },
            audio:AudioBindingV1 {
                disposition:"produced".into(),container:"wav".into(),audio_sha256:Some(wav_sha),
                byte_length:Some(wav_bytes.len()),sample_rate_hz:Some(44100),
                channels:Some(2),frames:Some(32)
            },
            verification:VerificationBindingV1 {
                midi_score_conformance:EvidenceResultV1 {verdict:"not_run".into(),verifier_id:None,verifier_version:None,receipt_sha256:None,reason:Some("fixture".into())},
                rendered_score_conformance:EvidenceResultV1 {verdict:"not_run".into(),verifier_id:None,verifier_version:None,receipt_sha256:None,reason:Some("fixture".into())},
                protected_audio_preservation:EvidenceResultV1 {verdict:"not_run".into(),verifier_id:None,verifier_version:None,receipt_sha256:None,reason:Some("fixture".into())},
                human_evaluation:EvidenceResultV1 {verdict:"not_run".into(),verifier_id:None,verifier_version:None,receipt_sha256:None,reason:Some("fixture".into())}
            },
            attestation:AttestationV1 {
                kind:"unsigned_local".into(),identity:"synthetic-test".into(),key_id:None,signature:None
            },
            digest_recipe:RECEIPT_DIGEST_RECIPE_V1.into(),
            receipt_sha256:d('0')
        };
        let value = serde_json::to_value(&receipt).unwrap();
        receipt.receipt_sha256 = receipt_digest(&value).unwrap();
        let sealed = serde_json::to_vec(&receipt).unwrap();
        let parsed = verify_receipt_bytes(&sealed).unwrap();
        let report = verify_bundle(&root, &parsed).unwrap();
        assert!(report.bundle_integrity_verified);
        assert!(report.midi_event_manifest_verified);
        assert!(report.wav_integrity_verified);
        assert!(!report.acceptance_eligible);

        // Re-seal a receipt around an invocation file whose hash is correct
        // but whose claimed MIDI input differs. The semantic join must fail.
        let mut altered_invocation: Value = serde_json::from_slice(&invocation).unwrap();
        altered_invocation["input_midi_sha256"] = Value::String(d('a'));
        let altered_invocation_bytes = serde_json::to_vec(&altered_invocation).unwrap();
        std::fs::write(root.join("render-invocation.json"), &altered_invocation_bytes).unwrap();
        let mut altered_receipt_value = serde_json::to_value(&parsed).unwrap();
        altered_receipt_value["execution"]["invocation_sha256"] =
            Value::String(sha256_hex(&altered_invocation_bytes));
        let altered_receipt_bytes = serde_json::to_vec(&seal(altered_receipt_value)).unwrap();
        let altered_receipt = verify_receipt_bytes(&altered_receipt_bytes).unwrap();
        assert!(verify_bundle(&root, &altered_receipt).unwrap_err().contains("render invocation fields"));

        std::fs::write(root.join("render-invocation.json"), &invocation).unwrap();

        // A re-sealed but cross-attempt invocation is not replayable into this attempt.
        let mut wrong_attempt: Value = serde_json::from_slice(&invocation).unwrap();
        wrong_attempt["attempt_id"] = Value::String("different-attempt".into());
        let wrong_attempt_bytes = serde_json::to_vec(&wrong_attempt).unwrap();
        std::fs::write(root.join("render-invocation.json"), &wrong_attempt_bytes).unwrap();
        let mut wrong_attempt_receipt: Value = serde_json::to_value(&parsed).unwrap();
        wrong_attempt_receipt["execution"]["invocation_sha256"] =
            Value::String(sha256_hex(&wrong_attempt_bytes));
        let wrong_attempt_sealed = serde_json::to_vec(&seal(wrong_attempt_receipt)).unwrap();
        let wrong_attempt_receipt = verify_receipt_bytes(&wrong_attempt_sealed).unwrap();
        assert!(verify_bundle(&root, &wrong_attempt_receipt).unwrap_err().contains("render invocation fields"));

        std::fs::write(root.join("render-invocation.json"), &invocation).unwrap();
        let mut corrupted = midi_bytes;
        let index = corrupted.iter().rposition(|x| *x == 60).unwrap();
        corrupted[index] = 61;
        std::fs::write(root.join("performance.mid"), corrupted).unwrap();
        assert!(verify_bundle(&root, &parsed).is_err());

        let _ = std::fs::remove_dir_all(&root);
    }
}

//! Build an unsigned standard in-toto Statement for verified MEL audio evidence.
//!
//! This is a subject/predicate adapter only. It does not construct a DSSE
//! envelope, attach a signature, verify a signer, or create a trust root.
//! #13254 remains the owner of DSSE, in-toto envelope authentication and
//! signer-role/trust-policy verification.

use serde::Serialize;

use crate::audio_evidence_window::{
    AudioEvidenceWindowManifestV1, EvidenceConditionV1, VerifiedWindowV1,
};
use crate::audio_revision_binding::audit_revision_with_audio_evidence;
use crate::evidence_digest::{canonical_json_sha256, sha256_hex};
use crate::temporal_revision::RevisionCaseV2;

const STATEMENT_TYPE_V1: &str = "https://in-toto.io/Statement/v1";
const PREDICATE_TYPE_V1: &str =
    "https://luminousdynamics.io/attestations/mel/audio-evidence-materialization/v1";
const DSSE_PAYLOAD_TYPE_IN_TOTO: &str = "application/vnd.in-toto+json";

#[derive(Debug, Clone, PartialEq, Serialize)]
struct InTotoDigestV1 {
    sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct InTotoSubjectV1 {
    name: String,
    digest: InTotoDigestV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct AudioEvidencePredicateV1 {
    schema: &'static str,
    evidence_level: &'static str,
    experiment_unit_id: String,
    target_source_lineage_id: String,
    evidence_source_lineage_id: String,
    query_id: String,
    query_text_sha256: String,
    evidence_condition: EvidenceConditionV1,
    target_audio_sha256: String,
    target_audio_bytes_recomputed: bool,
    source_audio_sha256: String,
    source_audio_byte_length: u64,
    rights_scope_receipt_sha256: String,
    decoded_pcm_sha256: String,
    decoded_pcm_byte_length: u64,
    sample_rate_hz: u32,
    channels: u16,
    total_frames: u64,
    sample_format: crate::audio_evidence_window::PcmSampleFormatV1,
    decoder_identity_sha256: String,
    decode_profile_sha256: String,
    model_identity_sha256: String,
    runtime_identity_sha256: String,
    decoding_config_sha256: String,
    prompt_sha256: String,
    adapter_sha256: String,
    retrieval_policy_sha256: String,
    selected_windows: Vec<VerifiedWindowV1>,
    full_manifest_sha256: String,
    context_manifest_sha256: String,
    revision_report_sha256: String,
    max_output_tokens: u32,
    actual_output_tokens: u32,
    max_wall_time_ms: u64,
    elapsed_wall_time_ms: u64,
    source_audio_bytes_recomputed: bool,
    decoded_pcm_bytes_recomputed: bool,
    window_payloads_recomputed: bool,
    context_manifest_recomputed: bool,
    rights_scope_authenticated: bool,
    decoder_identity_authenticated: bool,
    model_runtime_identity_authenticated: bool,
    runner_receipt_verified: bool,
    request_body_bytes_recomputed: bool,
    request_body_delivery_attested: bool,
    model_input_attested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct InTotoStatementV1 {
    #[serde(rename = "_type")]
    statement_type: &'static str,
    subject: Vec<InTotoSubjectV1>,
    #[serde(rename = "predicateType")]
    predicate_type: &'static str,
    predicate: AudioEvidencePredicateV1,
}

/// Exact in-toto payload bytes ready for the existing #13254 DSSE signer.
/// No DSSE envelope or signature is created by this module.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnsignedInTotoAudioEvidenceStatementV1 {
    pub payload_type: &'static str,
    /// Compact JSON encoding of an in-toto Statement v1.
    pub statement_json: String,
    /// SHA-256 over the exact statement_json UTF-8 bytes to be signed.
    pub statement_payload_sha256: String,
    /// Explicitly false: this module does not sign the statement.
    pub dsse_envelope_created: bool,
    /// Explicitly false: no signer or trust-root verification occurs here.
    pub signature_verified: bool,
}

pub fn build_audio_evidence_in_toto_statement(
    revision_case: &RevisionCaseV2,
    evidence_manifest: &AudioEvidenceWindowManifestV1,
    source_audio_bytes: &[u8],
    decoded_pcm_bytes: &[u8],
    request_body_bytes: &[u8],
) -> Result<UnsignedInTotoAudioEvidenceStatementV1, String> {
    let audit = audit_revision_with_audio_evidence(
        revision_case,
        evidence_manifest,
        source_audio_bytes,
        decoded_pcm_bytes,
        request_body_bytes,
    )?;

    let revision_report_sha256 = canonical_json_sha256(&audit.revision_report)
        .map_err(|error| format!("failed canonical revision-report digest: {error}"))?;
    let audio_report_sha256 = canonical_json_sha256(&audit.audio_evidence_report)
        .map_err(|error| format!("failed canonical audio-evidence report digest: {error}"))?;

    let mut subjects = Vec::with_capacity(6 + audit.audio_evidence_report.verified_windows.len());
    subjects.push(subject(
        format!(
            "target-audio-reference/{}",
            audit.audio_evidence_report.target_audio_sha256
        ),
        audit.audio_evidence_report.target_audio_sha256.clone(),
    ));
    subjects.push(subject(
        format!("source-audio/{}", audit.audio_evidence_report.source_audio_sha256),
        audit.audio_evidence_report.source_audio_sha256.clone(),
    ));
    subjects.push(subject(
        format!("decoded-pcm/{}", audit.audio_evidence_report.decoded_pcm_sha256),
        audit.audio_evidence_report.decoded_pcm_sha256.clone(),
    ));
    for window in &audit.audio_evidence_report.verified_windows {
        subjects.push(subject(
            format!("audio-window/{}", window.window_id),
            window.recomputed_sha256.clone(),
        ));
    }
    subjects.push(subject(
        format!(
            "client-request-body/{}",
            audit.audio_evidence_report.request_body_sha256
        ),
        audit.audio_evidence_report.request_body_sha256.clone(),
    ));
    subjects.push(subject(
        format!(
            "audio-context-manifest/{}",
            audit.audio_evidence_report.evidence_context_manifest_sha256
        ),
        audit.audio_evidence_report.evidence_context_manifest_sha256.clone(),
    ));
    subjects.push(subject(
        format!(
            "audio-evidence-manifest/{}",
            audit.audio_evidence_report.full_manifest_sha256
        ),
        audit.audio_evidence_report.full_manifest_sha256.clone(),
    ));
    subjects.push(subject(
        format!("audio-evidence-report/{audio_report_sha256}"),
        audio_report_sha256,
    ));
    subjects.push(subject(
        format!("temporal-revision-report/{revision_report_sha256}"),
        revision_report_sha256.clone(),
    ));

    let predicate = AudioEvidencePredicateV1 {
        schema: "mel-audio-evidence-materialization-predicate-v1",
        evidence_level: "DigestLinked",
        experiment_unit_id: audit.audio_evidence_report.experiment_unit_id.clone(),
        target_source_lineage_id: audit
            .audio_evidence_report
            .target_source_lineage_id
            .clone(),
        evidence_source_lineage_id: audit.audio_evidence_report.source_lineage_id.clone(),
        query_id: audit.audio_evidence_report.query_id.clone(),
        query_text_sha256: evidence_manifest.query_text_sha256.clone(),
        evidence_condition: audit.audio_evidence_report.evidence_condition,
        target_audio_sha256: audit.audio_evidence_report.target_audio_sha256.clone(),
        target_audio_bytes_recomputed: audit.audio_evidence_report.target_audio_sha256
            == audit.audio_evidence_report.source_audio_sha256
            && audit.audio_evidence_report.source_audio_bytes_recomputed,
        source_audio_sha256: audit.audio_evidence_report.source_audio_sha256.clone(),
        source_audio_byte_length: evidence_manifest.source_audio.byte_length,
        rights_scope_receipt_sha256: evidence_manifest
            .source_audio
            .rights_scope_receipt_sha256
            .clone(),
        decoded_pcm_sha256: audit.audio_evidence_report.decoded_pcm_sha256.clone(),
        decoded_pcm_byte_length: evidence_manifest.decoded_pcm.byte_length,
        sample_rate_hz: evidence_manifest.decoded_pcm.sample_rate_hz,
        channels: evidence_manifest.decoded_pcm.channels,
        total_frames: evidence_manifest.decoded_pcm.total_frames,
        sample_format: evidence_manifest.decoded_pcm.sample_format,
        decoder_identity_sha256: evidence_manifest.decoded_pcm.decoder_identity_sha256.clone(),
        decode_profile_sha256: evidence_manifest.decoded_pcm.decode_profile_sha256.clone(),
        model_identity_sha256: evidence_manifest.model_identity_sha256.clone(),
        runtime_identity_sha256: evidence_manifest.runtime_identity_sha256.clone(),
        decoding_config_sha256: evidence_manifest.decoding_config_sha256.clone(),
        prompt_sha256: evidence_manifest.context.prompt_sha256.clone(),
        adapter_sha256: evidence_manifest.context.adapter_sha256.clone(),
        retrieval_policy_sha256: evidence_manifest.context.retrieval_policy_sha256.clone(),
        selected_windows: audit.audio_evidence_report.verified_windows.clone(),
        full_manifest_sha256: audit.audio_evidence_report.full_manifest_sha256.clone(),
        context_manifest_sha256: audit
            .audio_evidence_report
            .evidence_context_manifest_sha256
            .clone(),
        revision_report_sha256,
        max_output_tokens: revision_case.max_output_tokens,
        actual_output_tokens: revision_case.actual_output_tokens,
        max_wall_time_ms: revision_case.max_wall_time_ms,
        elapsed_wall_time_ms: revision_case.elapsed_wall_time_ms,
        source_audio_bytes_recomputed: audit.audio_evidence_report.source_audio_bytes_recomputed,
        decoded_pcm_bytes_recomputed: audit.audio_evidence_report.decoded_pcm_bytes_recomputed,
        window_payloads_recomputed: audit.audio_evidence_report.window_payloads_recomputed,
        context_manifest_recomputed: audit.audio_evidence_report.context_manifest_recomputed,
        rights_scope_authenticated: audit.audio_evidence_report.rights_scope_authenticated,
        decoder_identity_authenticated: audit.audio_evidence_report.decoder_identity_authenticated,
        model_runtime_identity_authenticated: audit
            .audio_evidence_report
            .model_runtime_identity_authenticated,
        runner_receipt_verified: audit.audio_evidence_report.runner_receipt_verified,
        request_body_bytes_recomputed: audit
            .audio_evidence_report
            .request_body_bytes_recomputed,
        request_body_delivery_attested: audit
            .audio_evidence_report
            .request_body_delivery_attested,
        model_input_attested: audit.model_input_attested,
    };

    let statement = InTotoStatementV1 {
        statement_type: STATEMENT_TYPE_V1,
        subject: subjects,
        predicate_type: PREDICATE_TYPE_V1,
        predicate,
    };
    let statement_json = serde_json::to_string(&statement)
        .map_err(|error| format!("failed serializing in-toto Statement: {error}"))?;
    let statement_payload_sha256 = sha256_hex(statement_json.as_bytes());

    Ok(UnsignedInTotoAudioEvidenceStatementV1 {
        payload_type: DSSE_PAYLOAD_TYPE_IN_TOTO,
        statement_json,
        statement_payload_sha256,
        dsse_envelope_created: false,
        signature_verified: false,
    })
}

fn subject(name: String, sha256: String) -> InTotoSubjectV1 {
    InTotoSubjectV1 {
        name,
        digest: InTotoDigestV1 { sha256 },
    }
}

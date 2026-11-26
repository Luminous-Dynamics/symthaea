//! Exact audio evidence-window materialization checks for Melothaea.
//!
//! This module recomputes the source-file, decoded-PCM and selected-window
//! SHA-256 digests from supplied bytes, then verifies the ordered context
//! manifest. It does not authenticate the decoder, the request sender, a
//! remote API, or model-internal audio consumption. Those assurance stages
//! compose with #13254 and remain explicit false values in the report.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::evidence_digest::{canonical_json_sha256, sha256_hex};

const MANIFEST_SCHEMA_V1: &str = "mel-audio-evidence-window-v1";
const REPORT_SCHEMA_V1: &str = "mel-audio-evidence-verification-report-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceConditionV1 {
    LocalRetrieved,
    YokedRetrieved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcmSampleFormatV1 {
    U8,
    S16Le,
    S24Le,
    S32Le,
    F32Le,
}

impl PcmSampleFormatV1 {
    pub fn bytes_per_sample(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::S16Le => 2,
            Self::S24Le => 3,
            Self::S32Le | Self::F32Le => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAudioArtifactV1 {
    pub audio_sha256: String,
    pub byte_length: u64,
    /// Digest-shaped reference only; legal/purpose authority is verified elsewhere.
    pub rights_scope_receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecodedPcmArtifactV1 {
    pub pcm_sha256: String,
    pub byte_length: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub total_frames: u64,
    pub sample_format: PcmSampleFormatV1,
    pub decoder_identity_sha256: String,
    pub decode_profile_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceWindowV1 {
    pub window_id: String,
    pub start_frame: u64,
    pub end_frame: u64,
    pub window_pcm_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceContextItemV1 {
    /// Zero-based order in the context presented at the adapter boundary.
    pub position: u32,
    pub window_id: String,
    pub payload_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceContextV1 {
    pub prompt_sha256: String,
    pub adapter_sha256: String,
    pub retrieval_policy_sha256: String,
    pub items: Vec<EvidenceContextItemV1>,
    /// Digest over EvidenceContextDigestPayloadV1, with array order preserved.
    pub context_manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioEvidenceWindowManifestV1 {
    pub schema: String,
    pub experiment_unit_id: String,
    pub query_id: String,
    pub query_text: String,
    pub query_text_sha256: String,
    pub model_identity_sha256: String,
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub evidence_condition: EvidenceConditionV1,
    /// Digest of the exact audio artifact whose content is being queried.
    pub target_audio_sha256: String,
    /// Exact source bytes from which the selected evidence windows originate.
    pub source_audio: SourceAudioArtifactV1,
    /// The byte-exact decoded PCM representation used for frame slicing.
    pub decoded_pcm: DecodedPcmArtifactV1,
    pub windows: Vec<EvidenceWindowV1>,
    pub context: EvidenceContextV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerifiedWindowV1 {
    pub window_id: String,
    pub position: u32,
    pub start_frame: u64,
    pub end_frame: u64,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub declared_sha256: String,
    pub recomputed_sha256: String,
    pub digest_matches: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioEvidenceVerificationReportV1 {
    pub schema: &'static str,
    pub experiment_unit_id: String,
    pub query_id: String,
    pub evidence_condition: EvidenceConditionV1,
    pub target_audio_sha256: String,
    pub source_audio_sha256: String,
    pub rights_scope_receipt_sha256: String,
    pub decoded_pcm_sha256: String,
    pub evidence_context_manifest_sha256: String,
    pub full_manifest_sha256: String,
    pub source_audio_bytes_recomputed: bool,
    pub decoded_pcm_bytes_recomputed: bool,
    pub window_payloads_recomputed: bool,
    pub context_manifest_recomputed: bool,
    pub verified_window_count: usize,
    pub verified_windows: Vec<VerifiedWindowV1>,
    /// Decoder/profile identifiers are supplied metadata until #13254 verifies
    /// their signed runner/tool attestation.
    pub rights_scope_authenticated: bool,
    pub decoder_identity_authenticated: bool,
    pub model_runtime_identity_authenticated: bool,
    pub runner_receipt_verified: bool,
    /// This V1 adapter binds a context manifest, not the exact HTTP/API body.
    pub request_body_recomputed: bool,
    /// Input delivered to an adapter is not proof of model-internal consumption.
    pub model_input_attested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct EvidenceContextDigestPayloadV1<'a> {
    schema: &'static str,
    experiment_unit_id: &'a str,
    query_id: &'a str,
    query_text_sha256: &'a str,
    evidence_condition: EvidenceConditionV1,
    target_audio_sha256: &'a str,
    source_audio_sha256: &'a str,
    source_audio_byte_length: u64,
    rights_scope_receipt_sha256: &'a str,
    decoded_pcm_sha256: &'a str,
    decoded_pcm_byte_length: u64,
    sample_rate_hz: u32,
    channels: u16,
    total_frames: u64,
    sample_format: PcmSampleFormatV1,
    decoder_identity_sha256: &'a str,
    decode_profile_sha256: &'a str,
    windows: &'a [EvidenceWindowV1],
    model_identity_sha256: &'a str,
    runtime_identity_sha256: &'a str,
    decoding_config_sha256: &'a str,
    prompt_sha256: &'a str,
    adapter_sha256: &'a str,
    retrieval_policy_sha256: &'a str,
    items: &'a [EvidenceContextItemV1],
}

pub fn verify_audio_evidence_manifest_json(
    manifest_json: &str,
    source_audio_bytes: &[u8],
    decoded_pcm_bytes: &[u8],
) -> Result<AudioEvidenceVerificationReportV1, String> {
    let manifest: AudioEvidenceWindowManifestV1 = serde_json::from_str(manifest_json)
        .map_err(|error| format!("invalid audio evidence manifest JSON: {error}"))?;
    verify_audio_evidence_manifest(&manifest, source_audio_bytes, decoded_pcm_bytes)
}

pub fn verify_audio_evidence_manifest(
    manifest: &AudioEvidenceWindowManifestV1,
    source_audio_bytes: &[u8],
    decoded_pcm_bytes: &[u8],
) -> Result<AudioEvidenceVerificationReportV1, String> {
    if manifest.schema != MANIFEST_SCHEMA_V1 {
        return Err(format!("unsupported audio evidence schema: {}", manifest.schema));
    }
    require_text(&manifest.experiment_unit_id, "experiment_unit_id")?;
    require_text(&manifest.query_id, "query_id")?;
    require_text(&manifest.query_text, "query_text")?;
    validate_digest(&manifest.target_audio_sha256, "target_audio_sha256")?;
    if sha256_hex(manifest.query_text.as_bytes()) != manifest.query_text_sha256 {
        return Err("query text bytes do not match declared SHA-256".into());
    }
    validate_digest(&manifest.query_text_sha256, "query_text_sha256")?;
    validate_digest(&manifest.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&manifest.runtime_identity_sha256, "runtime_identity_sha256")?;
    validate_digest(&manifest.decoding_config_sha256, "decoding_config_sha256")?;
    validate_digest(&manifest.source_audio.audio_sha256, "source_audio.audio_sha256")?;
    validate_digest(
        &manifest.source_audio.rights_scope_receipt_sha256,
        "source_audio.rights_scope_receipt_sha256",
    )?;
    validate_digest(&manifest.decoded_pcm.pcm_sha256, "decoded_pcm.pcm_sha256")?;
    validate_digest(
        &manifest.decoded_pcm.decoder_identity_sha256,
        "decoded_pcm.decoder_identity_sha256",
    )?;
    validate_digest(
        &manifest.decoded_pcm.decode_profile_sha256,
        "decoded_pcm.decode_profile_sha256",
    )?;

    if source_audio_bytes.is_empty() {
        return Err("source audio byte payload must not be empty".into());
    }
    validate_declared_length(
        source_audio_bytes,
        manifest.source_audio.byte_length,
        "source_audio",
    )?;
    let source_audio_sha256 = sha256_hex(source_audio_bytes);
    if source_audio_sha256 != manifest.source_audio.audio_sha256 {
        return Err("source audio bytes do not match declared SHA-256".into());
    }

    validate_declared_length(
        decoded_pcm_bytes,
        manifest.decoded_pcm.byte_length,
        "decoded_pcm",
    )?;
    if decoded_pcm_bytes.is_empty() {
        return Err("decoded PCM byte payload must not be empty".into());
    }
    let decoded_pcm_sha256 = sha256_hex(decoded_pcm_bytes);
    if decoded_pcm_sha256 != manifest.decoded_pcm.pcm_sha256 {
        return Err("decoded PCM bytes do not match declared SHA-256".into());
    }

    if manifest.decoded_pcm.sample_rate_hz == 0 || manifest.decoded_pcm.channels == 0 {
        return Err("decoded PCM sample rate and channel count must be positive".into());
    }

    match manifest.evidence_condition {
        EvidenceConditionV1::LocalRetrieved
            if manifest.source_audio.audio_sha256 != manifest.target_audio_sha256 =>
        {
            return Err("local evidence must originate from the exact target audio digest".into());
        }
        EvidenceConditionV1::YokedRetrieved
            if manifest.source_audio.audio_sha256 == manifest.target_audio_sha256 =>
        {
            return Err("yoked evidence must originate from a different audio digest".into());
        }
        _ => {}
    }

    if manifest.windows.is_empty() {
        return Err("evidence window list must not be empty".into());
    }

    let bytes_per_frame = u64::from(manifest.decoded_pcm.channels)
        .checked_mul(manifest.decoded_pcm.sample_format.bytes_per_sample())
        .ok_or_else(|| "decoded PCM bytes-per-frame overflow".to_string())?;
    let expected_pcm_bytes = manifest
        .decoded_pcm
        .total_frames
        .checked_mul(bytes_per_frame)
        .ok_or_else(|| "decoded PCM expected byte-length overflow".to_string())?;
    if expected_pcm_bytes != manifest.decoded_pcm.byte_length {
        return Err("decoded PCM byte length does not match frame/channel/sample format".into());
    }

    let mut window_ids = BTreeSet::new();
    let mut verified_windows = Vec::with_capacity(manifest.windows.len());
    let mut window_digests = std::collections::BTreeMap::new();

    for window in &manifest.windows {
        require_text(&window.window_id, "window_id")?;
        validate_digest(&window.window_pcm_sha256, "window_pcm_sha256")?;
        if !window_ids.insert(window.window_id.as_str()) {
            return Err(format!("duplicate evidence window ID: {}", window.window_id));
        }
        if window.start_frame >= window.end_frame
            || window.end_frame > manifest.decoded_pcm.total_frames
        {
            return Err(format!(
                "evidence window {} is empty or outside decoded PCM frame range",
                window.window_id
            ));
        }

        let byte_offset = window
            .start_frame
            .checked_mul(bytes_per_frame)
            .ok_or_else(|| format!("window {} byte offset overflow", window.window_id))?;
        let frame_count = window.end_frame - window.start_frame;
        let byte_length = frame_count
            .checked_mul(bytes_per_frame)
            .ok_or_else(|| format!("window {} byte length overflow", window.window_id))?;
        let byte_end = byte_offset
            .checked_add(byte_length)
            .ok_or_else(|| format!("window {} byte end overflow", window.window_id))?;
        if byte_end > decoded_pcm_bytes.len() as u64 {
            return Err(format!("window {} byte range exceeds decoded PCM", window.window_id));
        }
        let start = usize::try_from(byte_offset)
            .map_err(|_| format!("window {} offset cannot fit host usize", window.window_id))?;
        let end = usize::try_from(byte_end)
            .map_err(|_| format!("window {} end cannot fit host usize", window.window_id))?;
        let recomputed_sha256 = sha256_hex(&decoded_pcm_bytes[start..end]);
        let digest_matches = recomputed_sha256 == window.window_pcm_sha256;
        if !digest_matches {
            return Err(format!("window {} payload SHA-256 mismatch", window.window_id));
        }
        window_digests.insert(window.window_id.as_str(), recomputed_sha256.clone());
        verified_windows.push(VerifiedWindowV1 {
            window_id: window.window_id.clone(),
            position: u32::MAX,
            start_frame: window.start_frame,
            end_frame: window.end_frame,
            byte_offset,
            byte_length,
            declared_sha256: window.window_pcm_sha256.clone(),
            recomputed_sha256,
            digest_matches,
        });
    }

    validate_context(manifest, &window_digests)?;
    let context_manifest_sha256 = context_digest(manifest)?;
    if context_manifest_sha256 != manifest.context.context_manifest_sha256 {
        return Err("ordered context manifest SHA-256 mismatch".into());
    }

    for (position, item) in manifest.context.items.iter().enumerate() {
        let index = verified_windows
            .iter()
            .position(|window| window.window_id == item.window_id)
            .ok_or_else(|| format!("context references unknown window: {}", item.window_id))?;
        verified_windows[index].position = u32::try_from(position)
            .map_err(|_| "context item count exceeds u32".to_string())?;
    }
    verified_windows.sort_by_key(|window| window.position);

    let full_manifest_sha256 = canonical_json_sha256(manifest)
        .map_err(|error| format!("failed canonical evidence manifest encoding: {error}"))?;
    Ok(AudioEvidenceVerificationReportV1 {
        schema: REPORT_SCHEMA_V1,
        experiment_unit_id: manifest.experiment_unit_id.clone(),
        query_id: manifest.query_id.clone(),
        evidence_condition: manifest.evidence_condition,
        target_audio_sha256: manifest.target_audio_sha256.clone(),
        source_audio_sha256,
        rights_scope_receipt_sha256: manifest.source_audio.rights_scope_receipt_sha256.clone(),
        decoded_pcm_sha256,
        evidence_context_manifest_sha256: manifest.context.context_manifest_sha256.clone(),
        full_manifest_sha256,
        source_audio_bytes_recomputed: true,
        decoded_pcm_bytes_recomputed: true,
        window_payloads_recomputed: true,
        context_manifest_recomputed: true,
        verified_window_count: verified_windows.len(),
        verified_windows,
        rights_scope_authenticated: false,
        decoder_identity_authenticated: false,
        model_runtime_identity_authenticated: false,
        runner_receipt_verified: false,
        request_body_recomputed: false,
        model_input_attested: false,
    })
}

fn validate_context(
    manifest: &AudioEvidenceWindowManifestV1,
    window_digests: &std::collections::BTreeMap<&str, String>,
) -> Result<(), String> {
    validate_digest(&manifest.context.prompt_sha256, "context.prompt_sha256")?;
    validate_digest(&manifest.context.adapter_sha256, "context.adapter_sha256")?;
    validate_digest(
        &manifest.context.retrieval_policy_sha256,
        "context.retrieval_policy_sha256",
    )?;
    validate_digest(
        &manifest.context.context_manifest_sha256,
        "context.context_manifest_sha256",
    )?;
    if manifest.context.items.len() != manifest.windows.len() {
        return Err("context must contain each declared evidence window exactly once".into());
    }
    let mut seen = BTreeSet::new();
    for (expected_position, item) in manifest.context.items.iter().enumerate() {
        if item.position as usize != expected_position {
            return Err("context item positions must be contiguous, zero-based and ordered".into());
        }
        if !seen.insert(item.window_id.as_str()) {
            return Err(format!("context repeats window ID: {}", item.window_id));
        }
        let declared_window_digest = window_digests
            .get(item.window_id.as_str())
            .ok_or_else(|| format!("context references undeclared window: {}", item.window_id))?;
        if declared_window_digest != &item.payload_sha256 {
            return Err(format!(
                "context payload digest does not match materialized window {}",
                item.window_id
            ));
        }
    }
    Ok(())
}

fn context_digest(manifest: &AudioEvidenceWindowManifestV1) -> Result<String, String> {
    let payload = EvidenceContextDigestPayloadV1 {
        schema: "mel-audio-evidence-context-v1",
        experiment_unit_id: &manifest.experiment_unit_id,
        query_id: &manifest.query_id,
        query_text_sha256: &manifest.query_text_sha256,
        evidence_condition: manifest.evidence_condition,
        target_audio_sha256: &manifest.target_audio_sha256,
        source_audio_sha256: &manifest.source_audio.audio_sha256,
        source_audio_byte_length: manifest.source_audio.byte_length,
        rights_scope_receipt_sha256: &manifest.source_audio.rights_scope_receipt_sha256,
        decoded_pcm_sha256: &manifest.decoded_pcm.pcm_sha256,
        decoded_pcm_byte_length: manifest.decoded_pcm.byte_length,
        sample_rate_hz: manifest.decoded_pcm.sample_rate_hz,
        channels: manifest.decoded_pcm.channels,
        total_frames: manifest.decoded_pcm.total_frames,
        sample_format: manifest.decoded_pcm.sample_format,
        decoder_identity_sha256: &manifest.decoded_pcm.decoder_identity_sha256,
        decode_profile_sha256: &manifest.decoded_pcm.decode_profile_sha256,
        windows: &manifest.windows,
        model_identity_sha256: &manifest.model_identity_sha256,
        runtime_identity_sha256: &manifest.runtime_identity_sha256,
        decoding_config_sha256: &manifest.decoding_config_sha256,
        prompt_sha256: &manifest.context.prompt_sha256,
        adapter_sha256: &manifest.context.adapter_sha256,
        retrieval_policy_sha256: &manifest.context.retrieval_policy_sha256,
        items: &manifest.context.items,
    };
    canonical_json_sha256(&payload)
        .map_err(|error| format!("failed canonical context manifest encoding: {error}"))
}

fn validate_declared_length(
    bytes: &[u8],
    declared_length: u64,
    label: &str,
) -> Result<(), String> {
    let actual_length = u64::try_from(bytes.len())
        .map_err(|_| format!("{label} input length cannot fit u64"))?;
    if actual_length != declared_length {
        return Err(format!("{label} byte length mismatch"));
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::evidence_digest::sha256_hex;

    const SOURCE_AUDIO: &[u8] = b"MEL-GROUND-009 synthetic source audio bytes v1\n";
    const PCM: &[u8] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    fn manifest() -> AudioEvidenceWindowManifestV1 {
        let mut manifest = AudioEvidenceWindowManifestV1 {
            schema: MANIFEST_SCHEMA_V1.into(),
            experiment_unit_id: "mel-grounding-evidence-window-fixture-001".into(),
            query_id: "piano-entry-query-001".into(),
            query_text: "Identify every piano entrance in the supplied recording.".into(),
            query_text_sha256: sha256_hex(b"Identify every piano entrance in the supplied recording."),
            model_identity_sha256: "7777777777777777777777777777777777777777777777777777777777777777".into(),
            runtime_identity_sha256: "8888888888888888888888888888888888888888888888888888888888888888".into(),
            decoding_config_sha256: "9999999999999999999999999999999999999999999999999999999999999999".into(),
            evidence_condition: EvidenceConditionV1::LocalRetrieved,
            target_audio_sha256: sha256_hex(SOURCE_AUDIO),
            source_audio: SourceAudioArtifactV1 {
                audio_sha256: sha256_hex(SOURCE_AUDIO),
                byte_length: SOURCE_AUDIO.len() as u64,
                rights_scope_receipt_sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into(),
            },
            decoded_pcm: DecodedPcmArtifactV1 {
                pcm_sha256: sha256_hex(PCM),
                byte_length: PCM.len() as u64,
                sample_rate_hz: 48_000,
                channels: 2,
                total_frames: 8,
                sample_format: PcmSampleFormatV1::S16Le,
                decoder_identity_sha256: "1111111111111111111111111111111111111111111111111111111111111111".into(),
                decode_profile_sha256: "2222222222222222222222222222222222222222222222222222222222222222".into(),
            },
            windows: vec![
                EvidenceWindowV1 {
                    window_id: "window-001".into(),
                    start_frame: 1,
                    end_frame: 3,
                    window_pcm_sha256: sha256_hex(&PCM[4..12]),
                },
                EvidenceWindowV1 {
                    window_id: "window-002".into(),
                    start_frame: 5,
                    end_frame: 8,
                    window_pcm_sha256: sha256_hex(&PCM[20..32]),
                },
            ],
            context: EvidenceContextV1 {
                prompt_sha256: "3333333333333333333333333333333333333333333333333333333333333333".into(),
                adapter_sha256: "4444444444444444444444444444444444444444444444444444444444444444".into(),
                retrieval_policy_sha256: "5555555555555555555555555555555555555555555555555555555555555555".into(),
                items: vec![
                    EvidenceContextItemV1 {
                        position: 0,
                        window_id: "window-001".into(),
                        payload_sha256: sha256_hex(&PCM[4..12]),
                    },
                    EvidenceContextItemV1 {
                        position: 1,
                        window_id: "window-002".into(),
                        payload_sha256: sha256_hex(&PCM[20..32]),
                    },
                ],
                context_manifest_sha256: String::new(),
            },
        };
        manifest.context.context_manifest_sha256 = context_digest(&manifest).unwrap();
        manifest
    }

    #[test]
    fn exact_source_pcm_window_and_context_digests_verify() {
        let manifest = manifest();
        let report = verify_audio_evidence_manifest(&manifest, SOURCE_AUDIO, PCM).unwrap();
        assert!(report.source_audio_bytes_recomputed);
        assert!(report.decoded_pcm_bytes_recomputed);
        assert!(report.window_payloads_recomputed);
        assert!(report.context_manifest_recomputed);
        assert_eq!(report.verified_window_count, 2);
        assert_eq!(report.verified_windows[0].byte_offset, 4);
        assert_eq!(report.verified_windows[0].byte_length, 8);
        assert_eq!(report.verified_windows[1].byte_offset, 20);
        assert_eq!(report.verified_windows[1].byte_length, 12);
        assert_eq!(report.full_manifest_sha256.len(), 64);
        assert_eq!(report.evidence_context_manifest_sha256.len(), 64);
        assert!(!report.rights_scope_authenticated);
        assert!(!report.decoder_identity_authenticated);
        assert!(!report.model_runtime_identity_authenticated);
        assert!(!report.runner_receipt_verified);
        assert!(!report.request_body_recomputed);
        assert!(!report.model_input_attested);
    }

    #[test]
    fn one_byte_source_mutation_is_rejected() {
        let mut bytes = SOURCE_AUDIO.to_vec();
        bytes[0] ^= 1;
        assert!(verify_audio_evidence_manifest(&manifest(), &bytes, PCM).is_err());
    }

    #[test]
    fn one_sample_pcm_mutation_is_rejected() {
        let mut bytes = PCM.to_vec();
        bytes[5] ^= 1;
        assert!(verify_audio_evidence_manifest(&manifest(), SOURCE_AUDIO, &bytes).is_err());
    }

    #[test]
    fn local_evidence_must_bind_exact_target_audio() {
        let mut input = manifest();
        input.target_audio_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn yoked_evidence_must_bind_different_target_audio() {
        let mut input = manifest();
        input.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
        input.target_audio_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_ok());
    }

    #[test]
    fn window_boundaries_are_half_open_and_frame_aligned() {
        let mut input = manifest();
        input.windows[0].start_frame = 0;
        input.windows[0].end_frame = 0;
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.windows[0].end_frame = 9;
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn context_must_contain_each_window_once_in_exact_order() {
        let mut input = manifest();
        input.context.items.swap(0, 1);
        input.context.context_manifest_sha256 = context_digest(&input).unwrap();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.context.items[1].position = 4;
        input.context.context_manifest_sha256 = context_digest(&input).unwrap();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn context_digest_detects_changed_query_model_runtime_or_decode_context() {
        let mut input = manifest();
        input.query_text = "different query".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.query_text_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.model_identity_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.runtime_identity_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.decoding_config_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.context.prompt_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.context.adapter_sha256 =
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn context_payload_cannot_substitute_another_window_digest() {
        let mut input = manifest();
        input.context.items[0].payload_sha256 =
            "7777777777777777777777777777777777777777777777777777777777777777".into();
        input.context.context_manifest_sha256 = context_digest(&input).unwrap();
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn malformed_lengths_and_clock_metadata_are_rejected() {
        let mut input = manifest();
        input.decoded_pcm.byte_length += 1;
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());

        input = manifest();
        input.decoded_pcm.channels = 0;
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn multiplication_overflow_is_rejected_before_slicing() {
        let mut input = manifest();
        input.decoded_pcm.total_frames = u64::MAX;
        input.decoded_pcm.byte_length = u64::MAX;
        let tiny = &PCM[..];
        assert!(verify_audio_evidence_manifest(&input, SOURCE_AUDIO, tiny).is_err());
    }
}

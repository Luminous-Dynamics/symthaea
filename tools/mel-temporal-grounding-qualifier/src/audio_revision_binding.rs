//! Bind recomputed audio evidence windows to a scored temporal-revision case.
//!
//! This adapter runs the byte verifier and the canonical temporal revision
//! scorer in the same evaluation path. It proves that the supplied bytes,
//! query, source lineage, model/runtime configuration and selected window
//! belong to the same experiment case. It does not prove that the model
//! service accepted or internally consumed those bytes.

use serde::Serialize;

use crate::audio_evidence_window::{
    verify_audio_evidence_manifest, AudioEvidenceVerificationReportV1,
    AudioEvidenceWindowManifestV1, EvidenceConditionV1 as AudioEvidenceConditionV1,
};
use crate::evidence_digest::sha256_hex;
use crate::temporal_revision::{
    audit_revision_v2, compare_matched_evidence_arms,
    EvidenceConditionV1 as RevisionEvidenceConditionV1, MatchedEvidenceContrastV1,
    RevisionCaseV2, RevisionReportV1,
};

const REPORT_SCHEMA_V1: &str = "mel-audio-bound-revision-report-v1";

pub struct AudioEvidenceRevisionArmInput<'a> {
    pub revision_case: &'a RevisionCaseV2,
    /// Must be None for NoEvidence and Some for LocalRetrieved/YokedRetrieved.
    pub evidence_manifest: Option<&'a AudioEvidenceWindowManifestV1>,
    pub source_audio_bytes: Option<&'a [u8]>,
    pub decoded_pcm_bytes: Option<&'a [u8]>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VerifiedAudioEvidenceTripletReportV1 {
    pub contrast: MatchedEvidenceContrastV1,
    pub no_evidence_revision_report: RevisionReportV1,
    pub local_audio_bound_report: AudioBoundRevisionAuditV1,
    pub yoked_audio_bound_report: AudioBoundRevisionAuditV1,
    pub no_evidence_arm_has_no_materialization: bool,
    pub local_evidence_bytes_verified: bool,
    pub yoked_evidence_bytes_verified: bool,
    /// The byte/context binding is verified; model-internal consumption remains false.
    pub model_input_attested: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AudioBoundRevisionAuditV1 {
    pub schema: &'static str,
    pub revision_report: RevisionReportV1,
    pub audio_evidence_report: AudioEvidenceVerificationReportV1,
    pub experiment_unit_binding_verified: bool,
    pub source_lineage_binding_verified: bool,
    pub query_binding_verified: bool,
    pub target_audio_binding_verified: bool,
    pub model_runtime_binding_verified: bool,
    pub prompt_adapter_policy_binding_verified: bool,
    pub selected_window_binding_verified: bool,
    pub materialized_audio_bytes_bound_to_revision_case: bool,
    /// This remains false until a trusted model/runtime receipt is verified.
    pub model_input_attested: bool,
}

/// Verify an entire NoEvidence / LocalRetrieved / YokedRetrieved triplet
/// before computing its paired evidence-ablation contrast. Unverified metadata
/// alone cannot produce a confirmatory contrast through this API.
pub fn audit_verified_audio_evidence_triplet(
    no_evidence: AudioEvidenceRevisionArmInput<'_>,
    local: AudioEvidenceRevisionArmInput<'_>,
    yoked: AudioEvidenceRevisionArmInput<'_>,
) -> Result<VerifiedAudioEvidenceTripletReportV1, String> {
    let no_case = no_evidence.revision_case;
    if no_case.evidence_condition != RevisionEvidenceConditionV1::NoEvidence
        || no_case.evidence.is_some()
        || no_evidence.evidence_manifest.is_some()
        || no_evidence.source_audio_bytes.is_some()
        || no_evidence.decoded_pcm_bytes.is_some()
    {
        return Err("NoEvidence arm must not include evidence references, manifests or bytes".into());
    }
    let no_evidence_revision_report = audit_revision_v2(no_case)?;

    let local_case = local.revision_case;
    let local_manifest = local
        .evidence_manifest
        .ok_or_else(|| "LocalRetrieved arm requires an audio evidence manifest".to_string())?;
    let local_source = local
        .source_audio_bytes
        .ok_or_else(|| "LocalRetrieved arm requires raw source audio bytes".to_string())?;
    let local_pcm = local
        .decoded_pcm_bytes
        .ok_or_else(|| "LocalRetrieved arm requires raw decoded PCM bytes".to_string())?;
    if local_case.evidence_condition != RevisionEvidenceConditionV1::LocalRetrieved {
        return Err("local arm revision condition must be LocalRetrieved".into());
    }
    let local_audio_bound_report =
        audit_revision_with_audio_evidence(local_case, local_manifest, local_source, local_pcm)?;

    let yoked_case = yoked.revision_case;
    let yoked_manifest = yoked
        .evidence_manifest
        .ok_or_else(|| "YokedRetrieved arm requires an audio evidence manifest".to_string())?;
    let yoked_source = yoked
        .source_audio_bytes
        .ok_or_else(|| "YokedRetrieved arm requires raw source audio bytes".to_string())?;
    let yoked_pcm = yoked
        .decoded_pcm_bytes
        .ok_or_else(|| "YokedRetrieved arm requires raw decoded PCM bytes".to_string())?;
    if yoked_case.evidence_condition != RevisionEvidenceConditionV1::YokedRetrieved {
        return Err("yoked arm revision condition must be YokedRetrieved".into());
    }
    let yoked_audio_bound_report =
        audit_revision_with_audio_evidence(yoked_case, yoked_manifest, yoked_source, yoked_pcm)?;

    let contrast_reports =
        compare_matched_evidence_arms(&[no_case.clone(), local_case.clone(), yoked_case.clone()])?;
    if contrast_reports.len() != 1 {
        return Err("verified evidence triplet must produce exactly one matched contrast".into());
    }
    let contrast = contrast_reports
        .into_iter()
        .next()
        .ok_or_else(|| "verified evidence contrast unexpectedly missing".to_string())?;

    Ok(VerifiedAudioEvidenceTripletReportV1 {
        contrast,
        no_evidence_revision_report,
        local_audio_bound_report,
        yoked_audio_bound_report,
        no_evidence_arm_has_no_materialization: true,
        local_evidence_bytes_verified: true,
        yoked_evidence_bytes_verified: true,
        model_input_attested: false,
    })
}

pub fn audit_revision_with_audio_evidence(
    revision_case: &RevisionCaseV2,
    evidence_manifest: &AudioEvidenceWindowManifestV1,
    source_audio_bytes: &[u8],
    decoded_pcm_bytes: &[u8],
) -> Result<AudioBoundRevisionAuditV1, String> {
    let revision_report = audit_revision_v2(revision_case)?;
    let evidence = revision_case
        .evidence
        .as_ref()
        .ok_or_else(|| "audio evidence binding requires a revision evidence reference".to_string())?;

    let expected_audio_condition = match revision_case.evidence_condition {
        RevisionEvidenceConditionV1::NoEvidence => {
            return Err("NoEvidence revision cases must not provide an audio-window manifest".into());
        }
        RevisionEvidenceConditionV1::LocalRetrieved => AudioEvidenceConditionV1::LocalRetrieved,
        RevisionEvidenceConditionV1::YokedRetrieved => AudioEvidenceConditionV1::YokedRetrieved,
    };
    if evidence_manifest.evidence_condition != expected_audio_condition {
        return Err("revision evidence condition differs from audio-window manifest".into());
    }

    let evidence_report = verify_audio_evidence_manifest(
        evidence_manifest,
        source_audio_bytes,
        decoded_pcm_bytes,
    )?;

    if evidence_manifest.experiment_unit_id != revision_case.experiment_unit_id {
        return Err("audio evidence experiment unit differs from revision case".into());
    }
    if evidence_manifest.target_source_lineage_id != revision_case.source_lineage_id {
        return Err("audio evidence target source lineage differs from revision case".into());
    }
    match expected_audio_condition {
        AudioEvidenceConditionV1::LocalRetrieved
            if evidence_manifest.source_lineage_id != revision_case.source_lineage_id =>
        {
            return Err("local evidence source lineage differs from target source lineage".into());
        }
        AudioEvidenceConditionV1::YokedRetrieved
            if evidence_manifest.source_lineage_id == revision_case.source_lineage_id =>
        {
            return Err("yoked evidence must come from a distinct source-work lineage".into());
        }
        _ => {}
    }
    if evidence_manifest.query_id != revision_case.truth.query_id
        || evidence_manifest.query_text != revision_case.truth.query_text
        || evidence_manifest.query_text_sha256
            != sha256_hex(revision_case.truth.query_text.as_bytes())
    {
        return Err("audio evidence query identity differs from revision case".into());
    }
    if evidence_manifest.target_audio_sha256 != revision_case.truth.artifact.audio_sha256 {
        return Err("audio evidence target artifact differs from revision scoring artifact".into());
    }
    if evidence_manifest.model_identity_sha256 != revision_case.model_identity_sha256
        || evidence_manifest.runtime_identity_sha256 != revision_case.runtime_identity_sha256
        || evidence_manifest.decoding_config_sha256 != revision_case.decoding_config_sha256
    {
        return Err("audio evidence model/runtime/decoding identity differs from revision case".into());
    }
    if evidence_manifest.context.prompt_sha256 != revision_case.prompt_sha256
        || evidence_manifest.context.adapter_sha256 != revision_case.adapter_sha256
        || evidence_manifest.context.retrieval_policy_sha256 != evidence.retrieval_config_sha256
    {
        return Err("audio evidence prompt/adapter/retrieval policy differs from revision case".into());
    }
    if evidence.source_audio_sha256 != evidence_report.source_audio_sha256
        || evidence.source_sample_rate_hz != evidence_manifest.decoded_pcm.sample_rate_hz
    {
        return Err("revision evidence source identity or sample rate differs from verified bytes".into());
    }

    let selected_window_match_count = evidence_manifest
        .windows
        .iter()
        .filter(|window| {
            window.start_frame == evidence.start_frame
                && window.end_frame == evidence.end_frame
                && window.window_pcm_sha256 == evidence.evidence_payload_sha256
        })
        .count();
    if selected_window_match_count != 1 {
        return Err(
            "revision evidence window must match exactly one recomputed manifest window".into(),
        );
    }

    match expected_audio_condition {
        AudioEvidenceConditionV1::LocalRetrieved => {
            let artifact = &revision_case.truth.artifact;
            let pcm = &evidence_manifest.decoded_pcm;
            if evidence_manifest.source_audio.audio_sha256 != artifact.audio_sha256
                || pcm.sample_rate_hz != artifact.sample_rate_hz
                || pcm.channels != artifact.channels
                || pcm.total_frames != artifact.total_frames
            {
                return Err("local evidence PCM clock differs from the exact scored target audio".into());
            }
        }
        AudioEvidenceConditionV1::YokedRetrieved => {
            if evidence_manifest.source_audio.audio_sha256 == revision_case.truth.artifact.audio_sha256 {
                return Err("yoked evidence reuses the target recording".into());
            }
        }
    }

    Ok(AudioBoundRevisionAuditV1 {
        schema: REPORT_SCHEMA_V1,
        revision_report,
        audio_evidence_report: evidence_report,
        experiment_unit_binding_verified: true,
        source_lineage_binding_verified: true,
        query_binding_verified: true,
        target_audio_binding_verified: true,
        model_runtime_binding_verified: true,
        prompt_adapter_policy_binding_verified: true,
        selected_window_binding_verified: true,
        materialized_audio_bytes_bound_to_revision_case: true,
        model_input_attested: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::audio_evidence_in_toto::build_audio_evidence_in_toto_statement;
    use crate::audio_evidence_window::recompute_audio_context_manifest_sha256;
    use crate::evidence_digest::sha256_hex;
    use crate::temporal_grounding::{
        AudioArtifactClockV1, FrameIntervalV1, GroundTruthStateV1,
        GroundingPredictionStateV1, TemporalGroundTruthV1, TemporalGroundingPlanV1,
        TemporalGroundingPredictionV1, TEMPORAL_GROUNDING_EVALUATOR_V1,
        TEMPORAL_GROUNDING_SCHEMA_V1,
    };

    const SOURCE_AUDIO: &[u8] = b"MEL-GROUND-009 synthetic source audio bytes v1\n";
    const PCM: &[u8] = &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    ];

    fn revision_case() -> RevisionCaseV2 {
        let audio = AudioArtifactClockV1 {
            audio_sha256: "b12dbd7c993ac1f0ddf6c0cef16c605ce48da060738b83837e39bdadf9df83ef".into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 8,
        };
        let intervals = vec![FrameIntervalV1 {
            start_frame: 1,
            end_frame: 3,
        }];
        let query_text = "Identify every piano entrance in the supplied recording.";
        let prediction = TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: "piano-entry-query-001".into(),
            artifact: audio.clone(),
            state: GroundingPredictionStateV1::Occurrences {
                intervals: intervals.clone(),
            },
        };
        RevisionCaseV2 {
            schema: "mel-temporal-revision-case-v2".into(),
            case_id: "audio-bound-revision-case-001".into(),
            experiment_unit_id: "mel-grounding-evidence-window-fixture-001".into(),
            source_lineage_id: "synthetic-source-lineage-001".into(),
            model_identity_sha256: "7777777777777777777777777777777777777777777777777777777777777777".into(),
            prompt_sha256: "3333333333333333333333333333333333333333333333333333333333333333".into(),
            adapter_sha256: "4444444444444444444444444444444444444444444444444444444444444444".into(),
            runtime_identity_sha256: "8888888888888888888888888888888888888888888888888888888888888888".into(),
            decoding_config_sha256: "9999999999999999999999999999999999999999999999999999999999999999".into(),
            max_output_tokens: 512,
            actual_output_tokens: 128,
            max_wall_time_ms: 30_000,
            elapsed_wall_time_ms: 4_000,
            evidence_condition: RevisionEvidenceConditionV1::LocalRetrieved,
            evidence: Some(crate::temporal_revision::RevisionEvidenceV1 {
                source_audio_sha256: audio.audio_sha256.clone(),
                source_sample_rate_hz: 48_000,
                evidence_payload_sha256: "3fc77626f4c70d9e28aa4c1cd901fcf28a7a6a690788342dcb45884f5dc3acb0".into(),
                retrieval_config_sha256: "5555555555555555555555555555555555555555555555555555555555555555".into(),
                start_frame: 1,
                end_frame: 3,
            }),
            actions: vec![crate::temporal_revision::RevisionActionV1::Other],
            truth: TemporalGroundTruthV1 {
                schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
                query_id: "piano-entry-query-001".into(),
                query_text: query_text.into(),
                artifact: audio,
                annotation_policy_version: "synthetic-annotation-v1".into(),
                annotation_set_sha256: "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd".into(),
                annotator_count: 2,
                state: GroundTruthStateV1::Present,
                intervals,
                status_reason: None,
            },
            plan: TemporalGroundingPlanV1 {
                evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
                minimum_interval_iou: 0.5,
            },
            draft_prediction: prediction.clone(),
            final_prediction: prediction,
        }
    }

    fn evidence_manifest() -> AudioEvidenceWindowManifestV1 {
        serde_json::from_str(include_str!("../fixtures/mel-audio-evidence-window.synthetic.json"))
            .unwrap()
    }

    #[test]
    fn exact_audio_materialization_and_revision_metadata_join() {
        let report = audit_revision_with_audio_evidence(
            &revision_case(),
            &evidence_manifest(),
            SOURCE_AUDIO,
            PCM,
        )
        .unwrap();
        assert_eq!(report.revision_report.final_score.f1, Some(1.0));
        assert!(report.materialized_audio_bytes_bound_to_revision_case);
        assert!(report.experiment_unit_binding_verified);
        assert!(report.source_lineage_binding_verified);
        assert!(report.query_binding_verified);
        assert!(report.target_audio_binding_verified);
        assert!(report.model_runtime_binding_verified);
        assert!(report.prompt_adapter_policy_binding_verified);
        assert!(report.selected_window_binding_verified);
        assert_eq!(report.audio_evidence_report.verified_window_count, 2);
        assert!(!report.model_input_attested);
    }

    #[test]
    fn yoked_audio_bytes_bind_to_the_target_query_without_relabeling_target_identity() {
        let mut case = revision_case();
        case.evidence_condition = RevisionEvidenceConditionV1::YokedRetrieved;
        let mut manifest = evidence_manifest();
        let yoked_source = b"MEL-GROUND-009 synthetic yoked source audio v1\\n";
        let yoked_pcm: Vec<u8> = PCM.iter().map(|sample| sample ^ 0x5a).collect();
        let yoked_digest = sha256_hex(yoked_source);
        manifest.evidence_condition = AudioEvidenceConditionV1::YokedRetrieved;
        manifest.source_audio.audio_sha256 = yoked_digest.clone();
        manifest.source_audio.byte_length = yoked_source.len() as u64;
        manifest.source_lineage_id = "synthetic-yoked-source-lineage-001".into();
        manifest.decoded_pcm.pcm_sha256 = sha256_hex(&yoked_pcm);
        manifest.windows[0].window_pcm_sha256 = sha256_hex(&yoked_pcm[4..12]);
        manifest.windows[1].window_pcm_sha256 = sha256_hex(&yoked_pcm[20..32]);
        manifest.context.items[0].payload_sha256 = manifest.windows[0].window_pcm_sha256.clone();
        manifest.context.items[1].payload_sha256 = manifest.windows[1].window_pcm_sha256.clone();
        case.evidence.as_mut().unwrap().source_audio_sha256 = yoked_digest;
        case.evidence.as_mut().unwrap().evidence_payload_sha256 =
            manifest.windows[0].window_pcm_sha256.clone();
        manifest.context.context_manifest_sha256 =
            recompute_audio_context_manifest_sha256(&manifest).unwrap();

        let report = audit_revision_with_audio_evidence(
            &case,
            &manifest,
            yoked_source,
            &yoked_pcm,
        )
        .unwrap();
        assert_eq!(report.audio_evidence_report.evidence_condition, AudioEvidenceConditionV1::YokedRetrieved);
        assert_ne!(
            report.audio_evidence_report.source_audio_sha256,
            report.audio_evidence_report.target_audio_sha256
        );
        assert_ne!(
            report.audio_evidence_report.source_lineage_id,
            report.audio_evidence_report.target_source_lineage_id
        );
        assert!(report.materialized_audio_bytes_bound_to_revision_case);
        assert!(!report.model_input_attested);

        let payload = build_audio_evidence_in_toto_statement(
            &case,
            &manifest,
            yoked_source,
            &yoked_pcm,
        )
        .unwrap();
        let statement: serde_json::Value =
            serde_json::from_str(&payload.statement_json).unwrap();
        assert_eq!(
            statement["predicate"]["target_source_lineage_id"],
            "synthetic-source-lineage-001"
        );
        assert_eq!(
            statement["predicate"]["evidence_source_lineage_id"],
            "synthetic-yoked-source-lineage-001"
        );
        assert_eq!(statement["predicate"]["target_audio_bytes_recomputed"], false);
        assert_eq!(statement["predicate"]["model_input_attested"], false);
    }

    #[test]
    fn unsigned_in_toto_statement_binds_recomputed_audio_and_revision_subjects() {
        let output = build_audio_evidence_in_toto_statement(
            &revision_case(),
            &evidence_manifest(),
            SOURCE_AUDIO,
            PCM,
        )
        .unwrap();

        assert_eq!(output.payload_type, "application/vnd.in-toto+json");
        assert!(!output.dsse_envelope_created);
        assert!(!output.signature_verified);
        assert_eq!(
            output.statement_payload_sha256,
            sha256_hex(output.statement_json.as_bytes())
        );

        let statement: serde_json::Value =
            serde_json::from_str(&output.statement_json).unwrap();
        assert_eq!(
            statement["_type"],
            "https://in-toto.io/Statement/v1"
        );
        assert_eq!(
            statement["predicateType"],
            "https://luminousdynamics.io/attestations/mel/audio-evidence-materialization/v1"
        );
        let subjects = statement["subject"].as_array().unwrap();
        let subject_names: Vec<_> = subjects
            .iter()
            .filter_map(|subject| subject["name"].as_str())
            .collect();
        assert!(subject_names.iter().any(|name| name.starts_with("target-audio-reference/")));
        assert!(subject_names.iter().any(|name| name.starts_with("source-audio/")));
        assert!(subject_names.iter().any(|name| name == &"audio-window/window-001"));
        assert!(subject_names.iter().any(|name| name == &"audio-window/window-002"));
        assert!(subject_names.iter().any(|name| name.starts_with("audio-context-manifest/")));
        assert!(subject_names.iter().any(|name| name.starts_with("audio-evidence-manifest/")));
        assert!(subject_names.iter().any(|name| name.starts_with("temporal-revision-report/")));

        assert_eq!(
            statement["predicate"]["evidence_level"],
            "DigestLinked"
        );
        assert_eq!(
            statement["predicate"]["target_source_lineage_id"],
            "synthetic-source-lineage-001"
        );
        assert_eq!(
            statement["predicate"]["evidence_source_lineage_id"],
            "synthetic-source-lineage-001"
        );
        assert_eq!(statement["predicate"]["target_audio_bytes_recomputed"], true);
        assert_eq!(statement["predicate"]["source_audio_bytes_recomputed"], true);
        assert_eq!(statement["predicate"]["sample_rate_hz"], 48_000);
        assert_eq!(statement["predicate"]["channels"], 2);
        assert_eq!(statement["predicate"]["total_frames"], 8);
        assert_eq!(statement["predicate"]["sample_format"], "S16Le");
        assert_eq!(statement["predicate"]["max_output_tokens"], 512);
        assert_eq!(statement["predicate"]["actual_output_tokens"], 128);
        assert_eq!(statement["predicate"]["max_wall_time_ms"], 30_000);
        assert_eq!(statement["predicate"]["elapsed_wall_time_ms"], 4_000);
        assert_eq!(statement["predicate"]["window_payloads_recomputed"], true);
        assert_eq!(statement["predicate"]["rights_scope_authenticated"], false);
        assert_eq!(statement["predicate"]["runner_receipt_verified"], false);
        assert_eq!(statement["predicate"]["model_input_attested"], false);
    }

    #[test]
    fn mismatched_source_lineage_is_rejected_even_when_all_bytes_hash() {
        let mut case = revision_case();
        case.source_lineage_id = "different-lineage".into();
        assert!(
            audit_revision_with_audio_evidence(
                &case,
                &evidence_manifest(),
                SOURCE_AUDIO,
                PCM,
            )
            .is_err()
        );
    }

    #[test]
    fn mismatched_model_prompt_or_adapter_is_rejected() {
        let manifest = evidence_manifest();
        let mut case = revision_case();
        case.model_identity_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(audit_revision_with_audio_evidence(&case, &manifest, SOURCE_AUDIO, PCM).is_err());

        case = revision_case();
        case.prompt_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(audit_revision_with_audio_evidence(&case, &manifest, SOURCE_AUDIO, PCM).is_err());

        case = revision_case();
        case.adapter_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();
        assert!(audit_revision_with_audio_evidence(&case, &manifest, SOURCE_AUDIO, PCM).is_err());
    }

    #[test]
    fn selected_window_must_match_exact_frames_and_payload_digest() {
        let mut case = revision_case();
        case.evidence.as_mut().unwrap().start_frame = 2;
        assert!(
            audit_revision_with_audio_evidence(
                &case,
                &evidence_manifest(),
                SOURCE_AUDIO,
                PCM,
            )
            .is_err()
        );
    }

    #[test]
    fn duplicate_matching_window_references_are_rejected_as_ambiguous() {
        let case = revision_case();
        let mut manifest = evidence_manifest();
        let mut duplicate = manifest.windows[0].clone();
        duplicate.window_id = "window-duplicate".into();
        manifest.windows.push(duplicate);
        manifest.context.items.push(crate::audio_evidence_window::EvidenceContextItemV1 {
            position: 2,
            window_id: "window-duplicate".into(),
            payload_sha256: manifest.windows[2].window_pcm_sha256.clone(),
        });
        manifest.context.context_manifest_sha256 =
            recompute_audio_context_manifest_sha256(&manifest).unwrap();
        assert!(
            audit_revision_with_audio_evidence(&case, &manifest, SOURCE_AUDIO, PCM).is_err()
        );
    }

    #[test]
    fn no_evidence_case_cannot_be_promoted_by_a_materialization_manifest() {
        let mut case = revision_case();
        case.evidence_condition = RevisionEvidenceConditionV1::NoEvidence;
        case.evidence = None;
        assert!(
            audit_revision_with_audio_evidence(
                &case,
                &evidence_manifest(),
                SOURCE_AUDIO,
                PCM,
            )
            .is_err()
        );
    }
}

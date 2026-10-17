// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-scoped product policy for repertoire operations.
//!
//! This module does not decide copyright law. It blocks captured streaming
//! audio by product policy and fails closed when operation-specific evidence
//! is missing. User assertions are never treated as verified grants.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepertoireActionV1 {
    OpenExternalPlayer,
    JoinLiveJam,
    GenerateOriginalFromBrief,
    AnalyzeSourceAudio,
    TranscribeComposition,
    CreateArrangement,
    PrivateRehearsal,
    PerformCoverPublicly,
    RecordCover,
    PublishCoverAudio,
    SynchronizeCoverToVideo,
    ExportSongTranscription,
    TrainModel,
    RedistributeSourceAudio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsLayerV1 {
    Composition,
    SoundRecording,
    SourcePlatformTerms,
    Arrangement,
    PerformerConsent,
    PublicPerformance,
    CoverRecording,
    RecordingDistribution,
    Synchronization,
    Redistribution,
    ModelTraining,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RightsEvidenceStatusV1 {
    VerifiedGrant,
    UserAssertion,
    ExplicitlyDenied,
    Unknown,
}

/// StreamingPlatformAudio means audio bytes reached the analysis path.
/// ExternalPlayerLink is metadata/URL only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepertoireSourceV1 {
    ExternalPlayerLink,
    StreamingPlatformAudio,
    UserSelectedLocalFile,
    LicensedCatalogAsset,
    CreatorOwnedAsset,
    VerifiedPublicDomainAsset,
    LiveRoomInput,
    UserAuthoredSymbolicWork,
    TextBrief,
}

impl RepertoireSourceV1 {
    fn is_audio_asset(self) -> bool {
        matches!(
            self,
            Self::UserSelectedLocalFile
                | Self::LicensedCatalogAsset
                | Self::CreatorOwnedAsset
                | Self::VerifiedPublicDomainAsset
        )
    }

    fn needs_platform_terms(self) -> bool {
        matches!(self, Self::LicensedCatalogAsset)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepertoireProcessingModeV1 {
    EphemeralLocal,
    PersistentLocal,
    LicensedService,
    Cloud,
}

/// evidence_ref should identify an auditable grant/contract/attestation, not
/// contain source audio. Verified grants require a current territory scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RightsGrantV1 {
    pub layer: RightsLayerV1,
    pub status: RightsEvidenceStatusV1,
    pub permitted_actions: Vec<RepertoireActionV1>,
    pub territory: Option<String>,
    pub valid_until_unix_seconds: Option<u64>,
    pub evidence_ref: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepertoireUseRequestV1 {
    pub action: RepertoireActionV1,
    pub source: RepertoireSourceV1,
    pub processing_mode: RepertoireProcessingModeV1,
    pub territory: Option<String>,
    pub now_unix_seconds: u64,
    pub uses_source_master_in_output: bool,
}

/// These outcomes describe product policy, not a legal determination.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RepertoireGateOutcomeV1 {
    AllowedWithoutProtectedSourceUse,
    AllowedByRecordedGrant,
    AllowedInExplicitLocalMode,
    BlockedByProductPolicy,
    BlockedBySourceTerms,
    NeedsRightsReview { missing_layers: Vec<RightsLayerV1> },
}

/// Evaluate whether product-policy scopes are supported by the supplied evidence.
///
/// User assertions unlock only narrow ephemeral-local analysis/private rehearsal
/// or consented live-room joining. Cover recording, publication, performance,
/// full-transcription export, model training and redistribution require verified
/// grants. This is not legal advice and does not infer statutory exceptions.
pub fn evaluate_repertoire_use_v1(
    request: &RepertoireUseRequestV1,
    grants: &[RightsGrantV1],
) -> RepertoireGateOutcomeV1 {
    if request.source == RepertoireSourceV1::StreamingPlatformAudio {
        return RepertoireGateOutcomeV1::BlockedByProductPolicy;
    }

    if request.source == RepertoireSourceV1::ExternalPlayerLink {
        return if request.action == RepertoireActionV1::OpenExternalPlayer {
            RepertoireGateOutcomeV1::AllowedWithoutProtectedSourceUse
        } else {
            RepertoireGateOutcomeV1::BlockedByProductPolicy
        };
    }

    if request.action == RepertoireActionV1::OpenExternalPlayer {
        return RepertoireGateOutcomeV1::BlockedByProductPolicy;
    }

    if request.action == RepertoireActionV1::GenerateOriginalFromBrief {
        return if request.source == RepertoireSourceV1::TextBrief
            && !request.uses_source_master_in_output
        {
            RepertoireGateOutcomeV1::AllowedWithoutProtectedSourceUse
        } else {
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::Composition],
            }
        };
    }

    if request.uses_source_master_in_output && !request.source.is_audio_asset() {
        return RepertoireGateOutcomeV1::BlockedByProductPolicy;
    }

    if request.action == RepertoireActionV1::JoinLiveJam
        && request.source != RepertoireSourceV1::LiveRoomInput
    {
        return RepertoireGateOutcomeV1::BlockedByProductPolicy;
    }

    if request.source == RepertoireSourceV1::LiveRoomInput
        && request.action != RepertoireActionV1::JoinLiveJam
    {
        return RepertoireGateOutcomeV1::NeedsRightsReview {
            missing_layers: vec![RightsLayerV1::PerformerConsent],
        };
    }

    let required = required_layers(request);
    if has_current_source_terms_denial(request, grants) {
        return RepertoireGateOutcomeV1::BlockedBySourceTerms;
    }

    let allow_user_assertion = allows_explicit_local_assertion(request);
    let mut missing = Vec::new();
    let mut relied_on_user_assertion = false;

    for layer in required {
        if has_usable_grant(
            layer,
            RightsEvidenceStatusV1::VerifiedGrant,
            request,
            grants,
        ) {
            continue;
        }

        if allow_user_assertion
            && has_usable_grant(
                layer,
                RightsEvidenceStatusV1::UserAssertion,
                request,
                grants,
            )
        {
            relied_on_user_assertion = true;
            continue;
        }

        if !missing.contains(&layer) {
            missing.push(layer);
        }
    }

    if !missing.is_empty() {
        return RepertoireGateOutcomeV1::NeedsRightsReview {
            missing_layers: missing,
        };
    }

    if relied_on_user_assertion {
        RepertoireGateOutcomeV1::AllowedInExplicitLocalMode
    } else {
        RepertoireGateOutcomeV1::AllowedByRecordedGrant
    }
}

fn required_layers(request: &RepertoireUseRequestV1) -> Vec<RightsLayerV1> {
    use RepertoireActionV1 as A;
    use RightsLayerV1 as L;

    let mut layers = match request.action {
        A::OpenExternalPlayer | A::GenerateOriginalFromBrief => Vec::new(),
        A::JoinLiveJam => vec![L::PerformerConsent],
        A::AnalyzeSourceAudio => vec![L::SoundRecording],
        A::TranscribeComposition => vec![L::SoundRecording, L::Composition],
        A::CreateArrangement => {
            let mut required = vec![L::Composition, L::Arrangement];
            if request.source.is_audio_asset() {
                required.insert(0, L::SoundRecording);
            }
            required
        }
        A::PrivateRehearsal => {
            let mut required = vec![L::Composition];
            if request.source.is_audio_asset() {
                required.insert(0, L::SoundRecording);
            }
            required
        }
        A::PerformCoverPublicly => vec![L::Composition, L::PublicPerformance],
        A::RecordCover => vec![L::Composition, L::CoverRecording],
        A::PublishCoverAudio => vec![
            L::Composition,
            L::CoverRecording,
            L::RecordingDistribution,
        ],
        A::SynchronizeCoverToVideo => vec![
            L::Composition,
            L::CoverRecording,
            L::Synchronization,
        ],
        A::ExportSongTranscription => vec![L::Composition, L::Redistribution],
        A::TrainModel => vec![L::SoundRecording, L::Composition, L::ModelTraining],
        A::RedistributeSourceAudio => vec![L::SoundRecording, L::Redistribution],
    };

    if request.uses_source_master_in_output
        && matches!(
            request.action,
            A::PerformCoverPublicly
                | A::RecordCover
                | A::PublishCoverAudio
                | A::SynchronizeCoverToVideo
        )
    {
        layers.push(L::SoundRecording);
    }

    if request.source.needs_platform_terms() && action_reads_source_audio(request) {
        layers.push(L::SourcePlatformTerms);
    }

    layers
}

fn action_reads_source_audio(request: &RepertoireUseRequestV1) -> bool {
    use RepertoireActionV1 as A;

    match request.action {
        A::AnalyzeSourceAudio
        | A::TranscribeComposition
        | A::CreateArrangement
        | A::TrainModel
        | A::RedistributeSourceAudio => true,
        A::PrivateRehearsal => request.source.is_audio_asset(),
        A::PerformCoverPublicly
        | A::RecordCover
        | A::PublishCoverAudio
        | A::SynchronizeCoverToVideo => request.uses_source_master_in_output,
        A::OpenExternalPlayer
        | A::JoinLiveJam
        | A::GenerateOriginalFromBrief
        | A::ExportSongTranscription => false,
    }
}

fn allows_explicit_local_assertion(request: &RepertoireUseRequestV1) -> bool {
    if request.processing_mode != RepertoireProcessingModeV1::EphemeralLocal
        || request.uses_source_master_in_output
    {
        return false;
    }

    match request.action {
        RepertoireActionV1::AnalyzeSourceAudio
        | RepertoireActionV1::PrivateRehearsal => {
            request.source == RepertoireSourceV1::UserSelectedLocalFile
        }
        RepertoireActionV1::JoinLiveJam => request.source == RepertoireSourceV1::LiveRoomInput,
        _ => false,
    }
}

fn has_current_source_terms_denial(
    request: &RepertoireUseRequestV1,
    grants: &[RightsGrantV1],
) -> bool {
    if !request.source.needs_platform_terms() || !action_reads_source_audio(request) {
        return false;
    }

    grants.iter().any(|grant| {
        grant.layer == RightsLayerV1::SourcePlatformTerms
            && grant.status == RightsEvidenceStatusV1::ExplicitlyDenied
            && grant.permitted_actions.contains(&request.action)
            && grant_scope_matches(grant, request)
            && has_evidence_reference(grant)
    })
}

fn has_usable_grant(
    layer: RightsLayerV1,
    status: RightsEvidenceStatusV1,
    request: &RepertoireUseRequestV1,
    grants: &[RightsGrantV1],
) -> bool {
    grants.iter().any(|grant| {
        grant.layer == layer
            && grant.status == status
            && grant.permitted_actions.contains(&request.action)
            && grant_scope_matches(grant, request)
            && has_evidence_reference(grant)
    })
}

fn has_evidence_reference(grant: &RightsGrantV1) -> bool {
    grant
        .evidence_ref
        .as_deref()
        .is_some_and(|reference| !reference.trim().is_empty())
}

fn grant_scope_matches(grant: &RightsGrantV1, request: &RepertoireUseRequestV1) -> bool {
    if let Some(expiry) = grant.valid_until_unix_seconds {
        if request.now_unix_seconds >= expiry {
            return false;
        }
    }

    match (grant.territory.as_deref(), request.territory.as_deref()) {
        (Some(g), Some(r)) => {
            g.eq_ignore_ascii_case("worldwide") || g.eq_ignore_ascii_case(r)
        }
        (Some(g), None) => g.eq_ignore_ascii_case("worldwide"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_800_000_000;

    fn request(
        action: RepertoireActionV1,
        source: RepertoireSourceV1,
        processing_mode: RepertoireProcessingModeV1,
    ) -> RepertoireUseRequestV1 {
        RepertoireUseRequestV1 {
            action,
            source,
            processing_mode,
            territory: Some("ZA".to_string()),
            now_unix_seconds: NOW,
            uses_source_master_in_output: false,
        }
    }

    fn grant(
        layer: RightsLayerV1,
        status: RightsEvidenceStatusV1,
        action: RepertoireActionV1,
    ) -> RightsGrantV1 {
        RightsGrantV1 {
            layer,
            status,
            permitted_actions: vec![action],
            territory: Some("ZA".to_string()),
            valid_until_unix_seconds: Some(NOW + 10_000),
            evidence_ref: Some("evidence:test-grant".to_string()),
        }
    }

    #[test]
    fn original_brief_needs_no_protected_source_asset() {
        let req = request(
            RepertoireActionV1::GenerateOriginalFromBrief,
            RepertoireSourceV1::TextBrief,
            RepertoireProcessingModeV1::EphemeralLocal,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[]),
            RepertoireGateOutcomeV1::AllowedWithoutProtectedSourceUse
        );
    }

    #[test]
    fn captured_streaming_audio_is_blocked_even_with_other_grants() {
        let req = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::StreamingPlatformAudio,
            RepertoireProcessingModeV1::Cloud,
        );
        let evidence = grant(
            RightsLayerV1::SoundRecording,
            RightsEvidenceStatusV1::VerifiedGrant,
            RepertoireActionV1::AnalyzeSourceAudio,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[evidence]),
            RepertoireGateOutcomeV1::BlockedByProductPolicy
        );
    }

    #[test]
    fn external_player_link_only_opens_the_player() {
        let open = request(
            RepertoireActionV1::OpenExternalPlayer,
            RepertoireSourceV1::ExternalPlayerLink,
            RepertoireProcessingModeV1::EphemeralLocal,
        );
        let analyse = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::ExternalPlayerLink,
            RepertoireProcessingModeV1::EphemeralLocal,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&open, &[]),
            RepertoireGateOutcomeV1::AllowedWithoutProtectedSourceUse
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&analyse, &[]),
            RepertoireGateOutcomeV1::BlockedByProductPolicy
        );
    }

    #[test]
    fn licensed_catalog_needs_platform_terms_as_well_as_recording_scope() {
        let req = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::LicensedCatalogAsset,
            RepertoireProcessingModeV1::LicensedService,
        );
        let recording = grant(
            RightsLayerV1::SoundRecording,
            RightsEvidenceStatusV1::VerifiedGrant,
            RepertoireActionV1::AnalyzeSourceAudio,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[recording]),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::SourcePlatformTerms]
            }
        );
    }

    #[test]
    fn verified_grants_must_match_territory_and_expiry() {
        let req = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::UserSelectedLocalFile,
            RepertoireProcessingModeV1::Cloud,
        );
        let mut evidence = grant(
            RightsLayerV1::SoundRecording,
            RightsEvidenceStatusV1::VerifiedGrant,
            RepertoireActionV1::AnalyzeSourceAudio,
        );
        evidence.territory = Some("US".to_string());
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[evidence.clone()]),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::SoundRecording]
            }
        );

        evidence.territory = Some("ZA".to_string());
        evidence.valid_until_unix_seconds = Some(NOW);
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[evidence]),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::SoundRecording]
            }
        );
    }

    #[test]
    fn user_assertion_only_unlocks_ephemeral_local_analysis() {
        let local = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::UserSelectedLocalFile,
            RepertoireProcessingModeV1::EphemeralLocal,
        );
        let assertion = grant(
            RightsLayerV1::SoundRecording,
            RightsEvidenceStatusV1::UserAssertion,
            RepertoireActionV1::AnalyzeSourceAudio,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&local, std::slice::from_ref(&assertion)),
            RepertoireGateOutcomeV1::AllowedInExplicitLocalMode
        );

        let mut cloud = local.clone();
        cloud.processing_mode = RepertoireProcessingModeV1::Cloud;
        assert_eq!(
            evaluate_repertoire_use_v1(&cloud, std::slice::from_ref(&assertion)),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::SoundRecording]
            }
        );

        let mut transcription = local;
        transcription.action = RepertoireActionV1::TranscribeComposition;
        assert_eq!(
            evaluate_repertoire_use_v1(&transcription, &[assertion]),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![
                    RightsLayerV1::SoundRecording,
                    RightsLayerV1::Composition
                ]
            }
        );
    }

    #[test]
    fn publishing_cover_requires_all_three_cover_scopes() {
        let req = request(
            RepertoireActionV1::PublishCoverAudio,
            RepertoireSourceV1::CreatorOwnedAsset,
            RepertoireProcessingModeV1::LicensedService,
        );
        let grants = [
            grant(
                RightsLayerV1::Composition,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::PublishCoverAudio,
            ),
            grant(
                RightsLayerV1::CoverRecording,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::PublishCoverAudio,
            ),
        ];
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &grants),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::RecordingDistribution]
            }
        );
    }

    #[test]
    fn new_cover_does_not_require_original_master_unless_embedded() {
        let mut req = request(
            RepertoireActionV1::RecordCover,
            RepertoireSourceV1::CreatorOwnedAsset,
            RepertoireProcessingModeV1::LicensedService,
        );
        let grants = [
            grant(
                RightsLayerV1::Composition,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::RecordCover,
            ),
            grant(
                RightsLayerV1::CoverRecording,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::RecordCover,
            ),
        ];
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &grants),
            RepertoireGateOutcomeV1::AllowedByRecordedGrant
        );

        req.uses_source_master_in_output = true;
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &grants),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::SoundRecording]
            }
        );
    }

    #[test]
    fn explicit_platform_denial_blocks_licensed_source_processing() {
        let req = request(
            RepertoireActionV1::AnalyzeSourceAudio,
            RepertoireSourceV1::LicensedCatalogAsset,
            RepertoireProcessingModeV1::LicensedService,
        );
        let grants = [
            grant(
                RightsLayerV1::SourcePlatformTerms,
                RightsEvidenceStatusV1::ExplicitlyDenied,
                RepertoireActionV1::AnalyzeSourceAudio,
            ),
            grant(
                RightsLayerV1::SoundRecording,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::AnalyzeSourceAudio,
            ),
        ];
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &grants),
            RepertoireGateOutcomeV1::BlockedBySourceTerms
        );
    }

    #[test]
    fn arrangement_from_symbolic_work_does_not_require_master_recording() {
        let req = request(
            RepertoireActionV1::CreateArrangement,
            RepertoireSourceV1::UserAuthoredSymbolicWork,
            RepertoireProcessingModeV1::LicensedService,
        );
        let grants = [
            grant(
                RightsLayerV1::Composition,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::CreateArrangement,
            ),
            grant(
                RightsLayerV1::Arrangement,
                RightsEvidenceStatusV1::VerifiedGrant,
                RepertoireActionV1::CreateArrangement,
            ),
        ];
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &grants),
            RepertoireGateOutcomeV1::AllowedByRecordedGrant
        );
    }

    #[test]
    fn public_performance_is_separate_from_cover_recording_rights() {
        let req = request(
            RepertoireActionV1::PerformCoverPublicly,
            RepertoireSourceV1::CreatorOwnedAsset,
            RepertoireProcessingModeV1::LicensedService,
        );
        let composition = grant(
            RightsLayerV1::Composition,
            RightsEvidenceStatusV1::VerifiedGrant,
            RepertoireActionV1::PerformCoverPublicly,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[composition]),
            RepertoireGateOutcomeV1::NeedsRightsReview {
                missing_layers: vec![RightsLayerV1::PublicPerformance]
            }
        );
    }

    #[test]
    fn live_jam_needs_participant_consent_not_stream_capture() {
        let req = request(
            RepertoireActionV1::JoinLiveJam,
            RepertoireSourceV1::LiveRoomInput,
            RepertoireProcessingModeV1::EphemeralLocal,
        );
        let consent = grant(
            RightsLayerV1::PerformerConsent,
            RightsEvidenceStatusV1::UserAssertion,
            RepertoireActionV1::JoinLiveJam,
        );
        assert_eq!(
            evaluate_repertoire_use_v1(&req, &[consent]),
            RepertoireGateOutcomeV1::AllowedInExplicitLocalMode
        );
    }
}

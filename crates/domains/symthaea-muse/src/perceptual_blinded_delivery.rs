// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1DR: blinded runtime stimulus-delivery contract.
//!
//! P1D correctly separates public opaque clip identifiers from the semantic
//! baseline/intervention codebook. This layer makes that separation operational:
//! a trusted pre-collection builder may compile the private audit into a minimal
//! opaque routing manifest, but the runtime delivery service does not receive
//! `StimulusArmV1` or the private audit.
//!
//! This is transport/codebook blinding, not a claim that an adversarial listener
//! cannot inspect or compare downloaded waveform bytes. In ABX, X intentionally
//! equals A or B as an auditory stimulus; raw-waveform forensic comparison is a
//! procedural/study-environment threat, not something HTTP metadata can solve.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_participant_schedule::{
        PerceptualCohortSlotsV1, PerceptualParticipantScheduleAuditV1,
        PerceptualParticipantScheduleBookV1, PrivateTrialMappingV1,
        validate_perceptual_participant_schedule,
    },
    perceptual_stimulus_pack::{
        FrozenC6fRenderSubjectBindingV1, FrozenPerceptualStimulusPackV1,
        REQUIRED_CHANNEL_COUNT, REQUIRED_SAMPLE_RATE_HZ, StimulusArmV1,
    },
    perceptual_study_protocol::FrozenPerceptualStudyProtocolV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BLINDED_DELIVERY_MANIFEST_VERSION: &str =
    "mel003-perceptual-blinded-delivery-manifest-v1";
pub const BLINDED_DELIVERY_SERVICE_VERSION: &str =
    "mel003-perceptual-blinded-delivery-service-v1";
pub const BLINDED_DELIVERY_CONTENT_TYPE: &str = "audio/wav";
pub const BLINDED_DELIVERY_CONTENT_DISPOSITION: &str =
    "inline; filename=\"stimulus.wav\"";
pub const BLINDED_DELIVERY_CACHE_CONTROL: &str =
    "no-store, no-cache, must-revalidate";
pub const BLINDED_DELIVERY_URL_PREFIX: &str = "/study/media/";
pub const CANONICAL_PCM16_WAV_HEADER_BYTES: usize = 44;
pub const CANONICAL_PCM16_STEREO_BYTES_PER_FRAME: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindedDeliveryServiceIdentityV1 {
    pub source_revision: String,
    pub service_binary_sha256: String,
    pub environment_sha256: String,
    pub implementation_version: String,
}

/// Qualified participant-facing asset identity supplied by the byte-level
/// qualification lineage. Deliberately contains no semantic arm label.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindedQualifiedAssetV1 {
    pub item_id: String,
    pub seed: u64,
    pub output_file_sha256: String,
    pub output_pcm_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
    pub content_length_bytes: usize,
}

/// Protected runtime route. There is intentionally no `StimulusArmV1` field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindedDeliveryRouteV1 {
    pub opaque_clip_id: String,
    pub item_id: String,
    pub seed: u64,
    pub output_file_sha256: String,
    pub output_pcm_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
    pub content_length_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantMediaMetadataPolicyV1 {
    pub content_type: String,
    pub content_disposition: String,
    pub cache_control: String,
    pub url_prefix: String,
    pub expose_etag: bool,
    pub expose_content_digest: bool,
    pub expose_semantic_filename: bool,
    pub expose_asset_sha256: bool,
    pub range_requests_enabled: bool,
}

impl Default for ParticipantMediaMetadataPolicyV1 {
    fn default() -> Self {
        Self {
            content_type: BLINDED_DELIVERY_CONTENT_TYPE.into(),
            content_disposition: BLINDED_DELIVERY_CONTENT_DISPOSITION.into(),
            cache_control: BLINDED_DELIVERY_CACHE_CONTROL.into(),
            url_prefix: BLINDED_DELIVERY_URL_PREFIX.into(),
            expose_etag: false,
            expose_content_digest: false,
            expose_semantic_filename: false,
            expose_asset_sha256: false,
            range_requests_enabled: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantBlindingLimitV1 {
    /// A technically motivated participant who obtains raw media may compare
    /// waveform/sample content. P1DR does not claim cryptographic resistance to
    /// this attack; study procedure/environment must address it.
    RawWaveformForensicComparisonNotPrevented,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenBlindedDeliveryManifestV1 {
    pub manifest_version: String,
    pub protocol_sha256: String,
    pub stimulus_pack_sha256: String,
    pub public_schedule_sha256: String,
    /// Exact commitment to the byte/sample qualification artifact.
    pub stimulus_qualification_sha256: String,
    pub service: BlindedDeliveryServiceIdentityV1,
    pub participant_media_policy: ParticipantMediaMetadataPolicyV1,
    pub blinding_limits: Vec<ParticipantBlindingLimitV1>,
    pub route_count: usize,
    /// Canonically sorted by opaque clip ID; runtime lookup relies on this.
    pub routes: Vec<BlindedDeliveryRouteV1>,
    pub manifest_sha256: String,
}

/// Participant-visible media metadata. Asset hashes are intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParticipantMediaResponseV1 {
    pub opaque_clip_id: String,
    pub media_url: String,
    pub content_type: String,
    pub content_disposition: String,
    pub cache_control: String,
    pub content_length_bytes: usize,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlindedDeliveryIssueV1 {
    InvalidProtocol,
    InvalidStimulusPack,
    InvalidParticipantSchedule,
    ProtocolSerializationFailed,
    ProtocolDigestMismatch,
    StimulusPackSerializationFailed,
    StimulusPackDigestMismatch,
    ScheduleSerializationFailed,
    ScheduleDigestMismatch,
    WrongManifestVersion,
    InvalidQualificationDigest,
    QualificationDigestMismatch,
    InvalidServiceDigest { field: String },
    InvalidServiceRevision,
    WrongServiceVersion,
    WrongPolicy,
    MissingBlindingLimit,
    WrongQualifiedAssetCount { found: usize, expected: usize },
    DuplicateQualifiedAssetDigest { digest: String },
    InvalidQualifiedAssetDigest { item_id: String, field: String },
    UnknownQualifiedAsset { item_id: String, digest: String },
    QualifiedAssetGeometryMismatch { item_id: String, digest: String },
    QualifiedAssetLengthMismatch { item_id: String, digest: String },
    WrongRouteCount { found: usize, expected: usize },
    RoutesNotCanonical { index: usize },
    DuplicateRouteClipId { clip_id: String },
    MissingPublicClipId { clip_id: String },
    UnexpectedRouteClipId { clip_id: String },
    RoutePublicItemMismatch { clip_id: String },
    RouteUnknownAsset { clip_id: String },
    RouteGeometryMismatch { clip_id: String },
    RouteLengthMismatch { clip_id: String },
    PrivateAuditRouteMismatch { clip_id: String },
    ManifestSerializationFailed,
    ManifestDigestMismatch,
}

pub fn build_blinded_delivery_manifest(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    private_audit: &PerceptualParticipantScheduleAuditV1,
    stimulus_qualification_sha256: &str,
    qualified_assets: &[BlindedQualifiedAssetV1],
    service: BlindedDeliveryServiceIdentityV1,
) -> Result<FrozenBlindedDeliveryManifestV1, Vec<BlindedDeliveryIssueV1>> {
    let issues = validate_build_inputs(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        schedule,
        private_audit,
        stimulus_qualification_sha256,
        qualified_assets,
        &service,
    );
    if !issues.is_empty() {
        return Err(issues);
    }

    let asset_by_digest: BTreeMap<_, _> = qualified_assets
        .iter()
        .map(|asset| (asset.output_file_sha256.as_str(), asset))
        .collect();
    let mut routes = Vec::new();
    for participant in &private_audit.participants {
        for trial in &participant.trials {
            for (clip_id, arm) in trial_clip_arms(trial) {
                let output_digest = output_digest_for_arm(trial, arm);
                let asset = asset_by_digest
                    .get(output_digest)
                    .expect("validated qualified assets contain every P1C output");
                routes.push(BlindedDeliveryRouteV1 {
                    opaque_clip_id: clip_id.to_string(),
                    item_id: trial.item_id.clone(),
                    seed: trial.seed,
                    output_file_sha256: asset.output_file_sha256.clone(),
                    output_pcm_sha256: asset.output_pcm_sha256.clone(),
                    sample_rate_hz: asset.sample_rate_hz,
                    channel_count: asset.channel_count,
                    frame_count: asset.frame_count,
                    content_length_bytes: asset.content_length_bytes,
                });
            }
        }
    }
    routes.sort_by(|left, right| left.opaque_clip_id.cmp(&right.opaque_clip_id));

    let mut manifest = FrozenBlindedDeliveryManifestV1 {
        manifest_version: BLINDED_DELIVERY_MANIFEST_VERSION.into(),
        protocol_sha256: canonical_json_sha256(protocol)
            .expect("validated protocol must serialize canonically"),
        stimulus_pack_sha256: canonical_json_sha256(stimulus_pack)
            .expect("validated stimulus pack must serialize canonically"),
        public_schedule_sha256: canonical_json_sha256(schedule)
            .expect("validated participant schedule must serialize canonically"),
        stimulus_qualification_sha256: stimulus_qualification_sha256.into(),
        service,
        participant_media_policy: ParticipantMediaMetadataPolicyV1::default(),
        blinding_limits: vec![
            ParticipantBlindingLimitV1::RawWaveformForensicComparisonNotPrevented,
        ],
        route_count: routes.len(),
        routes,
        manifest_sha256: String::new(),
    };
    manifest.manifest_sha256 = delivery_manifest_commitment(&manifest)
        .expect("blinded delivery manifest must serialize canonically");

    let validation = validate_blinded_delivery_manifest_public(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        schedule,
        stimulus_qualification_sha256,
        qualified_assets,
        &manifest,
    );
    let post_close = verify_blinded_delivery_manifest_after_reveal(
        stimulus_pack,
        private_audit,
        qualified_assets,
        &manifest,
    );
    if validation.is_empty() && post_close.is_empty() {
        Ok(manifest)
    } else {
        let mut combined = validation;
        combined.extend(post_close);
        Err(combined)
    }
}

/// Validation available to the blinded runtime/collector without private-audit
/// access. This proves schedule coverage and registered-asset routing, but not
/// which semantic arm a given opaque ID was randomized to.
pub fn validate_blinded_delivery_manifest_public(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    expected_stimulus_qualification_sha256: &str,
    qualified_assets: &[BlindedQualifiedAssetV1],
    manifest: &FrozenBlindedDeliveryManifestV1,
) -> Vec<BlindedDeliveryIssueV1> {
    let mut issues = Vec::new();
    if !protocol.validate().is_empty() {
        issues.push(BlindedDeliveryIssueV1::InvalidProtocol);
    }
    if !stimulus_pack.validate(protocol, render_binding).is_empty() {
        issues.push(BlindedDeliveryIssueV1::InvalidStimulusPack);
    }
    if !validate_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        schedule,
        None,
    )
    .is_empty()
    {
        issues.push(BlindedDeliveryIssueV1::InvalidParticipantSchedule);
    }

    match canonical_json_sha256(protocol) {
        Ok(value) if value == manifest.protocol_sha256 => {}
        Ok(_) => issues.push(BlindedDeliveryIssueV1::ProtocolDigestMismatch),
        Err(_) => issues.push(BlindedDeliveryIssueV1::ProtocolSerializationFailed),
    }
    match canonical_json_sha256(stimulus_pack) {
        Ok(value) if value == manifest.stimulus_pack_sha256 => {}
        Ok(_) => issues.push(BlindedDeliveryIssueV1::StimulusPackDigestMismatch),
        Err(_) => issues.push(BlindedDeliveryIssueV1::StimulusPackSerializationFailed),
    }
    match canonical_json_sha256(schedule) {
        Ok(value) if value == manifest.public_schedule_sha256 => {}
        Ok(_) => issues.push(BlindedDeliveryIssueV1::ScheduleDigestMismatch),
        Err(_) => issues.push(BlindedDeliveryIssueV1::ScheduleSerializationFailed),
    }
    if manifest.manifest_version != BLINDED_DELIVERY_MANIFEST_VERSION {
        issues.push(BlindedDeliveryIssueV1::WrongManifestVersion);
    }
    if !is_sha256(expected_stimulus_qualification_sha256)
        || !is_sha256(&manifest.stimulus_qualification_sha256)
    {
        issues.push(BlindedDeliveryIssueV1::InvalidQualificationDigest);
    }
    if manifest.stimulus_qualification_sha256 != expected_stimulus_qualification_sha256 {
        issues.push(BlindedDeliveryIssueV1::QualificationDigestMismatch);
    }
    validate_service(&manifest.service, &mut issues);
    if manifest.participant_media_policy != ParticipantMediaMetadataPolicyV1::default() {
        issues.push(BlindedDeliveryIssueV1::WrongPolicy);
    }
    if manifest.blinding_limits
        != vec![ParticipantBlindingLimitV1::RawWaveformForensicComparisonNotPrevented]
    {
        issues.push(BlindedDeliveryIssueV1::MissingBlindingLimit);
    }

    let qualified = validate_qualified_assets(stimulus_pack, qualified_assets, &mut issues);
    let public_clip_items = public_clip_item_map(schedule);
    if manifest.route_count != public_clip_items.len() || manifest.routes.len() != manifest.route_count {
        issues.push(BlindedDeliveryIssueV1::WrongRouteCount {
            found: manifest.routes.len(),
            expected: public_clip_items.len(),
        });
    }
    for (index, pair) in manifest.routes.windows(2).enumerate() {
        if pair[0].opaque_clip_id >= pair[1].opaque_clip_id {
            issues.push(BlindedDeliveryIssueV1::RoutesNotCanonical { index: index + 1 });
        }
    }

    let asset_by_digest: BTreeMap<_, _> = qualified
        .iter()
        .map(|asset| (asset.output_file_sha256.as_str(), *asset))
        .collect();
    let mut seen_routes = BTreeSet::new();
    for route in &manifest.routes {
        if !seen_routes.insert(route.opaque_clip_id.as_str()) {
            issues.push(BlindedDeliveryIssueV1::DuplicateRouteClipId {
                clip_id: route.opaque_clip_id.clone(),
            });
            continue;
        }
        let Some((expected_item, expected_seed)) = public_clip_items.get(route.opaque_clip_id.as_str())
        else {
            issues.push(BlindedDeliveryIssueV1::UnexpectedRouteClipId {
                clip_id: route.opaque_clip_id.clone(),
            });
            continue;
        };
        if route.item_id.as_str() != *expected_item || route.seed != *expected_seed {
            issues.push(BlindedDeliveryIssueV1::RoutePublicItemMismatch {
                clip_id: route.opaque_clip_id.clone(),
            });
        }
        let Some(asset) = asset_by_digest.get(route.output_file_sha256.as_str()) else {
            issues.push(BlindedDeliveryIssueV1::RouteUnknownAsset {
                clip_id: route.opaque_clip_id.clone(),
            });
            continue;
        };
        if route.item_id != asset.item_id
            || route.seed != asset.seed
            || route.output_pcm_sha256 != asset.output_pcm_sha256
        {
            issues.push(BlindedDeliveryIssueV1::RouteUnknownAsset {
                clip_id: route.opaque_clip_id.clone(),
            });
        }
        if route.sample_rate_hz != asset.sample_rate_hz
            || route.channel_count != asset.channel_count
            || route.frame_count != asset.frame_count
        {
            issues.push(BlindedDeliveryIssueV1::RouteGeometryMismatch {
                clip_id: route.opaque_clip_id.clone(),
            });
        }
        if route.content_length_bytes != asset.content_length_bytes {
            issues.push(BlindedDeliveryIssueV1::RouteLengthMismatch {
                clip_id: route.opaque_clip_id.clone(),
            });
        }
    }
    for clip_id in public_clip_items.keys() {
        if !seen_routes.contains(clip_id) {
            issues.push(BlindedDeliveryIssueV1::MissingPublicClipId {
                clip_id: (*clip_id).into(),
            });
        }
    }

    match delivery_manifest_commitment(manifest) {
        Ok(value) if value == manifest.manifest_sha256 => {}
        Ok(_) => issues.push(BlindedDeliveryIssueV1::ManifestDigestMismatch),
        Err(_) => issues.push(BlindedDeliveryIssueV1::ManifestSerializationFailed),
    }
    issues
}

/// Post-close verification with semantic audit authority. Rebuilds every
/// expected opaque-ID → exact-output route and requires the frozen manifest to
/// match. A swapped route can pass public registered-asset validation but must
/// fail here.
pub fn verify_blinded_delivery_manifest_after_reveal(
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    private_audit: &PerceptualParticipantScheduleAuditV1,
    qualified_assets: &[BlindedQualifiedAssetV1],
    manifest: &FrozenBlindedDeliveryManifestV1,
) -> Vec<BlindedDeliveryIssueV1> {
    let mut issues = Vec::new();
    let asset_by_digest: BTreeMap<_, _> = qualified_assets
        .iter()
        .map(|asset| (asset.output_file_sha256.as_str(), asset))
        .collect();
    let route_by_clip: BTreeMap<_, _> = manifest
        .routes
        .iter()
        .map(|route| (route.opaque_clip_id.as_str(), route))
        .collect();

    for participant in &private_audit.participants {
        for trial in &participant.trials {
            for (clip_id, arm) in trial_clip_arms(trial) {
                let expected_digest = output_digest_for_arm(trial, arm);
                let Some(asset) = asset_by_digest.get(expected_digest) else {
                    issues.push(BlindedDeliveryIssueV1::RouteUnknownAsset {
                        clip_id: clip_id.into(),
                    });
                    continue;
                };
                let Some(route) = route_by_clip.get(clip_id) else {
                    issues.push(BlindedDeliveryIssueV1::MissingPublicClipId {
                        clip_id: clip_id.into(),
                    });
                    continue;
                };
                if route.item_id != trial.item_id
                    || route.seed != trial.seed
                    || route.output_file_sha256 != asset.output_file_sha256
                    || route.output_pcm_sha256 != asset.output_pcm_sha256
                    || route.sample_rate_hz != asset.sample_rate_hz
                    || route.channel_count != asset.channel_count
                    || route.frame_count != asset.frame_count
                    || route.content_length_bytes != asset.content_length_bytes
                {
                    issues.push(BlindedDeliveryIssueV1::PrivateAuditRouteMismatch {
                        clip_id: clip_id.into(),
                    });
                }
            }
        }
    }

    // The audit itself must still refer only to the two pack outputs per item.
    let pack_by_seed: BTreeMap<_, _> = stimulus_pack
        .items
        .iter()
        .map(|pair| (pair.seed, pair))
        .collect();
    for participant in &private_audit.participants {
        for trial in &participant.trials {
            if let Some(pair) = pack_by_seed.get(&trial.seed) {
                if trial.baseline_output_sha256 != pair.baseline.output_sha256
                    || trial.intervention_output_sha256 != pair.intervention.output_sha256
                {
                    issues.push(BlindedDeliveryIssueV1::PrivateAuditRouteMismatch {
                        clip_id: trial.abx_trial_id.clone(),
                    });
                }
            }
        }
    }
    issues
}

pub fn resolve_participant_media(
    manifest: &FrozenBlindedDeliveryManifestV1,
    opaque_clip_id: &str,
) -> Option<ParticipantMediaResponseV1> {
    let route = manifest
        .routes
        .binary_search_by(|route| route.opaque_clip_id.as_str().cmp(opaque_clip_id))
        .ok()
        .and_then(|index| manifest.routes.get(index))?;
    let policy = &manifest.participant_media_policy;
    Some(ParticipantMediaResponseV1 {
        opaque_clip_id: route.opaque_clip_id.clone(),
        media_url: format!("{}{}", policy.url_prefix, route.opaque_clip_id),
        content_type: policy.content_type.clone(),
        content_disposition: policy.content_disposition.clone(),
        cache_control: policy.cache_control.clone(),
        content_length_bytes: route.content_length_bytes,
        sample_rate_hz: route.sample_rate_hz,
        channel_count: route.channel_count,
        frame_count: route.frame_count,
    })
}

pub fn delivery_manifest_commitment(
    manifest: &FrozenBlindedDeliveryManifestV1,
) -> Result<String, serde_json::Error> {
    let mut unsigned = manifest.clone();
    unsigned.manifest_sha256.clear();
    canonical_json_sha256(&unsigned)
}

fn validate_build_inputs(
    protocol: &FrozenPerceptualStudyProtocolV1,
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    cohort: &PerceptualCohortSlotsV1,
    schedule: &PerceptualParticipantScheduleBookV1,
    private_audit: &PerceptualParticipantScheduleAuditV1,
    qualification_sha256: &str,
    qualified_assets: &[BlindedQualifiedAssetV1],
    service: &BlindedDeliveryServiceIdentityV1,
) -> Vec<BlindedDeliveryIssueV1> {
    let mut issues = Vec::new();
    if !validate_perceptual_participant_schedule(
        protocol,
        stimulus_pack,
        render_binding,
        cohort,
        schedule,
        Some(private_audit),
    )
    .is_empty()
    {
        issues.push(BlindedDeliveryIssueV1::InvalidParticipantSchedule);
    }
    if !is_sha256(qualification_sha256) {
        issues.push(BlindedDeliveryIssueV1::InvalidQualificationDigest);
    }
    validate_service(service, &mut issues);
    validate_qualified_assets(stimulus_pack, qualified_assets, &mut issues);
    issues
}

fn validate_service(
    service: &BlindedDeliveryServiceIdentityV1,
    issues: &mut Vec<BlindedDeliveryIssueV1>,
) {
    for (field, digest) in [
        ("service_binary_sha256", service.service_binary_sha256.as_str()),
        ("environment_sha256", service.environment_sha256.as_str()),
    ] {
        if !is_sha256(digest) {
            issues.push(BlindedDeliveryIssueV1::InvalidServiceDigest {
                field: field.into(),
            });
        }
    }
    if !is_git_sha1(&service.source_revision) {
        issues.push(BlindedDeliveryIssueV1::InvalidServiceRevision);
    }
    if service.implementation_version != BLINDED_DELIVERY_SERVICE_VERSION {
        issues.push(BlindedDeliveryIssueV1::WrongServiceVersion);
    }
}

fn validate_qualified_assets<'a>(
    stimulus_pack: &FrozenPerceptualStimulusPackV1,
    qualified_assets: &'a [BlindedQualifiedAssetV1],
    issues: &mut Vec<BlindedDeliveryIssueV1>,
) -> Vec<&'a BlindedQualifiedAssetV1> {
    let expected_count = stimulus_pack.items.len().saturating_mul(2);
    if qualified_assets.len() != expected_count {
        issues.push(BlindedDeliveryIssueV1::WrongQualifiedAssetCount {
            found: qualified_assets.len(),
            expected: expected_count,
        });
    }
    let pack_outputs: BTreeMap<_, _> = stimulus_pack
        .items
        .iter()
        .flat_map(|pair| {
            [
                (
                    pair.baseline.output_sha256.as_str(),
                    (pair.item_id.as_str(), pair.seed, &pair.baseline),
                ),
                (
                    pair.intervention.output_sha256.as_str(),
                    (pair.item_id.as_str(), pair.seed, &pair.intervention),
                ),
            ]
        })
        .collect();
    let mut seen = BTreeSet::new();
    let mut accepted = Vec::new();
    for asset in qualified_assets {
        for (field, digest) in [
            ("output_file_sha256", asset.output_file_sha256.as_str()),
            ("output_pcm_sha256", asset.output_pcm_sha256.as_str()),
        ] {
            if !is_sha256(digest) {
                issues.push(BlindedDeliveryIssueV1::InvalidQualifiedAssetDigest {
                    item_id: asset.item_id.clone(),
                    field: field.into(),
                });
            }
        }
        if !seen.insert(asset.output_file_sha256.as_str()) {
            issues.push(BlindedDeliveryIssueV1::DuplicateQualifiedAssetDigest {
                digest: asset.output_file_sha256.clone(),
            });
            continue;
        }
        let Some((expected_item, expected_seed, manifest_asset)) =
            pack_outputs.get(asset.output_file_sha256.as_str())
        else {
            issues.push(BlindedDeliveryIssueV1::UnknownQualifiedAsset {
                item_id: asset.item_id.clone(),
                digest: asset.output_file_sha256.clone(),
            });
            continue;
        };
        if asset.item_id.as_str() != *expected_item || asset.seed != *expected_seed {
            issues.push(BlindedDeliveryIssueV1::UnknownQualifiedAsset {
                item_id: asset.item_id.clone(),
                digest: asset.output_file_sha256.clone(),
            });
        }
        if asset.sample_rate_hz != manifest_asset.sample_rate_hz
            || asset.channel_count != manifest_asset.channel_count
            || asset.frame_count != manifest_asset.frame_count
            || asset.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ
            || asset.channel_count != REQUIRED_CHANNEL_COUNT
        {
            issues.push(BlindedDeliveryIssueV1::QualifiedAssetGeometryMismatch {
                item_id: asset.item_id.clone(),
                digest: asset.output_file_sha256.clone(),
            });
        }
        let expected_length = CANONICAL_PCM16_WAV_HEADER_BYTES
            .checked_add(asset.frame_count.saturating_mul(CANONICAL_PCM16_STEREO_BYTES_PER_FRAME));
        if expected_length != Some(asset.content_length_bytes) {
            issues.push(BlindedDeliveryIssueV1::QualifiedAssetLengthMismatch {
                item_id: asset.item_id.clone(),
                digest: asset.output_file_sha256.clone(),
            });
        }
        accepted.push(asset);
    }
    accepted
}

fn public_clip_item_map(
    schedule: &PerceptualParticipantScheduleBookV1,
) -> BTreeMap<&str, (&str, u64)> {
    let mut clips = BTreeMap::new();
    for participant in &schedule.schedules {
        for trial in &participant.abx_trials {
            for clip in [&trial.a_clip_id, &trial.b_clip_id, &trial.x_clip_id] {
                clips.insert(clip.as_str(), (trial.item_id.as_str(), trial.seed));
            }
        }
        for trial in &participant.directional_trials {
            for clip in [&trial.left_clip_id, &trial.right_clip_id] {
                clips.insert(clip.as_str(), (trial.item_id.as_str(), trial.seed));
            }
        }
    }
    clips
}

fn trial_clip_arms(trial: &PrivateTrialMappingV1) -> [(&str, StimulusArmV1); 5] {
    [
        (trial.a_clip_id.as_str(), trial.a_arm),
        (trial.b_clip_id.as_str(), trial.b_arm),
        (trial.x_clip_id.as_str(), trial.x_arm),
        (trial.left_clip_id.as_str(), trial.left_arm),
        (trial.right_clip_id.as_str(), trial.right_arm),
    ]
}

fn output_digest_for_arm(trial: &PrivateTrialMappingV1, arm: StimulusArmV1) -> &str {
    match arm {
        StimulusArmV1::Baseline => trial.baseline_output_sha256.as_str(),
        StimulusArmV1::Intervention => trial.intervention_output_sha256.as_str(),
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_git_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn participant_policy_exposes_no_content_derived_identifiers() {
        let policy = ParticipantMediaMetadataPolicyV1::default();
        assert!(!policy.expose_etag);
        assert!(!policy.expose_content_digest);
        assert!(!policy.expose_semantic_filename);
        assert!(!policy.expose_asset_sha256);
        assert!(!policy.range_requests_enabled);
        assert_eq!(policy.content_type, "audio/wav");
        assert_eq!(policy.content_disposition, "inline; filename=\"stimulus.wav\"");
        assert_eq!(policy.cache_control, "no-store, no-cache, must-revalidate");
        assert_eq!(policy.url_prefix, "/study/media/");
    }

    #[test]
    fn runtime_route_type_is_arm_free_and_response_is_hash_free() {
        let route = BlindedDeliveryRouteV1 {
            opaque_clip_id: "opaque".into(),
            item_id: "item".into(),
            seed: 3,
            output_file_sha256: "a".repeat(64),
            output_pcm_sha256: "b".repeat(64),
            sample_rate_hz: 44_100,
            channel_count: 2,
            frame_count: 100,
            content_length_bytes: 444,
        };
        let mut manifest = FrozenBlindedDeliveryManifestV1 {
            manifest_version: BLINDED_DELIVERY_MANIFEST_VERSION.into(),
            protocol_sha256: "c".repeat(64),
            stimulus_pack_sha256: "d".repeat(64),
            public_schedule_sha256: "e".repeat(64),
            stimulus_qualification_sha256: "f".repeat(64),
            service: BlindedDeliveryServiceIdentityV1 {
                source_revision: "1".repeat(40),
                service_binary_sha256: "2".repeat(64),
                environment_sha256: "3".repeat(64),
                implementation_version: BLINDED_DELIVERY_SERVICE_VERSION.into(),
            },
            participant_media_policy: ParticipantMediaMetadataPolicyV1::default(),
            blinding_limits: vec![
                ParticipantBlindingLimitV1::RawWaveformForensicComparisonNotPrevented,
            ],
            route_count: 1,
            routes: vec![route],
            manifest_sha256: String::new(),
        };
        manifest.manifest_sha256 = delivery_manifest_commitment(&manifest).unwrap();
        let response = resolve_participant_media(&manifest, "opaque").unwrap();
        let json = serde_json::to_string(&response).unwrap();
        assert!(!json.contains(&"a".repeat(64)));
        assert!(!json.contains(&"b".repeat(64)));
        assert!(!json.contains("Baseline"));
        assert!(!json.contains("Intervention"));
        assert_eq!(response.media_url, "/study/media/opaque");
    }

    #[test]
    fn manifest_commitment_changes_when_route_target_changes() {
        let base_route = BlindedDeliveryRouteV1 {
            opaque_clip_id: "clip".into(),
            item_id: "item".into(),
            seed: 3,
            output_file_sha256: "a".repeat(64),
            output_pcm_sha256: "b".repeat(64),
            sample_rate_hz: 44_100,
            channel_count: 2,
            frame_count: 100,
            content_length_bytes: 444,
        };
        let make = |route: BlindedDeliveryRouteV1| FrozenBlindedDeliveryManifestV1 {
            manifest_version: BLINDED_DELIVERY_MANIFEST_VERSION.into(),
            protocol_sha256: "c".repeat(64),
            stimulus_pack_sha256: "d".repeat(64),
            public_schedule_sha256: "e".repeat(64),
            stimulus_qualification_sha256: "f".repeat(64),
            service: BlindedDeliveryServiceIdentityV1 {
                source_revision: "1".repeat(40),
                service_binary_sha256: "2".repeat(64),
                environment_sha256: "3".repeat(64),
                implementation_version: BLINDED_DELIVERY_SERVICE_VERSION.into(),
            },
            participant_media_policy: ParticipantMediaMetadataPolicyV1::default(),
            blinding_limits: vec![
                ParticipantBlindingLimitV1::RawWaveformForensicComparisonNotPrevented,
            ],
            route_count: 1,
            routes: vec![route],
            manifest_sha256: String::new(),
        };
        let left = delivery_manifest_commitment(&make(base_route.clone())).unwrap();
        let mut changed = base_route;
        changed.output_file_sha256 = "9".repeat(64);
        let right = delivery_manifest_commitment(&make(changed)).unwrap();
        assert_ne!(left, right);
    }
}

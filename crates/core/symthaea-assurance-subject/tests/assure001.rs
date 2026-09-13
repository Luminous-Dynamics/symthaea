// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{DigestSha256, StableId};
use symthaea_assurance_subject::{
    AiSubjectManifest, AiSurfaceKind, CommitmentMethod, MaterialCommitment, SubjectError,
    SurfaceBinding, SurfaceLocator, SurfaceProfile, SurfaceState, UnavailabilityReason,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(byte: char) -> DigestSha256 {
    DigestSha256::new(std::iter::repeat_n(byte, 64).collect::<String>()).unwrap()
}

fn commit(byte: char) -> MaterialCommitment {
    MaterialCommitment::artifact_bytes(digest(byte))
}

fn locator(name: &str) -> SurfaceLocator {
    SurfaceLocator::new(Some(id("provider-a")), id(name), Some(id("v1")))
}

fn binding(kind: AiSurfaceKind, name: &str, state: SurfaceState) -> SurfaceBinding {
    SurfaceBinding::applicable(kind, locator(name), state).unwrap()
}

fn model_runtime_profile() -> SurfaceProfile {
    SurfaceProfile::new(
        id("model-runtime-v1"),
        vec![AiSurfaceKind::Model, AiSurfaceKind::Runtime],
    )
    .unwrap()
}

fn exact_manifest() -> AiSubjectManifest {
    AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('a')),
            ),
            binding(
                AiSurfaceKind::Runtime,
                "runtime-image",
                SurfaceState::Known(commit('b')),
            ),
        ],
    )
    .unwrap()
}

#[test]
fn profile_identity_is_surface_order_independent() {
    let left = SurfaceProfile::new(
        id("profile-v1"),
        vec![AiSurfaceKind::Runtime, AiSurfaceKind::Model],
    )
    .unwrap();
    let right = SurfaceProfile::new(
        id("profile-v1"),
        vec![AiSurfaceKind::Model, AiSurfaceKind::Runtime],
    )
    .unwrap();
    assert_eq!(left.digest(), right.digest());
}

#[test]
fn duplicate_profile_surface_fails_closed() {
    let error = SurfaceProfile::new(
        id("duplicate-profile"),
        vec![AiSurfaceKind::Model, AiSurfaceKind::Model],
    )
    .unwrap_err();
    assert!(matches!(error, SubjectError::DuplicateProfileSurface(_)));
}

#[test]
fn manifest_identity_is_binding_order_independent() {
    let left = exact_manifest();
    let right = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(
                AiSurfaceKind::Runtime,
                "runtime-image",
                SurfaceState::Known(commit('b')),
            ),
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('a')),
            ),
        ],
    )
    .unwrap();
    assert_eq!(left.manifest_id(), right.manifest_id());
}

#[test]
fn omission_is_not_treated_as_unknown() {
    let error = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Unknown,
        )],
    )
    .unwrap_err();
    assert!(matches!(error, SubjectError::MissingSurface(_)));
}

#[test]
fn unexpected_surface_fails_closed() {
    let error = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(AiSurfaceKind::Model, "model-alias", SurfaceState::Unknown),
            binding(AiSurfaceKind::Runtime, "runtime", SurfaceState::Unknown),
            binding(AiSurfaceKind::Policy, "policy", SurfaceState::Unknown),
        ],
    )
    .unwrap_err();
    assert!(matches!(error, SubjectError::UnexpectedSurface(_)));
}

#[test]
fn duplicate_binding_fails_closed() {
    let error = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(AiSurfaceKind::Model, "model-a", SurfaceState::Unknown),
            binding(AiSurfaceKind::Model, "model-b", SurfaceState::Unknown),
            binding(AiSurfaceKind::Runtime, "runtime", SurfaceState::Unknown),
        ],
    )
    .unwrap_err();
    assert!(matches!(error, SubjectError::DuplicateBinding(_)));
}

#[test]
fn applicable_surface_requires_locator() {
    let error = SurfaceBinding::new(AiSurfaceKind::Model, None, SurfaceState::Unknown).unwrap_err();
    assert!(matches!(error, SubjectError::MissingLocator(_)));
}

#[test]
fn not_applicable_surface_rejects_locator() {
    let error = SurfaceBinding::new(
        AiSurfaceKind::DeploymentEnvelope,
        Some(locator("unused-deployment-envelope")),
        SurfaceState::NotApplicable,
    )
    .unwrap_err();
    assert!(matches!(error, SubjectError::LocatorOnNotApplicable(_)));
}

#[test]
fn completeness_states_are_identity_distinct() {
    let make = |state| {
        AiSubjectManifest::new(
            id("agent-a"),
            SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap(),
            vec![binding(AiSurfaceKind::Model, "model-alias", state)],
        )
        .unwrap()
    };

    let known = make(SurfaceState::Known(commit('a')));
    let unknown = make(SurfaceState::Unknown);
    let unavailable = make(SurfaceState::Unavailable(
        UnavailabilityReason::ProviderDoesNotExpose,
    ));
    let not_applicable = AiSubjectManifest::new(
        id("agent-a"),
        SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap(),
        vec![SurfaceBinding::not_applicable(AiSurfaceKind::Model)],
    )
    .unwrap();

    assert_ne!(known.manifest_id(), unknown.manifest_id());
    assert_ne!(unknown.manifest_id(), unavailable.manifest_id());
    assert_ne!(unavailable.manifest_id(), not_applicable.manifest_id());
    assert_ne!(known.manifest_id(), not_applicable.manifest_id());
}

#[test]
fn completeness_controls_committed_identity_not_replayability() {
    let complete = AiSubjectManifest::new(
        id("agent-a"),
        SurfaceProfile::new(
            id("model-deployment"),
            vec![AiSurfaceKind::Model, AiSurfaceKind::DeploymentEnvelope],
        )
        .unwrap(),
        vec![
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('a')),
            ),
            SurfaceBinding::not_applicable(AiSurfaceKind::DeploymentEnvelope),
        ],
    )
    .unwrap();
    let complete_summary = complete.completeness();
    assert_eq!(complete_summary.known, 1);
    assert_eq!(complete_summary.not_applicable, 1);
    assert!(complete_summary.has_complete_committed_identity());

    let unknown = AiSubjectManifest::new(
        id("agent-a"),
        SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Unknown,
        )],
    )
    .unwrap();
    assert!(!unknown.completeness().has_complete_committed_identity());

    let unavailable = AiSubjectManifest::new(
        id("agent-a"),
        SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Unavailable(UnavailabilityReason::ProviderDoesNotExpose),
        )],
    )
    .unwrap();
    assert!(!unavailable.completeness().has_complete_committed_identity());
}

#[test]
fn immutable_commitment_change_changes_subject_identity() {
    let left = exact_manifest();
    let right = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('c')),
            ),
            binding(
                AiSurfaceKind::Runtime,
                "runtime-image",
                SurfaceState::Known(commit('b')),
            ),
        ],
    )
    .unwrap();
    assert_ne!(left.manifest_id(), right.manifest_id());
}

#[test]
fn commitment_method_is_bound_into_subject_identity() {
    let profile = SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap();
    let artifact_bytes = AiSubjectManifest::new(
        id("agent-a"),
        profile.clone(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::new(
                CommitmentMethod::ArtifactBytesSha256,
                digest('a'),
            )),
        )],
    )
    .unwrap();
    let provider_token = AiSubjectManifest::new(
        id("agent-a"),
        profile,
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::provider_revision_token(
                id("provider-revision-v1"),
                digest('a'),
            )),
        )],
    )
    .unwrap();

    assert_ne!(artifact_bytes.manifest_id(), provider_token.manifest_id());
}

#[test]
fn descriptor_schema_is_bound_into_subject_identity() {
    let profile = SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap();
    let left = AiSubjectManifest::new(
        id("agent-a"),
        profile.clone(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::canonical_descriptor(
                id("model-descriptor-v1"),
                digest('a'),
            )),
        )],
    )
    .unwrap();
    let right = AiSubjectManifest::new(
        id("agent-a"),
        profile,
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::canonical_descriptor(
                id("model-descriptor-v2"),
                digest('a'),
            )),
        )],
    )
    .unwrap();

    assert_ne!(left.manifest_id(), right.manifest_id());
}

#[test]
fn provider_token_namespace_is_bound_into_subject_identity() {
    let profile = SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap();
    let left = AiSubjectManifest::new(
        id("agent-a"),
        profile.clone(),
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::provider_revision_token(
                id("provider-token-v1"),
                digest('a'),
            )),
        )],
    )
    .unwrap();
    let right = AiSubjectManifest::new(
        id("agent-a"),
        profile,
        vec![binding(
            AiSurfaceKind::Model,
            "model-alias",
            SurfaceState::Known(MaterialCommitment::provider_revision_token(
                id("provider-token-v2"),
                digest('a'),
            )),
        )],
    )
    .unwrap();

    assert_ne!(left.manifest_id(), right.manifest_id());
}

#[test]
fn provider_alias_metadata_does_not_masquerade_as_same_subject() {
    let profile = SurfaceProfile::new(id("model-only"), vec![AiSurfaceKind::Model]).unwrap();
    let left = AiSubjectManifest::new(
        id("agent-a"),
        profile.clone(),
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("rolling-alias"), Some(id("v1"))),
                SurfaceState::Known(commit('a')),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let renamed = AiSubjectManifest::new(
        id("agent-a"),
        profile.clone(),
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("other-alias"), Some(id("v1"))),
                SurfaceState::Known(commit('a')),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let reversioned = AiSubjectManifest::new(
        id("agent-a"),
        profile,
        vec![
            SurfaceBinding::applicable(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("rolling-alias"), Some(id("v2"))),
                SurfaceState::Known(commit('a')),
            )
            .unwrap(),
        ],
    )
    .unwrap();

    assert_ne!(left.manifest_id(), renamed.manifest_id());
    assert_ne!(left.manifest_id(), reversioned.manifest_id());
}

#[test]
fn semantic_subject_key_is_bound_into_identity() {
    let left = exact_manifest();
    let right = AiSubjectManifest::new(
        id("agent-b"),
        model_runtime_profile(),
        vec![
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('a')),
            ),
            binding(
                AiSurfaceKind::Runtime,
                "runtime-image",
                SurfaceState::Known(commit('b')),
            ),
        ],
    )
    .unwrap();
    assert_ne!(left.manifest_id(), right.manifest_id());
}

#[test]
fn external_dependency_registration_changes_profile_identity() {
    let no_remote = SurfaceProfile::external_ai_v1(vec![]).unwrap();
    let with_remote = SurfaceProfile::external_ai_v1(vec![id("retrieval-index")]).unwrap();
    assert_ne!(no_remote.digest(), with_remote.digest());
}

#[test]
fn duplicate_external_dependency_registration_fails_closed() {
    let error = SurfaceProfile::external_ai_v1(vec![id("tool-a"), id("tool-a")]).unwrap_err();
    assert!(matches!(error, SubjectError::DuplicateProfileSurface(_)));
}

#[test]
fn core_bridge_tracks_exact_assure001_identity() {
    let left = exact_manifest();
    assert_eq!(
        left.core_subject_id().unwrap(),
        left.core_subject_id().unwrap()
    );

    let changed = AiSubjectManifest::new(
        id("agent-a"),
        model_runtime_profile(),
        vec![
            binding(
                AiSurfaceKind::Model,
                "model-alias",
                SurfaceState::Known(commit('a')),
            ),
            binding(
                AiSurfaceKind::Runtime,
                "runtime-image",
                SurfaceState::Known(commit('c')),
            ),
        ],
    )
    .unwrap();

    assert_ne!(left.manifest_id(), changed.manifest_id());
    assert_ne!(
        left.core_subject_id().unwrap(),
        changed.core_subject_id().unwrap()
    );
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_assurance_core::{DigestSha256, StableId};
use symthaea_assurance_subject::{
    AiSubjectManifest, AiSurfaceKind, MaterialCommitment, SurfaceBinding, SurfaceLocator,
    SurfaceProfile, SurfaceState, UnavailabilityReason,
};

fn id(value: &str) -> StableId {
    StableId::new(value).unwrap()
}

fn digest(value: &str) -> DigestSha256 {
    DigestSha256::new(value).unwrap()
}

#[test]
fn canonical_subject_golden_vector_v1() {
    let profile = SurfaceProfile::new(
        id("golden-profile"),
        vec![AiSurfaceKind::Model, AiSurfaceKind::Custom(id("aaa"))],
    )
    .unwrap();

    assert_eq!(
        profile.digest().as_str(),
        "2b328394cac0ede56082fc9e66622f5a8921a52cdd298f177d7ace4503fb1af1"
    );

    let manifest = AiSubjectManifest::new(
        id("golden-agent"),
        profile,
        vec![
            SurfaceBinding::new(
                AiSurfaceKind::Model,
                SurfaceLocator::new(Some(id("provider-a")), id("rolling-model"), Some(id("v7"))),
                SurfaceState::Unavailable(UnavailabilityReason::ProviderDoesNotExpose),
            ),
            SurfaceBinding::new(
                AiSurfaceKind::Custom(id("aaa")),
                SurfaceLocator::new(None, id("custom-surface"), None),
                SurfaceState::Known(MaterialCommitment::artifact_bytes(digest(
                    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                ))),
            ),
        ],
    )
    .unwrap();

    assert_eq!(
        manifest.manifest_id().as_str(),
        "7ed1d417bf78b112ccd732ad8223156dea317c8ecfaf971e05301e86171a9e17"
    );
}

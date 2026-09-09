// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Qualification of the fabrication compatibility surface after the shared
//! digest cutover. These assignments are intentionally compile-time type
//! identity checks, not merely equal-byte comparisons.

use symthaea_evidence_plane::crypto_digest::{
    Sha256 as SharedSha256, Sha256Digest as SharedDigest, sha256 as shared_sha256,
};
use symthaea_fabrication_kernel::crypto_digest::{
    Sha256 as CompatSha256, Sha256Digest as CompatDigest, sha256 as compat_sha256,
};
use symthaea_fabrication_kernel::{
    Sha256 as RootSha256, Sha256Digest as RootDigest, sha256 as root_sha256,
};

#[test]
fn fabrication_module_and_root_are_exact_shared_types() {
    let shared: SharedDigest = shared_sha256(b"shared-digest-migration");
    let compat: CompatDigest = shared;
    let root: RootDigest = compat;

    let back_to_shared: SharedDigest = root;
    assert_eq!(back_to_shared, shared_sha256(b"shared-digest-migration"));
    assert_eq!(compat_sha256(b"shared-digest-migration"), back_to_shared);
    assert_eq!(root_sha256(b"shared-digest-migration"), back_to_shared);
}

#[test]
fn incremental_hasher_is_the_same_type_across_all_paths() {
    let shared = SharedSha256::new();
    let mut compat: CompatSha256 = shared;
    compat.update(b"abc");
    let root: RootSha256 = compat;
    assert_eq!(root.finalize(), shared_sha256(b"abc"));
}

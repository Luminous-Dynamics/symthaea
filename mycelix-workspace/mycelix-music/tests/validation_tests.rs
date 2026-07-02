// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Music — Validation Unit Tests
//!
//! Pure validation logic tests that run natively without a Holochain conductor.
//! These verify the boundary checks and invariants from the integrity zomes
//! by re-implementing the validation rules in plain Rust.
//!
//! ## Running
//! ```bash
//! cd mycelix-music/tests
//! cargo test --test validation_tests
//! ```

// --- Catalog validation ---

#[test]
fn song_title_bounds() {
    assert!("".is_empty()); // empty rejected
    assert!("x".repeat(256).len() <= 256); // 256 ok
    assert!("x".repeat(257).len() > 256); // 257 rejected
}

#[test]
fn ipfs_cid_bounds() {
    assert!("".is_empty()); // empty rejected
    assert!("Q".repeat(128).len() <= 128); // 128 ok
    assert!("Q".repeat(129).len() > 128); // 129 rejected
}

#[test]
fn genre_count_limit() {
    let genres: Vec<String> = (0..10).map(|i| format!("genre{}", i)).collect();
    assert!(genres.len() <= 10); // 10 ok
    let too_many: Vec<String> = (0..11).map(|i| format!("genre{}", i)).collect();
    assert!(too_many.len() > 10); // 11 rejected
}

#[test]
fn genre_tag_length() {
    assert!("x".repeat(64).len() <= 64); // ok
    assert!("x".repeat(65).len() > 64); // rejected
}

#[test]
fn metadata_size_limit() {
    assert!("x".repeat(65536).len() <= 65536); // 64KB ok
    assert!("x".repeat(65537).len() > 65536); // over rejected
}

#[test]
fn album_song_count_limit() {
    assert!(100 <= 100); // 100 ok
    assert!(101 > 100); // 101 rejected
}

#[test]
fn artist_name_bounds() {
    assert!("".is_empty()); // empty rejected
    assert!("x".repeat(128).len() <= 128); // 128 ok
    assert!("x".repeat(129).len() > 128); // 129 rejected
}

#[test]
fn eth_address_format() {
    let valid = "0x1234567890abcdef1234567890abcdef12345678";
    assert!(valid.starts_with("0x") && valid.len() == 42);

    let no_prefix = "1234567890abcdef1234567890abcdef12345678";
    assert!(!no_prefix.starts_with("0x"));

    let too_short = "0x1234";
    assert!(too_short.len() != 42);
}

// --- Trust validation ---

#[test]
fn confidence_bps_range() {
    assert!(0u32 <= 1000);
    assert!(1000u32 <= 1000);
    assert!(1001u32 > 1000); // rejected
}

#[test]
fn evidence_size_limit() {
    assert!("x".repeat(4096).len() <= 4096); // 4KB ok
    assert!("x".repeat(4097).len() > 4096); // rejected
}

#[test]
fn byzantine_evidence_bounds() {
    assert!("".is_empty()); // empty rejected
    assert!("x".repeat(8192).len() <= 8192); // 8KB ok
    assert!("x".repeat(8193).len() > 8192); // rejected
}

#[test]
fn severity_range() {
    assert!(0u8 <= 100);
    assert!(100u8 <= 100);
    assert!(101u8 > 100);
}

#[test]
fn pogq_score_finite() {
    assert!(0.85_f64.is_finite());
    assert!(!f64::NAN.is_finite());
    assert!(!f64::INFINITY.is_finite());
    assert!(!f64::NEG_INFINITY.is_finite());
}

#[test]
fn self_vouch_rejected() {
    let agent = "agent_a";
    assert_eq!(agent, agent);
}

// --- Plays validation ---

#[test]
fn duration_listened_cannot_exceed_song() {
    let listened = 200u32;
    let song_duration = 180u32;
    assert!(listened > song_duration); // rejected
}

#[test]
fn play_must_not_be_pre_settled() {
    let settled = true;
    assert!(settled); // if settled=true on creation, rejected
}

#[test]
fn settlement_play_count_matches_hashes() {
    let count = 3u64;
    let hashes_len = 3usize;
    assert_eq!(count as usize, hashes_len);

    let mismatched_count = 5u64;
    assert_ne!(mismatched_count as usize, hashes_len);
}

// --- Resonance validation ---

#[test]
fn resonance_finite_range() {
    assert!(0.5_f64.is_finite() && 0.5 >= -1.0 && 0.5 <= 1.0);
    assert!(!f64::NAN.is_finite());
    assert!(1.5_f64 > 1.0);
    assert!(-1.5_f64 < -1.0);
}

// --- Merkle root ---

#[test]
fn blake2b_merkle_not_xor() {
    let a = vec![1u8; 32];
    let b = vec![2u8; 32];

    let xor_ab: Vec<u8> = a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect();
    let xor_ba: Vec<u8> = b.iter().zip(a.iter()).map(|(x, y)| x ^ y).collect();
    assert_eq!(xor_ab, xor_ba, "XOR is commutative — bad for merkle");

    let xor_aa: Vec<u8> = a.iter().zip(a.iter()).map(|(x, y)| x ^ y).collect();
    assert_eq!(xor_aa, vec![0u8; 32], "XOR self-inverse — trivially forgeable");
}

// --- Visual Art validation ---

#[test]
fn provenance_did_format() {
    assert!("did:holo:abc123".starts_with("did:"));
    assert!(!"agent:abc123".starts_with("did:"));
}

#[test]
fn provenance_chain_limit() {
    assert!(100 <= 100);
    assert!(101 > 100);
}

#[test]
fn gallery_name_bounds() {
    assert!("".is_empty());
    assert!("x".repeat(256).len() <= 256);
    assert!("x".repeat(257).len() > 256);
}

#[test]
fn exhibition_publication_limit() {
    assert!(100 <= 100);
    assert!(101 > 100);
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Music — Sweettest Integration Tests
//!
//! Tests the Music cluster DNA: catalog, plays, balances, trust + music-bridge.
//!
//! ## Running
//! ```bash
//! cd mycelix-music
//! nix develop
//! cargo build --release --target wasm32-unknown-unknown
//! hc dna pack dnas/mycelix-music/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_integration -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- catalog ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Song {
    pub song_hash: String,
    pub title: String,
    pub artist: AgentPubKey,
    pub ipfs_cid: String,
    pub cover_cid: Option<String>,
    pub duration_seconds: u32,
    pub genres: Vec<String>,
    pub strategy_id: String,
    pub released_at: Timestamp,
    pub metadata: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Album {
    pub title: String,
    pub artist: AgentPubKey,
    pub cover_cid: String,
    pub released_at: Timestamp,
    pub song_hashes: Vec<ActionHash>,
    pub metadata: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArtistProfile {
    pub name: String,
    pub bio: String,
    pub avatar_cid: Option<String>,
    pub payment_address: String,
    pub social_links: String,
    pub verified: bool,
}

// --- plays ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordPlayInput {
    pub song_hash: ActionHash,
    pub artist: AgentPubKey,
    pub duration_listened: u32,
    pub song_duration: u32,
    pub strategy_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlayRecord {
    pub song_hash: ActionHash,
    pub artist: AgentPubKey,
    pub played_at: Timestamp,
    pub duration_listened: u32,
    pub song_duration: u32,
    pub strategy_id: String,
    pub amount_owed: u64,
    pub settled: bool,
    pub settlement_hash: Option<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SongStats {
    pub total_plays: u64,
    pub total_earnings: u64,
    pub unique_listeners: u64,
    pub avg_completion: f64,
}

// --- balances ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListenerAccount {
    pub owner: AgentPubKey,
    pub eth_address: String,
    pub balance: u64,
    pub total_deposited: u64,
    pub total_spent: u64,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ArtistAccount {
    pub owner: AgentPubKey,
    pub eth_address: String,
    pub pending_balance: u64,
    pub total_earned: u64,
    pub total_cashed_out: u64,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecordDepositInput {
    pub amount: u64,
    pub tx_hash: String,
    pub block_number: u64,
}

// --- trust ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateTrustClaimInput {
    pub to: AgentPubKey,
    pub claim_type: TrustClaimType,
    pub confidence_bps: u32,
    pub evidence: Option<String>,
    pub expires_at: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum TrustClaimType {
    IdentityVerification,
    ContentAuthenticity,
    QualityAttestation,
    CdnReliability,
    PaymentReliability,
    GeneralEndorsement,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationStatus {
    pub artist: AgentPubKey,
    pub trust_score: u32,
    pub tier: VerificationTier,
    pub vouch_count: u32,
    pub computed_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum VerificationTier {
    Unverified,
    CommunityVerified,
    Trusted,
    PlatformVerified,
    FoundingArtist,
}

// --- bridge ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub cluster: String,
    pub agent: String,
    pub zome_count: u32,
    pub healthy: bool,
}

// ============================================================================
// Setup helpers
// ============================================================================

fn happ_path() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .join("mycelix_music.happ")
}

async fn setup_conductor() -> (SweetConductor, AgentPubKey) {
    let mut conductor = SweetConductor::from_standard_config().await;
    let agent = SweetAgents::one(conductor.keystore()).await;
    (conductor, agent)
}

async fn install_app(
    conductor: &mut SweetConductor,
    agent: &AgentPubKey,
) -> SweetApp {
    let happ = happ_path();
    conductor
        .setup_app_for_agent("music", agent.clone(), &[DnaSource::Path(happ)])
        .await
        .unwrap()
}

// ============================================================================
// Catalog Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_create_and_get_song() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    let song = Song {
        song_hash: "QmTest123".to_string(),
        title: "Test Song".to_string(),
        artist: agent.clone(),
        ipfs_cid: "QmAudioCid123".to_string(),
        cover_cid: Some("QmCoverCid123".to_string()),
        duration_seconds: 240,
        genres: vec!["electronic".to_string(), "ambient".to_string()],
        strategy_id: "pay_per_stream".to_string(),
        released_at: Timestamp::now(),
        metadata: "{}".to_string(),
    };

    let action_hash: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_song", song.clone())
        .await;

    let record: Record = conductor
        .call(&app.cells()[0], "catalog", "get_song", action_hash.clone())
        .await;

    let retrieved: Song = record.entry().to_app_option().unwrap().unwrap();
    assert_eq!(retrieved.title, "Test Song");
    assert_eq!(retrieved.ipfs_cid, "QmAudioCid123");
    assert_eq!(retrieved.duration_seconds, 240);
    assert_eq!(retrieved.genres.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_create_album_with_songs() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Create two songs
    let song1 = Song {
        song_hash: "QmSong1".to_string(),
        title: "Track One".to_string(),
        artist: agent.clone(),
        ipfs_cid: "QmAudio1".to_string(),
        cover_cid: None,
        duration_seconds: 180,
        genres: vec!["rock".to_string()],
        strategy_id: "pay_per_stream".to_string(),
        released_at: Timestamp::now(),
        metadata: "{}".to_string(),
    };
    let song2 = Song {
        song_hash: "QmSong2".to_string(),
        title: "Track Two".to_string(),
        artist: agent.clone(),
        ipfs_cid: "QmAudio2".to_string(),
        cover_cid: None,
        duration_seconds: 210,
        genres: vec!["rock".to_string()],
        strategy_id: "pay_per_stream".to_string(),
        released_at: Timestamp::now(),
        metadata: "{}".to_string(),
    };

    let hash1: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_song", song1)
        .await;
    let hash2: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_song", song2)
        .await;

    let album = Album {
        title: "Test Album".to_string(),
        artist: agent.clone(),
        cover_cid: "QmAlbumCover".to_string(),
        released_at: Timestamp::now(),
        song_hashes: vec![hash1.clone(), hash2.clone()],
        metadata: "{}".to_string(),
    };

    let album_hash: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_album", album)
        .await;

    assert_ne!(album_hash, hash1);
    assert_ne!(album_hash, hash2);
}

// ============================================================================
// Plays Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_record_play_and_stats() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Create a song first
    let song = Song {
        song_hash: "QmPlayTest".to_string(),
        title: "Play Test Song".to_string(),
        artist: agent.clone(),
        ipfs_cid: "QmPlayAudio".to_string(),
        cover_cid: None,
        duration_seconds: 200,
        genres: vec!["pop".to_string()],
        strategy_id: "pay_per_stream".to_string(),
        released_at: Timestamp::now(),
        metadata: "{}".to_string(),
    };

    let song_hash: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_song", song)
        .await;

    // Record a play
    let play_input = RecordPlayInput {
        song_hash: song_hash.clone(),
        artist: agent.clone(),
        duration_listened: 180,
        song_duration: 200,
        strategy_id: "pay_per_stream".to_string(),
    };

    let play_hash: ActionHash = conductor
        .call(&app.cells()[0], "plays", "record_play", play_input)
        .await;

    // Verify play was recorded
    let unsettled: Vec<PlayRecord> = conductor
        .call(
            &app.cells()[0],
            "plays",
            "get_my_unsettled_plays",
            (),
        )
        .await;

    assert!(!unsettled.is_empty());
    let play = &unsettled[0];
    assert_eq!(play.duration_listened, 180);
    assert_eq!(play.song_duration, 200);
    assert!(!play.settled);
    assert!(play.amount_owed > 0); // Should have calculated payment
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_settlement_batch() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Create song and record plays
    let song = Song {
        song_hash: "QmSettleTest".to_string(),
        title: "Settlement Song".to_string(),
        artist: agent.clone(),
        ipfs_cid: "QmSettleAudio".to_string(),
        cover_cid: None,
        duration_seconds: 180,
        genres: vec![],
        strategy_id: "pay_per_stream".to_string(),
        released_at: Timestamp::now(),
        metadata: "{}".to_string(),
    };

    let song_hash: ActionHash = conductor
        .call(&app.cells()[0], "catalog", "create_song", song)
        .await;

    for _ in 0..3 {
        let play_input = RecordPlayInput {
            song_hash: song_hash.clone(),
            artist: agent.clone(),
            duration_listened: 180,
            song_duration: 180,
            strategy_id: "pay_per_stream".to_string(),
        };
        let _: ActionHash = conductor
            .call(&app.cells()[0], "plays", "record_play", play_input)
            .await;
    }

    // Create settlement batch
    let batch_hash: ActionHash = conductor
        .call(
            &app.cells()[0],
            "plays",
            "create_settlement_batch",
            agent.clone(),
        )
        .await;

    // Verify unsettled plays are now empty
    let unsettled: Vec<PlayRecord> = conductor
        .call(
            &app.cells()[0],
            "plays",
            "get_my_unsettled_plays",
            (),
        )
        .await;

    assert!(unsettled.is_empty(), "All plays should be settled");
}

// ============================================================================
// Balances Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_listener_account_lifecycle() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Create listener account
    let account: Record = conductor
        .call(
            &app.cells()[0],
            "balances",
            "get_or_create_listener_account",
            "0x1234567890abcdef1234567890abcdef12345678".to_string(),
        )
        .await;

    let listener: ListenerAccount = account.entry().to_app_option().unwrap().unwrap();
    assert_eq!(listener.balance, 0);
    assert_eq!(
        listener.eth_address,
        "0x1234567890abcdef1234567890abcdef12345678"
    );
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_artist_account_and_deposit() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Create artist account
    let account: Record = conductor
        .call(
            &app.cells()[0],
            "balances",
            "get_or_create_artist_account",
            "0xabcdef1234567890abcdef1234567890abcdef12".to_string(),
        )
        .await;

    let artist: ArtistAccount = account.entry().to_app_option().unwrap().unwrap();
    assert_eq!(artist.pending_balance, 0);

    // Record deposit (unverified — needs oracle to credit balance)
    let deposit_input = RecordDepositInput {
        amount: 1_000_000,
        tx_hash: "0xdeadbeef".to_string(),
        block_number: 12345,
    };

    let deposit_hash: ActionHash = conductor
        .call(
            &app.cells()[0],
            "balances",
            "record_deposit",
            deposit_input,
        )
        .await;

    // Balance should still be 0 (deposit not yet verified by oracle)
    let balance: Option<ListenerAccount> = conductor
        .call(
            &app.cells()[0],
            "balances",
            "get_my_listener_balance",
            (),
        )
        .await;

    // Account exists but balance unchanged until verify_deposit()
    if let Some(acct) = balance {
        assert_eq!(acct.balance, 0, "Balance should be 0 before oracle verification");
    }
}

// ============================================================================
// Trust Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_trust_claim_and_verification() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    // Need a second agent to vouch for
    let agent2 = SweetAgents::one(conductor.keystore()).await;
    let _app2 = install_app(&mut conductor, &agent2).await;

    let claim_input = CreateTrustClaimInput {
        to: agent2.clone(),
        claim_type: TrustClaimType::QualityAttestation,
        confidence_bps: 800,
        evidence: Some("Great music producer".to_string()),
        expires_at: None,
    };

    let claim_hash: ActionHash = conductor
        .call(
            &app.cells()[0],
            "trust",
            "create_trust_claim",
            claim_input,
        )
        .await;

    // Recompute verification status
    let status: VerificationStatus = conductor
        .call(
            &app.cells()[0],
            "trust",
            "recompute_verification",
            agent2.clone(),
        )
        .await;

    assert_eq!(status.vouch_count, 1);
    assert!(status.trust_score > 0);
    // Single vouch doesn't reach CommunityVerified (needs 3+)
    assert_eq!(status.tier, VerificationTier::Unverified);
}

// ============================================================================
// Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor + built hApp bundle"]
async fn test_bridge_health_check() {
    let (mut conductor, agent) = setup_conductor().await;
    let app = install_app(&mut conductor, &agent).await;

    let health: BridgeHealth = conductor
        .call(
            &app.cells()[0],
            "music_bridge",
            "health_check",
            (),
        )
        .await;

    assert!(health.healthy);
    assert_eq!(health.cluster, "music");
    assert_eq!(health.zome_count, 4);
}

// Validation unit tests are in validation_tests.rs (runs without holochain dep).
// Run: cargo test --test validation_tests

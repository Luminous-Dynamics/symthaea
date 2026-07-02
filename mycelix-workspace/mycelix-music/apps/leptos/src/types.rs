// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Mirror types from the music DNA zomes.
//! These must be serde-compatible with the Holochain entry types.

use serde::{Deserialize, Serialize};

// --- Catalog ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Song {
    pub song_hash: String,
    pub title: String,
    pub artist: String, // AgentPubKey serialized
    pub ipfs_cid: String,
    pub cover_cid: Option<String>,
    pub duration_seconds: u32,
    pub genres: Vec<String>,
    pub strategy_id: String,
    pub released_at: i64, // Timestamp as micros
    pub metadata: String,
}

impl Song {
    /// Construct an IPFS gateway URL for the audio file.
    pub fn audio_url(&self) -> String {
        format!("https://ipfs.io/ipfs/{}", self.ipfs_cid)
    }

    /// Construct an IPFS gateway URL for the cover art.
    pub fn cover_url(&self) -> Option<String> {
        self.cover_cid
            .as_ref()
            .map(|cid| format!("https://ipfs.io/ipfs/{}", cid))
    }

    /// Format duration as mm:ss.
    pub fn duration_display(&self) -> String {
        let mins = self.duration_seconds / 60;
        let secs = self.duration_seconds % 60;
        format!("{mins}:{secs:02}")
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Album {
    pub title: String,
    pub artist: String,
    pub cover_cid: String,
    pub released_at: i64,
    pub song_hashes: Vec<String>,
    pub metadata: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtistProfile {
    pub name: String,
    pub bio: String,
    pub avatar_cid: Option<String>,
    pub payment_address: String,
    pub social_links: String,
    pub verified: bool,
}

// --- Plays ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RecordPlayInput {
    pub song_hash: String,
    pub artist: String,
    pub duration_listened: u32,
    pub song_duration: u32,
    pub strategy_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayRecord {
    pub song_hash: String,
    pub artist: String,
    pub played_at: i64,
    pub duration_listened: u32,
    pub song_duration: u32,
    pub strategy_id: String,
    pub amount_owed: u64,
    pub settled: bool,
    pub settlement_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SongStats {
    pub total_plays: u64,
    pub total_earnings: u64,
    pub unique_listeners: u64,
    pub avg_completion: f64,
}

// --- Balances ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListenerAccount {
    pub owner: String,
    pub eth_address: String,
    pub balance: u64,
    pub total_deposited: u64,
    pub total_spent: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtistAccount {
    pub owner: String,
    pub eth_address: String,
    pub pending_balance: u64,
    pub total_earned: u64,
    pub total_cashed_out: u64,
}

// --- Trust ---

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationStatus {
    pub artist: String,
    pub trust_score: u32,
    pub tier: VerificationTier,
    pub vouch_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum VerificationTier {
    Unverified,
    CommunityVerified,
    Trusted,
    PlatformVerified,
    FoundingArtist,
}

impl VerificationTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Unverified => "Unverified",
            Self::CommunityVerified => "Community Verified",
            Self::Trusted => "Trusted",
            Self::PlatformVerified => "Platform Verified",
            Self::FoundingArtist => "Founding Artist",
        }
    }

    pub fn color_class(&self) -> &'static str {
        match self {
            Self::Unverified => "tier-unverified",
            Self::CommunityVerified => "tier-community",
            Self::Trusted => "tier-trusted",
            Self::PlatformVerified => "tier-platform",
            Self::FoundingArtist => "tier-founding",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RepeatMode { None, One, All }
impl RepeatMode {
    pub fn icon(&self) -> &'static str { match self { Self::None => "\u{1f501}", Self::One => "\u{1f502}", Self::All => "\u{1f501}" } }
    pub fn next(&self) -> Self { match self { Self::None => Self::All, Self::All => Self::One, Self::One => Self::None } }
}

pub fn mock_verification(artist: &str) -> VerificationTier {
    match artist { "mock-artist-1" => VerificationTier::CommunityVerified, "mock-artist-2" => VerificationTier::Trusted, _ => VerificationTier::Unverified }
}

// --- Mock data for when conductor is unavailable ---

pub fn mock_songs() -> Vec<Song> {
    vec![
        Song {
            song_hash: "mock-1".into(),
            title: "Decentralized Dreams".into(),
            artist: "mock-artist-1".into(),
            ipfs_cid: "QmDemo1".into(),
            cover_cid: None,
            duration_seconds: 234,
            genres: vec!["Electronic".into(), "Ambient".into()],
            strategy_id: "pay_per_stream".into(),
            released_at: 0,
            metadata: "{}".into(),
        },
        Song {
            song_hash: "mock-2".into(),
            title: "Zero-Cost Serenade".into(),
            artist: "mock-artist-2".into(),
            ipfs_cid: "QmDemo2".into(),
            cover_cid: None,
            duration_seconds: 187,
            genres: vec!["Indie".into(), "Folk".into()],
            strategy_id: "gift".into(),
            released_at: 0,
            metadata: "{}".into(),
        },
        Song {
            song_hash: "mock-3".into(),
            title: "Mycelium Network".into(),
            artist: "mock-artist-1".into(),
            ipfs_cid: "QmDemo3".into(),
            cover_cid: None,
            duration_seconds: 312,
            genres: vec!["Rock".into()],
            strategy_id: "patronage".into(),
            released_at: 0,
            metadata: "{}".into(),
        },
    ]
}

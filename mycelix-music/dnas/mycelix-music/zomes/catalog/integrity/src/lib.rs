//! Catalog Integrity Zome
//!
//! Defines the entry types and validation rules for the music catalog.
//! Songs, albums, and artist profiles are stored here.

use hdi::prelude::*;

/// Song entry - core content unit in Mycelix Music
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Song {
    /// Unique identifier (content hash)
    pub song_hash: String,
    /// Song title
    pub title: String,
    /// Artist's agent public key
    pub artist: AgentPubKey,
    /// IPFS CID for the audio file
    pub ipfs_cid: String,
    /// Cover art IPFS CID (optional)
    pub cover_cid: Option<String>,
    /// Duration in seconds
    pub duration_seconds: u32,
    /// Genre tags
    pub genres: Vec<String>,
    /// Economic strategy ID
    pub strategy_id: String,
    /// Release timestamp
    pub released_at: Timestamp,
    /// Additional metadata (JSON)
    pub metadata: String,
}

/// Album entry - collection of songs
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Album {
    /// Album title
    pub title: String,
    /// Artist's agent public key
    pub artist: AgentPubKey,
    /// Cover art IPFS CID
    pub cover_cid: String,
    /// Release timestamp
    pub released_at: Timestamp,
    /// Song hashes in order
    pub song_hashes: Vec<ActionHash>,
    /// Additional metadata
    pub metadata: String,
}

/// Artist profile entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ArtistProfile {
    /// Display name
    pub name: String,
    /// Bio/description
    pub bio: String,
    /// Profile image IPFS CID
    pub avatar_cid: Option<String>,
    /// Ethereum address for payments
    pub payment_address: String,
    /// Social links (JSON)
    pub social_links: String,
    /// Verified status (set by trust zome)
    pub verified: bool,
}

/// Link types for the catalog
#[hdk_link_types]
pub enum LinkTypes {
    /// Artist -> Songs they created
    ArtistToSongs,
    /// Artist -> Albums they created
    ArtistToAlbums,
    /// Album -> Songs in the album
    AlbumToSongs,
    /// Genre tag -> Songs
    GenreToSongs,
    /// All songs anchor
    AllSongs,
    /// All artists anchor
    AllArtists,
}

/// Entry types for the catalog zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Song(Song),
    Album(Album),
    ArtistProfile(ArtistProfile),
}

/// Validate song creation
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Song(song) => validate_create_song(song, action),
                EntryTypes::Album(album) => validate_create_album(album, action),
                EntryTypes::ArtistProfile(profile) => validate_create_profile(profile, action),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Song(song) => {
                    validate_update_song(song, action, original_action_hash)
                }
                EntryTypes::Album(album) => {
                    validate_update_album(album, action, original_action_hash)
                }
                EntryTypes::ArtistProfile(profile) => {
                    validate_update_profile(profile, action, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ArtistToSongs => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ArtistToAlbums => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AlbumToSongs => Ok(ValidateCallbackResult::Valid),
            LinkTypes::GenreToSongs => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllSongs => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllArtists => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_song(song: Song, action: Create) -> ExternResult<ValidateCallbackResult> {
    // Song must have a title
    if song.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Song title cannot be empty".to_string(),
        ));
    }

    // Song must have an IPFS CID
    if song.ipfs_cid.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Song must have an IPFS CID".to_string(),
        ));
    }

    // Artist must be the author
    if song.artist != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Song artist must match the action author".to_string(),
        ));
    }

    // Duration must be positive
    if song.duration_seconds == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Song duration must be greater than 0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_album(album: Album, action: Create) -> ExternResult<ValidateCallbackResult> {
    // Album must have a title
    if album.title.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Album title cannot be empty".to_string(),
        ));
    }

    // Artist must be the author
    if album.artist != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Album artist must match the action author".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_profile(
    _profile: ArtistProfile,
    _action: Create,
) -> ExternResult<ValidateCallbackResult> {
    // Profiles can be created by anyone for themselves
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_song(
    _song: Song,
    action: Update,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Only the original author can update
    let original_action = must_get_action(original_action_hash)?;
    if original_action.action().author() != &action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only the original author can update a song".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_album(
    _album: Album,
    action: Update,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_action = must_get_action(original_action_hash)?;
    if original_action.action().author() != &action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only the original author can update an album".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_profile(
    _profile: ArtistProfile,
    action: Update,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_action = must_get_action(original_action_hash)?;
    if original_action.action().author() != &action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only the original author can update their profile".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

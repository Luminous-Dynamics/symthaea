// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Catalog Coordinator Zome
//!
//! Provides the callable functions for managing the music catalog.
//! Handles song uploads, album creation, and artist profile management.
//! Holochain 0.6 compatible (hdk 0.6)

use catalog_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_civic, civic_requirement_proposal, GovernanceRequirement,
};

/// Consciousness gate: requires at least Participant tier for write operations.
fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<()> {
    gate_civic("music_bridge", requirement, action_name).map(|_| ())
}

/// Helper to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Create a new song entry
#[hdk_extern]
pub fn create_song(song: Song) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_proposal(), "create_song")?;
    let action_hash = create_entry(EntryTypes::Song(song.clone()))?;

    // Link from artist to song
    let artist_path = Path::from(format!("artists/{}", song.artist));
    let artist_hash = ensure_path(artist_path, LinkTypes::ArtistToSongs)?;
    create_link(
        artist_hash,
        action_hash.clone(),
        LinkTypes::ArtistToSongs,
        (),
    )?;

    // Link to all songs anchor
    let all_songs_path = Path::from("all_songs");
    let all_songs_hash = ensure_path(all_songs_path, LinkTypes::AllSongs)?;
    create_link(
        all_songs_hash,
        action_hash.clone(),
        LinkTypes::AllSongs,
        (),
    )?;

    // Link from each genre
    for genre in &song.genres {
        let genre_path = Path::from(format!("genres/{}", genre.to_lowercase()));
        let genre_hash = ensure_path(genre_path, LinkTypes::GenreToSongs)?;
        create_link(
            genre_hash,
            action_hash.clone(),
            LinkTypes::GenreToSongs,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get a song by its action hash
#[hdk_extern]
pub fn get_song(action_hash: ActionHash) -> ExternResult<Option<Song>> {
    let record = get(action_hash, GetOptions::default())?;
    match record {
        Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
        None => Ok(None),
    }
}

/// Get all songs by an artist
#[hdk_extern]
pub fn get_songs_by_artist(artist: AgentPubKey) -> ExternResult<Vec<Song>> {
    let artist_path = Path::from(format!("artists/{}", artist));
    let typed_path = artist_path.typed(LinkTypes::ArtistToSongs)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::ArtistToSongs)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut songs = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(song) = get_song(action_hash)? {
                songs.push(song);
            }
        }
    }
    Ok(songs)
}

/// Get all songs (paginated)
#[derive(Serialize, Deserialize, Debug)]
pub struct GetAllSongsInput {
    pub limit: usize,
    pub offset: usize,
}

#[hdk_extern]
pub fn get_all_songs(input: GetAllSongsInput) -> ExternResult<Vec<Song>> {
    let all_songs_path = Path::from("all_songs");
    let typed_path = all_songs_path.typed(LinkTypes::AllSongs)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AllSongs)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut songs = Vec::new();
    for link in links.into_iter().skip(input.offset).take(input.limit) {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(song) = get_song(action_hash)? {
                songs.push(song);
            }
        }
    }
    Ok(songs)
}

/// Get songs by genre
#[hdk_extern]
pub fn get_songs_by_genre(genre: String) -> ExternResult<Vec<Song>> {
    let genre_path = Path::from(format!("genres/{}", genre.to_lowercase()));
    let typed_path = genre_path.typed(LinkTypes::GenreToSongs)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::GenreToSongs)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut songs = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(song) = get_song(action_hash)? {
                songs.push(song);
            }
        }
    }
    Ok(songs)
}

/// Create an album
#[hdk_extern]
pub fn create_album(album: Album) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_proposal(), "create_album")?;
    let action_hash = create_entry(EntryTypes::Album(album.clone()))?;

    // Link from artist to album
    let artist_path = Path::from(format!("artists/{}", album.artist));
    let artist_hash = ensure_path(artist_path, LinkTypes::ArtistToAlbums)?;
    create_link(
        artist_hash,
        action_hash.clone(),
        LinkTypes::ArtistToAlbums,
        (),
    )?;

    // Link album to its songs
    for song_hash in &album.song_hashes {
        create_link(
            action_hash.clone(),
            song_hash.clone(),
            LinkTypes::AlbumToSongs,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get an album with its songs
#[derive(Serialize, Deserialize, Debug)]
pub struct AlbumWithSongs {
    pub album: Album,
    pub songs: Vec<Song>,
}

#[hdk_extern]
pub fn get_album_with_songs(action_hash: ActionHash) -> ExternResult<Option<AlbumWithSongs>> {
    let record = get(action_hash.clone(), GetOptions::default())?;
    let album: Option<Album> = match record {
        Some(r) => r.entry().to_app_option().map_err(|e| wasm_error!(e))?,
        None => return Ok(None),
    };

    let album = match album {
        Some(a) => a,
        None => return Ok(None),
    };

    // Get linked songs
    let filter = LinkTypeFilter::try_from(LinkTypes::AlbumToSongs)?;
    let links = get_links(
        LinkQuery::new(action_hash, filter),
        GetStrategy::default(),
    )?;

    let mut songs = Vec::new();
    for link in links {
        if let Some(song_hash) = link.target.into_action_hash() {
            if let Some(song) = get_song(song_hash)? {
                songs.push(song);
            }
        }
    }

    Ok(Some(AlbumWithSongs { album, songs }))
}

/// Create or update artist profile
#[hdk_extern]
pub fn set_artist_profile(profile: ArtistProfile) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_proposal(), "set_artist_profile")?;
    let my_agent = agent_info()?.agent_initial_pubkey;
    let profile_path = Path::from(format!("profile/{}", my_agent));
    let typed_path = profile_path.typed(LinkTypes::AllArtists)?;

    // Check if profile already exists
    let filter = LinkTypeFilter::try_from(LinkTypes::AllArtists)?;
    let existing_links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter.clone()),
        GetStrategy::default(),
    )?;

    if let Some(link) = existing_links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            // Update existing profile
            let updated_hash = update_entry(action_hash, EntryTypes::ArtistProfile(profile))?;
            return Ok(updated_hash);
        }
    }

    // Create new profile
    let action_hash = create_entry(EntryTypes::ArtistProfile(profile))?;

    // Link from profile path
    typed_path.ensure()?;
    create_link(
        typed_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllArtists,
        (),
    )?;

    // Link to all artists
    let all_artists_path = Path::from("all_artists");
    let all_artists_hash = ensure_path(all_artists_path, LinkTypes::AllArtists)?;
    create_link(
        all_artists_hash,
        action_hash.clone(),
        LinkTypes::AllArtists,
        (),
    )?;

    Ok(action_hash)
}

/// Get artist profile by agent pub key
#[hdk_extern]
pub fn get_artist_profile(agent: AgentPubKey) -> ExternResult<Option<ArtistProfile>> {
    let profile_path = Path::from(format!("profile/{}", agent));
    let typed_path = profile_path.typed(LinkTypes::AllArtists)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AllArtists)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            let record = get(action_hash, GetOptions::default())?;
            return match record {
                Some(r) => Ok(r.entry().to_app_option().map_err(|e| wasm_error!(e))?),
                None => Ok(None),
            };
        }
    }

    Ok(None)
}

/// Get my profile
#[hdk_extern]
pub fn get_my_profile(_: ()) -> ExternResult<Option<ArtistProfile>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_artist_profile(my_agent)
}

/// Search songs by title (basic implementation)
#[hdk_extern]
pub fn search_songs(query: String) -> ExternResult<Vec<Song>> {
    let all_songs_path = Path::from("all_songs");
    let typed_path = all_songs_path.typed(LinkTypes::AllSongs)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AllSongs)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let query_lower = query.to_lowercase();
    let mut matches = Vec::new();

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(song) = get_song(action_hash)? {
                if song.title.to_lowercase().contains(&query_lower) {
                    matches.push(song);
                }
            }
        }
    }

    Ok(matches)
}

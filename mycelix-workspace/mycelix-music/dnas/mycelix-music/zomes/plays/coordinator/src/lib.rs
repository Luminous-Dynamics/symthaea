// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Plays Coordinator Zome
//!
//! ZERO-COST STREAMING: The Heart of Mycelix Music
//!
//! How it works:
//! 1. Listener plays a song -> PlayRecord created on THEIR source chain (FREE!)
//! 2. Plays accumulate locally with calculated amounts owed
//! 3. Periodically, plays are batched into SettlementBatches
//! 4. Only the batch settlement touches the blockchain (paid, but amortized)
//!
//! Result: Artists get paid for EVERY play, listeners pay near-zero fees
//! Holochain 0.6 compatible (hdk 0.6)

use hdk::prelude::*;
use plays_integrity::*;
use mycelix_bridge_common::{gate_civic, civic_requirement_voting, GovernanceRequirement};

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

/// Record a song play - THIS IS FREE (just writes to local source chain)
#[hdk_extern]
pub fn record_play(input: RecordPlayInput) -> ExternResult<ActionHash> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Calculate amount owed based on strategy
    let amount_owed = calculate_play_amount(&input.strategy_id, input.duration_listened, input.song_duration);

    let play = PlayRecord {
        song_hash: input.song_hash.clone(),
        artist: input.artist.clone(),
        played_at: sys_time()?,
        duration_listened: input.duration_listened,
        song_duration: input.song_duration,
        strategy_id: input.strategy_id,
        amount_owed,
        settled: false,
        settlement_hash: None,
    };

    let action_hash = create_entry(&EntryTypes::PlayRecord(play))?;

    // Link from listener to their plays
    let listener_path = Path::from(format!("listener_plays/{}", my_agent));
    let listener_hash = ensure_path(listener_path, LinkTypes::ListenerToPlays)?;
    create_link(
        listener_hash,
        action_hash.clone(),
        LinkTypes::ListenerToPlays,
        (),
    )?;

    // Link from song to plays (for artist analytics)
    create_link(
        input.song_hash,
        action_hash.clone(),
        LinkTypes::SongToPlays,
        (),
    )?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordPlayInput {
    pub song_hash: ActionHash,
    pub artist: AgentPubKey,
    pub duration_listened: u32,
    pub song_duration: u32,
    pub strategy_id: String,
}

/// Calculate payment amount based on strategy
fn calculate_play_amount(strategy_id: &str, duration_listened: u32, song_duration: u32) -> u64 {
    // Base rate: 0.001 USD per full play (in wei: ~400000000000000 at $0.40/xDAI)
    let base_rate: u64 = 400_000_000_000_000; // 0.0004 xDAI

    // Calculate completion percentage
    let completion = if song_duration > 0 {
        (duration_listened as f64 / song_duration as f64).min(1.0)
    } else {
        0.0
    };

    // Only count plays over 30 seconds or 50% completion
    if duration_listened < 30 && completion < 0.5 {
        return 0;
    }

    // Apply strategy multiplier
    let multiplier = match strategy_id {
        "premium" => 2.0,
        "patronage" => 1.5,
        "gift" => 0.0, // Gift economy = free
        "pay_per_stream" => 1.0,
        _ => 1.0,
    };

    ((base_rate as f64) * completion * multiplier) as u64
}

/// Get my unsettled plays
#[hdk_extern]
pub fn get_my_unsettled_plays(_: ()) -> ExternResult<Vec<PlayRecord>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let listener_path = Path::from(format!("listener_plays/{}", my_agent));
    let typed_path = listener_path.typed(LinkTypes::ListenerToPlays)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::ListenerToPlays)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut unsettled = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(play) = record
                    .entry()
                    .to_app_option::<PlayRecord>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if !play.settled {
                        unsettled.push(play);
                    }
                }
            }
        }
    }

    Ok(unsettled)
}

/// Get total amount I owe (unsettled plays)
#[hdk_extern]
pub fn get_my_balance_owed(_: ()) -> ExternResult<BalanceOwed> {
    let plays = get_my_unsettled_plays(())?;

    let mut total_amount: u64 = 0;
    let mut play_count: u64 = 0;
    let mut by_artist: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    for play in plays {
        total_amount += play.amount_owed;
        play_count += 1;
        *by_artist.entry(play.artist.to_string()).or_insert(0) += play.amount_owed;
    }

    Ok(BalanceOwed {
        total_amount,
        play_count,
        by_artist: by_artist.into_iter().collect(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BalanceOwed {
    pub total_amount: u64,
    pub play_count: u64,
    pub by_artist: Vec<(String, u64)>,
}

/// Create a settlement batch for an artist
#[hdk_extern]
pub fn create_settlement_batch(artist: AgentPubKey) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_voting(), "create_settlement_batch")?;
    let plays = get_my_unsettled_plays(())?;

    // Filter plays for this artist
    let artist_plays: Vec<PlayRecord> = plays
        .into_iter()
        .filter(|p| p.artist == artist)
        .collect();

    if artist_plays.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "No unsettled plays for this artist".to_string()
        )));
    }

    // Calculate totals
    let play_count = artist_plays.len() as u64;
    let total_amount: u64 = artist_plays.iter().map(|p| p.amount_owed).sum();

    // Collect play hashes (we need to get them from links)
    let my_agent = agent_info()?.agent_initial_pubkey;
    let listener_path = Path::from(format!("listener_plays/{}", my_agent));
    let typed_path = listener_path.typed(LinkTypes::ListenerToPlays)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::ListenerToPlays)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut play_hashes = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(play) = record
                    .entry()
                    .to_app_option::<PlayRecord>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if !play.settled && play.artist == artist {
                        play_hashes.push(action_hash);
                    }
                }
            }
        }
    }

    // Create merkle root (simplified - just hash all play hashes together)
    let merkle_root = compute_merkle_root(&play_hashes);

    let batch = SettlementBatch {
        artist: artist.clone(),
        play_count,
        total_amount,
        play_hashes: play_hashes.clone(),
        merkle_root,
        created_at: sys_time()?,
        status: SettlementStatus::Pending,
        tx_hash: None,
    };

    let batch_hash = create_entry(&EntryTypes::SettlementBatch(batch))?;

    // Link batch to artist
    let artist_settlements_path = Path::from(format!("settlements/{}", artist));
    let artist_settlements_hash = ensure_path(artist_settlements_path, LinkTypes::ArtistToSettlements)?;
    create_link(
        artist_settlements_hash,
        batch_hash.clone(),
        LinkTypes::ArtistToSettlements,
        (),
    )?;

    // Link plays to settlement
    for play_hash in play_hashes {
        create_link(
            play_hash,
            batch_hash.clone(),
            LinkTypes::PlayToSettlement,
            (),
        )?;
    }

    Ok(batch_hash)
}

/// Compute a BLAKE2b-256 merkle root from action hashes.
///
/// Uses proper cryptographic hashing (BLAKE2b) instead of XOR folding.
/// XOR is commutative and self-inverse, making it trivially forgeable.
fn compute_merkle_root(hashes: &[ActionHash]) -> Vec<u8> {
    use hdk::prelude::holo_hash::blake2b_256;

    if hashes.is_empty() {
        return vec![0u8; 32];
    }

    let mut current: Vec<Vec<u8>> = hashes
        .iter()
        .map(|h| h.get_raw_39().to_vec())
        .collect();

    while current.len() > 1 {
        let mut next = Vec::new();
        for chunk in current.chunks(2) {
            let combined = if chunk.len() == 2 {
                [chunk[0].as_slice(), chunk[1].as_slice()].concat()
            } else {
                // Odd leaf: duplicate it
                [chunk[0].as_slice(), chunk[0].as_slice()].concat()
            };
            next.push(blake2b_256(&combined).to_vec());
        }
        current = next;
    }

    current.into_iter().next().unwrap_or_else(|| vec![0u8; 32])
}

/// Get pending settlements for an artist
#[hdk_extern]
pub fn get_pending_settlements(artist: AgentPubKey) -> ExternResult<Vec<SettlementBatch>> {
    let artist_settlements_path = Path::from(format!("settlements/{}", artist));
    let typed_path = artist_settlements_path.typed(LinkTypes::ArtistToSettlements)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::ArtistToSettlements)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut pending = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(batch) = record
                    .entry()
                    .to_app_option::<SettlementBatch>()
                    .map_err(|e| wasm_error!(e))?
                {
                    if batch.status == SettlementStatus::Pending {
                        pending.push(batch);
                    }
                }
            }
        }
    }

    Ok(pending)
}

/// Get play statistics for a song (for artists)
#[derive(Serialize, Deserialize, Debug)]
pub struct SongStats {
    pub total_plays: u64,
    pub total_earnings: u64,
    pub unique_listeners: u64,
    pub avg_completion: f64,
}

#[hdk_extern]
pub fn get_song_stats(song_hash: ActionHash) -> ExternResult<SongStats> {
    let filter = LinkTypeFilter::try_from(LinkTypes::SongToPlays)?;
    let links = get_links(
        LinkQuery::new(song_hash, filter),
        GetStrategy::default(),
    )?;

    let mut total_plays: u64 = 0;
    let mut total_earnings: u64 = 0;
    let mut listeners: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_completion: f64 = 0.0;

    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(play) = record
                    .entry()
                    .to_app_option::<PlayRecord>()
                    .map_err(|e| wasm_error!(e))?
                {
                    total_plays += 1;
                    total_earnings += play.amount_owed;

                    // Get the author of this play record for unique listener count
                    listeners.insert(record.action().author().to_string());

                    if play.song_duration > 0 {
                        total_completion +=
                            play.duration_listened as f64 / play.song_duration as f64;
                    }
                }
            }
        }
    }

    let avg_completion = if total_plays > 0 {
        total_completion / total_plays as f64
    } else {
        0.0
    };

    Ok(SongStats {
        total_plays,
        total_earnings,
        unique_listeners: listeners.len() as u64,
        avg_completion,
    })
}

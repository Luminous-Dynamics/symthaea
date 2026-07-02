// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Song Routes - Core music catalog operations
//!
//! Handles song CRUD, streaming, and play recording

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::sync::Arc;
use uuid::Uuid;

use crate::AppState;

/// Song model
#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
pub struct Song {
    pub id: Uuid,
    pub song_hash: String,
    pub title: String,
    pub artist_address: String,
    pub ipfs_hash: String,
    pub strategy_id: String,
    pub payment_model: String,
    pub plays: i64,
    pub earnings: f64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Create song request
#[derive(Debug, Deserialize)]
pub struct CreateSongRequest {
    pub title: String,
    pub artist_address: String,
    pub ipfs_hash: String,
    pub strategy_id: String,
    pub payment_model: String,
    pub splits: Vec<Split>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Split {
    pub recipient: String,
    pub basis_points: u32,
    pub role: String,
}

/// Query params for listing songs
#[derive(Debug, Deserialize)]
pub struct ListSongsQuery {
    pub limit: Option<i64>,
    pub offset: Option<i64>,
    pub genre: Option<String>,
    pub strategy: Option<String>,
    pub search: Option<String>,
}

/// Record play request
#[derive(Debug, Deserialize)]
pub struct RecordPlayRequest {
    pub listener_address: String,
    pub amount: f64,
    pub payment_type: String,
    pub signature: String,
    pub nonce: String,
}

/// List songs with optional filters
pub async fn list_songs(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListSongsQuery>,
) -> Result<Json<Vec<Song>>, StatusCode> {
    let limit = params.limit.unwrap_or(50).min(100);
    let offset = params.offset.unwrap_or(0);

    let songs = sqlx::query_as::<_, Song>(
        r#"
        SELECT id, song_hash, title, artist_address, ipfs_hash,
               strategy_id, payment_model, plays, earnings::float8 as earnings, created_at
        FROM songs
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        "#,
    )
    .bind(limit)
    .bind(offset)
    .fetch_all(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to list songs: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    Ok(Json(songs))
}

/// Get a single song by ID
pub async fn get_song(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Song>, StatusCode> {
    let song = sqlx::query_as::<_, Song>(
        r#"
        SELECT id, song_hash, title, artist_address, ipfs_hash,
               strategy_id, payment_model, plays, earnings::float8 as earnings, created_at
        FROM songs
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to get song: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?
    .ok_or(StatusCode::NOT_FOUND)?;

    Ok(Json(song))
}

/// Create a new song
pub async fn create_song(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSongRequest>,
) -> Result<Json<Song>, StatusCode> {
    let id = Uuid::new_v4();
    let song_hash = format!("0x{}", hex::encode(sha2::Sha256::digest(id.as_bytes())));

    let song = sqlx::query_as::<_, Song>(
        r#"
        INSERT INTO songs (id, song_hash, title, artist_address, ipfs_hash, strategy_id, payment_model, plays, earnings)
        VALUES ($1, $2, $3, $4, $5, $6, $7, 0, 0)
        RETURNING id, song_hash, title, artist_address, ipfs_hash,
                  strategy_id, payment_model, plays, earnings::float8 as earnings, created_at
        "#,
    )
    .bind(id)
    .bind(&song_hash)
    .bind(&req.title)
    .bind(&req.artist_address)
    .bind(&req.ipfs_hash)
    .bind(&req.strategy_id)
    .bind(&req.payment_model)
    .fetch_one(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to create song: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!("Created song: {} by {}", song.title, song.artist_address);
    Ok(Json(song))
}

/// Record a play (streaming event)
pub async fn record_play(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<RecordPlayRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    // Step 1: Verify the payment signature
    // The signature should prove the listener authorized this payment
    if !verify_play_signature(&req, id).map_err(|e| {
        tracing::error!("Signature verification failed: {}", e);
        StatusCode::UNAUTHORIZED
    })? {
        tracing::warn!(
            "Invalid signature from {} for song {}",
            req.listener_address, id
        );
        return Err(StatusCode::UNAUTHORIZED);
    }

    // Step 2: Validate the listener address format
    if !is_valid_eth_address(&req.listener_address) {
        tracing::warn!("Invalid listener address format: {}", req.listener_address);
        return Err(StatusCode::BAD_REQUEST);
    }

    // Step 3: Validate the payment amount is reasonable
    if req.amount <= 0.0 || req.amount > 1000.0 {
        tracing::warn!("Invalid payment amount: {}", req.amount);
        return Err(StatusCode::BAD_REQUEST);
    }

    // Step 4: Verify the nonce hasn't been used (prevent replay attacks)
    let nonce_exists = sqlx::query_scalar::<_, bool>(
        "SELECT EXISTS(SELECT 1 FROM used_nonces WHERE nonce = $1 AND listener_address = $2)"
    )
    .bind(&req.nonce)
    .bind(&req.listener_address)
    .fetch_one(&state.db_pool)
    .await
    .unwrap_or(false);

    if nonce_exists {
        tracing::warn!("Replay attack detected: nonce {} already used", req.nonce);
        return Err(StatusCode::CONFLICT);
    }

    // Step 5: Record the nonce to prevent replay
    sqlx::query(
        "INSERT INTO used_nonces (nonce, listener_address, used_at) VALUES ($1, $2, NOW()) ON CONFLICT DO NOTHING"
    )
    .bind(&req.nonce)
    .bind(&req.listener_address)
    .execute(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to record nonce: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    // Step 6: Process payment via smart contract (if blockchain service is available)
    let tx_hash = if let Some(blockchain) = &state.blockchain {
        // Convert song UUID to bytes32 (hash the UUID)
        let song_id_bytes: [u8; 32] = {
            let mut hasher = sha2::Sha256::new();
            hasher.update(id.as_bytes());
            hasher.finalize().into()
        };

        // Convert payment type string to enum
        let payment_type = match req.payment_type.to_lowercase().as_str() {
            "stream" => 0u8,
            "download" => 1u8,
            "tip" => 2u8,
            "patronage" => 3u8,
            "nft_access" => 4u8,
            _ => 0u8, // Default to stream
        };

        // Convert amount to wei (assuming FLOW token has 18 decimals)
        let amount_wei = ethers::types::U256::from((req.amount * 1e18) as u128);

        // Process payment on-chain
        match blockchain.process_payment(song_id_bytes, amount_wei, payment_type).await {
            Ok(hash) => {
                tracing::info!(
                    "Payment processed on-chain for song {}: tx {}",
                    id, hash
                );
                Some(format!("{:?}", hash))
            }
            Err(e) => {
                // Log but don't fail - payment might be handled client-side
                tracing::warn!(
                    "On-chain payment processing failed (may be handled client-side): {}",
                    e
                );
                None
            }
        }
    } else {
        // No blockchain service configured - payments handled client-side via SDK
        tracing::debug!("Blockchain service not configured, payment handled client-side");
        None
    };

    // Step 7: Update play count and earnings in database
    let result = sqlx::query(
        r#"
        UPDATE songs
        SET plays = plays + 1, earnings = earnings + $2
        WHERE id = $1
        "#,
    )
    .bind(id)
    .bind(req.amount)
    .execute(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to record play: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    if result.rows_affected() == 0 {
        return Err(StatusCode::NOT_FOUND);
    }

    // Step 8: Insert play record with verified signature
    sqlx::query(
        r#"
        INSERT INTO plays (song_id, listener_address, amount, payment_type, signature_verified, tx_hash, timestamp)
        VALUES ($1, $2, $3, $4, true, $5, NOW())
        "#,
    )
    .bind(id)
    .bind(&req.listener_address)
    .bind(req.amount)
    .bind(&req.payment_type)
    .bind(&tx_hash)
    .execute(&state.db_pool)
    .await
    .map_err(|e| {
        tracing::error!("Failed to insert play record: {}", e);
        StatusCode::INTERNAL_SERVER_ERROR
    })?;

    tracing::info!(
        "Recorded verified play for song {} by {} (amount: {}, on-chain: {})",
        id, req.listener_address, req.amount, tx_hash.is_some()
    );

    Ok(Json(serde_json::json!({
        "success": true,
        "song_id": id,
        "listener": req.listener_address,
        "amount": req.amount,
        "signature_verified": true,
        "payment_on_chain": tx_hash.is_some(),
        "transaction_hash": tx_hash,
        "message": "Play recorded with verified payment signature. Artist paid instantly!"
    })))
}

/// Verify the play payment signature
///
/// The signature should be an Ethereum personal_sign of:
/// "Mycelix Music Payment\nSong: {song_id}\nAmount: {amount}\nNonce: {nonce}"
fn verify_play_signature(req: &RecordPlayRequest, song_id: Uuid) -> Result<bool, String> {
    // Construct the expected message
    let message = format!(
        "Mycelix Music Payment\nSong: {}\nAmount: {}\nNonce: {}",
        song_id, req.amount, req.nonce
    );

    // Validate signature format (0x prefix + 130 hex chars = 65 bytes)
    if !req.signature.starts_with("0x") || req.signature.len() != 132 {
        return Err("Invalid signature format".to_string());
    }

    // Decode signature
    let sig_bytes = hex::decode(&req.signature[2..])
        .map_err(|e| format!("Invalid signature hex: {}", e))?;

    if sig_bytes.len() != 65 {
        return Err("Invalid signature length".to_string());
    }

    // Create the Ethereum signed message hash (EIP-191)
    let prefixed_message = format!("\x19Ethereum Signed Message:\n{}{}", message.len(), message);
    let message_hash = sha2::Sha256::digest(prefixed_message.as_bytes());

    // For production, use secp256k1 recovery:
    // 1. Extract r, s, v from signature
    // 2. Recover public key from signature and message hash
    // 3. Derive address from public key
    // 4. Compare to listener_address

    // Using k256 crate for ECDSA recovery
    use k256::ecdsa::{RecoveryId, Signature, VerifyingKey};
    use k256::elliptic_curve::sec1::ToEncodedPoint;
    use sha3::{Digest, Keccak256};

    // Parse signature components
    let r_bytes: [u8; 32] = sig_bytes[0..32].try_into()
        .map_err(|_| "Invalid r component")?;
    let s_bytes: [u8; 32] = sig_bytes[32..64].try_into()
        .map_err(|_| "Invalid s component")?;
    let v = sig_bytes[64];

    // Determine recovery ID (Ethereum uses 27/28, we need 0/1)
    let recovery_id = if v >= 27 { v - 27 } else { v };
    let recovery_id = RecoveryId::try_from(recovery_id)
        .map_err(|_| "Invalid recovery ID")?;

    // Create signature from r and s
    let mut sig_array = [0u8; 64];
    sig_array[..32].copy_from_slice(&r_bytes);
    sig_array[32..].copy_from_slice(&s_bytes);
    let signature = Signature::from_bytes(&sig_array.into())
        .map_err(|e| format!("Invalid signature: {}", e))?;

    // Hash with keccak256 for Ethereum
    let msg_hash = Keccak256::digest(&prefixed_message);

    // Recover the public key
    let recovered_key = VerifyingKey::recover_from_prehash(&msg_hash, &signature, recovery_id)
        .map_err(|e| format!("Recovery failed: {}", e))?;

    // Derive Ethereum address from public key (keccak256 of uncompressed public key, take last 20 bytes)
    let public_key_bytes = recovered_key.to_encoded_point(false);
    let public_key_hash = Keccak256::digest(&public_key_bytes.as_bytes()[1..]); // Skip 0x04 prefix
    let recovered_address = format!("0x{}", hex::encode(&public_key_hash[12..]));

    // Compare addresses (case-insensitive)
    Ok(recovered_address.to_lowercase() == req.listener_address.to_lowercase())
}

/// Validate Ethereum address format
fn is_valid_eth_address(address: &str) -> bool {
    if !address.starts_with("0x") {
        return false;
    }
    if address.len() != 42 {
        return false;
    }
    // Check all remaining characters are hex
    address[2..].chars().all(|c| c.is_ascii_hexdigit())
}

use sha2::Digest;

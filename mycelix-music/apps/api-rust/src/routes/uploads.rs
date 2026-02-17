//! Upload Routes - IPFS file uploads
//!
//! Handles music file uploads to IPFS with Web3.Storage

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::sync::Arc;

use crate::AppState;

#[derive(Debug, Serialize)]
pub struct UploadResponse {
    pub success: bool,
    pub ipfs_hash: String,
    pub size: u64,
    pub content_type: String,
    pub gateway_url: String,
}

/// Maximum file size (100MB)
const MAX_FILE_SIZE: u64 = 100 * 1024 * 1024;

/// Allowed MIME types
const ALLOWED_TYPES: &[&str] = &[
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/aac",
];

/// Upload file to IPFS
pub async fn upload_file(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<UploadResponse>, StatusCode> {
    while let Some(field) = multipart.next_field().await.map_err(|e| {
        tracing::error!("Failed to read multipart field: {}", e);
        StatusCode::BAD_REQUEST
    })? {
        let name = field.name().unwrap_or("").to_string();
        if name != "file" {
            continue;
        }

        let content_type = field
            .content_type()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "application/octet-stream".to_string());

        // Validate content type
        if !ALLOWED_TYPES.contains(&content_type.as_str()) {
            tracing::warn!("Rejected upload with content type: {}", content_type);
            return Err(StatusCode::UNSUPPORTED_MEDIA_TYPE);
        }

        // Read file data
        let data = field.bytes().await.map_err(|e| {
            tracing::error!("Failed to read file data: {}", e);
            StatusCode::BAD_REQUEST
        })?;

        // Check file size
        if data.len() as u64 > MAX_FILE_SIZE {
            tracing::warn!("Rejected upload: file too large ({} bytes)", data.len());
            return Err(StatusCode::PAYLOAD_TOO_LARGE);
        }

        // Upload to IPFS
        let cursor = std::io::Cursor::new(data.to_vec());
        let response = state
            .ipfs_client
            .add(cursor)
            .await
            .map_err(|e| {
                tracing::error!("Failed to upload to IPFS: {}", e);
                StatusCode::INTERNAL_SERVER_ERROR
            })?;

        let ipfs_hash = response.hash;
        let size = data.len() as u64;

        tracing::info!(
            "Uploaded file to IPFS: {} ({} bytes, {})",
            ipfs_hash,
            size,
            content_type
        );

        return Ok(Json(UploadResponse {
            success: true,
            ipfs_hash: ipfs_hash.clone(),
            size,
            content_type,
            gateway_url: format!("https://w3s.link/ipfs/{}", ipfs_hash),
        }));
    }

    Err(StatusCode::BAD_REQUEST)
}

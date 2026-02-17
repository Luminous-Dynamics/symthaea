//! Data Models
//!
//! Shared data structures for the Mycelix Music platform

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Song entity
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
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

/// Play record
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Play {
    pub id: Uuid,
    pub song_id: Uuid,
    pub listener_address: String,
    pub amount: f64,
    pub payment_type: String,
    pub tx_hash: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Artist profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artist {
    pub address: String,
    pub total_songs: i64,
    pub total_plays: i64,
    pub total_earnings: f64,
    pub strategies_used: Vec<String>,
    pub joined_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Economic strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub strategy_id: String,
    pub payment_model: PaymentModel,
    pub splits: Vec<Split>,
    pub min_payment: f64,
    pub protocol_fee_bps: u32,
}

/// Payment model types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentModel {
    PayPerStream,
    GiftEconomy,
    Subscription,
    Patronage,
    NftGated,
    PayWhatYouWant,
    Auction,
    Freemium,
    TimeBarter,
    Download,
    StakingGated,
}

/// Revenue split configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub recipient: String,
    pub basis_points: u32,
    pub role: String,
}

/// API error response
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub message: String,
    pub status_code: u16,
}

impl ApiError {
    pub fn new(error: &str, message: &str, status_code: u16) -> Self {
        Self {
            error: error.to_string(),
            message: message.to_string(),
            status_code,
        }
    }

    pub fn not_found(resource: &str) -> Self {
        Self::new("not_found", &format!("{} not found", resource), 404)
    }

    pub fn bad_request(message: &str) -> Self {
        Self::new("bad_request", message, 400)
    }

    pub fn internal_error() -> Self {
        Self::new("internal_error", "An unexpected error occurred", 500)
    }
}

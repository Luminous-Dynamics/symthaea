//! Economic Strategies Routes
//!
//! Manage and preview economic strategies for artists

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

/// Available economic strategy
#[derive(Debug, Serialize)]
pub struct EconomicStrategy {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub min_payment: f64,
    pub default_protocol_fee_bps: u32,
    pub supports_free_listening: bool,
    pub supports_tips: bool,
    pub supports_subscriptions: bool,
}

/// Split preview request
#[derive(Debug, Deserialize)]
pub struct PreviewSplitsRequest {
    pub amount: f64,
    pub splits: Vec<SplitConfig>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SplitConfig {
    pub recipient: String,
    pub basis_points: u32,
    pub role: String,
}

/// Split preview response
#[derive(Debug, Serialize)]
pub struct PreviewSplitsResponse {
    pub gross_amount: f64,
    pub protocol_fee: f64,
    pub net_amount: f64,
    pub distributions: Vec<Distribution>,
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub recipient: String,
    pub role: String,
    pub amount: f64,
    pub percentage: f64,
}

/// List all available strategies
pub async fn list_strategies(
    State(_state): State<Arc<AppState>>,
) -> Json<Vec<EconomicStrategy>> {
    // These mirror the Solidity contracts
    let strategies = vec![
        EconomicStrategy {
            id: "pay-per-stream-v1".into(),
            name: "Pay Per Stream".into(),
            description: "Listeners pay $0.01 per stream. Instant royalty distribution.".into(),
            category: "direct-payment".into(),
            min_payment: 0.01,
            default_protocol_fee_bps: 100, // 1%
            supports_free_listening: false,
            supports_tips: true,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "gift-economy-v1".into(),
            name: "Gift Economy".into(),
            description: "Free listening with CGC rewards. Optional tips to artist.".into(),
            category: "community".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 100,
            supports_free_listening: true,
            supports_tips: true,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "subscription-v1".into(),
            name: "Subscription".into(),
            description: "Monthly fee for unlimited listening.".into(),
            category: "recurring".into(),
            min_payment: 5.0,
            default_protocol_fee_bps: 200, // 2%
            supports_free_listening: false,
            supports_tips: true,
            supports_subscriptions: true,
        },
        EconomicStrategy {
            id: "patronage-v1".into(),
            name: "Patronage".into(),
            description: "Recurring support from dedicated fans.".into(),
            category: "recurring".into(),
            min_payment: 1.0,
            default_protocol_fee_bps: 100,
            supports_free_listening: true,
            supports_tips: false,
            supports_subscriptions: true,
        },
        EconomicStrategy {
            id: "nft-gated-v1".into(),
            name: "NFT Gated".into(),
            description: "Exclusive content for NFT holders.".into(),
            category: "token-gated".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 250, // 2.5%
            supports_free_listening: true,
            supports_tips: true,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "pay-what-you-want-v1".into(),
            name: "Pay What You Want".into(),
            description: "Listener chooses amount. No minimum.".into(),
            category: "flexible".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 100,
            supports_free_listening: true,
            supports_tips: true,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "auction-v1".into(),
            name: "Auction".into(),
            description: "Time-limited bidding for exclusive releases.".into(),
            category: "auction".into(),
            min_payment: 1.0,
            default_protocol_fee_bps: 500, // 5%
            supports_free_listening: false,
            supports_tips: false,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "freemium-v1".into(),
            name: "Freemium".into(),
            description: "Free tier with premium features.".into(),
            category: "tiered".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 150,
            supports_free_listening: true,
            supports_tips: true,
            supports_subscriptions: true,
        },
        EconomicStrategy {
            id: "time-barter-v1".into(),
            name: "Time Barter (TEND)".into(),
            description: "Exchange TEND tokens for access. No fiat required.".into(),
            category: "alternative-currency".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 0, // No fee for TEND
            supports_free_listening: false,
            supports_tips: false,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "download-v1".into(),
            name: "Pay Per Download".into(),
            description: "One-time payment to own the file.".into(),
            category: "direct-payment".into(),
            min_payment: 0.99,
            default_protocol_fee_bps: 100,
            supports_free_listening: false,
            supports_tips: false,
            supports_subscriptions: false,
        },
        EconomicStrategy {
            id: "staking-gated-v1".into(),
            name: "Staking Gated".into(),
            description: "Stake tokens to access content.".into(),
            category: "token-gated".into(),
            min_payment: 0.0,
            default_protocol_fee_bps: 50,
            supports_free_listening: true,
            supports_tips: true,
            supports_subscriptions: false,
        },
    ];

    Json(strategies)
}

/// Preview how splits would work for a given amount
pub async fn preview_splits(
    State(_state): State<Arc<AppState>>,
    Path(strategy_id): Path<String>,
    Json(req): Json<PreviewSplitsRequest>,
) -> Result<Json<PreviewSplitsResponse>, StatusCode> {
    // Get protocol fee for strategy (simplified)
    let protocol_fee_bps: u32 = match strategy_id.as_str() {
        "pay-per-stream-v1" => 100,
        "gift-economy-v1" => 100,
        "subscription-v1" => 200,
        "nft-gated-v1" => 250,
        "auction-v1" => 500,
        _ => 100,
    };

    let gross_amount = req.amount;
    let protocol_fee = gross_amount * (protocol_fee_bps as f64 / 10000.0);
    let net_amount = gross_amount - protocol_fee;

    // Calculate distributions
    let total_bps: u32 = req.splits.iter().map(|s| s.basis_points).sum();
    if total_bps != 10000 {
        return Err(StatusCode::BAD_REQUEST);
    }

    let distributions: Vec<Distribution> = req
        .splits
        .iter()
        .map(|split| {
            let amount = net_amount * (split.basis_points as f64 / 10000.0);
            Distribution {
                recipient: split.recipient.clone(),
                role: split.role.clone(),
                amount,
                percentage: split.basis_points as f64 / 100.0,
            }
        })
        .collect();

    Ok(Json(PreviewSplitsResponse {
        gross_amount,
        protocol_fee,
        net_amount,
        distributions,
    }))
}

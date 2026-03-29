// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! API routes

use std::sync::Arc;

use axum::Router;
use tokio::sync::broadcast;

use crate::config::Config;
use crate::services::{BridgeClient, HolochainService, StorageService, TrustCacheService};
use mycelix_identity_client::{IdentityClient, IdentityClientConfig, FallbackMode};

pub mod ai;
pub mod auth;
pub mod bridge;
pub mod claims;
pub mod did;
pub mod emails;
pub mod trust;
pub mod trust_graph;
pub mod ws;

pub use ws::{EventBroadcast, WsEvent, notify_new_mail, notify_trust_update};

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub config: Config,
    pub holochain: Arc<HolochainService>,
    pub trust_cache: Arc<TrustCacheService>,
    pub storage: Arc<StorageService>,
    /// Bridge client for cross-hApp communication
    pub bridge: Arc<BridgeClient>,
    /// Identity client for DID resolution and credential verification
    pub identity: Arc<IdentityClient>,
    /// Broadcast channel for real-time WebSocket events
    pub event_broadcast: EventBroadcast,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        let holochain = Arc::new(HolochainService::new(config.clone()));
        let storage = Arc::new(StorageService::new(&config));
        let bridge = Arc::new(BridgeClient::new(config.clone()));

        // Create trust cache with Bridge integration for cross-hApp reputation
        let trust_cache = Arc::new(TrustCacheService::new_with_bridge(
            &config,
            holochain.clone(),
            bridge.clone(),
        ));

        // Create identity client with Warn fallback mode (don't fail if unavailable)
        let identity_config = IdentityClientConfig {
            conductor_url: config.conductor_url().to_string(),
            enable_cache: true,
            cache_ttl_secs: 300,
            fallback_mode: FallbackMode::Warn,
            identity_role_name: "identity".to_string(),
        };
        let identity = Arc::new(IdentityClient::new(identity_config));

        // Create broadcast channel with 100 event buffer
        let (event_broadcast, _) = broadcast::channel(100);

        Self {
            config,
            holochain,
            trust_cache,
            storage,
            bridge,
            identity,
            event_broadcast,
        }
    }

    /// Initialize async components after creation
    ///
    /// This wires up the HolochainService to the BridgeClient so that
    /// Bridge zome calls can be made through the existing Holochain connection.
    pub async fn initialize(&self) {
        // Set the HolochainService reference on the BridgeClient
        // This enables real Bridge zome calls instead of stub data
        self.bridge.set_holochain(self.holochain.clone()).await;
        tracing::debug!("AppState: Wired HolochainService to BridgeClient for Bridge zome calls");
    }
}

/// Create the API router with all routes
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .nest("/api/auth", auth::router())
        .nest("/api/emails", emails::router())
        .nest("/api/trust", trust::router())
        .nest("/api/did", did::router())
        .nest("/api/claims", claims::router())
        .nest("/api/ai", ai::router())
        .nest("/api/trust-graph", trust_graph::router())
        .nest("/api/bridge", bridge::router())
        .nest("/ws", ws::router())
        .with_state(state)
}

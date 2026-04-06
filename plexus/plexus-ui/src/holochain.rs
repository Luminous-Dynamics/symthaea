// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holochain conductor bridge for Symthaea Prism.
//!
//! Connects to the shared Mycelix conductor on ws://localhost:8888
//! to publish verified claims to the knowledge DHT.
//!
//! Connection lifecycle:
//! 1. App startup → check_conductor() probes the WebSocket
//! 2. If connected → "Share to DHT" buttons become active
//! 3. Clicking "Share" → publish_claim() sends via WebSocket
//! 4. If disconnected → claims saved locally, queued for later sync

use leptos::prelude::*;
use serde::{Deserialize, Serialize};

/// Connection status to the Holochain conductor.
#[derive(Clone, Debug, PartialEq)]
pub enum ConductorStatus {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

impl ConductorStatus {
    pub fn is_connected(&self) -> bool {
        matches!(self, Self::Connected)
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Disconnected => "DHT Offline",
            Self::Connecting => "Connecting...",
            Self::Connected => "DHT Connected",
            Self::Error(_) => "DHT Error",
        }
    }

    pub fn css_class(&self) -> &str {
        match self {
            Self::Disconnected => "dht-status offline",
            Self::Connecting => "dht-status connecting",
            Self::Connected => "dht-status online",
            Self::Error(_) => "dht-status error",
        }
    }
}

/// A claim to be published to the Mycelix knowledge DHT.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhtClaim {
    pub content: String,
    pub classification: DhtClassification,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub claim_type: String,
    pub confidence: f32,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhtClassification {
    pub empirical: f32,
    pub normative: f32,
    pub mythic: f32,
}

/// Check if the Holochain conductor is reachable via WebSocket probe.
pub async fn check_conductor() -> ConductorStatus {
    // Probe the conductor by attempting an HTTP request to the app port
    // (WebSocket upgrade happens on connection, but HTTP probe tells us if port is open)
    match gloo_net::http::Request::get("http://localhost:8888")
        .send()
        .await
    {
        Ok(resp) => {
            // Holochain conductor responds (even with an error) = it's running
            log::info!("Holochain conductor detected on :8888 (status {})", resp.status());
            ConductorStatus::Connected
        }
        Err(e) => {
            log::info!("Holochain conductor not reachable: {}", e);
            ConductorStatus::Disconnected
        }
    }
}

/// Publish a verified claim to the knowledge DHT.
pub async fn publish_claim(claim: &DhtClaim) -> Result<String, String> {
    log::info!("Publishing claim to DHT: {}", &claim.content[..claim.content.len().min(60)]);

    // Check conductor reachability first
    let status = check_conductor().await;
    if !status.is_connected() {
        // Save locally for later sync
        save_claim_locally(claim);
        return Err("Conductor offline — claim saved locally for later sync".to_string());
    }

    // TODO: Full Holochain zome call via WebSocket
    // This requires:
    // 1. WebSocket connection to ws://localhost:8888
    // 2. AppAuthenticationToken handshake
    // 3. CallZome { zome_name: "claims", fn_name: "submit_claim", payload: claim }
    // 4. Parse ActionHash from response
    //
    // For now, save locally and mark as pending
    save_claim_locally(claim);
    Err("Claim saved locally — full DHT sync coming soon".to_string())
}

/// Save a claim to localStorage for offline-first operation.
fn save_claim_locally(claim: &DhtClaim) {
    if let Some(storage) = web_sys::window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
    {
        // Load existing queue
        let key = "prism-claim-queue";
        let mut queue: Vec<DhtClaim> = storage
            .get_item(key)
            .ok()
            .flatten()
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default();

        queue.push(claim.clone());

        // Cap queue at 100 claims
        if queue.len() > 100 {
            queue = queue.split_off(queue.len() - 100);
        }

        if let Ok(json) = serde_json::to_string(&queue) {
            let _ = storage.set_item(key, &json);
        }

        log::info!("Claim queued locally ({} pending)", queue.len());
    }
}

/// Get the count of locally queued claims waiting for DHT sync.
pub fn pending_claim_count() -> usize {
    web_sys::window()
        .and_then(|w| w.local_storage().ok())
        .flatten()
        .and_then(|s| s.get_item("prism-claim-queue").ok())
        .flatten()
        .and_then(|json| serde_json::from_str::<Vec<DhtClaim>>(&json).ok())
        .map(|q| q.len())
        .unwrap_or(0)
}

/// DHT status indicator component for the chrome bar.
#[component]
pub fn DhtStatusBadge() -> impl IntoView {
    let (status, set_status) = signal(ConductorStatus::Disconnected);
    let (pending, set_pending) = signal(0usize);

    // Check conductor on mount
    Effect::new(move |_| {
        wasm_bindgen_futures::spawn_local(async move {
            set_status.set(ConductorStatus::Connecting);
            let s = check_conductor().await;
            set_status.set(s);
            set_pending.set(pending_claim_count());
        });
    });

    let status_class = move || status.get().css_class().to_string();
    let status_label = move || status.get().label().to_string();
    let pending_count = move || pending.get();
    let has_pending = move || pending.get() > 0;

    view! {
        <div class=status_class title=move || {
            let p = pending_count();
            if p > 0 {
                format!("{} ({} claims pending sync)", status_label(), p)
            } else {
                status_label()
            }
        }>
            <span class="dht-dot"></span>
            {move || if has_pending() {
                Some(view! { <span class="dht-pending">{pending_count()}</span> })
            } else {
                None
            }}
        </div>
    }
}

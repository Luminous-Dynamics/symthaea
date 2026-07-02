// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holochain conductor bridge for Prism.
//!
//! Connects to the shared Mycelix conductor on ws://localhost:8888
//! to publish verified claims to the Knowledge DHT.
//!
//! Bridge lifecycle:
//! 1. App startup → check_conductor() probes the WebSocket
//! 2. If connected → "Share to DHT" buttons become active
//! 3. Clicking "Share" → publish_claim() sends via BrowserWsTransport
//! 4. If disconnected → claims saved locally, queued for later sync

use leptos::prelude::*;
use mycelix_claim_types::{EmpiricalLevel, MaterialityLevel, NormativeLevel};
use serde::{Deserialize, Serialize};

const CONDUCTOR_URL: &str = "ws://localhost:8888";
const APP_ID: &str = "mycelix-knowledge";

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

/// A claim to be published to the Mycelix Knowledge DHT.
/// Uses unified E/N/M types from mycelix-claim-types.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DhtClaim {
    pub content: String,
    pub empirical: EmpiricalLevel,
    pub normative: NormativeLevel,
    pub materiality: MaterialityLevel,
    pub sources: Vec<String>,
    pub tags: Vec<String>,
    pub claim_type: String,
    pub confidence: f64,
}

/// Check if the Holochain conductor is reachable.
pub async fn check_conductor() -> ConductorStatus {
    match gloo_net::http::Request::get("http://localhost:8888")
        .send()
        .await
    {
        Ok(resp) => {
            log::info!(
                "Holochain conductor detected on :8888 (status {})",
                resp.status()
            );
            ConductorStatus::Connected
        }
        Err(e) => {
            log::info!("Holochain conductor not reachable: {}", e);
            ConductorStatus::Disconnected
        }
    }
}

/// Publish a verified claim to the Knowledge DHT via WebSocket zome call.
pub async fn publish_claim(claim: &DhtClaim) -> Result<String, String> {
    log::info!(
        "Publishing claim to DHT: {}",
        &claim.content[..claim.content.len().min(60)]
    );

    let status = check_conductor().await;
    if !status.is_connected() {
        save_claim_locally(claim);
        return Err("Conductor offline — claim saved locally for later sync".to_string());
    }

    // Build zome call payload matching Knowledge's submit_claim input
    #[derive(Serialize)]
    struct SubmitClaimInput {
        id: String,
        content: String,
        classification: SubmitClassification,
        author: String,
        sources: Vec<String>,
        tags: Vec<String>,
        claim_type: String,
        confidence: f64,
    }

    #[derive(Serialize)]
    struct SubmitClassification {
        empirical: f64,
        normative: f64,
        mythic: f64,
    }

    let input = SubmitClaimInput {
        id: format!("prism-{}", js_sys::Date::now() as u64),
        content: claim.content.clone(),
        classification: SubmitClassification {
            empirical: claim.empirical.as_f32() as f64,
            normative: claim.normative.as_f32() as f64,
            mythic: claim.materiality.as_f32() as f64,
        },
        author: "did:mycelix:prism-browser".to_string(),
        sources: claim.sources.clone(),
        tags: claim.tags.clone(),
        claim_type: claim.claim_type.clone(),
        confidence: claim.confidence,
    };

    // Connect via BrowserWsTransport and call the zome
    let transport = mycelix_leptos_client::BrowserWsTransport::new();
    let client = mycelix_leptos_client::HolochainClient::new(transport, APP_ID, "knowledge");

    match client.connect(CONDUCTOR_URL, None).await {
        Ok(()) => {
            match client
                .call_zome::<SubmitClaimInput, serde_json::Value>("claims", "submit_claim", &input)
                .await
            {
                Ok(response) => {
                    log::info!("Claim published to DHT successfully");
                    Ok(format!("Published: {:?}", response))
                }
                Err(e) => {
                    log::warn!("DHT zome call failed: {e} — saving locally");
                    save_claim_locally(claim);
                    Err(format!("Zome call failed: {e} — saved locally"))
                }
            }
        }
        Err(e) => {
            log::warn!("WebSocket connection failed: {e} — saving locally");
            save_claim_locally(claim);
            Err(format!("Connection failed: {e} — saved locally"))
        }
    }
}

/// Sync all locally queued claims to the DHT.
pub async fn sync_pending_claims() -> usize {
    let claims = load_local_queue();
    if claims.is_empty() {
        return 0;
    }

    let status = check_conductor().await;
    if !status.is_connected() {
        return 0;
    }

    let mut synced = 0;
    for claim in &claims {
        if publish_claim(claim).await.is_ok() {
            synced += 1;
        }
    }

    if synced > 0 {
        let remaining: Vec<DhtClaim> = claims.into_iter().skip(synced).collect();
        save_queue(&remaining);
    }
    synced
}

// --- localStorage queue ---

fn save_claim_locally(claim: &DhtClaim) {
    let mut queue = load_local_queue();
    queue.push(claim.clone());
    if queue.len() > 100 {
        queue = queue.split_off(queue.len() - 100);
    }
    save_queue(&queue);
    log::info!("Claim queued locally ({} pending)", queue.len());
}

fn load_local_queue() -> Vec<DhtClaim> {
    web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("prism-claim-queue").ok().flatten())
        .and_then(|json| serde_json::from_str(&json).ok())
        .unwrap_or_default()
}

fn save_queue(queue: &[DhtClaim]) {
    if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
        if let Ok(json) = serde_json::to_string(queue) {
            let _ = storage.set_item("prism-claim-queue", &json);
        }
    }
}

pub fn pending_claim_count() -> usize {
    load_local_queue().len()
}

/// DHT status indicator component for the chrome bar.
#[component]
pub fn DhtStatusBadge() -> impl IntoView {
    let (status, set_status) = signal(ConductorStatus::Disconnected);
    let (pending, set_pending) = signal(0usize);

    Effect::new(move |_| {
        wasm_bindgen_futures::spawn_local(async move {
            set_status.set(ConductorStatus::Connecting);
            let s = check_conductor().await;
            set_status.set(s.clone());
            set_pending.set(pending_claim_count());

            // Auto-sync pending claims if connected
            if s.is_connected() {
                let synced = sync_pending_claims().await;
                if synced > 0 {
                    log::info!("Auto-synced {synced} pending claims to DHT");
                    set_pending.set(pending_claim_count());
                }
            }
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

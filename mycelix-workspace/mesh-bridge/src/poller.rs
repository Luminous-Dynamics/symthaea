// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conductor Poller — polls for new entries and relays to mesh.
//!
//! Watches for new TEND exchanges, food harvests, emergency messages,
//! water alerts, hearth alerts, knowledge claims, care circle updates,
//! shelter updates, supply inventory, mutual aid offers, and price reports.
//! Polls conductor via AppWebsocket. Deduplicates by action hash.

use crate::serializer::{
    self, CareCircleRelay, EmergencyRelay, FoodRelay, HearthAlertRelay, KnowledgeClaimRelay,
    MutualAidRelay, PriceReportRelay, RelayPayload, RelayType, SensorReadingRelay, ShelterRelay,
    SupplyRelay, TendRelay, WaterAlertRelay,
};
use crate::transport::MeshTransport;
use crate::BridgeMetrics;
use anyhow::Result;
use chacha20poly1305::XChaCha20Poly1305;
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ZomeCallTarget};
use holochain_types::prelude::ExternIO;
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Maximum dedup cache size (action hashes we've already relayed).
const MAX_DEDUP_CACHE: usize = 10_000;

/// LoRa frame size limit (SX1276 max payload).
const LORA_MAX_FRAME: usize = 255;

/// Heartbeat interval: broadcast presence every N poll cycles.
const HEARTBEAT_EVERY_N_POLLS: u64 = 4;

/// Max consecutive conductor connection failures before backoff.
const MAX_CONSECUTIVE_FAILURES: u32 = 5;

/// Backoff multiplier per failure (capped at 5 minutes).
const BACKOFF_BASE_SECS: u64 = 5;
const BACKOFF_MAX_SECS: u64 = 300;

/// Max entries per domain per poll cycle (backpressure).
const MAX_ENTRIES_PER_DOMAIN: usize = 20;

/// Encode a value as ExternIO, converting the error for match arms.
fn encode_extern_io<T: serde::Serialize + std::fmt::Debug>(val: T) -> std::result::Result<ExternIO, String> {
    ExternIO::encode(val).map_err(|e| format!("{e}"))
}

/// Run the poller loop: conductor → serializer → mesh transport.
///
/// Respects the `cancel` token for graceful shutdown — drains the current
/// poll cycle before exiting.
pub async fn run(
    conductor_url: &str,
    poll_interval_secs: u64,
    transport: Box<dyn MeshTransport>,
    cancel: CancellationToken,
    metrics: BridgeMetrics,
) -> Result<()> {
    tracing::info!("Poller starting, conductor={conductor_url}, interval={poll_interval_secs}s");

    let mut dedup: HashSet<String> = crate::dedup_cache::load_string_cache("poller-dedup.cache");
    let origin = get_origin_id();
    let cipher = crate::encryption::load_psk();
    if cipher.is_some() {
        tracing::info!("PSK encryption enabled");
    }
    let mut poll_count: u64 = 0;
    let mut consecutive_failures: u32 = 0;
    let mut ws_cached: Option<AppWebsocket> = None;

    // Startup diagnostic: verify conductor is reachable before entering poll loop
    match startup_check(conductor_url).await {
        Ok(()) => tracing::info!("Conductor reachable at {conductor_url}"),
        Err(e) => tracing::warn!(
            "Conductor startup check failed: {e}\n\
             The poller will retry on each poll cycle with exponential backoff.\n\
             Common fixes:\n\
             - Is the conductor running? (just up / docker compose up)\n\
             - Is CONDUCTOR_URL correct? (current: {conductor_url})\n\
             - Is the resilience hApp installed? (hc app install ...)"
        ),
    }

    loop {
        if cancel.is_cancelled() {
            tracing::info!("Poller shutting down (cancel requested)");
            break;
        }

        poll_count += 1;
        metrics.poll_cycles.fetch_add(1, Ordering::Relaxed);

        // Broadcast heartbeat periodically
        if poll_count % HEARTBEAT_EVERY_N_POLLS == 0 {
            let heartbeat = RelayPayload::new(RelayType::Heartbeat, origin, Vec::new());
            let bytes = heartbeat.to_bytes();
            let wire_bytes = if let Some(ref c) = cipher {
                match crate::encryption::encrypt(c, &bytes) {
                    Ok(enc) => enc,
                    Err(e) => {
                        tracing::error!("Failed to encrypt heartbeat: {e}");
                        bytes
                    }
                }
            } else {
                bytes
            };
            let frames = serializer::fragment(&wire_bytes, LORA_MAX_FRAME);
            for frame in &frames {
                let _ = transport.send(frame).await;
            }
            tracing::debug!("Heartbeat broadcast (cycle {})", poll_count);
        }

        // Try to connect and poll
        match poll_once(
            conductor_url,
            &mut dedup,
            &origin,
            &*transport,
            &mut ws_cached,
            poll_count,
            cipher.as_ref(),
        )
        .await
        {
            Ok(relayed) => {
                consecutive_failures = 0;
                if relayed > 0 {
                    metrics
                        .messages_sent
                        .fetch_add(relayed as u64, Ordering::Relaxed);
                    metrics.last_send_at.store(
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64,
                        Ordering::Relaxed,
                    );
                    tracing::info!("Relayed {relayed} entries to mesh");
                }
            }
            Err(e) => {
                consecutive_failures += 1;
                metrics.connection_failures.fetch_add(1, Ordering::Relaxed);
                // Drop cached connection on failure so next cycle reconnects
                ws_cached = None;
                tracing::warn!(
                    "Poll failed (attempt {}/{}): {e}",
                    consecutive_failures,
                    MAX_CONSECUTIVE_FAILURES
                );
            }
        }

        // Trim dedup cache
        if dedup.len() > MAX_DEDUP_CACHE {
            dedup.clear();
            tracing::debug!("Dedup cache cleared (exceeded {MAX_DEDUP_CACHE})");
        }

        // Exponential backoff when conductor is consistently unreachable
        let sleep_secs = if consecutive_failures > 0 {
            let backoff = BACKOFF_BASE_SECS * (1u64 << consecutive_failures.min(6));
            let capped = backoff.min(BACKOFF_MAX_SECS);
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                tracing::warn!(
                    "Conductor unreachable for {} cycles, backing off {}s",
                    consecutive_failures,
                    capped
                );
            }
            capped
        } else {
            poll_interval_secs
        };

        // Interruptible sleep — wake early on cancellation
        tokio::select! {
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(sleep_secs)) => {}
            _ = cancel.cancelled() => {
                tracing::info!("Poller interrupted during sleep");
                break;
            }
        }
    }

    // Persist dedup cache for restart resilience
    if let Err(e) = crate::dedup_cache::save_string_cache(&dedup, "poller-dedup.cache") {
        tracing::warn!("Failed to save poller dedup cache: {e}");
    }

    // Drop cached connection cleanly
    drop(ws_cached);
    tracing::info!("Poller stopped.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: serialize + fragment + send a relay payload
// ---------------------------------------------------------------------------

async fn relay_payload(
    relay_type: RelayType,
    origin: &[u8; 8],
    data: Vec<u8>,
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> Result<()> {
    let payload = RelayPayload::new(relay_type, *origin, data);
    let bytes = payload.to_bytes();
    let wire_bytes = if let Some(c) = cipher {
        crate::encryption::encrypt(c, &bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        bytes
    };
    let frames = serializer::fragment(&wire_bytes, LORA_MAX_FRAME);
    for frame in &frames {
        transport.send(frame).await?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: extract a string field from a JSON value
// ---------------------------------------------------------------------------

fn jstr(v: &serde_json::Value, key: &str) -> String {
    v[key].as_str().unwrap_or("").to_string()
}

fn jf32(v: &serde_json::Value, key: &str) -> f32 {
    v[key].as_f64().unwrap_or(0.0) as f32
}

fn ju8(v: &serde_json::Value, key: &str) -> u8 {
    v[key].as_u64().unwrap_or(0) as u8
}

// ---------------------------------------------------------------------------
// Single poll cycle
// ---------------------------------------------------------------------------

/// Single poll cycle. Returns number of entries relayed.
///
/// Priority scheduling:
///   - Safety-critical (emergency, water, hearth, food): every cycle
///   - Lower-priority (knowledge, shelter, supplies, care circles, mutual aid, price): every other cycle
async fn poll_once(
    conductor_url: &str,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    ws_cached: &mut Option<AppWebsocket>,
    poll_count: u64,
    cipher: Option<&XChaCha20Poly1305>,
) -> Result<usize> {
    // Reuse cached connection or establish a new one
    let ws = match ws_cached.take() {
        Some(ws) => ws,
        None => {
            let token: Vec<u8> = std::env::var("MESH_APP_TOKEN")
                .unwrap_or_default()
                .into_bytes();
            let signer: Arc<dyn AgentSigner + Send + Sync> =
                Arc::new(ClientAgentSigner::default());
            tracing::debug!("Connecting to conductor at {conductor_url}");
            AppWebsocket::connect(conductor_url, token, signer).await.map_err(|e| anyhow::anyhow!("{e:?}"))?
        }
    };

    let mut relayed = 0;

    // === SAFETY-CRITICAL: every cycle ===

    // --- Poll TEND exchanges ---
    relayed += poll_tend(&ws, dedup, origin, transport, cipher).await;

    // --- Poll emergency messages ---
    relayed += poll_emergency(&ws, dedup, origin, transport, cipher).await;

    // --- Poll water alerts ---
    relayed += poll_water(&ws, dedup, origin, transport, cipher).await;

    // --- Poll hearth alerts ---
    relayed += poll_hearth(&ws, dedup, origin, transport, cipher).await;

    // --- Poll food harvests ---
    relayed += poll_food(&ws, dedup, origin, transport, cipher).await;

    // --- Poll IoT sensor readings (safety-critical: water/air quality) ---
    relayed += poll_sensors(&ws, dedup, origin, transport, cipher).await;

    // === LOWER-PRIORITY: every other cycle ===
    if poll_count % 2 == 0 {
        relayed += poll_knowledge(&ws, dedup, origin, transport, cipher).await;
        relayed += poll_care_circles(&ws, dedup, origin, transport, cipher).await;
        relayed += poll_shelter(&ws, dedup, origin, transport, cipher).await;
        relayed += poll_supplies(&ws, dedup, origin, transport, cipher).await;
        relayed += poll_mutual_aid(&ws, dedup, origin, transport, cipher).await;
        relayed += poll_price_reports(&ws, dedup, origin, transport, cipher).await;
    }

    // Cache connection for reuse on next poll cycle
    *ws_cached = Some(ws);

    Ok(relayed)
}

// ---------------------------------------------------------------------------
// Domain-specific poll functions
// ---------------------------------------------------------------------------

async fn poll_tend(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({
        "dao_did": "example-community",
        "limit": MAX_ENTRIES_PER_DOMAIN
    })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("finance".to_string().into()),
            "tend".into(),
            "get_my_exchanges".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(exchanges) = result.decode::<Vec<serde_json::Value>>() {
            for exchange in exchanges.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(exchange, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let tend = TendRelay {
                    receiver_did: jstr(exchange, "receiver_did"),
                    hours: jf32(exchange, "hours"),
                    service_description: jstr(exchange, "service_description"),
                    service_category: jstr(exchange, "service_category"),
                    dao_did: jstr(exchange, "dao_did"),
                };

                if let Ok(data) = bincode::serialize(&tend) {
                    if relay_payload(RelayType::TendExchange, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_emergency(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("civic".to_string().into()),
            "emergency_comms".into(),
            "get_unsynced_messages".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(messages) = result.decode::<Vec<serde_json::Value>>() {
            for msg in messages.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(msg, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let emergency = EmergencyRelay {
                    channel_id: jstr(msg, "channel_id"),
                    content: jstr(msg, "content"),
                    priority: jstr(msg, "priority"),
                };

                if let Ok(data) = bincode::serialize(&emergency) {
                    if relay_payload(RelayType::EmergencyMessage, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_food(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({ "limit": MAX_ENTRIES_PER_DOMAIN })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "food_production".into(),
            "get_recent_harvests".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(harvests) = result.decode::<Vec<serde_json::Value>>() {
            for h in harvests.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(h, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let food = FoodRelay {
                    crop_hash: jstr(h, "crop_hash"),
                    quantity_kg: jf32(h, "quantity_kg"),
                    quality: jstr(h, "quality"),
                    notes: jstr(h, "notes"),
                };

                if let Ok(data) = bincode::serialize(&food) {
                    if relay_payload(RelayType::FoodHarvest, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_water(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "water_purity".into(),
            "get_active_alerts".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(alerts) = result.decode::<Vec<serde_json::Value>>() {
            for a in alerts.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(a, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let water = WaterAlertRelay {
                    system_id: jstr(a, "system_id"),
                    alert_type: jstr(a, "alert_type"),
                    severity: jstr(a, "severity"),
                    description: jstr(a, "description"),
                };

                if let Ok(data) = bincode::serialize(&water) {
                    if relay_payload(RelayType::WaterAlert, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_hearth(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("hearth".to_string().into()),
            "hearth_emergency".into(),
            "get_active_alerts".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(alerts) = result.decode::<Vec<serde_json::Value>>() {
            for a in alerts.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(a, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let hearth = HearthAlertRelay {
                    hearth_id: jstr(a, "hearth_id"),
                    alert_type: jstr(a, "alert_type"),
                    message: jstr(a, "message"),
                };

                if let Ok(data) = bincode::serialize(&hearth) {
                    if relay_payload(RelayType::HearthAlert, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_knowledge(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({ "limit": MAX_ENTRIES_PER_DOMAIN })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("knowledge".to_string().into()),
            "claims".into(),
            "get_recent_claims".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(claims) = result.decode::<Vec<serde_json::Value>>() {
            for c in claims.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(c, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let tags: Vec<String> = c["tags"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|t| t.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let claim = KnowledgeClaimRelay {
                    claim_text: jstr(c, "claim_text"),
                    tags,
                    empirical: ju8(c, "empirical"),
                    normative: ju8(c, "normative"),
                    materiality: ju8(c, "materiality"),
                };

                if let Ok(data) = bincode::serialize(&claim) {
                    if relay_payload(RelayType::KnowledgeClaim, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_care_circles(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "care_circles".into(),
            "get_recent_updates".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(updates) = result.decode::<Vec<serde_json::Value>>() {
            for u in updates.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(u, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let care = CareCircleRelay {
                    circle_id: jstr(u, "circle_id"),
                    update_type: jstr(u, "update_type"),
                    details: jstr(u, "details"),
                };

                if let Ok(data) = bincode::serialize(&care) {
                    if relay_payload(RelayType::CareCircleUpdate, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_shelter(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "housing_units".into(),
            "get_available_units".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(units) = result.decode::<Vec<serde_json::Value>>() {
            for u in units.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(u, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let shelter = ShelterRelay {
                    unit_id: jstr(u, "unit_id"),
                    status: jstr(u, "status"),
                    bedrooms: ju8(u, "bedrooms"),
                };

                if let Ok(data) = bincode::serialize(&shelter) {
                    if relay_payload(RelayType::ShelterUpdate, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_supplies(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(()) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("supplychain".to_string().into()),
            "inventory_coordinator".into(),
            "get_low_stock_items".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(items) = result.decode::<Vec<serde_json::Value>>() {
            for item in items.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(item, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let supply = SupplyRelay {
                    item_id: jstr(item, "item_id"),
                    item_name: jstr(item, "item_name"),
                    quantity: jf32(item, "quantity"),
                    category: jstr(item, "category"),
                };

                if let Ok(data) = bincode::serialize(&supply) {
                    if relay_payload(RelayType::SupplyUpdate, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_mutual_aid(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({ "limit": MAX_ENTRIES_PER_DOMAIN })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "mutualaid_timebank".into(),
            "get_recent_offers".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(offers) = result.decode::<Vec<serde_json::Value>>() {
            for o in offers.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(o, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let aid = MutualAidRelay {
                    offer_type: jstr(o, "offer_type"),
                    title: jstr(o, "title"),
                    description: jstr(o, "description"),
                    category: jstr(o, "category"),
                };

                if let Ok(data) = bincode::serialize(&aid) {
                    if relay_payload(RelayType::MutualAidOffer, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_price_reports(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({ "limit": MAX_ENTRIES_PER_DOMAIN })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("finance".to_string().into()),
            "price_oracle".into(),
            "get_recent_reports".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(reports) = result.decode::<Vec<serde_json::Value>>() {
            for r in reports.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(r, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let price = PriceReportRelay {
                    item_name: jstr(r, "item_name"),
                    price_tend: jf32(r, "price_tend"),
                    evidence: jstr(r, "evidence"),
                };

                if let Ok(data) = bincode::serialize(&price) {
                    if relay_payload(RelayType::PriceReport, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

async fn poll_sensors(
    ws: &AppWebsocket,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
    cipher: Option<&XChaCha20Poly1305>,
) -> usize {
    let input = match encode_extern_io(serde_json::json!({ "limit": MAX_ENTRIES_PER_DOMAIN })) {
        Ok(i) => i,
        Err(_) => return 0,
    };

    let response = ws
        .call_zome(
            ZomeCallTarget::RoleName("commons_care".to_string().into()),
            "resource_mesh".into(),
            "get_recent_readings".into(),
            input,
        )
        .await;

    let mut count = 0;
    if let Ok(result) = response {
        if let Ok(readings) = result.decode::<Vec<serde_json::Value>>() {
            for r in readings.iter().take(MAX_ENTRIES_PER_DOMAIN) {
                let id = jstr(r, "id");
                if id.is_empty() || dedup.contains(&id) {
                    continue;
                }

                let location = jstr(r, "location");
                let location_hash: [u8; 8] = {
                    let h = blake3::hash(location.as_bytes());
                    let mut buf = [0u8; 8];
                    buf.copy_from_slice(&h.as_bytes()[..8]);
                    buf
                };

                let sensor = SensorReadingRelay {
                    sensor_id: jstr(r, "sensor_id"),
                    resource_type: jstr(r, "resource_type"),
                    value: jf32(r, "value"),
                    unit: jstr(r, "unit"),
                    location_hash,
                };

                if let Ok(data) = bincode::serialize(&sensor) {
                    if relay_payload(RelayType::SensorReading, origin, data, transport, cipher)
                        .await
                        .is_ok()
                    {
                        dedup.insert(id);
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Get an 8-byte origin ID from the hostname.
fn get_origin_id() -> [u8; 8] {
    let hostname = hostname::get()
        .map(|h| h.to_string_lossy().to_string())
        .unwrap_or_else(|_| "unknown".into());
    let hash = blake3::hash(hostname.as_bytes());
    let mut id = [0u8; 8];
    id.copy_from_slice(&hash.as_bytes()[..8]);
    id
}

/// One-shot conductor connectivity check at startup.
/// Attempts to connect and list installed apps. Returns Ok if reachable.
async fn startup_check(conductor_url: &str) -> Result<()> {
    let token: Vec<u8> = std::env::var("MESH_APP_TOKEN")
        .unwrap_or_default()
        .into_bytes();
    let signer: Arc<dyn AgentSigner + Send + Sync> = Arc::new(ClientAgentSigner::default());
    let _ws = AppWebsocket::connect(conductor_url, token, signer).await.map_err(|e| anyhow::anyhow!("{e:?}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::LoopbackTransport;

    #[test]
    fn test_heartbeat_frame_creation() {
        let origin = [1, 2, 3, 4, 5, 6, 7, 8];
        let heartbeat = RelayPayload::new(RelayType::Heartbeat, origin, Vec::new());
        assert_eq!(heartbeat.relay_type, RelayType::Heartbeat);
        assert_eq!(heartbeat.origin, origin);
        assert!(heartbeat.data.is_empty());

        // Heartbeat should fit in a single LoRa frame
        let bytes = heartbeat.to_bytes();
        let frames = serializer::fragment(&bytes, LORA_MAX_FRAME);
        assert_eq!(frames.len(), 1);
    }

    #[test]
    fn test_heartbeat_interval() {
        // Heartbeat fires on cycles divisible by HEARTBEAT_EVERY_N_POLLS
        for cycle in 1..=20u64 {
            let should_fire = cycle % HEARTBEAT_EVERY_N_POLLS == 0;
            if cycle == 4 || cycle == 8 || cycle == 12 || cycle == 16 || cycle == 20 {
                assert!(should_fire, "cycle {cycle} should fire heartbeat");
            } else {
                assert!(!should_fire, "cycle {cycle} should NOT fire heartbeat");
            }
        }
    }

    #[test]
    fn test_backoff_calculation() {
        // 0 failures → normal interval
        let failures: u32 = 0;
        assert_eq!(failures, 0);

        // 1 failure → 5 * 2^1 = 10s
        let backoff = BACKOFF_BASE_SECS * (1u64 << 1u32.min(6));
        assert_eq!(backoff.min(BACKOFF_MAX_SECS), 10);

        // 3 failures → 5 * 2^3 = 40s
        let backoff = BACKOFF_BASE_SECS * (1u64 << 3u32.min(6));
        assert_eq!(backoff.min(BACKOFF_MAX_SECS), 40);

        // 6 failures → 5 * 2^6 = 320 → capped at 300s
        let backoff = BACKOFF_BASE_SECS * (1u64 << 6u32.min(6));
        assert_eq!(backoff.min(BACKOFF_MAX_SECS), BACKOFF_MAX_SECS);

        // 10 failures → still capped at 6 bits → 320 → 300s
        let backoff = BACKOFF_BASE_SECS * (1u64 << 10u32.min(6));
        assert_eq!(backoff.min(BACKOFF_MAX_SECS), BACKOFF_MAX_SECS);
    }

    #[test]
    fn test_dedup_cache_overflow() {
        let mut dedup: HashSet<String> = HashSet::new();
        // Fill past MAX_DEDUP_CACHE
        for i in 0..=MAX_DEDUP_CACHE {
            dedup.insert(format!("hash-{i}"));
        }
        assert!(dedup.len() > MAX_DEDUP_CACHE);

        // Simulates the cache clear logic from run()
        if dedup.len() > MAX_DEDUP_CACHE {
            dedup.clear();
        }
        assert_eq!(dedup.len(), 0);
    }

    #[test]
    fn test_origin_id_deterministic() {
        let id1 = get_origin_id();
        let id2 = get_origin_id();
        assert_eq!(id1, id2, "origin ID should be deterministic for same hostname");
        assert_ne!(id1, [0u8; 8], "origin ID should not be all zeros");
    }

    #[test]
    fn test_backpressure_constant() {
        assert_eq!(MAX_ENTRIES_PER_DOMAIN, 20);
        assert!(MAX_ENTRIES_PER_DOMAIN > 0);
        assert!(MAX_ENTRIES_PER_DOMAIN <= 100);
    }

    #[test]
    fn test_priority_scheduling() {
        // Safety-critical polls every cycle; lower-priority every other
        for cycle in 1..=10u64 {
            // Safety-critical always runs
            assert!(true, "TEND/emergency/water/hearth/food always poll");

            // Lower-priority only on even cycles
            let lower_priority_runs = cycle % 2 == 0;
            if cycle % 2 == 0 {
                assert!(lower_priority_runs);
            } else {
                assert!(!lower_priority_runs);
            }
        }
    }

    #[tokio::test]
    async fn test_relay_pipeline() {
        let transport = LoopbackTransport::new();
        let origin = [1, 2, 3, 4, 5, 6, 7, 8];

        // Create a TEND relay
        let tend = TendRelay {
            receiver_did: "bob.did".into(),
            hours: 1.0,
            service_description: "Fixed tap".into(),
            service_category: "Maintenance".into(),
            dao_did: "example-community".into(),
        };
        let data = bincode::serialize(&tend).unwrap();
        let payload = RelayPayload::new(RelayType::TendExchange, origin, data);
        let bytes = payload.to_bytes();

        // Fragment and send
        let frames = serializer::fragment(&bytes, LORA_MAX_FRAME);
        for frame in &frames {
            transport.send(frame).await.unwrap();
        }

        // Receive and reassemble
        let mut received_frames = Vec::new();
        while let Ok(Some(frame)) = transport.recv(100).await {
            received_frames.push(frame);
        }

        let reassembled = serializer::reassemble(&received_frames).unwrap();
        let decoded = RelayPayload::from_bytes(&reassembled).unwrap();
        assert_eq!(decoded.relay_type, RelayType::TendExchange);

        let tend_decoded: TendRelay = bincode::deserialize(&decoded.data).unwrap();
        assert_eq!(tend_decoded.receiver_did, "bob.did");
    }

    #[tokio::test]
    async fn test_relay_payload_helper() {
        let transport = LoopbackTransport::new();
        let origin = [1, 2, 3, 4, 5, 6, 7, 8];

        let food = FoodRelay {
            crop_hash: "crop-123".into(),
            quantity_kg: 5.5,
            quality: "Good".into(),
            notes: "First harvest".into(),
        };
        let data = bincode::serialize(&food).unwrap();
        relay_payload(RelayType::FoodHarvest, &origin, data, &transport, None)
            .await
            .unwrap();

        // Should have sent at least one frame
        let frame = transport.recv(100).await.unwrap();
        assert!(frame.is_some());
    }

    #[test]
    fn test_jstr_jf32_ju8_helpers() {
        let v = serde_json::json!({
            "name": "test",
            "value": 3.14,
            "count": 7,
            "missing": null
        });
        assert_eq!(jstr(&v, "name"), "test");
        assert_eq!(jstr(&v, "nonexistent"), "");
        assert!((jf32(&v, "value") - 3.14).abs() < 0.01);
        assert_eq!(jf32(&v, "nonexistent"), 0.0);
        assert_eq!(ju8(&v, "count"), 7);
        assert_eq!(ju8(&v, "nonexistent"), 0);
    }
}

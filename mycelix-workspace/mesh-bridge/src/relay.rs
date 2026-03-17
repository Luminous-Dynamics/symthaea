//! Mesh Relay — receives mesh packets and replays as zome calls.
//!
//! Listens on the mesh transport for incoming relay payloads,
//! deserializes them, deduplicates by content hash, and replays
//! as zome calls on the local Holochain conductor.

use crate::serializer::{
    self, CareCircleRelay, EmergencyRelay, FoodRelay, HearthAlertRelay, KnowledgeClaimRelay,
    MutualAidRelay, PriceReportRelay, RelayPayload, RelayType, ShelterRelay, SupplyRelay,
    TendRelay, WaterAlertRelay,
};
use crate::transport::MeshTransport;
use crate::BridgeMetrics;
use anyhow::Result;
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ExternIO, ZomeCallTarget};
use crate::dedup_cache::LruBinaryCache;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// Timeout for fragment reassembly (seconds).
const REASSEMBLY_TIMEOUT_SECS: u64 = 30;

/// Maximum in-flight reassembly buffers.
const MAX_REASSEMBLY_BUFFERS: usize = 64;

/// Maximum age of a relayed payload (seconds). Payloads older than this
/// are rejected to prevent replay attacks from stale mesh traffic.
const MAX_PAYLOAD_AGE_SECS: u64 = 3600; // 1 hour

/// Maximum future timestamp tolerance (seconds). Prevents spoofed
/// timestamps from bypassing the age check.
const MAX_FUTURE_TOLERANCE_SECS: u64 = 300; // 5 minutes

/// Maximum payload data size (bytes). Prevents memory exhaustion
/// from adversarial oversized messages.
const MAX_PAYLOAD_DATA_SIZE: usize = 65_536; // 64 KB

/// Maximum inbound messages per minute per peer. Prevents flooding.
const MAX_MESSAGES_PER_MINUTE_PER_PEER: u32 = 60;

/// Run the relay loop: mesh transport → deserializer → conductor.
///
/// Respects the `cancel` token for graceful shutdown — finishes processing
/// any in-flight payload before exiting.
pub async fn run(
    conductor_url: &str,
    transport: Box<dyn MeshTransport>,
    cancel: CancellationToken,
    metrics: BridgeMetrics,
) -> Result<()> {
    tracing::info!("Relay starting, conductor={conductor_url}");

    let cipher = crate::encryption::load_psk();
    if cipher.is_some() {
        tracing::info!("PSK encryption enabled (relay)");
    }

    let mut dedup = LruBinaryCache::from_hash_set(
        crate::dedup_cache::load_binary_cache("relay-dedup.cache"),
        10_000,
    );
    let mut reassembly: HashMap<[u8; 4], ReassemblyBuffer> = HashMap::new();
    let mut ws_cached: Option<AppWebsocket> = None;
    let mut known_peers: HashSet<[u8; 8]> = HashSet::new();
    let mut peer_rate: HashMap<[u8; 8], PeerRateTracker> = HashMap::new();

    loop {
        if cancel.is_cancelled() {
            tracing::info!("Relay shutting down (cancel requested)");
            break;
        }

        match transport.recv(5000).await {
            Ok(Some(frame)) => {
                if frame.len() < 8 {
                    metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Extract hash prefix for reassembly grouping
                let hash_prefix: [u8; 4] = frame[4..8].try_into().unwrap_or([0; 4]);

                // Add to reassembly buffer
                let buffer = reassembly
                    .entry(hash_prefix)
                    .or_insert_with(|| ReassemblyBuffer::new());
                buffer.add_frame(frame);

                // Try reassembly
                if let Some(payload_bytes) = buffer.try_reassemble() {
                    reassembly.remove(&hash_prefix);

                    // --- Decrypt if PSK is configured ---
                    let decrypted_bytes = if let Some(ref c) = cipher {
                        match crate::encryption::decrypt(c, &payload_bytes) {
                            Ok(plain) => plain,
                            Err(e) => {
                                tracing::warn!("Decryption failed (wrong key or plaintext?): {e}");
                                metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                        }
                    } else {
                        payload_bytes
                    };

                    match RelayPayload::from_bytes(&decrypted_bytes) {
                        Ok(payload) => {
                            // --- Security: validate payload size ---
                            if payload.data.len() > MAX_PAYLOAD_DATA_SIZE {
                                tracing::warn!(
                                    reason = "oversized",
                                    size_bytes = payload.data.len(),
                                    relay_type = ?payload.relay_type,
                                    origin = ?&payload.origin[..4],
                                    "REJECTED: payload data too large"
                                );
                                metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            // --- Security: validate timestamp (anti-replay) ---
                            let now = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .unwrap_or_default()
                                .as_secs();
                            if payload.timestamp > now + MAX_FUTURE_TOLERANCE_SECS {
                                tracing::warn!(
                                    reason = "future_timestamp",
                                    payload_ts = payload.timestamp,
                                    now_ts = now,
                                    origin = ?&payload.origin[..4],
                                    "REJECTED: payload timestamp in the future"
                                );
                                metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            if now.saturating_sub(payload.timestamp) > MAX_PAYLOAD_AGE_SECS {
                                tracing::warn!(
                                    reason = "expired",
                                    payload_ts = payload.timestamp,
                                    age_secs = now.saturating_sub(payload.timestamp),
                                    origin = ?&payload.origin[..4],
                                    "REJECTED: payload timestamp too old"
                                );
                                metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            // --- Security: per-peer rate limiting ---
                            let tracker = peer_rate
                                .entry(payload.origin)
                                .or_insert_with(PeerRateTracker::new);
                            if !tracker.allow() {
                                tracing::warn!(
                                    reason = "rate_limited",
                                    origin = ?&payload.origin[..4],
                                    "REJECTED: peer exceeds rate limit"
                                );
                                metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }

                            // Track peer
                            if known_peers.insert(payload.origin) {
                                metrics.peers_seen.fetch_add(1, Ordering::Relaxed);
                            }

                            // Dedup by content hash
                            if dedup.contains(&payload.content_hash) {
                                tracing::debug!("Duplicate payload, skipping");
                                continue;
                            }
                            dedup.insert(payload.content_hash);

                            // Replay as zome call with cached connection
                            match replay_payload(conductor_url, &payload, &mut ws_cached).await {
                                Ok(()) => {
                                    if payload.relay_type != RelayType::Heartbeat {
                                        metrics
                                            .messages_received
                                            .fetch_add(1, Ordering::Relaxed);
                                        metrics.last_recv_at.store(
                                            std::time::SystemTime::now()
                                                .duration_since(std::time::UNIX_EPOCH)
                                                .unwrap_or_default()
                                                .as_millis()
                                                as u64,
                                            Ordering::Relaxed,
                                        );
                                    }
                                }
                                Err(e) => {
                                    // Drop cached connection on failure
                                    ws_cached = None;
                                    metrics
                                        .connection_failures
                                        .fetch_add(1, Ordering::Relaxed);
                                    tracing::warn!("Replay failed: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            metrics.fragments_dropped.fetch_add(1, Ordering::Relaxed);
                            tracing::warn!("Failed to deserialize relay payload: {e}");
                        }
                    }
                }

                // Clean up stale reassembly buffers — count dropped
                let before = reassembly.len();
                cleanup_stale(&mut reassembly);
                let dropped = before.saturating_sub(reassembly.len());
                if dropped > 0 {
                    metrics
                        .fragments_dropped
                        .fetch_add(dropped as u64, Ordering::Relaxed);
                }
            }
            Ok(None) => {
                // Timeout, clean stale buffers
                cleanup_stale(&mut reassembly);
            }
            Err(e) => {
                tracing::warn!("Transport recv error: {e}");
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }

        // LRU eviction handled automatically by LruBinaryCache
    }

    // Persist dedup cache for restart resilience
    if let Err(e) = crate::dedup_cache::save_lru_cache(&dedup, "relay-dedup.cache") {
        tracing::warn!("Failed to save relay dedup cache: {e}");
    }

    // Drop cached connection cleanly
    drop(ws_cached);
    tracing::info!("Relay stopped.");
    Ok(())
}

struct ReassemblyBuffer {
    frames: Vec<Vec<u8>>,
    created_at: std::time::Instant,
}

impl ReassemblyBuffer {
    fn new() -> Self {
        Self {
            frames: Vec::new(),
            created_at: std::time::Instant::now(),
        }
    }

    fn add_frame(&mut self, frame: Vec<u8>) {
        self.frames.push(frame);
    }

    fn try_reassemble(&self) -> Option<Vec<u8>> {
        serializer::reassemble(&self.frames)
    }

    fn is_stale(&self) -> bool {
        self.created_at.elapsed().as_secs() > REASSEMBLY_TIMEOUT_SECS
    }
}

/// Sliding-window rate tracker per peer.
struct PeerRateTracker {
    /// Timestamps of recent messages (ring buffer).
    window: std::collections::VecDeque<std::time::Instant>,
}

impl PeerRateTracker {
    fn new() -> Self {
        Self {
            window: std::collections::VecDeque::with_capacity(
                MAX_MESSAGES_PER_MINUTE_PER_PEER as usize + 1,
            ),
        }
    }

    /// Returns true if the peer is within rate limits.
    fn allow(&mut self) -> bool {
        let now = std::time::Instant::now();
        let one_minute_ago = now - std::time::Duration::from_secs(60);

        // Expire old entries
        while self.window.front().map_or(false, |t| *t < one_minute_ago) {
            self.window.pop_front();
        }

        if self.window.len() >= MAX_MESSAGES_PER_MINUTE_PER_PEER as usize {
            return false;
        }

        self.window.push_back(now);
        true
    }
}

fn cleanup_stale(reassembly: &mut HashMap<[u8; 4], ReassemblyBuffer>) {
    reassembly.retain(|_, buf| !buf.is_stale());
    // Cap total buffers
    while reassembly.len() > MAX_REASSEMBLY_BUFFERS {
        if let Some(oldest_key) = reassembly
            .iter()
            .min_by_key(|(_, buf)| buf.created_at)
            .map(|(k, _)| *k)
        {
            reassembly.remove(&oldest_key);
        } else {
            break;
        }
    }
}

/// Connect to the local Holochain conductor for replay.
async fn connect_conductor(conductor_url: &str) -> Result<AppWebsocket> {
    let token: Vec<u8> = std::env::var("MESH_APP_TOKEN")
        .unwrap_or_default()
        .into_bytes();
    let signer: Arc<dyn AgentSigner + Send + Sync> =
        Arc::new(ClientAgentSigner::default());
    let ws = AppWebsocket::connect(conductor_url, token, signer).await?;
    Ok(ws)
}

/// Replay a relay payload as a zome call on the local conductor.
/// Reuses a cached AppWebsocket connection when available.
async fn replay_payload(
    conductor_url: &str,
    payload: &RelayPayload,
    ws_cached: &mut Option<AppWebsocket>,
) -> Result<()> {
    // Heartbeats don't need a conductor connection
    if payload.relay_type == RelayType::Heartbeat {
        tracing::debug!("Mesh heartbeat from peer {:?}", &payload.origin);
        return Ok(());
    }

    // Reuse cached connection or establish a new one
    let ws = match ws_cached.take() {
        Some(ws) => ws,
        None => {
            tracing::debug!("Connecting to conductor at {conductor_url}");
            connect_conductor(conductor_url).await?
        }
    };

    match payload.relay_type {
        RelayType::TendExchange => {
            let tend: TendRelay = bincode::deserialize(&payload.data)?;
            tracing::info!(
                "Replaying TEND exchange: mesh-peer → {}, {}h",
                tend.receiver_did,
                tend.hours
            );

            let input = ExternIO::encode(serde_json::json!({
                "receiver_did": tend.receiver_did,
                "hours": tend.hours,
                "service_description": tend.service_description,
                "service_category": tend.service_category,
                "dao_did": tend.dao_did,
            }))?;
            ws.call_zome(
                ZomeCallTarget::RoleName("finance".to_string().into()),
                "tend".into(),
                "record_exchange".into(),
                input,
            )
            .await?;
        }
        RelayType::FoodHarvest => {
            let food: FoodRelay = bincode::deserialize(&payload.data)?;
            tracing::info!(
                "Replaying food harvest: {}kg {}",
                food.quantity_kg,
                food.quality
            );

            let input = ExternIO::encode(serde_json::json!({
                "quantity_kg": food.quantity_kg,
                "quality": food.quality,
                "notes": "relayed via mesh",
            }))?;
            ws.call_zome(
                ZomeCallTarget::RoleName("commons_care".to_string().into()),
                "food_production".into(),
                "record_harvest".into(),
                input,
            )
            .await?;
        }
        RelayType::EmergencyMessage => {
            let emergency: EmergencyRelay = bincode::deserialize(&payload.data)?;
            tracing::info!(
                "Replaying emergency message: [{}] {}",
                emergency.priority,
                &emergency.content[..emergency.content.len().min(50)]
            );

            let input = ExternIO::encode(serde_json::json!({
                "channel_id": emergency.channel_id,
                "content": emergency.content,
                "priority": emergency.priority,
            }))?;
            ws.call_zome(
                ZomeCallTarget::RoleName("civic".to_string().into()),
                "emergency_comms".into(),
                "send_message".into(),
                input,
            )
            .await?;
        }
        RelayType::WaterAlert => {
            let water: WaterAlertRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying water alert: {} — {}", water.system_id, water.severity);
            let input = ExternIO::encode(serde_json::json!({
                "system_id": water.system_id, "alert_type": water.alert_type,
                "severity": water.severity, "description": water.description,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("commons_care".to_string().into()),
                "water_purity".into(), "raise_alert".into(), input).await?;
        }
        RelayType::HearthAlert => {
            let hearth: HearthAlertRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying hearth alert: {}", hearth.hearth_id);
            let input = ExternIO::encode(serde_json::json!({
                "hearth_id": hearth.hearth_id, "alert_type": hearth.alert_type, "message": hearth.message,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("hearth".to_string().into()),
                "hearth_emergency".into(), "raise_alert".into(), input).await?;
        }
        RelayType::KnowledgeClaim => {
            let claim: KnowledgeClaimRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying knowledge claim: {}...", &claim.claim_text[..claim.claim_text.len().min(40)]);
            let input = ExternIO::encode(serde_json::json!({
                "claim_text": claim.claim_text, "tags": claim.tags,
                "empirical": claim.empirical, "normative": claim.normative, "materiality": claim.materiality,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("knowledge".to_string().into()),
                "claims".into(), "submit_claim".into(), input).await?;
        }
        RelayType::CareCircleUpdate => {
            let circle: CareCircleRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying care circle update: {}", circle.circle_id);
            let input = ExternIO::encode(serde_json::json!({
                "circle_id": circle.circle_id, "update_type": circle.update_type, "details": circle.details,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("commons_care".to_string().into()),
                "care_circles".into(), "post_update".into(), input).await?;
        }
        RelayType::ShelterUpdate => {
            let shelter: ShelterRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying shelter update: unit {}", shelter.unit_id);
            let input = ExternIO::encode(serde_json::json!({
                "unit_id": shelter.unit_id, "status": shelter.status, "bedrooms": shelter.bedrooms,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("commons_care".to_string().into()),
                "housing_units".into(), "update_status".into(), input).await?;
        }
        RelayType::SupplyUpdate => {
            let supply: SupplyRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying supply update: {} ({})", supply.item_name, supply.quantity);
            let input = ExternIO::encode(serde_json::json!({
                "item_id": supply.item_id, "item_name": supply.item_name,
                "quantity": supply.quantity, "category": supply.category,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("supplychain".to_string().into()),
                "inventory_coordinator".into(), "update_stock".into(), input).await?;
        }
        RelayType::MutualAidOffer => {
            let aid: MutualAidRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying mutual aid {}: {}", aid.offer_type, aid.title);
            let input = ExternIO::encode(serde_json::json!({
                "title": aid.title, "description": aid.description, "category": aid.category,
            }))?;
            let fn_name = if aid.offer_type == "Request" { "create_service_request" } else { "create_service_offer" };
            ws.call_zome(ZomeCallTarget::RoleName("commons_care".to_string().into()),
                "mutualaid_timebank".into(), fn_name.into(), input).await?;
        }
        RelayType::PriceReport => {
            let price: PriceReportRelay = bincode::deserialize(&payload.data)?;
            tracing::info!("Replaying price report: {} = {} TEND", price.item_name, price.price_tend);
            let input = ExternIO::encode(serde_json::json!({
                "item_name": price.item_name, "price_tend": price.price_tend, "evidence": price.evidence,
            }))?;
            ws.call_zome(ZomeCallTarget::RoleName("finance".to_string().into()),
                "price_oracle".into(), "report_price".into(), input).await?;
        }
        RelayType::Heartbeat => unreachable!(), // handled above
    }

    // Cache connection for reuse
    *ws_cached = Some(ws);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reassembly_buffer_stale() {
        let buf = ReassemblyBuffer::new();
        assert!(!buf.is_stale());
    }

    #[test]
    fn test_cleanup_stale_empty() {
        let mut map: HashMap<[u8; 4], ReassemblyBuffer> = HashMap::new();
        cleanup_stale(&mut map);
        assert!(map.is_empty());
    }

    #[test]
    fn test_reassembly_buffer_add_and_try() {
        let mut buf = ReassemblyBuffer::new();
        assert!(buf.frames.is_empty());

        // Single frame that doesn't form a valid payload → reassemble returns None
        buf.add_frame(vec![0; 4]);
        assert_eq!(buf.frames.len(), 1);
        assert!(buf.try_reassemble().is_none());
    }

    #[test]
    fn test_reassembly_buffer_valid_reassembly() {
        use crate::serializer;

        let data = b"test relay payload for reassembly";
        let frames = serializer::fragment(data, 20);
        assert!(frames.len() > 1);

        let mut buf = ReassemblyBuffer::new();
        for frame in &frames {
            buf.add_frame(frame.clone());
        }

        let result = buf.try_reassemble();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), data);
    }

    #[test]
    fn test_cleanup_stale_caps_at_max() {
        let mut map: HashMap<[u8; 4], ReassemblyBuffer> = HashMap::new();

        // Insert more than MAX_REASSEMBLY_BUFFERS entries
        for i in 0..=(MAX_REASSEMBLY_BUFFERS + 10) {
            let key = (i as u32).to_le_bytes();
            map.insert(key, ReassemblyBuffer::new());
        }
        assert!(map.len() > MAX_REASSEMBLY_BUFFERS);

        cleanup_stale(&mut map);
        assert!(map.len() <= MAX_REASSEMBLY_BUFFERS);
    }

    #[test]
    fn test_content_hash_dedup() {
        let mut dedup = LruBinaryCache::new(10_000);

        let hash1: [u8; 32] = blake3::hash(b"payload-1").into();
        let hash2: [u8; 32] = blake3::hash(b"payload-2").into();

        assert!(!dedup.contains(&hash1));
        assert!(dedup.insert(hash1));
        assert!(dedup.contains(&hash1));
        assert!(!dedup.contains(&hash2));

        // Duplicate insert returns false and doesn't grow
        assert!(!dedup.insert(hash1));
        assert_eq!(dedup.len(), 1);

        assert!(dedup.insert(hash2));
        assert_eq!(dedup.len(), 2);
    }

    #[test]
    fn test_peer_rate_tracker_allows_within_limit() {
        let mut tracker = PeerRateTracker::new();
        for _ in 0..MAX_MESSAGES_PER_MINUTE_PER_PEER {
            assert!(tracker.allow());
        }
        // Next should be rejected
        assert!(!tracker.allow());
    }

    #[test]
    fn test_peer_rate_tracker_window_expiry() {
        let mut tracker = PeerRateTracker::new();
        // Fill up the window
        for _ in 0..MAX_MESSAGES_PER_MINUTE_PER_PEER {
            assert!(tracker.allow());
        }
        assert!(!tracker.allow());

        // Manually expire all entries by pushing old timestamps
        tracker.window.clear();
        // After clearing, should allow again
        assert!(tracker.allow());
    }

    #[test]
    fn test_payload_age_constants() {
        assert!(MAX_PAYLOAD_AGE_SECS > 0);
        assert!(MAX_PAYLOAD_AGE_SECS <= 86400, "max age should be <= 24h");
        assert!(MAX_FUTURE_TOLERANCE_SECS > 0);
        assert!(MAX_FUTURE_TOLERANCE_SECS <= 600, "future tolerance should be <= 10min");
        assert!(MAX_PAYLOAD_DATA_SIZE > 0);
        assert!(MAX_PAYLOAD_DATA_SIZE <= 1_048_576, "max data should be <= 1MB");
        assert!(MAX_MESSAGES_PER_MINUTE_PER_PEER >= 10);
    }

    #[test]
    fn test_dedup_lru_eviction_at_cap() {
        let mut dedup = LruBinaryCache::new(100);
        for i in 0..200u32 {
            dedup.insert(blake3::hash(&i.to_le_bytes()).into());
        }
        // LRU keeps exactly capacity entries
        assert_eq!(dedup.len(), 100);
        // Oldest entries (0..100) should be evicted
        assert!(!dedup.contains(&blake3::hash(&0u32.to_le_bytes()).into()));
        // Newest entries should be present
        assert!(dedup.contains(&blake3::hash(&199u32.to_le_bytes()).into()));
    }
}

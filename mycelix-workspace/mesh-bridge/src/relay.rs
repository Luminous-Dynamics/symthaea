//! Mesh Relay — receives mesh packets and replays as zome calls.
//!
//! Listens on the mesh transport for incoming relay payloads,
//! deserializes them, deduplicates by content hash, and replays
//! as zome calls on the local Holochain conductor.

use crate::serializer::{self, EmergencyRelay, FoodRelay, RelayPayload, RelayType, TendRelay};
use crate::transport::MeshTransport;
use anyhow::Result;
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ExternIO, ZomeCallTarget};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Timeout for fragment reassembly (seconds).
const REASSEMBLY_TIMEOUT_SECS: u64 = 30;

/// Maximum in-flight reassembly buffers.
const MAX_REASSEMBLY_BUFFERS: usize = 64;

/// Run the relay loop: mesh transport → deserializer → conductor.
pub async fn run(conductor_url: &str, transport: Box<dyn MeshTransport>) -> Result<()> {
    tracing::info!("Relay starting, conductor={conductor_url}");

    let mut dedup: HashSet<[u8; 32]> = HashSet::new();
    let mut reassembly: HashMap<[u8; 4], ReassemblyBuffer> = HashMap::new();
    let mut ws_cached: Option<AppWebsocket> = None;

    loop {
        match transport.recv(5000).await {
            Ok(Some(frame)) => {
                if frame.len() < 8 {
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

                    match RelayPayload::from_bytes(&payload_bytes) {
                        Ok(payload) => {
                            // Dedup by content hash
                            if dedup.contains(&payload.content_hash) {
                                tracing::debug!("Duplicate payload, skipping");
                                continue;
                            }
                            dedup.insert(payload.content_hash);

                            // Replay as zome call with cached connection
                            match replay_payload(conductor_url, &payload, &mut ws_cached).await {
                                Ok(()) => {}
                                Err(e) => {
                                    // Drop cached connection on failure
                                    ws_cached = None;
                                    tracing::warn!("Replay failed: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to deserialize relay payload: {e}");
                        }
                    }
                }

                // Clean up stale reassembly buffers
                cleanup_stale(&mut reassembly);
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

        // Trim dedup
        if dedup.len() > 10_000 {
            dedup.clear();
        }
    }
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
        let mut dedup: HashSet<[u8; 32]> = HashSet::new();

        let hash1 = blake3::hash(b"payload-1").into();
        let hash2 = blake3::hash(b"payload-2").into();

        assert!(!dedup.contains(&hash1));
        dedup.insert(hash1);
        assert!(dedup.contains(&hash1));
        assert!(!dedup.contains(&hash2));

        // Duplicate insert doesn't grow set
        dedup.insert(hash1);
        assert_eq!(dedup.len(), 1);

        dedup.insert(hash2);
        assert_eq!(dedup.len(), 2);
    }

    #[test]
    fn test_dedup_overflow_clear() {
        let mut dedup: HashSet<[u8; 32]> = HashSet::new();
        for i in 0..10_001u32 {
            dedup.insert(blake3::hash(&i.to_le_bytes()).into());
        }
        assert!(dedup.len() > 10_000);

        // Simulates the relay loop overflow logic
        if dedup.len() > 10_000 {
            dedup.clear();
        }
        assert_eq!(dedup.len(), 0);
    }
}

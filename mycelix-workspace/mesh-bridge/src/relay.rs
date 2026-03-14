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

                            // Replay as zome call
                            if let Err(e) = replay_payload(conductor_url, &payload).await {
                                tracing::warn!("Replay failed: {e}");
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
async fn replay_payload(conductor_url: &str, payload: &RelayPayload) -> Result<()> {
    match payload.relay_type {
        RelayType::TendExchange => {
            let tend: TendRelay = bincode::deserialize(&payload.data)?;
            tracing::info!(
                "Replaying TEND exchange: mesh-peer → {}, {}h",
                tend.receiver_did,
                tend.hours
            );

            let ws = connect_conductor(conductor_url).await?;
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

            let ws = connect_conductor(conductor_url).await?;
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

            let ws = connect_conductor(conductor_url).await?;
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
        RelayType::Heartbeat => {
            tracing::debug!("Mesh heartbeat from peer {:?}", &payload.origin);
        }
    }
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
}

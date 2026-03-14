//! Conductor Poller — polls for new entries and relays to mesh.
//!
//! Watches for new TEND exchanges, food harvests, and emergency messages
//! by polling conductor via AppWebsocket. Deduplicates by action hash.

use crate::serializer::{self, EmergencyRelay, RelayPayload, RelayType, TendRelay};
use crate::transport::MeshTransport;
use anyhow::Result;
use holochain_client::{AgentSigner, AppWebsocket, ClientAgentSigner, ExternIO, ZomeCallTarget};
use std::collections::HashSet;
use std::sync::Arc;

/// Maximum dedup cache size (action hashes we've already relayed).
const MAX_DEDUP_CACHE: usize = 10_000;

/// LoRa frame size limit (SX1276 max payload).
const LORA_MAX_FRAME: usize = 255;

/// Run the poller loop: conductor → serializer → mesh transport.
pub async fn run(
    conductor_url: &str,
    poll_interval_secs: u64,
    transport: Box<dyn MeshTransport>,
) -> Result<()> {
    tracing::info!("Poller starting, conductor={conductor_url}, interval={poll_interval_secs}s");

    let mut dedup: HashSet<String> = HashSet::new();
    let origin = get_origin_id();

    loop {
        // Try to connect and poll
        match poll_once(conductor_url, &mut dedup, &origin, &*transport).await {
            Ok(relayed) => {
                if relayed > 0 {
                    tracing::info!("Relayed {relayed} entries to mesh");
                }
            }
            Err(e) => {
                tracing::warn!("Poll failed (conductor may be offline): {e}");
            }
        }

        // Trim dedup cache
        if dedup.len() > MAX_DEDUP_CACHE {
            dedup.clear();
            tracing::debug!("Dedup cache cleared (exceeded {MAX_DEDUP_CACHE})");
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(poll_interval_secs)).await;
    }
}

/// Single poll cycle. Returns number of entries relayed.
///
/// Connects to the Holochain conductor via AppWebsocket, polls for new
/// TEND exchanges and emergency messages, serializes them into compact
/// relay payloads, and sends them over the mesh transport.
///
/// Requires MESH_APP_TOKEN env var for authentication.
async fn poll_once(
    conductor_url: &str,
    dedup: &mut HashSet<String>,
    origin: &[u8; 8],
    transport: &dyn MeshTransport,
) -> Result<usize> {
    // Auth token from environment (set by resilience-bootstrap.sh)
    let token: Vec<u8> = std::env::var("MESH_APP_TOKEN")
        .unwrap_or_default()
        .into_bytes();

    // For mesh bridge, we use a noop signer — the conductor handles signing
    // for calls from authenticated local connections.
    let signer: Arc<dyn AgentSigner + Send + Sync> =
        Arc::new(ClientAgentSigner::default());

    let ws = AppWebsocket::connect(conductor_url, token, signer).await?;
    let mut relayed = 0;

    // --- Poll TEND exchanges ---
    let tend_input = ExternIO::encode(serde_json::json!({
        "dao_did": "roodepoort-resilience",
        "limit": 50
    }))?;

    let tend_response = ws
        .call_zome(
            ZomeCallTarget::RoleName("finance".to_string().into()),
            "tend".into(),
            "get_my_exchanges".into(),
            tend_input,
        )
        .await;

    if let Ok(result) = tend_response {
        if let Ok(exchanges) = result.decode::<Vec<serde_json::Value>>() {
            for exchange in &exchanges {
                let id = exchange["id"].as_str().unwrap_or("");
                if id.is_empty() || dedup.contains(id) {
                    continue;
                }

                let tend = TendRelay {
                    receiver_did: exchange["receiver_did"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                    hours: exchange["hours"].as_f64().unwrap_or(0.0) as f32,
                    service_description: exchange["service_description"]
                        .as_str()
                        .unwrap_or("")
                        .to_string(),
                    service_category: exchange["service_category"]
                        .as_str()
                        .unwrap_or("GeneralAssistance")
                        .to_string(),
                    dao_did: exchange["dao_did"]
                        .as_str()
                        .unwrap_or("roodepoort-resilience")
                        .to_string(),
                };

                let data = bincode::serialize(&tend)?;
                let payload = RelayPayload::new(RelayType::TendExchange, *origin, data);
                let bytes = payload.to_bytes();
                let frames = serializer::fragment(&bytes, LORA_MAX_FRAME);
                for frame in &frames {
                    transport.send(frame).await?;
                }

                dedup.insert(id.to_string());
                relayed += 1;
            }
        }
    }

    // --- Poll emergency messages ---
    let civic_role: ZomeCallTarget = ZomeCallTarget::RoleName("civic".to_string().into());
    let emergency_input = ExternIO::encode(())?;

    let emergency_response = ws
        .call_zome(
            civic_role,
            "emergency_comms".into(),
            "get_unsynced_messages".into(),
            emergency_input,
        )
        .await;

    if let Ok(result) = emergency_response {
        if let Ok(messages) = result.decode::<Vec<serde_json::Value>>() {
            for msg in &messages {
                let id = msg["id"].as_str().unwrap_or("");
                if id.is_empty() || dedup.contains(id) {
                    continue;
                }

                let emergency = EmergencyRelay {
                    channel_id: msg["channel_id"].as_str().unwrap_or("").to_string(),
                    content: msg["content"].as_str().unwrap_or("").to_string(),
                    priority: msg["priority"].as_str().unwrap_or("Routine").to_string(),
                };

                let data = bincode::serialize(&emergency)?;
                let payload = RelayPayload::new(RelayType::EmergencyMessage, *origin, data);
                let bytes = payload.to_bytes();
                let frames = serializer::fragment(&bytes, LORA_MAX_FRAME);
                for frame in &frames {
                    transport.send(frame).await?;
                }

                dedup.insert(id.to_string());
                relayed += 1;
            }
        }
    }

    Ok(relayed)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::LoopbackTransport;

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
            dao_did: "roodepoort".into(),
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
}

//! Compact binary serialization for mesh relay.
//!
//! NOT full Holochain records — minimum fields to replay as zome calls.
//! Designed for LoRa constraints (255-byte frames, ~20kbps).

use serde::{Deserialize, Serialize};

/// Message types relayable over mesh.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RelayType {
    TendExchange = 1,
    FoodHarvest = 2,
    EmergencyMessage = 3,
    Heartbeat = 255,
}

/// Compact relay payload — stripped to minimum fields for replay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RelayPayload {
    /// Message type tag
    pub relay_type: RelayType,
    /// BLAKE3 hash of the content for dedup
    pub content_hash: [u8; 32],
    /// Originating node ID (truncated agent pubkey)
    pub origin: [u8; 8],
    /// Unix timestamp (seconds)
    pub timestamp: u64,
    /// Payload bytes (type-specific)
    pub data: Vec<u8>,
}

/// TEND exchange relay data — just enough to replay as record_exchange.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TendRelay {
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: String,
    pub dao_did: String,
}

/// Food harvest relay data — enough to replay as record_harvest.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FoodRelay {
    pub crop_hash: String,
    pub quantity_kg: f32,
    pub quality: String,
    pub notes: String,
}

/// Emergency message relay data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmergencyRelay {
    pub channel_id: String,
    pub content: String,
    pub priority: String,
}

impl RelayPayload {
    /// Serialize to compact binary (bincode).
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }

    /// Deserialize from binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        bincode::deserialize(bytes)
    }

    /// Create a new relay payload with auto-computed content hash.
    pub fn new(relay_type: RelayType, origin: [u8; 8], data: Vec<u8>) -> Self {
        let content_hash = blake3::hash(&data).into();
        Self {
            relay_type,
            content_hash,
            origin,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            data,
        }
    }
}

/// Fragment a payload for LoRa (max frame size).
/// Uses simple chunking with sequence numbers.
/// FEC can be added on top (see Symthaea's lora_fragment.rs for XOR FEC).
pub fn fragment(payload: &[u8], max_frame: usize) -> Vec<Vec<u8>> {
    if max_frame < 8 {
        return vec![payload.to_vec()];
    }
    // Header: [total_fragments: u16, fragment_index: u16, content_hash: 4 bytes]
    let header_size = 8;
    let chunk_size = max_frame - header_size;
    let chunks: Vec<&[u8]> = payload.chunks(chunk_size).collect();
    let total = chunks.len() as u16;
    let hash_prefix: [u8; 4] = blake3::hash(payload).as_bytes()[..4].try_into().unwrap();

    chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut frame = Vec::with_capacity(max_frame);
            frame.extend_from_slice(&total.to_le_bytes());
            frame.extend_from_slice(&(i as u16).to_le_bytes());
            frame.extend_from_slice(&hash_prefix);
            frame.extend_from_slice(chunk);
            frame
        })
        .collect()
}

/// Reassemble fragments into the original payload.
/// Returns None if not all fragments received yet.
pub fn reassemble(frames: &[Vec<u8>]) -> Option<Vec<u8>> {
    if frames.is_empty() {
        return None;
    }

    // Parse header from first frame
    let first = &frames[0];
    if first.len() < 8 {
        return None;
    }
    let total = u16::from_le_bytes([first[0], first[1]]) as usize;
    let hash_prefix: [u8; 4] = first[4..8].try_into().ok()?;

    if frames.len() < total {
        return None;
    }

    // Sort by fragment index, validate hash prefix matches
    let mut indexed: Vec<(usize, &[u8])> = Vec::new();
    for frame in frames {
        if frame.len() < 8 {
            continue;
        }
        let idx = u16::from_le_bytes([frame[2], frame[3]]) as usize;
        let hp: [u8; 4] = frame[4..8].try_into().ok()?;
        if hp != hash_prefix {
            continue;
        }
        indexed.push((idx, &frame[8..]));
    }

    indexed.sort_by_key(|(idx, _)| *idx);

    if indexed.len() < total {
        return None;
    }

    let mut payload = Vec::new();
    for (_, data) in &indexed[..total] {
        payload.extend_from_slice(data);
    }

    // Verify hash prefix matches
    let computed: [u8; 4] = blake3::hash(&payload).as_bytes()[..4].try_into().ok()?;
    if computed != hash_prefix {
        return None;
    }

    Some(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_serialization() {
        let payload = RelayPayload::new(
            RelayType::TendExchange,
            [1, 2, 3, 4, 5, 6, 7, 8],
            b"test data".to_vec(),
        );
        let bytes = payload.to_bytes();
        let decoded = RelayPayload::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.relay_type, RelayType::TendExchange);
        assert_eq!(decoded.data, b"test data");
        assert_eq!(decoded.content_hash, payload.content_hash);
    }

    #[test]
    fn test_fragment_reassemble() {
        let data = b"Hello, this is a test payload that should be fragmented into multiple frames for LoRa transmission.";
        let frames = fragment(data, 20);
        assert!(frames.len() > 1);
        let reassembled = reassemble(&frames).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_fragment_single_frame() {
        let data = b"small";
        let frames = fragment(data, 255);
        assert_eq!(frames.len(), 1);
        let reassembled = reassemble(&frames).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_incomplete_reassembly() {
        let data = b"Hello, this is a test payload that should be fragmented.";
        let frames = fragment(data, 20);
        assert!(frames.len() > 2);
        // Only pass first frame — should return None
        assert!(reassemble(&frames[..1]).is_none());
    }

    #[test]
    fn test_content_hash() {
        let payload = RelayPayload::new(
            RelayType::EmergencyMessage,
            [0; 8],
            b"emergency!".to_vec(),
        );
        let expected = blake3::hash(b"emergency!");
        assert_eq!(payload.content_hash, <[u8; 32]>::from(expected));
    }

    #[test]
    fn test_tend_relay_serialization() {
        let relay = TendRelay {
            receiver_did: "alice.did".into(),
            hours: 2.5,
            service_description: "Fixed plumbing".into(),
            service_category: "Maintenance".into(),
            dao_did: "roodepoort".into(),
        };
        let bytes = bincode::serialize(&relay).unwrap();
        let decoded: TendRelay = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.receiver_did, "alice.did");
        assert!((decoded.hours - 2.5).abs() < f32::EPSILON);
    }
}

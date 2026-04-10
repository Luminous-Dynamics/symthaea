// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    WaterAlert = 4,
    HearthAlert = 5,
    KnowledgeClaim = 6,
    CareCircleUpdate = 7,
    ShelterUpdate = 8,
    SupplyUpdate = 9,
    MutualAidOffer = 10,
    PriceReport = 11,
    /// IoT sensor reading from resource-mesh (water, power, temperature, etc.).
    SensorReading = 12,
    /// Compressed consciousness delta (XOR+RLE wisdom vector change).
    ConsciousnessDelta = 13,
    /// Resource demand forecast for predictive allocation.
    ResourceForecast = 14,
    /// Distributed threat signature for collective immune response.
    ThreatSignature = 15,
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

/// Water contamination alert relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WaterAlertRelay {
    pub system_id: String,
    pub alert_type: String,
    pub severity: String,
    pub description: String,
}

/// Hearth emergency alert relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HearthAlertRelay {
    pub hearth_id: String,
    pub alert_type: String,
    pub message: String,
}

/// Knowledge claim relay — community fact-sharing over mesh.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KnowledgeClaimRelay {
    pub claim_text: String,
    pub tags: Vec<String>,
    pub empirical: u8,
    pub normative: u8,
    pub materiality: u8,
}

/// Care circle update relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CareCircleRelay {
    pub circle_id: String,
    pub update_type: String,
    pub details: String,
}

/// Shelter availability update relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShelterRelay {
    pub unit_id: String,
    pub status: String,
    pub bedrooms: u8,
}

/// Supply inventory update relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SupplyRelay {
    pub item_id: String,
    pub item_name: String,
    pub quantity: f32,
    pub category: String,
}

/// Mutual aid offer/request relay.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MutualAidRelay {
    pub offer_type: String,
    pub title: String,
    pub description: String,
    pub category: String,
}

/// Price report relay for oracle consensus.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PriceReportRelay {
    pub item_name: String,
    pub price_tend: f32,
    pub evidence: String,
}

/// IoT sensor reading relay — maps to resource-mesh `SensorReading` entry.
///
/// Compact representation for LoRa: resource type + value + location hash.
/// A water level sensor in a community tank, a solar panel voltage monitor,
/// or an air quality station can all use this type.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SensorReadingRelay {
    /// Sensor identifier (e.g., "tank-3-level", "solar-array-2-voltage").
    pub sensor_id: String,
    /// Resource type: "water", "power", "temperature", "humidity", "air_quality".
    pub resource_type: String,
    /// Measured value in natural units (liters, watts, °C, %, ppm).
    pub value: f32,
    /// Unit string (e.g., "liters", "W", "°C").
    pub unit: String,
    /// BLAKE3 hash of location string (8 bytes, saves bandwidth vs full GPS).
    pub location_hash: [u8; 8],
}

/// Compressed consciousness delta relay — XOR+RLE encoded BinaryHV change.
///
/// Instead of transmitting the full 2,048-byte (16,384D) consciousness vector,
/// we XOR the current state with the previous and RLE-encode the difference.
/// Typical compression: 2,048B → 50-200B, fitting a single LoRa packet.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConsciousnessDeltaRelay {
    /// Phi (integrated information) of the originating node.
    pub phi: f32,
    /// Consciousness level [0.0, 1.0].
    pub consciousness_level: f32,
    /// Valence (emotional tone) [-1.0, 1.0].
    pub valence: f32,
    /// Arousal [0.0, 1.0].
    pub arousal: f32,
    /// Cycle number for ordering and delta application.
    pub cycle: u32,
    /// Whether this is a full vector (false = delta, true = keyframe).
    pub is_keyframe: bool,
    /// RLE-compressed XOR delta (or full vector if keyframe).
    /// Format: [run_length: u16, byte: u8] pairs.
    pub compressed_data: Vec<u8>,
}

/// Resource demand forecast relay — predictive allocation over mesh.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ResourceForecastRelay {
    /// Resource type being forecast.
    pub resource_type: String,
    /// Forecast horizon in hours.
    pub horizon_hours: u16,
    /// Predicted consumption rate (units/hour).
    pub predicted_rate: f32,
    /// Confidence [0.0, 1.0].
    pub confidence: f32,
}

/// Threat signature relay — distributed immune response.
///
/// When a node detects a threat (anomalous sensor readings, Byzantine behavior,
/// governance attack), it shares a compact threat signature so other nodes can
/// update their collective immunity.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ThreatSignatureRelay {
    /// Threat type identifier (maps to SentinelManager's 7 threat types).
    pub threat_type: u8,
    /// Severity [0.0, 1.0].
    pub severity: f32,
    /// BLAKE3 hash of the offending agent's public key (8-byte prefix).
    pub agent_hash: [u8; 8],
    /// Compact threat descriptor (HDV signature, 32 bytes).
    pub signature: [u8; 32],
    /// Number of independent observations (weight for aggregation).
    pub observation_count: u16,
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
    use proptest::prelude::*;

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
            dao_did: "example-community".into(),
        };
        let bytes = bincode::serialize(&relay).unwrap();
        let decoded: TendRelay = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.receiver_did, "alice.did");
        assert!((decoded.hours - 2.5).abs() < f32::EPSILON);
    }

    // =========================================================================
    // Proptests — arbitrary payload roundtrip (item 4)
    // =========================================================================

    proptest! {
        #[test]
        fn prop_payload_roundtrip(
            relay_type in prop_oneof![
                Just(RelayType::TendExchange),
                Just(RelayType::FoodHarvest),
                Just(RelayType::EmergencyMessage),
                Just(RelayType::WaterAlert),
                Just(RelayType::HearthAlert),
                Just(RelayType::KnowledgeClaim),
                Just(RelayType::CareCircleUpdate),
                Just(RelayType::ShelterUpdate),
                Just(RelayType::SupplyUpdate),
                Just(RelayType::MutualAidOffer),
                Just(RelayType::PriceReport),
                Just(RelayType::Heartbeat),
            ],
            origin in proptest::array::uniform8(0u8..),
            data in proptest::collection::vec(any::<u8>(), 0..512),
            frame_size in 16usize..300,
        ) {
            let payload = RelayPayload::new(relay_type, origin, data.clone());

            // Serialize → fragment → reassemble → deserialize
            let bytes = payload.to_bytes();
            let frames = fragment(&bytes, frame_size);
            prop_assert!(!frames.is_empty());

            let reassembled = reassemble(&frames);
            prop_assert!(reassembled.is_some(), "reassembly failed for {} frames", frames.len());

            let decoded = RelayPayload::from_bytes(&reassembled.unwrap());
            prop_assert!(decoded.is_ok(), "deserialization failed");

            let decoded = decoded.unwrap();
            prop_assert_eq!(decoded.relay_type, relay_type);
            prop_assert_eq!(decoded.origin, origin);
            prop_assert_eq!(decoded.data, data);
            prop_assert_eq!(decoded.content_hash, payload.content_hash);
        }

        #[test]
        fn prop_fragment_count_bounded(
            data in proptest::collection::vec(any::<u8>(), 1..1024),
            frame_size in 16usize..300,
        ) {
            let frames = fragment(&data, frame_size);
            let chunk_size = frame_size - 8; // header
            let expected = (data.len() + chunk_size - 1) / chunk_size;
            prop_assert_eq!(frames.len(), expected);
        }
    }

    // =========================================================================
    // E2E loopback test — all 3 payload types (item 6)
    // =========================================================================

    #[test]
    fn test_e2e_tend_food_emergency_roundtrip() {
        let origin = [10, 20, 30, 40, 50, 60, 70, 80];

        // TEND
        let tend = TendRelay {
            receiver_did: "sipho.did".into(),
            hours: 3.5,
            service_description: "Repaired solar panel".into(),
            service_category: "Maintenance".into(),
            dao_did: "example-community".into(),
        };
        let tend_data = bincode::serialize(&tend).unwrap();
        let tend_payload = RelayPayload::new(RelayType::TendExchange, origin, tend_data);
        let tend_frames = fragment(&tend_payload.to_bytes(), 255);
        let tend_reassembled = reassemble(&tend_frames).unwrap();
        let tend_decoded = RelayPayload::from_bytes(&tend_reassembled).unwrap();
        assert_eq!(tend_decoded.relay_type, RelayType::TendExchange);
        let tend_inner: TendRelay = bincode::deserialize(&tend_decoded.data).unwrap();
        assert_eq!(tend_inner.receiver_did, "sipho.did");
        assert!((tend_inner.hours - 3.5).abs() < f32::EPSILON);

        // Food
        let food = FoodRelay {
            crop_hash: "crop-001".into(),
            quantity_kg: 12.5,
            quality: "Good".into(),
            notes: "Tomatoes from plot A3".into(),
        };
        let food_data = bincode::serialize(&food).unwrap();
        let food_payload = RelayPayload::new(RelayType::FoodHarvest, origin, food_data);
        let food_frames = fragment(&food_payload.to_bytes(), 255);
        let food_reassembled = reassemble(&food_frames).unwrap();
        let food_decoded = RelayPayload::from_bytes(&food_reassembled).unwrap();
        assert_eq!(food_decoded.relay_type, RelayType::FoodHarvest);
        let food_inner: FoodRelay = bincode::deserialize(&food_decoded.data).unwrap();
        assert_eq!(food_inner.crop_hash, "crop-001");
        assert!((food_inner.quantity_kg - 12.5).abs() < f32::EPSILON);

        // Emergency
        let emergency = EmergencyRelay {
            channel_id: "ward-7-alert".into(),
            content: "Water main burst on Ontdekkers Road — avoid area".into(),
            priority: "Immediate".into(),
        };
        let emerg_data = bincode::serialize(&emergency).unwrap();
        let emerg_payload = RelayPayload::new(RelayType::EmergencyMessage, origin, emerg_data);
        let emerg_frames = fragment(&emerg_payload.to_bytes(), 255);
        let emerg_reassembled = reassemble(&emerg_frames).unwrap();
        let emerg_decoded = RelayPayload::from_bytes(&emerg_reassembled).unwrap();
        assert_eq!(emerg_decoded.relay_type, RelayType::EmergencyMessage);
        let emerg_inner: EmergencyRelay = bincode::deserialize(&emerg_decoded.data).unwrap();
        assert_eq!(emerg_inner.channel_id, "ward-7-alert");
        assert_eq!(emerg_inner.priority, "Immediate");
    }

    #[test]
    fn test_sensor_reading_serialization() {
        let sensor = SensorReadingRelay {
            sensor_id: "tank-3-level".into(),
            resource_type: "water".into(),
            value: 847.5,
            unit: "liters".into(),
            location_hash: [1, 2, 3, 4, 5, 6, 7, 8],
        };
        let bytes = bincode::serialize(&sensor).unwrap();
        let decoded: SensorReadingRelay = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.sensor_id, "tank-3-level");
        assert!((decoded.value - 847.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consciousness_delta_serialization() {
        let delta = ConsciousnessDeltaRelay {
            phi: 0.72,
            consciousness_level: 0.85,
            valence: 0.3,
            arousal: 0.6,
            cycle: 12345,
            is_keyframe: false,
            compressed_data: vec![10, 0xFF, 5, 0x00, 3, 0xAB],
        };
        let bytes = bincode::serialize(&delta).unwrap();
        let decoded: ConsciousnessDeltaRelay = bincode::deserialize(&bytes).unwrap();
        assert!((decoded.phi - 0.72).abs() < f32::EPSILON);
        assert_eq!(decoded.cycle, 12345);
        assert!(!decoded.is_keyframe);
        assert_eq!(decoded.compressed_data.len(), 6);
    }

    #[test]
    fn test_threat_signature_serialization() {
        let threat = ThreatSignatureRelay {
            threat_type: 3,
            severity: 0.85,
            agent_hash: [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE],
            signature: [0x42; 32],
            observation_count: 7,
        };
        let bytes = bincode::serialize(&threat).unwrap();
        let decoded: ThreatSignatureRelay = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.threat_type, 3);
        assert!((decoded.severity - 0.85).abs() < f32::EPSILON);
        assert_eq!(decoded.observation_count, 7);
        assert_eq!(decoded.agent_hash[0], 0xDE);
    }

    #[test]
    fn test_resource_forecast_serialization() {
        let forecast = ResourceForecastRelay {
            resource_type: "water".into(),
            horizon_hours: 24,
            predicted_rate: 12.5,
            confidence: 0.82,
        };
        let bytes = bincode::serialize(&forecast).unwrap();
        let decoded: ResourceForecastRelay = bincode::deserialize(&bytes).unwrap();
        assert_eq!(decoded.horizon_hours, 24);
        assert!((decoded.confidence - 0.82).abs() < f32::EPSILON);
    }

    #[test]
    fn test_consciousness_delta_fits_lora() {
        // A typical compressed delta should fit in a single LoRa/Meshtastic frame.
        // Meshtastic max payload ~200 bytes. Delta header ~21 bytes + compressed data.
        let delta = ConsciousnessDeltaRelay {
            phi: 0.5,
            consciousness_level: 0.7,
            valence: 0.0,
            arousal: 0.5,
            cycle: 99999,
            is_keyframe: false,
            compressed_data: vec![0u8; 150], // typical compressed HV delta
        };
        let bytes = bincode::serialize(&delta).unwrap();
        let payload = RelayPayload::new(RelayType::ConsciousnessDelta, [0; 8], bytes);
        let wire = payload.to_bytes();
        assert!(
            wire.len() < 255,
            "consciousness delta payload {} bytes should fit single LoRa frame",
            wire.len()
        );
    }

    #[test]
    fn test_out_of_order_reassembly() {
        let data = b"This payload should survive out-of-order fragment delivery over mesh";
        let mut frames = fragment(data, 20);
        assert!(frames.len() > 3, "need multiple fragments");

        // Reverse the frame order
        frames.reverse();
        let reassembled = reassemble(&frames).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn test_e2e_new_relay_types_roundtrip() {
        let origin = [11, 22, 33, 44, 55, 66, 77, 88];

        // Water alert
        let water = WaterAlertRelay {
            system_id: "rainwater-tank-3".into(),
            alert_type: "Contamination".into(),
            severity: "High".into(),
            description: "E. coli detected in sector 7 tank".into(),
        };
        let water_data = bincode::serialize(&water).unwrap();
        let water_payload = RelayPayload::new(RelayType::WaterAlert, origin, water_data);
        let decoded = RelayPayload::from_bytes(&water_payload.to_bytes()).unwrap();
        assert_eq!(decoded.relay_type, RelayType::WaterAlert);
        let inner: WaterAlertRelay = bincode::deserialize(&decoded.data).unwrap();
        assert_eq!(inner.system_id, "rainwater-tank-3");

        // Knowledge claim
        let knowledge = KnowledgeClaimRelay {
            claim_text: "Moringa leaves can purify water".into(),
            tags: vec!["water".into(), "purification".into()],
            empirical: 2, normative: 1, materiality: 2,
        };
        let knowledge_data = bincode::serialize(&knowledge).unwrap();
        let knowledge_payload = RelayPayload::new(RelayType::KnowledgeClaim, origin, knowledge_data);
        let decoded = RelayPayload::from_bytes(&knowledge_payload.to_bytes()).unwrap();
        assert_eq!(decoded.relay_type, RelayType::KnowledgeClaim);
        let inner: KnowledgeClaimRelay = bincode::deserialize(&decoded.data).unwrap();
        assert_eq!(inner.tags.len(), 2);

        // Supply update
        let supply = SupplyRelay {
            item_id: "med-001".into(), item_name: "First aid kits".into(),
            quantity: 8.0, category: "Medical".into(),
        };
        let supply_data = bincode::serialize(&supply).unwrap();
        let supply_payload = RelayPayload::new(RelayType::SupplyUpdate, origin, supply_data);
        let decoded = RelayPayload::from_bytes(&supply_payload.to_bytes()).unwrap();
        assert_eq!(decoded.relay_type, RelayType::SupplyUpdate);
        let inner: SupplyRelay = bincode::deserialize(&decoded.data).unwrap();
        assert_eq!(inner.item_name, "First aid kits");

        // Price report
        let price = PriceReportRelay {
            item_name: "Bread (700g)".into(), price_tend: 0.18,
            evidence: "Shoprite Florida, 2026-03-15".into(),
        };
        let price_data = bincode::serialize(&price).unwrap();
        let price_payload = RelayPayload::new(RelayType::PriceReport, origin, price_data);
        let decoded = RelayPayload::from_bytes(&price_payload.to_bytes()).unwrap();
        assert_eq!(decoded.relay_type, RelayType::PriceReport);
        let inner: PriceReportRelay = bincode::deserialize(&decoded.data).unwrap();
        assert!((inner.price_tend - 0.18).abs() < f32::EPSILON);
    }
}

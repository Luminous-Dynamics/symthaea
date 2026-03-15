//! Integration tests — full mesh bridge pipeline via LoopbackTransport.
//!
//! Verifies: serialize → fragment → send → recv → reassemble → deserialize
//! for all payload types, simulating the poller→transport→relay path.

use mycelix_mesh_bridge::serializer::{
    self, EmergencyRelay, FoodRelay, RelayPayload, RelayType, TendRelay,
};
use mycelix_mesh_bridge::transport::{LoopbackTransport, MeshTransport};

/// LoRa frame size limit.
const LORA_MAX_FRAME: usize = 255;

/// Helper: send a payload through the full pipeline, return the decoded payload.
async fn roundtrip(transport: &LoopbackTransport, payload: &RelayPayload) -> RelayPayload {
    let bytes = payload.to_bytes();
    let frames = serializer::fragment(&bytes, LORA_MAX_FRAME);

    // Send all frames
    for frame in &frames {
        transport.send(frame).await.unwrap();
    }

    // Receive all frames
    let mut received = Vec::new();
    for _ in 0..frames.len() {
        let frame = transport.recv(1000).await.unwrap().expect("expected frame");
        received.push(frame);
    }

    // Reassemble and decode
    let reassembled = serializer::reassemble(&received).expect("reassembly failed");
    RelayPayload::from_bytes(&reassembled).expect("deserialization failed")
}

#[tokio::test]
async fn test_tend_exchange_pipeline() {
    let transport = LoopbackTransport::new();
    let origin = [11, 22, 33, 44, 55, 66, 77, 88];

    let tend = TendRelay {
        receiver_did: "nomsa.did".into(),
        hours: 2.0,
        service_description: "Fixed leaking roof".into(),
        service_category: "Construction".into(),
        dao_did: "roodepoort-resilience".into(),
    };
    let data = bincode::serialize(&tend).unwrap();
    let payload = RelayPayload::new(RelayType::TendExchange, origin, data);

    let decoded = roundtrip(&transport, &payload).await;
    assert_eq!(decoded.relay_type, RelayType::TendExchange);
    assert_eq!(decoded.origin, origin);
    assert_eq!(decoded.content_hash, payload.content_hash);

    let inner: TendRelay = bincode::deserialize(&decoded.data).unwrap();
    assert_eq!(inner.receiver_did, "nomsa.did");
    assert!((inner.hours - 2.0).abs() < f32::EPSILON);
    assert_eq!(inner.service_category, "Construction");
    assert_eq!(inner.dao_did, "roodepoort-resilience");
}

#[tokio::test]
async fn test_food_harvest_pipeline() {
    let transport = LoopbackTransport::new();
    let origin = [1; 8];

    let food = FoodRelay {
        crop_hash: "plot-a3-tomato".into(),
        quantity_kg: 8.75,
        quality: "Excellent".into(),
        notes: "First harvest of the season".into(),
    };
    let data = bincode::serialize(&food).unwrap();
    let payload = RelayPayload::new(RelayType::FoodHarvest, origin, data);

    let decoded = roundtrip(&transport, &payload).await;
    assert_eq!(decoded.relay_type, RelayType::FoodHarvest);

    let inner: FoodRelay = bincode::deserialize(&decoded.data).unwrap();
    assert_eq!(inner.crop_hash, "plot-a3-tomato");
    assert!((inner.quantity_kg - 8.75).abs() < f32::EPSILON);
    assert_eq!(inner.quality, "Excellent");
}

#[tokio::test]
async fn test_emergency_message_pipeline() {
    let transport = LoopbackTransport::new();
    let origin = [255; 8];

    let emergency = EmergencyRelay {
        channel_id: "ward-7-main".into(),
        content: "Load shedding Stage 6 expected 18:00-22:00. Charge devices now.".into(),
        priority: "Flash".into(),
    };
    let data = bincode::serialize(&emergency).unwrap();
    let payload = RelayPayload::new(RelayType::EmergencyMessage, origin, data);

    let decoded = roundtrip(&transport, &payload).await;
    assert_eq!(decoded.relay_type, RelayType::EmergencyMessage);

    let inner: EmergencyRelay = bincode::deserialize(&decoded.data).unwrap();
    assert_eq!(inner.channel_id, "ward-7-main");
    assert_eq!(inner.priority, "Flash");
    assert!(inner.content.contains("Stage 6"));
}

#[tokio::test]
async fn test_heartbeat_pipeline() {
    let transport = LoopbackTransport::new();
    let origin = [42; 8];

    let payload = RelayPayload::new(RelayType::Heartbeat, origin, Vec::new());

    let decoded = roundtrip(&transport, &payload).await;
    assert_eq!(decoded.relay_type, RelayType::Heartbeat);
    assert_eq!(decoded.origin, origin);
    assert!(decoded.data.is_empty());
}

#[tokio::test]
async fn test_multiple_payloads_interleaved() {
    let transport = LoopbackTransport::new();
    let origin = [7; 8];

    // Send TEND, then emergency — verify both arrive correctly
    let tend = TendRelay {
        receiver_did: "thabo.did".into(),
        hours: 0.5,
        service_description: "Tutoring".into(),
        service_category: "Education".into(),
        dao_did: "roodepoort".into(),
    };
    let tend_data = bincode::serialize(&tend).unwrap();
    let tend_payload = RelayPayload::new(RelayType::TendExchange, origin, tend_data);

    let emerg = EmergencyRelay {
        channel_id: "general".into(),
        content: "Community meeting at 15:00".into(),
        priority: "Routine".into(),
    };
    let emerg_data = bincode::serialize(&emerg).unwrap();
    let emerg_payload = RelayPayload::new(RelayType::EmergencyMessage, origin, emerg_data);

    // Send both
    let tend_decoded = roundtrip(&transport, &tend_payload).await;
    let emerg_decoded = roundtrip(&transport, &emerg_payload).await;

    assert_eq!(tend_decoded.relay_type, RelayType::TendExchange);
    assert_eq!(emerg_decoded.relay_type, RelayType::EmergencyMessage);

    let tend_inner: TendRelay = bincode::deserialize(&tend_decoded.data).unwrap();
    assert_eq!(tend_inner.receiver_did, "thabo.did");

    let emerg_inner: EmergencyRelay = bincode::deserialize(&emerg_decoded.data).unwrap();
    assert_eq!(emerg_inner.content, "Community meeting at 15:00");
}

#[tokio::test]
async fn test_dedup_across_payloads() {
    // Same data → same content_hash → relay should dedup
    let data = b"identical-payload".to_vec();
    let p1 = RelayPayload::new(RelayType::TendExchange, [1; 8], data.clone());
    let p2 = RelayPayload::new(RelayType::TendExchange, [1; 8], data);

    // Content hashes match (same data → same BLAKE3)
    assert_eq!(p1.content_hash, p2.content_hash);

    // Different data → different hashes
    let p3 = RelayPayload::new(RelayType::TendExchange, [1; 8], b"different".to_vec());
    assert_ne!(p1.content_hash, p3.content_hash);
}

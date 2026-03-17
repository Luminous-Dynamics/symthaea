//! Stress & adversarial property tests for the mesh bridge.
//!
//! Validates resilience against malformed frames, duplicate storms,
//! oversized payloads, and hostile input patterns.

use mycelix_mesh_bridge::serializer::{
    self, EmergencyRelay, FoodRelay, RelayPayload, RelayType, TendRelay,
};
use mycelix_mesh_bridge::transport::{LoopbackTransport, MeshTransport};
use proptest::prelude::*;

// =============================================================================
// Adversarial fragmentation
// =============================================================================

proptest! {
    /// Truncated frames must never panic — reassemble returns None.
    #[test]
    fn prop_truncated_frames_never_panic(
        data in proptest::collection::vec(any::<u8>(), 1..512),
        frame_size in 16usize..255,
        truncate_at in 0usize..8,
    ) {
        let frames = serializer::fragment(&data, frame_size);
        if frames.is_empty() {
            return Ok(());
        }

        // Truncate the first frame at an arbitrary point
        let mut damaged = frames.clone();
        if !damaged[0].is_empty() {
            let cut = truncate_at.min(damaged[0].len().saturating_sub(1));
            damaged[0].truncate(cut);
        }

        // Must not panic — may return None or Some
        let _ = serializer::reassemble(&damaged);
    }

    /// Garbage bytes never cause panics in reassemble.
    #[test]
    fn prop_garbage_frames_never_panic(
        garbage in proptest::collection::vec(
            proptest::collection::vec(any::<u8>(), 0..300),
            1..20,
        ),
    ) {
        // Feed random bytes — must not panic
        let _ = serializer::reassemble(&garbage);
    }

    /// Duplicate frames don't cause panics — reassembly may succeed or fail
    /// depending on how duplicates interact with the index parsing.
    #[test]
    fn prop_duplicate_frames_no_panic(
        data in proptest::collection::vec(any::<u8>(), 10..256),
        frame_size in 16usize..255,
        dup_count in 1usize..5,
    ) {
        let frames = serializer::fragment(&data, frame_size);
        if frames.is_empty() {
            return Ok(());
        }

        // Duplicate every frame N times
        let mut with_dups: Vec<Vec<u8>> = Vec::new();
        for frame in &frames {
            for _ in 0..=dup_count {
                with_dups.push(frame.clone());
            }
        }

        // Must not panic — result may be Some or None
        let result = serializer::reassemble(&with_dups);
        // If reassembly succeeds, it must produce the original data
        if let Some(reassembled) = result {
            prop_assert_eq!(reassembled, data);
        }
    }

    /// Shuffled frames (without duplication) still reassemble correctly.
    #[test]
    fn prop_shuffled_frames_reassemble(
        data in proptest::collection::vec(any::<u8>(), 10..256),
        frame_size in 16usize..255,
        seed in any::<u64>(),
    ) {
        let frames = serializer::fragment(&data, frame_size);
        if frames.is_empty() {
            return Ok(());
        }

        // Shuffle using deterministic seed (no duplicates)
        let mut shuffled = frames.clone();
        let mut rng_state = seed;
        for i in (1..shuffled.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            shuffled.swap(i, j);
        }

        let result = serializer::reassemble(&shuffled);
        prop_assert!(result.is_some(), "shuffled frames broke reassembly");
        prop_assert_eq!(result.unwrap(), data);
    }

    /// Oversized payloads fragment correctly and roundtrip.
    #[test]
    fn prop_oversized_payload_roundtrip(
        data in proptest::collection::vec(any::<u8>(), 1000..4096),
    ) {
        let payload = RelayPayload::new(RelayType::TendExchange, [1; 8], data.clone());
        let bytes = payload.to_bytes();

        // Fragment at LoRa max (255 bytes)
        let frames = serializer::fragment(&bytes, 255);
        prop_assert!(frames.len() > 1, "large payload should produce multiple frames");

        let reassembled = serializer::reassemble(&frames);
        prop_assert!(reassembled.is_some());
        let decoded = RelayPayload::from_bytes(&reassembled.unwrap());
        prop_assert!(decoded.is_ok());
        prop_assert_eq!(decoded.unwrap().data, data);
    }

    /// Empty data field is valid (heartbeat pattern).
    #[test]
    fn prop_empty_data_roundtrip(
        relay_type in prop_oneof![
            Just(RelayType::Heartbeat),
            Just(RelayType::TendExchange),
        ],
        origin in proptest::array::uniform8(0u8..),
    ) {
        let payload = RelayPayload::new(relay_type, origin, Vec::new());
        let bytes = payload.to_bytes();
        let frames = serializer::fragment(&bytes, 255);
        let reassembled = serializer::reassemble(&frames).unwrap();
        let decoded = RelayPayload::from_bytes(&reassembled).unwrap();
        prop_assert!(decoded.data.is_empty());
        prop_assert_eq!(decoded.relay_type, relay_type);
    }
}

// =============================================================================
// Adversarial deserialization
// =============================================================================

proptest! {
    /// Arbitrary bytes fed to from_bytes never panic — only Ok or Err.
    #[test]
    fn prop_arbitrary_bytes_safe_deserialize(
        bytes in proptest::collection::vec(any::<u8>(), 0..1024),
    ) {
        // Must not panic
        let _ = RelayPayload::from_bytes(&bytes);
    }

    /// Arbitrary bytes fed to TendRelay deserialization never panic.
    #[test]
    fn prop_arbitrary_tend_deserialize(
        bytes in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let _ = bincode::deserialize::<TendRelay>(&bytes);
    }

    /// Arbitrary bytes fed to EmergencyRelay deserialization never panic.
    #[test]
    fn prop_arbitrary_emergency_deserialize(
        bytes in proptest::collection::vec(any::<u8>(), 0..512),
    ) {
        let _ = bincode::deserialize::<EmergencyRelay>(&bytes);
    }
}

// =============================================================================
// Content hash integrity
// =============================================================================

proptest! {
    /// Different data always produces different content hashes.
    #[test]
    fn prop_content_hash_collision_resistance(
        data1 in proptest::collection::vec(any::<u8>(), 1..256),
        data2 in proptest::collection::vec(any::<u8>(), 1..256),
    ) {
        if data1 == data2 {
            return Ok(());
        }
        let p1 = RelayPayload::new(RelayType::TendExchange, [0; 8], data1);
        let p2 = RelayPayload::new(RelayType::TendExchange, [0; 8], data2);
        prop_assert_ne!(p1.content_hash, p2.content_hash);
    }

    /// Same data always produces same content hash (deterministic).
    #[test]
    fn prop_content_hash_deterministic(
        data in proptest::collection::vec(any::<u8>(), 0..256),
    ) {
        let p1 = RelayPayload::new(RelayType::FoodHarvest, [0; 8], data.clone());
        let p2 = RelayPayload::new(RelayType::FoodHarvest, [0; 8], data);
        prop_assert_eq!(p1.content_hash, p2.content_hash);
    }
}

// =============================================================================
// Loopback transport stress
// =============================================================================

#[tokio::test]
async fn test_loopback_high_throughput() {
    let transport = LoopbackTransport::new();

    // Send 1000 frames rapidly
    for i in 0u32..1000 {
        transport.send(&i.to_le_bytes()).await.unwrap();
    }

    // Receive all 1000
    for i in 0u32..1000 {
        let frame = transport.recv(100).await.unwrap().expect("missing frame");
        assert_eq!(frame, i.to_le_bytes());
    }

    // Buffer should be empty now
    let none = transport.recv(10).await.unwrap();
    assert!(none.is_none());
}

#[tokio::test]
async fn test_loopback_concurrent_send_recv() {
    let transport = LoopbackTransport::new();
    let send_transport = transport.clone_box();

    let sender = tokio::spawn(async move {
        for i in 0u32..100 {
            send_transport.send(&i.to_le_bytes()).await.unwrap();
            tokio::task::yield_now().await;
        }
    });

    let mut received = Vec::new();
    for _ in 0..100 {
        loop {
            match transport.recv(500).await.unwrap() {
                Some(frame) => {
                    received.push(u32::from_le_bytes(frame.try_into().unwrap()));
                    break;
                }
                None => continue,
            }
        }
    }

    sender.await.unwrap();
    assert_eq!(received.len(), 100);
    // Should be in order (FIFO)
    for (i, val) in received.iter().enumerate() {
        assert_eq!(*val, i as u32);
    }
}

// =============================================================================
// Timestamp validation
// =============================================================================

#[test]
fn test_payload_timestamp_reasonable() {
    let payload = RelayPayload::new(RelayType::Heartbeat, [0; 8], vec![]);

    // Timestamp should be recent (within last 10 seconds)
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    assert!(payload.timestamp <= now);
    assert!(payload.timestamp >= now - 10);
}

// =============================================================================
// Max payload size bounds
// =============================================================================

#[test]
fn test_max_relay_types_all_serializable() {
    let types = [
        RelayType::TendExchange,
        RelayType::FoodHarvest,
        RelayType::EmergencyMessage,
        RelayType::WaterAlert,
        RelayType::HearthAlert,
        RelayType::KnowledgeClaim,
        RelayType::CareCircleUpdate,
        RelayType::ShelterUpdate,
        RelayType::SupplyUpdate,
        RelayType::MutualAidOffer,
        RelayType::PriceReport,
        RelayType::Heartbeat,
    ];

    for rt in types {
        let payload = RelayPayload::new(rt, [42; 8], vec![1, 2, 3]);
        let bytes = payload.to_bytes();
        assert!(!bytes.is_empty(), "failed to serialize {rt:?}");
        let decoded = RelayPayload::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.relay_type, rt);
    }
}

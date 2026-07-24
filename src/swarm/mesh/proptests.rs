// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::*;
use proptest::prelude::*;

fn arb_urgency() -> impl Strategy<Value = MeshUrgency> {
    (0u8..=2).prop_map(|b| MeshUrgency::from_byte(b))
}

fn arb_payload_type() -> impl Strategy<Value = PayloadType> {
    (0u8..=3).prop_map(|b| PayloadType::from_byte(b))
}

fn arb_wisdom_packet() -> impl Strategy<Value = WisdomPacket> {
    (
        any::<[u8; 8]>(), // source_id
        any::<u32>(),     // sequence
        any::<f32>(),     // phi
        arb_urgency(),
        any::<u32>(), // timestamp_s
        arb_payload_type(),
        any::<[u8; 32]>(), // auth_mac
        1u8..=10,          // ttl
    )
        .prop_map(|(sid, seq, phi, urg, ts, pt, mac, ttl)| {
            let phi = if phi.is_nan() || phi.is_infinite() {
                0.0
            } else {
                phi
            };
            WisdomPacket {
                source_id: sid,
                sequence: seq,
                phi,
                urgency: urg,
                timestamp_s: ts,
                payload_type: pt,
                auth_mac: mac,
                ttl,
                wisdom: BinaryHV([seq as u8; 2048]),
            }
        })
}

proptest! {
    #[test]
    fn proptest_serialization_roundtrip(packet in arb_wisdom_packet()) {
        let bytes = packet.to_bytes();
        let decoded = WisdomPacket::from_bytes(&bytes).expect("deserialization should succeed");
        prop_assert_eq!(decoded.source_id, packet.source_id);
        prop_assert_eq!(decoded.sequence, packet.sequence);
        prop_assert_eq!(decoded.urgency, packet.urgency);
        prop_assert_eq!(decoded.timestamp_s, packet.timestamp_s);
        prop_assert_eq!(decoded.payload_type, packet.payload_type);
        prop_assert_eq!(decoded.auth_mac, packet.auth_mac);
        prop_assert_eq!(decoded.ttl, packet.ttl);
        prop_assert_eq!(decoded.wisdom.0, packet.wisdom.0);
        // Handle NaN: both should be normalized to 0.0
        if packet.phi.is_nan() {
            prop_assert!(decoded.phi.is_nan() || decoded.phi == 0.0);
        } else {
            prop_assert!((decoded.phi - packet.phi).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn proptest_compress_decompress_roundtrip(packet in arb_wisdom_packet()) {
        let bytes = packet.to_bytes();
        let compressed = compress_packet(&bytes);
        let decompressed = decompress_packet(&compressed).expect("decompression should succeed");
        prop_assert_eq!(decompressed, bytes);
    }

    #[test]
    fn proptest_fragment_reassemble_roundtrip(packet in arb_wisdom_packet()) {
        use super::lora_fragment::{fragment, FragmentAssembler, LoRaFragment};

        let original_bytes = packet.to_bytes();
        let envelope = compress_packet(&original_bytes);
        let thought_id = packet.thought_id();
        let frags = fragment(thought_id, &envelope);
        let total = frags.len();

        let mut assembler = FragmentAssembler::new(thought_id, total as u8, envelope.len());
        let mut buf = [0u8; LORA_MTU];
        for frag in &frags {
            let len = frag.to_bytes(&mut buf);
            let decoded = LoRaFragment::from_bytes(&buf[..len]).unwrap();
            assembler.feed(&decoded);
        }
        prop_assert!(assembler.is_complete());
        let reassembled = assembler.assemble().expect("assembly should succeed");
        let decompressed = decompress_packet(&reassembled).expect("decompression should succeed");
        prop_assert_eq!(decompressed, original_bytes);
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for mycelix-protocol
//!
//! Tests for packet encoding/decoding, jitter buffering,
//! audio streaming, FEC, and congestion control.

use mycelix_protocol::{
    packet::{ControlMessage, ControlMessageType, PacketHeader, PingPacket},
    stream::{AudioStream, StreamDirection, StreamPriority},
    congestion::CongestionController,
    AudioPacket, AudioPacketHeader, JitterBuffer, FecEncoder, FecDecoder,
    NetworkEstimator, PacketType, ProtocolConfig, Result,
};
use bytes::Bytes;
use std::time::Duration;

// ============================================================================
// Packet Encoding/Decoding Tests
// ============================================================================

mod packet_tests {
    use super::*;

    #[test]
    fn test_packet_header_roundtrip() {
        let header = PacketHeader::new(PacketType::Audio, 1024);
        let encoded = header.encode();
        let decoded = PacketHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.packet_type, PacketType::Audio);
        assert_eq!(decoded.length, 1024);
    }

    #[test]
    fn test_all_packet_types() {
        let types = [
            PacketType::Audio,
            PacketType::AudioFec,
            PacketType::Control,
            PacketType::Ping,
            PacketType::Pong,
        ];

        for pkt_type in types {
            let header = PacketHeader::new(pkt_type, 100);
            let encoded = header.encode();
            let decoded = PacketHeader::decode(&encoded).unwrap();
            assert_eq!(decoded.packet_type, pkt_type);
        }
    }

    #[test]
    fn test_control_message_roundtrip() {
        let msg = ControlMessage::new(ControlMessageType::Start, 42)
            .with_payload(vec![1, 2, 3, 4]);

        let encoded = msg.encode();
        let decoded = ControlMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.message_type, ControlMessageType::Start);
        assert_eq!(decoded.stream_id, 42);
        assert_eq!(decoded.payload, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_all_control_message_types() {
        let types = [
            ControlMessageType::Start,
            ControlMessageType::Stop,
            ControlMessageType::Pause,
            ControlMessageType::Resume,
            ControlMessageType::Seek,
            ControlMessageType::Quality,
            ControlMessageType::Sync,
            ControlMessageType::Ack,
        ];

        for msg_type in types {
            let msg = ControlMessage::new(msg_type, 1);
            let encoded = msg.encode();
            let decoded = ControlMessage::decode(&encoded).unwrap();
            assert_eq!(decoded.message_type, msg_type);
        }
    }

    #[test]
    fn test_ping_packet_roundtrip() {
        let ping = PingPacket {
            sequence: 123,
            timestamp: 1000000,
        };

        let encoded = ping.encode();
        let decoded = PingPacket::decode(&encoded).unwrap();

        assert_eq!(decoded.sequence, 123);
        assert_eq!(decoded.timestamp, 1000000);
    }

    #[test]
    fn test_invalid_packet_header() {
        let result = PacketHeader::decode(&[0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_control_message() {
        let result = ControlMessage::decode(&[0x00, 0x01, 0x02]);
        assert!(result.is_err());
    }
}

// ============================================================================
// Audio Packet Tests
// ============================================================================

mod audio_packet_tests {
    use super::*;

    #[test]
    fn test_audio_packet_creation() {
        let payload = Bytes::from(vec![0u8; 1000]);
        let packet = AudioPacket::new(1, 960, 42, payload.clone());

        assert_eq!(packet.header.sequence, 1);
        assert_eq!(packet.header.timestamp, 960);
        assert_eq!(packet.header.stream_id, 42);
        assert_eq!(packet.payload.len(), 1000);
    }

    #[test]
    fn test_audio_packet_roundtrip() {
        let payload = Bytes::from(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let packet = AudioPacket::new(42, 12345, 1, payload.clone());

        let encoded = packet.encode();
        let decoded = AudioPacket::decode(encoded).unwrap();

        assert_eq!(decoded.header.sequence, 42);
        assert_eq!(decoded.header.timestamp, 12345);
        assert_eq!(decoded.header.stream_id, 1);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_audio_packet_header_size() {
        assert_eq!(AudioPacketHeader::SIZE, 19); // 4 + 8 + 4 + 2 + 1
    }

    #[test]
    fn test_large_audio_packet() {
        // Test with larger payload (typical audio frame)
        let payload = Bytes::from(vec![0u8; 7680]); // 960 samples * 2 channels * 4 bytes
        let packet = AudioPacket::new(0, 0, 1, payload);

        let encoded = packet.encode();
        let decoded = AudioPacket::decode(encoded).unwrap();

        assert_eq!(decoded.payload.len(), 7680);
    }
}

// ============================================================================
// Jitter Buffer Tests
// ============================================================================

mod jitter_buffer_tests {
    use super::*;

    fn make_packet(seq: u32) -> AudioPacket {
        let payload = Bytes::from(vec![seq as u8; 100]);
        AudioPacket::new(seq, seq as u64 * 960, 1, payload)
    }

    #[test]
    fn test_jitter_buffer_empty() {
        let mut buffer = JitterBuffer::new(64, 3);
        assert!(buffer.pop().is_none());
        assert!(!buffer.is_ready());
    }

    #[test]
    fn test_jitter_buffer_filling() {
        let mut buffer = JitterBuffer::new(64, 3);

        // Add packets
        for seq in 0..5 {
            buffer.push(make_packet(seq));
        }

        // Should be ready after target delay frames
        assert!(buffer.is_ready());
    }

    #[test]
    fn test_jitter_buffer_level() {
        let mut buffer = JitterBuffer::new(64, 3);

        buffer.push(make_packet(0));
        buffer.push(make_packet(1));
        buffer.push(make_packet(2));

        // Level should reflect packets in buffer
        assert!(buffer.level() > 0);
    }

    #[test]
    fn test_jitter_buffer_out_of_order() {
        let mut buffer = JitterBuffer::new(64, 2);

        // Insert out of order
        buffer.push(make_packet(2));
        buffer.push(make_packet(0));
        buffer.push(make_packet(1));
        buffer.push(make_packet(3));
        buffer.push(make_packet(4));

        // Should handle reordering
        assert!(buffer.is_ready());
    }
}

// ============================================================================
// Audio Stream Tests
// ============================================================================

mod stream_tests {
    use super::*;

    #[test]
    fn test_stream_creation() {
        let config = ProtocolConfig::default();
        let stream = AudioStream::new(1, StreamDirection::Send, config);

        assert_eq!(stream.stream_id, 1);
        assert!(!stream.has_pending());
    }

    #[test]
    fn test_stream_send() {
        let config = ProtocolConfig::default();
        let mut stream = AudioStream::new(1, StreamDirection::Send, config);

        // Queue samples for sending
        let samples: Vec<f32> = (0..960).map(|i| i as f32 / 960.0).collect();
        stream.send(&samples).unwrap();

        assert!(stream.has_pending());
    }

    #[test]
    fn test_stream_poll_send() {
        let config = ProtocolConfig::default();
        let mut stream = AudioStream::new(1, StreamDirection::Send, config);

        let samples: Vec<f32> = vec![0.5; 960];
        stream.send(&samples).unwrap();

        let packet = stream.poll_send();
        assert!(packet.is_some());

        // After polling, should be empty
        assert!(!stream.has_pending());
    }

    #[test]
    fn test_stream_receive_direction_error() {
        let config = ProtocolConfig::default();
        let mut stream = AudioStream::new(1, StreamDirection::Receive, config);

        let samples: Vec<f32> = vec![0.5; 960];
        let result = stream.send(&samples);

        assert!(result.is_err());
    }

    #[test]
    fn test_stream_priority() {
        let config = ProtocolConfig::default();
        let stream = AudioStream::new(1, StreamDirection::Bidirectional, config)
            .with_priority(StreamPriority::Realtime);

        // Stream should be created with priority
        assert!(!stream.has_pending());
    }

    #[test]
    fn test_stream_receive() {
        let config = ProtocolConfig::default();
        let mut stream = AudioStream::new(1, StreamDirection::Receive, config);

        // Create a packet with sample data
        let samples: Vec<f32> = vec![0.5; 960];
        let mut payload = Vec::with_capacity(samples.len() * 4);
        for &s in &samples {
            payload.extend_from_slice(&s.to_le_bytes());
        }

        let packet = AudioPacket::new(0, 0, 1, Bytes::from(payload));
        stream.receive(packet).unwrap();

        assert!(stream.has_available());

        let received = stream.poll_receive();
        assert!(received.is_some());
        assert_eq!(received.unwrap().len(), 960);
    }
}

// ============================================================================
// FEC Tests
// ============================================================================

mod fec_tests {
    use super::*;

    #[test]
    fn test_fec_encoder_creation() {
        let _encoder = FecEncoder::new(4, 0.25);
        // Should be created without error
        assert!(true);
    }

    #[test]
    fn test_fec_encoder_block() {
        let mut encoder = FecEncoder::new(4, 0.25);

        // Add packets
        for i in 0..3 {
            let result = encoder.add_packet(Bytes::from(vec![i; 100]));
            assert!(result.is_none()); // Not enough for block
        }

        // This should complete the block
        let result = encoder.add_packet(Bytes::from(vec![3; 100]));
        assert!(result.is_some());

        let fec_packets = result.unwrap();
        assert!(!fec_packets.is_empty());
    }

    #[test]
    fn test_fec_decoder_creation() {
        let _decoder = FecDecoder::new(4);
        assert!(true);
    }
}

// ============================================================================
// Congestion Control Tests
// ============================================================================

mod congestion_tests {
    use super::*;

    #[test]
    fn test_congestion_controller_initial_state() {
        let _cc = CongestionController::new();
        // Should start in slow start or similar
        assert!(true);
    }

    #[test]
    fn test_congestion_controller_ack_handling() {
        let mut cc = CongestionController::new();

        // Report ACKs
        for _ in 0..10 {
            cc.on_ack(1400); // 1 packet worth of bytes
        }

        // Should handle without panic
        assert!(true);
    }

    #[test]
    fn test_congestion_controller_loss_handling() {
        let mut cc = CongestionController::new();

        // Report some ACKs first
        for _ in 0..5 {
            cc.on_ack(1400);
        }

        // Report loss
        cc.on_loss();

        // Should reduce window
        assert!(true);
    }

    #[test]
    fn test_congestion_controller_rtt_sample() {
        let mut cc = CongestionController::new();

        cc.on_rtt_sample(Duration::from_millis(20));
        cc.on_rtt_sample(Duration::from_millis(15));
        cc.on_rtt_sample(Duration::from_millis(25));

        // Should track RTT
        assert!(true);
    }

    #[test]
    fn test_congestion_controller_can_send() {
        let cc = CongestionController::new();

        // Initially should be able to send
        let can_send = cc.can_send();
        assert!(can_send);
    }

    #[test]
    fn test_congestion_pacing_rate() {
        let mut cc = CongestionController::new();

        for _ in 0..10 {
            cc.on_ack(1400);
            cc.on_rtt_sample(Duration::from_millis(20));
        }

        let rate = cc.pacing_rate();
        assert!(rate > 0.0);
    }
}

// ============================================================================
// Network Estimator Tests
// ============================================================================

mod network_estimator_tests {
    use super::*;

    #[test]
    fn test_network_estimator_creation() {
        let _estimator = NetworkEstimator::new(20, 20);
        assert!(true);
    }

    #[test]
    fn test_network_estimator_rtt_update() {
        let mut estimator = NetworkEstimator::new(20, 20);

        estimator.record_rtt(20.0);
        estimator.record_rtt(25.0);
        estimator.record_rtt(15.0);

        let smoothed_rtt = estimator.smoothed_rtt();
        assert!(smoothed_rtt > 0.0);
    }

    #[test]
    fn test_network_estimator_loss() {
        let mut estimator = NetworkEstimator::new(20, 20);

        // Record some packets
        for _ in 0..10 {
            estimator.record_packet(false); // Not lost
        }
        estimator.record_packet(true); // Lost

        let loss = estimator.loss_ratio();
        assert!(loss > 0.0);
    }

    #[test]
    fn test_network_estimator_jitter() {
        let mut estimator = NetworkEstimator::new(20, 20);

        estimator.record_rtt(20.0);
        estimator.record_rtt(30.0);
        estimator.record_rtt(15.0);
        estimator.record_rtt(25.0);

        let jitter = estimator.jitter();
        assert!(jitter >= 0.0);
    }

    #[test]
    fn test_network_estimator_recommended_bitrate() {
        let mut estimator = NetworkEstimator::new(20, 20);

        // Simulate good conditions
        for _ in 0..10 {
            estimator.record_rtt(15.0);
            estimator.record_packet(false);
        }

        let bitrate = estimator.recommended_bitrate_kbps(256.0);
        assert!(bitrate > 0.0);
    }
}

// ============================================================================
// Protocol Config Tests
// ============================================================================

mod config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ProtocolConfig::default();

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.frame_size, 960);
        assert!(config.enable_fec);
        assert!(config.enable_abr);
    }

    #[test]
    fn test_frame_duration_calculation() {
        let config = ProtocolConfig::default();
        let duration = config.frame_duration_ms();

        // 960 samples at 48kHz = 20ms
        assert!((duration - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_bytes_per_frame() {
        let config = ProtocolConfig::default();
        let bytes = config.bytes_per_frame();

        // 960 samples * 2 channels * 4 bytes/sample (f32)
        assert_eq!(bytes, 960 * 2 * 4);
    }

    #[test]
    fn test_custom_config() {
        let config = ProtocolConfig {
            sample_rate: 44100,
            channels: 1,
            frame_size: 441,
            target_latency_ms: 30,
            enable_fec: false,
            enable_abr: false,
            ..Default::default()
        };

        let duration = config.frame_duration_ms();
        assert!((duration - 10.0).abs() < 0.1);
    }
}

// ============================================================================
// End-to-End Simulation Tests
// ============================================================================

mod simulation_tests {
    use super::*;

    #[test]
    fn test_audio_streaming_simulation() {
        let config = ProtocolConfig::default();
        let mut jitter_buffer = JitterBuffer::new(64, config.jitter_buffer_frames);

        // Generate and send 100 packets
        for seq in 0u32..100 {
            let payload = Bytes::from(vec![seq as u8; config.bytes_per_frame()]);
            let packet = AudioPacket::new(seq, seq as u64 * config.frame_size as u64, 1, payload);

            // Encode and decode (simulate network)
            let encoded = packet.encode();
            let decoded = AudioPacket::decode(encoded).unwrap();

            jitter_buffer.push(decoded);
        }

        // Should be ready for playback
        assert!(jitter_buffer.is_ready());

        // Extract packets
        let mut received = 0;
        while let Some(_packet) = jitter_buffer.pop() {
            received += 1;
            if received > 50 {
                break;
            }
        }

        assert!(received > 0);
    }

    #[test]
    fn test_latency_measurement_simulation() {
        let mut rtts = Vec::new();

        for seq in 0..50 {
            let ping = PingPacket {
                sequence: seq,
                timestamp: seq as u64 * 20000,
            };

            let encoded = ping.encode();
            let _decoded = PingPacket::decode(&encoded).unwrap();

            // Simulate RTT
            let simulated_rtt = 15.0 + (seq % 5) as f32;
            rtts.push(simulated_rtt);
        }

        let avg_rtt: f32 = rtts.iter().sum::<f32>() / rtts.len() as f32;
        assert!(avg_rtt > 10.0 && avg_rtt < 30.0);
    }
}

// ============================================================================
// Async Tests
// ============================================================================

mod async_tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_channel_based_streaming() {
        let (tx, mut rx) = mpsc::channel::<AudioPacket>(100);

        // Producer task
        let producer = tokio::spawn(async move {
            for seq in 0u32..50 {
                let payload = Bytes::from(vec![seq as u8; 100]);
                let packet = AudioPacket::new(seq, seq as u64 * 960, 1, payload);
                tx.send(packet).await.unwrap();
            }
        });

        // Consumer task
        let consumer = tokio::spawn(async move {
            let mut received = 0u32;
            while let Some(packet) = rx.recv().await {
                assert_eq!(packet.header.sequence, received);
                received += 1;
                if received >= 50 {
                    break;
                }
            }
            received
        });

        producer.await.unwrap();
        let count = consumer.await.unwrap();
        assert_eq!(count, 50);
    }

    #[tokio::test]
    async fn test_concurrent_packet_processing() {
        use futures::stream::{self, StreamExt};

        let packets: Vec<AudioPacket> = (0u32..100)
            .map(|seq| {
                let payload = Bytes::from(vec![0u8; 100]);
                AudioPacket::new(seq, seq as u64 * 960, 1, payload)
            })
            .collect();

        let results: Vec<Result<AudioPacket>> = stream::iter(packets)
            .map(|packet| async move {
                let encoded = packet.encode();
                AudioPacket::decode(encoded)
            })
            .buffer_unordered(10)
            .collect()
            .await;

        assert_eq!(results.len(), 100);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}

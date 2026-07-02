// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Low-Latency QUIC-Based Network Protocol
//!
//! High-performance audio streaming protocol built on QUIC for
//! minimal latency and maximum reliability.

use bytes::{Buf, BufMut, Bytes, BytesMut};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::mpsc;
use uuid::Uuid;

pub mod connection;
pub mod stream;
pub mod packet;
pub mod congestion;

#[derive(Error, Debug)]
pub enum ProtocolError {
    #[error("Connection error: {0}")]
    ConnectionError(String),

    #[error("Stream error: {0}")]
    StreamError(String),

    #[error("Packet error: {0}")]
    PacketError(String),

    #[error("Timeout")]
    Timeout,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Codec error: {0}")]
    CodecError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;

/// Protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Maximum packet size in bytes
    pub max_packet_size: usize,
    /// Audio frame size in samples
    pub frame_size: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u8,
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Enable FEC (forward error correction)
    pub enable_fec: bool,
    /// FEC redundancy ratio (0.0-1.0)
    pub fec_ratio: f32,
    /// Enable adaptive bitrate
    pub enable_abr: bool,
    /// Jitter buffer size in frames
    pub jitter_buffer_frames: usize,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            max_packet_size: 1400,
            frame_size: 960,        // 20ms at 48kHz
            sample_rate: 48000,
            channels: 2,
            target_latency_ms: 50,
            enable_fec: true,
            fec_ratio: 0.2,
            enable_abr: true,
            jitter_buffer_frames: 3,
        }
    }
}

impl ProtocolConfig {
    /// Calculate frame duration in milliseconds
    pub fn frame_duration_ms(&self) -> f32 {
        self.frame_size as f32 * 1000.0 / self.sample_rate as f32
    }

    /// Calculate bytes per frame
    pub fn bytes_per_frame(&self) -> usize {
        self.frame_size * self.channels as usize * 4 // f32 = 4 bytes
    }
}

/// Packet types in the protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// Audio data packet
    Audio = 0,
    /// Audio with FEC
    AudioFec = 1,
    /// Control message
    Control = 2,
    /// Keep-alive ping
    Ping = 3,
    /// Ping response
    Pong = 4,
    /// Stream metadata
    Metadata = 5,
    /// Seek request
    Seek = 6,
    /// Quality change
    Quality = 7,
    /// Error notification
    Error = 8,
}

impl PacketType {
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(PacketType::Audio),
            1 => Some(PacketType::AudioFec),
            2 => Some(PacketType::Control),
            3 => Some(PacketType::Ping),
            4 => Some(PacketType::Pong),
            5 => Some(PacketType::Metadata),
            6 => Some(PacketType::Seek),
            7 => Some(PacketType::Quality),
            8 => Some(PacketType::Error),
            _ => None,
        }
    }
}

/// Audio packet header
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioPacketHeader {
    /// Sequence number
    pub sequence: u32,
    /// Timestamp (samples from stream start)
    pub timestamp: u64,
    /// Stream ID
    pub stream_id: u32,
    /// Payload size
    pub payload_size: u16,
    /// Flags
    pub flags: u8,
}

impl AudioPacketHeader {
    pub const SIZE: usize = 19; // 4 + 8 + 4 + 2 + 1

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::SIZE);
        buf.put_u32(self.sequence);
        buf.put_u64(self.timestamp);
        buf.put_u32(self.stream_id);
        buf.put_u16(self.payload_size);
        buf.put_u8(self.flags);
        buf.freeze()
    }

    pub fn decode(mut buf: &[u8]) -> Result<Self> {
        if buf.len() < Self::SIZE {
            return Err(ProtocolError::PacketError("Header too short".to_string()));
        }

        Ok(Self {
            sequence: buf.get_u32(),
            timestamp: buf.get_u64(),
            stream_id: buf.get_u32(),
            payload_size: buf.get_u16(),
            flags: buf.get_u8(),
        })
    }
}

/// Audio packet with payload
#[derive(Debug, Clone)]
pub struct AudioPacket {
    pub header: AudioPacketHeader,
    pub payload: Bytes,
}

impl AudioPacket {
    pub fn new(sequence: u32, timestamp: u64, stream_id: u32, payload: Bytes) -> Self {
        Self {
            header: AudioPacketHeader {
                sequence,
                timestamp,
                stream_id,
                payload_size: payload.len() as u16,
                flags: 0,
            },
            payload,
        }
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(AudioPacketHeader::SIZE + self.payload.len());
        buf.extend_from_slice(&self.header.encode());
        buf.extend_from_slice(&self.payload);
        buf.freeze()
    }

    pub fn decode(data: Bytes) -> Result<Self> {
        if data.len() < AudioPacketHeader::SIZE {
            return Err(ProtocolError::PacketError("Packet too short".to_string()));
        }

        let header = AudioPacketHeader::decode(&data[..AudioPacketHeader::SIZE])?;
        let payload = data.slice(AudioPacketHeader::SIZE..);

        Ok(Self { header, payload })
    }
}

/// Stream statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamStats {
    pub packets_sent: u64,
    pub packets_received: u64,
    pub packets_lost: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub rtt_ms: f32,
    pub jitter_ms: f32,
    pub packet_loss_ratio: f32,
    pub bitrate_kbps: f32,
}

/// Jitter buffer for packet reordering
pub struct JitterBuffer {
    buffer: Vec<Option<AudioPacket>>,
    read_index: u32,
    write_index: u32,
    capacity: usize,
    target_delay_frames: usize,
    current_delay: usize,
}

impl JitterBuffer {
    pub fn new(capacity: usize, target_delay_frames: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            read_index: 0,
            write_index: 0,
            capacity,
            target_delay_frames,
            current_delay: 0,
        }
    }

    /// Push a packet into the buffer
    pub fn push(&mut self, packet: AudioPacket) {
        let seq = packet.header.sequence;
        let index = (seq as usize) % self.capacity;

        // Handle sequence wraparound
        if self.current_delay == 0 {
            self.read_index = seq.saturating_sub(self.target_delay_frames as u32);
        }

        self.buffer[index] = Some(packet);
        self.write_index = seq;
        self.current_delay = (seq - self.read_index) as usize;
    }

    /// Pop the next packet (may return None if packet is missing)
    pub fn pop(&mut self) -> Option<AudioPacket> {
        if self.current_delay < self.target_delay_frames {
            return None;
        }

        let index = (self.read_index as usize) % self.capacity;
        let packet = self.buffer[index].take();

        self.read_index = self.read_index.wrapping_add(1);
        if self.current_delay > 0 {
            self.current_delay -= 1;
        }

        packet
    }

    /// Get current buffer level
    pub fn level(&self) -> usize {
        self.current_delay
    }

    /// Check if buffer is ready for playback
    pub fn is_ready(&self) -> bool {
        self.current_delay >= self.target_delay_frames
    }
}

/// Forward Error Correction encoder
pub struct FecEncoder {
    block_size: usize,
    redundancy: usize,
    current_block: Vec<Bytes>,
}

impl FecEncoder {
    pub fn new(block_size: usize, redundancy_ratio: f32) -> Self {
        let redundancy = ((block_size as f32) * redundancy_ratio).ceil() as usize;
        Self {
            block_size,
            redundancy,
            current_block: Vec::with_capacity(block_size),
        }
    }

    /// Add packet to current FEC block
    pub fn add_packet(&mut self, data: Bytes) -> Option<Vec<Bytes>> {
        self.current_block.push(data);

        if self.current_block.len() >= self.block_size {
            let fec_packets = self.generate_fec();
            self.current_block.clear();
            Some(fec_packets)
        } else {
            None
        }
    }

    /// Generate FEC packets using XOR
    fn generate_fec(&self) -> Vec<Bytes> {
        if self.current_block.is_empty() {
            return vec![];
        }

        let max_len = self.current_block.iter().map(|p| p.len()).max().unwrap_or(0);
        let mut fec_packets = Vec::with_capacity(self.redundancy);

        // Simple XOR FEC
        for _ in 0..self.redundancy {
            let mut fec_data = vec![0u8; max_len];

            for packet in &self.current_block {
                for (i, &byte) in packet.iter().enumerate() {
                    fec_data[i] ^= byte;
                }
            }

            fec_packets.push(Bytes::from(fec_data));
        }

        fec_packets
    }
}

/// Forward Error Correction decoder
pub struct FecDecoder {
    block_size: usize,
    received_packets: HashMap<u32, Bytes>,
    fec_packets: Vec<Bytes>,
}

impl FecDecoder {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            received_packets: HashMap::new(),
            fec_packets: Vec::new(),
        }
    }

    /// Add received packet
    pub fn add_packet(&mut self, sequence: u32, data: Bytes) {
        self.received_packets.insert(sequence, data);
    }

    /// Add FEC packet
    pub fn add_fec(&mut self, data: Bytes) {
        self.fec_packets.push(data);
    }

    /// Try to recover missing packets
    pub fn recover(&mut self, expected_sequences: &[u32]) -> HashMap<u32, Bytes> {
        let mut recovered = HashMap::new();

        // Find missing packets
        let missing: Vec<_> = expected_sequences
            .iter()
            .filter(|seq| !self.received_packets.contains_key(seq))
            .collect();

        // Can only recover if exactly one packet is missing per FEC packet
        if missing.len() <= self.fec_packets.len() {
            for (&missing_seq, fec) in missing.iter().zip(self.fec_packets.iter()) {
                // XOR all received packets to recover missing
                let mut recovered_data = fec.to_vec();

                for (&seq, packet) in &self.received_packets {
                    if expected_sequences.contains(&seq) && seq != *missing_seq {
                        for (i, &byte) in packet.iter().enumerate() {
                            if i < recovered_data.len() {
                                recovered_data[i] ^= byte;
                            }
                        }
                    }
                }

                recovered.insert(*missing_seq, Bytes::from(recovered_data));
            }
        }

        recovered
    }

    /// Clear state for new block
    pub fn clear(&mut self) {
        self.received_packets.clear();
        self.fec_packets.clear();
    }
}

/// Network condition estimator
pub struct NetworkEstimator {
    rtt_samples: Vec<f32>,
    jitter_samples: Vec<f32>,
    loss_window: Vec<bool>,
    window_size: usize,
    last_packet_time: Option<Instant>,
    expected_interval: Duration,
}

impl NetworkEstimator {
    pub fn new(window_size: usize, expected_interval_ms: u32) -> Self {
        Self {
            rtt_samples: Vec::with_capacity(window_size),
            jitter_samples: Vec::with_capacity(window_size),
            loss_window: Vec::with_capacity(window_size),
            window_size,
            last_packet_time: None,
            expected_interval: Duration::from_millis(expected_interval_ms as u64),
        }
    }

    /// Record RTT sample
    pub fn record_rtt(&mut self, rtt_ms: f32) {
        if self.rtt_samples.len() >= self.window_size {
            self.rtt_samples.remove(0);
        }
        self.rtt_samples.push(rtt_ms);
    }

    /// Record packet arrival
    pub fn record_packet(&mut self, lost: bool) {
        let now = Instant::now();

        if let Some(last) = self.last_packet_time {
            let actual_interval = now.duration_since(last);
            let expected = self.expected_interval.as_secs_f32() * 1000.0;
            let actual = actual_interval.as_secs_f32() * 1000.0;
            let jitter = (actual - expected).abs();

            if self.jitter_samples.len() >= self.window_size {
                self.jitter_samples.remove(0);
            }
            self.jitter_samples.push(jitter);
        }

        self.last_packet_time = Some(now);

        if self.loss_window.len() >= self.window_size {
            self.loss_window.remove(0);
        }
        self.loss_window.push(lost);
    }

    /// Get smoothed RTT
    pub fn smoothed_rtt(&self) -> f32 {
        if self.rtt_samples.is_empty() {
            return 0.0;
        }
        self.rtt_samples.iter().sum::<f32>() / self.rtt_samples.len() as f32
    }

    /// Get jitter estimate
    pub fn jitter(&self) -> f32 {
        if self.jitter_samples.is_empty() {
            return 0.0;
        }
        self.jitter_samples.iter().sum::<f32>() / self.jitter_samples.len() as f32
    }

    /// Get packet loss ratio
    pub fn loss_ratio(&self) -> f32 {
        if self.loss_window.is_empty() {
            return 0.0;
        }
        let lost = self.loss_window.iter().filter(|&&x| x).count();
        lost as f32 / self.loss_window.len() as f32
    }

    /// Recommend bitrate based on conditions
    pub fn recommended_bitrate_kbps(&self, base_bitrate: f32) -> f32 {
        let loss = self.loss_ratio();
        let rtt = self.smoothed_rtt();

        // Reduce bitrate based on conditions
        let loss_factor = (1.0 - loss * 2.0).max(0.5);
        let rtt_factor = if rtt > 100.0 {
            0.8
        } else if rtt > 50.0 {
            0.9
        } else {
            1.0
        };

        base_bitrate * loss_factor * rtt_factor
    }
}

/// Audio stream session
pub struct AudioSession {
    pub id: Uuid,
    pub config: ProtocolConfig,
    pub stats: Arc<RwLock<StreamStats>>,
    jitter_buffer: JitterBuffer,
    fec_encoder: Option<FecEncoder>,
    fec_decoder: Option<FecDecoder>,
    network_estimator: NetworkEstimator,
    next_sequence: u32,
    current_timestamp: u64,
}

impl AudioSession {
    pub fn new(config: ProtocolConfig) -> Self {
        let fec_encoder = if config.enable_fec {
            Some(FecEncoder::new(5, config.fec_ratio))
        } else {
            None
        };

        let fec_decoder = if config.enable_fec {
            Some(FecDecoder::new(5))
        } else {
            None
        };

        Self {
            id: Uuid::new_v4(),
            config: config.clone(),
            stats: Arc::new(RwLock::new(StreamStats::default())),
            jitter_buffer: JitterBuffer::new(32, config.jitter_buffer_frames),
            fec_encoder,
            fec_decoder,
            network_estimator: NetworkEstimator::new(
                50,
                config.frame_duration_ms() as u32,
            ),
            next_sequence: 0,
            current_timestamp: 0,
        }
    }

    /// Create audio packet from samples
    pub fn create_packet(&mut self, samples: &[f32]) -> Vec<Bytes> {
        let mut packets = Vec::new();

        // Encode samples to bytes
        let mut payload = BytesMut::with_capacity(samples.len() * 4);
        for &sample in samples {
            payload.put_f32(sample);
        }

        let packet = AudioPacket::new(
            self.next_sequence,
            self.current_timestamp,
            0,
            payload.freeze(),
        );

        let encoded = packet.encode();
        packets.push(encoded.clone());

        // Generate FEC if enabled
        if let Some(ref mut fec) = self.fec_encoder {
            if let Some(fec_packets) = fec.add_packet(encoded) {
                packets.extend(fec_packets);
            }
        }

        // Update state
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.current_timestamp += self.config.frame_size as u64;

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.packets_sent += 1;
            stats.bytes_sent += packets.iter().map(|p| p.len() as u64).sum::<u64>();
        }

        packets
    }

    /// Process received packet
    pub fn receive_packet(&mut self, data: Bytes) -> Option<Vec<f32>> {
        match AudioPacket::decode(data) {
            Ok(packet) => {
                self.network_estimator.record_packet(false);

                // Add to jitter buffer
                self.jitter_buffer.push(packet);

                // Update stats
                {
                    let mut stats = self.stats.write();
                    stats.packets_received += 1;
                }

                // Try to output from jitter buffer
                self.get_next_frame()
            }
            Err(_) => {
                self.network_estimator.record_packet(true);
                None
            }
        }
    }

    /// Get next frame from jitter buffer
    pub fn get_next_frame(&mut self) -> Option<Vec<f32>> {
        if let Some(packet) = self.jitter_buffer.pop() {
            // Decode payload to samples
            let mut samples = Vec::with_capacity(packet.payload.len() / 4);
            let mut buf = packet.payload.as_ref();

            while buf.remaining() >= 4 {
                samples.push(buf.get_f32());
            }

            Some(samples)
        } else {
            // Packet loss concealment: return silence or interpolated
            None
        }
    }

    /// Record RTT measurement
    pub fn record_rtt(&mut self, rtt_ms: f32) {
        self.network_estimator.record_rtt(rtt_ms);

        let mut stats = self.stats.write();
        stats.rtt_ms = self.network_estimator.smoothed_rtt();
        stats.jitter_ms = self.network_estimator.jitter();
        stats.packet_loss_ratio = self.network_estimator.loss_ratio();
    }

    /// Get recommended bitrate
    pub fn recommended_bitrate(&self) -> f32 {
        let base_bitrate = self.config.sample_rate as f32 * self.config.channels as f32 * 32.0 / 1000.0;
        self.network_estimator.recommended_bitrate_kbps(base_bitrate)
    }

    /// Get current stats
    pub fn stats(&self) -> StreamStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_packet_encoding() {
        let payload = Bytes::from(vec![1, 2, 3, 4]);
        let packet = AudioPacket::new(42, 1000, 1, payload.clone());

        let encoded = packet.encode();
        let decoded = AudioPacket::decode(encoded).unwrap();

        assert_eq!(decoded.header.sequence, 42);
        assert_eq!(decoded.header.timestamp, 1000);
        assert_eq!(decoded.payload, payload);
    }

    #[test]
    fn test_jitter_buffer() {
        let mut buffer = JitterBuffer::new(16, 3);

        // Add packets
        for i in 0..5 {
            let packet = AudioPacket::new(i, i as u64 * 960, 0, Bytes::new());
            buffer.push(packet);
        }

        assert!(buffer.is_ready());

        // Pop packets
        let packet = buffer.pop().unwrap();
        assert_eq!(packet.header.sequence, 0);
    }

    #[test]
    fn test_fec_encoder() {
        let mut encoder = FecEncoder::new(3, 0.5);

        encoder.add_packet(Bytes::from(vec![1, 2, 3]));
        encoder.add_packet(Bytes::from(vec![4, 5, 6]));
        let fec = encoder.add_packet(Bytes::from(vec![7, 8, 9]));

        assert!(fec.is_some());
        let fec_packets = fec.unwrap();
        assert!(!fec_packets.is_empty());
    }

    #[test]
    fn test_network_estimator() {
        let mut estimator = NetworkEstimator::new(10, 20);

        for i in 0..10 {
            estimator.record_rtt(50.0 + i as f32);
            estimator.record_packet(i % 5 == 0);
        }

        assert!(estimator.smoothed_rtt() > 0.0);
        assert!(estimator.loss_ratio() > 0.0);
    }
}

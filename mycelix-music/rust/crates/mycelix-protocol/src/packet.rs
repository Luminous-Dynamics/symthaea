// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Packet encoding and decoding utilities

use crate::{PacketType, ProtocolError, Result};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};

/// Generic packet header
#[derive(Debug, Clone)]
pub struct PacketHeader {
    pub packet_type: PacketType,
    pub flags: u8,
    pub length: u16,
}

impl PacketHeader {
    pub const SIZE: usize = 4;

    pub fn new(packet_type: PacketType, length: u16) -> Self {
        Self {
            packet_type,
            flags: 0,
            length,
        }
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(Self::SIZE);
        buf.put_u8(self.packet_type as u8);
        buf.put_u8(self.flags);
        buf.put_u16(self.length);
        buf.freeze()
    }

    pub fn decode(mut data: &[u8]) -> Result<Self> {
        if data.len() < Self::SIZE {
            return Err(ProtocolError::PacketError("Header too short".to_string()));
        }

        let type_byte = data.get_u8();
        let packet_type = PacketType::from_u8(type_byte)
            .ok_or_else(|| ProtocolError::PacketError(format!("Invalid packet type: {}", type_byte)))?;

        Ok(Self {
            packet_type,
            flags: data.get_u8(),
            length: data.get_u16(),
        })
    }
}

/// Control message types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessageType {
    /// Request to start streaming
    Start,
    /// Request to stop streaming
    Stop,
    /// Pause playback
    Pause,
    /// Resume playback
    Resume,
    /// Seek to position
    Seek,
    /// Change quality
    Quality,
    /// Request sync
    Sync,
    /// Acknowledge
    Ack,
}

/// Control message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlMessage {
    pub message_type: ControlMessageType,
    pub stream_id: u32,
    pub sequence: u32,
    pub payload: Vec<u8>,
}

impl ControlMessage {
    pub fn new(message_type: ControlMessageType, stream_id: u32) -> Self {
        Self {
            message_type,
            stream_id,
            sequence: 0,
            payload: Vec::new(),
        }
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::new();
        buf.put_u8(self.message_type as u8);
        buf.put_u32(self.stream_id);
        buf.put_u32(self.sequence);
        buf.put_u16(self.payload.len() as u16);
        buf.extend_from_slice(&self.payload);
        buf.freeze()
    }

    pub fn decode(mut data: &[u8]) -> Result<Self> {
        if data.len() < 11 {
            return Err(ProtocolError::PacketError("Control message too short".to_string()));
        }

        let type_byte = data.get_u8();
        let message_type = match type_byte {
            0 => ControlMessageType::Start,
            1 => ControlMessageType::Stop,
            2 => ControlMessageType::Pause,
            3 => ControlMessageType::Resume,
            4 => ControlMessageType::Seek,
            5 => ControlMessageType::Quality,
            6 => ControlMessageType::Sync,
            7 => ControlMessageType::Ack,
            _ => return Err(ProtocolError::PacketError(format!("Invalid control type: {}", type_byte))),
        };

        let stream_id = data.get_u32();
        let sequence = data.get_u32();
        let payload_len = data.get_u16() as usize;

        if data.len() < payload_len {
            return Err(ProtocolError::PacketError("Payload truncated".to_string()));
        }

        let payload = data[..payload_len].to_vec();

        Ok(Self {
            message_type,
            stream_id,
            sequence,
            payload,
        })
    }
}

/// Ping packet for RTT measurement
#[derive(Debug, Clone)]
pub struct PingPacket {
    pub sequence: u32,
    pub timestamp: u64,
}

impl PingPacket {
    pub fn new(sequence: u32) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self { sequence, timestamp }
    }

    pub fn encode(&self) -> Bytes {
        let mut buf = BytesMut::with_capacity(12);
        buf.put_u32(self.sequence);
        buf.put_u64(self.timestamp);
        buf.freeze()
    }

    pub fn decode(mut data: &[u8]) -> Result<Self> {
        if data.len() < 12 {
            return Err(ProtocolError::PacketError("Ping too short".to_string()));
        }

        Ok(Self {
            sequence: data.get_u32(),
            timestamp: data.get_u64(),
        })
    }

    /// Calculate RTT from pong response
    pub fn calculate_rtt(&self) -> f32 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        (now - self.timestamp) as f32 / 1000.0 // Convert to milliseconds
    }
}

/// Metadata packet for stream information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataPacket {
    pub stream_id: u32,
    pub sample_rate: u32,
    pub channels: u8,
    pub codec: String,
    pub bitrate: u32,
    pub duration_ms: u64,
    pub title: Option<String>,
    pub artist: Option<String>,
}

impl MetadataPacket {
    pub fn encode(&self) -> Result<Bytes> {
        let json = serde_json::to_vec(self)
            .map_err(|e| ProtocolError::CodecError(e.to_string()))?;
        Ok(Bytes::from(json))
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| ProtocolError::CodecError(e.to_string()))
    }
}

/// Quality change request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequest {
    pub stream_id: u32,
    pub target_bitrate: u32,
    pub reason: QualityChangeReason,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QualityChangeReason {
    UserRequest,
    BandwidthDrop,
    BandwidthIncrease,
    BufferUnderrun,
    BufferOverrun,
}

impl QualityRequest {
    pub fn encode(&self) -> Result<Bytes> {
        let json = serde_json::to_vec(self)
            .map_err(|e| ProtocolError::CodecError(e.to_string()))?;
        Ok(Bytes::from(json))
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        serde_json::from_slice(data)
            .map_err(|e| ProtocolError::CodecError(e.to_string()))
    }
}

/// Packet builder for constructing protocol packets
pub struct PacketBuilder {
    buffer: BytesMut,
}

impl PacketBuilder {
    pub fn new() -> Self {
        Self {
            buffer: BytesMut::with_capacity(1500),
        }
    }

    pub fn audio(mut self, sequence: u32, timestamp: u64, stream_id: u32, payload: &[u8]) -> Bytes {
        let header = PacketHeader::new(PacketType::Audio, payload.len() as u16 + 16);
        self.buffer.extend_from_slice(&header.encode());
        self.buffer.put_u32(sequence);
        self.buffer.put_u64(timestamp);
        self.buffer.put_u32(stream_id);
        self.buffer.extend_from_slice(payload);
        self.buffer.freeze()
    }

    pub fn control(mut self, message: &ControlMessage) -> Bytes {
        let encoded = message.encode();
        let header = PacketHeader::new(PacketType::Control, encoded.len() as u16);
        self.buffer.extend_from_slice(&header.encode());
        self.buffer.extend_from_slice(&encoded);
        self.buffer.freeze()
    }

    pub fn ping(mut self, ping: &PingPacket) -> Bytes {
        let encoded = ping.encode();
        let header = PacketHeader::new(PacketType::Ping, encoded.len() as u16);
        self.buffer.extend_from_slice(&header.encode());
        self.buffer.extend_from_slice(&encoded);
        self.buffer.freeze()
    }

    pub fn pong(mut self, ping: &PingPacket) -> Bytes {
        let encoded = ping.encode();
        let header = PacketHeader::new(PacketType::Pong, encoded.len() as u16);
        self.buffer.extend_from_slice(&header.encode());
        self.buffer.extend_from_slice(&encoded);
        self.buffer.freeze()
    }

    pub fn metadata(mut self, metadata: &MetadataPacket) -> Result<Bytes> {
        let encoded = metadata.encode()?;
        let header = PacketHeader::new(PacketType::Metadata, encoded.len() as u16);
        self.buffer.extend_from_slice(&header.encode());
        self.buffer.extend_from_slice(&encoded);
        Ok(self.buffer.freeze())
    }
}

impl Default for PacketBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Packet parser for decoding received packets
pub struct PacketParser;

impl PacketParser {
    /// Parse a packet and return its type and payload
    pub fn parse(data: &[u8]) -> Result<(PacketType, Bytes)> {
        let header = PacketHeader::decode(data)?;

        if data.len() < PacketHeader::SIZE + header.length as usize {
            return Err(ProtocolError::PacketError("Packet truncated".to_string()));
        }

        let payload = Bytes::copy_from_slice(&data[PacketHeader::SIZE..PacketHeader::SIZE + header.length as usize]);

        Ok((header.packet_type, payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_header() {
        let header = PacketHeader::new(PacketType::Audio, 100);
        let encoded = header.encode();
        let decoded = PacketHeader::decode(&encoded).unwrap();

        assert_eq!(decoded.packet_type, PacketType::Audio);
        assert_eq!(decoded.length, 100);
    }

    #[test]
    fn test_control_message() {
        let msg = ControlMessage::new(ControlMessageType::Start, 42);
        let encoded = msg.encode();
        let decoded = ControlMessage::decode(&encoded).unwrap();

        assert_eq!(decoded.message_type, ControlMessageType::Start);
        assert_eq!(decoded.stream_id, 42);
    }

    #[test]
    fn test_ping_packet() {
        let ping = PingPacket::new(1);
        let encoded = ping.encode();
        let decoded = PingPacket::decode(&encoded).unwrap();

        assert_eq!(decoded.sequence, 1);
    }

    #[test]
    fn test_packet_builder() {
        let packet = PacketBuilder::new().audio(0, 0, 0, &[1, 2, 3, 4]);
        assert!(packet.len() > 0);
    }
}

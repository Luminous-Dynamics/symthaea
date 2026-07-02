// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio stream management

use crate::{AudioPacket, ProtocolConfig, ProtocolError, Result, StreamStats};
use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use uuid::Uuid;

/// Stream direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamDirection {
    Send,
    Receive,
    Bidirectional,
}

/// Stream priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StreamPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Realtime = 3,
}

/// Audio stream
pub struct AudioStream {
    pub id: Uuid,
    pub stream_id: u32,
    direction: StreamDirection,
    priority: StreamPriority,
    config: ProtocolConfig,
    /// Outgoing packet queue
    send_queue: VecDeque<AudioPacket>,
    /// Incoming packet buffer
    receive_buffer: VecDeque<AudioPacket>,
    /// Next expected sequence number
    next_sequence: u32,
    /// Current timestamp
    current_timestamp: u64,
    /// Stream creation time
    created_at: Instant,
    /// Last activity time
    last_activity: Instant,
    /// Stream statistics
    stats: StreamStats,
    /// Is stream active
    active: bool,
}

impl AudioStream {
    pub fn new(stream_id: u32, direction: StreamDirection, config: ProtocolConfig) -> Self {
        let now = Instant::now();
        Self {
            id: Uuid::new_v4(),
            stream_id,
            direction,
            priority: StreamPriority::Normal,
            config,
            send_queue: VecDeque::new(),
            receive_buffer: VecDeque::new(),
            next_sequence: 0,
            current_timestamp: 0,
            created_at: now,
            last_activity: now,
            stats: StreamStats::default(),
            active: true,
        }
    }

    pub fn with_priority(mut self, priority: StreamPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Queue samples for sending
    pub fn send(&mut self, samples: &[f32]) -> Result<()> {
        if self.direction == StreamDirection::Receive {
            return Err(ProtocolError::InvalidState("Cannot send on receive-only stream".to_string()));
        }

        // Create packet from samples
        let mut payload = Vec::with_capacity(samples.len() * 4);
        for &sample in samples {
            payload.extend_from_slice(&sample.to_le_bytes());
        }

        let packet = AudioPacket::new(
            self.next_sequence,
            self.current_timestamp,
            self.stream_id,
            Bytes::from(payload),
        );

        self.send_queue.push_back(packet);
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.current_timestamp += self.config.frame_size as u64;
        self.last_activity = Instant::now();

        Ok(())
    }

    /// Get next packet to transmit
    pub fn poll_send(&mut self) -> Option<AudioPacket> {
        let packet = self.send_queue.pop_front()?;
        self.stats.packets_sent += 1;
        self.stats.bytes_sent += packet.payload.len() as u64;
        Some(packet)
    }

    /// Receive a packet
    pub fn receive(&mut self, packet: AudioPacket) -> Result<()> {
        if self.direction == StreamDirection::Send {
            return Err(ProtocolError::InvalidState("Cannot receive on send-only stream".to_string()));
        }

        self.receive_buffer.push_back(packet);
        self.stats.packets_received += 1;
        self.last_activity = Instant::now();

        Ok(())
    }

    /// Get received samples
    pub fn poll_receive(&mut self) -> Option<Vec<f32>> {
        let packet = self.receive_buffer.pop_front()?;
        self.stats.bytes_received += packet.payload.len() as u64;

        // Decode samples
        let mut samples = Vec::with_capacity(packet.payload.len() / 4);
        let mut buf = packet.payload.as_ref();

        while buf.len() >= 4 {
            let bytes: [u8; 4] = buf[..4].try_into().unwrap();
            samples.push(f32::from_le_bytes(bytes));
            buf = &buf[4..];
        }

        Some(samples)
    }

    /// Check if stream has pending data to send
    pub fn has_pending(&self) -> bool {
        !self.send_queue.is_empty()
    }

    /// Check if stream has received data available
    pub fn has_available(&self) -> bool {
        !self.receive_buffer.is_empty()
    }

    /// Get stream statistics
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Get stream uptime
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Close the stream
    pub fn close(&mut self) {
        self.active = false;
    }

    /// Check if stream is active
    pub fn is_active(&self) -> bool {
        self.active
    }
}

/// Stream manager for multiple concurrent streams
pub struct StreamManager {
    streams: Vec<AudioStream>,
    config: ProtocolConfig,
    max_streams: usize,
    next_stream_id: u32,
}

impl StreamManager {
    pub fn new(config: ProtocolConfig, max_streams: usize) -> Self {
        Self {
            streams: Vec::new(),
            config,
            max_streams,
            next_stream_id: 0,
        }
    }

    /// Create a new stream
    pub fn create_stream(&mut self, direction: StreamDirection) -> Result<u32> {
        if self.streams.len() >= self.max_streams {
            return Err(ProtocolError::StreamError("Max streams reached".to_string()));
        }

        let stream_id = self.next_stream_id;
        self.next_stream_id += 1;

        let stream = AudioStream::new(stream_id, direction, self.config.clone());
        self.streams.push(stream);

        Ok(stream_id)
    }

    /// Get stream by ID
    pub fn get_stream(&mut self, stream_id: u32) -> Option<&mut AudioStream> {
        self.streams.iter_mut().find(|s| s.stream_id == stream_id)
    }

    /// Close a stream
    pub fn close_stream(&mut self, stream_id: u32) -> Result<()> {
        if let Some(stream) = self.get_stream(stream_id) {
            stream.close();
            Ok(())
        } else {
            Err(ProtocolError::StreamError(format!("Stream {} not found", stream_id)))
        }
    }

    /// Get all active streams
    pub fn active_streams(&self) -> Vec<u32> {
        self.streams
            .iter()
            .filter(|s| s.is_active())
            .map(|s| s.stream_id)
            .collect()
    }

    /// Poll all streams for outgoing packets (prioritized)
    pub fn poll_all(&mut self) -> Vec<(u32, AudioPacket)> {
        let mut packets = Vec::new();

        // Sort by priority (highest first)
        self.streams.sort_by(|a, b| b.priority.cmp(&a.priority));

        for stream in &mut self.streams {
            if stream.is_active() {
                while let Some(packet) = stream.poll_send() {
                    packets.push((stream.stream_id, packet));
                }
            }
        }

        packets
    }

    /// Clean up closed streams
    pub fn cleanup(&mut self) {
        self.streams.retain(|s| s.is_active());
    }
}

/// Multiplexed stream with sub-streams for different audio components
pub struct MultiplexedStream {
    main_stream: AudioStream,
    sub_streams: Vec<AudioStream>,
}

impl MultiplexedStream {
    pub fn new(stream_id: u32, config: ProtocolConfig) -> Self {
        Self {
            main_stream: AudioStream::new(stream_id, StreamDirection::Bidirectional, config),
            sub_streams: Vec::new(),
        }
    }

    /// Add a sub-stream for mixing
    pub fn add_sub_stream(&mut self, stream: AudioStream) {
        self.sub_streams.push(stream);
    }

    /// Mix all sub-streams into main stream
    pub fn mix(&mut self) -> Option<Vec<f32>> {
        let mut mixed: Option<Vec<f32>> = None;

        // Get main stream samples
        if let Some(samples) = self.main_stream.poll_receive() {
            mixed = Some(samples);
        }

        // Mix in sub-streams
        for sub in &mut self.sub_streams {
            if let Some(samples) = sub.poll_receive() {
                match &mut mixed {
                    Some(m) => {
                        for (i, sample) in samples.iter().enumerate() {
                            if i < m.len() {
                                m[i] += sample;
                            }
                        }
                    }
                    None => {
                        mixed = Some(samples);
                    }
                }
            }
        }

        // Normalize to prevent clipping
        if let Some(ref mut m) = mixed {
            let max = m.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if max > 1.0 {
                for sample in m.iter_mut() {
                    *sample /= max;
                }
            }
        }

        mixed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_stream() {
        let config = ProtocolConfig::default();
        let mut stream = AudioStream::new(0, StreamDirection::Send, config);

        let samples = vec![0.5f32; 960];
        stream.send(&samples).unwrap();

        assert!(stream.has_pending());
        let packet = stream.poll_send().unwrap();
        assert_eq!(packet.header.sequence, 0);
    }

    #[test]
    fn test_stream_manager() {
        let config = ProtocolConfig::default();
        let mut manager = StreamManager::new(config, 10);

        let stream_id = manager.create_stream(StreamDirection::Bidirectional).unwrap();
        assert!(manager.get_stream(stream_id).is_some());
    }
}

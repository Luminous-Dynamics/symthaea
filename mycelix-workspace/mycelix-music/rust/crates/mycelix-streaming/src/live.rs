// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Live streaming support

use crate::{Segment, StreamQuality, StreamingError, StreamingResult};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

/// Live stream state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveStreamState {
    Idle,
    Starting,
    Live,
    Paused,
    Ending,
    Ended,
}

/// Live stream configuration
#[derive(Debug, Clone)]
pub struct LiveStreamConfig {
    pub segment_duration: f32,
    pub playlist_size: usize,
    pub quality_levels: Vec<StreamQuality>,
    pub low_latency: bool,
    pub target_latency_ms: u32,
}

impl Default for LiveStreamConfig {
    fn default() -> Self {
        Self {
            segment_duration: 6.0,
            playlist_size: 5,
            quality_levels: vec![StreamQuality::Medium, StreamQuality::High],
            low_latency: false,
            target_latency_ms: 3000,
        }
    }
}

/// Live segment (simplified for live streaming)
#[derive(Debug, Clone)]
pub struct LiveSegment {
    pub sequence: u32,
    pub duration: f32,
    pub data: bytes::Bytes,
    pub is_init: bool,
}

/// Live stream manager
pub struct LiveStream {
    id: Uuid,
    config: LiveStreamConfig,
    state: RwLock<LiveStreamState>,
    segments: RwLock<VecDeque<Arc<LiveSegment>>>,
    sequence: RwLock<u32>,
    event_tx: broadcast::Sender<LiveStreamEvent>,
}

/// Live stream events
#[derive(Debug, Clone)]
pub enum LiveStreamEvent {
    Started { stream_id: Uuid },
    SegmentReady { sequence: u32 },
    StateChanged { old: LiveStreamState, new: LiveStreamState },
    Ended { stream_id: Uuid },
    Error { message: String },
}

impl LiveStream {
    pub fn new(config: LiveStreamConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);
        Self {
            id: Uuid::new_v4(),
            config,
            state: RwLock::new(LiveStreamState::Idle),
            segments: RwLock::new(VecDeque::new()),
            sequence: RwLock::new(0),
            event_tx,
        }
    }

    pub fn id(&self) -> Uuid {
        self.id
    }

    pub fn subscribe(&self) -> broadcast::Receiver<LiveStreamEvent> {
        self.event_tx.subscribe()
    }

    pub async fn start(&self) -> StreamingResult<()> {
        let mut state = self.state.write().await;
        let old_state = *state;
        *state = LiveStreamState::Starting;

        let _ = self.event_tx.send(LiveStreamEvent::StateChanged {
            old: old_state,
            new: LiveStreamState::Starting,
        });

        // Transition to live
        *state = LiveStreamState::Live;
        let _ = self.event_tx.send(LiveStreamEvent::Started { stream_id: self.id });
        let _ = self.event_tx.send(LiveStreamEvent::StateChanged {
            old: LiveStreamState::Starting,
            new: LiveStreamState::Live,
        });

        Ok(())
    }

    pub async fn push_segment(&self, segment: LiveSegment) -> StreamingResult<()> {
        let state = self.state.read().await;
        if *state != LiveStreamState::Live {
            return Err(StreamingError::StreamNotLive);
        }
        drop(state);

        let mut segments = self.segments.write().await;
        let mut sequence = self.sequence.write().await;

        // Maintain playlist size
        while segments.len() >= self.config.playlist_size {
            segments.pop_front();
        }

        let seq = *sequence;
        *sequence += 1;

        segments.push_back(Arc::new(segment));

        let _ = self.event_tx.send(LiveStreamEvent::SegmentReady { sequence: seq });

        Ok(())
    }

    pub async fn get_segments(&self) -> Vec<Arc<LiveSegment>> {
        let segments = self.segments.read().await;
        segments.iter().cloned().collect()
    }

    pub async fn get_segment(&self, sequence: u32) -> Option<Arc<LiveSegment>> {
        let segments = self.segments.read().await;
        let current_seq = *self.sequence.read().await;

        if sequence >= current_seq {
            return None;
        }

        let playlist_start = current_seq.saturating_sub(self.config.playlist_size as u32);
        if sequence < playlist_start {
            return None;
        }

        let index = (sequence - playlist_start) as usize;
        segments.get(index).cloned()
    }

    pub async fn generate_playlist(&self) -> String {
        let segments = self.segments.read().await;
        let sequence = *self.sequence.read().await;
        let target_duration = self.config.segment_duration.ceil() as u32;
        let media_sequence = sequence.saturating_sub(segments.len() as u32);

        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:6\n");
        playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_duration));
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", media_sequence));

        if self.config.low_latency {
            playlist.push_str("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES\n");
            playlist.push_str(&format!(
                "#EXT-X-PART-INF:PART-TARGET={:.3}\n",
                self.config.segment_duration / 6.0
            ));
        }

        for (i, segment) in segments.iter().enumerate() {
            let seg_seq = media_sequence + i as u32;
            playlist.push_str(&format!("#EXTINF:{:.3},\n", segment.duration));
            playlist.push_str(&format!("segment_{}.ts\n", seg_seq));
        }

        playlist
    }

    pub async fn end(&self) -> StreamingResult<()> {
        let mut state = self.state.write().await;
        let old_state = *state;
        *state = LiveStreamState::Ending;

        let _ = self.event_tx.send(LiveStreamEvent::StateChanged {
            old: old_state,
            new: LiveStreamState::Ending,
        });

        *state = LiveStreamState::Ended;
        let _ = self.event_tx.send(LiveStreamEvent::Ended { stream_id: self.id });

        Ok(())
    }

    pub async fn state(&self) -> LiveStreamState {
        *self.state.read().await
    }
}

/// Low-latency HLS extensions
pub mod ll_hls {
    /// Part (partial segment) for LL-HLS
    #[derive(Debug, Clone)]
    pub struct Part {
        pub duration: f32,
        pub independent: bool,
        pub data: bytes::Bytes,
    }

    /// Generate LL-HLS playlist with parts
    pub fn generate_ll_playlist(
        segments: &[(u32, f32, Vec<Part>)],
        media_sequence: u32,
        part_target: f32,
    ) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:9\n");
        playlist.push_str(&format!(
            "#EXT-X-TARGETDURATION:{}\n",
            segments.iter().map(|(_, d, _)| *d).fold(0.0f32, f32::max).ceil() as u32
        ));
        playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", media_sequence));
        playlist.push_str("#EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,PART-HOLD-BACK=0.5\n");
        playlist.push_str(&format!("#EXT-X-PART-INF:PART-TARGET={:.3}\n\n", part_target));

        for (seq, duration, parts) in segments {
            // Output parts
            for (i, part) in parts.iter().enumerate() {
                let independent = if part.independent { ",INDEPENDENT=YES" } else { "" };
                playlist.push_str(&format!(
                    "#EXT-X-PART:DURATION={:.3},URI=\"segment_{}_part_{}.m4s\"{}\n",
                    part.duration, seq, i, independent
                ));
            }

            // Output segment
            playlist.push_str(&format!("#EXTINF:{:.3},\n", duration));
            playlist.push_str(&format!("segment_{}.m4s\n", seq));
        }

        playlist
    }
}

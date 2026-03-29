// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HLS (HTTP Live Streaming) support

use crate::{Segment, StreamQuality, StreamingResult};

/// HLS playlist builder
pub struct HlsPlaylistBuilder {
    target_duration: u32,
    segments: Vec<Segment>,
    quality: StreamQuality,
}

impl HlsPlaylistBuilder {
    pub fn new(target_duration: u32, quality: StreamQuality) -> Self {
        Self {
            target_duration,
            segments: Vec::new(),
            quality,
        }
    }

    pub fn add_segment(&mut self, segment: Segment) {
        self.segments.push(segment);
    }

    pub fn build_master_playlist(&self, variants: &[StreamQuality]) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:6\n\n");

        for variant in variants {
            let bandwidth = variant.bitrate();
            let resolution = variant.resolution();
            playlist.push_str(&format!(
                "#EXT-X-STREAM-INF:BANDWIDTH={},RESOLUTION={},CODECS=\"{}\"\n",
                bandwidth,
                resolution,
                variant.codec()
            ));
            playlist.push_str(&format!("{}.m3u8\n\n", variant.name()));
        }

        playlist
    }

    pub fn build_variant_playlist(&self) -> String {
        let mut playlist = String::from("#EXTM3U\n");
        playlist.push_str("#EXT-X-VERSION:6\n");
        playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", self.target_duration));
        playlist.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
        playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n\n");

        for (i, segment) in self.segments.iter().enumerate() {
            if segment.is_init {
                playlist.push_str(&format!(
                    "#EXT-X-MAP:URI=\"init_{}.mp4\"\n",
                    self.quality.name()
                ));
            } else {
                playlist.push_str(&format!("#EXTINF:{:.3},\n", segment.duration));
                playlist.push_str(&format!("segment_{}_{}.ts\n", self.quality.name(), i));
            }
        }

        playlist.push_str("#EXT-X-ENDLIST\n");
        playlist
    }
}

/// Generate HLS segments from audio data
pub fn generate_hls_segments(
    audio_data: &[u8],
    segment_duration: f32,
    quality: StreamQuality,
) -> StreamingResult<Vec<Segment>> {
    // Simplified segment generation - in production would use actual encoding
    let total_duration = audio_data.len() as f32 / (quality.bitrate() as f32 / 8.0);
    let num_segments = (total_duration / segment_duration).ceil() as usize;

    let mut segments = Vec::with_capacity(num_segments + 1);

    // Add init segment
    segments.push(Segment {
        id: format!("init_{}", quality.name()),
        sequence: 0,
        duration: 0.0,
        start_time: 0.0,
        data: bytes::Bytes::from(vec![0u8; 1024]), // Init segment placeholder
        quality,
        is_init: true,
        encryption: None,
    });

    // Add media segments
    let bytes_per_segment = audio_data.len() / num_segments.max(1);
    for i in 0..num_segments {
        let start = i * bytes_per_segment;
        let end = ((i + 1) * bytes_per_segment).min(audio_data.len());
        let start_time = i as f32 * segment_duration;

        segments.push(Segment {
            id: format!("segment_{}_{}", quality.name(), i),
            sequence: (i + 1) as u32,
            duration: segment_duration,
            start_time,
            data: bytes::Bytes::copy_from_slice(&audio_data[start..end]),
            quality,
            is_init: false,
            encryption: None,
        });
    }

    Ok(segments)
}

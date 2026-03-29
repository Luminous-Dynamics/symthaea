// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DASH (Dynamic Adaptive Streaming over HTTP) support

use crate::{Segment, StreamQuality, StreamingResult};

/// DASH MPD (Media Presentation Description) builder
pub struct MpdBuilder {
    duration_seconds: f32,
    min_buffer_time: f32,
    profiles: String,
}

impl MpdBuilder {
    pub fn new(duration_seconds: f32) -> Self {
        Self {
            duration_seconds,
            min_buffer_time: 2.0,
            profiles: "urn:mpeg:dash:profile:isoff-on-demand:2011".to_string(),
        }
    }

    pub fn min_buffer_time(mut self, seconds: f32) -> Self {
        self.min_buffer_time = seconds;
        self
    }

    pub fn build(&self, variants: &[StreamQuality]) -> String {
        let duration_iso = format!("PT{:.3}S", self.duration_seconds);

        let mut mpd = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     profiles="{}"
     type="static"
     mediaPresentationDuration="{}"
     minBufferTime="PT{:.1}S">
  <Period>"#,
            self.profiles, duration_iso, self.min_buffer_time
        );

        mpd.push_str(r#"
    <AdaptationSet contentType="audio" mimeType="audio/mp4" segmentAlignment="true">"#);

        for variant in variants {
            mpd.push_str(&format!(
                r#"
      <Representation id="{}" bandwidth="{}" codecs="{}">
        <SegmentTemplate timescale="44100" duration="441000"
                         initialization="init_{}.m4s"
                         media="segment_{}_$Number$.m4s"/>
      </Representation>"#,
                variant.name(),
                variant.bitrate(),
                variant.codec(),
                variant.name(),
                variant.name()
            ));
        }

        mpd.push_str(r#"
    </AdaptationSet>
  </Period>
</MPD>"#);

        mpd
    }
}

/// Generate DASH segments from audio data
pub fn generate_dash_segments(
    audio_data: &[u8],
    segment_duration: f32,
    quality: StreamQuality,
) -> StreamingResult<Vec<Segment>> {
    // Similar to HLS but with DASH-specific formatting
    let total_duration = audio_data.len() as f32 / (quality.bitrate() as f32 / 8.0);
    let num_segments = (total_duration / segment_duration).ceil() as usize;

    let mut segments = Vec::with_capacity(num_segments + 1);

    // Add init segment (ftyp + moov boxes would go here)
    segments.push(Segment {
        id: format!("init_{}", quality.name()),
        sequence: 0,
        duration: 0.0,
        start_time: 0.0,
        data: bytes::Bytes::from(vec![0u8; 1024]),
        quality,
        is_init: true,
        encryption: None,
    });

    // Add media segments (moof + mdat boxes would go here)
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

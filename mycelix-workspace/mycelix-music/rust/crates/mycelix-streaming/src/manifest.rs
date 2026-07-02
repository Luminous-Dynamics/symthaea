// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manifest generation for HLS and DASH

use crate::StreamQuality;

/// Generate HLS master playlist
pub fn generate_hls_master(track_id: &str, variants: &[StreamQuality]) -> String {
    let mut playlist = String::from("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:6\n\n");

    for variant in variants {
        let bandwidth = variant.bitrate();
        playlist.push_str(&format!(
            "#EXT-X-STREAM-INF:BANDWIDTH={},CODECS=\"{}\"\n",
            bandwidth,
            variant.codec()
        ));
        playlist.push_str(&format!(
            "/api/v1/tracks/{}/stream/{}.m3u8\n\n",
            track_id,
            variant.name()
        ));
    }

    playlist
}

/// Generate HLS variant playlist
pub fn generate_hls_variant(
    track_id: &str,
    quality: &StreamQuality,
    segment_duration: f32,
    total_duration: f32,
) -> String {
    let num_segments = (total_duration / segment_duration).ceil() as u32;
    let target_duration = segment_duration.ceil() as u32;

    let mut playlist = String::from("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:6\n");
    playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", target_duration));
    playlist.push_str("#EXT-X-MEDIA-SEQUENCE:0\n");
    playlist.push_str("#EXT-X-PLAYLIST-TYPE:VOD\n");
    playlist.push_str(&format!(
        "#EXT-X-MAP:URI=\"/api/v1/tracks/{}/stream/{}/init.mp4\"\n\n",
        track_id,
        quality.name()
    ));

    for i in 0..num_segments {
        let seg_duration = if i == num_segments - 1 {
            total_duration - (i as f32 * segment_duration)
        } else {
            segment_duration
        };

        playlist.push_str(&format!("#EXTINF:{:.3},\n", seg_duration));
        playlist.push_str(&format!(
            "/api/v1/tracks/{}/stream/{}/segment_{}.ts\n",
            track_id,
            quality.name(),
            i
        ));
    }

    playlist.push_str("#EXT-X-ENDLIST\n");
    playlist
}

/// Generate DASH MPD manifest
pub fn generate_dash_mpd(
    track_id: &str,
    variants: &[StreamQuality],
    total_duration: f32,
    segment_duration: f32,
) -> String {
    let duration_iso = format!("PT{:.3}S", total_duration);
    let segment_duration_samples = (segment_duration * 44100.0) as u32;

    let mut mpd = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
     xsi:schemaLocation="urn:mpeg:dash:schema:mpd:2011 DASH-MPD.xsd"
     profiles="urn:mpeg:dash:profile:isoff-on-demand:2011"
     type="static"
     mediaPresentationDuration="{}"
     minBufferTime="PT2S">
  <Period id="0" start="PT0S">
    <AdaptationSet contentType="audio" mimeType="audio/mp4" lang="und" segmentAlignment="true">
"#,
        duration_iso
    );

    for variant in variants {
        mpd.push_str(&format!(
            r#"      <Representation id="{}" bandwidth="{}" codecs="{}" audioSamplingRate="44100">
        <SegmentTemplate timescale="44100" duration="{}"
                         initialization="/api/v1/tracks/{}/stream/{}/init.m4s"
                         media="/api/v1/tracks/{}/stream/{}/segment_$Number$.m4s"
                         startNumber="0"/>
      </Representation>
"#,
            variant.name(),
            variant.bitrate(),
            variant.codec(),
            segment_duration_samples,
            track_id,
            variant.name(),
            track_id,
            variant.name()
        ));
    }

    mpd.push_str(r#"    </AdaptationSet>
  </Period>
</MPD>"#);

    mpd
}

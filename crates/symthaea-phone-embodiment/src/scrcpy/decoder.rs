// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HEVC → RGBA decoder via `ffmpeg-next` 7.x.
//!
//! Scope: feed raw HEVC NAL units (as parsed out of the scrcpy v2.4 wire by
//! [`super::wire`]) into ffmpeg's hardware-fallback HEVC decoder, get YUV
//! frames out, swscale them to 8-bit RGBA, hand back to callers as
//! [`DecodedFrame`].
//!
//! # Why ffmpeg-next over rav1d/dav1d
//!
//! See `symthaea/docs/phase_1b_codec_probe.md` and roadmap v1.4. Short
//! version: Pixel 8 Pro on Android 16 has hardware HEVC encoding via
//! `c2.exynos.hevc.encoder` but no hardware AV1 encode. HEVC HW gives ~50%
//! better compression than H.264 — close enough to AV1's 30-40% that the
//! USB 2.0 budget relief survives the pivot. ffmpeg-next is the most
//! production-grade HEVC decoder available in the Rust ecosystem.
//!
//! # Lifecycle
//!
//! ```ignore
//! let mut dec = HevcDecoder::new()?;
//! for packet in scrcpy_stream.frames() {
//!     for frame in dec.decode_packet(&packet.payload)? {
//!         // frame.rgba is width*height*4 bytes
//!     }
//! }
//! ```
//!
//! `decode_packet` may return zero, one, or many [`DecodedFrame`]s per
//! input packet — HEVC's reorder buffer means input packets and output
//! frames are not 1:1. Callers must drain via the returned `Vec`.
//!
//! # Threading
//!
//! ffmpeg's HEVC decoder is internally multi-threaded by default. We do
//! not constrain `set_threading` here — the cognitive loop is single
//! consumer, single producer over the wire, and ffmpeg picks a sensible
//! thread count from the host CPU.

use std::sync::Once;

use ffmpeg_next as ffmpeg;

/// One swscale-converted RGBA frame ready for the vision manifold.
#[derive(Debug, Clone)]
pub struct DecodedFrame {
    pub width: u32,
    pub height: u32,
    /// Tightly packed RGBA8 — `width * height * 4` bytes. Row stride
    /// equals `width * 4` (no padding) because we configure swscale's
    /// output plane that way.
    pub rgba: Vec<u8>,
    /// Best-effort presentation timestamp in microseconds, mirrored from
    /// the upstream scrcpy frame header's `pts_micros()` when the caller
    /// supplied it. `None` for config packets and PTS-less frames.
    pub pts_micros: Option<u64>,
}

/// Errors out of the HEVC decode pipeline.
#[derive(Debug)]
pub enum DecodeError {
    /// `ffmpeg::decoder::find(HEVC)` returned None — should be impossible
    /// on any sane ffmpeg build, but worth checking explicitly.
    HevcCodecNotFound,
    /// ffmpeg returned an error from any of init / send_packet /
    /// receive_frame / swscale.
    Ffmpeg(ffmpeg::Error),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::HevcCodecNotFound => {
                write!(f, "ffmpeg HEVC decoder not found in this build")
            }
            DecodeError::Ffmpeg(e) => write!(f, "ffmpeg error: {e}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl From<ffmpeg::Error> for DecodeError {
    fn from(e: ffmpeg::Error) -> Self {
        DecodeError::Ffmpeg(e)
    }
}

/// `ffmpeg::init()` is idempotent but not free — guard it with a Once so
/// repeated `HevcDecoder::new()` calls are cheap.
static FFMPEG_INIT: Once = Once::new();

fn ensure_ffmpeg_initialized() -> Result<(), DecodeError> {
    let mut result: Result<(), DecodeError> = Ok(());
    FFMPEG_INIT.call_once(|| {
        if let Err(e) = ffmpeg::init() {
            result = Err(DecodeError::Ffmpeg(e));
        }
    });
    result
}

/// HEVC decoder owning an opened ffmpeg video decoder context and a lazy
/// swscale converter that materializes once we see the first decoded
/// frame's actual dimensions and pixel format.
pub struct HevcDecoder {
    decoder: ffmpeg::decoder::Video,
    /// Swscale context cached on (input format, input dims) — rebuilt if
    /// the upstream stream ever changes resolution mid-flight (which
    /// scrcpy v2.4 does support via its dynamic-resolution hooks).
    scaler: Option<CachedScaler>,
}

struct CachedScaler {
    in_format: ffmpeg::format::Pixel,
    in_width: u32,
    in_height: u32,
    ctx: ffmpeg::software::scaling::Context,
}

impl HevcDecoder {
    /// Build a fresh HEVC decoder. Cheap after the first call (ffmpeg
    /// init is guarded by a Once).
    ///
    /// Threading: configured for **slice-parallel** decoding via
    /// `set_threading(Type::Slice, count=0)`. count=0 lets ffmpeg pick
    /// `nproc` worker threads. `Type::Slice` parallelizes the work for
    /// a SINGLE frame across cores, so the first decoded frame comes
    /// out as soon as input is available — unlike `Type::Frame` which
    /// pipelines multiple frames in parallel and needs more input
    /// buffered before emitting anything (the I.B.6 sustain run caught
    /// that variant producing 0 frames from a 12-packet burst).
    /// Single-threaded default was ~100 ms per 720p frame on a 16-core
    /// Ryzen, capping us at 10 fps. Slice-threaded is the right tradeoff
    /// for a real-time consumer.
    pub fn new() -> Result<Self, DecodeError> {
        ensure_ffmpeg_initialized()?;
        let codec =
            ffmpeg::decoder::find(ffmpeg::codec::Id::HEVC).ok_or(DecodeError::HevcCodecNotFound)?;
        let mut context = ffmpeg::codec::Context::new_with_codec(codec);
        context.set_threading(ffmpeg::codec::threading::Config {
            kind: ffmpeg::codec::threading::Type::Slice,
            count: 0,
        });
        let decoder = context.decoder().video()?;
        Ok(Self {
            decoder,
            scaler: None,
        })
    }

    /// Feed one raw HEVC packet (NAL payload from `wire::FrameHeader`)
    /// and drain any frames the decoder is ready to release.
    ///
    /// Returns a `Vec<DecodedFrame>` because HEVC's reorder buffer makes
    /// the input/output packet ratio variable: a leading config packet
    /// produces zero frames, a keyframe right after a B-frame burst can
    /// produce several at once.
    ///
    /// `pts_micros` is the PTS the caller already extracted from the
    /// scrcpy frame header — we just copy it onto every frame this
    /// packet produces. ffmpeg has its own internal PTS plumbing but for
    /// raw NAL streams without a container it's unreliable.
    pub fn decode_packet(
        &mut self,
        nal_payload: &[u8],
        pts_micros: Option<u64>,
    ) -> Result<Vec<DecodedFrame>, DecodeError> {
        let packet = ffmpeg::Packet::copy(nal_payload);
        self.decoder.send_packet(&packet)?;
        self.drain(pts_micros)
    }

    /// Flush the decoder's reorder buffer at end-of-stream.
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>, DecodeError> {
        self.decoder.send_eof()?;
        self.drain(None)
    }

    fn drain(&mut self, pts_micros: Option<u64>) -> Result<Vec<DecodedFrame>, DecodeError> {
        let mut out = Vec::new();
        let mut yuv = ffmpeg::frame::Video::empty();
        loop {
            match self.decoder.receive_frame(&mut yuv) {
                Ok(()) => {
                    let rgba = self.convert_to_rgba(&yuv)?;
                    out.push(DecodedFrame {
                        width: yuv.width(),
                        height: yuv.height(),
                        rgba,
                        pts_micros,
                    });
                }
                // EAGAIN means "no more frames right now, send another packet"
                // EOF means "all frames drained after send_eof"
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::util::error::EAGAIN => {
                    break;
                }
                Err(ffmpeg::Error::Eof) => break,
                Err(e) => return Err(DecodeError::Ffmpeg(e)),
            }
        }
        Ok(out)
    }

    fn convert_to_rgba(&mut self, yuv: &ffmpeg::frame::Video) -> Result<Vec<u8>, DecodeError> {
        let in_format = yuv.format();
        let in_width = yuv.width();
        let in_height = yuv.height();

        // Rebuild the scaler if format or dimensions changed.
        let needs_rebuild = match &self.scaler {
            Some(s) => {
                s.in_format != in_format || s.in_width != in_width || s.in_height != in_height
            }
            None => true,
        };
        if needs_rebuild {
            let ctx = ffmpeg::software::scaling::Context::get(
                in_format,
                in_width,
                in_height,
                ffmpeg::format::Pixel::RGBA,
                in_width,
                in_height,
                ffmpeg::software::scaling::Flags::BILINEAR,
            )?;
            self.scaler = Some(CachedScaler {
                in_format,
                in_width,
                in_height,
                ctx,
            });
        }

        let scaler = self.scaler.as_mut().expect("scaler set above");

        let mut rgba_frame = ffmpeg::frame::Video::empty();
        scaler.ctx.run(yuv, &mut rgba_frame)?;

        // Plane 0 of an RGBA frame is the packed RGBA buffer. Stride may
        // be > width*4 if ffmpeg added padding; copy row-by-row to get a
        // tightly-packed output.
        let stride = rgba_frame.stride(0);
        let row_bytes = (in_width as usize) * 4;
        let plane = rgba_frame.data(0);
        let mut out = Vec::with_capacity(row_bytes * in_height as usize);
        for row in 0..in_height as usize {
            let start = row * stride;
            out.extend_from_slice(&plane[start..start + row_bytes]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constructor test: proves ffmpeg init succeeds, the HEVC codec is
    /// available in this build, and the decoder context opens without
    /// error. This is the minimum viable smoke test we can run without
    /// real HEVC bitstream data.
    #[test]
    fn hevc_decoder_constructs() {
        let dec = HevcDecoder::new();
        assert!(
            dec.is_ok(),
            "HevcDecoder::new() failed — ffmpeg init or HEVC codec lookup broken: {:?}",
            dec.err()
        );
    }

    #[test]
    fn ffmpeg_init_is_idempotent() {
        // Two constructors back-to-back must both succeed; the Once guard
        // ensures the second call doesn't double-init ffmpeg.
        let _ = HevcDecoder::new().unwrap();
        let _ = HevcDecoder::new().unwrap();
    }

    #[test]
    fn empty_packet_produces_no_frames() {
        // Feeding an empty packet to a fresh decoder should not panic and
        // should not emit any frames (the decoder needs at least a config
        // packet's worth of NAL data before it can produce output).
        let mut dec = HevcDecoder::new().unwrap();
        // ffmpeg may legitimately return AVERROR(EINVAL) on a zero-byte
        // packet — that's an Err from send_packet, which is the correct
        // behavior for malformed input. We only assert the decoder
        // doesn't panic; either Ok([]) or Err is acceptable here.
        let _ = dec.decode_packet(&[], None);
    }

    /// End-to-end offline decode test: feed the real recorded wire bytes
    /// from `tests/data/sample.hevc.wire` through `wire::parse_frame_header`
    /// and `HevcDecoder::decode_packet`, assert at least one decoded RGBA
    /// frame comes out with the expected dimensions.
    ///
    /// The asset is captured from a live Pixel 8 Pro by
    /// `examples/record_scrcpy_sample.rs`. Its layout matches what
    /// `ScrcpyCaptureStream::next_frame` consumes from the wire, minus
    /// the 64-byte device-meta block and 12-byte video header (which the
    /// recorder strips before saving). Each entry is:
    ///
    /// ```text
    /// [12-byte FrameHeader][packet_size bytes of HEVC NAL]
    /// ```
    ///
    /// The asset contains 10 frames: 1 config (SPS/PPS), 1 keyframe, 8
    /// P-frames. The decoder should produce ≥ 1 RGBA frame from the
    /// keyframe + P-frames after the config packet primes it.
    #[test]
    fn end_to_end_decode_recorded_wire_asset() {
        use crate::scrcpy::wire;

        const ASSET: &[u8] = include_bytes!("../../tests/data/sample.hevc.wire");

        let mut dec = HevcDecoder::new().expect("HevcDecoder::new");
        let mut cursor = 0usize;
        let mut frames_in: usize = 0;
        let mut frames_out: Vec<DecodedFrame> = Vec::new();
        let mut config_count = 0;
        let mut keyframe_count = 0;

        while cursor + wire::FRAME_HEADER_LEN <= ASSET.len() {
            let header =
                wire::parse_frame_header(&ASSET[cursor..]).expect("parse_frame_header on asset");
            cursor += wire::FRAME_HEADER_LEN;
            let payload_end = cursor + header.packet_size as usize;
            assert!(
                payload_end <= ASSET.len(),
                "packet_size {} overruns asset (cursor={cursor}, len={})",
                header.packet_size,
                ASSET.len()
            );
            let payload = &ASSET[cursor..payload_end];
            cursor = payload_end;

            if header.is_config() {
                config_count += 1;
            }
            if header.is_key_frame() {
                keyframe_count += 1;
            }
            frames_in += 1;

            let produced = dec
                .decode_packet(payload, header.pts_micros())
                .expect("decode_packet on real HEVC bytes");
            frames_out.extend(produced);
        }

        // Drain any frames left in the HEVC reorder buffer at EOS.
        let flushed = dec.flush().expect("flush");
        frames_out.extend(flushed);

        // Sanity: we processed all 10 wire entries from the recorder run.
        assert_eq!(
            frames_in, 10,
            "expected 10 wire frames in asset, got {frames_in}"
        );
        assert_eq!(
            cursor,
            ASSET.len(),
            "decoder did not consume the entire asset"
        );
        assert!(
            config_count >= 1,
            "asset must contain at least one config packet (SPS/PPS), got {config_count}"
        );
        assert!(
            keyframe_count >= 1,
            "asset must contain at least one keyframe, got {keyframe_count}"
        );

        // The full vertical produced at least one decoded RGBA frame.
        // We don't assert an exact count because HEVC's reorder buffer
        // makes the in/out ratio variable, but the keyframe + at least
        // some P-frames must round-trip.
        assert!(
            !frames_out.is_empty(),
            "decoder produced 0 frames from {frames_in} input packets ({config_count} config, {keyframe_count} keyframe)"
        );

        // Every produced frame must be a coherent RGBA8 image.
        for (i, f) in frames_out.iter().enumerate() {
            assert!(
                f.width > 0 && f.height > 0,
                "frame {i} has zero dimensions: {}x{}",
                f.width,
                f.height
            );
            let expected = (f.width as usize) * (f.height as usize) * 4;
            assert_eq!(
                f.rgba.len(),
                expected,
                "frame {i} has {}-byte buffer, expected {expected} for {}x{} RGBA8",
                f.rgba.len(),
                f.width,
                f.height
            );
            // PTS is sourced from the wire frame headers — the asset
            // recorder shows non-zero PTS on every non-config packet, so
            // we should see at least one Some(_) here.
            // (Allowed to be None on config-derived frames if HEVC ever
            // produces them, which it usually doesn't.)
        }

        // First decoded frame should match the encoder's output dimensions
        // from the recorder: 328x720 (the device side picked these for
        // max_size=720 on a portrait-oriented Pixel 8 Pro).
        let f0 = &frames_out[0];
        assert_eq!(
            (f0.width, f0.height),
            (328, 720),
            "expected first decoded frame at 328x720 (recorded asset's encoder picked these); got {}x{}",
            f0.width,
            f0.height
        );
    }

    #[test]
    fn decoded_frame_struct_layout() {
        // Sanity check on the public DecodedFrame surface — we don't want
        // to silently drop the pts_micros field in a future refactor.
        let f = DecodedFrame {
            width: 1920,
            height: 1080,
            rgba: vec![0u8; 1920 * 1080 * 4],
            pts_micros: Some(33_333),
        };
        assert_eq!(f.rgba.len(), (f.width * f.height * 4) as usize);
        assert_eq!(f.pts_micros, Some(33_333));
    }
}

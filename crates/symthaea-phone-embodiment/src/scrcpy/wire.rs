// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! scrcpy v2.4 binary wire-format parser.
//!
//! Pure data layer: every function in this module takes `&[u8]` and returns
//! parsed structs. There is no I/O. The connector that actually pulls bytes
//! from the TCP socket lives in [`super`] (the `mod.rs` lifecycle layer).
//!
//! # Wire layout (matches scrcpy v2.4 server with `send_device_meta=true`,
//! # `send_frame_meta=true`, `tunnel_forward=false`)
//!
//! Stream begins with the **device metadata block** (one-shot):
//!
//! ```text
//! +----------------------------------------+
//! | device_name : 64 bytes UTF-8 NUL-pad   |
//! +----------------------------------------+
//! ```
//!
//! Then the **video header** (one-shot, after device meta):
//!
//! ```text
//! +-------------+-------+--------+
//! | codec_id    | width | height |
//! | 4 BE bytes  | 4 BE  | 4 BE   |
//! +-------------+-------+--------+
//! ```
//!
//! `codec_id` is a fourcc literal: `b"h265"` for HEVC (the v1.4 primary),
//! `b"h264"` for H.264, `b"av01"` for AV1.
//!
//! Then a stream of **frame packets**, each being a 12-byte header followed
//! by a variable-length payload:
//!
//! ```text
//! +---------------------+-------------+-------------------+
//! | pts_with_flags      | packet_size | payload (NAL data)|
//! | 8 BE bytes          | 4 BE bytes  | packet_size bytes |
//! +---------------------+-------------+-------------------+
//! ```
//!
//! `pts_with_flags` packs three things into a u64:
//! - bit 63: [`PACKET_FLAG_CONFIG`] — payload is codec config (e.g. SPS/PPS
//!   for HEVC). Always also has the PTS field set to [`NO_PTS`].
//! - bit 62: [`PACKET_FLAG_KEY_FRAME`] — payload is a keyframe (IDR for
//!   HEVC).
//! - bits 0–61: presentation timestamp in microseconds, OR the [`NO_PTS`]
//!   sentinel meaning "no usable PTS, treat as side-channel data."
//!
//! References:
//! - scrcpy `app/src/main/java/com/genymobile/scrcpy/video/SurfaceEncoder.java`
//! - scrcpy `app/src/main/java/com/genymobile/scrcpy/util/IO.java`
//! - scrcpy `app/scrcpy.h` constants `SC_PACKET_FLAG_*`, `SC_PACKET_NO_PTS`

/// Length of the device-name block prepended once at stream start.
pub const DEVICE_NAME_LEN: usize = 64;

/// Length of the per-stream video header (codec_id + width + height).
pub const VIDEO_HEADER_LEN: usize = 12;

/// Length of the per-frame header (pts_with_flags + packet_size).
pub const FRAME_HEADER_LEN: usize = 12;

/// PTS-or-flags top bit: payload is a config packet.
pub const PACKET_FLAG_CONFIG: u64 = 1u64 << 63;

/// PTS-or-flags second bit: payload is a keyframe.
pub const PACKET_FLAG_KEY_FRAME: u64 = 1u64 << 62;

/// Mask isolating the PTS field (lower 62 bits) from the flag bits.
pub const PTS_MASK: u64 = (1u64 << 62) - 1;

/// Sentinel value meaning "no usable PTS" — used by config packets and
/// side-channel data. Equal to [`PTS_MASK`] after the flag bits are
/// stripped, so detection is `(pts_with_flags & PTS_MASK) == NO_PTS`.
pub const NO_PTS: u64 = PTS_MASK;

/// Parse errors from the binary framing layer.
///
/// Stringly-typed on purpose — see the design note on `super::ScrcpyError`.
#[derive(Debug, PartialEq, Eq)]
pub enum WireError {
    /// Caller handed in fewer bytes than the requested header needs.
    ShortRead { expected: usize, got: usize },
    /// `codec_id` fourcc isn't one we understand.
    UnknownCodec([u8; 4]),
    /// Width or height is zero — the encoder is misbehaving.
    ZeroDimensions { width: u32, height: u32 },
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::ShortRead { expected, got } => {
                write!(
                    f,
                    "scrcpy wire short read: need {expected} bytes, have {got}"
                )
            }
            WireError::UnknownCodec(b) => {
                write!(
                    f,
                    "scrcpy wire unknown codec_id: {:?} ({})",
                    b,
                    String::from_utf8_lossy(b)
                )
            }
            WireError::ZeroDimensions { width, height } => {
                write!(f, "scrcpy wire zero dimensions: {width}x{height}")
            }
        }
    }
}

impl std::error::Error for WireError {}

/// Codec identifier as parsed from the 4-byte fourcc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireCodec {
    H265,
    H264,
    Av1,
}

impl WireCodec {
    /// Parse a 4-byte fourcc. Returns the matching variant or
    /// [`WireError::UnknownCodec`] for unrecognized identifiers.
    pub fn from_fourcc(fourcc: [u8; 4]) -> Result<Self, WireError> {
        match &fourcc {
            b"h265" => Ok(WireCodec::H265),
            b"h264" => Ok(WireCodec::H264),
            b"av01" => Ok(WireCodec::Av1),
            _ => Err(WireError::UnknownCodec(fourcc)),
        }
    }
}

/// Decoded device metadata block (the first 64 bytes of the stream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMeta {
    pub name: String,
}

/// Parse a 64-byte device-meta block into a trimmed UTF-8 name.
///
/// Trailing NUL bytes are stripped. Non-UTF-8 byte sequences are
/// lossily replaced — the device name is informational only and
/// shouldn't crash the stream over an exotic locale.
pub fn parse_device_meta(bytes: &[u8]) -> Result<DeviceMeta, WireError> {
    if bytes.len() < DEVICE_NAME_LEN {
        return Err(WireError::ShortRead {
            expected: DEVICE_NAME_LEN,
            got: bytes.len(),
        });
    }
    let raw = &bytes[..DEVICE_NAME_LEN];
    let trimmed = match raw.iter().position(|&b| b == 0) {
        Some(i) => &raw[..i],
        None => raw,
    };
    Ok(DeviceMeta {
        name: String::from_utf8_lossy(trimmed).into_owned(),
    })
}

/// Decoded video header (12 bytes after the device-meta block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoHeader {
    pub codec: WireCodec,
    pub width: u32,
    pub height: u32,
}

/// Parse the 12-byte video header.
pub fn parse_video_header(bytes: &[u8]) -> Result<VideoHeader, WireError> {
    if bytes.len() < VIDEO_HEADER_LEN {
        return Err(WireError::ShortRead {
            expected: VIDEO_HEADER_LEN,
            got: bytes.len(),
        });
    }
    let mut fourcc = [0u8; 4];
    fourcc.copy_from_slice(&bytes[0..4]);
    let codec = WireCodec::from_fourcc(fourcc)?;
    let width = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let height = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if width == 0 || height == 0 {
        return Err(WireError::ZeroDimensions { width, height });
    }
    Ok(VideoHeader {
        codec,
        width,
        height,
    })
}

/// Decoded per-frame packet header.
///
/// The flag bits are surfaced via [`is_config`](Self::is_config) and
/// [`is_key_frame`](Self::is_key_frame). The PTS is surfaced as
/// `Option<u64>` — `None` means the [`NO_PTS`] sentinel was observed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHeader {
    /// Raw pts_with_flags word as it came off the wire. Useful for
    /// re-emitting the header on a forwarding path without re-encoding.
    pub raw_pts_with_flags: u64,
    /// Payload length in bytes. The next `packet_size` bytes after the
    /// 12-byte header form the NAL payload.
    pub packet_size: u32,
}

impl FrameHeader {
    /// Whether bit 63 (config-packet flag) is set.
    pub fn is_config(&self) -> bool {
        (self.raw_pts_with_flags & PACKET_FLAG_CONFIG) != 0
    }

    /// Whether bit 62 (keyframe flag) is set.
    pub fn is_key_frame(&self) -> bool {
        (self.raw_pts_with_flags & PACKET_FLAG_KEY_FRAME) != 0
    }

    /// PTS in microseconds, or `None` if the [`NO_PTS`] sentinel was sent.
    pub fn pts_micros(&self) -> Option<u64> {
        let masked = self.raw_pts_with_flags & PTS_MASK;
        if masked == NO_PTS { None } else { Some(masked) }
    }
}

/// Parse a 12-byte frame header. Does not consume the payload.
pub fn parse_frame_header(bytes: &[u8]) -> Result<FrameHeader, WireError> {
    if bytes.len() < FRAME_HEADER_LEN {
        return Err(WireError::ShortRead {
            expected: FRAME_HEADER_LEN,
            got: bytes.len(),
        });
    }
    let raw_pts_with_flags = u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    let packet_size = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    Ok(FrameHeader {
        raw_pts_with_flags,
        packet_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- device meta ---

    #[test]
    fn device_meta_strips_trailing_nuls() {
        let mut buf = [0u8; DEVICE_NAME_LEN];
        let name = b"Pixel 8 Pro";
        buf[..name.len()].copy_from_slice(name);
        // remaining bytes already 0
        let meta = parse_device_meta(&buf).unwrap();
        assert_eq!(meta.name, "Pixel 8 Pro");
    }

    #[test]
    fn device_meta_handles_full_64_byte_name() {
        let buf: [u8; DEVICE_NAME_LEN] = [b'A'; DEVICE_NAME_LEN];
        let meta = parse_device_meta(&buf).unwrap();
        assert_eq!(meta.name.len(), 64);
        assert!(meta.name.chars().all(|c| c == 'A'));
    }

    #[test]
    fn device_meta_short_read_errors() {
        let buf = [0u8; 32];
        let err = parse_device_meta(&buf).unwrap_err();
        assert_eq!(
            err,
            WireError::ShortRead {
                expected: 64,
                got: 32
            }
        );
    }

    #[test]
    fn device_meta_lossy_utf8() {
        let mut buf = [0u8; DEVICE_NAME_LEN];
        // 0xff is not valid UTF-8 — must not panic, must produce *something*
        buf[0] = 0xff;
        buf[1] = b'X';
        let meta = parse_device_meta(&buf).unwrap();
        assert!(meta.name.contains('X'));
    }

    // --- video header ---

    #[test]
    fn video_header_h265_2992x1344() {
        // codec_id="h265" then width=2992 (0x0BB0) then height=1344 (0x0540) BE
        let mut buf = [0u8; VIDEO_HEADER_LEN];
        buf[0..4].copy_from_slice(b"h265");
        buf[4..8].copy_from_slice(&2992u32.to_be_bytes());
        buf[8..12].copy_from_slice(&1344u32.to_be_bytes());
        let hdr = parse_video_header(&buf).unwrap();
        assert_eq!(hdr.codec, WireCodec::H265);
        assert_eq!(hdr.width, 2992);
        assert_eq!(hdr.height, 1344);
    }

    #[test]
    fn video_header_h264_1080x1920() {
        let mut buf = [0u8; VIDEO_HEADER_LEN];
        buf[0..4].copy_from_slice(b"h264");
        buf[4..8].copy_from_slice(&1080u32.to_be_bytes());
        buf[8..12].copy_from_slice(&1920u32.to_be_bytes());
        let hdr = parse_video_header(&buf).unwrap();
        assert_eq!(hdr.codec, WireCodec::H264);
        assert_eq!(hdr.width, 1080);
        assert_eq!(hdr.height, 1920);
    }

    #[test]
    fn video_header_av01_recognized() {
        let mut buf = [0u8; VIDEO_HEADER_LEN];
        buf[0..4].copy_from_slice(b"av01");
        buf[4..8].copy_from_slice(&720u32.to_be_bytes());
        buf[8..12].copy_from_slice(&1280u32.to_be_bytes());
        let hdr = parse_video_header(&buf).unwrap();
        assert_eq!(hdr.codec, WireCodec::Av1);
    }

    #[test]
    fn video_header_unknown_codec_rejected() {
        let mut buf = [0u8; VIDEO_HEADER_LEN];
        buf[0..4].copy_from_slice(b"vp09");
        buf[4..8].copy_from_slice(&720u32.to_be_bytes());
        buf[8..12].copy_from_slice(&1280u32.to_be_bytes());
        let err = parse_video_header(&buf).unwrap_err();
        assert!(matches!(err, WireError::UnknownCodec(c) if &c == b"vp09"));
    }

    #[test]
    fn video_header_zero_width_rejected() {
        let mut buf = [0u8; VIDEO_HEADER_LEN];
        buf[0..4].copy_from_slice(b"h265");
        buf[4..8].copy_from_slice(&0u32.to_be_bytes());
        buf[8..12].copy_from_slice(&1344u32.to_be_bytes());
        let err = parse_video_header(&buf).unwrap_err();
        assert_eq!(
            err,
            WireError::ZeroDimensions {
                width: 0,
                height: 1344,
            }
        );
    }

    #[test]
    fn video_header_short_read_errors() {
        let buf = [0u8; 8];
        let err = parse_video_header(&buf).unwrap_err();
        assert_eq!(
            err,
            WireError::ShortRead {
                expected: 12,
                got: 8
            }
        );
    }

    // --- frame header ---

    fn build_frame_header(pts_with_flags: u64, size: u32) -> [u8; FRAME_HEADER_LEN] {
        let mut buf = [0u8; FRAME_HEADER_LEN];
        buf[0..8].copy_from_slice(&pts_with_flags.to_be_bytes());
        buf[8..12].copy_from_slice(&size.to_be_bytes());
        buf
    }

    #[test]
    fn frame_header_normal_keyframe() {
        // pts = 1_000_000 us (1 second), keyframe flag set, no config flag
        let pts_with_flags = PACKET_FLAG_KEY_FRAME | 1_000_000u64;
        let buf = build_frame_header(pts_with_flags, 524_288);
        let hdr = parse_frame_header(&buf).unwrap();
        assert_eq!(hdr.packet_size, 524_288);
        assert!(hdr.is_key_frame());
        assert!(!hdr.is_config());
        assert_eq!(hdr.pts_micros(), Some(1_000_000));
    }

    #[test]
    fn frame_header_config_packet_no_pts() {
        // config + NO_PTS sentinel
        let pts_with_flags = PACKET_FLAG_CONFIG | NO_PTS;
        let buf = build_frame_header(pts_with_flags, 32);
        let hdr = parse_frame_header(&buf).unwrap();
        assert!(hdr.is_config());
        assert!(!hdr.is_key_frame());
        assert_eq!(hdr.pts_micros(), None);
        assert_eq!(hdr.packet_size, 32);
    }

    #[test]
    fn frame_header_normal_frame_no_flags() {
        // Plain inter-frame, PTS = 33_333 us (~30 fps frame 1)
        let pts_with_flags = 33_333u64;
        let buf = build_frame_header(pts_with_flags, 4096);
        let hdr = parse_frame_header(&buf).unwrap();
        assert!(!hdr.is_config());
        assert!(!hdr.is_key_frame());
        assert_eq!(hdr.pts_micros(), Some(33_333));
    }

    #[test]
    fn frame_header_keyframe_with_real_pts_after_2_seconds() {
        // Realistic Pixel 8 Pro burst: 2 s in, keyframe with valid PTS
        let pts_us = 2_000_000u64;
        let pts_with_flags = PACKET_FLAG_KEY_FRAME | pts_us;
        let buf = build_frame_header(pts_with_flags, 1_048_576);
        let hdr = parse_frame_header(&buf).unwrap();
        assert!(hdr.is_key_frame());
        assert!(!hdr.is_config());
        assert_eq!(hdr.pts_micros(), Some(2_000_000));
    }

    #[test]
    fn frame_header_short_read_errors() {
        let buf = [0u8; 4];
        let err = parse_frame_header(&buf).unwrap_err();
        assert_eq!(
            err,
            WireError::ShortRead {
                expected: 12,
                got: 4
            }
        );
    }

    #[test]
    fn pts_mask_constants_self_consistent() {
        // PACKET_FLAG_CONFIG and PACKET_FLAG_KEY_FRAME must be the only
        // bits outside PTS_MASK.
        let flags = PACKET_FLAG_CONFIG | PACKET_FLAG_KEY_FRAME;
        assert_eq!(flags & PTS_MASK, 0);
        assert_eq!(flags | PTS_MASK, u64::MAX);
        // NO_PTS is exactly PTS_MASK by construction.
        assert_eq!(NO_PTS, PTS_MASK);
    }
}

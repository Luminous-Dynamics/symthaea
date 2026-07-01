// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end scrcpy capture stream — the vertical that combines lifecycle,
//! TCP transport, [`super::wire`] parsing, and [`super::decoder`] HEVC decode
//! into one consumable that hands back ready-to-use RGBA frames.
//!
//! # Lifecycle
//!
//! ```ignore
//! use std::path::Path;
//! use symthaea_phone_embodiment::scrcpy::{stream::ScrcpyCaptureStream, ScrcpyOptions};
//!
//! let jar = Path::new("crates/symthaea-phone-embodiment/vendor/scrcpy-server-v2.4.jar");
//! let opts = ScrcpyOptions::cybernetic_defaults("41201FDJG000UM", 8400);
//! let mut stream = ScrcpyCaptureStream::launch(jar, &opts)?;
//!
//! println!("Capturing {}x{} from {}",
//!          stream.video_header().width,
//!          stream.video_header().height,
//!          stream.device_meta().name);
//!
//! while let Some(frame) = stream.next_frame()? {
//!     // frame.rgba is width*height*4 bytes, ready for the vision manifold
//! }
//! ```
//!
//! On `Drop`, the underlying [`super::ScrcpyHandle`] kills the device-side
//! `app_process` and removes the `adb reverse` tunnel — no manual cleanup.
//!
//! # Threading model
//!
//! Synchronous, single-threaded. The cognitive loop already runs sync
//! against `PhoneBridge`, so this matches. A read timeout (default 100 ms)
//! prevents `next_frame()` from blocking the cognitive cycle forever if
//! the device stalls.
//!
//! Async/tokio adoption is the Phase I.C work (the QUIC transport swap)
//! and lives in a separate crate boundary.

use std::collections::VecDeque;
use std::io::{self, ErrorKind, Read};
use std::net::TcpStream;
use std::path::Path;
use std::time::Duration;

use super::decoder::{DecodeError, DecodedFrame, HevcDecoder};
use super::wire::{self, FrameHeader, WireCodec, WireError};
use super::{
    ScrcpyError, ScrcpyHandle, ScrcpyOptions, VideoCodec, accept_from_server, bind_host_listener,
    start_scrcpy,
};

/// Default read timeout for [`ScrcpyCaptureStream::next_frame`].
///
/// 500 ms is generous enough that a momentary device hiccup on USB
/// doesn't trip a false timeout, while still short enough that a stalled
/// device shows up within ~2 cognitive cycles. The original 100 ms was
/// too tight: the recorder example empirically needed 500 ms to read
/// back-to-back frame headers on the Pixel 8 Pro USB link without
/// hitting `EWOULDBLOCK` on partial reads. Phase I.B.6 sustain run
/// confirmed 100 ms produced 0 frames over 60 s.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_millis(500);

/// Errors flowing out of the end-to-end capture stream.
#[derive(Debug)]
pub enum StreamError {
    /// Lifecycle layer failed (JAR push, reverse tunnel, app_process spawn).
    Lifecycle(ScrcpyError),
    /// Wire-format parser rejected a header.
    Wire(WireError),
    /// HEVC decoder returned an error.
    Decode(DecodeError),
    /// I/O error on the TCP socket.
    Io(io::Error),
    /// The codec the device actually sent doesn't match what we requested.
    /// This catches the case where scrcpy silently fell back to a different
    /// encoder (which it does on some Android builds when the requested
    /// codec init fails).
    CodecMismatch {
        expected: WireCodec,
        actual: WireCodec,
    },
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Lifecycle(e) => write!(f, "scrcpy lifecycle: {e}"),
            StreamError::Wire(e) => write!(f, "scrcpy wire: {e}"),
            StreamError::Decode(e) => write!(f, "scrcpy decode: {e}"),
            StreamError::Io(e) => write!(f, "scrcpy I/O: {e}"),
            StreamError::CodecMismatch { expected, actual } => write!(
                f,
                "scrcpy codec mismatch: requested {expected:?}, device sent {actual:?}"
            ),
        }
    }
}

impl std::error::Error for StreamError {}

impl From<ScrcpyError> for StreamError {
    fn from(e: ScrcpyError) -> Self {
        StreamError::Lifecycle(e)
    }
}

impl From<WireError> for StreamError {
    fn from(e: WireError) -> Self {
        StreamError::Wire(e)
    }
}

impl From<DecodeError> for StreamError {
    fn from(e: DecodeError) -> Self {
        StreamError::Decode(e)
    }
}

impl From<io::Error> for StreamError {
    fn from(e: io::Error) -> Self {
        StreamError::Io(e)
    }
}

/// End-to-end capture: device-side scrcpy-server → reverse tunnel → wire
/// parser → HEVC decoder → packed RGBA frames.
///
/// Owns the [`ScrcpyHandle`] so cleanup of the device-side process and the
/// `adb reverse` tunnel happens automatically on `Drop`.
pub struct ScrcpyCaptureStream {
    /// RAII guard for the device-side server process and reverse tunnel.
    /// Field order matters: dropped before `tcp` so server tear-down can
    /// signal the kernel before the local socket closes.
    _handle: ScrcpyHandle,
    /// Connection to the host-side reverse-tunnel TCP port.
    tcp: TcpStream,
    /// Lazy HEVC decoder. Cached scaler rebuilds on dimension changes.
    decoder: HevcDecoder,
    /// Device metadata read once during the initial handshake.
    device_meta: wire::DeviceMeta,
    /// Video header read once during the initial handshake.
    video_header: wire::VideoHeader,
    /// Drained frames waiting for the caller. The decoder may produce
    /// multiple frames per input packet because of HEVC's reorder buffer.
    pending: VecDeque<DecodedFrame>,
    /// Cumulative HEVC payload bytes pulled off the wire (header bytes
    /// excluded). The Phase I.B.6 observability harness reads this for
    /// its "sealed bytes/sec" metric — distinct from the size of the
    /// decoded RGBA buffers, which is constant per frame.
    wire_bytes_total: u64,
    /// Cumulative wire-frame count (1 per packet read off TCP, distinct
    /// from `DecodedFrame` count which depends on HEVC reorder buffer).
    wire_packets_total: u64,
}

impl ScrcpyCaptureStream {
    /// Full handshake: spawn the server, wait for it, connect TCP, read
    /// device meta + video header, build the decoder.
    pub fn launch(jar_path: &Path, opts: &ScrcpyOptions) -> Result<Self, StreamError> {
        Self::launch_with_timeout(jar_path, opts, DEFAULT_READ_TIMEOUT)
    }

    /// As [`launch`](Self::launch) but with an explicit per-frame read
    /// timeout. Use this when running on a slow link where 100 ms isn't
    /// enough headroom for a frame round-trip.
    pub fn launch_with_timeout(
        jar_path: &Path,
        opts: &ScrcpyOptions,
        read_timeout: Duration,
    ) -> Result<Self, StreamError> {
        // 1. Bind the HOST listener BEFORE starting the server. scrcpy v2
        //    in tunnel_forward=false mode has the device-side server
        //    *connect* to its local abstract socket, which adb reverse
        //    bridges into our host listener. If we don't bind first, the
        //    server's connect attempt fires before we're ready.
        let listener = bind_host_listener(opts.tcp_port)?;
        // 2. Spawn the device-side server (this pushes the JAR with SHA
        //    verification, opens the reverse tunnel, then spawns the JVM).
        let handle = start_scrcpy(jar_path, opts)?;
        // 3. Wait for the server to call connect() on its local abstract
        //    socket. Default 5s is generous for USB; LAN/WAN may need more.
        let mut tcp = accept_from_server(&listener, Duration::from_secs(5))?;
        // Force blocking mode again — on Linux, accept() on a non-blocking
        // listener (which `accept_from_server` uses internally for its
        // poll-with-deadline implementation) can produce sockets that
        // inherit `O_NONBLOCK` despite the caller setting it back to
        // false on the listener. Belt and suspenders: re-assert blocking
        // on the actual stream we're returning. Without this, every
        // `read_exact` inside `next_frame` returns `WouldBlock`
        // immediately and the harness spins at ~600k ops/sec collecting
        // zero frames (Phase I.B.6 caught this on the first run).
        tcp.set_nonblocking(false)?;
        tcp.set_read_timeout(Some(read_timeout))?;
        tcp.set_nodelay(true)?;
        // (No dummy-byte read: cybernetic_defaults intentionally omits
        // send_dummy_byte because v2.4 only sends it on the audio socket,
        // and we run with audio=false. Reading one here would shift the
        // entire stream by 1 byte and corrupt the device-meta block.)
        // 4. Read the one-shot device-meta block (64 bytes).
        let mut dm_buf = [0u8; wire::DEVICE_NAME_LEN];
        tcp.read_exact(&mut dm_buf)?;
        let device_meta = wire::parse_device_meta(&dm_buf)?;
        // 5. Read the one-shot video header (12 bytes).
        let mut vh_buf = [0u8; wire::VIDEO_HEADER_LEN];
        tcp.read_exact(&mut vh_buf)?;
        let video_header = wire::parse_video_header(&vh_buf)?;
        // 6. Verify the device honored our codec request — catches the
        //    fallback case where a device with the requested codec
        //    misconfigured silently picks a different encoder.
        let expected_wire = wire_codec_for(opts.codec);
        if video_header.codec != expected_wire {
            return Err(StreamError::CodecMismatch {
                expected: expected_wire,
                actual: video_header.codec,
            });
        }
        // 7. Build the HEVC decoder. (Note: today this is hardcoded to HEVC
        //    because the v1.4 codec primary is HEVC. When the fallback
        //    ladder fires for H.264 / AV1, the decoder construction
        //    branches on `video_header.codec` — left as a Phase I.B.4.5
        //    follow-up since those are gated behind `h264-fallback` /
        //    `av1-research` features that aren't in the default scrcpy
        //    build path.)
        let decoder = HevcDecoder::new()?;
        Ok(Self {
            _handle: handle,
            tcp,
            decoder,
            device_meta,
            video_header,
            pending: VecDeque::new(),
            wire_bytes_total: 0,
            wire_packets_total: 0,
        })
    }

    /// Cumulative HEVC payload bytes read off the wire since launch.
    /// The "sealed bytes/sec" metric in the v1.1 roadmap observability
    /// spec is the derivative of this counter.
    pub fn wire_bytes_total(&self) -> u64 {
        self.wire_bytes_total
    }

    /// Cumulative wire-packet count (1 per scrcpy frame header read off
    /// TCP, distinct from decoded frame count because the HEVC reorder
    /// buffer can output 0..N frames per input packet).
    pub fn wire_packets_total(&self) -> u64 {
        self.wire_packets_total
    }

    /// Device name as sent in the initial 64-byte device-meta block.
    pub fn device_meta(&self) -> &wire::DeviceMeta {
        &self.device_meta
    }

    /// Codec, width, height as sent in the 12-byte video header.
    pub fn video_header(&self) -> &wire::VideoHeader {
        &self.video_header
    }

    /// Pull the next decoded RGBA frame, blocking up to the read timeout.
    ///
    /// Returns:
    /// - `Ok(Some(frame))` — a frame is available (either freshly decoded
    ///   or already in the reorder-buffer queue from a prior call).
    /// - `Ok(None)` — read timed out OR the stream cleanly closed. The
    ///   caller should retry on a future cognitive cycle, or treat
    ///   sustained `None`s as a stalled device.
    /// - `Err(_)` — protocol error, decode error, or unrecoverable I/O.
    ///
    /// Internally this loops: if the reorder buffer already has frames,
    /// it returns one immediately. Otherwise it reads the next 12-byte
    /// frame header + payload from the wire, feeds the payload to the
    /// HEVC decoder, drains any newly produced frames into the queue, and
    /// returns one. A single wire packet may produce zero, one, or many
    /// frames depending on HEVC reorder buffer state.
    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>, StreamError> {
        loop {
            if let Some(frame) = self.pending.pop_front() {
                return Ok(Some(frame));
            }
            match self.read_one_packet()? {
                None => return Ok(None),
                Some((header, payload)) => {
                    let decoded = self.decoder.decode_packet(&payload, header.pts_micros())?;
                    for f in decoded {
                        self.pending.push_back(f);
                    }
                    // loop — if pending is now non-empty we'll pop on the
                    // next iteration; if not, we'll read another packet.
                }
            }
        }
    }

    /// Read one [header + payload] pair off the wire.
    ///
    /// Returns `Ok(None)` cleanly on timeout or EOF so [`next_frame`]
    /// can surface "no frame yet" without losing the distinction from
    /// hard errors.
    fn read_one_packet(&mut self) -> Result<Option<(FrameHeader, Vec<u8>)>, StreamError> {
        let mut hdr_buf = [0u8; wire::FRAME_HEADER_LEN];
        match self.tcp.read_exact(&mut hdr_buf) {
            Ok(()) => {}
            Err(e) if is_timeout_or_eof(&e) => return Ok(None),
            Err(e) => return Err(StreamError::Io(e)),
        }
        let header = wire::parse_frame_header(&hdr_buf)?;
        let mut payload = vec![0u8; header.packet_size as usize];
        match self.tcp.read_exact(&mut payload) {
            Ok(()) => {
                self.wire_bytes_total += header.packet_size as u64;
                self.wire_packets_total += 1;
                Ok(Some((header, payload)))
            }
            Err(e) if is_timeout_or_eof(&e) => Ok(None),
            Err(e) => Err(StreamError::Io(e)),
        }
    }

    /// Drain any frames left in the HEVC reorder buffer at end-of-stream.
    ///
    /// Call once after `next_frame()` returns `Ok(None)` for a sustained
    /// number of cycles and the caller has decided the stream is done.
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>, StreamError> {
        Ok(self.decoder.flush()?)
    }
}

fn wire_codec_for(opt: VideoCodec) -> WireCodec {
    match opt {
        VideoCodec::H265 => WireCodec::H265,
        VideoCodec::H264 => WireCodec::H264,
        VideoCodec::Av1 => WireCodec::Av1,
    }
}

fn is_timeout_or_eof(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::UnexpectedEof
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_error_display_lifecycle() {
        let inner = ScrcpyError::JarMissing(std::path::PathBuf::from("/nope.jar"));
        let err = StreamError::from(inner);
        let s = err.to_string();
        assert!(s.starts_with("scrcpy lifecycle:"), "got: {s}");
        assert!(s.contains("/nope.jar"), "got: {s}");
    }

    #[test]
    fn stream_error_display_wire() {
        let inner = WireError::ShortRead {
            expected: 64,
            got: 32,
        };
        let err = StreamError::from(inner);
        assert!(err.to_string().starts_with("scrcpy wire:"));
    }

    #[test]
    fn stream_error_display_codec_mismatch() {
        let err = StreamError::CodecMismatch {
            expected: WireCodec::H265,
            actual: WireCodec::H264,
        };
        let s = err.to_string();
        assert!(s.contains("requested H265"), "got: {s}");
        assert!(s.contains("device sent H264"), "got: {s}");
    }

    #[test]
    fn wire_codec_mapping_round_trips() {
        assert_eq!(wire_codec_for(VideoCodec::H265), WireCodec::H265);
        assert_eq!(wire_codec_for(VideoCodec::H264), WireCodec::H264);
        assert_eq!(wire_codec_for(VideoCodec::Av1), WireCodec::Av1);
    }

    #[test]
    fn timeout_classification() {
        let timed_out = io::Error::new(ErrorKind::TimedOut, "deadline");
        let would_block = io::Error::new(ErrorKind::WouldBlock, "wb");
        let eof = io::Error::new(ErrorKind::UnexpectedEof, "eof");
        let other = io::Error::new(ErrorKind::ConnectionReset, "rst");
        assert!(is_timeout_or_eof(&timed_out));
        assert!(is_timeout_or_eof(&would_block));
        assert!(is_timeout_or_eof(&eof));
        assert!(!is_timeout_or_eof(&other));
    }

    #[test]
    fn default_read_timeout_matches_recorder_empirical_value() {
        // 500 ms — empirically validated by the recorder example and
        // the Phase I.B.6 sustain run. 100 ms (the original guess) was
        // too tight and produced 0 frames over 60 s on USB.
        assert_eq!(DEFAULT_READ_TIMEOUT, Duration::from_millis(500));
    }
}

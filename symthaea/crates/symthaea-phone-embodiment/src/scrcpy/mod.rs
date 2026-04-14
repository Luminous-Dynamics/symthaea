// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! scrcpy-server lifecycle + binary framing parser + HEVC decode.
//!
//! This module is gated behind the `scrcpy` feature.
//!
//! Submodules:
//! - [`wire`] — pure binary-framing parser for scrcpy v2.4's wire protocol
//!   (device meta, video header, per-frame headers). No I/O — operates on
//!   `&[u8]` slices so it can be tested with handcrafted byte sequences.
//!
//! `mod.rs` itself owns the lifecycle layer:
//!
//! 1. **SHA-verified push** of the vendored JAR to `/data/local/tmp`
//! 2. **`adb reverse` tunnel** binding `localabstract:scrcpy` to a host TCP port
//! 3. **`app_process` spawn** of the server with the v1.4-decided codec
//!    (HEVC via `c2.exynos.hevc.encoder` — the Pixel 8 Pro hardware encoder)
//! 4. **Drop-based cleanup** of both the server process and the reverse tunnel
//!
//! # Codec decision
//!
//! See `symthaea/docs/phase_1b_codec_probe.md` for the full probe results.
//! Short version: Pixel 8 Pro on Android 16 only exposes HW encoders for
//! H.264 and HEVC. AV1 is software-only on this device, which is untenable
//! at 30 fps. HEVC HW gives ~50% better compression than H.264 — close
//! enough to AV1's 30-40% that the USB 2.0 budget relief survives.

pub mod wire;
pub mod decoder;
pub mod stream;

use std::io;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

/// Default vendored scrcpy-server JAR + SHA. Rebuild instructions in
/// `crates/symthaea-phone-embodiment/vendor/README.md`.
pub const VENDORED_JAR_NAME: &str = "scrcpy-server-v2.4.jar";
pub const VENDORED_JAR_SHA256: &str =
    "93c272b7438605c055e127f7444064ed78fa9ca49f81156777fd201e79ce7ba3";
/// scrcpy-server protocol version we vendor and parse.
pub const SCRCPY_VERSION: &str = "2.4";
/// Where the server lands on the device after push.
pub const DEVICE_JAR_PATH: &str = "/data/local/tmp/scrcpy-server.jar";
/// Local abstract socket name the server binds on the device side.
pub const DEVICE_ABSTRACT_SOCKET: &str = "scrcpy";

/// Codec the v1.4 roadmap pivot selected after the I.B.0 probe.
///
/// Variants intentionally use the scrcpy CLI spelling (`h265`, not `hevc`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// Hardware HEVC via `c2.exynos.hevc.encoder` (primary on Pixel 8 Pro).
    H265,
    /// Hardware H.264 via `c2.exynos.h264.encoder` (fallback).
    H264,
    /// Software AV1 via `c2.google.av1.encoder`. Research tier only —
    /// unsuitable for sustained 30 fps on a phone, kept for Phase II.5
    /// compressed-domain experiments.
    Av1,
}

impl VideoCodec {
    pub fn cli_name(self) -> &'static str {
        match self {
            VideoCodec::H265 => "h265",
            VideoCodec::H264 => "h264",
            VideoCodec::Av1 => "av1",
        }
    }
}

/// Zero-latency tuning options applied to every scrcpy-server invocation.
///
/// These are the "glass-to-glass latency over cinematic quality" defaults
/// from v1.3 of the roadmap. Values can still be overridden per-run via
/// [`ScrcpyOptions::extra_args`].
#[derive(Debug, Clone)]
pub struct ScrcpyOptions {
    pub serial: String,
    pub codec: VideoCodec,
    /// Locked encoder pacing matched to the cognitive cycle.
    pub max_fps: u32,
    /// Host-side TCP port the reverse tunnel will land on. The host parser
    /// connects to `127.0.0.1:tcp_port` once the server is up.
    pub tcp_port: u16,
    /// Disables device-side frame buffering (`--display-buffer=0`).
    /// True by default for cybernetic loop use; set false only if you want
    /// the device to smooth jitter at the cost of latency.
    pub display_buffer_zero: bool,
    /// Mute audio capture entirely. We have no audio path in Phase I.B.
    pub audio: bool,
    /// Extra `key=value` server args appended verbatim. Empty by default.
    /// Tuning bundles like `video_codec_options=profile=1` go here once
    /// they have been validated by the codec probe (I.B.0).
    pub extra_args: Vec<String>,
}

impl ScrcpyOptions {
    /// Glass-to-glass-latency defaults targeting the Pixel 8 Pro per v1.4.
    pub fn cybernetic_defaults(serial: impl Into<String>, tcp_port: u16) -> Self {
        Self {
            serial: serial.into(),
            codec: VideoCodec::H265,
            max_fps: 30,
            tcp_port,
            display_buffer_zero: true,
            audio: false,
            extra_args: Vec::new(),
        }
    }

    /// Render the `key=value` server arguments scrcpy v2.4 expects.
    fn to_server_args(&self) -> Vec<String> {
        let mut args = vec![
            format!("video=true"),
            format!("audio={}", self.audio),
            format!("video_codec={}", self.codec.cli_name()),
            format!("max_fps={}", self.max_fps),
            format!("control=true"),
            format!("send_device_meta=true"),
            format!("send_frame_meta=true"),
            // Use a tunnel_forward=false reverse-tunnel topology so we
            // listen on the host side via `adb reverse`.
            format!("tunnel_forward=false"),
            format!("log_level=info"),
        ];
        if self.display_buffer_zero {
            args.push("display_buffer=0".to_string());
        }
        args.extend(self.extra_args.iter().cloned());
        args
    }
}

/// Errors flowing out of the scrcpy lifecycle layer.
///
/// We deliberately keep this small and stringly-typed: the higher levels
/// (`PhoneBridge`, the soak observability harness) only need to log and
/// retry, not pattern-match.
#[derive(Debug)]
pub enum ScrcpyError {
    /// JAR file missing on disk before push.
    JarMissing(PathBuf),
    /// Vendored JAR SHA256 didn't match the constant.
    JarShaMismatch { expected: String, actual: String },
    /// `adb` command failed (push, reverse, shell, etc.). The string
    /// captures stderr or a context message.
    AdbFailed(String),
    /// I/O during JAR hashing or process spawn.
    Io(io::Error),
}

impl std::fmt::Display for ScrcpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScrcpyError::JarMissing(p) => write!(f, "scrcpy-server JAR not found at {}", p.display()),
            ScrcpyError::JarShaMismatch { expected, actual } => write!(
                f,
                "scrcpy-server JAR SHA256 mismatch: expected {expected}, got {actual}"
            ),
            ScrcpyError::AdbFailed(msg) => write!(f, "adb command failed: {msg}"),
            ScrcpyError::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for ScrcpyError {}

impl From<io::Error> for ScrcpyError {
    fn from(e: io::Error) -> Self {
        ScrcpyError::Io(e)
    }
}

/// Verify a file's SHA256 against the expected hex digest.
///
/// Returns `Ok(())` on match. Errors out with [`ScrcpyError::JarShaMismatch`]
/// otherwise — which is the correct response to supply-chain tampering, bit
/// rot, or accidental version drift in the vendor directory.
#[cfg(feature = "scrcpy")]
pub fn verify_sha256(path: &Path, expected_hex: &str) -> Result<(), ScrcpyError> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| match e.kind() {
        io::ErrorKind::NotFound => ScrcpyError::JarMissing(path.to_path_buf()),
        _ => ScrcpyError::Io(e),
    })?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let actual = hex_encode(&hasher.finalize());
    if actual.eq_ignore_ascii_case(expected_hex) {
        Ok(())
    } else {
        Err(ScrcpyError::JarShaMismatch {
            expected: expected_hex.to_string(),
            actual,
        })
    }
}

/// Tiny lowercase-hex encoder so the `scrcpy` feature doesn't drag a
/// `hex` crate just for one digest.
#[cfg(feature = "scrcpy")]
fn hex_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(TABLE[(b >> 4) as usize] as char);
        out.push(TABLE[(b & 0x0f) as usize] as char);
    }
    out
}

/// Manages a live scrcpy-server instance on the device.
///
/// On `Drop`, both the server child process and the `adb reverse` tunnel
/// are cleaned up. This is intentionally aggressive — leaving a stray
/// reverse tunnel around poisons subsequent runs, and leaving a stray
/// app_process around drains the device battery.
pub struct ScrcpyHandle {
    serial: String,
    tcp_port: u16,
    /// Active server child (an `adb shell` wrapping `app_process`).
    server: Option<Child>,
    /// Whether `adb reverse` was successfully established and needs removal.
    reverse_active: bool,
}

impl ScrcpyHandle {
    pub fn serial(&self) -> &str {
        &self.serial
    }
    pub fn tcp_port(&self) -> u16 {
        self.tcp_port
    }
}

impl Drop for ScrcpyHandle {
    fn drop(&mut self) {
        // Best-effort cleanup. We swallow errors because Drop must not panic
        // and the user already moved past this lifecycle.
        if let Some(mut child) = self.server.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if self.reverse_active {
            let _ = Command::new("adb")
                .args([
                    "-s",
                    &self.serial,
                    "reverse",
                    "--remove",
                    &format!("localabstract:{}", DEVICE_ABSTRACT_SOCKET),
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// Push the vendored scrcpy-server JAR to the device after SHA verification.
///
/// The verification step is not optional: it catches the three failure modes
/// listed in the vendor README (supply-chain tampering, accidental
/// corruption, version drift). The cost is one read + one SHA256 of a 70 KB
/// file, which is microseconds.
#[cfg(feature = "scrcpy")]
pub fn push_scrcpy_server(serial: &str, jar_path: &Path) -> Result<(), ScrcpyError> {
    verify_sha256(jar_path, VENDORED_JAR_SHA256)?;
    let output = Command::new("adb")
        .args([
            "-s",
            serial,
            "push",
            jar_path
                .to_str()
                .ok_or_else(|| ScrcpyError::AdbFailed("non-UTF8 JAR path".into()))?,
            DEVICE_JAR_PATH,
        ])
        .output()?;
    if !output.status.success() {
        return Err(ScrcpyError::AdbFailed(format!(
            "adb push: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Establish the reverse tunnel: device's `localabstract:scrcpy` ↔ host
/// `127.0.0.1:<tcp_port>`.
///
/// Must be called *before* spawning the server, so that when the server
/// connects to the device-side socket the host listener already exists.
#[cfg(feature = "scrcpy")]
pub fn open_reverse_tunnel(serial: &str, tcp_port: u16) -> Result<(), ScrcpyError> {
    let output = Command::new("adb")
        .args([
            "-s",
            serial,
            "reverse",
            &format!("localabstract:{}", DEVICE_ABSTRACT_SOCKET),
            &format!("tcp:{}", tcp_port),
        ])
        .output()?;
    if !output.status.success() {
        return Err(ScrcpyError::AdbFailed(format!(
            "adb reverse: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

/// Spawn `app_process` to start the scrcpy-server, returning a handle that
/// owns both the server child process and the reverse tunnel cleanup.
///
/// This does NOT block waiting for the server to be ready. The host should
/// `connect("127.0.0.1:tcp_port")` and the kernel will hold the connection
/// until the server side calls `accept()`. Sub-100 ms in practice on USB.
#[cfg(feature = "scrcpy")]
pub fn start_scrcpy(jar_path: &Path, opts: &ScrcpyOptions) -> Result<ScrcpyHandle, ScrcpyError> {
    push_scrcpy_server(&opts.serial, jar_path)?;
    open_reverse_tunnel(&opts.serial, opts.tcp_port)?;

    let server_args = opts.to_server_args();
    // Build the shell command line. Quoting matters: we go through `adb
    // shell` which does its own word-splitting, and the server args are
    // already in `key=value` form which is shell-safe.
    let shell_cmd = format!(
        "CLASSPATH={} app_process / com.genymobile.scrcpy.Server {} {}",
        DEVICE_JAR_PATH,
        SCRCPY_VERSION,
        server_args.join(" ")
    );

    let child = Command::new("adb")
        .args(["-s", &opts.serial, "shell", &shell_cmd])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| {
            // If spawn itself failed, we already opened the reverse tunnel —
            // tear it back down so we don't leak it.
            let _ = Command::new("adb")
                .args([
                    "-s",
                    &opts.serial,
                    "reverse",
                    "--remove",
                    &format!("localabstract:{}", DEVICE_ABSTRACT_SOCKET),
                ])
                .status();
            ScrcpyError::Io(e)
        })?;

    Ok(ScrcpyHandle {
        serial: opts.serial.clone(),
        tcp_port: opts.tcp_port,
        server: Some(child),
        reverse_active: true,
    })
}

/// Probe-mode invocation: run `list_encoders=true` and capture stdout.
///
/// Returns the raw output for the caller to parse. Used by I.B.0 to lock
/// the codec ladder, and by future runs as a startup sanity check (the
/// roadmap v1.4 promotes the probe to a Phase I.B startup task).
#[cfg(feature = "scrcpy")]
pub fn list_encoders(serial: &str, jar_path: &Path) -> Result<String, ScrcpyError> {
    push_scrcpy_server(serial, jar_path)?;
    let output = Command::new("adb")
        .args([
            "-s",
            serial,
            "shell",
            &format!(
                "CLASSPATH={} app_process / com.genymobile.scrcpy.Server {} \
                 list_encoders=true audio=false log_level=info",
                DEVICE_JAR_PATH, SCRCPY_VERSION
            ),
        ])
        .stderr(Stdio::piped())
        .stdout(Stdio::piped())
        .output()?;
    // list_encoders writes to stdout *and* stderr depending on log routing
    // on different scrcpy builds; concatenate both for the parser.
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(combined)
}

/// Wait briefly for the server to call `accept()` on the abstract socket
/// after a `start_scrcpy`. Useful when the caller wants to be sure the
/// tunnel is live before issuing the host-side TCP connect.
///
/// Default timeout 1 second is generous for USB; LAN/WAN may need more.
#[cfg(feature = "scrcpy")]
pub fn wait_for_server_ready(handle: &ScrcpyHandle, timeout: Duration) -> Result<(), ScrcpyError> {
    use std::net::{SocketAddr, TcpStream};
    let addr: SocketAddr = ([127, 0, 0, 1], handle.tcp_port).into();
    let deadline = std::time::Instant::now() + timeout;
    loop {
        match TcpStream::connect_timeout(&addr, Duration::from_millis(100)) {
            Ok(stream) => {
                // Drop the connection — caller will open its own.
                drop(stream);
                return Ok(());
            }
            Err(_) if std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                return Err(ScrcpyError::Io(e));
            }
        }
    }
}

#[cfg(all(test, feature = "scrcpy"))]
mod tests {
    use super::*;

    #[test]
    fn options_defaults_are_cybernetic() {
        let opts = ScrcpyOptions::cybernetic_defaults("serial123", 8400);
        assert_eq!(opts.codec, VideoCodec::H265);
        assert_eq!(opts.max_fps, 30);
        assert_eq!(opts.tcp_port, 8400);
        assert!(opts.display_buffer_zero);
        assert!(!opts.audio);
        let args = opts.to_server_args();
        assert!(args.iter().any(|a| a == "video_codec=h265"));
        assert!(args.iter().any(|a| a == "max_fps=30"));
        assert!(args.iter().any(|a| a == "display_buffer=0"));
        assert!(args.iter().any(|a| a == "audio=false"));
        assert!(args.iter().any(|a| a == "tunnel_forward=false"));
    }

    #[test]
    fn codec_cli_names_match_scrcpy_v24() {
        assert_eq!(VideoCodec::H265.cli_name(), "h265");
        assert_eq!(VideoCodec::H264.cli_name(), "h264");
        assert_eq!(VideoCodec::Av1.cli_name(), "av1");
    }

    #[test]
    fn hex_encode_lowercase_known_value() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0x10, 0xab]), "00ff10ab");
    }

    #[test]
    fn verify_sha256_round_trips() {
        // Write a tiny temp file, hash it externally, verify our checker matches.
        use sha2::{Digest, Sha256};
        let dir = std::env::temp_dir();
        let path = dir.join(format!("symthaea_scrcpy_sha_test_{}", std::process::id()));
        std::fs::write(&path, b"hello scrcpy").unwrap();
        let mut h = Sha256::new();
        h.update(b"hello scrcpy");
        let expected = hex_encode(&h.finalize());
        assert!(verify_sha256(&path, &expected).is_ok());
        // Wrong hash → mismatch error
        let wrong = "0".repeat(64);
        assert!(matches!(
            verify_sha256(&path, &wrong),
            Err(ScrcpyError::JarShaMismatch { .. })
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn extra_args_are_appended_verbatim() {
        let mut opts = ScrcpyOptions::cybernetic_defaults("s", 8400);
        opts.extra_args.push("video_codec_options=profile=1".to_string());
        let args = opts.to_server_args();
        assert!(args
            .iter()
            .any(|a| a == "video_codec_options=profile=1"));
    }
}

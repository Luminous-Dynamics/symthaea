// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-shot recorder for the Phase I.B.5 offline test asset.
//!
//! Connects to a live device via the existing scrcpy lifecycle, captures
//! N raw wire frames (header + payload) at a small `max_size` resolution,
//! and writes the bytes to `tests/data/sample.hevc.wire`. The offline
//! decoder test then consumes that asset via `include_bytes!` and runs
//! the full vertical (`wire::parse_*` → `HevcDecoder::decode_packet`)
//! without needing a connected device.
//!
//! # Usage
//!
//! ```text
//! cd symthaea
//! nix develop --command cargo run --example record_scrcpy_sample \
//!     --features scrcpy --release \
//!     -- 41201FDJG000UM 10
//! ```
//!
//! Args:
//! - serial (default: 41201FDJG000UM)
//! - frame count (default: 10)
//!
//! The recorder forces `max_size=720` and `max_fps=30` to keep the asset
//! small (~50–200 KB target). It also forces `video_codec_options` empty
//! so the encoder picks its own profile — the codec ladder probe (I.B.0)
//! already validated which encoder will run on this device.

use std::env;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};

use symthaea_phone_embodiment::scrcpy::wire::{
    DEVICE_NAME_LEN, FRAME_HEADER_LEN, VIDEO_HEADER_LEN, parse_device_meta, parse_frame_header,
    parse_video_header,
};
use symthaea_phone_embodiment::scrcpy::{
    DEVICE_JAR_PATH, ScrcpyOptions, VENDORED_JAR_NAME, accept_from_server, bind_host_listener,
    start_scrcpy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let serial = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "41201FDJG000UM".to_string());
    let frame_count: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    let crate_dir = env!("CARGO_MANIFEST_DIR");
    let jar = PathBuf::from(crate_dir)
        .join("vendor")
        .join(VENDORED_JAR_NAME);
    let out_path = PathBuf::from(crate_dir)
        .join("tests")
        .join("data")
        .join("sample.hevc.wire");

    println!("=== Phase I.B.5 wire recorder ===");
    println!("device serial : {serial}");
    println!("frame target  : {frame_count}");
    println!("vendored jar  : {}", jar.display());
    println!("output asset  : {}", out_path.display());
    println!();

    // Use a small max_size so the asset stays under ~200 KB at 10 frames.
    // The codec is HEVC (locked by the v1.4 roadmap pivot — see
    // docs/phase_1b_codec_probe.md). Port 8401 to avoid clashing with
    // any other dev session that might be using 8400.
    let mut opts = ScrcpyOptions::cybernetic_defaults(serial.clone(), 8401);
    opts.extra_args.push("max_size=720".to_string());

    println!(
        "Binding host listener on 127.0.0.1:{} BEFORE server spawn...",
        opts.tcp_port
    );
    let listener = bind_host_listener(opts.tcp_port)?;
    println!("Host listener ready.\n");

    println!("Spawning scrcpy-server (start_scrcpy pushes JAR + opens reverse tunnel + spawns)...");
    println!("  device JAR target: {DEVICE_JAR_PATH}");
    println!("  cybernetic options:");
    for a in opts.extra_args.iter() {
        println!("    extra: {a}");
    }
    println!("    video_codec=h265 (HW c2.exynos.hevc.encoder)");
    println!("    max_fps=30");
    println!("    tunnel_forward=false (host listens, server connects via reverse tunnel)");
    println!();

    let handle = start_scrcpy(&jar, &opts)?;
    println!("Server spawned. Waiting for incoming connection (5s budget)...");
    let mut tcp = accept_from_server(&listener, Duration::from_secs(5))?;
    tcp.set_read_timeout(Some(Duration::from_millis(500)))?;
    tcp.set_nodelay(true)?;
    println!("Accepted.");

    // 1. Read + parse + discard the device-meta block (64 bytes)
    let mut dm_buf = [0u8; DEVICE_NAME_LEN];
    tcp.read_exact(&mut dm_buf)?;
    let dm = parse_device_meta(&dm_buf)?;
    println!("Device name    : {}", dm.name);

    // 2. Read + parse + discard the video header (12 bytes)
    let mut vh_buf = [0u8; VIDEO_HEADER_LEN];
    tcp.read_exact(&mut vh_buf)?;
    let vh = parse_video_header(&vh_buf)?;
    println!("Stream codec   : {:?}", vh.codec);
    println!("Stream size    : {}x{}", vh.width, vh.height);
    println!();

    // 3. Capture N raw wire packets — frame headers + payloads, byte-for-byte.
    //    The offline decoder test will re-parse these via wire::parse_frame_header
    //    and feed each payload to HevcDecoder::decode_packet.
    let mut buf: Vec<u8> = Vec::with_capacity(64 * 1024);
    let mut config_seen = false;
    let mut keyframe_seen = false;
    let mut bytes_written: usize = 0;
    let started_at = Instant::now();

    for i in 0..frame_count {
        let mut hdr_buf = [0u8; FRAME_HEADER_LEN];
        tcp.read_exact(&mut hdr_buf)?;
        let hdr = parse_frame_header(&hdr_buf)?;
        let mut payload = vec![0u8; hdr.packet_size as usize];
        tcp.read_exact(&mut payload)?;

        if hdr.is_config() {
            config_seen = true;
        }
        if hdr.is_key_frame() {
            keyframe_seen = true;
        }

        let kind = if hdr.is_config() {
            "CONFIG"
        } else if hdr.is_key_frame() {
            "KEY   "
        } else {
            "P-FRM "
        };
        let pts = hdr
            .pts_micros()
            .map(|p| format!("{p:>10} us"))
            .unwrap_or_else(|| "       NO_PTS".to_string());
        println!(
            "  frame {:>2}: {kind} pts={pts} size={:>7} bytes",
            i, hdr.packet_size
        );

        buf.extend_from_slice(&hdr_buf);
        buf.extend_from_slice(&payload);
        bytes_written += FRAME_HEADER_LEN + payload.len();
    }

    let elapsed = started_at.elapsed();
    println!();
    println!(
        "Captured {} packets in {:.3}s",
        frame_count,
        elapsed.as_secs_f32()
    );
    println!("Total wire bytes  : {bytes_written}");
    println!("Saw config packet : {config_seen}");
    println!("Saw key frame     : {keyframe_seen}");
    println!();

    if !config_seen || !keyframe_seen {
        println!("WARNING: did not capture both a config packet and a key frame.");
        println!("The offline decoder test needs both to initialize the HEVC decoder.");
        println!("Try increasing the frame count (current: {frame_count}).");
    }

    // 4. Write the asset to disk.
    let mut out = std::fs::File::create(&out_path)?;
    out.write_all(&buf)?;
    out.sync_all()?;
    println!("Wrote {} bytes to {}", buf.len(), out_path.display());

    // ScrcpyHandle drop tears down server + reverse tunnel.
    drop(handle);
    println!("Server torn down. Done.");
    Ok(())
}

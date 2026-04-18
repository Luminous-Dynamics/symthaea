// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase II.A: LZ4 compression-ratio measurement for the sealed RDP wire.
//!
//! Captures real phone frames through the Phase I.B scrcpy streaming path,
//! seals each with `seal_frame`, then LZ4-compresses the sealed payload and
//! measures the size delta. Produces three numbers the roadmap's Phase II.A
//! STOP gate needs:
//!
//!   1. Mean sealed bytes per frame (raw).
//!   2. Mean sealed+LZ4 bytes per frame.
//!   3. Compression ratio (raw / lz4'd).
//!
//! The STOP gate: if projected sealed+LZ4 throughput at 30 fps is ≤ 8.75 MB/s
//! (25% of USB 2.0 ceiling), Phase II compression-ladder work beyond LZ4 is
//! deferred as optional polish. See `docs/HOLON_SOMA_ROADMAP.md` Phase II.A.
//!
//! This is an OFFLINE measurement — sealed bytes never traverse the wire.
//! It answers "what would LZ4 save" without requiring a transport change.

#[cfg(not(feature = "phase-2a-lz4"))]
fn main() {
    eprintln!("Requires: --features phase-2a-lz4 (nix develop)");
    std::process::exit(1);
}

#[cfg(feature = "phase-2a-lz4")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{anyhow, bail, Context};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use symthaea::swarm::rdp_holon_bridge::SomaRdpServer;
    use symthaea::swarm::rdp_protocol::{RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::seal_frame;
    use symthaea_phone_embodiment::streaming_bridge::StreamingPhoneBridge;

    const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];
    const DEFAULT_JAR: &str = "crates/symthaea-phone-embodiment/vendor/scrcpy-server-v2.4.jar";

    // The STOP gate in absolute bytes/sec at 30 fps. Derived from 25% of
    // the 35 MB/s USB 2.0 ceiling measured in Phase I.A, documented in
    // HOLON_SOMA_ROADMAP.md Phase II.A.
    const STOP_GATE_BPS_30FPS: f64 = 8.75 * 1024.0 * 1024.0;
    const TARGET_FPS: u32 = 30;

    #[derive(Debug)]
    struct Config {
        serial: String,
        duration_s: u64,
        fps_budget: u32,
        width: u32,
        height: u32,
        jar: PathBuf,
        tcp_port: u16,
    }

    fn parse_args() -> anyhow::Result<Config> {
        let mut config = Config {
            serial: std::env::var("ADB_SERIAL").unwrap_or_else(|_| "41201FDJG000UM".to_string()),
            duration_s: 15,
            fps_budget: 15,
            width: 1008,
            height: 2244,
            jar: PathBuf::from(DEFAULT_JAR),
            tcp_port: 8401,
        };

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--serial" => {
                    config.serial = args
                        .next()
                        .ok_or_else(|| anyhow!("--serial needs a value"))?;
                }
                "--duration" => {
                    config.duration_s = args
                        .next()
                        .ok_or_else(|| anyhow!("--duration needs a value"))?
                        .parse()
                        .context("parse --duration")?;
                }
                "--fps" => {
                    config.fps_budget = args
                        .next()
                        .ok_or_else(|| anyhow!("--fps needs a value"))?
                        .parse()
                        .context("parse --fps")?;
                }
                "--jar" => {
                    config.jar =
                        PathBuf::from(args.next().ok_or_else(|| anyhow!("--jar needs a value"))?);
                }
                "--tcp-port" => {
                    config.tcp_port = args
                        .next()
                        .ok_or_else(|| anyhow!("--tcp-port needs a value"))?
                        .parse()
                        .context("parse --tcp-port")?;
                }
                _ => bail!("unknown argument {arg}"),
            }
        }

        if config.fps_budget == 0 {
            bail!("--fps must be greater than zero");
        }
        if config.duration_s == 0 {
            bail!("--duration must be greater than zero");
        }
        Ok(config)
    }

    fn test_session(label: &str, is_initiator: bool) -> RdpSession {
        let mut session = RdpSession::new(
            label.into(),
            "peer".into(),
            RdpSessionConfig::default(),
            is_initiator,
        );
        session.on_connected();
        session.on_handshake_complete(PLACEHOLDER_KEY);
        session
    }

    fn frame_kind(frame: &RdpFrame) -> &'static str {
        match frame {
            RdpFrame::Full(_) => "Full",
            RdpFrame::Delta(_) => "Delta",
            _ => "Other",
        }
    }

    fn capture(config: &Config) -> anyhow::Result<Vec<RdpFrame>> {
        if !config.jar.exists() {
            bail!(
                "scrcpy-server JAR not found at {}; pass --jar PATH or run from symthaea/ dir",
                config.jar.display()
            );
        }

        let mut streaming = StreamingPhoneBridge::launch(
            config.serial.clone(),
            config.width,
            config.height,
            &config.jar,
            config.tcp_port,
        )
        .map_err(|error| anyhow!("scrcpy launch failed: {error}"))?;

        let mut soma = SomaRdpServer::new(
            config.width,
            config.height,
            config.fps_budget.min(255) as u8,
            config.fps_budget,
        );
        soma.start();

        let deadline = Instant::now() + Duration::from_secs(config.duration_s);
        let mut frames = Vec::new();

        while Instant::now() < deadline {
            match streaming.capture_streaming(1.0 / config.fps_budget as f32) {
                Ok(Some((_tel, rgba, width, height))) => {
                    soma.tick(&rgba, width, height, 0.65);
                    frames.extend(soma.drain_frames());
                }
                Ok(None) => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                Err(error) => bail!("scrcpy capture failed: {error}"),
            }
        }

        if frames.is_empty() {
            bail!("no RDP frames captured in {}s", config.duration_s);
        }
        Ok(frames)
    }

    fn mean(values: &[usize]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64
        }
    }

    let config = parse_args()?;
    eprintln!(
        "Phase II.A LZ4 measurement: serial={} duration={}s fps-budget={}",
        config.serial, config.duration_s, config.fps_budget
    );

    let frames = capture(&config)?;
    let n = frames.len();
    eprintln!("Captured {n} RDP frame(s), sealing + measuring...");

    let mut sealer = test_session("phase-2a-sealer", true);

    // LZ4 must happen BEFORE AEAD sealing — ciphertext is pseudorandom
    // and doesn't compress. The roadmap specifies "LZ4 wrap before
    // sealing" for this reason. We measure:
    //   bincode_bytes = RdpFrame::to_bin()           (serialized, pre-seal)
    //   lz4_bytes     = lz4(bincode_bytes)           (compressed pre-seal)
    //   sealed_raw    = seal_frame(frame)            (bincode + AEAD)
    //   aead_overhead = sealed_raw - bincode_bytes   (nonce + tag, constant)
    //   predicted_sealed_lz4 = lz4_bytes + aead_overhead
    //
    // The aead_overhead term is what currently ships and is unchanged
    // whether we LZ4 before or not; the compression ratio is the ratio
    // of the bincode payloads, which carries through the AEAD wrap.

    let mut bincode_full: Vec<usize> = Vec::new();
    let mut bincode_delta: Vec<usize> = Vec::new();
    let mut lz4_full: Vec<usize> = Vec::new();
    let mut lz4_delta: Vec<usize> = Vec::new();
    let mut sealed_raw_full: Vec<usize> = Vec::new();
    let mut sealed_raw_delta: Vec<usize> = Vec::new();
    let mut total_seal_us: u128 = 0;
    let mut total_lz4_us: u128 = 0;

    for frame in &frames {
        let bincode_bytes = frame.to_bin().context("RdpFrame::to_bin failed")?;

        let lz4_start = Instant::now();
        let compressed = lz4_flex::block::compress_prepend_size(&bincode_bytes);
        total_lz4_us += lz4_start.elapsed().as_micros();

        let seal_start = Instant::now();
        let sealed = seal_frame(frame, &mut sealer).context("seal_frame failed")?;
        total_seal_us += seal_start.elapsed().as_micros();

        match frame {
            RdpFrame::Full(_) => {
                bincode_full.push(bincode_bytes.len());
                lz4_full.push(compressed.len());
                sealed_raw_full.push(sealed.len());
            }
            RdpFrame::Delta(_) => {
                bincode_delta.push(bincode_bytes.len());
                lz4_delta.push(compressed.len());
                sealed_raw_delta.push(sealed.len());
            }
            _ => {}
        }
    }

    let n_full = bincode_full.len();
    let n_delta = bincode_delta.len();
    let bincode_full_mean = mean(&bincode_full);
    let bincode_delta_mean = mean(&bincode_delta);
    let lz4_full_mean = mean(&lz4_full);
    let lz4_delta_mean = mean(&lz4_delta);
    let sealed_full_mean = mean(&sealed_raw_full);
    let sealed_delta_mean = mean(&sealed_raw_delta);

    // AEAD overhead is constant per frame (nonce + Poly1305 tag ≈ 28-32
    // bytes). sealed_raw_mean - bincode_mean gives it empirically.
    let aead_overhead_full = (sealed_full_mean - bincode_full_mean).max(0.0);
    let aead_overhead_delta = (sealed_delta_mean - bincode_delta_mean).max(0.0);

    // Predicted sealed-with-LZ4 size: lz4'd bincode + same AEAD overhead.
    let predicted_sealed_lz4_full = lz4_full_mean + aead_overhead_full;
    let predicted_sealed_lz4_delta = lz4_delta_mean + aead_overhead_delta;

    let bincode_total: usize =
        bincode_full.iter().sum::<usize>() + bincode_delta.iter().sum::<usize>();
    let lz4_total: usize = lz4_full.iter().sum::<usize>() + lz4_delta.iter().sum::<usize>();
    let overall_ratio = if lz4_total > 0 {
        bincode_total as f64 / lz4_total as f64
    } else {
        0.0
    };

    println!("======================================================================");
    println!("  Phase II.A — LZ4 compression-ratio measurement");
    println!("  (LZ4 BEFORE AEAD seal — ciphertext is pseudorandom, won't compress)");
    println!("======================================================================");
    println!();
    println!("  Captured: {n} RDP frames ({n_full} Full, {n_delta} Delta)");
    println!();
    println!("  Per-frame bytes (Full  n={n_full}):");
    println!("    bincode raw:               {:>10.0}", bincode_full_mean);
    println!("    bincode + LZ4:             {:>10.0}", lz4_full_mean);
    println!("    sealed (current wire):     {:>10.0}", sealed_full_mean);
    println!(
        "    predicted sealed+LZ4:      {:>10.0}",
        predicted_sealed_lz4_full
    );
    println!(
        "    AEAD overhead:             {:>10.0}",
        aead_overhead_full
    );
    println!();
    println!("  Per-frame bytes (Delta n={n_delta}):");
    println!(
        "    bincode raw:               {:>10.0}",
        bincode_delta_mean
    );
    println!("    bincode + LZ4:             {:>10.0}", lz4_delta_mean);
    println!("    sealed (current wire):     {:>10.0}", sealed_delta_mean);
    println!(
        "    predicted sealed+LZ4:      {:>10.0}",
        predicted_sealed_lz4_delta
    );
    println!(
        "    AEAD overhead:             {:>10.0}",
        aead_overhead_delta
    );
    println!();
    println!("  Compression ratio (bincode / lz4): {overall_ratio:.2}x");
    println!();
    println!(
        "  Seal  latency mean: {:>6} us/frame",
        total_seal_us / n.max(1) as u128
    );
    println!(
        "  LZ4   latency mean: {:>6} us/frame",
        total_lz4_us / n.max(1) as u128
    );
    println!();

    // Project 30 fps throughput. Prefer Delta-only basis if any captured
    // (steady-state signal). The "raw" projection uses current sealed
    // wire bytes; the "lz4" projection uses predicted sealed+LZ4.
    let projection_base_raw = if n_delta > 0 {
        sealed_delta_mean
    } else {
        mean(&sealed_raw_full)
    };
    let projection_base_lz4 = if n_delta > 0 {
        predicted_sealed_lz4_delta
    } else {
        predicted_sealed_lz4_full
    };
    let proj_raw_30fps = projection_base_raw * TARGET_FPS as f64;
    let proj_lz4_30fps = projection_base_lz4 * TARGET_FPS as f64;
    let stop_gate = STOP_GATE_BPS_30FPS;

    println!("  Projection to {TARGET_FPS} fps (steady-state Delta basis if available):");
    println!(
        "    raw @ 30 fps:  {:>10.0} B/s ({:.3} MB/s)",
        proj_raw_30fps,
        proj_raw_30fps / (1024.0 * 1024.0)
    );
    println!(
        "    lz4 @ 30 fps:  {:>10.0} B/s ({:.3} MB/s)",
        proj_lz4_30fps,
        proj_lz4_30fps / (1024.0 * 1024.0)
    );
    println!(
        "    STOP gate:     {:>10.0} B/s ({:.3} MB/s)",
        stop_gate,
        stop_gate / (1024.0 * 1024.0)
    );
    println!();

    // STOP-gate verdict. Separate verdicts for raw and lz4 — both
    // matter: if raw already clears the gate, LZ4 is pure polish; if
    // raw doesn't but lz4 does, LZ4 is the minimum-viable path.
    let raw_passes = proj_raw_30fps <= stop_gate;
    let lz4_passes = proj_lz4_30fps <= stop_gate;

    println!("  Phase II.A decision:");
    if raw_passes {
        println!("    raw SEALED stream already ≤ 8.75 MB/s STOP gate. LZ4 is polish.");
        println!("    Phase II.B (attention backchannel) is also polish per the roadmap.");
        println!("    Phase II.C (deeper compression) NOT REQUIRED for USB 2.0 target.");
    } else if lz4_passes {
        println!("    raw sealed stream EXCEEDS the gate, but LZ4 CLEARS it.");
        println!("    LZ4 is the minimum-viable compression for the 30 fps USB target.");
        println!("    Phase II.B + II.C remain deferred unless WAN deployment demands.");
    } else {
        println!("    both raw and LZ4 EXCEED the 8.75 MB/s gate.");
        println!("    Proceed to Phase II.C (sparse patches, content-adaptive, delta coding).");
    }
    println!();
    println!("  (Note: Full-frame bootstraps dominate short captures. Run longer");
    println!("   with on-screen motion to get a steady-state Delta-only ratio.)");
    println!();
    println!("======================================================================");

    Ok(())
}

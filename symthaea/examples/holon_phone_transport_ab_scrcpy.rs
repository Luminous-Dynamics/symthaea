// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Live phone-content A/B harness for Holon RDP transports, driven by
//! the Phase I.B scrcpy persistent capture path.
//!
//! Captures real phone frames through `StreamingPhoneBridge` (HEVC wire
//! decoded via ffmpeg-next), converts them to `RdpFrame`s with
//! `SomaRdpServer`, then replays the exact same frame vector through the
//! baseline WebSocket transport and the Phase I.C QUIC transport.
//!
//! This closes the roadmap's "Real scrcpy-stream A/B" item for Phase I.C
//! — the earlier `holon_phone_transport_ab` used the ADB screenshot
//! polling path and explicitly noted that a scrcpy-derived checkpoint
//! was still open.

#[cfg(not(all(feature = "holon-viewer", feature = "phone-scrcpy")))]
fn main() {
    eprintln!("Requires: --features holon-viewer,phone-scrcpy (nix develop for ffmpeg + libclang)");
    std::process::exit(1);
}

#[cfg(all(feature = "holon-viewer", feature = "phone-scrcpy"))]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{Context, anyhow, bail};
    use futures_util::{SinkExt, StreamExt};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    use symthaea::api::holon::{HolonHttpState, holon_router};
    use symthaea::swarm::quic_transport::{run_viewer_quic_client, spawn_holon_quic_server};
    use symthaea::swarm::rdp_holon_bridge::SomaRdpServer;
    use symthaea::swarm::rdp_protocol::{InputEvent, InputFrame, RdpFrame, RdpSessionConfig};
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::{open_frame, open_input, seal_frame, seal_input};
    use symthaea_phone_embodiment::streaming_bridge::StreamingPhoneBridge;

    const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];
    // 10 s, not 2. scrcpy full-frame bootstraps traverse the QUIC
    // reliable-stream fallback (for oversized sealed frames) which is
    // slower than the datagram path and can exceed 2 s on localhost
    // with heavy scrcpy keyframe payloads. 10 s is generous for the
    // A/B; per-frame time is what we measure, not the deadline.
    const FRAME_TIMEOUT: Duration = Duration::from_secs(10);
    const DEFAULT_JAR: &str = "crates/symthaea-phone-embodiment/vendor/scrcpy-server-v2.4.jar";
    const DEFAULT_TCP_PORT: u16 = 8408; // Dev/test range 8400-8409 per PORTS.md

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

    #[derive(Debug)]
    struct LatencyStats {
        samples: usize,
        p50_us: u128,
        p99_us: u128,
        max_us: u128,
    }

    fn parse_args() -> anyhow::Result<Config> {
        let mut config = Config {
            serial: std::env::var("ADB_SERIAL").unwrap_or_else(|_| "41201FDJG000UM".to_string()),
            duration_s: 5,
            fps_budget: 30,
            width: 1008,
            height: 2244,
            jar: PathBuf::from(DEFAULT_JAR),
            tcp_port: DEFAULT_TCP_PORT,
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
                "--width" => {
                    config.width = args
                        .next()
                        .ok_or_else(|| anyhow!("--width needs a value"))?
                        .parse()
                        .context("parse --width")?;
                }
                "--height" => {
                    config.height = args
                        .next()
                        .ok_or_else(|| anyhow!("--height needs a value"))?
                        .parse()
                        .context("parse --height")?;
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

    fn display_frame_id(frame: &RdpFrame) -> Option<u64> {
        match frame {
            RdpFrame::Full(frame) => Some(frame.frame_id),
            RdpFrame::Delta(frame) => Some(frame.frame_id),
            _ => None,
        }
    }

    fn sample_input(sequence: u64) -> InputFrame {
        InputFrame {
            sequence,
            timestamp_ms: 0,
            events: vec![InputEvent::Pointer {
                x: 0.25,
                y: 0.75,
                button: 1,
                pressed: true,
            }],
        }
    }

    fn stats(samples: &[Duration]) -> LatencyStats {
        let mut micros: Vec<u128> = samples.iter().map(|sample| sample.as_micros()).collect();
        micros.sort_unstable();
        let last = micros.len().saturating_sub(1);
        LatencyStats {
            samples: micros.len(),
            p50_us: micros[(last * 50) / 100],
            p99_us: micros[(last * 99) / 100],
            max_us: micros[last],
        }
    }

    fn capture_phone_frames_scrcpy(config: &Config) -> anyhow::Result<Vec<RdpFrame>> {
        if !config.jar.exists() {
            bail!(
                "scrcpy-server JAR not found at {}; pass --jar PATH or run from repo root",
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

        // Align SomaRdpServer's advertised cap with the encoder's pacing.
        let mut soma = SomaRdpServer::new(
            config.width,
            config.height,
            config.fps_budget.min(255) as u8,
            config.fps_budget,
        );
        soma.start();

        let deadline = Instant::now() + Duration::from_secs(config.duration_s);
        let mut frames = Vec::new();
        let mut stream_frames_observed: u64 = 0;

        while Instant::now() < deadline {
            let outcome = streaming
                .capture_streaming(1.0 / config.fps_budget as f32)
                .map_err(|error| anyhow!("scrcpy capture failed: {error}"))?;
            let Some((tel, rgba, width, height)) = outcome else {
                // Read timeout inside the scrcpy stream — loop and
                // retry. Matches cognitive-loop "no frame ready"
                // semantics documented on capture_streaming.
                std::thread::sleep(Duration::from_millis(10));
                continue;
            };
            let rgba: Vec<u8> = rgba;
            stream_frames_observed += 1;
            soma.tick(&rgba, width, height, 0.65);
            let drained = soma.drain_frames();
            for frame in drained {
                if let Some(id) = display_frame_id(&frame) {
                    println!(
                        "captured frame={} source={}x{} prediction_error={:.3}",
                        id, width, height, tel.prediction_error
                    );
                }
                frames.push(frame);
            }
        }

        if frames.is_empty() {
            bail!(
                "scrcpy capture produced no RDP frames after {} device frames in {}s",
                stream_frames_observed,
                config.duration_s
            );
        }

        println!(
            "scrcpy stream: {} device frames observed, {} RDP frames produced",
            stream_frames_observed,
            frames.len()
        );
        Ok(frames)
    }

    async fn wait_for_status(
        status: &Arc<Mutex<String>>,
        prefix: &str,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        while started.elapsed() < timeout {
            let current = status.lock().map(|guard| guard.clone()).unwrap_or_default();
            if current.starts_with(prefix) {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let current = status.lock().map(|guard| guard.clone()).unwrap_or_default();
        bail!("timed out waiting for status {prefix:?}; last status: {current}");
    }

    async fn wait_for_inbound_input(
        state: &HolonHttpState,
        session: &mut RdpSession,
        sequence: u64,
        timeout: Duration,
    ) -> anyhow::Result<()> {
        let started = Instant::now();
        while started.elapsed() < timeout {
            for sealed in state.drain_rdp_inbound() {
                let input = open_input(&sealed, session).context("open inbound input")?;
                if input.sequence == sequence {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        bail!("timed out waiting for inbound input sequence {sequence}");
    }

    async fn run_ws(frames: &[RdpFrame]) -> anyhow::Result<LatencyStats> {
        let (tx, _rx) = std::sync::mpsc::channel();
        let state = Arc::new(HolonHttpState::new(tx));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("bind WS scrcpy A/B listener")?;
        let addr = listener
            .local_addr()
            .context("read WS scrcpy A/B address")?;
        let router = holon_router(state.clone());
        let server = tokio::spawn(async move { axum::serve(listener, router).await });

        let url = format!("ws://{addr}/holon/ws");
        let (mut socket, _response) = connect_async(&url)
            .await
            .with_context(|| format!("connect WS scrcpy A/B client to {url}"))?;
        let mut sender_session = test_session("scrcpy-ab-ws-server", true);
        let mut receiver_session = test_session("scrcpy-ab-ws-client", false);
        let mut samples = Vec::with_capacity(frames.len());

        for frame in frames {
            let Some(expected_id) = display_frame_id(frame) else {
                continue;
            };
            let sealed = seal_frame(frame, &mut sender_session).context("seal WS scrcpy frame")?;
            let sent_at = Instant::now();
            state.push_rdp_outbound(sealed);

            loop {
                let msg = tokio::time::timeout(FRAME_TIMEOUT, socket.next())
                    .await
                    .context("timed out waiting for WS scrcpy frame")?
                    .ok_or_else(|| anyhow!("WS scrcpy stream ended"))?
                    .context("receive WS scrcpy message")?;
                match msg {
                    Message::Binary(bytes) => {
                        let received = open_frame(&bytes, &mut receiver_session)
                            .context("open WS scrcpy frame")?;
                        if display_frame_id(&received) == Some(expected_id) {
                            samples.push(sent_at.elapsed());
                            break;
                        }
                    }
                    Message::Close(_) => bail!("WS scrcpy server closed during run"),
                    _ => {}
                }
            }
        }

        let input = sample_input(1);
        let sealed_input =
            seal_input(&input, &mut receiver_session).context("seal WS scrcpy reverse input")?;
        socket
            .send(Message::Binary(sealed_input.into()))
            .await
            .context("send WS scrcpy reverse input")?;
        wait_for_inbound_input(&state, &mut sender_session, input.sequence, FRAME_TIMEOUT).await?;

        server.abort();
        Ok(stats(&samples))
    }

    async fn run_quic(frames: &[RdpFrame]) -> anyhow::Result<LatencyStats> {
        let (tx, _rx) = std::sync::mpsc::channel();
        let state = Arc::new(HolonHttpState::new(tx));
        let server = spawn_holon_quic_server(
            state.clone(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        )
        .context("spawn QUIC scrcpy A/B server")?;

        let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let status = Arc::new(Mutex::new(String::new()));
        let session = Arc::new(Mutex::new(test_session("scrcpy-ab-quic-client", false)));
        let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});

        let client = tokio::spawn(run_viewer_quic_client(
            server.local_addr,
            "127.0.0.1",
            session,
            frame_tx,
            input_rx,
            status.clone(),
            repaint,
        ));
        wait_for_status(&status, "connected", FRAME_TIMEOUT).await?;

        let mut sender_session = test_session("scrcpy-ab-quic-server", true);
        let mut samples = Vec::with_capacity(frames.len());

        for frame in frames {
            let Some(expected_id) = display_frame_id(frame) else {
                continue;
            };
            let sealed =
                seal_frame(frame, &mut sender_session).context("seal QUIC scrcpy frame")?;
            let sent_at = Instant::now();
            state.push_rdp_outbound(sealed);

            loop {
                let received = tokio::time::timeout(FRAME_TIMEOUT, frame_rx.recv())
                    .await
                    .context("timed out waiting for QUIC scrcpy frame")?
                    .ok_or_else(|| anyhow!("QUIC scrcpy client closed"))?;
                if display_frame_id(&received) == Some(expected_id) {
                    samples.push(sent_at.elapsed());
                    break;
                }
            }
        }

        let input = sample_input(1);
        input_tx
            .send(input.clone())
            .map_err(|error| anyhow!("send QUIC scrcpy reverse input failed: {error}"))?;
        wait_for_inbound_input(&state, &mut sender_session, input.sequence, FRAME_TIMEOUT).await?;

        client.abort();
        Ok(stats(&samples))
    }

    let config = parse_args()?;
    println!(
        "Capturing scrcpy frames: serial={} duration={}s fps-budget={} nominal={}x{} jar={} port={}",
        config.serial,
        config.duration_s,
        config.fps_budget,
        config.width,
        config.height,
        config.jar.display(),
        config.tcp_port
    );
    let frames = capture_phone_frames_scrcpy(&config)?;
    println!(
        "Captured {} RDP frame(s); replaying transports...",
        frames.len()
    );

    let ws = run_ws(&frames).await.context("WS scrcpy A/B failed")?;
    let quic = run_quic(&frames).await.context("QUIC scrcpy A/B failed")?;

    println!(
        "WS   samples={} p50={}us p99={}us max={}us",
        ws.samples, ws.p50_us, ws.p99_us, ws.max_us
    );
    println!(
        "QUIC samples={} p50={}us p99={}us max={}us",
        quic.samples, quic.p50_us, quic.p99_us, quic.max_us
    );
    println!("Reverse input path OK for WS and QUIC");
    println!("Source: scrcpy persistent HEVC capture via StreamingPhoneBridge");

    Ok(())
}
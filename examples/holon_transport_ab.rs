// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Synthetic localhost A/B latency harness for Holon RDP transports.
//!
//! This does not replace the real scrcpy-stream A/B from the roadmap. It gives
//! Phase I.C a repeatable headless baseline: same sealed `FullFrame` payloads,
//! same localhost host, WebSocket vs QUIC transport.

#[cfg(not(feature = "holon-viewer"))]
fn main() {
    eprintln!("Requires: --features holon-viewer");
    std::process::exit(1);
}

#[cfg(feature = "holon-viewer")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use anyhow::{Context, anyhow, bail};
    use futures_util::{SinkExt, StreamExt};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    use symthaea::api::holon::{HolonHttpState, holon_router};
    use symthaea::swarm::quic_transport::{run_viewer_quic_client, spawn_holon_quic_server};
    use symthaea::swarm::rdp_codec::TILE_SIZE;
    use symthaea::swarm::rdp_protocol::{
        FullFrame, InputEvent, InputFrame, QuantizedPatch, RdpFrame, RdpSessionConfig,
    };
    use symthaea::swarm::rdp_session::RdpSession;
    use symthaea::swarm::rdp_wire::{open_frame, open_input, seal_frame, seal_input};

    const PLACEHOLDER_KEY: [u8; 32] = [0x42; 32];
    const DEFAULT_FRAMES: u64 = 30;
    const FRAME_TIMEOUT: Duration = Duration::from_secs(2);

    #[derive(Debug)]
    struct LatencyStats {
        samples: usize,
        p50_us: u128,
        p99_us: u128,
        max_us: u128,
    }

    fn frame_count_from_args() -> anyhow::Result<u64> {
        let mut frames = DEFAULT_FRAMES;
        for arg in std::env::args().skip(1) {
            if let Some(value) = arg.strip_prefix("--frames=") {
                frames = value
                    .parse::<u64>()
                    .with_context(|| format!("parse frame count from {arg}"))?;
            } else {
                bail!("unknown argument {arg}; expected --frames=N");
            }
        }
        if frames == 0 {
            bail!("--frames must be greater than zero");
        }
        Ok(frames)
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

    fn sample_full_frame(frame_id: u64, cols: u16, rows: u16, seed: u8) -> RdpFrame {
        let patches: Vec<QuantizedPatch> = (0..(cols as usize * rows as usize))
            .map(|idx| {
                let tile_x = (idx % cols as usize) as u8;
                let tile_y = (idx / cols as usize) as u8;
                let values: Vec<i8> = (0..(TILE_SIZE * TILE_SIZE))
                    .map(|p| {
                        let px = (p % TILE_SIZE) as u8;
                        let py = (p / TILE_SIZE) as u8;
                        let v = tile_x
                            .wrapping_add(tile_y.wrapping_mul(3))
                            .wrapping_add(px.wrapping_mul(2))
                            .wrapping_add(py)
                            .wrapping_add(seed);
                        (v as i16 - 128) as i8
                    })
                    .collect();
                QuantizedPatch { values }
            })
            .collect();

        RdpFrame::Full(FullFrame {
            frame_id,
            timestamp_ms: 0,
            patch_cols: cols,
            patch_rows: rows,
            patches,
            consciousness_level: 0.65,
            harmony: "transport-ab".into(),
        })
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

    fn display_frame_id(frame: &RdpFrame) -> Option<u64> {
        match frame {
            RdpFrame::Full(frame) => Some(frame.frame_id),
            RdpFrame::Delta(frame) => Some(frame.frame_id),
            _ => None,
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

    async fn run_ws(frames: u64) -> anyhow::Result<LatencyStats> {
        let (tx, _rx) = std::sync::mpsc::channel();
        let state = Arc::new(HolonHttpState::new(tx));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("bind WS A/B listener")?;
        let addr = listener.local_addr().context("read WS A/B address")?;
        let router = holon_router(state.clone());
        let server = tokio::spawn(async move { axum::serve(listener, router).await });

        let url = format!("ws://{addr}/holon/ws");
        let (mut socket, _response) = connect_async(&url)
            .await
            .with_context(|| format!("connect WS A/B client to {url}"))?;
        let mut sender_session = test_session("ab-ws-server", true);
        let mut receiver_session = test_session("ab-ws-client", false);
        let mut samples = Vec::with_capacity(frames as usize);

        for id in 1..=frames {
            let frame = sample_full_frame(id, 4, 4, id as u8);
            let sealed = seal_frame(&frame, &mut sender_session).context("seal WS A/B frame")?;
            let sent_at = Instant::now();
            state.push_rdp_outbound(sealed);

            loop {
                let msg = tokio::time::timeout(FRAME_TIMEOUT, socket.next())
                    .await
                    .context("timed out waiting for WS A/B frame")?
                    .ok_or_else(|| anyhow!("WS A/B stream ended"))?
                    .context("receive WS A/B message")?;
                match msg {
                    Message::Binary(bytes) => {
                        let received = open_frame(&bytes, &mut receiver_session)
                            .context("open WS A/B frame")?;
                        if display_frame_id(&received) == Some(id) {
                            samples.push(sent_at.elapsed());
                            break;
                        }
                    }
                    Message::Close(_) => bail!("WS A/B server closed during run"),
                    _ => {}
                }
            }
        }

        let input = sample_input(1);
        let sealed_input =
            seal_input(&input, &mut receiver_session).context("seal WS reverse input")?;
        socket
            .send(Message::Binary(sealed_input.into()))
            .await
            .context("send WS reverse input")?;
        wait_for_inbound_input(&state, &mut sender_session, input.sequence, FRAME_TIMEOUT).await?;

        server.abort();
        Ok(stats(&samples))
    }

    async fn run_quic(frames: u64) -> anyhow::Result<LatencyStats> {
        let (tx, _rx) = std::sync::mpsc::channel();
        let state = Arc::new(HolonHttpState::new(tx));
        let server = spawn_holon_quic_server(
            state.clone(),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        )
        .context("spawn QUIC A/B server")?;

        let (frame_tx, mut frame_rx) = tokio::sync::mpsc::unbounded_channel();
        let (input_tx, input_rx) = tokio::sync::mpsc::unbounded_channel();
        let status = Arc::new(Mutex::new(String::new()));
        let session = Arc::new(Mutex::new(test_session("ab-quic-client", false)));
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

        let mut sender_session = test_session("ab-quic-server", true);
        let mut samples = Vec::with_capacity(frames as usize);

        for id in 1..=frames {
            let frame = sample_full_frame(id, 4, 4, id as u8);
            let sealed = seal_frame(&frame, &mut sender_session).context("seal QUIC A/B frame")?;
            let sent_at = Instant::now();
            state.push_rdp_outbound(sealed);

            loop {
                let received = tokio::time::timeout(FRAME_TIMEOUT, frame_rx.recv())
                    .await
                    .context("timed out waiting for QUIC A/B frame")?
                    .ok_or_else(|| anyhow!("QUIC A/B client closed"))?;
                if display_frame_id(&received) == Some(id) {
                    samples.push(sent_at.elapsed());
                    break;
                }
            }
        }

        let input = sample_input(1);
        input_tx
            .send(input.clone())
            .map_err(|error| anyhow!("send QUIC reverse input to client task failed: {error}"))?;
        wait_for_inbound_input(&state, &mut sender_session, input.sequence, FRAME_TIMEOUT).await?;

        client.abort();
        Ok(stats(&samples))
    }

    let frames = frame_count_from_args()?;
    let ws = run_ws(frames).await.context("WS A/B run failed")?;
    let quic = run_quic(frames).await.context("QUIC A/B run failed")?;

    println!(
        "WS   samples={} p50={}us p99={}us max={}us",
        ws.samples, ws.p50_us, ws.p99_us, ws.max_us
    );
    println!(
        "QUIC samples={} p50={}us p99={}us max={}us",
        quic.samples, quic.p50_us, quic.p99_us, quic.max_us
    );
    println!("Reverse input path OK for WS and QUIC");

    Ok(())
}

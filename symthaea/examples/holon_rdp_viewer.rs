// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Holon RDP Viewer — egui desktop client for the Phase I.A binary wire.
//!
//! ## Status — Phase I.A.2 Pieces 1+2 done
//!
//! **Piece 1 (egui window shell)**: ✅ HolonViewerApp + FrameBuffer blit
//! to `egui::TextureHandle`.
//!
//! **Piece 2 (tokio-tungstenite WS client)**: ✅ dedicated tokio runtime
//! thread connects to `ws://localhost:7778/holon/ws`, split into read
//! (`Message::Binary` → `open_frame` → mpsc → egui) and write (egui
//! pointer events → `InputFrame` → `seal_input` → `Message::Binary`)
//! tasks. Egui's sync `update()` drains the frame channel via
//! `tokio::sync::mpsc::UnboundedReceiver::try_recv` which works from a
//! non-async context.
//!
//! **Phase I.C (QUIC transport A/B)**: ✅ `--transport=quic|ws`. QUIC uses
//! unreliable datagrams for sealed RDP frames and a reliable QUIC stream for
//! sealed input events while leaving the wire envelope unchanged.
//!
//! **Piece 3 (PQC handshake unblock)**: ⏳ deferred — placeholder key
//! `[0x42; 32]` is safe for localhost development. Real KEM handshake
//! lands when network deployment requires it (Phase I.B or later).
//!
//! ## Architecture — sync/async bridge
//!
//! ```text
//!   main thread (egui event loop, sync)
//!    ├─ HolonViewerApp::update()
//!    │    ├─ frame_rx.try_recv() loop → viewer.apply_frame
//!    │    ├─ refresh_texture() on new frame
//!    │    └─ pointer events → input_tx.send(InputFrame)
//!    └─ holds Arc<Mutex<RdpSession>> for seal/open
//!
//!   ws thread (tokio Runtime, async)
//!    ├─ connect_async("ws://localhost:7778/holon/ws")
//!    ├─ read task: ws_stream.next() → open_frame → frame_tx.send
//!    │             + ctx.request_repaint()
//!    └─ write task: input_rx.recv() → seal_input → ws_sink.send
//!
//!   or
//!
//!   quic thread (tokio Runtime, async)
//!    ├─ connect("quic://127.0.0.1:7779")
//!    ├─ read task: datagram reassembly → open_frame → frame_tx.send
//!    │             + ctx.request_repaint()
//!    └─ write task: input_rx.recv() → seal_input → reliable stream write
//! ```
//!
//! ## Usage
//!
//! ```bash
//! # Terminal 1 — Holon WS server (other end of the wire)
//! cargo run --release --bin symthaea-holon \
//!     --features api_module,mesh-encryption,phone
//!
//! # Terminal 2 — viewer
//! cargo run --release --example holon_rdp_viewer --features holon-viewer
//! ```
//!
//! The viewer defaults to `--transport=ws --url ws://localhost:7778/holon/ws`.
//! For Phase I.C A/B, use `--transport=quic --url quic://127.0.0.1:7779`.

#[cfg(not(feature = "holon-viewer"))]
fn main() {
    eprintln!(
        "Requires: --features holon-viewer\n\
         \n\
         This example needs the egui window, tokio-tungstenite client,\n\
         and the mesh-encryption + api_module features. The `holon-viewer`\n\
         feature pulls all of them together.\n"
    );
    std::process::exit(1);
}

#[cfg(feature = "holon-viewer")]
fn main() -> eframe::Result<()> {
    use eframe::egui;

    // Default to WS on localhost:7778. Phase I.C adds `--transport=quic`
    // for direct A/B against the new QUIC path.
    let args: Vec<String> = std::env::args().collect();
    let transport = args
        .iter()
        .position(|a| a == "--transport")
        .and_then(|i| args.get(i + 1))
        .map(|value| value.to_lowercase())
        .unwrap_or_else(|| "ws".to_string());
    let url = args
        .iter()
        .position(|a| a == "--url")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| match transport.as_str() {
            "quic" => "quic://127.0.0.1:7779".to_string(),
            _ => "ws://localhost:7778/holon/ws".to_string(),
        });

    let transport = match transport.as_str() {
        "quic" => ViewerTransport::Quic,
        _ => ViewerTransport::Ws,
    };

    const PHONE_W: u32 = 1008;
    const PHONE_H: u32 = 2244;
    const TILE_COLS: u16 = 16;
    const TILE_ROWS: u16 = 35;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([540.0, 1200.0])
            .with_min_inner_size([400.0, 600.0])
            .with_title("Holon RDP Viewer — Phase I.A.2"),
        ..Default::default()
    };

    eframe::run_native(
        "Holon RDP Viewer",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(HolonViewerApp::new(
                PHONE_W,
                PHONE_H,
                TILE_COLS,
                TILE_ROWS,
                transport,
                url.clone(),
                cc.egui_ctx.clone(),
            )))
        }),
    )
}

#[cfg(feature = "holon-viewer")]
#[derive(Clone, Copy, Debug)]
enum ViewerTransport {
    Ws,
    Quic,
}

#[cfg(feature = "holon-viewer")]
struct HolonViewerApp {
    /// The RDP frame buffer — receives `apply_full_frame`/`apply_delta_frame`
    /// from the WS read task.
    viewer: symthaea::swarm::rdp_holon_bridge::HolonRdpViewer,
    /// Shared session for opening inbound envelopes + sealing outbound
    /// input frames. Held behind Mutex because both the main thread (for
    /// Piece 1 test pattern injection via apply_frame) and the WS thread
    /// (for open/seal) need access.
    session: std::sync::Arc<std::sync::Mutex<symthaea::swarm::rdp_session::RdpSession>>,
    /// egui texture handle that holds the uploaded FrameBuffer pixels.
    texture: Option<eframe::egui::TextureHandle>,
    /// Last `frames_received` we observed — change detection for texture refresh.
    last_frames_seen: u64,
    width: u32,
    height: u32,
    /// Test pattern seed (Piece 1 carry-over; still useful as a "does
    /// the blit path work?" smoke test when the WS server isn't running).
    test_seed: u8,
    /// Channel from the WS read task → egui update loop.
    /// `try_recv()` works synchronously; we drain on every update().
    frame_rx: tokio::sync::mpsc::UnboundedReceiver<symthaea::swarm::rdp_protocol::RdpFrame>,
    /// Channel from egui pointer events → WS write task.
    input_tx: tokio::sync::mpsc::UnboundedSender<symthaea::swarm::rdp_protocol::InputFrame>,
    /// Human-readable connection status shown in the UI.
    connection_status: std::sync::Arc<std::sync::Mutex<String>>,
    /// Monotonic sequence counter for outbound InputFrames.
    input_seq: u64,
    /// Human-readable status line shown in the UI.
    status: String,
    /// Tokio runtime thread handle — dropped on app exit so the runtime
    /// shuts down cleanly. `Option` so `take()` on Drop if needed.
    _ws_thread: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "holon-viewer")]
impl HolonViewerApp {
    fn new(
        width: u32,
        height: u32,
        tile_cols: u16,
        tile_rows: u16,
        transport: ViewerTransport,
        endpoint: String,
        ctx: eframe::egui::Context,
    ) -> Self {
        use std::sync::{Arc, Mutex};
        use symthaea::swarm::rdp_holon_bridge::HolonRdpViewer;
        use symthaea::swarm::rdp_protocol::RdpSessionConfig;
        use symthaea::swarm::rdp_session::RdpSession;

        let mut viewer = HolonRdpViewer::new(width, height, tile_cols, tile_rows);
        viewer.start();

        // Piece 3 placeholder: a fixed 32-byte key both sides must agree on
        // out-of-band. NOT SAFE for any network deployment — only unblocks
        // localhost development until the PQC handshake (Track 2.5) lands.
        let mut session = RdpSession::new(
            "holon-viewer".into(),
            "holon-server".into(),
            RdpSessionConfig::default(),
            true, // is_initiator
        );
        session.on_connected();
        session.on_handshake_complete([0x42; 32]);
        let session = Arc::new(Mutex::new(session));

        // Two unbounded channels: frames flow in from WS → egui,
        // input flows out from egui → WS. Unbounded because the producer
        // side is always the slower side (ADB capture ~4fps, user clicks
        // ~1Hz), so backpressure would just drop events we care about.
        let (frame_tx, frame_rx) =
            tokio::sync::mpsc::unbounded_channel::<symthaea::swarm::rdp_protocol::RdpFrame>();
        let (input_tx, input_rx) =
            tokio::sync::mpsc::unbounded_channel::<symthaea::swarm::rdp_protocol::InputFrame>();

        let connection_status = Arc::new(Mutex::new("connecting...".to_string()));

        // Spawn the WS client on a dedicated tokio runtime thread. The
        // runtime is created inside the thread so it's owned + dropped
        // there; no need to expose it to the main thread.
        let ws_thread = {
            let session = session.clone();
            let connection_status = connection_status.clone();
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name("holon-transport-client".into())
                .spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            if let Ok(mut s) = connection_status.lock() {
                                *s = format!("runtime build failed: {e}");
                            }
                            return;
                        }
                    };
                    match transport {
                        ViewerTransport::Ws => rt.block_on(ws_client_task(
                            endpoint,
                            session,
                            frame_tx,
                            input_rx,
                            connection_status,
                            ctx,
                        )),
                        ViewerTransport::Quic => rt.block_on(quic_client_task(
                            endpoint,
                            session,
                            frame_tx,
                            input_rx,
                            connection_status,
                            ctx,
                        )),
                    }
                })
                .ok()
        };

        Self {
            viewer,
            session,
            texture: None,
            last_frames_seen: 0,
            width,
            height,
            test_seed: 0,
            frame_rx,
            input_tx,
            connection_status,
            input_seq: 0,
            status: format!(
                "Viewer ready. Frame buffer: {width}×{height}, tile grid: {tile_cols}×{tile_rows}. Connecting via {:?}...",
                transport
            ),
            _ws_thread: ws_thread,
        }
    }

    /// Inject a synthetic `FullFrame` into the viewer (Piece 1 smoke test).
    ///
    /// Still useful even with Piece 2 wired up — lets you verify the blit
    /// path without needing a WS server running on the other end.
    fn load_test_pattern(&mut self) {
        use symthaea::swarm::rdp_codec::TILE_SIZE;
        use symthaea::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpFrame};

        let cols = self.viewer.frame_buffer.tile_cols as usize;
        let rows = self.viewer.frame_buffer.tile_rows as usize;
        let tile_size = TILE_SIZE;
        let seed = self.test_seed;

        let patches: Vec<QuantizedPatch> = (0..(cols * rows))
            .map(|idx| {
                let tile_x = (idx % cols) as u8;
                let tile_y = (idx / cols) as u8;
                let values: Vec<i8> = (0..(tile_size * tile_size))
                    .map(|p| {
                        let px = (p % tile_size) as u8;
                        let py = (p / tile_size) as u8;
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

        let full = FullFrame {
            frame_id: self.last_frames_seen + 1,
            timestamp_ms: 0,
            patch_cols: cols as u16,
            patch_rows: rows as u16,
            patches,
            consciousness_level: 0.65,
            harmony: "test-pattern".into(),
        };

        let frame = RdpFrame::Full(full);
        if self.viewer.apply_frame(&frame) {
            self.test_seed = self.test_seed.wrapping_add(7);
            self.status = format!(
                "Test pattern loaded. frames_received={}",
                self.viewer.frames_received
            );
        }
    }

    /// Refresh the egui texture from the viewer's frame buffer.
    fn refresh_texture(&mut self, ctx: &eframe::egui::Context) {
        use eframe::egui::{ColorImage, TextureOptions};

        let rgba = self.viewer.frame_buffer.as_rgba();
        let image =
            ColorImage::from_rgba_unmultiplied([self.width as usize, self.height as usize], rgba);

        match self.texture.as_mut() {
            Some(handle) => handle.set(image, TextureOptions::default()),
            None => {
                self.texture =
                    Some(ctx.load_texture("holon_frame", image, TextureOptions::default()));
            }
        }
    }

    /// Drain the frame channel (non-blocking) and apply every received
    /// `RdpFrame` to the viewer. Called once per egui `update()`.
    fn drain_frame_channel(&mut self) {
        use tokio::sync::mpsc::error::TryRecvError;
        loop {
            match self.frame_rx.try_recv() {
                Ok(frame) => {
                    self.viewer.apply_frame(&frame);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
    }
}

#[cfg(feature = "holon-viewer")]
impl eframe::App for HolonViewerApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        use eframe::egui;

        // 1. Drain any frames that arrived from the WS task since last tick.
        self.drain_frame_channel();

        // 2. Refresh the texture if new frames arrived.
        if self.viewer.frames_received != self.last_frames_seen {
            self.last_frames_seen = self.viewer.frames_received;
            self.refresh_texture(ctx);
        }

        // Snapshot connection status for display.
        let conn_status = self
            .connection_status
            .lock()
            .ok()
            .map(|s| s.clone())
            .unwrap_or_else(|| "mutex poisoned".to_string());

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Holon RDP Viewer");
                ui.separator();
                ui.label(format!(
                    "{}×{} · frames={}",
                    self.width, self.height, self.viewer.frames_received
                ));
                ui.separator();
                ui.label(format!("ws: {conn_status}"));
                if ui.button("Load test pattern").clicked() {
                    self.load_test_pattern();
                }
            });
            ui.label(&self.status);
        });

        // 3. Render the frame buffer.
        egui::CentralPanel::default().show(ctx, |ui| {
            let image_response = match self.texture.as_ref() {
                Some(texture) => {
                    let available = ui.available_size();
                    let aspect = self.width as f32 / self.height as f32;
                    let w = available.x.min(available.y * aspect);
                    let h = w / aspect;
                    Some(
                        ui.add(
                            egui::Image::new((texture.id(), egui::vec2(w, h)))
                                .sense(egui::Sense::click_and_drag()),
                        ),
                    )
                }
                None => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label("(no frame yet — WS connecting or load test pattern)");
                    });
                    None
                }
            };

            // 4. Pointer events → InputFrame → WS write task.
            if let Some(response) = image_response {
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        // Map the click position from egui's image rect
                        // back to the 0.0..1.0 normalized space the
                        // InputFrame expects.
                        let rect = response.rect;
                        let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                        let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);
                        self.send_pointer(nx, ny, true);
                    }
                }
            }
        });
    }
}

#[cfg(feature = "holon-viewer")]
impl HolonViewerApp {
    /// Build and send a `Pointer` InputFrame to the WS write task.
    ///
    /// Normalized coords (0.0..1.0) — the remote end denormalizes to
    /// native phone screen pixels. Matches the pattern in
    /// `examples/phone_rdp_share.rs::input_frame_to_action`.
    fn send_pointer(&mut self, nx: f32, ny: f32, pressed: bool) {
        use symthaea::swarm::rdp_protocol::{InputEvent, InputFrame};

        self.input_seq += 1;
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let frame = InputFrame {
            sequence: self.input_seq,
            timestamp_ms,
            events: vec![InputEvent::Pointer {
                x: nx,
                y: ny,
                button: 1,
                pressed,
            }],
        };

        if let Err(e) = self.input_tx.send(frame) {
            self.status = format!("input_tx send failed: {e}");
        } else {
            self.status = format!("sent pointer ({nx:.2}, {ny:.2}) seq={}", self.input_seq);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// WS client task — runs inside the dedicated tokio runtime thread
// ═══════════════════════════════════════════════════════════════════════

#[cfg(feature = "holon-viewer")]
async fn ws_client_task(
    url: String,
    session: std::sync::Arc<std::sync::Mutex<symthaea::swarm::rdp_session::RdpSession>>,
    frame_tx: tokio::sync::mpsc::UnboundedSender<symthaea::swarm::rdp_protocol::RdpFrame>,
    mut input_rx: tokio::sync::mpsc::UnboundedReceiver<symthaea::swarm::rdp_protocol::InputFrame>,
    connection_status: std::sync::Arc<std::sync::Mutex<String>>,
    ctx: eframe::egui::Context,
) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::{connect_async, tungstenite::Message};

    // Set status → connecting.
    if let Ok(mut s) = connection_status.lock() {
        *s = format!("connecting to {url}");
    }

    // Establish the WebSocket connection.
    let (ws_stream, _resp) = match connect_async(&url).await {
        Ok(x) => x,
        Err(e) => {
            if let Ok(mut s) = connection_status.lock() {
                *s = format!("connect failed: {e}");
            }
            return;
        }
    };

    if let Ok(mut s) = connection_status.lock() {
        *s = "connected".to_string();
    }
    ctx.request_repaint();

    let (mut ws_sink, mut ws_stream) = ws_stream.split();

    // Read task: pulls binary messages, opens via RdpSession, forwards
    // to the egui frame channel + requests a repaint.
    let read_ctx = ctx.clone();
    let read_session = session.clone();
    let read_status = connection_status.clone();
    let read_task = async move {
        while let Some(msg) = ws_stream.next().await {
            match msg {
                Ok(Message::Binary(bytes)) => {
                    // `rdp_wire::open_frame` takes `&mut RdpSession`, so
                    // we need the lock. Lock contention is minimal — egui
                    // only holds it for send_pointer's seal path.
                    let opened = {
                        let mut session = match read_session.lock() {
                            Ok(s) => s,
                            Err(_) => return, // poisoned
                        };
                        symthaea::swarm::rdp_wire::open_frame(&bytes, &mut session)
                    };
                    match opened {
                        Ok(frame) => {
                            if frame_tx.send(frame).is_err() {
                                // egui side dropped; exit.
                                return;
                            }
                            read_ctx.request_repaint();
                        }
                        Err(e) => {
                            if let Ok(mut s) = read_status.lock() {
                                *s = format!("open_frame error: {e}");
                            }
                        }
                    }
                }
                Ok(Message::Close(_)) => {
                    if let Ok(mut s) = read_status.lock() {
                        *s = "remote closed".to_string();
                    }
                    return;
                }
                Ok(_) => {
                    // Ignore Text/Ping/Pong — telemetry text goes there.
                }
                Err(e) => {
                    if let Ok(mut s) = read_status.lock() {
                        *s = format!("ws recv error: {e}");
                    }
                    return;
                }
            }
        }
    };

    // Write task: drains the input channel, seals each InputFrame, sends
    // as Message::Binary.
    let write_session = session.clone();
    let write_status = connection_status.clone();
    let write_task = async move {
        while let Some(input) = input_rx.recv().await {
            let sealed = {
                let mut session = match write_session.lock() {
                    Ok(s) => s,
                    Err(_) => return,
                };
                symthaea::swarm::rdp_wire::seal_input(&input, &mut session)
            };
            match sealed {
                Ok(bytes) => {
                    if ws_sink.send(Message::Binary(bytes.into())).await.is_err() {
                        if let Ok(mut s) = write_status.lock() {
                            *s = "ws send failed".to_string();
                        }
                        return;
                    }
                }
                Err(e) => {
                    if let Ok(mut s) = write_status.lock() {
                        *s = format!("seal_input error: {e}");
                    }
                }
            }
        }
    };

    // Race the two tasks — first one to complete (disconnect / error)
    // ends the whole client.
    tokio::select! {
        _ = read_task => {}
        _ = write_task => {}
    }
}

#[cfg(feature = "holon-viewer")]
async fn quic_client_task(
    endpoint: String,
    session: std::sync::Arc<std::sync::Mutex<symthaea::swarm::rdp_session::RdpSession>>,
    frame_tx: tokio::sync::mpsc::UnboundedSender<symthaea::swarm::rdp_protocol::RdpFrame>,
    input_rx: tokio::sync::mpsc::UnboundedReceiver<symthaea::swarm::rdp_protocol::InputFrame>,
    connection_status: std::sync::Arc<std::sync::Mutex<String>>,
    ctx: eframe::egui::Context,
) {
    use std::sync::Arc;

    let (remote_addr, server_name) =
        match symthaea::swarm::quic_transport::resolve_quic_endpoint(&endpoint, 7779).await {
            Ok(value) => value,
            Err(error) => {
                if let Ok(mut status) = connection_status.lock() {
                    *status = format!("invalid QUIC endpoint: {error}");
                }
                return;
            }
        };

    let repaint: Arc<dyn Fn() + Send + Sync> = Arc::new(move || ctx.request_repaint());
    if let Err(error) = symthaea::swarm::quic_transport::run_viewer_quic_client(
        remote_addr,
        &server_name,
        session,
        frame_tx,
        input_rx,
        connection_status.clone(),
        repaint,
    )
    .await
    {
        if let Ok(mut status) = connection_status.lock() {
            *status = format!("quic client failed: {error}");
        }
    }
}

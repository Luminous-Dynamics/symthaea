// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Holon RDP Viewer — egui desktop client for the Phase I.A binary wire.
//!
//! ## Status — Phase I.A.2 execution in progress
//!
//! **Piece 1 (egui window shell)**: ✅ this file. Creates an eframe window
//! that owns a `HolonRdpViewer` and blits its `FrameBuffer` to an
//! `egui::TextureHandle` on every `update()`. Exposes a "Load test
//! pattern" button that injects a synthetic `FullFrame` so Piece 1 is
//! runnable and visually verifiable before Piece 2 wires in a real
//! WebSocket source.
//!
//! **Piece 2 (tokio-tungstenite WS client)**: ⏳ next commit. Replaces
//! the synthetic pattern with a live stream from `ws://localhost:7778`.
//!
//! **Piece 3 (PQC handshake unblock)**: ⏳ later commit. Swaps the
//! placeholder `[0x42; 32]` session key for a real KEM-derived key.
//! Blocked on `service.rs:844` TODO. Safe for localhost deployment
//! until then.
//!
//! ## Three pieces must land together (documented in scaffold commit)
//!
//! See the commit message on `c40b218081` and the worktree commit
//! `2c535fd2fc` for the coupling argument. Splitting these three pieces
//! across independent sessions would force the tests to be written twice
//! and leave a half-secured wire on disk.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --release --example holon_rdp_viewer --features holon-viewer
//! ```

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

    // Default viewer resolution matches the Pixel 8 Pro native screen at
    // the 128×128 vision manifold target times the 64-pixel codec tile size
    // — 128 * 64 = 8192 is far larger than the phone, so we use the actual
    // Pixel resolution 1008×2244 which gives ~16×35 tile grid at 64-pixel
    // tiles.
    const PHONE_W: u32 = 1008;
    const PHONE_H: u32 = 2244;
    const TILE_COLS: u16 = 16; // 1008 / 64 = 15.75 → 16
    const TILE_ROWS: u16 = 35; // 2244 / 64 = 35.06 → 35 (rounds down, last row may clip)

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // Scale down 2× for desktop display (phone is 1008×2244, that's taller
            // than most monitors) — the texture blit will downscale on upload.
            .with_inner_size([540.0, 1200.0])
            .with_min_inner_size([400.0, 600.0])
            .with_title("Holon RDP Viewer — Phase I.A.2 Piece 1"),
        ..Default::default()
    };

    eframe::run_native(
        "Holon RDP Viewer",
        native_options,
        Box::new(|_cc| Ok(Box::new(HolonViewerApp::new(PHONE_W, PHONE_H, TILE_COLS, TILE_ROWS)))),
    )
}

#[cfg(feature = "holon-viewer")]
struct HolonViewerApp {
    /// The RDP frame buffer — receives `apply_full_frame`/`apply_delta_frame`
    /// from either the test-pattern injector (Piece 1) or the WS client
    /// (Piece 2, next commit).
    viewer: symthaea::swarm::rdp_holon_bridge::HolonRdpViewer,
    /// egui texture handle that holds the uploaded FrameBuffer pixels.
    /// Re-uploaded on every update() when the buffer has new content.
    texture: Option<eframe::egui::TextureHandle>,
    /// Last `frames_received` we observed on the viewer — used to detect
    /// when a new frame has arrived and the texture needs refreshing.
    last_frames_seen: u64,
    /// Cached width and height of the frame buffer for texture sizing.
    width: u32,
    height: u32,
    /// Test pattern seed so repeated clicks produce visibly different
    /// patterns (Piece 1 only — Piece 2 removes this).
    test_seed: u8,
    /// Human-readable status line shown in the UI.
    status: String,
}

#[cfg(feature = "holon-viewer")]
impl HolonViewerApp {
    fn new(width: u32, height: u32, tile_cols: u16, tile_rows: u16) -> Self {
        use symthaea::swarm::rdp_holon_bridge::HolonRdpViewer;
        let mut viewer = HolonRdpViewer::new(width, height, tile_cols, tile_rows);
        viewer.start();
        Self {
            viewer,
            texture: None,
            last_frames_seen: 0,
            width,
            height,
            test_seed: 0,
            status: format!(
                "Piece 1 ready. Frame buffer: {width}×{height}, tile grid: {tile_cols}×{tile_rows}. Click 'Load test pattern' to verify the blit path."
            ),
        }
    }

    /// Inject a synthetic `FullFrame` into the viewer's `FrameBuffer`.
    ///
    /// This exercises the same `HolonRdpViewer::apply_frame` code path that
    /// Piece 2 will drive from real WebSocket frames, just with a locally
    /// constructed `RdpFrame::Full` whose patches are deterministic
    /// gradients based on `self.test_seed`. Lets us visually verify the
    /// full Piece 1 stack (RdpFrame → FrameBuffer → egui texture → screen)
    /// before any network code exists.
    fn load_test_pattern(&mut self) {
        use symthaea::swarm::rdp_codec::TILE_SIZE;
        use symthaea::swarm::rdp_protocol::{FullFrame, QuantizedPatch, RdpFrame};

        let cols = self.viewer.frame_buffer.tile_cols as usize;
        let rows = self.viewer.frame_buffer.tile_rows as usize;
        let tile_size = TILE_SIZE;
        let seed = self.test_seed;

        // Build tile_cols × tile_rows patches. Each patch is a gradient
        // that varies spatially so adjacent tiles look distinct after
        // the per-tile i8 dequantization in `FrameBuffer::apply_full_frame`.
        let patches: Vec<QuantizedPatch> = (0..(cols * rows))
            .map(|idx| {
                let tile_x = (idx % cols) as u8;
                let tile_y = (idx / cols) as u8;
                let values: Vec<i8> = (0..(tile_size * tile_size))
                    .map(|p| {
                        let px = (p % tile_size) as u8;
                        let py = (p / tile_size) as u8;
                        // Gradient that wraps through i8 range — gives
                        // visible tile boundaries and shifts with seed.
                        let v = tile_x
                            .wrapping_add(tile_y.wrapping_mul(3))
                            .wrapping_add(px.wrapping_mul(2))
                            .wrapping_add(py)
                            .wrapping_add(seed);
                        // Map u8 0..255 → i8 -128..127 so apply_full_frame
                        // treats it as a valid quantized patch.
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
        let applied = self.viewer.apply_frame(&frame);
        if applied {
            self.test_seed = self.test_seed.wrapping_add(7);
            self.status = format!(
                "Test pattern loaded. frame_id={}, frames_received={}, seed={}",
                self.last_frames_seen + 1,
                self.viewer.frames_received,
                self.test_seed
            );
        } else {
            self.status = "apply_frame returned false — check viewer state".into();
        }
    }

    /// Refresh the egui texture from the viewer's frame buffer.
    ///
    /// Called from `update()` whenever `frames_received` has advanced.
    /// Uses `egui::ColorImage::from_rgba_unmultiplied` to wrap the RGBA
    /// bytes produced by `FrameBuffer::as_rgba()`, then uploads via
    /// `Context::load_texture` (or re-uploads the existing handle).
    fn refresh_texture(&mut self, ctx: &eframe::egui::Context) {
        use eframe::egui::{ColorImage, TextureOptions};

        let rgba = self.viewer.frame_buffer.as_rgba();
        let image = ColorImage::from_rgba_unmultiplied(
            [self.width as usize, self.height as usize],
            rgba,
        );

        match self.texture.as_mut() {
            Some(handle) => handle.set(image, TextureOptions::default()),
            None => {
                self.texture = Some(ctx.load_texture(
                    "holon_frame",
                    image,
                    TextureOptions::default(),
                ));
            }
        }
    }
}

#[cfg(feature = "holon-viewer")]
impl eframe::App for HolonViewerApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        use eframe::egui;

        // Detect new frames and refresh the texture.
        if self.viewer.frames_received != self.last_frames_seen {
            self.last_frames_seen = self.viewer.frames_received;
            self.refresh_texture(ctx);
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Holon RDP Viewer");
                ui.separator();
                ui.label(format!(
                    "{}×{} · frames={}",
                    self.width, self.height, self.viewer.frames_received
                ));
                if ui.button("Load test pattern").clicked() {
                    self.load_test_pattern();
                }
            });
            ui.label(&self.status);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            match self.texture.as_ref() {
                Some(texture) => {
                    // Fit the phone-resolution texture into the available
                    // panel by preserving aspect ratio.
                    let available = ui.available_size();
                    let aspect = self.width as f32 / self.height as f32;
                    let w = available.x.min(available.y * aspect);
                    let h = w / aspect;
                    ui.image((texture.id(), egui::vec2(w, h)));
                }
                None => {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label("(no frame received yet — click 'Load test pattern')");
                    });
                }
            }
        });
    }
}

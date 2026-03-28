// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Egui renderer for Sovereign RDP client.
//!
//! Renders the reconstructed `FrameBuffer` as a GPU texture in an egui panel,
//! captures pointer and keyboard events as `InputEvent`s, and displays a
//! consciousness-aware status overlay.
//!
//! ## Feature Gate
//!
//! Requires `gui` feature (egui + eframe).

#[cfg(feature = "gui")]
use eframe::egui;

use super::rdp_client::FrameBuffer;
use super::rdp_protocol::InputEvent;
use super::rdp_session::SessionState;

/// RDP renderer state for egui integration.
#[cfg(feature = "gui")]
pub struct RdpEguiRenderer {
    /// GPU texture handle for the remote desktop frame.
    texture: Option<egui::TextureHandle>,
    /// Collected input events this frame (drained by caller).
    pending_input: Vec<InputEvent>,
    /// Display name for the connection.
    pub connection_name: String,
}

#[cfg(feature = "gui")]
impl RdpEguiRenderer {
    /// Create a new renderer.
    pub fn new(connection_name: &str) -> Self {
        Self {
            texture: None,
            pending_input: Vec::new(),
            connection_name: connection_name.to_string(),
        }
    }

    /// Render the remote desktop and capture input.
    ///
    /// Call this within an egui panel. Returns whether the frame was updated.
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        frame_buffer: &FrameBuffer,
        session_state: SessionState,
        server_consciousness: f32,
        current_fps: f32,
    ) -> bool {
        let mut updated = false;

        // Status bar at top.
        ui.horizontal(|ui| {
            // Connection status indicator.
            let (color, label) = match session_state {
                SessionState::Active => (egui::Color32::from_rgb(68, 255, 136), "ACTIVE"),
                SessionState::ViewOnly => (egui::Color32::from_rgb(255, 170, 68), "VIEW ONLY"),
                SessionState::Connecting | SessionState::Handshaking => {
                    (egui::Color32::from_rgb(68, 136, 255), "CONNECTING")
                }
                SessionState::Authenticating => {
                    (egui::Color32::from_rgb(68, 136, 255), "AUTHENTICATING")
                }
                SessionState::Rekeying => (egui::Color32::from_rgb(170, 68, 255), "REKEYING"),
                SessionState::Closing | SessionState::Closed => {
                    (egui::Color32::from_rgb(255, 68, 68), "DISCONNECTED")
                }
            };
            ui.colored_label(color, format!("● {}", label));
            ui.separator();

            // Consciousness level.
            ui.label(format!("Φ {:.1}%", server_consciousness * 100.0));
            ui.separator();

            // FPS.
            ui.label(format!("{:.0} FPS", current_fps));
            ui.separator();

            // Connection name.
            ui.label(&self.connection_name);
        });

        ui.separator();

        // Remote desktop frame.
        if frame_buffer.has_content {
            // Upload pixel buffer as egui texture.
            let image = egui::ColorImage::from_rgba_unmultiplied(
                [frame_buffer.width as usize, frame_buffer.height as usize],
                &frame_buffer.pixels,
            );

            let texture = self.texture.get_or_insert_with(|| {
                ui.ctx()
                    .load_texture("rdp-frame", image.clone(), egui::TextureOptions::NEAREST)
            });
            texture.set(image, egui::TextureOptions::NEAREST);

            // Render the texture filling available space.
            let available = ui.available_size();
            let aspect = frame_buffer.width as f32 / frame_buffer.height as f32;
            let (render_w, render_h) = if available.x / available.y > aspect {
                (available.y * aspect, available.y)
            } else {
                (available.x, available.x / aspect)
            };

            let (response, _painter) = ui.allocate_painter(
                egui::Vec2::new(render_w, render_h),
                egui::Sense::click_and_drag(),
            );

            // Draw the texture.
            let rect = response.rect;
            ui.painter().image(
                texture.id(),
                rect,
                egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );

            // Capture pointer input (normalized to 0.0-1.0 within the image).
            if let Some(pos) = response.interact_pointer_pos() {
                let nx = ((pos.x - rect.min.x) / rect.width()).clamp(0.0, 1.0);
                let ny = ((pos.y - rect.min.y) / rect.height()).clamp(0.0, 1.0);

                let button = if response.clicked() || response.dragged() {
                    1u8
                } else if response.secondary_clicked() {
                    2u8
                } else {
                    0u8
                };

                self.pending_input.push(InputEvent::Pointer {
                    x: nx,
                    y: ny,
                    button,
                    pressed: response.clicked() || response.dragged(),
                });
            }

            updated = true;
        } else {
            // No content yet — show connecting message.
            ui.centered_and_justified(|ui| {
                ui.label(
                    egui::RichText::new("Waiting for first frame...")
                        .size(24.0)
                        .color(egui::Color32::from_rgb(100, 100, 180)),
                );
            });
        }

        // Capture keyboard input.
        ui.input(|input| {
            for event in &input.events {
                match event {
                    egui::Event::Key {
                        key,
                        pressed,
                        modifiers,
                        ..
                    } => {
                        let code = egui_key_to_code(key);
                        let mods = egui_modifiers_to_flags(modifiers);
                        self.pending_input.push(InputEvent::Key {
                            code,
                            pressed: *pressed,
                            modifiers: mods,
                        });
                    }
                    _ => {}
                }
            }
        });

        updated
    }

    /// Drain collected input events.
    pub fn drain_input(&mut self) -> Vec<InputEvent> {
        std::mem::take(&mut self.pending_input)
    }
}

/// Convert egui Key to a platform-independent key code.
/// Uses a simple mapping — extend as needed.
#[cfg(feature = "gui")]
fn egui_key_to_code(key: &egui::Key) -> u32 {
    match key {
        egui::Key::A => 65,
        egui::Key::B => 66,
        egui::Key::C => 67,
        egui::Key::D => 68,
        egui::Key::E => 69,
        egui::Key::F => 70,
        egui::Key::G => 71,
        egui::Key::H => 72,
        egui::Key::I => 73,
        egui::Key::J => 74,
        egui::Key::K => 75,
        egui::Key::L => 76,
        egui::Key::M => 77,
        egui::Key::N => 78,
        egui::Key::O => 79,
        egui::Key::P => 80,
        egui::Key::Q => 81,
        egui::Key::R => 82,
        egui::Key::S => 83,
        egui::Key::T => 84,
        egui::Key::U => 85,
        egui::Key::V => 86,
        egui::Key::W => 87,
        egui::Key::X => 88,
        egui::Key::Y => 89,
        egui::Key::Z => 90,
        egui::Key::Num0 => 48,
        egui::Key::Num1 => 49,
        egui::Key::Num2 => 50,
        egui::Key::Num3 => 51,
        egui::Key::Num4 => 52,
        egui::Key::Num5 => 53,
        egui::Key::Num6 => 54,
        egui::Key::Num7 => 55,
        egui::Key::Num8 => 56,
        egui::Key::Num9 => 57,
        egui::Key::Space => 32,
        egui::Key::Enter => 13,
        egui::Key::Escape => 27,
        egui::Key::Tab => 9,
        egui::Key::Backspace => 8,
        egui::Key::Delete => 127,
        egui::Key::ArrowUp => 0x100,
        egui::Key::ArrowDown => 0x101,
        egui::Key::ArrowLeft => 0x102,
        egui::Key::ArrowRight => 0x103,
        _ => 0,
    }
}

/// Convert egui Modifiers to flags byte.
#[cfg(feature = "gui")]
fn egui_modifiers_to_flags(mods: &egui::Modifiers) -> u8 {
    let mut flags = 0u8;
    if mods.shift {
        flags |= 1;
    }
    if mods.ctrl {
        flags |= 2;
    }
    if mods.alt {
        flags |= 4;
    }
    if mods.mac_cmd || mods.command {
        flags |= 8;
    }
    flags
}

// Non-gui stub so the module compiles without the feature.
#[cfg(not(feature = "gui"))]
pub struct RdpEguiRenderer;

#[cfg(not(feature = "gui"))]
impl RdpEguiRenderer {
    pub fn new(_name: &str) -> Self {
        Self
    }
}

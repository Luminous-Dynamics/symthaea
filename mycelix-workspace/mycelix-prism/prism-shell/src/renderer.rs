// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GPU renderer for Prism using vello + wgpu.
//!
//! Features:
//! - E: Visual security chrome (zone badge, safety indicator, threat count)
//! - F: Reader mode styling (warm background, comfortable typography)
//! - Handles all PaintContent variants (Text, Rule, Fill)

use std::sync::Arc;

use anyhow::Result;
use vello::kurbo::Stroke;
use vello::kurbo::{Affine, Line, Rect, RoundedRect, RoundedRectRadii};
use vello::peniko::{Blob, Color, Fill, FontData};
use vello::util::{RenderContext, RenderSurface};
use vello::wgpu;
use vello::{AaConfig, Glyph, RenderParams, Renderer, RendererOptions, Scene};
use winit::window::Window;

use prism_common::{ContentZone, SafetyLevel};
use prism_layout::{PaintContent, PaintRect};

const FONT_REGULAR: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");
const FONT_BOLD: &[u8] = include_bytes!("../assets/fonts/DejaVuSans-Bold.ttf");

/// Chrome bar height in pixels.
const CHROME_HEIGHT: f64 = 48.0;
/// Address bar height.
const ADDRESS_BAR_HEIGHT: f64 = 32.0;
/// Total chrome (top bar + address bar).
pub const TOTAL_CHROME: f64 = CHROME_HEIGHT + ADDRESS_BAR_HEIGHT + 4.0;

/// Security state displayed in the chrome.
#[derive(Debug, Clone)]
pub struct SecurityState {
    pub zone: ContentZone,
    pub safety_level: SafetyLevel,
    pub threat_count: usize,
    pub url: String,
    pub title: String,
    /// If Some, show this text in address bar instead of url (user is typing).
    pub address_bar_text: Option<String>,
    /// Whether address bar is focused.
    pub address_bar_focused: bool,
}

impl Default for SecurityState {
    fn default() -> Self {
        Self {
            zone: ContentZone::Local,
            safety_level: SafetyLevel::Green,
            threat_count: 0,
            url: "prism://welcome".to_string(),
            title: "Welcome to Prism".to_string(),
            address_bar_text: None,
            address_bar_focused: false,
        }
    }
}

pub struct PrismRenderer {
    pub context: RenderContext,
    pub renderers: Vec<Option<Renderer>>,
    pub scene: Scene,
    font_regular: FontData,
    font_bold: FontData,
}

impl PrismRenderer {
    pub fn new() -> Self {
        Self {
            context: RenderContext::new(),
            renderers: vec![],
            scene: Scene::new(),
            font_regular: FontData::new(Blob::new(Arc::new(FONT_REGULAR.to_vec())), 0),
            font_bold: FontData::new(Blob::new(Arc::new(FONT_BOLD.to_vec())), 0),
        }
    }

    pub fn create_surface(&mut self, window: Arc<Window>) -> Result<RenderSurface<'static>> {
        let size = window.inner_size();
        let surface = pollster::block_on(self.context.create_surface(
            window,
            size.width.max(1),
            size.height.max(1),
            wgpu::PresentMode::AutoVsync,
        ))?;

        self.renderers
            .resize_with(self.context.devices.len(), || None);
        self.renderers[surface.dev_id].get_or_insert_with(|| {
            Renderer::new(
                &self.context.devices[surface.dev_id].device,
                RendererOptions::default(),
            )
            .expect("Failed to create vello Renderer")
        });

        Ok(surface)
    }

    pub fn render(
        &mut self,
        surface: &mut RenderSurface<'_>,
        paint_rects: &[PaintRect],
        scroll_y: f32,
        security: &SecurityState,
    ) -> Result<()> {
        let w = surface.config.width;
        let h = surface.config.height;

        self.scene.reset();

        // Page background — warm off-white for reader mode
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new([0.98, 0.97, 0.95, 1.0]),
            None,
            &Rect::new(0.0, TOTAL_CHROME, w as f64, h as f64),
        );

        // Content area
        self.draw_content(paint_rects, scroll_y, w as f64, h as f64);

        // Chrome (drawn OVER content so it doesn't scroll)
        self.draw_chrome(w as f64, h as f64, security);

        // GPU render
        let device_handle = &self.context.devices[surface.dev_id];
        self.renderers[surface.dev_id]
            .as_mut()
            .unwrap()
            .render_to_texture(
                &device_handle.device,
                &device_handle.queue,
                &self.scene,
                &surface.target_view,
                &RenderParams {
                    base_color: Color::new([0.06, 0.12, 0.18, 1.0]),
                    width: w,
                    height: h,
                    antialiasing_method: AaConfig::Area,
                },
            )?;

        let surface_texture = surface.surface.get_current_texture()?;
        let mut encoder =
            device_handle
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Prism Blit"),
                });
        surface.blitter.copy(
            &device_handle.device,
            &mut encoder,
            &surface.target_view,
            &surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default()),
        );
        device_handle.queue.submit([encoder.finish()]);
        surface_texture.present();
        let _ = device_handle.device.poll(wgpu::PollType::Poll);

        Ok(())
    }

    // ── Chrome (Feature E: Visual Security Feedback) ──

    fn draw_chrome(&mut self, w: f64, _h: f64, security: &SecurityState) {
        // Top bar background
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new([0.08, 0.14, 0.20, 1.0]),
            None,
            &Rect::new(0.0, 0.0, w, CHROME_HEIGHT),
        );

        // Brand
        self.draw_text("Prism", 14.0, 30.0, 18.0, true, [0.35, 0.85, 0.80, 1.0]);

        // Title (truncated)
        let title = if security.title.len() > 50 {
            format!("{}...", &security.title[..47])
        } else {
            security.title.clone()
        };
        self.draw_text(&title, 110.0, 30.0, 13.0, false, [0.7, 0.75, 0.78, 1.0]);

        // Safety indicator dot (right side)
        let (safety_color, safety_label) = match security.safety_level {
            SafetyLevel::Green => ([0.2, 0.8, 0.4, 1.0], "Safe"),
            SafetyLevel::Yellow => ([0.9, 0.8, 0.2, 1.0], "Caution"),
            SafetyLevel::Orange => ([0.95, 0.55, 0.15, 1.0], "Warning"),
            SafetyLevel::Red => ([0.9, 0.2, 0.2, 1.0], "Danger"),
        };
        let dot_x = w - 80.0;
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new(safety_color),
            None,
            &vello::kurbo::Circle::new((dot_x, 24.0), 5.0),
        );
        self.draw_text(safety_label, dot_x + 10.0, 28.0, 11.0, false, safety_color);

        // Threat count badge (if any)
        if security.threat_count > 0 {
            let badge_x = dot_x - 50.0;
            self.scene.fill(
                Fill::NonZero,
                Affine::IDENTITY,
                Color::new([0.8, 0.15, 0.15, 1.0]),
                None,
                &RoundedRect::new(badge_x, 14.0, badge_x + 36.0, 34.0, 10.0),
            );
            self.draw_text(
                &format!("{}", security.threat_count),
                badge_x + 12.0,
                29.0,
                12.0,
                true,
                [1.0, 1.0, 1.0, 1.0],
            );
        }

        // Address bar area
        let bar_y = CHROME_HEIGHT;
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new([0.12, 0.18, 0.24, 1.0]),
            None,
            &Rect::new(0.0, bar_y, w, bar_y + ADDRESS_BAR_HEIGHT),
        );

        // Zone badge (left of address bar)
        let (zone_color, zone_label) = match security.zone {
            ContentZone::Private => ([0.8, 0.2, 0.2, 1.0], "PRIVATE"),
            ContentZone::Local => ([0.85, 0.7, 0.2, 1.0], "LOCAL"),
            ContentZone::Public => ([0.2, 0.7, 0.4, 1.0], "PUBLIC"),
        };
        let badge_y = bar_y + 4.0;
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new(zone_color),
            None,
            &RoundedRect::new(10.0, badge_y, 80.0, badge_y + 24.0, 4.0),
        );
        self.draw_text(
            zone_label,
            16.0,
            badge_y + 17.0,
            10.0,
            true,
            [1.0, 1.0, 1.0, 1.0],
        );

        // Address bar
        let url_x = 90.0;
        let bar_bg = if security.address_bar_focused {
            [0.25, 0.31, 0.38, 1.0] // Lighter when focused
        } else {
            [0.18, 0.24, 0.30, 1.0]
        };
        self.scene.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::new(bar_bg),
            None,
            &RoundedRect::new(url_x, badge_y, w - 10.0, badge_y + 24.0, 4.0),
        );
        let display_text = security
            .address_bar_text
            .as_deref()
            .unwrap_or(&security.url);
        let display_text = if display_text.len() > 80 {
            &display_text[..80]
        } else {
            display_text
        };
        let text_color = if security.address_bar_focused {
            [0.9, 0.92, 0.95, 1.0]
        } else {
            [0.7, 0.75, 0.8, 1.0]
        };
        self.draw_text(
            display_text,
            url_x + 8.0,
            badge_y + 17.0,
            12.0,
            false,
            text_color,
        );

        // Blinking cursor when focused
        if security.address_bar_focused {
            let cursor_x = url_x + 8.0 + display_text.len() as f64 * 12.0 * 0.52;
            self.scene.stroke(
                &Stroke::new(1.5),
                Affine::IDENTITY,
                Color::new([0.4, 0.85, 0.8, 1.0]),
                None,
                &Line::new((cursor_x, badge_y + 4.0), (cursor_x, badge_y + 20.0)),
            );
        }

        // Separator line
        let sep_y = bar_y + ADDRESS_BAR_HEIGHT;
        self.scene.stroke(
            &Stroke::new(1.0),
            Affine::IDENTITY,
            Color::new([0.2, 0.26, 0.32, 1.0]),
            None,
            &Line::new((0.0, sep_y), (w, sep_y)),
        );
    }

    // ── Content Rendering ──

    fn draw_content(&mut self, rects: &[PaintRect], scroll_y: f32, _w: f64, h: f64) {
        let offset_y = TOTAL_CHROME as f32;

        for rect in rects {
            let y = rect.y - scroll_y + offset_y;

            // Cull offscreen
            if y + rect.height < 0.0 || y > h as f32 {
                continue;
            }

            match &rect.content {
                PaintContent::Text {
                    text,
                    font_size,
                    bold,
                    italic: _,
                    color,
                    is_link,
                    ..
                } => {
                    self.draw_text(
                        text,
                        rect.x as f64,
                        y as f64 + *font_size as f64,
                        *font_size,
                        *bold,
                        *color,
                    );

                    // Underline links
                    if *is_link {
                        let uw = text.len() as f64 * *font_size as f64 * 0.52;
                        let uy = y as f64 + *font_size as f64 + 2.0;
                        self.scene.stroke(
                            &Stroke::new(1.0),
                            Affine::IDENTITY,
                            Color::new(*color),
                            None,
                            &Line::new((rect.x as f64, uy), (rect.x as f64 + uw, uy)),
                        );
                    }
                }
                PaintContent::Rule { color, thickness } => {
                    self.scene.stroke(
                        &Stroke::new(*thickness as f64),
                        Affine::IDENTITY,
                        Color::new(*color),
                        None,
                        &Line::new(
                            (rect.x as f64, y as f64),
                            (rect.x as f64 + rect.width as f64, y as f64),
                        ),
                    );
                }
                PaintContent::Fill {
                    color,
                    corner_radius,
                } => {
                    let r = *corner_radius as f64;
                    self.scene.fill(
                        Fill::NonZero,
                        Affine::IDENTITY,
                        Color::new(*color),
                        None,
                        &RoundedRect::from_rect(
                            Rect::new(
                                rect.x as f64,
                                y as f64,
                                (rect.x + rect.width) as f64,
                                (y + rect.height) as f64,
                            ),
                            RoundedRectRadii::from_single_radius(r),
                        ),
                    );
                }
            }
        }
    }

    // ── Text Drawing ──

    fn draw_text(
        &mut self,
        text: &str,
        x: f64,
        y: f64,
        font_size: f32,
        bold: bool,
        color: [f32; 4],
    ) {
        let font = if bold {
            &self.font_bold
        } else {
            &self.font_regular
        };
        let vello_color = Color::new(color);

        let font_ref = match skrifa::raw::FontRef::from_index(font.data.as_ref(), 0) {
            Ok(f) => f,
            Err(_) => return,
        };

        let charmap = skrifa::MetadataProvider::charmap(&font_ref);
        let sz = skrifa::instance::Size::new(font_size);
        let glyph_metrics = skrifa::MetadataProvider::glyph_metrics(
            &font_ref,
            sz,
            skrifa::instance::LocationRef::default(),
        );

        let mut pen_x = 0.0_f32;
        let glyphs: Vec<Glyph> = text
            .chars()
            .filter_map(|ch| {
                let gid = charmap.map(ch)?;
                let advance = glyph_metrics.advance_width(gid).unwrap_or(font_size * 0.5);
                let gx = pen_x;
                pen_x += advance;
                Some(Glyph {
                    id: gid.to_u32(),
                    x: gx,
                    y: 0.0,
                })
            })
            .collect();

        if !glyphs.is_empty() {
            self.scene
                .draw_glyphs(font)
                .font_size(font_size)
                .transform(Affine::translate((x, y)))
                .glyph_transform(None)
                .brush(&vello_color)
                .hint(true)
                .draw(Fill::NonZero, glyphs.into_iter());
        }
    }
}

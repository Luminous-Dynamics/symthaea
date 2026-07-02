// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prism — consciousness-aware pure Rust browser.
//!
//! Features:
//! 1. Live navigation (type URL → fetch → render)
//! 2. Clickable links (click <a> → navigate)
//! 3. Epistemic search (type query → HDC search → results)
//! 4. Back/forward history
//! 5. Security chrome (zone badge, safety dot, threat count)
//! 6. Reader mode (centered content, comfortable typography)

mod renderer;

use std::sync::Arc;

use anyhow::Result;
use vello::util::RenderSurface;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::Window;

use prism_common::{ContentZone, SafetyLevel};
use prism_layout::{LayoutConfig, PaintContent, PaintRect};
use prism_privacy::ConsentStore;
use prism_reflex::ReflexArc;
use prism_search::SearchEngine;
use renderer::{PrismRenderer, SecurityState};

const WELCOME_HTML: &str = include_str!("welcome.html");
const LAYOUT_CONFIG: LayoutConfig = LayoutConfig {
    max_content_width: 680.0,
    base_font_size: 16.0,
    line_height: 1.6,
    char_width_ratio: 0.52,
    reader_mode: true,
};

// ═══════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════

struct TextField {
    content: String,
    cursor: usize,
    focused: bool,
}

impl TextField {
    fn new(text: &str) -> Self {
        Self {
            content: text.to_string(),
            cursor: text.len(),
            focused: false,
        }
    }
    fn insert(&mut self, ch: &str) {
        self.content.insert_str(self.cursor, ch);
        self.cursor += ch.len();
    }
    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.content.remove(self.cursor);
        }
    }
}

#[derive(Clone)]
struct HistoryEntry {
    url: String,
    html: String,
    scroll_y: f32,
}

struct ActiveState {
    surface: Box<RenderSurface<'static>>,
    window: Arc<Window>,
    paint_rects: Vec<PaintRect>,
    security: SecurityState,
    scroll_y: f32,
    address_bar: TextField,
    history: Vec<HistoryEntry>,
    history_idx: usize,
    current_html: String,
    current_url: String,
    mouse_pos: (f32, f32),
}

struct PrismApp {
    renderer: PrismRenderer,
    reflex: ReflexArc,
    search: SearchEngine,
    tokio_rt: tokio::runtime::Runtime,
    state: Option<ActiveState>,
}

impl PrismApp {
    fn new() -> Self {
        Self {
            renderer: PrismRenderer::new(),
            reflex: ReflexArc::new(),
            search: SearchEngine::with_seed_claims(),
            tokio_rt: tokio::runtime::Runtime::new().expect("tokio"),
            state: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// NAVIGATION
// ═══════════════════════════════════════════════════════════════════════

fn is_url(input: &str) -> bool {
    input.starts_with("http://")
        || input.starts_with("https://")
        || input.starts_with("prism://")
        || input.contains("://")
}

fn navigate(
    reflex: &ReflexArc,
    search: &SearchEngine,
    rt: &tokio::runtime::Runtime,
    input: &str,
    vw: f32,
    vh: f32,
) -> (String, String, Vec<PaintRect>, SecurityState) {
    if is_url(input) {
        // URL navigation
        navigate_url(reflex, rt, input, vw, vh)
    } else {
        // Epistemic search
        navigate_search(search, input, vw, vh)
    }
}

fn navigate_url(
    reflex: &ReflexArc,
    rt: &tokio::runtime::Runtime,
    url_str: &str,
    vw: f32,
    vh: f32,
) -> (String, String, Vec<PaintRect>, SecurityState) {
    // Handle prism:// internal URLs
    if url_str == "prism://welcome" || url_str.starts_with("prism://") {
        return load_html(reflex, WELCOME_HTML, url_str, vw, vh);
    }

    // Fetch live URL
    let client = prism_net::PrismClient::new();
    match rt.block_on(client.fetch(url_str)) {
        Ok(page) => {
            let html = page.html.clone();
            let (rects, security) =
                process_page(reflex, &page.dom, url_str, &page.metadata, vw, vh);
            (url_str.to_string(), html, rects, security)
        }
        Err(e) => {
            let error_html = format!(
                r#"<html><head><title>Error</title></head><body>
                <h1>Navigation Error</h1>
                <p>Could not load <strong>{}</strong></p>
                <p>{}</p>
                <p>Check the URL and try again.</p>
                </body></html>"#,
                url_str, e
            );
            load_html(reflex, &error_html, url_str, vw, vh)
        }
    }
}

fn navigate_search(
    search: &SearchEngine,
    query: &str,
    vw: f32,
    vh: f32,
) -> (String, String, Vec<PaintRect>, SecurityState) {
    let results = search.search(query, 10);
    let html = prism_search::render_search_results_html(query, &results);
    let url = format!("prism://search?q={}", query);
    let dom = prism_dom::parse_html(&html);
    let rects = prism_layout::layout_dom_with_config(&dom, vw, vh, &LAYOUT_CONFIG);

    let security = SecurityState {
        zone: ContentZone::Local,
        safety_level: SafetyLevel::Green,
        threat_count: 0,
        url: url.clone(),
        title: format!("Search: {}", query),
        ..Default::default()
    };

    (url, html, rects, security)
}

fn load_html(
    reflex: &ReflexArc,
    html: &str,
    url_str: &str,
    vw: f32,
    vh: f32,
) -> (String, String, Vec<PaintRect>, SecurityState) {
    let dom = prism_dom::parse_html(html);
    let url =
        url::Url::parse(url_str).unwrap_or_else(|_| url::Url::parse("prism://error").unwrap());
    let title = dom.title().unwrap_or_else(|| "Untitled".to_string());
    let pre = reflex.pre_fetch(&url, false, false);
    let post = reflex.post_parse(&dom, &pre);
    let consent = ConsentStore::new();
    let zone = consent.resolve_zone(post.zone, url.host_str().unwrap_or(""));
    let rects = prism_layout::layout_dom_with_config(&dom, vw, vh, &LAYOUT_CONFIG);

    let security = SecurityState {
        zone,
        safety_level: post.safety_level,
        threat_count: post.threats.len(),
        url: url_str.to_string(),
        title,
        ..Default::default()
    };

    (url_str.to_string(), html.to_string(), rects, security)
}

fn process_page(
    reflex: &ReflexArc,
    dom: &prism_dom::DomTree,
    url_str: &str,
    metadata: &prism_net::FetchMetadata,
    vw: f32,
    vh: f32,
) -> (Vec<PaintRect>, SecurityState) {
    let url = &metadata.url;
    let title = dom.title().unwrap_or_else(|| "Untitled".to_string());
    let pre = reflex.pre_fetch(url, metadata.has_auth_headers, metadata.has_cookies);
    let post = reflex.post_parse(dom, &pre);
    let consent = ConsentStore::new();
    let zone = consent.resolve_zone(post.zone, url.host_str().unwrap_or(""));
    let rects = prism_layout::layout_dom_with_config(dom, vw, vh, &LAYOUT_CONFIG);

    let security = SecurityState {
        zone,
        safety_level: post.safety_level,
        threat_count: post.threats.len(),
        url: url_str.to_string(),
        title,
        ..Default::default()
    };

    (rects, security)
}

/// Hit-test PaintRects for link clicks.
fn hit_test_link(rects: &[PaintRect], mouse_x: f32, mouse_y: f32, scroll_y: f32) -> Option<String> {
    let content_y = mouse_y - renderer::TOTAL_CHROME as f32 + scroll_y;
    for rect in rects {
        if mouse_x >= rect.x
            && mouse_x < rect.x + rect.width
            && content_y >= rect.y
            && content_y < rect.y + rect.height
        {
            if let PaintContent::Text {
                is_link: true,
                href: Some(ref url),
                ..
            } = rect.content
            {
                return Some(url.clone());
            }
        }
    }
    None
}

/// Resolve a potentially relative URL against a base URL.
fn resolve_url(href: &str, current_url: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") || href.starts_with("prism://") {
        return href.to_string();
    }
    if let Ok(base) = url::Url::parse(current_url) {
        if let Ok(resolved) = base.join(href) {
            return resolved.to_string();
        }
    }
    href.to_string()
}

// ═══════════════════════════════════════════════════════════════════════
// WINDOWED MODE
// ═══════════════════════════════════════════════════════════════════════

impl ApplicationHandler for PrismApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(LogicalSize::new(1024, 768))
                        .with_title("Prism"),
                )
                .expect("window"),
        );

        let surface = self
            .renderer
            .create_surface(window.clone())
            .expect("surface");
        let size = window.inner_size();
        let (url, html, rects, security) = navigate(
            &self.reflex,
            &self.search,
            &self.tokio_rt,
            "prism://welcome",
            size.width as f32,
            size.height as f32,
        );

        self.state = Some(ActiveState {
            surface: Box::new(surface),
            window,
            paint_rects: rects,
            security,
            scroll_y: 0.0,
            address_bar: TextField::new(&url),
            history: vec![HistoryEntry {
                url: url.clone(),
                html: html.clone(),
                scroll_y: 0.0,
            }],
            history_idx: 0,
            current_html: html,
            current_url: url,
            mouse_pos: (0.0, 0.0),
        });
    }

    fn suspended(&mut self, _el: &ActiveEventLoop) {
        self.state = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _wid: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = &mut self.state else { return };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.renderer.context.resize_surface(
                        &mut state.surface,
                        size.width,
                        size.height,
                    );
                    let dom = prism_dom::parse_html(&state.current_html);
                    state.paint_rects = prism_layout::layout_dom_with_config(
                        &dom,
                        size.width as f32,
                        size.height as f32,
                        &LAYOUT_CONFIG,
                    );
                    state.window.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                };
                state.scroll_y = (state.scroll_y - dy).max(0.0);
                state.window.request_redraw();
            }

            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_pos = (position.x as f32, position.y as f32);
            }

            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                let (mx, my) = state.mouse_pos;

                // Click in chrome area → focus address bar
                if my < renderer::TOTAL_CHROME as f32 {
                    state.address_bar.focused = true;
                    state.address_bar.cursor = state.address_bar.content.len();
                    state.window.request_redraw();
                    return;
                }

                // Click in content → check for links
                if let Some(href) = hit_test_link(&state.paint_rects, mx, my, state.scroll_y) {
                    let resolved = resolve_url(&href, &state.current_url);
                    state.history[state.history_idx].scroll_y = state.scroll_y;
                    // Navigate
                    let size = state.window.inner_size();
                    let (url, html, rects, sec) = navigate(
                        &self.reflex,
                        &self.search,
                        &self.tokio_rt,
                        &resolved,
                        size.width as f32,
                        size.height as f32,
                    );
                    state.current_url = url.clone();
                    state.current_html = html;
                    state.paint_rects = rects;
                    state.security = sec;
                    state.scroll_y = 0.0;
                    state.address_bar.content = url;
                    state.address_bar.cursor = state.address_bar.content.len();
                    state.address_bar.focused = false;
                    state.history.truncate(state.history_idx + 1);
                    state.history.push(HistoryEntry {
                        url: state.current_url.clone(),
                        html: state.current_html.clone(),
                        scroll_y: 0.0,
                    });
                    state.history_idx = state.history.len() - 1;
                    state.window.request_redraw();
                    return;
                }

                // Click on content → unfocus
                state.address_bar.focused = false;
                state.window.request_redraw();
            }

            WindowEvent::KeyboardInput {
                event: ref key_event,
                is_synthetic: false,
                ..
            } => {
                if !key_event.state.is_pressed() {
                    return;
                }

                // Global shortcuts
                match key_event.logical_key.as_ref() {
                    Key::Named(NamedKey::F6) => {
                        state.address_bar.focused = true;
                        state.address_bar.cursor = state.address_bar.content.len();
                        state.window.request_redraw();
                        return;
                    }
                    Key::Named(NamedKey::BrowserBack) if state.history_idx > 0 => {
                        state.history_idx -= 1;
                        let entry = state.history[state.history_idx].clone();
                        let size = state.window.inner_size();
                        let (url, html, rects, security) = load_html(
                            &self.reflex,
                            &entry.html,
                            &entry.url,
                            size.width as f32,
                            size.height as f32,
                        );
                        state.current_url = url.clone();
                        state.current_html = html;
                        state.paint_rects = rects;
                        state.security = security;
                        state.scroll_y = entry.scroll_y;
                        state.address_bar = TextField::new(&url);
                        state.window.request_redraw();
                        return;
                    }
                    Key::Named(NamedKey::BrowserForward)
                        if state.history_idx + 1 < state.history.len() =>
                    {
                        state.history_idx += 1;
                        let entry = state.history[state.history_idx].clone();
                        let size = state.window.inner_size();
                        let (url, html, rects, security) = load_html(
                            &self.reflex,
                            &entry.html,
                            &entry.url,
                            size.width as f32,
                            size.height as f32,
                        );
                        state.current_url = url.clone();
                        state.current_html = html;
                        state.paint_rects = rects;
                        state.security = security;
                        state.scroll_y = entry.scroll_y;
                        state.address_bar = TextField::new(&url);
                        state.window.request_redraw();
                        return;
                    }
                    _ => {}
                }

                if state.address_bar.focused {
                    match key_event.logical_key.as_ref() {
                        Key::Named(NamedKey::Enter) => {
                            let input = state.address_bar.content.clone();
                            if input.is_empty() {
                                return;
                            }
                            state.history[state.history_idx].scroll_y = state.scroll_y;
                            // Navigate or search
                            let size = state.window.inner_size();
                            let (url, html, rects, sec) = navigate(
                                &self.reflex,
                                &self.search,
                                &self.tokio_rt,
                                &input,
                                size.width as f32,
                                size.height as f32,
                            );
                            state.current_url = url.clone();
                            state.current_html = html;
                            state.paint_rects = rects;
                            state.security = sec;
                            state.scroll_y = 0.0;
                            state.address_bar.content = url;
                            state.address_bar.cursor = state.address_bar.content.len();
                            state.address_bar.focused = false;
                            state.history.truncate(state.history_idx + 1);
                            state.history.push(HistoryEntry {
                                url: state.current_url.clone(),
                                html: state.current_html.clone(),
                                scroll_y: 0.0,
                            });
                            state.history_idx = state.history.len() - 1;
                        }
                        Key::Named(NamedKey::Escape) => {
                            state.address_bar.content = state.current_url.clone();
                            state.address_bar.cursor = state.address_bar.content.len();
                            state.address_bar.focused = false;
                        }
                        Key::Named(NamedKey::Backspace) => state.address_bar.backspace(),
                        Key::Named(NamedKey::ArrowLeft) => {
                            if state.address_bar.cursor > 0 {
                                state.address_bar.cursor -= 1;
                            }
                        }
                        Key::Named(NamedKey::ArrowRight) => {
                            if state.address_bar.cursor < state.address_bar.content.len() {
                                state.address_bar.cursor += 1;
                            }
                        }
                        Key::Named(NamedKey::Home) => state.address_bar.cursor = 0,
                        Key::Named(NamedKey::End) => {
                            state.address_bar.cursor = state.address_bar.content.len()
                        }
                        _ => {
                            if let Some(text) = &key_event.text {
                                if !text.chars().all(|c| c.is_control()) {
                                    state.address_bar.insert(text);
                                }
                            }
                        }
                    }
                    state.window.request_redraw();
                } else {
                    // Content-focused shortcuts
                    match key_event.logical_key.as_ref() {
                        Key::Named(NamedKey::Space) => {
                            state.scroll_y += 400.0;
                            state.window.request_redraw();
                        }
                        _ => {}
                    }
                }
            }

            WindowEvent::Ime(winit::event::Ime::Commit(text)) => {
                if state.address_bar.focused {
                    state.address_bar.insert(&text);
                    state.window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let sec = SecurityState {
                    address_bar_text: if state.address_bar.focused {
                        Some(state.address_bar.content.clone())
                    } else {
                        None
                    },
                    address_bar_focused: state.address_bar.focused,
                    ..state.security.clone()
                };
                if let Err(e) = self.renderer.render(
                    &mut state.surface,
                    &state.paint_rects,
                    state.scroll_y,
                    &sec,
                ) {
                    log::error!("Render: {}", e);
                }
            }
            _ => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TEXT MODE
// ═══════════════════════════════════════════════════════════════════════

fn run_text_mode() {
    let reflex = ReflexArc::new();
    let search = SearchEngine::with_seed_claims();
    let rt = tokio::runtime::Runtime::new().unwrap();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║             SYMTHAEA PRISM v0.2.0 — Text Mode                  ║");
    println!("║    Pure Rust │ Zero C++ │ HDC Search │ Powered by Symthaea     ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Demo 1: Welcome page
    println!("\x1b[36m── Navigate: prism://welcome ──\x1b[0m\n");
    let (_, _, rects, sec) = navigate(&reflex, &search, &rt, "prism://welcome", 720.0, 600.0);
    print_page_chrome(&sec);
    print_content(&rects, 5);

    // Demo 2: Epistemic search
    for query in &[
        "ocean acidification",
        "consciousness",
        "rust programming",
        "quantum physics",
    ] {
        println!("\x1b[36m── Search: {} ──\x1b[0m\n", query);
        let results = search.search(query, 5);
        println!(
            "  {} results from {} claims:\n",
            results.len(),
            search.claim_count()
        );
        for (i, r) in results.iter().enumerate() {
            let e = match r.empirical_level {
                prism_common::EmpiricalLevel::E4 => "\x1b[32mE4\x1b[0m",
                prism_common::EmpiricalLevel::E3 => "\x1b[33mE3\x1b[0m",
                _ => "E?",
            };
            println!(
                "  {}. [{}] ({:.0}%) {}",
                i + 1,
                e,
                r.rank_score() * 100.0,
                r.content
            );
            if let Some(src) = r.sources.first() {
                println!("     Source: {}", src);
            }
        }
        println!();
    }

    // Demo 3: Live fetch (if network available)
    println!("\x1b[36m── Navigate: https://example.com ──\x1b[0m\n");
    let (_, _, rects, sec) = navigate(&reflex, &search, &rt, "https://example.com", 720.0, 600.0);
    print_page_chrome(&sec);
    print_content(&rects, 5);

    // Summary
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  1. Navigation    Type URL or search query → Enter             ║");
    println!("║  2. Links         Click <a> elements to navigate               ║");
    println!(
        "║  3. Search        {} claims indexed with 16,384-bit HDC    ║",
        format!("{:>3}", search.claim_count())
    );
    println!("║  4. Security      Reflex arc + 3-zone privacy + encoding gate  ║");
    println!("║  5. Reader mode   680px centered, 1.6 line height              ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Run with display:  cargo run -p prism-shell                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
}

fn print_page_chrome(sec: &SecurityState) {
    let zone = match sec.zone {
        ContentZone::Private => "\x1b[41;97m PRIVATE \x1b[0m",
        ContentZone::Local => "\x1b[43;30m LOCAL \x1b[0m",
        ContentZone::Public => "\x1b[42;97m PUBLIC \x1b[0m",
    };
    let safety = match sec.safety_level {
        SafetyLevel::Green => "\x1b[32m●\x1b[0m",
        SafetyLevel::Yellow => "\x1b[33m●\x1b[0m",
        SafetyLevel::Orange => "\x1b[38;5;208m●\x1b[0m",
        SafetyLevel::Red => "\x1b[31m●\x1b[0m",
    };
    println!("  {} {} {} {}\n", zone, sec.url, safety, sec.title);
}

fn print_content(rects: &[PaintRect], max_lines: usize) {
    let mut printed = 0;
    for rect in rects {
        if printed >= max_lines {
            println!("  ...");
            break;
        }
        if let PaintContent::Text {
            text,
            bold,
            font_size,
            ..
        } = &rect.content
        {
            if *bold && *font_size >= 28.0 {
                println!("  \x1b[1;97m{}\x1b[0m", text);
            } else if *bold {
                println!("  \x1b[1m{}\x1b[0m", text);
            } else {
                println!("  {}", text);
            }
            printed += 1;
        }
    }
    println!();
}

// ═══════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    env_logger::init();

    match EventLoop::new() {
        Ok(event_loop) => {
            let mut app = PrismApp::new();
            event_loop.run_app(&mut app)?;
        }
        Err(_) => run_text_mode(),
    }

    Ok(())
}

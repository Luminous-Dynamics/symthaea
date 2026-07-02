// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Canvas2D renderer for the kinship web visualization.
//!
//! Renders nodes, edges with breathing, gratitude particles, and
//! the homeostatic void meditative state.

use leptos::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::spawn_local;
use std::cell::RefCell;
use std::rc::Rc;

use super::shared_state::*;
use super::layout;
use crate::hearth_context::use_hearth;
use mycelix_leptos_core::use_homeostasis;
use mycelix_leptos_core::use_thermodynamic;
use crate::themes::{use_theme, HearthTheme};
use hearth_leptos_types::*;

const CANVAS_W: f64 = 800.0;
const CANVAS_H: f64 = 500.0;

/// Unique ID for the canvas element so we can find it in the DOM.
const CANVAS_ID: &str = "hearth-kinship-canvas";

/// Map HearthTheme to canvas colors.
fn theme_to_canvas(theme: HearthTheme) -> CanvasTheme {
    match theme {
        HearthTheme::Ember => CanvasTheme::default(),
        HearthTheme::Tide => CanvasTheme {
            bg: (8.0/255.0, 12.0/255.0, 18.0/255.0),
            primary: (100.0/255.0, 168.0/255.0, 216.0/255.0),
            text: (212.0/255.0, 220.0/255.0, 232.0/255.0),
            glow: (100.0/255.0, 168.0/255.0, 216.0/255.0),
        },
        HearthTheme::Canopy => CanvasTheme {
            bg: (8.0/255.0, 10.0/255.0, 6.0/255.0),
            primary: (140.0/255.0, 196.0/255.0, 100.0/255.0),
            text: (216.0/255.0, 228.0/255.0, 208.0/255.0),
            glow: (140.0/255.0, 196.0/255.0, 100.0/255.0),
        },
        HearthTheme::Dusk => CanvasTheme {
            bg: (12.0/255.0, 8.0/255.0, 14.0/255.0),
            primary: (212.0/255.0, 136.0/255.0, 156.0/255.0),
            text: (228.0/255.0, 216.0/255.0, 232.0/255.0),
            glow: (212.0/255.0, 136.0/255.0, 156.0/255.0),
        },
        HearthTheme::Stone => CanvasTheme {
            bg: (10.0/255.0, 11.0/255.0, 12.0/255.0),
            primary: (136.0/255.0, 180.0/255.0, 212.0/255.0),
            text: (220.0/255.0, 224.0/255.0, 228.0/255.0),
            glow: (136.0/255.0, 180.0/255.0, 212.0/255.0),
        },
        HearthTheme::Meadow => CanvasTheme {
            bg: (240.0/255.0, 238.0/255.0, 232.0/255.0),
            primary: (90.0/255.0, 138.0/255.0, 58.0/255.0),
            text: (42.0/255.0, 36.0/255.0, 24.0/255.0),
            glow: (90.0/255.0, 138.0/255.0, 58.0/255.0),
        },
        HearthTheme::Cosmos => CanvasTheme {
            bg: (4.0/255.0, 5.0/255.0, 10.0/255.0),
            primary: (160.0/255.0, 180.0/255.0, 224.0/255.0),
            text: (200.0/255.0, 204.0/255.0, 224.0/255.0),
            glow: (160.0/255.0, 180.0/255.0, 224.0/255.0),
        },
        HearthTheme::Clay => CanvasTheme {
            bg: (12.0/255.0, 8.0/255.0, 6.0/255.0),
            primary: (196.0/255.0, 120.0/255.0, 80.0/255.0),
            text: (232.0/255.0, 220.0/255.0, 212.0/255.0),
            glow: (196.0/255.0, 120.0/255.0, 80.0/255.0),
        },
        HearthTheme::Frost => CanvasTheme {
            bg: (248.0/255.0, 249.0/255.0, 252.0/255.0),
            primary: (58.0/255.0, 92.0/255.0, 140.0/255.0),
            text: (26.0/255.0, 28.0/255.0, 36.0/255.0),
            glow: (58.0/255.0, 92.0/255.0, 140.0/255.0),
        },
        HearthTheme::Circuit => CanvasTheme {
            bg: (6.0/255.0, 8.0/255.0, 10.0/255.0),
            primary: (64.0/255.0, 200.0/255.0, 176.0/255.0),
            text: (192.0/255.0, 212.0/255.0, 224.0/255.0),
            glow: (64.0/255.0, 200.0/255.0, 176.0/255.0),
        },
    }
}

/// Bond health -> RGB color (green -> amber -> red).
fn bond_color(strength_bp: u32) -> (f64, f64, f64) {
    let t = strength_bp as f64 / BOND_MAX as f64;
    if t >= 0.7 {
        (0.29, 0.87, 0.50)
    } else if t >= 0.4 {
        let blend = (t - 0.4) / 0.3;
        (0.96 - blend * 0.67, 0.62 + blend * 0.25, 0.04 + blend * 0.46)
    } else {
        (0.94, 0.27, 0.27)
    }
}

/// Get the 2D context from our canvas in the DOM.
fn get_canvas_ctx() -> Option<web_sys::CanvasRenderingContext2d> {
    let document = web_sys::window()?.document()?;
    let el = document.get_element_by_id(CANVAS_ID)?;
    let canvas: web_sys::HtmlCanvasElement = el.dyn_into().ok()?;
    canvas.get_context("2d").ok()??.dyn_into().ok()
}

/// The Leptos component that owns the canvas and animation loop.
#[component]
pub fn KinshipCanvas() -> impl IntoView {
    let hearth = use_hearth();
    let homeostasis = use_homeostasis();
    let thermo = use_thermodynamic();
    let theme_state = use_theme();

    let viz_state = new_shared_viz_state(CANVAS_W, CANVAS_H);

    // Sync hearth data into viz state reactively
    let sync_state = viz_state.clone();
    Effect::new(move |_| {
        let members = hearth.members.get();
        let bonds = hearth.bonds.get();
        let presence = hearth.presence.get();
        let gratitude = hearth.gratitude.get();
        let is_home = homeostasis.in_homeostasis.get();
        let torpor = thermo.torpor_level.get();

        let current_theme = theme_state.current.get();

        let mut state = sync_state.borrow_mut();
        state.homeostasis = is_home;
        state.torpor = torpor;
        state.theme = theme_to_canvas(current_theme);

        // Sync nodes (preserve positions if already placed)
        let existing: std::collections::HashMap<String, (f64, f64)> = state.nodes.iter()
            .map(|n| (n.id.clone(), (n.x, n.y)))
            .collect();

        state.nodes = members.iter().enumerate().map(|(i, m)| {
            let (x, y) = existing.get(&m.agent).cloned().unwrap_or_else(|| {
                let angle = (i as f64 / members.len().max(1) as f64) * std::f64::consts::TAU;
                let r = 100.0;
                (CANVAS_W / 2.0 + angle.cos() * r, CANVAS_H / 2.0 + angle.sin() * r)
            });
            let home = presence.iter()
                .find(|p| p.agent == m.agent)
                .map(|p| p.status == PresenceStatusType::Home)
                .unwrap_or(false);
            let color = if m.role.is_guardian() {
                [0.98, 0.75, 0.15]
            } else if m.role.is_minor() {
                [0.65, 0.55, 0.98]
            } else {
                [0.83, 0.65, 0.45]
            };
            VisNode {
                id: m.agent.clone(),
                label: m.display_name.clone(),
                x, y, vx: 0.0, vy: 0.0,
                radius: if m.role.is_guardian() { 22.0 } else { 16.0 },
                color,
                is_guardian: m.role.is_guardian(),
                presence_home: home,
            }
        }).collect();

        // Sync edges (preserve breath phase)
        let old_edges = state.edges.clone();
        state.edges = bonds.iter().map(|b| {
            let phase = old_edges.iter()
                .find(|e| e.id == b.hash)
                .map(|e| e.breath_phase)
                .unwrap_or(0.0);
            VisEdge {
                id: b.hash.clone(),
                source: b.member_a.clone(),
                target: b.member_b.clone(),
                strength_bp: b.strength_bp,
                bond_type_label: b.bond_type.label().to_string(),
                breath_phase: phase,
            }
        }).collect();

        // Spawn particles for most recent gratitude
        if let Some(g) = gratitude.first() {
            let already = state.particles.iter()
                .any(|p| p.from == g.from_agent && p.to == g.to_agent);
            if !already {
                state.particles.push(VisParticle {
                    from: g.from_agent.clone(),
                    to: g.to_agent.clone(),
                    t: 0.0,
                    ttl: 3.0,
                });
            }
        }

        // Consume pending tend (set by canvas click)
        if let Some(edge_id) = state.pending_tend.take() {
            drop(state); // release borrow before calling action
            crate::hearth_actions::tend_bond(edge_id.clone());
            let mut state = sync_state.borrow_mut();
            state.flash_edge = Some(edge_id);
            state.flash_ttl = 0.6;
        }
    });

    // Start animation loop + click handler after DOM mount
    let anim_state = viz_state.clone();
    let click_state = viz_state.clone();
    spawn_local(async move {
        gloo_timers::future::TimeoutFuture::new(50).await;
        start_animation_loop(anim_state);
        install_click_handler(click_state);
    });

    view! {
        <canvas
            id=CANVAS_ID
            width=CANVAS_W as u32
            height=CANVAS_H as u32
            class="kinship-canvas"
            role="img"
            aria-label="kinship web — family members as nodes connected by bonds of varying strength"
        />
    }
}

/// Start the requestAnimationFrame loop. Acquires the canvas from DOM each call.
fn start_animation_loop(state: SharedVizState) {
    let frame_fn: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let frame_fn_clone = frame_fn.clone();
    let last_time = Rc::new(RefCell::new(0.0_f64));

    *frame_fn.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let window = match web_sys::window() {
            Some(w) => w,
            None => return,
        };

        // Timing
        let now = window.performance().map(|p| p.now() / 1000.0).unwrap_or(0.0);
        let mut lt = last_time.borrow_mut();
        let dt = if *lt == 0.0 { 1.0 / 60.0 } else { (now - *lt).min(0.05) };
        *lt = now;
        drop(lt);

        // Physics
        {
            let mut s = state.borrow_mut();
            if !s.homeostasis {
                layout::step_layout(&mut s, dt);
            }
            layout::step_breathing(&mut s, dt);
            layout::step_particles(&mut s, dt);
            // Tick flash timer
            if s.flash_ttl > 0.0 {
                s.flash_ttl -= dt;
                if s.flash_ttl <= 0.0 {
                    s.flash_edge = None;
                }
            }
        }

        // Render (acquire canvas from DOM, not from stale ref)
        if let Some(ctx) = get_canvas_ctx() {
            let s = state.borrow();
            render_frame(&ctx, &s);
        }

        // Next frame
        if let Some(ref cb) = *frame_fn_clone.borrow() {
            let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }) as Box<dyn FnMut()>));

    if let Some(window) = web_sys::window() {
        if let Some(ref cb) = *frame_fn.borrow() {
            let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
        }
    }

    // Leak to keep alive
    std::mem::forget(frame_fn);
}

/// Install click handler on the canvas for bond tending.
fn install_click_handler(state: SharedVizState) {
    let document = match web_sys::window().and_then(|w| w.document()) {
        Some(d) => d,
        None => return,
    };
    let canvas = match document.get_element_by_id(CANVAS_ID) {
        Some(c) => c,
        None => return,
    };

    let closure = Closure::wrap(Box::new(move |event: web_sys::MouseEvent| {
        let target = event.target().unwrap();
        let canvas_el: &web_sys::HtmlCanvasElement = target.unchecked_ref();
        let rect = canvas_el.get_bounding_client_rect();

        // Scale click coords to canvas logical coords
        let scale_x = CANVAS_W / rect.width();
        let scale_y = CANVAS_H / rect.height();
        let cx = (event.client_x() as f64 - rect.left()) * scale_x;
        let cy = (event.client_y() as f64 - rect.top()) * scale_y;

        let mut s = state.borrow_mut();
        if let Some(edge_id) = s.edge_at_point(cx, cy, 20.0) {
            s.pending_tend = Some(edge_id);
        }
    }) as Box<dyn FnMut(web_sys::MouseEvent)>);

    let _ = canvas.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref());
    std::mem::forget(closure);
}

fn render_frame(ctx: &web_sys::CanvasRenderingContext2d, state: &VisualizationState) {
    let w = state.canvas_width;
    let h = state.canvas_height;

    // Clear with theme background
    let (br, bg, bb) = state.theme.bg;
    ctx.set_fill_style_str(&format!("rgb({},{},{})", (br*255.0) as u32, (bg*255.0) as u32, (bb*255.0) as u32));
    ctx.fill_rect(0.0, 0.0, w, h);

    if state.homeostasis {
        render_void(ctx, state);
        return;
    }

    if state.nodes.is_empty() {
        let (tr, tg, tb) = state.theme.text;
        ctx.set_fill_style_str(&format!("rgba({},{},{},0.3)", (tr*255.0) as u32, (tg*255.0) as u32, (tb*255.0) as u32));
        ctx.set_font("14px Inter, sans-serif");
        ctx.set_text_align("center");
        ctx.set_text_baseline("middle");
        let _ = ctx.fill_text("the web is forming…", w / 2.0, h / 2.0);
        return;
    }

    let torpor_alpha = 1.0 - state.torpor * 0.6;

    // Draw edges (bonds) with breathing + flash on tend
    for edge in &state.edges {
        let src = state.node_pos(&edge.source);
        let tgt = state.node_pos(&edge.target);
        if let (Some((x1, y1)), Some((x2, y2))) = (src, tgt) {
            let is_flashing = state.flash_edge.as_ref() == Some(&edge.id) && state.flash_ttl > 0.0;
            let (r, g, b) = if is_flashing {
                // Flash bright golden
                (1.0, 0.85, 0.3)
            } else {
                bond_color(edge.strength_bp)
            };
            let health = edge.strength_bp as f64 / BOND_MAX as f64;
            let breath = (edge.breath_phase.sin() * 0.5 + 0.5) * 0.3 + 0.7;
            let alpha = if is_flashing {
                (state.flash_ttl / 0.6).min(1.0) * 0.8 + 0.2
            } else {
                health * breath * torpor_alpha
            };
            let line_width = if is_flashing {
                4.0 + (state.flash_ttl / 0.6) * 4.0
            } else {
                1.0 + health * 3.0
            };

            ctx.set_stroke_style_str(&format!("rgba({},{},{},{:.2})",
                (r * 255.0) as u32, (g * 255.0) as u32, (b * 255.0) as u32, alpha));
            ctx.set_line_width(line_width * breath);
            ctx.begin_path();
            ctx.move_to(x1, y1);
            ctx.line_to(x2, y2);
            ctx.stroke();
        }
    }

    // Draw gratitude particles
    for particle in &state.particles {
        let src = state.node_pos(&particle.from);
        let tgt = state.node_pos(&particle.to);
        if let (Some((x1, y1)), Some((x2, y2))) = (src, tgt) {
            let px = x1 + (x2 - x1) * particle.t;
            let py = y1 + (y2 - y1) * particle.t;
            let alpha = (particle.ttl / 3.0).min(1.0) * torpor_alpha;
            let radius = 4.0 + (1.0 - particle.t) * 3.0;

            ctx.set_fill_style_str(&format!("rgba(251,191,36,{:.2})", alpha));
            ctx.begin_path();
            let _ = ctx.arc(px, py, radius, 0.0, std::f64::consts::TAU);
            ctx.fill();

            ctx.set_fill_style_str(&format!("rgba(251,191,36,{:.2})", alpha * 0.3));
            ctx.begin_path();
            let _ = ctx.arc(px, py, radius * 2.5, 0.0, std::f64::consts::TAU);
            ctx.fill();
        }
    }

    // Draw nodes
    for node in &state.nodes {
        let alpha = torpor_alpha;
        let [r, g, b] = node.color;

        // Presence glow ring
        if node.presence_home {
            ctx.set_stroke_style_str(&format!("rgba(74,222,128,{:.2})", alpha * 0.4));
            ctx.set_line_width(2.0);
            ctx.begin_path();
            let _ = ctx.arc(node.x, node.y, node.radius + 6.0, 0.0, std::f64::consts::TAU);
            ctx.stroke();
        }

        // Node filled circle
        ctx.set_fill_style_str(&format!("rgba({},{},{},{:.2})",
            (r * 255.0) as u32, (g * 255.0) as u32, (b * 255.0) as u32, alpha));
        ctx.begin_path();
        let _ = ctx.arc(node.x, node.y, node.radius, 0.0, std::f64::consts::TAU);
        ctx.fill();

        // Initial letter inside the node
        ctx.set_fill_style_str(&format!("rgba(10,10,8,{:.2})", alpha * 0.9));
        ctx.set_font(&format!("{}px Inter, sans-serif", (node.radius * 0.9) as u32));
        ctx.set_text_align("center");
        ctx.set_text_baseline("middle");
        let initial = node.label.chars().next().unwrap_or('?').to_string();
        let _ = ctx.fill_text(&initial, node.x, node.y);

        // Name label below (theme text color)
        let (tr, tg, tb) = state.theme.text;
        ctx.set_fill_style_str(&format!("rgba({},{},{},{:.2})", (tr*255.0) as u32, (tg*255.0) as u32, (tb*255.0) as u32, alpha * 0.8));
        ctx.set_font("11px Inter, sans-serif");
        ctx.set_text_baseline("top");
        let _ = ctx.fill_text(&node.label, node.x, node.y + node.radius + 6.0);
    }
}

fn render_void(ctx: &web_sys::CanvasRenderingContext2d, state: &VisualizationState) {
    let w = state.canvas_width;
    let h = state.canvas_height;
    let cx = w / 2.0;
    let cy = h / 2.0;
    let (pr, pg, pb) = state.theme.primary;
    let (tr, tg, tb) = state.theme.text;

    let now = web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0);
    let breath = (now * 0.5).sin() * 0.15 + 0.25;

    // Inner glow (theme primary)
    ctx.set_fill_style_str(&format!("rgba({},{},{},{:.3})",
        (pr*255.0) as u32, (pg*255.0) as u32, (pb*255.0) as u32, breath));
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, 50.0, 0.0, std::f64::consts::TAU);
    ctx.fill();

    // Outer glow
    ctx.set_fill_style_str(&format!("rgba({},{},{},{:.3})",
        (pr*255.0) as u32, (pg*255.0) as u32, (pb*255.0) as u32, breath * 0.25));
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, 110.0, 0.0, std::f64::consts::TAU);
    ctx.fill();

    // Text
    ctx.set_fill_style_str(&format!("rgba({},{},{},{:.2})",
        (tr*255.0) as u32, (tg*255.0) as u32, (tb*255.0) as u32, breath + 0.2));
    ctx.set_font("14px Inter, sans-serif");
    ctx.set_text_align("center");
    ctx.set_text_baseline("middle");
    let _ = ctx.fill_text("all is well.", cx, cy + 140.0);
}

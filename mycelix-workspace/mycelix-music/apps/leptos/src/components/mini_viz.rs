// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Miniature waveform visualization for the player bar.
//! Shows a breathing, consciousness-colored waveform when music is playing.

use leptos::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::app::PlayerState;

const MINI_W: u32 = 120;
const MINI_H: u32 = 40;

/// A small animated waveform that lives in the player bar.
#[component]
pub fn MiniViz() -> impl IntoView {
    let player = expect_context::<PlayerState>();
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();

    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else { return };
        let canvas: HtmlCanvasElement = canvas_el.into();
        canvas.set_width(MINI_W);
        canvas.set_height(MINI_H);

        let ctx = match canvas.get_context("2d") {
            Ok(Some(c)) => match c.dyn_into::<CanvasRenderingContext2d>() {
                Ok(ctx) => ctx,
                Err(_) => return,
            },
            _ => return,
        };

        let frame_ref: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
        let frame_clone = frame_ref.clone();

        let closure = Closure::wrap(Box::new(move || {
            let time = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or(0.0);

            let playing = player.is_playing.get_untracked();
            let progress_frac = {
                let d = player.duration.get_untracked();
                if d > 0.0 { (player.progress.get_untracked() / d).min(1.0) } else { 0.0 }
            };

            let w = MINI_W as f64;
            let h = MINI_H as f64;

            // Clear
            ctx.clear_rect(0.0, 0.0, w, h);

            if playing {
                // Animated waveform
                let mid = h / 2.0;
                let amplitude = 0.3 + (time * 0.5).sin().abs() * 0.5; // breathing

                // Waveform (emotion-colored via CSS variable read)
                ctx.set_stroke_style_str("rgba(139, 92, 246, 0.8)");
                ctx.set_line_width(1.5);
                ctx.begin_path();

                let bars = 32;
                for i in 0..bars {
                    let t = i as f64 / bars as f64;
                    let x = t * w;

                    // Composite wave from multiple frequencies
                    let wave = (t * 8.0 + time * 2.0).sin() * 0.6
                        + (t * 13.0 + time * 3.0).sin() * 0.3
                        + (t * 21.0 + time * 1.5).sin() * 0.1;

                    let y = mid + wave * amplitude * (h * 0.4);

                    if i == 0 { ctx.move_to(x, y); } else { ctx.line_to(x, y); }
                }
                ctx.stroke();

                // Progress indicator (thin line at current position)
                let px = progress_frac * w;
                ctx.set_stroke_style_str("rgba(255, 255, 255, 0.4)");
                ctx.set_line_width(1.0);
                ctx.begin_path();
                ctx.move_to(px, 2.0);
                ctx.line_to(px, h - 2.0);
                ctx.stroke();
            } else {
                // Idle: faint flatline
                ctx.set_stroke_style_str("rgba(136, 136, 160, 0.3)");
                ctx.set_line_width(1.0);
                ctx.begin_path();
                ctx.move_to(0.0, h / 2.0);
                ctx.line_to(w, h / 2.0);
                ctx.stroke();
            }

            if let Some(window) = web_sys::window() {
                if let Some(ref cb) = *frame_clone.borrow() {
                    let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
                }
            }
        }) as Box<dyn FnMut()>);

        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
        }
        *frame_ref.borrow_mut() = Some(closure);
    });

    view! {
        <canvas node_ref=canvas_ref class="mini-viz" width="120" height="40" />
    }
}

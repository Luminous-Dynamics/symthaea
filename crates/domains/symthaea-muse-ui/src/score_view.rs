// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research Mode's Score view: a piano roll of the shared current piece's
//! performed notes, with a playhead synced to the shared `<audio>`
//! element and click-to-seek. Ported from `pianoRoll()` in
//! `studio/index.html`, with one deliberate simplification: the legacy
//! page draws the static notes once into a cached `ImageData` and only
//! repaints the playhead line each frame; this redraws everything
//! (notes + playhead) every frame instead. A typical piece has a few
//! dozen notes, so the extra draw calls are cheap, and it avoids a whole
//! class of `ImageData`/devicePixelRatio bugs that would be hard to get
//! right without a browser to check them against.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlAudioElement, HtmlCanvasElement};

use crate::api::{self, PerformedVoice};
use crate::state::MuseState;

// pub(crate): shared with orchestration_view.rs so the same voice keeps
// the same color across Score's piano roll and the Orchestration view.
pub(crate) fn voice_color(name: &str) -> &'static str {
    match name {
        "Melody" => "#d9a05b",
        "Doubling" => "rgba(217,160,91,0.45)",
        "Harmony" => "#7a6d5c",
        "Bass" => "#7ba05b",
        "Counter" => "#6f9bbf",
        _ => "#999",
    }
}

fn midi_of(frequency: f64) -> f64 {
    69.0 + 12.0 * (frequency / 440.0).log2()
}

#[component]
pub fn ScoreView(muse: MuseState) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let voices = RwSignal::new(Vec::<PerformedVoice>::new());
    let load_status = RwSignal::new(String::new());

    // Refetch notes whenever the shared current piece changes.
    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            voices.set(Vec::new());
            return;
        };
        let id = c.id;
        load_status.set("loading score…".to_string());
        spawn_local(async move {
            match api::fetch_notes(api::DEFAULT_BACKEND, id).await {
                Ok(v) => {
                    voices.set(v);
                    load_status.set(String::new());
                }
                Err(e) => load_status.set(format!("score unavailable: {e}")),
            }
        });
    });

    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        start_score_loop(canvas_el.into(), voices, muse.audio_ref);
    });

    view! {
        <div class="score-view">
            <canvas
                node_ref=canvas_ref
                class="score-canvas"
                title="click to seek"
                on:click=move |ev: leptos::ev::MouseEvent| {
                    let Some(canvas_el) = canvas_ref.get_untracked() else { return };
                    let canvas: HtmlCanvasElement = canvas_el.into();
                    let rect = canvas.get_bounding_client_rect();
                    let frac = if rect.width() > 0.0 {
                        ((ev.client_x() as f64 - rect.left()) / rect.width()).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    if let Some(audio) = muse.audio_ref.get_untracked() {
                        let audio: HtmlAudioElement = audio.into();
                        let dur = audio.duration();
                        if dur.is_finite() && dur > 0.0 {
                            audio.set_current_time(frac * dur);
                        } else {
                            let _ = audio.play();
                        }
                    }
                }
            />
            <p class="legend">
                {move || voices.get().into_iter().map(|v| {
                    let color = voice_color(&v.name);
                    view! {
                        <span class="legend-item">
                            <b style=format!("color:{color}")>"■"</b>
                            " " {v.name} " · " {v.instrument} "  "
                        </span>
                    }
                }).collect_view()}
            </p>
            <p class="status-line">{move || load_status.get()}</p>
        </div>
    }
}

/// Same `requestAnimationFrame` + `on_cleanup`/`SendWrapper` pattern as
/// `pages.rs::start_visualizer` — see that function's doc for why the
/// `SendWrapper` is there. Kept as a separate copy rather than shared
/// with the Listen visualizer since the two draw entirely different
/// things and factoring out just the rAF plumbing would leave a thin,
/// harder-to-follow abstraction for ~15 lines of boilerplate.
fn start_score_loop(
    canvas: HtmlCanvasElement,
    voices: RwSignal<Vec<PerformedVoice>>,
    audio_ref: NodeRef<leptos::html::Audio>,
) {
    let ctx = match canvas.get_context("2d") {
        Ok(Some(ctx)) => match ctx.dyn_into::<CanvasRenderingContext2d>() {
            Ok(ctx) => ctx,
            Err(_) => return,
        },
        _ => return,
    };

    let frame_closure: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let frame_closure_clone = frame_closure.clone();
    let last_frame_id: Rc<Cell<i32>> = Rc::new(Cell::new(0));
    let last_frame_id_clone = last_frame_id.clone();

    let closure = Closure::wrap(Box::new(move || {
        let w = canvas.client_width().max(1) as u32;
        let h = canvas.client_height().max(1) as u32;
        // `set_width`/`set_height` clear the canvas's backing store even
        // when the value doesn't change — only touch them on an actual
        // resize, since every frame redraws unconditionally below anyway.
        if canvas.width() != w {
            canvas.set_width(w);
        }
        if canvas.height() != h {
            canvas.set_height(h);
        }

        let vs = voices.get_untracked();
        let audio_el = audio_ref
            .get_untracked()
            .map(|a| -> HtmlAudioElement { a.into() });
        draw_score_frame(&ctx, w as f64, h as f64, &vs, audio_el.as_ref());

        if let Some(window) = web_sys::window() {
            if let Some(ref cb) = *frame_closure_clone.borrow() {
                if let Ok(id) = window.request_animation_frame(cb.as_ref().unchecked_ref()) {
                    last_frame_id_clone.set(id);
                }
            }
        }
    }) as Box<dyn FnMut()>);

    if let Some(window) = web_sys::window() {
        if let Ok(id) = window.request_animation_frame(closure.as_ref().unchecked_ref()) {
            last_frame_id.set(id);
        }
    }
    *frame_closure.borrow_mut() = Some(closure);

    let cleanup = send_wrapper::SendWrapper::new(move || {
        if let Some(window) = web_sys::window() {
            let _ = window.cancel_animation_frame(last_frame_id.get());
        }
        *frame_closure.borrow_mut() = None;
    });
    on_cleanup(move || cleanup.take()());
}

fn draw_score_frame(
    ctx: &CanvasRenderingContext2d,
    w: f64,
    h: f64,
    voices: &[PerformedVoice],
    audio: Option<&HtmlAudioElement>,
) {
    ctx.clear_rect(0.0, 0.0, w, h);

    let mut t_max = 0.0_f64;
    let mut m_lo = 127.0_f64;
    let mut m_hi = 0.0_f64;
    for v in voices {
        for n in &v.notes {
            t_max = t_max.max(n.start_time + n.duration);
            let m = midi_of(n.frequency);
            m_lo = m_lo.min(m);
            m_hi = m_hi.max(m);
        }
    }
    if t_max <= 0.0 {
        return;
    }
    m_lo -= 2.0;
    m_hi += 2.0;
    let x = |t: f64| t / t_max * w;
    let y = |m: f64| h - (m - m_lo) / (m_hi - m_lo) * h;
    let note_h = (h / (m_hi - m_lo) * 0.85).max(2.0);

    for v in voices {
        ctx.set_fill_style_str(voice_color(&v.name));
        for n in &v.notes {
            ctx.set_global_alpha(0.35 + 0.6 * n.velocity.min(1.0));
            let _ = ctx.fill_rect(
                x(n.start_time),
                y(midi_of(n.frequency)) - note_h / 2.0,
                x(n.duration).max(1.5),
                note_h,
            );
        }
    }
    ctx.set_global_alpha(1.0);

    if let Some(audio) = audio {
        let dur = audio.duration();
        if dur.is_finite() && dur > 0.0 {
            let px = audio.current_time() / dur * w;
            ctx.set_fill_style_str("rgba(232,223,210,0.85)");
            let _ = ctx.fill_rect(px, 0.0, 1.5, h);
        }
    }
}

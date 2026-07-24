// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The persistent bottom playback bar — visible across every route
//! (Listen/Create/Research/Liked/Atlas), matching the design mockups'
//! layout: piece identity, restart/play-pause/next transport, volume, and
//! a meter/duration readout.
//!
//! Everything shown here is real data already on `Candidate` (`style`,
//! `meter`, `duration_secs`) or the shared playback reducer's state
//! (`MuseState::playback`, `volume`) — deliberately does NOT show BPM or
//! key, since
//! neither is fetched anywhere client-side today (that data lives behind
//! Research's Harmony/Overview endpoints, which aren't wired to the
//! global player); adding a per-piece fetch just to fill this bar would be
//! new scope beyond a visual pass, not reuse of existing data.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlAudioElement, HtmlCanvasElement, HtmlInputElement};

use crate::audio_reactivity;
use crate::icons::{
    BarsIcon, HeartIcon, NextIcon, PauseIcon, PlayIcon, RestartIcon, VolumeIcon, WavesIcon,
};
use crate::palette::{self, Palette};
use crate::playback::PlaybackPhase;
use crate::state::{MuseState, TimelineMode, VizMode};

/// `mm:ss`, matching the format the legacy studio page and the design
/// spec's timeline both use for playback position.
fn format_secs(s: f64) -> String {
    let s = s.max(0.0) as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

#[component]
pub fn PlayerBar(muse: MuseState) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let mini_viz_ref = NodeRef::<leptos::html::Canvas>::new();

    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        start_progress_wave(canvas_el.into(), muse);
    });

    Effect::new(move |_| {
        let Some(canvas_el) = mini_viz_ref.get() else {
            return;
        };
        start_mini_viz(canvas_el.into(), muse.viz_mode, muse.current_style);
    });

    view! {
        <footer class="player-bar" class:visible=move || muse.current.get().is_some()>
            <div class="player-viz-corner">
                <canvas node_ref=mini_viz_ref class="player-mini-viz" />
            </div>
            <div class="player-progress">
                // Timeline-display selector: Bar vs Wave — deliberately
                // independent of the Listen hero's own mode selector (now
                // in `pages.rs`'s canvas overlay). This one controls only
                // how THIS timeline strip renders, not the hero map; the
                // two used to be conflated under one shared toggle sitting
                // confusingly next to the timeline while actually
                // controlling the hero canvas instead.
                <div class="timeline-mode-icons">
                    {TimelineMode::ALL
                        .into_iter()
                        .map(|m| {
                            let icon = match m {
                                TimelineMode::Bar => view! { <BarsIcon /> }.into_any(),
                                TimelineMode::Wave => view! { <WavesIcon /> }.into_any(),
                            };
                            let label = m.label();
                            view! {
                                <button
                                    type="button"
                                    class="icon-btn timeline-mode-btn"
                                    class:sel=move || muse.timeline_mode.get() == m
                                    title=label
                                    aria-label=label
                                    on:click=move |_| muse.timeline_mode.set(m)
                                >
                                    {icon}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>
                // The real seek control: a plain range input, kept fully
                // keyboard/screen-reader operable (arrow keys, Home/End,
                // announced value) exactly as before — it's visually
                // transparent, with the decorative wave canvas layered on
                // top purely for looks (`pointer-events: none`, so clicks
                // and drags pass straight through to this input). Keeping
                // scrubbing on the native control avoids reimplementing
                // pointer capture, touch, and focus semantics a hand-rolled
                // canvas drag handler would otherwise have to get right.
                <div class="player-wave-track">
                    <canvas node_ref=canvas_ref class="player-wave" />
                    <input
                        type="range"
                        class="player-range"
                        min="0"
                        max=move || {
                            muse.playback.get().duration_seconds.unwrap_or(0.001).max(0.001).to_string()
                        }
                        step="0.1"
                        aria-label="Seek"
                        prop:value=move || muse.playback.get().position_seconds.to_string()
                        on:input=move |ev| {
                            let target: HtmlInputElement = event_target(&ev);
                            if let Ok(v) = target.value().parse::<f64>() {
                                muse.seek(v);
                            }
                        }
                    />
                </div>
                <span
                    class="player-time"
                    style=move || {
                        let p = palette::palette_for(&muse.current_style.get());
                        format!("--time-a: rgb({}); --time-b: rgb({});", p.a, p.b)
                    }
                >
                    <span class="player-time-elapsed">
                        {move || format_secs(muse.playback.get().position_seconds)}
                    </span>
                    <span class="player-time-sep">"/"</span>
                    <span class="player-time-total">
                        {move || format_secs(muse.playback.get().duration_seconds.unwrap_or(0.0))}
                    </span>
                </span>
            </div>
            {move || {
                muse.current
                    .get()
                    .map(|c| {
                        let p = palette::palette_for(&c.style);
                        let dot_style = format!(
                            "background: rgb({}); box-shadow: 0 0 10px rgba({},0.6);",
                            p.a,
                            p.a,
                        );
                        // Tints the play button's slow "lava lamp" glow to
                        // this piece's own palette rather than a fixed
                        // color pair — consistent with every other
                        // palette-driven surface (dot, waves, radial).
                        let play_btn_style =
                            format!("--lava-a: rgb({}); --lava-b: rgb({});", p.a, p.b);
                        view! {
                            <div class="player-identity">
                                <span class="player-dot" style=dot_style></span>
                                <div class="player-text">
                                    <span class="player-title">{c.title.clone()}</span>
                                    <span class="player-sub">
                                        {format!("{} · {}/4 · {:.0}s", c.style, c.meter, c.duration_secs)}
                                    </span>
                                </div>
                            </div>

                            <div class="player-transport">
                                <button
                                    type="button"
                                    class="icon-btn"
                                    title="Restart"
                                    on:click=move |_| muse.restart()
                                >
                                    <RestartIcon />
                                </button>
                                <button
                                    type="button"
                                    class="icon-btn play-btn"
                                    style=play_btn_style
                                    title=move || {
                                        if muse.playback.get().phase == PlaybackPhase::Playing {
                                            "Pause"
                                        } else {
                                            "Play"
                                        }
                                    }
                                    on:click=move |_| muse.toggle_play()
                                >
                                    {move || {
                                        if muse.playback.get().phase == PlaybackPhase::Playing {
                                            view! { <PauseIcon /> }.into_any()
                                        } else {
                                            view! { <PlayIcon /> }.into_any()
                                        }
                                    }}
                                </button>
                                <button
                                    type="button"
                                    class="icon-btn"
                                    title="Next Piece"
                                    on:click=move |_| muse.next_piece(true)
                                >
                                    <NextIcon />
                                </button>
                            </div>

                            <div class="player-secondary">
                                <button
                                    type="button"
                                    class="icon-btn heart-btn"
                                    class:kept=move || muse.kept.get()
                                    title=move || if muse.kept.get() { "Kept" } else { "Keep" }
                                    on:click=move |_| muse.keep()
                                >
                                    <HeartIcon filled=muse.kept.get() />
                                </button>
                                <span class="volume-control">
                                    <VolumeIcon />
                                    <input
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.01"
                                        prop:value=move || muse.volume.get().to_string()
                                        on:input=move |ev| {
                                            let target: HtmlInputElement = event_target(&ev);
                                            if let Ok(v) = target.value().parse::<f64>() {
                                                muse.volume.set(v);
                                                if let Some(audio) = muse.audio_ref.get_untracked() {
                                                    let audio: HtmlAudioElement = audio.into();
                                                    audio.set_volume(v);
                                                }
                                            }
                                        }
                                    />
                                </span>
                                <select
                                    class="renderer-select"
                                    title="Render backend"
                                    aria-label="Render backend"
                                    on:change=move |ev| {
                                        let v = event_target_value(&ev);
                                        muse.renderer_preference
                                            .set(match v.as_str() {
                                                "native" => Some("native"),
                                                "fluidsynth" => Some("fluidsynth"),
                                                _ => None,
                                            });
                                    }
                                >
                                    <option value="auto">"Auto"</option>
                                    <option value="fluidsynth">"FluidSynth"</option>
                                    <option value="native">"Native"</option>
                                </select>
                            </div>
                        }
                            .into_any()
                    })
            }}
        </footer>
    }
}

/// Self-scheduling `requestAnimationFrame` loop for the small corner
/// preview — reuses `pages::draw_frame` (the exact same visual identity
/// as the Listen hero canvas, just smaller) so switching modes anywhere
/// looks consistent everywhere. `Still` mode still redraws every frame
/// (cheap — one frozen scene) rather than special-casing the loop to
/// stop, keeping this function as simple as `pages::start_visualizer`.
fn start_mini_viz(
    canvas: HtmlCanvasElement,
    viz_mode: RwSignal<VizMode>,
    current_style: RwSignal<String>,
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
        let t = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() / 1000.0)
            .unwrap_or(0.0)
            * 0.5;

        let w = canvas.client_width().max(1) as u32;
        let h = canvas.client_height().max(1) as u32;
        if canvas.width() != w {
            canvas.set_width(w);
        }
        if canvas.height() != h {
            canvas.set_height(h);
        }

        let palette = palette::palette_for(&current_style.get_untracked());
        // Deliberately all `None` — this 40px corner preview stays a small
        // decorative echo of the mode, not a second live-audio consumer;
        // the main hero canvas is the one place playback progress, real
        // analyser spectrum, and composition evidence actually drive the
        // drawing.
        crate::pages::draw_frame(
            &ctx,
            w as f64,
            h as f64,
            viz_mode.get_untracked(),
            &palette,
            t,
            None,
            None,
            None,
        );

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

/// Self-scheduling `requestAnimationFrame` loop for the wave scrub bar —
/// same structure as `pages.rs`'s `start_visualizer` (read state via
/// `get_untracked()` every frame rather than tracking it, so this loop
/// itself is never a reactive dependency). Runs for the app's lifetime
/// since `PlayerBar` is mounted once outside `<Routes>` alongside the
/// shared `<audio>` element.
fn start_progress_wave(canvas: HtmlCanvasElement, muse: MuseState) {
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
        let t = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now() / 1000.0)
            .unwrap_or(0.0);

        let w = canvas.client_width().max(1) as u32;
        let h = canvas.client_height().max(1) as u32;
        if canvas.width() != w {
            canvas.set_width(w);
        }
        if canvas.height() != h {
            canvas.set_height(h);
        }

        // `audio_reactivity::waveform()` reads the one shared analysis
        // graph — `None` before anything has ever played, or if Web Audio
        // construction failed; `draw_wave` reads that as "fall back to
        // idle decoration." A PAUSED analyser reports near-silence — a
        // dead flatline — so paused states get the calm-sea idle too,
        // matching the Listen visualizer's paused behavior.
        let playing = muse.playback.get_untracked().phase == PlaybackPhase::Playing;
        let samples = if playing {
            audio_reactivity::waveform()
        } else {
            None
        };

        let palette = palette::palette_for(&muse.current_style.get_untracked());
        let playback = muse.playback.get_untracked();
        draw_wave(
            &ctx,
            w as f64,
            h as f64,
            &palette,
            t,
            playback.position_seconds,
            playback.duration_seconds.unwrap_or(0.0),
            samples.as_ref().map(|s| s.as_slice()).unwrap_or(&[]),
            muse.timeline_mode.get_untracked(),
        );

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

/// A soft translucent glow behind the wave, tinted to the current piece's
/// palette — the "calm background" the plain dark bar was missing. Cheap
/// (one gradient fill) and drawn first so the wave lines sit on top of it.
fn draw_glow(ctx: &CanvasRenderingContext2d, w: f64, h: f64, palette: &Palette) {
    let grad = ctx.create_linear_gradient(0.0, 0.0, w, 0.0);
    let _ = grad.add_color_stop(0.0, &format!("rgba({}, 0.02)", palette.b));
    let _ = grad.add_color_stop(0.5, &format!("rgba({}, 0.14)", palette.a));
    let _ = grad.add_color_stop(1.0, &format!("rgba({}, 0.02)", palette.b));
    ctx.set_fill_style_canvas_gradient(&grad);
    ctx.fill_rect(0.0, 0.0, w, h);
}

/// Real waveform/bar display when a piece is actually playing (see
/// `audio_reactivity::waveform`) — genuinely reactive to the music rather
/// than decoration, in either of two layouts the user picks between via
/// `TimelineMode`: a continuous line (`Wave`, this app's original look) or
/// a compact amplitude-envelope bar strip (`Bar`). Falls back to
/// `draw_calm_sea_wave`'s synthetic idle motion before anything has played
/// yet or if Web Audio construction failed — that idle fallback is shared
/// by both modes, since it represents "nothing to show yet", not a
/// Bar-vs-Wave distinction. The actual seek control is the transparent
/// range input layered on top of this canvas (see `PlayerBar`'s doc
/// comment) — this is display only.
fn draw_wave(
    ctx: &CanvasRenderingContext2d,
    w: f64,
    h: f64,
    palette: &Palette,
    t: f64,
    current_time: f64,
    duration: f64,
    samples: &[u8],
    mode: TimelineMode,
) {
    ctx.clear_rect(0.0, 0.0, w, h);
    draw_glow(ctx, w, h, palette);
    ctx.set_line_cap("round");

    if samples.is_empty() {
        draw_calm_sea_wave(ctx, w, h, palette, t);
    } else {
        match mode {
            TimelineMode::Wave => {
                let n = samples.len();
                for pass in 0..3usize {
                    let color = if pass == 1 { palette.a } else { palette.b };
                    let alpha = 0.85 - pass as f64 * 0.22;
                    let scale = 1.0 - pass as f64 * 0.22;
                    ctx.set_stroke_style_str(&format!("rgba({color}, {alpha:.3})"));
                    ctx.set_line_width(2.4 - pass as f64 * 0.5);
                    ctx.begin_path();
                    for (i, sample) in samples.iter().enumerate() {
                        let x = (i as f64 / (n as f64 - 1.0)) * w;
                        let centered = (*sample as f64 - 128.0) / 128.0;
                        let y = h / 2.0 + centered * scale * (h * 0.42);
                        if i == 0 {
                            ctx.move_to(x, y);
                        } else {
                            ctx.line_to(x, y);
                        }
                    }
                    let _ = ctx.stroke();
                }
            }
            TimelineMode::Bar => {
                // A compact amplitude-envelope strip: real samples bucketed
                // into N columns, each bar's height the bucket's peak
                // deviation from center — genuinely audio-reactive, not a
                // plain decorative progress fill.
                const N: usize = 64;
                let bucket = (samples.len() / N).max(1);
                ctx.set_line_cap("round");
                for i in 0..N {
                    let start = i * bucket;
                    if start >= samples.len() {
                        break;
                    }
                    let end = (start + bucket).min(samples.len());
                    let peak = samples[start..end]
                        .iter()
                        .map(|&s| ((s as f64) - 128.0).abs() / 128.0)
                        .fold(0.0_f64, f64::max);
                    let x = (i as f64 + 0.5) * (w / N as f64);
                    let len = (h * 0.12) + peak * (h * 0.36);
                    let alpha = 0.35 + peak * 0.6;
                    ctx.set_stroke_style_str(&format!("rgba({}, {alpha:.3})", palette.a));
                    ctx.set_line_width((w / N as f64 * 0.5).max(1.5));
                    ctx.begin_path();
                    ctx.move_to(x, h / 2.0 - len);
                    ctx.line_to(x, h / 2.0 + len);
                    let _ = ctx.stroke();
                }
            }
        }
    }

    if duration > 0.0 {
        let frac = (current_time / duration).clamp(0.0, 1.0);
        let px = frac * w;
        ctx.set_stroke_style_str(&format!("rgba({}, 0.9)", palette.a));
        ctx.set_line_width(2.0);
        ctx.begin_path();
        ctx.move_to(px, 0.0);
        ctx.line_to(px, h);
        let _ = ctx.stroke();
    }
}

/// The calm-sea idle — shown while paused, before anything has played yet
/// (no analyser tap exists), or if Web Audio construction fails, so the
/// bar is never blank and never a dead flatline. Three long-wavelength
/// swells drifting an order of magnitude slower than the old idle motion
/// (~20s per crossing, alternating directions), each with a translucent
/// fill below its crest so the bar reads as water at rest, matching the
/// Listen visualizer's paused scene (`pages.rs::draw_calm_sea`).
fn draw_calm_sea_wave(ctx: &CanvasRenderingContext2d, w: f64, h: f64, palette: &Palette, t: f64) {
    let n = 96usize;
    for pass in 0..3usize {
        let color = if pass == 1 { palette.a } else { palette.b };
        let alpha = 0.40 - pass as f64 * 0.10;
        let amp = h * (0.16 - pass as f64 * 0.035);
        let base_y = h * (0.46 + pass as f64 * 0.10);
        let dir = if pass % 2 == 0 { 1.0 } else { -1.0 };
        let phase =
            dir * t * std::f64::consts::TAU * (0.05 + pass as f64 * 0.014) + pass as f64 * 2.1;
        ctx.set_stroke_style_str(&format!("rgba({color}, {alpha:.3})"));
        ctx.set_line_width(2.0 - pass as f64 * 0.4);
        ctx.begin_path();
        for i in 0..n {
            let x = (i as f64 / (n as f64 - 1.0)) * w;
            let k = (x / w) * std::f64::consts::TAU * 1.6;
            let v = (k + phase).sin() * 0.65 + (k * 2.3 - phase * 0.6).sin() * 0.35;
            let y = base_y + v * amp;
            if i == 0 {
                ctx.move_to(x, y);
            } else {
                ctx.line_to(x, y);
            }
        }
        let _ = ctx.stroke();
        // Water body under the crest.
        ctx.line_to(w, h);
        ctx.line_to(0.0, h);
        ctx.close_path();
        ctx.set_fill_style_str(&format!("rgba({color}, 0.05)"));
        let _ = ctx.fill();
    }
}

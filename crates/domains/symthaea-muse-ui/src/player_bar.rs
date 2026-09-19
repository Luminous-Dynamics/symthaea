// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! The persistent bottom playback bar — visible across every route.
//!
//! Presentation and actions follow the exact `PlaybackSource` currently loaded
//! in the shared reducer, not `MuseState::current`. This matters for review
//! auditions: imported works, etudes, qualification artifacts, and future
//! Studio alternatives can be audible without becoming the canonical candidate.

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

fn format_secs(s: f64) -> String {
    let s = s.max(0.0) as u64;
    format!("{}:{:02}", s / 60, s % 60)
}

/// Palette for the thing actually loaded into the transport. Review subjects
/// intentionally use `palette_for`'s neutral/Classical fallback rather than the
/// stale style of whatever canonical candidate happened to play previously.
fn audible_palette(muse: MuseState) -> Palette {
    let playback = muse.playback.get_untracked();
    let style = playback
        .source
        .as_ref()
        .map(|source| source.presentation.palette_style())
        .unwrap_or("Review");
    palette::palette_for(style)
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
        start_mini_viz(canvas_el.into(), muse.viz_mode, muse);
    });

    view! {
        <footer
            class="player-bar"
            class:visible=move || muse.playback.get().source.is_some()
        >
            <div class="player-viz-corner">
                <canvas node_ref=mini_viz_ref class="player-mini-viz" />
            </div>
            <div class="player-progress">
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
                        let playback = muse.playback.get();
                        let style = playback
                            .source
                            .as_ref()
                            .map(|source| source.presentation.palette_style())
                            .unwrap_or("Review");
                        let p = palette::palette_for(style);
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
                muse.playback.get().source.map(|source| {
                    let presentation = source.presentation;
                    let can_keep = presentation.can_keep();
                    let can_advance = presentation.can_advance_journey();
                    let can_change_renderer = presentation.can_change_renderer();
                    let p = palette::palette_for(presentation.palette_style());
                    let dot_style = format!(
                        "background: rgb({}); box-shadow: 0 0 10px rgba({},0.6);",
                        p.a,
                        p.a,
                    );
                    let play_btn_style =
                        format!("--lava-a: rgb({}); --lava-b: rgb({});", p.a, p.b);
                    let title = presentation.title;
                    let subtitle = presentation.subtitle.unwrap_or_default();

                    view! {
                        <div class="player-identity">
                            <span class="player-dot" style=dot_style></span>
                            <div class="player-text">
                                <span class="player-title">{title}</span>
                                <span class="player-sub">{subtitle}</span>
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
                            {can_advance.then(|| view! {
                                <button
                                    type="button"
                                    class="icon-btn"
                                    title="Next Piece"
                                    on:click=move |_| muse.next_piece(true)
                                >
                                    <NextIcon />
                                </button>
                            })}
                        </div>

                        <div class="player-secondary">
                            {can_keep.then(|| view! {
                                <button
                                    type="button"
                                    class="icon-btn heart-btn"
                                    class:kept=move || muse.kept.get()
                                    title=move || if muse.kept.get() { "Kept" } else { "Keep" }
                                    on:click=move |_| muse.keep()
                                >
                                    <HeartIcon filled=muse.kept.get() />
                                </button>
                            })}
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
                            {can_change_renderer.then(|| view! {
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
                            })}
                        </div>
                    }
                    .into_any()
                })
            }}
        </footer>
    }
}

fn start_mini_viz(canvas: HtmlCanvasElement, viz_mode: RwSignal<VizMode>, muse: MuseState) {
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

        let palette = audible_palette(muse);
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

        let playing = muse.playback.get_untracked().phase == PlaybackPhase::Playing;
        let samples = if playing {
            audio_reactivity::waveform()
        } else {
            None
        };

        let palette = audible_palette(muse);
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

fn draw_glow(ctx: &CanvasRenderingContext2d, w: f64, h: f64, palette: &Palette) {
    let grad = ctx.create_linear_gradient(0.0, 0.0, w, 0.0);
    let _ = grad.add_color_stop(0.0, &format!("rgba({}, 0.02)", palette.b));
    let _ = grad.add_color_stop(0.5, &format!("rgba({}, 0.14)", palette.a));
    let _ = grad.add_color_stop(1.0, &format!("rgba({}, 0.02)", palette.b));
    ctx.set_fill_style_canvas_gradient(&grad);
    ctx.fill_rect(0.0, 0.0, w, h);
}

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
        ctx.line_to(w, h);
        ctx.line_to(0.0, h);
        ctx.close_path();
        ctx.set_fill_style_str(&format!("rgba({color}, 0.05)"));
        let _ = ctx.fill();
    }
}

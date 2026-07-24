// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Route targets for the three top-level modes.
//!
//! `ListenPage` is Listen Mode P0 (`UI_mocks/MUSE_LISTEN_MODE_VISUALIZATION_DESIGN_SPEC.md`
//! §21): persistent playback, now-playing hero, Hybrid (here:
//! Radial/Bars/Waves) visualizer, Why This Piece?, Keep, Next Piece.
//! Ported from the legacy `studio/index.html`'s Listen tab, keeping only
//! the *idle* (non-live) visualizer math — that page also drives the same
//! canvas from a live WebAudio `AnalyserNode`; wiring that up is a
//! deliberate follow-up. Playback state itself lives in `MuseState`
//! (`state.rs`), shared across all three modes.
//!
//! `ResearchPage` has a real Overview panel (the shared current piece's
//! already-computed metrics — no new analysis, just exposing what
//! `/api/compose` already returns), a real Score view
//! (`score_view::ScoreView`, a piano roll from `/api/notes/{id}`), and a
//! real Evidence panel (`evidence_view::EvidenceView`, motif/sonority/
//! cadence/orchestration/structural-activity from `/api/piece/{id}/
//! listen-bundle`), but not yet the rest of Research Mode P0.
//! `CreatePage` lives in `create_page.rs` and is re-exported below.

use std::cell::Cell;
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use leptos_router::components::A;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::api::{self, ListenCompositionBundle, MotifsSummary, SectionInfo};
use crate::audio_reactivity;
use crate::evidence_view::EvidenceView;
use crate::icons::{RadialIcon, StillIcon};
use crate::journey::JourneyPolicy;
use crate::palette::{self, Palette};
use crate::playback::PlaybackPhase;
use crate::score_view::ScoreView;
use crate::state::{MuseState, VizMode};

pub use crate::create_page::CreatePage;

#[component]
pub fn ListenPage() -> impl IntoView {
    let muse = use_context::<MuseState>().expect("MuseState provided by App");
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    // Shared with the player bar's mini preview (`state.rs::MuseState::viz_mode`)
    // — one choice, reflected everywhere, not two toggles that could disagree.
    let viz_mode = muse.viz_mode;

    // Live current-section indicator: fetches the same `/api/motifs/{id}`
    // structure Research Mode's Motifs view uses, so Listen Mode can
    // answer "what section is playing" without leaving the player. Refetches
    // whenever the current piece changes; `has_structure: false` (the
    // honest answer for 6+ form kinds that bypass Form entirely — Fugue,
    // Sonata, ground forms, etc.) means `active_section` below just never
    // matches, and the badge stays hidden rather than showing something
    // fabricated.
    let sections = RwSignal::new(None::<MotifsSummary>);
    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            sections.set(None);
            return;
        };
        let id = c.id;
        spawn_local(async move {
            let result = api::fetch_motifs(api::DEFAULT_BACKEND, id).await;
            // Stale-response guard: if the user has already moved to another
            // piece by the time this resolves, discard it rather than
            // showing evidence for the wrong piece — a slower response for
            // piece A landing after a faster one for piece B is a real race,
            // not a hypothetical.
            if muse.current.get_untracked().map(|c| c.id) != Some(id) {
                return;
            }
            match result {
                Ok(s) => sections.set(Some(s)),
                Err(_) => sections.set(None),
            }
        });
    });
    let active_section = move || -> Option<SectionInfo> {
        let t = muse.playback.get().position_seconds;
        sections
            .get()
            .into_iter()
            .flat_map(|s| s.sections)
            .find(|sec| t >= sec.start_seconds && t < sec.end_seconds)
    };

    // The composition-evidence layer for the Radial map: real section
    // arcs/phrase boundaries/cadence markers/motif-occurrence positions
    // from `GET /api/piece/{id}/listen-bundle` — the same endpoint Research
    // Mode's Evidence view (`evidence_view.rs`) already uses. This is a
    // separate fetch from `sections` above on purpose: `sections` (bars +
    // key, from `/api/motifs/{id}`) drives the human-readable badge text;
    // this one (exact seconds + intensity, from the fuller bundle) drives
    // the visual map. Wrapped in `Rc` (not cloned by value) because the draw
    // loop reads it every animation frame — an `Rc` clone there is a
    // refcount bump, not a deep copy of every section/phrase/cadence/motif
    // vector at 60fps.
    // `Rc` isn't `Send + Sync`, so this needs `LocalStorage` (sound: wasm32
    // is single-threaded) rather than the default `RwSignal::new`, which
    // requires `SyncStorage`.
    let composition = RwSignal::new_local(None::<Rc<CompositionMap>>);
    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            composition.set(None);
            return;
        };
        let id = c.id;
        spawn_local(async move {
            let result = api::fetch_listen_bundle(api::DEFAULT_BACKEND, id).await;
            // Same stale-response guard as `sections` above: a slower
            // response for a piece the user has already navigated away
            // from must never overwrite the evidence for the piece
            // actually playing now.
            if muse.current.get_untracked().map(|c| c.id) != Some(id) {
                return;
            }
            match result {
                Ok(env) => {
                    composition.set(Some(Rc::new(CompositionMap::from_bundle(&env.payload))))
                }
                Err(_) => composition.set(None),
            }
        });
    });

    // First piece + start the visualizer once the canvas exists.
    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        if muse.current.get_untracked().is_none() {
            muse.next_piece(false);
        }
        start_visualizer(
            canvas_el.into(),
            viz_mode,
            muse.current_style,
            muse,
            composition,
        );
    });

    // Room background follows the style, like `applyPalette()` in the
    // legacy page — a real (tracked) reactive effect, unlike the drawn
    // visualization itself, since this is cheap and only needs to run on
    // style change, not every frame.
    Effect::new(move |_| {
        let style = muse.current_style.get();
        let Some(canvas_el) = canvas_ref.get_untracked() else {
            return;
        };
        let canvas: HtmlCanvasElement = canvas_el.into();
        let p = palette::palette_for(&style);
        // `.style()` is disambiguated via UFCS: `leptos::prelude::*`'s
        // `ElementExt::style()` (a reactive style-string builder) shadows
        // the inherent `web_sys::HtmlElement::style()` this needs.
        let _ = web_sys::HtmlElement::style(&canvas).set_property(
            "background",
            &format!(
                "radial-gradient(ellipse at 50% 62%, {}, {} 78%)",
                p.bg0, p.bg1
            ),
        );
    });

    let status_text = move || {
        muse.status.get()
            + if muse.composing.get() {
                " (warming next piece…)"
            } else {
                ""
            }
    };

    view! {
        <div class="panel listen-hero">
            <div class="listen-canvas-wrap">
                <canvas
                    node_ref=canvas_ref
                    class="listen-canvas"
                    title="click to play / pause"
                    on:click=move |_| muse.toggle_play()
                />
                <div class="listen-title-overlay">
                    <span class="listen-title-kicker">"Muse Listen"</span>
                    <h2 class="listen-title-name">
                        {move || muse.current.get().map(|c| c.title).unwrap_or_default()}
                    </h2>
                    <p class="listen-title-meta">
                        {move || {
                            muse.current
                                .get()
                                .map(|c| format!("{} · {}/4 · {:.0}s", c.style, c.meter, c.duration_secs))
                                .unwrap_or_default()
                        }}
                    </p>
                </div>
                <div class="listen-mode-selector">
                    {VizMode::ALL
                        .into_iter()
                        .map(|m| {
                            let icon = match m {
                                VizMode::Radial => view! { <RadialIcon /> }.into_any(),
                                VizMode::Still => view! { <StillIcon /> }.into_any(),
                            };
                            let label = m.label();
                            view! {
                                <button
                                    type="button"
                                    class="icon-btn listen-mode-btn"
                                    class:sel=move || muse.viz_mode.get() == m
                                    title=label
                                    aria-label=label
                                    on:click=move |_| muse.viz_mode.set(m)
                                >
                                    {icon}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>
                {move || {
                    active_section()
                        .map(|sec| {
                            let bars = if sec.end_bar > sec.start_bar {
                                format!("bars {}–{}", sec.start_bar + 1, sec.end_bar)
                            } else {
                                format!("bar {}", sec.start_bar + 1)
                            };
                            view! {
                                <div class="current-section-badge">
                                    <span class="current-section-kicker">"Current section"</span>
                                    <span class="current-section-role">{sec.role}</span>
                                    <span class="current-section-detail">
                                        {format!("{} {} · {}", sec.key_tonic, sec.key_tonality, bars)}
                                    </span>
                                </div>
                            }
                        })
                }}
                {move || {
                    muse.current
                        .get()
                        .map(|c| {
                            let p = palette::palette_for(&c.style);
                            let dot_style = format!("background: rgb({});", p.a);
                            view! {
                                <div class="current-form-badge">
                                    <span class="current-form-dot" style=dot_style></span>
                                    <span class="current-form-text">
                                        <span class="current-section-kicker">"Current form"</span>
                                        <span class="current-form-name">{c.grammar}</span>
                                    </span>
                                </div>
                            }
                        })
                }}
            </div>

            <div class="listen-info">
                <p class="muted">
                    {move || muse.current.get().map(|c| {
                        let traits = c.card.map(|card| card.traits.join(" · ")).unwrap_or_default();
                        let render_note = if c.renderer != "fluidsynth" {
                            " · native render (soundfont unavailable)"
                        } else {
                            ""
                        };
                        format!("{traits}{render_note}")
                    }).unwrap_or_default()}
                </p>
                <details class="why-this-piece-details">
                    <summary>"Why this piece unfolds this way"</summary>
                    <p class="why-this-piece">
                        {move || muse.current.get().map(|c| c.why.join(" ")).unwrap_or_default()}
                    </p>
                </details>
                <p class="status-line">{status_text}</p>

                <div class="journey-policy-selector" role="group" aria-label="Listen journey policy">
                    {JourneyPolicy::ALL
                        .into_iter()
                        .map(|policy| {
                            let label = policy.label();
                            view! {
                                <button
                                    type="button"
                                    class="journey-policy-btn"
                                    class:sel=move || muse.journey_policy() == policy
                                    title=label
                                    on:click=move |_| muse.set_journey_policy(policy)
                                >
                                    {label}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>

                <div class="listen-actions">
                    <button
                        type="button"
                        class="heart-btn"
                        class:kept=move || muse.kept.get()
                        on:click=move |_| muse.keep()
                    >
                        {move || if muse.kept.get() { "♥ kept" } else { "♡ keep" }}
                    </button>
                    <button type="button" on:click=move |_| muse.next_piece(true)>"Next Piece"</button>
                    <a
                        class="link-btn"
                        href=move || muse.current.get().map(|c| api::audio_url(api::DEFAULT_BACKEND, c.id)).unwrap_or_default()
                        download=move || muse.current.get().map(|c| format!("muse_seed{}.wav", c.seed)).unwrap_or_default()
                    >
                        "WAV"
                    </a>
                    <a
                        class="link-btn"
                        href=move || muse.current.get().map(|c| api::midi_url(api::DEFAULT_BACKEND, c.id)).unwrap_or_default()
                        download=move || muse.current.get().map(|c| format!("muse_seed{}.mid", c.seed)).unwrap_or_default()
                    >
                        "MIDI"
                    </a>
                </div>

                <p class="muted small">
                    "Open in "
                    <A href="/create">"Create"</A>
                    " to compose with full control, or "
                    <A href="/research">"Research"</A>
                    " to see how this piece is built."
                </p>
            </div>
        </div>
    }
}

/// Starts the self-scheduling `requestAnimationFrame` draw loop and
/// registers cleanup to cancel it when `ListenPage` unmounts (navigating
/// to Create/Research) — without this the loop runs forever against a
/// detached canvas. `viz_mode`/`style`/`muse` are read with
/// `get_untracked()` every frame rather than tracked, matching
/// `symthaea-web`'s `TopologyPage` pattern: the loop itself must not be a
/// reactive dependency of the signals it reads, or every mode/style
/// change would spawn a second concurrent loop.
fn start_visualizer(
    canvas: HtmlCanvasElement,
    viz_mode: RwSignal<VizMode>,
    current_style: RwSignal<String>,
    muse: MuseState,
    composition: RwSignal<Option<Rc<CompositionMap>>, leptos::prelude::LocalStorage>,
) {
    let ctx = match canvas.get_context("2d") {
        Ok(Some(ctx)) => match ctx.dyn_into::<CanvasRenderingContext2d>() {
            Ok(ctx) => ctx,
            Err(_) => return,
        },
        _ => return,
    };

    // Checked once at mount, not per-frame — a live media-query listener
    // that could flip mid-session is more machinery than this needs today;
    // revisiting the setting means reloading the page, an acceptable
    // tradeoff for a real reduced-motion mode over none at all.
    let reduced_motion = web_sys::window()
        .and_then(|w| w.match_media("(prefers-reduced-motion: reduce)").ok())
        .flatten()
        .map(|m| m.matches())
        .unwrap_or(false);

    let frame_closure: Rc<std::cell::RefCell<Option<Closure<dyn FnMut()>>>> =
        Rc::new(std::cell::RefCell::new(None));
    let frame_closure_clone = frame_closure.clone();
    let last_frame_id: Rc<Cell<i32>> = Rc::new(Cell::new(0));
    let last_frame_id_clone = last_frame_id.clone();

    let closure = Closure::wrap(Box::new(move || {
        // Hidden tabs: skip the (relatively expensive) draw + backing-store
        // work entirely rather than trust browsers' rAF throttling alone —
        // still reschedules, so drawing resumes immediately on refocus.
        let hidden = web_sys::window()
            .and_then(|w| w.document())
            .map(|d| d.hidden())
            .unwrap_or(false);

        if !hidden {
            let time = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or(0.0);
            // Reduced motion keeps real playback-driven state (progress,
            // audio reactivity, section/form badges) but freezes the
            // decorative idle drift `t` otherwise drives — per this app's
            // accessibility contract, meaning stays, ornamental motion goes.
            let t = if reduced_motion { 0.0 } else { time * 0.5 };

            // HiDPI: draw at the real device pixel density (capped at 2x —
            // higher buys little and costs real fill-rate) so the canvas
            // isn't visibly soft on anything above 1x. `ctx.scale` after
            // resizing the backing store lets all drawing code below keep
            // working in CSS-pixel coordinates.
            let dpr = web_sys::window()
                .map(|w| w.device_pixel_ratio())
                .unwrap_or(1.0)
                .clamp(1.0, 2.0);
            let w = canvas.client_width().max(1) as f64;
            let h = canvas.client_height().max(1) as f64;
            let w_px = (w * dpr).round() as u32;
            let h_px = (h * dpr).round() as u32;
            // `set_width`/`set_height` clear the canvas's backing store
            // even when the value doesn't change — only touch them (and
            // re-apply the DPR transform, which a resize also resets) on
            // an actual resize, since every frame redraws unconditionally
            // below anyway.
            if canvas.width() != w_px || canvas.height() != h_px {
                canvas.set_width(w_px);
                canvas.set_height(h_px);
                let _ = ctx.set_transform(dpr, 0.0, 0.0, dpr, 0.0, 0.0);
            }

            let playback = muse.playback.get_untracked();
            let playing = playback.phase == PlaybackPhase::Playing;
            let progress = playback
                .duration_seconds
                .filter(|d| *d > 0.0)
                .map(|d| (playback.position_seconds / d).clamp(0.0, 1.0));
            // Real audio, not synthetic motion, whenever something is
            // actually playing — `spectrum()` (frequency-domain) feeds
            // bass/spoke energy (see `draw_frame`'s doc comment). The hero
            // visualizer has no use for raw `waveform()` samples anymore:
            // that time-domain tap now belongs entirely to the player
            // bar's own timeline (`TimelineMode::Wave`). `None` (paused,
            // not yet played, or Web Audio construction failed) falls back
            // to the idle decorative signal inside `draw_frame`.
            let spectrum = if playing {
                audio_reactivity::spectrum()
            } else {
                None
            };

            let palette = palette::palette_for(&current_style.get_untracked());
            let comp = composition.get_untracked();
            draw_frame(
                &ctx,
                w,
                h,
                viz_mode.get_untracked(),
                &palette,
                t,
                progress,
                spectrum.as_ref().map(|s| s.as_slice()),
                comp.as_deref(),
            );
        }

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

    // `on_cleanup` requires `Send + Sync` (Leptos's bound is uniform across
    // CSR/SSR); `Rc`/`RefCell` are neither, but wasm32 is single-threaded
    // so wrapping in `SendWrapper` is sound here — the standard pattern
    // for this exact situation in Leptos CSR apps.
    let cleanup = send_wrapper::SendWrapper::new(move || {
        if let Some(window) = web_sys::window() {
            let _ = window.cancel_animation_frame(last_frame_id.get());
        }
        // Dropping the closure (by clearing the Rc's content) after
        // cancelling is what actually frees the JS-side function; keeping
        // it alive past this point would leak it.
        *frame_closure.borrow_mut() = None;
    });
    on_cleanup(move || cleanup.take()());
}

/// The Listen hero visualizer — `VizMode::Radial`'s whole-piece map, or
/// `Still`, its frozen variant. (`Bar`/`Wave` used to be modes here too;
/// they're now `TimelineMode` in the player bar instead — see
/// `state.rs`'s doc comments for why.) Real reactivity comes from
/// `spectrum` (see `audio_reactivity::spectrum`) whenever something is
/// actually playing, not synthetic sine motion. `progress` (0.0-1.0
/// through the piece, when duration is known) drives a real playhead on
/// the Radial ring rather than decoration — the one thing in this view
/// genuinely tied to where playback is. Geometry is proportional to
/// `min(w, h)` throughout so the scene never clips regardless of the
/// canvas's actual size.
pub(crate) fn draw_frame(
    ctx: &CanvasRenderingContext2d,
    w: f64,
    h: f64,
    mode: VizMode,
    palette: &Palette,
    t: f64,
    progress: Option<f64>,
    spectrum: Option<&[u8]>,
    composition: Option<&CompositionMap>,
) {
    ctx.clear_rect(0.0, 0.0, w, h);
    // Real audio, when we have it: bass energy and the per-spoke/per-bar
    // scalar come from actual frequency-domain data
    // (`audio_reactivity::spectrum`, low-index bins are low frequencies),
    // not from time-domain waveform amplitude — a waveform sample tells you
    // how loud the signal is right now, not how much bass is in it.
    // `samples` (waveform) stays reserved for Waves mode's raw line, where
    // preserving polarity/zero-crossings is exactly what's wanted. Falls
    // back to the original synthetic idle drift when nothing is playing.
    let (bass, mid): (f64, Box<dyn Fn(usize) -> f64>) = match spectrum {
        Some(s) if !s.is_empty() => {
            let bass_bins = &s[..s.len().min(6)];
            let bass_energy = bass_bins.iter().map(|&b| b as f64 / 255.0).sum::<f64>()
                / bass_bins.len().max(1) as f64;
            let s = s.to_vec();
            (
                bass_energy.clamp(0.0, 1.0),
                Box::new(move |i: usize| {
                    let idx = i * s.len() / 72.max(1);
                    (s.get(idx).copied().unwrap_or(0) as f64 / 255.0).clamp(0.0, 1.0)
                }),
            )
        }
        _ => (
            0.12 + 0.05 * (t * 1.3).sin(),
            Box::new(move |i: usize| 0.06 + 0.04 * (t * 2.0 + i as f64 * 0.4).sin()),
        ),
    };

    match mode {
        VizMode::Radial => {
            draw_radial(ctx, w, h, palette, bass, &mid, progress, true, composition);
        }
        VizMode::Still => {
            // A single frozen frame — the Radial scene at rest, for
            // anyone who finds continuous motion distracting rather than
            // immersive. Same drawing code, just with no reactive samples.
            let mid_still = |i: usize| -> f64 { 0.06 + 0.02 * (i as f64 * 0.4).sin() };
            draw_radial(
                ctx,
                w,
                h,
                palette,
                0.12,
                &mid_still,
                progress,
                false,
                composition,
            );
        }
    }
}

/// The Radial primitive: a whole-piece map, not a spinning decoration.
/// Structural positions (the 72 spokes) stay angularly fixed — only their
/// brightness/length respond to audio — so what rotates is meaning
/// (nothing does) rather than wall-clock time. `progress`, when known,
/// draws a real playhead on the outer ring: the one part of this view
/// genuinely tied to where playback actually is. `composition`, when the
/// evidence bundle has loaded, draws the actual composed structure — real
/// section boundaries, phrase closes, cadences, and scored motif returns —
/// just outside the ring, so the map reads as "the living architecture of
/// the piece" rather than only an audio-reactive orb. All geometry scales
/// off `min(w, h)` so the scene stays fully visible at any canvas size
/// instead of the fixed-pixel radii clipping on shorter canvases.
#[allow(clippy::too_many_arguments)]
fn draw_radial(
    ctx: &CanvasRenderingContext2d,
    w: f64,
    h: f64,
    palette: &Palette,
    bass: f64,
    mid: &dyn Fn(usize) -> f64,
    progress: Option<f64>,
    twinkle: bool,
    composition: Option<&CompositionMap>,
) {
    let cx = w / 2.0;
    let cy = h * 0.52;
    let base = w.min(h);
    // `r0` is the audio-reactive interior — glow and spoke origin are
    // allowed to breathe with bass energy. `outer_ring`/`evidence_ring`
    // deliberately do NOT derive from it: they anchor the playhead, the
    // ring itself, and every piece of composition evidence (sections,
    // phrases, cadences, motifs), and per this view's own design principle
    // — "musical structure remains spatially stable; playback and audio
    // make it come alive" — none of that may drift just because the music
    // got louder. Fixed high enough that even a max-bass spoke tip
    // (r0_max + 0.02 + spoke_max_len = 0.21 + 0.02 + 0.19 = 0.42 * base)
    // stays safely inside the ring.
    let r0 = base * 0.15 + bass * base * 0.06;
    let spoke_inner = r0 + base * 0.02;
    let spoke_max_len = base * 0.19;
    let outer_ring = base * 0.44;
    let evidence_ring = base * 0.48;

    if let Ok(glow) = ctx.create_radial_gradient(cx, cy, base * 0.01, cx, cy, r0 * 2.1) {
        let _ = glow.add_color_stop(
            0.0,
            &format!("rgba({}, {:.3})", palette.a, 0.5 + bass * 0.5),
        );
        let _ = glow.add_color_stop(0.45, &format!("rgba({}, 0.22)", palette.b));
        let _ = glow.add_color_stop(1.0, &format!("rgba({}, 0)", palette.b));
        ctx.set_fill_style_canvas_gradient(&glow);
        ctx.begin_path();
        let _ = ctx.arc(cx, cy, r0 * 2.1, 0.0, std::f64::consts::TAU);
        let _ = ctx.fill();
    }

    // Angularly stationary — no `+ t * k` term. Audio brightens/lengthens
    // each spoke in place; the map itself never spins.
    for i in 0..72usize {
        let a = (i as f64 / 72.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
        let m = mid(i);
        let len = base * 0.03 + m * spoke_max_len;
        let r1 = spoke_inner;
        let r2 = spoke_inner + len;
        ctx.set_stroke_style_str(&format!("rgba({}, {:.3})", palette.b, 0.18 + m * 0.7));
        ctx.set_line_width((base * 0.006).max(1.4));
        ctx.begin_path();
        ctx.move_to(cx + a.cos() * r1, cy + a.sin() * r1);
        ctx.line_to(cx + a.cos() * r2, cy + a.sin() * r2);
        let _ = ctx.stroke();
    }

    ctx.set_stroke_style_str("rgba(163, 150, 138, 0.16)");
    ctx.set_line_width(1.0);
    ctx.begin_path();
    let _ = ctx.arc(cx, cy, outer_ring, 0.0, std::f64::consts::TAU);
    let _ = ctx.stroke();

    if let Some(map) = composition {
        draw_composition_evidence(ctx, cx, cy, outer_ring, evidence_ring, base, palette, map);
    }

    // The real playhead: the one element of this map that actually moves
    // with playback, not with wall-clock time. Starts at 12 o'clock,
    // sweeps clockwise with `progress`.
    if let Some(p) = progress {
        let a = p * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
        let x = cx + a.cos() * outer_ring;
        let y = cy + a.sin() * outer_ring;
        ctx.set_fill_style_str(&format!("rgba({}, 0.95)", palette.a));
        ctx.begin_path();
        let _ = ctx.arc(x, y, (base * 0.012).max(3.0), 0.0, std::f64::consts::TAU);
        let _ = ctx.fill();
        // A short trailing arc behind the playhead so its direction of
        // travel (and thus "where we've been") reads at a glance.
        ctx.set_stroke_style_str(&format!("rgba({}, 0.5)", palette.a));
        ctx.set_line_width((base * 0.006).max(1.5));
        ctx.begin_path();
        let _ = ctx.arc(cx, cy, outer_ring, a - 0.35, a);
        let _ = ctx.stroke();
    }

    draw_scattered_dots(ctx, cx, cy, r0, outer_ring, palette, bass, twinkle);
}

/// A loose field of small dots scattered across the ring — deterministic
/// per-index positions via a SplitMix64-style hash (well-distributed
/// across the full `[0, 1)` range, unlike a naive `i * constant >> 8`
/// which only ever spans a narrow sliver for small `i`), so the same ~40
/// positions appear every frame. Purely decorative texture (no per-note/
/// per-motif source to honestly place these against yet), but per this
/// view's own rule — every animated part responds to the music or shows
/// structure, nothing moves for its own sake — brightness tracks real
/// `bass` energy rather than wall-clock time; positions themselves never
/// move at all.
fn draw_scattered_dots(
    ctx: &CanvasRenderingContext2d,
    cx: f64,
    cy: f64,
    r0: f64,
    outer_ring: f64,
    palette: &Palette,
    bass: f64,
    twinkle: bool,
) {
    fn hash01(seed: u64) -> f64 {
        let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        (x as f64) / (u64::MAX as f64)
    }

    const N: usize = 40;
    let band = (outer_ring - r0).max(1.0);
    for i in 0..N {
        let h1 = hash01(i as u64 * 2 + 1);
        let h2 = hash01(i as u64 * 2 + 2);
        let angle = h1 * std::f64::consts::TAU;
        let radius = r0 + h2 * band;
        let phase = if twinkle {
            0.35 + 0.5 * bass.clamp(0.0, 1.0)
        } else {
            0.55
        };
        let size = 1.0 + h1 * 1.6;
        let x = cx + angle.cos() * radius;
        let y = cy + angle.sin() * radius;
        ctx.set_fill_style_str(&format!("rgba({}, {:.3})", palette.a, phase * 0.85));
        ctx.begin_path();
        let _ = ctx.arc(x, y, size, 0.0, std::f64::consts::TAU);
        let _ = ctx.fill();
    }
}

/// A lightweight, drawing-only extraction of `ListenCompositionBundle` —
/// section/phrase/cadence/motif-occurrence *positions*, nothing else.
/// Cloned once per animation frame (cheap: a handful of scalars), unlike
/// the full bundle which also carries every symbolic note event and the
/// sonority/orchestration timelines. Empty fields (e.g. `motifs` when the
/// bundle's own conservative occurrence threshold finds none — see
/// `evidence_view.rs`'s doc comment) are drawn as honestly absent, never
/// fabricated.
#[derive(Clone, Debug, Default)]
pub(crate) struct CompositionMap {
    pub duration_seconds: f64,
    pub sections: Vec<CompositionSection>,
    pub phrases: Vec<CompositionPhrase>,
    pub cadences: Vec<f64>,
    /// `(start_seconds, end_seconds)` of each scored motif occurrence.
    pub motifs: Vec<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub(crate) struct CompositionSection {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub intensity: f32,
    /// `SectionRegion::role` from the backend — a real semantic slug (e.g.
    /// Rondo's "refrain"/"episode"/"return", Passacaglia's "cycle"/
    /// "climax") for the forms `expected_section_labels` in `muse_studio.rs`
    /// recognizes, or the literal string `"region"` as an honest "no
    /// identity signal available" marker for every other form. Drawing code
    /// must treat `"region"` as meaning exactly that, not as one more
    /// distinct identity to color by.
    pub role: String,
}

#[derive(Clone, Debug)]
pub(crate) struct CompositionPhrase {
    pub end_seconds: f64,
    pub cadential: bool,
}

impl CompositionMap {
    fn from_bundle(payload: &ListenCompositionBundle) -> Self {
        Self {
            duration_seconds: payload.duration_seconds,
            sections: payload
                .sections
                .iter()
                .map(|s| CompositionSection {
                    start_seconds: s.start.seconds,
                    end_seconds: s.end.seconds,
                    intensity: s.intensity,
                    role: s.role.clone(),
                })
                .collect(),
            phrases: payload
                .phrases
                .iter()
                .map(|p| CompositionPhrase {
                    end_seconds: p.end.seconds,
                    cadential: p.closes_with_cadential_marker,
                })
                .collect(),
            cadences: payload.cadences.iter().map(|c| c.at.seconds).collect(),
            motifs: payload
                .motif_occurrences
                .iter()
                .map(|o| (o.start.seconds, o.end.seconds))
                .collect(),
        }
    }
}

/// The composition-evidence layer: real section arcs, phrase-boundary
/// ticks, cadence markers, and scored motif-occurrence positions, drawn as
/// a band just outside the audio-reactive spoke ring. This is what turns
/// the Radial view from "an audio-reactive orb" into a map of the piece's
/// actual architecture — every position here traces to a real field on
/// `GET /api/piece/{id}/listen-bundle`, normalized against the bundle's
/// own `duration_seconds` (kept separate from the playhead's
/// playback-driven `progress` fraction, since the two duration sources —
/// symbolic score vs. rendered audio — can differ by rendering padding).
#[allow(clippy::too_many_arguments)]
fn draw_composition_evidence(
    ctx: &CanvasRenderingContext2d,
    cx: f64,
    cy: f64,
    outer_ring: f64,
    evidence_ring: f64,
    base: f64,
    palette: &Palette,
    map: &CompositionMap,
) {
    let dur = map.duration_seconds.max(0.001);
    let angle_at = |seconds: f64| -> f64 {
        (seconds / dur).clamp(0.0, 1.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2
    };

    // Section arcs: colored by formal identity when the backend actually
    // provides one. `SectionRegion::role` carries real semantic slugs (e.g.
    // Rondo's refrain/episode/return, Passacaglia's cycle/climax) for the
    // handful of forms `expected_section_labels` (`muse_studio.rs`)
    // recognizes — coloring by first-seen-role-position means both
    // "return" sections in a Rondo share a color, and Passacaglia's lone
    // "climax" correctly stands out instead of blending into the
    // surrounding cycles the way plain index-parity would. Every other
    // form's sections all report the literal role `"region"` — an honest
    // "no identity signal available" marker, not one more identity to
    // group sections by — so those fall back to alternating strictly by
    // position, which is still enough to separate adjacent section
    // boundaries visually. Thickness/opacity always track the section's
    // own intensity regardless of which coloring path is used.
    let mut seen_roles: Vec<&str> = Vec::new();
    ctx.set_line_cap("butt");
    for (i, s) in map.sections.iter().enumerate() {
        let a0 = angle_at(s.start_seconds);
        let a1 = angle_at(s.end_seconds).max(a0 + 0.01);
        let color_index = if s.role == "region" {
            i
        } else {
            match seen_roles.iter().position(|r| *r == s.role.as_str()) {
                Some(pos) => pos,
                None => {
                    seen_roles.push(s.role.as_str());
                    seen_roles.len() - 1
                }
            }
        };
        let color = if color_index % 2 == 0 {
            &palette.a
        } else {
            &palette.b
        };
        let alpha = 0.18 + (s.intensity as f64).clamp(0.0, 1.0) * 0.32;
        ctx.set_stroke_style_str(&format!("rgba({color}, {alpha:.3})"));
        ctx.set_line_width((base * 0.018).max(3.0));
        ctx.begin_path();
        let _ = ctx.arc(cx, cy, evidence_ring, a0, a1);
        let _ = ctx.stroke();
    }

    // Phrase-boundary ticks at each phrase's close — brighter where the
    // score marks a genuine cadential arrival, dimmer for a plain phrase
    // break, so the eye can tell "this is where a musical thought ends"
    // from "this is where a cadence lands" (drawn separately below).
    let tick_r1 = outer_ring - base * 0.008;
    let tick_r2 = evidence_ring + base * 0.006;
    for p in &map.phrases {
        let a = angle_at(p.end_seconds);
        let color = if p.cadential {
            &palette.a
        } else {
            "163, 150, 138"
        };
        ctx.set_stroke_style_str(&format!("rgba({color}, 0.5)"));
        ctx.set_line_width(1.3);
        ctx.begin_path();
        ctx.move_to(cx + a.cos() * tick_r1, cy + a.sin() * tick_r1);
        ctx.line_to(cx + a.cos() * tick_r2, cy + a.sin() * tick_r2);
        let _ = ctx.stroke();
    }

    // Cadence markers: exact score-marked arrivals, sitting on the ring
    // itself.
    for at in &map.cadences {
        let a = angle_at(*at);
        let x = cx + a.cos() * outer_ring;
        let y = cy + a.sin() * outer_ring;
        ctx.set_fill_style_str(&format!("rgba({}, 0.9)", palette.a));
        ctx.begin_path();
        let _ = ctx.arc(x, y, (base * 0.008).max(2.0), 0.0, std::f64::consts::TAU);
        let _ = ctx.fill();
    }

    // Scored motif-occurrence markers: small outlined rings, between the
    // spoke ring and the section-arc band. Honestly absent (no markers
    // drawn) when the bundle's conservative similarity threshold found no
    // qualifying returns — never invented.
    let motif_r = (outer_ring + evidence_ring) / 2.0;
    for (start, _end) in &map.motifs {
        let a = angle_at(*start);
        let x = cx + a.cos() * motif_r;
        let y = cy + a.sin() * motif_r;
        ctx.set_stroke_style_str(&format!("rgba({}, 0.85)", palette.b));
        ctx.set_line_width(1.6);
        ctx.begin_path();
        let _ = ctx.arc(x, y, (base * 0.011).max(2.5), 0.0, std::f64::consts::TAU);
        let _ = ctx.stroke();
    }
}

/// Research Mode: Overview (the current shared piece's already-computed
/// metrics — `/api/compose` returns all of these; nothing here triggers
/// new analysis) plus a real Score view (`score_view::ScoreView`).
/// Harmony/Motifs/Orchestration/etc. views are not built yet.
#[component]
pub fn ResearchPage() -> impl IntoView {
    let muse = use_context::<MuseState>().expect("MuseState provided by App");

    view! {
        <div class="panel">
            <h2>"Research — Overview"</h2>
            {move || match muse.current.get() {
                None => view! {
                    <p class="muted">
                        "Nothing playing yet — open "
                        <A href="/">"Listen"</A>
                        " to start a piece, then come back here to see how it's built."
                    </p>
                }.into_any(),
                Some(c) => view! {
                    <dl class="metric-grid">
                        <dt>"Style"</dt><dd>{c.style.clone()}</dd>
                        <dt>"Meter"</dt><dd>{format!("{}/4", c.meter)}</dd>
                        <dt>"Duration"</dt><dd>{format!("{:.1}s", c.duration_secs)}</dd>
                        <dt>"Grammar"</dt><dd>{c.grammar.clone()}</dd>
                        {c.ending.clone().map(|e| view! {
                            <dt>"Ending"</dt><dd>{e}</dd>
                        })}
                        <dt title="Integration of the score-as-system — spectral MIP over the voice×segment dependency graph. Score analysis, not consciousness.">
                            "Φ (structural integration)"
                        </dt>
                        <dd>{format!("{:.3}", c.phi)}</dd>
                        <dt title="Vertical/consonance coherence">"Local coherence"</dt>
                        <dd>{format!("{:.3}", c.local_coherence)}</dd>
                        <dt title="Long-range/motif coherence">"Global coherence"</dt>
                        <dd>{format!("{:.3}", c.global_coherence)}</dd>
                        {c.similarity.map(|s| view! {
                            <dt title="Cosine similarity to the prompt in CLAP space">"Prompt similarity"</dt>
                            <dd>{format!("{s:.3}")}</dd>
                        })}
                    </dl>
                }.into_any(),
            }}
        </div>

        <div class="panel">
            <h2>"Score"</h2>
            <ScoreView muse=muse />
        </div>

        <div class="panel">
            <h2>"Evidence"</h2>
            <EvidenceView muse=muse />
        </div>
    }
}

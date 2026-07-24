// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Muse Atlas — Phase 1 + lenses: a diagnostic 2D map of Muse's OWN
//! generated output (in-session candidates + persisted kept/liked pieces),
//! from `GET /api/atlas`. Points are placed by the same structural
//! fingerprints the diversity census measures, projected to 2D via a
//! deterministic top-2-PCA (`symthaea_music_theory::fingerprint::project_2d`)
//! — this is a diagnostic scatter, not a semantically-labeled map.
//!
//! A "lens" (Combined/Motif & Form/Harmony/Rhythm/Orchestration) is just a
//! server-side reweighting of the SAME fingerprint before projection — not
//! a new embedding. Selecting one re-fetches `/api/atlas?lens=...` and the
//! points visibly move. Selecting a point also fetches
//! `/api/atlas/compare?a=...&b=<its nearest neighbor>` and shows a real,
//! computed per-layer distance breakdown ("why nearby") — not a fabricated
//! explanation.
//!
//! Deliberately NOT the full "Muse Atlas" design essay's architecture: no
//! rights/provenance model, no external/human registered works, no
//! lineage graph, no zoomable tile system, no custom user-defined weight
//! sliders (a small preset menu only). Fetches once on mount (plus manual
//! Refresh/lens change), same pattern as `liked_page.rs`, since it isn't
//! tied to whatever's currently playing.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

use crate::api::{self, AtlasCompareResponse, AtlasPoint};
use crate::palette;

/// The lens preset menu — `(value sent to the API, human label)`. Kept as a
/// small, understandable menu rather than a slider wall, per the Atlas
/// design essay's own recommendation.
const LENSES: [(&str, &str); 5] = [
    ("combined", "Combined identity"),
    ("motif_form", "Motif & Form"),
    ("harmony", "Harmony"),
    ("rhythm", "Rhythm & Groove"),
    ("orchestration", "Orchestration"),
];

fn lens_label(value: &str) -> &'static str {
    LENSES
        .iter()
        .find(|(v, _)| *v == value)
        .map(|(_, l)| *l)
        .unwrap_or("Combined identity")
}

#[component]
pub fn AtlasPage() -> impl IntoView {
    let points = RwSignal::new(Vec::<AtlasPoint>::new());
    let status = RwSignal::new(String::new());
    let loaded = RwSignal::new(false);
    let lens = RwSignal::new("combined".to_string());
    let selected = RwSignal::new(None::<AtlasPoint>);
    let comparison = RwSignal::new(None::<AtlasCompareResponse>);
    let compare_status = RwSignal::new(String::new());
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();

    let reload = move || {
        status.set("loading…".to_string());
        let requested_lens = lens.get_untracked();
        spawn_local(async move {
            match api::fetch_atlas(api::DEFAULT_BACKEND, Some(&requested_lens)).await {
                Ok(s) => {
                    lens.set(s.lens.clone());
                    points.set(s.points);
                    status.set(String::new());
                }
                Err(e) => {
                    points.set(Vec::new());
                    status.set(format!("couldn't reach the atlas endpoint — {e}"));
                }
            }
            loaded.set(true);
        });
    };

    Effect::new(move |_| {
        if !loaded.get_untracked() {
            reload();
        }
    });

    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        draw_atlas(canvas_el.into(), points);
    });

    // Selecting a point fetches "why nearby" against its precomputed
    // nearest neighbor (by RAW, lens-independent fingerprint distance) —
    // the compare-to-nearest-neighbor default the design essay suggested,
    // kept simple rather than an explicit two-point picker.
    Effect::new(move |_| {
        let Some(p) = selected.get() else {
            comparison.set(None);
            compare_status.set(String::new());
            return;
        };
        let Some(nearest) = p.nearest_id.clone() else {
            comparison.set(None);
            compare_status.set("only one piece on the map — nothing to compare to".to_string());
            return;
        };
        compare_status.set("loading…".to_string());
        spawn_local(async move {
            match api::fetch_atlas_compare(api::DEFAULT_BACKEND, &p.id, &nearest).await {
                Ok(c) => {
                    comparison.set(Some(c));
                    compare_status.set(String::new());
                }
                Err(e) => {
                    comparison.set(None);
                    compare_status.set(format!("couldn't compare — {e}"));
                }
            }
        });
    });

    view! {
        <div class="panel">
            <div class="liked-header">
                <h2>"Muse Atlas"</h2>
                <p class="muted small">
                    "A diagnostic map of Muse's own generated output — where it repeatedly \
                     generates, and where the keepers are. Position is a 2D projection of each \
                     piece's structural fingerprint — not a semantic coordinate, just relative \
                     distance. Switch lenses below to reweight which layers of that fingerprint \
                     drive the layout."
                </p>
                <div class="atlas-lens-row">
                    <label for="atlas-lens">"Lens"</label>
                    <select
                        id="atlas-lens"
                        on:change=move |ev| {
                            let v = event_target_value(&ev);
                            lens.set(v);
                            reload();
                        }
                    >
                        {LENSES.iter().map(|(value, label)| {
                            let value = value.to_string();
                            view! {
                                <option value=value.clone() selected=move || lens.get() == value>
                                    {*label}
                                </option>
                            }
                        }).collect_view()}
                    </select>
                    <button type="button" on:click=move |_| reload()>"Refresh"</button>
                </div>
            </div>

            {move || (!status.get().is_empty()).then(|| view! { <p class="muted">{status.get()}</p> })}
            {move || (loaded.get() && status.get().is_empty() && points.get().is_empty()).then(|| view! {
                <p class="muted">"Nothing to map yet — compose or keep a few pieces first."</p>
            })}

            <canvas
                node_ref=canvas_ref
                class="atlas-canvas"
                title="click a point for details"
                on:click=move |ev: leptos::ev::MouseEvent| {
                    let Some(canvas_el) = canvas_ref.get_untracked() else { return };
                    let canvas: HtmlCanvasElement = canvas_el.into();
                    let rect = canvas.get_bounding_client_rect();
                    let px = ev.client_x() as f64 - rect.left();
                    let py = ev.client_y() as f64 - rect.top();
                    let pts = points.get_untracked();
                    if let Some(p) = nearest_point(&pts, px, py, canvas.width() as f64, canvas.height() as f64) {
                        selected.set(Some(p));
                    }
                }
            />

            {move || selected.get().map(|p| view! {
                <div class="atlas-inspector">
                    <h3>{p.title.clone()}</h3>
                    <p class="muted">
                        {format!("{} · {:.1}s{}", p.style, p.duration_secs, if p.kept { " · kept" } else { "" })}
                        " · viewing: " {move || lens_label(&lens.get())}
                    </p>
                    {move || (!compare_status.get().is_empty()).then(|| view! {
                        <p class="muted small">{compare_status.get()}</p>
                    })}
                    {move || comparison.get().map(|c| view! {
                        <div class="atlas-why-nearby">
                            <h4>"Why nearby: " {c.b_title.clone()}</h4>
                            <ul>
                                {c.layers.iter().map(|l| {
                                    let close = l.distance < 0.15;
                                    view! {
                                        <li>
                                            {format!(
                                                "{} {} (distance: {:.2})",
                                                if close { "Similar" } else { "Different" },
                                                l.name,
                                                l.distance,
                                            )}
                                        </li>
                                    }
                                }).collect_view()}
                            </ul>
                            <p class="muted small">
                                {format!("total distance: {:.2}", c.total_distance)}
                            </p>
                        </div>
                    })}
                </div>
            })}
        </div>
    }
}

/// Same layout math `draw_atlas_frame` uses, factored out so click-hit-testing
/// and drawing can never disagree about where a point actually is.
fn layout(points: &[AtlasPoint], w: f64, h: f64) -> Vec<(f64, f64)> {
    if points.is_empty() {
        return Vec::new();
    }
    let (mut x_lo, mut x_hi, mut y_lo, mut y_hi) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for p in points {
        x_lo = x_lo.min(p.x);
        x_hi = x_hi.max(p.x);
        y_lo = y_lo.min(p.y);
        y_hi = y_hi.max(p.y);
    }
    let x_span = (x_hi - x_lo).max(1e-9);
    let y_span = (y_hi - y_lo).max(1e-9);
    let pad = 24.0_f64;
    points
        .iter()
        .map(|p| {
            let nx = (p.x - x_lo) / x_span;
            let ny = (p.y - y_lo) / y_span;
            (
                pad + nx * (w - 2.0 * pad),
                pad + (1.0 - ny) * (h - 2.0 * pad),
            )
        })
        .collect()
}

fn nearest_point(points: &[AtlasPoint], px: f64, py: f64, w: f64, h: f64) -> Option<AtlasPoint> {
    let coords = layout(points, w, h);
    coords
        .iter()
        .zip(points.iter())
        .map(|(&(cx, cy), p)| {
            let d = ((cx - px).powi(2) + (cy - py).powi(2)).sqrt();
            (d, p)
        })
        .filter(|(d, _)| *d < 16.0)
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .map(|(_, p)| p.clone())
}

fn draw_atlas(canvas: HtmlCanvasElement, points: RwSignal<Vec<AtlasPoint>>) {
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
        if canvas.width() != w {
            canvas.set_width(w);
        }
        if canvas.height() != h {
            canvas.set_height(h);
        }
        let pts = points.get_untracked();
        draw_atlas_frame(&ctx, w as f64, h as f64, &pts);

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

fn draw_atlas_frame(ctx: &CanvasRenderingContext2d, w: f64, h: f64, points: &[AtlasPoint]) {
    ctx.clear_rect(0.0, 0.0, w, h);
    if points.is_empty() {
        return;
    }
    let coords = layout(points, w, h);
    for ((cx, cy), p) in coords.iter().zip(points.iter()) {
        let pal = palette::palette_for(&p.style);
        let color = format!("rgb({})", pal.a);
        ctx.set_fill_style_str(&color);
        let radius = if p.kept { 6.0 } else { 4.0 };
        let _ = ctx.begin_path();
        let _ = ctx.arc(*cx, *cy, radius, 0.0, std::f64::consts::TAU);
        ctx.fill();
        if p.kept {
            ctx.set_stroke_style_str("rgba(232,223,210,0.9)");
            ctx.set_line_width(1.5);
            let _ = ctx.begin_path();
            let _ = ctx.arc(*cx, *cy, radius + 2.0, 0.0, std::f64::consts::TAU);
            ctx.stroke();
        }
    }
}

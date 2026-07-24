// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research Mode's Evidence views: motif, sonority/harmony, cadence,
//! orchestration, and structural-activity evidence for the shared current
//! piece — the Leptos port of the vanilla-JS Research view's
//! `renderResearch()`/`renderAnalysisAvailability()` (`studio/muse-studio.js`,
//! landed as part of this branch's series-3 work). Every rendered value
//! traces to a real field on `GET /api/piece/{id}/listen-bundle`'s
//! response; a layer the backend hasn't populated yet renders as an
//! honest "Not emitted" line, never a fabricated placeholder — see each
//! item's [`api::EvidenceBasis`] badge for whether it's directly observed,
//! deterministically reconstructed, or a bounded, disclosed inference.

use leptos::prelude::*;
use leptos::task::spawn_local;
use wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

use crate::api::{
    self, BundleEnvelope, EvidenceBasis, EvidenceStatus, ListenCompositionBundle, MusicalTime,
};
use crate::state::MuseState;

fn fmt_time(t: &MusicalTime) -> String {
    format!("{:.1}s", t.seconds)
}

fn status_label(status: &EvidenceStatus) -> &'static str {
    match status {
        EvidenceStatus::Observed => "Observed",
        EvidenceStatus::Reconstructed => "Reconstructed",
        EvidenceStatus::Inferred => "Inferred",
    }
}

fn status_class(status: &EvidenceStatus) -> &'static str {
    match status {
        EvidenceStatus::Observed => "evidence-observed",
        EvidenceStatus::Reconstructed => "evidence-reconstructed",
        EvidenceStatus::Inferred => "evidence-inferred",
    }
}

/// A compact status pill for one piece of evidence — the status plus, on
/// hover, its source method and any disclosed limitations. This is the
/// mechanism that keeps "Inferred" (a bounded proxy, e.g. resonance) from
/// reading the same as "Observed" (copied straight off the score/recipe).
#[component]
fn EvidenceBadge(basis: EvidenceBasis) -> impl IntoView {
    let title = if basis.limitations.is_empty() {
        basis.source_method.clone()
    } else {
        format!("{} — {}", basis.source_method, basis.limitations.join("; "))
    };
    let confidence = basis
        .confidence
        .map(|c| format!(" ({:.0}%)", c * 100.0))
        .unwrap_or_default();
    view! {
        <span class=format!("evidence-badge {}", status_class(&basis.status)) title=title>
            {format!("{}{confidence}", status_label(&basis.status))}
        </span>
    }
}

fn render_sections(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    if payload.sections.is_empty() {
        return view! { <p class="muted">"Sections: not emitted."</p> }.into_any();
    }
    view! {
        <ul class="evidence-list">
            {payload.sections.iter().map(|s| view! {
                <li>
                    <b>{s.label.clone()}</b>" · "{s.role.clone()}
                    " · "{fmt_time(&s.start)}" – "{fmt_time(&s.end)}
                    " · intensity "{format!("{:.2}", s.intensity)}
                    <span class="badge-note">{s.source_method.clone()}</span>
                </li>
            }).collect_view()}
        </ul>
    }
    .into_any()
}

fn render_motifs(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    if payload.motif_definitions.is_empty() {
        return view! { <p class="muted">"Motifs: not emitted."</p> }.into_any();
    }
    let occurrences_for = |motif_id: &str| {
        payload
            .motif_occurrences
            .iter()
            .filter(|o| o.motif_id == motif_id)
            .cloned()
            .collect::<Vec<_>>()
    };
    view! {
        <ul class="evidence-list">
            {payload.motif_definitions.iter().map(|m| {
                let occs = occurrences_for(&m.id);
                view! {
                    <li>
                        <div>
                            <b>{m.label.clone()}</b>" · degrees "{format!("{:?}", m.degrees)}
                            " "<EvidenceBadge basis=m.basis.clone() />
                        </div>
                        {if occs.is_empty() {
                            view! { <p class="muted indent">"No scored occurrences above threshold."</p> }.into_any()
                        } else {
                            view! {
                                <ul class="evidence-sublist">
                                    {occs.into_iter().map(|o| view! {
                                        <li>
                                            {fmt_time(&o.start)}" – "{fmt_time(&o.end)}
                                            " · "{o.transformation.clone()}
                                            " · similarity "{format!("{:.2}", o.similarity)}
                                            " "<EvidenceBadge basis=o.basis.clone() />
                                        </li>
                                    }).collect_view()}
                                </ul>
                            }.into_any()
                        }}
                    </li>
                }
            }).collect_view()}
        </ul>
    }
    .into_any()
}

fn render_cadences(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    if payload.cadences.is_empty() {
        return view! { <p class="muted">"Cadences: not emitted."</p> }.into_any();
    }
    view! {
        <ul class="evidence-list">
            {payload.cadences.iter().map(|c| view! {
                <li>
                    {fmt_time(&c.at)}" · arrives on "{c.arrival_pitch_name.clone()}
                    " · "{c.voice_role.clone()}
                    " "<EvidenceBadge basis=c.basis.clone() />
                </li>
            }).collect_view()}
        </ul>
    }
    .into_any()
}

fn render_sonority(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    if payload.sonorities.is_empty() {
        return view! { <p class="muted">"Sonority: not emitted."</p> }.into_any();
    }
    view! {
        <ul class="evidence-list">
            {payload.sonorities.iter().map(|s| {
                let degree = match (&s.home_key_degree, &s.home_key_function) {
                    (Some(d), Some(f)) => format!(" · degree {d} ({f})"),
                    (Some(d), None) => format!(" · degree {d}"),
                    _ => String::new(),
                };
                view! {
                    <li>
                        {fmt_time(&s.start)}" – "{fmt_time(&s.end)}
                        " · "{s.pitch_classes.join(" ")}
                        {degree}
                        " "<EvidenceBadge basis=s.basis.clone() />
                    </li>
                }
            }).collect_view()}
        </ul>
    }
    .into_any()
}

fn render_orchestration(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    if payload.orchestration.is_empty() {
        return view! { <p class="muted">"Orchestration: not emitted."</p> }.into_any();
    }
    view! {
        <table class="evidence-table">
            <thead>
                <tr><th>"Region"</th><th>"Voices"</th><th>"Register"</th><th>"Mean velocity"</th><th></th></tr>
            </thead>
            <tbody>
                {payload.orchestration.iter().map(|o| {
                    let register = match (o.register_min_midi, o.register_max_midi) {
                        (Some(lo), Some(hi)) => format!("MIDI {lo}–{hi}"),
                        _ => "—".to_string(),
                    };
                    let voices = o.active_voices.iter()
                        .map(|v| format!("{} ({})", v.voice_role, v.note_count))
                        .collect::<Vec<_>>()
                        .join(", ");
                    view! {
                        <tr>
                            <td>{fmt_time(&o.start)}" – "{fmt_time(&o.end)}</td>
                            <td>{voices}</td>
                            <td>{register}</td>
                            <td>{format!("{:.2}", o.mean_velocity)}</td>
                            <td><EvidenceBadge basis=o.basis.clone() /></td>
                        </tr>
                    }
                }).collect_view()}
            </tbody>
        </table>
    }
    .into_any()
}

/// A small static sparkline for the (optional) resonance curve — energy/
/// density/motion over time. Draws once per mount/data-change; unlike
/// Atlas or the piano-roll this isn't synced to live playback, so no rAF
/// loop is needed.
#[component]
fn ResonanceChart(samples: Vec<api::ResonanceCurve>, basis: EvidenceBasis) -> impl IntoView {
    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();
    let curve = samples.into_iter().next();

    Effect::new(move |_| {
        let Some(canvas) = canvas_ref.get() else {
            return;
        };
        let Some(curve) = &curve else { return };
        let Ok(Some(ctx)) = canvas.get_context("2d") else {
            return;
        };
        let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() else {
            return;
        };
        let w = canvas.client_width().max(1) as u32;
        let h = canvas.client_height().max(1) as u32;
        canvas.set_width(w);
        canvas.set_height(h);
        let (w, h) = (w as f64, h as f64);
        ctx.clear_rect(0.0, 0.0, w, h);
        if curve.samples.is_empty() {
            return;
        }
        let duration = curve
            .samples
            .last()
            .map(|s| s.at.seconds.max(0.001))
            .unwrap_or(1.0);
        let plot = |field: fn(&api::ResonanceCurve, usize) -> f64, color: &str| {
            ctx.begin_path();
            ctx.set_stroke_style_str(color);
            ctx.set_line_width(1.5);
            for (i, s) in curve.samples.iter().enumerate() {
                let x = (s.at.seconds / duration) * w;
                let y = h - field(curve, i).clamp(0.0, 1.0) * h;
                if i == 0 {
                    ctx.move_to(x, y);
                } else {
                    ctx.line_to(x, y);
                }
            }
            ctx.stroke();
        };
        plot(|c, i| c.samples[i].energy as f64, "rgb(215,164,95)");
        plot(|c, i| c.samples[i].density as f64, "rgb(102,174,190)");
        plot(|c, i| c.samples[i].motion as f64, "rgb(125,100,200)");
    });

    view! {
        <div class="resonance-chart">
            <canvas node_ref=canvas_ref style="width:100%;height:64px;display:block;"></canvas>
            <div class="chart-legend">
                <span style="color:rgb(215,164,95)">"■ energy"</span>
                " "
                <span style="color:rgb(102,174,190)">"■ density"</span>
                " "
                <span style="color:rgb(125,100,200)">"■ motion"</span>
                " "<EvidenceBadge basis=basis />
            </div>
        </div>
    }
}

fn render_resonance(payload: &ListenCompositionBundle) -> impl IntoView + use<> {
    match &payload.resonance {
        None => view! { <p class="muted">"Structural activity: not emitted."</p> }.into_any(),
        Some(curve) => {
            view! { <ResonanceChart samples=vec![curve.clone()] basis=curve.basis.clone() /> }
                .into_any()
        }
    }
}

#[component]
pub fn EvidenceView(muse: MuseState) -> impl IntoView {
    let bundle = RwSignal::new(None::<BundleEnvelope<ListenCompositionBundle>>);
    let status = RwSignal::new(String::new());

    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            bundle.set(None);
            return;
        };
        let id = c.id;
        status.set("loading evidence…".to_string());
        spawn_local(async move {
            match api::fetch_listen_bundle(api::DEFAULT_BACKEND, id).await {
                Ok(b) => {
                    bundle.set(Some(b));
                    status.set(String::new());
                }
                Err(e) => {
                    bundle.set(None);
                    status.set(format!("couldn't load evidence — {e}"));
                }
            }
        });
    });

    view! {
        <div class="evidence-view">
            {move || {
                let s = status.get();
                (!s.is_empty()).then(|| view! { <p class="muted">{s}</p> })
            }}
            {move || bundle.get().map(|env| {
                let payload = env.payload.clone();
                view! {
                    <div>
                        {(!env.warnings.is_empty()).then(|| view! {
                            <ul class="evidence-warnings">
                                {env.warnings.iter().map(|w| view! { <li>{w.message.clone()}</li> }).collect_view()}
                            </ul>
                        })}
                        <h3>"Sections"</h3>
                        {render_sections(&payload)}
                        <h3>"Motifs"</h3>
                        {render_motifs(&payload)}
                        <h3>"Cadences"</h3>
                        {render_cadences(&payload)}
                        <h3>"Sonority"</h3>
                        {render_sonority(&payload)}
                        <h3>"Orchestration"</h3>
                        {render_orchestration(&payload)}
                        <h3>"Structural activity"</h3>
                        {render_resonance(&payload)}
                    </div>
                }
            })}
        </div>
    }
}

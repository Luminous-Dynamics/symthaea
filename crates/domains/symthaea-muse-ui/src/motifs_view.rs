// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research Mode's Motifs view: the piece's discrete motif-return structure
//! — one colored, proportionally-sized block per section along a timeline,
//! labeled by role ("A", "B", "A (return)", "C") and key — from
//! `GET /api/motifs/{id}`.
//!
//! `has_structure: false` is the honest, expected answer for the 6+ form
//! kinds that bypass the period/`Form` pipeline entirely (Fugue, ProgSuite,
//! Sonata, Renaissance, Opera, the 3 ground forms) — this view renders that
//! as an explanatory sentence, never as an empty/broken timeline. See
//! `motifs_summary()`'s doc comment in `muse_studio.rs` for why those form
//! kinds have no `Form` to report.

use leptos::prelude::*;
use leptos::task::spawn_local;

use crate::api::{self, MotifsSummary, SectionInfo};
use crate::state::MuseState;

/// Stable color per section role, independent of key or bar position — so
/// a returning "A" section visually reads as the same material, matching
/// the whole point of a motif-return view.
fn role_color(role: &str) -> &'static str {
    match role {
        "A" | "A (return)" => "#d9a05b",
        "B" => "#6f9bbf",
        "C" => "#8fbf6f",
        _ => "#b08fbf",
    }
}

#[component]
pub fn MotifsView(muse: MuseState) -> impl IntoView {
    let summary = RwSignal::new(None::<MotifsSummary>);
    let load_status = RwSignal::new(String::new());

    Effect::new(move |_| {
        let Some(c) = muse.current.get() else {
            summary.set(None);
            return;
        };
        let id = c.id;
        load_status.set("loading motifs…".to_string());
        spawn_local(async move {
            match api::fetch_motifs(api::DEFAULT_BACKEND, id).await {
                Ok(s) => {
                    summary.set(Some(s));
                    load_status.set(String::new());
                }
                Err(e) => load_status.set(format!("motifs unavailable: {e}")),
            }
        });
    });

    view! {
        <div class="motifs-view">
            {move || match summary.get() {
                None => view! { <p class="muted small">{move || load_status.get()}</p> }.into_any(),
                Some(s) if !s.has_structure => view! {
                    <p class="muted">
                        "This piece's form doesn't have discrete motif-return structure \
                         (its section pipeline is bypassed by design — see Fugue/Sonata/\
                         ground-form composers, which build their own internal shape)."
                    </p>
                }.into_any(),
                Some(s) => {
                    let total = s.sections.iter().map(|sec| sec.end_seconds).fold(0.0_f64, f64::max).max(1.0);
                    view! {
                        <div class="motifs-timeline">
                            {s.sections.iter().map(|sec: &SectionInfo| {
                                let color = role_color(&sec.role);
                                let left_pct = (sec.start_seconds / total * 100.0).clamp(0.0, 100.0);
                                let width_pct = ((sec.end_seconds - sec.start_seconds) / total * 100.0).clamp(1.0, 100.0 - left_pct);
                                let key = format!("{} {}", sec.key_tonic, sec.key_tonality);
                                let bars = format!("bars {}\u{2013}{}", sec.start_bar, sec.end_bar);
                                view! {
                                    <div
                                        class="motifs-section"
                                        style=format!("left:{left_pct:.1}%;width:{width_pct:.1}%;background:{color};")
                                        title=format!("{} \u{00b7} {key} \u{00b7} {bars}", sec.role)
                                    >
                                        <span class="motifs-section-role">{sec.role.clone()}</span>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                        <div class="motifs-legend">
                            {{
                                let mut seen = std::collections::HashSet::new();
                                s.sections.iter()
                                    .filter(|sec| seen.insert(sec.role.clone()))
                                    .map(|sec| {
                                        view! {
                                            <span class="legend-item">
                                                <b style=format!("color:{}", role_color(&sec.role))>"■"</b>
                                                {format!(" {}", sec.role)}
                                            </span>
                                        }
                                    }).collect_view()
                            }}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

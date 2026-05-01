// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Praxis domain — learning dashboard with interactive SVG explorations.

use leptos::prelude::*;
use sensorium_games::math::unit_circle::UnitCircleExplorer;
use sensorium_games::physics::projectile::ProjectileExplorer;
use sensorium_games::progress::provide_progress_store;
use sensorium_viz::{bar_chart::Bar, BarChart};

#[derive(Clone, Copy, PartialEq)]
enum EduView {
    Dashboard,
    Games,
    Review,
    SkillMap,
    Credentials,
}

#[derive(Clone)]
struct EduState {
    level: RwSignal<u32>,
    xp: RwSignal<u64>,
    streak: RwSignal<u32>,
    mastery_pct: RwSignal<u32>,
    due_cards: RwSignal<u32>,
}

impl EduState {
    fn new() -> Self {
        Self {
            level: RwSignal::new(7),
            xp: RwSignal::new(4280),
            streak: RwSignal::new(12),
            mastery_pct: RwSignal::new(68),
            due_cards: RwSignal::new(18),
        }
    }
}

#[component]
pub fn PraxisOverview() -> impl IntoView {
    let (active_view, set_active_view) = signal(EduView::Dashboard);
    let (_progress_read, _progress_write) = provide_progress_store();
    let state = EduState::new();
    provide_context(state.clone());

    view! {
        <div class="praxis-content">
            <div class="governance-nav">
                <button class=move || if active_view.get() == EduView::Dashboard { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(EduView::Dashboard)>"Dashboard"</button>
                <button class=move || if active_view.get() == EduView::Games { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(EduView::Games)>"Explorations"</button>
                <button class=move || if active_view.get() == EduView::Review { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(EduView::Review)>"Review"</button>
                <button class=move || if active_view.get() == EduView::SkillMap { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(EduView::SkillMap)>"Skill Map"</button>
                <button class=move || if active_view.get() == EduView::Credentials { "domain-nav-btn active" } else { "domain-nav-btn" }
                    on:click=move |_| set_active_view.set(EduView::Credentials)>"Credentials"</button>
            </div>

            <Show when=move || active_view.get() == EduView::Dashboard>
                <div class="commons-stats-grid">
                    <div class="thought-card">
                        <div class="thought-type" style="color: #2563EB">"LEVEL"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.level.get().to_string()}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">{move || format!("{} XP total", state.xp.get())}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #60A5FA">"STREAK"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{}d", state.streak.get())}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">{move || if state.streak.get() >= 7 { "1.5x bonus active" } else { "Keep going!" }}</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #f59e0b">"DUE"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || state.due_cards.get().to_string()}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"Cards to review"</p>
                    </div>
                    <div class="thought-card">
                        <div class="thought-type" style="color: #22c55e">"MASTERY"</div>
                        <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">{move || format!("{}%", state.mastery_pct.get())}</p>
                        <p style="font-size: 0.7rem; color: var(--text-muted)">"CAPS curriculum"</p>
                    </div>
                </div>
                <h3 class="section-title">"Top Skills"</h3>
                <BarChart data=vec![
                    Bar { label: "Multiply".into(), value: 85.0, color: "#2563EB".into() },
                    Bar { label: "Fractions".into(), value: 72.0, color: "#60A5FA".into() },
                    Bar { label: "Algebra".into(), value: 68.0, color: "#60A5FA".into() },
                    Bar { label: "Geometry".into(), value: 55.0, color: "#93c5fd".into() },
                    Bar { label: "Stats".into(), value: 42.0, color: "#93c5fd".into() },
                ] width=350.0 height=180.0 />
            </Show>

            <Show when=move || active_view.get() == EduView::Games>
                <GameSelector />
            </Show>

            <Show when=move || active_view.get() == EduView::Review>
                <div class="thought-card">
                    <h3 class="section-title">"Spaced Repetition Review"</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary)">"18 cards due. SM-2 schedules reviews at optimal intervals."</p>
                </div>
            </Show>
            <Show when=move || active_view.get() == EduView::SkillMap>
                <div class="thought-card">
                    <h3 class="section-title">"Skill Constellation"</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary)">"100+ CAPS topics as a knowledge DAG."</p>
                </div>
            </Show>
            <Show when=move || active_view.get() == EduView::Credentials>
                <div class="thought-card">
                    <h3 class="section-title">"Verifiable Credentials"</h3>
                    <p style="font-size: 0.85rem; color: var(--text-secondary)">"W3C VCs for completed courses."</p>
                </div>
            </Show>
        </div>
    }
}

#[component]
fn GameSelector() -> impl IntoView {
    let (active_game, set_active_game) = signal(None::<String>);

    let games = vec![
        (
            "unit-circle",
            "Unit Circle",
            "Explore sin, cos, tan",
            "#2563EB",
        ),
        (
            "projectile",
            "Projectile Motion",
            "h = v\u{2080}t - \u{00bd}gt\u{00b2}",
            "#22c55e",
        ),
        (
            "parabola",
            "Parabola Explorer",
            "y = a(x-p)\u{00b2} + q",
            "#8b5cf6",
        ),
        (
            "circuits",
            "Electric Circuits",
            "Ohm's law, series/parallel",
            "#f59e0b",
        ),
        ("tangent", "Tangent Lines", "Derivative as slope", "#ec4899"),
        (
            "stats",
            "Statistics",
            "Mean, median, distributions",
            "#06b6d4",
        ),
        (
            "analytical",
            "Analytical Geometry",
            "Distance, midpoint",
            "#059669",
        ),
    ];

    view! {
        <div>
            {move || match active_game.get() {
                Some(ref game) => {
                    let back = move |_| set_active_game.set(None);
                    match game.as_str() {
                        "unit-circle" => view! {
                            <div>
                                <button class="domain-nav-btn" style="margin-bottom: 0.75rem;" on:click=back>"\u{2190} Back"</button>
                                <UnitCircleExplorer node_id="CAPS.Mathematics.Gr12.P2.TRIG".to_string() />
                            </div>
                        }.into_any(),
                        "projectile" => view! {
                            <div>
                                <button class="domain-nav-btn" style="margin-bottom: 0.75rem;" on:click=back>"\u{2190} Back"</button>
                                <ProjectileExplorer node_id="CAPS.PhysicalSciences.Gr12.P1.MECH1".to_string() />
                            </div>
                        }.into_any(),
                        _ => view! {
                            <div class="thought-card">
                                <p style="color: var(--text-muted)">"This exploration will be available soon."</p>
                                <button class="domain-nav-btn" style="margin-top: 0.5rem;" on:click=back>"\u{2190} Back"</button>
                            </div>
                        }.into_any(),
                    }
                },
                None => view! {
                    <div>
                        <h3 class="section-title">"Interactive Explorations"</h3>
                        <p style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 1rem;">
                            "Reactive SVG diagrams \u{2014} sliders, math, and shapes. Not GPU games."
                        </p>
                        <div class="proposals-list">
                            {games.iter().map(|(id, name, desc, color)| {
                                let id = id.to_string();
                                let name = *name;
                                let desc = *desc;
                                let color = *color;
                                let id_click = id.clone();
                                view! {
                                    <div class="proposal-card" style=format!("border-left: 3px solid {color}; cursor: pointer;")
                                        on:click=move |_| set_active_game.set(Some(id_click.clone()))>
                                        <h3 class="proposal-title" style=format!("color: {color}")>{name}</h3>
                                        <p style="font-size: 0.8rem; color: var(--text-secondary)">{desc}</p>
                                    </div>
                                }
                            }).collect::<Vec<_>>()}
                        </div>
                    </div>
                }.into_any(),
            }}
        </div>
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

use crate::curriculum::{curriculum_graph, CurriculumGraph, CurriculumNode, ProgressStatus, ProgressStore, use_progress};

#[derive(Clone, Copy, PartialEq)]
enum StewardshipTrack {
    Mycelix,
    Symthaea,
}

impl StewardshipTrack {
    fn title(self) -> &'static str {
        match self {
            StewardshipTrack::Mycelix => "Mycelix",
            StewardshipTrack::Symthaea => "Symthaea",
        }
    }

    fn subtitle(self) -> &'static str {
        match self {
            StewardshipTrack::Mycelix => "Decentralized civic infrastructure and governance.",
            StewardshipTrack::Symthaea => "Consciousness computing, moral algebra, and cognitive architecture.",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            StewardshipTrack::Mycelix => "🕸️",
            StewardshipTrack::Symthaea => "🧠",
        }
    }

    fn css_class(self) -> &'static str {
        match self {
            StewardshipTrack::Mycelix => "stewardship-track-mycelix",
            StewardshipTrack::Symthaea => "stewardship-track-symthaea",
        }
    }
}

fn course_number(id: &str) -> u32 {
    id.split('-').nth(1).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0)
}

fn track_nodes(graph: &'static CurriculumGraph, prefix: &str) -> Vec<&'static CurriculumNode> {
    let mut nodes: Vec<&CurriculumNode> = graph.nodes.iter().filter(|n| n.id.starts_with(prefix)).collect();
    nodes.sort_by(|a, b| {
        course_number(&a.id)
            .cmp(&course_number(&b.id))
            .then_with(|| a.id.cmp(&b.id))
    });
    nodes
}

fn group_by_level(nodes: &[&'static CurriculumNode]) -> Vec<(&'static str, Vec<&'static CurriculumNode>)> {
    let mut undergraduate = Vec::new();
    let mut graduate = Vec::new();
    let mut doctoral = Vec::new();
    let mut other = Vec::new();

    for &node in nodes {
        match node.grade_levels.first().map(|s| s.as_str()) {
            Some("Undergraduate") => undergraduate.push(node),
            Some("Graduate") => graduate.push(node),
            Some("Doctoral") => doctoral.push(node),
            _ => other.push(node),
        }
    }

    let mut grouped = Vec::new();
    if !undergraduate.is_empty() {
        grouped.push(("Undergraduate", undergraduate));
    }
    if !graduate.is_empty() {
        grouped.push(("Graduate", graduate));
    }
    if !doctoral.is_empty() {
        grouped.push(("Doctoral", doctoral));
    }
    if !other.is_empty() {
        grouped.push(("Other", other));
    }
    grouped
}

fn progress_counts(nodes: &[&CurriculumNode], progress: &ProgressStore) -> (usize, usize, usize) {
    let mastered = nodes
        .iter()
        .filter(|n| progress.get(&n.id).status == ProgressStatus::Mastered)
        .count();
    let studying = nodes
        .iter()
        .filter(|n| progress.get(&n.id).status == ProgressStatus::Studying)
        .count();
    let remaining = nodes.len().saturating_sub(mastered + studying);
    (mastered, studying, remaining)
}

#[component]
pub fn StewardshipPage() -> impl IntoView {
    let progress = use_progress();
    let graph = curriculum_graph();
    let mycelix_nodes = track_nodes(graph, "MYC-");
    let symthaea_nodes = track_nodes(graph, "SYM-");
    let (active_track, set_active_track) = signal(StewardshipTrack::Mycelix);

    view! {
        <div class="praxis-skill-map">
            <a href="/" class="refuge-back-link">"\u{2190} Home"</a>
            <h1 style="font-size: 1.6rem; margin: 1rem 0 0.5rem">"Stewardship"</h1>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6; max-width: 820px">
                "The operational manual for the Mycelix mesh and the Symthaea cognitive loop. Progress unlocks through"
                " prerequisite mastery embedded in the graph."
            </p>
            <div class="stewardship-portfolio-link">
                <a href="/profile" class="portfolio-link">"Open Stewardship Portfolio \u{2192}"</a>
                <span>"Publish verified credentials to the professional graph."</span>
            </div>

            <div class="stewardship-tabs" role="tablist" aria-label="Stewardship tracks">
                <button
                    class=move || format!("stewardship-tab mycelix {}", if active_track.get() == StewardshipTrack::Mycelix { "active" } else { "" })
                    role="tab"
                    aria-selected=move || active_track.get() == StewardshipTrack::Mycelix
                    on:click=move |_| set_active_track.set(StewardshipTrack::Mycelix)
                >
                    <span class="track-icon">"🕸️"</span>
                    "Mycelix"
                </button>
                <button
                    class=move || format!("stewardship-tab symthaea {}", if active_track.get() == StewardshipTrack::Symthaea { "active" } else { "" })
                    role="tab"
                    aria-selected=move || active_track.get() == StewardshipTrack::Symthaea
                    on:click=move |_| set_active_track.set(StewardshipTrack::Symthaea)
                >
                    <span class="track-icon">"🧠"</span>
                    "Symthaea"
                </button>
            </div>

            {move || {
                match active_track.get() {
                    StewardshipTrack::Mycelix => {
                        render_track(graph, StewardshipTrack::Mycelix, mycelix_nodes.clone(), progress).into_any()
                    }
                    StewardshipTrack::Symthaea => {
                        render_track(graph, StewardshipTrack::Symthaea, symthaea_nodes.clone(), progress).into_any()
                    }
                }
            }}
        </div>
    }
}

fn render_track(
    graph: &'static CurriculumGraph,
    track: StewardshipTrack,
    nodes: Vec<&'static CurriculumNode>,
    progress: ReadSignal<ProgressStore>,
) -> impl IntoView {
    let grouped = group_by_level(&nodes);

    view! {
        <section class=format!("stewardship-section {}", track.css_class())>
            <div class="stewardship-track-header">
                <span class="track-icon">{track.icon()}</span>
                <div>
                    <h2>{track.title()}</h2>
                    <p>{track.subtitle()}</p>
                </div>
            </div>

            {move || {
                let p = progress.get();
                let (mastered, studying, remaining) = progress_counts(&nodes, &p);

                let sections: Vec<_> = grouped.iter().map(|(level, level_nodes)| {
                    let (lvl_mastered, lvl_studying, lvl_remaining) = progress_counts(level_nodes, &p);
                    let items: Vec<_> = level_nodes.iter().map(|n| {
                        let status = p.get(&n.id).status;
                        let requires = graph.requires_for(&n.id);
                        let missing: Vec<&str> = requires.iter()
                            .copied()
                            .filter(|pid| p.get(pid).status != ProgressStatus::Mastered)
                            .collect();
                        let unlocked = missing.is_empty();
                        let prereq_titles: Vec<String> = requires.iter()
                            .filter_map(|pid| graph.node(pid).map(|node| node.title.clone()))
                            .collect();
                        let missing_titles: Vec<String> = missing.iter()
                            .filter_map(|pid| graph.node(pid).map(|node| node.title.clone()))
                            .collect();

                        let (icon, color) = match status {
                            ProgressStatus::Mastered => ("\u{1F333}", "var(--mastery-green)"),
                            ProgressStatus::Studying => ("\u{1F33F}", "var(--mastery-yellow)"),
                            ProgressStatus::NotStarted => ("\u{1F331}", "var(--text-tertiary)"),
                        };
                        let href = format!("/study/{}", n.id);
                        let border_color = if unlocked { color } else { "var(--border)" };

                        view! {
                            <div class="stewardship-item">
                                {if unlocked {
                                    view! {
                                        <a href=href class="pathway-node" style=format!("border-left-color: {}", border_color)>
                                            <span style="font-size: 1rem">{icon}</span>
                                            <span style="flex: 1; font-size: 0.9rem; font-weight: 600">{n.title.clone()}</span>
                                            <span style="font-size: 0.7rem; color: var(--text-tertiary)">{status.label()}</span>
                                        </a>
                                    }.into_any()
                                } else {
                                    view! {
                                        <div class="pathway-node locked" style=format!("border-left-color: {}", border_color)>
                                            <span style="font-size: 1rem">"\u{1F512}"</span>
                                            <span style="flex: 1; font-size: 0.9rem; font-weight: 600">{n.title.clone()}</span>
                                            <span style="font-size: 0.7rem; color: var(--text-tertiary)">"Locked"</span>
                                        </div>
                                    }.into_any()
                                }}

                                <div class="stewardship-description">{n.description.clone()}</div>

                                {if requires.is_empty() {
                                    view! { <span></span> }.into_any()
                                } else if missing_titles.is_empty() {
                                    view! {
                                        <div class="stewardship-prereqs">
                                            "Prerequisites complete: " {prereq_titles.join(", ")}
                                        </div>
                                    }.into_any()
                                } else {
                                    view! {
                                        <div class="stewardship-prereqs stewardship-prereqs-missing">
                                            "Requires: " {missing_titles.join(", ")}
                                        </div>
                                    }.into_any()
                                }}
                            </div>
                        }
                    }).collect();

                    view! {
                        <div class="stewardship-level">
                            <div class="stewardship-level-header">
                                <h3>{level.to_string()}</h3>
                                <span class="stewardship-level-meta">
                                    {lvl_mastered}" mastered \u{00B7} "{lvl_studying}" in progress \u{00B7} "{lvl_remaining}" remaining"
                                </span>
                            </div>
                            <div class="stewardship-track-list">{items}</div>
                        </div>
                    }
                }).collect();

                view! {
                    <div class="stewardship-progress">
                        {mastered}" mastered \u{00B7} "{studying}" in progress \u{00B7} "{remaining}" remaining"
                    </div>
                    <div class="stewardship-levels">{sections}</div>
                }.into_any()
            }}
        </section>
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Career Pathways page — linear progression views for post-secondary subjects.
//!
//! Shows career paths as vertical progressions from fundamentals to expert,
//! using the existing prerequisite graph. Same data as the constellation,
//! different lens — focused on career direction rather than exploration.

use leptos::prelude::*;

use crate::curriculum::{curriculum_graph, use_progress, ProgressStatus};

/// All subjects that have post-secondary nodes, with display names and descriptions.
const PATHWAYS: &[(&str, &str, &str)] = &[
    ("Cybersecurity", "Cybersecurity Career", "From security fundamentals to penetration testing and research"),
    ("Computer Science", "Computer Science", "ACM CS2013 Body of Knowledge — algorithms, systems, AI"),
    ("Financial Literacy", "Financial Literacy", "Earning, saving, investing, and protecting your money"),
    ("Critical Thinking", "Critical Thinking", "Arguments, evidence, biases, and decision-making"),
    ("Digital Literacy", "Digital Literacy", "Information, safety, creation, and citizenship online"),
    ("Health", "Health Literacy", "Body, nutrition, mental health, and first aid"),
    ("Emotional Intelligence", "Emotional Intelligence", "Self-awareness, empathy, and relationship skills"),
    ("Communication", "Communication", "Writing, speaking, listening, and persuasion"),
    ("Systems Thinking", "Systems Thinking", "Feedback loops, emergence, and leverage points"),
    ("Sustainability", "Sustainability", "Climate, resources, and environmental justice"),
    ("Learning Science", "Learning How to Learn", "Metacognition, spaced repetition, and deliberate practice"),
    ("Statistics", "Statistics & Data", "Probability, inference, bias, and visualisation"),
    ("Civic Literacy", "Civic Literacy", "Government, rights, voting, and engagement"),
    ("Philosophy", "Philosophy", "Logic, ethics, epistemology, and metaphysics"),
    ("Consciousness Computing", "Consciousness Computing", "HDC, IIT, active inference, and embodiment"),
    ("Decentralized Systems", "Decentralized Systems", "Holochain, Mycelix governance, and identity"),
    ("Economics", "Economics", "Micro, macro, behavioral, and policy"),
    ("Mathematics", "University Mathematics", "Calculus, linear algebra, differential equations, and beyond"),
    ("Physics", "University Physics", "Classical mechanics, E&M, quantum, and thermodynamics"),
    // Trades
    ("Electrical Trade", "Electrical Trade", "Safety, domestic wiring, three-phase, solar PV, wireman's licence"),
    ("Plumbing Trade", "Plumbing Trade", "Water supply, drainage, solar geysers"),
    ("Welding Trade", "Welding Trade", "Arc, MIG, TIG — from basics to aluminium"),
    ("Automotive Trade", "Automotive Trade", "Engines, electrical, brakes, diagnostics"),
    ("Construction Trade", "Construction Trade", "Foundations, brickwork, roofing, plan reading"),
    ("Agriculture", "Agriculture", "Crop production, small-scale farming, food gardens"),
    ("IT Support", "IT Support", "Hardware, networking, help desk"),
    ("Hospitality", "Hospitality", "Food safety, restaurant service, management"),
    // Legal
    ("Legal Awareness", "Legal Awareness", "Constitutional law, employment law, consumer rights, tax, data privacy, business registration"),
];

/// Returns a CSS class suffix for the pathway category based on subject name.
fn pathway_category_class(subject: &str) -> &'static str {
    match subject {
        "Cybersecurity" | "Computer Science" | "Consciousness Computing"
        | "Decentralized Systems" | "IT Support" => "pathway-card-tech",

        "Critical Thinking" | "Philosophy" | "Learning Science" | "Statistics"
        | "Economics" | "Mathematics" | "Physics" => "pathway-card-academic",

        "Financial Literacy" | "Health" | "Emotional Intelligence" | "Communication"
        | "Civic Literacy" | "Digital Literacy" | "Legal Awareness"
        | "Hospitality" => "pathway-card-life",

        "Sustainability" | "Agriculture" | "Systems Thinking" => "pathway-card-sustainability",

        // Trades default to tech
        "Electrical Trade" | "Plumbing Trade" | "Welding Trade"
        | "Automotive Trade" | "Construction Trade" => "pathway-card-tech",

        _ => "",
    }
}

#[component]
pub fn PathwaysPage() -> impl IntoView {
    let (selected, set_selected) = signal::<Option<String>>(None);
    let progress = use_progress();

    view! {
        <div class="praxis-skill-map">
            <a href="/" class="refuge-back-link">"\u{2190} Home"</a>
            <h1 style="font-size: 1.5rem; margin: 1rem 0 0.5rem">"Career Pathways"</h1>
            <p style="color: var(--text-secondary); margin-bottom: 1.5rem; line-height: 1.6">
                "Explore learning paths beyond school. Each pathway shows a progression from fundamentals to expertise."
            </p>

            {move || match selected.get() {
                None => {
                    // Show pathway cards
                    view! {
                        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 0.75rem">
                            {PATHWAYS.iter().map(|(subject, display_name, desc)| {
                                let graph = curriculum_graph();
                                let p = progress.get();
                                let subj = subject.to_string();
                                let count = graph.nodes.iter()
                                    .filter(|n| n.subject_area == subj && n.grade_levels.iter().any(|g| !g.starts_with("Grade")))
                                    .count();

                                if count == 0 { return view! { <span></span> }.into_any(); }

                                let mastered = graph.nodes.iter()
                                    .filter(|n| n.subject_area == subj && n.grade_levels.iter().any(|g| !g.starts_with("Grade")))
                                    .filter(|n| p.get(&n.id).status == ProgressStatus::Mastered)
                                    .count();

                                let subj_click = subject.to_string();
                                let display = display_name.to_string();
                                let description = desc.to_string();

                                let cat_class = pathway_category_class(subject);
                                view! {
                                    <button
                                        class=format!("pathway-card {}", cat_class)
                                        on:click=move |_| set_selected.set(Some(subj_click.clone()))
                                    >
                                        <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem">{display}</div>
                                        <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem; line-height: 1.4">{description}</div>
                                        <div style="font-size: 0.7rem; color: var(--text-tertiary)">
                                            {count}" topics"
                                            {if mastered > 0 {
                                                format!(" \u{00B7} {} mastered", mastered)
                                            } else {
                                                String::new()
                                            }}
                                        </div>
                                    </button>
                                }.into_any()
                            }).collect::<Vec<_>>()}
                        </div>
                    }.into_any()
                }

                Some(ref subject) => {
                    let graph = curriculum_graph();
                    let p = progress.get();
                    let subj = subject.clone();

                    let display_name = PATHWAYS.iter()
                        .find(|(s, _, _)| *s == subj)
                        .map(|(_, d, _)| *d)
                        .unwrap_or(&subj);

                    // Group nodes by level
                    let mut undergrad = Vec::new();
                    let mut graduate = Vec::new();
                    let mut doctoral = Vec::new();
                    let mut adult = Vec::new();

                    for n in &graph.nodes {
                        if n.subject_area != subj { continue; }
                        let gl = n.grade_levels.first().map(|s| s.as_str()).unwrap_or("");
                        match gl {
                            "Undergraduate" => undergrad.push(n),
                            "Graduate" => graduate.push(n),
                            "Doctoral" => doctoral.push(n),
                            "Adult" => adult.push(n),
                            _ => {}
                        }
                    }

                    let subj_back = subject.clone();

                    view! {
                        <button
                            class="refuge-back-link"
                            style="background: none; border: none; cursor: pointer; font-family: inherit"
                            on:click=move |_| set_selected.set(None)
                        >"\u{2190} All Pathways"</button>

                        <h2 style="font-size: 1.25rem; margin: 0.75rem 0">{display_name}</h2>

                        // Render each level
                        {render_level("Life Skills", &adult, &p)}
                        {render_level("Undergraduate", &undergrad, &p)}
                        {render_level("Graduate", &graduate, &p)}
                        {render_level("Doctoral", &doctoral, &p)}
                    }.into_any()
                }
            }}
        </div>
    }
}

fn render_level(label: &'static str, nodes: &[&crate::curriculum::CurriculumNode], progress: &crate::curriculum::ProgressStore) -> impl IntoView {
    if nodes.is_empty() {
        return view! { <span></span> }.into_any();
    }

    let items: Vec<_> = nodes.iter().map(|n| {
        let status = progress.get(&n.id).status;
        let href = format!("/study/{}", n.id);
        let title = n.title.clone();
        let (icon, color) = match status {
            ProgressStatus::Mastered => ("\u{1F333}", "var(--mastery-green)"),
            ProgressStatus::Studying => ("\u{1F33F}", "var(--mastery-yellow)"),
            ProgressStatus::NotStarted => ("\u{1F331}", "var(--text-tertiary)"),
        };

        view! {
            <a href=href class="pathway-node" style=format!("border-left-color: {}", color)>
                <span style="font-size: 1rem">{icon}</span>
                <span style="flex: 1; font-size: 0.85rem">{title}</span>
                <span style="font-size: 0.7rem; color: var(--text-tertiary)">{status.label()}</span>
            </a>
        }
    }).collect();

    view! {
        <div style="margin-bottom: 1.5rem">
            <h3 style="font-size: 0.8rem; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem">{label}</h3>
            <div style="display: flex; flex-direction: column; gap: 2px">
                {items}
            </div>
        </div>
    }.into_any()
}

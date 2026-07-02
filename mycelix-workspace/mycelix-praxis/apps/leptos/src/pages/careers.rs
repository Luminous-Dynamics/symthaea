// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Professional Path Visualizer — Dynamic Root-to-Fruit Career Mapping.
//! Calculates match percentages based on actual curriculum graph data.

use leptos::prelude::*;
use crate::curriculum::{curriculum_graph, use_progress, ProgressStatus, CurriculumNode};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetabolicSector {
    All,
    DigitalMesh,
    HabitatEngineering,
    BioregionalVitality,
    SovereignGovernance,
    IndustrialMetabolism,
}

impl MetabolicSector {
    pub fn label(&self) -> &'static str {
        match self {
            MetabolicSector::All => "All Sectors",
            MetabolicSector::DigitalMesh => "Digital Mesh",
            MetabolicSector::HabitatEngineering => "Habitat Engineering",
            MetabolicSector::BioregionalVitality => "Bioregional Vitality",
            MetabolicSector::SovereignGovernance => "Sovereign Governance",
            MetabolicSector::IndustrialMetabolism => "Industrial Metabolism",
        }
    }
    
    pub fn all() -> &'static [MetabolicSector] {
        &[
            MetabolicSector::All,
            MetabolicSector::DigitalMesh,
            MetabolicSector::HabitatEngineering,
            MetabolicSector::BioregionalVitality,
            MetabolicSector::SovereignGovernance,
            MetabolicSector::IndustrialMetabolism,
        ]
    }
}

#[component]
pub fn CareerPathPage() -> impl IntoView {
    let progress = use_progress();
    let (selected_sector, set_selected_sector) = signal(MetabolicSector::All);
    let (selected_domain, set_selected_domain) = signal("All".to_string());
    let (selected_difficulty, set_selected_difficulty) = signal("All".to_string());
    let (search_text, set_search_text) = signal("".to_string());
    
    // Derived: Get all unique domains with industry mappings
    let domains = Memo::new(move |_| {
        let graph = curriculum_graph();
        let mut d: Vec<String> = graph.nodes.iter()
            .filter(|n| !n.industry_mappings.is_empty())
            .map(|n| n.domain.clone())
            .collect();
        d.sort();
        d.dedup();
        let mut all = vec!["All".to_string()];
        all.extend(d);
        all
    });

    // Derived: Find all nodes that represent professional certifications (filtered by sector, domain, search, and difficulty)
    let certifications = Memo::new(move |_| {
        let graph = curriculum_graph();
        let p = progress.get();
        let sector_filter = selected_sector.get();
        let domain_filter = selected_domain.get();
        let diff_filter = selected_difficulty.get();
        let search = search_text.get().to_lowercase();
        
        let mut certs: Vec<_> = graph.nodes.iter()
            .filter(|n| !n.industry_mappings.is_empty())
            .filter(|n| domain_filter == "All" || n.domain == domain_filter)
            .filter(|n| diff_filter == "All" || n.difficulty == diff_filter)
            .filter(|n| search.is_empty() || n.title.to_lowercase().contains(&search) || n.domain.to_lowercase().contains(&search))
            .filter(|n| {
                if sector_filter == MetabolicSector::All { return true; }
                match sector_filter {
                    MetabolicSector::DigitalMesh => n.subject_area.contains("Computer") || n.domain.contains("IT") || n.domain.contains("Cyber"),
                    MetabolicSector::HabitatEngineering => n.subject_area.contains("Engineering") || n.domain.contains("Habitat") || n.domain.contains("Building"),
                    MetabolicSector::BioregionalVitality => n.subject_area.contains("Health") || n.domain.contains("Vitality") || n.domain.contains("Care"),
                    MetabolicSector::SovereignGovernance => n.subject_area.contains("Social") || n.domain.contains("Law") || n.domain.contains("Governance"),
                    MetabolicSector::IndustrialMetabolism => n.subject_area.contains("Engineering") || n.domain.contains("Trades") || n.domain.contains("Industrial"),
                    _ => true,
                }
            })
            .map(|n| {
                // Calculate coverage based on mastered prerequisites
                let prereqs = graph.prereqs_for(&n.id);
                let mastered_set = p.mastered_ids();
                
                let missing: Vec<(String, bool)> = prereqs.iter()
                    .filter(|pid| p.get(*pid).status != ProgressStatus::Mastered)
                    .map(|pid| {
                        let is_ready = graph.prereqs_for(pid).iter().all(|ppid| mastered_set.contains(*ppid));
                        let title = graph.node(pid).map(|node| node.title.clone()).unwrap_or(pid.to_string());
                        (title, is_ready)
                    })
                    .collect();
                
                let coverage = if prereqs.is_empty() { 
                    if p.get(&n.id).status == ProgressStatus::Mastered { 100 } else { 0 }
                } else {
                    let mastered_count = prereqs.iter()
                        .filter(|pid| mastered_set.contains(*pid))
                        .count();
                    ((mastered_count as f32 / prereqs.len() as f32) * 100.0) as u8
                };

                let status = if p.get(&n.id).status == ProgressStatus::Mastered { "Certified" }
                             else if coverage >= 80 { "Ready to Certify" }
                             else if coverage > 0 { "In Progress" }
                             else { "Locked" };

                let industry_code = n.industry_mappings.first()
                    .map(|m| m.code.clone())
                    .unwrap_or_else(|| "CERT".to_string());

                (n.clone(), coverage, status, missing, industry_code)
            })
            .collect();
            
        // PRIORITY SORTING: Highest coverage first
        certs.sort_by(|a, b| b.1.cmp(&a.1));
        certs
    });

    // ACTION: Copy Sovereign Resume to Clipboard
    let copy_resume = move |_| {
        let cert_data = certifications.get();
        let top_matches: Vec<_> = cert_data.iter()
            .filter(|(_, coverage, _, _, _)| *coverage >= 50)
            .take(3)
            .map(|(node, coverage, _, _, code)| {
                let mastery_hash = format!("{:x}", md5::compute(format!("{}-mastery", node.id)));
                format!("{} ({}) - {}% Match [Hash: {}...]", node.title, code, coverage, &mastery_hash[0..8])
            })
            .collect();
        
        let resume_text = format!(
            "VERIFIABLE SOVEREIGN PROFESSIONAL SUMMARY\nPlatform: Mycelix Praxis (CLR 2.0 Compliant)\nAgent DID: did:mycelix:praxis-alpha-student\n\nTOP INDUSTRY ALIGNMENTS:\n- {}\n\nROOT MASTERY:\nLocal contribution verified by community consensus and Symthaea Auditor.\n\nVERIFICATION GATEWAY: https://mycelix.org/verify/praxis-alpha-student",
            top_matches.join("\n- ")
        );
        
        #[cfg(web)]
        {
            let window = web_sys::window().expect("no global `window` exists");
            let navigator = window.navigator();
            let clipboard = navigator.clipboard();
            let _ = clipboard.write_text(&resume_text);
        }
    };

    view! {
        <div class="career-path-page">
            <header class="career-header">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem">
                    <h2>"Professional Path Visualizer"</h2>
                    <button 
                        class="btn-primary"
                        on:click=copy_resume
                    >
                        "\u{1F4C4} Copy Sovereign Resume"
                    </button>
                </div>
                <p class="career-subtitle">"Real-time semantic mapping between your local mastery and global industry standards."</p>

                <div class="metabolic-sector-filter" style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1rem">
                    {MetabolicSector::all().iter().map(|s| {
                        let s = *s;
                        view! {
                            <button 
                                class=move || if selected_sector.get() == s { "sector-btn active" } else { "sector-btn" }
                                on:click=move |_| set_selected_sector.set(s)
                                style="padding: 0.4rem 0.8rem; border-radius: 4px; border: 1px solid var(--border); background: var(--surface); cursor: pointer; font-size: 0.8rem"
                            >
                                {s.label()}
                            </button>
                        }
                    }).collect_view()}
                </div>

                <div class="career-search-bar" style="max-width: 600px; margin: 0.5rem auto 1rem auto">
                    <input 
                        type="text" 
                        placeholder="Search certifications (e.g. CompTIA, Solar, Finance)..."
                        style="width: 100%; padding: 0.75rem 1.25rem; border-radius: 30px; border: 2px solid var(--border); background: var(--surface); color: var(--text); font-size: 1rem"
                        prop:value=search_text
                        on:input=move |ev| set_search_text.set(event_target_value(&ev))
                    />
                </div>

                <div class="difficulty-filter-group" style="display: flex; gap: 0.5rem; justify-content: center; margin-bottom: 1rem">
                    {["All", "Beginner", "Intermediate", "Advanced", "Expert"].iter().map(|&d| {
                        let d_for_check = d.to_string();
                        let d_for_click = d.to_string();
                        view! {
                            <button 
                                class=move || if selected_difficulty.get() == d_for_check { "filter-pill active" } else { "filter-pill" }
                                on:click=move |_| set_selected_difficulty.set(d_for_click.clone())
                                style="font-size: 0.75rem; padding: 0.25rem 0.75rem; border-radius: 15px; border: 1px solid var(--border); background: var(--surface-low); cursor: pointer"
                            >
                                {d}
                            </button>
                        }
                    }).collect_view()}
                </div>
                
                <div class="domain-filter-scroll" style="overflow-x: auto; white-space: nowrap; padding-bottom: 0.5rem">
                    {move || domains.get().into_iter().map(|d| {
                        let d_check = d.clone();
                        let d_click = d.clone();
                        view! {
                            <button 
                                class=move || if selected_domain.get() == d_check { "domain-pill active" } else { "domain-pill" }
                                on:click=move |_| set_selected_domain.set(d_click.clone())
                                style="display: inline-block; margin-right: 0.5rem; padding: 0.4rem 1rem; border-radius: 20px; border: 1px solid var(--border); background: var(--surface-low); cursor: pointer"
                            >{d}</button>
                        }
                    }).collect_view()}
                </div>
            </header>

            <div class="career-visualizer-container">
                <div class="visualizer-roots">
                    <h3>"Sovereign Roots"</h3>
                    <div class="root-nodes">
                        {move || {
                            let p = progress.get();
                            let graph = curriculum_graph();
                            graph.nodes.iter()
                                .filter(|n| p.get(&n.id).status == ProgressStatus::Mastered)
                                .take(12)
                                .map(|n| {
                                    view! {
                                        <div class="root-node-pill">
                                            <span class="pill-icon">"\u{1F331}"</span>
                                            {n.title.clone()}
                                        </div>
                                    }
                                }).collect_view()
                        }}
                    </div>
                </div>

                <div class="visualizer-trunk">
                    <div class="trunk-line"></div>
                </div>

                <div class="visualizer-fruits">
                    <h3>"Global Fruits"</h3>
                    <div class="fruit-nodes">
                        {move || certifications.get().into_iter().map(|(node, coverage, status, missing, industry_code)| {
                            let title = node.title.clone();
                            view! {
                                <FruitCard 
                                    title=title
                                    node=node
                                    coverage=coverage 
                                    status=status
                                    category=node.domain.clone()
                                    missing=missing
                                    industry_code=industry_code
                                />
                            }
                        }).collect_view()}
                    </div>
                </div>
            </div>
        </div>
    }
}

#[component]
fn FruitCard(
    title: String,
    node: CurriculumNode,
    coverage: u8,
    status: &'static str,
    category: String,
    missing: Vec<(String, bool)>, // (Title, is_ready_to_learn)
    industry_code: String,
) -> impl IntoView {
    let (show_gap, set_show_gap) = signal(false);
    let color = if coverage > 80 { "var(--success)" }
                else if coverage > 50 { "var(--warning)" }
                else { "var(--text-tertiary)" };

    view! {
        <div class="fruit-card" on:click=move |_| set_show_gap.update(|v| *v = !*v)>
            <div class="fruit-header">
                <span class="fruit-category">{category}</span>
                <span class="fruit-code-badge" style="background: var(--surface-high); padding: 0.1rem 0.4rem; border-radius: 4px; font-size: 0.65rem; font-weight: 800; border: 1px solid var(--border)">
                    {industry_code}
                </span>
                <span class="fruit-status">{status}</span>
            </div>
            <h4>{title}</h4>

            {move || node.economic_signals.as_ref().map(|s| {
                view! {
                    <div class="economic-signals" style="display: flex; gap: 0.5rem; margin-bottom: 0.8rem">
                        <span class="signal-badge" style="font-size: 0.65rem; background: var(--surface-low); padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid var(--primary); color: var(--primary); font-weight: 700">
                            {format!("${}k+", s.average_starting_salary / 1000)}
                        </span>
                        <span class="signal-badge" style="font-size: 0.65rem; background: var(--surface-low); padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid var(--success); color: var(--success); font-weight: 700">
                            {format!("{} Demand", s.market_demand)}
                        </span>
                        {move || if s.local_demand_multiplier > 1.0 {
                            view! {
                                <span class="signal-badge" style="font-size: 0.65rem; background: var(--warning-low); padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid var(--warning); color: var(--warning); font-weight: 800">
                                    {format!("COMMUNITY NEED: {}x TEND", s.local_demand_multiplier)}
                                </span>
                            }.into_any()
                        } else {
                            view! { <span></span> }.into_any()
                        }}
                    </div>
                }
            })}

            <div class="coverage-bar-wrap">
                <div style="display: flex; justify-content: space-between; font-size: 0.7rem; margin-bottom: 0.2rem">
                    <span>"Match Coverage"</span>
                    <span style=format!("color: {}; font-weight: 700", color)>{coverage}"%"</span>
                </div>
                <div class="progress-bar" style="height: 6px">
                    <div class="progress-bar-fill" style=format!("width: {}%; background-color: {}", coverage, color)></div>
                </div>
            </div>

            <div class="root-stack-viz" style="margin-top: 1rem; display: flex; flex-wrap: wrap; gap: 0.3rem">
                {move || {
                    (0..5).map(|i| {
                        let is_mastered = i < (coverage / 20) as usize;
                        view! {
                            <div 
                                style=format!("width: 10px; height: 10px; border-radius: 50%; background: {}", 
                                    if is_mastered { "var(--success)" } else { "var(--surface-high)" })
                                title=if is_mastered { "Mastered Core Competency" } else { "Missing Requirement" }
                            ></div>
                        }
                    }).collect_view()
                }}
            </div>

            {move || if show_gap.get() && !missing.is_empty() {
                view! {
                    <div class="gap-analysis">
                        <h6 style="margin-bottom: 0.5rem">"Next Steps (Mastery Path):"</h6>
                        <ul style="list-style: none; padding: 0; margin: 0">
                            {missing.iter().map(|(m, is_ready)| {
                                let style = if *is_ready { "color: var(--warning); font-weight: 600" } else { "color: var(--text-tertiary)" };
                                let icon = if *is_ready { "\u{2192} [READY] " } else { "\u{1F512} [LOCKED] " };
                                view! { 
                                    <li style=format!("font-size: 0.75rem; margin-bottom: 0.2rem; {}", style)>
                                        {icon} {m}
                                    </li> 
                                }
                            }).collect_view()}
                        </ul>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
            
            <button class="btn-sm btn-outline" style="width: 100%; margin-top: 1rem">"Request Legacy Transcript"</button>
        </div>
    }
}

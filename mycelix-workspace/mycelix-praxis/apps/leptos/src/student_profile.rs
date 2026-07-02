// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Student Profile management and civilizational role visualization.

use leptos::prelude::*;
use serde::{Deserialize, Serialize};
use crate::curriculum::{curriculum_graph, use_progress, ProgressStatus};
use crate::persistence;
use crate::pages::careers::MetabolicSector;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct StudentProfile {
    pub name: String,
    pub did: String,
    pub grade: u8,
    pub biography: String,
    pub primary_interests: Vec<String>,
    pub is_hollow_state: bool, // Plausible Deniability for high-crime areas
}

#[component]
pub fn ProfilePage() -> impl IntoView {
    let progress = use_progress();
    let profile = persistence::load::<StudentProfile>("praxis_profile").unwrap_or_default();
    
    // Derived: Metabolic Readiness Stats
    let metabolic_readiness = Memo::new(move |_| {
        let graph = curriculum_graph();
        let p = progress.get();
        
        let sectors = vec![
            MetabolicSector::DigitalMesh,
            MetabolicSector::HabitatEngineering,
            MetabolicSector::BioregionalVitality,
            MetabolicSector::SovereignGovernance,
            MetabolicSector::IndustrialMetabolism,
        ];
        
        sectors.into_iter().map(|sector| {
            let sector_certs: Vec<_> = graph.nodes.iter()
                .filter(|n| !n.industry_mappings.is_empty())
                .filter(|n| {
                    match sector {
                        MetabolicSector::DigitalMesh => n.subject_area.contains("Computer") || n.domain.contains("IT") || n.domain.contains("Cyber"),
                        MetabolicSector::HabitatEngineering => n.subject_area.contains("Engineering") || n.domain.contains("Habitat") || n.domain.contains("Building"),
                        MetabolicSector::BioregionalVitality => n.subject_area.contains("Health") || n.domain.contains("Vitality") || n.domain.contains("Care"),
                        MetabolicSector::SovereignGovernance => n.subject_area.contains("Social") || n.domain.contains("Law") || n.domain.contains("Governance"),
                        MetabolicSector::IndustrialMetabolism => n.subject_area.contains("Engineering") || n.domain.contains("Trades") || n.domain.contains("Industrial"),
                        _ => true,
                    }
                })
                .collect();
            
            if sector_certs.is_empty() { return (sector, 0u8); }
            
            let total_coverage: f32 = sector_certs.iter().map(|n| {
                let prereqs = graph.prereqs_for(&n.id);
                if prereqs.is_empty() { return if p.get(&n.id).status == ProgressStatus::Mastered { 100.0 } else { 0.0 }; }
                let mastered_count = prereqs.iter().filter(|pid| p.get(*pid).status == ProgressStatus::Mastered).count();
                (mastered_count as f32 / prereqs.len() as f32) * 100.0
            }).sum();
            
            (sector, (total_coverage / sector_certs.len() as f32) as u8)
        }).collect::<Vec<_>>()
    });

    view! {
        <div class="profile-page">
            <header class="profile-header">
                <div class="profile-avatar">"\u{1F331}"</div>
                <h2>{profile.name.clone()}</h2>
                <div class="profile-did">{profile.did.clone()}</div>
            </header>

            <section class="readiness-section">
                <h3>"Metabolic Readiness Radar"</h3>
                <p class="section-subtitle">"Your average match coverage across civilizational sectors."</p>
                
                <div class="radar-container" style="margin-top: 2rem">
                    {move || metabolic_readiness.get().into_iter().map(|(sector, score)| {
                        let color = match sector {
                            MetabolicSector::DigitalMesh => "var(--primary)",
                            MetabolicSector::HabitatEngineering => "var(--accent)",
                            MetabolicSector::BioregionalVitality => "var(--success)",
                            MetabolicSector::SovereignGovernance => "var(--warning)",
                            MetabolicSector::IndustrialMetabolism => "var(--error)",
                            _ => "var(--text-tertiary)",
                        };
                        view! {
                            <div class="radar-bar-row" style="margin-bottom: 1rem">
                                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 0.3rem">
                                    <span style="font-weight: 600">{sector.label()}</span>
                                    <span>{score}"%"</span>
                                </div>
                                <div class="progress-bar" style="height: 10px; background: var(--surface-high); border-radius: 5px; overflow: hidden">
                                    <div class="progress-bar-fill" style=format!("width: {}%; background: {}; height: 100%; transition: width 0.5s ease", score, color)></div>
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>

            <section class="profile-details">
                <div class="detail-card">
                    <h4>"Biography"</h4>
                    <p>{profile.biography.clone()}</p>
                </div>
                <div class="detail-card">
                    <h4>"Core Interests"</h4>
                    <div class="interest-pills">
                        {profile.primary_interests.iter().map(|i| view! { <span class="interest-pill">{i}</span> }).collect_view()}
                    </div>
                </div>
            </section>

            <section class="sovereign-failsafe" style="margin-top: 3rem; padding: 2rem; background: var(--error-low); border: 1px solid var(--error); border-radius: 12px">
                <h3 style="color: var(--error)">"Sovereign Failsafe (Ahimsa)"</h3>
                <p style="font-size: 0.85rem; line-height: 1.5; color: var(--text)">
                    "You retain sovereign ownership of your knowledge graph. This failsafe ensures you can never be trapped by this protocol. One click to export your entire history, reputation, and artifacts."
                </p>
                <button 
                    class="btn-primary" 
                    style="margin-top: 1rem; background: var(--error); border-color: var(--error)"
                    on:click=move |_| {
                        // Trigger 1-click JSON/HDC export logic
                    }
                >
                    "\u{1F513} Emancipate My Data"
                </button>
            </section>
        </div>
    }
}

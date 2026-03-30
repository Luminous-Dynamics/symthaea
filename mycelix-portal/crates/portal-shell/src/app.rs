// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Portal — the Consciousness Orb interface.
//!
//! No sidebar. No pages. No navigation.
//! The portal IS the orb. Domains orbit it. You zoom into aspects of yourself.

use leptos::prelude::*;
use portal_domain_trait::ConsciousnessTier;

use crate::identity::{PortalIdentity, VaultState};

/// Experiential phenotype — how you perceive your consciousness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phenotype {
    /// Spatial: consciousness as sphere with orbiting domains
    Orb,
    /// Chronological: consciousness as a river of events
    Stream,
    /// Categorical: consciousness as a living garden
    Garden,
    /// Minimal: consciousness as a heartbeat
    Pulse,
}

impl Phenotype {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Orb => "Orb",
            Self::Stream => "Stream",
            Self::Garden => "Garden",
            Self::Pulse => "Pulse",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Orb => "See yourself as a sphere of consciousness with domains orbiting around you",
            Self::Stream => "Experience your data as a flowing river of events through time",
            Self::Garden => "Tend your domains like plots in a living garden",
            Self::Pulse => "Just the heartbeat — minimal, focused, fast",
        }
    }
}

/// A domain node orbiting the consciousness orb.
#[derive(Clone, Debug)]
pub struct OrbitalDomain {
    pub id: &'static str,
    pub name: &'static str,
    pub bio_name: &'static str,
    pub color: &'static str,
    pub glow: &'static str,
    pub min_tier: ConsciousnessTier,
    /// Angle offset in the orbital ring (radians)
    pub angle: f64,
    /// Activity level (0.0-1.0) — drives glow intensity
    pub activity: f64,
}

/// Event flowing through the portal — cross-domain data movement.
#[derive(Clone, Debug)]
pub struct PortalEvent {
    pub from_domain: &'static str,
    pub description: String,
    pub energy: f64, // How much this event affects the orb
}

#[component]
pub fn App() -> impl IntoView {
    let identity = PortalIdentity::new();
    provide_context(identity.clone());

    let active_domain: RwSignal<Option<String>> = RwSignal::new(None);
    let phenotype: RwSignal<Option<Phenotype>> = RwSignal::new(None);
    provide_context(active_domain);
    provide_context(phenotype);

    // Event stream — cross-domain energy flowing to the orb
    let events: RwSignal<Vec<PortalEvent>> = RwSignal::new(vec![
        PortalEvent {
            from_domain: "health",
            description: "FL gradient contributed to diabetes study".into(),
            energy: 0.05,
        },
        PortalEvent {
            from_domain: "health",
            description: "$42 metabolic yield from research".into(),
            energy: 0.03,
        },
    ]);

    // All orbital domains
    let domains = vec![
        OrbitalDomain {
            id: "health", name: "Health", bio_name: "Homeostasis",
            color: "#0D7377", glow: "#06D6C8",
            min_tier: ConsciousnessTier::Participant,
            angle: 0.0, activity: 0.8,
        },
        OrbitalDomain {
            id: "governance", name: "Governance", bio_name: "Consensus",
            color: "#7C3AED", glow: "#A78BFA",
            min_tier: ConsciousnessTier::Citizen,
            angle: std::f64::consts::TAU / 8.0, activity: 0.4,
        },
        OrbitalDomain {
            id: "edunet", name: "Education", bio_name: "Growth",
            color: "#2563EB", glow: "#60A5FA",
            min_tier: ConsciousnessTier::Participant,
            angle: 2.0 * std::f64::consts::TAU / 8.0, activity: 0.3,
        },
        OrbitalDomain {
            id: "finance", name: "Finance", bio_name: "Metabolism",
            color: "#D97706", glow: "#FBBF24",
            min_tier: ConsciousnessTier::Citizen,
            angle: 3.0 * std::f64::consts::TAU / 8.0, activity: 0.6,
        },
        OrbitalDomain {
            id: "commons", name: "Commons", bio_name: "Mutualism",
            color: "#059669", glow: "#34D399",
            min_tier: ConsciousnessTier::Citizen,
            angle: 4.0 * std::f64::consts::TAU / 8.0, activity: 0.2,
        },
        OrbitalDomain {
            id: "hearth", name: "Hearth", bio_name: "Kinship",
            color: "#DB2777", glow: "#F472B6",
            min_tier: ConsciousnessTier::Participant,
            angle: 5.0 * std::f64::consts::TAU / 8.0, activity: 0.5,
        },
        OrbitalDomain {
            id: "knowledge", name: "Knowledge", bio_name: "Noosphere",
            color: "#0891B2", glow: "#22D3EE",
            min_tier: ConsciousnessTier::Steward,
            angle: 6.0 * std::f64::consts::TAU / 8.0, activity: 0.1,
        },
        OrbitalDomain {
            id: "space", name: "Space", bio_name: "Cosmos",
            color: "#4F46E5", glow: "#818CF8",
            min_tier: ConsciousnessTier::Guardian,
            angle: 7.0 * std::f64::consts::TAU / 8.0, activity: 0.05,
        },
    ];

    let tier = identity.tier;
    let consciousness = identity.consciousness_score;
    let vault = identity.vault;

    let no_vault = move || vault.get() == VaultState::NoVault;
    let has_vault = move || vault.get() != VaultState::NoVault;
    let is_orbital = move || active_domain.get().is_none();
    let is_zoomed = move || active_domain.get().is_some();

    view! {
        <div class="portal-universe">
            {move || {
                if vault.get() == VaultState::NoVault {
                    // FIRST BREATH — seed waiting
                    view! {
                        <div class="first-breath-universe">
                            <div class="first-breath-seed-container">
                                <div class="seed-pulse" />
                                <div class="seed-pulse delay-1" />
                                <div class="seed-pulse delay-2" />
                                <div class="seed-core" />
                            </div>
                            <h1 class="first-breath-title">"First Breath"</h1>
                            <p class="first-breath-text">
                                "One identity. Every domain. Your consciousness made sovereign."
                            </p>
                            <button class="first-breath-cta" on:click=move |_| {
                                vault.set(VaultState::Unlocked);
                            }>"Breathe"</button>
                        </div>
                    }.into_any()
                } else if phenotype.get().is_none() {
                    // PHENOTYPE SELECTION — "How do you see yourself?"
                    view! {
                        <div class="phenotype-selection">
                            <h1 class="phenotype-question">"How do you see yourself?"</h1>
                            <div class="phenotype-grid">
                                {[Phenotype::Orb, Phenotype::Stream, Phenotype::Garden, Phenotype::Pulse]
                                    .iter().map(|p| {
                                    let p = *p;
                                    view! {
                                        <button class="phenotype-card" on:click=move |_| phenotype.set(Some(p))>
                                            <div class=format!("phenotype-preview {}", p.label().to_lowercase()) />
                                            <span class="phenotype-name">{p.label()}</span>
                                            <span class="phenotype-desc">{p.description()}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }.into_any()
                } else if active_domain.get().is_none() {
                    // MAIN VIEW — rendered according to phenotype
                    let ph = phenotype.get().unwrap_or(Phenotype::Orb);
                    match ph {
                    Phenotype::Stream => view! {
                        <div class="stream-view">
                            <div class="stream-header">
                                <span class="orb-score">{format!("{:.0}%", consciousness.get() * 100.0)}</span>
                                <span class="orb-tier">{tier.get().label()}</span>
                                <button class="phenotype-switch" on:click=move |_| phenotype.set(None)>"Switch View"</button>
                            </div>
                            <div class="stream-timeline">
                                {events.get().iter().map(|e| {
                                    view! {
                                        <div class="stream-event-card">
                                            <div class="stream-event-dot" />
                                            <div class="stream-event-body">
                                                <span class="stream-event-text">{e.description.clone()}</span>
                                                <span class="stream-event-domain">{e.from_domain}</span>
                                            </div>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            <div class="stream-domains">
                                {domains.iter().filter(|d| d.min_tier <= ConsciousnessTier::from_score(consciousness.get_untracked())).map(|d| {
                                    let id = d.id;
                                    view! {
                                        <button class="stream-domain-chip" style=format!("border-color: {}", d.color)
                                            on:click=move |_| { active_domain.set(Some(id.to_string())); }>
                                            {d.bio_name}
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }.into_any(),
                    Phenotype::Garden => view! {
                        <div class="garden-view">
                            <div class="garden-header">
                                <span class="orb-score">{format!("{:.0}%", consciousness.get() * 100.0)}</span>
                                <span class="orb-tier">{tier.get().label()}</span>
                                <button class="phenotype-switch" on:click=move |_| phenotype.set(None)>"Switch View"</button>
                            </div>
                            <div class="garden-plots">
                                {domains.iter().map(|d| {
                                    let id = d.id;
                                    let accessible = d.min_tier <= ConsciousnessTier::from_score(consciousness.get_untracked());
                                    let activity = d.activity;
                                    view! {
                                        <button
                                            class=move || if accessible { "garden-plot alive" } else { "garden-plot dormant" }
                                            style=format!("--plot-color: {}; --plot-glow: {}; --plot-height: {:.0}%;", d.color, d.glow, activity * 100.0)
                                            disabled=!accessible
                                            on:click=move |_| { if accessible { active_domain.set(Some(id.to_string())); } }
                                        >
                                            <div class="plot-growth" />
                                            <span class="plot-name">{d.bio_name}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                        </div>
                    }.into_any(),
                    Phenotype::Pulse => view! {
                        <div class="pulse-view">
                            <div class="pulse-center">
                                <div class="pulse-beat" />
                                <span class="pulse-score">{format!("{:.0}%", consciousness.get() * 100.0)}</span>
                            </div>
                            <div class="pulse-vitals">
                                {domains.iter().filter(|d| d.activity > 0.3 && d.min_tier <= ConsciousnessTier::from_score(consciousness.get_untracked())).map(|d| {
                                    let id = d.id;
                                    view! {
                                        <button class="pulse-vital" style=format!("color: {}", d.glow)
                                            on:click=move |_| { active_domain.set(Some(id.to_string())); }>
                                            <span class="vital-name">{d.bio_name}</span>
                                            <span class="vital-activity">{format!("{:.0}%", d.activity * 100.0)}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            <button class="phenotype-switch" on:click=move |_| phenotype.set(None)>"Switch View"</button>
                        </div>
                    }.into_any(),
                    _ => view! {
                        <div class="orbital-view">
                            <div class="consciousness-orb">
                                <div class="orb-core" />
                                <div class="orb-ring ring-identity" />
                                <div class="orb-ring ring-reputation" />
                                <div class="orb-ring ring-community" />
                                <div class="orb-ring ring-engagement" />
                                <div class="orb-label">
                                    <span class="orb-score">{move || format!("{:.0}%", consciousness.get() * 100.0)}</span>
                                    <span class="orb-tier">{move || tier.get().label()}</span>
                                </div>
                            </div>
                            <div class="orbital-ring">
                                {domains.iter().map(|d| {
                                    let id = d.id;
                                    let bio = d.bio_name;
                                    let color = d.color;
                                    let glow = d.glow;
                                    let angle = d.angle;
                                    let activity = d.activity;
                                    let name = d.name;
                                    let accessible = d.min_tier <= ConsciousnessTier::from_score(consciousness.get_untracked());
                                    view! {
                                        <button
                                            class=move || if accessible { "orbital-node accessible" } else { "orbital-node locked" }
                                            style=format!("--node-angle: {:.2}rad; --node-color: {}; --node-glow: {}; --node-activity: {:.2};", angle, color, glow, activity)
                                            disabled=!accessible
                                            on:click=move |_| { if accessible { active_domain.set(Some(id.to_string())); } }
                                            aria-label=format!("{}", name)
                                        >
                                            <span class="node-dot" />
                                            <span class="node-name">{bio}</span>
                                        </button>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            <div class="event-stream">
                                {events.get().iter().map(|e| {
                                    view! {
                                        <div class="stream-event">
                                            <span class="event-dot" />
                                            <span>{e.description.clone()}</span>
                                        </div>
                                    }
                                }).collect::<Vec<_>>()}
                            </div>
                            <button class="phenotype-switch orb-switch" on:click=move |_| phenotype.set(None)>"Switch View"</button>
                        </div>
                    }.into_any(),
                    } // end match
                } else {
                    view! {
                        <div class="domain-view">
                            <button class="domain-back" on:click=move |_| active_domain.set(None)>
                                "← Return to Orb"
                            </button>
                            <h1 class="bio-title">{move || active_domain.get().unwrap_or_default()}</h1>
                            <p class="bio-subtitle">"Domain content renders here"</p>
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Portal — the Consciousness Orb interface.
//!
//! No sidebar. No pages. No navigation.
//! The portal IS the orb. Domains orbit it. You zoom into aspects of yourself.

use leptos::prelude::*;
use portal_domain_trait::ConsciousnessTier;

use crate::identity::{PortalIdentity, VaultState};

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

    // Active (zoomed-in) domain — None means orbital view
    let active_domain: RwSignal<Option<String>> = RwSignal::new(None);
    provide_context(active_domain);

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
                } else if active_domain.get().is_none() {
                    view! {
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
                        </div>
                    }.into_any()
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

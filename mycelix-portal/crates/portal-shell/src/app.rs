// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mycelix Portal — the Consciousness Orb interface.
//!
//! No sidebar. No pages. No navigation.
//! The portal IS the orb. Domains orbit it. You zoom into aspects of yourself.

use leptos::prelude::*;
use portal_domain_trait::ConsciousnessTier;

use crate::background::HomeostasisBackground;
use crate::identity::{ConductorStatus, PortalIdentity, VaultState};
use crate::pages::DomainContent;

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
    pub _energy: f64,
}

#[component]
pub fn App() -> impl IntoView {
    let identity = PortalIdentity::new();
    provide_context(identity.clone());

    let active_domain: RwSignal<Option<String>> = RwSignal::new(None);

    // Load saved phenotype from localStorage
    let saved_phenotype = web_sys::window()
        .and_then(|w| w.local_storage().ok().flatten())
        .and_then(|s| s.get_item("mycelix_phenotype").ok().flatten())
        .and_then(|v| match v.as_str() {
            "orb" => Some(Phenotype::Orb),
            "stream" => Some(Phenotype::Stream),
            "garden" => Some(Phenotype::Garden),
            "pulse" => Some(Phenotype::Pulse),
            _ => None,
        });
    let phenotype: RwSignal<Option<Phenotype>> = RwSignal::new(saved_phenotype);

    // Save phenotype changes to localStorage
    Effect::new(move |_| {
        if let Some(p) = phenotype.get() {
            if let Some(storage) = web_sys::window()
                .and_then(|w| w.local_storage().ok().flatten())
            {
                let _ = storage.set_item("mycelix_phenotype", p.label().to_lowercase().as_str());
            }
        }
    });
    provide_context(active_domain);
    provide_context(phenotype);

    // Event stream — cross-domain energy flowing to the orb
    let events: RwSignal<Vec<PortalEvent>> = RwSignal::new(vec![
        PortalEvent {
            from_domain: "health",
            description: "FL gradient contributed to diabetes study".into(),
            _energy: 0.05,
        },
        PortalEvent {
            from_domain: "health",
            description: "$42 metabolic yield from research".into(),
            _energy: 0.03,
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
            angle: std::f64::consts::TAU / 11.0, activity: 0.4,
        },
        OrbitalDomain {
            id: "praxis", name: "Education", bio_name: "Growth",
            color: "#2563EB", glow: "#60A5FA",
            min_tier: ConsciousnessTier::Participant,
            angle: 2.0 * std::f64::consts::TAU / 11.0, activity: 0.3,
        },
        OrbitalDomain {
            id: "finance", name: "Finance", bio_name: "Metabolism",
            color: "#D97706", glow: "#FBBF24",
            min_tier: ConsciousnessTier::Citizen,
            angle: 3.0 * std::f64::consts::TAU / 11.0, activity: 0.6,
        },
        OrbitalDomain {
            id: "commons", name: "Commons", bio_name: "Mutualism",
            color: "#059669", glow: "#34D399",
            min_tier: ConsciousnessTier::Citizen,
            angle: 4.0 * std::f64::consts::TAU / 11.0, activity: 0.2,
        },
        OrbitalDomain {
            id: "hearth", name: "Hearth", bio_name: "Kinship",
            color: "#DB2777", glow: "#F472B6",
            min_tier: ConsciousnessTier::Participant,
            angle: 5.0 * std::f64::consts::TAU / 11.0, activity: 0.5,
        },
        OrbitalDomain {
            id: "knowledge", name: "Knowledge", bio_name: "Noosphere",
            color: "#0891B2", glow: "#22D3EE",
            min_tier: ConsciousnessTier::Steward,
            angle: 6.0 * std::f64::consts::TAU / 11.0, activity: 0.1,
        },
        OrbitalDomain {
            id: "space", name: "Space", bio_name: "Cosmos",
            color: "#4F46E5", glow: "#818CF8",
            min_tier: ConsciousnessTier::Guardian,
            angle: 7.0 * std::f64::consts::TAU / 11.0, activity: 0.05,
        },
        OrbitalDomain {
            id: "lucid", name: "LUCID", bio_name: "Inner Eye",
            color: "#8B5CF6", glow: "#C4B5FD",
            min_tier: ConsciousnessTier::Participant,
            angle: 8.0 * std::f64::consts::TAU / 11.0, activity: 0.45,
        },
        OrbitalDomain {
            id: "mail", name: "Mail", bio_name: "Signal Stream",
            color: "#DC2626", glow: "#FCA5A5",
            min_tier: ConsciousnessTier::Participant,
            angle: 9.0 * std::f64::consts::TAU / 11.0, activity: 0.7,
        },
        OrbitalDomain {
            id: "admin", name: "System", bio_name: "Nervous System",
            color: "#475569", glow: "#94A3B8",
            min_tier: ConsciousnessTier::Guardian,
            angle: 10.0 * std::f64::consts::TAU / 11.0, activity: 0.15,
        },
    ];

    let tier = identity.tier;
    let consciousness = identity.consciousness_score;
    let vault = identity.vault;

    let _no_vault = move || vault.get() == VaultState::NoVault;
    let _has_vault = move || vault.get() != VaultState::NoVault;
    let _is_orbital = move || active_domain.get().is_none();
    let _is_zoomed = move || active_domain.get().is_some();

    let conductor = identity.conductor_status;

    // Root class reflects conductor state — CSS responds
    let universe_class = move || {
        if conductor.get() == ConductorStatus::Connected {
            "portal-universe conductor-live"
        } else {
            "portal-universe"
        }
    };

    view! {
        <div class=universe_class>
            <HomeostasisBackground />

            // Sovereignty badge (top-left)
            <div class="sovereignty-badge">
                <span class="sovereignty-shield">"◈"</span>
                <span class="sovereignty-count">"0 leaks"</span>
            </div>

            // Conductor status badge (top-right)
            <div class="conductor-badge">
                <span class=move || match conductor.get() {
                    ConductorStatus::Connected => "conductor-dot connected",
                    ConductorStatus::Connecting => "conductor-dot connecting",
                    ConductorStatus::Mock => "conductor-dot mock",
                } />
                <span>{move || match conductor.get() {
                    ConductorStatus::Connected => "Live",
                    ConductorStatus::Connecting => "Probing",
                    ConductorStatus::Mock => "Local",
                }}</span>
            </div>
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
                                // First Breath sound — a rising tone
                                play_breath_tone();
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
                                            on:click=move |_| { if accessible { play_domain_tone(id); active_domain.set(Some(id.to_string())); } }
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
                            // SVG connections — mycelial tendrils from orb to each node
                            <svg class="orbital-connections" viewBox="-250 -250 500 500" aria-hidden="true">
                                {domains.iter().map(|d| {
                                    let radius = 180.0;
                                    let x = radius * d.angle.cos();
                                    let y = radius * d.angle.sin();
                                    let opacity = if d.min_tier <= ConsciousnessTier::from_score(consciousness.get_untracked()) {
                                        0.15 + d.activity * 0.3
                                    } else { 0.03 };
                                    view! {
                                        <line x1="0" y1="0" x2=x.to_string() y2=y.to_string()
                                            stroke=d.glow stroke-width="1" opacity=opacity.to_string()
                                            stroke-dasharray="4 8" />
                                    }
                                }).collect::<Vec<_>>()}
                            </svg>

                            // Domain nodes — positioned by computed angle
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
                                    // Compute position with JS-compatible math
                                    let radius = 180.0;
                                    let x = radius * angle.cos();
                                    let y = radius * angle.sin();
                                    view! {
                                        <button
                                            class=move || if accessible { "orbital-node accessible" } else { "orbital-node locked" }
                                            style=format!(
                                                "left: calc(50% + {:.0}px - 30px); top: calc(50% + {:.0}px - 20px); --node-color: {}; --node-glow: {}; --node-activity: {:.2};",
                                                x, y, color, glow, activity
                                            )
                                            disabled=!accessible
                                            on:click=move |_| { if accessible { play_domain_tone(id); active_domain.set(Some(id.to_string())); } }
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
                    let domain_id = active_domain.get().unwrap_or_default();
                    let domain_data = domains.iter().find(|d| d.id == domain_id).cloned();
                    view! {
                        <div class="domain-view" style=format!("--domain-color: {}; --domain-glow: {};",
                            domain_data.as_ref().map(|d| d.color).unwrap_or("#0D7377"),
                            domain_data.as_ref().map(|d| d.glow).unwrap_or("#06D6C8"))>
                            <button class="domain-back" on:click=move |_| active_domain.set(None)>
                                "← Return"
                            </button>
                            <div class="domain-header">
                                <div class="domain-orb-mini" />
                                <div>
                                    <h1 class="domain-title">{domain_data.as_ref().map(|d| d.bio_name).unwrap_or("Unknown")}</h1>
                                    <p class="domain-subtitle">{domain_data.as_ref().map(|d| d.name).unwrap_or("")}</p>
                                </div>
                                {domain_app_url(&domain_id).map(|url| view! {
                                    <a class="domain-launch-btn" href=url target="_blank" rel="noopener">
                                        "Open Full App →"
                                    </a>
                                })}
                            </div>
                            <div class="domain-stats">
                                <div class="domain-stat">
                                    <span class="stat-value">{format!("{:.0}%", domain_data.as_ref().map(|d| d.activity * 100.0).unwrap_or(0.0))}</span>
                                    <span class="stat-label">"Activity"</span>
                                </div>
                                <div class="domain-stat">
                                    <span class="stat-value">{move || format!("{:.0}%", consciousness.get() * 100.0)}</span>
                                    <span class="stat-label">"Consciousness"</span>
                                </div>
                                <div class="domain-stat">
                                    <span class="stat-value">{move || tier.get().label()}</span>
                                    <span class="stat-label">"Tier"</span>
                                </div>
                            </div>
                            <DomainContent domain_id=domain_id.clone() />
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}

/// Map domain IDs to standalone app URLs.
fn domain_app_url(domain_id: &str) -> Option<&'static str> {
    match domain_id {
        "governance" => Some("http://localhost:8110"),
        "hearth" => Some("http://localhost:8096"),
        "praxis" => Some("http://localhost:8107"),
        "health" => Some("http://localhost:8111"),
        "music" => Some("http://localhost:8121"),
        _ => None,
    }
}

/// Play a domain entry tone — pitch based on domain character.
fn play_domain_tone(domain_id: &str) {
    let freq = match domain_id {
        "health" => 262.0,     // C4 — grounded, organic
        "governance" => 330.0, // E4 — structured, clear
        "finance" => 392.0,    // G4 — active, energetic
        "praxis" => 349.0,     // F4 — growing, warm
        "commons" => 294.0,    // D4 — communal, solid
        "hearth" => 277.0,     // C#4 — intimate, warm
        "knowledge" => 440.0,  // A4 — resonant, bright
        "space" => 220.0,      // A3 — vast, deep
        _ => 330.0,
    };
    let script = format!(r#"
        (function() {{
            try {{
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain); gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime({freq}, ctx.currentTime);
                gain.gain.setValueAtTime(0, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0.08, ctx.currentTime + 0.3);
                gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 1.5);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 1.5);
            }} catch(e) {{}}
        }})()
    "#);
    let _ = js_sys::eval(&script);
}

/// Play a rising tone on First Breath using Web Audio API.
fn play_breath_tone() {
    let _window = match web_sys::window() {
        Some(w) => w,
        None => return,
    };

    // Create AudioContext via JS eval (web-sys AudioContext needs feature flags)
    let result = js_sys::eval(r#"
        (function() {
            try {
                var ctx = new (window.AudioContext || window.webkitAudioContext)();
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime(220, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 2);
                gain.gain.setValueAtTime(0, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.5);
                gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 3);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 3);
            } catch(e) {}
        })()
    "#);
    let _ = result;
}

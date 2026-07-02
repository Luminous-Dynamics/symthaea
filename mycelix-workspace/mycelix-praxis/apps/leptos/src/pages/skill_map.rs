// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Knowledge Garden / Skill Map — Dynamic, Context-Aware Node Exploration.

use leptos::prelude::*;
use crate::curriculum::{curriculum_graph, use_progress, use_set_progress, ProgressStatus, CurriculumNode, display_subject, Grade, Subject};
use crate::location::use_biome;
use crate::components::hardware::{HardwareScanner, PresenceValidator, SafetyGuard};

#[component]
pub fn SkillMapPage() -> impl IntoView {
    let progress = use_progress();
    let subject = crate::curriculum::use_subject();
    let set_subject = crate::curriculum::use_set_subject();
    let grade = crate::curriculum::use_grade();
    let set_grade = crate::curriculum::use_set_grade();
    let (selected_id, set_selected_id) = signal(None::<String>);
    let (focus_zpd, set_focus_zpd) = signal(false);

    view! {
        <div class="skill-map-page">
            <header class="garden-header">
                <h2>"Knowledge Garden"</h2>
                <p>"Explore the civilizational substrate."</p>
                
                <div class="garden-stats">
                    {move || {
                        let p = progress.get();
                        let mastered = p.mastered_count();
                        let studying = p.studying_count();
                        let total = curriculum_graph().nodes.len();
                        view! {
                            <span>{mastered}" Mastered | "{studying}" Studying | "{total}" Seeds"</span>
                        }
                    }}
                </div>
            </header>

            // Filter Section
            <div class="garden-filters" style="margin-bottom: 2rem; display: flex; gap: 1rem; flex-wrap: wrap">
                <button class="btn-outline" on:click=move |_| set_focus_zpd.update(|v| *v = !*v)>
                    {move || if focus_zpd.get() { "Show All" } else { "Show Ready to Learn" }}
                </button>
            </div>

use crate::components::gl_canvas::GardenCanvas;

// ...
            // Constellation View (LOD-Optimized)
            <div class="constellation-container" style="height: 70vh; background: var(--surface-low); border-radius: 12px; position: relative; overflow: hidden">
                <GardenCanvas node_count=curriculum_graph().nodes.len() />
                
                // Floating Action Layer (Optimistic DOM)
                <div 
                    class="node-sprout" 
                    style="position: absolute; top: 30%; left: 40%; cursor: pointer"
                    on:click=move |_| set_selected_id.set(Some("VOC-AGR-101".to_string()))
                >
                    <div class="sprout-icon">"\u{1F331}"</div>
                    <span style="font-size: 0.7rem">"AGR-101"</span>
                </div>
            </div>

            // Detail Panel
            {move || selected_id.get().and_then(|id| {
                curriculum_graph().node(&id).cloned()
            }).map(|node| {
                view! {
                    <NodeDetail 
                        node=node 
                        on_close=move || set_selected_id.set(None)
                    />
                }
            })}
        </div>
    }
}

#[component]
fn NodeDetail(
    node: CurriculumNode,
    on_close: impl Fn() + 'static,
) -> impl IntoView {
    let progress = use_progress();
    let set_progress = use_set_progress();
    let biome = use_biome();
    let (artifact_url, set_artifact_url) = signal("".to_string());

    let node_id = node.id.clone();
    let status = move || progress.get().get(&node_id).status;

    // Bioregional Morphing Logic
    let morphed_description = move || {
        let b = biome.get();
        let base_desc = node.description.clone();
        
        if let Some(params) = &node.biome_parameters {
            if params.hardiness_zones.contains(&b.hardiness_zone) {
                format!("{} \n\n[BIOREGIONAL ADVICE]: In your Hardiness Zone ({}), this technique is optimal. Current Season ({}) is perfect for execution.", base_desc, b.hardiness_zone, b.current_season)
            } else {
                format!("{} \n\n[BIOREGIONAL WARNING]: Your Hardiness Zone ({}) may require modifications for this technique (Original: Zone {:?}).", base_desc, b.hardiness_zone, params.hardiness_zones)
            }
        } else {
            base_desc
        }
    };

    view! {
        <div class="praxis-detail" style="position: fixed; right: 0; top: 0; width: 400px; height: 100vh; background: var(--surface); border-left: 1px solid var(--border); padding: 2rem; z-index: 1000; overflow-y: auto">
            <header style="display: flex; justify-content: space-between; align-items: flex-start">
                <div>
                    <h3 style="margin: 0">{node.title.clone()}</h3>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary); margin-top: 0.2rem">
                        "ID: "{node.id.clone()}" | TYPE: "{node.node_type.clone()}
                    </div>
                </div>
                <button class="btn-close" on:click=move |_| on_close()>"\u{00D7}"</button>
            </header>

            // HDC Rosetta Badge
            {move || node.hdc_anchor.as_ref().map(|h| {
                view! {
                    <div class="hdc-badge" style="margin-top: 1rem; font-size: 0.6rem; font-family: monospace; background: var(--surface-high); padding: 0.3rem 0.6rem; border-radius: 4px; border: 1px solid var(--primary-low)">
                        "HDC ANCHOR: "{&h.vector_hash[0..16]}"..."
                    </div>
                }
            })}

            // Bioregional Description
            <div class="node-description" style="margin-top: 1.5rem; font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap">
                {morphed_description}
            </div>

            // Wisdom Bridge
            {move || node.wisdom_bridge.as_ref().map(|w| {
                view! {
                    <div class="wisdom-bridge-box" style="margin: 1.5rem 0; padding: 1rem; background: var(--surface-low); border-left: 4px solid var(--accent); border-radius: 4px">
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: var(--accent); font-weight: 700">"Ancestral Root"</div>
                        <h5 style="margin: 0.2rem 0">{&w.tradition}": "{&w.pattern_name}</h5>
                        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; font-style: italic">{&w.description}</p>
                    </div>
                use crate::components::hardware::{HardwareScanner, PresenceValidator, SafetyGuard};
                use crate::components::voice_ops::VoiceCommandCenter;

                // ...
                            // Artifact Ledger (Proof of Craft)
                            <section class="artifact-ledger" style="margin-top: 2rem; padding-top: 2rem; border-top: 1px solid var(--border)">
                                <h4>"Artifact Ledger"</h4>
                                <p style="font-size: 0.75rem; color: var(--text-tertiary)">"Ground your digital record with physical proof (Proof-of-Craft)."</p>

                                {
                                    let nid = node.id.clone();
                                    let ntype = node.node_type.clone();
                                    move || if ntype == "Physical" {
                                        view! {
                                            <PresenceValidator 
                                                node_id=nid.clone() 
                                                on_verified=move || {
                                                    // Trigger verified_presence callback
                                                } 
                                            />
                                            <VoiceCommandCenter />
                                        }.into_any()
                                    } else {
                                        view! { <span></span> }.into_any()
                                    }
                                }

                <input 
                    type="text" 
                    placeholder="Link to GitHub, CAD, or Video..." 
                    style="width: 100%; margin-top: 0.5rem"
                    prop:value=artifact_url
                    on:input=move |ev| set_artifact_url.set(event_target_value(&ev))
                />
                <button class="btn-sm btn-outline" style="margin-top: 0.5rem; width: 100%">"Upload Artifact"</button>
            </section>

            // Kinetic Proof: IMU-based Verification
            {
                let nid = node.id.clone();
                move || if node.node_type == "Physical" {
                    view! {
                        <section class="kinetic-proof" style="margin-top: 1.5rem; padding: 1.25rem; background: var(--surface-high); border: 1px solid var(--primary-low); border-radius: 8px">
                            <h4 style="margin: 0; color: var(--primary)">"Kinetic Proof (IMU)"</h4>
                            <p style="font-size: 0.7rem; margin: 0.5rem 0 1rem 0">
                                "Verify manual labor via motion sensors. No camera required."
                            </p>
                            <button class="btn-sm btn-primary" style="width: 100%">
                                "\u{1F3CB}\u{FE0F} Start Motion Tracking"
                            </button>
                        </section>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }
            }

            // Embodied Energy Profile
            {move || if !node.abstract_resources.is_empty() {
                view! {
                    <section class="energy-profile" style="margin-top: 1.5rem; padding: 1rem; background: var(--surface-low); border-radius: 8px">
                        <div style="font-size: 0.7rem; text-transform: uppercase; color: var(--text-tertiary)">"Thermodynamic Cost"</div>
                        <div style="display: flex; justify-content: space-between; margin-top: 0.5rem; font-size: 0.8rem">
                            <span>"Embodied Energy (MJ):"</span>
                            <span style="font-weight: 700; color: var(--warning)">
                                {node.abstract_resources.iter().map(|r| r.embodied_energy_joules).sum::<u64>() / 1000000}" MJ"
                            </span>
                        </div>
                    </section>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}

            // Gaia Oracle: Verification by Reality
            {
                let gaia = node.gaia_target.clone();
                move || gaia.as_ref().map(|g| {
                    view! {
                        <section class="gaia-verification" style="margin-top: 2rem; padding: 1.5rem; background: linear-gradient(135deg, var(--surface-high), var(--success-low)); border: 2px solid var(--success); border-radius: 8px">
                            <h4 style="margin: 0; color: var(--success)">"Gaia Oracle: Active"</h4>
                            <p style="font-size: 0.75rem; line-height: 1.4; margin: 0.5rem 0 1rem 0">
                                "Mastery is pending physical confirmation of environmental flourishing."
                            </p>
                            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700">
                                <span>{format!("Target: {}", g.metric_type)}</span>
                                <span style="color: var(--success)">"+{:.1}% Delta"</span>
                            </div>
                            <div class="mining-bar" style="height: 8px; background: var(--surface); border-radius: 4px; margin-top: 0.5rem; overflow: hidden">
                                <div class="mining-bar-fill" style="width: 65%; height: 100%; background: var(--success)"></div>
                            </div>
                        </section>
                    }
                })
            }

            // Holonic Lab: Capability Grants (Proof of Access)
            {
                let node_title = node.title.clone();
                let tags = node.tags.clone();
                let mastered = status() == ProgressStatus::Mastered;
                move || if mastered && (tags.contains(&"solar".to_string()) || tags.contains(&"mfg".to_string()) || tags.contains(&"robotics".to_string())) {
                    view! {
                        <SafetyGuard>
                            <section class="hardware-grants" style="margin-top: 2rem; padding: 1.5rem; background: var(--surface-high); border: 2px solid var(--accent); border-radius: 8px">
                                <h4 style="margin: 0; color: var(--accent)">"Holonic Lab Access"</h4>
                                <p style="font-size: 0.75rem; line-height: 1.4; margin: 0.5rem 0 1rem 0">
                                    "As a master of this node, you are eligible for a physical hardware control key."
                                </p>
                                <HardwareScanner 
                                    device_type=node_title.clone() 
                                    on_linked=move |_grant| {
                                        // Store capability grant locally
                                    } 
                                />
                            </section>
                        </SafetyGuard>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }
            }

            // Status Control
            <footer style="margin-top: 2rem; display: flex; gap: 0.5rem">
                <button 
                    class=move || if status() == ProgressStatus::Mastered { "btn-primary active" } else { "btn-outline" }
                    on:click=move |_| set_progress.update(|p| p.set_status(&node_id, ProgressStatus::Mastered))
                >
                    "Mark Mastered"
                </button>

                <button
                    class="btn-outline"
                    style="font-size: 0.8rem; border-color: var(--primary)"
                    title="Propose a local translation or idiom for this topic (e.g. Zulu, Sotho, Slang)"
                    on:click=move |_| {
                        // Trigger community localization modal (Isibaya Translation)
                    }
                >
                    "\u{1F30D} Localize"
                </button>
            </footer>
        </div>
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Isibaya Governance — Local DAO interface for merit weighting and hardware access.

use leptos::prelude::*;
use crate::curriculum::{curriculum_graph, Subject};

#[component]
pub fn GovernancePage() -> impl IntoView {
    let (voting_active, set_voting_active) = signal(false);
    
    view! {
        <div class="governance-page">
            <header class="gov-header">
                <h2>"Isibaya Governance"</h2>
                <p class="subtitle">"Direct community control over the local merit economy."</p>
            </header>

            <div class="gov-grid">
                <section class="merit-steering">
                    <h3>"Local Merit Weighting"</h3>
                    <p style="font-size: 0.8rem; color: var(--text-tertiary)">
                        "Vote to prioritize specific curriculum tracks based on local physical needs."
                    </p>
                    
                    <div class="steering-list" style="margin-top: 1.5rem">
                        {move || {
                            let graph = curriculum_graph();
                            graph.subjects().into_iter().take(5).map(|s| {
                                view! {
                                    <div class="steering-row" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding: 1rem; background: var(--surface-low); border-radius: 8px">
                                        <span style="font-weight: 600">{s}</span>
                                        <div style="display: flex; align-items: center; gap: 1rem">
                                            <span style="font-family: monospace; color: var(--success)">"1.0x"</span>
                                            <button class="btn-sm btn-outline" on:click=move |_| set_voting_active.set(true)>"Vote \u{2191}"</button>
                                        </div>
                                    </div>
                                }
                            }).collect_view()
                        }}
                    </div>
                </section>

                <section class="hardware-council">
                    <h3>"Hardware Council"</h3>
                    <div class="council-card" style="padding: 1.5rem; background: var(--surface-high); border-radius: 12px; border: 1px solid var(--accent-low)">
                        <div style="font-size: 0.7rem; color: var(--accent); font-weight: 800; text-transform: uppercase">"Active Proposals"</div>
                        <h4 style="margin: 0.5rem 0">"Grant Access: Warehouse CNC-01"</h4>
                        <p style="font-size: 0.8rem; line-height: 1.5">
                            "Should students with 'Digital Fabrication' mastery be granted automatic Cap-Grants for the industrial mill?"
                        </p>
                        <div style="display: flex; gap: 0.5rem; margin-top: 1rem">
                            <button class="btn-sm btn-primary" style="flex: 1">"AYE (82%)"</button>
                            <button class="btn-sm btn-outline" style="flex: 1">"NAY (18%)"</button>
                        </div>
                    </div>
                </section>

                <section class="epistemic-markets" style="grid-column: span 2; margin-top: 2rem; padding: 2rem; background: var(--surface-high); border-radius: 12px; border: 1px solid var(--primary-low)">
                    <div style="display: flex; justify-content: space-between; align-items: center">
                        <h3>"Epistemic Markets (Truth Staking)"</h3>
                        <span class="badge" style="background: var(--primary); color: var(--text-on-primary)">"Quadratic Staking Active"</span>
                    </div>
                    <p class="subtitle">"Stake your reputation on physical outcomes. Accuracy increases your governing weight."</p>

                    <div class="market-card" style="margin-top: 1.5rem; padding: 1.5rem; background: var(--surface); border-radius: 8px; border: 1px solid var(--border)">
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; text-transform: uppercase; color: var(--text-tertiary)">
                            <span>"Predictive Metric: VAAL-RIVER-TURBIDITY"</span>
                            <span>"Closes: 24h"</span>
                        </div>
                        <h4 style="margin: 0.5rem 0">"Proposal: Biomimetic Water Filter Prototype"</h4>
                        <p style="font-size: 0.85rem">"Will this design reduce turbidity by >15% within 48 hours?"</p>
                        
                        <div class="prediction-distribution" style="margin: 1.5rem 0; height: 40px; display: flex; align-items: flex-end; gap: 2px">
                            {(0..20).map(|i| {
                                let height = if i < 10 { i * 4 } else { (20 - i) * 4 };
                                view! { <div style=format!("flex: 1; height: {}%; background: var(--primary-low); border-radius: 2px", height)></div> }
                            }).collect_view()}
                        </div>
                        
                        <div style="display: flex; gap: 1rem; align-items: center">
                            <input type="number" placeholder="Stake PHI..." style="flex: 1; padding: 0.5rem; border-radius: 4px; border: 1px solid var(--border); background: var(--surface-low)" />
                            <button class="btn-primary">"Submit Prediction"</button>
                        </div>
                        <div style="font-size: 0.6rem; color: var(--text-tertiary); margin-top: 0.8rem; text-align: center">
                            "Influence = sqrt(Stake). High-conviction minority payouts active."
                        </div>
                    </div>
                </section>
            </div>

            {move || if voting_active.get() {
                view! {
                    <div class="gov-modal-overlay">
                        <div class="gov-modal">
                            <h4>"Cast Your Merit Vote"</h4>
                            <p>"Allocating your Liquid Reputation stake to steer the community's learning velocity."</p>
                            <input type="range" min="0" max="100" style="width: 100%" />
                            <div style="display: flex; justify-content: flex-end; gap: 0.5rem; margin-top: 1.5rem">
                                <button class="btn-outline" on:click=move |_| set_voting_active.set(false)>"Cancel"</button>
                                <button class="btn-primary" on:click=move |_| set_voting_active.set(false)>"Commit Vote"</button>
                            </div>
                        </div>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}
        </div>
    }
}

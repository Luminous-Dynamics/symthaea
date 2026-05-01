// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::contexts::commons_context::use_commons;
use crate::contexts::commons_actions;

#[component]
pub fn FoodPage() -> impl IntoView {
    let commons = use_commons();
    view! {
        <div class="food-page" data-page="food" role="main">
            <h1 class="page-title">"Food"</h1>
            <p class="page-subtitle">"grow, share, nourish — food sovereignty begins here"</p>

            <section data-section="plots" aria-label="community plots">
                <h2 class="section-title">"Growing spaces"</h2>
                <div class="plot-grid" role="list">
                    {move || commons.plots.get().into_iter().map(|p| {
                        let steward = p.steward_did.split(':').last().unwrap_or(&p.steward_did).to_string();
                        view! {
                            <div class="plot-card" data-plot-hash=p.hash.clone() data-area=format!("{:.0}", p.area_sqm) role="listitem">
                                <h3 class="plot-name">{p.name}</h3>
                                <div class="plot-meta">
                                    <span>{format!("{:.0}m² — {} crops", p.area_sqm, p.crop_count)}</span>
                                    <span>{format!("steward: {steward}")}</span>
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>

            <section data-section="soil" aria-label="soil health and compost">
                <h2 class="section-title">"Soil & Compost"</h2>
                <p class="section-subtitle">"regenerating the substrate of life through Terra Preta"</p>
                
                <div class="compost-grid" role="list">
                    {move || commons.compost_batches.get().into_iter().map(|b| {
                        view! {
                            <div class="compost-card" data-compost-hash=b.hash.clone() data-status=b.status.to_lowercase() role="listitem">
                                <h3 class="compost-name">{b.name}</h3>
                                <div class="compost-type">{b.method}</div>
                                <div class="compost-metrics">
                                    <div class="metric">
                                        <span class="label">"Temp"</span>
                                        <span class="value">{format!("{:.1}°C", b.temperature_c)}</span>
                                    </div>
                                    <div class="metric">
                                        <span class="label">"pH"</span>
                                        <span class="value">{format!("{:.1}", b.ph)}</span>
                                    </div>
                                    <div class="metric highlight">
                                        <span class="label">"CO2e Seq."</span>
                                        <span class="value">{format!("{:.0}kg", b.carbon_seq_est)}</span>
                                    </div>
                                </div>
                                <div class="compost-reward-box">
                                    <span class="reward-label">"Potential Reward:"</span>
                                    <span class="reward-value">{format!("{:.1} TEND", b.reward_tend)}</span>
                                </div>
                                <div class="compost-footer">
                                    <div class="compost-status-tag">{b.status.clone()}</div>
                                    {b.can_claim.then(|| {
                                        let hash = b.hash.clone();
                                        view! {
                                            <button
                                                class="btn-claim"
                                                on:click=move |_| commons_actions::claim_compost_reward(hash.clone())
                                            >
                                                "Claim TEND"
                                            </button>
                                        }
                                    })}
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>

                <div class="recipe-promo">
                    <h3>"Community Soil Recipes"</h3>
                    <p>"Discover traditional practices for high-integrity Biochar charging and Bokashi fermentation."</p>
                    <button class="btn-secondary">"View Recipes"</button>
                </div>
            </section>

            <section data-section="markets" aria-label="food markets">
                <h2 class="section-title">"Markets & food banks"</h2>
                <div class="market-grid" role="list">
                    {move || commons.markets.get().into_iter().map(|m| {
                        let type_label = m.market_type.label().to_string();
                        let type_label2 = type_label.clone();
                        view! {
                            <div class="market-card" data-market-hash=m.hash.clone() data-type=type_label role="listitem">
                                <h3 class="market-name">{m.name}</h3>
                                <div class="market-meta">
                                    <span>{type_label2}</span>
                                    <span>{format!("{} listings", m.listing_count)}</span>
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>

            <section data-section="listings" aria-label="available food">
                <h2 class="section-title">"Available now"</h2>
                <div class="food-list" role="list">
                    {move || commons.food_listings.get().into_iter().filter(|f| f.available).map(|f| {
                        let producer = f.producer_did.split(':').last().unwrap_or(&f.producer_did).to_string();
                        let price = if f.price_per_kg == 0.0 { "free / donation".to_string() } else { format!("{:.2} TEND/kg", f.price_per_kg) };
                        view! {
                            <div class="food-card" data-food-hash=f.hash.clone() data-organic=f.organic.to_string() role="listitem">
                                <h3 class="food-name">{f.product_name}</h3>
                                <div class="food-meta">
                                    <span>{format!("{:.1}kg", f.quantity_kg)}</span>
                                    <span>{price}</span>
                                    {f.organic.then(|| view! { <span class="organic-badge">"organic"</span> })}
                                    <span>{format!("from {producer}")}</span>
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </section>
        </div>
    }
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::data::types::SelectedItem;
use crate::state::globe_state::GlobeState;

#[component]
pub fn InfoPanel() -> impl IntoView {
    let globe_state = expect_context::<GlobeState>();

    let visible = move || globe_state.selected.read().is_some();
    let close = move |_| globe_state.selected.set(None);

    view! {
        <div class=move || if visible() { "info-panel" } else { "info-panel hidden" }>
            <button class="close-btn" on:click=close>{"\u{2715}"}</button>
            {move || {
                let selected = globe_state.selected.read();
                match selected.as_ref() {
                    Some(SelectedItem::GeothermalNode(n)) => view! {
                        <div>
                            <h2>{n.name.clone()}</h2>
                            <div class="subtitle">"Geothermal Node"</div>
                            <div class="stat-row"><span class="stat-label">"Region"</span><span class="stat-value">{n.region.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Capacity"</span><span class="stat-value">{format!("{} MW", n.capacity_mw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Temperature"</span><span class="stat-value">{format!("{}°C", n.temperature_c)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Type"</span><span class="stat-value">{n.node_type.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Status"</span><span class="stat-value">{n.status.clone()}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::MaglevCorridor(c)) => view! {
                        <div>
                            <h2>{c.name.clone()}</h2>
                            <div class="subtitle">"Maglev Corridor"</div>
                            <div class="stat-row"><span class="stat-label">"From"</span><span class="stat-value">{c.from_name.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"To"</span><span class="stat-value">{c.to_name.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Distance"</span><span class="stat-value">{format!("{:.0} km", c.distance_km)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Travel Time"</span><span class="stat-value">{format!("{:.0} min", c.travel_time_min)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Type"</span><span class="stat-value">{if c.submarine { "Submarine" } else { "Land" }}</span></div>
                            <div class="stat-row"><span class="stat-label">"Seismic Risk"</span><span class="stat-value">{c.seismic_risk.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Cost"</span><span class="stat-value">{format!("${:.0}B", c.cost_billion_usd)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Capacity"</span><span class="stat-value">{format!("{} pax/hr", c.capacity_pax_hr)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Geothermal"</span><span class="stat-value">{if c.geothermal_powered { "Yes" } else { "No" }}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::ResontiaVault(v)) => view! {
                        <div>
                            <h2>{v.name.clone()}</h2>
                            <div class="subtitle">"Resontia Vault"</div>
                            <div class="stat-row"><span class="stat-label">"Capacity"</span><span class="stat-value">{format!("{} people", v.capacity_people)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Heat Rejection"</span><span class="stat-value">{format!("{:.0} MW", v.heat_rejection_mw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Blast Doors"</span><span class="stat-value">{v.blast_doors.to_string()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Status"</span><span class="stat-value">{v.status.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Terra Lumina"</span><span class="stat-value">{v.terra_lumina_id.clone()}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::TerraLuminaSite(s)) => view! {
                        <div>
                            <h2>{s.name.clone()}</h2>
                            <div class="subtitle">{format!("{} | {} | Score {}", s.country, s.tier, s.score)}</div>
                            <div class="stat-row"><span class="stat-label">"Geothermal"</span><span class="stat-value">{format!("{:.1} GW", s.geothermal_gw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Solar"</span><span class="stat-value">{format!("{:.1} GW", s.solar_gw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Hydro"</span><span class="stat-value">{format!("{:.1} GW", s.hydro_gw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Total Renewable"</span><span class="stat-value">{format!("{:.1} GW", s.total_renewable_gw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Phase 1"</span><span class="stat-value">{format!("\u{20ac}{:.0}B", s.phase1_billion_eur)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Total Cost"</span><span class="stat-value">{format!("\u{20ac}{:.0}B", s.total_billion_eur)}</span></div>
                            <div class="stat-row"><span class="stat-label">"IRR"</span><span class="stat-value">{format!("{:.1}%", s.irr_percent)}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::Site(s)) => view! {
                        <div>
                            <h2>{s.name.clone()}</h2>
                            <div class="subtitle">{format!("{:?} | {}", s.energy_type, s.country)}</div>
                            <div class="stat-row"><span class="stat-label">"Capacity"</span><span class="stat-value">{format!("{:.0} MW", s.capacity_mw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Status"</span><span class="stat-value">{s.status.clone()}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::EarthRegion(r)) => view! {
                        <div>
                            <h2>{r.name.clone()}</h2>
                            <div class="subtitle">"Earth Region"</div>
                            <div class="stat-row"><span class="stat-label">"Population"</span><span class="stat-value">{format!("{:.0}M", r.population_m)}</span></div>
                            <div class="stat-row"><span class="stat-label">"GDP/capita"</span><span class="stat-value">{format!("${:.0}", r.gdp_per_capita)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Education"</span><span class="stat-value">{format!("{:.0}%", r.education_index * 100.0)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Phi (mean)"</span><span class="stat-value">{format!("{:.2}", r.phi_mean)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Climate Risk"</span><span class="stat-value">{format!("{:.0}%", r.climate_vulnerability * 100.0)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Infrastructure"</span><span class="stat-value">{format!("{:.0}%", r.infrastructure * 100.0)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Spaceport"</span><span class="stat-value">{if r.spaceport { "Yes" } else { "No" }}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::NuclearSite(n)) => view! {
                        <div>
                            <h2>{n.name.clone()}</h2>
                            <div class="subtitle">{format!("{} | {}", n.reactor_type.label(), n.country)}</div>
                            <div class="stat-row"><span class="stat-label">"Capacity"</span><span class="stat-value">{format!("{:.0} MW", n.capacity_mw)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Operator"</span><span class="stat-value">{n.operator.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Status"</span><span class="stat-value">{n.status.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Commissioned"</span><span class="stat-value">{n.commission_year.to_string()}</span></div>
                        </div>
                    }.into_any(),
                    Some(SelectedItem::FossilDeposit(d)) => view! {
                        <div>
                            <h2>{d.name.clone()}</h2>
                            <div class="subtitle">{format!("{:?} | {}", d.fuel_type, d.country)}</div>
                            <div class="stat-row"><span class="stat-label">"Reserves"</span><span class="stat-value">{format!("{:.0} Mboe", d.proven_reserves_mboe)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Production"</span><span class="stat-value">{format!("{:.0} Mboe/yr", d.annual_production_mboe)}</span></div>
                            <div class="stat-row"><span class="stat-label">"Status"</span><span class="stat-value">{d.status.clone()}</span></div>
                            <div class="stat-row"><span class="stat-label">"Discovered"</span><span class="stat-value">{d.discovery_year.to_string()}</span></div>
                            {terra_atlas_core::economics::compute_eroi(d).map(|e| {
                                let tier = terra_atlas_core::economics::EroiTier::from_eroi(e);
                                view! {
                                    <div class="stat-row"><span class="stat-label">"EROI"</span><span class="stat-value">{format!("{:.0}:1 ({})", e, tier.label())}</span></div>
                                }
                            })}
                        </div>
                    }.into_any(),
                    None => view! { <div /> }.into_any(),
                }
            }}
        </div>
    }
}

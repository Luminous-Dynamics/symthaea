// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::data::types::HoverInfo;
use crate::state::globe_state::GlobeState;

#[component]
pub fn Tooltip() -> impl IntoView {
    let globe_state = expect_context::<GlobeState>();

    let visible = move || globe_state.hovered.read().is_some();
    let pos = move || globe_state.mouse_pos.get();

    let content = move || {
        let hover = globe_state.hovered.read();
        match hover.as_ref() {
            Some(HoverInfo::GeothermalNode(n)) => {
                format!("{}\n{} MW | {}°C | {}", n.name, n.capacity_mw, n.temperature_c, n.status)
            }
            Some(HoverInfo::MaglevCorridor(c)) => {
                let mode = if c.submarine { "Submarine" } else { "Land" };
                format!("{}\n{:.0} km | {} min | {} | ${:.0}B", c.name, c.distance_km, c.travel_time_min, mode, c.cost_billion_usd)
            }
            Some(HoverInfo::ResontiaVault(v)) => {
                format!("{}\n{} people | {:.0} MW heat rejection | {}", v.name, v.capacity_people, v.heat_rejection_mw, v.status)
            }
            Some(HoverInfo::TerraLuminaSite(s)) => {
                format!("{} ({})\nScore: {} | {:.1} GW renewable | IRR {:.1}%", s.name, s.country, s.score, s.total_renewable_gw, s.irr_percent)
            }
            Some(HoverInfo::Site(s)) => {
                format!("{}\n{:?} | {:.0} MW", s.name, s.energy_type, s.capacity_mw)
            }
            Some(HoverInfo::EarthRegion(r)) => {
                format!("{}\nPop {:.0}M | GDP ${:.0}k | Φ {:.2} | Climate Risk {:.0}%",
                    r.name, r.population_m, r.gdp_per_capita / 1000.0, r.phi_mean, r.climate_vulnerability * 100.0)
            }
            None => String::new(),
        }
    };

    view! {
        <div
            class="tooltip"
            style=move || {
                let (x, y) = pos();
                let vis = if visible() { "1" } else { "0" };
                format!("left: {}px; top: {}px; opacity: {vis}", x, y)
            }
        >
            <div class="tooltip-name">
                {move || {
                    let text = content();
                    let mut lines = text.lines();
                    lines.next().unwrap_or("").to_string()
                }}
            </div>
            <div class="tooltip-detail">
                {move || {
                    let text = content();
                    let lines: Vec<&str> = text.lines().skip(1).collect();
                    lines.join(" | ")
                }}
            </div>
        </div>
    }
}

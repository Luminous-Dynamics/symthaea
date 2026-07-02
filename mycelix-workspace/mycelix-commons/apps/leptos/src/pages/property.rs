// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn PropertyPage() -> impl IntoView {
    view! {
        <div class="page property-page">
            <h1>"Property Registry"</h1>
            <p class="page-desc">"Community-owned land and property records on the DHT."</p>

            <div class="filter-bar">
                <input type="text" placeholder="Search properties..." class="search-input" />
                <select class="filter-select">
                    <option>"All Types"</option>
                    <option>"Residential"</option>
                    <option>"Agricultural"</option>
                    <option>"Community Space"</option>
                    <option>"Conservation"</option>
                </select>
            </div>

            <div class="property-list">
                <div class="property-card">
                    <h3>"Riverside Community Garden"</h3>
                    <span class="property-type">"Agricultural"</span>
                    <p>"2.4 hectares — Managed by Riverside Cooperative"</p>
                    <div class="property-meta">
                        <span>"Stewards: 12"</span>
                        <span>"Since: 2025-03"</span>
                    </div>
                </div>
                <div class="property-card">
                    <h3>"Makers Workshop"</h3>
                    <span class="property-type">"Community Space"</span>
                    <p>"450 sqm — Open workshop with tools library"</p>
                    <div class="property-meta">
                        <span>"Stewards: 8"</span>
                        <span>"Since: 2024-11"</span>
                    </div>
                </div>
                <div class="property-card">
                    <h3>"Hillside Housing Block A"</h3>
                    <span class="property-type">"Residential"</span>
                    <p>"24 units — Housing cooperative with shared amenities"</p>
                    <div class="property-meta">
                        <span>"Residents: 67"</span>
                        <span>"Since: 2024-06"</span>
                    </div>
                </div>
            </div>
        </div>
    }
}

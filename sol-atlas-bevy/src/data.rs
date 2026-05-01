// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Static data loading for the Bevy globe.
//! Includes JSON at compile time from sol-atlas-leptos/assets/data/.

use sol_atlas_core::types::LoadedData;

const SITES_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/sites-clustered.json");
const MAGLEV_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/maglev-network.json");
const VAULTS_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/resontia-vaults.json");
const TERRA_LUMINA_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/terra-lumina-sites.json");
const REGIONS_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/earth-regions.json");
const SUPPLY_ROUTES_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/supply-routes.json");
const CLIMATE_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/climate-projects.json");
const INFRA_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/infrastructure.json");
const FOSSIL_DEPOSITS_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/fossil-deposits.json");
const NUCLEAR_SITES_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/nuclear-sites.json");
const EARTHQUAKES_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/usgs-earthquakes.json");
const FIRES_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/nasa-firms.json");
const STORMS_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/nasa-eonet.json");
const VOLCANOES_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/volcanoes.json");
const MAJOR_CITIES_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/major-cities-1m.json");
const CHOKEPOINTS_JSON: &str = include_str!("../../sol-atlas-leptos/assets/data/chokepoints.json");
const CRITICAL_INFRA_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/critical-infrastructure.json");
const SHIPPING_LANES_JSON: &str =
    include_str!("../../sol-atlas-leptos/assets/data/shipping-lanes-simplified.json");

/// Load simplified shipping lane routes (Vec of Vec of [lon, lat]).
pub fn load_shipping_lanes() -> Vec<Vec<[f64; 2]>> {
    sol_atlas_core::data::parse_shipping_lanes(SHIPPING_LANES_JSON)
}

/// Load all static datasets at compile time.
pub fn load_all() -> LoadedData {
    sol_atlas_core::data::load_all(
        SITES_JSON,
        MAGLEV_JSON,
        VAULTS_JSON,
        TERRA_LUMINA_JSON,
        REGIONS_JSON,
        SUPPLY_ROUTES_JSON,
        CLIMATE_JSON,
        INFRA_JSON,
        FOSSIL_DEPOSITS_JSON,
        NUCLEAR_SITES_JSON,
        EARTHQUAKES_JSON,
        FIRES_JSON,
        STORMS_JSON,
        VOLCANOES_JSON,
        MAJOR_CITIES_JSON,
        CHOKEPOINTS_JSON,
        CRITICAL_INFRA_JSON,
    )
}

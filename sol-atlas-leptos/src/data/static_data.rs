// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Static JSON data loading — delegates parsing to terra-atlas-core.

use sol_atlas_core::types::LoadedData;

const SITES_JSON: &str = include_str!("../../assets/data/sites-clustered.json");
const MAGLEV_JSON: &str = include_str!("../../assets/data/maglev-network.json");
const VAULTS_JSON: &str = include_str!("../../assets/data/resontia-vaults.json");
const TERRA_LUMINA_JSON: &str = include_str!("../../assets/data/terra-lumina-sites.json");
const REGIONS_JSON: &str = include_str!("../../assets/data/earth-regions.json");
const SUPPLY_ROUTES_JSON: &str = include_str!("../../assets/data/supply-routes.json");
const CLIMATE_JSON: &str = include_str!("../../assets/data/climate-projects.json");
const INFRA_JSON: &str = include_str!("../../assets/data/infrastructure.json");
const FOSSIL_DEPOSITS_JSON: &str = include_str!("../../assets/data/fossil-deposits.json");
const NUCLEAR_SITES_JSON: &str = include_str!("../../assets/data/nuclear-sites.json");

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
    )
}

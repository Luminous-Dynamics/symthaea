// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Holochain zome call wrappers for Terra Atlas data.
// These will be wired to real zome calls via mycelix-leptos-client
// when the atlas DNA is deployed. For now, they return empty vecs
// (static data is the fallback).

use super::types::*;

pub async fn fetch_all_sites() -> Vec<Site> {
    // TODO: call atlas-sites/list_sites via BrowserWsTransport
    log::debug!("Holochain: fetch_all_sites (stub — using static data)");
    Vec::new()
}

pub async fn fetch_geothermal_nodes() -> Vec<GeothermalNode> {
    // TODO: call atlas-infrastructure/list_geothermal_nodes
    log::debug!("Holochain: fetch_geothermal_nodes (stub)");
    Vec::new()
}

pub async fn fetch_maglev_corridors() -> Vec<MaglevCorridor> {
    // TODO: call atlas-infrastructure/list_corridors
    log::debug!("Holochain: fetch_maglev_corridors (stub)");
    Vec::new()
}

pub async fn fetch_vaults() -> Vec<ResontiaVault> {
    // TODO: call atlas-infrastructure/list_vaults
    log::debug!("Holochain: fetch_vaults (stub)");
    Vec::new()
}

pub async fn fetch_terra_lumina_sites() -> Vec<TerraLuminaSite> {
    // TODO: call atlas-infrastructure/list_terra_lumina_sites
    log::debug!("Holochain: fetch_terra_lumina_sites (stub)");
    Vec::new()
}

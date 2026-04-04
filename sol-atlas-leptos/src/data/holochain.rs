// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Holochain zome call wrappers for Sol Atlas data.
//!
//! When compiled with `--features holochain`, these functions call the
//! atlas DNA via `HolochainCtx`. Without the feature, they return empty
//! vecs (static data is the fallback).

use sol_atlas_core::types::*;

/// Fetch all energy sites from atlas-sites zome.
pub async fn fetch_all_sites() -> Vec<Site> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<Site>>("atlas", "atlas_sites", "list_sites", &()).await {
            Ok(sites) => {
                log::info!("[holochain] Fetched {} sites from DHT", sites.len());
                return sites;
            }
            Err(e) => log::warn!("[holochain] fetch_all_sites failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch geothermal nodes from atlas-infrastructure zome.
pub async fn fetch_geothermal_nodes() -> Vec<GeothermalNode> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<GeothermalNode>>("atlas", "atlas_infrastructure", "list_geothermal_nodes", &()).await {
            Ok(nodes) => {
                log::info!("[holochain] Fetched {} geothermal nodes from DHT", nodes.len());
                return nodes;
            }
            Err(e) => log::warn!("[holochain] fetch_geothermal_nodes failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch maglev corridors from atlas-infrastructure zome.
pub async fn fetch_maglev_corridors() -> Vec<MaglevCorridor> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<MaglevCorridor>>("atlas", "atlas_infrastructure", "list_corridors", &()).await {
            Ok(corridors) => {
                log::info!("[holochain] Fetched {} corridors from DHT", corridors.len());
                return corridors;
            }
            Err(e) => log::warn!("[holochain] fetch_maglev_corridors failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch Resontia vaults from atlas-infrastructure zome.
pub async fn fetch_vaults() -> Vec<ResontiaVault> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<ResontiaVault>>("atlas", "atlas_infrastructure", "list_vaults", &()).await {
            Ok(vaults) => {
                log::info!("[holochain] Fetched {} vaults from DHT", vaults.len());
                return vaults;
            }
            Err(e) => log::warn!("[holochain] fetch_vaults failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch Terra Lumina sites from atlas-infrastructure zome.
pub async fn fetch_terra_lumina_sites() -> Vec<TerraLuminaSite> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<TerraLuminaSite>>("atlas", "atlas_infrastructure", "list_terra_lumina_sites", &()).await {
            Ok(sites) => {
                log::info!("[holochain] Fetched {} terra lumina sites from DHT", sites.len());
                return sites;
            }
            Err(e) => log::warn!("[holochain] fetch_terra_lumina_sites failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch fossil deposits from atlas-infrastructure zome.
pub async fn fetch_fossil_deposits() -> Vec<FossilDeposit> {
    #[cfg(feature = "holochain")]
    {
        let ctx = mycelix_leptos_core::holochain_provider::use_holochain();
        match ctx.call_zome::<(), Vec<FossilDeposit>>("atlas", "atlas_infrastructure", "list_fossil_deposits", &()).await {
            Ok(deposits) => {
                log::info!("[holochain] Fetched {} fossil deposits from DHT", deposits.len());
                return deposits;
            }
            Err(e) => log::warn!("[holochain] fetch_fossil_deposits failed: {e}"),
        }
    }
    Vec::new()
}

/// Fetch nuclear sites (future — not yet in atlas DNA).
pub async fn fetch_nuclear_sites() -> Vec<NuclearSite> {
    log::debug!("[holochain] fetch_nuclear_sites (stub — using static data)");
    Vec::new()
}

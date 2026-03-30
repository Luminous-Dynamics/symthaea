// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::data::types::*;

#[derive(Clone)]
pub struct DataState {
    pub sites: RwSignal<Vec<Site>>,
    pub geothermal_nodes: RwSignal<Vec<GeothermalNode>>,
    pub maglev_corridors: RwSignal<Vec<MaglevCorridor>>,
    pub resontia_vaults: RwSignal<Vec<ResontiaVault>>,
    pub terra_lumina_sites: RwSignal<Vec<TerraLuminaSite>>,
    pub earth_regions: RwSignal<Vec<EarthRegion>>,
    pub supply_routes: RwSignal<Vec<SupplyRoute>>,
    pub climate_projects: RwSignal<Vec<ClimateProject>>,
    pub emergency_shelters: RwSignal<Vec<EmergencyShelter>>,
    pub health_facilities: RwSignal<Vec<HealthFacility>>,
    pub robotics_dispatch: RwSignal<Vec<RoboticsDispatch>>,
}

impl DataState {
    pub fn new() -> Self {
        Self {
            sites: RwSignal::new(Vec::new()),
            geothermal_nodes: RwSignal::new(Vec::new()),
            maglev_corridors: RwSignal::new(Vec::new()),
            resontia_vaults: RwSignal::new(Vec::new()),
            terra_lumina_sites: RwSignal::new(Vec::new()),
            earth_regions: RwSignal::new(Vec::new()),
            supply_routes: RwSignal::new(Vec::new()),
            climate_projects: RwSignal::new(Vec::new()),
            emergency_shelters: RwSignal::new(Vec::new()),
            health_facilities: RwSignal::new(Vec::new()),
            robotics_dispatch: RwSignal::new(Vec::new()),
        }
    }

    pub fn set_all(&self, data: LoadedData) {
        self.sites.set(data.sites);
        self.geothermal_nodes.set(data.geothermal_nodes);
        self.maglev_corridors.set(data.maglev_corridors);
        self.resontia_vaults.set(data.resontia_vaults);
        self.terra_lumina_sites.set(data.terra_lumina_sites);
        self.earth_regions.set(data.earth_regions);
        self.supply_routes.set(data.supply_routes);
        self.climate_projects.set(data.climate_projects);
        self.emergency_shelters.set(data.emergency_shelters);
        self.health_facilities.set(data.health_facilities);
        self.robotics_dispatch.set(data.robotics_dispatch);
    }
}

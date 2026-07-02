// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;

pub mod city_scale_logic {
    use super::*;
    /// Placeholder for `GamePhase::CityScale` ("Phase 11: City-Scale Governance
    /// Demonstration", see symtropy/src/resources.rs). Intentionally a no-op:
    /// the ecology-sim stack it should eventually drive
    /// (symtropy-lifesim-core -> symtropy-colony / symtropy-mycelium -> symtropy-basin)
    /// exists and builds, but nothing wires it to this plugin or to `GamePhase::CityScale`
    /// yet. Needs a city-scale gameplay design pass before implementing.
    pub struct CityScalePlugin<S> {
        pub state: S,
    }
    impl<S: States> Plugin for CityScalePlugin<S> {
        fn build(&self, _app: &mut App) {}
    }
}

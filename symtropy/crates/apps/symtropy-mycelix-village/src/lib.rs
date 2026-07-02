// SPDX-License-Identifier: AGPL-3.0-or-later

use bevy::prelude::*;

pub mod city_scale_logic {
    use super::*;
    pub struct CityScalePlugin<S> {
        pub state: S,
    }
    impl<S: States> Plugin for CityScalePlugin<S> {
        fn build(&self, _app: &mut App) {}
    }
}

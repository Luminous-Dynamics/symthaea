// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use std::collections::HashSet;

use crate::components::planet_nav::PlanetTarget;
use crate::data::types::{HoverInfo, Layer, SelectedItem};

#[derive(Clone)]
pub struct GlobeState {
    pub active_layers: RwSignal<HashSet<Layer>>,
    pub hovered: RwSignal<Option<HoverInfo>>,
    pub selected: RwSignal<Option<SelectedItem>>,
    pub mouse_pos: RwSignal<(f32, f32)>,
    pub timeline_year: RwSignal<u32>,
    pub show_core: RwSignal<bool>,
    pub focused_planet: RwSignal<Option<PlanetTarget>>,
}

impl GlobeState {
    pub fn new() -> Self {
        // Default: show geothermal and maglev layers
        let mut default_layers = HashSet::new();
        default_layers.insert(Layer::Geothermal);
        default_layers.insert(Layer::Maglev);
        default_layers.insert(Layer::ResontiaVaults);

        Self {
            active_layers: RwSignal::new(default_layers),
            hovered: RwSignal::new(None),
            selected: RwSignal::new(None),
            mouse_pos: RwSignal::new((0.0, 0.0)),
            timeline_year: RwSignal::new(150), // default: Maturation epoch
            show_core: RwSignal::new(false),
            focused_planet: RwSignal::new(None),
        }
    }

    pub fn toggle_layer(&self, layer: Layer) {
        self.active_layers.update(|layers| {
            if layers.contains(&layer) {
                layers.remove(&layer);
            } else {
                layers.insert(layer);
            }
        });
    }

    pub fn is_layer_active(&self, layer: Layer) -> bool {
        self.active_layers.read().contains(&layer)
    }
}

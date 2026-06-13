// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Implementation of the stable cognitive bridge.

use bevy::prelude::*;
use crate::{CognitiveBridge, SymbolicPredicate};
use symtropy_core_stable::hdc::ContinuousHV;

/// A lightweight agent that closes the vision-action loop.
pub struct SpacetimeCrystallizedAgent {
    pub current_vision: Option<ContinuousHV>,
    pub motor_output: Vec<f32>,
}

impl CognitiveBridge for SpacetimeCrystallizedAgent {
    fn update_perception(&mut self, vision_hv: ContinuousHV) {
        self.current_vision = Some(vision_hv);
    }
    
    fn act(&mut self, _dt: f32) -> Vec<f32> {
        if let Some(ref hv) = self.current_vision {
            let val = hv.values.iter().sum::<f32>().clamp(0.0, 1.0);
            self.motor_output = vec![val, val];
        }
        self.motor_output.clone()
    }

    fn query_symbols(&self) -> Vec<Box<dyn SymbolicPredicate>> {
        vec![]
    }
}

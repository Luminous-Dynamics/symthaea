// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use crate::{PersistentFeature, TopologicalFeature};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub dimension: usize,
    pub points: Vec<(f64, f64)>,
}

pub fn export_persistence(features: &[PersistentFeature]) -> Vec<PersistenceDiagram> {
    let mut diagrams = Vec::new();
    for d in 0..=2 {
        let f_type = match d {
            0 => TopologicalFeature::Component,
            1 => TopologicalFeature::Cycle,
            _ => TopologicalFeature::Void,
        };
        let points = features
            .iter()
            .filter(|f| f.feature_type == f_type)
            .map(|f| (f.birth, f.death))
            .collect();
        diagrams.push(PersistenceDiagram {
            dimension: d,
            points,
        });
    }
    diagrams
}

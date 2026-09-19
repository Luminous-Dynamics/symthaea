// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Internal hash support for duplicate prior-art hit-class validation.

use crate::novelty_synthesis::PriorArtHitClass;
use std::hash::{Hash, Hasher};

impl Hash for PriorArtHitClass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let tag: u8 = match self {
            PriorArtHitClass::ExactSubject => 0,
            PriorArtHitClass::CompositionNeighbor => 1,
            PriorArtHitClass::StructuralNeighbor => 2,
            PriorArtHitClass::ProcessNeighbor => 3,
        };
        tag.hash(state);
    }
}

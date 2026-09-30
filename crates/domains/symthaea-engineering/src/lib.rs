// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Engineering reasoning facade for Symthaea.

#![deny(unsafe_code)]

pub mod provenance_binding;
pub mod provenance_graph;

use serde::{Deserialize, Serialize};
use symthaea_broca::{BrocaConfig, BrocaGenerator, ThoughtChannels};
use symthaea_causal_reasoning::causal_calculus::StructuralCausalModel;
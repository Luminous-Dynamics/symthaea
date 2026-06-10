// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear material attribution via HDC similarity and CfC temporal backdating.
//!
//! Genesis Mission Challenge 25: Nuclear Attribution.
//! Encodes isotopic signatures into 16,384-dimensional hypervectors,
//! provides O(1) temporal backdating via CfC closed-form evolution, and
//! matches unknown samples against a reference database for source attribution.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod attribution;
pub mod decay_model;
pub mod encoder;
pub mod isotope;
pub mod safeguards;

pub use attribution::{AttributionResult, NuclearAttributionAgent};
pub use decay_model::{
    AgeEstimate, BackdatedResult, DECAY_HORIZON_LABELS, DECAY_HORIZONS, IsotopeDecayModel,
};
pub use encoder::IsotopicHdcEncoder;
pub use isotope::{IsotopicSignature, NuclearSource};
pub use safeguards::{
    SAFEGUARDS_HORIZON_LABELS, SAFEGUARDS_HORIZONS, SafeguardsFepAction, SafeguardsFepAgent,
    SafeguardsHdcEncoder, SafeguardsPredictor, SafeguardsReading,
};

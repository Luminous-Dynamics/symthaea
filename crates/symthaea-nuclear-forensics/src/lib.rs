//! Nuclear material attribution via HDC similarity and CfC temporal backdating.
//!
//! Genesis Mission Challenge 25: Nuclear Attribution.
//! Encodes isotopic signatures into 16,384-dimensional hypervectors,
//! provides O(1) temporal backdating via CfC closed-form evolution, and
//! matches unknown samples against a reference database for source attribution.

#![deny(unsafe_code)]

pub mod attribution;
pub mod decay_model;
pub mod encoder;
pub mod isotope;

pub use attribution::{AttributionResult, NuclearAttributionAgent};
pub use decay_model::{IsotopeDecayModel, DECAY_HORIZONS, DECAY_HORIZON_LABELS};
pub use encoder::IsotopicHdcEncoder;
pub use isotope::{IsotopicSignature, NuclearSource};

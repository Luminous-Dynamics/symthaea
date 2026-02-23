//! Vocal tract controller — re-exported from `symthaea-vocal-tract` sub-crate.
//!
//! See `crates/symthaea-vocal-tract/src/controller.rs` for the canonical implementation.
//!
//! This module adds a `train_controller_on_phoneme_db()` helper that accepts a
//! `FormantDatabase` (main-crate type), resolving all phoneme targets into a slice
//! before delegating to the sub-crate's `VocalTractController::train_on_phoneme_targets()`.

pub use symthaea_vocal_tract::controller::{
    ProsodyCorrection, ProsodyHead, SpeakerProfile, VocalTractConfig, VocalTractController,
};

use super::formant_targets::FormantDatabase;
use symthaea_core::genesis::GenesisSeed;

/// Train the controller on all phonemes from a `FormantDatabase`.
///
/// Wrapper around [`VocalTractController::train_on_phoneme_targets`] that handles
/// `FormantDatabase` → `&[(&str, &FormantTarget)]` conversion. Returns the average
/// loss from the final epoch.
pub fn train_controller_on_phoneme_db(
    controller: &mut VocalTractController,
    genesis: &GenesisSeed,
    db: &FormantDatabase,
    epochs: usize,
) -> f32 {
    let phonemes = db.all_phonemes();
    let targets: Vec<(&str, &super::formant_targets::FormantTarget)> = phonemes
        .iter()
        .filter_map(|name| db.lookup(name).map(|t| (name.as_str(), t)))
        .collect();
    controller.train_on_phoneme_targets(genesis, &targets, epochs)
}

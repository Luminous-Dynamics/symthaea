// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vocal tract controller — re-exported from `symthaea-vocal-tract` sub-crate.
//!
//! See `crates/symthaea-vocal-tract/src/controller.rs` for the canonical implementation.
//!
//! This module adds a `train_controller_on_phoneme_db()` helper that accepts a
//! `FormantDatabase` (main-crate type), resolving all phoneme targets into a slice
//! before delegating to the sub-crate's `VocalTractController::train_on_phoneme_targets()`.

pub use symthaea_vocal_tract::controller::{
    ProsodyCorrection, ProsodyHead, SpeakerProfile, TrainingHyperparams, VocalTractConfig,
    VocalTractController,
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

/// Train the controller on common vowel transitions for smooth coarticulation.
///
/// Generates all pairwise transitions between the 6 cardinal vowels
/// (AH, IY, UW, AE, AA, EH) and trains via BPTT sequence training.
/// Call this AFTER `train_controller_on_phoneme_db` to add transition smoothness
/// on top of static phoneme accuracy.
pub fn train_controller_transitions(
    controller: &mut VocalTractController,
    genesis: &GenesisSeed,
    db: &FormantDatabase,
    epochs: usize,
) -> f32 {
    let cardinal_vowels = ["AH", "IY", "UW", "AE", "AA", "EH"];

    // Build all pairwise transition tuples
    let mut pairs = Vec::new();
    for &from in &cardinal_vowels {
        for &to in &cardinal_vowels {
            if from != to {
                if let (Some(from_t), Some(to_t)) = (db.lookup(from), db.lookup(to)) {
                    pairs.push((from, from_t, to, to_t));
                }
            }
        }
    }

    // Convert to the format train_on_transitions expects
    let pair_refs: Vec<(
        &str,
        &super::formant_targets::FormantTarget,
        &str,
        &super::formant_targets::FormantTarget,
    )> = pairs
        .iter()
        .map(|(f, ft, t, tt)| (*f, *ft, *t, *tt))
        .collect();

    controller.train_on_transitions(genesis, &pair_refs, epochs)
}

/// Refine the controller's output projection using least-squares optimization.
///
/// Call AFTER gradient-based training. Computes the analytical optimal output weights
/// that minimize squared formant error across all phonemes simultaneously. `blend`
/// controls interpolation: 0.0 = keep gradient weights, 1.0 = full LS replacement.
pub fn refine_controller_ls(
    controller: &mut VocalTractController,
    genesis: &GenesisSeed,
    db: &FormantDatabase,
    blend: f32,
) {
    let phonemes = db.all_phonemes();
    let targets: Vec<(&str, &super::formant_targets::FormantTarget)> = phonemes
        .iter()
        .filter_map(|name| db.lookup(name).map(|t| (name.as_str(), t)))
        .collect();
    controller.refine_output_projection_ls(genesis, &targets, blend);
}

/// Train the controller on consonant-vowel and vowel-consonant transitions.
///
/// Generates CV (consonant→vowel) and VC (vowel→consonant) transition pairs
/// from 6 consonants (P, T, K, S, M, N) and 3 cardinal vowels (AH, IY, UW),
/// producing 36 training pairs. Call this AFTER `train_controller_on_phoneme_db`
/// to improve coarticulation smoothness at syllable boundaries.
pub fn train_controller_cv_vc_transitions(
    controller: &mut VocalTractController,
    genesis: &GenesisSeed,
    db: &FormantDatabase,
    epochs: usize,
) -> f32 {
    let consonants = ["P", "T", "K", "S", "M", "N"];
    let vowels = ["AH", "IY", "UW"];

    // Build CV and VC transition pairs
    let mut pairs = Vec::new();
    for &c in &consonants {
        for &v in &vowels {
            // CV transition (consonant → vowel)
            if let (Some(ct), Some(vt)) = (db.lookup(c), db.lookup(v)) {
                pairs.push((c, ct, v, vt));
            }
            // VC transition (vowel → consonant)
            if let (Some(vt), Some(ct)) = (db.lookup(v), db.lookup(c)) {
                pairs.push((v, vt, c, ct));
            }
        }
    }

    let pair_refs: Vec<(
        &str,
        &super::formant_targets::FormantTarget,
        &str,
        &super::formant_targets::FormantTarget,
    )> = pairs
        .iter()
        .map(|(f, ft, t, tt)| (*f, *ft, *t, *tt))
        .collect();

    controller.train_on_transitions(genesis, &pair_refs, epochs)
}

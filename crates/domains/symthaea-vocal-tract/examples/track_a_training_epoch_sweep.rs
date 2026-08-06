// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Does more training budget help, or does it get worse? Follow-up to
//! `track_a_trained_smoke_test.rs`, whose 40-epoch run found final_loss
//! (0.1596) WORSE than first_epoch_loss (0.1470) on the full 39-phoneme
//! table -- an honest negative result
//! (`TRACK_A_TRAINED_SMOKE_TEST_RESULT_2026-07-30.md`), disclosed as
//! plausibly under-training but not confirmed.
//!
//! IMPORTANT mechanism this harness accounts for:
//! `train_on_phoneme_targets_configured`'s learning rate follows a cosine
//! schedule *within* one call, from `lr_peak` at epoch 0 down to `lr_min`
//! at the final epoch (see controller.rs: `progress = epoch / epochs`,
//! `cos_factor = 0.5*(1+cos(progress*PI))`). This means each call to
//! `train_on_phoneme_targets(..., N)` is a SELF-CONTAINED full anneal
//! scaled to N epochs -- calling it repeatedly in small chunks would
//! restart the anneal at high LR every time (a warm-restart pattern), NOT
//! continue a single long decay. So to test "does a bigger training
//! budget help", this harness trains a FRESH controller (same genesis,
//! same deterministic init) for each budget in ONE call each, not
//! chunked -- an apples-to-apples comparison of "one full anneal at
//! budget N" across several N values.
//!
//! ```bash
//! cargo run --release -p symthaea-vocal-tract --example track_a_training_epoch_sweep
//! ```

use symthaea_core::genesis::GenesisSeed;
use symthaea_vocal_tract::controller::{VocalTractConfig, VocalTractController};
use symthaea_vocal_tract::phonetics::{PhonemeClass, arpabet_articulation};
use symthaea_vocal_tract::types::{FormantTarget, SourceType};

/// Same 39-phoneme table as track_a_trained_smoke_test.rs, same derivation
/// method (from `arpabet_articulation`, not hand-picked).
const CANONICAL_PHONEMES: &[&str] = &[
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW", "P",
    "B", "T", "D", "K", "G", "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH", "M", "N", "NG", "L",
    "R", "W", "Y", "CH", "JH",
];

fn target_for_phoneme(symbol: &str) -> FormantTarget {
    let meta = arpabet_articulation(symbol);
    let duration_ms = match meta.class {
        PhonemeClass::Vowel => 150.0,
        PhonemeClass::Nasal => 100.0,
        PhonemeClass::Liquid | PhonemeClass::Glide => 90.0,
        PhonemeClass::Fricative => 120.0,
        PhonemeClass::Affricate => 100.0,
        PhonemeClass::Stop => 80.0,
        PhonemeClass::Silence => 0.0,
    };
    match meta.class {
        PhonemeClass::Vowel => FormantTarget::vowel(meta.f1, meta.f2, meta.f3, duration_ms),
        PhonemeClass::Stop => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Stop)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Stop)
            }
        }
        PhonemeClass::Fricative => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Fricative)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Fricative)
            }
        }
        PhonemeClass::Nasal => {
            FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Nasal)
        }
        PhonemeClass::Liquid | PhonemeClass::Glide => {
            FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Liquid)
        }
        PhonemeClass::Affricate => {
            if meta.voiced {
                FormantTarget::voiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Affricate)
            } else {
                FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                    .with_manner(SourceType::Affricate)
            }
        }
        PhonemeClass::Silence => {
            FormantTarget::unvoiced_consonant(meta.f1, meta.f2, meta.f3, duration_ms)
                .with_manner(SourceType::Silent)
        }
    }
}

/// Epoch budgets to sweep. 1 and 40 replicate the prior smoke test's two
/// data points (first_epoch_loss / final_loss); 10 and 100 probe whether
/// more budget ever turns the trend around.
///
/// Trimmed from an original 9-point sweep (up to 4000 epochs) after the
/// first 4 points (1/10/40/100 = 151 cumulative epoch-equivalents) took
/// ~30 real minutes on this host -- roughly 12s/epoch across the full
/// 39-phoneme table. At that rate 200/500/1000/2000/4000 would have cost
/// several more hours; not run, see
/// `TRACK_A_TRAINED_SMOKE_TEST_RESULT_2026-07-30.md`'s epoch-sweep
/// section for the real 4-point result and what it does/doesn't show.
const EPOCH_BUDGETS: &[usize] = &[1, 10, 40, 100];

fn main() {
    let genesis = GenesisSeed::from_phrase("track-a-trained-smoke::training");
    let config = VocalTractConfig::default();

    let phoneme_targets: Vec<(&str, FormantTarget)> = CANONICAL_PHONEMES
        .iter()
        .map(|&sym| (sym, target_for_phoneme(sym)))
        .collect();
    let target_refs: Vec<(&str, &FormantTarget)> =
        phoneme_targets.iter().map(|(name, t)| (*name, t)).collect();

    println!("=== symthaea-vocal-tract Track A epoch-budget sweep ===");
    println!(
        "Each row trains a FRESH controller (same genesis/init) for exactly that many epochs \
         in ONE call -- NOT chunked, since the LR schedule is a self-contained cosine anneal \
         per call (see module doc). {} phonemes, same table as track_a_trained_smoke_test.rs.\n",
        target_refs.len()
    );

    let mut prev_loss: Option<f32> = None;
    let mut min_loss = f32::INFINITY;
    let mut min_loss_epochs = 0usize;

    for &epochs in EPOCH_BUDGETS {
        let mut ctrl = VocalTractController::new(&genesis, &config);
        let loss = ctrl.train_on_phoneme_targets(&genesis, &target_refs, epochs);
        let trend = match prev_loss {
            Some(p) if loss < p => "IMPROVING",
            Some(p) if loss > p => "WORSENING",
            Some(_) => "FLAT",
            None => "(baseline)",
        };
        println!("epochs={epochs:5} -> loss={loss:.4}  [{trend}]");
        if loss < min_loss {
            min_loss = loss;
            min_loss_epochs = epochs;
        }
        prev_loss = Some(loss);
    }

    println!(
        "\nBest loss observed: {min_loss:.4} at epochs={min_loss_epochs}. \
         Monotonic trend across the full sweep: {}",
        if min_loss_epochs == *EPOCH_BUDGETS.first().unwrap() {
            "more training NEVER helped -- consistent with genuine interference/instability, \
             not just under-training"
        } else if min_loss_epochs == *EPOCH_BUDGETS.last().unwrap() {
            "loss kept improving through the largest budget tested -- consistent with \
             under-training, worth trying even more epochs"
        } else {
            "loss improved then got worse again -- an interior optimum, not simply \
             under-training nor simply unstable"
        }
    );
}

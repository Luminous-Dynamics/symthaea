// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Universal Temporal Grammar Demo
//!
//! ONE ARCHITECTURE - MANY DOMAINS
//!
//! Demonstrates the same CfC + Sparse HDC + Hierarchical Grammar
//! working across completely different signal types:
//!
//! - Human Speech (phonemes)
//! - Whale Calls (bioacoustics)
//! - Heart Rhythms (ECG)
//! - Brain Waves (EEG)
//! - Musical Notes

use symthaea_stt::temporal_grammar::{DomainConfig, TemporalEvent, TemporalGrammar};

fn separator(c: char, n: usize) {
    println!("{}", std::iter::repeat(c).take(n).collect::<String>());
}

fn header(title: &str) {
    println!();
    separator('=', 70);
    println!("  {}", title);
    separator('=', 70);
    println!();
}

fn subheader(title: &str) {
    println!();
    separator('-', 70);
    println!("  {}", title);
    separator('-', 70);
    println!();
}

fn main() {
    header("UNIVERSAL TEMPORAL GRAMMAR");
    println!("  One Architecture. Many Domains. Infinite Possibilities.");
    println!();
    println!("  Core Equation: H_t = Π(H_{{t-1}}) ⊗ (Φ_event ⊗ Φ_duration ⊗ Φ_intensity)");
    println!();
    println!("  Features:");
    println!("    - CfC neurons with adaptive time constants");
    println!("    - Sparse HDC (10% density) for 10× capacity");
    println!("    - Predictive feedback for biological plausibility");
    println!("    - Hierarchical event binding");

    // =========================================================================
    // DOMAIN 1: HUMAN SPEECH
    // =========================================================================
    header("DOMAIN 1: HUMAN SPEECH");

    let mut speech = TemporalGrammar::speech();
    let stats = speech.stats();
    println!("  Configuration:");
    println!("    Categories:  {} phonemes", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!(
        "    Hierarchy:   {} levels (phonemes → words)",
        stats.hierarchy_depth
    );

    subheader("Training: English CVC Words");

    // Train on common CVC patterns
    let words = [
        (
            "CAT",
            vec![("K", 0.08, 0.7), ("AE", 0.12, 0.8), ("T", 0.06, 0.6)],
        ),
        (
            "DOG",
            vec![("D", 0.06, 0.7), ("AO", 0.15, 0.75), ("G", 0.08, 0.65)],
        ),
        (
            "BIT",
            vec![("B", 0.05, 0.6), ("IH", 0.08, 0.7), ("T", 0.06, 0.6)],
        ),
        (
            "SIT",
            vec![("S", 0.10, 0.5), ("IH", 0.08, 0.7), ("T", 0.06, 0.6)],
        ),
        (
            "PEN",
            vec![("P", 0.07, 0.7), ("EH", 0.10, 0.75), ("N", 0.08, 0.5)],
        ),
    ];

    for (word, phonemes) in &words {
        let events: Vec<TemporalEvent> = phonemes
            .iter()
            .enumerate()
            .map(|(i, (ph, dur, int))| {
                let start = phonemes[..i].iter().map(|(_, d, _)| d).sum();
                TemporalEvent::new(ph, speech.category_id(ph).unwrap_or(0), start, *dur, *int)
            })
            .collect();

        for _ in 0..20 {
            speech.train_sequence(&events);
        }
        println!(
            "    Trained: {} /{}/",
            word,
            phonemes
                .iter()
                .map(|(p, _, _)| *p)
                .collect::<Vec<_>>()
                .join(" ")
        );
    }

    subheader("Discrimination Test: Trained vs Novel");

    // Test trained word
    let cat_events: Vec<TemporalEvent> = vec![
        TemporalEvent::new("K", speech.category_id("K").unwrap(), 0.0, 0.08, 0.7),
        TemporalEvent::new("AE", speech.category_id("AE").unwrap(), 0.08, 0.12, 0.8),
        TemporalEvent::new("T", speech.category_id("T").unwrap(), 0.20, 0.06, 0.6),
    ];

    // Test novel word (not trained)
    let map_events: Vec<TemporalEvent> = vec![
        TemporalEvent::new("M", speech.category_id("M").unwrap(), 0.0, 0.08, 0.5),
        TemporalEvent::new("AE", speech.category_id("AE").unwrap(), 0.08, 0.10, 0.75),
        TemporalEvent::new("P", speech.category_id("P").unwrap(), 0.18, 0.07, 0.6),
    ];

    let score_cat = speech.score_sequence(&cat_events);
    let score_map = speech.score_sequence(&map_events);

    println!("    CAT (trained):   {:.4}", score_cat);
    println!("    MAP (novel):     {:.4}", score_map);
    println!("    Discrimination:  {:+.4}", score_cat - score_map);

    // =========================================================================
    // DOMAIN 2: CETACEAN BIOACOUSTICS
    // =========================================================================
    header("DOMAIN 2: CETACEAN BIOACOUSTICS");

    let mut whale = TemporalGrammar::cetacean();
    let stats = whale.stats();
    println!("  Configuration:");
    println!("    Categories:  {} call types", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!(
        "    Hierarchy:   {} levels (calls → codas)",
        stats.hierarchy_depth
    );

    subheader("Training: Sperm Whale Coda Patterns");

    // Sperm whale coda: rhythmic click sequences
    let coda_3plus1: Vec<TemporalEvent> = vec![
        TemporalEvent::new("click", whale.category_id("click").unwrap(), 0.0, 0.02, 0.9),
        TemporalEvent::new(
            "click",
            whale.category_id("click").unwrap(),
            0.1,
            0.02,
            0.85,
        ),
        TemporalEvent::new("click", whale.category_id("click").unwrap(), 0.2, 0.02, 0.8),
        TemporalEvent::new(
            "silence",
            whale.category_id("silence").unwrap(),
            0.22,
            0.3,
            0.0,
        ),
        TemporalEvent::new(
            "click",
            whale.category_id("click").unwrap(),
            0.52,
            0.02,
            0.9,
        ),
    ];

    let coda_regular: Vec<TemporalEvent> = vec![
        TemporalEvent::new("click", whale.category_id("click").unwrap(), 0.0, 0.02, 0.8),
        TemporalEvent::new(
            "click",
            whale.category_id("click").unwrap(),
            0.15,
            0.02,
            0.8,
        ),
        TemporalEvent::new(
            "click",
            whale.category_id("click").unwrap(),
            0.30,
            0.02,
            0.8,
        ),
        TemporalEvent::new(
            "click",
            whale.category_id("click").unwrap(),
            0.45,
            0.02,
            0.8,
        ),
    ];

    for _ in 0..30 {
        whale.train_sequence(&coda_3plus1);
    }
    println!("    Trained: 3+1 Coda (click-click-click---click)");

    subheader("Species Discrimination: Sperm vs Dolphin");

    // Dolphin whistle pattern (very different)
    let dolphin_whistle: Vec<TemporalEvent> = vec![
        TemporalEvent::new(
            "whistle_up",
            whale.category_id("whistle_up").unwrap(),
            0.0,
            0.5,
            0.6,
        ),
        TemporalEvent::new(
            "whistle_down",
            whale.category_id("whistle_down").unwrap(),
            0.5,
            0.4,
            0.5,
        ),
        TemporalEvent::new(
            "whistle_flat",
            whale.category_id("whistle_flat").unwrap(),
            0.9,
            0.3,
            0.55,
        ),
    ];

    let score_sperm = whale.score_sequence(&coda_3plus1);
    let score_regular = whale.score_sequence(&coda_regular);
    let score_dolphin = whale.score_sequence(&dolphin_whistle);

    println!("    Sperm 3+1 (trained):  {:.4}", score_sperm);
    println!("    Sperm Regular:        {:.4}", score_regular);
    println!("    Dolphin Whistle:      {:.4}", score_dolphin);
    println!(
        "    Sperm vs Dolphin:     {:+.4}",
        score_sperm - score_dolphin
    );

    // =========================================================================
    // DOMAIN 3: CARDIAC (ECG)
    // =========================================================================
    header("DOMAIN 3: CARDIAC MONITORING (ECG)");

    let mut ecg = TemporalGrammar::ecg();
    let stats = ecg.stats();
    println!("  Configuration:");
    println!("    Categories:  {} wave types", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!(
        "    Prediction:  {} (for regular rhythms)",
        if stats.predictive_feedback {
            "ON"
        } else {
            "OFF"
        }
    );

    subheader("Training: Normal Sinus Rhythm");

    // Normal PQRST complex
    let normal_beat: Vec<TemporalEvent> = vec![
        TemporalEvent::new("P", ecg.category_id("P").unwrap(), 0.0, 0.08, 0.25),
        TemporalEvent::new("Q", ecg.category_id("Q").unwrap(), 0.12, 0.02, 0.15),
        TemporalEvent::new("R", ecg.category_id("R").unwrap(), 0.14, 0.04, 1.0), // Tallest
        TemporalEvent::new("S", ecg.category_id("S").unwrap(), 0.18, 0.03, 0.3),
        TemporalEvent::new("T", ecg.category_id("T").unwrap(), 0.28, 0.16, 0.4),
    ];

    for _ in 0..50 {
        ecg.train_sequence(&normal_beat);
    }
    println!("    Trained: Normal P-QRS-T complex (50 reps)");

    subheader("Arrhythmia Detection");

    // PVC - Premature Ventricular Contraction (abnormal)
    let pvc: Vec<TemporalEvent> = vec![
        TemporalEvent::new("PVC", ecg.category_id("PVC").unwrap(), 0.0, 0.16, 0.95),
        TemporalEvent::new(
            "baseline",
            ecg.category_id("baseline").unwrap(),
            0.16,
            0.4,
            0.05,
        ),
    ];

    // Atrial Fibrillation pattern (irregular)
    let afib: Vec<TemporalEvent> = vec![
        TemporalEvent::new("AF", ecg.category_id("AF").unwrap(), 0.0, 0.05, 0.3),
        TemporalEvent::new("R", ecg.category_id("R").unwrap(), 0.08, 0.04, 0.9),
        TemporalEvent::new("AF", ecg.category_id("AF").unwrap(), 0.15, 0.04, 0.25),
        TemporalEvent::new("R", ecg.category_id("R").unwrap(), 0.25, 0.04, 0.85),
    ];

    let score_normal = ecg.score_sequence(&normal_beat);
    let score_pvc = ecg.score_sequence(&pvc);
    let score_afib = ecg.score_sequence(&afib);

    println!("    Normal Sinus (trained): {:.4}", score_normal);
    println!("    PVC (abnormal):         {:.4}", score_pvc);
    println!("    AFib (irregular):       {:.4}", score_afib);
    println!();
    println!("    Normal vs PVC:   {:+.4}", score_normal - score_pvc);
    println!("    Normal vs AFib:  {:+.4}", score_normal - score_afib);

    if score_normal > score_pvc && score_normal > score_afib {
        println!();
        println!("    [PASS] Correctly identifies normal rhythm as most familiar");
    }

    // =========================================================================
    // DOMAIN 4: NEURAL (EEG)
    // =========================================================================
    header("DOMAIN 4: BRAIN MONITORING (EEG)");

    let mut eeg = TemporalGrammar::eeg();
    let stats = eeg.stats();
    println!("  Configuration:");
    println!("    Categories:  {} pattern types", stats.num_categories);
    println!("    Sparsity:    {:.0}%", stats.sparsity * 100.0);
    println!(
        "    Hierarchy:   {} levels (patterns → epochs → stages)",
        stats.hierarchy_depth
    );

    subheader("Training: Stage 2 Sleep Signature");

    // Stage 2 sleep: spindles and K-complexes
    let stage2_sleep: Vec<TemporalEvent> = vec![
        TemporalEvent::new("theta", eeg.category_id("theta").unwrap(), 0.0, 1.0, 0.4),
        TemporalEvent::new(
            "spindle",
            eeg.category_id("spindle").unwrap(),
            1.0,
            0.5,
            0.6,
        ),
        TemporalEvent::new("theta", eeg.category_id("theta").unwrap(), 1.5, 0.8, 0.35),
        TemporalEvent::new(
            "k_complex",
            eeg.category_id("k_complex").unwrap(),
            2.3,
            0.3,
            0.8,
        ),
        TemporalEvent::new("theta", eeg.category_id("theta").unwrap(), 2.6, 1.0, 0.4),
    ];

    for _ in 0..40 {
        eeg.train_sequence(&stage2_sleep);
    }
    println!("    Trained: Stage 2 (theta + spindles + K-complexes)");

    subheader("Sleep Stage Discrimination");

    // REM sleep: mixed frequency, low amplitude
    let rem_sleep: Vec<TemporalEvent> = vec![
        TemporalEvent::new("beta", eeg.category_id("beta").unwrap(), 0.0, 0.5, 0.3),
        TemporalEvent::new("theta", eeg.category_id("theta").unwrap(), 0.5, 0.7, 0.25),
        TemporalEvent::new("alpha", eeg.category_id("alpha").unwrap(), 1.2, 0.4, 0.2),
        TemporalEvent::new("beta", eeg.category_id("beta").unwrap(), 1.6, 0.6, 0.35),
    ];

    // Deep sleep (Stage 3): delta waves
    let deep_sleep: Vec<TemporalEvent> = vec![
        TemporalEvent::new("delta", eeg.category_id("delta").unwrap(), 0.0, 2.0, 0.9),
        TemporalEvent::new("delta", eeg.category_id("delta").unwrap(), 2.0, 1.8, 0.85),
        TemporalEvent::new("delta", eeg.category_id("delta").unwrap(), 3.8, 2.2, 0.9),
    ];

    let score_stage2 = eeg.score_sequence(&stage2_sleep);
    let score_rem = eeg.score_sequence(&rem_sleep);
    let score_deep = eeg.score_sequence(&deep_sleep);

    println!("    Stage 2 (trained):  {:.4}", score_stage2);
    println!("    REM Sleep:          {:.4}", score_rem);
    println!("    Deep Sleep (N3):    {:.4}", score_deep);
    println!();
    println!("    Stage2 vs REM:   {:+.4}", score_stage2 - score_rem);
    println!("    Stage2 vs Deep:  {:+.4}", score_stage2 - score_deep);

    // =========================================================================
    // SUMMARY
    // =========================================================================
    header("ARCHITECTURE SUMMARY");

    println!("  The Universal Temporal Grammar uses IDENTICAL code for:");
    println!();
    println!("    Domain         │ Events              │ Discrimination");
    println!("    ───────────────┼─────────────────────┼────────────────");
    println!(
        "    Speech         │ Phonemes            │ {:+.4}",
        score_cat - score_map
    );
    println!(
        "    Bioacoustics   │ Whale Calls         │ {:+.4}",
        score_sperm - score_dolphin
    );
    println!(
        "    Cardiology     │ ECG Waves           │ {:+.4}",
        score_normal - score_pvc
    );
    println!(
        "    Neurology      │ EEG Patterns        │ {:+.4}",
        score_stage2 - score_rem
    );
    println!();

    let avg_disc = ((score_cat - score_map).abs()
        + (score_sperm - score_dolphin).abs()
        + (score_normal - score_pvc).abs()
        + (score_stage2 - score_rem).abs())
        / 4.0;

    println!("    Average Discrimination: {:+.4}", avg_disc);
    println!();

    header("THE INSIGHT");

    println!("  All temporal signals share the same structure:");
    println!();
    println!("    EVENTS + TIMING + INTENSITY → SEQUENCES → GRAMMAR");
    println!();
    println!("  Whether it's:");
    println!("    - Phonemes in speech      (50-200ms)");
    println!("    - Clicks in whale songs   (10-50ms)");
    println!("    - QRS complexes in ECG    (60-100ms)");
    println!("    - Spindles in EEG         (500-1500ms)");
    println!();
    println!("  The same architecture learns them all:");
    println!();
    println!("    CfC (adaptive ears) + HDC (symbolic mind) + Hierarchy (grammar)");
    println!();

    separator('=', 70);
    println!("  UNIVERSAL TEMPORAL GRAMMAR DEMO COMPLETE");
    separator('=', 70);
}

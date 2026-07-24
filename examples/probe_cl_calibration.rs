// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CL/safety-tier calibration probe (post-scale-fix, 2026-07-18).
//!
//! Keystone Phase 5 restored genuine signal magnitudes and consciousness_level
//! now saturates ~0.95 in every regime — the Φ→MotorSafetyLevel tiers
//! (Green ≥ 0.6 / Yellow ≥ 0.3 / Orange ≥ 0.1) likely sit at constant Green,
//! making graduated motor safety a constant again (safety-relevant).
//!
//! This probe measures, per input regime: the CL distribution, safety-tier
//! occupancy/transitions, and the MCE bottleneck telemetry (which dimension
//! binds, softmin, weighted sum) — the data needed to recalibrate the input
//! normalizations (e.g. structural_boost = avg·0.5 pegs at 1.0 for post-fix
//! Φ ~8–50) rather than guessing constants.
//!
//! Run: cargo run --release --example probe_cl_calibration

use std::collections::BTreeMap;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const CYCLES_PER_REGIME: usize = 300;

fn tier(cl: f64) -> &'static str {
    if cl >= 0.6 {
        "Green"
    } else if cl >= 0.3 {
        "Yellow"
    } else if cl >= 0.1 {
        "Orange"
    } else {
        "Red"
    }
}

fn main() {
    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some("cl-calibration-2026-07-18".into());
    config.async_training = false;
    let mut svc = CognitiveLoopService::new(config).expect("service");

    let regimes: Vec<(&str, Vec<&str>)> = vec![
        (
            "repetitive",
            vec!["the system hums quietly in the background"],
        ),
        (
            "varied",
            vec![
                "The water cycle moves moisture from oceans to clouds to rain.",
                "Is it acceptable to lie to protect a friend from harm?",
                "Two plus two equals four, and four plus four equals eight.",
                "The market fell three percent on news of the supply shortage.",
                "A gentle rain began to fall as the travelers reached the shelter.",
                "What is the meaning of a life well lived?",
            ],
        ),
        (
            "alarming",
            vec![
                "URGENT: fire detected in the server room, evacuate immediately!",
                "Critical failure: coolant pressure dropping, meltdown risk rising!",
                "Intruder alert: perimeter breach at the north gate right now!",
            ],
        ),
    ];

    for (name, inputs) in &regimes {
        let mut cls: Vec<f64> = Vec::new();
        let mut tiers: BTreeMap<&str, usize> = BTreeMap::new();
        let mut transitions = 0usize;
        let mut prev_tier = "";
        let mut bottlenecks: BTreeMap<String, usize> = BTreeMap::new();
        let mut softmins: Vec<f64> = Vec::new();
        let mut weighted: Vec<f64> = Vec::new();

        for i in 0..CYCLES_PER_REGIME {
            let r = svc.cycle(inputs[i % inputs.len()]);
            let cl = r.metadata.consciousness.consciousness_level;
            if cl > 0.0 {
                cls.push(cl);
                let t = tier(cl);
                *tiers.entry(t).or_default() += 1;
                if !prev_tier.is_empty() && t != prev_tier {
                    transitions += 1;
                }
                prev_tier = t;
            }
            let (bn, sm, ws) = svc.mce_summary();
            if !bn.is_empty() {
                *bottlenecks.entry(bn).or_default() += 1;
            }
            softmins.push(sm);
            weighted.push(ws);
        }

        cls.sort_by(|a, b| a.total_cmp(b));
        let mean = cls.iter().sum::<f64>() / cls.len().max(1) as f64;
        let q = |p: f64| cls[((cls.len() - 1) as f64 * p) as usize];
        println!("regime {name}:");
        println!(
            "  CL n={} min={:.4} q25={:.4} med={:.4} q75={:.4} max={:.4} mean={:.4} range={:.4}",
            cls.len(),
            q(0.0),
            q(0.25),
            q(0.5),
            q(0.75),
            q(1.0),
            mean,
            q(1.0) - q(0.0)
        );
        println!("  tiers: {tiers:?} | transitions: {transitions}");
        println!(
            "  mce softmin mean={:.4} weighted mean={:.4} | bottlenecks: {bottlenecks:?}",
            softmins.iter().sum::<f64>() / softmins.len().max(1) as f64,
            weighted.iter().sum::<f64>() / weighted.len().max(1) as f64,
        );
        println!();
    }
}

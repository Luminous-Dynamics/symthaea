// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Step 3: action semantics and perception-to-action sensitivity in the main loop.
//!
//! # What the main loop treats as an action (established by code audit)
//!
//! The live, ungated action pipeline is `step_fep_active_inference`
//! (`helpers/cycle_extracted.rs:734`), called every cycle from
//! `cycle_phase_dynamics/mod.rs:2156`. It does a real
//! `perceive -> select_action -> act`.
//!
//! **Observation space — 4 scalars, all introspective:**
//! `prediction_error`, `coherence`, `prediction_confidence`, `effective_learning_rate`.
//!
//! **Action space — 4 discrete internal control adjustments:**
//! 0 boost learning rate · 1 reset sensory precision · 2 boost exploration ·
//! 3 tighten trust via precision.
//!
//! **No action touches the world, or even the representation.** Every one adjusts
//! the loop's own learning hyperparameters. This is self-regulation, not agency.
//! The consequence matters for the whole program: an agent whose actions cannot
//! alter what it observes next *from the world* cannot generate interventional
//! data, and so cannot learn distinctions that only interventions reveal.
//!
//! Perception therefore reaches action through a 4-scalar bottleneck of which
//! only 2 (`prediction_error`, `coherence`) carry any perceptual signal at all;
//! the other 2 are pure internal state.
//!
//! # Two levels, so a null is attributable
//!
//! - **L1 (actuator)** — sweep `(prediction_error, coherence)` directly into
//!   `step_fep_active_inference` on a fresh service per grid point. Does the
//!   selected action vary with its own inputs *at all*?
//! - **L2 (channel)** — feed genuinely different perceptual regimes through
//!   `cycle()` and record both the realized `prediction_error` and the selected
//!   action. Does perception move the observation enough to change the action?
//!
//! Running both makes a null attributable rather than merely observed: L1 null
//! means the actuator is inert; L1 live + L2 null means the channel is too narrow
//! or too saturated to drive it. That is the diagnostic discipline this repo
//! adopted after several "subsystem is inert" findings turned out to be
//! goal-miscalibration or saturation instead.
//!
//! Run: `cargo run --release --example action_semantics_probe`

use std::collections::BTreeMap;

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const ACTION_NAMES: [&str; 4] = [
    "0 boost-LR",
    "1 reset-precision",
    "2 boost-exploration",
    "3 tighten-trust",
];

/// Cycles per regime in L2.
const L2_CYCLES: usize = 120;

fn config() -> CognitiveLoopConfig {
    let mut c = CognitiveLoopConfig::with_cfc();
    c.enable_validation_overlay = false;
    c
}

fn service() -> CognitiveLoopService {
    CognitiveLoopService::new(config()).expect("service")
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// L1 — is the actuator sensitive to its own inputs?
// ═══════════════════════════════════════════════════════════════════════════

fn level1_actuator_sensitivity() -> bool {
    println!("\n═══ L1. ACTUATOR SENSITIVITY (direct input sweep) ═══");
    println!(
        "  Fresh service per grid point, so each row is 'action as a function of the\n  \
         FIRST observation' with no state carryover confound."
    );
    println!(
        "\n{:>6} {:>6}  {:<20} {:>34}",
        "PE", "coh", "action", "probabilities"
    );

    let pes = [0.0_f32, 0.2, 0.4, 0.6, 0.8, 1.0];
    let cohs = [0.0_f32, 0.5, 1.0];
    let mut seen: BTreeMap<usize, usize> = BTreeMap::new();
    let mut all_probs: Vec<Vec<f64>> = Vec::new();
    let mut selected_per_row: Vec<usize> = Vec::new();

    for &pe in pes.iter() {
        for &coh in cohs.iter() {
            let mut s = service();
            let (idx, probs) = s.probe_fep_action(pe, coh);
            *seen.entry(idx).or_insert(0) += 1;
            let shown: Vec<String> = probs.iter().map(|p| format!("{p:.4}")).collect();
            println!(
                "{pe:>6.2} {coh:>6.2}  {:<20} {:>34}",
                ACTION_NAMES.get(idx).copied().unwrap_or("?"),
                shown.join(" ")
            );
            selected_per_row.push(idx);
            all_probs.push(probs);
        }
    }

    let distinct_actions = seen.len();
    // Per-action probability spread across the grid: if the actuator is inert the
    // probabilities are identical everywhere, not merely the argmax.
    let n_actions = all_probs.first().map(|p| p.len()).unwrap_or(0);
    let mut max_spread = 0.0_f64;
    for a in 0..n_actions {
        let col: Vec<f64> = all_probs.iter().filter_map(|p| p.get(a).copied()).collect();
        let spread = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - col.iter().cloned().fold(f64::INFINITY, f64::min);
        max_spread = max_spread.max(spread);
    }

    // Does the SELECTED action match the argmax of the probability vector it
    // returns? A persistent mismatch means the reported probabilities do not
    // describe the choice being made, and one of the two is decorative.
    let mut argmax_mismatch = 0usize;
    for (row, probs) in all_probs.iter().enumerate() {
        if let Some((best, _)) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        {
            if Some(&best) != selected_per_row.get(row) {
                argmax_mismatch += 1;
            }
        }
    }
    // Degeneracy: are any two actions assigned bit-identical probability in every
    // row? That would mean the value computation cannot tell them apart at all.
    let mut degenerate_pairs = Vec::new();
    for a in 0..n_actions {
        for b in (a + 1)..n_actions {
            if all_probs.iter().all(|p| (p[a] - p[b]).abs() < f64::EPSILON) {
                degenerate_pairs.push((a, b));
            }
        }
    }

    println!(
        "\n  distinct actions selected across grid : {distinct_actions} of 4  \
         (NOT EVIDENCE -- fixed RNG seed + fresh service per point forces this)"
    );
    println!(
        "  rows where selection != prob argmax   : {argmax_mismatch} of {}  \
         (EXPECTED -- selection is stochastic, not argmax)",
        all_probs.len()
    );
    println!("  bit-identical action pairs            : {degenerate_pairs:?}");
    println!("  max probability spread across grid    : {max_spread:.6}");
    // Threshold TIGHTENED 2026-07-31 after the first run. The original
    // `max_spread > 1e-6` passed on any nonzero response at all, and duly
    // reported "responds" for a grid that selected ONE action at all 18 points
    // with a 2% probability movement. That is the same rule-design failure as
    // Step 1.5's half-vs-full recovery metric: a threshold set where anything
    // clears it measures nothing. "Responds" now requires either genuinely
    // different actions or a probability shift large enough to change a choice.
    // CORRECTED 2026-07-31: `distinct_actions` is NOT evidence about the actuator.
    // select_action samples stochastically from the softmax (agent.rs:316) using an
    // RNG whose state is a fixed constant at construction (agent.rs:132), and this
    // sweep uses a fresh service per point -- so every point draws the identical
    // first random number and therefore the identical action, by construction.
    // Only the PROBABILITY vector is deterministic given the inputs, so only it can
    // speak to sensitivity.
    let live = max_spread > 0.10;
    println!(
        "  L1 verdict                            : {}",
        if live {
            "ACTUATOR RESPONDS to its inputs"
        } else {
            "ACTUATOR INERT — identical action and probabilities everywhere"
        }
    );
    live
}

// ═══════════════════════════════════════════════════════════════════════════
// L2 — does perception move the observation enough to change the action?
// ═══════════════════════════════════════════════════════════════════════════

struct RegimeResult {
    name: &'static str,
    pe_mean: f64,
    pe_sd: f64,
    action_hist: BTreeMap<usize, usize>,
}

fn level2_channel(regimes: &[(&'static str, Vec<&'static str>)]) -> Vec<RegimeResult> {
    println!("\n═══ L2. PERCEPTION → ACTION CHANNEL ═══");
    println!(
        "  {L2_CYCLES} cycles per regime through the real cycle(), recording the realized\n  \
         prediction_error and the selected FEP action."
    );

    let mut out = Vec::new();
    for (name, inputs) in regimes {
        let mut s = service();
        let mut pes = Vec::new();
        let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
        for i in 0..L2_CYCLES {
            let input = inputs[i % inputs.len()];
            let r = s.cycle(input);
            pes.push(r.prediction_error as f64);
            *hist.entry(r.metadata.fep.fep_action).or_insert(0) += 1;
        }
        out.push(RegimeResult {
            name,
            pe_mean: mean(&pes),
            pe_sd: stddev(&pes),
            action_hist: hist,
        });
    }

    println!(
        "\n{:<14} {:>10} {:>10}   {}",
        "regime", "PE mean", "PE sd", "action histogram"
    );
    for r in &out {
        let hist: Vec<String> = (0..4)
            .map(|a| format!("{}:{}", a, r.action_hist.get(&a).copied().unwrap_or(0)))
            .collect();
        println!(
            "{:<14} {:>10.4} {:>10.4}   {}",
            r.name,
            r.pe_mean,
            r.pe_sd,
            hist.join("  ")
        );
    }
    out
}

fn main() {
    println!("Step 3 — action semantics & perception-to-action sensitivity");
    println!(
        "\nESTABLISHED BY AUDIT (see module docs):\n  \
         observation = [prediction_error, coherence, prediction_confidence, effective_lr]\n  \
         action      = 4 internal hyperparameter adjustments; NONE touches the world\n  \
         call site   = cycle_phase_dynamics/mod.rs:2156, ungated, every cycle"
    );

    let l1_live = level1_actuator_sensitivity();

    let regimes: Vec<(&'static str, Vec<&'static str>)> = vec![
        (
            "repetitive",
            vec!["the same sentence again", "the same sentence again"],
        ),
        (
            "varied",
            vec![
                "a heron lifts from the shallows at dawn",
                "quarterly revenue fell short of the forecast",
                "the compiler rejected a lifetime annotation",
                "salt changes how bread dough ferments",
            ],
        ),
        (
            "alarming",
            vec![
                "the reactor coolant pressure is falling fast",
                "someone is trapped under the collapsed beam",
                "the pathogen has jumped to a second host",
            ],
        ),
    ];
    let l2 = level2_channel(&regimes);

    println!("\n═══ VERDICT ═══");
    let pe_means: Vec<f64> = l2.iter().map(|r| r.pe_mean).collect();
    let pe_range = pe_means.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - pe_means.iter().cloned().fold(f64::INFINITY, f64::min);

    // Do the regimes differ in which actions they select at all?
    let mut action_sets: Vec<Vec<usize>> = Vec::new();
    for r in &l2 {
        let mut keys: Vec<usize> = r.action_hist.keys().copied().collect();
        keys.sort_unstable();
        action_sets.push(keys);
    }
    let identical_action_sets = action_sets.windows(2).all(|w| w[0] == w[1]);
    // Compare distributions, not just support: same actions in very different
    // proportions is still sensitivity.
    let dominant: Vec<usize> = l2
        .iter()
        .map(|r| {
            r.action_hist
                .iter()
                .max_by_key(|&(_, c)| *c)
                .map(|(a, _)| *a)
                .unwrap_or(usize::MAX)
        })
        .collect();
    let identical_dominant = dominant.windows(2).all(|w| w[0] == w[1]);

    println!("  L1 actuator responds to its inputs   : {l1_live}");
    println!("  PE range across regimes              : {pe_range:.4}");
    println!("  regimes share the same action support: {identical_action_sets}");
    println!("  regimes share the same dominant action: {identical_dominant}");

    let verdict = if !l1_live {
        "ACTUATOR INERT — action selection does not respond to its own inputs. \
         Perception-to-action sensitivity is impossible regardless of the channel."
    } else if pe_range < 0.01 {
        "CHANNEL FLAT — the actuator responds, but perception barely moves prediction \
         error across very different regimes, so it cannot drive action."
    } else if identical_dominant && identical_action_sets {
        "CHANNEL PRESENT BUT NOT DECISIVE — PE differs across regimes yet the action \
         distribution does not shift its dominant choice."
    } else {
        "SENSITIVE — different perceptual regimes produce different action behavior."
    };
    println!("\n  {verdict}");

    println!(
        "\n  SCOPE: even a SENSITIVE verdict does not establish agency. Every action in\n  \
         this space adjusts an internal learning hyperparameter; none changes what the\n  \
         loop will observe from the world next. The intervention-generation problem is\n  \
         structural here, not a tuning issue."
    );
    println!("\nDone.");
}

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MA-001R-delta multi-seed shuffled check — direct follow-up to §15's disclosed candidate
//! explanation (now that the RNG-bias hypothesis has been ruled out, see
//! `ma001r_delta_rng_bias_check.rs`): is Shuffled's residual movement beyond the
//! balanced-decorrelated control (0.1206) just a single-seed finite-sample artifact, or does it
//! persist across many independent seeds?
//!
//! Runs the shuffled-context control (raw-observation mechanism) across 10 independent RNG seeds
//! and reports the mean and spread of post-training `delta_predicted`, compared against the
//! single-seed values already on record: Equal-outcome 0.0714, balanced-decorrelated 0.1206,
//! shuffled (seed 1) 0.1821.
//!
//! Run: `cargo run -p symthaea-alife --example ma001r_delta_shuffled_multiseed --release`

use symthaea_alife::ma001l::{DeltaRuleConfig, DeltaRuleLearner};
use symthaea_alife::ma001r::{Ma001rConfig, Ma001rProbe};
use symthaea_alife::organism::OrganismConfig;

const ORGANISM_SEED: u64 = 1;
const RNG_SEEDS: [u64; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

fn social_cfg() -> OrganismConfig {
    OrganismConfig {
        social_enabled: true,
        ..OrganismConfig::default()
    }
}

fn main() {
    let cfg = Ma001rConfig::default();
    println!(
        "MA-001R-delta multi-seed shuffled check -- {} independent RNG seeds\n",
        RNG_SEEDS.len()
    );
    println!("Reference values already on record (single default RNG seed = 1):");
    println!("  Equal-outcome control:      0.0714");
    println!("  Balanced-decorrelated ctrl: 0.1206");
    println!("  Shuffled (this seed):       0.1821\n");

    let mut results = Vec::with_capacity(RNG_SEEDS.len());
    for &rng_seed in &RNG_SEEDS {
        let mut probe = Ma001rProbe::new(social_cfg(), ORGANISM_SEED, cfg);
        probe.set_learning_pathway(false, false);
        let delta_rule =
            DeltaRuleLearner::new(DeltaRuleConfig::default(), &probe.organism.agent.model);
        let mut rng_state = 0x9E3779B97F4A7C15u64 ^ rng_seed;
        probe.run_shuffled_with_delta_rule_from_raw_observation(
            &delta_rule,
            0,
            cfg.training_ticks,
            &mut rng_state,
        );
        let post = probe.counterfactual_reading();
        println!(
            "  rng_seed={rng_seed:2}: post-training delta_predicted = {:.4}",
            post.delta_predicted
        );
        results.push(post.delta_predicted);
    }

    let mean: f64 = results.iter().sum::<f64>() / results.len() as f64;
    let min = results.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = results.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let variance: f64 =
        results.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / results.len() as f64;
    let std_dev = variance.sqrt();

    println!("\n=== Summary across {} seeds ===", RNG_SEEDS.len());
    println!("  mean   = {mean:.4}");
    println!("  std_dev = {std_dev:.4}");
    println!("  min    = {min:.4}");
    println!("  max    = {max:.4}");
    println!(
        "\nFor comparison: Equal-outcome=0.0714, Balanced-decorrelated=0.1206, single-seed Shuffled=0.1821"
    );

    let mean_close_to_balanced = (mean - 0.1206).abs() <= 0.1206 * 0.20;
    let mean_close_to_single_seed = (mean - 0.1821).abs() <= 0.1821 * 0.20;
    let high_variance = std_dev > mean * 0.30;

    println!(
        "\nVERDICT: {}",
        if mean_close_to_balanced && !mean_close_to_single_seed {
            "The multi-seed MEAN converges toward the balanced-decorrelated control's value, not \
            the single default seed's elevated 0.1821 -- CONFIRMS the residual was a single-seed \
            finite-sample artifact, not a systematic property of per-tick randomized assignment. \
            The default seed (1) happened to land on the high end of the distribution."
        } else if mean_close_to_single_seed && !high_variance {
            "The multi-seed mean stays close to the single-seed value with LOW variance across \
            seeds -- this REFUTES the finite-sample-artifact hypothesis. Shuffled's residual \
            movement beyond the balanced-decorrelated control appears to be a systematic property \
            of per-tick independent randomization itself (as opposed to a fixed periodic \
            schedule), not chance. This would need a mechanistic explanation, not yet identified."
        } else if high_variance {
            "High variance across seeds (std_dev is a large fraction of the mean) -- the specific \
            post-training delta_predicted is itself noisy/seed-sensitive, and averaging is doing \
            real work here. Report the full mean+spread rather than trusting any single seed's \
            result as representative -- this itself is a methodological finding about how many \
            seeds this kind of measurement needs."
        } else {
            "Mixed result -- report exact numbers, mean, and spread rather than forcing a verdict."
        }
    );
}

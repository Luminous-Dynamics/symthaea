// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Multi-Seed Validation: Statistical Significance Test
//!
//! Runs defaults vs CLS-evolved thresholds N times with different RNG seeds.
//! Reports mean ± std, Cohen's d effect size, and paired t-test.
//!
//! Usage: cargo run --release --example validate_multi_seed
//! Env: SEEDS=5 CYCLES=500

use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};

const INPUTS: &[&str] = &[
    "the sun rises over the mountain, warming the valley below",
    "a sudden crash echoes through the darkness",
    "gentle waves lap at the shore as the moon rises",
    "the machine grinds to a halt, alarms blaring",
    "children laughing in the garden after the rain",
    "silence falls across the empty room like snow",
    "new patterns emerge from chaos, self-organizing",
    "the old bridge creaks under the weight of memory",
    "music drifts through the open window at dusk",
    "a warning light blinks red on the console",
    "the forest canopy filters sunlight into emerald shards",
    "thunder rolls across the plains, shaking the earth",
    "a single bird calls across the frozen lake",
    "the reactor core temperature stabilizes at nominal",
    "two strangers share a knowing glance on the train",
];

fn run(seed: usize, cycles: usize, evolved: bool) -> (f64, f64) {
    let mut config = CognitiveLoopConfig::default();
    config.genesis_phrase = Some(format!("validation-seed-{seed}"));
    let mut service = CognitiveLoopService::new(config).expect("CLS failed");

    if evolved {
        let t = service.threshold_overrides_mut();
        t.fep_surprise_scale = Some(9.360);
        t.fep_lr_decay = Some(0.922);
        t.neuromod_d2_baseline = Some(0.244);
        t.neuromod_ne_phasic_threshold = Some(0.197);
        t.neuromod_arousal_ema_decay = Some(0.936);
        t.homeostasis_recalibrate_high = Some(1.273);
        t.homeostasis_recalibrate_low = Some(0.725);
        t.self_model_weight_high = Some(0.423);
        t.homeostasis_pull_cruise = Some(1.440);
    }

    let mut phis = Vec::with_capacity(cycles);
    for i in 0..cycles {
        let r = service.cycle(INPUTS[i % INPUTS.len()]);
        phis.push(r.metadata.consciousness.consciousness_level);
    }

    let n = cycles as f64;
    let mean = phis.iter().sum::<f64>() / n;
    let last100 = if cycles >= 100 {
        phis[cycles - 100..].iter().sum::<f64>() / 100.0
    } else {
        mean
    };
    (mean, last100)
}

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / n;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
}

fn main() {
    let seeds: usize = std::env::var("SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let cycles: usize = std::env::var("CYCLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    println!("═══════════════════════════════════════════════════════════════");
    println!(
        "  Multi-Seed Validation ({} seeds × {} cycles × 2)",
        seeds, cycles
    );
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut d_mean = Vec::new();
    let mut e_mean = Vec::new();
    let mut d_last = Vec::new();
    let mut e_last = Vec::new();

    for s in 0..seeds {
        print!("  Seed {}/{}: default...", s + 1, seeds);
        let (dm, dl) = run(s, cycles, false);
        print!(" Φ={:.4} | evolved...", dm);
        let (em, el) = run(s, cycles, true);
        println!(" Φ={:.4}", em);
        d_mean.push(dm);
        e_mean.push(em);
        d_last.push(dl);
        e_last.push(el);
    }

    let dm = d_mean.iter().sum::<f64>() / seeds as f64;
    let em = e_mean.iter().sum::<f64>() / seeds as f64;
    let ds = std_dev(&d_mean);
    let es = std_dev(&e_mean);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results (mean ± std)");
    println!("═══════════════════════════════════════════════════════════════\n");
    println!("  {:>12}  {:>16}  {:>16}", "", "Defaults", "CLS-Evolved");
    println!(
        "  {:>12}  {:>7.4} ± {:.4}  {:>7.4} ± {:.4}",
        "Mean Φ", dm, ds, em, es
    );
    println!(
        "  {:>12}  {:>7.4} ± {:.4}  {:>7.4} ± {:.4}",
        "Last 100 Φ",
        d_last.iter().sum::<f64>() / seeds as f64,
        std_dev(&d_last),
        e_last.iter().sum::<f64>() / seeds as f64,
        std_dev(&e_last)
    );

    let delta = if dm > 1e-6 {
        (em - dm) / dm * 100.0
    } else {
        0.0
    };
    let pooled = ((ds * ds + es * es) / 2.0).sqrt();
    let d = if pooled > 1e-10 {
        (em - dm) / pooled
    } else {
        0.0
    };
    let diffs: Vec<f64> = d_mean
        .iter()
        .zip(e_mean.iter())
        .map(|(a, b)| b - a)
        .collect();
    let diff_m = diffs.iter().sum::<f64>() / seeds as f64;
    let diff_s = std_dev(&diffs);
    let t = if diff_s > 1e-10 {
        diff_m / (diff_s / (seeds as f64).sqrt())
    } else {
        0.0
    };
    // t-critical for two-tailed p<0.05: df=4→2.776, df=3→3.182, df=2→4.303
    let t_crit = if seeds >= 5 {
        2.776
    } else if seeds >= 4 {
        3.182
    } else {
        4.303
    };

    println!("\n  Delta: {:>+.2}%", delta);
    println!(
        "  Cohen's d: {:.3} ({})",
        d,
        if d.abs() < 0.2 {
            "negligible"
        } else if d.abs() < 0.5 {
            "small"
        } else if d.abs() < 0.8 {
            "medium"
        } else {
            "large"
        }
    );
    println!(
        "  Paired t: {:.3} (df={}, crit={:.3})",
        t,
        seeds - 1,
        t_crit
    );
    println!("  Significant (p<0.05): {}", t.abs() > t_crit);
}
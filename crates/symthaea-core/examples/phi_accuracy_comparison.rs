// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi Approximation Accuracy Comparison
//!
//! Compares every available Φ approximation method against the TruePhiCalculator
//! (Shannon-entropy MIP) across controlled network topologies.
//!
//! ## Methods under test
//!
//! | Method                  | Type         | Complexity |
//! |-------------------------|--------------|------------|
//! | `ExhaustivePartition`   | Binary HV    | O(2^n)     |
//! | `SampledPartition`      | Binary HV    | O(n)       |
//! | `SpectralConnectivity`  | Binary HV    | O(n²)      |
//! | `TruePhiCalculator`     | Continuous HV| O(2^n)     |
//! | `compute_phi_fast` (EI) | Continuous HV| O(n²)      |
//!
//! ## Topologies
//!
//! * **Independent** — no shared structure; Φ should → 0
//! * **Correlated** — all derived from one base vector; Φ should be high
//! * **Ring** — each node shares half its basis with neighbours
//! * **Star** — one hub vector bound to each spoke; strong hub, weak spokes
//! * **Fully-connected** — all pairs share equal coupling; medium Φ
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example phi_accuracy_comparison
//! # With extended sizes (slower, more accurate):
//! cargo run --example phi_accuracy_comparison -- --extended
//! ```

use symthaea_core::consciousness_metrics::TruePhiCalculator;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::tiered_phi::{ApproximationTier, TieredPhi};
use symthaea_core::hdc::unified_hv::ContinuousHV;

const HDC_DIM: usize = 1024; // Reduced from 16384 for faster benchmarks; set to 16384 for production

// ─── Topology Builders ───────────────────────────────────────────────────────

fn independent_continuous(n: usize, seed_offset: u64) -> Vec<ContinuousHV> {
    (0..n)
        .map(|i| ContinuousHV::random(HDC_DIM, seed_offset + i as u64 * 1_000_000))
        .collect()
}

fn correlated_continuous(n: usize, strength: f32, seed_offset: u64) -> Vec<ContinuousHV> {
    let base = ContinuousHV::random(HDC_DIM, seed_offset);
    (0..n)
        .map(|i| {
            let noise = ContinuousHV::random(HDC_DIM, seed_offset + i as u64 + 1);
            ContinuousHV::weighted_bundle(&[&base, &noise], &[strength, 1.0 - strength])
        })
        .collect()
}

fn ring_continuous(n: usize, coupling: f32, seed_offset: u64) -> Vec<ContinuousHV> {
    let bases: Vec<ContinuousHV> = (0..n)
        .map(|i| ContinuousHV::random(HDC_DIM, seed_offset + i as u64))
        .collect();
    (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            ContinuousHV::weighted_bundle(&[&bases[i], &bases[next]], &[1.0 - coupling, coupling])
        })
        .collect()
}

fn star_continuous(n: usize, hub_strength: f32, seed_offset: u64) -> Vec<ContinuousHV> {
    let hub = ContinuousHV::random(HDC_DIM, seed_offset);
    let mut nodes = vec![hub.clone()]; // hub is node 0
    for i in 1..n {
        let spoke_base = ContinuousHV::random(HDC_DIM, seed_offset + i as u64);
        let spoke = ContinuousHV::weighted_bundle(
            &[&hub, &spoke_base],
            &[hub_strength, 1.0 - hub_strength],
        );
        nodes.push(spoke);
    }
    nodes
}

fn fully_connected_continuous(n: usize, coupling: f32, seed_offset: u64) -> Vec<ContinuousHV> {
    // All pairs share `coupling` of the same global basis
    let global = ContinuousHV::random(HDC_DIM, seed_offset);
    (0..n)
        .map(|i| {
            let local = ContinuousHV::random(HDC_DIM, seed_offset + i as u64 + 1);
            ContinuousHV::weighted_bundle(&[&global, &local], &[coupling, 1.0 - coupling])
        })
        .collect()
}

// Binary HV versions (for TieredPhi)
fn independent_binary(n: usize, seed_offset: u64) -> Vec<BinaryHV> {
    (0..n)
        .map(|i| BinaryHV::random(seed_offset + i as u64 * 1_000_000))
        .collect()
}

fn correlated_binary(n: usize, strength_bits: usize, seed_offset: u64) -> Vec<BinaryHV> {
    let base = BinaryHV::random(seed_offset);
    (0..n)
        .map(|i| {
            let mut v = BinaryHV::random(seed_offset + i as u64 + 1);
            // Copy `strength_bits` fraction of base bits into v
            let bits_to_copy = (BinaryHV::dimension() * strength_bits) / 100;
            for b in 0..bits_to_copy {
                let bit = base.get_bit(b);
                v.set_bit(b, bit);
            }
            v
        })
        .collect()
}

fn ring_binary(n: usize, coupling_bits: usize, seed_offset: u64) -> Vec<BinaryHV> {
    let bases: Vec<BinaryHV> = (0..n)
        .map(|i| BinaryHV::random(seed_offset + i as u64))
        .collect();
    (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            let mut v = bases[i].clone();
            let bits_to_copy = (BinaryHV::dimension() * coupling_bits) / 100;
            for b in 0..bits_to_copy {
                v.set_bit(b, bases[next].get_bit(b));
            }
            v
        })
        .collect()
}

fn star_binary(n: usize, hub_bits: usize, seed_offset: u64) -> Vec<BinaryHV> {
    let hub = BinaryHV::random(seed_offset);
    let mut nodes = vec![hub.clone()];
    for i in 1..n {
        let mut spoke = BinaryHV::random(seed_offset + i as u64);
        let bits_to_copy = (BinaryHV::dimension() * hub_bits) / 100;
        for b in 0..bits_to_copy {
            spoke.set_bit(b, hub.get_bit(b));
        }
        nodes.push(spoke);
    }
    nodes
}

fn fully_connected_binary(n: usize, coupling_bits: usize, seed_offset: u64) -> Vec<BinaryHV> {
    let global = BinaryHV::random(seed_offset);
    (0..n)
        .map(|i| {
            let mut local = BinaryHV::random(seed_offset + i as u64 + 1);
            let bits_to_copy = (BinaryHV::dimension() * coupling_bits) / 100;
            for b in 0..bits_to_copy {
                local.set_bit(b, global.get_bit(b));
            }
            local
        })
        .collect()
}

// ─── Result Types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TopologyResult {
    name: &'static str,
    n: usize,
    // Continuous HV methods
    true_phi: f64, // TruePhiCalculator exhaustive MIP
    phi_ei: f64,   // compute_effective_information_normalized (fast, no MIP)
    // Binary HV methods
    exhaustive: f64, // ExhaustivePartition
    sampled: f64,    // SampledPartition
    spectral: f64,   // SpectralConnectivity (deprecated for Φ, kept for calibration)
    // Timing (microseconds)
    time_true_us: u128,
    time_ei_us: u128,
    time_exhaustive_us: u128,
    time_sampled_us: u128,
    time_spectral_us: u128,
}

impl TopologyResult {
    /// Absolute error of each approx vs true_phi (continuous MIP = ground truth)
    fn error_exhaustive(&self) -> f64 {
        (self.exhaustive - self.true_phi).abs()
    }
    fn error_sampled(&self) -> f64 {
        (self.sampled - self.true_phi).abs()
    }
    fn error_spectral(&self) -> f64 {
        (self.spectral - self.true_phi).abs()
    }
    fn error_ei(&self) -> f64 {
        (self.phi_ei - self.true_phi).abs()
    }
}

// ─── Benchmark Runner ────────────────────────────────────────────────────────

fn run_topology(
    name: &'static str,
    n: usize,
    continuous: Vec<ContinuousHV>,
    binary: Vec<BinaryHV>,
) -> TopologyResult {
    let calc = TruePhiCalculator::new();
    let mut tiered_exact = TieredPhi::new(ApproximationTier::ExhaustivePartition);
    let mut tiered_sampled = TieredPhi::new(ApproximationTier::SampledPartition);
    #[allow(deprecated)]
    let mut tiered_spectral = TieredPhi::new(ApproximationTier::SpectralConnectivity);

    // True Phi (continuous, exhaustive MIP)
    let t0 = std::time::Instant::now();
    let true_result = calc.compute_true_phi(&continuous);
    let time_true_us = t0.elapsed().as_micros();

    // Fast EI (continuous, no MIP - raw mutual information)
    let t0 = std::time::Instant::now();
    let phi_ei = calc.compute_effective_information_normalized(&continuous);
    let time_ei_us = t0.elapsed().as_micros();

    // Exhaustive Partition (binary)
    let t0 = std::time::Instant::now();
    let exhaustive = tiered_exact.compute(&binary);
    let time_exhaustive_us = t0.elapsed().as_micros();

    // Sampled Partition (binary)
    let t0 = std::time::Instant::now();
    let sampled = tiered_sampled.compute(&binary);
    let time_sampled_us = t0.elapsed().as_micros();

    // Spectral Connectivity (binary, deprecated for Phi)
    let t0 = std::time::Instant::now();
    #[allow(deprecated)]
    let spectral = tiered_spectral.compute(&binary);
    let time_spectral_us = t0.elapsed().as_micros();

    TopologyResult {
        name,
        n,
        true_phi: true_result.phi,
        phi_ei,
        exhaustive,
        sampled,
        spectral,
        time_true_us,
        time_ei_us,
        time_exhaustive_us,
        time_sampled_us,
        time_spectral_us,
    }
}

// ─── Pearson Correlation ─────────────────────────────────────────────────────

fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let num: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - mx) * (y - my))
        .sum();
    let dx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = ys.iter().map(|y| (y - my).powi(2)).sum::<f64>().sqrt();
    if dx * dy < 1e-12 {
        0.0
    } else {
        num / (dx * dy)
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

fn main() {
    let extended = std::env::args().any(|a| a == "--extended");

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      Phi Approximation Accuracy Comparison — Mycelix        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("HDC Dimension: {}", HDC_DIM);
    println!(
        "Mode: {}",
        if extended {
            "extended (slow)"
        } else {
            "standard"
        }
    );
    println!();

    // Test cases: (name, n_components)
    let sizes: &[usize] = if extended {
        &[3, 4, 5, 6, 7, 8]
    } else {
        &[3, 4, 5, 6]
    };

    let mut all_results: Vec<TopologyResult> = Vec::new();

    for &n in sizes {
        println!(
            "── n = {} components ─────────────────────────────────────────",
            n
        );

        // Independent
        let cont = independent_continuous(n, 0);
        let bin = independent_binary(n, 0);
        all_results.push(run_topology("independent", n, cont, bin));

        // Correlated (strong: 80% shared basis)
        let cont = correlated_continuous(n, 0.80, 100);
        let bin = correlated_binary(n, 80, 100);
        all_results.push(run_topology("correlated-80%", n, cont, bin));

        // Ring (50% coupling to neighbor)
        let cont = ring_continuous(n, 0.50, 200);
        let bin = ring_binary(n, 50, 200);
        all_results.push(run_topology("ring-50%", n, cont, bin));

        // Star (60% hub influence on spokes)
        let cont = star_continuous(n, 0.60, 300);
        let bin = star_binary(n, 60, 300);
        all_results.push(run_topology("star-60%", n, cont, bin));

        // Fully-connected (40% global coupling)
        let cont = fully_connected_continuous(n, 0.40, 400);
        let bin = fully_connected_binary(n, 40, 400);
        all_results.push(run_topology("fully-connected-40%", n, cont, bin));

        // Print per-n table
        println!();
        println!(
            "  {:<22} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "Topology", "TruePhi", "Exhaust", "Sampled", "Spectral", "FastEI"
        );
        println!("  {}", "─".repeat(70));
        for r in all_results.iter().filter(|r| r.n == n) {
            println!(
                "  {:<22} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
                r.name, r.true_phi, r.exhaustive, r.sampled, r.spectral, r.phi_ei
            );
        }
        println!();
    }

    // ─── Summary Statistics ───────────────────────────────────────────────

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                  Accuracy Summary (vs TruePhi)              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let true_vals: Vec<f64> = all_results.iter().map(|r| r.true_phi).collect();
    let exhaust_vals: Vec<f64> = all_results.iter().map(|r| r.exhaustive).collect();
    let sampled_vals: Vec<f64> = all_results.iter().map(|r| r.sampled).collect();
    let spectral_vals: Vec<f64> = all_results.iter().map(|r| r.spectral).collect();
    let ei_vals: Vec<f64> = all_results.iter().map(|r| r.phi_ei).collect();

    let r_exhaust = pearson(&true_vals, &exhaust_vals);
    let r_sampled = pearson(&true_vals, &sampled_vals);
    let r_spectral = pearson(&true_vals, &spectral_vals);
    let r_ei = pearson(&true_vals, &ei_vals);

    let mean_err_exhaust: f64 = all_results
        .iter()
        .map(|r| r.error_exhaustive())
        .sum::<f64>()
        / all_results.len() as f64;
    let mean_err_sampled: f64 =
        all_results.iter().map(|r| r.error_sampled()).sum::<f64>() / all_results.len() as f64;
    let mean_err_spectral: f64 =
        all_results.iter().map(|r| r.error_spectral()).sum::<f64>() / all_results.len() as f64;
    let mean_err_ei: f64 =
        all_results.iter().map(|r| r.error_ei()).sum::<f64>() / all_results.len() as f64;

    let max_err_exhaust: f64 = all_results
        .iter()
        .map(|r| r.error_exhaustive())
        .fold(0.0_f64, f64::max);
    let max_err_sampled: f64 = all_results
        .iter()
        .map(|r| r.error_sampled())
        .fold(0.0_f64, f64::max);
    let max_err_spectral: f64 = all_results
        .iter()
        .map(|r| r.error_spectral())
        .fold(0.0_f64, f64::max);
    let max_err_ei: f64 = all_results
        .iter()
        .map(|r| r.error_ei())
        .fold(0.0_f64, f64::max);

    let avg_time_exhaust: f64 = all_results
        .iter()
        .map(|r| r.time_exhaustive_us as f64)
        .sum::<f64>()
        / all_results.len() as f64;
    let avg_time_sampled: f64 = all_results
        .iter()
        .map(|r| r.time_sampled_us as f64)
        .sum::<f64>()
        / all_results.len() as f64;
    let avg_time_spectral: f64 = all_results
        .iter()
        .map(|r| r.time_spectral_us as f64)
        .sum::<f64>()
        / all_results.len() as f64;
    let avg_time_ei: f64 =
        all_results.iter().map(|r| r.time_ei_us as f64).sum::<f64>() / all_results.len() as f64;

    println!(
        "  {:<22}  {:>8}  {:>9}  {:>9}  {:>11}",
        "Method", "Pearson r", "Mean |err|", "Max |err|", "Avg time µs"
    );
    println!("  {}", "─".repeat(72));

    let mut rows = vec![
        (
            "Exhaustive (binary)",
            r_exhaust,
            mean_err_exhaust,
            max_err_exhaust,
            avg_time_exhaust,
        ),
        (
            "Sampled   (binary)",
            r_sampled,
            mean_err_sampled,
            max_err_sampled,
            avg_time_sampled,
        ),
        (
            "Spectral  (binary)",
            r_spectral,
            mean_err_spectral,
            max_err_spectral,
            avg_time_spectral,
        ),
        (
            "FastEI   (cont.)",
            r_ei,
            mean_err_ei,
            max_err_ei,
            avg_time_ei,
        ),
    ];
    // Sort by mean error ascending (best first)
    rows.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (i, (name, r, me, mx, t)) in rows.iter().enumerate() {
        let medal = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };
        println!(
            "  {} {:<20}  {:>8.4}  {:>9.4}  {:>9.4}  {:>11.1}",
            medal, name, r, me, mx, t
        );
    }

    println!();

    // ─── Verdict ─────────────────────────────────────────────────────────

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         Verdict                             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let best = rows.first().unwrap();
    println!(
        "  Best approximation vs TruePhi:  {} (mean |err| = {:.4})",
        best.0, best.2
    );
    println!();
    println!("  SpectralConnectivity Pearson r = {:.4}", r_spectral);
    if r_spectral < 0.0 {
        println!("  ⚠ CONFIRMED: Spectral connectivity is ANTI-correlated with true Φ.");
        println!("    Do NOT use for consciousness gating — use Sampled or Exhaustive.");
    } else if r_spectral < 0.3 {
        println!("  ⚠ Spectral connectivity has weak positive correlation with true Φ.");
        println!("    Still not reliable for consciousness gating.");
    } else {
        println!("  Spectral connectivity shows useful correlation with true Φ on this test set.");
    }
    println!();

    // ─── Governance Gate Calibration Guidance ────────────────────────────

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            Governance Gate Calibration Guidance             ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Current gates (from CONSCIOUSNESS_METRICS.md):");
    println!("    basic         >= 0.2");
    println!("    proposal      >= 0.3");
    println!("    voting        >= 0.4");
    println!("    constitutional>= 0.6");
    println!();

    let independent_phi: f64 = all_results
        .iter()
        .filter(|r| r.name == "independent")
        .map(|r| r.true_phi)
        .sum::<f64>()
        / all_results
            .iter()
            .filter(|r| r.name == "independent")
            .count()
            .max(1) as f64;
    let correlated_phi: f64 = all_results
        .iter()
        .filter(|r| r.name == "correlated-80%")
        .map(|r| r.true_phi)
        .sum::<f64>()
        / all_results
            .iter()
            .filter(|r| r.name == "correlated-80%")
            .count()
            .max(1) as f64;

    println!("  Observed true-Φ range on test topologies:");
    println!("    Independent systems: mean Φ ≈ {:.4}", independent_phi);
    println!("    Correlated  systems: mean Φ ≈ {:.4}", correlated_phi);
    println!();
    println!("  These empirical ranges should inform gate calibration.");
    println!("  Run `cargo run --example phi_accuracy_comparison -- --extended`");
    println!("  for larger systems and more reliable statistics.");
    println!();
    println!("  Done.");
}

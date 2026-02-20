//! Byzantine Tolerance Phase Diagram Benchmark
//!
//! Sweeps Byzantine fraction from 0% to 40% across three aggregation strategies:
//! - Equal reputation via pipeline (all nodes rep=0.8, multi-signal detection on)
//! - Low-rep Byzantine via pipeline (honest=0.9, byz=0.3, detection off,
//!   relies on reputation-weighted hybrid BFT to down-weight adversaries)
//! - Krum (direct `krum()` call, immune to detection gating)
//!
//! Produces an ASCII table and phase diagram showing convergence boundaries.

use mycelix_fl_core::consciousness_plugin::ConsciousnessAwareByzantinePlugin;
use mycelix_fl_core::plugins::{ByzantinePlugin, PipelinePlugins};
use mycelix_fl_core::{krum, GradientUpdate, PipelineConfig, UnifiedPipeline};
use std::collections::HashMap;

/// Total number of participants in each trial.
const TOTAL_NODES: usize = 20;

/// Gradient dimension for synthetic updates.
const GRADIENT_DIM: usize = 20;

/// Value used by honest nodes for each gradient element.
const HONEST_VALUE: f32 = 0.5;

/// Value used by Byzantine nodes for each gradient element.
const BYZANTINE_VALUE: f32 = 100.0;

/// MSE convergence threshold: aggregated result must be within this of the honest mean.
const CONVERGENCE_THRESHOLD: f64 = 1.0;

/// Byzantine fractions to sweep (as percentages, converted to node counts).
const BYZANTINE_PERCENTAGES: &[usize] = &[0, 5, 10, 15, 20, 25, 30, 34, 40];

/// Result of a single trial.
#[derive(Debug, Clone)]
struct TrialResult {
    _byz_pct: usize,
    mse: f64,
    converged: bool,
    _method: &'static str,
    error_msg: Option<String>,
}

/// Build gradient updates for the given number of Byzantine nodes.
fn build_updates(num_byzantine: usize) -> Vec<GradientUpdate> {
    let num_honest = TOTAL_NODES - num_byzantine;
    let mut updates = Vec::with_capacity(TOTAL_NODES);

    for i in 0..num_honest {
        updates.push(GradientUpdate::new(
            format!("honest_{}", i),
            1,
            vec![HONEST_VALUE; GRADIENT_DIM],
            100,
            0.1,
        ));
    }

    for i in 0..num_byzantine {
        updates.push(GradientUpdate::new(
            format!("byz_{}", i),
            1,
            vec![BYZANTINE_VALUE; GRADIENT_DIM],
            100,
            0.1,
        ));
    }

    updates
}

/// Compute MSE of aggregated gradients vs the honest mean (HONEST_VALUE per dim).
fn compute_mse(gradients: &[f32]) -> f64 {
    let honest_mean = HONEST_VALUE as f64;
    let sum_sq: f64 = gradients
        .iter()
        .map(|g| {
            let diff = *g as f64 - honest_mean;
            diff * diff
        })
        .sum();
    sum_sq / gradients.len() as f64
}

/// Run the equal-reputation pipeline (multi-signal detection ON).
fn run_equal_rep_trial(updates: &[GradientUpdate], byz_pct: usize) -> TrialResult {
    let config = PipelineConfig::default();
    let mut pipeline = UnifiedPipeline::new(config);

    let reputations: HashMap<String, f32> = updates
        .iter()
        .map(|u| (u.participant_id.clone(), 0.8))
        .collect();

    match pipeline.aggregate(updates, &reputations) {
        Ok(result) => {
            let mse = compute_mse(&result.aggregated.gradients);
            TrialResult {
                _byz_pct: byz_pct,
                mse,
                converged: mse < CONVERGENCE_THRESHOLD,
                _method: "EqualRep",
                error_msg: None,
            }
        }
        Err(e) => TrialResult {
            _byz_pct: byz_pct,
            mse: f64::INFINITY,
            converged: false,
            _method: "EqualRep",
            error_msg: Some(format!("{}", e)),
        },
    }
}

/// Run the low-rep-Byzantine pipeline (multi-signal detection OFF, relies on
/// hybrid BFT reputation weighting to suppress low-reputation adversaries).
fn run_low_rep_trial(updates: &[GradientUpdate], byz_pct: usize) -> TrialResult {
    let mut config = PipelineConfig::default();
    // Disable multi-signal detection so the pipeline doesn't early-terminate
    // when the detected Byzantine fraction exceeds max_byzantine_tolerance.
    // Instead, hybrid BFT rep^2 weighting + trimming handles adversaries.
    config.multi_signal_detection = false;
    // Increase trim fraction to aggressively clip outlier dimensions.
    config.trim_fraction = 0.2;

    let mut pipeline = UnifiedPipeline::new(config);

    let reputations: HashMap<String, f32> = updates
        .iter()
        .map(|u| {
            let rep = if u.participant_id.starts_with("honest") {
                0.9
            } else {
                0.3
            };
            (u.participant_id.clone(), rep)
        })
        .collect();

    match pipeline.aggregate(updates, &reputations) {
        Ok(result) => {
            let mse = compute_mse(&result.aggregated.gradients);
            TrialResult {
                _byz_pct: byz_pct,
                mse,
                converged: mse < CONVERGENCE_THRESHOLD,
                _method: "LowRepByz",
                error_msg: None,
            }
        }
        Err(e) => TrialResult {
            _byz_pct: byz_pct,
            mse: f64::INFINITY,
            converged: false,
            _method: "LowRepByz",
            error_msg: Some(format!("{}", e)),
        },
    }
}

/// Run krum directly.
fn run_krum_trial(updates: &[GradientUpdate], byz_pct: usize) -> TrialResult {
    match krum(updates, 1) {
        Ok(gradients) => {
            let mse = compute_mse(&gradients);
            TrialResult {
                _byz_pct: byz_pct,
                mse,
                converged: mse < CONVERGENCE_THRESHOLD,
                _method: "Krum",
                error_msg: None,
            }
        }
        Err(e) => TrialResult {
            _byz_pct: byz_pct,
            mse: f64::INFINITY,
            converged: false,
            _method: "Krum",
            error_msg: Some(format!("{}", e)),
        },
    }
}

/// Run consciousness-aware pipeline (honest=high phi, byz=low phi, multi-signal OFF).
fn run_consciousness_trial(
    updates: &[GradientUpdate],
    byz_pct: usize,
    num_byzantine: usize,
) -> TrialResult {
    let mut config = PipelineConfig::default();
    config.multi_signal_detection = false;
    config.trim_fraction = 0.2;

    let mut pipeline = UnifiedPipeline::new(config);

    let reputations: HashMap<String, f32> = updates
        .iter()
        .map(|u| (u.participant_id.clone(), 0.8))
        .collect();

    let mut consciousness_plugin = ConsciousnessAwareByzantinePlugin::new();
    let mut consciousness_scores = HashMap::new();
    for u in updates {
        let score = if u.participant_id.starts_with("honest") {
            0.7
        } else {
            0.08 // Below veto threshold (0.1)
        };
        consciousness_scores.insert(u.participant_id.clone(), score);
    }
    consciousness_plugin.set_consciousness_scores(consciousness_scores);

    let mut plugins = PipelinePlugins {
        compression: None,
        byzantine: vec![&mut consciousness_plugin],
        verification: None,
    };

    match pipeline.aggregate_with_plugins(updates, &reputations, &mut plugins) {
        Ok(result) => {
            let mse = compute_mse(&result.result.aggregated.gradients);
            TrialResult {
                _byz_pct: byz_pct,
                mse,
                converged: mse < CONVERGENCE_THRESHOLD,
                _method: "Conscious",
                error_msg: None,
            }
        }
        Err(e) => TrialResult {
            _byz_pct: byz_pct,
            mse: f64::INFINITY,
            converged: false,
            _method: "Conscious",
            error_msg: Some(format!("{}", e)),
        },
    }
}

fn main() {
    println!("=============================================================");
    println!("  Byzantine Tolerance Phase Diagram Benchmark");
    println!(
        "  {} nodes, {} gradient dims, honest={}, byz={}",
        TOTAL_NODES, GRADIENT_DIM, HONEST_VALUE, BYZANTINE_VALUE
    );
    println!("  Convergence threshold: MSE < {}", CONVERGENCE_THRESHOLD);
    println!("=============================================================\n");

    let mut equal_rep_results: Vec<TrialResult> = Vec::new();
    let mut low_rep_results: Vec<TrialResult> = Vec::new();
    let mut krum_results: Vec<TrialResult> = Vec::new();
    let mut consciousness_results: Vec<TrialResult> = Vec::new();

    for &byz_pct in BYZANTINE_PERCENTAGES {
        let num_byzantine = (TOTAL_NODES * byz_pct + 50) / 100; // round to nearest
        let updates = build_updates(num_byzantine);

        equal_rep_results.push(run_equal_rep_trial(&updates, byz_pct));
        low_rep_results.push(run_low_rep_trial(&updates, byz_pct));
        krum_results.push(run_krum_trial(&updates, byz_pct));
        consciousness_results.push(run_consciousness_trial(&updates, byz_pct, num_byzantine));
    }

    // ── ASCII Table ──────────────────────────────────────────────────────
    println!("{:-<109}", "");
    println!(
        "| {:>5} | {:>6} | {:>12} {:>6} | {:>12} {:>6} | {:>12} {:>6} | {:>12} {:>6} |",
        "Byz%",
        "Nodes",
        "EqualRep",
        "Conv?",
        "LowRepByz",
        "Conv?",
        "Krum",
        "Conv?",
        "Conscious",
        "Conv?"
    );
    println!("{:-<109}", "");

    for i in 0..BYZANTINE_PERCENTAGES.len() {
        let byz_pct = BYZANTINE_PERCENTAGES[i];
        let num_byz = (TOTAL_NODES * byz_pct + 50) / 100;

        let eq = &equal_rep_results[i];
        let lr = &low_rep_results[i];
        let kr = &krum_results[i];
        let cs = &consciousness_results[i];

        let fmt_mse = |r: &TrialResult| -> String {
            if let Some(ref e) = r.error_msg {
                let short: String = e.chars().take(12).collect();
                format!("{:>12}", short)
            } else if r.mse > 9999.0 {
                format!("{:>12.1}", r.mse)
            } else {
                format!("{:>12.6}", r.mse)
            }
        };
        let fmt_conv = |r: &TrialResult| -> &str {
            if r.error_msg.is_some() {
                "ERR"
            } else if r.converged {
                "PASS"
            } else {
                "FAIL"
            }
        };

        println!(
            "| {:>4}% | {:>2}/{:>2} | {} {:>6} | {} {:>6} | {} {:>6} | {} {:>6} |",
            byz_pct,
            num_byz,
            TOTAL_NODES,
            fmt_mse(eq),
            fmt_conv(eq),
            fmt_mse(lr),
            fmt_conv(lr),
            fmt_mse(kr),
            fmt_conv(kr),
            fmt_mse(cs),
            fmt_conv(cs),
        );
    }
    println!("{:-<109}", "");

    // ── ASCII Phase Diagram ──────────────────────────────────────────────
    println!("\n  Phase Diagram (PASS = converged, FAIL = diverged, ERR = pipeline error)");
    println!("  -----------------------------------------------------------------------");
    println!(
        "  {:>10} | {}",
        "Method",
        BYZANTINE_PERCENTAGES
            .iter()
            .map(|p| format!("{:>5}%", p))
            .collect::<Vec<_>>()
            .join(" ")
    );
    println!(
        "  {:-<10}-+-{:-<width$}",
        "",
        "",
        width = BYZANTINE_PERCENTAGES.len() * 6
    );

    let methods_display: &[(&str, &[TrialResult])] = &[
        ("EqualRep", &equal_rep_results),
        ("LowRepByz", &low_rep_results),
        ("Krum", &krum_results),
        ("Conscious", &consciousness_results),
    ];

    for (name, results) in methods_display {
        let cells: Vec<String> = results
            .iter()
            .map(|r| {
                if r.error_msg.is_some() {
                    format!("{:>6}", "[ERR]")
                } else if r.converged {
                    format!("{:>6}", "[OK]")
                } else {
                    format!("{:>6}", "[XX]")
                }
            })
            .collect();
        println!("  {:>10} | {}", name, cells.join(""));
    }
    println!(
        "  {:-<10}-+-{:-<width$}",
        "",
        "",
        width = BYZANTINE_PERCENTAGES.len() * 6
    );
    println!(
        "  Legend: [OK] = converged (MSE<{:.1}), [XX] = diverged, [ERR] = pipeline error\n",
        CONVERGENCE_THRESHOLD
    );

    // ── Interpretation ───────────────────────────────────────────────────
    // Find safety boundaries per method
    let find_boundary = |results: &[TrialResult]| -> usize {
        let mut last_converged = 0;
        for (i, r) in results.iter().enumerate() {
            if r.converged {
                last_converged = BYZANTINE_PERCENTAGES[i];
            }
        }
        last_converged
    };

    let eq_boundary = find_boundary(&equal_rep_results);
    let lr_boundary = find_boundary(&low_rep_results);
    let kr_boundary = find_boundary(&krum_results);
    let cs_boundary = find_boundary(&consciousness_results);

    println!("  Safety Boundaries (last converging %):");
    println!("    EqualRep:   {}%", eq_boundary);
    println!("    LowRepByz:  {}%", lr_boundary);
    println!("    Krum:        {}%", kr_boundary);
    println!("    Conscious:   {}%", cs_boundary);
    println!();

    // ── Verification Tests ───────────────────────────────────────────────
    println!("=============================================================");
    println!("  Verification Tests");
    println!("=============================================================\n");

    let mut pass_count = 0u32;
    let mut fail_count = 0u32;

    macro_rules! verify {
        ($name:expr, $cond:expr, $msg:expr) => {
            if $cond {
                println!("  [PASS] {}", $name);
                pass_count += 1;
            } else {
                println!("  [FAIL] {} -- {}", $name, $msg);
                fail_count += 1;
            }
        };
    }

    // Test 1: At 0% Byzantine, all methods converge (MSE < 0.01)
    {
        let eq_0 = &equal_rep_results[0];
        let lr_0 = &low_rep_results[0];
        let kr_0 = &krum_results[0];
        verify!(
            "0% Byz: EqualRep converges (MSE < 0.01)",
            eq_0.converged && eq_0.mse < 0.01,
            format!("MSE = {:.6}", eq_0.mse)
        );
        verify!(
            "0% Byz: LowRepByz converges (MSE < 0.01)",
            lr_0.converged && lr_0.mse < 0.01,
            format!("MSE = {:.6}", lr_0.mse)
        );
        verify!(
            "0% Byz: Krum converges (MSE < 0.01)",
            kr_0.converged && kr_0.mse < 0.01,
            format!("MSE = {:.6}", kr_0.mse)
        );
    }

    // Test 2: LowRepByz extends safety boundary beyond EqualRep
    //   Reputation disparity (honest=0.9, byz=0.3) with rep^2 weighting
    //   means byz nodes contribute weight 0.09 vs honest 0.81.
    //   This must extend convergence range vs equal-reputation.
    {
        verify!(
            "LowRepByz boundary > EqualRep boundary (reputation helps)",
            lr_boundary > eq_boundary,
            format!("LowRepByz={}%, EqualRep={}%", lr_boundary, eq_boundary)
        );
    }

    // Test 3: At 40% EqualRep, pipeline fails to converge
    {
        let idx_40 = BYZANTINE_PERCENTAGES.iter().position(|&p| p == 40).unwrap();
        let eq_40 = &equal_rep_results[idx_40];
        verify!(
            "40% Byz: EqualRep fails to converge",
            !eq_40.converged,
            format!("MSE = {:.6} (expected divergence)", eq_40.mse)
        );
    }

    // Test 4: Krum survives up to 30% Byzantine
    //   Krum(num_select=1) picks the gradient with the smallest sum of
    //   distances to its (n-2) nearest neighbors. With 6/20 Byzantine,
    //   14 honest nodes cluster tightly, so Krum always picks one.
    {
        let idx_30 = BYZANTINE_PERCENTAGES.iter().position(|&p| p == 30).unwrap();
        let kr_30 = &krum_results[idx_30];
        verify!(
            "30% Byz: Krum converges",
            kr_30.converged,
            format!(
                "MSE = {:.6}{}",
                kr_30.mse,
                kr_30
                    .error_msg
                    .as_deref()
                    .map(|e| format!(", err: {}", e))
                    .unwrap_or_default()
            )
        );
    }

    // Test 5: Krum is the most resilient method in this benchmark
    //   Since Krum selects the single closest-to-neighbors gradient, it
    //   tolerates up to (n-2)/2 Byzantine nodes = 9/20 = 45%.
    {
        verify!(
            "Krum boundary >= 34% (most resilient)",
            kr_boundary >= 34,
            format!("Krum boundary = {}%", kr_boundary)
        );
    }

    // Test 6: EqualRep fails very early (pipeline detection catches Byzantines
    //   but the equal weighting means even a few adversaries poison the mean)
    {
        verify!(
            "EqualRep boundary <= 5% (no reputation protection)",
            eq_boundary <= 5,
            format!("EqualRep boundary = {}%", eq_boundary)
        );
    }

    // Test 7: LowRepByz converges at 20% Byzantine
    //   With 4/20 Byzantine (rep=0.3), rep^2 voting power = 4*0.09 = 0.36
    //   vs honest 16*0.81 = 12.96. Byzantine effective fraction = 2.7%.
    {
        let idx_20 = BYZANTINE_PERCENTAGES.iter().position(|&p| p == 20).unwrap();
        let lr_20 = &low_rep_results[idx_20];
        verify!(
            "20% Byz: LowRepByz converges",
            lr_20.converged,
            format!(
                "MSE = {:.6}{}",
                lr_20.mse,
                lr_20
                    .error_msg
                    .as_deref()
                    .map(|e| format!(", err: {}", e))
                    .unwrap_or_default()
            )
        );
    }

    println!("\n  Results: {} passed, {} failed", pass_count, fail_count);
    println!("=============================================================\n");

    if fail_count > 0 {
        std::process::exit(1);
    }
}

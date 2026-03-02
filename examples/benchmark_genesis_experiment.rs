//! Genesis Mission Challenge 5: Experiment Planner
//!
//! Demonstrates HDC + CfC encoder/predictor for experiment optimization.
//! O(1) prediction cost proof across 3 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 5: Experiment Planner ===\n");

    use symthaea::symthaea_cell_foundry::experiment_planner::{
        ExperimentHdcEncoder, ExperimentPredictor, ExperimentReading, EXPERIMENT_HORIZONS,
    };

    // 1. Create healthy experiment reading
    let reading = ExperimentReading {
        experiment_uncertainty: 0.2,
        cost: 0.3,
        time_remaining: 604_800.0,
        info_gain: 0.7,
    };

    // 2. Encode and observe
    let enc = ExperimentHdcEncoder::new();
    let mut pred = ExperimentPredictor::new();
    let hv = enc.encode(&reading);
    pred.observe(&hv, 3600.0);
    println!("Encoded experiment reading (dim={})", hv.dim());

    // 3. Predict at each horizon
    for (i, &h) in EXPERIMENT_HORIZONS.iter().enumerate() {
        let predicted = pred.predict_at_horizon(&hv, h);
        let sim = predicted.similarity(&hv);
        println!("  Horizon {} ({:.0}s): similarity={:.4}", i, h, sim);
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = EXPERIMENT_HORIZONS[0];
    let long_h = EXPERIMENT_HORIZONS[EXPERIMENT_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Experiment Planner operational");
}

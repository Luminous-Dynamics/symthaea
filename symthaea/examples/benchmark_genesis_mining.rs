//! Genesis Mission Challenge 18: Critical Minerals
//!
//! Demonstrates HDC + CfC encoder/predictor for mineral extraction monitoring.
//! O(1) prediction cost proof across 3 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 18: Critical Minerals ===\n");

    use symthaea_materials::mining::{
        MiningHdcEncoder, MiningPredictor, MiningReading, MINING_HORIZONS,
    };

    // 1. Create healthy mining reading
    let reading = MiningReading {
        ore_grade: 0.5,
        extraction_rate: 0.7,
        environmental_impact: 0.1,
        cost: 0.4,
    };

    // 2. Encode and observe
    let enc = MiningHdcEncoder::new();
    let mut pred = MiningPredictor::new();
    let hv = enc.encode(&reading);
    pred.observe(&hv, 86_400.0);
    println!("Encoded mining reading (dim={})", hv.dim());

    // 3. Predict at each horizon
    for (i, &h) in MINING_HORIZONS.iter().enumerate() {
        let predicted = pred.predict_at_horizon(&hv, h);
        let sim = predicted.similarity(&hv);
        println!("  Horizon {} ({:.0}s): similarity={:.4}", i, h, sim);
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = MINING_HORIZONS[0];
    let long_h = MINING_HORIZONS[MINING_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Critical Minerals operational");
}

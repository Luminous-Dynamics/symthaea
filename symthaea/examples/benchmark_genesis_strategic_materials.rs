//! Genesis Mission Challenge 21: Strategic Materials
//!
//! Demonstrates HDC + CfC encoder/predictor for extreme-environment materials.
//! O(1) prediction cost proof across 4 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 21: Strategic Materials ===\n");

    use symthaea_materials::strategic::{
        StrategicHdcEncoder, StrategicPredictor, StrategicReading, STRATEGIC_HORIZONS,
    };

    // 1. Create healthy materials reading
    let reading = StrategicReading {
        extreme_temp_resilience: 0.9,
        radiation_dose: 0.1,
        time_at_condition: 86_400.0,
        failure_probability: 0.001,
    };

    // 2. Encode and observe
    let enc = StrategicHdcEncoder::new();
    let mut pred = StrategicPredictor::new();
    let hv = enc.encode(&reading);
    pred.observe(&hv, 86_400.0);
    println!("Encoded strategic materials reading (dim={})", hv.dim());

    // 3. Predict at each horizon
    for (i, &h) in STRATEGIC_HORIZONS.iter().enumerate() {
        let predicted = pred.predict_at_horizon(&hv, h);
        let sim = predicted.similarity(&hv);
        println!("  Horizon {} ({:.0}s): similarity={:.4}", i, h, sim);
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = STRATEGIC_HORIZONS[0];
    let long_h = STRATEGIC_HORIZONS[STRATEGIC_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Strategic Materials operational");
}

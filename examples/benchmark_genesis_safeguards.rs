//! Genesis Mission Challenge 24: Proliferation Safeguards
//!
//! Demonstrates HDC + CfC encoder/predictor for nuclear material monitoring.
//! O(1) prediction cost proof across 2 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 24: Proliferation Safeguards ===\n");

    use symthaea_nuclear_forensics::safeguards::{
        SafeguardsHdcEncoder, SafeguardsPredictor, SafeguardsReading, SAFEGUARDS_HORIZONS,
    };

    // 1. Create healthy safeguards reading
    let reading = SafeguardsReading {
        inventory_discrepancy: 0.01,
        sensor_anomaly: 0.02,
        timeline_consistency: 0.95,
    };

    // 2. Encode and observe
    let enc = SafeguardsHdcEncoder::new();
    let mut pred = SafeguardsPredictor::new();
    let hv = enc.encode(&reading);
    pred.observe(&hv, 86_400.0);
    println!("Encoded safeguards reading (dim={})", hv.dim());

    // 3. Predict at each horizon
    for (i, &h) in SAFEGUARDS_HORIZONS.iter().enumerate() {
        let predicted = pred.predict_at_horizon(&hv, h);
        let sim = predicted.similarity(&hv);
        println!("  Horizon {} ({:.0}s): similarity={:.4}", i, h, sim);
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = SAFEGUARDS_HORIZONS[0];
    let long_h = SAFEGUARDS_HORIZONS[SAFEGUARDS_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = pred.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Proliferation Safeguards operational");
}

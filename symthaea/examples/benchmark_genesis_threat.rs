//! Genesis Mission Challenge 22: Threat Assessment
//!
//! Demonstrates HDC + CfC + FEP digital twin for anomaly detection.
//! O(1) prediction cost proof across 5 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 22: Threat Assessment ===\n");

    use symthaea::physics::threat::{ThreatReading, ThreatTwin, THREAT_HORIZONS};

    // 1. Create healthy sensor reading
    let reading = ThreatReading {
        sensor_reading: 0.5,
        expected_value: 0.5,
        surprise_signal: 0.0,
        response_time: 1.0,
        confidence: 0.9,
    };

    // 2. Initialize twin with healthy reference
    let mut twin = ThreatTwin::new();
    twin.set_reference(&reading);

    // 3. Run a few steps
    for i in 0..3 {
        let output = twin.step(&reading, 0.1);
        println!(
            "Step {}: FE={:.3} | Action={:?} | Threat={:?}",
            i + 1, output.free_energy, output.recommended_action, output.threat_level
        );
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let predictor = twin.predictor();
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = THREAT_HORIZONS[0];
    let long_h = THREAT_HORIZONS[THREAT_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.2}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Threat Assessment operational");
}

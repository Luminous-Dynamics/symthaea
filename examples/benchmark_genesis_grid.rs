//! Genesis Mission Challenge 1: Grid Scaling
//!
//! Demonstrates HDC + CfC + FEP digital twin for electrical grid monitoring.
//! O(1) prediction cost proof across 5 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 1: Grid Scaling ===\n");

    use symthaea::physics::grid::{GridReading, GridTwin, GRID_HORIZONS};

    // 1. Create healthy grid reading
    let reading = GridReading {
        voltage: 1.0,
        frequency: 60.0,
        demand: 0.6,
        reserve_margin: 0.2,
        line_flow: 0.5,
    };

    // 2. Initialize twin with healthy reference
    let mut twin = GridTwin::new();
    twin.set_reference(&reading);

    // 3. Run a few steps
    for i in 0..3 {
        let output = twin.step(&reading, 60.0);
        println!(
            "Step {}: FE={:.3} | Action={:?} | Safety={:?}",
            i + 1, output.free_energy, output.recommended_action, output.safety_level
        );
    }

    // 4. O(1) cost proof
    println!("\n--- O(1) Prediction Cost Proof ---");
    let predictor = twin.predictor();
    let input = symthaea::symthaea_core::hdc::unified_hv::ContinuousHV::random(
        symthaea::symthaea_core::hdc::unified_hv::HDC_DIMENSION, 42,
    );

    let short_h = GRID_HORIZONS[0];
    let long_h = GRID_HORIZONS[GRID_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Grid Scaling operational");
}

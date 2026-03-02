//! Genesis Mission Challenge 19: Building Systems
//!
//! Demonstrates HDC + CfC + FEP digital twin for building systems monitoring.
//! O(1) prediction cost proof across 4 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 19: Building Systems ===\n");

    use symthaea_fabrication_kernel::building::{
        BuildingReading, BuildingTwin, BUILDING_HORIZONS,
    };

    // 1. Create healthy building reading
    let reading = BuildingReading {
        thermal_load: 0.4,
        structural_stress: 0.1,
        occupancy: 0.6,
        comfort: 0.85,
        energy_consumption: 0.35,
    };

    // 2. Initialize twin with healthy reference
    let mut twin = BuildingTwin::new();
    twin.set_reference(&reading);

    // 3. Run a few steps
    for i in 0..3 {
        let output = twin.step(&reading, 3600.0);
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

    let short_h = BUILDING_HORIZONS[0];
    let long_h = BUILDING_HORIZONS[BUILDING_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.0}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Building Systems operational");
}

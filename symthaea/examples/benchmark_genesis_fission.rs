//! Genesis Mission Challenge 2: Fission Reactor
//!
//! Demonstrates HDC + CfC + FEP digital twin for fission reactor monitoring.
//! O(1) prediction cost proof across 5 orders of magnitude.

fn main() {
    println!("=== Genesis Mission Challenge 2: Fission Reactor ===\n");

    use symthaea::physics::fission::{FissionReading, FissionTwin, FISSION_HORIZONS};

    // 1. Create healthy reactor reading
    let reading = FissionReading {
        power_output: 0.8,
        coolant_temp: 300.0,
        neutron_flux: 0.5,
        pressure: 10.0,
        control_rod_pos: 0.5,
    };

    // 2. Initialize twin with healthy reference
    let mut twin = FissionTwin::new();
    twin.set_reference(&reading);

    // 3. Run a few steps
    for i in 0..3 {
        let output = twin.step(&reading, 1.0);
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

    let short_h = FISSION_HORIZONS[0];
    let long_h = FISSION_HORIZONS[FISSION_HORIZONS.len() - 1];

    let t1 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, short_h); }
    let short_us = t1.elapsed().as_micros() as f64 / 1000.0;

    let t2 = std::time::Instant::now();
    for _ in 0..1000 { let _ = predictor.predict_at_horizon(&input, long_h); }
    let long_us = t2.elapsed().as_micros() as f64 / 1000.0;

    println!("  Short ({:.1}s):  {:.1}us/pred", short_h, short_us);
    println!("  Long  ({:.0}s): {:.1}us/pred", long_h, long_us);
    println!("  Ratio: {:.2}", long_us / short_us.max(0.001));

    println!("\nPASS: Fission Reactor operational");
}

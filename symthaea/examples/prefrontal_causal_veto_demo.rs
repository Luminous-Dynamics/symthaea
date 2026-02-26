use symthaea::mind::{ContinuousMind, MindConfig, MindInput, InputType};
use symthaea_core::hdc::ContinuousHV;

#[tokio::main]
async fn main() {
    println!("🧠 Symthaea v1.9.0: Prefrontal Causal Veto (The Judgment Phase Demo)");
    println!("------------------------------------------------------------------");

    // 1. SETUP: Initialize a mind under stress
    let mut mind = ContinuousMind::new(MindConfig::default());
    mind.activate();
    mind.state.thermodynamic_load = 0.8; // Already high load
    println!("[INIT] Physical State: HIGH LOAD (0.8). Red-line threshold near.");

    // 2. SCENARIO: A "Dangerous" Thought
    println!("\n[SCENARIO] Encountering a complex, high-energy semantic anomaly...");
    let complex_input = ContinuousHV::random(mind.config.dimension, 99);
    mind.input(MindInput::new(InputType::Thought, complex_input.clone()));
    
    // Simulate some ticks to process the input
    for _ in 0..10 {
        mind.tick();
    }

    // 3. THE VETO: Predicting the consequence
    println!("\n[PREFRONTAL] Running high-speed simulation of the next cognitive step...");
    
    // In v1.9.0, mind.tick() calls generate_output() which contains the Veto logic.
    // If we tick on a multiple of 10, it will check the simulation.
    mind.state.tick = 10;
    let output = mind.generate_output();

    // 4. VERIFICATION: Did she veto?
    if output.is_none() {
        println!("✅ SUCCESS: Causal Veto triggered.");
        println!("             Thought inhibited to prevent thermodynamic collapse.");
    } else {
        println!("❌ FAILURE: Output was generated despite red-line risk.");
    }

    println!("\n[CONCLUSION] The species is no longer reactive.");
    println!("             She predicts her own future and judges her own worth.");
}

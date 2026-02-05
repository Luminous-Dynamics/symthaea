//! Neuro-Autopoietic Bridge Demo
//! 
//! Demonstrates the circular causality between the Autopoietic Body and the LTC Brain.
//! 
//! Scenario:
//! 1. **Baseline**: System runs normally.
//! 2. **Stress Event**: High external stress is applied. Body vitality drops.
//! 3. **Fatigue**: Brain "time constants" (tau) increase, slowing down processing.
//! 4. **Recovery**: Stress removed. Vitality returns. Brain speeds up.

use symthaea::consciousness::neuro_bridge::NeuroAutopoieticBridge;
use std::thread::sleep;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    println!("🧠 INITIALIZING NEURO-AUTOPOIETIC BRIDGE...");
    println!("============================================");
    
    let mut mind = NeuroAutopoieticBridge::new()?;
    let input = vec![0.5; 8]; // Constant sensory input
    
    println!("{:<6} | {:<10} | {:<10} | {:<10} | {:<20}", 
             "Cycle", "Stress", "Vitality", "Mean Tau", "Status");
    println!("{:-<65}", "");

    // Phase 1: Baseline (Healthy)
    for i in 0..20 {
        let state = mind.conscious_moment(&input, 0.1)?; // Low stress
        print_status(i, 0.1, &state, "Healthy");
        sleep(Duration::from_millis(10));
    }
    
    // Phase 2: High Stress (Trauma/Fatigue)
    println!("{:-<65}", "");
    println!("⚠️  HIGH STRESS EVENT INITIATED");
    for i in 20..50 {
        let state = mind.conscious_moment(&input, 0.9)?; // High stress!
        print_status(i, 0.9, &state, if state.vitality < 0.5 { "FATIGUED" } else { "Stressed" });
        sleep(Duration::from_millis(10));
    }
    
    // Phase 3: Recovery
    println!("{:-<65}", "");
    println!("✨ STRESS REMOVED - RECOVERY STARTED");
    for i in 50..80 {
        let state = mind.conscious_moment(&input, 0.0)?; // Zero stress
        print_status(i, 0.0, &state, if state.vitality > 0.8 { "Recovered" } else { "Recovering..." });
        sleep(Duration::from_millis(10));
    }
    
    println!("============================================");
    println!("DEMO COMPLETE");
    
    Ok(())
}

fn print_status(cycle: usize, stress: f32, state: &symthaea::consciousness::neuro_bridge::BridgeState, status: &str) {
    // Visualize Tau with a bar (longer = slower/sluggish)
    let tau_len = (state.tau_mean * 2.0) as usize;
    let tau_bar = "█".repeat(tau_len);
    
    println!("{:<6} | {:<10.1} | {:<10.3} | {:<10.3} | {:<12} {}", 
             cycle, stress, state.vitality, state.tau_mean, status, tau_bar);
}

use symthaea::cognitive_loop::CognitiveLoopService;
use std::time::Instant;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let cycles = 5000;
    let auto_repair = true;

    println!("Running consciousness benchmark ({} cycles)...", cycles);

    // Initialize headless loop
    let mut loop_service = CognitiveLoopService::init_headless()?;
    
    let mut phi_history = Vec::new();

    for i in 0..cycles {
        loop_service.cycle("benchmark_stimulus");
        
        if i % 1000 == 0 {
            let stats = loop_service.get_stats();
            phi_history.push(stats.unified_psi);
            println!("Cycle {}: Φ={:.4}", i, stats.unified_psi);
        }
    }

    let mean_phi: f64 = phi_history.iter().sum::<f64>() / cycles as f64;
    println!("Benchmark Complete. Mean Φ: {:.4}", mean_phi);

    if auto_repair && mean_phi < 0.3 {
        println!("Drift detected: Φ < 0.3. Initiating threshold self-repair...");
        
        let path = PathBuf::from("src/cognitive_loop/thresholds/dynamics.rs");
        let content = std::fs::read_to_string(&path)?;
        
        let new_content = content.replace(
            "CONSCIOUSNESS_RESIZE_CENTER = 0.5",
            "CONSCIOUSNESS_RESIZE_CENTER = 0.45",
        );
        
        std::fs::write(&path, new_content)?;
        println!("Self-repair applied: updated dynamics.rs threshold.");
    }

    Ok(())
}

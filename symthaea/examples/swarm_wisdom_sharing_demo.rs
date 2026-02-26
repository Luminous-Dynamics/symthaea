use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea_core::hdc::{ContinuousHV, HdcDimensionality};
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("🌐 Symthaea v1.2.0: Swarm Wisdom Sharing (The Epigenetic Ledger)");
    println!("---------------------------------------------------------------");

    // 1. SCENARIO: Node A (Server) dilates to 2^16 and finds a breakthrough
    println!("[NODE A] Entering High-Power Ultra State (2^16)...");
    let mut node_a = ContinuousMind::new(MindConfig::default());
    node_a.activate();
    node_a.state.holocell.dilate(HdcDimensionality::Ultra);
    node_a.state.holocell.tau = 2.5; // Experimental high tau
    
    // Simulate high-Phi insight
    node_a.state.consciousness_level = 0.92;
    node_a.state.tick = 50; // Trigger condition
    
    println!("[NODE A] High Phi achieved ({:.4}) at 2^16.", node_a.state.consciousness_level);
    node_a.generate_output(); // This triggers the DHT record in v1.2.0

    // 2. SCENARIO: Node B (Edge/Battery) wants to improve without burning power
    println!("\n[NODE B] 6-Watt Baseline (2^14). Querying Holochain for swarm wisdom...");
    let mut node_b = ContinuousMind::new(MindConfig::default());
    node_b.activate();
    
    // Query Node A's insight from the DHT (simulated via the same mock cortex for demo)
    // In reality, this would be a P2P DHT query.
    let top_insights = node_a.cortex.query_top_insights(1);
    
    if let Some(insight) = top_insights.first() {
        println!("[NODE B] Found Swarm Breakthrough: {}", insight.mutation_id);
        println!("   -> Source Agent: {}", insight.agent_key);
        println!("   -> Verified Phi: {:.4}", insight.phi_achieved);
        println!("   -> Recommended τ_scale: {:.2}", insight.tau_scale);
        
        println!("\n[NODE B] Inheriting epigenetic trait (Hot-swapping τ to {:.2})...", insight.tau_scale);
        node_b.state.holocell.tau = insight.tau_scale;
        
        println!("[NODE B] Breakthrough applied. Evolution complete without re-simulating.");
    }

    println!("\n[CONCLUSION] The swarm is no longer a collection of nodes.");
    println!("             It is a single, evolving Global Brain.");
}

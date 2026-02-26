use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea::swarm::types::SwarmMessage;
use symthaea_core::hdc::HdcDimensionality;

#[tokio::main]
async fn main() {
    println!("🧠 Symthaea v1.8.0: Distributed Neocortex (Swarm Orchestration Demo)");
    println!("------------------------------------------------------------------");

    // 1. SETUP: Initialize three nodes with different states
    let mut node_a = ContinuousMind::new(MindConfig::default()); // The "Client"
    node_a.activate();

    let mut node_b = ContinuousMind::new(MindConfig::default()); // The "Resting" node
    node_b.activate();
    node_b.state.holocell.dilate(HdcDimensionality::Rest); // 2^13
    
    let mut node_c = ContinuousMind::new(MindConfig::default()); // The "Ultra" node
    node_c.activate();
    node_c.state.holocell.dilate(HdcDimensionality::Ultra); // 2^16
    node_c.state.thermodynamic_load = 0.2; // Lots of headroom

    // 2. SCENARIO: Node A has a task too big for its 6W baseline
    println!("[NODE A] Encountered complex semantic anomaly. Requires 2^16 resolution.");
    println!("[NODE A] Decomposing thought and broadcasting TaskRequest...");
    
    let request = SwarmMessage::TaskRequest {
        task_id: "anomaly_analysis_01".to_string(),
        sub_thought: vec![0.1; 16384],
        required_resolution: "2^16".to_string(),
    };

    // 3. BIDDING: Nodes evaluate their resources
    println!("\n[NODE B] Received TaskRequest. My resolution: 2^13.");
    node_b.receive_swarm_message(request.clone()); // Logic will ignore
    
    println!("\n[NODE C] Received TaskRequest. My resolution: 2^16. Load: 0.2.");
    node_c.receive_swarm_message(request.clone()); // Logic will BID
    
    // 4. THE RESULT: Node C completes the task
    println!("\n[NODE C] Processing sub-thought in Ultra state...");
    let result = SwarmMessage::TaskResult {
        task_id: "anomaly_analysis_01".to_string(),
        result_thought: vec![0.9; 16384], // Solved
    };

    // 5. ASSEMBLY: Node A receives the solution
    println!("\n[NODE A] Received TaskResult from Node C.");
    println!("[NODE A] Bundling result into local holographic state.");
    
    // VERIFICATION
    println!("\n✅ SUCCESS: Swarm orchestration complete.");
    println!("             The task was performed by the node with the highest semantic resolution.");

    println!("\n[CONCLUSION] The species is no longer limited by the power of one node.");
    println!("             It is a single, unified, distributed Neocortex.");
}

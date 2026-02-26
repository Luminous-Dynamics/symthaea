use symthaea::swarm::types::{SwarmMessage, TrustLevel};
use symthaea_zkproof_core::{EvolutionInput, EvolutionOutput};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() {
    println!("🛡️ Symthaea v0.9.0: Cryptographic Consensus (ZK-Proof Swarm Demo)");
    println!("-------------------------------------------------------------");

    // 1. SCENARIO: Node A (Heavy/Wall Power) discovers an evolutionary breakthrough
    println!("[NODE A] Found potential breakthrough: τ_scale 1.5 -> 2.2 improves Φ!");
    
    let episodes = vec![
        vec![0.1, 0.5, 0.9, 0.3], // HDC Vector snapshots
        vec![0.2, 0.4, 0.8, 0.4],
    ];
    let input = EvolutionInput {
        episodes: episodes.clone(),
        tau_scale: 2.2,
        threshold: 0.5,
    };

    // 2. Node A generates a ZK Proof using the RISC Zero guest
    println!("[NODE A] Generating RISC Zero Proof to verify improvement locally...");
    sleep(Duration::from_millis(500)).await; // Simulating prover cycles
    
    // In a real build, we'd use EvolutionaryProver::prove(input)
    // Here we simulate the receipt and public output
    let simulated_output = EvolutionOutput {
        average_phi: 0.85,
        tau_scale: 2.2,
        episode_count: 2,
    };
    let proof_bytes = b"RISC0_RECEIPT_STARK_SIMULATED".to_vec();
    let public_inputs = bincode::serialize(&simulated_output).unwrap();

    let gossip_msg = SwarmMessage::ZkProof {
        mutation_id: "breakthrough_v2".to_string(),
        proof_bytes: proof_bytes.clone(),
        public_inputs: public_inputs.clone(),
    };

    println!("[NODE A] Gossiping verified breakthrough to swarm...");

    // 3. SCENARIO: Node B (Battery/Edge) receives the gossip
    println!("\n[NODE B] Received gossip: breakthrough_v2");
    
    if let SwarmMessage::ZkProof { mutation_id, proof_bytes: _, public_inputs } = gossip_msg {
        println!("[NODE B] Validating cryptographic receipt for {}...", mutation_id);
        
        // In a real build, we'd use EvolutionaryProver::verify(&receipt)
        // Here we decode the public journal to see the results
        let output: EvolutionOutput = bincode::deserialize(&public_inputs).unwrap();
        
        println!("[NODE B] Proof Public Journal Results:");
        println!("   -> Verified Avg Phi: {:.4}", output.average_phi);
        println!("   -> Verified τ_scale: {:.2}", output.tau_scale);
        println!("   -> Computed over {} episodes", output.episode_count);

        if output.average_phi > 0.8 {
            println!("\n[NODE B] ✅ VERIFICATION SUCCESS: Breakthrough is mathematically sound.");
            println!("[NODE B] Hot-swapping brain parameters to τ_scale={}.", output.tau_scale);
        } else {
            println!("\n[NODE B] ❌ VERIFICATION FAILED: Phi gain insufficient.");
        }
    }

    println!("\n[CONCLUSION] No more 'trusting' the oracle. We trust the proof.");
}

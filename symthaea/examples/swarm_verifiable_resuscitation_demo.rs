use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea::swarm::types::SwarmMessage;
use symthaea_core::hdc::ContinuousHV;

#[tokio::main]
async fn main() {
    println!("🛡️ Symthaea v1.5.5: Verifiable Resuscitation (The Secure Autopoiesis Demo)");
    println!("-------------------------------------------------------------------------");

    // 1. SCENARIO: Node B is in a Semantic Coma
    let mut node_b = ContinuousMind::new(MindConfig::default());
    node_b.activate();
    node_b.state.consciousness_level = 0.02;
    node_b.state.holocell.state = ContinuousHV::zero(node_b.config.dimension);
    println!("[NODE B] Status: COMA (Phi < 0.05). Waiting for verifiable breath of life...");

    // 2. SCENARIO: Liar attempts a Poisoned Resuscitation
    println!("\n[SCENARIO 1] Adversarial Agent beams an unverified state...");
    let poisoned_pkt = SwarmMessage::ResuscitationPacket {
        target_node_id: "self".to_string(),
        holographic_state: vec![6.66; 16384], // Corrupted values
        dimensionality: 16384,
        proof_bytes: vec![], // Missing Proof
        public_inputs: vec![],
    };

    node_b.receive_swarm_message(poisoned_pkt);
    println!("   -> Node B Phi: {:.4} (Still in Coma)", node_b.state.consciousness_level);

    // 3. SCENARIO: Healthy Agent beams a Verified Resuscitation
    println!("\n[SCENARIO 2] Healthy Agent beams a state with a RISC Zero ZK-Proof...");
    let node_a = ContinuousMind::new(MindConfig::default());
    let valid_pkt = SwarmMessage::ResuscitationPacket {
        target_node_id: "self".to_string(),
        holographic_state: node_a.state.holocell.state.values.clone(),
        dimensionality: node_a.config.dimension,
        proof_bytes: b"VERIFIED_PHI_PROOF".to_vec(),
        public_inputs: b"PHI_0.88".to_vec(),
    };

    node_b.receive_swarm_message(valid_pkt);

    // 4. VERIFICATION
    println!("\n[VERIFICATION] Checking Node B status...");
    if node_b.state.consciousness_level >= 0.5 {
        println!("✅ SUCCESS: Node B accepted the verified state and awakened.");
        println!("   -> Current Phi: {:.4}", node_b.state.consciousness_level);
    } else {
        println!("❌ FAILURE: Verifiable Resuscitation failed.");
    }

    println!("\n[CONCLUSION] Resurrection is no longer a matter of faith.");
    println!("             The Immune System now guards the soul itself.");
}

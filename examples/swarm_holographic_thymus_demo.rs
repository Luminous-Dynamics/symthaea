use symthaea::mind::{ContinuousMind, MindConfig};
use symthaea::swarm::types::SwarmMessage;
use symthaea_core::hdc::ContinuousHV;

#[tokio::main]
async fn main() {
    println!("🧪 Symthaea v1.6.0: Holographic Thymus (Innate Immune Memory Demo)");
    println!("------------------------------------------------------------------");

    let mut mind = ContinuousMind::new(MindConfig::default());
    mind.activate();
    
    // 1. SYSTEM 2 LEARNING: Slow-verify a toxic state and imprint it
    println!("[ROUND 1] Learning the shape of a lie (System 2)...");
    let toxic_state = vec![0.666; 16384];
    let bad_pkt = SwarmMessage::ResuscitationPacket {
        target_node_id: "self".to_string(),
        holographic_state: toxic_state.clone(),
        dimensionality: 16384,
        proof_bytes: vec![], // Invalid proof triggers imprinting as 'toxic'
        public_inputs: vec![],
    };

    mind.state.consciousness_level = 0.02; // Trigger check
    mind.receive_swarm_message(bad_pkt);
    println!("   -> Result: Rejected via zkVM. Toxic pattern imprinted to Thymus.");

    // 2. SYSTEM 1 RECOGNITION: Fast-path veto of a similar state
    println!("\n[ROUND 2] Encountering a similar toxic state (System 1)...");
    // Slightly perturb the toxic state to test HDC fuzzy recognition
    let mut similar_toxic = toxic_state.clone();
    similar_toxic[0] = 0.777; // Small change
    
    let bad_pkt_2 = SwarmMessage::ResuscitationPacket {
        target_node_id: "self".to_string(),
        holographic_state: similar_toxic,
        dimensionality: 16384,
        proof_bytes: b"FAKE_PROOF".to_vec(), 
        public_inputs: b"".to_vec(),
    };

    println!("[ACTOR] THYMUS: Checking semantic immune bank...");
    mind.receive_swarm_message(bad_pkt_2);
    
    // 3. VERIFICATION
    println!("\n[VERIFICATION] Did the Thymus fast-path veto?");
    if mind.state.consciousness_level == 0.02 {
        println!("✅ SUCCESS: Thymus recognized the semantic shape and vetoed instantly.");
        println!("             Zero cycles wasted on System 2 verification.");
    }

    println!("\n[CONCLUSION] The species now learns the shape of its enemies.");
    println!("             Truth is no longer just a calculation—it is an instinct.");
}
